"""
Convert PyTorch Geometric graph-convolution layers into n2v GNN layer specs.

Supported layers (all layers of one model must share a family):
    * ``GCNConv``  -> :class:`GCNLayerSpec`; the dense adjacency is built with
      PyG's own ``gcn_norm`` using the layer's ``improved`` /
      ``add_self_loops`` / ``normalize`` settings.
    * ``SAGEConv`` -> :class:`SAGELayerSpec`; ``aggr`` 'mean' (row-normalized
      adjacency) or 'add'/'sum' (count adjacency).  ``project`` and
      ``normalize`` must be off.
    * ``GINEConv`` -> :class:`GINELayerSpec` ('pyg' variant); ``nn`` must be
      ``Sequential(Linear, ReLU, Linear)`` and ``aggr`` 'add'.

Activations between layers live in the model's ``forward`` rather than in the
layer modules, so they are passed explicitly to
:meth:`n2v.nn.GraphNeuralNetwork.from_pyg`.

``torch_geometric`` is an optional dependency (``pip install n2v[gnn]``); it is
only imported when a conversion runs.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from n2v.utils.gnn_loader import GCNLayerSpec, SAGELayerSpec, GINELayerSpec


def _require_pyg():
    try:
        import torch_geometric  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without PyG
        raise ImportError(
            "Converting PyTorch Geometric models requires torch_geometric; "
            "install it with `pip install n2v[gnn]`") from exc


def _np(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64)


def _bias(linear_or_param) -> Optional[np.ndarray]:
    return None if linear_or_param is None else _np(linear_or_param).reshape(-1)


def collect_pyg_convs(model) -> List:
    """Return the supported PyG conv layers of ``model`` in registration order.

    ``model`` may be a single conv layer, a sequence of conv layers, or any
    ``torch.nn.Module`` whose conv layers are registered in forward order.
    """
    _require_pyg()
    from torch_geometric.nn import GCNConv, SAGEConv, GINEConv

    supported = (GCNConv, SAGEConv, GINEConv)
    if isinstance(model, supported):
        return [model]
    if isinstance(model, (list, tuple)):
        convs = list(model)
    else:
        convs = [m for m in model.modules() if isinstance(m, supported)]
    if not convs:
        raise ValueError("No GCNConv, SAGEConv, or GINEConv layers found in model")
    bad = [type(c).__name__ for c in convs if not isinstance(c, supported)]
    if bad:
        raise TypeError(f"Unsupported layer types: {bad}")
    kinds = {type(c).__name__ for c in convs}
    if len(kinds) != 1:
        raise ValueError(f"All conv layers must be the same type; got {sorted(kinds)}")
    for c in convs:
        if c.flow != "source_to_target":
            raise ValueError(f"{type(c).__name__} with flow='{c.flow}' is not supported")
    return convs


def _dense(edge_index: np.ndarray, weights: np.ndarray, num_nodes: int) -> np.ndarray:
    """A[dst, src] += w, so that (A @ X)[i] aggregates messages x_j -> x_i."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    np.add.at(A, (edge_index[1], edge_index[0]), weights)
    return A


def _gcn_adjacency(conv, edge_index, num_nodes: int, edge_weight=None) -> np.ndarray:
    import torch
    from torch_geometric.nn.conv.gcn_conv import gcn_norm

    ei = torch.as_tensor(edge_index, dtype=torch.long)
    ew = None if edge_weight is None else torch.as_tensor(edge_weight, dtype=torch.float64)
    if conv.normalize:
        ei, ew = gcn_norm(ei, ew, num_nodes, conv.improved, conv.add_self_loops,
                          conv.flow, dtype=torch.float64)
    elif ew is None:
        ew = torch.ones(ei.shape[1], dtype=torch.float64)
    return _dense(ei.cpu().numpy(), ew.cpu().numpy(), num_nodes)


def _sage_adjacency(aggr: str, edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    A = _dense(edge_index, np.ones(edge_index.shape[1]), num_nodes)
    if aggr == "mean":
        deg = A.sum(axis=1, keepdims=True)
        A = np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)
    return A


def convert_pyg(
    model,
    edge_index,
    num_nodes: int,
    edge_attr=None,
    edge_weight=None,
) -> Tuple[List[Union[GCNLayerSpec, SAGELayerSpec, GINELayerSpec]], dict]:
    """Convert PyG conv layers to n2v layer specs plus the graph structure.

    Args:
        model: Conv layer, sequence of conv layers, or module containing them.
        edge_index: (2, m) 0-indexed edge list (tensor or array), PyG convention
            (row 0 = source, row 1 = target).
        num_nodes: Number of nodes N.
        edge_attr: (m, E_in) edge features; required for GINEConv.
        edge_weight: Optional (m,) edge weights for GCNConv.

    Returns:
        ``(layers, graph)`` where ``graph`` holds the keyword arguments for
        :class:`n2v.nn.GraphNeuralNetwork` (``adjacency``, or ``edge_index`` /
        ``E`` / ``gine_variant`` for GINE).
    """
    convs = collect_pyg_convs(model)
    ei = np.asarray(edge_index.detach().cpu().numpy() if hasattr(edge_index, "detach")
                    else edge_index, dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, m); got {ei.shape}")
    kind = type(convs[0]).__name__

    if kind == "GCNConv":
        settings = {(c.improved, c.add_self_loops, c.normalize) for c in convs}
        if len(settings) != 1:
            raise ValueError("All GCNConv layers must share improved/add_self_loops/normalize")
        if any(c.aggr != "add" for c in convs):
            raise ValueError("GCNConv with aggr other than 'add' is not supported")
        layers = [GCNLayerSpec(_np(c.lin.weight).T, _bias(c.bias)) for c in convs]
        return layers, {"adjacency": _gcn_adjacency(convs[0], ei, num_nodes, edge_weight)}

    if kind == "SAGEConv":
        aggrs = {str(c.aggr) for c in convs}
        if len(aggrs) != 1 or not aggrs <= {"mean", "add", "sum"}:
            raise ValueError(f"SAGEConv aggr must be one of mean/add/sum and shared; got {aggrs}")
        if any(c.project or c.normalize for c in convs):
            raise ValueError("SAGEConv with project=True or normalize=True is not supported")
        layers = []
        for c in convs:
            W_edge = _np(c.lin_l.weight).T
            W_node = _np(c.lin_r.weight).T if c.root_weight else np.zeros_like(W_edge)
            layers.append(SAGELayerSpec(W_node, W_edge, _bias(c.lin_l.bias)))
        return layers, {"adjacency": _sage_adjacency(aggrs.pop(), ei, num_nodes)}

    # GINEConv
    import torch.nn as tnn

    if edge_attr is None:
        raise ValueError("GINEConv requires edge_attr")
    E = np.asarray(edge_attr.detach().cpu().numpy() if hasattr(edge_attr, "detach")
                   else edge_attr, dtype=np.float64)
    layers = []
    for c in convs:
        mlp = c.nn
        if not (isinstance(mlp, tnn.Sequential) and len(mlp) == 3
                and isinstance(mlp[0], tnn.Linear) and isinstance(mlp[1], tnn.ReLU)
                and isinstance(mlp[2], tnn.Linear)):
            raise ValueError("GINEConv nn must be Sequential(Linear, ReLU, Linear)")
        if c.aggr != "add":
            raise ValueError("GINEConv with aggr other than 'add' is not supported")
        F_in = mlp[0].in_features
        if c.lin is not None:
            W_edge, b_edge = _np(c.lin.weight).T, _bias(c.lin.bias)
        else:
            if E.shape[1] != F_in:
                raise ValueError(f"GINEConv without edge_dim needs edge_attr width {F_in}")
            W_edge, b_edge = np.eye(F_in), None
        eps = float(c.eps.item()) if hasattr(c.eps, "item") else float(c.eps)
        layers.append(GINELayerSpec(_np(mlp[0].weight).T, _bias(mlp[0].bias),
                                    _np(mlp[2].weight).T, _bias(mlp[2].bias),
                                    W_edge, b_edge, eps))
    return layers, {"edge_index": ei, "E": E, "gine_variant": "pyg"}
