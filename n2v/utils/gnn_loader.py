"""
Loader for GNN checkpoints stored as MATLAB .mat files (NNV's GNN format).

The same file format is consumed by NNV's ``gnn2nnv.m``, so one exported
checkpoint can be verified with both tools.  To verify a PyTorch Geometric
model directly, use :meth:`n2v.nn.GraphNeuralNetwork.from_pyg` instead.

**File schema.**  All weights are stored in NNV orientation, ``(F_in, F_out)`` (the transpose
of ``torch.nn.Linear.weight``); biases are ``(F_out, 1)`` and optional.
Layers are numbered from 1 and applied in index order.

``model_type`` (string) selects the layer family:

* ``'gcn'`` -- ``best_params.mult{i}.{Weights, Bias}``; ``ANorm_g`` is the
  dense ``(N, N)`` normalized adjacency (e.g. PyG ``gcn_norm`` with self
  loops), oriented so that ``Y = ANorm_g @ X @ W + b``.
* ``'sage'`` -- ``best_params.sage{i}.{NodeWeights, EdgeWeights, Bias}``
  (root and neighbor projections); ``A_adj`` is the ``(N, N)`` aggregation
  matrix, ``Y = X @ NodeWeights + A_adj @ X @ EdgeWeights + b``.
* ``'gine_conv'`` (PyG ``GINEConv``) and ``'gine_pretrain'`` (Hu et al.
  GIN with edge features) -- ``best_params.conv{i}.mlp1``, ``.mlp2`` (each
  ``{Weights, Bias}``), the edge projection ``.edge_linear`` (gine_conv) or
  ``.edge_proj`` (gine_pretrain) as ``(E_in, F_in)`` weights, and an optional
  scalar ``.eps`` (gine_conv).  The graph is given by 1-indexed ``src`` /
  ``dst`` columns ``(m, 1)``, edge features ``E_edge`` ``(m, E_in)``, and
  optional per-edge weights ``a`` ``(m, 1)``.  For gine_pretrain the self
  loops must be present in ``src`` / ``dst``.

Optional fields: ``activations`` (``'none'`` disables the inter-layer ReLU
of GCN/SAGE; default ReLU after every layer), ``X_test_g`` / ``Y_test_g`` /
``python_predictions`` (``(k, 1)`` cell arrays of per-instance ``(N, F)``
matrices, used for parity checks), and ``X_max`` / ``Y_max`` normalization
constants.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from scipy.io import loadmat


# ============================================================ Layer specs

@dataclass
class GCNLayerSpec:
    """One graph-convolution layer (weights + bias). Bias may be None."""
    W: np.ndarray   # (F_in, F_out)
    b: Optional[np.ndarray]  # (F_out,) or None


@dataclass
class SAGELayerSpec:
    """One GraphSAGE layer: self + neighbor projections and a bias."""
    W_node: np.ndarray        # (F_in, F_out) self/root weight
    W_edge: np.ndarray        # (F_in, F_out) neighbor weight
    b: Optional[np.ndarray]   # (F_out,) or None


@dataclass
class GINELayerSpec:
    """One GINEConv layer: edge projection + 2-layer MLP + self-loop eps."""
    W1: np.ndarray            # (F_in, hidden)  MLP layer 1
    b1: Optional[np.ndarray]  # (hidden,)
    W2: np.ndarray            # (hidden, F_out) MLP layer 2
    b2: Optional[np.ndarray]  # (F_out,)
    W_edge: np.ndarray        # (E_in, F_in)    edge projection
    b_edge: Optional[np.ndarray]  # (F_in,)
    eps: float = 0.0          # self-loop scaling


@dataclass
class NormStats:
    """Optional feature-/target-wise normalization saved alongside the model."""
    X_max: Optional[np.ndarray] = None
    Y_max: Optional[np.ndarray] = None


# ============================================================ Model container

@dataclass
class GNNModel:
    """Parsed GNN checkpoint with everything needed for evaluation and reach.

    Fields are populated according to ``model_type``:
        * gcn  -> ``layers`` are GCNLayerSpec, ``A_norm`` is the dense adjacency.
        * sage -> ``layers`` are SAGELayerSpec, ``A_adj`` is the binary adjacency.
        * gine -> ``layers`` are GINELayerSpec, ``edge_index`` / ``E`` /
          ``edge_weights`` describe the graph.

    ``gcn_layers`` and ``adjacency`` are convenience aliases so existing GCN
    code keeps working unchanged.
    """
    model_type: str
    layers: List[Union[GCNLayerSpec, SAGELayerSpec, GINELayerSpec]]
    has_relu: bool
    A_norm: Optional[np.ndarray] = None          # gcn
    A_adj: Optional[np.ndarray] = None           # sage
    edge_index: Optional[np.ndarray] = None      # gine (2, m) 0-indexed
    E: Optional[np.ndarray] = None               # gine edge features (m, E_in)
    edge_weights: Optional[np.ndarray] = None    # gine per-edge weights (m,)
    X_test: List[np.ndarray] = field(default_factory=list)
    Y_test: List[np.ndarray] = field(default_factory=list)
    python_predictions: List[np.ndarray] = field(default_factory=list)
    norm_stats: NormStats = field(default_factory=NormStats)
    source_path: Optional[Path] = None

    @property
    def gcn_layers(self) -> List:
        """Back-compat alias used by the GCN reach helpers and tests."""
        return self.layers

    @property
    def adjacency(self) -> Optional[np.ndarray]:
        """Adjacency for matrix-based layers (gcn -> A_norm, sage -> A_adj)."""
        return self.A_norm if self.A_norm is not None else self.A_adj

    @property
    def gine_variant(self) -> Optional[str]:
        """GINE architecture variant: 'pyg' (gine_conv) or 'hugine' (gine_pretrain)."""
        if self.model_type == "gine_conv":
            return "pyg"
        if self.model_type == "gine_pretrain":
            return "hugine"
        return None

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_test_instances(self) -> int:
        return len(self.X_test)


# GCNModel kept as an alias so existing imports/annotations keep working.
GCNModel = GNNModel


# ============================================================ scipy helpers

def _decode_model_type(raw) -> str:
    """Pull the scalar string out of scipy's nested ndarray wrappers."""
    arr = np.asarray(raw)
    if arr.dtype.kind in ("U", "S"):
        return str(arr.flatten()[0])
    if arr.dtype == object:
        return _decode_model_type(arr.flatten()[0])
    raise ValueError(f"Cannot decode string from {raw!r}")


def _unwrap_per_instance(arr) -> List[np.ndarray]:
    """Convert a (k, 1) object array of per-instance matrices to a list."""
    arr = np.asarray(arr)
    out = []
    for i in range(arr.shape[0]):
        cell = arr[i, 0] if arr.ndim == 2 else arr[i]
        out.append(np.asarray(cell, dtype=np.float64))
    return out


def _record(struct):
    """Return the scalar struct record from a scipy struct array."""
    return struct[0, 0] if struct.ndim == 2 else struct[0]


def _sorted_fields(record, prefix: str) -> List[str]:
    """Field names of ``record`` starting with ``prefix``, ordered by index."""
    names = list(record.dtype.names or [])
    return sorted(
        [n for n in names if n.startswith(prefix)],
        key=lambda n: int(n[len(prefix):]),
    )


def _weights(block, name: str) -> np.ndarray:
    block = block[0, 0] if block.ndim == 2 else block[0]
    return np.asarray(block[name], dtype=np.float64)


def _opt_bias(block, name: str, out_dim: int) -> Optional[np.ndarray]:
    block = block[0, 0] if block.ndim == 2 else block[0]
    if name not in (block.dtype.names or []):
        return None
    raw = np.asarray(block[name], dtype=np.float64).reshape(-1)
    return raw if raw.size == out_dim else None


# ============================================================ per-type parsers

def _parse_gcn(best_params) -> List[GCNLayerSpec]:
    record = _record(best_params)
    layers = []
    for name in _sorted_fields(record, "mult"):
        block = record[name]
        W = _weights(block, "Weights")
        layers.append(GCNLayerSpec(W=W, b=_opt_bias(block, "Bias", W.shape[1])))
    return layers


def _parse_sage(best_params) -> List[SAGELayerSpec]:
    record = _record(best_params)
    layers = []
    for name in _sorted_fields(record, "sage"):
        block = record[name]
        W_node = _weights(block, "NodeWeights")
        W_edge = _weights(block, "EdgeWeights")
        layers.append(
            SAGELayerSpec(W_node=W_node, W_edge=W_edge,
                          b=_opt_bias(block, "Bias", W_node.shape[1]))
        )
    return layers


def _parse_gine(best_params, edge_field: str) -> List[GINELayerSpec]:
    record = _record(best_params)
    layers = []
    for name in _sorted_fields(record, "conv"):
        conv = _record(record[name])
        W1 = _weights(conv["mlp1"], "Weights")
        b1 = _opt_bias(conv["mlp1"], "Bias", W1.shape[1])
        W2 = _weights(conv["mlp2"], "Weights")
        b2 = _opt_bias(conv["mlp2"], "Bias", W2.shape[1])
        W_edge = _weights(conv[edge_field], "Weights")
        b_edge = _opt_bias(conv[edge_field], "Bias", W_edge.shape[1])
        eps = 0.0
        if "eps" in (conv.dtype.names or []):
            eps = float(np.asarray(conv["eps"]).reshape(-1)[0])
        layers.append(GINELayerSpec(W1, b1, W2, b2, W_edge, b_edge, eps))
    return layers


def _parse_test_data(raw):
    X_test = _unwrap_per_instance(raw["X_test_g"]) if "X_test_g" in raw else []
    Y_test = _unwrap_per_instance(raw["Y_test_g"]) if "Y_test_g" in raw else []
    if "python_predictions" in raw:
        pp = np.asarray(raw["python_predictions"])
        py_pred = _unwrap_per_instance(pp) if pp.dtype == object else [np.asarray(pp, dtype=np.float64)]
    else:
        py_pred = []
    norm = NormStats()
    if "X_max" in raw:
        norm.X_max = np.asarray(raw["X_max"], dtype=np.float64).reshape(-1)
    if "Y_max" in raw:
        norm.Y_max = np.asarray(raw["Y_max"], dtype=np.float64).reshape(-1)
    return X_test, Y_test, py_pred, norm


def _edge_index_from_mat(raw) -> np.ndarray:
    """Build a 0-indexed (2, m) edge_index from the 1-indexed src/dst columns."""
    src = np.asarray(raw["src"], dtype=np.int64).reshape(-1) - 1
    dst = np.asarray(raw["dst"], dtype=np.int64).reshape(-1) - 1
    return np.vstack([src, dst])


# ============================================================ public loader

def load_gnn_mat(path) -> GNNModel:
    """Load a GNN .mat checkpoint (schema in the module docstring).

    Dispatches on ``model_type``; raises NotImplementedError for other types.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GNN checkpoint not found: {path}")

    raw = loadmat(str(path))
    model_type = _decode_model_type(raw["model_type"]).lower()
    X_test, Y_test, py_pred, norm = _parse_test_data(raw)

    has_relu = True
    if "activations" in raw:
        try:
            has_relu = _decode_model_type(raw["activations"]).lower() != "none"
        except Exception:
            has_relu = True

    common = dict(
        model_type=model_type, X_test=X_test, Y_test=Y_test,
        python_predictions=py_pred, norm_stats=norm, source_path=path,
    )

    if model_type == "gcn":
        return GNNModel(
            layers=_parse_gcn(raw["best_params"]), has_relu=has_relu,
            A_norm=np.asarray(raw["ANorm_g"], dtype=np.float64), **common,
        )

    if model_type == "sage":
        return GNNModel(
            layers=_parse_sage(raw["best_params"]), has_relu=has_relu,
            A_adj=np.asarray(raw["A_adj"], dtype=np.float64), **common,
        )

    if model_type in ("gine_conv", "gine_pretrain"):
        edge_field = "edge_linear" if model_type == "gine_conv" else "edge_proj"
        edge_index = _edge_index_from_mat(raw)
        E = np.asarray(raw["E_edge"], dtype=np.float64) if "E_edge" in raw else None
        ew = np.asarray(raw["a"], dtype=np.float64).reshape(-1) if "a" in raw else None
        # GINE carries its own internal ReLUs; no extra inter-layer ReLU.
        return GNNModel(
            layers=_parse_gine(raw["best_params"], edge_field), has_relu=False,
            edge_index=edge_index, E=E, edge_weights=ew, **common,
        )

    raise NotImplementedError(
        f"GNN model_type '{model_type}' is not yet supported "
        "(supported: gcn, sage, gine_conv, gine_pretrain)."
    )
