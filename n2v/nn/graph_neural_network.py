"""
GraphNeuralNetwork: a high-level wrapper for GNN reachability.

Owns an ordered stack of graph-conv layer specs together with the graph
structure (adjacency for GCN/SAGE, or edge list + edge features for GINE) and
exposes ``evaluate`` (concrete forward) and ``reach`` (set propagation),
mirroring the role of :class:`n2v.nn.NeuralNetwork` for feed-forward models.

Layer specs are the dataclasses produced by :func:`n2v.utils.load_gnn_mat`
(:class:`GCNLayerSpec`, :class:`SAGELayerSpec`, :class:`GINELayerSpec`); the
wrapper dispatches per spec type to the corresponding reach op.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from n2v.sets import GraphStar
from n2v.utils.gnn_loader import (
    GNNModel, GCNLayerSpec, SAGELayerSpec, GINELayerSpec, load_gnn_mat,
)
from n2v.utils.subgraph import khop_subgraph_matrix, khop_subgraph_edges
from n2v.utils.gnn_falsify import falsify_node_bounds, FalsifyResult
from n2v.nn.layer_ops.gcn_reach import (
    gcn_graph_star, gcn_evaluate, relu_graph_star,
)
from n2v.nn.layer_ops.sage_reach import sage_graph_star, sage_evaluate
from n2v.nn.layer_ops.gine_reach import gine_graph_star, gine_evaluate


def _single(sets: List[GraphStar], what: str) -> GraphStar:
    """Return the only set, refusing to silently pick one of several.

    Exact reach yields a union of GraphStars; taking any one of them would
    under-approximate the reachable set.
    """
    if len(sets) != 1:
        raise ValueError(
            f"{what} holds {len(sets)} GraphStars (exact reach yields a union); "
            f"use the list attribute instead")
    return sets[0]


def _is_box(gs: GraphStar) -> bool:
    """True if ``gs`` equals its bounding box (so box samples stay inside it).

    Holds when every generator moves a single (node, feature) entry, no entry
    has two generators, and every constraint is a single-predicate bound no
    tighter than the predicate bounds.
    """
    G = gs.V[..., 1:].reshape(-1, gs.V.shape[-1] - 1)
    if G.size and (np.any(np.count_nonzero(G, axis=0) > 1)
                   or np.any(np.count_nonzero(G, axis=1) > 1)):
        return False
    if gs.C.size == 0:
        return True
    if gs.predicate_lb is None or gs.predicate_ub is None:
        return False
    lb, ub = gs.predicate_lb.reshape(-1), gs.predicate_ub.reshape(-1)
    for row, d in zip(gs.C, gs.d.reshape(-1)):
        nz = np.flatnonzero(row)
        if nz.size != 1:
            return False
        j = nz[0]
        c = row[j]
        if (c > 0 and d / c < ub[j] - 1e-12) or (c < 0 and d / c > lb[j] + 1e-12):
            return False
    return True


@dataclass
class VerifyResult:
    """Verdict from :meth:`GraphNeuralNetwork.verify`."""
    status: str                              # 'verified' | 'falsified' | 'unknown'
    reach_sets: List[GraphStar] = None       # output reach set = union of these
    counterexample: "FalsifyResult" = None   # populated when status == 'falsified'

    @property
    def reach_set(self) -> GraphStar:
        """The output reach set when it is a single GraphStar (approx reach)."""
        return _single(self.reach_sets, "VerifyResult.reach_sets")


@dataclass
class SubgraphResult:
    """Per-target output of :meth:`GraphNeuralNetwork.reach_subgraph`."""
    target_node: int          # original 0-indexed node
    target_local_idx: int     # its row index within each output set
    outputs: List[GraphStar]  # reach set over the subgraph nodes = union of these
    n_sub_nodes: int
    n_sub_edges: int

    @property
    def output(self) -> GraphStar:
        """The reach set when it is a single GraphStar (approx reach)."""
        return _single(self.outputs, "SubgraphResult.outputs")

    def target_ranges(self, **kwargs):
        """(lb, ub) feature vectors for the target node only (hull over all outputs)."""
        ranges = [s.get_ranges(**kwargs) for s in self.outputs]
        lb = np.min([r[0][self.target_local_idx] for r in ranges], axis=0)
        ub = np.max([r[1][self.target_local_idx] for r in ranges], axis=0)
        return lb, ub


class GraphNeuralNetwork:
    """Reachability wrapper around a stack of graph-convolution layers.

    Args:
        layers: Ordered layer specs (all the same family: GCN, SAGE, or GINE).
        adjacency: Dense adjacency for GCN (normalized) or SAGE (binary).
        has_relu: Apply ReLU after every GCN/SAGE layer (ignored for GINE, which
            carries its own internal ReLUs).
        edge_index: (2, m) 0-indexed edge list (GINE only).
        E: Edge features (m, E_in) (GINE only).
        edge_weights: Per-edge aggregation weights (m,) (GINE only).
        gine_variant: 'hugine' (gine_pretrain) or 'pyg' (gine_conv).
        relu_layers: Optional per-layer flags: ReLU after layer ``i`` iff
            ``relu_layers[i]``.  Overrides the defaults, which are ``has_relu``
            on every GCN/SAGE layer, every layer but the last for 'hugine', and
            none for 'pyg' (MLP-internal ReLUs of GINE are always applied).
    """

    def __init__(
        self,
        layers: Sequence,
        adjacency: Optional[np.ndarray] = None,
        has_relu: bool = True,
        edge_index: Optional[np.ndarray] = None,
        E: Optional[np.ndarray] = None,
        edge_weights: Optional[np.ndarray] = None,
        gine_variant: str = "hugine",
        relu_layers: Optional[Sequence[bool]] = None,
    ):
        if len(layers) == 0:
            raise ValueError("GraphNeuralNetwork requires at least one layer")
        kinds = {type(l).__name__ for l in layers}
        if len(kinds) != 1:
            raise ValueError(f"All layers must be the same type; got {sorted(kinds)}")

        self.layers = list(layers)
        self.layer_kind = next(iter(kinds))
        self.adjacency = None if adjacency is None else np.asarray(adjacency, dtype=np.float64)
        self.has_relu = has_relu
        self.edge_index = None if edge_index is None else np.asarray(edge_index)
        self.E = None if E is None else np.asarray(E, dtype=np.float64)
        self.edge_weights = None if edge_weights is None else np.asarray(edge_weights, dtype=np.float64)
        self.gine_variant = gine_variant

        n = len(self.layers)
        if relu_layers is None:
            if self.layer_kind != "GINELayerSpec":
                relu_layers = [has_relu] * n
            elif gine_variant == "hugine":
                relu_layers = [i < n - 1 for i in range(n)]
            else:
                relu_layers = [False] * n
        if len(relu_layers) != n:
            raise ValueError(f"relu_layers has {len(relu_layers)} entries for {n} layers")
        self.relu_layers = [bool(r) for r in relu_layers]

        if self.layer_kind == "GINELayerSpec":
            if self.edge_index is None or self.E is None:
                raise ValueError("GINE network requires edge_index and E")
        else:
            if self.adjacency is None:
                raise ValueError(f"{self.layer_kind} network requires an adjacency matrix")

    # ----------------------------------------------------------------- builders

    @classmethod
    def from_model(cls, model: GNNModel) -> "GraphNeuralNetwork":
        """Build from a parsed :class:`GNNModel` (see :func:`load_gnn_mat`)."""
        return cls(
            layers=model.layers,
            adjacency=model.adjacency,
            has_relu=model.has_relu,
            edge_index=model.edge_index,
            E=model.E,
            edge_weights=model.edge_weights,
            gine_variant=model.gine_variant or "hugine",
        )

    @classmethod
    def from_mat(cls, path) -> "GraphNeuralNetwork":
        """Load a .mat checkpoint and wrap it."""
        return cls.from_model(load_gnn_mat(path))

    @classmethod
    def from_pyg(
        cls,
        model,
        edge_index,
        num_nodes: int,
        edge_attr=None,
        edge_weight=None,
        relu_hidden: bool = True,
        relu_output: bool = False,
    ) -> "GraphNeuralNetwork":
        """Build from PyTorch Geometric ``GCNConv`` / ``SAGEConv`` / ``GINEConv`` layers.

        Args:
            model: A conv layer, a sequence of conv layers, or a module whose
                conv layers are registered in forward order (see
                :func:`n2v.utils.gnn_pyg.convert_pyg` for supported settings).
            edge_index / num_nodes / edge_attr / edge_weight: The graph the
                model runs on (``edge_attr`` is required for GINEConv).
            relu_hidden: The model applies ReLU after every layer but the last.
            relu_output: The model applies ReLU after the last layer.

        Requires ``torch_geometric`` (``pip install n2v[gnn]``).
        """
        from n2v.utils.gnn_pyg import convert_pyg

        layers, graph = convert_pyg(model, edge_index, num_nodes,
                                    edge_attr=edge_attr, edge_weight=edge_weight)
        n = len(layers)
        relu_layers = [relu_hidden if i < n - 1 else relu_output for i in range(n)]
        return cls(layers, relu_layers=relu_layers, **graph)

    # --------------------------------------------------------------- properties

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        return (
            f"GraphNeuralNetwork(kind={self.layer_kind}, layers={self.num_layers}, "
            f"variant={self.gine_variant if self.layer_kind == 'GINELayerSpec' else 'n/a'})"
        )

    # ------------------------------------------------------------ forward eval

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Concrete forward pass over node features ``X`` (N, F_in)."""
        Y = np.asarray(X, dtype=np.float64)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GCNLayerSpec):
                Y = gcn_evaluate(Y, self.adjacency, layer.W, layer.b)
                if self.relu_layers[i]:
                    Y = np.maximum(Y, 0.0)
            elif isinstance(layer, SAGELayerSpec):
                Y = sage_evaluate(Y, self.adjacency, layer.W_node, layer.W_edge, layer.b)
                if self.relu_layers[i]:
                    Y = np.maximum(Y, 0.0)
            elif isinstance(layer, GINELayerSpec):
                Y = gine_evaluate(Y, self.E, self.edge_index, layer,
                                  self.edge_weights, self.gine_variant, self.relu_layers[i])
            else:
                raise TypeError(f"Unsupported layer spec: {type(layer).__name__}")
        return Y

    # --------------------------------------------------------------- reach

    def reach(
        self,
        input_set: GraphStar,
        method: str = "approx",
        relax_factor: float = 0.5,
        lp_solver: str = "default",
    ) -> List[GraphStar]:
        """Propagate a GraphStar through the layer stack.

        Args:
            input_set: Input GraphStar over node features.
            method: 'approx' (triangle-relaxation ReLU; one set) or 'exact'
                (ReLU splitting; a union of up to 2^k sets for k sign-crossing
                neurons).  'exact' is supported for GCN and SAGE only.
            relax_factor / lp_solver: forwarded to the ReLU relaxation.  As in
                :func:`relu_star_approx`, ``relax_factor=0`` under 'approx'
                means exact splitting (so it is rejected for GINE too); every
                value > 0 gives the same standard triangle relaxation.

        Returns:
            List of output GraphStars whose union is the reach set (length 1
            under approx-star).
        """
        if not isinstance(input_set, GraphStar):
            raise TypeError(f"reach expects a GraphStar, got {type(input_set).__name__}")
        if method not in ("approx", "exact"):
            raise ValueError(f"GNN reach supports method='approx' or 'exact'; got '{method}'")
        if self.layer_kind == "GINELayerSpec" and (method == "exact" or relax_factor == 0.0):
            raise ValueError(
                "GINE reach supports approx-star only (method='approx', relax_factor > 0)")

        sets: List[GraphStar] = [input_set]
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GCNLayerSpec):
                sets = [gcn_graph_star(s, layer.W, layer.b, self.adjacency) for s in sets]
                if self.relu_layers[i]:
                    sets = self._relu(sets, method, relax_factor, lp_solver)
            elif isinstance(layer, SAGELayerSpec):
                sets = [sage_graph_star(s, layer.W_node, layer.W_edge, layer.b, self.adjacency)
                        for s in sets]
                if self.relu_layers[i]:
                    sets = self._relu(sets, method, relax_factor, lp_solver)
            elif isinstance(layer, GINELayerSpec):
                sets = [
                    gine_graph_star(s, self.E, self.edge_index, layer, self.edge_weights,
                                    self.gine_variant, apply_output_relu=self.relu_layers[i],
                                    relax_factor=relax_factor, lp_solver=lp_solver)
                    for s in sets
                ]
            else:
                raise TypeError(f"Unsupported layer spec: {type(layer).__name__}")
        return sets

    @staticmethod
    def _relu(sets, method, relax_factor, lp_solver):
        out = []
        for s in sets:
            out.extend(relu_graph_star(s, method=method,
                                       relax_factor=relax_factor, lp_solver=lp_solver))
        return out

    # ------------------------------------------------------- verify + falsify

    def verify(
        self,
        input_set: GraphStar,
        spec_lb: np.ndarray,
        spec_ub: np.ndarray,
        target_nodes: Optional[Sequence[int]] = None,
        falsify: bool = True,
        method: str = "approx",
        relax_factor: float = 0.5,
        lp_solver: str = "default",
        **falsify_kwargs,
    ) -> "VerifyResult":
        """Verify outputs stay in [spec_lb, spec_ub]; falsify the 'unknown' gap.

        Runs sound reachability first.  If every output reach set proves the
        box holds, returns 'verified'.  Otherwise (when ``falsify``) searches
        the input box for a concrete counterexample; a hit returns 'falsified'
        with a witness, a miss returns 'unknown'.

        With ``falsify``, ``input_set`` must be a box GraphStar (e.g. from
        ``from_bounds``): the falsifier searches its bounding box, which must
        not contain points outside the set.  ``method`` / ``relax_factor`` /
        ``lp_solver`` are forwarded to :meth:`reach`.
        """
        from n2v.utils.gnn_verify import verify_node_bounds

        if falsify and not _is_box(input_set):
            raise ValueError(
                "verify(falsify=True) needs a box input set (GraphStar.from_bounds); "
                "pass falsify=False to verify a general GraphStar")
        outs = self.reach(input_set, method=method,
                          relax_factor=relax_factor, lp_solver=lp_solver)
        status = verify_node_bounds(outs, spec_lb, spec_ub, target_nodes)
        if status == "verified" or not falsify:
            return VerifyResult(status=status, reach_sets=outs)

        in_lb, in_ub = input_set.get_ranges()
        res = falsify_node_bounds(self.evaluate, in_lb, in_ub, spec_lb, spec_ub,
                                  target_nodes=target_nodes, **falsify_kwargs)
        if res.found:
            return VerifyResult(status="falsified", reach_sets=outs, counterexample=res)
        return VerifyResult(status="unknown", reach_sets=outs)

    # --------------------------------------------------------- subgraph reach

    def reach_subgraph(
        self,
        input_set: GraphStar,
        target_nodes: Sequence[int],
        method: str = "approx",
        relax_factor: float = 0.5,
        lp_solver: str = "default",
    ) -> List[SubgraphResult]:
        """Per-target k-hop subgraph reachability (scalable, exact per target).

        For each target node, extracts its k-hop neighborhood (k = number of
        message-passing layers), prunes the input GraphStar to that subset, and
        runs reach on the small sub-network.  The target node's output is
        mathematically identical to full-graph reach because message passing is
        k-hop local — only boundary nodes compute unused intermediate values.

        Args:
            input_set: Full-graph input GraphStar (N nodes).
            target_nodes: 0-indexed node indices to verify.
            method / relax_factor / lp_solver: forwarded to :meth:`reach`.

        Returns:
            One :class:`SubgraphResult` per target node.
        """
        if not isinstance(input_set, GraphStar):
            raise TypeError(f"reach_subgraph expects a GraphStar, got {type(input_set).__name__}")

        k = self.num_layers                      # all layers are message-passing
        is_gine = self.layer_kind == "GINELayerSpec"
        results: List[SubgraphResult] = []

        for t in target_nodes:
            if is_gine:
                sub_nodes, sub_ei, sub_E, sub_ew, t_local = khop_subgraph_edges(
                    int(t), k, self.edge_index, self.E, self.edge_weights,
                    num_nodes=input_set.N)
                sub_gs = input_set.extract_subgraph(sub_nodes)
                sub_net = GraphNeuralNetwork(
                    self.layers, edge_index=sub_ei, E=sub_E, edge_weights=sub_ew,
                    gine_variant=self.gine_variant, relu_layers=self.relu_layers)
                n_sub_edges = sub_ei.shape[1]
            else:
                sub_nodes, sub_A, t_local = khop_subgraph_matrix(int(t), k, self.adjacency)
                sub_gs = input_set.extract_subgraph(sub_nodes, sub_adjacency=sub_A)
                sub_net = GraphNeuralNetwork(self.layers, adjacency=sub_A,
                                             relu_layers=self.relu_layers)
                n_sub_edges = int(np.count_nonzero(sub_A) - np.count_nonzero(np.diag(sub_A)))

            outs = sub_net.reach(sub_gs, method=method,
                                 relax_factor=relax_factor, lp_solver=lp_solver)
            results.append(SubgraphResult(
                target_node=int(t), target_local_idx=t_local, outputs=outs,
                n_sub_nodes=len(sub_nodes), n_sub_edges=n_sub_edges))
        return results
