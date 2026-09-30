"""
Reachability and forward evaluation for GINE (edge-feature) graph layers.

Two architecture variants are supported, matching NNV's two MATLAB GINE layers
and the two GINE ``model_type`` values of :func:`n2v.utils.load_gnn_mat`:

* ``'hugine'`` — Hu et al. 2020 GIN+E (model_type ``gine_pretrain``,
  HuGINEConvLayer.m).  Messages are raw sums (no message ReLU); self-loops are
  already present in ``edge_index`` (no separate ``(1+eps)*x`` term); a ReLU is
  applied after every layer except the last (regression output).

      agg_v = Σ_{(u,v)∈E} w_uv * (x_u + E_uv @ W_edge + b_edge)
      out_v = ReLU(agg_v @ W1 + b1) @ W2 + b2
      h_v   = ReLU(out_v)   on all but the final layer

* ``'pyg'`` — PyTorch-Geometric GINEConv (model_type ``gine_conv``,
  GINEConvLayer.m).  Messages carry a ReLU, the self term is ``(1+eps)*x``:

      msg_uv     = ReLU(x_u + E_uv @ W_edge + b_edge)
      combined_v = (1+eps)*x_v + Σ_u w_uv * msg_uv
      out_v      = ReLU(combined_v @ W1 + b1) @ W2 + b2

Node-feature perturbation is supported throughout; edge-feature perturbation
(``E`` given as a GraphStar) is supported by :func:`gine_graph_star` and
:func:`gine_stack_reach`.  The ReLUs are delegated to the existing Star-level relu_star_approx, so the
predicate space grows at each ReLU and constant terms ride on the center.
"""

from typing import List, Optional, Sequence

import numpy as np

from n2v.sets import GraphStar, Star
from n2v.nn.layer_ops.relu_reach import relu_star_approx


def _normalize_edges(edge_index, edge_weights, m):
    """Return (src, dst, weights) as 0-indexed int arrays and a float weight vector."""
    edge_index = np.asarray(edge_index)
    if edge_index.shape != (2, m):
        raise ValueError(f"edge_index shape {edge_index.shape} != (2, m={m})")
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    if edge_weights is None:
        weights = np.ones(m, dtype=np.float64)
    else:
        weights = np.asarray(edge_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != m:
            raise ValueError(f"edge_weights size {weights.shape[0]} != m={m}")
    return src, dst, weights


def _bias_row(b):
    return None if b is None else np.asarray(b, dtype=np.float64).reshape(-1)[np.newaxis, :]


def _map_features(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply X -> X @ W to every generator of a (N, F_in, K) basis tensor."""
    return np.einsum("nfk,fg->ngk", V, W)


def _relu_on_basis(V, C, d, pred_lb, pred_ub, relax_factor, lp_solver):
    """Apply approx-star ReLU to a (rows, cols, K) basis tensor.

    Returns (V_out, C_out, d_out, pred_lb_out, pred_ub_out); V_out carries the
    (possibly enlarged) generator count produced by the relaxation.
    """
    rows, cols, K = V.shape
    flat = V.reshape(rows * cols, K)
    star = Star(flat, C if (C is not None and C.size) else None,
                d if (d is not None and d.size) else None, pred_lb, pred_ub)
    outs = relu_star_approx([star], relax_factor=relax_factor, lp_solver=lp_solver)
    if len(outs) != 1:
        # Dropping any piece would under-approximate the reachable set.
        raise RuntimeError(f"approx-star ReLU returned {len(outs)} sets; expected 1")
    out = outs[0]
    K_out = out.V.shape[1]
    return (out.V.reshape(rows, cols, K_out),
            out.C, out.d, out.predicate_lb, out.predicate_ub)


def _blkdiag(C1, n1, C2, n2):
    """Block-diagonal stack of two constraint matrices over disjoint predicates."""
    r1 = C1.shape[0] if (C1 is not None and C1.size) else 0
    r2 = C2.shape[0] if (C2 is not None and C2.size) else 0
    out = np.zeros((r1 + r2, n1 + n2), dtype=np.float64)
    if r1:
        out[:r1, :n1] = C1
    if r2:
        out[r1:, n1:] = C2
    return out


def _stack_rows(a, b):
    a = np.zeros((0, 1)) if (a is None or a.size == 0) else a.reshape(-1, 1)
    b = np.zeros((0, 1)) if (b is None or b.size == 0) else b.reshape(-1, 1)
    return np.vstack([a, b])


def _build_edge_message(in_set, E_set, src, W_edge, b_edge):
    """Combine gathered node features and projected edge features into one Star.

    Node and edge perturbations live in disjoint predicate spaces, so the
    message basis is their Minkowski sum: centers add, generator blocks sit
    side by side, and the constraints stack block-diagonally.

    Returns (V_msg, C, d, lb, ub, V_self_full) where V_self_full lifts the node
    state X into the combined predicate space (zeros on the edge predicates).
    """
    n_node, n_edge = in_set.nVar, E_set.nVar
    N, F_in = in_set.N, in_set.F
    m = src.shape[0]

    E_proj_V = _map_features(E_set.V, W_edge)              # (m, F_in, 1+n_edge)
    if b_edge is not None:
        E_proj_V[:, :, 0] += np.asarray(b_edge, dtype=np.float64).reshape(-1)[np.newaxis, :]

    V_node = in_set.V[src, :, :]                           # (m, F_in, 1+n_node)

    K = 1 + n_node + n_edge
    V_msg = np.zeros((m, F_in, K), dtype=np.float64)
    V_msg[:, :, 0] = V_node[:, :, 0] + E_proj_V[:, :, 0]
    V_msg[:, :, 1:1 + n_node] = V_node[:, :, 1:]
    V_msg[:, :, 1 + n_node:] = E_proj_V[:, :, 1:]

    C = _blkdiag(in_set.C, n_node, E_set.C, n_edge)
    d = _stack_rows(in_set.d, E_set.d)
    lb = _stack_rows(in_set.predicate_lb, E_set.predicate_lb)
    ub = _stack_rows(in_set.predicate_ub, E_set.predicate_ub)

    V_self_full = np.zeros((N, F_in, K), dtype=np.float64)
    V_self_full[:, :, :1 + n_node] = in_set.V             # node state, edge preds zero
    return V_msg, C, d, lb, ub, V_self_full


# ================================================================ forward eval

def gine_evaluate(
    X: np.ndarray,
    E: np.ndarray,
    edge_index: np.ndarray,
    spec,
    edge_weights: Optional[np.ndarray] = None,
    variant: str = "hugine",
    apply_output_relu: bool = False,
) -> np.ndarray:
    """Forward pass for a single GINE layer (see module docstring for variants)."""
    X = np.asarray(X, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    N, F_in = X.shape
    m = E.shape[0]
    src, dst, w = _normalize_edges(edge_index, edge_weights, m)

    E_proj = E @ spec.W_edge
    if spec.b_edge is not None:
        E_proj = E_proj + np.asarray(spec.b_edge, dtype=np.float64).reshape(-1)[np.newaxis, :]

    edge_msg = X[src, :] + E_proj
    if variant == "pyg":
        edge_msg = np.maximum(edge_msg, 0.0)

    agg = np.zeros((N, F_in), dtype=np.float64)
    np.add.at(agg, dst, w[:, np.newaxis] * edge_msg)

    if variant == "pyg":
        combined = (1.0 + spec.eps) * X + agg
    else:  # hugine: self-loop already in edge_index
        combined = agg

    H = combined @ spec.W1
    if spec.b1 is not None:
        H = H + np.asarray(spec.b1, dtype=np.float64).reshape(-1)[np.newaxis, :]
    H = np.maximum(H, 0.0)
    Y = H @ spec.W2
    if spec.b2 is not None:
        Y = Y + np.asarray(spec.b2, dtype=np.float64).reshape(-1)[np.newaxis, :]

    if apply_output_relu:
        Y = np.maximum(Y, 0.0)
    return Y


# ================================================================ reachability

def gine_graph_star(
    in_set: GraphStar,
    E: np.ndarray,
    edge_index: np.ndarray,
    spec,
    edge_weights: Optional[np.ndarray] = None,
    variant: str = "hugine",
    apply_output_relu: bool = False,
    relax_factor: float = 0.5,
    lp_solver: str = "default",
) -> GraphStar:
    """Reach for a single GINE layer on a GraphStar input (node-only perturbation).

    Args:
        in_set: Input GraphStar (node features uncertain).
        E: Edge features.  Either a constant (m, E_in) array (node-only
            perturbation) or a GraphStar over (m, E_in) for edge perturbation,
            in which case node and edge predicate spaces are combined.
        edge_index: (2, m) edge list, 0-indexed [src; dst].
        spec: Object with attributes W1, b1, W2, b2, W_edge, b_edge, eps.
        edge_weights: Optional per-edge aggregation weights (m,); defaults to 1.
        variant: 'hugine' (gine_pretrain) or 'pyg' (gine_conv).
        apply_output_relu: Apply a ReLU after MLP2 (hugine intermediate layers).
        relax_factor: Must be > 0; ``relax_factor=0`` selects exact ReLU
            splitting in :func:`relu_star_approx`, which GINE does not support.

    Returns:
        Output GraphStar with feature dimension F_out.
    """
    if not isinstance(in_set, GraphStar):
        raise TypeError(f"gine_graph_star expects GraphStar, got {type(in_set).__name__}")
    if relax_factor == 0.0:
        raise ValueError(
            "GINE reach supports approx-star only; relax_factor=0 selects exact "
            "ReLU splitting")

    N, F_in, K_in = in_set.V.shape
    edge_perturbed = isinstance(E, GraphStar)
    m = E.N if edge_perturbed else np.asarray(E).shape[0]
    src, dst, w = _normalize_edges(edge_index, edge_weights, m)

    if edge_perturbed:
        # combined node+edge predicate space (Minkowski sum of the two Stars)
        V_edge, C, d, lb, ub, V_self_full = _build_edge_message(
            in_set, E, src, spec.W_edge, spec.b_edge)
    else:
        E = np.asarray(E, dtype=np.float64)
        E_proj = E @ spec.W_edge
        if spec.b_edge is not None:
            E_proj = E_proj + np.asarray(spec.b_edge, dtype=np.float64).reshape(-1)[np.newaxis, :]
        V_edge = in_set.V[src, :, :].copy()        # (m, F_in, K_in)
        V_edge[:, :, 0] += E_proj
        C, d, lb, ub = in_set.C, in_set.d, in_set.predicate_lb, in_set.predicate_ub
        V_self_full = in_set.V                      # already in the message space

    K0 = V_edge.shape[2]
    if variant == "pyg":
        # message ReLU couples predicates -> space may grow to K_msg
        V_edge, C, d, lb, ub = _relu_on_basis(V_edge, C, d, lb, ub, relax_factor, lp_solver)
    K = V_edge.shape[2]

    # weighted aggregate edges -> nodes
    V_agg = np.zeros((N, F_in, K), dtype=np.float64)
    np.add.at(V_agg, dst, w[:, np.newaxis, np.newaxis] * V_edge)

    if variant == "pyg":
        # self term (1+eps)*x re-expressed in the (possibly enlarged) space
        V_self = np.zeros((N, F_in, K), dtype=np.float64)
        V_self[:, :, :K0] = V_self_full
        combined = (1.0 + spec.eps) * V_self + V_agg
    else:  # hugine: self-loop already aggregated via edge_index
        combined = V_agg

    # MLP layer 1 + ReLU
    V_mlp = _map_features(combined, spec.W1)
    if spec.b1 is not None:
        V_mlp[:, :, 0] += np.asarray(spec.b1, dtype=np.float64).reshape(-1)[np.newaxis, :]
    V_mlp, C, d, lb, ub = _relu_on_basis(V_mlp, C, d, lb, ub, relax_factor, lp_solver)

    # MLP layer 2 (linear)
    V_out = _map_features(V_mlp, spec.W2)
    if spec.b2 is not None:
        V_out[:, :, 0] += np.asarray(spec.b2, dtype=np.float64).reshape(-1)[np.newaxis, :]

    # optional inter-layer ReLU (hugine, all but the final layer)
    if apply_output_relu:
        V_out, C, d, lb, ub = _relu_on_basis(V_out, C, d, lb, ub, relax_factor, lp_solver)

    return GraphStar(V_out, C, d, lb, ub, adjacency=in_set.adjacency)


def gine_stack_evaluate(
    X: np.ndarray,
    E: np.ndarray,
    edge_index: np.ndarray,
    layers: Sequence,
    variant: str = "hugine",
    edge_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Forward pass through a stack of GINE layers.

    For 'hugine', a ReLU is applied after every layer except the last (matching
    PowerFlowGINEPretrain); 'pyg' has no inter-layer ReLU.
    """
    n_layers = len(layers)
    Y = X
    for i, layer in enumerate(layers):
        out_relu = (variant == "hugine") and (i < n_layers - 1)
        Y = gine_evaluate(Y, E, edge_index, layer, edge_weights, variant, out_relu)
    return Y


def gine_stack_reach(
    in_set: GraphStar,
    E: np.ndarray,
    edge_index: np.ndarray,
    layers: Sequence,
    variant: str = "hugine",
    edge_weights: Optional[np.ndarray] = None,
    relax_factor: float = 0.5,
    lp_solver: str = "default",
) -> List[GraphStar]:
    """Reach through a stack of GINE layers using approx-star ReLU.

    ``E`` may be a constant array (node-only) or a GraphStar (edge
    perturbation).  Note: when ``E`` is a GraphStar, each layer re-combines a
    fresh copy of the edge predicates, which treats the edge uncertainty as
    independent across layers.  This is a sound over-approximation (the true
    correlated-edge reachable set is contained), but looser than threading a
    single shared edge-predicate identity through the stack.
    """
    n_layers = len(layers)
    sets: List[GraphStar] = [in_set]
    for i, layer in enumerate(layers):
        out_relu = (variant == "hugine") and (i < n_layers - 1)
        sets = [
            gine_graph_star(s, E, edge_index, layer, edge_weights, variant,
                            apply_output_relu=out_relu,
                            relax_factor=relax_factor, lp_solver=lp_solver)
            for s in sets
        ]
    return sets
