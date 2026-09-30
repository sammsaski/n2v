"""
Reachability and forward evaluation for GraphSAGE convolution layers.

A SAGEConv layer computes Y = X @ W_node + (A @ X) @ W_edge + b, where A is a
raw binary adjacency matrix (sum aggregation, no self-loops, no degree
normalization).  Both the self transform and the neighbor transform are linear
in X, so on a GraphStar the layer is exact: it reduces to a sum of two affine
maps that share the same predicate variables.

Mirrors SAGEConvLayer.m from NNV's MATLAB GNN implementation.
ReLU between layers is delegated to the existing Star-level relaxation via
relu_graph_star, exactly as the GCN stack does.
"""

from typing import List, Optional, Sequence

import numpy as np

from n2v.sets import GraphStar
from n2v.nn.layer_ops.gcn_reach import relu_graph_star


def sage_evaluate(
    X: np.ndarray,
    A: np.ndarray,
    W_node: np.ndarray,
    W_edge: np.ndarray,
    b: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Forward pass for a single SAGEConv layer.

    Mirrors SAGEConvLayer.evaluate from the MATLAB GNNV implementation.

    Args:
        X: Node features (N, F_in).
        A: Binary adjacency (N, N), already oriented for ``A @ X`` aggregation.
        W_node: Self (root) weight matrix (F_in, F_out).
        W_edge: Neighbor weight matrix (F_in, F_out).
        b: Optional bias (F_out,) or (F_out, 1).

    Returns:
        Output node features (N, F_out).
    """
    X = np.asarray(X, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    W_node = np.asarray(W_node, dtype=np.float64)
    W_edge = np.asarray(W_edge, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, F); got shape {X.shape}")
    if A.shape[0] != A.shape[1] or A.shape[0] != X.shape[0]:
        raise ValueError(f"A shape {A.shape} incompatible with X {X.shape}")
    if W_node.shape != W_edge.shape:
        raise ValueError(
            f"W_node {W_node.shape} and W_edge {W_edge.shape} must match"
        )
    if W_node.shape[0] != X.shape[1]:
        raise ValueError(f"W shape {W_node.shape} incompatible with F_in={X.shape[1]}")

    Y = X @ W_node + (A @ X) @ W_edge
    if b is not None:
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if b.shape[0] != W_node.shape[1]:
            raise ValueError(f"Bias size {b.shape[0]} != output features {W_node.shape[1]}")
        Y = Y + b[np.newaxis, :]
    return Y


def sage_graph_star(
    in_set: GraphStar,
    W_node: np.ndarray,
    W_edge: np.ndarray,
    b: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
) -> GraphStar:
    """Reach for a SAGEConv layer on a GraphStar input.

    SAGEConv is linear in X for fixed A, so the reach is exact.  The self term
    ``X @ W_node`` and the neighbor term ``(A @ X) @ W_edge`` are computed as
    predicate-preserving affine maps and summed; both share the input's
    predicate variables and constraints.

    Args:
        in_set: Input GraphStar.
        W_node: Self weight (F_in, F_out).
        W_edge: Neighbor weight (F_in, F_out).
        b: Optional bias.
        A: Binary adjacency (N, N).  If None, the GraphStar's own adjacency
            attribute is used.

    Returns:
        Output GraphStar with feature dimension F_out.
    """
    if not isinstance(in_set, GraphStar):
        raise TypeError(f"sage_graph_star expects GraphStar, got {type(in_set).__name__}")

    if A is None:
        if in_set.adjacency is None:
            raise ValueError("No adjacency available; pass A or attach it to the GraphStar")
        A = in_set.adjacency

    node_term = in_set.affine_map_right(W_node, b)          # X @ W_node + b
    edge_term = in_set.affine_map_left(A).affine_map_right(W_edge)  # (A @ X) @ W_edge
    return node_term.add_set(edge_term)


def sage_stack_evaluate(
    X: np.ndarray,
    A: np.ndarray,
    layers: Sequence,
    has_relu: bool = True,
) -> np.ndarray:
    """Forward pass through a stack of SAGEConv(+ReLU) layers.

    Args:
        X: Node features (N, F_in).
        A: Binary adjacency (N, N).
        layers: Sequence of objects with attributes ``W_node``, ``W_edge``,
            ``b`` (e.g. ``SAGELayerSpec`` from ``gnn_loader``).
        has_relu: If True, apply ReLU after every layer.

    Returns:
        Output node features.
    """
    Y = X
    for layer in layers:
        Y = sage_evaluate(Y, A, layer.W_node, layer.W_edge, layer.b)
        if has_relu:
            Y = np.maximum(Y, 0.0)
    return Y


def sage_stack_reach(
    in_set: GraphStar,
    layers: Sequence,
    has_relu: bool = True,
    A: Optional[np.ndarray] = None,
    relax_factor: float = 0.5,
    lp_solver: str = "default",
) -> List[GraphStar]:
    """Reach through a stack of SAGEConv(+ReLU) layers using approx-star ReLU.

    Returns:
        List of output GraphStars.  approx-star produces a single set per
        ReLU, so the list length stays at 1 throughout the stack.
    """
    if A is None:
        if in_set.adjacency is None:
            raise ValueError("No adjacency available; pass A or attach it to the GraphStar")
        A = in_set.adjacency

    sets: List[GraphStar] = [in_set]
    for layer in layers:
        sets = [sage_graph_star(s, layer.W_node, layer.W_edge, layer.b, A) for s in sets]
        if has_relu:
            next_sets: List[GraphStar] = []
            for s in sets:
                next_sets.extend(
                    relu_graph_star(s, method="approx",
                                    relax_factor=relax_factor, lp_solver=lp_solver)
                )
            sets = next_sets
    return sets
