"""
Reachability and forward evaluation for graph-convolutional layers.

A GCN layer computes Y = (A_norm @ X) @ W + b, broadcasting bias across nodes.
Both ops are linear in X, so on a GraphStar the layer reduces to two affine
maps composed in order: aggregate, then per-node feature transform.

ReLU on a GraphStar is delegated to the existing Star-level relu_star_approx
via flatten / unflatten — no new activation logic needed.
"""

from typing import List, Optional, Sequence

import numpy as np

from n2v.sets import GraphStar, Star
from n2v.nn.layer_ops.relu_reach import relu_star_approx, relu_star_exact


def gcn_evaluate(
    X: np.ndarray,
    A_norm: np.ndarray,
    W: np.ndarray,
    b: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Forward pass for a single GCN layer.

    Mirrors GCNLayer.evaluate from the MATLAB GNNV implementation.

    Args:
        X: Node features (N, F_in).
        A_norm: Normalized adjacency (N, N).
        W: Weight matrix (F_in, F_out).
        b: Optional bias (F_out,) or (F_out, 1).

    Returns:
        Output node features (N, F_out).
    """
    X = np.asarray(X, dtype=np.float64)
    A_norm = np.asarray(A_norm, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, F); got shape {X.shape}")
    if A_norm.shape[0] != A_norm.shape[1] or A_norm.shape[0] != X.shape[0]:
        raise ValueError(f"A_norm shape {A_norm.shape} incompatible with X {X.shape}")
    if W.shape[0] != X.shape[1]:
        raise ValueError(f"W shape {W.shape} incompatible with X feature dim {X.shape[1]}")

    Y = (A_norm @ X) @ W
    if b is not None:
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if b.shape[0] != W.shape[1]:
            raise ValueError(f"Bias size {b.shape[0]} != output features {W.shape[1]}")
        Y = Y + b[np.newaxis, :]
    return Y


def gcn_graph_star(
    in_set: GraphStar,
    W: np.ndarray,
    b: Optional[np.ndarray] = None,
    A_norm: Optional[np.ndarray] = None,
) -> GraphStar:
    """Reach for a GCN layer on a GraphStar input.

    Args:
        in_set: Input GraphStar.
        W: Weight matrix (F_in, F_out).
        b: Optional bias.
        A_norm: Normalized adjacency.  If None, the GraphStar's own
            adjacency attribute is used.

    Returns:
        Output GraphStar with the new feature dimension F_out.
    """
    if not isinstance(in_set, GraphStar):
        raise TypeError(f"gcn_graph_star expects GraphStar, got {type(in_set).__name__}")

    if A_norm is None:
        if in_set.adjacency is None:
            raise ValueError("No adjacency available; pass A_norm or attach it to the GraphStar")
        A_norm = in_set.adjacency

    return in_set.affine_map_left(A_norm).affine_map_right(W, b)


def relu_graph_star(
    in_set: GraphStar,
    method: str = "approx",
    relax_factor: float = 0.5,
    lp_solver: str = "default",
) -> List[GraphStar]:
    """Reach for an element-wise ReLU on a GraphStar.

    Reuses the existing Star-level ReLU implementation by flattening to
    Star, applying the relaxation, then reshaping back into one or more
    GraphStars.  'approx' returns a single set; 'exact' may return many.

    Args:
        in_set: Input GraphStar.
        method: 'approx' (triangle relaxation, single output) or 'exact'
            (Star splitting, possibly many outputs).  Default 'approx'
            matches NNV's GNN reach, which uses approx-star throughout.

    Returns:
        List of output GraphStars.  Always non-empty for non-trivial inputs.
    """
    if not isinstance(in_set, GraphStar):
        raise TypeError(f"relu_graph_star expects GraphStar, got {type(in_set).__name__}")

    flat = in_set.to_star()
    if method == "approx":
        out_stars = relu_star_approx([flat], relax_factor=relax_factor, lp_solver=lp_solver)
    elif method == "exact":
        out_stars = relu_star_exact([flat], lp_solver=lp_solver)
    else:
        raise ValueError(f"Unknown ReLU method '{method}'; expected 'approx' or 'exact'")

    return [
        GraphStar.from_star(s, in_set.N, in_set.F, adjacency=in_set.adjacency)
        for s in out_stars
    ]


def gcn_stack_evaluate(
    X: np.ndarray,
    A_norm: np.ndarray,
    layers: Sequence,
    has_relu: bool = True,
) -> np.ndarray:
    """Forward pass through a stack of GCN(+ReLU) layers.

    Args:
        X: Node features (N, F_in).
        A_norm: Normalized adjacency (N, N).
        layers: Sequence of objects with attributes ``W`` and ``b`` (e.g.,
            ``GCNLayerSpec`` from ``gnn_loader``).
        has_relu: If True, apply ReLU after every layer (matches the
            MATLAB ``gnn2nnv.m`` default).

    Returns:
        Output node features.
    """
    Y = X
    for layer in layers:
        Y = gcn_evaluate(Y, A_norm, layer.W, layer.b)
        if has_relu:
            Y = np.maximum(Y, 0.0)
    return Y


def gcn_stack_reach(
    in_set: GraphStar,
    layers: Sequence,
    has_relu: bool = True,
    A_norm: Optional[np.ndarray] = None,
    relax_factor: float = 0.5,
    lp_solver: str = "default",
) -> List[GraphStar]:
    """Reach through a stack of GCN(+ReLU) layers using approx-star ReLU.

    Returns:
        List of output GraphStars.  approx-star produces a single set per
        ReLU, so the list length stays at 1 throughout the stack.
    """
    if A_norm is None:
        if in_set.adjacency is None:
            raise ValueError("No adjacency available; pass A_norm or attach it to the GraphStar")
        A_norm = in_set.adjacency

    sets: List[GraphStar] = [in_set]
    for layer in layers:
        sets = [gcn_graph_star(s, layer.W, layer.b, A_norm) for s in sets]
        if has_relu:
            next_sets: List[GraphStar] = []
            for s in sets:
                next_sets.extend(
                    relu_graph_star(s, method="approx",
                                    relax_factor=relax_factor, lp_solver=lp_solver)
                )
            sets = next_sets
    return sets
