"""
Graph-level pooling reachability.

Sum and mean pooling collapse a GraphStar of shape (N, F) into a graph-level
embedding of shape (1, F).  Both operations are linear in the basis tensor
so reachability is exact: the predicate constraints are unchanged.

Mirrors AddPoolLayer.m in NNV's MATLAB GNN implementation, with mean pooling
added.
"""

from typing import Optional

import numpy as np

from n2v.sets import GraphStar, Star


def sum_pool_evaluate(X: np.ndarray) -> np.ndarray:
    """Forward pass: g = sum(X, axis=0), shape (1, F)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, F); got shape {X.shape}")
    return X.sum(axis=0, keepdims=True)


def mean_pool_evaluate(X: np.ndarray) -> np.ndarray:
    """Forward pass: g = mean(X, axis=0), shape (1, F)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, F); got shape {X.shape}")
    return X.mean(axis=0, keepdims=True)


def sum_pool_graph_star(in_set: GraphStar) -> GraphStar:
    """Reach for graph-level sum pooling on a GraphStar.

    Returns a GraphStar of shape (1, F) — exact, since summation is linear.
    """
    if not isinstance(in_set, GraphStar):
        raise TypeError(f"sum_pool_graph_star expects GraphStar, got {type(in_set).__name__}")

    # V has shape (N, F, nVar+1); collapse the node axis.
    new_V = in_set.V.sum(axis=0, keepdims=True)
    return GraphStar(
        new_V,
        in_set.C if in_set.C.size else None,
        in_set.d if in_set.d.size else None,
        in_set.predicate_lb,
        in_set.predicate_ub,
        adjacency=None,  # adjacency no longer meaningful after pooling
    )


def mean_pool_graph_star(in_set: GraphStar) -> GraphStar:
    """Reach for graph-level mean pooling on a GraphStar.

    Returns a GraphStar of shape (1, F) — exact, since the average is a
    fixed scaling of the sum.
    """
    if not isinstance(in_set, GraphStar):
        raise TypeError(f"mean_pool_graph_star expects GraphStar, got {type(in_set).__name__}")
    if in_set.N == 0:
        raise ValueError("Cannot mean-pool a GraphStar with zero nodes")

    new_V = in_set.V.mean(axis=0, keepdims=True)
    return GraphStar(
        new_V,
        in_set.C if in_set.C.size else None,
        in_set.d if in_set.d.size else None,
        in_set.predicate_lb,
        in_set.predicate_ub,
        adjacency=None,
    )


def graph_star_to_star(in_set: GraphStar) -> Star:
    """Convert a graph-level (1, F) GraphStar into a Star of dim F.

    This is the bridge from the GNN encoder to a standard fully-connected
    classification head, which uses the existing linear_reach machinery.
    """
    if not isinstance(in_set, GraphStar):
        raise TypeError(f"graph_star_to_star expects GraphStar, got {type(in_set).__name__}")
    if in_set.N != 1:
        raise ValueError(
            f"graph_star_to_star requires a pooled GraphStar with N==1; got N={in_set.N}"
        )
    return in_set.to_star()
