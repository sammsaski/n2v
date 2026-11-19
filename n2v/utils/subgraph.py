"""
k-hop subgraph extraction for scalable GNN verification.

For a k-layer message-passing GNN, a target node's output depends only on its
k-hop neighborhood.  Extracting that neighborhood and running reachability on
the small subgraph yields a result for the target node that is *identical* to
full-graph reachability (exact, not an approximation) — only boundary nodes at
distance k compute unused intermediate values.  This keeps per-node LP sizes
bounded regardless of full-graph size (a sparse power grid yields ~15-25 node
subgraphs for a 3-layer GNN), which is what makes IEEE-118-scale verification
tractable.

Reference: CORA GNN verification (TMLR), Corollary 17; SCIP-MPNN (ICML 2024).
Translated from NNV's khop_subgraph.m / khop_subgraph_matrix.m.
"""

from typing import List, Optional, Tuple

import numpy as np


def _bfs_khop(target_node: int, k: int, neighbors: List[np.ndarray], N: int) -> np.ndarray:
    """Return sorted 0-indexed node set within k hops of ``target_node``."""
    visited = np.zeros(N, dtype=bool)
    visited[target_node] = True
    frontier = [target_node]
    for _ in range(k):
        nxt = []
        for node in frontier:
            for nb in neighbors[node]:
                if not visited[nb]:
                    visited[nb] = True
                    nxt.append(nb)
        frontier = nxt
        if not frontier:
            break
    return np.flatnonzero(visited)            # sorted ascending


def khop_subgraph_matrix(
    target_node: int,
    k: int,
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """k-hop neighborhood for matrix-based GNNs (GCN / SAGE).

    The submatrix of ``A`` preserves full-graph degree normalization, so the
    target node's result is exact.

    Args:
        target_node: 0-indexed node to center on.
        k: number of hops (= number of message-passing layers).
        A: (N, N) adjacency (GCN normalized, or SAGE binary).

    Returns:
        sub_nodes: sorted 0-indexed original node indices in the subgraph.
        sub_A: (n_sub, n_sub) submatrix of A.
        target_local_idx: target node's position within sub_nodes.
    """
    A = np.asarray(A, dtype=np.float64)
    N = A.shape[0]
    if not (0 <= target_node < N):
        raise ValueError(f"target_node {target_node} out of range [0, {N})")

    # Neighbors from the sparsity pattern of A (row OR column nonzero).
    pattern = (A != 0.0) | (A.T != 0.0)
    neighbors = [np.flatnonzero(pattern[i]) for i in range(N)]

    sub_nodes = _bfs_khop(target_node, k, neighbors, N)
    target_local_idx = int(np.searchsorted(sub_nodes, target_node))
    sub_A = A[np.ix_(sub_nodes, sub_nodes)]
    return sub_nodes, sub_A, target_local_idx


def khop_subgraph_edges(
    target_node: int,
    k: int,
    edge_index: np.ndarray,
    E: Optional[np.ndarray] = None,
    edge_weights: Optional[np.ndarray] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]:
    """k-hop neighborhood for edge-list GNNs (GINE).

    Args:
        target_node: 0-indexed node to center on.
        k: number of hops (= number of message-passing layers).
        edge_index: (2, m) edge list [src; dst], 0-indexed.
        E: optional (m, E_in) edge features.
        edge_weights: optional (m,) aggregation weights.
        num_nodes: number of nodes N; defaults to ``edge_index.max() + 1``,
            which misses trailing nodes that have no edges.

    Returns:
        sub_nodes: sorted 0-indexed original node indices.
        sub_edge_index: (2, m_sub) edge list remapped to local 0-indexed nodes.
        sub_E: (m_sub, E_in) edge features for kept edges (or None).
        sub_edge_weights: (m_sub,) weights for kept edges (or None).
        target_local_idx: target node's position within sub_nodes.
    """
    edge_index = np.asarray(edge_index)
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be (2, m); got {edge_index.shape}")
    src, dst = edge_index[0], edge_index[1]
    N_edges = int(edge_index.max()) + 1 if edge_index.size else 0
    N = N_edges if num_nodes is None else int(num_nodes)
    if N < N_edges:
        raise ValueError(f"edge_index references node {N_edges - 1} but num_nodes={N}")
    if not (0 <= target_node < N):
        raise ValueError(f"target_node {target_node} out of range [0, {N})")

    # Undirected neighbor lookup from the (possibly directed) edge list.
    neighbors: List[List[int]] = [[] for _ in range(N)]
    for s, d in zip(src.tolist(), dst.tolist()):
        neighbors[s].append(d)
        neighbors[d].append(s)
    neighbors = [np.asarray(nb, dtype=np.int64) for nb in neighbors]

    sub_nodes = _bfs_khop(target_node, k, neighbors, N)
    node_map = -np.ones(N, dtype=np.int64)
    node_map[sub_nodes] = np.arange(sub_nodes.size)
    target_local_idx = int(node_map[target_node])

    in_sub = np.zeros(N, dtype=bool)
    in_sub[sub_nodes] = True
    edge_mask = in_sub[src] & in_sub[dst]
    sub_edge_index = np.vstack([node_map[src[edge_mask]], node_map[dst[edge_mask]]])
    sub_E = None if E is None else np.asarray(E)[edge_mask]
    sub_ew = None if edge_weights is None else np.asarray(edge_weights).reshape(-1)[edge_mask]
    return sub_nodes, sub_edge_index, sub_E, sub_ew, target_local_idx
