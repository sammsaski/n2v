"""Unit tests for k-hop subgraph extraction and GraphStar.extract_subgraph."""

import numpy as np
import pytest

from n2v.sets import GraphStar
from n2v.utils import khop_subgraph_matrix, khop_subgraph_edges


pytestmark = pytest.mark.unit


# ============================================================ matrix k-hop


def _path_adjacency(n):
    """0-1-2-...-(n-1) path graph with self-loops."""
    A = np.eye(n)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A


def test_matrix_khop_grows_with_hops():
    A = _path_adjacency(7)
    for k, expect in [(0, {3}), (1, {2, 3, 4}), (2, {1, 2, 3, 4, 5})]:
        sub, subA, loc = khop_subgraph_matrix(3, k, A)
        assert set(sub.tolist()) == expect
        assert sub[loc] == 3                       # local index points back to target
        assert subA.shape == (len(sub), len(sub))


def test_matrix_khop_submatrix_values_preserved():
    A = _path_adjacency(6)
    sub, subA, _ = khop_subgraph_matrix(0, 2, A)   # nodes {0,1,2}
    np.testing.assert_array_equal(subA, A[np.ix_(sub, sub)])


def test_matrix_khop_out_of_range():
    A = _path_adjacency(4)
    with pytest.raises(ValueError):
        khop_subgraph_matrix(9, 1, A)


# ============================================================ edge-list k-hop


def test_edge_khop_remaps_and_filters():
    # path 0-1-2-3 as a directed edge list (both directions) + self loops
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    edges += [(i, i) for i in range(4)]
    edge_index = np.array(edges).T
    E = np.arange(len(edges)).reshape(-1, 1).astype(float)
    sub_nodes, sub_ei, sub_E, sub_ew, loc = khop_subgraph_edges(1, 1, edge_index, E)
    assert set(sub_nodes.tolist()) == {0, 1, 2}     # 1-hop around node 1
    assert sub_nodes[loc] == 1
    # all remapped endpoints are valid local indices
    assert sub_ei.min() >= 0 and sub_ei.max() < len(sub_nodes)
    # only edges with both endpoints kept survive; features track the mask
    assert sub_E.shape[0] == sub_ei.shape[1]


def test_edge_khop_weights_optional():
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    sub_nodes, sub_ei, sub_E, sub_ew, loc = khop_subgraph_edges(0, 5, edge_index)
    assert sub_E is None and sub_ew is None
    assert set(sub_nodes.tolist()) == {0, 1, 2}     # whole connected graph


# ====================================================== GraphStar.extract_subgraph


def test_extract_subgraph_prunes_dead_predicates():
    # node 0 perturbed, node 1 fixed -> slicing to {1} drops node-0 predicates
    lb = np.array([[-1.0, -1.0], [2.0, 3.0]])
    ub = np.array([[1.0, 1.0], [2.0, 3.0]])
    gs = GraphStar.from_bounds(lb, ub)
    assert gs.nVar == 2                              # two live entries on node 0
    sub = gs.extract_subgraph([1])
    assert sub.N == 1 and sub.nVar == 0             # node 1 is constant
    np.testing.assert_allclose(sub.center(), [[2.0, 3.0]])


def test_extract_subgraph_keeps_relevant_predicates_and_bounds():
    lb = np.array([[-1.0, -1.0], [-2.0, -2.0], [5.0, 5.0]])
    ub = np.array([[1.0, 1.0], [2.0, 2.0], [5.0, 5.0]])
    gs = GraphStar.from_bounds(lb, ub)
    sub = gs.extract_subgraph([0, 1])               # keep perturbed nodes
    slb, sub_ub = sub.get_ranges()
    np.testing.assert_allclose(slb, lb[[0, 1]], atol=1e-9)
    np.testing.assert_allclose(sub_ub, ub[[0, 1]], atol=1e-9)


def test_extract_subgraph_attaches_adjacency():
    gs = GraphStar.from_bounds(np.zeros((3, 2)), np.ones((3, 2)))
    A = np.eye(2)
    sub = gs.extract_subgraph([0, 1], sub_adjacency=A)
    assert sub.adjacency is not None and sub.adjacency.shape == (2, 2)


def test_extract_subgraph_sound_with_coupled_constraints():
    """Regression: pruning a predicate coupled to a kept one must RELAX (stay
    sound), never pin it to 0 (which shrank the set below true reachable)."""
    # node0 = 5 + a0, node1 = 7 + a1, with a0 + a1 <= 0.5, a0,a1 in [-1, 1].
    V = np.zeros((2, 1, 3))
    V[0, 0, 0], V[0, 0, 1] = 5.0, 1.0
    V[1, 0, 0], V[1, 0, 2] = 7.0, 1.0
    gs = GraphStar(V, np.array([[1.0, 1.0]]), np.array([[0.5]]),
                   np.array([[-1.0], [-1.0]]), np.array([[1.0], [1.0]]))
    full_lb, full_ub = gs.get_ranges()
    sub = gs.extract_subgraph([0])
    slb, sub_ub = sub.get_ranges()
    # subgraph reach for the kept node must CONTAIN the full-graph reach
    assert sub_ub[0, 0] >= full_ub[0, 0] - 1e-9
    assert slb[0, 0] <= full_lb[0, 0] + 1e-9


def test_extract_subgraph_random_coupled_constraints_sound():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n, F = int(rng.integers(3, 6)), 2
        V = np.zeros((n, F, n + 1))
        V[:, :, 0] = rng.normal(size=(n, F))
        for i in range(n):
            V[i, 0, i + 1] = 1.0
        nc = int(rng.integers(1, 4))
        gs = GraphStar(V, rng.normal(size=(nc, n)), rng.uniform(0.5, 2.0, size=(nc, 1)),
                       -np.ones((n, 1)), np.ones((n, 1)))
        keep = sorted(rng.choice(n, size=int(rng.integers(1, n)), replace=False))
        full_lb, full_ub = gs.get_ranges()
        slb, sub_ub = gs.extract_subgraph(keep).get_ranges()
        for li, gi in enumerate(keep):
            assert np.all(sub_ub[li] >= full_ub[gi] - 1e-6)
            assert np.all(slb[li] <= full_lb[gi] + 1e-6)


def test_edge_khop_num_nodes_allows_trailing_isolated_nodes():
    edge_index = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        khop_subgraph_edges(3, 1, edge_index)
    sub_nodes, sub_ei, _, _, loc = khop_subgraph_edges(3, 1, edge_index, num_nodes=4)
    assert sub_nodes.tolist() == [3] and loc == 0 and sub_ei.shape == (2, 0)
    with pytest.raises(ValueError):
        khop_subgraph_edges(0, 1, edge_index, num_nodes=1)
