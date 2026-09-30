"""Exactness + scaling tests for GraphNeuralNetwork.reach_subgraph.

k-hop subgraph reach must reproduce full-graph reach bounds at the target node
exactly (message passing is k-hop local), for every layer family.  Synthetic
graphs run always; the IEEE-118 checkpoints exercise real-scale subgraphs.
"""

import numpy as np
import pytest

from n2v.nn import GraphNeuralNetwork, SubgraphResult
from n2v.sets import GraphStar
from n2v.utils import load_gnn_mat, GCNLayerSpec, SAGELayerSpec, GINELayerSpec
from tests.fixtures.gnn import checkpoint


pytestmark = pytest.mark.unit


def _ring_adjacency(n):
    A = np.eye(n)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
    return A


def _ring_edges(n):
    src, dst = [], []
    for i in range(n):
        for a, b in [(i, (i + 1) % n), ((i + 1) % n, i), (i, i)]:
            src.append(a); dst.append(b)
    return np.array([src, dst])


def _assert_subgraph_matches_full(net, gs, targets):
    full = net.reach(gs)[0]
    flb, fub = full.get_ranges()
    results = net.reach_subgraph(gs, targets)
    assert all(isinstance(r, SubgraphResult) for r in results)
    for r in results:
        slb, sub = r.target_ranges()
        np.testing.assert_allclose(slb, flb[r.target_node], atol=1e-7)
        np.testing.assert_allclose(sub, fub[r.target_node], atol=1e-7)
    return results


def test_subgraph_exact_for_gcn_synthetic():
    rng = np.random.default_rng(0)
    N, F = 8, 3
    A = _ring_adjacency(N)
    layers = [GCNLayerSpec(rng.normal(size=(F, F)), rng.normal(size=(F,))) for _ in range(2)]
    net = GraphNeuralNetwork(layers, adjacency=A, has_relu=True)
    base = rng.normal(size=(N, F))
    gs = GraphStar.from_bounds(base - 0.02, base + 0.02, adjacency=A)
    res = _assert_subgraph_matches_full(net, gs, [0, 3, 6])
    assert all(r.n_sub_nodes < N for r in res)      # locality actually shrank the graph


def test_subgraph_exact_for_sage_synthetic():
    rng = np.random.default_rng(1)
    N, F = 9, 3
    A = _ring_adjacency(N)
    layers = [SAGELayerSpec(rng.normal(size=(F, F)), rng.normal(size=(F, F)),
                            rng.normal(size=(F,))) for _ in range(2)]
    net = GraphNeuralNetwork(layers, adjacency=A, has_relu=True)
    base = rng.normal(size=(N, F))
    gs = GraphStar.from_bounds(base - 0.02, base + 0.02, adjacency=A)
    _assert_subgraph_matches_full(net, gs, [0, 4, 8])


def test_subgraph_exact_for_gine_synthetic():
    rng = np.random.default_rng(2)
    N, F, hidden, E_in = 10, 3, 5, 2
    edge_index = _ring_edges(N)
    m = edge_index.shape[1]
    E = rng.normal(size=(m, E_in))
    layers = [GINELayerSpec(
        rng.normal(size=(F, hidden)), rng.normal(size=(hidden,)),
        rng.normal(size=(hidden, F)), rng.normal(size=(F,)),
        rng.normal(size=(E_in, F)), rng.normal(size=(F,)), 0.0) for _ in range(2)]
    net = GraphNeuralNetwork(layers, edge_index=edge_index, E=E, gine_variant="hugine")
    base = rng.normal(size=(N, F))
    gs = GraphStar.from_bounds(base - 0.01, base + 0.01)
    res = _assert_subgraph_matches_full(net, gs, [0, 5])
    assert all(r.n_sub_nodes < N for r in res)


@pytest.mark.parametrize("case", ["gcn", "sage", "gine_pretrain"])
def test_subgraph_exact_on_ieee118(case):
    path = checkpoint(case, "ieee118")
    if not path.exists():
        pytest.skip(f"requires {path}")
    model = load_gnn_mat(path)
    net = GraphNeuralNetwork.from_model(model)
    X = model.X_test[0]
    N = X.shape[0]
    eps = 1e-3
    gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=model.adjacency)
    results = _assert_subgraph_matches_full(net, gs, [0, 40, 80])
    # subgraphs must be strictly smaller than the full 118-node graph
    assert all(r.n_sub_nodes < N for r in results)
