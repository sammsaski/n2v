"""Exact (ReLU-splitting) reach for GCN/SAGE, and approx-only enforcement for GINE.

Exact reach returns a union of GraphStars.  Every consumer must use the whole
union: keeping any single piece under-approximates the reachable set and can
certify a false property (regression for verify()/reach_subgraph() taking
``reach(...)[0]``).
"""

import numpy as np
import pytest

from n2v.nn import GraphNeuralNetwork
from n2v.sets import GraphStar
from n2v.utils import GCNLayerSpec, SAGELayerSpec, GINELayerSpec
from n2v.nn.layer_ops.gine_reach import gine_graph_star


pytestmark = pytest.mark.unit

EPS = 0.2
PATH_A = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)


def _toy(kind, seed=2):
    """3-node path graph, 2 layers, F=2; seed 2 splits into tens of pieces."""
    rng = np.random.default_rng(seed)
    N, F = 3, 2
    if kind == "gcn":
        deg = PATH_A.sum(1)
        A = PATH_A / np.sqrt(np.outer(deg, deg))
        layers = [GCNLayerSpec(rng.normal(size=(F, F)), rng.normal(size=F) * 0.1)
                  for _ in range(2)]
    else:
        A = PATH_A
        layers = [SAGELayerSpec(rng.normal(size=(F, F)), rng.normal(size=(F, F)) * 0.5,
                                rng.normal(size=F) * 0.1) for _ in range(2)]
    net = GraphNeuralNetwork(layers, adjacency=A)
    base = rng.normal(size=(N, F)) * 0.2
    gs = GraphStar.from_bounds(base - EPS, base + EPS, adjacency=A)
    return net, gs, base


def _samples(net, base, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = [base + rng.uniform(-EPS, EPS, size=base.shape) for _ in range(n)]
    # include box vertices, where ReLU splits matter most
    X += [base + EPS * rng.choice([-1.0, 1.0], size=base.shape) for _ in range(n)]
    return np.array([net.evaluate(x) for x in X])


def _hull(sets):
    ranges = [s.get_ranges() for s in sets]
    return (np.min([r[0] for r in ranges], axis=0),
            np.max([r[1] for r in ranges], axis=0))


@pytest.mark.parametrize("kind", ["gcn", "sage"])
def test_exact_reach_union_contains_samples(kind):
    net, gs, base = _toy(kind)
    sets = net.reach(gs, method="exact")
    assert len(sets) > 1
    boxes = [s.get_ranges() for s in sets]
    for Y in _samples(net, base):
        assert any(np.all(Y >= lb - 1e-7) and np.all(Y <= ub + 1e-7) for lb, ub in boxes)


@pytest.mark.parametrize("kind", ["gcn", "sage"])
def test_exact_hull_within_approx(kind):
    net, gs, _ = _toy(kind)
    ex_lb, ex_ub = _hull(net.reach(gs, method="exact"))
    ap_lb, ap_ub = net.reach(gs)[0].get_ranges()
    assert np.all(ex_lb >= ap_lb - 1e-7) and np.all(ex_ub <= ap_ub + 1e-7)


@pytest.mark.parametrize("kind", ["gcn", "sage"])
@pytest.mark.parametrize("opts", [{"method": "exact"},
                                  {"method": "approx", "relax_factor": 0.0}])
def test_verify_exact_no_false_verified(kind, opts):
    """A spec fitted to one exact piece is violated by real outputs -> never 'verified'."""
    net, gs, base = _toy(kind)
    # relax_factor=0 yields the exact pieces too (and predates method='exact')
    lb0, ub0 = net.reach(gs, relax_factor=0.0)[0].get_ranges()
    spec_lb, spec_ub = lb0 - 1e-6, ub0 + 1e-6        # strictly contains piece 0
    Y = _samples(net, base)
    assert np.any((Y < spec_lb) | (Y > spec_ub))       # spec really is violated

    res = net.verify(gs, spec_lb, spec_ub, falsify=False, **opts)
    assert res.status != "verified"
    assert len(res.reach_sets) > 1


@pytest.mark.parametrize("kind", ["gcn", "sage"])
def test_verify_exact_proves_hull_spec(kind):
    net, gs, _ = _toy(kind)
    lb, ub = _hull(net.reach(gs, method="exact"))
    res = net.verify(gs, lb - 1e-6, ub + 1e-6, falsify=False, method="exact")
    assert res.status == "verified"


def test_verify_result_single_set_accessor():
    net, gs, _ = _toy("gcn")
    approx = net.verify(gs, -np.inf * np.ones((3, 2)), np.inf * np.ones((3, 2)), falsify=False)
    assert approx.reach_set is approx.reach_sets[0]
    exact = net.verify(gs, -np.inf * np.ones((3, 2)), np.inf * np.ones((3, 2)),
                       falsify=False, method="exact")
    with pytest.raises(ValueError):
        exact.reach_set


@pytest.mark.parametrize("kind", ["gcn", "sage"])
def test_reach_subgraph_exact_keeps_all_pieces(kind):
    net, gs, _ = _toy(kind)
    full_lb, full_ub = _hull(net.reach(gs, method="exact"))
    for r in net.reach_subgraph(gs, [0, 2], method="exact"):
        assert len(r.outputs) > 1
        with pytest.raises(ValueError):
            r.output
        lb, ub = r.target_ranges()
        np.testing.assert_allclose(lb, full_lb[r.target_node], atol=1e-6)
        np.testing.assert_allclose(ub, full_ub[r.target_node], atol=1e-6)


def _gine_net():
    rng = np.random.default_rng(0)
    F, H, Fe = 2, 3, 2
    layer = GINELayerSpec(rng.normal(size=(F, H)), rng.normal(size=H),
                          rng.normal(size=(H, F)), rng.normal(size=F),
                          rng.normal(size=(Fe, F)), None, 0.0)
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    E = rng.normal(size=(4, Fe))
    net = GraphNeuralNetwork([layer], edge_index=edge_index, E=E, gine_variant="pyg")
    gs = GraphStar.from_bounds(-0.5 * np.ones((3, F)), 0.5 * np.ones((3, F)))
    return net, gs, layer, edge_index, E


@pytest.mark.parametrize("opts", [{"method": "exact"},
                                  {"method": "approx", "relax_factor": 0.0}])
def test_gine_rejects_exact(opts):
    net, gs, layer, edge_index, E = _gine_net()
    with pytest.raises(ValueError):
        net.reach(gs, **opts)
    with pytest.raises(ValueError):
        net.verify(gs, -np.ones((3, 2)), np.ones((3, 2)), falsify=False, **opts)
    with pytest.raises(ValueError):
        net.reach_subgraph(gs, [1], **opts)


def test_gine_layer_op_rejects_relax_factor_zero():
    _, gs, layer, edge_index, E = _gine_net()
    with pytest.raises(ValueError):
        gine_graph_star(gs, E, edge_index, layer, variant="pyg", relax_factor=0.0)
