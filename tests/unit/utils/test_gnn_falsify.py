"""Tests for GNN counterexample search (falsification) and the verify method."""

import numpy as np
import pytest

from n2v.sets import GraphStar
from n2v.nn import GraphNeuralNetwork
from n2v.utils import load_gnn_mat, SAGELayerSpec, falsify_node_bounds
from tests.fixtures.gnn import checkpoint


pytestmark = pytest.mark.unit


def _toy_net(seed=0):
    rng = np.random.default_rng(seed)
    n, F = 5, 3
    A = np.eye(n)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    layers = [SAGELayerSpec(rng.normal(size=(F, F)), rng.normal(size=(F, F)),
                            rng.normal(size=(F,))) for _ in range(2)]
    return GraphNeuralNetwork(layers, adjacency=A, has_relu=True), rng


# ============================================================ falsify_node_bounds


def test_falsify_finds_real_counterexample():
    net, rng = _toy_net(1)
    base = rng.normal(size=(5, 3))
    eps = 0.05
    lb, ub = base - eps, base + eps
    Yc = net.evaluate(base)                       # reachable
    # spec that excludes the reachable center -> a counterexample exists
    res = falsify_node_bounds(net.evaluate, lb, ub, Yc + 0.05, Yc + 100.0, seed=0)
    assert res.found
    # the witness is a genuine violation: in the input box and out of the spec box
    Xc = res.counterexample
    assert np.all(Xc >= lb - 1e-9) and np.all(Xc <= ub + 1e-9)
    Yv = net.evaluate(Xc)
    assert np.any(Yv < Yc + 0.05 - 1e-9)
    assert res.margin > 0


def test_falsify_reports_none_when_spec_holds():
    """When the spec contains the whole reachable set, no counterexample exists."""
    net, rng = _toy_net(2)
    base = rng.normal(size=(5, 3))
    eps = 0.05
    lb, ub = base - eps, base + eps
    # bound the true range by dense sampling, then give the spec a safety margin
    samples = np.array([net.evaluate(lb + rng.uniform(size=(5, 3)) * (ub - lb))
                        for _ in range(400)])
    smin, smax = samples.min(0), samples.max(0)
    res = falsify_node_bounds(net.evaluate, lb, ub, smin - 1.0, smax + 1.0, seed=1)
    assert not res.found


def test_falsify_never_returns_spurious_witness():
    """Any returned counterexample must be a real forward-eval violation."""
    net, rng = _toy_net(3)
    for _ in range(10):
        base = rng.normal(size=(5, 3))
        eps = 0.05
        lb, ub = base - eps, base + eps
        spec_lb = rng.normal(size=(5, 3))
        spec_ub = spec_lb + rng.uniform(0.5, 3.0, size=(5, 3))
        res = falsify_node_bounds(net.evaluate, lb, ub, spec_lb, spec_ub, seed=2)
        if res.found:
            Yv = net.evaluate(res.counterexample)
            assert np.any(Yv < spec_lb - 1e-9) or np.any(Yv > spec_ub + 1e-9)


def test_falsify_target_nodes_subset():
    net, rng = _toy_net(4)
    base = rng.normal(size=(5, 3))
    eps = 0.05
    lb, ub = base - eps, base + eps
    Yc = net.evaluate(base)
    # violate only node 2's lower bound; restrict the search to node 2
    spec_lb = Yc - 100.0
    spec_lb[2] = Yc[2] + 0.05
    res = falsify_node_bounds(net.evaluate, lb, ub, spec_lb, Yc + 100.0,
                              target_nodes=[2], seed=0)
    assert res.found
    assert np.any(net.evaluate(res.counterexample)[2] < spec_lb[2] - 1e-9)


# ================================================================ net.verify


def test_verify_method_verified():
    net, rng = _toy_net(5)
    base = rng.normal(size=(5, 3))
    gs = GraphStar.from_bounds(base - 0.02, base + 0.02, adjacency=net.adjacency)
    lb, ub = net.reach(gs)[0].get_ranges()
    r = net.verify(gs, lb - 0.5, ub + 0.5)
    assert r.status == "verified" and r.counterexample is None


def test_verify_method_falsified_with_witness():
    net, rng = _toy_net(6)
    base = rng.normal(size=(5, 3))
    gs = GraphStar.from_bounds(base - 0.05, base + 0.05, adjacency=net.adjacency)
    Yc = net.evaluate(base)
    r = net.verify(gs, Yc + 0.05, Yc + 100.0)
    assert r.status == "falsified"
    assert r.counterexample is not None and r.counterexample.found
    # witness genuinely violates
    assert np.any(net.evaluate(r.counterexample.counterexample) < Yc + 0.05 - 1e-9)


def test_verify_method_falsify_disabled_gives_unknown():
    net, rng = _toy_net(7)
    base = rng.normal(size=(5, 3))
    gs = GraphStar.from_bounds(base - 0.05, base + 0.05, adjacency=net.adjacency)
    Yc = net.evaluate(base)
    r = net.verify(gs, Yc + 0.05, Yc + 100.0, falsify=False)
    assert r.status == "unknown" and r.counterexample is None


def test_verify_falsified_implies_real_violation_on_checkpoint():
    path = checkpoint("sage")
    if not path.exists():
        pytest.skip(f"requires {path}")
    model = load_gnn_mat(path)
    net = GraphNeuralNetwork.from_model(model)
    X = model.X_test[0]
    gs = GraphStar.from_bounds(X - 0.02, X + 0.02, adjacency=model.adjacency)
    Yc = net.evaluate(X)
    r = net.verify(gs, Yc + 0.05, Yc + 100.0)
    assert r.status == "falsified"
    Xc = r.counterexample.counterexample
    assert np.all(Xc >= X - 0.02 - 1e-9) and np.all(Xc <= X + 0.02 + 1e-9)
    assert np.any(net.evaluate(Xc) < Yc + 0.05 - 1e-9)


def test_verify_falsify_requires_box_input():
    net, rng = _toy_net(6)
    base = rng.normal(size=(5, 3))
    box = GraphStar.from_bounds(base - 0.05, base + 0.05, adjacency=net.adjacency)
    # couple two predicates: the bounding box now contains points outside the set
    coupled = GraphStar(box.V, np.vstack([box.C, np.ones((1, box.nVar))]),
                        np.vstack([box.d, [[0.0]]]), box.predicate_lb, box.predicate_ub,
                        adjacency=net.adjacency)
    spec_lb, spec_ub = -np.full((5, 3), 1e3), np.full((5, 3), 1e3)
    with pytest.raises(ValueError, match="box input"):
        net.verify(coupled, spec_lb, spec_ub)
    assert net.verify(coupled, spec_lb, spec_ub, falsify=False).status == "verified"
    assert net.verify(box, spec_lb, spec_ub).status == "verified"
