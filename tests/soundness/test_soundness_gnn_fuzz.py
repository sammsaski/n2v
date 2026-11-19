"""
Broad randomized soundness battery for the GNN reach + verify pipeline.

Complements the per-layer soundness suites with:
  * property-based fuzzing over random topology / weights / eps / depth,
  * subgraph-reach soundness (target output contained, at scale),
  * higher-eps regimes (more crossing neurons => more relaxation),
  * degenerate graphs (eps=0, self-loops only, parallel edges, isolated targets),
  * verifier soundness: NO false "verified" — a reachable point that violates a
    spec must yield "unknown", never "verified".

Everything is sound if the reach box contains every sampled forward output and
the verifier never certifies a box that a reachable point escapes.
"""

import numpy as np
import pytest

from n2v.sets import GraphStar
from n2v.nn import GraphNeuralNetwork
from n2v.utils import (
    load_gnn_mat, GCNLayerSpec, SAGELayerSpec, GINELayerSpec, verify_node_bounds,
)
from tests.fixtures.gnn import checkpoint


TOL = 1e-6


# ======================================================== random model factories

def _rand_adjacency(rng, n, p=0.4, normalized=False):
    A = (rng.uniform(size=(n, n)) < p).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)
    if normalized:
        d = A.sum(1, keepdims=True)
        A = A / np.sqrt(d) / np.sqrt(d.T)
    return A


def _rand_edges(rng, n, m):
    src = np.concatenate([rng.integers(0, n, m), np.arange(n)])   # incl. self-loops
    dst = np.concatenate([rng.integers(0, n, m), np.arange(n)])
    return np.vstack([src, dst])


def _gcn_net(rng, n, feats):
    A = _rand_adjacency(rng, n, normalized=True)
    layers = [GCNLayerSpec(rng.normal(size=(feats[i], feats[i + 1])),
                           rng.normal(size=(feats[i + 1],))) for i in range(len(feats) - 1)]
    return GraphNeuralNetwork(layers, adjacency=A, has_relu=True)


def _sage_net(rng, n, feats):
    A = _rand_adjacency(rng, n)
    layers = [SAGELayerSpec(rng.normal(size=(feats[i], feats[i + 1])),
                            rng.normal(size=(feats[i], feats[i + 1])),
                            rng.normal(size=(feats[i + 1],))) for i in range(len(feats) - 1)]
    return GraphNeuralNetwork(layers, adjacency=A, has_relu=True)


def _gine_net(rng, n, feats, variant, e_in=2):
    edge_index = _rand_edges(rng, n, 2 * n)
    m = edge_index.shape[1]
    E = rng.normal(size=(m, e_in))
    layers = []
    for i in range(len(feats) - 1):
        hid = feats[i + 1] + 1
        layers.append(GINELayerSpec(
            rng.normal(size=(feats[i], hid)), rng.normal(size=(hid,)),
            rng.normal(size=(hid, feats[i + 1])), rng.normal(size=(feats[i + 1],)),
            rng.normal(size=(e_in, feats[i])), rng.normal(size=(feats[i],)),
            0.2 if variant == "pyg" else 0.0))
    return GraphNeuralNetwork(layers, edge_index=edge_index, E=E, gine_variant=variant)


def _is_gine(net):
    return net.layer_kind == "GINELayerSpec"


def _make_input(net, rng, n, f_in, eps):
    base = rng.normal(size=(n, f_in))
    adj = None if _is_gine(net) else net.adjacency
    return base, GraphStar.from_bounds(base - eps, base + eps, adjacency=adj)


def _assert_reach_contains_samples(net, base, gs, eps, rng, k=80):
    out = net.reach(gs)[0]
    lb, ub = out.get_ranges()
    for _ in range(k):
        Xi = base + rng.uniform(-eps, eps, size=base.shape)
        Yi = net.evaluate(Xi)
        assert np.all(Yi >= lb - TOL) and np.all(Yi <= ub + TOL)
    return out, lb, ub


# ============================================================ property-based fuzz

@pytest.mark.parametrize("family", ["gcn", "sage", "gine_hugine", "gine_pyg"])
def test_fuzz_reach_soundness(family):
    rng = np.random.default_rng(hash(family) % 2**32)
    for _ in range(8):                                   # 8 random models per family
        n = int(rng.integers(4, 9))
        depth = int(rng.integers(1, 4))
        feats = [int(rng.integers(2, 5)) for _ in range(depth + 1)]
        eps = float(rng.choice([1e-3, 5e-3, 1e-2]))
        if family == "gcn":
            net = _gcn_net(rng, n, feats)
        elif family == "sage":
            net = _sage_net(rng, n, feats)
        else:
            net = _gine_net(rng, n, feats, "hugine" if "hugine" in family else "pyg")
        base, gs = _make_input(net, rng, n, feats[0], eps)
        _assert_reach_contains_samples(net, base, gs, eps, rng, k=60)


# ============================================================ subgraph soundness

@pytest.mark.parametrize("family", ["gcn", "sage", "gine_hugine"])
def test_fuzz_subgraph_reach_soundness(family):
    """Per-target subgraph reach box must contain the target's forward output."""
    rng = np.random.default_rng(99 + len(family))
    for _ in range(6):
        n = int(rng.integers(6, 12))
        feats = [int(rng.integers(2, 4)) for _ in range(3)]
        eps = 5e-3
        net = (_gcn_net(rng, n, feats) if family == "gcn"
               else _sage_net(rng, n, feats) if family == "sage"
               else _gine_net(rng, n, feats, "hugine"))
        base, gs = _make_input(net, rng, n, feats[0], eps)
        targets = sorted(rng.choice(n, size=min(3, n), replace=False).tolist())
        results = net.reach_subgraph(gs, targets)
        for r in results:
            tlb, tub = r.target_ranges()
            for _ in range(40):
                Xi = base + rng.uniform(-eps, eps, size=base.shape)
                Yi = net.evaluate(Xi)[r.target_node]
                assert np.all(Yi >= tlb - TOL) and np.all(Yi <= tub + TOL)


# ============================================================ higher-eps regimes

@pytest.mark.parametrize("case,eps", [("gcn", 0.05), ("gcn", 0.1),
                                      ("sage", 0.05), ("sage", 0.1)])
def test_high_eps_real_checkpoint_soundness(case, eps):
    path = checkpoint(case)
    if not path.exists():
        pytest.skip(f"requires {path}")
    model = load_gnn_mat(path)
    net = GraphNeuralNetwork.from_model(model)
    rng = np.random.default_rng(int(eps * 1000))
    X = model.X_test[0]
    gs = GraphStar.from_bounds(X - eps, X + eps, adjacency=model.adjacency)
    out = net.reach(gs)[0]
    lb, ub = out.get_ranges()
    for _ in range(120):
        Y = net.evaluate(X + rng.uniform(-eps, eps, size=X.shape))
        assert np.all(Y >= lb - TOL) and np.all(Y <= ub + TOL)


# ============================================================ degenerate graphs

def test_eps_zero_point_set():
    rng = np.random.default_rng(0)
    net = _sage_net(rng, 4, [2, 3, 2])
    X = rng.normal(size=(4, 2))
    gs = GraphStar.from_bounds(X, X, adjacency=net.adjacency)   # zero width
    out = net.reach(gs)[0]
    lb, ub = out.get_ranges()
    assert out.nVar == 0
    assert np.allclose(lb, ub, atol=1e-9)
    assert np.allclose(out.center(), net.evaluate(X), atol=1e-9)


def test_gine_self_loops_only_sound():
    rng = np.random.default_rng(1)
    n = 4
    edge_index = np.vstack([np.arange(n), np.arange(n)])        # only self-loops
    E = rng.normal(size=(n, 2))
    layers = [GINELayerSpec(rng.normal(size=(2, 3)), rng.normal(size=(3,)),
                            rng.normal(size=(3, 2)), rng.normal(size=(2,)),
                            rng.normal(size=(2, 2)), rng.normal(size=(2,)), 0.0)]
    net = GraphNeuralNetwork(layers, edge_index=edge_index, E=E, gine_variant="hugine")
    base = rng.normal(size=(n, 2))
    gs = GraphStar.from_bounds(base - 0.05, base + 0.05)
    out = net.reach(gs)[0]
    lb, ub = out.get_ranges()
    for _ in range(150):
        Y = net.evaluate(base + rng.uniform(-0.05, 0.05, size=base.shape))
        assert np.all(Y >= lb - TOL) and np.all(Y <= ub + TOL)


def test_gine_parallel_duplicate_edges_sound():
    rng = np.random.default_rng(2)
    edge_index = np.array([[0, 0, 1, 0, 1], [1, 1, 0, 0, 1]])   # duplicated 0->1
    E = rng.normal(size=(edge_index.shape[1], 2))
    layers = [GINELayerSpec(rng.normal(size=(2, 3)), rng.normal(size=(3,)),
                            rng.normal(size=(3, 2)), rng.normal(size=(2,)),
                            rng.normal(size=(2, 2)), rng.normal(size=(2,)), 0.0)]
    net = GraphNeuralNetwork(layers, edge_index=edge_index, E=E, gine_variant="hugine")
    base = rng.normal(size=(2, 2))
    gs = GraphStar.from_bounds(base - 0.03, base + 0.03)
    out = net.reach(gs)[0]
    lb, ub = out.get_ranges()
    for _ in range(150):
        Y = net.evaluate(base + rng.uniform(-0.03, 0.03, size=base.shape))
        assert np.all(Y >= lb - TOL) and np.all(Y <= ub + TOL)


# ===================================================== verifier soundness (key)

@pytest.mark.parametrize("family", ["gcn", "sage", "gine_hugine"])
def test_verifier_no_false_verified(family):
    """A reachable point that violates the spec must NOT be certified 'verified'.

    The network's own center output is reachable; a spec box that excludes it
    must come back 'unknown' (sound), never 'verified' (which would be a false
    safety certificate — the worst verifier bug).
    """
    rng = np.random.default_rng(7 + len(family))
    n, feats, eps = 6, [3, 4, 3], 1e-2
    net = (_gcn_net(rng, n, feats) if family == "gcn"
           else _sage_net(rng, n, feats) if family == "sage"
           else _gine_net(rng, n, feats, "hugine"))
    base, gs = _make_input(net, rng, n, feats[0], eps)
    out = net.reach(gs)[0]
    Yc = net.evaluate(base)                       # reachable (center of input box)
    assert out.contains(Yc)

    # spec lower bound just ABOVE the reachable center -> center violates it
    spec_lb = Yc + 0.1
    spec_ub = Yc + 100.0
    verdict = verify_node_bounds([out], spec_lb, spec_ub)
    assert verdict != "verified", (
        f"{family}: verifier certified a box that a reachable point escapes")

    # sanity: a box that contains the whole reach set IS verified
    lb, ub = out.get_ranges()
    assert verify_node_bounds([out], lb - 0.5, ub + 0.5) == "verified"


def test_verifier_no_false_verified_on_real_checkpoint():
    path = checkpoint("gcn")
    if not path.exists():
        pytest.skip(f"requires {path}")
    model = load_gnn_mat(path)
    net = GraphNeuralNetwork.from_model(model)
    X = model.X_test[0]
    gs = GraphStar.from_bounds(X - 1e-2, X + 1e-2, adjacency=model.adjacency)
    out = net.reach(gs)[0]
    Yc = net.evaluate(X)
    assert verify_node_bounds([out], Yc + 0.05, Yc + 50.0) != "verified"
    lb, ub = out.get_ranges()
    assert verify_node_bounds([out], lb - 0.5, ub + 0.5) == "verified"
