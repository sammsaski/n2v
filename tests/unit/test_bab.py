"""Branch-and-bound engine: correctness + soundness.

Uses a small ReLU net whose triangle (approx-star) relaxation is loose over the
full input box, so single-shot reach is UNKNOWN for a threshold between the true
maximum and the relaxed bound — but input splitting tightens the relaxation and
BaB verifies it. Also checks falsification and that the budget yields a sound
UNKNOWN (never an over-claim).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Star
from n2v.sets.halfspace import HalfSpace
from n2v.nn.neural_network import NeuralNetwork
from n2v.utils.verify_specification import verify_specification
from n2v.nn.bab import verify_bab_model, verify_bab_relu

LB = np.array([-1.0, -1.0])
UB = np.array([1.0, 1.0])


def _net(seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    m.eval()
    return m


def _true_max(m):
    g = np.linspace(-1, 1, 201)
    xs = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2).astype(np.float32)
    with torch.no_grad():
        return float(m(torch.from_numpy(xs)).numpy().max())


def _star_max(m):
    out = NeuralNetwork(m).reach(Star.from_bounds(LB, UB), method="approx")
    return max(float(np.asarray(s.get_ranges()[1]).reshape(-1)[0]) for s in out)


def _spec_leq(thresh):  # unsafe {Y >= thresh} == {-Y <= -thresh}
    return HalfSpace(np.array([[-1.0]]), np.array([-thresh]))


def _gap_net():
    """Pick a seed whose approx-star bound is meaningfully looser than truth."""
    for seed in range(20):
        m = _net(seed)
        tmax, smax = _true_max(m), _star_max(m)
        if smax - tmax > 0.05:
            return m, tmax, smax
    pytest.skip("no relaxation gap found")


def test_bab_verifies_what_single_shot_cannot():
    m, tmax, smax = _gap_net()
    thresh = tmax + 0.5 * (smax - tmax)        # true_max < thresh < star_max
    spec = _spec_leq(thresh)
    out = NeuralNetwork(m).reach(Star.from_bounds(LB, UB), method="approx")
    assert verify_specification(out, spec).verdict == "UNKNOWN"   # bounding alone fails
    res = verify_bab_model(m, LB, UB, spec, branch="widest",
                           max_nodes=2000, falsify_method="random")
    assert res.verdict == "VERIFIED", res
    assert res.splits >= 1


def test_bab_verified_is_sound():
    m, tmax, smax = _gap_net()
    thresh = tmax + 0.5 * (smax - tmax)
    res = verify_bab_model(m, LB, UB, _spec_leq(thresh), branch="widest",
                           max_nodes=2000, falsify_method="random")
    assert res.verdict == "VERIFIED"
    rng = np.random.default_rng(1)
    xs = rng.uniform(-1, 1, size=(20000, 2)).astype(np.float32)
    with torch.no_grad():
        ys = m(torch.from_numpy(xs)).numpy().reshape(-1)
    assert np.all(ys <= thresh + 1e-5)         # no true input violates the proven bound


def test_bab_falsifies_false_property():
    m, tmax, smax = _gap_net()
    thresh = tmax - 0.2 * (smax - tmax)        # below true max -> falsifiable
    res = verify_bab_model(m, LB, UB, _spec_leq(thresh), branch="widest",
                           max_nodes=2000, falsify_method="random+pgd")
    assert res.verdict == "FALSIFIED", res
    x = np.asarray(res.counterexample).reshape(-1).astype(np.float32)
    with torch.no_grad():
        y = float(m(torch.from_numpy(x).unsqueeze(0)).reshape(-1)[0])
    assert y >= thresh - 1e-4                  # counterexample really violates


def test_exact_relu_split_is_complete():
    # n2v's method='exact' performs complete ReLU (neuron) splitting via
    # relu_star_exact; it verifies the same relaxation gap that 'approx' (a
    # single triangle-relaxed star) cannot. This is the per-neuron split
    # completeness already in the toolbox (BaB is its budget-controlled cousin).
    m, tmax, smax = _gap_net()
    spec = _spec_leq(tmax + 0.5 * (smax - tmax))
    approx = NeuralNetwork(m).reach(Star.from_bounds(LB, UB), method="approx")
    assert verify_specification(approx, spec).verdict == "UNKNOWN"
    exact = NeuralNetwork(m).reach(Star.from_bounds(LB, UB), method="exact")
    assert verify_specification(exact, spec).verdict == "UNSAT"
    assert len(exact) > 1                       # it actually split


def test_bab_relu_neuron_split_verifies():
    # Neuron-split BaB verifies the relaxation gap by forcing unstable ReLUs,
    # pruning far below the 2^16 worst case.
    m, tmax, smax = _gap_net()
    spec = _spec_leq(tmax + 0.5 * (smax - tmax))
    res = verify_bab_relu(m, LB, UB, spec, falsify_method="random", max_nodes=3000)
    assert res.verdict == "VERIFIED", res
    assert res.nodes < 3000


def test_bab_relu_neuron_split_sound():
    m, tmax, smax = _gap_net()
    res = verify_bab_relu(m, LB, UB, _spec_leq(tmax + 0.5 * (smax - tmax)),
                          falsify_method="random", max_nodes=3000)
    assert res.verdict == "VERIFIED"
    rng = np.random.default_rng(2)
    xs = rng.uniform(-1, 1, size=(20000, 2)).astype(np.float32)
    with torch.no_grad():
        ys = m(torch.from_numpy(xs)).numpy().reshape(-1)
    assert np.all(ys <= tmax + 0.5 * (smax - tmax) + 1e-5)


def test_bab_relu_neuron_split_falsifies():
    m, tmax, smax = _gap_net()
    res = verify_bab_relu(m, LB, UB, _spec_leq(tmax - 0.2 * (smax - tmax)),
                          falsify_method="random+pgd", max_nodes=3000)
    assert res.verdict == "FALSIFIED", res


def test_bab_budget_is_sound_unknown():
    m, tmax, smax = _gap_net()
    res = verify_bab_model(m, LB, UB, _spec_leq(tmax + 0.5 * (smax - tmax)),
                           branch="widest", max_nodes=1, falsify_method="random")
    assert res.verdict in ("UNKNOWN", "VERIFIED")
    if res.verdict == "UNKNOWN":
        assert "budget" in res.reason or "timeout" in res.reason
