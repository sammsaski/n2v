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
    """Pick a seed whose approx-star bound is meaningfully looser than truth.

    Every seed tried must satisfy ``smax >= tmax``: ``tmax`` is a float32 grid
    maximum, a lower bound on the true maximum, so an approx-star bound below
    it (beyond float32 forward rounding) is an unsound reach. Finding no gap
    is a failure, not a skip: the tests below exist for exactly that regime.
    """
    for seed in range(20):
        m = _net(seed)
        tmax, smax = _true_max(m), _star_max(m)
        assert smax >= tmax - 1e-5, (
            f"approx-star bound {smax} is below the sampled true maximum "
            f"{tmax} (seed {seed}): the relaxation is unsound")
        if smax - tmax > 0.05:
            return m, tmax, smax
    pytest.fail("no seed in range(20) gives a relaxation gap above 0.05; the "
                "BaB tests need one (pin a seed that does)")


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
    # no true input violates the proven bound; no slack: thresh lies at least
    # 0.025 above the sampled maximum, far beyond float32 rounding
    assert np.all(ys <= thresh)


def test_bab_falsifies_false_property():
    # The witness must lie in the box and violate the property at zero
    # tolerance on the batch-1 float32 forward, which falsify() re-checks
    # before it returns (a 1e-4 slack here accepted a safe point).
    m, tmax, smax = _gap_net()
    thresh = tmax - 0.2 * (smax - tmax)        # below true max -> falsifiable
    res = verify_bab_model(m, LB, UB, _spec_leq(thresh), branch="widest",
                           max_nodes=2000, falsify_method="random+pgd")
    assert res.verdict == "FALSIFIED", res
    x64 = np.asarray(res.counterexample, dtype=np.float64).reshape(-1)
    assert np.all(LB <= x64) and np.all(x64 <= UB), x64
    x = x64.astype(np.float32)
    with torch.no_grad():
        y = float(m(torch.from_numpy(x).unsqueeze(0)).reshape(-1)[0])
    assert y >= thresh                         # counterexample really violates


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
    assert np.all(ys <= tmax + 0.5 * (smax - tmax))     # no slack, as above


def test_bab_relu_neuron_split_falsifies():
    # The witness is the falsifier's input: inside the box, and the batch-1
    # float32 forward reaches the threshold at zero tolerance.
    m, tmax, smax = _gap_net()
    thresh = tmax - 0.2 * (smax - tmax)
    res = verify_bab_relu(m, LB, UB, _spec_leq(thresh),
                          falsify_method="random+pgd", max_nodes=3000)
    assert res.verdict == "FALSIFIED", res
    x64 = np.asarray(res.counterexample, dtype=np.float64).reshape(-1)
    assert np.all(LB <= x64) and np.all(x64 <= UB), x64
    with torch.no_grad():
        y = float(m(torch.from_numpy(x64.astype(np.float32)).unsqueeze(0))
                  .reshape(-1)[0])
    assert y >= thresh


def test_bab_budget_is_sound_unknown():
    # One reach of the whole box does not prove this threshold (see
    # test_bab_verifies_what_single_shot_cannot), so a search allowed one
    # node must end UNKNOWN after that node; VERIFIED here would be the
    # budget answered as a proof.
    m, tmax, smax = _gap_net()
    res = verify_bab_model(m, LB, UB, _spec_leq(tmax + 0.5 * (smax - tmax)),
                           branch="widest", max_nodes=1, falsify_method="random")
    assert res.verdict == "UNKNOWN", res
    assert res.nodes == 1
    assert "budget" in res.reason or "timeout" in res.reason
