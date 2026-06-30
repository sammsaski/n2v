"""Unit tests for LP-over-P bound tightening."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.star import Star
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.tighten import neuron_bounds


def _seq(*dims, seed=0):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers).double()


def _star_with_constraints(seed):
    """A star carrying non-trivial C/d (triangle rows from relaxed ReLUs), with
    the net + input box so we can sample exact outputs that lie in [[S]]."""
    rng = np.random.default_rng(seed)
    net = _seq(3, 6, 6, 4, seed=seed)
    layers = extract_layers(net)
    lb = rng.uniform(-1, -0.2, size=3)
    ub = rng.uniform(0.2, 1, size=3)
    S_out, _ = relaxed_reach(Star.from_bounds(lb, ub), layers)
    assert S_out.C.size > 0  # has triangle constraints
    return S_out, net, lb, ub


def _exact_outputs(net, lb, ub, n, seed=0):
    """Exact model outputs over the input box -- provably a subset of [[S_out]]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(lb, ub, size=(n, len(lb)))
    with torch.no_grad():
        return net(torch.as_tensor(X, dtype=torch.float64)).numpy()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lp_cpu_matches_get_range(seed):
    """The full (prefilter=False) path LPs every dim and matches Star.get_range."""
    S, _, _, _ = _star_with_constraints(seed)
    lb, ub = neuron_bounds(S, "lp_cpu", prefilter=False)
    for i in range(S.dim):
        gmin, gmax = S.get_range(i)
        assert lb[i, 0] == pytest.approx(gmin, abs=1e-6)
        assert ub[i, 0] == pytest.approx(gmax, abs=1e-6)


def _classify(l, u):
    return "inactive" if u <= 0 else ("active" if l >= 0 else "crossing")


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_prefilter_reach_identical_to_full(seed):
    """
    The OBBT pre-filter is result-identical to LPing every dim: same ReLU
    classification on every neuron, and identical [l,u] on every CROSSING
    (relaxed) neuron -- which is all the reach output depends on. (Box-stable
    neurons keep box values, but those never reach the output star.)
    """
    S, _, _, _ = _star_with_constraints(seed)
    lb_pf, ub_pf = neuron_bounds(S, "lp_cpu", prefilter=True)
    lb_full, ub_full = neuron_bounds(S, "lp_cpu", prefilter=False)
    for i in range(S.dim):
        cls_full = _classify(lb_full[i, 0], ub_full[i, 0])
        assert _classify(lb_pf[i, 0], ub_pf[i, 0]) == cls_full
        if cls_full == "crossing":
            assert lb_pf[i, 0] == pytest.approx(lb_full[i, 0], abs=1e-6)
            assert ub_pf[i, 0] == pytest.approx(ub_full[i, 0], abs=1e-6)


def test_prefilter_skips_lp_when_all_box_stable():
    """A star whose every dim is box-stable returns the box interval verbatim
    (no LP solved) -- the best case for the pre-filter."""
    # y = alpha in [1, 2] x [-3, -1]: dim 0 box-active, dim 1 box-inactive.
    V = np.array([[1.5, 0.5, 0.0], [-2.0, 0.0, 1.0]])
    S = Star(V, np.zeros((0, 2)), np.zeros((0, 1)),
             np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    lb_box, ub_box = S.estimate_ranges()
    lb, ub = neuron_bounds(S, "lp_cpu", prefilter=True)
    assert np.allclose(lb, lb_box) and np.allclose(ub, ub_box)
    # neither dim crosses zero -> both provably stable
    assert ub[0, 0] >= 0 >= lb_box[0, 0] - 1 and lb[0, 0] >= 0  # dim0 active
    assert ub[1, 0] <= 0                                         # dim1 inactive


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lp_bounds_tighter_than_box_and_sound(seed):
    S, net, lb, ub = _star_with_constraints(seed)
    lb_box, ub_box = S.estimate_ranges()
    lb_lp, ub_lp = neuron_bounds(S, "lp_cpu")
    # tighter-or-equal to the box interval
    assert np.all(lb_lp >= lb_box - 1e-7)
    assert np.all(ub_lp <= ub_box + 1e-7)
    # and strictly tighter somewhere (C/d actually constrains the range)
    assert np.any(lb_lp > lb_box + 1e-6) or np.any(ub_lp < ub_box - 1e-6)
    # sound: every exact output (a subset of [[S]]) is within the LP bounds
    Y = _exact_outputs(net, lb, ub, 400, seed)
    assert np.all(Y >= lb_lp.flatten() - 1e-6)
    assert np.all(Y <= ub_lp.flatten() + 1e-6)


def test_nvar_zero_returns_center():
    V = np.array([[1.5], [-2.0]])  # dim 2, nVar 0
    S = Star(V, np.zeros((0, 0)), np.zeros((0, 1)))
    lb, ub = neuron_bounds(S, "lp_cpu")
    assert lb.shape == (2, 1) and ub.shape == (2, 1)
    assert np.allclose(lb, V) and np.allclose(ub, V)


@pytest.mark.skipif(
    not __import__("n2v.utils.lpsolver_gpu", fromlist=["gpu_available"]).gpu_available(),
    reason="no CUDA",
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lp_gpu_encloses_lp_cpu(seed):
    """NS-certified GPU bounds must be OUTWARD of the exact CPU LP bounds."""
    S, _, _, _ = _star_with_constraints(seed)
    lb_cpu, ub_cpu = neuron_bounds(S, "lp_cpu")
    lb_gpu, ub_gpu = neuron_bounds(S, "lp_gpu")
    # outward enclosure: lb_gpu <= lb_cpu, ub_gpu >= ub_cpu (within tiny tol)
    assert np.all(lb_gpu <= lb_cpu + 1e-5)
    assert np.all(ub_gpu >= ub_cpu - 1e-5)


@pytest.mark.skipif(
    not __import__("n2v.utils.lpsolver_gpu", fromlist=["gpu_available"]).gpu_available(),
    reason="no CUDA",
)
@pytest.mark.skipif(
    not __import__("n2v.utils.lpsolver_gpu", fromlist=["gpu_available"]).gpu_available(),
    reason="no CUDA",
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prefilter_gpu_stays_sound(seed):
    """
    The prefiltered lp_gpu path is a sound over-approximation: its bounds enclose
    the EXACT range (so a neuron is never falsely stabilised). lp_gpu is not
    bit-identical to full LP (batched PDHG is batch-composition-sensitive), but
    soundness -- box-clipped NS still outward of the true range -- always holds.
    """
    S, _, _, _ = _star_with_constraints(seed)
    lb_pf, ub_pf = neuron_bounds(S, "lp_gpu", prefilter=True)
    lb_exact, ub_exact = neuron_bounds(S, "lp_cpu", prefilter=False)   # exact HiGHS range
    assert np.all(lb_pf <= lb_exact + 1e-5), "GPU prefilter lb not outward of exact"
    assert np.all(ub_pf >= ub_exact - 1e-5), "GPU prefilter ub not outward of exact"
