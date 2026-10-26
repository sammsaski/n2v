"""Unit tests for LP-over-P bound tightening."""

import numpy as np
import pytest
import torch

from n2v.sets.star import Star
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.tighten import contract_input_boxes, neuron_bounds
from tests.unit.refine.conftest import seq_net as _seq


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
    """lp_cpu LPs every dim and matches Star.get_range."""
    S, _, _, _ = _star_with_constraints(seed)
    lb, ub = neuron_bounds(S, "lp_cpu")
    for i in range(S.dim):
        gmin, gmax = S.get_range(i)
        assert lb[i, 0] == pytest.approx(gmin, abs=1e-6)
        assert ub[i, 0] == pytest.approx(gmax, abs=1e-6)


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


def test_lp_cpu_ranges_are_padded_outward():
    from n2v.refine.tighten import _lp_ranges
    rng = np.random.default_rng(2)
    V = np.hstack([rng.standard_normal((4, 1)), rng.standard_normal((4, 3))])
    S = Star(V, np.zeros((0, 3)), np.zeros((0, 1)),
             -np.ones((3, 1)), np.ones((3, 1)))
    # exact optimum over the [-1, 1]^3 box (no polytope rows): c -/+ sum|V|
    lb_x = V[:, 0] - np.abs(V[:, 1:]).sum(axis=1)
    ub_x = V[:, 0] + np.abs(V[:, 1:]).sum(axis=1)
    pad_lo = 1e-7 * (1.0 + np.abs(lb_x))
    pad_hi = 1e-7 * (1.0 + np.abs(ub_x))
    lo, hi = _lp_ranges(S, "lp_cpu", np.arange(4))
    assert np.all(lo <= lb_x - 0.5 * pad_lo)
    assert np.all(hi >= ub_x + 0.5 * pad_hi)


def test_contract_input_boxes_rejects_inf_boxes():
    """Certified contraction is undefined on infinite predicate intervals
    (0 * inf = NaN); it must fail loudly instead of returning poisoned boxes."""
    # alpha_0 in [-1, 1] carried only by C/d rows (box +/-inf on that dim)
    V = np.hstack([np.zeros((2, 1)), np.eye(2)])
    C = np.array([[1.0, 0.0], [-1.0, 0.0]])
    d = np.array([[1.0], [1.0]])
    S = Star(V, C, d,
             np.array([[-np.inf], [-1.0]]), np.array([[np.inf], [1.0]]))
    with pytest.raises(ValueError, match="finite predicate boxes"):
        contract_input_boxes(
            [(S.C, S.d)], [(S.predicate_lb, S.predicate_ub)], n_dims=2,
            device="cpu")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dual_gpu_bounds_enclose_lp_cpu(seed, engine_device):
    """NS-certified GPU bounds must be OUTWARD of the exact CPU LP bounds."""
    S, _, _, _ = _star_with_constraints(seed)
    lb_cpu, ub_cpu = neuron_bounds(S, "lp_cpu")
    lb_gpu, ub_gpu = neuron_bounds(S, "dual_gpu")
    # outward enclosure: lb_gpu <= lb_cpu, ub_gpu >= ub_cpu (within tiny tol)
    assert np.all(lb_gpu <= lb_cpu + 1e-5)
    assert np.all(ub_gpu >= ub_cpu - 1e-5)
