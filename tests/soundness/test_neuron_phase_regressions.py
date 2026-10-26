"""Regressions for how ReLU / LeakyReLU / Sign reach decide a neuron's phase.

  * An LP optimum decides a phase only after it is padded outward (the HiGHS
    tolerance). Measured HiGHS error: a max reported 5e-10 below the true max
    on thin wedges. So a neuron whose true max is +2e-10 but whose LP reports
    -3e-10 is never zeroed or scaled as inactive (ReLU relax modes, LeakyReLU,
    Sign), and a raw bound within the pad of 0 keeps ONE star with a sliver
    variable instead of splitting (exact ReLU / LeakyReLU). A bound of exactly
    0 therefore no longer doubles the exact star count per neuron.
  * An LP that returns no optimum means 'undecided', never 'empty': a star is
    dropped only when certified empty, and exact ReLU no longer crashes on it.
  * The phase LPs run on HiGHS whatever solver the caller asked for; SCS/OSQP
    report 'optimal' maxima ~1e-5 below the true max.
  * The triangle relaxations keep the vertex (u, u) (no ``+ 1e-10`` in the
    slope denominator), and the LeakyReLU one holds for any slope (it was
    empty for a slope above 1, where LeakyReLU is concave).
  * Sign keeps sign(0) = 0, has no tolerance pointing the unsound way, and
    keeps the ImageStar type when exact mode splits.
  * The parallel exact ReLU re-raises a worker failure instead of dropping
    that star's output.
  * ReLU on an ImageZono returns an ImageZono (HWC order for Flatten).
  * Hexatope/Octatope ReLU: a dead row or a box bound of 0 is not split.
  * Exact reach widens the pre-pass (IBP) bounds before they decide a phase,
    and splits a neuron whose whole range lies within the pads (the sliver
    would be decoupled from x and many times wider than the range), unless
    the side the LP points to is confirmed by certifying the other empty.
  * LeakyReLU on a Box holds for any slope (it left out f(0) = 0 for a
    negative slope).
  * Precomputed bounds that cannot describe the layer are not applied: an
    array of another size raises (the approximate paths broadcast a size-1
    array to every neuron), bounds disjoint from the star's own enclosure
    are dropped (they zeroed a neuron truly in [1, 2]), and a NaN bound is
    ignored (it left a crossing neuron unrelaxed).
"""

from fractions import Fraction as F

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn import NeuralNetwork
from n2v.nn.layer_ops.leakyrelu_reach import (
    leakyrelu_box,
    leakyrelu_star_approx,
    leakyrelu_star_exact,
)
from n2v.nn.layer_ops.relu_reach import (
    _step_relu,
    relu_hexatope,
    relu_octatope,
    relu_star_approx,
    relu_star_exact,
    relu_zono_approx,
)
from n2v.nn.layer_ops.sign_reach import sign_star
from n2v.sets import Box, Hexatope, Octatope, Star
from n2v.sets.halfspace import HalfSpace
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils import lpsolver
from n2v.utils.lp_solver_enum import LPSolver, resolve
from n2v.utils.verify_specification import verify_specification


LP_ERR = 5e-10   # HiGHS error measured at the zero boundary of thin wedges


def _x_star(lo, hi):
    """1-D star ``x = c + r alpha``, ``alpha in [-1, 1]``, over ``[lo, hi]``."""
    return Star.from_bounds(np.array([[lo]]), np.array([[hi]]))


def _exact_max(S, i=0):
    """Exact rational max of row ``i`` of a box-only star."""
    return F(float(S.V[i, 0])) + sum(abs(F(float(g))) for g in S.V[i, 1:])


@pytest.fixture
def lp_error(monkeypatch):
    """Every Star LP errs by LP_ERR in the unsafe direction: maxima reported
    low, minima reported high."""
    orig_range, orig_max, orig_min = Star.get_range, Star.get_max, Star.get_min

    def get_range(self, index, lp_solver=LPSolver.DEFAULT):
        lo, hi = orig_range(self, index, lp_solver)
        if lo is None or hi is None:
            return lo, hi
        return lo + LP_ERR, hi - LP_ERR

    def get_max(self, index, lp_solver=LPSolver.DEFAULT):
        v = orig_max(self, index, lp_solver)
        return None if v is None else v - LP_ERR

    def get_min(self, index, lp_solver=LPSolver.DEFAULT):
        v = orig_min(self, index, lp_solver)
        return None if v is None else v + LP_ERR

    monkeypatch.setattr(Star, "get_range", get_range)
    monkeypatch.setattr(Star, "get_max", get_max)
    monkeypatch.setattr(Star, "get_min", get_min)


@pytest.fixture
def lp_fails(monkeypatch):
    """Every Star range LP returns no optimum (a solver failure or a false
    'infeasible'); the certified emptiness check itself still works."""
    monkeypatch.setattr(Star, "get_range", lambda self, i, s=None: (None, None))
    monkeypatch.setattr(Star, "get_max", lambda self, i, s=None: None)
    monkeypatch.setattr(Star, "get_min", lambda self, i, s=None: None)


# ---------------------------------------------------------------------------
# Exact ReLU: LP failure / empty stars, parallel failures
# ---------------------------------------------------------------------------

def _empty_star():
    V = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])   # x = alpha
    C = np.array([[-1.0, -1.0], [1.0, 1.0]])           # a0+a1 >= 1.5, <= 1
    d = np.array([[-1.5], [1.0]])
    return Star(V, C, d, -np.ones((2, 1)), np.ones((2, 1)))


def test_exact_relu_drops_a_certified_empty_star():
    S = _empty_star()
    assert S.get_range(0) == (None, None)
    assert relu_star_exact([S], parallel=False) == []
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU()).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.eye(2, dtype=torch.float64))
        net[0].bias.zero_()
    assert NeuralNetwork(net).reach(S, method="exact") == []


def test_exact_relu_keeps_a_star_whose_lp_fails(lp_fails):
    S = _x_star(-1.0, 1.0)
    out = relu_star_exact([S], parallel=False)
    assert len(out) == 2   # undecided -> split, not dropped
    lbs = [o.estimate_ranges()[0][0, 0] for o in out]
    ubs = [o.estimate_ranges()[1][0, 0] for o in out]
    assert min(lbs) <= 0.0 and max(ubs) >= 1.0


def test_exact_relu_float32_net_does_not_crash_and_covers_outputs():
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2), nn.ReLU())
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1., 1.], [1., 0.], [0., 1.]]))
        net[0].bias.copy_(torch.tensor([-1., 1., 1.]))
        net[2].weight.copy_(torch.tensor([[0., 1e-3, 1e-3], [1., 0., 0.]]))
        net[2].bias.copy_(torch.tensor([-3e-3, -0.25]))
    S = Star.from_bounds(np.zeros((2, 1)), np.ones((2, 1)))
    R = NeuralNetwork(net).reach(S, method="exact")
    assert len(R) >= 1
    rng = np.random.default_rng(0)
    X = np.vstack([rng.uniform(0, 1, (30, 2)), [[0, 0], [1, 1], [1, 0], [0, 1]]])
    with torch.no_grad():   # the same float32 weights, evaluated in float64
        Y = net.double()(torch.tensor(X)).numpy()
    for y in Y:
        assert any(r.contains(y) for r in R), f"output {y} not in the reach set"


def test_parallel_exact_relu_reraises_a_worker_failure():
    S = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    bad = (np.zeros(3), np.zeros(3))   # wrong shape: every worker raises
    with pytest.raises(ValueError):
        relu_star_exact([S, S], parallel=False, precomputed_bounds=bad)
    with pytest.raises(ValueError):
        relu_star_exact([S, S], parallel=True, n_workers=2,
                        precomputed_bounds=bad)


# ---------------------------------------------------------------------------
# Exact ReLU: bounds of exactly 0 do not split, pre-pass bounds are widened
# ---------------------------------------------------------------------------

def test_step_relu_keeps_one_star_on_an_lp_bound_of_exactly_zero():
    # x = alpha on [-1, 1] cut to alpha >= 0: the interval bound crosses 0,
    # the LP min is exactly 0.
    S = Star(np.array([[0.0, 1.0]]), np.array([[-1.0]]), np.array([[0.0]]),
             -np.ones((1, 1)), np.ones((1, 1)))
    out = _step_relu(S, 0)
    assert len(out) == 1
    lo, hi = out[0].estimate_ranges()
    assert lo[0, 0] <= 0.0 and hi[0, 0] >= 1.0


def test_second_relu_does_not_split_again():
    torch.manual_seed(0)
    lin = nn.Linear(2, 6).double()
    S = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    one = NeuralNetwork(nn.Sequential(lin, nn.ReLU())).reach(S, method="exact")
    two = NeuralNetwork(nn.Sequential(lin, nn.ReLU(), nn.ReLU())).reach(
        S, method="exact")
    assert len(two) == len(one)


def test_zero_bias_net_star_count_and_coverage():
    # Every child star contains the origin, so every sign-stable deeper
    # neuron has an LP bound of exactly 0 (241 stars when each was split).
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(2, 4, bias=False), nn.ReLU(),
                        nn.Linear(4, 4, bias=False), nn.ReLU(),
                        nn.Linear(4, 1, bias=False)).double()
    S = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    R = NeuralNetwork(net).reach(S, method="exact")
    assert len(R) <= 24
    rng = np.random.default_rng(0)
    X = np.vstack([rng.uniform(-1, 1, (40, 2)), [[0, 0], [1, 1], [-1, 1]]])
    with torch.no_grad():
        Y = net(torch.tensor(X)).numpy()
    for y in Y:
        assert any(r.contains(y) for r in R), f"output {y} not in the reach set"


def test_exact_relu_widens_round_to_nearest_prepass_bounds():
    from n2v.utils.bounds_precomputation import _ibp_linear
    lb, ub = np.array([0.0, 0.0]), np.array([0.1, 0.7])
    W, b = np.array([[1.0, 1.0]]), np.array([-0.7999999999999999])
    true_max = F(0.1) + F(0.7) + F(float(b[0]))
    assert true_max > 0
    S = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1)).affine_map(
        W, b.reshape(-1, 1))
    plb, pub = _ibp_linear(lb, ub, W, b)
    out = relu_star_exact([S], parallel=False,
                          precomputed_bounds=(plb.reshape(-1, 1),
                                              pub.reshape(-1, 1)))
    assert max(F(float(o.estimate_ranges()[1][0, 0])) for o in out) >= true_max
    out = leakyrelu_star_exact([S], gamma=0.01,
                               precomputed_bounds=(plb.reshape(-1, 1),
                                                   pub.reshape(-1, 1)))
    assert max(F(float(o.estimate_ranges()[1][0, 0])) for o in out) >= true_max


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_thin_neuron_on_one_side_of_zero_keeps_its_side(sign):
    # x = sign * 4.5e-8 * (a0 + a1) with a0 + a1 >= 1: the interval bound
    # crosses 0, the LP range sign * [4.5e-8, 9e-8] lies within the pads.
    # Exact either way (no sliver variable); where the other side can be
    # certified empty (highspy), one star.
    V = np.array([[0.0, sign * 4.5e-8, sign * 4.5e-8]])
    S = Star(V, np.array([[-1.0, -1.0]]), np.array([[-1.0]]),
             -np.ones((2, 1)), np.ones((2, 1)))
    for out, gamma in ((_step_relu(S, 0), 0.0),
                       (leakyrelu_star_exact([S], gamma=0.01), 0.01)):
        assert all(o.nVar == S.nVar for o in out)
        if lpsolver._HAS_HIGHSPY:
            assert len(out) == 1
            assert np.array_equal(out[0].V, V if sign > 0 else gamma * V)


def test_thin_neuron_keeps_both_sides_when_the_lp_sign_is_wrong(lp_error):
    # true range [-4e-10, 9e-8], the LP reports min +1e-10: x <= 0 is not
    # empty, so the neuron must still be split (not passed through as y = x)
    S = _x_star(-4e-10, 9e-8)
    x_lo = float(S.V[0, 0] - S.V[0, 1])            # x at alpha = -1
    for out, gamma in ((relu_star_exact([S], parallel=False), 0.0),
                       (leakyrelu_star_exact([S], gamma=0.01), 0.01)):
        f_lo = gamma * x_lo
        hit = False
        for o in out:
            plb, pub = o.predicate_lb.copy(), o.predicate_ub.copy()
            plb[0, 0] = pub[0, 0] = -1.0
            lo, hi = Star(o.V, o.C, o.d, plb, pub).estimate_ranges()
            hit |= lo[0, 0] - 1e-20 <= f_lo <= hi[0, 0] + 1e-20
        assert hit, f"f(x) = {f_lo} at x = {x_lo} is not in the reach set"


@pytest.mark.parametrize("act, true_max", [(nn.ReLU(), 2.25),
                                           (nn.LeakyReLU(0.01), 2.245)])
@pytest.mark.parametrize("scale", [1e-8, 3e-8])
def test_exact_reach_splits_a_neuron_whose_whole_range_is_within_the_pad(
        act, true_max, scale):
    # Pre-activations of size ~scale, amplified by 1/scale: each hidden
    # neuron's LP range crosses 0 but lies within the pads on both sides.
    # A sliver there is decoupled from x and ~8x wider than the range (reach
    # max 23.75 for scale 1e-8); the split keeps the reach exact.
    net = nn.Sequential(nn.Linear(2, 2), act, nn.Linear(2, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 1.0], [1.0, -1.0]]) * scale)
        net[0].bias.copy_(torch.tensor([-0.5, 0.25]) * scale)
        net[2].weight.copy_(torch.tensor([[1.0, 1.0]]) / scale)
        net[2].bias.zero_()
    S = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    R = NeuralNetwork(net).reach(S, method="exact")
    reach_max = max(r.get_max(0, "linprog") for r in R)
    assert true_max - 1e-6 <= reach_max <= true_max + 1e-6


# ---------------------------------------------------------------------------
# Phase LPs: padding, failures, solver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("relax_method", ["range", "area", "bound"])
def test_relax_modes_pad_the_lp_bound(lp_error, relax_method):
    S = _x_star(-1.0, 2e-10)
    out = relu_star_approx([S], relax_factor=0.1, relax_method=relax_method)
    assert F(float(out[0].estimate_ranges()[1][0, 0])) >= _exact_max(S)


@pytest.mark.parametrize("relax_method", ["range", "area", "bound"])
def test_relax_modes_pad_the_lp_min(lp_error, relax_method):
    # true min -4e-10, LP reports +1e-10: passing the neuron through as
    # y = x would give y = -4e-10 at alpha = -1, where relu(x) = 0
    S = _x_star(-4e-10, 1.0)
    out = relu_star_approx([S], relax_factor=0.1, relax_method=relax_method)[0]
    plb, pub = out.predicate_lb.copy(), out.predicate_ub.copy()
    plb[0, 0] = pub[0, 0] = -1.0
    at_lo = Star(out.V, out.C, out.d, plb, pub)
    assert at_lo.estimate_ranges()[1][0, 0] >= 0.0


@pytest.mark.parametrize("relax_method", ["range", "area", "bound"])
def test_relax_modes_keep_a_neuron_whose_lp_fails(lp_fails, relax_method):
    S = _x_star(-1.0, 1.0)
    out = relu_star_approx([S], relax_factor=0.1, relax_method=relax_method)
    assert out[0].estimate_ranges()[1][0, 0] >= 1.0


@pytest.mark.parametrize("relax_method", ["range", "area", "bound"])
def test_relax_modes_on_a_real_highs_undershoot(relax_method):
    """End to end on the thin wedge where HiGHS reported max -3.1e-10 for a
    true +2e-10.

    Documentation of the original instance: since lpsolver returns a
    certified outward optimum, HiGHS no longer undershoots here and this test
    passes without the phase pad. ``test_relax_modes_pad_the_lp_bound`` is the
    test that fails if the pad is removed.
    """
    A = np.array([[0.5979199160832872, -0.8015558458092329],
                  [-0.5979200128356061, 0.8015557736369121],
                  [-0.5657147846899222, 0.5751376701289588],
                  [-0.2537215332365208, 0.38524816005163154],
                  [0.6333020210359026, -0.14099972221182444]])
    b = np.array([-0.3252656027156358, 0.32526558642845826, 0.3557666286869708,
                  0.3924354746944929, 0.3974011595362077])
    f = [-0.8082119740543274, -0.14570769566213082]
    c0 = -0.02002580519576891
    w = np.array([-0.08632645297449967, 0.34139766883127765])  # inside C/d
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([f, [1e-3, 0.0]], dtype=torch.float64))
        net[0].bias.copy_(torch.tensor([c0, 0.0], dtype=torch.float64))
        net[2].weight.copy_(torch.tensor([[1e7, 0.0]], dtype=torch.float64))
        net[2].bias.zero_()
        z = net(torch.tensor(w[None, :])).item()
    assert z >= 1e-4
    S = Star(np.hstack([np.zeros((2, 1)), np.eye(2)]), A, b.reshape(-1, 1),
             -np.ones((2, 1)), np.ones((2, 1)))
    spec = HalfSpace(np.array([[-1.0]]), np.array([[-1e-4]]))   # z >= 1e-4
    R = NeuralNetwork(net).reach(S, method="approx", relax_method=relax_method,
                                 relax_factor=0.5)
    assert verify_specification(R, spec).verdict != "UNSAT"


def test_leakyrelu_pads_the_lp_bound(lp_error):
    gamma = 0.01
    S = _x_star(-1.0, 2e-10)
    for out in (leakyrelu_star_exact([S], gamma=gamma),
                leakyrelu_star_approx([S], gamma=gamma)):
        top = max(F(float(o.estimate_ranges()[1][0, 0])) for o in out)
        assert top >= _exact_max(S)


def test_leakyrelu_keeps_a_star_whose_lp_fails(lp_fails):
    gamma = 0.1
    S = _x_star(-1.0, 1.0)
    for out in (leakyrelu_star_exact([S], gamma=gamma),
                leakyrelu_star_approx([S], gamma=gamma)):
        assert len(out) >= 1
        assert max(o.estimate_ranges()[1][0, 0] for o in out) >= 1.0
        assert min(o.estimate_ranges()[0][0, 0] for o in out) <= -gamma


def test_sign_pads_the_lp_bound(lp_error):
    # true min -4e-10 (sign -1), LP reports +1e-10
    S = _x_star(-4e-10, 1.0)
    for method in ("exact", "approx"):
        out = sign_star(None, [S], method=method)
        assert min(o.estimate_ranges()[0][0, 0] for o in out) <= -1.0


def test_sign_keeps_a_star_whose_lp_fails(lp_fails):
    S = _x_star(-1.0, 1.0)
    for method in ("exact", "approx"):
        out = sign_star(None, [S], method=method)
        assert min(o.estimate_ranges()[0][0, 0] for o in out) <= -1.0
        assert max(o.estimate_ranges()[1][0, 0] for o in out) >= 1.0


def test_phase_lps_use_highs_whatever_the_requested_solver(monkeypatch):
    seen = {"get_range": [], "get_max": [], "get_min": []}
    for name in seen:
        orig = getattr(Star, name)

        def spy(self, index, lp_solver=LPSolver.DEFAULT, _orig=orig, _log=seen[name]):
            _log.append(lp_solver)
            return _orig(self, index, lp_solver)

        monkeypatch.setattr(Star, name, spy)
    S = _x_star(-1.0, 1.0)
    relu_star_exact([S], lp_solver="SCS", parallel=False)
    for relax_method in ("range", "area", "bound"):
        relu_star_approx([S], relax_factor=0.1, lp_solver="SCS",
                         relax_method=relax_method)
    leakyrelu_star_exact([S], lp_solver="SCS")
    leakyrelu_star_approx([S], lp_solver="SCS")
    sign_star(None, [S], method="exact", lp_solver="SCS")
    assert all(seen.values())
    for solvers in seen.values():
        assert all(resolve(s, allow_sentinel=False).is_scipy() for s in solvers)


def test_exact_relu_keeps_a_neuron_whose_max_is_inside_the_solver_shortfall(inexact_scs):
    """The exact ReLU decides phases with HiGHS plus an outward pad whatever
    solver the caller names (``_bound_lp_solver``). A neuron whose true max is
    half an inexact solver's shortfall must keep its active piece: deciding
    the phase from the inexact optimum would call the neuron inactive and drop
    every reachable output above zero, a false UNSAT downstream.

    The shortfall is injected (``inexact_scs``), so the test is the same on
    every host; the earlier version searched for a real SCS shortfall and
    failed its own setup whenever SCS happened to be accurate."""
    rng = np.random.default_rng(0)
    n = 6
    X = Star.from_bounds(-np.ones((n, 1)), np.ones((n, 1)))
    W = rng.normal(size=(8, n))
    b = rng.normal(size=(8, 1))
    Cx = rng.normal(size=(5, X.nVar))
    dx = rng.uniform(0.2, 1.0, size=(5, 1))
    Xc = Star(X.V, np.vstack([X.C, Cx]), np.vstack([X.d, dx]),
              X.predicate_lb, X.predicate_ub)
    S = Xc.affine_map(W, b)
    M = S.get_max(0, "highs")
    b2 = b.copy()
    b2[0, 0] -= M - inexact_scs / 2        # true max of neuron 0: delta / 2 > 0
    T = Xc.affine_map(W, b2)
    true_max = T.get_max(0, "highs")
    assert true_max == pytest.approx(inexact_scs / 2, abs=1e-8)
    assert T.get_max(0, "SCS") < 0         # the inexact solver calls it inactive
    outs = relu_star_exact([T], lp_solver="SCS", parallel=False)
    assert max(o.get_max(0, "highs") or -1.0 for o in outs) >= 0.5 * true_max


# ---------------------------------------------------------------------------
# Triangle relaxations keep the vertex (u, u)
# ---------------------------------------------------------------------------

def _graph_points_satisfy(out, S, f):
    """Every (alpha, f(x(alpha))) for alpha in {-1, 0, 1} satisfies the output
    star's rows and predicate box in exact arithmetic (predicate variables:
    alpha, then the relaxation variable)."""
    c, r = F(float(S.V[0, 0])), F(float(S.V[0, 1]))
    C, d = out.C, out.d.reshape(-1)
    plb, pub = out.predicate_lb.reshape(-1), out.predicate_ub.reshape(-1)
    for alpha in (F(-1), F(0), F(1)):
        point = [alpha, f(c + r * alpha)]
        for j, v in enumerate(point):
            assert F(float(plb[j])) <= v <= F(float(pub[j]))
        for i in range(C.shape[0]):
            lhs = sum(F(float(C[i, j])) * point[j] for j in range(2))
            assert lhs <= F(float(d[i])), (alpha, i, float(lhs - F(float(d[i]))))


@pytest.mark.parametrize("lo,hi", [(-1.0, 1e-4), (-1e-4, 1e-4), (-3.0, 7.0),
                                   (-1e-12, 1.0), (-0.7, 0.3),
                                   # the rounded secant row needs its slack
                                   (-38.832520223889176, 1.4232359386968964e-05)])
def test_relu_triangle_contains_the_graph(lo, hi):
    S = _x_star(lo, hi)
    out = relu_star_approx([S], relax_method="standard")[0]
    assert out.nVar == 2
    _graph_points_satisfy(out, S, lambda x: max(F(0), x))


@pytest.mark.parametrize("gamma", [0.01, 2.0, -0.5])
@pytest.mark.parametrize("lo,hi", [(-1.0, 1e-4), (-1e-4, 1e-4), (-3.0, 7.0),
                                   (-0.7, 0.3),
                                   # the rounded gamma*x row needs its slack
                                   (-2.4416780152088147, 2.4416780152088147)])
def test_leakyrelu_triangle_contains_the_graph(lo, hi, gamma):
    # gamma > 1 makes LeakyReLU concave (the relaxation used to be empty)
    g = F(gamma)
    S = _x_star(lo, hi)
    out = leakyrelu_star_approx([S], gamma=gamma)[0]
    assert out.nVar == 2
    _graph_points_satisfy(out, S, lambda x: x if x >= 0 else g * x)


def test_relu_triangle_upper_bound_reaches_relu_of_u():
    S = _x_star(-1e-4, 1e-4)
    out = relu_star_approx([S], relax_method="standard")[0]
    z = out.affine_map(np.array([[1e6]]))
    assert z.get_box().ub[0, 0] >= 100.0 - 1e-6


# ---------------------------------------------------------------------------
# Sign: sign(0) = 0, no unsound tolerance, ImageStar kept
# ---------------------------------------------------------------------------

def _sign_values(out, i=0):
    """Constant values of coordinate ``i`` over exact-mode output stars."""
    vals = set()
    for o in out:
        lo, hi = o.estimate_ranges()
        assert lo[i, 0] == hi[i, 0]
        vals.add(float(lo[i, 0]))
    return vals


def test_sign_exact_keeps_minus_one_just_below_zero():
    S = _x_star(-5e-11, 1.0)          # sign(-5e-11) = -1
    assert -1.0 in _sign_values(sign_star(None, [S], method="exact"))


def test_sign_of_a_box_touching_zero_includes_zero():
    S = _x_star(0.0, 1.0)             # sign(0) = 0
    assert _sign_values(sign_star(None, [S], method="exact")) == {0.0, 1.0}
    lo, hi = sign_star(None, [S], method="approx")[0].estimate_ranges()
    assert lo[0, 0] == 0.0 and hi[0, 0] == 1.0


def test_sign_of_a_crossing_neuron_includes_zero():
    S = _x_star(-1.0, 1.0)
    assert _sign_values(sign_star(None, [S], method="exact")) == {-1.0, 0.0, 1.0}


def test_sign_of_a_dead_unit_is_zero():
    S = Star(np.array([[0.2, 0.1], [0.0, 0.0]]), np.zeros((0, 1)),
             np.zeros((0, 1)), -np.ones((1, 1)), np.ones((1, 1)))
    for method in ("exact", "approx"):
        out = sign_star(None, [S], method=method)
        assert _sign_values(out, 1) == {0.0}
        assert _sign_values(out, 0) == {1.0}


def test_sign_exact_split_keeps_the_imagestar_type():
    lb = np.array([-1.0, 0.5]).reshape(1, 2, 1)
    ub = np.array([1.0, 1.5]).reshape(1, 2, 1)
    S = ImageStar.from_bounds(lb, ub, height=1, width=2, num_channels=1)
    out = sign_star(None, [S], method="exact")
    assert len(out) == 3
    assert all(isinstance(o, ImageStar) for o in out)


# ---------------------------------------------------------------------------
# Zono ReLU keeps ImageZono; Hexatope/Octatope zero bounds
# ---------------------------------------------------------------------------

def test_relu_on_imagezono_keeps_hwc_order_for_flatten():
    m = nn.Sequential(nn.ReLU(), nn.Flatten(), nn.Linear(4, 1, bias=False)).double()
    with torch.no_grad():
        m[2].weight[:] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])   # CHW index 1
    x_chw = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=torch.float64)
    y = m(x_chw).item()                                         # 2.0
    hwc = x_chw[0].permute(1, 2, 0).numpy()
    Z = ImageZono.from_bounds(hwc, hwc, 1, 2, 2)
    assert isinstance(relu_zono_approx([Z])[0], ImageZono)
    out = NeuralNetwork(m).reach(Z, method="approx")[0]
    lo, hi = out.get_bounds()
    assert lo[0, 0] <= y <= hi[0, 0]


@pytest.mark.parametrize("relu, cls", [(relu_hexatope, Hexatope),
                                       (relu_octatope, Octatope)])
def test_hexoct_relu_does_not_split_exact_zero_bounds(relu, cls):
    # two dead coordinates, one crossing, one with box lower bound 0
    H = cls.from_bounds(np.array([0.0, 0.0, -1.0, 0.0]),
                        np.array([0.0, 0.0, 1.0, 2.0]))
    assert len(relu([H], solver="lp")) == 2


# ---------------------------------------------------------------------------
# LeakyReLU on a Box, any slope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma", [0.01, 2.0, -0.5])
@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (-2.0, -1.0), (0.5, 2.0),
                                   (-3.0, 0.5)])
def test_leakyrelu_box_is_the_range_for_any_slope(lo, hi, gamma):
    # f(x) = x for x >= 0, gamma*x otherwise: its range on [lo, hi] is spanned
    # by f(lo), f(hi) and, when lo < 0 < hi, f(0) = 0 (a negative slope used
    # to give [gamma*lo, hi], leaving out 0 for [-1, 1], and an inverted box
    # for [-2, -1])
    def f(x):
        return x if x >= 0 else gamma * x

    vals = [f(lo), f(hi)] + ([0.0] if lo < 0 < hi else [])
    out = leakyrelu_box([Box(np.array([[lo]]), np.array([[hi]]))], gamma)[0]
    assert out.lb[0, 0] <= min(vals) and out.ub[0, 0] >= max(vals)
    assert out.lb[0, 0] >= min(vals) - 1e-12 and out.ub[0, 0] <= max(vals) + 1e-12


def test_leakyrelu_negative_slope_box_reach_keeps_f_of_zero():
    net = nn.Sequential(nn.Linear(1, 1), nn.LeakyReLU(-0.5),
                        nn.Linear(1, 1)).double()
    with torch.no_grad():
        for m in (net[0], net[2]):
            m.weight.fill_(1.0)
            m.bias.zero_()
    R = NeuralNetwork(net).reach(Box(np.array([[-1.0]]), np.array([[1.0]])),
                                 method="approx")
    spec = HalfSpace(np.array([[1.0]]), np.array([[0.1]]))   # unsafe: y <= 0.1
    assert verify_specification(R, spec).verdict != "UNSAT"  # f(0) = 0



# ---------------------------------------------------------------------------
# Precomputed bounds that cannot describe the layer
# ---------------------------------------------------------------------------

_PRE_CONSUMERS = {
    "relu_standard": lambda S, pre: relu_star_approx(
        [S], relax_method="standard", precomputed_bounds=pre),
    "relu_range": lambda S, pre: relu_star_approx(
        [S], relax_factor=0.5, relax_method="range", precomputed_bounds=pre),
    "relu_area": lambda S, pre: relu_star_approx(
        [S], relax_factor=0.5, relax_method="area", precomputed_bounds=pre),
    "relu_bound": lambda S, pre: relu_star_approx(
        [S], relax_factor=0.5, relax_method="bound", precomputed_bounds=pre),
    "leakyrelu_approx": lambda S, pre: leakyrelu_star_approx(
        [S], gamma=0.1, precomputed_bounds=pre),
    "relu_exact": lambda S, pre: relu_star_exact(
        [S], parallel=False, precomputed_bounds=pre),
    "leakyrelu_exact": lambda S, pre: leakyrelu_star_exact(
        [S], gamma=0.1, precomputed_bounds=pre),
}


@pytest.mark.parametrize("consumer", sorted(_PRE_CONSUMERS))
def test_precomputed_bounds_of_another_size_raise(consumer):
    # A pre-pass that mis-sized the node gave one neuron's bounds, [-5, -4];
    # np.maximum broadcast them to all three neurons and zeroed the layer.
    S = Star.from_bounds(np.ones((3, 1)), 2.0 * np.ones((3, 1)))
    with pytest.raises(ValueError):
        _PRE_CONSUMERS[consumer](S, (np.array([[-5.0]]), np.array([[-4.0]])))


@pytest.mark.parametrize("consumer", sorted(_PRE_CONSUMERS))
def test_precomputed_bounds_disjoint_from_the_star_are_dropped(consumer):
    # x0 in [1, 2], x1 in [-1, 1]. Bounds of [-5, -4] for x0 (another neuron's
    # bounds) do not meet the star's own enclosure, so they are not the
    # bounds of this layer: y0 = 2 must stay reachable.
    S = Star.from_bounds(np.array([[1.0], [-1.0]]), np.array([[2.0], [1.0]]))
    pre = (np.array([[-5.0], [-1.0]]), np.array([[-4.0], [1.0]]))
    out = _PRE_CONSUMERS[consumer](S, pre)
    assert max(o.get_ranges()[1][0, 0] for o in out) >= 2.0 - 1e-9


@pytest.mark.parametrize("consumer", sorted(_PRE_CONSUMERS))
def test_nan_precomputed_bound_is_ignored(consumer):
    # x0 = x in [-1, 1] (crossing), x1 = x + 2 (active). A NaN bound for x0
    # made every phase test false, so x0 was kept as the identity and the
    # output lost (f(-1), 1): y0 = -1 came with y1 = 1 instead.
    S = _x_star(-1.0, 1.0).affine_map(np.array([[1.0], [1.0]]),
                                      np.array([[0.0], [2.0]]))
    pre = (np.array([[np.nan], [1.0]]), np.array([[np.nan], [3.0]]))
    out = _PRE_CONSUMERS[consumer](S, pre)
    gamma = 0.1 if consumer.startswith("leaky") else 0.0
    for x in (-1.0, -0.5, 0.0, 1.0):
        y = np.array([x if x >= 0 else gamma * x, x + 2.0])
        assert any(o.contains(y) for o in out), y
