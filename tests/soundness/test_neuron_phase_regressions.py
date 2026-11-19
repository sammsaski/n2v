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
  * The cross-cut at the end of the file: a neuron whose true range crosses
    zero by 2^-70 (at its minimum, at its maximum, or both), or by 2.8e-17
    that a sum to nearest turns into exactly 0, through every kernel that
    decides a phase (ReLU, LeakyReLU and Sign on a Star, exact and every
    approximate rule; on a Zono; on a Box) by ``reach_layer`` and through the
    model entry. A phase test with a tolerance the unsafe way, or a bound
    source that rounds onto 0, gives the neuron one phase there.

Oracle: the exact image in rationals (``tests/_oracles/rational.py``), read
from the stored arrays of the input set; membership of its vertices in a
returned set decided in fractions with no slack unless a test states one.
Float64 replays of a model are checked by one LP on each returned set's own
rows (``assert_covered``). No reference comes from n2v.
"""

from fractions import Fraction as F

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper
from onnx2torch.node_converters.functions import OnnxFunction

from n2v.nn import NeuralNetwork
from n2v.nn.layer_ops.dispatcher import reach_layer
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
from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.sets.halfspace import HalfSpace
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils import lpsolver
from n2v.utils.lp_solver_enum import LPSolver, resolve
from n2v.utils.model_loader import load_onnx
from n2v.utils.verify_specification import verify_specification
from tests._oracles import rational as R
from tests.cell_support import (assert_covered, reach_model, reach_one_layer,
                                save_onnx)


LP_ERR = 5e-10   # HiGHS error measured at the zero boundary of thin wedges


def _x_star(lo, hi):
    """1-D star ``x = c + r alpha``, ``alpha in [-1, 1]``, over ``[lo, hi]``."""
    return Star.from_bounds(np.array([[lo]]), np.array([[hi]]))


def _exact_max(S, i=0):
    """Exact rational max of row ``i`` of a box-only star."""
    return F(float(S.V[i, 0])) + sum(abs(F(float(g))) for g in S.V[i, 1:])


# ---------------------------------------------------------------------------
# The exact reference: no n2v method computes it
# ---------------------------------------------------------------------------

def _pieces(input_set):
    """The identity on ``input_set`` as exact pieces: its predicate polytope
    (the rows and the finite predicate bounds, read from the stored arrays by
    ``rational.star``) mapped by its own centre and generators. A Box reads as
    the box of its bounds, a Zono as its generators over [-1, 1]."""
    c, G, C, d, lb, ub = R.star(input_set)
    n = len(lb)
    A, b = [list(row) for row in C], list(d)
    for j in range(n):
        unit = [F(int(i == j)) for i in range(n)]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    return R.affine(R.polytope(A, b), G, c)


def _widened(computed, slack):
    """A set read by ``rational.star``, grown by the box [-slack, slack] in
    every output coordinate (one new predicate variable per coordinate)."""
    c, G, C, d, lb, ub = computed
    n = len(c)
    G = [list(row) + [F(slack) if i == j else F(0) for j in range(n)]
         for i, row in enumerate(G)]
    C = [list(row) + [F(0)] * n for row in C]
    return c, G, C, d, list(lb) + [F(-1)] * n, list(ub) + [F(1)] * n


def _assert_holds_image(pieces, sets, what, slack=0.0):
    """Every piece of the exact image has all its vertices in one returned
    set, decided in fractions with no slack (a convex set that holds the
    vertices of a polytope holds the polytope). ``slack`` grows every
    returned set by a box of that half-width first; only the LeakyReLU Zono
    cell passes one, see there."""
    if not sets:
        raise AssertionError(f"{what}: no set returned")
    readers = [R.Membership(_widened(R.star(s), slack) if slack
                            else R.star(s)) for s in sets]
    for i, piece in enumerate(pieces):
        image = piece.image()
        if any(all(member(y) for y in image) for member in readers):
            continue
        outside = [y for y in image if not any(m(y) for m in readers)]
        shown = ([float(v) for v in outside[0]] if outside else
                 "none; its vertices are spread over several sets")
        raise AssertionError(
            f"{what}: piece {i} of the {len(pieces)} pieces of the exact "
            f"image is held by none of the {len(sets)} returned sets; a "
            f"vertex in none of them: {shown}")


def _holds_at(sets, alpha0, y):
    """Is the output ``y`` in a returned set at a point whose predicate
    variable 0 is ``alpha0``? Exact. The inputs that use it are stars on one
    predicate variable, which every rule keeps as variable 0; pinning it ties
    an output to its input point, so a neuron given the wrong phase shows
    although its own interval still covers the value."""
    for s in sets:
        c, G, C, d, lb, ub = R.star(s)
        lb, ub = list(lb), list(ub)
        lb[0] = ub[0] = F(alpha0)
        if R.member((c, G, C, d, lb, ub), [F(v) for v in y]):
            return True
    return False


def _query_holds_each(sets, points, what):
    """The library's own membership query, ``Star.contains``, reports every
    point of ``points`` as a member of some returned set. Called only on true
    outputs that the test's independent oracle has already shown to be in
    the result, so a False is the query's own error: its feasibility system
    (the equality rows ``V a = y - c`` stacked as two opposite inequalities,
    then certified empty) calls a reachable output unreachable. That is the
    answer a user reads who asks the reach result whether an output is
    reachable."""
    for y in points:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if not any(s.contains(y) for s in sets):
            raise AssertionError(
                f"{what}: Star.contains reports the true output {y.tolist()}, "
                f"which the result holds, in none of the {len(sets)} "
                f"returned sets")


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
    """An empty Star (a0 + a1 >= 1.5 and <= 1) is certified empty and dropped
    by exact ReLU, by the kernel and through the model entry. Catches an
    emptiness check (``Star.is_empty_set``, its Farkas certificate, the row
    scaling) that calls a certified infeasible system non-empty: the star is
    kept and its sets hold points no input reaches."""
    S = _empty_star()
    assert S.get_range(0) == (None, None)
    assert relu_star_exact([S], parallel=False) == []
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU()).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.eye(2, dtype=torch.float64))
        net[0].bias.zero_()
    assert NeuralNetwork(net).reach(S, method="exact") == []


def test_exact_relu_keeps_a_star_whose_lp_fails(lp_fails):
    """No range LP returns an optimum: the neuron is undecided and split,
    never dropped as empty, and both pieces of the exact image (0 for x <= 0,
    x for x >= 0) are in a returned set, decided in fractions."""
    S = _x_star(-1.0, 1.0)
    out = relu_star_exact([S], parallel=False)
    assert len(out) == 2   # undecided -> split, not dropped
    _assert_holds_image(R.relu(_pieces(S)), out, "exact ReLU, LPs failing")


def test_exact_relu_float32_net_does_not_crash_and_covers_outputs():
    """Exact reach of a float32 Linear, ReLU, Linear, ReLU net whose second-
    layer neurons touch zero (weights 1e-3 and bias -3e-3, rounded to
    float32), through the model entry: the float64 evaluation of the stored
    weights at 34 points is covered, by one LP on each returned set's rows,
    and the result's own membership query (``Star.contains``) agrees on
    them. Catches a crash on a float32 layer, a neuron near zero given the
    wrong phase, and a membership query that calls a reachable output
    unreachable."""
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2), nn.ReLU())
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1., 1.], [1., 0.], [0., 1.]]))
        net[0].bias.copy_(torch.tensor([-1., 1., 1.]))
        net[2].weight.copy_(torch.tensor([[0., 1e-3, 1e-3], [1., 0., 0.]]))
        net[2].bias.copy_(torch.tensor([-3e-3, -0.25]))
    S = Star.from_bounds(np.zeros((2, 1)), np.ones((2, 1)))
    sets = NeuralNetwork(net).reach(S, method="exact")
    assert len(sets) >= 1
    rng = np.random.default_rng(0)
    X = np.vstack([rng.uniform(0, 1, (30, 2)), [[0, 0], [1, 1], [1, 0], [0, 1]]])
    with torch.no_grad():   # the same float32 weights, evaluated in float64
        Y = net.double()(torch.tensor(X)).numpy()
    # membership by one LP on each returned set's own rows (the test oracle,
    # not n2v's contains)
    assert_covered(sets, list(Y), "exact reach of a float32 net")
    _query_holds_each(sets, Y, "exact reach of a float32 net")


def test_parallel_exact_relu_reraises_a_worker_failure():
    """The process-pool exact ReLU re-raises a worker's exception (a ValueError
    from bounds of another size), as the sequential path does. Catches a
    pool that skips the failed star and returns the others' outputs, which
    drops every state the failed star reaches (a false UNSAT downstream)."""
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
    """An LP minimum of exactly 0 (x = alpha cut to alpha >= 0, while the
    interval bound crosses 0) is kept in one star, which holds the exact
    image in fractions. Catches a split on a bound of exactly 0 (the star
    count doubled for each such neuron) and a sliver on the wrong side."""
    S = Star(np.array([[0.0, 1.0]]), np.array([[-1.0]]), np.array([[0.0]]),
             -np.ones((1, 1)), np.ones((1, 1)))
    out = _step_relu(S, 0)
    assert len(out) == 1
    _assert_holds_image(R.relu(_pieces(S)), out,
                        "_step_relu on an LP minimum of exactly 0")


def test_second_relu_does_not_split_again():
    """A second ReLU after an exact ReLU finds every neuron of every child
    decided and splits nothing: the exact reach of ReLU, ReLU has as many
    stars as that of one ReLU. Catches LP bounds that lose their certificate
    and land inside the range, so that a decided neuron is split again."""
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
    """A zero-bias net through the model entry: every child star contains the
    origin, so every sign-stable deeper neuron has an LP bound of exactly 0.
    At most 24 stars (241 when each such neuron was split), and the float64
    outputs of 43 points covered, by one LP on each set's rows. Catches the
    split of such neurons and a sliver on the wrong side."""
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(2, 4, bias=False), nn.ReLU(),
                        nn.Linear(4, 4, bias=False), nn.ReLU(),
                        nn.Linear(4, 1, bias=False)).double()
    S = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    sets = NeuralNetwork(net).reach(S, method="exact")
    assert len(sets) <= 24
    rng = np.random.default_rng(0)
    X = np.vstack([rng.uniform(-1, 1, (40, 2)), [[0, 0], [1, 1], [-1, 1]]])
    with torch.no_grad():
        Y = net(torch.tensor(X)).numpy()
    assert_covered(sets, list(Y), "exact reach of a zero-bias net")


def test_exact_relu_widens_round_to_nearest_prepass_bounds():
    """Exact ReLU and LeakyReLU given interval bounds computed to nearest whose
    upper bound 0.0 lies ulps inside the true maximum 2.8e-17: the exact
    image is held, in fractions. Catches pre-pass bounds that decide a phase
    without the LP pad (the neuron zeroed or scaled), and a pad built from
    the wrong bound.

    The bounds are summed to nearest here. The case used to take them from
    ``bounds_precomputation._ibp_linear``, which now rounds outward (upper
    bound 2.1e-15 on these data), so the bound it handed the kernel no longer
    lay inside the range and the case could not fail with the pad removed;
    the exact kernels still promise to widen given bounds before they decide
    a phase (``_relu_single_star_exact``)."""
    lb, ub = np.array([0.0, 0.0]), np.array([0.1, 0.7])
    W, b = np.array([[1.0, 1.0]]), np.array([-0.7999999999999999])
    true_max = F(0.1) + F(0.7) + F(float(b[0]))
    S = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1)).affine_map(
        W, b.reshape(-1, 1))
    plb, pub = W @ lb + b, W @ ub + b          # summed to nearest
    if not (true_max > 0 and pub[0] == 0.0):
        pytest.fail(f"setup: the upper bound {pub[0]} summed to nearest is "
                    f"not 0.0 inside the true maximum {float(true_max)}")
    out = relu_star_exact([S], parallel=False,
                          precomputed_bounds=(plb.reshape(-1, 1),
                                              pub.reshape(-1, 1)))
    # the exact image, whose top is relu(true_max) = true_max, in fractions
    _assert_holds_image(R.relu(_pieces(S)), out,
                        "exact ReLU with round-to-nearest IBP bounds")
    out = leakyrelu_star_exact([S], gamma=0.01,
                               precomputed_bounds=(plb.reshape(-1, 1),
                                                   pub.reshape(-1, 1)))
    _assert_holds_image(R.leaky_relu(_pieces(S), 0.01), out,
                        "exact LeakyReLU with round-to-nearest IBP bounds")


@pytest.mark.parametrize("op", ["ReLU", "LeakyReLU"])
def test_exact_relu_widens_round_to_nearest_prepass_lower_bounds(op):
    """The lower side of the case above, through ``reach_layer``: entry 0 =
    0.7999999999999999 + 0.1 a0 + 0.7 a1 has the true minimum -2.8e-17 and
    the lower bound 0.0 when summed to nearest; entries 1 = 4 + a0 and 2 =
    4 + a1 pin the predicate point. Exact ReLU and LeakyReLU (slope 0.25)
    given these bounds must widen the lower bound by the LP pad before they
    test ``lb < 0``: taken as given, entry 0 is kept as the identity, and
    relu(x) = 0 (or 0.25 x) at the minimum leaves the exact image, decided in
    fractions. Catches a pad applied to the upper bound only (or to
    neither), which the upper side cannot show."""
    S = _phase_input("Star", "min", "rounding")
    c, G = S.V[:, 0], S.V[:, 1:]
    plb, pub = c - np.abs(G).sum(axis=1), c + np.abs(G).sum(axis=1)
    true_min = F(float(c[0])) - F(float(G[0, 0])) - F(float(G[0, 1]))
    if not (true_min < 0 and plb[0] == 0.0):
        pytest.fail(f"setup: the lower bound {plb[0]} summed to nearest is "
                    f"not 0.0 above the true minimum {float(true_min)}")
    layer, rule = ((nn.ReLU(), R.relu) if op == "ReLU" else
                   (nn.LeakyReLU(0.25), lambda p: R.leaky_relu(p, 0.25)))
    what = f"exact {op} given a lower bound summed to nearest"
    out = reach_one_layer(layer, [S], "exact", what,
                          precomputed_bounds=(plb.reshape(-1, 1),
                                              pub.reshape(-1, 1)))
    _assert_holds_image(rule(_pieces(S)), out, what)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_thin_neuron_on_one_side_of_zero_keeps_its_side(sign):
    # x = sign * 4.5e-8 * (a0 + a1) with a0 + a1 >= 1: the interval bound
    # crosses 0, the LP range sign * [4.5e-8, 9e-8] lies within the pads.
    # Exact either way (no sliver variable); where the other side can be
    # certified empty (highspy), one star.
    """A thin neuron whose LP range lies within the pads on one side of 0 while
    its interval bound crosses 0: exact ReLU and LeakyReLU keep one star,
    the identity or the row scaled by the slope, with no sliver variable,
    because the other side is certified empty. Catches a side taken without
    the certificate, the wrong side, and a sliver where the exact answer is
    one phase."""
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
    # empty, so the neuron must still be split (not passed through as y = x).
    # The output at alpha = -1 is checked in fractions; the LeakyReLU slope
    # is dyadic, so the scaled row of the x <= 0 piece is exact (a slope such
    # as 0.01 is rounded to nearest there, ROUND-9, marked elsewhere).
    """A thin neuron with true range [-4e-10, 9e-8] whose LP minimum is
    reported +1e-10: the side x <= 0 is not empty, so the neuron must still
    be split, and f(x) at alpha = -1 must be in a returned set (fractions,
    predicate variable pinned). Catches a side taken from the raw LP sign
    without certifying the other side empty."""
    S = _x_star(-4e-10, 9e-8)
    x_lo = F(float(S.V[0, 0])) - F(float(S.V[0, 1]))   # x at alpha = -1
    for out, gamma in ((relu_star_exact([S], parallel=False), 0.0),
                       (leakyrelu_star_exact([S], gamma=0.125), 0.125)):
        f_lo = F(gamma) * x_lo
        assert _holds_at(out, -1, [f_lo]), (
            f"f(x) = {float(f_lo)} at x = {float(x_lo)} (alpha = -1) is in "
            f"none of the {len(out)} returned sets (slope {gamma})")


@pytest.mark.parametrize("act, true_max", [(nn.ReLU(), 2.25),
                                           (nn.LeakyReLU(0.01), 2.245)])
@pytest.mark.parametrize("scale", [1e-8, 3e-8])
def test_exact_reach_splits_a_neuron_whose_whole_range_is_within_the_pad(
        act, true_max, scale):
    """Pre-activations of size ~scale, amplified by 1/scale: each hidden
    neuron's LP range crosses 0 but lies within the pads on both sides. A
    sliver there is decoupled from x and ~8x wider than the range (reach max
    23.75 for scale 1e-8); the split keeps the reach exact.

    Reference: the exact image of the stored (float32-valued) weights in
    fractions; ``true_max`` is its maximum to about 1e-7. Every piece must be
    in one returned set and the largest vertex of the returned sets must not
    exceed the true maximum, both with a pad of 1e-12, which the slope 0.01
    needs: its products are rounded to nearest (ROUND-9, marked elsewhere),
    about 1e-16 here."""
    net = nn.Sequential(nn.Linear(2, 2), act, nn.Linear(2, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 1.0], [1.0, -1.0]]) * scale)
        net[0].bias.copy_(torch.tensor([-0.5, 0.25]) * scale)
        net[2].weight.copy_(torch.tensor([[1.0, 1.0]]) / scale)
        net[2].bias.zero_()
    S = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    sets = NeuralNetwork(net).reach(S, method="exact")
    hidden = R.affine(_pieces(S), net[0].weight.detach().numpy(),
                      net[0].bias.detach().numpy())
    slope = getattr(act, "negative_slope", 0.0)
    pieces = R.affine(R.leaky_relu(hidden, slope),
                      net[2].weight.detach().numpy(),
                      net[2].bias.detach().numpy())
    exact_max = max(y[0] for p in pieces for y in p.image())
    if abs(float(exact_max) - true_max) > 1e-7:
        pytest.fail(f"setup: the exact maximum {float(exact_max)} is not "
                    f"the expected {true_max}")
    what = f"exact {type(act).__name__} with pre-activations of size {scale}"
    _assert_holds_image(pieces, sets, what, slack=1e-12)
    reach_max = max(v[0] for s in sets for v in R.set_vertices(R.star(s)))
    assert reach_max <= exact_max + F(1e-12), (
        f"{what}: the reach reaches {float(reach_max)}, the network only "
        f"{float(exact_max)}")


# ---------------------------------------------------------------------------
# Phase LPs: padding, failures, solver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("relax_method", ["range", "area", "bound"])
def test_relax_modes_pad_the_lp_bound(lp_error, relax_method):
    """Relaxed ReLU (range, area, bound) on x in [-1, 2e-10] whose LP maximum
    is reported 5e-10 low: the neuron must stay crossing, so x at the true
    maximum (alpha = 1) is held, in fractions. Catches an LP maximum used
    without the outward pad (the neuron zeroed)."""
    S = _x_star(-1.0, 2e-10)
    out = relu_star_approx([S], relax_factor=0.1, relax_method=relax_method)
    # relu(x) = x at the true maximum (alpha = 1), in fractions
    assert _holds_at(out, 1, [_exact_max(S)]), (
        f"relax {relax_method}: the output {float(_exact_max(S))} at the true "
        f"maximum of a neuron whose LP maximum is reported 5e-10 low is lost")


@pytest.mark.parametrize("relax_method", ["range", "area", "bound"])
def test_relax_modes_pad_the_lp_min(lp_error, relax_method):
    # true min -4e-10, LP reports +1e-10: passing the neuron through as
    # y = x would give y = -4e-10 at alpha = -1, where relu(x) = 0
    """Relaxed ReLU on x in [-4e-10, 1] whose LP minimum is reported 5e-10
    high: relu(x) = 0 at alpha = -1 must be held, in fractions with the
    predicate variable pinned. Catches an LP minimum used without the pad,
    or padded the wrong way (the neuron passed through as y = x)."""
    S = _x_star(-4e-10, 1.0)
    out = relu_star_approx([S], relax_factor=0.1, relax_method=relax_method)
    assert _holds_at(out, -1, [0]), (
        f"relax {relax_method}: relu(x) = 0 at the true minimum (alpha = -1) "
        f"is lost when the LP minimum is reported 5e-10 high")


@pytest.mark.parametrize("relax_method", ["range", "area", "bound"])
def test_relax_modes_keep_a_neuron_whose_lp_fails(lp_fails, relax_method):
    """Relaxed ReLU when no range LP returns an optimum: the interval estimate
    is kept and the exact image held, in fractions. Catches a failed LP read
    as an optimum of 0 (the neuron zeroed)."""
    S = _x_star(-1.0, 1.0)
    out = relu_star_approx([S], relax_factor=0.1, relax_method=relax_method)
    _assert_holds_image(R.relu(_pieces(S)), out,
                        f"relax {relax_method} with every LP failing")


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
    """Exact and approx LeakyReLU on x in [-1, 2e-10] whose LP maximum is
    reported 5e-10 low: x at the true maximum (alpha = 1) is held, in
    fractions. Catches an LP maximum used without the pad (the neuron scaled
    as always negative)."""
    gamma = 0.01
    S = _x_star(-1.0, 2e-10)
    for method, out in (("exact", leakyrelu_star_exact([S], gamma=gamma)),
                        ("approx", leakyrelu_star_approx([S], gamma=gamma))):
        # f(x) = x at the true maximum (alpha = 1), in fractions
        assert _holds_at(out, 1, [_exact_max(S)]), (
            f"LeakyReLU {method}: the output at the true maximum of a neuron "
            f"whose LP maximum is reported 5e-10 low is lost")


def test_leakyrelu_approx_pads_the_lp_min(lp_error):
    """Approx LeakyReLU through ``reach_layer`` on x in [-4e-10, 1], whose LP
    minimum is reported +1e-10: the neuron still crosses zero only if the
    minimum is padded outward before ``lb < 0`` is tested. Passed through as
    y = x instead, the output at alpha = -1 is x, not slope x, and the true
    output (alpha = -1, slope x) is lost; decided in fractions with the
    predicate variable pinned. The minimum side of
    ``test_leakyrelu_pads_the_lp_bound`` (catches a raw LP minimum, or a
    padding of the wrong sign, in ``_leakyrelu_single_star_approx``)."""
    gamma = 0.25
    S = _x_star(-4e-10, 1.0)
    out = reach_one_layer(nn.LeakyReLU(gamma), [S], "approx",
                          "approx LeakyReLU with an LP minimum 5e-10 high")
    x_lo = F(float(S.V[0, 0])) - F(float(S.V[0, 1]))   # x at alpha = -1
    assert _holds_at(out, -1, [F(gamma) * x_lo]), (
        f"approx LeakyReLU: f(x) = {float(F(gamma) * x_lo)} at the true "
        f"minimum x = {float(x_lo)} is lost when the LP minimum is reported "
        f"5e-10 high")


def test_leakyrelu_keeps_a_star_whose_lp_fails(lp_fails):
    """Exact and approx LeakyReLU (slope 0.1) when no range LP returns an
    optimum: the exact image on [-1, 1] is held, in fractions. Catches a
    failed LP read as a decided phase or as an empty star."""
    gamma = 0.1
    S = _x_star(-1.0, 1.0)
    for method, out in (("exact", leakyrelu_star_exact([S], gamma=gamma)),
                        ("approx", leakyrelu_star_approx([S], gamma=gamma))):
        _assert_holds_image(R.leaky_relu(_pieces(S), gamma), out,
                            f"LeakyReLU {method} with every LP failing")


def test_sign_pads_the_lp_bound(lp_error):
    # true min -4e-10 (sign -1), LP reports +1e-10
    """Exact and approx Sign on x in [-4e-10, 1] whose LP minimum is reported
    5e-10 high: the sign -1 is kept, every piece of the exact image held in
    fractions. Catches a raw LP optimum deciding a sign (a step: an LP error
    flips the output by 1)."""
    S = _x_star(-4e-10, 1.0)
    for method in ("exact", "approx"):
        out = sign_star(None, [S], method=method)
        _assert_holds_image(R.sign(_pieces(S)), out,
                            f"Sign {method}, LP minimum reported 5e-10 high")


def test_sign_keeps_a_star_whose_lp_fails(lp_fails):
    """Exact and approx Sign when no range LP returns an optimum: -1, 0 and +1
    are all held, in fractions. Catches a failed LP read as a decided sign
    or as an empty star."""
    S = _x_star(-1.0, 1.0)
    for method in ("exact", "approx"):
        out = sign_star(None, [S], method=method)
        _assert_holds_image(R.sign(_pieces(S)), out,
                            f"Sign {method} with every LP failing")


def test_phase_lps_use_highs_whatever_the_requested_solver(monkeypatch):
    """Every phase LP of exact and relaxed ReLU, exact and approx LeakyReLU and
    exact Sign runs on a HiGHS backend when the caller names SCS (a spy on
    ``Star.get_range``, ``get_max``, ``get_min``). Catches a phase LP routed
    to the caller's solver, whose optima stop up to 3.6e-5 inside the range."""
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
    # The reference: the input point where neuron 0 is largest, found by
    # scipy's HiGHS on the input's own rows (no n2v call), and the true
    # output there, relu(W a + b2) in float64, checked by one LP on each
    # returned set's rows. Zeroing neuron 0 moves it by delta / 2 = 1e-5.
    from scipy.optimize import linprog
    best = linprog(-W[0], A_ub=Xc.C, b_ub=Xc.d.ravel(), bounds=[(-1, 1)] * n,
                   method="highs")
    if best.status != 0:
        pytest.fail(f"setup: the reference LP did not solve ({best.message})")
    y = np.maximum(W @ best.x + b2.ravel(), 0.0)
    if not y[0] > 0.25 * inexact_scs:
        pytest.fail(f"setup: neuron 0 is {y[0]} at the reference point")
    assert_covered(outs, [y], "exact ReLU asked for SCS, neuron 0 at its "
                   "true maximum")


# ---------------------------------------------------------------------------
# Triangle relaxations keep the vertex (u, u)
# ---------------------------------------------------------------------------

def _graph_points_satisfy(out, S, f):
    """Every (alpha, f(x(alpha))) for alpha in {-1, 0, 1} and at the kink
    x(alpha) = 0 satisfies the output star's rows and predicate box in exact
    arithmetic (predicate variables: alpha, then the relaxation variable).
    The ends and the kink are the vertices of the graph over the neuron's
    range, so a relaxation that holds them holds the graph."""
    c, r = F(float(S.V[0, 0])), F(float(S.V[0, 1]))
    C, d = out.C, out.d.reshape(-1)
    plb, pub = out.predicate_lb.reshape(-1), out.predicate_ub.reshape(-1)
    for alpha in (F(-1), F(0), F(1), -c / r):
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
    """Standard approx ReLU on one crossing neuron: its triangle rows hold the
    graph at both ends and at the kink, in fractions; wide, narrow and
    lopsided ranges, and one whose secant row needs its rounding slack.
    Catches a face below the vertex (u, u), a slope rounded the unsafe way
    and a lower face of the wrong sign."""
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
    """Approx LeakyReLU on one crossing neuron, slopes 0.01 (convex), 2
    (concave, every inequality flips) and -0.5: its rows hold the graph at
    both ends and at the kink, in fractions. Catches a face below a vertex,
    the convex rows used for a concave slope, a predicate bound of the new
    variable that leaves out f(l), and a rounded row without its slack."""
    g = F(gamma)
    S = _x_star(lo, hi)
    out = leakyrelu_star_approx([S], gamma=gamma)[0]
    assert out.nVar == 2
    _graph_points_satisfy(out, S, lambda x: x if x >= 0 else g * x)


def test_relu_triangle_upper_bound_reaches_relu_of_u():
    """The box of the relaxed output, scaled by 1e6, reaches 1e6 relu(u)
    (u the stored upper end), up to a pad of 1e-12 of that value for the
    rounding of the scaling (7b): a triangle face below the vertex (u, u)
    loses outputs near the top, and the range query of the result is where a
    user reads it."""
    S = _x_star(-1e-4, 1e-4)
    out = relu_star_approx([S], relax_method="standard")[0]
    z = out.affine_map(np.array([[1e6]]))
    top = F(1e6) * _exact_max(S)
    assert F(float(z.get_box().ub[0, 0])) >= top * (1 - F(1e-12))


# ---------------------------------------------------------------------------
# Sign: sign(0) = 0, no unsound tolerance, ImageStar kept
# ---------------------------------------------------------------------------

def _sign_values(out, i=0):
    """The values coordinate ``i`` takes at the vertices of the returned
    stars, in fractions: a constant output gives its one value, and a star
    cut to an empty polytope gives none, whatever constant it carries (an
    empty -1 child would otherwise count as holding the output -1)."""
    vals = set()
    for o in out:
        vals.update(float(v[i]) for v in R.set_vertices(R.star(o)))
    return vals


def test_sign_exact_keeps_minus_one_just_below_zero():
    """Exact Sign on x in [-5e-11, 1]: a non-empty -1 child exists. Catches a
    tolerance that calls a bound of -5e-11 zero, and a -1 child cut to an
    empty polytope."""
    S = _x_star(-5e-11, 1.0)          # sign(-5e-11) = -1
    assert -1.0 in _sign_values(sign_star(None, [S], method="exact"))


def test_sign_of_a_box_touching_zero_includes_zero():
    """Sign on x in [0, 1] exactly: the exact method returns exactly the values
    0 and 1 (no -1, which no input has: exactness is claimed), the approx
    method holds both pieces of the exact image, in fractions, and the range
    a user reads from it (``estimate_ranges``, over the new variable's
    predicate box [0, 1]) holds [0, 1], one-sided. Catches sign(0) = 0
    dropped when a bound is exactly 0 (taken as positive), a neuron with one
    bound of exactly 0 taken as the constant 0 (+1 lost), and an interval
    range query that reads a predicate box as symmetric ([0, 1] as [0, 0])."""
    S = _x_star(0.0, 1.0)             # sign(0) = 0
    assert _sign_values(sign_star(None, [S], method="exact")) == {0.0, 1.0}
    out = sign_star(None, [S], method="approx")
    _assert_holds_image(R.sign(_pieces(S)), out, "approx Sign on [0, 1]")
    lo, hi = out[0].estimate_ranges()
    assert lo[0, 0] <= 0.0 and hi[0, 0] >= 1.0, (
        f"the range query of approx Sign on [0, 1] gives [{lo[0, 0]}, "
        f"{hi[0, 0]}], which leaves out a value of the result")


def test_sign_of_a_crossing_neuron_includes_zero():
    """Exact Sign on x in [-1, 1] returns exactly the values -1, 0 and 1.
    Catches the child x = 0 dropped (sign(0) = 0 lost; the two halves alone
    give -1 and 1)."""
    S = _x_star(-1.0, 1.0)
    assert _sign_values(sign_star(None, [S], method="exact")) == {-1.0, 0.0, 1.0}


def test_sign_of_a_dead_unit_is_zero():
    """A row with no generator (the constant 0) is the constant 0 under both
    methods, beside a positive neuron that stays +1. Catches the dead-unit
    test reading a zero row as negative or positive, and a constant written
    to the wrong row."""
    S = Star(np.array([[0.2, 0.1], [0.0, 0.0]]), np.zeros((0, 1)),
             np.zeros((0, 1)), -np.ones((1, 1)), np.ones((1, 1)))
    for method in ("exact", "approx"):
        out = sign_star(None, [S], method=method)
        assert _sign_values(out, 1) == {0.0}
        assert _sign_values(out, 0) == {1.0}


def test_sign_exact_split_keeps_the_imagestar_type():
    """Exact Sign on an ImageStar that splits keeps every child an ImageStar.
    Catches a split child returned as a flat star in the (H, W, C) order,
    which a following Flatten reads as (C, H, W)."""
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
    """ReLU, Flatten, Linear on a (2, 1, 2) ImageZono of one point: ReLU
    returns an ImageZono and the model's output 2.0 is in the result (one LP
    on the generators). Catches the ReLU result returned as a flat Zono, so
    that Flatten skips its reorder and the Linear reads another neuron (a
    false UNSAT)."""
    m = nn.Sequential(nn.ReLU(), nn.Flatten(), nn.Linear(4, 1, bias=False)).double()
    with torch.no_grad():
        m[2].weight[:] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])   # CHW index 1
    x_chw = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=torch.float64)
    y = m(x_chw).item()                                         # 2.0
    hwc = x_chw[0].permute(1, 2, 0).numpy()
    Z = ImageZono.from_bounds(hwc, hwc, 1, 2, 2)
    assert isinstance(relu_zono_approx([Z])[0], ImageZono)
    out = NeuralNetwork(m).reach(Z, method="approx")
    # one LP on the returned zonotope's own generators, not its get_bounds
    assert_covered(out, [[y]], "ReLU, Flatten, Linear on an ImageZono")


@pytest.mark.parametrize("relu, cls", [(relu_hexatope, Hexatope),
                                       (relu_octatope, Octatope)])
def test_hexoct_relu_does_not_split_exact_zero_bounds(relu, cls):
    # two dead coordinates, one crossing, one with box lower bound 0
    """Hexatope and Octatope ReLU with solver 'lp' on two dead coordinates, one
    crossing [-1, 1] and one with box lower bound 0: exactly two sets (only
    the crossing neuron split), holding the corners of the exact image (LP
    oracle). Catches a padded zero bound turned crossing (2^k sets for k
    such neurons) and a crossing neuron given one phase."""
    H = cls.from_bounds(np.array([0.0, 0.0, -1.0, 0.0]),
                        np.array([0.0, 0.0, 1.0, 2.0]))
    out = relu([H], solver="lp")
    assert len(out) == 2
    # the crossing neuron keeps both phases: the corners of the exact image
    # [0, 0] x [0, 0] x [0, 1] x [0, 2] are in the result, by one LP on the
    # rows (the split pieces carry solver optima with no pad, HEXOCT-4, about
    # 1e-8 here, within the oracle's slack)
    corners = [np.array([0.0, 0.0, a, b]) for a in (0.0, 1.0) for b in (0.0, 2.0)]
    assert_covered(out, corners, f"ReLU on a {cls.__name__} with exact zero bounds")


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
    """LeakyReLU on a Box, slopes 0.01, 2 and -0.5, crossing, negative,
    positive and lopsided intervals: the result holds f(lo), f(hi) and, for
    a crossing interval, f(0) = 0. Catches [slope lo, hi] for a negative
    slope (f(0) left out, an all-negative box inverted) and the ends taken
    from the wrong bound."""
    def f(x):
        return x if x >= 0 else gamma * x

    vals = [f(lo), f(hi)] + ([0.0] if lo < 0 < hi else [])
    out = leakyrelu_box([Box(np.array([[lo]]), np.array([[hi]]))], gamma)[0]
    assert out.lb[0, 0] <= min(vals) and out.ub[0, 0] >= max(vals)


def test_leakyrelu_negative_slope_box_reach_keeps_f_of_zero():
    """Linear, LeakyReLU(-0.5), Linear on the Box [-1, 1] through the model
    entry and ``verify_specification``: the output f(0) = 0 lies in the
    unsafe region y <= 0.1, so the verdict must not be UNSAT. Catches f(0)
    dropped on the public route and a verdict fast path that reads a Box
    wrongly."""
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

def _through_reach_layer(slope, method, **opts):
    """A consumer of precomputed bounds: one ReLU (``slope`` None) or
    LeakyReLU layer reached through ``reach_layer``, the public route that
    hands the option to the kernel."""
    def consume(S, pre):
        layer = nn.ReLU() if slope is None else nn.LeakyReLU(slope)
        return reach_one_layer(
            layer, [S], method, f"{type(layer).__name__} {method} {opts} with "
            f"precomputed bounds", precomputed_bounds=pre, **opts)
    return consume


_PRE_CONSUMERS = {
    "relu_standard": _through_reach_layer(None, "approx",
                                          relax_method="standard"),
    "relu_range": _through_reach_layer(None, "approx", relax_factor=0.5,
                                       relax_method="range"),
    "relu_area": _through_reach_layer(None, "approx", relax_factor=0.5,
                                      relax_method="area"),
    "relu_bound": _through_reach_layer(None, "approx", relax_factor=0.5,
                                       relax_method="bound"),
    "leakyrelu_approx": _through_reach_layer(0.1, "approx"),
    "relu_exact": _through_reach_layer(None, "exact"),
    "leakyrelu_exact": _through_reach_layer(0.1, "exact"),
}


def _pre_rule(consumer):
    """The exact rule a consumer of ``_PRE_CONSUMERS`` computes."""
    if consumer.startswith("leaky"):
        return lambda p: R.leaky_relu(p, 0.1)
    return R.relu


@pytest.mark.parametrize("consumer", sorted(_PRE_CONSUMERS))
def test_precomputed_bounds_of_another_size_raise(consumer):
    # A pre-pass that mis-sized the node gave one neuron's bounds, [-5, -4];
    # np.maximum broadcast them to all three neurons and zeroed the layer.
    """Precomputed bounds of one entry for a layer of three, through
    ``reach_layer``: every consumer (ReLU standard, range, area, bound and
    exact; LeakyReLU approx and exact) raises ValueError. Catches a size-1
    array broadcast to every neuron (the layer zeroed) and the option
    dropped on its way to the kernel."""
    S = Star.from_bounds(np.ones((3, 1)), 2.0 * np.ones((3, 1)))
    with pytest.raises(ValueError):
        _PRE_CONSUMERS[consumer](S, (np.array([[-5.0]]), np.array([[-4.0]])))


@pytest.mark.parametrize("consumer", sorted(_PRE_CONSUMERS))
def test_precomputed_bounds_disjoint_from_the_star_are_dropped(consumer):
    # x0 in [1, 2], x1 in [-1, 1]. Bounds of [-5, -4] for x0 (another neuron's
    # bounds) do not meet the star's own enclosure, so they are not the
    # bounds of this layer: y0 = 2 must stay reachable.
    """Bounds [-5, -4] for a neuron truly in [1, 2] miss the star's own
    enclosure: they are dropped, the exact image is held (fractions) and the
    result's range query reaches 2. Catches bounds of another node applied
    (the neuron zeroed)."""
    S = Star.from_bounds(np.array([[1.0], [-1.0]]), np.array([[2.0], [1.0]]))
    pre = (np.array([[-5.0], [-1.0]]), np.array([[-4.0], [1.0]]))
    out = _PRE_CONSUMERS[consumer](S, pre)
    _assert_holds_image(_pre_rule(consumer)(_pieces(S)), out,
                        f"{consumer} given bounds of other neurons")
    # and the range query of the result reaches y0 = 2 (a pad of 1e-12 for
    # the centre added to the LP optimum to nearest, SET-5)
    assert max(o.get_ranges()[1][0, 0] for o in out) >= 2.0 - 2e-12


@pytest.mark.parametrize("consumer", sorted(_PRE_CONSUMERS))
def test_nan_precomputed_bound_is_ignored(consumer):
    # x0 = x in [-1, 1] (crossing), x1 = x + 2 (active). A NaN bound for x0
    # made every phase test false, so x0 was kept as the identity and the
    # output lost (f(-1), 1): y0 = -1 came with y1 = 1 instead.
    """A NaN precomputed bound for the crossing neuron x0 = x on [-1, 1],
    beside x1 = x + 2: it is ignored and the exact image held, in fractions;
    the result's own membership query (``Star.contains``) agrees on four of
    its points. Catches a NaN that makes every phase test false, so x0 is
    kept as the identity and (f(-1), 1) is lost, and a membership query that
    calls a reachable output unreachable (the outputs have entries of both
    signs, so ``y - c`` does too)."""
    S = _x_star(-1.0, 1.0).affine_map(np.array([[1.0], [1.0]]),
                                      np.array([[0.0], [2.0]]))
    pre = (np.array([[np.nan], [1.0]]), np.array([[np.nan], [3.0]]))
    out = _PRE_CONSUMERS[consumer](S, pre)
    _assert_holds_image(_pre_rule(consumer)(_pieces(S)), out,
                        f"{consumer} given a NaN bound")
    # true outputs (f(x), x + 2), members by the exact check above (the
    # slope 0.1 times -1 and -1/2 is a float)
    gamma = 0.1 if consumer.startswith("leaky") else 0.0
    _query_holds_each(out, [[x if x >= 0 else gamma * x, x + 2.0]
                            for x in (-1.0, -0.5, 0.0, 1.0)],
                      f"{consumer} given a NaN bound")


# ---------------------------------------------------------------------------
# A neuron that crosses zero by a tiny margin, through every kernel
# ---------------------------------------------------------------------------
#
# The cross-cut of this file. Entry 0 of the input crosses zero by a margin
# chosen here; entries 1 = 4 + a0 and 2 = 4 + a1 are stably active and pin
# the predicate point, so an entry 0 given the wrong phase moves the joint
# output off the exact image even where its own interval still covers the
# value (relu(x) = 0 against x = -eps). The margins:
#
# - "2^-70": entry 0 = +-0.5 + 0.5 a0 + 2^-70 a1, range [-2^-70, 1 + 2^-70]
#   ("min"), [-1 - 2^-70, 2^-70] ("max") or [-2^-70, 2^-70] ("both"). The
#   margin is below every tolerance a phase test could plausibly carry and
#   below one unit in the last place of the other terms, so a bound that
#   adds the generators before the centre rounds onto 0.
# - "rounding": entry 0 = +-0.7999999999999999 + 0.1 a0 + 0.7 a1, whose end
#   is -2.8e-17 ("min") or +2.8e-17 ("max") in exact arithmetic and exactly 0
#   when the three terms are summed to nearest. It tests the bound source
#   (``Star.estimate_ranges``, ``Zono.get_bounds``), which every kernel of a
#   lane reads the same way, so it runs on one consumer per lane and method.
#   ReLU and LeakyReLU on a Zono are left out here: their relaxations round
#   to nearest (ROUND-10, ROUND-16, marked in the ReLU and LeakyReLU cells)
#   and the exact check would see that, not a phase.
# - "2^-40", the LeakyReLU Zono cell alone: its band slope and centre are
#   rounded to nearest (ROUND-16), which no data with a margin far below
#   one can avoid, so this cell grows every returned zonotope by 2^-48 (16
#   units in the last place of 1, a bound on the rounding of the band
#   arithmetic on data of magnitude at most 1) and takes a margin of 2^-40,
#   so that a wrong phase moves the output by (1 - 0.25) 2^-40, 192 times
#   that slack.
#
# Box cells store the tiny end of entry 0 as a bound (c +- |g| would round it
# away). ReLU on a Box makes no phase decision (``np.maximum`` of the
# bounds) and has no cell; Hexatope and Octatope ReLU split pieces lose up to
# 1e-7 whatever the margin (HEXOCT-4, marked in
# test_soundness_range_rounding.py), so their tiny crossing is that defect.

_RTN = 0.7999999999999999      # 0.1 + 0.7 rounded to nearest


def _sign():
    return OnnxFunction("Sign")


# name: (layer factory, set type, method, options, exact rule)
_PHASE_CELLS = {
    "ReLU-Star-exact": (nn.ReLU, "Star", "exact", {}, R.relu),
    "ReLU-Star-approx": (nn.ReLU, "Star", "approx", {}, R.relu),
    "ReLU-Star-approx-range": (nn.ReLU, "Star", "approx", {
        "relax_method": "range", "relax_factor": 0.25}, R.relu),
    "ReLU-Star-approx-area": (nn.ReLU, "Star", "approx", {
        "relax_method": "area", "relax_factor": 0.25}, R.relu),
    "ReLU-Star-approx-bound": (nn.ReLU, "Star", "approx", {
        "relax_method": "bound", "relax_factor": 0.25}, R.relu),
    "LeakyReLU-Star-exact": (lambda: nn.LeakyReLU(0.25), "Star", "exact", {},
                             lambda p: R.leaky_relu(p, 0.25)),
    "LeakyReLU-Star-approx": (lambda: nn.LeakyReLU(0.25), "Star", "approx",
                              {}, lambda p: R.leaky_relu(p, 0.25)),
    "Sign-Star-exact": (_sign, "Star", "exact", {}, R.sign),
    "Sign-Star-approx": (_sign, "Star", "approx", {}, R.sign),
    "ReLU-Zono": (nn.ReLU, "Zono", "approx", {}, R.relu),
    "LeakyReLU-Zono": (lambda: nn.LeakyReLU(0.25), "Zono", "approx", {},
                       lambda p: R.leaky_relu(p, 0.25)),
    "Sign-Zono": (_sign, "Zono", "approx", {}, R.sign),
    "LeakyReLU-Box": (lambda: nn.LeakyReLU(-0.5), "Box", "approx", {},
                      lambda p: R.leaky_relu(p, -0.5)),
    "Sign-Box": (_sign, "Box", "approx", {}, R.sign),
}

_PHASE_GRID = (
    [(cell, side, "2^-70") for cell in _PHASE_CELLS
     if cell != "LeakyReLU-Zono" for side in ("min", "max")]
    # a thin neuron: the split of the exact rules, the dead-unit test of Sign
    + [(cell, "both", "2^-70") for cell in (
        "ReLU-Star-exact", "LeakyReLU-Star-exact", "Sign-Star-exact",
        "Sign-Star-approx", "Sign-Zono", "Sign-Box")]
    + [(cell, side, "rounding") for cell in (
        "ReLU-Star-exact", "ReLU-Star-approx", "Sign-Star-exact",
        "Sign-Star-approx", "Sign-Zono") for side in ("min", "max")]
    + [("LeakyReLU-Zono", side, "2^-40") for side in ("min", "max")])


def _phase_input(kind, side, margin):
    """The input set of a cell of ``_PHASE_GRID`` (see the banner)."""
    if margin == "rounding":
        c0, g0 = (_RTN if side == "min" else -_RTN), [0.1, 0.7]
    else:
        eps = 2.0 ** -70 if margin == "2^-70" else 2.0 ** -40
        c0, g0 = {"min": (0.5, [0.5, eps]), "max": (-0.5, [0.5, eps]),
                  "both": (0.0, [0.0, eps])}[side]
    c = np.array([c0, 4.0, 4.0])
    G = np.array([g0, [1.0, 0.0], [0.0, 1.0]])
    if kind == "Star":
        return Star(np.hstack([c[:, None], G]),
                    np.vstack([np.eye(2), -np.eye(2)]), np.ones((4, 1)),
                    -np.ones((2, 1)), np.ones((2, 1)))
    if kind == "Zono":
        return Zono(c.reshape(-1, 1), G)
    lo0, hi0 = {"min": (-eps, 1.0), "max": (-1.0, eps),
                "both": (-eps, eps)}[side]
    return Box(np.array([lo0, 3.0, 3.0]), np.array([hi0, 5.0, 5.0]))


@pytest.mark.parametrize("cell,side,margin", [
    pytest.param(*p, id="-".join(p)) for p in _PHASE_GRID])
def test_a_neuron_that_crosses_zero_by_a_tiny_margin_keeps_both_phases(
        cell, side, margin):
    """Every kernel that decides a neuron's phase, on its set type and
    method, through ``reach_layer``: entry 0 crosses zero by the margin of the
    banner, and every piece of the exact image must have its vertices in one
    returned set, in fractions with no slack (2^-48 for the LeakyReLU Zono
    cell, see the banner).

    Catches a phase test that carries a tolerance the unsafe way (``lb <
    -1e-12`` for crossing, ``ub <= 1e-12`` for inactive, ``abs(lb) <= 1e-12``
    for a bound of 0, a dead unit taken from bounds near 0, a -1 child or
    lower value of Sign made only below -1e-12), on the interval estimate of
    the exact and approximate Star rules, the zonotope bounds and the Box
    bounds, and a bound source that sums to nearest so that the end of the
    range lands on 0 ("rounding"): each fixes entry 0 in one phase, and the
    output of the other phase escapes (relu(x) = 0 or slope x against the
    identity x at the minimum, x against 0 at the maximum, the sign -1 or +1
    lost)."""
    make, kind, method, opts, rule = _PHASE_CELLS[cell]
    input_set = _phase_input(kind, side, margin)
    what = f"{cell}, entry 0 crossing zero at its {side} by {margin}"
    out = reach_one_layer(make(), [input_set], method, what, **opts)
    _assert_holds_image(rule(_pieces(input_set)), out, what,
                        slack=2.0 ** -48 if cell == "LeakyReLU-Zono" else 0.0)


# Linear then the activation: the same input through the model entry.
_W_TINY = np.array([[0.5, 2.0 ** -70], [1.0, 0.0], [0.0, 1.0]])
_B_TINY = np.array([0.5, 4.0, 4.0])


@pytest.mark.parametrize("op,method", [
    (op, method) for op in ("ReLU", "LeakyReLU", "Sign")
    for method in ("exact", "approx")])
def test_a_tiny_crossing_through_the_model_entry_keeps_both_phases(
        tmp_path, op, method):
    """Linear (entry 0 = 0.5 + 0.5 x0 + 2^-70 x1 on [-1, 1]^2, entries 1 and
    2 pin x) then the activation, reached through ``reach_pytorch_model`` on
    a Star (a torch Sequential; Sign as the module ``load_onnx`` builds for an
    ONNX Gemm and Sign, every number dyadic and a float32). Every piece of
    the exact image must be in one returned set, in fractions. Catches the
    phase errors of the cell above on the public route, where the walk hands
    the kernel the set the Linear made."""
    if op == "Sign":
        f32 = lambda a: np.asarray(a, dtype=np.float32)      # noqa: E731
        model = load_onnx(save_onnx(
            tmp_path, [helper.make_node("Gemm", ["X", "W", "B"], ["H"],
                                        transB=1),
                       helper.make_node("Sign", ["H"], ["Y"])],
            [1, 2], [1, 3], {"W": f32(_W_TINY), "B": f32(_B_TINY)},
            name="tiny_sign"))
        rule = R.sign
    else:
        act = nn.ReLU() if op == "ReLU" else nn.LeakyReLU(0.25)
        model = nn.Sequential(nn.Linear(2, 3), act).double()
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor(_W_TINY))
            model[0].bias.copy_(torch.tensor(_B_TINY))
        rule = R.relu if op == "ReLU" else (lambda p: R.leaky_relu(p, 0.25))
    input_set = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    what = f"Linear then {op}, {method}, through the model entry"
    out = reach_model(model, input_set, method, what)
    pieces = rule(R.affine(_pieces(input_set), _W_TINY, _B_TINY))
    _assert_holds_image(pieces, out, what)
