"""Soundness cells of the certified LP bound and of the outward-rounded bound
arithmetic: ``n2v/utils/ns_certificate.py`` (the Neumaier-Shcherbina
certificate ``ns_bound`` / ``ns_bounds_population``, the interval enclosures
``outward_interval_bounds`` / ``outward_interval_slack``, the spec pull-back
``outward_spec_rows``, and the directed-rounding helpers ``_add_dir``,
``_mul_dir``, ``_div_dir``, ``outward_center_radius``, ``_widen_sum``,
``_widen_relative``) and the certified GPU LP entries of
``n2v/utils/lpsolver_gpu.py`` (run on CPU torch, ``device='cpu'``).

The promise checked everywhere: a value returned as a bound holds the exact
value it bounds, compared one-sided with zero slack in ``fractions.Fraction``
of the floats as given.

- An LP bound holds the exact optimum, found by enumerating the vertices of
  the feasible polytope in rationals (``tests/_oracles/rational.py``), for
  every multiplier vector handed in: certificate-tight, zero, random,
  negative, non-finite, huge and tiny ones.
- A directed-rounding helper returns a float at or beyond the exact result
  in its direction, over values that round, overflow, underflow, signed
  zeros and infinities, and keeps a result exact where its docstring says
  it is exact.
- An interval enclosure holds the exact range over the box; a slack or a
  pulled-back right-hand side makes the caller's round-to-nearest value a
  rigorous one.

Each helper is also reached through a public entry: ``Box`` arithmetic and
``to_zono``, a model reach of constant Mul, Div, ReduceSum, Sigmoid, Tanh
and LeakyReLU, ``Star.estimate_ranges``, ``Zono.get_bounds``,
``Star.intersect_half_space``, ``verify_specification`` and the refinement
engine's ``neuron_bounds``, ``contract_input_boxes`` and
``certified_prune_wave`` on CPU torch.
"""

import math
import sys
from fractions import Fraction as F
from functools import lru_cache

import numpy as np
import pytest

from n2v.utils.ns_certificate import (
    _add_dir, _div_dir, _mul_dir, _widen_relative, _widen_sum, ns_bound,
    ns_bounds_population, outward_center_radius, outward_interval_bounds,
    outward_interval_slack, outward_spec_rows)
from tests._oracles.rational import vertices
from tests.known_bugs import known_bug

MAX = sys.float_info.max
ETA = 5e-324


# --------------------------------------------------------------------------- #
# Exact references                                                              #
# --------------------------------------------------------------------------- #
def q(x):
    """The rational a finite float is."""
    return F(float(x))


@lru_cache(maxsize=None)
def _vertex_list(key):
    A, b, lb, ub = key
    n = len(lb)
    rows = [[q(v) for v in r] for r in A]
    rhs = [q(v) for v in b]
    for j in range(n):
        e = [F(0)] * n
        e[j] = F(1)
        rows.append(e)
        rhs.append(q(ub[j]))
        rows.append([-v for v in e])
        rhs.append(-q(lb[j]))
    return vertices(rows, rhs)


def exact_opt(c, A, b, lb, ub, minimize):
    """The exact optimum of min/max c'x over {A x <= b, lb <= x <= ub} (a
    finite box), the best value over the vertices of the polytope."""
    A = [] if A is None else [tuple(float(v) for v in r) for r in np.atleast_2d(A)]
    b = [] if b is None else [float(v) for v in np.ravel(b)]
    key = (tuple(A), tuple(b), tuple(float(v) for v in np.ravel(lb)),
           tuple(float(v) for v in np.ravel(ub)))
    vs = _vertex_list(key)
    if not vs:
        pytest.fail("test setup: the LP is infeasible")
    vals = [sum(q(ci) * vi for ci, vi in zip(np.ravel(c), v)) for v in vs]
    return min(vals) if minimize else max(vals)


def holds(bound, exact, lower):
    """``bound`` is a valid lower (upper) bound of the rational ``exact``:
    not NaN, an infinity only on the side that claims nothing, else compared
    in rationals with zero slack."""
    bound = float(bound)
    if math.isnan(bound):
        return False
    if math.isinf(bound):
        return bound < 0 if lower else bound > 0
    return q(bound) <= exact if lower else q(bound) >= exact


def ns_exact(c, y, A, b, lb, ub):
    """The exact Neumaier-Shcherbina value b'y + sum_j max(r_j lb_j, r_j ub_j),
    r = c - A'y, of the multipliers y clamped to >= 0 (every value in
    rationals; the weak-duality upper bound of max c'x the certificate
    rounds outward)."""
    yq = [max(q(v), F(0)) for v in np.ravel(y)]
    A = np.atleast_2d(A)
    r = [q(c[j]) - sum(q(A[i, j]) * yq[i] for i in range(len(yq)))
         for j in range(len(c))]
    return (sum(q(bi) * yi for bi, yi in zip(np.ravel(b), yq))
            + sum(max(rj * q(l), rj * q(u)) for rj, l, u in zip(r, lb, ub)))


# --------------------------------------------------------------------------- #
# LP fixtures                                                                   #
# --------------------------------------------------------------------------- #
def _tight_lp(seed, n, m, dyadic=False):
    """An LP over a non-unit box whose optimum a known multiplier vector
    certifies up to rounding: rows tight at a box point x*, c = A'y* rounded
    (so x* is optimal and the certificate of y* is razor-tight there, where an
    inward rounding of the certificate shows). Data far from +-1, both signs;
    one row slack with multiplier 0 when m > 1."""
    rng = np.random.default_rng(seed)
    scale = 4.0 if dyadic else 1.0
    lb = np.round(rng.uniform(-9.0, -0.5, n) * scale) / scale
    ub = lb + np.round(rng.uniform(1.0, 12.0, n) * scale) / scale
    if not dyadic:
        lb = lb + rng.uniform(-0.3, 0.3, n)
        ub = ub + rng.uniform(-0.3, 0.3, n)
    x = lb + (ub - lb) * (rng.integers(1, 4, n) / 4.0)
    A = np.round(rng.uniform(-3.0, 3.0, (m, n)) * 8.0) / 8.0
    if not dyadic:
        A = A + rng.uniform(-0.05, 0.05, (m, n))
    b = A @ x
    y = np.round(rng.uniform(0.25, 2.0, m) * 4.0) / 4.0
    if not dyadic:
        y = y + rng.uniform(-0.1, 0.1, m)
    if m > 1:
        b[-1] += 1.375
        y[-1] = 0.0
    c = A.T @ y
    return c, A, b, lb, ub, y


TIGHT_LPS = ([("dyadic", s, n, m) for s, (n, m) in enumerate([(1, 1), (2, 1), (2, 3), (3, 2)])]
             + [("float", 10 + s, n, m) for s, (n, m) in enumerate([(1, 2), (2, 2), (3, 3), (3, 1)])])


def _lp(spec):
    kind, seed, n, m = spec
    return _tight_lp(seed, n, m, dyadic=(kind == "dyadic"))


def _multipliers(kind, y_star, rng):
    m = y_star.size
    if kind == "certificate":
        return y_star.copy()
    if kind == "zero":
        return np.zeros(m)
    if kind == "random":
        return rng.uniform(0.0, 3.0, m)
    if kind == "negative":
        return rng.uniform(-3.0, 3.0, m) - 1.0
    if kind == "nonfinite":
        return np.array([[np.nan, np.inf, -np.inf][i % 3] for i in range(m)])
    if kind == "huge":
        return rng.uniform(0.5, 2.0, m) * 1e200
    if kind == "tiny":
        return rng.uniform(0.5, 2.0, m) * 1e-300
    raise ValueError(kind)


MULTIPLIER_KINDS = ["certificate", "zero", "random", "negative", "nonfinite",
                    "huge", "tiny"]


# --------------------------------------------------------------------------- #
# ns_bound                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", MULTIPLIER_KINDS)
def test_ns_bound_holds_the_exact_optimum_for_any_multipliers(kind):
    """``ns_bound`` is an upper bound of the exact maximum and a lower bound
    of the exact minimum for every multiplier vector, in rationals with zero
    slack. The certificate-tight multipliers put the computed value within
    rounding of the optimum (catches an inflation applied inward, an endpoint
    chosen for the wrong sign of the reduced cost, the minimisation reduced
    with the wrong sign); negative ones must be clamped and non-finite ones
    zeroed (weak duality needs y >= 0)."""
    rng = np.random.default_rng(7)
    bad = []
    for spec in TIGHT_LPS:
        c, A, b, lb, ub, y_star = _lp(spec)
        y = _multipliers(kind, y_star, rng)
        for minimize, cc in ((False, c), (True, -c)):
            bound = ns_bound(cc, y, A=A, b=b, lb=lb, ub=ub, minimize=minimize)
            exact = exact_opt(cc, A, b, lb, ub, minimize)
            if not holds(bound, exact, lower=minimize):
                bad.append((spec, minimize, float(bound), float(exact)))
    assert not bad, bad


def test_ns_bound_with_exactly_optimal_multipliers_still_moves_outward():
    """Dyadic LPs whose certificate with the exact optimal multipliers equals
    the optimum in exact arithmetic (c = A'y* exactly, every row with y* > 0
    tight at an optimal box point): the returned bound must still be on the
    outer side, at or beyond the optimum, never inside it by its own rounding
    allowance (an inflation subtracted instead of added)."""
    cases = [
        # max x1 + x2 s.t. x1 + x2 <= 1, x in [0, 1]^2: optimum 1, y* = 1
        (np.array([1.0, 1.0]), np.array([[1.0, 1.0]]), np.array([1.0]),
         np.zeros(2), np.ones(2), np.array([1.0])),
        # max 3x1 - x2 s.t. 3x1 - x2 <= 0.75, -x2 <= 2.5, box [-4, 6] x [-3, 5]
        (np.array([3.0, -1.0]), np.array([[3.0, -1.0], [0.0, -1.0]]),
         np.array([0.75, 2.5]), np.array([-4.0, -3.0]), np.array([6.0, 5.0]),
         np.array([1.0, 0.0])),
        # max 0.5 x s.t. 0.25 x <= 1.5 (y* = 2), x in [-10, 20]: optimum 3
        (np.array([0.5]), np.array([[0.25]]), np.array([1.5]),
         np.array([-10.0]), np.array([20.0]), np.array([2.0])),
    ]
    for c, A, b, lb, ub, y in cases:
        assert ns_exact(c, y, A, b, lb, ub) == exact_opt(c, A, b, lb, ub, False)
        for minimize, cc in ((False, c), (True, -c)):
            bound = ns_bound(cc, y, A=A, b=b, lb=lb, ub=ub, minimize=minimize)
            exact = exact_opt(cc, A, b, lb, ub, minimize)
            assert holds(bound, exact, lower=minimize), (c, minimize, bound, exact)


@pytest.mark.parametrize("rows", ["none", "empty", "A-without-b"])
def test_ns_bound_box_only_holds_the_exact_box_optimum(rows):
    """With no general rows (``A``/``b`` None, an empty ``b``, or ``A``
    without ``b``) the certificate is the box bound: per coordinate the
    endpoint that maximises c_j x_j. Coefficients of both signs and a zero
    one, endpoints far from +-1 and not dyadic; checked both senses in
    rationals with zero slack (catches the endpoint chosen for the wrong
    sign and the minimisation reduced with the wrong sign)."""
    c = np.array([2.5, -0.3, 0.0, 1.1])
    lb = np.array([-7.3, 0.2, -1.0, -0.1])
    ub = np.array([4.1, 9.7, 3.0, 0.3])
    A, b = {"none": (None, None), "empty": (np.zeros((0, 4)), np.zeros(0)),
            "A-without-b": (np.ones((1, 4)), None)}[rows]
    for minimize in (False, True):
        bound = ns_bound(c, np.zeros(0), A=A, b=b, lb=lb, ub=ub,
                         minimize=minimize)
        exact = exact_opt(c, None, None, lb, ub, minimize)
        assert holds(bound, exact, lower=minimize), (minimize, bound, exact)


def test_ns_bound_on_an_infinite_box_claims_nothing_wrong():
    """``lb``/``ub`` left out stand for an unbounded box, and an infinite
    entry may sit beside a finite one. The bound must still hold the exact
    optimum and never be NaN: a coordinate whose reduced cost and its error
    are exactly 0 contributes 0 (not 0 * inf = NaN), and one whose reduced
    cost may be nonzero makes the bound infinite on the side that claims
    nothing. Catches a dropped 0 * inf mask in the box term or in its error
    term."""
    A = np.array([[1.0, 0.0]])
    b = np.array([0.75])
    # (numpy warns about the 0 * inf it masks afterwards; the value counts)
    with np.errstate(invalid="ignore"):
        # max x1 s.t. x1 <= 0.75, x2 free (coefficient 0): optimum 0.75
        for lb, ub in ((None, None),
                       (np.array([-np.inf, -2.0]), np.array([np.inf, 3.0])),
                       (np.array([-5.0, -np.inf]), np.array([5.0, np.inf]))):
            for y in (np.array([1.0]), np.array([0.5]), np.array([0.0])):
                bound = ns_bound(np.array([1.0, 0.0]), y, A=A, b=b, lb=lb,
                                 ub=ub, minimize=False)
                assert holds(bound, F(3, 4), lower=False), (lb, y, bound)
        # min x1 over x1 <= 0.75 with x1 unbounded below: only -inf holds
        bound = ns_bound(np.array([1.0, 0.0]), np.array([0.5]), A=A, b=b,
                         minimize=True)
        assert bound == -np.inf
        # box only, the infinite coordinate has coefficient exactly 0: the
        # bound is the finite box bound of the other coordinate (docstring:
        # "the product must be 0, not inf * 0 = NaN")
        lb = np.array([-1.3, -np.inf])
        ub = np.array([2.7, np.inf])
        for minimize in (False, True):
            bound = ns_bound(np.array([-1.5, 0.0]), np.zeros(0), lb=lb, ub=ub,
                             minimize=minimize)
            exact = q(-1.5) * (q(-1.3) if not minimize else q(2.7))
            assert math.isfinite(bound) and holds(bound, exact, lower=minimize), bound


# max x1 + 0.5 x2 s.t. -x1 - 0.5 x2 <= -0.25 over [0, 1.5] x [-1, 2]: the row
# is slack at the optimum 2.5; the multiplier -1 cancels the objective
# (reduced cost 0), so an unclamped certificate is b'y = 0.25 < 2.5.
_SLACK_ROW_LP = (np.array([1.0, 0.5]), np.array([[-1.0, -0.5]]), np.array([-0.25]),
                 np.array([0.0, -1.0]), np.array([1.5, 2.0]))


@pytest.mark.parametrize("y", [-1.0, -0.4, -np.inf, np.nan])
def test_certificates_do_not_use_a_negative_or_non_finite_multiplier(y):
    """Weak duality needs y >= 0: a negative multiplier on a row that is
    slack at the optimum, chosen so that it cancels the objective, gives a
    value below the exact maximum unless it is clamped to 0, and a non-finite
    one must be zeroed. Both ``ns_bound`` and ``ns_bounds_population`` must
    hold the exact optimum (2.5 for the maximum, 0.25 is what the unclamped
    certificate says)."""
    c, A, b, lb, ub = _SLACK_ROW_LP
    yv = np.array([y])
    for minimize, cc in ((False, c), (True, -c)):
        exact = exact_opt(cc, A, b, lb, ub, minimize)
        bound = ns_bound(cc, yv, A=A, b=b, lb=lb, ub=ub, minimize=minimize)
        assert holds(bound, exact, lower=minimize), ("scalar", minimize, bound)
    C = np.stack([c, -c], axis=1)[None]
    Y = np.full((1, 1, 2), y)
    r = ns_bounds_population(C, Y, A[None], b[None], lb[None], ub[None], [False, True])
    for j, minimize in enumerate((False, True)):
        exact = exact_opt(C[0, :, j], A, b, lb, ub, minimize)
        assert holds(r[0, j], exact, lower=minimize), ("population", minimize, r[0, j])


# The certificate is evaluated in round-to-nearest float64 and its rounding
# allowance is relative to the magnitudes of the terms. A product that
# underflows carries an absolute error (up to half the smallest subnormal)
# that no relative term bounds, and a product that overflows or meets an
# infinite right-hand side with multiplier 0 makes the value NaN.
_NAN_DEFECT = ("the certificate returns NaN when a product inside it is not "
               "finite (multipliers whose products overflow, or a right-hand "
               "side of +inf with multiplier 0); NaN holds no optimum and a "
               "NaN neuron bound reads as stably active")
_UNDERFLOW_DEFECT = ("the certificate's rounding allowance is relative only: a "
                     "product that underflows to a subnormal or to zero carries an "
                     "absolute error it does not cover, so the bound lies inside "
                     "the exact optimum by a subnormal amount")

# (c, y, A, b, lb, ub): an LP whose certificate meets a non-finite product
_NAN_CASES = {
    "rhs-inf-multiplier-zero": (
        np.array([1.0, 0.5]), np.array([1.0, 0.0]),
        np.array([[1.0, 1.0], [1.0, 0.0]]), np.array([0.5, np.inf]),
        -np.ones(2), np.ones(2)),
    "multiplier-products-overflow": (
        np.array([1.0, -0.5]), np.array([1e308, 0.0]),
        np.array([[1.0, 2.0], [-3.0, 1.0]]), np.array([1.0, 2.0]),
        np.array([-1.0, 0.5]), np.array([2.0, 1.5])),
    # neighbours on the same rows: a huge multiplier whose products stay
    # finite, and an infinite multiplier (zeroed by design)
    "multiplier-huge-but-finite": (
        np.array([1.0, -0.5]), np.array([1e300, 1e300]),
        np.array([[1.0, 2.0], [-3.0, 1.0]]), np.array([1.0, 2.0]),
        np.array([-1.0, 0.5]), np.array([2.0, 1.5])),
    "multiplier-infinite": (
        np.array([1.0, -0.5]), np.array([np.inf, 1.0]),
        np.array([[1.0, 2.0], [-3.0, 1.0]]), np.array([1.0, 2.0]),
        np.array([-1.0, 0.5]), np.array([2.0, 1.5])),
}


def _exact_opt_finite_rows(c, A, b, lb, ub, minimize):
    keep = np.isfinite(b)
    return exact_opt(c, A[keep], b[keep], lb, ub, minimize)


@pytest.mark.parametrize("case", [
    pytest.param("rhs-inf-multiplier-zero", marks=known_bug(17, _NAN_DEFECT)),
    pytest.param("multiplier-products-overflow", marks=known_bug(17, _NAN_DEFECT)),
    "multiplier-huge-but-finite",
    "multiplier-infinite",
])
def test_ns_bound_is_a_bound_when_a_product_is_not_finite(case):
    """``ns_bound`` returns a value that holds the exact optimum (an infinity
    on the side that claims nothing is a valid answer) and never NaN, also
    when a product of its sums is not finite. A row with right-hand side +inf
    is vacuous (Star.convex_hull documents such rows)."""
    c, y, A, b, lb, ub = _NAN_CASES[case]
    for minimize, cc in ((False, c), (True, -c)):
        with np.errstate(over="ignore", invalid="ignore"):
            bound = ns_bound(cc, y, A=A, b=b, lb=lb, ub=ub, minimize=minimize)
        exact = _exact_opt_finite_rows(cc, A, b, lb, ub, minimize)
        assert holds(bound, exact, lower=minimize), (minimize, bound, float(exact))


_TWO = F(2)
# (c, A, b, lb, ub, exact max, exact min)
_UNDERFLOW_CASES = {
    # max 2^-600 x over [0, 2^-600] is 2^-1200: the box term underflows to 0
    "box-term": (np.array([2.0 ** -600]), None, None, np.array([0.0]),
                 np.array([2.0 ** -600])),
    # c = 0, one row 2^-600 x <= -2^-561 over [-2^40, 2^40], multiplier
    # 2^-476: A'y = 2^-1076 underflows to 0 while b'y = -2^-1037 does not
    "row-term": (np.array([0.0]), np.array([[2.0 ** -600]]),
                 np.array([-(2.0 ** -561)]), np.array([-(2.0 ** 40)]),
                 np.array([2.0 ** 40])),
    # neighbour: the box-term case scaled into the normal range
    "box-term-normal-range": (np.array([2.0 ** -300]), None, None,
                              np.array([0.0]), np.array([2.0 ** -300])),
}
_UNDERFLOW_Y = {"box-term": np.zeros(0), "row-term": np.array([2.0 ** -476]),
                "box-term-normal-range": np.zeros(0)}


@pytest.mark.parametrize("case", [
    pytest.param("box-term", marks=known_bug(17, _UNDERFLOW_DEFECT)),
    pytest.param("row-term", marks=known_bug(17, _UNDERFLOW_DEFECT)),
    "box-term-normal-range",
])
def test_ns_bound_covers_products_that_underflow(case):
    """The certificate holds the exact optimum when its products underflow
    into the subnormal range or to zero (the case ``outward_interval_bounds``
    covers with its half-subnormal term per product). Both senses, zero
    slack in rationals."""
    c, A, b, lb, ub = _UNDERFLOW_CASES[case]
    y = _UNDERFLOW_Y[case]
    for minimize, cc in ((False, c), (True, -c)):
        bound = ns_bound(cc, y, A=A, b=b, lb=lb, ub=ub, minimize=minimize)
        exact = exact_opt(cc, A, b, lb, ub, minimize)
        assert holds(bound, exact, lower=minimize), (minimize, bound, float(exact))


# --------------------------------------------------------------------------- #
# ns_bounds_population                                                          #
# --------------------------------------------------------------------------- #
def _population(specs, kinds, seed=3, pad_rows=2):
    """Stack LPs of different sizes into one population: zero-padded rows
    (with arbitrary positive multipliers on them) and zero-width padded
    variables. Objective column j of every star is a maximisation of
    c (flag False) or a minimisation of -c (flag True), alternating and
    starting with a minimisation, each with its own multiplier kind."""
    rng = np.random.default_rng(seed)
    lps = [_lp(s) for s in specs]
    B = len(lps)
    n = max(lp[0].size for lp in lps)
    m = max(lp[1].shape[0] for lp in lps) + pad_rows
    k = len(kinds)
    A = np.zeros((B, m, n))
    b = np.zeros((B, m))
    lb = np.zeros((B, n))
    ub = np.zeros((B, n))
    C = np.zeros((B, n, k))
    Y = np.zeros((B, m, k))
    flags = [j % 2 == 0 for j in range(k)]
    for s, (c, As, bs, l, u, y_star) in enumerate(lps):
        ms, ns_ = As.shape
        A[s, :ms, :ns_] = As
        b[s, :ms] = bs
        lb[s, :ns_] = l
        ub[s, :ns_] = u
        for j, kind in enumerate(kinds):
            C[s, :ns_, j] = -c if flags[j] else c
            Y[s, :ms, j] = _multipliers(kind, y_star, rng)
            Y[s, ms:, j] = rng.uniform(0.0, 5.0, m - ms)
    return lps, A, b, lb, ub, C, Y, flags


def test_ns_bounds_population_holds_every_exact_optimum():
    """Every (star, objective) entry of ``ns_bounds_population`` holds the
    exact optimum of its own LP: stars with their own rows and boxes (a
    swapped or shared box shows), mixed minimise flags starting with a
    minimisation (a sign not flipped back shows), zero-padded rows carrying
    arbitrary multipliers, and multipliers of every kind."""
    specs = TIGHT_LPS
    kinds = ["certificate", "certificate", "random", "negative", "nonfinite",
             "zero", "tiny", "huge"]
    lps, A, b, lb, ub, C, Y, flags = _population(specs, kinds)
    r = ns_bounds_population(C, Y, A, b, lb, ub, flags)
    assert r.shape == (len(specs), len(kinds))
    bad = []
    for s, (c, As, bs, l, u, _) in enumerate(lps):
        for j in range(len(kinds)):
            cc = -c if flags[j] else c
            exact = exact_opt(cc, As, bs, l, u, flags[j])
            if not holds(r[s, j], exact, lower=flags[j]):
                bad.append((s, j, float(r[s, j]), float(exact)))
    assert not bad, bad


def test_ns_bounds_population_without_rows_is_the_exact_box_bound():
    """With no rows at all (``m = 0``) every entry is the box bound of its
    star, at or beyond the exact box optimum in rationals, both senses."""
    rng = np.random.default_rng(11)
    B, n, k = 3, 3, 4
    lb = rng.uniform(-8.0, 1.0, (B, n))
    ub = lb + rng.uniform(0.5, 9.0, (B, n))
    C = rng.uniform(-3.0, 3.0, (B, n, k))
    flags = [True, False, False, True]
    r = ns_bounds_population(C, np.zeros((B, 0, k)), np.zeros((B, 0, n)),
                             np.zeros((B, 0)), lb, ub, flags)
    for s in range(B):
        for j in range(k):
            exact = exact_opt(C[s, :, j], None, None, lb[s], ub[s], flags[j])
            assert holds(r[s, j], exact, lower=flags[j]), (s, j, r[s, j])


@pytest.mark.parametrize("case", [
    pytest.param("multiplier-products-overflow", marks=known_bug(17, _NAN_DEFECT)),
    "multiplier-huge-but-finite",
])
def test_ns_bounds_population_is_a_bound_when_a_product_is_not_finite(case):
    """The vectorised certificate never returns NaN: a population entry whose
    multiplier products overflow must still be a bound (an infinity on the
    side that claims nothing is one)."""
    c, y, A, b, lb, ub = _NAN_CASES[case]
    C = np.stack([c, -c], axis=1)[None]
    Y = np.stack([y, y], axis=1)[None]
    flags = [False, True]
    with np.errstate(over="ignore", invalid="ignore"):
        r = ns_bounds_population(C, Y, A[None], b[None], lb[None], ub[None], flags)
    for j in range(2):
        exact = exact_opt(C[0, :, j], A, b, lb, ub, flags[j])
        assert holds(r[0, j], exact, lower=flags[j]), (j, r[0, j], float(exact))


@pytest.mark.parametrize("case", [
    pytest.param("row-term", marks=known_bug(17, _UNDERFLOW_DEFECT)),
    "box-term-normal-range",
])
def test_ns_bounds_population_covers_products_that_underflow(case):
    """The vectorised certificate holds the exact optimum when its products
    underflow (same LPs as the scalar case)."""
    c, A, b, lb, ub = _UNDERFLOW_CASES[case]
    y = _UNDERFLOW_Y[case]
    m = 0 if A is None else A.shape[0]
    Ap = np.zeros((1, m, 1)) if A is None else A[None]
    bp = np.zeros((1, m)) if b is None else b[None]
    C = np.stack([c, -c], axis=1)[None]
    Y = np.stack([y, y], axis=1)[None] if m else np.zeros((1, 0, 2))
    flags = [False, True]
    r = ns_bounds_population(C, Y, Ap, bp, lb[None], ub[None], flags)
    for j in range(2):
        exact = exact_opt(C[0, :, j], A, b, lb, ub, flags[j])
        assert holds(r[0, j], exact, lower=flags[j]), (j, r[0, j], float(exact))


# --------------------------------------------------------------------------- #
# Directed rounding: _add_dir, _mul_dir, _div_dir                               #
# --------------------------------------------------------------------------- #
# Values that round (0.1 + 0.2, 1 + 2^-60, 1/3), sit near the overflow and
# underflow thresholds, are subnormal, signed zeros or infinite.
_VALS = [0.0, -0.0, 0.1, 0.2, -0.3, 1.0, -1.0, 2.0 ** -60, 1.0 + 2.0 ** -52,
         3.0, 1.0 / 3.0, -7.25, 0.7, 1.5e-160, -1.3e-160, 1e-300, 1e300,
         1.6e308, -1.5e308, MAX, -MAX, ETA, -3 * ETA, 2.2250738585072014e-308,
         math.inf, -math.inf, math.nan]
_PAIRS = [(a, b) for a in _VALS for b in _VALS]
_A = np.array([p[0] for p in _PAIRS])
_B = np.array([p[1] for p in _PAIRS])


def _ext(v):
    """A float as a rational, or the infinity itself."""
    return v if math.isinf(v) else q(v)


def _is_bound(out, exact, d):
    """``out`` lies at or beyond ``exact`` toward ``d * inf``. ``exact`` is a
    rational, an infinity (the exact extended-real result) or None (the
    operation has no value: only ``d * inf`` claims nothing)."""
    out = float(out)
    if math.isnan(out):
        return False
    if exact is None:
        return out == d * math.inf
    if isinstance(exact, float):          # an infinite exact result
        return out == exact or out == d * math.inf
    if math.isinf(out):
        return out * d > 0
    return q(out) <= exact if d < 0 else q(out) >= exact


def _exact_sum(a, b):
    if math.isnan(a) or math.isnan(b):
        return None
    if math.isinf(a) and math.isinf(b) and a != b:
        return None
    if math.isinf(a) or math.isinf(b):
        return a if math.isinf(a) else b
    return q(a) + q(b)


def _exact_product(a, b):
    if math.isnan(a) or math.isnan(b):
        return None
    if (a == 0 and math.isinf(b)) or (b == 0 and math.isinf(a)):
        return F(0)               # docstring: an infinite endpoint stands for unbounded reals
    if math.isinf(a) or math.isinf(b):
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return q(a) * q(b)


def _exact_quotient(a, b):
    if math.isnan(a) or math.isnan(b):
        return None
    if b == 0 or (math.isinf(a) and math.isinf(b)):
        return None
    if math.isinf(a):
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    if math.isinf(b):
        return F(0)
    return q(a) / q(b)


@pytest.mark.parametrize("op", ["add", "mul", "div"])
@pytest.mark.parametrize("d", [-1.0, 1.0])
def test_directed_operation_bounds_the_exact_result(op, d):
    """``_add_dir``, ``_mul_dir`` and ``_div_dir`` return, for every pair of
    the grid, a float at or beyond the exact result in direction ``d``
    (rationals of the operands, zero slack): a sum or product that rounds
    toward the bound moves one ulp, one that overflows keeps the largest
    double on the far side, an underflowing product leaves zero, and an
    operation with no value (inf - inf, a zero divisor of either sign,
    inf / inf, a NaN operand) gives ``d * inf``, never NaN. Catches a TwoSum test that
    ignores the direction, a dropped overflow branch, an exactness test that
    lost its negation, and a factor of +-1 taken as exact in a quotient."""
    fn, ref = {"add": (_add_dir, _exact_sum), "mul": (_mul_dir, _exact_product),
               "div": (_div_dir, _exact_quotient)}[op]
    out = fn(_A, _B, d)
    bad = [(a, b, float(o)) for a, b, o in zip(_A, _B, out)
           if not _is_bound(o, ref(float(a), float(b)), d)]
    assert not bad, bad[:10]


def test_directed_operations_keep_exact_results_exact():
    """Where the docstrings say the one rounding is exact, the result is the
    exact value in both directions: a sum whose TwoSum error is 0, a product
    with a factor 0 or +-1 (and 0 times inf, which is 0), a quotient with a
    zero numerator or a divisor of +-1. A bound moved off an exact value is
    wrong by the documented contract (and turns an exact zero into a
    crossing neuron downstream)."""
    for d in (-1.0, 1.0):
        sums = [(1.0, 2.0), (0.5, 0.25), (-7.25, 0.0), (MAX, -MAX), (ETA, ETA),
                (0.1, 0.0), (1e300, -1e300)]
        for a, b in sums:
            assert float(_add_dir(a, b, d)) == a + b, ("add", a, b, d)
        prods = [(0.1, 1.0), (-1.0, 0.7), (0.0, 1e300), (-3.3, -1.0),
                 (0.0, math.inf), (math.inf, 0.0), (MAX, 1.0)]
        for a, b in prods:
            expect = 0.0 if (a == 0 or b == 0) else a * b
            assert float(_mul_dir(a, b, d)) == expect, ("mul", a, b, d)
        quots = [(0.0, 0.3), (0.7, 1.0), (-0.1, -1.0), (MAX, -1.0)]
        for a, b in quots:
            assert float(_div_dir(a, b, d)) == a / b, ("div", a, b, d)


def test_box_minkowski_sum_holds_every_exact_sum():
    """Public route of ``_add_dir``: ``Box.minkowski_sum`` contains every
    x + y, in rationals: 0.1 + 0.2 rounds up (the lower bound must move
    down), 0.5 + 0.7 rounds down, and [1.5e308, 1.6e308] with itself
    overflows (the lower bound must stay finite at or below 3e308, the upper
    bound may be +inf)."""
    from n2v.sets import Box
    a = Box(np.array([[0.1], [0.5], [1.5e308], [-2.3]]),
            np.array([[0.5], [0.7], [1.6e308], [-1.1]]))
    b = Box(np.array([[0.2], [0.7], [1.5e308], [0.3]]),
            np.array([[0.7], [0.9], [1.6e308], [4.4]]))
    with np.errstate(over="ignore"):
        s = a.minkowski_sum(b)
    for i in range(4):
        lo = _exact_sum(float(a.lb[i, 0]), float(b.lb[i, 0]))
        hi = _exact_sum(float(a.ub[i, 0]), float(b.ub[i, 0]))
        assert _is_bound(s.lb[i, 0], lo, -1.0), (i, s.lb[i, 0])
        assert _is_bound(s.ub[i, 0], hi, 1.0), (i, s.ub[i, 0])
    assert math.isfinite(float(s.lb[2, 0]))


def _const_model(tmp_path, op, k):
    from onnx import helper
    from n2v.nn import NeuralNetwork
    from n2v.utils.model_loader import load_onnx
    from tests.cell_support import save_onnx
    path = save_onnx(tmp_path, [helper.make_node(op, ["X", "k"], ["Y"])],
                     [1, k.size], [1, k.size], inits={"k": k.astype(np.float32)},
                     name=op.lower())
    return NeuralNetwork(load_onnx(path))


@pytest.mark.parametrize("op", ["Mul", "Div"])
def test_constant_mul_and_div_of_a_box_hold_the_exact_image(tmp_path, op):
    """Public route of ``_mul_dir`` and ``_div_dir``: a model reach of an ONNX
    Mul or Div by a constant on a Box. The exact image of each coordinate,
    in rationals of the stored float32 constant and the float64 box, lies
    inside the returned Box with zero slack. Constants above and below 1,
    both signs, 1/3 that rounds; numerators of 1 (a numerator of +-1 is not
    exact in a quotient)."""
    from n2v.sets import Box
    k = np.array([3.0, -0.7, 0.3, -1.5, 3.0], dtype=np.float32)
    lb = np.array([0.1, -2.3, 1.0, -0.1, 0.5])
    ub = np.array([0.2, 4.1, 1.0, 0.3, 1.0])
    net = _const_model(tmp_path, op, k)
    out = net.reach(Box(lb.reshape(-1, 1), ub.reshape(-1, 1)), method="approx")
    assert len(out) == 1
    lo, hi = np.ravel(out[0].lb), np.ravel(out[0].ub)
    assert lo.size == k.size
    for i in range(k.size):
        kq = q(k[i])
        ends = [q(lb[i]) * kq, q(ub[i]) * kq] if op == "Mul" else [q(lb[i]) / kq, q(ub[i]) / kq]
        assert q(lo[i]) <= min(ends) and q(hi[i]) >= max(ends), (op, i, lo[i], hi[i])


# --------------------------------------------------------------------------- #
# outward_center_radius                                                         #
# --------------------------------------------------------------------------- #
_INTERVALS = [
    (0.0, 1.0), (1.9662155629226508, 3.1938129720301345), (0.1, 0.7),
    (-7.3, -7.3), (-1e308, 1.7e308), (-MAX, MAX), (MAX, MAX), (-MAX, -1e308),
    (ETA, 3 * ETA), (-3 * ETA, ETA), (2.0 ** -1060, 2.0 ** -1060 + 7 * ETA),
    (-0.1, 0.2), (1e15 + 0.5, 1e15 + 3.5), (-math.inf, 2.0), (3.0, math.inf),
    (-math.inf, math.inf), (-math.inf, -5.5),
]


def test_outward_center_radius_contains_the_interval():
    """``[c - r, c + r]`` contains ``[lb, ub]`` in rationals for every
    interval: inexact midpoints (the radius must be the larger of the two
    directed differences), zero width, magnitudes at the overflow threshold
    (halving before the sum), subnormals, and infinite bounds (centre 0,
    radius +inf). The centre is always finite (an infinite centre holds no
    finite point and turns into NaN under any map)."""
    lb = np.array([i[0] for i in _INTERVALS])
    ub = np.array([i[1] for i in _INTERVALS])
    c, r = outward_center_radius(lb, ub)
    for i, (lo, hi) in enumerate(_INTERVALS):
        ci, ri = float(c[i]), float(r[i])
        assert math.isfinite(ci) and not math.isnan(ri) and ri >= 0, (lo, hi, ci, ri)
        if math.isinf(lo) or math.isinf(hi):
            assert ci == 0.0 and ri == math.inf, (lo, hi, ci, ri)
            continue
        assert ri == math.inf or (q(ci) - q(ri) <= q(lo) and q(ci) + q(ri) >= q(hi)), \
            (lo, hi, ci, ri)
    # an exact midpoint and half-width are returned unchanged
    c, r = outward_center_radius(np.array([0.0, -6.0]), np.array([1.0, 2.0]))
    assert c.tolist() == [0.5, -2.0] and r.tolist() == [0.5, 4.0]


def test_box_to_zono_contains_the_box():
    """Public route of ``outward_center_radius``: ``Box.to_zono()`` and
    ``Zono.from_bounds`` hold every corner of the box in rationals (centre
    minus and plus the sum of generator magnitudes per coordinate), and a box
    with an infinite bound gets a finite centre."""
    from n2v.sets import Box, Zono
    lb = np.array([1.9662155629226508, 0.1, -7.3, -1e308, 0.0])
    ub = np.array([3.1938129720301345, 0.7, -7.3, 1.7e308, np.inf])
    for Z in (Box(lb.reshape(-1, 1), ub.reshape(-1, 1)).to_zono(),
              Zono.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))):
        c = np.ravel(Z.c)
        V = np.asarray(Z.V)
        assert np.all(np.isfinite(c)), c
        for i in range(lb.size):
            rad = sum(abs(q(v)) for v in V[i] if math.isfinite(v))
            if math.isinf(ub[i]):
                assert np.any(np.isinf(V[i])), V[i]
                continue
            assert q(c[i]) - rad <= q(lb[i]) and q(c[i]) + rad >= q(ub[i]), (i, c[i], V[i])


# --------------------------------------------------------------------------- #
# _widen_sum and _widen_relative                                                #
# --------------------------------------------------------------------------- #
def test_widen_sum_encloses_a_round_to_nearest_sum():
    """``_widen_sum`` moves a round-to-nearest sum of ``count`` products plus
    a constant outward far enough to hold the exact sum, for both bounds
    (in rationals, zero slack), at magnitudes from 1e-200 to 1e200; a count
    of 0 (a sum of exact zeros) is left exactly as it is, and an infinite or
    overflowing input gives an infinite bound on its own side."""
    rng = np.random.default_rng(5)
    lo_bad = hi_bad = 0
    for _ in range(400):
        cnt = int(rng.integers(1, 7))
        xs = rng.uniform(-3.0, 3.0, cnt) * 10.0 ** rng.integers(-200, 200, cnt).astype(float)
        ws = rng.uniform(-3.0, 3.0, cnt)
        c0 = float(rng.uniform(-3.0, 3.0))
        s = c0
        for w, x in zip(ws, xs):
            s = s + w * x
        exact = q(c0) + sum(q(w) * q(x) for w, x in zip(ws, xs))
        mag = abs(c0) + sum(abs(w * x) for w, x in zip(ws, xs))
        lo, hi = _widen_sum(np.array([s]), np.array([s]), np.array([mag]),
                            np.array([cnt]))
        lo_bad += q(lo[0]) > exact
        hi_bad += q(hi[0]) < exact
    assert lo_bad == 0 and hi_bad == 0, (lo_bad, hi_bad)
    lo, hi = _widen_sum(np.array([0.3, -2.5]), np.array([0.7, 4.0]),
                        np.array([0.7, 4.0]), np.array([0, 0]))
    assert lo.tolist() == [0.3, -2.5] and hi.tolist() == [0.7, 4.0]
    with np.errstate(over="ignore", invalid="ignore"):
        lo, hi = _widen_sum(np.array([-np.inf, 1.0]), np.array([2.0, np.inf]),
                            np.array([np.inf, MAX]), np.array([2, 3]))
    assert lo[0] == -np.inf and hi[1] == np.inf


def test_widen_relative_encloses_a_value_of_relative_accuracy():
    """``_widen_relative`` moves a value computed with relative error at most
    gamma_ops outward far enough to hold the exact value, for inputs of
    either sign and magnitudes from 1e-300 to 1e300 (both bounds, rationals,
    zero slack); a value at an input of exactly 0 is left unchanged
    (sigmoid(0) and tanh(0) are exact), and only there."""
    rng = np.random.default_rng(9)
    bad = []
    for _ in range(600):
        true = q(float(rng.uniform(-5.0, 5.0)) * 10.0 ** int(rng.integers(-300, 300)))
        comp = float(true * (1 + F(float(rng.uniform(-1.0, 1.0))) * 4 * F(2) ** -53))
        x = float(rng.choice([-2.5, -0.1, 3.0, 1e-300]))
        lo, hi = _widen_relative(np.array([comp]), np.array([comp]),
                                 np.array([x]), np.array([x]), 4)
        if q(lo[0]) > true or q(hi[0]) < true:
            bad.append((comp, x))
    assert not bad, bad[:5]
    lo, hi = _widen_relative(np.array([0.5, 0.0]), np.array([0.5, 0.0]),
                             np.array([0.0, -0.0]), np.array([0.0, 0.0]), 4)
    assert lo.tolist() == [0.5, 0.0] and hi.tolist() == [0.5, 0.0]


_WIDEN_F32_DEFECT = ("_widen_relative does its arithmetic in the dtype of its "
                     "inputs (the other directed-rounding helpers convert to "
                     "float64 first): float32 bounds come back unmoved, because "
                     "a pad of gamma_ops relative rounds away in float32; no "
                     "public route passes float32 today")


@pytest.mark.parametrize("dtype", [
    "float64",
    pytest.param("float32", marks=known_bug(7, _WIDEN_F32_DEFECT)),
])
def test_widen_relative_moves_bounds_held_in_any_float_type(dtype):
    """Bounds whose values are within gamma_4 relative of the true value
    must be moved past it, also when they are handed in as float32 numbers
    (a float32 value is a float64 value; the module comment says the
    operands are converted to float64 first). Here the true value lies just
    below the lower bound and just above the upper bound."""
    v = np.array([0.30000001192092896, -2.5], dtype=dtype)       # float32 numbers
    true_lo = [q(x) * (1 - F(2) ** -52) if x > 0 else q(x) * (1 + F(2) ** -52) for x in v]
    true_hi = [q(x) * (1 + F(2) ** -52) if x > 0 else q(x) * (1 - F(2) ** -52) for x in v]
    lo, hi = _widen_relative(v, v, np.array([1.0, -1.0], dtype=dtype),
                             np.array([1.0, -1.0], dtype=dtype), 4)
    for i in range(2):
        assert q(lo[i]) <= true_lo[i] and q(hi[i]) >= true_hi[i], (i, lo[i], hi[i])


def _mp_sigmoid(x):
    import mpmath
    return 1 / (1 + mpmath.exp(-mpmath.mpf(float(x))))


def _mp_tanh(x):
    import mpmath
    return mpmath.tanh(mpmath.mpf(float(x)))


@pytest.mark.parametrize("act", ["Sigmoid", "Tanh"])
def test_sigmoid_and_tanh_of_a_box_hold_the_exact_values(act):
    """Public route of ``_widen_relative``: a model reach of Sigmoid or Tanh
    on a Box. The library function is accurate to a few ulps, not correctly
    rounded, so each bound must lie at or beyond the exact value of the
    function at its endpoint (mpmath at 40 digits) for endpoints of both
    signs, far from and near 0, and deep in the range where sigmoid
    underflows; an endpoint of exactly 0 keeps its exact value. Catches a
    widening skipped for negative inputs and a missing absolute pad."""
    import mpmath
    import torch
    from n2v.nn import NeuralNetwork
    from n2v.sets import Box
    mpmath.mp.dps = 40
    rng = np.random.default_rng(13)
    # [-800, -750]: sigmoid underflows to 0 there while the true value is
    # positive, so the upper bound needs the absolute (subnormal) pad
    lb = np.concatenate([[-0.09247290128736017, -4.0682184936093595,
                          -3.650187839310152, 0.0, -30.0, -800.0],
                         -rng.uniform(0.01, 8.0, 58)])
    ub = np.concatenate([[0.5, 1.0, 2.0, 0.0, 30.0, -750.0],
                         rng.uniform(0.01, 8.0, 58)])
    f = _mp_sigmoid if act == "Sigmoid" else _mp_tanh
    layer = torch.nn.Sigmoid() if act == "Sigmoid" else torch.nn.Tanh()
    with np.errstate(over="ignore"):       # exp(800) overflows to inf: 1 / inf = 0
        out = NeuralNetwork(torch.nn.Sequential(layer)).reach(
            Box(lb.reshape(-1, 1), ub.reshape(-1, 1)), method="approx")
    lo, hi = np.ravel(out[0].lb), np.ravel(out[0].ub)
    bad = [i for i in range(lb.size)
           if mpmath.mpf(float(lo[i])) > f(lb[i]) or mpmath.mpf(float(hi[i])) < f(ub[i])]
    assert not bad, [(lb[i], lo[i]) for i in bad[:5]]


def test_reduce_sum_of_a_box_holds_the_exact_sum(tmp_path):
    """Public route of ``_widen_sum``: a model reach of an ONNX ReduceSum on
    a Box. Both returned bounds hold the exact sum of the box's lower and
    upper endpoints in rationals (endpoints that do not add exactly, both
    signs, magnitudes far from 1)."""
    from onnx import helper
    from n2v.nn import NeuralNetwork
    from n2v.sets import Box
    from n2v.utils.model_loader import load_onnx
    from tests.cell_support import save_onnx
    lb = np.array([0.1, 0.2, -0.7, 1e-3, -13.3])
    ub = np.array([0.3, 0.5, 1.1, 0.7, 8.9])
    path = save_onnx(tmp_path, [helper.make_node("ReduceSum", ["X", "ax"], ["Y"],
                                                 keepdims=1)],
                     [1, lb.size], [1, 1], inits={"ax": np.array([1], dtype=np.int64)})
    out = NeuralNetwork(load_onnx(path)).reach(
        Box(lb.reshape(-1, 1), ub.reshape(-1, 1)), method="approx")
    lo, hi = float(np.ravel(out[0].lb)[0]), float(np.ravel(out[0].ub)[0])
    assert q(lo) <= sum(q(v) for v in lb), lo
    assert q(hi) >= sum(q(v) for v in ub), hi


# --------------------------------------------------------------------------- #
# outward_interval_bounds and outward_interval_slack                            #
# --------------------------------------------------------------------------- #
def _special_values(rng, size):
    """Values of both signs and magnitudes 1e-5 to 1e5, with exact zeros,
    +-1 and products that underflow mixed in."""
    kind = rng.integers(0, 7, size)
    v = rng.uniform(-3.0, 3.0, size) * 10.0 ** rng.integers(-5, 5, size)
    v = np.where(kind == 0, 0.0, v)
    v = np.where(kind == 1, np.sign(v), v)
    return np.where(kind == 2, v * 1e-160, v)


def _row_sets(seed, count):
    """Random (centre, generators, lb, ub) row sets with infinite box entries
    on both sides and exact-zero generators on them."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = int(rng.integers(1, 5))
        rows = int(rng.integers(1, 5))
        lb = _special_values(rng, n)
        ub = lb + np.abs(_special_values(rng, n))
        if rng.uniform() < 0.25:
            lb[rng.integers(0, n)] = -np.inf
        if rng.uniform() < 0.25:
            ub[rng.integers(0, n)] = np.inf
        G = _special_values(rng, (rows, n))
        G = np.where(rng.uniform(size=(rows, n)) < 0.3, 0.0, G)
        yield _special_values(rng, rows), G, lb, ub


def exact_row_range(c, g, lb, ub):
    """The exact range of c + g'a over the box, as rationals; None for an
    unbounded side (a nonzero generator meets an infinite endpoint)."""
    lo = hi = q(c)
    for gj, l, u in zip(g, lb, ub):
        if gj == 0:
            continue
        el, eu = (l, u) if gj > 0 else (u, l)
        lo = None if (lo is None or math.isinf(el)) else lo + q(gj) * q(el)
        hi = None if (hi is None or math.isinf(eu)) else hi + q(gj) * q(eu)
    return lo, hi


def _encloses(lo, hi, elo, ehi):
    lo, hi = float(lo), float(hi)
    ok_lo = lo == -math.inf if elo is None else (not math.isnan(lo) and (lo == -math.inf or q(lo) <= elo))
    ok_hi = hi == math.inf if ehi is None else (not math.isnan(hi) and (hi == math.inf or q(hi) >= ehi))
    return ok_lo and ok_hi


# Named cases: (centre, generators, lb, ub) of one row each
_HAND_ROWS = {
    "two-generators-round": (0.1, [0.2, 0.3], [-1.0, -1.0], [1.0, 1.0]),
    "one-generator-twosum": (0.7, [0.1], [-1.0], [1.0]),
    "one-generator-rounding-endpoint": (0.1, [0.7], [0.3], [0.9]),
    "subnormal-product": (0.0, [1.5e-160], [1.3e-160], [1.0]),
    "subnormal-products-summed": (0.0, [1.5e-160, -1.1e-160, 1.7e-160],
                                  [1.3e-160, 1.9e-160, 1.3e-160], [1.0, 1.0, 1.0]),
    "infinite-lower-nonzero-generator": (1.0, [2.0, 0.5], [-np.inf, -1.0], [1.0, 1.0]),
    "infinite-endpoint-zero-generator": (1.0, [0.0, 0.5], [-np.inf, -1.0], [np.inf, 1.0]),
    "overflow": (1e308, [1e308, 1e308], [0.9, 0.5], [1.0, 1.0]),
}


def test_outward_interval_bounds_hold_the_exact_range():
    """``outward_interval_bounds`` holds the exact range of c + V a over the
    box for every row, in rationals with zero slack: sums of several
    products moved outward (an inward move or a dropped underflow term
    shows), single generators on an endpoint of 0 or +-1 (TwoSum, moved only
    when the error points inward), single generators on an endpoint that
    rounds, subnormal products, infinite endpoints (a nonzero generator
    makes that side infinite, a zero one contributes 0), overflow."""
    sets = list(_row_sets(17, 300))
    sets += [(np.array([c]), np.array([g]), np.array(l), np.array(u))
             for c, g, l, u in _HAND_ROWS.values()]
    bad = []
    with np.errstate(over="ignore", invalid="ignore"):
        for c, G, lb, ub in sets:
            lo, hi = outward_interval_bounds(c, G, lb, ub)
            for i in range(G.shape[0]):
                if not _encloses(lo[i], hi[i], *exact_row_range(c[i], G[i], lb, ub)):
                    bad.append((c[i], G[i].tolist(), lb.tolist(), ub.tolist(), lo[i], hi[i]))
    assert not bad, bad[:5]


def test_outward_interval_bounds_without_generators_is_the_centre():
    """With no predicate variable (a point set) the enclosure is the centre
    itself, exactly, in the centre's own shape (a multi-axis image centre
    included), and with no row the result is empty; through the public
    route a point Star's ``estimate_ranges`` is its centre."""
    from n2v.sets import Star
    c = np.array([[0.1, -2.7, 3.3], [1e300, -0.0, 5e-324]])
    lo, hi = outward_interval_bounds(c, np.zeros((2, 3, 0)), np.zeros(0), np.zeros(0))
    assert lo.shape == (2, 3) and hi.shape == (2, 3)
    assert lo.tolist() == c.tolist() and hi.tolist() == c.tolist()
    lo, hi = outward_interval_bounds(np.zeros(0), np.zeros((0, 3)), -np.ones(3), np.ones(3))
    assert lo.shape == (0,) and hi.shape == (0,)
    S = Star(np.array([[0.1], [-2.7]]), np.zeros((0, 0)), np.zeros((0, 1)))
    lo, hi = S.estimate_ranges()
    assert np.ravel(lo).tolist() == [0.1, -2.7] and np.ravel(hi).tolist() == [0.1, -2.7]


def test_outward_interval_bounds_keep_exact_bounds_exact():
    """The documented exact cases are returned unchanged: a direction whose
    products are all exact zeros is the centre (0.3 + 0.5 a1 + 0.7 a2 over
    [0, 0.3] x [0, 0.6] has the lower bound 0.3 exactly), a row with no
    generator is its centre in both directions, and a single generator on an
    endpoint of +-1 whose sum is exact stays exact ([0, 1] written as
    0.5 + 0.5 a). A bound moved off an exact zero turns a stably inactive
    neuron into a crossing one."""
    lo, hi = outward_interval_bounds(np.array([0.3, -0.7, 0.5]),
                                     np.array([[0.5, 0.7], [0.0, 0.0], [0.5, 0.0]]),
                                     np.array([0.0, 0.0]), np.array([0.3, 0.6]))
    assert lo[0] == 0.3 and lo[1] == -0.7 and hi[1] == -0.7 and lo[2] == 0.5
    lo, hi = outward_interval_bounds(np.array([0.5, -2.0]), np.array([[0.5], [-2.0]]),
                                     np.array([-1.0]), np.array([1.0]))
    assert lo.tolist() == [0.0, -4.0] and hi.tolist() == [1.0, 0.0]


def test_outward_interval_bounds_on_a_basis_that_does_not_merge_and_on_several_row_blocks():
    """Two shapes that take the other branches of the row blocking: a
    multi-axis basis whose leading axes do not merge into rows (a transposed
    image basis: each block is gathered by index) and a basis so wide that
    the rows are processed in several blocks. Every entry, at its own index,
    holds the exact range in rationals (a block written to the wrong rows,
    or rows left unwritten, shows)."""
    rng = np.random.default_rng(23)
    base = rng.uniform(-3.0, 3.0, (3, 4, 3)) * 10.0 ** rng.integers(-3, 3, (3, 4, 3))
    G = base.transpose(1, 0, 2)                     # (4, 3, nVar), not mergeable
    c = rng.uniform(-5.0, 5.0, (4, 3))
    lb = np.array([-1.3, 0.2, -7.0])
    ub = np.array([2.9, 0.7, -0.4])
    lo, hi = outward_interval_bounds(c, G, lb, ub)
    assert lo.shape == (4, 3)
    for i in range(4):
        for j in range(3):
            assert _encloses(lo[i, j], hi[i, j], *exact_row_range(c[i, j], G[i, j], lb, ub)), (i, j)
    # 2^19 + 3 generators: the blocks hold 3 rows each, so 7 rows take 3 blocks
    n = (1 << 19) + 3
    rows = 7
    G = np.zeros((rows, n))
    cols = rng.integers(0, n, (rows, 3))
    for i in range(rows):
        G[i, cols[i]] = rng.uniform(-3.0, 3.0, 3) * (i + 1)
    lbw = np.full(n, -1.0)
    ubw = np.full(n, 1.0)
    lbw[cols.ravel()] = rng.uniform(-9.0, -0.1, cols.size)
    ubw[cols.ravel()] = rng.uniform(0.1, 9.0, cols.size)
    c = rng.uniform(-5.0, 5.0, rows)
    lo, hi = outward_interval_bounds(c, G, lbw, ubw)
    for i in range(rows):
        nz = np.flatnonzero(G[i])
        assert _encloses(lo[i], hi[i], *exact_row_range(c[i], G[i, nz], lbw[nz], ubw[nz])), i


def test_star_estimate_ranges_and_zono_get_bounds_hold_the_exact_range():
    """Public route of ``outward_interval_bounds``: ``Star.estimate_ranges``
    and ``Zono.get_bounds`` hold the exact range of every entry in rationals
    (two generators that round, a single generator on +-1, a subnormal
    product, an infinite predicate bound with a nonzero and with a zero
    generator), and keep the exact bound of an all-zero-product direction."""
    from n2v.sets import Star, Zono
    # predicate variables: a1 in [0, 0.3], a2 in [0, 0.6], a3 in [-1, 1],
    # a4 in [-inf, 1] (with the row a4 <= 1)
    V = np.array([[0.1, 0.2, 0.3, 0.0, 0.0],      # two generators that round
                  [0.3, 0.5, 0.7, 0.0, 0.0],      # lower bound exactly 0.3
                  [1.0, 0.0, 0.0, 0.5, 2.0],      # nonzero generator on -inf
                  [0.7, 0.0, 0.0, 0.1, 0.0],      # one generator on +-1 (TwoSum)
                  [-3.3, 7.1, 0.0, -0.9, 0.0]])   # zero generator on -inf
    lb = np.array([0.0, 0.0, -1.0, -np.inf])
    ub = np.array([0.3, 0.6, 1.0, 1.0])
    S = Star(V, np.array([[0.0, 0.0, 0.0, 1.0]]), np.array([[1.0]]),
             lb.reshape(-1, 1), ub.reshape(-1, 1))
    lo, hi = S.estimate_ranges()
    for i in range(V.shape[0]):
        assert _encloses(lo[i, 0], hi[i, 0], *exact_row_range(V[i, 0], V[i, 1:], lb, ub)), i
    assert lo[1, 0] == 0.3 and lo[2, 0] == -np.inf and math.isfinite(lo[4, 0])
    S2 = Star(np.array([[0.0, 1.5e-160]]), np.zeros((0, 1)), np.zeros((0, 1)),
              np.array([[1.3e-160]]), np.array([[1.0]]))
    lo, hi = S2.estimate_ranges()
    assert _encloses(lo[0, 0], hi[0, 0], *exact_row_range(0.0, [1.5e-160], [1.3e-160], [1.0]))
    Z = Zono(np.array([[0.7], [0.1], [-2.3]]),
             np.array([[0.1, 0.0], [0.2, 0.3], [0.0, 0.0]]))
    lo, hi = Z.get_bounds()
    for i in range(3):
        assert _encloses(lo[i, 0], hi[i, 0],
                         *exact_row_range(Z.c[i, 0], Z.V[i], [-1.0, -1.0], [1.0, 1.0])), i
    assert lo[2, 0] == -2.3 and hi[2, 0] == -2.3


def _rn_interval(c, g, lb, ub):
    """The caller's round-to-nearest interval of c + g'a, summed left to
    right, 0 * inf taken as 0."""
    lo = hi = float(c)
    for gj, l, u in zip(g, lb, ub):
        if gj == 0:
            continue
        lo = lo + gj * (l if gj > 0 else u)
        hi = hi + gj * (u if gj > 0 else l)
    return lo, hi


def test_outward_interval_slack_makes_the_round_to_nearest_interval_rigorous():
    """A caller that sums c + V a over the box in round-to-nearest and then
    subtracts the returned slack from the lower bound and adds it to the
    upper one (both again in round-to-nearest) holds the exact range, in
    rationals with zero slack: rows of one generator (two roundings), of
    several, of products that underflow (the half-subnormal term), at
    magnitudes far from 1. A row with no nonzero generator gets slack 0
    exactly; a nonzero generator on an infinite box entry gives slack +inf,
    a zero one a finite slack."""
    sets = list(_row_sets(29, 300))
    sets += [(np.array([c]), np.array([g]), np.array(l), np.array(u))
             for name, (c, g, l, u) in _HAND_ROWS.items() if name != "overflow"]
    bad = []
    for c, G, lb, ub in sets:
        s = outward_interval_slack(np.abs(c), G, lb, ub)
        for i in range(G.shape[0]):
            elo, ehi = exact_row_range(c[i], G[i], lb, ub)
            nz = G[i] != 0
            if not nz.any():
                if s[i] != 0.0:
                    bad.append(("zero row", c[i], s[i]))
                continue
            if elo is None or ehi is None:
                if s[i] != np.inf:
                    bad.append(("unbounded row", c[i], G[i].tolist(), s[i]))
                continue
            if not math.isfinite(s[i]):
                bad.append(("bounded row, slack not finite", c[i], G[i].tolist(), s[i]))
                continue
            lo, hi = _rn_interval(c[i], G[i], lb, ub)
            if q(lo - s[i]) > elo or q(hi + s[i]) < ehi:
                bad.append((c[i], G[i].tolist(), lb.tolist(), ub.tolist(), float(s[i])))
    assert not bad, bad[:5]


@pytest.mark.parametrize("gamma", [0.01, 2.0, -0.5])
@pytest.mark.parametrize("lo,hi", [(-2.4416780152088147, 2.4416780152088147),
                                   (-0.7, 0.3), (-3.0, 7.0), (-1e-4, 1e-4),
                                   (-1.0, 1e-4)])
def test_leakyrelu_star_reach_holds_the_true_outputs(lo, hi, gamma):
    """Public route of ``outward_interval_slack``: the approx LeakyReLU of a
    one-dimensional Star moves the right-hand sides of its relaxation rows
    outward by this slack. For input points at both ends of the range, at 0
    and inside (all rational), the true output f(x) is a member of the
    returned set, and so is the pair (x, f(x)) of the set's graph: the reach
    keeps the input's predicate variables as the first ones (the documented
    prefix that witnesses and input contraction rely on), so the input entry
    is c + r a1 over the returned set's own rows. Decided exactly in
    rationals. At the range ends the rows pass through the graph, so a slack
    that is too small (no safety factor, one term short) cuts the pair off
    by an ulp. Slopes below and above 1 and negative (concave)."""
    import torch
    from n2v.nn import NeuralNetwork
    from n2v.sets import Star
    from tests._oracles.rational import member, star
    S = Star.from_bounds(np.array([[lo]]), np.array([[hi]]))
    out = NeuralNetwork(torch.nn.Sequential(torch.nn.LeakyReLU(gamma))).reach(
        S, method="approx")
    assert len(out) == 1
    c, r = q(S.V[0, 0]), q(S.V[0, 1])
    g = q(gamma)
    oc, oG, oC, od, olb, oub = star(out[0])
    if len(oG[0]) < 1:
        pytest.fail("test setup: the output has no predicate variable")
    graph = ([c] + list(oc), [[r] + [F(0)] * (len(oG[0]) - 1)] + [list(row) for row in oG],
             oC, od, olb, oub)
    missing = []
    for a in (F(-1), F(-1, 2), F(-1, 3), F(0), F(1, 4), F(1)):
        x = c + r * a
        y = x if x >= 0 else g * x
        if not member((oc, oG, oC, od, olb, oub), [y]):
            missing.append(("output", float(a)))
        if not member(graph, [x, y]):
            missing.append(("graph", float(a)))
    assert not missing, missing


# --------------------------------------------------------------------------- #
# outward_spec_rows                                                             #
# --------------------------------------------------------------------------- #
def _corners(lb, ub):
    pts = [[]]
    for l, u in zip(lb, ub):
        pts = [p + [v] for p in pts for v in (q(l), q(u), (q(l) + q(u)) / 2)]
    return pts


def test_outward_spec_rows_keep_every_point_of_the_unsafe_region():
    """Every predicate point a of the box whose exact image y = c + V'a lies
    in the unsafe half-space ``H y <= g`` satisfies the pulled-back row
    ``(H V') a <= rhs`` exactly, in rationals: stars with a nonzero centre
    (the centre term subtracted, not added), specs that touch the set at a
    corner (g is the rounded value of H y there, so the slack must be added
    to the right-hand side), with and without the predicate box handed in
    (without it only the centre point a = 0 is promised), rows that nearly
    cancel at the touching point (where only the rounding of H V' over the
    predicate box covers the pulled-back row), and a star with no predicate
    variables."""
    rng = np.random.default_rng(31)
    bad = []
    for t in range(150):
        dim = int(rng.integers(1, 4))
        nv = int(rng.integers(1, 4))
        k = int(rng.integers(1, 3))
        V = rng.uniform(-3.0, 3.0, (dim, nv + 1)) * 10.0 ** rng.integers(-2, 3, (dim, nv + 1))
        H = rng.uniform(-3.0, 3.0, (k, dim))
        plb = -rng.uniform(0.1, 5.0, nv)
        pub = rng.uniform(0.1, 5.0, nv)
        pts = _corners(plb, pub)
        touch = pts[int(rng.integers(0, len(pts)))]
        Vq = [[q(v) for v in row] for row in V]
        img = lambda a: [Vq[i][0] + sum(Vq[i][j + 1] * a[j] for j in range(nv))  # noqa: E731
                         for i in range(dim)]
        Hq = [[q(v) for v in row] for row in H]
        Hy = lambda a: [sum(Hq[r][i] * yi for i, yi in enumerate(img(a)))  # noqa: E731
                        for r in range(k)]
        g = np.array([float(v) for v in Hy(touch)])
        for with_box in (True, False):
            Ha, rhs = (outward_spec_rows(H, g, V, plb, pub) if with_box
                       else outward_spec_rows(H, g, V))
            check = pts if with_box else [[F(0)] * nv]
            for a in check:
                hy = Hy(a)
                for r in range(k):
                    if hy[r] <= q(g[r]):
                        lhs = sum(q(Ha[r, j]) * a[j] for j in range(nv))
                        if lhs > q(rhs[r]):
                            bad.append((t, with_box, r))
    # rows that nearly cancel at the touching corner (c = 0, the last entry of
    # H solves H V' a = 0 up to rounding): the right-hand side is then tiny and
    # only the rounding of H V' times the predicate box can cover the rounding
    # of the pulled-back row
    for t in range(60):
        dim = int(rng.integers(2, 4))
        nv = int(rng.integers(2, 4))
        V = np.hstack([np.zeros((dim, 1)),
                       rng.uniform(-3.0, 3.0, (dim, nv)) * 10.0 ** rng.integers(-2, 3, (dim, nv))])
        plb = -rng.uniform(0.1, 5.0, nv)
        pub = rng.uniform(0.1, 5.0, nv)
        pts = _corners(plb, pub)
        touch = pts[int(rng.integers(0, len(pts)))]
        y = [sum(q(V[i, j + 1]) * touch[j] for j in range(nv)) for i in range(dim)]
        if y[-1] == 0:
            continue
        H = rng.uniform(-3.0, 3.0, (1, dim))
        H[0, -1] = float(-sum(q(H[0, i]) * y[i] for i in range(dim - 1)) / y[-1])
        e = sum(q(H[0, i]) * y[i] for i in range(dim))
        g = np.array([float(e)])
        if q(g[0]) < e:
            g = np.nextafter(g, np.inf)
        Ha, rhs = outward_spec_rows(H, g, V, plb, pub)
        if sum(q(Ha[0, j]) * touch[j] for j in range(nv)) > q(rhs[0]):
            bad.append(("cancelling", t))
    # no predicate variables: a point star exactly on the boundary
    for c0 in (0.1, -2.7, 1e-3):
        V = np.array([[c0], [0.3]])
        H = np.array([[0.7, -1.3]])
        g = np.array([float(q(0.7) * q(c0) - q(1.3) * q(0.3))])
        if q(g[0]) < q(0.7) * q(c0) - q(1.3) * q(0.3):
            g = np.nextafter(g, np.inf)
        Ha, rhs = outward_spec_rows(H, g, V)
        assert Ha.shape == (1, 0)
        if not q(rhs[0]) >= 0:
            bad.append(("point star", c0, float(rhs[0])))
    assert not bad, bad[:5]


def test_a_touching_star_is_not_proved_disjoint_from_the_unsafe_region():
    """Public route of ``outward_spec_rows``: a reach set that touches the
    unsafe region (y in [0, 1], unsafe y >= 1) or lies partly in it with a
    nonzero centre (y in [1, 3], unsafe y >= 2.5) is not UNSAT for
    ``verify_specification``, and ``Star.intersect_half_space`` keeps the
    touching point (a member of the result, decided exactly in rationals on
    its rows)."""
    from n2v.sets import HalfSpace, Star
    from n2v.utils.verify_specification import verify_specification
    from tests._oracles.rational import member, star
    cases = [(0.0, 1.0, 1.0), (1.0, 3.0, 2.5), (-0.3, 0.7, 0.7), (0.1, 0.30000000000000004, 0.30000000000000004)]
    for lo, hi, thr in cases:
        S = Star.from_bounds(np.array([[lo]]), np.array([[hi]]))
        spec = HalfSpace(np.array([[-1.0]]), np.array([[-thr]]))
        verdict = verify_specification([S], spec).verdict
        assert verdict != "UNSAT", (lo, hi, thr, verdict)
        T = S.intersect_half_space(np.array([[-1.0]]), np.array([[-thr]]))
        assert member(star(T), [q(hi)]), (lo, hi, thr)


# --------------------------------------------------------------------------- #
# Certified GPU LP entries on CPU torch, and the refinement engine routes        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows", ["rows", "box-only"])
def test_solve_lp_batch_gpu_holds_every_exact_optimum(rows):
    """``solve_lp_batch_gpu`` (device 'cpu', a short descent) returns for
    every objective a bound that holds the exact optimum of its LP: a lower
    bound for a minimisation, an upper bound for a maximisation, with flags
    mixed and the first one a minimisation (one flag reused for all shows),
    on rows that cut the box (the box handed to the certificate in the wrong
    order shows) and on the box alone (the scalar certificate per objective).
    Zero slack in rationals."""
    from n2v.utils.lpsolver_gpu import solve_lp_batch_gpu
    rng = np.random.default_rng(37)
    bad = []
    for spec in TIGHT_LPS:
        c, A, b, lb, ub, _ = _lp(spec)
        objs = [-c, c, rng.uniform(-3.0, 3.0, c.size), -c, c]
        flags = [True, False, True, False, True]
        AA, bb = (A, b) if rows == "rows" else (None, None)
        out = solve_lp_batch_gpu(objs, A=AA, b=bb, lb=lb, ub=ub,
                                 minimize_flags=flags, max_iters=25, device="cpu")
        assert len(out) == len(objs)
        for o, f, v in zip(objs, flags, out):
            exact = exact_opt(o, AA, bb, lb, ub, f)
            if not holds(v, exact, lower=f):
                bad.append((spec, f, v, float(exact)))
    assert not bad, bad


def test_solve_lp_batch_gpu_without_a_box_claims_nothing_wrong():
    """``lb``/``ub`` left out mean an unbounded box: the certified values
    must still hold the exact optimum (max x1 + 2 x2 over the rows alone is
    unbounded above, so only +inf holds; the minimum likewise), never NaN."""
    from n2v.utils.lpsolver_gpu import solve_lp_batch_gpu
    A = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]])
    b = np.array([1.0, 0.5, 0.25])
    out = solve_lp_batch_gpu([np.array([1.0, 2.0])] * 2, A=A, b=b,
                             minimize_flags=[False, True], max_iters=10,
                             device="cpu")
    assert out[0] == np.inf and out[1] == -np.inf, out


def test_solve_lp_population_gpu_holds_every_exact_optimum():
    """``solve_lp_population_gpu`` (device 'cpu') over stars with their own
    rows and boxes, zero-padded rows and variables, a warm start that puts
    weight on padded rows, and flags mixed starting with a minimisation:
    every (star, objective) bound holds the exact optimum of that star's own
    LP in rationals (a box read from the wrong star or the wrong endpoint, a
    sign not flipped back, a padded row that is not inert shows)."""
    from n2v.utils.lpsolver_gpu import solve_lp_population_gpu
    kinds = ["certificate", "random", "zero", "certificate", "negative"]
    lps, A, b, lb, ub, C, Y, flags = _population(TIGHT_LPS, kinds)
    for Y0 in (None, Y):
        r = solve_lp_population_gpu(A, b, lb, ub, C, flags, max_iters=25,
                                    Y0=Y0, device="cpu")
        assert r.shape == (len(lps), len(kinds)) and r.dtype == np.float64
        bad = []
        for s, (c, As, bs, l, u, _) in enumerate(lps):
            for j in range(len(kinds)):
                cc = -c if flags[j] else c
                exact = exact_opt(cc, As, bs, l, u, flags[j])
                if not holds(r[s, j], exact, lower=flags[j]):
                    bad.append((Y0 is None, s, j, float(r[s, j]), float(exact)))
        assert not bad, bad


def _star_with_rows():
    """A Star over three predicate variables with two rows that cut its box
    (bounds far from +-1, not dyadic), and its outputs crossing zero."""
    from n2v.sets import Star
    V = np.array([[0.3, 1.7, -0.9, 0.4],
                  [-0.2, -2.3, 0.6, 1.1],
                  [1.1, 0.5, 0.5, -0.5],
                  [-4.0, 0.25, 0.1, 0.3]])
    C = np.array([[1.0, 1.0, 0.5], [-0.7, 0.3, 1.0]])
    d = np.array([[0.9], [1.3]])
    lb = np.array([-2.1, -1.4, -3.3])
    ub = np.array([1.7, 2.6, 0.8])
    return Star(V, C, d, lb.reshape(-1, 1), ub.reshape(-1, 1))


def exact_star_ranges(S, C=None, d=None):
    """Exact min and max of every entry of a Star over its predicate
    polytope (rows C, d and the predicate box), from the vertices."""
    C = S.C if C is None else C
    d = S.d if d is None else d
    lb = np.ravel(S.predicate_lb)
    ub = np.ravel(S.predicate_ub)
    lo, hi = [], []
    for i in range(S.dim):
        lo.append(q(S.V[i, 0]) + exact_opt(S.V[i, 1:], C, np.ravel(d), lb, ub, True))
        hi.append(q(S.V[i, 0]) + exact_opt(S.V[i, 1:], C, np.ravel(d), lb, ub, False))
    return lo, hi


@pytest.fixture
def cpu_engine(monkeypatch):
    """The refinement engine resolves ``device=None`` through
    ``gpu_available``; force the CPU-torch path whatever the host has."""
    monkeypatch.setattr("n2v.utils.lpsolver_gpu.gpu_available", lambda: False)


@pytest.mark.parametrize("rows", ["rows", "box-only"])
def test_neuron_bounds_dual_gpu_hold_the_exact_ranges(cpu_engine, rows):
    """Public route of ``solve_lp_batch_gpu`` and the certificate:
    ``n2v.refine.tighten.neuron_bounds(S, 'dual_gpu')`` on CPU torch holds
    the exact range of every entry of the Star over its predicate polytope
    (rationals, zero slack), for a Star whose rows cut the box (the rows
    path and the vectorised certificate) and for one with no rows (the
    box-only path and the scalar certificate)."""
    from n2v.refine.tighten import neuron_bounds
    from n2v.sets import Star
    S = _star_with_rows()
    if rows == "box-only":
        S = Star(S.V, np.zeros((0, S.nVar)), np.zeros((0, 1)), S.predicate_lb,
                 S.predicate_ub)
    lo, hi = neuron_bounds(S, "dual_gpu", gpu_iters=40)
    elo, ehi = exact_star_ranges(S)
    for i in range(S.dim):
        assert holds(lo[i, 0], elo[i], lower=True), (i, lo[i, 0], float(elo[i]))
        assert holds(hi[i, 0], ehi[i], lower=False), (i, hi[i, 0], float(ehi[i]))


@known_bug(17, _NAN_DEFECT)
def test_neuron_bounds_dual_gpu_on_a_star_with_a_vacuous_row(cpu_engine):
    """A constraint row whose right-hand side is +inf holds every point
    (``Star.convex_hull`` documents such rows); the certified ranges of
    ``neuron_bounds(S, 'dual_gpu')`` must still hold the exact ranges and
    never be NaN (a NaN bound reads as a stably active neuron). Neighbour:
    ``test_neuron_bounds_dual_gpu_hold_the_exact_ranges[rows]``, the same
    Star without the vacuous row."""
    from n2v.refine.tighten import neuron_bounds
    from n2v.sets import Star
    S0 = _star_with_rows()
    S = Star(S0.V, np.vstack([S0.C, [[1.0, 0.0, 0.0]]]),
             np.vstack([S0.d, [[np.inf]]]), S0.predicate_lb, S0.predicate_ub)
    with np.errstate(invalid="ignore", over="ignore"):
        lo, hi = neuron_bounds(S, "dual_gpu", gpu_iters=40)
    elo, ehi = exact_star_ranges(S0)
    for i in range(S.dim):
        assert holds(lo[i, 0], elo[i], lower=True), (i, lo[i, 0])
        assert holds(hi[i, 0], ehi[i], lower=False), (i, hi[i, 0])


def test_contract_input_boxes_hold_the_exact_input_ranges():
    """Public route of ``solve_lp_population_gpu``:
    ``contract_input_boxes(..., device='cpu')`` returns for each system a
    box that holds the exact range of every leading predicate variable over
    ``{C a <= d, box}`` (rationals, zero slack), and calls a system empty
    only when it is: a system whose feasible set is a single corner must
    not be called empty."""
    from n2v.refine.tighten import contract_input_boxes
    systems = [
        (np.array([[1.0, 1.0, 0.5], [-0.7, 0.3, 1.0]]), np.array([[0.9], [1.3]])),
        (np.array([[1.0, 1.0]]), np.array([[-2.0]])),           # one point: (-1, -1)
        (np.array([[0.3, -1.1, 0.0], [2.0, 0.0, -1.0], [0.0, 1.0, 1.0]]),
         np.array([[0.2], [1.5], [0.1]])),
    ]
    boxes = [(np.array([-2.1, -1.4, -3.3]), np.array([1.7, 2.6, 0.8])),
             (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
             (np.array([-4.5, -0.3, -2.0]), np.array([3.25, 5.5, 2.0]))]
    out = contract_input_boxes(systems, boxes, 2, gpu_iters=40, device="cpu")
    for (C, d), (lb, ub), (nlb, nub, empty) in zip(systems, boxes, out):
        assert not empty, (C, nlb, nub)
        for i in range(2):
            e = np.zeros(lb.size)
            e[i] = 1.0
            assert holds(nlb[i, 0], exact_opt(e, C, d, lb, ub, True), lower=True), (i, nlb)
            assert holds(nub[i, 0], exact_opt(e, C, d, lb, ub, False), lower=False), (i, nub)


def test_certified_prune_wave_never_prunes_a_star_that_meets_the_unsafe_region():
    """Public route of ``outward_spec_rows`` and the population certificate:
    ``certified_prune_wave(..., device='cpu')`` must not prune a star that
    meets the unsafe region, here at exactly one point: a box star whose
    corner lies on the unsafe half-space (the certificate is exact there, so
    a pulled-back right-hand side moved inward shows), the same star with a
    row that cuts the box, and a point star on the boundary."""
    from n2v.refine import LinearSpec
    from n2v.refine.witness import certified_prune_wave
    from n2v.sets import Star
    V = np.array([[0.5, 0.25, -0.75], [-1.0, 1.5, 0.5]])
    lb = np.array([-2.0, -1.0])
    ub = np.array([1.0, 3.0])
    # x0 = 0.5 + 0.25 a1 - 0.75 a2 has its maximum 1.5 at (1, -1)
    box_star = Star(V, np.zeros((0, 2)), np.zeros((0, 1)), lb.reshape(-1, 1),
                    ub.reshape(-1, 1))
    cut_star = Star(V, np.array([[1.0, 1.0]]), np.array([[0.0]]), lb.reshape(-1, 1),
                    ub.reshape(-1, 1))
    point = Star(np.array([[1.5], [-1.25]]), np.zeros((0, 0)), np.zeros((0, 1)))
    spec = LinearSpec(G=np.array([[-1.0, 0.0]]), g=np.array([-1.5]))   # x0 >= 1.5
    pruned = certified_prune_wave([box_star, cut_star, point], spec,
                                  max_iters=40, device="cpu")
    assert pruned == [False, False, False], pruned
