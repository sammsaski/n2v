"""Regressions for rounding in the set conversions and the directed-rounding
helpers (n2v/sets/star.py, n2v/sets/box.py, n2v/utils/ns_certificate.py).

- ``Star.estimate_range(s)`` on a star with no predicate box took raw LP
  optima from the configured default solver with no outward pad, so under
  SCS the "over-approximate" range left out feasible points. An LP with no
  optimum (infeasible or unbounded) came back as ``None`` / NaN.
- ``Box.to_zono`` (and so ``Box.to_star`` / ``Star.from_bounds``) used the
  round-to-nearest midpoint and half-width, so the zonotope missed an
  endpoint of the box by an ulp in about half of all boxes.
- ``_mul_dir`` gave NaN for ``0 * inf``, so a Box with an infinite endpoint
  times a zero (constant or endpoint) got NaN bounds. ``_add_dir`` and
  ``_div_dir`` gave NaN for ``inf - inf`` / ``inf / inf``, and ``_div_dir``
  a decided infinite bound for a zero divisor.
- The LP bounds for a star with no predicate box were box-free LPs, which
  get no NS certificate: HiGHS reported a finite optimum for an unbounded
  direction of slope 1e-9, and a pad sized by ``optimum + center`` was far
  below the optimum's own error when the center cancelled it.
- ``Box.single_partition`` ended its last piece an ulp below ``ub``, and
  ``Star.convex_hull`` rounded its center and half-difference to nearest
  and, for an operand whose predicate excludes 0, did not contain the other
  operand at all.

Every check is against exact rationals. Convention: SAT = unsafe (the
unsafe region is ``G y <= g``), UNSAT = safe.
"""

from fractions import Fraction

import numpy as np
import pytest

from n2v.config import config
from n2v.nn.reach import _mul_boxes, _mul_sets_by_constant
from n2v.sets import Box, Star
from n2v.utils.lpsolver import solve_lp
from n2v.utils.ns_certificate import _add_dir, _div_dir, _mul_dir


def _dot(row, x):
    return sum(Fraction(float(a)) * Fraction(float(b)) for a, b in zip(row, x))


# ---------------------------------------------------------------------------
# Star.estimate_range(s) with no predicate box
# ---------------------------------------------------------------------------

@pytest.fixture
def scs_default():
    import cvxpy as cp
    if "SCS" not in cp.installed_solvers():
        pytest.skip("SCS not installed")
    original = config.lp_solver
    config.lp_solver = "SCS"
    try:
        yield
    finally:
        config.lp_solver = original


def test_estimate_ranges_without_predicate_box_contain_feasible_points(scs_default):
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(10):
        n = 4
        C = np.vstack([np.eye(n), -np.eye(n), rng.normal(size=(4, n))])
        d = np.vstack([np.ones((2 * n, 1)), rng.uniform(0.2, 1.0, (4, 1))])
        V = np.hstack([rng.normal(size=(3, 1)), rng.normal(size=(3, n))])
        S = Star(V, C, d)
        assert S.predicate_lb is None
        lo, hi = S.estimate_ranges()
        for i in range(S.dim):
            lo_i, hi_i = S.estimate_range(i)
            for minimize in (True, False):
                x, _, status, _ = solve_lp(V[i, 1:], A=C, b=d,
                                           lp_solver="highs", minimize=minimize)
                if status != "optimal":
                    continue
                x = np.asarray(x).ravel()
                if any(_dot(r, x) > Fraction(float(di))
                       for r, di in zip(C, d.ravel())):
                    continue    # not exactly feasible: not a witness
                val = Fraction(float(V[i, 0])) + _dot(V[i, 1:], x)
                assert Fraction(float(lo[i, 0])) <= val <= Fraction(float(hi[i, 0]))
                assert Fraction(lo_i) <= val <= Fraction(hi_i)
                checked += 1
    assert checked >= 20


def test_estimate_range_without_predicate_box_is_never_undecided_or_nan():
    # x <= -1 and x >= 1: infeasible, no predicate box.
    S = Star(np.array([[0.0, 1.0]]), np.array([[1.0], [-1.0]]),
             np.array([[-1.0], [-1.0]]))
    assert S.estimate_range(0) == (-np.inf, np.inf)
    lo, hi = S.estimate_ranges()
    assert lo[0, 0] == -np.inf and hi[0, 0] == np.inf
    # The LP range falls back to estimate_range when its LP fails.
    lo, hi = S.get_ranges()
    assert not np.isnan(lo).any() and not np.isnan(hi).any()

    # x <= 1 only: unbounded below. The predicate variable has no finite
    # box, so the max is not trusted either.
    S = Star(np.array([[0.0, 1.0]]), np.array([[1.0]]), np.array([[1.0]]))
    lo_i, hi_i = S.estimate_range(0)
    assert lo_i == -np.inf and hi_i > 1.0


@pytest.mark.parametrize("eps", [1e-8, 1e-10, 1e-12])
def test_estimate_range_without_predicate_box_is_unbounded_along_a_small_slope(eps):
    # x = a1 + eps * a2, a1 in [-1, 1], a2 >= 0: unbounded above, along a
    # slope below HiGHS's dual feasibility tolerance.
    S = Star(np.array([[0.0, 1.0, eps]]),
             np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
             np.array([[1.0], [1.0], [0.0]]))
    lo, hi = S.estimate_ranges()
    lo_i, hi_i = S.estimate_range(0)
    for a2 in (1e12, 1e100):                       # (1, a2) is feasible
        val = Fraction(1) + Fraction(eps) * Fraction(a2)
        # (a Fraction compares exactly with a float, infinities included)
        assert val <= float(hi[0, 0]) and val <= hi_i
    assert lo[0, 0] <= -1.0 and lo_i <= -1.0


def test_estimate_range_without_predicate_box_survives_center_cancellation():
    # x = c + f a with q |a| <= 1 and c = -f / q rounded: the maximum is
    # about 0 while the LP optimum is about 1e11.
    rng = np.random.default_rng(0)
    for _ in range(60):
        q = float(rng.choice([3.0, 7.0, 11.0, 0.3, 0.7]))
        f = float(rng.uniform(1e9, 1e11))
        c = -float(Fraction(f) / Fraction(q))
        S = Star(np.array([[c, f]]), np.array([[q], [-q]]),
                 np.array([[1.0], [1.0]]))
        lo_i, hi_i = S.estimate_range(0)
        lo, hi = S.estimate_ranges()
        a = 1.0 / q
        while Fraction(q) * Fraction(a) > 1:       # largest feasible float
            a = np.nextafter(a, -np.inf)
        for alpha in (a, -a):
            val = Fraction(c) + Fraction(f) * Fraction(alpha)
            assert lo_i <= val <= hi_i
            assert float(lo[0, 0]) <= val <= float(hi[0, 0])


def test_estimate_ranges_without_predicate_box_keep_constant_rows_exact():
    S = Star(np.array([[0.3, 0.0], [0.1, 1.0]]), np.array([[1.0], [-1.0]]),
             np.array([[1.0], [1.0]]))
    lo, hi = S.estimate_ranges()
    assert lo[0, 0] == hi[0, 0] == 0.3


# ---------------------------------------------------------------------------
# Box.to_zono
# ---------------------------------------------------------------------------

def _covers(box, c, G):
    """The exact set ``c + G alpha``, ``alpha in [-1, 1]``, contains ``box``."""
    c = np.asarray(c).ravel()
    for i in range(box.dim):
        r = sum(abs(Fraction(float(v))) for v in G[i])
        ci = Fraction(float(c[i]))
        if not (ci - r <= Fraction(float(box.lb[i, 0]))
                and ci + r >= Fraction(float(box.ub[i, 0]))):
            return False
    return True


def test_box_to_zono_contains_the_box_exactly():
    rng = np.random.default_rng(0)
    for t in range(600):
        if t % 2:
            lb = rng.uniform(-10, 10, size=4)
            ub = lb + rng.uniform(0, 1, size=4) * 10.0 ** rng.integers(-15, 1, size=4)
        else:
            # zero-straddling, with independent magnitudes on each side
            lb = -rng.uniform(0, 1, size=4) * 10.0 ** rng.integers(-8, 2, size=4)
            ub = rng.uniform(0, 1, size=4) * 10.0 ** rng.integers(-8, 2, size=4)
        box = Box(lb, ub)
        Z = box.to_zono()
        assert _covers(box, Z.c, Z.V)
        S = box.to_star()
        assert _covers(box, S.V[:, 0], S.V[:, 1:])


def test_box_to_zono_edge_cases():
    # Exact midpoint and half-width: unchanged.
    Z = Box([0.0, 0.0, 3.0], [1.0, 2.0, 3.0]).to_zono()
    assert np.array_equal(Z.c.ravel(), [0.5, 1.0, 3.0])
    assert np.array_equal(Z.V, [[0.5, 0.0], [0.0, 1.0], [0.0, 0.0]])
    # A midpoint whose sum overflows, and a width near the largest double.
    for lb, ub in (([1e308], [1.7e308]), ([-1.7e308], [1.79e308])):
        box = Box(lb, ub)
        Z = box.to_zono()
        assert np.isfinite(Z.c).all() and _covers(box, Z.c, Z.V)
    # Unbounded coordinates: a finite center and an infinite radius.
    Z = Box([0.0, -np.inf], [np.inf, np.inf]).to_zono()
    assert np.array_equal(Z.c.ravel(), [0.0, 0.0])
    assert np.all(np.abs(Z.V).sum(axis=1) == np.inf)


def test_box_single_partition_covers_the_box():
    rng = np.random.default_rng(0)
    for _ in range(500):
        lb = rng.uniform(-5, 5)
        ub = lb + rng.uniform(0, 3)
        k = int(rng.integers(2, 9))
        parts = Box([lb, 0.0], [ub, 1.0]).single_partition(0, k)
        assert parts[0].lb[0, 0] == lb and parts[-1].ub[0, 0] == ub
        for p, q in zip(parts, parts[1:]):
            assert p.ub[0, 0] == q.lb[0, 0]


def _hull_point(H, S1, S2, a1, a2, lam):
    """``H``'s predicate point (exact rationals) for ``lam`` times the point
    of ``S1`` at ``a1`` plus ``1 - lam`` times that of ``S2`` at ``a2``:
    ``[lam a1, (1 - lam) a2, lam, lam s]``, the slack ``s`` solving each
    inexact center difference exactly."""
    k = H.nVar - S1.nVar - S2.nVar - 1
    s = []
    for j in range(k):
        i = int(np.flatnonzero(H.V[:, -k + j])[0])
        r = (Fraction(float(S1.V[i, 0])) - Fraction(float(S2.V[i, 0]))
             - Fraction(float(H.V[i, S1.nVar + S2.nVar + 1])))
        s.append(r / Fraction(float(H.V[i, -k + j])))
    return ([lam * a for a in a1] + [(1 - lam) * a for a in a2] + [lam]
            + [lam * v for v in s])


def _in_star_exactly(H, z, y):
    """``z`` satisfies ``H``'s constraints and maps to ``y``, exactly."""
    for row, di in zip(H.C, H.d.ravel()):
        if _dot_frac(row, z) > Fraction(float(di)):
            return False
    if H.predicate_lb is not None:
        for zi, l, u in zip(z, H.predicate_lb.ravel(), H.predicate_ub.ravel()):
            if not Fraction(float(l)) <= zi <= Fraction(float(u)):
                return False
    return all(Fraction(float(H.V[i, 0])) + _dot_frac(H.V[i, 1:], z) == y[i]
               for i in range(H.dim))


def _dot_frac(row, z):
    return sum(Fraction(float(a)) * zi for a, zi in zip(row, z))


def test_star_convex_hull_contains_both_operands_exactly():
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(40):
        c1, c2 = rng.uniform(-5, 5, 3), rng.uniform(-5, 5, 3)
        # point stars, and box stars
        cases.append((Star(c1.reshape(-1, 1)), Star(c2.reshape(-1, 1))))
        cases.append((Star.from_bounds(c1, c1 + 1e-3),
                      Star.from_bounds(c2, c2 + rng.uniform(0, 1, 3))))
    # A predicate that excludes 0: {10 + a, a in [0.5, 1]} with the point 0.
    far = Star(np.array([[10.0, 1.0]]), np.array([[1.0], [-1.0]]),
               np.array([[1.0], [-0.5]]))
    cases.append((Star(np.array([[0.0]])), far))
    cases.append((far, Star(np.array([[0.0]]))))
    for S1, S2 in cases:
        H = S1.convex_hull(S2)
        for S, which in ((S1, 1), (S2, 0)):
            # the operand's predicate points at the corners of a box that
            # contains its predicate polytope, kept when feasible exactly
            lo, hi = (S.predicate_lb, S.predicate_ub) if S.predicate_lb is not None \
                else (np.full((S.nVar, 1), 0.5), np.full((S.nVar, 1), 1.0))
            for corner in (lo, hi):
                a = [Fraction(float(v)) for v in corner.ravel()]
                if any(_dot_frac(r, a) > Fraction(float(di))
                       for r, di in zip(S.C, S.d.ravel())):
                    continue
                y = [Fraction(float(S.V[i, 0])) + _dot_frac(S.V[i, 1:], a)
                     for i in range(S.dim)]
                zeros1 = [Fraction(0)] * S1.nVar
                zeros2 = [Fraction(0)] * S2.nVar
                z = (_hull_point(H, S1, S2, a, zeros2, Fraction(1)) if which
                     else _hull_point(H, S1, S2, zeros1, a, Fraction(0)))
                assert _in_star_exactly(H, z, y)


# ---------------------------------------------------------------------------
# Directed-rounding helpers
# ---------------------------------------------------------------------------

_INF = np.inf
_SPECIAL = [0.0, -0.0, 1.0, -1.0, _INF, -_INF, 5e-324, -5e-324, 2.2e-308,
            1.7976931348623157e308, -1.7976931348623157e308, 1e200, -1e200,
            1e-200, 0.1, -0.3, 3.0]


def _exact(op, a, b):
    """``('num', Fraction)``, ``('inf', sign)`` or ``('none',)`` (no value),
    with ``0 * inf = 0``."""
    ia, ib = np.isinf(a), np.isinf(b)
    sa, sb = np.sign(a), np.sign(b)
    if op == "add":
        if not ia and not ib:
            return ("num", Fraction(a) + Fraction(b))
        if ia and ib and sa != sb:
            return ("none",)
        return ("inf", sa if ia else sb)
    if op == "mul":
        if a == 0.0 or b == 0.0:
            return ("num", Fraction(0))
        if not ia and not ib:
            return ("num", Fraction(a) * Fraction(b))
        return ("inf", sa * sb)
    if b == 0.0 or (ia and ib):
        return ("none",)
    if ib:
        return ("num", Fraction(0))
    if ia:
        return ("inf", sa * sb)
    return ("num", Fraction(a) / Fraction(b))


def _is_bound(r, e, d):
    if np.isnan(r):
        return False
    if e[0] == "none":
        return r == d * _INF
    if e[0] == "inf":
        return r == e[1] * _INF if d == e[1] else True
    if np.isinf(r):
        return np.sign(r) == d
    return Fraction(r) <= e[1] if d < 0 else Fraction(r) >= e[1]


@pytest.mark.parametrize("op, fn", [("add", _add_dir), ("mul", _mul_dir),
                                    ("div", _div_dir)])
def test_directed_rounding_is_a_bound_of_the_exact_value(op, fn):
    rng = np.random.default_rng(1)
    vals = _SPECIAL + list(rng.normal(size=60) * 10.0 ** rng.integers(-30, 30, 60))
    A = np.repeat(vals, len(vals))
    B = np.tile(vals, len(vals))
    for d in (-1.0, 1.0):
        R = fn(A, B, d)
        bad = [(a, b, r) for a, b, r in zip(A, B, R)
               if not _is_bound(float(r), _exact(op, float(a), float(b)), d)]
        assert not bad, bad[:5]


@pytest.mark.parametrize("zero", [0.0, -0.0])
@pytest.mark.parametrize("inf", [np.inf, -np.inf])
def test_zero_times_infinity_is_zero(zero, inf):
    for d in (-1.0, 1.0):
        assert _mul_dir(np.array([zero]), np.array([inf]), d)[0] == 0.0
        assert _mul_dir(np.array([inf]), np.array([zero]), d)[0] == 0.0


def test_box_times_zero_with_infinite_endpoint_has_no_nan_bounds():
    out = _mul_sets_by_constant([Box([0.0, 1.0], [np.inf, 2.0])],
                                np.array([0.0, 1.0]))[0]
    assert np.array_equal(out.lb.ravel(), [0.0, 1.0])
    assert np.array_equal(out.ub.ravel(), [0.0, 2.0])
    out = _mul_boxes(Box([0.0], [1.0]), Box([1.0], [np.inf]))
    assert out.lb[0, 0] == 0.0 and out.ub[0, 0] == np.inf
    out = _mul_boxes(Box([0.0], [0.0]), Box([-np.inf], [np.inf]))
    assert out.lb[0, 0] == 0.0 and out.ub[0, 0] == 0.0
