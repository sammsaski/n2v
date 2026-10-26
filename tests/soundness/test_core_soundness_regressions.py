"""Soundness regressions for the core (non-refine) fixes of the 2026-09 hunts.

Each test pins one measured failure mode with a minimal reproduction:

  * LP emptiness certificates and solver failures (fixes 4, 5, 10, 12): a
    solver 'infeasible' is not a certificate at tiny scales (presolve
    artefact), and failure statuses must raise ``LPFailure`` or degrade the
    verdict to UNKNOWN, never read as "empty".
  * Outward interval enclosures (8, 15, 19): ``estimate_ranges`` encloses the
    exact rational interval and the exact-LP range.
  * Outward spec rows (12, including the touching case): a set that touches
    the unsafe region at exact margin 0 must not be rounded into "disjoint",
    also when the products underflow into the subnormal range.
  * Infinite predicate boxes (13): ``0 * inf`` must not poison interval
    bounds; ``ensure_finite_predicate_box`` finitizes over the full polytope.
  * The zonotope ReLU error generator (15) is kept when it is small.
  * ImageStar specs are checked in CHW order (16).
  * Box-derived stars carry the raw float64 input bounds, and the falsify
    wrapper snaps attack points into the float64 box (18b).
  * Falsification accepts outputs at zero tolerance (18c).
  * VNNLIB 2.0 equality bounds intersect with earlier bounds (20).
"""

import importlib
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import reach_pytorch_model
from n2v.sets.box import Box
from n2v.sets.halfspace import HalfSpace
from n2v.sets.image_star import ImageStar
from n2v.sets.predicate_box import ensure_finite_predicate_box
from n2v.sets.star import Star
from n2v.sets.zono import Zono
from n2v.utils.lpsolver import (LPFailure, check_feasibility,
                                confirm_infeasible, solve_lp_no_presolve)
from n2v.utils.ns_certificate import outward_interval_slack, outward_spec_rows
from n2v.utils.verify_specification import (_verify_specification_star,
                                            verify_specification)


# ---------------------------------------------------------------------------
# LP emptiness certificates
# ---------------------------------------------------------------------------

def test_confirm_infeasible_decides_plain_cases():
    A = np.eye(2)
    lb, ub = np.zeros(2), np.ones(2)
    assert not confirm_infeasible(A, np.array([1.0, 1.0]), lb, ub)
    assert confirm_infeasible(A, np.array([-1.0, 1.0]), lb, ub)
    assert check_feasibility(A=A, b=np.array([1.0, 1.0]), lb=lb, ub=ub)
    assert not check_feasibility(A=A, b=np.array([-1.0, 1.0]), lb=lb, ub=ub)


def test_confirm_infeasible_respects_equalities():
    # x0 = 0.5 with x0 <= 0.25 : infeasible; x0 = 0.5 with x0 <= 0.75 : feasible
    A = np.array([[1.0, 0.0]])
    Aeq = np.array([[1.0, 0.0]])
    lb, ub = np.zeros(2), np.ones(2)
    assert confirm_infeasible(A, np.array([0.25]), lb, ub, Aeq=Aeq, beq=np.array([0.5]))
    assert not confirm_infeasible(A, np.array([0.75]), lb, ub, Aeq=Aeq, beq=np.array([0.5]))


@pytest.mark.parametrize("scale", [1e-6, 1e-7, 1e-8, 1e-9])
def test_known_feasible_tiny_scale_systems_are_never_declared_empty(scale):
    """Every system here contains a known point, so 'empty' is wrong by
    construction. Under default presolve HiGHS mislabels a fraction of them
    at these scales (measured 5-75 / 300); the robust path must not."""
    rng = np.random.default_rng(0)
    n_wrong = 0
    for _ in range(60):
        n, m = 6, 8
        x0 = rng.uniform(-1, 1, n) * scale
        A = rng.standard_normal((m, n))
        b = A @ x0 + rng.uniform(0.0, 1.0, m) * scale       # x0 strictly feasible
        lb = x0 - rng.uniform(0.0, 1.0, n) * scale
        ub = x0 + rng.uniform(0.0, 1.0, n) * scale
        if not check_feasibility(A=A, b=b, lb=lb, ub=ub):
            n_wrong += 1
    assert n_wrong == 0


def test_infeasible_by_a_small_margin_is_still_certified():
    """A pinned child infeasible by 1e-6 (the contraction cross-check's
    'genuinely infeasible' kind) must be certified empty: the Farkas/NS
    certificate works at any margin, and a per-row 1e-7 slack that compounds
    to ~n*1e-7 must not veto it."""
    rng = np.random.default_rng(7)
    n_in, n_missed = 3, 0
    for _ in range(40):
        nvar = n_in + int(rng.integers(0, 4))
        lb, ub = -np.ones(nvar), np.ones(nvar)
        ub[n_in:] = rng.uniform(0.1, 2.0, nvar - n_in); lb[n_in:] = 0.0
        g = rng.normal(size=nvar); g[n_in:] = 0.0
        c = -float(np.abs(g).sum()) - 1e-6          # g.a >= sum|g| + 1e-6: impossible
        A = np.vstack([np.eye(nvar)[:n_in], -np.eye(nvar)[:n_in], -g[None, :]])
        b = np.concatenate([np.ones(2 * n_in), [c]])
        if not confirm_infeasible(A, b, lb, ub):
            n_missed += 1
        assert not check_feasibility(A=A, b=b, lb=lb, ub=ub) or n_missed
    assert n_missed == 0


def test_solve_lp_no_presolve_returns_optimum():
    pytest.importorskip("highspy")
    A = np.eye(2)
    x, fval, st, _ = solve_lp_no_presolve(np.ones(2), A, np.ones(2),
                                          np.zeros(2), np.ones(2))
    assert st == "optimal"
    assert np.allclose(x, 0.0) and abs(fval) < 1e-12


def test_verify_specification_box_confirms_before_disjoint():
    from n2v.utils.verify_specification import _verify_specification_box
    box = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))
    # unsafe region y0 <= -1 : disjoint from the box
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-1.0]]))
    assert _verify_specification_box(box, hs) is True
    # unsafe region y0 <= 0.5 : intersects
    hs2 = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.5]]))
    assert _verify_specification_box(box, hs2) is False


# ---------------------------------------------------------------------------
# LP failure is not a certificate of emptiness
# ---------------------------------------------------------------------------

_FAILURE_STATUSES = ["unbounded", "failed: iteration limit", "error: numerical"]


@pytest.mark.parametrize("status", _FAILURE_STATUSES)
def test_check_feasibility_raises_on_failure(status, monkeypatch):
    import n2v.utils.lpsolver as lpmod

    # Both solves fail: the careful solve that decides on its own when it can
    # certify, and the default solve it falls back to.
    monkeypatch.setattr(lpmod, "_feasibility_robust",
                        lambda *a, **k: ("failed: simulated", None))
    monkeypatch.setattr(lpmod, "solve_lp",
                        lambda *a, **k: (None, None, status, None))
    with pytest.raises(LPFailure):
        lpmod.check_feasibility(A=np.array([[1.0]]), b=np.array([1.0]),
                                lb=np.array([-1.0]), ub=np.array([1.0]))


def test_check_feasibility_still_reports_genuine_infeasibility(monkeypatch):
    import n2v.utils.lpsolver as lpmod

    monkeypatch.setattr(lpmod, "solve_lp",
                        lambda *a, **k: (None, None, "infeasible", None))
    # x <= -2 with x >= -1: infeasible by a margin of 1 (robustly confirmed)
    assert lpmod.check_feasibility(A=np.array([[1.0]]), b=np.array([-2.0]),
                                   lb=np.array([-1.0]),
                                   ub=np.array([1.0])) is False
    # x <= 1 with x in [-1, 1] is FEASIBLE: a solver 'infeasible' on it is a
    # presolve artefact and must NOT be reported as empty.
    assert lpmod.check_feasibility(A=np.array([[1.0]]), b=np.array([1.0]),
                                   lb=np.array([-1.0]),
                                   ub=np.array([1.0])) is True


def test_verify_specification_box_raises_on_failure(monkeypatch):
    import n2v.utils.verify_specification as vsmod

    class _Res:
        status = 4              # scipy: numerical difficulties
        success = False
        message = "numerical difficulties encountered"

    from n2v.sets import Box, HalfSpace

    monkeypatch.setattr(vsmod, "linprog", lambda *a, **k: _Res())
    box = Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    # Two rows, each individually feasible over the box, so the routine
    # reaches the simultaneous-feasibility LP that we stubbed.
    hs = HalfSpace(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.5, 0.5]))
    with pytest.raises(LPFailure):
        vsmod._verify_specification_box(box, hs)


def test_verify_specification_degrades_to_unknown_on_lp_failure(monkeypatch,
                                                                caplog):
    import logging

    import n2v.utils.verify_specification as vs
    from n2v.sets import HalfSpace

    def boom(group, reach_set):
        raise LPFailure("simulated solver failure")

    monkeypatch.setattr(vs, "_group_disjoint_from_reach_set", boom)
    S = Star.from_bounds(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([-5.0]))  # would be disjoint
    with caplog.at_level(logging.WARNING, logger=vs.__name__):
        res = vs.verify_specification([S], hs)
    assert "LP did not decide" in caplog.text
    assert res.verdict == "UNKNOWN"


def test_lp_failure_status_never_reads_as_infeasible():
    """The HiGHS wrapper used to collapse EVERY non-optimal status into
    'infeasible' -- so a solver failure (e.g. a NaN-poisoned model) became
    an emptiness certificate. Failures must surface as failures."""
    from n2v.utils.lpsolver import solve_lp

    _, _, status, _ = solve_lp(
        np.array([1.0]), A=None, b=None,
        lb=np.array([np.nan]), ub=np.array([1.0]))
    assert status != 'infeasible'


# ---------------------------------------------------------------------------
# Interval enclosures
# ---------------------------------------------------------------------------

def _exact_interval(V, lb, ub):
    """Exact rational interval bound of c + V alpha over the box."""
    dim, ncol = V.shape
    lo, hi = [], []
    for i in range(dim):
        c = Fraction(float(V[i, 0]))
        a = c
        b = c
        for j in range(ncol - 1):
            g = Fraction(float(V[i, 1 + j]))
            l = Fraction(float(lb[j]))
            u = Fraction(float(ub[j]))
            a += min(g * l, g * u)
            b += max(g * l, g * u)
        lo.append(a)
        hi.append(b)
    return lo, hi


@pytest.mark.parametrize("seed", range(12))
def test_estimate_ranges_encloses_exact_rational_interval(seed):
    """The float interval must contain the exact rational interval computed
    from the same float inputs: lb_float <= lb_exact and ub_float >= ub_exact.
    Without outward inflation this fails on a measurable fraction of random
    stars by a few ulps."""
    rng = np.random.default_rng(seed)
    dim, nVar = 7, 25
    # ill-conditioned mixes: large and tiny generators, cancellation
    V = rng.standard_normal((dim, nVar + 1))
    V[:, 1:] *= 10.0 ** rng.integers(-6, 7, size=(dim, nVar))
    lb = -rng.uniform(0.1, 3.0, size=nVar)
    ub = rng.uniform(0.1, 3.0, size=nVar)
    S = Star(V, np.zeros((0, nVar)), np.zeros((0, 1)),
             lb.reshape(-1, 1), ub.reshape(-1, 1))
    flb, fub = S.estimate_ranges()
    elb, eub = _exact_interval(V, lb, ub)
    for i in range(dim):
        assert Fraction(float(flb[i, 0])) <= elb[i], (seed, i)
        assert Fraction(float(fub[i, 0])) >= eub[i], (seed, i)
        a, b = S.estimate_range(i)
        assert Fraction(a) <= elb[i] and Fraction(b) >= eub[i]


def test_outward_slack_is_tiny_nonnegative_and_inf_safe():
    g = np.array([[1.0, -2.0, 0.0], [0.5, 0.5, 3.0]])
    lb = np.array([-1.0, -1.0, -np.inf])
    ub = np.array([1.0, 1.0, np.inf])
    s = outward_interval_slack(np.array([0.1, 0.2]), g, lb, ub)
    assert s[0] >= 0.0 and np.isfinite(s[0]) and s[0] < 1e-12
    assert np.isinf(s[1])  # nonzero generator against an infinite bound
    # finite case is a few ulps relative
    g2 = np.ones((1, 100))
    s2 = outward_interval_slack(np.array([0.0]), g2, -np.ones(100), np.ones(100))
    assert 0.0 < s2[0] < 1e-11


def test_interval_bound_encloses_exact_lp_on_constrained_stars():
    """estimate_ranges (box interval) must enclose get_ranges (exact LP over
    the polytope) -- the property every neuron classification relies on."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        nVar, dim = 6, 5
        V = rng.standard_normal((dim, nVar + 1))
        C = rng.standard_normal((4, nVar))
        d = np.abs(rng.standard_normal((4, 1))) + 0.5
        S = Star(V, C, d, -np.ones((nVar, 1)), np.ones((nVar, 1)))
        ilb, iub = S.estimate_ranges()
        glb, gub = S.get_ranges()
        # get_ranges does not pad: the tolerance absorbs the LP optimum's own
        # 1e-7 solver tolerance.
        pad_l = 1e-7 * (1.0 + np.abs(glb.ravel()))
        pad_u = 1e-7 * (1.0 + np.abs(gub.ravel()))
        assert np.all(ilb.ravel() <= glb.ravel() + pad_l + 1e-12)
        assert np.all(iub.ravel() >= gub.ravel() - pad_u - 1e-12)


def test_estimate_ranges_is_nan_free_with_infinite_box():
    V = np.array([[1.0, 1.0, 0.0],
                  [2.0, 0.0, 1.0],
                  [3.0, 0.0, 0.0]])          # dim 3, nVar 2; row 2 ignores var 1
    C = np.array([[1.0, 0.0], [-1.0, 0.0]])
    d = np.array([[1.0], [1.0]])
    S = Star(V, C, d, np.array([[-np.inf], [-1.0]]), np.array([[np.inf], [1.0]]))
    lb, ub = S.estimate_ranges()
    assert not np.isnan(lb).any() and not np.isnan(ub).any()
    assert lb[0, 0] == -np.inf and ub[0, 0] == np.inf          # truthful
    assert np.isfinite(lb[1, 0]) and np.isfinite(ub[1, 0])
    assert lb[1, 0] <= 1.0 <= ub[1, 0]
    assert lb[2, 0] <= 3.0 <= ub[2, 0]                         # zero row: finite


# ---------------------------------------------------------------------------
# Zonotope ReLU keeps the error generator
# ---------------------------------------------------------------------------

def test_zono_relu_keeps_small_error_generator():
    from n2v.nn.layer_ops.relu_reach import relu_zono_approx
    # neuron 0 crosses zero with a tiny range -> err ~ 2.5e-11 (< old 1e-10 cut)
    c = np.array([[0.0], [1.0]])
    V = np.array([[1e-10, 0.0], [0.0, 1.0]])
    Z = Zono(c, V)
    R = relu_zono_approx([Z])[0]
    # neuron 0 must own a NEW nonzero generator (neuron 1, whose exact lower
    # bound is 0, may legitimately get one too under the outward enclosure)
    assert R.V.shape[1] >= Z.V.shape[1] + 1, "error generator was dropped"
    assert np.any(R.V[0, Z.V.shape[1]:] != 0.0), "neuron 0 error generator dropped"
    # semantic check: the point x = 0 maps to relu(0) = 0, which the
    # generator-less (line) approximation cannot represent.
    lb, ub = R.get_bounds()
    assert lb[0, 0] <= 0.0 <= ub[0, 0]
    assert ub[0, 0] >= 1e-10


# ---------------------------------------------------------------------------
# Outward spec rows (the touching case)
# ---------------------------------------------------------------------------

def test_outward_spec_rows_never_exclude_a_touching_point():
    """Place a box corner ``a`` exactly on the closed unsafe region
    ``H (c + V' a) <= g`` (``g`` is the exact rational value rounded up to a
    float) and check, in exact arithmetic, that the rows returned by
    ``outward_spec_rows`` still admit it. Plain rows ``(H V', g - H c)``
    exclude the touching corner on 168 of these 400 trials; the outward rows
    on none."""
    rng = np.random.default_rng(0)
    n_excluded = 0
    for _ in range(400):
        dim, nvar, k = 4, 5, 2
        V = (rng.standard_normal((dim, nvar + 1))
             * 10.0 ** rng.integers(-3, 4, size=(dim, nvar + 1)))
        H = rng.standard_normal((k, dim))
        lb, ub = -np.ones(nvar), np.ones(nvar)
        a = rng.choice([-1.0, 1.0], size=nvar)
        exact = [sum(Fraction(float(H[r, i]))
                     * (Fraction(float(V[i, 0]))
                        + sum(Fraction(float(V[i, 1 + j])) * Fraction(float(a[j]))
                              for j in range(nvar)))
                     for i in range(dim))
                 for r in range(k)]
        g = np.array([np.nextafter(float(e), np.inf) if Fraction(float(e)) < e
                      else float(e) for e in exact])
        assert all(Fraction(float(g[r])) >= exact[r] for r in range(k))
        H_alpha, rhs = outward_spec_rows(H, g, V, lb, ub)
        lhs = [sum(Fraction(float(H_alpha[r, j])) * Fraction(float(a[j]))
                   for j in range(nvar))
               for r in range(k)]
        if any(lhs[r] > Fraction(float(rhs[r])) for r in range(k)):
            n_excluded += 1
    assert n_excluded == 0


def test_verify_specification_touching_region_is_not_unsat():
    """Exactly, ``H y = 3 - alpha <= 2`` iff ``alpha >= 1``: the star touches
    the unsafe region at ``alpha = 1``, so UNSAT is wrong. In float64,
    ``H @ c`` cancels ``2**53 + 3 - 2**53`` to 4, and plain spec rows then
    prove a disjointness that does not hold (a false UNSAT); the outward rows
    must not."""
    c = np.array([[2.0**53], [3.0], [2.0**53]])
    V = np.hstack([c, np.array([[0.0], [0.0], [1.0]])])
    S = Star(V, np.zeros((0, 1)), np.zeros((0, 1)),
             -np.ones((1, 1)), np.ones((1, 1)))
    H = np.array([[1.0, 1.0, -1.0]])
    g = np.array([[2.0]])
    # Exact arithmetic says "not UNSAT" whatever the BLAS summation order does,
    # so the assertion holds unconditionally; the rounding only makes it bite.
    assert verify_specification([S], HalfSpace(H, g)).verdict != "UNSAT"


def test_outward_spec_rows_admit_a_touching_point_with_subnormal_products():
    """The touching test above at a scale where every product of ``H`` with
    ``V`` underflows into the subnormal range. There a product carries an
    absolute error of up to ``eta / 2`` and the relative slack
    ``gamma * magnitude`` rounds to 0, so the rows must also charge the
    underflow. Without that term the corner is excluded in most trials."""
    rng = np.random.default_rng(1)
    s = 2.0 ** -537
    eta = Fraction(2) ** -1074
    n_excluded = 0
    for _ in range(200):
        dim, nvar, k = 3, 2, 2
        # Products are half-integer multiples of eta: many round (ties
        # included) when computed in float64.
        V = rng.integers(-9, 10, size=(dim, nvar + 1)) * 0.5 * s
        H = rng.integers(-9, 10, size=(k, dim)) * 1.0 * s
        lb, ub = -np.ones(nvar), np.ones(nvar)
        a = rng.choice([-1.0, 1.0], size=nvar)
        exact = [sum(Fraction(float(H[r, i]))
                     * (Fraction(float(V[i, 0]))
                        + sum(Fraction(float(V[i, 1 + j])) * Fraction(float(a[j]))
                              for j in range(nvar)))
                     for i in range(dim))
                 for r in range(k)]
        # the smallest float at or above the exact value (a multiple of eta)
        g = np.array([float(-((-e) // eta) * eta) for e in exact])
        assert all(Fraction(float(g[r])) >= exact[r] for r in range(k))
        H_alpha, rhs = outward_spec_rows(H, g, V, lb, ub)
        lhs = [sum(Fraction(float(H_alpha[r, j])) * Fraction(float(a[j]))
                   for j in range(nvar))
               for r in range(k)]
        if any(lhs[r] > Fraction(float(rhs[r])) for r in range(k)):
            n_excluded += 1
    assert n_excluded == 0


@pytest.mark.parametrize("nvar", [0, 1])
def test_verify_specification_subnormal_touching_point_is_not_unsat(nvar):
    """Exactly, ``H c = 2 * 1.5 * 2**-1074 = g``: the star touches the unsafe
    region. In float64 each product rounds (half to even) to ``2 * 2**-1074``,
    so ``g - H c`` is ``-2**-1074``; the interval pre-check of
    ``_verify_specification_star`` then called the star disjoint (a false
    UNSAT) because the relative slack of the spec rows underflowed to 0."""
    eta = 2.0 ** -1074
    H = np.array([[2.0 ** -537, 2.0 ** -537]])
    c = np.full((2, 1), 1.5 * 2.0 ** -537)
    g = np.array([[3 * eta]])
    exact = sum(Fraction(float(H[0, i])) * Fraction(float(c[i, 0]))
                for i in range(2))
    assert exact == Fraction(float(g[0, 0]))
    S = Star(np.hstack([c, np.zeros((2, nvar))]), np.zeros((0, nvar)),
             np.zeros((0, 1)), -np.ones((nvar, 1)), np.ones((nvar, 1)))
    assert _verify_specification_star(S, HalfSpace(H, g)) is False
    assert verify_specification([S], HalfSpace(H, g)).verdict != "UNSAT"


# ---------------------------------------------------------------------------
# ImageStar specs use CHW order
# ---------------------------------------------------------------------------

def test_imagestar_spec_uses_chw_order():
    # H=1, W=2, C=2, one generator; the centre of (w, c) is 10 (2c + w), so
    # the CHW flattening is [0, 10, 20, 30] and the HWC one [0, 20, 10, 30].
    V = np.zeros((1, 2, 2, 2))
    for w in range(2):
        for c in range(2):
            V[0, w, c, 0] = 10.0 * (2 * c + w)
    V[..., 1] = 0.5
    IS = ImageStar(V, np.zeros((0, 1)), np.zeros((0, 1)),
                   -np.ones(1), np.ones(1))
    # y1 <= 12: CHW y1 is in [9.5, 10.5] (meets it), HWC y1 in [19.5, 20.5]
    hs = HalfSpace(np.array([[0.0, 1.0, 0.0, 0.0]]), np.array([[12.0]]))
    assert _verify_specification_star(IS.to_star(), hs) is True
    assert verify_specification([IS], hs).verdict != "UNSAT"


# ---------------------------------------------------------------------------
# Infinite predicate boxes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def inf_box_star():
    """alpha_0 in [-1, 1] carried ONLY by C/d rows (predicate box +/-inf on
    that dim), alpha_1 in [-1, 1] by the box."""
    V = np.hstack([np.zeros((2, 1)), np.eye(2)])
    C = np.array([[1.0, 0.0], [-1.0, 0.0]])          # -1 <= alpha_0 <= 1
    d = np.array([[1.0], [1.0]])
    return Star(V, C, d,
                np.array([[-np.inf], [-1.0]]), np.array([[np.inf], [1.0]]))


def test_finite_box_passes_through_by_identity(inf_box_star):
    S = inf_box_star
    S_fin = Star(S.V, S.C, S.d, np.array([[-1.0], [-1.0]]),
                 np.array([[1.0], [1.0]]))
    S2, empty = ensure_finite_predicate_box(S_fin)
    assert S2 is S_fin and not empty


def test_finitized_box_is_outward_of_the_polytope(inf_box_star):
    S = inf_box_star
    S2, empty = ensure_finite_predicate_box(S)
    assert not empty
    lb = S2.predicate_lb.flatten()
    ub = S2.predicate_ub.flatten()
    assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))
    # dim 0's true range is [-1, 1] (from C/d): outward containment, tight
    assert lb[0] <= -1.0 <= 1.0 <= ub[0]
    assert lb[0] == pytest.approx(-1.0, abs=1e-6)
    assert ub[0] == pytest.approx(1.0, abs=1e-6)
    # dim 1's finite bounds are untouched
    assert lb[1] == -1.0 and ub[1] == 1.0


def test_none_box_is_bounded_from_cd_rows(inf_box_star):
    S = inf_box_star
    C = np.vstack([S.C, [[0.0, 1.0], [0.0, -1.0]]])  # also bound alpha_1
    d = np.vstack([S.d, [[0.5], [0.5]]])
    S_none = Star(S.V, C, d)                          # no predicate box at all
    S2, empty = ensure_finite_predicate_box(S_none)
    assert not empty
    assert np.all(np.isfinite(S2.predicate_lb))
    assert np.all(np.isfinite(S2.predicate_ub))
    assert S2.predicate_ub.flatten()[1] == pytest.approx(0.5, abs=1e-6)


def test_unbounded_dim_raises(inf_box_star):
    S = inf_box_star
    S_unb = Star(S.V, np.zeros((0, 2)), np.zeros((0, 1)),
                 S.predicate_lb.copy(), S.predicate_ub.copy())
    with pytest.raises(ValueError, match="unbounded"):
        ensure_finite_predicate_box(S_unb)


def test_nan_box_entries_raise(inf_box_star):
    """NaN box entries are corruption, not intervals: they used to make the
    LP wrapper fail in a way that masqueraded as an emptiness certificate
    (vacuous UNSAT on a ground-truth-SAT star). They must raise."""
    S = inf_box_star
    plb = S.predicate_lb.copy()
    plb[0, 0] = np.nan
    S_nan = Star(S.V, S.C, S.d, plb, S.predicate_ub.copy())
    with pytest.raises(ValueError, match="NaN"):
        ensure_finite_predicate_box(S_nan)


def test_nan_cd_rows_raise(inf_box_star):
    S = inf_box_star
    C_bad = S.C.copy()
    C_bad[0, 0] = np.nan
    S_bad = Star(S.V, C_bad, S.d, S.predicate_lb.copy(),
                 S.predicate_ub.copy())
    with pytest.raises(ValueError, match="non-finite"):
        ensure_finite_predicate_box(S_bad)


def test_mixed_none_box_is_bounded(inf_box_star):
    """Docstring-promised: one side None, the other finite. Every dim must
    be bounded below by a C/d row for the None side to be recoverable."""
    S = inf_box_star
    C = np.vstack([S.C, [[0.0, -1.0]]])           # also alpha_1 >= -0.5
    d = np.vstack([S.d, [[0.5]]])
    S_mix = Star(S.V, C, d, None, np.array([[1.0], [1.0]]))
    S2, empty = ensure_finite_predicate_box(S_mix)
    assert not empty
    assert np.all(np.isfinite(S2.predicate_lb))
    assert S2.predicate_lb.flatten()[0] == pytest.approx(-1.0, abs=1e-6)
    assert S2.predicate_lb.flatten()[1] == pytest.approx(-0.5, abs=1e-6)


def test_infeasible_polytope_is_reported_empty_and_reaches_nothing(inf_box_star):
    S = inf_box_star
    C_bad = np.array([[1.0, 0.0], [-1.0, 0.0]])
    d_bad = np.array([[-2.0], [1.0]])                 # alpha_0 <= -2 & >= -1
    S_emp = Star(S.V, C_bad, d_bad, S.predicate_lb.copy(),
                 S.predicate_ub.copy())
    _, empty = ensure_finite_predicate_box(S_emp)
    assert empty
    # y = relu(x0) + relu(x1); an empty input set reaches nothing
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.eye(2, dtype=torch.float64))
        net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[1.0, 1.0]], dtype=torch.float64))
        net[2].bias.zero_()
    for method in ("approx", "exact"):
        assert reach_pytorch_model(net, S_emp, method=method) == [], method


@pytest.mark.parametrize("method", ["approx", "exact"])
def test_plain_reach_with_infinite_predicate_box_is_an_enclosure(method):
    """A star whose box is (-inf, inf) but whose C/d rows bound it must reach
    a set that contains every sampled image (it used to reach an unsound set
    and yield false UNSAT)."""
    from n2v.nn.reach import reach_pytorch_model
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, 2)).double()
    V = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    C = np.vstack([np.eye(2), -np.eye(2)])
    d = np.ones((4, 1))
    S = Star(V, C, d, np.full((2, 1), -np.inf), np.full((2, 1), np.inf))
    out = reach_pytorch_model(net, S, method=method)
    sets = out if isinstance(out, list) else [out]
    lbs, ubs = zip(*[T.estimate_ranges() for T in sets])
    lo = np.min(np.hstack(lbs), axis=1)
    hi = np.max(np.hstack(ubs), axis=1)
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (500, 2))
    with torch.no_grad():
        Y = net(torch.as_tensor(X)).numpy()
    assert np.all(Y.min(axis=0) >= lo - 1e-9)
    assert np.all(Y.max(axis=0) <= hi + 1e-9)


# ---------------------------------------------------------------------------
# Raw input bounds and the falsify snap into the float64 box
# ---------------------------------------------------------------------------

def test_box_to_star_carries_raw_bounds():
    rng = np.random.default_rng(3)
    lb = rng.uniform(-1, 1, 5) * 0.37
    ub = lb + rng.uniform(0.01, 0.3, 5) * 0.91
    S = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))
    assert np.array_equal(S._raw_lb, lb) and np.array_equal(S._raw_ub, ub)


def test_falsify_snaps_attack_point_into_float64_box(monkeypatch):
    """The attacks work in float32, so the point they return can sit outside
    the float64 box by rounding. The wrapper snaps it into the box and keeps
    it only if the snapped point still satisfies the property."""
    # a plain ``import n2v.utils.falsify`` resolves to the re-exported
    # function, not the module
    fz = importlib.import_module("n2v.utils.falsify")
    model = nn.Sequential(nn.Linear(1, 1)).double()
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.zero_()
    lb, ub = np.array([0.0]), np.array([0.1])
    x_out = np.array([float(np.float32(0.1))])        # 1.49e-9 above ub
    assert x_out[0] > ub[0]
    monkeypatch.setattr(fz, "_falsify_impl",
                        lambda *a, **k: (0, (x_out, np.array([0.1]))))
    # unsafe: y >= 0.05, which the snapped point still satisfies
    prop = HalfSpace(np.array([[-1.0]]), np.array([[-0.05]]))
    result, cex = fz.falsify(model, lb, ub, prop, method="random")
    assert result == 0
    x = np.asarray(cex[0]).ravel()
    assert np.all(lb <= x) and np.all(x <= ub)
    # unsafe: y >= 0.1000000001, which only the out-of-box point satisfies
    prop2 = HalfSpace(np.array([[-1.0]]), np.array([[-0.1000000001]]))
    assert fz.falsify(model, lb, ub, prop2, method="random") == (2, None)


# ---------------------------------------------------------------------------
# Falsification acceptance at zero tolerance
# ---------------------------------------------------------------------------

def test_falsify_output_check_is_zero_tolerance():
    from n2v.utils.falsify import _output_satisfies_property
    hs = HalfSpace(np.array([[1.0]]), np.array([[0.0]]))    # unsafe: y <= 0
    assert _output_satisfies_property(np.array([[0.0]]), [[hs]])
    assert _output_satisfies_property(np.array([[-1e-12]]), [[hs]])
    assert not _output_satisfies_property(np.array([[5e-9]]), [[hs]])


# ---------------------------------------------------------------------------
# VNNLIB 2.0 bound application intersects
# ---------------------------------------------------------------------------

def test_vnnlib2_equality_intersects_with_prior_bounds():
    from n2v.utils.vnnlib2 import _apply_bounds
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    _apply_bounds(lb, ub, [(0, "upper", 0.25), (0, "both", 0.5)])
    assert lb[0] == 0.5 and ub[0] == 0.25          # contradictory -> empty, not widened
    lb, ub = np.array([-1.0]), np.array([1.0])
    _apply_bounds(lb, ub, [(0, "both", 0.5)])
    assert lb[0] == 0.5 and ub[0] == 0.5
