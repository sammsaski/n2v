"""Soundness of the LP certificates of ``n2v.utils.lpsolver``.

Two promises that every Star kernel rests on, checked against exact
references in ``fractions`` (``tests/_oracles/rational.py``), never against
another n2v path:

* **An LP optimum used as a bound holds the true optimum.** Over a finite box
  every optimum of ``solve_lp``, ``solve_lp_batch``, ``solve_lp_no_presolve``
  and the Star range queries is the more outward of the solver's value and a
  weak-duality bound (``_outward_optimum``, ``_outward_optima``); the LP
  ranges a relaxation is built on are padded outward (``_lp_pad``,
  ``_outward_lp_ranges``) and solved with HiGHS (``_bound_lp_solver``). The
  oracle is the exact optimum by vertex enumeration, compared one-sided with
  zero slack, on data whose optimum is not a float64 sum, so that a solver
  value rounded to nearest lands inside about half the time.
* **A system is called empty only when it is empty.** ``check_feasibility``,
  ``confirm_infeasible`` and the public ``Star.is_empty_set`` and
  ``Star.contains`` (``_can_certify``, ``_stacked_rows``,
  ``_careful_emptiness``, ``_feasibility_robust``,
  ``_farkas_certified_empty``) must never call a feasible system empty, on
  every solver route: systems feasible in one point, by 1e-9, by 1e-13, on a
  row of large magnitude, with equality rows of both signs, with rows of
  different binary scales, with no box, a partly infinite box and no rows,
  each with its feasible point checked in fractions. Systems that are empty by
  a clear margin (checked by an exact refutation) must be shown empty. A solver
  that reports 'infeasible' wrongly or fails is simulated (a stubbed solve) so
  that the cases do not depend on a host where the solver errs: an
  'infeasible' whose multipliers prove nothing, an unconfirmed default solve
  and an undecided solve must never become emptiness.

Convention: SAT = unsafe, UNSAT = safe; a false 'empty' deletes a piece of a
reach result and is a false UNSAT.
"""

import warnings
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest

import n2v.utils.lpsolver as lpmod
from n2v.config import config
from n2v.sets import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.utils.lpsolver import (LPFailure, _bound_lp_solver,
                                _farkas_certified_empty, _outward_lp_ranges,
                                _outward_optima, _outward_optimum,
                                check_feasibility, confirm_infeasible,
                                solve_lp, solve_lp_batch,
                                solve_lp_no_presolve)
from tests._oracles.rational import decide, dot, frac, vertices
from tests.known_bugs import known_bug


# --------------------------------------------------------------------------- #
# Exact references                                                              #
# --------------------------------------------------------------------------- #
def _box_rows(lb, ub, n):
    """The finite entries of a box as rows ``e_j x <= ub_j``, ``-e_j x <=
    -lb_j``, in fractions."""
    rows, rhs = [], []
    for j in range(n):
        if ub is not None and np.isfinite(ub[j]):
            e = [F(0)] * n
            e[j] = F(1)
            rows.append(e)
            rhs.append(frac(ub[j]))
        if lb is not None and np.isfinite(lb[j]):
            e = [F(0)] * n
            e[j] = F(-1)
            rows.append(e)
            rhs.append(-frac(lb[j]))
    return rows, rhs


def _exact_point_is_feasible(s, x):
    """``x`` (fractions) satisfies every row, equality and box entry of the
    system ``s`` exactly."""
    n = len(x)
    ok = True
    if s.A is not None:
        ok &= all(dot([frac(v) for v in r], x) <= frac(bi)
                  for r, bi in zip(s.A, s.b))
    if s.Aeq is not None:
        ok &= all(dot([frac(v) for v in r], x) == frac(bi)
                  for r, bi in zip(s.Aeq, s.beq))
    rows, rhs = _box_rows(s.lb, s.ub, n)
    ok &= all(dot(r, x) <= ri for r, ri in zip(rows, rhs))
    return ok


def _exact_is_empty(s):
    """True when the system ``s`` has no point: ``decide`` returns a
    refutation that it has checked itself."""
    n = (s.A if s.A is not None else s.Aeq).shape[1]
    rows = [] if s.A is None else [[frac(v) for v in r] for r in s.A]
    rhs = [] if s.A is None else [frac(v) for v in s.b]
    brows, brhs = _box_rows(s.lb, s.ub, n)
    # an inverted interval lb_j > ub_j is empty by its two box rows
    eq = [] if s.Aeq is None else [[frac(v) for v in r] for r in s.Aeq]
    beq = [] if s.Aeq is None else [frac(v) for v in s.beq]
    x, proof = decide(eq, beq, rows + brows, rhs + brhs)
    return x is None and proof is not None


def _exact_range(A, b, lb, ub, f):
    """Exact ``(min, max)`` of ``f x`` over ``{A x <= b, lb <= x <= ub}``
    (finite box) by vertex enumeration."""
    n = len(f)
    rows = [] if A is None else [[frac(v) for v in r] for r in A]
    rhs = [] if A is None else [frac(v) for v in b]
    brows, brhs = _box_rows(lb, ub, n)
    V = vertices(rows + brows, rhs + brhs)
    if not V:
        pytest.fail("the reference polytope is empty")
    fv = [frac(v) for v in f]
    vals = [dot(fv, v) for v in V]
    return min(vals), max(vals)


def _exact_ranges_of_rows(A, b, lb, ub, G):
    """Exact ``(min, max)`` of every row of ``G`` over one polytope."""
    n = G.shape[1]
    rows = [] if A is None else [[frac(v) for v in r] for r in A]
    rhs = [] if A is None else [frac(v) for v in b]
    brows, brhs = _box_rows(lb, ub, n)
    V = vertices(rows + brows, rhs + brhs)
    out = []
    for g in G:
        gv = [frac(v) for v in g]
        vals = [dot(gv, v) for v in V]
        out.append((min(vals), max(vals)))
    return out


# --------------------------------------------------------------------------- #
# Systems                                                                       #
# --------------------------------------------------------------------------- #
_S = SimpleNamespace


def _sys(A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, x=None):
    arr = (lambda v: None if v is None else np.asarray(v, dtype=np.float64))
    return _S(A=arr(A), b=arr(b), Aeq=arr(Aeq), beq=arr(beq), lb=arr(lb),
              ub=arr(ub), x=x)


_N, _BIG = 50, 1e10
_UNIT = (np.zeros(2), np.ones(2))
_FEASIBLE = {
    # its only point is the box corner x_j = 2e8, on both rows: a row slack
    # applied inward makes it empty by ~1e-3
    "one point on two rows at 1e10": _sys(
        np.vstack([np.ones(_N), -np.ones(_N)]), [_BIG, -_BIG],
        lb=np.zeros(_N), ub=np.full(_N, _BIG / _N),
        x=[F(_BIG / _N)] * _N),
    # the line x + y / 2 = 100 given as two rows, no box
    "on a row, no box": _sys([[1.0, 0.5], [-1.0, -0.5]], [100.0, -100.0],
                             x=[F(100), F(0)]),
    "on a row, finite box": _sys([[1.0, 0.5], [-1.0, -0.5]], [100.0, -100.0],
                                 lb=[-300.0, -300.0], ub=[300.0, 300.0],
                                 x=[F(100), F(0)]),
    "feasible by 1e-9, finite box": _sys([[1.0], [-1.0]], [1.0 + 1e-9, -1.0],
                                         lb=[-4.0], ub=[4.0], x=[F(1)]),
    "feasible by 1e-13, no box": _sys([[1.0], [-1.0]], [1.0 + 1e-13, -1.0],
                                      x=[F(1)]),
    "on the boundary, finite box": _sys([[1.0], [-1.0]], [1.0, -1.0],
                                        lb=[0.0], ub=[2.0], x=[F(1)]),
    # 8 x <= 1 and x >= 1/8 (row scales 2^4 and 2^-2): the one point 1/8
    "one point on rows of different scales": _sys(
        [[8.0], [-0.25]], [1.0, -0.03125], lb=[-10.0], ub=[10.0],
        x=[F(1, 8)]),
    # x0 + x1 + x2 >= 0.3 over [0, 0.1]^3: in exact arithmetic the corner
    # (0.1, 0.1, 0.1) clears the row by 2.8e-17, the only feasible points
    "non-dyadic corner by 3e-17": _sys(
        [[-1.0, -1.0, -1.0]], [-0.3], lb=np.zeros(3), ub=np.full(3, 0.1),
        x=[F(0.1)] * 3),
    # the line a1 + a2 = 2 as two rows, a1 in [0, inf), a2 in [0, 1]: no
    # certificate can be formed, the status path decides
    "on a row, partly infinite box": _sys(
        [[1.0, 1.0], [-1.0, -1.0]], [2.0, -2.0], lb=[0.0, 0.0],
        ub=[np.inf, 1.0], x=[F(1), F(1)]),
    "box only": _sys(lb=[0.0, -1.0], ub=[1.0, 2.0], x=[F(0), F(0)]),
    # equality rows with a positive and with a negative right-hand side
    "equality a1 + a2 = 1/2 in the unit box": _sys(
        Aeq=[[1.0, 1.0]], beq=[0.5], lb=_UNIT[0], ub=_UNIT[1],
        x=[F(1, 4), F(1, 4)]),
    "equality a1 - a2 = -1/4 and a row, unit box": _sys(
        [[1.0, 1.0]], [1.0], Aeq=[[1.0, -1.0]], beq=[-0.25], lb=_UNIT[0],
        ub=_UNIT[1], x=[F(0), F(1, 4)]),
    "equality a1 + a2 = 3/2, unit box by rows only": _sys(
        np.vstack([np.eye(2), -np.eye(2)]), [1.0, 1.0, 0.0, 0.0],
        Aeq=[[1.0, 1.0]], beq=[1.5], x=[F(3, 4), F(3, 4)]),
}

_EMPTY = {
    "empty by 1/2, finite box": _sys([[1.0, 1.0], [-1.0, -1.0]], [1.0, -1.5],
                                     lb=-3 * np.ones(2), ub=3 * np.ones(2)),
    "empty by 1/2, no box": _sys([[1.0, 1.0], [-1.0, -1.0]], [1.0, -1.5]),
    # 8 x <= 1 and x >= 0.4: the multipliers of the scaled rows must be
    # divided by the row scales to cancel the column
    "rows of different scales": _sys([[8.0], [-0.25]], [1.0, -0.1],
                                     lb=[-10.0], ub=[10.0]),
    "partly infinite box": _sys([[1.0, 1.0]], [-1.0], lb=[0.0, 0.0],
                                ub=[np.inf, 1.0]),
    # an inverted interval [1, 0]: the solver gives no dual ray for it
    "inverted predicate interval": _sys([[1.0, 1.0]], [5.0], lb=[1.0, 0.0],
                                        ub=[0.0, 1.0]),
    "equality a1 + a2 = 3 in the unit box": _sys(
        Aeq=[[1.0, 1.0]], beq=[3.0], lb=_UNIT[0], ub=_UNIT[1]),
    "equality a1 + a2 = -1/2 in the unit box": _sys(
        Aeq=[[1.0, 1.0]], beq=[-0.5], lb=_UNIT[0], ub=_UNIT[1]),
    "equality a1 + a2 = 3 and a row, unit box": _sys(
        [[1.0, -1.0]], [0.25], Aeq=[[1.0, 1.0]], beq=[3.0], lb=_UNIT[0],
        ub=_UNIT[1]),
    "equality a1 + a2 = 3, unit box by rows only": _sys(
        np.vstack([np.eye(2), -np.eye(2)]), [1.0, 1.0, 0.0, 0.0],
        Aeq=[[1.0, 1.0]], beq=[3.0]),
}

# check_feasibility routes, one solver per code path: the highspy batch
# members ('default' resolves to linprog; 'highs' takes the same path) take
# one careful solve with a certificate; highs-ds (highs-ipm takes the same
# path) takes the default scipy solve and then confirm_infeasible; a CVXPY
# solver takes its own default solve, whose undecided statuses raise
# LPFailure (allowed: that is not a verdict). The route without highspy is
# run in each case as well.
_HIGHS_ROUTES = ("default", "highs-ds")
_CVXPY_ROUTES = ("CLARABEL",)


def _check(s, solver):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return check_feasibility(A=s.A, b=s.b, Aeq=s.Aeq, beq=s.beq, lb=s.lb,
                                 ub=s.ub, lp_solver=solver)


def _star_of(s):
    """The system as a Star and a point: with no equality rows the star with
    basis ``[0 | I]`` whose emptiness is the system's; with equality rows the
    star with basis ``[0 | Aeq]`` and the point ``beq``, whose membership is
    the system's feasibility."""
    col = (lambda v: None if v is None else v.reshape(-1, 1))
    if s.Aeq is None:
        n = s.A.shape[1] if s.A is not None else s.lb.size
        V = np.hstack([np.zeros((n, 1)), np.eye(n)])
        point = None
    else:
        n = s.Aeq.shape[1]
        V = np.hstack([np.zeros((s.Aeq.shape[0], 1)), s.Aeq])
        point = s.beq
    C = s.A if s.A is not None else np.zeros((0, n))
    d = s.b.reshape(-1, 1) if s.b is not None else np.zeros((0, 1))
    return Star(V, C, d, col(s.lb), col(s.ub)), point


def _no_highspy(monkeypatch):
    monkeypatch.setattr(lpmod, "_HAS_HIGHSPY", False)


# --------------------------------------------------------------------------- #
# A feasible system is never shown empty                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_FEASIBLE))
def test_a_feasible_system_is_never_shown_empty(name, monkeypatch):
    """Every route returns 'not shown empty' on a system with a point (checked
    in fractions). Catches a row slack applied inward on the certificate path
    (the 1e10 corner) and on the status path (the row with no box), an
    equality written with a wrong sign (a1 + a2 = 1/2 read as a1 + a2 <= 1/2
    and a1 + a2 <= -1/2), a default solve's 'infeasible' or failure taken as
    emptiness, and a solver limit of a CVXPY solver read as emptiness."""
    s = _FEASIBLE[name]
    if not _exact_point_is_feasible(s, s.x):
        pytest.fail(f"the witness of {name!r} is not feasible")
    wrong = []
    if confirm_infeasible(s.A, s.b, s.lb, s.ub, Aeq=s.Aeq, beq=s.beq):
        wrong.append("confirm_infeasible")
    for solver in _HIGHS_ROUTES:
        if _check(s, solver) is not True:
            wrong.append(f"check_feasibility[{solver}]")
    for solver in _CVXPY_ROUTES:
        try:
            if _check(s, solver) is not True:
                wrong.append(f"check_feasibility[{solver}]")
        except LPFailure:
            pass
    star, point = _star_of(s)
    for solver in ("default", "highs-ds"):
        if point is None:
            if star.is_empty_set(lp_solver=solver):
                wrong.append(f"Star.is_empty_set[{solver}]")
        elif not star.contains(point, lp_solver=solver):
            wrong.append(f"Star.contains[{solver}]")
    with monkeypatch.context() as m:
        _no_highspy(m)
        if confirm_infeasible(s.A, s.b, s.lb, s.ub, Aeq=s.Aeq, beq=s.beq):
            wrong.append("confirm_infeasible without highspy")
        if _check(s, "default") is not True:
            wrong.append("check_feasibility without highspy")
    assert not wrong, f"{name}: shown empty by {wrong}"


# --------------------------------------------------------------------------- #
# A clearly empty system is shown empty                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_EMPTY))
def test_a_clearly_empty_system_is_shown_empty(name, monkeypatch):
    """Every HiGHS route shows the system empty (an exact refutation exists).
    Catches a certificate that is no longer formed: the dual ray tried with
    one sign only, the ray of scaled rows multiplied by the row scales, an
    equality whose negated copy keeps the sign of its right-hand side, a
    partly infinite box taken as certifiable, the equality rows left out of
    the confirmation or of the certified first pass (``Star.contains`` of a
    point outside), and an 'infeasible' without a dual ray no longer decided
    on the status path (the inverted interval). CVXPY solvers may refuse with
    LPFailure but never answer 'not empty' wrongly here either."""
    s = _EMPTY[name]
    if not _exact_is_empty(s):
        pytest.fail(f"{name!r} is not empty in exact arithmetic")
    missed = []
    if not confirm_infeasible(s.A, s.b, s.lb, s.ub, Aeq=s.Aeq, beq=s.beq):
        missed.append("confirm_infeasible")
    for solver in _HIGHS_ROUTES:
        if _check(s, solver) is not False:
            missed.append(f"check_feasibility[{solver}]")
    for solver in _CVXPY_ROUTES:
        try:
            if _check(s, solver) is not False:
                missed.append(f"check_feasibility[{solver}]")
        except LPFailure:
            pass
    star, point = _star_of(s)
    for solver in ("default", "highs-ds"):
        if point is None:
            if not star.is_empty_set(lp_solver=solver):
                missed.append(f"Star.is_empty_set[{solver}]")
        elif star.contains(point, lp_solver=solver):
            missed.append(f"Star.contains[{solver}]")
    with monkeypatch.context() as m:
        _no_highspy(m)
        if not confirm_infeasible(s.A, s.b, s.lb, s.ub, Aeq=s.Aeq, beq=s.beq):
            missed.append("confirm_infeasible without highspy")
        if _check(s, "default") is not False:
            missed.append("check_feasibility without highspy")
    assert not missed, f"{name}: not shown empty by {missed}"


# --------------------------------------------------------------------------- #
# A solver that errs is not believed                                            #
# --------------------------------------------------------------------------- #
# Feasible stars and multiplier vectors that a solver might return with a
# wrong 'infeasible'. None of them proves emptiness: (1, 1) on x = 1 has the
# weak-duality bound 0 exactly, with either sign; (1) on x <= -1/2 over
# [-1, 2] has b'y < 0 but certifies only if the box term takes the wrong end
# of the column; (1, 1 + 2^-20) no longer cancels the column and certifies
# only in the same way.
_ONE_POINT = (np.array([[1.0], [-1.0]]), np.array([1.0, -1.0]), [0.0], [2.0])
_HALF_LINE = (np.array([[1.0]]), np.array([-0.5]), [-1.0], [2.0])
_FALSE_RAYS = {
    "bound exactly 0": (_ONE_POINT, [-1.0, -1.0]),
    "bound exactly 0, other sign": (_ONE_POINT, [1.0, 1.0]),
    "multipliers that do not cancel": (_ONE_POINT, [-1.0, -1.0 - 2.0 ** -20]),
    "certifies only with the wrong box end": (_HALF_LINE, [-1.0]),
}


@pytest.mark.parametrize("name", sorted(_FALSE_RAYS))
def test_an_infeasible_report_whose_multipliers_prove_nothing_is_not_believed(
        name, monkeypatch):
    """The careful solve is made to report 'infeasible' with multipliers that
    prove nothing, on a feasible star. The Farkas check must reject them
    (``_farkas_certified_empty`` directly and through ``Star.is_empty_set``).
    Catches a check that accepts a bound that is not strictly negative (a
    tolerance for rounding) and one that hands the box ends to the bound in
    the wrong order."""
    (A, b, lb, ub), ray = _FALSE_RAYS[name]
    x = [F(1)] if A.shape[0] == 2 else [F(-1, 2)]
    if not _exact_point_is_feasible(_sys(A, b, lb=lb, ub=ub), x):
        pytest.fail("the star is not feasible")
    ray = np.asarray(ray, dtype=np.float64)
    assert not _farkas_certified_empty(A, b, np.array(lb), np.array(ub), ray)
    monkeypatch.setattr(lpmod, "_feasibility_robust",
                        lambda A_, b_, lb_, ub_: ("infeasible", ray))
    S = Star(np.array([[0.0, 1.0]]), A, b.reshape(-1, 1),
             np.array([lb]), np.array([ub]))
    assert not S.is_empty_set()


def test_a_correct_farkas_vector_certifies_with_either_sign():
    """Exact multipliers of an empty system certify it, given with the sign
    HiGHS uses (negative) or the other one; with no rows, no ray, a missing
    or an infinite box no certificate is formed. Catches a check that tries
    one sign of the ray only (HiGHS's negative ray is then clipped to zero)
    and a check that skips its guards."""
    A = np.array([[1.0, 1.0], [-1.0, -1.0]])
    b = np.array([1.0, -1.5])                       # empty by 1/2
    lb, ub = -3 * np.ones(2), 3 * np.ones(2)
    for ray in ([1.0, 1.0], [-1.0, -1.0], [-2.0, -2.0]):
        assert _farkas_certified_empty(A, b, lb, ub, np.array(ray)), ray
    # 8 x <= 1 and -x / 4 <= -1/10: the multipliers (1/8, 4) cancel the column
    A2, b2 = np.array([[8.0], [-0.25]]), np.array([1.0, -0.1])
    assert _farkas_certified_empty(A2, b2, np.array([-10.0]), np.array([10.0]),
                                   np.array([-0.125, -4.0]))
    y = np.array([1.0, 1.0])
    assert not _farkas_certified_empty(A, b, None, ub, y)
    assert not _farkas_certified_empty(A, b, lb, np.array([3.0, np.inf]), y)
    assert not _farkas_certified_empty(A, b, lb, ub, None)
    assert not _farkas_certified_empty(np.zeros((0, 2)), np.zeros(0), lb, ub,
                                       np.zeros(0))


@pytest.mark.parametrize("box", ["none", "partly infinite", "finite, highs-ds"])
def test_a_default_infeasible_report_is_confirmed_before_emptiness(
        box, monkeypatch):
    """The default solve is made to report 'infeasible' on a feasible system
    (the presolve artefact measured on tiny-scale systems) on each route that
    takes the default solve: no box, a partly infinite box, and a finite box
    with a solver outside the certified first pass. It must be confirmed by
    the careful test, which finds the point, so the answer is 'not shown
    empty'. Catches a 'return False' on the default solve's word."""
    A = np.array([[1.0, 1.0], [-1.0, -1.0]])
    b = np.array([1.0, -1.0])                        # the line x0 + x1 = 1
    lb = ub = None
    solver = "default"
    if box == "partly infinite":
        lb, ub = np.array([0.0, 0.0]), np.array([np.inf, 1.0])
    elif box.startswith("finite"):
        lb, ub, solver = np.zeros(2), np.ones(2), "highs-ds"
    monkeypatch.setattr(lpmod, "solve_lp",
                        lambda *a, **k: (None, None, "infeasible", {}))
    assert check_feasibility(A=A, b=b, lb=lb, ub=ub, lp_solver=solver) is True
    S = Star(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), A, b.reshape(-1, 1),
             None if lb is None else lb.reshape(-1, 1),
             None if ub is None else ub.reshape(-1, 1))
    assert not S.is_empty_set(lp_solver=solver)


@pytest.mark.parametrize("status", ["failed: kUnknown", "unbounded",
                                    "error: solver crashed", "user_limit"])
def test_an_undecided_default_solve_raises_and_is_never_emptiness(
        status, monkeypatch):
    """A default solve that decides nothing (a failure, a limit, 'unbounded'
    on a zero objective) raises LPFailure; it is never read as 'empty'. A
    feasible box-free system and a finite box with highs-ds, both of which
    take the default solve. Catches an undecided status mapped to False."""
    monkeypatch.setattr(lpmod, "solve_lp",
                        lambda *a, **k: (None, None, status, {}))
    A = np.array([[1.0], [-1.0]])
    b = np.array([1.0, 0.0])                         # [0, 1]
    with pytest.raises(LPFailure):
        check_feasibility(A=A, b=b)
    with pytest.raises(LPFailure):
        check_feasibility(A=A, b=b, lb=np.zeros(1), ub=np.ones(1),
                          lp_solver="highs-ds")
    with pytest.raises(LPFailure):
        Star(np.array([[0.0, 1.0]]), A, b.reshape(-1, 1), None,
             None).is_empty_set()


def test_without_highspy_a_failed_careful_solve_is_not_emptiness(monkeypatch):
    """Without highspy no certificate can be formed and emptiness rests on
    two scipy solves agreeing. The default solve is made to report
    'infeasible' on the feasible star [0, 1] and the careful (presolve-off)
    solve to fail with scipy status 4: the two do not agree, so the star is
    not shown empty. Catches a careful solve whose failure is read as
    'infeasible'."""
    def fake(c, options=None, **kw):
        if (options or {}).get("presolve", True) is False:
            return SimpleNamespace(status=4, success=False, x=None, fun=None,
                                   nit=0, message="Numerical difficulties.")
        return SimpleNamespace(
            status=2, success=False, x=None, fun=None, nit=0,
            message="The problem is infeasible. (HiGHS Status 8: "
                    "model_status is Infeasible)")
    _no_highspy(monkeypatch)
    monkeypatch.setattr(lpmod, "_scipy_linprog", fake)
    A, b = np.array([[1.0], [-1.0]]), np.array([1.0, 0.0])
    assert not confirm_infeasible(A, b, None, None)
    assert not confirm_infeasible(A, b, np.zeros(1), np.ones(1))
    assert check_feasibility(A=A, b=b) is True
    assert not Star(np.array([[0.0, 1.0]]), A, b.reshape(-1, 1), None,
                    None).is_empty_set()


# The SET-7 system: a0 - 2**-41 a1 <= 0, a0 >= 0, |a1| <= 2**30 (all as rows)
# and a0 >= 3 * 2**-13 (a cut x >= 3 of x = 2**13 a0). The point
# a = (2**-11, 2**30) meets every row; HiGHS drops the entry 2**-41
# (4.5e-13 <= 1e-12) and sees a0 <= 0, so its solves say 'infeasible'.
_TINY_A = np.array([[1.0, -2.0 ** -41], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
                    [-1.0, 0.0]])
_TINY_B = np.array([0.0, 0.0, 2.0 ** 30, 2.0 ** 30, -3.0 * 2.0 ** -13])
_TINY_DROPPED = known_bug(
    17, "HiGHS drops constraint coefficients at or below 1e-12 and a system "
        "without a finite box has no certificate: the status path of the "
        "emptiness check reads a0 - 2**-41 a1 <= 0 as a0 <= 0 and calls a "
        "feasible system empty, a false UNSAT behind verify_specification "
        "(registry SET-7)")


@pytest.mark.parametrize("box", [pytest.param("none", marks=_TINY_DROPPED),
                                 "finite"])
def test_a_coefficient_below_the_highs_threshold_does_not_empty_a_system(box):
    """The SET-7 system has the point (2**-11, 2**30) (fractions), but HiGHS
    drops its entry 2**-41. With no box the emptiness check accepts the
    solver's 'infeasible' (marked); with the finite box [0, 2**-10] x
    [-2**30, 2**30] the Farkas check on the stored rows rejects HiGHS's ray
    (its bound is 2**-13 > 0), so the system is not shown empty: a natural
    solver error that the certificate must catch. Catches a Farkas check
    that takes the box ends in the wrong order."""
    point = [F(1, 2 ** 11), F(2 ** 30)]
    lb = ub = None
    if box == "finite":
        lb, ub = np.array([0.0, -2.0 ** 30]), np.array([2.0 ** -10, 2.0 ** 30])
    if not _exact_point_is_feasible(_sys(_TINY_A, _TINY_B, lb=lb, ub=ub),
                                    point):
        pytest.fail("the point is not feasible")
    col = (lambda v: None if v is None else v.reshape(-1, 1))
    S = Star(np.hstack([np.zeros((2, 1)), np.eye(2)]), _TINY_A,
             _TINY_B.reshape(-1, 1), col(lb), col(ub))
    shown_empty = {
        "check_feasibility": check_feasibility(A=_TINY_A, b=_TINY_B, lb=lb,
                                               ub=ub) is False,
        "confirm_infeasible": confirm_infeasible(_TINY_A, _TINY_B, lb, ub),
        "Star.is_empty_set": S.is_empty_set(),
    }
    assert not any(shown_empty.values()), shown_empty


# --------------------------------------------------------------------------- #
# The tol option of confirm_infeasible                                          #
# --------------------------------------------------------------------------- #
def test_confirm_infeasible_tol_loosens_the_rows_of_the_status_path():
    """x <= 1 and x >= 1.0001 is empty by 1e-4. Without a box the 'infeasible'
    status is accepted on rows loosened by tol (1 + |b_i|): with tol = 1e-3
    the loosened rows overlap (x = 1.0005 meets both), so it is not shown
    empty; with the default tol it is. With a finite box the certificate
    decides and tol plays no part. Catches tol not handed to the shared core,
    and a tol that is subtracted."""
    A, b = np.array([[1.0], [-1.0]]), np.array([1.0, -1.0001])
    tol = 1e-3
    x = 1.0005                      # meets both loosened rows
    assert x <= b[0] + tol * (1 + abs(b[0]))
    assert -x <= b[1] + tol * (1 + abs(b[1]))
    assert confirm_infeasible(A, b, None, None, tol=tol) is False
    assert confirm_infeasible(A, b, None, None) is True
    assert confirm_infeasible(A, b, np.array([-4.0]), np.array([4.0]),
                              tol=tol) is True


def test_check_feasibility_refuses_a_system_without_a_dimension():
    """No rows, no equality rows and no box: there is nothing to decide, and
    the call raises ValueError rather than answering. An unknown solver name
    raises ValueError as well."""
    with pytest.raises(ValueError):
        check_feasibility()
    with pytest.raises(ValueError):
        check_feasibility(A=np.eye(1), b=np.ones(1), lp_solver="not-a-solver")


# --------------------------------------------------------------------------- #
# LP optima used as bounds hold the exact optimum                               #
# --------------------------------------------------------------------------- #
def _polytopes(seed, count, rows):
    """Polytopes over 3 variables with non-dyadic rows (multiples of 1/3000)
    around a known interior point, a finite box of uneven width, and an
    objective with non-dyadic entries; scales 1e-2 to 1e2."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        scale = 10.0 ** int(rng.integers(-2, 3))
        x0 = rng.uniform(-1, 1, 3) * scale
        A = np.round(rng.normal(size=(rows, 3)), 3) / 3.0 if rows else None
        b = (A @ x0 + rng.uniform(0.01, 0.3, rows) * scale * 0.1
             if rows else None)
        lb = x0 - rng.uniform(0.3, 1.7, 3) * scale
        ub = x0 + rng.uniform(0.3, 1.7, 3) * scale
        f = np.round(rng.normal(size=3), 4) / 3.0
        out.append((A, b, lb, ub, f))
    return out


def _raw_optima_inside(cases):
    """How many raw scipy-HiGHS optima (no certificate) lie strictly inside
    the exact range: the cases can show a missing certificate only if some
    do."""
    from scipy.optimize import linprog
    count = 0
    for A, b, lb, ub, f in cases:
        lo_ex, hi_ex = _exact_range(A, b, lb, ub, f)
        kw = dict(A_ub=A, b_ub=b, bounds=list(zip(lb, ub)), method="highs")
        count += frac(linprog(f, **kw).fun) > lo_ex
        count += frac(-linprog(-f, **kw).fun) < hi_ex
    return count


def _optima_by_route(A, b, lb, ub, f):
    """``(min, max)`` of ``f x`` from every route that returns an LP optimum
    used as a bound: the single solves (highspy: default; scipy: highs-ds,
    highs-ipm), the batch solves (highspy and scipy), the presolve-off solve,
    and the Star queries on the star with basis ``[0 | f]`` (zero centre, so
    the centre addition plays no part)."""
    out = {}
    for s in ("default", "highs-ds", "highs-ipm"):
        out[f"solve_lp[{s}]"] = (
            solve_lp(f, A, b, lb=lb, ub=ub, lp_solver=s, minimize=True)[1],
            solve_lp(f, A, b, lb=lb, ub=ub, lp_solver=s, minimize=False)[1])
    for s in ("default", "highs-ds"):
        out[f"solve_lp_batch[{s}]"] = tuple(solve_lp_batch(
            [f, f], A, b, lb, ub, [True, False], lp_solver=s))
    out["solve_lp_no_presolve"] = (
        solve_lp_no_presolve(f, A, b, lb, ub, minimize=True)[1],
        solve_lp_no_presolve(f, A, b, lb, ub, minimize=False)[1])
    C = A if A is not None else np.zeros((0, 3))
    d = b.reshape(-1, 1) if b is not None else np.zeros((0, 1))
    S = Star(np.hstack([np.zeros((1, 1)), f.reshape(1, -1)]), C, d,
             lb.reshape(-1, 1), ub.reshape(-1, 1))
    out["Star.get_min/get_max"] = (S.get_min(0), S.get_max(0))
    lo, hi = S.get_ranges()
    out["Star.get_ranges"] = (lo[0, 0], hi[0, 0])
    return out


@pytest.mark.parametrize("rows", [4, 0], ids=["with rows", "box only"])
def test_lp_optima_used_as_bounds_hold_the_exact_optimum(rows):
    """Over a finite box every returned minimum is at or below the exact
    minimum and every maximum at or above the exact maximum, in fractions with
    zero slack, on every route. The raw solver optimum lies inside on about
    half of these LPs (checked below), so a certificate that keeps the inward
    of the two values, or is skipped when the LP has no rows, fails here."""
    cases = _polytopes(17 + rows, 10, rows)
    if _raw_optima_inside(cases) == 0:
        pytest.fail("no raw optimum lies inside the exact range: the cases "
                    "can no longer show a missing certificate")
    inside = []
    for k, (A, b, lb, ub, f) in enumerate(cases):
        lo_ex, hi_ex = _exact_range(A, b, lb, ub, f)
        for route, (lo, hi) in _optima_by_route(A, b, lb, ub, f).items():
            if lo is None or frac(lo) > lo_ex:
                inside.append((k, route, "min"))
            if hi is None or frac(hi) < hi_ex:
                inside.append((k, route, "max"))
    assert not inside, inside


def test_a_batch_of_more_than_256_optima_holds_every_exact_range():
    """A star of dimension 150 over 3 predicate variables with constraint rows
    gives one batch of 300 optima, certified in chunks of 256. Every range
    must hold the exact one (zero centre). Catches certified bounds written
    to the wrong objectives past the first chunk."""
    rng = np.random.default_rng(5)
    dim = 150
    C = np.round(rng.normal(size=(4, 3)), 3) / 3.0
    x0 = rng.uniform(-0.5, 0.5, 3)
    d = C @ x0 + rng.uniform(0.05, 0.3, 4)
    lb, ub = -1.3 * np.ones(3), 1.7 * np.ones(3)
    G = rng.normal(size=(dim, 3)) / 7.0
    S = Star(np.hstack([np.zeros((dim, 1)), G]), C, d.reshape(-1, 1),
             lb.reshape(-1, 1), ub.reshape(-1, 1))
    lo, hi = S.get_ranges(parallel=False)
    exact = _exact_ranges_of_rows(C, d, lb, ub, G)
    inside = [i for i, (lo_ex, hi_ex) in enumerate(exact)
              if frac(lo[i, 0]) > lo_ex or frac(hi[i, 0]) < hi_ex]
    assert not inside, inside


def test_the_certificate_moves_an_inward_optimum_outward():
    """A solver value one step inside the exact optimum (injected) is moved
    outward by the weak-duality bound, for a minimum and a maximum, with rows
    and without, singly (``_outward_optimum``) and in a batch whose chunks
    hold two objectives each, with mixed flags (``_outward_optima``). The
    exact optima: min x0 + x1 s.t. x0 + 2 x1 >= 1 over [0, 1]^2 is 1/2 (dual
    1/2); max x0 + x1 s.t. x0 + 2 x1 <= 2 is 3/2 (dual 1/2); min and max of
    (1/3) x0 - (2/3) x1 over [-0.7, 0.3] x [-0.1, 0.9] at box corners."""
    box = (np.zeros(2), np.ones(2))
    f = np.ones(2)
    A_lo, b_lo = np.array([[-1.0, -2.0]]), np.array([-1.0])
    A_hi, b_hi = np.array([[1.0, 2.0]]), np.array([2.0])
    y = np.array([0.5])
    step = 2.0 ** -40
    got = _outward_optimum(0.5 + step, f, y, A_lo, b_lo, box, True)
    assert frac(got) <= F(1, 2)
    got = _outward_optimum(1.5 - step, f, y, A_hi, b_hi, box, False)
    assert frac(got) >= F(3, 2)
    g = np.array([1 / 3, -2 / 3])
    gbox = (np.array([-0.7, -0.1]), np.array([0.3, 0.9]))
    lo_ex = frac(g[0]) * frac(-0.7) + frac(g[1]) * frac(0.9)
    hi_ex = frac(g[0]) * frac(0.3) + frac(g[1]) * frac(-0.1)
    lo_in, hi_in = float(lo_ex) + 1e-12, float(hi_ex) - 1e-12
    assert frac(_outward_optimum(lo_in, g, None, None, None, gbox,
                                 True)) <= lo_ex
    assert frac(_outward_optimum(hi_in, g, None, None, None, gbox,
                                 False)) >= hi_ex
    # a batch of five over the same rows, two per chunk, flags mixed
    results = [0.5 + step, 1.5 - step, 0.5 + step, 1.5 - step, 0.5 + step]
    A2 = np.vstack([A_lo, A_hi])
    b2 = np.concatenate([b_lo, b_hi])
    certify = [(i, f, np.array([0.5, 0.0]) if i % 2 == 0
                else np.array([0.0, 0.5]), i % 2 == 0) for i in range(5)]
    # over both rows the minimum is still 1/2 and the maximum still 3/2
    _outward_optima(results, certify, A2, b2, box, chunk=2)
    for i, v in enumerate(results):
        if i % 2 == 0:
            assert frac(v) <= F(1, 2), (i, v)
        else:
            assert frac(v) >= F(3, 2), (i, v)
    results = [lo_in, hi_in, lo_in]
    _outward_optima(results, [(0, g, None, True), (1, g, None, False),
                              (2, g, None, True)], None, None, gbox, chunk=2)
    assert frac(results[0]) <= lo_ex and frac(results[2]) <= lo_ex
    assert frac(results[1]) >= hi_ex


# --------------------------------------------------------------------------- #
# The pad and the solver of the relaxation boxes                                #
# --------------------------------------------------------------------------- #
def test_padded_lp_ranges_hold_the_exact_range_on_both_sides_of_zero():
    """``_outward_lp_ranges`` on LP ranges that a solver left 5e-10 inside
    (the measured HiGHS undershoot) of exact ranges whose ends lie below -1,
    in (-1, 0), at 0, in (0, 1) and above 1: the padded range holds the exact
    one. A row with some zero generator entries is padded like any other; a
    row with none is the exact centre. Catches a pad computed from the signed
    optimum (inward below -1), a pad with no absolute part (nothing at 0), an
    upper end moved down, and a row taken for a constant when one of its
    entries is zero."""
    ends = [(-1000.0, -4.0), (-4.0, -0.5), (-0.5, 0.0), (0.0, 0.5),
            (0.5, 4.0), (4.0, 1000.0), (-4.0, 4.0)]
    lo_ex = np.array([e[0] for e in ends])
    hi_ex = np.array([e[1] for e in ends])
    V = np.zeros((len(ends) + 1, 3))
    V[:-1, 0] = 0.5 * (lo_ex + hi_ex)
    V[:-1, 1] = 0.5 * (hi_ex - lo_ex)
    V[1:-1:2, 2] = 1.0       # odd rows: no zero entry; even rows: one zero
    V[-1, 0] = 0.1           # a constant row (no generator)
    lo_in = np.append(lo_ex + 5e-10, 0.25)
    hi_in = np.append(hi_ex - 5e-10, 0.75)
    lo, hi = _outward_lp_ranges(lo_in, hi_in, V)
    assert np.all(lo[:-1] <= lo_ex), (lo[:-1] - lo_ex)
    assert np.all(hi[:-1] >= hi_ex), (hi[:-1] - hi_ex)
    assert lo[-1] == hi[-1] == 0.1


@pytest.mark.parametrize("requested, configured, expected", [
    ("SCS", None, LPSolver.LINPROG),
    ("highs-ds", None, LPSolver.HIGHS_DS),
    (LPSolver.DEFAULT, "highs-ipm", LPSolver.HIGHS_IPM),
    (None, "highs-ds", LPSolver.HIGHS_DS),
    ("default", "SCS", LPSolver.LINPROG),
])
def test_bound_lps_run_on_highs_and_keep_a_highs_method(
        requested, configured, expected):
    """The LPs whose optimum is used as a bound run on HiGHS: a CVXPY solver
    (requested or set globally) is replaced by LINPROG, a HiGHS method the
    caller or the global setting picks is kept. Catches the reroute dropped,
    a HiGHS method replaced, and the default not resolved against the global
    setting."""
    try:
        if configured is not None:
            config.lp_solver = configured
        assert _bound_lp_solver(requested) is expected
    finally:
        config.reset()


def test_bound_lp_solver_refuses_an_unknown_name():
    with pytest.raises(ValueError):
        _bound_lp_solver("not-a-solver")
