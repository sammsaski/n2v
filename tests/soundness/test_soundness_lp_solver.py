"""Soundness cells for the LP wrapper ``n2v/utils/lpsolver.py``.

What the wrapper promises, cell by cell:

- ``solve_lp`` and ``solve_lp_batch`` return, for each objective, a value on
  the outward side of the exact optimum (a minimum not above it, a maximum
  not below it) and within the documented pad ``1e-7 * (1 + |opt|)`` of it,
  on every HiGHS backend (the highspy path behind ``'default'``,
  ``'linprog'`` and ``'highs'``; scipy's ``'highs-ds'`` and ``'highs-ipm'``),
  for single and batched calls, with a finite predicate box (the certificate
  pushes a round-to-nearest optimum outward) and without one (rows only, a
  half-open box). A batch mixes minimise and maximise in an order that is not
  sorted, so an answer handed to the wrong objective is seen.
- An LP with no optimum has no value: an infeasible system is ``None`` and
  status ``'infeasible'``, an unbounded objective is ``None`` and status
  ``'unbounded'`` (never a finite number that a caller would use as a bound).
- ``check_feasibility`` and ``confirm_infeasible`` call a system empty only
  when it is empty, with and without a finite box.
- The CVXPY backends (``'CLARABEL'``, ``'SCS'``, ``'OSQP'``) answer the LP that
  was asked (sense, rows, bounds) to the solver's own accuracy. That their
  values are not pushed outward is registry row SET-6, marked in
  ``test_soundness_range_rounding.py``, so these cells do not test
  outwardness.

Oracle: the exact optimum in ``fractions.Fraction`` by enumerating the
vertices of the bounded polytope ``{A x <= b, lb <= x <= ub}``
(``tests/_oracles/rational.py``: ``vertices``, ``is_bounded``); no n2v code
computes a reference. Data: the problems without a finite box are dyadic
(their optima are exact in float64, so a value one ulp inside fails with zero
slack); the problems with a finite box have random float64 entries, where
the raw HiGHS optimum lands inside the exact one about half of the time and
only the certificate keeps it outward. There is no layout in this file: every
vector is in the variable order of ``A``'s columns.
"""

import types
import warnings
from fractions import Fraction

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sparse
from scipy.optimize import OptimizeResult

import n2v.utils.lpsolver as lpsolver
from n2v.config import config
from n2v.sets import Star
from n2v.sets.predicate_box import ensure_finite_predicate_box
from n2v.utils.lpsolver import (LPFailure, _solve_lp_careful,
                                check_feasibility, confirm_infeasible,
                                solve_lp, solve_lp_batch,
                                solve_lp_no_presolve)
from tests._oracles import rational as R
from tests.known_bugs import known_bug

# Every HiGHS solver name, for the loops of a cell.
HIGHS_SINGLE = ("default", "linprog", "highs", "highs-ds", "highs-ipm")
HIGHS_BATCH = ("default", "linprog", "highs-ds", "highs-ipm")
# One name per distinct route, for the grids: 'linprog' and 'highs' take the
# same highspy path, so 'highs' is run in the loops only; 'default' differs
# from 'linprog' by its resolution through the global setting.
HIGHS_ROUTES = ("default", "linprog", "highs-ds", "highs-ipm")
CVXPY_SOLVERS = ("CLARABEL", "SCS", "OSQP")
# The documented pad of an LP optimum used as a bound (lpsolver._LP_PAD).
PAD = 1e-7


# --------------------------------------------------------------------------- #
# The exact reference                                                           #
# --------------------------------------------------------------------------- #
def _rows(A, b, lb=None, ub=None):
    """``A x <= b`` plus the finite entries of ``lb``/``ub`` as Fraction rows."""
    A = np.asarray(A, dtype=np.float64)
    rows = [[Fraction(float(v)) for v in row] for row in A]
    rhs = [Fraction(float(v)) for v in np.asarray(b, dtype=np.float64).ravel()]
    n = A.shape[1]
    for bound, sign in ((ub, 1), (lb, -1)):
        if bound is None:
            continue
        for j, v in enumerate(np.asarray(bound, dtype=np.float64).ravel()):
            if np.isfinite(v):
                unit = [Fraction(0)] * n
                unit[j] = Fraction(sign)
                rows.append(unit)
                rhs.append(sign * Fraction(float(v)))
    return rows, rhs


def _vertices(A, b, lb=None, ub=None):
    rows, rhs = _rows(A, b, lb, ub)
    if not R.is_bounded(rows):
        pytest.fail("test setup: the reference polytope is unbounded")
    found = R.vertices(rows, rhs)
    if not found:
        pytest.fail("test setup: the reference polytope is empty")
    return found


def _exact(f, verts, minimize):
    values = [R.dot([Fraction(float(v)) for v in f], x) for x in verts]
    return min(values) if minimize else max(values)


def _check(value, exact, minimize, what):
    """``value`` is outward of ``exact`` with zero slack and within the
    documented pad of it."""
    assert value is not None, f"{what}: no value for an LP with an optimum"
    assert np.isfinite(value), f"{what}: value {value!r}"
    got = Fraction(float(value))
    if minimize:
        assert got <= exact, (f"{what}: minimum {value!r} is above the exact "
                              f"minimum {float(exact)!r} by "
                              f"{float(got - exact):.3g}")
    else:
        assert got >= exact, (f"{what}: maximum {value!r} is below the exact "
                              f"maximum {float(exact)!r} by "
                              f"{float(exact - got):.3g}")
    pad = PAD * (1 + abs(float(exact)))
    assert abs(float(got - exact)) <= pad, (
        f"{what}: {'minimum' if minimize else 'maximum'} {value!r} is "
        f"{float(abs(got - exact)):.3g} away from the exact optimum "
        f"{float(exact)!r}, more than the pad {pad:.3g}")


def _single(f, A, b, lb, ub, solver, minimize):
    _, value, status, _ = solve_lp(f, A, b, lb=lb, ub=ub, lp_solver=solver,
                                   minimize=minimize)
    assert status == "optimal", f"solver {solver}: status {status!r}"
    return value


# --------------------------------------------------------------------------- #
# Problems                                                                      #
# --------------------------------------------------------------------------- #
# A dyadic polytope given by rows only (no predicate box). Every vertex is
# dyadic (checked by exact enumeration when the file was written), so every
# optimum below is a float64 number and a solver that is exact on such data
# has no rounding to forgive. The polytope reaches negative values in every
# coordinate (a solver that bounds the variables below by 0 misses them), it
# is not symmetric (a sign error moves every optimum), and rows 6 and 7 have
# their largest coefficient below 0.5 and bind at optima of the objectives
# below (a row scaling that forgets the right-hand side moves them).
_DYADIC_A = np.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.25, -0.25, -0.25],
    [-0.125, -0.125, 0.125],
    [1.0, 0.0, -1.0],
    [-0.5, -0.5, 0.0],
    [-1.0, 1.0, 1.0],
])
_DYADIC_B = np.array([1.5, 2.25, 0.75, 1.25, 2.0, 0.5, -0.0625, 0.28125,
                      -0.5, 0.75, 1.75])
# Objectives whose twelve optima (min and max of each) are distinct dyadic
# numbers, so an answer handed to the wrong position or sense is seen.
_DYADIC_F = [np.array(v) for v in (
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    [1.0, -0.5, 0.25], [-0.25, 0.75, -1.0], [0.5, 0.5, -0.125])]


def _dyadic(x):
    d = Fraction(x).denominator
    return d & (d - 1) == 0


def _dyadic_vertices(A, b, lb=None, ub=None):
    """``_vertices`` of a problem that must have dyadic vertices only: a
    non-dyadic vertex would make an exact-optimum check depend on the
    rounding of the solver (registry row SET-6, not this file's cell)."""
    verts = _vertices(A, b, lb, ub)
    if not all(_dyadic(x) for v in verts for x in v):
        pytest.fail("test setup: the problem has a vertex that is not dyadic")
    return verts


def _finite_box_problem(seed):
    """Random float64 rows around an interior point x0 and a finite box that
    is not symmetric about zero; the rows bind (slack 0.05 to 0.5)."""
    rng = np.random.default_rng(seed)
    n, m = 3, 4
    x0 = rng.uniform(-1, 1, n) + np.array([0.0, 0.5, -0.25])
    A = rng.standard_normal((m, n))
    b = A @ x0 + rng.uniform(0.05, 0.5, m)
    lb = x0 - rng.uniform(0.5, 2.0, n)
    ub = x0 + rng.uniform(0.5, 2.0, n)
    objectives = [rng.standard_normal(n) for _ in range(3)]
    return A, b, lb, ub, objectives


_SEEDS = tuple(range(8))

# Unsorted, mixed: (objective index, minimise?)
_MIXED_ORDER = [(2, False), (0, True), (1, False), (0, False), (2, True),
                (1, True), (3, False), (4, True), (5, False), (3, True),
                (5, True), (4, False)]


# --------------------------------------------------------------------------- #
# Single solves                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("solver", HIGHS_ROUTES)
def test_single_solve_with_a_finite_box_is_outward_and_within_the_pad(solver):
    """Random float64 LPs over a finite box, min and max of three objectives:
    the raw HiGHS optimum is round-to-nearest and lands inside the exact one
    about half of the time. Catches a certificate that keeps the more inward
    of the solver value and the rigorous bound, a certificate built for the
    negated objective, a sign slip in the maximisation and a value handed to
    the wrong sense (single solves on highspy and on scipy)."""
    for seed in _SEEDS:
        A, b, lb, ub, objectives = _finite_box_problem(seed)
        verts = _vertices(A, b, lb, ub)
        for k, f in enumerate(objectives):
            for minimize in (True, False):
                value = _single(f, A, b, lb, ub, solver, minimize)
                _check(value, _exact(f, verts, minimize), minimize,
                       f"solve_lp({solver}) seed {seed} objective {k}")


@pytest.mark.parametrize("box", ["none", "half_open"])
@pytest.mark.parametrize("solver", HIGHS_ROUTES)
def test_single_solve_without_a_finite_box_is_exact_on_dyadic_rows(solver,
                                                                   box):
    """A polytope given by rows only (``lb = ub = None``) or with only a lower
    box ``lb = -10``: no certificate can repair a wrong solve, so the value
    must be the exact optimum on the outward side. Catches a variable lower
    bound of 0 put in for a missing ``lb``, a maximisation that returns the
    negated minimum, rows added as ``A x >= b``, and a row scaling that
    divides the coefficients and not the right-hand side."""
    lb, ub = (None, None) if box == "none" else (np.full(3, -10.0), None)
    verts = _dyadic_vertices(_DYADIC_A, _DYADIC_B, lb, ub)
    for k, f in enumerate(_DYADIC_F):
        for minimize in (True, False):
            value = _single(f, _DYADIC_A, _DYADIC_B, lb, ub, solver, minimize)
            _check(value, _exact(f, verts, minimize), minimize,
                   f"solve_lp({solver}, box={box}) objective {k}")


@pytest.mark.parametrize("box", ["finite", "none"])
def test_solve_without_presolve_is_outward_and_within_the_pad(box):
    """``solve_lp_no_presolve``, the careful re-solve that the default solve
    and the predicate-box finitizer fall back to: same promise as
    ``solve_lp``. Catches a sense or row error in that separate model."""
    if box == "finite":
        problems = [_finite_box_problem(seed) for seed in _SEEDS[:4]]
    else:
        problems = [(_DYADIC_A, _DYADIC_B, None, None, _DYADIC_F)]
    for A, b, lb, ub, objectives in problems:
        verts = _vertices(A, b, lb, ub)
        for k, f in enumerate(objectives):
            for minimize in (True, False):
                _, value, status, _ = solve_lp_no_presolve(f, A, b, lb, ub,
                                                           minimize=minimize)
                assert status == "optimal", status
                _check(value, _exact(f, verts, minimize), minimize,
                       f"solve_lp_no_presolve(box={box}) objective {k}")


def test_equality_rows_are_kept():
    """``Aeq x = beq`` on the dyadic polytope (x0 - x2 = -1, a plane through
    its interior): the solve goes to scipy whatever the solver name, and the
    value is the exact optimum of the cut polytope. Catches equality rows
    dropped or read as one-sided."""
    Aeq, beq = np.array([[1.0, 0.0, -1.0]]), np.array([-1.0])
    A = np.vstack([_DYADIC_A, Aeq, -Aeq])
    b = np.concatenate([_DYADIC_B, beq, -beq])
    verts = _dyadic_vertices(A, b)
    for solver in ("default", "highs-ds"):
        for k, f in enumerate(_DYADIC_F):
            for minimize in (True, False):
                _, value, status, _ = solve_lp(
                    f, _DYADIC_A, _DYADIC_B, Aeq=Aeq, beq=beq,
                    lp_solver=solver, minimize=minimize)
                assert status == "optimal", status
                _check(value, _exact(f, verts, minimize), minimize,
                       f"solve_lp({solver}) with Aeq, objective {k}")


def test_the_default_solver_follows_the_global_setting():
    """``lp_solver='default'`` resolves to ``config.lp_solver``; with the
    global default set to scipy's dual simplex the answer must still be the
    exact optimum (catches a resolution that drops the sense or the rows on
    the scipy route)."""
    verts = _dyadic_vertices(_DYADIC_A, _DYADIC_B)
    try:
        config.lp_solver = "highs-ds"
        for k, f in enumerate(_DYADIC_F):
            for minimize in (True, False):
                value = _single(f, _DYADIC_A, _DYADIC_B, None, None,
                                "default", minimize)
                _check(value, _exact(f, verts, minimize), minimize,
                       f"default -> highs-ds, objective {k}")
        values = solve_lp_batch(_DYADIC_F, _DYADIC_A, _DYADIC_B,
                                minimize_flags=[False] * len(_DYADIC_F))
        for k, (f, value) in enumerate(zip(_DYADIC_F, values)):
            _check(value, _exact(f, verts, False), False,
                   f"batch default -> highs-ds, objective {k}")
    finally:
        config.reset()


# --------------------------------------------------------------------------- #
# Batched solves                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("solver", HIGHS_BATCH)
def test_batch_with_a_finite_box_mixes_senses_in_any_order(solver):
    """One batch per random problem: its three objectives, each minimised and
    maximised, in an unsorted mixed order. Catches the batch certificate
    keeping the inward value, the scipy batch loop certifying the negated
    objective, an inverted objective sense, and answers given to the wrong
    position of the batch."""
    order = [(2, False), (0, True), (1, False), (0, False), (2, True),
             (1, True)]
    for seed in _SEEDS:
        A, b, lb, ub, objectives = _finite_box_problem(seed)
        verts = _vertices(A, b, lb, ub)
        values = solve_lp_batch([objectives[k] for k, _ in order], A, b, lb,
                                ub, [mn for _, mn in order], lp_solver=solver)
        assert len(values) == len(order)
        for (k, minimize), value in zip(order, values):
            _check(value, _exact(objectives[k], verts, minimize), minimize,
                   f"solve_lp_batch({solver}) seed {seed} objective {k} "
                   f"{'min' if minimize else 'max'}")


@pytest.mark.parametrize("box", ["none", "half_open"])
@pytest.mark.parametrize("solver", HIGHS_BATCH)
def test_batch_without_a_finite_box_is_exact_on_dyadic_rows(solver, box):
    """The dyadic polytope with no finite box, twelve objectives mixing min
    and max in an unsorted order. Catches the scipy batch loop that returns
    a maximum negated, the highspy batch with its sense inverted, rows added
    the wrong way round and the row-scaling slip, which no certificate can
    repair here."""
    lb, ub = (None, None) if box == "none" else (np.full(3, -10.0), None)
    verts = _dyadic_vertices(_DYADIC_A, _DYADIC_B, lb, ub)
    values = solve_lp_batch([_DYADIC_F[k] for k, _ in _MIXED_ORDER],
                            _DYADIC_A, _DYADIC_B, lb, ub,
                            [mn for _, mn in _MIXED_ORDER], lp_solver=solver)
    assert len(values) == len(_MIXED_ORDER)
    for (k, minimize), value in zip(_MIXED_ORDER, values):
        _check(value, _exact(_DYADIC_F[k], verts, minimize), minimize,
               f"solve_lp_batch({solver}, box={box}) objective {k} "
               f"{'min' if minimize else 'max'}")


def test_batch_without_flags_minimises_every_objective():
    """``minimize_flags=None`` means minimise all (the documented default).
    Catches a default that maximises, or that reads the missing flags as
    "maximise" for some positions."""
    verts = _dyadic_vertices(_DYADIC_A, _DYADIC_B)
    for solver in HIGHS_BATCH:
        values = solve_lp_batch(_DYADIC_F, _DYADIC_A, _DYADIC_B,
                                lp_solver=solver)
        for k, (f, value) in enumerate(zip(_DYADIC_F, values)):
            _check(value, _exact(f, verts, True), True,
                   f"solve_lp_batch({solver}) default flags, objective {k}")


# --------------------------------------------------------------------------- #
# No optimum: infeasible and unbounded                                          #
# --------------------------------------------------------------------------- #
# x0 + x1 <= -1 inside the box [0, 1]^2 given by rows: empty.
_EMPTY_A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
                     [1.0, 1.0]])
_EMPTY_B = np.array([1.0, 0.0, 1.0, 0.0, -1.0])
# x0 in [-1, 1], x1 <= 1 and nothing below: min x1 has no optimum.
_OPEN_A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
_OPEN_B = np.array([1.0, 1.0, 1.0])


@pytest.mark.parametrize("box", ["none", "finite"])
@pytest.mark.parametrize("solver", HIGHS_ROUTES)
def test_an_empty_system_has_no_optimum(solver, box):
    """No value may come back for an empty polytope, single or batched: a
    number there would be a bound on nothing."""
    lb, ub = (None, None) if box == "none" else (np.full(2, -2.0),
                                                 np.full(2, 2.0))
    for minimize in (True, False):
        _, value, status, _ = solve_lp(np.array([1.0, -1.0]), _EMPTY_A,
                                       _EMPTY_B, lb=lb, ub=ub,
                                       lp_solver=solver, minimize=minimize)
        assert value is None and status == "infeasible", (value, status)
    if solver in HIGHS_BATCH:
        values = solve_lp_batch([np.array([1.0, 0.0]), np.array([0.0, 1.0])],
                                _EMPTY_A, _EMPTY_B, lb, ub, [False, True],
                                lp_solver=solver)
        assert values == [None, None], values


@pytest.mark.parametrize("solver", HIGHS_ROUTES)
def test_an_unbounded_objective_has_no_finite_value(solver):
    """min x1 over a set open below: status ``'unbounded'`` and no value; in a
    batch the unbounded objective is None (or minus infinity), never a finite
    number such as a placeholder 0, while the bounded objective next to it
    keeps its exact value."""
    _, value, status, _ = solve_lp(np.array([0.0, 1.0]), _OPEN_A, _OPEN_B,
                                   lp_solver=solver, minimize=True)
    assert value is None and status == "unbounded", (value, status)
    if solver in HIGHS_BATCH:
        values = solve_lp_batch(
            [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])],
            _OPEN_A, _OPEN_B, None, None, [False, True, False],
            lp_solver=solver)
        assert values[1] is None or values[1] == -np.inf, values
        # max x0 = 1 and max x1 = 1 over the set (exact, dyadic)
        for position in (0, 2):
            _check(values[position], Fraction(1), False,
                   f"solve_lp_batch({solver}) bounded objective {position} "
                   f"next to an unbounded one")


# --------------------------------------------------------------------------- #
# Emptiness                                                                     #
# --------------------------------------------------------------------------- #
def _tiny_feasible(seed, scale):
    """Rows of size ``scale`` around a known point x0 (slack 1e-3 of the
    row), with box rows at the same scale and no variable bounds; x0 is
    checked to satisfy every row exactly."""
    rng = np.random.default_rng(seed)
    n = 3
    x0 = rng.uniform(-1, 1, n)
    A = rng.standard_normal((5, n)) * scale
    b = A @ x0 + np.abs(rng.standard_normal(5)) * scale * 1e-3
    A = np.vstack([A, np.eye(n) * scale, -np.eye(n) * scale])
    b = np.concatenate([b, (x0 + 2) * scale, -(x0 - 2) * scale])
    rows, rhs = _rows(A, b)
    x = [Fraction(float(v)) for v in x0]
    if not all(R.dot(r, x) <= v for r, v in zip(rows, rhs)):
        pytest.fail("test setup: x0 is not in the polytope")
    return A, b


@pytest.mark.parametrize("box", ["none", "finite", "half_open"])
def test_a_feasible_system_is_never_reported_empty(box):
    """The dyadic polytope, a thin slab of it (row 0 pushed to 1/1024 above
    the smallest x0) and polytopes with rows of size 1e-6 to 1e-12: each holds
    a known point, so ``check_feasibility`` must say True and
    ``confirm_infeasible`` False, with a finite box, with none and with a box
    open above. Without a finite box no certificate exists and only two
    agreeing solves can call a system empty; that a first solve's
    'infeasible' alone is not taken as emptiness is checked by
    ``test_an_unconfirmed_infeasible_report_is_not_emptiness``. Catches rows
    added as ``A x >= b`` and a status misread as infeasible."""
    systems = [(_DYADIC_A, _DYADIC_B)]
    slab_b = _DYADIC_B.copy()
    slab_b[0] = -1.875 + 1.0 / 1024          # x0 in [-15/8, -15/8 + 2^-10]
    _vertices(_DYADIC_A, slab_b)             # setup check: the slab holds points
    systems.append((_DYADIC_A, slab_b))
    systems += [_tiny_feasible(seed, scale) for seed, scale in
                enumerate((1e-6, 1e-8, 1e-9, 1e-10, 1e-12))]
    for k, (A, b) in enumerate(systems):
        n = A.shape[1]
        lb, ub = {"none": (None, None),
                  "finite": (np.full(n, -10.0), np.full(n, 10.0)),
                  "half_open": (np.full(n, -10.0), None)}[box]
        assert check_feasibility(A=A, b=b, lb=lb, ub=ub) is True, k
        assert not confirm_infeasible(A, b, lb, ub), k


@pytest.mark.parametrize("box", ["none", "finite"])
def test_an_empty_system_is_reported_empty(box):
    """The empty system above (and the dyadic polytope cut by x0 >= 2,
    beyond its row x0 <= 1.5) must be reported empty by both functions.
    Catches an emptiness test that never says empty (a caller would then
    keep searching an empty branch; not unsound, but the promise of
    ``confirm_infeasible``)."""
    cut_A = np.vstack([_DYADIC_A, [[-1.0, 0.0, 0.0]]])
    cut_b = np.concatenate([_DYADIC_B, [-2.0]])
    for A, b in ((_EMPTY_A, _EMPTY_B), (cut_A, cut_b)):
        n = A.shape[1]
        lb, ub = ((None, None) if box == "none"
                  else (np.full(n, -10.0), np.full(n, 10.0)))
        assert check_feasibility(A=A, b=b, lb=lb, ub=ub) is False
        assert confirm_infeasible(A, b, lb, ub)


# --------------------------------------------------------------------------- #
# CVXPY backends                                                                #
# --------------------------------------------------------------------------- #
# The solvers' own accuracy on these problems, relative to 1 + |optimum|:
# about ten times the largest error measured on the unmodified library
# (CLARABEL 6.7e-10, SCS 7.7e-6, OSQP 4.3e-6). A wrong sense, row or bound
# moves an optimum by 1/32 or
# more.
_CVXPY_TOL = {"CLARABEL": 1e-8, "SCS": 1e-4, "OSQP": 1e-4}


def _cvxpy(call):
    with warnings.catch_warnings():
        # CVXPY's note on a solve that stopped short ('Solution may be
        # inaccurate'), which the iteration-limit cells provoke on purpose;
        # the status and the value are checked. Nothing else is silenced.
        warnings.filterwarnings("ignore", message="Solution may be inaccurate",
                                category=UserWarning, module="cvxpy")
        return call()


@pytest.mark.parametrize("solver", CVXPY_SOLVERS)
def test_cvxpy_backends_solve_the_lp_that_was_asked(solver):
    """The dyadic polytope with and without a finite box, min and max of six
    objectives, single and batched in an unsorted mixed order: each value
    within the solver's accuracy of the exact optimum (``_CVXPY_TOL``,
    relative). Catches a dropped row or bound, a sense slip and answers given
    to the wrong objective on the CVXPY route, which no other test runs.
    Outwardness is not checked here: the CVXPY values are not certified
    (registry row SET-6)."""
    tol = _CVXPY_TOL[solver]
    for lb, ub in ((None, None), (np.array([-1.0, -1.0, -0.25]),
                                  np.array([1.0, 0.5, 1.5]))):
        verts = _dyadic_vertices(_DYADIC_A, _DYADIC_B, lb, ub)
        for k, f in enumerate(_DYADIC_F):
            for minimize in (True, False):
                _, value, status, _ = _cvxpy(lambda: solve_lp(
                    f, _DYADIC_A, _DYADIC_B, lb=lb, ub=ub, lp_solver=solver,
                    minimize=minimize))
                exact = float(_exact(f, verts, minimize))
                assert status in ("optimal", "optimal_inaccurate"), status
                assert abs(value - exact) <= tol * (1 + abs(exact)), (
                    solver, k, minimize, value, exact)
        values = _cvxpy(lambda: solve_lp_batch(
            [_DYADIC_F[k] for k, _ in _MIXED_ORDER], _DYADIC_A, _DYADIC_B,
            lb, ub, [mn for _, mn in _MIXED_ORDER], lp_solver=solver))
        for (k, minimize), value in zip(_MIXED_ORDER, values):
            exact = float(_exact(_DYADIC_F[k], verts, minimize))
            assert value is not None, (solver, k, minimize)
            assert abs(value - exact) <= tol * (1 + abs(exact)), (
                solver, k, minimize, value, exact)


@pytest.mark.parametrize("solver", CVXPY_SOLVERS)
def test_cvxpy_backends_give_no_value_without_an_optimum(solver):
    """An empty system and an unbounded objective on the CVXPY route: no
    value, and the status of what the solver proved or a failure (a
    first-order solver may say it is unsure; it may not answer). The empty
    system is never called 'unbounded' and the unbounded objective never
    'infeasible', the status a caller reads as "this set is empty". Catches
    the CVXPY status 'unbounded' mapped to 'infeasible' and either verdict
    turned into a value."""
    _, value, status, _ = _cvxpy(lambda: solve_lp(
        np.array([1.0, -1.0]), _EMPTY_A, _EMPTY_B, lp_solver=solver))
    assert value is None, (status, value)
    assert status == "infeasible" or status.startswith("failed"), status
    _, value, status, _ = _cvxpy(lambda: solve_lp(
        np.array([0.0, 1.0]), _OPEN_A, _OPEN_B, lp_solver=solver,
        minimize=True))
    assert value is None, (status, value)
    assert status == "unbounded" or status.startswith("failed"), status


# --------------------------------------------------------------------------- #
# A box alone, sparse rows, feasibility without inequality rows                 #
# --------------------------------------------------------------------------- #
_BOX_LB = np.array([-1.0, -0.5, 0.25])
_BOX_UB = np.array([0.75, 1.0, 1.5])
_BOX_F = [np.array([0.5, 0.5, -0.125]), np.array([-1.0, 0.25, 0.0]),
          np.array([0.0, -0.75, 1.0])]
_BOX_ORDER = [(1, False), (0, True), (2, True), (0, False), (2, False),
              (1, True)]


def _no_rows(rows):
    return (None, None) if rows == "none" else (np.zeros((0, 3)), np.zeros(0))


@pytest.mark.parametrize("rows", ["none", "zero-rows"])
@pytest.mark.parametrize("solver", HIGHS_ROUTES)
def test_an_lp_on_a_box_alone_is_outward_and_within_the_pad(solver, rows):
    """No row at all (``A = None``) or a row matrix with no rows, over the
    dyadic box [-1, 3/4] x [-1/2, 1] x [1/4, 3/2]: each optimum is a corner,
    single (both senses) and batched (unsorted mixed senses). Catches a
    certificate built for rows that do not exist, the box bounds read in the
    wrong order, and an empty row block that breaks the model."""
    A, b = _no_rows(rows)
    verts = _vertices(np.zeros((0, 3)), np.zeros(0), _BOX_LB, _BOX_UB)
    for k, f in enumerate(_BOX_F):
        for minimize in (True, False):
            value = _single(f, A, b, _BOX_LB, _BOX_UB, solver, minimize)
            _check(value, _exact(f, verts, minimize), minimize,
                   f"solve_lp({solver}) on a box, rows={rows}, objective {k}")
    if solver in HIGHS_BATCH:
        values = solve_lp_batch([_BOX_F[k] for k, _ in _BOX_ORDER], A, b,
                                _BOX_LB, _BOX_UB,
                                [mn for _, mn in _BOX_ORDER], lp_solver=solver)
        for (k, minimize), value in zip(_BOX_ORDER, values):
            _check(value, _exact(_BOX_F[k], verts, minimize), minimize,
                   f"solve_lp_batch({solver}) on a box, rows={rows}, "
                   f"objective {k}")


@pytest.mark.parametrize("rows", [
    "none",
    pytest.param("zero-rows", marks=known_bug(
        17, "solve_lp on a CVXPY solver crashes on a row matrix with no rows "
            "(A of shape (0, n), b of shape (0,)): CVXPY raises 'Invalid "
            "dimensions (0, n)' while every HiGHS route answers",
        raises=ValueError))])
@pytest.mark.parametrize("solver", CVXPY_SOLVERS)
def test_cvxpy_backends_solve_an_lp_on_a_box_alone(solver, rows):
    """The box-only LPs above on the CVXPY route: each value within the
    solver's accuracy of the exact corner value (``_CVXPY_TOL``; outwardness
    is registry row SET-6). Catches the box bounds dropped or exchanged, and
    a crash on an empty row block."""
    A, b = _no_rows(rows)
    verts = _vertices(np.zeros((0, 3)), np.zeros(0), _BOX_LB, _BOX_UB)
    tol = _CVXPY_TOL[solver]
    for k, f in enumerate(_BOX_F):
        for minimize in (True, False):
            _, value, status, _ = _cvxpy(lambda: solve_lp(
                f, A, b, lb=_BOX_LB, ub=_BOX_UB, lp_solver=solver,
                minimize=minimize))
            exact = float(_exact(f, verts, minimize))
            assert status in ("optimal", "optimal_inaccurate"), status
            assert abs(value - exact) <= tol * (1 + abs(exact)), (
                solver, k, minimize, value, exact)


@pytest.mark.parametrize("solver", ["highs-ds", "highs-ipm"])
def test_sparse_rows_on_the_scipy_route_are_kept(solver):
    """``scipy.sparse`` inequality rows, and sparse equality rows, on the
    single scipy solve (the route that accepts them; no certificate is formed
    for sparse rows, and the dyadic optima need none): the exact optimum of
    each objective in both senses. Catches sparse rows dropped, or passed
    without their right-hand side."""
    import scipy.sparse as sparse
    A = sparse.csr_matrix(_DYADIC_A)
    Aeq, beq = sparse.csr_matrix([[1.0, 0.0, -1.0]]), np.array([-1.0])
    plain = _dyadic_vertices(_DYADIC_A, _DYADIC_B)
    cut = _dyadic_vertices(
        np.vstack([_DYADIC_A, Aeq.toarray(), -Aeq.toarray()]),
        np.concatenate([_DYADIC_B, beq, -beq]))
    for k, f in enumerate(_DYADIC_F):
        for minimize in (True, False):
            _, value, status, _ = solve_lp(f, A, _DYADIC_B, lp_solver=solver,
                                           minimize=minimize)
            assert status == "optimal", status
            _check(value, _exact(f, plain, minimize), minimize,
                   f"solve_lp({solver}) with sparse A, objective {k}")
            _, value, status, _ = solve_lp(f, A, _DYADIC_B, Aeq=Aeq, beq=beq,
                                           lp_solver=solver,
                                           minimize=minimize)
            assert status == "optimal", status
            _check(value, _exact(f, cut, minimize), minimize,
                   f"solve_lp({solver}) with sparse A and Aeq, objective {k}")


@pytest.mark.parametrize("empty", [False, True])
def test_feasibility_questions_without_inequality_rows(empty):
    """Systems with no inequality rows: equality rows over the box [0, 1]^2
    (x0 + x1 = 1, or = 3, which the box cannot meet) and a box alone (or a
    box with crossed bounds). ``check_feasibility`` is True exactly when the
    system holds a point and ``confirm_infeasible`` is True exactly when it
    does not; the exact feasibility in rationals decides. Catches the
    dimension read from the wrong argument, equality rows written as one
    inequality, and the box ignored when there are no rows."""
    Aeq = np.array([[1.0, 1.0]])
    beq = np.array([3.0 if empty else 1.0])
    lb, ub = np.zeros(2), np.ones(2)
    unit = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
    box_rows = unit + [[-v for v in row] for row in unit]
    truth = R.feasible([[Fraction(1), Fraction(1)]], [Fraction(beq[0])],
                       box_rows, [Fraction(1), Fraction(1), Fraction(0),
                                  Fraction(0)]) is not None
    if truth == empty:
        pytest.fail("test setup: the exact feasibility is not as intended")
    assert check_feasibility(Aeq=Aeq, beq=beq, lb=lb, ub=ub) is truth
    assert confirm_infeasible(None, None, lb, ub, Aeq=Aeq, beq=beq) is (
        not truth)
    lo, hi = (np.ones(2), np.zeros(2)) if empty else (lb, ub)
    assert check_feasibility(lb=lo, ub=hi) is (not empty)
    assert confirm_infeasible(None, None, lo, hi) is empty


def test_feasibility_without_any_dimension_refuses():
    """No rows and no bounds: the dimension is unknown, ``ValueError``."""
    with pytest.raises(ValueError):
        check_feasibility()


def test_batch_refuses_mismatched_inputs():
    """``solve_lp_batch`` with no objectives returns an empty list; flags of
    another length than the objectives, or objectives of different lengths,
    are a ``ValueError`` (never a value for the wrong objective)."""
    assert solve_lp_batch([], _DYADIC_A, _DYADIC_B) == []
    with pytest.raises(ValueError):
        solve_lp_batch(_DYADIC_F[:2], _DYADIC_A, _DYADIC_B,
                       minimize_flags=[True])
    with pytest.raises(ValueError):
        solve_lp_batch([_DYADIC_F[0], _DYADIC_F[1][:2]], _DYADIC_A,
                       _DYADIC_B)


# --------------------------------------------------------------------------- #
# The pad of an LP optimum used as a bound, and the solver for such LPs         #
# --------------------------------------------------------------------------- #
def test_the_pad_of_an_lp_optimum_is_outward_for_every_sign():
    """``_lp_pad(v)``, the pad every range-reading kernel adds to an LP
    optimum, is the documented ``1e-7 (1 + |v|)``: positive for optima below
    -1, at 0 and above 1, scalar and array (exact comparison). Catches a pad
    computed from the signed optimum (negative below -1, so both padded
    bounds move inward)."""
    from n2v.utils.lpsolver import _LP_PAD, _lp_pad
    values = [-1e6, -3.5, -1.0, -0.25, 0.0, 0.5, 2.0, 1e6]
    for v in values + [np.array(values)]:
        pad = np.atleast_1d(np.asarray(_lp_pad(v), dtype=np.float64))
        for p, x in zip(pad, np.atleast_1d(v)):
            want = Fraction(_LP_PAD) * (1 + abs(Fraction(float(x))))
            assert p > 0 and abs(Fraction(float(p)) - want) <= want * Fraction(
                1, 2 ** 50), f"_lp_pad({x!r}) = {p!r}, documented {float(want)!r}"


def test_padded_lp_ranges_move_outward_and_keep_constants():
    """``_outward_lp_ranges(lo, hi, V)`` pads every entry whose generator row
    has a nonzero (rows with some zero entries included) outward by
    ``_lp_pad`` on both sides, strictly, and returns an entry whose generator
    row is all zero as its centre exactly, also for an image-shaped basis.
    Catches the upper bound moved inward (an integer maximum floored), an
    entry with some zero generators taken for a constant, and the pad
    dropped."""
    from n2v.utils.lpsolver import _lp_pad, _outward_lp_ranges
    V = np.array([[0.5, 1.0, 0.0, 0.0],      # one zero, one nonzero
                  [-2.0, 0.0, 0.0, 0.0],     # constant
                  [3.0, 0.25, -0.5, 0.125],  # no zero
                  [-1.5, 0.0, 0.0, 2.0]])    # zeros and a nonzero
    lo = np.array([-0.5, -2.0, 2.125, -3.5])
    hi = np.array([1.5, -2.0, 3.875, 0.0])
    for basis in (V, V.reshape(2, 2, 1, 4)):
        new_lo, new_hi = _outward_lp_ranges(lo, hi, basis)
        assert new_lo[1] == -2.0 and new_hi[1] == -2.0, (new_lo, new_hi)
        for i in (0, 2, 3):
            assert new_lo[i] < lo[i] and new_hi[i] > hi[i], (i, new_lo, new_hi)
            assert Fraction(float(new_lo[i])) <= (
                Fraction(float(lo[i])) - Fraction(float(_lp_pad(lo[i]))) / 2)
            assert Fraction(float(new_hi[i])) >= (
                Fraction(float(hi[i])) + Fraction(float(_lp_pad(hi[i]))) / 2)


@pytest.mark.parametrize("asked, kept", [
    ("SCS", False), ("CLARABEL", False), ("OSQP", False),
    ("highs-ds", True), ("highs-ipm", True), ("linprog", True),
    ("highs", True)])
def test_bound_lps_are_routed_to_highs(asked, kept):
    """``_bound_lp_solver``, which picks the solver for every LP whose
    optimum becomes a bound inside a reach: a CVXPY solver is replaced by
    HiGHS (the pad is sized for HiGHS; SCS optima lie up to 5.7e-5 inside
    the range), and a HiGHS variant the caller chose is kept. Catches the
    requested solver returned unchanged."""
    from n2v.utils.lp_solver_enum import LPSolver
    from n2v.utils.lpsolver import _bound_lp_solver
    got = _bound_lp_solver(asked)
    assert got.is_scipy(), f"bound LPs asked for {asked!r} run on {got!r}"
    if kept:
        assert got == LPSolver(asked), (asked, got)


# --------------------------------------------------------------------------- #
# Rows of very different scales                                                 #
# --------------------------------------------------------------------------- #
# Each problem is a polytope given by rows only, so no certificate can repair
# a wrong solve, and every vertex is a float64 number (the exact optima come
# from vertex enumeration in rationals, not from these notes). Each puts one
# step of the row handling to work:
#
# - "tiny-entry": x0 - 2**-33 x1 <= 0 with 0 <= x1 <= 2**33, so x0 <= 1. The
#   entry 2**-33 (1.2e-10) lies between HiGHS's default small-matrix threshold
#   (1e-9) and the 1e-12 every model here asks for; a model built with the
#   default drops it and solves x0 <= 0.
# - "wide-row": 1024 x0 - 2**-33 x1 <= 0 with x1 <= 2**43, so x0 <= 1 again.
#   Scaling this row into [0.5, 1) would push 2**-33 below 1e-12; the row
#   scaling must stop short of that.
# - "huge-row": 2**66 x0 + 2**-30 x1 <= 2**66 with x1 fixed at 0 by two rows
#   (a degenerate coordinate), so x0 <= 1. HiGHS rejects a row with an entry
#   above 1e15 and the wide range of the row stops the scaling early, so only
#   the cap that scales every row below 2**49 lets the row reach HiGHS (lost,
#   x0 has no upper bound).
# - "row-factors": the dyadic polytope with every row multiplied by its own
#   power of two between 2**-40 and 2**40 (exact: the same polytope). Row 6,
#   which binds, then has entries of 2**-42 (2.3e-13): it must be scaled up
#   before HiGHS sees it, and every right-hand side must be scaled with its
#   row.
# - "entry-below-threshold": the tiny-entry problem with 2**-41 (4.5e-13),
#   below the lowered threshold, where HiGHS drops it (registry SET-7).
_ROW_FACTORS = 2.0 ** np.array([-30, 27, -12, 40, 3, -27, -40, 30, 0, -20,
                                12])


def _tiny_entry(entry):
    return (np.array([[1.0, -entry], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            np.array([0.0, 0.0, 1.0 / entry, 0.0]))


_SCALED_PROBLEMS = {
    "tiny-entry": _tiny_entry(2.0 ** -33),
    "wide-row": (np.array([[1024.0, -2.0 ** -33], [-1.0, 0.0], [0.0, 1.0],
                           [0.0, -1.0]]),
                 np.array([0.0, 0.0, 2.0 ** 43, 0.0])),
    "huge-row": (np.array([[2.0 ** 66, 2.0 ** -30], [-1.0, 0.0], [0.0, 1.0],
                           [0.0, -1.0]]),
                 np.array([2.0 ** 66, 0.0, 0.0, 0.0])),
    "row-factors": (_DYADIC_A * _ROW_FACTORS[:, None],
                    _DYADIC_B * _ROW_FACTORS),
    "entry-below-threshold": _tiny_entry(2.0 ** -41),
}

_ENTRY_DROPPED = known_bug(
    17, "HiGHS drops constraint coefficients at or below 1e-12 and an LP "
        "without a finite box has no certificate that would catch it: "
        "solve_lp, solve_lp_batch, the careful re-solve and the Star range "
        "queries answer max x0 = 0 over x0 - 2**-41 x1 <= 0, 0 <= x1 <= "
        "2**41, for the exact maximum 1 (registry SET-7)")


def _float_vertices(A, b, lb=None, ub=None):
    """``_vertices`` of a problem whose vertices must all be float64 numbers
    (so that an exact optimum leaves no rounding to forgive)."""
    verts = _vertices(A, b, lb, ub)
    if not all(Fraction(float(x)) == x for v in verts for x in v):
        pytest.fail("test setup: a vertex is not a float64 number")
    return verts


@pytest.mark.parametrize("problem", [
    pytest.param(name, marks=(_ENTRY_DROPPED
                              if name == "entry-below-threshold" else ()))
    for name in _SCALED_PROBLEMS])
def test_rows_of_very_different_scales_are_solved_as_given(problem):
    """The problems above with no finite box, min and max of each coordinate
    and of their sum: the exact optimum, on the outward side, from every
    HiGHS route (single solves, batches in an unsorted mixed order, the
    careful re-solve and its wrapper) and through the public routes (the
    range queries of the Star ``x = alpha`` over the same rows, and the
    finite predicate box that ``ensure_finite_predicate_box`` computes for
    it). Catches a HiGHS model built without the lowered small-matrix
    threshold, scipy not asked for it, a row scaling that ignores the
    guard on its smallest entry, drops the cap below 2**49, does not scale
    small rows up or forgets a right-hand side: each solves another
    polytope, and nothing here can certify the answer back."""
    A, b = _SCALED_PROBLEMS[problem]
    n = A.shape[1]
    verts = _float_vertices(A, b)
    objectives = [np.eye(n)[i] for i in range(n)] + [np.ones(n)]
    exact = {(k, mn): _exact(f, verts, mn) for k, f in enumerate(objectives)
             for mn in (True, False)}
    for (k, minimize), opt in exact.items():
        f = objectives[k]
        for solver in HIGHS_SINGLE:
            _check(_single(f, A, b, None, None, solver, minimize), opt,
                   minimize, f"solve_lp({solver}) {problem} objective {k}")
        for name, solve in (("solve_lp_no_presolve", solve_lp_no_presolve),
                            ("_solve_lp_careful", _solve_lp_careful)):
            _, value, status, _ = solve(f, A, b, None, None,
                                        minimize=minimize)
            assert status == "optimal", (name, problem, k, status)
            _check(value, opt, minimize, f"{name} {problem} objective {k}")
    order = [(k, (k + j) % 2 == 0) for j in (0, 1)
             for k in reversed(range(len(objectives)))]
    for solver in HIGHS_BATCH:
        values = solve_lp_batch([objectives[k] for k, _ in order], A, b,
                                None, None, [mn for _, mn in order],
                                lp_solver=solver)
        for (k, minimize), value in zip(order, values):
            _check(value, exact[k, minimize], minimize,
                   f"solve_lp_batch({solver}) {problem} objective {k}")
    star = Star(np.hstack([np.zeros((n, 1)), np.eye(n)]), A, b.reshape(-1, 1))
    for solver in ("default", "highs-ds"):
        lo, hi = star.get_ranges(lp_solver=solver)
        for i in range(n):
            what = f"Star({solver}) {problem} entry {i}"
            _check(lo[i, 0], exact[i, True], True, f"get_ranges {what}")
            _check(hi[i, 0], exact[i, False], False, f"get_ranges {what}")
            _check(star.get_min(i, solver), exact[i, True], True,
                   f"get_min {what}")
            _check(star.get_max(i, solver), exact[i, False], False,
                   f"get_max {what}")
    boxed, empty = ensure_finite_predicate_box(star)
    assert not empty, problem
    for i in range(n):
        assert Fraction(float(boxed.predicate_lb[i, 0])) <= exact[i, True], (
            f"{problem}: predicate box lower end {i} above the exact minimum")
        assert Fraction(float(boxed.predicate_ub[i, 0])) >= exact[i, False], (
            f"{problem}: predicate box upper end {i} below the exact maximum")


# --------------------------------------------------------------------------- #
# Equality rows with a finite box, and on the CVXPY route                       #
# --------------------------------------------------------------------------- #
_EQ_AEQ, _EQ_BEQ = np.array([[1.0, 0.0, -1.0]]), np.array([-1.0])
_EQ_LB = np.array([-2.0, -1.0, -0.5])
_EQ_UB = np.array([1.5, 0.75, 2.0])


def _cut_vertices(lb=None, ub=None):
    """The dyadic polytope cut by ``x0 - x2 = -1``, inside the box if given."""
    A = np.vstack([_DYADIC_A, _EQ_AEQ, -_EQ_AEQ])
    b = np.concatenate([_DYADIC_B, _EQ_BEQ, -_EQ_BEQ])
    return _dyadic_vertices(A, b, lb, ub)


def test_equality_rows_with_a_finite_box_keep_the_exact_optimum():
    """``Aeq x = beq`` with a finite box that cuts the polytope: every HiGHS
    name sends the solve to scipy, and the value is the exact optimum of the
    cut polytope in the box (dyadic, so the solver's own value is exact and
    within the pad). Catches the outward certificate applied although
    equality rows are present: it knows only the inequality rows and the
    box, so it returns a bound of the polytope without the cut."""
    verts = _cut_vertices(_EQ_LB, _EQ_UB)
    for solver in HIGHS_SINGLE:
        for k, f in enumerate(_DYADIC_F):
            for minimize in (True, False):
                _, value, status, _ = solve_lp(
                    f, _DYADIC_A, _DYADIC_B, Aeq=_EQ_AEQ, beq=_EQ_BEQ,
                    lb=_EQ_LB, ub=_EQ_UB, lp_solver=solver, minimize=minimize)
                assert status == "optimal", (solver, status)
                _check(value, _exact(f, verts, minimize), minimize,
                       f"solve_lp({solver}) with Aeq and a box, objective {k}")


@pytest.mark.parametrize("solver", CVXPY_SOLVERS)
def test_cvxpy_backends_keep_equality_rows(solver):
    """The equality cut above on the CVXPY route, without and with the finite
    box: each value within the solver's accuracy of the exact optimum of the
    cut polytope (``_CVXPY_TOL``; outwardness is registry row SET-6). Catches
    the equality rows dropped or written as one inequality on that route,
    which the other equality cells do not run."""
    tol = _CVXPY_TOL[solver]
    for lb, ub in ((None, None), (_EQ_LB, _EQ_UB)):
        verts = _cut_vertices(lb, ub)
        for k, f in enumerate(_DYADIC_F):
            for minimize in (True, False):
                _, value, status, _ = _cvxpy(lambda: solve_lp(
                    f, _DYADIC_A, _DYADIC_B, Aeq=_EQ_AEQ, beq=_EQ_BEQ, lb=lb,
                    ub=ub, lp_solver=solver, minimize=minimize))
                exact = float(_exact(f, verts, minimize))
                assert status in ("optimal", "optimal_inaccurate"), status
                assert abs(value - exact) <= tol * (1 + abs(exact)), (
                    solver, lb is not None, k, minimize, value, exact)


# --------------------------------------------------------------------------- #
# Solver failures are failures, never verdicts                                  #
# --------------------------------------------------------------------------- #
def _limit_problem():
    """A feasible LP in 4 variables over the box [-1, 1]^4 (x = 0 meets every
    row, since b > 0; data in multiples of 1/64) whose solve, stopped by an
    iteration limit on either scipy method, ends with an infeasible primal
    iterate: scipy's message then reads 'primal_status is Infeasible'
    (measured; the cells below do not depend on it, only the catch of a
    status read from the message does)."""
    rng = np.random.default_rng(3)
    A = np.round(rng.normal(size=(6, 4)) * 64) / 64
    b = np.round(rng.uniform(1, 2, size=6) * 64) / 64
    c = np.round(rng.normal(size=4) * 64) / 64
    if not all(Fraction(float(v)) > 0 for v in b):
        pytest.fail("test setup: x = 0 is not a point of the LP")
    return A, b, c, -np.ones(4), np.ones(4)


_LIMITS = {"highs-ds": {"options": {"maxiter": 1}},
           "highs-ipm": {"options": {"maxiter": 0}},
           "CLARABEL": {"max_iter": 1},
           "OSQP": {"max_iter": 1}}


@pytest.mark.parametrize("solver", list(_LIMITS))
def test_a_solve_stopped_by_a_limit_is_a_failure_not_a_verdict(solver):
    """An iteration limit passed through ``solve_lp``'s solver keywords, on a
    feasible LP, on the empty system and on the set open below: whatever the
    stopped solve says, 'infeasible' only for the empty system, 'unbounded'
    only for min x1 over the open set, and a value only with an optimal
    status and then the exact optimum (to the solver's accuracy). Catches a
    stopped scipy solve reported 'infeasible' because its message names an
    infeasible primal iterate, and a CVXPY limit status ('user_limit') mapped
    to 'infeasible'; a caller reads 'infeasible' as an empty set, which is
    vacuously safe. SCS is left out: stopped after one iteration it reports
    'optimal_inaccurate' with its iterate's value, which ``solve_lp`` passes
    on under that status (registry SET-6: CVXPY optima are used as they
    come)."""
    keywords = _LIMITS[solver]
    tol = _CVXPY_TOL.get(solver, PAD)
    A, b, c, lb, ub = _limit_problem()
    verts = R.vertices(*_rows(A, b, lb, ub))    # bounded: the box is finite
    if not verts:
        pytest.fail("test setup: the LP has no vertex")
    cases = [("feasible", c, A, b, lb, ub, mn, _exact(c, verts, mn))
             for mn in (True, False)]
    cases += [("empty", np.array([1.0, -1.0]), _EMPTY_A, _EMPTY_B, None,
               None, mn, None) for mn in (True, False)]
    cases += [("open", np.array([0.0, 1.0]), _OPEN_A, _OPEN_B, None, None,
               True, "unbounded"),
              ("open", np.array([0.0, 1.0]), _OPEN_A, _OPEN_B, None, None,
               False, Fraction(1))]
    for name, f, A_, b_, lb_, ub_, minimize, exact in cases:
        _, value, status, _ = _cvxpy(lambda: solve_lp(
            f, A_, b_, lb=lb_, ub=ub_, lp_solver=solver, minimize=minimize,
            **keywords))
        what = f"{solver} {keywords} on the {name} LP, minimize={minimize}"
        if status == "infeasible":
            assert name == "empty", f"{what}: 'infeasible'"
        if status == "unbounded":
            assert exact == "unbounded", f"{what}: 'unbounded'"
        if status in ("optimal", "optimal_inaccurate"):
            assert isinstance(exact, Fraction), f"{what}: value {value!r}"
            assert abs(value - float(exact)) <= tol * (1 + abs(exact)), (
                f"{what}: {value!r} for the exact {float(exact)!r}")
        else:
            assert value is None, f"{what}: status {status!r}, value {value!r}"


@pytest.mark.parametrize("solver", ["highs-ds", "highs-ipm"])
def test_a_model_error_is_a_failure_not_a_verdict(solver):
    """``scipy.sparse`` rows reach HiGHS unscaled, and a row with an entry
    above HiGHS's limit of 1e15 (2**54 x0 + x1 <= 2**54) is a model error,
    which scipy files under its infeasible code 2. The LP is feasible
    (x = 0), so the status must be a failure and carry no value, in both
    senses. Catches a status read from scipy's code 2 alone."""
    A = sparse.csr_matrix(np.array([[2.0 ** 54, 1.0], [-1.0, 0.0],
                                    [0.0, 1.0]]))
    b = np.array([2.0 ** 54, 0.5, 1.0])
    for minimize in (True, False):
        _, value, status, _ = solve_lp(np.array([1.0, 0.0]), A, b,
                                       lb=-np.ones(2), ub=np.ones(2),
                                       lp_solver=solver, minimize=minimize)
        assert value is None and status.startswith("failed"), (status, value)


def _scipy_result(code, message, n, m):
    return OptimizeResult(
        x=np.zeros(n) if code == 0 else None, fun=0.25 if code == 0 else None,
        status=code, success=code == 0, message=message, nit=1,
        ineqlin=OptimizeResult(marginals=np.zeros(m),
                               residual=np.zeros(m)))


# Every status code scipy's HiGHS interface reports, with its own message
# (scipy/optimize/_linprog_highs.py), and what the code proves.
_SCIPY_STATUSES = {
    "optimal": (0, "Optimization terminated successfully. (HiGHS Status 7: "
                   "Optimal)", "optimal"),
    "iteration-limit": (1, "Iteration limit reached. (HiGHS Status 14: "
                           "model_status is Iteration limit reached; "
                           "primal_status is Infeasible)", "failed"),
    "time-limit": (1, "Time limit reached. (HiGHS Status 13: model_status is "
                      "Time limit reached; primal_status is Infeasible)",
                   "failed"),
    "infeasible": (2, "The problem is infeasible. (HiGHS Status 8: "
                      "model_status is Infeasible; primal_status is None)",
                   "infeasible"),
    "model-error": (2, "(HiGHS Status 2: model_status is Model error; "
                       "primal_status is None)", "failed"),
    "unbounded": (3, "The problem is unbounded. (HiGHS Status 10: "
                     "model_status is Unbounded; primal_status is Feasible)",
                  "unbounded"),
    "unbounded-or-infeasible": (4, "The problem is unbounded or infeasible. "
                                   "(HiGHS Status 9: model_status is "
                                   "Primal infeasible or unbounded; "
                                   "primal_status is Infeasible)", "failed"),
    "solve-error": (4, "(HiGHS Status 4: model_status is Solve error; "
                       "primal_status is Infeasible)", "failed"),
}


@pytest.mark.parametrize("case", list(_SCIPY_STATUSES))
def test_every_scipy_status_means_what_it_proves(case, monkeypatch):
    """scipy's ``linprog`` replaced by a stub that reports one status code
    with scipy's own message, on the scipy routes (``solve_lp`` and the
    ``solve_lp_batch`` loop on 'highs-ds'): code 0 gives the value, code 2
    with "The problem is infeasible" alone gives 'infeasible', code 3 with
    "The problem is unbounded" alone gives 'unbounded', and every limit,
    model error and numerical failure is a failure with no value (the batch
    gives None for all but the optimum). Catches a verdict read from the
    message text or from the code alone: a failure read as 'infeasible' is
    an empty set to the caller, which is vacuously safe."""
    code, message, proves = _SCIPY_STATUSES[case]
    m, n = _OPEN_A.shape
    calls = []

    def stub(c, *args, **kwargs):
        calls.append(kwargs.get("options", {}))
        return _scipy_result(code, message, n, m)

    monkeypatch.setattr(lpsolver, "scipy_linprog", stub)
    f = np.array([1.0, 0.0])
    _, value, status, _ = solve_lp(f, _OPEN_A, _OPEN_B, lp_solver="highs-ds")
    if proves == "optimal":
        assert status == "optimal" and value == 0.25, (status, value)
    elif proves == "failed":
        assert status.startswith("failed") and value is None, (status, value)
    else:
        assert status == proves and value is None, (status, value)
    values = solve_lp_batch([f, f], _OPEN_A, _OPEN_B, None, None,
                            [True, False], lp_solver="highs-ds")
    want = [0.25, -0.25] if proves == "optimal" else [None, None]
    assert values == want, (case, values)
    assert calls, "the stub was not called"


class _ReportedStatus(cp.Problem):
    """A CVXPY problem whose solve reports ``report`` with the value 0.25."""

    report = "optimal"
    solver_stats = types.SimpleNamespace(solver_name="stub", num_iters=0,
                                         setup_time=0.0, solve_time=0.0)

    def solve(self, *args, **kwargs):
        return None

    @property
    def status(self):
        return type(self).report

    @property
    def value(self):
        return 0.25


# Every status CVXPY defines and what it proves for the LP that was asked.
_CVXPY_STATUSES = {
    "optimal": "optimal", "optimal_inaccurate": "optimal",
    "infeasible": "infeasible", "unbounded": "unbounded",
    "infeasible_inaccurate": "failed", "unbounded_inaccurate": "failed",
    "infeasible_or_unbounded": "failed", "user_limit": "failed",
    "solver_error": "failed"}


@pytest.mark.parametrize("report", list(_CVXPY_STATUSES))
def test_every_cvxpy_status_means_what_it_proves(report, monkeypatch):
    """CVXPY's problem replaced by one that reports a status, on the CVXPY
    routes (``solve_lp`` and ``solve_lp_batch`` on 'CLARABEL'): an optimal
    status (accurate or not) gives the value under that status, 'infeasible'
    and 'unbounded' give their verdict with no value, and every other status
    (the inaccurate verdicts, 'infeasible_or_unbounded', a limit, a solver
    error) gives no value and neither verdict (the batch gives None for all
    but an optimum). Catches an uncertified or undecided status mapped to
    'infeasible' or 'unbounded', and a value given with it."""
    proves = _CVXPY_STATUSES[report]
    monkeypatch.setattr(_ReportedStatus, "report", report)
    monkeypatch.setattr(lpsolver, "cp", types.SimpleNamespace(
        Variable=cp.Variable, Minimize=cp.Minimize, Maximize=cp.Maximize,
        Problem=_ReportedStatus))
    f = np.array([1.0, 0.0])
    _, value, status, _ = solve_lp(f, _OPEN_A, _OPEN_B, lp_solver="CLARABEL")
    if proves == "optimal":
        assert status == report and value == 0.25, (status, value)
    elif proves == "failed":
        assert value is None, (status, value)
        assert status not in ("optimal", "optimal_inaccurate", "infeasible",
                              "unbounded"), status
    else:
        assert status == proves and value is None, (status, value)
    values = solve_lp_batch([f], _OPEN_A, _OPEN_B, lp_solver="CLARABEL")
    assert values == ([0.25] if proves == "optimal" else [None]), values


def _raising(*args, **kwargs):
    raise RuntimeError("solver crashed")


def test_an_exception_inside_a_solver_is_a_failure(monkeypatch):
    """scipy's ``linprog`` and CVXPY's solve replaced by functions that raise:
    ``solve_lp`` returns an 'error: ...' status with no value on the scipy and
    the CVXPY routes, and the scipy batch gives None with a warning. Catches
    an exception turned into a verdict or a value."""
    monkeypatch.setattr(lpsolver, "scipy_linprog", _raising)
    monkeypatch.setattr(_ReportedStatus, "solve", _raising)
    monkeypatch.setattr(lpsolver, "cp", types.SimpleNamespace(
        Variable=cp.Variable, Minimize=cp.Minimize, Maximize=cp.Maximize,
        Problem=_ReportedStatus))
    f = np.array([1.0, 0.0])
    for solver in ("highs-ds", "highs-ipm", "CLARABEL"):
        _, value, status, _ = solve_lp(f, _OPEN_A, _OPEN_B, lp_solver=solver)
        assert value is None and status.startswith("error"), (solver, status)
    with pytest.warns(RuntimeWarning, match="solver crashed"):
        assert solve_lp_batch([f], _OPEN_A, _OPEN_B,
                              lp_solver="highs-ds") == [None]


# --------------------------------------------------------------------------- #
# Feasibility through the solve paths                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("box", ["none", "half_open", "finite"])
def test_an_unconfirmed_infeasible_report_is_not_emptiness(box, monkeypatch):
    """``check_feasibility`` on the dyadic polytope and its thin slab while
    every ``solve_lp`` call reports 'infeasible' (stubbed): a first solve's
    'infeasible' is not a certificate, so the systems, which hold points,
    must still not be called empty. The finite box runs on 'highs-ds', the
    route on which that first solve decides (with highspy and a finite box
    the careful solve decides alone). Catches ``check_feasibility`` reading
    the first solve's 'infeasible' as emptiness without confirming it."""
    reported = []

    def infeasible(*args, **kwargs):
        reported.append(kwargs.get("lp_solver"))
        return None, None, "infeasible", {}

    monkeypatch.setattr(lpsolver, "solve_lp", infeasible)
    slab_b = _DYADIC_B.copy()
    slab_b[0] = -1.875 + 1.0 / 1024
    _vertices(_DYADIC_A, slab_b)            # setup check: the slab holds points
    lb, ub, solver = {"none": (None, None, "default"),
                      "half_open": (np.full(3, -10.0), None, "default"),
                      "finite": (np.full(3, -10.0), np.full(3, 10.0),
                                 "highs-ds")}[box]
    for b in (_DYADIC_B, slab_b):
        assert check_feasibility(A=_DYADIC_A, b=b, lb=lb, ub=ub,
                                 lp_solver=solver) is True
    if not reported:
        pytest.fail("test setup: check_feasibility made no first solve")


@pytest.mark.parametrize("solver", CVXPY_SOLVERS)
def test_feasibility_on_a_cvxpy_solver(solver):
    """``check_feasibility`` with a CVXPY solver, whose first solve runs on
    the CVXPY route: a system that holds points (the dyadic polytope, the
    thin slab, and rows of size 1e-8 around a known point) is never called
    empty, and the two empty systems are called empty, with no box and a
    finite one. ``LPFailure`` (the solver did not decide) is allowed in both
    cases. Catches a CVXPY status that is not a proof of emptiness taken as
    one, and an emptiness that the route never reports."""
    slab_b = _DYADIC_B.copy()
    slab_b[0] = -1.875 + 1.0 / 1024
    feasible = [(_DYADIC_A, _DYADIC_B), (_DYADIC_A, slab_b),
                _tiny_feasible(2, 1e-8)]
    cut_A = np.vstack([_DYADIC_A, [[-1.0, 0.0, 0.0]]])
    cut_b = np.concatenate([_DYADIC_B, [-2.0]])
    empty = [(_EMPTY_A, _EMPTY_B), (cut_A, cut_b)]
    for systems, want in ((feasible, True), (empty, False)):
        for A, b in systems:
            n = A.shape[1]
            for lb, ub in ((None, None), (np.full(n, -10.0),
                                          np.full(n, 10.0))):
                try:
                    got = _cvxpy(lambda: check_feasibility(
                        A=A, b=b, lb=lb, ub=ub, lp_solver=solver))
                except LPFailure:
                    continue
                assert got is want, (solver, want, lb is None)


# --------------------------------------------------------------------------- #
# Rows of zeros, the careful re-solve without an optimum, no variables          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rhs", [-1.0, 0.0])
def test_a_row_of_zeros_still_constrains(rhs):
    """The dyadic polytope with one more row that is all zero, ``0 x <= rhs``:
    with rhs -1 the system is empty and every route must say so (status
    'infeasible' and no value, None in a batch); with rhs 0 the row holds
    everywhere and every optimum is that of the dyadic polytope. Catches an
    all-zero row dropped before it reaches the solver (a star whose
    constraint lost its last generator keeps such a row)."""
    A = np.vstack([_DYADIC_A, np.zeros((1, 3))])
    b = np.concatenate([_DYADIC_B, [rhs]])
    f = _DYADIC_F[3]
    verts = _dyadic_vertices(_DYADIC_A, _DYADIC_B) if rhs == 0.0 else None
    for minimize in (True, False):
        results = [(s, *solve_lp(f, A, b, lp_solver=s, minimize=minimize)[1:3])
                   for s in HIGHS_SINGLE]
        results.append(("solve_lp_no_presolve", *solve_lp_no_presolve(
            f, A, b, None, None, minimize=minimize)[1:3]))
        results += [(f"batch {s}", solve_lp_batch(
            [f], A, b, None, None, [minimize], lp_solver=s)[0],
            "batch") for s in HIGHS_BATCH]
        for name, value, status in results:
            if rhs < 0:
                assert value is None, (name, value)
                assert status in ("infeasible", "batch"), (name, status)
            else:
                _check(value, _exact(f, verts, minimize), minimize,
                       f"{name} with a zero row")


def test_the_careful_solves_have_no_optimum_without_one():
    """``solve_lp_no_presolve`` and ``_solve_lp_careful`` (the re-solve of
    an undecided LP and the solve behind the finite predicate box) on the
    empty system, on a box with crossed bounds and on the set open below:
    'infeasible' for the first two and 'unbounded' for min x1 over the last,
    no value for any, while max x1 over that set keeps its exact value 1.
    Catches an unbounded objective reported 'infeasible' (an empty set to
    the caller) and a placeholder value given for an LP with no optimum."""
    for solve in (solve_lp_no_presolve, _solve_lp_careful):
        for minimize in (True, False):
            for A, b, lb, ub in ((_EMPTY_A, _EMPTY_B, None, None),
                                 (_OPEN_A, _OPEN_B, np.array([1.0, -1.0]),
                                  np.array([0.0, 1.0]))):
                _, value, status, _ = solve(np.array([1.0, -1.0]), A, b, lb,
                                            ub, minimize=minimize)
                assert value is None and status == "infeasible", (
                    solve.__name__, status, value)
        _, value, status, _ = solve(np.array([0.0, 1.0]), _OPEN_A, _OPEN_B,
                                    None, None, minimize=True)
        assert value is None and status == "unbounded", (solve.__name__,
                                                           status, value)
        _, value, status, _ = solve(np.array([0.0, 1.0]), _OPEN_A, _OPEN_B,
                                    None, None, minimize=False)
        assert status == "optimal", (solve.__name__, status)
        _check(value, Fraction(1), False, f"{solve.__name__} max x1")


def test_an_lp_without_variables_is_never_reported_empty():
    """An LP in zero variables with no rows: its one point (the empty vector)
    has the value 0. Every route answers 0, refuses with ``ValueError`` or
    reports a failure; none may call it 'infeasible' (an empty set to a
    caller) or 'unbounded', or give a value other than 0. Catches HiGHS's
    status for a model with no columns mapped to a verdict."""
    f = np.zeros(0)
    calls = [(f"solve_lp({s})", lambda s=s: solve_lp(f, lp_solver=s))
             for s in HIGHS_SINGLE + CVXPY_SOLVERS]
    calls.append(("solve_lp_no_presolve",
                  lambda: solve_lp_no_presolve(f, None, None, None, None)))
    for name, call in calls:
        try:
            _, value, status, _ = _cvxpy(call)
        except ValueError:
            continue
        assert status not in ("infeasible", "unbounded"), (name, status)
        assert value is None or value == 0, (name, value)
    for s in HIGHS_BATCH + ("CLARABEL",):
        try:
            with warnings.catch_warnings():
                # the scipy batch warns of a solve that raised and gives None
                warnings.filterwarnings("ignore", message="LP solve failed",
                                        category=RuntimeWarning)
                values = _cvxpy(lambda: solve_lp_batch([f], lp_solver=s))
        except ValueError:
            continue
        assert values in ([None], [0.0]), (s, values)
