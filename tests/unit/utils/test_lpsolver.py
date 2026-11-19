"""Tests for LP solver backends in n2v.utils.lpsolver."""

import warnings

import numpy as np
import pytest

from n2v.utils.lpsolver import (
    _HAS_HIGHSPY,
    solve_lp,
    solve_lp_batch,
)
from tests.known_bugs import known_bug


class TestSolveLpBatch:
    """Tests for solve_lp_batch."""

    def test_batch_matches_sequential(self):
        """Batch results match calling solve_lp individually.

        Problem: 0 <= x, y <= 1.
        Objectives: min x, max x, min y, max y, min x+y, max x+y.
        """
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        objectives = [
            np.array([1.0, 0.0]),  # x
            np.array([1.0, 0.0]),  # x
            np.array([0.0, 1.0]),  # y
            np.array([0.0, 1.0]),  # y
            np.array([1.0, 1.0]),  # x+y
            np.array([1.0, 1.0]),  # x+y
        ]
        minimize_flags = [True, False, True, False, True, False]

        batch_results = solve_lp_batch(
            objectives, lb=lb, ub=ub, minimize_flags=minimize_flags
        )

        # Compute sequential results via solve_lp
        sequential_results = []
        for obj, do_min in zip(objectives, minimize_flags):
            _, fval, status, _ = solve_lp(obj, lb=lb, ub=ub, minimize=do_min)
            sequential_results.append(fval)

        assert len(batch_results) == len(sequential_results)
        for i, (batch_val, seq_val) in enumerate(
            zip(batch_results, sequential_results)
        ):
            assert batch_val is not None, f"Batch result {i} is None"
            assert seq_val is not None, f"Sequential result {i} is None"
            assert batch_val == pytest.approx(
                seq_val, abs=1e-6
            ), f"Mismatch at index {i}: batch={batch_val}, sequential={seq_val}"

        # Verify expected values
        expected = [0.0, 1.0, 0.0, 1.0, 0.0, 2.0]
        for i, (val, exp) in enumerate(zip(batch_results, expected)):
            assert val == pytest.approx(exp, abs=1e-6), (
                f"Objective {i}: expected {exp}, got {val}"
            )

    def test_batch_with_lp_solver_param(self):
        """The lp_solver parameter is accepted and produces correct results."""
        lb = np.array([0.0])
        ub = np.array([5.0])
        objectives = [np.array([1.0])]

        result = solve_lp_batch(
            objectives, lb=lb, ub=ub, minimize_flags=[True], lp_solver='linprog'
        )
        assert len(result) == 1
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_batch_empty_objectives(self):
        """Empty list returns empty results."""
        result = solve_lp_batch([])
        assert result == []

    def test_batch_no_constraints(self):
        """Works with only bounds, no A/b constraints."""
        lb = np.array([2.0, 3.0])
        ub = np.array([5.0, 7.0])
        objectives = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ]
        minimize_flags = [True, False]

        results = solve_lp_batch(objectives, lb=lb, ub=ub, minimize_flags=minimize_flags)

        assert len(results) == 2
        assert results[0] == pytest.approx(2.0, abs=1e-6)  # min x = lb[0]
        assert results[1] == pytest.approx(7.0, abs=1e-6)  # max y = ub[1]

    def test_batch_mismatched_lengths_raises(self):
        """ValueError when objectives and minimize_flags lengths differ."""
        objectives = [np.array([1.0]), np.array([2.0])]
        minimize_flags = [True]  # length 1 vs 2 objectives

        with pytest.raises(ValueError, match="minimize_flags length"):
            solve_lp_batch(objectives, minimize_flags=minimize_flags)

    def test_batch_mismatched_objective_dims_raises(self):
        """ValueError when objectives have different lengths."""
        objectives = [np.array([1.0, 2.0]), np.array([1.0])]

        with pytest.raises(ValueError, match="same length"):
            solve_lp_batch(objectives)

    def test_batch_warns_on_solver_error(self):
        """Fallback path warns instead of silently returning None."""
        from unittest.mock import patch

        objectives = [np.array([1.0])]
        lb = np.array([0.0])
        ub = np.array([1.0])

        # Mock scipy_linprog to raise, guaranteeing the warning path
        with patch('n2v.utils.lpsolver.scipy_linprog', side_effect=RuntimeError("mock failure")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                results = solve_lp_batch(
                    objectives,
                    lb=lb,
                    ub=ub,
                    minimize_flags=[True],
                    lp_solver='highs-ds',  # Forces scipy fallback, not highspy batch
                )

        assert len(results) == 1
        assert results[0] is None
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) >= 1
        assert "LP solve failed for objective 0" in str(runtime_warnings[0].message)
        assert "mock failure" in str(runtime_warnings[0].message)

    def test_constrained_problem(self):
        """Cross-validate batch vs sequential on x+y<=3, x>=0, y>=0."""
        A = np.array([[1.0, 1.0]])
        b_vec = np.array([3.0])
        lb = np.array([0.0, 0.0])

        objectives = [
            np.array([1.0, 0.0]),  # x
            np.array([1.0, 0.0]),  # x
            np.array([0.0, 1.0]),  # y
            np.array([0.0, 1.0]),  # y
            np.array([1.0, 1.0]),  # x+y
            np.array([1.0, 1.0]),  # x+y
        ]
        minimize_flags = [True, False, True, False, True, False]

        batch_results = solve_lp_batch(
            objectives, A=A, b=b_vec, lb=lb, minimize_flags=minimize_flags
        )

        sequential_results = []
        for obj, do_min in zip(objectives, minimize_flags):
            _, fval, status, _ = solve_lp(obj, A=A, b=b_vec, lb=lb, minimize=do_min)
            sequential_results.append(fval)

        assert len(batch_results) == len(sequential_results)
        for i, (batch_val, seq_val) in enumerate(
            zip(batch_results, sequential_results)
        ):
            assert batch_val is not None, f"Batch result {i} is None"
            assert seq_val is not None, f"Sequential result {i} is None"
            assert batch_val == pytest.approx(
                seq_val, abs=1e-6
            ), f"Mismatch at index {i}: batch={batch_val}, sequential={seq_val}"

        # Expected: min x=0, max x=3, min y=0, max y=3, min x+y=0, max x+y=3
        expected = [0.0, 3.0, 0.0, 3.0, 0.0, 3.0]
        for i, (val, exp) in enumerate(zip(batch_results, expected)):
            assert val == pytest.approx(exp, abs=1e-6), (
                f"Objective {i}: expected {exp}, got {val}"
            )


@pytest.mark.skipif(not _HAS_HIGHSPY, reason="highspy not installed")
class TestSolveLpHighspy:
    """Tests for the direct highspy backend in solve_lp."""

    def test_highspy_matches_scipy(self):
        """Direct highspy gives the same result as scipy for a constrained
        LP, and both give the exact minimum -5/4 (at x = 1/2, y = 7/4), on
        the outward side: two n2v paths that agree can be wrong together."""
        from n2v.utils.lpsolver import _solve_lp_highspy, _solve_lp_scipy

        A = np.array([[1.0, 2.0], [-1.0, 0.0]])
        b = np.array([[4.0], [-0.5]])
        lb = np.array([0.0, 0.0])
        ub = np.array([3.0, 3.0])
        f = np.array([1.0, -1.0])

        x_h, fval_h, status_h, _ = _solve_lp_highspy(
            f, A, b, lb, ub, minimize=True,
        )
        x_s, fval_s, status_s, _ = _solve_lp_scipy(
            f, A, b, lb=lb, ub=ub, minimize=True,
        )

        assert status_h == 'optimal'
        assert status_s == 'optimal'
        assert fval_h == pytest.approx(fval_s, abs=1e-6)
        for fval in (fval_h, fval_s):
            assert fval <= -1.25 and fval == pytest.approx(-1.25, abs=1e-9)

    def test_highspy_maximization(self):
        """Maximization returns correct result."""
        from n2v.utils.lpsolver import _solve_lp_highspy

        lb = np.array([0.0, 0.0])
        ub = np.array([5.0, 5.0])
        f = np.array([1.0, 1.0])

        x, fval, status, _ = _solve_lp_highspy(
            f, lb=lb, ub=ub, minimize=False,
        )

        assert status == 'optimal'
        assert fval == pytest.approx(10.0, abs=1e-6)

    def test_highspy_unbounded(self):
        """Unbounded LP returns 'unbounded' status, not a value."""
        from n2v.utils.lpsolver import _solve_lp_highspy

        # min x subject to x + y <= 10, no lower bounds
        A = np.array([[1.0, 1.0]])
        b = np.array([[10.0]])
        f = np.array([1.0, -1.0])

        x, fval, status, _ = _solve_lp_highspy(f, A, b)

        assert x is None
        assert fval is None
        assert status == 'unbounded'

    def test_highspy_infeasible(self):
        """Infeasible LP returns None."""
        from n2v.utils.lpsolver import _solve_lp_highspy

        # x >= 5 and x <= 1: infeasible
        lb = np.array([5.0])
        ub = np.array([1.0])
        f = np.array([1.0])

        x, fval, status, _ = _solve_lp_highspy(
            f, lb=lb, ub=ub,
        )

        assert x is None
        assert fval is None
        assert status == 'infeasible'

    def test_routing_uses_highspy(self):
        """solve_lp routes to highspy when available and solver is 'linprog'."""
        lb = np.array([0.0])
        ub = np.array([1.0])
        f = np.array([1.0])

        _, fval, status, info = solve_lp(
            f, lb=lb, ub=ub, lp_solver='linprog', minimize=True,
        )

        assert status == 'optimal'
        assert fval == pytest.approx(0.0, abs=1e-6)
        assert info['solver'] == 'highspy'


# The HiGHS model the solvers build with presolve on (lowered small-matrix
# threshold, row-scaled rows) ended in kUnknown on a feasible 220-row star
# from ml4acopf that a presolve-off solve decides at once. The batch path
# recorded None, the single path 'failed', and the McCormick product then
# raised: four instances that are UNSAT came out unknown. Such an LP is now
# re-solved once with presolve off. HiGHS's kUnknown depends on its version,
# so these tests make the presolve-on model report it.
# Triangle-like polytope: x + 2y <= 4, x >= 0.5, box [0, 3]^2.
_A = np.array([[1.0, 2.0], [-1.0, 0.0]])
_B = np.array([4.0, -0.5])
_LB = np.zeros(2)
_UB = np.full(2, 3.0)
_F = np.array([1.0, -1.0])      # min f'x = -1.25 at (0.5, 1.75); max 3 at (3, 0)


def _report_undecided(monkeypatch, presolve_off_too=False):
    """Make every HiGHS model built with presolve on (and, with
    ``presolve_off_too``, every model) report kUnknown after a solve."""
    import highspy
    import n2v.utils.lpsolver as lpsolver

    class _Undecided:
        def __init__(self, h):
            self._h = h

        def __getattr__(self, name):
            return getattr(self._h, name)

        def getModelStatus(self):
            return highspy.HighsModelStatus.kUnknown

    real = lpsolver._new_highs

    def new_highs(presolve=True):
        h = real(presolve=presolve)
        return _Undecided(h) if (presolve or presolve_off_too) else h

    monkeypatch.setattr(lpsolver, "_new_highs", new_highs)


@pytest.mark.skipif(not _HAS_HIGHSPY, reason="highspy not installed")
class TestHighspyUndecidedRetry:
    """An LP the presolve-on HiGHS model leaves undecided is re-solved with
    presolve off; a failure of that re-solve stays a failure."""

    def test_batch_resolves_undecided_objectives(self, monkeypatch):
        _report_undecided(monkeypatch)
        lo, hi = solve_lp_batch([_F, _F], _A, _B, _LB, _UB, [True, False],
                                lp_solver='linprog')
        assert lo is not None and hi is not None
        assert lo <= -1.25 and lo == pytest.approx(-1.25, abs=1e-9)
        assert hi >= 3.0 and hi == pytest.approx(3.0, abs=1e-9)

    def test_single_resolves_undecided_lp(self, monkeypatch):
        _report_undecided(monkeypatch)
        _, fval, status, info = solve_lp(_F, _A, _B, lb=_LB, ub=_UB,
                                         lp_solver='linprog', minimize=True)
        assert status == 'optimal'
        assert info['solver'] == 'highspy-nopresolve'
        assert fval <= -1.25 and fval == pytest.approx(-1.25, abs=1e-9)

    def test_star_range_and_mccormick_product(self, monkeypatch):
        """The failure surfaced as Star.get_range -> (None, None) and a
        ValueError from the McCormick product of two stars."""
        from n2v.nn.reach import _mul_stars_mccormick
        from n2v.sets import Star

        _report_undecided(monkeypatch)
        V = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 1.0]])
        S = Star(V, _A, _B.reshape(-1, 1), _LB.reshape(-1, 1),
                 _UB.reshape(-1, 1))
        lo, hi = S.get_range(0, 'linprog')
        assert lo == pytest.approx(0.5, abs=1e-9) and lo <= 0.5
        assert hi == pytest.approx(3.0, abs=1e-9) and hi >= 3.0

        T = Star(V[::-1].copy(), _A, _B.reshape(-1, 1), _LB.reshape(-1, 1),
                 _UB.reshape(-1, 1))
        Z = _mul_stars_mccormick(S, T, 'linprog')
        zlo, zhi = Z.get_ranges('linprog')
        # sampled products x0 * (y + 0.5) and (y + 0.5) * x0 lie in the range
        rng = np.random.default_rng(0)
        pts = rng.uniform(_LB, _UB, size=(2000, 2))
        pts = pts[(pts @ _A.T <= _B).all(axis=1)]
        prod = pts[:, 0] * (pts[:, 1] + 0.5)
        assert np.all(zlo.ravel() <= prod.min()) and np.all(zhi.ravel() >= prod.max())

    def test_failed_retry_stays_a_failure(self, monkeypatch):
        _report_undecided(monkeypatch, presolve_off_too=True)
        assert solve_lp_batch([_F], _A, _B, _LB, _UB, [True],
                              lp_solver='linprog') == [None]
        x, fval, status, _ = solve_lp(_F, _A, _B, lb=_LB, ub=_UB,
                                      lp_solver='linprog', minimize=True)
        assert x is None and fval is None
        assert status.startswith('failed')


# --------------------------------------------------------------------------- #
# The point a solve returns                                                     #
# --------------------------------------------------------------------------- #
# x + 2y <= 4, x >= 0.5 and x - y <= 2.5 in the box [0, 3]^2 (three rows for
# two variables): min x - y = -5/4 at the single vertex (1/2, 7/4); max x - y
# = 5/2 on the whole edge from (5/2, 0) to (3, 1/2).
_PA = np.array([[1.0, 2.0], [-1.0, 0.0], [1.0, -1.0]])
_PB = np.array([4.0, -0.5, 2.5])


def _exact_optimum(f, A, b, lb, ub, minimize):
    from fractions import Fraction
    from tests._oracles import rational as R
    rows = [[Fraction(float(v)) for v in r] for r in A]
    rhs = [Fraction(float(v)) for v in b]
    n = A.shape[1]
    for j in range(n):
        for sign, bound in ((1, ub), (-1, lb)):
            unit = [Fraction(0)] * n
            unit[j] = Fraction(sign)
            rows.append(unit)
            rhs.append(sign * Fraction(float(bound[j])))
    values = [R.dot([Fraction(float(v)) for v in f], x)
              for x in R.vertices(rows, rhs)]
    if not values:
        pytest.fail("test setup: the polytope is empty")
    return min(values) if minimize else max(values)


@pytest.mark.parametrize("route", [
    "default", "linprog", "highs", "highs-ds", "highs-ipm", "CLARABEL",
    "solve_lp_no_presolve", "_solve_lp_careful"])
def test_the_returned_point_attains_the_optimum(route):
    """``x`` of ``solve_lp`` (each HiGHS name and a CVXPY solver), of the
    careful re-solve and of its wrapper: a point of the polytope (every row
    and bound within the solver's tolerance) whose objective value is the
    exact optimum, for a minimum at a single vertex and a maximum on a whole
    edge, with three rows for two variables. ``x`` is what the refinement
    engine takes as a counterexample candidate (``refine/witness.py``).
    Catches the row activities ``A x`` returned in place of ``x``, a point
    of the other sense, and a point read from another model."""
    from n2v.utils.lpsolver import _solve_lp_careful, solve_lp_no_presolve
    tol = 1e-6 if route == "CLARABEL" else 1e-9
    f = np.array([1.0, -1.0])
    for minimize in (True, False):
        if route == "solve_lp_no_presolve":
            x, fval, status, _ = solve_lp_no_presolve(f, _PA, _PB, _LB, _UB,
                                                      minimize=minimize)
        elif route == "_solve_lp_careful":
            x, fval, status, _ = _solve_lp_careful(f, _PA, _PB, _LB, _UB,
                                                   minimize=minimize)
        else:
            x, fval, status, _ = solve_lp(f, _PA, _PB, lb=_LB, ub=_UB,
                                          lp_solver=route, minimize=minimize)
        assert status == 'optimal', (route, status)
        x = np.asarray(x, dtype=np.float64).ravel()
        assert x.shape == (2,), (route, x)
        assert np.all(_PA @ x <= _PB + tol * (1 + np.abs(_PB))), (route, x)
        assert np.all(x >= _LB - tol) and np.all(x <= _UB + tol), (route, x)
        exact = float(_exact_optimum(f, _PA, _PB, _LB, _UB, minimize))
        assert abs(f @ x - exact) <= tol * (1 + abs(exact)), (
            route, minimize, x, f @ x, exact)


# --------------------------------------------------------------------------- #
# Without highspy                                                               #
# --------------------------------------------------------------------------- #
def test_without_highspy_every_entry_answers_on_scipy(monkeypatch):
    """With highspy absent (``_HAS_HIGHSPY`` set to False, as on an install
    without it) the default solver runs on scipy: ``solve_lp`` and the batch
    loop give the exact optimum of min and max x - y over the three rows with
    and without the box [0, 3]^2 (outward, within the pad of 1e-7 relative),
    ``_solve_lp_careful`` does the same on scipy with presolve off, and
    ``solve_lp_no_presolve`` reports a failure rather than a value. Catches
    the fallback routes reading the sense, the rows or the bounds wrongly,
    and a careful re-solve that answers without a solver."""
    import n2v.utils.lpsolver as lpsolver
    from fractions import Fraction
    monkeypatch.setattr(lpsolver, "_HAS_HIGHSPY", False)
    f = np.array([1.0, -1.0])
    box_rows = np.vstack([np.eye(2), -np.eye(2)])
    for lb, ub, A, b in ((_LB, _UB, _PA, _PB),
                         (None, None, np.vstack([_PA, box_rows]),
                          np.concatenate([_PB, _UB, -_LB]))):
        exact = {mn: _exact_optimum(f, _PA, _PB, _LB, _UB, mn)
                 for mn in (True, False)}

        def check(value, minimize, what):
            assert value is not None, what
            got = Fraction(float(value))
            assert (got <= exact[minimize]) if minimize else (
                got >= exact[minimize]), (what, value)
            assert abs(float(got - exact[minimize])) <= 1e-7 * (
                1 + abs(float(exact[minimize]))), (what, value)

        for minimize in (True, False):
            _, fval, status, info = lpsolver.solve_lp(f, A, b, lb=lb, ub=ub,
                                                      minimize=minimize)
            assert status == 'optimal', status
            assert info['solver'].startswith('scipy'), info
            check(fval, minimize, "solve_lp")
            _, fval, status, _ = lpsolver._solve_lp_careful(f, A, b, lb, ub,
                                                            minimize=minimize)
            assert status == 'optimal', status
            check(fval, minimize, "_solve_lp_careful")
            x, fval, status, _ = lpsolver.solve_lp_no_presolve(
                f, A, b, lb, ub, minimize=minimize)
            assert x is None and fval is None and status.startswith('failed')
        lo, hi = lpsolver.solve_lp_batch([f, f], A, b, lb, ub, [True, False])
        check(lo, True, "solve_lp_batch minimum")
        check(hi, False, "solve_lp_batch maximum")


# --------------------------------------------------------------------------- #
# scipy's numerical-trouble retry                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("retry_decides", [True, False])
def test_a_numerical_failure_is_retried_without_presolve(monkeypatch,
                                                         retry_decides):
    """scipy's ``linprog`` stubbed to report numerical trouble (status 4)
    whenever presolve is on: the scipy routes (``solve_lp`` and the batch
    loop on 'highs-ds') solve again with presolve off, and with that solve
    deciding give its exact optimum; when the retry fails too, the result is
    a failure with no value. Catches the retry dropped, run with presolve
    still on, or its failure taken over a decided first solve."""
    import n2v.utils.lpsolver as lpsolver
    from scipy.optimize import OptimizeResult
    real = lpsolver.scipy_linprog
    calls = []

    def troubled(c, *args, options=None, **kwargs):
        options = dict(options or {})
        calls.append(options.get('presolve', True))
        if options.get('presolve', True) is not False or not retry_decides:
            return OptimizeResult(
                x=None, fun=None, status=4, success=False, nit=0,
                message="(HiGHS Status 4: model_status is Solve error; "
                        "primal_status is Infeasible)")
        return real(c, *args, options=options, **kwargs)

    monkeypatch.setattr(lpsolver, "scipy_linprog", troubled)
    f = np.array([1.0, -1.0])
    _, fval, status, _ = lpsolver.solve_lp(f, _PA, _PB, lp_solver='highs-ds',
                                           minimize=True, lb=_LB, ub=_UB)
    values = lpsolver.solve_lp_batch([f, f], _PA, _PB, _LB, _UB,
                                     [True, False], lp_solver='highs-ds')
    assert calls and calls[0] is not False, calls
    assert False in calls, f"no solve with presolve off: {calls}"
    if retry_decides:
        assert status == 'optimal', status
        assert fval <= -1.25 and fval == pytest.approx(-1.25, abs=1e-9)
        assert values[0] <= -1.25 and values[0] == pytest.approx(-1.25,
                                                                  abs=1e-9)
        assert values[1] >= 2.5 and values[1] == pytest.approx(2.5, abs=1e-9)
    else:
        assert fval is None and status.startswith('failed'), status
        assert values == [None, None], values


# --------------------------------------------------------------------------- #
# Solver keywords and malformed problems                                        #
# --------------------------------------------------------------------------- #
def _limit_lp():
    """A feasible LP in 4 variables (x = 0 meets every row) that the scipy
    methods do not finish in one iteration (data in multiples of 1/64)."""
    rng = np.random.default_rng(3)
    A = np.round(rng.normal(size=(6, 4)) * 64) / 64
    b = np.round(rng.uniform(1, 2, size=6) * 64) / 64
    c = np.round(rng.normal(size=4) * 64) / 64
    return c, A, b, -np.ones(4), np.ones(4)


@pytest.mark.parametrize("solver, keywords", [
    pytest.param("default", {"options": {"maxiter": 1}}, marks=known_bug(
        17, "solve_lp drops its solver keywords on the direct highspy route "
            "('default', 'linprog' and 'highs' without equality rows): "
            "options={'maxiter': 1}, which the same call honours on "
            "'highs-ds' and 'highs-ipm', is ignored and the LP is solved to "
            "optimality, as is any time limit given the same way"),
        id="default"),
    pytest.param("highs-ds", {"options": {"maxiter": 1}}, id="highs-ds"),
    pytest.param("highs-ipm", {"options": {"maxiter": 1}}, id="highs-ipm"),
    pytest.param("CLARABEL", {"max_iter": 1}, id="CLARABEL"),
    pytest.param("OSQP", {"max_iter": 1}, id="OSQP")])
def test_solver_options_reach_the_solver_or_are_refused(solver, keywords):
    """``solve_lp(..., **keywords)``, which its docstring forwards to the
    solver, with an iteration limit of 1 (scipy's ``options={'maxiter': 1}``,
    CVXPY's ``max_iter=1``) on an LP that needs more than one iteration: the
    limit must stop the solve (no value and no optimal status or verdict) or
    the call must refuse the keyword (``NotImplementedError``,
    ``ValueError`` or ``TypeError``). An optimum means the keyword was
    dropped. Catches solver keywords dropped on a route."""
    c, A, b, lb, ub = _limit_lp()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Solution may be "
                                    "inaccurate", category=UserWarning)
            _, fval, status, _ = solve_lp(c, A, b, lb=lb, ub=ub,
                                          lp_solver=solver, **keywords)
    except (NotImplementedError, ValueError, TypeError):
        return
    assert fval is None and status not in (
        'optimal', 'optimal_inaccurate', 'infeasible', 'unbounded'), (
        status, fval)


# Each case keeps a finite box wherever its kind allows, so that every way of
# reading the arrays has a finite optimum and an answer shows as a value.
_MALFORMED = {
    # A with a column count other than len(f), wider and narrower
    "columns": [dict(A=np.hstack([_PA, np.ones((3, 1))]), b=_PB, lb=_LB,
                     ub=_UB),
                dict(A=_PA[:, :1], b=_PB, lb=_LB, ub=_UB)],
    # lb or ub with a size other than len(f)
    "bounds": [dict(A=_PA, b=_PB, lb=_LB[:1], ub=_UB),
               dict(A=_PA, b=_PB, lb=_LB, ub=np.full(3, 3.0))],
    # rows given without their right-hand side
    "rhs-missing": [dict(A=_PA, lb=_LB, ub=_UB),
                    dict(A=_PA, b=_PB, Aeq=np.array([[1.0, 1.0]]), lb=_LB,
                         ub=_UB)],
    # a right-hand side of another length than the rows (the neighbour)
    "rhs-length": [dict(A=_PA, b=_PB[:2], lb=_LB, ub=_UB)],
}


def _malformed_marker(kind, what):
    return known_bug(
        17, f"solve_lp and solve_lp_batch answer an LP whose {what} instead "
            f"of refusing it with ValueError: the highspy route answers "
            f"another LP (rows naming a missing column are dropped, a short "
            f"bound array is read past its end), and the scipy and CVXPY "
            f"routes answer where they can broadcast or ignore an argument")


@pytest.mark.parametrize("kind", [
    pytest.param("columns", marks=_malformed_marker(
        "columns", "rows have another number of columns than the objective")),
    pytest.param("bounds", marks=_malformed_marker(
        "bounds", "bounds have another size than the objective")),
    pytest.param("rhs-missing", marks=_malformed_marker(
        "rhs-missing", "rows come without their right-hand side")),
    "rhs-length"])
def test_a_malformed_lp_is_refused(kind):
    """An LP whose arrays do not fit together cannot be read: ``solve_lp``
    on each route (highspy, scipy, CVXPY) and ``solve_lp_batch`` must raise
    ``ValueError``, or at least give neither a value nor a verdict (a
    failure status, None in a batch). A value answers some other LP than the
    one given (rows dropped or broadcast, a bound read from beyond the
    array), and 'infeasible' would call it empty. Catches a shape check that
    is missing on a route."""
    f = np.array([1.0, -1.0])
    for args in _MALFORMED[kind]:
        for solver in ("default", "highs-ds", "CLARABEL"):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="LP solve failed",
                                            category=RuntimeWarning)
                    single = solve_lp(f, lp_solver=solver, **args)
                    batch = solve_lp_batch(
                        [f], args.get('A'), args.get('b'), args.get('lb'),
                        args.get('ub'), [True], lp_solver=solver) if (
                        'Aeq' not in args) else [None]
            except ValueError:
                continue
            what = f"{kind} {sorted(args)} on {solver}"
            assert single[1] is None and single[2] not in (
                'optimal', 'optimal_inaccurate', 'infeasible', 'unbounded'), (
                f"{what}: solve_lp answered {single[1:3]}")
            assert batch == [None], f"{what}: solve_lp_batch answered {batch}"


def test_an_unknown_solver_name_is_refused():
    """``solve_lp`` and ``solve_lp_batch`` with a solver name that is not one
    (``'simplex'``) raise ``ValueError``, and with a name of another type
    (``3``) ``TypeError``, before any solve; the names are matched without
    regard to case, so ``'HIGHS-DS'`` and ``'clarabel'`` answer with the
    exact minimum -5/4. Catches an unknown name passed on to a backend that
    picks some solver of its own (CVXPY's default) or answers nothing."""
    f = np.array([1.0, -1.0])
    for call in (lambda s: solve_lp(f, _PA, _PB, lb=_LB, ub=_UB, lp_solver=s),
                 lambda s: solve_lp_batch([f], _PA, _PB, _LB, _UB,
                                          lp_solver=s)):
        with pytest.raises(ValueError, match="Unknown LP solver"):
            call('simplex')
        with pytest.raises(TypeError):
            call(3)
    for name, tol in (('HIGHS-DS', 1e-9), ('clarabel', 1e-6)):
        _, fval, status, _ = solve_lp(f, _PA, _PB, lb=_LB, ub=_UB,
                                      lp_solver=name)
        assert status == 'optimal', (name, status)
        assert fval == pytest.approx(-1.25, abs=tol), (name, fval)


@pytest.mark.skipif(not _HAS_HIGHSPY, reason="highspy not installed")
def test_new_highs_models_carry_the_documented_options():
    """``_new_highs`` builds every highspy model of the module: silent, with
    the small-matrix threshold lowered from HiGHS's 1e-9 to 1e-12 (entries
    above it reach the solver) and, when asked, presolve off (the careful
    re-solve and the feasibility check behind ``confirm_infeasible`` rely on
    it: presolve was measured to call tiny-scale feasible systems
    infeasible). Read back from the model, exact. Catches the threshold not
    set and the presolve flag ignored or inverted (the latter changes no
    certified answer, only which of two solves decides)."""
    from n2v.utils.lpsolver import _new_highs
    for presolve, expected in ((True, 'choose'), (False, 'off')):
        h = _new_highs(presolve=presolve)
        assert h.getOptionValue('small_matrix_value')[1] == 1e-12, presolve
        assert h.getOptionValue('presolve')[1] == expected, presolve
        assert h.getOptionValue('output_flag')[1] is False, presolve
    assert _new_highs().getOptionValue('presolve')[1] == 'choose'
