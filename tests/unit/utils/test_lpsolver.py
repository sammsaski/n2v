"""Tests for LP solver backends in n2v.utils.lpsolver."""

import warnings

import numpy as np
import pytest

from n2v.utils.lpsolver import (
    _HAS_HIGHSPY,
    solve_lp,
    solve_lp_batch,
)


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
        """Direct highspy gives same results as scipy for a constrained LP."""
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
