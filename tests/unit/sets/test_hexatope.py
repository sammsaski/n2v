"""Tests for set representations."""

import pytest
import numpy as np
from n2v.sets import Star, Zono, Box, HalfSpace, Hexatope, Octatope

class TestHexatope:
    """Tests for Hexatope set."""

    def test_creation(self, simple_hexatope):
        """Test Hexatope creation."""
        assert simple_hexatope.dim == 3
        # V1 soundness fix: Hexatopes include anchor variable, so nVar = dim + 1
        assert simple_hexatope.nVar == 4  # 1 anchor + 3 dimensions
        pytest.assert_hexatope_valid(simple_hexatope)

    def test_from_bounds(self):
        """Test Hexatope creation from bounds."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        assert hexatope.dim == 3
        # V1 soundness fix: nVar = dim + 1 (includes anchor variable)
        assert hexatope.nVar == 4
        pytest.assert_hexatope_valid(hexatope)

        # Check bounds are preserved
        assert hexatope.state_lb is not None
        assert hexatope.state_ub is not None
        np.testing.assert_allclose(hexatope.state_lb, lb, atol=1e-6)
        np.testing.assert_allclose(hexatope.state_ub, ub, atol=1e-6)

    def test_affine_map(self, simple_hexatope):
        """Test affine transformation."""
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0]])
        b = np.array([[0.5], [0.5]])

        result = simple_hexatope.affine_map(W, b)

        assert result.dim == 2
        assert result.nVar == simple_hexatope.nVar
        pytest.assert_hexatope_valid(result)

    def test_estimate_ranges(self, simple_hexatope):
        """Test range estimation."""
        lb, ub = simple_hexatope.estimate_ranges()

        assert lb.shape == (simple_hexatope.dim, 1)
        assert ub.shape == (simple_hexatope.dim, 1)
        assert np.all(lb <= ub)

        # Check that state bounds are updated
        assert simple_hexatope.state_lb is not None
        assert simple_hexatope.state_ub is not None

    def test_get_bounds(self, simple_hexatope):
        """Test bounds computation."""
        lb, ub = simple_hexatope.get_bounds()

        assert lb.shape == (simple_hexatope.dim, 1)
        assert ub.shape == (simple_hexatope.dim, 1)
        assert np.all(lb <= ub)

    def test_identity_transformation(self):
        """Test identity transformation preserves bounds."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply identity transformation
        W = np.eye(2)
        b = np.zeros((2, 1))
        result = hexatope.affine_map(W, b)

        result_lb, result_ub = result.estimate_ranges()
        np.testing.assert_allclose(result_lb, lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, ub, atol=1e-6)

    def test_translation(self):
        """Test pure translation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Translate by [2, 3]
        W = np.eye(2)
        b = np.array([[2.0], [3.0]])
        result = hexatope.affine_map(W, b)

        result_lb, result_ub = result.estimate_ranges()
        expected_lb = np.array([[2.0], [3.0]])
        expected_ub = np.array([[3.0], [4.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_scaling(self):
        """Test scaling transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Scale by 2
        W = np.eye(2) * 2
        result = hexatope.affine_map(W)

        result_lb, result_ub = result.estimate_ranges()
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.0], [2.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_dimension_reduction(self):
        """Test dimension reduction."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Project to 2D and sum third dimension
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0]])
        result = hexatope.affine_map(W)

        assert result.dim == 2
        result_lb, result_ub = result.estimate_ranges()

        # First dimension: [0, 1]
        # Second dimension: [0, 2] (sum of two [0, 1] ranges)
        assert result_lb[0] <= 0.0 + 1e-6
        assert result_ub[0] >= 1.0 - 1e-6
        assert result_lb[1] <= 0.0 + 1e-6
        assert result_ub[1] >= 2.0 - 1e-6

    def test_is_empty_set(self, simple_hexatope):
        """Test emptiness checking."""
        # Simple hexatope from bounds should not be empty
        assert not simple_hexatope.is_empty_set()

    def test_contains_point_inside(self):
        """Test point containment for point inside."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        point_in = np.array([[0.5], [0.5]])
        assert hexatope.contains(point_in)

    def test_hexatope_to_box_conversion(self, simple_hexatope):
        """Test conversion to Box."""
        box = simple_hexatope.get_box(use_mcf=False)

        assert box.dim == simple_hexatope.dim
        assert np.all(box.lb <= box.ub)


    # Exact reachability tests for Hexatope
    def test_exact_simple_box_2d(self):
        """Test exact bounds for simple 2D box."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        lb_computed, ub_computed = hexatope.get_bounds()

        assert np.allclose(lb_computed, lb, atol=1e-6)
        assert np.allclose(ub_computed, ub, atol=1e-6)

    def test_exact_affine_transformed(self):
        """Test exact bounds after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply scaling: y = 2x
        W = np.eye(2) * 2
        hexatope_transformed = hexatope.affine_map(W)

        lb_computed, ub_computed = hexatope_transformed.get_bounds()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.0], [2.0]])

        assert np.allclose(lb_computed, expected_lb, atol=1e-6)
        assert np.allclose(ub_computed, expected_ub, atol=1e-6)

    def test_exact_dimension_reduction(self):
        """Test exact bounds after dimension reduction."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Project to 2D: y = [x_0, x_1 + x_2]
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0]])
        hexatope_projected = hexatope.affine_map(W)

        lb_computed, ub_computed = hexatope_projected.get_bounds()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1.0], [2.0]])

        assert np.allclose(lb_computed, expected_lb, atol=1e-6)
        assert np.allclose(ub_computed, expected_ub, atol=1e-6)

    def test_exact_vs_estimate(self):
        """Verify exact bounds are tighter or equal to estimates."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply complex transformation
        W = np.array([[1.0, 0.5, 0.0],
                      [0.0, 1.0, 0.5]])
        hexatope_transformed = hexatope.affine_map(W)

        lb_exact, ub_exact = hexatope_transformed.get_bounds()
        lb_estimate, ub_estimate = hexatope_transformed.estimate_ranges()

        # Exact should be contained in estimate
        assert np.all(lb_exact >= lb_estimate - 1e-6)
        assert np.all(ub_exact <= ub_estimate + 1e-6)

    # ========================================================================
    # Additional DCS Tests
    # ========================================================================

    def test_dcs_creation_basic(self):
        """Test DifferenceConstraintSystem creation."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        dcs = DifferenceConstraintSystem(num_vars=3)
        assert dcs.num_vars == 3
        assert len(dcs.constraints) == 0

    def test_dcs_add_constraint(self):
        """Test adding constraints to DCS."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        dcs = DifferenceConstraintSystem(num_vars=3)
        dcs.add_constraint(0, 1, 5.0)  # x0 - x1 <= 5
        dcs.add_constraint(1, 2, 3.0)  # x1 - x2 <= 3

        assert len(dcs.constraints) == 2
        assert dcs.constraints[0].i == 0
        assert dcs.constraints[0].j == 1
        assert dcs.constraints[0].b == 5.0

    def test_dcs_add_constraint_invalid_indices(self):
        """Test that invalid indices raise error."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        dcs = DifferenceConstraintSystem(num_vars=3)

        with pytest.raises(ValueError, match="Invalid variable indices"):
            dcs.add_constraint(5, 1, 1.0)  # i out of range

        with pytest.raises(ValueError, match="Invalid variable indices"):
            dcs.add_constraint(0, -1, 1.0)  # j negative

    def test_dcs_to_matrix_form(self):
        """Test conversion of DCS to matrix form."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        dcs = DifferenceConstraintSystem(num_vars=2)
        dcs.add_constraint(0, 1, 3.0)  # x0 - x1 <= 3

        A, b = dcs.to_matrix_form()

        # Should have 1 constraint: [1, -1] x <= 3
        assert A.shape == (1, 2)
        assert b.shape == (1,)
        np.testing.assert_array_equal(A[0], [1, -1])
        assert b[0] == 3.0

    def test_dcs_is_feasible_true(self):
        """Test feasibility check for feasible DCS."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        dcs = DifferenceConstraintSystem(num_vars=3)
        dcs.add_constraint(0, 1, 5.0)  # x0 - x1 <= 5
        dcs.add_constraint(1, 2, 3.0)  # x1 - x2 <= 3
        dcs.add_constraint(2, 0, 10.0)  # x2 - x0 <= 10

        # This should be feasible (no negative cycle)
        assert dcs.is_feasible()

    def test_dcs_is_feasible_false(self):
        """Test feasibility check for infeasible DCS (negative cycle)."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        dcs = DifferenceConstraintSystem(num_vars=3)
        dcs.add_constraint(0, 1, 1.0)   # x0 - x1 <= 1
        dcs.add_constraint(1, 2, 1.0)   # x1 - x2 <= 1
        dcs.add_constraint(2, 0, -3.0)  # x2 - x0 <= -3

        # Sum around cycle: 1 + 1 + (-3) = -1 < 0 → negative cycle
        assert not dcs.is_feasible()

    def test_dcs_copy(self):
        """Test DCS deep copy."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        dcs = DifferenceConstraintSystem(num_vars=2)
        dcs.add_constraint(0, 1, 5.0)

        dcs_copy = dcs.copy()

        assert dcs_copy.num_vars == dcs.num_vars
        assert len(dcs_copy.constraints) == len(dcs.constraints)
        assert dcs_copy.constraints[0].i == dcs.constraints[0].i

        # Modify copy shouldn't affect original
        dcs_copy.add_constraint(1, 0, 2.0)
        assert len(dcs.constraints) == 1
        assert len(dcs_copy.constraints) == 2

    # ========================================================================
    # MCF vs LP Solver Comparison Tests
    # ========================================================================

    # @pytest.mark.skip(reason="get_range with MCF may return None - implementation issue")
    def test_get_range_mcf_vs_lp(self):
        """Test that MCF and LP solvers give same results."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Get range using both methods
        lb_mcf, ub_mcf = hexatope.get_range(0, use_mcf=True)
        lb_lp, ub_lp = hexatope.get_range(0, use_mcf=False)

        # Check both returned valid results
        assert lb_mcf is not None and ub_mcf is not None
        assert lb_lp is not None and ub_lp is not None

        # Results should be very close
        np.testing.assert_allclose(lb_mcf, lb_lp, atol=1e-5)
        np.testing.assert_allclose(ub_mcf, ub_lp, atol=1e-5)

    def test_get_bounds_mcf_vs_lp(self):
        """Test that MCF and LP solvers give same bounds."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        bounds_mcf = hexatope.get_bounds(use_mcf=True)
        bounds_lp = hexatope.get_bounds(use_mcf=False)

        np.testing.assert_allclose(bounds_mcf[0], bounds_lp[0], atol=1e-5)
        np.testing.assert_allclose(bounds_mcf[1], bounds_lp[1], atol=1e-5)

    # @pytest.mark.skip(reason="optimize_linear with MCF may return None - implementation issue")
    def test_optimize_linear_mcf_vs_lp(self):
        """Test that MCF and LP give same optimization results."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Maximize x0 + x1
        objective = np.array([1.0, 1.0])

        result_mcf = hexatope.optimize_linear(objective, maximize=True, use_mcf=True)
        result_lp = hexatope.optimize_linear(objective, maximize=True, use_mcf=False)

        # Check both returned valid results
        assert result_mcf is not None
        assert result_lp is not None

        # Both should find optimal value ≈ 2.0
        np.testing.assert_allclose(result_mcf, result_lp, atol=1e-5)

    # ========================================================================
    # Edge Cases and Error Handling
    # ========================================================================

    def test_from_bounds_1d(self):
        """Test creation from 1D bounds."""
        lb = np.array([[2.0]])
        ub = np.array([[5.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        assert hexatope.dim == 1
        # V1 soundness fix: nVar = dim + 1 (includes anchor)
        assert hexatope.nVar == 2  # 1 anchor + 1 dimension

        computed_lb, computed_ub = hexatope.get_bounds()
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_from_bounds_mismatched_dimensions(self):
        """Test that mismatched dimensions raise error."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])  # Wrong dimension

        with pytest.raises(ValueError):
            Hexatope.from_bounds(lb, ub)

    def test_affine_map_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        hexatope = Hexatope.from_bounds(np.array([[0.0], [0.0]]),
                                        np.array([[1.0], [1.0]]))

        W = np.eye(3)  # Wrong dimension

        with pytest.raises(ValueError):
            hexatope.affine_map(W)

    def test_contains_point_outside(self):
        """Test point containment for point outside."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        point_out = np.array([[5.0], [5.0]])
        assert not hexatope.contains(point_out)

    def test_contains_point_on_boundary(self):
        """Test point containment for point on boundary."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        point_boundary = np.array([[1.0], [1.0]])
        assert hexatope.contains(point_boundary)

    def test_get_range_invalid_index(self):
        """Test that invalid index raises error."""
        hexatope = Hexatope.from_bounds(np.array([[0.0], [0.0]]),
                                        np.array([[1.0], [1.0]]))

        with pytest.raises((IndexError, ValueError)):
            hexatope.get_range(5)

    def test_intersect_half_space_basic(self):
        """Test half-space intersection."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Intersect with constraint on generators (2 generators for 2D box)
        # The constraint should be on the generator space, not state space
        H = np.array([[1.0, 0.0]])  # Constraint on first generator
        g = np.array([[0.5]])

        result = hexatope.intersect_half_space(H, g)

        # Result should be a valid hexatope
        pytest.assert_hexatope_valid(result)

        # Bounds should be constrained
        result_lb, result_ub = result.get_bounds()
        assert result_ub[0] <= 0.5 + 1e-5

    def test_to_star_conversion(self):
        """Test conversion to Star set."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # Star should represent similar region
        star_lb, star_ub = star.get_ranges()
        hex_lb, hex_ub = hexatope.get_bounds()

        # Bounds should be close (Star may be looser)
        np.testing.assert_allclose(star_lb, hex_lb, atol=1e-3)
        np.testing.assert_allclose(star_ub, hex_ub, atol=1e-3)

    # ========================================================================
    # Numerical Stability Tests
    # ========================================================================

    def test_large_bounds(self):
        """Test with large bound values."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1e6], [1e6]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Should handle large values
        computed_lb, computed_ub = hexatope.get_bounds()
        assert np.all(computed_lb >= lb - 1e-3)
        assert np.all(computed_ub <= ub + 1e-3)

    def test_small_bounds(self):
        """Test with very small bound ranges."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1e-6], [1e-6]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Should handle small values
        computed_lb, computed_ub = hexatope.get_bounds()
        assert np.all(computed_lb >= -1e-5)
        assert np.all(computed_ub <= ub + 1e-5)

    def test_negative_bounds(self):
        """Test with negative bounds."""
        lb = np.array([[-10.0], [-5.0]])
        ub = np.array([[-1.0], [0.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        computed_lb, computed_ub = hexatope.get_bounds()
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)


