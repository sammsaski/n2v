"""Tests for set representations."""

import pytest
import numpy as np
from n2v.sets import Star, Zono, Box, HalfSpace, Hexatope, Octatope

class TestOctatope:
    """Tests for Octatope set."""

    def test_creation(self, simple_octatope):
        """Test Octatope creation."""
        assert simple_octatope.dim == 3
        assert simple_octatope.nVar == 3
        pytest.assert_octatope_valid(simple_octatope)

    def test_from_bounds(self):
        """Test Octatope creation from bounds."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        assert octatope.dim == 3
        assert octatope.nVar == 3
        pytest.assert_octatope_valid(octatope)

        # Check bounds are preserved
        assert octatope.state_lb is not None
        assert octatope.state_ub is not None
        np.testing.assert_allclose(octatope.state_lb, lb, atol=1e-6)
        np.testing.assert_allclose(octatope.state_ub, ub, atol=1e-6)

    def test_affine_map(self, simple_octatope):
        """Test affine transformation."""
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0]])
        b = np.array([[0.5], [0.5]])

        result = simple_octatope.affine_map(W, b)

        assert result.dim == 2
        assert result.nVar == simple_octatope.nVar
        pytest.assert_octatope_valid(result)

    def test_estimate_ranges(self, simple_octatope):
        """Test range estimation."""
        lb, ub = simple_octatope.estimate_ranges()

        assert lb.shape == (simple_octatope.dim, 1)
        assert ub.shape == (simple_octatope.dim, 1)
        assert np.all(lb <= ub)

        # Check that state bounds are updated
        assert simple_octatope.state_lb is not None
        assert simple_octatope.state_ub is not None

    def test_get_bounds(self, simple_octatope):
        """Test bounds computation."""
        lb, ub = simple_octatope.get_bounds()

        assert lb.shape == (simple_octatope.dim, 1)
        assert ub.shape == (simple_octatope.dim, 1)
        assert np.all(lb <= ub)

    def test_identity_transformation(self):
        """Test identity transformation preserves bounds."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Apply identity transformation
        W = np.eye(2)
        b = np.zeros((2, 1))
        result = octatope.affine_map(W, b)

        result_lb, result_ub = result.estimate_ranges()
        np.testing.assert_allclose(result_lb, lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, ub, atol=1e-6)

    def test_translation(self):
        """Test pure translation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Translate by [2, 3]
        W = np.eye(2)
        b = np.array([[2.0], [3.0]])
        result = octatope.affine_map(W, b)

        result_lb, result_ub = result.estimate_ranges()
        expected_lb = np.array([[2.0], [3.0]])
        expected_ub = np.array([[3.0], [4.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_scaling(self):
        """Test scaling transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Scale by 2
        W = np.eye(2) * 2
        result = octatope.affine_map(W)

        result_lb, result_ub = result.estimate_ranges()
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.0], [2.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_dimension_reduction(self):
        """Test dimension reduction."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Project to 2D and sum third dimension
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0]])
        result = octatope.affine_map(W)

        assert result.dim == 2
        result_lb, result_ub = result.estimate_ranges()

        # First dimension: [0, 1]
        # Second dimension: [0, 2] (sum of two [0, 1] ranges)
        assert result_lb[0] <= 0.0 + 1e-6
        assert result_ub[0] >= 1.0 - 1e-6
        assert result_lb[1] <= 0.0 + 1e-6
        assert result_ub[1] >= 2.0 - 1e-6

    def test_is_empty_set(self, simple_octatope):
        """Test emptiness checking."""
        # Simple octatope from bounds should not be empty
        assert not simple_octatope.is_empty_set()

    def test_contains_point_inside(self):
        """Test point containment for point inside."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        point_in = np.array([[0.5], [0.5]])
        assert octatope.contains(point_in)

    def test_octatope_to_box_conversion(self, simple_octatope):
        """Test conversion to Box."""
        box = simple_octatope.get_box(use_mcf=False)

        assert box.dim == simple_octatope.dim
        assert np.all(box.lb <= box.ub)

    def test_utvpi_feasibility(self):
        """Test UTVPI constraint system feasibility."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Should be feasible
        assert octatope.utvpi.is_feasible()
        assert not octatope.is_empty_set()

    """Exact reachability tests for Octatope."""

    def test_exact_simple_box_2d(self):
        """Test exact bounds for simple 2D box."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        octatope = Octatope.from_bounds(lb, ub)

        lb_computed, ub_computed = octatope.get_bounds()

        assert np.allclose(lb_computed, lb, atol=1e-6)
        assert np.allclose(ub_computed, ub, atol=1e-6)

    def test_exact_affine_transformed(self):
        """Test exact bounds after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Apply scaling: y = 2x
        W = np.eye(2) * 2
        octatope_transformed = octatope.affine_map(W)

        lb_computed, ub_computed = octatope_transformed.get_bounds()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.0], [2.0]])

        assert np.allclose(lb_computed, expected_lb, atol=1e-6)
        assert np.allclose(ub_computed, expected_ub, atol=1e-6)

    def test_exact_dimension_reduction(self):
        """Test exact bounds after dimension reduction."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Project to 2D: y = [x_0, x_1 + x_2]
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0]])
        octatope_projected = octatope.affine_map(W)

        lb_computed, ub_computed = octatope_projected.get_bounds()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1.0], [2.0]])

        assert np.allclose(lb_computed, expected_lb, atol=1e-6)
        assert np.allclose(ub_computed, expected_ub, atol=1e-6)

    def test_exact_vs_estimate(self):
        """Verify exact bounds are tighter or equal to estimates."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Apply complex transformation
        W = np.array([[1.0, 0.5, 0.0],
                      [0.0, 1.0, 0.5]])
        octatope_transformed = octatope.affine_map(W)

        lb_exact, ub_exact = octatope_transformed.get_bounds()
        lb_estimate, ub_estimate = octatope_transformed.estimate_ranges()

        # Exact should be contained in estimate
        assert np.all(lb_exact >= lb_estimate - 1e-6)
        assert np.all(ub_exact <= ub_estimate + 1e-6)

    def test_hexatope_octatope_consistency(self):
        """Hexatope and Octatope should give same results for boxes."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])

        hexatope = Hexatope.from_bounds(lb, ub)
        octatope = Octatope.from_bounds(lb, ub)

        lb_hex, ub_hex = hexatope.get_bounds()
        lb_oct, ub_oct = octatope.get_bounds()

        assert np.allclose(lb_hex, lb_oct, atol=1e-6)
        assert np.allclose(ub_hex, ub_oct, atol=1e-6)

    # ========================================================================
    # Additional UTVPI Tests
    # ========================================================================

    def test_utvpi_creation_basic(self):
        """Test UTVPIConstraintSystem creation."""
        from n2v.sets.octatope import UTVPIConstraintSystem

        utvpi = UTVPIConstraintSystem(num_vars=3)
        assert utvpi.num_vars == 3
        assert len(utvpi.constraints) == 0

    def test_utvpi_add_constraint(self):
        """Test adding UTVPI constraints."""
        from n2v.sets.octatope import UTVPIConstraintSystem

        utvpi = UTVPIConstraintSystem(num_vars=3)
        utvpi.add_constraint(0, 1, 1, 1, 5.0)  # x0 + x1 <= 5
        utvpi.add_constraint(1, 2, 1, -1, 3.0)  # x1 - x2 <= 3

        assert len(utvpi.constraints) == 2
        assert utvpi.constraints[0].i == 0
        assert utvpi.constraints[0].j == 1
        assert utvpi.constraints[0].ai == 1
        assert utvpi.constraints[0].aj == 1
        assert utvpi.constraints[0].b == 5.0

    def test_utvpi_constraint_validation(self):
        """Test UTVPI coefficient validation."""
        from n2v.sets.octatope import UTVPIConstraint

        # Valid coefficients
        c1 = UTVPIConstraint(0, 1, 1, -1, 5.0)
        assert c1.ai == 1 and c1.aj == -1

        # Invalid coefficient (not in {-1, 0, 1})
        with pytest.raises(ValueError, match="UTVPI coefficients"):
            UTVPIConstraint(0, 1, 2, 1, 5.0)

        # Both coefficients zero
        with pytest.raises(ValueError, match="At least one coefficient"):
            UTVPIConstraint(0, 1, 0, 0, 5.0)

    def test_utvpi_add_constraint_invalid_indices(self):
        """Test that invalid indices raise error."""
        from n2v.sets.octatope import UTVPIConstraintSystem

        utvpi = UTVPIConstraintSystem(num_vars=3)

        with pytest.raises(ValueError, match="Invalid variable indices"):
            utvpi.add_constraint(5, 1, 1, 1, 1.0)  # i out of range

        with pytest.raises(ValueError, match="Invalid variable indices"):
            utvpi.add_constraint(0, -1, 1, 1, 1.0)  # j negative

    def test_utvpi_to_matrix_form(self):
        """Test conversion of UTVPI to matrix form."""
        from n2v.sets.octatope import UTVPIConstraintSystem

        utvpi = UTVPIConstraintSystem(num_vars=2)
        utvpi.add_constraint(0, 1, 1, -1, 3.0)  # x0 - x1 <= 3

        A, b = utvpi.to_matrix_form()

        # Should have 1 constraint: [1, -1] x <= 3
        assert A.shape == (1, 2)
        assert b.shape == (1,)
        np.testing.assert_array_equal(A[0], [1, -1])
        assert b[0] == 3.0

    def test_utvpi_to_dcs_conversion(self):
        """Test UTVPI to DCS conversion."""
        from n2v.sets.octatope import UTVPIConstraintSystem

        utvpi = UTVPIConstraintSystem(num_vars=2)
        utvpi.add_constraint(0, 1, 1, -1, 3.0)  # x0 - x1 <= 3

        dcs = utvpi.to_dcs()

        # DCS should have 2*2 = 4 variables (x+_0, x-_0, x+_1, x-_1)
        assert dcs.num_vars == 4
        # Should have constraints from UTVPI conversion
        assert len(dcs.constraints) > 0

    def test_utvpi_is_feasible_true(self):
        """Test feasibility check for feasible UTVPI system."""
        from n2v.sets.octatope import UTVPIConstraintSystem

        utvpi = UTVPIConstraintSystem(num_vars=3)
        utvpi.add_constraint(0, 1, 1, 1, 5.0)   # x0 + x1 <= 5
        utvpi.add_constraint(1, 2, 1, -1, 3.0)  # x1 - x2 <= 3
        utvpi.add_constraint(2, 0, -1, -1, 10.0)  # -x2 - x0 <= 10

        # Should be feasible
        assert utvpi.is_feasible()

    def test_utvpi_is_feasible_false(self):
        """Test feasibility check for infeasible UTVPI system."""
        from n2v.sets.octatope import UTVPIConstraintSystem

        utvpi = UTVPIConstraintSystem(num_vars=2)
        utvpi.add_constraint(0, 1, 1, 0, 1.0)   # x0 <= 1
        utvpi.add_constraint(0, 1, -1, 0, -3.0)  # -x0 <= -3 (i.e., x0 >= 3)

        # Infeasible: x0 <= 1 and x0 >= 3
        assert not utvpi.is_feasible()

    def test_utvpi_copy(self):
        """Test UTVPI deep copy."""
        from n2v.sets.octatope import UTVPIConstraintSystem

        utvpi = UTVPIConstraintSystem(num_vars=2)
        utvpi.add_constraint(0, 1, 1, -1, 5.0)

        utvpi_copy = utvpi.copy()

        assert utvpi_copy.num_vars == utvpi.num_vars
        assert len(utvpi_copy.constraints) == len(utvpi.constraints)
        assert utvpi_copy.constraints[0].i == utvpi.constraints[0].i

        # Modify copy shouldn't affect original
        utvpi_copy.add_constraint(1, 0, 1, 1, 2.0)
        assert len(utvpi.constraints) == 1
        assert len(utvpi_copy.constraints) == 2

    # ========================================================================
    # Edge Cases and Error Handling
    # ========================================================================

    def test_from_bounds_1d(self):
        """Test creation from 1D bounds."""
        lb = np.array([[2.0]])
        ub = np.array([[5.0]])
        octatope = Octatope.from_bounds(lb, ub)

        assert octatope.dim == 1
        assert octatope.nVar == 1

        computed_lb, computed_ub = octatope.get_bounds()
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_from_bounds_mismatched_dimensions(self):
        """Test that mismatched dimensions raise error."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])  # Wrong dimension

        with pytest.raises(ValueError):
            Octatope.from_bounds(lb, ub)

    def test_affine_map_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        octatope = Octatope.from_bounds(np.array([[0.0], [0.0]]),
                                        np.array([[1.0], [1.0]]))

        W = np.eye(3)  # Wrong dimension

        with pytest.raises(ValueError):
            octatope.affine_map(W)

    def test_contains_point_outside(self):
        """Test point containment for point outside."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        point_out = np.array([[5.0], [5.0]])
        assert not octatope.contains(point_out)

    def test_contains_point_on_boundary(self):
        """Test point containment for point on boundary."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        point_boundary = np.array([[1.0], [1.0]])
        assert octatope.contains(point_boundary)

    def test_to_star_conversion(self):
        """Test conversion to Star set."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # Star should represent similar region
        star_lb, star_ub = star.get_ranges()
        oct_lb, oct_ub = octatope.get_bounds()

        # Bounds should be close (Star may be looser)
        np.testing.assert_allclose(star_lb, oct_lb, atol=1e-3)
        np.testing.assert_allclose(star_ub, oct_ub, atol=1e-3)

    # ========================================================================
    # Numerical Stability Tests
    # ========================================================================

    def test_large_bounds(self):
        """Test with large bound values."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1e6], [1e6]])
        octatope = Octatope.from_bounds(lb, ub)

        # Should handle large values
        computed_lb, computed_ub = octatope.get_bounds()
        assert np.all(computed_lb >= lb - 1e-3)
        assert np.all(computed_ub <= ub + 1e-3)

    def test_small_bounds(self):
        """Test with very small bound ranges."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1e-6], [1e-6]])
        octatope = Octatope.from_bounds(lb, ub)

        # Should handle small values
        computed_lb, computed_ub = octatope.get_bounds()
        assert np.all(computed_lb >= -1e-5)
        assert np.all(computed_ub <= ub + 1e-5)

    def test_negative_bounds(self):
        """Test with negative bounds."""
        lb = np.array([[-10.0], [-5.0]])
        ub = np.array([[-1.0], [0.0]])
        octatope = Octatope.from_bounds(lb, ub)

        computed_lb, computed_ub = octatope.get_bounds()
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_get_bounds_mcf_vs_lp(self):
        """Test that MCF and LP solvers give same bounds."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        octatope = Octatope.from_bounds(lb, ub)

        bounds_mcf = octatope.get_bounds(use_mcf=True)
        bounds_lp = octatope.get_bounds(use_mcf=False)

        np.testing.assert_allclose(bounds_mcf[0], bounds_lp[0], atol=1e-5)
        np.testing.assert_allclose(bounds_mcf[1], bounds_lp[1], atol=1e-5)


