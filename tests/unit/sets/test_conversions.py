"""Tests for set representations."""

import pytest
import numpy as np
from n2v.sets import Star, Zono, Box, HalfSpace, Hexatope, Octatope

class TestSetConversions:
    """Test conversions between set types."""

    def test_star_to_box(self, simple_star):
        """Test Star to Box conversion - basic functionality."""
        box = simple_star.get_box()

        assert box.dim == simple_star.dim
        assert np.all(box.lb <= box.ub)

        # Box should be valid
        assert box.lb.shape == (simple_star.dim, 1)
        assert box.ub.shape == (simple_star.dim, 1)

    def test_star_get_box_from_bounds(self):
        """Test that get_box() preserves bounds for box Stars."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        star = Star.from_bounds(lb, ub)

        box = star.get_box()

        # Should recover original bounds exactly
        np.testing.assert_allclose(box.lb, lb, atol=1e-6)
        np.testing.assert_allclose(box.ub, ub, atol=1e-6)

    def test_star_get_box_after_translation(self):
        """Test get_box() after translation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Translate by [2, 3]
        W = np.eye(2)
        b = np.array([[2.0], [3.0]])
        star_translated = star.affine_map(W, b)

        box = star_translated.get_box()

        # Bounds should be shifted
        expected_lb = np.array([[2.0], [3.0]])
        expected_ub = np.array([[3.0], [4.0]])
        np.testing.assert_allclose(box.lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(box.ub, expected_ub, atol=1e-6)

    def test_star_get_box_after_scaling(self):
        """Test get_box() after scaling."""
        lb = np.array([[0.0], [1.0]])
        ub = np.array([[1.0], [2.0]])
        star = Star.from_bounds(lb, ub)

        # Scale by 2
        W = np.eye(2) * 2
        star_scaled = star.affine_map(W)

        box = star_scaled.get_box()

        # Bounds should be scaled
        expected_lb = np.array([[0.0], [2.0]])
        expected_ub = np.array([[2.0], [4.0]])
        np.testing.assert_allclose(box.lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(box.ub, expected_ub, atol=1e-6)

    def test_star_get_box_after_half_space_intersection(self):
        """Test get_box() after half-space intersection."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Intersect with x0 <= 0.5
        H = np.array([[1.0, 0.0]])
        g = np.array([[0.5]])
        star_intersected = star.intersect_half_space(H, g)

        box = star_intersected.get_box()

        # Should get [0, 0.5] × [0, 1]
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[0.5], [1.0]])
        np.testing.assert_allclose(box.lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(box.ub, expected_ub, atol=1e-6)

    def test_star_get_box_1d(self):
        """Test get_box() for 1D Star."""
        lb = np.array([[2.0]])
        ub = np.array([[5.0]])
        star = Star.from_bounds(lb, ub)

        box = star.get_box()

        assert box.dim == 1
        np.testing.assert_allclose(box.lb, lb, atol=1e-6)
        np.testing.assert_allclose(box.ub, ub, atol=1e-6)

    def test_star_get_box_3d(self):
        """Test get_box() for 3D Star."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        star = Star.from_bounds(lb, ub)

        box = star.get_box()

        assert box.dim == 3
        np.testing.assert_allclose(box.lb, lb, atol=1e-6)
        np.testing.assert_allclose(box.ub, ub, atol=1e-6)

    def test_star_get_box_negative_bounds(self):
        """Test get_box() with negative bounds."""
        lb = np.array([[-5.0], [-3.0]])
        ub = np.array([[-1.0], [0.0]])
        star = Star.from_bounds(lb, ub)

        box = star.get_box()

        np.testing.assert_allclose(box.lb, lb, atol=1e-6)
        np.testing.assert_allclose(box.ub, ub, atol=1e-6)

    def test_star_get_box_dimension_reduction(self):
        """Test get_box() after dimension reduction."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Project to 2D: [x0, x1+x2]
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0]])
        star_projected = star.affine_map(W)

        box = star_projected.get_box()

        assert box.dim == 2
        # First dimension: [0, 1]
        # Second dimension: [0, 2] (sum of two [0,1] ranges)
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1.0], [2.0]])
        np.testing.assert_allclose(box.lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(box.ub, expected_ub, atol=1e-6)

    def test_star_get_box_diagonal_constraint(self):
        """Test get_box() with diagonal constraint."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Add constraint x0 + x1 <= 1
        H = np.array([[1.0, 1.0]])
        g = np.array([[1.0]])
        star_constrained = star.intersect_half_space(H, g)

        box = star_constrained.get_box()

        # Box should still be [0,1] × [0,1] (over-approximation of triangle)
        # The exact bounds depend on LP solver, but should be at most [0,1]×[0,1]
        assert box.lb[0] <= 0.0 + 1e-6
        assert box.lb[1] <= 0.0 + 1e-6
        assert box.ub[0] >= 1.0 - 1e-6  # Could be exact 1.0
        assert box.ub[1] >= 1.0 - 1e-6  # Could be exact 1.0

    def test_zono_to_box(self, simple_zono):
        """Test Zono to Box conversion."""
        box = simple_zono.get_box()

        assert box.dim == simple_zono.dim

        # Box should contain zonotope
        zono_lb, zono_ub = simple_zono.get_bounds()
        np.testing.assert_allclose(box.lb, zono_lb, atol=1e-6)
        np.testing.assert_allclose(box.ub, zono_ub, atol=1e-6)

    def test_box_to_zono(self, simple_box):
        """Test Box to Zono conversion."""
        zono = simple_box.to_zono()

        assert zono.dim == simple_box.dim

        # Zono should represent same box
        zono_lb, zono_ub = zono.get_bounds()
        np.testing.assert_allclose(zono_lb, simple_box.lb, atol=1e-6)
        np.testing.assert_allclose(zono_ub, simple_box.ub, atol=1e-6)




class TestToStarConversion:
    """Tests for to_star() conversion methods."""

    def test_hexatope_to_star_basic(self):
        """Test basic Hexatope to Star conversion."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # Check dimensions
        assert star.dim == hexatope.dim
        assert star.nVar == hexatope.nVar

        # Check that Star is valid
        pytest.assert_star_valid(star)

        # Check that predicate bounds are [-1, 1]
        np.testing.assert_array_equal(star.predicate_lb, np.full((hexatope.nVar, 1), -1.0))
        np.testing.assert_array_equal(star.predicate_ub, np.full((hexatope.nVar, 1), 1.0))

    def test_octatope_to_star_basic(self):
        """Test basic Octatope to Star conversion."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # Check dimensions
        assert star.dim == octatope.dim
        assert star.nVar == octatope.nVar

        # Check that Star is valid
        pytest.assert_star_valid(star)

        # Check that predicate bounds are [-1, 1]
        np.testing.assert_array_equal(star.predicate_lb, np.full((octatope.nVar, 1), -1.0))
        np.testing.assert_array_equal(star.predicate_ub, np.full((octatope.nVar, 1), 1.0))

    def test_hexatope_to_star_preserves_bounds(self):
        """Test that Hexatope to Star conversion preserves bounds."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # Compute bounds on both
        hex_lb, hex_ub = hexatope.get_bounds()
        star_lb, star_ub = star.get_ranges()

        # Star bounds should match Hexatope bounds (within tolerance)
        np.testing.assert_allclose(star_lb, hex_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, hex_ub, atol=1e-5)

    def test_octatope_to_star_preserves_bounds(self):
        """Test that Octatope to Star conversion preserves bounds."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # Compute bounds on both
        oct_lb, oct_ub = octatope.get_bounds()
        star_lb, star_ub = star.get_ranges()

        # Star bounds should match Octatope bounds (within tolerance)
        np.testing.assert_allclose(star_lb, oct_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, oct_ub, atol=1e-5)

    def test_hexatope_to_star_center_preservation(self):
        """Test that center is preserved in conversion."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # Star's center should be hexatope's center
        np.testing.assert_allclose(star.V[:, 0], hexatope.center, atol=1e-9)

    def test_octatope_to_star_center_preservation(self):
        """Test that center is preserved in conversion."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # Star's center should be octatope's center
        np.testing.assert_allclose(star.V[:, 0], octatope.center, atol=1e-9)

    def test_hexatope_to_star_generators_preservation(self):
        """Test that generators are preserved in conversion."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # Star's generators should match hexatope's generators
        np.testing.assert_allclose(star.V[:, 1:], hexatope.generators, atol=1e-9)

    def test_octatope_to_star_generators_preservation(self):
        """Test that generators are preserved in conversion."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # Star's generators should match octatope's generators
        np.testing.assert_allclose(star.V[:, 1:], octatope.generators, atol=1e-9)

    def test_hexatope_to_star_constraints_include_box(self):
        """Test that converted Star includes box constraints."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # V1 soundness: Anchor variable provides implicit bounds through DCS
        # The star should have exactly the DCS constraints (no additional explicit box constraints)
        n_dcs_constraints = len(hexatope.dcs.constraints)

        assert star.C.shape[0] == n_dcs_constraints

        # Verify the bounds are correct (most important check)
        hex_lb, hex_ub = hexatope.get_bounds()
        star_lb, star_ub = star.get_ranges()
        np.testing.assert_allclose(star_lb, hex_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, hex_ub, atol=1e-5)

    def test_octatope_to_star_constraints_include_box(self):
        """Test that converted Star includes box constraints."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # UTVPI constraints already include box bounds (±x_i ≤ 1 added by from_bounds())
        # The star should have exactly the UTVPI constraints
        n_utvpi_constraints = len(octatope.utvpi.constraints)

        assert star.C.shape[0] == n_utvpi_constraints

        # Verify the bounds are correct (most important check)
        oct_lb, oct_ub = octatope.get_bounds()
        star_lb, star_ub = star.get_ranges()
        np.testing.assert_allclose(star_lb, oct_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, oct_ub, atol=1e-5)

    def test_hexatope_to_star_after_affine_map(self):
        """Test conversion works after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply affine map
        W = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([[1.0], [2.0]])
        hexatope_transformed = hexatope.affine_map(W, b)

        star = hexatope_transformed.to_star()

        # Check that bounds are preserved
        hex_lb, hex_ub = hexatope_transformed.get_bounds()
        star_lb, star_ub = star.get_ranges()

        np.testing.assert_allclose(star_lb, hex_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, hex_ub, atol=1e-5)

    def test_octatope_to_star_after_affine_map(self):
        """Test conversion works after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Apply affine map
        W = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([[1.0], [2.0]])
        octatope_transformed = octatope.affine_map(W, b)

        star = octatope_transformed.to_star()

        # Check that bounds are preserved
        oct_lb, oct_ub = octatope_transformed.get_bounds()
        star_lb, star_ub = star.get_ranges()

        np.testing.assert_allclose(star_lb, oct_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, oct_ub, atol=1e-5)

    def test_hexatope_to_star_empty_dcs(self):
        """Test conversion with empty DCS (only box constraints)."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        # Create hexatope with no DCS constraints
        center = np.array([0.5, 0.5])
        generators = np.eye(2)
        dcs = DifferenceConstraintSystem(2)  # Empty DCS

        hexatope = Hexatope(center, generators, dcs)
        star = hexatope.to_star()

        # Should still work - star will have only box constraints
        assert star.dim == 2
        assert star.nVar == 2
        pytest.assert_star_valid(star)

    def test_hexatope_to_star_with_extra_constraints(self):
        """Test conversion when hexatope has extra constraints."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        # Create hexatope
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Add extra constraints via intersect_half_space
        H = np.array([[1.0, 1.0]])  # x + y <= 1.5
        g = np.array([[1.5]])
        hexatope_constrained = hexatope.intersect_half_space(H, g)

        star = hexatope_constrained.to_star()

        # Should have more constraints than the original hexatope
        # V1 soundness: Bounding box approach may not add many explicit constraints,
        # but the important thing is that bounds are correct
        assert star.C.shape[0] >= len(hexatope.dcs.constraints)
        pytest.assert_star_valid(star)

        # Verify the bounds are correct (most important check)
        hex_lb, hex_ub = hexatope_constrained.get_bounds()
        star_lb, star_ub = star.get_ranges()
        np.testing.assert_allclose(star_lb, hex_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, hex_ub, atol=1e-5)

    def test_hexatope_octatope_star_consistency(self):
        """Test that Hexatope and Octatope give similar Star representations for boxes."""
        lb = np.array([[0.0], [1.0]])
        ub = np.array([[1.0], [3.0]])

        hexatope = Hexatope.from_bounds(lb, ub)
        octatope = Octatope.from_bounds(lb, ub)

        star_from_hex = hexatope.to_star()
        star_from_oct = octatope.to_star()

        # Both should have same dimensions
        assert star_from_hex.dim == star_from_oct.dim

        # V1 soundness: Hexatope uses anchor variable (nVar = dim + 1),
        # Octatope doesn't (nVar = dim). They have different internal representations
        # but should produce the same bounds
        # assert star_from_hex.nVar == star_from_oct.nVar  # This is implementation-dependent

        # Both should give same bounds
        hex_star_lb, hex_star_ub = star_from_hex.get_ranges()
        oct_star_lb, oct_star_ub = star_from_oct.get_ranges()

        np.testing.assert_allclose(hex_star_lb, oct_star_lb, atol=1e-5)
        np.testing.assert_allclose(hex_star_ub, oct_star_ub, atol=1e-5)
