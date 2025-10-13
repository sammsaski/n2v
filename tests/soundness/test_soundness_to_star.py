"""
Soundness tests for to_star() conversion methods.

These tests verify that the conversion from Hexatope/Octatope to Star is sound:
- If a point is in the Hexatope/Octatope, it must also be in the resulting Star
- The Star set overapproximates the Hexatope/Octatope (bounds are preserved or expanded)
"""

import pytest
import numpy as np
from n2v.sets import Star, Hexatope, Octatope


class TestHexatopeToStarSoundness:
    """Soundness tests for Hexatope to Star conversion."""

    def test_soundness_point_containment_corners(self):
        """Test that corner points of Hexatope are in converted Star."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)
        star = hexatope.to_star()

        # Test all corner points
        corners = [
            np.array([[0.0], [0.0]]),
            np.array([[1.0], [0.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0], [1.0]]),
        ]

        for corner in corners:
            assert hexatope.contains(corner), f"Hexatope should contain corner {corner.T}"
            assert star.contains(corner), f"Star should contain corner {corner.T}"

    def test_soundness_point_containment_interior(self):
        """Test that interior points of Hexatope are in converted Star."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)
        star = hexatope.to_star()

        # Test interior points
        interior_points = [
            np.array([[0.5], [0.5]]),
            np.array([[0.25], [0.75]]),
            np.array([[0.75], [0.25]]),
            np.array([[0.1], [0.9]]),
        ]

        for point in interior_points:
            if hexatope.contains(point):
                assert star.contains(point), f"Star should contain point {point.T} that is in Hexatope"

    def test_soundness_bounds_overapproximate(self):
        """Test that Star bounds overapproximate Hexatope bounds."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        hexatope = Hexatope.from_bounds(lb, ub)
        star = hexatope.to_star()

        hex_lb, hex_ub = hexatope.get_bounds()
        star_lb, star_ub = star.get_ranges()

        # Star bounds should contain Hexatope bounds
        assert np.all(star_lb <= hex_lb + 1e-6), "Star lower bounds should be <= Hexatope lower bounds"
        assert np.all(star_ub >= hex_ub - 1e-6), "Star upper bounds should be >= Hexatope upper bounds"

    def test_soundness_after_affine_transformation(self):
        """Test soundness is preserved after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply affine transformation
        W = np.array([[2.0, 1.0], [0.0, 3.0]])
        b = np.array([[1.0], [2.0]])
        hexatope_transformed = hexatope.affine_map(W, b)

        star = hexatope_transformed.to_star()

        # Test that transformed corners are preserved
        corners = [
            np.array([[0.0], [0.0]]),
            np.array([[1.0], [0.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0], [1.0]]),
        ]

        for corner in corners:
            # Apply same transformation to corner
            transformed_corner = W @ corner + b

            if hexatope_transformed.contains(transformed_corner):
                assert star.contains(transformed_corner), \
                    f"Star should contain transformed corner {transformed_corner.T}"

    def test_soundness_random_points(self):
        """Test soundness with random points in the Hexatope."""
        np.random.seed(42)
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)
        star = hexatope.to_star()

        # Generate random points in the box
        n_samples = 50
        for _ in range(n_samples):
            point = np.random.uniform(lb.flatten(), ub.flatten()).reshape(-1, 1)

            # Only test points that are actually in the hexatope
            if hexatope.contains(point):
                assert star.contains(point), \
                    f"Star should contain point {point.T} that is in Hexatope"

    def test_soundness_with_extra_constraints(self):
        """Test soundness when Hexatope has extra constraints."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Add constraint: x + y <= 1.5
        H = np.array([[1.0, 1.0]])
        g = np.array([[1.5]])
        hexatope_constrained = hexatope.intersect_half_space(H, g)

        star = hexatope_constrained.to_star()

        # Test points that satisfy the constraint
        test_points = [
            np.array([[0.5], [0.5]]),  # sum = 1.0 < 1.5
            np.array([[0.7], [0.7]]),  # sum = 1.4 < 1.5
            np.array([[0.2], [0.8]]),  # sum = 1.0 < 1.5
        ]

        for point in test_points:
            if hexatope_constrained.contains(point):
                assert star.contains(point), \
                    f"Star should contain point {point.T} in constrained Hexatope"

    def test_soundness_center_point(self):
        """Test that the center point is always contained."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        hexatope = Hexatope.from_bounds(lb, ub)
        star = hexatope.to_star()

        center = hexatope.center.reshape(-1, 1)

        assert hexatope.contains(center), "Hexatope should contain its center"
        assert star.contains(center), "Star should contain Hexatope's center"


class TestOctatopeToStarSoundness:
    """Soundness tests for Octatope to Star conversion."""

    def test_soundness_point_containment_corners(self):
        """Test that corner points of Octatope are in converted Star."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)
        star = octatope.to_star()

        # Test all corner points
        corners = [
            np.array([[0.0], [0.0]]),
            np.array([[1.0], [0.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0], [1.0]]),
        ]

        for corner in corners:
            assert octatope.contains(corner), f"Octatope should contain corner {corner.T}"
            assert star.contains(corner), f"Star should contain corner {corner.T}"

    def test_soundness_point_containment_interior(self):
        """Test that interior points of Octatope are in converted Star."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)
        star = octatope.to_star()

        # Test interior points
        interior_points = [
            np.array([[0.5], [0.5]]),
            np.array([[0.25], [0.75]]),
            np.array([[0.75], [0.25]]),
            np.array([[0.1], [0.9]]),
        ]

        for point in interior_points:
            if octatope.contains(point):
                assert star.contains(point), f"Star should contain point {point.T} that is in Octatope"

    def test_soundness_bounds_overapproximate(self):
        """Test that Star bounds overapproximate Octatope bounds."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        octatope = Octatope.from_bounds(lb, ub)
        star = octatope.to_star()

        oct_lb, oct_ub = octatope.get_bounds()
        star_lb, star_ub = star.get_ranges()

        # Star bounds should contain Octatope bounds
        assert np.all(star_lb <= oct_lb + 1e-6), "Star lower bounds should be <= Octatope lower bounds"
        assert np.all(star_ub >= oct_ub - 1e-6), "Star upper bounds should be >= Octatope upper bounds"

    def test_soundness_after_affine_transformation(self):
        """Test soundness is preserved after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Apply affine transformation
        W = np.array([[2.0, 1.0], [0.0, 3.0]])
        b = np.array([[1.0], [2.0]])
        octatope_transformed = octatope.affine_map(W, b)

        star = octatope_transformed.to_star()

        # Test that transformed corners are preserved
        corners = [
            np.array([[0.0], [0.0]]),
            np.array([[1.0], [0.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0], [1.0]]),
        ]

        for corner in corners:
            # Apply same transformation to corner
            transformed_corner = W @ corner + b

            if octatope_transformed.contains(transformed_corner):
                assert star.contains(transformed_corner), \
                    f"Star should contain transformed corner {transformed_corner.T}"

    def test_soundness_random_points(self):
        """Test soundness with random points in the Octatope."""
        np.random.seed(42)
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)
        star = octatope.to_star()

        # Generate random points in the box
        n_samples = 50
        for _ in range(n_samples):
            point = np.random.uniform(lb.flatten(), ub.flatten()).reshape(-1, 1)

            # Only test points that are actually in the octatope
            if octatope.contains(point):
                assert star.contains(point), \
                    f"Star should contain point {point.T} that is in Octatope"

    def test_soundness_with_extra_constraints(self):
        """Test soundness when Octatope has extra constraints."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Add constraint: x + y <= 1.5
        H = np.array([[1.0, 1.0]])
        g = np.array([[1.5]])
        octatope_constrained = octatope.intersect_half_space(H, g)

        star = octatope_constrained.to_star()

        # Test points that satisfy the constraint
        test_points = [
            np.array([[0.5], [0.5]]),  # sum = 1.0 < 1.5
            np.array([[0.7], [0.7]]),  # sum = 1.4 < 1.5
            np.array([[0.2], [0.8]]),  # sum = 1.0 < 1.5
        ]

        for point in test_points:
            if octatope_constrained.contains(point):
                assert star.contains(point), \
                    f"Star should contain point {point.T} in constrained Octatope"

    def test_soundness_center_point(self):
        """Test that the center point is always contained."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        octatope = Octatope.from_bounds(lb, ub)
        star = octatope.to_star()

        center = octatope.center.reshape(-1, 1)

        assert octatope.contains(center), "Octatope should contain its center"
        assert star.contains(center), "Star should contain Octatope's center"


class TestConversionConsistency:
    """Test consistency between Hexatope and Octatope conversions."""

    def test_hexatope_octatope_star_same_bounds(self):
        """Test that Hexatope and Octatope Stars have consistent bounds for boxes."""
        lb = np.array([[0.0], [1.0]])
        ub = np.array([[1.0], [3.0]])

        hexatope = Hexatope.from_bounds(lb, ub)
        octatope = Octatope.from_bounds(lb, ub)

        star_from_hex = hexatope.to_star()
        star_from_oct = octatope.to_star()

        hex_star_lb, hex_star_ub = star_from_hex.get_ranges()
        oct_star_lb, oct_star_ub = star_from_oct.get_ranges()

        # Both should give same bounds (within tolerance)
        np.testing.assert_allclose(hex_star_lb, oct_star_lb, atol=1e-5,
                                   err_msg="Hexatope and Octatope Star lower bounds should match")
        np.testing.assert_allclose(hex_star_ub, oct_star_ub, atol=1e-5,
                                   err_msg="Hexatope and Octatope Star upper bounds should match")

    def test_both_contain_same_test_points(self):
        """Test that Stars from Hexatope and Octatope contain the same test points."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])

        hexatope = Hexatope.from_bounds(lb, ub)
        octatope = Octatope.from_bounds(lb, ub)

        star_from_hex = hexatope.to_star()
        star_from_oct = octatope.to_star()

        test_points = [
            np.array([[0.5], [0.5]]),
            np.array([[0.0], [0.0]]),
            np.array([[1.0], [1.0]]),
            np.array([[0.25], [0.75]]),
        ]

        for point in test_points:
            hex_contains = star_from_hex.contains(point)
            oct_contains = star_from_oct.contains(point)

            assert hex_contains == oct_contains, \
                f"Both Stars should agree on containment of point {point.T}"
