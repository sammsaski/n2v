"""
Tests for verify_specification function.
"""

import pytest
import numpy as np
from n2v.utils.verify_specification import verify_specification
from n2v.sets import Star, HalfSpace


class TestVerifySpecificationBasic:
    """Basic tests for verify_specification."""

    def test_single_halfspace_satisfied(self):
        """Test verification with single halfspace that is satisfied (no intersection)."""
        # Create a simple star: unit box [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Property: x1 >= 2 (represented as -x1 <= -2)
        # This should NOT intersect with [0,1] x [0,1]
        G = np.array([[-1, 0]], dtype=np.float32)
        g = np.array([[-2]], dtype=np.float32)
        halfspace = HalfSpace(G, g)

        result = verify_specification([star], halfspace)

        # Should be satisfied (no intersection)
        assert result == 1

    def test_single_halfspace_unknown(self):
        """Test verification with single halfspace that intersects (unknown)."""
        # Create a simple star: unit box [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Property: x1 <= 0.5
        # This WILL intersect with [0,1] x [0,1]
        G = np.array([[1, 0]], dtype=np.float32)
        g = np.array([[0.5]], dtype=np.float32)
        halfspace = HalfSpace(G, g)

        result = verify_specification([star], halfspace)

        # Should be unknown (intersection exists)
        assert result == 2

    def test_single_halfspace_from_dict(self):
        """Test verification with property as dictionary."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Property as dict (VNN-LIB format)
        G = np.array([[-1, 0]], dtype=np.float32)
        g = np.array([[-2]], dtype=np.float32)
        halfspace = HalfSpace(G, g)
        property_dict = {'Hg': halfspace}

        result = verify_specification([star], property_dict)

        assert result == 1

    def test_single_halfspace_from_list_of_dicts(self):
        """Test verification with property as list of dicts (VNN-LIB format)."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Property as list of dicts
        G = np.array([[-1, 0]], dtype=np.float32)
        g = np.array([[-2]], dtype=np.float32)
        halfspace = HalfSpace(G, g)
        property_list = [{'Hg': halfspace}]

        result = verify_specification([star], property_list)

        assert result == 1


class TestVerifySpecificationMultipleHalfspaces:
    """Tests for verification with multiple halfspaces (OR logic)."""

    def test_multiple_halfspaces_all_satisfied(self):
        """Test with multiple halfspaces where none intersect (satisfied)."""
        # Star: [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Multiple halfspaces (OR) - none should intersect
        # Property 1: x1 >= 2 (outside to the right)
        G1 = np.array([[-1, 0]], dtype=np.float32)
        g1 = np.array([[-2]], dtype=np.float32)
        hs1 = HalfSpace(G1, g1)

        # Property 2: x2 >= 2 (outside above)
        G2 = np.array([[0, -1]], dtype=np.float32)
        g2 = np.array([[-2]], dtype=np.float32)
        hs2 = HalfSpace(G2, g2)

        result = verify_specification([star], [hs1, hs2])

        # All halfspaces satisfied (no intersection)
        assert result == 1

    def test_multiple_halfspaces_one_intersects(self):
        """Test with multiple halfspaces where one intersects (unknown)."""
        # Star: [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Multiple halfspaces (OR)
        # Property 1: x1 >= 2 (doesn't intersect)
        G1 = np.array([[-1, 0]], dtype=np.float32)
        g1 = np.array([[-2]], dtype=np.float32)
        hs1 = HalfSpace(G1, g1)

        # Property 2: x1 <= 0.5 (DOES intersect)
        G2 = np.array([[1, 0]], dtype=np.float32)
        g2 = np.array([[0.5]], dtype=np.float32)
        hs2 = HalfSpace(G2, g2)

        result = verify_specification([star], [hs1, hs2])

        # One intersects -> unknown
        assert result == 2


class TestVerifySpecificationMultipleStars:
    """Tests for verification with multiple reach sets."""

    def test_multiple_stars_all_satisfied(self):
        """Test with multiple stars where all satisfy property."""
        # Star 1: [0,1] x [0,1]
        star1 = Star.from_bounds(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        # Star 2: [0.5,1.5] x [0.5,1.5]
        star2 = Star.from_bounds(np.array([[0.5], [0.5]]), np.array([[1.5], [1.5]]))

        # Property: x1 >= 2 (neither star intersects)
        G = np.array([[-1, 0]], dtype=np.float32)
        g = np.array([[-2]], dtype=np.float32)
        halfspace = HalfSpace(G, g)

        result = verify_specification([star1, star2], halfspace)

        # All stars satisfy property
        assert result == 1

    def test_multiple_stars_one_intersects(self):
        """Test with multiple stars where one intersects property."""
        # Star 1: [0,1] x [0,1]
        star1 = Star.from_bounds(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        # Star 2: [2,3] x [2,3] (this will intersect)
        star2 = Star.from_bounds(np.array([[2.0], [2.0]]), np.array([[3.0], [3.0]]))

        # Property: x1 >= 2 (star2 intersects)
        G = np.array([[-1, 0]], dtype=np.float32)
        g = np.array([[-2]], dtype=np.float32)
        halfspace = HalfSpace(G, g)

        result = verify_specification([star1, star2], halfspace)

        # One star intersects -> unknown
        assert result == 2


class TestVerifySpecificationEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_reach_set(self):
        """Test with empty reach set list."""
        G = np.array([[1, 0]], dtype=np.float32)
        g = np.array([[5]], dtype=np.float32)
        halfspace = HalfSpace(G, g)

        result = verify_specification([], halfspace)

        # No reach sets -> property satisfied (vacuously true)
        assert result == 1

    def test_invalid_property_type(self):
        """Test with invalid property type."""
        star = Star.from_bounds(np.array([[0.0]]), np.array([[1.0]]))

        # Invalid property type
        with pytest.raises(TypeError):
            verify_specification([star], "invalid_property")

    def test_higher_dimensional_verification(self):
        """Test verification in higher dimensional space."""
        # 4D star: [0,1]^4
        lb = np.array([[0.0], [0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Property: x1 + x2 + x3 + x4 >= 5 (doesn't intersect [0,1]^4)
        G = np.array([[-1, -1, -1, -1]], dtype=np.float32)
        g = np.array([[-5]], dtype=np.float32)
        halfspace = HalfSpace(G, g)

        result = verify_specification([star], halfspace)

        # Should be satisfied (max sum is 4, need >= 5)
        assert result == 1


class TestVerifySpecificationRealWorld:
    """Tests simulating real-world verification scenarios."""

    def test_robustness_verification_scenario(self):
        """Test typical robustness verification scenario."""
        # Reachable output set (e.g., class 0 vs class 1 scores)
        # Suppose nominal output is [0.8, 0.2] for [class0, class1]
        # Reachable set might be [0.7, 0.9] x [0.1, 0.3]
        lb = np.array([[0.7], [0.1]])
        ub = np.array([[0.9], [0.3]])
        reach_star = Star.from_bounds(lb, ub)

        # Property: class 0 should always be higher than class 1
        # i.e., output[0] > output[1]
        # Unsafe region: output[1] >= output[0]
        # Represented as: output[1] - output[0] >= 0
        # Or in standard form: -output[0] + output[1] <= 0
        G = np.array([[-1, 1]], dtype=np.float32)
        g = np.array([[0]], dtype=np.float32)
        unsafe_region = HalfSpace(G, g)

        result = verify_specification([reach_star], unsafe_region)

        # Check if reachable set intersects unsafe region
        # [0.7, 0.9] x [0.1, 0.3]: max(output[1] - output[0]) = 0.3 - 0.7 = -0.4 < 0
        # So no intersection -> verified robust
        assert result == 1

    def test_multi_class_verification_scenario(self):
        """Test multi-class robustness verification."""
        # 3-class output: [class0, class1, class2]
        # True class is 0, reachable output approximately [0.7-0.9, 0.05-0.15, 0.05-0.15]
        lb = np.array([[0.7], [0.05], [0.05]])
        ub = np.array([[0.9], [0.15], [0.15]])
        reach_star = Star.from_bounds(lb, ub)

        # Property: class 0 should be highest
        # Unsafe region (OR of two conditions):
        # 1) class1 >= class0  ->  -class0 + class1 <= 0
        # 2) class2 >= class0  ->  -class0 + class2 <= 0

        G1 = np.array([[-1, 1, 0]], dtype=np.float32)
        g1 = np.array([[0]], dtype=np.float32)
        unsafe1 = HalfSpace(G1, g1)

        G2 = np.array([[-1, 0, 1]], dtype=np.float32)
        g2 = np.array([[0]], dtype=np.float32)
        unsafe2 = HalfSpace(G2, g2)

        result = verify_specification([reach_star], [unsafe1, unsafe2])

        # Neither unsafe condition should intersect
        # max(class1 - class0) = 0.15 - 0.7 = -0.55 < 0 ✓
        # max(class2 - class0) = 0.15 - 0.7 = -0.55 < 0 ✓
        assert result == 1
