"""
Tests for verify_specification function.
"""

import pytest
import numpy as np
from n2v.utils.verify_specification import verify_specification
from n2v.sets import Star, Box, HalfSpace
from n2v.sets.probabilistic_box import ProbabilisticBox


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

    def test_infeasible_intersection(self):
        """Test when intersection exists but is infeasible (empty Star).

        This is a critical edge case: intersect_half_space may return a Star object,
        but that Star represents an empty/infeasible set. The verification should
        check is_empty_set() and treat it as UNSAT (verified).

        This test addresses the bug found in ACAS Xu verification where all
        intersections were infeasible but verification returned UNKNOWN instead of UNSAT.
        """
        # Create a star: [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Property: x1 + x2 >= 10 (impossible for [0,1] x [0,1])
        # This creates a geometric intersection, but the Star will be infeasible
        # (no points in [0,1] x [0,1] can sum to >= 10)
        G = np.array([[-1, -1]], dtype=np.float32)
        g = np.array([[-10]], dtype=np.float32)
        halfspace = HalfSpace(G, g)

        result = verify_specification([star], halfspace)

        # Intersection is infeasible -> property satisfied (UNSAT)
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
        # Or in standard form: -output[1] + output[0] <= 0  =>  output[0] - output[1] <= 0
        G = np.array([[1, -1]], dtype=np.float32)
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
        # 1) class1 >= class0  ->  class1 - class0 >= 0  ->  class0 - class1 <= 0
        # 2) class2 >= class0  ->  class2 - class0 >= 0  ->  class0 - class2 <= 0

        G1 = np.array([[1, -1, 0]], dtype=np.float32)
        g1 = np.array([[0]], dtype=np.float32)
        unsafe1 = HalfSpace(G1, g1)

        G2 = np.array([[1, 0, -1]], dtype=np.float32)
        g2 = np.array([[0]], dtype=np.float32)
        unsafe2 = HalfSpace(G2, g2)

        result = verify_specification([reach_star], [unsafe1, unsafe2])

        # Neither unsafe condition should intersect
        # max(class1 - class0) = 0.15 - 0.7 = -0.55 < 0 ✓
        # max(class2 - class0) = 0.15 - 0.7 = -0.55 < 0 ✓
        assert result == 1


class TestVerifySpecificationMultiGroup:
    """Tests for multi-group property handling (AND logic across groups).

    VNN-LIB properties can have multiple top-level asserts that are ANDed.
    verify_specification must check all groups, not just the first.
    """

    def test_star_multi_group_all_disjoint_returns_satisfied(self):
        """When each group individually doesn't intersect, result is satisfied."""
        star = Star.from_bounds(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        # Group 0: x1 >= 0.5 (intersects [0,1]^2)
        hs0 = HalfSpace(np.array([[-1, 0]], dtype=np.float64), np.array([[-0.5]]))
        # Group 1: x2 >= 2 (does NOT intersect [0,1]^2)
        hs1 = HalfSpace(np.array([[0, -1]], dtype=np.float64), np.array([[-2.0]]))

        prop = [{'Hg': hs0}, {'Hg': hs1}]
        result = verify_specification([star], prop)

        # Group 1 is infeasible → no input satisfies both → satisfied
        assert result == 1

    def test_star_multi_group_all_intersect_returns_unknown(self):
        """When all groups individually intersect, result is unknown."""
        star = Star.from_bounds(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        # Group 0: x1 >= 0.5 (intersects)
        hs0 = HalfSpace(np.array([[-1, 0]], dtype=np.float64), np.array([[-0.5]]))
        # Group 1: x2 >= 0.5 (intersects)
        hs1 = HalfSpace(np.array([[0, -1]], dtype=np.float64), np.array([[-0.5]]))

        prop = [{'Hg': hs0}, {'Hg': hs1}]
        result = verify_specification([star], prop)

        # Both groups feasible simultaneously (e.g., x=(0.7, 0.7)) → unknown
        assert result == 2

    def test_box_multi_group_second_group_infeasible(self):
        """Box with multi-group property where second group is infeasible."""
        box = Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        # Group 0: x1 >= 0.3 (intersects)
        hs0 = HalfSpace(np.array([[-1, 0]], dtype=np.float64), np.array([[-0.3]]))
        # Group 1: x2 >= 5.0 (impossible in [0,1]^2)
        hs1 = HalfSpace(np.array([[0, -1]], dtype=np.float64), np.array([[-5.0]]))

        prop = [{'Hg': hs0}, {'Hg': hs1}]
        result = verify_specification([box], prop)

        assert result == 1, "Second group infeasible → satisfied"

    def test_multi_group_with_or_within_group(self):
        """Multi-group where one group has OR of halfspaces."""
        star = Star.from_bounds(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        # Group 0: x1 >= 0.8 OR x1 <= 0.2 (partially intersects)
        hs0a = HalfSpace(np.array([[-1, 0]], dtype=np.float64), np.array([[-0.8]]))
        hs0b = HalfSpace(np.array([[1, 0]], dtype=np.float64), np.array([[0.2]]))
        # Group 1: x2 >= 5.0 (impossible)
        hs1 = HalfSpace(np.array([[0, -1]], dtype=np.float64), np.array([[-5.0]]))

        prop = [{'Hg': [hs0a, hs0b]}, {'Hg': hs1}]
        result = verify_specification([star], prop)

        assert result == 1, "Group 1 infeasible → satisfied despite group 0 intersecting"

    def test_single_group_backwards_compatible(self):
        """Single-group property (list with one dict) should work as before."""
        star = Star.from_bounds(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        hs = HalfSpace(np.array([[-1, 0]], dtype=np.float64), np.array([[-2.0]]))
        prop = [{'Hg': hs}]
        result = verify_specification([star], prop)

        assert result == 1


class TestVerifySpecificationBox:
    """Tests for verify_specification with Box inputs (no Star conversion)."""

    def test_box_disjoint_single_halfspace(self):
        """Box clearly disjoint from halfspace should return 1 (satisfied)."""
        box = Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        # x1 >= 2 → -x1 <= -2: impossible for [0,1]^2
        G = np.array([[-1, 0]], dtype=np.float64)
        g = np.array([[-2]], dtype=np.float64)
        hs = HalfSpace(G, g)

        result = verify_specification([box], hs)
        assert result == 1

    def test_box_intersecting_single_halfspace(self):
        """Box intersecting halfspace should return 2 (unknown)."""
        box = Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        # x1 <= 0.5: intersects [0,1]^2
        G = np.array([[1, 0]], dtype=np.float64)
        g = np.array([[0.5]], dtype=np.float64)
        hs = HalfSpace(G, g)

        result = verify_specification([box], hs)
        assert result == 2

    def test_box_multirow_halfspace_disjoint(self):
        """Box disjoint from multi-row halfspace (like yolo's 5x21125)."""
        box = Box(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

        # Two constraints: x1 >= 5 AND x2 >= 5
        # Neither can be satisfied in [0,1]^3
        G = np.array([[-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        g = np.array([[-5], [-5]], dtype=np.float64)
        hs = HalfSpace(G, g)

        result = verify_specification([box], hs)
        assert result == 1

    def test_box_multirow_halfspace_intersecting(self):
        """Box intersects multi-row halfspace (all rows feasible simultaneously)."""
        box = Box(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

        # x1 <= 0.5 AND x2 <= 0.5: both satisfiable in [0,1]^3 at e.g. (0,0,0)
        G = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        g = np.array([[0.5], [0.5]], dtype=np.float64)
        hs = HalfSpace(G, g)

        result = verify_specification([box], hs)
        assert result == 2

    def test_box_multiple_halfspaces_or_logic(self):
        """Box with multiple halfspaces (OR logic) — one intersects."""
        box = Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        # hs1: x1 >= 2 (disjoint)
        hs1 = HalfSpace(np.array([[-1, 0]]), np.array([[-2]]))
        # hs2: x1 <= 0.5 (intersects)
        hs2 = HalfSpace(np.array([[1, 0]]), np.array([[0.5]]))

        result = verify_specification([box], [hs1, hs2])
        assert result == 2

    def test_box_multiple_halfspaces_all_disjoint(self):
        """Box with multiple halfspaces (OR logic) — all disjoint."""
        box = Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        hs1 = HalfSpace(np.array([[-1, 0]]), np.array([[-2]]))
        hs2 = HalfSpace(np.array([[0, -1]]), np.array([[-2]]))

        result = verify_specification([box], [hs1, hs2])
        assert result == 1

    def test_box_dict_property_format(self):
        """Box works with dict property format from vnnlib."""
        box = Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        G = np.array([[-1, 0]], dtype=np.float64)
        g = np.array([[-2]], dtype=np.float64)
        hs = HalfSpace(G, g)

        result = verify_specification([box], [{'Hg': hs}])
        assert result == 1

    def test_probabilistic_box_uses_box_path(self):
        """ProbabilisticBox (inherits Box) should use the Box fast path."""
        pbox = ProbabilisticBox(
            lb=np.array([0.0, 0.0]),
            ub=np.array([1.0, 1.0]),
            m=100, ell=99, epsilon=0.01
        )

        # Disjoint: x1 >= 2
        G = np.array([[-1, 0]], dtype=np.float64)
        g = np.array([[-2]], dtype=np.float64)
        hs = HalfSpace(G, g)

        result = verify_specification([pbox], hs)
        assert result == 1

    def test_box_high_dimensional_disjoint(self):
        """High-dimensional Box should be fast (no Star conversion).

        At 21K dims (yolo scale), Star conversion takes ~55s due to creating
        a 21K x 21K matrix. Box interval arithmetic should take < 1s.
        """
        import time
        n = 21125  # yolo output dimension

        box = Box(np.zeros(n), np.ones(n))

        # sum(x) >= 2*n: impossible for [0,1]^n (max sum = n)
        G = -np.ones((1, n), dtype=np.float64)
        g = np.array([[-2 * n]], dtype=np.float64)
        hs = HalfSpace(G, g)

        t0 = time.time()
        result = verify_specification([box], hs)
        elapsed = time.time() - t0

        assert result == 1
        assert elapsed < 5.0, f"Box verify_specification took {elapsed:.1f}s, expected < 5s"

    def test_box_matches_star_result(self):
        """Box path should give same result as Star path for equivalent sets."""
        lb = np.array([[0.7], [0.1]])
        ub = np.array([[0.9], [0.3]])

        box = Box(lb, ub)
        star = Star.from_bounds(lb, ub)

        # Robustness: class0 - class1 <= 0 (unsafe region)
        G = np.array([[1, -1]], dtype=np.float32)
        g = np.array([[0]], dtype=np.float32)
        hs = HalfSpace(G, g)

        result_box = verify_specification([box], hs)
        result_star = verify_specification([star], hs)
        assert result_box == result_star
