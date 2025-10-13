"""
Soundness tests for ReLU layer reachability.

These tests verify that ReLU layer reachability produces mathematically correct results
for all set representations (Star, Zono, Box) and methods (exact, approx).
"""

import pytest
import numpy as np
import torch.nn as nn
from n2v.sets import Star, Zono, Box, Hexatope, Octatope
from n2v.nn.layer_ops.relu_reach import (
    relu_star_exact, relu_star_approx,
    relu_zono_approx, relu_box,
    relu_hexatope, relu_octatope
)


class TestReLUStarExactSoundness:
    """Soundness tests for exact ReLU reachability with Star sets."""

    def test_all_positive_input(self):
        """Test ReLU with all-positive input (no splitting needed)."""
        # Input: [1, 2] x [1, 2]  (all positive)
        lb = np.array([[1.0], [1.0]])
        ub = np.array([[2.0], [2.0]])
        input_star = Star.from_bounds(lb, ub)

        # Apply ReLU
        output_stars = relu_star_exact([input_star])

        # Ground truth: ReLU([1,2]) = [1,2] (unchanged)
        assert len(output_stars) == 1  # No splitting
        output_lb, output_ub = output_stars[0].estimate_ranges()

        assert np.allclose(output_lb, lb, atol=1e-6)
        assert np.allclose(output_ub, ub, atol=1e-6)

    def test_all_negative_input(self):
        """Test ReLU with all-negative input (zeros out)."""
        # Input: [-2, -1] x [-2, -1]  (all negative)
        lb = np.array([[-2.0], [-2.0]])
        ub = np.array([[-1.0], [-1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Apply ReLU
        output_stars = relu_star_exact([input_star])

        # Ground truth: ReLU([-2,-1]) = [0,0]
        assert len(output_stars) == 1  # No splitting needed
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected = np.array([[0.0], [0.0]])
        assert np.allclose(output_lb, expected, atol=1e-6)
        assert np.allclose(output_ub, expected, atol=1e-6)

    def test_single_neuron_crossing_zero(self):
        """Test ReLU with input crossing zero (requires splitting)."""
        # Input: [-1, 1] (crosses zero)
        lb = np.array([[-1.0]])
        ub = np.array([[1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Apply ReLU
        output_stars = relu_star_exact([input_star])

        # Ground truth: Should split into two cases
        # Case 1: x <= 0 → ReLU(x) = 0
        # Case 2: x > 0 → ReLU(x) = x in (0, 1]
        # Union covers [0, 1]
        assert len(output_stars) == 2

        # Collect all output ranges
        all_outputs = []
        for star in output_stars:
            if not star.is_empty_set():
                lb_out, ub_out = star.estimate_ranges()
                all_outputs.append((lb_out[0, 0], ub_out[0, 0]))

        # Verify union covers [0, 1]
        min_out = min(x[0] for x in all_outputs)
        max_out = max(x[1] for x in all_outputs)

        assert min_out <= 0.0 + 1e-6
        assert max_out >= 1.0 - 1e-6

    def test_two_neurons_one_positive_one_negative(self):
        """Test ReLU with one positive and one negative dimension."""
        # Input: [1, 2] x [-2, -1]
        # x1 is all positive, x2 is all negative
        lb = np.array([[1.0], [-2.0]])
        ub = np.array([[2.0], [-1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Apply ReLU
        output_stars = relu_star_exact([input_star])

        # Ground truth:
        # y1 = ReLU([1,2]) = [1,2]
        # y2 = ReLU([-2,-1]) = [0,0]
        assert len(output_stars) == 1  # No splitting needed
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected_lb = np.array([[1.0], [0.0]])
        expected_ub = np.array([[2.0], [0.0]])

        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_two_neurons_both_crossing_zero(self):
        """Test ReLU with both dimensions crossing zero."""
        # Input: [-1, 1] x [-1, 1]
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Apply ReLU
        output_stars = relu_star_exact([input_star])

        # Ground truth: Should split into 2^2 = 4 cases
        # But some may be infeasible, so we have 1 to 4 stars
        assert 1 <= len(output_stars) <= 4

        # Verify union covers [0, 1] x [0, 1]
        lb_out = np.ones((2, 1)) * np.inf
        ub_out = np.ones((2, 1)) * -np.inf

        for star in output_stars:
            if not star.is_empty_set():
                lb_temp, ub_temp = star.estimate_ranges()
                lb_out = np.minimum(lb_out, lb_temp)
                ub_out = np.maximum(ub_out, ub_temp)

        # Union should cover [0, 1] x [0, 1]
        assert lb_out[0, 0] <= 0.0 + 1e-6
        assert lb_out[1, 0] <= 0.0 + 1e-6
        assert ub_out[0, 0] >= 1.0 - 1e-6
        assert ub_out[1, 0] >= 1.0 - 1e-6

    def test_boundary_case_at_zero(self):
        """Test ReLU with input exactly at zero."""
        # Input: [0, 0] x [0, 1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[0.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Apply ReLU
        output_stars = relu_star_exact([input_star])

        # Ground truth: First dimension stays 0, second stays [0, 1]
        assert len(output_stars) >= 1

        # Check that output covers expected range
        lb_out = np.ones((2, 1)) * np.inf
        ub_out = np.ones((2, 1)) * -np.inf

        for star in output_stars:
            if not star.is_empty_set():
                lb_temp, ub_temp = star.estimate_ranges()
                lb_out = np.minimum(lb_out, lb_temp)
                ub_out = np.maximum(ub_out, ub_temp)

        assert np.allclose(lb_out, np.array([[0.0], [0.0]]), atol=1e-6)
        assert np.allclose(ub_out, np.array([[0.0], [1.0]]), atol=1e-6)


class TestReLUStarApproxSoundness:
    """Soundness tests for approximate ReLU reachability with Star sets."""

    def test_overapproximation_property(self):
        """Test that approximate ReLU over-approximates exact result."""
        # Input crossing zero: [-1, 1] x [-1, 1]
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Compute exact
        exact_stars = relu_star_exact([input_star])

        # Compute approx with various relaxation factors
        for relax_factor in [0.0, 0.5, 1.0]:
            approx_stars = relu_star_approx([input_star], relax_factor=relax_factor)

            # Get bounds for exact (union of all stars)
            # Use get_ranges() for exact stars to get tight LP-based bounds
            exact_lb = np.ones((2, 1)) * np.inf
            exact_ub = np.ones((2, 1)) * -np.inf
            for star in exact_stars:
                if not star.is_empty_set():
                    lb_temp, ub_temp = star.get_ranges()
                    exact_lb = np.minimum(exact_lb, lb_temp)
                    exact_ub = np.maximum(exact_ub, ub_temp)

            # Get bounds for approx
            # Use estimate_ranges() for approx stars (fast over-approximation)
            approx_lb = np.ones((2, 1)) * np.inf
            approx_ub = np.ones((2, 1)) * -np.inf
            for star in approx_stars:
                if not star.is_empty_set():
                    lb_temp, ub_temp = star.estimate_ranges()
                    approx_lb = np.minimum(approx_lb, lb_temp)
                    approx_ub = np.maximum(approx_ub, ub_temp)

            # Soundness: approx should contain exact
            # approx_lb <= exact_lb and exact_ub <= approx_ub
            assert np.all(approx_lb <= exact_lb + 1e-6), \
                f"Approx lower bound {approx_lb} > exact lower bound {exact_lb}"
            assert np.all(exact_ub <= approx_ub + 1e-6), \
                f"Exact upper bound {exact_ub} > approx upper bound {approx_ub}"

    def test_relaxation_factor_monotonicity(self):
        """Test that larger relaxation factor gives larger over-approximation."""
        lb = np.array([[-1.0]])
        ub = np.array([[1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Compute with different relaxation factors
        results = {}
        for rf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            output_stars = relu_star_approx([input_star], relax_factor=rf)
            lb_out, ub_out = output_stars[0].estimate_ranges()
            results[rf] = (lb_out[0, 0], ub_out[0, 0])

        # Larger relaxation should give looser bounds
        # (lower lb, higher ub)
        for rf1 in [0.0, 0.25, 0.5, 0.75]:
            for rf2 in [rf1 + 0.25]:
                if rf2 <= 1.0:
                    lb1, ub1 = results[rf1]
                    lb2, ub2 = results[rf2]
                    # rf2 should have looser bounds
                    assert lb2 <= lb1 + 1e-6, f"RF {rf2} has tighter lower bound than {rf1}"


class TestReLUZonoSoundness:
    """Soundness tests for approximate ReLU with Zonotope sets."""

    def test_all_positive_input(self):
        """Test ReLU zonotope with all-positive input."""
        # Zonotope: center=[1, 1], generator=[0.5, 0.5]
        # Represents [0.5, 1.5] x [0.5, 1.5] (all positive)
        c = np.array([[1.0], [1.0]])
        V = np.array([[0.5], [0.5]])
        input_zono = Zono(c, V)

        # Apply ReLU
        output_zonos = relu_zono_approx([input_zono])

        # Ground truth: Should be unchanged (all positive)
        assert len(output_zonos) == 1
        assert np.allclose(output_zonos[0].c, c, atol=1e-6)
        assert np.allclose(output_zonos[0].V, V, atol=1e-6)

    def test_all_negative_input(self):
        """Test ReLU zonotope with all-negative input."""
        # Zonotope: center=[-1, -1], generator=[0.5, 0.5]
        # Represents [-1.5, -0.5] x [-1.5, -0.5] (all negative)
        c = np.array([[-1.0], [-1.0]])
        V = np.array([[0.5], [0.5]])
        input_zono = Zono(c, V)

        # Apply ReLU
        output_zonos = relu_zono_approx([input_zono])

        # Ground truth: Should be zero
        assert len(output_zonos) == 1
        assert np.allclose(output_zonos[0].c, np.zeros((2, 1)), atol=1e-6)


class TestReLUBoxSoundness:
    """Soundness tests for ReLU with Box sets."""

    def test_all_positive_input(self):
        """Test ReLU box with all-positive input."""
        lb = np.array([[1.0], [1.0]])
        ub = np.array([[2.0], [2.0]])
        input_box = Box(lb, ub)

        # Apply ReLU
        output_boxes = relu_box([input_box])

        # Ground truth: unchanged
        assert len(output_boxes) == 1
        assert np.allclose(output_boxes[0].lb, lb, atol=1e-6)
        assert np.allclose(output_boxes[0].ub, ub, atol=1e-6)

    def test_all_negative_input(self):
        """Test ReLU box with all-negative input."""
        lb = np.array([[-2.0], [-2.0]])
        ub = np.array([[-1.0], [-1.0]])
        input_box = Box(lb, ub)

        # Apply ReLU
        output_boxes = relu_box([input_box])

        # Ground truth: all zeros
        assert len(output_boxes) == 1
        assert np.allclose(output_boxes[0].lb, np.zeros((2, 1)), atol=1e-6)
        assert np.allclose(output_boxes[0].ub, np.zeros((2, 1)), atol=1e-6)

    def test_crossing_zero(self):
        """Test ReLU box with input crossing zero."""
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        input_box = Box(lb, ub)

        # Apply ReLU
        output_boxes = relu_box([input_box])

        # Ground truth: [0, 1] x [0, 1]
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1.0], [1.0]])

        assert len(output_boxes) == 1
        assert np.allclose(output_boxes[0].lb, expected_lb, atol=1e-6)
        assert np.allclose(output_boxes[0].ub, expected_ub, atol=1e-6)


class TestReLUEdgeCases:
    """Edge case soundness tests for ReLU layer."""

    def test_very_small_positive_values(self):
        """Test ReLU with very small positive values."""
        lb = np.array([[1e-10], [1e-10]])
        ub = np.array([[1e-9], [1e-9]])
        input_star = Star.from_bounds(lb, ub)

        # Apply ReLU
        output_stars = relu_star_exact([input_star])

        # Ground truth: Should pass through unchanged (all positive)
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        assert np.allclose(output_lb, lb, atol=1e-12)
        assert np.allclose(output_ub, ub, atol=1e-12)

    def test_very_large_positive_values(self):
        """Test ReLU with very large positive values."""
        lb = np.array([[1e6], [1e6]])
        ub = np.array([[1e7], [1e7]])
        input_star = Star.from_bounds(lb, ub)

        # Apply ReLU
        output_stars = relu_star_exact([input_star])

        # Ground truth: Should pass through unchanged
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        assert np.allclose(output_lb, lb, rtol=1e-6)
        assert np.allclose(output_ub, ub, rtol=1e-6)

    def test_asymmetric_crossing_zero(self):
        """Test ReLU with asymmetric range crossing zero."""
        # Input: [-2, 1] (more negative than positive)
        lb = np.array([[-2.0]])
        ub = np.array([[1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Apply ReLU
        output_stars = relu_star_exact([input_star])

        # Ground truth: Output should cover [0, 1]
        lb_out = np.inf
        ub_out = -np.inf

        for star in output_stars:
            if not star.is_empty_set():
                lb_temp, ub_temp = star.estimate_ranges()
                lb_out = min(lb_out, lb_temp[0, 0])
                ub_out = max(ub_out, ub_temp[0, 0])

        assert lb_out <= 0.0 + 1e-6
        assert ub_out >= 1.0 - 1e-6

    def test_high_dimensional_input(self):
        """Test ReLU with high-dimensional input."""
        # 10D input crossing zero in each dimension
        lb = np.ones((10, 1)) * -1.0
        ub = np.ones((10, 1)) * 1.0
        input_star = Star.from_bounds(lb, ub)

        # Apply ReLU
        output_stars = relu_star_exact([input_star])

        # Ground truth: Output should cover [0, 1]^10
        # (may have many stars due to splitting)
        assert len(output_stars) >= 1

        # Verify union covers expected range
        lb_out = np.ones((10, 1)) * np.inf
        ub_out = np.ones((10, 1)) * -np.inf

        for star in output_stars:
            if not star.is_empty_set():
                lb_temp, ub_temp = star.estimate_ranges()
                lb_out = np.minimum(lb_out, lb_temp)
                ub_out = np.maximum(ub_out, ub_temp)

        # Each dimension should cover [0, 1]
        assert np.all(lb_out <= 0.0 + 1e-6)
        assert np.all(ub_out >= 1.0 - 1e-6)


class TestReLUHexatopeSoundness:
    """Soundness tests for ReLU with Hexatope sets."""

    def test_all_positive_input(self):
        """Test ReLU hexatope with all-positive input."""
        lb = np.array([[1.0], [1.0]])
        ub = np.array([[2.0], [2.0]])
        input_hexatope = Hexatope.from_bounds(lb, ub)

        # Apply ReLU
        output_hexatopes = relu_hexatope([input_hexatope])

        # Ground truth: unchanged
        assert len(output_hexatopes) == 1
        output_lb, output_ub = output_hexatopes[0].estimate_ranges()
        assert np.allclose(output_lb, lb, atol=1e-6)
        assert np.allclose(output_ub, ub, atol=1e-6)

    def test_all_negative_input(self):
        """Test ReLU hexatope with all-negative input."""
        lb = np.array([[-2.0], [-2.0]])
        ub = np.array([[-1.0], [-1.0]])
        input_hexatope = Hexatope.from_bounds(lb, ub)

        # Apply ReLU
        output_hexatopes = relu_hexatope([input_hexatope])

        # Ground truth: all zeros
        assert len(output_hexatopes) == 1
        output_lb, output_ub = output_hexatopes[0].estimate_ranges()
        assert np.allclose(output_lb, np.zeros((2, 1)), atol=1e-6)
        assert np.allclose(output_ub, np.zeros((2, 1)), atol=1e-6)

    def test_crossing_zero(self):
        """Test ReLU hexatope with input crossing zero."""
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        input_hexatope = Hexatope.from_bounds(lb, ub)

        # Apply ReLU
        output_hexatopes = relu_hexatope([input_hexatope])

        # Ground truth: [0, 1] x [0, 1]
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1.0], [1.0]])

        assert len(output_hexatopes) == 1
        output_lb, output_ub = output_hexatopes[0].estimate_ranges()
        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_mixed_dimensions(self):
        """Test ReLU with one positive and one negative dimension."""
        # x1 all positive, x2 all negative
        lb = np.array([[1.0], [-2.0]])
        ub = np.array([[2.0], [-1.0]])
        input_hexatope = Hexatope.from_bounds(lb, ub)

        # Apply ReLU
        output_hexatopes = relu_hexatope([input_hexatope])

        # Ground truth: [1,2] x [0,0]
        expected_lb = np.array([[1.0], [0.0]])
        expected_ub = np.array([[2.0], [0.0]])

        assert len(output_hexatopes) == 1
        output_lb, output_ub = output_hexatopes[0].estimate_ranges()
        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_asymmetric_crossing(self):
        """Test ReLU with asymmetric range crossing zero."""
        # More negative than positive
        lb = np.array([[-2.0]])
        ub = np.array([[1.0]])
        input_hexatope = Hexatope.from_bounds(lb, ub)

        # Apply ReLU
        output_hexatopes = relu_hexatope([input_hexatope])

        # Ground truth: [0, 1]
        expected_lb = np.array([[0.0]])
        expected_ub = np.array([[1.0]])

        assert len(output_hexatopes) == 1
        output_lb, output_ub = output_hexatopes[0].estimate_ranges()
        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_boundary_at_zero(self):
        """Test ReLU with input exactly at zero."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[0.0], [1.0]])
        input_hexatope = Hexatope.from_bounds(lb, ub)

        # Apply ReLU
        output_hexatopes = relu_hexatope([input_hexatope])

        # Ground truth: [0,0] x [0,1]
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[0.0], [1.0]])

        assert len(output_hexatopes) == 1
        output_lb, output_ub = output_hexatopes[0].estimate_ranges()
        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)


class TestReLUOctatopeSoundness:
    """Soundness tests for ReLU with Octatope sets."""

    def test_all_positive_input(self):
        """Test ReLU octatope with all-positive input."""
        lb = np.array([[1.0], [1.0]])
        ub = np.array([[2.0], [2.0]])
        input_octatope = Octatope.from_bounds(lb, ub)

        # Apply ReLU
        output_octatopes = relu_octatope([input_octatope])

        # Ground truth: unchanged
        assert len(output_octatopes) == 1
        output_lb, output_ub = output_octatopes[0].estimate_ranges()
        assert np.allclose(output_lb, lb, atol=1e-6)
        assert np.allclose(output_ub, ub, atol=1e-6)

    def test_all_negative_input(self):
        """Test ReLU octatope with all-negative input."""
        lb = np.array([[-2.0], [-2.0]])
        ub = np.array([[-1.0], [-1.0]])
        input_octatope = Octatope.from_bounds(lb, ub)

        # Apply ReLU
        output_octatopes = relu_octatope([input_octatope])

        # Ground truth: all zeros
        assert len(output_octatopes) == 1
        output_lb, output_ub = output_octatopes[0].estimate_ranges()
        assert np.allclose(output_lb, np.zeros((2, 1)), atol=1e-6)
        assert np.allclose(output_ub, np.zeros((2, 1)), atol=1e-6)

    def test_crossing_zero(self):
        """Test ReLU octatope with input crossing zero."""
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        input_octatope = Octatope.from_bounds(lb, ub)

        # Apply ReLU
        output_octatopes = relu_octatope([input_octatope])

        # Ground truth: [0, 1] x [0, 1]
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1.0], [1.0]])

        assert len(output_octatopes) == 1
        output_lb, output_ub = output_octatopes[0].estimate_ranges()
        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_mixed_dimensions(self):
        """Test ReLU with one positive and one negative dimension."""
        # x1 all positive, x2 all negative
        lb = np.array([[1.0], [-2.0]])
        ub = np.array([[2.0], [-1.0]])
        input_octatope = Octatope.from_bounds(lb, ub)

        # Apply ReLU
        output_octatopes = relu_octatope([input_octatope])

        # Ground truth: [1,2] x [0,0]
        expected_lb = np.array([[1.0], [0.0]])
        expected_ub = np.array([[2.0], [0.0]])

        assert len(output_octatopes) == 1
        output_lb, output_ub = output_octatopes[0].estimate_ranges()
        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_asymmetric_crossing(self):
        """Test ReLU with asymmetric range crossing zero."""
        # More negative than positive
        lb = np.array([[-2.0]])
        ub = np.array([[1.0]])
        input_octatope = Octatope.from_bounds(lb, ub)

        # Apply ReLU
        output_octatopes = relu_octatope([input_octatope])

        # Ground truth: [0, 1]
        expected_lb = np.array([[0.0]])
        expected_ub = np.array([[1.0]])

        assert len(output_octatopes) == 1
        output_lb, output_ub = output_octatopes[0].estimate_ranges()
        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_boundary_at_zero(self):
        """Test ReLU with input exactly at zero."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[0.0], [1.0]])
        input_octatope = Octatope.from_bounds(lb, ub)

        # Apply ReLU
        output_octatopes = relu_octatope([input_octatope])

        # Ground truth: [0,0] x [0,1]
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[0.0], [1.0]])

        assert len(output_octatopes) == 1
        output_lb, output_ub = output_octatopes[0].estimate_ranges()
        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)
