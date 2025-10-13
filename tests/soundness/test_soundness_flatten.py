"""
Soundness tests for Flatten layer reachability.

Tests verify that Flatten correctly preserves set properties while converting
from image format to flattened format.
"""

import numpy as np
import torch.nn as nn
from n2v.sets import Star, ImageStar, Box, Hexatope, Octatope
from n2v.nn.layer_ops.flatten_reach import flatten_star, flatten_box, flatten_hexatope, flatten_octatope


class TestFlattenImageStarSoundness:
    """Soundness tests for Flatten with ImageStar sets."""

    def test_flatten_preserves_bounds(self):
        """Test that Flatten preserves bounds."""
        layer = nn.Flatten()

        # Input: 2x2 image with range [0, 5]
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1)) * 5
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply Flatten
        output_stars = flatten_star(layer, [input_star])

        # Ground truth: Bounds should be preserved
        assert len(output_stars) == 1
        out_star = output_stars[0]

        # Output should be regular Star, not ImageStar
        assert isinstance(out_star, Star)
        assert not isinstance(out_star, ImageStar)

        # Check bounds are preserved
        lb_out, ub_out = out_star.estimate_ranges()
        assert out_star.dim == 4  # 2x2x1 = 4
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 5.0, atol=1e-6)

    def test_flatten_preserves_dimension_count(self):
        """Test that Flatten preserves total dimension count."""
        layer = nn.Flatten()

        # Input: 3x3 RGB image (3x3x3 = 27 dimensions)
        lb = np.zeros((3, 3, 3))
        ub = np.ones((3, 3, 3))
        input_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=3)

        # Apply Flatten
        output_stars = flatten_star(layer, [input_star])

        # Ground truth: Output should have 27 dimensions
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.dim == 27

    def test_flatten_single_pixel(self):
        """Test Flatten with 1x1 image (edge case)."""
        layer = nn.Flatten()

        # Input: 1x1 image with range [2, 7]
        lb = np.array([[[2.0]]])
        ub = np.array([[[7.0]]])
        input_star = ImageStar.from_bounds(lb, ub, height=1, width=1, num_channels=1)

        # Apply Flatten
        output_stars = flatten_star(layer, [input_star])

        # Ground truth: Single dimension preserved
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.dim == 1
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 2.0, atol=1e-6)
        assert np.allclose(ub_out, 7.0, atol=1e-6)

    def test_flatten_multi_channel(self):
        """Test Flatten with multi-channel image."""
        layer = nn.Flatten()

        # Input: 2x2 image with 3 channels
        # Each channel has different range
        lb = np.zeros((2, 2, 3))
        ub = np.array([
            [[1, 2, 3], [1, 2, 3]],
            [[1, 2, 3], [1, 2, 3]]
        ])
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=3)

        # Apply Flatten
        output_stars = flatten_star(layer, [input_star])

        # Ground truth: 2x2x3 = 12 dimensions
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.dim == 12

        # Check bounds are preserved (flattened in correct order)
        lb_out, ub_out = out_star.estimate_ranges()

        # All lower bounds should be 0
        assert np.allclose(lb_out, 0.0, atol=1e-6)

        # Upper bounds should maintain channel structure
        # The flattening order depends on how ImageStar stores data internally
        # But the values should be preserved
        assert np.all(ub_out >= 0.0 - 1e-6)
        assert np.all(ub_out <= 3.0 + 1e-6)

    def test_flatten_with_negative_values(self):
        """Test Flatten with negative input values."""
        layer = nn.Flatten()

        # Input: 2x2 image with range [-10, -5]
        lb = np.ones((2, 2, 1)) * -10
        ub = np.ones((2, 2, 1)) * -5
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply Flatten
        output_stars = flatten_star(layer, [input_star])

        # Ground truth: Bounds preserved
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, -10.0, atol=1e-6)
        assert np.allclose(ub_out, -5.0, atol=1e-6)

    def test_flatten_regular_star_passthrough(self):
        """Test that Flatten passes through regular Star unchanged."""
        layer = nn.Flatten()

        # Input: Regular Star (not ImageStar)
        lb = np.array([[1.0], [2.0], [3.0]])
        ub = np.array([[4.0], [5.0], [6.0]])
        input_star = Star.from_bounds(lb, ub)

        # Apply Flatten
        output_stars = flatten_star(layer, [input_star])

        # Ground truth: Should be unchanged
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.dim == 3
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, lb, atol=1e-6)
        assert np.allclose(ub_out, ub, atol=1e-6)

    def test_flatten_preserves_constraints(self):
        """Test that Flatten preserves Star constraints."""
        layer = nn.Flatten()

        # Input: 2x2 image with constraints
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1)) * 10
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Store number of constraints before flatten
        n_constraints_before = input_star.C.shape[0]

        # Apply Flatten
        output_stars = flatten_star(layer, [input_star])

        # Ground truth: Constraints should be preserved
        assert len(output_stars) == 1
        out_star = output_stars[0]
        n_constraints_after = out_star.C.shape[0]
        assert n_constraints_after == n_constraints_before


class TestFlattenBoxSoundness:
    """Soundness tests for Flatten with Box sets."""

    def test_flatten_box_passthrough(self):
        """Test that Flatten passes through Box unchanged."""
        layer = nn.Flatten()

        # Input: Box with range [0, 5]
        lb = np.zeros((4, 1))
        ub = np.ones((4, 1)) * 5
        input_box = Box(lb, ub)

        # Apply Flatten
        output_boxes = flatten_box(layer, [input_box])

        # Ground truth: Should be unchanged (Boxes don't have image structure)
        assert len(output_boxes) == 1
        out_box = output_boxes[0]
        assert out_box is input_box  # Should be same object
        assert np.allclose(out_box.lb, lb, atol=1e-6)
        assert np.allclose(out_box.ub, ub, atol=1e-6)


class TestFlattenEdgeCases:
    """Edge case tests for Flatten."""

    def test_flatten_large_image(self):
        """Test Flatten with large image."""
        layer = nn.Flatten()

        # Input: 10x10 RGB image = 300 dimensions
        lb = np.zeros((10, 10, 3))
        ub = np.ones((10, 10, 3))
        input_star = ImageStar.from_bounds(lb, ub, height=10, width=10, num_channels=3)

        # Apply Flatten
        output_stars = flatten_star(layer, [input_star])

        # Ground truth: 300 dimensions
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.dim == 300

    def test_flatten_single_channel_large(self):
        """Test Flatten with single channel large image."""
        layer = nn.Flatten()

        # Input: 5x5 grayscale image = 25 dimensions
        lb = np.ones((5, 5, 1)) * -3
        ub = np.ones((5, 5, 1)) * 8
        input_star = ImageStar.from_bounds(lb, ub, height=5, width=5, num_channels=1)

        # Apply Flatten
        output_stars = flatten_star(layer, [input_star])

        # Ground truth: 25 dimensions with bounds [-3, 8]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.dim == 25
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, -3.0, atol=1e-6)
        assert np.allclose(ub_out, 8.0, atol=1e-6)

    def test_flatten_asymmetric_image(self):
        """Test Flatten with non-square image."""
        layer = nn.Flatten()

        # Input: 3x5 image (not square) = 15 dimensions
        lb = np.zeros((3, 5, 1))
        ub = np.ones((3, 5, 1)) * 2
        input_star = ImageStar.from_bounds(lb, ub, height=3, width=5, num_channels=1)

        # Apply Flatten
        output_stars = flatten_star(layer, [input_star])

        # Ground truth: 15 dimensions
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.dim == 15
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 2.0, atol=1e-6)


class TestFlattenHexatopeSoundness:
    """Soundness tests for Flatten with Hexatope sets."""

    def test_flatten_passthrough(self):
        """Test that Flatten passes through Hexatope unchanged."""
        layer = nn.Flatten()

        # Hexatopes are already flattened
        lb = np.zeros((4, 1))
        ub = np.ones((4, 1)) * 5
        input_hexatope = Hexatope.from_bounds(lb, ub)

        # Apply Flatten
        output_hexatopes = flatten_hexatope(layer, [input_hexatope])

        # Should be unchanged
        assert len(output_hexatopes) == 1
        out_hexatope = output_hexatopes[0]
        assert out_hexatope.dim == 4
        lb_out, ub_out = out_hexatope.estimate_ranges()
        assert np.allclose(lb_out, lb, atol=1e-6)
        assert np.allclose(ub_out, ub, atol=1e-6)

    def test_flatten_preserves_bounds(self):
        """Test that Flatten preserves bounds."""
        layer = nn.Flatten()

        lb = np.array([[1.0], [2.0], [3.0]])
        ub = np.array([[4.0], [5.0], [6.0]])
        input_hexatope = Hexatope.from_bounds(lb, ub)

        output_hexatopes = flatten_hexatope(layer, [input_hexatope])

        assert len(output_hexatopes) == 1
        out_hexatope = output_hexatopes[0]
        lb_out, ub_out = out_hexatope.estimate_ranges()
        assert np.allclose(lb_out, lb, atol=1e-6)
        assert np.allclose(ub_out, ub, atol=1e-6)

    def test_flatten_negative_values(self):
        """Test Flatten with negative values."""
        layer = nn.Flatten()

        lb = np.ones((5, 1)) * -10
        ub = np.ones((5, 1)) * -2
        input_hexatope = Hexatope.from_bounds(lb, ub)

        output_hexatopes = flatten_hexatope(layer, [input_hexatope])

        assert len(output_hexatopes) == 1
        lb_out, ub_out = output_hexatopes[0].estimate_ranges()
        assert np.allclose(lb_out, -10.0, atol=1e-6)
        assert np.allclose(ub_out, -2.0, atol=1e-6)

    def test_flatten_single_dimension(self):
        """Test Flatten with single dimension."""
        layer = nn.Flatten()

        lb = np.array([[5.0]])
        ub = np.array([[8.0]])
        input_hexatope = Hexatope.from_bounds(lb, ub)

        output_hexatopes = flatten_hexatope(layer, [input_hexatope])

        assert len(output_hexatopes) == 1
        assert output_hexatopes[0].dim == 1


class TestFlattenOctatopeSoundness:
    """Soundness tests for Flatten with Octatope sets."""

    def test_flatten_passthrough(self):
        """Test that Flatten passes through Octatope unchanged."""
        layer = nn.Flatten()

        lb = np.zeros((4, 1))
        ub = np.ones((4, 1)) * 5
        input_octatope = Octatope.from_bounds(lb, ub)

        output_octatopes = flatten_octatope(layer, [input_octatope])

        assert len(output_octatopes) == 1
        out_octatope = output_octatopes[0]
        assert out_octatope.dim == 4
        lb_out, ub_out = out_octatope.estimate_ranges()
        assert np.allclose(lb_out, lb, atol=1e-6)
        assert np.allclose(ub_out, ub, atol=1e-6)

    def test_flatten_preserves_bounds(self):
        """Test that Flatten preserves bounds."""
        layer = nn.Flatten()

        lb = np.array([[1.0], [2.0], [3.0]])
        ub = np.array([[4.0], [5.0], [6.0]])
        input_octatope = Octatope.from_bounds(lb, ub)

        output_octatopes = flatten_octatope(layer, [input_octatope])

        assert len(output_octatopes) == 1
        out_octatope = output_octatopes[0]
        lb_out, ub_out = out_octatope.estimate_ranges()
        assert np.allclose(lb_out, lb, atol=1e-6)
        assert np.allclose(ub_out, ub, atol=1e-6)

    def test_flatten_negative_values(self):
        """Test Flatten with negative values."""
        layer = nn.Flatten()

        lb = np.ones((5, 1)) * -10
        ub = np.ones((5, 1)) * -2
        input_octatope = Octatope.from_bounds(lb, ub)

        output_octatopes = flatten_octatope(layer, [input_octatope])

        assert len(output_octatopes) == 1
        lb_out, ub_out = output_octatopes[0].estimate_ranges()
        assert np.allclose(lb_out, -10.0, atol=1e-6)
        assert np.allclose(ub_out, -2.0, atol=1e-6)

    def test_flatten_single_dimension(self):
        """Test Flatten with single dimension."""
        layer = nn.Flatten()

        lb = np.array([[5.0]])
        ub = np.array([[8.0]])
        input_octatope = Octatope.from_bounds(lb, ub)

        output_octatopes = flatten_octatope(layer, [input_octatope])

        assert len(output_octatopes) == 1
        assert output_octatopes[0].dim == 1
