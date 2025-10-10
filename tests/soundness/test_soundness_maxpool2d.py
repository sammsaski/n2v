"""
Soundness tests for MaxPool2D layer reachability.

Tests verify mathematical correctness of MaxPool2D operations.
MaxPool is a non-linear operation that may require splitting for exact analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import ImageStar
from n2v.nn.layer_ops.maxpool2d_reach import maxpool2d_star


class TestMaxPool2DImageStarExact:
    """Soundness tests for exact MaxPool2D with ImageStar sets."""

    def test_all_positive_values_unique_max(self):
        """Test MaxPool2D with all positive values where max is unique."""
        # 2x2 MaxPool with stride=2 (non-overlapping)
        layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 2x2 image where each pixel is unique and positive
        # [[1, 2],    -> max pool -> [[4]]
        #  [3, 4]]
        lb = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (C, H, W)
        ub = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        lb = lb.transpose(1, 2, 0)  # (H, W, C)
        ub = ub.transpose(1, 2, 0)
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply MaxPool2D
        output_stars = maxpool2d_star(layer, [input_star], method='exact')

        # Ground truth: Output should be [[4]] (max of [1,2,3,4])
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 1
        assert out_star.width == 1
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 4.0, atol=1e-6)
        assert np.allclose(ub_out, 4.0, atol=1e-6)

    def test_overlapping_windows(self):
        """Test MaxPool2D with stride < kernel_size (overlapping windows)."""
        # 2x2 MaxPool with stride=1 (overlapping windows)
        layer = nn.MaxPool2d(kernel_size=2, stride=1)

        # Input: 3x3 image
        # [[1, 2, 1],    -> [[4, 5],   (4 windows)
        #  [3, 4, 5],        [6, 7]]
        #  [2, 6, 7]]
        lb = np.array([
            [[1.0], [2.0], [1.0]],
            [[3.0], [4.0], [5.0]],
            [[2.0], [6.0], [7.0]]
        ])
        ub = lb.copy()
        input_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        # Apply MaxPool2D
        output_stars = maxpool2d_star(layer, [input_star], method='exact')

        # Ground truth:
        # Window (0,0): max([1,2,3,4]) = 4
        # Window (0,1): max([2,1,4,5]) = 5
        # Window (1,0): max([3,4,2,6]) = 6
        # Window (1,1): max([4,5,6,7]) = 7
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 2
        assert out_star.width == 2

        # Check output values
        lb_out, ub_out = out_star.estimate_ranges()
        lb_out = lb_out.reshape(2, 2, 1)
        ub_out = ub_out.reshape(2, 2, 1)

        assert np.allclose(lb_out[0, 0, 0], 4.0, atol=1e-6)
        assert np.allclose(lb_out[0, 1, 0], 5.0, atol=1e-6)
        assert np.allclose(lb_out[1, 0, 0], 6.0, atol=1e-6)
        assert np.allclose(lb_out[1, 1, 0], 7.0, atol=1e-6)

    def test_max_with_uncertainty(self):
        """Test MaxPool2D with uncertain max (splitting required)."""
        # 2x2 MaxPool
        layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 2x2 image where values overlap
        # Pixel values: [1,2] x [1,2] x [1,2] x [1,2]
        # Max could be any pixel in range [1, 2]
        lb = np.ones((2, 2, 1))
        ub = np.ones((2, 2, 1)) * 2
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply MaxPool2D
        output_stars = maxpool2d_star(layer, [input_star], method='exact')

        # Ground truth: Exact method may split, but union should give [1, 2]
        assert len(output_stars) >= 1

        # Compute union of all output stars
        lb_min = np.inf
        ub_max = -np.inf
        for star in output_stars:
            lb_temp, ub_temp = star.get_ranges()
            lb_min = min(lb_min, lb_temp.min())
            ub_max = max(ub_max, ub_temp.max())

        assert lb_min >= 1.0 - 1e-6  # Max of [1,2] is at least 1
        assert ub_max <= 2.0 + 1e-6  # Max of [1,2] is at most 2


class TestMaxPool2DImageStarApprox:
    """Soundness tests for approximate MaxPool2D with ImageStar sets."""

    def test_all_positive_approx(self):
        """Test approximate MaxPool2D with all positive values."""
        # 2x2 MaxPool
        layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 2x2 image with range [1, 4]
        lb = np.ones((2, 2, 1))
        ub = np.ones((2, 2, 1)) * 4
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply MaxPool2D (approximate)
        output_stars = maxpool2d_star(layer, [input_star], method='approx')

        # Ground truth: Approx should over-approximate
        # Max of [1,4] x 4 pixels is at most 4
        assert len(output_stars) >= 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()

        # Upper bound should be 4 (max possible)
        assert ub_out.max() <= 4.0 + 1e-6
        # Lower bound should be at least 1 (conservative)
        assert lb_out.min() >= 1.0 - 1e-6

    def test_overapproximation_property(self):
        """Test that approximate MaxPool over-approximates exact result."""
        # 2x2 MaxPool
        layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 2x2 image with specific range [0, 3]
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1)) * 3
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Compute exact
        exact_stars = maxpool2d_star(layer, [input_star], method='exact')

        # Compute approx
        approx_stars = maxpool2d_star(layer, [input_star], method='approx')

        # Get bounds for exact (union of all stars)
        exact_lb = np.inf
        exact_ub = -np.inf
        for star in exact_stars:
            lb_temp, ub_temp = star.get_ranges()
            exact_lb = min(exact_lb, lb_temp.min())
            exact_ub = max(exact_ub, ub_temp.max())

        # Get bounds for approx
        approx_lb = np.inf
        approx_ub = -np.inf
        for star in approx_stars:
            lb_temp, ub_temp = star.estimate_ranges()
            approx_lb = min(approx_lb, lb_temp.min())
            approx_ub = max(approx_ub, ub_temp.max())

        # Soundness: approx should contain exact
        assert approx_lb <= exact_lb + 1e-6, \
            f"Approx lower bound {approx_lb} > exact lower bound {exact_lb}"
        assert exact_ub <= approx_ub + 1e-6, \
            f"Exact upper bound {exact_ub} > approx upper bound {approx_ub}"


class TestMaxPool2DEdgeCases:
    """Edge case tests for MaxPool2D."""

    def test_1x1_pooling(self):
        """Test 1x1 MaxPool (identity operation)."""
        # 1x1 MaxPool should be identity
        layer = nn.MaxPool2d(kernel_size=1, stride=1)

        # Input: 3x3 image
        lb = np.zeros((3, 3, 1))
        ub = np.ones((3, 3, 1)) * 5
        input_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        # Apply MaxPool2D
        output_stars = maxpool2d_star(layer, [input_star], method='exact')

        # Ground truth: Should be unchanged (identity)
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 3
        assert out_star.width == 3
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 5.0, atol=1e-6)

    def test_negative_values(self):
        """Test MaxPool with negative values."""
        # 2x2 MaxPool
        layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 2x2 image with negative values
        # [[-1, -2],    -> max pool -> [[-1]]  (max of negatives)
        #  [-3, -4]]
        lb = np.array([[[-1.0], [-2.0]], [[-3.0], [-4.0]]])
        ub = lb.copy()
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply MaxPool2D
        output_stars = maxpool2d_star(layer, [input_star], method='exact')

        # Ground truth: Max of [-1,-2,-3,-4] = -1
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, -1.0, atol=1e-6)
        assert np.allclose(ub_out, -1.0, atol=1e-6)

    def test_multi_channel(self):
        """Test MaxPool with multiple channels (RGB)."""
        # 2x2 MaxPool on 3-channel image
        layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 2x2 RGB image (channels pooled independently)
        lb = np.zeros((2, 2, 3))
        ub = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=3)

        # Apply MaxPool2D
        output_stars = maxpool2d_star(layer, [input_star], method='exact')

        # Ground truth: Each channel pooled independently
        # Channel 0: max([0,1]) = 1
        # Channel 1: max([0,2]) = 2
        # Channel 2: max([0,3]) = 3
        assert len(output_stars) >= 1

        # Check that output has correct dimensions
        out_star = output_stars[0]
        assert out_star.height == 1
        assert out_star.width == 1
        assert out_star.num_channels == 3
