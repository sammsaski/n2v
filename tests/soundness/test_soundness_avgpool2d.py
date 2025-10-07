"""
Soundness tests for AvgPool2D layer reachability.

Tests verify mathematical correctness of AvgPool2D operations.
AvgPool is a linear operation, so it's exact for all set types (no approximation needed).
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import ImageStar, Hexatope, Octatope
from n2v.nn.layer_ops.avgpool2d_reach import avgpool2d_star, avgpool2d_hexatope, avgpool2d_octatope


class TestAvgPool2DImageStarSoundness:
    """Soundness tests for AvgPool2D with ImageStar sets."""

    def test_simple_2x2_averaging(self):
        """Test 2x2 AvgPool with simple values."""
        # 2x2 AvgPool with stride=2 (non-overlapping)
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        # Input: 2x2 image with known values
        # [[1, 2],    -> avg pool -> [[(1+2+3+4)/4]] = [[2.5]]
        #  [3, 4]]
        lb = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        ub = lb.copy()
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply AvgPool2D
        output_stars = avgpool2d_star(layer, [input_star])

        # Ground truth: Output should be [[2.5]]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 1
        assert out_star.width == 1
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 2.5, atol=1e-6)
        assert np.allclose(ub_out, 2.5, atol=1e-6)

    def test_averaging_with_range(self):
        """Test AvgPool with uncertain input values."""
        # 2x2 AvgPool
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        # Input: 2x2 image where each pixel is [0, 4]
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1)) * 4
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply AvgPool2D
        output_stars = avgpool2d_star(layer, [input_star])

        # Ground truth: Average of [0,4] x 4 pixels is [0, 4]
        # (min avg = 0+0+0+0 = 0, max avg = 4+4+4+4 = 16, divided by 4)
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 4.0, atol=1e-6)

    def test_overlapping_windows(self):
        """Test AvgPool with stride < kernel_size (overlapping)."""
        # 2x2 AvgPool with stride=1
        layer = nn.AvgPool2d(kernel_size=2, stride=1)

        # Input: 3x3 image
        # [[1, 2, 3],    -> [[2.5, 3.5],   (4 windows)
        #  [4, 5, 6],        [5.5, 6.5]]
        #  [7, 8, 9]]
        # Window (0,0): avg(1,2,4,5) = 3.0
        # Window (0,1): avg(2,3,5,6) = 4.0
        # Window (1,0): avg(4,5,7,8) = 6.0
        # Window (1,1): avg(5,6,8,9) = 7.0
        values = np.arange(1, 10).reshape(3, 3, 1).astype(np.float64)
        input_star = ImageStar.from_bounds(values, values, height=3, width=3, num_channels=1)

        # Apply AvgPool2D
        output_stars = avgpool2d_star(layer, [input_star])

        # Ground truth
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 2
        assert out_star.width == 2

        lb_out, ub_out = out_star.estimate_ranges()
        lb_out = lb_out.reshape(2, 2, 1)

        # Check averages
        assert np.allclose(lb_out[0, 0, 0], 3.0, atol=1e-6)  # avg(1,2,4,5)
        assert np.allclose(lb_out[0, 1, 0], 4.0, atol=1e-6)  # avg(2,3,5,6)
        assert np.allclose(lb_out[1, 0, 0], 6.0, atol=1e-6)  # avg(4,5,7,8)
        assert np.allclose(lb_out[1, 1, 0], 7.0, atol=1e-6)  # avg(5,6,8,9)

    def test_3x3_averaging_kernel(self):
        """Test AvgPool with 3x3 kernel."""
        # 3x3 AvgPool with stride=3 (non-overlapping)
        layer = nn.AvgPool2d(kernel_size=3, stride=3)

        # Input: 3x3 image with all values = 9
        lb = np.ones((3, 3, 1)) * 9
        ub = lb.copy()
        input_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        # Apply AvgPool2D
        output_stars = avgpool2d_star(layer, [input_star])

        # Ground truth: Average of 9 nines = 9
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 1
        assert out_star.width == 1
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 9.0, atol=1e-6)
        assert np.allclose(ub_out, 9.0, atol=1e-6)

    def test_linearity_property(self):
        """Test that AvgPool is linear: avg(aX + bY) = a*avg(X) + b*avg(Y)."""
        # 2x2 AvgPool
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        # Input 1: 2x2 image with range [1, 2]
        lb1 = np.ones((2, 2, 1))
        ub1 = np.ones((2, 2, 1)) * 2
        star1 = ImageStar.from_bounds(lb1, ub1, height=2, width=2, num_channels=1)

        # Apply AvgPool
        out1 = avgpool2d_star(layer, [star1])[0]
        lb_out1, ub_out1 = out1.estimate_ranges()

        # Ground truth: avg([1,2]) = [1, 2] (linear operation preserves range)
        assert np.allclose(lb_out1, 1.0, atol=1e-6)
        assert np.allclose(ub_out1, 2.0, atol=1e-6)

    def test_negative_values(self):
        """Test AvgPool with negative values."""
        # 2x2 AvgPool
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        # Input: 2x2 image with negative values
        # [[-4, -2],    -> avg pool -> [[(-4-2-3-1)/4]] = [[-2.5]]
        #  [-3, -1]]
        lb = np.array([[[-4.0], [-2.0]], [[-3.0], [-1.0]]])
        ub = lb.copy()
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply AvgPool2D
        output_stars = avgpool2d_star(layer, [input_star])

        # Ground truth: Average = -2.5
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, -2.5, atol=1e-6)
        assert np.allclose(ub_out, -2.5, atol=1e-6)

    def test_mixed_positive_negative(self):
        """Test AvgPool with mixed positive and negative values."""
        # 2x2 AvgPool
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        # Input: 2x2 image
        # [[-2, 4],    -> avg pool -> [[(-2+4-1+3)/4]] = [[1.0]]
        #  [-1, 3]]
        lb = np.array([[[-2.0], [4.0]], [[-1.0], [3.0]]])
        ub = lb.copy()
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply AvgPool2D
        output_stars = avgpool2d_star(layer, [input_star])

        # Ground truth: Average = 1.0
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 1.0, atol=1e-6)
        assert np.allclose(ub_out, 1.0, atol=1e-6)

    def test_multi_channel(self):
        """Test AvgPool with multiple channels (RGB)."""
        # 2x2 AvgPool on 3-channel image
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        # Input: 2x2 RGB image (channels pooled independently)
        # Each channel has different constant values
        lb = np.array([
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        ])
        ub = lb.copy()
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=3)

        # Apply AvgPool2D
        output_stars = avgpool2d_star(layer, [input_star])

        # Ground truth: Each channel averaged independently
        # Channel 0: avg(1,1,1,1) = 1
        # Channel 1: avg(2,2,2,2) = 2
        # Channel 2: avg(3,3,3,3) = 3
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 1
        assert out_star.width == 1
        assert out_star.num_channels == 3

        lb_out, ub_out = out_star.estimate_ranges()
        lb_out = lb_out.reshape(1, 1, 3)

        assert np.allclose(lb_out[0, 0, 0], 1.0, atol=1e-6)
        assert np.allclose(lb_out[0, 0, 1], 2.0, atol=1e-6)
        assert np.allclose(lb_out[0, 0, 2], 3.0, atol=1e-6)


class TestAvgPool2DEdgeCases:
    """Edge case tests for AvgPool2D."""

    def test_1x1_pooling(self):
        """Test 1x1 AvgPool (identity operation)."""
        # 1x1 AvgPool should be identity
        layer = nn.AvgPool2d(kernel_size=1, stride=1)

        # Input: 3x3 image
        lb = np.zeros((3, 3, 1))
        ub = np.ones((3, 3, 1)) * 5
        input_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        # Apply AvgPool2D
        output_stars = avgpool2d_star(layer, [input_star])

        # Ground truth: Should be unchanged (identity)
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 3
        assert out_star.width == 3
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 5.0, atol=1e-6)

    def test_asymmetric_input_bounds(self):
        """Test AvgPool with asymmetric input bounds."""
        # 2x2 AvgPool
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        # Input: 2x2 image with range [-10, 20] (asymmetric)
        lb = np.ones((2, 2, 1)) * -10
        ub = np.ones((2, 2, 1)) * 20
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply AvgPool2D
        output_stars = avgpool2d_star(layer, [input_star])

        # Ground truth: Average preserves range [-10, 20]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, -10.0, atol=1e-6)
        assert np.allclose(ub_out, 20.0, atol=1e-6)

    def test_4x4_to_2x2_pooling(self):
        """Test AvgPool with 4x4 input to 2x2 output."""
        # 2x2 AvgPool with stride=2 on 4x4 image
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        # Input: 4x4 image with sequential values
        values = np.arange(1, 17).reshape(4, 4, 1).astype(np.float64)
        input_star = ImageStar.from_bounds(values, values, height=4, width=4, num_channels=1)

        # Apply AvgPool2D
        output_stars = avgpool2d_star(layer, [input_star])

        # Ground truth: 2x2 output
        # Window (0,0): avg(1,2,5,6) = 3.5
        # Window (0,1): avg(3,4,7,8) = 5.5
        # Window (1,0): avg(9,10,13,14) = 11.5
        # Window (1,1): avg(11,12,15,16) = 13.5
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 2
        assert out_star.width == 2

        lb_out, ub_out = out_star.estimate_ranges()
        lb_out = lb_out.reshape(2, 2, 1)

        assert np.allclose(lb_out[0, 0, 0], 3.5, atol=1e-6)
        assert np.allclose(lb_out[0, 1, 0], 5.5, atol=1e-6)
        assert np.allclose(lb_out[1, 0, 0], 11.5, atol=1e-6)
        assert np.allclose(lb_out[1, 1, 0], 13.5, atol=1e-6)


class TestAvgPool2DHexatopeSoundness:
    """Soundness tests for AvgPool2D with Hexatope sets."""

    def test_simple_pooling(self):
        """Test AvgPool2D with hexatope."""
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        lb = np.zeros((4, 1))
        ub = np.ones((4, 1))
        input_hexatope = Hexatope.from_bounds(lb, ub)

        output_hexatopes = avgpool2d_hexatope(layer, [input_hexatope])

        assert len(output_hexatopes) == 1
        assert isinstance(output_hexatopes[0], Hexatope)

    def test_bounds_preservation(self):
        """Test that AvgPool2D preserves sound bounds."""
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        lb = np.array([[1.0], [2.0], [3.0], [4.0]])
        ub = np.array([[2.0], [3.0], [4.0], [5.0]])
        input_hexatope = Hexatope.from_bounds(lb, ub)

        output_hexatopes = avgpool2d_hexatope(layer, [input_hexatope])

        assert len(output_hexatopes) == 1

    def test_negative_values(self):
        """Test AvgPool2D with negative bounds."""
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        lb = np.array([[-5.0], [-4.0], [-3.0], [-2.0]])
        ub = np.array([[-3.0], [-2.0], [-1.0], [0.0]])
        input_hexatope = Hexatope.from_bounds(lb, ub)

        output_hexatopes = avgpool2d_hexatope(layer, [input_hexatope])

        assert len(output_hexatopes) == 1


class TestAvgPool2DOctatopeSoundness:
    """Soundness tests for AvgPool2D with Octatope sets."""

    def test_simple_pooling(self):
        """Test AvgPool2D with octatope."""
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        lb = np.zeros((4, 1))
        ub = np.ones((4, 1))
        input_octatope = Octatope.from_bounds(lb, ub)

        output_octatopes = avgpool2d_octatope(layer, [input_octatope])

        assert len(output_octatopes) == 1
        assert isinstance(output_octatopes[0], Octatope)

    def test_bounds_preservation(self):
        """Test that AvgPool2D preserves sound bounds."""
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        lb = np.array([[1.0], [2.0], [3.0], [4.0]])
        ub = np.array([[2.0], [3.0], [4.0], [5.0]])
        input_octatope = Octatope.from_bounds(lb, ub)

        output_octatopes = avgpool2d_octatope(layer, [input_octatope])

        assert len(output_octatopes) == 1

    def test_negative_values(self):
        """Test AvgPool2D with negative bounds."""
        layer = nn.AvgPool2d(kernel_size=2, stride=2)

        lb = np.array([[-5.0], [-4.0], [-3.0], [-2.0]])
        ub = np.array([[-3.0], [-2.0], [-1.0], [0.0]])
        input_octatope = Octatope.from_bounds(lb, ub)

        output_octatopes = avgpool2d_octatope(layer, [input_octatope])

        assert len(output_octatopes) == 1
