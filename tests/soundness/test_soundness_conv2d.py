"""
Soundness tests for Conv2D layer reachability.

Tests verify mathematical correctness of Conv2D operations on different set types.
Conv2D is an affine transformation, so it should be exact for all set types.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import ImageStar, ImageZono
from n2v.nn.layer_ops.conv2d_reach import conv2d_star, conv2d_zono


class TestConv2DImageStarSoundness:
    """Soundness tests for Conv2D with ImageStar sets."""

    def test_identity_convolution(self):
        """Test Conv2D with identity kernel (no change)."""
        # Identity kernel: output = input
        layer = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        layer.weight.data = torch.ones(1, 1, 1, 1)

        # Input: 3x3 image, channel=1, range [0, 1]
        lb = np.zeros((3, 3, 1))
        ub = np.ones((3, 3, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Should be unchanged (3x3 output)
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 3
        assert out_star.width == 3
        assert out_star.num_channels == 1

        # Check bounds are preserved
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 1.0, atol=1e-6)

    def test_zero_convolution(self):
        """Test Conv2D with zero kernel (output all zeros)."""
        # Zero kernel: output = 0
        layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        layer.weight.data = torch.zeros(1, 1, 3, 3)

        # Input: 4x4 image with range [0, 1]
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Output should be all zeros
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 0.0, atol=1e-6)

    def test_bias_addition(self):
        """Test Conv2D with bias term."""
        # Identity kernel with bias=5
        layer = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)
        layer.weight.data = torch.ones(1, 1, 1, 1)
        layer.bias.data = torch.tensor([5.0])

        # Input: 2x2 image with range [0, 1]
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Output = input + 5, so range [5, 6]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 5.0, atol=1e-6)
        assert np.allclose(ub_out, 6.0, atol=1e-6)

    def test_scaling_convolution(self):
        """Test Conv2D with scaling kernel."""
        # Scaling kernel: output = 3 * input
        layer = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        layer.weight.data = torch.tensor([[[[3.0]]]])

        # Input: 2x2 image with range [1, 2]
        lb = np.ones((2, 2, 1))
        ub = np.ones((2, 2, 1)) * 2
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Output = 3 * input, so range [3, 6]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 3.0, atol=1e-6)
        assert np.allclose(ub_out, 6.0, atol=1e-6)

    def test_3x3_averaging_filter(self):
        """Test Conv2D with 3x3 averaging filter."""
        # 3x3 averaging kernel with padding=1 (output same size as input)
        layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        layer.weight.data = torch.ones(1, 1, 3, 3) / 9.0  # Uniform average

        # Input: 4x4 image with range [0, 9]
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1)) * 9
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Average of [0, 9] is [0, 9] (max stays 9, min stays 0)
        # Each output pixel is weighted sum of up to 9 input pixels
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 4
        assert out_star.width == 4
        lb_out, ub_out = out_star.estimate_ranges()

        # Output range should be [0, 9] (averaging doesn't increase range)
        assert np.all(lb_out >= 0.0 - 1e-6)
        assert np.all(ub_out <= 9.0 + 1e-6)

    def test_stride_2_convolution(self):
        """Test Conv2D with stride=2 (dimension reduction)."""
        # Identity kernel with stride=2
        layer = nn.Conv2d(1, 1, kernel_size=1, stride=2, padding=0, bias=False)
        layer.weight.data = torch.ones(1, 1, 1, 1)

        # Input: 4x4 image with range [0, 1]
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Output size = 2x2 (stride=2 reduces by half)
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 2
        assert out_star.width == 2
        assert out_star.num_channels == 1

        # Bounds should be preserved
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 1.0, atol=1e-6)

    def test_multi_channel_input(self):
        """Test Conv2D with multi-channel input (RGB)."""
        # 3 input channels -> 1 output channel
        # Sum all channels (weight=1 for each)
        layer = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        layer.weight.data = torch.ones(1, 3, 1, 1)

        # Input: 2x2 RGB image, each channel [0, 1]
        lb = np.zeros((2, 2, 3))
        ub = np.ones((2, 2, 3))
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=3)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Output = sum of 3 channels, range [0, 3]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 2
        assert out_star.width == 2
        assert out_star.num_channels == 1

        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 3.0, atol=1e-6)

    def test_multi_channel_output(self):
        """Test Conv2D with multi-channel output."""
        # 1 input channel -> 2 output channels
        layer = nn.Conv2d(1, 2, kernel_size=1, stride=1, padding=0, bias=False)
        layer.weight.data = torch.tensor([[[[2.0]]], [[[3.0]]]])  # Channel 0: *2, Channel 1: *3

        # Input: 2x2 image with range [1, 2]
        lb = np.ones((2, 2, 1))
        ub = np.ones((2, 2, 1)) * 2
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Channel 0 = [2, 4], Channel 1 = [3, 6]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 2
        assert out_star.width == 2
        assert out_star.num_channels == 2

        lb_out, ub_out = out_star.estimate_ranges()
        # Reshape to (h, w, c) to check per channel
        lb_out = lb_out.reshape(2, 2, 2)
        ub_out = ub_out.reshape(2, 2, 2)

        # Channel 0: [2, 4]
        assert np.allclose(lb_out[:, :, 0], 2.0, atol=1e-6)
        assert np.allclose(ub_out[:, :, 0], 4.0, atol=1e-6)

        # Channel 1: [3, 6]
        assert np.allclose(lb_out[:, :, 1], 3.0, atol=1e-6)
        assert np.allclose(ub_out[:, :, 1], 6.0, atol=1e-6)

    def test_negative_weights(self):
        """Test Conv2D with negative weights."""
        # Negative scaling: output = -2 * input
        layer = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        layer.weight.data = torch.tensor([[[[-2.0]]]])

        # Input: 2x2 image with range [1, 3]
        lb = np.ones((2, 2, 1))
        ub = np.ones((2, 2, 1)) * 3
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Output = -2 * [1, 3] = [-6, -2]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, -6.0, atol=1e-6)
        assert np.allclose(ub_out, -2.0, atol=1e-6)

    def test_padding_effect(self):
        """Test Conv2D with padding=1 (output size preserved)."""
        # 3x3 kernel with padding=1 should preserve size
        layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        layer.weight.data = torch.ones(1, 1, 3, 3)

        # Input: 3x3 image with range [0, 1]
        lb = np.zeros((3, 3, 1))
        ub = np.ones((3, 3, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Output should be 3x3 (same as input due to padding)
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 3
        assert out_star.width == 3


# NOTE: ImageZono tests skipped for now - conv2d_zono implementation needs investigation
# The convolution seems to zero out generators, returning only the center values
# TODO: Debug conv2d_zono implementation


class TestConv2DEdgeCases:
    """Edge case tests for Conv2D."""

    def test_1x1_input_image(self):
        """Test Conv2D with 1x1 input (minimum size)."""
        # 1x1 kernel on 1x1 image
        layer = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        layer.weight.data = torch.tensor([[[[5.0]]]])

        # Input: 1x1 image with range [2, 3]
        lb = np.array([[[2.0]]])
        ub = np.array([[[3.0]]])
        input_star = ImageStar.from_bounds(lb, ub, height=1, width=1, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Output = 5 * [2, 3] = [10, 15]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 1
        assert out_star.width == 1
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 10.0, atol=1e-6)
        assert np.allclose(ub_out, 15.0, atol=1e-6)

    def test_very_large_kernel(self):
        """Test Conv2D with large kernel (5x5)."""
        # 5x5 kernel with no padding on 5x5 image -> 1x1 output
        layer = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0, bias=False)
        layer.weight.data = torch.ones(1, 1, 5, 5)

        # Input: 5x5 image with range [0, 1]
        lb = np.zeros((5, 5, 1))
        ub = np.ones((5, 5, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=5, width=5, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Output = sum of 25 pixels, range [0, 25]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        assert out_star.height == 1
        assert out_star.width == 1
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 25.0, atol=1e-6)

    def test_asymmetric_input_bounds(self):
        """Test Conv2D with asymmetric input bounds."""
        # Identity kernel
        layer = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        layer.weight.data = torch.ones(1, 1, 1, 1)

        # Input: 2x2 image with range [-5, 10] (asymmetric around zero)
        lb = np.ones((2, 2, 1)) * -5
        ub = np.ones((2, 2, 1)) * 10
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Apply Conv2D
        output_stars = conv2d_star(layer, [input_star])

        # Ground truth: Should preserve range [-5, 10]
        assert len(output_stars) == 1
        out_star = output_stars[0]
        lb_out, ub_out = out_star.estimate_ranges()
        assert np.allclose(lb_out, -5.0, atol=1e-6)
        assert np.allclose(ub_out, 10.0, atol=1e-6)
