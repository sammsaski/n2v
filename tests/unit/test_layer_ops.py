"""
Tests for layer operations: Linear, ReLU, Conv2D, MaxPool2D, AvgPool2D, Flatten.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box, ImageStar, ImageZono
from n2v.nn.layer_ops import (
    linear_reach, relu_reach, conv2d_reach,
    maxpool2d_reach, avgpool2d_reach, flatten_reach
)


class TestLinearReach:
    """Tests for Linear layer reachability."""

    def test_linear_star(self, simple_star):
        """Test Linear layer with Star."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = linear_reach.linear_star(layer, [simple_star])

        assert len(result) == 1
        assert result[0].dim == 2
        pytest.assert_star_valid(result[0])

    def test_linear_zono(self, simple_zono):
        """Test Linear layer with Zono."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = linear_reach.linear_zono(layer, [simple_zono])

        assert len(result) == 1
        assert result[0].dim == 2
        pytest.assert_zono_valid(result[0])

    def test_linear_box(self, simple_box):
        """Test Linear layer with Box."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = linear_reach.linear_box(layer, [simple_box])

        assert len(result) == 1
        assert result[0].dim == 2

    def test_linear_preserves_exactness(self):
        """Test that Linear is exact (no over-approximation)."""
        layer = nn.Linear(2, 2)
        # Set to identity + constant
        with torch.no_grad():
            layer.weight.data = torch.eye(2)
            layer.bias.data = torch.ones(2)

        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        result = linear_reach.linear_star(layer, [star])[0]

        # Expected: [0, 1] + 1 = [1, 2]
        result.estimate_ranges()
        np.testing.assert_allclose(result.state_lb, np.array([[1.0], [1.0]]), atol=1e-5)
        np.testing.assert_allclose(result.state_ub, np.array([[2.0], [2.0]]), atol=1e-5)


class TestReLUReach:
    """Tests for ReLU activation reachability."""

    def test_relu_star_exact_always_active(self):
        """Test ReLU with always active neuron."""
        # Star with all positive bounds
        lb = np.array([[1.0], [2.0]])
        ub = np.array([[2.0], [3.0]])
        star = Star.from_bounds(lb, ub)

        result = relu_reach.relu_star_exact([star])

        # Should not split
        assert len(result) == 1
        pytest.assert_star_valid(result[0])

    def test_relu_star_exact_always_inactive(self):
        """Test ReLU with always inactive neuron."""
        # Star with all negative bounds
        lb = np.array([[-2.0], [-3.0]])
        ub = np.array([[-1.0], [-2.0]])
        star = Star.from_bounds(lb, ub)

        result = relu_reach.relu_star_exact([star])

        # Should not split, but zero out
        assert len(result) == 1
        result[0].estimate_ranges()
        np.testing.assert_allclose(result[0].state_lb, np.zeros((2, 1)), atol=1e-6)
        np.testing.assert_allclose(result[0].state_ub, np.zeros((2, 1)), atol=1e-6)

    def test_relu_star_exact_splitting(self):
        """Test ReLU with uncertain neuron (causes splitting)."""
        # Star with crossing zero
        lb = np.array([[-1.0], [1.0]])
        ub = np.array([[1.0], [2.0]])
        star = Star.from_bounds(lb, ub)

        result = relu_reach.relu_star_exact([star])

        # Should split on first neuron
        assert len(result) == 2  # One for active, one for inactive
        for s in result:
            pytest.assert_star_valid(s)

    def test_relu_zono_approx(self, simple_zono):
        """Test ReLU with Zonotope (over-approximation)."""
        result = relu_reach.relu_zono_approx([simple_zono])

        assert len(result) == 1
        pytest.assert_zono_valid(result[0])

    def test_relu_box(self):
        """Test ReLU with Box."""
        lb = np.array([[-1.0], [-2.0], [1.0]])
        ub = np.array([[1.0], [0.5], [2.0]])
        box = Box(lb, ub)

        result = relu_reach.relu_box([box])

        assert len(result) == 1
        # Check ReLU applied correctly (Box has .lb and .ub attributes)
        result_lb = result[0].lb
        result_ub = result[0].ub
        np.testing.assert_allclose(result_lb, np.array([[0.0], [0.0], [1.0]]), atol=1e-6)
        np.testing.assert_allclose(result_ub, np.array([[1.0], [0.5], [2.0]]), atol=1e-6)


class TestConv2DReach:
    """Tests for Conv2D layer reachability."""

    def test_conv2d_star(self, simple_image_star):
        """Test Conv2D with ImageStar."""
        layer = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        layer.eval()

        result = conv2d_reach.conv2d_star(layer, [simple_image_star])

        assert len(result) == 1
        assert result[0].num_channels == 2
        assert result[0].height == 4  # Same with padding
        assert result[0].width == 4
        pytest.assert_image_star_valid(result[0])

    def test_conv2d_star_exact(self):
        """Test that Conv2D is exact (linear operation)."""
        layer = nn.Conv2d(1, 1, kernel_size=1)
        # Set to identity
        with torch.no_grad():
            layer.weight.data = torch.ones(1, 1, 1, 1)
            layer.bias.data = torch.zeros(1)

        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        result = conv2d_reach.conv2d_star(layer, [img_star])[0]

        # Should preserve bounds (identity conv)
        result.estimate_ranges()
        assert np.all(result.state_lb >= -0.01)
        assert np.all(result.state_ub <= 1.01)

    def test_conv2d_strided(self, simple_image_star):
        """Test Conv2D with stride."""
        layer = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)
        layer.eval()

        result = conv2d_reach.conv2d_star(layer, [simple_image_star])

        assert len(result) == 1
        assert result[0].height == 2  # 4 -> 2 with stride 2
        assert result[0].width == 2

    def test_conv2d_zono(self, simple_image_zono):
        """Test Conv2D with ImageZono."""
        layer = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        layer.eval()

        result = conv2d_reach.conv2d_zono(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].num_channels == 2


class TestMaxPool2DReach:
    """Tests for MaxPool2D layer reachability."""

    def test_maxpool2d_star_exact(self, simple_image_star):
        """Test MaxPool2D exact method."""
        layer = nn.MaxPool2d(2, 2)

        result = maxpool2d_reach.maxpool2d_star(
            layer, [simple_image_star],
            method='exact'
        )

        assert len(result) >= 1  # May split
        assert result[0].height == 2  # 4 -> 2
        assert result[0].width == 2
        for star in result:
            pytest.assert_image_star_valid(star)

    def test_maxpool2d_star_approx(self, simple_image_star):
        """Test MaxPool2D approximate method."""
        layer = nn.MaxPool2d(2, 2)

        result = maxpool2d_reach.maxpool2d_star(
            layer, [simple_image_star],
            method='approx'
        )

        # Approx doesn't split
        assert len(result) == 1
        assert result[0].height == 2
        assert result[0].width == 2

    def test_maxpool2d_zono(self, simple_image_zono):
        """Test MaxPool2D with ImageZono."""
        layer = nn.MaxPool2d(2, 2)

        result = maxpool2d_reach.maxpool2d_zono(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].height == 2
        assert result[0].width == 2


class TestAvgPool2DReach:
    """Tests for AvgPool2D layer reachability."""

    def test_avgpool2d_star(self, simple_image_star):
        """Test AvgPool2D with ImageStar."""
        layer = nn.AvgPool2d(2, 2)

        result = avgpool2d_reach.avgpool2d_star(layer, [simple_image_star])

        # AvgPool never splits (linear operation)
        assert len(result) == 1
        assert result[0].height == 2
        assert result[0].width == 2
        pytest.assert_image_star_valid(result[0])

    def test_avgpool2d_no_splitting(self):
        """Test that AvgPool2D never splits stars (key property)."""
        lb = np.zeros((8, 8, 1))
        ub = np.ones((8, 8, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=8, width=8, num_channels=1)

        layer = nn.AvgPool2d(2, 2)

        result = avgpool2d_reach.avgpool2d_star(layer, [img_star])

        # Should NEVER split (this is the key advantage!)
        assert len(result) == 1

    def test_avgpool2d_exact(self):
        """Test that AvgPool2D is exact."""
        # Create known image
        image = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(image, image, height=4, width=4, num_channels=1)

        layer = nn.AvgPool2d(2, 2)

        result = avgpool2d_reach.avgpool2d_star(layer, [img_star])[0]

        # Average of 1s should be 1
        result.estimate_ranges()
        np.testing.assert_allclose(result.state_lb, np.ones((4, 1)), atol=1e-6)
        np.testing.assert_allclose(result.state_ub, np.ones((4, 1)), atol=1e-6)

    def test_avgpool2d_zono(self, simple_image_zono):
        """Test AvgPool2D with ImageZono."""
        layer = nn.AvgPool2d(2, 2)

        result = avgpool2d_reach.avgpool2d_zono(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].height == 2
        assert result[0].width == 2


class TestFlattenReach:
    """Tests for Flatten layer reachability."""

    def test_flatten_star(self, simple_image_star):
        """Test Flatten with ImageStar."""
        layer = nn.Flatten()

        result = flatten_reach.flatten_star(layer, [simple_image_star])

        assert len(result) == 1
        assert isinstance(result[0], Star)
        assert result[0].dim == 16  # 4*4*1
        pytest.assert_star_valid(result[0])

    def test_flatten_zono(self, simple_image_zono):
        """Test Flatten with ImageZono."""
        layer = nn.Flatten()

        result = flatten_reach.flatten_zono(layer, [simple_image_zono])

        assert len(result) == 1
        assert isinstance(result[0], Zono)
        assert result[0].dim == 16


class TestLayerCombinations:
    """Test combinations of layers."""

    # def test_conv_relu_sequence(self, simple_image_star):
    #     """Test Conv2D followed by ReLU."""
    #     conv = nn.Conv2d(1, 2, 3, padding=1)
    #     conv.eval()

    #     # Conv2D
    #     after_conv = conv2d_reach.conv2d_star(conv, [simple_image_star])
    #     assert len(after_conv) == 1

    #     # ReLU (may split)
    #     after_relu = relu_reach.relu_star_exact(after_conv)
    #     assert len(after_relu) >= 1

    def test_conv_avgpool_flatten(self, simple_image_star):
        """Test typical CNN sequence."""
        conv = nn.Conv2d(1, 2, 3, padding=1)
        avgpool = nn.AvgPool2d(2, 2)
        flatten = nn.Flatten()

        conv.eval()

        # Conv2D
        after_conv = conv2d_reach.conv2d_star(conv, [simple_image_star])

        # AvgPool
        after_pool = avgpool2d_reach.avgpool2d_star(avgpool, after_conv)
        assert len(after_pool) == len(after_conv)  # No splitting!

        # Flatten
        after_flatten = flatten_reach.flatten_star(flatten, after_pool)
        assert isinstance(after_flatten[0], Star)
