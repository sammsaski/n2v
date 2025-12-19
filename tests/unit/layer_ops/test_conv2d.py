"""Tests for layer operations."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box, ImageStar, ImageZono, Hexatope, Octatope
from n2v.nn.layer_ops import (
    linear_reach, relu_reach, conv2d_reach,
    maxpool2d_reach, avgpool2d_reach, flatten_reach
)

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


