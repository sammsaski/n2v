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


