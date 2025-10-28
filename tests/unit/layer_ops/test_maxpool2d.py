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


