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


