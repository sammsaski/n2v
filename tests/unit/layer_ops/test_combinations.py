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

