"""Tests for GlobalAvgPool layer reachability."""

import pytest
import numpy as np
import torch.nn as nn
from n2v.sets import ImageStar, ImageZono
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestGlobalAvgPoolReach:
    """Tests for GlobalAvgPool layer reachability."""

    def test_global_avgpool_star_identity_bounds(self):
        """GlobalAvgPool on uniform [0,1] ImageStar should give [0,1] per channel."""
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        layer = nn.AdaptiveAvgPool2d(1)

        result = reach_layer(layer, [img_star], method='exact')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, ImageStar)
        assert out.height == 1
        assert out.width == 1
        assert out.num_channels == 1

    def test_global_avgpool_star_known_bounds(self):
        """GlobalAvgPool on [2,4] should give bounds containing [2,4]."""
        lb = np.ones((3, 3, 2)) * 2
        ub = np.ones((3, 3, 2)) * 4
        img_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        layer = nn.AdaptiveAvgPool2d(1)

        result = reach_layer(layer, [img_star], method='exact')

        assert len(result) == 1
        out = result[0]
        assert out.height == 1
        assert out.width == 1
        assert out.num_channels == 2

        lb_out, ub_out = out.estimate_ranges()
        lb_out = lb_out.flatten()
        ub_out = ub_out.flatten()
        assert np.all(lb_out >= 2.0 - 1e-6)
        assert np.all(ub_out <= 4.0 + 1e-6)

    def test_global_avgpool_zono(self):
        """GlobalAvgPool on ImageZono."""
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_zono = ImageZono.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        layer = nn.AdaptiveAvgPool2d(1)

        result = reach_layer(layer, [img_zono], method='approx')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, ImageZono)
        assert out.height == 1
        assert out.width == 1
        assert out.num_channels == 1

    def test_global_avgpool_multichannel(self):
        """GlobalAvgPool preserves channel count."""
        lb = np.zeros((3, 3, 4))
        ub = np.ones((3, 3, 4))
        img_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=4)

        layer = nn.AdaptiveAvgPool2d(1)

        result = reach_layer(layer, [img_star], method='exact')

        assert len(result) == 1
        assert result[0].num_channels == 4
        assert result[0].height == 1
        assert result[0].width == 1
