"""Tests for Pad layer reachability."""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import ImageStar, ImageZono
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestZeroPad2dStar:
    """Test ZeroPad2d reachability with ImageStar."""

    def test_zeropad_imagestar_dimensions(self):
        """ZeroPad2d should expand spatial dimensions."""
        lb = np.zeros((3, 3, 2))
        ub = np.ones((3, 3, 2))
        img_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        pad = nn.ZeroPad2d((1, 1, 1, 1))  # left, right, top, bottom

        result = reach_layer(pad, [img_star], method='exact')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, ImageStar)
        assert out.height == 5  # 3 + 1 + 1
        assert out.width == 5
        assert out.num_channels == 2

    def test_zeropad_bounds_preserved(self):
        """Original content region should have unchanged bounds; padded region should be zero."""
        lb = np.ones((2, 2, 1)) * 2.0
        ub = np.ones((2, 2, 1)) * 4.0
        img_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        pad = nn.ZeroPad2d((1, 1, 1, 1))

        result = reach_layer(pad, [img_star], method='exact')
        out = result[0]
        lb_out, ub_out = out.estimate_ranges()

        # Reshape to (H, W, C) for checking
        lb_img = lb_out.reshape(4, 4, 1)
        ub_img = ub_out.reshape(4, 4, 1)

        # Padded border should be [0, 0]
        assert np.allclose(lb_img[0, :, :], 0.0)
        assert np.allclose(ub_img[0, :, :], 0.0)
        assert np.allclose(lb_img[-1, :, :], 0.0)
        assert np.allclose(ub_img[-1, :, :], 0.0)

        # Interior should be [2, 4]
        assert np.all(lb_img[1:3, 1:3, :] >= 2.0 - 1e-6)
        assert np.all(ub_img[1:3, 1:3, :] <= 4.0 + 1e-6)

    def test_asymmetric_padding(self):
        """Asymmetric padding (different left/right/top/bottom)."""
        lb = np.zeros((2, 3, 1))
        ub = np.ones((2, 3, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=2, width=3, num_channels=1)

        pad = nn.ZeroPad2d((0, 2, 1, 0))  # left=0, right=2, top=1, bottom=0

        result = reach_layer(pad, [img_star], method='exact')
        out = result[0]
        assert out.height == 3   # 2 + 1 + 0
        assert out.width == 5    # 3 + 0 + 2


class TestZeroPad2dZono:
    """Test ZeroPad2d reachability with ImageZono."""

    def test_zeropad_imagezono(self):
        """ZeroPad2d should work with ImageZono."""
        lb = np.zeros((3, 3, 1))
        ub = np.ones((3, 3, 1))
        img_zono = ImageZono.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        pad = nn.ZeroPad2d((1, 1, 1, 1))

        result = reach_layer(pad, [img_zono], method='approx')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, ImageZono)
        assert out.height == 5
        assert out.width == 5

    def test_zeropad_imagezono_bounds(self):
        """Padded region should have zero bounds in ImageZono."""
        lb = np.ones((2, 2, 1))
        ub = np.ones((2, 2, 1)) * 3.0
        img_zono = ImageZono.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        pad = nn.ZeroPad2d(1)  # pad=1 on all sides

        result = reach_layer(pad, [img_zono], method='approx')
        out = result[0]
        lb_out, ub_out = out.get_bounds()

        # First row (padding) should be zero
        lb_img = lb_out.reshape(4, 4, 1)
        ub_img = ub_out.reshape(4, 4, 1)
        assert np.allclose(lb_img[0, :, :], 0.0)
        assert np.allclose(ub_img[0, :, :], 0.0)
