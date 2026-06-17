"""ConvTranspose2D reachability: exact affine map on ImageStar/ImageZono.

Pins: degenerate (point) ImageStar through conv_transpose2d reproduces
torch's output exactly (in HWC order); sampled containment over a box;
stride/padding/output_padding variants; loud raise on flat Star input.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Star, ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops.dispatcher import reach_layer


def _point_imagestar(vec_chw, C, H, W):
    img = vec_chw.reshape(C, H, W).transpose(1, 2, 0)
    flat = img.reshape(-1, 1)
    return ImageStar.from_bounds(flat, flat, height=H, width=W,
                                 num_channels=C)


def _torch_hwc(layer, vec_chw, C, H, W):
    with torch.no_grad():
        y = layer(torch.tensor(vec_chw, dtype=torch.float32)
                  .reshape(1, C, H, W)).numpy()[0]
    return y.transpose(1, 2, 0).flatten()


class TestConvTranspose2dStar:
    @pytest.mark.parametrize("kwargs", [
        dict(kernel_size=3),
        dict(kernel_size=4, stride=2, padding=1),
        dict(kernel_size=3, stride=2, output_padding=1),
        dict(kernel_size=3, stride=1, padding=1, dilation=2),
    ])
    def test_degenerate_matches_torch(self, kwargs):
        torch.manual_seed(0)
        C, H, W = 3, 5, 5
        layer = nn.ConvTranspose2d(C, 2, **kwargs)
        vec = np.arange(C * H * W, dtype=np.float64) / 10.0

        out = reach_layer(layer, [_point_imagestar(vec, C, H, W)],
                          'approx')[0]
        lo, hi = out.get_ranges()
        lo = np.asarray(lo).flatten()
        hi = np.asarray(hi).flatten()
        expected = _torch_hwc(layer, vec, C, H, W)
        np.testing.assert_allclose(lo, expected, atol=1e-5)
        np.testing.assert_allclose(hi, expected, atol=1e-5)

    def test_containment_over_box(self):
        torch.manual_seed(1)
        C, H, W = 2, 4, 4
        layer = nn.ConvTranspose2d(C, 3, 4, stride=2, padding=1)
        center = np.arange(C * H * W, dtype=np.float64) / 8.0
        lb, ub = center - 0.1, center + 0.1

        img_lb = lb.reshape(C, H, W).transpose(1, 2, 0).reshape(-1, 1)
        img_ub = ub.reshape(C, H, W).transpose(1, 2, 0).reshape(-1, 1)
        s = ImageStar.from_bounds(img_lb, img_ub, height=H, width=W,
                                  num_channels=C)
        out = reach_layer(layer, [s], 'approx')[0]
        lo, hi = out.get_ranges()
        lo = np.asarray(lo).flatten()
        hi = np.asarray(hi).flatten()

        rng = np.random.default_rng(5)
        for _ in range(100):
            x = rng.uniform(lb, ub)
            y = _torch_hwc(layer, x, C, H, W)
            assert np.all(y >= lo - 1e-5) and np.all(y <= hi + 1e-5)

    def test_flat_star_raises(self):
        layer = nn.ConvTranspose2d(1, 1, 3)
        s = Star.from_bounds(np.zeros(9), np.ones(9))
        with pytest.raises(ValueError, match="requires ImageStar"):
            reach_layer(layer, [s], 'approx')


class TestConvTranspose2dZono:
    def test_degenerate_matches_torch(self):
        torch.manual_seed(2)
        C, H, W = 2, 3, 3
        layer = nn.ConvTranspose2d(C, 2, 3, stride=2)
        vec = np.arange(C * H * W, dtype=np.float64) / 5.0
        img = vec.reshape(C, H, W).transpose(1, 2, 0).reshape(-1, 1)
        z = ImageZono(img, np.zeros((C * H * W, 0)), H, W, C)
        out = reach_layer(layer, [z], 'zono')[0]
        lo, hi = out.get_ranges()
        expected = _torch_hwc(layer, vec, C, H, W)
        np.testing.assert_allclose(np.asarray(lo).flatten(), expected,
                                   atol=1e-5)
        np.testing.assert_allclose(np.asarray(hi).flatten(), expected,
                                   atol=1e-5)
