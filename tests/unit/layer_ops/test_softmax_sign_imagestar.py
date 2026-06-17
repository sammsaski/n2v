"""Task 2.6 pins: Softmax interval relaxation, Sign on ImageStar
(I-36), ImageStar constant add/sub, and the NCHW->HWC ordering of
full-size constants (traffic_signs failure chain).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Star, Box
from n2v.sets.image_star import ImageStar
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.layer_ops import sign_reach, softmax_reach
from n2v.nn.reach import _mul_sets_by_constant


def _np_softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class TestSoftmaxStar:
    def test_degenerate_is_exact(self):
        x = np.array([1.0, -2.0, 0.5, 3.0])
        s = Star.from_bounds(x, x)
        out = reach_layer(nn.Softmax(dim=-1), [s], 'approx')[0]
        lo, hi = out.get_ranges()
        np.testing.assert_allclose(np.asarray(lo).flatten(),
                                   _np_softmax(x), atol=1e-9)
        np.testing.assert_allclose(np.asarray(hi).flatten(),
                                   _np_softmax(x), atol=1e-9)

    def test_containment_over_box(self):
        center = np.array([0.5, -1.0, 2.0])
        lb, ub = center - 0.3, center + 0.3
        s = Star.from_bounds(lb, ub)
        out = reach_layer(nn.Softmax(dim=1), [s], 'approx')[0]
        lo, hi = out.get_ranges()
        lo = np.asarray(lo).flatten()
        hi = np.asarray(hi).flatten()
        rng = np.random.default_rng(3)
        for _ in range(300):
            y = _np_softmax(rng.uniform(lb, ub))
            assert np.all(y >= lo - 1e-9) and np.all(y <= hi + 1e-9)

    def test_inner_axis_raises(self):
        s = Star.from_bounds(np.zeros(4), np.ones(4))
        with pytest.raises(NotImplementedError, match="axis"):
            reach_layer(nn.Softmax(dim=2), [s], 'approx')

    def test_box_path(self):
        lb = np.array([0.0, 1.0])
        ub = np.array([0.5, 1.5])
        out = softmax_reach.softmax_box(nn.Softmax(dim=-1),
                                        [Box(lb, ub)])[0]
        rng = np.random.default_rng(4)
        for _ in range(200):
            y = _np_softmax(rng.uniform(lb, ub))
            assert np.all(y >= out.lb.flatten() - 1e-9)
            assert np.all(y <= out.ub.flatten() + 1e-9)


class TestSignImageStar:
    """I-36: Sign used to call the flat get_range(i) signature on
    ImageStar and crashed; it now flattens, applies, and restores the
    spatial type."""

    def test_imagestar_roundtrip(self):
        H = W = 2
        C = 1
        vals = np.array([-1.5, -0.2, 0.3, 2.0]).reshape(-1, 1)
        s = ImageStar.from_bounds(vals, vals, height=H, width=W,
                                  num_channels=C)
        out = sign_reach.sign_star(None, [s], method='approx',
                                   lp_solver='linprog')[0]
        assert isinstance(out, ImageStar)
        lo, hi = out.get_ranges()
        expected = np.sign(vals.flatten())
        np.testing.assert_allclose(np.asarray(lo).flatten(), expected,
                                   atol=1e-9)
        np.testing.assert_allclose(np.asarray(hi).flatten(), expected,
                                   atol=1e-9)


class TestImageStarConstantOps:
    """Full-size ONNX constants are (C, H, W) flat and must be permuted
    to the HWC storage order — both for mul-by-constant and the add/sub
    translation (the cifar100 bug class, model-internal edition)."""

    def _point_imagestar(self, vec_chw, C, H, W):
        img = vec_chw.reshape(C, H, W).transpose(1, 2, 0).reshape(-1, 1)
        return ImageStar.from_bounds(img, img, height=H, width=W,
                                     num_channels=C)

    def test_mul_full_size_constant_is_nchw(self):
        C, H, W = 2, 2, 3
        x = np.arange(C * H * W, dtype=np.float64) + 1.0
        scale = np.linspace(0.5, 2.0, C * H * W)  # NCHW flat
        s = self._point_imagestar(x, C, H, W)
        out = _mul_sets_by_constant([s], scale)[0]
        lo, _ = out.get_ranges()
        expected = (x * scale).reshape(C, H, W).transpose(1, 2, 0).flatten()
        np.testing.assert_allclose(np.asarray(lo).flatten(), expected,
                                   atol=1e-9)

    def test_channelwise_scale(self):
        C, H, W = 3, 2, 2
        x = np.ones(C * H * W)
        scale = np.array([1.0, 2.0, 3.0])
        s = self._point_imagestar(x, C, H, W)
        out = _mul_sets_by_constant([s], scale)[0]
        lo, _ = out.get_ranges()
        expected = (x.reshape(C, H, W) * scale.reshape(C, 1, 1)) \
            .transpose(1, 2, 0).flatten()
        np.testing.assert_allclose(np.asarray(lo).flatten(), expected,
                                   atol=1e-9)
