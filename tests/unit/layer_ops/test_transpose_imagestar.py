"""ImageStar-aware OnnxTranspose (task 2.3).

Transpose is a bijection on tensor entries -> exact row permutation of V,
constraints untouched. The only risk is layout, so entries are pinned
against torch.permute as ground truth, and flat sets reject any perm
that could reorder data (full flat support arrives with shape tracking,
task 2.4) instead of silently permuting rows.

Conventions: ImageStar stores V as (H, W, C, nVar+1) representing the
NCHW tensor (1, C, H, W); perms include the batch axis and must keep it
first.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.reach import _handle_reshape
from n2v.sets import Star
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

from onnx2torch.node_converters.transpose import OnnxTranspose

C, H, W = 2, 3, 4
N = C * H * W


def _arange_imagestar():
    """Degenerate ImageStar whose NCHW tensor is arange(N)."""
    chw = np.arange(N, dtype=np.float64).reshape(C, H, W)
    hwc = chw.transpose(1, 2, 0)
    return ImageStar.from_bounds(hwc, hwc, height=H, width=W,
                                 num_channels=C)


class TestImageStarLayoutPin:
    @pytest.mark.parametrize("perm", [
        [0, 2, 3, 1],   # NCHW -> NHWC
        [0, 3, 1, 2],   # NCHW -> NWCH
        [0, 2, 1, 3],   # swap C and H
        [0, 1, 3, 2],   # swap H and W
    ])
    def test_entries_match_torch_permute(self, perm):
        img = _arange_imagestar()
        out = reach_layer(OnnxTranspose(perm=perm), [img], 'approx')[0]
        assert isinstance(out, ImageStar)

        t = torch.arange(N, dtype=torch.float64).reshape(1, C, H, W)
        expect = t.permute(perm)[0].numpy()  # (c', h', w') in NCHW terms
        ce, he, we = expect.shape
        assert (out.num_channels, out.height, out.width) == (ce, he, we)
        for c in range(ce):
            for h in range(he):
                for w in range(we):
                    assert out.V[h, w, c, 0] == expect[c, h, w]

    def test_constraints_preserved(self):
        lb = np.zeros((H, W, C))
        ub = np.ones((H, W, C))
        img = ImageStar.from_bounds(lb, ub, height=H, width=W,
                                    num_channels=C)
        out = reach_layer(OnnxTranspose(perm=[0, 2, 3, 1]), [img],
                          'approx')[0]
        np.testing.assert_array_equal(out.C, img.C)
        np.testing.assert_array_equal(out.d, img.d)
        assert out.V.shape[-1] == img.V.shape[-1]

    def test_batch_moving_perm_raises(self):
        img = _arange_imagestar()
        with pytest.raises(NotImplementedError):
            reach_layer(OnnxTranspose(perm=[2, 0, 1, 3]), [img], 'approx')


class TestImageZonoLayoutPin:
    def test_entries_match_torch_permute(self):
        chw = np.arange(N, dtype=np.float64).reshape(C, H, W)
        hwc = chw.transpose(1, 2, 0)
        z = ImageZono.from_bounds(hwc, hwc, H, W, C)
        out = reach_layer(OnnxTranspose(perm=[0, 2, 3, 1]), [z],
                          'approx')[0]
        t = torch.arange(N, dtype=torch.float64).reshape(1, C, H, W)
        expect = t.permute([0, 2, 3, 1])[0].numpy()  # (c',h',w')
        ce, he, we = expect.shape
        got = out.c.flatten().reshape(he, we, ce).transpose(2, 0, 1)
        np.testing.assert_array_equal(got, expect)


class TestFlatSetGuards:
    """Flat sets have no tensor shape: only perms that provably leave the
    flat layout unchanged (batch axis moves only) are allowed; anything
    else raises until shape tracking (2.4) lands. This REPLACES the old
    silent row-permutation, which mis-read axis indices as row indices."""

    def test_identity_noop(self):
        star = Star.from_bounds(np.arange(3.0), np.arange(3.0) + 1)
        out = reach_layer(OnnxTranspose(perm=[0, 1, 2]), [star], 'approx')[0]
        np.testing.assert_array_equal(out.V, star.V)

    def test_batch_move_is_flat_identity(self):
        """(1, N) -> (N, 1): only the size-1 batch axis moves."""
        star = Star.from_bounds(np.arange(3.0), np.arange(3.0) + 1)
        out = reach_layer(OnnxTranspose(perm=[1, 0]), [star], 'approx')[0]
        np.testing.assert_array_equal(out.V, star.V)

    def test_data_reordering_perm_raises(self):
        star = Star.from_bounds(np.arange(3.0), np.arange(3.0) + 1)
        with pytest.raises(NotImplementedError):
            reach_layer(OnnxTranspose(perm=[0, 2, 1]), [star], 'approx')

    def test_none_perm_on_flat_raises(self):
        star = Star.from_bounds(np.arange(3.0), np.arange(3.0) + 1)
        with pytest.raises(NotImplementedError):
            reach_layer(OnnxTranspose(perm=None), [star], 'approx')


class TestPipelineSoundness:
    """Conv2d -> Transpose(NCHW->NHWC) -> Reshape(flat) -> Gemm: the
    traffic_signs shape, checked end-to-end against PyTorch."""

    def _build(self, seed=0):
        torch.manual_seed(seed)
        conv = nn.Conv2d(C, 3, kernel_size=3, padding=1)
        gemm = nn.Linear(3 * H * W, 5)

        def forward(t):  # t: (1, C, H, W)
            z = conv(t)
            z = z.permute(0, 2, 3, 1)        # NCHW -> NHWC
            z = z.reshape(1, -1)             # ONNX-style flatten
            return gemm(z)

        return conv, gemm, forward

    def _reach(self, conv, gemm, img):
        z = reach_layer(conv, [img], 'approx')
        z = reach_layer(OnnxTranspose(perm=[0, 2, 3, 1]), z, 'approx')
        z = _handle_reshape(z, (1, 3 * H * W))
        return reach_layer(gemm, z, 'approx')[0]

    def test_degenerate_exactness(self):
        conv, gemm, forward = self._build()
        chw = np.linspace(-1, 1, N).reshape(C, H, W)
        hwc = chw.transpose(1, 2, 0)
        img = ImageStar.from_bounds(hwc, hwc, height=H, width=W,
                                    num_channels=C)
        out = self._reach(conv, gemm, img)
        lo, hi = out.estimate_ranges()
        lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
        with torch.no_grad():
            y = forward(torch.tensor(chw[np.newaxis],
                                     dtype=torch.float32)).numpy().flatten()
        assert float(np.max(hi - lo)) < 1e-6
        np.testing.assert_allclose((lo + hi) / 2, y, atol=1e-5)

    def test_sampled_containment(self):
        conv, gemm, forward = self._build(seed=1)
        lb = np.full((H, W, C), -0.5)
        ub = np.full((H, W, C), 0.5)
        img = ImageStar.from_bounds(lb, ub, height=H, width=W,
                                    num_channels=C)
        out = self._reach(conv, gemm, img)
        lo, hi = out.estimate_ranges()
        lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
        rng = np.random.default_rng(3)
        for _ in range(200):
            chw = rng.uniform(-0.5, 0.5, size=(C, H, W))
            with torch.no_grad():
                y = forward(torch.tensor(chw[np.newaxis],
                                         dtype=torch.float32)
                            ).numpy().flatten()
            assert np.all(y >= lo - 1e-5) and np.all(y <= hi + 1e-5)
