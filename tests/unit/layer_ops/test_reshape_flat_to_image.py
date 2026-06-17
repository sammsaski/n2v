"""Flat-Star -> ImageStar Reshape (task 2.2).

A Reshape is a bijection on tensor entries: x = c + V·alpha implies
reshape(x) = reshape(c) + reshape(V)·alpha, so permuting V's rows is
EXACT — no approximation. The only possible defect is a layout error,
so these tests pin the permutation at the index level, the roundtrip
identity against the existing flatten branch, and end-to-end semantics
against PyTorch through a Gemm->Reshape->Conv2d->ReLU pipeline.

Convention under test: flat-Star rows are in ONNX CHW-flattened order;
ImageStar stores V as (H, W, C, nVar+1).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import _handle_reshape
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Star
from n2v.sets.image_star import ImageStar

C, H, W = 2, 3, 4
N = C * H * W


def _arange_star():
    """Degenerate star (point) whose entries are 0..N-1 in flat order."""
    vals = np.arange(N, dtype=np.float64)
    return Star.from_bounds(vals, vals)


class TestLayoutPin:
    def test_index_level_layout(self):
        """ImageStar center at (h, w, c) must equal flat entry
        c*H*W + h*W + w (ONNX CHW order)."""
        star = _arange_star()
        out = _handle_reshape([star], (1, C, H, W))[0]  # leading batch dim
        assert isinstance(out, ImageStar)
        assert (out.height, out.width, out.num_channels) == (H, W, C)
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    assert out.V[h, w, c, 0] == c * H * W + h * W + w

    def test_roundtrip_identity(self):
        """flat -> image -> flat must return a bit-identical star
        (the new branch is the exact inverse of the flatten branch)."""
        rng = np.random.default_rng(0)
        lb = rng.normal(size=N)
        star = Star.from_bounds(lb, lb + 1.0)
        img = _handle_reshape([star], (1, C, H, W))[0]
        back = _handle_reshape([img], (1, N))[0]
        assert isinstance(back, Star) and not isinstance(back, ImageStar)
        np.testing.assert_array_equal(back.V, star.V)
        np.testing.assert_array_equal(back.C, star.C)

    def test_minus_one_target(self):
        star = _arange_star()
        out = _handle_reshape([star], (-1, C, H, W))[0]
        assert isinstance(out, ImageStar)

    def test_flat_target_stays_flat(self):
        star = _arange_star()
        out = _handle_reshape([star], (1, N))[0]
        assert isinstance(out, Star) and not isinstance(out, ImageStar)

    def test_size_mismatch_raises(self):
        star = _arange_star()
        with pytest.raises(ValueError):
            _handle_reshape([star], (1, C, H, W + 1))


class TestPipelineSoundness:
    """Gemm -> Reshape -> Conv2d -> ReLU: the soundnessbench shape."""

    def _build(self, seed=0):
        torch.manual_seed(seed)
        gemm = nn.Linear(5, N)
        conv = nn.Conv2d(C, 3, kernel_size=3, padding=1)
        relu = nn.ReLU()

        def forward(t):
            z = gemm(t)
            z = z.reshape(-1, C, H, W)
            return relu(conv(z))

        return gemm, conv, relu, forward

    def _reach(self, gemm, conv, relu, star):
        z = reach_layer(gemm, [star], 'approx')
        z = _handle_reshape(z, (-1, C, H, W))
        z = reach_layer(conv, z, 'approx')
        return reach_layer(relu, z, 'approx')[0]

    def test_degenerate_input_exactness(self):
        """Point input -> reach must collapse to torch's forward output
        (torch reshape uses the same NCHW semantics as ONNX)."""
        gemm, conv, relu, forward = self._build()
        x = np.linspace(-1.0, 1.0, 5)
        star = Star.from_bounds(x, x)
        out = self._reach(gemm, conv, relu, star)
        lo, hi = out.estimate_ranges()
        lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
        with torch.no_grad():
            y = forward(torch.tensor(x, dtype=torch.float32)
                        .unsqueeze(0)).numpy()
        y_hwc = y[0].transpose(1, 2, 0).flatten()  # reach sets are HWC
        assert float(np.max(hi - lo)) < 1e-6
        np.testing.assert_allclose((lo + hi) / 2, y_hwc, atol=1e-5)

    def test_sampled_containment(self):
        gemm, conv, relu, forward = self._build(seed=1)
        lb, ub = np.full(5, -1.0), np.ones(5)
        star = Star.from_bounds(lb, ub)
        out = self._reach(gemm, conv, relu, star)
        lo, hi = out.estimate_ranges()
        lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
        rng = np.random.default_rng(2)
        for _ in range(300):
            x = rng.uniform(lb, ub)
            with torch.no_grad():
                y = forward(torch.tensor(x, dtype=torch.float32)
                            .unsqueeze(0)).numpy()
            y_hwc = y[0].transpose(1, 2, 0).flatten()
            assert np.all(y_hwc >= lo - 1e-5) and np.all(y_hwc <= hi + 1e-5)
