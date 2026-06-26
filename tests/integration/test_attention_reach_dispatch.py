"""General-dispatcher reach for transformer self-attention (GitHub #44).

Guards the wiring that lets ``NeuralNetwork.reach`` complete the sound pipeline
on a ViT-style graph: the ``set@set`` bilinear MatMuls (Q@Kᵀ and A@V), the
batched attention Softmax, and the head-split Reshape (which must stay flat, not
be reinterpreted as a (C, H, W) image). A tiny multi-head self-attention is
exported to ONNX, loaded through the real loader, and reached as a Star; the
result must soundly enclose every sampled forward output.

See ``.claude/resolved.md`` §A I-39 / GitHub #44.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn import NeuralNetwork
from n2v.nn.reach import _const_set_like
from n2v.sets import Star, Zono, Box
from n2v.utils.model_loader import load_onnx


class _MiniAttn(nn.Module):
    """Minimal multi-head self-attention with a flat (1, L*D) interface."""

    def __init__(self, L: int = 3, D: int = 6, heads: int = 2):
        super().__init__()
        self.L, self.D, self.H, self.dh = L, D, heads, D // heads
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)

    def forward(self, x):  # x: (1, L*D)
        x = x.reshape(1, self.L, self.D)
        qkv = self.qkv(x).reshape(1, self.L, 3, self.H, self.dh)
        q = qkv[:, :, 0].transpose(1, 2)           # (1, H, L, dh)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) * (self.dh ** -0.5)   # set@set -> (1,H,L,L)
        a = torch.softmax(s, dim=-1)
        o = (a @ v).transpose(1, 2).reshape(1, self.L, self.D)   # set@set
        return self.out(o).reshape(1, self.L * self.D)


def test_attention_general_reach_is_sound(tmp_path):
    """net.reach(approx) completes on a ViT-style graph and soundly encloses
    the model's forward outputs (set@set MatMul + softmax + head-split)."""
    torch.manual_seed(0)
    L, D = 3, 6
    N = L * D
    model = _MiniAttn(L, D, heads=2).eval()
    x = torch.randn(1, N)
    onnx_path = tmp_path / "mini_attn.onnx"
    torch.onnx.export(model, x, str(onnx_path),
                      input_names=["x"], output_names=["y"], opset_version=11)

    gm = load_onnx(str(onnx_path))
    x0 = x.numpy().reshape(-1).astype(np.float64)
    eps = 0.05
    out = NeuralNetwork(gm).reach(Star.from_bounds(x0 - eps, x0 + eps),
                                  method="approx")
    o = out[0] if isinstance(out, list) else out
    assert o.dim == N  # pipeline completed end-to-end

    rlb, rub = (np.asarray(t).reshape(-1) for t in o.estimate_ranges())

    def fwd(xx):
        return gm(torch.tensor(xx.reshape(1, N).astype(np.float32))
                  ).detach().numpy().reshape(-1)

    rng = np.random.default_rng(0)
    for _ in range(500):
        xx = x0 + rng.uniform(-eps, eps, size=x0.shape)
        y = fwd(xx)
        assert np.all(y >= rlb - 1e-5) and np.all(y <= rub + 1e-5)
    # box corners too
    for xx in (x0 - eps, x0 + eps):
        y = fwd(xx)
        assert np.all(y >= rlb - 1e-5) and np.all(y <= rub + 1e-5)


class _ReshapeTransposeConv(nn.Module):
    """Reshape -> Transpose(to NCHW) -> Conv2d: the discriminator must walk past
    the transpose to the conv and materialize an ImageStar (else the conv crashes
    on a flat Star). Regression guard for the head-split reshape fix not breaking
    genuine flat -> image -> conv paths."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)

    def forward(self, x):  # x: (1, 2*4*4)
        x = x.reshape(1, 4, 4, 2).permute(0, 3, 1, 2)  # -> (1, 2, 4, 4) via Transpose
        return self.conv(x).reshape(1, -1)


def test_reshape_transpose_conv_stays_supported(tmp_path):
    """A reshape feeding a conv THROUGH a transpose must still materialize an
    ImageStar (the _reshape_feeds_spatial walk looks past OnnxTranspose)."""
    torch.manual_seed(0)
    N = 2 * 4 * 4
    model = _ReshapeTransposeConv().eval()
    x = torch.randn(1, N)
    onnx_path = tmp_path / "r2t2c.onnx"
    torch.onnx.export(model, x, str(onnx_path), opset_version=11)

    gm = load_onnx(str(onnx_path))
    x0 = x.numpy().reshape(-1).astype(np.float64)
    eps = 0.05
    out = NeuralNetwork(gm).reach(Star.from_bounds(x0 - eps, x0 + eps),
                                  method="approx")
    o = out[0] if isinstance(out, list) else out
    rlb, rub = (np.asarray(t).reshape(-1) for t in o.estimate_ranges())

    def fwd(xx):
        return gm(torch.tensor(xx.reshape(1, N).astype(np.float32))
                  ).detach().numpy().reshape(-1)

    rng = np.random.default_rng(0)
    for _ in range(300):
        y = fwd(x0 + rng.uniform(-eps, eps, size=x0.shape))
        assert np.all(y >= rlb - 1e-5) and np.all(y <= rub + 1e-5)


@pytest.mark.parametrize("ctor", [
    lambda lb, ub: Star.from_bounds(lb, ub),
    lambda lb, ub: Zono.from_bounds(lb, ub),
    lambda lb, ub: Box(lb, ub),
])
def test_const_set_like_is_exact_constant(ctor):
    """A materialized constant operand (the ViT cls-token Concat case) is an
    exact point at the given values, sharing the template's predicate system."""
    template = ctor(np.array([-1.0, 0.0, 2.0]), np.array([1.0, 0.5, 3.0]))
    const = np.array([7.0, -3.0])
    cset = _const_set_like(template, const)
    assert cset.dim == const.size
    lb, ub = (cset.estimate_ranges() if hasattr(cset, "estimate_ranges")
              else cset.get_bounds())
    np.testing.assert_allclose(np.asarray(lb).reshape(-1), const, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ub).reshape(-1), const, atol=1e-9)
