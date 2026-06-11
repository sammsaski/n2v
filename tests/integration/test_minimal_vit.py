"""Synthetic minimal-ViT integration test (vit_2023 benchmark target).

The VNN-COMP ``vit_2023`` smoke test instance is ``pgd_2_3_16.onnx`` (2
blocks, 3 heads, dim 16). This file builds an equivalent PyTorch ViT
**using n2v's wrapper modules** and exercises the full
``NeuralNetwork(model).reach(...)`` path on a small L∞ input box. The
intent is to expose every dispatch gap between n2v's current state and
end-to-end ViT verification, then close them one gap at a time.

The test is deliberately tiny (4x4 image, 2x2 patches -> 4 tokens + CLS
= 5 tokens, dim=4, 1 head per block, 1 block) so the reach completes in
milliseconds when it works. The expansion path (matching
``pgd_2_3_16.onnx``) is wired but skipped until each gap is closed.

If reach() raises a ``NotImplementedError``, the error message names the
specific primitive that needs a handler (see PR12_FIX_LIST T1-6 / T1-7).
This test therefore behaves as a checklist: each red XFAIL becomes green
as the matching commit lands.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from n2v.nn import NeuralNetwork
from n2v.sets import Box, Star
from n2v.sets.image_zono import ImageZono
from n2v.sets.image_star import ImageStar
from n2v.nn.layers import (
    CLSToken,
    PatchEmbed,
    SoftmaxAttention,
)


# ---------------------------------------------------------------------------
# Synthetic ViT building blocks
# ---------------------------------------------------------------------------


class MultiheadSelfAttention(nn.Module):
    """A self-attention wrapper that projects x -> Q/K/V via three Linears
    then runs n2v's SoftmaxAttention on each head.

    Single-head case (the tiny test below) keeps the shape manipulation
    minimal so we can iterate against the dispatcher gap-by-gap.
    """

    def __init__(self, dim: int, n_heads: int = 1):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attn = SoftmaxAttention(d_head=self.d_head)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, dim)
        q = self.q_proj(x)  # (B, L, dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Single-head case: pass directly. SoftmaxAttention's forward
        # computes softmax(q @ k.T / sqrt(d_head)) @ v.
        out = self.attn(q, k, v)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x -> x + Attn(LN(x)) -> x + MLP(LN(x))."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_in = nn.Linear(dim, mlp_dim)
        self.mlp_out = nn.Linear(mlp_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual paths use Python-level operator.add (silently dropped
        # pre-Tier-0; fail-loud now).
        x = x + self.attn(self.norm1(x))
        h = self.mlp_out(F.gelu(self.mlp_in(self.norm2(x)), approximate="tanh"))
        x = x + h
        return x


class MinimalViT(nn.Module):
    """Tiny synthetic ViT modelled on VNN-COMP vit_2023 (pgd_2_3_16).

    Default dims chosen so a single L∞ reach is tractable:
      image    : 4 x 4 x 1
      patch    : 2 x 2     ->  4 patches
      CLS      : 1 token   ->  L = 5 tokens
      dim      : 4
      heads    : 1
      blocks   : 1
      classes  : 2
    """

    def __init__(
        self,
        img_size: int = 4,
        patch_size: int = 2,
        in_channels: int = 1,
        dim: int = 4,
        n_heads: int = 1,
        n_blocks: int = 1,
        n_classes: int = 2,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            in_channels=in_channels, embed_dim=dim, patch_size=patch_size,
        )
        n_patches = (img_size // patch_size) ** 2
        self.n_tokens = n_patches + 1  # +1 for CLS
        self.cls_token = CLSToken(dim=dim)
        # Learnable positional embedding (additive); keep tiny for the test.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, n_heads=n_heads, mlp_ratio=mlp_ratio)
             for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)        # (B, L=n_patches, dim)
        x = self.cls_token(x)          # (B, L+1, dim)
        x = x + self.pos_embed         # (B, L+1, dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # CLS-token classifier head.
        cls_h = x[:, 0]                # (B, dim)
        return self.head(cls_h)


# ---------------------------------------------------------------------------
# Reach-end-to-end tests (XFAIL-marked while gaps remain)
# ---------------------------------------------------------------------------


def _make_input_image_zono(model: nn.Module, img_size: int = 4) -> ImageZono:
    """L∞ ImageZono around a fixed seed image, radius 1/255 (VNN-COMP-style).

    Conv2d is dispatched for Zono/ImageZono inputs (the Box helper exists
    but raises by design -- Conv2d needs explicit image shape, which
    Box does not carry). For ViT verification ``ImageZono`` is the right
    input type at the entry of patch embedding; downstream layers receive
    flat sets after PatchEmbed's flatten+transpose.
    """
    torch.manual_seed(0)
    c = model.patch_embed.in_channels
    img = torch.randn(1, c, img_size, img_size)
    radius = 1.0 / 255.0
    lb = (img - radius).detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
    ub = (img + radius).detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
    return ImageZono.from_bounds(lb, ub, height=img_size, width=img_size, num_channels=c)


def test_minimal_vit_zono_reach_completes():
    """T-ViT: build a tiny ViT, run NeuralNetwork.reach with an ImageZono
    input, and assert reach() completes without raising. This test is the
    canonical ViT-enable check; success means the dispatch chain is
    complete for the full vit_2023 architecture (modulo Star paths).

    Acceptance: output set has 2 dims (n_classes) and finite bounds, and
    a Monte-Carlo containment check confirms soundness (the reach over-
    approximation contains 100 concrete forward outputs).
    """
    torch.manual_seed(42)
    model = MinimalViT()
    model.eval()

    input_zono = _make_input_image_zono(model)
    out_sets = NeuralNetwork(model).reach(
        input_zono, method="approx", n_tokens=model.n_tokens,
    )

    assert len(out_sets) == 1
    out = out_sets[0]
    assert out.dim == 2  # n_classes
    out_lb, out_ub = out.get_bounds()
    assert np.all(np.isfinite(out_lb.flatten()))
    assert np.all(np.isfinite(out_ub.flatten()))

    # Soundness: sample inputs from the input box and check concrete
    # outputs lie inside the reach bounds.
    z_lb, z_ub = input_zono.get_bounds()
    rng = np.random.default_rng(0)
    samples = rng.uniform(
        z_lb.flatten(), z_ub.flatten(), size=(64, z_lb.size),
    ).astype(np.float32)
    imgs = torch.from_numpy(samples.reshape(
        64, model.patch_embed.in_channels, 4, 4,
    ))
    with torch.no_grad():
        outs = model(imgs).numpy()
    assert np.all(outs >= out_lb.flatten() - 1e-6), \
        "Concrete output min escapes reach lower bound (UNSOUND)."
    assert np.all(outs <= out_ub.flatten() + 1e-6), \
        "Concrete output max escapes reach upper bound (UNSOUND)."


def test_minimal_vit_image_star_reach_raises_until_t1_1_audit_N10():
    """PR-1 audit N10: was ``xfail(strict=False)`` which passes whether
    the test raises OR succeeds -- including an unsound XPASS where the
    Star path silently completes with the wrong reach. Convert to
    ``pytest.raises(NotImplementedError, match='multi-token')`` so the
    pin is specific: the Star path MUST raise on multi-token input
    until PR12_FIX_LIST T1-1 (per-group LayerNorm Star reach) lands.
    When T1-1 lands, this test will fail and the assertion can be
    replaced with the original soundness check.
    """
    torch.manual_seed(42)
    model = MinimalViT()
    model.eval()

    zono = _make_input_image_zono(model)
    lb, ub = zono.get_bounds()
    img_size = 4
    c = model.patch_embed.in_channels
    input_star = ImageStar.from_bounds(
        lb, ub, height=img_size, width=img_size, num_channels=c,
    )
    with pytest.raises(NotImplementedError, match="multi-token"):
        NeuralNetwork(model).reach(input_star, method="approx")
