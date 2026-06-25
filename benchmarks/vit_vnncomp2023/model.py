"""PyTorch reimplementation of the VNN-COMP 2023 ViT benchmark models.

The benchmark ships ONNX (opset 9) that onnx2torch cannot ingest (it fails on
``Slice`` v1, and after an opset upgrade on the per-token BatchNorm). The
exported graph replaced the training-time LayerNorm with **BatchNorm1d over the
embedding-channel axis** and pools with **ReduceMean over tokens** (not a
cls-token gather). This module mirrors the *exported ONNX* so that we verify the
deployed model, and loads weights directly from the ONNX initializers.

Faithful to within ~1e-6 of onnxruntime (see ``tests`` parity check). Ported
from the W11-vit-vnncomp-reconciliation reimplementation, itself reverse-
engineered from the ONNX initializer dump:

    block 0:  cls_token (48,)  pos (Np+1, 48)  Conv(3->48, k=patch, s=patch)
    block 1.i.0:  PreNorm-BN -> Attention(Q,K,V,Out = 48)
    block 1.i.1:  PreNorm-BN -> FF(48 -> 96 -> 48) with ReLU
    block 2:      BN(48) -> Linear(48, 10) on the mean-pooled tokens

Heads = 3, dim_head = 16, scale = 1/4. Two configs:
  * pgd_2_3_16: patch 16, depth 2  (5 tokens)
  * ibp_3_3_8:  patch 8,  depth 3  (17 tokens)
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
import torch.nn as nn


class Attn(nn.Module):
    def __init__(self, dim=48, heads=3, dim_head=16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner = heads * dim_head
        self.query = nn.Linear(dim, inner, bias=True)
        self.key = nn.Linear(dim, inner, bias=True)
        self.value = nn.Linear(dim, inner, bias=True)
        self.out = nn.Linear(inner, dim, bias=True)

    def forward(self, x):
        # x: (B, N, dim)
        B, N, _ = x.shape
        H, D = self.heads, self.dim_head
        q = self.query(x).reshape(B, N, H, D).transpose(1, 2)  # (B,H,N,D)
        k = self.key(x).reshape(B, N, H, D).transpose(1, 2)
        v = self.value(x).reshape(B, N, H, D).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,N,N)
        att = att.softmax(dim=-1)
        out = att @ v  # (B,H,N,D)
        out = out.transpose(1, 2).reshape(B, N, H * D)
        return self.out(out)


class FF(nn.Module):
    def __init__(self, dim=48, hidden=96):
        super().__init__()
        self.l0 = nn.Linear(dim, hidden, bias=True)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        return self.l3(self.relu(self.l0(x)))


class BNAcrossSeq(nn.Module):
    """BatchNorm1d over the embedding-channel axis of (B, N, C), eval mode.

    The ONNX graph transposes to (B, C, N), applies BatchNorm with running
    stats of shape (C,), then transposes back. In eval mode this is a fixed
    per-channel affine ``y = (x-mean)/sqrt(var+eps)*weight + bias``, hence an
    exact affine map for reachability.
    """

    def __init__(self, dim=48):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.eps = 1e-5

    def affine(self):
        """Return per-channel (scale, shift) so that ``y = scale*x + shift``."""
        scale = self.weight.detach().cpu().numpy() / (
            (self.running_var.detach().cpu().numpy() + self.eps) ** 0.5)
        shift = (self.bias.detach().cpu().numpy()
                 - scale * self.running_mean.detach().cpu().numpy())
        return scale, shift

    def forward(self, x):
        m = self.running_mean.view(1, 1, -1)
        v = self.running_var.view(1, 1, -1)
        w = self.weight.view(1, 1, -1)
        b = self.bias.view(1, 1, -1)
        return (x - m) / torch.sqrt(v + self.eps) * w + b


class TransformerBlock(nn.Module):
    def __init__(self, dim=48, heads=3, dim_head=16, mlp_dim=96):
        super().__init__()
        self.norm_attn = BNAcrossSeq(dim)
        self.attn = Attn(dim, heads=heads, dim_head=dim_head)
        self.norm_ff = BNAcrossSeq(dim)
        self.ff = FF(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class ViT_BN(nn.Module):
    """Mirrors the exported ONNX. ``patch_size`` & ``depth`` differ between
    ``ibp_3_3_8`` (8, 3) and ``pgd_2_3_16`` (16, 2)."""

    def __init__(self, image_size=32, patch_size=8, depth=3, dim=48,
                 heads=3, dim_head=16, mlp_dim=96, num_classes=10):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.n_tokens = n_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(dim))
        self.positions = nn.Parameter(torch.zeros(n_patches + 1, dim))
        self.projection = nn.Conv2d(3, dim, kernel_size=patch_size,
                                    stride=patch_size, bias=True)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, heads=heads, dim_head=dim_head,
                             mlp_dim=mlp_dim)
            for _ in range(depth)
        ])
        self.head_norm = BNAcrossSeq(dim)
        self.head = nn.Linear(dim, num_classes, bias=True)

    def forward(self, x):
        # x: (B, 3, 32, 32) -- already normalised by mean/std outside.
        B = x.shape[0]
        z = self.projection(x)            # (B, dim, H/p, W/p)
        z = z.flatten(2).transpose(1, 2)  # (B, N, dim)
        cls = self.cls_token.view(1, 1, -1).expand(B, 1, -1)
        z = torch.cat([cls, z], dim=1)    # (B, N+1, dim)
        z = z + self.positions.unsqueeze(0)
        for blk in self.blocks:
            z = blk(z)
        z = z.mean(dim=1)                  # ReduceMean over tokens -> (B, dim)
        z = self.head_norm(z.unsqueeze(1)).squeeze(1)
        return self.head(z)


def _np_from_init(init):
    return torch.from_numpy(onnx.numpy_helper.to_array(init).copy()).float()


def load_from_onnx(model: ViT_BN, onnx_path: str | Path) -> None:
    """Load weights from the benchmark ONNX into ``model`` (strict)."""
    proto = onnx.load(str(onnx_path))
    inits = {init.name: _np_from_init(init) for init in proto.graph.initializer}
    sd = model.state_dict()
    dim = model.dim

    sd["cls_token"] = inits["0.cls_token"].view(dim)
    sd["positions"] = inits["0.positions"]
    sd["projection.weight"] = inits["0.projection.weight"]
    sd["projection.bias"] = inits["0.projection.bias"]
    sd["head_norm.weight"] = inits["2.1.weight"]
    sd["head_norm.bias"] = inits["2.1.bias"]
    sd["head_norm.running_mean"] = inits["2.1.running_mean"]
    sd["head_norm.running_var"] = inits["2.1.running_var"]
    sd["head.weight"] = inits["2.2.weight"]
    sd["head.bias"] = inits["2.2.bias"]

    depth = len(model.blocks)
    for i in range(depth):
        sd[f"blocks.{i}.norm_attn.weight"] = inits[f"1.{i}.0.fn.0.norm.weight"]
        sd[f"blocks.{i}.norm_attn.bias"] = inits[f"1.{i}.0.fn.0.norm.bias"]
        sd[f"blocks.{i}.norm_attn.running_mean"] = inits[f"1.{i}.0.fn.0.norm.running_mean"]
        sd[f"blocks.{i}.norm_attn.running_var"] = inits[f"1.{i}.0.fn.0.norm.running_var"]
        sd[f"blocks.{i}.attn.query.bias"] = inits[f"1.{i}.0.fn.1.query.bias"]
        sd[f"blocks.{i}.attn.key.bias"] = inits[f"1.{i}.0.fn.1.key.bias"]
        sd[f"blocks.{i}.attn.value.bias"] = inits[f"1.{i}.0.fn.1.value.bias"]
        sd[f"blocks.{i}.attn.out.bias"] = inits[f"1.{i}.0.fn.1.out.bias"]
        sd[f"blocks.{i}.norm_ff.weight"] = inits[f"1.{i}.1.fn.0.norm.weight"]
        sd[f"blocks.{i}.norm_ff.bias"] = inits[f"1.{i}.1.fn.0.norm.bias"]
        sd[f"blocks.{i}.norm_ff.running_mean"] = inits[f"1.{i}.1.fn.0.norm.running_mean"]
        sd[f"blocks.{i}.norm_ff.running_var"] = inits[f"1.{i}.1.fn.0.norm.running_var"]
        sd[f"blocks.{i}.ff.l0.bias"] = inits[f"1.{i}.1.fn.1.0.bias"]
        sd[f"blocks.{i}.ff.l3.bias"] = inits[f"1.{i}.1.fn.1.3.bias"]

    # Weight-bearing MatMul nodes (RHS is an initializer): per block the order
    # is Q, K, V, OUT, FF.l0, FF.l3. (QK^T and Attn@V have activation inputs on
    # both sides, so they are excluded.) ONNX MatMul is y = x @ W with W shape
    # (in, out); torch Linear weight is (out, in), hence the transpose.
    matmul_nodes = [
        n for n in proto.graph.node
        if n.op_type == "MatMul" and any(inp in inits for inp in n.input)
    ]
    expected = ["query", "key", "value", "out", "ff.l0", "ff.l3"]
    if len(matmul_nodes) != 6 * depth:
        raise ValueError(
            f"expected {6*depth} weight-matmul nodes, got {len(matmul_nodes)}")
    for i in range(depth):
        for j, key in enumerate(expected):
            node = matmul_nodes[i * 6 + j]
            wname = next(n for n in node.input if n in inits)
            w_torch = inits[wname].transpose(0, 1).contiguous()
            pkey = (f"blocks.{i}.{key}.weight" if "ff" in key
                    else f"blocks.{i}.attn.{key}.weight")
            sd[pkey] = w_torch

    model.load_state_dict(sd, strict=True)
    model.eval()


def build_model(name: str, onnx_path: str | Path) -> ViT_BN:
    name = name.lower()
    if name == "ibp_3_3_8":
        m = ViT_BN(image_size=32, patch_size=8, depth=3)
    elif name == "pgd_2_3_16":
        m = ViT_BN(image_size=32, patch_size=16, depth=2)
    else:
        raise ValueError(f"unknown model name {name!r}")
    load_from_onnx(m, onnx_path)
    return m
