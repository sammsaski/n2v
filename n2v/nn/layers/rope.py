"""Rotary Positional Embedding wrapper.

Position is treated as a constant; given a fixed position index the
rotation is affine, which is the regime n2v reachability covers.
"""

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary positional embedding (RoPE)."""

    def __init__(self, dim: int, base: float = 10000.0, max_len: int = 4096):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = int(dim)
        self.base = float(base)
        self.max_len = int(max_len)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = x.size(-2)
        cos = self.cos[:l]
        sin = self.sin[:l]
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin
