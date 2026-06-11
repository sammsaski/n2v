"""Sinusoidal positional encoding (additive, fixed)."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal positional encodings to ``(B, L, D)`` input."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        if int(dim) <= 0:
            raise ValueError(f"PositionalEncoding dim must be positive, got {dim}")
        self.dim = int(dim)
        self.max_len = int(max_len)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        # For odd `dim`, `torch.arange(0, dim, 2)` has one more column than
        # `pe[:, 1::2]`. Slice div_term for the odd-position cosine channels
        # so the assignment shapes match.
        pe[:, 0::2] = torch.sin(position * div_term)
        n_cos = pe[:, 1::2].size(1)
        pe[:, 1::2] = torch.cos(position * div_term[:n_cos])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
