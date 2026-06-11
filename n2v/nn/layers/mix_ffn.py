"""MixFFN (SegFormer): Linear -> 3x3 depthwise conv -> GELU -> Linear."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MixFFN(nn.Module):
    """SegFormer Mix-FFN block (channel-mixing + spatial mixing).

    Forward expects an ``(B, L, D)`` sequence where ``L`` is a perfect
    square so the intermediate ``(B, D, H, W)`` form can be constructed.
    Non-square sequence lengths raise ``ValueError``.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else 4 * dim)
        self.fc1 = nn.Linear(self.dim, self.hidden_dim)
        self.dwconv = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, groups=self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        b, l, d = x.shape
        h = int(l ** 0.5)
        if h * h != l:
            raise ValueError(
                f"MixFFN requires a square sequence length; got L={l} "
                f"(not a perfect square)."
            )
        x_2d = x.transpose(1, 2).reshape(b, d, h, h)
        x_2d = self.dwconv(x_2d)
        x = x_2d.flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)
