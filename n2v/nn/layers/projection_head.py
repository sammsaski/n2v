"""Projection head: Linear -> activation -> Linear (CLIP / DINO style)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head with a configurable activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.out_features = int(out_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation if activation is not None else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
