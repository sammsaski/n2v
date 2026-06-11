"""Parallel residual block (GPT-J style)."""

import torch
import torch.nn as nn


class ParallelResidual(nn.Module):
    """y = x + sublayer_a(x) + sublayer_b(x)."""

    def __init__(self, sublayer_a: nn.Module, sublayer_b: nn.Module):
        super().__init__()
        self.sublayer_a = sublayer_a
        self.sublayer_b = sublayer_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sublayer_a(x) + self.sublayer_b(x)
