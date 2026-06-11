"""Selective Feature Fusion (SFF) wrapper.

Combines two feature streams via an elementwise sigmoid gate:
``y = sigma(g) * a + (1 - sigma(g)) * b``  where  ``g = W_g [a; b]``.
"""

import torch
import torch.nn as nn


class SelectiveFeatureFusion(nn.Module):
    """Weighted fusion of two feature streams via a sigmoid gate."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return g * a + (1.0 - g) * b
