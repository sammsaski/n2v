"""GRN (Global Response Normalisation) wrapper."""

import torch
import torch.nn as nn


class GRN(nn.Module):
    """Global Response Normalisation as used in ConvNeXt-v2.

    For input ``x`` of shape ``(*, C)``::

        gx = ||x||_{2} along (H, W)            # (B, 1, 1, C)
        nx = gx / (mean(gx, dim=-1) + eps)
        y  = gamma * (x * nx) + beta + x
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = x.norm(p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x
