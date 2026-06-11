"""LayerScale wrapper: per-channel learnable scaling."""

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """Per-channel learnable scaling factor.

    ``y = gamma * x`` where ``gamma`` is a learnable vector of length
    ``dim``. Used in DeiT, ConvNeXt and many ViT variants.
    """

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.dim = int(dim)
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma
