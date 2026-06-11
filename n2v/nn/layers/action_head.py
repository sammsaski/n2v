"""Action head: linear projection from feature space to action space."""

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Affine action head used in VLA (vision-language-action) models."""

    def __init__(self, in_features: int, action_dim: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.action_dim = int(action_dim)
        self.proj = nn.Linear(in_features, action_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
