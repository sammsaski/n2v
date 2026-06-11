"""2D concatenation along the channel dimension."""

import torch
import torch.nn as nn


class Concat2D(nn.Module):
    """Concatenate ``(B, C_i, H, W)`` tensors along channel dim."""

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(list(inputs), dim=1)
