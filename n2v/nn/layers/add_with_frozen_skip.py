"""Add the running activation with a stored (frozen) skip tensor."""

import torch
import torch.nn as nn


class AddWithFrozenSkip(nn.Module):
    """Add a frozen (buffer) skip tensor to the current activation.

    Useful for verifying networks where a residual side has been
    precomputed and is therefore a constant from the reachability
    perspective.
    """

    def __init__(self, skip: torch.Tensor):
        super().__init__()
        self.register_buffer("skip", skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.skip
