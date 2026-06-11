"""CLS token: prepends a learnable token to a sequence."""

import torch
import torch.nn as nn


class CLSToken(nn.Module):
    """Prepend a learnable ``[CLS]`` token to an ``(B, L, D)`` sequence."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        cls = self.token.expand(b, -1, -1)
        return torch.cat([cls, x], dim=1)
