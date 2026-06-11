"""DropPath (stochastic depth) wrapper.

For reachability the layer is treated as identity at inference time
(``model.eval()``), matching standard PyTorch behaviour.
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Per-sample DropPath (stochastic depth)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        p = float(drop_prob)
        if not (0.0 <= p < 1.0):
            raise ValueError(
                f"DropPath drop_prob must be in [0, 1), got {p}. Values "
                f">= 1 would zero out every sample (keep_prob == 0 causes "
                f"division by zero in forward)."
            )
        self.drop_prob = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask
