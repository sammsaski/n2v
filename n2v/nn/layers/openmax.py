"""OpenMax wrapper: open-set recognition variant of softmax.

The full Bendale & Boult (2016) OpenMax uses Weibull-distribution
parameters fit on a calibration set to recalibrate logits before
softmax. This wrapper is a *simplified* variant: it applies a
rank-based linear taper to the top-``alpha`` logits, then softmax
over ``num_classes + 1`` augmented classes (the extra class is the
"unknown" mass). The Weibull parameters are reserved as buffers so a
downstream subclass can plug in the full calibration without changing
the dispatcher contract; the default ``forward`` does not consult them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenMax(nn.Module):
    """Rank-tapered OpenMax-like activation. See module docstring."""

    def __init__(self, num_classes: int, alpha: int = 10):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = int(alpha)
        # Buffers for a future Weibull-calibrated subclass; unused here.
        self.mean_vec = nn.Parameter(torch.zeros(num_classes, num_classes), requires_grad=False)
        self.weibull_scale = nn.Parameter(torch.ones(num_classes), requires_grad=False)
        self.weibull_shape = nn.Parameter(torch.ones(num_classes), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank = torch.argsort(x, dim=-1, descending=True)
        recalib = torch.ones_like(x)
        for i in range(min(self.alpha, x.size(-1))):
            w = float(self.alpha - i) / float(self.alpha)
            idx = rank[..., i:i + 1]
            recalib.scatter_(-1, idx, w)
        recalib_x = x * recalib
        unknown = (x - recalib_x).sum(dim=-1, keepdim=True)
        augmented = torch.cat([recalib_x, unknown], dim=-1)
        return F.softmax(augmented, dim=-1)
