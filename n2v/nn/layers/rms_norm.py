"""RMSNorm wrapper module."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    ``y = x / sqrt(mean(x**2) + eps) * weight``

    Mirrors ``nnvla.components.rmsnorm`` so users can author models that
    n2v's dispatcher recognises by ``isinstance``.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        out = x / rms
        if self.weight is not None:
            out = out * self.weight
        return out
