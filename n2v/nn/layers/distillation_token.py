"""Distillation token (DeiT): a learnable token prepended to a sequence.

Despite the DeiT paper presenting the distillation token as appearing
*after* CLS in the final token order, this wrapper simply prepends a
learnable token at the start of the sequence. To replicate the
[CLS, distillation, patch_1, ..., patch_N] ordering, apply
``DistillationToken`` *first* on the patch sequence, then ``CLSToken``
on the result.
"""

import torch
import torch.nn as nn


class DistillationToken(nn.Module):
    """Prepend a learnable distillation token to an ``(B, L, D)`` sequence.

    See module docstring for token-ordering recipe when combining with
    :class:`~n2v.nn.layers.CLSToken`.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tok = self.token.expand(b, -1, -1)
        return torch.cat([tok, x], dim=1)
