"""CausalMask wrapper: adds an additive lower-triangular mask."""

import torch
import torch.nn as nn


class CausalMask(nn.Module):
    """Lower-triangular causal mask applied additively to logits.

    Adds ``-inf`` (or a large negative value) to the upper triangle of an
    ``(L, L)`` attention logit matrix.
    """

    def __init__(self, max_len: int = 4096, fill_value: float = -1e9):
        super().__init__()
        self.max_len = int(max_len)
        self.fill_value = float(fill_value)
        mask = torch.full((max_len, max_len), fill_value)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        l = logits.size(-1)
        return logits + self.mask[:l, :l]
