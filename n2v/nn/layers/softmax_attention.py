"""SoftmaxAttention wrapper module.

Wraps a vanilla scaled dot-product softmax attention so that n2v's
dispatcher can detect the op via ``isinstance``. The forward path
manually computes ``softmax(q @ k.T / sqrt(d_head)) @ v`` rather than
delegating to ``F.scaled_dot_product_attention``, so PyTorch-specific
options like ``dropout_p``, ``is_causal`` and ``scale`` are not
supported here.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxAttention(nn.Module):
    """Plain scaled dot-product softmax attention.

    Forward signature: ``forward(q, k, v, attn_mask=None)``.
    """

    def __init__(self, d_head: Optional[int] = None):
        super().__init__()
        self.d_head = d_head

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        d_head = self.d_head if self.d_head is not None else q.size(-1)
        scale = 1.0 / math.sqrt(d_head)
        logits = (q @ k.transpose(-1, -2)) * scale
        if attn_mask is not None:
            logits = logits + attn_mask
        attn = F.softmax(logits, dim=-1)
        return attn @ v
