"""T5-style relative attention bias (additive bucketed bias)."""

import math

import torch
import torch.nn as nn


class RelativeAttentionBiasT5(nn.Module):
    """Learnable bucketed relative position bias added to attention logits."""

    def __init__(
        self,
        num_buckets: int = 32,
        max_distance: int = 128,
        n_heads: int = 8,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.n_heads = int(n_heads)
        self.bidirectional = bool(bidirectional)
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        ret = 0
        n = -relative_position
        num_buckets = self.num_buckets
        if self.bidirectional:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = n.abs()
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, q_len: int, k_len: int) -> torch.Tensor:
        context_position = torch.arange(q_len, dtype=torch.long, device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(k_len, dtype=torch.long, device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)
        return values.permute(2, 0, 1).unsqueeze(0)
