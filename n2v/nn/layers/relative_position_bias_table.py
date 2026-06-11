"""Swin-style relative position bias table (additive learned bias)."""

import torch
import torch.nn as nn


class RelativePositionBiasTable(nn.Module):
    """Swin Transformer style relative position bias table.

    Given a window size ``W`` and ``n_heads`` heads, stores a learnable
    ``(2W-1)^2 x n_heads`` table that is indexed by relative position to
    produce a per-head additive bias of shape ``(n_heads, W*W, W*W)``.
    """

    def __init__(self, window_size: int, n_heads: int):
        super().__init__()
        self.window_size = int(window_size)
        self.n_heads = int(n_heads)
        n_rel = (2 * window_size - 1) ** 2
        self.bias_table = nn.Parameter(torch.zeros(n_rel, n_heads))

        coords = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(coords, coords, indexing="ij")).flatten(1)
        rel = grid[:, :, None] - grid[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        index = rel.sum(-1)
        self.register_buffer("relative_position_index", index, persistent=False)

    def forward(self) -> torch.Tensor:
        bias = self.bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        return bias.permute(2, 0, 1).contiguous()
