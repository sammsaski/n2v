"""SegFormer-style overlapping patch embedding."""

import torch
import torch.nn as nn


class OverlapPatchEmbed(nn.Module):
    """Conv2d with kernel > stride for overlapping patches."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        patch_size: int = 7,
        stride: int = 4,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        padding = (patch_size - 1) // 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride, padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)
