"""ViT-style patch embedding: stride-equal-kernel Conv2d + flatten."""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image-to-patch embedding via a single Conv2d with stride=patch_size."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
