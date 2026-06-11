"""Segment embedding (e.g. BERT type-id embedding)."""

import torch
import torch.nn as nn


class SegmentEmbedding(nn.Module):
    """Add a per-segment learned embedding to an ``(B, L, D)`` sequence."""

    def __init__(self, num_segments: int, dim: int):
        super().__init__()
        self.num_segments = int(num_segments)
        self.dim = int(dim)
        self.embedding = nn.Embedding(num_segments, dim)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        return x + self.embedding(segment_ids)
