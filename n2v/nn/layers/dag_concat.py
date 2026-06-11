"""DAG concat: explicit multi-input concatenation node."""

import torch
import torch.nn as nn


class DagConcat(nn.Module):
    """Concatenate N input tensors along ``dim``."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = int(dim)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(list(inputs), dim=self.dim)
