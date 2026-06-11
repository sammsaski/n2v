"""Pooler: take the CLS token's representation and project + tanh."""

import torch
import torch.nn as nn


class Pooler(nn.Module):
    """BERT-style pooler: ``tanh(W @ x[:, 0])``."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_token = x[:, 0]
        return self.activation(self.dense(first_token))
