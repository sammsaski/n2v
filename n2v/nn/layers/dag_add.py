"""DAG add: explicit multi-input addition node."""

import torch
import torch.nn as nn


class DagAdd(nn.Module):
    """Sum N input tensors.

    Wraps elementwise addition as a single nn.Module so the
    n2v dispatcher can detect it without relying on bare
    ``+`` / ``torch.add`` calls in the model graph.
    """

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) == 0:
            raise ValueError("DagAdd requires at least one input")
        out = inputs[0]
        for t in inputs[1:]:
            out = out + t
        return out
