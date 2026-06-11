"""Tied-weight linear (decoder weight = embedding weight, transposed)."""

import torch
import torch.nn as nn


class TiedLinear(nn.Module):
    """Linear layer whose weight is tied to another module's weight.

    Common pattern in language models: the unembedding linear shares
    weights with the input embedding matrix.
    """

    def __init__(self, source: nn.Module, bias: bool = False):
        super().__init__()
        if not hasattr(source, "weight"):
            raise TypeError("TiedLinear source must expose a 'weight' tensor")
        self._source = [source]  # avoid registering as submodule
        out_features, in_features = source.weight.shape
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        return self._source[0].weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)
