"""TiedLinear reachability.

TiedLinear is a plain linear layer whose weight is shared with another
module (e.g. a token embedding). At reachability time it is identical
to :class:`torch.nn.Linear`, so this module simply forwards to
:mod:`linear_reach` after materialising the tied weight into a
synthetic ``nn.Linear``.

Coverage matches nnVLA: Box, Star, Zono, Hexatope, Octatope.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _as_linear(layer) -> nn.Linear:
    weight = layer.weight
    bias = getattr(layer, "bias", None)
    out_features, in_features = weight.shape
    dummy = nn.Linear(in_features, out_features, bias=bias is not None)
    with torch.no_grad():
        dummy.weight.copy_(weight.detach().float())
        if bias is not None:
            dummy.bias.copy_(bias.detach().float())
    return dummy


def tied_linear_star(layer, input_stars: List[Star]) -> List[Star]:
    return linear_reach.linear_star(_as_linear(layer), input_stars)


def tied_linear_box(layer, input_boxes: List[Box]) -> List[Box]:
    return linear_reach.linear_box(_as_linear(layer), input_boxes)


def tied_linear_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    return linear_reach.linear_zono(_as_linear(layer), input_zonos)


def tied_linear_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    return linear_reach.linear_hexatope(_as_linear(layer), input_sets)


def tied_linear_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    return linear_reach.linear_octatope(_as_linear(layer), input_sets)
