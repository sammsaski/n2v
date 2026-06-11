"""ActionHead reachability: affine projection to action space.

Identical to a single Linear; forwards through :mod:`linear_reach` via
the inner ``proj`` submodule.
"""

from __future__ import annotations

from typing import List

import torch.nn as nn

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _proj(layer) -> nn.Linear:
    return layer.proj


def action_head_star(layer, input_stars: List[Star]) -> List[Star]:
    return linear_reach.linear_star(_proj(layer), input_stars)


def action_head_box(layer, input_boxes: List[Box]) -> List[Box]:
    return linear_reach.linear_box(_proj(layer), input_boxes)


def action_head_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    return linear_reach.linear_zono(_proj(layer), input_zonos)


def action_head_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    return linear_reach.linear_hexatope(_proj(layer), input_sets)


def action_head_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    return linear_reach.linear_octatope(_proj(layer), input_sets)
