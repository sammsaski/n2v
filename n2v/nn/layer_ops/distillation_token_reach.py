"""DistillationToken reachability: prepend a second learnable token.

Identical reachability pattern to :mod:`cls_token_reach`.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import cls_token_reach


def distillation_token_box(layer, input_boxes: List[Box]) -> List[Box]:
    return cls_token_reach.cls_token_box(layer, input_boxes)


def distillation_token_star(layer, input_stars: List[Star]) -> List[Star]:
    return cls_token_reach.cls_token_star(layer, input_stars)


def distillation_token_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    return cls_token_reach.cls_token_zono(layer, input_zonos)
