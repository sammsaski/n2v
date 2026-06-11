"""RelativePositionBiasTable (Swin) reachability.

The forward produces a fixed bias tensor whose entries are values
indexed from the learnable ``bias_table`` parameter. The reach result
is therefore the per-position interval ``[min(bias_table), max(bias_table)]``
of that table — a sound box-shaped constant set independent of the
input. Returning a zero degenerate set would be unsound for any
nonzero trained bias.

Coverage matches nnVLA: Box, Star, Zono (and Hex/Oct as a constant set).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.sets.image_star import ImageStar


def _bias_extrema(layer) -> Tuple[float, float]:
    """Return ``(min, max)`` over the layer's ``bias_table``."""
    table = layer.bias_table.detach().cpu().numpy()
    return float(table.min()), float(table.max())


def relative_position_bias_table_box(layer, input_boxes: List[Box]) -> List[Box]:
    lo, hi = _bias_extrema(layer)
    out: List[Box] = []
    for b in input_boxes:
        out.append(Box(np.full_like(b.lb, lo), np.full_like(b.ub, hi)))
    return out


def relative_position_bias_table_star(layer, input_stars: List[Star]) -> List[Star]:
    """Sound constant set bounded by the learned bias-table extrema."""
    lo, hi = _bias_extrema(layer)
    out: List[Star] = []
    for s in input_stars:
        is_image = isinstance(s, ImageStar)
        dim = s.to_star().dim if is_image else s.dim
        new_star = Star.from_bounds(np.full((dim, 1), lo), np.full((dim, 1), hi))
        if is_image:
            new_star = new_star.to_image_star(s.height, s.width, s.num_channels)
        out.append(new_star)
    return out


def relative_position_bias_table_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    lo, hi = _bias_extrema(layer)
    out: List[Zono] = []
    for z in input_zonos:
        out.append(Zono.from_bounds(np.full((z.dim, 1), lo), np.full((z.dim, 1), hi)))
    return out


def relative_position_bias_table_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    lo, hi = _bias_extrema(layer)
    out: List[Hexatope] = []
    for s in input_sets:
        out.append(Hexatope.from_bounds(np.full((s.dim, 1), lo), np.full((s.dim, 1), hi)))
    return out


def relative_position_bias_table_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    lo, hi = _bias_extrema(layer)
    out: List[Octatope] = []
    for s in input_sets:
        out.append(Octatope.from_bounds(np.full((s.dim, 1), lo), np.full((s.dim, 1), hi)))
    return out
