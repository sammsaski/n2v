"""ActionTokenizer reachability.

ActionTokenizer maps continuous actions to integer token IDs via
uniform binning. The output is a discrete-valued tensor; for sound
bounds we lift to the integer range ``[0, n_bins - 1]`` per dimension.

Coverage matches nnVLA: Box + Star (box-lifted), no Zono (output is
integer-valued so a zonotope generator basis is not meaningful).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar


def action_tokenizer_box(layer, input_boxes: List[Box]) -> List[Box]:
    n_bins = int(layer.n_bins) - 1
    out: List[Box] = []
    for b in input_boxes:
        out.append(
            Box(np.zeros_like(b.lb), np.full_like(b.ub, n_bins, dtype=np.float64))
        )
    return out


def action_tokenizer_star_approx(layer, input_stars: List[Star]) -> List[Star]:
    """Integer-valued output bounded to [0, n_bins-1] per coordinate.

    Output dimension equals input dimension, so ImageStar shape is
    preserved if present.
    """
    n_bins = int(layer.n_bins) - 1
    out: List[Star] = []
    for s in input_stars:
        is_image = isinstance(s, ImageStar)
        dim = s.to_star().dim if is_image else s.dim
        new_star = Star.from_bounds(np.zeros((dim, 1)), np.full((dim, 1), float(n_bins)))
        if is_image:
            new_star = new_star.to_image_star(s.height, s.width, s.num_channels)
        out.append(new_star)
    return out
