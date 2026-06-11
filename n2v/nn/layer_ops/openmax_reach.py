"""OpenMax reachability.

OpenMax produces an augmented probability distribution over
``num_classes + 1`` outputs. Each entry lies in [0, 1] and the row
sums to 1, so the sound box reach per coordinate is [0, 1].

Coverage matches nnVLA: Box + Star (box-lifted).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star


def openmax_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        n = b.dim + 1  # augmented unknown class
        out.append(Box(np.zeros((n, 1)), np.ones((n, 1))))
    return out


def openmax_star_approx(layer, input_stars: List[Star]) -> List[Star]:
    """Output dim = num_classes + 1.

    ImageStar shape is intentionally *not* preserved: OpenMax adds an
    "unknown" class so output dimensionality differs from input.
    Downstream layers must treat the result as a flat Star.
    """
    out: List[Star] = []
    for s in input_stars:
        n = s.dim + 1
        out.append(Star.from_bounds(np.zeros((n, 1)), np.ones((n, 1))))
    return out
