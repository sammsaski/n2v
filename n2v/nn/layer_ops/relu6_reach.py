"""ReLU6 activation reachability.

ReLU6(x) = min(max(x, 0), 6). Monotone non-decreasing, piecewise linear.
The exact Star reach can be obtained by composing ReLU with a clamp at
6, but the practical implementation uses a single three-region linear
relaxation as in nnVLA's ``relu6/methods/crown.py``.

Coverage (matches nnVLA): Box (IBP), Star (CROWN-style approx).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star


def _relu6(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, 0.0), 6.0)


# ---------------------------------------------------------------------------
# Box (IBP)
# ---------------------------------------------------------------------------

def relu6_box(input_boxes: List[Box]) -> List[Box]:
    """ReLU6 is monotone; applying it to the bounds is exact."""
    return [Box(_relu6(b.lb), _relu6(b.ub)) for b in input_boxes]


# ---------------------------------------------------------------------------
# Star (CROWN-style approx)
# ---------------------------------------------------------------------------

def relu6_star_approx(input_stars: List[Star]) -> List[Star]:
    """Box-lifted Star approx for ReLU6, preserving ImageStar shape."""

    def _box(lb: np.ndarray, ub: np.ndarray):
        return _relu6(lb), _relu6(ub)

    return apply_box_lift_star(input_stars, _box)
