"""HardSwish activation reachability.

HardSwish(x) = x * ReLU6(x + 3) / 6.  Piecewise polynomial-of-degree-2,
non-monotone (small dip near x = -1.5 where the value is -0.375).
Coverage matches nnVLA: Box (IBP), Star (CROWN-style approx).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star


_HSWISH_X_MIN = -1.5
_HSWISH_F_MIN = -0.375  # HardSwish(-1.5)


def _hardswish(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def hardswish_box(input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        lb = b.lb.flatten()
        ub = b.ub.flatten()
        fl = _hardswish(lb)
        fu = _hardswish(ub)
        contains_min = (lb <= _HSWISH_X_MIN) & (ub >= _HSWISH_X_MIN)
        out_lb = np.where(contains_min, _HSWISH_F_MIN, np.minimum(fl, fu))
        out_ub = np.maximum(fl, fu)
        out.append(Box(out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)))
    return out


def hardswish_star_approx(input_stars: List[Star]) -> List[Star]:
    """Box-lifted Star reach, preserving ImageStar shape."""

    def _box(lb: np.ndarray, ub: np.ndarray):
        box = hardswish_box([Box(lb, ub)])[0]
        return box.lb, box.ub

    return apply_box_lift_star(input_stars, _box)
