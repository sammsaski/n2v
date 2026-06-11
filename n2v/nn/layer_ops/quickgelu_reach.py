"""QuickGELU activation reachability.

QuickGELU(x) = x * sigmoid(1.702 * x).  Smooth non-monotone (small dip
near x ≈ -1.18). Box reach accounts for the dip; Star reach uses a
sound box-lifted relaxation matching nnVLA's CROWN fallback.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star


# QuickGELU(x) = x * sigmoid(1.702 * x). The global minimum is at the
# unique negative root of d/dx (x * sigmoid(1.702 x)) = 0; numerically
# x* ≈ -0.7517 with f* ≈ -0.1638. Use slightly conservative values so the
# sound lower bound never underestimates the dip.
_QGELU_X_MIN = -0.7517
_QGELU_F_MIN = -0.1639


def _quickgelu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-1.702 * x))


def quickgelu_box(input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        lb = b.lb.flatten()
        ub = b.ub.flatten()
        fl = _quickgelu(lb)
        fu = _quickgelu(ub)
        contains_min = (lb <= _QGELU_X_MIN) & (ub >= _QGELU_X_MIN)
        out_lb = np.where(contains_min, _QGELU_F_MIN, np.minimum(fl, fu))
        out_ub = np.maximum(fl, fu)
        out.append(Box(out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)))
    return out


def quickgelu_star_approx(input_stars: List[Star]) -> List[Star]:
    """Box-lifted Star reach, preserving ImageStar shape."""

    def _box(lb: np.ndarray, ub: np.ndarray):
        box = quickgelu_box([Box(lb, ub)])[0]
        return box.lb, box.ub

    return apply_box_lift_star(input_stars, _box)
