"""SiLU / Swish activation reachability.

SiLU(x) = x * sigmoid(x). Smooth, non-monotone (small dip near x ≈ -1.28).
Coverage matches nnVLA: Box (IBP, dip-aware), Star (CROWN-style approx).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star


# PR-1 audit I1: the previous ``_SILU_X_MIN = -1.2785`` was the rounded
# value of the true argmin -1.2784645614 — but rounded AWAY from zero by
# ~3.5e-5. The point check ``(lb <= x_min) & (ub >= x_min)`` then missed
# narrow boxes that bracket the TRUE argmin but exclude the rounded
# constant (e.g. Box [-1.27848, -1.27840] contains the true argmin but
# lb = -1.27848 > -1.2785 so the check returned False), producing an
# above-floor lower bound (~2.6e-11 unsound). Fix: use the true argmin to
# ~16 digits (scipy-verified) and add a tiny inward guard band.
_SILU_X_MIN = -1.2784645614377839   # true SiLU argmin (scipy-verified)
_SILU_F_MIN = -0.27846454277        # SiLU(_SILU_X_MIN), rounded AWAY from
                                    # zero from true -0.2784645427610738
                                    # (>= 1e-12 below the true minimum).
_SILU_XMIN_GUARD = 5e-12            # inward guard against fp endpoint slip


def _silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


def silu_box(input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        lb = b.lb.flatten()
        ub = b.ub.flatten()
        fl = _silu(lb)
        fu = _silu(ub)
        contains_min = (
            (lb <= _SILU_X_MIN + _SILU_XMIN_GUARD)
            & (ub >= _SILU_X_MIN - _SILU_XMIN_GUARD)
        )
        out_lb = np.where(contains_min, _SILU_F_MIN, np.minimum(fl, fu))
        out_ub = np.maximum(fl, fu)
        out.append(Box(out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)))
    return out


def silu_star_approx(input_stars: List[Star]) -> List[Star]:
    """Box-lifted Star reach, preserving ImageStar shape."""

    def _box(lb: np.ndarray, ub: np.ndarray):
        box = silu_box([Box(lb, ub)])[0]
        return box.lb, box.ub

    return apply_box_lift_star(input_stars, _box)
