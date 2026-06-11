"""ELU activation reachability.

ELU(x) = x for x >= 0, alpha*(exp(x) - 1) for x < 0. Smooth and monotone
non-decreasing FOR alpha >= 0. PyTorch permits negative alpha for which
the negative branch is strictly DECREASING and the endpoint-evaluation
soundness argument below fails.

Coverage matches nnVLA: Box (IBP), Star (CROWN-style linear relaxation).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star


def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x >= 0, x, alpha * (np.exp(np.minimum(x, 0.0)) - 1.0))


def _check_alpha(alpha: float) -> None:
    """T0-4 (audit C-high): the endpoint-evaluation box reach below assumes
    ELU is monotone non-decreasing, which is only true for ``alpha >= 0``.
    PyTorch permits negative alpha. Mounting evidence MC: with alpha=-1
    over [-3, 2], elu_box reports lb=0.9502 but the true min is 0.0;
    79128/100000 samples escape. Fail loud rather than under-approximate.
    """
    if alpha < 0:
        raise NotImplementedError(
            f"elu_reach requires alpha >= 0 (the box/star reach assumes "
            f"ELU is monotone, true only for non-negative alpha). Got "
            f"alpha={alpha}. Non-monotone ELU reach is a follow-up; see "
            f"PR12_FIX_LIST T4 polish."
        )


def elu_box(input_boxes: List[Box], alpha: float = 1.0) -> List[Box]:
    """ELU is monotone non-decreasing (for alpha >= 0); apply directly to bounds."""
    _check_alpha(alpha)
    return [Box(_elu(b.lb, alpha), _elu(b.ub, alpha)) for b in input_boxes]


def elu_star_approx(input_stars: List[Star], alpha: float = 1.0) -> List[Star]:
    """Box-lifted Star over-approximation, preserving ImageStar shape."""
    _check_alpha(alpha)

    def _box(lb: np.ndarray, ub: np.ndarray):
        return (
            _elu(lb.flatten(), alpha).reshape(-1, 1),
            _elu(ub.flatten(), alpha).reshape(-1, 1),
        )

    return apply_box_lift_star(input_stars, _box)
