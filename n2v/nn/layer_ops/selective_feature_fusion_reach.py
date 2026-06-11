"""SelectiveFeatureFusion reachability.

SFF computes a sigmoid gate from concatenated inputs, then blends:
``y = g * a + (1 - g) * b`` with ``g = sigmoid(W [a; b])``.

Sound Box reach exploits ``g in [0, 1]``: the output of each entry
lies in the convex hull of the corresponding ``(a_i, b_i)`` intervals,
i.e. ``[min(a_lb, b_lb), max(a_ub, b_ub)]``.

Coverage per nnVLA: Box only.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box


def selective_feature_fusion_box(
    input_boxes: List[Box], extras: List[List[Box]]
) -> List[Box]:
    if not extras:
        return input_boxes
    out: List[Box] = []
    for i, a in enumerate(input_boxes):
        b = extras[0][i]
        lb = np.minimum(a.lb, b.lb)
        ub = np.maximum(a.ub, b.ub)
        out.append(Box(lb, ub))
    return out
