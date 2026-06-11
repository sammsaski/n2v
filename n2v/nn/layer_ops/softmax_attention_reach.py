"""SoftmaxAttention reachability.

For QKV inputs ``q in R^{L_q x d}``, ``k in R^{L_k x d}``, ``v in R^{L_k x d}``::

    A = softmax(q @ k.T / sqrt(d))   # row-stochastic, entries in [0, 1]
    y = A @ v

Sound Box reach exploits ``A_ij in [0, 1]`` and ``sum_j A_ij = 1``: the
output of row ``i`` lies in the convex hull of the value rows (which is
contained in the per-column box bounds of ``v``).

For Star, the same box-hull argument lifts to a Star via
``Star.from_bounds``. Tighter Star reach following nnVLA's
``softmax_attention/methods/star.py`` is research-grade and remains a
TODO (tracked in the draft PR description).

Coverage matches nnVLA: Box (sound), Star (sound, box-lifted approx).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar


def _row_attention_box(v_lb: np.ndarray, v_ub: np.ndarray, l_q: int) -> tuple[np.ndarray, np.ndarray]:
    """Bound y_i for each query row i by the column-wise min/max of v."""
    col_lb = v_lb.min(axis=0, keepdims=True)
    col_ub = v_ub.max(axis=0, keepdims=True)
    out_lb = np.tile(col_lb, (l_q, 1))
    out_ub = np.tile(col_ub, (l_q, 1))
    return out_lb, out_ub


def softmax_attention_box(
    q_box: List[Box], k_box: List[Box], v_box: List[Box], l_q: int, d_v: int
) -> List[Box]:
    out: List[Box] = []
    for q, k, v in zip(q_box, k_box, v_box):
        v_lb = v.lb.reshape(-1, d_v)
        v_ub = v.ub.reshape(-1, d_v)
        out_lb, out_ub = _row_attention_box(v_lb, v_ub, l_q)
        out.append(Box(out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)))
    return out


def softmax_attention_star_approx(
    q_star: List[Star], k_star: List[Star], v_star: List[Star], l_q: int, d_v: int
) -> List[Star]:
    """Box-lifted Star reach.

    Output shape ``(l_q, d_v)`` matches V's per-row layout. If V was an
    ImageStar (e.g. when QKV come from a patch-embed pipeline) the
    output is wrapped back as an ImageStar with the original ``(H, W,
    C)`` so downstream image-aware layers see consistent metadata.
    """
    out: List[Star] = []
    for q, k, v in zip(q_star, k_star, v_star):
        is_image = isinstance(v, ImageStar)
        base_v = v.to_star() if is_image else v
        v_lb, v_ub = base_v.estimate_ranges()
        v_lb = v_lb.reshape(-1, d_v)
        v_ub = v_ub.reshape(-1, d_v)
        out_lb, out_ub = _row_attention_box(v_lb, v_ub, l_q)
        new_star = Star.from_bounds(out_lb.reshape(-1, 1), out_ub.reshape(-1, 1))
        if is_image:
            new_star = new_star.to_image_star(v.height, v.width, v.num_channels)
        out.append(new_star)
    return out
