"""LinearAttention reachability — multi-input ``phi(q) @ (phi(k).T @ v)``.

The full linear-attention output is
``y = phi(q) @ (phi(k).T @ v)`` with ``phi(x) = elu(x) + 1`` (a
Performers-style positive kernel feature map). Reachability requires
bounds on all three of ``phi(q)``, ``phi(k).T @ v``, and their
composition.

The previous version of this module bounded only ``phi(x)`` for a
single input, which silently verifies the wrong operation. The
helpers here now bound the *attention output* itself, given Q/K/V
streams; they raise ``NotImplementedError`` for the legacy single-
input call shape rather than returning a misleading box.

Coverage matches nnVLA: Box, Star (box-lifted).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star


def _phi_box(box: Box) -> Box:
    """Bound ``phi(x) = elu(x) + 1`` elementwise. Always non-negative."""
    lb = np.where(box.lb >= 0, box.lb + 1.0, np.exp(np.minimum(box.lb, 0.0)))
    ub = np.where(box.ub >= 0, box.ub + 1.0, np.exp(np.minimum(box.ub, 0.0)))
    return Box(lb, ub)


def linear_attention_box(
    q_box: List[Box],
    k_box: List[Box] | None = None,
    v_box: List[Box] | None = None,
    l_q: int = 1,
    d_v: int = 1,
) -> List[Box]:
    """Sound box reach for ``phi(q) @ (phi(k).T @ v)``.

    Q/K/V are passed as separate set lists; each entry shaped per-
    token (L * d_head). When K/V aren't given (legacy single-input
    call), we raise to avoid the previous unsound shortcut.
    """
    if k_box is None or v_box is None:
        raise NotImplementedError(
            "LinearAttention reach requires Q/K/V streams. The previous "
            "single-input shortcut was unsound — pass k_box and v_box."
        )
    out: List[Box] = []
    for q, k, v in zip(q_box, k_box, v_box):
        phi_q = _phi_box(q)
        phi_k = _phi_box(k)
        # Bound (phi(k).T @ v) via interval matmul, treating each as
        # a flattened (L_k, d) matrix.
        v_lb = v.lb.reshape(-1, d_v)
        v_ub = v.ub.reshape(-1, d_v)
        phi_k_lb = phi_k.lb.reshape(-1, d_v)
        phi_k_ub = phi_k.ub.reshape(-1, d_v)
        # T1-9 (audit C7): SOUND four-corner interval product. For each
        # term phi(k)_{ij} * v_{ij} the bound is
        #   [min, max] over the four corners
        #     {phi_lb * v_lb, phi_lb * v_ub, phi_ub * v_lb, phi_ub * v_ub}
        # then summed over i. The previous version used only two corners
        # (phi_lb * v_lb, phi_ub * v_ub) which is sound ONLY when v is
        # non-negative; for negative v the true minimum corner is
        # phi_ub * v_lb, which was DROPPED -> reach EXCLUDED true outputs.
        # Counterexample (audit C7): phi_k in [1,3], v in [-2,1] ->
        # true kv range [-6, 3], previous code returned [-2, 3] (missing
        # the true minimum -6). The four-corner formula correctly returns
        # min(1*-2, 1*1, 3*-2, 3*1)=-6 and max=3.
        c0 = phi_k_lb * v_lb
        c1 = phi_k_lb * v_ub
        c2 = phi_k_ub * v_lb
        c3 = phi_k_ub * v_ub
        elementwise_lb = np.minimum(np.minimum(c0, c1), np.minimum(c2, c3))
        elementwise_ub = np.maximum(np.maximum(c0, c1), np.maximum(c2, c3))
        kv_lb = elementwise_lb.sum(axis=0)
        kv_ub = elementwise_ub.sum(axis=0)
        # phi(q) @ kv (broadcast over l_q): per-row of phi(q) we get a
        # length-d_v output. Conservatively bound by per-feature extremes.
        phi_q_lb = phi_q.lb.reshape(-1, d_v)
        phi_q_ub = phi_q.ub.reshape(-1, d_v)
        # Conservative product of axis-aligned intervals.
        cands = np.stack([
            phi_q_lb * kv_lb, phi_q_lb * kv_ub,
            phi_q_ub * kv_lb, phi_q_ub * kv_ub,
        ])
        out_lb = cands.min(axis=0).reshape(-1, 1)
        out_ub = cands.max(axis=0).reshape(-1, 1)
        out.append(Box(out_lb, out_ub))
    return out


def linear_attention_star_approx(
    q_star: List[Star],
    k_star: List[Star] | None = None,
    v_star: List[Star] | None = None,
    l_q: int = 1,
    d_v: int = 1,
) -> List[Star]:
    """Box-lifted Star reach for the full attention output."""
    if k_star is None or v_star is None:
        raise NotImplementedError(
            "LinearAttention reach requires Q/K/V streams. Pass k_star and v_star."
        )
    # Reduce each Star to its IBP envelope, run the Box path, lift back.
    def _to_box(s: Star) -> Box:
        lb, ub = s.estimate_ranges()
        return Box(lb, ub)

    q_boxes = [_to_box(s) for s in q_star]
    k_boxes = [_to_box(s) for s in k_star]
    v_boxes = [_to_box(s) for s in v_star]
    box_out = linear_attention_box(q_boxes, k_boxes, v_boxes, l_q=l_q, d_v=d_v)
    return [Star.from_bounds(b.lb, b.ub) for b in box_out]
