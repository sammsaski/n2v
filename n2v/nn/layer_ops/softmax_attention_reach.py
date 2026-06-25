"""Sound softmax reachability for transformer attention weights.

The attention weights are ``A = softmax(S)`` over the key axis. This module
provides the **correlated row-softmax interval bound**, which is the exact
per-element range of ``A_j`` over a box of logits and is the tightest sound box
transformer (it is also the IBP bound used for non-regression intersection with
the symbolic Star path).

Monotonicity of softmax makes the bound exact for a box input:
``dA_j/dS_j = A_j(1-A_j) >= 0`` (increasing in the own logit) and
``dA_j/dS_k = -A_j A_k <= 0`` for ``k != j`` (decreasing in every other logit).
So ``A_j`` is maximised at ``S_j = s_hi_j`` with all other ``S_k = s_lo_k``, and
minimised at ``S_j = s_lo_j`` with all other ``S_k = s_hi_k`` -- independent
coordinates, so those corners are jointly attainable. See
``docs/theory/sound-vit-reach.md`` SS3.3.

The symbolic exp/rowsum/reciprocal/normalize Star construction (Slice 1, the
precision win) is a separate follow-up; this module deliberately ships only the
sound bound. ``softmax_attn_star`` here returns the box-precision enclosure
(``Star.from_bounds`` of the correlated row bound) -- sound, and exactly the
non-regression fallback the design intersects the symbolic path with. It does
NOT pretend to keep predicate correlation.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from n2v.sets import Star, Zono, Box


def _softmax_last(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def correlated_row_softmax_bounds(
    s_lo: np.ndarray, s_hi: np.ndarray, axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact per-element range of ``softmax(S)`` over the logit box ``[s_lo,s_hi]``.

    Returns (A_lb, A_ub), same shape as the inputs, with ``0 <= A_lb <= A_ub <=
    1``. For each index ``j`` along ``axis``:
        A_ub_j = softmax(s_lo everywhere, s_hi at j)_j
        A_lb_j = softmax(s_hi everywhere, s_lo at j)_j
    """
    s_lo = np.moveaxis(np.asarray(s_lo, dtype=np.float64), axis, -1)
    s_hi = np.moveaxis(np.asarray(s_hi, dtype=np.float64), axis, -1)
    if np.any(s_hi < s_lo - 1e-12):
        raise ValueError("softmax bounds: s_hi < s_lo somewhere")
    n = s_lo.shape[-1]
    a_lb = np.empty_like(s_lo)
    a_ub = np.empty_like(s_hi)
    for j in range(n):
        m_up = s_lo.copy()
        m_up[..., j] = s_hi[..., j]
        a_ub[..., j] = _softmax_last(m_up)[..., j]
        m_lo = s_hi.copy()
        m_lo[..., j] = s_lo[..., j]
        a_lb[..., j] = _softmax_last(m_lo)[..., j]
    # Clamp into [0,1] to absorb FP (the true values are already in range).
    a_lb = np.clip(a_lb, 0.0, 1.0)
    a_ub = np.clip(a_ub, 0.0, 1.0)
    return np.moveaxis(a_lb, -1, axis), np.moveaxis(a_ub, -1, axis)


def _resolve_axis(axis: int, ndim: int) -> int:
    a = axis if axis >= 0 else axis + ndim
    if not (0 <= a < ndim):
        raise ValueError(f"softmax axis {axis} out of range for ndim {ndim}")
    return a


def _bounds_to_flat(
    s_lo: np.ndarray, s_hi: np.ndarray,
    shape: Sequence[int], axis: int,
) -> Tuple[np.ndarray, np.ndarray]:
    shape = tuple(int(s) for s in shape)
    if int(np.prod(shape)) != s_lo.size:
        raise ValueError(
            f"softmax shape {shape} (prod={int(np.prod(shape))}) does not match "
            f"logit dim {s_lo.size}")
    a = _resolve_axis(axis, len(shape))
    a_lb, a_ub = correlated_row_softmax_bounds(
        s_lo.reshape(shape), s_hi.reshape(shape), axis=a)
    return a_lb.reshape(-1), a_ub.reshape(-1)


def softmax_attn_box(
    input_boxes: List[Box], shape: Sequence[int], axis: int = -1,
) -> List[Box]:
    """Sound softmax over ``axis`` for Box logits (exact correlated row bound)."""
    out: List[Box] = []
    for box in input_boxes:
        a_lb, a_ub = _bounds_to_flat(
            np.asarray(box.lb).reshape(-1), np.asarray(box.ub).reshape(-1),
            shape, axis)
        out.append(Box(a_lb, a_ub))
    return out


def softmax_attn_zono(
    input_zonos: List[Zono], shape: Sequence[int], axis: int = -1,
) -> List[Zono]:
    """Sound softmax for Zono logits via the correlated row bound (box-encoded)."""
    out: List[Zono] = []
    for z in input_zonos:
        lo, hi = z.get_bounds()
        a_lb, a_ub = _bounds_to_flat(
            np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1), shape, axis)
        out.append(Zono.from_bounds(a_lb, a_ub))
    return out


def softmax_attn_star(
    input_stars: List[Star], shape: Sequence[int], axis: int = -1,
    lp_solver: str = "default", bounds: str = "estimate",
) -> List[Star]:
    """Sound softmax for Star logits (box-precision enclosure).

    Concretises logits to their LP-exact per-dim range, applies the exact
    correlated row bound, and re-encodes via ``Star.from_bounds``. Sound; this
    is the non-regression fallback that the symbolic Star softmax (Slice 1) will
    intersect with. It does not preserve predicate correlation.
    """
    out: List[Star] = []
    for star in input_stars:
        if bounds == "lp":
            lo, hi = star.get_ranges(lp_solver=lp_solver)
        elif bounds == "estimate":
            lo, hi = star.estimate_ranges()
        else:
            raise ValueError(f"unknown bounds mode {bounds!r}")
        a_lb, a_ub = _bounds_to_flat(
            np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1), shape, axis)
        out.append(Star.from_bounds(a_lb, a_ub))
    return out
