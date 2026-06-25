"""Bilinear (set @ set) matmul reachability for transformer attention.

n2v's ``_handle_onnx_matmul`` handles ``set @ constant`` exactly (an affine
map). This module handles the harder ``set @ set`` case, where *both* operands
carry uncertainty -- the two bilinear products inside self-attention:

* scores ``S = scale * (Q @ K^T)``  (Q, K both depend on the input), and
* output ``O = A @ V``              (A, V both depend on the input).

Both are bilinear, hence not affine, so the result is a *sound
over-approximation*, never the exact image.

Mode ``"concretize"`` (default) takes per-dimension interval bounds of each
operand and applies a sound interval matmul (Rump midpoint--radius form), then
re-encodes the result as a fresh box-set. This matches the CROWN strategy of
concretising the bilinear term into an interval: it is sound for operands of
arbitrary sign and drops only the operand cross-correlation (an
over-approximation, never an under-approximation). See
``docs/theory/sound-vit-reach.md`` SS3.2/SS3.4/SS5.

Mode ``"mccormick"`` (symbolic, predicate-preserving) is the precision lever
tracked as Slice 2 in the design doc and is not implemented yet; it raises
rather than silently falling back, to avoid any ambiguity about what was
computed.

Conventions: operands are passed as lists of sets (n2v carries a list to
support branching) together with their *tensor* shapes (the flat set dimension
reshaped row-major). Standard batched-matmul semantics apply: the last two axes
are the matrix dims and any leading axes broadcast, exactly like ``np.matmul``.
"""

from typing import List, Sequence, Tuple

import numpy as np

from n2v.sets import Star, Zono, Box


def _as_flat(lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lb, ub) as 1-D float64 arrays regardless of (n,) vs (n, 1)."""
    return (np.asarray(lb, dtype=np.float64).reshape(-1),
            np.asarray(ub, dtype=np.float64).reshape(-1))


def _star_bounds(star, bounds: str, lp_solver: str):
    """Per-dim bounds of a Star. ``bounds='lp'`` is exact (``get_ranges``);
    ``bounds='estimate'`` uses the predicate-box estimate -- pure numpy, a sound
    over-approximation in general and *exact* when the star has no constraint
    rows (the usual case in the concretise pipeline before any relaxation).
    Both are sound (the estimate is a superset)."""
    if bounds == "lp":
        return _as_flat(*star.get_ranges(lp_solver=lp_solver))
    if bounds == "estimate":
        return _as_flat(*star.estimate_ranges())
    raise ValueError(f"unknown bounds mode {bounds!r}")


def _interval_matmul(
    al: np.ndarray, au: np.ndarray, bl: np.ndarray, bu: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sound interval matmul ``C = A @ B`` (Rump midpoint--radius form).

    For ``A in [al, au]``, ``B in [bl, bu]`` (batched over leading axes):

        Am = (al+au)/2,  Ar = (au-al)/2,   Bm, Br likewise
        Cm = Am @ Bm
        Cr = |Am| @ Br + Ar @ |Bm| + Ar @ Br
        C  in [Cm - Cr, Cm + Cr]

    Soundness: each product term's deviation from ``Am*Bm`` is bounded by
    ``|Am|*Br + Ar*|Bm| + Ar*Br``; summing over the contraction index gives
    ``Cr``. The enclosure therefore contains every ``A@B`` with ``A in [al,au]``,
    ``B in [bl,bu]``. Shapes follow ``np.matmul`` broadcasting.
    """
    am = 0.5 * (al + au)
    ar = 0.5 * (au - al)
    bm = 0.5 * (bl + bu)
    br = 0.5 * (bu - bl)
    cm = am @ bm
    cr = np.abs(am) @ br + ar @ np.abs(bm) + ar @ br
    return cm - cr, cm + cr


def _scaled_bounds(
    cl: np.ndarray, cu: np.ndarray, scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multiply an interval by a (possibly negative) scalar, sign-safe."""
    p, q = scale * cl, scale * cu
    return np.minimum(p, q), np.maximum(p, q)


def _check_shapes(dim: int, shape: Sequence[int], which: str) -> Tuple[int, ...]:
    shape = tuple(int(s) for s in shape)
    if int(np.prod(shape)) != dim:
        raise ValueError(
            f"{which} shape {shape} (prod={int(np.prod(shape))}) does not match "
            f"set dim {dim}")
    if len(shape) < 2:
        raise ValueError(f"{which} shape {shape} must have >= 2 axes for matmul")
    return shape


def _concretize_bounds(
    left_lb, left_ub, right_lb, right_ub,
    left_shape, right_shape, scale,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared core: interval matmul of two operand boxes -> flat output box."""
    al = left_lb.reshape(left_shape)
    au = left_ub.reshape(left_shape)
    bl = right_lb.reshape(right_shape)
    bu = right_ub.reshape(right_shape)
    cl, cu = _interval_matmul(al, au, bl, bu)
    cl, cu = _scaled_bounds(cl, cu, scale)
    return cl.reshape(-1), cu.reshape(-1)


def _reject_unimplemented_mode(mode: str) -> None:
    if mode == "mccormick":
        raise NotImplementedError(
            "bilinear_matmul mode='mccormick' (symbolic, predicate-preserving) "
            "is not implemented yet -- see docs/theory/sound-vit-reach.md Slice 2. "
            "Use mode='concretize' (the sound default).")
    if mode != "concretize":
        raise ValueError(f"unknown bilinear_matmul mode {mode!r}")


def bilinear_matmul_box(
    left_boxes: List[Box], right_boxes: List[Box],
    left_shape: Sequence[int], right_shape: Sequence[int],
    scale: float = 1.0, mode: str = "concretize",
) -> List[Box]:
    """Sound ``scale * (left @ right)`` for Box operands (interval matmul)."""
    _reject_unimplemented_mode(mode)
    if len(left_boxes) != len(right_boxes):
        raise ValueError(
            f"operand list lengths differ: {len(left_boxes)} vs "
            f"{len(right_boxes)}")
    out: List[Box] = []
    for lb_box, rb_box in zip(left_boxes, right_boxes):
        ls = _check_shapes(lb_box.dim, left_shape, "left")
        rs = _check_shapes(rb_box.dim, right_shape, "right")
        ll, lu = _as_flat(lb_box.lb, lb_box.ub)
        rl, ru = _as_flat(rb_box.lb, rb_box.ub)
        cl, cu = _concretize_bounds(ll, lu, rl, ru, ls, rs, scale)
        out.append(Box(cl, cu))
    return out


def bilinear_matmul_zono(
    left_zonos: List[Zono], right_zonos: List[Zono],
    left_shape: Sequence[int], right_shape: Sequence[int],
    scale: float = 1.0, mode: str = "concretize",
) -> List[Zono]:
    """Sound ``scale * (left @ right)`` for Zono operands.

    Zonotopes cannot represent the bilinear coupling, so we concretise each
    operand to its interval bounds (exact for a Zono), apply the sound interval
    matmul, and re-encode the result as a box-zonotope.
    """
    _reject_unimplemented_mode(mode)
    if len(left_zonos) != len(right_zonos):
        raise ValueError(
            f"operand list lengths differ: {len(left_zonos)} vs "
            f"{len(right_zonos)}")
    out: List[Zono] = []
    for lz, rz in zip(left_zonos, right_zonos):
        ls = _check_shapes(lz.dim, left_shape, "left")
        rs = _check_shapes(rz.dim, right_shape, "right")
        ll, lu = _as_flat(*lz.get_bounds())
        rl, ru = _as_flat(*rz.get_bounds())
        cl, cu = _concretize_bounds(ll, lu, rl, ru, ls, rs, scale)
        out.append(Zono.from_bounds(cl, cu))
    return out


def av_envelope_star(
    a_lb: np.ndarray, a_ub: np.ndarray, v_star: Star,
    H: int, M: int, K: int, D: int,
) -> Star:
    """Sound symbolic ``O = A @ V`` keeping V's predicates (value-path correlation).

    ``A`` is a concretized interval ``[a_lb, a_ub]`` of attention weights, shape
    ``(H, M, K)`` with ``a_lb >= 0`` (softmax invariant); ``v_star`` is a symbolic
    Star of dim ``H*K*D`` (row-major ``(H, K, D)``). Returns a Star of dim
    ``H*M*D`` (row-major ``(H, M, D)``) where each output is a fresh predicate
    constrained by the sign-aware A·V envelope (affine in V's predicates), so the
    output stays correlated with V (hence the network input). See
    ``docs/theory/sound-vit-reach.md`` SS3.4/SS6bis.
    """
    a_lb = np.asarray(a_lb, dtype=np.float64).reshape(H, M, K)
    a_ub = np.asarray(a_ub, dtype=np.float64).reshape(H, M, K)
    if np.any(a_lb < -1e-12):
        raise ValueError("av_envelope_star requires a_lb >= 0 (softmax invariant)")
    vlo, vhi = (b.reshape(H, K, D) for b in v_star.estimate_ranges())
    n_old = v_star.nVar
    cV = v_star.V[:, 0].reshape(H, K, D)
    GV = v_star.V[:, 1:].reshape(H, K, D, n_old)

    alb = a_lb[:, :, :, None]      # (H,M,K,1)
    aub = a_ub[:, :, :, None]
    vl = vlo[:, None, :, :]        # (H,1,K,D)
    vh = vhi[:, None, :, :]
    pos = vl >= 0
    neg = vh <= 0
    mixed = ~(pos | neg)
    width = np.where(mixed, vh - vl, 1.0)
    up_slope_mixed = (aub * vh - alb * vl) / width
    low_slope_mixed = (alb * vh - aub * vl) / width
    up_slope = np.where(pos, aub, np.where(neg, alb, up_slope_mixed))
    low_slope = np.where(pos, alb, np.where(neg, aub, low_slope_mixed))
    up_bias = np.where(mixed, alb * vl - up_slope_mixed * vl, 0.0)
    low_bias = np.where(mixed, aub * vl - low_slope_mixed * vl, 0.0)
    up_slope = np.maximum(up_slope, 0.0)      # FP guard (no-op when a_lb>=0)
    low_slope = np.maximum(low_slope, 0.0)

    n_out = H * M * D
    coef_up = np.einsum("hmkd,hkdn->hmdn", up_slope, GV).reshape(n_out, n_old)
    coef_low = np.einsum("hmkd,hkdn->hmdn", low_slope, GV).reshape(n_out, n_old)
    const_up = (np.einsum("hmkd,hkd->hmd", up_slope, cV)
                + up_bias.sum(axis=2)).reshape(-1)
    const_low = (np.einsum("hmkd,hkd->hmd", low_slope, cV)
                 + low_bias.sum(axis=2)).reshape(-1)
    o_lb = (np.einsum("hmkd,hkd->hmd", low_slope, vlo)
            + low_bias.sum(axis=2)).reshape(-1)
    o_ub = (np.einsum("hmkd,hkd->hmd", up_slope, vhi)
            + up_bias.sum(axis=2)).reshape(-1)

    V_out = np.zeros((n_out, 1 + n_old + n_out))
    V_out[:, 1 + n_old:] = np.eye(n_out)      # O_k = alpha_new_k

    old_C = np.asarray(v_star.C, dtype=np.float64).reshape(-1, n_old) \
        if np.asarray(v_star.C).size else np.zeros((0, n_old))
    old_d = np.asarray(v_star.d, dtype=np.float64).reshape(-1, 1) \
        if np.asarray(v_star.d).size else np.zeros((0, 1))
    nc = old_C.shape[0]
    I = np.eye(n_out)
    C = np.zeros((nc + 2 * n_out, n_old + n_out))
    C[:nc, :n_old] = old_C
    C[nc:nc + n_out, :n_old] = -coef_up        # O <= up facet
    C[nc:nc + n_out, n_old:] = I
    C[nc + n_out:, :n_old] = coef_low          # O >= low facet
    C[nc + n_out:, n_old:] = -I
    d = np.vstack([old_d, const_up.reshape(-1, 1), (-const_low).reshape(-1, 1)])

    def _pb(p, fill):
        return (np.asarray(p, dtype=np.float64).reshape(-1, 1) if p is not None
                else np.full((n_old, 1), fill))
    plb = np.vstack([_pb(v_star.predicate_lb, -np.inf), o_lb.reshape(-1, 1)])
    pub = np.vstack([_pb(v_star.predicate_ub, np.inf), o_ub.reshape(-1, 1)])
    return Star(V_out, C, d, plb, pub)


def bilinear_matmul_star(
    left_stars: List[Star], right_stars: List[Star],
    left_shape: Sequence[int], right_shape: Sequence[int],
    scale: float = 1.0, mode: str = "concretize",
    lp_solver: str = "default", bounds: str = "estimate",
) -> List[Star]:
    """Sound ``scale * (left @ right)`` for Star operands.

    In ``concretize`` mode the operands are reduced to LP-exact per-dimension
    interval bounds (``get_ranges``), combined with the sound interval matmul,
    and the result re-encoded as a box-star via ``Star.from_bounds``. This is
    the CROWN-style concretisation of the bilinear term: sound, and it drops
    only the operand cross-correlation. Downstream softmax keeps its own
    relaxation symbolic, so this is not a full per-block box-lift.
    """
    _reject_unimplemented_mode(mode)
    if len(left_stars) != len(right_stars):
        raise ValueError(
            f"operand list lengths differ: {len(left_stars)} vs "
            f"{len(right_stars)}")
    out: List[Star] = []
    for ls_star, rs_star in zip(left_stars, right_stars):
        ls = _check_shapes(ls_star.dim, left_shape, "left")
        rs = _check_shapes(rs_star.dim, right_shape, "right")
        ll, lu = _star_bounds(ls_star, bounds, lp_solver)
        rl, ru = _star_bounds(rs_star, bounds, lp_solver)
        cl, cu = _concretize_bounds(ll, lu, rl, ru, ls, rs, scale)
        out.append(Star.from_bounds(cl, cu))
    return out
