"""Sound reachability driver for the VNN-COMP 2023 ViT benchmark.

Sequences n2v's sound reach ops directly over the ViT_BN computation graph
(bypassing the fx tracer, which onnx2torch cannot feed for these models). Every
step is a sound over-approximation, so the final logit set encloses all
reachable outputs; an instance is VERIFIED robust iff the margin
``Y_label - Y_i`` has a positive lower bound for every i != label.

This is the Star pipeline. Affine ops (conv, BatchNorm, Linear, residual,
mean-pool, head) are exact and predicate-preserving; the attention bilinears
(QK^T, A.V) and softmax are sound relaxations. In the default "concretize" mode
the bilinear/softmax steps box-lift (matching CROWN's concretisation of the
attention weights) -- sound, ~IBP precision. Symbolic (predicate-preserving)
attention is the precision upgrade (design doc Slices 1-2).

Multi-head reshape is handled as exact row-permutations of the token-major
state, so all heads run in single batched matmul calls.
"""

from typing import Tuple

import numpy as np

from n2v.sets import Star, ImageStar
from n2v.nn.layer_ops.conv2d_reach import conv2d_star
from n2v.nn.layer_ops.relu_reach import relu_star_approx
from n2v.nn.layer_ops.bilinear_matmul_reach import (
    bilinear_matmul_star, av_envelope_star)
from n2v.nn.layer_ops.softmax_attention_reach import softmax_attn_star
from n2v.nn.reach import _add_sets


def _prefix_add(x: Star, full: Star) -> Star:
    """Sound + exact residual ``x + full`` when x's predicates are a PREFIX of
    full's (the attention/MLP case: the branch only appends relaxation
    predicates to the shared stream). Pads x's generators with zeros for the
    appended predicates and adds over full's (superset) constraint system. This
    keeps the residual correlation that a block-diagonal join drops (design
    SS6.2 / I-35). Falls back to the caller for non-prefix operands."""
    nx, nf = x.nVar, full.nVar
    if nx > nf or x.dim != full.dim:
        raise ValueError("prefix_add: x is not a prefix-compatible operand")
    Vx = np.zeros((x.dim, 1 + nf))
    Vx[:, :1 + nx] = x.V
    return Star(Vx + full.V, full.C, full.d, full.predicate_lb, full.predicate_ub)


def _lin(layer) -> Tuple[np.ndarray, np.ndarray]:
    W = layer.weight.detach().cpu().numpy().astype(np.float64)
    b = layer.bias.detach().cpu().numpy().astype(np.float64)
    return W, b


def _per_token_affine(star: Star, W: np.ndarray, b: np.ndarray, L: int) -> Star:
    """Apply the same affine map ``W x + b`` to each of the L tokens of a
    token-major (L*din,) state. Exact, predicate-preserving."""
    full = np.kron(np.eye(L), W)
    bias = np.tile(b, L).reshape(-1, 1)
    return star.affine_map(full, bias)


def _permute_rows(star: Star, shape, axes) -> Star:
    """Reorder the state rows as if reshaping to ``shape`` and transposing by
    ``axes`` (exact: a permutation, predicate system unchanged)."""
    idx = np.arange(int(np.prod(shape))).reshape(shape).transpose(axes).reshape(-1)
    return Star(star.V[idx, :].copy(), star.C, star.d,
                star.predicate_lb, star.predicate_ub)


def _prepend_const_token(star: Star, cls_vec: np.ndarray) -> Star:
    """Prepend a constant token (zero generators) to a token-major state."""
    dim = cls_vec.size
    old_dim, ncol = star.V.shape
    newV = np.zeros((dim + old_dim, ncol))
    newV[:dim, 0] = cls_vec.reshape(-1)
    newV[dim:, :] = star.V
    return Star(newV, star.C, star.d, star.predicate_lb, star.predicate_ub)


class ViTReacher:
    def __init__(self, model, mode: str = "concretize", relax_factor: float = 0.5):
        self.m = model
        self.mode = mode
        self.relax_factor = relax_factor

    def _attn(self, h: Star, attn, L: int, dim: int) -> Star:
        H, D = attn.heads, attn.dim_head
        scale = float(attn.scale)
        Wq, bq = _lin(attn.query)
        Wk, bk = _lin(attn.key)
        Wv, bv = _lin(attn.value)
        q = _per_token_affine(h, Wq, bq, L)   # (L*inner,)
        k = _per_token_affine(h, Wk, bk, L)
        v = _per_token_affine(h, Wv, bv, L)
        # token-major (t,h,d) -> per-head batched layouts (exact permutations)
        qp = _permute_rows(q, (L, H, D), (1, 0, 2))   # (H,L,D)
        vp = _permute_rows(v, (L, H, D), (1, 0, 2))   # (H,L,D)
        kT = _permute_rows(k, (L, H, D), (1, 2, 0))   # (H,D,L)
        S = bilinear_matmul_star([qp], [kT], (H, L, D), (H, D, L),
                                 scale=scale, mode="concretize")[0]   # (H,L,L)
        A = softmax_attn_star([S], (H, L, L), axis=-1)[0]            # (H,L,L)
        if self.mode.startswith("symbolic"):
            # Keep the value path symbolic: O affine in V (input-correlated).
            a_lb, a_ub = (b.reshape(H, L, L) for b in A.estimate_ranges())
            O = av_envelope_star(a_lb, a_ub, vp, H, L, L, D)         # (H,L,D)
        else:
            O = bilinear_matmul_star([A], [vp], (H, L, L), (H, L, D),
                                     mode="concretize")[0]
        Ob = _permute_rows(O, (H, L, D), (1, 0, 2))   # -> (L,inner) token-major
        Wo, bo = _lin(attn.out)
        return _per_token_affine(Ob, Wo, bo, L)

    def _ff(self, h: Star, ff, L: int) -> Star:
        Wl0, bl0 = _lin(ff.l0)
        z = _per_token_affine(h, Wl0, bl0, L)
        z = relu_star_approx([z], relax_factor=self.relax_factor)[0]
        Wl3, bl3 = _lin(ff.l3)
        return _per_token_affine(z, Wl3, bl3, L)

    def _add(self, x: Star, branch: Star) -> Star:
        """Residual x + branch. In symbolic mode use the provenance-aware
        prefix-aligned add (keeps stream correlation); else the sound
        block-diagonal join."""
        if self.mode.startswith("symbolic"):
            try:
                return _prefix_add(x, branch)
            except ValueError:
                pass
        return _add_sets([x], [branch], "add")[0]

    def _block(self, x: Star, blk, L: int, dim: int) -> Star:
        sc, sh = blk.norm_attn.affine()
        h = _per_token_affine(x, np.diag(sc), sh, L)
        a = self._attn(h, blk.attn, L, dim)
        x = self._add(x, a)
        sc2, sh2 = blk.norm_ff.affine()
        h2 = _per_token_affine(x, np.diag(sc2), sh2, L)
        f = self._ff(h2, blk.ff, L)
        x = self._add(x, f)
        return x

    def reach(self, input_imagestar: ImageStar) -> Star:
        m = self.m
        cs = conv2d_star(m.projection, [input_imagestar])[0]   # ImageStar
        Hp, Wp, dim = cs.height, cs.width, cs.num_channels
        star = cs.to_star()                                    # (Lp*dim,) token-major
        star = _prepend_const_token(star, m.cls_token.detach().cpu().numpy())
        L = Hp * Wp + 1
        pos = m.positions.detach().cpu().numpy().reshape(-1, 1)
        star = star.affine_map(np.eye(L * dim), pos)           # + positions
        for blk in m.blocks:
            star = self._block(star, blk, L, dim)
        # ReduceMean over tokens
        P = np.zeros((dim, L * dim))
        for t in range(L):
            P[np.arange(dim), t * dim + np.arange(dim)] = 1.0 / L
        star = star.affine_map(P)
        sc, sh = m.head_norm.affine()
        star = star.affine_map(np.diag(sc), sh.reshape(-1, 1))
        Wh, bh = _lin(m.head)
        return star.affine_map(Wh, bh.reshape(-1, 1))          # (10,)

    def margins(self, out: Star, label: int, margin_mode: str = "estimate",
                lp_solver: str = "default") -> dict:
        """Per-class lower bounds on the margin ``Y_label - Y_i``.

        ``estimate`` (default): predicate-box estimate of the margin direction --
        pure numpy, a sound lower bound (ignores constraint rows, so >= the LP
        value), fast even for huge stars. ``lp``: exact ``get_min`` over the full
        predicate polytope -- tighter, but expensive when nVar/nC are large.
        """
        K = out.dim
        out_margins = {}
        for i in range(K):
            if i == label:
                continue
            c = np.zeros((1, K))
            c[0, label] = 1.0
            c[0, i] = -1.0
            mstar = out.affine_map(c)
            if margin_mode == "lp":
                mlb = mstar.get_min(0, lp_solver=lp_solver)
            else:
                lb, _ = mstar.estimate_ranges()
                mlb = float(lb.reshape(-1)[0])
            out_margins[i] = float(mlb) if mlb is not None else None
        return out_margins

    def verify(self, lb_chw: np.ndarray, ub_chw: np.ndarray, label: int,
               margin_mode: str = "estimate", lp_solver: str = "default") -> dict:
        """Reach the logit set for the input box and check robustness margins.

        lb_chw/ub_chw: (3,H,W) normalized lower/upper bounds. Returns status in
        {'verified','unknown'} (verified iff every margin lower bound > 0).
        """
        H, W = lb_chw.shape[1], lb_chw.shape[2]
        # ImageStar is (H,W,C); transpose from (C,H,W)
        lb_hwc = np.transpose(lb_chw, (1, 2, 0)).reshape(-1)
        ub_hwc = np.transpose(ub_chw, (1, 2, 0)).reshape(-1)
        img = ImageStar.from_bounds(lb_hwc, ub_hwc, H, W, lb_chw.shape[0])
        out = self.reach(img)
        m = self.margins(out, label, margin_mode=margin_mode, lp_solver=lp_solver)
        verified = all(v is not None and v > 0 for v in m.values())
        return {"status": "verified" if verified else "unknown",
                "margins": m, "out_star": out}
