"""Sound reachability driver for the VNN-COMP 2023 ViT benchmark.

Sequences n2v's sound reach ops directly over the ViT_BN computation graph
(bypassing the fx tracer, which onnx2torch cannot feed for these models). Every
step is a sound over-approximation, so the final logit set encloses all
reachable outputs; an instance is VERIFIED robust iff the margin
``Y_label - Y_i`` has a positive lower bound for every i != label.

Generic over the set representation (``set_type`` in {Star, Zono, Box}) for
coverage: affine ops (conv, BatchNorm, Linear, residual, mean-pool, head) are
exact/predicate-preserving; attention bilinears (QK^T, A.V) and softmax are
sound relaxations. The ``symbolic-av`` mode (Star only) keeps the attention
value path + residual stream input-correlated (CROWN-class); ``concretize`` (all
types) box-lifts the attention output (IBP-class). Box/Zono are sound but looser
than Star; they exist for completeness of layer coverage.

Multi-head reshape is handled as exact row-permutations of the token-major
state, so all heads run in single batched matmul calls.
"""

from typing import Tuple

import numpy as np

from n2v.sets import Star, Zono, Box, ImageStar, ImageZono
from n2v.nn.layer_ops.conv2d_reach import conv2d_star, conv2d_zono
from n2v.nn.layer_ops.relu_reach import (
    relu_star_approx, relu_zono_approx, relu_box)
from n2v.nn.layer_ops.bilinear_matmul_reach import (
    bilinear_matmul_star, bilinear_matmul_zono, bilinear_matmul_box,
    av_envelope_star)
from n2v.nn.layer_ops.softmax_attention_reach import (
    softmax_attn_star, softmax_attn_zono, softmax_attn_box)
from n2v.nn.reach import _add_sets


def _lin(layer) -> Tuple[np.ndarray, np.ndarray]:
    W = layer.weight.detach().cpu().numpy().astype(np.float64)
    b = layer.bias.detach().cpu().numpy().astype(np.float64)
    return W, b


def _prefix_add(x: Star, full: Star) -> Star:
    """Sound + exact residual ``x + full`` when x's predicates are a PREFIX of
    full's (the attention/MLP case: the branch only appends relaxation
    predicates to the shared stream). Keeps the residual correlation that a
    block-diagonal join drops (design SS6.2 / I-35)."""
    nx, nf = x.nVar, full.nVar
    if nx > nf or x.dim != full.dim:
        raise ValueError("prefix_add: x is not a prefix-compatible operand")
    Vx = np.zeros((x.dim, 1 + nf))
    Vx[:, :1 + nx] = x.V
    return Star(Vx + full.V, full.C, full.d, full.predicate_lb, full.predicate_ub)


def _permute_rows(s, shape, axes):
    """Reorder state rows as if reshaping to ``shape`` and transposing by
    ``axes`` (exact permutation; set-type aware)."""
    idx = np.arange(int(np.prod(shape))).reshape(shape).transpose(axes).reshape(-1)
    if isinstance(s, Star):
        return Star(s.V[idx, :].copy(), s.C, s.d, s.predicate_lb, s.predicate_ub)
    if isinstance(s, Zono):
        return Zono(s.c.reshape(-1, 1)[idx].copy(), s.V[idx, :].copy())
    # Box
    lb = np.asarray(s.lb).reshape(-1)[idx]
    ub = np.asarray(s.ub).reshape(-1)[idx]
    return Box(lb.copy(), ub.copy())


def _prepend_const_token(s, cls_vec: np.ndarray):
    """Prepend a constant token (zero uncertainty) to a token-major state."""
    cls_vec = np.asarray(cls_vec, dtype=np.float64).reshape(-1)
    d = cls_vec.size
    if isinstance(s, Star):
        old_dim, ncol = s.V.shape
        V = np.zeros((d + old_dim, ncol))
        V[:d, 0] = cls_vec
        V[d:, :] = s.V
        return Star(V, s.C, s.d, s.predicate_lb, s.predicate_ub)
    if isinstance(s, Zono):
        c = np.vstack([cls_vec.reshape(-1, 1), s.c.reshape(-1, 1)])
        V = np.vstack([np.zeros((d, s.V.shape[1])), s.V])
        return Zono(c, V)
    lb = np.concatenate([cls_vec, np.asarray(s.lb).reshape(-1)])
    ub = np.concatenate([cls_vec, np.asarray(s.ub).reshape(-1)])
    return Box(lb, ub)


class ViTReacher:
    def __init__(self, model, mode: str = "concretize", relax_factor: float = 0.5,
                 set_type=Star):
        self.m = model
        self.mode = mode
        self.relax_factor = relax_factor
        self.set_type = set_type

    # ---- set-type-generic primitives ----
    def _bilinear(self, left, right, lshape, rshape, scale=1.0):
        if self.set_type is Star:
            return bilinear_matmul_star([left], [right], lshape, rshape,
                                        scale=scale, mode="concretize")[0]
        if self.set_type is Zono:
            return bilinear_matmul_zono([left], [right], lshape, rshape, scale=scale)[0]
        return bilinear_matmul_box([left], [right], lshape, rshape, scale=scale)[0]

    def _softmax(self, s, shape, axis=-1):
        if self.set_type is Star:
            return softmax_attn_star([s], shape, axis)[0]
        if self.set_type is Zono:
            return softmax_attn_zono([s], shape, axis)[0]
        return softmax_attn_box([s], shape, axis)[0]

    def _relu(self, s):
        if self.set_type is Star:
            return relu_star_approx([s], relax_factor=self.relax_factor)[0]
        if self.set_type is Zono:
            return relu_zono_approx([s])[0]
        return relu_box([s])[0]

    def _add(self, x, branch):
        if self.set_type is Star and self.mode.startswith("symbolic"):
            try:
                return _prefix_add(x, branch)
            except ValueError:
                pass
        return _add_sets([x], [branch], "add")[0]

    def _patch_embed(self, lb_chw, ub_chw):
        """Conv patch-embed -> token-major flat set of self.set_type."""
        H, W, Cin = lb_chw.shape[1], lb_chw.shape[2], lb_chw.shape[0]
        lb_hwc = np.transpose(lb_chw, (1, 2, 0)).reshape(-1)
        ub_hwc = np.transpose(ub_chw, (1, 2, 0)).reshape(-1)
        if self.set_type is Zono:
            iz = ImageZono.from_bounds(lb_hwc, ub_hwc, H, W, Cin)
            cz = conv2d_zono(self.m.projection, [iz])[0]
            return cz.to_zono(), cz.height, cz.width, cz.num_channels
        # Star and Box both route the (exact) conv through ImageStar.
        ist = ImageStar.from_bounds(lb_hwc, ub_hwc, H, W, Cin)
        cs = conv2d_star(self.m.projection, [ist])[0]
        star = cs.to_star()
        if self.set_type is Box:
            lb, ub = star.estimate_ranges()   # exact interval (input box -> affine conv)
            return Box(lb.reshape(-1), ub.reshape(-1)), cs.height, cs.width, cs.num_channels
        return star, cs.height, cs.width, cs.num_channels

    # ---- ViT blocks ----
    def _attn(self, h, attn, L: int, dim: int):
        H, D = attn.heads, attn.dim_head
        scale = float(attn.scale)
        q = _per_token_affine(h, *_lin(attn.query), L)
        k = _per_token_affine(h, *_lin(attn.key), L)
        v = _per_token_affine(h, *_lin(attn.value), L)
        qp = _permute_rows(q, (L, H, D), (1, 0, 2))   # (H,L,D)
        vp = _permute_rows(v, (L, H, D), (1, 0, 2))
        kT = _permute_rows(k, (L, H, D), (1, 2, 0))   # (H,D,L)
        S = self._bilinear(qp, kT, (H, L, D), (H, D, L), scale=scale)   # (H,L,L)
        A = self._softmax(S, (H, L, L), axis=-1)
        if self.set_type is Star and self.mode.startswith("symbolic"):
            a_lb, a_ub = (b.reshape(H, L, L) for b in A.estimate_ranges())
            O = av_envelope_star(a_lb, a_ub, vp, H, L, L, D)
        else:
            O = self._bilinear(A, vp, (H, L, L), (H, L, D))
        Ob = _permute_rows(O, (H, L, D), (1, 0, 2))   # -> (L,inner)
        return _per_token_affine(Ob, *_lin(attn.out), L)

    def _ff(self, h, ff, L: int):
        z = _per_token_affine(h, *_lin(ff.l0), L)
        z = self._relu(z)
        return _per_token_affine(z, *_lin(ff.l3), L)

    def _block(self, x, blk, L: int, dim: int):
        sc, sh = blk.norm_attn.affine()
        h = _per_token_affine(x, np.diag(sc), sh, L)
        x = self._add(x, self._attn(h, blk.attn, L, dim))
        sc2, sh2 = blk.norm_ff.affine()
        h2 = _per_token_affine(x, np.diag(sc2), sh2, L)
        x = self._add(x, self._ff(h2, blk.ff, L))
        return x

    def reach(self, lb_chw=None, ub_chw=None, input_imagestar=None):
        """Reach the 10-logit output set. Pass (lb_chw, ub_chw) normalized
        (C,H,W) bounds; ``input_imagestar`` is accepted for back-compat (Star)."""
        m = self.m
        if isinstance(lb_chw, ImageStar):  # back-compat: reach(imagestar)
            input_imagestar, lb_chw = lb_chw, None
        if input_imagestar is not None:    # legacy Star path
            cs = conv2d_star(m.projection, [input_imagestar])[0]
            Hp, Wp, dim = cs.height, cs.width, cs.num_channels
            star = cs.to_star()
        else:
            star, Hp, Wp, dim = self._patch_embed(
                np.asarray(lb_chw, float), np.asarray(ub_chw, float))
        star = _prepend_const_token(star, m.cls_token.detach().cpu().numpy())
        L = Hp * Wp + 1
        pos = m.positions.detach().cpu().numpy().reshape(-1, 1)
        star = star.affine_map(np.eye(L * dim), pos)
        for blk in m.blocks:
            star = self._block(star, blk, L, dim)
        P = np.zeros((dim, L * dim))
        for t in range(L):
            P[np.arange(dim), t * dim + np.arange(dim)] = 1.0 / L
        star = star.affine_map(P)
        sc, sh = m.head_norm.affine()
        star = star.affine_map(np.diag(sc), sh.reshape(-1, 1))
        Wh, bh = _lin(m.head)
        return star.affine_map(Wh, bh.reshape(-1, 1))

    def margins(self, out, label: int, margin_mode: str = "estimate",
                lp_solver: str = "default") -> dict:
        K = out.dim
        res = {}
        for i in range(K):
            if i == label:
                continue
            c = np.zeros((1, K)); c[0, label] = 1.0; c[0, i] = -1.0
            ms = out.affine_map(c)
            if isinstance(ms, Box):
                mlb = float(np.asarray(ms.lb).reshape(-1)[0])
            elif isinstance(ms, Star) and margin_mode == "lp":
                mlb = ms.get_min(0, lp_solver=lp_solver)
            elif isinstance(ms, Star):
                lb, _ = ms.estimate_ranges(); mlb = float(lb.reshape(-1)[0])
            else:  # Zono
                lb, _ = ms.get_bounds(); mlb = float(np.asarray(lb).reshape(-1)[0])
            res[i] = float(mlb) if mlb is not None else None
        return res

    def verify(self, lb_chw, ub_chw, label, margin_mode="estimate",
               lp_solver="default") -> dict:
        out = self.reach(np.asarray(lb_chw, float), np.asarray(ub_chw, float))
        m = self.margins(out, label, margin_mode=margin_mode, lp_solver=lp_solver)
        verified = all(v is not None and v > 0 for v in m.values())
        return {"status": "verified" if verified else "unknown",
                "margins": m, "out_star": out}


def _per_token_affine(s, W: np.ndarray, b: np.ndarray, L: int):
    """Apply the same affine map ``W x + b`` to each of L tokens (exact)."""
    full = np.kron(np.eye(L), W)
    bias = np.tile(b, L).reshape(-1, 1)
    return s.affine_map(full, bias)
