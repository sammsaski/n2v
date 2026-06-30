"""LP-free CROWN verification of VNN-COMP ViT models.

The package home of the ViT CROWN verifier: it reconstructs the VNN-COMP 2023 ViT
(``ViT_BN``) from its ONNX initializers (the exported opset-9 graph -- Slice v1 +
per-token BatchNorm -- is not ingested by onnx2torch), lowers it to the flat op DAG
consumed by :mod:`n2v.nn.crown_reach`, and verifies a robustness property with
forward IBP -> CROWN intermediate-bound refinement -> backward CROWN (+ optional
alpha-optimization). **No LP solver anywhere** -- this is the route that scales
where the star+LP attention reach did not.

Provenance: this is an independent, from-scratch Python translation of the *method*
in NNV's ``ViTCrown.m`` (NNV Team, ViT track) -- a MATLAB reference read for the
algorithm only. It uses **no NNV code or files**, installs/imports **no NNV**, and
(like ``ViTCrown.m`` itself) is **not** a port of auto_LiRPA / alpha,beta-CROWN
(also math references only). The relaxations (ReLU triangle, exp/reciprocal convex
envelopes, McCormick bilinear, backward linear bounds, alpha-relaxation) are
standard published techniques, not copied code. n2v gains no new dependency: these
modules import only numpy / torch / n2v.

The benchmark driver (``benchmarks/vit_vnncomp2023``) and the VNN-COMP runner both
import from here.
"""
from __future__ import annotations

import re

import numpy as np

from n2v.nn import crown_reach as cr

# CIFAR-10 normalization baked OUTSIDE the ONNX (the ONNX input is the normalized
# image). Used only by the convenience raw-image entry points (eps_box / norm_img);
# the VNN-COMP path takes the vnnlib box directly (already in model space).
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD = np.array([0.2023, 0.1994, 0.2010])


# =========================================================================== #
#  ViT_BN architecture (mirrors the exported ONNX) + ONNX initializer loader   #
# =========================================================================== #
def _build_torch_vit(patch_size, depth, dim=48, heads=3, dim_head=16, mlp_dim=96,
                     num_classes=10):
    import torch
    import torch.nn as nn

    class BNAcrossSeq(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d)); self.bias = nn.Parameter(torch.zeros(d))
            self.register_buffer("running_mean", torch.zeros(d))
            self.register_buffer("running_var", torch.ones(d))
            self.eps = 1e-5

        def affine(self):
            scale = self.weight.detach().cpu().numpy() / (
                (self.running_var.detach().cpu().numpy() + self.eps) ** 0.5)
            shift = (self.bias.detach().cpu().numpy()
                     - scale * self.running_mean.detach().cpu().numpy())
            return scale, shift

        def forward(self, x):
            m = self.running_mean.view(1, 1, -1); v = self.running_var.view(1, 1, -1)
            w = self.weight.view(1, 1, -1); b = self.bias.view(1, 1, -1)
            return (x - m) / torch.sqrt(v + self.eps) * w + b

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads; self.dim_head = dim_head
            self.scale = dim_head ** -0.5
            inner = heads * dim_head
            self.query = nn.Linear(dim, inner); self.key = nn.Linear(dim, inner)
            self.value = nn.Linear(dim, inner); self.out = nn.Linear(inner, dim)

        def forward(self, x):
            B, N, _ = x.shape; H, D = self.heads, self.dim_head
            q = self.query(x).reshape(B, N, H, D).transpose(1, 2)
            k = self.key(x).reshape(B, N, H, D).transpose(1, 2)
            v = self.value(x).reshape(B, N, H, D).transpose(1, 2)
            att = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            out = (att @ v).transpose(1, 2).reshape(B, N, H * D)
            return self.out(out)

    class FF(nn.Module):
        def __init__(self):
            super().__init__()
            self.l0 = nn.Linear(dim, mlp_dim); self.relu = nn.ReLU(); self.l3 = nn.Linear(mlp_dim, dim)

        def forward(self, x):
            return self.l3(self.relu(self.l0(x)))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_attn = BNAcrossSeq(dim); self.attn = Attn()
            self.norm_ff = BNAcrossSeq(dim); self.ff = FF()

        def forward(self, x):
            x = x + self.attn(self.norm_attn(x))
            return x + self.ff(self.norm_ff(x))

    class ViT_BN(nn.Module):
        def __init__(self):
            super().__init__()
            n_patches = (32 // patch_size) ** 2
            self.patch_size = patch_size; self.depth = depth; self.dim = dim
            self.heads = heads; self.dim_head = dim_head
            self.n_tokens = n_patches + 1
            self.cls_token = nn.Parameter(torch.zeros(dim))
            self.positions = nn.Parameter(torch.zeros(n_patches + 1, dim))
            self.projection = nn.Conv2d(3, dim, patch_size, stride=patch_size)
            self.blocks = nn.ModuleList([Block() for _ in range(depth)])
            self.head_norm = BNAcrossSeq(dim); self.head = nn.Linear(dim, num_classes)

        def forward(self, x):
            B = x.shape[0]
            z = self.projection(x).flatten(2).transpose(1, 2)
            cls = self.cls_token.view(1, 1, -1).expand(B, 1, -1)
            z = torch.cat([cls, z], dim=1) + self.positions.unsqueeze(0)
            for blk in self.blocks:
                z = blk(z)
            z = z.mean(dim=1)
            z = self.head_norm(z.unsqueeze(1)).squeeze(1)
            return self.head(z)

    return ViT_BN()


def _infer_arch(inits):
    """Infer (patch_size, depth) from the loaded initializers (``0.positions``
    shape -> token count -> patch size; ``1.<i>.`` keys -> depth)."""
    n_patches = int(inits["0.positions"].shape[0]) - 1
    patch_size = int(round((32 * 32 / n_patches) ** 0.5))
    depth = len({int(m.group(1)) for k in inits
                 if (m := re.match(r"1\.(\d+)\.", k))})
    return patch_size, depth


def load_vit_onnx(onnx_path):
    """Reconstruct a loaded ``ViT_BN`` torch model from a VNN-COMP ViT ONNX file
    (architecture inferred from the initializers; weights loaded strictly)."""
    import onnx
    import torch
    proto = onnx.load(str(onnx_path))
    inits = {i.name: torch.from_numpy(onnx.numpy_helper.to_array(i).copy()).float()
             for i in proto.graph.initializer}
    patch_size, depth = _infer_arch(inits)
    model = _build_torch_vit(patch_size, depth)
    sd = model.state_dict(); dim = model.dim
    sd["cls_token"] = inits["0.cls_token"].view(dim)
    sd["positions"] = inits["0.positions"]
    sd["projection.weight"] = inits["0.projection.weight"]
    sd["projection.bias"] = inits["0.projection.bias"]
    sd["head_norm.weight"] = inits["2.1.weight"]; sd["head_norm.bias"] = inits["2.1.bias"]
    sd["head_norm.running_mean"] = inits["2.1.running_mean"]
    sd["head_norm.running_var"] = inits["2.1.running_var"]
    sd["head.weight"] = inits["2.2.weight"]; sd["head.bias"] = inits["2.2.bias"]
    for i in range(depth):
        sd[f"blocks.{i}.norm_attn.weight"] = inits[f"1.{i}.0.fn.0.norm.weight"]
        sd[f"blocks.{i}.norm_attn.bias"] = inits[f"1.{i}.0.fn.0.norm.bias"]
        sd[f"blocks.{i}.norm_attn.running_mean"] = inits[f"1.{i}.0.fn.0.norm.running_mean"]
        sd[f"blocks.{i}.norm_attn.running_var"] = inits[f"1.{i}.0.fn.0.norm.running_var"]
        sd[f"blocks.{i}.attn.query.bias"] = inits[f"1.{i}.0.fn.1.query.bias"]
        sd[f"blocks.{i}.attn.key.bias"] = inits[f"1.{i}.0.fn.1.key.bias"]
        sd[f"blocks.{i}.attn.value.bias"] = inits[f"1.{i}.0.fn.1.value.bias"]
        sd[f"blocks.{i}.attn.out.bias"] = inits[f"1.{i}.0.fn.1.out.bias"]
        sd[f"blocks.{i}.norm_ff.weight"] = inits[f"1.{i}.1.fn.0.norm.weight"]
        sd[f"blocks.{i}.norm_ff.bias"] = inits[f"1.{i}.1.fn.0.norm.bias"]
        sd[f"blocks.{i}.norm_ff.running_mean"] = inits[f"1.{i}.1.fn.0.norm.running_mean"]
        sd[f"blocks.{i}.norm_ff.running_var"] = inits[f"1.{i}.1.fn.0.norm.running_var"]
        sd[f"blocks.{i}.ff.l0.bias"] = inits[f"1.{i}.1.fn.1.0.bias"]
        sd[f"blocks.{i}.ff.l3.bias"] = inits[f"1.{i}.1.fn.1.3.bias"]
    matmul_nodes = [n for n in proto.graph.node
                    if n.op_type == "MatMul" and any(inp in inits for inp in n.input)]
    expected = ["query", "key", "value", "out", "ff.l0", "ff.l3"]
    if len(matmul_nodes) != 6 * depth:
        raise ValueError(f"expected {6*depth} weight-matmul nodes, got {len(matmul_nodes)}")
    for i in range(depth):
        for j, key in enumerate(expected):
            node = matmul_nodes[i * 6 + j]
            wname = next(n for n in node.input if n in inits)
            w_torch = inits[wname].transpose(0, 1).contiguous()
            pkey = (f"blocks.{i}.{key}.weight" if "ff" in key
                    else f"blocks.{i}.attn.{key}.weight")
            sd[pkey] = w_torch
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.double()        # lower in float64 (exact BatchNorm affine)


# =========================================================================== #
#  op builders + lowering (mirrors NNV ViTCrown.toOps, e-major / column-major) #
# =========================================================================== #
def _affine_op(in_idx, W, b, mat):
    W = np.asarray(W, dtype=np.float64); b = np.asarray(b, dtype=np.float64).reshape(-1)
    return dict(type="affine", **{"in": in_idx}, W=W, b=b, dim=W.shape[0], mat=mat)


def _per_token_op(in_idx, W, b, N):
    W = np.asarray(W, dtype=np.float64)
    full = np.kron(W, np.eye(N))
    bias = np.repeat(np.asarray(b, dtype=np.float64).reshape(-1), N)
    return _affine_op(in_idx, full, bias, (N, W.shape[0]))


def _bn_op(in_idx, scale, shift, N, E):
    d = np.repeat(np.asarray(scale, float).reshape(-1), N)
    bsh = np.repeat(np.asarray(shift, float).reshape(-1), N)
    return _affine_op(in_idx, np.diag(d), bsh, (N, E))


def _slice_op(in_idx, h, D, N, E):
    idx = h * D * N + np.arange(D * N)
    W = np.zeros((D * N, N * E)); W[np.arange(D * N), idx] = 1.0
    return _affine_op(in_idx, W, np.zeros(D * N), (N, D))


def _lin(layer):
    return (layer.weight.detach().cpu().numpy().astype(np.float64),
            layer.bias.detach().cpu().numpy().astype(np.float64))


def _patch_embed(model):
    p = model.patch_size; Hg = 32 // p; E = model.dim; N = model.n_tokens
    proj_w = model.projection.weight.detach().cpu().numpy().astype(np.float64)
    proj_b = model.projection.bias.detach().cpu().numpy().astype(np.float64)
    cls = model.cls_token.detach().cpu().numpy().astype(np.float64)
    pos = model.positions.detach().cpu().numpy().astype(np.float64)
    Wpe = np.zeros((N * E, 3 * 32 * 32)); bpe = np.zeros(N * E)
    for e in range(E):
        bpe[e * N + 0] = cls[e] + pos[0, e]
    for gh in range(Hg):
        for gw in range(Hg):
            n = gh * Hg + gw + 1
            for e in range(E):
                bpe[e * N + n] = proj_b[e] + pos[n, e]
                for c in range(3):
                    for ii in range(p):
                        for jj in range(p):
                            col = c * 1024 + (gh * p + ii) * 32 + (gw * p + jj)
                            Wpe[e * N + n, col] = proj_w[e, c, ii, jj]
    return Wpe, bpe


def to_ops(model):
    """Lower a ``ViT_BN`` (this module's or the benchmark's -- duck-typed on the
    same attribute names) to a CROWN op DAG."""
    E = model.dim; H = model.heads; D = model.dim_head; N = model.n_tokens
    scale = float(model.blocks[0].attn.scale)
    ops = []

    def push(op):
        ops.append(op); return len(ops) - 1

    push(dict(type="input", **{"in": []}, dim=3 * 32 * 32, mat=None))
    Wpe, bpe = _patch_embed(model)
    x = push(_affine_op(0, Wpe, bpe, (N, E)))

    for blk in model.blocks:
        sc1, sh1 = blk.norm_attn.affine()
        xn = push(_bn_op(x, sc1, sh1, N, E))
        Wq, bq = _lin(blk.attn.query); Wk, bk = _lin(blk.attn.key)
        Wv, bv = _lin(blk.attn.value); Wo, bo = _lin(blk.attn.out)
        qIdx = push(_per_token_op(xn, Wq, bq, N))
        kIdx = push(_per_token_op(xn, Wk, bk, N))
        vIdx = push(_per_token_op(xn, Wv, bv, N))
        Wsum = np.zeros((N, N * N))
        for ii in range(N):
            Wsum[ii, np.arange(N) * N + ii] = 1.0
        Wexp = np.zeros((N * N, N))
        for kk in range(N):
            for ii in range(N):
                Wexp[kk * N + ii, ii] = 1.0
        head_outs = []
        for h in range(H):
            qh = push(_slice_op(qIdx, h, D, N, E))
            kh = push(_slice_op(kIdx, h, D, N, E))
            vh = push(_slice_op(vIdx, h, D, N, E))
            sc = push(dict(type="bmatmul", **{"in": [qh, kh]}, mode="abt",
                           ra=N, ca=D, rb=N, cb=D, dim=N * N, mat=(N, N)))
            scs = push(_affine_op(sc, scale * np.eye(N * N), np.zeros(N * N), (N, N)))
            expi = push(dict(type="exp", **{"in": scs}, dim=N * N, mat=(N, N)))
            sumi = push(_affine_op(expi, Wsum, np.zeros(N), (N, 1)))
            reci = push(dict(type="reciprocal", **{"in": sumi}, dim=N, mat=(N, 1)))
            rexp = push(_affine_op(reci, Wexp, np.zeros(N * N), (N, N)))
            ai = push(dict(type="eprod", **{"in": [expi, rexp]}, dim=N * N, mat=(N, N)))
            oh = push(dict(type="bmatmul", **{"in": [ai, vh]}, mode="ab",
                           ra=N, ca=N, rb=N, cb=D, dim=N * D, mat=(N, D)))
            head_outs.append(oh)
        oc = push(dict(type="concat", **{"in": head_outs}, dim=N * E, mat=(N, E)))
        ao = push(_per_token_op(oc, Wo, bo, N))
        x = push(dict(type="add", **{"in": [x, ao]}, dim=N * E, mat=(N, E)))

        sc2, sh2 = blk.norm_ff.affine()
        xn2 = push(_bn_op(x, sc2, sh2, N, E))
        W1, b1 = _lin(blk.ff.l0); W2, b2 = _lin(blk.ff.l3)
        f1 = push(_per_token_op(xn2, W1, b1, N))
        r = push(dict(type="relu", **{"in": f1}, dim=ops[f1]["dim"], mat=ops[f1]["mat"]))
        f2 = push(_per_token_op(r, W2, b2, N))
        x = push(dict(type="add", **{"in": [x, f2]}, dim=N * E, mat=(N, E)))

    Wm = np.zeros((E, N * E))
    for e in range(E):
        Wm[e, e * N + np.arange(N)] = 1.0 / N
    z = push(_affine_op(x, Wm, np.zeros(E), (E, 1)))
    hsc, hsh = model.head_norm.affine()
    zb = push(_affine_op(z, np.diag(hsc), hsh, (E, 1)))
    Wh, bh = _lin(model.head)
    push(_affine_op(zb, Wh, bh, (10, 1)))
    return ops


# =========================================================================== #
#  instance helpers + verdict                                                  #
# =========================================================================== #
def norm_img(img01):
    img01 = np.asarray(img01, dtype=np.float64)
    return ((img01 - MEAN[:, None, None]) / STD[:, None, None]).reshape(-1)


def eps_box(img01, eps=1.0 / 255):
    """Raw [3,32,32] in [0,1] -> normalized per-pixel (lb,ub) in (c,h,w) C-order."""
    img01 = np.asarray(img01, dtype=np.float64)
    lo = np.clip(img01 - eps, 0, 1); hi = np.clip(img01 + eps, 0, 1)
    lb = ((lo - MEAN[:, None, None]) / STD[:, None, None]).reshape(-1)
    ub = ((hi - MEAN[:, None, None]) / STD[:, None, None]).reshape(-1)
    return lb, ub


def margin_spec(label):
    """C (9x10): row r is logit_label - logit_other_r (argmax preservation)."""
    others = [i for i in range(10) if i != label]
    C = np.zeros((9, 10))
    for r, i in enumerate(others):
        C[r, label] = 1.0; C[r, i] = -1.0
    return C


def crown_margins(ops, lb, ub, C, offset=None, refine=True, refine_iters=2,
                  refine_max_dim=np.inf, alpha=False, alpha_iter=60, alpha_lr=0.1,
                  verbose=False):
    """LP-free CROWN lower bound of ``C @ logits - offset`` over the box [lb,ub].
    ``offset`` (per-row, default 0) lets the spec encode general half-spaces
    ``G@Y > g`` (pass C=G, offset=g). Returns the per-row margin lower bounds."""
    if offset is None:
        offset = np.zeros(C.shape[0])
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    if refine:
        cl, cu, mmaps = cr.refine_bounds(ops, lb, ub, iters=refine_iters,
                                         max_dim=refine_max_dim)
    else:
        cl, cu = cr.forward_ibp(ops, lb, ub)
        mmaps = cr.precompute_maps(ops, cl, cu)
    if alpha:
        margins, _ = cr.optimize_alpha(ops, lb, ub, cl, cu, C, n_iter=alpha_iter,
                                       lr=alpha_lr, verbose=verbose, offset=offset)
    else:
        m, _, _ = cr.backward_crown(ops, lb, ub, cl, cu, C, mmaps=mmaps)
        margins = np.asarray(m).reshape(-1)
    return margins - offset


def verify_instance(ops, img01, label, eps=1.0 / 255, **kw):
    """Robustness verdict for a RAW image instance (convenience entry point).
    Returns (robust, margins, info)."""
    lb, ub = eps_box(img01, eps)
    C = margin_spec(label)
    margins = crown_margins(ops, lb, ub, C, **kw)
    return bool(np.all(margins > 0)), margins, {"lb": lb, "ub": ub, "C": C}


def verify_halfspace_group_safe(ops, lb, ub, G, g, **kw):
    """A half-space group is ORed (a counterexample satisfies ANY row). It is
    provably unreachable -- the box is SAFE w.r.t. this group -- iff EVERY row's
    unsafe condition ``G_k @ Y <= g_k`` is impossible, i.e. CROWN proves
    ``G_k @ Y > g_k``. Returns (group_safe, per_row_margin)."""
    G = np.atleast_2d(np.asarray(G, dtype=np.float64))
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    margins = crown_margins(ops, lb, ub, G, offset=g, **kw)
    return bool(np.all(margins > 0)), margins
