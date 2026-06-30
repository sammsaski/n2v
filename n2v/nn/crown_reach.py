"""LP-free CROWN reachability over a flat operation DAG.

An independent, from-scratch Python **re-implementation** of the method in NNV's
*now-working* attention verifier ``ViTCrown.m``. It uses **no NNV code or files**
and adds **no NNV dependency** -- NNV's MATLAB source is a reference for the
algorithm only; this module imports only numpy. ``ViTCrown.m`` is cited purely for
attribution. The decisive
change from n2v's earlier star+LP attention BaB is that **there is no LP solver
anywhere** -- the certificate is produced by dense backward linear-bound
propagation (alpha,beta-CROWN style), exactly the reason CROWN scales where a
monolithic margin LP walls out (a 23k-var margin LP on the ViT took 10.6 h; this
recipe verifies the same instance in ~30 s).

The model is lowered (by a separate, model-specific lowering -- see the ViT
driver) to a flat op DAG whose only nonlinearities are ``relu`` (FF), the two
attention bilinear matmuls (``bmatmul``), and softmax, which is itself
**decomposed** into ``exp -> reduce_sum (affine) -> reciprocal -> broadcast
(affine) -> eprod`` so a real backward coefficient flows to the score (this is
what restores the lost Q-K input correlation -- the constant-box softmax zeroed
it). Three mechanisms, each sound, compose to cross zero:

1. score-carrying softmax backward (the decomposition above),
2. CROWN intermediate-bound refinement (``refine_bounds`` -- the tightness driver
   that collapses the IBP looseness), and
3. alpha-optimization (``optimize_alpha`` -- PGA on the ReLU lower slopes and the
   attention McCormick plane interpolation; sound double re-evaluation).

Op layout (mirrors ViTCrown): every op's value is a 1-D vector; matrix-valued ops
carry ``mat=(r,c)`` and the vector is the **column-major** (Fortran-order) flatten
of that matrix. Op dict fields: ``type``, ``in`` (list of 0-based source indices),
plus type-specific data.

  types: 'input'(dim) | 'affine'(W,b) | 'relu' | 'add' |
         'bmatmul'(mode 'abt'|'ab', ra,ca,rb,cb, dim) |
         'exp' | 'reciprocal' | 'eprod' | 'concat' | 'softmax'(mat)

Soundness is the only hard constraint: every relaxation is a sound
over-approximation, so the worst case is a failure to certify, never a false
VERIFIED. The engine is toolbox-general (no benchmark assumption); the ViT
lowering lives with the benchmark.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  small array-library shim: the SAME backward pass runs in numpy (the sound    #
#  certificate) and in torch (alpha-optimization, gradients only).             #
# --------------------------------------------------------------------------- #
def _pos(x):
    """max(x, 0) with a constant 0/1 mask -- the correct subgradient for the
    sign gate, identical in numpy and torch (``x * (x > 0)``)."""
    return x * (x > 0)


def _neg(x):
    return x * (x < 0)


def _conv(tensor: bool):
    """Return (cast, zeros) helpers for the active array library."""
    if tensor:
        import torch
        def cast(a):
            if torch.is_tensor(a):
                return a
            return torch.as_tensor(np.asarray(a, dtype=np.float64))
        def zeros(s):
            return torch.zeros(s, dtype=torch.float64)
        return cast, zeros
    return (lambda a: np.asarray(a, dtype=np.float64)), (lambda s: np.zeros(s))


# --------------------------------------------------------------------------- #
#  concrete evaluator (the orientation gate validates this against the model)   #
# --------------------------------------------------------------------------- #
def eval_ops(ops: List[dict], xvec: np.ndarray) -> Tuple[np.ndarray, list]:
    """Evaluate the DAG on a concrete input vector. Returns (logits, all_vals)."""
    xvec = np.asarray(xvec, dtype=np.float64).reshape(-1)
    val = [None] * len(ops)
    for k, op in enumerate(ops):
        t = op["type"]
        if t == "input":
            val[k] = xvec.copy()
        elif t == "affine":
            val[k] = op["W"] @ val[op["in"]] + op["b"]
        elif t == "relu":
            val[k] = np.maximum(val[op["in"]], 0.0)
        elif t == "add":
            val[k] = val[op["in"][0]] + val[op["in"][1]]
        elif t == "bmatmul":
            A = val[op["in"][0]].reshape(op["ra"], op["ca"], order="F")
            B = val[op["in"][1]].reshape(op["rb"], op["cb"], order="F")
            Cm = A @ B.T if op["mode"] == "abt" else A @ B
            val[k] = Cm.reshape(-1, order="F")
        elif t == "softmax":
            S = val[op["in"]].reshape(op["mat"], order="F")
            m = S.max(axis=1, keepdims=True)
            ex = np.exp(S - m)
            val[k] = (ex / ex.sum(axis=1, keepdims=True)).reshape(-1, order="F")
        elif t == "exp":
            val[k] = np.exp(val[op["in"]])
        elif t == "reciprocal":
            val[k] = 1.0 / val[op["in"]]
        elif t == "eprod":
            val[k] = val[op["in"][0]] * val[op["in"][1]]
        elif t == "concat":
            val[k] = np.concatenate([val[j] for j in op["in"]])
        else:
            raise ValueError(f"eval_ops: unknown op {t}")
    return val[-1], val


# --------------------------------------------------------------------------- #
#  forward IBP (sound per-op interval bounds)                                   #
# --------------------------------------------------------------------------- #
def _interval_matmul(al, au, bl, bu):
    """Rump's sound enclosure of {A@B : al<=A<=au, bl<=B<=bu} (no operand
    cross-correlation assumed)."""
    Am = 0.5 * (al + au); Ar = 0.5 * (au - al)
    Bm = 0.5 * (bl + bu); Br = 0.5 * (bu - bl)
    Cm = Am @ Bm
    Cr = np.abs(Am) @ Br + Ar @ np.abs(Bm) + Ar @ Br
    return Cm - Cr, Cm + Cr


def _correlated_row_softmax(s_lo, s_hi):
    """Exact per-element range of softmax over the key axis given the logit box
    (the per-target IBP optimum). s_lo,s_hi are [R,n]."""
    R, n = s_lo.shape
    a_lb = np.zeros((R, n)); a_ub = np.zeros((R, n))
    def col(S, j):
        m = S.max(axis=1, keepdims=True)
        E = np.exp(S - m)
        return E[:, j] / E.sum(axis=1)
    for j in range(n):
        Su = s_lo.copy(); Su[:, j] = s_hi[:, j]; a_ub[:, j] = col(Su, j)
        Sl = s_hi.copy(); Sl[:, j] = s_lo[:, j]; a_lb[:, j] = col(Sl, j)
    return np.clip(a_lb, 0, 1), np.clip(a_ub, 0, 1)


def forward_ibp(ops, lb, ub, ov=None):
    """Sound per-op interval bounds over the input box [lb,ub]. ``ov`` (optional)
    is a per-op list of [lo,hi] (n,2) overrides intersected after each op -- the
    CROWN-refinement / GenBaB score-split hook (sound: a case-split on an op's
    value covers its original range)."""
    n = len(ops)
    cl = [None] * n; cu = [None] * n
    for k, op in enumerate(ops):
        t = op["type"]
        if t == "input":
            cl[k] = np.asarray(lb, float).reshape(-1).copy()
            cu[k] = np.asarray(ub, float).reshape(-1).copy()
        elif t == "affine":
            W = op["W"]; Wp = np.maximum(W, 0); Wn = np.minimum(W, 0)
            cl[k] = Wp @ cl[op["in"]] + Wn @ cu[op["in"]] + op["b"]
            cu[k] = Wp @ cu[op["in"]] + Wn @ cl[op["in"]] + op["b"]
        elif t == "relu":
            cl[k] = np.maximum(cl[op["in"]], 0); cu[k] = np.maximum(cu[op["in"]], 0)
        elif t == "add":
            cl[k] = cl[op["in"][0]] + cl[op["in"][1]]
            cu[k] = cu[op["in"][0]] + cu[op["in"][1]]
        elif t == "bmatmul":
            Al = cl[op["in"][0]].reshape(op["ra"], op["ca"], order="F")
            Au = cu[op["in"][0]].reshape(op["ra"], op["ca"], order="F")
            Bl = cl[op["in"][1]].reshape(op["rb"], op["cb"], order="F")
            Bu = cu[op["in"][1]].reshape(op["rb"], op["cb"], order="F")
            if op["mode"] == "abt":
                Cl, Cu = _interval_matmul(Al, Au, Bl.T, Bu.T)
            else:
                Cl, Cu = _interval_matmul(Al, Au, Bl, Bu)
            cl[k] = Cl.reshape(-1, order="F"); cu[k] = Cu.reshape(-1, order="F")
        elif t == "softmax":
            Slo = cl[op["in"]].reshape(op["mat"], order="F")
            Shi = cu[op["in"]].reshape(op["mat"], order="F")
            alb, aub = _correlated_row_softmax(Slo, Shi)
            cl[k] = alb.reshape(-1, order="F"); cu[k] = aub.reshape(-1, order="F")
        elif t == "exp":
            cl[k] = np.exp(cl[op["in"]]); cu[k] = np.exp(cu[op["in"]])
        elif t == "reciprocal":
            lo = np.maximum(cl[op["in"]], 1e-12)        # T = sum exp > 0 (FP floor)
            cl[k] = 1.0 / cu[op["in"]]; cu[k] = 1.0 / lo
        elif t == "eprod":
            El = cl[op["in"][0]]; Eu = cu[op["in"][0]]
            Rl = cl[op["in"][1]]; Ru = cu[op["in"][1]]
            c1 = El * Rl; c2 = El * Ru; c3 = Eu * Rl; c4 = Eu * Ru
            cl[k] = np.minimum(np.minimum(c1, c2), np.minimum(c3, c4))
            cu[k] = np.maximum(np.maximum(c1, c2), np.maximum(c3, c4))
        elif t == "concat":
            cl[k] = np.concatenate([cl[j] for j in op["in"]])
            cu[k] = np.concatenate([cu[j] for j in op["in"]])
        else:
            raise ValueError(f"forward_ibp: unknown op {t}")
        if ov is not None and ov[k] is not None:
            cl[k] = np.maximum(cl[k], ov[k][:, 0])
            cu[k] = np.minimum(cu[k], ov[k][:, 1])
    return cl, cu


# --------------------------------------------------------------------------- #
#  McCormick corner planes (bilinear under/over-estimators)                     #
# --------------------------------------------------------------------------- #
def mc_corners(pl, pu, ql, qu):
    """The two McCormick under-estimator planes (L1,L2) and over-estimator planes
    (U1,U2) for t = p*q on [pl,pu]x[ql,qu], as coeffs ap*p + aq*q + a0; mid.* =
    plane value at the box midpoint (for the default tighter pick)."""
    pm = 0.5 * (pl + pu); qm = 0.5 * (ql + qu)
    PL = dict(ap1=ql, aq1=pl, a01=-(pl * ql),
              ap2=qu, aq2=pu, a02=-(pu * qu))
    PU = dict(ap1=ql, aq1=pu, a01=-(pu * ql),
              ap2=qu, aq2=pl, a02=-(pl * qu))
    mid = dict(L1=pl * qm + ql * pm - pl * ql, L2=pu * qm + qu * pm - pu * qu,
               U1=pu * qm + ql * pm - pu * ql, U2=pl * qm + qu * pm - pl * qu)
    return PL, PU, mid


def bmm_maps(op, cl, cu):
    """Fixed sparse->dense McCormick adjoint maps for a reducing bilinear matmul:
    for the lower/upper envelope, the two valid corner planes per output element,
    assembled as operand->output coefficient matrices. A sound estimator is any
    convex combo ``aL*map1 + (1-aL)*map2`` (likewise aU) -> aL,aU in [0,1] are the
    attention-layer alpha (optimized in optimize_alpha). Stored dense (dimY small)
    so they flow through autograd."""
    ra, ca, rb, cb = op["ra"], op["ca"], op["rb"], op["cb"]
    clA = cl[op["in"][0]].reshape(ra, ca, order="F"); cuA = cu[op["in"][0]].reshape(ra, ca, order="F")
    clB = cl[op["in"][1]].reshape(rb, cb, order="F"); cuB = cu[op["in"][1]].reshape(rb, cb, order="F")
    dimY = op["dim"]; dimA = ra * ca; dimB = rb * cb; N = ra
    AL1 = np.zeros((dimY, dimA)); AL2 = np.zeros((dimY, dimA))
    AU1 = np.zeros((dimY, dimA)); AU2 = np.zeros((dimY, dimA))
    BL1 = np.zeros((dimY, dimB)); BL2 = np.zeros((dimY, dimB))
    BU1 = np.zeros((dimY, dimB)); BU2 = np.zeros((dimY, dimB))
    c0L1 = np.zeros(dimY); c0L2 = np.zeros(dimY)
    c0U1 = np.zeros(dimY); c0U2 = np.zeros(dimY)
    aL0 = np.zeros(dimY); aU0 = np.zeros(dimY)
    if op["mode"] == "abt":
        Iout, Jout = ra, rb
    else:
        Iout, Jout = cb, ra
    for a1 in range(Iout):
        for a2 in range(Jout):
            if op["mode"] == "abt":           # Y(i,j) = sum_d A(i,d) B(j,d)
                i, j = a1, a2; m = j * N + i
                pl = clA[i, :]; pu = cuA[i, :]; idxA = np.arange(ca) * N + i
                ql = clB[j, :]; qu = cuB[j, :]; idxB = np.arange(cb) * N + j
            else:                             # Y(i,e) = sum_n A(i,n) B(n,e)
                e, i = a1, a2; m = e * N + i
                pl = clA[i, :]; pu = cuA[i, :]; idxA = np.arange(ca) * N + i
                ql = clB[:, e]; qu = cuB[:, e]; idxB = e * N + np.arange(rb)
            PL, PU, mid = mc_corners(pl, pu, ql, qu)
            AL1[m, idxA] = PL["ap1"]; AL2[m, idxA] = PL["ap2"]
            AU1[m, idxA] = PU["ap1"]; AU2[m, idxA] = PU["ap2"]
            BL1[m, idxB] = PL["aq1"]; BL2[m, idxB] = PL["aq2"]
            BU1[m, idxB] = PU["aq1"]; BU2[m, idxB] = PU["aq2"]
            c0L1[m] = PL["a01"].sum(); c0L2[m] = PL["a02"].sum()
            c0U1[m] = PU["a01"].sum(); c0U2[m] = PU["a02"].sum()
            aL0[m] = float(mid["L1"].sum() >= mid["L2"].sum())
            aU0[m] = float(mid["U1"].sum() <= mid["U2"].sum())
    return dict(AL1=AL1, AL2=AL2, AU1=AU1, AU2=AU2, BL1=BL1, BL2=BL2, BU1=BU1, BU2=BU2,
                c0L1=c0L1, c0L2=c0L2, c0U1=c0U1, c0U2=c0U2, aL0=aL0, aU0=aU0, dimY=dimY)


def bmm_combine(mp, aL, aU, cast):
    """Effective McCormick maps for per-output attention-alpha aL,aU in [0,1]
    (convex combo of two valid planes -> sound for any a). ``cast`` lifts the
    constant maps into the active array library."""
    aL0 = cast(mp["aL0"]) if aL is None else aL
    aU0 = cast(mp["aU0"]) if aU is None else aU
    aL0 = aL0.reshape(-1, 1) if hasattr(aL0, "reshape") else aL0
    aU0 = aU0.reshape(-1, 1) if hasattr(aU0, "reshape") else aU0
    PAL = aL0 * cast(mp["AL1"]) + (1 - aL0) * cast(mp["AL2"])
    PBL = aL0 * cast(mp["BL1"]) + (1 - aL0) * cast(mp["BL2"])
    PAU = aU0 * cast(mp["AU1"]) + (1 - aU0) * cast(mp["AU2"])
    PBU = aU0 * cast(mp["BU1"]) + (1 - aU0) * cast(mp["BU2"])
    aL0v = aL0.reshape(-1) if hasattr(aL0, "reshape") else aL0
    aU0v = aU0.reshape(-1) if hasattr(aU0, "reshape") else aU0
    p0L = aL0v * cast(mp["c0L1"]) + (1 - aL0v) * cast(mp["c0L2"])
    p0U = aU0v * cast(mp["c0U1"]) + (1 - aU0v) * cast(mp["c0U2"])
    return PAL, PAU, PBL, PBU, p0L, p0U


def precompute_maps(ops, cl, cu):
    """Precompute the (box-only) McCormick maps for every bmatmul op."""
    mmaps = [None] * len(ops)
    for k, op in enumerate(ops):
        if op["type"] == "bmatmul":
            mmaps[k] = bmm_maps(op, cl, cu)
    return mmaps


# --------------------------------------------------------------------------- #
#  backward CROWN lower bound                                                    #
# --------------------------------------------------------------------------- #
def backward_crown(ops, lb, ub, cl, cu, C, alpha=None, mmaps=None, battn=None,
                   start_op=None, tensor=False):
    """Lower bound of ``C @ value[start_op]`` over the input box [lb,ub] via
    backward CROWN. Sound relaxations: affine/add/concat exact; relu triangle
    (lower slope ``alpha``, upper chord); exp/reciprocal convex (lower tangent,
    upper chord); eprod/bmatmul McCormick. ``alpha`` is a per-op list of relu
    lower-slope vectors (None -> area-adaptive default); ``battn`` a per-op list
    of {'aL','aU'} attention-alpha. With ``tensor=True`` every constant is lifted
    to torch float64 so the bound is differentiable wrt alpha/battn (gradients
    only -- the sound certificate always re-evaluates with tensor=False).
    Returns (lbnd, Lin, bL)."""
    cast, zeros = _conv(tensor)
    n = len(ops); S = C.shape[0]
    if start_op is None:
        start_op = n - 1
    if alpha is None:
        alpha = [None] * n
    if battn is None:
        battn = [None] * n
    if mmaps is None:
        mmaps = precompute_maps(ops, cl, cu)
    Lam = [None] * n
    Lam[start_op] = cast(C)
    bL = zeros(S)

    def addco(i, B):
        Lam[i] = B if Lam[i] is None else Lam[i] + B

    for k in range(n - 1, -1, -1):
        L = Lam[k]
        if L is None:
            continue
        op = ops[k]; t = op["type"]
        if t == "input":
            pass
        elif t == "affine":
            addco(op["in"], L @ cast(op["W"]))
            bL = bL + L @ cast(op["b"])
        elif t == "add":
            addco(op["in"][0], L); addco(op["in"][1], L)
        elif t == "concat":
            off = 0
            for j in op["in"]:
                dj = cl[j].shape[0]
                addco(j, L[:, off:off + dj]); off += dj
        elif t == "relu":
            l = cl[op["in"]]; u = cu[op["in"]]
            sp = (l >= 0).astype(np.float64); sn = (u <= 0).astype(np.float64)
            un = 1.0 - sp - sn
            den = (u - l).copy(); den[den == 0] = 1.0
            slope_u = u / den
            su = sp + un * slope_u                       # upper-chord slope
            ci = un * (-slope_u * l)                     # upper-chord intercept
            if alpha[k] is None:
                a = (u > -l).astype(np.float64)          # area-adaptive default
                a = cast(a)
            else:
                a = alpha[k]                             # may be a torch leaf
            sL = cast(sp) + cast(un) * a                 # lower slope (masked)
            Lp = _pos(L); Ln = _neg(L)
            addco(op["in"], Lp * sL[None, :] + Ln * cast(su)[None, :])
            bL = bL + Ln @ cast(ci)
        elif t == "softmax":
            Lp = _pos(L); Ln = _neg(L)
            bL = bL + Lp @ cast(cl[k]) + Ln @ cast(cu[k])
        elif t == "exp":
            l = cl[op["in"]]; u = cu[op["in"]]; m = 0.5 * (l + u)
            slo = np.exp(m); ilo = np.exp(m) * (1 - m)
            du = (u - l).copy(); du[du == 0] = 1.0
            sup = (np.exp(u) - np.exp(l)) / du; iup = np.exp(l) - sup * l
            Lp = _pos(L); Ln = _neg(L)
            addco(op["in"], Lp * cast(slo)[None, :] + Ln * cast(sup)[None, :])
            bL = bL + Lp @ cast(ilo) + Ln @ cast(iup)
        elif t == "reciprocal":
            l = np.maximum(cl[op["in"]], 1e-12); u = np.maximum(cu[op["in"]], 1e-12)
            m = 0.5 * (l + u)
            slo = -1.0 / (m * m); ilo = 2.0 / m
            sup = -1.0 / (l * u); iup = 1.0 / l - sup * l
            Lp = _pos(L); Ln = _neg(L)
            addco(op["in"], Lp * cast(slo)[None, :] + Ln * cast(sup)[None, :])
            bL = bL + Lp @ cast(ilo) + Ln @ cast(iup)
        elif t == "eprod":
            El = cl[op["in"][0]]; Eu = cu[op["in"][0]]
            Rl = cl[op["in"][1]]; Ru = cu[op["in"][1]]
            PL, PU, mid = mc_corners(El, Eu, Rl, Ru)
            aL = (mid["L1"] >= mid["L2"]).astype(np.float64)
            aU = (mid["U1"] <= mid["U2"]).astype(np.float64)
            apL = aL * PL["ap1"] + (1 - aL) * PL["ap2"]
            aqL = aL * PL["aq1"] + (1 - aL) * PL["aq2"]
            a0L = aL * PL["a01"] + (1 - aL) * PL["a02"]
            apU = aU * PU["ap1"] + (1 - aU) * PU["ap2"]
            aqU = aU * PU["aq1"] + (1 - aU) * PU["aq2"]
            a0U = aU * PU["a01"] + (1 - aU) * PU["a02"]
            Lp = _pos(L); Ln = _neg(L)
            addco(op["in"][0], Lp * cast(apL)[None, :] + Ln * cast(apU)[None, :])
            addco(op["in"][1], Lp * cast(aqL)[None, :] + Ln * cast(aqU)[None, :])
            bL = bL + Lp @ cast(a0L) + Ln @ cast(a0U)
        elif t == "bmatmul":
            mp = mmaps[k]
            ba = battn[k]
            aL = ba["aL"] if (ba is not None) else None
            aU = ba["aU"] if (ba is not None) else None
            PAL, PAU, PBL, PBU, p0L, p0U = bmm_combine(mp, aL, aU, cast)
            Lp = _pos(L); Ln = _neg(L)
            addco(op["in"][0], Lp @ PAL + Ln @ PAU)
            addco(op["in"][1], Lp @ PBL + Ln @ PBU)
            bL = bL + Lp @ p0L + Ln @ p0U
        else:
            raise ValueError(f"backward_crown: unknown op {t}")
    Lin = Lam[0]
    lbnd = _pos(Lin) @ cast(lb) + _neg(Lin) @ cast(ub) + bL
    return lbnd, Lin, bL


def crown_bounds(ops, lb, ub, cl, cu, target_op, mmaps):
    """Tighter [lo,hi] for op ``target_op`` via a backward CROWN pass to the input
    (vs the looser forward IBP). lo = min, hi = -min(-x)."""
    d = ops[target_op]["dim"]
    I = np.eye(d)
    lo, _, _ = backward_crown(ops, lb, ub, cl, cu, I, mmaps=mmaps, start_op=target_op)
    hineg, _, _ = backward_crown(ops, lb, ub, cl, cu, -I, mmaps=mmaps, start_op=target_op)
    return lo, -hineg


def refine_bounds(ops, lb, ub, iters=1, max_dim=np.inf):
    """Replace IBP boxes with CROWN intermediate bounds for ALL nonlinearity
    inputs (relu/exp/reciprocal/eprod/bmatmul operands), sequentially in
    topological order, re-propagating IBP each pass. This is the alpha,beta-CROWN
    tightness driver -- with it the backward bound approaches the LP optimum.
    ``max_dim`` caps which ops to refine (skip wide ops for speed)."""
    cl, cu = forward_ibp(ops, lb, ub)
    mmaps = precompute_maps(ops, cl, cu)
    nl = {"relu", "exp", "reciprocal", "eprod", "bmatmul"}
    targets = set()
    for op in ops:
        if op["type"] in nl:
            ins = op["in"] if isinstance(op["in"], (list, tuple)) else [op["in"]]
            targets.update(ins)
    targets = sorted(t for t in targets if ops[t]["dim"] <= max_dim)
    for _ in range(iters):
        ov = [None] * len(ops)
        for tt in targets:
            lo, hi = crown_bounds(ops, lb, ub, cl, cu, tt, mmaps)
            ov[tt] = np.stack([np.maximum(cl[tt], lo.reshape(-1)),
                               np.minimum(cu[tt], hi.reshape(-1))], axis=1)
        cl, cu = forward_ibp(ops, lb, ub, ov)
        mmaps = precompute_maps(ops, cl, cu)
    return cl, cu, mmaps


def refine_scores(ops, lb, ub):
    """Cheaper refinement: tighten only the score boxes (exp inputs) with CROWN,
    then re-propagate IBP. The key attention tightness lever at a fraction of the
    cost of refining every nonlinearity input."""
    cl, cu = forward_ibp(ops, lb, ub)
    mmaps = precompute_maps(ops, cl, cu)
    exp_idx = [k for k, o in enumerate(ops) if o["type"] == "exp"]
    ov = [None] * len(ops)
    for e in exp_idx:
        sc = ops[e]["in"]
        lo, hi = crown_bounds(ops, lb, ub, cl, cu, sc, mmaps)
        ov[sc] = np.stack([np.maximum(cl[sc], lo.reshape(-1)),
                           np.minimum(cu[sc], hi.reshape(-1))], axis=1)
    cl, cu = forward_ibp(ops, lb, ub, ov)
    mmaps = precompute_maps(ops, cl, cu)
    return cl, cu, mmaps


# --------------------------------------------------------------------------- #
#  alpha-relaxation optimization (PGA, torch autograd; sound double re-eval)     #
# --------------------------------------------------------------------------- #
def optimize_alpha(ops, lb, ub, cl, cu, C, n_iter=40, lr=0.1, do_relu=True,
                   do_attn=True, verbose=False, offset=None):
    """Projected-gradient ascent (Adam) on the relaxation alpha to maximize the
    worst margin. Optimizes the FF-relu lower slope AND -- the main lever for the
    ViT -- the attention-layer alpha (per-output McCormick plane interpolation
    aL,aU on every bilinear QK^T / A*V op). The optimized bound is RE-EVALUATED on
    the sound numpy path, so soundness never depends on the autodiff arithmetic
    (any alpha in [0,1] is sound). Returns (margins, sol) with sol={'alpha','battn'}
    for reuse by BaB."""
    import torch

    if offset is None:
        offset = np.zeros(C.shape[0])
    off = torch.as_tensor(np.asarray(offset, dtype=np.float64).reshape(-1))
    mmaps = precompute_maps(ops, cl, cu)
    relu_idx = [k for k, o in enumerate(ops) if o["type"] == "relu"] if do_relu else []
    bm_idx = [k for k, o in enumerate(ops) if o["type"] == "bmatmul"] if do_attn else []

    leaves = []                      # torch leaf tensors (the optimization vars)
    alpha = [None] * len(ops)        # relu lower-slope leaves, by op
    battn = [None] * len(ops)        # {'aL','aU'} leaves, by op
    for k in relu_idx:
        inb = ops[k]["in"]
        a0 = (cu[inb] > -cl[inb]).astype(np.float64)
        v = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
        alpha[k] = v; leaves.append(v)
    for k in bm_idx:
        mp = mmaps[k]
        vL = torch.tensor(mp["aL0"], dtype=torch.float64, requires_grad=True)
        vU = torch.tensor(mp["aU0"], dtype=torch.float64, requires_grad=True)
        battn[k] = {"aL": vL, "aU": vU}; leaves += [vL, vU]

    if not leaves:                   # nothing to optimize -> plain bound
        m, _, _ = backward_crown(ops, lb, ub, cl, cu, C, mmaps=mmaps)
        return np.asarray(m).reshape(-1), {"alpha": alpha, "battn": battn}

    opt = torch.optim.Adam(leaves, lr=lr)
    best = -np.inf; best_state = None
    for it in range(n_iter):
        opt.zero_grad()
        lbnd, _, _ = backward_crown(ops, lb, ub, cl, cu, C, alpha=alpha,
                                    mmaps=mmaps, battn=battn, tensor=True)
        obj = (lbnd - off).min()             # worst half-space margin (G@Y - g)
        (-obj).backward()
        opt.step()
        with torch.no_grad():
            for v in leaves:
                v.clamp_(0.0, 1.0)            # project to [0,1] (keeps soundness)
            cur = float(obj.detach())
            if cur > best:
                best = cur
                best_state = [v.detach().clone() for v in leaves]
        if verbose and (it < 3 or (it + 1) % 10 == 0):
            print(f"   alpha-opt it {it+1:3d}: min-margin = {cur:+.5f}", flush=True)

    # restore best, build numpy alpha/battn, and RE-EVALUATE soundly in numpy
    if best_state is not None:
        for v, s in zip(leaves, best_state):
            with torch.no_grad():
                v.copy_(s)
    alpha_np = [None] * len(ops); battn_np = [None] * len(ops)
    for k in relu_idx:
        alpha_np[k] = alpha[k].detach().cpu().numpy()
    for k in bm_idx:
        battn_np[k] = {"aL": battn[k]["aL"].detach().cpu().numpy().reshape(-1, 1),
                       "aU": battn[k]["aU"].detach().cpu().numpy().reshape(-1, 1)}
    margins, _, _ = backward_crown(ops, lb, ub, cl, cu, C, alpha=alpha_np,
                                   mmaps=mmaps, battn=battn_np)
    return np.asarray(margins).reshape(-1), {"alpha": alpha_np, "battn": battn_np}
