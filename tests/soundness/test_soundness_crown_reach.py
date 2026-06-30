"""Soundness of the LP-free CROWN op-DAG engine (``n2v.nn.crown_reach``), the
n2v port of NNV's ``ViTCrown`` attention verifier.

Every relaxation (relu triangle, exp/reciprocal convex envelopes, eprod/bmatmul
McCormick) is a sound over-approximation, so the backward bound must be a true
LOWER bound and the forward/CROWN boxes must ENCLOSE every reachable value. The
tests build a self-contained single-head attention+FF DAG (exercising both
bilinear modes and the full softmax decomposition ``exp->sum->reciprocal->
broadcast->eprod``) and check, by Monte-Carlo over the input box:

  (1) forward IBP boxes enclose every sampled op value;
  (2) the backward CROWN margin is <= every sampled true margin;
  (3) CROWN intermediate-bound refinement stays sound (still encloses) and is no
      looser than IBP;
  (4) alpha-optimization's RE-EVALUATED margin is still sound (<= true) -- the
      'sound double re-evaluation' guarantee (any alpha in [0,1] is sound).

No LP solver is involved anywhere.
"""
import numpy as np
import pytest

from n2v.nn import crown_reach as cr


# --------------------------------------------------------------------------- #
#  build a self-contained single-head attention + FF DAG over a token state    #
# --------------------------------------------------------------------------- #
def _build_attn_dag(N, D, rng, with_ff=True):
    """input = token state (dim N*D, e-major) -> Q,K,V per-state affine ->
    S=scale*QK^T -> softmax(decomposed) -> O=A*V -> [FF relu] -> scalar margin."""
    E = D                                   # single head
    nin = N * E
    ops = []

    def push(op):
        ops.append(op); return len(ops) - 1

    def aff(in_idx, W, b, mat):
        return dict(type="affine", **{"in": in_idx},
                    W=np.asarray(W, float), b=np.asarray(b, float).reshape(-1),
                    dim=np.asarray(W).shape[0], mat=mat)

    push(dict(type="input", **{"in": []}, dim=nin, mat=None))
    Wq = 0.5 * rng.standard_normal((nin, nin)); bq = 0.1 * rng.standard_normal(nin)
    Wk = 0.5 * rng.standard_normal((nin, nin)); bk = 0.1 * rng.standard_normal(nin)
    Wv = 0.5 * rng.standard_normal((nin, nin)); bv = 0.1 * rng.standard_normal(nin)
    q = push(aff(0, Wq, bq, (N, E)))
    k = push(aff(0, Wk, bk, (N, E)))
    v = push(aff(0, Wv, bv, (N, E)))
    scale = 1.0 / np.sqrt(D)
    sc = push(dict(type="bmatmul", **{"in": [q, k]}, mode="abt",
                   ra=N, ca=D, rb=N, cb=D, dim=N * N, mat=(N, N)))
    scs = push(aff(sc, scale * np.eye(N * N), np.zeros(N * N), (N, N)))
    expi = push(dict(type="exp", **{"in": scs}, dim=N * N, mat=(N, N)))
    Wsum = np.zeros((N, N * N))
    for i in range(N):
        Wsum[i, np.arange(N) * N + i] = 1.0
    sumi = push(aff(expi, Wsum, np.zeros(N), (N, 1)))
    reci = push(dict(type="reciprocal", **{"in": sumi}, dim=N, mat=(N, 1)))
    Wexp = np.zeros((N * N, N))
    for kk in range(N):
        for i in range(N):
            Wexp[kk * N + i, i] = 1.0
    rexp = push(aff(reci, Wexp, np.zeros(N * N), (N, N)))
    ai = push(dict(type="eprod", **{"in": [expi, rexp]}, dim=N * N, mat=(N, N)))
    o = push(dict(type="bmatmul", **{"in": [ai, v]}, mode="ab",
                  ra=N, ca=N, rb=N, cb=D, dim=N * D, mat=(N, D)))
    last = o
    if with_ff:
        W1 = 0.6 * rng.standard_normal((N * D, N * D)); b1 = 0.1 * rng.standard_normal(N * D)
        f1 = push(aff(o, W1, b1, (N, D)))
        r = push(dict(type="relu", **{"in": f1}, dim=N * D, mat=(N, D)))
        W2 = 0.6 * rng.standard_normal((N * D, N * D)); b2 = 0.1 * rng.standard_normal(N * D)
        last = push(aff(r, W2, b2, (N, D)))
    # DAG ends at the feature output (dim N*D); the 3 'margin' functionals C are
    # applied by the verifier as the backward spec (not baked into the DAG).
    Cm = rng.standard_normal((3, N * D))
    return ops, nin, Cm


def _box(nin, rng, width=0.4):
    c = 0.5 * rng.standard_normal(nin)
    return c - width / 2, c + width / 2


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_forward_ibp_encloses_all_ops(seed):
    rng = np.random.default_rng(seed)
    ops, nin, _ = _build_attn_dag(3, 2, rng)
    lb, ub = _box(nin, rng)
    cl, cu = cr.forward_ibp(ops, lb, ub)
    worst = 0.0
    for _ in range(400):
        xi = lb + (ub - lb) * rng.random(nin)
        _, vals = cr.eval_ops(ops, xi)
        for kk in range(len(ops)):
            worst = max(worst, float(np.max(cl[kk] - vals[kk])),
                        float(np.max(vals[kk] - cu[kk])))
    assert worst < 1e-9, f"forward IBP not sound: overshoot {worst:.2e}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_backward_crown_is_true_lower_bound(seed):
    rng = np.random.default_rng(100 + seed)
    ops, nin, C = _build_attn_dag(3, 2, rng)
    lb, ub = _box(nin, rng)
    cl, cu = cr.forward_ibp(ops, lb, ub)
    mb, _, _ = cr.backward_crown(ops, lb, ub, cl, cu, C)
    mc_min = np.full(C.shape[0], np.inf)
    for _ in range(2000):
        xi = lb + (ub - lb) * rng.random(nin)
        logit, _ = cr.eval_ops(ops, xi)
        mc_min = np.minimum(mc_min, C @ logit)
    assert np.all(mb <= mc_min + 1e-7), \
        f"backward bound exceeds a true margin: {mb} vs MC {mc_min}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_refine_is_sound_and_not_looser(seed):
    rng = np.random.default_rng(200 + seed)
    ops, nin, C = _build_attn_dag(3, 2, rng)
    lb, ub = _box(nin, rng)
    cl0, cu0 = cr.forward_ibp(ops, lb, ub)
    clr, cur, _ = cr.refine_bounds(ops, lb, ub, iters=2)
    # refined boxes still enclose samples, and are no looser than IBP
    for kk in range(len(ops)):
        assert np.all(clr[kk] >= cl0[kk] - 1e-9) and np.all(cur[kk] <= cu0[kk] + 1e-9), \
            f"refine looser than IBP at op {kk}"
    worst = 0.0
    for _ in range(400):
        xi = lb + (ub - lb) * rng.random(nin)
        _, vals = cr.eval_ops(ops, xi)
        for kk in range(len(ops)):
            worst = max(worst, float(np.max(clr[kk] - vals[kk])),
                        float(np.max(vals[kk] - cur[kk])))
    assert worst < 1e-9, f"refined boxes not sound: overshoot {worst:.2e}"
    # refinement should not REDUCE the certified margin (monotone tightening)
    mb0, _, _ = cr.backward_crown(ops, lb, ub, cl0, cu0, C)
    mbr, _, _ = cr.backward_crown(ops, lb, ub, clr, cur, C)
    assert np.all(mbr >= mb0 - 1e-7)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alpha_optimized_margin_still_sound(seed):
    pytest.importorskip("torch")
    rng = np.random.default_rng(300 + seed)
    ops, nin, C = _build_attn_dag(3, 2, rng)
    lb, ub = _box(nin, rng)
    cl, cu, _ = cr.refine_bounds(ops, lb, ub, iters=1)
    margins, sol = cr.optimize_alpha(ops, lb, ub, cl, cu, C, n_iter=20, lr=0.1)
    mc_min = np.full(C.shape[0], np.inf)
    for _ in range(2000):
        xi = lb + (ub - lb) * rng.random(nin)
        logit, _ = cr.eval_ops(ops, xi)
        mc_min = np.minimum(mc_min, C @ logit)
    assert np.all(margins <= mc_min + 1e-7), \
        "alpha-optimized (re-evaluated) margin is not a sound lower bound"


def test_bmatmul_both_modes_mccormick_sound():
    # isolate the two bilinear modes: bound <= true product over the box
    rng = np.random.default_rng(7)
    for mode, (ra, ca, rb, cb) in [("abt", (3, 2, 3, 2)), ("ab", (3, 3, 3, 2))]:
        nin = ra * ca + rb * cb
        ops = [dict(type="input", **{"in": []}, dim=nin, mat=None)]
        Sa = np.zeros((ra * ca, nin)); Sa[:, :ra * ca] = np.eye(ra * ca)
        Sb = np.zeros((rb * cb, nin)); Sb[:, ra * ca:] = np.eye(rb * cb)
        ops.append(dict(type="affine", **{"in": 0}, W=Sa, b=np.zeros(ra * ca),
                        dim=ra * ca, mat=(ra, ca)))
        ops.append(dict(type="affine", **{"in": 0}, W=Sb, b=np.zeros(rb * cb),
                        dim=rb * cb, mat=(rb, cb)))
        dimY = (ra * rb) if mode == "abt" else (cb * ra)
        ops.append(dict(type="bmatmul", **{"in": [1, 2]}, mode=mode,
                        ra=ra, ca=ca, rb=rb, cb=cb, dim=dimY, mat=None))
        C = rng.standard_normal((4, dimY))   # spec applied to the bmatmul output
        lb, ub = _box(nin, rng, width=0.6)
        cl, cu = cr.forward_ibp(ops, lb, ub)
        mb, _, _ = cr.backward_crown(ops, lb, ub, cl, cu, C)
        mc = np.full(4, np.inf)
        for _ in range(3000):
            xi = lb + (ub - lb) * rng.random(nin)
            logit, _ = cr.eval_ops(ops, xi)
            mc = np.minimum(mc, C @ logit)
        assert np.all(mb <= mc + 1e-7), f"{mode} bmatmul bound unsound: {mb} vs {mc}"
