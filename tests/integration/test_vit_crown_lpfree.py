"""End-to-end check of the LP-free CROWN ViT verifier (n2v port of NNV
``ViTCrown``) on the VNN-COMP 2023 benchmark.

  (1) Orientation gate: the lowered op DAG reproduces the torch ViT forward pass
      to machine precision (this validates every layout convention: e-major
      token state, column-major matrix flatten, kron per-token maps, head
      slice/concat, the conv patch-embed affine, and the softmax decomposition).
  (2) Single-shot soundness: the backward CROWN margin is a true lower bound (no
      LP), matched against Monte-Carlo samples of the torch model.
  (3) [slow] NNV "instance 11" (0-based idx 10) is VERIFIED robust LP-free with
      the full refine + alpha recipe, and the certified margin is never violated.

The benchmark assets live under ``benchmarks/vit_vnncomp2023`` (ONNX + instance
npz); the test skips cleanly if they are absent.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "benchmarks", "vit_vnncomp2023")
ONNX = os.path.join(BENCH, "onnx", "ibp_3_3_8.onnx")
NPZ = os.path.join(BENCH, "instances", "ibp_3_3_8.npz")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(ONNX) and os.path.exists(NPZ)),
    reason="ViT benchmark assets (onnx/instances) not present")

sys.path.insert(0, BENCH)


def _load():
    from model import build_model
    import vit_crown as vc
    from n2v.nn import crown_reach as cr
    m = build_model("ibp_3_3_8", ONNX).double()   # lower in float64 (exact BN affine)
    ops = vc.to_ops(m)
    z = np.load(NPZ)
    return m, ops, vc, cr, z["images"], z["labels"]


def test_orientation_gate_dag_matches_torch():
    m, ops, vc, cr, imgs, labels = _load()
    md = m
    maxerr = 0.0
    for k in range(5):
        x = imgs[k].astype(np.float64)
        logit_dag, _ = cr.eval_ops(ops, vc.norm_img(x))
        xt = (x - vc.MEAN[:, None, None]) / vc.STD[:, None, None]
        with torch.no_grad():
            logit_t = md(torch.tensor(xt[None], dtype=torch.float64)).numpy().reshape(-1)
        maxerr = max(maxerr, float(np.max(np.abs(logit_dag - logit_t))))
    assert maxerr < 1e-10, f"DAG/torch parity failed: {maxerr:.2e}"


def test_single_shot_backward_is_sound_lower_bound():
    m, ops, vc, cr, imgs, labels = _load()
    md = m
    k = 10; x = imgs[k].astype(np.float64); y = int(labels[k])
    lb, ub = vc.eps_box(x, 1 / 255); C = vc.margin_spec(y)
    cl, cu = cr.forward_ibp(ops, lb, ub)
    mb, _, _ = cr.backward_crown(ops, lb, ub, cl, cu, C)
    rng = np.random.default_rng(0)
    Xn = lb + (ub - lb) * rng.random((4000, lb.size))
    with torch.no_grad():
        lg = md(torch.tensor(Xn.reshape(-1, 3, 32, 32), dtype=torch.float64)).numpy()
    tm = (C @ lg.T).T
    assert np.all(mb <= tm.min(axis=0) + 1e-7), "single-shot bound exceeds a true margin"


@pytest.mark.slow
def test_instance11_verified_lpfree_and_sound():
    m, ops, vc, cr, imgs, labels = _load()
    md = m
    k = 10; x = imgs[k].astype(np.float64); y = int(labels[k])   # NNV "instance 11"
    robust, margins, _ = vc.verify_instance(ops, x, y, eps=1 / 255, refine=True,
                                            refine_iters=2, alpha=True, alpha_iter=60)
    cert = float(margins.min())
    assert robust and cert > 0, f"inst 11 should verify LP-free, got min margin {cert}"
    # the certified lower bound must never exceed a true margin (soundness)
    lb, ub = vc.eps_box(x, 1 / 255); C = vc.margin_spec(y)
    rng = np.random.default_rng(1)
    Xn = lb + (ub - lb) * rng.random((20000, lb.size))
    with torch.no_grad():
        lg = md(torch.tensor(Xn.reshape(-1, 3, 32, 32), dtype=torch.float64)).numpy()
    assert (C @ lg.T).T.min() >= cert - 1e-6, "certified margin violated by a sample"
