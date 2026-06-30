"""Run the LP-free CROWN ViT verifier (method translated from NNV ``ViTCrown``) on the VNN-COMP
2023 benchmark instances.

Pipeline per instance (NO LP solver anywhere): forward IBP -> CROWN
intermediate-bound refinement -> backward CROWN (+ optional alpha-optimization).
An instance is VERIFIED robust iff every margin lower bound is > 0.

Examples
--------
  # verify NNV "instance 11" (0-based idx 10) with the full recipe
  python run_vit_crown.py --model ibp_3_3_8 --instances 10 --alpha

  # reproduce NNV's near-miss generalization table
  python run_vit_crown.py --model ibp_3_3_8 --instances 10 28 0 88 45 34 86 --alpha

  # first N instances, refine-only (faster, no alpha)
  python run_vit_crown.py --model ibp_3_3_8 --num 20
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from model import build_model            # noqa: E402
import vit_crown as vc                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibp_3_3_8", choices=["ibp_3_3_8", "pgd_2_3_16"])
    ap.add_argument("--instances", type=int, nargs="*", default=None,
                    help="explicit 0-based instance indices")
    ap.add_argument("--num", type=int, default=None, help="first N instances")
    ap.add_argument("--eps", type=float, default=1.0 / 255)
    ap.add_argument("--refine-iters", type=int, default=2)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--alpha", action="store_true", help="run alpha-optimization")
    ap.add_argument("--alpha-iter", type=int, default=60)
    ap.add_argument("--alpha-lr", type=float, default=0.1)
    a = ap.parse_args()

    m = build_model(a.model, os.path.join(HERE, "onnx", f"{a.model}.onnx")).double()
    ops = vc.to_ops(m)
    z = np.load(os.path.join(HERE, "instances", f"{a.model}.npz"))
    imgs, labels = z["images"], z["labels"]

    if a.instances is not None:
        idxs = a.instances
    elif a.num is not None:
        idxs = list(range(min(a.num, len(imgs))))
    else:
        idxs = [10]

    print(f"== {a.model}: LP-free CROWN verify, eps={a.eps*255:.3f}/255, "
          f"refine={'off' if a.no_refine else a.refine_iters}, alpha={a.alpha} ==",
          flush=True)
    nver = 0
    for k in idxs:
        x = imgs[k].astype(np.float64); y = int(labels[k])
        t0 = time.time()
        robust, margins, _ = vc.verify_instance(
            ops, x, y, eps=a.eps, refine=not a.no_refine, refine_iters=a.refine_iters,
            alpha=a.alpha, alpha_iter=a.alpha_iter, alpha_lr=a.alpha_lr)
        dt = time.time() - t0
        nver += robust
        print(f"  [{k:3d}] label={y} min_margin={float(margins.min()):+.5f} "
              f"{'VERIFIED' if robust else 'unknown '} ({dt:.0f}s)", flush=True)
    print(f"== VERIFIED {nver}/{len(idxs)} ==", flush=True)


if __name__ == "__main__":
    main()
