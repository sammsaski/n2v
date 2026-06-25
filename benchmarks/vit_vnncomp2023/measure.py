"""Honest measurement of the sound ViT verifier.

For each model: (a) verified count at full eps=1/255, and (b) the per-instance
certified radius eps* (max eps the verifier certifies, by bisection) for a
sample -- the verifier's real strength, comparable to W11's eps_star table.
Uses fast sound `estimate` margins (a lower bound on the method's strength; LP
margins would be marginally tighter via the ReLU constraints).
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from model import build_model
from reach import ViTReacher
from n2v.sets import ImageStar

MEAN = np.array([0.4914, 0.4822, 0.4465]); STD = np.array([0.2023, 0.1994, 0.2010])
FULL = 1.0 / 255


def imgstar(x, eps):
    lb = (np.clip(x - eps, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
    ub = (np.clip(x + eps, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
    return ImageStar.from_bounds(np.transpose(lb, (1, 2, 0)).reshape(-1).astype(np.float64),
                                 np.transpose(ub, (1, 2, 0)).reshape(-1).astype(np.float64),
                                 32, 32, 3)


def min_margin(R, x, eps, label):
    out = R.reach(imgstar(x, eps))
    m = R.margins(out, label, margin_mode="estimate")
    return min(v for v in m.values() if v is not None)


def eps_star(R, x, label, hi=2.0 / 255, steps=7):
    # bisection on the largest eps with min_margin > 0
    lo = 0.0
    if min_margin(R, x, hi, label) > 0:
        return hi  # certifies beyond the search ceiling
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if min_margin(R, x, mid, label) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def run(name, n_full, n_epsstar):
    z = np.load(os.path.join(HERE, "instances", f"{name}.npz"))
    imgs, labels = z["images"], z["labels"]
    R = ViTReacher(build_model(name, os.path.join(HERE, "onnx", f"{name}.onnx")))
    nver = 0
    print(f"\n== {name}: verified @ full eps=1/255 ==", flush=True)
    for k in range(min(n_full, len(imgs))):
        x = imgs[k].astype(np.float64); y = int(labels[k])
        t0 = time.time(); mm = min_margin(R, x, FULL, y); dt = time.time() - t0
        v = mm > 0; nver += v
        print(f"  [{k}] {'VERIFIED' if v else 'unknown '} margin={mm:+.3f} ({dt:.1f}s)", flush=True)
    print(f"== {name}: {nver}/{min(n_full,len(imgs))} verified at full eps ==", flush=True)
    print(f"== {name}: eps* (x255) for {n_epsstar} instances ==", flush=True)
    es = []
    for k in range(min(n_epsstar, len(imgs))):
        x = imgs[k].astype(np.float64); y = int(labels[k])
        t0 = time.time(); e = eps_star(R, x, y); dt = time.time() - t0
        es.append(e)
        print(f"  [{k}] eps*={e*255:.4f}/255 ({dt:.1f}s)", flush=True)
    if es:
        print(f"== {name}: eps* median={np.median(es)*255:.4f}/255 "
              f"max={np.max(es)*255:.4f}/255 ==", flush=True)
    return nver


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgd-full", type=int, default=30)
    ap.add_argument("--pgd-epsstar", type=int, default=10)
    ap.add_argument("--ibp-full", type=int, default=15)
    ap.add_argument("--ibp-epsstar", type=int, default=5)
    a = ap.parse_args()
    run("pgd_2_3_16", a.pgd_full, a.pgd_epsstar)
    run("ibp_3_3_8", a.ibp_full, a.ibp_epsstar)
    print("\nMEASURE_DONE", flush=True)
