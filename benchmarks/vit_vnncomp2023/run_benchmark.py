"""Run sound verification over the VNN-COMP 2023 ViT benchmark instances.

For each (model, instance) the input is the L-inf eps-box around a CIFAR image
in normalized space; an instance is VERIFIED iff every margin Y_label - Y_i has
a positive lower bound. Tallies verified/unknown/falsified and writes a CSV.

Instance set: the first ``num_instances`` rows of each model's index.txt
(deterministic, reproducible). The competition draws 100 per model via a seed;
this fixed selection defines a reproducible 200-instance set.

Data (data.pt, index.txt) is read from the W11 mirror by default; pass --data-root
to point elsewhere. ONNX is the vendored copy under onnx/.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root for `n2v`
sys.path.insert(0, HERE)
from model import build_model           # noqa: E402
from reach import ViTReacher            # noqa: E402
from n2v.sets import ImageStar          # noqa: E402

MEAN = np.array([0.4914, 0.4822, 0.4465])
STD = np.array([0.2023, 0.1994, 0.2010])
DEFAULT_DATA = ("C:/Users/veriv/Documents/nnVLA/research/"
                "W11-vit-vnncomp-reconciliation/code/external/ViT_vnncomp2023/models")
MODELS = {"pgd_2_3_16": dict(patch=16, depth=2),
          "ibp_3_3_8": dict(patch=8, depth=3)}


def input_box(x_pix, eps):
    lb = (np.clip(x_pix - eps, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
    ub = (np.clip(x_pix + eps, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
    return lb.astype(np.float64), ub.astype(np.float64)


def run_model(name, num_instances, eps, mode, data_root, writer, relax_factor):
    onnx_path = os.path.join(HERE, "onnx", f"{name}.onnx")
    m = build_model(name, onnx_path)
    R = ViTReacher(m, mode=mode, relax_factor=relax_factor)
    npz = os.path.join(HERE, "instances", f"{name}.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        imgs, labels = z["images"], z["labels"]
    else:  # fall back to the external data.pt mirror
        data = torch.load(os.path.join(data_root, name, "data.pt"))
        imgs, labels = data[0].numpy(), data[1].numpy()
    n = min(num_instances, len(imgs))
    nver = 0
    for k in range(n):
        x = np.asarray(imgs[k])
        y = int(labels[k])
        lb, ub = input_box(x, eps)
        t0 = time.time()
        try:
            res = R.verify(lb, ub, y)
            status = res["status"]
            mm = min((v for v in res["margins"].values() if v is not None),
                     default=None)
        except Exception as e:  # noqa: BLE001
            status, mm = f"error:{type(e).__name__}", None
        dt = time.time() - t0
        nver += (status == "verified")
        writer.writerow([name, k, y, status, mm, f"{dt:.2f}"])
        print(f"[{name} {k+1}/{n}] {status:9s} "
              f"margin={mm if mm is None else round(mm,4)} ({dt:.1f}s) "
              f"running_verified={nver}")
    return nver, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--num-instances", type=int, default=100)
    ap.add_argument("--eps", type=float, default=1.0 / 255)
    ap.add_argument("--mode", default="concretize")
    ap.add_argument("--relax-factor", type=float, default=0.5)
    ap.add_argument("--data-root", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(HERE, "results.csv"))
    args = ap.parse_args()

    total_v, total_n = 0, 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instance", "label", "status", "min_margin", "time_s"])
        for name in args.models:
            v, n = run_model(name, args.num_instances, args.eps, args.mode,
                             args.data_root, w, args.relax_factor)
            total_v += v
            total_n += n
            print(f"== {name}: {v}/{n} verified ==")
    print(f"==== TOTAL: {total_v}/{total_n} verified "
          f"(eps={args.eps*255:.3f}/255, mode={args.mode}) ====")


if __name__ == "__main__":
    main()
