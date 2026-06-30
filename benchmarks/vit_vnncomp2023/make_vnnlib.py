"""Emit a standard VNN-COMP robustness VNNLIB spec for a ViT benchmark instance.

The VNN-COMP 2023 ViT specs are L-inf argmax-robustness properties: a per-pixel
input box (in the ONNX input space, i.e. the *normalized* image — the ONNX expects
normalized input) and an unsafe output condition ``OR_i (Y_i >= Y_label)`` (some
other class beats the true class). ``X`` variables follow the ONNX input order
(C,H,W row-major); ``Y`` are the 10 logits.

Usage:
  python make_vnnlib.py --model ibp_3_3_8 --instance 10 --eps 0.0039215686 \
      --out instances/vit_ibp_idx10.vnnlib
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def make_vnnlib(img01, label, eps, out_path):
    from n2v.nn.vit_crown import eps_box   # normalized box (ONNX input space)
    lb, ub = eps_box(img01, eps)
    n = lb.size
    lines = [f"; ViT robustness, label={label}, eps={eps}, {n} inputs"]
    lines += [f"(declare-const X_{i} Real)" for i in range(n)]
    lines += [f"(declare-const Y_{i} Real)" for i in range(10)]
    lines.append("\n; input box (ONNX input order, C,H,W)")
    for i in range(n):
        lines.append(f"(assert (<= X_{i} {ub[i]:.9f}))")
        lines.append(f"(assert (>= X_{i} {lb[i]:.9f}))")
    lines.append("\n; unsafe: some other class >= the true class (robustness CE)")
    ors = " ".join(f"(and (>= Y_{i} Y_{label}))" for i in range(10) if i != label)
    lines.append(f"(assert (or {ors}))")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibp_3_3_8")
    ap.add_argument("--instance", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1.0 / 255)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(os.path.join(HERE, "instances", f"{a.model}.npz"))
    img = z["images"][a.instance].astype(np.float64); label = int(z["labels"][a.instance])
    out = a.out or os.path.join(HERE, "instances",
                                f"vit_{a.model}_idx{a.instance}.vnnlib")
    make_vnnlib(img, label, a.eps, out)
    print(f"wrote {out}  (label={label}, eps={a.eps})")


if __name__ == "__main__":
    main()
