#!/usr/bin/env python3
"""Falsification bake-off: for a set of target instances (default = the gold-SAT
instances our sweep MISSED), try a menu of falsification methods/budgets and
report which finds an onnxruntime-VALID counterexample, and how fast.

Falsification runs before reach in the real runner and carries zero soundness
risk (every CE is ORT-revalidated), so any method that cracks a missed gold-SAT
is a free +10. This harness is how we pick the per-benchmark falsify config.

Usage: falsify_bakeoff.py CATEGORY [CATEGORY...]   # bakes off missed-SATs
       falsify_bakeoff.py --instance CAT ONNX VNNLIB
"""
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

_RUNNER_DIR = "/isis/home/sasakis/v/tools/n2v/examples/Submission/VNN_COMP2026"
sys.path.insert(0, _RUNNER_DIR)
sys.path.insert(0, "/isis/home/sasakis/v/tools/n2v/examples/VNN-COMP")

import n2v  # noqa: E402
from n2v.utils import load_vnnlib, falsify  # noqa: E402
from n2v.utils.model_loader import load_onnx  # noqa: E402
from n2v.utils.onnx_validate import onnx_forward, in_unsafe_region  # noqa: E402
from n2v.utils.falsify import _extract_halfspace_groups  # noqa: E402

BENCH = "/isis/home/sasakis/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks"
BENCH2026 = "/isis/home/sasakis/v/other/VNNCOMP/vnncomp2026_benchmarks/benchmarks"
GOLD_DIR = "/isis/home/sasakis/v/other/VNNCOMP/vnncomp2025_results"
TOOLS = ["alpha_beta_crown", "cora", "neuralsat", "nnenum", "nnv", "pyrat", "rover", "sobolbox"]
SWEEP = "/isis/home/sasakis/v/tools/n2v/examples/Submission/VNN_COMP2026/sweep/results/sweep/results.csv"

# The strategy menu. Each: (label, method, n_samples, kwargs).
MENU = [
    ("random_100", "random", 100, {}),
    ("random_5k", "random", 5000, {}),
    ("random_50k", "random", 50000, {}),
    ("apgd_nr1_ns30", "random+apgd", 100, {"n_restarts": 1, "n_steps": 30}),
    ("apgd_nr5_ns100", "random+apgd", 200, {"n_restarts": 5, "n_steps": 100}),
    ("square_20k", "random+square", 200, {"n_iters": 20000}),
    ("strong", "strong", 1000, {}),
]


def bn(p):
    return os.path.basename(p.strip())


def build_gold():
    agg = defaultdict(lambda: {"sat": 0, "unsat": 0})
    for t in TOOLS:
        p = os.path.join(GOLD_DIR, t, "results.csv")
        if not os.path.isfile(p):
            continue
        for row in csv.reader(open(p)):
            if len(row) < 5:
                continue
            k = (row[0].strip(), bn(row[1]), bn(row[2]))
            r = row[4].strip().lower()
            if r == "sat":
                agg[k]["sat"] += 1
            elif r == "unsat":
                agg[k]["unsat"] += 1
    return {k: ("sat" if v["sat"] else ("unsat" if v["unsat"] else "unknown"))
            for k, v in agg.items()}


def resolve(cat, onnx_bn, vnnlib_bn):
    """Find the onnx/vnnlib on disk for a (cat, basename) pair."""
    for base in (BENCH, BENCH2026):
        for verdir in ("", "2.0", "1.0"):
            od = os.path.join(base, cat, verdir, "onnx", onnx_bn)
            vd = os.path.join(base, cat, verdir, "vnnlib", vnnlib_bn)
            if os.path.isfile(od) and os.path.isfile(vd):
                return od, vd
            # some corpora keep flat onnx/ at the category root
    # glob fallback
    import glob
    for base in (BENCH, BENCH2026):
        o = glob.glob(os.path.join(base, cat, "**", onnx_bn), recursive=True)
        v = glob.glob(os.path.join(base, cat, "**", vnnlib_bn), recursive=True)
        if o and v:
            return o[0], v[0]
    return None, None


def missed_sats(cats):
    gold = build_gold()
    rows = list(csv.DictReader(open(SWEEP)))
    out = []
    for r in rows:
        if r["category"] not in cats:
            continue
        k = (r["category"], bn(r["onnx"]), bn(r["vnnlib"]))
        if gold.get(k) == "sat" and r["result"].strip().lower() not in ("sat", "unsat"):
            od, vd = resolve(*k)
            if od:
                out.append((r["category"], od, vd))
            else:
                print(f"  ! could not resolve {k}", file=sys.stderr)
    return out


def try_falsify(model, vnnlib_path, onnx_path, method, n_samples, kwargs):
    """Return (found_valid_ce, secs). Mirrors runner stage-1 (per-pair + ORT gate)."""
    prop = load_vnnlib(vnnlib_path)
    pairs = prop.get("pairs", [])
    ishape = None
    import onnx
    m = onnx.load(onnx_path)
    init = {i.name for i in m.graph.initializer}
    inp = [i for i in m.graph.input if i.name not in init][0]
    dims = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim)
    if len(dims) > 1 and dims[0] in (0, 1):
        dims = dims[1:]
    ishape = dims
    t0 = time.time()
    per = max(20, n_samples // max(len(pairs), 1))
    for pair in pairs:
        try:
            lb = np.asarray(pair["lb"], dtype=np.float64).reshape(ishape)
            ub = np.asarray(pair["ub"], dtype=np.float64).reshape(ishape)
            res, cex = falsify(model, lb, ub, pair["prop"], method=method,
                               n_samples=per, seed=42, **kwargs)
            if res == 0 and cex is not None:
                y = onnx_forward(onnx_path, cex[0])
                if in_unsafe_region(y, _extract_halfspace_groups(pair["prop"])):
                    return True, time.time() - t0
        except Exception as e:  # noqa: BLE001
            pass
    return False, time.time() - t0


def main():
    args = sys.argv[1:]
    if args and args[0] == "--instance":
        cat, od, vd = args[1], args[2], args[3]
        targets = [(cat, od, vd)]
    else:
        cats = set(args) if args else {"sat_relu"}
        targets = missed_sats(cats)

    n2v.set_parallel(True, n_workers=int(os.environ.get("N2V_WORKERS", "16")))
    n2v.set_lp_solver("linprog")

    print(f"Bake-off over {len(targets)} missed gold-SAT instances\n")
    labels = [m[0] for m in MENU]
    print(f"{'instance':<42}" + "".join(f"{l[:13]:>14}" for l in labels))
    wins = defaultdict(int)
    for cat, od, vd in targets:
        model = load_onnx(od)
        cells = []
        for label, method, ns, kw in MENU:
            found, secs = try_falsify(model, vd, od, method, ns, kw)
            cells.append(f"{'OK' if found else '..'} {secs:5.1f}s")
            if found:
                wins[label] += 1
        name = f"{cat[:14]}/{bn(od)[:26]}"
        print(f"{name:<42}" + "".join(f"{c:>14}" for c in cells), flush=True)
    print(f"\nwins by strategy (out of {len(targets)}):")
    for label in labels:
        print(f"  {label:<20}{wins[label]}")


if __name__ == "__main__":
    main()
