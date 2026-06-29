#!/usr/bin/env python3
"""Cross-reference our sweep verdicts against VNN-COMP 2025 ground truth.

Gold rule (VNN-COMP): an instance's true result is SAT if ANY tool found a
valid counterexample, else it is assumed UNSAT. Built from the per-tool merged
results.csv files in vnncomp2025_results/.

For each of our swept instances we join on (category, onnx-basename,
vnnlib-basename) and classify against gold, separately flagging UNSATs that came
from a probabilistic (unsound) config -- those are the -150 landmines under the
+10/0/-150 scoring model.

Usage: analyze_gold.py [our_results.csv]
"""
import csv
import os
import sys
from collections import defaultdict

GOLD_DIR = "/isis/home/sasakis/v/other/VNNCOMP/vnncomp2025_results"
TOOLS = ["alpha_beta_crown", "cora", "neuralsat", "nnenum", "nnv", "pyrat", "rover", "sobolbox"]
OUR = sys.argv[1] if len(sys.argv) > 1 else \
    "/isis/home/sasakis/v/tools/n2v/examples/Submission/VNN_COMP2026/sweep/results/sweep/results.csv"

sys.path.insert(0, "/isis/home/sasakis/v/tools/n2v/examples/VNN-COMP")
from benchmark_configs import get_config  # noqa: E402

VERDICTS = {"sat", "unsat", "unknown", "timeout"}


def bn(p):
    return os.path.basename(p.strip())


def build_gold():
    """key (category, onnx_bn, vnnlib_bn) -> {'sat':n,'unsat':n,'tools_sat':set}."""
    agg = defaultdict(lambda: {"sat": 0, "unsat": 0, "tools_sat": set()})
    for tool in TOOLS:
        path = os.path.join(GOLD_DIR, tool, "results.csv")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for row in csv.reader(f):
                if len(row) < 5:
                    continue
                cat, onnx, vnnlib = row[0].strip(), row[1], row[2]
                res = row[4].strip().lower()
                key = (cat, bn(onnx), bn(vnnlib))
                if res == "sat":
                    agg[key]["sat"] += 1
                    agg[key]["tools_sat"].add(tool)
                elif res == "unsat":
                    agg[key]["unsat"] += 1
    gold = {}
    for key, v in agg.items():
        gold[key] = "sat" if v["sat"] > 0 else ("unsat" if v["unsat"] > 0 else "unknown")
    return gold, agg


def is_probabilistic_only(cat, onnx, vnnlib):
    try:
        cfg = get_config(cat, onnx, vnnlib)
    except Exception:
        return False
    methods = [m for m, _ in cfg.get("reach_methods", [])]
    return bool(methods) and all(m == "probabilistic" for m in methods)


def main():
    gold, agg = build_gold()
    print(f"gold instances: {len(gold)}  "
          f"(sat={sum(v=='sat' for v in gold.values())}, "
          f"unsat={sum(v=='unsat' for v in gold.values())})\n")

    rows = list(csv.DictReader(open(OUR)))
    # Per-category tallies.
    cats = defaultdict(lambda: defaultdict(int))
    landmines, missed_sat, false_sat = [], [], []
    matched = 0
    for r in rows:
        cat, onnx, vnnlib = r["category"], r["onnx"], r["vnnlib"]
        ours = r["result"].strip().lower()
        ours = ours if ours in VERDICTS else "error"
        key = (cat, bn(onnx), bn(vnnlib))
        g = gold.get(key)
        C = cats[cat]
        C["n"] += 1
        if g is None:
            C["no_gold"] += 1
            continue
        matched += 1
        prob = is_probabilistic_only(cat, onnx, vnnlib)
        if ours == "sat":
            if g == "sat":
                C["correct_sat"] += 1
            else:
                C["FALSE_SAT"] += 1
                false_sat.append((cat, key[1], key[2], g))
        elif ours == "unsat":
            if g == "sat":
                C["FALSE_UNSAT"] += 1
                landmines.append((cat, key[1], key[2], prob, sorted(agg[key]["tools_sat"])))
            else:
                C["correct_unsat" + ("_PROB" if prob else "")] += 1
        else:  # unknown/timeout/error
            if g == "sat":
                C["missed_sat"] += 1
                missed_sat.append((cat, key[1], key[2], ours, sorted(agg[key]["tools_sat"])))
            elif g == "unsat":
                C["missed_unsat"] += 1
            else:
                C["both_unknown"] += 1

    # Report.
    hdr = ["cat", "n", "match", "cor_sat", "cor_uns", "uns_PROB",
           "FALSE_UNS", "miss_sat", "miss_uns"]
    print(f"{'category':<40}{'n':>4}{'mat':>4}{'cSAT':>5}{'cUNS':>5}"
          f"{'cUNS_P':>7}{'F_UNS':>6}{'mSAT':>5}{'mUNS':>5}")
    tot = defaultdict(int)
    for cat in sorted(cats):
        C = cats[cat]
        print(f"{cat:<40}{C['n']:>4}{C['n']-C['no_gold']:>4}{C['correct_sat']:>5}"
              f"{C['correct_unsat']:>5}{C['correct_unsat_PROB']:>7}"
              f"{C['FALSE_UNSAT']:>6}{C['missed_sat']:>5}{C['missed_unsat']:>5}")
        for k, v in C.items():
            tot[k] += v
    print("-" * 81)
    print(f"{'TOTAL':<40}{tot['n']:>4}{tot['n']-tot['no_gold']:>4}{tot['correct_sat']:>5}"
          f"{tot['correct_unsat']:>5}{tot['correct_unsat_PROB']:>7}"
          f"{tot['FALSE_UNSAT']:>6}{tot['missed_sat']:>5}{tot['missed_unsat']:>5}")

    print(f"\n=== {len(landmines)} CURRENT FALSE-UNSAT LANDMINES (gold=SAT, we said unsat) ===")
    for cat, o, v, prob, ts in landmines:
        print(f"  [{'PROB' if prob else 'SOUND!'}] {cat:<28}{o[:34]:<34}{v[:28]:<28} sat-tools={ts}")

    print(f"\n=== current-UNSAT-from-PROBABILISTIC summary (zero-risk: these become unknown) ===")
    pc = defaultdict(int)
    for r in rows:
        if r["result"].strip().lower() == "unsat" and \
           is_probabilistic_only(r["category"], r["onnx"], r["vnnlib"]):
            pc[r["category"]] += 1
    for cat, n in sorted(pc.items()):
        print(f"  {cat:<40}{n} unsat via probabilistic -> would concede to unknown")
    print(f"  total probabilistic-unsat we'd drop: {sum(pc.values())}")

    print(f"\n=== {len(missed_sat)} MISSED SATs (gold=SAT, we unknown/timeout) -- falsification targets ===")
    ms = defaultdict(int)
    for cat, o, v, ours, ts in missed_sat:
        ms[cat] += 1
    for cat, n in sorted(ms.items(), key=lambda x: -x[1]):
        print(f"  {cat:<40}{n} missed sat")
    if false_sat:
        print(f"\n!!! {len(false_sat)} FALSE SAT (gold=unsat, we said sat) -- investigate ORT gate !!!")
        for cat, o, v, g in false_sat:
            print(f"  {cat:<28}{o[:34]:<34}{v}")


if __name__ == "__main__":
    main()
