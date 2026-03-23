#!/usr/bin/env python3
"""
Level 2: Compare Star, Hexatope, and Octatope on a Small Network

Runs a 5->10->5->1 feedforward network with fixed random weights and compares
Star (exact/approx) with Hexatope and Octatope (approx only, with LP and
MCF solver variants for bounds extraction).

Usage:
    python compare_small.py                  # default epsilon=0.05
    python compare_small.py --epsilon 0.1    # larger perturbation
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn

import n2v
from n2v.sets import Star, Hexatope, Octatope
from n2v.nn import NeuralNetwork

n2v.set_lp_solver('linprog')


def build_model():
    """Create a 5->10->5->1 network with fixed random weights."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )
    model.eval()
    return model


def get_bounds_for_sets(output_sets, set_type, solver=None):
    """
    Compute aggregate lower/upper bounds across a list of output sets.

    Returns (lb, ub, elapsed) where lb/ub are 1-D arrays and elapsed is
    the wall-clock time for bounds extraction.
    """
    t0 = time.time()
    lb_all = None
    ub_all = None
    for s in output_sets:
        if set_type == 'star':
            lb_i, ub_i = s.get_ranges()
        else:
            lb_i, ub_i = s.get_ranges(solver=solver)
        lb_i = lb_i.flatten()
        ub_i = ub_i.flatten()
        if lb_all is None:
            lb_all = lb_i.copy()
            ub_all = ub_i.copy()
        else:
            lb_all = np.minimum(lb_all, lb_i)
            ub_all = np.maximum(ub_all, ub_i)
    elapsed = time.time() - t0
    return lb_all, ub_all, elapsed


def run_config(net, input_set, method, set_type, solver=None, label=""):
    """
    Run reachability + bounds extraction for one configuration.

    Returns a dict with timing breakdown and bounds, or None on failure.
    """
    # --- reach ---
    t0 = time.time()
    try:
        output_sets = net.reach(input_set, method=method)
    except Exception as e:
        print(f"  {label}: REACH FAILED -- {e}")
        return None
    reach_time = time.time() - t0

    # --- bounds ---
    try:
        lb, ub, bounds_time = get_bounds_for_sets(output_sets, set_type, solver)
    except Exception as e:
        print(f"  {label}: BOUNDS FAILED -- {e}")
        return None

    total_time = reach_time + bounds_time
    return {
        'label': label,
        'method': method,
        'set_type': set_type,
        'solver': solver,
        'n_sets': len(output_sets),
        'lb': lb,
        'ub': ub,
        'reach_time': reach_time,
        'bounds_time': bounds_time,
        'total_time': total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Level 2: Compare Star/Hexatope/Octatope on a small network")
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Perturbation radius around center (default: 0.05)')
    args = parser.parse_args()
    eps = args.epsilon

    print("=" * 72)
    print("Level 2: Small Network Comparison  (5 -> 10 -> 5 -> 1)")
    print("=" * 72)

    # --- Model ---
    model = build_model()
    net = NeuralNetwork(model)

    # --- Input bounds ---
    np.random.seed(0)
    center = np.array([0.5, -0.3, 0.1, 0.8, -0.2])
    lb = (center - eps).reshape(-1, 1).astype(np.float64)
    ub = (center + eps).reshape(-1, 1).astype(np.float64)
    print(f"\nInput dimension : 5")
    print(f"Epsilon         : {eps}")
    print(f"Center          : {center}")
    print(f"Bounds          : [{center[0]-eps:.4f}, {center[0]+eps:.4f}] x ... "
          f"x [{center[-1]-eps:.4f}, {center[-1]+eps:.4f}]")

    # --- Build input sets ---
    star_in = Star.from_bounds(lb, ub)
    hex_in = Hexatope.from_bounds(lb, ub)
    oct_in = Octatope.from_bounds(lb, ub)

    # --- Define configurations ---
    # Hex/oct only support approx mode (exact was removed for soundness)
    configs = [
        # (input_set, method, set_type, solver, label)
        (star_in,  'exact',  'star',     None,  'Star exact'),
        (star_in,  'approx', 'star',     None,  'Star approx'),
        (hex_in,   'approx', 'hexatope', 'lp',  'Hexatope approx (LP)'),
        (oct_in,   'approx', 'octatope', 'lp',  'Octatope approx (LP)'),
    ]

    # --- Run all ---
    results = []
    print(f"\n{'─' * 72}")
    print(f"{'Configuration':<30s}  {'Reach':>7s}  {'Bounds':>7s}  "
          f"{'Total':>7s}  {'Sets':>4s}")
    print(f"{'─' * 72}")

    for (inp, method, stype, solver, label) in configs:
        r = run_config(net, inp, method, stype, solver=solver, label=label)
        if r is not None:
            results.append(r)
            print(f"  {label:<28s}  {r['reach_time']:7.3f}s  "
                  f"{r['bounds_time']:7.3f}s  {r['total_time']:7.3f}s  "
                  f"{r['n_sets']:>4d}")
        else:
            results.append(None)

    # --- Reference: Star exact bounds ---
    ref = results[0]  # Star exact
    if ref is None:
        print("\nStar exact failed; cannot compute comparisons.")
        return

    ref_lb = ref['lb']
    ref_ub = ref['ub']
    ref_width = ref_ub - ref_lb  # per-dimension width

    # --- Bound tightness table ---
    print(f"\n{'─' * 72}")
    print("Bound Tightness vs Star Exact")
    print(f"{'─' * 72}")
    print(f"  {'Configuration':<28s}  {'LB':>10s}  {'UB':>10s}  "
          f"{'Width':>10s}  {'Overhead':>10s}")
    print(f"  {'':─<28s}  {'':─>10s}  {'':─>10s}  {'':─>10s}  {'':─>10s}")

    for r in results:
        if r is None:
            continue
        w = r['ub'] - r['lb']
        # Average width overhead across dimensions
        if np.all(ref_width > 0):
            overhead_pct = np.mean((w - ref_width) / ref_width) * 100.0
        else:
            overhead_pct = 0.0
        overhead_str = f"{overhead_pct:+.2f}%"

        print(f"  {r['label']:<28s}  {r['lb'][0]:>10.6f}  {r['ub'][0]:>10.6f}  "
              f"{w[0]:>10.6f}  {overhead_str:>10s}")

    # --- Soundness checks ---
    print(f"\n{'─' * 72}")
    print("Soundness Checks")
    print(f"{'─' * 72}")

    tol = 1e-6
    all_pass = True

    for r in results:
        if r is None or r['label'] == 'Star exact':
            continue

        # Every approx method should over-approximate Star exact:
        # approx_lb <= star_exact_lb and approx_ub >= star_exact_ub
        lb_ok = np.all(r['lb'] <= ref_lb + tol)
        ub_ok = np.all(r['ub'] >= ref_ub - tol)
        ok = lb_ok and ub_ok

        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  {r['label']} contains Star exact: {status}")
        if not ok:
            if not lb_ok:
                diffs = r['lb'] - ref_lb
                worst = np.max(diffs)
                print(f"    LB FAIL: worst violation = {worst:.8f}")
            if not ub_ok:
                diffs = ref_ub - r['ub']
                worst = np.max(diffs)
                print(f"    UB FAIL: worst violation = {worst:.8f}")

    print()
    if all_pass:
        print("All soundness checks PASSED.")
    else:
        print("Some soundness checks FAILED!")

    # --- Summary ---
    print(f"\n{'─' * 72}")
    print("Reference Bounds (Star exact)")
    print(f"{'─' * 72}")
    for dim_i in range(len(ref_lb)):
        print(f"  dim {dim_i}: [{ref_lb[dim_i]:.6f}, {ref_ub[dim_i]:.6f}]  "
              f"width={ref_width[dim_i]:.6f}")

    print(f"\n{'=' * 72}")
    print("Done.")

    # Assert for CI / automated testing
    assert all_pass, "Soundness checks failed!"


if __name__ == "__main__":
    main()
