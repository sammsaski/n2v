#!/usr/bin/env python3
"""
Level 1: Compare verification methods on a tiny 2->3->1 network.

Compares Star (exact and approx) with Hexatope and Octatope (approx only).
Verifies soundness by checking that all approx methods' bounds contain the
Star exact bounds (the tightest sound reference).
"""

import time
import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Star, Hexatope, Octatope
from n2v.nn import NeuralNetwork


def build_model():
    """Build a 2->3->1 feedforward network with fixed weights."""
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )
    model.eval()

    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([
            [ 1.0,  0.5],
            [-0.5,  1.0],
            [ 0.3, -0.7],
        ]))
        model[0].bias.copy_(torch.tensor([0.1, -0.2, 0.0]))
        model[2].weight.copy_(torch.tensor([[0.6, -0.4, 0.8]]))
        model[2].bias.copy_(torch.tensor([0.05]))

    return model


def compute_bounds(output_sets, set_type):
    """Compute overall output bounds from a list of output sets (union)."""
    lb_all = np.inf
    ub_all = -np.inf

    for s in output_sets:
        if set_type == "star":
            lb, ub = s.get_ranges()
        else:
            lb, ub = s.get_ranges(solver='lp')
        lb_all = min(lb_all, float(lb.flatten()[0]))
        ub_all = max(ub_all, float(ub.flatten()[0]))

    return lb_all, ub_all


def main():
    print("=" * 70)
    print("Level 1: Tiny Network Comparison (2 -> 3 -> 1)")
    print("=" * 70)

    # Build model and define input bounds
    model = build_model()
    net = NeuralNetwork(model)

    lb = np.array([-1.0, -1.0]).reshape(-1, 1)
    ub = np.array([ 1.0,  1.0]).reshape(-1, 1)

    # Configurations: (label, set_class, set_type_key, method)
    # Hex/oct only support approx mode (exact was removed for soundness)
    configs = [
        ("Star exact",      Star,     "star",     "exact"),
        ("Star approx",     Star,     "star",     "approx"),
        ("Hexatope approx", Hexatope, "hexatope", "approx"),
        ("Octatope approx", Octatope, "octatope", "approx"),
    ]

    results = []

    for label, SetClass, set_key, method in configs:
        input_set = SetClass.from_bounds(lb.astype(np.float64), ub.astype(np.float64))
        t0 = time.time()
        output_sets = net.reach(input_set, method=method)
        elapsed = time.time() - t0
        out_lb, out_ub = compute_bounds(output_sets, set_key)
        n_sets = len(output_sets)
        results.append((label, out_lb, out_ub, n_sets, elapsed))

    # Print summary table
    print()
    header = f"{'Method':<20s} {'LB':>10s} {'UB':>10s} {'#Sets':>6s} {'Time(s)':>8s}"
    print(header)
    print("-" * len(header))
    for label, out_lb, out_ub, n_sets, elapsed in results:
        print(f"{label:<20s} {out_lb:>10.6f} {out_ub:>10.6f} {n_sets:>6d} {elapsed:>8.4f}")

    # Soundness checks
    star_exact_lb, star_exact_ub = results[0][1], results[0][2]

    print()
    print("=" * 70)
    print("Soundness Checks")
    print("=" * 70)

    tol = 1e-6
    all_pass = True

    # Check: every approx method should over-approximate Star exact
    # i.e., approx_lb <= star_exact_lb and approx_ub >= star_exact_ub
    for label, out_lb, out_ub, _, _ in results:
        if label == "Star exact":
            continue  # skip reference
        lb_ok = out_lb <= star_exact_lb + tol
        ub_ok = out_ub >= star_exact_ub - tol
        ok = lb_ok and ub_ok
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  {label} contains Star exact: {status}")
        if not ok:
            if not lb_ok:
                print(f"    LB FAIL: {out_lb:.8f} > {star_exact_lb:.8f}")
            if not ub_ok:
                print(f"    UB FAIL: {out_ub:.8f} < {star_exact_ub:.8f}")

    print()
    if all_pass:
        print("All soundness checks PASSED.")
    else:
        print("Some soundness checks FAILED!")
    print("=" * 70)

    # Assert for CI / automated testing
    assert all_pass, "Soundness checks failed!"


if __name__ == "__main__":
    main()
