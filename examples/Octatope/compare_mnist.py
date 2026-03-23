"""
MNIST Verification: Star vs Hexatope

Compares verification approaches on the fc_mnist_small model (784->32->16->10):
  1. Star exact (LP)           — ground truth, exponential splitting
  2. Star approx (LP)          — triangle relaxation, polynomial
  3. Hexatope approx (LP)      — DCS bounding box, sequential LP

For the Star methods, the LP calls are in get_range (neuron classification)
and are fast because stars can represent arbitrary half-spaces natively.

For the Hexatope methods, the bottleneck is intersect_half_space (Algorithm 5.1),
which requires n*(n-1) optimizations where n = number of generators (784 for MNIST).

Usage:
    python compare_mnist.py
    python compare_mnist.py --epsilon 0.01
    python compare_mnist.py --layers 1        # only first layer (fast)
"""

import argparse
import numpy as np
import time
import sys
import os

import torch
import torch.nn as nn

import n2v
from n2v.sets import Star, Hexatope
from n2v.nn import NeuralNetwork


def load_mnist_model():
    """Load the fc_mnist_small model (784 -> 32 -> 16 -> 10)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "CompareNNV", "models",
                              "fc_mnist_small", "fc_mnist_small.pth")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please ensure examples/CompareNNV/models/fc_mnist_small/ exists.")
        sys.exit(1)

    state = torch.load(model_path, map_location='cpu', weights_only=False)

    model = nn.Sequential(
        nn.Linear(784, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
    )
    model[0].weight.data = state['fc1.weight']
    model[0].bias.data = state['fc1.bias']
    model[2].weight.data = state['fc2.weight']
    model[2].bias.data = state['fc2.bias']
    model[4].weight.data = state['fc3.weight']
    model[4].bias.data = state['fc3.bias']
    model.eval()
    return model


def make_partial_model(model, n_layers):
    """Extract the first n_layers (Linear+ReLU pairs) from the model."""
    # model is: Linear, ReLU, Linear, ReLU, Linear
    # n_layers=1 => Linear, ReLU (indices 0,1)
    # n_layers=2 => Linear, ReLU, Linear, ReLU (indices 0,1,2,3)
    # n_layers=3 => full model (indices 0,1,2,3,4)
    if n_layers >= 3:
        return model
    end_idx = n_layers * 2  # each layer is Linear+ReLU
    return nn.Sequential(*list(model.children())[:end_idx])


def extract_bounds(output_sets, set_type='star'):
    """Extract overall bounds from a list of output sets."""
    lbs, ubs = [], []
    for s in output_sets:
        if set_type == 'star':
            lb, ub = s.get_ranges()
        else:
            lb, ub = s.get_ranges(solver='lp')
        lbs.append(lb)
        ubs.append(ub)
    overall_lb = np.min(lbs, axis=0)
    overall_ub = np.max(ubs, axis=0)
    return overall_lb, overall_ub


def run_config(net, input_set, method, set_type, label, timeout=None):
    """Run a single verification configuration with timing."""
    print(f"  {label}...", end='', flush=True)
    t0 = time.time()
    try:
        output_sets = net.reach(input_set, method=method)
        t_reach = time.time() - t0

        t1 = time.time()
        lb, ub = extract_bounds(output_sets, set_type)
        t_bounds = time.time() - t1

        t_total = t_reach + t_bounds
        print(f" {t_total:.1f}s ({len(output_sets)} sets)")
        return {
            'label': label,
            'lb': lb, 'ub': ub,
            'n_sets': len(output_sets),
            'reach_time': t_reach,
            'bounds_time': t_bounds,
            'total_time': t_total,
            'success': True,
        }
    except Exception as e:
        t_total = time.time() - t0
        print(f" FAILED ({t_total:.1f}s): {e}")
        return {
            'label': label,
            'success': False,
            'total_time': t_total,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="MNIST verification: Star vs Hexatope")
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="L-inf perturbation radius (default: 0.02)")
    parser.add_argument("--layers", type=int, default=1,
                        help="Number of layers to verify through (1-3, default: 1)")
    args = parser.parse_args()

    n2v.set_lp_solver('linprog')

    print("=" * 78)
    print("MNIST Verification: Star vs Hexatope")
    print("=" * 78)
    print()

    # Load model
    full_model = load_mnist_model()
    model = make_partial_model(full_model, args.layers)
    layer_desc = {1: "Linear(784,32) + ReLU", 2: "2 layers through ReLU",
                  3: "full network (784->32->16->10)"}
    print(f"Model:   fc_mnist_small")
    print(f"Layers:  {args.layers} — {layer_desc.get(args.layers, str(args.layers))}")
    print(f"Epsilon: {args.epsilon}")
    print()

    # Create input sets
    center = np.full(784, 0.5)
    lb = np.clip(center - args.epsilon, 0, 1).reshape(-1, 1).astype(np.float64)
    ub = np.clip(center + args.epsilon, 0, 1).reshape(-1, 1).astype(np.float64)

    star_input = Star.from_bounds(lb, ub)
    hex_input = Hexatope.from_bounds(lb, ub)

    print(f"Input: 784-dim, epsilon={args.epsilon}")
    print(f"  Star:     dim={star_input.dim}, nVar={star_input.nVar}")
    print(f"  Hexatope: dim={hex_input.dim}, generators={hex_input.generators.shape[1]}")
    print()

    net = NeuralNetwork(model)

    # ── Run configurations ──────────────────────────────────────────

    print("Running verification methods:")
    print("-" * 78)

    results = []

    # 1. Star exact
    r = run_config(net, star_input, 'exact', 'star', 'Star exact (LP)')
    results.append(r)

    # 2. Star approx
    r = run_config(net, star_input, 'approx', 'star', 'Star approx (LP)')
    results.append(r)

    # 3. Hexatope approx (LP) — default solver=None uses LP in bounding box
    r = run_config(net, hex_input, 'approx', 'hex', 'Hexatope approx (LP)')
    results.append(r)

    # ── Results table ───────────────────────────────────────────────

    print()
    print("=" * 78)
    print("Results")
    print("=" * 78)

    # Get star exact as reference
    star_exact = next((r for r in results if r['label'] == 'Star exact (LP)' and r['success']), None)

    print(f"\n{'Method':<28s}  {'Reach':>8s}  {'Bounds':>8s}  {'Total':>8s}  {'Sets':>5s}")
    print("-" * 78)
    for r in results:
        if r['success']:
            print(f"  {r['label']:<26s}  {r['reach_time']:7.1f}s  {r['bounds_time']:7.1f}s  "
                  f"{r['total_time']:7.1f}s  {r['n_sets']:5d}")
        else:
            print(f"  {r['label']:<26s}  {'FAILED':>8s}  {'':>8s}  "
                  f"{r['total_time']:7.1f}s  {'':>5s}")

    # Output bounds comparison (only for successful runs with star exact reference)
    successful = [r for r in results if r['success']]
    if star_exact and len(successful) > 1:
        ref_lb = star_exact['lb']
        ref_ub = star_exact['ub']
        out_dim = len(ref_lb)

        print(f"\nOutput bounds (dim 0..{min(out_dim, 5)-1} of {out_dim}):")
        print(f"  {'Method':<28s}", end='')
        for d in range(min(out_dim, 5)):
            print(f"  {'dim '+str(d):>14s}", end='')
        print()
        print("-" * 78)
        for r in successful:
            print(f"  {r['label']:<28s}", end='')
            for d in range(min(out_dim, 5)):
                width = r['ub'][d, 0] - r['lb'][d, 0]
                print(f"  [{r['lb'][d,0]:+.3f},{r['ub'][d,0]:+.3f}]", end='')
            print()

        # Soundness check
        print(f"\nSoundness (hex/oct bounds must contain star exact):")
        for r in successful:
            if r['label'] == 'Star exact (LP)':
                continue
            sound_lb = np.all(r['lb'] <= ref_lb + 1e-6)
            sound_ub = np.all(r['ub'] >= ref_ub - 1e-6)
            sound = sound_lb and sound_ub
            status = "PASS" if sound else "FAIL"
            print(f"  {r['label']:<28s}  {status}")
            if not sound:
                for d in range(out_dim):
                    if r['lb'][d, 0] > ref_lb[d, 0] + 1e-6:
                        print(f"    dim {d} LB: {r['lb'][d,0]:.6f} > {ref_lb[d,0]:.6f}")
                    if r['ub'][d, 0] < ref_ub[d, 0] - 1e-6:
                        print(f"    dim {d} UB: {r['ub'][d,0]:.6f} < {ref_ub[d,0]:.6f}")

    print()
    print("=" * 78)
    print("Analysis")
    print("=" * 78)
    print(f"Input dimension: 784 (MNIST 28x28)")
    print(f"Hexatope generators: 784")
    print(f"Bounding box cost per crossing neuron: 784*783 = 614,112 optimizations")
    print()
    print("Star sets can add half-space constraints for free (just append a row to C,d).")
    print("Hexatopes must compute the DCS bounding box (Algorithm 5.1) at each split,")
    print("solving optimizations via LP one at a time.")


if __name__ == "__main__":
    main()
