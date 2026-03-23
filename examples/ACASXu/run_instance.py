#!/usr/bin/env python3
"""
Verify a single ACAS Xu instance following NNV's VNN-COMP strategy.

Strategy:
1. Falsification (random sampling, optionally with PGD)
2. Two-stage verification for prop_3/4: approx first, then exact if needed
3. Single-stage verification for other properties: exact directly

Output format (for parsing by bash script):
    RESULT:<SAT|UNSAT|UNKNOWN|ERROR>
    TIME:<seconds>
    METHOD:<falsification|approx|exact|approx+exact>
    CEX:<counterexample in VNN-COMP format> (only if SAT)
"""

import os
import sys
import time
import argparse

import numpy as np

import n2v
from n2v.nn import NeuralNetwork
from n2v.sets import Star
from n2v.utils import load_vnnlib, falsify
from n2v.utils.verify_specification import verify_specification
from n2v.utils.model_loader import load_onnx


def format_counterexample(input_vec: np.ndarray, output_vec: np.ndarray) -> str:
    """Format counterexample in VNN-COMP format.

    Format:
        ((X_0  value)
        (X_1  value)
        ...
        (Y_0  value)
        (Y_1  value)
        ...)
    """
    lines = []
    for i, val in enumerate(input_vec):
        lines.append(f"(X_{i}  {val})")
    for i, val in enumerate(output_vec):
        lines.append(f"(Y_{i}  {val})")
    return "(" + "\n".join(lines) + ")"


def main():
    parser = argparse.ArgumentParser(description='Verify single ACAS Xu instance')
    parser.add_argument('onnx', help='Path to ONNX file')
    parser.add_argument('vnnlib', help='Path to VNN-LIB file')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--falsify-method', type=str, default='random',
                        choices=['random', 'pgd', 'random+pgd'],
                        help='Falsification method (default: random)')
    parser.add_argument('--falsify-samples', type=int, default=500,
                        help='Number of random falsification samples (default: 500)')
    parser.add_argument('--pgd-restarts', type=int, default=10,
                        help='Number of PGD restarts (default: 10)')
    parser.add_argument('--pgd-steps', type=int, default=50,
                        help='Number of PGD steps per restart (default: 50)')
    args = parser.parse_args()

    onnx_path = args.onnx
    vnnlib_path = args.vnnlib
    n_workers = args.workers
    falsify_method = args.falsify_method
    n_falsify_samples = args.falsify_samples
    pgd_restarts = args.pgd_restarts
    pgd_steps = args.pgd_steps

    vnnlib_name = os.path.basename(vnnlib_path)
    t_start = time.time()

    try:
        # Load model and property
        model = load_onnx(onnx_path)
        prop = load_vnnlib(vnnlib_path)
        lb = prop['lb']
        ub = prop['ub']
        property_spec = prop['prop']

        # Handle multiple input regions (e.g., prop_6)
        # Convert single region to list for uniform handling
        if not isinstance(lb, list):
            lb_list = [lb]
            ub_list = [ub]
        else:
            lb_list = lb
            ub_list = ub

        # Determine property number from filename
        prop_num = int(vnnlib_name.split('_')[1].split('.')[0])
        use_two_stage = prop_num in [3, 4]

        # Step 1: Falsification (try each input region)
        for lb_region, ub_region in zip(lb_list, ub_list):
            falsify_result, cex = falsify(
                model, lb_region, ub_region, property_spec,
                method=falsify_method,
                n_samples=n_falsify_samples,
                n_restarts=pgd_restarts,
                n_steps=pgd_steps,
                seed=42
            )

            if falsify_result == 0:
                # Found counterexample
                total_time = time.time() - t_start
                print(f"RESULT:SAT")
                print(f"TIME:{total_time:.3f}")
                print(f"METHOD:falsification")
                if cex is not None:
                    cex_str = format_counterexample(cex[0], cex[1])
                    print(f"CEX:{cex_str}")
                return 0

        # Step 2: Reachability analysis
        # Configure parallelism
        if n_workers:
            n2v.set_parallel(True, n_workers=n_workers)
        else:
            import multiprocessing
            n2v.set_parallel(True, n_workers=multiprocessing.cpu_count())
        n2v.set_lp_solver('linprog')

        # Create neural network wrapper
        net = NeuralNetwork(model)

        # Verify each input region
        # Property holds (UNSAT) only if ALL regions are safe
        # Property violated (SAT) if ANY region has a counterexample
        all_unsat = True
        any_sat = False
        method_used = "exact"

        for lb_region, ub_region in zip(lb_list, ub_list):
            # Create input Star set for this region
            lb_col = lb_region.reshape(-1, 1).astype(np.float32)
            ub_col = ub_region.reshape(-1, 1).astype(np.float32)
            input_set = Star.from_bounds(lb_col, ub_col)

            region_method = "exact"

            # For prop_3/4: Try approx first
            if use_two_stage:
                try:
                    reach_sets = net.reach(input_set, method='approx')
                    verify_result = verify_specification(reach_sets, property_spec)

                    if verify_result == 1:
                        # UNSAT via approx for this region
                        continue
                    elif verify_result == 0:
                        # SAT via approx
                        any_sat = True
                        total_time = time.time() - t_start
                        print(f"RESULT:SAT")
                        print(f"TIME:{total_time:.3f}")
                        print(f"METHOD:approx")
                        return 0
                    # else: UNKNOWN, continue to exact
                    region_method = "approx+exact"
                except Exception:
                    pass  # Continue to exact

            # Exact reachability
            reach_sets = net.reach(input_set, method='exact')
            verify_result = verify_specification(reach_sets, property_spec)

            if region_method == "approx+exact":
                method_used = "approx+exact"

            if verify_result == 0:
                any_sat = True
                total_time = time.time() - t_start
                print(f"RESULT:SAT")
                print(f"TIME:{total_time:.3f}")
                print(f"METHOD:{method_used}")
                return 0
            elif verify_result != 1:
                all_unsat = False

        # All regions verified
        if all_unsat:
            result_str = "UNSAT"
        elif any_sat:
            result_str = "SAT"
        else:
            result_str = "UNKNOWN"

        total_time = time.time() - t_start
        print(f"RESULT:{result_str}")
        print(f"TIME:{total_time:.3f}")
        print(f"METHOD:{method_used}")

    except Exception as e:
        total_time = time.time() - t_start
        print(f"RESULT:ERROR")
        print(f"TIME:{total_time:.3f}")
        print(f"METHOD:none")
        print(f"ERROR:{str(e)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
