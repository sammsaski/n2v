#!/usr/bin/env python3
"""
Generic VNN-COMP instance verifier.

Verifies a single ONNX model against a VNNLIB specification using a
3-stage strategy: falsification -> approx reachability -> exact reachability.

Usage:
    python run_instance.py <onnx_model> <vnnlib_spec> [options]

Output:
    Prints one of: sat, unsat, unknown, timeout, error (VNN-COMP compliant)
    If sat, prints counterexample on next line.
"""

import os
import sys
import time
import argparse
import multiprocessing

import numpy as np

import n2v
from n2v.nn import NeuralNetwork
from n2v.utils import load_vnnlib, verify_specification, falsify
from n2v.utils.model_loader import load_onnx

# Import prepare_instance from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_instance import get_input_shape, create_input_set

# VNN-COMP result strings (lowercase)
RESULT_SAT = "sat"
RESULT_UNSAT = "unsat"
RESULT_UNKNOWN = "unknown"
RESULT_TIMEOUT = "timeout"
RESULT_ERROR = "error"


def format_counterexample(input_vec: np.ndarray, output_vec: np.ndarray) -> str:
    """
    Format counterexample in VNN-COMP format.

    Format:
        ((X_0  value)
        (X_1  value)
        ...
        (Y_0  value)
        (Y_1  value)
        ...)
    """
    input_vec = np.asarray(input_vec).flatten()
    output_vec = np.asarray(output_vec).flatten()
    lines = []
    for i, val in enumerate(input_vec):
        lines.append(f"(X_{i}  {val})")
    for i, val in enumerate(output_vec):
        lines.append(f"(Y_{i}  {val})")
    return "(" + "\n".join(lines) + ")"


def verify_instance(
    onnx_path: str,
    vnnlib_path: str,
    no_falsify: bool = False,
    no_approx: bool = False,
    no_exact: bool = False,
    falsify_method: str = 'random+pgd',
    falsify_samples: int = 500,
    pgd_restarts: int = 10,
    pgd_steps: int = 50,
    workers: int = None,
    parallel_regions: bool = False,
    precompute_bounds: bool = False,
) -> dict:
    """
    Verify a single instance using 3-stage strategy.

    Args:
        onnx_path: Path to ONNX model
        vnnlib_path: Path to VNNLIB specification
        no_falsify: Skip falsification stage
        no_approx: Skip approximate reachability stage
        no_exact: Skip exact reachability stage
        falsify_method: Falsification method ('random', 'pgd', 'random+pgd')
        falsify_samples: Number of random samples for falsification
        pgd_restarts: Number of PGD restarts
        pgd_steps: Number of PGD steps per restart
        workers: Number of parallel workers (None = CPU count)
        parallel_regions: Verify disjunctive input regions in parallel
        precompute_bounds: Enable Zono pre-pass for dead neuron elimination

    Returns:
        Dictionary with keys:
        - 'result': one of 'sat', 'unsat', 'unknown', 'error'
        - 'time': wall-clock time in seconds
        - 'method': which stage produced the result
        - 'counterexample': formatted string (only if sat)
    """
    t_start = time.time()

    try:
        # Load model and property
        model = load_onnx(onnx_path)
        prop = load_vnnlib(vnnlib_path)
        input_shape = get_input_shape(onnx_path)

        lb_raw = prop['lb']
        ub_raw = prop['ub']
        property_spec = prop['prop']

        # Normalize to list of regions
        if not isinstance(lb_raw, list):
            lb_list = [lb_raw]
            ub_list = [ub_raw]
        else:
            lb_list = lb_raw
            ub_list = ub_raw

        # Stage 1: Falsification
        if not no_falsify:
            for lb_region, ub_region in zip(lb_list, ub_list):
                try:
                    falsify_result, cex = falsify(
                        model, lb_region, ub_region, property_spec,
                        method=falsify_method,
                        n_samples=falsify_samples,
                        n_restarts=pgd_restarts,
                        n_steps=pgd_steps,
                        seed=42,
                    )
                    if falsify_result == 0 and cex is not None:
                        return {
                            'result': RESULT_SAT,
                            'time': time.time() - t_start,
                            'method': 'falsification',
                            'counterexample': format_counterexample(cex[0], cex[1]),
                        }
                except Exception:
                    # Falsification may fail for CNN models (flat input vs 4D expected)
                    # Silently skip to next stage
                    pass

        # Configure parallel LP solving
        if workers is None:
            workers = multiprocessing.cpu_count()
        n2v.set_parallel(True, n_workers=workers)
        n2v.set_lp_solver('linprog')

        net = NeuralNetwork(model)

        # Track which regions still need exact verification
        unknown_regions = []

        # Stage 2: Approximate reachability
        if not no_approx:
            # Parallel path for multiple disjunctive regions
            if parallel_regions and len(lb_list) > 1:
                from n2v.utils.vnncomp import verify_regions_parallel
                parallel_result = verify_regions_parallel(
                    model, list(zip(lb_list, ub_list)), property_spec,
                    method='approx', n_workers=workers,
                    precompute_bounds=precompute_bounds,
                )
                if parallel_result['result'] == 'sat':
                    return {
                        'result': RESULT_SAT,
                        'time': time.time() - t_start,
                        'method': 'approx',
                        'counterexample': None,
                    }
                elif parallel_result['result'] == 'unsat':
                    return {
                        'result': RESULT_UNSAT,
                        'time': time.time() - t_start,
                        'method': 'approx',
                        'counterexample': None,
                    }
                else:
                    # Some regions unknown — fall through to exact
                    unknown_regions = [
                        (lb_list[i], ub_list[i])
                        for i, r in enumerate(parallel_result['per_region'])
                        if r['result'] != 1
                    ]
            else:
                # Sequential path
                all_unsat = True
                for lb_region, ub_region in zip(lb_list, ub_list):
                    input_set = create_input_set(lb_region, ub_region, input_shape)
                    try:
                        reach_sets = net.reach(
                            input_set, method='approx',
                            precompute_bounds=precompute_bounds,
                        )
                        verdict = verify_specification(reach_sets, property_spec)

                        if verdict == 0:
                            # SAT — property violated
                            return {
                                'result': RESULT_SAT,
                                'time': time.time() - t_start,
                                'method': 'approx',
                                'counterexample': None,
                            }
                        elif verdict == 1:
                            # UNSAT for this region
                            continue
                        else:
                            # UNKNOWN — need exact for this region
                            all_unsat = False
                            unknown_regions.append((lb_region, ub_region))
                    except Exception:
                        all_unsat = False
                        unknown_regions.append((lb_region, ub_region))

                if all_unsat:
                    return {
                        'result': RESULT_UNSAT,
                        'time': time.time() - t_start,
                        'method': 'approx',
                        'counterexample': None,
                    }
        else:
            # If skipping approx, all regions need exact
            unknown_regions = list(zip(lb_list, ub_list))

        # Stage 3: Exact reachability
        if not no_exact and unknown_regions:
            # Parallel path for exact
            if parallel_regions and len(unknown_regions) > 1:
                from n2v.utils.vnncomp import verify_regions_parallel
                parallel_result = verify_regions_parallel(
                    model, unknown_regions, property_spec,
                    method='exact', n_workers=workers,
                    precompute_bounds=precompute_bounds,
                )
                if parallel_result['result'] == 'sat':
                    return {
                        'result': RESULT_SAT,
                        'time': time.time() - t_start,
                        'method': 'exact',
                        'counterexample': None,
                    }
                elif parallel_result['result'] == 'unsat':
                    return {
                        'result': RESULT_UNSAT,
                        'time': time.time() - t_start,
                        'method': 'exact',
                        'counterexample': None,
                    }
            else:
                # Sequential path
                all_unsat = True
                for lb_region, ub_region in unknown_regions:
                    input_set = create_input_set(lb_region, ub_region, input_shape)
                    try:
                        reach_sets = net.reach(
                            input_set, method='exact',
                            precompute_bounds=precompute_bounds,
                        )
                        verdict = verify_specification(reach_sets, property_spec)

                        if verdict == 0:
                            return {
                                'result': RESULT_SAT,
                                'time': time.time() - t_start,
                                'method': 'exact',
                                'counterexample': None,
                            }
                        elif verdict != 1:
                            all_unsat = False
                    except Exception:
                        all_unsat = False

                if all_unsat:
                    return {
                        'result': RESULT_UNSAT,
                        'time': time.time() - t_start,
                        'method': 'exact',
                        'counterexample': None,
                    }

        return {
            'result': RESULT_UNKNOWN,
            'time': time.time() - t_start,
            'method': 'none',
            'counterexample': None,
        }

    except Exception as e:
        return {
            'result': RESULT_ERROR,
            'time': time.time() - t_start,
            'method': 'none',
            'counterexample': None,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='VNN-COMP instance verifier')
    parser.add_argument('onnx', help='Path to ONNX model file')
    parser.add_argument('vnnlib', help='Path to VNNLIB specification file')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Timeout in seconds (handled by shell, informational only)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--no-falsify', action='store_true',
                        help='Skip falsification stage')
    parser.add_argument('--no-approx', action='store_true',
                        help='Skip approximate reachability stage')
    parser.add_argument('--no-exact', action='store_true',
                        help='Skip exact reachability stage')
    parser.add_argument('--falsify-method', type=str, default='random+pgd',
                        choices=['random', 'pgd', 'random+pgd'],
                        help='Falsification method (default: random+pgd)')
    parser.add_argument('--falsify-samples', type=int, default=500,
                        help='Number of random falsification samples')
    parser.add_argument('--pgd-restarts', type=int, default=10,
                        help='Number of PGD restarts')
    parser.add_argument('--pgd-steps', type=int, default=50,
                        help='Number of PGD steps per restart')
    parser.add_argument('--parallel-regions', action='store_true',
                        help='Verify input regions in parallel')
    parser.add_argument('--category', type=str, default=None,
                        help='Benchmark category for per-benchmark config')
    parser.add_argument('--precompute-bounds', action='store_true',
                        help='Enable Zono pre-pass for dead neuron elimination')
    args = parser.parse_args()

    result = verify_instance(
        onnx_path=args.onnx,
        vnnlib_path=args.vnnlib,
        no_falsify=args.no_falsify,
        no_approx=args.no_approx,
        no_exact=args.no_exact,
        falsify_method=args.falsify_method,
        falsify_samples=args.falsify_samples,
        pgd_restarts=args.pgd_restarts,
        pgd_steps=args.pgd_steps,
        workers=args.workers,
        parallel_regions=args.parallel_regions,
        precompute_bounds=args.precompute_bounds,
    )

    # Print VNN-COMP compliant output
    print(result['result'])
    if result['result'] == RESULT_SAT and result.get('counterexample'):
        print(result['counterexample'])

    return 0


if __name__ == "__main__":
    sys.exit(main())
