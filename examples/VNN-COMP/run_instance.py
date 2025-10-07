#!/usr/bin/env python3
"""
Generic VNN-COMP instance verifier.

Verifies a single ONNX model against a VNNLIB specification.
Strategy: falsification -> reachability methods (from benchmark_configs.py).

Usage:
    python run_instance.py <onnx_model> <vnnlib_spec> [options]

Output:
    Prints one of: sat, unsat, unknown, timeout, error (VNN-COMP compliant)
    If sat, prints counterexample on next line.
"""

import logging
import os
import sys
import time
import argparse
import multiprocessing

import numpy as np

logger = logging.getLogger(__name__)

import n2v
from n2v.nn import NeuralNetwork
from n2v.utils import load_vnnlib, falsify
from n2v.utils.verify_specification import verify_specification
from n2v.utils.model_loader import load_onnx

# Import prepare_instance from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_instance import get_input_shape, create_input_set
from benchmark_configs import get_config

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
    category: str = None,
    workers: int = None,
    no_falsify: bool = False,
) -> dict:
    """
    Verify a single instance.

    Strategy: falsification first, then iterate through reach_methods from
    benchmark_configs.py until one produces a definitive result.

    Args:
        onnx_path: Path to ONNX model
        vnnlib_path: Path to VNNLIB specification
        category: Benchmark category for per-benchmark config (uses default if None)
        workers: Number of parallel LP workers (None = CPU count)
        no_falsify: If True, skip the falsification stage

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

        # Get benchmark config (falls back to DEFAULT_CONFIG if category is None/unknown)
        cfg = get_config(category, onnx_path, vnnlib_path)
        falsify_samples = cfg.get('n_rand', 100)
        falsify_method = cfg.get('falsify_method', 'random+pgd')

        # Configure parallel LP solving
        if workers is None:
            workers = multiprocessing.cpu_count()
        n2v.set_parallel(True, n_workers=workers)
        n2v.set_lp_solver('linprog')

        # Stage 1: Falsification
        if no_falsify:
            logger.info("Falsification skipped (--no-falsify)")
        for lb_region, ub_region in zip(lb_list, ub_list):
            if no_falsify:
                break
            try:
                lb_shaped = np.asarray(lb_region, dtype=np.float64).reshape(input_shape)
                ub_shaped = np.asarray(ub_region, dtype=np.float64).reshape(input_shape)
                falsify_result, cex = falsify(
                    model, lb_shaped, ub_shaped, property_spec,
                    method=falsify_method,
                    n_samples=falsify_samples,
                    seed=42,
                )
                if falsify_result == 0 and cex is not None:
                    return {
                        'result': RESULT_SAT,
                        'time': time.time() - t_start,
                        'method': 'falsification',
                        'counterexample': format_counterexample(cex[0], cex[1]),
                    }
            except Exception as e:
                logger.debug("Falsification failed for region: %s", e)

        # Stage 2: Reachability (iterate through configured methods)
        net = NeuralNetwork(model)

        for reach_method, reach_kwargs in cfg['reach_methods']:
            all_unsat = True
            for lb_region, ub_region in zip(lb_list, ub_list):
                input_set = create_input_set(lb_region, ub_region, input_shape)
                try:
                    extra_kwargs = dict(reach_kwargs)
                    if reach_method == 'probabilistic':
                        extra_kwargs['input_shape'] = input_shape
                    else:
                        extra_kwargs['precompute_bounds'] = 'ibp'

                    reach_sets = net.reach(
                        input_set, method=reach_method,
                        **extra_kwargs,
                    )
                    verdict = verify_specification(reach_sets, property_spec)

                    if verdict == 0:
                        return {
                            'result': RESULT_SAT,
                            'time': time.time() - t_start,
                            'method': reach_method,
                            'counterexample': None,
                        }
                    elif verdict == 1:
                        continue
                    else:
                        all_unsat = False
                except NotImplementedError as e:
                    logger.warning("Unsupported layer in %s: %s — trying next method", reach_method, e)
                    all_unsat = False
                    break  # Skip remaining regions, try next method
                except Exception as e:
                    logger.warning("Error during %s verification: %s", reach_method, e)
                    all_unsat = False

            if all_unsat:
                return {
                    'result': RESULT_UNSAT,
                    'time': time.time() - t_start,
                    'method': reach_method,
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    parser = argparse.ArgumentParser(description='VNN-COMP instance verifier')
    parser.add_argument('onnx', help='Path to ONNX model file')
    parser.add_argument('vnnlib', help='Path to VNNLIB specification file')
    parser.add_argument('--category', type=str, default=None,
                        help='Benchmark category for per-benchmark config')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel LP workers (default: CPU count)')
    args = parser.parse_args()

    result = verify_instance(
        onnx_path=args.onnx,
        vnnlib_path=args.vnnlib,
        category=args.category,
        workers=args.workers,
    )

    # Print VNN-COMP compliant output
    print(result['result'])
    if result['result'] == RESULT_SAT and result.get('counterexample'):
        print(result['counterexample'])
    if result['result'] == RESULT_ERROR and result.get('error'):
        print(f"# Error: {result['error']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
