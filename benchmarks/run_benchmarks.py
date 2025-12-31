#!/usr/bin/env python3
"""
n2v Benchmark Runner

Runs performance benchmarks for n2v verification methods and reports timing results.

Usage:
    python run_benchmarks.py                    # Run all, compare to latest.json
    python run_benchmarks.py --category cnn     # Run CNN benchmarks only
    python run_benchmarks.py --save             # Run all and update latest.json

Workflow:
    1. Make code changes
    2. Run: python run_benchmarks.py            # Compare against baseline
    3. If improvements are acceptable:
       Run: python run_benchmarks.py --save     # Update baseline
    4. Commit the updated latest.json with your changes
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from scipy.io import loadmat

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from local benchmarks config (not CompareNNV config)
from benchmarks.config import (
    ALL_BENCHMARKS,
    BenchmarkConfig,
    METHOD_CONFIG,
    get_benchmarks_by_model,
    get_benchmarks_by_method,
    get_benchmarks_by_category,
)

# Import n2v components
from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.nn import NeuralNetwork
from n2v.utils.model_loader import load_onnx


# Paths
MODELS_DIR = Path(__file__).parent.parent / 'examples' / 'CompareNNV' / 'models'
SAMPLES_DIR = Path(__file__).parent.parent / 'examples' / 'CompareNNV' / 'samples'
RESULTS_DIR = Path(__file__).parent / 'results'


# NNV timing data from CompareNNV experiments (2024-12-31)
# These are static since NNV doesn't change - avoids re-running MATLAB experiments
NNV_TIMINGS = {
    # FC MNIST
    'fc_mnist_exact': 1.2946,
    'fc_mnist_approx': 0.2172,
    'fc_mnist_area_0.25': 0.1553,
    'fc_mnist_area_0.50': 0.1241,
    'fc_mnist_area_0.75': 0.1132,
    'fc_mnist_range_0.25': 0.1380,
    'fc_mnist_range_0.50': 0.0786,
    'fc_mnist_range_0.75': 0.0674,
    'fc_mnist_small_exact': 0.0586,
    'fc_mnist_small_approx': 0.0530,
    # CNN conv+relu
    'cnn_conv_relu_approx': 4.2755,
    'cnn_conv_relu_area_0.25': 3.2630,
    'cnn_conv_relu_area_0.50': 2.1223,
    'cnn_conv_relu_area_0.75': 1.0712,
    'cnn_conv_relu_range_0.25': 3.1747,
    'cnn_conv_relu_range_0.50': 2.1289,
    'cnn_conv_relu_range_0.75': 1.0695,
    # CNN avgpool
    'cnn_avgpool_approx': 0.1386,
    'cnn_avgpool_area_0.25': 0.1053,
    'cnn_avgpool_area_0.50': 0.1054,
    'cnn_avgpool_area_0.75': 0.0824,
    'cnn_avgpool_range_0.25': 0.1116,
    'cnn_avgpool_range_0.50': 0.1027,
    'cnn_avgpool_range_0.75': 0.0795,
    # CNN maxpool
    'cnn_maxpool_approx': 17.6356,
    'cnn_maxpool_area_0.25': 17.5725,
    'cnn_maxpool_area_0.50': 17.1401,
    'cnn_maxpool_area_0.75': 16.6098,
    'cnn_maxpool_range_0.25': 17.3391,
    'cnn_maxpool_range_0.50': 16.9986,
    'cnn_maxpool_range_0.75': 16.5728,
    # Toy models
    'toy_fc_4_3_2_zono': 0.0069,
    'toy_fc_4_3_2_box': 0.0574,
    'toy_fc_8_4_2_zono': 0.0014,
    'toy_fc_8_4_2_box': 0.0295,
}


def load_test_sample(sample_path: Path, model_type: str = 'mnist') -> Dict[str, Any]:
    """Load test sample from .mat file."""
    data = loadmat(str(sample_path))

    if model_type == 'mnist':
        image = data['image'].astype(np.float64)
        if image.max() > 1:
            image = image / 255.0
        label = int(data['label'].flatten()[0])
        return {'image': image, 'label': label}
    else:
        input_vec = data['input'].astype(np.float64).flatten()
        output_dim = int(data['output_dim'].flatten()[0])
        return {'input': input_vec, 'output_dim': output_dim}


def create_input_set(sample_data: dict, epsilon: float, set_type: str, model_type: str = 'mnist'):
    """Create input set based on set type."""
    if model_type == 'mnist':
        image = sample_data['image']
        lb = np.maximum(image - epsilon, 0)
        ub = np.minimum(image + epsilon, 1)
        lb_flat = lb.flatten().reshape(-1, 1)
        ub_flat = ub.flatten().reshape(-1, 1)

        if set_type == 'star':
            return Star.from_bounds(lb_flat, ub_flat)
        elif set_type == 'imagestar':
            height, width, num_channels = 28, 28, 1
            return ImageStar.from_bounds(lb_flat, ub_flat, height, width, num_channels)
        elif set_type == 'zono':
            return Zono.from_bounds(lb_flat, ub_flat)
        elif set_type == 'box':
            return Box(lb_flat, ub_flat)
        else:
            raise ValueError(f"Unknown set type: {set_type}")
    else:
        input_vec = sample_data['input']
        lb = input_vec - epsilon
        ub = input_vec + epsilon
        lb_flat = lb.reshape(-1, 1)
        ub_flat = ub.reshape(-1, 1)

        if set_type in ['star', 'imagestar']:
            return Star.from_bounds(lb_flat, ub_flat)
        elif set_type == 'zono':
            return Zono.from_bounds(lb_flat, ub_flat)
        elif set_type == 'box':
            return Box(lb_flat, ub_flat)
        else:
            raise ValueError(f"Unknown set type: {set_type}")


def run_single_benchmark(bench: BenchmarkConfig, warmup: bool = True) -> Dict[str, Any]:
    """Run a single benchmark and return results."""
    model_type = 'toy' if bench.model.startswith('toy') else 'mnist'

    # Load model
    model_dir = MODELS_DIR / bench.model
    onnx_path = model_dir / f"{bench.model}.onnx"
    if not onnx_path.exists():
        return {'success': False, 'error': f"Model not found: {onnx_path}"}

    net = load_onnx(str(onnx_path))

    # Load sample
    sample_path = SAMPLES_DIR / f"{bench.model}_sample.mat"
    if not sample_path.exists():
        return {'success': False, 'error': f"Sample not found: {sample_path}"}

    sample_data = load_test_sample(sample_path, model_type)

    # Create input set
    input_set = create_input_set(sample_data, bench.epsilon, bench.set_type, model_type)

    # Configure method
    method_config = METHOD_CONFIG[bench.method]
    reach_method = method_config['method']
    reach_kwargs = method_config['kwargs'].copy()
    if bench.relax_factor is not None:
        reach_kwargs['relax_factor'] = bench.relax_factor

    # Create network wrapper
    net_wrapper = NeuralNetwork(net)

    # Warmup run (optional)
    if warmup:
        try:
            _ = net_wrapper.reach(input_set, method=reach_method, **reach_kwargs)
        except Exception:
            pass  # Ignore warmup errors

    # Timed run
    try:
        t_start = time.perf_counter()
        output_sets = net_wrapper.reach(input_set, method=reach_method, **reach_kwargs)
        elapsed = time.perf_counter() - t_start

        return {
            'success': True,
            'time': elapsed,
            'num_output_sets': len(output_sets),
            'error': None,
        }
    except Exception as e:
        return {
            'success': False,
            'time': None,
            'num_output_sets': 0,
            'error': str(e),
        }


def run_benchmarks(
    benchmarks: List[BenchmarkConfig],
    skip_slow: bool = True,
    warmup: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run multiple benchmarks and return aggregated results."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_benchmarks': len(benchmarks),
        'skipped': 0,
        'succeeded': 0,
        'failed': 0,
        'total_time': 0.0,
        'benchmarks': {},
    }

    for i, bench in enumerate(benchmarks):
        if skip_slow and bench.skip_by_default:
            if verbose:
                print(f"[{i+1}/{len(benchmarks)}] Skipping: {bench.name} (slow)")
            results['skipped'] += 1
            continue

        if verbose:
            print(f"[{i+1}/{len(benchmarks)}] Running: {bench.name}...", end=' ', flush=True)

        result = run_single_benchmark(bench, warmup=warmup)

        if result['success']:
            results['succeeded'] += 1
            results['total_time'] += result['time']
            if verbose:
                print(f"{result['time']:.3f}s")
        else:
            results['failed'] += 1
            if verbose:
                print(f"FAILED: {result['error']}")

        results['benchmarks'][bench.name] = {
            **bench.to_dict(),
            **result,
        }

    return results


def compare_results(current: Dict, baseline: Dict) -> None:
    """Compare current results to baseline and print comparison."""
    print("\n" + "=" * 85)
    print("COMPARISON TO BASELINE")
    print("=" * 85)
    print(f"{'Benchmark':<35} {'Current':>10} {'Baseline':>10} {'Change':>10} {'Speedup':>10}")
    print("-" * 85)

    improvements = []
    regressions = []

    for name, curr_data in current['benchmarks'].items():
        if name not in baseline['benchmarks']:
            continue

        base_data = baseline['benchmarks'][name]

        if not curr_data['success'] or not base_data['success']:
            continue

        curr_time = curr_data['time']
        base_time = base_data['time']
        change = (curr_time - base_time) / base_time * 100
        speedup = base_time / curr_time if curr_time > 0 else float('inf')

        # Format change percentage (pad to fixed width before adding color)
        change_text = f"{change:+.1f}%"
        speedup_text = f"{speedup:.2f}x"

        if change < -5:  # More than 5% faster
            change_str = f"\033[32m{change_text:>10}\033[0m"  # Green
            speedup_str = f"\033[32m{speedup_text:>10}\033[0m"
            improvements.append((name, change, speedup))
        elif change > 5:  # More than 5% slower
            change_str = f"\033[31m{change_text:>10}\033[0m"  # Red
            speedup_str = f"\033[31m{speedup_text:>10}\033[0m"
            regressions.append((name, change, speedup))
        else:
            change_str = f"{change_text:>10}"
            speedup_str = f"{speedup_text:>10}"

        print(f"{name:<35} {curr_time:>10.3f}s {base_time:>10.3f}s {change_str} {speedup_str}")

    print("-" * 85)
    print(f"Improvements (>5% faster): {len(improvements)}")
    print(f"Regressions (>5% slower):  {len(regressions)}")

    if regressions:
        print("\nRegressions:")
        for name, change, speedup in sorted(regressions, key=lambda x: -x[1]):
            print(f"  {name}: {change:+.1f}% ({speedup:.2f}x)")


def compare_to_nnv(results: Dict) -> None:
    """Compare current results to NNV timings and print comparison."""
    print("\n" + "=" * 85)
    print("COMPARISON TO NNV (MATLAB)")
    print("=" * 85)
    print(f"{'Benchmark':<35} {'n2v':>10} {'NNV':>10} {'Speedup':>10}")
    print("-" * 85)

    faster = []
    slower = []

    for name, data in results['benchmarks'].items():
        if not data['success']:
            continue

        # Look up NNV timing
        nnv_time = NNV_TIMINGS.get(name)
        if nnv_time is None:
            continue

        n2v_time = data['time']
        speedup = nnv_time / n2v_time if n2v_time > 0 else float('inf')

        # Format speedup (pad to fixed width before adding color)
        speedup_text = f"{speedup:.1f}x"

        if speedup > 1.5:  # n2v is faster
            speedup_str = f"\033[32m{speedup_text:>10}\033[0m"  # Green
            faster.append((name, speedup))
        elif speedup < 0.67:  # n2v is slower (NNV >1.5x faster)
            speedup_str = f"\033[31m{speedup_text:>10}\033[0m"  # Red
            slower.append((name, speedup))
        else:
            speedup_str = f"{speedup_text:>10}"

        print(f"{name:<35} {n2v_time:>10.3f}s {nnv_time:>10.3f}s {speedup_str}")

    print("-" * 85)
    print(f"n2v faster (>1.5x): {len(faster)}")
    print(f"n2v slower (<0.67x): {len(slower)}")


def print_summary(results: Dict) -> None:
    """Print benchmark summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total benchmarks: {results['total_benchmarks']}")
    print(f"Succeeded:        {results['succeeded']}")
    print(f"Failed:           {results['failed']}")
    print(f"Skipped:          {results['skipped']}")
    print(f"Total time:       {results['total_time']:.2f}s")
    print()

    # Print detailed results by category
    categories = {}
    for name, data in results['benchmarks'].items():
        cat = data.get('category', 'misc')
        if cat not in categories:
            categories[cat] = {'total': 0, 'time': 0.0}
        if data['success']:
            categories[cat]['total'] += 1
            categories[cat]['time'] += data['time']

    print(f"{'Category':<10} {'Count':>8} {'Total Time':>12} {'Avg Time':>12}")
    print("-" * 45)
    for cat, stats in sorted(categories.items()):
        avg = stats['time'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{cat:<10} {stats['total']:>8} {stats['time']:>12.2f}s {avg:>12.3f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Run n2v benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run_benchmarks.py                    # Run all, compare to latest.json
  python run_benchmarks.py --category cnn     # Run CNN benchmarks only
  python run_benchmarks.py --save             # Run all and update latest.json
  python run_benchmarks.py --lp-solver linprog  # Use scipy linprog (HiGHS)
'''
    )
    parser.add_argument('--model', type=str, help='Run benchmarks for specific model')
    parser.add_argument('--method', type=str, help='Run benchmarks for specific method')
    parser.add_argument('--category', type=str, help='Run benchmarks for specific category (fc, cnn, toy)')
    parser.add_argument('--include-slow', action='store_true', help='Include slow benchmarks (CNN exact)')
    parser.add_argument('--no-warmup', action='store_true', help='Skip warmup runs')
    parser.add_argument('--save', action='store_true', help='Save results to latest.json (updates baseline)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--lp-solver', type=str, default='default',
                        help='LP solver: default (CVXPY/CLARABEL), linprog (scipy/HiGHS)')

    args = parser.parse_args()

    # Set LP solver globally
    if args.lp_solver != 'default':
        import n2v
        n2v.set_lp_solver(args.lp_solver)
        print(f"Using LP solver: {args.lp_solver}")

    # Select benchmarks
    if args.model:
        benchmarks = get_benchmarks_by_model(args.model)
    elif args.method:
        benchmarks = get_benchmarks_by_method(args.method)
    elif args.category:
        benchmarks = get_benchmarks_by_category(args.category)
    else:
        benchmarks = ALL_BENCHMARKS

    if not benchmarks:
        print("No benchmarks found matching criteria")
        return 1

    print(f"Running {len(benchmarks)} benchmarks...")

    # Run benchmarks
    results = run_benchmarks(
        benchmarks,
        skip_slow=not args.include_slow,
        warmup=not args.no_warmup,
        verbose=not args.quiet,
    )

    # Print summary
    if not args.quiet:
        print_summary(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    latest_file = RESULTS_DIR / 'latest.json'

    # Always compare to latest.json if it exists
    if latest_file.exists():
        with open(latest_file) as f:
            baseline = json.load(f)
        compare_results(results, baseline)
    else:
        print("\nNo baseline found (latest.json). Use --save to create one.")

    # Always compare to NNV timings
    if not args.quiet:
        compare_to_nnv(results)

    # Only save if --save flag is passed
    if args.save:
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {latest_file}")
    else:
        print(f"\nResults NOT saved (use --save to update latest.json)")

    return 0 if results['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
