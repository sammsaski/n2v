#!/usr/bin/env python3
"""
n2v Benchmark Runner

Runs performance benchmarks for n2v verification methods and reports timing results.

Usage:
    python run_benchmarks.py                    # Run all benchmarks
    python run_benchmarks.py --model fc_mnist   # Run specific model
    python run_benchmarks.py --category cnn     # Run category
    python run_benchmarks.py --save-baseline    # Save as baseline
    python run_benchmarks.py --compare baseline.json  # Compare to baseline
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
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE")
    print("=" * 70)
    print(f"{'Benchmark':<35} {'Current':>10} {'Baseline':>10} {'Change':>12}")
    print("-" * 70)

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

        if change < -5:  # More than 5% faster
            change_str = f"\033[32m{change:+.1f}%\033[0m"  # Green
            improvements.append((name, change))
        elif change > 5:  # More than 5% slower
            change_str = f"\033[31m{change:+.1f}%\033[0m"  # Red
            regressions.append((name, change))
        else:
            change_str = f"{change:+.1f}%"

        print(f"{name:<35} {curr_time:>10.3f}s {base_time:>10.3f}s {change_str:>12}")

    print("-" * 70)
    print(f"Improvements (>5% faster): {len(improvements)}")
    print(f"Regressions (>5% slower):  {len(regressions)}")

    if regressions:
        print("\nRegressions:")
        for name, change in sorted(regressions, key=lambda x: -x[1]):
            print(f"  {name}: {change:+.1f}%")


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
    parser = argparse.ArgumentParser(description='Run n2v benchmarks')
    parser.add_argument('--model', type=str, help='Run benchmarks for specific model')
    parser.add_argument('--method', type=str, help='Run benchmarks for specific method')
    parser.add_argument('--category', type=str, help='Run benchmarks for specific category (fc, cnn, toy)')
    parser.add_argument('--include-slow', action='store_true', help='Include slow benchmarks (CNN exact)')
    parser.add_argument('--no-warmup', action='store_true', help='Skip warmup runs')
    parser.add_argument('--compare', type=str, help='Compare to baseline file (e.g., latest.json)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

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

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)

    # Save as latest (this becomes the new baseline for future comparisons)
    latest_file = RESULTS_DIR / 'latest.json'
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {latest_file}")

    # Compare to baseline if requested
    if args.compare:
        baseline_path = Path(args.compare)
        if not baseline_path.exists():
            baseline_path = RESULTS_DIR / args.compare
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline = json.load(f)
            compare_results(results, baseline)
        else:
            print(f"Baseline file not found: {args.compare}")

    return 0 if results['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
