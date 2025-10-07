#!/usr/bin/env python3
"""
Run all Python verification experiments.

This script provides a convenient way to run all experiments defined
in the configuration.

Usage:
    python run_all.py              # Run all experiments
    python run_all.py --parallel   # Run experiments in parallel (experimental)
    python run_all.py --model fc_mnist  # Run specific model
    python run_all.py --skip-cnn-exact  # Skip exact method for CNN models (recommended)
"""

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALL_EXPERIMENTS, get_experiments_by_model
from python.verify import run_verification


def run_single_experiment(exp_id: int):
    """Run a single experiment (for parallel execution)."""
    from config import get_experiment
    base_dir = Path(__file__).parent.parent

    exp = get_experiment(exp_id)
    try:
        result = run_verification(exp, base_dir, verbose=False)
        return {
            'id': exp.id,
            'model': exp.model,
            'method': exp.method,
            'success': result['success'],
            'robust': result['robust'],
            'time': result['computation_time'],
        }
    except Exception as e:
        return {
            'id': exp.id,
            'model': exp.model,
            'method': exp.method,
            'success': 0,
            'robust': -1,
            'time': 0,
            'error': str(e),
        }


def run_all_sequential(experiments, base_dir: Path, verbose: bool = True):
    """Run all experiments sequentially."""
    results = []
    total = len(experiments)

    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{total}] Running: {exp.model} / {exp.method}", end='')
        if exp.relax_factor:
            print(f" (relax={exp.relax_factor})")
        else:
            print()

        try:
            result = run_verification(exp, base_dir, verbose=verbose)
            results.append({
                'id': exp.id,
                'model': exp.model,
                'method': exp.method,
                'success': result['success'],
                'robust': result['robust'],
                'time': result['computation_time'],
            })
            print(f"  Completed in {result['computation_time']:.2f}s")
        except Exception as e:
            results.append({
                'id': exp.id,
                'model': exp.model,
                'method': exp.method,
                'success': 0,
                'robust': -1,
                'time': 0,
                'error': str(e),
            })
            print(f"  ERROR: {e}")

    return results


def run_all_parallel(experiments, max_workers: int = 4):
    """Run experiments in parallel."""
    results = []
    exp_ids = [exp.id for exp in experiments]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_experiment, exp_id): exp_id
                   for exp_id in exp_ids}

        for future in as_completed(futures):
            exp_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "OK" if result['success'] else "FAIL"
                print(f"[{len(results)}/{len(exp_ids)}] Exp {exp_id}: {status}")
            except Exception as e:
                results.append({
                    'id': exp_id,
                    'success': 0,
                    'error': str(e),
                })
                print(f"[{len(results)}/{len(exp_ids)}] Exp {exp_id}: ERROR - {e}")

    return results


def print_summary(results, skipped_count: int = 0):
    """Print summary of all results."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    succeeded = sum(1 for r in results if r.get('success', 0))
    failed = len(results) - succeeded
    total_time = sum(r.get('time', 0) for r in results)

    print(f"\nTotal experiments: {len(results) + skipped_count}")
    print(f"Succeeded: {succeeded}/{len(results) + skipped_count}")
    print(f"Skipped:   {skipped_count}/{len(results) + skipped_count}")
    print(f"Failed:    {failed}/{len(results) + skipped_count}")
    print(f"Total time: {total_time:.2f} seconds")

    print(f"\n{'ID':<4} {'Model':<20} {'Method':<20} {'Status':<8} {'Robust':<8} {'Time':<10}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.get('id', 0)):
        status = "OK" if r.get('success', 0) else "FAIL"
        robust = "YES" if r.get('robust', -1) == 1 else "NO"
        time_str = f"{r.get('time', 0):.2f}s"
        print(f"{r.get('id', '?'):<4} {r.get('model', '?'):<20} {r.get('method', '?'):<20} {status:<8} {robust:<8} {time_str:<10}")


def should_skip_experiment(exp, skip_cnn_exact: bool) -> bool:
    """Check if an experiment should be skipped."""
    if skip_cnn_exact:
        is_cnn = exp.model.startswith('cnn')
        is_exact = exp.method == 'exact'
        if is_cnn and is_exact:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Run all Python verification experiments')
    parser.add_argument('--model', type=str, help='Run specific model only')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for parallel')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--skip-cnn-exact', action='store_true',
                        help='Skip exact method for CNN models (avoids memory issues)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Select experiments
    if args.model:
        experiments = get_experiments_by_model(args.model)
    else:
        experiments = ALL_EXPERIMENTS

    # Filter out skipped experiments
    skipped_count = 0
    if args.skip_cnn_exact:
        original_count = len(experiments)
        experiments = [e for e in experiments if not should_skip_experiment(e, args.skip_cnn_exact)]
        skipped_count = original_count - len(experiments)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} CNN exact experiments (use without --skip-cnn-exact to include)")

    print(f"Running {len(experiments)} experiments...")
    start_time = time.time()

    # Run experiments
    if args.parallel:
        results = run_all_parallel(experiments, max_workers=args.workers)
    else:
        results = run_all_sequential(experiments, base_dir, verbose=args.verbose)

    total_time = time.time() - start_time
    print(f"\nTotal wall time: {total_time:.2f} seconds")

    print_summary(results, skipped_count=skipped_count)


if __name__ == '__main__':
    main()
