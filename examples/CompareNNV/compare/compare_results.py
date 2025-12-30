#!/usr/bin/env python3
"""
Compare NNV and n2v verification results.

This script loads results from both implementations and generates
detailed comparison reports and plots.

Usage:
    python compare_results.py                    # Compare all experiments
    python compare_results.py --exp-id 1         # Compare specific experiment
    python compare_results.py --model fc_mnist   # Compare specific model
    python compare_results.py --exp-id 1 --plot  # Compare and generate plots
    python compare_results.py --model fc_mnist --plot  # Generate plots for model
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import csv

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ALL_EXPERIMENTS, ExperimentConfig, get_experiment


@dataclass
class ComparisonResult:
    """Results of comparing NNV and n2v outputs."""
    experiment_id: int
    model: str
    method: str
    set_type: str
    relax_factor: Optional[float]

    # Status
    n2v_success: bool
    nnv_success: bool
    both_exist: bool

    # Bound comparisons
    max_lb_diff: float
    max_ub_diff: float
    mean_lb_diff: float
    mean_ub_diff: float

    # Robustness comparison
    n2v_robust: int
    nnv_robust: int
    robustness_match: bool

    # Output set count comparison
    n2v_num_sets: int
    nnv_num_sets: int
    num_sets_match: bool

    # Nominal output comparison
    max_nominal_diff: float

    # Time comparison
    n2v_time: float
    nnv_time: float

    # Soundness checks
    bounds_equivalent: bool  # Within tolerance
    n2v_tighter_lb: int   # Number of classes where n2v LB > NNV LB
    n2v_looser_lb: int    # Number of classes where n2v LB < NNV LB
    n2v_tighter_ub: int   # Number of classes where n2v UB < NNV UB
    n2v_looser_ub: int    # Number of classes where n2v UB > NNV UB


def load_n2v_results(exp: ExperimentConfig, base_dir: Path) -> Optional[dict]:
    """Load n2v verification results."""
    output_dir = exp.get_output_dir(base_dir, tool='n2v')
    result_file = output_dir / exp.get_result_filename()

    if not result_file.exists():
        return None

    data = loadmat(str(result_file))
    return extract_results(data)


def load_nnv_results(exp: ExperimentConfig, base_dir: Path) -> Optional[dict]:
    """Load MATLAB NNV verification results."""
    output_dir = exp.get_output_dir(base_dir, tool='nnv')
    result_file = output_dir / exp.get_result_filename()

    if not result_file.exists():
        return None

    data = loadmat(str(result_file))

    # Handle MATLAB nested structure if present
    # Check for 'expResults' (new naming) or 'results' (legacy)
    struct_key = None
    if 'expResults' in data:
        struct_key = 'expResults'
    elif 'results' in data:
        struct_key = 'results'

    if struct_key:
        data = data[struct_key]
        if hasattr(data, 'dtype') and data.dtype.names:
            # Structured array - extract fields
            extracted = {}
            for name in data.dtype.names:
                val = data[name][0, 0]
                if val.size == 1:
                    extracted[name] = val.item()
                else:
                    extracted[name] = val.flatten()
            return extracted

    return extract_results(data)


def extract_results(data: dict) -> dict:
    """Extract results from loaded .mat data."""
    results = {}
    for key, value in data.items():
        if key.startswith('_'):
            continue
        if isinstance(value, np.ndarray):
            if value.size == 1:
                results[key] = value.item()
            else:
                results[key] = value.flatten() if value.ndim > 1 and 1 in value.shape else value
        else:
            results[key] = value
    return results


def compare_experiment(exp: ExperimentConfig, base_dir: Path, tolerance: float = 1e-6) -> ComparisonResult:
    """
    Compare n2v and NNV results for a single experiment.

    Args:
        exp: Experiment configuration
        base_dir: Base directory for CompareNNV
        tolerance: Tolerance for considering bounds equivalent

    Returns:
        ComparisonResult dataclass
    """
    n2v_data = load_n2v_results(exp, base_dir)
    nnv_data = load_nnv_results(exp, base_dir)

    # Initialize with defaults
    result = ComparisonResult(
        experiment_id=exp.id,
        model=exp.model,
        method=exp.method,
        set_type=exp.set_type,
        relax_factor=exp.relax_factor,
        n2v_success=n2v_data is not None and n2v_data.get('success', 0) == 1,
        nnv_success=nnv_data is not None and nnv_data.get('success', 0) == 1,
        both_exist=n2v_data is not None and nnv_data is not None,
        max_lb_diff=float('inf'),
        max_ub_diff=float('inf'),
        mean_lb_diff=float('inf'),
        mean_ub_diff=float('inf'),
        n2v_robust=-1,
        nnv_robust=-1,
        robustness_match=False,
        n2v_num_sets=0,
        nnv_num_sets=0,
        num_sets_match=False,
        max_nominal_diff=float('inf'),
        n2v_time=0,
        nnv_time=0,
        bounds_equivalent=False,
        n2v_tighter_lb=0,
        n2v_looser_lb=0,
        n2v_tighter_ub=0,
        n2v_looser_ub=0,
    )

    if not result.both_exist:
        return result

    # Extract arrays
    n2v_lb = np.array(n2v_data['output_lb']).flatten()
    n2v_ub = np.array(n2v_data['output_ub']).flatten()
    nnv_lb = np.array(nnv_data['output_lb']).flatten()
    nnv_ub = np.array(nnv_data['output_ub']).flatten()

    # Bound differences
    lb_diff = np.abs(n2v_lb - nnv_lb)
    ub_diff = np.abs(n2v_ub - nnv_ub)

    result.max_lb_diff = float(np.max(lb_diff))
    result.max_ub_diff = float(np.max(ub_diff))
    result.mean_lb_diff = float(np.mean(lb_diff))
    result.mean_ub_diff = float(np.mean(ub_diff))

    # Robustness
    result.n2v_robust = int(n2v_data.get('robust', -1))
    result.nnv_robust = int(nnv_data.get('robust', -1))
    result.robustness_match = result.n2v_robust == result.nnv_robust

    # Number of output sets
    result.n2v_num_sets = int(n2v_data.get('num_output_sets', 0))
    result.nnv_num_sets = int(nnv_data.get('num_output_sets', 0))
    result.num_sets_match = result.n2v_num_sets == result.nnv_num_sets

    # Nominal output
    n2v_nominal = np.array(n2v_data.get('nominal_output', [])).flatten()
    nnv_nominal = np.array(nnv_data.get('nominal_output', [])).flatten()
    if len(n2v_nominal) > 0 and len(nnv_nominal) > 0:
        result.max_nominal_diff = float(np.max(np.abs(n2v_nominal - nnv_nominal)))

    # Times
    result.n2v_time = float(n2v_data.get('computation_time', 0))
    result.nnv_time = float(nnv_data.get('computation_time', 0))

    # Bounds equivalence
    result.bounds_equivalent = (result.max_lb_diff < tolerance and result.max_ub_diff < tolerance)

    # Soundness checks
    for i in range(len(n2v_lb)):
        if n2v_lb[i] > nnv_lb[i] + tolerance:
            result.n2v_tighter_lb += 1
        elif n2v_lb[i] < nnv_lb[i] - tolerance:
            result.n2v_looser_lb += 1

        if n2v_ub[i] < nnv_ub[i] - tolerance:
            result.n2v_tighter_ub += 1
        elif n2v_ub[i] > nnv_ub[i] + tolerance:
            result.n2v_looser_ub += 1

    return result


def print_comparison(result: ComparisonResult, verbose: bool = True):
    """Print comparison results."""
    print(f"\n{'='*70}")
    print(f"Experiment {result.experiment_id}: {result.model} / {result.method}")
    if result.relax_factor:
        print(f"Relax Factor: {result.relax_factor}")
    print(f"{'='*70}")

    if not result.both_exist:
        if not result.n2v_success:
            print("n2v results: MISSING")
        if not result.nnv_success:
            print("NNV results: MISSING")
        return

    print(f"\n1. STATUS")
    print(f"   n2v success: {result.n2v_success}")
    print(f"   NNV success: {result.nnv_success}")

    print(f"\n2. BOUND DIFFERENCES")
    print(f"   Max LB difference: {result.max_lb_diff:.2e}")
    print(f"   Max UB difference: {result.max_ub_diff:.2e}")
    print(f"   Mean LB difference: {result.mean_lb_diff:.2e}")
    print(f"   Mean UB difference: {result.mean_ub_diff:.2e}")
    print(f"   Bounds equivalent (< 1e-6): {'YES' if result.bounds_equivalent else 'NO'}")

    print(f"\n3. ROBUSTNESS")
    print(f"   n2v: {'ROBUST' if result.n2v_robust == 1 else 'NOT ROBUST'}")
    print(f"   NNV: {'ROBUST' if result.nnv_robust == 1 else 'NOT ROBUST'}")
    print(f"   Match: {'YES' if result.robustness_match else 'NO'}")

    print(f"\n4. OUTPUT SETS")
    print(f"   n2v: {result.n2v_num_sets}")
    print(f"   NNV: {result.nnv_num_sets}")
    print(f"   Match: {'YES' if result.num_sets_match else 'NO'}")

    print(f"\n5. NOMINAL OUTPUT")
    print(f"   Max difference: {result.max_nominal_diff:.2e}")

    print(f"\n6. COMPUTATION TIME")
    print(f"   n2v: {result.n2v_time:.4f}s")
    print(f"   NNV: {result.nnv_time:.4f}s")
    if result.nnv_time > 0:
        print(f"   Ratio (n2v/NNV): {result.n2v_time/result.nnv_time:.2f}x")

    print(f"\n7. SOUNDNESS")
    print(f"   n2v tighter LB: {result.n2v_tighter_lb} classes")
    print(f"   n2v looser LB: {result.n2v_looser_lb} classes")
    print(f"   n2v tighter UB: {result.n2v_tighter_ub} classes")
    print(f"   n2v looser UB: {result.n2v_looser_ub} classes")


def generate_summary_csv(results: List[ComparisonResult], output_path: Path):
    """Generate a CSV summary of all comparisons."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Experiment ID', 'Model', 'Method', 'Set Type', 'Relax Factor',
            'Both Exist', 'n2v Success', 'NNV Success',
            'Max LB Diff', 'Max UB Diff', 'Bounds Equiv',
            'Robustness Match', 'Num Sets Match',
            'n2v Time', 'NNV Time',
            'n2v Looser LB', 'n2v Looser UB'
        ])

        for r in results:
            writer.writerow([
                r.experiment_id, r.model, r.method, r.set_type, r.relax_factor or '',
                r.both_exist, r.n2v_success, r.nnv_success,
                f"{r.max_lb_diff:.2e}", f"{r.max_ub_diff:.2e}", r.bounds_equivalent,
                r.robustness_match, r.num_sets_match,
                f"{r.n2v_time:.4f}", f"{r.nnv_time:.4f}",
                r.n2v_looser_lb, r.n2v_looser_ub
            ])

    print(f"Summary saved to: {output_path}")


def plot_nominal_comparison(
    exp: ExperimentConfig,
    n2v_data: dict,
    nnv_data: dict,
    output_path: Path
):
    """
    Plot comparison of nominal (forward pass) outputs.

    Creates a bar chart comparing n2v and NNV nominal outputs side-by-side.
    """
    n2v_nominal = np.array(n2v_data.get('nominal_output', [])).flatten()
    nnv_nominal = np.array(nnv_data.get('nominal_output', [])).flatten()

    if len(n2v_nominal) == 0 or len(nnv_nominal) == 0:
        print("  Warning: Missing nominal output data, skipping plot")
        return

    num_classes = len(n2v_nominal)
    x = np.arange(num_classes)
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top plot: Side-by-side bar comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, n2v_nominal, width, label='n2v', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, nnv_nominal, width, label='NNV', color='darkorange', alpha=0.8)

    ax1.set_xlabel('Class')
    ax1.set_ylabel('Output Logit')
    ax1.set_title(f'Nominal Output Comparison: {exp.model}')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Highlight predicted class
    n2v_pred = int(n2v_data.get('nominal_pred', n2v_nominal.argmax()))
    ax1.axvline(x=n2v_pred, color='green', linestyle='--', alpha=0.5, label=f'Predicted: {n2v_pred}')

    # Bottom plot: Difference
    ax2 = axes[1]
    diff = n2v_nominal - nnv_nominal
    colors = ['green' if d >= 0 else 'red' for d in diff]
    ax2.bar(x, diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Difference (n2v - NNV)')
    ax2.set_title(f'Nominal Output Difference (max: {np.abs(diff).max():.2e})')
    ax2.set_xticks(x)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_reachable_set_comparison(
    exp: ExperimentConfig,
    n2v_data: dict,
    nnv_data: dict,
    output_path: Path
):
    """
    Plot comparison of reachable set bounds.

    Creates an error bar plot showing the reachable output ranges for both tools.
    """
    n2v_lb = np.array(n2v_data.get('output_lb', [])).flatten()
    n2v_ub = np.array(n2v_data.get('output_ub', [])).flatten()
    nnv_lb = np.array(nnv_data.get('output_lb', [])).flatten()
    nnv_ub = np.array(nnv_data.get('output_ub', [])).flatten()

    if len(n2v_lb) == 0 or len(nnv_lb) == 0:
        print("  Warning: Missing bound data, skipping plot")
        return

    num_classes = len(n2v_lb)
    x = np.arange(num_classes)

    # Get nominal outputs for reference points
    n2v_nominal = np.array(n2v_data.get('nominal_output', [])).flatten()
    nnv_nominal = np.array(nnv_data.get('nominal_output', [])).flatten()

    # Get true label
    true_label = int(n2v_data.get('test_label', -1))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top plot: Overlaid reachable sets
    ax1 = axes[0]

    # n2v bounds (blue)
    n2v_mid = (n2v_lb + n2v_ub) / 2
    n2v_err = (n2v_ub - n2v_lb) / 2
    ax1.errorbar(x - 0.15, n2v_mid, yerr=n2v_err, fmt='o', color='steelblue',
                 linewidth=2, capsize=5, capthick=2, markersize=8,
                 label='n2v', alpha=0.8)

    # NNV bounds (orange)
    nnv_mid = (nnv_lb + nnv_ub) / 2
    nnv_err = (nnv_ub - nnv_lb) / 2
    ax1.errorbar(x + 0.15, nnv_mid, yerr=nnv_err, fmt='s', color='darkorange',
                 linewidth=2, capsize=5, capthick=2, markersize=8,
                 label='NNV', alpha=0.8)

    # Nominal outputs as reference
    if len(n2v_nominal) > 0:
        ax1.scatter(x - 0.15, n2v_nominal, marker='x', color='navy', s=100,
                   zorder=5, label='n2v Nominal')
    if len(nnv_nominal) > 0:
        ax1.scatter(x + 0.15, nnv_nominal, marker='+', color='darkred', s=100,
                   zorder=5, label='NNV Nominal')

    # Highlight true class
    if true_label >= 0:
        ax1.axvspan(true_label - 0.4, true_label + 0.4, alpha=0.2, color='green',
                   label=f'True Class ({true_label})')

    ax1.set_xlabel('Class')
    ax1.set_ylabel('Output Logit')

    # Build title with method info
    title = f'Reachable Set Comparison: {exp.model} / {exp.method}'
    if exp.relax_factor:
        title += f' (relax={exp.relax_factor})'
    ax1.set_title(title)

    ax1.set_xticks(x)
    ax1.set_xlim(-0.5, num_classes - 0.5)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Bottom plot: Bound differences
    ax2 = axes[1]

    lb_diff = n2v_lb - nnv_lb  # Positive = n2v tighter (higher LB)
    ub_diff = n2v_ub - nnv_ub  # Negative = n2v tighter (lower UB)

    width = 0.35
    ax2.bar(x - width/2, lb_diff, width, label='LB diff (n2v - NNV)', color='steelblue', alpha=0.7)
    ax2.bar(x + width/2, ub_diff, width, label='UB diff (n2v - NNV)', color='darkorange', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax2.set_xlabel('Class')
    ax2.set_ylabel('Bound Difference')
    ax2.set_title(f'Bound Differences (max LB: {np.abs(lb_diff).max():.2e}, max UB: {np.abs(ub_diff).max():.2e})')
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add interpretation note
    fig.text(0.5, 0.01,
             'LB diff > 0: n2v tighter (higher lower bound)  |  UB diff < 0: n2v tighter (lower upper bound)',
             ha='center', fontsize=9, style='italic', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_experiment_comparison(exp: ExperimentConfig, base_dir: Path) -> bool:
    """
    Generate all comparison plots for an experiment.

    Returns True if plots were generated successfully.
    """
    n2v_data = load_n2v_results(exp, base_dir)
    nnv_data = load_nnv_results(exp, base_dir)

    if n2v_data is None:
        print(f"  Missing n2v results for experiment {exp.id}")
        return False
    if nnv_data is None:
        print(f"  Missing NNV results for experiment {exp.id}")
        return False

    # Create output directory
    plot_dir = base_dir / 'outputs' / 'comparisons' / 'plots' / exp.model

    # Generate filename suffix
    method_str = exp.method.replace('-', '_')
    if exp.relax_factor:
        method_str += f"_{exp.relax_factor:.2f}"

    # Plot nominal outputs
    nominal_path = plot_dir / f"nominal_{method_str}.png"
    plot_nominal_comparison(exp, n2v_data, nnv_data, nominal_path)

    # Plot reachable set bounds
    bounds_path = plot_dir / f"bounds_{method_str}.png"
    plot_reachable_set_comparison(exp, n2v_data, nnv_data, bounds_path)

    return True


def generate_all_plots(base_dir: Path, experiments: List[ExperimentConfig] = None):
    """Generate comparison plots for all (or specified) experiments."""
    if experiments is None:
        experiments = ALL_EXPERIMENTS

    print(f"\nGenerating plots for {len(experiments)} experiments...")

    success_count = 0
    for exp in experiments:
        print(f"\nExperiment {exp.id}: {exp.model} / {exp.method}")
        if plot_experiment_comparison(exp, base_dir):
            success_count += 1

    print(f"\nGenerated plots for {success_count}/{len(experiments)} experiments")
    print(f"Plots saved to: {base_dir / 'outputs' / 'comparisons' / 'plots'}")


def main():
    parser = argparse.ArgumentParser(description='Compare n2v and NNV verification results')
    parser.add_argument('--exp-id', type=int, help='Compare specific experiment')
    parser.add_argument('--model', type=str, help='Compare specific model')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Tolerance for equivalence')
    parser.add_argument('--summary-only', action='store_true', help='Only print summary')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--plot-only', action='store_true', help='Only generate plots (skip text comparison)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    if args.exp_id:
        exp = get_experiment(args.exp_id)

        if not args.plot_only:
            result = compare_experiment(exp, base_dir, tolerance=args.tolerance)
            print_comparison(result)

        if args.plot or args.plot_only:
            print("\nGenerating plots...")
            plot_experiment_comparison(exp, base_dir)

    elif args.model:
        from config import get_experiments_by_model
        experiments = get_experiments_by_model(args.model)

        if not args.plot_only:
            results = []
            for exp in experiments:
                result = compare_experiment(exp, base_dir, tolerance=args.tolerance)
                results.append(result)
                if not args.summary_only:
                    print_comparison(result)

            # Print summary
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            both_exist = sum(1 for r in results if r.both_exist)
            bounds_equiv = sum(1 for r in results if r.bounds_equivalent)
            robust_match = sum(1 for r in results if r.robustness_match)
            print(f"Total experiments: {len(results)}")
            print(f"Both results exist: {both_exist}")
            print(f"Bounds equivalent: {bounds_equiv}")
            print(f"Robustness match: {robust_match}")

        if args.plot or args.plot_only:
            generate_all_plots(base_dir, experiments)

    else:
        # Compare all experiments
        if not args.plot_only:
            results = []
            for exp in ALL_EXPERIMENTS:
                result = compare_experiment(exp, base_dir, tolerance=args.tolerance)
                results.append(result)
                if not args.summary_only:
                    print_comparison(result, verbose=False)

            # Generate summary
            summary_path = base_dir / 'outputs' / 'comparisons' / 'summary.csv'
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            generate_summary_csv(results, summary_path)

            # Print summary statistics
            print(f"\n{'='*70}")
            print("OVERALL SUMMARY")
            print(f"{'='*70}")

            both_exist = sum(1 for r in results if r.both_exist)
            bounds_equiv = sum(1 for r in results if r.bounds_equivalent)
            robust_match = sum(1 for r in results if r.robustness_match)
            looser_any = sum(1 for r in results if r.n2v_looser_lb > 0 or r.n2v_looser_ub > 0)

            print(f"Total experiments: {len(results)}")
            print(f"Both results exist: {both_exist}/{len(results)}")
            print(f"Bounds equivalent (< {args.tolerance}): {bounds_equiv}/{both_exist}")
            print(f"Robustness match: {robust_match}/{both_exist}")
            print(f"n2v has looser bounds: {looser_any}/{both_exist}")

            if both_exist > 0:
                avg_lb_diff = np.mean([r.max_lb_diff for r in results if r.both_exist])
                avg_ub_diff = np.mean([r.max_ub_diff for r in results if r.both_exist])
                print(f"\nAverage max LB difference: {avg_lb_diff:.2e}")
                print(f"Average max UB difference: {avg_ub_diff:.2e}")

        if args.plot or args.plot_only:
            generate_all_plots(base_dir)


if __name__ == '__main__':
    main()
