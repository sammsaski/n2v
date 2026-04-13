"""
Plots for the Coverage Validation Experiment.

Reads results from exp_coverage_validation.csv and produces two figures:
  1. Three-panel histogram of the three probabilistic claims
  2. Calibration scatter: ground truth vs pipeline-estimated robustness

Run this script after exp_coverage_validation.py has produced the CSV.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUTPUT_DIR, 'exp_coverage_validation.csv')
HISTOGRAM_PATH = os.path.join(OUTPUT_DIR, 'exp_coverage_validation_histograms.png')
CALIBRATION_PATH = os.path.join(OUTPUT_DIR, 'exp_coverage_validation_calibration.png')


def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at {CSV_PATH}. "
            f"Run exp_coverage_validation.py first."
        )
    df = pd.read_csv(CSV_PATH)
    # Drop failed rows for plotting
    df = df.dropna(subset=['conformal_coverage'])
    return df


def plot_histograms(df):
    """Three-panel histogram of the three probabilistic claims."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Theoretical bounds (assumes all runs used the same parameters)
    eps1 = df['epsilon_1'].iloc[0]
    eps2 = df['epsilon_2'].iloc[0]
    conformal_bound = 1 - eps1
    scenario_bound = eps2
    joint_bound = (1 - eps1) * (1 - eps2)

    # Left: conformal coverage
    ax = axes[0]
    ax.hist(
        df['conformal_coverage'].dropna(),
        bins=20, color='C0', alpha=0.7, edgecolor='black',
    )
    ax.axvline(
        conformal_bound, color='red', linestyle='--', linewidth=2,
        label=f'theoretical bound\n1 - $\\epsilon_1$ = {conformal_bound:.4f}',
    )
    mean_val = df['conformal_coverage'].mean()
    ax.axvline(
        mean_val, color='green', linestyle=':', linewidth=2,
        label=f'empirical mean = {mean_val:.4f}',
    )
    ax.set_xlabel('conformal coverage')
    ax.set_ylabel('count')
    ax.set_title('Conformal layer:\n$\\Pr[f(x) \\in R] \\geq 1 - \\epsilon_1$')
    ax.legend(fontsize=8, loc='upper left')

    # Middle: scenario violation rate
    ax = axes[1]
    violation_data = df['scenario_violation_rate'].dropna()
    if len(violation_data) > 0:
        ax.hist(
            violation_data,
            bins=20, color='C1', alpha=0.7, edgecolor='black',
        )
    ax.axvline(
        scenario_bound, color='red', linestyle='--', linewidth=2,
        label=f'theoretical bound\n$\\epsilon_2$ = {scenario_bound:.6f}',
    )
    if len(violation_data) > 0:
        mean_val = violation_data.mean()
        ax.axvline(
            mean_val, color='green', linestyle=':', linewidth=2,
            label=f'empirical mean = {mean_val:.6f}',
        )
    ax.set_xlabel('scenario violation rate')
    ax.set_ylabel('count')
    ax.set_title(
        'Scenario layer:\n$\\Pr[\\mathrm{violation} | y \\in R] '
        '\\leq \\epsilon_2$'
    )
    ax.legend(fontsize=8, loc='upper right')

    # Right: joint satisfaction
    ax = axes[2]
    ax.hist(
        df['joint_spec_satisfaction'].dropna(),
        bins=20, color='C2', alpha=0.7, edgecolor='black',
    )
    ax.axvline(
        joint_bound, color='red', linestyle='--', linewidth=2,
        label=f'theoretical bound\n$(1-\\epsilon_1)(1-\\epsilon_2)$ = '
              f'{joint_bound:.4f}',
    )
    mean_val = df['joint_spec_satisfaction'].mean()
    ax.axvline(
        mean_val, color='green', linestyle=':', linewidth=2,
        label=f'empirical mean = {mean_val:.4f}',
    )
    ax.set_xlabel('joint spec satisfaction')
    ax.set_ylabel('count')
    ax.set_title('Joint claim')
    ax.legend(fontsize=8, loc='upper left')

    plt.suptitle('Coverage Validation: Empirical vs Theoretical Bounds',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PATH, dpi=150)
    plt.close()
    print(f"Saved {HISTOGRAM_PATH}")


def plot_calibration(df):
    """Scatter of ground truth vs joint spec satisfaction,
    marker shape by pipeline outcome, marker color by n2v ground truth."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Color by n2v ground truth
    n2v_colors = {
        True: 'green',   # n2v says robust
        False: 'red',    # n2v says non-robust
        None: 'gray',    # n2v timed out
    }
    # Marker by pipeline outcome
    outcome_markers = {
        'verified': 'o',
        'falsified': 'X',
        'unknown': '^',
        'error': 's',
    }

    for n2v_value, color in n2v_colors.items():
        for outcome, marker in outcome_markers.items():
            if n2v_value is None:
                sub = df[df['n2v_spec_holds'].isna() & (df['outcome'] == outcome)]
            else:
                sub = df[(df['n2v_spec_holds'] == n2v_value) & (df['outcome'] == outcome)]
            if len(sub) == 0:
                continue
            label_n2v = (
                'n2v: robust' if n2v_value is True
                else 'n2v: non-robust' if n2v_value is False
                else 'n2v: timeout'
            )
            label = f'{label_n2v} + {outcome} (n={len(sub)})'
            ax.scatter(
                sub['ground_truth_robustness'],
                sub['joint_spec_satisfaction'],
                s=100, c=color, marker=marker,
                edgecolors='black', alpha=0.8,
                label=label,
            )

    # Diagonal
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5,
            label='y = x (perfect calibration)')
    ax.set_xlabel('Ground truth robustness rate (MC)')
    ax.set_ylabel('Joint spec satisfaction (pipeline)')
    ax.set_title('Calibration: pipeline outcome vs n2v ground truth')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(CALIBRATION_PATH, dpi=150)
    plt.close()
    print(f"Saved {CALIBRATION_PATH}")


def run():
    print(f"Loading data from {CSV_PATH}...")
    df = load_data()
    print(f"Loaded {len(df)} successful runs.")

    plot_histograms(df)
    plot_calibration(df)
    print("Done.")


if __name__ == '__main__':
    run()
