"""
Plot tightness ablation results.

Reads exp_tightness_ablation.csv and produces:
  - exp_tightness_ablation.png — grid of one panel per knob; each panel
    shows reach-set volume vs knob value (mean +/- std), with empirical
    coverage on a twin axis and a horizontal line at 1 - epsilon_1.
  - exp_tightness_ablation_pareto.png — volume vs delta_1 Pareto plot
    for the conformal block (epsilon_1, ell_offset, n_calib).
"""

import os
import csv
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUTPUT_DIR, 'exp_tightness_ablation.csv')
GRID_PATH = os.path.join(OUTPUT_DIR, 'exp_tightness_ablation.png')
PARETO_PATH = os.path.join(OUTPUT_DIR, 'exp_tightness_ablation_pareto.png')


# Knobs that benefit from a log x-axis (range spans >= 1 decade)
LOG_KNOBS = {'hidden', 'n_train', 'epochs', 'n_calib', 'epsilon_1'}

# Knob -> display label
KNOB_LABEL = {
    't': 'flow time t',
    'n_layers': 'flow depth (n_layers)',
    'hidden': 'flow width (hidden)',
    'n_train': 'N_train',
    'epochs': 'epochs',
    'coupling': 'OT coupling',
    'epsilon_1': 'epsilon_1',
    'ell_offset': 'ell offset (m - ell)',
    'n_calib': 'N_calib',
}


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k in ('block', 'knob', 'value', 'cfg_coupling', 'cfg_activation'):
                    continue
                if v == '' or v is None:
                    continue
                try:
                    row[k] = float(v) if '.' in v or 'e' in v.lower() else int(v)
                except ValueError:
                    pass
            rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, Any]], knob: str):
    """Group rows by knob value, return sorted list of (value, vol_mean, vol_std, cov_mean, cov_std, eps1)."""
    by_value = defaultdict(list)
    for r in rows:
        if r['knob'] != knob:
            continue
        by_value[r['value']].append(r)

    def coerce(v):
        try:
            return float(v)
        except ValueError:
            return v

    sortable = all(isinstance(coerce(v), (int, float)) for v in by_value.keys())
    if sortable:
        keys = sorted(by_value.keys(), key=lambda x: float(x))
    else:
        keys = sorted(by_value.keys())

    out = []
    for k in keys:
        runs = by_value[k]
        vols = np.array([float(r['volume']) for r in runs])
        covs = np.array([float(r['empirical_coverage']) for r in runs])
        delta1 = float(runs[0]['delta_1'])
        eps1 = float(runs[0]['cfg_epsilon_1'])
        out.append({
            'value': coerce(k),
            'vol_mean': vols.mean(),
            'vol_std': vols.std(ddof=0),
            'cov_mean': covs.mean(),
            'cov_std': covs.std(ddof=0),
            'delta_1': delta1,
            'epsilon_1': eps1,
        })
    return out


def plot_knob_panel(ax, knob: str, agg: List[Dict[str, Any]]):
    """One panel: volume vs knob, with coverage on twin axis."""
    values = [a['value'] for a in agg]
    vol_mean = np.array([a['vol_mean'] for a in agg])
    vol_std = np.array([a['vol_std'] for a in agg])
    cov_mean = np.array([a['cov_mean'] for a in agg])
    cov_std = np.array([a['cov_std'] for a in agg])

    is_numeric = all(isinstance(v, (int, float)) for v in values)

    if is_numeric:
        x = np.array(values, dtype=float)
        ax.errorbar(x, vol_mean, yerr=vol_std, marker='o', color='C0',
                    capsize=3, label='volume')
        if knob in LOG_KNOBS:
            ax.set_xscale('log')
    else:
        x = np.arange(len(values))
        ax.errorbar(x, vol_mean, yerr=vol_std, marker='o', color='C0',
                    capsize=3, label='volume')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in values])

    ax.set_xlabel(KNOB_LABEL.get(knob, knob))
    ax.set_ylabel('reach-set volume', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')

    ax2 = ax.twinx()
    if is_numeric:
        ax2.errorbar(x, cov_mean, yerr=cov_std, marker='s', color='C3',
                     capsize=3, alpha=0.7, label='coverage')
    else:
        ax2.errorbar(x, cov_mean, yerr=cov_std, marker='s', color='C3',
                     capsize=3, alpha=0.7, label='coverage')

    eps1 = agg[0]['epsilon_1']
    ax2.axhline(1.0 - eps1, color='C3', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('empirical coverage', color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')
    ax2.set_ylim(min(0.99, (1.0 - eps1) - 0.005), 1.001)

    ax.set_title(KNOB_LABEL.get(knob, knob))
    ax.grid(True, alpha=0.3)


def plot_grid(rows: List[Dict[str, Any]]):
    sweep_order = [
        ('A', 't'),
        ('A', 'n_layers'),
        ('A', 'hidden'),
        ('A', 'n_train'),
        ('A', 'epochs'),
        ('A', 'coupling'),
        ('B', 'epsilon_1'),
        ('B', 'ell_offset'),
        ('B', 'n_calib'),
    ]
    knobs = [k for _, k in sweep_order if any(r['knob'] == k for r in rows)]

    n = len(knobs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for ax, knob in zip(axes, knobs):
        agg = aggregate(rows, knob)
        plot_knob_panel(ax, knob, agg)

    for ax in axes[len(knobs):]:
        ax.axis('off')

    fig.suptitle('Tightness ablation: volume + coverage vs each knob', y=1.00,
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(GRID_PATH, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {GRID_PATH}")


def plot_pareto(rows: List[Dict[str, Any]]):
    """Volume vs delta_1 Pareto plot for the conformal block."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    block_b_knobs = [k for k in ('epsilon_1', 'ell_offset', 'n_calib')
                     if any(r['knob'] == k for r in rows)]
    colors = {'epsilon_1': 'C0', 'ell_offset': 'C1', 'n_calib': 'C2'}

    for knob in block_b_knobs:
        agg = aggregate(rows, knob)
        d1 = np.array([a['delta_1'] for a in agg])
        vol = np.array([a['vol_mean'] for a in agg])
        vol_std = np.array([a['vol_std'] for a in agg])
        ax.errorbar(d1, vol, yerr=vol_std, marker='o', color=colors[knob],
                    capsize=3, label=KNOB_LABEL[knob], linewidth=1.5)

        for a in agg:
            ax.annotate(
                f"{a['value']}",
                xy=(a['delta_1'], a['vol_mean']),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=7,
                color=colors[knob],
                alpha=0.8,
            )

    ax.set_xlabel('conformal confidence delta_1')
    ax.set_ylabel('reach-set volume')
    ax.set_title('Conformal Pareto: volume vs delta_1')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PARETO_PATH, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {PARETO_PATH}")


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Missing {CSV_PATH}. Run exp_tightness_ablation.py first."
        )
    rows = load_rows(CSV_PATH)
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")
    plot_grid(rows)
    plot_pareto(rows)


if __name__ == '__main__':
    main()
