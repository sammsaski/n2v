"""
Plot Flow Demographic-Parity Results
Generates a per-group favorable-rate bar chart from the verification CSV written
by verify_parity.py. One figure per dataset: a cluster of bars per model
(one bar per sensitive group, height = favorable rate on the flow samples, error
bar = the Clopper-Pearson interval). The shaded band over each cluster is the
parity floor c * max_rate, drawn across the top group's CP interval; a group
whose interval is entirely below the band is a proven violation. Each cluster is
annotated with both verdicts (flow population / real data).

This script can be run standalone or called from run_group_fairness.py.
Standalone: looks in the most recent results/group_fairness/<ts>/ subdir.
Runner-driven: uses `config['output_dir']` from the caller.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')                       # headless: write files, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_LABEL_COLOR = {'fair': 'green', 'unfair': 'red', 'inconclusive': 'gray'}


def _parse_groups(cell):
    """';'-joined per-group floats -> list[float]."""
    return [float(x) for x in str(cell).split(';')]


def _parse_intervals(cell):
    """';'-joined 'lo:hi' per-group intervals -> list[(lo, hi)]."""
    out = []
    for part in str(cell).split(';'):
        lo, hi = part.split(':')
        out.append((float(lo), float(hi)))
    return out


def main(config=None):
    ## Setup
    if config is None:
        # Standalone: pick the most recent results/group_fairness/<ts>/ subdir.
        # results/ lives in the FairN2V dir, one level up from this script.
        results_root = Path(__file__).resolve().parent.parent / 'results' / 'group_fairness'
        subdirs = [d for d in results_root.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(
                f"No results subdir found under {results_root}. Run verify_parity.py first.")
        config = {
            'output_dir': max(subdirs, key=lambda d: d.stat().st_mtime),
            'save_png': True,
            'save_pdf': True,
        }

    results_dir = config['output_dir']

    # Find the parity CSV in the results directory
    parity_files = list(results_dir.glob('parity_*.csv'))
    if not parity_files:
        raise FileNotFoundError(
            f"No parity_*.csv found in {results_dir}. Please run verify_parity.py first.")
    csv_parity = max(parity_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from:\n  {csv_parity}")

    ## Load CSV Data
    data = pd.read_csv(csv_parity, dtype={'GroupNames': str}).fillna({'GroupNames': ''})
    dataset = str(data['Dataset'].iloc[0])
    c = float(data['C'].iloc[0])

    # Per-model parsed rows, in file order
    rows = []
    for _, r in data.iterrows():
        rows.append({
            'model': r['Model'],
            'verdict': r['Verdict'],
            'real_verdict': r['RealVerdict'],
            'rates': _parse_groups(r['FlowRates']),
            'intervals': _parse_intervals(r['Intervals']),
        })
    k = len(rows[0]['rates'])
    names = data['GroupNames'].iloc[0].split(';') if data['GroupNames'].iloc[0] else None

    def group_label(g):
        return names[g] if names and g < len(names) else f'group {g}'

    ## Figure: per-group favorable-rate bars, one cluster per model
    x = np.arange(len(rows))
    width = 0.8 / k
    colors = plt.cm.tab10(np.arange(k))
    fig, ax = plt.subplots(figsize=(max(6.0, 2.4 * len(rows)), 5.0))

    cluster_tops = np.zeros(len(rows))
    for g in range(k):
        rates = np.array([row['rates'][g] for row in rows])
        lowers = np.array([row['rates'][g] - row['intervals'][g][0] for row in rows])
        uppers = np.array([row['intervals'][g][1] - row['rates'][g] for row in rows])
        errs = np.vstack([np.clip(lowers, 0, None), np.clip(uppers, 0, None)])
        offset = (g - (k - 1) / 2) * width
        ax.bar(x + offset, rates, width, yerr=errs, capsize=3,
               color=colors[g], label=group_label(g))
        cluster_tops = np.maximum(cluster_tops, rates + errs[1])
    cluster_tops = np.nan_to_num(cluster_tops, nan=0.0)      # flow not fit -> no bars

    # Per-model parity floor (c * top group's CP interval) and verdict annotation.
    for i, row in enumerate(rows):
        g_max = max(range(k), key=lambda g: row['rates'][g])
        lo_m, hi_m = row['intervals'][g_max]
        if lo_m == lo_m:                                     # NaN check: skip band if no flow
            ax.fill_between([x[i] - 0.45, x[i] + 0.45], c * lo_m, c * hi_m,
                            color='black', alpha=0.15, linewidth=0, zorder=0,
                            label='parity floor (80%-rule)' if i == 0 else None)
        ax.text(x[i], cluster_tops[i] + 0.02, f"flow: {str(row['verdict']).upper()}",
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=_LABEL_COLOR.get(row['verdict'], 'black'))
        ax.text(x[i], cluster_tops[i] + 0.07, f"real: {str(row['real_verdict']).upper()}",
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=_LABEL_COLOR.get(row['real_verdict'], 'black'))

    ax.set_xticks(x)
    ax.set_xticklabels([row['model'] for row in rows])
    ax.set_ylabel('P(favorable | group)')
    ax.set_ylim(0, min(1.0, cluster_tops.max() + 0.16))
    ax.set_title(f'Demographic parity (flow population): {dataset}  (c={c})')
    # Legend outside the axes so it never occludes a tall cluster or its label.
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()

    png_path = results_dir / f'parity_{dataset}.png'
    if config.get('save_png', True):
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
    if config.get('save_pdf', True):
        fig.savefig(results_dir / f'parity_{dataset}.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: parity_{dataset}.png/pdf")
    print(" ")
    print("======= FLOW PARITY PLOTTING COMPLETE ==========")
    return png_path


if __name__ == "__main__":
    main()
