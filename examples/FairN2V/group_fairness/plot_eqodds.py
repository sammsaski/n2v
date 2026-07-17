"""
Plot Flow Equalized-Odds Results
Generates a per-group TPR / FPR bar chart from the CSV written by
verify_eqodds.py. Two panels per dataset (left = TPR, right = FPR); each panel is
a cluster of bars per model, one bar per sensitive group (height = rate on the
flow cell, error bar = the Clopper-Pearson interval). The shaded band over each
cluster is the gap tolerance [min_rate, min_rate + tau] -- equalized odds asks
every group's bar to top out within tau of the lowest, so a bar poking above the
band is a gap > tau. Each cluster is annotated with that panel's sub-verdict.

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


def _draw_panel(ax, models, rates, intervals, subverdicts, real_subverdicts, *, k,
                names, tau, ylabel, legend):
    """One panel (TPR or FPR): per-model clusters of per-group bars + tau band.
    Bars are the flow rates; each cluster is labelled with the flow sub-verdict
    and the real-data one."""
    n = len(models)
    x = np.arange(n)
    width = 0.8 / k
    colors = plt.cm.tab10(np.arange(k))
    cluster_tops = np.zeros(n)
    for g in range(k):
        r = np.array([row[g] for row in rates])
        ci = np.array([iv[g] for iv in intervals])          # (n, 2): lo/hi per model
        errs = np.vstack([np.clip(r - ci[:, 0], 0, None), np.clip(ci[:, 1] - r, 0, None)])
        offset = (g - (k - 1) / 2) * width
        ax.bar(x + offset, r, width, yerr=errs, capsize=3, color=colors[g],
               label=(names(g) if legend else None))
        cluster_tops = np.maximum(cluster_tops, r + errs[1])
    cluster_tops = np.nan_to_num(cluster_tops, nan=0.0)      # flow not fit -> no bars
    for i, (rlist, flow_v, real_v) in enumerate(zip(rates, subverdicts, real_subverdicts)):
        rmin = min(rlist)
        if rmin == rmin:                                     # NaN check: skip band if no flow
            ax.fill_between([x[i] - 0.45, x[i] + 0.45], rmin, rmin + tau,
                            color='black', alpha=0.12, linewidth=0, zorder=0,
                            label=('gap tolerance τ' if (legend and i == 0) else None))
        ax.text(x[i], cluster_tops[i] + 0.01, f"flow: {str(flow_v).upper()}",
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                color=_LABEL_COLOR.get(flow_v, 'black'))
        ax.text(x[i], cluster_tops[i] + 0.06, f"real: {str(real_v).upper()}",
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                color=_LABEL_COLOR.get(real_v, 'black'))
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, min(1.0, cluster_tops.max() + 0.14))


def main(config=None):
    ## Setup
    if config is None:
        # Standalone: pick the most recent results/group_fairness/<ts>/ subdir.
        results_root = Path(__file__).resolve().parent.parent / 'results' / 'group_fairness'
        subdirs = [d for d in results_root.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(
                f"No results subdir found under {results_root}. Run verify_eqodds.py first.")
        config = {
            'output_dir': max(subdirs, key=lambda d: d.stat().st_mtime),
            'save_png': True,
            'save_pdf': True,
        }

    results_dir = config['output_dir']
    eqodds_files = list(results_dir.glob('eqodds_*.csv'))
    if not eqodds_files:
        raise FileNotFoundError(
            f"No eqodds_*.csv found in {results_dir}. Please run verify_eqodds.py first.")
    csv_path = max(eqodds_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from:\n  {csv_path}")

    data = pd.read_csv(csv_path, dtype={'GroupNames': str}).fillna({'GroupNames': ''})
    dataset = str(data['Dataset'].iloc[0])
    tau = float(data['Tau'].iloc[0])

    models = list(data['Model'])
    tpr_rates = [_parse_groups(r) for r in data['TPRRates']]
    tpr_ci = [_parse_intervals(r) for r in data['TPRIntervals']]
    fpr_rates = [_parse_groups(r) for r in data['FPRRates']]
    fpr_ci = [_parse_intervals(r) for r in data['FPRIntervals']]
    tpr_sub = list(data['TPRVerdict'])
    fpr_sub = list(data['FPRVerdict'])
    tpr_real_sub = list(data['RealTPRVerdict'])
    fpr_real_sub = list(data['RealFPRVerdict'])
    k = len(tpr_rates[0])
    names = data['GroupNames'].iloc[0].split(';') if data['GroupNames'].iloc[0] else None

    def group_label(g):
        return names[g] if names and g < len(names) else f'group {g}'

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(max(9.0, 3.2 * len(models)), 5.0))
    _draw_panel(axL, models, tpr_rates, tpr_ci, tpr_sub, tpr_real_sub, k=k,
                names=group_label, tau=tau,
                ylabel='TPR = P(favorable | group, Y=favorable)', legend=True)
    _draw_panel(axR, models, fpr_rates, fpr_ci, fpr_sub, fpr_real_sub, k=k,
                names=group_label, tau=tau,
                ylabel='FPR = P(favorable | group, Y=unfavorable)', legend=False)
    axL.set_title('True-positive rate  (= equal opportunity)')
    axR.set_title('False-positive rate')
    fig.suptitle(f'Equal opportunity + equalized odds: {dataset}  (τ={tau})\n'
                 f'cluster labels = flow / real sub-verdicts; TPR = equal opportunity, '
                 f'TPR+FPR = equalized odds')
    axL.legend(loc='upper left', bbox_to_anchor=(0.0, -0.08), ncol=k + 1, fontsize=8)
    fig.tight_layout()

    png_path = results_dir / f'eqodds_{dataset}.png'
    if config.get('save_png', True):
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
    if config.get('save_pdf', True):
        fig.savefig(results_dir / f'eqodds_{dataset}.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: eqodds_{dataset}.png/pdf")
    print(" ")
    print("======= EQUALIZED-ODDS PLOTTING COMPLETE ==========")
    return png_path


if __name__ == "__main__":
    main()
