"""
Plot FairN2V Results
Generates figures and LaTeX tables from verification CSV results.
Outputs:
    (1) LaTeX table for counterfactual fairness
    (2) Combined individual fairness stacked area plot
    (3) LaTeX table for timing results (separated by fairness type)

This script can be run standalone or called from run_individual_fairness.py.
Standalone: looks in the most recent results/individual_fairness/<ts>/ subdir.
Runner-driven: uses `config.output_dir` from caller workspace.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless: write files, no display
import matplotlib.pyplot as plt


def _print_table(headers, rows):
    """Print a column-aligned plain-text table to the console.

    First column is left-aligned (model names); the rest are right-aligned
    (numbers). A dashed rule separates the header from the body.
    """
    all_rows = [headers] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]

    def fmt(row):
        cells = [str(c).ljust(widths[i]) if i == 0 else str(c).rjust(widths[i])
                 for i, c in enumerate(row)]
        return '  '.join(cells)

    print(fmt(headers))
    print('  '.join('-' * w for w in widths))
    for row in rows:
        print(fmt(row))


def main(config=None):
    if config is None:
        # Standalone: pick the most recent results/individual_fairness/<ts>/ subdir.
        # results/ lives in the FairN2V dir, one level up from this script.
        results_root = Path(__file__).resolve().parent.parent / 'results' / 'individual_fairness'
        subdirs = [d for d in results_root.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(
                f"No results subdir found under {results_root}. Run verify_individual.py first.")
        config = {
            'output_dir': max(subdirs, key=lambda d: d.stat().st_mtime),
            'save_png': True,
            'save_pdf': True,
        }

    results_dir = config['output_dir']

    counterfactual_files = list(results_dir.glob('counterfactual_*.csv'))
    individual_files = list(results_dir.glob('individual_*.csv'))
    timing_files = list(results_dir.glob('timing_*.csv'))

    if not counterfactual_files or not individual_files or not timing_files:
        raise FileNotFoundError(
            f"CSV files not found in {results_dir}. Please run verify_individual.py first.")

    # Most recent file of each family
    csv_counterfactual = max(counterfactual_files, key=lambda p: p.stat().st_mtime)
    csv_individual = max(individual_files, key=lambda p: p.stat().st_mtime)
    csv_timing = max(timing_files, key=lambda p: p.stat().st_mtime)

    print("Loading results from:")
    print(f"  {csv_counterfactual}")
    print(f"  {csv_individual}")
    print(f"  {csv_timing}")

    counterfactual_data = pd.read_csv(csv_counterfactual)
    individual_data = pd.read_csv(csv_individual)
    timing_data = pd.read_csv(csv_timing)

    color_fair = '#2ecc71'
    color_unfair = '#e74c3c'

    models = sorted(individual_data['Model'].unique())

    # Fuller titles for figures/tables, one per model across every dataset
    # profile; models not listed fall back to their raw id. ACD-* are the
    # debiased counterparts of the same-numbered AC-* nets.
    model_display_names = {
        'AC-1': 'Adult Census - Small Model',
        'AC-3': 'Adult Census - Medium Model',
        'AC-4': 'Adult Census - Large Model',
        'ACD-1': 'Adult Census (Debiased) - Small Model',
        'ACD-3': 'Adult Census (Debiased) - Medium Model',
        'ACD-4': 'Adult Census (Debiased) - Large Model',
        'GC-1': 'German Credit - Model 1',
        'GC-2': 'German Credit - Model 2',
        'GC-3': 'German Credit - Model 3',
        'BM-5': 'Bank Marketing - Model 5',
        'BM-6': 'Bank Marketing - Model 6',
        'BM-7': 'Bank Marketing - Model 7',
        'FT-1': 'Folktables Income - Model 1',
        'FT-2': 'Folktables Income - Model 2',
        'FT-3': 'Folktables Income - Model 3',
    }

    ## LaTeX Table 1: Counterfactual Fairness (with timing)
    print(" ")
    print("======= COUNTERFACTUAL FAIRNESS (epsilon = 0) ==========")

    latex_cf_filename = results_dir / 'counterfactual_table.tex'
    with open(latex_cf_filename, 'w', encoding='utf-8') as f:
        f.write(r'\begin{table}[ht]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Counterfactual Fairness Verification Results ($\epsilon = 0$)}' + '\n')
        f.write(r'\label{tab:counterfactual_fairness}' + '\n')
        f.write(r'\begin{tabular}{lccc}' + '\n')
        f.write(r'\toprule' + '\n')
        f.write(r'Model & Fair (\%) & Unfair (\%) & Avg. Time (s) \\' + '\n')
        f.write(r'\midrule' + '\n')

        for _, row in counterfactual_data.iterrows():
            model_name = row['Model']
            display_name = model_display_names.get(model_name, model_name)
            timing_idx = (timing_data['Model'] == model_name) & (timing_data['Epsilon'] == 0)
            if timing_idx.any():
                avg_time = timing_data.loc[timing_idx, 'AvgTimePerSample'].iloc[0]
            else:
                avg_time = float('nan')

            f.write(rf'{display_name} & {row["FairPercent"]:.1f} & '
                    rf'{row["UnfairPercent"]:.1f} & {avg_time:.4f} \\' + '\n')

        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')

    # Readable table to the console (the .tex file holds the LaTeX version)
    cf_display_rows = []
    for _, row in counterfactual_data.iterrows():
        model_name = row['Model']
        display_name = model_display_names.get(model_name, model_name)
        timing_idx = (timing_data['Model'] == model_name) & (timing_data['Epsilon'] == 0)
        avg_time = (timing_data.loc[timing_idx, 'AvgTimePerSample'].iloc[0]
                    if timing_idx.any() else float('nan'))
        cf_display_rows.append([display_name, f"{row['FairPercent']:.1f}",
                                f"{row['UnfairPercent']:.1f}", f"{avg_time:.4f}"])
    _print_table(['Model', 'Fair %', 'Unfair %', 'Avg Time (s)'], cf_display_rows)
    print(" ")
    print(f"Saved: {latex_cf_filename}")

    ## Figure: Combined Individual Fairness (stacked area plot)
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5),
                             num='Individual Fairness - All Models')
    # Keep axes indexable even when there is a single model
    if n_models == 1:
        axes = [axes]

    for m, model_name in enumerate(models):
        ax = axes[m]

        model_data = individual_data[individual_data['Model'] == model_name]
        model_data = model_data.sort_values('Epsilon')

        epsilons = model_data['Epsilon'].values
        fair_pct = model_data['FairPercent'].values
        unfair_pct = model_data['UnfairPercent'].values

        # Position points at their true epsilon values (proportional x-axis),
        # so horizontal gaps reflect the real perturbation-size differences.
        x = np.asarray(epsilons, dtype=float)

        # Stacked area layers, bottom to top: fair, then unfair. Any unknown
        # share (post-timeout samples) is the unfilled gap up to 100.
        y1 = fair_pct
        y2 = y1 + unfair_pct

        ax.fill_between(x, 0, y1, color=color_fair, alpha=0.9, edgecolor='none',
                        label='Fair')
        ax.fill_between(x, y1, y2, color=color_unfair, alpha=0.9, edgecolor='none',
                        label='Unfair')
        ax.plot(x, y1, 'w-', linewidth=1.5)  # white edge between the areas

        ax.set_xlabel(r'Perturbation Level ($\epsilon$)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
        display_title = model_display_names.get(model_name, model_name)
        ax.set_title(display_title, fontweight='bold', fontsize=14)

        ax.set_xticks(x)
        eps_labels = [f'{e:.2f}' for e in epsilons]
        ax.set_xticklabels(eps_labels)

        # Small proportional padding so end points aren't on the spines
        ax.set_ylim(0, 100)
        pad = (x.max() - x.min()) * 0.03 if x.max() > x.min() else 0.01
        ax.set_xlim(x.min() - pad, x.max() + pad)

        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(False)  # grid on top

        if m == n_models - 1:  # legend only on the last subplot
            ax.legend(loc='upper right', prop={'weight': 'bold'}, frameon=True)

    fig.tight_layout()
    fig.patch.set_facecolor('white')

    if config['save_png']:
        fig.savefig(results_dir / 'individual_fairness_combined.png',
                    dpi=300, facecolor='white')
    if config['save_pdf']:
        fig.savefig(results_dir / 'individual_fairness_combined.pdf',
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved: individual_fairness_combined.png/pdf")

    ## LaTeX Table 2: Individual Fairness Timing (epsilon as columns, models as rows)
    print(" ")
    print("======= INDIVIDUAL FAIRNESS TIMING (seconds per sample) ==========")

    latex_timing_filename = results_dir / 'timing_table.tex'
    with open(latex_timing_filename, 'w', encoding='utf-8') as file:
        individual_timing = timing_data[timing_data['Epsilon'] > 0]
        epsilons_unique = np.unique(individual_timing['Epsilon'].values)
        n_eps = len(epsilons_unique)

        file.write(r'\begin{table}[ht]' + '\n')
        file.write(r'\centering' + '\n')
        file.write(r'\caption{Individual Fairness Verification Timing (seconds per sample)}' + '\n')
        file.write(r'\label{tab:individual_timing}' + '\n')

        # Column format: l for model, then a padded c per epsilon
        col_format = 'l'
        for _ in range(n_eps):
            col_format += r'@{\hskip 8pt}c'
        file.write(r'\begin{tabular}{' + col_format + '}' + '\n')
        file.write(r'\toprule' + '\n')

        # Two-row header: epsilon-spanning label, then the epsilon values
        file.write(rf' & \multicolumn{{{n_eps}}}{{c}}{{Perturbation Level ($\epsilon$)}} \\' + '\n')
        file.write(rf'\cmidrule(l){{2-{n_eps + 1}}}' + '\n')
        header = 'Model'
        for eps_val in epsilons_unique:
            header += f' & {eps_val:.2f}'
        file.write(header + r' \\' + '\n')
        file.write(r'\midrule' + '\n')

        for model_name in models:
            display_name = model_display_names.get(model_name, model_name)
            line = display_name
            for eps_val in epsilons_unique:
                idx = ((individual_timing['Model'] == model_name)
                       & (individual_timing['Epsilon'] == eps_val))
                if idx.any():
                    avg_time = individual_timing.loc[idx, 'AvgTimePerSample'].iloc[0]
                    line += f' & {avg_time:.4f}'
                else:
                    line += ' & --'
            file.write(line + r' \\' + '\n')

        file.write(r'\bottomrule' + '\n')
        file.write(r'\end{tabular}' + '\n')
        file.write(r'\end{table}' + '\n')

    # Readable table to the console (the .tex file holds the LaTeX version)
    eps_headers = [f'{e:.2f}' for e in epsilons_unique]
    timing_display_rows = []
    for model_name in models:
        display_name = model_display_names.get(model_name, model_name)
        cells = [display_name]
        for eps_val in epsilons_unique:
            idx = ((individual_timing['Model'] == model_name)
                   & (individual_timing['Epsilon'] == eps_val))
            if idx.any():
                cells.append(f"{individual_timing.loc[idx, 'AvgTimePerSample'].iloc[0]:.4f}")
            else:
                cells.append('--')
        timing_display_rows.append(cells)
    _print_table(['Model'] + eps_headers, timing_display_rows)
    print(" ")
    print(f"Saved: {latex_timing_filename}")

    print(" ")
    print("======= FairN2V PLOTTING COMPLETE ==========")
    print(f"Generated outputs in {results_dir}:")
    print("  1. counterfactual_table.tex (LaTeX table)")
    print("  2. individual_fairness_combined.png/pdf (Area plot)")
    print("  3. timing_table.tex (LaTeX table)")


if __name__ == "__main__":
    main()
