"""
Plot FairN2V Results
Generates figures and LaTeX tables from verification CSV results.
Outputs:
    (1) LaTeX table for counterfactual fairness
    (2) Combined individual fairness area plot (smooth lines with filled regions)
    (3) LaTeX table for timing results (separated by fairness type)

This script can be run standalone or called from run_fairn2v.py.
Standalone: looks in the most recent results/<ts>/subdir.
Runner-driven: uses `config.output_dir` from caller workspace.
"""

from pathlib import Path

import numpy as np
import pandas as pd
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
    ## Setup
    if config is None:
        # Standalone: pick the most recent results/<ts>/ subdir if present.
        script_dir = Path(__file__).resolve().parent
        results_root = script_dir / 'results'
        subdirs = [d for d in results_root.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(
                f"No results subdir found under {results_root}. Run adult_verify.py first.")
        config = {
            'output_dir': max(subdirs, key=lambda d: d.stat().st_mtime),
            'save_png': True,
            'save_pdf': True,
        }

    results_dir = config['output_dir']

    # Find the CSV files in the results directory
    counterfactual_files = list(results_dir.glob('counterfactual_*.csv'))
    individual_files = list(results_dir.glob('individual_*.csv'))
    timing_files = list(results_dir.glob('timing_*.csv'))

    # Check if files exist
    if not counterfactual_files or not individual_files or not timing_files:
        raise FileNotFoundError(
            f"CSV files not found in {results_dir}. Please run adult_verify.py first.")

    # Get the most recent file of each family (sorted by modification time)
    csv_counterfactual = max(counterfactual_files, key=lambda p: p.stat().st_mtime)
    csv_individual = max(individual_files, key=lambda p: p.stat().st_mtime)
    csv_timing = max(timing_files, key=lambda p: p.stat().st_mtime)

    print("Loading results from:")
    print(f"  {csv_counterfactual}")
    print(f"  {csv_individual}")
    print(f"  {csv_timing}")

    ## Load CSV Data
    # Counterfactual fairness data
    counterfactual_data = pd.read_csv(csv_counterfactual)

    # Individual fairness data
    individual_data = pd.read_csv(csv_individual)

    # Timing data
    timing_data = pd.read_csv(csv_timing)

    ## Figure Settings
    # Professional color scheme
    color_fair = '#2ecc71'
    color_unfair = '#e74c3c'

    # Get unique models
    models = sorted(individual_data['Model'].unique())

    # Model display names (fuller titles for figures/tables)
    # Based on actual architectures:
    # AC-1: 13→16→8→2 (~350 params) - Small
    # AC-3: 13→50→2 (~750 params) - Medium
    model_display_names = {
        'AC-1': 'Adult Census - Small Model',
        'AC-3': 'Adult Census - Medium Model',
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
            # Use display name if available
            display_name = model_display_names.get(model_name, model_name)
            # Find corresponding timing (epsilon = 0)
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

    ## Figure: Combined Individual Fairness (Stacked Area Plot - Professional Style)
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5),
                             num='Individual Fairness - All Models')
    # Keep axes indexable even when there is a single model
    if n_models == 1:
        axes = [axes]

    for m, model_name in enumerate(models):
        ax = axes[m]

        # Filter data for this model
        model_data = individual_data[individual_data['Model'] == model_name]

        # Sort by epsilon
        model_data = model_data.sort_values('Epsilon')

        # Extract values
        epsilons = model_data['Epsilon'].values
        fair_pct = model_data['FairPercent'].values
        unfair_pct = model_data['UnfairPercent'].values
        n_eps = len(epsilons)

        # Use discrete x positions
        x = np.arange(n_eps)

        # Stacked area layers (bottom to top: fair, then unfair)
        y1 = fair_pct         # Bottom layer: Fair
        y2 = y1 + unfair_pct  # Top layer: Unfair (should sum to 100)

        # Fill area for Fair (bottom, from 0 to y1)
        ax.fill_between(x, 0, y1, color=color_fair, alpha=0.9, edgecolor='none',
                        label='Fair')

        # Fill area for Unfair (top, from y1 to y2)
        ax.fill_between(x, y1, y2, color=color_unfair, alpha=0.9, edgecolor='none',
                        label='Unfair')

        # Add white edge line between areas for clarity
        ax.plot(x, y1, 'w-', linewidth=1.5)

        # Labels and formatting with bold fonts and larger size
        ax.set_xlabel(r'Perturbation Level ($\epsilon$)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)

        # Use full display name for title
        display_title = model_display_names.get(model_name, model_name)
        ax.set_title(display_title, fontweight='bold', fontsize=14)

        # Set x-axis with epsilon labels
        ax.set_xticks(x)
        eps_labels = [f'{e:.2f}' for e in epsilons]
        ax.set_xticklabels(eps_labels)

        # Axis limits
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.3, n_eps - 0.7)

        # Professional grid styling
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(False)  # Grid on top

        # Add legend only to the last subplot
        if m == n_models - 1:
            # Labels come from the fills (Fair drawn first, then Unfair)
            ax.legend(loc='upper right', prop={'weight': 'bold'}, frameon=True)

    # Adjust layout
    fig.tight_layout()
    fig.patch.set_facecolor('white')

    # Save combined figure with high resolution
    if config['save_png']:
        fig.savefig(results_dir / 'individual_fairness_combined.png',
                    dpi=300, facecolor='white')
    if config['save_pdf']:
        fig.savefig(results_dir / 'individual_fairness_combined.pdf',
                    bbox_inches='tight', facecolor='white')
    print("Saved: individual_fairness_combined.png/pdf")
    
    ## LaTeX Table 2: Individual Fairness Timing (Horizontal Layout)
    # Epsilon values as columns, models as rows
    print(" ")
    print("======= INDIVIDUAL FAIRNESS TIMING (seconds per sample) ==========")

    latex_timing_filename = results_dir / 'timing_table.tex'
    with open(latex_timing_filename, 'w', encoding='utf-8') as file:
        # Get unique epsilon values for individual fairness (epsilon > 0)
        individual_timing = timing_data[timing_data['Epsilon'] > 0]
        epsilons_unique = np.unique(individual_timing['Epsilon'].values)
        n_eps = len(epsilons_unique)

        file.write(r'\begin{table}[ht]' + '\n')
        file.write(r'\centering' + '\n')
        file.write(r'\caption{Individual Fairness Verification Timing (seconds per sample)}' + '\n')
        file.write(r'\label{tab:individual_timing}' + '\n')

        # Create column format with spacing: l for model, then c with padding for each epsilon
        col_format = 'l'
        for _ in range(n_eps):
            col_format += r'@{\hskip 8pt}c'
        file.write(r'\begin{tabular}{' + col_format + '}' + '\n')
        file.write(r'\toprule' + '\n')

        # Two-row header: first row spans epsilon columns with label
        file.write(rf' & \multicolumn{{{n_eps}}}{{c}}{{Perturbation Level ($\epsilon$)}} \\' + '\n')
        file.write(rf'\cmidrule(l){{2-{n_eps + 1}}}' + '\n')

        # Second header row with just epsilon values
        header = 'Model'
        for eps_val in epsilons_unique:
            header += f' & {eps_val:.2f}'
        file.write(header + r' \\' + '\n')
        file.write(r'\midrule' + '\n')

        # Data rows (one per model) with full display names
        for model_name in models:
            display_name = model_display_names.get(model_name, model_name)
            line = display_name

            for eps_val in epsilons_unique:
                # Find timing for this model and epsilon
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

    ## Summary
    print(" ")
    print("======= FairNNV PLOTTING COMPLETE ==========")
    print(f"Generated outputs in {results_dir}:")
    print("  1. counterfactual_table.tex (LaTeX table)")
    print("  2. individual_fairness_combined.png/pdf (Area plot)")
    print("  3. timing_table.tex (LaTeX table)")


if __name__ == "__main__":
    main()