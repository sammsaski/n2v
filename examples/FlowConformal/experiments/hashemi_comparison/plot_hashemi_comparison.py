"""
Plot the Hashemi comparison results.

Reads exp_hashemi_comparison.csv and produces:
  1. exp_hashemi_comparison_volume_ratios.png -- boxplots of clip/flow ratio per (network, radius)
  2. exp_hashemi_comparison_scaling.png       -- volume vs radius per network (flow, clip, exact)
  3. exp_hashemi_comparison_verdicts.png      -- verdict cross-tab per network, method vs exact
  4. exp_hashemi_comparison_runtimes.png      -- clip vs flow runtime scatter
  5. exp_hashemi_comparison_hero_banana.png   -- flow + clip + exact overlaid on banana output space
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.collections import PatchCollection

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'examples'))


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUTPUT_DIR, 'exp_hashemi_comparison.csv')


def plot_volume_ratios(df):
    """Boxplots of volume_ratio_clip_over_flow per (network, radius)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    networks = ['RotatedBananaNet', 'ThreeBlobClassifier']

    for ax, network in zip(axes, networks):
        sub = df[df['network'] == network]
        if len(sub) == 0:
            ax.axis('off')
            continue
        radii = sorted(sub['perturbation_radius'].unique())
        data = [sub[sub['perturbation_radius'] == r]['volume_ratio_clip_over_flow'].values
                for r in radii]
        bp = ax.boxplot(data, labels=[f'{r:g}' for r in radii], showfliers=True)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='parity')
        ax.set_yscale('log')
        ax.set_xlabel('perturbation radius')
        ax.set_ylabel('clip volume / flow volume')
        ax.set_title(network)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('Tightness ratio (clip / flow): > 1 means flow is tighter')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'exp_hashemi_comparison_volume_ratios.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path}')


def plot_scaling(df):
    """Volume vs radius per network, one line per method."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    networks = ['RotatedBananaNet', 'ThreeBlobClassifier']

    for ax, network in zip(axes, networks):
        sub = df[df['network'] == network]
        if len(sub) == 0:
            ax.axis('off')
            continue
        radii = sorted(sub['perturbation_radius'].unique())
        flow_means, flow_stds = [], []
        clip_means, clip_stds = [], []
        exact_means = []
        for r in radii:
            r_sub = sub[sub['perturbation_radius'] == r]
            flow_means.append(r_sub['flow_volume'].mean())
            flow_stds.append(r_sub['flow_volume'].std())
            clip_means.append(r_sub['clip_volume'].mean())
            clip_stds.append(r_sub['clip_volume'].std())
            exact_means.append(r_sub['exact_volume'].mean())

        radii_arr = np.array(radii)
        ax.errorbar(radii_arr, flow_means, yerr=flow_stds, marker='o',
                    color='C2', label='flow', capsize=3)
        ax.errorbar(radii_arr, clip_means, yerr=clip_stds, marker='s',
                    color='C3', label='clip', capsize=3)
        if not np.all(np.isnan(exact_means)):
            ax.plot(radii_arr, exact_means, marker='^', color='black',
                    linestyle='--', label='exact', linewidth=1.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('perturbation radius')
        ax.set_ylabel('reach-set volume')
        ax.set_title(network)
        ax.legend(loc='best')
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Volume vs perturbation radius')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'exp_hashemi_comparison_scaling.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path}')


def plot_verdicts(df):
    """Cross-tab: method verdict × exact verdict, one panel per (network, method)."""
    networks = ['RotatedBananaNet', 'ThreeBlobClassifier']
    methods = [('flow', 'flow_verdict'), ('clip', 'clip_verdict')]
    exact_col = 'exact_verdict'

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))

    for row_i, network in enumerate(networks):
        for col_i, (method_name, method_col) in enumerate(methods):
            ax = axes[row_i, col_i]
            sub = df[df['network'] == network]
            if len(sub) == 0:
                ax.axis('off')
                continue
            method_labels = ['verified', 'falsified', 'unknown']
            exact_labels = ['verified', 'falsified', 'unknown']
            mat = np.zeros((3, 3), dtype=int)
            for r_i, r_lbl in enumerate(method_labels):
                for c_i, c_lbl in enumerate(exact_labels):
                    mat[r_i, c_i] = (
                        (sub[method_col] == r_lbl) & (sub[exact_col] == c_lbl)
                    ).sum()
            im = ax.imshow(mat, cmap='Blues', aspect='auto')
            for r_i in range(3):
                for c_i in range(3):
                    text = str(mat[r_i, c_i])
                    color = 'white' if mat[r_i, c_i] > mat.max() * 0.5 else 'black'
                    ax.text(c_i, r_i, text, ha='center', va='center',
                            color=color, fontsize=11)
            ax.set_xticks(range(3))
            ax.set_xticklabels(exact_labels)
            ax.set_yticks(range(3))
            ax.set_yticklabels(method_labels)
            ax.set_xlabel('exact ground truth')
            ax.set_ylabel(f'{method_name} verdict')
            ax.set_title(f'{network} — {method_name}')

    fig.suptitle('Verdict cross-tabulation: method vs exact ground truth')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'exp_hashemi_comparison_verdicts.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path}')


def plot_runtimes(df):
    """Scatter of clip_runtime vs flow_runtime."""
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {'RotatedBananaNet': 'C0', 'ThreeBlobClassifier': 'C1'}
    for network, color in colors.items():
        sub = df[df['network'] == network]
        if len(sub) == 0:
            continue
        ax.scatter(sub['clip_runtime'], sub['flow_runtime'],
                   color=color, label=network, alpha=0.7, s=40)
    # y=x reference
    all_rt = np.concatenate([df['clip_runtime'].values, df['flow_runtime'].values])
    mx = all_rt.max() * 1.1
    ax.plot([0, mx], [0, mx], color='gray', linestyle='--',
            linewidth=1, label='y = x')
    ax.set_xlabel('clip runtime (s)')
    ax.set_ylabel('flow runtime (s)')
    ax.set_title('Per-run wall clock: flow vs clip')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'exp_hashemi_comparison_runtimes.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path}')


def plot_hero_banana(df):
    """Hero figure: for one banana seed+radius, plot flow reach set, clip
    reach set, and exact reach set on a single axis.

    Requires re-running the pipeline because the actual reach sets aren't
    stored in the CSV (only summary stats). We pick a middle-radius banana
    config and rebuild the three sets.
    """
    from FlowConformal.networks import RotatedBananaNet
    from FlowConformal.experiments.hashemi_comparison.exp_hashemi_comparison import (
        compute_exact_reach, star_polygon_2d, sample_l_inf_ball,
        CenteredFlowScore, BANANA_SPEC, N_TRAIN, N_CALIB,
        FLOW_HIDDEN, FLOW_N_LAYERS, FLOW_EPOCHS, FLOW_BATCH_SIZE,
        FLOW_LR, FLOW_COUPLING, EPSILON_1,
    )
    from n2v.probabilistic.flow import (
        VelocityField, FlowODE, FlowScore, train_flow, calibrate,
        ProbabilisticSet,
    )
    from n2v.probabilistic import verify as n2v_verify
    from n2v.sets.box import Box

    torch.manual_seed(0)
    banana = RotatedBananaNet()

    # Use test input 1 (middle point), radius 0.2, seed 0
    x_center = torch.tensor([0.5, 0.5], dtype=torch.float32)
    radius = 0.2
    seed = 0

    x_train = sample_l_inf_ball(x_center, radius, N_TRAIN, seed=seed, dim=2)
    x_calib = sample_l_inf_ball(x_center, radius, N_CALIB, seed=seed + 1_000_000, dim=2)
    with torch.no_grad():
        y_train = banana(x_train)
        y_calib = banana(x_calib)

    center = y_train.mean(dim=0)
    y_train_c = y_train - center

    torch.manual_seed(seed)
    vf = VelocityField(dim=2, hidden=FLOW_HIDDEN, n_layers=FLOW_N_LAYERS)
    flow_ode = FlowODE(vf)
    train_flow(vf, y_train_c, n_epochs=FLOW_EPOCHS,
               batch_size=FLOW_BATCH_SIZE, lr=FLOW_LR, coupling=FLOW_COUPLING)

    flow_score = CenteredFlowScore(FlowScore(flow_ode, t=1.0), center)
    with torch.no_grad():
        calib_scores = flow_score(y_calib)
    ell = N_CALIB - 1
    threshold = calibrate(calib_scores, ell).item()

    # Exact
    x_center_np = x_center.numpy()
    reach = compute_exact_reach(banana, x_center_np, radius, output_dim=2)
    stars = reach['stars']
    rng = np.random.default_rng(0)
    exact_polygons = []
    for s in stars:
        p = star_polygon_2d(s, rng)
        if p is not None:
            exact_polygons.append(p)

    # Clip
    lb_in = (x_center_np - radius).reshape(-1, 1)
    ub_in = (x_center_np + radius).reshape(-1, 1)
    input_box = Box(lb_in, ub_in)
    def model_fn(x_np):
        with torch.no_grad():
            return banana(torch.tensor(x_np, dtype=torch.float32)).numpy()
    clip_result = n2v_verify(
        model=model_fn, input_set=input_box,
        m=N_CALIB, ell=ell, epsilon=EPSILON_1,
        surrogate='clipping_block', training_samples=N_TRAIN, seed=seed,
    )
    clip_lb = clip_result.lb.flatten()
    clip_ub = clip_result.ub.flatten()

    # Flow boundary via ProbabilisticSet
    pset = ProbabilisticSet(
        score_fn=flow_score, threshold=threshold,
        m=N_CALIB, ell=ell, epsilon=EPSILON_1, dim=2,
    )

    # Bbox for flow contour
    all_pts = []
    for p in exact_polygons:
        all_pts.append(p)
    all_pts.append(np.array([[clip_lb[0], clip_lb[1]], [clip_ub[0], clip_ub[1]]]))
    all_pts = np.vstack(all_pts)
    lo = all_pts.min(axis=0) - 0.05 * (all_pts.max(axis=0) - all_pts.min(axis=0))
    hi = all_pts.max(axis=0) + 0.05 * (all_pts.max(axis=0) - all_pts.min(axis=0))
    bbox = (
        torch.tensor([lo[0], lo[1]], dtype=torch.float32),
        torch.tensor([hi[0], hi[1]], dtype=torch.float32),
    )
    flow_contours = pset.boundary_2d(resolution=400, bounds=bbox)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    # Exact reach set: filled light blue
    exact_patches = [MplPolygon(p, closed=True) for p in exact_polygons]
    exact_pc = PatchCollection(
        exact_patches,
        facecolor='#6baed6', edgecolor='#6baed6',
        linewidths=0.5, alpha=1.0, antialiased=False,
        label='exact',
    )
    ax.add_collection(exact_pc)

    # Clip reach set: red dashed rectangle
    rect = Rectangle(
        (clip_lb[0], clip_lb[1]),
        clip_ub[0] - clip_lb[0],
        clip_ub[1] - clip_lb[1],
        fill=False, edgecolor='#d62728', linewidth=2.0, linestyle='--',
    )
    ax.add_patch(rect)

    # Flow reach set: green contour
    for seg in flow_contours:
        ax.plot(seg[:, 0], seg[:, 1], color='#2ca02c', linewidth=2.5)

    ax.set_xlim(bbox[0][0].item(), bbox[1][0].item())
    ax.set_ylim(bbox[0][1].item(), bbox[1][1].item())
    ax.set_aspect('equal')
    ax.set_xlabel(r'$y_1$', fontsize=13)
    ax.set_ylabel(r'$y_2$', fontsize=13)
    ax.set_title(
        f'Banana output space (test_input=(0.5,0.5), radius={radius}, seed={seed})'
    )

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor='#6baed6', edgecolor='#6baed6',
              label='exact reach set (n2v Star)'),
        Line2D([0], [0], color='#d62728', linewidth=2.0, linestyle='--',
               label='clip reach set (Hashemi)'),
        Line2D([0], [0], color='#2ca02c', linewidth=2.5,
               label='flow reach set (ours)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'exp_hashemi_comparison_hero_banana.png')
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path}')


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f'Missing {CSV_PATH}. Run exp_hashemi_comparison.py first.'
        )
    df = pd.read_csv(CSV_PATH)
    print(f'Loaded {len(df)} rows from {CSV_PATH}')

    plot_volume_ratios(df)
    plot_scaling(df)
    plot_verdicts(df)
    plot_runtimes(df)
    plot_hero_banana(df)


if __name__ == '__main__':
    main()
