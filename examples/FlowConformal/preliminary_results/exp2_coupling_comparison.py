# examples/FlowConformal/exp2_coupling_comparison.py
"""
Experiment 2: Coupling Method Comparison

Compares Hungarian vs Sinkhorn OT coupling on CPU vs GPU:
  - Training wall-clock time
  - Flow quality (set boundary visualization)
  - Volume of flow-based reach set

Produces a 2x2 figure where each subfigure is the hero figure
for one (coupling, device) configuration.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'examples'))

from n2v.probabilistic.flow import (
    HyperrectScore,
    EllipsoidScore,
    BallScore,
    FlowScore,
    VelocityField,
    FlowODE,
    train_flow,
    calibrate,
    ProbabilisticSet,
)
from FlowConformal.networks import RotatedBananaNet


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'outputs', 'preliminary_results'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIGS = [
    ('Hungarian', 'CPU', 'hungarian', 'cpu'),
    ('Hungarian', 'GPU', 'hungarian', 'cuda'),
    ('Sinkhorn', 'CPU', 'sinkhorn', 'cpu'),
    ('Sinkhorn', 'GPU', 'sinkhorn', 'cuda'),
]


def train_and_evaluate(y_train, y_calib, y_test, coupling, device_str,
                       n_epochs=500, batch_size=256, lr=1e-3):
    """
    Train a flow model and evaluate it.

    Returns dict with: train_time, sets, volumes, coverages, bbox.
    """
    device = torch.device(device_str)
    center = y_train.mean(dim=0)

    # Move centered training data to device
    y_centered = (y_train - center).to(device)

    # Train flow
    vf = VelocityField(dim=2, hidden=128, n_layers=4).to(device)

    start = time.time()
    train_flow(vf, y_centered, n_epochs=n_epochs, batch_size=batch_size,
               lr=lr, coupling=coupling)
    train_time = time.time() - start

    # Move model back to CPU for scoring
    vf = vf.cpu()
    flow_ode = FlowODE(vf)

    flow_score_raw = FlowScore(flow_ode, t=1.0)

    class CenteredFlowScore:
        def __init__(self, flow_score, center):
            self.flow_score = flow_score
            self.center = center

        def __call__(self, y):
            return self.flow_score(y - self.center)

    flow_score = CenteredFlowScore(flow_score_raw, center)

    # Non-flow scores
    scales = (y_train - center).abs().max(dim=0).values.clamp(min=1e-8)
    hr_score = HyperrectScore(center, scales)

    cov = torch.cov(y_train.T)
    cov_inv = torch.linalg.inv(cov)
    ell_score = EllipsoidScore(center, cov_inv)

    ball_score = BallScore(center)

    # Calibrate
    ell = y_calib.shape[0] - 1
    epsilon = 0.001

    hr_threshold = calibrate(hr_score(y_calib), ell)
    ell_threshold = calibrate(ell_score(y_calib), ell)
    ball_threshold = calibrate(ball_score(y_calib), ell)
    flow_threshold = calibrate(flow_score(y_calib), ell)

    # Build sets
    sets = {
        'Hyperrectangle': ProbabilisticSet(
            hr_score, hr_threshold.item(), y_calib.shape[0], ell,
            epsilon, dim=2,
        ),
        'Ellipsoid': ProbabilisticSet(
            ell_score, ell_threshold.item(), y_calib.shape[0], ell,
            epsilon, dim=2,
        ),
        'L2 Ball': ProbabilisticSet(
            ball_score, ball_threshold.item(), y_calib.shape[0], ell,
            epsilon, dim=2,
        ),
        'Flow': ProbabilisticSet(
            flow_score, flow_threshold.item(), y_calib.shape[0], ell,
            epsilon, dim=2,
        ),
    }

    # Compute volumes and coverage
    y_test_np = y_test.numpy()
    pad = 0.5
    bbox = (
        torch.tensor([y_test_np[:, 0].min() - pad,
                       y_test_np[:, 1].min() - pad]),
        torch.tensor([y_test_np[:, 0].max() + pad,
                       y_test_np[:, 1].max() + pad]),
    )

    volumes = {}
    coverages = {}
    for name, pset in sets.items():
        vol, _ = pset.estimate_volume(n_samples=500_000, bounding_box=bbox)
        cov = pset.contains(y_test).float().mean().item()
        volumes[name] = vol
        coverages[name] = cov

    return {
        'train_time': train_time,
        'sets': sets,
        'volumes': volumes,
        'coverages': coverages,
        'bbox': bbox,
    }


def plot_subfigure(ax, y_test_np, result, title):
    """Plot one hero figure on the given axes."""
    ax.scatter(y_test_np[::10, 0], y_test_np[::10, 1],
               s=0.5, c='gray', alpha=0.3)

    styles = {
        'Hyperrectangle': {'color': 'red', 'linestyle': '--', 'linewidth': 1.5},
        'Ellipsoid': {'color': 'blue', 'linestyle': '--', 'linewidth': 1.5},
        'L2 Ball': {'color': 'orange', 'linestyle': ':', 'linewidth': 1.5},
        'Flow': {'color': 'green', 'linestyle': '-', 'linewidth': 2},
    }

    for name, pset in result['sets'].items():
        try:
            contours = pset.boundary_2d(resolution=200, bounds=result['bbox'])
            style = styles[name]
            for i, path in enumerate(contours):
                ax.plot(path[:, 0], path[:, 1],
                        label=name if i == 0 else None, **style)
        except Exception as e:
            print(f"  Warning: {name} boundary failed: {e}")

    flow_vol = result['volumes']['Flow']
    train_time = result['train_time']
    ax.set_title(f"{title}\nVol={flow_vol:.3f}, Time={train_time:.1f}s",
                 fontsize=10)
    ax.set_xlabel('$y_1$', fontsize=9)
    ax.set_ylabel('$y_2$', fontsize=9)


def run():
    torch.manual_seed(42)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. GPU configs will be skipped.")

    # --- Create network and datasets ---
    print("Creating RotatedBananaNet...")
    net = RotatedBananaNet()

    n_train = 10_000
    n_calib = 8_000
    n_test = 100_000

    print(f"Sampling: {n_train} train, {n_calib} calib, {n_test} test")
    with torch.no_grad():
        y_train = net(torch.rand(n_train, 2))
        y_calib = net(torch.rand(n_calib, 2))
        y_test = net(torch.rand(n_test, 2))

    y_test_np = y_test.numpy()

    # --- Run configurations ---
    results = {}
    for label, device_label, coupling, device_str in CONFIGS:
        config_name = f"{label}/{device_label}"

        if device_str == 'cuda' and not torch.cuda.is_available():
            print(f"\nSkipping {config_name} (no CUDA)")
            continue

        print(f"\n{'='*50}")
        print(f"Training: {config_name}")
        print(f"{'='*50}")

        result = train_and_evaluate(
            y_train, y_calib, y_test,
            coupling=coupling, device_str=device_str,
        )
        results[config_name] = result

        print(f"  Time: {result['train_time']:.1f}s")
        print(f"  Flow volume: {result['volumes']['Flow']:.4f}")
        print(f"  Flow coverage: {result['coverages']['Flow']:.4f}")

    # --- Timing table ---
    print(f"\n{'='*60}")
    print(f"{'Config':<25} {'Time (s)':>10} {'Flow Vol':>10} {'Coverage':>10}")
    print(f"{'-'*60}")
    for name, r in results.items():
        print(f"{name:<25} {r['train_time']:>10.1f} "
              f"{r['volumes']['Flow']:>10.4f} "
              f"{r['coverages']['Flow']:>10.4f}")

    # --- 2x2 figure ---
    print("\nGenerating comparison figure...")
    n_configs = len(results)
    ncols = 2
    nrows = (n_configs + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for idx, (config_name, result) in enumerate(results.items()):
        plot_subfigure(axes[idx], y_test_np, result, config_name)

    # Add legend to first subplot only
    if len(results) > 0:
        axes[0].legend(loc='upper left', fontsize=7)

    # Hide unused axes
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        'Coupling Comparison: Hungarian vs Sinkhorn, CPU vs GPU',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'exp2_coupling_comparison.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    run()
