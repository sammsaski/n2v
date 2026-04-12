"""
Experiment 1: Conservatism Demonstration (Hero Figure)

Produces a figure showing a 2D banana-shaped output distribution with
four reach set boundaries overlaid:
  - Hyperrectangle (red dashed)
  - Ellipsoid (blue dashed)
  - L2 Ball (orange dotted)
  - Flow-based (green solid)

All four sets are calibrated to the same coverage level via conformal
inference, demonstrating that the flow-based set hugs the banana shape
while the hyperrectangle wastes area on empty corners.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
sys.path.insert(0, project_root)
# Add examples/ to path for network imports
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


def run():
    torch.manual_seed(42)

    # --- 1. Create network and sample datasets ---
    print("Creating RotatedBananaNet...")
    net = RotatedBananaNet()

    n_train = 10_000
    n_calib = 8_000
    n_test = 100_000

    print(f"Sampling outputs: {n_train} train, {n_calib} calib, {n_test} test")
    with torch.no_grad():
        x_train = torch.rand(n_train, 2)
        y_train = net(x_train)

        x_calib = torch.rand(n_calib, 2)
        y_calib = net(x_calib)

        x_test = torch.rand(n_test, 2)
        y_test = net(x_test)

    # --- 2. Build score functions ---
    center = y_train.mean(dim=0)

    # Hyperrectangle
    scales = (y_train - center).abs().max(dim=0).values
    scales = scales.clamp(min=1e-8)
    hr_score = HyperrectScore(center, scales)

    # Ellipsoid
    cov = torch.cov(y_train.T)
    cov_inv = torch.linalg.inv(cov)
    ell_score = EllipsoidScore(center, cov_inv)

    # Ball
    ball_score = BallScore(center)

    # Flow — train on centered data
    print("Training flow matching model...")
    y_train_centered = y_train - center
    vf = VelocityField(dim=2, hidden=128, n_layers=4)
    flow_ode = FlowODE(vf)
    train_flow(vf, y_train_centered, n_epochs=500, batch_size=256, lr=1e-3,
               coupling='hungarian')
    print("Flow training complete.")

    # FlowScore operates on centered data
    flow_score_raw = FlowScore(flow_ode, t=1.0)

    # Wrapper that centers before scoring
    class CenteredFlowScore:
        def __init__(self, flow_score, center):
            self.flow_score = flow_score
            self.center = center

        def __call__(self, y):
            return self.flow_score(y - self.center)

    flow_score = CenteredFlowScore(flow_score_raw, center)

    # --- 3. Calibrate all scores ---
    ell = n_calib - 1  # rank for ~99.9% coverage
    epsilon = 0.001

    print(f"Calibrating with m={n_calib}, ell={ell}, epsilon={epsilon}")

    hr_threshold = calibrate(hr_score(y_calib), ell)
    ell_threshold = calibrate(ell_score(y_calib), ell)
    ball_threshold = calibrate(ball_score(y_calib), ell)
    flow_threshold = calibrate(flow_score(y_calib), ell)

    # --- 4. Build ProbabilisticSets ---
    sets = {
        'Hyperrectangle': ProbabilisticSet(
            hr_score, hr_threshold.item(), n_calib, ell, epsilon, dim=2
        ),
        'Ellipsoid': ProbabilisticSet(
            ell_score, ell_threshold.item(), n_calib, ell, epsilon, dim=2
        ),
        'L2 Ball': ProbabilisticSet(
            ball_score, ball_threshold.item(), n_calib, ell, epsilon, dim=2
        ),
        'Flow': ProbabilisticSet(
            flow_score, flow_threshold.item(), n_calib, ell, epsilon, dim=2
        ),
    }

    # --- 5. Compute volumes and coverage ---
    y_test_np = y_test.numpy()
    pad = 0.5
    bbox = (
        torch.tensor([y_test_np[:, 0].min() - pad,
                       y_test_np[:, 1].min() - pad]),
        torch.tensor([y_test_np[:, 0].max() + pad,
                       y_test_np[:, 1].max() + pad]),
    )

    print("\n--- Results ---")
    print(f"{'Set':<20} {'Volume':>10} {'Coverage':>10}")
    print("-" * 45)

    volumes = {}
    for name, pset in sets.items():
        vol, vol_se = pset.estimate_volume(n_samples=1_000_000,
                                           bounding_box=bbox)
        cov = pset.contains(y_test).float().mean().item()
        volumes[name] = vol
        print(f"{name:<20} {vol:>10.4f} {cov:>10.4f}")

    # Print volume ratios
    if 'Flow' in volumes:
        flow_vol = volumes['Flow']
        print(f"\nVolume ratios relative to Flow set:")
        for name, vol in volumes.items():
            if name != 'Flow':
                print(f"  {name}: {vol / flow_vol:.2f}x")

    # --- 6. Plot hero figure ---
    print("\nGenerating hero figure...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Scatter test outputs (subsample for readability)
    ax.scatter(y_test_np[::10, 0], y_test_np[::10, 1],
               s=0.5, c='gray', alpha=0.3, label='Test outputs')

    # Plot boundaries
    styles = {
        'Hyperrectangle': {'color': 'red', 'linestyle': '--', 'linewidth': 2},
        'Ellipsoid': {'color': 'blue', 'linestyle': '--', 'linewidth': 2},
        'L2 Ball': {'color': 'orange', 'linestyle': ':', 'linewidth': 2},
        'Flow': {'color': 'green', 'linestyle': '-', 'linewidth': 2.5},
    }

    for name, pset in sets.items():
        try:
            contours = pset.boundary_2d(resolution=300, bounds=bbox)
            style = styles[name]
            for i, path in enumerate(contours):
                ax.plot(path[:, 0], path[:, 1],
                        label=name if i == 0 else None, **style)
        except Exception as e:
            print(f"  Warning: could not plot {name} boundary: {e}")

    ax.set_xlabel('$y_1$')
    ax.set_ylabel('$y_2$')
    ax.set_title('Probabilistic Reach Sets: Hyperrectangle vs Flow-Based')
    ax.legend(loc='upper left')

    coverage_str = f"Coverage: {1 - epsilon:.1%}"
    guarantee = sets['Flow'].get_guarantee()
    conf_str = f"Confidence: {guarantee[1]:.4f}"
    ax.text(0.98, 0.02, f"{coverage_str}\n{conf_str}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat',
                                   alpha=0.5))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'exp1_conservatism.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    run()
