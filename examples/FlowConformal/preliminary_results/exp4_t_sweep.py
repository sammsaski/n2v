"""
Experiment 4: Flow Time (t) Sweep

Trains one flow on the banana network and sweeps the evaluation time
parameter t in [0, 1] to visualize how the reach set evolves. At t=0
the set is an L2 ball around the data center; at t=1 the set follows
the full learned flow geometry.

This is a preliminary diagnostic experiment. It does NOT measure
verification outcomes or hallucination rates — those are for the
ablation study later.

Produces:
  - A two-panel figure:
      Left: scatter of test outputs with 6 overlaid boundaries
            (t in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}) colored by t.
      Right: MC volume vs t curve (11 points).
  - A printed table of (t, threshold, volume, coverage).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'examples'))

from n2v.probabilistic.flow import (
    VelocityField, FlowODE, train_flow, calibrate,
    FlowScore, ProbabilisticSet,
)
from FlowConformal.networks import RotatedBananaNet


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'outputs', 'preliminary_results'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# t values for the sweep.
T_VALUES = [round(0.1 * i, 1) for i in range(11)]

# Subset of t values to plot as overlaid boundaries (keeps the figure
# readable while the volume curve uses all 11).
T_OVERLAY = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


class CenteredFlowScore:
    """Wraps a FlowScore to center inputs before scoring.

    The flow is trained on centered outputs, so every score evaluation
    must center the input by subtracting the data center first.
    """

    def __init__(self, flow_score, center):
        self.flow_score = flow_score
        self.center = center

    def __call__(self, y):
        return self.flow_score(y - self.center)


def run():
    torch.manual_seed(42)

    # --- 1. Setup: banana network + datasets ---
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

    center = y_train.mean(dim=0)
    y_train_centered = y_train - center

    # --- 2. Train flow once ---
    print("Training flow matching model (SiLU)...")
    vf = VelocityField(dim=2, hidden=128, n_layers=4)
    flow_ode = FlowODE(vf)
    train_flow(vf, y_train_centered, n_epochs=500, batch_size=256,
               lr=1e-3, coupling='sinkhorn')
    print("Flow training complete.")

    # --- 3. Sweep t ---
    n_calib_samples = n_calib
    ell = n_calib_samples - 1
    epsilon_1 = 0.001

    # Bounding box for MC volume and boundary extraction
    y_test_np = y_test.numpy()
    pad = 0.5
    bbox = (
        torch.tensor([y_test_np[:, 0].min() - pad,
                      y_test_np[:, 1].min() - pad]),
        torch.tensor([y_test_np[:, 0].max() + pad,
                      y_test_np[:, 1].max() + pad]),
    )

    print(f"\nSweeping t in {T_VALUES}...")
    print(f"{'t':>6} {'threshold':>12} {'volume':>12} {'coverage':>12}")
    print("-" * 44)

    results = []  # list of dicts per t
    for t in T_VALUES:
        # Build the score at this t.
        flow_score_raw = FlowScore(flow_ode, t=t)
        flow_score = CenteredFlowScore(flow_score_raw, center)

        # Calibrate
        with torch.no_grad():
            scores = flow_score(y_calib)
        threshold = calibrate(scores, ell).item()

        # Build ProbabilisticSet
        pset = ProbabilisticSet(
            score_fn=flow_score,
            threshold=threshold,
            m=n_calib_samples, ell=ell, epsilon=epsilon_1,
            dim=2,
        )

        # MC volume estimate
        vol, _ = pset.estimate_volume(n_samples=500_000, bounding_box=bbox)

        # Empirical coverage on test set
        with torch.no_grad():
            coverage = pset.contains(y_test).float().mean().item()

        # Extract boundary (needed only for the t values we overlay)
        boundary = None
        if t in T_OVERLAY:
            try:
                boundary = pset.boundary_2d(resolution=300, bounds=bbox)
            except Exception as e:
                print(f"  Warning: boundary extraction at t={t} failed: {e}")
                boundary = None

        results.append({
            't': t,
            'threshold': threshold,
            'volume': vol,
            'coverage': coverage,
            'boundary': boundary,
        })

        print(f"{t:>6.2f} {threshold:>12.4f} {vol:>12.4f} {coverage:>12.4f}")

    # --- 4. Plot ---
    print("\nGenerating figure...")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: scatter + overlaid boundaries, colored by t
    ax_left.scatter(
        y_test_np[::10, 0], y_test_np[::10, 1],
        s=0.5, c='gray', alpha=0.3, label='Test outputs'
    )

    cmap = plt.get_cmap('viridis')
    for r in results:
        if r['t'] not in T_OVERLAY or r['boundary'] is None:
            continue
        color = cmap(r['t'])
        label = f"t = {r['t']:.1f}"
        contours = r['boundary']
        for i, path in enumerate(contours):
            ax_left.plot(
                path[:, 0], path[:, 1],
                color=color,
                linewidth=1.8,
                label=label if i == 0 else None,
            )

    ax_left.set_xlabel(r'$y_1$')
    ax_left.set_ylabel(r'$y_2$')
    ax_left.set_title('Reach set boundaries as $t$ varies')
    ax_left.legend(loc='upper left', fontsize=8)

    # Right panel: volume curve
    ts = np.array([r['t'] for r in results])
    vols = np.array([r['volume'] for r in results])
    ax_right.plot(ts, vols, 'o-', color='C0', linewidth=2, markersize=7)
    ax_right.set_xlabel(r'flow time $t$')
    ax_right.set_ylabel('MC volume')
    ax_right.set_title('Reach set volume vs $t$')
    ax_right.grid(True, alpha=0.3)

    # Annotate endpoints
    ax_right.annotate(
        'L2 Ball',
        xy=(0.0, vols[0]),
        xytext=(0.1, vols[0] * 1.05),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', lw=0.8),
    )
    ax_right.annotate(
        'Full Flow',
        xy=(1.0, vols[-1]),
        xytext=(0.6, vols[-1] * 1.15 if vols[-1] > 0 else 0.1),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', lw=0.8),
    )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'exp4_t_sweep.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    run()
