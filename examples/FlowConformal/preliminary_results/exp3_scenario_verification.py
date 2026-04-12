"""
Experiment 3: Scenario-Based Verification on the Banana

Demonstrates the scenario optimization verification primitive on the
banana example, with preimage search for counterexample certification.

Two queries:
  1. A loose halfspace spec — should verify with a probabilistic certificate.
  2. A tight halfspace spec — the flow set contains points violating the
     spec, but these are flow hallucinations (not real network outputs).
     Preimage search distinguishes genuine counterexamples from hallucinations,
     producing an 'unknown' or 'falsified' outcome.

The experiment demonstrates the three-state outcome:
  - verified: joint probabilistic certificate holds
  - falsified: genuine counterexample found (real input x in I)
  - unknown: flow-set violation found but no real preimage (hallucination)
"""

import os
import sys
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
    VelocityField, FlowODE, train_flow, calibrate, compute_guarantee,
    FlowScore, scenario_verify_halfspace,
)
from FlowConformal.networks import RotatedBananaNet


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'outputs', 'preliminary_results'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run():
    torch.manual_seed(42)

    # --- 1. Setup: train flow + conformal calibration ---
    print("Creating RotatedBananaNet...")
    net = RotatedBananaNet()

    n_train = 10_000
    n_calib = 8_000

    print(f"Sampling: {n_train} train, {n_calib} calib")
    with torch.no_grad():
        y_train = net(torch.rand(n_train, 2))
        y_calib = net(torch.rand(n_calib, 2))

    center = y_train.mean(dim=0)
    y_train_centered = y_train - center

    print("Training flow matching model (SiLU)...")
    vf = VelocityField(dim=2, hidden=128, n_layers=4)
    flow_ode = FlowODE(vf)
    train_flow(vf, y_train_centered, n_epochs=500, batch_size=256, lr=1e-3,
               coupling='sinkhorn')
    print("Flow training complete.")

    # Calibrate
    flow_score = FlowScore(flow_ode, t=1.0)
    with torch.no_grad():
        scores = flow_score(y_calib - center)
    ell = n_calib - 1
    epsilon_1 = 0.001
    threshold_q = calibrate(scores, ell).item()
    _, delta_1 = compute_guarantee(m=n_calib, ell=ell, epsilon=epsilon_1)

    print(f"\nConformal calibration:")
    print(f"  threshold q = {threshold_q:.4f}")
    print(f"  epsilon_1 = {epsilon_1}")
    print(f"  delta_1 = {delta_1:.4f}")

    # --- 2. Verification: loose spec (should verify) ---
    # The flow operates on centered data. To check "y_orig[0] >= -10"
    # rewrite in centered coordinates: y_centered[0] >= -10 - center[0]
    # As a halfspace (w^T y_centered <= b): w = [-1, 0], b = 10 + center[0]
    print("\n--- Loose spec: original y[0] >= -10 ---")
    w_loose = np.array([-1.0, 0.0])
    b_loose = 10.0 + center[0].item()

    result_loose = scenario_verify_halfspace(
        flow_ode=flow_ode, threshold_q=threshold_q,
        w=w_loose, b=b_loose,
        n_samples=10_000, beta_2=0.001, t=1.0,
    )

    print(f"  outcome = {result_loose.outcome}")
    print(f"  epsilon_2 = {result_loose.epsilon_2:.6f}")
    print(f"  delta_2 = {result_loose.delta_2:.4f}")
    if result_loose.outcome == 'verified':
        eps_total = 1 - (1 - epsilon_1) * (1 - result_loose.epsilon_2)
        delta_total = delta_1 * result_loose.delta_2
        print(f"  joint epsilon_total = {eps_total:.6f}")
        print(f"  joint delta_total = {delta_total:.4f}")
        print(f"  -> with {delta_total:.1%} confidence, the network "
              f"satisfies y[0] >= -10 with prob >= {1 - eps_total:.4%}")

    # --- 3. Verification: tight spec with preimage search ---
    print("\n--- Tight spec: original y[0] >= 0.7 (with preimage search) ---")
    w_tight = np.array([-1.0, 0.0])
    b_tight = -0.7 + center[0].item()

    # The flow operates on centered outputs, so target_fn must also
    # produce centered outputs. Wrap the banana network with centering.
    def target_fn_centered(x):
        return net(x) - center

    result_tight = scenario_verify_halfspace(
        flow_ode=flow_ode, threshold_q=threshold_q,
        w=w_tight, b=b_tight,
        n_samples=10_000, beta_2=0.001, t=1.0,
        target_fn=target_fn_centered,
        input_set_bounds=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        preimage_n_restarts=10,
        preimage_n_steps=300,
        preimage_tolerance=0.05,
    )

    print(f"  outcome = {result_tight.outcome}")
    if result_tight.outcome == 'falsified':
        x_ce = result_tight.genuine_input
        y_real = net(torch.tensor(x_ce[None], dtype=torch.float32)).detach().numpy().squeeze()
        print(f"  GENUINE COUNTEREXAMPLE")
        print(f"  real input x = {x_ce}")
        print(f"  real output y = f(x) = {y_real}")
        print(f"  -> actual network produces y[0] = {y_real[0]:.4f} < 0.7 — spec violated.")
    elif result_tight.outcome == 'unknown':
        z_ce, y_ce, margin = result_tight.counterexample
        y_ce_orig = y_ce + center.numpy()
        print(f"  UNKNOWN (flow hallucination)")
        print(f"  flow set contains y = {y_ce_orig} with margin {margin:.4f},")
        print(f"  but no real input in [0,1]^2 maps to this point.")
        print(f"  The spec is on the boundary of what the flow set can certify.")

    # --- 4. Plot ---
    print("\nGenerating figure...")
    with torch.no_grad():
        y_test = net(torch.rand(50_000, 2))
    y_test_np = y_test.numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(y_test_np[:, 0], y_test_np[:, 1],
               s=0.5, c='gray', alpha=0.3, label='Test outputs')

    # Plot the spec line
    ax.axvline(0.7, color='red', linestyle='--',
               linewidth=1.5, label='Spec: y[0] >= 0.7')

    if result_tight.outcome == 'falsified':
        x_ce = result_tight.genuine_input
        y_real = net(torch.tensor(x_ce[None], dtype=torch.float32)).detach().numpy().squeeze()
        ax.scatter([y_real[0]], [y_real[1]],
                   s=200, marker='X', c='red',
                   edgecolors='black', linewidths=1.5, zorder=10,
                   label='Genuine counterexample')
    elif result_tight.outcome == 'unknown':
        z_ce, y_ce, margin = result_tight.counterexample
        y_ce_orig = y_ce + center.numpy()
        ax.scatter([y_ce_orig[0]], [y_ce_orig[1]],
                   s=200, marker='X', c='orange',
                   edgecolors='black', linewidths=1.5, zorder=10,
                   label='Flow hallucination (unknown)')

    ax.set_xlabel('$y_1$')
    ax.set_ylabel('$y_2$')
    title = f'Scenario Verification — outcome: {result_tight.outcome}'
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'exp3_scenario_verification.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    run()
