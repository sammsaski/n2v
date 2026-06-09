"""Verify the MNIST F2FMLP SNN using n2v.

Demonstrates local robustness verification against L-inf perturbations using
SpikingNeuralNetwork.reach() with both 'approx' and 'exact' methods.

Requires: models/mnist_snn.pt + models/mnist_snn_meta.json  (run train_snn.py first)

Verification task:
  Given a correctly classified test image x and perturbation radius epsilon,
  prove that every image x' with ||x' - x||_inf <= epsilon is classified the
  same way as x.

Approach:
  The full 784-dimensional perturbation box is too large for the 'exact'
  method (exponential in the number of symbolic dimensions). Instead we:
    1. Run 'approx' on the full k_approx-pixel perturbation - fast, may be
       inconclusive if the over-approximation is too loose.
    2. Run 'exact' on a focused k_exact-pixel subset - slower per run but
       sound and complete for those pixels.
  Both calls target the same epsilon; the difference is how many pixels are
  treated as symbolic vs fixed at their nominal value.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from n2v import SpikingNeuralNetwork, Box
from n2v.snn.model import F2FMLP
from n2v.snn.lp import feasible_latencies

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EPSILON   = 0.05   # L-inf perturbation radius
K_APPROX  = 20     # pixels treated as symbolic for the approx run
K_EXACT   = 2      # pixels treated as symbolic for the exact run (tractable)
N_IMAGES  = 5      # number of test images to verify


def load_model(model_path, meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    model = F2FMLP(
        input_size   = meta["input_size"],
        hidden_sizes = meta["hidden_sizes"],
        num_classes  = meta["num_classes"],
        beta         = meta["beta"],
        threshold    = meta["threshold"],
        num_steps    = meta["num_steps"],
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, meta


def select_pixels(x_flat, epsilon, num_steps, k):
    """Return indices of the k pixels with the most feasible latency choices.

    Only considers nonzero pixels: zero-valued pixels land in a uniform dark
    region where perturbation rarely changes the latency assignment, making
    them poor candidates for meaningful verification.
    """
    lb = np.clip(x_flat - epsilon, 0.0, 1.0)
    ub = np.clip(x_flat + epsilon, 0.0, 1.0)
    n_choices = np.array([
        len(feasible_latencies(float(lb[i]), float(ub[i]), num_steps))
        if x_flat[i] > 0.0 else 0
        for i in range(len(x_flat))
    ])
    return np.argsort(n_choices)[::-1][:k]


def make_box(x_flat, epsilon, perturb_idx):
    """Build a Box where only perturb_idx dimensions are symbolic."""
    lb = x_flat.reshape(-1, 1).copy()
    ub = x_flat.reshape(-1, 1).copy()
    for i in perturb_idx:
        lb[i] = max(0.0, x_flat[i] - epsilon)
        ub[i] = min(1.0, x_flat[i] + epsilon)
    return Box(lb, ub)


def check_certified(out_box, true_label):
    """Return True if true_label's lower bound exceeds all other upper bounds."""
    lb = out_box.lb.flatten()
    ub = out_box.ub.flatten()
    competitors = [c for c in range(len(lb)) if c != true_label]
    return all(lb[true_label] > ub[c] for c in competitors)


def print_bounds_table(out_box, true_label, prefix="  "):
    lb = out_box.lb.flatten()
    ub = out_box.ub.flatten()
    print(f"{prefix}{'Class':<7} {'LB':>10} {'UB':>10}")
    print(f"{prefix}{'-'*7} {'-'*10} {'-'*10}")
    for c in range(len(lb)):
        marker = "  <-- true" if c == true_label else ""
        print(f"{prefix}{c:<7} {lb[c]:>10.3f} {ub[c]:>10.3f}{marker}")


if __name__ == "__main__":
    # ---- Load model --------------------------------------------------------
    model_path = SCRIPT_DIR / "models" / "mnist_snn.pt"
    meta_path  = SCRIPT_DIR / "models" / "mnist_snn_meta.json"
    if not model_path.exists() or not meta_path.exists():
        print(f"ERROR: models not found in {SCRIPT_DIR / 'models'}. Run train_snn.py first.")
        sys.exit(1)

    print("=" * 62)
    print("  SNN Verification - MNIST F2FMLP")
    print("=" * 62)

    model, meta = load_model(model_path, meta_path)
    snn = SpikingNeuralNetwork(model)
    num_steps = model.num_steps

    print(f"\nModel : {snn}")
    print(f"        trained {meta['epochs']} epochs, test acc = {meta['test_acc']:.1f}%")

    print(f"\nSettings:")
    print(f"  epsilon  = {EPSILON}  (L-inf perturbation radius)")
    print(f"  k_approx = {K_APPROX} pixels symbolic  (approx method)")
    print(f"  k_exact  = {K_EXACT}  pixels symbolic  (exact method)")

    # ---- Load MNIST test set -----------------------------------------------
    test_ds = datasets.MNIST(str(SCRIPT_DIR / "data"), train=False, download=True,
                             transform=transforms.ToTensor())

    # ---- Verify N_IMAGES correctly classified images -----------------------
    verified_count = 0
    tried = 0

    for img_idx in range(len(test_ds)):
        image, true_label = test_ds[img_idx]
        x_flat = image.flatten().numpy().astype(np.float32)

        # Skip misclassified images.
        with torch.no_grad():
            pred = snn.forward(torch.from_numpy(x_flat)).argmax().item()
        if pred != true_label:
            continue

        tried += 1
        print(f"\n{'-' * 62}")
        print(f"  Image {img_idx:5d}  |  true label = {true_label}  |  predicted = {pred}")
        print(f"{'-' * 62}")

        # ---- Select symbolic pixels ----------------------------------------
        idx_approx = select_pixels(x_flat, EPSILON, num_steps, K_APPROX)
        idx_exact  = select_pixels(x_flat, EPSILON, num_steps, K_EXACT)

        n_choices_exact = [
            len(feasible_latencies(
                float(max(0.0, x_flat[i] - EPSILON)),
                float(min(1.0, x_flat[i] + EPSILON)),
                num_steps,
            ))
            for i in idx_exact
        ]
        n_branches = 1
        for nc in n_choices_exact:
            n_branches *= nc

        print(f"\n  Pixel selection:")
        print(f"    Approx: {K_APPROX} pixels, values "
              f"[{x_flat[idx_approx].min():.3f}, {x_flat[idx_approx].max():.3f}]")
        print(f"    Exact : {K_EXACT} pixels, indices {idx_exact.tolist()}, "
              f"latency choices {n_choices_exact}, -> {n_branches} LP branches")

        # ---- Approx reach --------------------------------------------------
        box_approx = make_box(x_flat, EPSILON, idx_approx)
        t0 = time.time()
        out_approx = snn.reach(box_approx, method='approx')[0]
        t_approx   = time.time() - t0
        cert_approx = check_certified(out_approx, true_label)

        decision_approx = "CERTIFIED ROBUST" if cert_approx else "unknown"
        print(f"\n  Approx reach  ({t_approx:.2f}s)  ->  {decision_approx}")
        print_bounds_table(out_approx, true_label)

        # ---- Exact reach ---------------------------------------------------
        box_exact = make_box(x_flat, EPSILON, idx_exact)
        t0 = time.time()
        out_exact  = snn.reach(box_exact, method='exact')[0]
        t_exact    = time.time() - t0
        cert_exact = check_certified(out_exact, true_label)

        decision_exact = "CERTIFIED ROBUST" if cert_exact else "unknown"
        print(f"\n  Exact reach   ({t_exact:.2f}s)  ->  {decision_exact}")
        print_bounds_table(out_exact, true_label)

        if cert_approx or cert_exact:
            verified_count += 1

        if tried >= N_IMAGES:
            break

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'=' * 62}")
    print(f"  Summary: {verified_count}/{tried} images certified robust")
    print(f"           (epsilon={EPSILON}, k_exact={K_EXACT} pixels)")
    print(f"{'=' * 62}")
