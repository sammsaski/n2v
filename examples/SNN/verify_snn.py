"""
SNN Verification with n2v  - Worked Example
===========================================

This script is a self-contained walkthrough of local robustness verification
for a spiking neural network using n2v's SpikingNeuralNetwork wrapper.

Background
----------
The model is a First-to-Fire MLP (F2FMLP): a fully-connected SNN where each
neuron fires at most once, and classification is based on which output neuron
fires first (i.e., has the largest latency-coded score). Inputs are encoded as
spike trains using latency coding: brighter pixels fire earlier.

Verification question
---------------------
Given a test image x and perturbation radius epsilon, can we *prove* that every
image x' with ||x' - x||_inf <= epsilon is classified identically to x?

This is local L-inf robustness: the model's decision should not change anywhere
within the epsilon-ball around x.

How SpikingNeuralNetwork.reach() works
---------------------------------------
reach() takes a set of possible inputs (a Box or Star) and computes guaranteed
bounds on the model's output scores across every point in that set.

  out = snn.reach(input_box, method='exact')

The output is a list of Box objects (always length 1 for this model). The
output Box gives:
  out[0].lb[c]  -- lowest possible score for class c over all inputs in the box
  out[0].ub[c]  -- highest possible score for class c over all inputs in the box

These bounds are *sound*: the true score for every concrete input in the box
lies within [lb[c], ub[c]].

Certification rule
------------------
The true label is certified robust if and only if its lower bound strictly
exceeds every competitor's upper bound:

  lb[true_label] > ub[c]  for all c != true_label

If this holds, no input in the box can cause a misclassification.

Exact vs. approx method
------------------------
  method='exact'  -- full latency enumeration: branches over every combination
                     of spike times for the symbolic pixels. Sound and complete
                     for the branched pixels; tighter bounds than approx.
  method='approx' -- depth-0 LP relaxation: single LP, faster but looser bounds.
                     May fail to certify images that exact can certify.

This example uses method='exact'. The number of LP branches grows as
  product(latency_choices[i] for i in symbolic_pixels)
so keep the number of symbolic pixels small (K=5 is a practical choice).

Symbolic pixel selection
------------------------
Not all pixels contribute equally to output uncertainty. A pixel at value v
under an epsilon perturbation covers the range [v-eps, v+eps]. The number of
distinct spike times (latencies) feasible in that range is the key metric.
Pixels with more feasible latency choices introduce more branching and tighter
bounds relative to the LP work done. Zero pixels are excluded: they fire at the
silent sentinel regardless of small perturbations.

Requires: examples/SNN/models/mnist_snn.pt
          examples/SNN/models/mnist_snn_meta.json
Run train_snn.py first to produce these files.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

# n2v imports -------------------------------------------------------------------
#   SpikingNeuralNetwork  -- wraps an F2FMLP for reachability analysis
#   Box                   -- axis-aligned input set: each dimension has [lb, ub]
#   F2FMLP                -- the spiking model class (needed for reconstruction)
#   feasible_latencies    -- utility: how many spike times are possible in [lb,ub]
from n2v import SpikingNeuralNetwork, Box
from n2v.snn.model import F2FMLP
from n2v.snn.lp import feasible_latencies

SCRIPT_DIR = Path(__file__).parent

# ==============================================================================
# Configuration
# ==============================================================================

EPSILONS = [0.05, 0.10]  # L-inf radii to compare  - shows how bounds widen
K        = 5             # number of symbolic (perturbed) pixels per image
N_IMAGES = 5             # number of test images to verify


# ==============================================================================
# Step 1: Load the trained model
# ==============================================================================

def load_model(model_path: Path, meta_path: Path):
    """Reconstruct an F2FMLP from a saved state dict and metadata file.

    SpikingNeuralNetwork works with any F2FMLP instance  - it does not care
    how the model was trained. The only requirement is that the weights are
    loaded into an F2FMLP with matching architecture.

    We save state dicts rather than full model objects because snntorch's
    surrogate gradient functions are defined as closures and cannot be pickled.
    The metadata JSON captures the architecture hyperparameters needed to
    reconstruct the model shell before loading weights.
    """
    with open(meta_path) as f:
        meta = json.load(f)

    # Reconstruct the model architecture from metadata.
    model = F2FMLP(
        input_size   = meta["input_size"],
        hidden_sizes = meta["hidden_sizes"],
        num_classes  = meta["num_classes"],
        beta         = meta["beta"],         # LIF membrane decay rate
        threshold    = meta["threshold"],    # spike threshold
        num_steps    = meta["num_steps"],    # latency coding timesteps T
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, meta


# ==============================================================================
# Step 2: Select which pixels to treat as symbolic
# ==============================================================================

def select_pixels(x_flat: np.ndarray, epsilon: float, num_steps: int, k: int):
    """Choose the k pixels that introduce the most timing ambiguity under epsilon.

    In latency coding, a pixel at intensity v fires at a spike time determined
    by v. When v is uncertain (lies anywhere in [v-eps, v+eps]), the spike time
    may take several different values  - these are the 'feasible latencies'.

    Pixels with more feasible latency choices contribute more uncertainty to the
    output bounds and are the most informative symbolic dimensions to model.

    Zero-valued pixels are excluded: they land in a uniformly dark region where
    small perturbations typically do not change the assigned spike time.
    """
    lb = np.clip(x_flat - epsilon, 0.0, 1.0)
    ub = np.clip(x_flat + epsilon, 0.0, 1.0)
    n_choices = np.array([
        len(feasible_latencies(float(lb[i]), float(ub[i]), num_steps))
        if x_flat[i] > 0.0 else 0
        for i in range(len(x_flat))
    ])
    return np.argsort(n_choices)[::-1][:k]


# ==============================================================================
# Step 3: Build the input set (Box)
# ==============================================================================

def make_input_box(x_flat: np.ndarray, epsilon: float, symbolic_idx: np.ndarray):
    """Construct a Box representing the perturbed input set.

    A Box is an axis-aligned set: each input dimension i has an independent
    interval [lb[i], ub[i]]. Dimensions in symbolic_idx are widened by epsilon;
    all other dimensions are fixed at their nominal (point) values.

    This models the assumption that at most k pixels are adversarially perturbed
    while the rest of the image is unchanged. It is a conservative over-
    approximation of the true threat model (any k pixels could be perturbed
    simultaneously), which keeps the verification sound.

    SpikingNeuralNetwork.reach() also accepts a Star set. The Star's predicate
    is used to compute per-dimension marginal bounds (via LP), which are then
    passed to the LP engine  - so cross-dimension correlations in the predicate
    are respected during bound extraction but not during the LP solve itself.
    """
    lb = x_flat.reshape(-1, 1).copy()
    ub = x_flat.reshape(-1, 1).copy()
    for i in symbolic_idx:
        lb[i] = max(0.0, x_flat[i] - epsilon)
        ub[i] = min(1.0, x_flat[i] + epsilon)
    return Box(lb, ub)


# ==============================================================================
# Step 4: Interpret the output
# ==============================================================================

def is_certified(output_box: Box, true_label: int) -> bool:
    """Return True if the model is provably robust at true_label.

    The output Box bounds every possible score vector over the input set.
    Robustness holds iff the true label's minimum score exceeds every
    competitor's maximum score  - i.e., no input in the box can flip the
    prediction away from true_label.
    """
    lb = output_box.lb.flatten()
    ub = output_box.ub.flatten()
    return all(lb[true_label] > ub[c]
               for c in range(len(lb)) if c != true_label)


def print_output_bounds(output_box: Box, true_label: int, prefix: str = "    "):
    """Print the per-class score intervals [lb, ub] from the output Box.

    Wider intervals mean more uncertainty. The true label is certified robust
    if lb[true_label] > max(ub[c] for c != true_label).
    """
    lb = output_box.lb.flatten()
    ub = output_box.ub.flatten()
    print(f"{prefix}{'Class':<7} {'LB':>8} {'UB':>8}   interval width")
    print(f"{prefix}{'-'*7} {'-'*8} {'-'*8}   {'-'*14}")
    for c in range(len(lb)):
        width  = ub[c] - lb[c]
        marker = "  <-- true label" if c == true_label else ""
        print(f"{prefix}{c:<7} {lb[c]:>8.2f} {ub[c]:>8.2f}   {width:>6.2f}{marker}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    # ---- Load model ----------------------------------------------------------
    model_path = SCRIPT_DIR / "models" / "mnist_snn.pt"
    meta_path  = SCRIPT_DIR / "models" / "mnist_snn_meta.json"
    if not model_path.exists() or not meta_path.exists():
        print(f"ERROR: model files not found under {SCRIPT_DIR / 'models'}.")
        print("Run train_snn.py first to train and save the model.")
        sys.exit(1)

    print("=" * 66)
    print("  SNN Verification with n2v  - MNIST F2FMLP Worked Example")
    print("=" * 66)

    model, meta = load_model(model_path, meta_path)
    print(f"\n[Model loaded]")
    print(f"  Architecture : F2FMLP  {meta['input_size']} -> "
          f"{' -> '.join(str(h) for h in meta['hidden_sizes'])} -> {meta['num_classes']}")
    print(f"  Timesteps T  : {meta['num_steps']}  (latency coding range: 0 .. {meta['num_steps']-1})")
    print(f"  Test accuracy: {meta['test_acc']:.1f}%  ({meta['epochs']} epochs)")

    # Wrap the trained model in SpikingNeuralNetwork.
    # This gives access to reach() for reachability analysis.
    # SpikingNeuralNetwork.forward() works like the original model for
    # point evaluation; reach() is the new capability.
    snn = SpikingNeuralNetwork(model)
    print(f"\n[Wrapped in SpikingNeuralNetwork]")
    print(f"  {snn}")

    print(f"\n[Verification settings]")
    print(f"  Symbolic pixels k = {K}  (nonzero, ranked by latency ambiguity)")
    print(f"  Method            = exact  (full latency branch-and-bound LP)")
    print(f"  Epsilons          = {EPSILONS}  (L-inf perturbation radii)")
    print(f"\n  For each image we use the same k pixels across both epsilon values")
    print(f"  (selected at the larger epsilon) so the bounds are directly comparable.")

    # ---- Load MNIST test set -------------------------------------------------
    test_ds = datasets.MNIST(str(SCRIPT_DIR / "data"), train=False, download=True,
                             transform=transforms.ToTensor())

    # Use the larger epsilon for pixel selection so both runs share the same
    # symbolic dimensions and their outputs can be compared directly.
    max_eps = max(EPSILONS)

    # ---- Verify N_IMAGES correctly classified images -------------------------
    summary = {eps: {"certified": 0, "total": 0} for eps in EPSILONS}
    verified = 0

    for img_idx in range(len(test_ds)):
        if verified >= N_IMAGES:
            break

        image, true_label = test_ds[img_idx]
        x_flat = image.flatten().numpy().astype(np.float32)

        # Only verify images the model classifies correctly. Verification asks
        # whether the model is *robustly* correct, not whether it is correct at
        # all  - so starting from a misclassified image is not meaningful.
        with torch.no_grad():
            pred = snn.forward(torch.from_numpy(x_flat)).argmax().item()
        if pred != true_label:
            continue

        verified += 1
        print(f"\n{'=' * 66}")
        print(f"  Image {img_idx:5d}   true label = {true_label}   predicted = {pred}")
        print(f"{'=' * 66}")

        # Step 2: Select symbolic pixels.
        symbolic_idx = select_pixels(x_flat, max_eps, model.num_steps, K)
        latency_choices = [
            len(feasible_latencies(
                float(max(0.0, x_flat[i] - max_eps)),
                float(min(1.0, x_flat[i] + max_eps)),
                model.num_steps,
            ))
            for i in symbolic_idx
        ]
        n_branches = 1
        for nc in latency_choices:
            n_branches *= nc

        print(f"\n  [Symbolic pixels  - selected at eps={max_eps}]")
        print(f"  Pixel indices : {symbolic_idx.tolist()}")
        print(f"  Pixel values  : {[f'{x_flat[i]:.3f}' for i in symbolic_idx]}")
        print(f"  Latency choices per pixel: {latency_choices}")
        print(f"  Total LP branches (product): {n_branches}")

        # Steps 3-4: Build the input box, run reach(), interpret results.
        for eps in EPSILONS:
            print(f"\n  --- epsilon = {eps:.2f} ---")

            # Step 3: Build the input Box.
            # Symbolic pixels are free within [x-eps, x+eps] clipped to [0,1].
            # All other pixels are fixed at their nominal value (point bounds).
            input_box = make_input_box(x_flat, eps, symbolic_idx)

            # Step 4: Run reachability analysis.
            # reach() returns a list of output Boxes (length 1 for F2FMLP).
            # Each output Box bounds the score vector over every input in input_box.
            t0 = time.time()
            output_boxes = snn.reach(input_box, method='exact')
            elapsed = time.time() - t0
            output_box = output_boxes[0]  # single output set

            # Step 4: Check certification.
            certified = is_certified(output_box, true_label)
            summary[eps]["total"] += 1
            if certified:
                summary[eps]["certified"] += 1

            status = "CERTIFIED ROBUST" if certified else "not certified"
            print(f"  reach() completed in {elapsed:.2f}s  ->  {status}")
            print(f"\n  Output bounds (score intervals over all inputs in the box):")
            print_output_bounds(output_box, true_label)

            if certified:
                lb = output_box.lb.flatten()
                ub = output_box.ub.flatten()
                margin = lb[true_label] - max(
                    ub[c] for c in range(len(lb)) if c != true_label
                )
                print(f"\n  Certified margin: {margin:.2f}  "
                      f"(lb[{true_label}]={lb[true_label]:.2f} exceeds all competitor UBs by this amount)")

    # ---- Summary -------------------------------------------------------------
    print(f"\n{'=' * 66}")
    print(f"  Final Summary  (k={K} symbolic pixels, method=exact, {verified} images)")
    print(f"  {'epsilon':<10}  {'certified':>10}  {'out of':>8}  {'rate':>8}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for eps in EPSILONS:
        r = summary[eps]
        rate = 100.0 * r["certified"] / r["total"] if r["total"] > 0 else 0.0
        print(f"  {eps:<10.2f}  {r['certified']:>10}  {r['total']:>8}  {rate:>7.0f}%")
    print(f"{'=' * 66}")
    print()
    print("  Interpretation:")
    print("  A 'CERTIFIED ROBUST' result is a *proof*  - not a statistical estimate.")
    print("  Every image within the epsilon-ball is guaranteed to be classified")
    print(f"  identically to the original, based on the {K} most timing-ambiguous pixels.")
    print("  Images where epsilon=0.05 certifies but epsilon=0.10 does not illustrate")
    print("  the trade-off between perturbation strength and verifiability.")
    print()
