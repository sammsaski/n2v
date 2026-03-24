"""Verify the fully connected MNIST classifier using n2v.

Demonstrates formal verification of the MNIST FC classifier using
Star-based reachability analysis (approx and exact).

Verification task: local robustness against L-inf adversarial perturbations.

Requires: models/mnist_fc_classifier.pth (run train_fc.py first).
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path
import time
import matplotlib.pyplot as plt

import n2v
from n2v.sets import Star
from n2v.nn import NeuralNetwork

n2v.set_lp_solver('linprog')
n2v.set_parallel(True)


class FCMnistClassifier(nn.Module):
    """Fully connected MNIST classifier."""

    def __init__(self):
        super(FCMnistClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 10)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # --- Check Prerequisites ---
    model_path = 'models/mnist_fc_classifier.pth'
    if not Path(model_path).exists():
        print(f"ERROR: {model_path} not found.")
        print("Please run train_fc.py first to train the model.")
        sys.exit(1)

    # --- Setup ---
    print("Setting up...")

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cpu')  # Verification on CPU

    # Create outputs directory for plots
    Path('outputs').mkdir(exist_ok=True)

    # --- Load Trained Model ---
    print("\nLoading trained model...")

    model = FCMnistClassifier()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded!")
    print(f"Test accuracy: {checkpoint['test_accuracy']:.2f}%")
    print(f"Architecture: {checkpoint['architecture']}")

    # --- Load Test Data ---
    print("\nLoading test data...")

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    print(f"Test samples: {len(test_dataset)}")

    # --- Find Correctly Classified Sample ---
    print("\nSelecting test image...")

    sample_idx = 0
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            pred = output.argmax(dim=1).item()
        if pred == label:
            sample_idx = i
            break

    test_image, true_label = test_dataset[sample_idx]

    with torch.no_grad():
        output = model(test_image.unsqueeze(0))
        pred_label = output.argmax(dim=1).item()

    print(f"Selected image {sample_idx}: True label = {true_label}, Predicted = {pred_label}")

    # --- Create Star from Bounds ---
    print("\nDefining input perturbation region...")

    epsilon = 0.02  # L-infinity perturbation magnitude
    num_classes = 10

    # Flatten image to vector
    img_vector = test_image.flatten().numpy().reshape(-1, 1)

    # Create bounds: [img - epsilon, img + epsilon] clipped to [0, 1]
    lb = np.clip(img_vector - epsilon, 0, 1)
    ub = np.clip(img_vector + epsilon, 0, 1)

    # Create Star set from bounds
    input_star = Star.from_bounds(lb, ub)

    print(f"Input Star Set:")
    print(f"  Dimension: {input_star.dim}")
    print(f"  Variables: {input_star.nVar}")
    print(f"  Perturbation: epsilon = {epsilon}")
    print(f"  Represents all images within L-inf ball of radius {epsilon}")

    # --- Run Approx Reachability ---
    print("\nStarting approx reachability analysis...")
    print("Approx method over-approximates ReLU (no splitting, polynomial time).\n")

    start_time = time.time()
    output_stars_approx = NeuralNetwork(model).reach(input_star, method='approx')
    elapsed_time_approx = time.time() - start_time

    print(f"Approx reachability complete!")
    print(f"  Time: {elapsed_time_approx:.2f} seconds")
    print(f"  Output stars: {len(output_stars_approx)}")

    # --- Extract Approx Bounds using get_ranges() ---
    print("\nComputing approx output bounds (LP-based)...")

    all_lbs_approx = []
    all_ubs_approx = []
    for star in output_stars_approx:
        star_lb, star_ub = star.get_ranges()
        all_lbs_approx.append(star_lb.flatten())
        all_ubs_approx.append(star_ub.flatten())

    all_lbs_approx = np.array(all_lbs_approx)
    all_ubs_approx = np.array(all_ubs_approx)

    overall_lb_approx = np.min(all_lbs_approx, axis=0)
    overall_ub_approx = np.max(all_ubs_approx, axis=0)

    # --- Run Exact Reachability ---
    print("\nStarting exact reachability analysis...")
    print("Exact method splits on ReLU neurons (exponential in worst case).\n")

    start_time = time.time()
    output_stars_exact = NeuralNetwork(model).reach(input_star, method='exact')
    elapsed_time_exact = time.time() - start_time

    print(f"Exact reachability complete!")
    print(f"  Time: {elapsed_time_exact:.2f} seconds")
    print(f"  Output stars: {len(output_stars_exact)}")

    # --- Extract Exact Bounds using get_ranges() ---
    print("\nComputing exact output bounds (LP-based)...")

    all_lbs_exact = []
    all_ubs_exact = []
    for star in output_stars_exact:
        star_lb, star_ub = star.get_ranges()
        all_lbs_exact.append(star_lb.flatten())
        all_ubs_exact.append(star_ub.flatten())

    all_lbs_exact = np.array(all_lbs_exact)
    all_ubs_exact = np.array(all_ubs_exact)

    overall_lb_exact = np.min(all_lbs_exact, axis=0)
    overall_ub_exact = np.max(all_ubs_exact, axis=0)

    # --- Verification Decisions ---
    approx_verified = all(
        overall_lb_approx[true_label] > overall_ub_approx[i]
        for i in range(num_classes) if i != true_label
    )
    approx_decision = "verified robust" if approx_verified else "unknown"

    exact_verified = all(
        overall_lb_exact[true_label] > overall_ub_exact[i]
        for i in range(num_classes) if i != true_label
    )
    exact_decision = (
        "verified robust" if exact_verified
        else "verified non-robust"
    )

    # --- Save Output Ranges Figure ---
    print("\nSaving output ranges figure...")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(num_classes)
    width = 0.35

    # Approx bounds as bars from lb to ub
    for i in x:
        color = 'steelblue'
        ax.bar(i - width/2, overall_ub_approx[i] - overall_lb_approx[i],
               bottom=overall_lb_approx[i], width=width, color=color,
               alpha=0.7, label='Approx' if i == 0 else '')

    # Exact bounds as bars from lb to ub
    for i in x:
        color = 'darkorange'
        ax.bar(i + width/2, overall_ub_exact[i] - overall_lb_exact[i],
               bottom=overall_lb_exact[i], width=width, color=color,
               alpha=0.7, label='Exact' if i == 0 else '')

    # Horizontal line at the approx lower bound of the true class
    ax.axhline(
        overall_lb_approx[true_label], color='red',
        linestyle='--', linewidth=1.5, alpha=0.7,
        label=f'True class ({true_label}) approx LB',
    )

    ax.set_xlabel('Output Class')
    ax.set_ylabel('Output Value')
    ax.set_title(f'Output Reachable Set Bounds (epsilon={epsilon})')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/fc_output_ranges.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to outputs/fc_output_ranges.png")

    # --- Print Verification Summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    print(f"\nModel: FC MNIST Classifier")
    print(f"Architecture: 784 -> 30 -> 30 -> 10")
    print(f"Test image: index {sample_idx}, true label = {true_label}, predicted = {pred_label}")
    print(f"Perturbation: epsilon = {epsilon} (L-inf)")

    print(f"\nApprox reachability:")
    print(f"  Time: {elapsed_time_approx:.2f}s")
    print(f"  Output stars: {len(output_stars_approx)}")
    print(f"  Decision: {approx_decision}")

    print(f"\nExact reachability:")
    print(f"  Time: {elapsed_time_exact:.2f}s")
    print(f"  Output stars: {len(output_stars_exact)}")
    print(f"  Decision: {exact_decision}")

    print(f"\nOutput bounds:")
    print(f"  {'Class':<8} {'Approx LB':>12} {'Approx UB':>12} {'Exact LB':>12} {'Exact UB':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for i in range(num_classes):
        marker = " <-- true" if i == true_label else ""
        print(f"  {i:<8} {overall_lb_approx[i]:>12.4f} {overall_ub_approx[i]:>12.4f} "
              f"{overall_lb_exact[i]:>12.4f} {overall_ub_exact[i]:>12.4f}{marker}")

    print("\n" + "=" * 60)
