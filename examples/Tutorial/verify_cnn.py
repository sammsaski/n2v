"""Verify the CNN MNIST classifier using n2v.

Demonstrates formal verification of the CNN MNIST classifier using
approximate ImageStar-based reachability analysis.

Verification task: local robustness against L-inf adversarial
perturbations.

Note: Exact reachability is not practical for CNNs at MNIST scale
due to the large number of ReLU neurons (thousands per layer).
Approximate reachability provides sound over-approximate bounds.

Requires: models/mnist_cnn_classifier.pth (run train_cnn.py first).
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
from n2v.sets import ImageStar
from n2v.nn import NeuralNetwork

n2v.set_lp_solver('linprog')
n2v.set_parallel(True)


class CNNMnistClassifier(nn.Module):
    """CNN MNIST classifier with AvgPool2d."""

    def __init__(self):
        super(CNNMnistClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 16, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    # --- Check Prerequisites ---
    model_path = 'models/mnist_cnn_classifier.pth'
    if not Path(model_path).exists():
        print(f"ERROR: {model_path} not found.")
        print("Please run train_cnn.py first to train the model.")
        sys.exit(1)

    # --- Setup ---
    print("Setting up...")

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cpu')

    Path('outputs').mkdir(exist_ok=True)

    # --- Load Trained CNN Model ---
    print("\nLoading trained model...")

    model = CNNMnistClassifier()
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

    print(f"Selected image {sample_idx}: "
          f"True label = {true_label}, Predicted = {pred_label}")

    # --- Create ImageStar from Bounds ---
    print("\nDefining input perturbation region (ImageStar)...")

    epsilon = 0.02
    num_classes = 10
    height, width = 28, 28
    num_channels = 1

    img_hwc = test_image.squeeze().numpy()
    img_hwc = img_hwc.reshape(height, width, num_channels)

    lb = np.clip(img_hwc - epsilon, 0, 1)
    ub = np.clip(img_hwc + epsilon, 0, 1)

    input_image_star = ImageStar.from_bounds(
        lb, ub,
        height=height,
        width=width,
        num_channels=num_channels,
    )

    print(f"Input ImageStar:")
    print(f"  Shape: {height}x{width}x{num_channels}")
    print(f"  Dimension: {input_image_star.dim}")
    print(f"  Variables: {input_image_star.nVar}")
    print(f"  Perturbation: epsilon = {epsilon}")

    # --- Run Approx Reachability ---
    print("\nStarting approx reachability analysis...")
    print("Approx over-approximates ReLU (no splitting).\n")

    start_time = time.time()
    output_stars = NeuralNetwork(model).reach(
        input_image_star, method='approx'
    )
    elapsed_time = time.time() - start_time

    print(f"Approx reachability complete!")
    print(f"  Time: {elapsed_time:.2f} seconds")
    print(f"  Output stars: {len(output_stars)}")

    # --- Extract Bounds ---
    print("\nComputing output bounds (LP-based)...")

    all_lbs = []
    all_ubs = []
    for star in output_stars:
        star_lb, star_ub = star.get_ranges()
        all_lbs.append(star_lb.flatten())
        all_ubs.append(star_ub.flatten())

    all_lbs = np.array(all_lbs)
    all_ubs = np.array(all_ubs)

    overall_lb = np.min(all_lbs, axis=0)
    overall_ub = np.max(all_ubs, axis=0)

    # --- Verification Decision ---
    approx_verified = all(
        overall_lb[true_label] > overall_ub[i]
        for i in range(num_classes) if i != true_label
    )
    approx_decision = (
        "verified robust" if approx_verified else "unknown"
    )

    # --- Save Output Ranges Figure ---
    print("\nSaving output ranges figure...")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(num_classes)

    for i in x:
        color = 'steelblue'
        ax.bar(i, overall_ub[i] - overall_lb[i],
               bottom=overall_lb[i], width=0.6, color=color,
               alpha=0.7)

    # Horizontal line at the lower bound of the true class
    ax.axhline(
        overall_lb[true_label], color='red',
        linestyle='--', linewidth=1.5, alpha=0.7,
        label=f'True class ({true_label}) LB',
    )

    ax.set_xlabel('Output Class')
    ax.set_ylabel('Output Value')
    ax.set_title(
        f'CNN Output Reachable Set Bounds '
        f'(approx, epsilon={epsilon})'
    )
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(
        'outputs/cnn_output_ranges.png', dpi=150,
        bbox_inches='tight',
    )
    plt.close()
    print("Plot saved to outputs/cnn_output_ranges.png")

    # --- Print Verification Summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    print(f"\nModel: CNN MNIST Classifier (AvgPool2d)")
    print(f"Architecture: Conv(8)-ReLU-AvgPool-Conv(16)-"
          f"ReLU-AvgPool-FC(10)")
    print(f"Test image: index {sample_idx}, "
          f"true label = {true_label}, "
          f"predicted = {pred_label}")
    print(f"Perturbation: epsilon = {epsilon} (L-inf)")
    print(f"Set type: ImageStar")

    print(f"\nApprox reachability:")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Output stars: {len(output_stars)}")
    print(f"  Decision: {approx_decision}")

    print(f"\nOutput bounds:")
    print(f"  {'Class':<8} {'LB':>12} {'UB':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12}")
    for i in range(num_classes):
        marker = " <-- true" if i == true_label else ""
        print(f"  {i:<8} {overall_lb[i]:>12.4f} "
              f"{overall_ub[i]:>12.4f}{marker}")

    print("\n" + "=" * 60)
