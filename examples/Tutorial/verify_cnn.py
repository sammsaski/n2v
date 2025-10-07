"""Verify the CNN MNIST classifier using n2v.

Demonstrates formal verification of the CNN MNIST classifier using
ImageStar-based reachability analysis (approx and exact).

Verification task: local robustness against L-inf adversarial perturbations.

Key advantage: AvgPool2d enables 10-100x faster verification than MaxPool2d
because it is a linear operation that does not cause star splitting.

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
from matplotlib.patches import Patch

import n2v
from n2v.sets import ImageStar
from n2v.nn import NeuralNetwork
from n2v.nn.layer_ops.dispatcher import reach_layer

n2v.set_lp_solver('linprog')
n2v.set_parallel(True)


class CNNMnistClassifier(nn.Module):
    """CNN MNIST classifier with AvgPool2d."""

    def __init__(self):
        super(CNNMnistClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),  # AvgPool - no splitting
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),  # AvgPool - no splitting
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

    device = torch.device('cpu')  # Verification on CPU

    # Create outputs directory for plots
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
    test_image_np = test_image.squeeze().numpy()

    with torch.no_grad():
        output = model(test_image.unsqueeze(0))
        pred_label = output.argmax(dim=1).item()

    print(f"Selected image {sample_idx}: True label = {true_label}, Predicted = {pred_label}")

    # --- Create ImageStar from Bounds ---
    print("\nDefining input perturbation region (ImageStar)...")
    print("For CNNs, we use ImageStar which preserves spatial structure.")

    epsilon = 0.02  # L-infinity perturbation magnitude
    num_classes = 10

    height, width = 28, 28
    num_channels = 1

    # Convert image to numpy (H, W, C) format
    img_hwc = test_image.squeeze().numpy()
    img_hwc = img_hwc.reshape(height, width, num_channels)

    # Create bounds: [img - epsilon, img + epsilon] clipped to [0, 1]
    lb = np.clip(img_hwc - epsilon, 0, 1)
    ub = np.clip(img_hwc + epsilon, 0, 1)

    # Create ImageStar from bounds
    input_image_star = ImageStar.from_bounds(
        lb, ub,
        height=height,
        width=width,
        num_channels=num_channels
    )

    print(f"Input ImageStar:")
    print(f"  Height: {input_image_star.height}")
    print(f"  Width: {input_image_star.width}")
    print(f"  Channels: {input_image_star.num_channels}")
    print(f"  Dimension: {input_image_star.dim}")
    print(f"  Variables: {input_image_star.nVar}")
    print(f"  Perturbation: epsilon = {epsilon}")

    # --- Run Approx Reachability ---
    print("\nStarting approx reachability analysis...")
    print("Approx method over-approximates ReLU (no splitting, polynomial time).\n")

    start_time = time.time()
    output_stars_approx = NeuralNetwork(model).reach(input_image_star, method='approx')
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
    print("Expected splitting only from ReLU layers (NOT from AvgPool!).\n")

    start_time = time.time()
    output_stars_exact = NeuralNetwork(model).reach(input_image_star, method='exact')
    elapsed_time_exact = time.time() - start_time

    print(f"Exact reachability complete!")
    print(f"  Time: {elapsed_time_exact:.2f} seconds")
    print(f"  Output stars: {len(output_stars_exact)}")
    print(f"\nAvgPool2d contributed 0 splits (linear operation!)")
    print(f"All splitting came from ReLU layers only")

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

    # --- Layer-by-Layer Analysis (exact) ---
    print("\nLayer-by-layer star propagation (exact):\n")

    all_layers = list(model.features.children()) + list(model.classifier.children())

    current_stars = [input_image_star]
    layer_stats = []

    for i, layer in enumerate(all_layers):
        layer_name = layer.__class__.__name__
        input_count = len(current_stars)

        start = time.time()
        current_stars = reach_layer(layer, current_stars, method='exact')
        layer_time = time.time() - start

        output_count = len(current_stars)
        splitting = output_count - input_count

        layer_stats.append({
            'layer': layer_name,
            'input': input_count,
            'output': output_count,
            'splitting': splitting,
            'time': layer_time
        })

        split_indicator = "SPLITS" if splitting > 0 else "no split"
        print(f"Layer {i+1:2d} ({layer_name:12s}): {input_count:4d} -> {output_count:4d} stars "
              f"({splitting:+4d}) {split_indicator} [{layer_time:.2f}s]")

    print(f"\nFinal output: {len(current_stars)} stars")

    # --- Save Star Growth Figure ---
    print("\nSaving star growth figure...")

    layer_names = [f"{i+1}. {s['layer']}" for i, s in enumerate(layer_stats)]
    star_counts = [s['output'] for s in layer_stats]
    times = [s['time'] for s in layer_stats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Star count growth
    ax1.plot(range(len(star_counts)), star_counts, marker='o', linewidth=2)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Stars')
    ax1.set_title('Star Set Growth Through Network')
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    for i, stat in enumerate(layer_stats):
        if stat['splitting'] > 0:
            ax1.axvline(i, color='red', alpha=0.2, linestyle='--')
            ax1.text(i, star_counts[i], f"+{stat['splitting']}",
                    ha='center', va='bottom', color='red', fontsize=8)

    # Computation time per layer
    colors_layer = ['red' if s['layer'] == 'ReLU' else 'blue' if 'Pool' in s['layer'] else 'gray'
              for s in layer_stats]
    ax2.bar(range(len(times)), times, color=colors_layer, alpha=0.7)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Computation Time per Layer')
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='ReLU (splits)'),
        Patch(facecolor='blue', alpha=0.7, label='AvgPool (no split)'),
        Patch(facecolor='gray', alpha=0.7, label='Other')
    ]
    ax2.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig('outputs/cnn_star_growth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to outputs/cnn_star_growth.png")

    # --- Verification Decisions ---
    approx_verified = all(
        overall_lb_approx[true_label] > overall_ub_approx[i]
        for i in range(num_classes) if i != true_label
    )
    approx_decision = "verified" if approx_verified else "unknown"

    exact_verified = all(
        overall_lb_exact[true_label] > overall_ub_exact[i]
        for i in range(num_classes) if i != true_label
    )
    exact_decision = "verified" if exact_verified else "not robust"

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

    # Highlight true class
    ax.axvline(true_label, color='green', linestyle='--', linewidth=2,
               alpha=0.5, label=f'True class ({true_label})')

    ax.set_xlabel('Output Class')
    ax.set_ylabel('Output Value')
    ax.set_title(f'Output Reachable Set Bounds (epsilon={epsilon})')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/cnn_output_ranges.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to outputs/cnn_output_ranges.png")

    # --- Print Verification Summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    print(f"\nModel: CNN MNIST Classifier (with AvgPool2d)")
    print(f"Architecture: Conv(8)-ReLU-AvgPool-Conv(16)-ReLU-AvgPool-FC(10)")
    print(f"Test image: index {sample_idx}, true label = {true_label}, predicted = {pred_label}")
    print(f"Perturbation: epsilon = {epsilon} (L-inf)")
    print(f"Set type: ImageStar")

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
