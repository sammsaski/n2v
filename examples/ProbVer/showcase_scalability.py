"""
showcase_scalability.py - Reach Set Comparison: Exact vs Approx vs Probabilistic

This script demonstrates that probabilistic verification produces tighter
bounds than approximate methods, while both contain the exact reach set.

Key insight: Approximate methods over-approximate at each layer, causing
bounds to accumulate conservatism. Probabilistic learns bounds directly
from samples, avoiding this issue.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os

import n2v
from n2v.sets import Star, Box
from n2v.probabilistic import verify


# Output directory for visualizations
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_network(input_dim, hidden_dim, n_hidden_layers, output_dim):
    """Create a fully-connected ReLU network."""
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    model = nn.Sequential(*layers)
    model.eval()
    return model


def count_relus(model):
    """Count the number of ReLU activations in a model."""
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.out_features
    # Subtract output layer (no ReLU after it)
    for module in list(model.modules())[::-1]:
        if isinstance(module, nn.Linear):
            total -= module.out_features
            break
    return total


def main():
    print("=" * 70)
    print("REACH SET COMPARISON: EXACT vs APPROX vs PROBABILISTIC")
    print("=" * 70)

    print("""
This showcase compares reach sets from three verification methods:
- Exact: True reachable set (sound & complete)
- Approx: Over-approximation (sound, conservative)
- Probabilistic: Coverage guarantee (NOT sound, but tighter)

We use a small network where exact is tractable to show ground truth.
""")

    torch.manual_seed(42)
    np.random.seed(42)

    # =========================================================================
    # Create a small network where exact is tractable
    # =========================================================================
    print("\n" + "=" * 70)
    print("NETWORK SETUP")
    print("=" * 70)

    # Small network: 2 -> 8 -> 8 -> 2
    model = create_network(2, 8, 2, 2)
    n_relus = count_relus(model)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nNetwork: 2 -> 8 -> 8 -> 2")
    print(f"Parameters: {n_params:,}")
    print(f"ReLUs: {n_relus}")

    # Input region
    center = np.array([0.5, 0.5])
    epsilon = 0.2
    lb = (center - epsilon).astype(np.float32)
    ub = (center + epsilon).astype(np.float32)
    input_box = Box(lb, ub)
    input_star = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))

    print(f"Input: Box with center=[0.5, 0.5], epsilon={epsilon}")

    # =========================================================================
    # Run all three methods
    # =========================================================================
    verifier = n2v.NeuralNetwork(model)

    def model_fn(x):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).numpy()

    # --- Exact ---
    print("\n--- Exact Reachability ---")
    start = time.time()
    exact_results = verifier.reach(input_star, method='exact')
    exact_time = time.time() - start
    print(f"Time: {exact_time:.3f}s")
    print(f"Number of stars: {len(exact_results)}")

    # Get exact bounds by taking union of all stars
    exact_lbs = []
    exact_ubs = []
    for star in exact_results:
        star_lb, star_ub = star.get_ranges()
        exact_lbs.append(star_lb.flatten())
        exact_ubs.append(star_ub.flatten())
    exact_lb = np.min(exact_lbs, axis=0)
    exact_ub = np.max(exact_ubs, axis=0)
    exact_width = np.mean(exact_ub - exact_lb)
    print(f"Bounds: [{exact_lb[0]:.4f}, {exact_ub[0]:.4f}] x [{exact_lb[1]:.4f}, {exact_ub[1]:.4f}]")
    print(f"Average width: {exact_width:.4f}")

    # --- Approx ---
    print("\n--- Approximate Reachability ---")
    start = time.time()
    approx_result = verifier.reach(input_star, method='approx')
    approx_time = time.time() - start
    approx_lb, approx_ub = approx_result[0].get_ranges()
    approx_lb = approx_lb.flatten()
    approx_ub = approx_ub.flatten()
    approx_width = np.mean(approx_ub - approx_lb)
    print(f"Time: {approx_time:.3f}s")
    print(f"Bounds: [{approx_lb[0]:.4f}, {approx_ub[0]:.4f}] x [{approx_lb[1]:.4f}, {approx_ub[1]:.4f}]")
    print(f"Average width: {approx_width:.4f}")

    # --- Probabilistic (Naive Surrogate) ---
    print("\n--- Probabilistic Reachability (Naive Surrogate) ---")
    start = time.time()
    prob_naive_result = verify(
        model=model_fn,
        input_set=input_box,
        m=1000,
        epsilon=0.01,  # 99% coverage
        surrogate='naive',
        training_samples=500,
        seed=42,
        verbose=False
    )
    prob_naive_time = time.time() - start
    prob_naive_lb = prob_naive_result.lb.flatten()
    prob_naive_ub = prob_naive_result.ub.flatten()
    prob_naive_width = np.mean(prob_naive_ub - prob_naive_lb)
    print(f"Time: {prob_naive_time:.3f}s")
    print(f"Bounds: [{prob_naive_lb[0]:.4f}, {prob_naive_ub[0]:.4f}] x [{prob_naive_lb[1]:.4f}, {prob_naive_ub[1]:.4f}]")
    print(f"Average width: {prob_naive_width:.4f}")
    print(f"{prob_naive_result.get_guarantee_string()}")

    # --- Probabilistic (Clipping Block Surrogate) ---
    print("\n--- Probabilistic Reachability (Clipping Block Surrogate) ---")
    start = time.time()
    prob_clip_result = verify(
        model=model_fn,
        input_set=input_box,
        m=500,
        epsilon=0.01,  # 99% coverage
        surrogate='clipping_block',
        training_samples=200,
        seed=42,
        verbose=False
    )
    prob_clip_time = time.time() - start
    prob_clip_lb = prob_clip_result.lb.flatten()
    prob_clip_ub = prob_clip_result.ub.flatten()
    prob_clip_width = np.mean(prob_clip_ub - prob_clip_lb)
    print(f"Time: {prob_clip_time:.3f}s")
    print(f"Bounds: [{prob_clip_lb[0]:.4f}, {prob_clip_ub[0]:.4f}] x [{prob_clip_lb[1]:.4f}, {prob_clip_ub[1]:.4f}]")
    print(f"Average width: {prob_clip_width:.4f}")
    print(f"{prob_clip_result.get_guarantee_string()}")

    # =========================================================================
    # Summary comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Width':<12} {'Time':<12} {'Soundness'}")
    print("-" * 70)
    print(f"{'Exact':<25} {exact_width:<12.4f} {exact_time:<12.3f}s {'Yes (complete)'}")
    print(f"{'Approx':<25} {approx_width:<12.4f} {approx_time:<12.3f}s {'Yes (over-approx)'}")
    print(f"{'Prob (Naive)':<25} {prob_naive_width:<12.4f} {prob_naive_time:<12.3f}s {'No (99% coverage)'}")
    print(f"{'Prob (Clipping Block)':<25} {prob_clip_width:<12.4f} {prob_clip_time:<12.3f}s {'No (99% coverage)'}")

    print(f"\nApprox is {approx_width / exact_width:.2f}x wider than Exact")
    print(f"Prob (Naive) is {prob_naive_width / exact_width:.2f}x wider than Exact")
    print(f"Prob (Clipping Block) is {prob_clip_width / exact_width:.2f}x wider than Exact")

    # =========================================================================
    # Generate Visualization
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Sample outputs for scatter plot
        n_samples = 2000
        sample_inputs = np.random.uniform(lb, ub, size=(n_samples, 2)).astype(np.float32)
        sample_outputs = model_fn(sample_inputs)

        # Helper function to plot reach sets on an axis
        def plot_reach_sets(ax, prob_lb, prob_ub, title):
            # Plot samples (ground truth outputs)
            ax.scatter(sample_outputs[:, 0], sample_outputs[:, 1],
                       alpha=0.5, s=2, c='gray', label='Sampled Outputs', zorder=1)

            # Plot approximate bounds (outermost, behind)
            approx_rect = plt.Rectangle(
                (approx_lb[0], approx_lb[1]),
                approx_ub[0] - approx_lb[0],
                approx_ub[1] - approx_lb[1],
                fill=True, facecolor=(0.204, 0.596, 0.859, 0.2),
                edgecolor=(0.102, 0.322, 0.463, 0.8),
                linewidth=1.0, label='Approx (Box)', zorder=2
            )
            ax.add_patch(approx_rect)

            # Plot exact bounding box (union of all star bounds) - orange/amber color
            exact_box_rect = plt.Rectangle(
                (exact_lb[0], exact_lb[1]),
                exact_ub[0] - exact_lb[0],
                exact_ub[1] - exact_lb[1],
                fill=True, facecolor=(0.953, 0.612, 0.071, 0.2),
                edgecolor=(0.718, 0.584, 0.043, 0.8),
                linewidth=1.0, label='Exact (Box)', zorder=3
            )
            ax.add_patch(exact_box_rect)

            # Plot probabilistic bounds
            prob_rect = plt.Rectangle(
                (prob_lb[0], prob_lb[1]),
                prob_ub[0] - prob_lb[0],
                prob_ub[1] - prob_lb[1],
                fill=True, facecolor=(0.906, 0.298, 0.235, 0.2),
                edgecolor=(0.573, 0.169, 0.129, 0.8),
                linewidth=1.0, label='Probabilistic (99%)', zorder=4
            )
            ax.add_patch(prob_rect)

            # Plot exact reach set as Star polytopes
            for i, star in enumerate(exact_results):
                vertices = star.get_vertices()
                if vertices is not None and len(vertices) >= 3:
                    poly = Polygon(vertices, fill=True,
                                   facecolor=(0.180, 0.800, 0.443, 0.25),
                                   edgecolor=(0.118, 0.522, 0.286, 0.8),
                                   linewidth=1.0,
                                   label='Exact (Star)' if i == 0 else None, zorder=5)
                    ax.add_patch(poly)

            ax.set_xlabel('Output Dimension 0')
            ax.set_ylabel('Output Dimension 1')
            ax.set_title(title)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Expand axis limits for better visibility
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min
            ax.set_xlim(x_min - 0.3 * x_range, x_max + 0.3 * x_range)
            ax.set_ylim(y_min - 0.3 * y_range, y_max + 0.3 * y_range)

        # Panel 1: Naive surrogate
        plot_reach_sets(axes[0], prob_naive_lb, prob_naive_ub, 'Naive Surrogate')

        # Panel 2: Clipping Block surrogate
        plot_reach_sets(axes[1], prob_clip_lb, prob_clip_ub, 'Clipping Block Surrogate')

        # Panel 3: Timing table
        ax_table = axes[2]
        ax_table.axis('off')

        # Create table data
        table_data = [
            ['Exact', f'{exact_time:.3f}s', f'{exact_width:.4f}', 'Yes'],
            ['Approx', f'{approx_time:.3f}s', f'{approx_width:.4f}', 'Yes'],
            ['Prob (Naive)', f'{prob_naive_time:.3f}s', f'{prob_naive_width:.4f}', 'No'],
            ['Prob (Clipping)', f'{prob_clip_time:.3f}s', f'{prob_clip_width:.4f}', 'No'],
        ]
        col_labels = ['Method', 'Time', 'Width', 'Sound']

        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colWidths=[0.28, 0.18, 0.2, 0.14]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.6)

        # Style the header row
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#e6e6e6')
            table[(0, i)].set_text_props(weight='bold')

        ax_table.set_title('Timing Comparison')

        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, 'scalability_showcase.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available - skipping visualization")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
REACH SET COMPARISON:

1. EXACT (Green):
   - True reachable set (sound & complete)
   - Tightest possible bounds
   - Expensive for large networks (exponential in ReLUs)

2. APPROXIMATE (Blue):
   - Sound over-approximation (contains exact)
   - Over-approximation accumulates through layers
   - Polynomial time, but bounds can be very conservative

3. PROBABILISTIC (Red):
   - NOT sound (only coverage guarantee)
   - Learns bounds from samples, avoids layer-wise over-approximation
   - Often tighter than approx, closer to exact
   - Constant time regardless of network size

KEY TRADEOFF:
   - Approx: Guaranteed to contain ALL outputs (sound)
   - Probabilistic: Guaranteed to contain 99% of outputs (coverage)

RECOMMENDATION:
   - Use exact when tractable (small networks)
   - Use approx when soundness is required
   - Use probabilistic for rapid iteration or when 99% coverage suffices
""")


if __name__ == "__main__":
    main()
