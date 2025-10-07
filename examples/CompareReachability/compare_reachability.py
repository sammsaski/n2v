"""
Reachability comparison: Exact vs Overapproximate vs Probabilistic (Naive).

Replicates the MATLAB NNV example from:
  nnv/examples/Tutorial/NN/compareReachability/reach_exact_vs_approx.m

This script:
1. Loads the same neural network (NeuralNetwork7_3.mat)
2. Computes exact reachable sets (multiple stars)
3. Computes overapproximate reachable set (single star)
4. Computes probabilistic (naive) reachable set bounds
5. Plots all results together, matching the MATLAB visualization
"""

import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path

import n2v
from n2v.sets import Star, Box
from n2v.probabilistic import verify as prob_verify


def load_matlab_network(mat_file: str) -> nn.Sequential:
    """
    Load neural network weights from MATLAB .mat file.

    The MATLAB network has:
    - 7 hidden layers with ReLU activation
    - 1 output layer (linear)

    Architecture: 3 -> 7 -> 7 -> 7 -> 7 -> 7 -> 7 -> 7 -> 2

    Args:
        mat_file: Path to NeuralNetwork7_3.mat

    Returns:
        PyTorch Sequential model
    """
    data = scipy.io.loadmat(mat_file)
    W = data['W']
    b = data['b']

    n_layers = W.shape[1]

    layers = []
    for i in range(n_layers):
        Wi = W[0, i]  # Weight matrix for layer i
        bi = b[0, i].flatten()  # Bias vector for layer i

        # Create linear layer
        linear = nn.Linear(Wi.shape[1], Wi.shape[0])
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(Wi, dtype=torch.float32))
            linear.bias.copy_(torch.tensor(bi, dtype=torch.float32))
        layers.append(linear)

        # Add ReLU for hidden layers (not the last layer)
        if i < n_layers - 1:
            layers.append(nn.ReLU())

    model = nn.Sequential(*layers)
    model.eval()

    return model


def evaluate_network_samples(model: nn.Module, lb: np.ndarray, ub: np.ndarray,
                             step: float = 0.2) -> np.ndarray:
    """
    Evaluate the network on a grid of sample inputs.

    Args:
        model: PyTorch model
        lb: Lower bounds of input region
        ub: Upper bounds of input region
        step: Step size for grid sampling

    Returns:
        Array of output points (n_samples, output_dim)
    """
    outputs = []

    for x1 in np.arange(lb[0], ub[0] + step/2, step):
        for x2 in np.arange(lb[1], ub[1] + step/2, step):
            for x3 in np.arange(lb[2], ub[2] + step/2, step):
                xi = torch.tensor([[x1, x2, x3]], dtype=torch.float32)
                with torch.no_grad():
                    yi = model(xi).numpy().flatten()
                outputs.append(yi)

    return np.array(outputs)


def plot_star_2d(ax, star: Star, color: str, alpha: float = 0.5,
                 edgecolor: str = 'black', linewidth: float = 1.0):
    """
    Plot a 2D Star set as a filled polygon.

    Args:
        ax: Matplotlib axis
        star: Star set to plot
        color: Fill color
        alpha: Transparency
        edgecolor: Edge color
        linewidth: Edge line width
    """
    vertices = star.get_vertices()

    if vertices is None or len(vertices) < 3:
        # Fall back to bounding box if vertices can't be computed
        lb, ub = star.get_ranges()
        lb = lb.flatten()
        ub = ub.flatten()
        rect_vertices = np.array([
            [lb[0], lb[1]],
            [ub[0], lb[1]],
            [ub[0], ub[1]],
            [lb[0], ub[1]]
        ])
        polygon = Polygon(rect_vertices, closed=True, facecolor=color,
                         edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
        ax.add_patch(polygon)
    else:
        polygon = Polygon(vertices[:, :2], closed=True, facecolor=color,
                         edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
        ax.add_patch(polygon)


def main():
    print("=" * 60)
    print("Compare Reachability: Exact vs Approx vs Probabilistic")
    print("=" * 60)

    # Get path to the .mat file
    script_dir = Path(__file__).parent
    mat_file = script_dir / "NeuralNetwork7_3.mat"

    if not mat_file.exists():
        raise FileNotFoundError(f"Network file not found: {mat_file}")

    # =========================================
    # Step 1: Load the neural network
    # =========================================
    print("\n[1] Loading neural network from MATLAB file...")
    model = load_matlab_network(str(mat_file))
    print(f"    Network architecture: {model}")

    # =========================================
    # Step 2: Define input set
    # =========================================
    print("\n[2] Defining input set...")
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])

    input_star = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))
    input_box = Box(lb.reshape(-1, 1), ub.reshape(-1, 1))

    print(f"    Input bounds: lb={lb}, ub={ub}")
    print(f"    Input Star: {input_star}")

    # =========================================
    # Step 3: Create verifier
    # =========================================
    verifier = n2v.NeuralNetwork(model)

    # =========================================
    # Step 4: Exact reachability
    # =========================================
    print("\n[3] Computing EXACT reachable set...")
    import time

    t_exact_start = time.time()
    exact_stars = verifier.reach(input_star, method='exact')
    t_exact = time.time() - t_exact_start

    print(f"    Time: {t_exact:.3f}s")
    print(f"    Number of output stars: {len(exact_stars)}")

    # =========================================
    # Step 5: Overapproximate reachability
    # =========================================
    print("\n[4] Computing OVERAPPROXIMATE reachable set...")

    t_approx_start = time.time()
    approx_stars = verifier.reach(input_star, method='approx')
    t_approx = time.time() - t_approx_start

    print(f"    Time: {t_approx:.3f}s")
    print(f"    Number of output stars: {len(approx_stars)}")

    # =========================================
    # Step 6: Probabilistic (naive) reachability
    # =========================================
    print("\n[5] Computing PROBABILISTIC (naive) reachable set...")

    # Create model function for probabilistic verification
    def model_fn(x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            return model(x_tensor).numpy()

    t_prob_naive_start = time.time()
    prob_naive_result = prob_verify(
        model=model_fn,
        input_set=input_box,
        m=5000,           # Calibration samples
        epsilon=0.01,     # 99% coverage
        surrogate='naive',
        seed=42,
        verbose=True
    )
    t_prob_naive = time.time() - t_prob_naive_start

    print(f"    Time: {t_prob_naive:.3f}s")
    print(f"    Probabilistic bounds: lb={prob_naive_result.lb.flatten()}, ub={prob_naive_result.ub.flatten()}")
    print(f"    {prob_naive_result.get_guarantee_string()}")

    # =========================================
    # Step 7: Sample evaluations
    # =========================================
    print("\n[6] Evaluating network on sample inputs...")
    sample_outputs = evaluate_network_samples(model, lb, ub, step=0.2)
    print(f"    Number of samples: {len(sample_outputs)}")

    # Evaluate corner points
    with torch.no_grad():
        y_lb = model(torch.tensor([lb], dtype=torch.float32)).numpy().flatten()
        y_ub = model(torch.tensor([ub], dtype=torch.float32)).numpy().flatten()
        y_mid = model(torch.tensor([(lb + ub) / 2], dtype=torch.float32)).numpy().flatten()

    # =========================================
    # Step 8: Compute exact bounds for zoomed view
    # =========================================
    print("\n[7] Computing exact reachable set bounds...")

    # Get overall bounds from exact stars
    all_exact_lbs = []
    all_exact_ubs = []
    for star in exact_stars:
        star_lb, star_ub = star.get_ranges()
        all_exact_lbs.append(star_lb.flatten())
        all_exact_ubs.append(star_ub.flatten())

    exact_lb = np.min(all_exact_lbs, axis=0)
    exact_ub = np.max(all_exact_ubs, axis=0)
    print(f"    Exact bounds: lb={exact_lb}, ub={exact_ub}")

    # Get approx bounds
    approx_lb, approx_ub = approx_stars[0].get_ranges()
    approx_lb = approx_lb.flatten()
    approx_ub = approx_ub.flatten()
    print(f"    Approx bounds: lb={approx_lb}, ub={approx_ub}")

    # =========================================
    # Step 9: Plot results
    # =========================================
    print("\n[8] Plotting results...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot colors for exact stars (cycling like MATLAB)
    exact_colors = ['red', 'green', 'blue', 'magenta', 'yellow']
    prob_naive_lb = prob_naive_result.lb.flatten()
    prob_naive_ub = prob_naive_result.ub.flatten()

    from matplotlib.patches import Patch

    # 1. Plot overapproximate set (cyan, background)
    for star in approx_stars:
        plot_star_2d(ax, star, color='cyan', alpha=0.8, edgecolor='black', linewidth=1.5)

    # 2. Plot probabilistic (naive) bounds (shaded with fine dashed edge)
    prob_naive_rect = plt.Rectangle(
        (prob_naive_lb[0], prob_naive_lb[1]),
        prob_naive_ub[0] - prob_naive_lb[0],
        prob_naive_ub[1] - prob_naive_lb[1],
        facecolor='lightgreen', alpha=0.25,
        edgecolor='darkgreen', linewidth=2.0,
        linestyle=(0, (3, 1)),  # Fine dashes: 3 on, 1 off
        label='Probabilistic (naive)'
    )
    ax.add_patch(prob_naive_rect)

    # 3. Plot exact sets (colored polygons)
    for i, star in enumerate(exact_stars):
        color = exact_colors[i % len(exact_colors)]
        plot_star_2d(ax, star, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

    # 5. Plot sample evaluations (black dots)
    ax.plot(sample_outputs[:, 0], sample_outputs[:, 1], 'k.', markersize=4, label='Samples')

    # 6. Plot corner evaluations (black x markers)
    ax.plot(y_lb[0], y_lb[1], 'kx', markersize=8, markeredgewidth=2)
    ax.plot(y_ub[0], y_ub[1], 'kx', markersize=8, markeredgewidth=2)
    ax.plot(y_mid[0], y_mid[1], 'kx', markersize=8, markeredgewidth=2)

    # Set axis properties
    ax.set_xlabel('Output Dimension 1', fontsize=12)
    ax.set_ylabel('Output Dimension 2', fontsize=12)
    ax.set_title('Output Reachable Sets', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.autoscale()

    # Add legend
    legend_elements = [
        Patch(facecolor='cyan', edgecolor='black', alpha=0.8, label='Overapproximate'),
        Patch(facecolor='lightgreen', edgecolor='darkgreen', alpha=0.25,
              linestyle=(0, (3, 1)), linewidth=2.0, label='Probabilistic (naive)'),
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Exact (multiple stars)'),
        plt.Line2D([0], [0], marker='.', color='black', linestyle='None', markersize=8, label='Sample evaluations'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = script_dir / "output_reachability_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"    Figure saved to: {output_file}")

    plt.show()

    # =========================================
    # Step 10: Print summary
    # =========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Exact reachability:           {len(exact_stars)} stars, {t_exact:.3f}s")
    print(f"  Approx reachability:          {len(approx_stars)} stars, {t_approx:.3f}s")
    print(f"  Probabilistic (naive):        1 box, {t_prob_naive:.3f}s")
    print(f"\n  Timing comparison:")
    print(f"    Exact is {t_exact/t_approx:.1f}x slower than approx")
    print(f"    Prob (naive) is {t_exact/t_prob_naive:.1f}x faster than exact")
    print("\n  Bounds comparison:")
    print(f"    Exact:              [{exact_lb[0]:.2f}, {exact_ub[0]:.2f}] x [{exact_lb[1]:.2f}, {exact_ub[1]:.2f}]")
    print(f"    Approx:             [{approx_lb[0]:.2f}, {approx_ub[0]:.2f}] x [{approx_lb[1]:.2f}, {approx_ub[1]:.2f}]")
    print(f"    Prob (naive):       [{prob_naive_lb[0]:.2f}, {prob_naive_ub[0]:.2f}] x [{prob_naive_lb[1]:.2f}, {prob_naive_ub[1]:.2f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
