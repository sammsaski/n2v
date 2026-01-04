"""
showcase_speed.py - Probabilistic Verification Speed Advantage

This script demonstrates when probabilistic verification outperforms
deterministic methods. As network size grows, deterministic methods
slow down exponentially while probabilistic stays constant.

Key insight: Probabilistic verification time depends only on the number
of samples (m), NOT on network architecture.
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


def benchmark_methods(model, lb, ub, timeout=30.0, skip_exact=False):
    """
    Benchmark different reachability methods on a model.

    Returns dict with timing results and whether method completed.
    """
    import signal

    results = {}

    # Create input sets
    input_star = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))
    input_box = Box(lb, ub)

    verifier = n2v.NeuralNetwork(model)

    # Timeout handler
    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError()

    # Exact method
    if skip_exact:
        results['exact'] = {'time': None, 'completed': False, 'timeout': True}
    else:
        start = time.time()
        try:
            # Set timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

            exact_result = verifier.reach(input_star, method='exact')
            exact_time = time.time() - start

            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)

            results['exact'] = {'time': exact_time, 'completed': True, 'timeout': False}
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            results['exact'] = {'time': None, 'completed': False, 'timeout': True}
        except Exception as e:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            results['exact'] = {'time': None, 'completed': False, 'error': str(e)}

    # Approx method
    start = time.time()
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        approx_result = verifier.reach(input_star, method='approx')
        approx_time = time.time() - start

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        results['approx'] = {'time': approx_time, 'completed': True, 'timeout': False}
    except TimeoutError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        results['approx'] = {'time': None, 'completed': False, 'timeout': True}
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        results['approx'] = {'time': None, 'completed': False, 'error': str(e)}

    # Probabilistic method
    def model_fn(x):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).numpy()

    start = time.time()
    try:
        prob_result = verify(
            model=model_fn,
            input_set=input_box,
            m=500,
            epsilon=0.05,
            surrogate='naive',
            seed=42,
            verbose=False
        )
        prob_time = time.time() - start
        results['probabilistic'] = {'time': prob_time, 'completed': True, 'timeout': False}
    except Exception as e:
        results['probabilistic'] = {'time': None, 'completed': False, 'error': str(e)}

    return results


def main():
    print("=" * 70)
    print("PROBABILISTIC VERIFICATION: SPEED ADVANTAGE")
    print("=" * 70)

    print("""
This showcase demonstrates the key speed advantage of probabilistic
verification: its runtime is INDEPENDENT of network architecture.

As networks grow deeper and wider, exact methods slow exponentially
(due to ReLU splitting), while probabilistic time stays constant.
""")

    # =========================================================================
    # Experiment 1: Varying Network Depth
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: VARYING NETWORK DEPTH")
    print("=" * 70)

    print("""
We'll benchmark networks with increasing depth (1-6 hidden layers).
Input: 4 dimensions, Hidden: 64 neurons, Output: 2 dimensions.
Perturbation: epsilon = 0.1 around center point.
Timeout: 30 seconds per method.

With 64 neurons per layer, each ReLU layer has more potential splits,
making the scaling behavior more pronounced.
""")

    torch.manual_seed(42)
    np.random.seed(42)

    depths = [1, 2, 3, 4, 5, 6]
    depth_results = []

    center = np.array([0.5, 0.5, 0.5, 0.5])
    epsilon = 0.1
    lb = (center - epsilon).astype(np.float32)
    ub = (center + epsilon).astype(np.float32)

    print(f"{'Depth':>8} {'Exact':>12} {'Approx':>12} {'Prob':>12}")
    print("-" * 50)

    skip_exact = False
    for depth in depths:
        model = create_network(4, 64, depth, 2)  # 64 neurons per layer
        results = benchmark_methods(model, lb, ub, timeout=30.0, skip_exact=skip_exact)
        depth_results.append(results)

        # Format exact time
        if results['exact']['time'] is not None:
            exact_str = f"{results['exact']['time']:.3f}s"
        elif results['exact'].get('timeout', False):
            exact_str = "TIMEOUT"
            skip_exact = True  # Skip exact for remaining deeper networks
        else:
            exact_str = "FAILED"

        # Format approx time
        if results['approx']['time'] is not None:
            approx_str = f"{results['approx']['time']:.3f}s"
        elif results['approx'].get('timeout', False):
            approx_str = "TIMEOUT"
        else:
            approx_str = "FAILED"

        # Format prob time
        if results['probabilistic']['time'] is not None:
            prob_str = f"{results['probabilistic']['time']:.3f}s"
        else:
            prob_str = "FAILED"

        print(f"{depth:>8} {exact_str:>12} {approx_str:>12} {prob_str:>12}")

    # =========================================================================
    # Experiment 2: Varying Network Width
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: VARYING NETWORK WIDTH")
    print("=" * 70)

    print("""
We'll benchmark networks with increasing width (8-128 neurons per layer).
Input: 4 dimensions, Depth: 2 hidden layers, Output: 2 dimensions.
Timeout: 30 seconds per method.
""")

    widths = [8, 16, 32, 64, 128]
    width_results = []

    print(f"{'Width':>8} {'Exact':>12} {'Approx':>12} {'Prob':>12}")
    print("-" * 50)

    skip_exact_width = False
    for width in widths:
        model = create_network(4, width, 2, 2)
        results = benchmark_methods(model, lb, ub, timeout=30.0, skip_exact=skip_exact_width)
        width_results.append(results)

        # Format exact time
        if results['exact']['time'] is not None:
            exact_str = f"{results['exact']['time']:.3f}s"
        elif results['exact'].get('timeout', False):
            exact_str = "TIMEOUT"
            skip_exact_width = True
        else:
            exact_str = "FAILED"

        # Format approx time
        if results['approx']['time'] is not None:
            approx_str = f"{results['approx']['time']:.3f}s"
        elif results['approx'].get('timeout', False):
            approx_str = "TIMEOUT"
        else:
            approx_str = "FAILED"

        # Format prob time
        if results['probabilistic']['time'] is not None:
            prob_str = f"{results['probabilistic']['time']:.3f}s"
        else:
            prob_str = "FAILED"

        print(f"{width:>8} {exact_str:>12} {approx_str:>12} {prob_str:>12}")

    # =========================================================================
    # Experiment 3: Large Network (Probabilistic Advantage)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: LARGE NETWORK (PROBABILISTIC SHINES)")
    print("=" * 70)

    print("""
For a larger network (4 layers, 64 neurons each), exact reachability
becomes very slow due to exponential ReLU splitting.
""")

    model = create_network(8, 64, 4, 4)

    center = np.array([0.5] * 8)
    epsilon = 0.05  # Smaller perturbation to give exact a chance
    lb = (center - epsilon).astype(np.float32)
    ub = (center + epsilon).astype(np.float32)

    print(f"\nNetwork: 8 -> 64 -> 64 -> 64 -> 64 -> 4")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Perturbation: epsilon = {epsilon}")

    # Time each method
    input_star = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))
    input_box = Box(lb, ub)
    verifier = n2v.NeuralNetwork(model)

    print("\n--- Approximate Reachability ---")
    start = time.time()
    approx_result = verifier.reach(input_star, method='approx')
    approx_time = time.time() - start
    approx_lb, approx_ub = approx_result[0].get_ranges()
    print(f"Time: {approx_time:.3f}s")
    print(f"Bound width: {np.mean(approx_ub - approx_lb):.4f}")

    print("\n--- Probabilistic Reachability ---")
    def model_fn(x):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).numpy()

    start = time.time()
    prob_result = verify(
        model=model_fn,
        input_set=input_box,
        m=1000,
        epsilon=0.05,
        surrogate='clipping_block',
        training_samples=500,
        seed=42,
        verbose=False
    )
    prob_time = time.time() - start
    print(f"Time: {prob_time:.3f}s")
    print(f"Bound width: {np.mean(prob_result.ub - prob_result.lb):.4f}")
    print(f"Coverage: {prob_result.coverage:.0%}")
    print(f"Confidence: {prob_result.confidence:.6f}")

    print("\n--- Exact Reachability (Limited Time) ---")
    print("Running exact with 30s timeout...")
    start = time.time()

    # Use hybrid to see how many stars accumulate
    hybrid_result = verifier.reach(
        input_star,
        method='hybrid',
        max_stars=500,
        timeout_per_layer=10.0,
        m=1000,
        epsilon=0.05,
        verbose=True
    )
    hybrid_time = time.time() - start

    print(f"\nHybrid completed in {hybrid_time:.3f}s")
    if hasattr(hybrid_result[0], 'coverage'):
        print("  -> Switched to probabilistic (exact was too slow)")
    else:
        print(f"  -> Completed deterministically with {len(hybrid_result)} stars")

    # =========================================================================
    # Generate Visualization
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Time vs Depth (line plot for better trend visualization)
        ax1 = axes[0]

        # Extract times, using None for timeouts
        exact_times = [r['exact']['time'] for r in depth_results]
        approx_times = [r['approx']['time'] for r in depth_results]
        prob_times = [r['probabilistic']['time'] for r in depth_results]

        # Plot lines, skipping None values
        valid_exact = [(d, t) for d, t in zip(depths, exact_times) if t is not None]
        valid_approx = [(d, t) for d, t in zip(depths, approx_times) if t is not None]
        valid_prob = [(d, t) for d, t in zip(depths, prob_times) if t is not None]

        if valid_exact:
            ax1.plot([x[0] for x in valid_exact], [x[1] for x in valid_exact],
                    'o-', label='Exact', color='#2ecc71', linewidth=2, markersize=8)
        if valid_approx:
            ax1.plot([x[0] for x in valid_approx], [x[1] for x in valid_approx],
                    's-', label='Approx', color='#3498db', linewidth=2, markersize=8)
        if valid_prob:
            ax1.plot([x[0] for x in valid_prob], [x[1] for x in valid_prob],
                    '^-', label='Probabilistic', color='#e74c3c', linewidth=2, markersize=8)

        # Mark timeouts with X markers at the timeout threshold
        timeout_threshold = 30.0
        timeout_depths_exact = [d for d, t in zip(depths, exact_times) if t is None]
        timeout_depths_approx = [d for d, t in zip(depths, approx_times) if t is None]

        if timeout_depths_exact:
            ax1.scatter(timeout_depths_exact, [timeout_threshold] * len(timeout_depths_exact),
                       marker='x', color='#2ecc71', s=100, linewidths=3, zorder=5)
        if timeout_depths_approx:
            ax1.scatter(timeout_depths_approx, [timeout_threshold] * len(timeout_depths_approx),
                       marker='x', color='#3498db', s=100, linewidths=3, zorder=5)

        # Add timeout line
        ax1.axhline(y=timeout_threshold, color='gray', linestyle='--', alpha=0.5, label='Timeout (30s)')

        ax1.set_xlabel('Number of Hidden Layers')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Verification Time vs Network Depth\n(64 neurons per layer)')
        ax1.set_xticks(depths)
        ax1.legend(loc='upper left')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=1e-4)

        # Plot 2: Time vs Width
        ax2 = axes[1]

        exact_times = [r['exact']['time'] for r in width_results]
        approx_times = [r['approx']['time'] for r in width_results]
        prob_times = [r['probabilistic']['time'] for r in width_results]

        valid_exact = [(w, t) for w, t in zip(widths, exact_times) if t is not None]
        valid_approx = [(w, t) for w, t in zip(widths, approx_times) if t is not None]
        valid_prob = [(w, t) for w, t in zip(widths, prob_times) if t is not None]

        if valid_exact:
            ax2.plot([x[0] for x in valid_exact], [x[1] for x in valid_exact],
                    'o-', label='Exact', color='#2ecc71', linewidth=2, markersize=8)
        if valid_approx:
            ax2.plot([x[0] for x in valid_approx], [x[1] for x in valid_approx],
                    's-', label='Approx', color='#3498db', linewidth=2, markersize=8)
        if valid_prob:
            ax2.plot([x[0] for x in valid_prob], [x[1] for x in valid_prob],
                    '^-', label='Probabilistic', color='#e74c3c', linewidth=2, markersize=8)

        # Mark timeouts
        timeout_widths_exact = [w for w, t in zip(widths, exact_times) if t is None]
        timeout_widths_approx = [w for w, t in zip(widths, approx_times) if t is None]

        if timeout_widths_exact:
            ax2.scatter(timeout_widths_exact, [timeout_threshold] * len(timeout_widths_exact),
                       marker='x', color='#2ecc71', s=100, linewidths=3, zorder=5)
        if timeout_widths_approx:
            ax2.scatter(timeout_widths_approx, [timeout_threshold] * len(timeout_widths_approx),
                       marker='x', color='#3498db', s=100, linewidths=3, zorder=5)

        ax2.axhline(y=timeout_threshold, color='gray', linestyle='--', alpha=0.5, label='Timeout (30s)')

        ax2.set_xlabel('Neurons per Hidden Layer')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Verification Time vs Network Width\n(2 hidden layers)')
        ax2.set_xticks(widths)
        ax2.legend(loc='upper left')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=1e-4)

        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, 'speed_comparison.png')
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
KEY OBSERVATIONS:

1. EXACT REACHABILITY:
   - Time grows exponentially with network depth (ReLU splitting)
   - May become infeasible for deep networks
   - Provides sound & complete bounds

2. APPROXIMATE REACHABILITY:
   - Linear time growth with network size
   - Bounds may be conservative
   - Provides sound (over-approximate) bounds

3. PROBABILISTIC REACHABILITY:
   - Nearly constant time regardless of network architecture
   - Time depends only on number of samples (m)
   - Provides coverage guarantee (NOT sound)

WHEN TO USE PROBABILISTIC:
   - Large networks (100+ ReLUs)
   - Black-box models (no architecture access)
   - When coverage guarantee (e.g., 95%) is acceptable
   - Initial rapid screening before detailed analysis

WHEN NOT TO USE PROBABILISTIC:
   - Safety-critical applications requiring soundness
   - Small networks where exact is fast enough
   - When 100% containment is required
""")


if __name__ == "__main__":
    main()
