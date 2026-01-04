"""
02_surrogates.py - Comparing Naive and Clipping Block Surrogates

This script demonstrates the two surrogate methods:
1. Naive Surrogate: Uses the center (mean) of training outputs
2. Clipping Block Surrogate: Projects onto convex hull via LP

The surrogate choice affects:
- Bound tightness (clipping block is tighter)
- Computation time (naive is faster)
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from n2v.probabilistic.surrogates.naive import NaiveSurrogate
from n2v.probabilistic.surrogates.clipping_block import (
    ClippingBlockSurrogate,
    BatchedClippingBlockSurrogate
)
from n2v.probabilistic import conformal_inference


def main():
    print("=" * 70)
    print("SURROGATE COMPARISON: NAIVE vs CLIPPING BLOCK")
    print("=" * 70)

    # =========================================================================
    # Part 1: Understanding Surrogates
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: WHAT IS A SURROGATE?")
    print("=" * 70)

    print("""
A surrogate model predicts "typical" outputs for inputs in the region.
The nonconformity score measures deviation from these predictions.

Surrogate types:
1. NAIVE: Predicts the center (mean) for all inputs
   - Fast O(1) prediction
   - May be conservative if outputs are correlated

2. CLIPPING BLOCK: Projects onto convex hull of training outputs
   - Solves an LP per prediction: min ||y - hull||_∞
   - Exploits correlation structure
   - Produces tighter bounds
""")

    # =========================================================================
    # Part 2: Naive Surrogate
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: NAIVE SURROGATE")
    print("=" * 70)

    np.random.seed(42)

    # Create training outputs with correlation
    t = 100  # Training samples
    n = 3    # Output dimensions

    # Outputs are correlated: when dim0 is high, dim1 and dim2 tend to be high
    base = np.random.randn(t, 1) * 2  # Shared component
    noise = np.random.randn(t, n) * 0.5  # Independent noise
    training_outputs = base + noise  # Correlated outputs

    print("Training outputs shape:", training_outputs.shape)
    print("Correlation matrix:")
    print(np.corrcoef(training_outputs.T))

    # Fit naive surrogate
    naive = NaiveSurrogate()
    naive.fit(training_outputs)

    naive_lb, naive_ub = naive.get_bounds()
    print(f"\nNaive surrogate:")
    print(f"  Center = {naive.center}")
    print(f"  Bounds before inflation: [{naive_lb}, {naive_ub}]")
    print(f"  (Note: lb = ub = center for naive surrogate)")

    # Predict for some test points
    test_outputs = np.array([
        [2.0, 2.5, 2.2],   # Near the high end
        [-2.0, -1.8, -2.1], # Near the low end
        [0.0, 0.0, 0.0]    # Near center
    ])

    naive_predictions = naive.predict(test_outputs)
    print(f"\nNaive predictions (always the center):")
    for i, (test, pred) in enumerate(zip(test_outputs, naive_predictions)):
        error = np.max(np.abs(test - pred))
        print(f"  Test {i}: {test} -> Prediction: {pred}, Max error: {error:.4f}")

    # =========================================================================
    # Part 3: Clipping Block Surrogate
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: CLIPPING BLOCK SURROGATE")
    print("=" * 70)

    print("""
The clipping block projects each point onto the convex hull of training outputs.
For a point y, it finds:
  min_{α} ||y - Σ_j α_j * training_j||_∞
  s.t.    Σ_j α_j = 1, α_j >= 0

This is an LP that finds the closest point in the convex hull.
""")

    # Fit clipping block surrogate
    clipping = ClippingBlockSurrogate(n_workers=1, verbose=False)
    clipping.fit(training_outputs)

    clipping_lb, clipping_ub = clipping.get_bounds()
    print(f"Clipping block surrogate:")
    print(f"  Bounds: [{clipping_lb}, {clipping_ub}]")
    print(f"  (These are min/max of training outputs)")

    # Predict for the same test points
    clipping_predictions = clipping.predict(test_outputs)
    print(f"\nClipping block predictions (projected onto hull):")
    for i, (test, pred) in enumerate(zip(test_outputs, clipping_predictions)):
        error = np.max(np.abs(test - pred))
        print(f"  Test {i}: {test}")
        print(f"         -> Projection: {pred}")
        print(f"         -> Max error: {error:.4f}")

    # =========================================================================
    # Part 4: Comparing Prediction Errors
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: COMPARING PREDICTION ERRORS")
    print("=" * 70)

    print("""
The key difference: clipping block produces SMALLER prediction errors
for points that lie within the convex hull of training outputs.
Smaller errors -> smaller nonconformity scores -> tighter bounds!
""")

    # Generate calibration outputs (some inside hull, some outside)
    m = 50
    calib_outputs = np.random.randn(m, n) * 1.5  # Some may be outside hull

    naive_preds = naive.predict(calib_outputs)
    clipping_preds = clipping.predict(calib_outputs)

    naive_errors = np.max(np.abs(calib_outputs - naive_preds), axis=1)
    clipping_errors = np.max(np.abs(calib_outputs - clipping_preds), axis=1)

    print(f"\nPrediction error comparison (m={m} samples):")
    print(f"  Naive - Mean max error: {np.mean(naive_errors):.4f}")
    print(f"  Naive - Max max error:  {np.max(naive_errors):.4f}")
    print(f"  Clipping - Mean max error: {np.mean(clipping_errors):.4f}")
    print(f"  Clipping - Max max error:  {np.max(clipping_errors):.4f}")

    improvement = (np.mean(naive_errors) - np.mean(clipping_errors)) / np.mean(naive_errors) * 100
    print(f"\n  Clipping block reduces mean error by {improvement:.1f}%")

    # =========================================================================
    # Part 5: Impact on Final Bounds
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 5: IMPACT ON FINAL BOUNDS")
    print("=" * 70)

    print("""
Let's run full conformal inference with both surrogates and compare
the final bound widths.
""")

    # Use more realistic sample sizes
    np.random.seed(123)
    t = 200   # Training samples
    m = 500   # Calibration samples
    n = 5     # Output dimensions
    ell = m - 1
    epsilon = 0.01

    # Generate correlated outputs (as would come from a neural network)
    shared = np.random.randn(t + m, 2) * 3
    noise = np.random.randn(t + m, n) * 0.3
    weights = np.random.randn(2, n)
    all_outputs = shared @ weights + noise

    training_outputs = all_outputs[:t]
    calibration_outputs = all_outputs[t:]

    # Naive surrogate
    print("Running with Naive surrogate...")
    naive = NaiveSurrogate()
    naive.fit(training_outputs)

    naive_training_errors = training_outputs - naive.predict(training_outputs)
    naive_calib_errors = calibration_outputs - naive.predict(calibration_outputs)

    naive_guarantee = conformal_inference(
        training_errors=naive_training_errors,
        calibration_errors=naive_calib_errors,
        m=m, ell=ell, epsilon=epsilon
    )

    naive_lb, naive_ub = naive.get_bounds()
    naive_final_lb = naive_lb - naive_guarantee.inflation
    naive_final_ub = naive_ub + naive_guarantee.inflation
    naive_widths = naive_final_ub - naive_final_lb

    # Clipping block surrogate
    print("Running with Clipping Block surrogate...")
    clipping = ClippingBlockSurrogate(n_workers=4, verbose=False)
    clipping.fit(training_outputs)

    clipping_training_errors = training_outputs - clipping.predict(training_outputs)
    clipping_calib_errors = calibration_outputs - clipping.predict(calibration_outputs)

    clipping_guarantee = conformal_inference(
        training_errors=clipping_training_errors,
        calibration_errors=clipping_calib_errors,
        m=m, ell=ell, epsilon=epsilon
    )

    clipping_lb, clipping_ub = clipping.get_bounds()
    clipping_final_lb = clipping_lb - clipping_guarantee.inflation
    clipping_final_ub = clipping_ub + clipping_guarantee.inflation
    clipping_widths = clipping_final_ub - clipping_final_lb

    # Compare
    print(f"\nResults comparison (m={m}, ε={epsilon}):")
    print("-" * 60)
    print(f"{'Dimension':>10} {'Naive Width':>15} {'Clipping Width':>15} {'Reduction':>12}")
    print("-" * 60)
    for k in range(n):
        reduction = (naive_widths[k] - clipping_widths[k]) / naive_widths[k] * 100
        print(f"{k:>10} {naive_widths[k]:>15.4f} {clipping_widths[k]:>15.4f} {reduction:>11.1f}%")

    avg_reduction = (np.mean(naive_widths) - np.mean(clipping_widths)) / np.mean(naive_widths) * 100
    print("-" * 60)
    print(f"{'Average':>10} {np.mean(naive_widths):>15.4f} {np.mean(clipping_widths):>15.4f} {avg_reduction:>11.1f}%")

    print(f"\nConfidence (same for both): {naive_guarantee.confidence:.6f}")

    # =========================================================================
    # Part 6: Timing Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 6: TIMING COMPARISON")
    print("=" * 70)

    print("""
The main trade-off: clipping block produces tighter bounds but is slower
because it solves an LP for each calibration sample.
""")

    # Time both methods
    np.random.seed(456)
    t = 100
    m = 200
    n = 10

    training_outputs = np.random.randn(t, n)
    calibration_outputs = np.random.randn(m, n)

    # Naive timing
    start = time.time()
    naive = NaiveSurrogate()
    naive.fit(training_outputs)
    naive.predict(calibration_outputs)
    naive_time = time.time() - start

    # Clipping block timing
    start = time.time()
    clipping = ClippingBlockSurrogate(n_workers=4, verbose=False)
    clipping.fit(training_outputs)
    clipping.predict(calibration_outputs)
    clipping_time = time.time() - start

    print(f"m={m} calibration samples, n={n} dimensions:")
    print(f"  Naive surrogate: {naive_time:.3f} seconds")
    print(f"  Clipping block:  {clipping_time:.3f} seconds")
    print(f"  Slowdown factor: {clipping_time/naive_time:.1f}x")

    print("""
Recommendations:
- Use 'naive' for quick screening or when tightness isn't critical
- Use 'clipping_block' for final verification or safety-critical applications
- For very large m, consider using BatchedClippingBlockSurrogate
""")

    # =========================================================================
    # Part 7: Visualization (saved to file)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 7: VISUALIZATION")
    print("=" * 70)

    # Create 2D visualization
    np.random.seed(789)
    t = 50
    m = 100

    # Correlated 2D outputs
    angle = np.random.randn(t) * 0.5
    training_2d = np.stack([np.cos(angle) + 0.3*np.random.randn(t),
                            np.sin(angle) + 0.3*np.random.randn(t)], axis=1)

    naive_2d = NaiveSurrogate()
    naive_2d.fit(training_2d)

    clipping_2d = ClippingBlockSurrogate(n_workers=1)
    clipping_2d.fit(training_2d)

    # Test points for visualization
    test_2d = np.random.randn(m, 2) * 1.2

    naive_proj = naive_2d.predict(test_2d)
    clipping_proj = clipping_2d.predict(test_2d)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Naive surrogate
    ax = axes[0]
    ax.scatter(training_2d[:, 0], training_2d[:, 1], c='blue', alpha=0.5, label='Training')
    ax.scatter(test_2d[:, 0], test_2d[:, 1], c='red', alpha=0.3, s=20, label='Test')
    ax.scatter([naive_2d.center[0]], [naive_2d.center[1]], c='green', s=100,
               marker='*', label='Center', zorder=5)
    # Draw lines from test points to center
    for i in range(min(20, m)):
        ax.plot([test_2d[i, 0], naive_2d.center[0]],
                [test_2d[i, 1], naive_2d.center[1]],
                'g-', alpha=0.2, linewidth=0.5)
    ax.set_title('Naive Surrogate\n(all points project to center)')
    ax.legend()
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    ax.axis('equal')

    # Clipping block surrogate
    ax = axes[1]
    ax.scatter(training_2d[:, 0], training_2d[:, 1], c='blue', alpha=0.5, label='Training')
    ax.scatter(test_2d[:, 0], test_2d[:, 1], c='red', alpha=0.3, s=20, label='Test')
    ax.scatter(clipping_proj[:, 0], clipping_proj[:, 1], c='green', alpha=0.3,
               s=20, label='Projections')
    # Draw lines from test points to projections
    for i in range(min(20, m)):
        ax.plot([test_2d[i, 0], clipping_proj[i, 0]],
                [test_2d[i, 1], clipping_proj[i, 1]],
                'g-', alpha=0.3, linewidth=0.5)
    ax.set_title('Clipping Block Surrogate\n(points project to convex hull)')
    ax.legend()
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig('surrogate_comparison.png', dpi=150)
    print("Visualization saved to: surrogate_comparison.png")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Surrogate Choice Summary:

| Surrogate       | Speed  | Tightness | Use Case                    |
|-----------------|--------|-----------|------------------------------|
| Naive           | Fast   | Looser    | Quick screening, simple models |
| Clipping Block  | Slower | Tighter   | Final verification, safety-critical |

The clipping block exploits correlation in outputs to produce tighter bounds.
For independent outputs, the improvement over naive may be smaller.
""")


if __name__ == "__main__":
    main()
