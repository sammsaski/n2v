"""
05_integration.py - Integration with n2v's NeuralNetwork.reach()

This script demonstrates how probabilistic verification integrates with n2v:
1. Using method='probabilistic' with NeuralNetwork.reach()
2. Using method='hybrid' for automatic switching
3. Comparing deterministic vs probabilistic methods
4. When to use each approach
"""

import numpy as np
import torch
import torch.nn as nn
import time

import n2v
from n2v.sets import Star, Box
from n2v.probabilistic import verify


def main():
    print("=" * 70)
    print("N2V INTEGRATION: NeuralNetwork.reach()")
    print("=" * 70)

    # =========================================================================
    # Part 1: Setup Model and Input Set
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: SETUP MODEL AND INPUT SET")
    print("=" * 70)

    # Create a small network for comparison
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )
    model.eval()

    print("Model architecture:")
    print(model)

    # Create n2v verifier
    verifier = n2v.NeuralNetwork(model)
    print(f"\nCreated n2v.NeuralNetwork verifier")

    # Define input region (small perturbation around a point)
    center = np.array([0.5, 0.5, 0.5, 0.5])
    epsilon = 0.1
    lb = center - epsilon
    ub = center + epsilon

    # Create input sets for different methods
    input_star = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))
    input_box = Box(lb, ub)

    print(f"\nInput region: Box centered at {center} with ε={epsilon}")
    print(f"  Lower bound: {lb}")
    print(f"  Upper bound: {ub}")

    # =========================================================================
    # Part 2: Exact Reachability
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: EXACT REACHABILITY (method='exact')")
    print("=" * 70)

    print("""
Exact reachability computes the precise reachable set by:
- Propagating Star sets through each layer
- Splitting at ReLU nodes to handle case analysis
- May produce multiple output Stars (exponential in ReLU count)
""")

    start = time.time()
    exact_result = verifier.reach(input_star, method='exact')
    exact_time = time.time() - start

    print(f"\nExact reachability results:")
    print(f"  Number of output Stars: {len(exact_result)}")
    print(f"  Time: {exact_time:.3f} seconds")

    # Compute overall bounds from all Stars
    # Note: Use get_ranges() for LP-based exact bounds, not estimate_ranges()
    # which is a fast over-approximation that doesn't account for constraints
    all_lb = []
    all_ub = []
    for star in exact_result:
        star_lb, star_ub = star.get_ranges()  # LP-based exact bounds
        all_lb.append(star_lb.flatten())
        all_ub.append(star_ub.flatten())

    exact_lb = np.min(np.stack(all_lb), axis=0)
    exact_ub = np.max(np.stack(all_ub), axis=0)

    print(f"\n  Output bounds:")
    print(f"    Lower: {exact_lb}")
    print(f"    Upper: {exact_ub}")
    print(f"    Width: {exact_ub - exact_lb}")

    # =========================================================================
    # Part 3: Approximate Reachability
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: APPROXIMATE REACHABILITY (method='approx')")
    print("=" * 70)

    print("""
Approximate reachability avoids ReLU splitting by over-approximating:
- Uses convex relaxation at ReLU nodes
- Produces a single output set (no exponential blowup)
- Sound but not complete (bounds may be conservative)
""")

    start = time.time()
    approx_result = verifier.reach(input_star, method='approx')
    approx_time = time.time() - start

    approx_star = approx_result[0]
    approx_lb, approx_ub = approx_star.get_ranges()  # LP-based exact bounds
    approx_lb = approx_lb.flatten()
    approx_ub = approx_ub.flatten()

    print(f"\nApproximate reachability results:")
    print(f"  Number of output Stars: {len(approx_result)}")
    print(f"  Time: {approx_time:.3f} seconds")
    print(f"\n  Output bounds:")
    print(f"    Lower: {approx_lb}")
    print(f"    Upper: {approx_ub}")
    print(f"    Width: {approx_ub - approx_lb}")

    # Compare to exact - soundness check
    # Approximate should OVER-approximate: approx_lb <= exact_lb, approx_ub >= exact_ub
    print(f"\n  Comparison to exact (soundness check):")
    print(f"    Over-approximation in lower: {exact_lb - approx_lb}  (should be >= 0)")
    print(f"    Over-approximation in upper: {approx_ub - exact_ub}  (should be >= 0)")

    # =========================================================================
    # Part 4: Probabilistic Reachability
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: PROBABILISTIC REACHABILITY (method='probabilistic')")
    print("=" * 70)

    print("""
Probabilistic reachability uses conformal inference:
- Model-agnostic (treats network as black box)
- Provides coverage guarantee (e.g., 99% of outputs in bounds)
- Constant time regardless of network size
""")

    start = time.time()
    prob_result = verifier.reach(
        input_box,  # Note: uses Box for probabilistic
        method='probabilistic',
        m=500,
        epsilon=0.05,  # 95% coverage
        surrogate='naive',
        seed=42,
        verbose=True
    )
    prob_time = time.time() - start

    prob_box = prob_result[0]

    print(f"\nProbabilistic reachability results:")
    print(f"  Time: {prob_time:.3f} seconds")
    print(f"\n  Output bounds:")
    print(f"    Lower: {prob_box.lb.flatten()}")
    print(f"    Upper: {prob_box.ub.flatten()}")
    print(f"    Width: {(prob_box.ub - prob_box.lb).flatten()}")
    print(f"\n  Guarantees:")
    print(f"    Coverage: {prob_box.coverage:.2f}")
    print(f"    Confidence: {prob_box.confidence:.4f}")
    print(f"    {prob_box.get_guarantee_string()}")

    # =========================================================================
    # Part 5: Hybrid Method
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 5: HYBRID METHOD (method='hybrid')")
    print("=" * 70)

    print("""
Hybrid method starts with exact/approximate and switches to probabilistic:
- Tries deterministic reachability layer by layer
- Switches to probabilistic if:
  - Star count exceeds max_stars
  - Layer computation exceeds timeout_per_layer
- Best of both worlds for medium-sized networks
""")

    start = time.time()
    hybrid_result = verifier.reach(
        input_star,
        method='hybrid',
        max_stars=500,           # Switch if > 500 stars
        timeout_per_layer=10.0,  # Switch if layer > 10 seconds
        m=500,
        epsilon=0.05,
        surrogate='naive',
        verbose=True
    )
    hybrid_time = time.time() - start

    print(f"\nHybrid reachability results:")
    print(f"  Time: {hybrid_time:.3f} seconds")
    print(f"  Output type: {type(hybrid_result[0]).__name__}")

    if hasattr(hybrid_result[0], 'coverage'):
        # Switched to probabilistic
        print(f"  (Switched to probabilistic mode)")
        print(f"  Coverage: {hybrid_result[0].coverage}")
    else:
        # Stayed deterministic
        print(f"  (Completed in deterministic mode)")
        print(f"  Number of Stars: {len(hybrid_result)}")

    # =========================================================================
    # Part 6: Method Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 6: METHOD COMPARISON")
    print("=" * 70)

    print("\n--- Results Summary ---")
    print(f"{'Method':<20} {'Time':>10} {'Bound Width':>15} {'Guarantee':>25}")
    print("-" * 75)

    exact_width = np.mean(exact_ub - exact_lb)
    print(f"{'Exact':<20} {exact_time:>10.3f}s {exact_width:>15.4f} {'Sound & Complete':>25}")

    approx_width = np.mean(approx_ub - approx_lb)
    print(f"{'Approx':<20} {approx_time:>10.3f}s {approx_width:>15.4f} {'Sound (over-approx)':>25}")

    prob_width = np.mean(prob_box.ub - prob_box.lb)
    print(f"{'Probabilistic':<20} {prob_time:>10.3f}s {prob_width:>15.4f} {f'{prob_box.coverage:.0%} coverage (not sound)':>25}")

    hybrid_out = hybrid_result[0]
    if hasattr(hybrid_out, 'coverage'):
        hybrid_width = np.mean(hybrid_out.ub - hybrid_out.lb)
        hybrid_guar = f'{hybrid_out.coverage:.0%} coverage (not sound)'
    else:
        hb_lb, hb_ub = hybrid_out.get_ranges()  # LP-based exact bounds
        hybrid_width = np.mean(hb_ub - hb_lb)
        hybrid_guar = 'Sound'
    print(f"{'Hybrid':<20} {hybrid_time:>10.3f}s {hybrid_width:>15.4f} {hybrid_guar:>25}")

    # =========================================================================
    # Part 7: When to Use Each Method
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 7: WHEN TO USE EACH METHOD")
    print("=" * 70)

    print("""
DECISION GUIDE:

┌────────────────────────────────────────────────────────────────────┐
│                         NETWORK SIZE                                │
├──────────────────┬─────────────────┬───────────────────────────────┤
│      Small       │     Medium      │           Large               │
│   (<100 ReLUs)   │  (100-1000)     │        (>1000 ReLUs)          │
├──────────────────┼─────────────────┼───────────────────────────────┤
│                  │                 │                               │
│  method='exact'  │ method='hybrid' │ method='probabilistic'        │
│                  │                 │                               │
│  - Precise       │ - Tries exact   │ - Constant time               │
│  - Sound         │ - Falls back to │ - Model-agnostic              │
│  - Complete      │   probabilistic │ - Coverage guarantee          │
│                  │   when needed   │                               │
└──────────────────┴─────────────────┴───────────────────────────────┘

USE CASES:

method='exact':
  ✓ Safety-critical applications requiring soundness
  ✓ Small networks (few ReLUs)
  ✓ Need precise bounds
  ✗ Not suitable for large networks (exponential time)

method='approx':
  ✓ Medium networks where exact is too slow
  ✓ Quick screening before detailed analysis
  ✓ When over-approximation is acceptable
  ✗ Bounds may be too conservative

method='probabilistic':
  ✓ Large networks where deterministic methods time out
  ✓ Black-box models (no access to architecture)
  ✓ External APIs or non-PyTorch models
  ✓ When high-probability coverage is acceptable
  ✗ NOT SOUND - outputs may fall outside bounds with probability ε

method='hybrid':
  ✓ Unknown network complexity
  ✓ Want to try exact first, fall back gracefully
  ✓ Medium-large networks
  ✗ May switch at unpredictable points
  ✗ If switches to probabilistic, loses soundness guarantee
""")

    # =========================================================================
    # Part 8: Direct verify() Usage
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 8: DIRECT verify() USAGE (For Black-Box Models)")
    print("=" * 70)

    print("""
For maximum flexibility, use verify() directly with any callable:
""")

    # Example with raw callable
    def my_black_box(x):
        """Could be any model - PyTorch, TensorFlow, API, etc."""
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).numpy()

    print("Direct verify() with any callable:")
    print("```python")
    print("from n2v.probabilistic import verify")
    print("from n2v.sets import Box")
    print("")
    print("result = verify(")
    print("    model=my_black_box,  # Any callable: np.array -> np.array")
    print("    input_set=Box(lb, ub),")
    print("    m=1000,")
    print("    epsilon=0.01,")
    print("    surrogate='clipping_block'")
    print(")")
    print("```")

    # Actually run it
    direct_result = verify(
        model=my_black_box,
        input_set=input_box,
        m=500,
        epsilon=0.05,
        surrogate='clipping_block',
        training_samples=250,
        seed=42,
        verbose=False
    )

    print(f"\nDirect verify() result:")
    print(f"  Bounds: [{direct_result.lb.flatten()}, {direct_result.ub.flatten()}]")
    print(f"  Coverage: {direct_result.coverage}")
    print(f"  Confidence: {direct_result.confidence:.6f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
n2v Integration Options:

1. verifier.reach(input, method='exact')
   - Precise, sound, complete
   - Best for small networks

2. verifier.reach(input, method='approx')
   - Over-approximate, sound
   - Best for medium networks

3. verifier.reach(input, method='probabilistic', m=..., epsilon=...)
   - Coverage guarantee, model-agnostic
   - Best for large networks

4. verifier.reach(input, method='hybrid', max_stars=..., timeout_per_layer=...)
   - Automatic fallback
   - Best when complexity is unknown

5. verify(model, input_set, ...) - Direct call
   - Maximum flexibility
   - Works with any callable model
""")


if __name__ == "__main__":
    main()
