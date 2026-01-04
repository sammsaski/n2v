"""
03_verify_pipeline.py - Full Probabilistic Verification Pipeline

This script demonstrates the complete verify() pipeline:
1. Model definition (any callable)
2. Input set specification
3. Running verification with different parameters
4. Interpreting results (ProbabilisticBox)
5. Validating the coverage guarantee empirically
"""

import numpy as np
import torch
import torch.nn as nn

from n2v.probabilistic import verify
from n2v.sets import Box


def main():
    print("=" * 70)
    print("PROBABILISTIC VERIFICATION PIPELINE")
    print("=" * 70)

    # =========================================================================
    # Part 1: Define a Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: DEFINE A MODEL")
    print("=" * 70)

    print("""
The verify() function works with ANY callable model that maps:
  numpy array (batch_size, input_dim) -> numpy array (batch_size, output_dim)

This can be:
- PyTorch model
- TensorFlow/Keras model
- ONNX model
- External API
- Any black-box function
""")

    # Create a simple PyTorch model
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(5, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 3)
    )
    model.eval()

    # Wrap it for numpy interface
    def model_fn(x: np.ndarray) -> np.ndarray:
        """Convert numpy input to torch, run model, convert back."""
        with torch.no_grad():
            x_torch = torch.tensor(x, dtype=torch.float32)
            y_torch = model(x_torch)
            return y_torch.numpy()

    print("Model architecture:")
    print(model)
    print(f"\nInput dimension: 5")
    print(f"Output dimension: 3")

    # Test the model
    test_input = np.random.randn(3, 5).astype(np.float32)
    test_output = model_fn(test_input)
    print(f"\nTest: model_fn(shape {test_input.shape}) -> shape {test_output.shape}")

    # =========================================================================
    # Part 2: Define Input Set
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: DEFINE INPUT SET")
    print("=" * 70)

    print("""
The input set is specified as a Box (hyperrectangle) with lower and upper bounds.
Samples are drawn uniformly from this region.
""")

    # Define input bounds
    lb = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
    ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    input_set = Box(lb, ub)

    print(f"Input set: Box")
    print(f"  Lower bounds: {lb}")
    print(f"  Upper bounds: {ub}")
    print(f"  Dimension: {input_set.dim}")

    # =========================================================================
    # Part 3: Run Verification
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: RUN VERIFICATION")
    print("=" * 70)

    print("""
The verify() function:
1. Samples training points from input_set
2. Fits a surrogate model to the outputs
3. Samples calibration points from input_set
4. Computes nonconformity scores
5. Inflates bounds to achieve the desired coverage
""")

    # Run verification with verbose output
    print("\n--- Running verify() with verbose=True ---\n")

    result = verify(
        model=model_fn,
        input_set=input_set,
        m=500,                   # Calibration samples
        ell=None,                # Default: m-1
        epsilon=0.05,            # 95% coverage
        surrogate='clipping_block',  # Tighter bounds
        training_samples=250,    # Samples for surrogate
        batch_size=100,
        seed=42,
        verbose=True
    )

    # =========================================================================
    # Part 4: Interpret Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: INTERPRET RESULTS")
    print("=" * 70)

    print("""
The result is a ProbabilisticBox, which is a Box with probabilistic guarantees.
""")

    print(f"\nResult type: {type(result).__name__}")
    print(f"\nOutput bounds:")
    print(f"  Lower bound: {result.lb.flatten()}")
    print(f"  Upper bound: {result.ub.flatten()}")
    print(f"  Bound widths: {(result.ub - result.lb).flatten()}")

    print(f"\nGuarantee parameters:")
    print(f"  m (calibration size): {result.m}")
    print(f"  ℓ (rank parameter): {result.ell}")
    print(f"  ε (miscoverage level): {result.epsilon}")

    print(f"\nComputed guarantees:")
    print(f"  Coverage δ₁ = 1 - ε = {result.coverage:.4f}")
    print(f"  Confidence δ₂ = {result.confidence:.6f}")

    print(f"\nGuarantee string:")
    print(f"  {result.get_guarantee_string()}")

    # =========================================================================
    # Part 5: Empirical Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 5: EMPIRICAL VALIDATION")
    print("=" * 70)

    print("""
Let's validate that the coverage guarantee holds empirically.
We'll sample many points from the input set and check what fraction
of outputs fall within the computed bounds.
""")

    # Sample many test points
    n_test = 10000
    np.random.seed(123)  # Different seed than verification

    test_inputs = np.random.uniform(lb, ub, size=(n_test, 5)).astype(np.float32)
    test_outputs = model_fn(test_inputs)

    # Check how many are inside the bounds
    lb_result = result.lb.flatten()
    ub_result = result.ub.flatten()

    inside = np.all((test_outputs >= lb_result) & (test_outputs <= ub_result), axis=1)
    empirical_coverage = np.mean(inside)

    print(f"Empirical validation with {n_test} test samples:")
    print(f"  Samples inside bounds: {np.sum(inside)}/{n_test}")
    print(f"  Empirical coverage: {empirical_coverage:.4f}")
    print(f"  Expected coverage: {result.coverage:.4f}")
    print(f"  Difference: {empirical_coverage - result.coverage:+.4f}")

    if empirical_coverage >= result.coverage:
        print("\n  ✓ Coverage guarantee is satisfied!")
    else:
        print(f"\n  Note: Empirical coverage ({empirical_coverage:.4f}) is slightly below")
        print(f"  expected ({result.coverage:.4f}). This can happen with probability")
        print(f"  1 - δ₂ = {1 - result.confidence:.6f} (about 1 in {int(1/(1-result.confidence))})")

    # =========================================================================
    # Part 6: Compare Parameters
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 6: PARAMETER SENSITIVITY")
    print("=" * 70)

    print("""
Let's see how different parameters affect the results.
""")

    # Compare different m values
    print("\n--- Effect of calibration size m ---")
    print(f"{'m':>8} {'Width':>12} {'Confidence':>12}")
    print("-" * 35)

    for m in [100, 250, 500, 1000]:
        result_m = verify(
            model=model_fn,
            input_set=input_set,
            m=m,
            epsilon=0.05,
            surrogate='naive',  # Fast for this comparison
            seed=42,
            verbose=False
        )
        avg_width = np.mean(result_m.ub - result_m.lb)
        print(f"{m:>8} {avg_width:>12.4f} {result_m.confidence:>12.6f}")

    print("\nNote: Larger m gives tighter bounds and higher confidence.")

    # Compare different epsilon values
    print("\n--- Effect of miscoverage level ε ---")
    print(f"{'ε':>8} {'Coverage':>10} {'Width':>12} {'Confidence':>12}")
    print("-" * 45)

    for eps in [0.001, 0.01, 0.05, 0.1]:
        result_eps = verify(
            model=model_fn,
            input_set=input_set,
            m=500,
            epsilon=eps,
            surrogate='naive',
            seed=42,
            verbose=False
        )
        avg_width = np.mean(result_eps.ub - result_eps.lb)
        print(f"{eps:>8.3f} {result_eps.coverage:>10.3f} {avg_width:>12.4f} {result_eps.confidence:>12.6f}")

    print("\nNote: Smaller ε (higher coverage) gives wider bounds but lower confidence.")

    # Compare surrogates
    print("\n--- Surrogate comparison ---")

    result_naive = verify(
        model=model_fn,
        input_set=input_set,
        m=500,
        epsilon=0.05,
        surrogate='naive',
        seed=42,
        verbose=False
    )

    result_clip = verify(
        model=model_fn,
        input_set=input_set,
        m=500,
        epsilon=0.05,
        surrogate='clipping_block',
        training_samples=250,
        seed=42,
        verbose=False
    )

    naive_width = np.mean(result_naive.ub - result_naive.lb)
    clip_width = np.mean(result_clip.ub - result_clip.lb)

    print(f"Naive surrogate:    avg width = {naive_width:.4f}")
    print(f"Clipping block:     avg width = {clip_width:.4f}")
    print(f"Improvement:        {(naive_width - clip_width) / naive_width * 100:.1f}%")

    # =========================================================================
    # Part 7: Using ProbabilisticBox
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 7: USING PROBABILISTICBOX")
    print("=" * 70)

    print("""
ProbabilisticBox inherits from Box, so you can use all Box methods.
""")

    print(f"\nProbabilisticBox methods:")
    print(f"  result.dim = {result.dim}")
    print(f"  result.get_range() = {result.get_range()}")
    print(f"  result.contains(point) = {result.contains(test_outputs[0])}")

    # Sample from the box
    samples = result.sample(5)
    print(f"\n  result.sample(5):")
    for i, s in enumerate(samples):
        print(f"    Sample {i}: {s}")

    # Convert to Star (note: loses probabilistic metadata)
    star = result.to_star()
    print(f"\n  result.to_star() -> {type(star).__name__}")
    print(f"    (Warning: Converting loses probabilistic metadata)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The verify() pipeline:

1. INPUTS:
   - model: Any callable (numpy -> numpy)
   - input_set: Box defining input region
   - m, epsilon: Coverage/confidence parameters

2. PROCESS:
   - Sample training data, fit surrogate
   - Sample calibration data, compute scores
   - Run conformal inference

3. OUTPUT:
   - ProbabilisticBox with bounds and guarantees

4. GUARANTEE:
   "With confidence δ₂, at least (1-ε) of outputs are in bounds"

Key parameters:
- Larger m → tighter bounds, higher confidence
- Smaller ε → wider bounds, higher coverage
- clipping_block → tighter than naive (but slower)
""")


if __name__ == "__main__":
    main()
