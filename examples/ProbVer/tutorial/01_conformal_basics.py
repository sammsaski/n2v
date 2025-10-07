"""
01_conformal_basics.py - Understanding Conformal Inference Fundamentals

This script demonstrates the core conformal inference components:
1. Confidence computation using the Beta CDF formula
2. Normalization (tau) computation from training errors
3. Nonconformity score calculation
4. Threshold selection (R_ell)
5. Inflation computation

These are the building blocks of probabilistic verification.
"""

import numpy as np
from scipy.stats import beta

# Import conformal inference primitives from n2v
from n2v.probabilistic import (
    compute_confidence,
    compute_normalization,
    compute_nonconformity_scores,
    compute_threshold,
    compute_inflation,
    conformal_inference,
    ConformalGuarantee
)


def main():
    print("=" * 70)
    print("CONFORMAL INFERENCE BASICS")
    print("=" * 70)

    # =========================================================================
    # Part 1: The ⟨ε, ℓ, m⟩ Guarantee Framework
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: THE ⟨ε, ℓ, m⟩ GUARANTEE FRAMEWORK")
    print("=" * 70)

    print("""
The probabilistic verification provides guarantees parameterized by:
  - m: Number of calibration samples
  - ℓ: Rank parameter (which sorted score to use as threshold)
  - ε: Miscoverage level (probability that output falls outside bounds)

The guarantee is:
  With probability δ₂ (confidence), at least 1-ε (coverage) of outputs
  from the input set will be contained in the computed bounds.

Where:
  δ₁ = 1 - ε          (coverage)
  δ₂ = 1 - betacdf(1-ε; ℓ, m+1-ℓ)  (confidence)
""")

    # =========================================================================
    # Part 2: Confidence Computation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: CONFIDENCE COMPUTATION")
    print("=" * 70)

    print("\nThe confidence δ₂ is computed from the incomplete Beta function:")
    print("  δ₂ = 1 - I_{1-ε}(ℓ, m+1-ℓ)")
    print("\nwhere I_x(a,b) is the regularized incomplete beta function.\n")

    # Example: varying m with fixed ℓ/m ratio and ε
    print("Example: How confidence varies with m (ℓ = m-1, ε = 0.001)")
    print("-" * 50)
    print(f"{'m':>10} {'ℓ':>10} {'ε':>10} {'Confidence δ₂':>15}")
    print("-" * 50)

    epsilon = 0.001
    for m in [100, 500, 1000, 5000, 8000, 10000]:
        ell = m - 1  # Second largest score
        confidence = compute_confidence(m, ell, epsilon)
        print(f"{m:>10} {ell:>10} {epsilon:>10.4f} {confidence:>15.6f}")

    print("\nNote: Larger m gives higher confidence in the coverage guarantee.")

    # Example: varying ε
    print("\n\nExample: How confidence varies with ε (m = 1000, ℓ = 999)")
    print("-" * 50)
    print(f"{'ε':>10} {'Coverage 1-ε':>15} {'Confidence δ₂':>15}")
    print("-" * 50)

    m = 1000
    ell = 999
    for epsilon in [0.001, 0.005, 0.01, 0.05, 0.1]:
        confidence = compute_confidence(m, ell, epsilon)
        coverage = 1 - epsilon
        print(f"{epsilon:>10.3f} {coverage:>15.3f} {confidence:>15.6f}")

    print("\nNote: Smaller ε (higher coverage) gives lower confidence.")

    # =========================================================================
    # Part 3: Normalization (tau) Computation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: NORMALIZATION (τ) COMPUTATION")
    print("=" * 70)

    print("""
The normalization vector τ scales errors dimension-wise:
  τ[k] = max(τ*, max_j |training_errors[j, k]|)

where τ* is a small value to prevent division by zero.
This ensures all dimensions contribute equally to the nonconformity score.
""")

    # Create example training errors with different scales
    np.random.seed(42)
    t, n = 100, 4  # 100 training samples, 4 output dimensions

    # Simulate training errors with different magnitudes per dimension
    training_errors = np.zeros((t, n))
    training_errors[:, 0] = np.random.randn(t) * 0.1   # Small errors
    training_errors[:, 1] = np.random.randn(t) * 1.0   # Medium errors
    training_errors[:, 2] = np.random.randn(t) * 10.0  # Large errors
    training_errors[:, 3] = np.random.randn(t) * 0.01  # Very small errors

    print("Training errors statistics:")
    print(f"  Dimension 0: max abs = {np.max(np.abs(training_errors[:, 0])):.4f}")
    print(f"  Dimension 1: max abs = {np.max(np.abs(training_errors[:, 1])):.4f}")
    print(f"  Dimension 2: max abs = {np.max(np.abs(training_errors[:, 2])):.4f}")
    print(f"  Dimension 3: max abs = {np.max(np.abs(training_errors[:, 3])):.4f}")

    tau = compute_normalization(training_errors)

    print(f"\nComputed normalization τ:")
    print(f"  τ = {tau}")
    print("\nNote: τ[k] captures the typical error scale in dimension k.")

    # =========================================================================
    # Part 4: Nonconformity Scores
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: NONCONFORMITY SCORES")
    print("=" * 70)

    print("""
The nonconformity score measures how "unusual" each calibration sample is:
  R_i = max_k( |error_i[k]| / τ[k] )

This is the normalized L∞ distance - the worst-case deviation across dimensions.
""")

    # Create example calibration errors
    m = 20  # 20 calibration samples
    calibration_errors = np.zeros((m, n))
    calibration_errors[:, 0] = np.random.randn(m) * 0.1
    calibration_errors[:, 1] = np.random.randn(m) * 1.0
    calibration_errors[:, 2] = np.random.randn(m) * 10.0
    calibration_errors[:, 3] = np.random.randn(m) * 0.01

    # Add a few outliers
    calibration_errors[5, 2] = 30.0   # Large outlier in dim 2
    calibration_errors[10, 0] = 0.5   # Outlier in dim 0 (5x typical)

    scores = compute_nonconformity_scores(calibration_errors, tau)

    print("Calibration errors (first 5 samples):")
    for i in range(5):
        print(f"  Sample {i}: {calibration_errors[i]}")
    print(f"  ...")
    print(f"  Sample 5 (outlier): {calibration_errors[5]}")
    print(f"  ...")
    print(f"  Sample 10 (outlier): {calibration_errors[10]}")

    print(f"\nNonconformity scores R (sorted):")
    sorted_scores = np.sort(scores)
    for i, s in enumerate(sorted_scores):
        marker = " (outlier)" if s > 2.0 else ""
        print(f"  R[{i+1}] = {s:.4f}{marker}")

    # =========================================================================
    # Part 5: Threshold Selection
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 5: THRESHOLD SELECTION (R_ℓ)")
    print("=" * 70)

    print("""
The threshold R_ℓ is the ℓ-th smallest nonconformity score.
  - ℓ = m: Use the largest score (most conservative)
  - ℓ = m-1: Use the second largest (typical choice)
  - Smaller ℓ: Tighter bounds but lower coverage guarantee

The threshold determines how much to inflate the surrogate bounds.
""")

    print("Threshold values for different ℓ:")
    print("-" * 40)
    print(f"{'ℓ':>5} {'R_ℓ':>10} {'Interpretation':>20}")
    print("-" * 40)

    for ell in [m, m-1, m-2, int(m*0.9), int(m*0.5)]:
        threshold = compute_threshold(scores, ell)
        pct = (ell / m) * 100
        print(f"{ell:>5} {threshold:>10.4f} {f'Top {100-pct:.0f}% excluded':>20}")

    # =========================================================================
    # Part 6: Inflation Computation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 6: INFLATION COMPUTATION")
    print("=" * 70)

    print("""
The inflation σ determines how much to expand the bounds:
  σ[k] = τ[k] * R_ℓ

Final bounds = surrogate_bounds ± σ
""")

    ell = m - 1
    threshold = compute_threshold(scores, ell)
    inflation = compute_inflation(tau, threshold)

    print(f"Using ℓ = {ell}, threshold R_ℓ = {threshold:.4f}")
    print(f"\nInflation per dimension:")
    print(f"  σ = τ * R_ℓ")
    for k in range(n):
        print(f"  σ[{k}] = {tau[k]:.4f} * {threshold:.4f} = {inflation[k]:.4f}")

    # =========================================================================
    # Part 7: Full Conformal Inference
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 7: FULL CONFORMAL INFERENCE (END-TO-END)")
    print("=" * 70)

    print("""
The conformal_inference() function combines all steps:
1. Compute normalization τ from training errors
2. Compute nonconformity scores for calibration samples
3. Select threshold R_ℓ (ℓ-th smallest score)
4. Compute inflation σ = τ * R_ℓ
5. Compute coverage and confidence guarantees
""")

    # Generate more realistic data
    np.random.seed(123)
    t = 500   # Training samples
    m = 1000  # Calibration samples
    n = 5     # Output dimensions
    ell = m - 1
    epsilon = 0.01

    # Simulate prediction errors (surrogate model errors)
    training_errors = np.random.randn(t, n) * np.array([0.1, 0.5, 1.0, 2.0, 0.05])
    calibration_errors = np.random.randn(m, n) * np.array([0.1, 0.5, 1.0, 2.0, 0.05])

    guarantee = conformal_inference(
        training_errors=training_errors,
        calibration_errors=calibration_errors,
        m=m,
        ell=ell,
        epsilon=epsilon
    )

    print(f"Parameters: m={m}, ℓ={ell}, ε={epsilon}")
    print(f"\nConformalGuarantee result:")
    print(f"  Coverage δ₁ = 1 - ε = {guarantee.coverage:.4f}")
    print(f"  Confidence δ₂ = {guarantee.confidence:.6f}")
    print(f"  Threshold R_ℓ = {guarantee.threshold:.4f}")
    print(f"  Inflation σ = {guarantee.inflation}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The conformal inference pipeline:
1. τ (normalization) - scales errors to be dimension-agnostic
2. R_i (scores) - measures how unusual each calibration sample is
3. R_ℓ (threshold) - the score that covers (ℓ/m) of calibration samples
4. σ (inflation) - how much to expand bounds: σ = τ * R_ℓ

The guarantee: With confidence δ₂, at least (1-ε) of future samples
will fall within bounds inflated by σ.
""")


if __name__ == "__main__":
    main()
