"""
04_deflation_pca.py - Dimensionality Reduction for High-Dimensional Outputs

This script demonstrates DeflationPCA for handling high-dimensional outputs:
1. Why standard PCA fails when samples << dimensions
2. How deflation-based PCA works
3. Using PCA with verify() for image segmentation-like outputs
"""

import numpy as np
import time

from n2v.probabilistic.dimensionality.deflation_pca import DeflationPCA


def main():
    print("=" * 70)
    print("DEFLATION PCA FOR HIGH-DIMENSIONAL OUTPUTS")
    print("=" * 70)

    # =========================================================================
    # Part 1: The Problem with High-Dimensional Outputs
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: THE PROBLEM WITH HIGH-DIMENSIONAL OUTPUTS")
    print("=" * 70)

    print("""
For models with high-dimensional outputs (e.g., semantic segmentation):
- Output dimension n could be 100,000+ (pixels × classes)
- Calibration samples m might be only 1,000-10,000 (computational limit)
- Computing per-dimension normalization τ requires many samples

Problem: When n >> m, we can't reliably estimate per-dimension statistics.

Solution: Reduce dimensionality using PCA, then apply conformal inference
in the lower-dimensional space.
""")

    # Example dimensions
    print("Example scenario:")
    print("  Image: 100 × 100 pixels, 10 classes = 100,000 output dimensions")
    print("  Calibration budget: 5,000 samples")
    print("  Ratio: 5,000 / 100,000 = 0.05 samples per dimension!")
    print("\nWe need dimensionality reduction.")

    # =========================================================================
    # Part 2: Why Standard PCA Fails
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: WHY STANDARD PCA FAILS")
    print("=" * 70)

    print("""
Standard PCA:
1. Compute n×n covariance matrix C = (1/t) X^T X
2. Find eigenvectors of C

Problems when t << n:
- Memory: n×n matrix may not fit in memory
- Singularity: C has rank ≤ t, so only t non-zero eigenvalues
- Numerics: Nearly singular matrices cause numerical issues
""")

    # Demonstrate the memory issue
    t = 500   # Number of samples
    n = 10000  # Output dimension

    print(f"\nExample: t={t} samples, n={n} dimensions")
    print(f"  Standard covariance matrix: {n}×{n} = {n*n:,} floats")
    print(f"  Memory needed: {n*n*8/1e9:.2f} GB (for float64)")
    print(f"  With n=100,000: {100000*100000*8/1e12:.0f} TB!")

    # =========================================================================
    # Part 3: Deflation-Based PCA
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: DEFLATION-BASED PCA")
    print("=" * 70)

    print("""
Deflation PCA finds principal components one at a time:

For each component:
  1. Initialize random direction a
  2. Gradient ascent to maximize variance: max ||X @ a||²
  3. Normalize: a = a / ||a||
  4. Deflate data: X = X - (X @ a) @ a.T

Advantages:
- Memory: O(t × n), not O(n²)
- Works when t << n
- Naturally handles rank-deficient data
""")

    # Create high-dimensional data
    np.random.seed(42)
    t = 200   # Samples
    n = 5000  # Output dimensions
    k = 20    # True latent dimensions

    print(f"\nGenerating synthetic data:")
    print(f"  Samples: {t}")
    print(f"  Dimensions: {n}")
    print(f"  True latent dimensions: {k}")

    # Generate data with low-rank structure
    latent = np.random.randn(t, k)  # Low-dimensional structure
    weights = np.random.randn(k, n) * 0.1  # Map to high dimension
    noise = np.random.randn(t, n) * 0.01  # Small noise
    X = latent @ weights + noise

    print(f"  Data shape: {X.shape}")
    print(f"  Data variance: {np.var(X):.6f}")

    # =========================================================================
    # Part 4: Using DeflationPCA
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: USING DEFLATIONPCA")
    print("=" * 70)

    # Create and fit PCA
    n_components = 50
    pca = DeflationPCA(n_components=n_components, verbose=True)

    print(f"\nFitting DeflationPCA with {n_components} components...")
    start = time.time()
    pca.fit(X)
    fit_time = time.time() - start
    print(f"Fitting took {fit_time:.2f} seconds")

    print(f"\nPCA attributes:")
    print(f"  components_ shape: {pca.components_.shape}")
    print(f"  mean_ shape: {pca.mean_.shape}")

    # Transform data
    X_reduced = pca.transform(X)
    print(f"\nTransformed data:")
    print(f"  Original shape: {X.shape}")
    print(f"  Reduced shape: {X_reduced.shape}")

    # Reconstruction
    X_reconstructed = pca.inverse_transform(X_reduced)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"\nReconstruction MSE: {reconstruction_error:.6f}")
    print(f"Original variance: {np.var(X):.6f}")
    print(f"Explained ratio: {1 - reconstruction_error/np.var(X):.4f}")

    # =========================================================================
    # Part 5: Variance Explained by Components
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 5: VARIANCE EXPLAINED BY COMPONENTS")
    print("=" * 70)

    print("""
Since our data has only ~20 true latent dimensions, most variance
should be captured by the first 20-30 components.
""")

    # Compute variance explained by each component
    variances = []
    X_centered = X - pca.mean_

    for i in range(min(n_components, 30)):
        projection = X_centered @ pca.components_[i]
        var_explained = np.var(projection)
        variances.append(var_explained)

    total_var = np.var(X)
    cumulative = np.cumsum(variances) / total_var

    print(f"\nVariance explained (first 10 components):")
    print(f"{'Component':>10} {'Var Explained':>15} {'Cumulative':>12}")
    print("-" * 40)
    for i in range(10):
        print(f"{i+1:>10} {variances[i]/total_var:>15.4f} {cumulative[i]:>12.4f}")

    print(f"\nComponents needed to explain 95% variance: ", end="")
    for i, cum in enumerate(cumulative):
        if cum >= 0.95:
            print(f"{i+1}")
            break
    else:
        print(f">{len(cumulative)}")

    # =========================================================================
    # Part 6: Comparison with Standard Approach
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 6: WHEN DEFLATION PCA HELPS")
    print("=" * 70)

    print("""
When to use Deflation PCA:
- Output dimension n > 10,000
- Number of samples t < n
- Memory constraints prevent storing n×n matrices

When NOT needed:
- n < 1,000 (standard methods work fine)
- n ≈ t (standard methods work)
- You have unlimited memory
""")

    # Memory comparison
    print("\nMemory comparison for n=50,000 outputs:")
    n_large = 50000
    print(f"  Standard covariance: {n_large}×{n_large} = {n_large*n_large*8/1e9:.1f} GB")
    print(f"  Deflation PCA:       {t}×{n_large} = {t*n_large*8/1e6:.1f} MB")

    # =========================================================================
    # Part 7: Using PCA with verify()
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 7: USING PCA WITH verify()")
    print("=" * 70)

    print("""
The verify() function has a pca_components parameter that automatically
applies DeflationPCA when output dimensions are high.

Example usage:
```python
result = verify(
    model=segmentation_model,
    input_set=input_box,
    m=5000,
    epsilon=0.01,
    surrogate='clipping_block',
    pca_components=100  # Reduce to 100 dimensions
)
```

Note: When using PCA, the bounds in the reduced space are transformed
back to the original space. This is approximate and may be conservative.
""")

    # Demonstrate with a simple example
    import torch
    import torch.nn as nn
    from n2v.probabilistic import verify
    from n2v.sets import Box

    # Create a model with "high-dimensional" output
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 500)  # 500-dim output
    )
    model.eval()

    def model_fn(x):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).numpy()

    # Input set
    lb = np.zeros(10)
    ub = np.ones(10)
    input_set = Box(lb, ub)

    print("\nRunning verify() WITHOUT PCA (500-dim output)...")
    result_no_pca = verify(
        model=model_fn,
        input_set=input_set,
        m=200,
        epsilon=0.05,
        surrogate='naive',
        seed=42,
        verbose=False
    )
    width_no_pca = np.mean(result_no_pca.ub - result_no_pca.lb)
    print(f"  Average bound width: {width_no_pca:.4f}")

    print("\nRunning verify() WITH PCA (reduce to 50 dimensions)...")
    result_with_pca = verify(
        model=model_fn,
        input_set=input_set,
        m=200,
        epsilon=0.05,
        surrogate='naive',
        pca_components=50,
        seed=42,
        verbose=False
    )
    width_with_pca = np.mean(result_with_pca.ub - result_with_pca.lb)
    print(f"  Average bound width: {width_with_pca:.4f}")

    print(f"\nNote: PCA may change bound tightness due to the transformation.")
    print(f"Use PCA primarily when memory/computation is a bottleneck.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
DeflationPCA for high-dimensional outputs:

WHEN TO USE:
- Output dimension n > 10,000
- Memory constraints (can't store n×n covariance)
- t << n (fewer samples than dimensions)

HOW IT WORKS:
- Finds principal directions one at a time via gradient ascent
- Deflates (removes) each found direction before finding the next
- Memory: O(t × n) instead of O(n²)

WITH verify():
- Set pca_components parameter
- Conformal inference runs in reduced space
- Bounds transformed back to original space (approximate)

RECOMMENDATION:
- First try without PCA if computationally feasible
- Use PCA when n > 10,000 or memory is limited
- Choose pca_components to explain ~95% of variance
""")


if __name__ == "__main__":
    main()
