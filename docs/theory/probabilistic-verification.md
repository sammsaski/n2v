# Probabilistic Verification in n2v

This document describes the theory and usage of probabilistic verification in n2v, a model-agnostic approach to computing output bounds with formal probabilistic guarantees.

## Overview

Traditional neural network verification computes exact or over-approximate reachable sets by analyzing the network layer by layer. While sound and complete, these methods can be slow for large networks due to exponential complexity from ReLU splitting.

**Probabilistic verification** provides an alternative approach:
- **Model-agnostic**: Treats the network as a black box
- **Scalable**: Constant-time regardless of network size
- **Formal guarantees**: Provides coverage and confidence bounds
- **Flexible**: Works with any callable model (PyTorch, TensorFlow, ONNX, APIs)

## Theoretical Foundation

### The ⟨ε, ℓ, m⟩ Guarantee Framework

Probabilistic verification provides guarantees parameterized by three values:

| Parameter | Description |
|-----------|-------------|
| **m** | Number of calibration samples |
| **ℓ** | Rank parameter (typically m-1) |
| **ε** | Miscoverage level |

These parameters provide two guarantees:

1. **Coverage (δ₁ = 1-ε)**: The probability that a random output from the input set falls inside the computed bounds.

2. **Confidence (δ₂)**: The probability that the coverage guarantee holds.

The confidence is computed as:
```
δ₂ = 1 - betacdf(1-ε; ℓ, m+1-ℓ)
```

where `betacdf` is the cumulative distribution function of the Beta distribution.

### Conformal Inference

The approach is based on **conformal prediction**, a distribution-free framework for constructing prediction sets with guaranteed coverage.

**Algorithm Overview:**

1. **Sample Training Set**: Draw `t` samples from the input set, compute outputs
2. **Fit Surrogate**: Train a surrogate model that predicts "typical" outputs
3. **Sample Calibration Set**: Draw `m` additional samples, compute outputs
4. **Compute Nonconformity Scores**: Measure how "atypical" each calibration output is
5. **Determine Threshold**: Select the ℓ-th largest score
6. **Inflate Bounds**: Expand surrogate bounds by the threshold

### Nonconformity Score

For each output `y` with surrogate prediction `ŷ` and dimension-wise normalization `τ`:

```
R(y) = max_k( |y[k] - ŷ[k]| / τ[k] )
```

This measures the worst-case normalized deviation across all output dimensions.

### Surrogate Models

n2v provides two surrogate types:

#### Naive Surrogate
- Uses the center (mean) of training outputs as prediction for all inputs
- Simple but potentially conservative
- Fast to compute

#### Clipping Block Surrogate
- Projects each calibration output onto the convex hull of training outputs
- Uses L∞ projection via linear programming
- Produces tighter bounds but requires LP solving

## API Reference

### Main Entry Point: `verify()`

```python
from n2v.probabilistic import verify
from n2v.sets import Box

result = verify(
    model,           # Callable: numpy array -> numpy array
    input_set,       # Box specifying input region
    m=8000,          # Calibration samples (default: 8000)
    ell=None,        # Rank parameter (default: m-1)
    epsilon=0.001,   # Miscoverage level (default: 0.001)
    surrogate='clipping_block',  # 'naive' or 'clipping_block'
    training_samples=None,       # Samples for surrogate (default: m//2)
    pca_components=None,         # Dimensionality reduction (optional)
    batch_size=100,              # Batch size for model inference
    seed=None,                   # Random seed for reproducibility
    verbose=False                # Print progress
)
```

**Returns:** `ProbabilisticBox` with output bounds and guarantee metadata.

### ProbabilisticBox

A `ProbabilisticBox` is a `Box` with additional probabilistic guarantee metadata:

```python
# Access bounds
lb, ub = result.lb, result.ub
lb, ub = result.get_range()

# Access guarantee parameters
result.m          # Calibration set size
result.ell        # Rank parameter
result.epsilon    # Miscoverage level
result.coverage   # 1 - epsilon
result.confidence # Computed from betacdf

# Inherits all Box methods
samples = result.sample(100)
star = result.to_star()  # Warning: loses probabilistic metadata
```

### NeuralNetwork.reach() Integration

```python
import n2v
from n2v.sets import Box

model = torch.nn.Sequential(...)
net = n2v.NeuralNetwork(model)
input_set = Box(lb, ub)

# Probabilistic method
result = net.reach(
    input_set,
    method='probabilistic',
    m=1000,
    epsilon=0.01,
    surrogate='clipping_block'
)

# Hybrid method (deterministic until threshold, then probabilistic)
result = net.reach(
    input_set,
    method='hybrid',
    max_stars=1000,        # Switch if star count exceeds
    timeout_per_layer=30.0 # Switch if layer takes too long
)
```

## Usage Examples

### Basic Usage

```python
import numpy as np
import torch.nn as nn
from n2v.probabilistic import verify
from n2v.sets import Box

# Create a simple model
model = nn.Sequential(
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20, 3)
)
model.eval()

# Model wrapper for numpy interface
def model_fn(x):
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()

# Define input region
lb = np.zeros(5)
ub = np.ones(5)
input_set = Box(lb, ub)

# Run probabilistic verification
result = verify(
    model=model_fn,
    input_set=input_set,
    m=1000,
    epsilon=0.05,  # 95% coverage
    surrogate='naive'
)

print(f"Output bounds: [{result.lb.flatten()}, {result.ub.flatten()}]")
print(f"Coverage: {result.coverage:.2%}")
print(f"Confidence: {result.confidence:.4f}")
```

### Comparing Surrogates

```python
# Naive surrogate (simpler, potentially wider bounds)
result_naive = verify(
    model=model_fn,
    input_set=input_set,
    m=1000,
    epsilon=0.05,
    surrogate='naive'
)

# Clipping block surrogate (tighter bounds)
result_clipping = verify(
    model=model_fn,
    input_set=input_set,
    m=1000,
    epsilon=0.05,
    surrogate='clipping_block',
    training_samples=500
)

# Compare bound widths
width_naive = np.mean(result_naive.ub - result_naive.lb)
width_clipping = np.mean(result_clipping.ub - result_clipping.lb)

print(f"Naive width: {width_naive:.4f}")
print(f"Clipping width: {width_clipping:.4f}")
```

### High-Dimensional Outputs with PCA

For models with high-dimensional outputs (e.g., image segmentation), use dimensionality reduction:

```python
result = verify(
    model=model_fn,
    input_set=input_set,
    m=1000,
    epsilon=0.01,
    surrogate='clipping_block',
    pca_components=50  # Reduce to 50 principal components
)
```

### External APIs and Black-Box Models

```python
def api_model(x):
    """Call external API for predictions."""
    # x is a numpy array of shape (batch_size, input_dim)
    responses = []
    for sample in x:
        response = requests.post(
            'https://api.example.com/predict',
            json={'input': sample.tolist()}
        )
        responses.append(response.json()['output'])
    return np.array(responses)

result = verify(
    model=api_model,
    input_set=input_set,
    m=500,  # Fewer samples due to API costs
    epsilon=0.05,
    surrogate='naive'
)
```

## Parameter Selection Guide

### Choosing m (Calibration Set Size)

| m | Coverage | Confidence | Use Case |
|---|----------|------------|----------|
| 100 | Moderate | Lower | Quick testing |
| 1000 | Good | High | Standard verification |
| 8000 | Excellent | Very high | Production/safety-critical |

Larger `m` provides:
- Tighter bounds (threshold is more accurate)
- Higher confidence in the coverage guarantee

### Choosing ε (Miscoverage Level)

| ε | Coverage | Trade-off |
|---|----------|-----------|
| 0.01 | 99% | Wider bounds, higher assurance |
| 0.05 | 95% | Balanced |
| 0.10 | 90% | Tighter bounds, lower assurance |

### Choosing ℓ (Rank Parameter)

The default `ℓ = m-1` provides a good balance. Lower values of `ℓ` (with adjusted ε) can provide tighter bounds at the cost of lower coverage.

### Naive vs Clipping Block

| Surrogate | Speed | Bound Tightness | When to Use |
|-----------|-------|-----------------|-------------|
| Naive | Fast | Conservative | Quick screening, simple models |
| Clipping Block | Slower | Tighter | Final verification, critical applications |

## Comparison with Deterministic Methods

| Aspect | Deterministic (exact/approx) | Probabilistic |
|--------|------------------------------|---------------|
| **Soundness** | **Yes** - all outputs guaranteed in bounds | **No** - ε fraction may be outside |
| **Guarantee** | 100% containment | (1-ε) coverage with confidence δ₂ |
| **Speed** | Depends on network size | Constant time |
| **Precision** | Exact (for exact methods) | High-probability bounds |
| **Model Access** | Layer-by-layer analysis | Black-box |
| **Scalability** | Limited by ReLU splitting | Unlimited |

**When to use probabilistic:**
- Large networks where deterministic methods time out
- Black-box models (no access to architecture)
- External APIs
- Quick screening before detailed analysis
- When coverage guarantees (not soundness) are acceptable

**When to use deterministic:**
- **Safety-critical applications requiring soundness**
- Small/medium networks where deterministic methods are tractable
- When 100% containment is required (not just high probability)

## Mathematical Details

### Confidence Computation

The confidence δ₂ is computed using the incomplete beta function:

```python
from scipy.stats import beta

def compute_confidence(epsilon, ell, m):
    """
    Compute confidence level for ⟨ε, ℓ, m⟩ guarantee.

    confidence = 1 - I_{1-ε}(ℓ, m+1-ℓ)

    where I_x(a,b) is the regularized incomplete beta function.
    """
    return 1 - beta.cdf(1 - epsilon, ell, m + 1 - ell)
```

### Nonconformity Score

The nonconformity score measures how far each calibration output is from the surrogate prediction:

```python
def compute_nonconformity_scores(calibration_outputs, surrogate_predictions, tau):
    """
    Compute normalized L∞ nonconformity scores.

    R_i = max_k( |y_i[k] - ŷ_i[k]| / τ[k] )
    """
    diff = np.abs(calibration_outputs - surrogate_predictions)
    normalized_diff = diff / tau
    return np.max(normalized_diff, axis=1)
```

### Inflation

The output bounds are computed by inflating the surrogate bounds by the threshold:

```python
def compute_bounds(surrogate_lb, surrogate_ub, threshold, tau):
    """
    Inflate surrogate bounds by threshold * tau.
    """
    inflation = threshold * tau
    lb = surrogate_lb - inflation
    ub = surrogate_ub + inflation
    return lb, ub
```

## References

1. Hashemi et al. "Scaling Data-Driven Probabilistic Reachability Analysis to High-Dimensional Image Classification" (ICLR 2026)

2. Hashemi et al. "Data-Driven Reachability Analysis via Conformal Prediction" (Older paper, ReLU surrogate approach)

3. Vovk et al. "Algorithmic Learning in a Random World" (Conformal prediction foundations)

## Implementation Files

```
n2v/probabilistic/
├── __init__.py              # Exports: verify
├── verify.py                # Main verify() function
├── conformal.py             # Conformal inference primitives
├── surrogates/
│   ├── __init__.py
│   ├── base.py              # Abstract Surrogate interface
│   ├── naive.py             # NaiveSurrogate
│   └── clipping_block.py    # ClippingBlockSurrogate
└── dimensionality/
    ├── __init__.py
    └── deflation_pca.py     # DeflationPCA for high-dim outputs

n2v/sets/
└── probabilistic_box.py     # ProbabilisticBox class

tests/unit/probabilistic/
├── test_probabilistic_box.py
├── test_conformal.py
├── test_surrogates.py
├── test_verify.py
├── test_deflation_pca.py
└── test_integration.py

tests/soundness/
└── test_soundness_probabilistic.py
```
