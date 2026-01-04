# Probabilistic Verification Examples

This directory contains examples and showcases demonstrating the probabilistic verification module in n2v.

## Overview

Probabilistic verification uses **conformal inference** to compute output bounds with coverage guarantees. Unlike deterministic methods, it treats the neural network as a black box.

**Key guarantee**: With confidence δ₂, at least (1-ε) of outputs from the input set are contained in the computed bounds.

**Important**: Probabilistic verification is **NOT sound**. Unlike `exact` and `approx` methods which guarantee all outputs are contained, probabilistic methods allow a fraction ε of outputs to fall outside the bounds. Use deterministic methods when soundness is required.

## Directory Structure

```
examples/ProbVer/
├── README.md                    # This file
├── showcase_speed.py            # Speed/scalability vs network depth and width
├── showcase_scalability.py      # Reach set comparison: exact vs approx vs probabilistic
├── showcase_blackbox.py         # Black-box model verification (ensembles, APIs)
├── tutorial/                    # Step-by-step tutorials
│   ├── 01_conformal_basics.py   # Conformal inference fundamentals
│   ├── 02_surrogates.py         # Naive vs clipping block surrogates
│   ├── 03_verify_pipeline.py    # Complete verification pipeline
│   ├── 04_deflation_pca.py      # Dimensionality reduction
│   └── 05_integration.py        # n2v NeuralNetwork.reach() integration
└── outputs/                     # Generated visualizations
    ├── speed_comparison.png     # Time vs depth/width plots
    ├── scalability_showcase.png # Side-by-side surrogate comparison + timing table
    └── blackbox_showcase.png    # Black-box verification results
```

## Showcase Scripts

These scripts demonstrate the **benefits** of probabilistic verification.

| Script | What It Shows |
|--------|---------------|
| `showcase_speed.py` | How probabilistic verification time stays constant as networks grow (depth/width), while exact methods slow down exponentially |
| `showcase_scalability.py` | Side-by-side comparison of reach sets from exact, approx, and probabilistic methods with timing table. Compares naive vs clipping block surrogates |
| `showcase_blackbox.py` | Unique capability: verify ensembles, external APIs, unsupported layers, non-differentiable models |

### Running Showcases

```bash
cd examples/ProbVer

# See the speed advantage
python showcase_speed.py

# See scalability to large networks
python showcase_scalability.py

# See black-box model verification
python showcase_blackbox.py
```

Generated visualizations are saved to the `outputs/` directory.

## Tutorial Scripts

Step-by-step tutorials in the `tutorial/` directory explain each component.

| File | Topic |
|------|-------|
| `01_conformal_basics.py` | Core conformal inference: confidence, normalization, scores, thresholds |
| `02_surrogates.py` | Comparison of naive vs clipping block surrogates |
| `03_verify_pipeline.py` | Complete verification pipeline with empirical validation |
| `04_deflation_pca.py` | Dimensionality reduction for high-dimensional outputs |
| `05_integration.py` | Using probabilistic verification with `NeuralNetwork.reach()` |

### Running Tutorials

```bash
cd examples/ProbVer/tutorial

# Start with the basics
python 01_conformal_basics.py

# Understand surrogate models
python 02_surrogates.py

# See the full pipeline
python 03_verify_pipeline.py

# Learn about high-dimensional outputs
python 04_deflation_pca.py

# Integration with n2v
python 05_integration.py
```

## Quick Start

### Basic Usage

```python
from n2v.probabilistic import verify
from n2v.sets import Box
import numpy as np

# Define your model (any callable)
def model(x):
    return my_neural_network(x)

# Define input region
lb = np.zeros(5)
ub = np.ones(5)
input_set = Box(lb, ub)

# Run verification
result = verify(
    model=model,
    input_set=input_set,
    m=1000,           # Calibration samples
    epsilon=0.05,     # 95% coverage
    surrogate='naive' # or 'clipping_block'
)

# Result is a ProbabilisticBox
print(f"Bounds: [{result.lb}, {result.ub}]")
print(f"Coverage: {result.coverage}")
print(f"Confidence: {result.confidence}")
```

### With NeuralNetwork.reach()

```python
import n2v
import torch.nn as nn

model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 2))
verifier = n2v.NeuralNetwork(model)

# Probabilistic method
result = verifier.reach(
    input_set,
    method='probabilistic',
    m=1000,
    epsilon=0.05
)

# Hybrid method (tries exact first)
result = verifier.reach(
    input_set,
    method='hybrid',
    max_stars=500
)
```

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `m` | Calibration set size | 500-10000 |
| `epsilon` | Miscoverage level (1-ε = coverage) | 0.001-0.1 |
| `ell` | Rank parameter | m-1 (default) |
| `surrogate` | Surrogate type | 'naive' or 'clipping_block' |

**Trade-offs**:
- Larger `m` → tighter bounds, higher confidence, more computation
- Smaller `epsilon` → higher coverage, wider bounds
- `clipping_block` → tighter bounds than `naive`, but slower

## When to Use Probabilistic Verification

| Scenario | Recommended Method | Guarantee |
|----------|-------------------|-----------|
| Safety-critical, need soundness | `method='exact'` | Sound & complete |
| Medium network, need soundness | `method='approx'` | Sound (over-approx) |
| Large network, coverage acceptable | `method='probabilistic'` | Coverage only (not sound) |
| Black-box model | `method='probabilistic'` | Coverage only (not sound) |
| Unknown complexity | `method='hybrid'` | Sound unless fallback triggers |

### Unique Probabilistic Capabilities

These scenarios **require** probabilistic verification:

| Scenario | Why Deterministic Fails |
|----------|-------------------------|
| Ensemble models | Multiple models combined |
| External APIs | No architecture access |
| Unsupported layers | LayerNorm, Softmax, custom ops |
| Non-differentiable ops | Argmax, discrete decisions |
| Very large networks | Exponential time in exact methods |

## Component Overview

```
n2v/probabilistic/
├── verify.py                 # Main entry point
├── conformal.py              # Conformal inference primitives
├── surrogates/
│   ├── naive.py              # Center-based surrogate
│   └── clipping_block.py     # Convex hull projection surrogate
└── dimensionality/
    └── deflation_pca.py      # PCA for high-dim outputs
```

## Further Reading

- [Full documentation](../../docs/probabilistic_verification.md)
- Research papers:
  - Hashemi et al. "Scaling Data-Driven Probabilistic Reachability Analysis to High-Dimensional Image Classification" (ICLR 2026)
  - Vovk et al. "Algorithmic Learning in a Random World" (Conformal prediction foundations)
