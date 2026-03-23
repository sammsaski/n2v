# NNV-Python: Neural Network Verification Tool

Python implementation of the Neural Network Verification (NNV) tool, supporting formal verification and reachability analysis for PyTorch neural networks.

**Translated from MATLAB NNV** | **PyTorch Native** | **ONNX Support** | **VNN-COMP Ready**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Layers](#supported-layers)
- [Set Representations](#set-representations)
- [Reachability Methods](#reachability-methods)
- [Probabilistic Verification](#probabilistic-verification)
- [CNN Verification](#cnn-verification)
- [Examples](#examples)
- [Falsification](#falsification)
- [Performance Tips](#performance-tips)
- [Testing](#testing)
- [API Reference](#api-reference)
- [ONNX Model Support](#onnx-model-support)
- [VNN-COMP Benchmarks](#vnn-comp-benchmarks)
- [Differences from MATLAB NNV](#differences-from-matlab-nnv)
- [Theoretical Foundations](#theoretical-foundations)
- [References](#references)

---

## Overview

NNV-Python is a Python port of the MATLAB NNV tool, designed for:
- **Formal verification** of neural network properties
- **Reachability analysis** using set-based methods
- **Robustness verification** against adversarial perturbations
- **PyTorch integration** for seamless model verification

### Why NNV-Python?

- **PyTorch Native**: Works directly with PyTorch models (no custom layer classes needed)
- **ONNX Support**: Load and verify ONNX models via onnx2torch
- **Exact & Approximate**: Multiple verification methods for speed/precision trade-offs
- **CNN Support**: Full support for convolutional networks with pooling layers
- **Probabilistic Verification**: Model-agnostic conformal inference for large/black-box models
- **VNN-COMP Ready**: Complete VNN-COMP 2025 benchmark infrastructure with 28 benchmarks
- **Extensible**: Easy to add new layers and methods

---

## Key Features

### Set-Based Representations
- **Star Sets**: Exact with linear constraints (`C*α ≤ d`)
- **Zonotopes**: Efficient over-approximations (`c + V*α`)
- **Boxes**: Fast interval-based bounds
- **ImageStar/ImageZono**: Image-aware representations for CNNs
- **Hexatope/Octatope**: Specialized polytopes with strongly polynomial optimization

### Layer Support (20+ layer types)
- **Fully Connected**: Linear layers (exact)
- **Convolutional**: Conv1D, Conv2D (exact)
- **Activations**: ReLU, LeakyReLU (exact/approx), Sigmoid, Tanh (approx), Sign (exact/approx)
- **Pooling**: MaxPool2D (exact/approx), AvgPool2D, GlobalAvgPool (exact, no splitting)
- **Normalization**: BatchNorm1d/2d (exact, fusible with preceding Linear/Conv)
- **Structural**: Flatten, Pad, Upsample, Transpose, Reshape, Concatenation, Slice, Split, Reduce
- **ONNX**: Full graph execution for Add, Sub, Mul, Div, MatMul, Neg, Cast, and more

### Verification Methods
- **Exact**: Sound and complete (may be slower)
- **Approximate**: Over-approximate (faster, still sound)
- **Probabilistic**: Model-agnostic with formal coverage guarantees
- **Hybrid**: Mix exact/approximate/probabilistic for optimal performance
- **Falsification**: Quick counterexample search via random sampling and PGD

---

## Installation

### Prerequisites

- Python >= 3.8
- Git (for cloning with submodules)

### Step 1: Clone with Submodules

When cloning the repository, make sure to include the submodules (specifically `onnx2torch`):

```bash
# Option 1: Clone with submodules
git clone --recurse-submodules <repository-url>

# Option 2: If already cloned, initialize submodules
git submodule update --init --recursive
```

### Step 2: Install Dependencies

```bash
cd n2v

# Install core dependencies
pip install torch numpy scipy cvxpy

# Install onnx2torch from the submodule (for ONNX model support)
pip install -e third_party/onnx2torch

# Install n2v in editable mode
pip install -e .
```

### Quick Install (All Dependencies)

```bash
pip install -r requirements.txt
pip install -e third_party/onnx2torch
pip install -e .
```

### Core Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- CVXPY >= 1.2.0
- onnx2torch (installed from submodule at `third_party/onnx2torch`)

---

## Quick Start

### Basic Example: Feedforward Network

```python
import torch
import torch.nn as nn
import numpy as np
import n2v as nnv
from n2v.sets import Star

# Define your PyTorch model
model = nn.Sequential(
    nn.Linear(3, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
model.eval()

# Create input specification (L∞ ball around a point)
center = np.array([0.5, 0.5, 0.5])
epsilon = 0.1
input_star = Star.from_bounds(center - epsilon, center + epsilon)

# Verify the network
verifier = nnv.NeuralNetwork(model)
output_stars = verifier.reach(input_star, method='exact')

print(f"Output: {len(output_stars)} reachable set(s)")

# Get output bounds (use get_ranges() for exact LP-based bounds)
for star in output_stars:
    lb, ub = star.get_ranges()
    print(f"Lower bounds: {lb.flatten()}")
    print(f"Upper bounds: {ub.flatten()}")
```

### CNN Example: Image Classification

```python
import torch.nn as nn
from n2v.sets import ImageStar

# Define CNN model
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),  # ⭐ AvgPool is exact and fast!
    nn.Flatten(),
    nn.Linear(16 * 14 * 14, 10)
)
model.eval()

# Create ImageStar input (28x28 MNIST image with perturbation)
image = np.random.rand(28, 28)
epsilon = 0.01
lb = np.maximum(image - epsilon, 0)
ub = np.minimum(image + epsilon, 1)

input_star = ImageStar.from_bounds(
    lb.reshape(28, 28, 1),
    ub.reshape(28, 28, 1),
    height=28, width=28, num_channels=1
)

# Verify CNN
verifier = nnv.NeuralNetwork(model)
output_stars = verifier.reach(input_star, method='exact')

print(f"CNN output: {len(output_stars)} reachable sets")
```

### Robustness Verification

```python
# Check if output class remains stable under perturbation
def check_robustness(output_stars, true_class):
    """Check if true_class has highest score in all reachable sets."""
    for star in output_stars:
        lb, ub = star.get_ranges()
        lb = lb.flatten()
        ub = ub.flatten()

        # Check if any other class could have higher score
        for i in range(len(lb)):
            if i != true_class:
                if ub[i] >= lb[true_class]:
                    return False  # Not robust
    return True  # Robust!

is_robust = check_robustness(output_stars, true_class=3)
print(f"Model is robust: {is_robust}")
```

---

## Supported Layers

### Exact (Linear) Layers

These layers are affine transformations — computed exactly with no approximation or splitting.

| Layer Type | Star/ImageStar | Zono/ImageZono | Box | Hex/Oct |
|------------|:--------------:|:--------------:|:---:|:-------:|
| **Linear** (`nn.Linear`) | ✅ | ✅ | ✅ | ✅ |
| **Conv2D** (`nn.Conv2d`) | ✅ | ✅ | — | — |
| **Conv1D** (`nn.Conv1d`) | ✅ | ✅ | ✅ | — |
| **BatchNorm** (`nn.BatchNorm1d/2d`) | ✅ | ✅ | ✅ | ✅ |
| **AvgPool2D** (`nn.AvgPool2d`) | ✅ | ✅ | — | — |
| **GlobalAvgPool** (`nn.AdaptiveAvgPool2d(1)`) | ✅ | ✅ | — | — |
| **Flatten** (`nn.Flatten`) | ✅ | ✅ | no-op | no-op |
| **Pad** (`nn.ZeroPad2d`, etc.) | ✅ | ✅ | — | — |
| **Upsample** (`nn.Upsample`, nearest) | ✅ | ✅ | — | — |
| **Reduce** (ReduceSum, ReduceMean) | ✅ | ✅ | ✅ | — |
| **Transpose** | ✅ | ✅ | ✅ | — |
| **Neg** | ✅ | ✅ | ✅ | ✅ |
| **Identity/Dropout/Cast** | no-op | no-op | no-op | no-op |

### Nonlinear Layers

These layers require splitting (exact) or relaxation (approx). See [docs/theoretical_foundations.md](docs/theoretical_foundations.md) for detailed algorithms.

| Layer Type | Star (exact) | Star (approx) | Zono | Box |
|------------|:------------:|:--------------:|:----:|:---:|
| **ReLU** (`nn.ReLU`) | Split | Triangle relax | Interval approx | Elementwise |
| **LeakyReLU** (`nn.LeakyReLU`) | Split | Triangle relax | Interval approx | Elementwise |
| **Sigmoid** (`nn.Sigmoid`) | — | S-curve relax | Interval approx | Monotone |
| **Tanh** (`nn.Tanh`) | — | S-curve relax | Interval approx | Monotone |
| **Sign** | Split | Parallelogram relax | Interval approx | Elementwise |
| **MaxPool2D** (`nn.MaxPool2d`) | Split + LP | New predicates | Bounds approx | — |

### ONNX Graph Operations

These operations are handled by the ONNX graph execution engine when loading models via `load_onnx()`.

| Operation | Description |
|-----------|-------------|
| OnnxReshape | Reshape with NCHW ↔ HWC format conversion |
| OnnxConcat | Concatenation along specified axis |
| OnnxSlice/SliceV9 | Rectangular slicing |
| OnnxSplit/Split13 | Splitting along axis |
| OnnxBinaryMath (Add, Sub, Mul, Div) | Element-wise arithmetic with constants |
| OnnxMatMul | Matrix multiplication with constant weights |

✅ = exact | — = not supported for this set type

---

## Set Representations

n2v provides 8 set representations with different expressiveness/speed trade-offs. See [docs/theoretical_foundations.md](docs/theoretical_foundations.md) for full mathematical definitions.

### Star Sets

Primary representation: **x = c + V*α** where **C*α ≤ d**

```python
from n2v.sets import Star

# From bounds (creates Star from hyperbox)
star = Star.from_bounds(lb, ub)

# Manual construction
V = np.array([[1, 0.1], [0, 0.2], [0, 0.3]])  # Basis matrix (center + generators)
C = np.array([[1, 0], [0, 1]])  # Constraints
d = np.array([[1], [1]])
star = Star(V, C, d)

# Operations
output = star.affine_map(W, b)       # Linear transformation
lb, ub = star.get_ranges()           # LP-based exact bounds
new_star = star.intersect_half_space(G, g)  # Intersect with G*x <= g
```

### ImageStar

Star sets for images with spatial structure (4D tensor: H x W x C x nVar+1):

```python
from n2v.sets import ImageStar

# Create from image bounds
image_star = ImageStar.from_bounds(
    lb_image,  # (height, width, channels)
    ub_image,  # (height, width, channels)
    height=28, width=28, num_channels=1
)

# Flatten for FC layers (CHW ordering for PyTorch compatibility)
regular_star = image_star.flatten_to_star()
```

### Zonotopes

Efficient representation: **x = c + V*α** where **-1 ≤ αᵢ ≤ 1**

Bounds are computed analytically (no LP needed): `lb[i] = c[i] - Σ|V[i,j]|`

```python
from n2v.sets import Zono

# From center and generators
zono = Zono(center, generators)

# From bounds
zono = Zono.from_bounds(lb, ub)

# Analytical bounds (fast, no LP)
lb, ub = zono.get_ranges()
```

### Box

Axis-aligned hyperrectangle. Fastest but most conservative.

```python
from n2v.sets import Box

box = Box(lb, ub)
```

### Hexatope / Octatope

Specialized polytopes using Difference Constraint Systems (DCS) and UTVPI constraints respectively. Enable **strongly polynomial optimization** via minimum cost flow instead of LP.

```python
from n2v.sets import Hexatope, Octatope
```

### ProbabilisticBox

Box with conformal inference metadata (coverage and confidence guarantees). Returned by probabilistic verification.

```python
from n2v.sets import ProbabilisticBox

# Returned by probabilistic verify()
result.coverage    # e.g. 0.99
result.confidence  # e.g. 0.997
```

---

## Reachability Methods

### Exact Methods

#### `'exact'`
- **Sound and complete** (exact reachability)
- Uses Star sets with ReLU splitting
- Best for: Small-medium networks, safety-critical applications
- Note: Can be slow due to exponential splitting

```python
output = verifier.reach(input_star, method='exact')
```

### Approximate Methods

#### `'approx'`
- Over-approximate using relaxed Star sets or Zonotopes
- Faster than exact (no splitting)
- Best for: Large networks, quick verification

```python
output = verifier.reach(input_star, method='approx')
```

### Probabilistic Methods

#### `'probabilistic'`
- Model-agnostic using conformal inference
- Works with any callable model (black-box)
- Provides formal coverage guarantees
- Best for: Very large networks, external APIs

```python
output = verifier.reach(input_box, method='probabilistic', m=1000, epsilon=0.01)
```

#### `'hybrid'`
- Starts with deterministic, switches to probabilistic when needed
- Automatic switching based on star count or timeout
- Best for: Medium-large networks with unknown complexity

```python
output = verifier.reach(input_star, method='hybrid', max_stars=1000)
```

### Method Comparison

| Method | Speed | Guarantee | Use Case |
|--------|-------|-----------|----------|
| `exact` | Slow | Sound & complete | Safety-critical, small nets |
| `approx` | Medium | Sound (over-approx) | Large networks |
| `probabilistic` | Fast | Coverage only (not sound) | Very large nets, black-box |
| `hybrid` | Adaptive | Depends on fallback | Unknown complexity |

---

## Probabilistic Verification

For large models or when deterministic methods are too slow, n2v provides **probabilistic verification** using conformal inference. This approach provides high-confidence output bounds without requiring network-specific analysis.

### Key Features

- **Model-agnostic**: Works with any callable model (PyTorch, TensorFlow, ONNX, even APIs)
- **Scalable**: Constant-time complexity regardless of network size
- **Coverage guarantee**: With high confidence, (1-ε) of outputs are contained in bounds
- **Not sound**: Unlike exact/approx methods, outputs may fall outside bounds with probability ε
- **No network analysis**: Treats the model as a black box

### Basic Usage

```python
from n2v.probabilistic import verify
from n2v.sets import Box
import numpy as np

# Define input region
lb = np.zeros(5)
ub = np.ones(5)
input_set = Box(lb, ub)

# Any callable model
def model(x):
    return my_neural_network(x)

# Run probabilistic verification
result = verify(
    model=model,
    input_set=input_set,
    m=1000,           # Number of calibration samples
    epsilon=0.01,     # 99% coverage guarantee
    surrogate='naive' # or 'clipping_block' for tighter bounds
)

# Result is a ProbabilisticBox with coverage guarantees
print(f"Output bounds: [{result.lb}, {result.ub}]")
print(f"Coverage: {result.coverage}")      # 0.99
print(f"Confidence: {result.confidence}")  # High probability guarantee holds
```

### Using with NeuralNetwork.reach()

```python
import n2v

model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3))
verifier = n2v.NeuralNetwork(model)

# Use probabilistic method
result = verifier.reach(
    input_set,
    method='probabilistic',
    m=1000,
    epsilon=0.01,
    surrogate='clipping_block'
)

pbox = result[0]  # ProbabilisticBox
```

### Hybrid Method

The hybrid method starts with deterministic reachability and automatically switches to probabilistic when thresholds are exceeded:

```python
result = verifier.reach(
    input_set,
    method='hybrid',
    max_stars=1000,           # Switch if star count exceeds this
    timeout_per_layer=30.0,   # Switch if layer takes too long
    m=500,                    # Probabilistic parameters
    epsilon=0.05
)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `m` | Calibration set size | 8000 |
| `ell` | Rank parameter (affects threshold) | m-1 |
| `epsilon` | Miscoverage level (1-epsilon = coverage) | 0.001 |
| `surrogate` | 'naive' (center-based) or 'clipping_block' (tighter) | 'clipping_block' |
| `training_samples` | Samples for clipping_block fitting | m//2 |

### When to Use Probabilistic Verification

| Scenario | Recommended Method | Note |
|----------|-------------------|------|
| Small network, safety-critical | `exact` | Sound guarantee required |
| Medium network | `approx` | Sound, faster than exact |
| Large network, coverage acceptable | `probabilistic` | Not sound, but scalable |
| External API / black-box model | `probabilistic` | Only option for black-box |
| Unknown complexity | `hybrid` | May lose soundness on fallback |

See [docs/probabilistic_verification.md](docs/probabilistic_verification.md) for detailed theory and advanced usage.

---

## CNN Verification

### Best Practices for CNN Verification

#### ⭐ Use AvgPool2D Instead of MaxPool2D

**AvgPool2D is LINEAR** → No star splitting → Much faster!

```python
# Good for verification ✅
nn.AvgPool2d(2, 2)  # Exact, no splitting!

# Slower for verification ⚠️
nn.MaxPool2d(2, 2)  # Can cause exponential splitting
```

**Performance Difference:**
- AvgPool: Input 4 stars → Output 4 stars ✅
- MaxPool: Input 4 stars → Output 16-256 stars ❌

#### Architecture Example

**Verification-Friendly CNN:**

```python
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),  # ⭐ Linear, exact, no splitting
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),  # ⭐ Still no splitting!
    nn.Flatten(),
    nn.Linear(32*7*7, 10)
)
```

### CNN Layer Propagation

For ImageStar through CNN:

1. **Conv2D**: Exact (affine) - `conv2d_star(layer, stars)`
2. **ReLU**: Exact with splitting - `relu_star_exact(stars)`
3. **AvgPool2D**: Exact, no splitting - `avgpool2d_star(layer, stars)` ⭐
4. **Flatten**: Reshaping - `flatten_star(layer, stars)`
5. **Linear**: Exact (affine) - `linear_star(layer, stars)`

### Layer-by-Layer Analysis

```python
from n2v.nn.layer_ops.dispatcher import reach_layer

layers = list(model.children())
current_stars = [input_star]

for i, layer in enumerate(layers):
    print(f"Layer {i}: {type(layer).__name__}")
    print(f"  Input: {len(current_stars)} stars")

    current_stars = reach_layer(
        layer, current_stars,
        method='exact',
        lp_solver='default'
    )

    print(f"  Output: {len(current_stars)} stars")
```

---

## Examples

See [`examples/`](examples/) directory for complete examples:

### Benchmark Examples
- **[`ACASXu/`](examples/ACASXu/)** - ACAS Xu benchmark (186 instances). See [examples/ACASXu/README.md](examples/ACASXu/README.md).
- **[`vnncomp/`](examples/VNN-COMP/)** - Full VNN-COMP 2025 infrastructure (28 benchmarks). See [examples/VNN-COMP/README.md](examples/VNN-COMP/README.md).

### Running ACAS Xu

```bash
cd examples/ACASXu

# Single instance
python run_instance.py onnx/ACASXU_run2a_1_1_batch_2000.onnx vnnlib/prop_1.vnnlib

# Full benchmark
./run_benchmark.sh --timeout 120 --falsify-method random+pgd
```

### Running VNN-COMP Smoke Test

```bash
cd examples/VNN-COMP

# Quick check: 1 instance per benchmark
./smoke_test.sh /path/to/vnncomp_benchmarks

# Full benchmark
./run_benchmark.sh /path/to/benchmarks/acasxu_2023 --timeout 120
```

---

## Falsification

Before running expensive reachability analysis, n2v can attempt to quickly find counterexamples using falsification techniques.

### Available Methods

| Method | Description | Speed |
|--------|-------------|-------|
| `random` | Uniform random sampling | Fast |
| `pgd` | Projected Gradient Descent | Moderate |
| `random+pgd` | Random first, then PGD | Combined |

### Usage

```python
from n2v.utils import falsify

# Try to find a counterexample via random sampling
result, counterexample = falsify(model, lb, ub, property_spec, method='random', n_samples=500)

# Use PGD for targeted search
result, counterexample = falsify(model, lb, ub, property_spec, method='pgd', n_restarts=10, n_steps=50)

# Combined approach: random first, then PGD
result, counterexample = falsify(model, lb, ub, property_spec, method='random+pgd')

# Check result
if result == 0:  # SAT - counterexample found
    inp, out = counterexample
    print(f"Found counterexample: input={inp}, output={out}")
elif result == 2:  # UNKNOWN - no counterexample found, need verification
    print("No counterexample found, running reachability analysis...")
```

**Note**: Falsification assumes hyperbox input sets (axis-aligned bounds `[lb, ub]`).

---

## Performance Tips

### 1. 🚀 Use scipy linprog (HiGHS) - 1.5-2x Faster

```python
import n2v

# Use scipy linprog with HiGHS solver (1.5-2x faster than default CVXPY)
n2v.set_lp_solver('linprog')

# Now all verification uses the faster LP solver
output = verifier.reach(input_star, method='exact')
```

**Speedup**: 1.5-2x faster than default CVXPY
**Why**: Lower problem setup overhead, efficient sparse matrix handling
**Best for**: Exact methods and relaxed methods that solve many LPs

See [docs/lp_solvers.md](docs/lp_solvers.md) for detailed LP solver comparison.

### 2. ⭐ Enable Parallel LP Solving

```python
import n2v

# Enable parallel solving with explicit worker count
n2v.set_parallel(True, n_workers=8)  # Use 8 parallel workers

# Or use all available CPU cores
import multiprocessing
n2v.set_parallel(True, n_workers=multiprocessing.cpu_count())

# Now all verification uses parallel LP solving
output = net.reach(input_star, method='exact')
```

**Note**: The default is 4 workers. For best performance, set `n_workers` to your CPU core count.

**Speedup**: 2-10x on multi-core systems depending on core count
**Best for**: Exact method with many output stars, high-dimensional problems

### 3. 🏆 Gurobi Support (TODO)

Gurobi support is planned but not yet implemented. When available, it will provide 10-100x speedup over open-source solvers.

- **License**: Free for academics at https://www.gurobi.com/academia/

### 4. ⭐ Use AvgPool2D Instead of MaxPool2D

```python
# Instead of:
nn.MaxPool2d(2, 2)  # Can cause exponential splitting

# Use:
nn.AvgPool2d(2, 2)  # Always exact, no splitting!
```

**Speedup**: 10-100x faster for verification!

### 5. Use Smaller Perturbations

```python
epsilon = 0.01  # Instead of 0.1
```

Reduces ReLU splitting significantly.

### 6. Choose Appropriate Method

```python
# For small networks (< 1000 neurons)
method = 'exact'

# For medium networks (1000-10000 neurons)
method = 'approx'

# For large networks (> 10000 neurons)
method = 'approx'
```

### 7. Monitor Star Count

```python
for layer in layers:
    current_stars = reach_layer(layer, current_stars)
    if len(current_stars) > 1000:
        print(f"Warning: {len(current_stars)} stars - consider approximate method")
```

### Performance Comparison

| Configuration | vs Default CVXPY |
|---------------|------------------|
| scipy linprog | 1.5-2x faster |
| + Parallel (4 cores) | 2-3x faster |

**Bottom line:** Use `n2v.set_lp_solver('linprog')` for best performance!

---

## Testing

### Running Tests

NNV-Python includes comprehensive unit and soundness tests (840+ total tests) using pytest.

```bash
# Run all tests (unit + soundness)
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run only unit tests
pytest tests/unit/

# Run only soundness tests
pytest tests/soundness/

# Run specific test file
pytest tests/unit/sets/test_star.py -v
pytest tests/soundness/test_soundness_relu.py -v

# Run specific test
pytest tests/unit/sets/test_star.py::TestStar::test_from_bounds -v
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and helpers
│
├── unit/                    # Unit tests (~620 tests)
│   ├── sets/                # Set representation tests (Star, Zono, Box, etc.)
│   ├── layer_ops/           # Layer operation tests (linear, relu, conv2d, etc.)
│   ├── core/                # Core functionality tests (dispatcher, parallel)
│   ├── utils/               # Utility tests (VNN-LIB parsing, solvers)
│   ├── probabilistic/       # Probabilistic verification tests
│   └── integration/         # End-to-end verification tests
│
└── soundness/               # Soundness tests (~200 tests)
    ├── test_soundness_linear.py           # Linear layer soundness
    ├── test_soundness_relu.py             # ReLU activation soundness
    ├── test_soundness_conv2d.py           # Conv2D soundness
    ├── test_soundness_maxpool2d.py        # MaxPool2D soundness
    ├── test_soundness_avgpool2d.py        # AvgPool2D soundness
    ├── test_soundness_flatten.py          # Flatten soundness
    └── test_soundness_probabilistic.py    # Probabilistic verification soundness
```

### Test Coverage

**Unit Tests** verify implementation correctness:
- Set Representations: Star, Zono, Box, ProbabilisticBox creation and operations
- Image Sets: ImageStar, ImageZono with spatial operations
- Layer Operations: Linear, ReLU, Conv2D, MaxPool2D, AvgPool2D, Flatten
- Probabilistic: Conformal inference, surrogates, verify interface
- Dispatcher: Layer-type routing and method selection
- VNN-LIB: Property file parsing
- Integration: End-to-end network verification workflows

**Soundness Tests** verify mathematical correctness:
- Exact vs. approximate methods produce sound results
- Over-approximations contain exact reachable sets
- Bounds computation is correct
- Ground truth validation for simple cases
- Probabilistic coverage and confidence guarantees hold empirically

See [tests/README.md](tests/README.md) for detailed test organization and [tests/soundness/README.md](tests/soundness/README.md) for soundness testing methodology.

### Writing New Tests

**Unit Tests**: Use the fixtures in `unit/conftest.py`:

```python
def test_my_feature(simple_star, simple_image_star):
    """Test description."""
    # Your test code
    result = my_function(simple_star)
    pytest.assert_star_valid(result)
```

**Soundness Tests**: Compare against ground truth or exact methods:

```python
def test_layer_soundness(self):
    """Verify layer produces sound results."""
    # Generate random input
    input_set = generate_random_star(dim=10)

    # Compute exact and approximate results
    exact_result = layer_exact(input_set)
    approx_result = layer_approx(input_set)

    # Verify soundness: approx should contain exact
    pytest.assert_soundness(exact_result, approx_result)
```

### Test Requirements

```bash
pip install pytest pytest-cov
```

### Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=n2v --cov-report=html tests/

# View in browser
open htmlcov/index.html
```

### Current Test Status

- **Total**: 846 passing, 2 skipped
- Unit tests cover sets, layer operations, dispatcher, VNN-LIB parsing, probabilistic verification, and integration
- Soundness tests verify mathematical correctness of all layer implementations including probabilistic guarantees

Skipped tests are for Conv2d with Zonotope (not yet implemented) and sklearn PCA comparison (optional dependency).

---

## API Reference

### Core Classes

#### `NeuralNetwork`

```python
verifier = nnv.NeuralNetwork(pytorch_model)

# Reachability
output_stars = verifier.reach(input_set, method='exact')

# With options
output_stars = verifier.reach(
    input_set,
    method='exact',
    lp_solver='default',
    verbose=True  # Show progress
)
```

#### `Star`

```python
# Construction
star = Star.from_bounds(lb, ub)
star = Star(V, C, d, pred_lb, pred_ub)

# Operations
output_star = star.affine_map(W, b)
new_star = star.intersect_half_space(G, g)
lb, ub = star.get_range(dim_index)  # LP-based exact bound for one dimension
lb, ub = star.get_ranges()          # LP-based exact bounds for all dimensions
```

**Important**: Always use `get_ranges()` for final bounds (LP-based, exact for the polytope).
Avoid `estimate_ranges()` for final results - it's a fast heuristic for internal use that ignores constraints.

#### `ImageStar`

```python
# Construction
img_star = ImageStar.from_bounds(lb_img, ub_img, h, w, c)

# Conversion
regular_star = img_star.flatten_to_star()

# Properties
shape = img_star.get_image_shape()  # (h, w, c)
```

### Layer Operations

```python
from n2v.nn.layer_ops.dispatcher import reach_layer

# Dispatch to appropriate layer operation (auto-detects set type)
output = reach_layer(
    layer,          # PyTorch layer (nn.Linear, nn.ReLU, etc.)
    input_sets,     # List of input sets (Star, Zono, Box, etc.)
    method='exact', # 'exact' or 'approx'
    **kwargs        # lp_solver, verbose, etc.
)
```

#### `ProbabilisticBox`

```python
from n2v.sets import ProbabilisticBox

# From verify() result
result = verify(model, input_set, m=1000, epsilon=0.01)

# Properties
result.coverage     # 0.99 (1 - epsilon)
result.confidence   # High probability guarantee holds
result.m            # Number of calibration samples
result.ell          # Rank parameter

# Inherits from Box - all Box methods work
lb, ub = result.get_range()
star = result.to_star()  # Warning: loses probabilistic metadata
```

### Utilities

```python
from n2v.utils import lpsolver

# Solve LP
result = lpsolver.solve_lp(
    f, A, b, Aeq, beq, lb, ub,
    solver='default',
    minimize=True
)
```

---

## ONNX Model Support

n2v can verify ONNX models directly using the bundled onnx2torch converter.

### Loading ONNX Models

```python
from n2v.utils import load_onnx

# Load ONNX model as PyTorch module
model = load_onnx('model.onnx')

# Verify as usual
verifier = nnv.NeuralNetwork(model)
output = verifier.reach(input_set, method='approx')
```

### Loading VNNLIB Specifications

```python
from n2v.utils import load_vnnlib

# Parse VNNLIB file (returns input bounds and output property)
regions = load_vnnlib('spec.vnnlib')
# regions is a list of (input_bounds, output_property) tuples
# Supports multiple disjunctive input regions
```

### Supported ONNX Operations

Beyond standard PyTorch layers, the ONNX graph engine handles: Reshape, Concat, Slice, Split, binary arithmetic (Add, Sub, Mul, Div), MatMul, Reduce (Sum, Mean), Resize/Upsample, Transpose, Neg, Cast, and Pad.

See the [Supported Layers](#supported-layers) section for the full list.

---

## VNN-COMP Benchmarks

n2v includes a complete VNN-COMP 2025 benchmark infrastructure supporting 28 benchmarks with per-benchmark tuned verification strategies.

### Quick Smoke Test

```bash
cd examples/VNN-COMP
./smoke_test.sh /path/to/vnncomp_benchmarks
```

### Single Instance

```bash
python examples/VNN-COMP/run_instance.py model.onnx spec.vnnlib --category acasxu_2023
```

### Verification Strategy

The runner uses a 3-stage pipeline: **falsification** (random + PGD) to find counterexamples quickly, then **approximate reachability** (sound, fast), then **exact reachability** (sound and complete, potentially slow). Each stage can prove the result and short-circuit remaining stages.

Per-benchmark configurations in `benchmark_configs.py` tune which stages run. For example, large ResNet models skip to probabilistic verification, while small ACAS Xu models run exact analysis.

See [examples/VNN-COMP/README.md](examples/VNN-COMP/README.md) for full documentation.

---

## Differences from MATLAB NNV

| Feature | MATLAB NNV | Python NNV |
|---------|-----------|------------|
| **Model Input** | Custom NN class | PyTorch nn.Module ✅ |
| **Layer Dispatch** | Custom layer classes | Direct PyTorch types ✅ |
| **LP Solver** | MATLAB linprog | CVXPY ✅ |
| **Language** | MATLAB | Python ✅ |
| **API Style** | MATLAB OOP | Pythonic ✅ |
| **Performance** | Good | Optimized for Python ✅ |

### Key Advantages

1. **No Custom Layer Classes**: Works directly with PyTorch layers
2. **PyTorch Native**: Seamless integration with PyTorch ecosystem
3. **Modern Python**: Type hints, clear APIs, modular design
4. **Open Source Tools**: CVXPY instead of proprietary MATLAB

---

## References

### Original MATLAB NNV

This tool is based on the MATLAB NNV tool:

```bibtex
@inproceedings{nnv2_cav2023,
  author = {Lopez, Diego Manzanas and Choi, Sung Woo and Tran, Hoang-Dung and Johnson, Taylor T.},
  title = {NNV 2.0: The Neural Network Verification Tool},
  booktitle = {CAV 2023},
  year = {2023},
  publisher = {Springer},
}
```

### Papers & Resources

- [NNV Tool Paper](https://link.springer.com/chapter/10.1007/978-3-031-37706-8_15)
- [MATLAB NNV GitHub](https://github.com/verivital/nnv)
- [Star Set Reachability (Tran et al., 2019)](https://arxiv.org/abs/1908.01739)
- [Conformal Prediction (Vovk et al., 2005)](https://link.springer.com/book/10.1007/978-3-031-06649-8) — Theory behind probabilistic verification
- [Abstract Interpretation for Neural Networks (Singh et al., 2019)](https://dl.acm.org/doi/10.1145/3290354) — Zonotope abstractions

---

## Theoretical Foundations

For detailed mathematical descriptions of all set representations, relaxation techniques, and algorithmic choices, see [docs/theoretical_foundations.md](docs/theoretical_foundations.md). This includes:

- Mathematical definitions of all 8 set types (Star, Zono, Box, ImageStar, ImageZono, ProbabilisticBox, Hexatope, Octatope)
- Exact vs approximate computation for every layer+set combination
- ReLU triangle relaxation, sigmoid/tanh S-curve relaxation, sign parallelogram relaxation
- MaxPool2D splitting and over-approximation strategies
- Conformal inference theory for probabilistic verification
- Optimization techniques (Zono pre-pass, LP solver selection, parallel computation)

---

## Development & Contributing

### Project Structure

```
n2v/
├── sets/              # Set representations
│   ├── star.py               # Star (primary set type)
│   ├── image_star.py         # ImageStar (4D for CNNs)
│   ├── zono.py               # Zonotope
│   ├── image_zono.py         # ImageZono (4D zonotope)
│   ├── box.py                # Axis-aligned hyperrectangle
│   ├── probabilistic_box.py  # Box with conformal guarantees
│   ├── hexatope.py           # DCS-constrained zonotope
│   ├── octatope.py           # UTVPI-constrained zonotope
│   └── halfspace.py          # Linear constraint representation
├── nn/                # Neural network verification
│   ├── neural_network.py     # NeuralNetwork wrapper class
│   ├── reach.py              # Top-level reachability orchestration
│   └── layer_ops/            # Layer-specific operations (20+ layers)
│       ├── dispatcher.py     # Routes by layer type and set type
│       ├── linear_reach.py
│       ├── relu_reach.py
│       ├── leakyrelu_reach.py
│       ├── sigmoid_reach.py
│       ├── tanh_reach.py
│       ├── sign_reach.py
│       ├── conv2d_reach.py
│       ├── conv1d_reach.py
│       ├── maxpool2d_reach.py
│       ├── avgpool2d_reach.py
│       ├── global_avgpool_reach.py
│       ├── batchnorm_reach.py
│       ├── flatten_reach.py
│       ├── pad_reach.py
│       ├── upsample_reach.py
│       └── reduce_reach.py
├── probabilistic/     # Probabilistic verification module
│   ├── verify.py             # Main verify() entry point
│   ├── conformal.py          # Conformal inference primitives
│   ├── surrogates/           # Surrogate models (naive, clipping_block)
│   └── dimensionality/       # Deflation PCA for high-dim outputs
├── utils/             # Utilities (LP solver, VNNLIB parser, falsification, etc.)
├── examples/
│   ├── ACASXu/               # ACAS Xu benchmark (186 instances)
│   └── vnncomp/              # VNN-COMP 2025 infrastructure (28 benchmarks)
└── docs/              # Documentation
    ├── theoretical_foundations.md  # Mathematical details and algorithms
    ├── probabilistic_verification.md
    └── lp_solvers.md
```

### Adding New Layers

1. Create operation file in `nn/layer_ops/<layer>_reach.py`
2. Add dispatch case in `layer_ops/dispatcher.py`
3. Implement for Star, Zono, Box (as applicable)

### Contributing

Contributions welcome! Please:
1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Submit PR with clear description

---

## License

MIT License - See LICENSE file for details

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/verivital/nnv/issues)
- **Email**: samuel.sasaki@vanderbilt.edu
- **Documentation**: This README and [docs/](docs/)

---

## Quick Links

- [Quick Start](#quick-start) - Get started in 5 minutes
- [Theoretical Foundations](docs/theoretical_foundations.md) - Mathematical details and algorithms
- [VNN-COMP Guide](examples/VNN-COMP/README.md) - Running VNN-COMP benchmarks
- [ACAS Xu Guide](examples/ACASXu/README.md) - ACAS Xu benchmark
- [Probabilistic Verification](docs/probabilistic_verification.md) - Conformal inference theory
- [LP Solvers](docs/lp_solvers.md) - Solver comparison and selection
- [CNN Verification](#cnn-verification) - CNN-specific guide
