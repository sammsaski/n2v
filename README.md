# NNV-Python: Neural Network Verification Tool

Python implementation of the Neural Network Verification (NNV) tool, supporting formal verification and reachability analysis for PyTorch neural networks.

**Translated from MATLAB NNV** | **PyTorch Native** | **Production Ready**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Layers](#supported-layers)
- [Set Representations](#set-representations)
- [Reachability Methods](#reachability-methods)
- [CNN Verification](#cnn-verification)
- [Examples](#examples)
- [Performance Tips](#performance-tips)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Differences from MATLAB NNV](#differences-from-matlab-nnv)
- [References](#references)

---

## Overview

NNV-Python is a Python port of the MATLAB NNV tool, designed for:
- **Formal verification** of neural network properties
- **Reachability analysis** using set-based methods
- **Robustness verification** against adversarial perturbations
- **PyTorch integration** for seamless model verification

### Why NNV-Python?

- ✅ **PyTorch Native**: Works directly with PyTorch models (no custom layer classes needed)
- ✅ **Exact & Approximate**: Multiple verification methods for speed/precision trade-offs
- ✅ **CNN Support**: Full support for convolutional networks with pooling layers
- ✅ **Efficient**: Optimized for both small and large networks
- ✅ **Extensible**: Easy to add new layers and methods

---

## Key Features

### Set-Based Representations
- **Star Sets**: Exact with linear constraints (`C*α ≤ d`)
- **Zonotopes**: Efficient over-approximations (`c + V*α`)
- **Boxes**: Fast interval-based bounds
- **ImageStar/ImageZono**: Image-aware representations for CNNs

### Layer Support
- **Fully Connected**: Linear layers (exact)
- **Convolutional**: Conv1D, Conv2D (exact)
- **Activations**: ReLU (exact with splitting), Sigmoid, Tanh
- **Pooling**: MaxPool2D (exact), **AvgPool2D (exact, no splitting!)** ⭐
- **Normalization**: BatchNorm, LayerNorm
- **Structural**: Flatten, Reshape, Concatenation

### Verification Methods
- **Exact**: Sound and complete (may be slower)
- **Approximate**: Over-approximate (faster, still sound)
- **Hybrid**: Mix exact/approximate for optimal performance

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
pip install -e utils/onnx2torch

# Install n2v in editable mode
pip install -e .
```

### Quick Install (All Dependencies)

```bash
pip install -r requirements.txt
pip install -e utils/onnx2torch
pip install -e .
```

### Core Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- CVXPY >= 1.2.0
- onnx2torch (installed from submodule at `utils/onnx2torch`)

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
output_stars = verifier.reach(input_star, method='exact-star')

print(f"Output: {len(output_stars)} reachable set(s)")

# Get output bounds
for star in output_stars:
    star.estimate_ranges()
    print(f"Lower bounds: {star.state_lb.flatten()}")
    print(f"Upper bounds: {star.state_ub.flatten()}")
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
output_stars = verifier.reach(input_star, method='exact-star')

print(f"CNN output: {len(output_stars)} reachable sets")
```

### Robustness Verification

```python
# Check if output class remains stable under perturbation
def check_robustness(output_stars, true_class):
    """Check if true_class has highest score in all reachable sets."""
    for star in output_stars:
        star.estimate_ranges()
        lb = star.state_lb.flatten()
        ub = star.state_ub.flatten()

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

| Layer Type | Star | Zono | Box | Notes |
|------------|------|------|-----|-------|
| **Linear** | ✅ Exact | ✅ Exact | ✅ Exact | Affine transformation |
| **Conv2D** | ✅ Exact | ✅ Exact | ❌ | Affine transformation |
| **ReLU** | ✅ Exact | ✅ Approx | ✅ Exact | Splitting in exact mode |
| **AvgPool2D** | ✅ Exact | ✅ Exact | ⚠️ | **Linear - no splitting!** ⭐ |
| **MaxPool2D** | ✅ Exact | ✅ Approx | ❌ | Can cause splitting |
| **Sigmoid/Tanh** | ✅ Approx | ✅ Approx | ✅ Approx | Nonlinear |
| **BatchNorm** | 🚧 | 🚧 | 🚧 | In progress |
| **Flatten** | ✅ | ✅ | ✅ | Reshaping |

✅ = Implemented | ⚠️ = Partial | 🚧 = In progress | ❌ = Not implemented

---

## Set Representations

### Star Sets

Represent sets as: **x = V * [1; α]** where **C*α ≤ d**

```python
from n2v.sets import Star

# From bounds
star = Star.from_bounds(lb, ub)

# Manual construction
V = np.array([[1, 0.1], [0, 0.2], [0, 0.3]])  # Basis matrix
C = np.array([[1, 0], [0, 1]])  # Constraints
d = np.array([[1], [1]])
star = Star(V, C, d)

# Operations
output = star.affine_map(W, b)  # Linear transformation
bounds_lb, bounds_ub = star.get_bounds()
```

### ImageStar

Star sets for images with spatial structure:

```python
from n2v.sets import ImageStar

# Create from image bounds
image_star = ImageStar.from_bounds(
    lb_image,  # (height, width, channels)
    ub_image,  # (height, width, channels)
    height=28, width=28, num_channels=1
)

# Flatten for FC layers
regular_star = image_star.flatten_to_star()
```

### Zonotopes

Efficient representation: **x = c + V*α** where **-1 ≤ α_i ≤ 1**

```python
from n2v.sets import Zono

# From center and generators
zono = Zono(center, generators)

# From bounds (creates zonotope)
zono = Zono.from_bounds(lb, ub)

# Fast over-approximation
bounds_lb, bounds_ub = zono.get_bounds()
```

---

## Reachability Methods

### Exact Methods

#### `'exact-star'`
- **Sound and complete** (exact reachability)
- Uses Star sets with ReLU splitting
- Best for: Small-medium networks, safety-critical applications
- Note: Can be slow due to exponential splitting

```python
output = verifier.reach(input_star, method='exact-star')
```

### Approximate Methods

#### `'approx-star'`
- Over-approximate using relaxed Star sets
- Faster than exact (no splitting)
- Best for: Large networks, quick verification

```python
output = verifier.reach(input_star, method='approx-star')
```

#### `'approx-zono'`
- Over-approximate using Zonotopes
- Very fast, more conservative
- Best for: Very large networks, initial screening

```python
output = verifier.reach(input_star, method='approx-zono')
```

### Method Comparison

| Method | Speed | Precision | Use Case |
|--------|-------|-----------|----------|
| `exact-star` | Slow | Exact | Safety-critical, small nets |
| `approx-star` | Medium | Tight | Large networks |
| `approx-zono` | Fast | Conservative | Very large nets, screening |

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
from n2v.nn.layers.layer_reach import reach_layer_star

layers = list(model.children())
current_stars = [input_star]

for i, layer in enumerate(layers):
    print(f"Layer {i}: {type(layer).__name__}")
    print(f"  Input: {len(current_stars)} stars")

    current_stars = reach_layer_star(
        layer, current_stars,
        method='exact',
        lp_solver='default'
    )

    print(f"  Output: {len(current_stars)} stars")
```

---

## Examples

See [`examples/`](examples/) directory for complete examples:

### Basic Examples
- **`simple_verification.py`** - Basic feedforward network
- **`mnist_verification.py`** - MNIST classifier verification

### CNN Examples
- **`mnist_cnn_verification.py`** - CNN without pooling
- **`mnist_cnn_maxpool_verification.py`** - CNN with MaxPool2D
- **`mnist_cnn_avgpool_verification.py`** - CNN with AvgPool2D ⭐ (recommended!)

### Running Examples

```bash
cd examples

# Basic verification
python simple_verification.py

# CNN with AvgPool (fastest!)
python mnist_cnn_avgpool_verification.py
```

See [examples/README.md](examples/README.md) for detailed documentation.

---

## Performance Tips

### 1. ⭐ Use AvgPool2D Instead of MaxPool2D

```python
# Instead of:
nn.MaxPool2d(2, 2)  # Can cause exponential splitting

# Use:
nn.AvgPool2d(2, 2)  # Always exact, no splitting!
```

**Speedup**: 10-100x faster for verification!

### 2. Use Smaller Perturbations

```python
epsilon = 0.01  # Instead of 0.1
```

Reduces ReLU splitting significantly.

### 3. Choose Appropriate Method

```python
# For small networks (< 1000 neurons)
method = 'exact-star'

# For medium networks (1000-10000 neurons)
method = 'approx-star'

# For large networks (> 10000 neurons)
method = 'approx-zono'
```

### 4. Use Approximate Methods for Early Layers

```python
# Exact for critical layers only
verifier.reach(input_star, method='exact-star', exact_layers=[5, 6, 7])
```

### 5. Monitor Star Count

```python
for layer in layers:
    current_stars = reach_layer_star(layer, current_stars)
    if len(current_stars) > 1000:
        print(f"Warning: {len(current_stars)} stars - consider approximate method")
```

---

## Testing

### Running Tests

NNV-Python includes comprehensive unit and soundness tests (200+ total tests) using pytest.

```bash
# Run all tests (unit + soundness)
pytest n2v/tests/

# Run with verbose output
pytest n2v/tests/ -v

# Run only unit tests
pytest n2v/tests/unit/

# Run only soundness tests
pytest n2v/tests/soundness/

# Run specific test file
pytest n2v/tests/unit/test_sets.py
pytest n2v/tests/soundness/test_soundness_relu.py -v

# Run specific test
pytest n2v/tests/unit/test_sets.py::TestStar::test_from_bounds -v
```

### Test Structure

```
n2v/tests/
├── conftest.py              # Shared fixtures and helpers
│
├── unit/                    # Unit tests (~130 tests)
│   ├── conftest.py          # Unit test fixtures
│   ├── test_sets.py         # Star, Zono, Box tests
│   ├── test_image_sets.py   # ImageStar, ImageZono tests
│   ├── test_layer_ops.py    # Layer operation tests
│   ├── test_dispatcher.py   # Layer dispatcher tests
│   ├── test_load_vnnlib.py  # VNN-LIB format parsing tests
│   ├── test_verify_specification.py  # Property verification tests
│   └── test_integration.py  # Full network verification tests
│
└── soundness/               # Soundness tests (~80 tests)
    ├── README.md            # Soundness testing methodology
    ├── test_soundness_linear.py     # Linear layer soundness
    ├── test_soundness_relu.py       # ReLU activation soundness
    ├── test_soundness_conv2d.py     # Conv2D soundness
    ├── test_soundness_maxpool2d.py  # MaxPool2D soundness
    ├── test_soundness_avgpool2d.py  # AvgPool2D soundness
    └── test_soundness_flatten.py    # Flatten soundness
```

### Test Coverage

**Unit Tests** verify implementation correctness:
- Set Representations: Star, Zono, Box creation and operations
- Image Sets: ImageStar, ImageZono with spatial operations
- Layer Operations: Linear, ReLU, Conv2D, MaxPool2D, AvgPool2D, Flatten
- Dispatcher: Layer-type routing and method selection
- VNN-LIB: Property file parsing
- Integration: End-to-end network verification workflows

**Soundness Tests** verify mathematical correctness:
- Exact vs. approximate methods produce sound results
- Over-approximations contain exact reachable sets
- Bounds computation is correct
- Ground truth validation for simple cases

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
pytest --cov=n2v --cov-report=html n2v/tests/

# View in browser
open htmlcov/index.html
```

### Current Test Status

- **Unit Tests**: 124 passing, 5 skipped (unimplemented features)
- **Soundness Tests**: 79 passing
- **Total**: 203 passing tests

Skipped tests are for features not yet implemented (e.g., `Box.contains()`, `Box.intersect()`, `Box.union()`, `Zono.reduce_order()`). These can be implemented in the future as needed.

---

## API Reference

### Core Classes

#### `NeuralNetwork`

```python
verifier = nnv.NeuralNetwork(pytorch_model)

# Reachability
output_stars = verifier.reach(input_set, method='exact-star')

# With options
output_stars = verifier.reach(
    input_set,
    method='exact-star',
    lp_solver='default',
    dis_opt='display'  # Show progress
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
lb, ub = star.get_range(dim_index)
star.estimate_ranges()  # Compute all bounds
```

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
from n2v.nn.layers.layer_reach import reach_layer_star

# Dispatch to appropriate layer operation
output = reach_layer_star(
    layer,          # PyTorch layer (nn.Linear, nn.ReLU, etc.)
    input_stars,    # List of input Stars
    method='exact', # 'exact' or 'approx'
    **kwargs        # lp_solver, dis_opt, etc.
)
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
- [Star Set Reachability](https://arxiv.org/abs/1908.01739)

---

## Development & Contributing

### Project Structure

```
n2v/
├── sets/              # Set representations (Star, Zono, Box, etc.)
├── nn/                # Neural network wrapper and layer operations
│   ├── neural_network.py
│   └── layers/
│       ├── layer_reach.py      # Layer dispatcher
│       └── operations/         # Layer-specific operations
│           ├── linear_reach.py
│           ├── relu_reach.py
│           ├── conv2d_reach.py
│           ├── avgpool2d_reach.py
│           └── maxpool2d_reach.py
├── utils/             # Utilities (LP solver, etc.)
└── examples/          # Example scripts
```

### Adding New Layers

1. Create operation file in `nn/layers/operations/`
2. Add case to `layer_reach.py` dispatcher
3. Implement for Star, Zono, Box (as applicable)

See [CHANGELOG.md](CHANGELOG.md) for development history.

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
- **Email**: diego.manzanas.lopez@vanderbilt.edu
- **Documentation**: This README and [CHANGELOG.md](CHANGELOG.md)

---

## Quick Links

- 📖 [Examples README](examples/README.md) - How to run examples
- 📋 [CHANGELOG](CHANGELOG.md) - Development history and changes
- 🚀 [Quick Start](#quick-start) - Get started in 5 minutes
- 🧠 [CNN Verification](#cnn-verification) - CNN-specific guide

---

**NNV-Python**: Bringing formal verification to PyTorch! 🚀
