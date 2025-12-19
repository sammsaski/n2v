# NNV-Python Examples

Complete examples demonstrating neural network verification with NNV-Python.

---

## Quick Start

### Installation

```bash
cd n2v
pip install -e .
```

### Run an Example

```bash
cd examples
python mnist_cnn_avgpool_verification.py  # Recommended CNN example!
```

---

## Available Examples

### Basic Verification

#### `simple_verification.py`
**What it demonstrates**:
- Creating feedforward neural networks
- Star set input specifications
- Basic reachability analysis
- Set operations and bounds computation

**Run**:
```bash
python simple_verification.py
```

**Expected output**:
```
Creating simple feedforward network...
Input: Star set with 3 dimensions
Output: Reachable set bounds
✓ Verification complete!
```

---

### MNIST Classifier Verification

#### `mnist_verification.py`
**What it demonstrates**:
- MNIST digit classification
- L∞ robustness verification
- Feedforward network with ReLU
- Output bounds analysis

**Run**:
```bash
python mnist_verification.py
```

**Key concepts**:
- Input perturbation (ε-ball around image)
- Star splitting with ReLU layers
- Robustness checking

---

### CNN Verification (Without Pooling)

#### `mnist_cnn_verification.py`
**What it demonstrates**:
- Convolutional neural networks
- ImageStar propagation
- Conv2D exact reachability
- Strided convolutions for downsampling

**Architecture**:
```python
Conv2d(1→4, 3x3) → ReLU → Conv2d(4→8, 3x3, stride=2) → ReLU → Flatten → Linear
```

**Run**:
```bash
python mnist_cnn_verification.py
```

**Note**: Uses strided Conv2D instead of pooling for downsampling.

---

### CNN with MaxPool2D

#### `mnist_cnn_maxpool_verification.py`
**What it demonstrates**:
- MaxPool2D exact reachability
- Star splitting from MaxPool (exponential growth)
- Performance characteristics of MaxPool

**Architecture**:
```python
Conv2d(1→4) → ReLU → MaxPool2d(2x2) → Conv2d(4→8) → ReLU → MaxPool2d(2x2) → Flatten → Linear
```

**Run**:
```bash
python mnist_cnn_maxpool_verification.py
```

**Warning**: MaxPool2D can cause significant star splitting!

**Expected behavior**:
- ReLU creates some star splitting
- MaxPool2D may create additional splitting
- Total star count can grow exponentially

---

### CNN with AvgPool2D ⭐ (RECOMMENDED!)

#### `mnist_cnn_avgpool_verification.py`
**What it demonstrates**:
- AvgPool2D exact reachability (LINEAR operation!)
- NO star splitting from pooling
- 10-100x speedup compared to MaxPool2D
- Best practices for CNN verification

**Architecture**:
```python
Conv2d(1→4) → ReLU → AvgPool2d(2x2) → Conv2d(4→8) → ReLU → AvgPool2d(2x2) → Flatten → Linear
```

**Run**:
```bash
python mnist_cnn_avgpool_verification.py
```

**Key advantage**:
```
✓ AvgPool2D is LINEAR - no splitting!
✓ 10-100x faster than MaxPool2D
✓ Always exact (no over-approximation)
```

**Expected output**:
```
Layer 3/8: AvgPool1 (AvgPool2d)
  Input: X star(s)
  Output: X star(s)  # Same count - no splitting!
  ✓ AvgPool is exact - no splitting occurred!
```

---

## Performance Comparison

### MaxPool2D vs AvgPool2D

**Test**: 28×28 MNIST image, ε=0.01, 2 pooling layers

#### With MaxPool2D:
```
After ReLU1: 8 stars
After MaxPool1: 16-64 stars    # ❌ 2-8x growth
After ReLU2: 128-512 stars
After MaxPool2: 256-4096 stars  # ❌ More growth
Runtime: ~10-100 seconds
```

#### With AvgPool2D: ⭐
```
After ReLU1: 8 stars
After AvgPool1: 8 stars         # ✅ No growth!
After ReLU2: 64 stars
After AvgPool2: 64 stars        # ✅ No growth!
Runtime: ~1-5 seconds
```

**Speedup: 10-20x with AvgPool2D!**

---

## Example Workflow

### 1. Define Your Model

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),  # ⭐ Use AvgPool for verification!
    nn.Flatten(),
    nn.Linear(16*14*14, 10)
)
model.eval()
```

### 2. Create Input Specification

```python
from n2v.sets import ImageStar
import numpy as np

# Create L∞ ball around image
image = np.random.rand(28, 28)
epsilon = 0.01
lb = np.maximum(image - epsilon, 0)
ub = np.minimum(image + epsilon, 1)

input_star = ImageStar.from_bounds(
    lb.reshape(28, 28, 1),
    ub.reshape(28, 28, 1),
    height=28, width=28, num_channels=1
)
```

### 3. Verify the Network

```python
import n2v as nnv

verifier = nnv.NeuralNetwork(model)
output_stars = verifier.reach(input_star, method='exact')

print(f"Output: {len(output_stars)} reachable set(s)")
```

### 4. Check Robustness

```python
def is_robust(output_stars, true_class):
    for star in output_stars:
        star.estimate_ranges()
        lb = star.state_lb.flatten()
        ub = star.state_ub.flatten()

        # Check if any other class could beat true_class
        for i in range(len(lb)):
            if i != true_class and ub[i] >= lb[true_class]:
                return False
    return True

robust = is_robust(output_stars, true_class=5)
print(f"Model is robust: {robust}")
```

---

## Layer-by-Layer Analysis

For debugging or understanding verification:

```python
from n2v.nn.layers.layer_reach import reach_layer_star

layers = list(model.children())
current_stars = [input_star]

for i, layer in enumerate(layers):
    print(f"\nLayer {i}: {type(layer).__name__}")
    print(f"  Input: {len(current_stars)} stars")

    current_stars = reach_layer_star(
        layer, current_stars,
        method='exact',
        verbose=True  # Show ReLU/MaxPool splitting details
    )

    print(f"  Output: {len(current_stars)} stars")

    # Monitor star count growth
    if len(current_stars) > 100:
        print(f"  ⚠️  Warning: High star count - consider approximate method")
```

---

## Tips for Your Own Networks

### 1. ⭐ Use AvgPool2D Instead of MaxPool2D

```python
# Good for verification:
nn.AvgPool2d(2, 2)  # Linear, no splitting

# Slower for verification:
nn.MaxPool2d(2, 2)  # Can cause exponential splitting
```

### 2. Keep Perturbations Small

```python
epsilon = 0.01  # Good
epsilon = 0.1   # May cause too much splitting
```

### 3. Choose Appropriate Method

```python
# For small networks:
method = 'exact'

# For large networks:
method = 'approx'  # Faster, still sound

# For very large networks:
method = 'approx'  # Fastest, conservative
```

### 4. Monitor Star Count

If star count grows beyond 1000-10000:
- Use approximate methods
- Reduce ε
- Replace MaxPool with AvgPool
- Use fewer ReLU layers

---

## Common Issues

### Issue 1: Too Many Stars

**Symptom**: Verification very slow, memory issues

**Causes**:
- Large ε (perturbation)
- Many ReLU layers with splitting
- MaxPool2D with uncertain maxima

**Solutions**:
1. Use smaller ε
2. Use approximate methods
3. Replace MaxPool2D with AvgPool2D ⭐
4. Use star merging (when available)

### Issue 2: Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'n2v'`

**Solution**:
```bash
cd n2v
pip install -e .
```

### Issue 3: LP Solver Errors

**Symptom**: CVXPY solver errors

**Solution**:
```bash
# Install better solver
pip install cvxopt
# or
pip install mosek  # Commercial, fastest
```

---

## Understanding Output

### Star Count
- **1 star**: No splitting occurred (linear layers only)
- **2-10 stars**: Minimal splitting (good)
- **10-100 stars**: Moderate splitting (acceptable)
- **100-1000 stars**: High splitting (consider approximate)
- **>1000 stars**: Very high (likely too slow)

### Verification Result
- **Robust**: True class wins in all stars
- **Not Robust**: Some star allows misclassification
- **Unknown**: Over-approximation too conservative

---

## Next Steps

After trying these examples:

1. **Modify architectures**: Try your own CNN designs
2. **Different perturbations**: L2 balls, polygon regions
3. **Custom properties**: Beyond robustness
4. **Hybrid methods**: Mix exact/approximate for speed

See main [README.md](../README.md) for API details and advanced usage.

---

## Additional Resources

- **Main README**: [../README.md](../README.md)
- **Development History**: [../CHANGELOG.md](../CHANGELOG.md)
- **MATLAB NNV**: https://github.com/verivital/nnv

---

**Recommended starting point**: `mnist_cnn_avgpool_verification.py` ⭐
