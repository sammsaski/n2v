# MNIST Classification and Verification Examples

Complete workflow for training and formally verifying MNIST classifiers using NNV-Python.

---

## Overview

This directory contains **4 Jupyter notebooks** demonstrating:

1. **Training** fully connected and CNN classifiers on MNIST
2. **Verifying** local robustness against adversarial perturbations
3. **Comparing** AvgPool2D vs MaxPool2D for verification efficiency

---

## Notebooks

### 1. Train Fully Connected Classifier
**File**: `01_train_fc_mnist.ipynb`

- Architecture: 784 → 128 → 64 → 10
- Training: 10 epochs with Adam optimizer
- Outputs: Trained model saved to `models/mnist_fc_classifier.pth`

**What you'll learn**:
- Loading and preprocessing MNIST data
- Defining a simple feedforward network
- Training loop with PyTorch
- Evaluating test accuracy

---

### 2. Verify Fully Connected Classifier
**File**: `02_verify_fc_mnist.ipynb`

- Loads trained FC model
- Creates Star set for perturbed input region
- Performs exact reachability analysis
- Verifies local robustness

**What you'll learn**:
- Creating Star sets from input bounds
- Using `reach_star_exact()` for verification
- Computing output bounds
- Checking robustness properties
- Analyzing safety margins

**Key Concepts**:
- ReLU layers cause star splitting
- Smaller ε → easier to verify
- Safety margin = true_class_lb - other_class_ub

---

### 3. Train CNN Classifier
**File**: `03_train_cnn_mnist.ipynb`

- Architecture: Conv(8) → ReLU → **AvgPool** → Conv(16) → ReLU → **AvgPool** → FC(10)
- **Uses AvgPool2D instead of MaxPool2D** ⭐
- Training: 10 epochs with Adam optimizer
- Outputs: Trained model saved to `models/mnist_cnn_classifier.pth`

**What you'll learn**:
- Building CNNs with PyTorch
- Why AvgPool2D is better for verification
- Visualizing learned filters
- Feature map visualization

**⭐ Design Decision**:
```python
# Instead of:
nn.MaxPool2d(2, 2)  # Non-linear → exponential splitting

# Use:
nn.AvgPool2d(2, 2)  # Linear → no splitting!
```

**Result**: 10-100x faster verification with similar accuracy!

---

### 4. Verify CNN Classifier
**File**: `04_verify_cnn_mnist.ipynb`

- Loads trained CNN model
- Creates **ImageStar** set (preserves spatial structure)
- Performs exact reachability analysis
- Layer-by-layer tracking of star growth
- Compares AvgPool vs MaxPool verification speed

**What you'll learn**:
- Using ImageStar for CNN verification
- Layer-by-layer reachability analysis
- Tracking star splitting through network
- Performance comparison: AvgPool vs MaxPool

**Key Results**:
- AvgPool2D contributes **0 star splits** (linear!)
- MaxPool2D causes **exponential splitting** (non-linear)
- **Speedup**: 10-100x faster with AvgPool
- Only ReLU layers split stars in AvgPool networks

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy matplotlib cvxpy scipy
pip install -e ../../  # Install n2v
```

### 2. Run Notebooks in Order

```bash
jupyter notebook
```

Open notebooks in sequence:
1. `01_train_fc_mnist.ipynb` → Train FC model
2. `02_verify_fc_mnist.ipynb` → Verify FC model
3. `03_train_cnn_mnist.ipynb` → Train CNN model
4. `04_verify_cnn_mnist.ipynb` → Verify CNN model

---

## Expected Outputs

### Trained Models

After running training notebooks:
```
examples/MNIST/models/
├── mnist_fc_classifier.pth    # FC model (~98% accuracy)
└── mnist_cnn_classifier.pth   # CNN model (~99% accuracy)
```

### Verification Results

Typical verification results:

**FC Model** (ε=0.02):
- Computation time: ~30-60s
- Output stars: 50-200 (depends on ReLU splitting)
- Robustness: Depends on test image

**CNN Model** (ε=0.02):
- Computation time: ~20-40s (faster than MaxPool!)
- Output stars: 30-100 (only from ReLU layers)
- **Key**: AvgPool adds 0 splits!

---

## Architecture Comparison

### Fully Connected Network

```
Input: 28×28 image (flattened to 784)
│
├─ Linear(784, 128)
├─ ReLU                    ⚠️ Splits stars
├─ Linear(128, 64)
├─ ReLU                    ⚠️ Splits stars
├─ Linear(64, 10)
│
Output: 10 class scores
```

**Verification Characteristics**:
- Simple architecture
- Moderate splitting (2 ReLU layers)
- Good for understanding basics

---

### CNN Network (with AvgPool)

```
Input: 28×28×1 image
│
├─ Conv2d(1→8, 3×3, pad=1)    28×28×8  (exact, no split)
├─ ReLU                        28×28×8  ⚠️ Splits stars
├─ AvgPool2d(2×2)              14×14×8  ✅ No split! (linear)
├─ Conv2d(8→16, 3×3, pad=1)   14×14×16 (exact, no split)
├─ ReLU                        14×14×16 ⚠️ Splits stars
├─ AvgPool2d(2×2)              7×7×16   ✅ No split! (linear)
├─ Flatten                     784      (no split)
├─ Linear(784, 10)             10       (exact, no split)
│
Output: 10 class scores
```

**Verification Characteristics**:
- Only ReLU layers split stars
- AvgPool2D: 0 splits (linear operation)
- **10-100x faster** than MaxPool version
- Practical for real-world verification

---

### CNN Network (with MaxPool) - Not Recommended

```
Input: 28×28×1 image
│
├─ Conv2d(1→8, 3×3, pad=1)    28×28×8  (exact, no split)
├─ ReLU                        28×28×8  ⚠️ Splits stars
├─ MaxPool2d(2×2)              14×14×8  ❌ SPLITS! (non-linear)
├─ Conv2d(8→16, 3×3, pad=1)   14×14×16 (exact, no split)
├─ ReLU                        14×14×16 ⚠️ Splits stars
├─ MaxPool2d(2×2)              7×7×16   ❌ SPLITS! (non-linear)
├─ Flatten                     784      (no split)
├─ Linear(784, 10)             10       (exact, no split)
│
Output: 10 class scores
```

**Verification Characteristics**:
- ReLU **AND** MaxPool split stars
- Exponential star growth
- **10-100x slower** than AvgPool
- Often impractical for verification

---

## Key Concepts Demonstrated

### 1. Set-Based Verification

**Star Sets**:
- Represent sets of inputs: `{x | x = c + V*α, C*α ≤ d}`
- Exact representation with linear constraints
- Propagate through network layers

**ImageStar**:
- Star sets with spatial structure (H×W×C)
- Efficient for CNN verification
- Preserves image dimensions

### 2. Reachability Analysis

```python
from n2v.nn.reach.reach_star import reach_star_exact

output_stars = reach_star_exact(model, [input_star])
```

Computes **exact** output reachable set:
- All possible outputs for inputs in input_star
- Sound and complete verification
- May split stars at non-linear layers

### 3. Local Robustness Verification

**Goal**: Verify that all inputs within ε of a point get same classification

**Method**:
1. Create input set: `[image - ε, image + ε]`
2. Compute output reachable set
3. Check: `true_class_lb > all_other_classes_ub`

**Result**:
- ✅ **Robust**: No adversarial examples in region
- ❌ **Not Robust**: Potential adversarial examples exist

### 4. Star Splitting

**Where it happens**:
- **ReLU layers**: Split at uncertain neurons
- **MaxPool2D layers**: Split at uncertain max operations
- **AvgPool2D layers**: Never split (linear!)

**Example**:
```
ReLU on 1 star with 10 uncertain neurons → up to 2^10 = 1024 stars
AvgPool on 1 star → exactly 1 star (no splitting!)
```

### 5. Performance Trade-offs

| Layer Type | Exact? | Splits? | Speed |
|------------|--------|---------|-------|
| Linear | ✅ Yes | ❌ No | Fast |
| Conv2D | ✅ Yes | ❌ No | Fast |
| **AvgPool2D** | ✅ Yes | ❌ No | **Fast** ⭐ |
| ReLU | ✅ Yes | ✅ Yes | Slow |
| MaxPool2D | ✅ Yes | ✅ Yes | Very Slow |

**Design Recommendation**: Use AvgPool2D for verifiable CNNs!

---

## Typical Workflow

### 1. Train a Model

```python
# 01_train_fc_mnist.ipynb or 03_train_cnn_mnist.ipynb
model = YourModel()
train(model, train_loader)
torch.save(model.state_dict(), 'model.pth')
```

### 2. Define Input Region

```python
# For FC networks
from n2v.sets import Star
epsilon = 0.02
lb = np.clip(image - epsilon, 0, 1)
ub = np.clip(image + epsilon, 0, 1)
input_star = Star.from_bounds(lb, ub)

# For CNNs
from n2v.sets import ImageStar
input_star = ImageStar.from_bounds(
    lb, ub, height=28, width=28, num_channels=1
)
```

### 3. Verify

```python
from n2v.nn.reach.reach_star import reach_star_exact

output_stars = reach_star_exact(model, [input_star])

# Compute bounds
for star in output_stars:
    star.estimate_ranges()

overall_lb = np.min([s.state_lb for s in output_stars], axis=0)
overall_ub = np.max([s.state_ub for s in output_stars], axis=0)
```

### 4. Check Robustness

```python
true_class = 5
robust = all(
    overall_lb[true_class] > overall_ub[i]
    for i in range(10) if i != true_class
)

if robust:
    print("✅ Verified robust!")
else:
    print("❌ Not robust - adversarial examples may exist")
```

---

## Performance Tips

### 1. Use AvgPool2D Instead of MaxPool2D

```python
# 10-100x faster!
nn.AvgPool2d(2, 2)  # instead of nn.MaxPool2d(2, 2)
```

### 2. Start with Small ε

```python
# Easier to verify
epsilon = 0.01  # instead of 0.1
```

### 3. Monitor Star Count

```python
if len(current_stars) > 1000:
    print("Warning: Too many stars, consider approximate method")
```

### 4. Use Layer-by-Layer Analysis

```python
from n2v.nn.layer_ops.dispatcher import reach_layer

# Track splitting to identify bottlenecks
for layer in model:
    current_stars = reach_layer(layer, current_stars)
    print(f"{layer}: {len(current_stars)} stars")
```

---

## Troubleshooting

### Issue: Verification takes too long

**Solutions**:
- Reduce ε (smaller perturbation)
- Use AvgPool2D instead of MaxPool2D
- Try approximate methods (coming soon)
- Verify fewer layers exactly

### Issue: Too many output stars

**Cause**: Excessive ReLU splitting

**Solutions**:
- Reduce network width
- Use smaller input regions
- Monitor star count per layer

### Issue: Out of memory

**Solutions**:
- Reduce batch of input stars
- Use approximate methods
- Verify on CPU (more memory)

### Issue: CVXPY solver warnings

**Cause**: LP solver convergence issues

**Solutions**:
- Usually harmless, results still valid
- Try different solver: `lp_solver='GLPK'`
- Check constraint matrix conditioning

---

## Learning Path

### Beginner
1. Run `01_train_fc_mnist.ipynb` - Understand MNIST training
2. Run `02_verify_fc_mnist.ipynb` - Learn basic verification
3. Understand Star sets and reachability

### Intermediate
4. Run `03_train_cnn_mnist.ipynb` - CNNs with AvgPool
5. Run `04_verify_cnn_mnist.ipynb` - CNN verification
6. Understand ImageStar and layer-by-layer analysis

### Advanced
7. Experiment with different architectures
8. Compare AvgPool vs MaxPool performance
9. Try different perturbation magnitudes
10. Analyze safety margins and robustness

---

## Expected Runtime

On a typical laptop (CPU):

| Notebook | Training Time | Verification Time |
|----------|--------------|-------------------|
| FC Training | ~5 min | - |
| FC Verification | - | ~30-60s (ε=0.02) |
| CNN Training | ~10 min | - |
| CNN Verification | - | ~20-40s (ε=0.02) |

**Note**: Verification time increases with:
- Larger ε (more uncertain neurons)
- Deeper networks (more layers)
- MaxPool instead of AvgPool (exponential!)

---

## References

### NNV-Python Documentation
- Main README: `../../README.md`

### Key Papers
- NNV Tool: [MATLAB NNV Repository](https://github.com/verivital/nnv)
- Star Sets: Tran et al., "Star-Based Reachability Analysis"
- Verification: "Formal Verification of Neural Networks"

### Related Tools
- PyTorch: https://pytorch.org/
- CVXPY: https://www.cvxpy.org/

---

## Next Steps

After completing these examples:

1. **Experiment**: Try different architectures, ε values, test images
2. **Extend**: Add more layers, try other datasets
3. **Compare**: Benchmark AvgPool vs MaxPool on your models
4. **Apply**: Verify your own trained models

---

**Happy Verifying!** 🚀

For questions or issues, see the main [README](../../README.md) or open an issue on GitHub.
