# Soundness Tests

This directory contains soundness tests for the NNV-Python verification tool. These tests verify that the tool produces **correct** results, not just that the code runs without errors.

## Purpose

**Soundness tests** ensure that:
1. Reachability analysis correctly computes the exact or over-approximated reachable set
2. Every layer operation produces mathematically correct results
3. Edge cases are handled properly
4. Results match known ground truth for simple examples

These are different from **unit tests** which check:
- Code compiles and runs
- API contracts are met
- No crashes on edge cases

## Test Organization

```
soundness/
├── README.md                          # This file
├── test_soundness_linear.py           # Linear layer soundness tests
├── test_soundness_relu.py             # ReLU layer soundness tests
├── test_soundness_conv2d.py           # Conv2d layer soundness tests
├── test_soundness_pooling.py          # Pooling layer soundness tests
├── test_soundness_other_layers.py     # Other layers (Flatten, etc.)
├── test_soundness_integration.py      # Multi-layer network tests
├── test_soundness_probabilistic.py    # Probabilistic verification soundness
└── ground_truth/                      # Ground truth data
    ├── linear_examples.json
    ├── relu_examples.json
    └── ...
```

## Test Methodology

For each layer operation and set representation:

1. **Define Ground Truth**: Create simple examples where the exact output is known
2. **Test Exact Methods**: Verify exact reachability produces correct results
3. **Test Approximate Methods**: Verify over-approximation contains exact result
4. **Test Edge Cases**: Empty sets, boundary points, degenerate cases
5. **Test All Combinations**: Every set type × every layer × every method

### Set Types to Test
- **Star**: Polytope representation
- **Zono**: Zonotope representation
- **Box**: Box (hyperrectangle) representation

### Layer Types to Test
- Linear (fully connected)
- ReLU activation
- Conv2d (convolutional)
- MaxPool2d, AvgPool2d
- Flatten, BatchNorm, etc.

### Reachability Methods
- **Exact**: Produces exact reachable set
- **Approx**: Over-approximation methods (various relaxation factors)

## Example Test Structure

```python
def test_linear_star_exact():
    """Test exact reachability through Linear layer with Star set.

    Ground truth: Linear transformation y = Wx + b
    - Input: Star representing [0,1] x [0,1]
    - Layer: Linear(2, 2) with W = [[1, 0], [0, 1]], b = [1, 1]
    - Expected: Star representing [1,2] x [1,2]
    """
    # Create simple input set
    lb = np.array([[0.0], [0.0]])
    ub = np.array([[1.0], [1.0]])
    input_star = Star.from_bounds(lb, ub)

    # Create layer with known weights
    layer = nn.Linear(2, 2)
    layer.weight.data = torch.eye(2)
    layer.bias.data = torch.ones(2)

    # Compute reachability
    output_stars = linear_star(layer, [input_star])

    # Verify ground truth
    assert len(output_stars) == 1
    output_lb, output_ub = output_stars[0].estimate_ranges()

    # Expected: [0,1] -> [0+1, 1+1] = [1,2]
    assert np.allclose(output_lb, [[1.0], [1.0]], atol=1e-6)
    assert np.allclose(output_ub, [[2.0], [2.0]], atol=1e-6)
```

## Soundness Properties to Verify

### 1. Correctness (Exact Methods)
For exact reachability:
```
Output set = { f(x) | x ∈ Input set }
```

### 2. Soundness (Over-approximation)
For approximate reachability:
```
{ f(x) | x ∈ Input set } ⊆ Output set
```

### 3. Non-emptiness Preservation
```
If Input set is non-empty, then Output set should be non-empty (unless impossible)
```

### 4. Bound Correctness
```
For all x in Input: lb ≤ f(x) ≤ ub
```

### 5. Monotonicity (for ReLU)
```
If Input1 ⊆ Input2, then ReLU(Input1) ⊆ ReLU(Input2)
```

### 6. Probabilistic Coverage Guarantee
For probabilistic verification with ⟨ε, ℓ, m⟩ parameters:
```
With high probability (confidence δ₂), at least (1-ε) fraction of
outputs from the input set are contained in the ProbabilisticBox.
```

Probabilistic soundness tests (`test_soundness_probabilistic.py`):
- Verify coverage guarantee holds empirically (sample N >> m points)
- Verify confidence guarantee holds across repeated runs
- Compare naive vs clipping_block surrogate tightness

## Running Soundness Tests

```bash
# Run all soundness tests
pytest tests/soundness/ -v

# Run specific soundness test suite
pytest tests/soundness/test_soundness_linear.py -v

# Run with detailed output
pytest tests/soundness/ -v -s

# Run only failed tests from last run
pytest tests/soundness/ --lf
```

## Adding New Soundness Tests

When adding a new layer operation or method:

1. Create test file: `test_soundness_<layer>.py`
2. For each set type (Star, Zono, Box):
   - Test exact method with known ground truth
   - Test approx methods verify over-approximation
   - Test edge cases (empty, degenerate, boundary)
3. Add ground truth examples to `ground_truth/`
4. Update this README

## Test Coverage Goals

- ✅ Every layer operation × set type × method combination
- ✅ All edge cases (empty sets, zero inputs, large values)
- ✅ Boundary conditions (ReLU split points, etc.)
- ✅ Known counterexamples from literature
- ✅ Integration tests with multiple layers

## References

1. Star Set: Tran et al. "Star-Based Reachability Analysis of Deep Neural Networks" (FM 2019)
2. Zonotope: Ghorbal et al. "Characterizing the Accuracy of the Zonotope Abstract Domain" (NASA 2009)
3. Verification Benchmarks: VNN-COMP 2020-2024 benchmarks
4. Probabilistic Verification: Hashemi et al. "Scaling Data-Driven Probabilistic Reachability Analysis" (ICLR 2026)
