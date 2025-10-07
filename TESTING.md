# NNV-Python Testing Guide

Complete testing infrastructure for neural network verification.

---

## Quick Start

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run Tests

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Run specific test file
pytest tests/test_sets.py

# Run with coverage
pytest --cov=nnv_py --cov-report=html
```

---

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test helpers
├── test_sets.py             # Star, Zono, Box set tests (25 tests)
├── test_image_sets.py       # ImageStar, ImageZono tests (17 tests)
├── test_layer_ops.py        # Layer operation tests (27 tests)
├── test_dispatcher.py       # Dispatcher system tests (18 tests)
└── test_integration.py      # End-to-end verification tests (15 tests)
```

**Total: 102+ unit and integration tests**

---

## Test Categories

### 1. Set Representations (`test_sets.py`)

Tests for core set representations:

**Star Sets**:
- Creation and validation
- From bounds conversion
- Affine transformations
- Half-space intersections
- Bounds computation
- Point containment
- Range estimation

**Zonotopes**:
- Creation and validation
- From bounds conversion
- Affine transformations
- Minkowski sum
- Order reduction
- Bounds computation

**Boxes**:
- Creation and validation
- Invalid bounds detection
- Affine transformations
- Point containment
- Set intersection
- Set union

**Set Conversions**:
- Star → Box
- Zono → Box
- Box → Zono

### 2. Image Sets (`test_image_sets.py`)

Tests for spatial set representations:

**ImageStar**:
- Multi-channel support
- Flattening operations
- Affine maps
- Bounds computation
- Dimension validation

**ImageZono**:
- Multi-channel support
- Flattening operations
- Affine maps
- Bounds computation

**Image Operations**:
- Flatten preserves bounds
- Multi-channel handling
- Reshape validation

### 3. Layer Operations (`test_layer_ops.py`)

Tests for neural network layers:

**Linear Layers**:
- Star propagation (exact)
- Zono propagation (exact)
- Box propagation (exact)
- Exactness verification (identity test)

**ReLU Activation**:
- Always active neurons (no splitting)
- Always inactive neurons (zeroing)
- Uncertain neurons (splitting behavior)
- Zonotope approximation
- Box propagation

**Conv2D Layers**:
- Star propagation (exact)
- Exactness verification
- Strided convolutions
- Zonotope propagation

**MaxPool2D Layers**:
- Exact method (with splitting)
- Approximate method (no splitting)
- Zonotope propagation

**AvgPool2D Layers** ⭐:
- Exact propagation
- **No splitting property** (key test!)
- Exactness verification
- Zonotope propagation

**Flatten Layers**:
- ImageStar → Star conversion
- ImageZono → Zono conversion
- Dimension preservation

**Layer Combinations**:
- Conv2D → ReLU sequence
- Conv2D → AvgPool → Flatten (typical CNN)

### 4. Dispatcher System (`test_dispatcher.py`)

Tests for layer routing:

**Star Dispatcher**:
- Linear dispatch
- ReLU dispatch
- Conv2D dispatch
- MaxPool2D dispatch
- AvgPool2D dispatch
- Flatten dispatch
- Sequential container handling
- Unknown layer error handling

**Zono Dispatcher**:
- All layer types with Zono sets
- Approximate method routing

**Box Dispatcher**:
- Supported layer types
- Box-specific operations

**Options Testing**:
- Exact vs approximate methods
- Display output verification

### 5. Integration Tests (`test_integration.py`)

End-to-end verification tests:

**Feedforward Networks**:
- Simple feedforward (exact method)
- Bounds preservation (identity network)
- Multiple ReLU layers (splitting accumulation)

**CNN Networks**:
- Simple CNN with exact method
- AvgPool no-splitting verification
- Strided convolution networks

**Robustness Verification**:
- Local robustness checking
- Adversarial example detection
- Decision boundary analysis

**Approximate Methods**:
- Zonotope approximate verification
- Speed comparison (approx vs exact)

**Edge Cases**:
- Single neuron networks
- Very small perturbations (ε=1e-6)
- Exact point inputs (no uncertainty)

---

## Fixtures (conftest.py)

### Set Fixtures

```python
simple_star         # 3D Star set with 2 variables
simple_zono         # 3D Zonotope with 2 generators
simple_box          # 3D Box with bounds [0,1]
simple_image_star   # 4×4×1 ImageStar
simple_image_zono   # 4×4×1 ImageZono
```

### Model Fixtures

```python
simple_linear_model    # Linear(3, 2)
simple_cnn_model       # Conv→ReLU→AvgPool→Flatten→Linear
cnn_with_maxpool       # Conv→ReLU→MaxPool→Flatten→Linear
```

### Helper Assertions

```python
pytest.assert_star_valid(star)           # Validate Star structure
pytest.assert_zono_valid(zono)           # Validate Zono structure
pytest.assert_image_star_valid(img_star) # Validate ImageStar
```

---

## Writing New Tests

### Basic Test Template

```python
def test_my_feature(simple_star):
    """Test description."""
    result = my_function(simple_star)

    assert result is not None
    pytest.assert_star_valid(result)
```

### Testing Exact vs Approximate

```python
def test_exact_vs_approx():
    """Compare exact and approximate methods."""
    # Exact
    exact = reach_star_exact(model, [input_star])

    # Approximate
    approx = reach_star_approx(model, [input_star])

    # Approx should have fewer stars (no splitting)
    assert len(approx) <= len(exact)
```

### Testing Bounds

```python
def test_bounds_preservation():
    """Test that bounds are preserved correctly."""
    star.estimate_ranges()

    np.testing.assert_allclose(
        star.state_lb, expected_lb, atol=1e-5
    )
    np.testing.assert_allclose(
        star.state_ub, expected_ub, atol=1e-5
    )
```

---

## Coverage Report

Generate detailed coverage:

```bash
# HTML report
pytest --cov=nnv_py --cov-report=html

# Terminal report
pytest --cov=nnv_py --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -e .
      - run: pip install pytest pytest-cov
      - run: pytest --cov=nnv_py
```

---

## Test Configuration

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts =
    -v
    --tb=short
    --strict-markers
```

---

## Best Practices

### 1. Test Organization

- One test file per module
- Group related tests in classes
- Use descriptive test names
- Document test purpose in docstrings

### 2. Fixtures

- Use shared fixtures for common setups
- Keep fixtures simple and focused
- Document fixture behavior

### 3. Assertions

- Use helper assertions for validation
- Check multiple properties
- Use `np.testing.assert_allclose` for floats

### 4. Edge Cases

- Test boundary conditions
- Test error handling
- Test extreme values

### 5. Performance

- Mark slow tests with `@pytest.mark.slow`
- Use smaller test cases when possible
- Skip expensive checks in unit tests

---

## Debugging Failed Tests

### Verbose Output

```bash
pytest -vv tests/test_sets.py::TestStar::test_from_bounds
```

### Print Debugging

```bash
pytest -s tests/test_sets.py
```

### Drop into Debugger

```bash
pytest --pdb tests/test_sets.py
```

### Run Last Failed

```bash
pytest --lf
```

---

## Known Issues

### LP Solver Warnings

Some tests may show CVXPY solver warnings. These are generally harmless:

```
CVXPY: Solver did not solve the problem. Using backup method.
```

To suppress:

```bash
pytest --disable-warnings
```

### Slow Tests

Some integration tests may be slow (>1s). Mark them:

```python
@pytest.mark.slow
def test_large_network():
    ...
```

Run without slow tests:

```bash
pytest -m "not slow"
```

---

## Future Test Additions

Potential areas for expansion:

1. **Property-based testing** with Hypothesis
2. **Parametric tests** for layer combinations
3. **Benchmark tests** for performance regression
4. **Fuzzing tests** for robustness
5. **Parallel test execution** with pytest-xdist
6. **GPU tests** for CUDA operations
7. **Memory profiling** tests

---

## Contributing Tests

When adding new features:

1. Add unit tests for new functions
2. Add integration tests for new workflows
3. Update fixtures if needed
4. Document test purpose
5. Ensure tests pass locally
6. Check coverage remains high

---

## Resources

- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **NumPy Testing**: https://numpy.org/doc/stable/reference/routines.testing.html

---

**Test Status**: ✅ 102+ tests covering all major functionality

See [README.md](README.md) for main documentation.
