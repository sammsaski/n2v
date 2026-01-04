# N2V Test Suite

This directory contains the complete test suite for the N2V (Neural Network Verification) library.

## Test Organization

The test suite is organized into two categories:

### 1. Unit Tests (`unit/`)

**~620 tests** that verify correct implementation and edge case handling.

These tests check that:
- Code compiles and runs without errors
- Methods handle edge cases correctly (empty inputs, boundary conditions, etc.)
- API contracts are maintained
- Integration between components works as expected

**Subdirectories:**
- `sets/` - Tests for Star, Zono, Box, ImageStar, Hexatope, Octatope, ProbabilisticBox
- `layer_ops/` - Tests for layer operations (Linear, ReLU, Conv2D, MaxPool2D, AvgPool2D, Flatten)
- `core/` - Tests for dispatcher and parallel processing
- `utils/` - Tests for VNN-LIB parsing and differentiable solvers
- `probabilistic/` - Tests for probabilistic verification (conformal inference, surrogates, verify)
- `integration/` - Integration tests for complete workflows

### 2. Soundness Tests (`soundness/`)

**~200 tests** (all passing) that verify mathematical correctness and soundness properties.

These tests check that:
- Operations produce mathematically correct results
- Approximate methods over-approximate exact results (soundness)
- Bounds are computed correctly
- Ground truth matches expected values for simple cases

**Files:**
- `test_soundness_linear.py` - Linear/fully-connected layers
- `test_soundness_relu.py` - ReLU activation (exact & approximate)
- `test_soundness_conv2d.py` - 2D convolution layers
- `test_soundness_maxpool2d.py` - Max pooling (exact & approximate)
- `test_soundness_avgpool2d.py` - Average pooling
- `test_soundness_flatten.py` - Flatten operations
- `test_soundness_parallel.py` - Parallel LP solving
- `test_soundness_to_star.py` - Conversion to Star sets
- `test_soundness_differentiable.py` - Differentiable verification
- `test_soundness_probabilistic.py` - Probabilistic verification guarantees

See [soundness/README.md](soundness/README.md) for detailed methodology.

## Running Tests

### Quick Start

```bash
# Run all tests (unit + soundness, ~846 passing)
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with quiet output (summary only)
pytest tests/ -q
```

### Test Categories

```bash
# Run only unit tests (~620 tests)
pytest tests/unit/

# Run only soundness tests (~200 tests)
pytest tests/soundness/

# Run only probabilistic verification tests
pytest tests/unit/probabilistic/ tests/soundness/test_soundness_probabilistic.py -v
```

### Specific Test Files

```bash
# Run specific test file
pytest tests/unit/sets/test_star.py -v
pytest tests/soundness/test_soundness_relu.py -v

# Run specific test class
pytest tests/unit/sets/test_star.py::TestStar -v

# Run specific test method
pytest tests/unit/sets/test_star.py::TestStar::test_from_bounds -v
```

### Advanced Options

```bash
# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Generate coverage report
pytest tests/ --cov=n2v --cov-report=html
```

## Test Philosophy

**Unit tests** focus on **implementation correctness**:
- Does the code run without errors?
- Are edge cases handled properly?
- Do components integrate correctly?

**Soundness tests** focus on **mathematical correctness**:
- Are the results mathematically sound?
- Do approximate methods over-approximate exact results?
- Are the computed bounds correct?

Both types of tests are essential:
- Unit tests catch bugs and regressions
- Soundness tests ensure the verification results are trustworthy

## Writing Tests

### Using Fixtures

Unit tests use shared fixtures defined in `unit/conftest.py`:

```python
def test_my_feature(simple_star, simple_image_star):
    """Test using predefined fixtures."""
    result = my_function(simple_star)
    assert result.dim == simple_star.dim
    pytest.assert_star_valid(result)
```

**Available fixtures:**
- `simple_star` - 3D Star set for basic tests
- `simple_zono` - 3D Zonotope for basic tests
- `simple_box` - 3D Box set for basic tests
- `simple_image_star` - 4x4x1 ImageStar for CNN tests
- `simple_image_zono` - 4x4x1 ImageZono for CNN tests

**Custom assertions:**
- `pytest.assert_star_valid(star)` - Verify Star set is well-formed
- `pytest.assert_zono_valid(zono)` - Verify Zonotope is well-formed

### Test Patterns

**Unit test pattern:**
```python
class TestMyFeature:
    """Tests for my new feature."""

    def test_basic_case(self, simple_star):
        """Test basic functionality."""
        result = my_operation(simple_star)
        assert result is not None
        assert result.dim == simple_star.dim

    def test_edge_case(self):
        """Test edge case handling."""
        # Create specific input for edge case
        edge_input = Star.from_bounds(lb, ub)
        result = my_operation(edge_input)
        assert result.is_valid()

    @pytest.mark.skip(reason="Feature not implemented yet")
    def test_future_feature(self):
        """Test for future feature."""
        pass
```

**Soundness test pattern:**
```python
class TestLayerSoundness:
    """Soundness tests for layer operation."""

    def test_exact_vs_approx_soundness(self):
        """Verify approximate over-approximates exact."""
        # Generate test input
        input_star = generate_random_star(dim=10)

        # Compute both methods
        exact_result = layer_reach_exact(input_star)
        approx_result = layer_reach_approx(input_star)

        # Verify soundness
        for exact_star in exact_result:
            exact_lb, exact_ub = exact_star.get_ranges()
            approx_lb, approx_ub = approx_result[0].get_ranges()

            # Approx should contain exact
            assert np.all(approx_lb <= exact_lb + 1e-6)
            assert np.all(approx_ub >= exact_ub - 1e-6)
```

### Test Requirements

```bash
# Install test dependencies
pip install pytest pytest-cov

# Optional: parallel test execution
pip install pytest-xdist
```

## Contributing

When adding new features:

1. **Always add unit tests** to verify the implementation works
   - Test basic functionality
   - Test edge cases (empty inputs, boundary conditions, etc.)
   - Test error handling

2. **Add soundness tests** for verification operations to ensure mathematical correctness
   - Compare exact vs. approximate methods
   - Verify bounds are correct
   - Check against ground truth for simple cases

3. **Follow existing patterns**
   - Use the existing test files as templates
   - Use shared fixtures from `conftest.py`
   - Use descriptive test names and docstrings

4. **Ensure all tests pass** before submitting
   ```bash
   pytest tests/
   ```

For more details on soundness testing methodology, see [soundness/README.md](soundness/README.md).
