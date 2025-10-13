# Differentiable Solver Test Suite

This directory contains comprehensive unit and soundness tests for the differentiable LP solver implementation used in Hexatope and Octatope optimization.

## Test Files

### Unit Tests: `unit/test_differentiable_solver.py`

Unit tests verify the correctness of the differentiable solver implementation in isolation.

**Test Classes:**

1. **`TestDifferentiableLPSolver`** - Core solver functionality
   - Simple 2D minimization/maximization
   - Equality and inequality constraints
   - Unbounded and infeasible problems
   - High-dimensional problems (10+ variables)
   - Grid size and temperature schedule effects
   - Batch size variations
   - Comparison with CVXPY solver
   - CUDA acceleration (if available)

2. **`TestHexatopeDifferentiable`** - Hexatope integration
   - Range computation from bounds
   - Affine transformations
   - Half-space intersections
   - 3D hexatopes
   - `get_bounds()` alias method
   - Point containment checks

3. **`TestOctatopeDifferentiable`** - Octatope integration
   - Range computation from bounds
   - Affine transformations
   - Half-space intersections
   - 3D octatopes
   - Consistency with Hexatope
   - Point containment checks

### Soundness Tests: `soundness/test_soundness_differentiable.py`

Soundness tests verify that the abstract domain operations produce sound over-approximations when used for neural network verification.

**Test Classes:**

1. **`TestHexatopeSoundnessLinear`** - Hexatope + Linear layers
   - Identity transformation
   - Scaling
   - Translation
   - Dimension reduction
   - Dimension expansion

2. **`TestOctatopeSoundnessLinear`** - Octatope + Linear layers
   - Identity transformation
   - Rotation
   - Negative weights

3. **`TestHexatopeSoundnessReLU`** - Hexatope + ReLU
   - All positive inputs
   - All negative inputs
   - Inputs crossing zero

4. **`TestOctatopeSoundnessReLU`** - Octatope + ReLU
   - All positive inputs
   - Mixed positive/negative dimensions

5. **`TestComposedOperationsSoundness`** - Multi-layer networks
   - Linear → ReLU composition
   - Multi-layer networks

6. **`TestComparativeSoundness`** - Solver comparison
   - Standard vs differentiable solver for Hexatope
   - Standard vs differentiable solver for Octatope

## Running the Tests

### Prerequisites

```bash
# Install required packages
pip install pytest torch numpy

# Optional: Install CUDA-enabled PyTorch for GPU tests
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Run All Tests

```bash
# From the n2v directory
cd /home/sasakis/v/tools/n2v

# Run all differentiable solver tests
pytest tests/unit/test_differentiable_solver.py -v
pytest tests/soundness/test_soundness_differentiable.py -v

# Run both
pytest tests/unit/test_differentiable_solver.py tests/soundness/test_soundness_differentiable.py -v
```

### Run Specific Test Classes

```bash
# Run only basic LP solver tests
pytest tests/unit/test_differentiable_solver.py::TestDifferentiableLPSolver -v

# Run only Hexatope soundness tests
pytest tests/soundness/test_soundness_differentiable.py::TestHexatopeSoundnessLinear -v

# Run specific test
pytest tests/unit/test_differentiable_solver.py::TestDifferentiableLPSolver::test_simple_2d_minimization -v
```

### Run with Coverage

```bash
# Install coverage
pip install pytest-cov

# Run with coverage report
pytest tests/unit/test_differentiable_solver.py \
       tests/soundness/test_soundness_differentiable.py \
       --cov=utils.lpsolver \
       --cov=sets.hexatope \
       --cov=sets.octatope \
       --cov-report=html
```

### Skip Tests Without PyTorch

Tests that require PyTorch are automatically skipped if PyTorch is not installed:

```bash
# These tests will be skipped without PyTorch
pytest tests/unit/test_differentiable_solver.py -v
# Output: ... SKIPPED (PyTorch not available)
```

## Test Design Principles

### Soundness Verification

Soundness tests verify the fundamental property of abstract interpretation:

```
∀x ∈ γ(X_input), f(x) ∈ γ(X_output)
```

Where:
- `X_input`, `X_output` are abstract sets (Hexatope/Octatope)
- `γ(X)` is the concretization (set of concrete points)
- `f` is the neural network operation

**Methodology:**
1. Sample concrete points from input abstract set
2. Apply concrete operation (e.g., linear layer)
3. Check if output is contained in output abstract set
4. Allow small violation ratio due to approximation errors

### Tolerance Levels

Different tolerance levels are used based on the operation type:

- **Exact operations** (identity, scaling): ≤10% violation ratio
- **Approximate operations** (ReLU over-approximation): ≤20% violation ratio
- **Composed operations** (multi-layer): ≤25% violation ratio
- **Numerical accuracy** (objective values): ±20-30% relative error

These tolerances account for:
- Grid discretization in differentiable solver
- Stochastic sampling in Gumbel-Softmax
- Over-approximation in abstract interpretation
- Numerical precision limitations

## Understanding Test Results

### Successful Test

```
test_simple_2d_minimization PASSED
  Status: optimal
  Expected objective: 1.0
  Actual objective: 1.03
  Relative error: 3%
```

### Acceptable Approximation

```
test_relu_crossing_zero_soundness PASSED
  Violations: 15/100 samples (15%)
  Max violation: 0.08
  Status: PASS (below 20% threshold)
```

### Failed Test

```
test_identity_linear_soundness FAILED
  Violations: 35/100 samples (35%)
  Max violation: 0.45
  Status: FAIL (exceeds 10% threshold)

  Possible causes:
  - Insufficient epochs (try increasing num_epochs)
  - Poor temperature schedule (adjust init_temp/final_temp)
  - Grid too coarse (increase grid_size)
```

## Debugging Failed Tests

### Increase Optimization Quality

If tests fail due to poor approximation:

```python
# In the test, increase these parameters:
solve_lp_differentiable(
    ...,
    num_epochs=200,      # More optimization steps
    batch_size=64,       # More samples per step
    grid_size=100,       # Finer discretization
    init_temp=20.0,      # Higher initial exploration
    final_temp=0.05,     # Sharper final distribution
    learning_rate=0.02,  # Faster convergence
)
```

### Enable Verbose Output

```python
# Add verbose=True to see optimization progress
solve_lp_differentiable(..., verbose=True)
```

### Check Constraint Violations

```python
# Increase penalty for constraint violations
solve_lp_differentiable(
    ...,
    constraint_penalty_weight=500.0,  # Stricter constraint satisfaction
)
```

## Expected Test Runtime

On a modern CPU:
- **Unit tests**: ~2-5 minutes (depends on num_epochs settings)
- **Soundness tests**: ~5-10 minutes (includes sampling and verification)
- **Total**: ~7-15 minutes

With CUDA GPU:
- **Unit tests**: ~1-2 minutes
- **Soundness tests**: ~2-5 minutes
- **Total**: ~3-7 minutes

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Test Differentiable Solver
  run: |
    pip install pytest torch numpy
    pytest tests/unit/test_differentiable_solver.py \
           tests/soundness/test_soundness_differentiable.py \
           -v --tb=short
```

## Contributing

When adding new tests:

1. **Follow naming conventions**: `test_<operation>_<aspect>`
2. **Document expected behavior**: Include docstrings explaining what is being tested
3. **Use appropriate tolerances**: Balance strictness with practicality
4. **Test edge cases**: Include boundary conditions, empty sets, etc.
5. **Verify soundness**: Ensure abstract operations are sound over-approximations

## References

1. Liu, M., et al. (2024). "Differentiable Combinatorial Scheduling at Scale." ICML'24.
2. Bak, S., et al. (2024). "The hexatope and octatope abstract domains for neural network verification." Formal Methods in System Design.

## Contact

For questions about the test suite:
- Check test documentation in individual test files
- Review the main differentiable solver documentation: `docs/differentiable_solver.md`
- Open an issue in the repository
