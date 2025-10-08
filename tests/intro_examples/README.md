# Introductory NNV Examples

This directory contains introductory examples from the NNV documentation, implemented in both **MATLAB** and **Python** for validation.

The goal is to ensure that N2V (Python) produces identical results to NNV (MATLAB) for basic operations, establishing confidence in the translation.

---

## Overview

These examples follow a validation pattern:

1. **MATLAB Script** - Runs in MATLAB NNV, saves all variables and results to `.mat` file
2. **Python Script** - Runs equivalent operations in N2V, compares against saved MATLAB results
3. **Pytest Tests** - Automated validation suite that checks Python implementation

---

## Example 9.D: Parallelogram Zonotope

**Reference**: NNV Introduction, Example 9.D

### Description

A simple 2D parallelogram zonotope demonstrating compact notation:

```
(2 + ε₁, 3 + ε₁ + ε₂)
```

**Compact Notation**: `(⟨2, 1, 0⟩, ⟨3, 1, 1⟩)`

Where:
- Center: `c = [2; 3]`
- Generator 1: `v₁ = [1; 1]`
- Generator 2: `v₂ = [0; 1]`

The zonotope is defined as: `Z = c + α₁*v₁ + α₂*v₂`, where `-1 ≤ αᵢ ≤ 1`

### Key Property

The **upper bound in the vertical dimension** is:
```
3 + ε₁ + ε₂ = 3 + 1 + 1 = 5  (when ε₁, ε₂ = 1)
```

This serves as a simple verification that bounds are computed correctly.

---

## Files

```
intro_examples/
├── README.md                           # This file
├── example_9d_parallelogram.m          # MATLAB script
├── example_9d_parallelogram.py         # Python script
└── saved_results/
    ├── example_9d_results.mat          # MATLAB output (ground truth)
    └── example_9d_python_results.mat   # Python output
```

### Test File

```
tests/
└── test_intro_examples.py              # Pytest validation suite
```

---

## Usage

### Step 1: Run MATLAB Script

**Prerequisite**: MATLAB with NNV installed and on the path

```matlab
% In MATLAB, from the n2v/tests/intro_examples directory
example_9d_parallelogram
```

**Output**:
- Console output showing all operations
- `saved_results/example_9d_results.mat` with all variables

### Step 2: Run Python Script

```bash
cd n2v/tests/intro_examples
python3 example_9d_parallelogram.py
```

**Output**:
- Console output showing all operations
- Comparison with MATLAB results (if available)
- `saved_results/example_9d_python_results.mat`

### Step 3: Run Automated Tests

```bash
cd n2v
pytest tests/test_intro_examples.py -v
```

**Test Coverage**:
- ✅ Zonotope creation
- ✅ Bounds computation
- ✅ Vertex enumeration
- ✅ Affine transformation
- ✅ Minkowski sum
- ✅ Point containment
- ✅ Compact notation validation

---

## What Gets Tested

### Zonotope Operations

1. **Creation**
   - Center vector `c`
   - Generator matrix `V`
   - Dimension validation

2. **Bounds Computation**
   - Lower bounds `lb`
   - Upper bounds `ub`
   - Verification: upper y-bound = 5

3. **Vertices**
   - All 4 corner points of parallelogram
   - Order-independent comparison

4. **Affine Map**
   - Transformation: `W*Z + b`
   - Test: `W = [2, 0; 0, 1]`, `b = [1; 0]`

5. **Minkowski Sum**
   - Sum with small box: `Z ⊕ Z₂`
   - Generator concatenation

6. **Point Containment**
   - Center (should be inside)
   - Corners (should be inside)
   - Points outside zonotope

---

## Validation Methodology

### Comparison Tolerance

All numerical comparisons use **absolute tolerance = 1e-10**:

```python
np.testing.assert_allclose(python_val, matlab_val, atol=1e-10)
```

### Why This Tolerance?

- Both implementations use `float64` precision
- Linear operations should be exact within numerical precision
- Small differences may arise from LP solver tolerances (not tested here)

### What We Compare

For each operation, we compare:

1. **Intermediate results** (centers, generators)
2. **Final outputs** (bounds, vertices)
3. **Derived quantities** (number of generators, dimensions)

---

## Expected Results

### Console Output (MATLAB)

```
========================================
Example 9.D: Parallelogram Zonotope
========================================

1. Creating parallelogram zonotope...
   Center c = [2; 3]
   Generators V = [1, 0; 1, 1]
   Dimension: 2
   Number of generators: 2
   ✓ Zonotope created

2. Computing bounds...
   Bounds:
     Dimension 1 (x): [1.0000000000, 3.0000000000]
     Dimension 2 (y): [1.0000000000, 5.0000000000]

   Verification: Upper bound in vertical dimension = 5.0000000000
   Expected: 3 + 1 + 1 = 5
   Match: ✓ YES

[... additional operations ...]
```

### Console Output (Python)

```
============================================================
Example 9.D: Parallelogram Zonotope (Python)
============================================================

1. Creating parallelogram zonotope...
   Center c = [2. 3.]
   Generators V =
[[1. 0.]
 [1. 1.]]
   Dimension: 2
   Number of generators: 2
   ✓ Zonotope created

2. Computing bounds...
   Bounds:
     Dimension 1 (x): [1.0000000000, 3.0000000000]
     Dimension 2 (y): [1.0000000000, 5.0000000000]

   Verification: Upper bound in vertical dimension = 5.0000000000
   Expected: 3 + 1 + 1 = 5
   Match: ✓ YES

[... comparison with MATLAB ...]

   Comparing Python vs MATLAB results:
   --------------------------------------------------------
   ✓ Center c: Match (max diff = 0.00e+00)
   ✓ Generators V: Match (max diff = 0.00e+00)
   ✓ Lower bounds: Match (max diff = 0.00e+00)
   ✓ Upper bounds: Match (max diff = 0.00e+00)
   [...]
   --------------------------------------------------------

   ✅ All comparisons passed! Python matches MATLAB.
```

### Pytest Output

```bash
$ pytest tests/test_intro_examples.py -v

tests/test_intro_examples.py::TestExample9DParallelogram::test_zonotope_creation PASSED
tests/test_intro_examples.py::TestExample9DParallelogram::test_bounds_computation PASSED
tests/test_intro_examples.py::TestExample9DParallelogram::test_vertices_computation PASSED
tests/test_intro_examples.py::TestExample9DParallelogram::test_affine_transformation PASSED
tests/test_intro_examples.py::TestExample9DParallelogram::test_minkowski_sum PASSED
tests/test_intro_examples.py::TestExample9DParallelogram::test_point_containment PASSED
tests/test_intro_examples.py::TestExample9DParallelogram::test_compact_notation PASSED
tests/test_intro_examples.py::TestExample9DParallelogram::test_parallelogram_corners PASSED
tests/test_intro_examples.py::TestExample9DWithoutMATLAB::test_basic_zonotope_creation PASSED
tests/test_intro_examples.py::TestExample9DWithoutMATLAB::test_upper_bound_verification PASSED
tests/test_intro_examples.py::TestExample9DWithoutMATLAB::test_bounds_formula PASSED
tests/test_intro_examples.py::TestExample9DWithoutMATLAB::test_center_is_contained PASSED
tests/test_intro_examples.py::TestExample9DWithoutMATLAB::test_corner_points PASSED
tests/test_intro_examples.py::TestExample9DWithoutMATLAB::test_outside_points PASSED

================================ 14 passed in 0.23s ================================
```

---

## Troubleshooting

### Issue: MATLAB results file not found

**Error**:
```
⚠️  MATLAB results not found at: saved_results/example_9d_results.mat
Please run the MATLAB script first: example_9d_parallelogram.m
```

**Solution**: Run the MATLAB script first to generate ground truth data.

### Issue: NNV not found in MATLAB

**Error**: `Undefined function or variable 'Zono'`

**Solution**: Add NNV to MATLAB path:
```matlab
addpath(genpath('/path/to/nnv/code/nnv'));
```

Or run `startup_nnv.m` from the NNV installation.

### Issue: Python imports fail

**Error**: `ModuleNotFoundError: No module named 'n2v'`

**Solution**: Install N2V in development mode:
```bash
cd n2v
pip install -e .
```

### Issue: Tests are skipped

**Message**: `SKIPPED [1] tests/test_intro_examples.py:XX: MATLAB results not found`

**Solution**: This is expected if you haven't run the MATLAB script yet. The Python-only tests (class `TestExample9DWithoutMATLAB`) will still run.

---

## Understanding the Parallelogram

### Visual Representation

```
        (1, 3) -------- (3, 5)
          /              /
         /              /
        /              /
    (1, 1) -------- (3, 3)
```

### Vertices

The 4 corners correspond to extreme values of `ε₁` and `ε₂`:

| ε₁  | ε₂  | Point     | Computation          |
|-----|-----|-----------|----------------------|
| -1  | -1  | (1, 1)    | (2-1+0, 3-1-1)       |
| +1  | -1  | (3, 3)    | (2+1+0, 3+1-1)       |
| +1  | +1  | (3, 5)    | (2+1+0, 3+1+1)       |
| -1  | +1  | (1, 3)    | (2-1+0, 3-1+1)       |

### Bounds

- **x-dimension**: `2 + ε₁·1 + ε₂·0` → `[2-1, 2+1]` = `[1, 3]`
- **y-dimension**: `3 + ε₁·1 + ε₂·1` → `[3-1-1, 3+1+1]` = `[1, 5]`

---

## Benefits of This Approach

1. **Ground Truth**: MATLAB results provide reference implementation
2. **Automated**: Pytest runs automatically in CI/CD
3. **Comprehensive**: Tests both exact values and derived properties
4. **Reproducible**: All inputs and expected outputs are saved
5. **Documentation**: Examples serve as usage tutorials

---

## Adding New Examples

To add a new introductory example:

1. **Create MATLAB script**: `example_XX_name.m`
2. **Create Python script**: `example_XX_name.py`
3. **Add tests**: Update `test_intro_examples.py`
4. **Document**: Update this README

Follow the same pattern:
- MATLAB saves to `saved_results/example_XX_results.mat`
- Python loads and compares
- Pytest validates

---

## References

- **NNV Documentation**: [NNV GitHub](https://github.com/verivital/nnv)
- **N2V Implementation**: [../sets/zono.py](../sets/zono.py)
- **Comparison Examples**: [../../examples/CompareMATLAB/](../../examples/CompareMATLAB/)

---

## Status

✅ **Example 9.D: Parallelogram** - Complete and validated

---

**Last Updated**: 2025-10-08
