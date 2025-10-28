# ACAS Xu Verification Examples

This directory contains examples for verifying ACAS Xu neural network properties using NNV-Python with support for multiple set representations.

## Overview

ACAS Xu (Airborne Collision Avoidance System X for Unmanned Aircraft) is a safety-critical system that uses neural networks to provide collision avoidance advisories. The verification of these networks is a challenging benchmark problem in the neural network verification community.

## Files

- `onnx/` - ACAS Xu neural networks in ONNX format (45 networks)
- `vnnlib/` - VNN-LIB property files (10 properties)
- **`verify_acasxu.py`** - **NEW**: Generalized verification script supporting all set types (Box, Zono, Star, Hexatope, Octatope)
- `verify_acasxu_star.py` - Original Star-specific verification script (backward compatibility)
- `verify_acasxu_comp.py` - Comparison script for benchmarking

## Usage

### Quick Start - Generalized Script

The new `verify_acasxu.py` supports all set types and methods:

```bash
# Default: Star sets with exact method
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib

# Fast verification with Box sets
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib --set box --method approx

# Precise verification with Hexatope and differentiable solver
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib --set hexatope --method exact-differentiable

# Parallel processing with Star sets
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib --set star --method exact --parallel --workers 4
```

### Supported Set Types and Methods

| Set Type | Supported Methods | Speed | Precision | Notes |
|----------|------------------|-------|-----------|-------|
| **box** | `approx` | ⚡⚡⚡ Fastest | ⭐ Basic | Interval arithmetic |
| **zono** | `approx` | ⚡⚡ Fast | ⭐⭐ Good | Zonotope representation |
| **star** | `exact`, `approx` | ⚡ Moderate | ⭐⭐⭐ High | Most flexible, parallel support |
| **hexatope** | `exact`, `exact-differentiable`, `approx` | 🐌 Slow | ⭐⭐⭐⭐ Very High | DCS constraints |
| **octatope** | `exact`, `exact-differentiable`, `approx` | 🐌 Slow | ⭐⭐⭐⭐ Very High | UTVPI constraints |

### Command-Line Arguments

#### Required Arguments
- **`network`**: Path to ONNX network file
- **`property`**: Path to VNN-LIB property file

#### Optional Arguments
- **`--set {box,zono,star,hexatope,octatope}`**: Set representation (default: `star`)
- **`--method {exact,exact-differentiable,approx}`**: Reachability method (default: `exact`)
  - `exact`: Exact reachability with splitting (uses CVXPY)
  - `exact-differentiable`: Exact with differentiable LP solver (Hexatope/Octatope only)
  - `approx`: Over-approximate reachability (faster, conservative)
- **`--timeout TIMEOUT`**: Timeout in seconds (default: 300.0)
- **`--parallel`**: Enable parallel processing (Star only)
- **`--workers N`**: Number of parallel workers (default: auto-detect)

### Examples by Set Type

```bash
# Box sets - Fastest, least precise
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set box --method approx

# Zonotope sets - Fast with good precision
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set zono --method approx

# Star sets with exact method (default)
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set star --method exact

# Star sets with approximate method (faster)
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set star --method approx

# Hexatope with exact method and CVXPY
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set hexatope --method exact

# Hexatope with differentiable LP solver (GPU-friendly)
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set hexatope --method exact-differentiable

# Octatope with approximate method
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set octatope --method approx

# Star sets with parallel processing
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set star --method exact --parallel --workers 4
```

### Comparison of Different Approaches

```bash
# Compare different set types with approximate methods
for set_type in box zono star; do
  echo "Testing with $set_type"
  python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
    --set $set_type --method approx
done

# Compare exact methods
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set star --method exact

python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set hexatope --method exact

python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set hexatope --method exact-differentiable
```

### Legacy Script

The original Star-specific script is still available:

```bash
python verify_acasxu_star.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib --method exact --parallel
```

### Example Output

```
================================================================================
ACAS Xu Verification
================================================================================
Network: ACASXU_run2a_1_4_batch_2000.onnx
Property: prop_3.vnnlib
================================================================================

================================================================================
Verifying: ACASXU_run2a_1_4_batch_2000.onnx with prop_3.vnnlib
================================================================================

1. Loading network...

2. Loading property...
   ✓ Property loaded: /path/to/vnnlib/prop_3.vnnlib
   Input dimension: 5
   Input bounds:
     X_0: [-0.303531, -0.298553]
     X_1: [-0.009549, 0.009549]
     X_2: [0.493380, 0.500000]
     X_3: [0.300000, 0.500000]
     X_4: [0.300000, 0.500000]
   Output properties: 1
   Property type: Single halfspace

3. Creating input set...
   ✓ Input Star created:
     Dimension: 5
     Number of variables: 5

4. Computing reachable set (method: exact)...
   ✓ Reachability completed in 111.25 seconds
   Number of output stars: 4346

   Output bounds:
     Y_0: [-0.002428, 0.008114]
     Y_1: [-0.003664, 0.005348]
     Y_2: [-0.005898, 0.008840]
     Y_3: [-0.014091, -0.005934]
     Y_4: [-0.014701, -0.000675]

5. Verifying specification...
   ✓ Verification completed in 7.19 seconds

================================================================================
VERIFICATION RESULT
================================================================================
  Result: UNSAT
  Status: ✅ Property holds (no intersection with unsafe region)

Timing:
  Reachability: 111.25s
  Verification: 7.19s
  Total: 118.50s
================================================================================


================================================================================
SUMMARY
================================================================================
Network: ACASXU_run2a_1_4_batch_2000.onnx
Property: prop_3.vnnlib
Result: UNSAT
Time: 118.50s
Method: exact
Output stars: 4346
================================================================================
```

## Verification Results

The verification returns one of three results:

- **UNSAT**: Property holds (verified safe)
- **SAT**: Property violated (counterexample exists)
- **UNKNOWN**: Cannot determine (typically with approximate methods)

## Implementation Status

### ✅ Completed Features

1. **ONNX Model Loading**
   - Full support for ACAS Xu ONNX models via `onnx2torch`
   - Handles GraphModule structure with custom operation types
   - Supports `OnnxMatMul`, `OnnxBinaryMathOperation`, and standard PyTorch layers

2. **VNN-LIB Parser** (`load_vnnlib.py`)
   - Parses VNN-LIB property files
   - Extracts input bounds (lb, ub)
   - Converts output constraints to HalfSpace objects
   - Handles comments, multiline statements, various formats

3. **HalfSpace Class** (`halfspace.py`)
   - Represents linear constraints: G @ x <= g
   - Point containment checking
   - Compatible with Star set operations

4. **Specification Verification** (`verify_specification.py`)
   - Verifies properties by checking intersection with unsafe regions
   - Handles single and multiple halfspaces (AND/OR logic)
   - Works with multiple output stars
   - **Correctly handles infeasible intersections** (checks `is_empty_set()`)
   - Returns: 1 (UNSAT/verified), 2 (UNKNOWN), 0 (SAT/violated)

5. **Reachability Analysis**
   - Exact-star reachability for precise verification
   - Approximate-star reachability for faster over-approximation
   - Support for GraphModule (FX graph) from ONNX models
   - Optimized handling of translation operations (Add/Sub)

### ⚠️ Known Limitations

1. **Performance**
   - Exact reachability is ~2x slower than MATLAB implementation
   - Multi-core processing not yet implemented (TODO in code)
   - MATLAB uses parallel processing via `numCores` parameter

2. **Optimizations Needed**
   - Fuse MatMul+Add operations into single Linear layers
   - Implement parallel ReLU processing
   - Consider JIT compilation for bottleneck operations

## Property Specifications

ACAS Xu properties test various safety conditions:

- **Property 1-2**: If the intruder is distant and slower, COC advisory should not be issued
- **Properties 3-4**: Test advisories for head-on scenarios (use approx first in MATLAB)
- **Properties 5-10**: Various other safety-critical scenarios

### Property Format

VNN-LIB properties define:
- **Input constraints**: Box constraints on 5D input space (scaled state information)
- **Output constraints**: Linear constraints on 5D output space (advisory scores)
- **Unsafe region**: Region where property is violated

Example (Property 3):
```smt
; Input bounds
(assert (<= X_0 -0.298552812))
(assert (>= X_0 -0.303531156))
...

; Unsafe if COC is minimal (Y_0 <= Y_i for all i)
(assert (<= Y_0 Y_1))
(assert (<= Y_0 Y_2))
(assert (<= Y_0 Y_3))
(assert (<= Y_0 Y_4))
```

## Implementation Notes

### Python vs MATLAB Comparison

**Performance** (for ACAS Xu 1_4 + prop_3 with exact):
- **MATLAB**: ~60s (with multi-core)
- **Python**: ~118s (single-core)

**Method Selection** (from MATLAB `run_vnncomp_instance.m`):
- For properties 3 and 4: Try `approx` (Star) first, fall back to `exact` (Star) if needed
- For other properties: Use `exact` (Star) directly

**Verification Logic**:
Both implementations check intersection with unsafe regions:
- No intersection → UNSAT (verified)
- Infeasible intersection → UNSAT (verified)  ← **Critical fix in Python**
- Feasible intersection → UNKNOWN/SAT

### Recent Bug Fixes

1. **Infeasible Intersection Handling** (Fixed)
   - Issue: `verify_specification` didn't check `is_empty_set()` for intersection Stars
   - All 4346 output stars had infeasible intersections but returned UNKNOWN instead of UNSAT
   - Fix: Added `is_empty_set()` check in verification logic
   - Test: Added `test_infeasible_intersection` to prevent regression

## Testing

Unit tests are provided in:
- `tests/test_load_vnnlib.py` - VNN-LIB parser tests
- `tests/test_verify_specification.py` - Specification verification tests (including infeasible intersection edge case)
- `tests/test_sets.py` - HalfSpace tests

Run tests:
```bash
pytest tests/test_verify_specification.py -v
pytest tests/test_load_vnnlib.py -v
```

## References

1. Katz, G., et al. "Reluplex: An efficient SMT solver for verifying deep neural networks." *CAV 2017*.
2. VNN-COMP: International Verification of Neural Networks Competition
3. ACAS Xu: Airborne Collision Avoidance System for unmanned aircraft
4. ONNX: Open Neural Network Exchange format
