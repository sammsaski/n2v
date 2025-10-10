# ACAS Xu Verification Example

This directory contains an example of verifying ACAS Xu neural network properties using NNV-Python.

## Overview

ACAS Xu (Airborne Collision Avoidance System X for Unmanned Aircraft) is a safety-critical system that uses neural networks to provide collision avoidance advisories. The verification of these networks is a challenging benchmark problem in the neural network verification community.

## Files

- `onnx/` - ACAS Xu neural networks in ONNX format (45 networks)
- `vnnlib/` - VNN-LIB property files (10 properties)
- `verify_acasxu.py` - Main verification script with command-line interface

## Usage

### Basic Usage

```bash
python verify_acasxu.py <network> <property> [options]
```

### Command-Line Arguments

- **`network`** (required): Path to ONNX network file (relative to script directory or absolute)
- **`property`** (required): Path to VNN-LIB property file (relative to script directory or absolute)
- **`--method {exact,approx}`** (optional): Reachability method (default: `exact`)
  - `exact`: Exact reachability analysis (slower, more precise)
  - `approx`: Approximate reachability analysis (faster, over-approximation)
- **`--timeout TIMEOUT`** (optional): Timeout in seconds (default: 300.0)

### Examples

```bash
# Verify with exact reachability (default)
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib

# Verify with approximate reachability (faster)
python verify_acasxu.py onnx/ACASXU_run2a_1_5_batch_2000.onnx vnnlib/prop_3.vnnlib --method approx

# With custom timeout
python verify_acasxu.py onnx/ACASXU_run2a_1_1_batch_2000.onnx vnnlib/prop_1.vnnlib --timeout 600

# Using absolute paths
python verify_acasxu.py /path/to/network.onnx /path/to/property.vnnlib

# Show help
python verify_acasxu.py --help
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
- **Properties 3-4**: Test advisories for head-on scenarios (use approx-star first in MATLAB)
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

**Performance** (for ACAS Xu 1_4 + prop_3 with exact-star):
- **MATLAB**: ~60s (with multi-core)
- **Python**: ~118s (single-core)

**Method Selection** (from MATLAB `run_vnncomp_instance.m`):
- For properties 3 and 4: Try `approx-star` first, fall back to `exact-star` if needed
- For other properties: Use `exact-star` directly

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
