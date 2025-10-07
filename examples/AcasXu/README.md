# ACAS Xu Verification Example

This directory contains an example of verifying ACAS Xu neural network properties using NNV-Python.

## Overview

ACAS Xu (Airborne Collision Avoidance System X for Unmanned Aircraft) is a safety-critical system that uses neural networks to provide collision avoidance advisories. The verification of these networks is a challenging benchmark problem in the neural network verification community.

## Files

- `onnx/` - ACAS Xu neural networks in ONNX format (45 networks)
- `vnnlib/` - VNN-LIB property files (10 properties)
- `verify_acasxu.py` - Verification script demonstrating the workflow

## Implementation Status

### ✅ Completed

1. **VNN-LIB Parser** (`load_vnnlib.py`)
   - Parses VNN-LIB property files
   - Extracts input bounds (lb, ub)
   - Converts output constraints to HalfSpace objects
   - Handles comments, multiline statements, various formats

2. **HalfSpace Class** (`halfspace.py`)
   - Represents linear constraints: G @ x <= g
   - Point containment checking
   - Compatible with Star set operations

3. **Specification Verification** (`verify_specification.py`)
   - Verifies properties by checking intersection with unsafe regions
   - Handles single and multiple halfspaces (AND/OR logic)
   - Works with multiple output stars
   - Returns results: 1 (verified), 2 (unknown), 0 (violated)

4. **Integration**
   - Full verification workflow from VNN-LIB to result
   - Star-based reachability analysis
   - Property verification

### ⚠️ Known Limitations

1. **ONNX Model Loading**
   - ACAS Xu ONNX models have weights embedded as model inputs
   - `onnx2torch` doesn't support the `Flatten` operation with this format
   - **Workaround needed**: Custom ONNX loader or ONNX Runtime wrapper

2. **Current Demo**
   - Uses a dummy network with the same architecture (6 hidden layers, 50 neurons each)
   - Demonstrates the complete verification workflow
   - Property loading and verification works correctly

## Usage

```bash
cd /Users/samuel/milos/rgit/nnv2/nnv_py/examples/AcasXu
python verify_acasxu.py
```

### Example Output

```
================================================================================
ACAS Xu Verification
================================================================================
Network: ACASXU_run2a_1_1_batch_2000.onnx
Property: prop_1.vnnlib
================================================================================

1. Loading network...
   ⚠️  Using dummy network (same architecture as ACAS Xu)

2. Loading property...
   ✓ Property loaded: prop_1.vnnlib
   Input dimension: 5
   Input bounds:
     X_0: [0.600000, 0.679858]
     X_1: [-0.500000, 0.500000]
     ...

3. Creating input set...
   ✓ Input Star created

4. Computing reachable set (method: approx)...
   ✓ Reachability completed in 0.05 seconds

5. Verifying specification...
   ✓ Verification completed

Result: UNKNOWN
```

## Property Specifications

ACAS Xu properties test various safety conditions:

- **Property 1**: If the intruder is distant and slower, COC advisory should not be issued
- **Property 2**: Similar to Property 1 but different input ranges
- **Properties 3-4**: Test advisories for head-on scenarios
- **Properties 5-10**: Various other safety-critical scenarios

### Property Format

VNN-LIB properties define:
- **Input constraints**: Box constraints on 5D input space
- **Output constraints**: Linear constraints on 5D output space (advisory scores)
- **Unsafe region**: Region where property is violated

Example (Property 1):
```smt
; Input: 5D (scaled state information)
(assert (>= X_0 0.6))
(assert (<= X_0 0.679857769))
...

; Unsafe if COC advisory >= 1500
(assert (>= Y_0 3.991125645861615))
```

## Next Steps

To complete ACAS Xu verification:

1. **Implement Custom ONNX Loader**
   - Extract weights from ONNX model inputs
   - Build PyTorch Sequential model manually
   - Handle all ACAS Xu-specific ONNX operations

2. **Optimize Verification**
   - Implement exact-star method
   - Add support for parallel verification
   - Implement early termination on violation

3. **Benchmark**
   - Run all 45 networks × 10 properties = 450 instances
   - Compare with other verification tools
   - Generate verification results table

## Testing

Unit tests are provided in:
- `tests/test_load_vnnlib.py` - VNN-LIB parser tests
- `tests/test_verify_specification.py` - Specification verification tests
- `tests/test_sets.py` - HalfSpace tests

Run tests:
```bash
cd /Users/samuel/milos/rgit/nnv2/nnv_py
pytest tests/test_load_vnnlib.py -v
pytest tests/test_verify_specification.py -v
pytest tests/test_sets.py::TestHalfSpace -v
```

## References

1. Katz, G., et al. "Reluplex: An efficient SMT solver for verifying deep neural networks." *CAV 2017*.
2. VNN-COMP: International Verification of Neural Networks Competition
3. ACAS Xu: Airborne Collision Avoidance System for unmanned aircraft

## Implementation Notes

### Python vs MATLAB

The Python implementation (`nnv_py`) mirrors the MATLAB implementation (`nnv`) with key differences:

**MATLAB** (`verify_specification.m`):
- Uses `Star.intersectHalfSpace()` method
- Returns 0/1/2 for violated/verified/unknown
- Handles ImageStar/Star conversion

**Python** (`verify_specification.py`):
- Uses `Star.intersect_half_space()` method
- Same return values (0/1/2)
- Type checking with isinstance
- More explicit type hints and documentation

Both implementations follow the same logic for single and multiple halfspace verification.
