# Octatope Reachability Testing and Profiling

This directory contains test scripts to verify and profile Octatope reachability analysis performance.

## Problem

Octatope exact reachability is significantly slower than Star exact reachability on ACAS Xu benchmarks (>1 hour vs ~5 minutes), suggesting a performance bottleneck that needs investigation.

## Test Scripts

### `test_octatope_reach.py`

Comprehensive benchmark suite that compares Octatope vs Star reachability on progressively larger networks.

**Usage:**
```bash
python test_octatope_reach.py
```

**Tests:**
1. **Tiny Network** (2-3-1): Quick sanity check, should complete in <1 second
2. **Small Network** (2-5-1): Should complete in a few seconds
3. **Medium Network** (5-10-5-1): Similar to small ACAS Xu, may take longer

**Output:**
- Timing comparison between Star and Octatope
- Success/failure status for each test
- Performance analysis and recommendations

**Example Output:**
```
TINY NETWORK:
  Star Exact:        0.234s  (8 sets)
  Octatope Exact:    2.156s  (8 sets)  [0.11x vs Star]
  Octatope Approx:   0.145s  (1 sets)  [1.61x vs Star]

⚠️  Octatope exact is 9.2x SLOWER than Star!
    This suggests a performance bottleneck in Octatope exact reachability.
```

### `profile_octatope.py`

Detailed profiling using Python's cProfile to identify specific bottlenecks.

**Usage:**
```bash
# Compare estimate_ranges() performance only
python profile_octatope.py --compare-ranges

# Run full cProfile analysis
python profile_octatope.py --full-profile

# Run both (default)
python profile_octatope.py
```

**Features:**
- Compares `estimate_ranges()` performance between Star and Octatope
- Shows top 30 time-consuming functions
- Filters for Octatope-specific functions
- Identifies LP solver bottlenecks

**What to Look For:**
1. **`estimate_ranges()`** - Called frequently during ReLU splitting, must be fast
2. **`_optimize_utvpi_lp()`** - LP solving for UTVPI constraints
3. **`relu_octatope_exact()`** - Main ReLU handling code
4. **CVXPY overhead** - May be slow for small LPs

## Common Performance Issues

### 1. Slow `estimate_ranges()`

If Octatope's `estimate_ranges()` is much slower than Star's, this is a major bottleneck since it's called for every neuron during exact reachability.

**Potential causes:**
- UTVPI to DCS conversion overhead
- Inefficient LP solving for range estimation
- Too many constraints in the converted problem

**Solution:**
- Cache range estimates when possible
- Optimize UTVPI constraint handling
- Consider using differentiable solver for batch optimization

### 2. Excessive ReLU Splitting

If Octatope creates many more output sets than Star during exact reachability, the problem may be in the splitting logic.

**Check:**
- Number of output sets for Octatope vs Star
- Are Octatope bounds tighter (causing more splits)?
- Is splitting happening unnecessarily?

### 3. LP Solver Overhead

CVXPY may have significant overhead for small LP problems that are solved repeatedly.

**Solution:**
- Try `method='exact-differentiable'` which uses a differentiable LP solver
- Consider caching LP solver instances
- Use a faster LP solver backend if available

## Debugging Steps

### Step 1: Run Basic Tests
```bash
python test_octatope_reach.py
```

Look at the timing ratios. If Octatope is >10x slower on tiny networks, there's a fundamental bottleneck.

### Step 2: Profile the Code
```bash
python profile_octatope.py --compare-ranges
```

This will show if `estimate_ranges()` is the bottleneck.

### Step 3: Full Profiling
```bash
python profile_octatope.py --full-profile
```

Identify the top time-consuming functions and focus optimization there.

### Step 4: Try Differentiable Solver
```bash
cd ../AcasXu
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set octatope --method exact-differentiable
```

If this is much faster, the bottleneck is in CVXPY/LP solving.

### Step 5: Compare with Hexatope

Hexatope uses similar techniques but with DCS instead of UTVPI:

```bash
python verify_acasxu.py onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib \
  --set hexatope --method exact
```

If Hexatope is also slow, the issue is in the shared DCS/UTVPI infrastructure. If Hexatope is fast, the issue is specific to UTVPI handling.

## Expected Performance

Based on the Hexatope/Octatope paper (Bak et al., FMSD 2024), Octatopes should be:
- More precise than Stars (tighter bounds, more expressive constraints)
- Potentially slower due to UTVPI → DCS conversion overhead
- Reasonable performance with proper optimization

If Octatope is >10x slower than Star on the same problem, there's likely an implementation issue.

## Recommended Optimizations

1. **Cache range estimates** - Avoid re-computing bounds for unchanged Octatopes
2. **Optimize UTVPI handling** - The conversion to DCS may be inefficient
3. **Use differentiable solver** - May be faster than CVXPY for repeated small LPs
4. **Parallelize splitting** - Similar to Star parallel processing
5. **Lazy constraint evaluation** - Don't convert all constraints if not needed

## Questions to Answer

1. Is `estimate_ranges()` the bottleneck? (**Most likely**)
2. Is UTVPI → DCS conversion expensive?
3. Is CVXPY overhead significant for small LPs?
4. Are we creating too many Octatope sets during splitting?
5. Can we use the differentiable solver more effectively?

## Next Steps

After identifying the bottleneck:

1. **If `estimate_ranges()` is slow**: Optimize the LP solving in `_optimize_utvpi_lp()`
2. **If splitting is the issue**: Review the ReLU handling logic
3. **If CVXPY is slow**: Switch to differentiable solver or optimize solver usage
4. **If constraint handling is slow**: Optimize UTVPI constraint system operations

## References

- Bak et al., "The hexatope and octatope abstract domains for neural network verification", FMSD 2024
- UTVPI constraints: Unit-Two-Variables-Per-Inequality
- DCS: Difference Constraint Systems
