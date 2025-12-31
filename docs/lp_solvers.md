# LP Solver Selection Guide

## Overview

n2v uses Linear Programming (LP) to compute tight bounds on set representations during neural network verification. The choice of LP solver significantly impacts performance, especially for methods that solve many LPs (exact reachability, relaxed methods).

This guide explains the available solvers and when to use each.

---

## Available Solvers

### scipy linprog (HiGHS) - Recommended

```python
import n2v
n2v.set_lp_solver('linprog')
```

- **Backend**: scipy.optimize.linprog with HiGHS solver
- **Speed**: 1.5-2x faster than CVXPY for typical Star set LPs
- **Strengths**: Minimal problem setup overhead, handles sparse matrices efficiently

**When to use:**
- Performance is critical
- Running many LPs (exact method, relaxed methods with low relax factors)
- Problem structure is simple (standard LP form)

```python
import n2v
n2v.set_lp_solver('linprog')

# Good for:
verifier.reach(input_star, method='exact')
verifier.reach(input_star, method='approx', relax_factor=0.25)
```

**HiGHS variants:**
```python
n2v.set_lp_solver('highs')      # Auto-select (same as 'linprog')
n2v.set_lp_solver('highs-ds')   # Dual simplex
n2v.set_lp_solver('highs-ipm')  # Interior point method
```

For most n2v use cases, the default HiGHS algorithm selection works best.

### CVXPY (Default)

```python
import n2v
n2v.set_lp_solver('default')  # Uses CLARABEL
n2v.set_lp_solver('ECOS')     # Alternative CVXPY solver
n2v.set_lp_solver('SCS')      # Alternative CVXPY solver
```

- **Backend**: CVXPY with CLARABEL (default) or other installed solvers
- **Speed**: Moderate (problem construction overhead)
- **Strengths**: Rich API, easy debugging, supports many solver backends

**When to use:**
- Debugging or developing new features
- Need access to solution details (dual values, slack, etc.)
- Problem has special structure that benefits from CVXPY's modeling

```python
import n2v
n2v.set_lp_solver('default')

# Good for:
# - Debugging verification issues
# - Prototyping new verification methods
```

### Gurobi (TODO)

Support for Gurobi is planned but not yet implemented. Gurobi is a commercial-grade LP solver that typically provides 10-100x speedup over open-source alternatives.

- **License**: Free for academics at https://www.gurobi.com/academia/

---

## Why scipy linprog is Faster

The performance difference comes from problem setup overhead, not solver speed:

| Component | CVXPY | scipy linprog |
|-----------|-------|---------------|
| Problem construction | ~57% of time | Minimal |
| Solver execution | ~26% of time | ~90% of time |
| Result extraction | ~17% of time | Minimal |

CVXPY builds an optimization graph for each problem, which adds overhead. scipy linprog directly passes matrices to HiGHS with minimal processing.

---

## Problem Structure in n2v

### Star Set LP Structure

When computing bounds for Star sets, n2v solves LPs of the form:

```
maximize/minimize  e_i^T * x
subject to:        x = V * [1; α]
                   C * α ≤ d
                   lb ≤ α ≤ ub
```

Characteristics:
- **Variables**: Number of Star predicates (typically 10-1000)
- **Constraints**: Linear predicates from previous layers
- **Sparsity**: Constraint matrices are ~99.9% sparse
- **Structure**: Mostly identity + triangle relaxation constraints

### Why HiGHS Works Well

HiGHS (High-performance Interior-point and Simplex) is well-suited because:

1. **Sparse matrix support**: Constraint matrices are highly sparse
2. **Warm starting**: Can reuse factorizations across similar LPs
3. **Dual simplex**: Effective for bound computation problems
4. **Low overhead**: Minimal problem transformation

---

## Benchmarking

To compare LP solver performance on your specific workload, use the benchmark suite:

```bash
# Compare default (CVXPY) vs scipy linprog
python benchmarks/run_benchmarks.py --lp-solver default
python benchmarks/run_benchmarks.py --lp-solver linprog
```

See [benchmarks/README.md](../benchmarks/README.md) for more options.

---

## Troubleshooting

### "Solver failed" errors

If you see solver failures with one backend, try another:

```python
# If linprog fails on specific problems
n2v.set_lp_solver('default')  # Fall back to CVXPY
```

### Numerical precision differences

Different solvers may return slightly different optimal values:
- Differences of ~1e-6 to 1e-10 are normal
- For bound computation, this is acceptable

### Memory usage

- scipy linprog: Lower memory (no graph construction)
- CVXPY: Higher memory (caches problem structure)
- For very large problems, prefer scipy linprog
