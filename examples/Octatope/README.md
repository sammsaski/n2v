# Hexatope/Octatope Verification Examples

Comparison scripts for hexatope and octatope set representations against star-based verification.

## Background

- **Star**: Primary set representation (`x = V * [1; a]` with `C*a <= d`). Supports exact reachability (splitting on crossing ReLUs) and approx (triangle relaxation). Bounds via LP.
- **Hexatope**: Difference Constraint System (DCS) — constraints of the form `x_i - x_j <= b`. Optimization reducible to min-cost flow (Theorems 5, 7 in Bak et al.). Less expressive than Stars but potentially faster per-optimization.
- **Octatope**: UTVPI system — constraints `a_i*x_i + a_j*x_j <= b` with coefficients in {-1, 0, +1}. Strictly more expressive than hexatopes, also MCF-reducible.

Hexatope/Octatope only support `method='approx'`. The `intersect_half_space` operation (Algorithm 5.1) always over-approximates via DCS/UTVPI bounding box, so there is no exact mode. Crossing ReLU neurons are split into active + inactive regions, with both kept as separate output sets. This is a sound over-approximation.

## Scripts

### `compare_tiny.py` — Level 1: Sanity Check

2->3->1 network with fixed weights, input bounds `[-1, 1]^2`.

```bash
python compare_tiny.py
```

Runs Star (exact/approx), Hexatope approx, and Octatope approx. Verifies soundness: hex/oct bounds must contain the Star exact bounds.

### `compare_small.py` — Level 2: Timing Comparison

5->10->5->1 network with `epsilon=0.05` perturbation.

```bash
python compare_small.py
python compare_small.py --epsilon 0.02  # tighter input region
```

Runs multiple configurations with LP and MCF solvers. Reports reach time, bounds extraction time, bound widths, and overhead vs Star exact.

## Solver API

Hexatope/Octatope methods require an explicit `solver` parameter:

```python
from n2v.sets import Hexatope, Octatope

h = Hexatope.from_bounds(lb, ub)
lb, ub = h.get_ranges(solver='lp')              # scipy linprog
lb, ub = h.get_ranges(solver='mcf')             # min-cost flow (NetworkX)
```

Star sets do not take a solver parameter — they always use LP on the predicate space.

## Known Limitations

- **Hexatope/Octatope are slow on large networks** (50+ neurons per layer) due to O(n^2) optimizations per `intersect_half_space` call.
- **MCF solver** can hang on problems with negative-cost edges from `intersect_half_space`.
