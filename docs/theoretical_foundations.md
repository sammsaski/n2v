# Theoretical Foundations

This document describes the mathematical representations, algorithmic choices, and relaxation techniques used throughout n2v. For each layer and set type, it explains whether the computation is **exact** or uses an **approximation**, and details the specific techniques employed.

---

## Table of Contents

- [Set Representations](#set-representations)
  - [Star](#star)
  - [ImageStar](#imagestar)
  - [Zonotope (Zono)](#zonotope-zono)
  - [ImageZono](#imagezono)
  - [Box](#box)
  - [ProbabilisticBox](#probabilisticbox)
  - [Hexatope](#hexatope)
  - [Octatope](#octatope)
- [Exact (Linear) Layer Operations](#exact-linear-layer-operations)
- [Nonlinear Activations](#nonlinear-activations)
  - [ReLU](#relu)
  - [LeakyReLU](#leakyrelu)
  - [Sigmoid](#sigmoid)
  - [Tanh](#tanh)
  - [Sign](#sign)
- [MaxPool2D](#maxpool2d)
- [Verification Algorithms](#verification-algorithms)
  - [Exact Reachability](#exact-reachability)
  - [Approximate Reachability](#approximate-reachability)
  - [Probabilistic Verification](#probabilistic-verification)
  - [Hybrid Verification](#hybrid-verification)
- [Falsification](#falsification)
- [Optimization Techniques](#optimization-techniques)
  - [Zonotope Pre-Pass for Bounds Tightening](#zonotope-pre-pass-for-bounds-tightening)
  - [LP Solver Selection](#lp-solver-selection)
  - [Parallel Computation](#parallel-computation)
  - [Zonotope Order Reduction](#zonotope-order-reduction)
- [Bounds Extraction: get_ranges() vs estimate_ranges()](#bounds-extraction-get_ranges-vs-estimate_ranges)

---

## Set Representations

### Star

The primary set representation. A Star set S is defined as:

```
S = { x ∈ ℝⁿ | x = c + V·α,  C·α ≤ d,  pred_lb ≤ α ≤ pred_ub }
```

where:
- `c ∈ ℝⁿ` is the center (stored as `V[:, 0]`)
- `V ∈ ℝⁿˣᵐ` is the generator matrix (stored as `V[:, 1:]`)
- `α ∈ ℝᵐ` are the predicate variables
- `C ∈ ℝᵖˣᵐ` is the constraint matrix
- `d ∈ ℝᵖ` is the constraint vector
- `pred_lb, pred_ub ∈ ℝᵐ` are box bounds on predicate variables

Stars are closed under affine maps and half-space intersections. Exact bounds are computed by solving linear programs (LPs) — one minimization and one maximization per output dimension.

### ImageStar

An extension of Star for image data. Instead of a 2D basis matrix, ImageStar uses a 4D tensor:

```
V ∈ ℝ^(H × W × C × (nVar+1))
```

where `V[:,:,:,0]` is the center image and `V[:,:,:,1:]` are generator images. This preserves spatial structure, allowing `conv2d`, pooling, and padding operations to be applied directly to the 4D tensor without flattening.

Conversion to a flat Star uses CHW (channel-first) ordering for PyTorch compatibility.

### Zonotope (Zono)

A zonotope Z is defined as:

```
Z = { c + V·α | -1 ≤ αᵢ ≤ 1  for all i }
```

where:
- `c ∈ ℝⁿ` is the center
- `V ∈ ℝⁿˣᵐ` is the generator matrix (m generators)

Zonotopes are closed under affine maps and Minkowski sums. Bounds are computed **analytically** without LP:

```
lb[i] = c[i] - Σⱼ |V[i,j]|
ub[i] = c[i] + Σⱼ |V[i,j]|
```

Zonotopes are less expressive than Stars (no arbitrary linear constraints), but bound computation is O(n·m) rather than requiring LP solves. They are primarily used for over-approximation in approximate methods and for the Zono pre-pass (see [Optimization Techniques](#zonotope-pre-pass-for-bounds-tightening)).

### ImageZono

The 4D analogue of Zono for image data, mirroring the relationship between Star and ImageStar.

### Box

The simplest representation — an axis-aligned hyperrectangle:

```
B = { x ∈ ℝⁿ | lb ≤ x ≤ ub }
```

A Box is a special case of both a Star and a Zonotope (with a diagonal generator matrix). It is the fastest representation for bound propagation but the most conservative (largest over-approximation).

### ProbabilisticBox

Inherits from Box and augments it with conformal inference metadata:

```
ProbabilisticBox = Box(lb, ub) + {m, ℓ, ε, coverage, confidence}
```

where:
- `m` = calibration set size
- `ℓ` = rank parameter (which order statistic is used as threshold)
- `ε` = miscoverage level
- `coverage = 1 - ε` = guaranteed fraction of outputs within bounds
- `confidence = 1 - B_{1-ε}(ℓ, m+1-ℓ)` = probability that the coverage guarantee holds

See [Probabilistic Verification](#probabilistic-verification) for the theory behind these guarantees.

### Hexatope

A constrained zonotope where constraints form a **Difference Constraint System (DCS)**:

```
H = { G·x + c | A·x ≤ b },    A is DCS
```

DCS constraints have the form `xᵢ - xⱼ ≤ b`, which can be represented as edges in a directed graph. This structure enables **strongly polynomial optimization** via minimum cost flow (MCF), avoiding LP entirely.

An anchor variable x₀ = 0 serves as a reference point, encoding absolute bounds as `xᵢ - x₀ ≤ u` and `x₀ - xᵢ ≤ -l`.

### Octatope

A constrained zonotope where constraints form a **UTVPI (Unit Two-Variable Per Inequality) system**:

```
O = { G·x + c | A·x ≤ b },    A is UTVPI
```

UTVPI constraints have the form `aᵢ·xᵢ + aⱼ·xⱼ ≤ b` where `aᵢ, aⱼ ∈ {-1, 0, +1}`. This is strictly more expressive than DCS. Optimization is performed by converting to a DCS via variable splitting (`xᵢ = 0.5·(xᵢ⁺ - xᵢ⁻)`), then using MCF on the expanded system.

---

## Exact (Linear) Layer Operations

The following layers are **affine (linear) transformations** and are computed **exactly** for all applicable set types. No approximation or relaxation is involved.

| Layer | Operation | Notes |
|-------|-----------|-------|
| **Linear** (`nn.Linear`) | `y = W·x + b` | Affine map applied to set generators |
| **Conv2D** (`nn.Conv2d`) | Convolution | Linear; applied directly to 4D ImageStar/ImageZono generators via `F.conv2d` |
| **Conv1D** (`nn.Conv1d`) | Convolution | Builds explicit convolution matrix, then applies as affine map |
| **BatchNorm** (`nn.BatchNorm1d/2d`) | `y = γ/√(σ²+ε) · x + (β - γμ/√(σ²+ε))` | Channel-wise affine in eval mode; can be fused with preceding Linear/Conv |
| **AvgPool2D** (`nn.AvgPool2d`) | Average pooling | **Linear** — averaging is a weighted sum. No splitting, no approximation |
| **GlobalAvgPool** (`nn.AdaptiveAvgPool2d(1)`) | Global spatial mean | Averages over H×W dimensions; linear |
| **Flatten** (`nn.Flatten`) | Reshape | Reorders 4D ImageStar to flat Star using CHW ordering |
| **Pad** (`nn.ZeroPad2d`, etc.) | Zero-padding | Extends spatial dimensions with zeros; linear |
| **Upsample** (`nn.Upsample`, nearest) | Nearest-neighbor upsampling | Replicates elements via `np.repeat`; linear |
| **Reduce** (`ReduceSum`, `ReduceMean`) | Sum/mean along axes | Linear operations |
| **Transpose** | Coordinate permutation | Permutes rows of generator matrix |
| **Neg** | Negation | `V → -V`, swaps bounds |
| **Identity/Dropout/Cast** | No-op | Passed through unchanged |

**Key insight**: AvgPool2D is often highlighted because it is a common alternative to MaxPool2D that avoids the exponential splitting problem entirely (see [MaxPool2D](#maxpool2d)).

### Set Type Coverage for Linear Operations

| Layer | Star/ImageStar | Zono/ImageZono | Box | Hexatope/Octatope |
|-------|----------------|----------------|-----|-------------------|
| Linear | Exact | Exact | Exact | Exact |
| Conv2D | Exact (ImageStar) | Exact (ImageZono) | — | — |
| Conv1D | Exact | Exact | Exact | — |
| BatchNorm | Exact | Exact | Exact | Exact (via diagonal Linear) |
| AvgPool2D | Exact (ImageStar) | Exact (ImageZono) | — | — |
| GlobalAvgPool | Exact (ImageStar) | Exact (ImageZono) | — | — |
| Flatten | Exact | Exact | No-op | No-op |
| Pad | Exact (ImageStar) | Exact (ImageZono) | — | — |
| Upsample | Exact (ImageStar) | Exact (ImageZono) | — | — |
| Reduce | Exact | Exact | Exact | — |

---

## Nonlinear Activations

These layers require splitting (for exact analysis) or relaxation (for over-approximate analysis). The specific techniques differ by activation function and set type.

### ReLU

```
ReLU(x) = max(0, x)
```

#### Exact Method (Star) / Splitting (Hexatope, Octatope)

For each neuron i with bounds [lᵢ, uᵢ]:

1. **Always inactive** (`uᵢ ≤ 0`): Output is 0. Set the i-th row of V to zero.
2. **Always active** (`lᵢ ≥ 0`): Output is x. No change needed.
3. **Crossing** (`lᵢ < 0 < uᵢ`): **Split** into two sub-Stars:
   - **Inactive case**: Add constraint `Vᵢ·α ≤ -cᵢ` (ensuring xᵢ ≤ 0), set output to 0
   - **Active case**: Add constraint `-Vᵢ·α ≤ cᵢ` (ensuring xᵢ ≥ 0), keep output as-is

Each crossing neuron doubles the number of Stars, leading to worst-case **2ᵏ** Stars for k crossing neurons. In practice, many splits produce empty (infeasible) Stars that are pruned.

#### Approximate Method — Triangle Relaxation (Star)

For each crossing neuron with bounds [l, u], introduce three linear constraints that form a **triangle** enclosing the ReLU graph:

```
y ≥ 0                           (lower bound from negative side)
y ≥ x                           (lower bound from positive side)
y ≤ u·(x - l) / (u - l)        (secant upper bound connecting (l,0) to (u,u))
```

This adds a new predicate variable per crossing neuron with the relaxation constraints, avoiding splitting entirely. The result is a single (larger) Star that over-approximates the exact reachable set.

#### Approximate Method — Interval Over-Approximation (Zono)

For zonotopes, ReLU is handled by computing the output center and adding error generators for crossing neurons:

- **Always inactive**: Center → 0, generators → 0
- **Always active**: No change
- **Crossing**: Compute the secant slope `λ = u/(u-l)`, shift center, and add a new error generator capturing the approximation gap

#### Box

Elementwise: `lb_out = max(0, lb)`, `ub_out = max(0, ub)`. Exact for the Box representation.

### LeakyReLU

```
LeakyReLU(x) = x        if x ≥ 0
             = γ·x      if x < 0
```

where γ is the negative slope (typically 0.01).

#### Exact Method (Star)

Same splitting strategy as ReLU, but the inactive case maps to `γ·x` instead of 0:
- **Inactive case**: Scale the i-th row of V by γ (instead of zeroing)
- **Active case**: Unchanged

#### Approximate Method — Modified Triangle Relaxation (Star)

For crossing neurons [l, u], the three constraints become:

```
y ≥ γ·x                              (lower bound: LeakyReLU line for x < 0)
y ≥ x                                (lower bound: identity for x ≥ 0)
y ≤ a·x + b                          (upper bound: secant from (l, γl) to (u, u))
    where a = (u - γl)/(u - l),  b = (γl·u - u·l)/(u - l) = 0 when simplified
```

#### Approximate Method (Zono)

Similar to ReLU Zono approximation but with secant slope adjusted for the negative slope γ.

### Sigmoid

```
σ(x) = 1 / (1 + e⁻ˣ)
```

Sigmoid is a smooth, bounded, monotonic nonlinearity. There is **no exact Star method** — only approximation is available (a warning is emitted if `method='exact'` is requested).

#### Approximate Method — Tangent/Secant S-Curve Relaxation (Star)

This relaxation technique (from NNV) partitions neurons into regions based on their bounds:

**Convex region** (`l ≥ 0`):
σ is convex-then-concave, but on [0, ∞) it starts convex. Two tangent lines form the upper bound, one secant line forms the lower bound:
- Upper bounds: tangent at l, tangent at u
- Lower bound: secant from (l, σ(l)) to (u, σ(u))

**Concave region** (`u ≤ 0`):
Mirror of the convex case:
- Lower bounds: tangent at l, tangent at u
- Upper bound: secant from (l, σ(l)) to (u, σ(u))

**Mixed region** (`l < 0 < u`):
Uses the inflection point properties of sigmoid (`σ(0) = 0.5`, `σ'(0) = 0.25`):
- 4 constraints using tangent and secant lines from both sides of the inflection point
- LP-based tightening selects the best tangent points

Each crossing neuron adds a new predicate variable with these relaxation constraints.

#### Zonotope and Box

- **Zono**: Interval over-approximation — evaluate σ on bounds, create interval-based zonotope
- **Box**: Direct application of monotonicity — `σ(lb) ≤ σ(x) ≤ σ(ub)`

### Tanh

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

Tanh shares the same S-curve structure as sigmoid. The implementation **reuses the sigmoid relaxation algorithm** with different function parameters:
- Inflection point: `f(0) = 0` (vs 0.5 for sigmoid)
- Derivative at inflection: `f'(0) = 1` (vs 0.25 for sigmoid)

All three region partitioning strategies (convex, concave, mixed) work identically with these substituted values.

### Sign

```
sign(x) = -1  if x < 0
         =  0  if x = 0
         = +1  if x > 0
```

Sign is a discontinuous nonlinearity used in binary neural networks (BNNs).

#### Exact Method (Star) — 2-Way Splitting

For each crossing neuron with bounds [l, u] where l < 0 < u:
- **Case 1** (`x ≤ 0`): Add constraint `Vᵢ·α ≤ -cᵢ`, map output to -1
- **Case 2** (`x ≥ 0`): Add constraint `-Vᵢ·α ≤ cᵢ`, map output to +1

Note: x = 0 has measure zero, so the 2-way split (ignoring the 0 case) is standard.

#### Approximate Method — Parallelogram Relaxation (Star)

For each crossing neuron, introduce a new predicate variable y ∈ [-1, +1] with 4 constraints forming a parallelogram in the (x, y) plane:

```
y ≥ -1                    (box lower bound)
y ≤ +1                    (box upper bound)
y ≥ (2/(u-l))·x - (u+l)/(u-l)    (secant from (l,-1) to (u,+1))
y ≤ (2/(u-l))·x - (u+l)/(u-l)    (parallel secant)
```

This produces a sound over-approximation without splitting.

#### Box

Elementwise interval evaluation: if lb > 0 → output 1; if ub < 0 → output -1; otherwise → [-1, 1].

---

## MaxPool2D

MaxPool2D is the only **pooling** layer that introduces nonlinearity, because `max` is not a linear operation.

### Exact Method (ImageStar) — Splitting with LP

For each pooling window and channel:
1. Compute bounds for all pixels in the window using LP
2. Find the pixel with the maximum lower bound (the "best candidate")
3. Identify all pixels whose upper bound ≥ that maximum lower bound (the "candidate set")
4. If only 1 candidate: use that pixel's value directly (no split needed)
5. If >1 candidate: **split** the Star — create one sub-Star per candidate, adding constraints ensuring that candidate is ≥ all others in the window

This can produce many splits for large pooling windows with overlapping bounds.

### Approximate Method (ImageStar) — New Predicate Variables

Instead of splitting, introduce a **new predicate variable** for each uncertain pooling window:
- The new variable is bounded by [lb, ub] of the pool window
- Add relaxation constraints: secant upper bound, linear lower bound
- This avoids splitting but over-approximates

### Zonotope, Hexatope, Octatope — Bounds Over-Approximation

For these set types, MaxPool is handled via interval arithmetic:
```
ub_out = max_pool2d(ub_in)
lb_out = -max_pool2d(-lb_in)    (equivalent to min_pool2d)
```

This is a sound but potentially loose over-approximation.

---

## Verification Algorithms

### Exact Reachability

**Method**: `'exact'`
**Guarantee**: Sound and complete
**Set types**: Star, ImageStar

Propagates sets layer-by-layer through the network. Linear layers transform sets exactly. Nonlinear layers (ReLU, MaxPool) split sets, producing multiple output Stars. The union of all output Stars is the exact reachable set.

**Complexity**: Worst-case exponential in the number of crossing neurons (due to splitting).

### Approximate Reachability

**Method**: `'approx'`
**Guarantee**: Sound (over-approximate) but not complete
**Set types**: Star, ImageStar, Zono, ImageZono, Box, Hexatope, Octatope

Uses relaxation techniques (triangle relaxation for ReLU, tangent/secant for sigmoid/tanh) instead of splitting. Always produces a **single** output set that contains the true reachable set.

**Complexity**: Polynomial in network size (no splitting).

**Trade-off**: May produce false positives (reporting "unknown" when a property actually holds) but never false negatives.

### Probabilistic Verification

**Method**: `'probabilistic'`
**Guarantee**: Coverage guarantee with high confidence (not sound in the traditional sense)
**Set types**: Any (model-agnostic, treats network as black box)

Uses **conformal inference** to produce output bounds with formal probabilistic guarantees. The approach works with any callable model — no network analysis required.

#### Algorithm (8 steps)

1. **Generate training samples**: Draw t samples uniformly from the input set
2. **Dimensionality reduction** (optional): Apply Deflation PCA if output dimension is high
3. **Fit surrogate model**: Learn a simple predictor of network outputs
   - *Naive*: Uses the mean of training outputs as prediction for all inputs
   - *Clipping Block*: Projects outputs onto the convex hull of training outputs via LP
4. **Generate calibration samples**: Draw m independent samples from the input set
5. **Compute calibration errors**: `eᵢ = outputᵢ - surrogate.predict(outputᵢ)`
6. **Conformal inference**:
   - Compute normalization factors: `τₖ = max(τ*, maxⱼ |eⱼₖ|)` per dimension k
   - Compute nonconformity scores: `Rᵢ = maxₖ(|eᵢₖ|/τₖ)` (normalized ∞-norm)
   - Select threshold: `R_ℓ` = ℓ-th smallest score
   - Compute confidence: `δ₂ = 1 - B_{1-ε}(ℓ, m+1-ℓ)` (Beta CDF)
7. **Compute final bounds**: `surrogate_bounds ± τ · R_ℓ` per dimension
8. **Return ProbabilisticBox** with coverage = 1-ε and confidence = δ₂

#### Guarantee

With confidence at least δ₂:

```
Pr[ f(x) ∈ [lb, ub]  for x ~ Uniform(input_set) ] ≥ 1 - ε
```

#### Surrogate Models

**Naive Surrogate**: Predicts the center (mean) of training outputs for all inputs. Simple but conservative — the inflation must account for all variation.

**Clipping Block Surrogate**: Projects each calibration output onto the convex hull of training outputs by solving an LP:

```
min_α  ||y - Σⱼ αⱼ·yⱼ||_∞
s.t.   Σⱼ αⱼ = 1,  αⱼ ≥ 0
```

This exploits correlation structure in the outputs, producing tighter bounds than naive.

#### Deflation PCA

For high-dimensional outputs (e.g., semantic segmentation with thousands of output neurons), standard PCA requires an n×n covariance matrix. Deflation PCA avoids this by extracting principal components one at a time via gradient ascent:

```
For each component i:
    max_a  (1/t) Σⱼ (aᵀzⱼ)²    s.t. ||a||₂ = 1
    Deflate: zⱼ ← zⱼ - (aᵀzⱼ)·a
```

This has complexity O(t·n) per component instead of O(n²), enabling PCA when n >> t.

### Hybrid Verification

**Method**: `'hybrid'`
**Guarantee**: Depends on where the switch occurs

Starts with deterministic (exact) reachability and monitors two thresholds:
- **Star count**: If the number of Stars exceeds `max_stars` (default: 1000)
- **Layer time**: If a single layer takes longer than `timeout_per_layer` (default: 30s)

When either threshold is exceeded, the method extracts bounds from the current Stars, constructs a Box, and runs probabilistic verification on the remaining layers.

This provides exact guarantees for the early (typically linear) layers and probabilistic guarantees for the deeper (more complex) layers.

---

## Falsification

Falsification attempts to find **counterexamples** (inputs that violate the property) before running expensive reachability analysis. If a counterexample is found, verification is unnecessary.

**Limitation**: The current implementation assumes hyperbox input sets (axis-aligned bounds `[lb, ub]`).

### Random Sampling

Uniformly sample inputs from `[lb, ub]`, evaluate the network, and check each output against the property specification.

- **Complexity**: O(n_samples · forward_pass)
- **Strengths**: Fast, broad exploration
- **Weaknesses**: May miss adversarial examples in small regions

### Projected Gradient Descent (PGD)

For each constraint group in the property, formulate a loss function and minimize via gradient descent:

```
loss = max over groups of (min over halfspaces in group of max(G·y - g))
```

Steps:
1. Initialize with random input from `[lb, ub]`
2. Compute gradient of loss w.r.t. input
3. Step: `x ← x - step_size · sign(∇loss)`
4. Project back onto `[lb, ub]`
5. Check if output violates the property

Multiple restarts with random initialization improve coverage.

### Combined (random+pgd)

Run random sampling first (fast, broad). If no counterexample found, run PGD (slower, targeted). Return immediately if either finds a counterexample.

---

## Optimization Techniques

### Zonotope Pre-Pass for Bounds Tightening

**Feature**: `precompute_bounds=True` in `reach()` kwargs

Before running exact Star reachability, perform a fast Zono pre-pass through the entire network. This produces intermediate layer bounds that are passed to subsequent layers as `precomputed_bounds`.

**Impact on ReLU**: With tighter input bounds, more neurons can be classified as "always active" or "always inactive" (dead neuron elimination), reducing the number of splits. This is particularly effective for deep networks where bounds would otherwise grow very loose.

**Trade-off**: The Zono pre-pass has a small fixed cost, but the savings from reduced splitting can be substantial.

### LP Solver Selection

n2v supports two LP solver backends:

| Solver | Command | Speed | Notes |
|--------|---------|-------|-------|
| CVXPY | `n2v.set_lp_solver('cvxpy')` | Baseline | Multiple backend solvers, more features |
| scipy linprog (HiGHS) | `n2v.set_lp_solver('linprog')` | **1.5-2x faster** | Lower setup overhead, efficient sparse handling |

The HiGHS solver (via scipy) is recommended for most use cases due to lower per-LP overhead, which matters when thousands of LPs are solved during exact verification.

### Parallel Computation

```python
n2v.set_parallel(True, n_workers=N)
```

Parallelism is applied at two levels:
1. **LP solving**: Multiple `get_range()` calls run concurrently via `ThreadPoolExecutor`
2. **Star processing**: Multiple Stars through ReLU are processed concurrently via `ProcessPoolExecutor`

The heuristic activates parallelism only for dimensions > 10 (to avoid threading overhead for small problems).

### Zonotope Order Reduction

When zonotopes accumulate too many generators (from Minkowski sums or nonlinear over-approximations), order reduction keeps the representation compact:

1. Sort generators by 2-norm (largest first)
2. Keep the `n_keep` largest generators
3. Over-approximate the remaining generators with an interval hull (diagonal generator matrix)

```
Total generators after reduction = n_keep + dim
If target_order = k, then n_keep = (k-1)·dim, so final count = k·dim
```

This is a sound over-approximation: the reduced zonotope contains the original.

---

## Bounds Extraction: get_ranges() vs estimate_ranges()

**`get_ranges()`**: Solves LPs to find the true min/max within the Star's polytope. This accounts for all constraints `C·α ≤ d` and gives exact bounds for the Star representation.

**`estimate_ranges()`**: Fast over-approximation using only predicate bounds (interval arithmetic on generators). Ignores the constraint matrix C entirely.

```
estimate: lb[i] = c[i] + pos_gens[i,:] @ pred_lb + neg_gens[i,:] @ pred_ub
          ub[i] = c[i] + pos_gens[i,:] @ pred_ub + neg_gens[i,:] @ pred_lb
```

**When to use which**:
- `get_ranges()`: Always use for **final verification results** and any bounds reported to users
- `estimate_ranges()`: Appropriate for **internal decisions** (ReLU splitting heuristics, quick screening) where speed matters and over-approximation is safe

Using `estimate_ranges()` for comparing exact vs approximate methods can give misleading results, as the approximate method may appear tighter than exact (because exact's constraints are being ignored in the estimate).

---

## Layer Support Summary

| Layer | Star (exact) | Star (approx) | Zono | Box | Hexatope/Octatope |
|-------|:------------:|:--------------:|:----:|:---:|:-----------------:|
| Linear | Exact | Exact | Exact | Exact | Exact |
| Conv2D | Exact* | Exact* | Exact* | — | — |
| Conv1D | Exact | Exact | Exact | Exact | — |
| BatchNorm | Exact | Exact | Exact | Exact | Exact |
| ReLU | Split | Triangle | Interval | Elementwise | Split/Relax |
| LeakyReLU | Split | Triangle | Interval | Elementwise | Split/Relax |
| Sigmoid | — | S-curve | Interval | Monotone | — |
| Tanh | — | S-curve | Interval | Monotone | — |
| Sign | Split | Parallelogram | Interval | Elementwise | — |
| MaxPool2D | Split+LP | New predicates | Bounds | — | Bounds |
| AvgPool2D | Exact* | Exact* | Exact* | — | — |
| GlobalAvgPool | Exact* | Exact* | Exact* | — | — |
| Flatten | Exact* | Exact* | No-op | No-op | No-op |
| Pad | Exact* | Exact* | Exact* | — | — |
| Upsample | Exact* | Exact* | Exact* | — | — |
| Reduce | Exact | Exact | Exact | Exact | — |
| Transpose | Exact | Exact | Exact | Exact | — |
| Neg | Exact | Exact | Exact | Exact | Exact |
| Identity/Dropout/Cast | No-op | No-op | No-op | No-op | No-op |

**Legend**: *image-aware set types (ImageStar/ImageZono) required for spatial operations

---

## ONNX Operations

n2v supports ONNX models (via onnx2torch) with the following additional operations handled in the graph execution engine:

| ONNX Operation | Implementation |
|----------------|---------------|
| OnnxReshape | Transpose/flatten with NCHW ↔ HWC conversion |
| OnnxConcat | Element-wise concatenation along specified axis |
| OnnxSlice/SliceV9 | Rectangular slicing with axis mapping |
| OnnxSplit/Split13 | Splitting along axis with size specification |
| OnnxBinaryMath (Add, Sub, Mul, Div) | One computed operand, one constant |
| OnnxMatMul | Matrix multiplication with constant weights |
| OnnxReduceSum/ReduceMean | Reduction along specified axes |
| OnnxResize | Nearest-neighbor upsampling (scale detection via forward probing) |
| OnnxTranspose | Coordinate permutation |
| OnnxNeg | Negation |
| OnnxCast | Type casting (no-op for reachability) |
| OnnxPad | Zero-padding |

Element-wise multiplication between two computed (non-constant) operands uses **McCormick relaxation** (4 envelope constraints per dimension) for Star sets.

---

## References

- Tran, H.D., et al. "Star-based reachability analysis of deep neural networks." *FM 2019*.
- Lopez, D.M., et al. "NNV 2.0: The Neural Network Verification Tool." *CAV 2023*.
- Vovk, V., et al. "Algorithmic Learning in a Random World." Springer, 2005. (Conformal inference)
- Singh, G., et al. "An abstract domain for certifying neural networks." *POPL 2019*. (Zonotope abstractions)
