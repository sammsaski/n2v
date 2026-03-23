# Development Status

Feature inventory and development roadmap for n2v.

Last updated: 2026-03-23

---

## Table of Contents

- [Completed Features](#completed-features)
  - [Set Representations](#set-representations)
  - [Layer Operations](#layer-operations)
  - [Verification Methods](#verification-methods)
  - [Falsification](#falsification)
  - [Performance Optimizations](#performance-optimizations)
  - [ONNX Support](#onnx-support)
  - [VNN-COMP Infrastructure](#vnn-comp-infrastructure)
  - [Testing](#testing)
  - [Documentation](#documentation)
- [Known Limitations](#known-limitations)
- [To-Do](#to-do)
  - [Missing Layers](#missing-layers)
  - [Missing Set Type Coverage](#missing-set-type-coverage)
  - [Falsification Improvements](#falsification-improvements)
  - [Infrastructure](#infrastructure)
  - [Documentation](#documentation-1)

---

## Completed Features

### Set Representations

| Set Type | Description | Status |
|----------|-------------|--------|
| Star | Polytopic constraints `x = c + V·α, C·α ≤ d` | Complete |
| ImageStar | 4D Star for CNNs (H, W, C, nVar+1) | Complete |
| Zono | Zonotope `x = c + V·α, α ∈ [-1,1]` | Complete |
| ImageZono | 4D Zono for CNNs | Complete |
| Box | Axis-aligned hyperrectangle `lb ≤ x ≤ ub` | Complete |
| ProbabilisticBox | Box + conformal inference guarantees (coverage, confidence) | Complete |
| Hexatope | DCS-constrained zonotope, min-cost flow optimization | Complete |
| Octatope | UTVPI-constrained zonotope, strongly polynomial optimization | Complete |
| HalfSpace | Linear constraint `G·x ≤ g` | Complete |

All set types support: `from_bounds()`, `affine_map()`, `get_ranges()`, `estimate_ranges()`, `contains()`, `is_empty_set()`, `to_star()`, `sample()`.

### Layer Operations

#### Exact (Linear) Layers

| Layer | Star/ImageStar | Zono/ImageZono | Box | Hex/Oct |
|-------|:--------------:|:--------------:|:---:|:-------:|
| Linear (`nn.Linear`) | Done | Done | Done | Done |
| Conv2D (`nn.Conv2d`) | Done | Done | -- | -- |
| Conv1D (`nn.Conv1d`) | Done | Done | Done | -- |
| BatchNorm (`nn.BatchNorm1d/2d`) | Done | Done | Done | Done (via diagonal Linear) |
| AvgPool2D (`nn.AvgPool2d`) | Done | Done | -- | -- |
| GlobalAvgPool (`nn.AdaptiveAvgPool2d(1)`) | Done | Done | -- | -- |
| Flatten (`nn.Flatten`) | Done | Done | No-op | No-op |
| Pad (`nn.ZeroPad2d`, `nn.ConstantPad2d`) | Done | Done | -- | -- |
| Upsample (`nn.Upsample`, nearest) | Done | Done | -- | -- |
| Reduce (ReduceSum, ReduceMean) | Done | Done | Done | -- |
| Transpose | Done | Done | Done | -- |
| Neg | Done | Done | Done | Done |
| Identity / Dropout / Cast | No-op | No-op | No-op | No-op |
| Sequential (`nn.Sequential`) | Recursive | Recursive | Recursive | Recursive |

#### Nonlinear Layers

| Layer | Star exact | Star approx | Zono | Box | Hex/Oct |
|-------|:----------:|:-----------:|:----:|:---:|:-------:|
| ReLU (`nn.ReLU`) | 2-way split | Triangle relaxation | Interval over-approx | Elementwise | Split / Relaxation |
| LeakyReLU (`nn.LeakyReLU`) | 2-way split | Modified triangle | Interval over-approx | Elementwise | -- |
| Sigmoid (`nn.Sigmoid`) | -- (approx only) | Tangent/secant S-curve | Interval over-approx | Monotone | -- |
| Tanh (`nn.Tanh`) | -- (approx only) | S-curve (shared w/ sigmoid) | Interval over-approx | Monotone | -- |
| Sign | 2-way split | Parallelogram relaxation | Interval over-approx | Elementwise | -- |
| MaxPool2D (`nn.MaxPool2d`) | Split + LP candidates | New predicate variables | Bounds over-approx | -- | Bounds over-approx |

### Verification Methods

| Method | Guarantee | Description |
|--------|-----------|-------------|
| `exact` | Sound and complete | Star splitting at nonlinear layers |
| `approx` | Sound (over-approximate) | Triangle/S-curve relaxation, no splitting |
| `probabilistic` | Coverage with confidence | Conformal inference, model-agnostic |
| `hybrid` | Mixed | Exact until star/time threshold, then probabilistic fallback |

### Falsification

| Method | Description |
|--------|-------------|
| `random` | Uniform sampling from input bounds |
| `pgd` | Projected Gradient Descent with multiple restarts |
| `random+pgd` | Random first, PGD if no counterexample found |

### Performance Optimizations

| Feature | Description |
|---------|-------------|
| scipy linprog (HiGHS) | 1.5-2x faster than CVXPY; `n2v.set_lp_solver('linprog')` |
| Parallel LP solving | Multi-worker via ThreadPoolExecutor; `n2v.set_parallel(True, n_workers=N)` |
| Zono pre-pass | Precompute intermediate bounds to eliminate stable neurons before exact ReLU splitting |
| BatchNorm fusion | Fuse BatchNorm into preceding Linear/Conv for fewer layers |
| 4D ImageStar ops | Conv2D and pooling applied directly to 4D tensors without flattening |

### ONNX Support

Models loaded via `load_onnx()` are converted to PyTorch `GraphModule` via onnx2torch. The graph execution engine handles these ONNX-specific operations:

| Operation | Description |
|-----------|-------------|
| OnnxReshape | Format conversion (NCHW to HWC and back) |
| OnnxConcat | Concatenation along specified axis |
| OnnxSlice / OnnxSliceV9 | Rectangular slicing with axis mapping |
| OnnxSplit / OnnxSplit13 | Splitting along axis |
| OnnxBinaryMathOperation | Add, Sub, Mul, Div (one computed operand, one constant) |
| OnnxMatMul | Matrix multiplication with constant weights |
| OnnxReduceSum / OnnxReduceMean | Reduction along specified axes |
| OnnxResize | Nearest-neighbor upsampling (scale detection via forward probing) |
| OnnxTranspose | Coordinate permutation |
| OnnxNeg | Negation |
| OnnxCast | Type casting (no-op) |
| OnnxPad | Zero-padding |

Element-wise multiplication between two computed (non-constant) Star operands uses McCormick relaxation (4 envelope constraints per dimension).

### VNN-COMP Infrastructure

| Component | Description |
|-----------|-------------|
| `run_instance.py` | 3-stage verifier: falsify, approx, exact |
| `prepare_instance.py` | ONNX model loading + VNNLIB parsing + input set creation |
| `benchmark_configs.py` | Per-benchmark strategies for 28 benchmarks |
| `smoke_test.sh` | 1 instance per benchmark, quick compatibility check |
| `run_benchmark.sh` | All instances from a single benchmark directory |
| VNN-COMP output format | `sat`/`unsat`/`unknown`/`timeout` + counterexample formatting |

### Testing

| Category | Count | Description |
|----------|-------|-------------|
| Unit tests | ~1060 | Sets, layer ops, dispatcher, VNNLIB parsing, probabilistic, integration |
| Soundness tests | ~190 | Mathematical correctness: exact contains approx, bounds valid, ground truth |
| Skipped tests | 2 | Conv2d Zonotope (not implemented), sklearn PCA (optional dep) |

Soundness test coverage exists for: Linear, ReLU, LeakyReLU, Sigmoid, Tanh, Conv2D, Conv1D, MaxPool2D, AvgPool2D, Flatten, Sign, Upsample, BatchNorm, Probabilistic.

### Documentation

| Document | Description |
|----------|-------------|
| `README.md` | Project overview, usage guide, API reference |
| `CLAUDE.md` | Architecture reference for AI assistants |
| `docs/theoretical_foundations.md` | Mathematical details for all layers, relaxations, algorithms |
| `docs/probabilistic_verification.md` | Conformal inference theory and API guide |
| `docs/lp_solvers.md` | LP solver selection and benchmarking guide |
| `examples/ACASXu/README.md` | ACAS Xu benchmark guide |
| `examples/VNN-COMP/README.md` | VNN-COMP benchmark guide |

---

## Known Limitations

1. **Falsification assumes hyperbox inputs** — Random sampling and PGD projection operate on axis-aligned bounds `[lb, ub]`. For polytope input sets (Star with non-trivial constraints), counterexamples outside the true input region may be tested, or valid counterexamples may be missed.

2. **Upsample only supports nearest-neighbor** — Bilinear and bicubic interpolation are not implemented. Only integer scale factors work.

3. **OnnxResize requires forward probing** — Scale factors are detected by running a dummy forward pass because not all ONNX attributes are stored.

4. **Hexatope/Octatope MCF solver** — Some edge cases in `get_range()` and `optimize_linear()` with minimum cost flow may return None. Tests for these cases are skipped.

5. **Conv2D requires image-aware sets** — Cannot apply Conv2D to a flat Star or Box; must use ImageStar or ImageZono with known spatial dimensions.

6. **Sigmoid/Tanh have no exact Star method** — Only approximation is available. A warning is emitted if `method='exact'` is requested.

---

## To-Do

### Missing Layers

#### High Priority
- [ ] **Softmax** (`nn.Softmax`) — Common output layer for classification. Nonlinear; needs over-approximation strategy.
- [ ] **LayerNorm** (`nn.LayerNorm`) — Used in transformers. Involves mean/variance computation over features.
- [ ] **GELU** (`nn.GELU`) — Common in transformers. Smooth nonlinearity; needs S-curve-style relaxation.

#### Medium Priority
- [ ] **Conv3D** (`nn.Conv3d`) — 3D convolutions for volumetric data.
- [ ] **TransposedConv2D** (`nn.ConvTranspose2d`) — Deconvolutions for generative models.
- [ ] **GlobalMaxPool** (`nn.AdaptiveMaxPool2d`) — Like MaxPool but over entire spatial dimensions.
- [ ] **MaxUnpool2D** (`nn.MaxUnpool2d`) — Inverse of MaxPool.
- [ ] **GroupConv** — Grouped convolutions (used in ResNeXt, MobileNet).

#### Low Priority
- [ ] **LSTM / GRU / RNN** — Recurrent layers. Would require unrolling or custom handling.
- [ ] **HardSigmoid / HardSwish** — Piecewise linear approximations to sigmoid/swish.
- [ ] **Attention / MultiHeadAttention** — Transformer core. Complex; involves softmax + matmul.

### Missing Set Type Coverage

These are operations that work for Star/ImageStar but not yet for other set types. Listed in rough priority order:

- [ ] Conv2D on Box (build convolution matrix, apply as affine map)
- [ ] MaxPool2D on Box (interval max over pooling windows)
- [ ] AvgPool2D on Box (interval mean — placeholder exists but doesn't process)
- [ ] Concat on Zono (generator concatenation along axis)
- [ ] Slice on Zono (generator slicing)
- [ ] Split on Zono (generator splitting)
- [ ] Residual Add on Zono (Minkowski sum of two Zonos from shared input)
- [ ] LeakyReLU on Hexatope/Octatope
- [ ] Sigmoid/Tanh on Hexatope/Octatope

### Falsification Improvements

- [ ] **Polytope input sampling** — Support non-hyperbox input sets via hit-and-run sampling or LP-based projection for the falsification module.
- [ ] **Gradient-free methods** — CMA-ES or other derivative-free optimizers as alternatives to PGD for networks with zero-gradient activations (e.g., Sign).

### Infrastructure

- [ ] **GitHub Actions CI/CD** — Automated test runs on push/PR.
- [ ] **Gurobi LP solver backend** — 10-100x faster than open-source solvers. Free for academics.
- [ ] **MNIST tutorial notebooks** — Verify/update to reflect current API.

### Documentation

- [ ] **CLAUDE.md layer table** — Update to match current state (some entries still show "In progress" for completed layers).
- [ ] **Benchmark suite** — Build a regression benchmark that measures correctness + tightness + speed (see `benchmarks/README.md`).
