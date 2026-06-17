# Layer-type soundness coverage (VNN-COMP corpus)

_Internal reference. Backs the "sound reach" claims in the benchmark
support doc (`examples/Submission/VNN_COMP2026/BENCHMARK_SUPPORT.md`).
Tests: `tests/soundness/test_layer_soundness_matrix.py`._

Inventory: **55 distinct ONNX op types across 605 unique models** in the
2026 benchmark corpus.

Soundness is a per-operator property: if every operator's reach output
encloses every true output, and the composition/join logic is sound,
then any network built from them is sound. So each op is validated on
small, tractable instances rather than whole models.

Ground truth: **onnxruntime** on single-node ONNX models (independent of
both n2v and onnx2torch), exercised through the real load+reach
pipeline; **torch** for the set-set composition primitives. Every
supported op is checked on **every set-type path it implements**
(Star/ImageStar, Zono/ImageZono, Box) across edge-case input regimes.

## A. Supported & soundness-validated

| Op(s) | Kind | Soundness test |
|---|---|---|
| Relu, LeakyRelu, Sigmoid, Tanh | activation (relaxation) | matrix + dedicated |
| Sign | activation (relaxation) | matrix + test_soundness_sign |
| Softmax | activation (relaxation) | matrix (star+box) |
| MaxPool | nonlinear pooling | matrix + test_soundness_maxpool2d |
| AveragePool, GlobalAveragePool | affine pooling | matrix + dedicated |
| Conv | affine | matrix + test_soundness_conv2d/1d |
| ConvTranspose | affine | matrix + test_conv_transpose2d |
| Gemm/Linear, MatMul | affine | matrix + test_soundness_linear |
| BatchNormalization | affine | matrix + test_soundness_batchnorm |
| Neg | affine | matrix |
| ReduceMean, ReduceSum | affine reduction | matrix + test_soundness_reduce |
| Round, Floor, Ceil | step relaxation | matrix |
| Pow (x^p, any constant non-negative integer p) | power relaxation | matrix (p=2,3,4,5; even=convex, odd=monotonic) |
| Sin, Cos | periodic bounded relaxation | matrix (arc tangent/secant + extremum box) |
| Resize/Upsample | affine | test_soundness_upsample |
| Add, Sub, Mul, Div (by constant) | affine | matrix (both operand orders) |
| Add, Sub, Mul, Div (two sets) | composition join | test_soundness_residual_add, test_soundness_mul_div_concat |
| Concat | structural / join | test_soundness_concat, test_soundness_mul_div_concat |
| Flatten, Reshape | structural (exact) | test_soundness_flatten/reshape |
| Transpose, Gather, Slice, Split, Squeeze, Unsqueeze | structural (exact) | test_shape_ops_onnx_oracle (entry-wise), matrix (Transpose/Pad spatial) |
| Pad | structural (exact) | matrix + test_soundness_pad |
| Cast, Dropout, Constant | identity / const-fold | test_onnx_cast, test_dropout |

Exact structural ops are entry-pinned against onnxruntime in
`test_shape_ops_onnx_oracle.py` (exactness ⇒ soundness with zero
over-approximation).

## B. Unsupported (support gaps — not soundness-testable)

These appear in the corpus but have no reach handler; they block only
the already-flagged hard benchmarks. Each is a future support item, not
a soundness gap.

| Op(s) | Blocks | Note |
|---|---|---|
| Erf, Sqrt, Exp | smart_turn | nonlinear (`Pow`, `Sin`, `Cos` now supported, 2026-06-13) |
| Clip, Max, Min, Expand, ArgMax, ScatterND, Where, Equal, Range, ConstantOfShape, Shape, dynamic Squeeze | cctsdb_yolo (perf-bound), ml4acopf, vit | control/shape ops |
| QuantizeLinear, DequantizeLinear | smart_turn | quantization (I-38) |

## Bugs found & fixed during this validation pass

1. **OnnxFunction-wrapped Tanh/Sigmoid** unhandled — a bare ONNX
   Tanh/Sigmoid node converts to the generic `OnnxFunction` wrapper,
   which the dispatcher only routed for Sign. Now routed by inspecting
   the wrapped function. (`dispatcher.py`)
2. **AvgPool ImageZono** crashed on list-typed kernel/stride/padding
   from onnx2torch (`int + list`). Normalized. (`avgpool2d_reach.py`)
3. **Mirrored `Sub(c, x)`** only handled `Star`; now covers
   Zono/Box/ImageStar/ImageZono. (`reach.py`)
4. **Dynamic Pad** (`OnnxPadDynamic`, opset-13 pads-as-input) silently
   applied zero padding → wrong (unpadded) output. Now resolves the
   pads from the node input and fails loud if unresolvable.
   (`reach.py`, `pad_reach.py`)

## Op support added

- **Pow** (`x^p`, constant non-negative integer `p`) — 2026-06-13.
  Sound per-neuron relaxation: even `p` convex (2 tangent lower + secant
  upper); odd `p` monotonic (convex/concave by sign, box fallback for
  sign-spanning); Zono/Box interval enclosure. New `pow_reach.py`,
  intercepted as `OnnxPow` in the graph executor (exponent read from
  `node.args[1]`). Unblocks nn4sys (x³) and the `Pow` part of
  ml4acopf/collins. `base**x` and non-integer exponents raise loudly.
- **Sin / Cos** — 2026-06-13. Sound periodic relaxation: exact range
  over `[l,u]` via endpoints + interior critical points (`±1` extrema);
  tangent/secant when the interval is within a monotonic single-curvature
  arc (`f'' = -f` for both), interval box otherwise; Zono/Box interval
  enclosure. New `trig_reach.py`, detected as `OnnxFunction` wrapping
  `torch.sin`/`torch.cos`. Unblocks the full ml4acopf models
  (14/118_ieee verified end-to-end).
