# VNN-COMP 2023 ViT benchmark — sound star-set verification

Sound reachability verification of the VNN-COMP 2023 ViT models
(`shizhouxing/ViT_vnncomp2023`) in n2v. CIFAR-10, L∞ ε = 1/255 (normalized),
property = argmax preservation (robustness). Two models: `pgd_2_3_16`
(depth 2, 5 tokens) and `ibp_3_3_8` (depth 3, 17 tokens); 100 instances each.

The design and soundness arguments are in
[`docs/theory/sound-vit-reach.md`](../../docs/theory/sound-vit-reach.md).

## Why a dedicated driver

n2v loads ONNX through onnx2torch, which cannot ingest these opset-9 models
(it fails on `Slice` v1, and after an opset upgrade on the per-token
BatchNorm). So instead of the fx-graph path we:

1. Reimplement the exported ONNX as a clean PyTorch module (`model.py`,
   `ViT_BN`) and load weights from the ONNX initializers. Parity with
   onnxruntime is ~1e-6 (`tests/integration/test_vit_model_parity.py`).
2. Sequence n2v's sound reach ops over that computation graph directly
   (`reach.py`, `ViTReacher`), bypassing the tracer.

Every step is a sound over-approximation, so the final logit set encloses all
reachable outputs; an instance is **verified** iff every margin
`Y_label - Y_i` has a positive lower bound.

## Files

- `model.py` — `ViT_BN` + `build_model(name, onnx_path)` (ONNX weight loader).
- `reach.py` — `ViTReacher`: conv → +cls/+pos → blocks(BN, attention, residual,
  MLP/ReLU, residual) → mean-pool → BN → head; margins. Attention uses the
  sound `bilinear_matmul` (QKᵀ, A·V) and `softmax_attention` reach ops; the
  multi-head reshape is exact row-permutation so all heads run in single
  batched matmul calls.
- `run_benchmark.py` — sweep instances, tally verified/unknown, write CSV.
- `onnx/` — vendored ONNX models (620 KB). `instances/` — vendored compact
  instance sets (100/model, ~0.4 MB each).

## Modes / precision

- **`mode='concretize'`** (default): the attention bilinears and softmax are
  box-lifted (matching CROWN's concretisation of the attention weights). Sound;
  ~IBP precision. `bounds='estimate'` keeps the forward pass pure-numpy fast.
- Symbolic predicate-preserving attention (design Slices 1–2) is the precision
  upgrade and is in progress.

Note: at full ε = 1/255, IBP-class precision certifies ≈0 (the sound reference
is α,β-CROWN at 79/200, which uses branch-and-bound); precision work targets
closing that gap within the sound star framework.

## Run

```bash
python benchmarks/vit_vnncomp2023/run_benchmark.py --num-instances 100 --eps 0.00392
```
