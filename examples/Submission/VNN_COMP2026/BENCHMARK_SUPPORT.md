# n2v — VNN-COMP 2026 Benchmark Support

_Status snapshot: 2026-06-13. Supersedes the 2026-06-10 snapshot
(which predated the reach-layer work and reported 14/34)._

This document records which VNN-COMP 2026 benchmarks n2v can handle, and
documents the layer and specification types that are not yet supported.
The companion internal doc `docs/layer_soundness_coverage.md` records the
per-operator soundness validation that backs the "sound reach" claims.

## What "support" means

Load the ONNX model, parse the VNNLIB spec, and run **sound** set-based
reachability (`approx` Star) through every layer to completion. This is
stricter than "solved": it excludes SAT found by falsification (forward
passes only) and UNSAT from the probabilistic / conformal method (a
coverage guarantee, **not** a sound proof). Every unsupported case fails
as an **error → `unknown`**, never a silent wrong verdict — there is no
soundness *violation*.

## How this was measured

Per-benchmark scorecard sweep (`scorecard_one.py`, subprocess-isolated,
150 s reach cap): for the **first instance** of each benchmark, probe
spec-parse + model-load + degenerate reach, then cross-reference the op
inventory (55 op types / 605 models) for intra-benchmark gaps. "✓" = the
first instance works end-to-end; the Caveat/Blocker column flags
instances or models that differ. A scorecard is a one-instance snapshot,
not a full-benchmark score; timeouts are at the smoke budget and usually
indicate per-benchmark tuning, not a hard limitation.

## Summary

- **Specs: 34/34 parse** (VNNLIB 1.0, 2.0, relational, nonlinear).
- **Models: 33/34 load** (only smart_turn fails — quantized transformer,
  I-38). vggnet16 downloaded + loads (opset shim); vit loads (shape-fold
  shim).
- **Sound reach: ~29 benchmarks clear** (incl. both relational). The
  remaining are either perf/tuning (collins, traffic_signs, ml4acopf
  300_ieee f64) or **fundamental frontiers that load+forward but whose
  sound reach needs new machinery** (vggnet16 scale, vit attention,
  cctsdb discrete control flow).

## Tier 1 — fully working (spec ✓, load ✓, sound reach ✓) — 27

| Benchmark | Note |
|---|---|
| acasxu_2023 | |
| adaptive_cruise_control_non_linear_2026 | nonlinear spec; model is a plain MLP |
| cersyve | |
| cgan_2023 | ConvTranspose |
| cgan2026 | ConvTranspose |
| challenging_certified_training_2026 | |
| cifar100_2024 | |
| collins_rul_cnn_2022 | |
| cora_2024 | |
| dist_shift_2023 | has `Shape`, but it constant-folds |
| linearizenn_2024 | |
| lsnc_relu | |
| malbeware | |
| metaroom_2023 | |
| ml4acopf_2024 | full non-linear 14/118_ieee reach (Pow + Sin/Cos); see 300_ieee caveat below |
| nn4sys | Pow (x³) |
| relusplitter | |
| relusplitter_2026 | |
| safenlp_2024 | |
| sat_relu | |
| soundnessbench | |
| soundnessbench_2026 | |
| test | |
| tinyimagenet_2024 | |
| tllverifybench_2023 | |
| traffic_signs_recognition_2023 | Sign+Softmax; full-eps *sound verification* is perf-bound (>900 s) |
| yolo_2023 | |

## Tier 1b — relational (two-network) — supported — 2

| Benchmark | Verdict path | Note |
|---|---|---|
| isomorphic_acasxu_2026 | sound joint reach | UNSAT (instance specs use a contradictory `and` — upstream and/or bug — so the unsafe region is empty) |
| monotonic_acasxu_2026 | falsify (SAT) / sound joint reach | SAT with a genuine coupled counterexample |

Relational engine: self-composition over the coupled joint input with a
prefix-aligned predicate join (`n2v/nn/relational.py`), wired into the
runner. Sound (joint reach encloses the true `[f(x_f); g(x_g)]`). Note:
the product relaxation relaxes f and g independently, so it cannot prove
properties that need a *tight* cross-network difference (e.g. exact
equivalence) — those stay UNKNOWN unless falsified. A precision limit,
not a soundness one.

## Tier 2 — partial: first instance works, some models do not — 2

| Benchmark | Blocker on the bad models |
|---|---|
| ml4acopf_2024 | `14/118_ieee` (linear and full) work. **`300_ieee` is float32-ill-conditioned** (squares large values; ort-f32 itself is ~1.6e-2 from the true f64 answer — *not* a conversion bug, I-37). n2v's **sound reach is float64 and accurate** (matches f64 to 2e-11); only the falsification lane should validate counterexamples in f64. |
| collins_aerospace_benchmark | first instance runs but **perf-bound** (>150 s); Pow is supported |

## Tier 3 — load OK, reach is a fundamental frontier (not an op gap) — 4

| Benchmark | Load | Reach | Blocker |
|---|---|---|---|
| smart_turn_multimodal_2026 | ✗ | — | **extended-track, future work** — fully QDQ-quantized transformer (199 DequantizeLinear + 119 activation-path QuantizeLinear + Erf/Sqrt/Exp); needs converters + quantize round-relaxation + nonlinear ops + transformer-scale perf. Not currently supported (I-38). |
| vggnet16_2022 | ✓ | scale | model downloaded (setup.sh) + **loads** (opset-8→13 upgrade shim; forward matches ort to 9.5e-7). All ops supported (Conv/Relu/MaxPool/Gemm/Flatten/Dropout). Full sound reach is **scalability-bound** — 224×224×3 = 150k-dim input + millions of ReLU neurons; needs the perf phase (IBP-bound ReLU, sparse input sets), not an op gap. |
| cctsdb_yolo_2023 | ✓ | frontier | **loads + forwards** (so falsification can find SAT). Sound reach blocked by the YOLO detection head's **discrete / data-dependent ops** (`ArgMax`, `ScatterND`, `Where`, `Equal`) — beyond sound set-based reach — plus piecewise-linear `Clip`/`Max`/`Min`/`Expand`. Frontier, not an op gap. |
| vit_2023 | ✓ | frontier | **loads** now (shape-fold shim folds away `Shape`/`ConstantOfShape`; forward matches ort 9.5e-7); affine ops supported. Sound reach **infeasible with star sets** — attention has 6 `set@set` (bilinear) MatMuls; a McCormick relaxation explodes to ~80k+ predicates. Transformer-verification frontier, not an op gap. |

## Remaining work to "everything end-to-end"

Every benchmark that *can* load now loads (33/34; only smart_turn is
load-blocked). The remaining gaps split into:

1. **Perf / tuning** — fit the supported-but-slow benchmarks into the
   competition budget: collins_aerospace, traffic_signs (full-eps sound
   verification), and the f64 falsification validation for ml4acopf
   300_ieee. The natural next phase.
2. **Fundamental reach frontiers** (load + forward, but sound reach needs
   machinery the star engine doesn't have):
   - **vggnet16** — scale (150k-dim input × millions of ReLU neurons);
     needs IBP-bound ReLU + sparse/zonotope input sets.
   - **vit** — bilinear self-attention (`set@set` MatMul); needs a
     dedicated attention relaxation.
   - **cctsdb_yolo** — discrete detection-head control flow
     (`ArgMax`/`ScatterND`/`Where`/`Equal`).
3. **Extended track** — smart_turn (quantized transformer, I-38).

Note: I-37 (ml4acopf 300_ieee) is **not** a conversion bug — float32
ill-conditioning; the float64 sound reach handles it correctly.

## Soundness note

The tool also has a **probabilistic** (conformal / flow) lane and a
**falsification** lane (random + PGD). Results from those carry a
coverage guarantee or a counterexample, **not** a sound proof, and must
be reported/scored accordingly. The sound vs. probabilistic choice per
benchmark is a tuning decision deferred to a later pass. This document is
exclusively about the **sound reach** lane.
