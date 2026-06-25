# CSV output schemas (FlowConformal paper experiments)

Authoritative schema reference for every CSV produced (or consumed) by
the FlowConformal paper-experiments pipeline. Figure / table generation
scripts depend on these schemas — do not rename columns or change
dtypes without updating this file and the consumers.

All paths are **relative to the n2v repo root**
(`/path/to/n2v`) unless stated otherwise.

## Conventions

- `verdict` is one of `UNSAT`, `SAT`, `UNKNOWN`, `TIMEOUT`, `ERROR`,
  `SKIPPED`, `NOT_APPLICABLE`. Sound-verifier CSVs use lowercase
  (`unsat`/`sat`/`unknown`/`timeout`); our pipeline uppercases.
- Wall-clock times are seconds (`*_s`). Empty string ("") indicates the
  field was not measured for that row (e.g. for `SKIPPED` rows).
- `coverage` / `coverage_empirical` are fractions in `[0, 1]` measured
  on a held-out test set.
- `epsilon_total` / `delta_total` are joint conformal-scenario certificate
  parameters. Smaller `epsilon_total` and larger `delta_total` are
  tighter.
- VNN-LIB file basenames (e.g. `prop_1.vnnlib`) and ONNX file basenames
  (e.g. `ACASXU_run2a_1_1_batch_2000.onnx`) are the canonical join keys
  for instance-level comparisons.
- One row per (instance, seed) for every per-instance CSV.

---

## 1. Experiment 1 — VNN-COMP subset (sound-verifier comparison)

### 1.1 `exp1_<benchmark>_ours.csv`

Path: `examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs/`

Active benchmarks (9): `acasxu_2023`, `collins_rul_cnn_2022`,
`dist_shift_2023`, `linearizenn_2024`, `tllverify_2023`,
`metaroom_2023`, `vit_2023`, `tinyimagenet_2024`, `cifar100_2024`.
(Two earlier benchmarks — `lsnc_relu` and `relusplitter` — were
dropped from the paper.)

| Column | Description |
|---|---|
| `benchmark` | benchmark name |
| `onnx_file` | ONNX basename |
| `vnnlib_file` | VNN-LIB basename |
| `verdict` | UNSAT / SAT / UNKNOWN / TIMEOUT / ERROR / SKIPPED |
| `wall_s` | end-to-end wall-clock |
| `train_s` | flow training wall-clock |
| `verify_s` | verification wall-clock |
| `vnncomp_timeout_s` | per-instance shell timeout from VNN-COMP `instances.csv` |
| `coverage` | empirical coverage on held-out test set |
| `q` | conformal threshold |
| `epsilon_total` | joint epsilon |
| `delta_total` | joint confidence |
| `amls_bounded_eps_2_upper` | AMLS-bounded ε₂ upper bound; "" when not run |
| `amls_bounded_detected_unsafe` | bool flag — True if a sample landed in U |
| `amls_levels_used` | adaptive AMLS levels actually run |
| `cex_x` | JSON `list[float]` counterexample input (SAT only) |
| `cex_y` | JSON counterexample output (SAT only) |
| `error` | error message when verdict ∈ {ERROR, SKIPPED, TIMEOUT} |
| `timestamp` | ISO-8601 UTC |

**Rows:** one per instance. ACAS Xu has 186; collins_rul_cnn 63;
dist_shift 73; linearizenn 61; tllverify 33; metaroom 100;
vit_2023, tinyimagenet_2024, cifar100_2024 each 200. (Exp 2 also
runs the latter three benchmarks at full 201; see §2.)

### 1.2 `exp1_<benchmark>_hashemi_clipping.csv`

| Column | Description |
|---|---|
| `benchmark`, `onnx_file`, `vnnlib_file`, `verdict`, `wall_s`, `vnncomp_timeout_s` | as in §1.1 |
| `m` | total Hashemi sample budget (default 8000) |
| `ell` | order-statistic index (default `m-1`) |
| `epsilon` | DKW miscoverage parameter (default 1e-3) |
| `coverage` | empirical coverage from `pbox.coverage` |
| `coverage_empirical` | held-out test-set coverage |
| `coverage_n_test` | held-out test-set size |
| `confidence` | DKW confidence (default 1 − ε) |
| `cex_x`, `cex_y`, `error`, `timestamp` | as in §1.1 |

### 1.3 `exp1_<benchmark>_hashemi_clipping_pca.csv`

Same as §1.2 plus `pca_components` (default 32).

### 1.4 `exp1_<benchmark>_saver.csv`

Sample-based DKW (Sivaramakrishnan et al., HSCC 2025). Applies to every
benchmark (sample-based, spec-shape-agnostic).

| Column | Description |
|---|---|
| `benchmark`, `verdict`, `wall_s`, `error`, `timestamp` | as elsewhere |
| `instance` | `<onnx_basename>+<vnnlib_basename>` |
| `timeout_s` | per-row budget |
| `beta` | DKW confidence (default 1e-3) |
| `dkw_epsilon` | DKW CDF tolerance (default 1e-2) |
| `delta` | allowed Pr[unsafe] (default 1e-3) |
| `n_samples` | DKW-bound sample count |
| `k_disjuncts` | # disjuncts in the unsafe-region union |
| `beta_per` | per-disjunct confidence (= β/k) |
| `delta_per` | per-disjunct delta (= δ/k) |
| `n_certified_disjuncts` | disjuncts whose Pr[unsafe] upper bound ≤ δ_per |
| `worst_prob_unsafe` | worst per-disjunct upper bound |
| `union_upper_bound_unsafe` | union-bound aggregate |
| `n_unsafe_samples` | samples that fell into any unsafe disjunct |

### 1.5 `exp1_<benchmark>_probstar.csv`

ProbStar / StarV (Tran et al.). Returns `NOT_APPLICABLE` whenever
StarV's `load_onnx_network` rejects the network.

| Column | Description |
|---|---|
| `benchmark`, `onnx_file`, `vnnlib_file`, `verdict`, `wall_s`, `vnncomp_timeout_s`, `error`, `timestamp` | as in §1.1 |
| `p_filter` | probability filter for ProbStar reach (0.0 = exact) |
| `lp_solver` | `gurobi` / `glpk` |
| `p_min` | lower bound on Pr[unsafe] |
| `p_max` | upper bound on Pr[unsafe] |
| `threshold` | Pr[unsafe] threshold above which to declare SAT (default 1e-3) |
| `n_disjuncts` | # disjuncts in the unsafe region |

### 1.6 `ground_truth.csv`

Path: `examples/FlowConformal/experiments/exp1_vnncomp_subset/ground_truth.csv`

Generated by `examples/FlowConformal/experiments/build_ground_truth.py`
from the §6 raw VNN-COMP CSVs.

**Consensus rule (SAT-wins):**
- If any tool reports `sat` → `ground_truth = 'sat'`. Tools that
  reported `unsat` for the same instance go in `dissenting_tools`
  (suspect soundness violations).
- Else if at least one tool reports `unsat` (and no SAT) →
  `ground_truth = 'unsat'`.
- Else (all tools timed out / errored / unknown) → `ground_truth = 'unknown'`.

| Column | Type | Description |
|---|---|---|
| `benchmark` | str | local benchmark name |
| `onnx_file` | str | ONNX basename |
| `vnnlib_file` | str | VNN-LIB basename |
| `ground_truth` | str | `sat` / `unsat` / `unknown` |
| `n_sat`, `n_unsat`, `n_timeout`, `n_unknown`, `n_error` | int | per-verdict tool count |
| `dissenting_tools` | str | comma-separated; only populated when GT=`sat` |
| `source_tools` | str | comma-separated tools that returned a definitive verdict |

---

## 2. Experiment 2 — Probabilistic-scale comparison

Path: `examples/FlowConformal/experiments/exp2_prob_scale/outputs/`

The Exp 2 benchmarks (`vit_2023`, `tinyimagenet_2024`, `cifar100_2024`)
are the high-output-dim subset of Exp 1. Both Exp 1 and Exp 2 runs
exist for these benchmarks; the headline paper table joins on the
Exp 2 outputs (which run all 200 instances vs. Exp 1's 100-instance
default).

### 2.1 `exp2_<benchmark>_ours.csv`

| Column | Description |
|---|---|
| `benchmark` | benchmark name |
| `instance` | benchmark-specific instance label (e.g. `pgd_2_3_16.onnx+pgd_2_3_16_2446.vnnlib`) |
| `verdict`, `wall_s`, `train_s`, `verify_s` | as in §1.1 |
| `timeout_s` | per-row budget |
| `coverage`, `q`, `epsilon_total`, `delta_total` | as in §1.1 |
| `amls_bounded_eps_2_upper`, `amls_bounded_detected_unsafe`, `amls_levels_used` | as in §1.1 |
| `cex_x`, `cex_y`, `error`, `timestamp` | as in §1.1 |

**Rows:** one per (benchmark, instance), K=1.

### 2.2 `exp2_<benchmark>_hashemi_clipping.csv`

Same as §1.2 but with the Exp 2 `instance` key (single column) instead
of split `onnx_file` / `vnnlib_file`.

### 2.3 `exp2_<benchmark>_hashemi_clipping_pca.csv`

Same as §1.3 plus `training_samples` (training-set size for the
clipping_block surrogate; default `m // 2`).

### 2.4 `exp2_<benchmark>_saver.csv`

Same schema as §1.4 (with `instance` key per Exp 2 convention).

### 2.5 `exp2_<benchmark>_probstar.csv`

Same schema as §1.5 (with `instance` key).

### 2.6 `exp2_<benchmark>_rs.csv`

Cohen et al. 2019 randomized smoothing. Classifier-only.

| Column | Description |
|---|---|
| `benchmark`, `instance`, `verdict`, `wall_s`, `timeout_s`, `error`, `timestamp` | as elsewhere |
| `sigma` | smoothing noise σ |
| `n0` | initial-estimate sample count |
| `n_certify` | certification sample count |
| `alpha` | certification confidence (one-sided) |
| `pred_class` | predicted class on the original input |
| `true_class` | ground-truth label from the spec |
| `l2_radius` | certified ℓ₂ radius around the input |
| `eps_linf_threshold_l2` | ℓ∞→ℓ₂ conversion of the spec ε for direct comparison |

### 2.7 `exp2_<benchmark>_alpha_beta_crown.csv`

Local αβ-CROWN re-run (the production paper table merges these with
the VNN-COMP-published numbers via §6).

| Column | Description |
|---|---|
| `benchmark`, `onnx_file`, `vnnlib_file`, `verdict`, `wall_s`, `timeout_s`, `cex_x`, `cex_y`, `error`, `timestamp` | as elsewhere |

### 2.8 `ground_truth.csv` (Exp 2 path)

Same schema as §1.6, separate file at
`examples/FlowConformal/experiments/exp2_prob_scale/ground_truth.csv`.

---

## 3. Experiment 3 — Synthetic volume-comparison

Path: `examples/FlowConformal/experiments/exp3_synthetic/outputs/`

Falsifier is OFF in Exp 3. Active benchmarks (7):

- `2d_banana` (`RotatedBananaNet` on `[0, 1]²`, exact area = 0.295)
- `3d_banana` (`ThreeBlobClassifier3D` on `[-1, 1]³`, cached MC vol = 213.73)
- `synth_2d`, `synth_3d`, `synth_5d`, `synth_10d`, `synth_20d`
  (identity-activation `OneLipschitzNet`, closed-form `vol = |det(W_total)| × prod(ub-lb)`)

Three sample-budget configs per (benchmark, method): `small` (m=1k),
`default` (m=8k), `large` (m=16k). Five seeds each. Filename suffix
encodes config: `_<config>` (omitted if running the canonical 5-seed
single-config sweep).

### 3.1 `exp3_<benchmark>_flow_<spec_type>_ours[_<config>].csv`

Produced by `exp3_run_ours.py`. The canonical spec is `unsat` (far-away
halfspace, UNSAT by construction); a `sat` variant exists for honest-
abstention probes.

| Column | Description |
|---|---|
| `benchmark` | benchmark name |
| `score` | nonconformity-score family (`flow` is the headline; `hyperrect` / `ellipsoid` / `gmm` are closed-form alternates) |
| `spec_type` | `unsat` / `sat` |
| `seed` | seed index |
| `verdict`, `wall_s`, `train_s`, `verify_s` | as elsewhere |
| `coverage`, `q`, `epsilon_total`, `delta_total` | as elsewhere |
| `amls_bounded_eps_2_upper`, `amls_levels_used` | as elsewhere |
| `volume_estimate` | MC volume of the calibrated reach set |
| `volume_exact` | closed-form (linear synth) or cached MC (banana) reach-set volume; "" when not derivable |
| `volume_ratio` | `volume_estimate / volume_exact`; "" when `volume_exact` is "" |
| `cex_x`, `cex_y`, `error`, `timestamp` | as elsewhere |

### 3.2 `exp3_<benchmark>_<spec_type>_hashemi_clipping[_<config>].csv`

Hashemi-clipping pbox volume comparison.

| Column | Description |
|---|---|
| `benchmark`, `spec_type`, `seed`, `verdict`, `wall_s`, `error`, `timestamp` | as elsewhere |
| `m`, `ell`, `epsilon` | as in §1.2 |
| `coverage`, `confidence` | from `pbox` |
| `volume_estimate_closedform` | `prod(pbox.ub - pbox.lb)` — exact for axis-aligned boxes |
| `volume_estimate_mc` | MC sanity-check: sample uniformly from a 1.1× padded bbox, count fraction inside pbox, multiply by padded-bbox volume — must match closed form modulo MC noise |
| `volume_exact` | as in §3.1 |
| `volume_ratio_closedform` | `volume_estimate_closedform / volume_exact` |
| `volume_ratio_mc` | `volume_estimate_mc / volume_exact` |
| `cex_x`, `cex_y` | "" (Exp 3 has no falsifier) |

### 3.3 `exp3_<benchmark>_<spec_type>_starset_approx[_<config>].csv`

Sound deterministic baseline using
`n2v.nn.reach.reach_pytorch_model(model, Star, method='approx')`.
Reports the bbox volume of the over-approximate output Star.

| Column | Description |
|---|---|
| `benchmark`, `spec_type`, `seed`, `verdict`, `wall_s`, `error`, `timestamp` | as elsewhere |
| `n_output_sets` | # output Stars / Boxes returned by approx reach |
| `starset_bbox_vol` | bbox volume of the union of output sets |
| `volume_exact` | as in §3.1 |
| `volume_ratio` | `starset_bbox_vol / volume_exact` |
| `cex_x`, `cex_y` | "" |

---

## 4. Experiment 4 — Scaling vs. network size

Path: `examples/FlowConformal/experiments/exp4_scaling/outputs/`

One-Lipschitz synthetic family at fixed width = 512, varying depth ∈
{2, 4, 8, 16, 24, 32, 40}. Per-method CSVs use the filename
`exp4_d<depth>_<method>.csv`.

### 4.1 `exp4_d<depth>_ours.csv`

| Column | Description |
|---|---|
| `depth`, `width`, `n_params` | architecture descriptors |
| `instance_idx` | 0-indexed instance within the depth |
| `method` | always `ours` for this CSV |
| `verdict`, `wall_s`, `train_s`, `verify_s` | as elsewhere |
| `timeout_s` | per-instance budget |
| `x_0_seed` | seed of the random base input |
| `eps` | ℓ∞ perturbation radius |
| `spec_threshold` | unsafe halfspace threshold (`y_0 ≥ spec_threshold`) |
| `empirical_max` | max observed `y_0` over the perturbation MC |
| `ground_truth` | `unsat` / `sat` based on `empirical_max` vs `spec_threshold` |
| `coverage`, `q`, `epsilon_total`, `delta_total` | as elsewhere |
| `amls_bounded_eps_2_upper`, `amls_bounded_detected_unsafe`, `amls_levels_used` | as elsewhere |
| `cex_x`, `cex_y`, `error`, `timestamp` | as elsewhere |

### 4.2 `exp4_d<depth>_alpha_beta_crown.csv` / `exp4_d<depth>_neuralsat.csv`

Sound verifiers, no calibration columns.

| Column | Description |
|---|---|
| `depth`, `width`, `n_params`, `instance_idx`, `method` | as in §4.1 |
| `verdict`, `wall_s`, `timeout_s`, `x_0_seed`, `eps` | as elsewhere |
| `spec_threshold`, `empirical_max`, `ground_truth` | as in §4.1 |
| `amls_bounded_eps_2_upper`, `amls_levels_used` | "" (sound verifiers don't compute these) — kept for schema parity with §4.1 |
| `cex_x`, `cex_y`, `error`, `timestamp` | as elsewhere |

### 4.3 `exp4_d<depth>_hashemi_clipping.csv`

§4.2 schema plus `m`, `ell`, `epsilon`, `coverage`, `confidence` (as
in §1.2).

---

## 5. Verification-method ablation

Path: `examples/FlowConformal/experiments/exp_ablation/outputs/`

`ablation_shared_flow.py` calibrates the flow ONCE per instance and
runs all selected verifiers against the shared `(flow, q)` tuple,
isolating the verification-method axis.

CSV: `ablation_shared_flow_<benchmark>_<method>.csv`. One per (method)
per benchmark.

| Column | Description |
|---|---|
| `instance_idx` | 0-indexed instance from the random sample |
| `box_idx` | 0-indexed box (ACAS Xu prop_3 / prop_4 are OR-of-input-regions) |
| `onnx_file`, `vnnlib_file` | basenames |
| `ground_truth` | from the per-benchmark ground-truth CSV |
| `method` | one of {`scenario`, `amls`, `is_tilted`, `amls_bounded`, `amls_bounded_union`, `raw_mc_uniform`} |
| `verdict` | as elsewhere |
| `q`, `coverage`, `epsilon_total` | as elsewhere |
| `amls_bounded_eps_2_upper` | as elsewhere |
| `flow_train_s` | shared calibration cost (constant per instance) |
| `verify_s` | per-method verification time |
| `wall_s` | `flow_train_s + verify_s` for that (instance, method) pair |
| `error`, `timestamp` | as elsewhere |

Active benchmarks: `acasxu_2023` (186 instances), `tllverify_2023` (32).

---

## 6. Sound-verifier results (read-only, from VNN-COMP)

Path:
`~/v/other/VNNCOMP/vnncomp2025_results/<verifier>/<2025_benchmark>/results.csv`

Verifiers (folder names): `alpha_beta_crown`, `neuralsat`, `nnenum`,
`nnv`, `pyrat`, `cora`, `rover`, `sobolbox`, `marabou` (when added).

Benchmark folder names are prefixed `2025_` (e.g.
`2025_acasxu_2023`).

VNN-COMP results.csv has **no header** and uses the schema:

```
benchmark, onnx_path, vnnlib_path, prepare_time_s, verdict, runtime_s
```

| Column | Description |
|---|---|
| col 0 | benchmark short-name |
| col 1 | full ONNX path relative to vnncomp2025 repo |
| col 2 | full VNN-LIB path relative to vnncomp2025 repo |
| col 3 | prepare time (s) |
| col 4 | verdict (lowercase) |
| col 5 | run-time (s) |

**Loader rule:** join keys are `(Path(col1).name, Path(col2).name)` —
i.e. **basenames** of the ONNX and VNN-LIB files. This matches our
CSVs' `(onnx_file, vnnlib_file)` columns.

---

## 7. JOIN keys (instance-level comparisons)

| Comparison | Left | Right | Join key |
|---|---|---|---|
| Exp 1: ours vs αβ-CROWN / NeuralSAT / PyRAT / CORA | `exp1_<bench>_ours.csv` | `~/v/other/VNNCOMP/vnncomp2025_results/<verifier>/2025_<bench>/results.csv` | `(onnx_file, vnnlib_file)` basenames |
| Exp 1: ours vs Hashemi / SaVer / ProbStar | `exp1_<bench>_ours.csv` | `exp1_<bench>_<baseline>.csv` | `(onnx_file, vnnlib_file)` (split on `+` for SaVer's `instance` field) |
| Exp 1: ours vs ground truth | `exp1_<bench>_ours.csv` | `exp1_vnncomp_subset/ground_truth.csv` | `(benchmark, onnx_file, vnnlib_file)` |
| Exp 2: ours vs probabilistic baseline | `exp2_<bench>_ours.csv` | `exp2_<bench>_<baseline>.csv` | `instance` |
| Exp 2: ours vs ground truth | `exp2_<bench>_ours.csv` | `exp2_prob_scale/ground_truth.csv` | derived from `instance` (split on `+` for VNN-COMP-style benchmarks) |
| Exp 3: per-method volume aggregation | `exp3_<bench>_*_ours_<config>.csv`, `exp3_<bench>_*_hashemi_clipping_<config>.csv`, `exp3_<bench>_*_starset_approx.csv` | (no join — same `seed` × `(benchmark, config)` slot per method) | `(benchmark, seed)` |
| Ablation aggregate (shared-flow) | `ablation_shared_flow_<benchmark>_<method>.csv` | (no join — direct compare across method CSVs) | `(instance_idx, box_idx)` |

The `instance` column in baseline CSVs uses the format
`<onnx_basename>+<vnnlib_basename>` for VNN-COMP-style benchmarks
(per `_common.load_vnncomp_instances`). Splitting on the first `+`
recovers the basenames.

---

## 8. Production sweep order

1. **Sound-verifier results** — already published in VNN-COMP; just
   pulled from §6.
2. **Exp 3** synthetic volume comparison (validates framework on
   benchmarks with closed-form / cached MC ground truth).
3. **Exp 1 ours** + 4 probabilistic baselines (Hashemi-clipping,
   Hashemi-clipping-PCA, SaVer, ProbStar).
4. **Exp 2 ours** + 4 probabilistic baselines + RS on cifar100.
5. **Exp 4 scaling** (synthetic 1-Lipschitz family, ours vs αβ-CROWN
   vs NeuralSAT vs Hashemi-clipping).
6. **Verification-method ablation** (shared-flow §5).
7. **Figure / table generation** under
   `examples/FlowConformal/paper/` reads all of the above.
