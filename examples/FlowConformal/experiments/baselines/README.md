# Probabilistic baselines for Exp 1 / Exp 2

Thin runner scripts that call external probabilistic-verification tools
on our Exp 1 (VNN-COMP subset) and Exp 2 (large-network) benchmarks.
**These tools are NOT integrated into the n2v interface** — they run
as standalone scripts that import from `~/v/other/` via `sys.path`
injection at runtime.

## Layout

```
baselines/
├── README.md                       # this file
├── _common.py                      # shared loaders, CSV schema, spec helpers
├── run_hashemi_clipping.py         # Hashemi clipping-block surrogate
├── run_rs.py                       # Cohen et al. randomized smoothing
├── run_saver.py                    # Sivaramakrishnan et al., HSCC 2025 (DKW path)
├── run_probstar.py                 # Tran et al. ProbStar/StarV
└── outputs/                        # per-runner CSVs
```

## Common CLI

Every runner accepts:

- `--benchmark <name>` — required. One of:
  - VNN-COMP 2025: `acasxu_2023`, `collins_rul_cnn_2022`, `cora_2024`,
    `dist_shift_2023`, `linearizenn_2024`, `malbeware`,
    `metaroom_2023`, `ml4acopf_2024`, `safenlp_2024`, `tllverify_2023`,
    `tinyimagenet_2024`, `vit_2023`, `yolo_2023`
  - Image classification: `cifar10_resnet110`
- `--smoke` — run 2 instances only (overrides `--instances`).
- `--instances N` — run first N instances (default 10).
- `--output-csv PATH` — override default output path.
- `--seed K` — master seed (default 0).

Default output is
`outputs/baseline_<name>_<benchmark>[_smoke].csv`.

## Runner-specific notes

### `run_hashemi_clipping.py`

Wraps `n2v.probabilistic.conformal_reach(surrogate='clipping_block')` (the
Hashemi pipeline already in n2v, NOT part of AMLS). The surrogate
fits a calibrated bounding box; we then check whether the box is
disjoint from the spec's unsafe halfspace using interval arithmetic.

- No external imports beyond n2v itself.
- Verdicts: `UNSAT` (box disjoint), `SAT` (sample falsifier hits unsafe
  halfspace), `UNKNOWN` (spec not ruled out by box bound), `ERROR`
  (load/run failure).
- Extra CSV fields: `m`, `ell`, `epsilon`, `coverage`, `confidence`.

### `run_rs.py` (Randomized Smoothing)

Imports `Smooth` from `~/v/other/smoothing/code/core.py`.

- **Applicable only** to classification benchmarks with a
  `make_classification_robustness_spec`-style spec
  (`cifar10_resnet110`, `cifar100_2024`). VNN-COMP regression benchmarks
  (collins_rul_cnn, cora, etc.) emit `verdict=NOT_APPLICABLE`.
- L∞ ε is converted to an L2 threshold `sqrt(d) * eps` for the
  certified radius comparison. This is a **conservative, sufficient
  condition** for L∞ robustness: passing it implies UNSAT, failing it
  may still be UNSAT (RS just can't prove it).
- **Pretrained weights required for `cifar10_resnet110`**: download
  `~/v/other/smoothing/models/cifar10/resnet110/noise_<sigma>/checkpoint.pth.tar`
  from the Cohen et al. Drive link (URL printed by the loader on
  failure). Without weights, the runner emits a TODO and exits with
  code 0 (graceful skip).
- CPU-only mode is supported via a re-implementation of `Smooth.certify`
  (the upstream code hard-codes `device='cuda'` for noise sampling).
- Extra CSV fields: `sigma`, `n0`, `n_certify`, `alpha`, `pred_class`,
  `true_class`, `l2_radius`, `eps_linf_threshold_l2`.

### `run_saver.py` (SaVer-Toolbox)

Imports `usingDKW` and `polytope` SDF from
`~/v/other/SaVer-Toolbox/SaVer_Toolbox/`.

- Spec is converted to a single polytope SDF. Single-disjunct specs
  (`HalfSpace`) are encoded directly. **OR-of-ANDs and AND-of-OR specs
  are not directly representable** as a single SDF and emit
  `NOT_APPLICABLE`.
- Multi-input-region specs (e.g. ACAS Xu prop_6) emit `NOT_APPLICABLE`.
- DKW chooses sample count from `(beta, dkw_epsilon)`. Defaults
  `beta=1e-3`, `dkw_epsilon=1e-2` give ≈ 38 005 samples.
- The DKW path itself does not require Gurobi; **the polytope SDF eval
  uses cvxpy when a sample falls outside the unsafe set**, defaulting
  to ECOS/SCS solvers (free). If you want Gurobi for cvxpy, set
  `CVXPY_DEFAULT_SOLVER=GUROBI` in the environment.
- The **Scenario** path of SaVer (not used here) does require Gurobi.
- Extra CSV fields: `beta`, `dkw_epsilon`, `delta`, `n_samples`,
  `empirical_cdf_at_0`, `certified_lb`, `n_unsafe_samples`.

### `run_probstar.py` (StarV)

Imports `Star`, `ProbStar`, `quantiVerifyBFS` from `~/v/other/StarV/`.

- **Restricted to piecewise-linear feedforward MLPs** (Linear+ReLU).
  The runner walks the network's children; on any unsupported layer
  (Conv2D, BatchNorm, MultiheadAttention, GeLU, etc.) it emits
  `verdict=NOT_APPLICABLE`. This matches ProbStar's intrinsic limits:
  attention-based ViTs and YOLOv5-style detectors are out of scope.
- **Gurobi is required** for `lp_solver='gurobi'` (default). Without a
  Gurobi license, pass `--lp-solver glpk` to fall back to GLPK (slower
  and may give different bounds). Without any working LP solver, every
  instance emits `verdict=ERROR`.
- ProbStar input is built as a truncated Gaussian centered at the
  midpoint of the box (mirrors StarV's tutorial pattern). The
  `--gauss-alpha` flag controls truncation aggressiveness:
  `sigma = (mu - lb) / alpha`.
- Verdict: `UNSAT` if `p_max < threshold`, `SAT` if `p_min > threshold`,
  else `UNKNOWN`. Threshold default `1e-3` matches our `α`.
- Extra CSV fields: `p_filter`, `lp_solver`, `p_min`, `p_max`,
  `threshold`.

## Usage examples

Smoke-test every runner on a small VNN-COMP benchmark:

```bash
cd /path/to/n2v
for runner in hashemi_naive hashemi_clipping rs saver probstar; do
    echo "=== $runner ==="
    timeout 120 python -u -m \
        examples.FlowConformal.experiments.baselines.run_$runner \
        --benchmark acasxu_2023 --smoke
done
```

Full Exp 2 sweep with Hashemi-clip (cheapest baseline):

```bash
python -u -m \
    examples.FlowConformal.experiments.baselines.run_hashemi_clipping \
    --benchmark cifar10_resnet110 --instances 100
```

## CSV schema

Every CSV starts with these columns:

| benchmark | instance | baseline | verdict | wall_s | error |

Then runner-specific columns are appended. See per-runner docstrings.

## Verdict definitions

| Verdict          | Meaning                                                |
|------------------|--------------------------------------------------------|
| `UNSAT`          | proved (probabilistically) safe                        |
| `SAT`            | counterexample / unsafe with prob >= 1-α               |
| `UNKNOWN`        | tool ran but cannot certify either direction at α      |
| `ERROR`          | tool crashed / pre-conditions failed                   |
| `NOT_APPLICABLE` | tool fundamentally does not support this instance     |

## TODOs / blockers

- **RS / cifar10_resnet110**: download Cohen-et-al pretrained weights
  (see `run_rs.py` docstring).
- **ProbStar**: requires Gurobi for non-trivial runs. Even after
  installing `gurobipy`, a Gurobi WLS license must be in
  `$HOME/gurobi.lic` or set via `GRB_LICENSE_FILE`.
- **SaVer**: cvxpy needs at least one solver (ECOS, SCS, or Gurobi).
  Install: `pip install ecos`.

## Constraints (non-goals)

- These runners are **not part of `n2v.probabilistic`**; do not import
  them from n2v internals.
- They are NOT covered by n2v's test suite — they live under
  `examples/` and are exercised via the smoke harness above.
- They should never be `git add`-ed in CI; user handles all commits.
