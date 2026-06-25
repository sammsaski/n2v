# FlowConformal — flow-matching-based probabilistic reachability

Runnable entry points for the flow-matching probabilistic-reachability
project. The library code (flow training, calibration, AMLS / scenario
/ IS spec verification, falsifier ensemble) lives in
`n2v/probabilistic/flow/`, `n2v/utils/verify_specification.py`, and
`n2v/utils/falsify.py`. This directory holds the demos, benchmarks,
paper-experiment scripts, and the paper-side figure / table
generators.

For the paper-experiment design see
[`docs/plans/2026-04-27-paper-experiments-design.md`](../../docs/plans/2026-04-27-paper-experiments-design.md).
Reproduction commands: [`REPRODUCING.md`](REPRODUCING.md). Every CSV
schema produced by this codebase: [`CSV_SCHEMAS.md`](CSV_SCHEMAS.md).

## Public library API

After the post-NeurIPS cleanup refactor the public surface is three
composable steps: ``falsify`` (optional, pre-reach SAT search), then
either the OO ``NeuralNetwork.reach(method='flow_matching')`` or the
model-agnostic free function ``flow_reach``, then
``verify_specification``. Both produce identical numbers on the same
seeds.

```python
from n2v.sets import Box
from n2v.nn import NeuralNetwork
from n2v.probabilistic import FlowReachConfig
from n2v.utils.falsify import falsify
from n2v.utils.verify_specification import (
    ProbVerifyConfig,
    verify_specification,
)

# Optional Stage 1: fast counterexample search.
sat_int, cex = falsify(model, lb=lb, ub=ub, property=spec, method='apgd',
                       seed=0, n_restarts=10, n_steps=100)
if sat_int == 0:
    # ... handle SAT (counterexample found)
    pass

# Stage 2: flow-matching probabilistic reach.
net = NeuralNetwork(model)
prob_set = net.reach(
    Box(lb, ub), method='flow_matching',
    config=FlowReachConfig(
        epsilon=0.001, m=8000, ell=7999,
        n_train=10_000, flow_epochs=5000, flow_config='base',
        seed=0,
    ),
)

# Stage 3: spec verification (UNSAT certification).
result = verify_specification(
    prob_set, spec,
    config=ProbVerifyConfig(
        method='amls_bounded',           # paper-production default
        n_samples=2000, beta=0.001,
        amls_bounded_eps_2_target=0.001,
        seed=0,
    ),
)
print(result.verdict, result.epsilon_total, result.q)
```

The same model-agnostic dispatch is also available as a free function
``n2v.probabilistic.flow_reach(model, box, config)``, which accepts any
callable ``y = model(x)`` (PyTorch ``nn.Module``, TensorFlow, JAX, ONNX
session, …). See ``conformal_reach`` for the surrogate-based variant
(returns a ``ProbabilisticBox``).

## Directory layout

```
FlowConformal/
├── networks.py                     toy networks (RotatedBananaNet, ThreeBlobClassifier3D)
├── README.md                       this file
├── REPRODUCING.md                  end-to-end reproduction steps
├── CSV_SCHEMAS.md                  every CSV column documented
│
├── benchmarks/                     small benchmarks + the ACAS Xu loader
│   ├── _common.py                  legacy-shaped result-dict shim around the new 3-stage API
│   ├── _common_analytical.py       analytical-ground-truth helpers
│   ├── demo_acasxu_single.py       single-instance ACAS Xu runner
│   ├── demo_rotated_linear.py / demo_rotated_linear_production.py
│   └── demo_identity_network.py    cube sanity check
│
├── experiments/                    paper-quality runs
│   ├── README.md                   design doc + execution order
│   ├── run_paper_sweeps.sh         single canonical launcher (--phase exp1|exp2|exp3|exp4|ablation|all)
│   ├── run_cell.sh                 VNN-COMP-style per-instance shell-timeout wrapper
│   ├── build_ground_truth.py       one-shot SAT-wins consensus generator (Exp 1 + Exp 2)
│   ├── _external_verifiers.py      αβ-CROWN / NeuralSAT / StarV subprocess wrappers
│   ├── _ground_truth_lookup.py     in-process GT helper used by smoke summaries
│   │
│   ├── baselines/                  shared probabilistic-baseline helpers
│   │   ├── run_hashemi_clipping.py    Hashemi clipping-block surrogate
│   │   ├── run_rs.py                  Cohen et al. randomized smoothing
│   │   ├── run_saver.py               SaVer (Sivaramakrishnan et al., HSCC 2025)
│   │   ├── run_probstar.py            ProbStar / StarV (Tran et al.)
│   │   └── _common.py
│   │
│   ├── exp1_vnncomp_subset/        Exp 1 — sound-verifier comparison
│   │   ├── exp1_run_ours.py                  ours (bounded AMLS), --benchmark X
│   │   ├── exp1_run_hashemi_clipping.py
│   │   ├── exp1_run_hashemi_clipping_pca.py  PCA variant (used for metaroom)
│   │   ├── exp1_run_saver.py / exp1_run_probstar.py
│   │   ├── ground_truth.csv                  pre-computed SAT-wins consensus
│   │   └── _benchmarks.py / _common.py / outputs/
│   │
│   ├── exp2_prob_scale/            Exp 2 — probabilistic-scale comparison
│   │   ├── exp2_run_ours.py                  ours, --benchmark X
│   │   ├── exp2_run_hashemi_clipping.py / exp2_run_hashemi_clipping_pca.py
│   │   ├── exp2_run_saver.py / exp2_run_probstar.py / exp2_run_rs.py
│   │   └── _benchmarks.py / _common.py / ground_truth.csv / outputs/
│   │
│   ├── exp3_synthetic/             Exp 3 — synthetic volume comparison
│   │   ├── exp3_run_ours.py                  ours, --benchmark/--score/--spec
│   │   ├── exp3_run_hashemi_clipping.py
│   │   ├── exp3_run_starset_approx.py        sound n2v approx-reach baseline
│   │   ├── _score_pipeline.py                hyperrect / ellipsoid / GMM / flow scores
│   │   ├── exact_volumes.py                  closed-form + cached MC reach volumes
│   │   └── _benchmarks.py / networks.py / outputs/
│   │
│   ├── exp4_scaling/               Exp 4 — controlled depth-scaling on 1-Lipschitz family
│   │   ├── exp4_run_{ours,hashemi_clipping,alpha_beta_crown,neuralsat}.py
│   │   ├── networks.py / instance_generator.py
│   │   └── _benchmarks.py / outputs/
│   │
│   └── exp_ablation/               Verification-method ablation
│       ├── ablation_shared_flow.py           shared (flow, q) per instance
│       └── outputs/
│
├── paper/                          paper-side figure / table generators
│   ├── _common.py / __init__.py
│   ├── tables/
│   │   ├── main_table.py                    % Solved (headline)
│   │   ├── main_table_recall.py             UNSAT-recall (full breakdown)
│   │   ├── main_table_recall_compact.py     UNSAT-recall (percentages only)
│   │   ├── tab5_shared_flow_ablation.py     verifier ablation
│   │   └── outputs/                         generated .tex
│   └── figures/
│       ├── fig4_exp4_scaling.py             accuracy + wall-clock vs network size
│       ├── fig5_exp3_volume_comparison.py   volume ratio per benchmark
│       ├── flow_matching_training/          training-progression overlay
│       └── _common.py
│
├── smokes/                         verify_exact_caches (re-derives MC volume cache)
└── utils/                          shared helpers (compute_exact_reach wrapper)
```

## How to run

All commands assume the n2v conda env is active
(`source /isis/home/sasakis/miniconda3/bin/activate n2v`); with it active `$CONDA`
is just the env's `python`:

```bash
CONDA=python
```

A single benchmark (~3 min):

```bash
$CONDA -m examples.FlowConformal.benchmarks.demo_identity_network
$CONDA -m examples.FlowConformal.benchmarks.demo_rotated_linear
$CONDA -m examples.FlowConformal.benchmarks.demo_acasxu_single
```

A paper experiment (single benchmark smoke):

```bash
$CONDA -m examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_ours \
    --benchmark acasxu_2023 --smoke
$CONDA -m examples.FlowConformal.experiments.exp3_synthetic.exp3_run_ours \
    --benchmark synth_5d --score flow --spec unsat --smoke
$CONDA -m examples.FlowConformal.experiments.exp_ablation.ablation_shared_flow \
    --benchmark acasxu_2023 --smoke
```

A baseline (probabilistic comparison, called via the per-experiment
runners above):

```bash
$CONDA -m examples.FlowConformal.experiments.baselines.run_saver --help
$CONDA -m examples.FlowConformal.experiments.baselines.run_probstar --help
```

A full sweep (best run with `nohup`):

```bash
nohup bash examples/FlowConformal/experiments/run_paper_sweeps.sh \
    > examples/FlowConformal/experiments/outputs/sweep_logs/sweep.log 2>&1 &
```

Or one phase at a time: `--phase exp1`, `exp2`, `exp3`, `exp4`, or
`ablation`.

A regenerated paper figure / table (assuming inputs already in
`outputs/`):

```bash
$CONDA -m examples.FlowConformal.paper.tables.main_table
$CONDA -m examples.FlowConformal.paper.tables.main_table_recall_compact
$CONDA -m examples.FlowConformal.paper.tables.tab5_shared_flow_ablation
$CONDA -m examples.FlowConformal.paper.figures.fig4_exp4_scaling
$CONDA -m examples.FlowConformal.paper.figures.fig5_exp3_volume_comparison
```

## Test suite

```bash
# Flow subset, fast tier (~1 min):
$CONDA -m pytest tests/unit/probabilistic/flow/ -m "not slow" -q

# Flow subset, full (~5 min):
$CONDA -m pytest tests/unit/probabilistic/flow/ -q

# Entire repo test suite (1629 tests, ~19 min):
$CONDA -m pytest tests/ -q
```

For fast iteration when not changing flow internals, append `-m 'not slow'`
to skip the slow tier.

## Default config knobs

The production ACAS Xu defaults live in
`examples/FlowConformal/experiments/exp1_vnncomp_subset/_benchmarks.py:PER_BENCHMARK_CONFIG`
(and analogous per-experiment files). Key knobs:

```python
# Stage-2 verification
verification_method        = 'amls_bounded'   # production default
amls_bounded_eps_2_target  = 0.001            # default = alpha; joint mult. bound 1-(1-α)(1-ε_2) ≈ 2α
m, ell, alpha              = 8000, 7999, 0.001
scenario_n_samples         = 2000             # AMLS samples per level
scenario_beta              = 0.001            # AMLS asymptotic-CI failure prob

# Flow training (ACAS Xu defaults)
flow_config = 'base'
n_train     = 5000
flow_epochs = 2000

# Falsifier (Stage 1, opt-in)
use_falsifier = True   # production runners turn this on for SAT detection
```

Per-benchmark overrides for the other Exp 1 / Exp 2 benchmarks
(`collins_rul_cnn_2022`, `dist_shift_2023`, `linearizenn_2024`,
`tllverify_2023`, `metaroom_2023`, `vit_2023`, `tinyimagenet_2024`,
`cifar100_2024`) are documented inline in each
`PER_BENCHMARK_CONFIG`.

## Bounded AMLS — design summary

Empirical probes during Phase 5d/5e found that unbounded AMLS
over-rejects on benchmarks where the conformal reach set is
disjoint from unsafe but the flow assigns small tail mass outside
the calibrated ball. **Bounded AMLS** restricts the rare-event
search to `||z|| <= q`, giving the right verdict on tllverify
(UNSAT, margin +20) where unbounded AMLS gave UNKNOWN.
Implementation: `n2v/probabilistic/flow/amls_bounded.py`. Design
and soundness argument:
[`docs/research/2026-04-28-bounded-amls-design.md`](../../docs/research/2026-04-28-bounded-amls-design.md).

The unbounded-AMLS code path remains
(`verification_method='amls'`) for the AMLS-vs-bounded-AMLS row in
the verifier-ablation table.
