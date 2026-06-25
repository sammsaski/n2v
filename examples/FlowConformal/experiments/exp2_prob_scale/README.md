# Exp 2: Probabilistic-Scale Comparison

Run flow-conformal on three large image-classification benchmarks
where sound verifiers are expected to TIMEOUT, and compare against
Hashemi-clipping (with PCA at high output dim), SaVer, ProbStar, and
Cohen-style randomized smoothing.

## Roster (3 benchmarks)

All three are multi-class image classification with cora-style nested
OR specs (`list[1 dict]` + `Hg = list[K HalfSpaces]`). Ours uses a
**uniform config** across all three — `mega + amls_bounded_union +
amls_max_levels=30` — so any cross-benchmark differences come from
the network/spec, not the verifier setup.

| short name           | spec disjuncts | ONNX size | Output dim | Per-row budget | Source |
|----------------------|----------------|-----------|------------|----------------|--------|
| `vit_2023`           | 9 (10-class)   | 0.3 MB    | 10         | 100 s | VNN-COMP 2023 |
| `cifar100_2024`      | 99 (100-class) | 9.7 MB    | 100        | 100 s | VNN-COMP 2024 |
| `tinyimagenet_2024`  | 199 (200-class)| 13.8 MB   | 200        | 100 s | VNN-COMP 2024 |

## Files

```
_benchmarks.py                    PER_BENCHMARK_CONFIG, deferred loaders, VNN-COMP path lookup
_common.py                        shared loaders, sweep harness
exp2_run_ours.py                  ours runner, --benchmark X
exp2_run_hashemi_clipping.py      Hashemi-clipping baseline (m=8000)
exp2_run_hashemi_clipping_pca.py  Hashemi-clipping with PCA (used for the headline rows)
exp2_run_saver.py                 SaVer (Sivaramakrishnan et al., HSCC 2025)
exp2_run_probstar.py              ProbStar / StarV
exp2_run_rs.py                    Cohen RS (cifar100_2024 only)
ground_truth.csv                  pre-computed SAT-wins consensus from VNN-COMP 2025 (8 tools)
outputs/                          per-(benchmark, method) CSVs land here
```

## Smoke

```bash
PY=python
cd /path/to/n2v
for bench in vit_2023 tinyimagenet_2024 cifar100_2024; do
  for tool in ours hashemi_clipping; do
    $PY -m examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_$tool \
        --benchmark $bench --smoke
  done
done
# RS smoke (cifar100_2024 only)
$PY -m examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_rs \
    --benchmark cifar100_2024 --smoke
```

## Full sweep

Use the project-level launcher (per-instance shell timeouts via
`run_cell.sh`):

```bash
bash examples/FlowConformal/experiments/run_paper_sweeps.sh --phase exp2
```

## Aggregation

The headline `% Solved` and `UNSAT-recall` numbers are computed by the
paper-side scripts under `examples/FlowConformal/paper/tables/`
(`main_table.py`, `main_table_recall.py`, `main_table_recall_compact.py`),
which read every `outputs/exp2_<bench>_<method>.csv` directly and join
against [`ground_truth.csv`](ground_truth.csv).

## Sound-verifier expectations (paper context)

For the "no sound verifier scales here" claim:

| benchmark | sound-verifier expectation |
|---|---|
| `vit_2023` | αβ-CROWN solves a small fraction; most UNKNOWN/TIMEOUT in VNN-COMP 2023 |
| `cifar100_2024` | αβ-CROWN solves some at 100 s budget; ResNet-medium pushes against the budget |
| `tinyimagenet_2024` | αβ-CROWN solves few; 200-class disjunctive spec amplifies search cost |

The headline tables read the published VNN-COMP 2025 results for
sound-verifier rows (αβ-CROWN, NeuralSAT, PyRAT, CORA) — no local
re-run.
