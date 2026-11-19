# Individual & Counterfactual Fairness (Exact Reachability)

Exact, **per-sample** fairness verification via Star-set reachability — the
original FairN2V pipeline, a Python port of NNV's MATLAB **FairNNV** example. For
the shared dataset / model / adapter infrastructure and the group-fairness
counterpart, see the [top-level README](../README.md).

## What it verifies

- **Counterfactual fairness** (ε = 0) — changing a sensitive attribute (flipping
  a binary one like sex, or switching a one-hot one like race to any other
  category) must not change the prediction.
- **Individual fairness** (ε > 0) — for every input, no perturbation within an
  ε-ball (combined with that change of the sensitive attribute) changes the
  prediction. Verified across multiple ε values.

Each verdict is summarized as a **Verified Fairness (VF)** score: the proportion
of test samples for which fairness is formally certified.

## How it works

For each test sample, [`verify_individual.py`](verify_individual.py) builds an
input **Star set** — the sensitive attribute pinned to its counterfactual
value(s), the perturbable numerical features widened by ±ε, everything clamped to
the normalized `[0, 1]` domain — propagates it through the (softmax-stripped) net
with exact reachability, and checks the output specification that the predicted
class is preserved. Because reachability is exact (no over-approximation), the
verdict is sound *and* complete. A one-hot sensitive attribute yields k−1
counterfactuals per sample (the prediction must hold across all of them);
binary yields one.

## Files

| File                   | Role |
|------------------------|------|
| `verify_individual.py` | Loads each ONNX model, runs reachability + verification, writes CSVs. |
| `plot_individual.py`   | Reads the latest CSVs, generates figures + LaTeX tables. |

The runner [`../run_individual_fairness.py`](../run_individual_fairness.py) sets
the config and chains the two. Both modules also run standalone (they fall back
to default paths under the FairN2V dir when `config` is not supplied).

## Running

From the FairN2V dir (so `models/`, `data/`, `results/` resolve):

```bash
python run_individual_fairness.py                  # Adult (default)
python run_individual_fairness.py --dataset german
python run_individual_fairness.py --dataset bank
python run_individual_fairness.py --dataset adult_debiased
python run_individual_fairness.py --dataset folktables
python run_individual_fairness.py --dataset folktables_race
```

The Adult run verifies AC-1, AC-3, AC-4 on 100 observations — counterfactual
fairness (ε = 0) plus individual fairness across the paper's ε grid — then writes
the CSVs, figure, and LaTeX tables to `results/individual_fairness/<timestamp>/`.
`--num-obs N` is auto-capped to the dataset size (German has only 150 samples),
and `--models GC-1 GC-2` overrides the profile's model list.

## Configuration parameters

Edit the `CONFIGURATION` block at the top of
[`../run_individual_fairness.py`](../run_individual_fairness.py), or pass a
pre-populated `config` dict to the step scripts' `main(config)` (the runner uses
`setdefault`, so caller-supplied values are preserved):

| Key                      | Default                          | Effect |
|--------------------------|----------------------------------|--------|
| `dataset`                | `'adult'`                        | Dataset profile (see `RUN_PROFILES` in `adapter.py`) |
| `model_list`             | profile default (`AC-1, AC-3, AC-4` for adult) | Which models to verify |
| `num_obs`                | `100` (capped to dataset size)   | Number of test observations |
| `random_seed`            | `500`                            | RNG seed (NumPy `default_rng`) |
| `timeout`                | `600`                            | Per-epsilon timeout (s) |
| `epsilon_counterfactual` | `[0.0]`                          | ε grid for counterfactual |
| `epsilon_individual`     | `[0.01,0.02,0.03,0.05,0.07,0.1]` | ε grid for individual |
| `save_png` / `save_pdf`  | `True`                           | Figure formats to write |

## Outputs

A timestamped subfolder `results/individual_fairness/<yymmdd-HHMMSS>/` per run:

- `counterfactual_<ts>.csv` — per-model fair / unfair %
- `individual_<ts>.csv`     — per-model × ε fair / unfair / unknown %
- `timing_<ts>.csv`         — per-model × ε total + per-sample time
- `counterfactual_table.tex` — booktabs-style LaTeX table
- `individual_fairness_combined.png` / `.pdf` — area plot across models
- `timing_table.tex`         — LaTeX timing table

## Expected runtime

Measured on a MacBook Pro, CPU only. Full sweep (3 models, ε ∈ {0, 0.01, 0.02,
0.03, 0.05, 0.07, 0.1}, 100 obs; plotting adds ~1 s):

| `--dataset`      | sweep time | dominated by (model @ ε=0.1) |
|------------------|-----------:|------------------------------|
| `adult`          |    ~60 s   | AC-4 (~0.31 s/sample)        |
| `adult_debiased` |    ~65 s   | ACD-4 (~0.33 s/sample)       |
| `bank`           |    ~72 s   | BM-7 (~0.48 s/sample)        |
| `german`         |    ~87 s   | GC-2 (~0.56 s/sample)        |
| `folktables`     |   ~460 s   | FT-3 (~2.6 s/sample)         |
| `folktables_race`|   ~630 s   | FT-3 (~3.4 s/sample)         |

A **smoke** run (`--models AC-1 --num-obs 10`) is **~2–3 s**. Per-sample cost
grows steeply with ε — larger input boxes mean more ReLU case-splitting in exact
reachability — and with width/depth, so the largest net at ε = 0.1 dominates;
ε = 0 (a single point) is near-instant. `folktables_race` is ~1.5× slower than
`folktables` on the same nets because its one-hot race attribute yields k−1 = 3
counterfactuals per sample (up to 3× the reachability calls, short-circuited at
the first violation).

## Validation against NNV

The four NNV-derived profiles were checked against NNV on MATLAB's exact seed-500
samples (`rng(500); randsample(...)`), across both fairness notions and the full
ε grid. **Adult**, **`adult_debiased`**, and **`bank`** reproduce the NNV results
*exactly* — bit-for-bit on every model and ε (48 of 48 cells). **`german`**
matches everywhere except 6 cells, each off by at most 1: those rows sit at a
perturbable feature's extreme, where this port builds a valid input box but
MATLAB fed `ImageStar` a degenerate `lb > ub` one. The divergence is a deliberate
soundness fix, not a mismatch — matching NNV there would mean reproducing the bug.
