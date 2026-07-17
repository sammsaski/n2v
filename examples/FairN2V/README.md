# FairN2V - Fairness Verification

Fairness verification of binary classifiers trained on tabular datasets — UCI
Adult-Income, German Credit, Bank Marketing, and folktables ACSIncome — for the
[`n2v`](../../) toolbox. FairN2V verifies fairness in **two complementary
senses**, each in its own subfolder:

| | [Individual / counterfactual](individual_fairness/) | [Group (parity, equal opportunity, equalized odds)](group_fairness/) |
|---|---|---|
| **Property** | per-sample: changing the sensitive attribute (± an ε-perturbation) doesn't change the prediction | population: demographic parity (80%-rule on the favorable rate), equal opportunity (TPR gap ≤ τ), and equalized odds (TPR **and** FPR gaps ≤ τ) |
| **Method** | exact Star-set reachability (sound + complete) | normalizing-flow sampling + Clopper–Pearson (n2v flow module) |
| **Verdict** | per-sample certified; Verified-Fairness score over the test set | fair / unfair / inconclusive at confidence `1 − δ` |
| **Runner** | [`run_individual_fairness.py`](run_individual_fairness.py) | [`run_group_fairness.py`](run_group_fairness.py) |
| **Details** | [individual_fairness/README.md](individual_fairness/README.md) | [group_fairness/README.md](group_fairness/README.md) |

Both share the same datasets, models, and the `DatasetAdapter` infrastructure
described below; only the fairness *question* and the verification *method*
differ.

The individual/counterfactual pipeline is a Python port of the MATLAB **FairNNV**
example that ships with NNV; the group-fairness pipeline follows the **VeriFair**
framing, certified with normalizing flows (a per-group flow as the population
model, Clopper–Pearson as the bound), for three notions —
demographic parity, **equal opportunity**, and **equalized odds** (the latter two
condition on the true label, so a model can fail one notion and pass another).
See [References](#references).

## Shared infrastructure: the adapter

Everything dataset-specific lives behind [`adapter.py`](adapter.py). A
`DatasetAdapter` holds the *nouns* — the normalized samples, the trained net,
per-feature clamps, and the fairness *declaration* (which feature is sensitive,
how it is encoded, which output index is favorable, how the net picks its class).
Both pipelines read these facts instead of hardcoding them. Two registries select
a dataset at run time via `--dataset`: `LOADERS` (what a dataset *is*) and
`RUN_PROFILES` (the data file + the models to verify).

**`folktables`**'s data and FT-* nets are built from scratch to show the
adapter extends cleanly to a new dataset. Built from the
ACSIncome task (predict income > $50k), with the ordinal race code `RAC1P`
one-hot encoded into a 4-way block (13 features total), it serves two profiles
from one dataset and one set of FT-* nets: `folktables` verifies **sex** (binary)
and `folktables_race` verifies **race** (one-hot). See
[Adding a dataset](#adding-a-dataset).

## Models

Fifteen ONNX classifiers in `models/`, grouped by the dataset profile whose
`model_list` (in [`adapter.py`](adapter.py)) selects them:

| Model        | Profile          | Architecture        | Notes |
|--------------|------------------|---------------------|-------|
| AC-1         | `adult`          | 13 → 16 → 8 → 2     | "Small": two narrow hidden layers |
| AC-3         | `adult`          | 13 → 50 → 2         | "Medium": one wider hidden layer  |
| AC-4         | `adult`          | 13 → 100 → 100 → 2  | "Large": two wide hidden layers   |
| ACD-1/3/4    | `adult_debiased` | same as AC-1/3/4    | Debiased (fairness-trained) Adult nets; same data + declaration as `adult` |
| GC-1         | `german`         | 20 → 50 → 2         | |
| GC-2         | `german`         | 20 → 100 → 2        | |
| GC-3         | `german`         | 20 → 9 → 2          | |
| BM-5         | `bank`           | 16 → 22 → 10 → 2    | |
| BM-6         | `bank`           | 16 → 9 → 9 → 2      | |
| BM-7         | `bank`           | 16 → 64 → 64 → 2    | |
| FT-1         | `folktables(_race)` | 13 → 16 → 8 → 2  | Trained in the prep repo; ~80% acc |
| FT-2         | `folktables(_race)` | 13 → 50 → 2      | Trained in the prep repo; ~80% acc |
| FT-3         | `folktables(_race)` | 13 → 100 → 100 → 2 | Trained in the prep repo; ~81% acc |

The FT-* nets are shared by both folktables profiles — `folktables` (sex) and
`folktables_race` (race) verify the *same* data and the *same* nets; only the
sensitive declaration in [`adapter.py`](adapter.py) differs.

Each model ends in a softmax; both pipelines strip it and work on the logits
(softmax is order-preserving, so the predicted class is unchanged).

## Data

The three NNV-derived `data/*.npz` are lossless NumPy conversions of the
corresponding `.mat` from the NNV source examples — they load with `np.load`
alone and contents are unchanged. `X` is samples × features and `y` is one-hot
labels (column 0 is the class used by the pipelines):

| File              | Source `.mat`                                       | `X`          | `y`         |
|-------------------|-----------------------------------------------------|--------------|-------------|
| `adult_data.npz`  | `…/examples/NNV3.0/FairNNV/data/adult_data.mat`     | `(9769, 13)` | `(9769, 2)` |
| `german_data.npz` | `…/examples/Submission/ICAIF24/data/german_data.mat`| `(150, 20)`  | `(150, 2)`  |
| `bank_data.npz`   | `…/examples/Submission/ICAIF24/data/bank_data.mat`  | `(6098, 16)` | `(6098, 2)` |

`folktables_data.npz` `(20000, 13)` / `(20000, 2)` is the exception: it is not
a `.mat` conversion. It is built from the folktables ACSIncome task (California, 2018
1-Year), one-hot encoding the ordinal `RAC1P` race code into a 4-way block (hence
13 columns, not the task's raw 10) and subsampling to 20 000 rows with a fixed
seed. Same `X`/`y` layout as the others (column 0 of `y` is 1 for income > $50k);
one file serves both the `folktables` (sex) and `folktables_race` (race)
profiles. The `adult_debiased` profile likewise reuses `adult_data.npz` (same
data; only the verified models differ).

## Layout

```
examples/FairN2V/
├── README.md                  This overview (shared infra + approach comparison)
├── adapter.py                 DatasetAdapter + per-dataset loaders (LOADERS / RUN_PROFILES)
├── run_individual_fairness.py Runner: exact-reachability pipeline
├── run_group_fairness.py      Runner: group-fairness pipeline (parity + equalized odds)
├── individual_fairness/       verify_individual.py, plot_individual.py, README.md
├── group_fairness/            flow_population.py, intervals.py, verify_parity.py,
│                              plot_parity.py, verify_eqodds.py, plot_eqodds.py, README.md
├── models/                    AC-*, ACD-*, GC-*, BM-*, FT-*.onnx
├── data/                      adult/german/bank/folktables_data.npz
└── results/                   Output, split per approach:
                               individual_fairness/<ts>/ and group_fairness/<ts>/
```

## Running

Requires the `n2v` package importable (from the repo root: `pip install -e .`)
and Python 3.9+; dependencies are in [`requirements.txt`](../../requirements.txt).
Run the two pipelines from this folder; each resolves `models/`, `data/`, and
`results/` automatically:

```bash
cd examples/FairN2V
python run_individual_fairness.py            # per-sample (counterfactual + individual)
python run_group_fairness.py                 # group (demographic parity + equalized odds)
```

Both accept `--dataset` (`adult` default, `adult_debiased`, `german`, `bank`,
`folktables`, `folktables_race`) and `--models`. For each pipeline's full flags,
outputs, configuration, and expected runtime, see its own README linked in the
table above.

## Adding a dataset

A dataset is a **loader** (what it is) and a **run profile** (how to run it). For
a dataset already in the shared npz `X`/`y` layout, adding it is a thin `load_*`
wrapper plus one `LOADERS` and one `RUN_PROFILES` entry in
[`adapter.py`](adapter.py) — the loader stamps the fairness *declaration*
(`sensitive_features`, `perturbable_features`, `sensitive_encoding`,
`output_size`, `class_type`, `positive_class`) onto a `DatasetAdapter`;
`_load_npz_adapter` does the loading, min-max normalization, and softmax-stripped
model wrapping. Both pipelines then pick it up for free.

`folktables` is the worked example of adding one *from scratch* (no
pre-existing data or ONNX). The scripts that build `folktables_data.npz` and train/export
the FT-* nets — and the conventions they must satisfy to line up with the adapter
(argmin nets so the loader can use `class_type='min'`; the same min-max stats the
adapter recomputes from the npz, so inputs line up) — live in a separate repo to
keep n2v small: **[dataset-prep-fairn2v](https://github.com/jhsu7769/dataset-prep-fairn2v)**.
Once the data and ONNX are in place, wire it like any other: `load_folktables` +
a `LOADERS` entry + a `RUN_PROFILES` entry, and add display names in the relevant
plot script.

`folktables_race` shows the cheap case — a new fairness *verb* over existing
data: the same `.npz` and FT-* nets with only a different declaration (sensitive
columns 9–12, `sensitive_encoding='onehot'`), i.e. one extra
`load_folktables_race` + `LOADERS` + `RUN_PROFILES` entry, no new data or models.

## References

- **FairNNV**: Tumlin, A.M., Manzanas Lopez, D., Robinette,
  P., Zhao, Y., Derr, T., Johnson, T.T. *FairNNV: The neural network
  verification tool for certifying fairness.* Proceedings of the 5th
  ACM International Conference on AI in Finance (ICAIF '24), 2024.
- **VeriFair**: Bastani, O., Zhang, X., Solar-Lezama, A. *Probabilistic
  Verification of Fairness Properties via Concentration.* OOPSLA 2019.
  arXiv:1812.02573.
- **Counterfactual fairness definition**: Kusner, M.J., Loftus, J.R.,
  Russell, C., Silva, R. *Counterfactual fairness.* NeurIPS 2017.
- **Adult-Income dataset**: Becker, B. & Kohavi, R. *Adult.*
  UCI Machine Learning Repository, 1996.
- **German-Credit dataset**: Hofmann, H. *Statlog (German Credit Data).*
  UCI Machine Learning Repository, 1994.
- **Bank-Marketing dataset**: Moro, S., Rita, P., Cortez, P. *Bank Marketing.*
  UCI Machine Learning Repository, 2014.
- **folktables / ACSIncome**: Ding, F., Hardt, M., Miller, J., Schmidt, L.
  *Retiring Adult: New Datasets for Fair Machine Learning.* NeurIPS 2021.

