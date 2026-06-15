# FairN2V - Fairness Verification of the Adult-Income Classifier

Exact reachability-based verification of two notions of fairness on
binary classifiers trained on the UCI Adult-Income dataset:

- **Counterfactual fairness** — flipping a sensitive attribute (e.g.
  sex) must not change the prediction. Verified at ε = 0.
- **Individual fairness** — for every input, no perturbation within an
  ε-ball (combined with a flip of the sensitive attribute) changes the
  prediction. Verified across multiple ε values.

Each verdict is summarized as a Verified Fairness (VF) score: the
proportion of test samples for which fairness is formally certified.

This example is a Python port — for the [`n2v`](../../) toolbox — of the
MATLAB **FairNNV** example that ships with NNV. The verification logic, models,
and dataset are carried over directly; see [References](#references). It has
been checked to reproduce the NNV results exactly on identical samples.

## References

- **FairNNV**: Tumlin, A.M., Manzanas Lopez, D., Robinette,
  P., Zhao, Y., Derr, T., Johnson, T.T. *FairNNV: The neural network
  verification tool for certifying fairness.* Proceedings of the 5th
  ACM International Conference on AI in Finance (ICAIF '24), 2024.
- **Counterfactual fairness definition**: Kusner, M.J., Loftus, J.R.,
  Russell, C., Silva, R. *Counterfactual fairness.* NeurIPS 2017.
- **Adult-Income dataset**: Dheeru & Efi. *UCI Machine Learning
  Repository — Adult.* 2017.

## Models

Two ONNX classifiers in `models/`:

| Model | Architecture           | Notes                       |
|-------|------------------------|-----------------------------|
| AC-1  | 13 → 16 → 8 → 2        | "Small": two narrow hidden layers |
| AC-3  | 13 → 50 → 2            | "Medium": one wider hidden layer  |

Each model ends in a softmax; the runner strips it before reachability and
verifies on the logits (softmax is order-preserving, so the predicted class is
unchanged and the output specification stays linear).

## Data

`data/adult_data.npz` is a lossless NumPy conversion of `adult_data.mat`
from the NNV source example
(`nnv/code/nnv/examples/NNV3.0/FairNNV/data/adult_data.mat`). It loads with
`np.load` alone — no scipy needed at run time. Contents are unchanged:

- `X`: `(9769, 13)` float64 — samples × features
- `y`: `(9769, 2)` float64 — one-hot labels (column 0 is the class used by
  the verification pipeline)

## Layout

```
examples/FairN2V/
├── README.md
├── run_fairn2v.py      Top-level runner; sets config and chains the steps
├── adult_verify.py     Loads ONNX, runs reachability + verification, writes CSVs
├── plot_results.py     Reads the latest CSVs, generates figures + LaTeX tables
├── models/
│   ├── AC-1.onnx
│   └── AC-3.onnx
├── data/
│   └── adult_data.npz
└── results/            Timestamped output (<yymmdd-HHMMSS>/)
```

`adult_verify.py` and `plot_results.py` can also run standalone — they
fall back to default paths in this folder when `config` is not already
in scope.

## Running

Requires the `n2v` package importable (from the repo root: `pip install -e .`)
and Python 3.9+; dependencies are in [`requirements.txt`](../../requirements.txt).
The runner and the two step scripts resolve `models/`, `data/`, and `results/`
relative to this folder, so no paths need configuring.

### Default sweep

```bash
cd examples/FairN2V
python run_fairn2v.py
```

Verifies AC-1 and AC-3 on 100 observations, counterfactual fairness
(ε = 0) plus individual fairness across the paper's ε grid, then writes the
CSVs, figure, and LaTeX tables to `results/<timestamp>/`.

### Smoke / custom run

There is no separate smoke flag. Either edit the `CONFIGURATION` block at the
top of [`run_fairn2v.py`](run_fairn2v.py) (e.g. `model_list=['AC-1']`,
`num_obs=10`, `epsilon_individual=[0.01]`, `timeout=120`) and run it the same
way, or call the step scripts' `main(config)` directly for a one-off:

```python
from pathlib import Path
import adult_verify, plot_results

config = {
    'models_dir': Path('models'), 'data_dir': Path('data'),
    'output_dir': Path('results/smoke'), 'data_file': 'adult_data.npz',
    'model_list': ['AC-1'], 'num_obs': 10, 'random_seed': 500, 'timeout': 120,
    'epsilon_counterfactual': [0.0], 'epsilon_individual': [0.01],
    'save_png': True, 'save_pdf': True,
}
adult_verify.main(config)
plot_results.main(config)
```

## Configuration parameters

Edit the `CONFIGURATION` block at the top of [`run_fairn2v.py`](run_fairn2v.py),
or pass a pre-populated `config` dict to the step scripts' `main(config)`
(the runner uses `setdefault`, so any caller-supplied values are preserved):

| Key                       | Default                          | Effect |
|---------------------------|----------------------------------|--------|
| `model_list`              | `['AC-1', 'AC-3']`               | Which models to verify (filenames without `.onnx`) |
| `num_obs`                 | `100`                            | Number of test observations |
| `random_seed`             | `500`                            | RNG seed (NumPy `default_rng`) |
| `timeout`                 | `600`                            | Per-epsilon timeout (s) |
| `epsilon_counterfactual`  | `[0.0]`                          | ε grid for counterfactual |
| `epsilon_individual`      | `[0.01,0.02,0.03,0.05,0.07,0.1]` | ε grid for individual |
| `save_png` / `save_pdf`   | `True`                           | Figure formats to write |

## Outputs

A timestamped subfolder `results/<yymmdd-HHMMSS>/` is created per run
and contains:

- `counterfactual_<ts>.csv` — per-model fair / unfair %
- `individual_<ts>.csv`     — per-model × ε fair / unfair / unknown %
- `timing_<ts>.csv`         — per-model × ε total + per-sample time
- `counterfactual_table.tex` — booktabs-style LaTeX table
- `individual_fairness_combined.png` / `.pdf` — area plot across models
- `timing_table.tex`         — LaTeX timing table

## Expected runtime

Measured on a MacBook Pro, CPU only (n2v runs on CPU here):

- **Smoke** (AC-1 only, 10 obs, ε ∈ {0.01}): **~2–3 s**
- **Default sweep** (2 models, 7 ε values, 100 obs, including plotting): **~20–30 s**

Per-sample cost grows with ε — larger input boxes mean more ReLU case-splitting
in exact reachability. AC-3 at ε = 0.1 dominates (~0.05 s/sample); ε = 0
(a single point) is near-instant.
