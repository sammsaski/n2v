# Group Fairness — Demographic Parity, Equal Opportunity, Equalized Odds (flow population)

Probabilistic, **population-level** group-fairness verification, built on the
`n2v.probabilistic.flow` normalizing-flow module. Where the
[individual-fairness](../individual_fairness/) pipeline proves a *per-sample*
property with exact Star reachability, this one certifies *distributional*
properties by sampling a learned population model of each group and bounding its
rates with Clopper–Pearson intervals. It checks **three** group-fairness notions,
following the **VeriFair** framing (Bastani, Zhang & Solar-Lezama, OOPSLA 2019)
with a learned-flow population model. For the shared dataset / model / adapter
infrastructure, see the [top-level README](../README.md).

## What it verifies

Let `μ_a = Pr[favorable | group = a]`. A single `run_group_fairness.py` reports
all three notions, each three-valued (**fair** / **unfair** / **inconclusive**):

**1. Demographic parity** — the **80%-rule** on the *marginal* favorable rate:

```
        min_a μ_a
Y  ≡   ───────────  ≥  c          (c = 0.8 by default)
        max_a μ_a
```

**2. Equal opportunity** and **3. Equalized odds** — both condition on the *true
label* `Y`; equal opportunity is the TPR half, equalized odds adds the FPR half
(so equalized odds is the stronger notion: passing it implies passing equal
opportunity, but not the reverse). Each gap must be within `τ` (default 0.1):

```
equal opportunity:  |TPR_a − TPR_b| ≤ τ                              for every group pair
equalized odds:     |TPR_a − TPR_b| ≤ τ   AND   |FPR_a − FPR_b| ≤ τ
   TPR_a = Pr[favorable | a, Y=favorable]     FPR_a = Pr[favorable | a, Y=unfavorable]
```

The two families use different metrics on purpose: parity is a *multiplicative*
ratio (the legal 80%-rule / disparate-impact convention), while the
label-conditioned notions use an *additive* gap `τ` (as Hardt et al. state
equalized odds). They aren't meant to be read on the same scale.

**Why several — they can disagree, and that's the point.** Demographic parity
*ignores* qualifications; the label-conditioned notions *condition* on them. A
model can **fail demographic parity but pass equal opportunity / equalized odds**
when the groups have different base rates but comparable (same-label) people are
treated equally — the disparity is in the world's label distribution, not the
model. On adult, **AC-4 is exactly this case**: unfair by demographic parity, fair
by both label-conditioned notions. And a model that equalizes TPR but not FPR
passes equal opportunity yet fails equalized odds. Neither notion is "correct";
reporting all three shows *which kind* of disparity (if any) is present.

## How it works

Demographic parity is a statement about a *distribution* of inputs, so there's no
finite input box to enumerate — we sample a population `P_{V|a}` per group. That
population is a **normalizing flow** trained on each group's real rows with n2v's
conditional flow matching (`train_flow` / `FlowODE`, `coupling='none'`).

- **Population model** — [`flow_population.py`](flow_population.py),
  `FlowGroupModel`: one flow per group. `sample(a, batch)` draws Gaussian latents,
  maps them with `FlowODE.inverse`, de-standardizes, clips to `[0, 1]`, and pins
  the sensitive columns to group `a`. A `mask` argument restricts training to a
  true-label cell (used by the label-conditioned notions).
- **Classifier** — `make_favorable_classifier(adapter)`: `net.forward` +
  `argmin`/`argmax` (per `class_type`) against `adapter.positive_class`.
- **Demographic parity** — [`verify_parity.py`](verify_parity.py): one flow per
  group, `n_samples` draws, two-sided Clopper–Pearson interval per rate, 80%-rule
  in difference form `AND_{a≠b}(μ_a − c·μ_b ≥ 0)`; budget `δ` split across the
  `k` group intervals.
- **Equal opportunity + equalized odds** — [`verify_eqodds.py`](verify_eqodds.py):
  per-(group, label) flows (`Y = favorable` cell → TPR, `Y = unfavorable` cell →
  FPR), a CP interval per rate, the absolute-gap test on TPR and on FPR; `δ` split
  `δ/2` per condition then across the `k` intervals. **Equal opportunity** = the
  TPR-gap verdict alone; **equalized odds** = TPR-gap **and** FPR-gap.

**Two verdicts per model.** Every notion is decided twice, with the same CP
interval + verdict logic ([`intervals.py`](intervals.py)) applied to two count
sources:

- **flow** — counts from `n_samples` draws of the fitted population `P_{V|a}`.
  The guarantee is *distributional* ("fair w.r.t. the fitted flow, confidence
  ≥ 1 − δ"); the interval is tight regardless of how much real data backs it.
- **real** — counts from the actual test rows. Assumption-free (it bounds only
  the empirical rate), so the interval widens on data-starved groups and the
  verdict honestly reads *inconclusive* rather than confidently wrong. It also
  still returns a verdict when a cell is too small to fit a flow.

The runner [`../run_group_fairness.py`](../run_group_fairness.py) sets the config
and chains both verifications and both plots. Every step module also runs
standalone (default paths under the FairN2V dir when `config` is not supplied).

## Files

| File                  | Role |
|-----------------------|------|
| `flow_population.py`  | `FlowGroupModel` (per-group / per-cell flow sampler) + `make_favorable_classifier`. |
| `intervals.py`        | `clopper_pearson`: the shared two-sided exact-binomial interval both verifiers use. |
| `verify_parity.py`    | Demographic parity: `main(config)` → `parity_<dataset>.csv`; `verify_model_flow`, `parity_verdict`. |
| `plot_parity.py`      | Demographic parity: `main(config)` reads the CSV → per-group rate bars + parity-floor band. |
| `verify_eqodds.py`    | Equal opportunity + equalized odds: `main(config)` → `eqodds_<dataset>.csv`; `verify_model_eqodds`, `gap_verdict`. |
| `plot_eqodds.py`      | Equal opportunity + equalized odds: `main(config)` reads the CSV → TPR / FPR bars + τ gap-tolerance band. |

## Running

From the FairN2V dir (so `models/`, `data/`, `results/` resolve):

```bash
python run_group_fairness.py                                 # Adult (default)
python run_group_fairness.py --dataset german
python run_group_fairness.py --dataset folktables_race       # 4-group one-hot
python run_group_fairness.py --models AC-1 AC-4 --n-samples 20000
```

`--dataset` selects the profile, `--models` overrides its model list, and
`--n-samples` sets the flow draws per group. A run trains `k` demographic-parity
flows **plus** `2k` equalized-odds cell flows per model (seconds each), so a full
sweep is a few minutes — `folktables_race` (4 groups) is the slowest.

## Configuration parameters

Edit the `CONFIGURATION` block at the top of
[`../run_group_fairness.py`](../run_group_fairness.py), or pass a pre-populated
`config` dict to the step modules' `main(config)`:

| Key             | Default                          | Effect |
|-----------------|----------------------------------|--------|
| `dataset`       | `'adult'`                        | Dataset profile (see `RUN_PROFILES` in `adapter.py`) |
| `model_list`    | profile default (`AC-1, AC-3, AC-4` for adult) | Which models to verify |
| `n_samples`     | `50000`                          | Flow draws per group for the CP intervals |
| `n_epochs`      | `300`                            | Training epochs per flow |
| `random_seed`   | `0`                              | RNG seed (flow init + sampling) |
| `c`             | `0.8`                            | Demographic-parity threshold (80%-rule constant) |
| `tau`           | `0.1`                            | Equal-opportunity / equalized-odds tolerance (max allowed TPR/FPR gap) |
| `delta`         | `0.05`                           | Failure budget per notion (verdict holds w.p. ≥ 1 − δ) |
| `save_png` / `save_pdf` | `True`                   | Figure formats to write |

## Outputs

A timestamped subfolder `results/group_fairness/<yymmdd-HHMMSS>/` per run:

- `parity_<dataset>.csv` / `.png` / `.pdf` — demographic parity: per model, the
  flow **and** real-data verdicts, per-group flow rates + CP intervals, and the
  real-data rates + CP intervals. The figure is one bar cluster per model, one bar
  per group; the shaded band is the parity floor `c · max_a μ_a` — a group below
  the band is a proven violation. Each cluster is labelled with both verdicts.
- `eqodds_<dataset>.csv` / `.png` / `.pdf` — equal opportunity + equalized odds:
  per model, the equal-opportunity (TPR) and equalized-odds (TPR+FPR) verdicts,
  each computed on the flow **and** on the real cells, with per-group rates + CP
  intervals for both. The figure has two panels (TPR | FPR); the shaded band is
  `[min_rate, min_rate + τ]` — a bar poking above it is a gap > τ.

Both CSVs also record each model's wall-clock verification time (`Runtime`
column); the verifiers print the median per run.

Each report's `summary()` prints the **flow** verdict (over the fitted
population) next to the **real-data** verdict (exact binomial on the actual
rows). On well-sampled groups the two agree; on small groups the flow can look
confident where the real-data verdict honestly reads inconclusive.

## Caveats

- **The flow verdict is relative to the fitted flow**, not the true
  data-generating process. Its Clopper–Pearson interval bounds only the
  Monte-Carlo error of the draws — not how well the flow matches the real
  population. This is why every notion is *also* reported on the real rows.
- **Small groups are unreliable, and equalized odds makes this worse.**
  Conditioning on the label shrinks each cell (adult's Female-favorable cell is
  already only 365 rows); on german / bank a cell can be too small to fit a flow.
  When that happens the flow verdict is reported as n/a, but the real-data verdict
  — an exact binomial on the *real* per-cell counts — still stands (and correctly
  reads inconclusive when the cell is tiny).
- Flows are trained with `coupling='none'` (vanilla CFM) for speed rather than
  the OT `'hungarian'` default — fine for a population density model.

## References

- **VeriFair**: Bastani, O., Zhang, X., Solar-Lezama, A. *Probabilistic
  Verification of Fairness Properties via Concentration.* OOPSLA 2019.
  arXiv:1812.02573.
- **Equalized odds**: Hardt, M., Price, E., Srebro, N. *Equality of Opportunity
  in Supervised Learning.* NeurIPS 2016. arXiv:1610.02413.
- **80%-rule / disparate impact**: Feldman, M., Friedler, S.A., Moeller, J.,
  Scheidegger, C., Venkatasubramanian, S. *Certifying and Removing Disparate
  Impact.* KDD 2015.
- **Flow matching**: Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, T.
  *Flow Matching for Generative Modeling.* ICLR 2023 (n2v uses the OT-CFM variant).
