# Group-fairness test suite

Validates that swapping VeriFair's exact method for **n2v's normalizing flow +
Clopper–Pearson** yields a verifier that (1) works and (2) is sound and
mathematically correct. The code under test lives in
`examples/FairN2V/group_fairness/` (imported via a `sys.path` insertion in
`conftest.py`, like the other example-code tests in this suite).

```bash
pytest tests/unit/fairness/                 # everything (~40s)
pytest tests/unit/fairness/ -m "not slow"   # pure logic/statistics core (~0.1s)
```

## What "sound" means here (and what the tests must show)

The guarantee is *conditional on the population model*: **given** the fitted
per-group population `P_{V|a}`, the verdict is wrong with probability ≤ δ, where
the randomness is the Monte-Carlo draws. That splits into two layers:

| Layer | Claim | Tested by |
|-------|-------|-----------|
| **Statistical / logical core** | Given i.i.d. draws from `P_{V\|a}` and a deterministic classifier, the CP intervals cover the true rates and the verdict logic is correct at confidence `1 − δ` | `test_clopper_pearson`, `test_parity_verdict`, `test_gap_verdict` — **pure math, no flow** |
| **Flow fidelity** | The flow matches the true data distribution | *Not a soundness claim* (VeriFair has the same caveat). Only **measured**, in `test_end_to_end`'s with-flow tier. |

The soundness burden is entirely on the core, and the core is independent of the
flow — swapping the population model cannot make a sound verdict unsound.

## Test map

**Soundness / correctness (the core):**
- `test_clopper_pearson.py` — the interval is the true Clopper–Pearson interval
  (independent binomial-tail oracle via `scipy.stats.binom`, not our Beta form)
  and its Monte-Carlo **coverage** is ≥ 1 − α for every `p`.
- `test_parity_verdict.py` — three-valued logic; the difference-form encoding
  `AND_{a≠b}(μ_a − c·μ_b ≥ 0)` is verdict-**equivalent to the ratio 80%-rule**;
  Monte-Carlo **soundness** (definitive-wrong ≤ δ, incl. the k=4 union bound); and
  a side-by-side showing a naive point-estimate verdict *violates* δ on the same
  data (so the soundness test has power).
- `test_gap_verdict.py` — the equalized-odds gap logic, the `_combine` truth
  table (EO = TPR-parity ∧ FPR-parity), and its Monte-Carlo soundness.

**Works (plumbing + recovery):**
- `test_population_sampler.py` — the classifier reproduces the net's decision,
  group labels / label masks select the right rows, sensitive columns are pinned.
- `test_end_to_end.py` — recovery of an **analytic** ground-truth rate: a
  bypass-flow tier (tight; isolates estimator + verdict) and a with-flow tier
  (looser; the flow-fidelity measurement) + reproducibility.

Slow tests (flow training, large Monte-Carlo) are marked `@pytest.mark.slow`.
Fixtures and the analytic-rate stand-in net live in `conftest.py`.
