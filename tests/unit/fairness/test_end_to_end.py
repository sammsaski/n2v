"""End-to-end recovery against an analytic ground truth ("it works"), plus
reproducibility.

`known_rate_adapter` has per-group favorable rates that are analytic under
`ThresholdNet` (0.5 and 2/3, ratio 0.75 -> truly UNFAIR). Two tiers:

  * BYPASS-FLOW -- sample the known distribution directly (numpy) and run the
    classifier + CP + verdict. Tight: isolates the estimator/verdict from any flow
    fit error, and checks the CP interval covers the true rate.
  * WITH-FLOW -- fit the actual `FlowGroupModel` and recover the same rates within
    a looser tolerance. This is a flow *fidelity* check (a modeling-quality
    measurement, explicitly NOT part of the soundness guarantee), plus a
    reproducibility check.
"""
import numpy as np
import pytest

from tests.unit.fairness.conftest import SEED
from flow_population import FlowGroupModel, make_favorable_classifier
from intervals import clopper_pearson
from verify_parity import parity_verdict


def test_bypass_flow_recovers_known_rates(known_rate_adapter):
    """Classifier + CP + verdict recover ground truth when we sample the known
    distribution directly (no flow)."""
    adapter, (r0, r1) = known_rate_adapter
    classify = make_favorable_classifier(adapter)
    rng = np.random.default_rng(SEED)
    N = 40_000

    def draw(a):
        x0 = rng.uniform(0.0, 1.0, N) if a == 0 else rng.uniform(0.25, 1.0, N)
        V = np.zeros((N, 3))
        V[:, 0], V[:, 1], V[:, 2] = x0, rng.uniform(0, 1, N), float(a)
        return V

    counts = []
    for a, true_rate in [(0, r0), (1, r1)]:
        s = int(classify(draw(a)).sum())
        counts.append((s, N))
        lo, hi = clopper_pearson(s, N, 0.025)
        assert lo <= true_rate <= hi                 # interval covers ground truth

    verdict, _ = parity_verdict(counts, c=0.8, delta=0.05)
    assert verdict is False                          # ratio 0.75 -> truly unfair


@pytest.mark.slow
def test_flow_recovers_known_rates(known_rate_adapter):
    """The fitted flow reproduces the analytic per-group rates within tolerance,
    and the verdict is correct. (Fidelity measurement, not a soundness claim.)"""
    adapter, (r0, r1) = known_rate_adapter
    model = FlowGroupModel(adapter, n_epochs=250, seed=SEED)
    classify = make_favorable_classifier(adapter)
    N = 30_000
    rates = [float(classify(model.sample(a, N)).mean()) for a in range(model.k)]
    assert rates[0] == pytest.approx(r0, abs=0.06)
    assert rates[1] == pytest.approx(r1, abs=0.06)

    counts = [(int(round(rt * N)), N) for rt in rates]
    verdict, _ = parity_verdict(counts, c=0.8, delta=0.05)
    assert verdict is False


@pytest.mark.slow
def test_reproducibility_same_seed(known_rate_adapter):
    """Same seed -> same fitted flow -> same samples (a verifier must be
    deterministic)."""
    adapter, _ = known_rate_adapter
    s1 = FlowGroupModel(adapter, n_epochs=30, seed=7).sample(0, 200)
    s2 = FlowGroupModel(adapter, n_epochs=30, seed=7).sample(0, 200)
    assert np.allclose(s1, s2)
