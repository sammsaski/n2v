"""Soundness + correctness of the demographic-parity verdict (`parity_verdict`).

Four things are checked:
  * the three-valued interval logic (fair / unfair / inconclusive) on constructed
    cases;
  * that the difference-form encoding AND_{a!=b}(mu_a - c*mu_b >= 0) is verdict-
    equivalent to the ratio 80%-rule min/max >= c (encoding faithfulness);
  * MONTE-CARLO SOUNDNESS: over repeated sampling, a *definitive* wrong verdict
    (certifying the wrong side of the 80%-rule) happens with probability <= delta,
    including the multi-group union bound (k = 4);
  * a mutation ("test the test"): a naive point-estimate verdict with no intervals
    EXCEEDS delta on the same near-boundary case, so the soundness test has power.
"""
import numpy as np
import pytest

from tests.unit.fairness.conftest import SEED
from verify_parity import parity_verdict


def _counts(rates, n):
    return [(int(round(r * n)), n) for r in rates]


_HUGE = 10 ** 7        # collapses the CP intervals so the verdict is definitive


# --------------------------- interval logic ---------------------------------

def test_unit_unfair():
    v, _ = parity_verdict(_counts([0.2, 0.9], _HUGE), c=0.8, delta=0.05)
    assert v is False


def test_unit_fair():
    v, _ = parity_verdict(_counts([0.85, 0.9], _HUGE), c=0.8, delta=0.05)   # ratio 0.94
    assert v is True


def test_unit_inconclusive():
    # near the boundary with a tiny sample: the intervals straddle the threshold.
    v, _ = parity_verdict(_counts([0.78, 0.98], 40), c=0.8, delta=0.05)
    assert v is None


# ------------------ difference-form  <=>  ratio 80%-rule --------------------

def test_difference_form_equivalent_to_ratio():
    """A *definitive* verdict must equal the exact ratio test. CP intervals
    bracket the point estimate, so any non-inconclusive verdict agrees with
    min/max >= c exactly; near the boundary the verdict may read inconclusive
    (the band scales with 1/max-rate, so small rates widen it) -- that is
    correct, not a mismatch, so only the definitive cases are checked."""
    rng = np.random.default_rng(SEED)
    for _ in range(3000):
        k = int(rng.integers(2, 5))
        rates = rng.uniform(0.05, 0.95, k)
        v, _ = parity_verdict(_counts(rates, _HUGE), c=0.8, delta=0.05)
        if v is not None:
            ratio = rates.min() / rates.max()
            assert v is bool(ratio >= 0.8)         # bool() -> Python singleton for `is`


def test_determinism():
    counts = _counts([0.4, 0.6], 500)
    assert parity_verdict(counts, c=0.8, delta=0.05) == parity_verdict(counts, c=0.8, delta=0.05)


# --------------------------- Monte-Carlo soundness --------------------------

def _naive_point_verdict(counts, c):
    """Unsound baseline: compare point-estimate rates with no confidence interval."""
    rates = [s / n for s, n in counts]
    return min(rates) / max(rates) >= c


@pytest.mark.slow
@pytest.mark.parametrize("rates,truly_fair", [
    ([0.71, 0.90], False),           # ratio 0.79 < 0.8  -> truly unfair
    ([0.90, 0.72], False),           # order shouldn't matter
    ([0.90, 0.90], True),            # ratio 1.0         -> truly fair
    ([0.83, 0.99], True),            # ratio 0.84        -> truly fair
    ([0.60, 0.70, 0.80, 0.90], False),   # k=4 union bound, ratio 0.667
])
def test_soundness_definitive_error_below_delta(rates, truly_fair):
    """P(certify the WRONG side) <= delta over repeated sampling."""
    delta, n, M = 0.05, 150, 4000
    rng = np.random.default_rng(SEED)
    wrong = 0
    for _ in range(M):
        counts = [(int(rng.binomial(n, p)), n) for p in rates]
        v, _ = parity_verdict(counts, c=0.8, delta=delta)
        if v is not None and v is not truly_fair:
            wrong += 1
    assert wrong / M <= delta


@pytest.mark.slow
def test_sound_vs_naive_on_a_boundary_case():
    """Power / mutation check, side by side on the SAME data. Truth is *just*
    fair (rates 0.73 / 0.90, ratio ~0.811 >= 0.8). The sound CP verdict certifies
    the wrong side <= delta of the time; the naive point-estimate verdict (no
    intervals) is wrong far more often -- so the soundness test is not vacuous and
    the intervals are doing real work."""
    rates, delta, n, M = [0.73, 0.90], 0.05, 150, 4000       # ratio 0.811 -> truly fair
    rng = np.random.default_rng(SEED)
    sound_wrong = naive_wrong = 0
    for _ in range(M):
        counts = [(int(rng.binomial(n, p)), n) for p in rates]
        v, _ = parity_verdict(counts, c=0.8, delta=delta)
        if v is False:                        # truth is fair, so `unfair` is wrong
            sound_wrong += 1
        if not _naive_point_verdict(counts, 0.8):
            naive_wrong += 1
    assert sound_wrong / M <= delta           # sound: definitive error within budget
    assert naive_wrong / M > delta            # naive: violates the guarantee
