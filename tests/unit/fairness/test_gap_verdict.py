"""Soundness + correctness of the equalized-odds gap test (`gap_verdict`) and the
notion combiner (`_combine`).

`gap_verdict` certifies |rate_a - rate_b| <= tau for every group pair from CP
intervals. Checked:
  * the three-valued interval logic on constructed cases;
  * MONTE-CARLO SOUNDNESS: a definitive wrong verdict (certifying the wrong side
    of the tau gap) happens with probability <= delta;
  * `_combine` (equalized odds = TPR-parity AND FPR-parity) on its full truth
    table.
"""
import numpy as np
import pytest

from tests.unit.fairness.conftest import SEED
from verify_eqodds import gap_verdict, _combine


def _counts(rates, n):
    return [(int(round(r * n)), n) for r in rates]


_HUGE = 10 ** 7


# --------------------------- interval logic ---------------------------------

def test_unit_fair():
    # gap 0.05 <= tau 0.1, intervals collapsed -> provably fair
    v, _ = gap_verdict(_counts([0.30, 0.35], _HUGE), tau=0.1, alpha=0.025)
    assert v is True


def test_unit_unfair():
    # gap 0.30 > tau 0.1 -> provably unfair
    v, _ = gap_verdict(_counts([0.20, 0.50], _HUGE), tau=0.1, alpha=0.025)
    assert v is False


def test_unit_inconclusive():
    # true gap ~0.1 with a small sample -> intervals straddle tau
    v, _ = gap_verdict(_counts([0.20, 0.30], 60), tau=0.1, alpha=0.025)
    assert v is None


def test_gap_is_symmetric_in_group_order():
    a = gap_verdict(_counts([0.2, 0.6], _HUGE), tau=0.1, alpha=0.025)[0]
    b = gap_verdict(_counts([0.6, 0.2], _HUGE), tau=0.1, alpha=0.025)[0]
    assert a is b is False


# ------------------------------- combiner -----------------------------------

@pytest.mark.parametrize("tpr,fpr,expected", [
    (True, True, True),
    (True, None, None),
    (None, True, None),
    (None, None, None),
    (True, False, False),
    (False, True, False),
    (False, None, False),
    (None, False, False),
    (False, False, False),
])
def test_combine_truth_table(tpr, fpr, expected):
    assert _combine(tpr, fpr) is expected


# --------------------------- Monte-Carlo soundness --------------------------

@pytest.mark.slow
@pytest.mark.parametrize("rates,within_tau", [
    ([0.30, 0.45], False),     # gap 0.15 > tau 0.1 -> truly unfair
    ([0.50, 0.52], True),      # gap 0.02          -> truly fair
    ([0.10, 0.40, 0.45], False),   # k=3, max gap 0.35
])
def test_soundness_definitive_error_below_delta(rates, within_tau):
    """P(certify the WRONG side of the tau gap) <= delta over repeated sampling.
    alpha = delta/k here matches one equalized-odds condition's budget after the
    delta/2 split across TPR and FPR would be applied upstream."""
    tau, delta, n, M = 0.1, 0.05, 400, 4000
    k = len(rates)
    alpha = delta / k
    rng = np.random.default_rng(SEED)
    wrong = 0
    for _ in range(M):
        counts = [(int(rng.binomial(n, p)), n) for p in rates]
        v, _ = gap_verdict(counts, tau=tau, alpha=alpha)
        if v is not None and v is not within_tau:
            wrong += 1
    assert wrong / M <= delta
