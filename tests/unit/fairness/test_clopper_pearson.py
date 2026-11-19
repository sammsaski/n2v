"""Soundness of the Clopper-Pearson interval (the bound we swapped in for
VeriFair's adaptive concentration).

`clopper_pearson` is implemented with the Beta-quantile form. We validate it two
ways that do NOT reuse that form:

  * an independent ORACLE using the exact binomial tail (scipy.stats.binom): the
    CP limits are, by definition, the p-values solving the binomial tail
    equalities, so we check those identities directly;
  * Monte-Carlo COVERAGE: the interval must contain the true p at least 1 - alpha
    of the time, for every p (CP is exact/conservative).
"""
import numpy as np
import pytest
from scipy.stats import binom

from tests.unit.fairness.conftest import SEED
from intervals import clopper_pearson


@pytest.mark.parametrize("s,n,alpha", [
    (0, 10, 0.05), (1, 10, 0.05), (5, 20, 0.10), (37, 100, 0.05),
    (99, 100, 0.05), (100, 100, 0.05), (3, 50, 0.01), (250, 500, 0.05),
])
def test_binomial_tail_oracle(s, n, alpha):
    """CP defining identities via the binomial tail (independent of our Beta impl):
      lower limit l:  P(Bin(n, l) >= s) = alpha/2
      upper limit u:  P(Bin(n, u) <= s) = alpha/2
    with the s=0 / s=n endpoints clamped to 0 / 1.
    """
    lo, hi = clopper_pearson(s, n, alpha)
    assert 0.0 <= lo <= hi <= 1.0
    if s == 0:
        assert lo == 0.0
    else:
        assert binom.sf(s - 1, n, lo) == pytest.approx(alpha / 2, abs=1e-6)
    if s == n:
        assert hi == 1.0
    else:
        assert binom.cdf(s, n, hi) == pytest.approx(alpha / 2, abs=1e-6)


def test_empty_cell_is_vacuous():
    """n == 0 (an empty real-data cell) must return [0, 1] -- no data, no
    information -- so the gap/ratio tests read it as inconclusive rather than
    dividing by zero. The real-count fallback relies on this."""
    assert clopper_pearson(0, 0, 0.05) == (0.0, 1.0)


@pytest.mark.parametrize("n", [10, 50, 200])
def test_contains_point_estimate_and_monotone(n):
    """The interval brackets s/n, and both limits are non-decreasing in s."""
    prev_lo, prev_hi = -1.0, -1.0
    for s in range(n + 1):
        lo, hi = clopper_pearson(s, n, 0.05)
        assert lo <= s / n <= hi
        assert lo >= prev_lo and hi >= prev_hi     # monotone in s
        prev_lo, prev_hi = lo, hi


@pytest.mark.slow
@pytest.mark.parametrize("p", [0.02, 0.1, 0.3, 0.5, 0.8, 0.95])
def test_coverage_monte_carlo(p):
    """Empirical coverage >= 1 - alpha for every p (CP is conservative). The 0.01
    slack absorbs Monte-Carlo noise (se ~ 0.0015 at M=20000); true CP coverage is
    >= 1-alpha exactly, so this only fails if the interval is genuinely too tight.
    """
    n, alpha, M = 80, 0.05, 20_000
    rng = np.random.default_rng(SEED)
    s = rng.binomial(n, p, M)
    covered = sum(lo <= p <= hi for lo, hi in (clopper_pearson(si, n, alpha) for si in s))
    coverage = covered / M
    assert coverage >= 1 - alpha - 0.01
