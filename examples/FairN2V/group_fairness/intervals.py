"""Exact-binomial confidence interval shared by the group-fairness verifiers.

Both verify_parity and verify_eqodds bound a per-group favorable rate the same
way -- a two-sided Clopper-Pearson interval on the ``(successes, trials)`` count,
whether those trials are flow draws (distributional verdict) or the real rows
(empirical verdict). Kept here so neither verifier has to import from the other.
"""
from __future__ import annotations

from scipy.stats import beta


def clopper_pearson(s: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided exact binomial CI ``[lo, hi]``, coverage >= 1 - alpha.

    ``n == 0`` (an empty cell) returns the vacuous ``[0, 1]`` -- no data, no
    information -- which flows through the gap/ratio tests as inconclusive.
    """
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if s == 0 else beta.ppf(alpha / 2, s, n - s + 1)
    hi = 1.0 if s == n else beta.ppf(1 - alpha / 2, s + 1, n - s)
    return lo, hi
