"""
Conformal calibration for flow-based reachability.

Provides score-agnostic calibration (sort scores, pick threshold)
and the double-step probabilistic guarantee computation.
"""

import torch
from scipy.stats import beta as beta_dist
from typing import Tuple


def calibrate(scores: torch.Tensor, ell: int) -> torch.Tensor:
    """
    Calibrate by selecting the ell-th smallest score as threshold.

    Args:
        scores: (m,) tensor of nonconformity scores.
        ell: Rank parameter (1-indexed). ell=m selects the maximum.

    Returns:
        Scalar tensor: the ell-th smallest score.
    """
    sorted_scores, _ = scores.sort()
    return sorted_scores[ell - 1]


def compute_guarantee(m: int, ell: int, epsilon: float) -> Tuple[float, float]:
    """
    Compute the double-step probabilistic guarantee.

    Pr[Pr[R_unseen <= threshold] > 1-epsilon] > 1 - betacdf(1-eps, ell, m+1-ell)

    Args:
        m: Calibration set size.
        ell: Rank parameter.
        epsilon: Miscoverage level.

    Returns:
        (coverage, confidence) where coverage = 1-epsilon and
        confidence = 1 - betacdf(1-epsilon, ell, m+1-ell).
    """
    coverage = 1.0 - epsilon
    confidence = 1.0 - beta_dist.cdf(coverage, ell, m + 1 - ell)
    return (coverage, float(confidence))
