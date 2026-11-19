"""Phase 1.3: Monte-Carlo volume estimator correctness on closed-form shapes."""

import math
import pytest
import torch

from n2v.probabilistic.flow.scores import (
    BallScore, HyperrectScore, EllipsoidScore,
)
from n2v.probabilistic.flow.sets import ProbabilisticSet


def _set_with_score(score_fn, threshold, dim):
    """Small constructor: wrap a NonconformityScore in ProbabilisticSet
    with dummy m/ell/epsilon since we only use estimate_volume."""
    return ProbabilisticSet(
        score_fn=score_fn, threshold=threshold,
        m=100, ell=99, epsilon=0.01, dim=dim,
    )


class TestHyperrectVolume:
    def test_axis_aligned_box_2d(self):
        score = HyperrectScore(center=torch.zeros(2), scales=torch.tensor([1.5, 0.8]))
        s = _set_with_score(score, threshold=1.0, dim=2)
        bbox = (torch.tensor([-2.0, -1.0]), torch.tensor([2.0, 1.0]))
        vol, se = s.estimate_volume(n_samples=200_000, bounding_box=bbox)
        expected = 2 * 1.0 * 1.5 * 2 * 1.0 * 0.8  # (2 q tau_1)(2 q tau_2)
        assert abs(vol - expected) < 3 * se, f"got {vol} +/- {se}, expected {expected}"


class TestBallVolume:
    def test_unit_ball_2d(self):
        score = BallScore(center=torch.zeros(2))
        s = _set_with_score(score, threshold=1.0, dim=2)
        bbox = (-torch.ones(2) * 1.2, torch.ones(2) * 1.2)
        vol, se = s.estimate_volume(n_samples=200_000, bounding_box=bbox)
        expected = math.pi  # pi r^2 with r=1
        assert abs(vol - expected) < 3 * se

    def test_unit_ball_3d(self):
        score = BallScore(center=torch.zeros(3))
        s = _set_with_score(score, threshold=1.0, dim=3)
        bbox = (-torch.ones(3) * 1.2, torch.ones(3) * 1.2)
        vol, se = s.estimate_volume(n_samples=500_000, bounding_box=bbox)
        expected = 4.0 / 3 * math.pi
        assert abs(vol - expected) < 3 * se


class TestEllipsoidVolume:
    def test_axis_ellipsoid_2d(self):
        sigmas = torch.tensor([2.0, 0.5])
        cov = torch.diag(sigmas ** 2)
        score = EllipsoidScore(center=torch.zeros(2), cov_inv=torch.linalg.inv(cov))
        s = _set_with_score(score, threshold=1.0, dim=2)
        bbox = (-torch.tensor([2.5, 0.7]), torch.tensor([2.5, 0.7]))
        vol, se = s.estimate_volume(n_samples=400_000, bounding_box=bbox)
        # Volume of {x : x^T Sigma^-1 x <= q^2} in R^d is
        # pi^{d/2} sqrt(det Sigma) q^d / Gamma(d/2+1)
        expected = math.pi * sigmas.prod().item()  # d=2
        assert abs(vol - expected) < 3 * se, f"got {vol}, expected {expected}"


class TestDeterminismAndConvergence:
    def test_seed_determinism(self):
        score = BallScore(center=torch.zeros(2))
        s = _set_with_score(score, threshold=1.0, dim=2)
        bbox = (-torch.ones(2), torch.ones(2))
        torch.manual_seed(0)
        v1, _ = s.estimate_volume(n_samples=10_000, bounding_box=bbox)
        torch.manual_seed(0)
        v2, _ = s.estimate_volume(n_samples=10_000, bounding_box=bbox)
        assert v1 == v2, f"{v1} != {v2} — determinism broken"

    def test_se_shrinks_with_N(self):
        score = BallScore(center=torch.zeros(2))
        s = _set_with_score(score, threshold=1.0, dim=2)
        bbox = (-torch.ones(2), torch.ones(2))
        torch.manual_seed(0)
        _, se_small = s.estimate_volume(n_samples=10_000, bounding_box=bbox)
        torch.manual_seed(0)
        _, se_big = s.estimate_volume(n_samples=100_000, bounding_box=bbox)
        # 10x more samples -> ~sqrt(10) smaller SE
        assert se_big < se_small * 0.5, f"SE didn't shrink: {se_big} vs {se_small}"


# ---------------------------------------------------------------------------
# Off-centre boxes and the default sampling box
# ---------------------------------------------------------------------------
from scipy.stats import binom  # noqa: E402

from tests.known_bugs import known_bug  # noqa: E402


def _halfwidth(n, p, alpha=1e-6):
    """Half-width of an interval that holds the share of a Bin(n, p) count
    with probability at least 1 - alpha."""
    lo, hi = binom.ppf(alpha / 2, n, p), binom.isf(alpha / 2, n, p)
    return max(n * p - lo, hi - n * p) / n


class TestBoxNotCentredAtTheOrigin:
    def test_volume_and_standard_error_of_a_disc_in_an_off_centre_box(self):
        """The unit disc at (1.5, 0.5) in the box [0.25, 2.75] x
        [-0.75, 1.75] (volume 6.25): the estimate lies in the two-sided 1e-6
        binomial interval of pi, and the standard error is the Bernoulli one
        at the sampled share p = vol / 6.25, sqrt(p (1 - p) / n) times the
        box volume, within 1e-3 relative (the n / (n - 1) of an unbiased
        standard deviation and float32 summation stay far below it; no
        randomness enters this second check). The tests above use boxes
        symmetric about the origin and the code's own standard error as
        tolerance. Catches samples drawn from the box reflected through the
        origin, the sum of the widths as box volume, and a standard error
        divided by n instead of sqrt(n) or inflated."""
        n = 200_000
        score = BallScore(center=torch.tensor([1.5, 0.5]))
        s = _set_with_score(score, threshold=1.0, dim=2)
        bbox = (torch.tensor([0.25, -0.75]), torch.tensor([2.75, 1.75]))
        vol, se = s.estimate_volume(n_samples=n, bounding_box=bbox)
        box_volume = 6.25
        p = math.pi / box_volume
        assert abs(vol - math.pi) <= _halfwidth(n, p) * box_volume, vol
        p_hat = vol / box_volume
        se_expected = math.sqrt(p_hat * (1 - p_hat) / n) * box_volume
        assert abs(se - se_expected) <= 1e-3 * se_expected, (se, se_expected)


class TestDefaultSamplingBox:
    def test_default_box_gives_the_volume_of_a_set_around_the_origin(self):
        """Without a bounding box estimate_volume samples the cube
        [-3q, 3q]^d; for the disc of radius q = 0.75 at the origin the
        estimate lies in the two-sided 1e-6 binomial interval of its area
        and the standard error is the Bernoulli one at the sampled share
        vol / (6q)**2 within 1e-3 relative (a deterministic check, as in the
        case above). Catches the cube's volume taken as r**d instead of
        (2r)**d."""
        n, q = 200_000, 0.75
        s = _set_with_score(BallScore(center=torch.zeros(2)), threshold=q, dim=2)
        vol, se = s.estimate_volume(n_samples=n)
        area, cube = math.pi * q * q, (6 * q) ** 2
        p = area / cube
        assert abs(vol - area) <= _halfwidth(n, p) * cube, (vol, area)
        p_hat = vol / cube
        se_expected = math.sqrt(p_hat * (1 - p_hat) / n) * cube
        assert abs(se - se_expected) <= 1e-3 * se_expected, (se, se_expected)

    @known_bug(13, "without a bounding box estimate_volume samples the cube "
                   "[-3q, 3q]^d around the origin whatever the set's centre, "
                   "so a set away from the origin is given volume 0 with "
                   "standard error 0")
    def test_default_box_gives_the_volume_of_a_set_away_from_the_origin(self):
        """The disc of radius 1 at (10, 10) has area pi; the estimate must
        be within five of its own (positive) standard errors of it. Today
        the default cube [-3, 3]^2 misses the disc and the answer is
        0 +/- 0."""
        s = _set_with_score(BallScore(center=torch.tensor([10.0, 10.0])),
                            threshold=1.0, dim=2)
        vol, se = s.estimate_volume(n_samples=200_000)
        assert se > 0 and abs(vol - math.pi) <= 5 * se, (vol, se)
