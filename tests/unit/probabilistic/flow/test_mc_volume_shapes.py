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
