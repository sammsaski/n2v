"""Unit tests for flow conformal calibration."""

import pytest
import torch
import numpy as np
from scipy.stats import beta


class TestCalibrate:
    """Tests for calibrate function."""

    def test_returns_ell_th_smallest(self):
        """Calibrate should return the ell-th order statistic."""
        from n2v.probabilistic.flow.calibrate import calibrate

        scores = torch.tensor([5.0, 3.0, 1.0, 4.0, 2.0])
        # sorted: [1, 2, 3, 4, 5]
        assert calibrate(scores, ell=1).item() == pytest.approx(1.0)
        assert calibrate(scores, ell=3).item() == pytest.approx(3.0)
        assert calibrate(scores, ell=5).item() == pytest.approx(5.0)

    def test_returns_tensor(self):
        """Calibrate should return a scalar tensor."""
        from n2v.probabilistic.flow.calibrate import calibrate

        scores = torch.tensor([1.0, 2.0, 3.0])
        result = calibrate(scores, ell=2)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_ell_equals_m(self):
        """ell=m should return the maximum score."""
        from n2v.probabilistic.flow.calibrate import calibrate

        scores = torch.tensor([10.0, 1.0, 5.0])
        assert calibrate(scores, ell=3).item() == pytest.approx(10.0)


class TestComputeGuarantee:
    """Tests for compute_guarantee function."""

    def test_coverage_is_one_minus_epsilon(self):
        """Coverage should be 1 - epsilon."""
        from n2v.probabilistic.flow.calibrate import compute_guarantee

        coverage, _ = compute_guarantee(m=1000, ell=999, epsilon=0.01)
        assert coverage == pytest.approx(0.99)

    def test_confidence_matches_beta_cdf(self):
        """Confidence should match the beta CDF formula."""
        from n2v.probabilistic.flow.calibrate import compute_guarantee

        m, ell, epsilon = 8000, 7999, 0.001
        _, confidence = compute_guarantee(m, ell, epsilon)
        expected = 1 - beta.cdf(1 - epsilon, ell, m + 1 - ell)
        assert confidence == pytest.approx(expected, abs=1e-10)

    def test_typical_values(self):
        """m=8000, ell=7999, epsilon=0.001 should give confidence ~0.997."""
        from n2v.probabilistic.flow.calibrate import compute_guarantee

        coverage, confidence = compute_guarantee(m=8000, ell=7999, epsilon=0.001)
        assert coverage == pytest.approx(0.999)
        assert 0.99 < confidence < 1.0

    def test_returns_tuple_of_floats(self):
        """Should return a tuple of two floats."""
        from n2v.probabilistic.flow.calibrate import compute_guarantee

        result = compute_guarantee(m=100, ell=99, epsilon=0.05)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestScoreAgnosticEmpiricalCoverage:
    """Cornerstone validity check: marginal empirical coverage must hit the
    target 1 - alpha for any valid score function. A failure here invalidates
    the conformal wrapper itself, not just a specific method.

    Marginal coverage is the average over both calibration and test randomness,
    which is the quantity conformal theory guarantees to be >= 1 - alpha. We
    check the average over many (calibration, test) draws, NOT per-trial
    floor crossings (which have Beta-distributed calibration variance that
    will produce legitimate per-trial misses at a 3-sigma test-set floor).
    """

    def test_multiple_scores_achieve_marginal_coverage(self):
        """1D toy: pushforward = N(0, 1). Score by ball and hyperrect. Both
        must hit marginal empirical coverage >= 1 - alpha within Bernoulli SE."""
        import math

        from n2v.probabilistic.flow.calibrate import calibrate
        from n2v.probabilistic.flow.scores import BallScore, HyperrectScore

        d = 1
        alpha = 0.01
        m = 2000
        n_test = 5000
        n_trials = 40
        # Total pooled test samples across all trials.
        total_samples = n_trials * n_test
        # Bernoulli SE on a 1 - alpha mean estimate.
        se = math.sqrt(alpha * (1 - alpha) / total_samples)
        floor = (1 - alpha) - 3 * se

        for score_name, make_score in [
            ('ball', lambda: BallScore(center=torch.zeros(d))),
            ('hyperrect', lambda: HyperrectScore(
                center=torch.zeros(d), scales=torch.ones(d))),
        ]:
            total_covered = 0
            for trial in range(n_trials):
                gen_cal = torch.Generator().manual_seed(trial)
                gen_test = torch.Generator().manual_seed(trial + 10_000)
                calib = torch.randn(m, d, generator=gen_cal)
                test = torch.randn(n_test, d, generator=gen_test)
                score_fn = make_score()
                calib_scores = score_fn(calib)
                ell = math.ceil((m + 1) * (1 - alpha))
                threshold = calibrate(calib_scores, ell).item()
                test_scores = score_fn(test)
                total_covered += int((test_scores <= threshold).sum().item())
            mean_cov = total_covered / total_samples
            assert mean_cov >= floor, (
                f"score={score_name}: marginal coverage {mean_cov:.5f} "
                f"below floor {floor:.5f} (target {1 - alpha:.5f})"
            )
