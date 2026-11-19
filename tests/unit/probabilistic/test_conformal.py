"""
Unit tests for conformal inference module.
"""

from fractions import Fraction
from math import comb

import pytest
import numpy as np
from scipy.stats import beta

from n2v.probabilistic.conformal import (
    ConformalGuarantee,
    compute_confidence,
    compute_normalization,
    compute_nonconformity_scores,
    compute_threshold,
    compute_inflation,
    conformal_inference
)


def _exact_confidence(m, ell, epsilon):
    """P[Binomial(m, 1 - epsilon) <= ell - 1] in rationals, ``epsilon`` a
    decimal string: the confidence 1 - BetaCDF(1 - epsilon; ell, m + 1 - ell)
    written as a finite sum (the shorter tail is summed)."""
    eps = Fraction(epsilon)
    q = 1 - eps
    if ell - 1 <= m // 2:
        return sum(comb(m, j) * q ** j * eps ** (m - j) for j in range(ell))
    return 1 - sum(comb(m, j) * q ** j * eps ** (m - j)
                   for j in range(ell, m + 1))


class TestComputeConfidence:
    """Tests for compute_confidence function."""

    def test_compute_confidence_basic(self):
        """Test basic confidence computation."""
        m = 1000
        ell = 999
        epsilon = 0.01

        result = compute_confidence(m, ell, epsilon)
        expected = 1 - beta.cdf(1 - epsilon, ell, m + 1 - ell)

        assert abs(result - expected) < 1e-10

    def test_compute_confidence_typical_values(self):
        """Test confidence for typical parameter combinations from papers."""
        # From the papers: m=8000, ell=7999, epsilon=0.001 gives ~0.997
        result = compute_confidence(m=8000, ell=7999, epsilon=0.001)
        assert 0.99 < result < 1.0

        # m=100000, ell=99999, epsilon=0.0001 gives ~0.9995
        result = compute_confidence(m=100000, ell=99999, epsilon=0.0001)
        assert 0.999 < result < 1.0

    def test_compute_confidence_ell_equals_m(self):
        """Test confidence when ell = m (most conservative)."""
        result = compute_confidence(m=100, ell=100, epsilon=0.05)
        expected = 1 - beta.cdf(0.95, 100, 1)

        assert abs(result - expected) < 1e-10

    def test_compute_confidence_returns_float(self):
        """Test that result is a float."""
        result = compute_confidence(m=100, ell=99, epsilon=0.01)
        assert isinstance(result, float)

    @pytest.mark.parametrize("m, ell, epsilon", [
        (1, 1, "0.5"), (10, 1, "0.1"), (10, 10, "0.1"), (10, 9, "0.25"),
        (50, 25, "0.5"), (100, 100, "0.05"), (200, 199, "0.01"),
        (1000, 990, "0.01"), (8000, 7999, "0.001"),
    ])
    def test_confidence_is_the_exact_binomial_tail(self, m, ell, epsilon):
        """The confidence of rank ell among m calibration scores at
        miscoverage epsilon is P[Binomial(m, 1 - epsilon) <= ell - 1], summed
        here in rationals (independent of scipy's Beta function). The grid
        holds ell = 1, ell = m, ell = m - 1 and mid ranks, small and large
        epsilon. Catches a Beta parameter off by one (m - ell for m + 1 - ell,
        which overstates the confidence and is NaN at ell = m), swapped Beta
        parameters, and epsilon used where 1 - epsilon belongs. Slack 1e-12:
        float64 evaluation of a value in [0, 1]."""
        got = compute_confidence(m, ell, float(epsilon))
        assert abs(got - float(_exact_confidence(m, ell, epsilon))) <= 1e-12


class TestComputeNormalization:
    """Tests for compute_normalization function."""

    def test_normalization_basic(self):
        """Test basic normalization computation."""
        training_errors = np.array([
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5],
            [0.2, 0.8, 1.2]
        ])

        tau = compute_normalization(training_errors)

        # tau[k] = max(tau*, max_j |training_errors[j, k]|)
        # Max abs values per dimension: [1.0, 2.0, 3.0]
        assert tau.shape == (3,)
        assert tau[0] >= 1.0
        assert tau[1] >= 2.0
        assert tau[2] >= 3.0

    def test_normalization_prevents_zero_division(self):
        """Test that tau has minimum value to prevent division by zero."""
        # All zeros
        training_errors = np.zeros((10, 5))

        tau = compute_normalization(training_errors)

        # Should have non-zero values
        assert np.all(tau > 0)

    def test_normalization_with_negative_values(self):
        """Test normalization with negative errors (uses abs)."""
        training_errors = np.array([
            [-1.0, 2.0],
            [0.5, -3.0],
        ])

        tau = compute_normalization(training_errors)

        # Max abs: [1.0, 3.0]
        assert tau[0] >= 1.0
        assert tau[1] >= 3.0

    def test_normalization_1d_input(self):
        """Test that 1D input is handled correctly."""
        training_errors = np.array([1.0, 2.0, 3.0])

        tau = compute_normalization(training_errors)

        assert tau.shape == (3,)

    def test_normalization_empty_raises_error(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_normalization(np.array([]))

    def test_normalization_unions_train_and_calib(self):
        """tau[k] = max over train + calib errors per component (Paper 1 eq 6)."""
        training_errors = np.array([
            [1.0, 0.0],
            [0.5, 0.0],
        ])
        calibration_errors = np.array([
            [0.0, 3.0],
            [0.0, 1.5],
        ])
        tau = compute_normalization(training_errors, calibration_errors)
        # Max over both: dim 0 max = 1.0 (train), dim 1 max = 3.0 (calib)
        assert tau[0] >= 1.0
        assert tau[1] >= 3.0

    def test_normalization_clipping_block_scenario(self):
        """Regression test for Finding 1: zero training errors must not
        collapse tau to the 1e-10 floor when calibration errors are provided."""
        training_errors = np.zeros((50, 2))  # clipping block scenario
        calibration_errors = np.column_stack([
            np.random.RandomState(0).randn(100) * 1.0,
            np.random.RandomState(1).randn(100) * 0.1,
        ])
        tau = compute_normalization(training_errors, calibration_errors)
        # tau should reflect the per-component calibration error scale,
        # not the 1e-10 floor.
        assert tau[0] > 0.5  # y_1 scale ~ 1.0
        assert tau[1] > 0.05  # y_2 scale ~ 0.1
        assert tau[1] < 0.5  # y_2 should NOT be inflated to the y_1 scale


class TestComputeNonconformityScores:
    """Tests for compute_nonconformity_scores function."""

    def test_nonconformity_scores_basic(self):
        """Test basic nonconformity score computation."""
        prediction_errors = np.array([
            [1.0, 2.0],
            [0.5, 1.0],
            [2.0, 0.5]
        ])
        tau = np.array([1.0, 1.0])

        scores = compute_nonconformity_scores(prediction_errors, tau)

        # R_i = max_k(|q_i[k]| / tau[k])
        # R_0 = max(1.0/1.0, 2.0/1.0) = 2.0
        # R_1 = max(0.5/1.0, 1.0/1.0) = 1.0
        # R_2 = max(2.0/1.0, 0.5/1.0) = 2.0
        expected = np.array([2.0, 1.0, 2.0])

        np.testing.assert_array_almost_equal(scores, expected)

    def test_nonconformity_scores_with_normalization(self):
        """Test scores with non-unit normalization."""
        prediction_errors = np.array([
            [2.0, 4.0],
            [1.0, 2.0]
        ])
        tau = np.array([2.0, 4.0])

        scores = compute_nonconformity_scores(prediction_errors, tau)

        # R_0 = max(2.0/2.0, 4.0/4.0) = max(1.0, 1.0) = 1.0
        # R_1 = max(1.0/2.0, 2.0/4.0) = max(0.5, 0.5) = 0.5
        expected = np.array([1.0, 0.5])

        np.testing.assert_array_almost_equal(scores, expected)

    def test_nonconformity_scores_with_center(self):
        """Test scores with non-zero center."""
        prediction_errors = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        tau = np.array([1.0, 1.0])
        center = np.array([1.0, 2.0])

        scores = compute_nonconformity_scores(prediction_errors, tau, center)

        # R_0 = max(|1.0-1.0|/1.0, |2.0-2.0|/1.0) = max(0, 0) = 0
        # R_1 = max(|3.0-1.0|/1.0, |4.0-2.0|/1.0) = max(2, 2) = 2
        expected = np.array([0.0, 2.0])

        np.testing.assert_array_almost_equal(scores, expected)

    @pytest.mark.parametrize("center, expected", [
        (None, [3.0, 2.0]),
        ([-1.0, 0.25], [2.0, 2.5]),
    ], ids=["no-center", "center"])
    def test_negative_errors_count_by_their_magnitude(self, center, expected):
        """R_i = max_k |q_i[k] - c[k]| / tau[k], by hand: errors [[-3, 0.5],
        [0.25, -1]], tau [1, 0.5]. The largest normalized error of each row is
        a negative one, so a score without the absolute value (which ignores
        outputs below the surrogate) gives [1, 0.25] or [0.5, 2.5]; a centre
        added instead of subtracted gives [4, 1.5]. Dyadic data: exact."""
        errors = np.array([[-3.0, 0.5], [0.25, -1.0]])
        tau = np.array([1.0, 0.5])
        c = None if center is None else np.array(center)
        scores = compute_nonconformity_scores(errors, tau, c)
        assert scores.tolist() == expected

    def test_nonconformity_scores_1d_input(self):
        """Test that 1D input is handled correctly."""
        prediction_errors = np.array([1.0, 2.0, 3.0])
        tau = np.array([1.0, 1.0, 1.0])

        scores = compute_nonconformity_scores(prediction_errors, tau)

        # Single sample: R = max(1.0, 2.0, 3.0) = 3.0
        assert scores.shape == (1,)
        assert scores[0] == 3.0


class TestComputeThreshold:
    """Tests for compute_threshold function."""

    def test_threshold_basic(self):
        """Test basic threshold selection."""
        scores = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        # Sorted: [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]

        # ell=1 gives smallest (1.0)
        assert compute_threshold(scores, ell=1) == 1.0

        # ell=4 gives 4th smallest (3.0)
        assert compute_threshold(scores, ell=4) == 3.0

        # ell=8 gives largest (9.0)
        assert compute_threshold(scores, ell=8) == 9.0

    def test_threshold_typical_ell_m_minus_1(self):
        """Test threshold with typical ell = m - 1."""
        np.random.seed(42)
        scores = np.random.rand(100)

        # ell = 99 gives second largest
        threshold = compute_threshold(scores, ell=99)
        sorted_scores = np.sort(scores)

        assert threshold == sorted_scores[98]

    def test_threshold_returns_float(self):
        """Test that threshold is a scalar float."""
        scores = np.array([1.0, 2.0, 3.0])
        threshold = compute_threshold(scores, ell=2)

        assert isinstance(threshold, (float, np.floating))


class TestComputeInflation:
    """Tests for compute_inflation function."""

    def test_inflation_basic(self):
        """Test basic inflation computation."""
        tau = np.array([1.0, 2.0, 3.0])
        threshold = 2.0

        inflation = compute_inflation(tau, threshold)

        # sigma[k] = tau[k] * threshold
        expected = np.array([2.0, 4.0, 6.0])

        np.testing.assert_array_almost_equal(inflation, expected)

    def test_inflation_preserves_shape(self):
        """Test that inflation has same shape as tau."""
        tau = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        threshold = 1.5

        inflation = compute_inflation(tau, threshold)

        assert inflation.shape == tau.shape


class TestConformalInference:
    """Tests for complete conformal_inference function."""

    def test_conformal_inference_basic(self):
        """Test complete conformal inference."""
        np.random.seed(42)

        t, n = 50, 10  # Training samples and output dimension
        m = 100  # Calibration samples
        ell = 99
        epsilon = 0.01

        training_errors = np.random.randn(t, n) * 0.1
        calibration_errors = np.random.randn(m, n) * 0.1

        guarantee = conformal_inference(
            training_errors, calibration_errors,
            m=m, ell=ell, epsilon=epsilon
        )

        # Check guarantee attributes
        assert isinstance(guarantee, ConformalGuarantee)
        assert guarantee.m == m
        assert guarantee.ell == ell
        assert guarantee.epsilon == epsilon
        assert guarantee.coverage == 1 - epsilon
        assert guarantee.threshold > 0
        assert guarantee.inflation.shape == (n,)
        assert np.all(guarantee.inflation >= 0)

    def test_conformal_inference_confidence_matches(self):
        """Test that confidence in result matches compute_confidence."""
        np.random.seed(42)

        training_errors = np.random.randn(50, 5)
        calibration_errors = np.random.randn(100, 5)

        m, ell, epsilon = 100, 99, 0.01

        guarantee = conformal_inference(
            training_errors, calibration_errors,
            m=m, ell=ell, epsilon=epsilon
        )

        expected_confidence = compute_confidence(m, ell, epsilon)

        assert abs(guarantee.confidence - expected_confidence) < 1e-10

    def test_conformal_inference_validates_m(self):
        """Test that conformal_inference validates m matches calibration size."""
        training_errors = np.random.randn(50, 5)
        calibration_errors = np.random.randn(100, 5)

        # m doesn't match calibration_errors.shape[0]
        with pytest.raises(ValueError, match="expected m="):
            conformal_inference(
                training_errors, calibration_errors,
                m=50,  # Wrong!
                ell=49, epsilon=0.01
            )

    def test_conformal_inference_validates_ell(self):
        """Test that conformal_inference validates ell."""
        training_errors = np.random.randn(50, 5)
        calibration_errors = np.random.randn(100, 5)

        with pytest.raises(ValueError, match="ell must be in"):
            conformal_inference(
                training_errors, calibration_errors,
                m=100, ell=101, epsilon=0.01  # ell > m
            )

    def test_conformal_inference_validates_epsilon(self):
        """Test that conformal_inference validates epsilon."""
        training_errors = np.random.randn(50, 5)
        calibration_errors = np.random.randn(100, 5)

        with pytest.raises(ValueError, match="epsilon must be in"):
            conformal_inference(
                training_errors, calibration_errors,
                m=100, ell=99, epsilon=0.0  # epsilon = 0
            )

    def test_conformal_inference_1d_inputs(self):
        """Test that 1D inputs are handled correctly."""
        training_errors = np.random.randn(5)  # 1D: 5 output dimensions
        calibration_errors = np.random.randn(10, 5)  # Need (10, 5)

        # This should work since 1D training_errors is reshaped to (1, 5)
        guarantee = conformal_inference(
            training_errors, calibration_errors,
            m=10, ell=9, epsilon=0.1
        )

        assert guarantee.inflation.shape == (5,)


class TestConformalInferenceByHand:
    """The whole pipeline of conformal_inference on errors computed by hand."""

    # tau = [2, 4]: 2 from the calibration errors, 4 from the training errors.
    TRAIN = np.array([[0.5, -4.0], [-0.25, 2.0]])
    CALIB = np.array([[1.0, -0.5], [-2.0, 1.0], [0.25, 3.0], [0.5, -1.5],
                      [-1.25, 0.5]])
    # scores max(|e0| / 2, |e1| / 4) = [0.5, 1.0, 0.75, 0.375, 0.625]

    @pytest.mark.parametrize("ell, threshold", [
        (1, 0.375), (2, 0.5), (4, 0.75), (5, 1.0),
    ])
    def test_threshold_inflation_and_confidence_by_hand(self, ell, threshold):
        """tau[k] = max over training and calibration errors of |e[k]|
        (today's formula, deferred PROB-6) = [2, 4]; the scores are
        [0.5, 1, 0.75, 0.375, 0.625]; the threshold is the ell-th smallest
        (ell 1-based); the inflation is tau times it; the confidence is the
        binomial tail for (m = 5, ell, epsilon = 0.25). Dyadic data, so the
        threshold and the inflation are exact. Catches the rank handed on as
        ell - 1 (the largest score at ell = 1), a descending sort, an
        inflation without tau, scores that average or drop the absolute
        value, tau from the smaller of the two maxima, and a confidence
        computed with the coverage in place of epsilon or with another rank."""
        g = conformal_inference(self.TRAIN, self.CALIB, m=5, ell=ell,
                                epsilon=0.25)
        assert g.threshold == threshold
        assert g.inflation.tolist() == [2.0 * threshold, 4.0 * threshold]
        assert (g.m, g.ell, g.epsilon, g.coverage) == (5, ell, 0.25, 0.75)
        assert abs(g.confidence - float(_exact_confidence(5, ell, "0.25"))) \
            <= 1e-12

    def test_one_dimensional_calibration_errors_are_one_sample(self):
        """A 1-D calibration error array is one sample of n outputs (m = 1).
        tau = max(|train|, |calib|) per output = [1, 2, 0.5]; the one score is
        max(0.5 / 1, 2 / 2, 0.25 / 0.5) = 1. Catches a 1-D array read as a
        column of m samples, which refuses the legal call for m = 1 or mixes
        the outputs."""
        train = np.array([[1.0, 0.0, 0.5], [0.0, 0.25, 0.0]])
        calib = np.array([0.5, -2.0, -0.25])
        g = conformal_inference(train, calib, m=1, ell=1, epsilon=0.5)
        assert g.threshold == 1.0
        assert g.inflation.tolist() == [1.0, 2.0, 0.5]

    @pytest.mark.parametrize("ell, epsilon", [
        (0, 0.1), (6, 0.1), (5, 1.0), (5, -0.25),
    ], ids=["ell-0", "ell-m-plus-1", "epsilon-1", "epsilon-negative"])
    def test_rank_and_level_outside_their_range_are_refused(self, ell, epsilon):
        """ell must be in [1, m] and epsilon in (0, 1); each end is refused
        with ValueError. Catches a check that lets ell = 0 through (the
        threshold is then the largest score, read from index -1, with a
        confidence for rank 0) or that accepts epsilon = 1 or a negative
        epsilon."""
        with pytest.raises(ValueError, match="ell must be in|epsilon must be in"):
            conformal_inference(self.TRAIN, self.CALIB, m=5, ell=ell,
                                epsilon=epsilon)


class TestConformalGuaranteeDataclass:
    """Tests for ConformalGuarantee dataclass."""

    def test_dataclass_fields(self):
        """Test that dataclass has expected fields."""
        guarantee = ConformalGuarantee(
            m=100,
            ell=99,
            epsilon=0.01,
            coverage=0.99,
            confidence=0.95,
            threshold=1.5,
            inflation=np.array([1.0, 2.0, 3.0])
        )

        assert guarantee.m == 100
        assert guarantee.ell == 99
        assert guarantee.epsilon == 0.01
        assert guarantee.coverage == 0.99
        assert guarantee.confidence == 0.95
        assert guarantee.threshold == 1.5
        np.testing.assert_array_equal(guarantee.inflation, np.array([1.0, 2.0, 3.0]))


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_calibration_sample(self):
        """Test with single calibration sample (m=1, ell=1)."""
        training_errors = np.random.randn(10, 3)
        calibration_errors = np.random.randn(1, 3)

        guarantee = conformal_inference(
            training_errors, calibration_errors,
            m=1, ell=1, epsilon=0.5
        )

        assert guarantee.m == 1
        assert guarantee.inflation.shape == (3,)

    def test_single_dimension(self):
        """Test with single output dimension."""
        training_errors = np.random.randn(50, 1)
        calibration_errors = np.random.randn(100, 1)

        guarantee = conformal_inference(
            training_errors, calibration_errors,
            m=100, ell=99, epsilon=0.01
        )

        assert guarantee.inflation.shape == (1,)

    def test_large_epsilon(self):
        """Test with large epsilon (low coverage)."""
        training_errors = np.random.randn(50, 5)
        calibration_errors = np.random.randn(100, 5)

        guarantee = conformal_inference(
            training_errors, calibration_errors,
            m=100, ell=99, epsilon=0.5
        )

        assert guarantee.coverage == 0.5

    def test_ell_equals_1(self):
        """Test with ell=1 (smallest score)."""
        training_errors = np.random.randn(50, 5)
        calibration_errors = np.random.randn(100, 5)

        guarantee = conformal_inference(
            training_errors, calibration_errors,
            m=100, ell=1, epsilon=0.01
        )

        # ell=1 gives smallest threshold, smallest inflation
        assert guarantee.threshold >= 0
