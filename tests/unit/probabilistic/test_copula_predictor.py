"""
Unit tests for CopulaConformalPredictor class.
"""

import numpy as np
import pytest

from n2v.probabilistic.copula.predictor import CopulaConformalPredictor


def simple_linear_model(x):
    """Simple linear model for testing."""
    return x @ np.array([[1, 0.5], [0.5, 1]]) + np.array([0.1, 0.2])


def correlated_model(x):
    """Model with correlated outputs."""
    y1 = x[:, 0] + 0.5 * x[:, 1]
    y2 = 0.8 * y1 + 0.2 * x[:, 1]  # y2 highly correlated with y1
    return np.column_stack([y1, y2])


class TestCopulaConformalPredictorCalibration:
    """Tests for predictor calibration."""

    def test_basic_calibration(self):
        """Test basic calibration."""
        predictor = CopulaConformalPredictor()

        predictor.calibrate(
            model=simple_linear_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            training_samples=200,
            m=500,
            ell=499,
            epsilon=0.01,
            seed=42
        )

        assert predictor.center is not None
        assert predictor.marginals is not None
        assert predictor.copula is not None
        assert predictor.threshold is not None
        assert predictor.d == 2
        assert predictor.coverage == 0.99
        assert 0 < predictor.confidence < 1

    def test_calibration_with_defaults(self):
        """Test calibration with default parameters."""
        predictor = CopulaConformalPredictor()

        predictor.calibrate(
            model=simple_linear_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            m=100,  # Small for speed
            seed=42
        )

        assert predictor.ell == 99  # Default: m - 1
        assert predictor.epsilon == 0.001  # Default

    def test_1d_output(self):
        """Test with 1D output."""

        def scalar_model(x):
            return x.sum(axis=1, keepdims=True)

        predictor = CopulaConformalPredictor()
        predictor.calibrate(
            model=scalar_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            training_samples=100,
            m=200,
            seed=42
        )

        assert predictor.d == 1

    def test_higher_dimensional_output(self):
        """Test with higher dimensional output."""

        def high_dim_model(x):
            return np.hstack([x, x ** 2, np.sin(x)])

        predictor = CopulaConformalPredictor()
        predictor.calibrate(
            model=high_dim_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            training_samples=300,
            m=500,
            seed=42
        )

        assert predictor.d == 6


class TestCopulaConformalPredictorScoring:
    """Tests for scoring and membership."""

    @pytest.fixture
    def calibrated_predictor(self):
        """Create a calibrated predictor for testing."""
        predictor = CopulaConformalPredictor()
        predictor.calibrate(
            model=simple_linear_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            training_samples=300,
            m=800,
            seed=42
        )
        return predictor

    def test_score_returns_float(self, calibrated_predictor):
        """Test that score returns a float."""
        y = calibrated_predictor.center + np.array([0.1, 0.1])
        score = calibrated_predictor.score(y)
        assert isinstance(score, float)

    def test_score_finite(self, calibrated_predictor):
        """Test that scores are finite."""
        for _ in range(10):
            y = calibrated_predictor.center + np.random.randn(2) * 0.1
            score = calibrated_predictor.score(y)
            assert np.isfinite(score)

    def test_score_center_low(self, calibrated_predictor):
        """Test that center has low score."""
        center_score = calibrated_predictor.score(calibrated_predictor.center)
        # Center should have low score (high density)
        assert center_score < calibrated_predictor.threshold

    def test_score_is_finite_for_nearby_points(self, calibrated_predictor):
        """Test that scores are finite for points near the center."""
        # Points near the center
        for offset in [0.01, 0.1, 0.2]:
            point = calibrated_predictor.center + np.array([offset, offset])
            score = calibrated_predictor.score(point)
            assert np.isfinite(score), f"Score not finite for offset {offset}"

    def test_contains_returns_bool(self, calibrated_predictor):
        """Test that contains returns boolean."""
        y = calibrated_predictor.center
        result = calibrated_predictor.contains(y)
        assert isinstance(result, bool)

    def test_contains_center(self, calibrated_predictor):
        """Test that center is contained."""
        assert calibrated_predictor.contains(calibrated_predictor.center)

    def test_contains_batch(self, calibrated_predictor):
        """Test batch containment check."""
        Y = calibrated_predictor.center + np.random.randn(10, 2) * 0.1
        result = calibrated_predictor.contains_batch(Y)

        assert result.shape == (10,)
        assert result.dtype == bool


class TestCopulaConformalPredictorSampling:
    """Tests for sampling from prediction region."""

    @pytest.fixture
    def calibrated_predictor(self):
        """Create a calibrated predictor for testing."""
        predictor = CopulaConformalPredictor()
        predictor.calibrate(
            model=simple_linear_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            training_samples=300,
            m=800,
            epsilon=0.01,
            seed=42
        )
        return predictor

    def test_sample_shape(self, calibrated_predictor):
        """Test that samples have correct shape."""
        samples = calibrated_predictor.sample(50, seed=42)
        assert samples.shape == (50, 2)

    def test_samples_in_region(self, calibrated_predictor):
        """Test that samples are in the prediction region."""
        samples = calibrated_predictor.sample(100, seed=42)

        for sample in samples:
            assert calibrated_predictor.contains(sample)

    def test_sample_reproducibility(self, calibrated_predictor):
        """Test that seed gives reproducible samples."""
        samples1 = calibrated_predictor.sample(20, seed=123)
        samples2 = calibrated_predictor.sample(20, seed=123)

        np.testing.assert_array_equal(samples1, samples2)


class TestCopulaConformalPredictorGuarantee:
    """Tests for guarantee computation."""

    def test_guarantee_values(self):
        """Test that guarantee values are correct."""
        predictor = CopulaConformalPredictor()
        predictor.calibrate(
            model=simple_linear_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            m=1000,
            ell=999,
            epsilon=0.01,
            seed=42
        )

        assert predictor.coverage == 0.99
        assert 0 < predictor.confidence < 1

        # Get guarantee tuple
        coverage, confidence = predictor.get_guarantee()
        assert coverage == 0.99
        assert confidence == predictor.confidence

    def test_guarantee_string(self):
        """Test guarantee string format."""
        predictor = CopulaConformalPredictor()
        predictor.calibrate(
            model=simple_linear_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            m=500,
            epsilon=0.01,
            seed=42
        )

        guarantee_str = predictor.get_guarantee_string()
        assert "Coverage" in guarantee_str or "coverage" in guarantee_str.lower()
        assert "confidence" in guarantee_str.lower()


class TestCopulaConformalPredictorBoundingBox:
    """Tests for bounding box estimation."""

    def test_bounding_box_shape(self):
        """Test bounding box has correct shape."""
        predictor = CopulaConformalPredictor()
        predictor.calibrate(
            model=simple_linear_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            m=500,
            seed=42
        )

        lb, ub = predictor.get_bounding_box(n_samples=1000, seed=42)

        assert lb.shape == (2,)
        assert ub.shape == (2,)
        assert np.all(lb < ub)

    def test_bounding_box_contains_center(self):
        """Test that bounding box contains center."""
        predictor = CopulaConformalPredictor()
        predictor.calibrate(
            model=simple_linear_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            m=500,
            seed=42
        )

        lb, ub = predictor.get_bounding_box(n_samples=1000, seed=42)

        assert np.all(lb <= predictor.center)
        assert np.all(predictor.center <= ub)


class TestCopulaConformalPredictorEdgeCases:
    """Tests for edge cases."""

    def test_uncalibrated_predictor_errors(self):
        """Test that uncalibrated predictor raises errors."""
        predictor = CopulaConformalPredictor()

        with pytest.raises(ValueError, match="must be calibrated"):
            predictor.score(np.array([0.5, 0.5]))

        with pytest.raises(ValueError, match="must be calibrated"):
            predictor.contains(np.array([0.5, 0.5]))

        with pytest.raises(ValueError, match="must be calibrated"):
            predictor.sample(10)

    def test_dimension_mismatch_error(self):
        """Test error on dimension mismatch."""
        predictor = CopulaConformalPredictor()
        predictor.calibrate(
            model=simple_linear_model,
            input_lb=np.array([0, 0]),
            input_ub=np.array([1, 1]),
            m=200,
            seed=42
        )

        with pytest.raises(ValueError, match="Expected 2 dimensions"):
            predictor.score(np.array([0.5, 0.5, 0.5]))  # Wrong dimension


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
