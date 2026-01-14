"""
Unit tests for CopulaRegion class.
"""

import numpy as np
import pytest

from n2v.sets import CopulaRegion, ProbabilisticBox
from n2v.probabilistic.copula.predictor import CopulaConformalPredictor


def simple_model(x):
    """Simple linear model for testing."""
    return x @ np.array([[1, 0.5], [0.5, 1]]) + np.array([0.1, 0.2])


@pytest.fixture
def calibrated_region():
    """Create a calibrated CopulaRegion for testing."""
    predictor = CopulaConformalPredictor()
    predictor.calibrate(
        model=simple_model,
        input_lb=np.array([0, 0]),
        input_ub=np.array([1, 1]),
        training_samples=300,
        m=800,
        epsilon=0.01,
        seed=42
    )
    return CopulaRegion(predictor)


class TestCopulaRegionConstruction:
    """Tests for CopulaRegion construction."""

    def test_basic_construction(self, calibrated_region):
        """Test basic construction."""
        assert calibrated_region.d == 2
        assert calibrated_region.m == 800
        assert calibrated_region.ell == 799
        assert calibrated_region.epsilon == 0.01
        assert calibrated_region.coverage == 0.99
        assert 0 < calibrated_region.confidence < 1

    def test_uncalibrated_predictor_error(self):
        """Test error when using uncalibrated predictor."""
        predictor = CopulaConformalPredictor()

        with pytest.raises(ValueError, match="must be calibrated"):
            CopulaRegion(predictor)

    def test_repr(self, calibrated_region):
        """Test string representation."""
        repr_str = repr(calibrated_region)
        assert "CopulaRegion" in repr_str
        assert "dim=2" in repr_str
        assert "coverage" in repr_str


class TestCopulaRegionContainment:
    """Tests for containment operations."""

    def test_contains_returns_bool(self, calibrated_region):
        """Test that contains returns boolean."""
        y = calibrated_region.center
        result = calibrated_region.contains(y)
        assert isinstance(result, bool)

    def test_contains_center(self, calibrated_region):
        """Test that center is contained."""
        assert calibrated_region.contains(calibrated_region.center)

    def test_contains_batch(self, calibrated_region):
        """Test batch containment."""
        Y = calibrated_region.center + np.random.randn(10, 2) * 0.05
        result = calibrated_region.contains_batch(Y)

        assert result.shape == (10,)
        assert result.dtype == bool


class TestCopulaRegionSampling:
    """Tests for sampling."""

    def test_sample_shape(self, calibrated_region):
        """Test sample shape."""
        samples = calibrated_region.sample(50, seed=42)
        assert samples.shape == (50, 2)

    def test_samples_in_region(self, calibrated_region):
        """Test that samples are contained in region."""
        samples = calibrated_region.sample(50, seed=42)

        for sample in samples:
            assert calibrated_region.contains(sample)

    def test_sample_reproducibility(self, calibrated_region):
        """Test sample reproducibility with seed."""
        samples1 = calibrated_region.sample(20, seed=123)
        samples2 = calibrated_region.sample(20, seed=123)

        np.testing.assert_array_equal(samples1, samples2)


class TestCopulaRegionConversion:
    """Tests for conversion to other types."""

    def test_to_box_returns_probabilistic_box(self, calibrated_region):
        """Test conversion to ProbabilisticBox."""
        box = calibrated_region.to_box(n_samples=1000, seed=42)

        assert isinstance(box, ProbabilisticBox)
        assert box.dim == 2
        assert box.m == calibrated_region.m
        assert box.ell == calibrated_region.ell
        assert box.epsilon == calibrated_region.epsilon

    def test_to_hyperrectangle_alias(self, calibrated_region):
        """Test that to_hyperrectangle is alias for to_box."""
        box1 = calibrated_region.to_box(n_samples=500, seed=42)
        box2 = calibrated_region.to_hyperrectangle(n_samples=500, seed=42)

        np.testing.assert_array_equal(box1.lb, box2.lb)
        np.testing.assert_array_equal(box1.ub, box2.ub)

    def test_get_bounding_box(self, calibrated_region):
        """Test bounding box computation."""
        lb, ub = calibrated_region.get_bounding_box(n_samples=1000, seed=42)

        assert lb.shape == (2,)
        assert ub.shape == (2,)
        assert np.all(lb < ub)

    def test_bounding_box_caching(self, calibrated_region):
        """Test that bounding box is cached."""
        lb1, ub1 = calibrated_region.get_bounding_box(n_samples=1000, seed=42)
        lb2, ub2 = calibrated_region.get_bounding_box(use_cache=True)

        np.testing.assert_array_equal(lb1, lb2)
        np.testing.assert_array_equal(ub1, ub2)

    def test_get_ranges(self, calibrated_region):
        """Test get_ranges method."""
        lb, ub = calibrated_region.get_ranges(n_samples=500, seed=42)

        assert lb.shape == (2, 1)
        assert ub.shape == (2, 1)


class TestCopulaRegionVolumeAnalysis:
    """Tests for volume analysis."""

    def test_volume_ratio_range(self, calibrated_region):
        """Test that volume ratio is in [0, 1]."""
        ratio = calibrated_region.volume_ratio(n_samples=5000, seed=42)

        assert 0 <= ratio <= 1

    def test_volume_reduction_complement(self, calibrated_region):
        """Test that volume_reduction = 1 - volume_ratio."""
        ratio = calibrated_region.volume_ratio(n_samples=5000, seed=42)
        reduction = calibrated_region.volume_reduction(n_samples=5000, seed=42)

        np.testing.assert_almost_equal(ratio + reduction, 1.0, decimal=5)


class TestCopulaRegionGuarantee:
    """Tests for guarantee methods."""

    def test_get_guarantee(self, calibrated_region):
        """Test get_guarantee returns tuple."""
        coverage, confidence = calibrated_region.get_guarantee()

        assert coverage == 0.99
        assert 0 < confidence < 1

    def test_get_guarantee_string(self, calibrated_region):
        """Test guarantee string."""
        guarantee_str = calibrated_region.get_guarantee_string()

        assert isinstance(guarantee_str, str)
        assert len(guarantee_str) > 0


class TestCopulaRegionProperties:
    """Tests for properties."""

    def test_center_property(self, calibrated_region):
        """Test center property."""
        center = calibrated_region.center
        assert center.shape == (2,)

    def test_threshold_property(self, calibrated_region):
        """Test threshold property."""
        threshold = calibrated_region.threshold
        assert isinstance(threshold, float)
        assert threshold > 0

    def test_correlation_property(self, calibrated_region):
        """Test correlation property."""
        corr = calibrated_region.correlation
        assert corr.shape == (2, 2)
        np.testing.assert_array_almost_equal(np.diag(corr), [1, 1])

    def test_dim_property(self, calibrated_region):
        """Test dim property."""
        assert calibrated_region.dim == 2


class TestCopulaRegionScore:
    """Tests for scoring."""

    def test_score_returns_float(self, calibrated_region):
        """Test that score returns float."""
        y = calibrated_region.center
        score = calibrated_region.score(y)
        assert isinstance(score, float)

    def test_score_center_in_region(self, calibrated_region):
        """Test that center is contained in the region."""
        center_score = calibrated_region.score(calibrated_region.center)
        # Center should have score below threshold (inside region)
        assert center_score <= calibrated_region.threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
