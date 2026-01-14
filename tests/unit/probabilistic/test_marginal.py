"""
Unit tests for marginal CDF classes.
"""

import numpy as np
import pytest
from scipy.stats import norm

from n2v.probabilistic.copula.marginal import KernelCDF, EmpiricalCDF, MarginalCDF


class TestKernelCDF:
    """Tests for KernelCDF class."""

    def test_basic_construction(self):
        """Test basic construction with valid data."""
        data = np.random.randn(100)
        cdf = KernelCDF(data)
        assert cdf.n == 100
        assert cdf.bandwidth > 0

    def test_silverman_bandwidth(self):
        """Test Silverman's rule bandwidth calculation."""
        np.random.seed(42)
        data = np.random.randn(1000)
        cdf = KernelCDF(data)

        # Silverman's rule: h = 1.06 * std * n^(-0.2)
        expected = 1.06 * np.std(data, ddof=1) * len(data) ** (-0.2)
        np.testing.assert_almost_equal(cdf.bandwidth, expected, decimal=10)

    def test_cdf_monotonic(self):
        """Test that CDF is monotonically increasing."""
        data = np.random.randn(200)
        cdf = KernelCDF(data)

        x = np.linspace(-3, 3, 50)
        cdf_values = cdf.cdf(x)

        # Check monotonicity
        assert np.all(np.diff(cdf_values) >= -1e-10)

    def test_cdf_bounds(self):
        """Test that CDF values are in [0, 1]."""
        data = np.random.randn(200)
        cdf = KernelCDF(data)

        x = np.linspace(-10, 10, 100)
        cdf_values = cdf.cdf(x)

        assert np.all(cdf_values >= 0)
        assert np.all(cdf_values <= 1)

    def test_cdf_approaches_bounds(self):
        """Test that CDF approaches 0 and 1 at extremes."""
        data = np.random.randn(500)
        cdf = KernelCDF(data)

        # Far left should be near 0
        assert cdf.cdf(-10) < 0.01

        # Far right should be near 1
        assert cdf.cdf(10) > 0.99

    def test_inverse_consistency(self):
        """Test that inverse(cdf(x)) ≈ x."""
        np.random.seed(42)
        data = np.random.randn(500)
        cdf = KernelCDF(data)

        # Test some x values within data range
        x_test = np.percentile(data, [10, 25, 50, 75, 90])

        for x in x_test:
            u = cdf.cdf(x)
            x_recovered = cdf.inverse(u)
            np.testing.assert_almost_equal(x_recovered, x, decimal=2)

    def test_inverse_bounds(self):
        """Test inverse at boundary values."""
        data = np.random.randn(200)
        cdf = KernelCDF(data)

        # Inverse of values near 0 and 1
        low = cdf.inverse(0.01)
        high = cdf.inverse(0.99)

        assert low < high
        assert cdf.cdf(low) < 0.1
        assert cdf.cdf(high) > 0.9

    def test_scalar_and_array_input(self):
        """Test that scalar and array inputs work correctly."""
        data = np.random.randn(200)
        cdf = KernelCDF(data)

        # Scalar input
        scalar_result = cdf.cdf(0.0)
        assert isinstance(scalar_result, float)

        # Array input
        array_result = cdf.cdf(np.array([0.0, 0.5, 1.0]))
        assert isinstance(array_result, np.ndarray)
        assert len(array_result) == 3

    def test_custom_bandwidth(self):
        """Test construction with custom bandwidth."""
        data = np.random.randn(100)
        cdf = KernelCDF(data, bandwidth=0.5)
        assert cdf.bandwidth == 0.5

    def test_minimum_samples(self):
        """Test that at least 2 samples are required."""
        with pytest.raises(ValueError, match="at least 2 data points"):
            KernelCDF(np.array([1.0]))

    def test_zero_variance_data(self):
        """Test handling of constant data (zero variance)."""
        data = np.ones(100)
        cdf = KernelCDF(data)

        # Should still work, bandwidth should be small but positive
        assert cdf.bandwidth > 0

    def test_log_density_finite(self):
        """Test that log_density returns finite values."""
        np.random.seed(42)
        data = np.random.randn(200)
        cdf = KernelCDF(data)

        x_test = np.linspace(-3, 3, 50)
        log_dens = cdf.log_density(x_test)

        assert np.all(np.isfinite(log_dens))

    def test_log_density_shape(self):
        """Test that log_density has correct output shape."""
        data = np.random.randn(100)
        cdf = KernelCDF(data)

        # Scalar input
        scalar_result = cdf.log_density(0.0)
        assert isinstance(scalar_result, float)

        # Array input
        array_result = cdf.log_density(np.array([0.0, 0.5, 1.0]))
        assert isinstance(array_result, np.ndarray)
        assert len(array_result) == 3

    def test_log_density_peaks_near_data(self):
        """Test that log_density is higher near data points."""
        np.random.seed(42)
        data = np.random.randn(500)
        cdf = KernelCDF(data)

        # Log density at mean should be higher than at far points
        log_dens_center = cdf.log_density(np.mean(data))
        log_dens_far = cdf.log_density(10.0)

        assert log_dens_center > log_dens_far

    def test_log_density_low_outside_support(self):
        """Test that log_density is low outside data support."""
        data = np.random.uniform(-1, 1, 500)
        cdf = KernelCDF(data)

        # Very far from data should have low density
        log_dens_inside = cdf.log_density(0.0)
        log_dens_outside = cdf.log_density(10.0)

        assert log_dens_inside > log_dens_outside


class TestEmpiricalCDF:
    """Tests for EmpiricalCDF class."""

    def test_basic_construction(self):
        """Test basic construction."""
        data = np.random.randn(100)
        cdf = EmpiricalCDF(data)
        assert cdf.n == 100

    def test_data_sorted(self):
        """Test that data is sorted internally."""
        data = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        cdf = EmpiricalCDF(data)
        assert np.all(np.diff(cdf.data) >= 0)

    def test_cdf_step_function(self):
        """Test that CDF is a step function."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        cdf = EmpiricalCDF(data)

        # Below all data
        assert cdf.cdf(0.5) == 0.0

        # At first data point
        assert cdf.cdf(1.0) == 0.25

        # Between points
        assert cdf.cdf(1.5) == 0.25
        assert cdf.cdf(2.5) == 0.5

        # Above all data
        assert cdf.cdf(5.0) == 1.0

    def test_cdf_monotonic(self):
        """Test monotonicity."""
        data = np.random.randn(100)
        cdf = EmpiricalCDF(data)

        x = np.linspace(-3, 3, 100)
        cdf_values = cdf.cdf(x)

        assert np.all(np.diff(cdf_values) >= 0)

    def test_inverse(self):
        """Test inverse CDF."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        cdf = EmpiricalCDF(data)

        # Inverse at various probabilities
        assert cdf.inverse(0.25) == 1.0
        assert cdf.inverse(0.50) == 2.0
        assert cdf.inverse(0.75) == 3.0

    def test_inverse_array(self):
        """Test inverse with array input."""
        data = np.random.randn(100)
        cdf = EmpiricalCDF(data)

        u = np.array([0.1, 0.5, 0.9])
        result = cdf.inverse(u)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_single_sample(self):
        """Test with single sample."""
        data = np.array([5.0])
        cdf = EmpiricalCDF(data)

        assert cdf.cdf(4.0) == 0.0
        assert cdf.cdf(5.0) == 1.0
        assert cdf.cdf(6.0) == 1.0

    def test_log_density_finite(self):
        """Test that log_density returns finite values."""
        data = np.random.randn(100)
        cdf = EmpiricalCDF(data)

        x_test = np.linspace(-3, 3, 50)
        log_dens = cdf.log_density(x_test)

        assert np.all(np.isfinite(log_dens))

    def test_log_density_shape(self):
        """Test that log_density has correct output shape."""
        data = np.random.randn(100)
        cdf = EmpiricalCDF(data)

        # Scalar input
        scalar_result = cdf.log_density(0.0)
        assert isinstance(scalar_result, float)

        # Array input
        array_result = cdf.log_density(np.array([0.0, 0.5, 1.0]))
        assert isinstance(array_result, np.ndarray)
        assert len(array_result) == 3


class TestMarginalCDFAbstract:
    """Tests for MarginalCDF abstract base class."""

    def test_cannot_instantiate(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MarginalCDF()

    def test_subclass_implementation(self):
        """Test that subclasses must implement all required methods."""

        class IncompleteCDF(MarginalCDF):
            def cdf(self, x):
                return 0.5

            def inverse(self, u):
                return 0.0

            # Missing log_density

        with pytest.raises(TypeError):
            IncompleteCDF()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
