"""
Unit tests for GaussianCopula class.
"""

import numpy as np
import pytest
from scipy.stats import norm

from n2v.probabilistic.copula.gaussian import GaussianCopula


class TestGaussianCopulaFitting:
    """Tests for copula fitting."""

    def test_basic_fit(self):
        """Test basic fitting with valid data."""
        np.random.seed(42)
        U = np.random.uniform(0, 1, size=(100, 3))
        copula = GaussianCopula()
        copula.fit(U)

        assert copula.d == 3
        assert copula.correlation.shape == (3, 3)
        assert copula.correlation_inv.shape == (3, 3)
        assert copula.log_det is not None

    def test_correlation_properties(self):
        """Test that fitted correlation matrix has correct properties."""
        np.random.seed(42)
        U = np.random.uniform(0, 1, size=(500, 4))
        copula = GaussianCopula()
        copula.fit(U)

        # Diagonal should be 1
        np.testing.assert_array_almost_equal(
            np.diag(copula.correlation), np.ones(4), decimal=10
        )

        # Should be symmetric
        np.testing.assert_array_almost_equal(
            copula.correlation, copula.correlation.T, decimal=10
        )

        # Should be positive definite (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvalsh(copula.correlation)
        assert np.all(eigenvalues > 0)

    def test_correlation_recovery(self):
        """Test that known correlation can be recovered."""
        np.random.seed(42)

        # Generate correlated normal data with known correlation
        rho = 0.7
        cov = np.array([[1, rho], [rho, 1]])
        Z = np.random.multivariate_normal([0, 0], cov, size=5000)

        # Transform to uniform
        U = norm.cdf(Z)

        # Fit copula
        copula = GaussianCopula()
        copula.fit(U)

        # Check correlation is recovered (approximately)
        np.testing.assert_almost_equal(
            copula.correlation[0, 1], rho, decimal=1
        )

    def test_independence_case(self):
        """Test that independent data gives identity correlation."""
        np.random.seed(42)

        # Generate independent uniform data
        U = np.random.uniform(0, 1, size=(2000, 3))

        copula = GaussianCopula()
        copula.fit(U)

        # Off-diagonal should be close to 0
        off_diag = copula.correlation - np.eye(3)
        assert np.max(np.abs(off_diag)) < 0.1

    def test_1d_case(self):
        """Test 1D case (trivial copula)."""
        U = np.random.uniform(0, 1, size=(100, 1))
        copula = GaussianCopula()
        copula.fit(U)

        assert copula.d == 1
        np.testing.assert_array_equal(copula.correlation, [[1.0]])

    def test_regularization(self):
        """Test that regularization ensures positive definiteness."""
        np.random.seed(42)

        # Create data that might give a near-singular correlation
        U = np.random.uniform(0, 1, size=(50, 5))
        U[:, 2] = U[:, 1] + 1e-10 * np.random.randn(50)  # Almost duplicate

        copula = GaussianCopula()
        copula.fit(U, min_eigenvalue=1e-6)

        # Should still be positive definite
        eigenvalues = np.linalg.eigvalsh(copula.correlation)
        assert np.all(eigenvalues >= 1e-6 - 1e-10)

    def test_minimum_samples(self):
        """Test that at least 2 samples are required."""
        U = np.array([[0.5, 0.5]])
        copula = GaussianCopula()

        with pytest.raises(ValueError, match="at least 2 samples"):
            copula.fit(U)


class TestGaussianCopulaDensity:
    """Tests for copula density computation."""

    def test_density_positive(self):
        """Test that density is always positive."""
        np.random.seed(42)
        U = np.random.uniform(0.1, 0.9, size=(100, 3))
        copula = GaussianCopula()
        copula.fit(U)

        for _ in range(10):
            u = np.random.uniform(0.1, 0.9, 3)
            density = copula.density(u)
            assert density > 0

    def test_log_density_finite(self):
        """Test that log density is finite for valid inputs."""
        np.random.seed(42)
        U = np.random.uniform(0.1, 0.9, size=(100, 3))
        copula = GaussianCopula()
        copula.fit(U)

        for _ in range(10):
            u = np.random.uniform(0.01, 0.99, 3)
            log_dens = copula.log_density(u)
            assert np.isfinite(log_dens)

    def test_density_vs_log_density(self):
        """Test that density and log_density are consistent."""
        np.random.seed(42)
        U = np.random.uniform(0.1, 0.9, size=(100, 2))
        copula = GaussianCopula()
        copula.fit(U)

        u = np.array([0.5, 0.5])
        density = copula.density(u)
        log_density = copula.log_density(u)

        np.testing.assert_almost_equal(np.log(density), log_density, decimal=10)

    def test_batch_density(self):
        """Test batch density computation."""
        np.random.seed(42)
        U = np.random.uniform(0.1, 0.9, size=(100, 3))
        copula = GaussianCopula()
        copula.fit(U)

        U_test = np.random.uniform(0.1, 0.9, size=(20, 3))
        log_densities = copula.log_density_batch(U_test)

        assert log_densities.shape == (20,)
        assert np.all(np.isfinite(log_densities))

        # Compare with individual computation
        for i in range(5):
            expected = copula.log_density(U_test[i])
            np.testing.assert_almost_equal(log_densities[i], expected, decimal=10)

    def test_independence_density(self):
        """Test that independent copula has density ≈ 1 (log ≈ 0)."""
        # For independent copula, c(u) ≈ 1 everywhere
        copula = GaussianCopula()
        copula.correlation = np.eye(3)
        copula.correlation_inv = np.eye(3)
        copula.log_det = 0.0
        copula.d = 3

        u = np.array([0.5, 0.5, 0.5])
        log_dens = copula.log_density(u)

        np.testing.assert_almost_equal(log_dens, 0.0, decimal=10)

    def test_density_dimension_mismatch(self):
        """Test error on dimension mismatch."""
        np.random.seed(42)
        U = np.random.uniform(0, 1, size=(100, 3))
        copula = GaussianCopula()
        copula.fit(U)

        with pytest.raises(ValueError, match="Expected 3 dimensions"):
            copula.log_density(np.array([0.5, 0.5]))  # Wrong dimension


class TestGaussianCopulaSampling:
    """Tests for copula sampling."""

    def test_sample_shape(self):
        """Test that samples have correct shape."""
        np.random.seed(42)
        U = np.random.uniform(0, 1, size=(100, 4))
        copula = GaussianCopula()
        copula.fit(U)

        samples = copula.sample(50, seed=42)
        assert samples.shape == (50, 4)

    def test_sample_range(self):
        """Test that samples are in [0, 1]."""
        np.random.seed(42)
        U = np.random.uniform(0, 1, size=(100, 3))
        copula = GaussianCopula()
        copula.fit(U)

        samples = copula.sample(1000, seed=42)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_sample_correlation(self):
        """Test that samples preserve correlation structure."""
        np.random.seed(42)

        # Generate data with known correlation
        rho = 0.8
        cov = np.array([[1, rho], [rho, 1]])
        Z = np.random.multivariate_normal([0, 0], cov, size=2000)
        U = norm.cdf(Z)

        copula = GaussianCopula()
        copula.fit(U)

        # Sample from copula
        samples = copula.sample(5000, seed=42)

        # Transform to normal and check correlation
        Z_samples = norm.ppf(np.clip(samples, 1e-10, 1 - 1e-10))
        sample_corr = np.corrcoef(Z_samples.T)[0, 1]

        np.testing.assert_almost_equal(sample_corr, rho, decimal=1)

    def test_sample_reproducibility(self):
        """Test that seed gives reproducible samples."""
        np.random.seed(42)
        U = np.random.uniform(0, 1, size=(100, 3))
        copula = GaussianCopula()
        copula.fit(U)

        samples1 = copula.sample(50, seed=123)
        samples2 = copula.sample(50, seed=123)

        np.testing.assert_array_equal(samples1, samples2)


class TestGaussianCopulaEdgeCases:
    """Tests for edge cases."""

    def test_unfitted_copula_errors(self):
        """Test that unfitted copula raises errors."""
        copula = GaussianCopula()

        with pytest.raises(ValueError, match="must be fitted"):
            copula.log_density(np.array([0.5, 0.5]))

        with pytest.raises(ValueError, match="must be fitted"):
            copula.sample(10)

        with pytest.raises(ValueError, match="must be fitted"):
            copula.get_correlation()

    def test_boundary_values(self):
        """Test handling of boundary values in density."""
        np.random.seed(42)
        U = np.random.uniform(0.1, 0.9, size=(100, 2))
        copula = GaussianCopula()
        copula.fit(U)

        # Values at boundaries should be clipped internally
        u_boundary = np.array([0.0, 1.0])
        log_dens = copula.log_density(u_boundary)
        assert np.isfinite(log_dens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
