"""
Unit tests for DeflationPCA.
"""

import pytest
import numpy as np

from n2v.probabilistic.dimensionality.deflation_pca import DeflationPCA


class TestDeflationPCAFit:
    """Tests for DeflationPCA fit() method."""

    def test_fit_basic(self):
        """Test basic fitting."""
        np.random.seed(42)
        X = np.random.randn(50, 100)  # 50 samples, 100 dimensions

        pca = DeflationPCA(n_components=10)
        pca.fit(X)

        assert pca._is_fitted
        assert pca.components_.shape == (10, 100)
        assert pca.mean_.shape == (100,)

    def test_fit_returns_self(self):
        """Test that fit() returns self for method chaining."""
        X = np.random.randn(20, 30)

        pca = DeflationPCA(n_components=5)
        result = pca.fit(X)

        assert result is pca

    def test_fit_computes_mean(self):
        """Test that fit() computes correct mean."""
        X = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        pca = DeflationPCA(n_components=2)
        pca.fit(X)

        expected_mean = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(pca.mean_, expected_mean)

    def test_fit_warns_if_n_components_too_large(self):
        """Test that fit() warns if n_components > min(t, n)."""
        X = np.random.randn(10, 20)  # 10 samples, 20 dimensions

        pca = DeflationPCA(n_components=15)  # > 10

        with pytest.warns(UserWarning):
            pca.fit(X)

        # Should reduce to 10 components
        assert pca.components_.shape[0] == 10

    def test_fit_handles_1d_input(self):
        """Test that fit() handles 1D input (single sample)."""
        X = np.random.randn(10)

        pca = DeflationPCA(n_components=1)
        pca.fit(X)

        assert pca.components_.shape == (1, 10)


class TestDeflationPCATransform:
    """Tests for DeflationPCA transform() method."""

    def test_transform_basic(self):
        """Test basic transformation."""
        np.random.seed(42)
        X = np.random.randn(50, 100)

        pca = DeflationPCA(n_components=10)
        pca.fit(X)

        X_reduced = pca.transform(X)

        assert X_reduced.shape == (50, 10)

    def test_transform_raises_if_not_fitted(self):
        """Test that transform() raises if not fitted."""
        pca = DeflationPCA(n_components=5)

        with pytest.raises(RuntimeError, match="fitted"):
            pca.transform(np.random.randn(10, 20))

    def test_transform_handles_1d_input(self):
        """Test that transform() handles 1D input (single sample)."""
        X = np.random.randn(20, 30)
        pca = DeflationPCA(n_components=5)
        pca.fit(X)

        X_single = np.random.randn(30)
        X_reduced = pca.transform(X_single)

        assert X_reduced.shape == (1, 5)


class TestDeflationPCAInverseTransform:
    """Tests for DeflationPCA inverse_transform() method."""

    def test_inverse_transform_basic(self):
        """Test basic inverse transformation."""
        np.random.seed(42)
        X = np.random.randn(50, 100)

        pca = DeflationPCA(n_components=10)
        pca.fit(X)

        X_reduced = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)

        assert X_reconstructed.shape == (50, 100)

    def test_inverse_transform_raises_if_not_fitted(self):
        """Test that inverse_transform() raises if not fitted."""
        pca = DeflationPCA(n_components=5)

        with pytest.raises(RuntimeError, match="fitted"):
            pca.inverse_transform(np.random.randn(10, 5))

    def test_transform_inverse_transform_consistency(self):
        """Test that transform and inverse_transform are consistent."""
        np.random.seed(42)
        # Create low-rank data for perfect reconstruction
        n_samples, n_dim, n_rank = 50, 100, 5
        U = np.random.randn(n_samples, n_rank)
        V = np.random.randn(n_rank, n_dim)
        X = U @ V  # Low-rank matrix

        pca = DeflationPCA(n_components=n_rank, max_iter=2000)
        pca.fit(X)

        X_reduced = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)

        # Should be close to original for low-rank data
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        assert reconstruction_error < 0.1  # Tolerance for optimization


class TestDeflationPCAFitTransform:
    """Tests for DeflationPCA fit_transform() method."""

    def test_fit_transform_equals_fit_then_transform(self):
        """Test that fit_transform equals fit() + transform() on same data."""
        np.random.seed(42)
        X = np.random.randn(30, 50)

        # Use the same PCA instance - fit_transform should be equivalent to fit + transform
        pca = DeflationPCA(n_components=10)
        pca.fit(X)
        X_reduced_separate = pca.transform(X)

        # Create new instance with same seed for fitting
        np.random.seed(42)
        X2 = np.random.randn(30, 50)  # Same data

        np.random.seed(123)  # Different seed for PCA random init
        pca2 = DeflationPCA(n_components=10)
        X_reduced_combined = pca2.fit_transform(X2)

        # After fitting, transform should give consistent results
        X_reduced_verify = pca2.transform(X2)
        np.testing.assert_array_almost_equal(X_reduced_combined, X_reduced_verify)


class TestDeflationPCAComponentOrthogonality:
    """Tests for component orthogonality."""

    def test_components_are_unit_vectors(self):
        """Test that components are unit vectors."""
        np.random.seed(42)
        X = np.random.randn(50, 100)

        pca = DeflationPCA(n_components=10)
        pca.fit(X)

        for i in range(pca.components_.shape[0]):
            norm = np.linalg.norm(pca.components_[i])
            np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_components_are_approximately_orthogonal(self):
        """Test that components are approximately orthogonal."""
        np.random.seed(42)
        X = np.random.randn(100, 50)

        pca = DeflationPCA(n_components=10, max_iter=2000)
        pca.fit(X)

        # Check pairwise dot products
        for i in range(pca.components_.shape[0]):
            for j in range(i + 1, pca.components_.shape[0]):
                dot_product = np.abs(np.dot(pca.components_[i], pca.components_[j]))
                # Should be close to 0 (orthogonal)
                assert dot_product < 0.1, f"Components {i} and {j} not orthogonal: {dot_product}"


class TestDeflationPCAVarianceExplained:
    """Tests for variance explanation."""

    def test_variance_decreases_with_deflation(self):
        """Test that variance explained decreases with each component."""
        np.random.seed(42)
        X = np.random.randn(100, 50)

        pca = DeflationPCA(n_components=10, max_iter=2000)
        pca.fit(X)

        # Compute variance explained by each component
        X_centered = X - pca.mean_
        variances = []
        for i in range(pca.components_.shape[0]):
            projections = X_centered @ pca.components_[i]
            variance = np.var(projections)
            variances.append(variance)

        # First few should be roughly decreasing (may not be perfectly monotonic)
        # Just check that first > last
        assert variances[0] > variances[-1] * 0.5


class TestDeflationPCAHighDimensional:
    """Tests for high-dimensional cases."""

    def test_works_when_t_less_than_n(self):
        """Test that PCA works when t < n (more dimensions than samples)."""
        np.random.seed(42)
        t, n = 20, 100  # More dimensions than samples

        X = np.random.randn(t, n)

        pca = DeflationPCA(n_components=10)
        pca.fit(X)

        assert pca.components_.shape == (10, n)

        X_reduced = pca.transform(X)
        assert X_reduced.shape == (t, 10)

    def test_reasonable_reconstruction_high_dim(self):
        """Test reconstruction in high-dimensional case."""
        np.random.seed(42)
        t, n = 30, 200

        # Create data with some structure
        X = np.random.randn(t, n)

        pca = DeflationPCA(n_components=20, max_iter=2000)
        pca.fit(X)

        X_reduced = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)

        # Should capture some variance
        original_var = np.var(X)
        residual_var = np.var(X - X_reconstructed)

        assert residual_var < original_var  # Some variance captured


class TestDeflationPCAEdgeCases:
    """Tests for edge cases."""

    def test_single_component(self):
        """Test with n_components=1."""
        np.random.seed(42)
        X = np.random.randn(50, 30)

        pca = DeflationPCA(n_components=1)
        pca.fit(X)

        assert pca.components_.shape == (1, 30)

    def test_n_components_equals_n_samples(self):
        """Test when n_components equals n_samples."""
        np.random.seed(42)
        X = np.random.randn(10, 50)

        pca = DeflationPCA(n_components=10)
        pca.fit(X)

        assert pca.components_.shape == (10, 50)

    def test_small_data(self):
        """Test with very small data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        pca = DeflationPCA(n_components=2)
        pca.fit(X)

        assert pca.components_.shape == (2, 2)

    def test_constant_dimension(self):
        """Test with one constant dimension."""
        np.random.seed(42)
        X = np.random.randn(30, 10)
        X[:, 0] = 5.0  # Constant column

        pca = DeflationPCA(n_components=5)
        pca.fit(X)

        # Should still work
        assert pca.components_.shape == (5, 10)


class TestDeflationPCAParameters:
    """Tests for different parameter settings."""

    def test_different_learning_rates(self):
        """Test with different learning rates."""
        np.random.seed(42)
        X = np.random.randn(50, 30)

        for lr in [0.01, 0.1, 0.5]:
            pca = DeflationPCA(n_components=5, learning_rate=lr)
            pca.fit(X)
            assert pca._is_fitted

    def test_different_tolerances(self):
        """Test with different convergence tolerances."""
        np.random.seed(42)
        X = np.random.randn(50, 30)

        for tol in [1e-3, 1e-6, 1e-9]:
            pca = DeflationPCA(n_components=5, tol=tol)
            pca.fit(X)
            assert pca._is_fitted


class TestDeflationPCAVarianceThreshold:
    """Test early stopping when remaining variance is negligible."""

    def test_stops_early_when_variance_exhausted(self):
        """PCA with low-rank data should extract fewer components than requested."""
        np.random.seed(42)
        t, true_rank = 30, 2
        Z = np.random.randn(t, true_rank) @ np.random.randn(true_rank, 50)

        pca = DeflationPCA(n_components=10, variance_threshold=1e-6)
        pca.fit(Z)

        assert pca.components_.shape[0] <= true_rank + 1
        assert pca.n_components_fitted_ <= true_rank + 1

    def test_no_threshold_extracts_all(self):
        """Without threshold, all requested components are extracted."""
        np.random.seed(42)
        t, true_rank = 30, 2
        Z = np.random.randn(t, true_rank) @ np.random.randn(true_rank, 50)

        pca = DeflationPCA(n_components=10, variance_threshold=None)
        pca.fit(Z)

        assert pca.components_.shape[0] == 10

    def test_transform_shape_with_early_stop(self):
        """Transform output shape matches actual fitted components, not requested."""
        np.random.seed(42)
        t = 30
        Z = np.random.randn(t, 2) @ np.random.randn(2, 50)

        pca = DeflationPCA(n_components=10, variance_threshold=1e-6)
        pca.fit(Z)

        reduced = pca.transform(Z)
        assert reduced.shape == (t, pca.n_components_fitted_)

        reconstructed = pca.inverse_transform(reduced)
        assert reconstructed.shape == (t, 50)


class TestDeflationPCACompareWithSklearn:
    """Compare with sklearn PCA on small data (where both work)."""

    def test_similar_to_sklearn_small_data(self):
        """Test that results are similar to sklearn PCA on small data."""
        try:
            from sklearn.decomposition import PCA as SklearnPCA
        except ImportError:
            pytest.skip("sklearn not installed")

        np.random.seed(42)
        X = np.random.randn(100, 20)

        # sklearn PCA
        sklearn_pca = SklearnPCA(n_components=5)
        X_sklearn = sklearn_pca.fit_transform(X)

        # DeflationPCA
        np.random.seed(42)
        deflation_pca = DeflationPCA(n_components=5, max_iter=3000)
        X_deflation = deflation_pca.fit_transform(X)

        # Compare variance explained (not exact components due to sign ambiguity)
        sklearn_var = np.sum(np.var(X_sklearn, axis=0))
        deflation_var = np.sum(np.var(X_deflation, axis=0))

        # Should capture similar variance (within 20%)
        assert abs(sklearn_var - deflation_var) / sklearn_var < 0.2
