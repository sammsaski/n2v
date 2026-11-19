"""
Unit tests for DeflationPCA.
"""

import tracemalloc
import warnings

import pytest
import numpy as np

from n2v.probabilistic import conformal_reach
from n2v.probabilistic.dimensionality.deflation_pca import DeflationPCA
from n2v.sets import Box
from tests.known_bugs import DID_NOT_RAISE, known_bug


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


# ---------------------------------------------------------------------------
# Data whose principal directions are known
# exactly, the stored mean and components as conformal_reach reads them, the
# options, the branches of the ascent, the public route, and today's defects.
# ---------------------------------------------------------------------------

def _hadamard(k):
    """The Sylvester Hadamard matrix of order 2**k: entries +-1, orthogonal columns, every column but the first sums
    to zero."""
    h = np.array([[1.0]])
    for _ in range(k):
        h = np.block([[h, h], [h, -h]])
    return h


def _principal_data(t, n, sds, mean):
    """t rows mean + S diag(sds) Q^T in n dimensions whose principal directions are known exactly.

    S holds columns 1..k of the t-by-t Hadamard matrix (entries +-1, zero mean, orthogonal, population variance 1),
    Q holds columns 1..k of the n-by-n Hadamard matrix divided by sqrt(n) (orthonormal). For t a power of two and
    n = 4 or 16 every entry is dyadic, so the data, their mean and their covariance are exact in float64. The centred
    covariance is Q diag(sds**2) Q^T: the principal directions are the columns of Q in order, with population
    variances sds**2, and the data have rank k. Returns (X, Q)."""
    k = len(sds)
    scores = _hadamard(int(np.log2(t)))[:, 1:k + 1] * np.asarray(sds, dtype=float)
    directions = _hadamard(int(np.log2(n)))[:, 1:k + 1] / np.sqrt(n)
    return np.asarray(mean, dtype=float) + scores @ directions.T, directions


# t < n takes the Gram branch of the ascent, t >= n the product branch. The means have both signs and magnitudes
# from below one to 100, so that a step that forgets the mean or applies it twice changes every result.
_BRANCHES = {
    "gram": (8, 16, np.arange(16) * 2.5 - 12.5),
    "product": (16, 4, np.array([10.5, -20.25, 0.75, 100.0])),
}

# With the default options (tol 1e-6 on the change of the objective, learning_rate 0.1) and the variances 64, 16, 4
# used below, the ascent for component i shrinks the tangent of its angle to the principal direction by
# r = (1 + 0.2 lambda_{i+1}) / (1 + 0.2 lambda_i) per step and stops when the objective changes by less than tol, which
# happens once sin^2 < tol / ((lambda_i - lambda_{i+1}) (1 / r^2 - 1)): at most 1.1e-7 for these data (for the third
# component, 4 against 0). So 1 - |cos| <= 1e-6 for every component, sin <= 1.5e-3, pairwise inner products of the
# components <= 3e-3, and the round trip on the fitted subspace is off by at most 3 * 1.5e-3 times |x - mean|_1.
_DEFAULT_COS_SLACK = 1e-6


def _alignment_gap(components, directions):
    """1 - |cos| between each fitted component and its principal direction."""
    return 1.0 - np.abs(np.sum(components * directions.T[:len(components)], axis=1))


class TestDeflationPCAPrincipalDirectionsOracle:
    """The components against principal directions known exactly, at the default options conformal_reach uses."""

    @pytest.mark.parametrize("branch", sorted(_BRANCHES))
    def test_components_are_the_principal_directions_at_the_default_options(self, branch):
        """On rank-3 data with variances 64, 16, 4 along known orthonormal directions, fit with the defaults stores the
        exact mean, three components that are the principal directions in order (1 - |cos| <= 1e-6, derived above) and
        nearly orthonormal, and transform followed by inverse_transform returns the training data (they lie on the
        fitted affine subspace) within the derived bound, while fit_transform equals transform. Catches an ascent that
        stops before it converges or steps against the gradient, a deflation that is not applied, a mean over all
        entries, a transform that does not centre, an inverse that forgets the mean, and a fit_transform that skips the
        centring; on the Gram branch (t < n) and on the product branch (t >= n)."""
        t, n, mean = _BRANCHES[branch]
        X, directions = _principal_data(t, n, (8.0, 4.0, 2.0), mean)
        pca = DeflationPCA(n_components=3)
        reduced = pca.fit_transform(X)
        np.testing.assert_array_equal(pca.mean_, mean)
        assert pca.components_.shape == (3, n) and pca.n_components_fitted_ == 3
        gaps = _alignment_gap(pca.components_, directions)
        assert np.all(gaps <= _DEFAULT_COS_SLACK), f"components are not the principal directions: 1 - |cos| = {gaps}"
        assert np.max(np.abs(pca.components_ @ pca.components_.T - np.eye(3))) <= 3e-3
        np.testing.assert_allclose(reduced, pca.transform(X), rtol=0, atol=1e-12 * (1 + np.abs(X).max()))
        z1 = np.abs(X - mean).sum(axis=1, keepdims=True)
        round_trip = pca.inverse_transform(reduced)
        assert np.all(np.abs(round_trip - X) <= 4.5e-3 * z1 + 1e-12 * (1 + np.abs(X))), (
            f"round trip off by {np.max(np.abs(round_trip - X))}")

    def test_full_rank_data_kept_at_full_rank_round_trip_every_point(self):
        """More components requested than min(t, n) on full-rank data in 4 dimensions (t = 16 > n): fit warns, keeps 4
        components that form an orthonormal basis, and inverse_transform(transform(x)) returns x for points far from
        the training data. tol = 0 (an option value that reads as false) makes the ascent run all max_iter = 2000 steps,
        so it converges to rounding: the round trip is checked to 1e-12 relative to |x - mean|_1. Catches a clamp to
        max(t, n) (16 rows), and any loss of orthonormality or of the mean in the round trip."""
        X, directions = _principal_data(16, 4, (8.0, 4.0, 2.0), np.array([10.5, -20.25, 0.75, 100.0]))
        X = X + np.outer(_hadamard(4)[:, 4], _hadamard(2)[:, 0] / 2.0)   # a fourth direction, variance 1
        pca = DeflationPCA(n_components=6, tol=0.0, max_iter=2000)
        with pytest.warns(UserWarning):
            pca.fit(X)
        assert pca.components_.shape == (4, 4) and pca.n_components_fitted_ == 4
        np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(4), rtol=0, atol=1e-12)
        points = np.array([[0.0, 0.0, 0.0, 0.0], [1e3, -1e3, 5.0, 0.5], [-7.0, 3.25, 11.0, -100.0]])
        round_trip = pca.inverse_transform(pca.transform(points))
        scale = 1.0 + np.abs(points - pca.mean_).sum(axis=1, keepdims=True)
        assert np.all(np.abs(round_trip - points) <= 1e-12 * scale), f"round trip off by {round_trip - points}"


class TestDeflationPCAStoredStateSoundness:
    """Promise point (2): conformal_reach maps bounds back through components_ and mean_ by interval arithmetic and
    maps the projections back through inverse_transform, so transform and inverse_transform must be exactly the maps
    those attributes define, on data other than the training data, and must keep no state from the data they see."""

    def test_transform_and_inverse_transform_are_the_maps_of_the_stored_mean_and_components(self):
        """For new points and for arbitrary reduced vectors (not the training data, far from them), transform is
        (x - mean_) @ components_.T and inverse_transform is r @ components_ + mean_, to 1e-12 relative; a 1-D argument
        gives the first row of the 2-D result; and neither call nor fit changes its argument. Catches a transform that
        does not centre or centres new data with their own mean, an inverse that forgets the mean, a 1-D reduced
        vector read as a column, and centring done in place on the caller's array (conformal_reach compares the very
        arrays it passed with the projections)."""
        X, _ = _principal_data(16, 4, (8.0, 4.0, 2.0), np.array([10.5, -20.25, 0.75, 100.0]))
        X_before = X.copy()
        pca = DeflationPCA(n_components=2).fit(X)
        np.testing.assert_array_equal(X, X_before)
        new = np.array([[1e3, -2.5, 7.0, 0.125], [-50.0, 60.0, -0.5, 300.0], [0.0, 0.0, 0.0, 0.0]])
        new_before = new.copy()
        reduced = pca.transform(new)
        np.testing.assert_array_equal(new, new_before)
        expected = (new - pca.mean_) @ pca.components_.T
        np.testing.assert_allclose(reduced, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(pca.transform(new[0]), expected[:1], rtol=1e-12, atol=1e-12)
        r = np.array([[3.0, -4.0], [0.5, 250.0]])
        r_before = r.copy()
        back = pca.inverse_transform(r)
        np.testing.assert_array_equal(r, r_before)
        expected_back = r @ pca.components_ + pca.mean_
        np.testing.assert_allclose(back, expected_back, rtol=1e-12, atol=1e-12)
        back_one = pca.inverse_transform(r[0])
        assert back_one.shape == (1, 4)
        np.testing.assert_allclose(back_one, expected_back[:1], rtol=1e-12, atol=1e-12)

    def test_a_second_fit_forgets_the_first(self):
        """The same object fitted twice answers for the second data only. A first fit on two samples clamps the three
        requested components to two, with a warning; a second fit on rank-3 data of 16 samples must keep all three
        requested components, store the exact mean of the new data and find their principal directions (bound derived
        above, which holds for any start). Catches a clamp that overwrites n_components, so that the object fits fewer
        components than requested ever after, and a mean or a component count kept from the first fit."""
        X, directions = _principal_data(16, 4, (8.0, 4.0, 2.0), np.array([10.5, -20.25, 0.75, 100.0]))
        pca = DeflationPCA(n_components=3)
        with pytest.warns(UserWarning):
            pca.fit(np.array([[1.0, -2.0, 3.0, 0.5], [-4.0, 5.0, 0.25, 7.0]]))
        if pca.n_components_fitted_ != 2:
            pytest.fail(f"the first fit should keep min(t, n) = 2 components, kept {pca.n_components_fitted_}")
        pca.fit(X)
        np.testing.assert_array_equal(pca.mean_, [10.5, -20.25, 0.75, 100.0])
        assert pca.components_.shape == (3, 4) and pca.n_components_fitted_ == 3
        gaps = _alignment_gap(pca.components_, directions)
        assert np.all(gaps <= _DEFAULT_COS_SLACK), f"components are not the principal directions: 1 - |cos| = {gaps}"


class TestDeflationPCAVarianceThresholdValues:
    """The early stop of Algorithm 1 on data whose variances are known exactly."""

    @pytest.mark.parametrize("threshold, kept", [(1.0, 2), (0.1, 3), (None, 4)])
    def test_threshold_keeps_exactly_the_components_above_it(self, threshold, kept):
        """Rank-3 data with variances 16, 4, 0.25 along known directions and a mean far from zero, four components
        requested: a threshold of 1 keeps 2 components, 0.1 keeps 3 (the fourth has no variance), none keeps all 4.
        Every kept row is a unit vector, n_components_fitted_ and the width of transform agree, and the first two
        kept components are the principal directions (bound derived as above: sin^2 < 1.4e-7 for these variances).
        Catches a stop that keeps an extra all-zero row, a residual variance computed on the uncentred data (it never
        falls below the threshold), and a count that reports the requested number instead of the fitted one."""
        X, directions = _principal_data(16, 4, (4.0, 2.0, 0.5), np.array([10.5, -20.25, 0.75, 100.0]))
        pca = DeflationPCA(n_components=4, variance_threshold=threshold).fit(X)
        assert pca.components_.shape == (kept, 4)
        assert pca.n_components_fitted_ == kept
        np.testing.assert_allclose(np.linalg.norm(pca.components_, axis=1), np.ones(kept), rtol=0, atol=1e-12)
        assert pca.transform(X).shape == (16, kept)
        assert np.all(_alignment_gap(pca.components_[:2], directions) <= _DEFAULT_COS_SLACK)


class TestDeflationPCAOptionsAndBranches:
    """The options of the ascent and its branches, with the random start replaced by a fixed one so that the result
    of the documented iteration can be computed independently: start from a / |a|, then max_iter times (or until the
    objective changes by less than tol) a <- a + learning_rate (2 / t) Z^T Z a, normalised."""

    @pytest.mark.parametrize("branch, learning_rate, max_iter, tol, steps", [
        ("gram", 0.05, 7, 0.0, 7),       # tol 0 never stops early: exactly max_iter steps
        ("product", 0.3, 5, 0.0, 5),
        ("product", 0.1, 50, 1e9, 2),    # the first step is compared with -inf; the second change is below tol
    ])
    def test_options_set_the_ascent_exactly(self, branch, learning_rate, max_iter, tol, steps, monkeypatch):
        """With the start fixed, the first component is the documented iteration run for the expected number of steps,
        to 1e-12. Catches an ignored learning_rate, max_iter or tol, a gradient without its factor 2 (Gram branch) or
        scaled by 1/n instead of 1/t (product branch), a step against the gradient, and an inverted convergence
        test."""
        t, n, mean = _BRANCHES[branch]
        X, _ = _principal_data(t, n, (8.0, 4.0, 2.0), mean)
        start = np.linspace(-1.0, 2.0, n) + 0.25
        calls = []

        def fixed_start(*shape):
            calls.append(shape)
            return start.copy()

        monkeypatch.setattr(np.random, "randn", fixed_start)
        pca = DeflationPCA(n_components=1, max_iter=max_iter, tol=tol, learning_rate=learning_rate).fit(X)
        if calls != [(n,)]:
            pytest.fail(f"the start is no longer drawn once by np.random.randn(n) (calls {calls}); update the stub")
        Z = X - mean
        a = start / np.linalg.norm(start)
        for _ in range(steps):
            a = a + learning_rate * (2.0 / t) * (Z.T @ (Z @ a))
            a = a / np.linalg.norm(a)
        np.testing.assert_allclose(pca.components_[0], a, rtol=0, atol=1e-12)

    def test_memory_error_on_the_gram_matrix_falls_back_to_the_product_branch(self):
        """When building the n-by-n Gram matrix raises MemoryError (as numpy does when it does not fit), the ascent
        continues on the product branch and still finds the principal directions. The data are an ndarray subclass whose
        product of an n-by-t array with a t-by-n array raises MemoryError, a deterministic stand-in for the allocation
        failure. Catches a fallback that keeps the Gram branch (it then reads a matrix that was never built)."""

        class GramRefusingArray(np.ndarray):
            refused = 0

            def __matmul__(self, other):
                if np.ndim(other) == 2 and self.ndim == 2 and self.shape[0] > self.shape[1] == np.shape(other)[0]:
                    type(self).refused += 1
                    raise MemoryError("stand-in: the n-by-n Gram matrix does not fit")
                return np.matmul(np.asarray(self), np.asarray(other))

        t, n, mean = _BRANCHES["gram"]
        X, directions = _principal_data(t, n, (8.0, 4.0, 2.0), mean)
        pca = DeflationPCA(n_components=3).fit(X.view(GramRefusingArray))
        if GramRefusingArray.refused == 0:
            pytest.fail("the Gram matrix was not attempted; the test no longer reaches the MemoryError fallback")
        gaps = _alignment_gap(np.asarray(pca.components_), directions)
        assert np.all(gaps <= _DEFAULT_COS_SLACK), f"1 - |cos| = {gaps}"


class TestDeflationPCADefects:
    """Defects of today's library, each with its passing neighbour."""

    @known_bug(13, "DeflationPCA.fit crashes with UnboundLocalError when t < n and t*n >= 1e8 (the regime the class is "
                   "written for, reached from conformal_reach with pca_components on wide outputs): the Gram flag "
                   "use_gram = (t < n) stays on although the Gram matrix is only built when t*n < 1e8",
               raises=UnboundLocalError)
    def test_fit_answers_when_samples_times_dimension_reach_1e8(self):
        """t = 10 samples in n = 10**7 dimensions (t*n = 1e8), held in float32 (400 MB) to keep the case small: rank-1
        data 3 + s_i u with u a dyadic unit vector, so the data, their mean and the centred data are exact and the
        principal direction is u. The 1e8 entries are what the defect needs; the case crashes within a second today
        and takes a few seconds once the fit answers. The neighbour with t*n < 1e8 is the Gram branch case of
        TestDeflationPCAPrincipalDirectionsOracle."""
        t, n = 10, 10_000_000
        u = np.zeros(n, dtype=np.float32)
        u[::610] = np.float32(2.0 ** -7)                  # 16394 entries of 1/128: norm 1.0003, normalised below
        u_unit = u.astype(np.float64) / np.linalg.norm(u.astype(np.float64))
        s = (np.arange(t, dtype=np.float32) - np.float32(4.5)) * np.float32(8.0)
        X = np.float32(3.0) + np.outer(s, u)
        pca = DeflationPCA(n_components=1, max_iter=50).fit(X)
        assert 1.0 - abs(float(pca.components_[0].astype(np.float64) @ u_unit)) <= 1e-6

    @known_bug(13, "DeflationPCA.fit with t < n builds the n-by-n matrix Z^T Z (n^2 8 bytes, 80 GB for n = 100000; the "
                   "measured peak is twice that, 64 MB for t = 5 samples in n = 2000 dimensions), the O(n^2) memory the "
                   "class docstring says deflation avoids")
    def test_fit_memory_grows_with_samples_times_dimension_not_dimension_squared(self):
        """The class promises PCA without the n-by-n covariance: for t = 5 samples in n = 2000 dimensions (80 kB of
        data), the peak memory numpy reports to tracemalloc during fit stays below 16 times the data plus 1 MB (a
        correct implementation needs a few t-by-n and n-sized arrays and a t-by-t matrix at most). The neighbour on
        small data is any fit of this file. max_iter = 2 keeps the case fast: the peak is reached when the Gram matrix is
        built, before the first step of the ascent, so the number of steps does not change what is measured."""
        X = np.random.randn(5, 2000)
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            base, _ = tracemalloc.get_traced_memory()
            DeflationPCA(n_components=1, max_iter=2).fit(X)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not was_tracing:
                tracemalloc.stop()
        assert peak - base <= 16 * X.nbytes + 2 ** 20, f"peak {(peak - base) / 1e6:.1f} MB for {X.nbytes / 1e3:.0f} kB"

    @known_bug(13, "DeflationPCA compares the absolute change of the objective with tol and steps with a fixed "
                   "learning_rate, so with the default options (the ones conformal_reach uses) on data of small scale "
                   "the ascent stops after its second step at a nearly random direction")
    def test_components_do_not_depend_on_the_scale_of_the_data(self):
        """The principal directions of c X are those of X. The data of the oracle above divided by 128 (top standard
        deviation 1/16, all values still dyadic), fitted with the default options for five starts, must give the same
        principal directions within the bound that holds at scale 1 (1 - |cos| <= 1e-6). The neighbour at scale 1 is
        test_components_are_the_principal_directions_at_the_default_options."""
        t, n, mean = _BRANCHES["gram"]
        X, directions = _principal_data(t, n, (8.0 / 128, 4.0 / 128, 2.0 / 128), mean / 128)
        worst = []
        for seed in range(5):
            np.random.seed(seed)
            pca = DeflationPCA(n_components=3).fit(X)
            worst.append(float(np.max(_alignment_gap(pca.components_, directions))))
        assert max(worst) <= _DEFAULT_COS_SLACK, f"1 - |cos| per start: {worst}"

    @pytest.mark.parametrize("width", [
        pytest.param(1, marks=known_bug(
            13, "DeflationPCA.transform answers for a width-1 array after a fit in several dimensions: X - mean_ "
                "broadcasts the column across all coordinates", raises=DID_NOT_RAISE)),
        2,
    ])
    def test_transform_refuses_a_width_that_does_not_fit(self, width):
        """After a fit in 4 dimensions, transform of an array of another width must refuse with ValueError. Width 2
        refuses today (numpy cannot broadcast it) and is the passing neighbour."""
        X, _ = _principal_data(16, 4, (8.0, 4.0, 2.0), np.array([10.5, -20.25, 0.75, 100.0]))
        pca = DeflationPCA(n_components=2).fit(X)
        with pytest.raises(ValueError):
            pca.transform(np.ones((3, width)))

    @known_bug(13, "DeflationPCA.fit on an empty array (0 rows) answers a NaN mean and no component instead of "
                   "refusing", raises=DID_NOT_RAISE)
    def test_fit_refuses_an_empty_training_set(self):
        """fit on zero rows must refuse with ValueError: the mean of nothing is undefined. NumPy's warnings about the
        empty mean and fit's own warning about the clamp are silenced so that the refusal is what is checked. The
        neighbour on one row is TestDeflationPCAFit::test_fit_handles_1d_input."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError):
                DeflationPCA(n_components=2).fit(np.zeros((0, 4)))


class _AffineRecordingModel:
    """A numpy model for conformal_reach with six outputs near a two-dimensional affine subspace (a small nonlinear
    residual outside it), keeping every batch of outputs it returns."""

    W = np.array([[1.0, -2.0, 0.5, 3.0, -1.0, 2.0], [0.5, 1.0, -1.5, 0.25, 2.0, -3.0]])
    RESIDUAL = np.array([1.0, -1.0, 2.0, 0.5, -0.5, 1.0])

    def __init__(self):
        self.calls = []

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        y = x @ self.W + 10.0 + 0.05 * np.sin(3.0 * x[:, :1]) * self.RESIDUAL
        self.calls.append(y.copy())
        return y


class TestConformalReachWithPCARouteSoundness:
    """Promise point (2) through the public entry: with pca_components set, conformal_reach maps the surrogate box back
    through components_ and mean_ and the projections through inverse_transform. With ell = m every calibration output
    lies in the returned box exactly when every mapped prediction lies in the mapped box (the inflation covers every
    calibration error); no statistics are needed."""

    T, M = 40, 60

    @pytest.mark.parametrize("surrogate", ["clipping_block", "naive"])
    def test_every_calibration_output_is_inside_the_box_when_ell_is_m(self, surrogate):
        """conformal_reach(pca_components=2, ell=m) on six outputs returns a box that holds every calibration output.
        Slack: HiGHS's primal feasibility tolerance 1e-7 times sqrt(2) times the sum of the training outputs' distances
        to their mean (how far a projection built from weights within that tolerance can move after the map back), plus
        a float64 pad of 1e-12. Catches bounds of the reduced surrogate that do not hold its predictions and a map back
        of the projections that is not the map of the bounds."""
        model = _AffineRecordingModel()
        result = conformal_reach(model, Box(np.array([-2.0, 0.5]), np.array([3.0, 4.0])), surrogate=surrogate,
                                 m=self.M, ell=self.M, training_samples=self.T, pca_components=2, batch_size=1000,
                                 epsilon=0.1, seed=5)
        if [c.shape[0] for c in model.calls] != [self.T, self.M]:
            pytest.fail(f"expected one training call of {self.T} rows and one calibration call of {self.M} rows, got "
                        f"{[c.shape for c in model.calls]}")
        training, calibration = model.calls
        lb = np.asarray(result.lb, dtype=float).ravel()
        ub = np.asarray(result.ub, dtype=float).ravel()
        spread = np.linalg.norm(training - training.mean(axis=0), axis=1).sum()
        slack = 1e-7 * np.sqrt(2.0) * spread + 1e-12 * (1.0 + np.abs(calibration) + np.abs(lb) + np.abs(ub))
        escape = np.maximum(lb - calibration, calibration - ub) - slack
        assert np.all(escape <= 0.0), f"{int(np.sum(np.any(escape > 0, axis=1)))} calibration outputs outside the box"
