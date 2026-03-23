"""
Unit tests for surrogate methods.
"""

import pytest
import numpy as np

from n2v.probabilistic.surrogates.base import Surrogate
from n2v.probabilistic.surrogates.naive import NaiveSurrogate
from n2v.probabilistic.surrogates.clipping_block import (
    ClippingBlockSurrogate,
    BatchedClippingBlockSurrogate
)


class TestNaiveSurrogateFit:
    """Tests for NaiveSurrogate fit() method."""

    def test_fit_computes_center_correctly(self):
        """Test that fit() computes the mean correctly."""
        training_outputs = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ])

        surrogate = NaiveSurrogate()
        surrogate.fit(training_outputs)

        expected_center = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(surrogate.center, expected_center)

    def test_fit_sets_dimension(self):
        """Test that fit() sets n_dim correctly."""
        training_outputs = np.random.randn(50, 10)

        surrogate = NaiveSurrogate()
        surrogate.fit(training_outputs)

        assert surrogate.n_dim == 10

    def test_fit_sets_fitted_flag(self):
        """Test that fit() sets _is_fitted flag."""
        training_outputs = np.random.randn(50, 5)

        surrogate = NaiveSurrogate()
        assert not surrogate._is_fitted

        surrogate.fit(training_outputs)
        assert surrogate._is_fitted

    def test_fit_handles_1d_input(self):
        """Test that fit() handles 1D input (single sample)."""
        training_outputs = np.array([1.0, 2.0, 3.0])

        surrogate = NaiveSurrogate()
        surrogate.fit(training_outputs)

        np.testing.assert_array_almost_equal(surrogate.center, training_outputs)


class TestNaiveSurrogatePredict:
    """Tests for NaiveSurrogate predict() method."""

    def test_predict_returns_center_for_all_inputs(self):
        """Test that predict() returns center for all inputs."""
        training_outputs = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        center = np.array([2.0, 3.0])

        calibration_outputs = np.array([
            [0.0, 0.0],
            [5.0, 5.0],
            [100.0, -100.0]
        ])

        surrogate = NaiveSurrogate()
        surrogate.fit(training_outputs)
        predictions = surrogate.predict(calibration_outputs)

        # All predictions should be the center
        assert predictions.shape == (3, 2)
        for i in range(3):
            np.testing.assert_array_almost_equal(predictions[i], center)

    def test_predict_raises_if_not_fitted(self):
        """Test that predict() raises if not fitted."""
        surrogate = NaiveSurrogate()

        with pytest.raises(RuntimeError, match="fitted"):
            surrogate.predict(np.array([[1.0, 2.0]]))

    def test_predict_handles_1d_input(self):
        """Test that predict() handles 1D input (single sample)."""
        training_outputs = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])

        surrogate = NaiveSurrogate()
        surrogate.fit(training_outputs)

        single_output = np.array([10.0, 20.0])
        predictions = surrogate.predict(single_output)

        assert predictions.shape == (1, 2)


class TestNaiveSurrogateGetBounds:
    """Tests for NaiveSurrogate get_bounds() method."""

    def test_get_bounds_returns_center_center(self):
        """Test that get_bounds() returns (center, center)."""
        training_outputs = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        expected_center = np.array([2.5, 3.5, 4.5])

        surrogate = NaiveSurrogate()
        surrogate.fit(training_outputs)

        lb, ub = surrogate.get_bounds()

        np.testing.assert_array_almost_equal(lb, expected_center)
        np.testing.assert_array_almost_equal(ub, expected_center)
        np.testing.assert_array_almost_equal(lb, ub)

    def test_get_bounds_raises_if_not_fitted(self):
        """Test that get_bounds() raises if not fitted."""
        surrogate = NaiveSurrogate()

        with pytest.raises(RuntimeError, match="fitted"):
            surrogate.get_bounds()


class TestClippingBlockSurrogateFit:
    """Tests for ClippingBlockSurrogate fit() method."""

    def test_fit_stores_vertices(self):
        """Test that fit() stores training outputs as vertices."""
        training_outputs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        surrogate = ClippingBlockSurrogate()
        surrogate.fit(training_outputs)

        np.testing.assert_array_equal(surrogate.vertices, training_outputs)
        assert surrogate.n_samples == 4
        assert surrogate.n_dim == 2

    def test_fit_computes_bounds(self):
        """Test that fit() computes min/max bounds of vertices."""
        training_outputs = np.array([
            [0.0, 1.0],
            [2.0, 0.0],
            [1.0, 3.0]
        ])

        surrogate = ClippingBlockSurrogate()
        surrogate.fit(training_outputs)

        expected_lb = np.array([0.0, 0.0])
        expected_ub = np.array([2.0, 3.0])

        np.testing.assert_array_almost_equal(surrogate.lb, expected_lb)
        np.testing.assert_array_almost_equal(surrogate.ub, expected_ub)


class TestClippingBlockSurrogatePredict:
    """Tests for ClippingBlockSurrogate predict() method."""

    def test_projection_is_inside_convex_hull(self):
        """Test that projection is inside (or on boundary of) convex hull."""
        # Simple triangle in 2D
        training_outputs = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 2.0]
        ])

        surrogate = ClippingBlockSurrogate(n_workers=1)
        surrogate.fit(training_outputs)

        # Point outside the triangle
        calibration_outputs = np.array([
            [3.0, 3.0],  # Outside
            [0.5, 0.5],  # Inside
            [-1.0, -1.0]  # Outside
        ])

        projections = surrogate.predict(calibration_outputs)

        # Check projections are inside bounds
        lb, ub = surrogate.get_bounds()
        for i in range(3):
            assert np.all(projections[i] >= lb - 1e-6)
            assert np.all(projections[i] <= ub + 1e-6)

    def test_projection_of_vertex_is_itself(self):
        """Test that projecting a vertex returns itself."""
        training_outputs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ])

        surrogate = ClippingBlockSurrogate(n_workers=1)
        surrogate.fit(training_outputs)

        # Project the vertices themselves
        projections = surrogate.predict(training_outputs)

        np.testing.assert_array_almost_equal(projections, training_outputs, decimal=5)

    def test_projection_inside_hull_is_itself(self):
        """Test that projecting a point inside the hull returns itself."""
        # Unit square
        training_outputs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        surrogate = ClippingBlockSurrogate(n_workers=1)
        surrogate.fit(training_outputs)

        # Point inside the square
        inside_point = np.array([[0.5, 0.5]])
        projection = surrogate.predict(inside_point)

        np.testing.assert_array_almost_equal(projection, inside_point, decimal=5)

    def test_projection_minimizes_linf_distance(self):
        """Test that projection achieves minimum L∞ distance to convex hull."""
        # Unit square in 2D
        training_outputs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        surrogate = ClippingBlockSurrogate(n_workers=1)
        surrogate.fit(training_outputs)

        # Point outside to the right of the square
        outside_point = np.array([[2.0, 0.5]])
        projection = surrogate.predict(outside_point)

        # The minimum L∞ distance to the unit square from (2, 0.5) is 1.0
        # The projection could be any point on the right edge (1, y) for y in [0, 1]
        # The LP may return a vertex or an interior point depending on solver
        linf_distance = np.max(np.abs(outside_point - projection))
        np.testing.assert_almost_equal(linf_distance, 1.0, decimal=5)

        # Verify projection is in convex hull
        lb, ub = surrogate.get_bounds()
        assert np.all(projection >= lb - 1e-6)
        assert np.all(projection <= ub + 1e-6)

    def test_predict_raises_if_not_fitted(self):
        """Test that predict() raises if not fitted."""
        surrogate = ClippingBlockSurrogate()

        with pytest.raises(RuntimeError, match="fitted"):
            surrogate.predict(np.array([[1.0, 2.0]]))

    def test_predict_raises_if_wrong_dimension(self):
        """Test that predict() raises if dimension doesn't match."""
        training_outputs = np.random.randn(10, 5)
        surrogate = ClippingBlockSurrogate()
        surrogate.fit(training_outputs)

        wrong_dim_outputs = np.random.randn(5, 3)  # Wrong dimension

        with pytest.raises(ValueError, match="dimension"):
            surrogate.predict(wrong_dim_outputs)


class TestClippingBlockSurrogateGetBounds:
    """Tests for ClippingBlockSurrogate get_bounds() method."""

    def test_get_bounds_returns_min_max_of_vertices(self):
        """Test that get_bounds() returns element-wise min/max of vertices."""
        training_outputs = np.array([
            [0.0, 1.0, 2.0],
            [3.0, 0.0, 1.0],
            [1.0, 2.0, 0.0]
        ])

        surrogate = ClippingBlockSurrogate()
        surrogate.fit(training_outputs)

        lb, ub = surrogate.get_bounds()

        expected_lb = np.array([0.0, 0.0, 0.0])
        expected_ub = np.array([3.0, 2.0, 2.0])

        np.testing.assert_array_almost_equal(lb, expected_lb)
        np.testing.assert_array_almost_equal(ub, expected_ub)

    def test_get_bounds_raises_if_not_fitted(self):
        """Test that get_bounds() raises if not fitted."""
        surrogate = ClippingBlockSurrogate()

        with pytest.raises(RuntimeError, match="fitted"):
            surrogate.get_bounds()


class TestBatchedClippingBlockSurrogate:
    """Tests for BatchedClippingBlockSurrogate."""

    def test_batched_same_results_as_non_batched(self):
        """Test that batched version produces same results."""
        np.random.seed(42)

        training_outputs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        calibration_outputs = np.random.rand(20, 2) * 2 - 0.5  # Some outside

        # Non-batched
        surrogate_normal = ClippingBlockSurrogate(n_workers=1)
        surrogate_normal.fit(training_outputs)
        projections_normal = surrogate_normal.predict(calibration_outputs)

        # Batched
        surrogate_batched = BatchedClippingBlockSurrogate(batch_size=5, n_workers=1)
        surrogate_batched.fit(training_outputs)
        projections_batched = surrogate_batched.predict(calibration_outputs)

        np.testing.assert_array_almost_equal(
            projections_normal, projections_batched, decimal=5
        )

    def test_batched_handles_small_batches(self):
        """Test batched version with batch_size larger than data."""
        training_outputs = np.random.randn(5, 3)
        calibration_outputs = np.random.randn(3, 3)  # Smaller than batch_size

        surrogate = BatchedClippingBlockSurrogate(batch_size=10)
        surrogate.fit(training_outputs)
        projections = surrogate.predict(calibration_outputs)

        assert projections.shape == calibration_outputs.shape


class TestSurrogateComputeErrors:
    """Tests for Surrogate.compute_errors() base class method."""

    def test_compute_errors_naive(self):
        """Test compute_errors() for NaiveSurrogate."""
        training_outputs = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        center = np.array([2.0, 3.0])

        calibration_outputs = np.array([
            [2.0, 3.0],  # Same as center -> error = 0
            [4.0, 5.0],  # error = [2, 2]
        ])

        surrogate = NaiveSurrogate()
        surrogate.fit(training_outputs)

        errors = surrogate.compute_errors(calibration_outputs)

        expected_errors = np.array([
            [0.0, 0.0],
            [2.0, 2.0]
        ])

        np.testing.assert_array_almost_equal(errors, expected_errors)

    def test_compute_errors_clipping_block(self):
        """Test compute_errors() for ClippingBlockSurrogate."""
        # Unit square
        training_outputs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        # Point inside: error should be 0
        # Point outside: error = point - projection
        calibration_outputs = np.array([
            [0.5, 0.5],  # Inside, error = 0
            [2.0, 0.5],  # Outside, projects to (1.0, 0.5), error = (1.0, 0.0)
        ])

        surrogate = ClippingBlockSurrogate(n_workers=1)
        surrogate.fit(training_outputs)

        errors = surrogate.compute_errors(calibration_outputs)

        # First point is inside, error should be ~0
        np.testing.assert_array_almost_equal(errors[0], [0.0, 0.0], decimal=5)

        # Second point projects to boundary
        assert errors[1, 0] > 0  # Should have positive error in x dimension

    def test_compute_errors_with_precomputed_projections(self):
        """Test compute_errors() with pre-computed projections."""
        training_outputs = np.random.randn(10, 3)
        calibration_outputs = np.random.randn(5, 3)

        surrogate = NaiveSurrogate()
        surrogate.fit(training_outputs)

        projections = surrogate.predict(calibration_outputs)
        errors_with_proj = surrogate.compute_errors(calibration_outputs, projections)
        errors_without_proj = surrogate.compute_errors(calibration_outputs)

        np.testing.assert_array_almost_equal(errors_with_proj, errors_without_proj)


class TestSurrogateInterface:
    """Tests for Surrogate abstract base class interface."""

    def test_naive_is_surrogate(self):
        """Test that NaiveSurrogate implements Surrogate interface."""
        assert issubclass(NaiveSurrogate, Surrogate)

    def test_clipping_block_is_surrogate(self):
        """Test that ClippingBlockSurrogate implements Surrogate interface."""
        assert issubclass(ClippingBlockSurrogate, Surrogate)

    def test_batched_is_clipping_block(self):
        """Test that BatchedClippingBlockSurrogate inherits from ClippingBlockSurrogate."""
        assert issubclass(BatchedClippingBlockSurrogate, ClippingBlockSurrogate)


class TestClippingBlockProcessPool:
    """Test that clipping block uses process-based parallelism."""

    def test_parallel_predict_matches_sequential(self):
        """Parallel and sequential predictions should give identical results."""
        np.random.seed(42)
        training = np.random.randn(15, 4)
        calibration = np.random.randn(10, 4)

        surr_seq = ClippingBlockSurrogate(n_workers=1)
        surr_seq.fit(training)
        proj_seq = surr_seq.predict(calibration)

        surr_par = ClippingBlockSurrogate(n_workers=2)
        surr_par.fit(training)
        proj_par = surr_par.predict(calibration)

        np.testing.assert_allclose(proj_seq, proj_par, atol=1e-6)


class TestParallelization:
    """Tests for parallel processing in ClippingBlockSurrogate."""

    def test_parallel_same_as_sequential(self):
        """Test that parallel processing gives same results as sequential."""
        np.random.seed(123)

        training_outputs = np.random.randn(20, 5)
        calibration_outputs = np.random.randn(30, 5)

        # Sequential
        surrogate_seq = ClippingBlockSurrogate(n_workers=1)
        surrogate_seq.fit(training_outputs)
        projections_seq = surrogate_seq.predict(calibration_outputs)

        # Parallel
        surrogate_par = ClippingBlockSurrogate(n_workers=4)
        surrogate_par.fit(training_outputs)
        projections_par = surrogate_par.predict(calibration_outputs)

        np.testing.assert_array_almost_equal(projections_seq, projections_par, decimal=5)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_training_sample(self):
        """Test with single training sample."""
        training_outputs = np.array([[1.0, 2.0, 3.0]])

        # Naive
        surrogate_naive = NaiveSurrogate()
        surrogate_naive.fit(training_outputs)
        np.testing.assert_array_equal(surrogate_naive.center, training_outputs[0])

        # Clipping block - convex hull of single point is the point itself
        surrogate_clipping = ClippingBlockSurrogate(n_workers=1)
        surrogate_clipping.fit(training_outputs)
        projection = surrogate_clipping.predict(np.array([[5.0, 5.0, 5.0]]))
        np.testing.assert_array_almost_equal(projection, training_outputs, decimal=5)

    def test_single_dimension(self):
        """Test with single output dimension."""
        training_outputs = np.array([[1.0], [2.0], [3.0]])

        surrogate = ClippingBlockSurrogate(n_workers=1)
        surrogate.fit(training_outputs)

        lb, ub = surrogate.get_bounds()
        assert lb[0] == 1.0
        assert ub[0] == 3.0

        # Point outside should project to boundary
        projection = surrogate.predict(np.array([[5.0]]))
        np.testing.assert_almost_equal(projection[0, 0], 3.0, decimal=5)  # Should project to max

    def test_high_dimensional(self):
        """Test with high-dimensional data."""
        np.random.seed(42)

        n_dim = 100
        training_outputs = np.random.randn(50, n_dim)
        calibration_outputs = np.random.randn(10, n_dim)

        surrogate = ClippingBlockSurrogate(n_workers=1)
        surrogate.fit(training_outputs)
        projections = surrogate.predict(calibration_outputs)

        assert projections.shape == (10, n_dim)
