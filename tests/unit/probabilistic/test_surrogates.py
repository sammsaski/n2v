"""
Unit tests for surrogate methods.
"""

import concurrent.futures
import itertools
import warnings
from types import SimpleNamespace

import pytest
import numpy as np

from n2v.probabilistic import conformal_reach
from n2v.probabilistic.surrogates import clipping_block as clipping_block_module
from n2v.probabilistic.surrogates.base import Surrogate
from n2v.probabilistic.surrogates.naive import NaiveSurrogate
from n2v.probabilistic.surrogates.clipping_block import (
    ClippingBlockSurrogate,
    BatchedClippingBlockSurrogate,
    _project_single_standalone,
)
from n2v.sets import Box
from tests.known_bugs import DID_NOT_RAISE, known_bug


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


# ---------------------------------------------------------------------------
# Independent oracles for the projection,
# the bounds as the conformal box uses them, the state kept between calls,
# the public route through conformal_reach, refusals, and today's defects.
# ---------------------------------------------------------------------------

# A right triangle with legs of different lengths (4 along x, 2 along y) and its corner at (-1, -0.5), so that
# coordinates have both signs and magnitudes above and below one. Its hull differs from its bounding box.
_TRIANGLE_CORNER = np.array([-1.0, -0.5])
_TRIANGLE = _TRIANGLE_CORNER + np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 2.0]])
# Points and their least L-infinity distance to the triangle, derived by hand in the frame q = point - corner,
# where the triangle is q_x >= 0, q_y >= 0, q_x / 4 + q_y / 2 <= 1. Beyond the slanted edge the L-infinity ball
# first touches it with its lower-left corner q - d (1, 1), so d = (q_x / 4 + q_y / 2 - 1) / (1 / 4 + 1 / 2) as
# long as q - d (1, 1) has no negative coordinate; otherwise the contact is on a leg or at a vertex.
_TRIANGLE_CASES = [
    ([0.0, 0.0], 0.0),         # inside (q = (1, 0.5)): projects to itself
    ([1.0, 0.5], 0.0),         # on the slanted edge (q = (2, 1)): projects to itself
    ([3.0, 2.5], 2.0),         # q = (4, 3): d = 1.5 / 0.75, nearest point q = (2, 1)
    ([9.0, 5.5], 6.0),         # q = (10, 6): d = 4.5 / 0.75, nearest point the vertex q = (4, 0)
    ([5.0, -1.5], 2.0),        # q = (6, -1): q_x <= 4 needs d >= 2, then the vertex q = (4, 0)
    ([-2.0, -3.5], 3.0),       # q = (-1, -3): q_y >= 0 needs d >= 3; the nearest points form a segment of the x-leg
    ([-3.0, 0.5], 2.0),        # q = (-2, 1): q_x >= 0 needs d >= 2; a segment of the y-leg
    ([0.5, 2.0], 5.0 / 6.0),   # q = (1.5, 2.5): d = 0.625 / 0.75, nearest point q = (2/3, 5/3)
    ([99.0, -50.5], 96.0),     # q = (100, -50): q_x <= 4 needs d >= 96, then the vertex q = (4, 0)
]

# A box hull in three dimensions given by its eight corners, of different widths per coordinate. The least
# L-infinity distance to a box is max_k max(lo_k - y_k, y_k - hi_k, 0).
_BOX_LO = np.array([-3.0, 0.5, -0.25])
_BOX_HI = np.array([2.0, 4.0, 0.25])
_BOX_CORNERS = np.array(list(itertools.product(*zip(_BOX_LO, _BOX_HI))))
_BOX_POINTS = np.array([
    [0.0, 1.0, 0.0],           # inside
    [2.0, 3.0, 0.25],          # on a face and an edge
    [5.0, 2.0, 0.0],           # beyond one face
    [-10.0, 7.0, 1.0],         # beyond a corner
    [100.0, -100.0, 50.0],     # far away
    [0.5, 0.25, 0.5],          # just outside two faces
])

# Membership and distance are compared with an absolute slack of 1e-9 (the points have magnitude at most about
# 100): the LP is solved by HiGHS in float64 and its vertex solutions are exact to rounding on these data.
_PROJ_TOL = 1e-9

_ROUTES = {
    # route name -> a fresh surrogate; the batched routes cover a partial last batch (and a last batch of one
    # row, which takes the serial branch although n_workers > 1), a single batch shorter than batch_size, and a row
    # count that is a multiple of batch_size (9 triangle points and 6 box points in batches of 3, each batch in the
    # process pool), where the last batch ends exactly at the last row.
    "serial": lambda: ClippingBlockSurrogate(n_workers=1),
    "parallel": lambda: ClippingBlockSurrogate(n_workers=2),
    "batched-partial-last-batch": lambda: BatchedClippingBlockSurrogate(batch_size=4, n_workers=2),
    "batched-one-short-batch": lambda: BatchedClippingBlockSurrogate(batch_size=64, n_workers=1),
    "batched-full-batches": lambda: BatchedClippingBlockSurrogate(batch_size=3, n_workers=2),
}


def _in_triangle(p, tol=_PROJ_TOL):
    q = np.asarray(p, dtype=float) - _TRIANGLE_CORNER
    return q[0] >= -tol and q[1] >= -tol and q[0] / 4.0 + q[1] / 2.0 <= 1.0 + tol


def _box_distance(y):
    return float(np.max(np.maximum(np.maximum(_BOX_LO - y, y - _BOX_HI), 0.0)))


class TestClippingBlockProjectionOracle:
    """The projection against closed forms, on every route that computes it."""

    @pytest.mark.parametrize("route", sorted(_ROUTES))
    @pytest.mark.parametrize("hull", ["triangle", "box3d"])
    def test_projection_is_a_nearest_point_of_the_hull(self, hull, route):
        """Every projection is a point of the hull whose L-infinity distance to the output is the least distance from
        the output to the hull, derived by hand (triangle) or by the closed form of a box (box3d), and it lies inside
        the box get_bounds returns. Catches an LP that encodes another problem (a sign slip in a constraint, weights
        read from the wrong slice of the solution, weights left unbounded so the projection goes onto the affine hull)
        on the serial branch, the process-pool branch and the batched surrogate, and a batched predict that leaves the
        rows of a partial last batch unprojected."""
        surrogate = _ROUTES[route]()
        if hull == "triangle":
            surrogate.fit(_TRIANGLE)
            points = np.array([p for p, _ in _TRIANGLE_CASES])
            distances = [d for _, d in _TRIANGLE_CASES]
            inside = _in_triangle
        else:
            surrogate.fit(_BOX_CORNERS)
            points = _BOX_POINTS
            distances = [_box_distance(y) for y in _BOX_POINTS]
            inside = lambda p: bool(np.all(p >= _BOX_LO - _PROJ_TOL) and np.all(p <= _BOX_HI + _PROJ_TOL))
        projections = surrogate.predict(points)
        assert projections.shape == points.shape
        lb, ub = surrogate.get_bounds()
        for y, p, d in zip(points, projections, distances):
            assert inside(p), f"projection {p} of {y} is not a point of the hull"
            assert abs(np.max(np.abs(y - p)) - d) <= _PROJ_TOL, (
                f"projection {p} of {y} is at L-infinity distance {np.max(np.abs(y - p))}, the least is {d}")
            assert np.all(p >= lb - _PROJ_TOL) and np.all(p <= ub + _PROJ_TOL), (
                f"projection {p} lies outside the bounds [{lb}, {ub}] of the surrogate")


class TestSurrogateBoundsHoldPredictionsSoundness:
    """Promise point (1): conformal_reach adds the inflation to get_bounds, so the bounds must hold every training
    output and every prediction of the surrogate itself."""

    # Per-coordinate minima (-3, -1, -6) and maxima (2, 7, 3) all differ from each other and from the extremes over
    # all entries, so a minimum or maximum taken over the wrong axis changes the answer.
    TRAINING = np.array([[-3.0, 7.0, 0.5], [2.0, -1.0, 0.25], [0.5, 4.0, -6.0], [1.0, 2.0, 3.0]])
    LB = np.array([-3.0, -1.0, -6.0])
    UB = np.array([2.0, 7.0, 3.0])
    POINTS = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [-10.0, 20.0, -30.0], [1.0, 2.0, 3.0], [2.0, -1.0, -6.0]])

    def test_clipping_block_bounds_are_the_coordinatewise_extremes_and_hold_every_projection(self):
        """get_bounds of the clipping block, built as conformal_reach builds it (batched, process pool), returns the
        per-coordinate minimum and maximum of the training outputs exactly, and every training output and every
        projection lies inside. Catches a lower bound taken over all entries (a scalar below the hull in every
        coordinate but one), an upper bound replaced by the lower one (the surrogate box collapses to a corner and the
        conformal box misses outputs) and a projection that leaves the hull."""
        surrogate = BatchedClippingBlockSurrogate(batch_size=1000)
        surrogate.fit(self.TRAINING)
        lb, ub = surrogate.get_bounds()
        assert lb.shape == (3,) and ub.shape == (3,)
        np.testing.assert_array_equal(lb, self.LB)
        np.testing.assert_array_equal(ub, self.UB)
        assert np.all(self.TRAINING >= lb) and np.all(self.TRAINING <= ub)
        projections = surrogate.predict(self.POINTS)
        assert np.all(projections >= lb - _PROJ_TOL) and np.all(projections <= ub + _PROJ_TOL), (
            f"projections {projections} leave the bounds [{lb}, {ub}]")

    def test_naive_bounds_are_the_mean_and_hold_every_prediction(self):
        """get_bounds of the naive surrogate is (mean, mean) of the training outputs, computed by hand, and every
        prediction is that point. Catches a bound or a prediction that is not the training mean."""
        surrogate = NaiveSurrogate()
        surrogate.fit(self.TRAINING)
        mean = np.array([0.125, 3.0, -0.5625])    # column sums 0.5, 12, -2.25 divided by 4, exact in binary
        lb, ub = surrogate.get_bounds()
        np.testing.assert_array_equal(lb, mean)
        np.testing.assert_array_equal(ub, mean)
        np.testing.assert_array_equal(surrogate.predict(self.POINTS), np.tile(mean, (len(self.POINTS), 1)))


class TestNaiveSurrogateValues:
    """The centre on data where a wrong statistic changes it, and the refusals of the naive surrogate."""

    def test_centre_is_the_mean_of_skewed_outputs_after_a_refit(self):
        """After a first fit on data of another width, a refit on skewed outputs sets the centre to their mean, which
        differs from their median in every coordinate (mean (2, 3, 0.5), median (0.5, 1, -1), by hand), and predict
        and get_bounds follow it. Catches a centre computed as the median or another statistic, and state kept from the
        first fit."""
        surrogate = NaiveSurrogate()
        surrogate.fit(np.array([[100.0, -100.0], [50.0, 25.0]]))
        training = np.array([[0.0, 10.0, -1.0], [0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [7.0, 2.0, 5.0]])
        surrogate.fit(training)
        mean = np.array([2.0, 3.0, 0.5])
        np.testing.assert_array_equal(surrogate.center, mean)
        assert surrogate.n_dim == 3
        np.testing.assert_array_equal(surrogate.predict(np.zeros((2, 3))), np.tile(mean, (2, 1)))
        lb, ub = surrogate.get_bounds()
        np.testing.assert_array_equal(lb, mean)
        np.testing.assert_array_equal(ub, mean)

    @known_bug(13, "NaiveSurrogate.predict does not check the width of the outputs against the fitted dimension: it "
                   "answers rows of the fitted width for outputs of another width (and compute_errors broadcasts a "
                   "width-1 array into wrong errors), where ClippingBlockSurrogate refuses with ValueError",
               raises=DID_NOT_RAISE)
    def test_predict_refuses_outputs_of_another_width(self):
        """predict of outputs whose width is not the fitted one must refuse with ValueError, as the clipping block does
        (`TestClippingBlockSurrogatePredict::test_predict_raises_if_wrong_dimension` is the passing neighbour). Catches a
        surrogate that answers for outputs it cannot have been fitted to."""
        surrogate = NaiveSurrogate()
        surrogate.fit(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))
        with pytest.raises(ValueError):
            surrogate.predict(np.zeros((4, 2)))

    @pytest.mark.parametrize("kind", [
        pytest.param("naive", marks=known_bug(
            13, "NaiveSurrogate.fit on an empty training set (0 rows) answers a NaN centre instead of refusing",
            raises=DID_NOT_RAISE)),
        "clipping_block",
    ])
    def test_fit_refuses_an_empty_training_set(self, kind):
        """fit on zero training outputs must refuse with ValueError: the mean or the hull of nothing is undefined, and
        a NaN centre turns every later box into NaN. The clipping block refuses today and is the passing neighbour.
        NumPy's warning about an empty mean is silenced here so that the refusal itself is what is checked."""
        surrogate = NaiveSurrogate() if kind == "naive" else ClippingBlockSurrogate(n_workers=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(ValueError):
                surrogate.fit(np.zeros((0, 3)))


class TestSurrogateBaseContract:
    """The abstract base class on its own: the interface it enforces and compute_errors."""

    @staticmethod
    def _subclass_without(missing):
        methods = {
            "fit": lambda self, training_outputs: None,
            "predict": lambda self, outputs: np.zeros_like(outputs),
            "get_bounds": lambda self: (np.zeros(1), np.zeros(1)),
        }
        del methods[missing]
        return type(f"SurrogateWithout_{missing}", (Surrogate,), methods)

    @pytest.mark.parametrize("missing", ["fit", "predict", "get_bounds"])
    def test_a_subclass_without_an_abstract_method_cannot_be_built(self, missing):
        """A subclass that does not define one of fit, predict and get_bounds cannot be instantiated (TypeError), while
        one that defines all three can. Catches a method of the interface that is no longer abstract, so that a
        surrogate without get_bounds is built and answers None where conformal_reach unpacks two bounds."""
        complete = type("CompleteSurrogate", (Surrogate,), {
            "fit": lambda self, training_outputs: None,
            "predict": lambda self, outputs: np.zeros_like(outputs),
            "get_bounds": lambda self: (np.zeros(1), np.zeros(1)),
        })
        complete()
        with pytest.raises(TypeError):
            self._subclass_without(missing)()

    def test_compute_errors_takes_the_given_projections_and_otherwise_predict(self):
        """compute_errors(outputs, projections) is outputs - projections for projections that differ from what predict
        would return, without calling predict; compute_errors(outputs) is outputs - predict(outputs). Expected values by
        hand. Catches errors of the wrong sign and given projections that are ignored."""
        calls = []

        class FixedPrediction(Surrogate):
            def fit(self, training_outputs):
                pass

            def predict(self, outputs):
                calls.append(outputs.shape)
                return np.full_like(outputs, 0.5)

            def get_bounds(self):
                return np.zeros(2), np.zeros(2)

        surrogate = FixedPrediction()
        outputs = np.array([[1.0, -2.0], [3.5, 0.0]])
        given = np.array([[0.25, 1.0], [-4.0, 2.0]])
        np.testing.assert_array_equal(surrogate.compute_errors(outputs, given),
                                      np.array([[0.75, -3.0], [7.5, -2.0]]))
        assert calls == [], "compute_errors called predict although projections were given"
        np.testing.assert_array_equal(surrogate.compute_errors(outputs), np.array([[0.5, -2.5], [3.0, -0.5]]))
        assert calls == [(2, 2)]


class TestSurrogateStateBetweenCalls:
    """Promise point (3) and the same object used twice: a surrogate keeps no state from the arrays it is given after
    fit, changes none of them, and hands out bounds the caller may change."""

    @pytest.mark.parametrize("route", ["serial", "parallel", "batched-partial-last-batch"])
    def test_clipping_block_keeps_its_own_copy_and_leaves_the_callers_arrays(self, route):
        """predict changes neither the outputs it is given nor the training array; changing the training array, the
        outputs or the returned bounds after the calls changes no later projection or bound. Catches projections
        written into the caller's outputs (through conformal_reach the calibration errors then become zero and the
        inflation collapses), vertices that alias the caller's array, and bounds that alias the surrogate's own."""
        training = _TRIANGLE.copy()
        surrogate = _ROUTES[route]()
        surrogate.fit(training)
        outputs = np.array([p for p, _ in _TRIANGLE_CASES])
        outputs_before = outputs.copy()
        first = surrogate.predict(outputs)
        np.testing.assert_array_equal(outputs, outputs_before)
        np.testing.assert_array_equal(training, _TRIANGLE)
        lb, ub = surrogate.get_bounds()
        lb_before, ub_before = lb.copy(), ub.copy()
        training += 100.0
        outputs -= 7.0
        lb -= 5.0
        ub += 5.0
        lb2, ub2 = surrogate.get_bounds()
        np.testing.assert_array_equal(lb2, lb_before)
        np.testing.assert_array_equal(ub2, ub_before)
        np.testing.assert_array_equal(surrogate.predict(outputs_before), first)

    def test_naive_get_bounds_returns_arrays_the_caller_may_change(self):
        """Shifting the returned lower bound in place leaves the upper bound, the next get_bounds and the predictions
        unchanged. Catches bounds that alias the centre."""
        surrogate = NaiveSurrogate()
        surrogate.fit(np.array([[1.0, 2.0], [3.0, 6.0]]))
        lb, ub = surrogate.get_bounds()
        lb -= 5.0
        np.testing.assert_array_equal(ub, [2.0, 4.0])
        lb2, ub2 = surrogate.get_bounds()
        np.testing.assert_array_equal(lb2, [2.0, 4.0])
        np.testing.assert_array_equal(ub2, [2.0, 4.0])
        np.testing.assert_array_equal(surrogate.predict(np.zeros((1, 2))), [[2.0, 4.0]])


class TestClippingBlockSingleRowsAndOptions:
    """One-dimensional arrays, the constructor options, and option values the surrogate must refuse."""

    @pytest.mark.parametrize("route", ["serial", "parallel", "batched-partial-last-batch"])
    def test_one_dimensional_output_is_one_row(self, route):
        """predict of one output given as a 1-D array returns one row, the unique nearest hull point (1, 0.5) of
        (3, 2.5), derived by hand in _TRIANGLE_CASES. Catches a 1-D output read as a column of rows of width 1 (then
        refused for its width) and a batched predict that does not reshape a 1-D output."""
        surrogate = _ROUTES[route]()
        surrogate.fit(_TRIANGLE)
        projection = surrogate.predict(np.array([3.0, 2.5]))
        assert projection.shape == (1, 2)
        assert np.max(np.abs(projection[0] - np.array([1.0, 0.5]))) <= _PROJ_TOL

    def test_one_dimensional_training_output_is_one_sample(self):
        """fit on one training output given as a 1-D array of length 3 makes a hull of that one point: the bounds are
        the point, and every output projects to it. Catches a 1-D training output read as three one-dimensional
        samples."""
        point = np.array([1.0, -2.0, 3.0])
        surrogate = ClippingBlockSurrogate(n_workers=1)
        surrogate.fit(point)
        assert surrogate.vertices.shape == (1, 3)
        lb, ub = surrogate.get_bounds()
        np.testing.assert_array_equal(lb, point)
        np.testing.assert_array_equal(ub, point)
        np.testing.assert_allclose(surrogate.predict(np.array([[5.0, 5.0, 5.0]])), [point], rtol=0, atol=_PROJ_TOL)

    def test_constructor_options_are_kept(self):
        """The options given to the constructors are the ones the surrogates hold. Catches a batched surrogate that does
        not forward n_workers and verbose, so that n_workers=1 is replaced by a pool of 4 processes."""
        plain = ClippingBlockSurrogate(n_workers=1, verbose=True)
        assert (plain.n_workers, plain.verbose) == (1, True)
        batched = BatchedClippingBlockSurrogate(batch_size=3, n_workers=1, verbose=True)
        assert (batched.batch_size, batched.n_workers, batched.verbose) == (3, 1, True)

    @pytest.mark.parametrize("batch_size", [
        pytest.param(0, marks=known_bug(
            13, "BatchedClippingBlockSurrogate does not check batch_size: 0 crashes predict with ZeroDivisionError "
                "instead of a refusal", raises=ZeroDivisionError)),
        pytest.param(-1, marks=known_bug(
            13, "BatchedClippingBlockSurrogate does not check batch_size: -1 makes predict return all-zero "
                "projections, which are not points of the hull, instead of refusing", raises=DID_NOT_RAISE)),
        1,
    ])
    def test_batch_size_must_be_positive(self, batch_size):
        """A batch_size that is not positive must be refused with ValueError, at construction or at predict; a
        positive one answers the hand-derived projections ((1, 0.5) for (3, 2.5), the vertex (3, -0.5) for (5, -1.5)).
        batch_size=1 is the passing neighbour."""
        points = np.array([[3.0, 2.5], [5.0, -1.5]])
        expected = np.array([[1.0, 0.5], [3.0, -0.5]])
        if batch_size >= 1:
            surrogate = BatchedClippingBlockSurrogate(batch_size=batch_size, n_workers=1)
            surrogate.fit(_TRIANGLE)
            assert np.max(np.abs(surrogate.predict(points) - expected)) <= _PROJ_TOL
            return
        with pytest.raises(ValueError):
            surrogate = BatchedClippingBlockSurrogate(batch_size=batch_size, n_workers=1)
            surrogate.fit(_TRIANGLE)
            surrogate.predict(points)

    @pytest.mark.parametrize("dtype", [
        np.float64,
        pytest.param(np.int64, marks=known_bug(
            13, "ClippingBlockSurrogate.predict allocates its result with the dtype of the outputs (np.zeros_like), so "
                "the projections of integer outputs are truncated to integers: (4, 4) onto the hull of (0, 0), (5, 0), "
                "(0, 5) gives (2, 2), at L-infinity distance 2, instead of (2.5, 2.5) at distance 1.5")),
    ])
    def test_projection_of_integer_data_is_the_nearest_point(self, dtype):
        """Training outputs and outputs held in an integer array are projected as the same values in floating point:
        the nearest point of the triangle (0, 0), (5, 0), (0, 5) to (4, 4) is (2.5, 2.5) (the ball first touches the
        slanted edge with its lower-left corner, d = (4 + 4 - 5) / 2). The float64 case is the passing neighbour."""
        surrogate = ClippingBlockSurrogate(n_workers=1)
        surrogate.fit(np.array([[0, 0], [5, 0], [0, 5]], dtype=dtype))
        projection = surrogate.predict(np.array([[4, 4]], dtype=dtype))
        assert np.max(np.abs(projection - np.array([[2.5, 2.5]]))) <= _PROJ_TOL, f"projection {projection}"


class _InProcessPool:
    """A stand-in for ProcessPoolExecutor: it records every pool that predict opens (its max_workers and how many calls
    were submitted to it) and runs each submitted call at once in this process, so the projections it returns are
    those of the function the real pool would run."""

    opened = []

    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.submitted = 0
        type(self).opened.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        self.submitted += 1
        future = concurrent.futures.Future()
        future.set_result(fn(*args))
        return future


# name -> (a fresh surrogate, the number of triangle points projected, the pools predict must open as
# (max_workers, calls submitted)). The batched surrogate with batch_size 4 sees batches of 4, 4 and 1 rows; the last
# batch has one row, which takes the serial branch.
_WORKER_CASES = {
    "plain-one-worker": (lambda: ClippingBlockSurrogate(n_workers=1), 9, []),
    "plain-three-workers": (lambda: ClippingBlockSurrogate(n_workers=3), 9, [(3, 9)]),
    "plain-three-workers-one-row": (lambda: ClippingBlockSurrogate(n_workers=3), 1, []),
    "batched-one-worker": (lambda: BatchedClippingBlockSurrogate(batch_size=4, n_workers=1), 9, []),
    "batched-two-workers": (lambda: BatchedClippingBlockSurrogate(batch_size=4, n_workers=2), 9, [(2, 4), (2, 4)]),
}


class TestClippingBlockWorkerOption:
    """The option n_workers by what predict does with it, not by the attribute it sets."""

    @pytest.mark.parametrize("case", sorted(_WORKER_CASES))
    def test_n_workers_decides_whether_and_how_wide_a_process_pool_is_opened(self, case, monkeypatch):
        """n_workers = 1 opens no process pool, n_workers = k > 1 opens pools of k workers and submits one call per row,
        and a single row is projected in this process; the batched surrogate forwards n_workers to every batch. The
        projections are at the hand-derived least distances of _TRIANGLE_CASES either way. Catches a predict that
        ignores n_workers (a pool for n_workers = 1, or always a pool of 4), a batched surrogate that does not forward
        n_workers, and a pool opened for a single row."""
        make, rows, expected = _WORKER_CASES[case]
        monkeypatch.setattr(clipping_block_module, "ProcessPoolExecutor", _InProcessPool)
        monkeypatch.setattr(_InProcessPool, "opened", [])
        surrogate = make()
        surrogate.fit(_TRIANGLE)
        points = np.array([p for p, _ in _TRIANGLE_CASES])[:rows]
        projections = surrogate.predict(points)
        assert [(pool.max_workers, pool.submitted) for pool in _InProcessPool.opened] == expected
        for y, p, (_, d) in zip(points, projections, _TRIANGLE_CASES):
            assert _in_triangle(p) and abs(np.max(np.abs(y - p)) - d) <= _PROJ_TOL, f"projection {p} of {y}"


class TestClippingBlockSolverFailureFallback:
    """The branch taken when the LP fails. The LP is always feasible and bounded, so no input found reaches it; a
    stubbed solver that reports a failure stands in for a numerical failure of HiGHS."""

    @pytest.mark.parametrize("route", ["serial", "standalone"])
    def test_a_failed_lp_returns_the_nearest_training_output(self, route, monkeypatch):
        """When linprog reports no success, the projection is the training output nearest in L-infinity: for (3, 2.5)
        the vertices of the triangle are at distances 4, 3 and 4, so the answer is (3, -0.5), a point of the hull inside
        get_bounds. Catches a fallback that returns the output itself (outside the hull, with a zero error that
        shrinks the conformal box) and one that returns another vertex. The serial route goes through predict; the
        standalone route calls the function the process pool runs, in this process, so that the stub applies."""
        calls = []

        def failing_linprog(*args, **kwargs):
            calls.append(1)
            return SimpleNamespace(success=False, status=4, x=None, message="stubbed numerical failure")

        monkeypatch.setattr(clipping_block_module, "linprog", failing_linprog)
        y = np.array([3.0, 2.5])
        if route == "serial":
            surrogate = ClippingBlockSurrogate(n_workers=1)
            surrogate.fit(_TRIANGLE)
            projection = surrogate.predict(y[None, :])[0]
            lb, ub = surrogate.get_bounds()
        else:
            projection = _project_single_standalone(y, _TRIANGLE)
            lb, ub = _TRIANGLE.min(axis=0), _TRIANGLE.max(axis=0)
        if not calls:
            pytest.fail("the stubbed linprog was not called; the test no longer reaches the fallback")
        np.testing.assert_array_equal(projection, [3.0, -0.5])
        assert np.all(projection >= lb) and np.all(projection <= ub)


class _RecordingModel:
    """A numpy model for conformal_reach that keeps every batch of outputs it returns, so that the test knows the
    training and the calibration outputs without relying on the order in which conformal_reach draws its samples."""

    def __init__(self):
        self.calls = []

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        # Two inputs to three outputs, nonlinear, so that the hull of the training outputs is not their bounding box;
        # outputs have both signs and magnitudes above and below one.
        y = np.stack([x[:, 0] + 2.0 * x[:, 1], 3.0 * x[:, 0] * x[:, 1] - 4.0, -x[:, 0] + 0.5 * x[:, 1] ** 2], axis=1)
        self.calls.append(y.copy())
        return y


class TestConformalReachRouteSoundness:
    """Promise point (1) through the public entry: conformal_reach adds the inflation to the surrogate's bounds. With
    ell = m the threshold is the largest calibration score, so the inflation in every coordinate is at least the
    calibration error |y - g(y)| there, and every calibration output y lies in the returned box exactly when every
    prediction g(y) lies in the surrogate's bounds. This needs no statistics."""

    T, M = 40, 60

    def _run(self, surrogate):
        model = _RecordingModel()
        result = conformal_reach(model, Box(np.array([-2.0, 0.5]), np.array([3.0, 4.0])), surrogate=surrogate,
                                 m=self.M, ell=self.M, training_samples=self.T, batch_size=1000, epsilon=0.1, seed=3)
        if [c.shape[0] for c in model.calls] != [self.T, self.M]:
            pytest.fail(f"expected one training call of {self.T} rows and one calibration call of {self.M} rows, got "
                        f"{[c.shape for c in model.calls]}")
        return np.asarray(result.lb, dtype=float).ravel(), np.asarray(result.ub, dtype=float).ravel(), model.calls

    def test_clipping_block_box_holds_the_hull_box_and_every_calibration_output(self):
        """The box of conformal_reach(surrogate='clipping_block', ell=m) holds the bounding box of the training outputs
        (zero slack: the inflation is subtracted from the per-coordinate minimum and added to the maximum) and every
        calibration output, with a slack of HiGHS's primal feasibility tolerance 1e-7 times the sum of the absolute
        training outputs in that coordinate (how far a projection built from weights within that tolerance can leave
        the hull) plus a float64 pad. Catches bounds that do not hold the surrogate's predictions (an upper bound
        replaced by the lower one) and projections written into the calibration outputs (the errors become zero)."""
        lb, ub, (training, calibration) = self._run("clipping_block")
        assert np.all(lb <= training.min(axis=0)) and np.all(ub >= training.max(axis=0)), (
            f"box [{lb}, {ub}] does not hold the training outputs' bounding box")
        slack = 1e-7 * np.abs(training).sum(axis=0) + 1e-12 * (1.0 + np.abs(calibration))
        escape = np.maximum(lb - calibration, calibration - ub) - slack
        assert np.all(escape <= 0.0), f"{int(np.sum(np.any(escape > 0, axis=1)))} calibration outputs outside the box"

    def test_naive_box_is_centred_on_the_training_mean_and_holds_every_calibration_output(self):
        """The box of conformal_reach(surrogate='naive', ell=m) is centred on the mean of the training outputs and holds
        every calibration output, with a float64 pad of 1e-12 relative to the magnitudes (the prediction is the mean
        itself, so no solver tolerance enters). Catches a centre that is not the training mean."""
        lb, ub, (training, calibration) = self._run("naive")
        mean = training.mean(axis=0)
        scale = 1.0 + np.abs(mean) + (ub - lb)
        assert np.all(np.abs((lb + ub) / 2.0 - mean) <= 1e-12 * scale), f"box centre {(lb + ub) / 2} != mean {mean}"
        slack = 1e-12 * (1.0 + np.abs(calibration) + scale)
        escape = np.maximum(lb - calibration, calibration - ub) - slack
        assert np.all(escape <= 0.0), f"{int(np.sum(np.any(escape > 0, axis=1)))} calibration outputs outside the box"
