"""Phase 1.4: 2D boundary_2d contour accuracy on closed-form shapes."""

import math
import numpy as np
import torch

from n2v.probabilistic.flow.scores import BallScore, HyperrectScore, EllipsoidScore
from n2v.probabilistic.flow.sets import ProbabilisticSet


def _mk(score, dim=2):
    return ProbabilisticSet(
        score_fn=score, threshold=1.0, m=100, ell=99, epsilon=0.01, dim=dim,
    )


def _max_boundary_residual(paths, residual_fn):
    """Return max |residual_fn(x, y)| over all points on all contour paths."""
    worst = 0.0
    for path in paths:
        if path.shape[0] == 0:
            continue
        r = np.abs(residual_fn(path[:, 0], path[:, 1]))
        worst = max(worst, float(r.max()))
    return worst


class TestBoundary2D:
    def test_unit_circle(self):
        s = _mk(BallScore(torch.zeros(2)))
        bounds = (-1.3 * torch.ones(2), 1.3 * torch.ones(2))
        paths = s.boundary_2d(resolution=300, bounds=bounds)
        worst = _max_boundary_residual(paths, lambda x, y: x ** 2 + y ** 2 - 1.0)
        assert worst < 0.02, f"unit circle boundary deviates by {worst}"

    def test_unit_square(self):
        s = _mk(HyperrectScore(torch.zeros(2), torch.ones(2)))
        bounds = (-1.3 * torch.ones(2), 1.3 * torch.ones(2))
        paths = s.boundary_2d(resolution=300, bounds=bounds)
        # boundary is max(|x|, |y|) = 1
        worst = _max_boundary_residual(
            paths, lambda x, y: np.maximum(np.abs(x), np.abs(y)) - 1.0
        )
        assert worst < 0.02

    def test_rotated_ellipse(self):
        # Ellipse with semi-axes (2, 0.5), rotated 30 degrees.
        sigmas = torch.tensor([2.0, 0.5])
        angle = math.pi / 6
        c, sn = math.cos(angle), math.sin(angle)
        R = torch.tensor([[c, -sn], [sn, c]])
        cov = R @ torch.diag(sigmas ** 2) @ R.T
        s = _mk(EllipsoidScore(torch.zeros(2), torch.linalg.inv(cov)))
        bounds = (-2.5 * torch.ones(2), 2.5 * torch.ones(2))
        paths = s.boundary_2d(resolution=400, bounds=bounds)
        # residual: (x, y) Sigma^-1 (x, y)^T - 1
        cov_inv_np = torch.linalg.inv(cov).numpy()
        def residual(x, y):
            xy = np.stack([x, y], axis=1)
            return np.einsum('ni,ij,nj->n', xy, cov_inv_np, xy) - 1.0
        worst = _max_boundary_residual(paths, residual)
        assert worst < 0.05, f"rotated ellipse residual {worst}"


# ---------------------------------------------------------------------------
# Completeness of the contour
# ---------------------------------------------------------------------------
from tests.known_bugs import known_bug  # noqa: E402


def _distance_to_circle(paths, centre, radius):
    """(largest distance of a contour vertex from the circle, largest
    distance from a point of the circle to the nearest contour vertex): the
    two directions of the Hausdorff distance. The residual checks above see
    only the first, which a contour that is clipped or missing passes."""
    if not paths:
        return math.inf, math.inf
    verts = np.concatenate(paths)
    centre = np.asarray(centre, dtype=np.float64)
    off = float(np.abs(np.linalg.norm(verts - centre, axis=1) - radius).max())
    angles = np.linspace(0.0, 2 * math.pi, 720, endpoint=False)
    circle = centre + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    gaps = np.linalg.norm(circle[:, None, :] - verts[None, :, :], axis=2).min(axis=1)
    return off, float(gaps.max())


class TestContourCompleteness:
    def test_contour_traces_the_whole_off_centre_circle_with_bounds_that_differ_per_axis(self):
        """The unit circle at (2.5, 0) on the grid [1.25, 3.75] x
        [-1.25, 1.5] (201 points per axis, spacing h at most 0.01375): every
        contour vertex is within 2h of the circle and every point of the
        circle within 2h of a vertex (consecutive vertices of a marching
        squares contour are at most one cell apart). Catches the grid built
        in (y, x) order (the contour of the mirrored set, here off the grid)
        and a y range read from the x entries of the bounds (the grid then
        misses the circle)."""
        centre = (2.5, 0.0)
        s = _mk(BallScore(torch.tensor(centre)))
        bounds = (torch.tensor([1.25, -1.25]), torch.tensor([3.75, 1.5]))
        paths = s.boundary_2d(resolution=201, bounds=bounds)
        tol = 2 * max(2.5, 2.75) / 200
        off, gap = _distance_to_circle(paths, centre, 1.0)
        assert off <= tol and gap <= tol, (off, gap, tol)

    def test_default_bounds_hold_a_set_around_the_origin(self):
        """Without bounds the grid is [-3q, 3q]^2; the circle of radius
        q = 1.5 at the origin is traced completely (300 points per axis,
        tolerance 2h with h = 9 / 299)."""
        s = ProbabilisticSet(score_fn=BallScore(torch.zeros(2)), threshold=1.5,
                             m=100, ell=99, epsilon=0.01, dim=2)
        paths = s.boundary_2d(resolution=300)
        tol = 2 * 9.0 / 299
        off, gap = _distance_to_circle(paths, (0.0, 0.0), 1.5)
        assert off <= tol and gap <= tol, (off, gap, tol)

    @known_bug(13, "without bounds boundary_2d evaluates the square "
                   "[-3q, 3q]^2 around the origin whatever the set's centre, "
                   "so the boundary of a set away from the origin is returned "
                   "as no contour at all")
    def test_default_bounds_hold_a_set_away_from_the_origin(self):
        """The unit circle at (10, 10), no bounds given: the returned contour
        must trace it. Today the default grid [-3, 3]^2 misses it and the
        result is an empty list."""
        s = _mk(BallScore(torch.tensor([10.0, 10.0])))
        paths = s.boundary_2d(resolution=300)
        off, gap = _distance_to_circle(paths, (10.0, 10.0), 1.0)
        assert off <= 0.1 and gap <= 0.1, (off, gap)


# ---------------------------------------------------------------------------
# State left by a call: the caller's matplotlib backend and figures
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


class TestCallerPlottingState:
    @pytest.mark.parametrize("backend", [
        pytest.param("agg", id="caller-on-agg"),
        pytest.param("svg", id="caller-on-svg", marks=known_bug(
            13, "boundary_2d calls matplotlib.use('Agg') on every call, so it "
                "switches the whole process to the Agg backend and leaves it "
                "there: a caller who plots with another backend (svg, pdf, a "
                "notebook's inline backend) loses it")),
    ])
    def test_boundary_2d_leaves_the_callers_backend_and_figures_alone(self, backend):
        """boundary_2d draws its contour on a figure of its own; the
        backend the caller chose and the figure the caller holds open are
        the same after the call (backend names compared without case, as
        matplotlib reads them). The reference is matplotlib's own state read
        before and after. Catches closing every figure instead of its own
        one (the caller's open figure is closed) and a backend switch that
        is not restored."""
        import matplotlib
        import matplotlib.pyplot as plt

        original = matplotlib.get_backend()
        plt.switch_backend(backend)
        held = plt.figure()
        try:
            before = matplotlib.get_backend().lower()
            paths = _mk(BallScore(torch.zeros(2))).boundary_2d(resolution=50)
            if not paths:
                pytest.fail("fixture: the unit circle gave no contour")
            assert plt.fignum_exists(held.number), "the caller's figure was closed"
            assert matplotlib.get_backend().lower() == before, (
                f"backend {before} became {matplotlib.get_backend()}")
        finally:
            plt.close(held)
            plt.switch_backend(original)
