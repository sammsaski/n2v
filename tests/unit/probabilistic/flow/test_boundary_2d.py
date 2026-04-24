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
