"""Unit tests for ProbabilisticSet."""

import pytest
import torch
import numpy as np


class TestProbabilisticSetContains:
    """Tests for membership queries."""

    def test_center_is_inside(self):
        """Center point should be inside the set."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        y = torch.tensor([[0.0, 0.0]])
        assert pset.contains(y).item() is True

    def test_far_point_is_outside(self):
        """Point far from center should be outside."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        y = torch.tensor([[10.0, 10.0]])
        assert pset.contains(y).item() is False

    def test_batch_contains(self):
        """Should handle batches, returning (batch,) boolean tensor."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        y = torch.tensor([[0.0, 0.0], [10.0, 10.0], [0.5, 0.5]])
        result = pset.contains(y)
        assert result.shape == (3,)
        assert result[0].item() is True
        assert result[1].item() is False
        assert result[2].item() is True


class TestProbabilisticSetGuarantee:
    """Tests for guarantee metadata."""

    def test_get_guarantee(self):
        """Should return (coverage, confidence) tuple."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=8000, ell=7999, epsilon=0.001, dim=2
        )

        coverage, confidence = pset.get_guarantee()
        assert coverage == pytest.approx(0.999)
        assert 0.99 < confidence < 1.0


class TestProbabilisticSetVolume:
    """Tests for MC volume estimation."""

    def test_ball_volume_2d(self):
        """Volume of L2 ball with radius r in 2D = pi*r^2."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        radius = 2.0
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=radius,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        bbox = (torch.tensor([-3.0, -3.0]), torch.tensor([3.0, 3.0]))
        volume, std_err = pset.estimate_volume(n_samples=500_000,
                                               bounding_box=bbox)
        expected = np.pi * radius**2
        # MC estimate should be within ~5% of true volume
        assert volume == pytest.approx(expected, rel=0.05)

    def test_hyperrect_volume_2d(self):
        """Volume of hyperrect should match closed-form."""
        from n2v.probabilistic.flow.scores import HyperrectScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        scales = torch.tensor([1.0, 1.0])
        score_fn = HyperrectScore(center, scales)
        q = 2.0
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=q,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        bbox = (torch.tensor([-5.0, -5.0]), torch.tensor([5.0, 5.0]))
        volume, std_err = pset.estimate_volume(n_samples=500_000,
                                               bounding_box=bbox)
        expected = score_fn.sublevel_set_volume(torch.tensor(q))
        assert volume == pytest.approx(expected, rel=0.05)


class TestProbabilisticSetBoundary2D:
    """Tests for 2D boundary extraction."""

    def test_returns_arrays(self):
        """boundary_2d should return list of (N, 2) numpy arrays."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=100, ell=99, epsilon=0.01, dim=2
        )

        bounds = (torch.tensor([-2.0, -2.0]), torch.tensor([2.0, 2.0]))
        contours = pset.boundary_2d(resolution=50, bounds=bounds)
        assert isinstance(contours, list)
        assert len(contours) > 0
        for path in contours:
            assert isinstance(path, np.ndarray)
            assert path.ndim == 2
            assert path.shape[1] == 2

    def test_raises_for_non_2d(self):
        """boundary_2d should raise ValueError for dim != 2."""
        from n2v.probabilistic.flow.scores import BallScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        center = torch.tensor([0.0, 0.0, 0.0])
        score_fn = BallScore(center)
        pset = ProbabilisticSet(
            score_fn=score_fn, threshold=1.0,
            m=100, ell=99, epsilon=0.01, dim=3
        )

        with pytest.raises(ValueError, match="dim=2"):
            pset.boundary_2d()
