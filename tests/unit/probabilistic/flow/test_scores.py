"""Unit tests for nonconformity score functions."""

import pytest
import torch
import numpy as np


class TestHyperrectScore:
    """Tests for HyperrectScore."""

    def test_zero_at_center(self):
        """Score at the center should be 0."""
        from n2v.probabilistic.flow.scores import HyperrectScore

        center = torch.tensor([1.0, 2.0])
        scales = torch.tensor([1.0, 1.0])
        score_fn = HyperrectScore(center, scales)

        y = center.unsqueeze(0)  # (1, 2)
        assert score_fn(y).item() == pytest.approx(0.0)

    def test_max_normalized_deviation(self):
        """Score should be max of |y_k - c_k| / tau_k."""
        from n2v.probabilistic.flow.scores import HyperrectScore

        center = torch.tensor([0.0, 0.0])
        scales = torch.tensor([1.0, 2.0])
        score_fn = HyperrectScore(center, scales)

        y = torch.tensor([[3.0, 2.0]])  # deviations: 3/1=3, 2/2=1
        assert score_fn(y).item() == pytest.approx(3.0)

    def test_batch_computation(self):
        """Should handle batches correctly."""
        from n2v.probabilistic.flow.scores import HyperrectScore

        center = torch.tensor([0.0, 0.0])
        scales = torch.tensor([1.0, 1.0])
        score_fn = HyperrectScore(center, scales)

        y = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
        scores = score_fn(y)
        assert scores.shape == (3,)
        assert scores[0].item() == pytest.approx(1.0)
        assert scores[1].item() == pytest.approx(2.0)
        assert scores[2].item() == pytest.approx(3.0)

    def test_sublevel_set_volume(self):
        """Volume of hyperrect {y : score(y) <= q} = prod(2*q*tau_k)."""
        from n2v.probabilistic.flow.scores import HyperrectScore

        center = torch.tensor([0.0, 0.0])
        scales = torch.tensor([2.0, 3.0])
        score_fn = HyperrectScore(center, scales)

        # Volume = 2*q*2 * 2*q*3 = 4q * 6q = 24q^2
        q = torch.tensor(1.5)
        expected = (2 * 1.5 * 2) * (2 * 1.5 * 3)
        assert score_fn.sublevel_set_volume(q) == pytest.approx(expected)


class TestEllipsoidScore:
    """Tests for EllipsoidScore."""

    def test_zero_at_center(self):
        """Score at the center should be 0."""
        from n2v.probabilistic.flow.scores import EllipsoidScore

        center = torch.tensor([1.0, 2.0])
        cov_inv = torch.eye(2)
        score_fn = EllipsoidScore(center, cov_inv)

        y = center.unsqueeze(0)
        assert score_fn(y).item() == pytest.approx(0.0)

    def test_identity_covariance_equals_l2(self):
        """With identity covariance, Mahalanobis = L2 distance."""
        from n2v.probabilistic.flow.scores import EllipsoidScore

        center = torch.tensor([0.0, 0.0])
        cov_inv = torch.eye(2)
        score_fn = EllipsoidScore(center, cov_inv)

        y = torch.tensor([[3.0, 4.0]])
        assert score_fn(y).item() == pytest.approx(5.0)

    def test_batch_computation(self):
        """Should handle batches correctly."""
        from n2v.probabilistic.flow.scores import EllipsoidScore

        center = torch.tensor([0.0, 0.0])
        cov_inv = torch.eye(2)
        score_fn = EllipsoidScore(center, cov_inv)

        y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        scores = score_fn(y)
        assert scores.shape == (2,)
        assert scores[0].item() == pytest.approx(1.0)
        assert scores[1].item() == pytest.approx(1.0)


class TestBallScore:
    """Tests for BallScore."""

    def test_zero_at_center(self):
        """Score at center should be 0."""
        from n2v.probabilistic.flow.scores import BallScore

        center = torch.tensor([1.0, 2.0])
        score_fn = BallScore(center)

        y = center.unsqueeze(0)
        assert score_fn(y).item() == pytest.approx(0.0)

    def test_l2_distance(self):
        """Score should be L2 norm from center."""
        from n2v.probabilistic.flow.scores import BallScore

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)

        y = torch.tensor([[3.0, 4.0]])
        assert score_fn(y).item() == pytest.approx(5.0)


class TestFlowScore:
    """Tests for FlowScore."""

    def test_output_shape(self):
        """Score should return (batch,) tensor."""
        from n2v.probabilistic.flow.scores import FlowScore

        # Use a dummy flow model that just returns its input
        class DummyFlow:
            def forward(self, y, t=1.0):
                return y

        score_fn = FlowScore(DummyFlow(), t=1.0)
        y = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        scores = score_fn(y)
        assert scores.shape == (2,)

    def test_with_identity_flow(self):
        """With identity flow, FlowScore = L2 norm."""
        from n2v.probabilistic.flow.scores import FlowScore

        class IdentityFlow:
            def forward(self, y, t=1.0):
                return y

        score_fn = FlowScore(IdentityFlow(), t=1.0)
        y = torch.tensor([[3.0, 4.0]])
        assert score_fn(y).item() == pytest.approx(5.0)

    def test_set_t(self):
        """set_t should update the flow time parameter."""
        from n2v.probabilistic.flow.scores import FlowScore

        class DummyFlow:
            def forward(self, y, t=1.0):
                return y * t  # scale by t

        score_fn = FlowScore(DummyFlow(), t=1.0)
        y = torch.tensor([[3.0, 4.0]])

        score_at_1 = score_fn(y).item()
        score_fn.set_t(0.5)
        score_at_05 = score_fn(y).item()

        assert score_at_1 == pytest.approx(5.0)
        assert score_at_05 == pytest.approx(2.5)
