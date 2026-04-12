"""Unit tests for flow matching training."""

import pytest
import torch
import numpy as np


class TestOTCoupling:
    """Tests for Hungarian OT coupling."""

    def test_output_shapes(self):
        """OT coupling should return tensors of same shape."""
        from n2v.probabilistic.flow.train import ot_coupling

        x0 = torch.randn(16, 2)
        x1 = torch.randn(16, 2)
        x0_c, x1_c = ot_coupling(x0, x1)
        assert x0_c.shape == (16, 2)
        assert x1_c.shape == (16, 2)

    def test_is_permutation(self):
        """OT coupling should be a permutation of the input rows."""
        from n2v.probabilistic.flow.train import ot_coupling

        x0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        x1 = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        x0_c, x1_c = ot_coupling(x0, x1)

        x0_sorted = x0[x0[:, 0].sort().indices]
        x0c_sorted = x0_c[x0_c[:, 0].sort().indices]
        torch.testing.assert_close(x0_sorted, x0c_sorted)


class TestSinkhornCoupling:
    """Tests for Sinkhorn OT coupling."""

    def test_output_shapes(self):
        """Sinkhorn coupling should return tensors of same shape."""
        from n2v.probabilistic.flow.train import sinkhorn_coupling

        x0 = torch.randn(16, 2)
        x1 = torch.randn(16, 2)
        x0_c, x1_c = sinkhorn_coupling(x0, x1)
        assert x0_c.shape == (16, 2)
        assert x1_c.shape == (16, 2)

    def test_is_permutation(self):
        """Sinkhorn coupling should be a permutation of the input rows."""
        from n2v.probabilistic.flow.train import sinkhorn_coupling

        x0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        x1 = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        x0_c, x1_c = sinkhorn_coupling(x0, x1)

        x0_sorted = x0[x0[:, 0].sort().indices]
        x0c_sorted = x0_c[x0_c[:, 0].sort().indices]
        torch.testing.assert_close(x0_sorted, x0c_sorted)

    def test_gpu_compatible(self):
        """Sinkhorn should work on GPU tensors if available."""
        from n2v.probabilistic.flow.train import sinkhorn_coupling

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x0 = torch.randn(16, 2, device='cuda')
        x1 = torch.randn(16, 2, device='cuda')
        x0_c, x1_c = sinkhorn_coupling(x0, x1)
        assert x0_c.device.type == 'cuda'
        assert x1_c.device.type == 'cuda'

    def test_agrees_with_hungarian_on_easy_case(self):
        """On well-separated clusters, Sinkhorn should match Hungarian."""
        from n2v.probabilistic.flow.train import ot_coupling, sinkhorn_coupling

        x0 = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        x1 = torch.tensor([[0.1, 0.1], [10.1, 0.1], [0.1, 10.1]])

        _, x1_h = ot_coupling(x0, x1)
        _, x1_s = sinkhorn_coupling(x0, x1)

        torch.testing.assert_close(x1_h, x1_s)


class TestTrainFlow:
    """Tests for the training loop."""

    def test_returns_model_and_losses(self):
        """train_flow should return (model, losses)."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        data = torch.randn(100, 2)
        model, losses = train_flow(
            vf, data, n_epochs=5, batch_size=32, lr=1e-3,
            coupling='none',
        )
        assert model is vf
        assert isinstance(losses, list)
        assert len(losses) == 5

    def test_loss_decreases(self):
        """Loss should generally decrease over training."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow

        torch.manual_seed(42)
        vf = VelocityField(dim=2, hidden=64, n_layers=3)
        data = torch.randn(500, 2) * 0.5 + 2.0
        _, losses = train_flow(
            vf, data, n_epochs=50, batch_size=64, lr=1e-3,
            coupling='none',
        )
        assert np.mean(losses[:5]) > np.mean(losses[-5:])

    def test_with_hungarian_coupling(self):
        """Training with Hungarian coupling should work."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        data = torch.randn(64, 2)
        model, losses = train_flow(
            vf, data, n_epochs=3, batch_size=32, lr=1e-3,
            coupling='hungarian',
        )
        assert len(losses) == 3

    def test_with_sinkhorn_coupling(self):
        """Training with Sinkhorn coupling should work."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        data = torch.randn(64, 2)
        model, losses = train_flow(
            vf, data, n_epochs=3, batch_size=32, lr=1e-3,
            coupling='sinkhorn',
        )
        assert len(losses) == 3

    def test_invalid_coupling_raises(self):
        """Invalid coupling string should raise ValueError."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        data = torch.randn(64, 2)
        with pytest.raises(ValueError, match="coupling"):
            train_flow(vf, data, n_epochs=1, coupling='invalid')
