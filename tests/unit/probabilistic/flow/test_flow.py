"""Unit tests for flow matching components."""

import pytest
import torch


class TestVelocityField:
    """Tests for VelocityField network."""

    def test_output_shape(self):
        """Output should be (batch, dim)."""
        from n2v.probabilistic.flow.model import VelocityField

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        t = torch.tensor([0.5, 0.5])
        y = torch.randn(2, 2)
        v = vf(t, y)
        assert v.shape == (2, 2)

    def test_scalar_t(self):
        """Should handle scalar t by expanding to batch size."""
        from n2v.probabilistic.flow.model import VelocityField

        vf = VelocityField(dim=3, hidden=32, n_layers=3)
        t = torch.tensor(0.5)
        y = torch.randn(5, 3)
        v = vf(t, y)
        assert v.shape == (5, 3)

    def test_different_dims(self):
        """Should work with various dimensionalities."""
        from n2v.probabilistic.flow.model import VelocityField

        for dim in [2, 5, 10]:
            vf = VelocityField(dim=dim, hidden=32, n_layers=3)
            t = torch.tensor(0.5)
            y = torch.randn(4, dim)
            v = vf(t, y)
            assert v.shape == (4, dim)


class TestFlowODE:
    """Tests for FlowODE wrapper."""

    def test_forward_output_shape(self):
        """Forward should return (batch, dim) tensor."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        flow = FlowODE(vf)
        y = torch.randn(4, 2)
        z = flow.forward(y, t=1.0, n_steps=10)
        assert z.shape == (4, 2)

    def test_forward_is_deterministic(self):
        """Same input should produce same output."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        flow = FlowODE(vf)
        y = torch.randn(3, 2)
        z1 = flow.forward(y, t=0.5, n_steps=10)
        z2 = flow.forward(y, t=0.5, n_steps=10)
        torch.testing.assert_close(z1, z2)

    def test_forward_t_zero_is_identity(self):
        """At t=0, the flow should be (approximately) the identity."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        flow = FlowODE(vf)
        y = torch.randn(4, 2)
        z = flow.forward(y, t=0.0, n_steps=10)
        torch.testing.assert_close(z, y, atol=1e-4, rtol=1e-4)

    def test_forward_trajectory_output_shape(self):
        """forward_trajectory should return (batch, len(t_values)) norms."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        flow = FlowODE(vf)
        y = torch.randn(4, 2)
        t_values = [0.25, 0.5, 0.75, 1.0]
        norms = flow.forward_trajectory(y, t_values, n_steps=10)
        assert norms.shape == (4, 4)

    def test_forward_trajectory_norms_nonnegative(self):
        """All norms should be non-negative."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        flow = FlowODE(vf)
        y = torch.randn(4, 2)
        norms = flow.forward_trajectory(y, [0.5, 1.0], n_steps=10)
        assert (norms >= 0).all()


class TestVelocityFieldActivation:
    """Tests for configurable activation."""

    def test_default_is_silu(self):
        """Default activation should be SiLU."""
        from n2v.probabilistic.flow.model import VelocityField

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        activations = [m for m in vf.net if isinstance(m, torch.nn.SiLU)]
        assert len(activations) > 0

    def test_relu_activation(self):
        """Should support ReLU activation."""
        from n2v.probabilistic.flow.model import VelocityField

        vf = VelocityField(dim=2, hidden=32, n_layers=3, activation='relu')
        activations = [m for m in vf.net if isinstance(m, torch.nn.ReLU)]
        assert len(activations) > 0
        silu_activations = [m for m in vf.net if isinstance(m, torch.nn.SiLU)]
        assert len(silu_activations) == 0

    def test_silu_activation_explicit(self):
        """Explicit silu should work."""
        from n2v.probabilistic.flow.model import VelocityField

        vf = VelocityField(dim=2, hidden=32, n_layers=3, activation='silu')
        activations = [m for m in vf.net if isinstance(m, torch.nn.SiLU)]
        assert len(activations) > 0

    def test_invalid_activation_raises(self):
        """Invalid activation should raise ValueError."""
        from n2v.probabilistic.flow.model import VelocityField

        with pytest.raises(ValueError, match="activation"):
            VelocityField(dim=2, hidden=32, n_layers=3, activation='gelu')


class TestFlowODEInverse:
    """Tests for the inverse direction (latent -> data)."""

    def test_inverse_output_shape(self):
        """Inverse should return (batch, dim) tensor."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        flow = FlowODE(vf)
        z = torch.randn(4, 2)
        y = flow.inverse(z, t=1.0, n_steps=10)
        assert y.shape == (4, 2)

    def test_inverse_t_zero_is_identity(self):
        """At t=0, inverse should be identity."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        flow = FlowODE(vf)
        z = torch.randn(4, 2)
        y = flow.inverse(z, t=0.0, n_steps=10)
        torch.testing.assert_close(y, z, atol=1e-4, rtol=1e-4)

    def test_inverse_then_forward_roundtrip(self):
        """forward(inverse(z)) should approximately equal z."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE

        torch.manual_seed(0)
        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        flow = FlowODE(vf)
        z = torch.randn(4, 2) * 0.5
        y = flow.inverse(z, t=1.0, n_steps=100)
        z_back = flow.forward(y, t=1.0, n_steps=100)
        torch.testing.assert_close(z_back, z, atol=1e-3, rtol=1e-3)

    def test_inverse_is_deterministic(self):
        """Same input should give same output."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE

        vf = VelocityField(dim=2, hidden=32, n_layers=3)
        flow = FlowODE(vf)
        z = torch.randn(3, 2)
        y1 = flow.inverse(z, t=0.5, n_steps=10)
        y2 = flow.inverse(z, t=0.5, n_steps=10)
        torch.testing.assert_close(y1, y2)
