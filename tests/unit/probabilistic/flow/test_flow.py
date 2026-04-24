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


class TestSinusoidalTimeEmbedding:
    """Tests for the optional sinusoidal time embedding on VelocityField."""

    def test_concat_is_default(self):
        """Default time_embed is 'concat' (pre-change behavior)."""
        from n2v.probabilistic.flow.model import VelocityField
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        assert vf.time_embed == 'concat'

    def test_sinusoidal_forward_runs(self):
        """Sinusoidal embedding runs end-to-end and returns correct shape."""
        from n2v.probabilistic.flow.model import VelocityField
        vf = VelocityField(
            dim=2, hidden=8, n_layers=3, time_embed='sinusoidal'
        )
        t = torch.rand(4)
        y = torch.randn(4, 2)
        v = vf(t, y)
        assert v.shape == (4, 2)

    def test_sinusoidal_differs_from_concat(self):
        """Sinusoidal and concat produce different outputs for same t, y."""
        from n2v.probabilistic.flow.model import VelocityField
        torch.manual_seed(0)
        vf_concat = VelocityField(
            dim=2, hidden=8, n_layers=3, time_embed='concat'
        )
        torch.manual_seed(0)
        vf_sinus = VelocityField(
            dim=2, hidden=8, n_layers=3, time_embed='sinusoidal'
        )
        t = torch.rand(4)
        y = torch.randn(4, 2)
        with torch.no_grad():
            out_concat = vf_concat(t, y)
            out_sinus = vf_sinus(t, y)
        # Different input dims -> different first-layer weights -> outputs differ
        assert not torch.allclose(out_concat, out_sinus)

    def test_invalid_time_embed_raises(self):
        """Unknown time_embed value raises ValueError."""
        from n2v.probabilistic.flow.model import VelocityField
        with pytest.raises(ValueError, match="time_embed"):
            VelocityField(dim=2, hidden=8, n_layers=3, time_embed='nonsense')


class TestResidualBlocks:
    """Tests for the optional residual-block architecture."""

    def test_default_is_non_residual(self):
        from n2v.probabilistic.flow.model import VelocityField
        vf = VelocityField(dim=2, hidden=8, n_layers=4)
        assert vf.residual is False

    def test_residual_forward_runs(self):
        from n2v.probabilistic.flow.model import VelocityField
        vf = VelocityField(dim=2, hidden=8, n_layers=4, residual=True)
        t = torch.rand(4)
        y = torch.randn(4, 2)
        v = vf(t, y)
        assert v.shape == (4, 2)

    def test_residual_differs_from_sequential(self):
        from n2v.probabilistic.flow.model import VelocityField
        torch.manual_seed(0)
        vf_seq = VelocityField(dim=2, hidden=8, n_layers=4, residual=False)
        torch.manual_seed(0)
        vf_res = VelocityField(dim=2, hidden=8, n_layers=4, residual=True)
        t = torch.rand(4)
        y = torch.randn(4, 2)
        with torch.no_grad():
            assert not torch.allclose(vf_seq(t, y), vf_res(t, y))


class TestLayerNorm:

    def test_default_no_layer_norm(self):
        from n2v.probabilistic.flow.model import VelocityField
        vf = VelocityField(dim=2, hidden=8, n_layers=4)
        assert vf.layer_norm is False

    def test_requires_residual(self):
        from n2v.probabilistic.flow.model import VelocityField
        with pytest.raises(ValueError, match="residual"):
            VelocityField(
                dim=2, hidden=8, n_layers=4,
                residual=False, layer_norm=True,
            )

    def test_layer_norm_forward_runs(self):
        from n2v.probabilistic.flow.model import VelocityField
        vf = VelocityField(
            dim=2, hidden=8, n_layers=4, residual=True, layer_norm=True
        )
        t = torch.rand(4)
        y = torch.randn(4, 2)
        assert vf(t, y).shape == (4, 2)


class TestZeroInitOutput:

    def test_default_is_false(self):
        from n2v.probabilistic.flow.model import VelocityField
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        assert vf.zero_init_output is False

    def test_zero_init_produces_zero_output(self):
        from n2v.probabilistic.flow.model import VelocityField
        vf = VelocityField(
            dim=2, hidden=8, n_layers=3, zero_init_output=True
        )
        t = torch.rand(4)
        y = torch.randn(4, 2)
        with torch.no_grad():
            out = vf(t, y)
        assert torch.allclose(out, torch.zeros_like(out))


class TestDiTLiteVelocityField:

    def test_import_exists(self):
        from n2v.probabilistic.flow.model import DiTLiteVelocityField  # noqa

    def test_forward_shape_banana(self):
        from n2v.probabilistic.flow.model import DiTLiteVelocityField
        vf = DiTLiteVelocityField(dim=2, hidden=64, n_blocks=2, n_heads=4)
        t = torch.rand(4)
        y = torch.randn(4, 2)
        assert vf(t, y).shape == (4, 2)

    def test_forward_shape_classifier(self):
        from n2v.probabilistic.flow.model import DiTLiteVelocityField
        vf = DiTLiteVelocityField(dim=3, hidden=64, n_blocks=2, n_heads=4)
        t = torch.rand(4)
        y = torch.randn(4, 3)
        assert vf(t, y).shape == (4, 3)

    def test_scalar_time_broadcasts(self):
        from n2v.probabilistic.flow.model import DiTLiteVelocityField
        vf = DiTLiteVelocityField(dim=2, hidden=64)
        t = torch.tensor(0.5)
        y = torch.randn(4, 2)
        assert vf(t, y).shape == (4, 2)


class TestDiTLiteLargeBatch:
    """Verify DiTLiteVelocityField handles batches > 65535 via chunking."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="efficient attention crash is CUDA-only",
    )
    def test_large_batch_forward_runs(self):
        """Batch of 100_000 points forwards without the efficient-attention crash."""
        from n2v.probabilistic.flow.model import DiTLiteVelocityField
        vf = DiTLiteVelocityField(
            dim=3, hidden=64, n_blocks=2, n_heads=4
        ).cuda()
        n = 100_000
        y = torch.randn(n, 3, device='cuda')
        t = torch.rand(n, device='cuda')
        with torch.no_grad():
            out = vf(t, y)
        assert out.shape == (n, 3)

    def test_chunked_output_matches_single_batch(self):
        """Chunking is numerically equivalent to a single forward for small batches."""
        from n2v.probabilistic.flow.model import DiTLiteVelocityField
        torch.manual_seed(0)
        vf = DiTLiteVelocityField(dim=3, hidden=64, n_blocks=2, n_heads=4)
        vf.eval()
        n = 256
        y = torch.randn(n, 3)
        t = torch.rand(n)
        with torch.no_grad():
            out_full = vf(t, y)
            # Compute in two halves by calling forward with slices
            out_half1 = vf(t[:128], y[:128])
            out_half2 = vf(t[128:], y[128:])
            out_concat = torch.cat([out_half1, out_half2], dim=0)
        torch.testing.assert_close(out_full, out_concat, atol=1e-5, rtol=1e-5)
