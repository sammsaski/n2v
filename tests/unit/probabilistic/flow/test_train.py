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
        x0_c, x1_c = sinkhorn_coupling(x0, x1, reg=0.05)
        assert x0_c.shape == (16, 2)
        assert x1_c.shape == (16, 2)

    def test_is_permutation(self):
        """Sinkhorn coupling should be a permutation of the input rows."""
        from n2v.probabilistic.flow.train import sinkhorn_coupling

        x0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        x1 = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        x0_c, x1_c = sinkhorn_coupling(x0, x1, reg=0.05)

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
        x0_c, x1_c = sinkhorn_coupling(x0, x1, reg=0.05)
        assert x0_c.device.type == 'cuda'
        assert x1_c.device.type == 'cuda'

    def test_agrees_with_hungarian_on_easy_case(self):
        """On well-separated clusters, Sinkhorn should match Hungarian."""
        from n2v.probabilistic.flow.train import ot_coupling, sinkhorn_coupling

        x0 = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        x1 = torch.tensor([[0.1, 0.1], [10.1, 0.1], [0.1, 10.1]])

        _, x1_h = ot_coupling(x0, x1)
        _, x1_s = sinkhorn_coupling(x0, x1, reg=0.05)

        torch.testing.assert_close(x1_h, x1_s)


def test_sinkhorn_coupling_requires_explicit_reg():
    """sinkhorn_coupling must not have a default reg parameter."""
    import torch
    from n2v.probabilistic.flow.train import sinkhorn_coupling
    x0 = torch.randn(16, 2)
    x1 = torch.randn(16, 2)
    # Should raise TypeError because reg has no default
    with pytest.raises(TypeError, match="missing.*required.*argument.*reg"):
        sinkhorn_coupling(x0, x1)


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


def test_train_flow_accepts_sinkhorn_iters():
    """train_flow should accept sinkhorn_iters as a keyword argument."""
    import torch
    from n2v.probabilistic.flow import VelocityField, train_flow
    torch.manual_seed(0)
    vf = VelocityField(dim=2, hidden=16, n_layers=2)
    data = torch.randn(64, 2)
    # This should not raise — passes explicit sinkhorn_iters
    train_flow(
        vf, data,
        n_epochs=2, batch_size=32, lr=1e-3,
        coupling='sinkhorn',
        sinkhorn_iters=5,
    )


def test_train_flow_accepts_sinkhorn_reg():
    """train_flow should accept sinkhorn_reg as a keyword argument."""
    import torch
    from n2v.probabilistic.flow import VelocityField, train_flow
    torch.manual_seed(0)
    vf = VelocityField(dim=2, hidden=16, n_layers=2)
    data = torch.randn(64, 2)
    # Passing a custom numeric reg should not raise
    train_flow(
        vf, data,
        n_epochs=2, batch_size=32, lr=1e-3,
        coupling='sinkhorn',
        sinkhorn_reg=0.1,
    )


def test_adaptive_sinkhorn_reg_has_floor():
    """Adaptive reg should never return less than 1e-6 even on degenerate data."""
    import torch
    from n2v.probabilistic.flow.train import compute_adaptive_sinkhorn_reg
    # Edge case: all-zero data. The probe noise is nonzero so distances
    # are still nonzero, but we still want a floor in case something odd happens.
    data = torch.zeros(10, 2)
    reg = compute_adaptive_sinkhorn_reg(data, alpha=0.1)
    assert reg >= 1e-6


def test_adaptive_sinkhorn_reg_stable_at_classifier_scale():
    """Adaptive reg should keep cost/reg in a numerically stable range
    for classifier-scale data (median squared distance ~9)."""
    import torch
    from n2v.probabilistic.flow.train import compute_adaptive_sinkhorn_reg
    torch.manual_seed(0)
    # Simulate classifier-r=1 output distribution scale
    data = torch.randn(1000, 3) * 1.8  # std per dim ~ 1.8
    reg = compute_adaptive_sinkhorn_reg(data, alpha=0.1)
    # Compute median cost/reg with fresh noise
    noise = torch.randn_like(data)
    cost_sq = (torch.cdist(noise, data, p=2) ** 2).median().item()
    ratio = cost_sq / reg
    # cost/reg should be < 30 so that exp(-cost/reg) > exp(-30) ≈ 1e-13
    assert ratio < 30, (
        f"cost/reg={ratio:.1f} too large, K entries will underflow"
    )


def test_train_flow_default_sinkhorn_reg_is_auto():
    """When sinkhorn_reg is not specified, train_flow should use adaptive reg."""
    import inspect
    from n2v.probabilistic.flow import train_flow
    sig = inspect.signature(train_flow)
    assert sig.parameters['sinkhorn_reg'].default == 'auto', (
        "Expected default sinkhorn_reg to be 'auto', got "
        f"{sig.parameters['sinkhorn_reg'].default}"
    )


def test_train_flow_auto_computes_adaptive_reg(monkeypatch):
    """train_flow with sinkhorn_reg='auto' should call compute_adaptive_sinkhorn_reg."""
    import torch
    from n2v.probabilistic.flow import VelocityField, train_flow
    import n2v.probabilistic.flow.train as train_mod

    calls = []
    original = train_mod.compute_adaptive_sinkhorn_reg

    def spy(training_outputs, **kwargs):
        value = original(training_outputs, **kwargs)
        calls.append(value)
        return value

    monkeypatch.setattr(train_mod, 'compute_adaptive_sinkhorn_reg', spy)

    torch.manual_seed(0)
    vf = VelocityField(dim=2, hidden=16, n_layers=2)
    data = torch.randn(64, 2) * 2.0
    train_flow(
        vf, data,
        n_epochs=1, batch_size=32, lr=1e-3,
        coupling='sinkhorn',
        sinkhorn_reg='auto',
    )
    assert len(calls) == 1, "compute_adaptive_sinkhorn_reg should be called once"
    assert calls[0] > 0
    assert calls[0] != 0.05, "adaptive reg should differ from the old hardcoded 0.05"


class TestEMA:

    def test_default_no_ema(self):
        """Default behavior is unchanged — no EMA."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        torch.manual_seed(0)
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        vf_trained, _ = train_flow(
            vf, data, n_epochs=5, batch_size=16, coupling='none'
        )
        # No exception; existing behavior preserved
        assert isinstance(vf_trained, VelocityField)

    def test_ema_changes_final_weights(self):
        """With EMA on, final weights differ from no-EMA run (same seed)."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow

        def run(use_ema):
            torch.manual_seed(0)
            vf = VelocityField(dim=2, hidden=8, n_layers=3)
            data = torch.randn(50, 2)
            vf_trained, _ = train_flow(
                vf, data, n_epochs=5, batch_size=16,
                coupling='none', use_ema=use_ema, ema_decay=0.9,
            )
            return [p.detach().clone() for p in vf_trained.parameters()]

        params_noema = run(use_ema=False)
        params_ema = run(use_ema=True)
        # At least one parameter differs
        any_diff = any(
            not torch.allclose(a, b)
            for a, b in zip(params_noema, params_ema)
        )
        assert any_diff


class TestAdamW:

    def test_default_is_adam(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        # Should run without raising (default optimizer='adam')
        train_flow(vf, data, n_epochs=2, batch_size=16, coupling='none')

    def test_adamw_runs(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        train_flow(
            vf, data, n_epochs=2, batch_size=16, coupling='none',
            optimizer='adamw', weight_decay=1e-4,
        )

    def test_invalid_optimizer_raises(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        with pytest.raises(ValueError, match="optimizer"):
            train_flow(
                vf, data, n_epochs=2, batch_size=16, coupling='none',
                optimizer='sgd',
            )


class TestLRWarmup:

    def test_default_no_warmup(self):
        """Default lr_warmup_frac=0.0 preserves existing cosine schedule."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        train_flow(vf, data, n_epochs=3, batch_size=16, coupling='none')

    def test_warmup_runs(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        train_flow(
            vf, data, n_epochs=10, batch_size=16, coupling='none',
            lr_warmup_frac=0.2,
        )


def test_adaptive_sinkhorn_reg_stable_at_banana_scale():
    """Adaptive reg should keep cost/reg in a numerically stable range
    for small-scale data (banana r=0.05-like, std ~0.03 per dim)."""
    import torch
    from n2v.probabilistic.flow.train import compute_adaptive_sinkhorn_reg
    torch.manual_seed(0)
    # Simulate banana r=0.05 output distribution scale
    data = torch.randn(1000, 2) * 0.03
    reg = compute_adaptive_sinkhorn_reg(data, alpha=0.1)
    # Compute median cost/reg with fresh noise (matching what Sinkhorn actually sees)
    noise = torch.randn_like(data)
    cost_sq = (torch.cdist(noise, data, p=2) ** 2).median().item()
    ratio = cost_sq / reg
    # cost/reg should be < 30 so that exp(-cost/reg) > exp(-30) ≈ 1e-13
    assert ratio < 30, (
        f"cost/reg={ratio:.1f} too large, K entries will underflow"
    )
    # Also verify it's not absurdly small (which would mean blurry OT)
    assert ratio > 0.01, (
        f"cost/reg={ratio:.4f} too small, Sinkhorn coupling will be essentially uniform"
    )


class TestGradClip:

    def test_default_is_none(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        train_flow(vf, data, n_epochs=2, batch_size=16, coupling='none')

    def test_grad_clip_runs(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        train_flow(
            vf, data, n_epochs=2, batch_size=16, coupling='none',
            grad_clip=1.0,
        )


class TestTimeSampling:

    def test_default_is_uniform(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        train_flow(vf, data, n_epochs=2, batch_size=16, coupling='none')

    def test_logit_normal_runs(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        train_flow(
            vf, data, n_epochs=2, batch_size=16, coupling='none',
            time_sampling='logit_normal',
        )

    def test_invalid_time_sampling_raises(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        with pytest.raises(ValueError, match="time_sampling"):
            train_flow(
                vf, data, n_epochs=2, batch_size=16, coupling='none',
                time_sampling='beta',
            )


class TestStandardizeOutputs:

    def test_default_no_standardization(self):
        """Default path — no buffers registered."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        vf_trained, _ = train_flow(
            vf, data, n_epochs=3, batch_size=16, coupling='none'
        )
        assert vf_trained.y_mean is None

    def test_standardization_registers_buffers(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2) * 10.0 + 5.0
        vf_trained, _ = train_flow(
            vf, data, n_epochs=3, batch_size=16, coupling='none',
            standardize_outputs=True,
        )
        assert vf_trained.y_mean is not None
        assert vf_trained.y_std is not None
        assert vf_trained.y_mean.shape == (2,)
        assert vf_trained.y_std.shape == (2,)
        # Mean and std should be close to the data stats
        torch.testing.assert_close(
            vf_trained.y_mean, data.mean(0), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            vf_trained.y_std, data.std(0), atol=1e-4, rtol=1e-4
        )

    def test_forward_runs_with_buffers(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2) * 10.0 + 5.0
        vf_trained, _ = train_flow(
            vf, data, n_epochs=3, batch_size=16, coupling='none',
            standardize_outputs=True,
        )
        y_test = torch.randn(4, 2) * 10.0 + 5.0
        t_test = torch.rand(4)
        out = vf_trained(t_test, y_test)
        assert out.shape == (4, 2)


class TestCouplingBatchSize:

    def test_default_is_none(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(200, 2)
        train_flow(
            vf, data, n_epochs=2, batch_size=32, coupling='sinkhorn',
        )

    def test_larger_coupling_batch_runs(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(200, 2)
        train_flow(
            vf, data, n_epochs=2, batch_size=32, coupling='sinkhorn',
            coupling_batch_size=128,
        )

    def test_non_multiple_raises(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(200, 2)
        with pytest.raises(ValueError, match="multiple"):
            train_flow(
                vf, data, n_epochs=2, batch_size=32, coupling='sinkhorn',
                coupling_batch_size=100,  # not a multiple of 32
            )


class TestRefreshData:

    def test_default_no_refresh(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        train_flow(vf, data, n_epochs=2, batch_size=16, coupling='none')

    def test_refresh_called_each_epoch(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        calls = {'count': 0}

        def refresher():
            calls['count'] += 1
            return torch.randn(50, 2)

        train_flow(
            vf, data, n_epochs=4, batch_size=16, coupling='none',
            refresh_data_each_epoch=refresher,
        )
        assert calls['count'] == 4


class TestFixedNoise:
    """Tests for the fixed_noise kwarg on train_flow (ReFlow entry point)."""

    def test_fixed_noise_runs(self):
        """train_flow with fixed_noise + coupling='none' completes training."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        torch.manual_seed(0)
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        noise = torch.randn(50, 2)
        vf_trained, losses = train_flow(
            vf, data,
            n_epochs=3, batch_size=16,
            coupling='none',
            fixed_noise=noise,
        )
        assert isinstance(vf_trained, VelocityField)
        assert len(losses) == 3

    def test_fixed_noise_uses_provided_values(self):
        """Different fixed_noise tensors → different trained weights."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow

        def run(noise):
            torch.manual_seed(0)
            vf = VelocityField(dim=2, hidden=8, n_layers=3)
            data = torch.randn(50, 2)
            vf_trained, _ = train_flow(
                vf, data,
                n_epochs=3, batch_size=16,
                coupling='none',
                fixed_noise=noise,
            )
            return [p.detach().clone() for p in vf_trained.parameters()]

        torch.manual_seed(0)
        noise_a = torch.randn(50, 2)
        torch.manual_seed(1)
        noise_b = torch.randn(50, 2)
        params_a = run(noise_a)
        params_b = run(noise_b)
        any_diff = any(
            not torch.allclose(a, b) for a, b in zip(params_a, params_b)
        )
        assert any_diff, "fixed_noise tensors had no effect on trained weights"

    def test_fixed_noise_requires_coupling_none(self):
        """fixed_noise with coupling != 'none' must raise ValueError."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        noise = torch.randn(50, 2)
        with pytest.raises(ValueError, match="coupling"):
            train_flow(
                vf, data,
                n_epochs=2, batch_size=16,
                coupling='sinkhorn',
                fixed_noise=noise,
            )

    def test_fixed_noise_shape_must_match(self):
        """fixed_noise with wrong shape must raise ValueError."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.train import train_flow
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        data = torch.randn(50, 2)
        noise = torch.randn(40, 2)  # wrong shape
        with pytest.raises(ValueError, match="shape"):
            train_flow(
                vf, data,
                n_epochs=2, batch_size=16,
                coupling='none',
                fixed_noise=noise,
            )


class TestGenerateReflowPairs:
    """Tests for generate_reflow_pairs helper."""

    def test_shapes_and_types(self):
        """Pairs have correct shape, dtype, and device."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.train import generate_reflow_pairs

        torch.manual_seed(0)
        vf = VelocityField(dim=3, hidden=8, n_layers=3)
        flow_ode = FlowODE(vf)

        z0, x1 = generate_reflow_pairs(
            flow_ode, n_pairs=64, dim=3, device='cpu', n_steps=20, seed=0,
        )
        assert z0.shape == (64, 3)
        assert x1.shape == (64, 3)
        assert z0.dtype == torch.float32
        assert x1.dtype == torch.float32
        assert z0.device.type == 'cpu'
        assert x1.device.type == 'cpu'

    def test_deterministic_with_seed(self):
        """Same seed -> same pairs."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.train import generate_reflow_pairs

        torch.manual_seed(0)
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        flow_ode = FlowODE(vf)

        z0_a, x1_a = generate_reflow_pairs(
            flow_ode, n_pairs=32, dim=2, device='cpu', n_steps=20, seed=42,
        )
        z0_b, x1_b = generate_reflow_pairs(
            flow_ode, n_pairs=32, dim=2, device='cpu', n_steps=20, seed=42,
        )
        torch.testing.assert_close(z0_a, z0_b)
        torch.testing.assert_close(x1_a, x1_b)

    def test_different_seeds_give_different_pairs(self):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.train import generate_reflow_pairs

        torch.manual_seed(0)
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        flow_ode = FlowODE(vf)

        z0_a, _ = generate_reflow_pairs(
            flow_ode, n_pairs=32, dim=2, device='cpu', n_steps=20, seed=0,
        )
        z0_b, _ = generate_reflow_pairs(
            flow_ode, n_pairs=32, dim=2, device='cpu', n_steps=20, seed=1,
        )
        assert not torch.allclose(z0_a, z0_b)

    def test_outputs_have_no_grad(self):
        """Returned tensors should not track grad (data generation path)."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.train import generate_reflow_pairs

        torch.manual_seed(0)
        vf = VelocityField(dim=2, hidden=8, n_layers=3)
        flow_ode = FlowODE(vf)

        z0, x1 = generate_reflow_pairs(
            flow_ode, n_pairs=16, dim=2, device='cpu', n_steps=10, seed=0,
        )
        assert not z0.requires_grad
        assert not x1.requires_grad
