"""Unit tests for flow matching components."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from tests.known_bugs import known_bug


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

    @pytest.mark.gpu
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


# ---------------------------------------------------------------------------
# Closed forms of the flow ODE, the sameness of forward and inverse, and the
# networks the velocity fields document.
#
# FlowODE.forward maps data to latent (integrates from t down to 0) and
# FlowODE.inverse maps latent to data (from 0 up to t). The flow set is
# defined with forward and the scenario certifier samples it with inverse, so
# the two must be inverse to each other within the integration tolerance and
# both must integrate over the stated times. References are flows known in
# closed form (a field polynomial in time: a shift; a linear field: a matrix
# exponential, or the polynomial of a fixed-step method), never n2v.
# ---------------------------------------------------------------------------

_C2 = torch.tensor([3.0, -0.75], dtype=torch.float64)
_B2 = torch.tensor([-6.0, 2.5], dtype=torch.float64)
_A2 = torch.tensor([[0.5, -1.25], [0.75, -0.5]], dtype=torch.float64)


class _ShiftField(nn.Module):
    """v(s, x) = c + s b, independent of x.

    forward(y, t) = y - c t - b t^2 / 2 and inverse(z, t) = z + c t + b t^2 / 2
    exactly; a solver that hands the field 1 - s instead of s agrees with this
    only at t = 1.
    """

    def __init__(self, c, b):
        super().__init__()
        self.register_buffer('c', c.clone())
        self.register_buffer('b', b.clone())

    def forward(self, t, x):
        return (self.c + t.to(x.dtype).unsqueeze(1) * self.b).expand_as(x)


class _LinearField(nn.Module):
    """v(s, x) = x A^T: forward(y, t) = expm(-A t) y, inverse(z, t) = expm(A t) z."""

    def __init__(self, A):
        super().__init__()
        self.register_buffer('A', A.clone())

    def forward(self, t, x):
        return x @ self.A.T


def _pts2(n, seed):
    g = torch.Generator().manual_seed(seed)
    return 3.0 * (2.0 * torch.rand(n, 2, generator=g, dtype=torch.float64) - 1.0)


def _step_matrix(method, Z):
    """One step of 'euler' (I + Z) or of a four-stage fourth-order
    Runge-Kutta method (the degree-4 Taylor polynomial of expm(Z)) on the
    linear system y' = M y with Z = h M."""
    I = torch.eye(Z.shape[0], dtype=Z.dtype)
    if method == 'euler':
        return I + Z
    return I + Z + Z @ Z / 2 + Z @ Z @ Z / 6 + Z @ Z @ Z @ Z / 24


class TestFlowODEExactFlows:
    """forward and inverse integrate over the stated times, in the stated
    direction, with the stated solver and tolerances."""

    @pytest.mark.parametrize('method,n_steps', [('dopri5', 5), ('rk4', 30)])
    def test_forward_and_inverse_match_the_exact_flow(self, method, n_steps):
        """On v = c + s b at t = 0.5: forward(y) = y - c t - b t^2/2 and
        inverse(z) = z + c t + b t^2/2. Both solvers are exact for a field of
        degree one in s; the time grid is float32, which moves each time by
        at most 2^-26 here, so the error is below t |b| 2^-25 (about 1e-7);
        the bound is 1e-5. Catches forward integrating from 0 up to t,
        inverse integrating from t down to 0, inverse returning its start
        point, inverse stopping at 1 instead of t, and either one handing the
        field the time 1 - s."""
        from n2v.probabilistic.flow.ode import FlowODE

        flow = FlowODE(_ShiftField(_C2, _B2))
        t = 0.5
        shift = _C2 * t + _B2 * t * t / 2
        y = _pts2(5, seed=0)
        f = flow.forward(y, t=t, n_steps=n_steps, method=method)
        i = flow.inverse(y, t=t, n_steps=n_steps, method=method)
        assert (f - (y - shift)).abs().max().item() <= 1e-5
        assert (i - (y + shift)).abs().max().item() <= 1e-5

    @pytest.mark.parametrize('method', ['euler', 'rk4'])
    def test_fixed_step_method_reaches_the_solver(self, method):
        """On v = A y at t = 0.5 with ``n_steps=4``, forward is S(-hA)^k y and
        inverse S(hA)^k z, S one step of the method and h = t / k. The step
        count is the subject of the marked cases of
        ``TestFlowODEStepCount``, so k = n_steps and k = n_steps - 1 are both
        accepted; the exact flow (what dopri5 returns) is neither. The time
        grid's float32 rounding changes the step sizes but not their sum,
        which moves the result by the square of that rounding (about 1e-15);
        the bound is 1e-10. Catches a method that does not reach ``odeint``,
        so 'euler' and 'rk4' silently run dopri5, in either direction."""
        from n2v.probabilistic.flow.ode import FlowODE

        flow = FlowODE(_LinearField(_A2))
        t, n_steps = 0.5, 4
        y = _pts2(5, seed=1)
        f = flow.forward(y, t=t, n_steps=n_steps, method=method)
        i = flow.inverse(y, t=t, n_steps=n_steps, method=method)
        err_f, err_i = [], []
        for k in (n_steps - 1, n_steps):
            h = t / k
            Sf = torch.linalg.matrix_power(_step_matrix(method, -h * _A2), k)
            Si = torch.linalg.matrix_power(_step_matrix(method, h * _A2), k)
            err_f.append((f - y @ Sf.T).abs().max().item())
            err_i.append((i - y @ Si.T).abs().max().item())
        assert min(err_f) <= 1e-10, err_f
        assert min(err_i) <= 1e-10, err_i

    def test_adaptive_tolerances_reach_the_solver(self):
        """dopri5 with atol = rtol = 1e-12 on float64 points of v = A y at
        t = 0.5 lands within about 1e-12 of expm(-A t) y (forward) and
        expm(A t) z (inverse); torchdiffeq's own defaults leave about 3e-8
        and n2v's default 1e-5 about 1e-5. The bound is 1e-10. Catches
        tolerances that do not reach ``odeint`` in either direction."""
        from n2v.probabilistic.flow.ode import FlowODE

        flow = FlowODE(_LinearField(_A2))
        t = 0.5
        y = _pts2(5, seed=2)
        f = flow.forward(y, t=t, n_steps=5, method='dopri5',
                         atol=1e-12, rtol=1e-12)
        i = flow.inverse(y, t=t, n_steps=5, method='dopri5',
                         atol=1e-12, rtol=1e-12)
        Ef = torch.linalg.matrix_exp(-_A2 * t)
        Ei = torch.linalg.matrix_exp(_A2 * t)
        assert (f - y @ Ef.T).abs().max().item() <= 1e-10
        assert (i - y @ Ei.T).abs().max().item() <= 1e-10

    def test_forward_trajectory_columns_follow_the_requested_times(self):
        """``forward_trajectory(y, [0.25, 0.5, 1.0])`` has one column per
        requested time, in that order, each ||forward(y, t_j)|| =
        ||y - c t_j - b t_j^2/2||. Catches columns in the wrong order (the
        descending copy of the times) and norms taken at the wrong time."""
        from n2v.probabilistic.flow.ode import FlowODE

        flow = FlowODE(_ShiftField(_C2, _B2))
        y = _pts2(4, seed=3)
        times = [0.25, 0.5, 1.0]
        norms = flow.forward_trajectory(y, times, n_steps=10)
        expected = torch.stack(
            [(y - _C2 * t - _B2 * t * t / 2).norm(dim=1) for t in times],
            dim=1)
        assert norms.shape == (4, 3)
        assert (norms - expected).abs().max().item() <= 1e-9


_STEP_COUNT_BUG = known_bug(
    13, "FlowODE.forward and FlowODE.inverse build their time grid with "
        "torch.linspace(.., n_steps) points, so a fixed-step solver takes "
        "n_steps - 1 steps although the docstring says n_steps is the exact "
        "number of steps; n_steps = 1 integrates nothing and returns the "
        "input unchanged")


class TestFlowODEStepCount:
    """For 'rk4' and 'euler' the docstrings say ``n_steps`` "is the exact
    number of steps"."""

    @pytest.mark.parametrize('direction,n_steps', [
        pytest.param('forward', 1, marks=_STEP_COUNT_BUG),
        ('forward', 2),
        pytest.param('inverse', 1, marks=_STEP_COUNT_BUG),
        ('inverse', 2),
    ])
    def test_fixed_step_count(self, direction, n_steps):
        """On a constant field c every rk4 step is exact, so any step count
        >= 1 moves a point by exactly c t (backward for forward, onward for
        inverse). ``n_steps = 1`` asks for one step; today the grid has a
        single point and nothing moves. ``n_steps = 2`` is the passing
        neighbour with the same reference."""
        from n2v.probabilistic.flow.ode import FlowODE

        flow = FlowODE(_ShiftField(_C2, torch.zeros(2, dtype=torch.float64)))
        y = _pts2(3, seed=4)
        t = 0.5
        if direction == 'forward':
            out, expected = flow.forward(y, t=t, n_steps=n_steps,
                                         method='rk4'), y - _C2 * t
        else:
            out, expected = flow.inverse(y, t=t, n_steps=n_steps,
                                         method='rk4'), y + _C2 * t
        assert (out - expected).abs().max().item() <= 1e-12


class TestForwardInvertsInverse:
    """The set is defined with forward and sampled with inverse."""

    def test_round_trip_at_half_time_with_the_verifier_defaults(self):
        """A VelocityField that reads t and y (output layer scaled so points
        move by more than 0.05), t = 0.5, rk4 with 30 grid points (the
        defaults of ProbVerifyConfig and FlowReachConfig): forward(inverse(z))
        returns z within 1e-4 (on this field the round trip is 6e-12 off in
        float64 and 2.4e-7 off in float32, the rounding of the float32
        network). Catches forward and inverse that integrate over
        different times (inverse to 1 instead of t, a time input of 1 - s on
        one side) on a field where nothing is in closed form."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE

        torch.manual_seed(1)
        vf = VelocityField(dim=2, hidden=16, n_layers=3)
        with torch.no_grad():
            vf.net[-1].weight.mul_(4.0)
            vf.net[-1].bias.mul_(4.0)
        flow = FlowODE(vf)
        z = torch.tensor([[0.5, -1.0], [-1.5, 0.25], [1.0, 1.0],
                          [0.0, -2.0]])
        with torch.no_grad():
            y = flow.inverse(z, t=0.5, n_steps=30, method='rk4')
            if (y - z).abs().max().item() <= 0.05:
                pytest.fail('setup: the field does not move the points')
            z_back = flow.forward(y, t=0.5, n_steps=30, method='rk4')
        assert (z_back - z).abs().max().item() <= 1e-4


class TestScenarioSamplesLandOnTheSet:
    """The public route of ``FlowODE.inverse``: the scenario method of
    ``verify_specification`` maps latent samples with inverse."""

    @pytest.mark.parametrize('n_ode_steps', [
        30,
        pytest.param(1, marks=known_bug(
            13, "with ProbVerifyConfig(n_ode_steps=1), which its own check "
                "accepts, FlowODE.inverse takes no step and returns the "
                "latent samples unchanged, so the scenario method samples "
                "the latent ball instead of the set and answers UNSAT for an "
                "unsafe band that holds 44 % of the set's mass")),
    ])
    def test_unsafe_band_inside_the_set_is_not_answered_unsat(self,
                                                              n_ode_steps):
        """Field v = c + s b with c = (-8, 0.5), b = (16, -1), flow time 0.5:
        the set {y : ||forward(y, 0.5)|| <= 1} is the unit disc around
        m = c/2 + b/8 = (-2, 0.125), and inverse(z, 0.5) = z + m. The unsafe
        band -2 <= y0 <= -1.25 holds the part 0 <= z0 <= 0.75 of the latent
        ball, 0.439 of the truncated Gaussian the scenario samples (by
        quadrature), so among 2000 samples the chance that none is unsafe is
        below 1e-300 and UNSAT is a false answer. A latent sample mapped to
        z (no step), z - m (wrong direction), z + c + b/2 (stopped at 1) or
        with time input 1 - s lands at y0 >= -1, outside the band, and the
        answer would be UNSAT. Catches each of those through the public
        entry. If ``n_steps`` is settled as a count of grid points, a
        configuration with one grid point may be refused with ``ValueError``
        instead; a refusal is not a false UNSAT, so it is accepted for
        ``n_ode_steps = 1`` only."""
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.scores import FlowScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet
        from n2v.sets import HalfSpace
        from n2v.utils.verify_specification import (
            ProbVerifyConfig, verify_specification,
        )

        c = torch.tensor([-8.0, 0.5])
        b = torch.tensor([16.0, -1.0])
        flow = FlowODE(_ShiftField(c, b))
        score = FlowScore(flow, t=0.5, n_steps=30, method='rk4')
        pset = ProbabilisticSet(score_fn=score, threshold=1.0, m=2000,
                                ell=2000, epsilon=0.01, dim=2)
        G = np.array([[-1.0, 0.0], [1.0, 0.0]])
        g = np.array([[2.0], [-1.25]])
        spec = HalfSpace(G, g)
        witness = torch.tensor([[-1.5, 0.125]])
        if not (bool(pset.contains(witness)[0])
                and np.all(G @ witness.numpy().T <= g)):
            pytest.fail('setup: the witness is not an unsafe member')
        try:
            res = verify_specification(
                pset, spec, config=ProbVerifyConfig(
                    method='scenario', n_samples=2000, seed=0, t=0.5,
                    n_ode_steps=n_ode_steps))
        except ValueError:
            if n_ode_steps == 1:
                return
            raise
        assert res.verdict != 'UNSAT'


def _act(name, x):
    """ReLU as max(x, 0) and SiLU as x * sigmoid(x), written out."""
    return x.clamp_min(0.0) if name == 'relu' else x * torch.sigmoid(x)


def _check_sinusoidal_freqs(freqs):
    """16 frequencies spaced logarithmically from 1 to 1000 (numpy's
    geomspace); stored in float32, hence the relative 1e-6."""
    ref = torch.tensor(np.geomspace(1.0, 1000.0, 16), dtype=freqs.dtype)
    torch.testing.assert_close(freqs, ref, rtol=1e-6, atol=0.0)


def _velocity_by_hand(vf, t, y, *, activation, n_layers, residual=False,
                      layer_norm=False, time_embed='concat', mean=None,
                      std=None):
    """The network VelocityField documents, composed from its own weights.

    Input [t, y_w] ('concat') or [sin(t f), cos(t f), y_w] ('sinusoidal'),
    y_w = (y - mean) / std when whitening buffers are set; then
    act(Linear), n_layers - 2 middle layers, each act(Linear) or the residual
    block h + Linear(act(LN(Linear(h)))) (LN only with layer_norm), and a
    final Linear whose output is multiplied by std. The layer counts are
    asserted, so a network with a layer too many or too few fails here.
    """
    t = t.expand(y.shape[0]) if t.dim() == 0 else t
    y_w = y if mean is None else (y - mean) / std
    if time_embed == 'concat':
        x = torch.cat([t.unsqueeze(1), y_w], dim=1)
    else:
        a = t.unsqueeze(1) * vf.sinusoidal_freqs.unsqueeze(0)
        x = torch.cat([a.sin(), a.cos(), y_w], dim=1)
    linears = [m for m in vf.net if isinstance(m, nn.Linear)]
    blocks = [m for m in vf.net if hasattr(m, 'lin1')]
    if residual:
        assert len(linears) == 2 and len(blocks) == n_layers - 2
    else:
        assert len(linears) == n_layers and not blocks
    h = _act(activation, linears[0](x))
    if residual:
        for blk in blocks:
            u = blk.lin1(h)
            if layer_norm:
                mu = u.mean(-1, keepdim=True)
                var = ((u - mu) ** 2).mean(-1, keepdim=True)
                u = ((u - mu) / torch.sqrt(var + blk.ln.eps) * blk.ln.weight
                     + blk.ln.bias)
            h = h + blk.lin2(_act(activation, u))
    else:
        for lin in linears[1:-1]:
            h = _act(activation, lin(h))
    out = linears[-1](h)
    return out if std is None else out * std


class TestVelocityFieldComputesTheDocumentedNetwork:
    """VelocityField's forward is the network its docstring describes, for
    every option; float64, so the comparison bound is 1e-12."""

    _t = torch.tensor([0.1, 0.4, 0.7, 0.95], dtype=torch.float64)

    @pytest.mark.parametrize('activation,n_layers', [('relu', 4),
                                                     ('silu', 2)])
    def test_plain_network_matches_hand_composition(self, activation,
                                                    n_layers):
        """Catches a layer count other than n_layers (one middle layer too
        many, including at n_layers = 2 where there is none), an activation
        hard-wired at any place, and a time input that is not concatenated
        in front of y."""
        from n2v.probabilistic.flow.model import VelocityField

        torch.manual_seed(0)
        vf = VelocityField(dim=3, hidden=8, n_layers=n_layers,
                           activation=activation).double()
        y = 3.0 * torch.randn(4, 3, dtype=torch.float64)
        with torch.no_grad():
            out = vf(self._t, y)
            hand = _velocity_by_hand(vf, self._t, y, activation=activation,
                                     n_layers=n_layers)
        assert (out - hand).abs().max().item() <= 1e-12

    @pytest.mark.parametrize('layer_norm', [False, True])
    def test_residual_network_matches_hand_composition(self, layer_norm):
        """Two residual blocks with ReLU; the LayerNorm weight and bias are
        set away from 1 and 0. Catches a block without its skip term, a
        LayerNorm applied to the block's input instead of to Linear(input),
        and blocks built with SiLU whatever the activation."""
        from n2v.probabilistic.flow.model import VelocityField

        torch.manual_seed(0)
        vf = VelocityField(dim=3, hidden=8, n_layers=4, activation='relu',
                           residual=True, layer_norm=layer_norm).double()
        with torch.no_grad():
            for m in vf.modules():
                if isinstance(m, nn.LayerNorm):
                    m.weight.uniform_(0.5, 2.0)
                    m.bias.uniform_(-1.0, 1.0)
        y = 3.0 * torch.randn(4, 3, dtype=torch.float64)
        with torch.no_grad():
            out = vf(self._t, y)
            hand = _velocity_by_hand(vf, self._t, y, activation='relu',
                                     n_layers=4, residual=True,
                                     layer_norm=layer_norm)
        assert (out - hand).abs().max().item() <= 1e-12

    def test_sinusoidal_embedding_matches_hand_composition(self):
        """The time enters as [sin(t f), cos(t f)] with the 16 frequencies
        spaced logarithmically from 1 to 1000; a scalar t is expanded to the
        batch. Catches frequencies spaced linearly, an embedding without its
        cosine half, and a scalar t that is not expanded."""
        from n2v.probabilistic.flow.model import VelocityField

        torch.manual_seed(0)
        vf = VelocityField(dim=3, hidden=8, n_layers=3,
                           time_embed='sinusoidal').double()
        _check_sinusoidal_freqs(vf.sinusoidal_freqs)
        y = 3.0 * torch.randn(4, 3, dtype=torch.float64)
        t = torch.tensor(0.3, dtype=torch.float64)
        with torch.no_grad():
            out = vf(t, y)
            hand = _velocity_by_hand(vf, t, y, activation='silu',
                                     n_layers=3, time_embed='sinusoidal')
        assert (out - hand).abs().max().item() <= 1e-12

    @pytest.mark.parametrize('time_embed', ['concat', 'sinusoidal'])
    def test_whitening_buffers_whiten_the_input_and_scale_the_output(
            self, time_embed):
        """With y_mean = (5, -3, 0.5) and y_std = (4, 0.25, 1.5), as
        ``train_flow(standardize_outputs=True)`` registers them, forward is
        std * net(t, (y - mean) / std): the network sees whitened points and
        the velocity is mapped back to the original coordinates. Catches a
        velocity divided by std instead of multiplied, and raw y fed to the
        network in either time embedding."""
        from n2v.probabilistic.flow.model import VelocityField

        torch.manual_seed(0)
        vf = VelocityField(dim=3, hidden=8, n_layers=3,
                           time_embed=time_embed).double()
        mean = torch.tensor([5.0, -3.0, 0.5], dtype=torch.float64)
        std = torch.tensor([4.0, 0.25, 1.5], dtype=torch.float64)
        vf.y_mean = mean
        vf.y_std = std
        y = mean + std * torch.randn(4, 3, dtype=torch.float64)
        with torch.no_grad():
            out = vf(self._t, y)
            hand = _velocity_by_hand(vf, self._t, y, activation='silu',
                                     n_layers=3, time_embed=time_embed,
                                     mean=mean, std=std)
        assert (out - hand).abs().max().item() <= 1e-12


def _dit_by_hand(dit, t, y):
    """DiTLite as documented: per coordinate a token value_proj(y_k) +
    coord_embed(k) + time_proj([sin(t f), cos(t f)]), the encoder over the
    coordinate tokens, out_proj to one value per coordinate."""
    a = t.unsqueeze(1) * dit.time_freqs.unsqueeze(0)
    t_tok = dit.time_proj(torch.cat([a.sin(), a.cos()], dim=1)).unsqueeze(1)
    tokens = (dit.value_proj(y.unsqueeze(-1))
              + dit.coord_embed.weight.unsqueeze(0) + t_tok)
    return dit.out_proj(dit.blocks(tokens)).squeeze(-1)


class TestDiTLiteComputesTheDocumentedNetwork:
    """DiTLiteVelocityField composes its parts as documented."""

    def test_forward_matches_hand_composition_of_its_parts(self):
        """Three coordinates, a different t per row, float64, eval mode (no
        dropout). Catches time tokens computed but not added (the velocity
        would not depend on t), coordinate embeddings left out (coordinates
        indistinguishable), and wrong time frequencies."""
        from n2v.probabilistic.flow.model import DiTLiteVelocityField

        torch.manual_seed(0)
        dit = DiTLiteVelocityField(dim=3, hidden=8, n_blocks=1,
                                   n_heads=2).double().eval()
        _check_sinusoidal_freqs(dit.time_freqs)
        t = torch.tensor([0.1, 0.4, 0.7, 0.95], dtype=torch.float64)
        y = 3.0 * torch.randn(4, 3, dtype=torch.float64)
        with torch.no_grad():
            out = dit(t, y)
            hand = _dit_by_hand(dit, t, y)
        assert (out - hand).abs().max().item() <= 1e-12


class TestDiTLiteVelocityOfARowDoesNotDependOnItsBatch:
    """The velocity of a row is the same alone and inside a batch above the
    chunk size of 32768 rows."""

    def test_batch_above_the_chunk_size_returns_one_row_per_input(self):
        """32773 rows (one full chunk and a partial one) of a small float64
        DiTLite in eval mode: one output row per input row, and the rows on
        both sides of the chunk boundary and at the ends equal the hand
        composition of those rows alone. Catches a chunk loop that drops or
        repeats a row per chunk (the output is shifted against the input)."""
        from n2v.probabilistic.flow.model import DiTLiteVelocityField

        torch.manual_seed(0)
        dit = DiTLiteVelocityField(dim=2, hidden=8, n_blocks=1,
                                   n_heads=2).double().eval()
        n = 32768 + 5
        g = torch.Generator().manual_seed(1)
        y = torch.randn(n, 2, generator=g, dtype=torch.float64)
        t = torch.rand(n, generator=g, dtype=torch.float64)
        idx = torch.tensor([0, 1, 32766, 32767, 32768, 32769, n - 1])
        with torch.no_grad():
            out = dit(t, y)
            assert out.shape == (n, 2)
            hand = _dit_by_hand(dit, t[idx], y[idx])
        assert (out[idx] - hand).abs().max().item() <= 1e-12
