"""Unit tests for LogDetFlowScore.

Tests are written TDD-first: they import from
`n2v.probabilistic.flow.logdet_scores`, which does not exist yet. They
should fail on import until the module is implemented.
"""

import math

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Small helper velocity fields used across tests.
#
# These mimic the (batch,) t, (batch, dim) y signature expected by FlowODE
# but have simple, analytically tractable forms. They are NOT `VelocityField`
# subclasses; they are just nn.Modules with the right forward signature.
# ---------------------------------------------------------------------------


class LinearVelocityField(nn.Module):
    """Constant linear velocity: v(t, x) = x @ A.T (t is ignored).

    Under this velocity, the flow from t=0 to t=1 is the linear map exp(A),
    so the inverse (integrating from t=1 to t=0) is exp(-A).
    """

    def __init__(self, A: torch.Tensor):
        super().__init__()
        # Register as a buffer so .to(device) works; shape (d, d).
        self.register_buffer('A', A.clone())

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (batch,) -- ignored because A is constant in t
        return x @ self.A.T


class ZeroVelocityField(nn.Module):
    """v(t, x) = 0 everywhere. The flow is the identity; log|det J| = 0."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _make_flow_ode(velocity_field):
    """Wrap a velocity field in a FlowODE-like object."""
    from n2v.probabilistic.flow.ode import FlowODE
    return FlowODE(velocity_field)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype():
    """Score should be (batch,) float32."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    dim = 3
    flow = _make_flow_ode(ZeroVelocityField(dim))
    score_fn = LogDetFlowScore(flow, t=1.0, n_steps=50)

    y = torch.randn(5, dim, dtype=torch.float32)
    scores = score_fn(y)

    assert scores.shape == (5,)
    assert scores.dtype == torch.float32


def test_linear_flow_matches_analytical():
    """For v(t, x) = A x (constant A), log-det should equal trace(A).

    The inverse map is phi(y) = e^{-A} y, and
        s_logdet(y) = (1/2) ||e^{-A} y||^2 + trace(A).
    """
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    # A = diag([0.1, -0.2, 0.3]), trace = 0.2
    diag_vals = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    A = torch.diag(diag_vals)
    trace_A = diag_vals.sum().item()
    assert abs(trace_A - 0.2) < 1e-12

    vfield = LinearVelocityField(A.to(torch.float32))
    flow = _make_flow_ode(vfield)

    # Use a reasonable number of steps for tight tolerance.
    score_fn = LogDetFlowScore(flow, t=1.0, n_steps=200)

    y = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    # Analytical phi(y) for a diagonal constant A: e^{-A} y is
    # componentwise exp(-diag) * y.
    exp_neg_diag = torch.exp(-diag_vals)  # float64
    z = exp_neg_diag * y[0].to(torch.float64)
    expected_score = 0.5 * (z ** 2).sum().item() + trace_A

    actual = score_fn(y).item()

    assert abs(actual - expected_score) < 1e-3, (
        f"expected {expected_score:.6f}, got {actual:.6f}, "
        f"diff = {actual - expected_score:.6f}"
    )


def test_composition_with_flowscore():
    """On v = 0 identity flow, LogDetFlowScore reduces to (1/2) ||y||^2."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    dim = 3
    flow = _make_flow_ode(ZeroVelocityField(dim))
    score_fn = LogDetFlowScore(flow, t=1.0, n_steps=50)

    y = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    expected = 0.5 * (y ** 2).sum(dim=1)

    actual = score_fn(y)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_deterministic():
    """Repeated calls on the same input must return the same score."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    torch.manual_seed(0)
    # Use a small random-weight linear flow to exercise real numerics.
    A = 0.1 * torch.randn(3, 3)
    flow = _make_flow_ode(LinearVelocityField(A))
    score_fn = LogDetFlowScore(flow, t=1.0, n_steps=100)

    y = torch.randn(4, 3)
    s1 = score_fn(y)
    s2 = score_fn(y)

    assert torch.allclose(s1, s2, atol=0.0, rtol=0.0)


def test_preserves_caller_device_on_cpu():
    """Input on CPU should produce output on CPU."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    dim = 3
    flow = _make_flow_ode(ZeroVelocityField(dim))
    score_fn = LogDetFlowScore(flow, t=1.0, n_steps=50)

    y = torch.randn(2, dim, device='cpu')
    scores = score_fn(y)

    assert scores.device.type == 'cpu'


def test_set_t_updates_t():
    """set_t should update the stored t used for subsequent calls."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    dim = 3
    flow = _make_flow_ode(ZeroVelocityField(dim))
    score_fn = LogDetFlowScore(flow, t=1.0)
    score_fn.set_t(0.5)
    assert score_fn.t == 0.5


def test_accepts_method_rk4():
    """rk4 is a valid solver choice."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    dim = 3
    flow = _make_flow_ode(ZeroVelocityField(dim))
    score_fn = LogDetFlowScore(flow, t=1.0, n_steps=30, method='rk4')
    y = torch.randn(2, dim)
    # On zero-flow, score should still be (1/2) ||y||^2 regardless of method.
    expected = 0.5 * (y ** 2).sum(dim=1)
    assert torch.allclose(score_fn(y), expected, atol=1e-5)


def test_accepts_atol_rtol():
    """atol and rtol should be settable and used by adaptive solvers."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    dim = 3
    flow = _make_flow_ode(ZeroVelocityField(dim))
    score_fn = LogDetFlowScore(
        flow, t=1.0, n_steps=30, method='dopri5', atol=1e-4, rtol=1e-4,
    )
    assert score_fn.atol == 1e-4
    assert score_fn.rtol == 1e-4
    # Sanity: still produces correct output on zero-flow.
    y = torch.randn(2, dim)
    expected = 0.5 * (y ** 2).sum(dim=1)
    assert torch.allclose(score_fn(y), expected, atol=1e-5)


def test_rk4_matches_dopri5_on_linear_flow():
    """rk4 and dopri5 should agree to 1e-3 on a simple linear flow."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    diag_vals = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    A = torch.diag(diag_vals)
    flow = _make_flow_ode(LinearVelocityField(A))

    y = torch.randn(5, 3)
    dopri5_scores = LogDetFlowScore(
        flow, t=1.0, n_steps=200, method='dopri5',
    )(y)
    rk4_scores = LogDetFlowScore(
        flow, t=1.0, n_steps=200, method='rk4',
    )(y)
    assert torch.allclose(dopri5_scores, rk4_scores, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='requires CUDA GPU')
def test_cpu_input_gpu_flow_returns_cpu_scores():
    """Input on CPU with flow on GPU: scores should come back on CPU."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    dim = 3
    vfield = ZeroVelocityField(dim).cuda()
    flow = _make_flow_ode(vfield)
    score_fn = LogDetFlowScore(flow, t=1.0, n_steps=30, method='rk4')

    y = torch.randn(5, dim, device='cpu')
    scores = score_fn(y)
    assert scores.device.type == 'cpu'
    expected = 0.5 * (y ** 2).sum(dim=1)
    assert torch.allclose(scores, expected, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='requires CUDA GPU')
def test_gpu_input_gpu_flow_returns_gpu_scores():
    """Input on GPU with flow on GPU: scores stay on GPU."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    dim = 3
    vfield = ZeroVelocityField(dim).cuda()
    flow = _make_flow_ode(vfield)
    score_fn = LogDetFlowScore(flow, t=1.0, n_steps=30, method='rk4')

    y = torch.randn(5, dim, device='cuda')
    scores = score_fn(y)
    assert scores.device.type == 'cuda'


def test_batch_size_chunks_correctly():
    """Chunked evaluation should match unchunked evaluation element-wise."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    A = 0.1 * torch.randn(3, 3, generator=torch.Generator().manual_seed(0))
    flow = _make_flow_ode(LinearVelocityField(A))

    y = torch.randn(
        50, 3, generator=torch.Generator().manual_seed(1),
    )

    unchunked = LogDetFlowScore(flow, t=1.0, n_steps=50, method='rk4')
    chunked = LogDetFlowScore(
        flow, t=1.0, n_steps=50, method='rk4', batch_size=7,
    )
    assert torch.allclose(unchunked(y), chunked(y), atol=1e-5)


def test_batch_size_none_means_no_chunking():
    """batch_size=None (default) should evaluate the whole input at once."""
    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

    flow = _make_flow_ode(ZeroVelocityField(3))
    score_fn = LogDetFlowScore(flow, t=1.0, n_steps=30, method='rk4')
    assert score_fn.batch_size is None

    y = torch.randn(100, 3)
    scores = score_fn(y)
    assert scores.shape == (100,)
