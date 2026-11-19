"""Unit tests for LogDetFlowScore and exact_trace.

The score is 0.5 ||psi(0)||^2 - L(0) for the coupled ODE that integrates the
flow and the trace of its Jacobian from the score's time t down to 0. The
cases check it against closed forms of linear fields, check that the time,
solver, step count and tolerances reach the integration, and check that the
score of a point does not depend on the other points of its batch.
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


@pytest.mark.gpu
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


@pytest.mark.gpu
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


# ---------------------------------------------------------------------------
# Closed forms, options and the sameness promise.
#
# For a linear field v(s, x) = A x the score has a closed form: psi(0) =
# expm(-A t) y and L(0) = -t tr(A), so s(y) = 0.5 ||expm(-A t) y||^2 + t tr(A);
# for v(s, x) = s A x the time t becomes t^2 / 2 in both terms. The matrix A
# below is not diagonal, so the trace (0.25) differs from the sum of all
# Jacobian entries (0.625). References come from torch.linalg.matrix_exp and
# hand-written traces, never from n2v.
# ---------------------------------------------------------------------------

from tests.known_bugs import known_bug  # noqa: E402

_A3 = torch.tensor([[0.5, -1.25, 0.25],
                    [0.75, -0.5, 2.0],
                    [-1.5, 0.125, 0.25]], dtype=torch.float64)


class TimeScaledLinearVelocityField(nn.Module):
    """v(s, x) = s * (x @ A.T): its flow from t down to 0 is expm(-A t^2/2)."""

    def __init__(self, A: torch.Tensor):
        super().__init__()
        self.register_buffer('A', A.clone())

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t.to(x.dtype).unsqueeze(1) * (x @ self.A.T)


class ConstantVelocityField(nn.Module):
    """v(s, x) = c: the flow is a shift by c times the elapsed time, the
    Jacobian is zero, and one Euler or rk4 step is exact."""

    def __init__(self, c: torch.Tensor):
        super().__init__()
        self.register_buffer('c', c.clone())

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.c.to(x.dtype).expand_as(x)


def _points(n, seed):
    g = torch.Generator().manual_seed(seed)
    return 3.0 * (2.0 * torch.rand(n, 3, generator=g, dtype=torch.float64)
                  - 1.0)


def _closed_form(y, tau, A=_A3):
    """0.5 ||expm(-A tau) y||^2 + tau tr(A)."""
    E = torch.linalg.matrix_exp(-A * tau)
    return 0.5 * ((y @ E.T) ** 2).sum(dim=1) + tau * torch.trace(A)


def _nonlinear_field(t_batch, x):
    """A field whose Jacobian has non-zero off-diagonal entries and that reads
    t; its trace, by hand: 2 x0 x1 + t + cos(x1) + x0 + x0."""
    return torch.stack([
        x[:, 0] ** 2 * x[:, 1] + t_batch * x[:, 0],
        torch.sin(x[:, 1]) + x[:, 0] * x[:, 1],
        x[:, 2] * x[:, 0] + x[:, 1] ** 3,
    ], dim=1)


@pytest.mark.parametrize('grad_mode', ['enabled', 'no_grad'])
def test_exact_trace_matches_the_hand_trace(grad_mode):
    """``exact_trace`` is the sum of the diagonal of the Jacobian: tr(A) for
    the linear field (0.25, while all entries sum to 0.625) and the hand
    trace of a nonlinear field that reads its time argument, row by row.
    The ``no_grad`` case runs the call inside ``torch.no_grad()``, as
    ``ProbabilisticSet.estimate_volume`` and ``boundary_2d`` do. Catches a
    trace that sums whole gradient rows (all Jacobian entries), a time that
    is not handed to the field, and gradients that stay off inside
    ``no_grad`` (the trace silently becomes 0)."""
    import contextlib
    from n2v.probabilistic.flow.logdet_scores import exact_trace

    x = _points(4, seed=0)
    t_batch = torch.tensor([0.25, -0.5, 1.5, 3.0], dtype=torch.float64)
    lin = LinearVelocityField(_A3)
    ctx = torch.no_grad() if grad_mode == 'no_grad' else contextlib.nullcontext()
    with ctx:
        tr_lin = exact_trace(lin, t_batch, x)
        tr_nl = exact_trace(_nonlinear_field, t_batch, x)
    torch.testing.assert_close(
        tr_lin, torch.full((4,), 0.25, dtype=torch.float64),
        rtol=0.0, atol=1e-12)
    hand = (2 * x[:, 0] * x[:, 1] + t_batch + torch.cos(x[:, 1])
            + 2 * x[:, 0])
    torch.testing.assert_close(tr_nl, hand, rtol=0.0, atol=1e-12)


class TestLogDetScoreMatchesItsClosedForm:
    """``LogDetFlowScore`` is 0.5 ||psi(0)||^2 - L(0) integrated from its own t
    with the solver, steps and tolerances it was given."""

    def test_linear_field_at_half_time_matches_the_closed_form(self):
        """t = 0.5 from the constructor, dopri5 at tolerances 1e-12, float64:
        within 1e-9 of 0.5 ||expm(-A/2) y||^2 + tr(A)/2 (the solver lands
        within about 1e-11; torchdiffeq's own default tolerances would leave
        about 4e-7). Catches integration from t = 1 instead of t, a log-det
        term added instead of subtracted, a trace that sums all Jacobian
        entries, and tolerances that do not reach the solver."""
        from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

        score = LogDetFlowScore(_make_flow_ode(LinearVelocityField(_A3)),
                                t=0.5, n_steps=5, method='dopri5',
                                atol=1e-12, rtol=1e-12)
        y = _points(4, seed=1)
        s = score(y)
        assert s.dtype == torch.float64
        assert (s - _closed_form(y, 0.5)).abs().max().item() <= 1e-9

    def test_set_t_moves_the_start_of_the_integration(self):
        """A score built with t = 1 and moved to t = 0.5 by ``set_t`` must
        give the t = 0.5 closed form; moved to t = 0 it must give
        0.5 ||y||^2 exactly (the flow has not moved a point at time 0).
        Catches a ``set_t`` or constructor t that the integration ignores and
        a wrong factor in the t = 0 shortcut."""
        from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

        score = LogDetFlowScore(_make_flow_ode(LinearVelocityField(_A3)),
                                t=1.0, n_steps=5, method='dopri5',
                                atol=1e-12, rtol=1e-12)
        y = _points(4, seed=2)
        score.set_t(0.5)
        assert (score(y) - _closed_form(y, 0.5)).abs().max().item() <= 1e-9
        score.set_t(0.0)
        assert torch.equal(score(y), 0.5 * (y ** 2).sum(dim=1))

    @pytest.mark.parametrize('method', ['euler', 'rk4'])
    def test_fixed_step_method_reaches_the_solver(self, method):
        """``method='euler'`` or ``'rk4'`` with ``n_steps=4`` at t = 0.5 gives
        0.5 ||S^k y||^2 + t tr(A), S one step of that method on y' = -A y
        (I - hA, or the degree-4 Taylor polynomial of expm(-hA)), h = t / k;
        the log-det term is exact for both since its derivative is constant.
        The step count is the subject of the marked case
        ``test_one_fixed_step_is_taken_when_n_steps_is_one``, so k = n_steps
        and k = n_steps - 1 are both accepted here; the exact flow is
        neither. Catches a method that does not reach ``odeint`` (dopri5 runs
        instead)."""
        from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

        t, n_steps = 0.5, 4
        score = LogDetFlowScore(_make_flow_ode(LinearVelocityField(_A3)),
                                t=t, n_steps=n_steps, method=method)
        y = _points(4, seed=3)
        s = score(y)
        I = torch.eye(3, dtype=torch.float64)
        errors = []
        for k in (n_steps - 1, n_steps):
            Z = -(t / k) * _A3
            S = I + Z if method == 'euler' else (
                I + Z + Z @ Z / 2 + Z @ Z @ Z / 6 + Z @ Z @ Z @ Z / 24)
            psi = y @ torch.linalg.matrix_power(S, k).T
            ref = 0.5 * (psi ** 2).sum(dim=1) + t * torch.trace(_A3)
            errors.append((s - ref).abs().max().item())
        assert min(errors) <= 1e-10, errors

    def test_time_dependent_field_matches_the_closed_form(self):
        """For v(s, x) = s A x at t = 0.5 the score is
        0.5 ||expm(-A t^2/2) y||^2 + tr(A) t^2/2. Catches a coupled ODE that
        hands the field the time 1 - s instead of s (the two agree only at
        t = 1)."""
        from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

        score = LogDetFlowScore(
            _make_flow_ode(TimeScaledLinearVelocityField(_A3)),
            t=0.5, n_steps=5, method='dopri5', atol=1e-12, rtol=1e-12)
        y = _points(4, seed=4)
        assert (score(y) - _closed_form(y, 0.125)).abs().max().item() <= 1e-9

    @pytest.mark.parametrize('n_steps', [
        pytest.param(1, marks=known_bug(
            13, "LogDetFlowScore builds its time grid with n_steps points, so "
                "a fixed-step solver takes n_steps - 1 steps although the "
                "docstring says n_steps is the exact number of steps; "
                "n_steps = 1 integrates nothing and the score is "
                "0.5 ||y||^2 for every field")),
        2,
    ])
    def test_one_fixed_step_is_taken_when_n_steps_is_one(self, n_steps):
        """The docstring: for fixed-step solvers ``n_steps`` "is the exact
        number of steps". On a constant field c every rk4 step is exact, so
        any step count >= 1 gives 0.5 ||y - c t||^2 (the Jacobian is zero).
        ``n_steps = 1`` asks for one step; today the grid has one point and
        no step is taken. ``n_steps = 2`` is the passing neighbour with the
        same reference."""
        from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

        c = torch.tensor([3.0, -1.0, 0.5], dtype=torch.float64)
        score = LogDetFlowScore(_make_flow_ode(ConstantVelocityField(c)),
                                t=0.5, n_steps=n_steps, method='rk4')
        y = _points(3, seed=5)
        expected = 0.5 * ((y - 0.5 * c) ** 2).sum(dim=1)
        assert (score(y) - expected).abs().max().item() <= 1e-12


class TestLogDetScoreOfAPointDoesNotDependOnItsBatch:
    """The score of a point alone equals its score inside a batch, to the
    integrator's tolerance, and each is the closed form of that point."""

    def test_chunks_and_single_points_match_the_closed_form(self):
        """``batch_size=4`` on 11 points (two full chunks and a partial one),
        dopri5 at 1e-12, t = 0.5: one score per point, each within 1e-9 of
        the closed form, and the score of a point alone too. Catches a chunk
        loop that scores the wrong slice or drops rows, and a score whose
        value for one point depends on the others beyond the tolerance."""
        from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

        score = LogDetFlowScore(_make_flow_ode(LinearVelocityField(_A3)),
                                t=0.5, n_steps=5, method='dopri5',
                                atol=1e-12, rtol=1e-12, batch_size=4)
        y = _points(11, seed=6)
        expected = _closed_form(y, 0.5)
        s = score(y)
        assert s.shape == (11,)
        assert (s - expected).abs().max().item() <= 1e-9
        for i in (0, 5, 10):
            assert abs(score(y[i:i + 1]).item() - expected[i].item()) <= 1e-9

    def test_set_membership_inside_no_grad_follows_the_closed_form(self):
        """The public route: ``ProbabilisticSet.contains`` with a
        ``LogDetFlowScore``, called inside ``torch.no_grad()`` as the set's
        volume, boundary and coverage routines call it. With t = 0.5 the
        score of y = expm(A t) z is 0.5 ||z||^2 + 0.125; threshold 1.125.
        ||z|| = 1.3 is inside (0.97), 1.45 and 1.5 are outside (1.176, 1.25);
        without the log-det term 1.45 would read as inside. Catches a trace
        that is silently 0 inside ``no_grad``, so the set queried there is
        not the set that was calibrated."""
        from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        score = LogDetFlowScore(_make_flow_ode(LinearVelocityField(_A3)),
                                t=0.5, n_steps=5, method='dopri5',
                                atol=1e-10, rtol=1e-10)
        pset = ProbabilisticSet(score_fn=score, threshold=1.125, m=2000,
                                ell=2000, epsilon=0.01, dim=3)
        d = _points(3, seed=7)
        d = d / d.norm(dim=1, keepdim=True)
        z = d * torch.tensor([[1.3], [1.45], [1.5]], dtype=torch.float64)
        y = z @ torch.linalg.matrix_exp(_A3 * 0.5).T
        with torch.no_grad():
            inside = pset.contains(y)
        assert inside.tolist() == [True, False, False]
