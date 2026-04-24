"""
Log-det-Jacobian-corrected flow nonconformity score (Experiment E1).

This module is *additive* and intentionally lives outside the production
scores/ode modules. It is imported directly from experiment scripts:

    from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore

Rationale: the existing `FlowScore` uses only ||phi(y)||_2, which is a
proxy for the density learned by flow matching. Under the instantaneous
change-of-variables formula, the actual negative log-density is

    -log p_theta(y) = (d/2) log(2*pi)
                    + (1/2) ||phi(y)||^2
                    + integral_0^1 trace(dv/dx(s, psi(s))) ds

where psi(t) is the trajectory satisfying psi(1) = y and
dpsi/dt = v(t, psi(t)). The integral is the log-det-Jacobian correction
that `FlowScore` omits.

`LogDetFlowScore` integrates the coupled ODE
    dpsi/dt = v(t, psi(t)),        psi(1) = y
    dL/dt   = trace(dv/dx(t, psi(t))),   L(1) = 0
backward from t=1 to t=0 and returns

    s_logdet(y) = (1/2) ||psi(0)||^2 - L(0)

which equals the integral-form expression above (see the module-level
docstring comment: backward integration of dL/dt with L(1)=0 gives
L(0) = -integral_0^1 trace(dv/dx) ds, so -L(0) is the integral term).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchdiffeq import odeint


def exact_trace(v_func, t_batch: Tensor, x: Tensor) -> Tensor:
    """trace(dv/dx) via d autograd calls; exact, not Hutchinson.

    Args:
        v_func: Callable taking (t_batch, x) -> (batch, d) velocity.
        t_batch: (batch,) time values passed to `v_func`.
        x: (batch, d) positions. A fresh leaf tensor is created internally
            so this is safe to call inside an ODE solver callback even if
            the caller's `x` already has requires_grad=True.

    Returns:
        (batch,) exact trace of the Jacobian dv/dx evaluated at each
        (t_batch[i], x[i]).
    """
    d = x.shape[1]
    # Detach and re-require grad so we own the autograd graph here.
    x_req = x.detach().requires_grad_(True)

    # Enable grad even if caller wrapped us in torch.no_grad().
    with torch.enable_grad():
        v = v_func(t_batch, x_req)  # (batch, d)
        trace = torch.zeros(x_req.shape[0], device=x_req.device, dtype=x_req.dtype)
        # If v does not depend on x at all (e.g. a constant field), autograd
        # will report no grad_fn. In that case the trace is identically zero.
        if not v.requires_grad:
            return trace
        for i in range(d):
            grad_i = torch.autograd.grad(
                v[:, i].sum(),
                x_req,
                create_graph=False,
                retain_graph=(i < d - 1),
                allow_unused=True,
            )[0]
            if grad_i is None:
                continue
            trace = trace + grad_i[:, i]
    return trace


class LogDetFlowScore:
    """Nonconformity score with log-det-Jacobian correction.

    Score formula:
        s(y) = (1/2) ||phi(y)||^2 + integral_0^1 trace(dv/dx(s, psi(s))) ds

    which is monotone in -log p_theta(y) under the change-of-variables
    formula. Compared to `FlowScore` (which uses only ||phi(y)||_2), this
    includes the learned density's Jacobian term. Hypothesized to reduce
    out-of-support bleed that the proxy score suffers from.
    """

    def __init__(
        self,
        flow_model,
        t: float = 1.0,
        n_steps: int = 30,
        method: str = 'dopri5',
        atol: float = 1e-5,
        rtol: float = 1e-5,
        batch_size: 'int | None' = None,
    ):
        """
        Args:
            flow_model: A FlowODE-like instance exposing ``.velocity_field``.
            t: Starting time of the backward integration (default 1.0).
            n_steps: Number of ODE grid points (default 30). For adaptive
                solvers this is the output-time resolution; internal stepping
                is chosen adaptively. For fixed-step solvers (rk4, euler)
                this is the exact number of steps.
            method: torchdiffeq solver name. ``'dopri5'`` (default, adaptive,
                accurate) or ``'rk4'`` / ``'euler'`` (fixed-step, fast).
            atol, rtol: absolute / relative tolerances. Ignored by fixed-step
                solvers.
            batch_size: if not None, chunk the input into this size, evaluate
                each chunk independently, and concatenate the scores. Bounds
                peak autograd-graph memory during MC volume estimation on
                400k+ points. Matches the semantics of FlowScore.batch_size.
        """
        self.flow_model = flow_model
        self.t = float(t)
        self.n_steps = int(n_steps)
        self.method = str(method)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.batch_size = batch_size

    def set_t(self, t: float):
        """Update the starting time parameter. Mirrors FlowScore.set_t."""
        self.t = float(t)

    def _flow_device(self):
        """Return the device of the underlying velocity field, if any."""
        vf = getattr(self.flow_model, 'velocity_field', self.flow_model)
        try:
            return next(vf.parameters()).device
        except (StopIteration, AttributeError):
            return None

    def _score_chunk(self, y: Tensor) -> Tensor:
        """Score a single chunk of points; no device movement or batching here.
        Assumes y is already on the flow's device."""
        if self.t == 0.0:
            return 0.5 * (y ** 2).sum(dim=1)

        batch = y.shape[0]
        device = y.device
        dtype = y.dtype

        v_field = self.flow_model.velocity_field

        def odefunc(t_val: Tensor, state):
            psi, _L = state
            t_batch = t_val.expand(psi.shape[0]).to(psi.dtype)
            with torch.no_grad():
                d_psi = v_field(t_batch, psi)
            d_L = exact_trace(v_field, t_batch, psi)
            return d_psi, d_L

        psi0 = y
        L0 = torch.zeros(batch, device=device, dtype=dtype)

        t_span = torch.linspace(self.t, 0.0, self.n_steps, device=device, dtype=dtype)

        if self.method in ('rk4', 'euler'):
            trajectory = odeint(odefunc, (psi0, L0), t_span, method=self.method)
        else:
            trajectory = odeint(
                odefunc, (psi0, L0), t_span,
                method=self.method, atol=self.atol, rtol=self.rtol,
            )
        psi_final = trajectory[0][-1]
        L_final = trajectory[1][-1]
        return 0.5 * (psi_final ** 2).sum(dim=1) - L_final

    def __call__(self, y: Tensor) -> Tensor:
        """
        Args:
            y: (batch, d) tensor of points to score.
        Returns:
            (batch,) tensor of scores, same dtype/device as the input ``y``.
        """
        src_device = y.device
        dev = self._flow_device()
        if dev is not None and y.device != dev:
            y = y.to(dev)

        bs = self.batch_size
        if bs is None or y.shape[0] <= bs:
            out = self._score_chunk(y)
        else:
            chunks = [
                self._score_chunk(y[i:i + bs])
                for i in range(0, y.shape[0], bs)
            ]
            out = torch.cat(chunks, dim=0)

        if out.device != src_device:
            out = out.to(src_device)
        return out
