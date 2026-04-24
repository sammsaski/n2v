"""
ODE integration for the learned flow.

The OT-CFM training convention is:
  t=0: noise x0 ~ N(0,I)
  t=1: data x1
  velocity v(t, x) points from noise toward data

To map data -> latent (Gaussian), we integrate BACKWARDS from t -> 0.
"""

import torch
import torch.nn as nn
from typing import List

from torchdiffeq import odeint


class FlowODE(nn.Module):
    """
    Wraps a VelocityField for ODE integration via torchdiffeq.

    Args:
        velocity_field: VelocityField module.
    """

    def __init__(self, velocity_field):
        super().__init__()
        self.velocity_field = velocity_field

    def forward(
        self,
        y: torch.Tensor,
        t: float = 1.0,
        n_steps: int = 100,
        method: str = 'dopri5',
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> torch.Tensor:
        """
        Integrate the flow from t to 0 (reverse direction).

        Maps data points y toward the latent (Gaussian) space by
        reversing the learned noise->data flow.

        Args:
            y: (batch, dim) data points.
            t: Starting time (flow time at which y lives).
            n_steps: Number of integration steps. For adaptive solvers
                (dopri5) this is the number of output grid points; internal
                stepping is chosen by the solver. For fixed-step solvers
                (rk4, euler) this is the exact number of steps.
            method: torchdiffeq solver. 'dopri5' (default, adaptive,
                accurate) or 'rk4'/'euler' (fixed-step, fast inference).
            atol, rtol: absolute/relative tolerances. Ignored by fixed-step
                solvers.

        Returns:
            (batch, dim) latent points.
        """
        if t == 0.0:
            return y.clone()

        t_span = torch.linspace(t, 0, n_steps, device=y.device)

        def odefunc(t_val, y_val):
            t_batch = t_val.expand(y_val.shape[0])
            return self.velocity_field(t_batch, y_val)

        if method in ('rk4', 'euler'):
            trajectory = odeint(odefunc, y, t_span, method=method)
        else:
            trajectory = odeint(
                odefunc, y, t_span, method=method,
                atol=atol, rtol=rtol,
            )
        return trajectory[-1]

    def inverse(
        self,
        z: torch.Tensor,
        t: float = 1.0,
        n_steps: int = 100,
        method: str = 'dopri5',
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> torch.Tensor:
        """
        Integrate the flow from 0 to t (forward direction).

        Maps latent (Gaussian) points z to data points by following
        the learned noise->data flow.

        Args:
            z: (batch, dim) latent points.
            t: Endpoint of integration (flow time at which to stop).
            n_steps: Number of integration steps.
            method: torchdiffeq solver.
            atol, rtol: tolerances for adaptive solvers.

        Returns:
            (batch, dim) data points.
        """
        if t == 0.0:
            return z.clone()

        t_span = torch.linspace(0, t, n_steps, device=z.device)

        def odefunc(t_val, y_val):
            t_batch = t_val.expand(y_val.shape[0])
            return self.velocity_field(t_batch, y_val)

        if method in ('rk4', 'euler'):
            trajectory = odeint(odefunc, z, t_span, method=method)
        else:
            trajectory = odeint(
                odefunc, z, t_span, method=method,
                atol=atol, rtol=rtol,
            )
        return trajectory[-1]

    def forward_trajectory(
        self,
        y: torch.Tensor,
        t_values: List[float],
        n_steps: int = 100,
    ) -> torch.Tensor:
        """
        Compute ||phi_t(y)||_2 at multiple t values in one ODE solve.

        Integrates backwards from max(t_values) to 0, recording norms
        at each requested t value.

        Args:
            y: (batch, dim) data points.
            t_values: List of t values (must be sorted ascending, all > 0).
            n_steps: Number of integration steps.

        Returns:
            (batch, len(t_values)) tensor of norms.
        """
        # Integrate from max t down to 0, collecting at each t_value
        t_descending = sorted(t_values, reverse=True)
        t_span = torch.tensor(
            [t_descending[0]] + t_descending + [0.0], device=y.device
        )
        # Remove duplicates while preserving order
        t_span = torch.unique_consecutive(t_span)

        def odefunc(t_val, y_val):
            t_batch = t_val.expand(y_val.shape[0])
            return self.velocity_field(t_batch, y_val)

        trajectory = odeint(
            odefunc, y, t_span, method='dopri5',
            atol=1e-5, rtol=1e-5,
        )

        # trajectory: (len(t_span), batch, dim)
        # Map each requested t_value to its position in t_span
        t_span_list = t_span.tolist()
        norms = []
        for tv in t_values:
            # Find the index in t_span closest to 0 when starting from tv
            # The norm at time tv is the state when ODE reaches t=0
            # But we need the norm at the *latent* point reached by
            # integrating from tv to 0. For the full trajectory from
            # max_t to 0, the state at t=0 is trajectory[-1].
            # For intermediate t_values, we need separate solves.
            pass

        # Simpler approach: separate forward() call per t_value
        norms = []
        for tv in t_values:
            with torch.no_grad():
                z = self.forward(y, t=tv, n_steps=n_steps)
                norms.append(z.norm(dim=1))

        return torch.stack(norms, dim=1)
