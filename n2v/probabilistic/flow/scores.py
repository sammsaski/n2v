"""
Nonconformity score functions for flow-based conformal reachability.

Each score maps a batch of output vectors to non-negative scalars.
The sublevel set {y : score(y) <= q} defines the prediction region.
"""

import torch
from torch import Tensor


class NonconformityScore:
    """Base class for nonconformity score functions.

    All scores: (batch, d) -> (batch,) non-negative scalars.
    """

    def __call__(self, y: Tensor) -> Tensor:
        raise NotImplementedError


class HyperrectScore(NonconformityScore):
    """Hyperrectangular score: max_k |y_k - c_k| / tau_k.

    The sublevel set is an axis-aligned box.
    This is the naive score from Hashemi et al. 2025.

    Args:
        center: (d,) center of the score function.
        scales: (d,) per-dimension normalization factors.
    """

    def __init__(self, center: Tensor, scales: Tensor):
        self.center = center
        self.scales = scales

    def __call__(self, y: Tensor) -> Tensor:
        return ((y - self.center).abs() / self.scales).max(dim=1).values

    def sublevel_set_volume(self, q: Tensor) -> float:
        """Closed-form volume of {y : score(y) <= q}."""
        return (2 * q * self.scales).prod().item()


class EllipsoidScore(NonconformityScore):
    """Mahalanobis score: sqrt((y-c)^T Sigma^{-1} (y-c)).

    The sublevel set is an ellipsoid.

    Args:
        center: (d,) center.
        cov_inv: (d, d) inverse covariance matrix.
    """

    def __init__(self, center: Tensor, cov_inv: Tensor):
        self.center = center
        self.cov_inv = cov_inv

    def __call__(self, y: Tensor) -> Tensor:
        diff = y - self.center
        return (diff @ self.cov_inv * diff).sum(dim=1).sqrt()


class BallScore(NonconformityScore):
    """L2 ball score: ||y - c||_2.

    The sublevel set is a Euclidean ball.

    Args:
        center: (d,) center.
    """

    def __init__(self, center: Tensor):
        self.center = center

    def __call__(self, y: Tensor) -> Tensor:
        return (y - self.center).norm(dim=1)


class FlowScore(NonconformityScore):
    """Flow-based score: ||phi_t(y)||_2.

    The sublevel set follows the geometry of the learned flow.

    Args:
        flow_model: Object with forward(y, t, ...) method (e.g., FlowODE).
        t: Flow time parameter (default 1.0).
        n_steps: Number of ODE integration steps.
        method: ODE solver name passed through to flow_model.forward.
            'dopri5' for adaptive (default), 'rk4'/'euler' for fast
            fixed-step inference.
        batch_size: if not None, chunk the incoming y into this size and
            concatenate — lets callers evaluate very large batches (e.g.
            MC volume) without OOM.
    """

    def __init__(self, flow_model, t: float = 1.0, n_steps: int = 100,
                 method: str = 'dopri5', batch_size: int | None = None,
                 atol: float = 1e-5, rtol: float = 1e-5):
        self.flow_model = flow_model
        self.t = t
        self.n_steps = n_steps
        self.method = method
        self.batch_size = batch_size
        self.atol = atol
        self.rtol = rtol

    def _flow_device(self) -> torch.device | None:
        """Return the device of the underlying velocity field, if any."""
        vf = getattr(self.flow_model, 'velocity_field', self.flow_model)
        try:
            return next(vf.parameters()).device
        except (StopIteration, AttributeError):
            return None

    def _integrate(self, y: Tensor) -> Tensor:
        return self.flow_model.forward(
            y, t=self.t, n_steps=self.n_steps, method=self.method,
            atol=self.atol, rtol=self.rtol,
        )

    def __call__(self, y: Tensor) -> Tensor:
        dev = self._flow_device()
        src_device = y.device
        if dev is not None and y.device != dev:
            y = y.to(dev)
        if self.batch_size is None or y.shape[0] <= self.batch_size:
            out = self._integrate(y).norm(dim=1)
        else:
            outs = []
            for i in range(0, y.shape[0], self.batch_size):
                outs.append(self._integrate(y[i:i + self.batch_size]).norm(dim=1))
            out = torch.cat(outs, dim=0)
        if dev is not None and out.device != src_device:
            out = out.to(src_device)
        return out

    def set_t(self, t: float):
        """Update the flow time parameter."""
        self.t = t
