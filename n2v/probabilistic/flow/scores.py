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
        flow_model: Object with forward(y, t) method (e.g., FlowODE).
        t: Flow time parameter (default 1.0).
    """

    def __init__(self, flow_model, t: float = 1.0):
        self.flow_model = flow_model
        self.t = t

    def __call__(self, y: Tensor) -> Tensor:
        z = self.flow_model.forward(y, t=self.t)
        return z.norm(dim=1)

    def set_t(self, t: float):
        """Update the flow time parameter."""
        self.t = t
