"""Helpers for Phase 0A Gaussian-benchmark flow-matching tests.

Each helper pairs a sampler for a Gaussian target with an analytical
reference velocity field derived from the affine conditional flow in
Lipman et al. 2024, eq. (4.52). These are the ground-truth objects the
trained flow should match.
"""

import math
import torch

from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.train import train_flow


def sample_gaussian(n, mean, cov, seed):
    """Draw n IID samples from N(mean, cov)."""
    gen = torch.Generator().manual_seed(seed)
    d = mean.shape[0]
    L = torch.linalg.cholesky(cov)
    z = torch.randn(n, d, generator=gen)
    return mean.unsqueeze(0) + z @ L.T


def sym_psd_sqrt(cov):
    """Symmetric PSD square root of a symmetric PSD matrix via eigendecomposition.

    For Gaussian-to-Gaussian OT between N(0, I) and N(b, cov), the OT-optimal
    map is T(z) = cov^{1/2} z + b, where cov^{1/2} is the *symmetric* PSD root
    (not an arbitrary Cholesky or R @ diag(sigma) factorization). Passing a
    non-symmetric factorization to affine_optimal_velocity gives a valid
    factorization of cov but NOT the OT velocity — the learned flow (trained
    with OT coupling) will approach cov^{1/2}, not the asymmetric factor.
    """
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=0.0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


def affine_optimal_velocity(x, t, A, b):
    """Analytical marginal velocity for target X_1 = A Z + b, X_0 = Z, Z ~ N(0,I).

    The affine conditional OT flow at time t is
        X_t = (1-t) X_0 + t (A X_0 + b) = ((1-t) I + t A) X_0 + t b.
    Its constant-along-trajectory velocity is X_1 - X_0 = (A - I) X_0 + b.
    The marginal velocity at position x is obtained by inverting the
    affine map:
        X_0 = M_t^{-1} (x - t b), where M_t = (1-t) I + t A,
    and substituting:
        u(t, x) = (A - I) M_t^{-1} (x - t b) + b.
    """
    d = x.shape[1]
    I = torch.eye(d, dtype=x.dtype, device=x.device)
    Mt = (1.0 - t) * I + t * A
    x_shift = x - t * b.unsqueeze(0)
    # Solve M_t x0^T = x_shift^T  <=>  x0 M_t^T = x_shift (row-vector form).
    x0 = torch.linalg.solve(Mt, x_shift.T).T
    return x0 @ (A - I).T + b.unsqueeze(0)


def train_small_flow(data, dim, n_epochs=300, batch_size=256, seed=0,
                     coupling='hungarian', use_ema=False,
                     standardize_outputs=False):
    """Small-network OT-CFM training used by Phase 0A tests.

    Defaults picked for speed — each test completes in <30s on CPU.
    """
    torch.manual_seed(seed)
    vf = VelocityField(dim=dim, hidden=64, n_layers=3, activation='silu')
    vf, _ = train_flow(
        vf,
        data,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=1e-3,
        coupling=coupling,
        use_ema=use_ema,
        standardize_outputs=standardize_outputs,
    )
    vf.eval()
    return FlowODE(vf)


def grid_points(d, n_per_side=10, range_=2.0, seed=0):
    """Return (n_per_side**2, d) sample points in [-range_, range_]^d.

    Uniformly random for simplicity; the name 'grid' is historical — in
    d>=2 a full tensor grid would blow up, so we sample instead. For d=1
    we use a deterministic linspace of n_per_side**2 points so the
    behavior is consistent across dimensions.
    """
    n = n_per_side ** 2
    if d == 1:
        lin = torch.linspace(-range_, range_, n)
        return lin.unsqueeze(1)
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(n, d, generator=gen) * 2 - 1) * range_


def pointwise_velocity_mse(flow_ode, analytical_fn, dim, n_t=5, n_x=100, seed=0):
    """Mean squared error between learned v_theta(t, x) and analytical v*(t, x)
    sampled at n_t time points x n_x spatial points.
    """
    torch.manual_seed(seed)
    vf = flow_ode.velocity_field
    ts = torch.linspace(0.1, 0.9, n_t)
    xs = grid_points(dim, n_per_side=int(math.sqrt(n_x)), seed=seed)
    sq = 0.0
    count = 0
    with torch.no_grad():
        for t_val in ts:
            t_batch = torch.full((xs.shape[0],), t_val.item())
            v_pred = vf(t_batch, xs)
            v_true = analytical_fn(xs, t_val.item())
            sq += (v_pred - v_true).pow(2).sum().item()
            count += xs.numel()
    return sq / count


def round_trip_error(flow_ode, y, n_steps=200):
    """||y - phi^{-1}(phi(y))||_2 mean over y batch.
    Backward-then-forward integration should recover y within solver tolerance.
    """
    with torch.no_grad():
        z = flow_ode.forward(y, t=1.0, n_steps=n_steps)
        y_back = flow_ode.inverse(z, t=1.0, n_steps=n_steps)
    return (y - y_back).norm(dim=1).mean().item()
