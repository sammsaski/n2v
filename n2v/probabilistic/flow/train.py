"""
OT-CFM training loop for flow matching.

Trains a velocity field to transport data toward a standard Gaussian.
Pairs (x0 ~ N(0,I), x1 = data), interpolates x_t = (1-t)*x0 + t*x1,
and regresses v_theta(t, x_t) against the target velocity x1 - x0.

Supports two OT coupling methods:
  - Hungarian (exact, O(n^3), CPU-only via scipy)
  - Sinkhorn (approximate, GPU-friendly, pure tensor ops)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List

from n2v.probabilistic.flow.model import VelocityField


def ot_coupling(
    x0: torch.Tensor, x1: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minibatch OT coupling via the Hungarian algorithm.

    Finds the permutation of (x0, x1) that minimizes total L2 cost.
    Requires CPU (uses scipy). O(n^3) per batch.

    Args:
        x0: (batch, dim) source samples.
        x1: (batch, dim) target samples.

    Returns:
        (x0_permuted, x1_permuted) with optimal coupling.
    """
    from scipy.optimize import linear_sum_assignment

    cost = torch.cdist(x0, x1, p=2).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return x0[row_ind], x1[col_ind]


def sinkhorn_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
    reg: float = 0.05,
    max_iters: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minibatch OT coupling via the Sinkhorn algorithm.

    Approximates optimal transport via entropic regularization.
    Pure tensor ops — works on CPU and GPU. O(n^2) per iteration.

    Args:
        x0: (batch, dim) source samples.
        x1: (batch, dim) target samples.
        reg: Entropic regularization strength (smaller = closer to exact OT).
        max_iters: Number of Sinkhorn iterations.

    Returns:
        (x0_permuted, x1_permuted) with approximate optimal coupling.
    """
    with torch.no_grad():
        cost = torch.cdist(x0, x1, p=2)
        K = torch.exp(-cost / reg)

        # Sinkhorn iterations (row/column normalization)
        u = torch.ones(x0.shape[0], device=x0.device)
        for _ in range(max_iters):
            v = 1.0 / (K.T @ u + 1e-8)
            u = 1.0 / (K @ v + 1e-8)

        # Transport plan
        plan = torch.diag(u) @ K @ torch.diag(v)

        # Extract hard assignment via argmax per row
        col_ind = plan.argmax(dim=1)
        row_ind = torch.arange(x0.shape[0], device=x0.device)

    return x0[row_ind], x1[col_ind]


def train_flow(
    velocity_field: VelocityField,
    training_outputs: torch.Tensor,
    n_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    coupling: str = 'hungarian',
) -> Tuple[VelocityField, List[float]]:
    """
    Train the velocity field using OT-CFM.

    Args:
        velocity_field: VelocityField module to train.
        training_outputs: (n, d) tensor of training data points.
        n_epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        coupling: OT coupling method. One of:
            'none' — random pairing (no OT)
            'hungarian' — exact OT via Hungarian algorithm (CPU-only)
            'sinkhorn' — approximate OT via Sinkhorn (GPU-friendly)

    Returns:
        (velocity_field, losses) — the trained model and per-epoch losses.

    Raises:
        ValueError: If coupling is not one of the valid options.
    """
    valid_couplings = ('none', 'hungarian', 'sinkhorn')
    if coupling not in valid_couplings:
        raise ValueError(
            f"coupling must be one of {valid_couplings}, got '{coupling}'"
        )

    optimizer = torch.optim.Adam(velocity_field.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    dataset = torch.utils.data.TensorDataset(training_outputs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (x1_batch,) in loader:
            # Sample source noise
            x0_batch = torch.randn_like(x1_batch)

            # OT coupling
            if coupling == 'hungarian':
                x0_batch, x1_batch = ot_coupling(x0_batch, x1_batch)
            elif coupling == 'sinkhorn':
                x0_batch, x1_batch = sinkhorn_coupling(x0_batch, x1_batch)

            # Sample time uniformly
            t = torch.rand(x1_batch.shape[0], device=x1_batch.device)

            # Interpolate
            x_t = (1 - t.unsqueeze(1)) * x0_batch + t.unsqueeze(1) * x1_batch

            # Target velocity
            target_v = x1_batch - x0_batch

            # Predicted velocity
            pred_v = velocity_field(t, x_t)

            loss = F.mse_loss(pred_v, target_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        losses.append(epoch_loss / max(n_batches, 1))

    return velocity_field, losses
