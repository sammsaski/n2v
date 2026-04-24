"""Input-set sampling utilities for probabilistic reachability.

Currently hosts the L_inf-ball uniform sampler that every benchmark
in this project uses as its input distribution ``P_X``. Previously
lived inside an experiment script under
``examples/FlowConformal/experiments/hashemi_comparison/v2_adaptive_reg/``;
promoted to library code because every probabilistic-reach harness
depends on it.
"""

from __future__ import annotations

import torch


def sample_l_inf_ball(
    x_center: torch.Tensor,
    radius: float,
    n_samples: int,
    seed: int,
    dim: int,
) -> torch.Tensor:
    """Uniform sample from the L-infinity ball ``B(x_center, radius)``.

    Draws ``n_samples`` points from the uniform distribution on
    ``{x : ||x - x_center||_inf <= radius}`` using a seeded RNG.
    Each dimension is drawn independently from ``Uniform([-radius, +radius])``
    and added to ``x_center`` — no rejection sampling, no sphere
    parametrization. This is the standard uniform-on-L_inf-ball
    factorization used throughout the NN verification literature.

    Args:
        x_center: ``(dim,)`` tensor giving the ball center. Any
            ``torch.Tensor`` of shape ``(dim,)`` is accepted and is
            broadcast across samples.
        radius: Scalar half-width of the ball (the "epsilon" in
            ``B(x_center, epsilon)``).
        n_samples: Number of points to draw.
        seed: Integer seed for the internal ``torch.Generator`` so the
            sample is deterministic across runs.
        dim: Dimensionality of the input space. Must match
            ``x_center.shape[0]``.

    Returns:
        ``(n_samples, dim)`` float32 tensor. Same dtype/device as
        ``x_center`` via broadcasting.
    """
    gen = torch.Generator().manual_seed(seed)
    perturbations = (
        torch.rand(n_samples, dim, generator=gen) * 2 - 1
    ) * radius
    return x_center + perturbations


def sample_box(
    lb: torch.Tensor,
    ub: torch.Tensor,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    """Uniform sample from the axis-aligned box ``[lb, ub]``.

    Generalizes :func:`sample_l_inf_ball` to asymmetric boxes, which is
    what VNN-LIB input specifications produce. Each dimension is drawn
    independently from ``Uniform([lb[k], ub[k]])``.

    The RNG construction matches :func:`sample_l_inf_ball` (fresh
    ``torch.Generator`` seeded with ``seed``, one ``torch.rand`` call for
    the whole batch) so the two functions agree to floating-point
    tolerance on the symmetric-box case.

    Args:
        lb: ``(dim,)`` tensor. Lower bound per dimension.
        ub: ``(dim,)`` tensor. Upper bound per dimension. Must satisfy
            ``lb <= ub`` componentwise.
        n_samples: Number of points to draw.
        seed: Integer RNG seed.

    Returns:
        ``(n_samples, dim)`` tensor. Same dtype/device as ``lb``.

    Raises:
        ValueError: if ``lb.shape != ub.shape`` or any ``lb[i] > ub[i]``.
    """
    if lb.shape != ub.shape:
        raise ValueError(
            f"lb.shape {tuple(lb.shape)} != ub.shape {tuple(ub.shape)}"
        )
    if (lb > ub).any():
        raise ValueError("lb must be <= ub componentwise")

    dim = lb.shape[0]
    gen = torch.Generator().manual_seed(seed)
    # Uniform in [-1, 1] then rescaled so we match sample_l_inf_ball's RNG
    # sequence on the symmetric case (center=(lb+ub)/2, radius=(ub-lb)/2).
    center = (lb + ub) * 0.5
    half_width = (ub - lb) * 0.5
    perturbations = (
        torch.rand(n_samples, dim, generator=gen) * 2 - 1
    ) * half_width
    return center + perturbations
