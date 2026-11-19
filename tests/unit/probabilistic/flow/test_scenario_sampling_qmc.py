"""Tests for QMC scenario sampling.

QMC (Sobol-based) sampling reduces estimator variance for the scenario
disjointness check. Validity is preserved because samples are
equidistributed on N(0, I) — same marginal distribution as i.i.d.
Gaussian, just lower variance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from n2v.probabilistic.flow.scenario_verify import (
    _qmc_sample_latents,
    certify_halfspace_disjoint,
)
from n2v.sets.halfspace import HalfSpace

from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow


# -------- _qmc_sample_latents helper tests --------


def test_qmc_shape_and_dtype():
    """Helper returns the requested shape and the requested dtype."""
    z = _qmc_sample_latents(100, 5, seed=0)
    assert z.shape == (100, 5)
    assert z.dtype == torch.float32


def test_qmc_dtype_override():
    """Helper respects an explicit dtype override."""
    z = _qmc_sample_latents(64, 3, seed=0, dtype=torch.float64)
    assert z.dtype == torch.float64
    assert z.shape == (64, 3)


def test_qmc_marginal_close_to_standard_normal():
    """Sobol+norm.ppf gives samples whose marginals are close to N(0, 1).

    Per-dim mean ~ 0 and per-dim std ~ 1 within reasonable tolerance
    for N=2000.
    """
    z = _qmc_sample_latents(2000, 5, seed=0)
    z_np = z.numpy()
    # Per-dim mean should be close to 0
    assert np.all(np.abs(z_np.mean(axis=0)) < 0.05), (
        f'mean drift: {z_np.mean(axis=0)}'
    )
    # Per-dim std should be close to 1
    assert np.all(np.abs(z_np.std(axis=0) - 1.0) < 0.1), (
        f'std drift: {z_np.std(axis=0)}'
    )


def test_qmc_lower_variance_than_uniform_for_smooth_integral():
    """QMC estimator of E[f(z)] for f smooth has lower variance than i.i.d.

    Use f(z) = sum(z**2). Truth = dim. Compare empirical variance of the
    estimator across n_trials trials of N=200 each.
    """
    dim = 5
    n_samples = 200
    n_trials = 50

    qmc_estimates = []
    iid_estimates = []
    for trial in range(n_trials):
        # Uniform i.i.d.
        torch.manual_seed(trial)
        z_iid = torch.randn(n_samples, dim)
        iid_estimates.append((z_iid ** 2).sum(dim=1).mean().item())
        # QMC
        z_qmc = _qmc_sample_latents(n_samples, dim, seed=trial)
        qmc_estimates.append((z_qmc ** 2).sum(dim=1).mean().item())

    iid_var = float(np.var(iid_estimates))
    qmc_var = float(np.var(qmc_estimates))
    # QMC should have noticeably lower variance for smooth integrand.
    # We don't require a specific factor, just lower.
    assert qmc_var < iid_var, (
        f'QMC variance {qmc_var} not less than i.i.d. variance {iid_var}'
    )


def test_qmc_finite_no_inf():
    """norm.ppf clipping must guarantee no inf/-inf output even at large N."""
    z = _qmc_sample_latents(4096, 8, seed=0)
    assert torch.isfinite(z).all()


# -------- Dispatcher integration tests --------


@pytest.mark.slow
def test_certify_halfspace_disjoint_qmc_default_unchanged():
    """Default sampling_strategy must remain 'uniform' (bit-for-bit unchanged
    behavior on the existing call signature)."""
    flow = _train_small_2d_flow(seed=0)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    # Call without sampling_strategy: must use 'uniform' under the hood
    # and return a sensible disjoint result for an unreachable polyhedron.
    res = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=200, beta_2=0.001, seed=0,
    )
    assert res.disjoint is True


@pytest.mark.slow
def test_certify_halfspace_disjoint_qmc_strategy_runs():
    """sampling_strategy='qmc' produces a valid disjoint result on a clearly
    unreachable polyhedron."""
    flow = _train_small_2d_flow(seed=0)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    res_qmc = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=200, beta_2=0.001, seed=0,
        sampling_strategy='qmc',
    )
    res_uniform = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=200, beta_2=0.001, seed=0,
        sampling_strategy='uniform',
    )
    # Both should certify disjoint (loose unsafe halfspace far from reach set)
    assert res_qmc.disjoint is True
    assert res_uniform.disjoint is True
    # Same epsilon_2 (depends only on N, beta_2)
    assert res_qmc.epsilon_2 == pytest.approx(
        math.log(1.0 / 0.001) / 200, rel=1e-6
    )


@pytest.mark.slow
def test_certify_halfspace_disjoint_invalid_sampling_strategy_raises():
    """Unknown sampling_strategy must raise ValueError with a helpful message."""
    flow = _train_small_2d_flow(seed=0)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    with pytest.raises(ValueError, match='sampling_strategy'):
        certify_halfspace_disjoint(
            flow_ode=flow, threshold_q=3.0, halfspace=hs,
            n_samples=100, beta_2=0.001, seed=0,
            sampling_strategy='not_a_strategy',
        )


@pytest.mark.slow
def test_certify_group_disjoint_threads_qmc_kwarg():
    """certify_group_disjoint accepts and forwards sampling_strategy='qmc'."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    flow = _train_small_2d_flow(seed=0)
    group = [HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))]
    res = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=200, beta_2=0.001, seed=0,
        sampling_strategy='qmc',
    )
    assert res.disjoint is True


@pytest.mark.slow
def test_certify_spec_disjoint_threads_qmc_kwarg():
    """certify_spec_disjoint accepts and forwards sampling_strategy='qmc'."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    flow = _train_small_2d_flow(seed=0)
    spec_groups = [[HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))]]
    res = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=spec_groups,
        n_samples=200, beta_2=0.001, seed=0,
        sampling_strategy='qmc',
    )
    assert res.unsat_certified is True


# -------- Seed and antithetic pairs --------
# The law of the QMC latents (one Sobol point per stratum of every coordinate
# after the inverse normal CDF) is checked in
# tests/soundness/test_soundness_scenario_verify.py.


def test_qmc_same_seed_gives_the_same_points():
    """The seed fixes the scrambling: two calls with the same seed return the
    same points bit for bit, another seed other points. Catches a seed that
    is not handed to the Sobol sampler (scrambling from system entropy)."""
    a = _qmc_sample_latents(64, 3, seed=9, dtype=torch.float64)
    b = _qmc_sample_latents(64, 3, seed=9, dtype=torch.float64)
    c = _qmc_sample_latents(64, 3, seed=10, dtype=torch.float64)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("n", [64, 63, 1])
def test_qmc_antithetic_pairs_each_point_with_its_mirror(n):
    """With antithetic=True the first ceil(n/2) rows are the Sobol points of
    the same seed and the next rows their negatives, in order; for odd n the
    last base point has no mirror. Catches the points repeated in place of
    mirrored, and a base count of floor(n/2) (one sample short for odd n)."""
    n_base = (n + 1) // 2
    z = _qmc_sample_latents(n, 3, seed=2, dtype=torch.float64, antithetic=True)
    base = _qmc_sample_latents(n_base, 3, seed=2, dtype=torch.float64)
    assert z.shape == (n, 3)
    assert torch.equal(z[:n_base], base)
    assert torch.equal(z[n_base:], -base[:n - n_base])


def test_qmc_antithetic_certifier_draws_exactly_n_samples():
    """certify_halfspace_disjoint with 'qmc+antithetic' hands exactly N latent
    points to the flow for odd N as well, and states ln(1/beta)/N."""
    class _Recording:
        def __init__(self):
            self.batches = []

        def inverse(self, z, **kwargs):
            self.batches.append(int(z.shape[0]))
            return z.detach().to(torch.float64).clone()

    flow = _Recording()
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-50.0]]))
    res = certify_halfspace_disjoint(flow, 1.0, hs, n_samples=101, beta_2=0.01, seed=0,
                                     sampling_strategy='qmc+antithetic')
    assert flow.batches == [101]
    assert res.n_samples_used == 101
    assert res.epsilon_2 == pytest.approx(math.log(100.0) / 101, rel=1e-15)
