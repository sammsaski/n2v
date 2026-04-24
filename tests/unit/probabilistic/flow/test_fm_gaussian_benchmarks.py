"""Phase 0A: Closed-form Gaussian benchmarks for flow matching.

For each target, we verify three things against the analytical flow:
1. Pointwise velocity MSE is small.
2. Round-trip y -> phi(y) -> phi^{-1}(phi(y)) recovers y.
3. Sample quality: mean and covariance of generated samples match target.
"""

import math
import pytest
import torch

from tests.unit.probabilistic.flow._fm_gaussian_helpers import (
    affine_optimal_velocity,
    pointwise_velocity_mse,
    round_trip_error,
    sample_gaussian,
    sym_psd_sqrt,
    train_small_flow,
)


@pytest.mark.slow
class TestIdentityTarget:
    """Target = N(0, I). Optimal velocity is zero everywhere."""

    def test_pointwise_velocity_near_zero(self):
        torch.manual_seed(0)
        d = 2
        data = torch.randn(4000, d)  # already N(0, I)
        flow = train_small_flow(data, dim=d, n_epochs=200, seed=0)
        I = torch.eye(d)
        zero_b = torch.zeros(d)

        def analytical(x, t):
            return affine_optimal_velocity(x, t, I, zero_b)

        mse = pointwise_velocity_mse(flow, analytical, dim=d)
        assert mse < 0.05, f"velocity MSE {mse} too high for identity target"

    def test_round_trip_recovers_data(self):
        torch.manual_seed(0)
        d = 2
        data = torch.randn(4000, d)
        flow = train_small_flow(data, dim=d, n_epochs=200, seed=0)
        y = torch.randn(64, d)
        err = round_trip_error(flow, y)
        assert err < 5e-3, f"round-trip error {err} too high"


@pytest.mark.slow
class TestShiftTarget:
    """Target = N(mu, I). Optimal velocity is constant: v*(t, x) = mu."""

    def test_pointwise_velocity_matches_mu(self):
        torch.manual_seed(0)
        d = 2
        mu = torch.tensor([3.0, -2.0])
        data = sample_gaussian(4000, mu, torch.eye(d), seed=0)
        flow = train_small_flow(data, dim=d, n_epochs=300, seed=0)
        I = torch.eye(d)

        def analytical(x, t):
            return affine_optimal_velocity(x, t, I, mu)

        mse = pointwise_velocity_mse(flow, analytical, dim=d)
        # Reference ||mu||^2 = 13; MSE ~0.3 corresponds to ~2% relative error,
        # which is the floor for a 300-epoch small-net training.
        assert mse < 0.5, f"velocity MSE {mse} too high for shift target"

    def test_samples_match_mean(self):
        torch.manual_seed(0)
        d = 2
        mu = torch.tensor([3.0, -2.0])
        data = sample_gaussian(4000, mu, torch.eye(d), seed=0)
        flow = train_small_flow(data, dim=d, n_epochs=300, seed=0)

        z = torch.randn(2000, d)
        with torch.no_grad():
            y_gen = flow.inverse(z, t=1.0, n_steps=100)
        empirical_mean = y_gen.mean(dim=0)
        assert (empirical_mean - mu).norm() < 0.2, (
            f"generated mean {empirical_mean} far from target {mu}"
        )


@pytest.mark.slow
class TestIsotropicScaleTarget:
    """Target = N(0, sigma^2 I). Optimal map is T(z) = sigma*z, so A = sigma*I.

    Tests that OT coupling correctly pairs nearby source/target samples and
    that the flow learns t-dependent velocity (constant across t in v*, but
    position-dependent)."""

    def test_pointwise_velocity_and_samples(self):
        torch.manual_seed(0)
        d = 2
        sigma = 2.0
        cov = (sigma ** 2) * torch.eye(d)
        data = sample_gaussian(4000, torch.zeros(d), cov, seed=0)
        flow = train_small_flow(data, dim=d, n_epochs=400, seed=0)
        A = sigma * torch.eye(d)
        b = torch.zeros(d)

        def analytical(x, t):
            return affine_optimal_velocity(x, t, A, b)

        mse = pointwise_velocity_mse(flow, analytical, dim=d)
        assert mse < 0.3, f"velocity MSE {mse} too high for scale target"

        z = torch.randn(2000, d)
        with torch.no_grad():
            y_gen = flow.inverse(z, t=1.0, n_steps=100)
        empirical_std = y_gen.std(dim=0)
        assert ((empirical_std - sigma).abs() < 0.3).all(), (
            f"generated std {empirical_std} far from target {sigma}"
        )


@pytest.mark.slow
class TestAxisAlignedAnisotropicTarget:
    """Target = N(0, diag(sigma^2)). Optimal A is diag(sigma)."""

    def test_pointwise_velocity_and_samples(self):
        torch.manual_seed(0)
        d = 2
        sigmas = torch.tensor([2.0, 0.5])
        cov = torch.diag(sigmas ** 2)
        data = sample_gaussian(4000, torch.zeros(d), cov, seed=0)
        flow = train_small_flow(data, dim=d, n_epochs=400, seed=0)
        # A = diag(sigma) is the symmetric PSD sqrt of diag(sigma^2) — matches OT.
        A = torch.diag(sigmas)
        b = torch.zeros(d)

        def analytical(x, t):
            return affine_optimal_velocity(x, t, A, b)

        mse = pointwise_velocity_mse(flow, analytical, dim=d)
        assert mse < 0.5, f"velocity MSE {mse} too high"

        z = torch.randn(2000, d)
        with torch.no_grad():
            y_gen = flow.inverse(z, t=1.0, n_steps=100)
        empirical_std = y_gen.std(dim=0)
        assert ((empirical_std - sigmas).abs() < 0.3).all(), (
            f"generated std {empirical_std} far from target {sigmas}"
        )


@pytest.mark.slow
class TestRotatedAnisotropicTarget:
    """Target = N(0, R diag(sigma^2) R^T). OT map is the symmetric PSD sqrt of
    cov, which is R diag(sigma) R^T — NOT R diag(sigma). The asymmetric factor
    A = R diag(sigma) is a valid factorization of cov but does not give the
    OT-optimal velocity that Hungarian coupling trains the flow toward.

    Included specifically because 'jumble of regions' on ThreeBlobClassifier3D
    is a plausible symptom of a rotation-handling bug; hyperrect/ball scores
    would mask rotation errors, but flow matching should learn the correct
    rotated target."""

    def test_pointwise_velocity_and_covariance(self):
        torch.manual_seed(0)
        d = 2
        sigmas = torch.tensor([2.0, 0.5])
        angle = math.pi / 6  # 30 degrees
        c, s = math.cos(angle), math.sin(angle)
        R = torch.tensor([[c, -s], [s, c]])
        cov = R @ torch.diag(sigmas ** 2) @ R.T
        data = sample_gaussian(4000, torch.zeros(d), cov, seed=0)
        flow = train_small_flow(data, dim=d, n_epochs=500, seed=0)
        # A = sym_psd_sqrt(cov) = R @ diag(sigma) @ R^T is the OT map.
        A = sym_psd_sqrt(cov)
        b = torch.zeros(d)

        def analytical(x, t):
            return affine_optimal_velocity(x, t, A, b)

        mse = pointwise_velocity_mse(flow, analytical, dim=d)
        assert mse < 0.6, f"velocity MSE {mse} too high for rotated target"

        z = torch.randn(4000, d)
        with torch.no_grad():
            y_gen = flow.inverse(z, t=1.0, n_steps=100)
        empirical_cov = torch.cov(y_gen.T)
        assert (empirical_cov - cov).norm() < 0.5, (
            f"generated cov {empirical_cov} differs from target {cov}"
        )


@pytest.mark.slow
class TestFullAffineTarget3D:
    """Target = N(mu, Sigma) in R^3 with arbitrary Sigma. A = chol(Sigma)."""

    def test_full_affine(self):
        torch.manual_seed(42)
        d = 3
        mu = torch.tensor([1.0, -0.5, 0.2])
        B = torch.tensor([[1.5, 0.3, 0.1], [0.3, 0.8, -0.2], [0.1, -0.2, 1.2]])
        cov = B @ B.T
        data = sample_gaussian(6000, mu, cov, seed=42)
        flow = train_small_flow(data, dim=d, n_epochs=600, seed=42)

        # A must be the symmetric PSD sqrt of cov (the OT map), not Cholesky.
        A = sym_psd_sqrt(cov)
        b = mu

        def analytical(x, t):
            return affine_optimal_velocity(x, t, A, b)

        mse = pointwise_velocity_mse(flow, analytical, dim=d, n_x=144)
        assert mse < 0.8, f"velocity MSE {mse} too high for 3D affine"

        z = torch.randn(4000, d)
        with torch.no_grad():
            y_gen = flow.inverse(z, t=1.0, n_steps=100)
        assert (y_gen.mean(dim=0) - mu).norm() < 0.25
        assert (torch.cov(y_gen.T) - cov).norm() < 0.8
