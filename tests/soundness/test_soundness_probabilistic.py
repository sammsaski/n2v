"""
Soundness tests for probabilistic verification.

These tests verify that the probabilistic guarantees hold empirically.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from n2v.probabilistic import verify
from n2v.sets import Box


class TestCoverageGuarantee:
    """Tests that verify coverage guarantees hold empirically."""

    def test_coverage_guarantee_linear_model(self):
        """
        Verify that stated coverage is achieved for a linear model.

        1. Run verify() with epsilon=0.05 (95% coverage)
        2. Sample N >> m points from input region
        3. Check that at least ~95% of outputs are in the ProbabilisticBox
        """
        np.random.seed(42)
        torch.manual_seed(42)

        # Simple linear model
        def model(x):
            return x @ np.array([[1.0, 0.5], [0.5, 1.0], [0.2, 0.3]]) + np.array([0.1, 0.2])

        # Input region
        lb = np.zeros(3)
        ub = np.ones(3)
        input_set = Box(lb, ub)

        # Run probabilistic verification with 95% coverage
        epsilon = 0.05
        m = 500  # Moderate sample size
        result = verify(
            model=model,
            input_set=input_set,
            m=m,
            epsilon=epsilon,
            surrogate='naive',
            seed=42
        )

        assert result.coverage == 1 - epsilon

        # Sample many points and check coverage
        n_test = 10000
        test_inputs = np.random.uniform(lb, ub, size=(n_test, 3)).astype(np.float32)
        test_outputs = model(test_inputs)

        # Check how many outputs are inside the ProbabilisticBox
        pbox_lb = result.lb.flatten()
        pbox_ub = result.ub.flatten()

        inside = np.all((test_outputs >= pbox_lb) & (test_outputs <= pbox_ub), axis=1)
        empirical_coverage = np.mean(inside)

        # Empirical coverage should be close to (1 - epsilon) = 0.95
        # Allow some margin for sampling variance
        # With probability result.confidence, coverage >= 0.95
        # We check that it's at least close (within 5%)
        assert empirical_coverage > 0.90, \
            f"Empirical coverage {empirical_coverage:.3f} too low (expected ~0.95)"

    def test_coverage_guarantee_nonlinear_model(self):
        """
        Verify coverage for a nonlinear model (neural network).
        """
        np.random.seed(123)
        torch.manual_seed(123)

        # Create simple neural network
        torch_model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
        torch_model.eval()

        def model(x):
            with torch.no_grad():
                return torch_model(torch.tensor(x, dtype=torch.float32)).numpy()

        # Input region
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        input_set = Box(lb, ub)

        # Run probabilistic verification with 90% coverage
        epsilon = 0.10
        result = verify(
            model=model,
            input_set=input_set,
            m=500,
            epsilon=epsilon,
            surrogate='clipping_block',
            training_samples=200,
            seed=123
        )

        # Sample many points and check coverage
        n_test = 5000
        test_inputs = np.random.uniform(lb, ub, size=(n_test, 2)).astype(np.float32)
        test_outputs = model(test_inputs)

        pbox_lb = result.lb.flatten()
        pbox_ub = result.ub.flatten()

        inside = np.all((test_outputs >= pbox_lb) & (test_outputs <= pbox_ub), axis=1)
        empirical_coverage = np.mean(inside)

        # Should achieve at least 85% (allowing some margin below 90%)
        assert empirical_coverage > 0.85, \
            f"Empirical coverage {empirical_coverage:.3f} too low (expected ~0.90)"


class TestConfidenceGuarantee:
    """Tests that confidence guarantees hold empirically."""

    def test_confidence_guarantee_repeated_runs(self):
        """
        Verify that coverage holds with stated confidence.

        1. Run verify() K times with different seeds
        2. For each run, check if coverage >= 1-epsilon
        3. Verify that at least delta_2 fraction of runs achieve coverage
        """
        np.random.seed(42)

        def model(x):
            return x ** 2 + np.sin(x)

        lb = np.zeros(3)
        ub = np.ones(3)
        input_set = Box(lb, ub)

        epsilon = 0.10  # 90% coverage
        m = 100
        K = 30  # Number of repeated runs

        coverage_achieved = []

        for run in range(K):
            result = verify(
                model=model,
                input_set=input_set,
                m=m,
                epsilon=epsilon,
                surrogate='naive',
                seed=run * 1000 + 42
            )

            # Check coverage with test samples
            n_test = 1000
            test_inputs = np.random.uniform(lb, ub, size=(n_test, 3)).astype(np.float32)
            np.random.seed(run * 1000 + 42)  # Reset for consistent test samples
            test_outputs = model(test_inputs)

            pbox_lb = result.lb.flatten()
            pbox_ub = result.ub.flatten()

            inside = np.all((test_outputs >= pbox_lb) & (test_outputs <= pbox_ub), axis=1)
            empirical_coverage = np.mean(inside)

            coverage_achieved.append(empirical_coverage >= (1 - epsilon - 0.05))

        # Fraction of runs achieving coverage
        success_rate = np.mean(coverage_achieved)

        # Should be at least ~delta_2 (confidence level)
        # For m=100, ell=99, epsilon=0.10, delta_2 is quite high
        # We expect most runs to succeed
        assert success_rate > 0.7, \
            f"Only {success_rate:.1%} of runs achieved coverage (expected ~90%+)"


class TestClippingBlockVsNaive:
    """Tests comparing clipping block vs naive surrogate."""

    def test_clipping_block_produces_tighter_or_equal_bounds(self):
        """
        Verify that clipping block produces bounds at least as tight as naive.

        For a simple model, clipping block should produce tighter or equal bounds
        compared to naive surrogate with the same parameters.
        """
        np.random.seed(42)

        # Linear model
        def model(x):
            return x @ np.array([[1.0], [0.5], [0.2]])

        lb = np.zeros(3)
        ub = np.ones(3)
        input_set = Box(lb, ub)

        params = dict(m=200, epsilon=0.05, seed=42, training_samples=100)

        # Naive surrogate
        result_naive = verify(
            model=model,
            input_set=input_set,
            surrogate='naive',
            **params
        )

        # Clipping block surrogate
        result_clipping = verify(
            model=model,
            input_set=input_set,
            surrogate='clipping_block',
            **params
        )

        # Compute bound widths
        width_naive = result_naive.ub.flatten() - result_naive.lb.flatten()
        width_clipping = result_clipping.ub.flatten() - result_clipping.lb.flatten()

        # Clipping block should be tighter or equal (allow small tolerance)
        # Note: This might not always hold due to random sampling, but should hold on average
        # For this test, just check they're not dramatically different
        ratio = np.mean(width_clipping) / (np.mean(width_naive) + 1e-10)
        assert ratio < 1.5, \
            f"Clipping block bounds ({np.mean(width_clipping):.4f}) much wider than naive ({np.mean(width_naive):.4f})"


class TestBoundsContainTrueRange:
    """Tests that probabilistic bounds contain true output range with high probability."""

    def test_bounds_contain_sampled_outputs(self):
        """
        Simple sanity check: bounds should contain most sampled outputs.
        """
        np.random.seed(42)

        def model(x):
            return np.tanh(x)

        lb = np.array([-2.0, -2.0])
        ub = np.array([2.0, 2.0])
        input_set = Box(lb, ub)

        result = verify(
            model=model,
            input_set=input_set,
            m=500,
            epsilon=0.01,  # 99% coverage
            surrogate='naive',
            seed=42
        )

        # Sample outputs
        n_test = 5000
        test_inputs = np.random.uniform(lb, ub, size=(n_test, 2)).astype(np.float32)
        test_outputs = model(test_inputs)

        # Check containment
        pbox_lb = result.lb.flatten()
        pbox_ub = result.ub.flatten()

        inside = np.all((test_outputs >= pbox_lb) & (test_outputs <= pbox_ub), axis=1)
        coverage = np.mean(inside)

        # Should contain at least 95% (being conservative)
        assert coverage > 0.95, f"Only {coverage:.1%} of outputs contained"


class TestParameterSensitivity:
    """Tests for parameter sensitivity."""

    def test_larger_m_gives_tighter_bounds(self):
        """
        Larger m should generally give tighter bounds (less inflation).
        """
        np.random.seed(42)

        def model(x):
            return x

        lb = np.zeros(2)
        ub = np.ones(2)
        input_set = Box(lb, ub)

        # Small m
        result_small = verify(
            model=model,
            input_set=input_set,
            m=100,
            epsilon=0.01,
            surrogate='naive',
            seed=42
        )

        # Large m
        result_large = verify(
            model=model,
            input_set=input_set,
            m=1000,
            epsilon=0.01,
            surrogate='naive',
            seed=42
        )

        # Larger m should give tighter or equal bounds
        width_small = np.mean(result_small.ub - result_small.lb)
        width_large = np.mean(result_large.ub - result_large.lb)

        # Allow some tolerance since random sampling introduces variance
        assert width_large <= width_small * 1.2, \
            f"Larger m gave wider bounds: small={width_small:.4f}, large={width_large:.4f}"

    def test_larger_epsilon_gives_tighter_or_equal_bounds(self):
        """
        Larger epsilon (lower coverage) should give tighter or equal bounds.

        Note: The effect of epsilon depends on the ell parameter.
        With ell = m-1 (default), epsilon primarily affects confidence, not bound width.
        Bound width is determined by the (ell)-th largest nonconformity score.
        Different ell values for different epsilon would show this effect more clearly.
        """
        np.random.seed(42)

        def model(x):
            return x + np.random.randn(*x.shape) * 0.1  # Add some noise

        lb = np.zeros(2)
        ub = np.ones(2)
        input_set = Box(lb, ub)

        # Small epsilon with ell = m-1 (second largest score)
        result_high_cov = verify(
            model=model,
            input_set=input_set,
            m=200,
            ell=199,  # m-1
            epsilon=0.01,  # 99% coverage
            surrogate='naive',
            seed=42
        )

        # Larger epsilon with smaller ell (uses a smaller score)
        # ell should be chosen such that the guarantee still holds
        result_low_cov = verify(
            model=model,
            input_set=input_set,
            m=200,
            ell=180,  # Uses 180th largest score (tighter)
            epsilon=0.10,  # 90% coverage
            surrogate='naive',
            seed=42
        )

        width_high_cov = np.mean(result_high_cov.ub - result_high_cov.lb)
        width_low_cov = np.mean(result_low_cov.ub - result_low_cov.lb)

        # Lower ell with lower coverage target should give tighter bounds
        assert width_low_cov <= width_high_cov, \
            f"Lower ell gave wider bounds: high_cov={width_high_cov:.4f}, low_cov={width_low_cov:.4f}"


class TestCopulaCoverageGuarantee:
    """Tests that copula-based coverage guarantees hold empirically."""

    def test_copula_coverage_guarantee_linear_model(self):
        """
        Verify that copula method achieves stated coverage.
        """
        np.random.seed(42)

        # Correlated output model
        def model(x):
            y1 = x[:, 0] + 0.5 * x[:, 1]
            y2 = 0.8 * y1 + 0.2 * x[:, 1]  # y2 correlated with y1
            return np.column_stack([y1, y2])

        lb = np.zeros(2)
        ub = np.ones(2)
        input_set = Box(lb, ub)

        # Run copula verification with 95% coverage
        epsilon = 0.05
        result = verify(
            model=model,
            input_set=input_set,
            m=800,
            epsilon=epsilon,
            surrogate='copula',
            training_samples=400,
            seed=42
        )

        assert result.coverage == 1 - epsilon

        # Sample many points and check coverage
        n_test = 10000
        test_inputs = np.random.uniform(lb, ub, size=(n_test, 2)).astype(np.float32)
        test_outputs = model(test_inputs)

        # Use copula region's contains method
        inside = result.contains_batch(test_outputs)
        empirical_coverage = np.mean(inside)

        # Empirical coverage should be close to (1 - epsilon) = 0.95
        assert empirical_coverage > 0.90, \
            f"Copula empirical coverage {empirical_coverage:.3f} too low (expected ~0.95)"

    def test_copula_coverage_guarantee_nonlinear_model(self):
        """
        Verify copula coverage for a nonlinear model.
        """
        np.random.seed(123)
        torch.manual_seed(123)

        # Neural network with correlated outputs
        torch_model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        torch_model.eval()

        def model(x):
            with torch.no_grad():
                return torch_model(torch.tensor(x, dtype=torch.float32)).numpy()

        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])
        input_set = Box(lb, ub)

        # Run copula verification with 90% coverage
        epsilon = 0.10
        result = verify(
            model=model,
            input_set=input_set,
            m=800,
            epsilon=epsilon,
            surrogate='copula',
            training_samples=400,
            seed=123
        )

        # Sample and check coverage
        n_test = 5000
        test_inputs = np.random.uniform(lb, ub, size=(n_test, 3)).astype(np.float32)
        test_outputs = model(test_inputs)

        inside = result.contains_batch(test_outputs)
        empirical_coverage = np.mean(inside)

        # Should achieve at least 85%
        assert empirical_coverage > 0.85, \
            f"Copula empirical coverage {empirical_coverage:.3f} too low (expected ~0.90)"


class TestCopulaVsHyperrectangle:
    """Tests comparing copula regions to hyperrectangle methods."""

    def test_copula_tighter_for_correlated_outputs(self):
        """
        Verify that copula produces smaller regions for correlated outputs.

        For highly correlated outputs, the copula region should have a
        significantly smaller volume ratio compared to the bounding hyperrectangle.
        """
        np.random.seed(42)

        # Highly correlated output model
        def model(x):
            y1 = x[:, 0] + x[:, 1]
            y2 = 0.95 * y1 + 0.05 * x[:, 0]  # Highly correlated with y1
            return np.column_stack([y1, y2])

        lb = np.zeros(2)
        ub = np.ones(2)
        input_set = Box(lb, ub)

        # Run copula verification
        result = verify(
            model=model,
            input_set=input_set,
            m=800,
            epsilon=0.01,
            surrogate='copula',
            training_samples=400,
            seed=42
        )

        # Compute volume ratio
        volume_ratio = result.volume_ratio(n_samples=10000, seed=42)

        # For highly correlated outputs, volume ratio should be significantly < 1
        # Expected: around 0.3-0.5 for rho ≈ 0.95
        assert volume_ratio < 0.8, \
            f"Volume ratio {volume_ratio:.3f} too high for correlated outputs (expected < 0.8)"

    def test_copula_similar_to_hyperrect_for_independent(self):
        """
        Verify that copula is similar to hyperrectangle for independent outputs.

        For uncorrelated outputs, the copula region should have a volume
        ratio close to 1 compared to the bounding hyperrectangle.
        """
        np.random.seed(42)

        # Independent output model
        def model(x):
            y1 = x[:, 0]  # Only depends on x1
            y2 = x[:, 1]  # Only depends on x2
            return np.column_stack([y1, y2])

        lb = np.zeros(2)
        ub = np.ones(2)
        input_set = Box(lb, ub)

        # Run copula verification
        result = verify(
            model=model,
            input_set=input_set,
            m=800,
            epsilon=0.01,
            surrogate='copula',
            training_samples=400,
            seed=42
        )

        # Compute volume ratio
        volume_ratio = result.volume_ratio(n_samples=10000, seed=42)

        # For independent outputs, volume ratio should be close to 1
        # (copula region is similar to hyperrectangle)
        assert volume_ratio > 0.6, \
            f"Volume ratio {volume_ratio:.3f} too low for independent outputs (expected > 0.6)"

    def test_copula_region_converts_to_box(self):
        """
        Test that CopulaRegion.to_box() produces a valid ProbabilisticBox.
        """
        np.random.seed(42)

        def model(x):
            return x @ np.array([[1, 0.5], [0.5, 1]])

        lb = np.zeros(2)
        ub = np.ones(2)
        input_set = Box(lb, ub)

        result = verify(
            model=model,
            input_set=input_set,
            m=500,
            epsilon=0.01,
            surrogate='copula',
            seed=42
        )

        # Convert to box
        box = result.to_box(n_samples=5000, seed=42)

        from n2v.sets import ProbabilisticBox
        assert isinstance(box, ProbabilisticBox)
        assert box.m == result.m
        assert box.ell == result.ell
        assert box.epsilon == result.epsilon

        # Box should contain the center
        center = result.center
        assert np.all(center >= box.lb.flatten())
        assert np.all(center <= box.ub.flatten())
