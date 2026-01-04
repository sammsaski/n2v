"""
Unit tests for verify.py module.
"""

import pytest
import numpy as np

from n2v.probabilistic import verify, ProbabilisticBox
from n2v.sets import Box


class TestVerifyBasic:
    """Basic tests for verify() function."""

    def test_verify_with_identity_model(self):
        """Test verify() with identity model (y = x)."""
        def identity_model(x):
            return x

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        input_set = Box(lb, ub)

        result = verify(
            model=identity_model,
            input_set=input_set,
            m=100,
            ell=99,
            epsilon=0.1,
            surrogate='naive',
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)
        assert result.dim == 2
        assert result.m == 100
        assert result.ell == 99
        assert result.epsilon == 0.1
        assert result.coverage == 0.9

    def test_verify_with_linear_model(self):
        """Test verify() with linear model (y = 2x + 1)."""
        def linear_model(x):
            return 2 * x + 1

        lb = np.array([0.0, 0.0, 0.0])
        ub = np.array([1.0, 1.0, 1.0])
        input_set = Box(lb, ub)

        result = verify(
            model=linear_model,
            input_set=input_set,
            m=100,
            ell=99,
            epsilon=0.1,
            surrogate='naive',
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)
        assert result.dim == 3

        # For y = 2x + 1 with x in [0, 1], output should be in [1, 3]
        # The bounds should contain this range
        assert np.all(result.lb <= 1.0 + 0.5)  # Some tolerance
        assert np.all(result.ub >= 3.0 - 0.5)  # Some tolerance

    def test_verify_with_relu_model(self):
        """Test verify() with simple ReLU model."""
        def relu_model(x):
            return np.maximum(0, x - 0.5)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        input_set = Box(lb, ub)

        result = verify(
            model=relu_model,
            input_set=input_set,
            m=100,
            ell=99,
            epsilon=0.1,
            surrogate='naive',
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)
        assert result.dim == 2


class TestVerifyReturns:
    """Tests for verify() return values."""

    def test_returns_probabilistic_box(self):
        """Test that verify() returns ProbabilisticBox."""
        def model(x):
            return x

        input_set = Box(np.zeros(5), np.ones(5))

        result = verify(
            model=model,
            input_set=input_set,
            m=50,
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)

    def test_coverage_matches_input_epsilon(self):
        """Test that coverage matches 1 - epsilon."""
        def model(x):
            return x

        input_set = Box(np.zeros(3), np.ones(3))

        for epsilon in [0.01, 0.05, 0.1]:
            result = verify(
                model=model,
                input_set=input_set,
                m=50,
                epsilon=epsilon,
                seed=42
            )

            assert result.epsilon == epsilon
            assert result.coverage == 1 - epsilon

    def test_m_and_ell_match_input(self):
        """Test that m and ell match input parameters."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        result = verify(
            model=model,
            input_set=input_set,
            m=100,
            ell=95,
            epsilon=0.05,
            seed=42
        )

        assert result.m == 100
        assert result.ell == 95


class TestVerifySurrogates:
    """Tests for different surrogate methods."""

    def test_naive_surrogate(self):
        """Test verify() with naive surrogate."""
        def model(x):
            return x

        input_set = Box(np.zeros(3), np.ones(3))

        result = verify(
            model=model,
            input_set=input_set,
            m=50,
            surrogate='naive',
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)

    def test_clipping_block_surrogate(self):
        """Test verify() with clipping_block surrogate."""
        def model(x):
            return x

        input_set = Box(np.zeros(3), np.ones(3))

        result = verify(
            model=model,
            input_set=input_set,
            m=50,
            surrogate='clipping_block',
            training_samples=25,
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)


class TestVerifyValidation:
    """Tests for input validation."""

    def test_invalid_input_set_type(self):
        """Test that non-Box input_set raises TypeError."""
        def model(x):
            return x

        with pytest.raises(TypeError, match="must be a Box"):
            verify(
                model=model,
                input_set=np.array([0, 1]),  # Not a Box
                m=50
            )

    def test_invalid_m_raises_error(self):
        """Test that invalid m raises ValueError."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        with pytest.raises(ValueError, match="m must be"):
            verify(model=model, input_set=input_set, m=0)

    def test_invalid_ell_raises_error(self):
        """Test that invalid ell raises ValueError."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        with pytest.raises(ValueError, match="ell must be in"):
            verify(model=model, input_set=input_set, m=50, ell=51)

        with pytest.raises(ValueError, match="ell must be in"):
            verify(model=model, input_set=input_set, m=50, ell=0)

    def test_invalid_epsilon_raises_error(self):
        """Test that invalid epsilon raises ValueError."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        with pytest.raises(ValueError, match="epsilon must be in"):
            verify(model=model, input_set=input_set, m=50, epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be in"):
            verify(model=model, input_set=input_set, m=50, epsilon=1.0)

    def test_invalid_surrogate_raises_error(self):
        """Test that invalid surrogate raises ValueError."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        with pytest.raises(ValueError, match="surrogate must be"):
            verify(model=model, input_set=input_set, m=50, surrogate='invalid')


class TestVerifyBatchedInference:
    """Tests for batched inference."""

    def test_batched_inference_produces_correct_shape(self):
        """Test that batched inference produces correct output shape."""
        call_count = [0]

        def model(x):
            call_count[0] += 1
            return x * 2

        input_set = Box(np.zeros(5), np.ones(5))

        result = verify(
            model=model,
            input_set=input_set,
            m=100,
            batch_size=25,  # 100 calibration / 25 = 4 batches
            training_samples=50,  # 50 training / 25 = 2 batches
            seed=42
        )

        assert result.dim == 5
        # Model should be called multiple times for batching
        assert call_count[0] > 1

    def test_small_batch_size(self):
        """Test with very small batch size."""
        def model(x):
            return x

        input_set = Box(np.zeros(3), np.ones(3))

        result = verify(
            model=model,
            input_set=input_set,
            m=30,
            batch_size=5,
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)


class TestVerifyReproducibility:
    """Tests for reproducibility with seed."""

    def test_same_seed_same_result(self):
        """Test that same seed produces same result."""
        def model(x):
            return x + np.sin(x)

        input_set = Box(np.zeros(3), np.ones(3))

        result1 = verify(
            model=model,
            input_set=input_set,
            m=50,
            seed=12345
        )

        result2 = verify(
            model=model,
            input_set=input_set,
            m=50,
            seed=12345
        )

        np.testing.assert_array_equal(result1.lb, result2.lb)
        np.testing.assert_array_equal(result1.ub, result2.ub)

    def test_different_seed_different_result(self):
        """Test that different seeds produce different results."""
        def model(x):
            return x + np.sin(x)

        input_set = Box(np.zeros(3), np.ones(3))

        result1 = verify(
            model=model,
            input_set=input_set,
            m=50,
            seed=12345
        )

        result2 = verify(
            model=model,
            input_set=input_set,
            m=50,
            seed=54321
        )

        # Results should be different (very unlikely to be identical)
        assert not np.allclose(result1.lb, result2.lb)


class TestVerifyDefaults:
    """Tests for default parameter values."""

    def test_default_ell_is_m_minus_1(self):
        """Test that default ell is m - 1."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        result = verify(
            model=model,
            input_set=input_set,
            m=100,
            # ell not specified
            seed=42
        )

        assert result.ell == 99  # m - 1

    def test_default_epsilon(self):
        """Test default epsilon is 0.001."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        result = verify(
            model=model,
            input_set=input_set,
            m=100,
            seed=42
        )

        assert result.epsilon == 0.001


class TestVerifyHighDimensional:
    """Tests for higher dimensional inputs/outputs."""

    def test_high_dimensional_input(self):
        """Test with high-dimensional input."""
        def model(x):
            return x @ np.random.randn(100, 10)  # Project to 10 dims

        np.random.seed(42)
        input_set = Box(np.zeros(100), np.ones(100))

        result = verify(
            model=model,
            input_set=input_set,
            m=50,
            seed=42
        )

        assert result.dim == 10

    def test_single_output_dimension(self):
        """Test with single output dimension."""
        def model(x):
            return np.sum(x, axis=1, keepdims=True)

        input_set = Box(np.zeros(5), np.ones(5))

        result = verify(
            model=model,
            input_set=input_set,
            m=50,
            seed=42
        )

        assert result.dim == 1


class TestVerifyImports:
    """Tests for module imports."""

    def test_import_verify_from_probabilistic(self):
        """Test that verify can be imported from n2v.probabilistic."""
        from n2v.probabilistic import verify as v
        assert callable(v)

    def test_import_probabilistic_box_from_probabilistic(self):
        """Test that ProbabilisticBox can be imported from n2v.probabilistic."""
        from n2v.probabilistic import ProbabilisticBox as PB
        assert PB is not None
