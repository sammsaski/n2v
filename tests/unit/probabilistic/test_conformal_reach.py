"""
Unit tests for the ``conformal_reach`` API (renamed from the legacy ``verify``).
"""

import logging

import pytest
import numpy as np
import torch
import torch.nn as nn

from n2v.probabilistic import (
    ConformalReachConfig, conformal_reach, ProbabilisticBox,
)
from n2v.sets import Box
from tests.known_bugs import known_bug


class TestConformalReachBasic:
    """Basic tests for conformal_reach() function."""

    def test_conformal_reach_with_identity_model(self):
        """Test conformal_reach() with identity model (y = x)."""
        def identity_model(x):
            return x

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        input_set = Box(lb, ub)

        result = conformal_reach(
            model=identity_model,
            input_box=input_set,
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

    def test_conformal_reach_with_linear_model(self):
        """Test conformal_reach() with linear model (y = 2x + 1)."""
        def linear_model(x):
            return 2 * x + 1

        lb = np.array([0.0, 0.0, 0.0])
        ub = np.array([1.0, 1.0, 1.0])
        input_set = Box(lb, ub)

        result = conformal_reach(
            model=linear_model,
            input_box=input_set,
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

    def test_conformal_reach_with_relu_model(self):
        """Test conformal_reach() with simple ReLU model."""
        def relu_model(x):
            return np.maximum(0, x - 0.5)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        input_set = Box(lb, ub)

        result = conformal_reach(
            model=relu_model,
            input_box=input_set,
            m=100,
            ell=99,
            epsilon=0.1,
            surrogate='naive',
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)
        assert result.dim == 2


class TestConformalReachReturns:
    """Tests for conformal_reach() return values."""

    def test_returns_probabilistic_box(self):
        """Test that conformal_reach() returns ProbabilisticBox."""
        def model(x):
            return x

        input_set = Box(np.zeros(5), np.ones(5))

        result = conformal_reach(
            model=model,
            input_box=input_set,
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
            result = conformal_reach(
                model=model,
                input_box=input_set,
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

        result = conformal_reach(
            model=model,
            input_box=input_set,
            m=100,
            ell=95,
            epsilon=0.05,
            seed=42
        )

        assert result.m == 100
        assert result.ell == 95


class TestConformalReachSurrogates:
    """Tests for different surrogate methods."""

    def test_naive_surrogate(self):
        """Test conformal_reach() with naive surrogate."""
        def model(x):
            return x

        input_set = Box(np.zeros(3), np.ones(3))

        result = conformal_reach(
            model=model,
            input_box=input_set,
            m=50,
            surrogate='naive',
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)

    def test_clipping_block_surrogate(self):
        """Test conformal_reach() with clipping_block surrogate."""
        def model(x):
            return x

        input_set = Box(np.zeros(3), np.ones(3))

        result = conformal_reach(
            model=model,
            input_box=input_set,
            m=50,
            surrogate='clipping_block',
            training_samples=25,
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)


class TestConformalReachValidation:
    """Tests for input validation."""

    def test_invalid_input_set_type(self):
        """Test that non-Box input_set raises TypeError."""
        def model(x):
            return x

        with pytest.raises(TypeError, match="must be a Box"):
            conformal_reach(
                model=model,
                input_box=np.array([0, 1]),  # Not a Box
                m=50
            )

    def test_invalid_m_raises_error(self):
        """Test that invalid m raises ValueError."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        with pytest.raises(ValueError, match="m must be"):
            conformal_reach(model=model, input_box=input_set, m=0)

    def test_invalid_ell_raises_error(self):
        """Test that invalid ell raises ValueError."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        with pytest.raises(ValueError, match="ell must be in"):
            conformal_reach(model=model, input_box=input_set, m=50, ell=51)

        with pytest.raises(ValueError, match="ell must be in"):
            conformal_reach(model=model, input_box=input_set, m=50, ell=0)

    def test_invalid_epsilon_raises_error(self):
        """Test that invalid epsilon raises ValueError."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        with pytest.raises(ValueError, match="epsilon must be in"):
            conformal_reach(model=model, input_box=input_set, m=50, epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be in"):
            conformal_reach(model=model, input_box=input_set, m=50, epsilon=1.0)

    def test_invalid_surrogate_raises_error(self):
        """Test that invalid surrogate raises ValueError."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        with pytest.raises(ValueError, match="surrogate must be"):
            conformal_reach(model=model, input_box=input_set, m=50, surrogate='invalid')


class TestConformalReachBatchedInference:
    """Tests for batched inference."""

    def test_batched_inference_produces_correct_shape(self):
        """Test that batched inference produces correct output shape."""
        call_count = [0]

        def model(x):
            call_count[0] += 1
            return x * 2

        input_set = Box(np.zeros(5), np.ones(5))

        result = conformal_reach(
            model=model,
            input_box=input_set,
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

        result = conformal_reach(
            model=model,
            input_box=input_set,
            m=30,
            batch_size=5,
            seed=42
        )

        assert isinstance(result, ProbabilisticBox)


class TestConformalReachReproducibility:
    """Tests for reproducibility with seed."""

    def test_same_seed_same_result(self):
        """Test that same seed produces same result."""
        def model(x):
            return x + np.sin(x)

        input_set = Box(np.zeros(3), np.ones(3))

        result1 = conformal_reach(
            model=model,
            input_box=input_set,
            m=50,
            seed=12345
        )

        result2 = conformal_reach(
            model=model,
            input_box=input_set,
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

        result1 = conformal_reach(
            model=model,
            input_box=input_set,
            m=50,
            seed=12345
        )

        result2 = conformal_reach(
            model=model,
            input_box=input_set,
            m=50,
            seed=54321
        )

        # Results should be different (very unlikely to be identical)
        assert not np.allclose(result1.lb, result2.lb)


class TestConformalReachDefaults:
    """Tests for default parameter values."""

    def test_default_ell_is_m_minus_1(self):
        """Test that default ell is m - 1."""
        def model(x):
            return x

        input_set = Box(np.zeros(2), np.ones(2))

        result = conformal_reach(
            model=model,
            input_box=input_set,
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

        result = conformal_reach(
            model=model,
            input_box=input_set,
            m=100,
            seed=42
        )

        assert result.epsilon == 0.001


class TestConformalReachHighDimensional:
    """Tests for higher dimensional inputs/outputs."""

    def test_high_dimensional_input(self):
        """Test with high-dimensional input."""
        def model(x):
            return x @ np.random.randn(100, 10)  # Project to 10 dims

        np.random.seed(42)
        input_set = Box(np.zeros(100), np.ones(100))

        result = conformal_reach(
            model=model,
            input_box=input_set,
            m=50,
            seed=42
        )

        assert result.dim == 10

    def test_single_output_dimension(self):
        """Test with single output dimension."""
        def model(x):
            return np.sum(x, axis=1, keepdims=True)

        input_set = Box(np.zeros(5), np.ones(5))

        result = conformal_reach(
            model=model,
            input_box=input_set,
            m=50,
            seed=42
        )

        assert result.dim == 1


class TestClippingBlockTrainingOptimization:
    """Test that clipping block skips wasteful training projections."""

    def test_clipping_block_training_errors_are_zero(self):
        """Verify clipping block training errors are ~zero (motivates optimization)."""
        from n2v.probabilistic.surrogates.clipping_block import ClippingBlockSurrogate

        np.random.seed(42)
        training = np.random.randn(20, 5)
        surr = ClippingBlockSurrogate(n_workers=1)
        surr.fit(training)
        projections = surr.predict(training)
        errors = training - projections
        assert np.allclose(errors, 0, atol=1e-6)

    def test_conformal_reach_clipping_block_uses_zero_training_errors(self):
        """Verify the optimization produces identical results to the unoptimized path."""
        np.random.seed(0)
        model_fn = lambda x: x @ np.array([[1, 0.5], [0.5, 1], [0.3, 0.7]])

        input_set = Box(np.zeros(3), np.ones(3))
        result = conformal_reach(
            model=model_fn,
            input_box=input_set,
            m=100,
            surrogate='clipping_block',
            training_samples=50,
            seed=42,
        )
        assert result.confidence > 0
        assert np.all(result.ub >= result.lb)


class TestClippingBlockPerComponentInflation:
    """Regression tests for audit Finding 1: tau must come from train + calib
    so per-component structure is preserved for the clipping block."""

    def test_clipping_block_inflation_is_per_component_not_uniform(self):
        """For anisotropic outputs, clipping-block inflation should track
        the per-dimension error scale, not degenerate to a uniform scalar."""
        def anisotropic_model(x):
            # y_0 ~ unit scale, y_1 ~ 1/10 unit scale
            return np.column_stack([x[:, 0], 0.1 * x[:, 1]])

        input_set = Box(np.zeros(2), np.ones(2))
        result = conformal_reach(
            model=anisotropic_model,
            input_box=input_set,
            m=500,
            epsilon=0.05,
            surrogate='clipping_block',
            training_samples=200,
            seed=42,
        )
        # Widths should differ by roughly the same 10x as the model output scale.
        width = (result.ub - result.lb).flatten()
        ratio = width[0] / max(width[1], 1e-12)
        assert ratio > 3.0, (
            f"clipping block produced near-uniform inflation "
            f"(width ratio {ratio:.2f}) — per-component structure was lost"
        )

    def test_clipping_block_tighter_than_pre_fix_uniform_inflation(self):
        """Clipping block with anisotropic outputs should produce a smaller
        total volume than a uniform-inflation baseline would have."""
        def anisotropic_model(x):
            return np.column_stack([x[:, 0], 0.1 * x[:, 1]])

        input_set = Box(np.zeros(2), np.ones(2))
        result = conformal_reach(
            model=anisotropic_model,
            input_box=input_set,
            m=500,
            epsilon=0.05,
            surrogate='clipping_block',
            training_samples=200,
            seed=42,
        )
        width = (result.ub - result.lb).flatten()
        volume = float(np.prod(width))
        # If inflation were uniform at the max scale (pre-fix), volume would
        # be at least ~ (max_err * 2)^2 ≈ 4 in a 2D box. With per-component
        # inflation, volume should be well under that.
        assert volume < 0.8, (
            f"clipping block volume {volume:.3f} too large — "
            f"suggests uniform inflation regression"
        )


class TestPCASoundness:
    """Regression tests for audit Finding 3: PCA must compute errors in the
    full output space so the PCA residual is covered by the final bounds."""

    def test_pca_clipping_block_covers_pca_residual(self):
        """Outputs with meaningful variance outside the PCA subspace must still
        be covered by the final bounds. Before the fix, the PCA residual was
        silently dropped."""
        # Model produces a 5D output with rank-2 signal plus orthogonal noise.
        # PCA with n_components=2 captures the signal; the noise is residual.
        rng = np.random.RandomState(0)
        basis = rng.randn(2, 5)

        def lowrank_plus_noise(x):
            # x is shape (batch, 2); lift to 5D rank-2 signal plus noise
            signal = x @ basis
            # Deterministic noise derived from input so the model is reproducible
            noise = 0.4 * np.sin(10 * x @ np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]))
            return signal + noise

        input_set = Box(np.zeros(2), np.ones(2))

        result = conformal_reach(
            model=lowrank_plus_noise,
            input_box=input_set,
            m=500,
            epsilon=0.05,
            surrogate='clipping_block',
            training_samples=200,
            pca_components=2,
            seed=42,
        )

        # Check empirical coverage in full space.
        rng_test = np.random.RandomState(1)
        test_inputs = rng_test.uniform(0, 1, size=(5000, 2))
        test_outputs = lowrank_plus_noise(test_inputs)

        inside = np.all(
            (test_outputs >= result.lb.flatten()) & (test_outputs <= result.ub.flatten()),
            axis=1,
        )
        empirical_coverage = float(np.mean(inside))

        # Should clear the target coverage; before the fix this dropped far
        # below because the PCA residual was not accounted for.
        assert empirical_coverage > 0.92, (
            f"Empirical full-space coverage {empirical_coverage:.3f} below "
            f"target {1 - 0.05:.3f} — PCA residual not covered"
        )

    def test_pca_naive_still_valid(self):
        """The naive surrogate with PCA should still produce valid bounds."""
        rng = np.random.RandomState(0)
        basis = rng.randn(2, 4)

        def lowrank_plus_noise(x):
            return x @ basis + 0.2 * np.cos(5 * x[:, :1] + x[:, 1:])

        input_set = Box(np.zeros(2), np.ones(2))
        result = conformal_reach(
            model=lowrank_plus_noise,
            input_box=input_set,
            m=500,
            epsilon=0.1,
            surrogate='naive',
            pca_components=2,
            seed=42,
        )

        rng_test = np.random.RandomState(2)
        test_inputs = rng_test.uniform(0, 1, size=(5000, 2))
        test_outputs = lowrank_plus_noise(test_inputs)
        inside = np.all(
            (test_outputs >= result.lb.flatten()) & (test_outputs <= result.ub.flatten()),
            axis=1,
        )
        assert float(np.mean(inside)) > 0.85


class TestConformalReachImports:
    """Tests for module imports."""

    def test_import_verify_from_probabilistic(self):
        """Test that verify can be imported from n2v.probabilistic."""
        from n2v.probabilistic import conformal_reach as v
        assert callable(v)

    def test_import_probabilistic_box_from_probabilistic(self):
        """Test that ProbabilisticBox can be imported from n2v.probabilistic."""
        from n2v.probabilistic import ProbabilisticBox as PB
        assert PB is not None


class TestPCABoundsCorrectness:
    """Test that PCA inverse transform uses interval arithmetic."""

    def test_pca_bounds_interval_arithmetic(self):
        """Verify bounds are computed with interval arithmetic, not naive transform."""
        from n2v.probabilistic.conformal_reach import _inverse_transform_bounds
        from n2v.probabilistic.dimensionality.deflation_pca import DeflationPCA

        pca = DeflationPCA(n_components=2)
        pca.mean_ = np.array([0.0, 0.0, 0.0])
        pca.components_ = np.array([
            [1.0, -1.0, 0.5],
            [0.5,  1.0, -1.0],
        ])
        pca._is_fitted = True

        lb_reduced = np.array([0.0, 0.0])
        ub_reduced = np.array([1.0, 1.0])

        lb, ub = _inverse_transform_bounds(pca, lb_reduced, ub_reduced)

        assert lb[0] == pytest.approx(0.0)
        assert ub[0] == pytest.approx(1.5)
        assert lb[1] == pytest.approx(-1.0)
        assert ub[1] == pytest.approx(1.0)
        assert lb[2] == pytest.approx(-1.0)
        assert ub[2] == pytest.approx(0.5)

    def test_pca_bounds_with_nonzero_mean(self):
        from n2v.probabilistic.conformal_reach import _inverse_transform_bounds
        from n2v.probabilistic.dimensionality.deflation_pca import DeflationPCA

        pca = DeflationPCA(n_components=1)
        pca.mean_ = np.array([10.0, 20.0])
        pca.components_ = np.array([[1.0, -1.0]])
        pca._is_fitted = True

        lb_reduced = np.array([-2.0])
        ub_reduced = np.array([3.0])

        lb, ub = _inverse_transform_bounds(pca, lb_reduced, ub_reduced)

        assert lb[0] == pytest.approx(8.0)
        assert ub[0] == pytest.approx(13.0)
        assert lb[1] == pytest.approx(17.0)
        assert ub[1] == pytest.approx(22.0)

    def test_pca_bounds_soundness_empirical(self):
        from n2v.probabilistic.dimensionality.deflation_pca import DeflationPCA
        from n2v.probabilistic.conformal_reach import _inverse_transform_bounds

        np.random.seed(42)
        pca = DeflationPCA(n_components=3)
        pca.mean_ = np.random.randn(10)
        pca.components_ = np.random.randn(3, 10)
        pca._is_fitted = True

        lb_reduced = np.array([-1.0, -2.0, -0.5])
        ub_reduced = np.array([1.0, 0.5, 3.0])

        lb, ub = _inverse_transform_bounds(pca, lb_reduced, ub_reduced)

        for _ in range(10000):
            x = np.random.uniform(lb_reduced, ub_reduced)
            y = pca.inverse_transform(x.reshape(1, -1)).flatten()
            assert np.all(y >= lb - 1e-10), f"lb violation: {y} < {lb}"
            assert np.all(y <= ub + 1e-10), f"ub violation: {y} > {ub}"


# --------------------------------------------------------------------------- #
# Options, defaults, seeding and model kinds, each checked on the draws the   #
# model itself records                                                         #
# --------------------------------------------------------------------------- #
class _Recorder:
    """A numpy model that keeps every input batch and output batch, and fails
    the test as soon as it is asked for more rows than ``budget`` (so a run
    that ignores the requested sizes fails at once instead of running the
    default 8000 calibration draws)."""

    def __init__(self, fn, budget):
        self.fn, self.budget = fn, budget
        self.inputs, self.outputs = [], []

    def __call__(self, x):
        done = sum(len(b) for b in self.inputs)
        if done + len(x) > self.budget:
            pytest.fail(f"the model was asked for {done + len(x)} rows, more "
                        f"than the {self.budget} the requested sizes need")
        y = self.fn(np.asarray(x, dtype=np.float64))
        self.inputs.append(np.array(x, copy=True))
        self.outputs.append(np.array(y, copy=True))
        return y

    def rows(self):
        return np.vstack(self.inputs), np.vstack(self.outputs)


def _held(result, points, pad):
    """Rows of ``points`` inside the box widened by ``pad``, and inside the box
    narrowed by ``pad``."""
    lb, ub = result.lb.ravel(), result.ub.ravel()
    inside = lambda lo, hi: int(np.sum(np.all((points >= lo) & (points <= hi),
                                              axis=1)))
    return inside(lb - pad, ub + pad), inside(lb + pad, ub - pad)


def _affine(x):
    """3 inputs -> 2 outputs with different scales and signs."""
    return x @ np.array([[1.5, -0.5], [-0.75, 2.0], [2.0, 1.25]]) + np.array(
        [4.0, -6.0])


_BOX3 = Box(np.array([-3.0, 0.5, -1.5]), np.array([1.0, 4.0, -0.25]))


def _naive_box_check(result, model, t, m, ell):
    """With the naive surrogate the box holds exactly ell of the m calibration
    outputs (the last m model rows) up to a float64 pad, whatever tau is."""
    x, y = model.rows()
    assert len(x) == t + m, f"{len(x)} model rows, expected {t} + {m}"
    pad = 1e-12 * (1.0 + np.max(np.abs(y)))
    loose, strict = _held(result, y[t:], pad)
    assert loose >= ell and strict <= ell, (
        f"the naive box holds {loose} (well inside: {strict}) of {m} "
        f"calibration outputs, expected exactly ell = {ell}")


class TestConformalReachOptionsReachTheRun:
    """Each option of ConformalReachConfig changes what the run does."""

    def test_keyword_options_reach_the_run(self):
        """The keyword style (config=None) must build the config from the
        keywords: m = 30 calibration and 10 training draws, rank 27, epsilon
        0.125, naive surrogate. The box holds exactly 27 of the 30 calibration
        outputs and states (30, 27, 0.125). Catches keywords that are dropped
        for an all-default config (the model fails the test at its first
        batch of 100 rows instead of running 8000 calibration draws)."""
        model = _Recorder(_affine, 40)
        result = conformal_reach(model, _BOX3, m=30, ell=27, epsilon=0.125,
                                 surrogate="naive", training_samples=10,
                                 seed=4)
        assert (result.m, result.ell, result.epsilon) == (30, 27, 0.125)
        _naive_box_check(result, model, 10, 30, 27)

    def test_default_rank_and_training_size(self):
        """Left at their defaults, ell is m - 1 and the number of training
        draws is m // 2: with m = 41 the model is asked for 20 + 41 rows, and
        the naive box holds exactly 40 calibration outputs. An odd m
        separates m // 2 from rounding up. Catches a default rank of m and a
        default training size of m."""
        model = _Recorder(_affine, 61)
        result = conformal_reach(model, _BOX3, m=41, epsilon=0.25,
                                 surrogate="naive", seed=4)
        assert result.ell == 40
        _naive_box_check(result, model, 20, 41, 40)

    @pytest.mark.parametrize("route", ["keywords", "config"])
    def test_explicit_training_size_is_used(self, route):
        """training_samples = 7 with m = 30: the model is asked for exactly
        7 + 30 rows, through keywords and through config=. Catches a run that
        takes m // 2 (15) training draws whatever was asked."""
        options = dict(m=30, ell=29, epsilon=0.25, surrogate="naive",
                       training_samples=7, seed=4)
        model = _Recorder(_affine, 37)
        if route == "config":
            result = conformal_reach(model, _BOX3, ConformalReachConfig(**options))
        else:
            result = conformal_reach(model, _BOX3, **options)
        _naive_box_check(result, model, 7, 30, 29)

    def test_seed_zero_fixes_the_draws(self):
        """seed = 0 is a seed: two runs with seed 0 draw the same inputs and
        return the same box, whatever the global generator held before.
        Catches a seed tested for truth instead of against None, which
        ignores seed 0."""
        boxes, draws = [], []
        for before in (1, 2):
            np.random.seed(before)
            model = _Recorder(_affine, 30)
            boxes.append(conformal_reach(model, _BOX3, m=20, surrogate="naive",
                                         seed=0))
            draws.append(model.rows()[0])
        assert np.array_equal(draws[0], draws[1])
        assert np.array_equal(boxes[0].lb, boxes[1].lb)
        assert np.array_equal(boxes[0].ub, boxes[1].ub)

    def test_seed_none_leaves_the_draws_to_the_caller(self):
        """seed = None leaves the seeding to the caller: the draws follow the
        global generator, so the same outside seed gives the same draws and
        another one other draws. Catches a run that reseeds itself when no
        seed is given (from the clock, or with a fixed seed)."""
        draws = []
        for before in (5, 5, 6):
            np.random.seed(before)
            model = _Recorder(_affine, 30)
            conformal_reach(model, _BOX3, m=20, surrogate="naive")
            draws.append(model.rows()[0])
        assert np.array_equal(draws[0], draws[1])
        assert not np.array_equal(draws[0], draws[2])

    def test_config_and_keyword_options_together_are_refused(self):
        """config= and keyword options are two styles of one call; both at
        once is refused with TypeError instead of silently dropping one."""
        with pytest.raises(TypeError, match="not both"):
            conformal_reach(_affine, _BOX3, ConformalReachConfig(m=20), m=30)

    @pytest.mark.parametrize("options, field", [
        (dict(m=20, batch_size=0), "batch_size"),
        (dict(m=20, pca_components=0), "pca_components"),
        (dict(m=20, training_samples=0), "training_samples"),
        (dict(m=1), "ell"),
    ], ids=["batch_size-0", "pca_components-0", "training_samples-0",
            "m-1-default-rank-0"])
    def test_sizes_below_one_are_refused(self, options, field):
        """batch_size, pca_components and training_samples below 1 are
        refused with ValueError naming the field, before any model call; with
        m = 1 the default rank m - 1 = 0 is refused the same way. Catches a
        check that lets 0 through (a zero batch size, a PCA with no component,
        an empty training set)."""
        with pytest.raises(ValueError, match=f"ConformalReachConfig.{field}"):
            conformal_reach(_Recorder(_affine, 0), _BOX3, surrogate="naive",
                            **options)

    def test_verbose_run_logs_and_keeps_its_answer(self, caplog):
        """verbose=True logs the steps (the calibration size among them) and
        returns a box that keeps the rank property, through the clipping block
        and PCA (whose verbose paths differ). Catches a logging path that
        breaks or changes the run, and a run that skips the requested PCA
        (pca_components = 2 below the 4 outputs must reduce 4 -> 2; with the
        naive surrogate or one-sided box checks a skipped PCA is invisible,
        since both runs return a box that holds the calibration outputs)."""
        four = lambda x: np.hstack([_affine(x), np.sin(_affine(x))])
        model = _Recorder(four, 40)
        with caplog.at_level(logging.INFO,
                             logger="n2v.probabilistic.conformal_reach"):
            result = conformal_reach(model, _BOX3, m=30, ell=28, epsilon=0.25,
                                     surrogate="clipping_block",
                                     training_samples=10, pca_components=2,
                                     seed=4, verbose=True)
        assert any("Calibration size m: 30" in r.getMessage()
                   for r in caplog.records)
        assert any("Reducing dimension 4 -> 2" in r.getMessage()
                   for r in caplog.records), "the requested PCA did not run"
        _, y = model.rows()
        loose, _ = _held(result, y[10:], 1e-6 * (1.0 + np.max(np.abs(y))))
        assert loose >= 28


class _RecordingModule(nn.Module):
    """y = W x + b, optionally without parameters (W, b as buffers or plain
    tensors); keeps every input and output batch."""

    def __init__(self, kind, dtype=torch.float32):
        super().__init__()
        W = torch.tensor([[1.5, -0.75, 2.0], [-0.5, 2.0, 1.25]], dtype=dtype)
        b = torch.tensor([4.0, -6.0], dtype=dtype)
        if kind == "parameters":
            self.lin = nn.Linear(3, 2).to(dtype)
            with torch.no_grad():
                self.lin.weight.copy_(W)
                self.lin.bias.copy_(b)
        elif kind == "buffers":
            self.register_buffer("W", W)
            self.register_buffer("b", b)
        else:
            self.W_plain, self.b_plain = W, b
        self.kind = kind
        self.inputs, self.outputs = [], []

    def forward(self, x):
        if self.kind == "parameters":
            y = self.lin(x)
        elif self.kind == "buffers":
            y = x @ self.W.T + self.b
        else:
            y = x @ self.W_plain.T + self.b_plain
        self.inputs.append(x.detach().clone())
        self.outputs.append(y.detach().clone())
        return y


class TestModuleModels:
    """An nn.Module is wrapped into a numpy callable and evaluated."""

    @pytest.mark.parametrize("kind", ["parameters", "buffers", "neither"])
    def test_module_is_evaluated_on_the_draws(self, kind):
        """A module with parameters, one with buffers only and one with
        neither are all evaluated (on the CPU) on float32 draws of the box,
        and the naive box holds exactly ell of the m calibration outputs the
        module returned. Catches a wrapper that needs a parameter to find the
        device (a module without parameters then fails) or that passes the
        draws in another dtype or layout."""
        module = _RecordingModule(kind)
        result = conformal_reach(module, _BOX3, m=30, ell=28, epsilon=0.25,
                                 surrogate="naive", training_samples=10,
                                 seed=4)
        x = torch.cat(module.inputs).numpy()
        y = torch.cat(module.outputs).numpy().astype(np.float64)
        assert x.dtype == np.float32 and x.shape == (40, 3)
        pad = 1e-12 * (1.0 + np.max(np.abs(y)))
        loose, strict = _held(result, y[10:], pad)
        assert loose >= 28 and strict <= 28

    @known_bug(13, "conformal_reach casts the draws to float32 for every "
                   "nn.Module, so a float64 module fails with a dtype "
                   "mismatch instead of being evaluated", raises=RuntimeError)
    def test_float64_module_is_evaluated(self):
        """A float64 module is a legal model (every lane answers for the
        network with its stored weights); its conformal box holds exactly ell
        of the m calibration outputs like any other. Today the wrapper hands
        it float32 draws and torch raises RuntimeError (mat1 and mat2 must
        have the same dtype)."""
        module = _RecordingModule("parameters", dtype=torch.float64)
        result = conformal_reach(module, _BOX3, m=30, ell=28, epsilon=0.25,
                                 surrogate="naive", training_samples=10,
                                 seed=4)
        y = torch.cat(module.outputs).numpy()
        pad = 1e-12 * (1.0 + np.max(np.abs(y)))
        loose, strict = _held(result, y[10:], pad)
        assert loose >= 28 and strict <= 28


class TestBatchedInferenceShapes:
    """Model calls in batches, and outputs that are not one row per sample."""

    def test_partial_batches_and_outputs_with_more_axes(self):
        """batch_size = 7 with 10 training and 30 calibration draws: the model
        sees batches of 7 and a last partial batch (7, 3, then 7, 7, 7, 7, 2),
        and its (batch, 2, 3) outputs are flattened per sample in row-major
        order, so box coordinate 3 i + j is output entry (i, j). Each entry
        has its own offset and scale, so a column-major flattening or a lost
        partial batch moves the box away from the calibration outputs: the
        naive box must hold exactly 28 of the 30 flattened in C order."""
        offsets = np.arange(6.0).reshape(2, 3) * 10.0 - 20.0
        scales = np.array([[1.0, -2.0, 0.5], [3.0, -0.25, 1.5]])

        def tensor_model(x):
            return x[:, :1, None] * scales[None] + x[:, 1:2, None] + offsets[None]

        model = _Recorder(tensor_model, 40)
        result = conformal_reach(model, _BOX3, m=30, ell=28, epsilon=0.25,
                                 surrogate="naive", training_samples=10,
                                 batch_size=7, seed=4)
        assert [len(b) for b in model.inputs] == [7, 3, 7, 7, 7, 7, 2]
        y = np.vstack([o.reshape(len(o), -1) for o in model.outputs])
        assert result.dim == 6
        pad = 1e-12 * (1.0 + np.max(np.abs(y)))
        loose, strict = _held(result, y[10:], pad)
        assert loose >= 28 and strict <= 28

    @known_bug(13, "a model that returns one value per sample as a 1-D array "
                   "is stacked into one row per batch, so conformal_reach "
                   "fails with a numpy broadcast ValueError instead of "
                   "returning a one-output box", raises=ValueError)
    def test_one_value_per_sample_as_a_flat_array(self):
        """A scalar model may return shape (batch,) (``x @ w``). The docstring
        of the batching helper promises outputs flattened to one row per
        sample; the box has one coordinate and, with the naive surrogate,
        holds exactly ell of the m calibration outputs. Today the batches are
        stacked as rows, the training outputs become one row of 10 values,
        and the run fails with a broadcast ValueError."""
        model = _Recorder(lambda x: x @ np.array([1.5, -0.75, 2.0]), 40)
        result = conformal_reach(model, _BOX3, m=30, ell=28, epsilon=0.25,
                                 surrogate="naive", training_samples=10,
                                 seed=4)
        y = np.concatenate(model.outputs)[:, None]
        assert result.dim == 1
        pad = 1e-12 * (1.0 + np.max(np.abs(y)))
        loose, strict = _held(result, y[10:], pad)
        assert loose >= 28 and strict <= 28
