"""Tests for ReLU reachability with precomputed bounds."""

import pytest
import numpy as np
import torch.nn as nn
from n2v.sets import Star, Zono
from n2v.nn.layer_ops.relu_reach import relu_star_exact, relu_star_approx


class TestReLUStarExactWithPrecomputedBounds:
    """Verify relu_star_exact with precomputed_bounds matches without."""

    def test_precomputed_bounds_does_not_change_result(self):
        """Results with and without precomputed bounds should be equivalent."""
        lb = np.array([[-1.0], [0.5], [-2.0], [1.0]])
        ub = np.array([[1.0], [1.5], [-0.5], [3.0]])
        star = Star.from_bounds(lb, ub)

        # Without precomputed bounds
        result_without = relu_star_exact([star])

        # With matching precomputed bounds (same as Zono estimate)
        zono = Zono.from_bounds(lb, ub)
        pre_lb, pre_ub = zono.estimate_ranges()
        result_with = relu_star_exact([star], precomputed_bounds=(pre_lb, pre_ub))

        # Should produce same number of output stars
        assert len(result_with) == len(result_without)

        # Output ranges should match
        for s_with, s_without in zip(
            sorted(result_with, key=lambda s: s.V[0, 0]),
            sorted(result_without, key=lambda s: s.V[0, 0]),
        ):
            lb_w, ub_w = s_with.get_ranges()
            lb_wo, ub_wo = s_without.get_ranges()
            np.testing.assert_allclose(lb_w, lb_wo, atol=1e-6)
            np.testing.assert_allclose(ub_w, ub_wo, atol=1e-6)

    def test_precomputed_bounds_skips_stable_neurons(self):
        """Neurons proven stable by precomputed bounds should not be split."""
        # Input: [-1, 1] x [-1, 1], but precomputed bounds say neuron 1 is always active
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Without: both neurons cross zero -> 2^2 = 4 stars max
        result_without = relu_star_exact([star])

        # With: neuron 1 is marked as always active (lb=0.5, ub=1.5)
        pre_lb = np.array([[-1.0], [0.5]])  # neuron 1: lb >= 0
        pre_ub = np.array([[1.0], [1.5]])
        result_with = relu_star_exact([star], precomputed_bounds=(pre_lb, pre_ub))

        # With precomputed bounds, only neuron 0 needs splitting -> at most 2 stars
        assert len(result_with) <= len(result_without)

    def test_precomputed_bounds_all_stable(self):
        """If all neurons are stable, no splitting should occur."""
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Precomputed bounds say: neuron 0 always active, neuron 1 always inactive
        pre_lb = np.array([[0.5], [-1.5]])
        pre_ub = np.array([[1.5], [-0.5]])
        result = relu_star_exact([star], precomputed_bounds=(pre_lb, pre_ub))

        # No splitting -- should return 1 star
        assert len(result) == 1

    def test_none_precomputed_bounds_is_noop(self):
        """Passing precomputed_bounds=None should behave identically to default."""
        lb = np.array([[-1.0], [0.5]])
        ub = np.array([[1.0], [1.5]])
        star = Star.from_bounds(lb, ub)

        result_default = relu_star_exact([star])
        result_none = relu_star_exact([star], precomputed_bounds=None)

        assert len(result_default) == len(result_none)


class TestReLUStarApproxWithPrecomputedBounds:
    """Verify relu_star_approx with precomputed_bounds."""

    def test_approx_precomputed_bounds_is_sound(self):
        """Approx with precomputed bounds must still contain all exact results."""
        lb = np.array([[-1.0], [-0.5], [0.5]])
        ub = np.array([[1.0], [0.5], [1.5]])
        star = Star.from_bounds(lb, ub)

        pre_lb = lb.copy()
        pre_ub = ub.copy()

        result_approx = relu_star_approx(
            [star], precomputed_bounds=(pre_lb, pre_ub)
        )
        result_exact = relu_star_exact([star])

        # Approx output should contain all exact output ranges
        approx_lb, approx_ub = result_approx[0].estimate_ranges()
        for exact_star in result_exact:
            exact_lb, exact_ub = exact_star.get_ranges()
            assert np.all(approx_lb <= exact_lb + 1e-6), \
                "Approx lower bound should be <= exact lower bound"
            assert np.all(approx_ub >= exact_ub - 1e-6), \
                "Approx upper bound should be >= exact upper bound"

    def test_approx_none_precomputed_bounds(self):
        """Passing None should behave identically to default."""
        lb = np.array([[-1.0], [0.5]])
        ub = np.array([[1.0], [1.5]])
        star = Star.from_bounds(lb, ub)

        result_default = relu_star_approx([star])
        result_none = relu_star_approx([star], precomputed_bounds=None)

        assert len(result_default) == len(result_none)
        lb_d, ub_d = result_default[0].estimate_ranges()
        lb_n, ub_n = result_none[0].estimate_ranges()
        np.testing.assert_allclose(lb_d, lb_n, atol=1e-10)
        np.testing.assert_allclose(ub_d, ub_n, atol=1e-10)
