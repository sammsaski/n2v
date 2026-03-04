"""Tests for LeakyReLU reachability with precomputed bounds."""

import numpy as np
from n2v.sets import Star, Zono
from n2v.nn.layer_ops.leakyrelu_reach import (
    leakyrelu_star_exact, leakyrelu_star_approx,
)


class TestLeakyReLUWithPrecomputedBounds:

    def test_exact_precomputed_matches_without(self):
        """Exact results with and without precomputed bounds should match."""
        lb = np.array([[-1.0], [0.5], [-2.0]])
        ub = np.array([[1.0], [1.5], [-0.5]])
        star = Star.from_bounds(lb, ub)

        result_without = leakyrelu_star_exact([star], gamma=0.1)

        zono = Zono.from_bounds(lb, ub)
        pre_lb, pre_ub = zono.estimate_ranges()
        result_with = leakyrelu_star_exact(
            [star], gamma=0.1, precomputed_bounds=(pre_lb, pre_ub)
        )

        assert len(result_with) == len(result_without)

    def test_approx_precomputed_is_sound(self):
        """Approx with precomputed bounds should contain exact results."""
        lb = np.array([[-1.0], [-0.5], [0.5]])
        ub = np.array([[1.0], [0.5], [1.5]])
        star = Star.from_bounds(lb, ub)

        result_approx = leakyrelu_star_approx(
            [star], gamma=0.1, precomputed_bounds=(lb, ub)
        )
        result_exact = leakyrelu_star_exact([star], gamma=0.1)

        approx_lb, approx_ub = result_approx[0].estimate_ranges()
        for exact_star in result_exact:
            exact_lb, exact_ub = exact_star.get_ranges()
            assert np.all(approx_lb <= exact_lb + 1e-6)
            assert np.all(approx_ub >= exact_ub - 1e-6)

    def test_none_is_noop(self):
        """precomputed_bounds=None should not change behavior."""
        lb = np.array([[-1.0], [0.5]])
        ub = np.array([[1.0], [1.5]])
        star = Star.from_bounds(lb, ub)

        r1 = leakyrelu_star_exact([star], gamma=0.1)
        r2 = leakyrelu_star_exact([star], gamma=0.1, precomputed_bounds=None)
        assert len(r1) == len(r2)
