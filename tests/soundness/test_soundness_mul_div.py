"""
Soundness tests for element-wise Mul/Div by constant.

Tests verify that for random points sampled from input set bounds,
applying the Mul/Div operation to each point produces outputs
contained within the reachable set obtained via _mul_sets_by_constant.
"""

import numpy as np
from n2v.sets import Star, Zono, Box
from n2v.nn.reach import _mul_sets_by_constant


class TestMulStarSoundness:
    """Soundness: Star Mul by mixed positive/negative scale."""

    def test_mul_star_soundness(self):
        """Sample 200 points, multiply by mixed scale, verify containment."""
        lb = np.array([0.0, -1.0, 0.5, -2.0])
        ub = np.array([1.0, 1.0, 2.0, 0.0])
        star = Star.from_bounds(lb, ub)

        scale = np.array([2.0, -3.0, 0.5, -1.0])

        result = _mul_sets_by_constant([star], scale)
        out = result[0]

        lb_out, ub_out = out.estimate_ranges()

        np.random.seed(42)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(4,))
            point = lb + alpha * (ub - lb)
            scaled_point = scale * point

            assert np.all(scaled_point >= lb_out.flatten() - 1e-5), (
                f"Output {scaled_point} below lower bound {lb_out.flatten()}"
            )
            assert np.all(scaled_point <= ub_out.flatten() + 1e-5), (
                f"Output {scaled_point} above upper bound {ub_out.flatten()}"
            )


class TestDivStarSoundness:
    """Soundness: Star Div by mixed divisors."""

    def test_div_star_soundness(self):
        """Sample 200 points, divide by mixed divisors, verify containment."""
        lb = np.array([1.0, -2.0, 0.0])
        ub = np.array([3.0, 2.0, 4.0])
        star = Star.from_bounds(lb, ub)

        divisor = np.array([2.0, -0.5, 4.0])
        reciprocal = 1.0 / divisor

        result = _mul_sets_by_constant([star], reciprocal)
        out = result[0]

        lb_out, ub_out = out.estimate_ranges()

        np.random.seed(123)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(3,))
            point = lb + alpha * (ub - lb)
            divided_point = point / divisor

            assert np.all(divided_point >= lb_out.flatten() - 1e-5), (
                f"Output {divided_point} below lower bound {lb_out.flatten()}"
            )
            assert np.all(divided_point <= ub_out.flatten() + 1e-5), (
                f"Output {divided_point} above upper bound {ub_out.flatten()}"
            )


class TestMulZonoSoundness:
    """Soundness: Zono Mul by mixed scale."""

    def test_mul_zono_soundness(self):
        """Sample 200 points, multiply by mixed scale, verify containment."""
        lb = np.array([-1.0, 0.0, 1.0, -0.5])
        ub = np.array([1.0, 2.0, 3.0, 0.5])
        zono = Zono.from_bounds(lb, ub)

        scale = np.array([-2.0, 0.5, 3.0, -1.0])

        result = _mul_sets_by_constant([zono], scale)
        out = result[0]

        lb_out, ub_out = out.get_bounds()

        np.random.seed(99)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(4,))
            point = lb + alpha * (ub - lb)
            scaled_point = scale * point

            assert np.all(scaled_point >= lb_out.flatten() - 1e-5), (
                f"Output {scaled_point} below lower bound {lb_out.flatten()}"
            )
            assert np.all(scaled_point <= ub_out.flatten() + 1e-5), (
                f"Output {scaled_point} above upper bound {ub_out.flatten()}"
            )


class TestMulBoxNegativeScaleSoundness:
    """Soundness: Box Mul with negative scale factors."""

    def test_mul_box_negative_scale_soundness(self):
        """Sample 200 points, multiply by negative scale, verify containment."""
        lb = np.array([1.0, -1.0, 2.0])
        ub = np.array([3.0, 1.0, 5.0])
        box = Box(lb, ub)

        scale = np.array([-2.0, -0.5, -3.0])

        result = _mul_sets_by_constant([box], scale)
        out = result[0]

        np.random.seed(7)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(3,))
            point = lb + alpha * (ub - lb)
            scaled_point = scale * point

            assert np.all(scaled_point >= out.lb.flatten() - 1e-5), (
                f"Output {scaled_point} below lower bound {out.lb.flatten()}"
            )
            assert np.all(scaled_point <= out.ub.flatten() + 1e-5), (
                f"Output {scaled_point} above upper bound {out.ub.flatten()}"
            )
