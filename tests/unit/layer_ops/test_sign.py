"""Tests for Sign activation reachability."""

import numpy as np
import pytest
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.sign_reach import sign_star, sign_zono, sign_box


class TestSignBox:
    def test_all_positive(self):
        box = Box(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        result = sign_box([box])
        np.testing.assert_allclose(result[0].lb, np.array([[1.0], [1.0]]))
        np.testing.assert_allclose(result[0].ub, np.array([[1.0], [1.0]]))

    def test_all_negative(self):
        box = Box(np.array([[-3.0], [-2.0]]), np.array([[-1.0], [-0.5]]))
        result = sign_box([box])
        np.testing.assert_allclose(result[0].lb, np.array([[-1.0], [-1.0]]))
        np.testing.assert_allclose(result[0].ub, np.array([[-1.0], [-1.0]]))

    def test_crossing_zero(self):
        box = Box(np.array([[-1.0]]), np.array([[1.0]]))
        result = sign_box([box])
        np.testing.assert_allclose(result[0].lb, np.array([[-1.0]]))
        np.testing.assert_allclose(result[0].ub, np.array([[1.0]]))

    def test_mixed_dims(self):
        box = Box(
            np.array([[1.0], [-3.0], [-1.0]]),
            np.array([[2.0], [-1.0], [1.0]])
        )
        result = sign_box([box])
        np.testing.assert_allclose(result[0].lb, np.array([[1.0], [-1.0], [-1.0]]))
        np.testing.assert_allclose(result[0].ub, np.array([[1.0], [-1.0], [1.0]]))


class TestSignZono:
    def test_all_positive(self):
        zono = Zono.from_bounds(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        result = sign_zono([zono])
        lb, ub = result[0].get_bounds()
        np.testing.assert_allclose(lb, np.array([[1.0], [1.0]]))
        np.testing.assert_allclose(ub, np.array([[1.0], [1.0]]))

    def test_crossing_soundness(self):
        zono = Zono.from_bounds(np.array([[-1.0], [-0.5]]), np.array([[1.0], [0.5]]))
        result = sign_zono([zono])
        lb, ub = result[0].get_bounds()
        assert np.all(lb <= -1.0 + 1e-10)
        assert np.all(ub >= 1.0 - 1e-10)


class TestSignStarApprox:
    def test_all_positive_exact_output(self):
        star = Star.from_bounds(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        result = sign_star(None, [star], method='approx')
        lb, ub = result[0].get_ranges()
        np.testing.assert_allclose(lb, np.array([[1.0], [1.0]]), atol=1e-6)
        np.testing.assert_allclose(ub, np.array([[1.0], [1.0]]), atol=1e-6)

    def test_all_negative_exact_output(self):
        star = Star.from_bounds(np.array([[-4.0], [-3.0]]), np.array([[-1.0], [-0.5]]))
        result = sign_star(None, [star], method='approx')
        lb, ub = result[0].get_ranges()
        np.testing.assert_allclose(lb, np.array([[-1.0], [-1.0]]), atol=1e-6)
        np.testing.assert_allclose(ub, np.array([[-1.0], [-1.0]]), atol=1e-6)

    def test_crossing_contains_both_values(self):
        star = Star.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        result = sign_star(None, [star], method='approx')
        lb, ub = result[0].get_ranges()
        assert lb[0, 0] <= -1.0 + 1e-6
        assert ub[0, 0] >= 1.0 - 1e-6

    def test_mixed_dimensions(self):
        star = Star.from_bounds(
            np.array([[1.0], [-3.0], [-1.0]]),
            np.array([[2.0], [-1.0], [1.0]])
        )
        result = sign_star(None, [star], method='approx')
        lb, ub = result[0].get_ranges()
        np.testing.assert_allclose(lb[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(ub[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(lb[1], -1.0, atol=1e-6)
        np.testing.assert_allclose(ub[1], -1.0, atol=1e-6)
        assert lb[2, 0] <= -1.0 + 1e-6
        assert ub[2, 0] >= 1.0 - 1e-6


class TestSignStarExact:
    def test_all_positive_single_star(self):
        star = Star.from_bounds(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        result = sign_star(None, [star], method='exact')
        assert len(result) == 1
        lb, ub = result[0].get_ranges()
        np.testing.assert_allclose(lb, 1.0, atol=1e-6)
        np.testing.assert_allclose(ub, 1.0, atol=1e-6)

    def test_one_crossing_two_stars(self):
        star = Star.from_bounds(np.array([[1.0], [-1.0]]), np.array([[2.0], [1.0]]))
        result = sign_star(None, [star], method='exact')
        assert len(result) == 2
        for s in result:
            lb, ub = s.get_ranges()
            np.testing.assert_allclose(lb[0], 1.0, atol=1e-6)
            np.testing.assert_allclose(ub[0], 1.0, atol=1e-6)
            assert abs(lb[1, 0] - ub[1, 0]) < 1e-6

    def test_soundness_random(self):
        np.random.seed(42)
        star = Star.from_bounds(
            np.array([[-1.0], [-0.5], [0.5]]),
            np.array([[1.0], [0.5], [1.5]])
        )
        result = sign_star(None, [star], method='exact')
        for _ in range(200):
            x = np.random.uniform([-1.0, -0.5, 0.5], [1.0, 0.5, 1.5])
            y = np.sign(x).reshape(-1, 1)
            contained = any(
                np.all(y >= s.get_ranges()[0] - 1e-6) and np.all(y <= s.get_ranges()[1] + 1e-6)
                for s in result
            )
            assert contained, f"sign({x}) = {y.flatten()} not in any output star"
