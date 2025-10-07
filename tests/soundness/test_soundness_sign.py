"""Soundness tests for Sign activation reachability."""

import numpy as np
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.sign_reach import sign_star, sign_zono, sign_box


class TestSignStarSoundness:
    def test_random_samples_approx(self):
        np.random.seed(42)
        center = np.random.randn(5, 1)
        star = Star.from_bounds(center - 1.0, center + 1.0)
        result = sign_star(None, [star], method='approx')
        lb, ub = result[0].get_ranges()
        for _ in range(500):
            x = np.random.uniform((center - 1.0).flatten(), (center + 1.0).flatten())
            y = np.sign(x).reshape(-1, 1)
            assert np.all(y >= lb - 1e-6)
            assert np.all(y <= ub + 1e-6)

    def test_approx_contains_exact(self):
        star = Star.from_bounds(
            np.array([[-0.5], [0.2], [-1.0]]),
            np.array([[0.3], [0.8], [-0.1]])
        )
        approx_result = sign_star(None, [star], method='approx')
        exact_result = sign_star(None, [star], method='exact')
        approx_lb, approx_ub = approx_result[0].get_ranges()
        for exact_star in exact_result:
            exact_lb, exact_ub = exact_star.get_ranges()
            assert np.all(exact_lb >= approx_lb - 1e-6)
            assert np.all(exact_ub <= approx_ub + 1e-6)


class TestSignZonoSoundness:
    def test_random_samples(self):
        np.random.seed(42)
        center = np.random.randn(5, 1)
        zono = Zono.from_bounds(center - 1.0, center + 1.0)
        result = sign_zono([zono])
        lb, ub = result[0].get_bounds()
        for _ in range(500):
            x = np.random.uniform((center - 1.0).flatten(), (center + 1.0).flatten())
            y = np.sign(x).reshape(-1, 1)
            assert np.all(y >= lb - 1e-6)
            assert np.all(y <= ub + 1e-6)


class TestSignBoxSoundness:
    def test_random_samples(self):
        np.random.seed(42)
        center = np.random.randn(5, 1)
        box = Box(center - 1.0, center + 1.0)
        result = sign_box([box])
        for _ in range(500):
            x = np.random.uniform((center - 1.0).flatten(), (center + 1.0).flatten())
            y = np.sign(x).reshape(-1, 1)
            assert np.all(y >= result[0].lb - 1e-6)
            assert np.all(y <= result[0].ub + 1e-6)
