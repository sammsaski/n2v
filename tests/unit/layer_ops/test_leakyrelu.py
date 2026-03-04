"""Tests for LeakyReLU layer reachability operations."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops import leakyrelu_reach
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestLeakyReLUStarExact:
    """Tests for LeakyReLU exact with Star sets."""

    def test_always_active(self):
        """All-positive bounds: identity."""
        lb = np.array([[1.0], [2.0]])
        ub = np.array([[2.0], [3.0]])
        star = Star.from_bounds(lb, ub)

        result = leakyrelu_reach.leakyrelu_star_exact([star], gamma=0.1)

        assert len(result) == 1
        pytest.assert_star_valid(result[0])
        out_lb, out_ub = result[0].estimate_ranges()
        np.testing.assert_allclose(out_lb, lb, atol=1e-6)
        np.testing.assert_allclose(out_ub, ub, atol=1e-6)

    def test_always_inactive(self):
        """All-negative bounds: scale by gamma."""
        lb = np.array([[-2.0], [-3.0]])
        ub = np.array([[-1.0], [-2.0]])
        star = Star.from_bounds(lb, ub)
        gamma = 0.1

        result = leakyrelu_reach.leakyrelu_star_exact([star], gamma=gamma)

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        np.testing.assert_allclose(out_lb, gamma * lb, atol=1e-6)
        np.testing.assert_allclose(out_ub, gamma * ub, atol=1e-6)

    def test_crossing_splits(self):
        """Crossing neuron: should split into 2."""
        lb = np.array([[-1.0], [1.0]])
        ub = np.array([[1.0], [2.0]])
        star = Star.from_bounds(lb, ub)

        result = leakyrelu_reach.leakyrelu_star_exact([star], gamma=0.1)

        assert len(result) == 2
        for s in result:
            pytest.assert_star_valid(s)

    def test_dispatch(self):
        """LeakyReLU routed through dispatcher."""
        lb = np.array([[-1.0], [1.0]])
        ub = np.array([[1.0], [2.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.2)

        result = reach_layer(layer, [star], method='exact')
        assert len(result) == 2

    def test_gamma_zero_equals_relu(self):
        """gamma=0 should match ReLU behavior."""
        from n2v.nn.layer_ops import relu_reach
        lb = np.array([[-1.0], [-0.5]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        relu_result = relu_reach.relu_star_exact([star])
        leaky_result = leakyrelu_reach.leakyrelu_star_exact([star], gamma=0.0)

        assert len(relu_result) == len(leaky_result)


class TestLeakyReLUStarApprox:
    """Tests for LeakyReLU approx with Star sets."""

    def test_no_splitting(self):
        """Approx should not split."""
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        result = leakyrelu_reach.leakyrelu_star_approx([star], gamma=0.1)

        assert len(result) == 1
        pytest.assert_star_valid(result[0])

    def test_dispatch_approx(self):
        """Approx routed through dispatcher."""
        lb = np.array([[-1.0]])
        ub = np.array([[1.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.1)

        result = reach_layer(layer, [star], method='approx')
        assert len(result) == 1


class TestLeakyReLUZono:
    """Tests for LeakyReLU with Zonotope."""

    def test_always_active(self):
        """All-positive Zono: identity."""
        zono = Zono.from_bounds(np.array([[1.0], [2.0]]), np.array([[2.0], [3.0]]))
        result = leakyrelu_reach.leakyrelu_zono_approx([zono], gamma=0.1)
        assert len(result) == 1
        np.testing.assert_allclose(result[0].c, zono.c, atol=1e-6)

    def test_dispatch_zono(self):
        """Zono routed through dispatcher."""
        zono = Zono.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        layer = nn.LeakyReLU(negative_slope=0.1)
        result = reach_layer(layer, [zono], method='approx')
        assert len(result) == 1
        pytest.assert_zono_valid(result[0])


class TestLeakyReLUBox:
    """Tests for LeakyReLU with Box."""

    def test_crossing_zero(self):
        """Box crossing zero: lb scaled, ub kept."""
        lb = np.array([[-1.0], [-2.0], [1.0]])
        ub = np.array([[1.0], [0.5], [2.0]])
        box = Box(lb, ub)
        gamma = 0.2

        result = leakyrelu_reach.leakyrelu_box([box], gamma=gamma)

        assert len(result) == 1
        np.testing.assert_allclose(result[0].lb, np.array([[-0.2], [-0.4], [1.0]]), atol=1e-6)
        np.testing.assert_allclose(result[0].ub, np.array([[1.0], [0.5], [2.0]]), atol=1e-6)

    def test_dispatch_box(self):
        """Box routed through dispatcher."""
        box = Box(np.array([[-1.0]]), np.array([[1.0]]))
        layer = nn.LeakyReLU(negative_slope=0.1)
        result = reach_layer(layer, [box], method='approx')
        assert len(result) == 1
