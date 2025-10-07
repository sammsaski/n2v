"""Tests for Sigmoid layer reachability operations."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops import sigmoid_reach
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestSigmoidStarApprox:
    """Tests for Sigmoid approx with Star sets."""

    def test_positive_bounds(self):
        """All-positive input: output in (0.5, 1)."""
        lb = np.array([[1.0], [2.0]])
        ub = np.array([[2.0], [3.0]])
        star = Star.from_bounds(lb, ub)

        result = sigmoid_reach.sigmoid_star_approx([star])

        assert len(result) == 1
        pytest.assert_star_valid(result[0])
        out_lb, out_ub = result[0].estimate_ranges()
        # sigmoid(1) ~ 0.731, sigmoid(3) ~ 0.953
        assert np.all(out_lb > 0.5)
        assert np.all(out_ub < 1.0)

    def test_negative_bounds(self):
        """All-negative input: output in (0, 0.5)."""
        lb = np.array([[-3.0], [-2.0]])
        ub = np.array([[-1.0], [-0.5]])
        star = Star.from_bounds(lb, ub)

        result = sigmoid_reach.sigmoid_star_approx([star])

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        assert np.all(out_lb > 0.0)
        assert np.all(out_ub < 0.5 + 1e-6)

    def test_mixed_bounds(self):
        """Mixed bounds spanning inflection point."""
        lb = np.array([[-2.0]])
        ub = np.array([[2.0]])
        star = Star.from_bounds(lb, ub)

        result = sigmoid_reach.sigmoid_star_approx([star])

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        # sigmoid(-2) ~ 0.119, sigmoid(2) ~ 0.881
        assert out_lb[0, 0] < 0.12 + 0.05
        assert out_ub[0, 0] > 0.88 - 0.05

    def test_constant_input(self):
        """Constant input (lb == ub): output is sigmoid(value)."""
        val = 1.0
        lb = np.array([[val]])
        ub = np.array([[val]])
        star = Star.from_bounds(lb, ub)

        result = sigmoid_reach.sigmoid_star_approx([star])

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        expected = 1.0 / (1.0 + np.exp(-val))
        np.testing.assert_allclose(out_lb[0, 0], expected, atol=1e-5)
        np.testing.assert_allclose(out_ub[0, 0], expected, atol=1e-5)

    def test_dispatch(self):
        """Sigmoid routed through dispatcher."""
        star = Star.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        layer = nn.Sigmoid()
        result = reach_layer(layer, [star], method='approx')
        assert len(result) == 1

    def test_exact_warns_and_uses_approx(self):
        """method='exact' emits warning and falls back to approx."""
        star = Star.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        layer = nn.Sigmoid()
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = reach_layer(layer, [star], method='exact')
            assert len(result) == 1
            assert any("approx" in str(warning.message).lower() for warning in w)


class TestSigmoidZono:
    """Tests for Sigmoid with Zonotope."""

    def test_zono_bounds(self):
        """Sigmoid Zono output should be bounded by [sigma(lb), sigma(ub)]."""
        lb = np.array([[-2.0], [0.0]])
        ub = np.array([[1.0], [3.0]])
        zono = Zono.from_bounds(lb, ub)

        result = sigmoid_reach.sigmoid_zono_approx([zono])

        assert len(result) == 1
        pytest.assert_zono_valid(result[0])
        out_lb, out_ub = result[0].get_bounds()
        from n2v.nn.layer_ops.sigmoid_reach import _sigmoid
        expected_lb = _sigmoid(lb)
        expected_ub = _sigmoid(ub)
        assert np.all(out_lb <= expected_lb + 1e-5)
        assert np.all(out_ub >= expected_ub - 1e-5)

    def test_dispatch_zono(self):
        """Zono routed through dispatcher."""
        zono = Zono.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        layer = nn.Sigmoid()
        result = reach_layer(layer, [zono], method='approx')
        assert len(result) == 1


class TestSigmoidBox:
    """Tests for Sigmoid with Box."""

    def test_box_bounds(self):
        """Sigmoid Box: output = [sigma(lb), sigma(ub)]."""
        lb = np.array([[-2.0], [0.0], [1.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        box = Box(lb, ub)

        result = sigmoid_reach.sigmoid_box([box])

        from n2v.nn.layer_ops.sigmoid_reach import _sigmoid
        np.testing.assert_allclose(result[0].lb, _sigmoid(lb), atol=1e-10)
        np.testing.assert_allclose(result[0].ub, _sigmoid(ub), atol=1e-10)

    def test_dispatch_box(self):
        """Box routed through dispatcher."""
        box = Box(np.array([[-1.0]]), np.array([[1.0]]))
        layer = nn.Sigmoid()
        result = reach_layer(layer, [box], method='approx')
        assert len(result) == 1
