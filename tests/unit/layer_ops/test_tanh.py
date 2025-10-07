"""Tests for Tanh layer reachability operations."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops import tanh_reach
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestTanhStarApprox:
    """Tests for Tanh approx with Star sets."""

    def test_positive_bounds(self):
        """All-positive input: output in (0, 1)."""
        lb = np.array([[0.5], [1.0]])
        ub = np.array([[2.0], [3.0]])
        star = Star.from_bounds(lb, ub)

        result = tanh_reach.tanh_star_approx([star])

        assert len(result) == 1
        pytest.assert_star_valid(result[0])
        out_lb, out_ub = result[0].estimate_ranges()
        assert np.all(out_lb > 0.0)
        assert np.all(out_ub < 1.0)

    def test_negative_bounds(self):
        """All-negative input: output in (-1, 0)."""
        lb = np.array([[-3.0], [-2.0]])
        ub = np.array([[-0.5], [-0.5]])
        star = Star.from_bounds(lb, ub)

        result = tanh_reach.tanh_star_approx([star])

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        assert np.all(out_lb > -1.0)
        assert np.all(out_ub < 0.0 + 1e-6)

    def test_mixed_bounds(self):
        """Mixed bounds spanning inflection point."""
        lb = np.array([[-2.0]])
        ub = np.array([[2.0]])
        star = Star.from_bounds(lb, ub)

        result = tanh_reach.tanh_star_approx([star])

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        # tanh(-2) ~ -0.964, tanh(2) ~ 0.964
        assert out_lb[0, 0] < -0.96 + 0.05
        assert out_ub[0, 0] > 0.96 - 0.05

    def test_constant_input(self):
        """Constant input: output is tanh(value)."""
        val = 1.0
        lb = np.array([[val]])
        ub = np.array([[val]])
        star = Star.from_bounds(lb, ub)

        result = tanh_reach.tanh_star_approx([star])

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        expected = np.tanh(val)
        np.testing.assert_allclose(out_lb[0, 0], expected, atol=1e-5)
        np.testing.assert_allclose(out_ub[0, 0], expected, atol=1e-5)

    def test_dispatch(self):
        """Tanh routed through dispatcher."""
        star = Star.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        layer = nn.Tanh()
        result = reach_layer(layer, [star], method='approx')
        assert len(result) == 1

    def test_exact_warns_and_uses_approx(self):
        """method='exact' emits warning and falls back to approx."""
        star = Star.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        layer = nn.Tanh()
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = reach_layer(layer, [star], method='exact')
            assert len(result) == 1
            assert any("approx" in str(warning.message).lower() for warning in w)


class TestTanhZono:
    """Tests for Tanh with Zonotope."""

    def test_zono_bounds(self):
        """Tanh Zono output bounded by [tanh(lb), tanh(ub)]."""
        lb = np.array([[-2.0], [0.0]])
        ub = np.array([[1.0], [3.0]])
        zono = Zono.from_bounds(lb, ub)

        result = tanh_reach.tanh_zono_approx([zono])

        assert len(result) == 1
        pytest.assert_zono_valid(result[0])
        out_lb, out_ub = result[0].get_bounds()
        assert np.all(out_lb <= np.tanh(lb) + 1e-5)
        assert np.all(out_ub >= np.tanh(ub) - 1e-5)

    def test_dispatch_zono(self):
        """Zono routed through dispatcher."""
        zono = Zono.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        layer = nn.Tanh()
        result = reach_layer(layer, [zono], method='approx')
        assert len(result) == 1


class TestTanhBox:
    """Tests for Tanh with Box."""

    def test_box_bounds(self):
        """Tanh Box: output = [tanh(lb), tanh(ub)]."""
        lb = np.array([[-2.0], [0.0], [1.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        box = Box(lb, ub)

        result = tanh_reach.tanh_box([box])

        np.testing.assert_allclose(result[0].lb, np.tanh(lb), atol=1e-10)
        np.testing.assert_allclose(result[0].ub, np.tanh(ub), atol=1e-10)

    def test_dispatch_box(self):
        """Box routed through dispatcher."""
        box = Box(np.array([[-1.0]]), np.array([[1.0]]))
        layer = nn.Tanh()
        result = reach_layer(layer, [box], method='approx')
        assert len(result) == 1
