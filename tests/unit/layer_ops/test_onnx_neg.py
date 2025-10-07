"""Tests for OnnxNeg dispatch — negation should multiply all set values by -1."""

import numpy as np
import pytest

from n2v.nn.layer_ops.dispatcher import reach_layer

# Import OnnxNeg (skip tests if onnx2torch not installed)
try:
    from onnx2torch.node_converters.neg import OnnxNeg
except ImportError:
    OnnxNeg = None

pytestmark = pytest.mark.skipif(OnnxNeg is None, reason="onnx2torch not installed")


class TestOnnxNegStar:
    """OnnxNeg should negate Star sets (multiply V by -1)."""

    def test_neg_star_center_is_negated(self, simple_star):
        layer = OnnxNeg()
        result = reach_layer(layer, [simple_star], method='exact')
        assert len(result) == 1
        neg_star = result[0]
        # Center (V[:, 0]) should be negated
        np.testing.assert_allclose(neg_star.V[:, 0], -simple_star.V[:, 0])

    def test_neg_star_generators_are_negated(self, simple_star):
        layer = OnnxNeg()
        result = reach_layer(layer, [simple_star], method='exact')
        neg_star = result[0]
        # All generators should be negated
        np.testing.assert_allclose(neg_star.V[:, 1:], -simple_star.V[:, 1:])

    def test_neg_star_constraints_preserved(self, simple_star):
        layer = OnnxNeg()
        result = reach_layer(layer, [simple_star], method='exact')
        neg_star = result[0]
        np.testing.assert_array_equal(neg_star.C, simple_star.C)
        np.testing.assert_array_equal(neg_star.d, simple_star.d)

    def test_neg_star_double_negation_is_identity(self, simple_star):
        layer = OnnxNeg()
        result = reach_layer(layer, [simple_star], method='exact')
        result2 = reach_layer(layer, result, method='exact')
        np.testing.assert_allclose(result2[0].V, simple_star.V)


class TestOnnxNegZono:
    """OnnxNeg should negate Zonotope sets."""

    def test_neg_zono_center_is_negated(self, simple_zono):
        layer = OnnxNeg()
        result = reach_layer(layer, [simple_zono], method='approx')
        assert len(result) == 1
        np.testing.assert_allclose(result[0].c, -simple_zono.c)

    def test_neg_zono_generators_are_negated(self, simple_zono):
        layer = OnnxNeg()
        result = reach_layer(layer, [simple_zono], method='approx')
        np.testing.assert_allclose(result[0].V, -simple_zono.V)


class TestOnnxNegBox:
    """OnnxNeg should negate Box sets (swap and negate bounds)."""

    def test_neg_box_bounds_swapped(self, simple_box):
        layer = OnnxNeg()
        result = reach_layer(layer, [simple_box], method='approx')
        assert len(result) == 1
        neg_box = result[0]
        # neg([0,1]) = [-1, 0]
        np.testing.assert_allclose(neg_box.lb, -simple_box.ub)
        np.testing.assert_allclose(neg_box.ub, -simple_box.lb)


class TestOnnxNegOtherTypes:
    """OnnxNeg should work for Hexatope and Octatope."""

    def test_neg_hexatope(self, simple_hexatope):
        layer = OnnxNeg()
        result = reach_layer(layer, [simple_hexatope], method='approx')
        assert len(result) == 1

    def test_neg_octatope(self, simple_octatope):
        layer = OnnxNeg()
        result = reach_layer(layer, [simple_octatope], method='approx')
        assert len(result) == 1
