"""Tests for OnnxTranspose dispatch — permutes dimensions of flat sets."""

import numpy as np
import pytest

from n2v.nn.layer_ops.dispatcher import reach_layer

try:
    from onnx2torch.node_converters.transpose import OnnxTranspose
except ImportError:
    OnnxTranspose = None

pytestmark = pytest.mark.skipif(OnnxTranspose is None, reason="onnx2torch not installed")


class TestOnnxTransposeStar:
    def test_transpose_reverses_dimensions(self, simple_star):
        layer = OnnxTranspose(perm=[2, 1, 0])
        result = reach_layer(layer, [simple_star], method='exact')
        assert len(result) == 1
        np.testing.assert_allclose(result[0].V[0, :], simple_star.V[2, :])
        np.testing.assert_allclose(result[0].V[1, :], simple_star.V[1, :])
        np.testing.assert_allclose(result[0].V[2, :], simple_star.V[0, :])

    def test_transpose_preserves_constraints(self, simple_star):
        layer = OnnxTranspose(perm=[2, 1, 0])
        result = reach_layer(layer, [simple_star], method='exact')
        np.testing.assert_array_equal(result[0].C, simple_star.C)
        np.testing.assert_array_equal(result[0].d, simple_star.d)

    def test_identity_permutation(self, simple_star):
        layer = OnnxTranspose(perm=[0, 1, 2])
        result = reach_layer(layer, [simple_star], method='exact')
        np.testing.assert_allclose(result[0].V, simple_star.V)


class TestOnnxTransposeZono:
    def test_transpose_reverses_dimensions(self, simple_zono):
        layer = OnnxTranspose(perm=[2, 1, 0])
        result = reach_layer(layer, [simple_zono], method='approx')
        assert len(result) == 1
        np.testing.assert_allclose(result[0].c[0], simple_zono.c[2])
        np.testing.assert_allclose(result[0].c[2], simple_zono.c[0])
        np.testing.assert_allclose(result[0].V[0, :], simple_zono.V[2, :])
        np.testing.assert_allclose(result[0].V[2, :], simple_zono.V[0, :])


class TestOnnxTransposeBox:
    def test_transpose_permutes_bounds(self):
        from n2v.sets import Box
        box = Box(
            np.array([[1.0], [2.0], [3.0]]),
            np.array([[4.0], [5.0], [6.0]])
        )
        layer = OnnxTranspose(perm=[2, 0, 1])
        result = reach_layer(layer, [box], method='approx')
        np.testing.assert_allclose(result[0].lb, np.array([[3.0], [1.0], [2.0]]))
        np.testing.assert_allclose(result[0].ub, np.array([[6.0], [4.0], [5.0]]))
