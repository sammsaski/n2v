"""Tests for OnnxTranspose dispatch on FLAT sets.

History note (task 2.3): these tests previously pinned a row-permutation
semantics (``V[perm, :]``) that mis-read ONNX *axis* indices as row
indices — wrong for any real tensor, and never exercised by a supported
benchmark. Flat sets carry no tensor shape, so the corrected behavior is:
perms that only move the size-1 batch axis are a flat no-op; anything
else raises until shape tracking lands. Full axis-permutation semantics
(with layout pins against torch.permute) are covered for ImageStar/
ImageZono in test_transpose_imagestar.py.
"""

import numpy as np
import pytest

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Box

try:
    from onnx2torch.node_converters.transpose import OnnxTranspose
except ImportError:
    OnnxTranspose = None

pytestmark = pytest.mark.skipif(OnnxTranspose is None, reason="onnx2torch not installed")


class TestOnnxTransposeStar:
    def test_identity_permutation(self, simple_star):
        layer = OnnxTranspose(perm=[0, 1, 2])
        result = reach_layer(layer, [simple_star], method='exact')
        assert len(result) == 1
        np.testing.assert_allclose(result[0].V, simple_star.V)
        np.testing.assert_array_equal(result[0].C, simple_star.C)
        np.testing.assert_array_equal(result[0].d, simple_star.d)

    def test_batch_axis_move_is_noop(self, simple_star):
        """(1, N) -> (N, 1): flat layout unchanged."""
        layer = OnnxTranspose(perm=[1, 0])
        result = reach_layer(layer, [simple_star], method='exact')
        np.testing.assert_allclose(result[0].V, simple_star.V)

    def test_data_reordering_perm_raises(self, simple_star):
        layer = OnnxTranspose(perm=[0, 2, 1])
        with pytest.raises(NotImplementedError):
            reach_layer(layer, [simple_star], method='exact')


class TestOnnxTransposeZono:
    def test_identity_permutation(self, simple_zono):
        layer = OnnxTranspose(perm=[0, 1, 2])
        result = reach_layer(layer, [simple_zono], method='approx')
        np.testing.assert_allclose(result[0].c, simple_zono.c)
        np.testing.assert_allclose(result[0].V, simple_zono.V)

    def test_data_reordering_perm_raises(self, simple_zono):
        layer = OnnxTranspose(perm=[2, 1, 0])
        with pytest.raises(NotImplementedError):
            reach_layer(layer, [simple_zono], method='approx')


class TestOnnxTransposeBox:
    def test_identity_permutation(self):
        box = Box(
            np.array([[1.0], [2.0], [3.0]]),
            np.array([[4.0], [5.0], [6.0]])
        )
        layer = OnnxTranspose(perm=[0, 1, 2])
        result = reach_layer(layer, [box], method='approx')
        np.testing.assert_allclose(result[0].lb, box.lb)
        np.testing.assert_allclose(result[0].ub, box.ub)

    def test_data_reordering_perm_raises(self):
        box = Box(
            np.array([[1.0], [2.0], [3.0]]),
            np.array([[4.0], [5.0], [6.0]])
        )
        layer = OnnxTranspose(perm=[2, 0, 1])
        with pytest.raises(NotImplementedError):
            reach_layer(layer, [box], method='approx')
