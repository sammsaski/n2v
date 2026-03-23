"""Tests for OnnxCast dispatch — cast should be pass-through identity for verification."""

import pytest

from n2v.nn.layer_ops.dispatcher import reach_layer

# Import OnnxCast (skip tests if onnx2torch not installed)
try:
    from onnx2torch.node_converters.cast import OnnxCast
    from onnx import TensorProto
except ImportError:
    OnnxCast = None
    TensorProto = None

pytestmark = pytest.mark.skipif(OnnxCast is None, reason="onnx2torch not installed")


class TestOnnxCastPassthrough:
    """OnnxCast should be identity for all set types (type cast is irrelevant for verification)."""

    def test_cast_star(self, simple_star):
        layer = OnnxCast(int(TensorProto.FLOAT))
        result = reach_layer(layer, [simple_star], method='exact')
        assert len(result) == 1
        assert result[0] is simple_star

    def test_cast_zono(self, simple_zono):
        layer = OnnxCast(int(TensorProto.FLOAT))
        result = reach_layer(layer, [simple_zono], method='approx')
        assert len(result) == 1
        assert result[0] is simple_zono

    def test_cast_box(self, simple_box):
        layer = OnnxCast(int(TensorProto.FLOAT))
        result = reach_layer(layer, [simple_box], method='approx')
        assert len(result) == 1
        assert result[0] is simple_box

    def test_cast_hexatope(self, simple_hexatope):
        layer = OnnxCast(int(TensorProto.FLOAT))
        result = reach_layer(layer, [simple_hexatope], method='approx')
        assert len(result) == 1
        assert result[0] is simple_hexatope

    def test_cast_octatope(self, simple_octatope):
        layer = OnnxCast(int(TensorProto.DOUBLE))
        result = reach_layer(layer, [simple_octatope], method='approx')
        assert len(result) == 1
        assert result[0] is simple_octatope

    def test_cast_image_star(self, simple_image_star):
        layer = OnnxCast(int(TensorProto.FLOAT))
        result = reach_layer(layer, [simple_image_star], method='exact')
        assert len(result) == 1
        assert result[0] is simple_image_star

    def test_cast_image_zono(self, simple_image_zono):
        layer = OnnxCast(int(TensorProto.FLOAT))
        result = reach_layer(layer, [simple_image_zono], method='approx')
        assert len(result) == 1
        assert result[0] is simple_image_zono
