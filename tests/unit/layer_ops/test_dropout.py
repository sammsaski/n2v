"""Tests for Dropout passthrough in the dispatcher."""

import pytest
import torch.nn as nn
from n2v.nn.layer_ops.dispatcher import reach_layer

from onnx2torch.node_converters.dropout import OnnxDropoutDynamic

# onnx2torch's opset-12/13 Dropout converter (OnnxDropoutDynamic) must also be
# treated as an inference-time identity — VGG-16 (issue #50) uses it in its
# classifier and reach must pass through it, not raise NotImplementedError.
_DROPOUT_CLASSES = [nn.Dropout, nn.Dropout2d, nn.Dropout3d, OnnxDropoutDynamic]


class TestDropoutPassthrough:
    """Dropout should be identity for all set types."""

    @pytest.mark.parametrize("dropout_cls", _DROPOUT_CLASSES)
    def test_dropout_star(self, simple_star, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_star], method='exact')
        assert len(result) == 1
        assert result[0] is simple_star

    @pytest.mark.parametrize("dropout_cls", _DROPOUT_CLASSES)
    def test_dropout_zono(self, simple_zono, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_zono], method='approx')
        assert len(result) == 1
        assert result[0] is simple_zono

    @pytest.mark.parametrize("dropout_cls", _DROPOUT_CLASSES)
    def test_dropout_box(self, simple_box, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_box], method='approx')
        assert len(result) == 1
        assert result[0] is simple_box

    @pytest.mark.parametrize("dropout_cls", _DROPOUT_CLASSES)
    def test_dropout_hexatope(self, simple_hexatope, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_hexatope], method='approx')
        assert len(result) == 1
        assert result[0] is simple_hexatope

    @pytest.mark.parametrize("dropout_cls", _DROPOUT_CLASSES)
    def test_dropout_octatope(self, simple_octatope, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_octatope], method='approx')
        assert len(result) == 1
        assert result[0] is simple_octatope

    @pytest.mark.parametrize("dropout_cls", _DROPOUT_CLASSES)
    def test_dropout_image_star(self, simple_image_star, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_image_star], method='exact')
        assert len(result) == 1
        assert result[0] is simple_image_star

    @pytest.mark.parametrize("dropout_cls", _DROPOUT_CLASSES)
    def test_dropout_image_zono(self, simple_image_zono, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_image_zono], method='approx')
        assert len(result) == 1
        assert result[0] is simple_image_zono
