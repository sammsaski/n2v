"""Tests for Dropout passthrough in the dispatcher."""

import pytest
import torch.nn as nn
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestDropoutPassthrough:
    """Dropout should be identity for all set types."""

    @pytest.mark.parametrize("dropout_cls", [nn.Dropout, nn.Dropout2d, nn.Dropout3d])
    def test_dropout_star(self, simple_star, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_star], method='exact')
        assert len(result) == 1
        assert result[0] is simple_star

    @pytest.mark.parametrize("dropout_cls", [nn.Dropout, nn.Dropout2d, nn.Dropout3d])
    def test_dropout_zono(self, simple_zono, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_zono], method='approx')
        assert len(result) == 1
        assert result[0] is simple_zono

    @pytest.mark.parametrize("dropout_cls", [nn.Dropout, nn.Dropout2d, nn.Dropout3d])
    def test_dropout_box(self, simple_box, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_box], method='approx')
        assert len(result) == 1
        assert result[0] is simple_box

    @pytest.mark.parametrize("dropout_cls", [nn.Dropout, nn.Dropout2d, nn.Dropout3d])
    def test_dropout_hexatope(self, simple_hexatope, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_hexatope], method='approx')
        assert len(result) == 1
        assert result[0] is simple_hexatope

    @pytest.mark.parametrize("dropout_cls", [nn.Dropout, nn.Dropout2d, nn.Dropout3d])
    def test_dropout_octatope(self, simple_octatope, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_octatope], method='approx')
        assert len(result) == 1
        assert result[0] is simple_octatope

    @pytest.mark.parametrize("dropout_cls", [nn.Dropout, nn.Dropout2d, nn.Dropout3d])
    def test_dropout_image_star(self, simple_image_star, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_image_star], method='exact')
        assert len(result) == 1
        assert result[0] is simple_image_star

    @pytest.mark.parametrize("dropout_cls", [nn.Dropout, nn.Dropout2d, nn.Dropout3d])
    def test_dropout_image_zono(self, simple_image_zono, dropout_cls):
        layer = dropout_cls()
        layer.eval()
        result = reach_layer(layer, [simple_image_zono], method='approx')
        assert len(result) == 1
        assert result[0] is simple_image_zono
