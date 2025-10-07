"""Tests for ReduceMean / ReduceSum layer reachability."""

import numpy as np
import pytest

try:
    from onnx2torch.node_converters.reduce import (
        OnnxReduceStaticAxes,
        OnnxReduceSumStaticAxes,
    )
    HAS_ONNX2TORCH = True
except ImportError:
    HAS_ONNX2TORCH = False

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.nn.layer_ops.dispatcher import reach_layer


pytestmark = pytest.mark.skipif(
    not HAS_ONNX2TORCH, reason="onnx2torch not installed"
)


class TestReduceMeanStarAllDims:
    """Test ReduceMean over all feature dims for a flat Star."""

    def test_reducemean_star_all_dims(self):
        """ReduceMean over axis [1] (ONNX, with batch dim) -> scalar Star.
        Center of output should be mean of input centers."""
        lb = np.array([0.0, 2.0, 4.0, 6.0])
        ub = np.array([1.0, 3.0, 5.0, 7.0])
        star = Star.from_bounds(lb, ub)

        # ONNX axis [1] = feature dim after batch dim
        layer = OnnxReduceStaticAxes('ReduceMean', axes=[1], keepdims=0)
        result = reach_layer(layer, [star], 'approx')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Star)
        assert out.dim == 1

        # Center should be mean of input centers
        input_center = (lb + ub) / 2.0
        expected_center = np.mean(input_center)
        assert np.allclose(out.V[0, 0], expected_center, atol=1e-10)


class TestReduceMeanStarKeepdims:
    """Test ReduceMean with keepdims=1."""

    def test_reducemean_star_keepdims(self):
        """ReduceMean with keepdims=1 -> output dim=1."""
        lb = np.array([0.0, 2.0, 4.0])
        ub = np.array([1.0, 3.0, 5.0])
        star = Star.from_bounds(lb, ub)

        layer = OnnxReduceStaticAxes('ReduceMean', axes=[1], keepdims=1)
        result = reach_layer(layer, [star], 'approx')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Star)
        assert out.dim == 1


class TestReduceSumStar:
    """Test ReduceSum for a flat Star."""

    def test_reducesum_star(self):
        """ReduceSum over axis [1] -> scalar Star.
        Center should be sum of input centers."""
        lb = np.array([1.0, 2.0, 3.0])
        ub = np.array([2.0, 3.0, 4.0])
        star = Star.from_bounds(lb, ub)

        layer = OnnxReduceSumStaticAxes(axes=[1], keepdims=0)
        result = reach_layer(layer, [star], 'approx')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Star)
        assert out.dim == 1

        # Center should be sum of input centers
        input_center = (lb + ub) / 2.0
        expected_center = np.sum(input_center)
        assert np.allclose(out.V[0, 0], expected_center, atol=1e-10)


class TestReduceMeanImageStarSpatial:
    """Test ReduceMean over spatial dims for an ImageStar."""

    def test_reducemean_imagestar_spatial(self):
        """ReduceMean over axes [2,3] (ONNX NCHW: H and W) -> like GlobalAvgPool.
        Output should be a flat Star with dim=channels."""
        h, w, c = 4, 4, 3
        lb = np.zeros((h, w, c))
        ub = np.ones((h, w, c))
        istar = ImageStar.from_bounds(lb, ub, height=h, width=w, num_channels=c)

        # ONNX axes [2, 3] = H, W in NCHW (after removing batch dim 0,
        # these become axes [1, 2] in CHW)
        layer = OnnxReduceStaticAxes('ReduceMean', axes=[2, 3], keepdims=0)
        result = reach_layer(layer, [istar], 'approx')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Star)
        assert out.dim == c  # channels remain


class TestReduceMeanZono:
    """Test ReduceMean for a flat Zono."""

    def test_reducemean_zono(self):
        """ReduceMean -> Zono dim=1, center = mean."""
        lb = np.array([0.0, 2.0, 4.0, 6.0])
        ub = np.array([1.0, 3.0, 5.0, 7.0])
        zono = Zono.from_bounds(lb, ub)

        layer = OnnxReduceStaticAxes('ReduceMean', axes=[1], keepdims=0)
        result = reach_layer(layer, [zono], 'approx')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Zono)
        assert out.dim == 1

        # Center should be mean of input centers
        input_center = (lb + ub) / 2.0
        expected_center = np.mean(input_center)
        assert np.allclose(out.c[0, 0], expected_center, atol=1e-10)


class TestReduceMeanBox:
    """Test ReduceMean for a Box."""

    def test_reducemean_box(self):
        """ReduceMean -> Box dim=1, bounds = mean of bounds."""
        lb = np.array([[0.0], [2.0], [4.0], [6.0]])
        ub = np.array([[1.0], [3.0], [5.0], [7.0]])
        box = Box(lb, ub)

        layer = OnnxReduceStaticAxes('ReduceMean', axes=[1], keepdims=0)
        result = reach_layer(layer, [box], 'approx')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Box)
        assert out.dim == 1

        # Bounds should be mean of original bounds
        assert np.allclose(out.lb[0, 0], np.mean(lb), atol=1e-10)
        assert np.allclose(out.ub[0, 0], np.mean(ub), atol=1e-10)
