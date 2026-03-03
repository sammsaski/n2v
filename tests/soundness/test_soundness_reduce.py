"""
Soundness tests for ReduceMean / ReduceSum layer reachability.

Tests verify that for random points sampled from the input set,
applying the reduction operation produces outputs contained in the
output set bounds.
"""

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
from n2v.nn.layer_ops.dispatcher import reach_layer


pytestmark = pytest.mark.skipif(
    not HAS_ONNX2TORCH, reason="onnx2torch not installed"
)


class TestReduceMeanStarSoundness:
    """Soundness: ReduceMean on Star, verify sampled points are within bounds."""

    def test_reducemean_star_soundness(self):
        """Create Star from bounds, apply ReduceMean.
        Sample 200 points, verify np.mean(point) within output bounds."""
        lb = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ub = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        star = Star.from_bounds(lb, ub)

        layer = OnnxReduceStaticAxes('ReduceMean', axes=[1], keepdims=0)
        result = reach_layer(layer, [star], 'approx')
        out = result[0]

        lb_out, ub_out = out.get_ranges()

        np.random.seed(42)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(5,))
            point = lb + alpha * (ub - lb)
            reduced = np.mean(point)

            assert reduced >= lb_out.flatten()[0] - 1e-6, (
                f"Reduced value {reduced} below lower bound {lb_out.flatten()[0]}"
            )
            assert reduced <= ub_out.flatten()[0] + 1e-6, (
                f"Reduced value {reduced} above upper bound {ub_out.flatten()[0]}"
            )


class TestReduceSumZonoSoundness:
    """Soundness: ReduceSum on Zono, verify sampled points are within bounds."""

    def test_reducesum_zono_soundness(self):
        """Create Zono from bounds, apply ReduceSum.
        Sample 200 points, verify np.sum(point) within output bounds."""
        lb = np.array([0.0, 1.0, 2.0, 3.0])
        ub = np.array([1.0, 2.0, 3.0, 4.0])
        zono = Zono.from_bounds(lb, ub)

        layer = OnnxReduceSumStaticAxes(axes=[1], keepdims=0)
        result = reach_layer(layer, [zono], 'approx')
        out = result[0]

        lb_out, ub_out = out.get_bounds()

        np.random.seed(99)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(4,))
            point = lb + alpha * (ub - lb)
            reduced = np.sum(point)

            assert reduced >= lb_out.flatten()[0] - 1e-6, (
                f"Reduced value {reduced} below lower bound {lb_out.flatten()[0]}"
            )
            assert reduced <= ub_out.flatten()[0] + 1e-6, (
                f"Reduced value {reduced} above upper bound {ub_out.flatten()[0]}"
            )


class TestReduceMeanBoxSoundness:
    """Soundness: ReduceMean on Box, verify sampled points are within bounds."""

    def test_reducemean_box_soundness(self):
        """Create Box from bounds, apply ReduceMean.
        Sample 200 points, verify np.mean(point) within output bounds."""
        lb = np.array([[0.0], [1.0], [2.0], [3.0]])
        ub = np.array([[1.0], [2.0], [3.0], [4.0]])
        box = Box(lb, ub)

        layer = OnnxReduceStaticAxes('ReduceMean', axes=[1], keepdims=0)
        result = reach_layer(layer, [box], 'approx')
        out = result[0]

        np.random.seed(7)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(4,))
            point = lb.flatten() + alpha * (ub.flatten() - lb.flatten())
            reduced = np.mean(point)

            assert reduced >= out.lb.flatten()[0] - 1e-6, (
                f"Reduced value {reduced} below lower bound {out.lb.flatten()[0]}"
            )
            assert reduced <= out.ub.flatten()[0] + 1e-6, (
                f"Reduced value {reduced} above upper bound {out.ub.flatten()[0]}"
            )
