"""
Soundness tests for Reshape operations.

Verify that reshape preserves set containment:
sample points from input, reshape via PyTorch, verify in output set.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, ImageStar, ImageZono
from n2v.nn.reach import _handle_reshape


class TestReshapeSoundness:
    """Reshape should preserve all points in the set."""

    def test_imagestar_flatten_containment(self):
        """Points in ImageStar should be in flattened Star."""
        lb = np.zeros((3, 3, 2))
        ub = np.ones((3, 3, 2))
        img_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        result = _handle_reshape([img_star], (1, -1))
        out_star = result[0]
        lb_out, ub_out = out_star.estimate_ranges()

        np.random.seed(42)
        for _ in range(100):
            # Sample in HWC format (matching ImageStar storage)
            point_hwc = np.random.uniform(0.0, 1.0, size=(3, 3, 2))
            # PyTorch flatten uses CHW order
            point_chw = point_hwc.transpose(2, 0, 1).flatten()

            assert np.all(point_chw >= lb_out.flatten() - 1e-6)
            assert np.all(point_chw <= ub_out.flatten() + 1e-6)

    def test_imagezono_flatten_containment(self):
        """Points in ImageZono should be in flattened Zono."""
        lb = np.ones((2, 2, 3)) * 2
        ub = np.ones((2, 2, 3)) * 5
        img_zono = ImageZono.from_bounds(lb, ub, height=2, width=2, num_channels=3)

        result = _handle_reshape([img_zono], (1, -1))
        out_zono = result[0]
        lb_out, ub_out = out_zono.get_bounds()

        np.random.seed(42)
        for _ in range(100):
            point_hwc = np.random.uniform(2.0, 5.0, size=(2, 2, 3))
            point_chw = point_hwc.transpose(2, 0, 1).flatten()

            assert np.all(point_chw >= lb_out.flatten() - 1e-6)
            assert np.all(point_chw <= ub_out.flatten() + 1e-6)

    def test_imagestar_multichannel_flatten_containment(self):
        """Points in multi-channel ImageStar should be in flattened Star."""
        lb = np.ones((4, 4, 3)) * -1
        ub = np.ones((4, 4, 3)) * 1
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=3)

        result = _handle_reshape([img_star], (1, -1))
        out_star = result[0]
        lb_out, ub_out = out_star.estimate_ranges()

        np.random.seed(123)
        for _ in range(100):
            point_hwc = np.random.uniform(-1.0, 1.0, size=(4, 4, 3))
            point_chw = point_hwc.transpose(2, 0, 1).flatten()

            assert np.all(point_chw >= lb_out.flatten() - 1e-6)
            assert np.all(point_chw <= ub_out.flatten() + 1e-6)
