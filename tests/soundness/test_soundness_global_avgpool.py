"""
Soundness tests for GlobalAvgPool layer reachability.

Tests verify that for random points sampled from the input set,
forwarding through PyTorch GlobalAvgPool produces outputs contained
in the reachable set.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import ImageStar, ImageZono
from n2v.nn.layer_ops.global_avgpool_reach import global_avgpool_star, global_avgpool_zono


class TestGlobalAvgPoolStarSoundness:
    """Soundness tests for GlobalAvgPool with ImageStar."""

    def test_uniform_bounds(self):
        """GlobalAvgPool on [0,1] ImageStar: output bounds should contain all samples."""
        lb = np.zeros((4, 4, 2))
        ub = np.ones((4, 4, 2))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=2)

        result = global_avgpool_star([img_star])
        out_star = result[0]
        lb_out, ub_out = out_star.estimate_ranges()

        layer = nn.AdaptiveAvgPool2d(1)

        np.random.seed(42)
        for _ in range(100):
            point = np.random.uniform(0.0, 1.0, size=(4, 4, 2))
            # PyTorch NCHW format
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )
            with torch.no_grad():
                # Output is (1, C, 1, 1), convert to (C,)
                pt_output = layer(pt_input).numpy().squeeze()

            assert np.all(pt_output >= lb_out.flatten() - 1e-6)
            assert np.all(pt_output <= ub_out.flatten() + 1e-6)

    def test_shifted_bounds(self):
        """GlobalAvgPool on [3,7] ImageStar: mean of uniform is in [3,7]."""
        lb = np.ones((3, 3, 1)) * 3
        ub = np.ones((3, 3, 1)) * 7
        img_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        result = global_avgpool_star([img_star])
        out_star = result[0]
        lb_out, ub_out = out_star.estimate_ranges()

        assert np.all(lb_out >= 3.0 - 1e-6)
        assert np.all(ub_out <= 7.0 + 1e-6)

    def test_negative_bounds(self):
        """GlobalAvgPool on [-5, -1] ImageStar."""
        lb = np.ones((2, 2, 1)) * -5
        ub = np.ones((2, 2, 1)) * -1
        img_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        result = global_avgpool_star([img_star])
        out_star = result[0]
        lb_out, ub_out = out_star.estimate_ranges()

        assert np.all(lb_out >= -5.0 - 1e-6)
        assert np.all(ub_out <= -1.0 + 1e-6)


class TestGlobalAvgPoolZonoSoundness:
    """Soundness tests for GlobalAvgPool with ImageZono."""

    def test_uniform_bounds(self):
        """GlobalAvgPool on [0,1] ImageZono: output bounds should contain all samples."""
        lb = np.zeros((4, 4, 2))
        ub = np.ones((4, 4, 2))
        img_zono = ImageZono.from_bounds(lb, ub, height=4, width=4, num_channels=2)

        result = global_avgpool_zono([img_zono])
        out_zono = result[0]
        lb_out, ub_out = out_zono.get_bounds()

        layer = nn.AdaptiveAvgPool2d(1)

        np.random.seed(42)
        for _ in range(100):
            point = np.random.uniform(0.0, 1.0, size=(4, 4, 2))
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()

            assert np.all(pt_output >= lb_out.flatten() - 1e-6)
            assert np.all(pt_output <= ub_out.flatten() + 1e-6)

    def test_scaling_check(self):
        """GlobalAvgPool on [2,4] ImageZono: bounds should contain [2,4]."""
        lb = np.ones((3, 3, 1)) * 2
        ub = np.ones((3, 3, 1)) * 4
        img_zono = ImageZono.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        result = global_avgpool_zono([img_zono])
        out_zono = result[0]
        lb_out, ub_out = out_zono.get_bounds()

        assert np.all(lb_out >= 2.0 - 1e-6)
        assert np.all(ub_out <= 4.0 + 1e-6)
