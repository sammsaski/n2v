"""
Soundness tests for Pad layer reachability.

Tests verify that for random points sampled from the input set,
forwarding through PyTorch ZeroPad2d produces outputs contained
in the reachable set.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import ImageStar, ImageZono
from n2v.nn.layer_ops.pad_reach import pad_star, pad_zono


class TestPadStarSoundness:
    """Soundness tests for ZeroPad2d with ImageStar."""

    def test_random_samples_contained(self):
        """Random samples through ZeroPad2d should be within reachable set bounds."""
        lb = np.ones((3, 3, 2)) * (-1)
        ub = np.ones((3, 3, 2))
        img_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        pad_layer = nn.ZeroPad2d((1, 1, 1, 1))

        result = pad_star(pad_layer, [img_star])
        out_star = result[0]
        lb_out, ub_out = out_star.estimate_ranges()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(-1.0, 1.0, size=(3, 3, 2))
            # PyTorch NCHW format
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )
            with torch.no_grad():
                # Output: (1, C, H', W') -> (H', W', C)
                pt_output = pad_layer(pt_input).numpy().squeeze().transpose(1, 2, 0)

            assert np.all(pt_output.flatten() >= lb_out.flatten() - 1e-6)
            assert np.all(pt_output.flatten() <= ub_out.flatten() + 1e-6)

    def test_pad_then_conv(self):
        """Pad followed by Conv: pipeline soundness check."""
        from n2v.nn.layer_ops.conv2d_reach import conv2d_star

        torch.manual_seed(42)
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1)) * 0.5
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        pad_layer = nn.ZeroPad2d(1)
        conv_layer = nn.Conv2d(1, 2, kernel_size=3, bias=True)

        padded = pad_star(pad_layer, [img_star])
        result = conv2d_star(conv_layer, padded)
        lb_out, ub_out = result[0].estimate_ranges()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(0.0, 0.5, size=(4, 4, 1))
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )
            with torch.no_grad():
                pt_padded = pad_layer(pt_input)
                pt_output = conv_layer(pt_padded).numpy().squeeze().transpose(1, 2, 0)

            assert np.all(pt_output.flatten() >= lb_out.flatten() - 1e-5)
            assert np.all(pt_output.flatten() <= ub_out.flatten() + 1e-5)


class TestPadZonoSoundness:
    """Soundness tests for ZeroPad2d with ImageZono."""

    def test_random_samples_contained(self):
        """Random samples through ZeroPad2d should be within ImageZono bounds."""
        lb = np.zeros((3, 3, 1))
        ub = np.ones((3, 3, 1))
        img_zono = ImageZono.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        pad_layer = nn.ZeroPad2d(1)

        result = pad_zono(pad_layer, [img_zono])
        out_zono = result[0]
        lb_out, ub_out = out_zono.get_bounds()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(0.0, 1.0, size=(3, 3, 1))
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )
            with torch.no_grad():
                # Output: (1, C, H', W') -> squeeze batch -> (C, H', W') -> (H', W', C)
                pt_output = pad_layer(pt_input).numpy().squeeze(0).transpose(1, 2, 0)

            assert np.all(pt_output.flatten() >= lb_out.flatten() - 1e-6)
            assert np.all(pt_output.flatten() <= ub_out.flatten() + 1e-6)
