"""
Soundness tests for residual (element-wise) set addition.

Tests verify that for random points sampled from the input set,
forwarding through two PyTorch layers and summing produces outputs
contained in the reachable set obtained via _add_sets.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.nn.reach import _add_sets
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.layer_ops.flatten_reach import flatten_zono
from n2v.sets import Star, ImageStar, ImageZono


class TestResidualAddStarSoundness:
    """Soundness tests for residual add with Star sets."""

    def test_linear_residual_star(self):
        """Two linear layers added: W1(x) + W2(x) should be within reach set bounds."""
        torch.manual_seed(42)

        W1 = nn.Linear(4, 4, bias=False)
        W2 = nn.Linear(4, 4, bias=False)

        # Create Star from bounds [0, 1] dim=4
        lb = np.zeros(4)
        ub = np.ones(4)
        star = Star.from_bounds(lb, ub)

        # Forward through each layer via reach_layer
        reach_W1 = reach_layer(W1, [star], 'approx')
        reach_W2 = reach_layer(W2, [star], 'approx')

        # Add the two reach sets
        result_sets = _add_sets(reach_W1, reach_W2, 'add')
        result = result_sets[0]

        # Get bounds from the result Star
        lb_out, ub_out = result.estimate_ranges()

        # Sample 200 random points from [0, 1]^4 and verify containment
        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(0.0, 1.0, size=(4,))
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out1 = W1(pt_input)
                out2 = W2(pt_input)
                pt_output = (out1 + out2).numpy().flatten()

            assert np.all(pt_output >= lb_out.flatten() - 1e-5), (
                f"Output {pt_output} below lower bound {lb_out.flatten()}"
            )
            assert np.all(pt_output <= ub_out.flatten() + 1e-5), (
                f"Output {pt_output} above upper bound {ub_out.flatten()}"
            )


class TestResidualAddImageStarSoundness:
    """Soundness tests for residual add with ImageStar sets."""

    def test_conv_residual_imagestar(self):
        """Two conv layers added: conv1(x) + conv2(x) should be within reach set bounds."""
        torch.manual_seed(42)

        conv1 = nn.Conv2d(1, 2, 3, padding=1)
        conv2 = nn.Conv2d(1, 2, 1)

        # Create ImageStar from bounds [0, 0.5] for 4x4x1 image
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1)) * 0.5
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        # Forward through each conv layer via reach_layer
        reach_conv1 = reach_layer(conv1, [img_star], 'approx')
        reach_conv2 = reach_layer(conv2, [img_star], 'approx')

        # Add the two reach sets
        result_sets = _add_sets(reach_conv1, reach_conv2, 'add')
        result = result_sets[0]

        # Get bounds from the result ImageStar
        lb_out, ub_out = result.estimate_ranges()

        # Sample 200 random points and verify containment
        np.random.seed(42)
        for _ in range(200):
            # Random point in HWC format within bounds
            point = np.random.uniform(0.0, 0.5, size=(4, 4, 1))

            # Convert to NCHW for PyTorch
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )

            with torch.no_grad():
                out1 = conv1(pt_input)  # (1, 2, 4, 4)
                out2 = conv2(pt_input)  # (1, 2, 4, 4)
                pt_output_nchw = (out1 + out2).numpy()

            # Convert PyTorch NCHW output to HWC for comparison with ImageStar bounds
            pt_output_hwc = pt_output_nchw[0].transpose(1, 2, 0)  # (4, 4, 2)
            pt_output_flat = pt_output_hwc.flatten()

            assert np.all(pt_output_flat >= lb_out.flatten() - 1e-5), (
                f"Output below lower bound. "
                f"Min diff: {np.min(pt_output_flat - lb_out.flatten())}"
            )
            assert np.all(pt_output_flat <= ub_out.flatten() + 1e-5), (
                f"Output above upper bound. "
                f"Max diff: {np.max(pt_output_flat - ub_out.flatten())}"
            )


class TestResidualAddZonoSoundness:
    """Soundness tests for residual add with Zono sets."""

    def test_linear_residual_zono(self):
        """Two linear layers added on flattened ImageZono: W1(x) + W2(x) within bounds."""
        torch.manual_seed(42)

        W1 = nn.Linear(9, 9, bias=True)
        W2 = nn.Linear(9, 9, bias=True)

        # Create ImageZono from bounds [0, 1] for 3x3x1 image
        lb = np.zeros((3, 3, 1))
        ub = np.ones((3, 3, 1))
        img_zono = ImageZono.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        # Flatten via flatten_zono
        flat_zonos = flatten_zono(nn.Flatten(), [img_zono])

        # Forward through each linear layer via reach_layer
        reach_W1 = reach_layer(W1, flat_zonos, 'approx')
        reach_W2 = reach_layer(W2, flat_zonos, 'approx')

        # Add the two reach sets
        result_sets = _add_sets(reach_W1, reach_W2, 'add')
        result = result_sets[0]

        # Get bounds from the result Zono
        lb_out, ub_out = result.get_bounds()

        # Sample 200 random points and verify containment
        np.random.seed(42)
        for _ in range(200):
            # Random point in HWC [0, 1]
            point_hwc = np.random.uniform(0.0, 1.0, size=(3, 3, 1))

            # Flatten in CHW order to match nn.Flatten behavior
            point_chw = point_hwc.transpose(2, 0, 1)  # (1, 3, 3)
            point_flat = point_chw.flatten()

            pt_input = torch.tensor(point_flat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out1 = W1(pt_input)
                out2 = W2(pt_input)
                pt_output = (out1 + out2).numpy().flatten()

            assert np.all(pt_output >= lb_out.flatten() - 1e-5), (
                f"Output {pt_output} below lower bound {lb_out.flatten()}"
            )
            assert np.all(pt_output <= ub_out.flatten() + 1e-5), (
                f"Output {pt_output} above upper bound {ub_out.flatten()}"
            )
