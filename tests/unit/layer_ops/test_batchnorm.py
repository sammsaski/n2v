"""Tests for BatchNorm layer reachability operations."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box, ImageStar, ImageZono
from n2v.nn.layer_ops.dispatcher import reach_layer


def _make_bn2d(num_features=3):
    """Create a BatchNorm2d layer in eval mode with non-trivial running stats."""
    bn = nn.BatchNorm2d(num_features)
    bn.eval()
    # Set non-trivial parameters
    with torch.no_grad():
        bn.weight.data = torch.tensor([2.0, 0.5, -1.0][:num_features], dtype=torch.float32)
        bn.bias.data = torch.tensor([0.1, -0.2, 0.3][:num_features], dtype=torch.float32)
        bn.running_mean.data = torch.tensor([1.0, 2.0, -0.5][:num_features], dtype=torch.float32)
        bn.running_var.data = torch.tensor([0.5, 1.5, 0.25][:num_features], dtype=torch.float32)
    return bn


def _make_bn1d(num_features=4):
    """Create a BatchNorm1d layer in eval mode with non-trivial running stats."""
    bn = nn.BatchNorm1d(num_features)
    bn.eval()
    # Set non-trivial parameters
    with torch.no_grad():
        bn.weight.data = torch.tensor([1.5, -0.5, 2.0, 0.8][:num_features], dtype=torch.float32)
        bn.bias.data = torch.tensor([0.1, 0.2, -0.3, 0.0][:num_features], dtype=torch.float32)
        bn.running_mean.data = torch.tensor([0.5, 1.0, -1.0, 0.0][:num_features], dtype=torch.float32)
        bn.running_var.data = torch.tensor([1.0, 0.5, 2.0, 0.25][:num_features], dtype=torch.float32)
    return bn


class TestBatchNorm2dStar:
    """Tests for BatchNorm2d with Star/ImageStar sets."""

    def test_batchnorm2d_imagestar_dispatch(self):
        """BN2d handled by dispatcher for ImageStar, output is ImageStar with correct dims."""
        num_channels = 3
        height, width = 4, 4
        bn = _make_bn2d(num_channels)

        lb = np.zeros((height, width, num_channels))
        ub = np.ones((height, width, num_channels))
        img_star = ImageStar.from_bounds(lb, ub, height=height, width=width, num_channels=num_channels)

        result = reach_layer(bn, [img_star], method='exact')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, ImageStar)
        assert out.height == height
        assert out.width == width
        assert out.num_channels == num_channels
        # V shape should be (H, W, C, nVar+1)
        assert out.V.shape[0] == height
        assert out.V.shape[1] == width
        assert out.V.shape[2] == num_channels
        pytest.assert_image_star_valid(out)

    def test_batchnorm2d_imagestar_bounds(self):
        """Output bounds contain corner points forwarded through PyTorch BN."""
        num_channels = 3
        height, width = 2, 2
        bn = _make_bn2d(num_channels)

        lb = np.random.uniform(0.0, 0.5, (height, width, num_channels))
        ub = lb + np.random.uniform(0.1, 0.5, (height, width, num_channels))
        img_star = ImageStar.from_bounds(lb, ub, height=height, width=width, num_channels=num_channels)

        result = reach_layer(bn, [img_star], method='exact')
        out = result[0]

        # Forward lb and ub through PyTorch BN and verify they are within estimated ranges
        # PyTorch BN expects (N, C, H, W) format
        lb_torch = torch.from_numpy(lb.transpose(2, 0, 1)).unsqueeze(0).float()  # (1, C, H, W)
        ub_torch = torch.from_numpy(ub.transpose(2, 0, 1)).unsqueeze(0).float()

        with torch.no_grad():
            out_lb_pt = bn(lb_torch).squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)
            out_ub_pt = bn(ub_torch).squeeze(0).permute(1, 2, 0).numpy()

        # Get estimated ranges from the output ImageStar
        est_lb, est_ub = out.estimate_ranges()
        est_lb = est_lb.reshape(height, width, num_channels)
        est_ub = est_ub.reshape(height, width, num_channels)

        # Both corner evaluations should be contained within estimated ranges
        # (with small tolerance for floating point)
        tol = 1e-6
        assert np.all(out_lb_pt >= est_lb - tol), \
            f"LB corner not contained: min diff = {(out_lb_pt - est_lb).min()}"
        assert np.all(out_lb_pt <= est_ub + tol), \
            f"LB corner exceeds UB: max diff = {(out_lb_pt - est_ub).max()}"
        assert np.all(out_ub_pt >= est_lb - tol), \
            f"UB corner not contained: min diff = {(out_ub_pt - est_lb).min()}"
        assert np.all(out_ub_pt <= est_ub + tol), \
            f"UB corner exceeds UB: max diff = {(out_ub_pt - est_ub).max()}"


class TestBatchNorm1dStar:
    """Tests for BatchNorm1d with Star sets."""

    def test_batchnorm1d_star_dispatch(self):
        """BN1d handled by dispatcher for Star, output is Star with correct dim."""
        num_features = 4
        bn = _make_bn1d(num_features)

        lb = np.array([[0.0], [0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        result = reach_layer(bn, [star], method='exact')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Star)
        assert out.dim == num_features
        pytest.assert_star_valid(out)


class TestBatchNormZono:
    """Tests for BatchNorm with Zonotope sets."""

    def test_batchnorm2d_imagezono_dispatch(self):
        """BN2d via dispatcher with ImageZono."""
        num_channels = 3
        height, width = 4, 4
        bn = _make_bn2d(num_channels)

        lb = np.zeros((height, width, num_channels))
        ub = np.ones((height, width, num_channels))
        img_zono = ImageZono.from_bounds(lb, ub, height=height, width=width, num_channels=num_channels)

        result = reach_layer(bn, [img_zono], method='approx')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, ImageZono)
        assert out.height == height
        assert out.width == width
        assert out.num_channels == num_channels


class TestBatchNormBox:
    """Tests for BatchNorm with Box sets."""

    def test_batchnorm1d_box_dispatch(self):
        """BN1d via dispatcher with Box."""
        num_features = 4
        bn = _make_bn1d(num_features)

        lb = np.array([[0.0], [0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0], [1.0]])
        box = Box(lb, ub)

        result = reach_layer(bn, [box], method='exact')

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Box)
        assert out.dim == num_features

        # Verify bounds are correct by comparing with PyTorch BN
        # Forward lb and ub through BN
        lb_torch = torch.from_numpy(lb.flatten()).unsqueeze(0).float()  # (1, num_features)
        ub_torch = torch.from_numpy(ub.flatten()).unsqueeze(0).float()

        with torch.no_grad():
            out_lb_pt = bn(lb_torch).squeeze(0).numpy().reshape(-1, 1)
            out_ub_pt = bn(ub_torch).squeeze(0).numpy().reshape(-1, 1)

        # For each dimension, the true min/max should be within box bounds
        # Note: negative scale flips lb/ub, so take elementwise min/max
        true_lb = np.minimum(out_lb_pt, out_ub_pt)
        true_ub = np.maximum(out_lb_pt, out_ub_pt)

        tol = 1e-6
        assert np.all(out.lb <= true_lb + tol), \
            f"Box lb not sound: max diff = {(out.lb - true_lb).max()}"
        assert np.all(out.ub >= true_ub - tol), \
            f"Box ub not sound: max diff = {(true_ub - out.ub).max()}"


class TestBatchNormAutoFuse:
    """Test that reach_pytorch_model auto-fuses BatchNorm."""

    def test_reach_with_conv_bn_model(self):
        """Full reach() with Conv+BN model should work via auto-fusion."""
        from n2v.nn import NeuralNetwork

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 2),
        )
        model.eval()

        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1)) * 0.1
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        net = NeuralNetwork(model)
        result = net.reach(input_star, method='approx')

        assert len(result) >= 1
        assert isinstance(result[0], Star)
