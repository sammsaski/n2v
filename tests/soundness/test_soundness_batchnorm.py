"""
Soundness tests for BatchNorm layer reachability.

Tests verify that for random points sampled from the input set,
forwarding through PyTorch BatchNorm produces outputs contained
in the reachable set.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops.batchnorm_reach import batchnorm_star, batchnorm_zono, batchnorm_box
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.utils.model_preprocessing import fuse_batchnorm


def _make_bn2d(num_features, weight, bias, running_mean, running_var):
    """Create a BatchNorm2d layer in eval mode with specified parameters."""
    bn = nn.BatchNorm2d(num_features)
    bn.eval()
    with torch.no_grad():
        bn.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        bn.bias.copy_(torch.tensor(bias, dtype=torch.float32))
        bn.running_mean.copy_(torch.tensor(running_mean, dtype=torch.float32))
        bn.running_var.copy_(torch.tensor(running_var, dtype=torch.float32))
    return bn


def _make_bn1d(num_features, weight, bias, running_mean, running_var):
    """Create a BatchNorm1d layer in eval mode with specified parameters."""
    bn = nn.BatchNorm1d(num_features)
    bn.eval()
    with torch.no_grad():
        bn.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        bn.bias.copy_(torch.tensor(bias, dtype=torch.float32))
        bn.running_mean.copy_(torch.tensor(running_mean, dtype=torch.float32))
        bn.running_var.copy_(torch.tensor(running_var, dtype=torch.float32))
    return bn


class TestBatchNorm2dStarSoundness:
    """Soundness tests for BatchNorm2d with ImageStar."""

    def test_random_samples_contained(self):
        """BN2d on [-1, 1] ImageStar: output bounds should contain all samples."""
        lb = -np.ones((4, 4, 2))
        ub = np.ones((4, 4, 2))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=2)

        bn = _make_bn2d(
            num_features=2,
            weight=[1.5, 0.8],
            bias=[0.1, -0.2],
            running_mean=[0.5, -0.3],
            running_var=[0.25, 1.0],
        )

        result = batchnorm_star(bn, [img_star])
        out_star = result[0]
        lb_out, ub_out = out_star.estimate_ranges()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(-1.0, 1.0, size=(4, 4, 2))
            # PyTorch expects NCHW format
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )
            with torch.no_grad():
                pt_output = bn(pt_input).numpy().squeeze()
                # Convert from CHW back to HWC for comparison
                pt_output = pt_output.transpose(1, 2, 0)

            assert np.all(pt_output >= lb_out.reshape(4, 4, 2) - 1e-5), (
                f"Sample below lower bound: min diff = {(pt_output - lb_out.reshape(4, 4, 2)).min()}"
            )
            assert np.all(pt_output <= ub_out.reshape(4, 4, 2) + 1e-5), (
                f"Sample above upper bound: max diff = {(pt_output - ub_out.reshape(4, 4, 2)).max()}"
            )

    def test_narrow_bounds(self):
        """BN2d on narrow [0.4, 0.6] input range: output range should be tight."""
        lb = np.ones((4, 4, 2)) * 0.4
        ub = np.ones((4, 4, 2)) * 0.6
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=2)

        bn = _make_bn2d(
            num_features=2,
            weight=[1.5, 0.8],
            bias=[0.1, -0.2],
            running_mean=[0.5, -0.3],
            running_var=[0.25, 1.0],
        )

        result = batchnorm_star(bn, [img_star])
        out_star = result[0]
        lb_out, ub_out = out_star.estimate_ranges()

        # With narrow input range [0.4, 0.6] (width 0.2), the BN scale can
        # amplify this. Channel 0: scale = 1.5/sqrt(0.25+eps) ~ 3.0, so
        # output width ~ 0.6. Channel 1: scale = 0.8/sqrt(1.0+eps) ~ 0.8,
        # so output width ~ 0.16. Both should be well under 1.0.
        widths = ub_out - lb_out
        assert np.all(widths < 1.0), (
            f"Output range too wide: max width = {widths.max()}"
        )


class TestBatchNorm1dStarSoundness:
    """Soundness tests for BatchNorm1d with plain Star."""

    def test_random_samples_contained(self):
        """BN1d on [0, 1] Star: output bounds should contain all samples."""
        dim = 5
        lb = np.zeros((dim, 1))
        ub = np.ones((dim, 1))
        star = Star.from_bounds(lb, ub)

        bn = _make_bn1d(
            num_features=dim,
            weight=[1.2, 0.9, 1.5, 0.7, 1.1],
            bias=[0.1, -0.3, 0.2, 0.0, -0.1],
            running_mean=[0.3, -0.1, 0.5, 0.2, -0.4],
            running_var=[0.5, 1.2, 0.3, 0.8, 1.5],
        )

        result = batchnorm_star(bn, [star])
        out_star = result[0]
        lb_out, ub_out = out_star.estimate_ranges()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(0.0, 1.0, size=(dim,))
            # PyTorch BN1d expects (N, C) input
            pt_input = torch.tensor(point[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                pt_output = bn(pt_input).numpy().squeeze()

            assert np.all(pt_output >= lb_out.flatten() - 1e-5), (
                f"Sample below lower bound: min diff = {(pt_output - lb_out.flatten()).min()}"
            )
            assert np.all(pt_output <= ub_out.flatten() + 1e-5), (
                f"Sample above upper bound: max diff = {(pt_output - ub_out.flatten()).max()}"
            )


class TestBatchNormZonoSoundness:
    """Soundness tests for BatchNorm2d with ImageZono."""

    def test_random_samples_contained(self):
        """BN2d on [0, 1] ImageZono: output bounds should contain all samples."""
        lb = np.zeros((3, 3, 2))
        ub = np.ones((3, 3, 2))
        img_zono = ImageZono.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        bn = _make_bn2d(
            num_features=2,
            weight=[1.3, 0.7],
            bias=[0.2, -0.1],
            running_mean=[0.4, -0.2],
            running_var=[0.6, 0.9],
        )

        result = batchnorm_zono(bn, [img_zono])
        out_zono = result[0]
        lb_out, ub_out = out_zono.get_bounds()

        layer_pt = bn

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(0.0, 1.0, size=(3, 3, 2))
            # PyTorch expects NCHW format
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )
            with torch.no_grad():
                pt_output = layer_pt(pt_input).numpy().squeeze()
                # Convert from CHW back to HWC, then flatten
                pt_output_hwc = pt_output.transpose(1, 2, 0)
                pt_output_flat = pt_output_hwc.flatten()

            assert np.all(pt_output_flat >= lb_out.flatten() - 1e-5), (
                f"Sample below lower bound: min diff = {(pt_output_flat - lb_out.flatten()).min()}"
            )
            assert np.all(pt_output_flat <= ub_out.flatten() + 1e-5), (
                f"Sample above upper bound: max diff = {(pt_output_flat - ub_out.flatten()).max()}"
            )


class TestBatchNormBoxSoundness:
    """Soundness tests for BatchNorm1d with Box."""

    def test_random_samples_contained(self):
        """BN1d on [0, 1] Box: output bounds should contain all samples."""
        dim = 4
        lb = np.zeros((dim, 1))
        ub = np.ones((dim, 1))
        box = Box(lb, ub)

        bn = _make_bn1d(
            num_features=dim,
            weight=[1.4, 0.6, 1.1, 0.9],
            bias=[-0.1, 0.3, 0.0, -0.2],
            running_mean=[0.2, -0.5, 0.1, 0.3],
            running_var=[0.4, 1.1, 0.7, 0.5],
        )

        result = batchnorm_box(bn, [box])
        out_box = result[0]

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(0.0, 1.0, size=(dim,))
            # PyTorch BN1d expects (N, C)
            pt_input = torch.tensor(point[np.newaxis], dtype=torch.float32)
            with torch.no_grad():
                pt_output = bn(pt_input).numpy().squeeze()

            assert np.all(pt_output >= out_box.lb.flatten() - 1e-5), (
                f"Sample below lower bound: min diff = {(pt_output - out_box.lb.flatten()).min()}"
            )
            assert np.all(pt_output <= out_box.ub.flatten() + 1e-5), (
                f"Sample above upper bound: max diff = {(pt_output - out_box.ub.flatten()).max()}"
            )


class TestBatchNormFusionSoundness:
    """Soundness tests for BatchNorm fusion with Conv2d."""

    def test_fused_vs_unfused_same_bounds(self):
        """Conv2d + BN2d fused vs unfused should produce same bounds."""
        torch.manual_seed(42)

        conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=True)
        bn = nn.BatchNorm2d(2)

        # Initialize BN with non-trivial running stats by running some data
        bn.train()
        dummy_data = torch.randn(10, 2, 4, 4)
        # Manually set running stats since we can't easily forward through conv+bn pair in training
        bn.eval()
        with torch.no_grad():
            bn.running_mean.copy_(torch.tensor([0.3, -0.1]))
            bn.running_var.copy_(torch.tensor([0.8, 1.2]))
            bn.weight.copy_(torch.tensor([1.2, 0.9]))
            bn.bias.copy_(torch.tensor([0.05, -0.15]))

        # Build unfused model
        unfused_model = nn.Sequential(conv, bn)
        unfused_model.eval()

        # Build fused model
        fused_model = fuse_batchnorm(unfused_model)
        fused_model.eval()

        # Create input ImageStar from bounds [0, 0.5]
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1)) * 0.5
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        # Forward through unfused: Conv2d then BN2d via reach_layer
        unfused_out = reach_layer(unfused_model[0], [img_star])  # Conv2d
        unfused_out = reach_layer(unfused_model[1], unfused_out)  # BN2d
        unfused_star = unfused_out[0]
        lb_unfused, ub_unfused = unfused_star.estimate_ranges()

        # Forward through fused: just the fused Conv2d
        fused_out = reach_layer(fused_model[0], [img_star])  # Fused Conv2d
        fused_star = fused_out[0]
        lb_fused, ub_fused = fused_star.estimate_ranges()

        # Bounds should be close (not identical due to floating point)
        np.testing.assert_allclose(lb_unfused, lb_fused, atol=1e-4), (
            f"Lower bounds differ: max diff = {np.abs(lb_unfused - lb_fused).max()}"
        )
        np.testing.assert_allclose(ub_unfused, ub_fused, atol=1e-4), (
            f"Upper bounds differ: max diff = {np.abs(ub_unfused - ub_fused).max()}"
        )
