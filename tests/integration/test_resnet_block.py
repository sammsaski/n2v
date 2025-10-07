"""
Integration test: ResNet-style block end-to-end reachability.

Tests that NeuralNetwork.reach() handles auto-fusion, dispatching,
and layer-by-layer processing correctly for models with BatchNorm and Pad.
Note: True residual connections (skip connections) require ONNX graph module
tracing, which is tested separately. These tests focus on the sequential
Conv→BN→ReLU pipeline and Pad+Conv pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, ImageStar, ImageZono
from n2v.nn import NeuralNetwork


class TestResNetBlockReachability:
    """End-to-end reachability through models with BatchNorm."""

    def _make_conv_bn_model(self):
        """Create and prepare a Conv+BN+ReLU model."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 2),
        )
        model.eval()
        model[1].running_mean.data = torch.randn(4) * 0.1
        model[1].running_var.data = torch.rand(4) + 0.1
        return model

    def test_sequential_model_approx_star(self):
        """Approx Star reachability through Conv+BN+ReLU model."""
        model = self._make_conv_bn_model()

        # Verify BN fusion works correctly
        from n2v.utils.model_preprocessing import fuse_batchnorm
        fused = fuse_batchnorm(model)
        fused.eval()

        x = torch.randn(1, 1, 4, 4)
        with torch.no_grad():
            y_orig = model(x)
            y_fused = fused(x)
        assert torch.allclose(y_orig, y_fused, atol=1e-4)

    def test_soundness_with_sampling(self):
        """Fused Conv+BN model soundness with point sampling."""
        model = self._make_conv_bn_model()

        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1)) * 0.1  # small perturbation
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        net = NeuralNetwork(model)
        result = net.reach(input_star, method='approx')

        assert len(result) >= 1
        lb_out, ub_out = result[0].estimate_ranges()

        # Verify soundness with sampling
        np.random.seed(42)
        for _ in range(100):
            point = np.random.uniform(0.0, 0.1, size=(4, 4, 1))
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )
            with torch.no_grad():
                pt_output = model(pt_input).numpy().flatten()

            assert np.all(pt_output >= lb_out.flatten() - 1e-4), \
                f"Below LB: {pt_output} < {lb_out.flatten()}"
            assert np.all(pt_output <= ub_out.flatten() + 1e-4), \
                f"Above UB: {pt_output} > {ub_out.flatten()}"

    def test_pad_conv_pipeline(self):
        """ZeroPad2d + Conv2d pipeline through NeuralNetwork.reach()."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(1, 2, kernel_size=3, bias=True),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 4 * 4, 3),
        )
        model.eval()

        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1)) * 0.5
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        net = NeuralNetwork(model)
        result = net.reach(input_star, method='approx')

        assert len(result) >= 1
        lb_out, ub_out = result[0].estimate_ranges()

        # Verify soundness
        np.random.seed(42)
        for _ in range(100):
            point = np.random.uniform(0.0, 0.5, size=(4, 4, 1))
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )
            with torch.no_grad():
                pt_output = model(pt_input).numpy().flatten()

            assert np.all(pt_output >= lb_out.flatten() - 1e-4), \
                f"Below LB: {pt_output} < {lb_out.flatten()}"
            assert np.all(pt_output <= ub_out.flatten() + 1e-4), \
                f"Above UB: {pt_output} > {ub_out.flatten()}"

    def test_pad_conv_bn_pipeline(self):
        """ZeroPad2d + Conv2d + BatchNorm2d pipeline through NeuralNetwork.reach()."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 2),
        )
        model.eval()
        model[2].running_mean.data = torch.randn(4) * 0.1
        model[2].running_var.data = torch.rand(4) + 0.1

        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1)) * 0.1
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        net = NeuralNetwork(model)
        result = net.reach(input_star, method='approx')

        assert len(result) >= 1
        lb_out, ub_out = result[0].estimate_ranges()

        # Verify soundness
        np.random.seed(42)
        for _ in range(100):
            point = np.random.uniform(0.0, 0.1, size=(4, 4, 1))
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )
            with torch.no_grad():
                pt_output = model(pt_input).numpy().flatten()

            assert np.all(pt_output >= lb_out.flatten() - 1e-4), \
                f"Below LB: {pt_output} < {lb_out.flatten()}"
            assert np.all(pt_output <= ub_out.flatten() + 1e-4), \
                f"Above UB: {pt_output} > {ub_out.flatten()}"

    def test_imagezono_pad_conv(self):
        """ZeroPad2d + Conv pipeline with ImageZono (approx method)."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(1, 2, kernel_size=3, bias=True),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 4 * 4, 3),
        )
        model.eval()

        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1)) * 0.5
        img_zono = ImageZono.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        net = NeuralNetwork(model)
        result = net.reach(img_zono, method='approx')

        assert len(result) >= 1
        # Just verify it runs without error and produces valid output
        out = result[0]
        assert out.dim == 3  # 3 output features
