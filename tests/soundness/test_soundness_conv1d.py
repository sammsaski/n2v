"""
Soundness tests for Conv1D layer reachability.

Conv1D is a linear operation, so it should be exact for all set types.
Tests verify that random samples from the input set always land in the output set.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.conv1d_reach import conv1d_star, conv1d_zono, conv1d_box


class TestConv1DStarSoundness:
    """Soundness tests for Conv1D with Star sets."""

    def _make_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                          padding=padding, bias=bias)
        torch.manual_seed(42)
        nn.init.uniform_(layer.weight, -1, 1)
        if bias:
            nn.init.uniform_(layer.bias, -0.5, 0.5)
        return layer

    def test_identity_kernel(self):
        """Conv1d with kernel_size=1, weight=1 preserves values."""
        layer = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        layer.weight.data = torch.ones(1, 1, 1)

        lb = np.zeros((8, 1))
        ub = np.ones((8, 1))
        star = Star.from_bounds(lb, ub)

        out = conv1d_star(layer, [star])
        lb_out, ub_out = out[0].estimate_ranges()
        assert np.allclose(lb_out, 0.0, atol=1e-6)
        assert np.allclose(ub_out, 1.0, atol=1e-6)

    def test_zero_kernel(self):
        """Conv1d with zero weights produces zero output (plus bias)."""
        layer = nn.Conv1d(1, 2, kernel_size=3, bias=True)
        layer.weight.data.zero_()
        layer.bias.data = torch.tensor([1.0, -1.0])

        lb = np.zeros((8, 1))
        ub = np.ones((8, 1))
        star = Star.from_bounds(lb, ub)

        out = conv1d_star(layer, [star])
        lb_out, ub_out = out[0].estimate_ranges()
        # Output dim = 2 * 6 = 12: first 6 should be 1.0, next 6 should be -1.0
        assert out[0].dim == 12
        assert np.allclose(lb_out[:6], 1.0, atol=1e-6)
        assert np.allclose(ub_out[:6], 1.0, atol=1e-6)
        assert np.allclose(lb_out[6:], -1.0, atol=1e-6)
        assert np.allclose(ub_out[6:], -1.0, atol=1e-6)

    def test_random_samples_in_bounds(self):
        """Random samples from input Star produce outputs within output Star bounds."""
        layer = self._make_layer(1, 4, kernel_size=3)
        np.random.seed(0)
        center = np.random.rand(8, 1)
        star = Star.from_bounds(center - 0.1, center + 0.1)

        out = conv1d_star(layer, [star])
        lb_out, ub_out = out[0].estimate_ranges()

        for _ in range(500):
            x = np.random.uniform(center.flatten() - 0.1, center.flatten() + 0.1)
            x_torch = torch.tensor(x, dtype=torch.float32).reshape(1, 1, 8)
            with torch.no_grad():
                y = layer(x_torch).flatten().numpy().reshape(-1, 1).astype(np.float64)
            assert np.all(y >= lb_out - 1e-4), f"Below lower bound"
            assert np.all(y <= ub_out + 1e-4), f"Above upper bound"

    def test_multi_channel_input(self):
        """Conv1d with multiple input channels."""
        layer = self._make_layer(3, 4, kernel_size=2)
        # Input: 3 channels, length 5 -> flat dim = 15
        np.random.seed(1)
        center = np.random.rand(15, 1)
        star = Star.from_bounds(center - 0.05, center + 0.05)

        out = conv1d_star(layer, [star])
        lb_out, ub_out = out[0].estimate_ranges()

        for _ in range(200):
            x = np.random.uniform(center.flatten() - 0.05, center.flatten() + 0.05)
            x_torch = torch.tensor(x, dtype=torch.float32).reshape(1, 3, 5)
            with torch.no_grad():
                y = layer(x_torch).flatten().numpy().reshape(-1, 1).astype(np.float64)
            assert np.all(y >= lb_out - 1e-4)
            assert np.all(y <= ub_out + 1e-4)

    def test_stride_and_padding(self):
        """Conv1d with stride>1 and padding."""
        layer = self._make_layer(1, 2, kernel_size=3, stride=2, padding=1)
        np.random.seed(2)
        center = np.random.rand(10, 1)
        star = Star.from_bounds(center - 0.1, center + 0.1)

        out = conv1d_star(layer, [star])

        for _ in range(200):
            x = np.random.uniform(center.flatten() - 0.1, center.flatten() + 0.1)
            x_torch = torch.tensor(x, dtype=torch.float32).reshape(1, 1, 10)
            with torch.no_grad():
                y = layer(x_torch).flatten().numpy().reshape(-1, 1).astype(np.float64)
            lb_out, ub_out = out[0].estimate_ranges()
            assert np.all(y >= lb_out - 1e-4)
            assert np.all(y <= ub_out + 1e-4)

    def test_output_dimension(self):
        """Verify output dimension matches PyTorch."""
        layer = self._make_layer(2, 8, kernel_size=4, stride=2, padding=1)
        # Input: 2 channels, length 16 -> flat dim = 32
        star = Star.from_bounds(np.zeros((32, 1)), np.ones((32, 1)))
        out = conv1d_star(layer, [star])

        # PyTorch output shape
        x_torch = torch.randn(1, 2, 16)
        with torch.no_grad():
            y_torch = layer(x_torch)
        expected_dim = y_torch.numel()

        assert out[0].dim == expected_dim


class TestConv1DZonoSoundness:
    """Soundness tests for Conv1D with Zonotope sets."""

    def test_random_samples_in_bounds(self):
        """Random samples from input Zono produce outputs within output Zono bounds."""
        layer = nn.Conv1d(1, 4, kernel_size=3, bias=True)
        torch.manual_seed(42)
        nn.init.uniform_(layer.weight, -1, 1)
        nn.init.uniform_(layer.bias, -0.5, 0.5)

        np.random.seed(0)
        center = np.random.rand(8, 1)
        zono = Zono.from_bounds(center - 0.1, center + 0.1)

        out = conv1d_zono(layer, [zono])
        lb_out, ub_out = out[0].get_bounds()

        for _ in range(500):
            x = np.random.uniform(center.flatten() - 0.1, center.flatten() + 0.1)
            x_torch = torch.tensor(x, dtype=torch.float32).reshape(1, 1, 8)
            with torch.no_grad():
                y = layer(x_torch).flatten().numpy().reshape(-1, 1).astype(np.float64)
            assert np.all(y >= lb_out - 1e-4)
            assert np.all(y <= ub_out + 1e-4)


class TestConv1DBoxSoundness:
    """Soundness tests for Conv1D with Box sets."""

    def test_random_samples_in_bounds(self):
        """Random samples from input Box produce outputs within output Box bounds."""
        layer = nn.Conv1d(1, 4, kernel_size=3, bias=True)
        torch.manual_seed(42)
        nn.init.uniform_(layer.weight, -1, 1)
        nn.init.uniform_(layer.bias, -0.5, 0.5)

        np.random.seed(0)
        center = np.random.rand(8, 1)
        box = Box(center - 0.1, center + 0.1)

        out = conv1d_box(layer, [box])

        for _ in range(500):
            x = np.random.uniform(center.flatten() - 0.1, center.flatten() + 0.1)
            x_torch = torch.tensor(x, dtype=torch.float32).reshape(1, 1, 8)
            with torch.no_grad():
                y = layer(x_torch).flatten().numpy().reshape(-1, 1).astype(np.float64)
            assert np.all(y >= out[0].lb - 1e-4)
            assert np.all(y <= out[0].ub + 1e-4)
