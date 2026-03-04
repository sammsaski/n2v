"""
Soundness tests for Tanh layer reachability.

Tests verify that for random points sampled from the input set,
forwarding through PyTorch Tanh produces outputs contained
in the reachable set.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestTanhStarApproxSoundness:
    """Soundness tests for approximate Tanh with Star sets."""

    def test_positive_region_sampling(self):
        """Tanh on [0.5, 2.0] (convex region): all samples within bounds."""
        lb = np.array([[0.5], [1.0]])
        ub = np.array([[2.0], [3.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.Tanh()

        result = reach_layer(layer, [star], method='approx')
        out_lb, out_ub = result[0].estimate_ranges()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()
            assert np.all(pt_output >= out_lb.flatten() - 1e-5)
            assert np.all(pt_output <= out_ub.flatten() + 1e-5)

    def test_negative_region_sampling(self):
        """Tanh on [-3.0, -0.5] (concave region): all samples within bounds."""
        lb = np.array([[-3.0], [-2.0]])
        ub = np.array([[-0.5], [-0.5]])
        star = Star.from_bounds(lb, ub)
        layer = nn.Tanh()

        result = reach_layer(layer, [star], method='approx')
        out_lb, out_ub = result[0].estimate_ranges()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()
            assert np.all(pt_output >= out_lb.flatten() - 1e-5)
            assert np.all(pt_output <= out_ub.flatten() + 1e-5)

    def test_mixed_region_sampling(self):
        """Tanh on [-2.0, 2.0] (mixed region): all samples within bounds."""
        lb = np.array([[-2.0], [-1.0], [0.0]])
        ub = np.array([[2.0], [1.0], [3.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.Tanh()

        result = reach_layer(layer, [star], method='approx')
        out_lb, out_ub = result[0].estimate_ranges()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()
            assert np.all(pt_output >= out_lb.flatten() - 1e-5)
            assert np.all(pt_output <= out_ub.flatten() + 1e-5)

    def test_exact_falls_back_to_approx(self):
        """Tanh with method='exact' should still work (falls back to approx)."""
        lb = np.array([[-1.0]])
        ub = np.array([[1.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.Tanh()

        result = reach_layer(layer, [star], method='exact')
        assert len(result) == 1


class TestTanhZonoSoundness:
    """Soundness tests for Tanh with Zonotope sets."""

    def test_zono_sampling(self):
        """Tanh Zono: all sampled outputs within bounds."""
        lb = np.array([[-2.0], [-1.0], [0.5]])
        ub = np.array([[1.0], [2.0], [3.0]])
        zono = Zono.from_bounds(lb, ub)
        layer = nn.Tanh()

        result = reach_layer(layer, [zono], method='approx')
        out_lb, out_ub = result[0].get_bounds()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()
            assert np.all(pt_output >= out_lb.flatten() - 1e-5)
            assert np.all(pt_output <= out_ub.flatten() + 1e-5)


class TestTanhBoxSoundness:
    """Soundness tests for Tanh with Box sets."""

    def test_box_sampling(self):
        """Tanh Box: all sampled outputs within bounds."""
        lb = np.array([[-2.0], [-1.0], [0.5]])
        ub = np.array([[1.0], [2.0], [3.0]])
        box = Box(lb, ub)
        layer = nn.Tanh()

        result = reach_layer(layer, [box], method='approx')
        out = result[0]

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()
            assert np.all(pt_output >= out.lb.flatten() - 1e-5)
            assert np.all(pt_output <= out.ub.flatten() + 1e-5)
