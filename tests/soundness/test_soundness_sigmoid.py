"""
Soundness tests for Sigmoid layer reachability.

Tests verify that for random points sampled from the input set,
forwarding through PyTorch Sigmoid produces outputs contained
in the reachable set.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestSigmoidStarApproxSoundness:
    """Soundness tests for approximate Sigmoid with Star sets."""

    def test_positive_region_sampling(self):
        """Sigmoid on [0.5, 2.0] (convex region): all samples within bounds."""
        lb = np.array([[0.5], [1.0]])
        ub = np.array([[2.0], [3.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.Sigmoid()

        result = reach_layer(layer, [star], method='approx')
        out_lb, out_ub = result[0].estimate_ranges()

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()
            assert np.all(pt_output >= out_lb.flatten() - 1e-5), \
                f"Below lb: {(pt_output - out_lb.flatten()).min()}"
            assert np.all(pt_output <= out_ub.flatten() + 1e-5), \
                f"Above ub: {(pt_output - out_ub.flatten()).max()}"

    def test_negative_region_sampling(self):
        """Sigmoid on [-3.0, -0.5] (concave region): all samples within bounds."""
        lb = np.array([[-3.0], [-2.0]])
        ub = np.array([[-0.5], [-0.5]])
        star = Star.from_bounds(lb, ub)
        layer = nn.Sigmoid()

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
        """Sigmoid on [-2.0, 2.0] (mixed region): all samples within bounds."""
        lb = np.array([[-2.0], [-1.0], [0.0]])
        ub = np.array([[2.0], [1.0], [3.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.Sigmoid()

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
        """Sigmoid with method='exact' should still work (falls back to approx)."""
        lb = np.array([[-1.0]])
        ub = np.array([[1.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.Sigmoid()

        result = reach_layer(layer, [star], method='exact')
        assert len(result) == 1


class TestSigmoidZonoSoundness:
    """Soundness tests for Sigmoid with Zonotope sets."""

    def test_zono_sampling(self):
        """Sigmoid Zono: all sampled outputs within bounds."""
        lb = np.array([[-2.0], [-1.0], [0.5]])
        ub = np.array([[1.0], [2.0], [3.0]])
        zono = Zono.from_bounds(lb, ub)
        layer = nn.Sigmoid()

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


class TestSigmoidBoxSoundness:
    """Soundness tests for Sigmoid with Box sets."""

    def test_box_sampling(self):
        """Sigmoid Box: all sampled outputs within bounds."""
        lb = np.array([[-2.0], [-1.0], [0.5]])
        ub = np.array([[1.0], [2.0], [3.0]])
        box = Box(lb, ub)
        layer = nn.Sigmoid()

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
