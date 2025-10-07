"""
Soundness tests for LeakyReLU layer reachability.

Tests verify that for random points sampled from the input set,
forwarding through PyTorch LeakyReLU produces outputs contained
in the reachable set.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestLeakyReLUStarExactSoundness:
    """Soundness tests for exact LeakyReLU with Star sets."""

    def test_all_positive_input(self):
        """LeakyReLU with all-positive input: identity."""
        lb = np.array([[1.0], [1.0]])
        ub = np.array([[2.0], [2.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.1)

        result = reach_layer(layer, [star], method='exact')

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        np.testing.assert_allclose(out_lb, lb, atol=1e-6)
        np.testing.assert_allclose(out_ub, ub, atol=1e-6)

    def test_all_negative_input(self):
        """LeakyReLU with all-negative input: scales by gamma."""
        lb = np.array([[-2.0], [-2.0]])
        ub = np.array([[-1.0], [-1.0]])
        star = Star.from_bounds(lb, ub)
        gamma = 0.1
        layer = nn.LeakyReLU(negative_slope=gamma)

        result = reach_layer(layer, [star], method='exact')

        assert len(result) == 1
        out_lb, out_ub = result[0].estimate_ranges()
        np.testing.assert_allclose(out_lb, gamma * lb, atol=1e-6)
        np.testing.assert_allclose(out_ub, gamma * ub, atol=1e-6)

    def test_crossing_zero_exact_sampling(self):
        """LeakyReLU with crossing zero: sample 200 points, verify containment."""
        lb = np.array([[-1.0], [-0.5], [0.5]])
        ub = np.array([[1.0], [1.0], [2.0]])
        star = Star.from_bounds(lb, ub)
        gamma = 0.2
        layer = nn.LeakyReLU(negative_slope=gamma)

        result = reach_layer(layer, [star], method='exact')

        # Collect union of output bounds
        union_lb = np.ones((3, 1)) * np.inf
        union_ub = np.ones((3, 1)) * -np.inf
        for s in result:
            if not s.is_empty_set():
                s_lb, s_ub = s.estimate_ranges()
                union_lb = np.minimum(union_lb, s_lb)
                union_ub = np.maximum(union_ub, s_ub)

        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(lb.flatten(), ub.flatten())
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pt_output = layer(pt_input).numpy().squeeze()

            assert np.all(pt_output >= union_lb.flatten() - 1e-5), \
                f"Sample below lb: {(pt_output - union_lb.flatten()).min()}"
            assert np.all(pt_output <= union_ub.flatten() + 1e-5), \
                f"Sample above ub: {(pt_output - union_ub.flatten()).max()}"

    def test_different_negative_slopes(self):
        """LeakyReLU soundness across several negative_slope values."""
        lb = np.array([[-2.0], [-1.0]])
        ub = np.array([[1.0], [2.0]])

        for gamma in [0.01, 0.1, 0.3, 0.5]:
            star = Star.from_bounds(lb, ub)
            layer = nn.LeakyReLU(negative_slope=gamma)
            result = reach_layer(layer, [star], method='exact')

            union_lb = np.ones((2, 1)) * np.inf
            union_ub = np.ones((2, 1)) * -np.inf
            for s in result:
                if not s.is_empty_set():
                    s_lb, s_ub = s.estimate_ranges()
                    union_lb = np.minimum(union_lb, s_lb)
                    union_ub = np.maximum(union_ub, s_ub)

            np.random.seed(42)
            for _ in range(50):
                point = np.random.uniform(lb.flatten(), ub.flatten())
                pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pt_output = layer(pt_input).numpy().squeeze()
                assert np.all(pt_output >= union_lb.flatten() - 1e-5)
                assert np.all(pt_output <= union_ub.flatten() + 1e-5)


class TestLeakyReLUStarApproxSoundness:
    """Soundness tests for approximate LeakyReLU with Star sets."""

    def test_approx_contains_exact(self):
        """Approx LeakyReLU should over-approximate exact result."""
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.1)

        exact = reach_layer(layer, [star], method='exact')
        approx = reach_layer(layer, [star], method='approx')

        # Exact union bounds (LP-based)
        exact_lb = np.ones((2, 1)) * np.inf
        exact_ub = np.ones((2, 1)) * -np.inf
        for s in exact:
            if not s.is_empty_set():
                s_lb, s_ub = s.get_ranges()
                exact_lb = np.minimum(exact_lb, s_lb)
                exact_ub = np.maximum(exact_ub, s_ub)

        # Approx bounds
        approx_lb, approx_ub = approx[0].estimate_ranges()

        assert np.all(approx_lb <= exact_lb + 1e-6)
        assert np.all(exact_ub <= approx_ub + 1e-6)

    def test_approx_sampling(self):
        """Approx LeakyReLU: all sampled outputs within bounds."""
        lb = np.array([[-2.0], [-1.0], [0.5]])
        ub = np.array([[1.0], [2.0], [3.0]])
        star = Star.from_bounds(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.2)

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


class TestLeakyReLUZonoSoundness:
    """Soundness tests for LeakyReLU with Zonotope sets."""

    def test_zono_sampling(self):
        """LeakyReLU Zono: all sampled outputs within bounds."""
        lb = np.array([[-1.0], [-2.0], [0.0]])
        ub = np.array([[1.0], [1.0], [2.0]])
        zono = Zono.from_bounds(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.1)

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


class TestLeakyReLUBoxSoundness:
    """Soundness tests for LeakyReLU with Box sets."""

    def test_box_sampling(self):
        """LeakyReLU Box: all sampled outputs within bounds."""
        lb = np.array([[-1.0], [-2.0], [0.5]])
        ub = np.array([[1.0], [1.0], [2.0]])
        box = Box(lb, ub)
        layer = nn.LeakyReLU(negative_slope=0.2)

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
