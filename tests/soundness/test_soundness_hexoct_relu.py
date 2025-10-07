"""
Soundness tests for Hexatope/Octatope ReLU reachability.

These tests verify that hex/oct ReLU output bounds CONTAIN the Star exact
bounds (ground truth). If hex/oct bounds are tighter than Star exact, the
over-approximation is unsound.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Star, Hexatope, Octatope
from n2v.nn import NeuralNetwork


class TestHexatopeReLUSoundness:
    """Hex/Oct approx ReLU must produce bounds that contain Star exact bounds."""

    def _build_model(self):
        """2->3->1 network with fixed weights for reproducibility."""
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        model.eval()
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([
                [ 1.0,  0.5],
                [-0.5,  1.0],
                [ 0.3, -0.7],
            ]))
            model[0].bias.copy_(torch.tensor([0.1, -0.2, 0.0]))
            model[2].weight.copy_(torch.tensor([[0.6, -0.4, 0.8]]))
            model[2].bias.copy_(torch.tensor([0.05]))
        return model

    def test_hexatope_approx_contains_star_exact(self):
        """Hexatope approx bounds must contain Star exact bounds."""
        model = self._build_model()
        net = NeuralNetwork(model)

        lb = np.array([[-1.0], [-1.0]], dtype=np.float64)
        ub = np.array([[ 1.0], [ 1.0]], dtype=np.float64)

        # Star exact = ground truth (tightest possible)
        star_out = net.reach(Star.from_bounds(lb, ub), method='exact')
        star_lbs = [s.get_ranges()[0] for s in star_out]
        star_ubs = [s.get_ranges()[1] for s in star_out]
        star_lb = np.min(star_lbs, axis=0)
        star_ub = np.max(star_ubs, axis=0)

        # Hexatope approx must contain star exact
        hex_out = net.reach(Hexatope.from_bounds(lb, ub), method='approx')
        hex_lbs = [s.get_ranges(solver='lp')[0] for s in hex_out]
        hex_ubs = [s.get_ranges(solver='lp')[1] for s in hex_out]
        hex_lb = np.min(hex_lbs, axis=0)
        hex_ub = np.max(hex_ubs, axis=0)

        # Soundness: hex must be wider than or equal to star
        np.testing.assert_array_less(hex_lb, star_lb + 1e-6,
            err_msg="Hexatope LB is tighter than Star exact — UNSOUND")
        np.testing.assert_array_less(star_ub, hex_ub + 1e-6,
            err_msg="Hexatope UB is tighter than Star exact — UNSOUND")

    def test_octatope_approx_contains_star_exact(self):
        """Octatope approx bounds must contain Star exact bounds."""
        model = self._build_model()
        net = NeuralNetwork(model)

        lb = np.array([[-1.0], [-1.0]], dtype=np.float64)
        ub = np.array([[ 1.0], [ 1.0]], dtype=np.float64)

        star_out = net.reach(Star.from_bounds(lb, ub), method='exact')
        star_lbs = [s.get_ranges()[0] for s in star_out]
        star_ubs = [s.get_ranges()[1] for s in star_out]
        star_lb = np.min(star_lbs, axis=0)
        star_ub = np.max(star_ubs, axis=0)

        oct_out = net.reach(Octatope.from_bounds(lb, ub), method='approx')
        oct_lbs = [s.get_ranges(solver='lp')[0] for s in oct_out]
        oct_ubs = [s.get_ranges(solver='lp')[1] for s in oct_out]
        oct_lb = np.min(oct_lbs, axis=0)
        oct_ub = np.max(oct_ubs, axis=0)

        np.testing.assert_array_less(oct_lb, star_lb + 1e-6,
            err_msg="Octatope LB is tighter than Star exact — UNSOUND")
        np.testing.assert_array_less(star_ub, oct_ub + 1e-6,
            err_msg="Octatope UB is tighter than Star exact — UNSOUND")

    def test_hexatope_approx_contains_star_exact_larger_network(self):
        """Soundness on 5->10->5->1 network."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(5, 10), nn.ReLU(),
            nn.Linear(10, 5), nn.ReLU(),
            nn.Linear(5, 1),
        )
        model.eval()
        net = NeuralNetwork(model)

        center = np.array([[0.5], [-0.3], [0.1], [0.8], [-0.2]])
        lb = (center - 0.05).astype(np.float64)
        ub = (center + 0.05).astype(np.float64)

        star_out = net.reach(Star.from_bounds(lb, ub), method='exact')
        star_lbs = [s.get_ranges()[0] for s in star_out]
        star_ubs = [s.get_ranges()[1] for s in star_out]
        star_lb = np.min(star_lbs, axis=0)
        star_ub = np.max(star_ubs, axis=0)

        hex_out = net.reach(Hexatope.from_bounds(lb, ub), method='approx')
        hex_lbs = [s.get_ranges(solver='lp')[0] for s in hex_out]
        hex_ubs = [s.get_ranges(solver='lp')[1] for s in hex_out]
        hex_lb = np.min(hex_lbs, axis=0)
        hex_ub = np.max(hex_ubs, axis=0)

        np.testing.assert_array_less(hex_lb, star_lb + 1e-6,
            err_msg="Hexatope LB is tighter than Star exact — UNSOUND")
        np.testing.assert_array_less(star_ub, hex_ub + 1e-6,
            err_msg="Hexatope UB is tighter than Star exact — UNSOUND")

    def test_octatope_approx_contains_star_exact_larger_network(self):
        """Soundness on 5->10->5->1 network."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(5, 10), nn.ReLU(),
            nn.Linear(10, 5), nn.ReLU(),
            nn.Linear(5, 1),
        )
        model.eval()
        net = NeuralNetwork(model)

        center = np.array([[0.5], [-0.3], [0.1], [0.8], [-0.2]])
        lb = (center - 0.05).astype(np.float64)
        ub = (center + 0.05).astype(np.float64)

        star_out = net.reach(Star.from_bounds(lb, ub), method='exact')
        star_lbs = [s.get_ranges()[0] for s in star_out]
        star_ubs = [s.get_ranges()[1] for s in star_out]
        star_lb = np.min(star_lbs, axis=0)
        star_ub = np.max(star_ubs, axis=0)

        oct_out = net.reach(Octatope.from_bounds(lb, ub), method='approx')
        oct_lbs = [s.get_ranges(solver='lp')[0] for s in oct_out]
        oct_ubs = [s.get_ranges(solver='lp')[1] for s in oct_out]
        oct_lb = np.min(oct_lbs, axis=0)
        oct_ub = np.max(oct_ubs, axis=0)

        np.testing.assert_array_less(oct_lb, star_lb + 1e-6,
            err_msg="Octatope LB is tighter than Star exact — UNSOUND")
        np.testing.assert_array_less(star_ub, oct_ub + 1e-6,
            err_msg="Octatope UB is tighter than Star exact — UNSOUND")


class TestGetRangesValidation:
    """Test that get_ranges never returns lb > ub."""

    def test_hexatope_get_ranges_lb_le_ub(self):
        """get_ranges must always return lb <= ub."""
        lb = np.array([[-1.0], [-1.0]], dtype=np.float64)
        ub = np.array([[ 1.0], [ 1.0]], dtype=np.float64)
        h = Hexatope.from_bounds(lb, ub)
        result_lb, result_ub = h.get_ranges(solver='lp')
        assert np.all(result_lb <= result_ub + 1e-10), \
            f"lb > ub: lb={result_lb.flatten()}, ub={result_ub.flatten()}"

    def test_octatope_get_ranges_lb_le_ub(self):
        """get_ranges must always return lb <= ub."""
        lb = np.array([[-1.0], [-1.0]], dtype=np.float64)
        ub = np.array([[ 1.0], [ 1.0]], dtype=np.float64)
        o = Octatope.from_bounds(lb, ub)
        result_lb, result_ub = o.get_ranges(solver='lp')
        assert np.all(result_lb <= result_ub + 1e-10), \
            f"lb > ub: lb={result_lb.flatten()}, ub={result_ub.flatten()}"
