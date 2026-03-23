"""End-to-end reachability tests for Hexatope and Octatope through NeuralNetwork.reach()."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Hexatope, Octatope, Star
from n2v.nn import NeuralNetwork


class TestHexatopeReachability:
    """End-to-end tests for hexatope reachability."""

    def _make_small_net(self):
        """Create a small 2->3->2 network."""
        net = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        torch.manual_seed(42)
        nn.init.uniform_(net[0].weight, -1, 1)
        nn.init.uniform_(net[0].bias, -0.5, 0.5)
        nn.init.uniform_(net[2].weight, -1, 1)
        nn.init.uniform_(net[2].bias, -0.5, 0.5)
        return net

    def test_approx_reachability_soundness(self):
        """Approx reachability should contain all concrete outputs."""
        net = self._make_small_net()
        nn_verifier = NeuralNetwork(net)

        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_hex = Hexatope.from_bounds(lb, ub)

        output_hexs = nn_verifier.reach(input_hex, method='approx')
        assert len(output_hexs) > 0

        # Sample from input and check containment
        np.random.seed(0)
        input_samples = input_hex.sample(100)
        for sample in input_samples:
            inp = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            out = net(inp).detach().numpy().flatten()
            contained = any(h.contains(out) for h in output_hexs)
            assert contained, f"Output {out} not contained in any output hexatope"

    def test_approx_vs_star_comparison(self):
        """Hexatope approx should produce comparable bounds to Star approx."""
        net = self._make_small_net()
        nn_verifier = NeuralNetwork(net)

        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])

        hex_out = nn_verifier.reach(Hexatope.from_bounds(lb, ub), method='approx')
        star_out = nn_verifier.reach(Star.from_bounds(lb, ub), method='approx')

        # Get bounds from both
        hex_lbs = [h.get_ranges(solver='lp')[0] for h in hex_out]
        hex_ubs = [h.get_ranges(solver='lp')[1] for h in hex_out]
        hex_lb = np.min(hex_lbs, axis=0)
        hex_ub = np.max(hex_ubs, axis=0)

        star_lbs = [s.get_ranges()[0] for s in star_out]
        star_ubs = [s.get_ranges()[1] for s in star_out]
        star_lb = np.min(star_lbs, axis=0)
        star_ub = np.max(star_ubs, axis=0)

        # Hexatope may be wider (less precise) but should overlap significantly
        assert np.all(hex_lb <= star_lb + 0.5), "Hexatope lb much wider than Star"
        assert np.all(hex_ub >= star_ub - 0.5), "Hexatope ub much tighter than Star"


class TestOctatopeReachability:
    """End-to-end tests for octatope reachability."""

    def _make_small_net(self):
        net = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        torch.manual_seed(42)
        nn.init.uniform_(net[0].weight, -1, 1)
        nn.init.uniform_(net[0].bias, -0.5, 0.5)
        nn.init.uniform_(net[2].weight, -1, 1)
        nn.init.uniform_(net[2].bias, -0.5, 0.5)
        return net

    def test_approx_reachability_soundness(self):
        """Approx reachability should contain all concrete outputs."""
        net = self._make_small_net()
        nn_verifier = NeuralNetwork(net)

        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_oct = Octatope.from_bounds(lb, ub)

        output_octs = nn_verifier.reach(input_oct, method='approx')
        assert len(output_octs) > 0

        np.random.seed(0)
        input_samples = input_oct.sample(100)
        for sample in input_samples:
            inp = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            out = net(inp).detach().numpy().flatten()
            contained = any(o.contains(out) for o in output_octs)
            assert contained, f"Output {out} not contained in any output octatope"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
