"""Tests for reach pipeline with precompute_bounds option."""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star
from n2v.nn import NeuralNetwork
from n2v.nn.reach import reach_pytorch_model


class TestReachWithPrecomputeBounds:

    def _make_model(self):
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        model.eval()
        return model

    def test_precompute_bounds_produces_valid_output(self):
        """reach with precompute_bounds=True should produce valid Star sets."""
        model = self._make_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        result = reach_pytorch_model(
            model, input_set, method='approx', precompute_bounds=True
        )
        assert len(result) >= 1
        assert result[0].V is not None

    def test_precompute_bounds_is_sound(self):
        """Output with precompute_bounds must contain all concrete outputs."""
        model = self._make_model()
        lb = np.array([[-0.5], [-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5], [0.5]])
        input_set = Star.from_bounds(lb, ub)

        result = reach_pytorch_model(
            model, input_set, method='approx', precompute_bounds=True
        )

        # Sample and verify containment
        np.random.seed(42)
        n_samples = 500
        samples = lb + (ub - lb) * np.random.rand(3, n_samples)

        out_lb, out_ub = result[0].estimate_ranges()

        for i in range(n_samples):
            x = torch.tensor(samples[:, i:i+1].T, dtype=torch.float32)
            with torch.no_grad():
                y = model(x).numpy().flatten()
            assert np.all(y >= out_lb.flatten() - 1e-5), \
                f"Sample {i} output below lower bound"
            assert np.all(y <= out_ub.flatten() + 1e-5), \
                f"Sample {i} output above upper bound"

    def test_precompute_false_is_default(self):
        """precompute_bounds=False should match default behavior exactly."""
        model = self._make_model()
        lb = np.array([[-0.5], [-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5], [0.5]])
        input_set = Star.from_bounds(lb, ub)

        result_default = reach_pytorch_model(model, input_set, method='approx')
        result_false = reach_pytorch_model(
            model, input_set, method='approx', precompute_bounds=False
        )

        lb_d, ub_d = result_default[0].estimate_ranges()
        lb_f, ub_f = result_false[0].estimate_ranges()
        np.testing.assert_allclose(lb_d, lb_f, atol=1e-10)
        np.testing.assert_allclose(ub_d, ub_f, atol=1e-10)

    def test_exact_with_precompute(self):
        """Exact method with precompute_bounds should also work."""
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        model.eval()
        lb = np.array([[-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5]])
        input_set = Star.from_bounds(lb, ub)

        result = reach_pytorch_model(
            model, input_set, method='exact', precompute_bounds=True
        )
        assert len(result) >= 1

    def test_neural_network_reach_forwards_precompute(self):
        """NeuralNetwork.reach() should forward precompute_bounds."""
        model = self._make_model()
        net = NeuralNetwork(model)
        lb = np.array([[-0.5], [-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5], [0.5]])
        input_set = Star.from_bounds(lb, ub)

        # Should not raise
        result = net.reach(input_set, method='approx', precompute_bounds=True)
        assert len(result) >= 1
