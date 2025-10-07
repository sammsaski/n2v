"""Tests for Zonotope pre-pass bounds precomputation."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono


class TestComputeIntermediateBounds:
    """Tests for compute_intermediate_bounds on Sequential models."""

    def _make_fc_model(self):
        """Create a small FC model: Linear(3,5) -> ReLU -> Linear(5,3) -> ReLU -> Linear(3,2)."""
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        model.eval()
        return model

    def test_returns_bounds_at_each_relu(self):
        """Pre-pass should return bounds keyed by layer index for each ReLU."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)

        # model layers: 0=Linear, 1=ReLU, 2=Linear, 3=ReLU, 4=Linear
        # Should have bounds at layer indices 1 and 3 (the two ReLUs)
        assert 1 in layer_bounds, "Should have bounds for first ReLU (layer 1)"
        assert 3 in layer_bounds, "Should have bounds for second ReLU (layer 3)"
        assert len(layer_bounds) == 2

    def test_bounds_shapes_match_layer_dims(self):
        """Bounds should have shape matching the layer input dimension."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)

        # First ReLU input has dim 5 (output of Linear(3,5))
        pre_lb, pre_ub = layer_bounds[1]
        assert pre_lb.shape == (5, 1)
        assert pre_ub.shape == (5, 1)

        # Second ReLU input has dim 3 (output of Linear(5,3))
        pre_lb, pre_ub = layer_bounds[3]
        assert pre_lb.shape == (3, 1)
        assert pre_ub.shape == (3, 1)

    def test_bounds_contain_sampled_points(self):
        """Pre-pass bounds must contain all concrete activations (soundness)."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)

        # Sample 500 points from input range
        np.random.seed(42)
        n_samples = 500
        samples = lb + (ub - lb) * np.random.rand(3, n_samples)

        # Forward through model layer by layer, check each ReLU input
        layers = list(model.children())
        for s in range(n_samples):
            x = torch.tensor(samples[:, s:s+1].T, dtype=torch.float32)
            for i, layer in enumerate(layers):
                if i in layer_bounds:
                    # Check: activation at this point is within bounds
                    activation = x.detach().numpy().flatten()
                    pre_lb, pre_ub = layer_bounds[i]
                    assert np.all(activation >= pre_lb.flatten() - 1e-6), \
                        f"Sample {s} below lower bound at layer {i}"
                    assert np.all(activation <= pre_ub.flatten() + 1e-6), \
                        f"Sample {s} above upper bound at layer {i}"
                x = layer(x)

    def test_lb_leq_ub(self):
        """Lower bounds must be <= upper bounds everywhere."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-0.5], [-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5], [0.5]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)

        for layer_id, (pre_lb, pre_ub) in layer_bounds.items():
            assert np.all(pre_lb <= pre_ub + 1e-10), \
                f"lb > ub at layer {layer_id}"

    def test_accepts_box_input(self):
        """Should accept Box as input set."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Box(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)
        assert len(layer_bounds) == 2

    def test_accepts_zono_input(self):
        """Should accept Zono as input set."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Zono.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)
        assert len(layer_bounds) == 2

    def test_leakyrelu_detected(self):
        """Should also store bounds before LeakyReLU layers."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = nn.Sequential(
            nn.Linear(3, 4),
            nn.LeakyReLU(0.1),
            nn.Linear(4, 2),
        )
        model.eval()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)
        assert 1 in layer_bounds  # LeakyReLU at index 1
        pre_lb, pre_ub = layer_bounds[1]
        assert pre_lb.shape == (4, 1)

    def test_stable_neurons_identified(self):
        """With biased weights, some neurons should be provably stable."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        # Create model where first Linear has large positive bias
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        model.eval()
        # Set bias so that output of Linear is always positive for small inputs
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(3, 2) * 0.1)
            model[0].bias.copy_(torch.tensor([10.0, -10.0, 0.0]))

        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)
        pre_lb, pre_ub = layer_bounds[1]

        # Neuron 0: bias=10, weight*input in [-0.1, 0.1] -> always active (lb >= 0)
        assert pre_lb[0, 0] >= 0, "Neuron 0 should be provably active"
        # Neuron 1: bias=-10, weight*input in [-0.1, 0.1] -> always inactive (ub <= 0)
        assert pre_ub[1, 0] <= 0, "Neuron 1 should be provably inactive"
