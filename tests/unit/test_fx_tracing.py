"""Tests for torch.fx tracing support — inline activations and functional ops."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from n2v.sets import Star
from n2v.nn import NeuralNetwork


class InlineReLUModel(nn.Module):
    """Model with inline ReLU (not registered as submodule) — untraceable."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


class FunctionalReLUModel(nn.Module):
    """Model using F.relu() functional call."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TorchReLUModel(nn.Module):
    """Model using torch.relu() call."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class InlineFlattenModel(nn.Module):
    """Model with inline Flatten (not registered)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x


class RegisteredReLUModel(nn.Module):
    """Reference model with ReLU registered as submodule (should always work)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class UntraceableModel(nn.Module):
    """Model with data-dependent control flow (cannot be traced)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 2)
        self.fc3 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        if x.sum() > 0:
            return self.fc2(x)
        return self.fc3(x)


def _make_input_star():
    """Helper to create a 3D input Star."""
    lb = np.array([[0.0], [0.0], [0.0]])
    ub = np.array([[1.0], [1.0], [1.0]])
    return Star.from_bounds(lb, ub)


def _get_reference_bounds(model, input_star):
    """Get bounds by sampling — ground truth for comparison.

    Samples many points from the input region, runs them through the model,
    and returns the min/max output values observed.
    """
    # Get bounds from the Star (estimate_ranges computes and caches them)
    star_lb, star_ub = input_star.estimate_ranges()
    lb = star_lb.flatten()
    ub = star_ub.flatten()
    n_samples = 10000
    samples = np.random.uniform(lb, ub, size=(n_samples, len(lb)))
    with torch.no_grad():
        outputs = model(torch.tensor(samples, dtype=torch.float32)).numpy()
    return outputs.min(axis=0), outputs.max(axis=0)


class TestInlineActivationRegression:
    """Regression tests for issue #3: inline activations silently skipped."""

    def _verify_model_reach(self, model, method='approx'):
        """Verify reach() output bounds contain sampled bounds (soundness)."""
        model.eval()
        input_star = _make_input_star()
        output_stars = NeuralNetwork(model).reach(input_star, method=method)

        # Get bounds from reach output
        reach_lb_list, reach_ub_list = [], []
        for star in output_stars:
            star_lb, star_ub = star.get_ranges()
            reach_lb_list.append(star_lb.flatten())
            reach_ub_list.append(star_ub.flatten())
        reach_lb = np.min(np.stack(reach_lb_list), axis=0)
        reach_ub = np.max(np.stack(reach_ub_list), axis=0)

        # Get sampled bounds (ground truth)
        sample_lb, sample_ub = _get_reference_bounds(model, input_star)

        # Reach bounds must contain sampled bounds (soundness)
        np.testing.assert_array_less(
            reach_lb - 1e-6, sample_lb,
            err_msg="Reach lower bound is above sampled minimum — unsound!"
        )
        np.testing.assert_array_less(
            sample_ub, reach_ub + 1e-6,
            err_msg="Reach upper bound is below sampled maximum — unsound!"
        )

    def test_inline_relu_raises_error(self):
        """Issue #3: inline nn.ReLU() must raise a clear error, not silently skip."""
        torch.manual_seed(42)
        model = InlineReLUModel()
        model.eval()
        input_star = _make_input_star()

        with pytest.raises(TypeError, match="F.relu"):
            NeuralNetwork(model).reach(input_star, method='approx')

    def test_functional_relu(self):
        """F.relu() must be captured by tracing."""
        torch.manual_seed(42)
        self._verify_model_reach(FunctionalReLUModel())

    def test_torch_relu(self):
        """torch.relu() must be captured by tracing."""
        torch.manual_seed(42)
        self._verify_model_reach(TorchReLUModel())

    def test_inline_flatten(self):
        """Inline torch.flatten() must be captured by tracing."""
        torch.manual_seed(42)
        self._verify_model_reach(InlineFlattenModel())

    def test_functional_sigmoid(self):
        """torch.sigmoid must be captured by tracing."""
        class SigmoidModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 5)
                self.fc2 = nn.Linear(5, 2)
            def forward(self, x):
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        torch.manual_seed(42)
        self._verify_model_reach(SigmoidModel())

    def test_functional_tanh(self):
        """torch.tanh must be captured by tracing."""
        class TanhModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 5)
                self.fc2 = nn.Linear(5, 2)
            def forward(self, x):
                x = torch.tanh(self.fc1(x))
                x = self.fc2(x)
                return x
        torch.manual_seed(42)
        self._verify_model_reach(TanhModel())

    def test_functional_leaky_relu(self):
        """F.leaky_relu with custom negative_slope must be captured."""
        class LeakyReLUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 5)
                self.fc2 = nn.Linear(5, 2)
            def forward(self, x):
                x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
                x = self.fc2(x)
                return x
        torch.manual_seed(42)
        self._verify_model_reach(LeakyReLUModel())

    def test_functional_vs_registered_parity(self):
        """Functional F.relu and registered nn.ReLU must produce identical bounds."""
        torch.manual_seed(42)
        functional_model = FunctionalReLUModel()
        registered_model = RegisteredReLUModel()

        # Copy weights so they're identical
        with torch.no_grad():
            registered_model.fc1.weight.copy_(functional_model.fc1.weight)
            registered_model.fc1.bias.copy_(functional_model.fc1.bias)
            registered_model.fc2.weight.copy_(functional_model.fc2.weight)
            registered_model.fc2.bias.copy_(functional_model.fc2.bias)

        input_star = _make_input_star()
        functional_stars = NeuralNetwork(functional_model).reach(input_star, method='approx')
        registered_stars = NeuralNetwork(registered_model).reach(input_star, method='approx')

        # Both should produce same bounds
        for f_star, r_star in zip(functional_stars, registered_stars):
            f_lb, f_ub = f_star.get_ranges()
            r_lb, r_ub = r_star.get_ranges()
            np.testing.assert_allclose(f_lb, r_lb, atol=1e-6)
            np.testing.assert_allclose(f_ub, r_ub, atol=1e-6)


class TestUntraceableModelError:
    """Test that untraceable models produce a clear error."""

    def test_untraceable_model_raises_typeerror(self):
        """Models with data-dependent control flow must raise TypeError."""
        model = UntraceableModel()
        model.eval()
        input_star = _make_input_star()

        with pytest.raises(TypeError, match="traceable by torch.fx"):
            NeuralNetwork(model).reach(input_star, method='approx')

    def test_inline_module_error_message_suggests_fix(self):
        """Error for inline nn.ReLU()(x) must suggest F.relu(x) as alternative."""
        model = InlineReLUModel()
        model.eval()

        with pytest.raises(TypeError, match="F.relu.*instead of.*nn.ReLU"):
            NeuralNetwork(model)


class TestSequentialModelParity:
    """Verify nn.Sequential models still work identically after the change."""

    def test_sequential_feedforward(self):
        """nn.Sequential model produces same results as before."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        model.eval()

        input_star = _make_input_star()
        output_stars = NeuralNetwork(model).reach(input_star, method='approx')

        assert len(output_stars) >= 1
        for star in output_stars:
            assert star.dim == 2
            pytest.assert_star_valid(star)

        # Soundness check
        sample_lb, sample_ub = _get_reference_bounds(model, input_star)
        star_lb, star_ub = output_stars[0].get_ranges()
        np.testing.assert_array_less(star_lb.flatten() - 1e-6, sample_lb)
        np.testing.assert_array_less(sample_ub, star_ub.flatten() + 1e-6)


class TestNeuralNetworkLayers:
    """Test that NeuralNetwork.layers reflects all ops after fx tracing."""

    def test_inline_relu_raises_in_init(self):
        """NeuralNetwork.__init__ must raise TypeError for inline nn.ReLU()."""
        model = InlineReLUModel()
        with pytest.raises(TypeError, match="F.relu"):
            NeuralNetwork(model)

    def test_functional_relu_layer_count(self):
        """NeuralNetwork.layers must include F.relu as a layer."""
        model = FunctionalReLUModel()
        net = NeuralNetwork(model)
        # fc1, relu, fc2 = 3 layers
        assert len(net.layers) == 3, (
            f"Expected 3 layers (fc1, relu, fc2), got {len(net.layers)}: "
            f"{[type(l).__name__ for l in net.layers]}"
        )
