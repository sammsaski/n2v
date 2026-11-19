"""Tests for torch.fx tracing support — inline activations and functional ops."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from n2v.sets import Star
from n2v.nn import NeuralNetwork
from tests.cell_support import assert_covered, replay_torch, samples


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


def _assert_holds_every_output(output_stars, model, what):
    """The approx method relaxes and never splits, so one input set gives one
    output set (a reach that loses ``method`` runs the splitting exact
    method); every output of the model, replayed in float64 on 128 points of
    the input box [0, 1]^3 (corners included), must be a member of a
    returned set by an LP on that set's own rows. (A comparison of the
    result's bounding box with the extremes of 10000 float32 samples and a
    slack of 1e-6 would let a wrong row pass.)"""
    assert len(output_stars) == 1, (
        f"{what}: the approx method returned {len(output_stars)} sets")
    assert_covered(output_stars, replay_torch(
        model, samples(np.zeros(3), np.ones(3)), (1, 3)), what)


class TestInlineActivationRegression:
    """Regression tests for issue #3: inline activations silently skipped."""

    def _verify_model_reach(self, model, method='approx'):
        """Verify that the reach result holds every replayed output."""
        model.eval()
        input_star = _make_input_star()
        output_stars = NeuralNetwork(model).reach(input_star, method=method)
        _assert_holds_every_output(output_stars, model,
                                   f"{type(model).__name__} approx reach")

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

        # Each holds every replayed output (the independent reference); the
        # two spellings take the same rule, so their bounds agree as well.
        _assert_holds_every_output(functional_stars, functional_model,
                                   "F.relu model")
        _assert_holds_every_output(registered_stars, registered_model,
                                   "nn.ReLU model")
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
        """Error for inline nn.ReLU()(x) must suggest F.relu(x) as alternative.

        ``NeuralNetwork.__init__`` defers the trace error so probabilistic
        methods can still run on un-traceable models; the error surfaces
        when a sound method is actually invoked.
        """
        model = InlineReLUModel()
        model.eval()
        input_star = _make_input_star()

        with pytest.raises(TypeError, match="F.relu.*instead of.*nn.ReLU"):
            NeuralNetwork(model).reach(input_star, method='approx')

    def test_untraceable_model_allows_probabilistic_methods(self):
        """Un-traceable models can still construct and reach via probabilistic
        methods (which don't need the layer inventory).

        Regression test for the post-NeurIPS refactor: ``NeuralNetwork()``
        used to eagerly ``torch.fx``-trace the model in ``__init__``,
        causing the ``flow_matching`` / ``conformal`` paths to fail for
        any model with data-dependent control flow (e.g. the ACAS Xu
        wrapper's ``if x.dim() == 2:`` reshape branch). The fix makes
        ``layers`` a lazy property — probabilistic methods never read
        it, so untraceable models work for those paths; the trace only
        runs (and raises) when sound methods need the layer inventory.
        """
        from n2v.sets import Box
        from n2v.probabilistic import FlowReachConfig
        import numpy as np
        model = UntraceableModel()  # 3-D input -> 2-D output, data-dep ctrl flow
        model.eval()

        # Construction succeeds (no eager trace).
        net = NeuralNetwork(model)
        assert net._layers_cache is None, (
            'Expected layers cache empty until first access')

        # Probabilistic reach proceeds — never reads ``layers``.
        # (UntraceableModel takes 3-D input.)
        input_box = Box(np.array([-0.1, -0.1, -0.1]),
                        np.array([ 0.1,  0.1,  0.1]))
        prob_set = net.reach(
            input_box, method='flow_matching',
            config=FlowReachConfig(
                epsilon=0.001, m=200, ell=199,
                n_train=200, flow_epochs=20, flow_config='base',
                seed=0,
            ),
        )
        assert prob_set is not None

        # And reading the ``layers`` property directly does raise.
        with pytest.raises(TypeError, match="traceable by torch.fx"):
            _ = net.layers


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

        for star in output_stars:
            assert star.dim == 2
            pytest.assert_star_valid(star)
        _assert_holds_every_output(output_stars, model,
                                   "nn.Sequential approx reach")


class TestNeuralNetworkLayers:
    """Test that NeuralNetwork.layers reflects all ops after fx tracing."""

    def test_inline_relu_raises_on_sound_reach(self):
        """Inline nn.ReLU() trace failure must surface on sound .reach().

        ``NeuralNetwork.__init__`` defers the trace error so probabilistic
        methods still work; the error surfaces when a sound method
        actually needs the layer inventory.
        """
        model = InlineReLUModel()
        input_star = _make_input_star()
        with pytest.raises(TypeError, match="F.relu"):
            NeuralNetwork(model).reach(input_star, method='approx')

    def test_functional_relu_layer_count(self):
        """NeuralNetwork.layers must include F.relu as a layer."""
        model = FunctionalReLUModel()
        net = NeuralNetwork(model)
        # fc1, relu, fc2 = 3 layers
        assert len(net.layers) == 3, (
            f"Expected 3 layers (fc1, relu, fc2), got {len(net.layers)}: "
            f"{[type(l).__name__ for l in net.layers]}"
        )
