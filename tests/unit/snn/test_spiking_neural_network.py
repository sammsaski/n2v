"""
Unit tests for SpikingNeuralNetwork: construction, forward, and reach().
"""

import pytest
pytest.importorskip("snntorch", reason="snntorch not installed; pip install n2v[snn]")

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Star
from n2v.snn.model import F2FMLP
from n2v.nn.spiking_neural_network import SpikingNeuralNetwork, SNNReachConfig


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSpikingNeuralNetworkConstruction:

    def test_valid_construction(self, tiny_model):
        snn = SpikingNeuralNetwork(tiny_model)
        assert snn.model is tiny_model
        assert snn.input_size == 4
        assert snn.output_size == (3,)

    def test_model_set_to_eval(self, tiny_model):
        snn = SpikingNeuralNetwork(tiny_model)
        assert not snn.model.training

    def test_non_module_raises_type_error(self):
        with pytest.raises(TypeError, match="nn.Module"):
            SpikingNeuralNetwork("not_a_module")  # type: ignore

    def test_matching_input_size_accepted(self, tiny_model):
        snn = SpikingNeuralNetwork(tiny_model, input_size=4)
        assert snn.input_size == 4

    def test_mismatched_input_size_raises(self, tiny_model):
        with pytest.raises(ValueError):
            SpikingNeuralNetwork(tiny_model, input_size=999)

    def test_infers_input_size_from_model(self, tiny_model):
        snn = SpikingNeuralNetwork(tiny_model)
        assert snn.input_size == tiny_model.fcs[0].in_features

    def test_infers_output_size_from_model(self, tiny_model):
        snn = SpikingNeuralNetwork(tiny_model)
        assert snn.output_size == (tiny_model.fcs[-1].out_features,)

    def test_repr_contains_sizes(self, tiny_model):
        snn = SpikingNeuralNetwork(tiny_model)
        r = repr(snn)
        assert "input_size=4" in r
        assert "output_size=(3,)" in r

    def test_no_layers_attribute(self, tiny_model):
        snn = SpikingNeuralNetwork(tiny_model)
        assert not hasattr(snn, 'layers')


# ---------------------------------------------------------------------------
# forward()
# ---------------------------------------------------------------------------

class TestSpikingNeuralNetworkForward:

    def test_forward_2d_input(self, snn_wrapper):
        x = torch.rand(2, 4).clamp(min=0.1)
        out = snn_wrapper.forward(x)
        assert out.shape == (2, 3)

    def test_forward_1d_input_unsqueezed(self, snn_wrapper):
        x = torch.rand(4).clamp(min=0.1)
        out = snn_wrapper.forward(x)
        assert out.shape == (1, 3)

    def test_forward_scores_non_negative(self, snn_wrapper):
        x = torch.rand(5, 4).clamp(min=0.1)
        out = snn_wrapper.forward(x)
        assert torch.all(out >= 0.0)

    def test_forward_returns_float_tensor(self, snn_wrapper):
        x = torch.rand(1, 4)
        out = snn_wrapper.forward(x)
        assert out.dtype == torch.float32

    def test_forward_no_gradient(self, snn_wrapper):
        x = torch.rand(1, 4, requires_grad=False)
        out = snn_wrapper.forward(x)
        assert not out.requires_grad


# ---------------------------------------------------------------------------
# reach() — return type and structure
# ---------------------------------------------------------------------------

class TestSpikingNeuralNetworkReachStructure:

    def test_reach_returns_list(self, snn_wrapper, tiny_box):
        result = snn_wrapper.reach(tiny_box, method='approx')
        assert isinstance(result, list)

    def test_reach_returns_single_box(self, snn_wrapper, tiny_box):
        result = snn_wrapper.reach(tiny_box, method='approx')
        assert len(result) == 1
        assert isinstance(result[0], Box)

    def test_reach_output_shape(self, snn_wrapper, tiny_box):
        out_box = snn_wrapper.reach(tiny_box, method='approx')[0]
        # num_classes == 3, shape (3, 1)
        assert out_box.lb.shape == (3, 1)
        assert out_box.ub.shape == (3, 1)

    def test_reach_lb_leq_ub(self, snn_wrapper, tiny_box):
        out_box = snn_wrapper.reach(tiny_box, method='approx')[0]
        assert np.all(out_box.lb <= out_box.ub)

    def test_reach_scores_non_negative(self, snn_wrapper, tiny_box):
        out_box = snn_wrapper.reach(tiny_box, method='approx')[0]
        assert np.all(out_box.lb >= 0.0)


# ---------------------------------------------------------------------------
# reach() — input set types
# ---------------------------------------------------------------------------

class TestSpikingNeuralNetworkReachInputTypes:

    def test_reach_with_box(self, snn_wrapper, tiny_box):
        result = snn_wrapper.reach(tiny_box, method='approx')
        assert len(result) == 1 and isinstance(result[0], Box)

    def test_reach_with_star(self, snn_wrapper, tiny_star):
        result = snn_wrapper.reach(tiny_star, method='approx')
        assert len(result) == 1 and isinstance(result[0], Box)

    def test_reach_with_star_exact(self, snn_wrapper, partial_star):
        # partial_star has only 2 symbolic dims — exercises Star.get_ranges() → LP split path
        result = snn_wrapper.reach(partial_star, method='exact')
        assert len(result) == 1 and isinstance(result[0], Box)

    def test_reach_unsupported_type_raises(self, snn_wrapper):
        from n2v.sets import Zono
        c = np.array([[0.5], [0.5], [0.5], [0.5]])
        V = np.eye(4) * 0.1
        z = Zono(c, V)
        with pytest.raises(TypeError):
            snn_wrapper.reach(z)


# ---------------------------------------------------------------------------
# reach() — method dispatch
# ---------------------------------------------------------------------------

class TestSpikingNeuralNetworkReachMethods:

    def test_approx_method(self, snn_wrapper, tiny_box):
        result = snn_wrapper.reach(tiny_box, method='approx')
        assert isinstance(result[0], Box)

    def test_exact_method(self, snn_wrapper, partial_box):
        # partial_box has only 2 symbolic dims → manageable for 'exact'
        result = snn_wrapper.reach(partial_box, method='exact')
        assert isinstance(result[0], Box)

    def test_exact_tighter_than_approx(self, snn_wrapper, partial_box):
        approx_box = snn_wrapper.reach(partial_box, method='approx')[0]
        exact_box  = snn_wrapper.reach(partial_box, method='exact')[0]
        # Exact bounds must be inside (or equal to) approx bounds
        assert np.all(exact_box.lb >= approx_box.lb - 1e-4)
        assert np.all(exact_box.ub <= approx_box.ub + 1e-4)

    def test_invalid_method_raises(self, snn_wrapper, tiny_box):
        with pytest.raises(ValueError):
            snn_wrapper.reach(tiny_box, method='nonsense')


# ---------------------------------------------------------------------------
# reach() — config handling
# ---------------------------------------------------------------------------

class TestSpikingNeuralNetworkReachConfig:

    def test_config_object_accepted(self, snn_wrapper, tiny_box):
        cfg = SNNReachConfig(method='approx')
        result = snn_wrapper.reach(tiny_box, config=cfg)
        assert isinstance(result[0], Box)

    def test_config_method_takes_priority(self, snn_wrapper, partial_box):
        # config.method overrides the positional method= parameter
        cfg = SNNReachConfig(method='exact')
        result = snn_wrapper.reach(partial_box, config=cfg)
        assert isinstance(result[0], Box)

    def test_config_and_kwargs_raises(self, snn_wrapper, tiny_box):
        cfg = SNNReachConfig(method='approx')
        with pytest.raises(TypeError):
            snn_wrapper.reach(tiny_box, config=cfg, tight_bounds=True)

    def test_kwargs_style(self, snn_wrapper, tiny_box):
        result = snn_wrapper.reach(tiny_box, method='approx', singleton_bounds=False)
        assert isinstance(result[0], Box)

    def test_label_kwarg_accepted(self, snn_wrapper, tiny_box):
        result = snn_wrapper.reach(tiny_box, method='approx', label=0)
        assert isinstance(result[0], Box)


# ---------------------------------------------------------------------------
# reach() — option flags
# ---------------------------------------------------------------------------

class TestSpikingNeuralNetworkReachOptions:

    def test_singleton_bounds_accepted(self, snn_wrapper, tiny_box):
        result = snn_wrapper.reach(tiny_box, method='approx', singleton_bounds=True)
        assert len(result) == 1 and isinstance(result[0], Box)

    def test_singleton_bounds_sound(self, snn_wrapper, tiny_box):
        # Enabling singleton_bounds must not produce bounds that exclude valid outputs.
        out_box = snn_wrapper.reach(tiny_box, method='approx', singleton_bounds=True)[0]
        assert np.all(out_box.lb <= out_box.ub)
        assert np.all(out_box.lb >= 0.0)


# ---------------------------------------------------------------------------
# reach() — dimension mismatch
# ---------------------------------------------------------------------------

class TestSpikingNeuralNetworkReachDimCheck:

    def test_wrong_dimension_raises(self, tiny_model):
        snn = SpikingNeuralNetwork(tiny_model, input_size=4)
        wrong_box = Box(
            np.array([[0.0], [0.0]]),
            np.array([[1.0], [1.0]]),
        )
        with pytest.raises(ValueError, match="dimension"):
            snn.reach(wrong_box)
