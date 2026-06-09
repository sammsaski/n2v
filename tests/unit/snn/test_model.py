"""
Unit tests for F2FMLP: construction, forward pass, and simulation.
"""

import pytest
pytest.importorskip("snntorch", reason="snntorch not installed; pip install n2v[snn]")

import numpy as np
import torch
from n2v.snn.model import F2FMLP


class TestF2FMLPConstruction:

    def test_default_construction(self):
        model = F2FMLP()
        assert model.num_steps == 10
        assert model.num_classes == 10
        assert model.hidden_sizes == [64]
        assert len(model.fcs) == 2   # one hidden + one output
        assert len(model.lifs) == 2

    def test_custom_construction(self, tiny_model):
        assert tiny_model.num_steps == 8
        assert tiny_model.num_classes == 3
        assert tiny_model.hidden_sizes == [8]
        assert len(tiny_model.fcs) == 2  # Linear(4→8) + Linear(8→3)

    def test_multiple_hidden_layers(self):
        model = F2FMLP(input_size=4, hidden_sizes=[16, 8], num_classes=3, num_steps=6)
        assert len(model.fcs) == 3   # 4→16, 16→8, 8→3
        assert len(model.lifs) == 3
        assert model.fcs[0].in_features == 4
        assert model.fcs[0].out_features == 16
        assert model.fcs[1].in_features == 16
        assert model.fcs[1].out_features == 8
        assert model.fcs[2].out_features == 3

    def test_layer_sizes(self, tiny_model):
        assert tiny_model.fcs[0].in_features == 4
        assert tiny_model.fcs[0].out_features == 8
        assert tiny_model.fcs[1].in_features == 8
        assert tiny_model.fcs[1].out_features == 3

    def test_is_nn_module(self, tiny_model):
        import torch.nn as nn
        assert isinstance(tiny_model, nn.Module)


class TestF2FMLPForward:

    def test_output_shape_single_sample(self, tiny_model):
        # spike_train shape: (B, D, T)
        spike_train = torch.zeros(1, 4, 8)
        scores = tiny_model(spike_train)
        assert scores.shape == (1, 3)

    def test_output_shape_batch(self, tiny_model):
        spike_train = torch.zeros(5, 4, 8)
        scores = tiny_model(spike_train)
        assert scores.shape == (5, 3)

    def test_scores_non_negative(self, tiny_model):
        # F2F scores are weighted sums of 0/1 spikes with positive weights,
        # so scores >= 0 always holds.
        torch.manual_seed(1)
        spike_train = (torch.rand(8, 4, 8) > 0.5).float()
        scores = tiny_model(spike_train)
        assert torch.all(scores >= 0.0), "F2F scores must be non-negative"

    def test_scores_deterministic(self, tiny_model):
        spike_train = torch.zeros(1, 4, 8)
        spike_train[0, 0, 0] = 1.0  # dim 0 fires at t=0
        s1 = tiny_model(spike_train)
        s2 = tiny_model(spike_train)
        assert torch.allclose(s1, s2)

    def test_no_spike_gives_zero_scores(self):
        # With zero biases, all-zero spike train produces zero membrane potential
        # at every timestep → no neuron ever fires → scores must be exactly zero.
        # (Cannot use tiny_model here: its biases can charge the membrane over T steps
        # even on zero input, causing spontaneous firing and non-zero scores.)
        import torch.nn as nn
        model = F2FMLP(input_size=4, hidden_sizes=[8], num_classes=3, num_steps=8)
        for fc in model.fcs:
            nn.init.zeros_(fc.bias)
        model.eval()
        spike_train = torch.zeros(1, 4, 8)
        scores = model(spike_train)
        assert torch.allclose(scores, torch.zeros(1, 3))

    def test_returns_float_tensor(self, tiny_model):
        spike_train = torch.zeros(1, 4, 8)
        scores = tiny_model(spike_train)
        assert scores.dtype == torch.float32


class TestF2FMLPSimulate:

    def test_simulate_output_shapes(self, tiny_model):
        spike_train = torch.zeros(4, 8)  # (D, T) single sample
        scores, hidden, output = tiny_model.simulate_with_patterns(spike_train)
        assert scores.shape == (3,)               # (num_classes,)
        assert hidden.shape == (8, 8)             # (num_steps, hidden_neurons)
        assert output.shape == (8, 3)             # (num_steps, num_classes)

    def test_simulate_binary_spikes(self, tiny_model):
        spike_train = torch.zeros(4, 8)
        _, hidden, output = tiny_model.simulate_with_patterns(spike_train)
        assert set(np.unique(hidden)).issubset({0, 1})
        assert set(np.unique(output)).issubset({0, 1})

    def test_simulate_scores_match_forward(self, tiny_model):
        torch.manual_seed(7)
        # Build a valid spike train: at most one 1 per input dim
        spike_train_2d = torch.zeros(4, 8)
        for d in range(4):
            t = d % 8
            spike_train_2d[d, t] = 1.0

        spike_train_3d = spike_train_2d.unsqueeze(0)  # (1, D, T)
        forward_scores = tiny_model(spike_train_3d).detach().numpy()[0]
        sim_scores, _, _ = tiny_model.simulate_with_patterns(spike_train_2d)

        np.testing.assert_allclose(forward_scores, sim_scores, atol=1e-5)

    def test_simulate_accepts_batch_dim(self, tiny_model):
        spike_train = torch.zeros(1, 4, 8)  # (B=1, D, T) with batch dim
        scores, hidden, output = tiny_model.simulate_with_patterns(spike_train)
        assert scores.shape == (3,)
