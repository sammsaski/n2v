"""
Unit tests for F2F latency encoding functions.
"""

import pytest
pytest.importorskip("snntorch", reason="snntorch not installed; pip install n2v[snn]")

import numpy as np
import torch
from n2v.snn.encoding import latency_from_values, encode_batch, spike_train_from_latencies


class TestLatencyFromValues:

    def test_maximum_value_fires_first(self):
        # x = 1.0 → latency = floor((T-1)*(1-1) + 0.5) = floor(0.5) = 0
        v = torch.tensor([[1.0]])
        lat = latency_from_values(v, num_steps=8)
        assert lat[0, 0].item() == 0

    def test_zero_value_is_silent(self):
        # x = 0.0 → sentinel = T (silent, never fires)
        v = torch.tensor([[0.0]])
        lat = latency_from_values(v, num_steps=8)
        assert lat[0, 0].item() == 8  # sentinel = num_steps

    def test_negative_value_is_silent(self):
        v = torch.tensor([[-0.5]])
        lat = latency_from_values(v, num_steps=8)
        assert lat[0, 0].item() == 8

    def test_midpoint_fires_in_middle(self):
        # x = 0.5 → floor(7*0.5 + 0.5) = floor(4.0) = 4
        v = torch.tensor([[0.5]])
        lat = latency_from_values(v, num_steps=8)
        assert lat[0, 0].item() == 4

    def test_monotone_larger_fires_earlier(self):
        # Larger values should produce smaller (earlier) latencies
        values = torch.tensor([[0.9, 0.7, 0.5, 0.3, 0.1]])
        lat = latency_from_values(values, num_steps=16)
        lats = lat[0].tolist()
        assert lats == sorted(lats), f"Latencies not monotone: {lats}"

    def test_all_silent_input(self):
        v = torch.zeros(1, 4)
        lat = latency_from_values(v, num_steps=8)
        assert torch.all(lat == 8)

    def test_output_dtype_is_long(self):
        v = torch.tensor([[0.5]])
        lat = latency_from_values(v, num_steps=8)
        assert lat.dtype == torch.long

    def test_batch_shape_preserved(self):
        v = torch.rand(3, 5)  # batch of 3, 5 dims
        lat = latency_from_values(v, num_steps=10)
        assert lat.shape == (3, 5)


class TestEncodeBatch:

    def test_output_shape(self):
        x = torch.rand(2, 4)
        spikes = encode_batch(x, num_steps=8)
        assert spikes.shape == (2, 4, 8)

    def test_binary_values(self):
        x = torch.rand(4, 6)
        spikes = encode_batch(x, num_steps=8)
        unique = torch.unique(spikes)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_non_silent_pixel_fires_exactly_once(self):
        # Each pixel with value > 0 should have exactly one timestep set to 1
        x = torch.rand(3, 5).clamp(min=0.1)  # all > 0, so non-silent
        spikes = encode_batch(x, num_steps=10)
        # sum over time axis should be 1 for every non-silent pixel
        fire_counts = spikes.sum(dim=2)  # (B, D)
        assert torch.all(fire_counts == 1.0), "Non-silent pixels must fire exactly once"

    def test_silent_pixel_never_fires(self):
        # Zero-valued pixels should never fire
        x = torch.zeros(2, 4)
        spikes = encode_batch(x, num_steps=8)
        assert torch.all(spikes == 0.0)

    def test_high_value_fires_early(self):
        # x = 1.0 → latency = 0 → fires at t=0
        x = torch.ones(1, 1)
        spikes = encode_batch(x, num_steps=8)  # (1, 1, 8)
        assert spikes[0, 0, 0].item() == 1.0
        assert spikes[0, 0, 1:].sum().item() == 0.0

    def test_spike_timestep_matches_latency(self):
        # For a known value, verify the spike fires at the correct timestep
        x = torch.tensor([[0.5]])  # latency = 4 for T=8
        spikes = encode_batch(x, num_steps=8)  # (1, 1, 8)
        assert spikes[0, 0, 4].item() == 1.0
        assert spikes[0, 0, :4].sum().item() == 0.0
        assert spikes[0, 0, 5:].sum().item() == 0.0


class TestSpikeTrainFromLatencies:

    def test_output_shape(self):
        latencies = np.array([0, 3, 7])
        spikes = spike_train_from_latencies(latencies, num_steps=8)
        assert spikes.shape == (3, 8)

    def test_binary_values(self):
        latencies = np.array([1, 4, 6])
        spikes = spike_train_from_latencies(latencies, num_steps=8)
        unique = torch.unique(spikes)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_each_dim_fires_at_correct_time(self):
        latencies = np.array([0, 2, 5])
        spikes = spike_train_from_latencies(latencies, num_steps=8)
        assert spikes[0, 0].item() == 1.0  # fires at t=0
        assert spikes[1, 2].item() == 1.0  # fires at t=2
        assert spikes[2, 5].item() == 1.0  # fires at t=5
        # All other entries should be 0
        assert spikes.sum().item() == 3.0

    def test_silent_sentinel_never_fires(self):
        latencies = np.array([8, 8])  # sentinel = num_steps = 8
        spikes = spike_train_from_latencies(latencies, num_steps=8)
        assert spikes.sum().item() == 0.0

    def test_returns_torch_tensor(self):
        latencies = np.array([1, 3])
        spikes = spike_train_from_latencies(latencies, num_steps=6)
        assert isinstance(spikes, torch.Tensor)

    def test_consistent_with_encode_batch(self):
        # Verify encode_batch and spike_train_from_latencies agree for a known input
        x = torch.tensor([[0.5, 1.0]])  # latencies: 4, 0 (for T=8)
        spikes_batch = encode_batch(x, num_steps=8)  # (1, 2, 8)

        latencies = np.array([4, 0])
        spikes_single = spike_train_from_latencies(latencies, num_steps=8)  # (2, 8)

        np.testing.assert_array_equal(
            spikes_batch[0].numpy(),  # (2, 8)
            spikes_single.numpy(),    # (2, 8)
        )
