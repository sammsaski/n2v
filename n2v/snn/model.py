"""
F2F (First-to-Fire) SNN model definition.

The F2FMLP model uses latency coding: brighter/larger input values fire earlier.
Each neuron fires at most once across all T timesteps (at-most-once constraint).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate


class F2FMLP(nn.Module):
    """Multi-hidden-layer F2F SNN for flattened input classification.

    Encoding: bright/large inputs fire at timestep 0 (earliest), dark/small inputs
    fire at timestep T-1 (latest), zero-valued inputs never fire (latency = T).

    Each neuron fires at most once across all T timesteps (the at-most-once
    property). The predicted class is the output neuron with the highest
    accumulated score, where earlier firing gives higher score.
    """

    def __init__(self, input_size: int = 784,
                 hidden_sizes: list[int] | tuple[int, ...] = (64,),
                 num_classes: int = 10, beta: float = 0.9,
                 threshold: float = 1.0, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.hidden_sizes = list(hidden_sizes)

        # Build one Linear + one LIF per layer (hidden layers + output layer).
        layer_sizes = [input_size] + list(hidden_sizes) + [num_classes]
        self.fcs = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])
        # One LIF per layer; all layers share the same beta and threshold.
        # fast_sigmoid is the surrogate gradient for training.
        self.lifs = nn.ModuleList([
            snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())
            for _ in range(len(layer_sizes) - 1)
        ])

    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Return F2F class scores from spike train shaped (B, input_size, T).

        At each timestep t, the current layer's input (x) is the spike output
        of the previous layer. The at-most-once constraint is enforced by the
        'fired' mask: once a neuron has spiked, its future spikes are zeroed.

        Score for class c = sum_t  (T - t) * output_spike[c, t]
        Earlier spikes contribute larger weights, so the first-to-fire class
        accumulates the highest score.
        """
        batch_size = spike_train.shape[0]
        n_layers = len(self.fcs)

        # Initialize membrane potentials for all layers.
        mems = [self.lifs[l].init_leaky() for l in range(n_layers)]

        # fired[l][i] tracks whether neuron i in layer l has already fired.
        # Shape: (batch_size, layer_width). Initialized to all-zeros (no neuron has fired).
        fired = [
            torch.zeros(batch_size, self.fcs[l].out_features, device=spike_train.device)
            for l in range(n_layers)
        ]

        # Accumulator for the F2F class scores.
        scores = torch.zeros(batch_size, self.num_classes, device=spike_train.device)

        for t in range(self.num_steps):
            # Input at timestep t: which input pixels fire at this timestep.
            x = spike_train[:, :, t]

            for l in range(n_layers):
                cur = self.fcs[l](x)                         # linear pre-activation
                spk_raw, mems[l] = self.lifs[l](cur, mems[l])  # LIF step: membrane update + spike

                # Enforce at-most-once: zero out spikes from neurons that already fired.
                spk = spk_raw * (1.0 - fired[l])

                # Record newly fired neurons so they cannot fire again.
                fired[l] = torch.clamp(fired[l] + spk, 0.0, 1.0)

                x = spk  # Spikes from this layer become input to the next layer.

            # F2F scoring: output spikes at timestep t contribute (T - t) to their class score.
            # t=0 contributes T (maximum), t=T-1 contributes 1 (minimum).
            scores = scores + float(self.num_steps - t) * x

        return scores

    @torch.no_grad()
    def simulate_with_patterns(self, spike_train: torch.Tensor):
        """Return (scores, hidden_spikes, output_spikes) for one sample.

        Used by the exhaustive fallback verifier, which needs the exact scores
        for every spike-timing combination, and by Monte Carlo sampling.

        hidden_spikes: uint8 array of shape (T, total_hidden_neurons)
        output_spikes: uint8 array of shape (T, num_classes)
        """
        if spike_train.ndim == 2:
            # Add batch dimension if a single sample was passed.
            spike_train = spike_train.unsqueeze(0)
        batch_size = spike_train.shape[0]
        n_layers = len(self.fcs)
        mems = [self.lifs[l].init_leaky() for l in range(n_layers)]
        fired = [
            torch.zeros(batch_size, self.fcs[l].out_features, device=spike_train.device)
            for l in range(n_layers)
        ]
        scores = torch.zeros(batch_size, self.num_classes, device=spike_train.device)
        layer_spikes = [[] for _ in range(n_layers)]  # collect per-timestep spike arrays

        for t in range(self.num_steps):
            x = spike_train[:, :, t]
            for l in range(n_layers):
                cur = self.fcs[l](x)
                spk_raw, mems[l] = self.lifs[l](cur, mems[l])
                spk = spk_raw * (1.0 - fired[l])
                fired[l] = torch.clamp(fired[l] + spk, 0.0, 1.0)
                x = spk
                # Save the spike pattern at this timestep for this layer.
                layer_spikes[l].append(spk.detach().cpu().numpy()[0])
            scores = scores + float(self.num_steps - t) * x

        # Stack to (T, neurons_in_layer) for each layer, cast to uint8 (0 or 1).
        stacked = [np.stack(layer_spikes[l], axis=0).astype(np.uint8) for l in range(n_layers)]

        # Concatenate all hidden layer spikes along the neuron axis.
        if len(stacked) > 1:
            hidden = np.concatenate(stacked[:-1], axis=1)  # all but last (output) layer
        else:
            hidden = np.zeros((self.num_steps, 0), dtype=np.uint8)
        output = stacked[-1]  # output layer spikes only
        return scores.detach().cpu().numpy()[0], hidden, output
