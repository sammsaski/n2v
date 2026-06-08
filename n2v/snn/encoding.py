"""
Latency encoding utilities for the F2F SNN.

Converts continuous input values to spike trains using latency coding:
larger values fire earlier (smaller timestep), smaller values fire later.
Zero values are silent (never fire).
"""

from __future__ import annotations

import numpy as np
import torch


def latency_from_values(values: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Encode normalized input values in [0, 1] to 0-based latency integers.

    Encoding formula: latency = round((T-1) * (1 - x))
      - x = 1.0  → latency = 0  (fires at timestep 0, the earliest)
      - x = 0.5  → latency ≈ (T-1)/2 (fires in the middle)
      - x = 0.0  → treated as background: latency = T (silent, never fires)
      - x ≤ 0    → silent sentinel value T

    Zero-valued inputs are background; assigning latency T means the spike
    loop (which runs t = 0..T-1) never emits a spike for them.
    """
    # Map brightness linearly to latency, round to nearest integer.
    z = torch.floor((num_steps - 1) * (1.0 - values) + 0.5).long()
    z = torch.clamp(z, 0, num_steps - 1)

    # Background pixels (value == 0) are assigned the silent sentinel T.
    silent = values <= 0
    z[silent] = num_steps
    return z


def encode_batch(images: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Convert a batch of images to a spike train tensor of shape (B, pixels, T).

    For each pixel, exactly one timestep is set to 1.0 (the latency timestep),
    or no timestep is set if the pixel is silent (background).
    """
    flat = images.view(images.shape[0], -1)       # flatten spatial dims: (B, pixels)
    lat = latency_from_values(flat, num_steps)     # (B, pixels) integer latencies

    # Build the spike train: one-hot along the time axis.
    spikes = torch.zeros(flat.shape[0], flat.shape[1], num_steps, device=flat.device)
    for t in range(num_steps):
        spikes[:, :, t] = (lat == t).float()      # 1 where this pixel fires at time t
    return spikes


def spike_train_from_latencies(latencies: np.ndarray, num_steps: int) -> torch.Tensor:
    """Build a single-sample spike train from a latency array.

    latencies[i] = t  → pixel i fires at timestep t
    latencies[i] = T  → pixel i is silent (no spike)

    Returns shape (num_pixels, num_steps).
    Used by the exhaustive fallback verifier and Monte Carlo sampling, where
    we need to simulate the model for one specific latency assignment.
    """
    spikes = np.zeros((latencies.shape[0], num_steps), dtype=np.float32)
    finite = latencies < num_steps          # True for non-silent pixels
    spikes[np.where(finite)[0], latencies[finite]] = 1.0
    return torch.from_numpy(spikes)
