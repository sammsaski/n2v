"""
Shared fixtures for n2v SNN unit tests.

All fixtures use lazy imports so that importing this conftest does not
trigger snntorch. Individual test files call pytest.importorskip('snntorch')
at module level to skip gracefully when snntorch is absent.
"""

import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Tiny model parameters — small enough for LP tests to run in milliseconds
# ---------------------------------------------------------------------------
INPUT_SIZE = 4
HIDDEN_SIZES = [8]
NUM_CLASSES = 3
NUM_STEPS = 8


@pytest.fixture(scope="module")
def tiny_model():
    """A minimal F2FMLP: 4 inputs → 8 hidden → 3 classes, T=8 steps."""
    from n2v.snn.model import F2FMLP
    torch.manual_seed(0)
    model = F2FMLP(
        input_size=INPUT_SIZE,
        hidden_sizes=HIDDEN_SIZES,
        num_classes=NUM_CLASSES,
        num_steps=NUM_STEPS,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def snn_wrapper(tiny_model):
    """SpikingNeuralNetwork wrapping the tiny test model."""
    from n2v.nn.spiking_neural_network import SpikingNeuralNetwork
    return SpikingNeuralNetwork(tiny_model)


@pytest.fixture
def tiny_box():
    """4-D Box with all dimensions symbolic."""
    from n2v.sets import Box
    lb = np.array([[0.2], [0.3], [0.1], [0.4]])
    ub = np.array([[0.5], [0.7], [0.6], [0.9]])
    return Box(lb, ub)


@pytest.fixture
def tiny_star():
    """4-D Star (axis-aligned, equivalent to tiny_box)."""
    from n2v.sets import Star
    lb = np.array([[0.2], [0.3], [0.1], [0.4]])
    ub = np.array([[0.5], [0.7], [0.6], [0.9]])
    return Star.from_bounds(lb, ub)


@pytest.fixture
def partial_box():
    """4-D Box where only 2 dimensions are symbolic (faster for 'exact' tests)."""
    from n2v.sets import Box
    lb = np.array([[0.2], [0.6], [0.6], [0.6]])
    ub = np.array([[0.8], [0.9], [0.6], [0.6]])
    return Box(lb, ub)
