"""Fixtures for profiler instrumentation tests."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn import NeuralNetwork
from n2v.sets import Star
from n2v.sets import ImageStar
from n2v.sets import Zono, Hexatope, Octatope


@pytest.fixture
def tiny_split_net():
    """``Linear(2->3) + ReLU`` designed so that over the input box ``[-1,1]^2``:

    - neuron 0: ``x0 + 2 in [1,3] > 0``   -> stable active
    - neuron 1: ``x0 - 2 in [-3,-1] < 0`` -> stable inactive
    - neuron 2: ``x0 in [-1,1]``          -> unstable (crosses 0)

    so the classification counters are hand-checkable.
    """
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        )
        model[0].bias.copy_(torch.tensor([2.0, -2.0, 0.0]))
    net = NeuralNetwork(model)
    X = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    return net, X


@pytest.fixture
def lp_oracle(monkeypatch):
    """Independent ground-truth counter of LP problems solved, intercepted where
    ``star.py`` looks up ``solve_lp_batch``. Compared against the profiler's
    ``n_lp_solves`` so LP counts are verified, not hand-guessed."""
    import n2v.sets.star as star_mod

    counter = {"n": 0}
    orig = star_mod.solve_lp_batch

    def wrapped(objectives, *args, **kwargs):
        counter["n"] += len(objectives)
        return orig(objectives, *args, **kwargs)

    monkeypatch.setattr(star_mod, "solve_lp_batch", wrapped)
    return counter


def _three_neuron_first_layer(activation):
    """Linear(2->3) + ``activation`` with the tiny_split_net weights."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), activation)
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        )
        model[0].bias.copy_(torch.tensor([2.0, -2.0, 0.0]))
    net = NeuralNetwork(model)
    X = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    return net, X


@pytest.fixture
def leaky_net():
    """Same classification structure as tiny_split_net, with LeakyReLU."""
    return _three_neuron_first_layer(nn.LeakyReLU())


@pytest.fixture
def sigmoid_net():
    """Linear(2->3) + Sigmoid; all three pre-activations vary over the box."""
    return _three_neuron_first_layer(nn.Sigmoid())


@pytest.fixture
def multi_layer_net():
    """Linear -> ReLU -> Linear -> ReLU (two of each layer type)."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3), nn.ReLU()
    )
    net = NeuralNetwork(model)
    X = Star.from_bounds(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
    return net, X


def _split_net_model():
    """The tiny_split_net Linear(2->3) model (no activation attached)."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU())
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        )
        model[0].bias.copy_(torch.tensor([2.0, -2.0, 0.0]))
    return NeuralNetwork(model)


@pytest.fixture
def zono_split_net():
    """tiny_split_net with a Zono input (LP-free over-approx ReLU)."""
    net = _split_net_model()
    X = Zono.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    return net, X


@pytest.fixture
def hexa_split_net():
    """tiny_split_net with a Hexatope input (splits crossing neurons; MCF)."""
    net = _split_net_model()
    X = Hexatope.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    return net, X


@pytest.fixture
def octa_split_net():
    """tiny_split_net with an Octatope input (splits crossing neurons; MCF)."""
    net = _split_net_model()
    X = Octatope.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    return net, X


@pytest.fixture
def cnn_net():
    """Conv2d -> ReLU -> MaxPool2d -> AvgPool2d over an 8x8x1 ImageStar.

    Exercises the conv/pool operation regions and MaxPool split counting.
    A small input box keeps the exact MaxPool split count modest.
    """
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 2, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AvgPool2d(2),
    )
    net = NeuralNetwork(model)
    H = W = 8
    C = 1
    lb = -0.1 * np.ones((H, W, C))
    ub = 0.1 * np.ones((H, W, C))
    X = ImageStar.from_bounds(lb, ub, H, W, C)
    return net, X
