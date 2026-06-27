"""Unit tests for the metadata-emitting fixed-phase relaxed reach (Phase 1)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.star import Star
from n2v.refine.reach_relaxed import (
    LinearLayer,
    ReluLayer,
    extract_layers,
    relaxed_reach,
)
from n2v.refine.types import NeuronKey, Phase


def _seq(*dims, seed=0):
    """Build a random FC ReLU net: Linear/ReLU/.../Linear."""
    torch.manual_seed(seed)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    net = nn.Sequential(*layers).double()
    return net


def _forward(model, X):
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float64)).cpu().numpy()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_relaxed_reach_is_sound_overapproximation(seed):
    """Every exact output sample must lie inside the relaxed output star."""
    rng = np.random.default_rng(seed)
    net = _seq(3, 6, 6, 2, seed=seed)
    layers = extract_layers(net)

    lb = rng.uniform(-1.0, 0.0, size=3)
    ub = rng.uniform(0.0, 1.0, size=3)
    S_in = Star.from_bounds(lb, ub)

    S_out, meta = relaxed_reach(S_in, layers)

    # Sample inputs, forward exactly, assert containment in the over-approx star.
    X = rng.uniform(lb, ub, size=(400, 3))
    Y = _forward(net, X)
    inside = S_out.contains(Y, method="lp")
    assert np.all(inside), f"{(~inside).sum()}/400 exact outputs outside the relaxed star"


def test_neuron_meta_is_recorded_correctly():
    """A hand-built net with a guaranteed-unstable neuron yields correct meta."""
    # 1 input -> 1 hidden (ReLU) -> 1 output.  hidden pre-activation = x, x in [-1, 1]
    # so the hidden neuron is unstable (l=-1<0<1=u).
    net = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[0].bias.copy_(torch.tensor([0.0], dtype=torch.float64))
        net[2].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[2].bias.copy_(torch.tensor([0.0], dtype=torch.float64))
    layers = extract_layers(net)

    S_in = Star.from_bounds(np.array([-1.0]), np.array([1.0]))
    S_out, meta = relaxed_reach(S_in, layers)

    assert len(meta) == 1
    nm = meta[0]
    assert nm.key == NeuronKey(0, 0)
    # pre-activation read of the hidden neuron is exactly the input alpha_0.
    assert nm.preact_center == pytest.approx(0.0)
    assert nm.preact_gens.shape == (1,)
    assert nm.preact_gens[0] == pytest.approx(1.0)
    assert nm.l < 0 < nm.u
    # fresh predicate column is appended after the single input variable.
    assert nm.pred_col == 1
    # output star: input var + 1 relaxed var = 2 predicate variables.
    assert S_out.nVar == 2


def test_fixed_phase_inactive_zeroes_neuron():
    """Forcing a neuron INACTIVE removes its contribution; ACTIVE keeps identity."""
    net = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[0].bias.copy_(torch.tensor([0.0], dtype=torch.float64))
        net[2].weight.copy_(torch.tensor([[1.0]], dtype=torch.float64))
        net[2].bias.copy_(torch.tensor([0.0], dtype=torch.float64))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0]), np.array([1.0]))
    key = NeuronKey(0, 0)

    # INACTIVE: output forced to 0 AND predicate restricted to x_hat <= 0 (x in
    # [-1, 0]); the output star is the point {0}.
    S_inact, meta_inact = relaxed_reach(S_in, layers, fixed={key: Phase.INACTIVE})
    assert meta_inact == []
    lo, hi = S_inact.get_ranges()
    assert lo[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert hi[0, 0] == pytest.approx(0.0, abs=1e-9)

    # ACTIVE: identity output, but the predicate is restricted to the active
    # subregion x_hat >= 0 (x in [0, 1]); so the exact output range is [0, 1],
    # NOT [-1, 1]. (Fixing a phase adds the phase constraint, like the exact split.)
    S_act, meta_act = relaxed_reach(S_in, layers, fixed={key: Phase.ACTIVE})
    assert meta_act == []
    lo, hi = S_act.get_ranges()
    assert lo[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert hi[0, 0] == pytest.approx(1.0, abs=1e-9)


def test_extract_layers_rejects_unsupported():
    net = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())
    with pytest.raises(TypeError):
        extract_layers(net)
