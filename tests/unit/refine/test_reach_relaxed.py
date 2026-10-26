"""Unit tests for the metadata-emitting fixed-phase relaxed reach."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.star import Star
from n2v.refine import Status, verify_refine
from n2v.refine.reach_relaxed import (
    ConvLayer,
    LinearLayer,
    ReluLayer,
    extract_layers,
    relaxed_reach,
)
from n2v.refine.types import LinearSpec, NeuronKey, Phase
from tests.unit.refine.conftest import _assert_honest_ce
from tests.unit.refine.conftest import seq_net as _seq


CPU_CONFIG = dict(bound_mode="box", prune_mode="exact", witness_mode="exact",
                  frontier_width=1)


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


# --------------------------------------------------------------------------- #
# extract_layers accepts only models whose forward() is the child chain       #
# --------------------------------------------------------------------------- #
class _Functional(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc2(x + torch.relu(self.fc1(x)))


class _ScaledSequential(nn.Sequential):
    def forward(self, x):
        return 2 * super().forward(x)


class _ScaledLinear(nn.Linear):
    def forward(self, x):
        return 2 * super().forward(x)


def _chain():
    return nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))


@pytest.mark.parametrize("make, match", [
    (_Functional, "nn.Sequential"),
    (_Residual, "nn.Sequential"),
    (lambda: _ScaledSequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)),
     "nn.Sequential"),
    (lambda: nn.Sequential(_ScaledLinear(2, 3), nn.ReLU(), nn.Linear(3, 1)),
     "overrides forward"),
], ids=["functional", "residual", "sequential-subclass", "leaf-subclass"])
def test_extract_layers_rejects_a_custom_forward(make, match):
    """The layer walk reads .children(); a custom forward() (functional ops,
    skip connections, rescaling) computes something the walk cannot see."""
    with pytest.raises(TypeError, match=match):
        extract_layers(make())


def test_extract_layers_flattens_nested_sequential():
    m = nn.Sequential(nn.Sequential(nn.Linear(2, 3), nn.ReLU()), nn.Linear(3, 1))
    assert [type(l) for l in extract_layers(m)] == [LinearLayer, ReluLayer,
                                                     LinearLayer]


def test_verify_refine_rejects_a_functional_model():
    torch.manual_seed(0)
    net = _Functional().double()
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    with pytest.raises(TypeError, match="nn.Sequential"):
        verify_refine(S_in, net, LinearSpec([[1.0]], [0.0]), **CPU_CONFIG)


def test_extract_layers_rejects_forward_hooks():
    m = _chain()
    handle = m[0].register_forward_hook(lambda mod, i, o: 2 * o)
    try:
        with pytest.raises(TypeError, match="forward hook"):
            extract_layers(m)
    finally:
        handle.remove()
    extract_layers(m)                     # accepted once the hook is gone


def test_extract_layers_rejects_an_instance_level_forward():
    m = _chain()
    m[0].forward = lambda x: 2 * nn.Linear.forward(m[0], x)
    with pytest.raises(TypeError):
        extract_layers(m)


def test_extract_layers_rejects_global_forward_hooks():
    handle = torch.nn.modules.module.register_module_forward_hook(
        lambda mod, i, o: None)
    try:
        with pytest.raises(TypeError, match="global"):
            extract_layers(_chain())
    finally:
        handle.remove()                   # leave the process-wide registry clean


def test_extract_layers_rejects_non_default_flatten_and_unflatten_dims():
    with pytest.raises(TypeError):
        extract_layers(nn.Sequential(nn.Flatten(0), nn.Linear(2, 1)))
    with pytest.raises(TypeError):
        extract_layers(nn.Sequential(nn.Linear(2, 4), nn.Unflatten(2, (2, 2)),
                                     nn.Flatten(), nn.Linear(4, 1)))


def test_extract_layers_rejects_batchnorm_without_running_stats():
    """Such a BatchNorm normalizes with the batch's own statistics even in
    eval mode, so it is not an affine map."""
    m = nn.Sequential(nn.Linear(2, 4), nn.Unflatten(1, (1, 2, 2)),
                      nn.BatchNorm2d(1, track_running_stats=False),
                      nn.Flatten(), nn.ReLU(), nn.Linear(4, 1))
    with pytest.raises(TypeError, match="track_running_stats"):
        extract_layers(m)


# --------------------------------------------------------------------------- #
# Conv layers                                                                  #
# --------------------------------------------------------------------------- #
def test_extract_layers_leaves_the_callers_conv_module_untouched():
    """The ConvLayer is built from a float64 eval-mode copy: the caller's
    float32 training-mode model keeps its dtype and mode and still runs."""
    m = nn.Sequential(nn.Linear(4, 16), nn.Unflatten(1, (1, 4, 4)),
                      nn.Conv2d(1, 1, 3), nn.Flatten(), nn.ReLU(),
                      nn.Linear(4, 1))
    m.train()
    layers = extract_layers(m)
    assert m[2].weight.dtype == torch.float32
    assert m[2].training is True
    m(torch.randn(3, 4))
    assert any(isinstance(l, ConvLayer) for l in layers)


def test_conv_layer_reach_is_sound_and_decides():
    """A flat-input Sequential that reshapes with nn.Unflatten before its
    conv: the relaxed reach contains sampled outputs and verify_refine decides
    both a far-UNSAT and a far-SAT spec."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(3, 8), nn.Unflatten(1, (2, 2, 2)),
                      nn.Conv2d(2, 2, 2), nn.Flatten(), nn.ReLU(),
                      nn.Linear(2, 2)).double().eval()
    layers = extract_layers(m)
    assert any(isinstance(l, ConvLayer) for l in layers)
    lb, ub = -np.ones(3), np.ones(3)
    S_in = Star.from_bounds(lb, ub)
    out, _ = relaxed_reach(S_in, layers)
    X = np.random.default_rng(0).uniform(lb, ub, size=(300, 3))
    inside = out.contains(_forward(m, X), method="lp")
    assert np.all(inside), f"{(~inside).sum()}/300 exact outputs outside"

    lo, hi = out.get_ranges()
    unsat = LinearSpec([[1.0, 0.0]], [float(lo[0, 0]) - 10.0])
    sat = LinearSpec([[1.0, 0.0]], [float(hi[0, 0]) + 10.0])
    assert verify_refine(S_in, m, unsat, **CPU_CONFIG).status == Status.UNSAT
    res = verify_refine(S_in, m, sat, **CPU_CONFIG)
    assert res.status == Status.SAT
    _assert_honest_ce(res, m, sat, lb, ub)


def test_conv_net_behind_a_flat_input_wrapper():
    """Supported conv usage: a flat-input wrapper as ``model`` with
    ``layers=extract_layers(inner, in_shape=(C, H, W))``. The wrapper itself
    is rejected by extract_layers (its forward() reshapes)."""
    torch.manual_seed(0)
    inner = nn.Sequential(nn.Conv2d(1, 2, 2), nn.Flatten(), nn.ReLU(),
                          nn.Linear(8, 1)).double().eval()

    class Flat(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x.reshape(x.shape[0], 1, 3, 3))

    flat = Flat(inner)
    with pytest.raises(TypeError):
        extract_layers(flat)
    layers = extract_layers(inner, in_shape=(1, 3, 3))
    lb, ub = -np.ones(9), np.ones(9)
    S_in = Star.from_bounds(lb, ub)

    # x = 0 is a counterexample on the closed boundary of the unsafe region
    with torch.no_grad():
        tau = float(flat(torch.zeros((1, 9), dtype=torch.float64)))
    sat = LinearSpec([[1.0]], [tau])
    res = verify_refine(S_in, flat, sat, layers=layers, **CPU_CONFIG)
    assert res.status == Status.SAT
    _assert_honest_ce(res, flat, sat, lb, ub)

    out, _ = relaxed_reach(S_in, layers)
    lo, hi = (float(v) for v in out.get_range(0))
    unsat = LinearSpec([[1.0]], [lo - 0.1 * (hi - lo)])
    res = verify_refine(S_in, flat, unsat, layers=layers, **CPU_CONFIG)
    assert res.status == Status.UNSAT


@pytest.mark.parametrize("kind", ["conv", "batchnorm"])
def test_conv_layer_generators_skip_the_bias(kind):
    """Generator columns go through a bias-free copy of the module, so a
    generator far below ulp(|bias|) survives exactly (adding the bias and
    subtracting it again would round it away); the anchor keeps the bias."""
    if kind == "conv":
        m = nn.Conv2d(1, 1, 1).double().eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.fill_(1.0)
    else:
        m = nn.BatchNorm2d(1).double().eval()
        with torch.no_grad():
            m.running_mean.fill_(-1.0)
            m.running_var.fill_(1.0 - m.eps)
            m.weight.fill_(1.0)
            m.bias.zero_()
    layer = ConvLayer(m, (1, 1, 1), (1, 1, 1))
    S = Star(np.array([[0.0, 1e-17, 3e-16]]), np.zeros((0, 2)),
             np.zeros((0, 1)), -np.ones(2), np.ones(2))
    V = layer.apply(S).V
    assert V[0, 0] == 1.0
    assert np.array_equal(V[0, 1:], np.array([1e-17, 3e-16]))
