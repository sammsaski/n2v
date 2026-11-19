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


# --------------------------------------------------------------------------- #
# Refusals of extract_layers                                                    #
# --------------------------------------------------------------------------- #
def _nested_chain():
    return nn.Sequential(nn.Sequential(nn.Linear(2, 3), nn.ReLU()),
                         nn.Linear(3, 1))


def _hook_nested_leaf(m):
    return m[0][0].register_forward_hook(lambda mod, i, o: 2 * o)


def _hook_the_model(m):
    return m.register_forward_hook(lambda mod, i, o: 2 * o)


def _pre_hook_a_leaf(m):
    return m[1].register_forward_pre_hook(lambda mod, args: (2 * args[0],))


@pytest.mark.parametrize("attach", [_hook_nested_leaf, _hook_the_model,
                                    _pre_hook_a_leaf],
                         ids=["forward-hook-in-a-nested-sequential",
                              "forward-hook-on-the-model-itself",
                              "forward-pre-hook-on-a-leaf"])
def test_extract_layers_rejects_a_hook_anywhere_in_the_model(attach):
    """A forward hook or pre-hook changes what model(x) computes, wherever it
    sits: on a module inside a nested Sequential, on the model itself, or
    before a leaf. Catches a scan that reads only the top-level children or
    only forward hooks; the same model is accepted once the hook is gone."""
    m = _nested_chain()
    handle = attach(m)
    try:
        with pytest.raises(TypeError, match="hook"):
            extract_layers(m)
    finally:
        handle.remove()
    assert len(extract_layers(m)) == 3


def test_extract_layers_rejects_global_forward_pre_hooks():
    """A global pre-hook changes the input of every module's forward()."""
    handle = torch.nn.modules.module.register_module_forward_pre_hook(
        lambda mod, args: None)
    try:
        with pytest.raises(TypeError, match="global"):
            extract_layers(_chain())
    finally:
        handle.remove()                   # leave the process-wide registry clean


@pytest.mark.parametrize("make", [
    lambda: nn.Sequential(nn.Conv2d(1, 1, 1), nn.Flatten()),
    lambda: nn.Sequential(nn.Linear(4, 8), nn.Unflatten(1, (2, 2, 2)),
                          nn.Flatten(), nn.BatchNorm2d(2)),
], ids=["conv-first-without-in-shape", "batchnorm-after-a-flatten"])
def test_extract_layers_rejects_a_conv_type_module_without_an_image_shape(make):
    """A conv-type module needs a (C, H, W) shape from ``in_shape`` or a
    preceding nn.Unflatten: none at all, or a flat one after nn.Flatten.
    Catches a Flatten that leaves the tracked image shape in place (the
    BatchNorm2d after it is then accepted)."""
    with pytest.raises(TypeError, match=r"\(C, H, W\)"):
        extract_layers(make())


@pytest.mark.parametrize("make", [
    lambda: nn.Sequential(nn.Linear(2, 8), nn.Unflatten(1, (2, 2, 2)),
                          nn.Flatten(1, 2), nn.Linear(2, 1)),
    lambda: nn.Sequential(nn.Linear(2, 8), nn.Unflatten(1, (2, 2, 2)),
                          nn.Conv2d(2, 2, 1), nn.Unflatten(1, (1, 2)),
                          nn.Flatten(), nn.Linear(8, 1)),
], ids=["flatten-1-2", "unflatten-of-an-image"])
def test_extract_layers_rejects_other_flatten_and_unflatten_dims(make):
    """Only the default Flatten and an Unflatten of a flat activation are the
    identity on the star's flat basis: Flatten(1, 2) keeps an image axis, so
    the Linear after it is applied along that axis by torch and to the whole
    flat vector by the layer list. Catches a Flatten check that reads only
    start_dim (Flatten(0) and Unflatten(2) are the cases of
    test_extract_layers_rejects_non_default_flatten_and_unflatten_dims)."""
    with pytest.raises(TypeError):
        extract_layers(make())


def test_extract_layers_keeps_the_float64_parameters_bit_for_bit():
    """The layer list holds the model's own float64 weights and biases, not
    a rounded copy (0.1 and 1/3 are not float32 numbers), and a Linear
    without bias gets a zero bias of its output size. With cancelling weights
    and inputs near 1e4 a float32 copy puts the reach set far from the true
    outputs (float64 torch, membership by an LP on the star's rows)."""
    from tests._oracles.membership import uncovered

    net = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2),
                        nn.ReLU(), nn.Linear(2, 3, bias=False)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[0.1], [0.1000001]]))
        net[0].bias.copy_(torch.tensor([1.0 / 3.0, -0.1]))
        net[2].weight.copy_(torch.tensor([[1.0, -1.0], [1e-5, 0.0]]))
        net[2].bias.copy_(torch.tensor([0.0, 0.0]))
        net[4].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.1, 0.2]]))
    layers = extract_layers(net)
    linears = [l for l in layers if isinstance(l, LinearLayer)]
    mods = [m for m in net if isinstance(m, nn.Linear)]
    for lay, mod in zip(linears, mods):
        assert lay.W.dtype == np.float64 and lay.b.dtype == np.float64
        assert np.array_equal(lay.W, mod.weight.detach().numpy())
        expected_b = (mod.bias.detach().numpy() if mod.bias is not None
                      else np.zeros(mod.out_features))
        assert np.array_equal(lay.b, expected_b)

    S_in = Star.from_bounds(np.array([[1.0e4]]), np.array([[2.0e4]]))
    out, _ = relaxed_reach(S_in, layers)
    Y = _forward(net, np.linspace(1.0e4, 2.0e4, 33).reshape(-1, 1))
    assert uncovered([out], Y) == []


# --------------------------------------------------------------------------- #
# relaxed_reach options, refusals and provenance                                #
# --------------------------------------------------------------------------- #
def _lp_star():
    """[-1, 1]^2 cut by the row x0 + x1 <= -1."""
    S = Star.from_bounds(np.array([[-1.0], [-1.0]]), np.array([[1.0], [1.0]]))
    return Star(S.V, np.vstack([S.C, [[1.0, 1.0]]]), np.vstack([S.d, [[-1.0]]]),
                S.predicate_lb, S.predicate_ub)


@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu", "dual_gpu"])
def test_bound_modes_read_the_constraint_rows_as_documented(bound_mode,
                                                            monkeypatch):
    """Neuron 0 is x0 + x1 + 1/2: its box range [-3/2, 5/2] crosses 0, its
    exact range over the rows is [-3/2, -1/2] (vertex enumeration), so it is
    stable inactive. Every mode records bounds that hold the exact range;
    lp_cpu documents the exact LP range padded by the solver tolerance, so
    its neuron is not relaxed and its upper bound is within that pad of
    -1/2; box documents the predicate-box interval and relaxes it. Catches
    lp_cpu answered with the box interval."""
    monkeypatch.setattr("n2v.utils.lpsolver_gpu.gpu_available", lambda: False)
    layers = [LinearLayer(np.array([[1.0, 1.0], [1.0, -1.0]]),
                          np.array([0.5, 0.25])), ReluLayer()]
    S_in = _lp_star()
    from tests._oracles import rational as R
    corners = R.vertices([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]],
                         [1, 1, 1, 1, -1])
    exact = [(min(sum(w * v for w, v in zip(row, x)) + b0 for x in corners),
              max(sum(w * v for w, v in zip(row, x)) + b0 for x in corners))
             for row, b0 in (((1, 1), 0.5), ((1, -1), 0.25))]
    out, meta = relaxed_reach(S_in, layers, bound_mode=bound_mode)
    lb, ub = out.relu_bounds[0]
    for j, (lo, hi) in enumerate(exact):
        assert lb[j] <= lo and ub[j] >= hi, f"neuron {j}: [{lb[j]}, {ub[j]}] misses [{lo}, {hi}]"
    keys = {m.key for m in meta}
    if bound_mode == "lp_cpu":
        assert NeuronKey(0, 0) not in keys
        assert ub[0] <= -0.5 + 1e-6
    if bound_mode == "box":
        assert NeuronKey(0, 0) in keys


def test_relaxed_reach_rejects_an_unknown_bound_mode():
    layers = extract_layers(_seq(2, 3, 1, seed=0))
    S_in = Star.from_bounds(np.array([[-1.0], [-1.0]]), np.array([[1.0], [1.0]]))
    with pytest.raises(ValueError, match="bound_mode"):
        relaxed_reach(S_in, layers, bound_mode="exact")


@pytest.mark.parametrize("neuron", [-1, 3], ids=["negative", "past-the-layer"])
def test_relaxed_reach_rejects_a_fixed_neuron_index_out_of_range(neuron):
    """A negative index would wrap and pin the last neuron's phase."""
    layers = extract_layers(_seq(2, 3, 1, seed=0))
    S_in = Star.from_bounds(np.array([[-1.0], [-1.0]]), np.array([[1.0], [1.0]]))
    with pytest.raises(ValueError, match="out of range"):
        relaxed_reach(S_in, layers, {NeuronKey(0, neuron): Phase.ACTIVE})


@pytest.mark.parametrize("upper_only", [True, False],
                         ids=["infinite-above", "infinite-both-sides"])
def test_relaxed_reach_refuses_non_finite_neuron_bounds(upper_only):
    """A predicate box infinite above gives the neuron x0 the bounds
    [-1, inf]: a NaN slope would follow, so the layer must raise."""
    lb = np.array([[-1.0], [-1.0]]) if upper_only else np.array([[-np.inf], [-1.0]])
    S_in = Star(np.hstack([np.zeros((2, 1)), np.eye(2)]), np.zeros((0, 2)),
                np.zeros((0, 1)), lb, np.array([[np.inf], [1.0]]))
    layers = [LinearLayer(np.eye(2), np.zeros(2)), ReluLayer()]
    with pytest.raises(ValueError, match="non-finite"):
        relaxed_reach(S_in, layers)


def test_relaxed_reach_rejects_an_unknown_layer_object():
    """A hand-built layer list holding a torch module is not reached as if
    that layer were the identity."""
    S_in = Star.from_bounds(np.array([[-1.0], [-1.0]]), np.array([[1.0], [1.0]]))
    layers = [LinearLayer(np.eye(2), np.zeros(2)), nn.ReLU()]
    with pytest.raises(TypeError, match="Unknown layer type"):
        relaxed_reach(S_in, layers)


@pytest.mark.parametrize("make", [
    lambda: [],
    lambda: extract_layers(nn.Sequential(nn.Flatten(), nn.Dropout(0.5))),
], ids=["no-layer", "flatten-and-dropout"])
def test_relaxed_reach_leaves_the_input_star_untouched_when_no_layer_changes_it(make):
    """The caller's input star is reused across calls: provenance goes on a
    fresh star, never on the input."""
    S_in = Star.from_bounds(np.array([[-1.0], [0.5]]), np.array([[1.0], [2.0]]))
    V = S_in.V.copy()
    out, meta = relaxed_reach(S_in, make(), {NeuronKey(0, 0): Phase.ACTIVE})
    assert out is not S_in and meta == []
    assert S_in.relax_meta is None and S_in.fixed is None
    assert S_in.checkpoints is None and S_in.bound_mode is None
    assert np.array_equal(out.V, V) and out.checkpoints == []


def test_relaxed_reach_records_a_copy_of_the_fixed_phases():
    """The star records the fixes it was reached under; a caller that
    changes its dict afterwards does not change them."""
    layers = extract_layers(_seq(2, 3, 3, 1, seed=1))
    S_in = Star.from_bounds(np.array([[-1.0], [-1.0]]), np.array([[1.0], [1.0]]))
    fixed = {NeuronKey(0, 0): Phase.ACTIVE}
    out, _ = relaxed_reach(S_in, layers, fixed)
    fixed[NeuronKey(1, 0)] = Phase.INACTIVE
    assert out.fixed == {NeuronKey(0, 0): Phase.ACTIVE}
