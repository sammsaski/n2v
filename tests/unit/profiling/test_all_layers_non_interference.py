"""Exhaustive non-interference + counter-accuracy across ALL layer types.

The load-bearing correctness property of an observation-only profiler: turning
profiling ON must produce **bit-identical** reach output to OFF. If it didn't,
both the measurements AND the tool's verdicts would be compromised when profiling
is enabled. We verify this for every dispatcher-supported layer type, across the
set representations it supports, driving the instrumented dispatcher entry
(``reach_layer``) directly.

A second block verifies counter ACCURACY (not just non-perturbation): population,
byte volume, FLOPs and classification counters equal independently-computed
ground truth.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Star, Zono, Box, ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops.dispatcher import reach_layer, _set_nbytes
from n2v.profiling import profile, LAYER

# Optional ONNX-wrapped op types (onnx2torch is a required dep, but guard anyway)
try:
    from onnx2torch.node_converters.neg import OnnxNeg
    from onnx2torch.node_converters.transpose import OnnxTranspose
    from onnx2torch.node_converters.reduce import OnnxReduceStaticAxes
    from onnx2torch.node_converters.roundings import OnnxRound
    _ONNX_OK = True
except Exception:  # pragma: no cover
    _ONNX_OK = False


# --------------------------------------------------------------------------- #
# Input builders
# --------------------------------------------------------------------------- #
def _flat_bounds(n=3):
    # crosses zero so activations actually split / relax
    return -0.5 * np.ones(n), 0.5 * np.ones(n)


def flat_star(n=3):
    lb, ub = _flat_bounds(n)
    return Star.from_bounds(lb, ub)


def flat_zono(n=3):
    lb, ub = _flat_bounds(n)
    return Zono.from_bounds(lb, ub)


def flat_box(n=3):
    lb, ub = _flat_bounds(n)
    return Box(lb.reshape(-1, 1), ub.reshape(-1, 1))


def img_star(h=4, w=4, c=2):
    lb = -0.3 * np.ones((h, w, c))
    ub = 0.3 * np.ones((h, w, c))
    return ImageStar.from_bounds(lb, ub, h, w, c)


def img_zono(h=4, w=4, c=2):
    lb = -0.3 * np.ones((h, w, c))
    ub = 0.3 * np.ones((h, w, c))
    return ImageZono.from_bounds(lb, ub, h, w, c)


def img_star_one_window():
    """2x2x1 ImageStar with an ambiguous argmax -> exact MaxPool splits into a
    bounded 2 stars (a full image would explode exponentially under exact)."""
    lb = np.array([[[0.4], [0.4]], [[-1.0], [-1.0]]])
    ub = np.array([[[0.6], [0.6]], [[-0.8], [-0.8]]])
    return ImageStar.from_bounds(lb, ub, 2, 2, 1)


def _linear(nin, nout):
    torch.manual_seed(0)
    return nn.Linear(nin, nout)


def _conv2d():
    torch.manual_seed(0)
    return nn.Conv2d(2, 3, 3, padding=1)


def _conv_transpose2d():
    torch.manual_seed(0)
    return nn.ConvTranspose2d(2, 3, 3, padding=1)


def _bn1d():
    torch.manual_seed(0)
    m = nn.BatchNorm1d(3)
    m.eval()
    return m


def _bn2d():
    torch.manual_seed(0)
    m = nn.BatchNorm2d(2)
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# Case table: (id, layer, inputs, method). Driven through reach_layer.
# --------------------------------------------------------------------------- #
def _cases():
    cases = [
        # --- flat dense / activation layers ---
        ("linear-star-exact", _linear(3, 4), [flat_star()], "exact"),
        ("linear-zono", _linear(3, 4), [flat_zono()], "approx"),
        ("linear-box", _linear(3, 4), [flat_box()], "approx"),
        ("batchnorm1d-star", _bn1d(), [flat_star()], "approx"),
        ("relu-star-exact", nn.ReLU(), [flat_star()], "exact"),
        ("relu-star-approx", nn.ReLU(), [flat_star()], "approx"),
        ("relu-zono", nn.ReLU(), [flat_zono()], "approx"),
        ("relu-box", nn.ReLU(), [flat_box()], "approx"),
        ("leakyrelu-star-exact", nn.LeakyReLU(0.1), [flat_star()], "exact"),
        ("leakyrelu-zono", nn.LeakyReLU(0.1), [flat_zono()], "approx"),
        ("leakyrelu-box", nn.LeakyReLU(0.1), [flat_box()], "approx"),
        ("sigmoid-star", nn.Sigmoid(), [flat_star()], "approx"),
        ("sigmoid-box", nn.Sigmoid(), [flat_box()], "approx"),
        ("tanh-star", nn.Tanh(), [flat_star()], "approx"),
        ("tanh-box", nn.Tanh(), [flat_box()], "approx"),
        ("softmax-star", nn.Softmax(dim=-1), [flat_star()], "approx"),
        ("dropout-passthrough", nn.Dropout(), [flat_star()], "approx"),
        ("identity-passthrough", nn.Identity(), [flat_star()], "approx"),
        # --- image / conv / pool layers ---
        ("conv2d-imagestar", _conv2d(), [img_star()], "exact"),
        ("conv_transpose2d-imagestar", _conv_transpose2d(), [img_star()], "exact"),
        ("conv1d-star", nn.Conv1d(2, 3, 3, padding=1), [flat_star(8)], "approx"),
        ("global_avgpool-imagestar", nn.AdaptiveAvgPool2d(1), [img_star()], "approx"),
        ("batchnorm2d-imagestar", _bn2d(), [img_star()], "approx"),
        # tiny single-window input: exact MaxPool splits but stays bounded
        ("maxpool2d-imagestar-exact", nn.MaxPool2d(2), [img_star_one_window()], "exact"),
        ("maxpool2d-imagestar-approx", nn.MaxPool2d(2), [img_star()], "approx"),
        ("avgpool2d-imagestar", nn.AvgPool2d(2), [img_star()], "approx"),
        ("flatten-imagestar", nn.Flatten(), [img_star()], "approx"),
        ("pad-imagestar", nn.ZeroPad2d(1), [img_star()], "approx"),
        ("upsample-imagestar", nn.Upsample(scale_factor=2), [img_star()], "approx"),
        # --- zono image ---
        ("conv2d-imagezono", _conv2d(), [img_zono()], "approx"),
        ("maxpool2d-imagezono", nn.MaxPool2d(2), [img_zono()], "approx"),
        ("avgpool2d-imagezono", nn.AvgPool2d(2), [img_zono()], "approx"),
    ]
    if _ONNX_OK:
        cases += [
            ("neg-star", OnnxNeg(), [flat_star()], "approx"),
            ("transpose-star", OnnxTranspose(perm=[0, 1]), [flat_star()], "exact"),
            # reduce over the feature dim (ONNX axis 1 = after the batch axis)
            ("reduce-star", OnnxReduceStaticAxes("ReduceMean", axes=[1], keepdims=0),
             [flat_star(4)], "approx"),
        ]
    return cases


_CASES = _cases()


def _repr_arrays(s):
    out = []
    for attr in ("V", "C", "d", "c", "lb", "ub", "center", "generators"):
        v = getattr(s, attr, None)
        if isinstance(v, np.ndarray):
            out.append((attr, v))
    return out


def _assert_identical(off, on):
    assert len(off) == len(on)
    for a, b in zip(off, on):
        aa, bb = _repr_arrays(a), _repr_arrays(b)
        assert aa and [k for k, _ in aa] == [k for k, _ in bb]
        for (ka, va), (kb, vb) in zip(aa, bb):
            assert np.array_equal(va, vb), f"{ka} perturbed by profiling"


# --------------------------------------------------------------------------- #
# Non-interference: ON output == OFF output, for every case.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_layer_non_interference(case):
    _id, layer, inputs, method = case

    off = reach_layer(layer, inputs, method)
    with profile(level="operation") as p:
        on = reach_layer(layer, inputs, method)

    _assert_identical(off, on)

    # the dispatcher always records a LAYER region with correct population/bytes
    layers = [r for r in p.records() if r.level == LAYER]
    assert len(layers) == 1
    rec = layers[0]
    assert rec.counters["n_sets_in"] == len(inputs)
    assert rec.counters["n_sets_out"] == len(on)
    assert rec.counters["set_bytes_out"] == sum(_set_nbytes(s) for s in on)
    assert rec.meta.get("layer_type") == type(layer).__name__


# --------------------------------------------------------------------------- #
# FLOPs must account for groups (depthwise/grouped convs connect each output
# channel to only in_channels/groups inputs).
# --------------------------------------------------------------------------- #
def test_conv2d_flops_accounts_for_groups():
    torch.manual_seed(0)
    plain = nn.Conv2d(2, 2, 3, padding=1)            # groups=1
    torch.manual_seed(0)
    depthwise = nn.Conv2d(2, 2, 3, padding=1, groups=2)  # each out-ch sees 1 in-ch

    def flops(layer):
        with profile(level="operation") as p:
            reach_layer(layer, [img_star(h=4, w=4, c=2)], "exact")
        return p.rollup()["totals"]["flops"]

    f_plain, f_depthwise = flops(plain), flops(depthwise)
    # same output shape; only in_channels/groups differs (2 vs 1) -> exactly 2x
    assert f_plain == 2 * f_depthwise


# --------------------------------------------------------------------------- #
# Edge cases.
# --------------------------------------------------------------------------- #
def test_edge_empty_input_no_region():
    """Empty input list short-circuits before opening a region (returns [])."""
    with profile() as p:
        out = reach_layer(nn.ReLU(), [], "exact")
    assert out == []
    assert [r.name for r in p.records()] == ["run"]  # no LAYER region opened


def test_edge_passthrough_returns_same_objects():
    """Pass-through layers (Identity/Dropout) return inputs unchanged, and the
    profiler records the region without perturbing them."""
    s = flat_star()
    for layer in (nn.Identity(), nn.Dropout()):
        with profile(level="layer") as p:
            out = reach_layer(layer, [s], "approx")
        assert len(out) == 1 and out[0] is s
        assert p.find(type(layer).__name__)[0].counters["n_sets_out"] == 1


def test_edge_all_stable_relu_no_split():
    """A ReLU whose pre-activations are all strictly one-signed never splits:
    n_unstable == 0, n_split == 0, population unchanged."""
    # bias +5 dominates inputs in [-0.5,0.5] -> all 3 neurons strictly active
    lin = _linear(3, 3)
    with torch.no_grad():
        lin.bias.copy_(torch.tensor([5.0, 5.0, 5.0]))
    pre = reach_layer(lin, [flat_star()], "exact")
    with profile(level="layer") as p:
        out = reach_layer(nn.ReLU(), pre, "exact")
    c = p.find("ReLU")[0].counters
    assert c["n_unstable"] == 0
    assert c.get("n_split", 0) == 0
    assert c["n_sets_out"] == c["n_sets_in"] == 1
    assert len(out) == 1


def test_edge_fully_crossing_relu_splits_2_to_the_k():
    """A ReLU whose every pre-activation crosses zero splits to 2^k stars."""
    # identity-ish weights, zero bias, inputs symmetric about 0 -> all cross
    lin = _linear(3, 3)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(3))
        lin.bias.copy_(torch.zeros(3))
    pre = reach_layer(lin, [flat_star()], "exact")
    with profile(level="layer") as p:
        out = reach_layer(nn.ReLU(), pre, "exact")
    c = p.find("ReLU")[0].counters
    assert c["n_unstable"] == 3
    assert len(out) == 2 ** 3
    assert c["n_split"] == c["n_sets_out"] - c["n_sets_in"]
