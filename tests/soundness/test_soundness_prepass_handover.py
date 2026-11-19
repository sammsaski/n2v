"""The bound pre-pass hands each nonlinear layer bounds of its own input, for
the input set being reached, in the order its kernel reads them; and reach
with the pre-pass on still holds every true output.

Component: ``n2v/utils/bounds_precomputation.py``: the entry
``compute_intermediate_bounds``, the input of each pass (``_extract_bounds``
with ``_hwc_flat_to_chw_flat`` for the interval pass, ``_convert_to_zono``
for the zonotope pass), the order of what the interval pass records
(``_record_spatial_bounds``), the walkers of the zonotope pass
(``_zono_sequential``, ``_zono_graphmodule``); and the handover in
``n2v/nn/reach.py`` (``_handle_graphmodule`` runs the pass once per reach
and gives ``layer_bounds[node.name]`` to the node of that name) and in the
dispatcher, which hands them to the Star and ImageStar ReLU and LeakyReLU
kernels (approx uses them as given, exact widens them by the LP pad and lets
them decide a neuron's phase).

Promise. Direct: every interval recorded for a nonlinear layer holds the
real value of that layer's input at every member point of the input set, in
the layer's own entry order ((height, width, channel) for an image layer,
the tensor's row-major order for a flat one), for every input set type, and
the recorded keys are exactly the nonlinear layers (child indices for an
``nn.Sequential``, node names for a graph). Through reach: with
``precompute_bounds`` 'ibp', 'zono' or True, on an ``nn.Sequential``, a
traced torch graph with functional activations and an ONNX file, approx and
exact, the result holds the exact image of the input set; a second reach of
the same model object holds its own input set.

A wrong handover shows only where the bounds a kernel reads are narrower
than the true range of its own input and still overlap the star's own
enclosure (a kernel drops a whole layer's bounds when one interval misses
its enclosure). So every neuron of the models here crosses zero with a range
of its own: bounds of another neuron, another layer, another input set or
another layout cut a true range and leave true outputs outside.

Oracles, none of them n2v: the real layer inputs in ``fractions.Fraction``
by an evaluator written here (Linear, Conv2d, ReLU, LeakyReLU, Flatten) on
dyadic data; the exact image of the input polytope in rationals
(``tests._oracles.rational``) with exact membership in each returned set;
point replay on torch in float64 or onnxruntime with membership decided by
the LP oracle on the returned sets' rows (``assert_covered``). Comparisons
are one-sided with zero slack against exact references.
"""

import itertools
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as TF
from onnx import helper

from n2v.sets import Zono
from n2v.utils.bounds_precomputation import compute_intermediate_bounds
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests.cell_support import (
    assert_covered, constrained_image_star, constrained_star, flat_set,
    image_set, reach_model, replay_onnx, replay_torch, samples, save_onnx,
)
from tests.known_bugs import known_bug

F = Fraction
_NONLINEAR = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)


# --------------------------------------------------------------------------- #
# Models                                                                        #
# --------------------------------------------------------------------------- #
def _set(layer, weight, bias=None):
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(np.asarray(weight, np.float64)))
        if bias is not None:
            layer.bias.copy_(torch.tensor(np.asarray(bias, np.float64)))
    return layer


# The flat model: x in [-3, 2] x [-1, 4]. Layer 1 ranges h0 in [-3, 9/2],
# h1 in [-7/2, 4], h2 in [-7/2, 3/2]; layer 2 (interval bounds) g0 in
# [-9/2, 19/4], g1 in [-2, 11/4], g2 in [-5/2, 13/4]. Every neuron crosses
# zero, really as well (g at the corners (-3, -1) and (2, 4) has opposite
# signs in every entry), and no two ranges are equal.
_FW1 = [[1.0, 0.5], [-0.5, 1.0], [0.25, -0.75]]
_FB1 = [0.5, -1.5, 0.25]
_FW2 = [[1.0, -1.0, 0.5], [-0.5, 0.25, 1.0], [0.5, 0.5, -1.0]]
_FB2 = [-0.5, 0.25, -1.0]
_FW3 = [[1.0, 0.5, -0.25], [-1.0, 1.0, 0.5], [0.25, -0.5, 1.0]]
_FB3 = [0.0, 0.125, -0.25]
_SLOPE = 0.25
_FLO, _FHI = np.array([-3.0, -1.0]), np.array([2.0, 4.0])


def _flat_sequential():
    """Linear(2, 3), ReLU, Linear(3, 3), LeakyReLU(1/4), Linear(3, 3)."""
    return nn.Sequential(
        _set(nn.Linear(2, 3).double(), _FW1, _FB1), nn.ReLU(),
        _set(nn.Linear(3, 3).double(), _FW2, _FB2), nn.LeakyReLU(_SLOPE),
        _set(nn.Linear(3, 3).double(), _FW3, _FB3)).eval()


class _Functional(nn.Module):
    """The flat model with ``F.relu`` and ``F.leaky_relu``: traced, the
    activations are call_function nodes, keyed by their own node names."""

    def __init__(self):
        super().__init__()
        self.fc1 = _set(nn.Linear(2, 3).double(), _FW1, _FB1)
        self.fc2 = _set(nn.Linear(3, 3).double(), _FW2, _FB2)
        self.fc3 = _set(nn.Linear(3, 3).double(), _FW3, _FB3)

    def forward(self, x):
        h = TF.relu(self.fc1(x))
        return self.fc3(TF.leaky_relu(self.fc2(h), negative_slope=_SLOPE))


def _flat_onnx(tmp_path):
    """The flat model as an ONNX file of MatMul, Add, Relu, LeakyRelu (the
    affine layers become graph operations the reach runs itself)."""
    inits, nodes, prev = {}, [], "X"
    for k, (W, b) in enumerate([(_FW1, _FB1), (_FW2, _FB2), (_FW3, _FB3)]):
        inits[f"W{k}"] = np.asarray(W, np.float32).T.copy()
        inits[f"B{k}"] = np.asarray(b, np.float32)
        nodes += [helper.make_node("MatMul", [prev, f"W{k}"], [f"m{k}"]),
                  helper.make_node("Add", [f"m{k}", f"B{k}"], [f"a{k}"])]
        prev = f"a{k}"
        if k == 0:
            nodes.append(helper.make_node("Relu", [prev], ["r0"]))
            prev = "r0"
        elif k == 1:
            nodes.append(helper.make_node("LeakyRelu", [prev], ["r1"],
                                          alpha=_SLOPE))
            prev = "r1"
    nodes[-1].output[0] = "Y"
    return save_onnx(tmp_path, nodes, [1, 2], [1, 3], inits, name="flat")


def _flat_pieces(input_pieces):
    """The exact image of the flat model on ``input_pieces``."""
    pieces = R.relu(R.affine(input_pieces, _FW1, _FB1))
    pieces = R.leaky_relu(R.affine(pieces, _FW2, _FB2), _SLOPE)
    return R.affine(pieces, _FW3, _FB3)


# The image model: an image (C, H, W) = (2, 2, 4), every pixel straddling 0
# with an interval of its own ([-a, b], a and b in 1 1/4 .. 3).
# Conv2d(2 -> 3, kernel (1, 2)) gives (3, 2, 3): channels, height and width
# all differ, so a transposed or swapped layout moves bounds onto other
# neurons; every one of its 18 outputs crosses 0, from about -7 .. -3 1/2 to
# about 4 1/4 .. 7, no two alike. Then ReLU, Flatten, a Linear(18, 18) that
# is diagonal with signs +1, -1, +1, ... and shifts -2, +2, -2, ... (each
# second-layer neuron then crosses 0 as well, with a range of its own), and
# ReLU. Each output follows one neuron of each layer, so a wrong bound on
# one neuron shows in the output; a dense layer would mix the relaxations of
# many neurons and hide it.
_IC, _IH, _IW = 2, 2, 4
_IK = np.array([[[[1.0, -0.5]], [[0.5, 0.25]]],
                [[[-1.0, 0.75]], [[0.25, -0.5]]],
                [[[0.5, 0.5]], [[-0.75, 1.0]]]])
_IKB = np.array([0.25, -0.125, 0.0])
_ISIGN = np.array([1.0, -1.0] * 9)
_IL1 = np.diag(_ISIGN)
_IL1B = -2.0 * _ISIGN
_IRNG = np.random.default_rng(7)
_IRNG.integers(-4, 5, size=(3, 18))        # keeps the box below unchanged
_ILO = -(_IRNG.integers(5, 13, size=(_IC, _IH, _IW)) / 4.0)
_IHI = _IRNG.integers(5, 13, size=(_IC, _IH, _IW)) / 4.0


def _image_sequential():
    return nn.Sequential(
        _set(nn.Conv2d(2, 3, (1, 2)).double(), _IK, _IKB), nn.ReLU(),
        nn.Flatten(), _set(nn.Linear(18, 18).double(), _IL1, _IL1B),
        nn.ReLU()).eval()


def _image_onnx(tmp_path):
    """The image model as an ONNX file: Conv, Relu, Flatten, Gemm, Relu (a
    Gemm with transB is loaded as a torch Linear)."""
    inits = {"K": _IK.astype(np.float32), "KB": _IKB.astype(np.float32),
             "L1": _IL1.astype(np.float32), "L1B": _IL1B.astype(np.float32)}
    nodes = [
        helper.make_node("Conv", ["X", "K", "KB"], ["c"], kernel_shape=[1, 2]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("Flatten", ["r"], ["f"], axis=1),
        helper.make_node("Gemm", ["f", "L1", "L1B"], ["g"], transB=1),
        helper.make_node("Relu", ["g"], ["Y"])]
    return save_onnx(tmp_path, nodes, [1, _IC, _IH, _IW], [1, 18], inits,
                     name="image")


# --------------------------------------------------------------------------- #
# Exact layer inputs                                                            #
# --------------------------------------------------------------------------- #
def _frac_array(a):
    a = np.asarray(a, np.float64)
    return np.array([F(float(v)) for v in a.ravel()],
                    dtype=object).reshape(a.shape)


def _exact_apply(layer, v):
    """``layer`` on the Fraction array ``v`` (no batch axis), exactly."""
    if isinstance(layer, nn.Linear):
        W = _frac_array(layer.weight.detach().numpy())
        out = W.dot(v.reshape(-1))
        if layer.bias is not None:
            out = out + _frac_array(layer.bias.detach().numpy())
        return out
    if isinstance(layer, nn.Conv2d):
        assert layer.stride == (1, 1) and layer.padding == (0, 0) \
            and layer.dilation == (1, 1) and layer.groups == 1
        W = _frac_array(layer.weight.detach().numpy())
        o, c, kh, kw = W.shape
        _, h, w = v.shape
        out = np.empty((o, h - kh + 1, w - kw + 1), dtype=object)
        b = (_frac_array(layer.bias.detach().numpy()) if layer.bias is not None
             else [F(0)] * o)
        for k, i, j in itertools.product(range(o), range(out.shape[1]),
                                         range(out.shape[2])):
            out[k, i, j] = b[k] + sum(
                W[k, ci, a, bb] * v[ci, i + a, j + bb]
                for ci in range(c) for a in range(kh) for bb in range(kw))
        return out
    if isinstance(layer, nn.ReLU):
        return np.vectorize(lambda t: max(t, F(0)), otypes=[object])(v)
    if isinstance(layer, nn.LeakyReLU):
        s = F(float(layer.negative_slope))
        return np.vectorize(lambda t: t if t >= 0 else s * t,
                            otypes=[object])(v)
    if isinstance(layer, nn.Flatten):
        return v.reshape(-1)
    raise AssertionError(f"no exact rule for {type(layer).__name__}")


def _layer_inputs(model, x):
    """The exact input of every child of the Sequential ``model`` at the
    point ``x`` (the network's order, no batch axis)."""
    v, inputs = _frac_array(x), []
    for layer in model:
        inputs.append(v)
        v = _exact_apply(layer, v)
    return inputs


def _entry_order(v):
    """A layer input as the recorded arrays list it: an image tensor
    (C, H, W) in (height, width, channel) order, a flat one as it is."""
    return v.transpose(1, 2, 0).reshape(-1) if v.ndim == 3 else v.reshape(-1)


def _child_index(target, key):
    """The child of the Sequential that a recorded key names: the key itself
    for the sequential walker, the call_module node's target for a graph."""
    if isinstance(key, int):
        return key
    node = next(n for n in target.graph.nodes if n.name == key)
    return int(node.target)


def _expected_keys(model, target):
    if not isinstance(target, fx.GraphModule):
        return {i for i, m in enumerate(model) if isinstance(m, _NONLINEAR)}
    return {n.name for n in target.graph.nodes if n.op == "call_module"
            and isinstance(model[int(n.target)], _NONLINEAR)}


def _check_recorded(model, target, bounds, points, what):
    """Every recorded interval holds the exact layer input at every point,
    and the keys are exactly the nonlinear layers of the model."""
    assert set(bounds) == _expected_keys(model, target), (
        f"{what}: recorded keys {sorted(map(str, bounds))}, the nonlinear "
        f"layers are {sorted(map(str, _expected_keys(model, target)))}")
    exact = [_layer_inputs(model, x) for x in points]
    for key, (lo, hi) in bounds.items():
        k = _child_index(target, key)
        lo, hi = _frac_array(lo).reshape(-1), _frac_array(hi).reshape(-1)
        for p, inputs in enumerate(exact):
            v = _entry_order(inputs[k])
            assert v.size == lo.size == hi.size, (
                f"{what}: layer {key} records {lo.size} entries, its input "
                f"has {v.size}")
            out = [i for i in range(v.size) if not lo[i] <= v[i] <= hi[i]]
            assert not out, (
                f"{what}: layer {key} ({type(model[k]).__name__}), point "
                f"{p}: entries {out} of {v.size} outside the recorded bounds; "
                f"first: real {float(v[out[0]])}, recorded "
                f"[{float(lo[out[0]])}, {float(hi[out[0]])}]")


def _arrays(s):
    """The arrays a set holds, copied; the ranges it caches about itself
    (``state_lb``, ``state_ub``) are left out, as ``cell_support`` does."""
    return {k: np.array(v, copy=True) for k, v in vars(s).items()
            if isinstance(v, np.ndarray) and k not in ("state_lb", "state_ub")}


# --------------------------------------------------------------------------- #
# Input sets with member points                                                 #
# --------------------------------------------------------------------------- #
def _wide_zono():
    """A Zono with four generators on two entries whose hull is the flat box
    [-3, 2] x [-1, 4] (the shape a relaxation leaves), with member points
    ``c + V a`` for the corners and half-grid points of ``a``."""
    c = np.array([[-0.5], [1.5]])
    V = np.array([[1.5, 0.0, 0.5, 0.5], [0.0, 1.25, 0.75, -0.5]])
    pts = [(c + V @ np.array(a).reshape(-1, 1)).ravel()
           for a in itertools.product([-1.0, -0.5, 0.0, 0.5, 1.0], repeat=4)]
    return Zono(c, V), pts[::5] + pts[-1:]


def _flat_inputs():
    """``{kind: (set, member points)}`` over the flat box: a Box, a box
    Star, a Star whose predicate variables outnumber its entries (from the
    wide Zono), a Star with a binding row that is not a box row, and the
    wide Zono. Every bound is beyond +-1 and every interval wider than 2,
    so reading the predicate box [-1, 1] in place of the state range
    misses."""
    pts = samples(_FLO, _FHI, 48)
    zono, zono_pts = _wide_zono()
    row_star, row_pts = constrained_star(_FLO, _FHI, n=48, seed=3)
    return {"Box": (flat_set("Box", _FLO, _FHI), pts),
            "Star": (flat_set("Star", _FLO, _FHI), pts),
            "Star+gens": (zono.to_star(), zono_pts),
            "Star+row": (row_star, row_pts),
            "Zono": (zono, zono_pts)}


def _image_inputs():
    pts = samples(_ILO, _IHI, 24)
    row_star, row_pts = constrained_image_star(_ILO, _IHI, n=24, seed=3)
    return {"ImageStar": (image_set("ImageStar", _ILO, _IHI), pts),
            "ImageStar+row": (row_star, row_pts),
            "ImageZono": (image_set("ImageZono", _ILO, _IHI), pts)}


# --------------------------------------------------------------------------- #
# 1. What the pass records, called directly                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("walker", ["sequential", "traced"])
@pytest.mark.parametrize("model_kind", ["flat", "image"])
def test_recorded_bounds_hold_every_layer_input(model_kind, walker, method):
    """``compute_intermediate_bounds`` on the flat model (Box, Star, a Star
    with more predicate variables than entries, a Star with a binding row,
    a Zono with more generators than entries) and on the image model
    (ImageStar, ImageStar with a binding row, ImageZono), on the walker of
    an ``nn.Sequential`` (keys are child indices) and of a traced graph
    (keys are node names): every recorded interval holds the exact input of
    its layer at every member point, in the layer's entry order, and the
    keys are the nonlinear layers. The input set comes back unchanged.

    Catches an input interval taken from one end of a Box, Star, ImageStar
    or Zono only, or from a Zono's ends in swapped order, or from a Star's
    predicate box instead of its state range (interval pass,
    ``_extract_bounds``); an input zonotope built from a Box's lower end
    only or from an ImageStar with height and width swapped
    (``_convert_to_zono``); an image set's ranges reordered to the interval
    pass's (C, H, W) layout with height and width swapped
    (``_hwc_flat_to_chw_flat``); bounds of a spatial layer reordered back
    with the wrong permutation, or recorded from one end only
    (``_record_spatial_bounds``); a zonotope's row bounds taken over the
    wrong generator range or the hull taken of the wrong ends
    (``_zono_row_bounds``, ``_zono_layer_bounds``); and bounds recorded from
    the sets leaving a layer instead of entering it (both walkers)."""
    model = _flat_sequential() if model_kind == "flat" else _image_sequential()
    target = model if walker == "sequential" else fx.symbolic_trace(model)
    inputs = _flat_inputs() if model_kind == "flat" else _image_inputs()
    for kind, (input_set, points) in inputs.items():
        what = f"{method} pass, {walker} walker, {model_kind} model, {kind}"
        before = _arrays(input_set)
        bounds = compute_intermediate_bounds(target, input_set, method=method)
        after = _arrays(input_set)
        if before.keys() != after.keys() or any(
                not np.array_equal(before[k], after[k]) for k in before):
            pytest.fail(f"{what}: the pre-pass changed the input set")
        _check_recorded(model, target, bounds, points, what)


@pytest.mark.parametrize("method", ["ibp", "zono"])
@pytest.mark.parametrize("shared", [
    pytest.param(False, id="distinct-relus"),
    pytest.param(True, id="one-relu-object-twice", marks=known_bug(
        16, "the sequential walkers of the bound pre-pass (_ibp_sequential, "
            "_zono_sequential) walk model.children(), which yields a module "
            "object once: an nn.Sequential that uses one ReLU object at two "
            "places is walked without its second use, the ReLU there gets no "
            "bounds and a later key names the wrong child with the bounds of "
            "another function")),
])
def test_a_sequential_that_reuses_a_module_records_every_nonlinear_layer(
        shared, method):
    """``nn.Sequential(Linear, act, Linear, act, Linear, ReLU)`` where the
    two ``act`` are one ``nn.ReLU`` object (a common way to write a model)
    or two: called directly on the Sequential, the pre-pass records exactly
    the three nonlinear layers (indices 1, 3, 5), each holding its exact
    input at every member point. The distinct-object case is the passing
    neighbour.

    Catches a walk over the model's distinct children instead of its
    positions: it computes another function after the repeated module and
    files bounds under the index of a Linear."""
    act = nn.ReLU()
    layers = [_set(nn.Linear(2, 3).double(), _FW1, _FB1), act,
              _set(nn.Linear(3, 3).double(), _FW2, _FB2),
              act if shared else nn.ReLU(),
              _set(nn.Linear(3, 3).double(), _FW3, _FB3), nn.ReLU()]
    model = nn.Sequential(*layers).eval()
    input_set, points = _flat_inputs()["Star"]
    bounds = compute_intermediate_bounds(model, input_set, method=method)
    _check_recorded(model, model, bounds, points,
                    f"{method} pass on a Sequential, shared={shared}")


# --------------------------------------------------------------------------- #
# 2. Through reach: the exact image of the flat model                           #
# --------------------------------------------------------------------------- #
def _input_pieces(input_set):
    """The input set as one exact piece: its predicate polytope (rows and
    finite bounds) mapped by its centre and generators, read from the
    stored arrays."""
    c, G, C, d, lb, ub = R.star(input_set)
    A, b = [list(row) for row in C], list(d)
    for j in range(len(lb)):
        unit = [F(int(i == j)) for i in range(len(lb))]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-u for u in unit])
            b.append(-lb[j])
    return R.affine(R.polytope(A, b), G, c)


def _assert_holds_image(pieces, sets, what):
    """Every piece of the exact image has all its vertices in one returned
    set, exactly (a convex set holding a polytope's vertices holds the
    polytope). A piece whose image is inside another piece's image (a
    region on a kink) is skipped; a vertex shared by several pieces is
    decided once per set."""
    images = [frozenset(p.image()) for p in pieces]
    keep = [i for i, im in enumerate(images)
            if not any(im < o or (im == o and j < i)
                       for j, o in enumerate(images) if j != i)]
    members = [R.Membership(R.star(s)) for s in sets]
    known = {}

    def inside(k, y):
        if (k, y) not in known:
            known[(k, y)] = members[k](y)
        return known[(k, y)]
    escaped, spread = [], []
    for i in keep:
        image = sorted(images[i])
        if any(all(inside(k, y) for y in image) for k in range(len(sets))):
            continue
        outside = [y for y in image
                   if not any(inside(k, y) for k in range(len(sets)))]
        if outside:
            escaped.append((i, [float(v) for v in outside[0]]))
        else:
            spread.append(i)
    assert not escaped, (
        f"{what}: {len(escaped)} of {len(keep)} pieces of the exact image "
        f"have a vertex in none of the {len(sets)} returned sets; first: "
        f"piece {escaped[0][0]}, vertex {escaped[0][1]}")
    assert not spread, (
        f"{what}: pieces {spread} of the exact image are spread over several "
        f"returned sets and in no single one")


_IMAGE_CACHE = {}


def _flat_image(kind):
    """The exact image of the flat model on the flat input set ``kind``,
    computed once per kind."""
    if kind not in _IMAGE_CACHE:
        _IMAGE_CACHE[kind] = _flat_pieces(
            _input_pieces(_flat_inputs()[kind][0]))
    return _IMAGE_CACHE[kind]


def _flat_model(entry, tmp_path):
    """``(model handed to reach, replay of points)`` for an entry route: an
    ``nn.Sequential`` (reach traces it), a module with functional
    activations (reach traces it), a graph traced by the caller (reach uses
    it as given) or an ONNX file loaded by ``load_onnx``."""
    if entry == "onnx":
        path = _flat_onnx(tmp_path)
        return load_onnx(path), lambda pts: replay_onnx(path, pts, (1, 2))
    model = {"sequential": _flat_sequential, "functional": _Functional,
             "traced": lambda: fx.symbolic_trace(_flat_sequential())}[entry]()
    return model, lambda pts: replay_torch(model, pts, (1, 2))


def _flat_reach_params():
    out = []
    for entry in ("sequential", "functional", "onnx"):
        for pre in ("ibp", "zono"):
            for method in ("approx", "exact"):
                out.append(pytest.param(entry, pre, method, "Star",
                                        id=f"{entry}-{pre}-{method}-Star"))
    for pre in ("ibp", "zono"):
        out.append(pytest.param("sequential", pre, "approx", "Star+row",
                                id=f"sequential-{pre}-approx-Star+row"))
    out.append(pytest.param("sequential", True, "exact", "Star",
                            id="sequential-true-exact-Star"))
    return out


@pytest.mark.parametrize("entry, pre, method, kind", _flat_reach_params())
def test_reach_with_the_pre_pass_holds_the_exact_image(entry, pre, method,
                                                      kind, tmp_path):
    """Reach of Linear, ReLU, Linear, LeakyReLU(1/4), Linear on a Star (and
    a Star with a binding row) with ``precompute_bounds`` 'ibp', 'zono' and
    True (read as 'ibp'), given as an ``nn.Sequential`` (call_module nodes
    once traced), as a module with functional activations (call_function
    nodes) and as an ONNX file (MatMul and Add run by the graph walker,
    ONNX node names): the result holds every vertex of the exact image,
    exactly, and every replayed output by the LP oracle. Under 'exact' the
    LeakyReLU receives the several pieces of the ReLU's split with one set
    of bounds, those of the whole input set.

    Catches bounds handed to the wrong node (another layer's, a producer's,
    a ReLU's output range instead of its input range), bounds of another
    input set, a functional activation or an ONNX node that is handed the
    bounds of another node, and a kernel that reads them at the wrong end;
    each cuts the true range of a crossing neuron, so true outputs fall
    outside."""
    model, replay = _flat_model(entry, tmp_path)
    input_set = _flat_inputs()[kind][0]
    what = f"{entry} model, precompute_bounds={pre}, {method}, {kind}"
    out = reach_model(model, input_set, method, what, precompute_bounds=pre)
    _assert_holds_image(_flat_image(kind), out, what)
    # 128 replayed points per row: under approx (one returned set); under
    # exact, where the exact image above is the verdict, 16.
    n = 128 if method == "approx" else 16
    points = (samples(_FLO, _FHI, n) if kind == "Star"
              else constrained_star(_FLO, _FHI, n=n, seed=3)[1])
    assert_covered(out, replay(points), what)


@pytest.mark.parametrize("pre", ["ibp", "zono"])
@pytest.mark.parametrize("entry", ["sequential", "traced", "onnx"])
def test_a_second_reach_of_one_model_holds_its_own_input_set(entry, pre,
                                                            tmp_path):
    """One model object reached twice with the pre-pass on: first on the
    corner box [-3, -2] x [-1, 0], where the first ReLU's neuron 0 is
    stably negative and the others are narrow, then on the whole box
    [-3, 2] x [-1, 4]. The second result holds the second set's exact image.

    Catches bounds kept from an earlier reach (on the model object, its
    graph or its modules) and handed to the next: the corner box's bounds
    overlap the whole box's, so the kernel keeps them, zeroes neuron 0 and
    cuts the others."""
    model, replay = _flat_model(entry, tmp_path)
    first = flat_set("Star", [-3.0, -1.0], [-2.0, 0.0])
    reach_model(model, first, "approx", f"{entry} first reach",
                precompute_bounds=pre)
    input_set, _ = _flat_inputs()["Star"]
    what = f"{entry} model, second reach, precompute_bounds={pre}"
    out = reach_model(model, input_set, "approx", what, precompute_bounds=pre)
    _assert_holds_image(_flat_image("Star"), out, what)
    assert_covered(out, replay(samples(_FLO, _FHI, 128)), what)


# --------------------------------------------------------------------------- #
# 3. Through reach: an image model                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pre", ["ibp", "zono"])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageStar+row"])
@pytest.mark.parametrize("entry", ["sequential", "onnx"])
def test_image_reach_with_the_pre_pass_holds_every_output(entry, kind, pre,
                                                          tmp_path):
    """Approx reach of Conv2d(2 -> 3, (1, 2)), ReLU, Flatten, a diagonal
    Linear(18, 18), ReLU on a (2, 2, 4) ImageStar (and one with a binding
    row) with ``precompute_bounds`` 'ibp' and 'zono', as an
    ``nn.Sequential`` and as an ONNX file: 128 replayed outputs (corners
    and grid points of the box, or member points on the cut face) are
    members of the result by the LP oracle.

    Catches recorded bounds of the convolution's output that reach the
    ReLU kernel in (channel, height, width) order or in another permutation
    of its (height, width, channel) rows, flat bounds after the Flatten in
    the image's row order, an input image read with height and width
    swapped, and an image's ranges read from one end: every neuron's range
    contains 0 and the ranges differ, so bounds of another neuron cut a
    true range, and each output follows one neuron of each layer."""
    if entry == "onnx":
        path = _image_onnx(tmp_path)
        model = load_onnx(path)
    else:
        model = _image_sequential()
    if kind == "ImageStar":
        input_set = image_set("ImageStar", _ILO, _IHI)
        points = samples(_ILO, _IHI, 128)
    else:
        input_set, points = constrained_image_star(_ILO, _IHI, n=128, seed=3)
    what = f"{entry} image model, precompute_bounds={pre}, {kind}"
    out = reach_model(model, input_set, "approx", what, precompute_bounds=pre,
                      input_shape=(_IC, _IH, _IW))
    outputs = (replay_onnx(path, points, (1, _IC, _IH, _IW))
               if entry == "onnx"
               else replay_torch(model, points, (1, _IC, _IH, _IW)))
    assert_covered(out, outputs, what)
