"""Soundness cells for the layers that reach passes through unchanged: Dropout,
Identity, Squeeze and Unsqueeze, the dropout functions and the value-keeping
tensor methods.

What is covered. The layers that ``n2v/nn/layer_ops/dispatcher.py`` returns
unchanged (``_PASSTHROUGH_TYPES``: ``nn.Identity``, ``nn.Dropout``,
``nn.Dropout2d``, ``nn.Dropout3d`` with a drop probability above zero in
evaluation mode, and the converter's ``OnnxDropoutDynamic`` that ONNX Dropout
of opset 12 and later loads as), and the converter's four Squeeze and
Unsqueeze modules (``_SHAPE_IDENTITY_TYPES``, static and dynamic axes), which
the dispatcher answers as the identity on a flat set and must refuse on an
image set. Every set type: Box, Zono, Star (both methods), Hexatope,
Octatope, ImageStar (both methods), ImageZono; through ``reach_layer`` with
several input sets at once, through ``load_onnx`` and
``reach_pytorch_model`` in a graph where the layer sits between two Gemm
nodes (behind an exact ReLU split, so the layer receives several sets), and
in an image graph where a pooling along the height reads the layout the
pass-through returned. The graph walker's own identity rules in
``n2v/nn/reach.py``: a dropout function whose training flag is the literal
False (``_is_inference_identity_function``) and a copy, layout or
float32/float64 cast method (``_is_identity_method``) hand their input's
sets on, every other spelling (a random dropout, a cast that rounds) is
refused; through a torch model on every set type.

Oracle. The exact image of an identity is its input. The input sets are
written from arrays (the builders of ``test_soundness_linear.py`` and
``test_soundness_flatten.py``) and their union is a list of rational pieces
(``tests/_oracles/rational.py``). In exact arithmetic with no slack, every
vertex of every input piece must lie in one returned set (``escapes``,
``straddlers``); for the exact method also every vertex of every returned
set must be a point of the input (``unreached``). A pass-through that keeps
the bounding box but moves a row leaves out the input's vertex on that row,
or the replayed points on the cut face. On top, member points are replayed
through the torch layer in float64 (or the graph on onnxruntime) and must be
members of the returned sets by one LP on each set's own rows
(``assert_covered``, slack ``LP_TOL`` only). Weights, biases and inputs are
dyadic, so float32, float64 and the fractions agree.

Layout. A flat set is in the tensor's row-major order. An image set stores
(height, width, channel); a pass-through must return an image set of the
same stored shape, so the exact pieces, built in (channel, height, width)
order, are permuted to the stored order before the exact check
(``hwc_index``), and the replay is compared with ``out_chw``.

Grids. The pass-through layer types share one branch per set lane, and so do
the four Squeeze and Unsqueeze modules: every type is run on the Star lane
under the exact method (the strongest check) and one type on each other
(kind, method), with a drop probability above zero, so that a branch that
scales by the keep probability shows. The image refusal is kept for one and
for two channels on both image lanes.

Defect of today's library marked here (strict expected failure): an ONNX
Dropout whose ``training_mode`` input is the constant True loads as
``OnnxDropoutDynamic``, whose converter and whose dispatcher branch both
ignore that input, and is answered as the identity although onnxruntime
zeroes entries at random and scales the rest (registry AFFINE-7). The same
answer is given on all seven set types; one
case per lane of the three that the model entry reaches most (Star, Zono,
Box) is marked.

Not covered here: a torch ``nn.Dropout`` in training mode is answered as the
evaluation-mode identity; ``NeuralNetwork`` and ``load_onnx`` call
``.eval()`` and DESIGN section 9 analyses the evaluation-mode module, so that
is the promise, not a defect. The bound pre-pass branches for these layers
belong to ``test_soundness_bounds_prepass.py``.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper
from onnx2torch.node_converters.dropout import OnnxDropoutDynamic
from onnx2torch.node_converters.squeeze import (OnnxSqueezeDynamicAxes,
                                                OnnxSqueezeStaticAxes)
from onnx2torch.node_converters.unsqueeze import (OnnxUnsqueezeDynamicAxes,
                                                  OnnxUnsqueezeStaticAxes)

from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests.cell_support import (assert_covered, flat_box, flat_set, hwc_index,
                                image_set, pixel_box, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx)
from tests.known_bugs import known_bug
from tests.soundness.test_soundness_flatten import (_C, _H, _SHAPE, _W,
                                                    _image_family)
from tests.soundness.test_soundness_linear import (_N_IN, _ROW,
                                                   _assert_holds_exact_image,
                                                   _box_generators, _build,
                                                   _exact_input, _family,
                                                   _member_points)

_FLAT_KIND_METHODS = (("Box", "approx"), ("Zono", "approx"),
                      ("Hexatope", "approx"), ("Octatope", "approx"),
                      ("Star", "approx"), ("Star", "exact"))
_IMAGE_KIND_METHODS = (("ImageStar", "exact"), ("ImageStar", "approx"),
                       ("ImageZono", "approx"))
_N_POINTS = 64
# Replayed member points per cell of this file (PROTOCOL 3.3); the builders
# below keep 64 by default for the files that import them.
_N_REPLAY = 128


# --------------------------------------------------------------------------- #
# Inputs and replays                                                            #
# --------------------------------------------------------------------------- #
def _thin(points, n=_N_POINTS):
    """At most ``n`` of ``points``, spread over the list (the first and the
    last kept), so the LP replay stays cheap; the exact check sees every
    vertex anyway."""
    if len(points) <= n:
        return list(points)
    keep = np.unique(np.linspace(0, len(points) - 1, n).round().astype(int))
    return [points[i] for i in keep]


def _flat_inputs(kind, n_points=_N_POINTS):
    """``(sets, pieces, points)``: a list of four flat sets of ``kind`` of
    four entries each, the shape a split leaves: one set cut by a binding row
    that is not a box row (a sheared generator for a Zono; a plain box of
    other bounds for a Box), placed 4 away from the others in every entry, so
    that the image of the first set alone cannot hold the others even after
    the bounding boxes of several layers; then three sets that cut another
    box along one coordinate at dyadic points (shares 1, 2 and 5 of 8).
    Their union as rational pieces, and at most ``n_points`` member points
    of the sets (half from the first set, half from the others)."""
    first, first_pieces, first_points = _family(
        kind, "one" if kind == "Box" else "row", seed=1, shift=4.0)
    rest, rest_pieces, rest_points = _family(kind, "several", seed=0)
    return (first + rest, first_pieces + rest_pieces,
            _thin(first_points, n_points // 2)
            + _thin(rest_points, n_points // 2))


def _image_inputs(kind, n_points=_N_POINTS):
    """``(sets, pieces, points)`` for a (2, 3, 5) image: one set cut by a
    binding row (a generator over two pixels for an ImageZono), then the
    three sets of a split along one pixel. Pieces are over the
    (C, H, W)-flattened tensor, points are (C, H, W) tensors, at most
    ``n_points`` of them."""
    first, first_pieces, first_points = _image_family(kind, "row")
    rest, rest_pieces, rest_points = _image_family(kind, "several")
    return (first + rest, first_pieces + rest_pieces,
            _thin(first_points, n_points // 2)
            + _thin(rest_points, n_points // 2))


def _wide_flat_family(kind):
    """``(sets, pieces, points)``: one flat set of ``_N_IN`` entries over
    ``flat_box(seed=2)`` times 8 (bounds up to about 8 away from zero, most
    widths above 2, one entry of zero width), cut by the binding row of
    ``kind`` (none for a Box; for a Zono a generator over two entries and two
    more generators than entries). Every predicate variable ranges over
    [-1, 1], so a result that reads one variable in place of another holds
    the outputs only while the data stays inside that range."""
    lo, hi = flat_box(_N_IN, seed=2, zero_width=True)
    c, G = _box_generators(8 * lo, 8 * hi)
    row, levels = None, 5
    if kind == "Zono":
        G[:, 0] += 0.5 * G[:, 1]
        more = np.array([[2.0, 0.0], [0.0, 1.0], [4.0, -2.0], [-1.0, 2.0]])
        more[hi == lo] = 0.0
        G, levels = np.hstack([G, more]), 3
    elif kind != "Box":
        row = _ROW[kind]
    return ([_build(kind, c, G, row)], _exact_input(c, G, row),
            _member_points(c, G, row, levels))


def _to_stored(pieces):
    """Rational pieces over the (C, H, W)-flattened image, mapped to the
    (H, W, C) order an image set stores."""
    order = hwc_index(_C, _H, _W)
    P = np.zeros((order.size, order.size))
    P[np.arange(order.size), order] = 1.0
    return R.affine(pieces, P.tolist())


def _replay(module, points, in_shape, *extra):
    """Outputs of ``module`` on ``points`` in float64, eval mode, each a flat
    vector in row-major order; ``extra`` are further inputs of the forward
    call (the axes tensor of the dynamic-axes modules)."""
    module = module.double().eval()
    out = []
    with torch.no_grad():
        for x in points:
            y = module(torch.as_tensor(np.asarray(x, dtype=np.float64)
                                       .reshape(in_shape)), *extra)
            out.append(y.numpy().astype(np.float64).reshape(-1))
    return out


def _height_pool_matrix():
    """The (2 * 2 * 5, 30) matrix of AvgPool2d with kernel (2, 1) and stride
    (1, 1) on a (2, 3, 5) image, both sides in (C, H, W) order, read off
    torch's float64 pooling of unit tensors. It averages each pixel with the
    one below it: pixels that are ``_W * _C`` apart in an image set's stored
    order, so it depends on which number is the height."""
    pool = nn.AvgPool2d((2, 1), stride=(1, 1))
    n = _C * _H * _W
    columns = []
    for j in range(n):
        unit = torch.zeros(n, dtype=torch.float64)
        unit[j] = 1.0
        columns.append(pool(unit.reshape(1, _C, _H, _W)).reshape(-1).numpy())
    return np.stack(columns, axis=1)


def _picking_weights(n_in, seed):
    """A (3, n_in) dyadic weight whose rows each read three entries."""
    rng = np.random.default_rng(seed)
    W = np.zeros((3, n_in))
    for row in range(3):
        cols = rng.choice(n_in, size=3, replace=False)
        W[row, cols] = rng.integers(1, 5, size=3) / 4.0
    return W


# --------------------------------------------------------------------------- #
# Pass-through layers through reach_layer                                       #
# --------------------------------------------------------------------------- #
# (name, layer factory, flat input shape for the replay). Dropout2d and
# Dropout3d get a 4-D and a 5-D tensor (torch warns on a 2-D one).
_PASSTHROUGH = (
    ("Identity", nn.Identity, (1, _N_IN)),
    ("Dropout(0.5)", lambda: nn.Dropout(p=0.5), (1, _N_IN)),
    ("Dropout2d(0.25)", lambda: nn.Dropout2d(p=0.25), (1, _N_IN, 1, 1)),
    ("Dropout3d(0.75)", lambda: nn.Dropout3d(p=0.75), (1, _N_IN, 1, 1, 1)),
    ("OnnxDropoutDynamic", OnnxDropoutDynamic, (1, _N_IN)),
)


def _flat_grid(entries, star_exact_all):
    """Cells ``(kind, method, *entry)`` with ids ``kind-method-name``: every
    entry on the Star lane under the exact method, the entry named
    ``star_exact_all[kind]`` on each other (kind, method)."""
    by_name = {entry[0]: entry for entry in entries}
    cells = [pytest.param("Star", "exact", *entry, id=f"Star-exact-{entry[0]}")
             for entry in entries]
    for kind, method in _FLAT_KIND_METHODS:
        if (kind, method) == ("Star", "exact"):
            continue
        entry = by_name[star_exact_all[kind, method]]
        cells.append(pytest.param(kind, method, *entry,
                                  id=f"{kind}-{method}-{entry[0]}"))
    return cells


# One type per other lane: Dropout(0.5) has a drop probability, so a branch
# that scales by the keep probability shows; Identity and OnnxDropoutDynamic
# have none.
_PASSTHROUGH_ONE = {(kind, method): "Dropout(0.5)"
                    for kind, method in _FLAT_KIND_METHODS}


@pytest.mark.parametrize("kind,method,name,factory,in_shape",
                         _flat_grid(_PASSTHROUGH, _PASSTHROUGH_ONE))
def test_pass_through_layer_returns_the_same_flat_sets(kind, method, name,
                                                       factory, in_shape):
    """``reach_layer`` on a pass-through layer with four flat input sets
    (one with a binding row, three of a split) must return the inputs:
    every input vertex in one returned set and every replayed point a member
    by the rows; for the exact method also no returned vertex outside the
    input. Catches a pass-through branch that answers only the first set
    of the list, that scales the set by the keep probability 1 - p (training
    time dropout without its rescaling), that negates or otherwise maps it,
    or that drops or changes a constraint row; and a layer type missing from
    the pass-through list (then refused, which fails here)."""
    layer = factory().eval()
    sets, pieces, points = _flat_inputs(kind, _N_REPLAY)
    what = f"{name} on {len(sets)} x {kind} ({method})"
    result = reach_one_layer(layer, sets, method, what)
    assert all(type(s) is type(sets[0]) for s in result), (
        f"{what}: returned {[type(s).__name__ for s in result]}")
    _assert_holds_exact_image(result, pieces, what, exact=(method == "exact"))
    assert_covered(result, _replay(layer, points, in_shape), what)


@pytest.mark.parametrize("kind,method,name,factory", [
    pytest.param(kind, method, "Dropout(0.5)", _PASSTHROUGH[1][1],
                 id=f"{kind}-{method}-Dropout(0.5)")
    for kind, method in _IMAGE_KIND_METHODS])
def test_pass_through_layer_returns_the_same_image_sets(kind, method, name,
                                                        factory):
    """The same on a (2, 3, 5) image set: the returned sets must be image
    sets of the same stored height, width and channels, holding exactly the
    input, so the next spatial layer reads the right pixels. Catches what the
    flat cell catches, on the image branches of the Star and Zono routers,
    and a pass-through that returns a flat set or swaps height and width."""
    layer = factory().eval()
    sets, pieces, points = _image_inputs(kind, _N_REPLAY)
    what = f"{name} on {len(sets)} x {kind} ({method})"
    result = reach_one_layer(layer, sets, method, what)
    assert all(type(s) is type(sets[0]) for s in result), (
        f"{what}: returned {[type(s).__name__ for s in result]}")
    _assert_holds_exact_image(result, _to_stored(pieces), what,
                              exact=(method == "exact"))
    assert_covered(result, _replay(layer, points, _SHAPE), what,
                   out_chw=(_C, _H, _W))


@pytest.mark.parametrize("layer", [
    pytest.param(nn.AlphaDropout(p=0.5), id="AlphaDropout")])
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_dropout_variants_outside_the_list_refuse_or_pass_through(kind,
                                                                  layer):
    """Alpha dropout is the identity in evaluation mode too, but it is not in
    the pass-through list. It must be refused (``NotImplementedError``) or
    answered with the input itself; never answered as another layer."""
    layer = layer.eval()
    sets, pieces, points = _flat_inputs(kind)
    what = f"{type(layer).__name__} on {kind}"
    try:
        result = reach_one_layer(layer, sets, "approx", what)
    except NotImplementedError:
        return
    _assert_holds_exact_image(result, pieces, what)
    assert_covered(result, _replay(layer, points, (1, _N_IN)), what)


# --------------------------------------------------------------------------- #
# Squeeze and Unsqueeze through reach_layer                                     #
# --------------------------------------------------------------------------- #
# (name, layer factory, input shape of the flat set's tensor, extra forward
# inputs). Each changes only the rank of the tensor, so its row-major entries
# are unchanged.
_SHAPE_OPS = (
    ("UnsqueezeStatic[1]", lambda: OnnxUnsqueezeStaticAxes(axes=[1]),
     (1, _N_IN), ()),
    ("SqueezeStatic[1]", lambda: OnnxSqueezeStaticAxes(axes=[1]),
     (1, 1, _N_IN), ()),
    ("SqueezeStatic[]", lambda: OnnxSqueezeStaticAxes(axes=None),
     (1, _N_IN, 1), ()),
    ("UnsqueezeDynamic[2]", OnnxUnsqueezeDynamicAxes, (1, _N_IN),
     (torch.tensor([2]),)),
    ("SqueezeDynamic[-1]", OnnxSqueezeDynamicAxes, (1, _N_IN, 1),
     (torch.tensor([-1]),)),
)

# One module per other lane, a different one on each.
_SHAPE_ONE = {("Star", "approx"): "SqueezeDynamic[-1]",
              ("Zono", "approx"): "UnsqueezeStatic[1]",
              ("Box", "approx"): "SqueezeStatic[]",
              ("Hexatope", "approx"): "UnsqueezeDynamic[2]",
              ("Octatope", "approx"): "SqueezeStatic[1]"}


@pytest.mark.parametrize("kind,method,name,factory,in_shape,extra",
                         _flat_grid(_SHAPE_OPS, _SHAPE_ONE))
def test_squeeze_and_unsqueeze_return_the_same_flat_sets(kind, method, name,
                                                         factory, in_shape,
                                                         extra):
    """``reach_layer`` on a Squeeze or Unsqueeze module with four flat input
    sets must return the inputs (every input vertex in one returned set, in
    exact arithmetic; for the exact method no returned vertex outside the
    input; replayed points members by the rows). Catches a shape
    identity that answers only the first set of a list (as an exact split
    hands it several), one that negates or maps the set, one that drops a
    row, and a module type missing from the shape-identity list."""
    layer = factory()
    sets, pieces, points = _flat_inputs(kind, _N_REPLAY)
    what = f"{name} on {len(sets)} x {kind} ({method})"
    result = reach_one_layer(layer, sets, method, what)
    assert all(type(s) is type(sets[0]) for s in result), (
        f"{what}: returned {[type(s).__name__ for s in result]}")
    _assert_holds_exact_image(result, pieces, what, exact=(method == "exact"))
    assert_covered(result, _replay(layer, points, in_shape, *extra), what)


# (kind, method, module name, channels): every module on each image lane
# (ImageStar under the two methods together, ImageZono), one and two
# channels on each.
_IMAGE_REFUSALS = (
    ("ImageStar", "exact", "UnsqueezeStatic[1]", 1),
    ("ImageStar", "exact", "SqueezeStatic[1]", 1),
    ("ImageStar", "exact", "SqueezeStatic[]", 2),
    ("ImageStar", "approx", "UnsqueezeDynamic[2]", 1),
    ("ImageStar", "approx", "SqueezeDynamic[-1]", 1),
    ("ImageZono", "approx", "UnsqueezeStatic[1]", 1),
    ("ImageZono", "approx", "SqueezeStatic[1]", 1),
    ("ImageZono", "approx", "SqueezeStatic[]", 1),
    ("ImageZono", "approx", "UnsqueezeDynamic[2]", 2),
    ("ImageZono", "approx", "SqueezeDynamic[-1]", 1),
)
_SHAPE_FACTORY = {name: factory for name, factory, _, _ in _SHAPE_OPS}


@pytest.mark.parametrize("kind,method,name,factory,channels", [
    pytest.param(kind, method, name, _SHAPE_FACTORY[name], channels,
                 id=f"{kind}-{method}-{name}-{channels}")
    for kind, method, name, channels in _IMAGE_REFUSALS])
def test_squeeze_and_unsqueeze_refuse_an_image_set(kind, method, name,
                                                   factory, channels):
    """An image set records (height, width, channels) and nothing else, so
    after a Squeeze or Unsqueeze it would describe a tensor of the old rank
    and the next axis-indexed layer would act on the wrong axis. Until the
    set tracks the tensor shape, the dispatcher must refuse
    (``NotImplementedError``), for one channel as for several: catches a
    refusal limited to images with several channels (on the ImageStar and
    on the ImageZono branch), and one that checks the ImageStar branch
    only."""
    lo, hi = pixel_box(channels, _H, _W)
    image = image_set(kind, lo, hi)
    with pytest.raises(NotImplementedError):
        reach_one_layer(factory(), [image, image_set(kind, lo, hi)], method,
                        f"{name} on a {channels}-channel {kind}")


# --------------------------------------------------------------------------- #
# The same layers inside ONNX graphs, through load_onnx and the model entry     #
# --------------------------------------------------------------------------- #
def _gemm_nodes(middle, relu=True):
    """Gemm(4 -> 3, bias), Relu (or nothing), the ``middle`` nodes (input
    "M", output "N"), Gemm(3 -> 3) with a zero bias (opset 10 needs the
    input)."""
    head = [helper.make_node("Gemm", ["X", "W1", "B1"],
                             ["A" if relu else "M"], transB=1)]
    if relu:
        head.append(helper.make_node("Relu", ["A"], ["M"]))
    return (head + middle
            + [helper.make_node("Gemm", ["N", "W2", "B2"], ["Y"], transB=1)])


def _weights():
    """Dyadic Gemm weights. Under the exact method the Relu splits the Star
    of the 'row' family into four sets, each holding vertices of the image
    that the other three miss, and W2 is square and invertible (determinant
    -3/8), so a node that drops any one set of the split shows (a map onto
    fewer entries can lay one piece's image inside another's)."""
    rng = np.random.default_rng(49)
    return {"W1": (rng.integers(-4, 5, (3, _N_IN)) / 4.0).astype(np.float32),
            "B1": (rng.integers(-8, 9, 3) / 8.0).astype(np.float32),
            "W2": (rng.integers(-4, 5, (3, 3)) / 4.0).astype(np.float32),
            "B2": np.zeros(3, np.float32)}


_GRAPHS = {
    # name: (middle nodes from "M" to "N", extra initializers, opset)
    "Dropout-opset13": ([helper.make_node("Dropout", ["M"], ["N"])], {}, 13),
    "Dropout-ratio-opset13": ([helper.make_node("Dropout", ["M", "r"], ["N"])],
                              {"r": np.array(0.25, np.float32)}, 13),
    "Dropout-opset10": ([helper.make_node("Dropout", ["M"], ["N"],
                                          ratio=0.5)], {}, 10),
    "Unsqueeze-Squeeze-opset13": (
        [helper.make_node("Unsqueeze", ["M", "a1"], ["U"]),
         helper.make_node("Squeeze", ["U", "a1"], ["N"])],
        {"a1": np.array([1], np.int64)}, 13),
    "Unsqueeze-Squeeze-opset11": (
        [helper.make_node("Unsqueeze", ["M"], ["U"], axes=[2]),
         helper.make_node("Squeeze", ["U"], ["N"], axes=[2])], {}, 11),
    # One pass-through and one shape node, each once: a sign flip in a
    # branch shows here, where a Squeeze after an Unsqueeze undoes it.
    "Dropout-Unsqueeze-Flatten-opset11": (
        [helper.make_node("Dropout", ["M"], ["D"], ratio=0.5),
         helper.make_node("Unsqueeze", ["D"], ["U"], axes=[1]),
         helper.make_node("Flatten", ["U"], ["N"], axis=1)], {}, 11),
}


def _graph(tmp_path, name, relu=True):
    """The ONNX file of graph ``name`` and its exact network (W1, b1, W2),
    dyadic. With ``relu`` the graph has a Relu, run on the set types whose
    ReLU is exact in rationals or padded (Star, Box); without it, on Zono,
    Hexatope and Octatope (whose ReLU rounds: registry ROUND-10, HEXOCT-4,
    cells of other files)."""
    middle, inits, opset = _GRAPHS[name]
    weights = _weights()
    path = save_onnx(tmp_path, _gemm_nodes(middle, relu), [1, _N_IN], [1, 3],
                     {**weights, **inits}, name=name.replace("-", "_"),
                     opset=opset)
    return path, (weights["W1"].astype(np.float64).tolist(),
                  weights["B1"].astype(np.float64).tolist(),
                  weights["W2"].astype(np.float64).tolist())


# Every graph (one per converter route) behind the exact Star split; the
# approx Star and the Box lane on a Dropout and a Squeeze route each; the
# Dropout and Unsqueeze routes on the Zono, Hexatope and Octatope lanes (no
# Relu there; a torch model cannot reach a Squeeze, it refuses
# ``x.squeeze``).
_GRAPH_CELLS = (
    [("Star", "exact", name) for name in _GRAPHS]
    + [("Star", "approx", "Dropout-opset13"),
       ("Star", "approx", "Unsqueeze-Squeeze-opset13"),
       ("Box", "approx", "Dropout-opset10"),
       ("Box", "approx", "Unsqueeze-Squeeze-opset11"),
       ("Zono", "approx", "Dropout-Unsqueeze-Flatten-opset11"),
       ("Hexatope", "approx", "Dropout-Unsqueeze-Flatten-opset11"),
       ("Octatope", "approx", "Dropout-Unsqueeze-Flatten-opset11")])


@pytest.mark.parametrize("kind,method,name", [
    pytest.param(kind, method, name, id=f"{kind}-{method}-{name}")
    for kind, method, name in _GRAPH_CELLS])
def test_pass_through_and_shape_nodes_in_an_onnx_graph(tmp_path, kind, method,
                                                       name):
    """Gemm, Relu (on Star and Box), then an ONNX Dropout (opset 10 loads as
    ``nn.Dropout``, opset 13 as ``OnnxDropoutDynamic``) or an Unsqueeze and a
    Squeeze (static and dynamic axes), or a Dropout, an Unsqueeze and a
    Flatten, then Gemm, through ``load_onnx`` and ``reach_pytorch_model``,
    replayed on onnxruntime. Behind the exact ReLU split the middle nodes
    receive four Stars at once, and the last Gemm reads the rows they
    return. Catches a pass-through or shape identity that keeps only one set
    of a split, that scales by the keep probability, that negates (the
    single-node graph; a Squeeze after an Unsqueeze undoes a sign flip), and
    a node loop that hands the second Gemm another node's sets. For the
    exact method the result must also have no point outside the image."""
    relu = kind in ("Star", "Box")
    path, (W1, b1, W2) = _graph(tmp_path, name, relu)
    sets, pieces, points = _family(kind, "one" if kind == "Box" else "row")
    what = f"Gemm, {'Relu, ' if relu else ''}{name}, Gemm on {kind} ({method})"
    result = reach_model(load_onnx(path), sets, method, what)
    hidden = R.affine(pieces, W1, b1)
    exact = R.affine(R.relu(hidden) if relu else hidden, W2)
    _assert_holds_exact_image(result, exact, what, exact=(method == "exact"))
    assert_covered(result, replay_onnx(path, _thin(points, _N_REPLAY),
                                       [1, _N_IN]), what)


def _image_dropout_graph(tmp_path):
    """X (1, 2, 3, 5), ONNX Dropout (opset 13), AveragePool with kernel
    (2, 1) and stride (1, 1) (each pixel averaged with the one below it),
    Flatten, Gemm(20 -> 3) whose rows read three pooled entries; and the
    exact map from the (C, H, W)-flattened input to the output."""
    W = _picking_weights(_C * (_H - 1) * _W, seed=43)
    nodes = [helper.make_node("Dropout", ["X"], ["D"]),
             helper.make_node("AveragePool", ["D"], ["P"], kernel_shape=[2, 1],
                              strides=[1, 1]),
             helper.make_node("Flatten", ["P"], ["F"], axis=1),
             helper.make_node("Gemm", ["F", "W"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, list(_SHAPE), [1, 3],
                     {"W": W.astype(np.float32)}, name="image_dropout")
    return path, (W @ _height_pool_matrix()).tolist()


@pytest.mark.parametrize("kind,method", _IMAGE_KIND_METHODS)
def test_dropout_on_an_image_set_in_an_onnx_graph(tmp_path, kind, method):
    """ONNX Dropout on a (2, 3, 5) image set, then an AveragePool along the
    height, Flatten and a Gemm, through ``load_onnx`` and
    ``reach_pytorch_model``, against the exact image and onnxruntime.
    The pooling pairs pixels that are a whole row apart in the stored order,
    so it reads the height and width the pass-through returned. Catches a
    pass-through that returns the image set with height and width exchanged
    (a pool or a flatten alone sees no difference), as a flat set, in another
    order, scaled, or only the first set; and the ONNX Dropout converter
    route on an image."""
    path, M = _image_dropout_graph(tmp_path)
    sets, pieces, points = _image_family(kind, "row")
    what = f"Dropout, AveragePool, Flatten, Gemm on {kind} ({method})"
    result = reach_model(load_onnx(path), sets, method, what)
    _assert_holds_exact_image(result, R.affine(pieces, M), what,
                              exact=(method == "exact"))
    assert_covered(result, replay_onnx(path, _thin(points, _N_REPLAY),
                                       list(_SHAPE)), what)


def _training_mode_graph(tmp_path, training_mode):
    """A lone ONNX Dropout (opset 13) with ratio 1/2 and a constant
    ``training_mode`` input."""
    return save_onnx(tmp_path, [helper.make_node("Dropout", ["X", "r", "t"],
                                                 ["Y"])],
                     [1, _N_IN], [1, _N_IN],
                     {"r": np.array(0.5, np.float32),
                      "t": np.array(training_mode)},
                     name=f"dropout_training_{training_mode}", opset=13)


def _masked_outputs(points, keep):
    """Every output ONNX Dropout can give on each point: each entry is kept
    and scaled by 1 / keep, or set to zero (the operator's definition with
    ``training_mode`` true), over all masks."""
    out = []
    for x in points:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        for mask in np.ndindex(*(2,) * x.size):
            out.append(np.where(np.array(mask) == 1, x / keep, 0.0))
    return out


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_dropout_with_training_mode_false_is_the_identity(tmp_path, kind):
    """ONNX Dropout whose ``training_mode`` input is the constant False is the
    identity: the answer must hold every onnxruntime output. The passing
    neighbour of the cell below (same graph builder, same replay)."""
    path = _training_mode_graph(tmp_path, False)
    lo, hi = flat_box(_N_IN, seed=3)
    points = samples(lo, hi, n=_N_REPLAY)
    outputs = replay_onnx(path, points, [1, _N_IN])
    if not all(np.array_equal(y, np.asarray(x).reshape(-1))
               for x, y in zip(points, outputs)):
        pytest.fail("onnxruntime changed the input with training_mode False")
    what = f"Dropout(training_mode=False) on {kind}"
    result = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                         what)
    assert_covered(result, outputs, what)


@pytest.mark.parametrize("kind", [
    pytest.param(kind, marks=known_bug(
        7, "an ONNX Dropout whose training_mode input is the constant True "
           "drops entries at random and scales the rest by 1/(1-ratio), but "
           "it loads as OnnxDropoutDynamic and the dispatcher passes it "
           "through as the identity, so the outputs onnxruntime gives lie "
           "outside the result"))
    for kind in ("Star", "Zono", "Box")])
def test_dropout_with_training_mode_true_refuses_or_holds_every_mask(
        tmp_path, kind):
    """ONNX Dropout whose ``training_mode`` input is the constant True is
    random: each entry is zeroed or scaled by 1/(1-ratio). The library must
    refuse (``NotImplementedError``) or return sets that hold the output of
    every mask. The reference is the operator's definition over all 16 masks,
    and onnxruntime is checked to give outputs of that family (and not the
    input), so the graph really is in training mode."""
    path = _training_mode_graph(tmp_path, True)
    lo, hi = flat_box(_N_IN, seed=3)
    points = samples(lo, hi, n=16)
    possible = _masked_outputs(points, keep=0.5)
    replayed = replay_onnx(path, points, [1, _N_IN])
    for i, y in enumerate(replayed):
        family = possible[16 * i:16 * (i + 1)]
        if not any(np.array_equal(y, z) for z in family):
            pytest.fail(f"onnxruntime output {y} of point {i} is no mask of "
                        f"the input")
    if all(np.array_equal(y, np.asarray(x).reshape(-1))
           for x, y in zip(points, replayed)):
        pytest.fail("onnxruntime did not apply the dropout")
    what = f"Dropout(training_mode=True) on {kind}"
    try:
        result = reach_model(load_onnx(path), flat_set(kind, lo, hi),
                             "approx", what)
    except NotImplementedError:
        return
    assert_covered(result, possible, what)


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_onnx_identity_node_refuses_or_returns_the_same_sets(tmp_path, kind):
    """ONNX Identity loads as the converter's ``OnnxCopyIdentity``, which is
    not in the pass-through list. It must be refused (``NotImplementedError``)
    or answered with exactly the input; never as another operation."""
    path = save_onnx(tmp_path, [helper.make_node("Identity", ["X"], ["Y"])],
                     [1, _N_IN], [1, _N_IN], name="identity")
    sets, pieces, points = _family(kind, "one" if kind == "Box" else "row")
    what = f"ONNX Identity on {kind}"
    try:
        result = reach_model(load_onnx(path), sets, "approx", what)
    except NotImplementedError:
        return
    _assert_holds_exact_image(result, pieces, what, exact=True)
    assert_covered(result, replay_onnx(path, _thin(points), [1, _N_IN]), what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_squeeze_of_a_one_channel_image_then_reduce_refuses_or_is_sound(
        tmp_path, kind):
    """ONNX Squeeze of the channel axis of a one-channel (1, 1, 3, 5) image,
    then ReduceSum over axis 1 (the height of the squeezed tensor), through
    ``load_onnx`` and ``reach_pytorch_model``. The library must refuse or
    hold every onnxruntime output: an image set passed through the Squeeze
    keeps its old rank, and the ReduceSum then sums the size-1 channel axis
    instead of the height."""
    lo, hi = pixel_box(1, _H, _W, seed=2)
    nodes = [helper.make_node("Squeeze", ["X", "a"], ["S"]),
             helper.make_node("ReduceSum", ["S", "a"], ["Y"], keepdims=0)]
    path = save_onnx(tmp_path, nodes, [1, 1, _H, _W], [1, _W],
                     {"a": np.array([1], np.int64)}, name="squeeze_reduce")
    what = f"Squeeze, ReduceSum on a one-channel {kind}"
    try:
        result = reach_model(load_onnx(path), image_set(kind, lo, hi),
                             "approx", what)
    except NotImplementedError:
        return
    assert_covered(result, replay_onnx(path, samples(lo, hi), [1, 1, _H, _W]),
                   what, stored_order=True)


# --------------------------------------------------------------------------- #
# Dropout functions and tensor methods in a torch model (the graph walker)      #
# --------------------------------------------------------------------------- #
def _dyadic_linear(n_in, n_out, seed):
    """A float64 Linear layer with dyadic weights and bias."""
    rng = np.random.default_rng(seed)
    layer = nn.Linear(n_in, n_out).double()
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(
            rng.integers(-4, 5, (n_out, n_in)) / 4.0))
        layer.bias.copy_(torch.from_numpy(rng.integers(-8, 9, n_out) / 8.0))
    return layer


def _affine_of(layer):
    return (layer.weight.detach().numpy().tolist(),
            layer.bias.detach().numpy().tolist())


class _FlatChain(nn.Module):
    """``nn.Identity``, fc1, a ReLU (``relu``), then ``nn.Dropout`` and every
    value-keeping spelling the graph walker reads as the identity, in a row,
    then fc2 and a final cast to float32.
    Traced, the modules are ``call_module`` nodes; the ``F.*`` dropout calls
    carry their training flag as the keyword False (``self.training`` in
    evaluation mode, a literal, or the default of the alpha variants);
    ``torch.dropout`` and ``torch.alpha_dropout`` carry it as the positional
    False. The methods are copies, layout changes and casts between float32
    and float64, with the extra keywords the rule drops. Float64 so that
    ``.double()`` is a cast of the network's own type. fc2 is square and
    invertible: a map onto fewer entries can lay the image of one piece of a
    split inside another's, and a node that drops a set then goes unseen."""

    def __init__(self, relu):
        super().__init__()
        self.relu = relu
        # fc1 splits the wide Star into three sets under the exact ReLU, each
        # holding vertices of the image that the other two miss; fc2 is
        # square and invertible (determinant -7/16), so the images of the
        # pieces stay apart and a node that drops any one set shows.
        self.fc1, self.fc2 = _dyadic_linear(_N_IN, 3, 65), _dyadic_linear(3, 3,
                                                                         66)
        self.drop, self.identity = nn.Dropout(p=0.5), nn.Identity()

    def forward(self, x):
        # The two modules apart, with fc1 (and its bias) between them: a
        # sign flip in the pass-through branch would cancel in a row.
        h = self.fc1(self.identity(x))
        if self.relu:
            h = F.relu(h)
        h = self.drop(h)
        h = F.dropout(h, 0.5, self.training)
        h = F.alpha_dropout(h, 0.5)
        h = F.feature_alpha_dropout(h, 0.25)
        h = torch.dropout(h, 0.5, False)
        h = torch.alpha_dropout(h, 0.5, False)
        h = h.clone().detach().contiguous()
        h = h.contiguous(memory_format=torch.contiguous_format)
        h = h.double().to(torch.float64).type(torch.float64)
        h = h.to(dtype=torch.float64, non_blocking=True).to(torch.float64,
                                                            copy=True)
        return self.fc2(h).float()


class _ImageChain(nn.Module):
    """An ``nn.Dropout2d`` module, the 2-D and feature dropout spellings and
    three methods on a (1, 2, 3, 5) image, then an AvgPool2d along the height
    (kernel (2, 1), stride (1, 1)), ``flatten(1)`` and a Linear(20 -> 3) that
    reads three pooled entries per row. One pass-through module only: two in
    a row would undo a branch that swaps height and width or flips a
    sign."""

    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout2d(p=0.25)
        self.pool = nn.AvgPool2d((2, 1), stride=(1, 1))
        self.fc = nn.Linear(_C * (_H - 1) * _W, 3, bias=False).double()
        with torch.no_grad():
            self.fc.weight.copy_(torch.from_numpy(
                _picking_weights(_C * (_H - 1) * _W, seed=63)))

    def forward(self, x):
        h = self.drop(x)
        h = F.dropout2d(h, 0.5, self.training)
        h = F.dropout(h, p=0.25, training=False)
        h = torch.feature_dropout(h, 0.5, False)
        h = torch.feature_alpha_dropout(h, 0.5, False)
        h = h.clone().contiguous().double()
        return self.fc(self.pool(h).flatten(1)).float()


_CHAIN_CELLS = (("Star", "exact"), ("Star", "approx"), ("Box", "approx"),
                ("Zono", "approx"), ("Hexatope", "approx"),
                ("Octatope", "approx"))


@pytest.mark.parametrize("kind,method", _CHAIN_CELLS)
def test_value_keeping_functions_and_methods_in_a_model_hold_the_image(
        kind, method):
    """A torch model ``nn.Identity``, fc1, ReLU (on Star and Box),
    ``nn.Dropout``, every evaluation-mode dropout function and value-keeping
    tensor method, fc2, through
    ``reach_pytorch_model``, on a set of bounds beyond plus and minus 1 cut
    by a binding row (a Zono with more generators than entries). The result
    must hold the exact image (and, for the exact method, have no point
    outside it) and every torch float64 output. Behind the exact ReLU split
    the identity nodes receive several Stars. Catches an identity node that
    hands on only one set of its list or another node's sets, a value-keeping
    spelling that is refused or mis-read (a keyword such as ``non_blocking``,
    ``copy`` or ``memory_format`` taken for a dtype, a positional flag of a
    ``torch.*`` builtin not read), and a dropout function answered as
    anything but the identity."""
    relu = kind in ("Star", "Box")
    model = _FlatChain(relu).eval()
    sets, pieces, points = _wide_flat_family(kind)
    what = f"fc1, dropout functions and tensor methods, fc2 on {kind} ({method})"
    result = reach_model(model, sets, method, what)
    hidden = R.affine(pieces, *_affine_of(model.fc1))
    exact = R.affine(R.relu(hidden) if relu else hidden,
                     *_affine_of(model.fc2))
    _assert_holds_exact_image(result, exact, what, exact=(method == "exact"))
    assert_covered(result, replay_torch(model, _thin(points, _N_REPLAY),
                                        (1, _N_IN)), what)


@pytest.mark.parametrize("kind,method", _IMAGE_KIND_METHODS)
def test_value_keeping_functions_and_methods_on_an_image_set(kind, method):
    """The image spellings (an ``nn.Dropout2d`` module, ``F.dropout2d``,
    ``torch.feature_dropout``,
    ``torch.feature_alpha_dropout`` in evaluation, copies) on a (2, 3, 5)
    image set, then a pooling along the height that reads the stored layout,
    flatten and Linear, through ``reach_pytorch_model``, against the exact
    image and torch float64. Catches an identity node that returns an image
    set flattened, with its height and width exchanged, or another node's
    sets, and an image dropout spelling that is refused."""
    model = _ImageChain().eval()
    sets, pieces, points = _image_family(kind, "row")
    what = f"image dropout functions and methods, pool, Linear on {kind} ({method})"
    result = reach_model(model, sets, method, what)
    W = model.fc.weight.detach().numpy()
    _assert_holds_exact_image(result, R.affine(pieces, (W @ _height_pool_matrix())
                                               .tolist()),
                              what, exact=(method == "exact"))
    assert_covered(result, replay_torch(model, _thin(points, _N_REPLAY),
                                        _SHAPE), what)


class _Middle(nn.Module):
    """fc1, ReLU, one spelling ``middle`` of ``h``, fc2 (float64)."""

    def __init__(self, middle):
        super().__init__()
        self.middle = middle
        self.fc1, self.fc2 = _dyadic_linear(_N_IN, 3, 64), _dyadic_linear(3, 2,
                                                                         65)

    def forward(self, x):
        h = F.relu(self.fc1(x))                                # noqa: F841
        return self.fc2(eval(self.middle))


# Spellings that change values, each by the code path that must see it: a
# random dropout by the traced keyword flag (True by default for F.dropout,
# given for the alpha variant whose default is False) and by the positional
# flag of a builtin; casts that round, by an argument-free method, a
# positional dtype, a keyword dtype and ``type``.
_VALUE_CHANGING = (
    "F.dropout(h, 0.5)",
    "F.alpha_dropout(h, 0.5, True)",
    "torch.dropout(h, 0.5, True)",
    "h.half().double()",
    "h.to(torch.int32).double()",
    "h.to(dtype=torch.float16).double()",
    "h.type(torch.bfloat16).double()",
)


@pytest.mark.parametrize("middle", _VALUE_CHANGING)
def test_value_changing_functions_and_methods_are_refused(middle):
    """A dropout function whose training flag is true (random) and a cast to
    an integer, boolean or narrower float type (which truncates or rounds)
    between two layers must be refused with ``NotImplementedError``, never
    passed through as the identity. Catches a training flag read only by
    its type, a flag looked up under the wrong keyword or position, ``half``
    added to the identity methods, and a dtype test that accepts every
    floating type."""
    model = _Middle(middle).eval()
    lo, hi = flat_box(_N_IN, seed=5)
    with pytest.raises(NotImplementedError):
        reach_model(model, flat_set("Star", lo, hi), "approx", middle)
