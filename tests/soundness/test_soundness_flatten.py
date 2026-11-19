"""Soundness cells of Flatten on every set type.

``nn.Flatten``, the ONNX Flatten node (axis 1 loads as ``nn.Flatten``, any
other axis as the converter's ``OnnxFlatten``), ``torch.flatten`` and the
tensor method ``x.flatten`` all reach the kernels of
``n2v/nn/layer_ops/flatten_reach.py`` through the dispatcher. No kernel reads
``start_dim``, ``end_dim``, ``axis`` or the method: an ImageStar becomes the
flat Star of ``ImageStar.flatten_to_star``, an ImageZono the flat Zono of
``_flatten_imagezono_chw``, both in the network's (channel, height, width)
order, and every flat set comes back as it is. Flatten only reorders
entries, so the layout is everything: an image set stores (height, width,
channel), the flattened tensor is (channel, height, width), and a result in
the stored order, a centre and generators in two different orders, or
height and width read the other way all leave the true outputs out. The
cells follow tests/README.md "A soundness cell".

Oracle. As in test_soundness_linear.py (whose builders are reused): the input
set is written here from arrays, its exact image is a union of rational
pieces, every exact vertex must be in a returned set with no slack, and for
the exact method every returned vertex must have a preimage; member points
replayed through the layer or the model in torch float64 (or on
onnxruntime) must be members by the LP oracle on the set's own rows.
Weights, biases and inputs are dyadic, so float32, float64 and the
fractions agree.

Inputs. An image of 2 channels, height 3 and width 5 with its own dyadic
interval per pixel (``pixel_box``). The exact check needs few predicate
variables, so in the "sparse" families three pixels at asymmetric places
have positive width and every other pixel is fixed at its own value (a
permuted pixel then moves a value); the "dense" families give every pixel
its interval and are checked by replay only; the "wide" family scales the
box by 8 (bounds from -16 to 20, widths 2 to 10) and adds ten generators
over three pixels each, so that it has more predicate variables
(generators) than entries. One set, three sets cut along one pixel (the
shape of an exact split) and, for an ImageStar, a binding row that is not a
box row (for an ImageZono a generator over two pixels). Flat sets: the
four-entry families of the Linear cells through ``reach_layer``, and the
sparse image box times 8 read as a flat vector of 30 entries, with
``input_shape``, through a model.

Routes. ``reach_layer``; a torch model through ``reach_pytorch_model`` and
``NeuralNetwork.reach``, with the layer written as a module, a function call
and a tensor method; an ONNX file through ``load_onnx`` with every spelling
of the axis; and three-node chains (Conv2d, Flatten, Linear on image sets;
BatchNormalization, Flatten, Gemm on flat sets). Since no kernel reads the
method, ImageStar under approx runs the code ImageStar under exact runs;
the grids over layer spellings keep ImageStar under exact (which also checks
the reverse direction) and ImageZono, and the cells of one layer keep both
methods.

Rows that refuse: ``method='exact'`` on a Zono, ImageZono, Box, Hexatope or
Octatope is refused by the model entry (``ValueError`` today).

Defects of today's library marked here (strict expected failures):

- AFFINE-6 (the row also holds the missing
  check on a Linear route): ONNX Flatten with axis 2 then Flatten(axis=1)
  then a Gemm over all 30 entries, which onnxruntime rejects, is answered.
- REACH-6, on Flatten a wrong answer and not a crash: a ``reach_layer`` list
  whose second set is an image set of another family than the first comes
  back with that image set unflattened, in its stored order.
- SHAPE-8, through a second trigger: an ONNX Flatten with axis 0 makes the
  shape pass drop every tensor shape, and a Reduce on either side then reads
  axis 1 of a flat set as every entry.

Marked elsewhere: ``Flatten(start_dim=4)`` handed to an image set (registry
AFFINE-5) in ``tests/contracts/test_layer_refusals.py``; the state bounds
that ``flatten_to_star`` passes on in the stored order (registry SET-9,
latent) in ``tests/soundness/test_soundness_image_star_sets.py``.
"""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper
from onnx2torch.node_converters.flatten import OnnxFlatten
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from n2v.nn import NeuralNetwork
from n2v.sets import ImageStar, ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests.known_bugs import DID_NOT_RAISE, known_bug
from tests.cell_support import (N_SAMPLES, assert_covered,
                                constrained_image_star, flat_set, hwc_index,
                                image_set, image_split_cover, pixel_box,
                                reach_model, reach_one_layer, replay_onnx,
                                replay_torch, samples, save_onnx)
from tests.soundness.test_soundness_linear import (
    _ROW, _assert_holds_exact_image, _box_generators, _build, _exact_input,
    _family, _full_row, _member_points)

_REFUSALS = (NotImplementedError, ValueError)

_C, _H, _W = 2, 3, 5
_N = _C * _H * _W
_SHAPE = (1, _C, _H, _W)
# (channel, row, column) of the pixels of positive width in the sparse
# families: their channel-first and stored positions all differ, and none is
# on the diagonal a height-width swap leaves in place.
_LIVE = ((0, 0, 4), (1, 2, 1), (1, 0, 3))


def _image_parts(family, seed=0):
    """``[(c, G, row)]`` in the network's (C, H, W) order for the sparse
    families: three live pixels, every other pixel fixed."""
    lo, hi = pixel_box(_C, _H, _W, seed=seed)
    mask = np.zeros((_C, _H, _W), dtype=bool)
    for pixel in _LIVE:
        mask[pixel] = True
    lo, hi = lo.reshape(-1), np.where(mask, hi, lo).reshape(-1)
    if family == "several":
        j = int(np.ravel_multi_index(_LIVE[0], (_C, _H, _W)))
        ends = lo[j] + (hi[j] - lo[j]) * np.array([0, 1, 3, 8]) / 8
        parts = []
        for a, b in zip(ends[:-1], ends[1:]):
            part_lo, part_hi = lo.copy(), hi.copy()
            part_lo[j], part_hi[j] = a, b
            parts.append(_box_generators(part_lo, part_hi) + (None,))
        return parts
    c, G = _box_generators(lo, hi)
    if family == "one":
        return [(c, G, None)]
    if family == "row":
        return [(c, G, _ROW["Star"])]
    if family == "shear":
        G[:, 0] += 0.5 * G[:, 1]               # a generator over two pixels
        return [(c, G, None)]
    raise ValueError(family)


def _image_build(kind, c, G, row=None):
    """The image set {c + G a : -1 <= a <= 1, row . a <= 1} of ``kind`` from
    (C, H, W)-ordered ``c`` and ``G``, stored in (H, W, C) order."""
    k = G.shape[1]
    order = hwc_index(_C, _H, _W)
    c_hwc, G_hwc = c[order], G[order, :]
    if kind == "ImageZono":
        if row is not None:
            raise ValueError("an ImageZono has no rows")
        return ImageZono(c_hwc.reshape(-1, 1), G_hwc, _H, _W, _C)
    row = _full_row(row, k)
    C = np.vstack([np.eye(k), -np.eye(k)]
                  + ([] if row is None else [row[None, :]]))
    return ImageStar(np.hstack([c_hwc[:, None], G_hwc]), C,
                     np.ones((C.shape[0], 1)), -np.ones((k, 1)),
                     np.ones((k, 1)), height=_H, width=_W, num_channels=_C)


def _image_family(kind, family):
    """``(sets, pieces, points)`` of a sparse image family: the n2v image
    sets, their exact union as rational pieces over the (C, H, W)-flattened
    tensor, and member points in that order."""
    if family == "row" and kind == "ImageZono":
        family = "shear"
    parts = _image_parts(family)
    levels = 3 if family == "several" else 5
    sets = [_image_build(kind, c, G, row) for c, G, row in parts]
    pieces = [p for c, G, row in parts for p in _exact_input(c, G, row)]
    points = [x for c, G, row in parts
              for x in _member_points(c, G, row, levels)]
    return sets, pieces, points


def _dense_family(kind, family):
    """``(sets, points)`` with every pixel live, points as (C, H, W) tensors:
    one set, three sets of a split, or the box cut by a binding row (for an
    ImageZono, a generator over two pixels)."""
    lo, hi = pixel_box(_C, _H, _W, seed=1)
    if family == "one":
        return [image_set(kind, lo, hi)], samples(lo, hi)
    if family == "several":
        return image_split_cover(kind, lo, hi, 3), samples(lo, hi)
    if family == "row" and kind == "ImageStar":
        star, points = constrained_image_star(lo, hi)
        return [star], points
    if family == "row":
        c, G = _box_generators(lo.reshape(-1), hi.reshape(-1))
        G[:, 0] += 0.5 * G[:, 1]
        cube = samples(-np.ones(G.shape[1]), np.ones(G.shape[1]))
        return ([_image_build(kind, c, G)],
                [(c + G @ a).reshape(_C, _H, _W) for a in cube])
    raise ValueError(family)


def _wide_family(kind):
    """``(sets, points)`` of the wide family, points as (C, H, W) tensors:
    every pixel live, the box of ``pixel_box(seed=2)`` times 8 (bounds from
    -16 to 20, widths 2 to 10, so every value lies well outside the [-1, 1]
    of a predicate variable), and ten more generators, each over three
    pixels, so that the set has 40 predicate variables (an ImageZono 40
    generators) for 30 entries: the shape a ReLU relaxation leaves, not the
    one ``from_bounds`` gives."""
    lo, hi = pixel_box(_C, _H, _W, seed=2)
    c, G = _box_generators(8 * lo.reshape(-1), 8 * hi.reshape(-1))
    rng = np.random.default_rng(7)
    extra = np.zeros((_N, 10))
    for j in range(extra.shape[1]):
        extra[rng.choice(_N, size=3, replace=False), j] = (
            rng.integers(-8, 9, size=3) / 4.0)
    G = np.hstack([G, extra])
    cube = samples(-np.ones(G.shape[1]), np.ones(G.shape[1]))
    return ([_image_build(kind, c, G)],
            [(c + G @ a).reshape(_C, _H, _W) for a in cube])


def _cube_points(c, G, row=None, n=N_SAMPLES):
    """``n`` member points of {c + G a : -1 <= a <= 1, row . a <= 1}: the
    corners and quarter-grid points of the predicate cube (``samples``),
    each one beyond the row moved along the first variable onto the cut face
    or dropped when that leaves the cube (as ``_member_points`` does). Dyadic
    when ``c``, ``G`` and the row are."""
    k = G.shape[1]
    row = _full_row(row, k)
    points = []
    for a in samples(-np.ones(k), np.ones(k), n=2 * n):
        a = a.copy()
        if row is not None and row @ a > 1:
            a[0] -= (row @ a - 1) / row[0]
            if abs(a[0]) > 1:
                continue
        points.append(c + G @ a)
    if len(points) < n:
        raise ValueError(f"only {len(points)} member points")
    return points[:n]


_IMAGE_KIND_METHODS = (("ImageStar", "exact"), ("ImageStar", "approx"),
                       ("ImageZono", "approx"))
# One (kind, method) per kernel path, for the grids over spellings of the
# layer: no kernel reads the method, so ImageStar under approx runs the code
# ImageStar under exact runs, and the exact case also checks the reverse
# direction.
_IMAGE_PATHS = (("ImageStar", "exact"), ("ImageZono", "approx"))


def _flatten_reference(layer, points):
    return replay_torch(layer, points, _SHAPE)


@pytest.mark.parametrize("family", ["one", "several", "row"])
@pytest.mark.parametrize("kind,method", _IMAGE_KIND_METHODS)
def test_flatten_of_an_image_set_is_the_channel_first_tensor(kind, method,
                                                             family):
    """``reach_layer(nn.Flatten())`` on a (2, 3, 5) image set must return
    flat sets in the network's (channel, height, width) order, holding the
    exact image. Catches a result in the stored (height, width, channel)
    order, centre and generators reordered differently, a generator attached
    to the transposed pixel, a lost constraint row, and a kernel that keeps
    only the first or only the last of several input sets."""
    layer = nn.Flatten()
    sets, pieces, points = _image_family(kind, family)
    what = f"Flatten of a sparse {kind} ({method}, {family})"
    result = reach_one_layer(layer, sets, method, what)
    assert not any(isinstance(s, (ImageStar, ImageZono)) for s in result), (
        f"{what}: the result is still an image set")
    _assert_holds_exact_image(result, pieces, what, exact=(method == "exact"))
    assert_covered(result, _flatten_reference(layer, points), what)


@pytest.mark.parametrize("family", ["one", "several", "row"])
@pytest.mark.parametrize("kind,method", [("ImageStar", "approx"),
                                         ("ImageZono", "approx")])
def test_flatten_of_a_dense_image_set_holds_every_output(kind, method, family):
    """The same Flatten with every pixel of positive width, checked by
    replay: a layout bug that only shows when every generator is live.
    ImageStar under exact runs the same code under the same check (replay
    has no reverse direction), so the dense family runs under approx."""
    layer = nn.Flatten()
    sets, points = _dense_family(kind, family)
    what = f"Flatten of a dense {kind} ({method}, {family})"
    result = reach_one_layer(layer, sets, method, what)
    assert_covered(result, _flatten_reference(layer, points), what)


@pytest.mark.parametrize("layer", [
    pytest.param(nn.Flatten(1, -1), id="start1-end-1"),
    pytest.param(nn.Flatten(start_dim=2), id="start2"),
    pytest.param(nn.Flatten(1, 2), id="start1-end2"),
    pytest.param(nn.Flatten(0), id="start0"),
    pytest.param(nn.Flatten(-3, -1), id="start-3"),
    pytest.param(nn.Flatten(-1), id="start-1"),
    pytest.param(OnnxFlatten(axis=2), id="onnx-axis2"),
])
@pytest.mark.parametrize("kind,method", _IMAGE_PATHS)
def test_flatten_with_other_axes_keeps_the_row_major_order(kind, method,
                                                          layer):
    """``start_dim``, ``end_dim`` and the ONNX ``axis`` change only the shape
    of the output tensor, never the row-major order of its entries: the flat
    result must hold the torch output read row-major. No kernel reads them,
    so every spelling (0, 1, -1, an inner axis, a negative start, the
    converter's module) must keep the promise; ``Flatten(-1)`` leaves the
    image tensor as it is. Catches a kernel or dispatcher that reads an axis
    as a layout instruction, and ``OnnxFlatten`` routed anywhere but the
    Flatten kernel."""
    sets, pieces, points = _image_family(kind, "several")
    what = f"{layer} of a sparse {kind} ({method})"
    result = reach_one_layer(layer, sets, method, what)
    _assert_holds_exact_image(result, pieces, what, exact=(method == "exact"))
    assert_covered(result, _flatten_reference(layer, points), what)


_FLAT_KIND_METHODS = (("Box", "approx"), ("Zono", "approx"),
                      ("Hexatope", "approx"), ("Octatope", "approx"),
                      ("Star", "approx"), ("Star", "exact"))


@pytest.mark.parametrize("kind,method,family", [
    pytest.param(kind, method, family, id=f"{kind}-{method}-{family}")
    for kind, method in _FLAT_KIND_METHODS
    for family in ("one", "several", "row")
    if not (kind == "Box" and family == "row")])
def test_flatten_of_a_flat_set_returns_the_same_sets(kind, method, family):
    """Flatten of a set that is already flat is the identity: every input set
    comes back, holding its points and nothing more (both directions, in
    exact arithmetic, against the set's rows). Catches a kernel that keeps
    only the first set of a list (Box, Hexatope, Octatope, Zono, Star) and
    one that drops or alters a row; the Hexatope and Octatope cells go
    through the dispatcher's own Flatten branch for those types."""
    sets, pieces, points = _family(kind, family)
    what = f"Flatten of a flat {kind} ({method}, {family})"
    result = reach_one_layer(nn.Flatten(), sets, method, what)
    _assert_holds_exact_image(result, pieces, what, exact=True)
    assert_covered(result, points, what)


# --------------------------------------------------------------------------- #
# Flatten, then a Linear layer that reads single pixels, through the models     #
# --------------------------------------------------------------------------- #
def _picking_linear(seed=5):
    """Linear(30, 4) whose rows each read a few single entries of the
    flattened tensor: every live pixel, the entry at a live pixel's stored
    position, and one fixed pixel each, so a permuted order changes every
    output. Dyadic weights and bias."""
    order = hwc_index(_C, _H, _W)
    live = [int(np.ravel_multi_index(p, (_C, _H, _W))) for p in _LIVE]
    stored = [int(np.flatnonzero(order == i)[0]) for i in live]
    weight = np.zeros((4, _N))
    weight[0, [live[0], live[1], stored[0]]] = [1.0, 0.5, -0.25]
    weight[1, [live[1], live[2], stored[1]]] = [1.0, -0.25, 0.5]
    weight[2, [live[2], stored[2], 7]] = [1.0, 0.5, 0.75]
    weight[3, :] = 0.125
    layer = nn.Linear(_N, 4)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight))
        layer.bias.copy_(torch.from_numpy(
            np.random.default_rng(seed).integers(-8, 9, size=4) / 8.0))
    return layer


class _FlattenFunction(nn.Module):
    def __init__(self, linear, how):
        super().__init__()
        self.linear, self.how = linear, how

    def forward(self, x):
        if self.how == "torch.flatten(x, 1)":
            x = torch.flatten(x, 1)
        elif self.how == "torch.flatten(x)":
            x = torch.flatten(x)
        elif self.how == "x.flatten(1)":
            x = x.flatten(1)
        elif self.how == "x.flatten(start_dim=1)":
            x = x.flatten(start_dim=1)
        return self.linear(x)


def _flatten_model(how):
    linear = _picking_linear()
    if how == "nn.Flatten":
        return nn.Sequential(nn.Flatten(), linear).eval()
    return _FlattenFunction(linear, how).eval()


def _exact_picking(pieces, linear):
    W = linear.weight.detach().double().numpy().tolist()
    b = linear.bias.detach().double().numpy().tolist()
    return R.affine(pieces, W, b)


_FORMS = ("nn.Flatten", "torch.flatten(x, 1)", "torch.flatten(x)",
          "x.flatten(1)", "x.flatten(start_dim=1)")


@pytest.mark.parametrize("how", _FORMS)
@pytest.mark.parametrize("kind,method", _IMAGE_PATHS)
def test_flatten_then_linear_in_a_model_reads_the_right_pixels(kind, method,
                                                               how):
    """Flatten written as a module, a function call or a tensor method, then
    a Linear layer that reads single pixels, through ``reach_pytorch_model``.
    Catches a flattened set in the stored order (every output reads the wrong
    pixel), a lost input set of a split, and a node loop that hands the
    Linear layer the image set instead of the flattened one."""
    model = _flatten_model(how)
    sets, pieces, points = _image_family(kind, "several")
    what = f"{how} then Linear on a sparse {kind} ({method})"
    result = reach_model(model, sets, method, what)
    _assert_holds_exact_image(result, _exact_picking(pieces, model[1] if how
                                                     == "nn.Flatten" else
                                                     model.linear), what,
                              exact=(method == "exact"))
    assert_covered(result, replay_torch(model, points, _SHAPE), what)


@pytest.mark.parametrize("family", ["row", "dense"])
@pytest.mark.parametrize("kind,method", _IMAGE_KIND_METHODS)
def test_flatten_then_linear_through_neural_network_reach(kind, method, family):
    """The same model through ``NeuralNetwork(model).reach``, with a binding
    row (a sheared generator for an ImageZono) and with every pixel live."""
    model = _flatten_model("nn.Flatten")
    network = NeuralNetwork(model)
    what = f"Flatten then Linear by NeuralNetwork.reach on {kind} ({method}, {family})"
    if family == "dense":
        sets, points = _dense_family(kind, "row")
        pieces = None
    else:
        sets, pieces, points = _image_family(kind, family)
    result = []
    for s in sets:
        result.extend(network.reach(s, method=method))
    if not result:
        pytest.fail(f"{what}: no sets returned")
    if pieces is not None:
        _assert_holds_exact_image(result, _exact_picking(pieces, model[1]),
                                  what, exact=(method == "exact"))
    assert_covered(result, replay_torch(model, points, _SHAPE), what)


def _onnx_flatten_gemm(tmp_path, axes, name):
    """ONNX graph: one Flatten node per entry of ``axes``, then Gemm with
    the picking weights (transB=1) and a bias."""
    linear = _picking_linear()
    W = linear.weight.detach().numpy().astype(np.float32)
    b = linear.bias.detach().numpy().astype(np.float32)
    nodes, prev = [], "X"
    for i, axis in enumerate(axes):
        nodes.append(helper.make_node("Flatten", [prev], [f"F{i}"], axis=axis))
        prev = f"F{i}"
    nodes.append(helper.make_node("Gemm", [prev, "W", "b"], ["Y"], transB=1))
    path = save_onnx(tmp_path, nodes, list(_SHAPE), [1, 4], {"W": W, "b": b},
                     name=name)
    return path, linear


@pytest.mark.parametrize("axes", [pytest.param((1,), id="axis1"),
                                  pytest.param((0,), id="axis0"),
                                  pytest.param((-3,), id="axis-3"),
                                  pytest.param((2, 0), id="axis2-then-axis0"),
                                  pytest.param((-1, 0), id="axis-1-then-axis0")])
@pytest.mark.parametrize("kind,method", _IMAGE_PATHS)
def test_onnx_flatten_then_gemm_reads_the_right_pixels(tmp_path, kind, method,
                                                       axes):
    """ONNX Flatten (axis 1 loads as nn.Flatten, every other axis as the
    converter's OnnxFlatten module, both routed to the Flatten kernel), then
    Gemm, loaded with ``load_onnx`` and replayed on onnxruntime. Every chain
    ends in the (1, 30) tensor the Gemm needs: ONNX Flatten(axis=k) makes the
    2-D tensor (prod(shape[:k]), prod(shape[k:])), so axis 0 and axis -3 give
    (1, 30) at once, and axis 2 or -1 give (2, 15) or (6, 5), which a
    second Flatten(axis=0) turns into (1, 30). (The converter's module for
    axis 0 raises in torch, so the shape pass records no tensor shape for
    this graph; nothing in it reads one, and onnxruntime is the reference.
    Where a node does read one, see
    ``test_onnx_flatten_between_two_reductions_refuses_or_holds_every_output``.)
    Catches a Flatten kernel or graph walk that reads the axis as a layout
    instruction (an order other than row-major) and a node loop that hands
    the Gemm the image set instead of the flattened one."""
    path, linear = _onnx_flatten_gemm(
        tmp_path, axes, "flat_" + "_".join(map(str, axes)).replace("-", "m"))
    sets, pieces, points = _image_family(kind, "several")
    what = f"ONNX Flatten{axes} then Gemm on a sparse {kind} ({method})"
    result = reach_model(load_onnx(path), sets, method, what)
    _assert_holds_exact_image(result, _exact_picking(pieces, linear), what,
                              exact=(method == "exact"))
    assert_covered(result, replay_onnx(path, points, list(_SHAPE)), what)


def _legal_partial_flatten(case):
    """Models where a Linear layer reads the last axis of a partly flattened
    tensor: torch and onnxruntime answer them (a per-channel or per-row map),
    the flat-set kernels hold no tensor shape."""
    rng = np.random.default_rng(9)
    if case == "Flatten(start_dim=2)":
        n_in, flatten = _H * _W, nn.Flatten(start_dim=2)
    elif case == "Flatten(1, 2)":
        n_in, flatten = _W, nn.Flatten(1, 2)
    else:
        raise ValueError(case)
    linear = nn.Linear(n_in, 3)
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(rng.integers(-4, 5, (3, n_in)) / 4.0))
        linear.bias.copy_(torch.from_numpy(rng.integers(-8, 9, 3) / 8.0))
    return nn.Sequential(flatten, linear).eval()


@pytest.mark.parametrize("case", ["Flatten(start_dim=2)", "Flatten(1, 2)",
                                  "onnx-axis2-gemm"])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_a_linear_layer_on_a_partly_flattened_image_refuses_or_is_sound(
        tmp_path, kind, case):
    """Flatten that keeps a leading axis, then a Linear (or Gemm) over the
    last axis: a legal network whose output is (channels, k) or (rows, k).
    The library must refuse (``NotImplementedError``, or the ``ValueError``
    a flat set of the wrong length raises today) or hold every output read
    row-major; it must never answer with the full flattened map."""
    sets, pieces, points = _image_family(kind, "one")
    what = f"{case} then Linear on {kind}"
    if case == "onnx-axis2-gemm":
        rng = np.random.default_rng(10)
        W = (rng.integers(-4, 5, (3, _H * _W)) / 4.0).astype(np.float32)
        path = save_onnx(tmp_path, [
            helper.make_node("Flatten", ["X"], ["F"], axis=2),
            helper.make_node("Gemm", ["F", "W"], ["Y"], transB=1)],
            list(_SHAPE), [_C, 3], {"W": W}, name="partial")
        model, outputs = load_onnx(path), replay_onnx(path, points, list(_SHAPE))
    else:
        model = _legal_partial_flatten(case)
        outputs = replay_torch(model, points, _SHAPE)
    try:
        result = reach_model(model, sets, "approx", what)
    except (NotImplementedError, ValueError):
        return
    assert_covered(result, outputs, what)


# --------------------------------------------------------------------------- #
# An ONNX Flatten chain that keeps a leading axis, then a full-width Gemm       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("axes", [pytest.param((2, 1), id="axis2-then-axis1")])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@known_bug(12, "ONNX Flatten with axis 2 or 3 keeps a leading axis ((2, 15) "
               "or (6, 5) for a (1, 2, 3, 5) input, and a second "
               "Flatten(axis=1) keeps it), so a following Gemm over 30 inputs "
               "fails in onnxruntime and torch; n2v keeps no tensor shape for "
               "the flattened set (the shape pass swallows the failing "
               "forward and records no shapes, and the Linear kernel never "
               "checks one), so it applies the 30-wide map to all 30 entries "
               "and answers instead of refusing (AFFINE-6; "
               "the row also holds the missing check on a "
               "Linear route)",
           raises=DID_NOT_RAISE)
def test_onnx_flatten_chain_that_onnxruntime_rejects_is_refused(tmp_path,
                                                               kind, axes):
    """ONNX Flatten(axis=2), then Flatten(axis=1), then a Gemm whose weight
    has 30 columns: the graph passes ``onnx.checker`` but the tensor reaching
    the Gemm is (2, 15), so onnxruntime raises. The library must refuse too
    (``NotImplementedError``, or the ``ValueError`` of a size mismatch), not
    answer for a network that does not run. One marked case per set family
    (the Star and the Zono kernels); axis 3, which gives (6, 5), runs the same
    code. The passing neighbours are the valid chains of
    ``test_onnx_flatten_then_gemm_reads_the_right_pixels`` (same builder,
    same replay), where the second Flatten has axis 0."""
    path, _ = _onnx_flatten_gemm(tmp_path, axes,
                                 "bad_" + "_".join(map(str, axes)))
    lo, hi = pixel_box(_C, _H, _W)
    # The setup must see onnxruntime reject the Gemm's shapes and nothing
    # else; a RuntimeError is not absorbed by the marker (pytest.fail would
    # be, since DID_NOT_RAISE is pytest's Failed).
    try:
        replay_onnx(path, [lo], list(_SHAPE))
    except InvalidArgument as exc:
        if "GEMM" not in str(exc).upper():
            raise RuntimeError(f"setup: onnxruntime rejected the graph for "
                               f"another reason: {exc}") from exc
    else:
        raise RuntimeError("onnxruntime ran the graph; the cell needs a "
                           "graph it rejects")
    with pytest.raises((NotImplementedError, ValueError)):
        reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                    f"ONNX Flatten{axes} then a 30-wide Gemm on {kind}")


# --------------------------------------------------------------------------- #
# The rows that refuse: exact on a set type with no exact method              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Zono", "ImageZono", "Box", "Hexatope",
                                  "Octatope"])
def test_exact_flatten_model_on_a_set_with_no_exact_method_is_refused(kind):
    """The rows (exact, Zono / ImageZono / Box / Hexatope / Octatope) of
    Flatten: the model entry refuses ``method='exact'`` on these set types
    before any layer (today ``ValueError``, '... only supports approx'; the
    planned ``UnsupportedError`` subclasses ``NotImplementedError``), on the
    image set and on a flat set given with ``input_shape``. Catches the model
    entry passing 'exact' through to a lane that never reads the method and
    answering under the name exact."""
    model = _flatten_model("nn.Flatten")
    lo, hi = pixel_box(_C, _H, _W)
    if kind == "ImageZono":
        input_set, options = image_set(kind, lo, hi), {}
    else:
        input_set = flat_set(kind, lo, hi)
        options = {"input_shape": (_C, _H, _W)}
    with pytest.raises(_REFUSALS):
        reach_model(model, input_set, "exact",
                    f"exact Flatten then Linear on {kind}", **options)


# --------------------------------------------------------------------------- #
# Conv2d, then Flatten, then Linear: the three-node chain on image sets       #
# --------------------------------------------------------------------------- #
def _conv_flatten_linear(seed=11):
    """Conv2d(2, 3, kernel (2, 1)), whose (3, 2, 5) output has as many
    entries as the input but another channel count and height, then
    Flatten, then Linear(30, 4); dyadic weights and biases."""
    rng = np.random.default_rng(seed)
    conv, linear = nn.Conv2d(_C, 3, (2, 1)), nn.Linear(_N, 4)
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(
            rng.integers(-4, 5, size=tuple(conv.weight.shape)) / 4.0))
        conv.bias.copy_(torch.from_numpy(rng.integers(-8, 9, size=3) / 8.0))
        linear.weight.copy_(torch.from_numpy(
            rng.integers(-4, 5, size=(4, _N)) / 4.0))
        linear.bias.copy_(torch.from_numpy(rng.integers(-8, 9, size=4) / 8.0))
    return nn.Sequential(conv, nn.Flatten(), linear).eval()


def _exact_conv_flatten_linear(pieces, model):
    """The exact image of ``model`` on ``pieces`` over the (C, H, W)-flattened
    input: the convolution as a matrix read off torch in float64 on unit
    tensors (exact: dyadic weights), Flatten as the identity on the
    channel-first order, then the Linear layer."""
    conv = copy.deepcopy(model[0]).double()
    with torch.no_grad():
        bias = conv(torch.zeros(_SHAPE, dtype=torch.float64)).reshape(-1)
        M = torch.stack([conv(torch.eye(_N, dtype=torch.float64)[j]
                              .reshape(_SHAPE)).reshape(-1) - bias
                         for j in range(_N)], dim=1)
    pieces = R.affine(pieces, M.numpy().tolist(), bias.numpy().tolist())
    return _exact_picking(pieces, model[2])


def _conv_flatten_gemm_onnx(tmp_path, model):
    """The same layers as an ONNX file: Conv, Flatten(axis=1), Gemm."""
    f32 = lambda t: t.detach().numpy().astype(np.float32)     # noqa: E731
    conv, linear = model[0], model[2]
    return save_onnx(tmp_path, [
        helper.make_node("Conv", ["X", "K", "kb"], ["A"], kernel_shape=[2, 1]),
        helper.make_node("Flatten", ["A"], ["F"], axis=1),
        helper.make_node("Gemm", ["F", "W", "b"], ["Y"], transB=1)],
        list(_SHAPE), [1, 4], {"K": f32(conv.weight), "kb": f32(conv.bias),
                               "W": f32(linear.weight), "b": f32(linear.bias)},
        name="conv_flatten_gemm")


@pytest.mark.parametrize("route", ["torch-split", "onnx-wide"])
@pytest.mark.parametrize("kind,method", _IMAGE_KIND_METHODS)
def test_conv_then_flatten_then_linear_holds_every_output(tmp_path, kind,
                                                          method, route):
    """The three-node chain Conv2d, Flatten, Linear through the model entry:
    the Flatten reads an image set that a Conv2d made (three channels, height
    2, generators over several pixels) and the Linear reads the rows the
    Flatten wrote. "torch-split": a torch model on the three sets of a split
    of the sparse family, checked against the exact image and torch float64.
    "onnx-wide": the same layers as an ONNX file on the wide family (inputs
    from -16 to 20, outputs to 74, more predicate variables or generators
    than entries), checked against onnxruntime. Catches, behind a
    convolution, a flattened set in the stored order, with centre and
    generators in different orders, with height and width read the other
    way or with swapped predicate bounds, and an ImageZono left
    unflattened."""
    model = _conv_flatten_linear()
    what = f"Conv2d, Flatten, Linear on {kind} ({method}, {route})"
    if route == "torch-split":
        sets, pieces, points = _image_family(kind, "several")
        result = reach_model(model, sets, method, what)
        _assert_holds_exact_image(result, _exact_conv_flatten_linear(
            pieces, model), what, exact=(method == "exact"))
        assert_covered(result, replay_torch(model, points, _SHAPE), what)
    else:
        path = _conv_flatten_gemm_onnx(tmp_path, model)
        sets, points = _wide_family(kind)
        result = reach_model(load_onnx(path), sets, method, what)
        assert_covered(result, replay_onnx(path, points, list(_SHAPE)), what)


# --------------------------------------------------------------------------- #
# A flat set of an image through BatchNormalization, Flatten, Gemm            #
# --------------------------------------------------------------------------- #
def _bn_flatten_gemm(tmp_path, seed=13):
    """``(path, affine)``: an ONNX graph X (1, 2, 3, 5), BatchNormalization
    (one scale and shift per channel, epsilon 0 and variances 4 and 1, so the
    scale is exact), Flatten(axis=1), Gemm(30 -> 4) with a bias, all
    parameters dyadic float32; ``affine(pieces)`` is its exact image over the
    (C, H, W)-flattened input."""
    rng = np.random.default_rng(seed)
    gamma, beta = np.array([0.5, -2.0]), np.array([0.25, 1.0])
    mean, var = np.array([0.5, -1.0]), np.array([4.0, 1.0])
    W = rng.integers(-4, 5, size=(4, _N)) / 4.0
    b = rng.integers(-8, 9, size=4) / 8.0
    f32 = lambda a: np.asarray(a, dtype=np.float32)            # noqa: E731
    path = save_onnx(tmp_path, [
        helper.make_node("BatchNormalization", ["X", "g", "be", "m", "v"],
                         ["A"], epsilon=0.0),
        helper.make_node("Flatten", ["A"], ["F"], axis=1),
        helper.make_node("Gemm", ["F", "W", "b"], ["Y"], transB=1)],
        list(_SHAPE), [1, 4], {"g": f32(gamma), "be": f32(beta),
                               "m": f32(mean), "v": f32(var), "W": f32(W),
                               "b": f32(b)}, name="bn_flatten_gemm")
    scale = np.repeat(gamma / np.sqrt(var), _H * _W)
    shift = np.repeat(beta - mean * gamma / np.sqrt(var), _H * _W)

    def affine(pieces):
        pieces = R.affine(pieces, np.diag(scale).tolist(), shift.tolist())
        return R.affine(pieces, W.tolist(), b.tolist())
    return path, affine


@pytest.mark.parametrize("kind,method", _FLAT_KIND_METHODS)
def test_flatten_of_a_flat_set_in_a_model_holds_every_output(tmp_path, kind,
                                                             method):
    """The flat rows in ordinary use: a flat set that stands for the
    (1, 2, 3, 5) input in the tensor's own order, given with
    ``input_shape`` (the competition route), through BatchNormalization,
    Flatten and Gemm loaded from ONNX, so the Gemm reads the rows the Flatten
    passed on. The set is the sparse image box times 8 (values from -16 to
    14, live widths 2 to 10), with the binding row of its type where it has
    rows. Checked against the exact image and onnxruntime. Catches a model
    entry or a Flatten that reads a flat set as an image in the stored
    (height, width, channel) order, and a Flatten that alters a row of a
    flat set on the model route."""
    path, affine = _bn_flatten_gemm(tmp_path)
    (c, G, _), = _image_parts("one")
    c, G = 8 * c, 8 * G
    row = _ROW.get(kind)
    what = f"BatchNormalization, Flatten, Gemm on a flat {kind} ({method})"
    result = reach_model(load_onnx(path), _build(kind, c, G, row), method,
                         what, input_shape=(_C, _H, _W))
    _assert_holds_exact_image(result, affine(_exact_input(c, G, row)), what,
                              exact=(method == "exact"))
    assert_covered(result, replay_onnx(path, _cube_points(c, G, row),
                                       list(_SHAPE)), what)


# --------------------------------------------------------------------------- #
# A list of two set families handed to reach_layer                            #
# --------------------------------------------------------------------------- #
_MIXED_FAMILIES = known_bug(
    7, "reach_layer chooses the Flatten kernel by the first set of the list "
       "and checks no other: an ImageZono after an ImageStar or a Star, an "
       "ImageStar after a Zono or an ImageZono, and any image set after a "
       "Box, Hexatope or Octatope comes back unflattened, in its stored "
       "(height, width, channel) order, and the Linear after it reads every "
       "entry at the wrong pixel (registry REACH-6, the dispatcher defect that "
       "crashes other kernels; on Flatten it is a wrong answer)")


@pytest.mark.parametrize("first", [
    pytest.param("ImageStar", marks=_MIXED_FAMILIES,
                 id="imagestar-then-imagezono"),
    pytest.param("Zono", id="zono-then-imagezono")])
def test_flatten_of_a_list_mixing_set_families_refuses_or_holds_every_output(
        first):
    """``reach_layer(nn.Flatten())`` on a list of two sets of different
    families whose second set is an ImageZono, then the picking Linear on
    the result through ``reach_layer``: the library must refuse
    (``NotImplementedError`` or ``ValueError``) or hold the true outputs of
    both inputs, the Linear's result those of Flatten then Linear and the
    Flatten's result those of Flatten, read row-major. Only a direct call
    builds such a list. The passing neighbour leads with a Zono, whose lane
    flattens the same ImageZono."""
    lo, hi = pixel_box(_C, _H, _W, seed=0)
    lo2, hi2 = pixel_box(_C, _H, _W, seed=3)
    lo2, hi2 = lo2 + 3.0, hi2 + 3.0
    lead = (image_set(first, lo, hi) if first.startswith("Image")
            else flat_set(first, lo, hi))
    sets = [lead, image_set("ImageZono", lo2, hi2)]
    points = samples(lo, hi, 64) + samples(lo2, hi2, 64, seed=1)
    linear = _picking_linear()
    what = f"Flatten of [{first}, ImageZono]"
    try:
        flat = reach_one_layer(nn.Flatten(), sets, "approx", what)
        out = reach_one_layer(linear, flat, "approx", what + " then Linear")
    except _REFUSALS:
        return
    assert_covered(out, replay_torch(nn.Sequential(nn.Flatten(), linear),
                                     points, _SHAPE), what + " then Linear")
    assert_covered(flat, replay_torch(nn.Flatten(), points, _SHAPE), what)


# --------------------------------------------------------------------------- #
# ONNX Flatten with axis 0 between two reductions                             #
# --------------------------------------------------------------------------- #
_SHAPE_RECORD_DROPPED = known_bug(
    12, "ONNX Flatten with axis 0 loads as the converter's OnnxFlatten, whose "
        "forward raises in torch on a rank-4 input; the shape pass then "
        "records no tensor shape for any node of the graph, and the "
        "ReduceMean and ReduceSum around the Flatten read axis 1 of the flat "
        "set as every entry, so the reach answers the mean of all 30 entries "
        "where the network gives half their sum (registry SHAPE-8, the dropped "
        "shape record, here triggered by the Flatten converter)")


@pytest.mark.parametrize("axis", [
    pytest.param(0, marks=_SHAPE_RECORD_DROPPED, id="axis0"),
    pytest.param(1, id="axis1")])
def test_onnx_flatten_between_two_reductions_refuses_or_holds_every_output(
        tmp_path, axis):
    """ReduceMean over the channel axis (keepdims), ONNX Flatten(axis), then
    ReduceSum over axis 1: legal for both axes (the Flatten gives (1, 15)),
    the output is half the sum of the 30 input entries. Reached on a flat
    Star with ``input_shape``, the competition route; the library must
    refuse or hold every onnxruntime output (corners of the box included).
    Catches a Flatten conversion whose torch forward fails and so takes the
    tensor shapes away from the shape-reading nodes around it. The axis 1
    case is the passing neighbour (same graph, same replay)."""
    path = save_onnx(tmp_path, [
        helper.make_node("ReduceMean", ["X"], ["R"], axes=[1], keepdims=1),
        helper.make_node("Flatten", ["R"], ["F"], axis=axis),
        helper.make_node("ReduceSum", ["F", "ax"], ["Y"], keepdims=0)],
        list(_SHAPE), [1], {"ax": np.array([1], dtype=np.int64)},
        name=f"reduce_flatten_{axis}_reduce")
    lo, hi = pixel_box(_C, _H, _W)
    points = samples(lo, hi)
    what = f"ReduceMean, Flatten(axis={axis}), ReduceSum on a flat Star"
    try:
        result = reach_model(load_onnx(path), flat_set("Star", lo, hi),
                             "approx", what, input_shape=(_C, _H, _W))
    except _REFUSALS:
        return
    assert_covered(result, replay_onnx(path, points, list(_SHAPE)), what)
