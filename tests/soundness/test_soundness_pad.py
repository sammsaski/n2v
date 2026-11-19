"""
Soundness cells for Pad reachability (``n2v/nn/layer_ops/pad_reach.py``, the
Pad branches of the dispatcher and the ONNX Pad handling of the graph walk in
``n2v/nn/reach.py``).

Rows: (method, set type) for the seven set types and the two methods. The
kernels read no option and run the same code for both methods. ImageStar
(exact and approx) and ImageZono (approx) answer: zero padding of the two
spatial axes is copied into the basis, so the result is the exact image. A
pad with a non-zero constant, another mode, a negative amount or a pad of the
channel or batch axis is refused with ``NotImplementedError``. A flat Star or
Zono crashes with ``TypeError`` (marked); Box, Hexatope and Octatope have no
rule in the dispatcher and are refused with ``NotImplementedError``. The model
entry refuses ``method='exact'`` on Zono, ImageZono, Box, Hexatope and
Octatope with ``ValueError`` before the walk (held by
``tests/soundness/test_soundness_graph_walk.py``).

Oracles: torch in float64, or onnxruntime for an ONNX Pad, on member points
(corners included) with membership decided by one LP on the returned set's
own rows (``tests/_oracles/membership.py``); and the exact image in rationals
(``tests/_oracles/rational.py``) from the definition of the pad below, which
reads torch's documented (left, right, top, bottom) order and shares no code
with n2v. Layout: a tensor is (C, H, W); an image set stores (H, W, C); every
comparison converts explicitly (``out_chw``, ``hwc_index``). Every image has
two channels and height different from width, and the four pad amounts
differ. The exact-image helpers live in ``test_soundness_conv2d.py``.
"""

import numpy as np
import pytest
import torch.nn as nn
from onnx import helper
from onnx2torch.node_converters.pad import OnnxPadDynamic, OnnxPadStatic

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import ImageStar, ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests.cell_support import (
    assert_covered, constrained_image_star, flat_box, flat_set, hwc_index,
    image_set, pixel_box, reach_model, reach_one_layer, replay_onnx,
    replay_torch, samples, save_onnx)
from tests.known_bugs import known_bug
from tests.soundness.test_soundness_conv2d import (
    _assert_exact_image, _chain, _distinct, _few_var_inputs, _inputs,
    _out_chw, _piece, _two_straddlers)

# id: ((C, H, W) of the input, a function that builds the layer). torch's
# 4-tuple is (left, right, top, bottom). In "same-sums" left + right equals
# top + bottom, so a pad read with the two axes exchanged keeps the output
# shape and only the placement shows it; in "tuple" every amount differs
# and the sums too, and the layer is a ConstantPad2d with the constant 0
# (ZeroPad2d is its subclass). torch stores an int padding as the same
# 4-tuple, so ZeroPad2d(2) runs the tuple code with equal amounts.
_PAD = {
    "same-sums": ((2, 3, 5), lambda: nn.ZeroPad2d((0, 3, 2, 1))),
    "tuple": ((2, 4, 3), lambda: nn.ConstantPad2d((1, 0, 3, 2), 0.0)),
}


def _make_pad(cfg):
    chw, build = _PAD[cfg]
    return build(), chw


def _exact_pad(chw, lrtb, value=0):
    """``f(x) -> (y, (C, Ho, Wo))``: the pad by its definition on a flat
    (C, H, W) list of Fractions: entry (c, h, w) goes to (c, h + top,
    w + left) of a frame filled with ``value``."""
    C, H, W = chw
    left, right, top, bottom = lrtb
    Ho, Wo = H + top + bottom, W + left + right
    fill = R.frac(value)

    def f(x):
        y = [fill] * (C * Ho * Wo)
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    y[(c * Ho + h + top) * Wo + w + left] = x[(c * H + h) * W + w]
        return y, (C, Ho, Wo)
    return f


def _pad_cells():
    out = []
    for cfg in _PAD:
        for v in [("ImageStar", "exact", "split"),
                  ("ImageZono", "approx", "split"),
                  ("ImageStar", "approx", "row")]:
            out.append(pytest.param(cfg, *v, id=f"{cfg}-{v[0]}-{v[1]}-{v[2]}"))
    return out


@pytest.mark.parametrize("cfg,kind,method,variant", _pad_cells())
def test_pad_layer_holds_every_torch_output(cfg, kind, method, variant):
    """ZeroPad2d and ConstantPad2d(value 0) through ``reach_layer`` on three
    sets at once or on an ImageStar with a binding row, against torch in
    float64, membership by LP on the rows. Catches a result that keeps only
    the first or only the last of several input sets, the torch 4-tuple read
    as (top, bottom, left, right), top and bottom (or left and right)
    exchanged, padding that copies the border instead of zeros, the padded
    generators flattened in another order than the centre, and an image set
    stored with its height and width exchanged."""
    layer, chw = _make_pad(cfg)
    sets, points = _inputs(kind, variant, chw)
    what = f"{layer} on {kind} ({variant}, {method})"
    result = reach_one_layer(layer, sets, method, what)
    assert_covered(result, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=_out_chw(layer, chw))


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageStar", "approx"),
                                         ("ImageZono", "approx")])
@pytest.mark.parametrize("cfg", list(_PAD))
def test_pad_layer_is_the_exact_image_in_rationals(cfg, kind, method):
    """Two input sets at once on three predicate variables each (the
    ImageStar cut by a binding row), centre and generators dense over every
    pixel with 30 significant bits (not representable in float32), the
    second star on an asymmetric predicate box. The exact image is the pad's
    definition in rationals; every vertex of it must be in one returned set
    and, for the exact method, every vertex of a returned set must have a
    preimage. Catches a generator placed at the wrong offset even when
    sampled points are inside, a lost input set, predicate bounds or rows
    not passed on as they are (rebuilt as the unit box, or the first set's
    given to every set), and a copy made in float32."""
    layer, chw = _make_pad(cfg)
    f = _exact_pad(chw, tuple(layer.padding))
    sets, pieces = [], []
    # 30 significant bits: a float32 round trip of the copied entries
    # moves them, so the exact check sees it
    for star, zono, c, G, star_rows, zono_rows in _few_var_inputs(
            chw, seed=3, bits=30):
        sets.append(star if kind == "ImageStar" else zono)
        piece, out_chw = _piece(f, c, G, star_rows if kind == "ImageStar"
                                else zono_rows)
        pieces.append(piece)
    what = f"{layer} on {kind} ({method}), exact image"
    result = reach_one_layer(layer, sets, method, what)
    # The reverse check enumerates the vertices of every returned set; a
    # result on many more variables than the three given would not finish.
    many = [getattr(s, "nVar", s.V.shape[-1]) for s in result
            if getattr(s, "nVar", s.V.shape[-1]) > 6]
    if many:
        pytest.fail(f"{what}: returned sets on {many} predicate variables, "
                    f"the exact check enumerates at most 6")
    _assert_exact_image(pieces, result, what, exact=(method == "exact"),
                        out_chw=out_chw)


def _wide_set(kind, chw, seed=11, extra=10):
    """``(set, points)``: an image set on more predicate variables than
    entries (``extra`` more: the shape a ReLU relaxation leaves), centre in
    [-8, 8] and generators in [-2, 2] (multiples of 1/8), so every entry
    ranges far beyond [-1, 1] over an interval much wider than 2. The
    ImageStar's predicate box is [-1, 3] on every other variable and
    [-1, 1] on the rest, cut by one binding row over three variables; the
    ImageZono's is the cube [-1, 1]. ``points`` are at least 128 member
    points in (C, H, W): predicate corners and quarter-grid points that keep
    the row, mapped through the basis without n2v (exact, dyadic)."""
    C, H, W = chw
    n = C * H * W
    m = n + extra
    rng = np.random.default_rng(seed)
    c = rng.integers(-64, 65, n) / 8.0
    G = rng.integers(-16, 17, (n, m)) / 8.0
    order = hwc_index(C, H, W)
    if kind == "ImageZono":
        lb, ub = -np.ones(m), np.ones(m)
        A, b = np.vstack([np.eye(m), -np.eye(m)]), np.concatenate([ub, -lb])
        s = ImageZono(c[order].reshape(-1, 1), G[order].copy(), H, W, C)
    else:
        lb = -np.ones(m)
        ub = np.where(np.arange(m) % 2 == 0, 3.0, 1.0)
        row = np.zeros(m)
        row[[0, 5, m - 1]] = [1.0, -0.5, 1.0]
        rhs = float(np.sum(np.maximum(row * lb, row * ub))) - 1.0
        A = np.vstack([np.eye(m), -np.eye(m), row])
        b = np.concatenate([ub, -lb, [rhs]])
        V = np.hstack([c[order][:, None], G[order]])
        s = ImageStar(V, A, b.reshape(-1, 1), lb.reshape(-1, 1),
                      ub.reshape(-1, 1), height=H, width=W, num_channels=C)
    points = []
    for attempt in range(100000):
        if len(points) >= 128:
            break
        if attempt % 2:
            a = np.where(rng.integers(0, 2, m) == 1, ub, lb)
        else:
            a = lb + (ub - lb) * rng.integers(0, 5, m) / 4.0
        if np.all(A @ a <= b):
            points.append((c + G @ a).reshape(C, H, W))
    if len(points) < 128:
        pytest.fail("could not draw 128 member points")
    return s, points


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageStar", "approx"),
                                         ("ImageZono", "approx")])
def test_pad_of_a_wide_set_with_more_variables_than_entries_holds_every_torch_output(
        kind, method):
    """ZeroPad2d((0, 3, 2, 1)) on a (2, 3, 5) ImageStar with 40 predicate
    variables (a binding row, a predicate box that is not [-1, 1]) or an
    ImageZono with 40 generators, entries far beyond [-1, 1]: the input
    shape that a ReLU relaxation leaves, which ``from_bounds`` never builds.
    torch in float64 on 128 member points, membership by LP on the rows.
    Catches the padded generators flattened in another order than the
    centre or padded on exchanged axes, and predicate bounds rebuilt as
    [-1, 1], which a unit predicate box and data inside [-1, 1] hide."""
    layer = nn.ZeroPad2d((0, 3, 2, 1))
    chw = (2, 3, 5)
    start, points = _wide_set(kind, chw)
    what = f"{layer} on a {kind} with more variables than entries ({method})"
    result = reach_one_layer(layer, [start], method, what)
    assert_covered(result, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=_out_chw(layer, chw))


@pytest.mark.parametrize("cfg,kind,method,variant", [
    ("same-sums", "ImageStar", "exact", "split"),
    ("tuple", "ImageZono", "approx", "split"),
    ("same-sums", "ImageStar", "approx", "row"),
])
def test_pad_inside_a_model_holds_every_torch_output(cfg, kind, method,
                                                     variant):
    """Pad between a channel-mixing 1x1 convolution and Flatten > Linear,
    through ``reach_pytorch_model``. Catches padded rows or a layout that
    only the downstream affine map and the Flatten conversion expose."""
    op, chw = _make_pad(cfg)
    model = _chain(op, chw)
    sets, points = _inputs(kind, variant, chw)
    what = f"conv > {op} > flatten > linear on {kind} ({method})"
    result = reach_model(model, sets, method, what)
    assert_covered(result, replay_torch(model, points, (1,) + chw), what)


def test_pad_after_an_exact_relu_split_holds_every_torch_output():
    """ReLU > ZeroPad2d > Flatten > Linear with the exact method on a box
    with two pixels that straddle zero: the pad receives the four sets of the
    exact split together. Catches a pad that returns the image of only the
    first or only the last of several input sets."""
    op, chw = _make_pad("same-sums")
    model = _chain(op, chw, head=nn.ReLU())
    lo, hi = _two_straddlers(chw)
    what = f"relu > {op} > flatten > linear, exact"
    result = reach_model(model, image_set("ImageStar", lo, hi), "exact", what)
    assert_covered(result, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what)


# --------------------------------------------------------------------------- #
# Pad layers n2v does not reach as zero padding: refused, or sound            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("build", [
    pytest.param(lambda: nn.ConstantPad2d((1, 0, 2, 1), 0.5),
                 id="constant-0.5"),
    pytest.param(lambda: nn.ConstantPad2d((0, 2, 1, 0), -2.0),
                 id="constant-minus-2"),
    pytest.param(lambda: nn.ReflectionPad2d((1, 2, 2, 0)), id="reflect"),
    pytest.param(lambda: nn.ReplicationPad2d((0, 2, 1, 3)), id="replicate"),
    pytest.param(lambda: nn.CircularPad2d((1, 2, 0, 1)), id="circular"),
    pytest.param(lambda: nn.ZeroPad2d((-1, 0, 1, 2)), id="negative"),
    # the module the converter builds for a Pad of opset 2 to 10 (and, with
    # -inf, in front of an asymmetrically padded MaxPool), built directly
    # because the converter loses a non-zero constant (marked below)
    pytest.param(lambda: OnnxPadStatic(pads=[0, 3, 2, 1], mode="constant",
                                       constant_value=0.5),
                 id="onnx-static-constant-0.5"),
])
def test_pad_that_is_not_zero_padding_refuses_or_is_sound(build, kind):
    """A pad with a non-zero constant, a border-copying mode, or a negative
    amount (torch crops): the reach must refuse (NotImplementedError) or
    answer with sets that hold every torch output. Catches a constant value
    or a mode that is not passed to the zero-padding check (of a torch
    ConstantPad2d or of an OnnxPadStatic), so the layer is reached as zero
    padding, and a negative amount read as 0."""
    layer = build()
    chw = (2, 3, 5)
    lo, hi = pixel_box(*chw, seed=1)
    what = f"{layer} on {kind}"
    try:
        result = reach_one_layer(layer, [image_set(kind, lo, hi)], "approx",
                                 what)
    except NotImplementedError:
        return
    assert_covered(result, replay_torch(layer, samples(lo, hi), (1,) + chw),
                   what, out_chw=_out_chw(layer, chw))


# --------------------------------------------------------------------------- #
# ONNX Pad: OnnxPadDynamic (opset 11 and 13) and OnnxPadStatic (opset 10)       #
# --------------------------------------------------------------------------- #
# ONNX order of a pads vector, all begins then all ends:
# [batch, channel, top, left, batch, channel, bottom, right].
# top 2, left 0, bottom 1, right 3: all four differ and top + bottom equals
# left + right, so heights and widths read exchanged keep the output shape.
_ONNX_PADS = [0, 0, 2, 0, 0, 0, 1, 3]


def _onnx_pad_chain(tmp_path, name, opset, pads, mode=None, value=None,
                    value_from_input=False, chw=(2, 3, 5)):
    """X -> Conv 1x1 -> Pad -> Conv 1x1 -> Y with dyadic float32 weights.
    From opset 11 the pads (and the value, when given) are inputs, before
    that attributes. ``mode=None`` leaves the mode attribute out (its
    default is 'constant'); ``value_from_input`` gives the constant as the
    scalar ReduceMean(X), which depends on the input. Returns ``(path, chw,
    out_chw)``; the output shape is the definition of the pad."""
    rng = np.random.default_rng(7)
    C, H, W = chw
    C2 = C + pads[1] + pads[5]
    out_chw = (2, H + pads[2] + pads[6], W + pads[3] + pads[7])
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    inits = {"a_w": f32(_distinct(rng, (C, C, 1, 1))),
             "a_b": f32(_distinct(rng, (C,))),
             "z_w": f32(_distinct(rng, (2, C2, 1, 1))),
             "z_b": f32(_distinct(rng, (2,)))}
    attrs = {} if mode is None else {"mode": mode}
    nodes = []
    if opset >= 11:
        inputs = ["T0", "pads"]
        inits["pads"] = np.asarray(pads, dtype=np.int64)
        if value is not None:
            inputs.append("value")
            inits["value"] = f32(value)
        elif value_from_input:
            inputs.append("value")
            nodes.append(helper.make_node("ReduceMean", ["X"], ["value"],
                                          keepdims=0))
        pad = helper.make_node("Pad", inputs, ["T1"], **attrs)
    else:
        if value is not None:
            attrs["value"] = float(value)
        pad = helper.make_node("Pad", ["T0"], ["T1"], pads=pads, **attrs)
    nodes += [helper.make_node("Conv", ["X", "a_w", "a_b"], ["T0"]), pad,
              helper.make_node("Conv", ["T1", "z_w", "z_b"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, C, H, W], [1, *out_chw], inits,
                     name=name, opset=opset)
    return path, chw, out_chw


def _onnx_inputs(kind, method, chw, seed=8):
    lo, hi = pixel_box(*chw, seed=seed)
    if kind == "ImageStar" and method == "approx":
        star, points = constrained_image_star(lo, hi, seed=seed)
        return star, points
    return image_set(kind, lo, hi), samples(lo, hi)


@pytest.mark.parametrize("opset,value,kind,method", [
    # OnnxPadDynamic, the constant input left out
    (13, None, "ImageStar", "exact"),
    (11, None, "ImageZono", "approx"),
    # OnnxPadDynamic, the constant 0 given as an input and checked
    (13, 0.0, "ImageStar", "exact"),
    (13, 0.0, "ImageStar", "approx"),
    (13, 0.0, "ImageZono", "approx"),
    # OnnxPadStatic, the attribute 'value' left out, or given as 0
    (10, None, "ImageZono", "approx"),
    (10, 0.0, "ImageStar", "exact"),
    (10, 0.0, "ImageZono", "approx"),
])
def test_onnx_pad_holds_every_onnxruntime_output(tmp_path, opset, value, kind,
                                                 method):
    """An ONNX Pad with top 2, left 0, bottom 1, right 3 and its mode
    attribute left out (default 'constant') between two 1x1 Convs, loaded
    with ``load_onnx`` (OnnxPadDynamic from opset 11, OnnxPadStatic before),
    replayed on onnxruntime. Catches the resolved pads read with the height
    and width entries exchanged or in the wrong order, an image placed at
    the wrong offset in the padded frame, a zero constant refused, and a
    padded ImageZono stored with its height and width exchanged (the Conv
    after the pad reads them)."""
    path, chw, out_chw = _onnx_pad_chain(
        tmp_path, f"pad{opset}", opset, _ONNX_PADS, value=value)
    start, points = _onnx_inputs(kind, method, chw)
    what = f"ONNX Pad opset {opset} value {value} on {kind} ({method})"
    result = reach_model(load_onnx(path), start, method, what)
    assert_covered(result, replay_onnx(path, points, [1, *chw]), what,
                   out_chw=out_chw)


_PAD_VALUE_ATTRIBUTE_LOST = known_bug(
    12, "an ONNX Pad of opset 2 to 10 keeps its constant in the attribute "
        "'value', which the onnx2torch converter does not read (it reads "
        "'constant_value'), so the loaded module pads with 0 and n2v reaches "
        "zero padding: every output at a padded position lies outside")


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("case", [
    pytest.param(dict(opset=13, mode="edge"), id="opset13-edge"),
    pytest.param(dict(opset=13, mode="reflect"), id="opset13-reflect"),
    pytest.param(dict(opset=10, mode="edge"), id="opset10-edge"),
    pytest.param(dict(opset=10, mode="reflect"), id="opset10-reflect"),
    pytest.param(dict(opset=13, value=0.5), id="opset13-value-0.5"),
    pytest.param(dict(opset=10, value=0.5), id="opset10-value-0.5",
                 marks=_PAD_VALUE_ATTRIBUTE_LOST),
    pytest.param(dict(opset=13, value_from_input=True),
                 id="opset13-value-from-input"),
    pytest.param(dict(opset=13, pads=[0, 1, 0, 0, 0, 0, 0, 0]),
                 id="opset13-channel-pad"),
    pytest.param(dict(opset=10, pads=[0, 1, 0, 0, 0, 0, 0, 0]),
                 id="opset10-channel-pad"),
    pytest.param(dict(opset=13, pads=[0, 0, 1, -1, 0, 0, 0, 2]),
                 id="opset13-negative"),
])
def test_onnx_pad_that_is_not_zero_padding_refuses_or_is_sound(tmp_path, case,
                                                               kind):
    """An ONNX Pad in edge or reflect mode, with a non-zero constant, with a
    constant computed from the input (ReduceMean of X), padding the channel
    axis, or with a negative amount (onnxruntime crops): the reach must
    refuse (NotImplementedError) or answer with sets that hold every
    onnxruntime output. The same loaded model is reached twice: the graph
    walk stores the pads on the module at the first visit, so a check made
    only then is skipped by the second reach. Catches a mode check that
    compares against the ONNX spelling while the loaded module stores
    torch's ('replicate'), a mode or a value not passed to the check, a
    value input that is not checked or is checked at the first reach only,
    a constant that cannot be resolved taken as 0, and a channel pad or a
    crop reached as something else."""
    opset = case["opset"]
    pads = case.get("pads", _ONNX_PADS)
    path, chw, out_chw = _onnx_pad_chain(
        tmp_path, "padx", opset, pads, mode=case.get("mode", "constant"),
        value=case.get("value"),
        value_from_input=case.get("value_from_input", False))
    lo, hi = pixel_box(*chw, seed=9)
    model = load_onnx(path)
    for visit in (1, 2):
        what = f"ONNX Pad {case} on {kind}, reach {visit} of one model"
        try:
            result = reach_model(model, image_set(kind, lo, hi), "approx",
                                 what)
        except NotImplementedError:
            continue
        assert_covered(result, replay_onnx(path, samples(lo, hi), [1, *chw]),
                       what, out_chw=out_chw)


# --------------------------------------------------------------------------- #
# Refusals                                                                      #
# --------------------------------------------------------------------------- #
_PAD_FLAT_TYPE_ERROR = known_bug(
    23, "Pad on a flat Star or Zono raises TypeError instead of the refusal "
        "of an unsupported combination (NotImplementedError, which "
        "UnsupportedError will subclass), so a caller that records a refusal "
        "and tries the next method stops instead", raises=TypeError)


@pytest.mark.parametrize("kind", [
    pytest.param("Star", marks=_PAD_FLAT_TYPE_ERROR),
    pytest.param("Zono", marks=_PAD_FLAT_TYPE_ERROR),
    "Box", "Hexatope", "Octatope"])
def test_pad_on_a_flat_set_is_refused(kind):
    """A flat set carries no height, width or channel count, so no rule pads
    it: ``reach_layer`` must refuse (NotImplementedError), never answer.
    Catches a lane that passes the pad through as the identity and a kernel
    that skips a set it cannot pad (an empty result)."""
    lo, hi = flat_box(30)
    method = "exact" if kind == "Star" else "approx"
    with pytest.raises(NotImplementedError):
        reach_layer(nn.ZeroPad2d((0, 3, 2, 1)), [flat_set(kind, lo, hi)],
                    method)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_onnx_pad_whose_pads_were_not_resolved_is_refused(kind):
    """An OnnxPadDynamic whose pads input was never resolved from the graph
    (a direct ``reach_layer`` call) has no pad amounts: the reach must
    refuse, never pad by zero amounts."""
    lo, hi = pixel_box(2, 3, 5)
    with pytest.raises(NotImplementedError):
        reach_layer(OnnxPadDynamic(mode="constant"),
                    [image_set(kind, lo, hi)], "approx")


class _DoubledPad(nn.ZeroPad2d):
    """A ZeroPad2d subclass whose forward is not the stock one: twice the
    padded image."""

    def forward(self, x):
        return 2.0 * super().forward(x)


class _NamedPad(nn.ZeroPad2d):
    """A ZeroPad2d subclass that keeps the stock forward (a user's own name
    for the layer)."""


_OWN_FORWARD = known_bug(
    7, "reach_layer matches a ZeroPad2d subclass by isinstance and pads with "
       "zeros by the stock rule, so a subclass whose forward is not the stock "
       "one (here twice the padded image) is answered for another function "
       "and every true output lies outside (REACH-9, on the "
       "Pad branch of the Star and of the Zono lane; the model entry traces "
       "into such a subclass and refuses its functional pad)")


@pytest.mark.parametrize("subclass", [
    pytest.param(_DoubledPad, marks=_OWN_FORWARD, id="own-forward"),
    pytest.param(_NamedPad, id="stock-forward")])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_pad_subclass_refuses_or_holds_every_torch_output(kind, subclass):
    """A ZeroPad2d subclass handed to ``reach_layer`` on a (2, 3, 5) pixel
    box: the reach must refuse it (``NotImplementedError``) or hold every
    output of the subclass's own forward (torch in float64 on 128 corners
    and grid points, membership by LP on the rows). Catches a subclass
    answered by the stock rule whatever its forward computes; the subclass
    with the stock forward is the passing neighbour of the marked case."""
    layer = subclass((0, 3, 2, 1))
    chw = (2, 3, 5)
    lo, hi = pixel_box(*chw, seed=26)
    what = f"{subclass.__name__} on {kind}"
    try:
        result = reach_one_layer(layer, [image_set(kind, lo, hi)], "approx",
                                 what)
    except NotImplementedError:
        return
    assert_covered(result, replay_torch(layer, samples(lo, hi), (1,) + chw),
                   what, out_chw=_out_chw(layer, chw))


_MIXED_LIST_TYPE_ERROR = known_bug(
    23, "pad_star handed an ImageZono after an ImageStar, and pad_zono "
        "handed an ImageStar after an ImageZono, raise TypeError ('pad_star "
        "requires ImageStar') instead of the refusal of an unsupported "
        "combination (NotImplementedError) or of an input that cannot be "
        "read (ValueError)", raises=TypeError)


@pytest.mark.parametrize("first,second", [
    pytest.param("ImageStar", "ImageZono", marks=_MIXED_LIST_TYPE_ERROR),
    pytest.param("ImageZono", "ImageStar", marks=_MIXED_LIST_TYPE_ERROR)])
def test_pad_on_a_list_of_mixed_set_types_refuses_or_is_sound(first, second):
    """An ImageStar and an ImageZono in one ``reach_layer`` call (only a
    direct call can build such a list). The dispatcher routes by the first
    set; the kernel must refuse the other (``ValueError`` for an input it
    cannot read, ``NotImplementedError`` for an unsupported combination),
    or answer with sets that hold the outputs of both. Catches a kernel
    that skips a set of the wrong type. Both orders raise ``TypeError``
    today; the passing neighbours are the several-set items of
    ``test_pad_layer_holds_every_torch_output`` on the same layer
    (``[tuple-ImageStar-exact-split]``, ``[tuple-ImageZono-approx-split]``)."""
    layer, chw = _make_pad("tuple")
    lo, hi = pixel_box(*chw)
    try:
        result = reach_layer(layer, [image_set(first, lo, hi),
                                     image_set(second, lo + 4, hi + 4)],
                             "approx")
    except (ValueError, NotImplementedError):
        return
    points = samples(lo, hi, 32) + samples(lo + 4, hi + 4, 32)
    assert_covered(result, replay_torch(layer, points, (1,) + chw),
                   f"{layer} on a mixed list", out_chw=_out_chw(layer, chw))
