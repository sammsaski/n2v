"""
Soundness cells for ONNX Reshape and the torch tensor methods ``view``,
``reshape`` and ``flatten`` as graph nodes.

The walker's Reshape rules (``_handle_reshape``, ``_resolve_shape``,
``_reshape_imagestar``, ``_reshape_imagezono``, ``_reshape_star``,
``_reshape_zono``, ``_reshape_box``, the keep-flat decision
``_reshape_feeds_spatial``) and the torch nodes that reach them (``x.view``,
``x.reshape``, ``x.flatten``, ``torch.flatten``), through the model entry
``reach_pytorch_model``: an ONNX file loaded by ``load_onnx`` and a torch model
traced by torch.fx. A direct ``reach_layer`` call with the ``OnnxReshape``
module is refused by every lane (it has no target shape).

A reshape keeps the tensor's row-major order, while an image set stores
height, width, channel: every rule that makes or reads an image set must
convert, which a square one-channel image hides. The image here is
(C, H, W) = (2, 3, 5) with a different interval per pixel.

Oracle: onnxruntime (or torch in float64) replays member points of the input
set, dyadic so float32 is exact, and every true output must be a member of a
returned set by an LP on the set's own rows (``tests/_oracles/membership``),
an image output converted to the stored order with ``out_chw``. The exact
cells compare with the exact image in rationals (``tests/_oracles/rational``)
of a Star with three predicate variables, both directions.

Several input sets are never handed to one ``reach_pytorch_model`` call
(``reach_model`` reaches each set on its own), so the loop of
``_handle_reshape`` over several sets is reached only after an exact split
earlier in the graph: the exact cells put a ReLU in front of a Reshape.
"""

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper

from n2v.sets import ImageStar, Star
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_image_star,
                                constrained_star, flat_box, flat_set,
                                hwc_index, image_set, pixel_box, reach_model,
                                replay_onnx, replay_torch, samples, save_onnx)
from tests.known_bugs import known_bug

_C, _H, _W = 2, 3, 5
_SHAPE = (1, _C, _H, _W)
_N = _C * _H * _W
# The model entry refuses ``method='exact'`` on a set type with no exact
# method with ``ValueError`` today; the planned refusal type subclasses
# ``NotImplementedError``.
_REFUSALS = (NotImplementedError, ValueError)


def _dyadic(rng, *shape):
    """Quarters in [-1, 1] as float32: exact in float32 and float64."""
    return (rng.integers(-4, 5, size=shape) / 4.0).astype(np.float32)


def _shape(values):
    return np.array(values, dtype=np.int64)


def _image_inputs(kind, form, lo, hi):
    """``(sets, member points in (C, H, W) order)`` of an image set kind;
    ``form`` is "box", "row" (a binding row that is not a box row, ImageStar
    only) or "rows-only" (the same constrained ImageStar given by its rows
    alone, with no predicate box: the model entry leaves an ImageStar as it
    is given)."""
    if form == "box":
        return [image_set(kind, lo, hi)], samples(lo, hi)
    s, points = constrained_image_star(lo, hi)
    if form == "rows-only":
        s = ImageStar(s.V, s.C, s.d, None, None, _H, _W, _C)
    return [s], points


def _flat_inputs(kind, form, lo, hi):
    """The same for a flat set kind, in the vector's own order."""
    if form == "box":
        return [flat_set(kind, lo, hi)], samples(lo, hi)
    s, points = constrained_star(lo, hi)
    return [s], points


# (kind, form) of the image cells: a binding row that is not a box row needs a
# predicate system, so only the ImageStar has the "row" form.
_IMAGE_FORMS = [("ImageStar", "box"), ("ImageStar", "row"),
                ("ImageZono", "box")]
_FLAT_FORMS = [("Star", "box"), ("Star", "row"), ("Zono", "box")]


# Batch-prefixed Reshape targets of the (1, 2, 3, 5) tensor: an image whose
# three axis lengths all differ from the input's (entries move between axes),
# the same with -1 inside it, a rank change on each side of three (the branch
# of every count but three) and a flattening.
_IMAGE_TARGETS = {
    "image-5x2x3": [1, 5, 2, 3],
    "image-5x-1x3": [1, 5, -1, 3],
    "rank-2": [1, 2, 15],
    "rank-5": [1, 2, 3, 5, 1],
    "flat": [1, -1],
}
_ALONE_CELLS = (
    [(target, kind, form) for target in ("image-5x2x3", "rank-2", "rank-5",
                                         "flat")
     for kind, form in _IMAGE_FORMS]
    + [("image-5x-1x3", "ImageStar", "box"), ("image-5x-1x3", "ImageZono", "box"),
       ("image-5x2x3", "ImageStar", "rows-only"),
       ("flat", "ImageStar", "rows-only")])


@pytest.mark.parametrize("target,kind,form", [
    pytest.param(*cell, id="-".join(cell)) for cell in _ALONE_CELLS])
def test_image_reshape_alone_holds_every_output(tmp_path, target, kind, form):
    """An ImageStar or ImageZono of (2, 3, 5) reshaped alone. The result is
    the reshaped tensor: for a 3-D target an image set must store exactly
    the target's (C, H, W) (a flat set is compared in row-major order), for
    any other target the result is flat in row-major order. The layout the
    outputs are converted to comes from the target, never from the result.
    Catches a flattening in stored height, width, channel order (the
    one-entry branch and the branch of every other rank), a centre and
    generators reordered differently (ImageZono), an ImageZono taking the
    plain Zono rule (it is a Zono subclass), a 3-D target stored without the
    channel-first to channel-last conversion or transposed, an image
    labelled with the input's height, width and channels, a -1 inside the
    image target resolved at the wrong axis, and a rule that mishandles an
    ImageStar with no predicate box."""
    shape = _IMAGE_TARGETS[target]
    out_shape = list(np.zeros(_SHAPE).reshape(shape).shape)
    path = save_onnx(tmp_path, [helper.make_node("Reshape", ["X", "S"], ["Y"])],
                     _SHAPE, out_shape, inits={"S": _shape(shape)})
    lo, hi = pixel_box(_C, _H, _W)
    sets, points = _image_inputs(kind, form, lo, hi)
    what = f"Reshape {target} of a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, _SHAPE), what,
                   out_chw=(tuple(out_shape[1:]) if len(out_shape) == 4
                            else None))


@pytest.mark.parametrize("kind,form", _IMAGE_FORMS)
@pytest.mark.parametrize("target", ["image-5x2x3", "rank-2", "flat"])
def test_image_reshape_between_affine_layers(tmp_path, target, kind, form):
    """Conv 1x1 -> Reshape -> Reshape(1, -1) -> Gemm on an image set whose
    box is the pixel box times 8 (bounds well beyond -1 and 1, intervals up
    to 10 wide): the rows the Reshape emits are consumed by a Gemm. Catches
    the layout bugs of the one-node cell in the form a downstream layer sees
    them, on a basis that the 1x1 convolution has mixed across channels, a
    Reshape rule that loses the binding row's correlation, and a rule that
    reads one predicate variable in place of another (invisible while every
    entry stays inside [-1, 1])."""
    rng = np.random.default_rng(2)
    inits = {"K": _dyadic(rng, _C, _C, 1, 1), "kb": _dyadic(rng, _C),
             "S": _shape(_IMAGE_TARGETS[target]), "F": _shape([1, -1]),
             "W2": _dyadic(rng, 3, _N), "b2": _dyadic(rng, 3)}
    nodes = [helper.make_node("Conv", ["X", "K", "kb"], ["A"]),
             helper.make_node("Reshape", ["A", "S"], ["R"]),
             helper.make_node("Reshape", ["R", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, _SHAPE, [1, 3], inits=inits)
    lo, hi = pixel_box(_C, _H, _W, seed=1)
    sets, points = _image_inputs(kind, form, 8 * lo, 8 * hi)
    what = f"Conv/Reshape {target}/Gemm on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, _SHAPE), what)


# A reshape to an image that feeds a convolution, directly or through layers
# the walker looks past (an activation, a Transpose).
_TO_CONV = {
    "5x2x3-direct": ([1, 5, 2, 3], None, (2, 1), 5),
    "5x2x3-relu": ([1, 5, 2, 3], "Relu", (2, 1), 5),
    "3x2x5-leakyrelu": ([1, 3, 2, 5], "LeakyRelu", (1, 2), 3),
    "5x2x3-transpose": ([1, 5, 2, 3], "Transpose", (2, 1), 5),
}


def _to_conv_model(tmp_path, case, in_shape, rng, name="m"):
    """Reshape(case) -> [middle] -> Conv -> Reshape(1, -1) -> Gemm(3); with
    no middle layer the Reshape feeds the Conv directly."""
    target, middle, kernel, channels = _TO_CONV[case]
    nodes = [helper.make_node("Reshape", ["X", "S"], ["R"])]
    after = list(np.zeros(target).shape)
    if middle == "Relu":
        nodes.append(helper.make_node("Relu", ["R"], ["M"]))
    elif middle == "LeakyRelu":
        nodes.append(helper.make_node("LeakyRelu", ["R"], ["M"], alpha=0.25))
    elif middle == "Transpose":
        nodes.append(helper.make_node("Transpose", ["R"], ["M"],
                                      perm=[0, 1, 3, 2]))
        after = [after[0], after[1], after[3], after[2]]
    conv_in = "R" if middle is None else "M"
    conv_out = [1, 2, after[2] - kernel[0] + 1, after[3] - kernel[1] + 1]
    n_conv = int(np.prod(conv_out))
    nodes += [helper.make_node("Conv", [conv_in, "K", "kb"], ["V"],
                               kernel_shape=list(kernel)),
              helper.make_node("Reshape", ["V", "F"], ["G"]),
              helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    inits = {"S": _shape(target), "F": _shape([1, -1]),
             "K": _dyadic(rng, 2, channels, *kernel), "kb": _dyadic(rng, 2),
             "W2": _dyadic(rng, 3, n_conv), "b2": _dyadic(rng, 3)}
    return save_onnx(tmp_path, nodes, in_shape, [1, 3], inits=inits,
                     name=name)


@pytest.mark.parametrize("kind,form", _FLAT_FORMS)
@pytest.mark.parametrize("case", list(_TO_CONV))
def test_flat_reshape_to_an_image_feeding_a_convolution(tmp_path, case, kind,
                                                        form):
    """A flat Star or Zono of 30 entries, its box times 8, reshaped to an
    image that a convolution with an asymmetric kernel consumes, directly or
    behind a ReLU, a LeakyReLU or a Transpose. The flat rows are in
    channel-first order and must be stored as height, width, channel.
    Catches the reshape without that transposition (``_reshape_star`` and
    the Zono 3-D branch, whose permutation may also be inverted), and a
    consumer walk that does not see the convolution behind a pass-through
    layer (the set then stays flat and the convolution is fed the wrong
    layout or refuses)."""
    rng = np.random.default_rng(4)
    path = _to_conv_model(tmp_path, case, (1, _N), rng)
    lo, hi = flat_box(_N, seed=2)
    sets, points = _flat_inputs(kind, form, 8 * lo, 8 * hi)
    what = f"Reshape {case}/Conv/Gemm on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, _N)), what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("case", ["5x2x3-direct", "5x2x3-transpose",
                                  "3x2x5-leakyrelu"])
def test_image_reshape_to_another_image_feeding_a_convolution(tmp_path, case,
                                                              kind):
    """An image set of (2, 3, 5) reshaped to an image of other axis lengths
    that a convolution consumes. Catches a 3-D image reshape that keeps the
    stored order of the input image (``_reshape_imagestar`` and
    ``_reshape_imagezono`` 3-D branches) or labels the result with the
    wrong height, width and channel count."""
    rng = np.random.default_rng(6)
    path = _to_conv_model(tmp_path, case, _SHAPE, rng)
    lo, hi = pixel_box(_C, _H, _W, seed=5)
    sets, points = _image_inputs(kind, "box", lo, hi)
    what = f"Reshape {case}/Conv/Gemm on a {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, _SHAPE), what)


@pytest.mark.parametrize("kind", ["Box", "Hexatope", "Octatope"])
def test_flat_reshape_into_a_convolution_on_a_set_without_a_conv_rule(
        tmp_path, kind):
    """Box, Hexatope and Octatope have no convolution rule: a flat set
    reshaped to an image in front of a Conv must be refused
    (NotImplementedError), never passed through as a flat set."""
    path = _to_conv_model(tmp_path, "5x2x3-direct", (1, _N),
                          np.random.default_rng(4))
    lo, hi = flat_box(_N, seed=2)
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                    f"Reshape/Conv on a {kind}")


# A 3-D reshape that feeds no convolution is not an image (a transformer head
# split): a flat set stays flat and the consumer works on the tensor's axes.
def _non_spatial_model(tmp_path, consumer, rng, in_shape=(1, _N),
                       target=(1, 2, 3, 5)):
    """Reshape(target) -> consumer -> Reshape(1, -1) -> Gemm(3). The
    consumer is a Softmax over the last axis or over axis 2, a ReLU then a
    Softmax over the last axis, or a MatMul with a constant over the last
    axis."""
    inits = {"S": _shape(target), "F": _shape([1, -1])}
    nodes = [helper.make_node("Reshape", ["X", "S"], ["R"])]
    n_mid = _N
    if consumer == "softmax-last-axis":
        nodes.append(helper.make_node("Softmax", ["R"], ["M"], axis=-1))
    elif consumer == "softmax-axis-2":
        nodes.append(helper.make_node("Softmax", ["R"], ["M"], axis=2))
    elif consumer == "relu-softmax-last-axis":
        nodes += [helper.make_node("Relu", ["R"], ["Q"]),
                  helper.make_node("Softmax", ["Q"], ["M"], axis=-1)]
    else:
        nodes.append(helper.make_node("MatMul", ["R", "B"], ["M"]))
        inits["B"] = _dyadic(rng, target[-1], 4)
        n_mid = _N // target[-1] * 4
    inits.update({"W2": _dyadic(rng, 3, n_mid), "b2": _dyadic(rng, 3)})
    nodes += [helper.make_node("Reshape", ["M", "F"], ["G"]),
              helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    return save_onnx(tmp_path, nodes, in_shape, [1, 3], inits=inits)


_NON_SPATIAL_CELLS = (
    [(consumer, kind, form)
     for consumer in ("softmax-last-axis", "softmax-axis-2", "matmul")
     for kind, form in _FLAT_FORMS]
    + [("softmax-last-axis", "Box", "box"),
       ("relu-softmax-last-axis", "Star", "box")])


@pytest.mark.parametrize("consumer,kind,form", [
    pytest.param(*cell, id="-".join(cell)) for cell in _NON_SPATIAL_CELLS])
def test_flat_reshape_without_a_spatial_consumer_stays_flat(tmp_path,
                                                            consumer, kind,
                                                            form):
    """Reshape of a flat set to (1, 2, 3, 5) followed by a Softmax over one
    axis, a ReLU and a Softmax, or a MatMul over the last axis, then Gemm:
    no convolution reads the reshaped value, so the set must stay flat in
    row-major order. Catches a walker that does not pass the keep-flat
    decision to the Reshape rule, a Star rule that ignores it (the value
    becomes an ImageStar read as channels, height, width, whose Softmax runs
    over the wrong entries), a consumer walk that takes a pass-through layer
    such as the ReLU for a spatial consumer, and a Box rule that permutes a
    3-D target as if it were an image."""
    rng = np.random.default_rng(8)
    path = _non_spatial_model(tmp_path, consumer, rng)
    lo, hi = flat_box(_N, seed=4)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"Reshape/{consumer}/Gemm on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, _N)), what)


_IMAGE_SOFTMAX_DEFECT = (
    "A Softmax over the last axis of an image set's tensor is computed as one "
    "softmax over all H*W*C entries (registry ACT-1). A Reshape of an "
    "ImageStar to a 3-D target builds an ImageStar whatever the keep-flat "
    "decision (_reshape_imagestar never reads it), so Reshape (1, 5, 2, 3) "
    "then Softmax(axis=-1) on a (2, 3, 5) ImageStar leaves 128 of 128 "
    "onnxruntime outputs outside, where the same model on a flat Star is "
    "answered soundly")


@pytest.mark.parametrize("consumer,kind", [
    pytest.param("softmax-last-axis", "ImageStar",
                 marks=known_bug(7, _IMAGE_SOFTMAX_DEFECT)),
    ("softmax-last-axis", "ImageZono"),
    ("matmul", "ImageStar"), ("matmul", "ImageZono")])
def test_image_reshape_without_a_spatial_consumer_refuses_or_holds_every_output(
        tmp_path, consumer, kind):
    """Reshape of a (2, 3, 5) image set to (1, 5, 2, 3) followed by a
    Softmax over the last axis or a MatMul over the last axis, then Gemm: no
    convolution reads the value. The library must refuse
    (NotImplementedError) or hold every onnxruntime output. Today the
    ImageZono and the MatMul consumer are refused and the ImageStar Softmax
    is answered wrongly (ACT-1 reached through the 3-D image reshape); the
    flat Star of the same tensor
    (``test_flat_reshape_without_a_spatial_consumer_stays_flat[softmax-last-axis-Star-box]``)
    is the passing neighbour with the same reference and check. Catches an
    image reshape that hands a non-spatial consumer a set it reads in the
    wrong layout."""
    rng = np.random.default_rng(8)
    path = _non_spatial_model(tmp_path, consumer, rng, in_shape=_SHAPE,
                              target=(1, 5, 2, 3))
    lo, hi = pixel_box(_C, _H, _W, seed=4)
    what = f"Reshape(1, 5, 2, 3)/{consumer}/Gemm on a {kind}"
    try:
        out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                          what)
    except NotImplementedError:
        return                      # a clear refusal keeps the promise
    assert_covered(out, replay_onnx(path, samples(lo, hi), _SHAPE), what)


_TWO_CONSUMERS_DEFECT = (
    "A Reshape whose value feeds both a convolution and a Softmax over the "
    "last axis is made an ImageStar for both consumers (the keep-flat walk "
    "finds the convolution), and the Softmax of an ImageStar is one softmax "
    "over all entries (registry ACT-1): on a flat Star 10 of 128 onnxruntime "
    "outputs lie outside (largest distance 0.537)")


@pytest.mark.parametrize("second", [
    pytest.param("softmax", marks=known_bug(7, _TWO_CONSUMERS_DEFECT)),
    "relu"])
def test_reshape_feeding_a_convolution_and_another_layer_refuses_or_holds_every_output(
        tmp_path, second):
    """A flat Star reshaped to (1, 5, 2, 3) whose value feeds a Conv and a
    second layer (a Softmax over the last axis, or a ReLU); both branches are
    flattened and joined by a Concat before a Gemm. The keep-flat walk sees
    the convolution, so the second layer receives an ImageStar. The library
    must refuse (NotImplementedError) or hold every onnxruntime output. The
    ReLU branch is the passing neighbour. Catches a consumer walk that stops
    at the first user of the reshape and a reshape whose single result is
    read wrongly by one of its consumers."""
    rng = np.random.default_rng(6)
    nodes = [helper.make_node("Reshape", ["X", "S"], ["R"]),
             helper.make_node("Conv", ["R", "K", "kb"], ["V"],
                              kernel_shape=[2, 1]),
             helper.make_node("Reshape", ["V", "F"], ["G1"]),
             (helper.make_node("Softmax", ["R"], ["M"], axis=-1)
              if second == "softmax" else
              helper.make_node("Relu", ["R"], ["M"])),
             helper.make_node("Reshape", ["M", "F"], ["G2"]),
             helper.make_node("Concat", ["G1", "G2"], ["G"], axis=1),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    inits = {"S": _shape([1, 5, 2, 3]), "F": _shape([1, -1]),
             "K": _dyadic(rng, 2, 5, 2, 1), "kb": _dyadic(rng, 2),
             "W2": _dyadic(rng, 3, 6 + _N), "b2": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, (1, _N), [1, 3], inits=inits)
    lo, hi = flat_box(_N, seed=2)
    what = f"Reshape/[Conv, {second}]/Concat/Gemm on a Star"
    try:
        out = reach_model(load_onnx(path), flat_set("Star", lo, hi), "approx",
                          what)
    except NotImplementedError:
        return                      # a clear refusal keeps the promise
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, _N)), what)


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box", "Hexatope",
                                  "Octatope"])
def test_flat_reshapes_keep_the_row_major_order_of_every_flat_kind(tmp_path,
                                                                   kind):
    """(1, 30) -> (1, 2, 15) -> (1, 6, 5) -> (1, -1) -> Gemm on every flat
    set kind, its box times 8: reshapes of a flat set that feed no
    convolution change no entry. Catches a flat rule that permutes rows (the
    image conversion applied to a 2-D or non-spatial target) or that drops
    the set."""
    rng = np.random.default_rng(9)
    nodes = [helper.make_node("Reshape", ["X", "S1"], ["A"]),
             helper.make_node("Reshape", ["A", "S2"], ["B"]),
             helper.make_node("Reshape", ["B", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    inits = {"S1": _shape([1, 2, 15]), "S2": _shape([1, 6, 5]),
             "F": _shape([1, -1]), "W2": _dyadic(rng, 3, _N),
             "b2": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, (1, _N), [1, 3], inits=inits)
    lo, hi = flat_box(_N, seed=6)
    lo, hi = 8 * lo, 8 * hi
    what = f"flat Reshapes/Gemm on a {kind}"
    out = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                      what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, _N)), what)


@pytest.mark.parametrize("kind", ["Star", "Zono", "ImageStar", "ImageZono"])
def test_reshape_of_a_set_with_more_variables_than_entries_holds_every_output(
        tmp_path, kind):
    """Relu -> Reshape(1, 5, 2, 3) -> Conv -> Reshape(1, -1) -> Gemm under
    approx, on a box times 8 whose entries cross zero: the ReLU relaxation
    hands the Reshape a Star or ImageStar with more predicate variables than
    entries, and a Zono or ImageZono with more generators than entries (the
    shape no ``from_bounds`` set has). Catches a reshape rule that reads the
    number of basis columns or generators from the number of entries, or
    that drops the relaxation's own variables."""
    rng = np.random.default_rng(15)
    image = kind.startswith("Image")
    in_shape = _SHAPE if image else (1, _N)
    nodes = [helper.make_node("Relu", ["X"], ["Q"]),
             helper.make_node("Reshape", ["Q", "S"], ["R"]),
             helper.make_node("Conv", ["R", "K", "kb"], ["V"],
                              kernel_shape=[2, 1]),
             helper.make_node("Reshape", ["V", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    inits = {"S": _shape([1, 5, 2, 3]), "F": _shape([1, -1]),
             "K": _dyadic(rng, 2, 5, 2, 1), "kb": _dyadic(rng, 2),
             "W2": _dyadic(rng, 3, 6), "b2": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, in_shape, [1, 3], inits=inits)
    if image:
        lo, hi = pixel_box(_C, _H, _W, seed=7)
        lo, hi = 8 * lo - 4, 8 * hi - 4
        input_set = image_set(kind, lo, hi)
    else:
        lo, hi = flat_box(_N, seed=7)
        lo, hi = 8 * lo, 8 * hi
        input_set = flat_set(kind, lo, hi)
    if not (np.any(lo < 0) and np.any(hi > 0) and np.any((lo < 0) & (hi > 0))):
        pytest.fail("the box has no entry that crosses zero")
    what = f"Relu/Reshape/Conv/Gemm on a {kind}"
    out = reach_model(load_onnx(path), input_set, "approx", what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), in_shape), what)


# --------------------------------------------------------------------------- #
# torch models: x.view, x.reshape, x.flatten, torch.flatten                     #
# --------------------------------------------------------------------------- #
class _ImageToLinear(nn.Module):
    """A (1, 2, 3, 5) input flattened by one spelling, then a Linear."""

    def __init__(self, how):
        super().__init__()
        self.fc = nn.Linear(_N, 3)
        self.how = how

    def forward(self, x):
        if self.how == "x.flatten(1)":
            h = x.flatten(1)
        elif self.how == "torch.flatten(x, 1)":
            h = torch.flatten(x, 1)
        elif self.how == "x.view(x.size(0), -1)":
            h = x.view(x.size(0), -1)
        elif self.how == "x.reshape(x.shape[0], 30)":
            h = x.reshape(x.shape[0], _N)
        else:
            h = x.view((1, _N))
        return self.fc(h)


class _FlatToConv(nn.Module):
    """A flat input of 30 entries viewed as (1, 5, 2, 3), a ReLU, a Conv2d
    with an asymmetric kernel, flatten and a Linear."""

    def __init__(self, how):
        super().__init__()
        self.conv = nn.Conv2d(5, 2, (2, 1))
        self.fc = nn.Linear(2 * 1 * 3, 3)
        self.how = how

    def forward(self, x):
        if self.how == "x.view(1, 5, 2, 3)":
            h = x.view(1, 5, 2, 3)
        elif self.how == "x.reshape(shape=(1, 5, 2, 3))":
            h = x.reshape(shape=(1, 5, 2, 3))
        elif self.how == "torch.reshape(x, (1, 5, 2, 3))":
            h = torch.reshape(x, (1, 5, 2, 3))
        else:
            h = x.reshape(-1, 5, 2, 3)
        return self.fc(self.conv(F.relu(h)).flatten(1))


def _dyadic_module(module, seed):
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.tensor(rng.integers(-4, 5, size=tuple(p.shape))
                                 / 4.0))
    return module.double().eval()


# Each torch spelling once on the ImageStar (its own argument parsing in the
# walker), the common view also on the ImageZono, and the two node kinds that
# reach a flat set's rule on every flat kind.
_FLATTEN_CELLS = (
    [(how, "ImageStar") for how in ("x.flatten(1)", "torch.flatten(x, 1)",
                                    "x.view(x.size(0), -1)",
                                    "x.reshape(x.shape[0], 30)",
                                    "x.view((1, 30))")]
    + [("x.view(x.size(0), -1)", "ImageZono")]
    + [(how, kind) for how in ("x.flatten(1)", "x.view(x.size(0), -1)")
       for kind in ("Star", "Zono", "Box")])


@pytest.mark.parametrize("how,kind", _FLATTEN_CELLS)
def test_torch_flatten_and_view_of_an_image_holds_every_output(how, kind):
    """Torch spellings of a flatten (tensor method ``flatten``,
    ``torch.flatten``, ``view`` and ``reshape`` with a traced batch entry
    from ``x.size(0)`` or ``x.shape[0]``, ``view`` with one tuple) on a
    (2, 3, 5) image, then a Linear. An image set must be read in
    channel-first order; a flat set with ``input_shape`` is already in that
    order. Catches a tensor-method or function node that flattens in the
    stored order or is taken for the identity, and a view whose batch entry
    (a traced size) is taken for a shape."""
    model = _dyadic_module(_ImageToLinear(how), seed=3)
    lo, hi = pixel_box(_C, _H, _W, seed=7)
    if kind.startswith("Image"):
        input_set, options = image_set(kind, lo, hi), {}
    else:
        input_set, options = flat_set(kind, lo, hi), {"input_shape": (_C, _H,
                                                                      _W)}
    what = f"{how} then Linear on a {kind}"
    out = reach_model(model, input_set, "approx", what, **options)
    assert_covered(out, replay_torch(model, samples(lo, hi), _SHAPE), what)


@pytest.mark.parametrize("kind", ["Star", "Zono"])
@pytest.mark.parametrize("how", ["x.view(1, 5, 2, 3)",
                                 "x.reshape(-1, 5, 2, 3)"])
def test_torch_view_of_a_flat_set_into_a_convolution(how, kind):
    """A torch ``view`` or ``reshape`` of a flat set to (1, 5, 2, 3), behind
    it a functional ReLU and a Conv2d: the walker must find the convolution
    past the ReLU and store the image as height, width, channel. Catches the
    reshape rule without the layout conversion and a consumer walk that
    stops at a function node."""
    model = _dyadic_module(_FlatToConv(how), seed=5)
    lo, hi = flat_box(_N, seed=8)
    what = f"{how}/ReLU/Conv2d on a {kind}"
    out = reach_model(model, flat_set(kind, lo, hi), "approx", what)
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, _N)), what)


def test_torch_view_of_a_flat_set_into_a_convolution_box_is_refused():
    """A Box has no convolution rule: the same model must be refused."""
    model = _dyadic_module(_FlatToConv("x.view(1, 5, 2, 3)"), seed=5)
    lo, hi = flat_box(_N, seed=8)
    with pytest.raises(NotImplementedError):
        reach_model(model, flat_set("Box", lo, hi), "approx", "view/Conv Box")


_KEYWORD_TARGET_DEFECT = (
    "The torch view/reshape branch of the graph walker reads the target "
    "shape from the positional arguments only: x.reshape(shape=(1, 5, 2, 3)) "
    "or x.view(size=...) gives the target (), and _reshape_star raises "
    "ValueError (Cannot reshape Star of dim 30 to shape ()) where torch "
    "answers; on an ImageStar the target () flattens the set, and a following "
    "Conv2d raises ValueError")


@known_bug(7, _KEYWORD_TARGET_DEFECT, raises=ValueError)
def test_torch_reshape_with_a_keyword_target_into_a_convolution():
    """``x.reshape(shape=(1, 5, 2, 3))``, the keyword spelling of the
    target, in the model of ``test_torch_view_of_a_flat_set_into_a_convolution``
    on a flat Star, whose ``[x.reshape(-1, 5, 2, 3)-Star]`` case is the
    passing neighbour with the same model, replay and check. The library
    must answer with a set that holds every torch output or refuse with
    NotImplementedError. Catches a walker that reads only the positional
    arguments of the node."""
    model = _dyadic_module(_FlatToConv("x.reshape(shape=(1, 5, 2, 3))"),
                           seed=5)
    lo, hi = flat_box(_N, seed=8)
    what = "x.reshape(shape=...)/ReLU/Conv2d on a Star"
    try:
        out = reach_model(model, flat_set("Star", lo, hi), "approx", what)
    except NotImplementedError:
        return                      # a clear refusal keeps the promise
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, _N)), what)


def test_torch_reshape_function_node_refuses_or_holds_every_output():
    """``torch.reshape(x, (1, 5, 2, 3))``, a function node, in the same
    model on a flat Star: today the walker has no rule for it and refuses
    (NotImplementedError); if it answers, every torch output must be held.
    Catches a function node that is mapped to another layer (a Flatten, the
    identity) and answered in the wrong layout."""
    model = _dyadic_module(_FlatToConv("torch.reshape(x, (1, 5, 2, 3))"),
                           seed=5)
    lo, hi = flat_box(_N, seed=8)
    what = "torch.reshape/ReLU/Conv2d on a Star"
    try:
        out = reach_model(model, flat_set("Star", lo, hi), "approx", what)
    except NotImplementedError:
        return                      # a clear refusal keeps the promise
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, _N)), what)


class _ViewToSoftmax(nn.Module):
    """A flat input of 30 entries viewed as (1, 5, 2, 3), a Softmax over the
    last axis, a flattening reshape and a Linear: a non-spatial view."""

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(_N, 3)

    def forward(self, x):
        return self.fc(self.softmax(x.view(1, 5, 2, 3)).reshape(1, -1))


@pytest.mark.parametrize("kind", ["Star", "Zono"])
def test_torch_view_of_a_flat_set_without_a_spatial_consumer_stays_flat(kind):
    """A torch ``view`` of a flat set to (1, 5, 2, 3) that feeds a Softmax
    over the last axis, not a convolution: the set must stay flat in
    row-major order. Catches the torch view branch of the walker without the
    keep-flat decision (a Star becomes an ImageStar whose Softmax runs over
    all entries; a Zono becomes an ImageZono, which the Softmax refuses)."""
    model = _dyadic_module(_ViewToSoftmax(), seed=3)
    lo, hi = flat_box(_N, seed=2)
    what = f"x.view(1, 5, 2, 3)/Softmax/Linear on a {kind}"
    out = reach_model(model, flat_set(kind, lo, hi), "approx", what)
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, _N)), what)


class _TracedView(nn.Module):
    """A flat input of 30 entries viewed with a traced entry in the target,
    a ReLU and a Linear."""

    def __init__(self, how):
        super().__init__()
        self.fc = nn.Linear(_N, 3)
        self.how = how

    def forward(self, x):
        if self.how == "x.view(x.size(0), 30)":
            h = x.view(x.size(0), _N)
        else:
            h = x.view(1, x.size(1))
        return self.fc(F.relu(h))


@pytest.mark.parametrize("how,may_refuse", [
    ("x.view(x.size(0), 30)", False), ("x.view(1, x.size(1))", True)])
def test_torch_view_with_a_traced_entry_holds_every_output(how, may_refuse):
    """A torch ``view`` whose target holds a traced size: in the batch entry
    (which the walker strips; it must answer) or in a kept entry (the
    walker cannot read it as a constant; it must answer soundly or refuse
    with NotImplementedError). Catches a traced entry read as a shape and a
    view that is passed the sets of another node."""
    model = _dyadic_module(_TracedView(how), seed=14)
    lo, hi = flat_box(_N, seed=14)
    what = f"{how}/ReLU/Linear on a Star"
    try:
        out = reach_model(model, flat_set("Star", lo, hi), "approx", what)
    except NotImplementedError:
        if may_refuse:
            return                  # a clear refusal keeps the promise
        raise
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, _N)), what)


# --------------------------------------------------------------------------- #
# The exact method through a Reshape: exact image in rationals                  #
# --------------------------------------------------------------------------- #
def _linear_part(tmp_path, nodes, in_shape, out_shape, inits, name):
    """``(A, a)`` with ``f(x) = A x + a`` for the affine sub-graph ``nodes``,
    read off onnxruntime on 0 and the unit vectors (exact for dyadic
    weights), checked on one more dyadic point."""
    path = save_onnx(tmp_path, nodes, in_shape, out_shape, inits=inits,
                     name=name)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    n = int(np.prod(in_shape))

    def run(x):
        return np.asarray(session.run(None, {"X": np.asarray(
            x, dtype=np.float32).reshape(in_shape)})[0],
            dtype=np.float64).reshape(-1)
    a = run(np.zeros(n))
    A = np.stack([run(e) - a for e in np.eye(n)], axis=1)
    probe = np.random.default_rng(0).integers(-4, 5, size=n) / 4.0
    if not np.array_equal(run(probe), A @ probe + a):
        pytest.fail(f"{name}: the sub-graph is not affine on dyadic points")
    return A, a


def _assert_exact(out, pieces, what):
    computed = [rational.star(s) for s in out]
    too_big = [k for k, s in enumerate(out) if s.nVar > 3]
    assert not too_big, (f"{what}: returned sets {too_big} have more than "
                         f"the 3 predicate variables of the input, which "
                         f"an exact reshape and ReLU never add")
    escaped = rational.escapes(pieces, *computed)
    assert not escaped, f"{what}: {len(escaped)} vertices of the true image " \
                        f"are in no returned set, first {escaped[0]}"
    spread = rational.straddlers(pieces, *computed)
    assert not spread, f"{what}: pieces {spread} of the true image are not " \
                       f"held by a single returned set"
    for k, s in enumerate(computed):
        extra = rational.unreached(pieces, rational.set_vertices(s))
        assert not extra, f"{what}: returned set {k} has {len(extra)} " \
                          f"vertices with no preimage, first {extra[0][1]}"


def _low_rank_input(kind, x0, E):
    """``x0 + E a``, a in [-1, 1]^3, as a flat Star of 30 entries or as the
    (2, 3, 5) ImageStar of the same tensor (``x0`` and ``E`` in the tensor's
    row-major order)."""
    k = E.shape[1]
    box = (-np.ones((k, 1)), np.ones((k, 1)))
    if kind == "Star":
        return Star(np.hstack([x0[:, None], E]), np.zeros((0, k)),
                    np.zeros((0, 1)), *box)
    order = hwc_index(_C, _H, _W)
    return ImageStar(np.hstack([x0[order, None], E[order]]), np.zeros((0, k)),
                     np.zeros((0, 1)), *box, height=_H, width=_W,
                     num_channels=_C)


@pytest.mark.parametrize("kind", ["Star", "ImageStar"])
def test_exact_reshape_into_a_convolution_then_relu_is_the_exact_image(
        tmp_path, kind):
    """Exact method: Reshape to (5, 2, 3) -> Conv (2x1) -> ReLU ->
    Reshape(1, -1) -> Gemm, on a flat Star of 30 entries and on an ImageStar
    of (2, 3, 5), each with three predicate variables. Both directions
    against the exact image in rationals. Catches every layout bug of the
    reshape rules (the exact image is missed) and an exact reach that does
    not reach the ReLU behind the reshape with ``method='exact'`` (an
    over-approximation has vertices without a preimage)."""
    rng = np.random.default_rng(12)
    in_shape = (1, _N) if kind == "Star" else _SHAPE
    K, kb = _dyadic(rng, 2, 5, 2, 1), _dyadic(rng, 2)
    W2, b2 = _dyadic(rng, 2, 6), _dyadic(rng, 2)
    inits = {"S": _shape([1, 5, 2, 3]), "K": K, "kb": kb,
             "F": _shape([1, -1])}

    def front(out):
        return [helper.make_node("Reshape", ["X", "S"], ["R"]),
                helper.make_node("Conv", ["R", "K", "kb"], ["V"],
                                 kernel_shape=[2, 1]),
                helper.make_node("Reshape", ["V", "F"], [out])]
    A, a = _linear_part(tmp_path, front("Y"), in_shape, [1, 6], inits,
                        "prefix")
    nodes = front("G") + [
        helper.make_node("Relu", ["G"], ["Q"]),
        helper.make_node("Gemm", ["Q", "W2", "b2"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, in_shape, [1, 2],
                     inits=dict(inits, W2=W2, b2=b2))
    x0 = rng.integers(-8, 9, size=_N) / 8.0
    E = rng.integers(-4, 5, size=(_N, 3)) / 8.0
    k = E.shape[1]
    input_set = _low_rank_input(kind, x0, E)
    what = f"exact Reshape/Conv/ReLU/Gemm on a low-rank {kind}"
    out = reach_model(load_onnx(path), input_set, "exact", what)
    pieces = rational.affine(rational.box([-1] * k, [1] * k), E.tolist(),
                             x0.tolist())
    pieces = rational.relu(rational.affine(pieces, A.tolist(), a.tolist()))
    pieces = rational.affine(pieces, W2.astype(float).tolist(),
                             b2.astype(float).tolist())
    _assert_exact(out, pieces, what)


@pytest.mark.parametrize("kind", ["Star", "ImageStar"])
def test_exact_reshape_of_the_pieces_of_an_exact_split_is_the_exact_image(
        tmp_path, kind):
    """Exact method: Relu -> Reshape to (5, 2, 3) -> Conv (2x1) ->
    Reshape(1, -1) -> Gemm on a flat Star of 30 entries and on the (2, 3, 5)
    ImageStar of the same tensor, ``x0 + E a`` with three predicate
    variables, scaled by 8, two entries crossing zero and the others stable:
    the ReLU splits the set into four pieces, so both Reshapes receive
    several sets (the only way a public entry hands ``_handle_reshape`` more
    than one set). Both directions against the exact image in rationals.
    Catches a Reshape that passes on fewer sets than it is given (an early
    return in its loop, a rule applied to the first set only) and the layout
    bugs of the rules on every piece."""
    rng = np.random.default_rng(12)
    in_shape = (1, _N) if kind == "Star" else _SHAPE
    K, kb = _dyadic(rng, 2, 5, 2, 1), _dyadic(rng, 2)
    W2, b2 = _dyadic(rng, 2, 6), _dyadic(rng, 2)
    inits = {"S": _shape([1, 5, 2, 3]), "K": K, "kb": kb,
             "F": _shape([1, -1])}

    def middle(src, out):
        return [helper.make_node("Reshape", [src, "S"], ["R"]),
                helper.make_node("Conv", ["R", "K", "kb"], ["V"],
                                 kernel_shape=[2, 1]),
                helper.make_node("Reshape", ["V", "F"], [out])]
    A, a = _linear_part(tmp_path, middle("X", "Y"), in_shape, [1, 6], inits,
                        "middle")
    nodes = ([helper.make_node("Relu", ["X"], ["Q"])] + middle("Q", "G")
             + [helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"],
                                 transB=1)])
    path = save_onnx(tmp_path, nodes, in_shape, [1, 2],
                     inits=dict(inits, W2=W2, b2=b2))
    data = np.random.default_rng(21)
    E = data.integers(-4, 5, size=(_N, 3)) / 8.0
    x0 = data.choice([-1, 1], size=_N) * (2 + data.integers(0, 4, size=_N) / 8.0)
    x0[[7, 22]] = 1 / 8.0                   # the two entries that cross zero
    E[7], E[22] = [0.5, -0.25, 0.375], [-0.25, 0.5, 0.25]
    x0, E = 8 * x0, 8 * E
    pieces = rational.relu(rational.affine(
        rational.box([-1] * 3, [1] * 3), E.tolist(), x0.tolist()))
    if len(pieces) < 2:
        pytest.fail("the ReLU does not split the input set")
    pieces = rational.affine(pieces, A.tolist(), a.tolist())
    pieces = rational.affine(pieces, W2.astype(float).tolist(),
                             b2.astype(float).tolist())
    what = f"exact Relu/Reshape/Conv/Reshape/Gemm on a low-rank {kind}"
    out = reach_model(load_onnx(path), _low_rank_input(kind, x0, E), "exact",
                      what)
    assert len(out) <= 4 * len(pieces), (
        f"{what}: {len(out)} returned sets for {len(pieces)} pieces")
    _assert_exact(out, pieces, what)


@pytest.mark.parametrize("kind", ["Zono", "ImageZono", "Box", "Hexatope",
                                  "Octatope"])
def test_exact_reshape_model_on_a_set_with_no_exact_method_is_refused(
        tmp_path, kind):
    """The rows (exact, Zono / ImageZono / Box / Hexatope / Octatope) of
    Reshape: the model entry refuses ``method='exact'`` on these set types
    before any node (today ``ValueError``, '... only supports approx'; the
    planned refusal type subclasses ``NotImplementedError``), on the image
    set and on a flat set given with ``input_shape``. The model is one each
    of these types answers under approx (two Reshapes and a Gemm). Catches
    the model entry passing 'exact' through to a walk that never reads the
    method and answering under the name exact."""
    rng = np.random.default_rng(9)
    nodes = [helper.make_node("Reshape", ["X", "S"], ["R"]),
             helper.make_node("Reshape", ["R", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    inits = {"S": _shape([1, 5, 2, 3]), "F": _shape([1, -1]),
             "W2": _dyadic(rng, 3, _N), "b2": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, _SHAPE, [1, 3], inits=inits)
    lo, hi = pixel_box(_C, _H, _W)
    if kind == "ImageZono":
        input_set, options = image_set(kind, lo, hi), {}
    else:
        input_set = flat_set(kind, lo, hi)
        options = {"input_shape": (_C, _H, _W)}
    with pytest.raises(_REFUSALS):
        reach_model(load_onnx(path), input_set, "exact",
                    f"exact Reshape/Gemm on a {kind}", **options)


# --------------------------------------------------------------------------- #
# A 0 in the target shape: ONNX copies the input's dimension (allowzero = 0)   #
# --------------------------------------------------------------------------- #
_ZERO_TARGET_DEFECT = (
    "ONNX Reshape reads a 0 entry of its target shape, which means 'copy the "
    "input's dimension' (allowzero=0, the default), as a literal 0: the "
    "reshape rules raise ValueError (Cannot reshape ... to shape (2, 0, 0)) "
    "where onnxruntime returns the input's entries in the same order")
_ZERO_DIVIDE_DEFECT = (
    "ONNX Reshape with a 0 entry next to -1 in its target shape divides by "
    "zero in _resolve_shape (the 0, 'copy the input's dimension', is read "
    "as a literal 0); numpy's divide-by-zero RuntimeWarning is an error")


_ZERO_TARGETS = {
    "explicit-1-2-3-5": ([1, 2, 3, 5], False),
    "batch-0-then-minus-1": ([0, -1], False),
    "copy-height-and-width": ([1, 2, 0, 0], True),
    "copy-channels-then-minus-1": ([1, 0, -1], True),
}
_ZERO_CELLS = [
    pytest.param(*_ZERO_TARGETS[t], kind, id=f"{t}-{kind}", marks=marks)
    for t, kind, marks in [
        ("explicit-1-2-3-5", "ImageStar", ()),
        ("explicit-1-2-3-5", "Star", ()),
        ("batch-0-then-minus-1", "ImageStar", ()),
        ("batch-0-then-minus-1", "Star", ()),
        ("copy-height-and-width", "ImageStar",
         known_bug(7, _ZERO_TARGET_DEFECT, raises=ValueError)),
        ("copy-height-and-width", "Star",
         known_bug(7, _ZERO_TARGET_DEFECT, raises=ValueError)),
        ("copy-channels-then-minus-1", "Star",
         known_bug(7, _ZERO_DIVIDE_DEFECT, raises=RuntimeWarning))]]


@pytest.mark.parametrize("target,may_refuse,kind", _ZERO_CELLS)
def test_reshape_with_a_copied_dimension_holds_every_output(tmp_path, target,
                                                            may_refuse, kind):
    """Reshape(X, target) -> Reshape(1, -1) -> Gemm on a (2, 3, 5) image,
    with a target that copies input dimensions by a 0 entry. The output
    tensor is the input's entries in the same row-major order; the library
    must answer with a set that holds every onnxruntime output, or refuse
    with NotImplementedError. The image rule and the flat rule fail in
    different places (numpy's reshape in ``_reshape_imagestar``, the product
    check of ``_reshape_star``), so each keeps one marked case; the 0 next
    to -1 fails earlier, in ``_resolve_shape``, on the same line for both,
    so it is marked once. The explicit target and a 0 in the batch entry
    (which the walker strips) are the passing neighbours with the same
    model, replay and check; they must answer, a refusal fails them."""
    rng = np.random.default_rng(10)
    nodes = [helper.make_node("Reshape", ["X", "S"], ["R"]),
             helper.make_node("Reshape", ["R", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    inits = {"S": _shape(target), "F": _shape([1, -1]),
             "W2": _dyadic(rng, 3, _N), "b2": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, _SHAPE, [1, 3], inits=inits)
    lo, hi = pixel_box(_C, _H, _W, seed=9)
    if kind == "ImageStar":
        input_set, options = image_set(kind, lo, hi), {}
    else:
        input_set, options = flat_set(kind, lo, hi), {"input_shape": (_C, _H,
                                                                      _W)}
    outputs = replay_onnx(path, samples(lo, hi), _SHAPE)
    if len(outputs[0]) != 3:
        pytest.fail("the reference model does not give 3 outputs")
    what = f"Reshape {target}/Gemm on a {kind}"
    try:
        out = reach_model(load_onnx(path), input_set, "approx", what,
                          **options)
    except NotImplementedError:
        if may_refuse:
            return                  # a clear refusal keeps the promise
        raise
    assert_covered(out, outputs, what)


# --------------------------------------------------------------------------- #
# A target whose first entry is not the batch axis                              #
# --------------------------------------------------------------------------- #
_LEADING_ENTRY_DEFECT = (
    "ONNX Reshape and torch view drop the first entry of every target shape "
    "as the batch axis: a target that moves the batch axis into the first "
    "dimension, such as [-1, 2, 2], [6, 2, 2] or [2, 12] on a (1, 6, 4) "
    "tensor, is reshaped to the remaining entries and the rule raises "
    "ValueError (Cannot reshape Star of dim 24 to shape (2, 2)) where "
    "onnxruntime returns the input's entries in the same order")


_LEADING_CELLS = [
    pytest.param([1, 6, 2, 2], "Star", id="batch-kept-1-6-2-2-Star"),
    pytest.param([1, 6, 2, 2], "Zono", id="batch-kept-1-6-2-2-Zono"),
    pytest.param([1, 24], "Star", id="batch-kept-1-24-Star"),
    pytest.param([1, 24], "Zono", id="batch-kept-1-24-Zono"),
    pytest.param([-1, 2, 2], "Star", id="first-entry-minus-1-2-2-Star",
                 marks=known_bug(7, _LEADING_ENTRY_DEFECT, raises=ValueError)),
    pytest.param([2, 12], "Zono", id="first-entry-2-12-Zono",
                 marks=known_bug(7, _LEADING_ENTRY_DEFECT, raises=ValueError)),
]


@pytest.mark.parametrize("target,kind", _LEADING_CELLS)
def test_reshape_whose_first_entry_is_not_the_batch_holds_every_output(
        tmp_path, target, kind):
    """Reshape(X, target) -> Relu -> Reshape(1, -1) -> Gemm on a flat set
    that stands for a (1, 6, 4) tensor (``input_shape`` (6, 4)). A reshape
    keeps the row-major order whatever the target, so the result must hold
    every onnxruntime output, or the call must be refused with
    NotImplementedError. The targets that keep the batch axis are the
    passing neighbours with the same model, replay and check. One marked
    case per flat rule: ``[-1, 2, 2]`` on the Star, ``[2, 12]`` on the Zono
    (the strip of the first entry in ``_handle_reshape`` is shared; the Box
    rule raises the same way, a Hexatope or Octatope is passed through and
    answered soundly)."""
    rng = np.random.default_rng(13)
    nodes = [helper.make_node("Reshape", ["X", "S"], ["R"]),
             helper.make_node("Relu", ["R"], ["Q"]),
             helper.make_node("Reshape", ["Q", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    inits = {"S": _shape(target), "F": _shape([1, -1]),
             "W2": _dyadic(rng, 3, 24), "b2": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, [1, 6, 4], [1, 3], inits=inits)
    lo, hi = flat_box(24, seed=11)
    outputs = replay_onnx(path, samples(lo, hi), (1, 6, 4))
    if len(outputs[0]) != 3:
        pytest.fail("the reference model does not give 3 outputs")
    what = f"Reshape {target}/Relu/Gemm on a {kind}"
    try:
        out = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                          what, input_shape=(6, 4))
    except NotImplementedError:
        if target[0] in (1,):
            raise                   # the batch-keeping targets must answer
        return                      # a clear refusal keeps the promise
    assert_covered(out, outputs, what)


_LEADING_IMAGE_DEFECT = (
    "ONNX Reshape drops the first entry of every target shape as the batch "
    "axis; on an image set a target whose remaining entries are three, such "
    "as [2, 15, 1, 1] on a (1, 2, 3, 5) tensor, is reshaped to (15, 1, 1) and "
    "numpy raises ValueError (cannot reshape array of size 930 into shape "
    "(15,1,1,31)) where onnxruntime returns the input's entries in the same "
    "order (a target whose remaining entries are not three is flattened, "
    "which is right by accident)")


@pytest.mark.parametrize("target,kind", [
    pytest.param([1, 2, 15, 1], "ImageStar", id="batch-kept-1-2-15-1-ImageStar"),
    pytest.param([1, 2, 15, 1], "ImageZono", id="batch-kept-1-2-15-1-ImageZono"),
    pytest.param([2, 15, 1, 1], "ImageStar", id="first-entry-2-15-1-1-ImageStar",
                 marks=known_bug(7, _LEADING_IMAGE_DEFECT, raises=ValueError)),
    pytest.param([2, 15, 1, 1], "ImageZono", id="first-entry-2-15-1-1-ImageZono",
                 marks=known_bug(7, _LEADING_IMAGE_DEFECT, raises=ValueError)),
])
def test_reshape_of_an_image_whose_first_entry_is_not_the_batch_holds_every_output(
        tmp_path, target, kind):
    """Reshape(X, target) -> Relu -> Reshape(1, -1) -> Gemm on a (2, 3, 5)
    image set. The image rules have no product check of their own, so the
    batch strip of ``_handle_reshape`` fails in numpy's reshape inside
    ``_reshape_imagestar`` and ``_reshape_imagezono``: one marked case each.
    The batch-keeping target of the same rank, which makes a (2, 15, 1)
    image, is the passing neighbour with the same model, replay and check;
    it must answer, a refusal fails it."""
    rng = np.random.default_rng(13)
    nodes = [helper.make_node("Reshape", ["X", "S"], ["R"]),
             helper.make_node("Relu", ["R"], ["Q"]),
             helper.make_node("Reshape", ["Q", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    inits = {"S": _shape(target), "F": _shape([1, -1]),
             "W2": _dyadic(rng, 3, _N), "b2": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, _SHAPE, [1, 3], inits=inits)
    lo, hi = pixel_box(_C, _H, _W, seed=9)
    outputs = replay_onnx(path, samples(lo, hi), _SHAPE)
    what = f"Reshape {target}/Relu/Gemm on a {kind}"
    try:
        out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                          what)
    except NotImplementedError:
        if target[0] == 1:
            raise                   # the batch-keeping target must answer
        return                      # a clear refusal keeps the promise
    assert_covered(out, outputs, what)


class _ViewWithoutBatch(nn.Module):
    """A (1, 6, 4) input viewed by one spelling, a ReLU, a flattening
    reshape and a Linear."""

    def __init__(self, how):
        super().__init__()
        self.fc = nn.Linear(24, 3)
        self.how = how

    def forward(self, x):
        if self.how == "x.view(6, 2, 2)":
            h = x.view(6, 2, 2)
        else:
            h = x.view(1, 6, 2, 2)
        return self.fc(F.relu(h).reshape(1, -1))


@pytest.mark.parametrize("how", [
    "x.view(1, 6, 2, 2)",
    pytest.param("x.view(6, 2, 2)",
                 marks=known_bug(7, _LEADING_ENTRY_DEFECT, raises=ValueError))])
def test_torch_view_whose_first_entry_is_not_the_batch_holds_every_output(how):
    """The torch route of the same defect: ``x.view(6, 2, 2)`` of a flat Star
    that stands for a (1, 6, 4) tensor (``input_shape`` (6, 4)), then a ReLU
    and a Linear. The torch view branch strips the first entry before
    ``_handle_reshape`` is called (it lets only that entry be a traced size),
    so a fix in the ONNX branch alone leaves it. ``x.view(1, 6, 2, 2)`` is
    the passing neighbour with the same model, replay and check; it must
    answer, a refusal fails it."""
    model = _dyadic_module(_ViewWithoutBatch(how), seed=7)
    lo, hi = flat_box(24, seed=3)
    what = f"{how}/ReLU/Linear on a Star"
    try:
        out = reach_model(model, flat_set("Star", lo, hi), "approx", what,
                          input_shape=(6, 4))
    except NotImplementedError:
        if how == "x.view(1, 6, 2, 2)":
            raise                   # the batch-keeping view must answer
        return                      # a clear refusal keeps the promise
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, 6, 4)), what)
