"""Soundness cells for ONNX BatchNormalization on flat sets whose tensor is
(N, C, L) or (N, C, H, W): the shape-aware graph handler
``n2v.nn.reach._handle_batchnorm_flat``.

The handler expands the per-channel scale and shift to one value per flat
entry (broadcast along the axes after the channel axis) and applies the
elementwise affine map; a Box takes the map with a rounding margin. It
returns nothing (and the dispatcher's BatchNorm takes over) for image sets,
for a set without a tracked tensor shape and for the plain (N, C) tensor.
BatchNorm on image sets, and BatchNorm1d / BatchNorm2d handed to the
dispatcher directly, are cells of ``test_soundness_batchnorm.py``; the
rounding margin of the Box fold on non-dyadic parameters is checked in exact
arithmetic there (the Box on a channel-by-length tensor, through this
handler) and in ``test_box_rounding_regressions.py``.

Oracle: onnxruntime replay of the same ONNX file on member points of the
input set. The data are dyadic and every channel has var + epsilon equal to
the square of a power of two (epsilon = 2^-10 is not zero), so float32 and
the real-number layer agree exactly and no slack is needed beyond the LP
tolerance of the membership check (``cell_support.assert_covered``, which
decides membership by an LP on the rows of each returned set).

Layout: a flat set holds the tensor's own row-major order (channel, then
the axes after it), so channel c owns a contiguous block of L entries; a
result that gives entry i the parameters of channel i mod C instead of
i // L misses the outputs.
"""

import numpy as np
import pytest
from onnx import helper

from n2v.utils.model_loader import load_onnx
from tests.cell_support import (assert_covered, constrained_star, flat_set,
                                image_set, reach_model, replay_onnx, samples,
                                save_onnx)
from tests.known_bugs import DID_NOT_RAISE, known_bug

EPS = 2.0 ** -10


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _i64(*values):
    return np.array(values, dtype=np.int64)


def _tensor_box(shape, crossing=(0, 4)):
    """Dyadic bounds ``(lo, hi)``: every entry its own interval, the entries
    in ``crossing`` straddle zero, the others lie on one side of it."""
    n = int(np.prod(shape))
    lo, hi = np.empty(n), np.empty(n)
    for i in range(n):
        half = (1 + i % 3) / 8.0
        centre = (i + 2) / 4.0 * (-1.0 if i % 3 == 2 else 1.0)
        if i in crossing:
            lo[i], hi[i] = -(2 + i % 5) / 8.0, (1 + i % 3) / 8.0
        else:
            lo[i], hi[i] = centre - half, centre + half
    return lo.reshape(shape), hi.reshape(shape)


def _bn_params(channels):
    """Per-channel (scale, bias, mean, var), all different across channels:
    a negative scale, a zero scale on the fourth channel when there is one,
    and on the second channel a bias equal to scale * mean / sqrt(var + eps)
    (the folded shift is then zero). var + eps is (1/2)^2, 1^2 or 2^2."""
    sd = np.array([0.5, 1.0, 2.0, 1.0, 0.5])[:channels]
    scale = np.array([1.5, -0.75, 2.0, 0.0, -1.25])[:channels]
    mean = np.array([0.25, -0.5, 1.0, 0.75, -1.5])[:channels]
    bias = np.array([-0.5, 0.0, 0.375, 1.25, 0.625])[:channels]
    if channels > 1:
        bias[1] = scale[1] * mean[1] / sd[1]
    return scale, bias, mean, sd ** 2 - EPS


def _bn_nodes(channels, before=None):
    scale, bias, mean, var = _bn_params(channels)
    inits = {"g": _f32(scale), "b": _f32(bias), "m": _f32(mean),
             "v": _f32(var)}
    nodes = []
    x = "X"
    if before is not None:
        nodes.append(helper.make_node(before, ["X"], ["A"]))
        x = "A"
    nodes.append(helper.make_node("BatchNormalization",
                                  [x, "g", "b", "m", "v"], ["Y"],
                                  epsilon=EPS))
    return nodes, inits


def _check(tmp_path, nodes, inits, in_shape, out_shape, kind, method, what,
           box=None, constrained=False):
    lo, hi = box if box is not None else _tensor_box(tuple(in_shape[1:]))
    path = save_onnx(tmp_path, nodes, in_shape, out_shape, inits=inits)
    if constrained:
        input_set, points = constrained_star(lo, hi)
    else:
        input_set, points = flat_set(kind, lo, hi), samples(lo, hi)
    outputs = replay_onnx(path, points, in_shape)
    sets = reach_model(load_onnx(path), input_set, method, what,
                       input_shape=tuple(in_shape[1:]))
    assert_covered(sets, outputs, what)


_KINDS = [("Star", "approx"), ("Star", "exact"), ("Zono", "approx"),
          ("Box", "approx")]


@pytest.mark.parametrize("shape", [[1, 3, 4], [1, 2, 3, 5]])
@pytest.mark.parametrize("kind,method", _KINDS)
def test_batchnorm_of_a_flat_set_holds_every_output(tmp_path, kind, method,
                                                    shape):
    """BatchNormalization on a flat set standing for a (1, 3, 4) and a
    (1, 2, 3, 5) tensor (BatchNorm1d and BatchNorm2d after loading), every
    channel with its own scale, bias, mean and variance. Catches the
    per-channel values tiled (channel i mod C) instead of broadcast over the
    axes after the channel (channel i // L), the shift built from the scale,
    and a Star or Zono path that applies the plain C-sized map to a longer
    vector."""
    nodes, inits = _bn_nodes(shape[1])
    _check(tmp_path, nodes, inits, shape, shape, kind, method,
           f"BatchNorm on {shape}, {kind} {method}")


@pytest.mark.parametrize("kind,method", _KINDS)
def test_batchnorm_after_an_exact_split_holds_every_output(tmp_path, kind,
                                                           method):
    """Relu then BatchNormalization on a (1, 3, 4) tensor with two entries
    across zero. With the exact method the BatchNorm receives a list of
    pieces, every one of which must be mapped (a handler that maps only the
    first set misses outputs); with the approx method it receives a Star or
    Zono with more predicate variables or generators than entries (the
    relaxation's), whose rows are not box rows. A Box never reaches the
    handler as a list here: its ReLU does not split. Catches the first piece
    mapped for every piece, and generators scaled by |scale| while the
    centre is mapped right (the relaxation's rows make the sign of each
    variable matter)."""
    nodes, inits = _bn_nodes(3, before="Relu")
    _check(tmp_path, nodes, inits, [1, 3, 4], [1, 3, 4], kind, method,
           f"Relu > BatchNorm, {kind} {method}")


@pytest.mark.parametrize("method", ["approx", "exact"])
def test_batchnorm_of_a_star_with_a_binding_row_holds_every_output(tmp_path,
                                                                   method):
    """Relu then BatchNormalization on a Star cut by a row that is not a box
    row, so its predicate polytope is not symmetric: the elementwise map must
    keep the meaning of every predicate variable under the rows, which stay.
    Catches a map that scales a channel's generators by |scale| (as a
    radius) while mapping the centre right: under a negative scale the
    channel's variables change sign, which a box Star, symmetric in every
    variable, cannot show; with the exact method also the first piece of the
    Relu's split mapped for every piece. A result that drops the rows is a
    superset and passes: this cell guards the variables, not the rows."""
    nodes, inits = _bn_nodes(3, before="Relu")
    _check(tmp_path, nodes, inits, [1, 3, 4], [1, 3, 4], "Star", method,
           f"BatchNorm on a constrained Star, {method}", constrained=True)


_CHAIN_KINDS = _KINDS + [("Hexatope", "approx"), ("Octatope", "approx")]


@pytest.mark.parametrize("kind,method,tensor", [
    pytest.param(kind, method, (1, 3, 4), id=f"{kind}-{method}")
    for kind, method in _CHAIN_KINDS] + [
    pytest.param(kind, method, (1, 2, 2, 3), id=f"{kind}-{method}-2x2x3")
    for kind, method in _CHAIN_KINDS])
def test_batchnorm_between_two_affine_layers_holds_every_output(
        tmp_path, kind, method, tensor):
    """Gemm (mixes all six inputs), Reshape to (1, 3, 4) or (1, 2, 2, 3)
    (BatchNorm1d or BatchNorm2d after loading), BatchNormalization, Relu,
    Flatten, Gemm with a few weights per row: the twelve entries of the
    BatchNorm output, with intervals wider than 2, are consumed downstream,
    so a parameter applied to the wrong entry changes the outputs. A
    Hexatope or Octatope skips the Relu (their ReLU relaxation takes seconds
    to a minute here and is not this operation)."""
    rng = np.random.default_rng(23)
    lo, hi = _tensor_box((6,), crossing=())
    W1 = rng.integers(-2, 3, size=(12, 6)) / 2.0
    b1 = rng.integers(-4, 5, size=12) / 4.0
    W2 = np.zeros((5, 12))
    for r in range(5):
        cols = rng.choice(12, size=4, replace=False)
        W2[r, cols] = rng.choice([-1.0, -0.5, 0.5, 1.0, 2.0], size=4)
    scale, bias, mean, var = _bn_params(tensor[1])
    inits = {"W1": _f32(W1), "b1": _f32(b1), "shape": _i64(*tensor),
             "g": _f32(scale), "b": _f32(bias), "m": _f32(mean),
             "v": _f32(var), "W2": _f32(W2)}
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["Z"], transB=1),
             helper.make_node("Reshape", ["Z", "shape"], ["T"]),
             helper.make_node("BatchNormalization", ["T", "g", "b", "m", "v"],
                              ["N"], epsilon=EPS),
             helper.make_node("Relu", ["N"], ["R"]),
             helper.make_node("Flatten", ["R"], ["F"], axis=1),
             helper.make_node("Gemm", ["F", "W2"], ["Y"], transB=1)]
    if kind in ("Hexatope", "Octatope"):
        del nodes[3]
        nodes[3] = helper.make_node("Flatten", ["N"], ["F"], axis=1)
    _check(tmp_path, nodes, inits, [1, 6], [1, 5], kind, method,
           f"Gemm > Reshape {tensor} > BatchNorm > Gemm, {kind} {method}",
           box=(lo, hi))


@pytest.mark.parametrize("kind,shape", [
    pytest.param(kind, [1, 3, 4], id=f"{kind}-1x3x4")
    for kind in ("Star", "Zono", "Box", "Hexatope", "Octatope")] + [
    pytest.param("Box", [1, 1, 4], id="Box-1x1x4")])
def test_batchnorm_of_a_flat_set_without_a_tensor_shape_refuses_or_holds_every_output(  # noqa: E501
        tmp_path, kind, shape):
    """BatchNormalization of a (1, 3, 4) or (1, 1, 4) tensor reached from a
    flat set with no ``input_shape``: the walker tracks no tensor shape (its
    dummy forward of a (1, 12) or (1, 4) tensor fails), so the graph handler
    declines and the dispatcher's per-feature rule meets a set longer than
    the parameter count. The promise is a refusal (``ValueError`` or
    ``NotImplementedError``) or a result that holds every onnxruntime
    output. Today every rule refuses the three channels (the map has fewer
    columns than the set has entries), and the Box rule broadcasts the one
    channel's parameters over the (1, 1, 4) tensor, which is its image.
    Catches a fallback that guesses a layout (the per-channel values
    repeated over the vector, channel i mod C) and answers."""
    nodes, inits = _bn_nodes(shape[1])
    lo, hi = _tensor_box(tuple(shape[1:]))
    path = save_onnx(tmp_path, nodes, shape, shape, inits=inits)
    what = f"BatchNorm on {shape} without a tensor shape, {kind}"
    try:
        sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                           what)
    except (NotImplementedError, ValueError):
        return
    assert_covered(sets, replay_onnx(path, samples(lo, hi), shape), what)


@pytest.mark.parametrize("kind,method", _KINDS)
def test_batchnorm_of_a_plain_vector_holds_every_output(tmp_path, kind,
                                                        method):
    """BatchNormalization on a (1, 5) tensor, where the flat set has one entry
    per channel: the handler returns nothing and the dispatcher's BatchNorm
    answers. Neighbour of the (N, C, L) cells: the same parameters and
    reference with the other path."""
    nodes, inits = _bn_nodes(5)
    _check(tmp_path, nodes, inits, [1, 5], [1, 5], kind, method,
           f"BatchNorm on (1, 5), {kind} {method}")


@pytest.mark.parametrize("kind,method", [("ImageStar", "approx"),
                                         ("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
def test_batchnorm_of_an_image_set_in_the_graph_holds_every_output(
        tmp_path, kind, method):
    """Relu then BatchNormalization on a (2, 3, 5) image set: the flat handler
    must leave an image set to the dispatcher's channel-aware BatchNorm,
    because the image set stores (H, W, C) and the flat handler expands the
    parameters in the tensor's (C, H, W) order; taking it would give pixel k
    the parameters of the wrong channel. Compared in the stored order."""
    shape = [1, 2, 3, 5]
    nodes, inits = _bn_nodes(2, before="Relu")
    lo, hi = _tensor_box((2, 3, 5))
    path = save_onnx(tmp_path, nodes, shape, shape, inits=inits)
    outputs = replay_onnx(path, samples(lo, hi), shape)
    sets = reach_model(load_onnx(path), image_set(kind, lo, hi), method,
                       f"Relu > BatchNorm on a {kind}, {method}",
                       input_shape=(2, 3, 5))
    assert_covered(sets, outputs, f"BatchNorm on a {kind}, {method}",
                   out_chw=(2, 3, 5))


_TRAINING_MODE_ANSWERED = known_bug(
    23, "an ONNX BatchNormalization with training_mode=1 and only the output "
        "Y is loaded as an inference-mode BatchNorm (the vendored converter "
        "does not read training_mode) and answered with the running "
        "statistics; onnxruntime and onnx's full check reject the model "
        "(training mode needs the three outputs), and in training mode Y is "
        "normalised by the batch's own statistics", raises=DID_NOT_RAISE)


@pytest.mark.parametrize("opset,attributes,outputs", [
    pytest.param(15, {"training_mode": 0}, 1, id="opset15-training_mode-0"),
    pytest.param(15, {"training_mode": 1}, 1, id="opset15-training_mode-1",
                 marks=_TRAINING_MODE_ANSWERED),
    pytest.param(7, {"spatial": 1}, 1, id="opset7-spatial-1")])
def test_batchnorm_onnx_attributes_answer_only_where_onnxruntime_answers(
        tmp_path, opset, attributes, outputs):
    """BatchNormalization of a (1, 3, 4) tensor with the attributes that
    change its meaning: ``training_mode`` (opset 14 and later: normalise by
    the batch's own statistics, which needs the three outputs) and
    ``spatial`` of opset 7 (a model of opset 7 is upgraded at load). The
    promise: when onnxruntime refuses the model, reach refuses it too
    (``ValueError`` or ``NotImplementedError``); when it answers, reach
    refuses or holds every onnxruntime output (LP on the rows). Catches a
    loader that drops ``training_mode`` and answers for the inference-mode
    layer, and an upgraded opset-7 layer read wrongly. ``training_mode=1``
    with only Y is answered today (marked); ``training_mode=0`` is its
    neighbour on the same reference. (``spatial=0`` of opsets 7 and 8, and
    ``training_mode=1`` with the three outputs, are refused by the vendored
    converter with ``NotImplementedError`` at load.)"""
    inits = dict(zip("gbmv", (_f32(v) for v in _bn_params(3))))
    names = ["Y", "rm", "rv"][:outputs]
    nodes = [helper.make_node("BatchNormalization", ["X", "g", "b", "m", "v"],
                              names, epsilon=EPS, **attributes)]
    path = save_onnx(tmp_path, nodes, [1, 3, 4], [1, 3, 4], inits=inits,
                     opset=opset)
    lo, hi = _tensor_box((3, 4))
    what = f"BatchNormalization opset {opset} {attributes}, {outputs} outputs"
    try:
        outputs_ort = replay_onnx(path, samples(lo, hi), [1, 3, 4])
    except Exception:  # noqa: BLE001 - onnxruntime refuses the model
        outputs_ort = None
    if outputs_ort is None:
        with pytest.raises((ValueError, NotImplementedError)):
            reach_model(load_onnx(path), flat_set("Star", lo, hi), "approx",
                        what, input_shape=(3, 4))
        return
    try:
        sets = reach_model(load_onnx(path), flat_set("Star", lo, hi),
                           "approx", what, input_shape=(3, 4))
    except (ValueError, NotImplementedError):
        return
    assert_covered(sets, outputs_ort, what)

