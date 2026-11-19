"""Reach coverage: a cell (operation, set type, method) either answers with
sets that contain every model output, or refuses (DESIGN sections 2, 17, 19).

This file holds the known gaps: cells that answer wrongly today, or answer
where DESIGN says refuse. Each is written to the promise and carries a strict
xfail naming the PLAN step that fixes it. The neighbouring cells that are right
today run unmarked, so each test's reference and layout are checked on a
passing case. The table of every supported cell arrives with PLAN step 14.

It also holds the refusals that no operation's own cells hold: an operation
for which n2v has no rule at all (ONNX Clip, which ``load_onnx`` turns into
``nn.ReLU6`` or ``OnnxClip``; torch functions next to the ones the graph walk
converts, such as ``F.relu6`` and ``torch.clamp``), and ``method='exact'`` on
a set type that has no exact rules. Each is checked on every set type: the
promise is a refusal, or an answer that holds every true output.

Reference: onnxruntime on dyadic inputs and weights, where float32 evaluation
is exact, or a float64 graph or torch replay in float64 where it is not.
Membership is one LP on the set's own rows (``tests/_oracles/membership.py``);
the only slack is the solver tolerance. A refusal is ``NotImplementedError``
(``UnsupportedError`` will subclass it); a cell that only promises "refuses or
holds every output" accepts ``ValueError`` (``ModelError``) too.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Star
from n2v.utils.model_loader import load_onnx
from tests._oracles.membership import hwc
from tests.cell_support import (assert_covered, flat_set, image_set, pixel_box,
                                reach_model, reach_one_layer, replay_onnx,
                                replay_torch, samples, save_onnx)
from tests.known_bugs import DID_NOT_RAISE, known_bug

# --------------------------------------------------------------------------- #
# Softmax on an image set                                                       #
# --------------------------------------------------------------------------- #
_C, _H, _W = 2, 3, 5                    # an asymmetric image: 30 entries

_SOFTMAX_AS_ONE_VECTOR = known_bug(
    7, "Softmax on an ImageStar is computed over all H*W*C entries as one "
       "vector, whatever the axis; model outputs escape the result")


@pytest.mark.parametrize("kind,axis", [
    pytest.param("ImageStar", 1, marks=_SOFTMAX_AS_ONE_VECTOR),
    pytest.param("ImageStar", -1, marks=_SOFTMAX_AS_ONE_VECTOR),
    ("ImageStar", 3),
    ("ImageZono", 1),
    ("ImageZono", -1),
])
def test_softmax_on_an_image_set_refuses_or_is_sound(tmp_path, kind, axis):
    """Softmax over the channels (axis 1), the width (axis -1 and 3) of a
    (1, 2, 3, 5) tensor, then Flatten. No rule computes a softmax along one
    axis of an image set, so the cell refuses until one exists. The ImageStar
    result today, one softmax over all 30 entries, misses all 128 model
    outputs (largest distance 0.93 over axis 1, 0.71 over axis -1). The graph
    is float64, so the replay is a float64 reference and the only slack is
    the LP tolerance."""
    path = save_onnx(tmp_path,
                     [helper.make_node("Softmax", ["X"], ["S"], axis=axis),
                      helper.make_node("Flatten", ["S"], ["Y"])],
                     [1, _C, _H, _W], [1, _C * _H * _W], name="softmax",
                     in_type="DOUBLE", out_type="DOUBLE")
    lo, hi = pixel_box(_C, _H, _W)
    what = f"Softmax(axis={axis}) on {kind}"
    try:
        sets = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx", what)
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, _C, _H, _W]),
                   what)


@pytest.mark.parametrize("kind", [
    pytest.param("ImageStar", marks=known_bug(
        7, "the Softmax layer on an ImageStar returns a flat Star computed "
           "over all H*W*C entries", raises=DID_NOT_RAISE)),
    "ImageZono",
])
@pytest.mark.parametrize("dim", [1, -1])
def test_softmax_layer_on_an_image_set_is_refused(kind, dim):
    lo, hi = pixel_box(_C, _H, _W)
    with pytest.raises(NotImplementedError):
        reach_layer(nn.Softmax(dim=dim), [image_set(kind, lo, hi)], "approx")


# --------------------------------------------------------------------------- #
# Rounding operations on an image set, read by a later Flatten                  #
# --------------------------------------------------------------------------- #
_FLAT_ZONO_IN_HWC = known_bug(
    7, "Round, Floor, Ceil and Sign on an ImageZono return a flat Zono in "
       "height-width-channel order, which the next Flatten reads as "
       "channel-height-width")


@pytest.mark.parametrize("kind", [
    pytest.param("ImageZono", marks=_FLAT_ZONO_IN_HWC),
    "ImageStar",
])
@pytest.mark.parametrize("op", ["Round", "Floor", "Ceil", "Sign"])
def test_rounding_op_on_an_image_set_keeps_the_pixel_order(tmp_path, op, kind):
    """op, Flatten, then an affine layer whose output i reads pixels i and
    i + 7 of the flattened tensor, so a permuted pixel order moves the
    outputs. Inputs, weights and the rounded values are dyadic."""
    n = _C * _H * _W
    weight = (np.eye(n) + 0.5 * np.roll(np.eye(n), 7, axis=1)).astype(np.float32)
    bias = (np.arange(n) % 3 - 1).astype(np.float32)
    path = save_onnx(tmp_path,
                     [helper.make_node(op, ["X"], ["A"]),
                      helper.make_node("Flatten", ["A"], ["F"]),
                      helper.make_node("Gemm", ["F", "W", "b"], ["Y"], transB=1)],
                     [1, _C, _H, _W], [1, n], {"W": weight, "b": bias}, name=op)
    lo, hi = pixel_box(_C, _H, _W)
    what = f"{op} on {kind}, then Flatten and Gemm"
    sets = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx", what)
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, _C, _H, _W]),
                   what)


# --------------------------------------------------------------------------- #
# AdaptiveAvgPool2d with an output larger than one pixel                        #
# --------------------------------------------------------------------------- #
_POOLED_AS_GLOBAL = known_bug(
    7, "AdaptiveAvgPool2d never reads output_size: every size is computed as "
       "global pooling to one pixel")


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("shape,output_size", [
    pytest.param((4, 6, 2), (2, 3), marks=_POOLED_AS_GLOBAL, id="4x6-to-2x3"),
    pytest.param((4, 4, 2), (2, 2), marks=_POOLED_AS_GLOBAL, id="4x4-to-2x2"),
    pytest.param((4, 6, 2), (1, 1), id="4x6-to-1x1"),
    pytest.param((4, 6, 2), (None, 1), marks=_POOLED_AS_GLOBAL,
                 id="4x6-to-4x1"),
])
def test_adaptive_average_pooling_reads_its_output_size(kind, shape,
                                                        output_size):
    """An output size that divides the input is average pooling over equal
    blocks; ``None`` keeps that axis (torch's own reading), so (None, 1) is
    the mean over the width alone. The reference is the block mean in
    numpy."""
    height, width, channels = shape
    out_h, out_w = (height if output_size[0] is None else output_size[0],
                    width if output_size[1] is None else output_size[1])
    lo, hi = pixel_box(channels, height, width)
    what = f"AdaptiveAvgPool2d({output_size}) on a {height}x{width} {kind}"
    sets = reach_layer(nn.AdaptiveAvgPool2d(output_size),
                       [image_set(kind, lo, hi)], "approx")
    assert len(sets) >= 1, f"{what}: reach returned no sets"
    n_out = out_h * out_w * channels
    assert [s.dim for s in sets] == [n_out] * len(sets), (
        f"{what}: the result has {sets[0].dim} entries, the layer's output "
        f"has {n_out}")
    blocks = (channels, out_h, height // out_h, out_w, width // out_w)
    outputs = [hwc(x.reshape(blocks).mean(axis=(2, 4)), channels, out_h, out_w)
               for x in samples(lo, hi)]
    assert_covered(sets, outputs, what, stored_order=True)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@known_bug(7, "AdaptiveAvgPool2d with an output size that does not divide "
              "the input is computed as global pooling", raises=DID_NOT_RAISE)
def test_adaptive_average_pooling_refuses_uneven_blocks(kind):
    """4x4 to 3x3 pools over overlapping blocks of unequal size; no rule."""
    lo, hi = pixel_box(2, 4, 4)
    with pytest.raises(NotImplementedError):
        reach_layer(nn.AdaptiveAvgPool2d((3, 3)), [image_set(kind, lo, hi)],
                    "approx")


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@known_bug(7, "AdaptiveAvgPool2d never reads output_size (POOL-1): an output "
              "size of 0, which torch answers with an empty tensor, comes "
              "back as a pool to one pixel of C entries")
def test_adaptive_average_pooling_to_an_empty_output_refuses_or_has_no_entries(
        kind):
    """``AdaptiveAvgPool2d(0)``: 0 is a size, not an absent one (a fix that
    reads ``output_size or 1`` turns it into the global pool), and torch maps
    the (1, 2, 4, 6) tensor to a (1, 2, 0, 0) tensor. The promise is a
    refusal or a result whose sets have no entries."""
    lo, hi = pixel_box(2, 4, 6)
    what = f"AdaptiveAvgPool2d(0) on {kind}"
    try:
        sets = reach_one_layer(nn.AdaptiveAvgPool2d(0),
                               [image_set(kind, lo, hi)], "approx", what)
    except (NotImplementedError, ValueError):
        return
    assert [s.dim for s in sets] == [0] * len(sets), (
        f"{what}: the result has {[s.dim for s in sets]} entries, the layer's "
        f"output has none")


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@known_bug(7, "AdaptiveAvgPool2d never reads output_size (POOL-1): "
              "AdaptiveAvgPool2d(None), whose torch forward raises TypeError, "
              "is answered as a pool to one pixel", raises=DID_NOT_RAISE)
def test_adaptive_average_pooling_without_an_output_size_is_refused(kind):
    """``AdaptiveAvgPool2d(None)`` is no layer torch can apply (its forward
    raises ``TypeError``); an answer is the image of some other function."""
    lo, hi = pixel_box(2, 4, 6)
    with pytest.raises((NotImplementedError, ValueError)):
        reach_layer(nn.AdaptiveAvgPool2d(None), [image_set(kind, lo, hi)],
                    "approx")


# --------------------------------------------------------------------------- #
# "exact" on an operation that has no exact rule                                #
# --------------------------------------------------------------------------- #
_EXACT_IS_A_RELAXATION = known_bug(
    18, "method='exact' runs the over-approximating rule of an operation "
        "that has no exact one and returns it as the exact result",
    raises=DID_NOT_RAISE)

# op: (nodes, initializers, input tensor shape, reach keywords)
_NO_EXACT_RULE = {
    "Sigmoid": ([helper.make_node("Sigmoid", ["X"], ["Y"])], {}, [1, 1], {}),
    "Tanh": ([helper.make_node("Tanh", ["X"], ["Y"])], {}, [1, 1], {}),
    "Sin": ([helper.make_node("Sin", ["X"], ["Y"])], {}, [1, 1], {}),
    "Cos": ([helper.make_node("Cos", ["X"], ["Y"])], {}, [1, 1], {}),
    "Pow": ([helper.make_node("Pow", ["X", "p"], ["Y"])],
            {"p": np.array(3.0, dtype=np.float32)}, [1, 1], {}),
    "Round": ([helper.make_node("Round", ["X"], ["Y"])], {}, [1, 1], {}),
    "Floor": ([helper.make_node("Floor", ["X"], ["Y"])], {}, [1, 1], {}),
    "Ceil": ([helper.make_node("Ceil", ["X"], ["Y"])], {}, [1, 1], {}),
    "Softmax": ([helper.make_node("Softmax", ["X"], ["Y"], axis=-1)], {},
                [1, 2], {}),
    # The product of two computed values X X. A Mul or Div of two computed
    # values is marked by test_soundness_set_arith_cells.py::
    # test_exact_product_or_quotient_is_refused on the same route.
    "MatMul": ([helper.make_node("MatMul", ["X", "X"], ["Y"])], {}, [2, 2],
               {"input_shape": (2, 2)}),
}


@pytest.mark.parametrize("op", sorted(_NO_EXACT_RULE))
@_EXACT_IS_A_RELAXATION
def test_exact_refuses_an_operation_with_no_exact_rule(tmp_path, op):
    """Each graph returns one Star under "exact" today, the same enclosure
    "approx" returns. The name "exact" states a guarantee the result does not
    have, so the call must refuse and name "approx"."""
    nodes, inits, shape, kwargs = _NO_EXACT_RULE[op]
    model = load_onnx(save_onnx(tmp_path, nodes, shape, shape, inits, name=op))
    n = int(np.prod(shape))
    star = Star.from_bounds(np.full((n, 1), -0.5), np.ones((n, 1)))
    with pytest.raises(NotImplementedError):
        reach_pytorch_model(model, star, method="exact", **kwargs)


@pytest.mark.parametrize("layer", [nn.Sigmoid(), nn.Tanh(), nn.Softmax(dim=-1)],
                         ids=["Sigmoid", "Tanh", "Softmax"])
@_EXACT_IS_A_RELAXATION
def test_exact_refuses_a_layer_with_no_exact_rule(layer):
    """Replaces ``test_exact_warns_and_uses_approx`` (unit Sigmoid and Tanh)
    and ``test_exact_falls_back_to_approx`` (soundness Sigmoid and Tanh),
    which asserted the downgrade."""
    star = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    with pytest.raises(NotImplementedError):
        reach_layer(layer, [star], "exact")


@pytest.mark.parametrize("op", ["Relu", "Sign"])
def test_exact_on_a_piecewise_linear_operation_is_sound(tmp_path, op):
    """The other side of the rule: these have exact rules and keep answering.
    The quarter grid of the box includes 0, where Sign is 0."""
    path = save_onnx(tmp_path, [helper.make_node(op, ["X"], ["Y"])], [1, 2],
                     [1, 2], name=op)
    lo, hi = np.array([-0.5, -1.0]), np.array([0.5, 1.0])
    what = f"{op} on Star under exact"
    sets = reach_model(load_onnx(path), flat_set("Star", lo, hi), "exact", what)
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, 2]), what)


# --------------------------------------------------------------------------- #
# Nearest-neighbour upsampling as PyTorch exports it from opset 13              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("roi", [
    pytest.param("", id="roi-omitted", marks=known_bug(
        7, "a Resize whose roi input is omitted reaches the reach rule with "
           "scales as a keyword, which the rule does not read: the layer is "
           "refused", raises=NotImplementedError)),
    pytest.param("roi", id="roi-given"),
])
def test_nearest_upsample_by_two(tmp_path, roi, kind):
    """Resize(mode=nearest, scales=[1, 1, 2, 2]). PyTorch's exporter omits
    the roi input from opset 13 on; the hand-written form names an empty one.
    Both are the same layer and copy every pixel into a 2x2 block."""
    channels, height, width = 2, 2, 3
    inits = {"scales": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)}
    if roi:
        inits[roi] = np.array([], dtype=np.float32)
    path = save_onnx(tmp_path,
                     [helper.make_node("Resize", ["X", roi, "scales"], ["Y"],
                                       mode="nearest", nearest_mode="floor",
                                       coordinate_transformation_mode="asymmetric")],
                     [1, channels, height, width],
                     [1, channels, 2 * height, 2 * width], inits, name="resize")
    lo, hi = pixel_box(channels, height, width)
    what = f"nearest Resize by 2 on {kind}"
    sets = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx", what)
    outputs = [hwc(y, channels, 2 * height, 2 * width) for y in
               replay_onnx(path, samples(lo, hi), [1, channels, height, width])]
    assert_covered(sets, outputs, what, stored_order=True)


# --------------------------------------------------------------------------- #
# Cast to a type that changes the value                                         #
# --------------------------------------------------------------------------- #
_CAST_IS_IDENTITY = known_bug(
    7, "Cast is reached as the identity whatever the target type; a cast to "
       "an integer, boolean or half-precision type changes the value")

# x0 crosses zero, x1 lies strictly between two integers, and x2 lies
# strictly between two neighbouring half-precision values.
_CAST_LO = np.array([-1.5, 0.25, 1 + 2.0 ** -12])
_CAST_HI = np.array([1.5, 0.75, 1 + 2.0 ** -11])


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
@pytest.mark.parametrize("to", [
    pytest.param("INT32", marks=_CAST_IS_IDENTITY),
    pytest.param("INT64", marks=_CAST_IS_IDENTITY),
    pytest.param("BOOL", marks=_CAST_IS_IDENTITY),
    pytest.param("FLOAT16", marks=_CAST_IS_IDENTITY),
    "DOUBLE",
    "FLOAT",
])
def test_cast_refuses_or_is_sound(tmp_path, to, kind):
    """One Cast node, whose output is the graph's output in the target type
    (onnxruntime's values, read as float64). Truncation sends x1 to 0, the
    boolean cast sends it to 1 and half precision sends x2 to 1; none of
    these is in the input interval the identity returns. A float32 or
    float64 target is the identity, so its cell catches a Cast rule that
    changes the set; one node and not a cast there and back, because a rule
    that undoes itself (a negation) gives the identity again after two."""
    path = save_onnx(tmp_path,
                     [helper.make_node("Cast", ["X"], ["Y"],
                                       to=getattr(TensorProto, to))],
                     [1, 3], [1, 3], name=f"cast_{to}", out_type=to)
    what = f"Cast to {to} on {kind}"
    try:
        sets = reach_model(load_onnx(path), flat_set(kind, _CAST_LO, _CAST_HI),
                           "approx", what)
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path, samples(_CAST_LO, _CAST_HI), [1, 3]),
                   what)


# --------------------------------------------------------------------------- #
# "exact" on a set type that has no exact rules                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", [
    pytest.param(kind, marks=known_bug(
        16, "reach_pytorch_model refuses method='exact' on a set type with no "
            "exact rules with ValueError ('... only supports approx'); the "
            "settled refusal of an unsupported combination is "
            "NotImplementedError (UnsupportedError naming 'approx', DESIGN "
            "section 7)", raises=ValueError))
    for kind in ("Zono", "Box", "Hexatope")])
def test_exact_on_a_set_type_without_exact_rules_is_refused(tmp_path, kind):
    """The model entry checks the method against the set type before the
    graph walk, in one branch for Zono (an ImageZono is a Zono), one for Box
    and one for Hexatope and Octatope. A Zono kernel answers under "exact"
    with its relaxation (REACH-8), so a check that let "exact" through would
    return a result that is not the exact image under a name that promises
    it. The passing neighbour on the same graph and replay, where the set
    type has exact rules, is ``test_exact_on_a_piecewise_linear_operation_is_
    sound[Relu]``."""
    path = save_onnx(tmp_path, [helper.make_node("Relu", ["X"], ["Y"])],
                     [1, 2], [1, 2], name="relu")
    lo, hi = np.array([-0.5, -1.0]), np.array([0.5, 1.0])
    with pytest.raises(NotImplementedError):
        reach_pytorch_model(load_onnx(path), flat_set(kind, lo, hi),
                            method="exact")


# --------------------------------------------------------------------------- #
# Operations with no reach rule: refused, or every output covered              #
# --------------------------------------------------------------------------- #
# One (1, 2, 2, 3) tensor whose entries lie on both sides of every kink of the
# operations below (-3, -1, 0, 0.5, 1, 3, 6), with intervals as wide as 14.5.
_KINK_LO = np.array([-8.0, -3.0, -0.5, 0.25, 5.0, -2.0,
                     6.5, -1.5, 0.75, -0.25, 2.0, -7.0]).reshape(2, 2, 3)
_KINK_HI = np.array([-6.0, 1.0, 0.25, 3.0, 8.0, 7.0,
                     9.0, -0.75, 1.5, 0.75, 6.5, 7.5]).reshape(2, 2, 3)
_KINK_SHAPE = (1, 2, 2, 3)
_ALL_KINDS = ("Star", "ImageStar", "Zono", "ImageZono", "Box", "Hexatope",
              "Octatope")
# Under "exact" the model entry refuses every set type but the Star family
# before any node is reached (test_exact_on_a_set_type_without_exact_rules_
# is_refused), so "exact" adds a cell only on Star and ImageStar.
_KIND_METHODS = ([(k, "approx") for k in _ALL_KINDS]
                 + [("Star", "exact"), ("ImageStar", "exact")])


def _kink_set(kind):
    if kind.startswith("Image"):
        return image_set(kind, _KINK_LO, _KINK_HI)
    return flat_set(kind, _KINK_LO, _KINK_HI)


def _refuses_or_covers(call, outputs, kind, what):
    """The cell's promise: ``call()`` raises ``NotImplementedError`` or
    ``ValueError``, or returns sets that hold every output, read as the
    (2, 2, 3) tensor (an image set in its stored order, a flat set in the
    tensor's own order)."""
    try:
        sets = call()
    except (NotImplementedError, ValueError):
        return
    assert_covered(sets, outputs, what, out_chw=_KINK_SHAPE[1:])


def _clip_node(low, high, opset):
    """An ONNX Clip node and its initializers: bounds as inputs from opset
    11 on (an empty name leaves one out), as attributes before."""
    if opset < 11:
        attrs = {k: float(v) for k, v in (("min", low), ("max", high))
                 if v is not None}
        return helper.make_node("Clip", ["X"], ["Y"], **attrs), {}
    inits, names = {}, ["X"]
    for name, value in (("lo", low), ("hi", high)):
        names.append(name if value is not None else "")
        if value is not None:
            inits[name] = np.array(value, dtype=np.float32)
    while names[-1] == "":
        names.pop()
    return helper.make_node("Clip", names, ["Y"]), inits


# variant: (min, max, opset, the module load_onnx builds)
_CLIPS = {
    "Clip(0,6)": (0.0, 6.0, 13, "ReLU6"),
    "Clip(-1,0.5)": (-1.0, 0.5, 13, "OnnxClip"),
    "Clip(0.5,none)": (0.5, None, 13, "OnnxClip"),
    "Clip(-1,0.5)-opset6": (-1.0, 0.5, 6, "OnnxClip"),
}


def _clip_cells():
    cells = []
    for variant in ("Clip(0,6)", "Clip(-1,0.5)"):
        for route in ("model", "reach_layer"):
            for kind, method in _KIND_METHODS:
                cells.append(pytest.param(
                    variant, kind, method, route,
                    id=f"{variant}-{kind}-{method}-{route}"))
    # The converter picks the module from the bound values and, before
    # opset 11, from attributes: one set type is enough for those routes.
    for variant in ("Clip(0.5,none)", "Clip(-1,0.5)-opset6"):
        cells.append(pytest.param(variant, "Star", "approx", "model",
                                  id=f"{variant}-Star-approx-model"))
    return cells


@pytest.mark.parametrize("variant,kind,method,route", _clip_cells())
def test_onnx_clip_refuses_or_is_sound(tmp_path, variant, kind, method,
                                       route):
    """ONNX Clip has no reach rule: ``load_onnx`` turns Clip(0, 6) into
    ``nn.ReLU6`` and other bounds into ``OnnxClip`` (Clip with min 0 and no
    max becomes ``nn.ReLU`` and belongs to the ReLU cells). Catches a lane of
    the dispatcher that answers a layer type it has no rule for (with the
    identity, or with the ReLU rule for ``nn.ReLU6``) and a converter that
    builds the wrong module from the bounds: the inputs lie on both sides of
    both bounds, so an answer that keeps a value the clip moves misses the
    onnxruntime output."""
    low, high, opset, module_name = _CLIPS[variant]
    node, inits = _clip_node(low, high, opset)
    shape = list(_KINK_SHAPE)
    path = save_onnx(tmp_path, [node], shape, shape, inits, name="clip",
                     opset=opset)
    model = load_onnx(path)
    modules = [m for n, m in model.named_modules() if n]
    if [type(m).__name__ for m in modules] != [module_name]:
        pytest.fail(f"{variant}: load_onnx built {modules}, the cell expects "
                    f"one {module_name}")
    outputs = replay_onnx(path, samples(_KINK_LO, _KINK_HI), shape)
    what = f"{variant} on {kind} under {method} via {route}"
    if route == "model":
        options = {} if kind.startswith("Image") else {"input_shape": _KINK_SHAPE}
        call = lambda: reach_model(model, _kink_set(kind), method, what,  # noqa: E731
                                   **options)
    else:
        call = lambda: reach_one_layer(modules[0], [_kink_set(kind)],  # noqa: E731
                                       method, what)
    _refuses_or_covers(call, outputs, kind, what)


class _Calls(nn.Module):
    """A model whose forward is one call of ``fn``, traced by ``torch.fx``
    into one ``call_function`` (or ``call_method``) node."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


# The relatives of F.relu, torch.sigmoid and torch.tanh, which the graph walk
# converts into layers (FUNCTION_TO_MODULE_CLS in n2v/nn/reach.py).
_RELATIVES = {
    "F.relu6": F.relu6,
    "F.hardtanh": F.hardtanh,
    "F.elu": F.elu,
    "F.gelu": F.gelu,
    "F.silu": F.silu,
    "F.softplus": F.softplus,
    "F.hardsigmoid": F.hardsigmoid,
    "F.hardswish": F.hardswish,
    "torch.clamp": lambda x: torch.clamp(x, min=-1.0, max=0.5),
    "x.clamp": lambda x: x.clamp(-1.0, 0.5),
}


@pytest.mark.parametrize("kind,method", _KIND_METHODS,
                         ids=[f"{k}-{m}" for k, m in _KIND_METHODS])
@pytest.mark.parametrize("fn", sorted(_RELATIVES))
def test_a_torch_function_without_a_reach_rule_refuses_or_is_sound(
        fn, kind, method):
    """A torch model that calls a function the graph walk has no rule for.
    Catches a function table entry that maps it onto a converted layer it
    resembles (relu6 onto ReLU, hardtanh onto Tanh, silu onto Sigmoid) and a
    walk that passes the sets through such a node, or through a tensor method
    it does not know (``x.clamp``), as if it were a copy. The inputs lie on
    both sides of every kink, so each of those answers misses a torch
    output (float64 replay)."""
    model = _Calls(_RELATIVES[fn])
    outputs = replay_torch(model, samples(_KINK_LO, _KINK_HI), _KINK_SHAPE)
    what = f"{fn} on {kind} under {method}"
    options = {} if kind.startswith("Image") else {"input_shape": _KINK_SHAPE}
    _refuses_or_covers(
        lambda: reach_model(model, _kink_set(kind), method, what, **options),
        outputs, kind, what)
