"""
Soundness tests for ReduceMean / ReduceSum layer reachability.

Every cell reaches an input set through one Reduce layer (``reach_layer``) or
through an ONNX or torch model and checks that every true output of member
points of the input set is a member of a returned set (or that the call
refuses where the library has no rule); the cells are described below.
"""

import numpy as np
import pytest

try:
    from onnx2torch.node_converters.reduce import (
        OnnxReduceStaticAxes,
        OnnxReduceSumStaticAxes,
    )
    HAS_ONNX2TORCH = True
except ImportError:
    HAS_ONNX2TORCH = False

from n2v.sets import Star, Zono, Box


pytestmark = pytest.mark.skipif(
    not HAS_ONNX2TORCH, reason="onnx2torch not installed"
)


# =========================================================================== #
# Soundness cells
#
# Cells: ReduceSum (the opset-13 node, axes as an input, and the opset-11
# node, axes as an attribute) and ReduceMean through ``reach_layer`` on
# ImageStar and ImageZono (``reduce_star``, ``reduce_zono``, ``_get_axes``,
# ``_reduce_hwc``) for one, several and every axis, negative axes, the batch
# axis, keepdims 0 and 1; on several input sets at once and on an ImageStar
# with a binding row; the non-affine reduce ops, which must be refused or
# answered soundly; flat Star, Zono and Box (``_reduce_flat_*``), which carry
# no tensor shape, and Hexatope and Octatope, which have no Reduce; the
# outward rounding of the Box reduction, in exact fractions; the chains of
# Reduce nodes through ``reach_layer`` that the flat kernel misreads
# (registry row AFFINE-4, marked); and the chains Conv, Reduce, Conv and
# Mul, Add, Reduce, Mul, Add through ``reach_pytorch_model`` for image sets
# and for flat sets with a known tensor shape (``_handle_onnx_reduce_flat``).
# For flat sets also: the model routes without ``input_shape`` (shape
# tracked, and no shape tracked: the flat kernel answers), a model whose
# shape record is lost to another node (marked, SHAPE-8), the torch
# model route and the opset-13 ReduceSum without an axes input (both refused
# today), the exact method in both directions in rationals, and the strict
# refusals of the rows the library does not answer (Hexatope, Octatope, the
# exact method on Zono and Box, the non-affine ops).
#
# Oracle: true outputs of sample points (corners included), by the ONNX
# definition of the reduction in numpy float64 for a layer (not the
# onnx2torch module's forward, which raises for an opset-13 ReduceSum with
# empty axes and keepdims=0), and by onnxruntime for a model: on a float64
# copy of the graph where a mean over 3 entries would be rounded in float32,
# on the float32 graph where every value is dyadic (onnxruntime has no
# float64 Conv); each output is a member of a returned set by the LP on the
# set's own rows. Layout: an image output (keepdims=1) is compared with
# ``out_chw``, which also checks the stored height, width and channel count;
# an output of lower rank is a flat set in the tensor's row-major order of
# the remaining axes.
# =========================================================================== #
import torch  # noqa: E402
from onnx import helper  # noqa: E402

from fractions import Fraction  # noqa: E402

from n2v.utils.model_loader import load_onnx  # noqa: E402
from n2v.sets.image_star import ImageStar  # noqa: E402
from n2v.sets.image_zono import ImageZono  # noqa: E402
from tests._oracles import rational  # noqa: E402
from tests.cell_support import (N_SAMPLES, assert_covered,  # noqa: E402
                                constrained_image_star,
                                constrained_star, flat_box, flat_set,
                                hwc_index, image_set, pixel_box, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx)
from tests.known_bugs import known_bug  # noqa: E402

_RC, _RH, _RW = 2, 3, 5
_RLO, _RHI = pixel_box(_RC, _RH, _RW, seed=9)
# The same image moved above zero (lowest pixel bound 1/2): a sum and a mean
# of positive entries have disjoint ranges, which a box straddling zero hides
# (the range of the mean lies inside that of the sum there).
_RLO_POS, _RHI_POS = _RLO - _RLO.min() + 0.5, _RHI - _RLO.min() + 0.5


def _reducer(op, axes, keepdims, noop=0):
    """The onnx2torch module of one reduce node. ``"ReduceSum"`` is the
    opset-13 node (axes as an input: ``OnnxReduceSumStaticAxes``; no axes is
    the empty constant input, ``axes=[]``, as the converter builds it),
    ``"ReduceSum11"`` the opset-11 node (axes as an attribute:
    ``OnnxReduceStaticAxes``); every other op is an ``OnnxReduceStaticAxes``,
    as the converter builds them."""
    if op == "ReduceSum":
        return OnnxReduceSumStaticAxes(axes=[] if axes is None else list(axes),
                                       keepdims=keepdims,
                                       noop_with_empty_axes=noop)
    return OnnxReduceStaticAxes("ReduceSum" if op == "ReduceSum11" else op,
                                None if axes is None else list(axes),
                                keepdims=keepdims)


_ONNX_REDUCE = {
    "ReduceSum": np.sum, "ReduceSum11": np.sum, "ReduceMean": np.mean,
    "ReduceMax": np.max, "ReduceMin": np.min, "ReduceProd": np.prod,
    "ReduceL1": lambda x, **k: np.sum(np.abs(x), **k),
    "ReduceL2": lambda x, **k: np.sqrt(np.sum(x * x, **k)),
    "ReduceLogSum": lambda x, **k: np.log(np.sum(x, **k)),
    "ReduceLogSumExp": lambda x, **k: np.log(np.sum(np.exp(x), **k)),
    "ReduceSumSquare": lambda x, **k: np.sum(x * x, **k),
}


def _true_outputs(nodes, points, in_shape):
    """The true outputs of a chain of reduce nodes ``(op, axes, keepdims[,
    noop])`` on ``points``, by the ONNX definition in numpy float64 (no
    axes: every axis, or the input itself for a ReduceSum with
    ``noop_with_empty_axes``; a negative axis counts from the end), each a
    flat vector in the output tensor's row-major order. Exact for dyadic
    points except for the division of a mean, rounded once in float64. Not
    the onnx2torch module's forward, which raises for an opset-13 ReduceSum
    with empty axes and keepdims=0, and which writes the axes it chose back
    into the module."""
    outputs = []
    for x in points:
        y = np.asarray(x, dtype=np.float64).reshape(in_shape)
        for node in nodes:
            op, axes, keepdims = node[:3]
            noop = node[3] if len(node) > 3 else 0
            if not axes:
                if noop:
                    continue
                axes = range(y.ndim)
            axes = tuple(sorted({a % y.ndim for a in axes}))
            y = _ONNX_REDUCE[op](y, axis=axes, keepdims=bool(keepdims))
        outputs.append(np.asarray(y, dtype=np.float64).reshape(-1))
    return outputs


def _reduced_shape(shape, axes, keepdims, noop=0):
    """The ONNX output shape of a reduction of a tensor of ``shape``."""
    if not axes:
        if noop:
            return tuple(shape)
        axes = range(len(shape))
    axes = tuple(sorted({a % len(shape) for a in axes}))
    return np.sum(np.zeros(shape), axis=axes, keepdims=bool(keepdims)).shape


def _rid(value):
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(map(str, value)) + "]"
    return str(value)


def _assert_holds(sets, outputs, out_shape, what):
    """An output of rank 4 is an image (compared with ``out_chw``); any other
    output is flat, in the tensor's row-major order."""
    if len(out_shape) == 4:
        assert_covered(sets, outputs, what, out_chw=tuple(out_shape[1:]))
    else:
        assert_covered(sets, outputs, what)


def _cut(lo, hi, shares=(1, 2, 5)):
    """The box cut along its first coordinate of positive width into parts
    of the given shares, as ``(lo, hi)`` pairs (dyadic for dyadic bounds)."""
    flat_lo, flat_hi = lo.reshape(-1), hi.reshape(-1)
    j = int(np.flatnonzero(flat_hi > flat_lo)[0])
    ends = np.cumsum((0,) + tuple(shares)) / float(sum(shares))
    cuts = flat_lo[j] + (flat_hi[j] - flat_lo[j]) * ends
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        part_lo, part_hi = flat_lo.copy(), flat_hi.copy()
        part_lo[j], part_hi[j] = a, b
        out.append((part_lo.reshape(lo.shape), part_hi.reshape(lo.shape)))
    return out


# --------------------------------------------------------------------------- #
# Image sets through reach_layer                                                #
# --------------------------------------------------------------------------- #
# The reductions of _IMAGE_POSITIVE run on the positive image only, where a
# mean answered as a sum (or a sum as a mean) cannot hide.
_IMAGE_REDUCTIONS = [
    ("ReduceMean", [2], 1), ("ReduceSum11", [3], 1),
    ("ReduceSum", [1, 2, 3], 1),
    ("ReduceSum", [0], 1), ("ReduceMean", [0, 3], 1),
    ("ReduceSum11", [-1], 1), ("ReduceSum", [2], 0),
    ("ReduceMean", [3], 0), ("ReduceSum11", [1, 3], 0),
    ("ReduceMean", [-3, -2], 0), ("ReduceSum", [0], 0), ("ReduceMean", [-2], 0),
]


_IMAGE_POSITIVE = [
    ("ReduceMean", [2, 3], 1), ("ReduceSum", [1], 1), ("ReduceMean", [1], 0),
    ("ReduceSum11", [3], 0), ("ReduceMean", None, 1), ("ReduceSum", None, 0),
    ("ReduceMean", [-1], 1),
]
# The image scaled by 8: bounds from -16 to 18 and intervals 2 to 10 wide,
# where every predicate variable of the box set still ranges over [-1, 1].
_RLO_WIDE, _RHI_WIDE = _RLO * 8, _RHI * 8
_IMAGE_WIDE = [("ReduceSum", [2], 0), ("ReduceMean", [1, 3], 1),
               ("ReduceSum11", [-1], 0)]
# The image with two pixels of zero width (an ImageStar gives them no
# predicate variable, an ImageZono a zero generator).
_RHI_FLAT = _RHI.copy()
_RHI_FLAT[1, 2, 4], _RHI_FLAT[0, 0, 1] = _RLO[1, 2, 4], _RLO[0, 0, 1]
_IMAGE_DEGENERATE = [("ReduceSum", [1], 0), ("ReduceMean", [2, 3], 1)]
_IMAGE_DATA = {"mixed": (_RLO, _RHI), "positive": (_RLO_POS, _RHI_POS),
               "wide": (_RLO_WIDE, _RHI_WIDE), "degenerate": (_RLO, _RHI_FLAT)}


@pytest.mark.parametrize("op,axes,keepdims,data",
                         [r + ("mixed",) for r in _IMAGE_REDUCTIONS]
                         + [r + ("positive",) for r in _IMAGE_POSITIVE]
                         + [r + ("wide",) for r in _IMAGE_WIDE]
                         + [r + ("degenerate",) for r in _IMAGE_DEGENERATE],
                         ids=_rid)
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_reduce_layer_on_an_image_set_holds_every_output(kind, op, axes,
                                                         keepdims, data):
    """ReduceSum or ReduceMean of a (2, 3, 5) image with a different interval
    per pixel, over the channel, height or width axis alone, two or all of
    them, a negative axis, the batch axis alone or with another one, with
    keepdims 0 and 1. Catches a tensor axis mapped onto the wrong stored axis
    (height and width swapped), a negative axis resolved off by one, mean and
    sum swapped (and an opset-13 ReduceSum read as a mean), the centre of an
    ImageZono reduced in another order than its generators, a flat result in
    stored order instead of the tensor's, a kept-dims result that stores
    its height and width swapped or its predicate bounds wrongly, keepdims
    read from the wrong attribute, a batch axis that is not dropped, a
    reduction with no axes taken for none, and a no-op test that ignores
    ``noop_with_empty_axes``. The image straddles zero (``mixed``), lies
    above it (``positive``), where a mean answered as a sum, or a sum as a
    mean, misses every output, or is scaled by 8 (``wide``: bounds well
    beyond -1 and 1, intervals wider than 2), where a result that reads one
    predicate variable for another can no longer hide inside [-1, 1], or has
    two pixels of zero width (``degenerate``: fewer predicate variables
    than entries on an ImageStar)."""
    layer = _reducer(op, axes, keepdims)
    lo, hi = _IMAGE_DATA[data]
    what = f"{op}(axes={axes}, keepdims={keepdims}) on {kind} ({data})"
    result = reach_one_layer(layer, [image_set(kind, lo, hi)], "approx", what)
    in_shape = (1, _RC, _RH, _RW)
    _assert_holds(result, _true_outputs([(op, axes, keepdims)],
                                        samples(lo, hi), in_shape),
                  _reduced_shape(in_shape, axes, keepdims), what)


def _extra_column_image(kind, lo_chw, hi_chw, extra=3, seed=0, n=N_SAMPLES):
    """``(set, points)``: an image set with ``extra`` more generator columns
    than entries, the shape an approximate ReLU leaves (an ImageZono with
    more generators than entries, an ImageStar with more predicate
    variables than entries), centred on the box [lo, hi] of a (C, H, W)
    image: one column per pixel of half its width, then ``extra`` columns
    of dyadic entries (multiples of 1/8 up to 1/2) shared by every pixel, so
    the set reaches beyond the box. The predicate is the cube [-1, 1]^k.
    ``points`` are member points as (C, H, W) arrays, computed without n2v
    from corners and half-grid predicate points (dyadic)."""
    channels, height, width = lo_chw.shape
    order = hwc_index(channels, height, width)
    centre = ((lo_chw + hi_chw) / 2).reshape(-1)[order]
    half = ((hi_chw - lo_chw) / 2).reshape(-1)[order]
    rng = np.random.default_rng(seed)
    G = np.hstack([np.diag(half),
                   rng.integers(-4, 5, size=(half.size, extra)) / 8.0])
    k = G.shape[1]
    if kind == "ImageZono":
        made = ImageZono(centre.reshape(-1, 1), G, height, width, channels)
    else:
        made = ImageStar(np.hstack([centre[:, None], G]),
                         np.vstack([np.eye(k), -np.eye(k)]),
                         np.ones((2 * k, 1)), -np.ones((k, 1)),
                         np.ones((k, 1)), height=height, width=width,
                         num_channels=channels)
    points = []
    for j in range(n):
        a = (rng.choice([-1.0, 1.0], size=k) if j % 2
             else rng.integers(-2, 3, size=k) / 2.0)
        if j < 2:
            a = np.full(k, -1.0 if j == 0 else 1.0)
        x = np.empty(half.size)
        x[order] = centre + G @ a
        points.append(x.reshape(channels, height, width))
    return made, points


@pytest.mark.parametrize("op,axes,keepdims", [("ReduceMean", [2, 3], 1),
                                              ("ReduceSum", [1], 0),
                                              ("ReduceSum11", [3], 1)], ids=_rid)
@pytest.mark.parametrize("kind,inputs", [("ImageStar", "several sets"),
                                         ("ImageZono", "several sets"),
                                         ("ImageStar", "binding row"),
                                         ("ImageStar", "extra columns"),
                                         ("ImageZono", "extra columns")])
def test_reduce_layer_on_several_or_cut_image_sets(kind, inputs, op, axes,
                                                   keepdims):
    """Three image sets of different sizes at once, an ImageStar cut by a
    binding row that is not a box row, and an image set with more generator
    columns than entries (an ImageZono with more generators, an ImageStar
    with more predicate variables, as an approximate ReLU leaves them).
    Catches a loop over the input sets that returns after the first set or
    reduces the first set for every set, a reduction that touches the
    constraint rows, and one that reads the generator count from the number
    of entries."""
    layer = _reducer(op, axes, keepdims)
    what = f"{op}(axes={axes}, keepdims={keepdims}) on {inputs} of {kind}"
    if inputs == "binding row":
        star, points = constrained_image_star(_RLO, _RHI, seed=5)
        sets = [star]
    elif inputs == "extra columns":
        made, points = _extra_column_image(kind, _RLO, _RHI, seed=6)
        sets = [made]
    else:
        sets = [image_set(kind, a, b) for a, b in _cut(_RLO, _RHI)]
        points = samples(_RLO, _RHI)
    result = reach_one_layer(layer, sets, "approx", what)
    in_shape = (1, _RC, _RH, _RW)
    _assert_holds(result, _true_outputs([(op, axes, keepdims)], points, in_shape),
                  _reduced_shape(in_shape, axes, keepdims), what)


# The reduce ops that are the identity over one element (``_get_reduce_op``
# answers them as a sum when every reduced axis has length 1) and the ones
# that never are (|x|, x^2, log x: always refused).
_IDENTITY_OPS = ("ReduceMax", "ReduceMin", "ReduceProd", "ReduceLogSumExp")
_NON_IDENTITY_OPS = ("ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceSumSquare")
# Images with one axis of length 1 (the height, the channels, the width in
# turn). For the identity ops: each image reduced over its axis of length 1
# (one element: answered), over the axis that a size lookup in the stored
# (height, width, channel) order would take for an axis of length 1 (more
# than one element: refused today, answered as a sum by that lookup), over
# the channels and the height of the first image (one axis of length 1 and
# one longer: refused today, answered by a test that asks for any axis of
# length 1 instead of every one), and over every axis of a (2, 1, 1) image
# (two elements: refused today, answered by a lookup that finds no sizes
# when the layer has no axes). The non-identity ops are refused over any
# axis; their one case is the single element, where an op added to the
# identity list would be answered as x.
_IMAGE_SHAPES = [((2, 1, 5), [2]), ((2, 1, 5), [1]), ((2, 1, 5), [1, 2]),
                 ((1, 3, 5), [1]), ((1, 3, 5), [3]), ((2, 3, 1), [3]),
                 ((2, 3, 1), [2]), ((2, 1, 1), None)]
_ONE_ELEMENT = [((2, 1, 5), [2]), ((1, 3, 5), [1]), ((2, 3, 1), [3])]
# Once an identity op is accepted, ``_get_reduce_op`` answers it as a sum
# whatever the op, so the answered single-element cases run for ReduceMax
# only; the refusals run for every op (each is its own entry of the
# identity list).
_NON_AFFINE_CELLS = ([(op, chw, axes) for op in _IDENTITY_OPS
                      for chw, axes in _IMAGE_SHAPES
                      if op == "ReduceMax" or (chw, axes) not in _ONE_ELEMENT]
                     + [(op, (2, 1, 5), [2]) for op in _NON_IDENTITY_OPS])


@pytest.mark.parametrize("op,chw,axes", _NON_AFFINE_CELLS,
                         ids=[f"{op}-{_rid(list(chw))}-{_rid(axes)}"
                              for op, chw, axes in _NON_AFFINE_CELLS])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_non_affine_reduce_on_an_image_set_is_refused_or_sound(kind, op, chw,
                                                               axes):
    """The non-affine reduce ops over an image with one axis of length 1, and
    over a whole (2, 1, 1) image. Over one element ReduceMax, ReduceMin,
    ReduceProd and ReduceLogSumExp are the identity and may be answered;
    ReduceL1, ReduceL2, ReduceSumSquare and ReduceLogSum are not (|x|, x^2,
    log x), and over more than one element none is affine. The call must
    refuse or hold every output. Catches an op added to the identity list, a
    size test that looks the sizes up in another axis order (the stored
    height, width, channel order, say: then a maximum over two channels of a
    one-row image is taken for one over the height and answered as a sum),
    one that accepts the op when any axis, not every axis, has length 1, and
    one that finds no sizes to test when the layer has no axes (every axis
    reduced). The strict refusals of these ops are
    ``tests/soundness/test_leftover_reduce.py::test_non_affine_reduce_is_refused``,
    and the answer of each identity op over one element is checked there
    too (``test_identity_non_affine_reduce_over_size_one_axes``).
    ReduceLogSum gets a positive image, where its value is defined, and so
    does the whole-image reduction, where a sum of the two entries is above
    their maximum (on entries of both signs it can hold it by accident)."""
    lo, hi = pixel_box(*chw, seed=11)
    if op == "ReduceLogSum" or axes is None:
        lo, hi = lo + 3, hi + 3
    layer = _reducer(op, axes, 1)
    what = f"{op}(axes={axes}) on a {chw} {kind}"
    try:
        result = reach_one_layer(layer, [image_set(kind, lo, hi)], "approx", what)
    except NotImplementedError:
        return
    in_shape = (1,) + tuple(chw)
    _assert_holds(result, _true_outputs([(op, axes, 1)], samples(lo, hi), in_shape),
                  _reduced_shape(in_shape, axes, 1), what)


@pytest.mark.parametrize("axes", [[4], [-5], [1, 4]], ids=_rid)
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_reduce_over_an_axis_the_image_does_not_have_raises(kind, axes):
    """An axis outside the rank-4 tensor: torch raises on the true network,
    so the reach must raise too (a ``ValueError`` for a malformed axis or a
    refusal) instead of reducing some other axis."""
    with pytest.raises((ValueError, NotImplementedError)):
        reach_one_layer(_reducer("ReduceMean", axes, 1),
                        [image_set(kind, _RLO, _RHI)], "approx",
                        f"ReduceMean(axes={axes}) on {kind}")


# --------------------------------------------------------------------------- #
# Flat sets through reach_layer                                                 #
# --------------------------------------------------------------------------- #
# A flat set carries no tensor shape. Each cell says which tensor it stands
# for; the kernel must refuse, or hold every output of that tensor's
# reduction (the registry's fix for AFFINE-4 refuses every Reduce on a set
# that does not carry its shape, which passes every cell here).
_FLAT_N = 6
_FLO, _FHI = flat_box(_FLAT_N, seed=13, zero_width=True)
# The same box moved above zero (lowest bound 1/2), as for the image: the
# ranges of a sum and of a mean of positive entries are disjoint, which a box
# straddling zero hides (the mean's range lies inside the sum's there).
_FLO_POS, _FHI_POS = _FLO - _FLO.min() + 0.5, _FHI - _FLO.min() + 0.5
# The box scaled by 8 (bounds from -15 to 15, intervals 3 to 14 wide), where
# every predicate variable of the box set still ranges over [-1, 1]: a
# result that reads one variable for another can no longer hide inside it.
# The positive box scaled by 8 (bounds 4 to 34) serves the means, whose
# wrong divisor a box straddling zero hides.
_FLO_WIDE, _FHI_WIDE = _FLO * 8, _FHI * 8
_FLAT_DATA = {"mixed": (_FLO, _FHI), "positive": (_FLO_POS, _FHI_POS),
              "wide": (_FLO_WIDE, _FHI_WIDE),
              "wide-positive": (_FLO_POS * 8, _FHI_POS * 8)}
_FLAT_REDUCTIONS = [
    ("ReduceSum", [1], 0), ("ReduceMean", [1], 1), ("ReduceSum11", [1], 1),
    ("ReduceSum", None, 1), ("ReduceMean", None, 0), ("ReduceMean", [0, 1], 0),
    ("ReduceSum11", [0], 0), ("ReduceMean", [0], 1),
]


def _extra_column_flat(kind, lo, hi, extra=3, seed=0, n=N_SAMPLES):
    """``(set, points)``: a flat Star with more predicate variables than
    entries, or a Zono with more generators than entries (the shape an
    approximate ReLU leaves), centred on the box [lo, hi]: one column per
    entry of half its width, then ``extra`` columns of dyadic entries
    (multiples of 1/8 up to 1/2) shared by every entry, so the set reaches
    beyond the box. The predicate is the cube [-1, 1]^k. ``points`` are
    member points computed without n2v from corners and half-grid predicate
    points (dyadic)."""
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    rng = np.random.default_rng(seed)
    G = np.hstack([np.diag(half),
                   rng.integers(-4, 5, size=(half.size, extra)) / 8.0])
    k = G.shape[1]
    if kind == "Zono":
        made = Zono(centre.reshape(-1, 1), G)
    else:
        made = Star(np.hstack([centre[:, None], G]),
                    np.vstack([np.eye(k), -np.eye(k)]), np.ones((2 * k, 1)),
                    -np.ones((k, 1)), np.ones((k, 1)))
    points = []
    for j in range(n):
        a = (rng.choice([-1.0, 1.0], size=k) if j % 2
             else rng.integers(-2, 3, size=k) / 2.0)
        if j < 2:
            a = np.full(k, -1.0 if j == 0 else 1.0)
        points.append(centre + G @ a)
    return made, points


def _flat_cell(kind, inputs, op, axes, keepdims, shape, data="mixed"):
    layer = _reducer(op, axes, keepdims)
    what = (f"{op}(axes={axes}, keepdims={keepdims}) on {inputs} of {kind} "
            f"standing for a {shape} tensor ({data})")
    lo, hi = _FLAT_DATA[data]
    if inputs == "binding row":
        star, points = constrained_star(lo, hi, seed=4)
        sets = [star]
    elif inputs == "extra columns":
        made, points = _extra_column_flat(kind, lo, hi, seed=6)
        sets = [made]
    else:
        boxes = [(lo, hi)] if inputs == "one set" else _cut(lo, hi)
        sets = [flat_set(kind, a, b) for a, b in boxes]
        points = samples(lo, hi)
    try:
        result = reach_one_layer(layer, sets, "approx", what)
    except NotImplementedError:
        return
    _assert_holds(result, _true_outputs([(op, axes, keepdims)], points, shape),
                  _reduced_shape(shape, axes, keepdims), what)


# Every reduction on the positive box for Star, Zono and Box; a sum and a
# mean on the wide box; Hexatope and Octatope, which have no Reduce rule (one
# dispatcher branch refuses every reduction: their strict refusal is
# test_reduce_on_a_hexatope_or_octatope_is_refused), once each.
_ROW_VECTOR_CELLS = (
    [(kind, "positive") + r for kind in ("Star", "Zono", "Box")
     for r in _FLAT_REDUCTIONS]
    + [(kind, data) + r for kind in ("Star", "Zono", "Box")
       for data, r in (("wide", ("ReduceSum", [1], 0)),
                       ("wide-positive", ("ReduceMean", None, 0)))]
    + [(kind, "mixed", "ReduceSum", [1], 0) for kind in ("Hexatope",
                                                         "Octatope")])


@pytest.mark.parametrize(
    "kind,data,op,axes,keepdims", _ROW_VECTOR_CELLS,
    ids=[f"{k}-{d}-{o}-{_rid(a)}-{kd}" for k, d, o, a, kd in _ROW_VECTOR_CELLS])
def test_reduce_layer_on_a_flat_set_of_a_row_vector(kind, data, op, axes,
                                                    keepdims):
    """A flat set of six entries that stands for a (1, 6) tensor (one entry
    of zero width) through ``reach_layer``: the feature axis in the opset-13
    form (axes as an input) and the opset-11 form (an attribute), no axes in
    both forms, the batch and feature axes, the batch axis alone, on a box
    above zero (``positive``), where the ranges of a sum and of a mean are
    disjoint, and on a box scaled by 8 whose entries straddle zero, are
    positive, negative and of zero width (``wide``; the mean on the positive
    box scaled by 8, ``wide-positive``). Catches mean and sum
    swapped, a mean without its division, a sum divided by the count, bounds
    of a Box mean or sum built from one side, a batch-only reduction that
    reduces the features (the past bug of
    ``_get_axes``), and a no-axes reduction read as some other axis.
    Hexatope and Octatope have no Reduce and must refuse (or answer
    soundly)."""
    _flat_cell(kind, "one set", op, axes, keepdims, (1, _FLAT_N), data)


@pytest.mark.parametrize("op,axes,keepdims", [("ReduceMean", [1], 0),
                                              ("ReduceSum", None, 1)], ids=_rid)
@pytest.mark.parametrize("kind,inputs", [("Star", "several sets"),
                                         ("Zono", "several sets"),
                                         ("Box", "several sets"),
                                         ("Star", "binding row"),
                                         ("Star", "extra columns"),
                                         ("Zono", "extra columns")])
def test_reduce_layer_on_several_or_cut_flat_sets(kind, inputs, op, axes,
                                                  keepdims):
    """Three flat sets of different sizes at once, a Star cut by a binding
    row, and a Star with more predicate variables than entries or a Zono
    with more generators than entries (the shape an approximate ReLU
    leaves), all on the positive box, where a mean and a sum differ.
    Catches a loop that returns after the first set, or reduces the first
    set for every set, mean and sum swapped on these input shapes, and a
    reduction that reads the generator count from the number of entries."""
    _flat_cell(kind, inputs, op, axes, keepdims, (1, _FLAT_N), "positive")


# Bounds whose float64 sum and mean both round inward (the sum of the lower
# bounds rounds up, the sum of the upper bounds rounds down), found by
# a search in exact fractions; each literal is the float it prints as.
_ROUND_LO = np.array([0.517, 0.325, 1.858, 0.363, -1.269])
_ROUND_HI = np.array([1.343, 1.9609999999999999, 3.1109999999999998,
                      2.2809999999999997, -0.5299999999999999])


@pytest.mark.parametrize("op,keepdims", [("ReduceSum", 0), ("ReduceSum11", 1),
                                         ("ReduceMean", 0)], ids=_rid)
@pytest.mark.parametrize("inputs", ["one box", "two boxes", "model"])
def test_box_reduce_is_rounded_outward_in_exact_arithmetic(tmp_path, inputs,
                                                          op, keepdims):
    """ReduceSum and ReduceMean of a Box over its feature axis, decided in
    exact fractions: the returned lower bound is at most the exact sum (or
    mean) of the stored lower bounds, and the returned upper bound at least
    that of the upper bounds, with no slack. The bounds are chosen so that
    plain float64 summation lands inside on both sides, so a bound computed
    without the outward rounding, or rounded the wrong way (either output of
    the rounding helper taken for the other bound), fails. With two boxes
    the rounding box is the second one, so a loop that rounds only the first
    set fails too. ``model``: the same box through a one-node ONNX model with
    ``input_shape``, where the graph handler builds the reduction matrix and
    the Box maps its bounds through it; catches bounds mapped through that
    matrix in round-to-nearest."""
    rounding = Box(_ROUND_LO.copy(), _ROUND_HI.copy())
    boxes = ([Box(_FLO.copy(), _FHI.copy()), rounding] if inputs == "two boxes"
             else [rounding])
    exact = []
    for b in boxes:
        lo, hi = (sum(Fraction(float(v)) for v in b.lb.reshape(-1)),
                  sum(Fraction(float(v)) for v in b.ub.reshape(-1)))
        if op == "ReduceMean":
            lo, hi = lo / b.dim, hi / b.dim
        exact.append((lo, hi))
    naive = (np.mean if op == "ReduceMean" else np.sum)
    if not (Fraction(float(naive(_ROUND_LO))) > exact[-1][0]
            and Fraction(float(naive(_ROUND_HI))) < exact[-1][1]):
        pytest.fail("setup: float64 summation of the rounding box does not "
                    "land inside on both sides, so the cell cannot see a "
                    "missing outward rounding")
    what = f"{op}(axes=[1]) on {inputs} of Box, exact fractions"
    try:
        if inputs == "model":
            nodes, inits = _reduce_nodes(op, [1], keepdims, "X", "Y", "r")
            path = save_onnx(tmp_path, nodes, [1, 5],
                             list(_reduced_shape((1, 5), [1], keepdims)),
                             inits=inits, name="rb", opset=_opset([op]))
            result = reach_model(load_onnx(path), boxes, "approx", what,
                                 input_shape=(5,))
        else:
            result = reach_one_layer(_reducer(op, [1], keepdims), boxes,
                                     "approx", what)
    except NotImplementedError:
        return     # a refusal keeps the promise (the fix of AFFINE-4 may refuse)
    assert len(result) == len(boxes), (
        f"{what}: {len(result)} boxes returned for {len(boxes)}")
    for k, (out, (lo, hi)) in enumerate(zip(result, exact)):
        assert out.dim == 1, f"{what}: box {k} has {out.dim} entries, not 1"
        got_lo = Fraction(float(np.asarray(out.lb).reshape(-1)[0]))
        got_hi = Fraction(float(np.asarray(out.ub).reshape(-1)[0]))
        assert got_lo <= lo, (
            f"{what}: box {k} lower bound {float(got_lo)!r} is above the exact "
            f"value {float(lo)!r} by {float(got_lo - lo):.3g}")
        assert got_hi >= hi, (
            f"{what}: box {k} upper bound {float(got_hi)!r} is below the exact "
            f"value {float(hi)!r} by {float(hi - got_hi):.3g}")


@pytest.mark.parametrize("op,axes,keepdims", [
    ("ReduceSum", [2], 0), ("ReduceMean", [-1], 1), ("ReduceSum11", [1, 2], 0),
    ("ReduceMean", [0, 2], 1), ("ReduceSum", [-2], 0)], ids=_rid)
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_reduce_layer_on_a_flat_set_of_a_matrix_refuses_or_is_sound(
        kind, op, axes, keepdims):
    """The same six entries standing for a (1, 2, 3) tensor, reduced over an
    axis other than 1 or over a negative one: the kernel cannot tell which
    entries form an axis, so it must refuse (or hold every output of the
    (1, 2, 3) tensor). Catches a flat kernel that reads axis 2, or a negative
    axis, as 'every entry'."""
    _flat_cell(kind, "one set", op, axes, keepdims, (1, 2, 3))


@pytest.mark.parametrize("axes", [[3], [-4]], ids=_rid)
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_reduce_over_an_axis_a_flat_tensor_does_not_have_raises(kind, axes):
    """An axis outside the rank-3 tensor the flat set stands for: torch raises
    on the true network, so the reach must raise too."""
    with pytest.raises((ValueError, NotImplementedError)):
        reach_one_layer(_reducer("ReduceSum11", axes, 0),
                        [flat_set(kind, _FLO, _FHI)], "approx",
                        f"ReduceSum(axes={axes}) on {kind}")


@pytest.mark.parametrize("kind,others", [("Star", "Zono"), ("Zono", "Box"),
                                         ("Box", "Star"), ("ImageStar", "Box"),
                                         ("ImageZono", "Star")])
def test_reduce_layer_on_a_list_of_mixed_set_types_raises(kind, others):
    """A list that holds a set of another type after the first: the kernel
    for the first set's type must raise ``TypeError`` (or refuse the Reduce
    altogether with ``NotImplementedError``) instead of skipping or
    misreading the second set."""
    first = (image_set(kind, _RLO, _RHI) if kind.startswith("Image")
             else flat_set(kind, _FLO, _FHI))
    if kind.startswith("Image"):
        second = flat_set(others, _RLO.reshape(-1), _RHI.reshape(-1))
    else:
        second = flat_set(others, _FLO, _FHI)
    with pytest.raises((TypeError, NotImplementedError)):
        reach_one_layer(_reducer("ReduceMean", [1], 1), [first, second],
                        "approx", f"ReduceMean on [{kind}, {others}]")


# "none" and "empty" would build the same layer (``_reducer`` gives the
# opset-13 ReduceSum the empty axes list for None, as the converter does), so
# every kind keeps the empty one.
_NOOP_CELLS = [([], kind) for kind in ("Star", "Zono", "Box", "ImageStar",
                                       "ImageZono")]


@pytest.mark.parametrize("axes,kind", _NOOP_CELLS,
                         ids=[f"{'none' if axes is None else 'empty'}-{kind}"
                              for axes, kind in _NOOP_CELLS])
def test_reduce_sum_with_no_axes_and_noop_is_the_identity(kind, axes):
    """ReduceSum with no axes and ``noop_with_empty_axes=1`` returns its input
    (whatever keepdims says). Catches a no-op test that reads the flag wrongly
    and reduces every axis, which changes the number of entries, and a
    kernel that skips the no-op test (``reduce_star``, ``reduce_zono``)."""
    layer = _reducer("ReduceSum", axes, 0, noop=1)
    what = f"ReduceSum(axes={axes}, noop) on {kind}"
    if kind.startswith("Image"):
        input_set, lo, hi = image_set(kind, _RLO, _RHI), _RLO, _RHI
        shape = (1, _RC, _RH, _RW)
    else:
        input_set, lo, hi = flat_set(kind, _FLO, _FHI), _FLO, _FHI
        shape = (1, _FLAT_N)
    result = reach_one_layer(layer, [input_set], "approx", what)
    _assert_holds(result, _true_outputs([("ReduceSum", axes, 0, 1)],
                                        samples(lo, hi), shape), shape, what)


# --------------------------------------------------------------------------- #
# Chains of Reduce nodes through reach_layer (registry row AFFINE-4)            #
# --------------------------------------------------------------------------- #
_FLAT_AXIS_ONE = known_bug(
    7, "A Reduce on a flat set through reach_layer takes axis 1 to mean every "
       "entry, which is right only when the set stands for a (1, n) tensor; "
       "after an image-set Reduce (which returns a bare flat set) or on a flat "
       "set that stands for a rank-3 tensor it returns a set with the wrong "
       "number of entries, or, after a second Reduce, one that leaves the "
       "true outputs out (registry AFFINE-4)")

_CHAINS = {
    # the image-set reductions alone keep the image: right today
    "mean3k-sum1k": [("ReduceMean", [3], 1), ("ReduceSum", [1], 1)],
    # ReduceMean over the width without keepdims gives a flat (1, 2, 3) set
    "mean3-sum1": [("ReduceMean", [3], 0), ("ReduceSum", [1], 0)],
    "mean3-mean1-sum1": [("ReduceMean", [3], 0), ("ReduceMean", [1], 0),
                         ("ReduceSum11", [1], 0)],
}


@pytest.mark.parametrize("chain", [
    "mean3k-sum1k",
    pytest.param("mean3-mean1-sum1", marks=_FLAT_AXIS_ONE)])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_reduce_chain_on_an_image_set_through_reach_layer(kind, chain):
    """Reduce nodes applied one after another with ``reach_layer`` to a
    (2, 3, 5) image: each call must refuse, or the last result must have the
    entry count of the true output and hold every output. ReduceMean over
    the width without keepdims leaves a (1, 2, 3) tensor as a flat set; the
    next ReduceMean over axis 1 averages the two channels (3 entries), where
    the flat kernel today averages all six (1 entry); the third node, a
    ReduceSum over axis 1, then gives a set of the right size whose value is
    the mean of the six, not the sum over the height of the channel means.
    One marked case per kind: the flat Star and the flat Zono kernel each
    take axis 1 for every entry (registry AFFINE-4); the kept-dims chain,
    which stays an image, is the passing neighbour; through the model, with
    the tensor shape tracked, the same chains are
    ``test_reduce_chain_on_an_image_set_through_the_model_is_sound``."""
    layers = [_reducer(*node) for node in _CHAINS[chain]]
    what = f"{chain} on {kind} through reach_layer"
    sets = [image_set(kind, _RLO, _RHI)]
    try:
        for layer in layers:
            sets = reach_one_layer(layer, sets, "approx", what)
    except NotImplementedError:
        return
    in_shape = (1, _RC, _RH, _RW)
    outputs = _true_outputs(_CHAINS[chain], samples(_RLO, _RHI), in_shape)
    shape = in_shape
    for node in _CHAINS[chain]:
        shape = _reduced_shape(shape, node[1], node[2])
    _assert_holds(sets, outputs, shape, what)


@pytest.mark.parametrize("chain,shape", [
    pytest.param([("ReduceMean", [1], 0), ("ReduceSum", [1], 0)], (1, 3, 4),
                 marks=_FLAT_AXIS_ONE, id="mean1-sum1-of-1x3x4"),
    pytest.param([("ReduceMean", [1], 1), ("ReduceSum", [1], 0)], (1, 12),
                 id="mean1k-sum1-of-1x12")])
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_reduce_chain_on_a_flat_set_of_a_matrix_through_reach_layer(kind, chain,
                                                                    shape):
    """The same flat set of 12 entries, reduced with ``reach_layer`` node by
    node, standing for a (1, 3, 4) tensor or for a (1, 12) row vector. For
    the (1, 3, 4) tensor ReduceMean over axis 1 has 4 outputs (the column
    means), where the flat kernel returns 1, the mean of all 12; the
    ReduceSum over axis 1 that follows then gives one entry, the right
    count, whose value (the sum over 12) is a quarter of the true one (the
    sum of the column means, which is the sum over 3).
    Each call must refuse, or the result must have the true entry count and
    hold every output. One marked case per flat kernel (Star, Zono, Box:
    registry AFFINE-4); the passing neighbour, with the same builder and
    reference, is the row vector, for which axis 1 is every entry (a
    kept-dims mean followed by a sum over its one entry)."""
    lo, hi = flat_box(12, seed=17)
    layers = [_reducer(*node) for node in chain]
    what = f"{chain} on {kind} standing for a {shape} tensor"
    sets = [flat_set(kind, lo, hi)]
    try:
        for layer in layers:
            sets = reach_one_layer(layer, sets, "approx", what)
    except NotImplementedError:
        return
    outputs = _true_outputs(chain, samples(lo, hi), shape)
    out_shape = shape
    for node in chain:
        out_shape = _reduced_shape(out_shape, node[1], node[2])
    _assert_holds(sets, outputs, out_shape, what)


# --------------------------------------------------------------------------- #
# Through the model entry point                                                 #
# --------------------------------------------------------------------------- #
def _reduce_nodes(op, axes, keepdims, x, y, tag, noop=0):
    """One ONNX reduce node and its initializers. ``"ReduceSum"`` is the
    opset-13 form (axes as an int64 input), ``"ReduceSum11"`` and the others
    the attribute form."""
    attrs = {"keepdims": keepdims}
    if op == "ReduceSum":
        if noop:
            attrs["noop_with_empty_axes"] = 1
        if axes is None:
            return [helper.make_node("ReduceSum", [x], [y], **attrs)], {}
        name = f"axes_{tag}"
        return ([helper.make_node("ReduceSum", [x, name], [y], **attrs)],
                {name: np.array(axes, np.int64)})
    if axes is not None:
        attrs["axes"] = list(axes)
    onnx_op = "ReduceSum" if op == "ReduceSum11" else op
    return [helper.make_node(onnx_op, [x], [y], **attrs)], {}


def _twin(tmp_path, nodes, in_shape, out_shape, inits, name, opset):
    """The graph in float32 (what n2v reaches) and the same graph in float64
    (the reference, so that a mean over 3 or 5 entries is exact to float64)."""
    path32 = save_onnx(tmp_path, nodes, in_shape, out_shape, inits=inits,
                       name=name, opset=opset)
    inits64 = {k: (v.astype(np.float64) if v.dtype == np.float32 else v)
               for k, v in inits.items()}
    path64 = save_onnx(tmp_path, nodes, in_shape, out_shape, inits=inits64,
                       name=name + "64", opset=opset, in_type="DOUBLE",
                       out_type="DOUBLE")
    return path32, path64


def _opset(nodes_ops):
    return 11 if "ReduceSum11" in nodes_ops else 13


# Conv 1: (2, 3, 5) -> (4, 2, 4), a 2x2 kernel. Every axis it leaves has a
# power-of-two length and every weight, bias and input is dyadic, so a mean
# over any of its axes is dyadic and the float32 graph replayed by
# onnxruntime is exact: no slack beyond the LP's is needed.
_RK1 = ((np.arange(32).reshape(4, 2, 2, 2) * 3 % 7 - 3) / 4).astype(np.float32)
_RB1 = np.array([0.25, -0.5, 0.125, 0.75], np.float32)
_RMID = (1, 4, _RH - 1, _RW - 1)


@pytest.mark.parametrize("op,axes,kernel", [("ReduceMean", [2], (1, 2)),
                                            ("ReduceMean", [1], (2, 1)),
                                            ("ReduceSum", [1], (1, 2)),
                                            ("ReduceSum11", [3], (2, 1)),
                                            ("ReduceMean", [-3], (2, 2)),
                                            ("ReduceMean", [-2, -1], (1, 1))],
                         ids=_rid)
@pytest.mark.parametrize("kind,inputs", [("ImageStar", "box"),
                                         ("ImageZono", "box"),
                                         ("ImageStar", "binding row")])
def test_conv_reduce_conv_holds_every_onnxruntime_output(tmp_path, kind, inputs,
                                                         op, axes, kernel):
    """Conv, a Reduce with keepdims=1, Conv on a (2, 3, 5) image through
    ``reach_pytorch_model`` (with ``input_shape``). The Reduce sees a
    (4, 2, 4) image; after a reduction over the channels it is (1, 2, 4),
    with height and width both above 1 and different, and the second Conv
    (an asymmetric kernel) reads it with its stored height, width and channel
    count. Catches a reduced ImageZono whose rows are right and whose height
    and width are recorded swapped, a reduction over the wrong axis (a
    negative axis resolved off by one included), and mean and sum swapped."""
    mid = _reduced_shape(_RMID, axes, 1)
    k2 = (np.arange(2 * mid[1] * kernel[0] * kernel[1])
          .reshape(2, mid[1], *kernel) % 5 - 2).astype(np.float32) / 4
    out = (1, 2, mid[2] - kernel[0] + 1, mid[3] - kernel[1] + 1)
    reduce_nodes, inits = _reduce_nodes(op, axes, 1, "A", "R", "r")
    nodes = ([helper.make_node("Conv", ["X", "K1", "b1"], ["A"])] + reduce_nodes
             + [helper.make_node("Conv", ["R", "K2", "b2"], ["Y"])])
    inits.update(K1=_RK1, b1=_RB1, K2=k2, b2=np.array([0.5, -0.25], np.float32))
    path = save_onnx(tmp_path, nodes, [1, _RC, _RH, _RW], list(out), inits=inits,
                     name="crc", opset=_opset([op]))
    what = f"Conv, {op}(axes={axes}), Conv on {kind} {inputs}"
    if inputs == "binding row":
        input_set, points = constrained_image_star(_RLO, _RHI, seed=12)
    else:
        input_set, points = image_set(kind, _RLO, _RHI), samples(_RLO, _RHI)
    sets = reach_model(load_onnx(path), input_set, "approx", what,
                       input_shape=(_RC, _RH, _RW))
    assert_covered(sets, replay_onnx(path, points, [1, _RC, _RH, _RW]), what,
                   out_chw=out[1:])


# Gemm after the flattened reduction: 3 outputs, dyadic weights.
def _gemm_weights(n_in):
    g = np.arange(3 * n_in).reshape(3, n_in)
    return ((g * 5 % 9 - 4) / 8).astype(np.float32)


@pytest.mark.parametrize("op,axes", [("ReduceMean", [2, 3]), ("ReduceSum", [1]),
                                     ("ReduceMean", [-2])], ids=_rid)
@pytest.mark.parametrize("kind,inputs", [("ImageStar", "box"),
                                         ("ImageZono", "box"),
                                         ("ImageStar", "binding row")])
def test_conv_reduce_gemm_holds_every_onnxruntime_output(tmp_path, kind, inputs,
                                                         op, axes):
    """Conv, a Reduce with keepdims=0, Flatten, Gemm on a (2, 3, 5) image
    through ``reach_pytorch_model`` (with ``input_shape``): the global
    average pool of a classifier head (ReduceMean over the height and the
    width), a channel sum (the opset-13 ReduceSum), and a mean over the
    height by a negative axis, which leaves the channels and the width. The
    flat set the Reduce emits is read by the Gemm in the tensor's (C, H, W)
    order. Catches a flat result in the stored (height, width, channel)
    order, a reduction over the wrong axis, and mean and sum swapped, in the
    chain a model has. Every value is dyadic (means over 2 or 8 entries), so
    the float32 graph replayed by onnxruntime is exact."""
    mid = _reduced_shape(_RMID, axes, 0)
    n_mid = int(np.prod(mid))
    reduce_nodes, inits = _reduce_nodes(op, axes, 0, "A", "R", "r")
    nodes = ([helper.make_node("Conv", ["X", "K1", "b1"], ["A"])] + reduce_nodes
             + [helper.make_node("Flatten", ["R"], ["F"]),
                helper.make_node("Gemm", ["F", "W2", "b2"], ["Y"], transB=1)])
    inits.update(K1=_RK1, b1=_RB1, W2=_gemm_weights(n_mid),
                 b2=np.array([0.5, -0.25, 0.125], np.float32))
    path = save_onnx(tmp_path, nodes, [1, _RC, _RH, _RW], [1, 3], inits=inits,
                     name="crg", opset=_opset([op]))
    what = f"Conv, {op}(axes={axes}, keepdims=0), Flatten, Gemm on {kind} {inputs}"
    if inputs == "binding row":
        input_set, points = constrained_image_star(_RLO, _RHI, seed=13)
    else:
        input_set, points = image_set(kind, _RLO, _RHI), samples(_RLO, _RHI)
    sets = reach_model(load_onnx(path), input_set, "approx", what,
                       input_shape=(_RC, _RH, _RW))
    assert_covered(sets, replay_onnx(path, points, [1, _RC, _RH, _RW]), what)


# --------------------------------------------------------------------------- #
# The exact method on image sets                                                #
# --------------------------------------------------------------------------- #
# A (2, 3, 4) image: every mean below is over 2, 4 or 8 entries, so the exact
# image in rationals is the float64 result with no rounding (a mean over 3
# entries would show the affine-map rounding of PLAN step 7b instead).
_XCHW = (2, 3, 4)
_EXACT_REDUCTIONS = [("ReduceSum", [2], 0), ("ReduceMean", [3], 1),
                     ("ReduceMean", [-3, -1], 0), ("ReduceSum11", None, 1)]


def _low_rank_image(kind, seed=3):
    """``(set, c, G, A, b)``: a (2, 3, 4) image whose 24 entries are
    ``c + G a`` over three predicate variables, stored height, width,
    channel (``c``, ``G`` dyadic), and the predicate polytope ``A a <= b``:
    the cube [-1, 1]^3, cut for an ImageStar by the row a0 + a1 - a2 <= 1/2,
    which binds (the corner (1, 1, -1) is cut off)."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(_XCHW))
    c = rng.integers(-8, 9, size=n) / 4.0
    G = rng.integers(-4, 5, size=(n, 3)) / 4.0
    if np.linalg.matrix_rank(G) < 3:
        pytest.fail("setup: the generators are not independent")
    A = np.vstack([np.eye(3), -np.eye(3)])
    b = np.ones(6)
    channels, height, width = _XCHW
    if kind == "ImageZono":
        made = ImageZono(c.reshape(-1, 1), G, height, width, channels)
    else:
        A, b = np.vstack([A, [1.0, 1.0, -1.0]]), np.append(b, 0.5)
        made = ImageStar(np.hstack([c[:, None], G]), A, b.reshape(-1, 1),
                         -np.ones((3, 1)), np.ones((3, 1)), height=height,
                         width=width, num_channels=channels)
    return made, c, G, A, b


def _exact_reduce_pieces(c, G, A, b, op, axes, keepdims):
    """The exact image of ``{c + G a : A a <= b}`` (stored height, width,
    channel) under the reduction, in the layout the result stores: the
    tensor's row-major order for a flat output, height, width, channel for
    an image (keepdims=1). The reduction matrix comes from the ONNX
    definition applied to the unit vectors (0 or 1 for a sum, divided by the
    count for a mean), in fractions."""
    in_shape = (1,) + _XCHW
    n = int(np.prod(_XCHW))
    to_chw = np.empty(n, dtype=int)
    to_chw[hwc_index(*_XCHW)] = np.arange(n)       # x_chw = x_stored[to_chw]
    sum_op = "ReduceSum11" if op == "ReduceMean" else op
    S = np.stack([_true_outputs([(sum_op, axes, 0)], [np.eye(n)[i]], in_shape)[0]
                  for i in range(n)], axis=1)
    weight = (Fraction(1, int(S.sum(axis=1)[0])) if op == "ReduceMean"
              else Fraction(1))
    R = [[weight * int(v) for v in row] for row in S]
    out_shape = _reduced_shape(in_shape, axes, keepdims)
    rows = list(range(len(R)))
    if len(out_shape) == 4:
        rows = list(hwc_index(*out_shape[1:]))
    M = [[sum((R[j][i] * Fraction(float(G[to_chw[i], k])) for i in range(n)),
              Fraction(0)) for k in range(G.shape[1])] for j in rows]
    m = [sum((R[j][i] * Fraction(float(c[to_chw[i]])) for i in range(n)),
             Fraction(0)) for j in rows]
    return rational.affine(rational.polytope(A.tolist(), b.tolist()), M, m), out_shape


@pytest.mark.parametrize("op,axes,keepdims", _EXACT_REDUCTIONS, ids=_rid)
@pytest.mark.parametrize("kind,route", [("ImageStar", "model"),
                                        ("ImageStar", "reach_layer"),
                                        ("ImageZono", "reach_layer")])
def test_exact_reduce_on_an_image_set_is_the_exact_image(tmp_path, kind, route,
                                                         op, axes, keepdims):
    """``method='exact'`` on an image whose entries are an affine image of
    three predicate variables (an ImageStar cut by a binding row), through
    a one-node ONNX model and through ``reach_layer``; on an ImageZono
    through ``reach_layer``, which answers ``exact`` there (the model entry
    refuses it: the cell below). Both directions against the exact image in
    rationals with no slack: every vertex of the exact image is in the
    returned set, and every vertex of the returned set is in the exact
    image. Catches a reduction over the wrong entries or with the wrong
    weight, a result that loses or widens the constraint rows or the
    predicate bounds (a set that is sound but not the image), and a result
    stored in the wrong layout."""
    made, c, G, A, b = _low_rank_image(kind)
    pieces, out_shape = _exact_reduce_pieces(c, G, A, b, op, axes, keepdims)
    what = f"exact {op}(axes={axes}, keepdims={keepdims}) on {kind} via {route}"
    if route == "model":
        nodes, inits = _reduce_nodes(op, axes, keepdims, "X", "Y", "r")
        path = save_onnx(tmp_path, nodes, [1] + list(_XCHW), list(out_shape),
                         inits=inits, name="xr", opset=_opset([op]))
        sets = reach_model(load_onnx(path), made, "exact", what,
                           input_shape=_XCHW)
    else:
        sets = reach_one_layer(_reducer(op, axes, keepdims), [made], "exact",
                               what)
    assert len(sets) == 1, f"{what}: {len(sets)} sets for one linear map"
    out = sets[0]
    if len(out_shape) == 4:
        stored = (getattr(out, "height", None), getattr(out, "width", None),
                  getattr(out, "num_channels", None))
        assert stored == (out_shape[2], out_shape[3], out_shape[1]), (
            f"{what}: the result stores {stored} (height, width, channels) "
            f"for a {out_shape} output")
    computed = rational.star(out)
    assert len(computed[0]) == len(pieces[0].c), (
        f"{what}: the result has {len(computed[0])} entries, the output "
        f"{len(pieces[0].c)}")
    if len(computed[4]) > 4:
        pytest.fail(f"{what}: the result has {len(computed[4])} predicate "
                    f"variables, more than the vertex enumeration takes")
    escaped = rational.escapes(pieces, computed)
    assert not escaped, (f"{what}: {len(escaped)} vertices of the exact image "
                         f"are outside the result, e.g. "
                         f"{[float(v) for v in escaped[0][1]]}")
    extra = rational.unreached(pieces, rational.set_vertices(computed))
    assert not extra, (f"{what}: {len(extra)} vertices of the result are not "
                       f"in the exact image, e.g. "
                       f"{[float(v) for v in extra[0][1]]}")


def test_exact_on_an_image_zono_through_the_model_is_refused(tmp_path):
    """The model entry has no exact method for a zonotope: ``method='exact'``
    on an ImageZono through a ReduceMean graph raises ``ValueError`` before
    the walk (the row (exact, ImageZono) refuses; whether ``ValueError`` is
    the right type is decided in ``tests/contracts/test_reach_coverage.py``).
    Catches an exact request answered with a zonotope's result."""
    nodes, inits = _reduce_nodes("ReduceMean", [3], 1, "X", "Y", "r")
    path = save_onnx(tmp_path, nodes, [1] + list(_XCHW), [1, 2, 3, 1],
                     inits=inits, name="xz", opset=13)
    made = _low_rank_image("ImageZono")[0]
    with pytest.raises((ValueError, NotImplementedError)):
        reach_model(load_onnx(path), made, "exact", "exact on ImageZono",
                    input_shape=_XCHW)


# --------------------------------------------------------------------------- #
# Entry routes that refuse today                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("keepdims,noop", [(0, 0), (1, 0), (0, 1)],
                         ids=["keepdims0", "keepdims1", "noop"])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_reduce_sum_without_an_axes_input_refuses_or_is_sound(tmp_path, kind,
                                                              keepdims, noop):
    """An opset-13 ReduceSum with no axes input (every axis, or the input
    itself with ``noop_with_empty_axes``) loads as onnx2torch's
    ``OnnxReduceSumDynamicAxes``, which the reach does not take today:
    through the model and through ``reach_layer`` (the module alone, whose
    forward without an axes tensor reduces the same) the call must refuse,
    or hold every onnxruntime output. Catches a handler for the
    dynamic-axes module that reads its axes, keepdims or no-op flag
    wrongly."""
    from onnx2torch.node_converters.reduce import OnnxReduceSumDynamicAxes
    attrs = {"keepdims": keepdims}
    if noop:
        attrs["noop_with_empty_axes"] = 1
    in_shape = [1, _RC, _RH, _RW]
    out_shape = in_shape if noop else ([1, 1, 1, 1] if keepdims else [])
    path = save_onnx(tmp_path, [helper.make_node("ReduceSum", ["X"], ["Y"],
                                                 **attrs)],
                     in_shape, out_shape, name="rd", opset=13)
    outputs = replay_onnx(path, samples(_RLO, _RHI), in_shape)
    out_chw = tuple(out_shape[1:]) if len(out_shape) == 4 else None
    what = f"ReduceSum without axes (keepdims={keepdims}, noop={noop}) on {kind}"
    for route in ("model", "reach_layer"):
        try:
            if route == "model":
                sets = reach_model(load_onnx(path), image_set(kind, _RLO, _RHI),
                                   "approx", what, input_shape=(_RC, _RH, _RW))
            else:
                sets = reach_one_layer(OnnxReduceSumDynamicAxes(keepdims, noop),
                                       [image_set(kind, _RLO, _RHI)], "approx",
                                       what)
        except NotImplementedError:
            continue
        assert_covered(sets, outputs, f"{what} via {route}", out_chw=out_chw)


class _MeanOverHeight(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=2, keepdim=True)


class _SumOverHeightAndWidth(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=(2, 3))


_TORCH_REDUCERS = {
    "onnx2torch-module": (lambda: torch.nn.Sequential(
        OnnxReduceStaticAxes("ReduceMean", [2], keepdims=1)), (2, 1, 5)),
    "tensor-method": (lambda: _SumOverHeightAndWidth(), None),
    "mean-method": (lambda: _MeanOverHeight(), (2, 1, 5)),
}


@pytest.mark.parametrize("model", sorted(_TORCH_REDUCERS))
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_reduce_in_a_torch_model_refuses_or_is_sound(kind, model):
    """A torch model (not an ONNX file) that reduces: an onnx2torch Reduce
    module inside ``nn.Sequential`` (torch.fx traces into it, so the walk sees
    ``torch.mean``), and a forward that calls ``x.mean`` or ``x.sum``. The
    walk has no rule for these today and refuses; the promise is a refusal
    or a result that holds every torch float64 output. Catches a rule added
    for them that reduces the wrong axis or keeps the wrong layout."""
    build, out_chw = _TORCH_REDUCERS[model]
    module = build()
    points = samples(_RLO, _RHI)
    outputs = replay_torch(module, points, (1, _RC, _RH, _RW))
    what = f"{model} on {kind} through the torch model entry"
    try:
        sets = reach_model(module, image_set(kind, _RLO, _RHI), "approx", what)
    except NotImplementedError:
        return
    assert_covered(sets, outputs, what, out_chw=out_chw)


@pytest.mark.parametrize("axis", [0, -1, 2])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_reduce_after_an_image_reduce_without_input_shape_refuses_or_is_sound(
        tmp_path, kind, axis):
    """ReduceMean over the width with keepdims=0 (a flat set of the (1, 2, 3)
    tensor), then ReduceSum over axis 0, -1 or 2 of it, as one ONNX graph
    through ``reach_pytorch_model`` without ``input_shape``. An ImageZono
    gets no tracked tensor shape then, so the second Reduce takes the flat
    kernel, which knows no shape: it must refuse, or hold every onnxruntime
    output (axis 1 is wrong today, registry SHAPE-3, marked in
    ``tests/soundness/test_soundness_tensor_shape.py::test_an_image_set_without_input_shape_holds_every_output``;
    these are the other spellings of the axis). The ImageStar, whose shape
    is tracked, is the neighbour. Catches a flat kernel that reads the batch
    axis, a negative axis or an inner axis of an unknown shape as some other
    reduction."""
    nodes = (_reduce_nodes("ReduceMean", [3], 0, "X", "A", "0")[0]
             + _reduce_nodes("ReduceSum11", [axis], 0, "A", "Y", "1")[0])
    in_shape = [1, _RC, _RH, _RW]
    out_shape = list(_reduced_shape((1, _RC, _RH), [axis], 0))
    path32, path64 = _twin(tmp_path, nodes, in_shape, out_shape, {}, "rn", 11)
    what = f"ReduceMean(3), ReduceSum({axis}) on {kind} without input_shape"
    try:
        sets = reach_model(load_onnx(path32), image_set(kind, _RLO, _RHI),
                           "approx", what)
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path64, samples(_RLO, _RHI), in_shape),
                   what)


def _flat_model(tmp_path, shape, reduce_spec, relu=False, name="mrm"):
    """X ``shape`` -> Mul -> Add [-> Relu] -> Reduce -> Mul -> Add, with
    elementwise dyadic constants in front of and behind the Reduce."""
    op, axes, keepdims, noop = reduce_spec
    out = _reduced_shape(shape, axes, keepdims, noop)
    n, m = int(np.prod(shape)), int(np.prod(out))
    g, h = np.arange(n), np.arange(m)
    inits = dict(m1=((g % 4 + 1) / 2).reshape(shape).astype(np.float32),
                 a1=((g % 3 - 1) / 4).reshape(shape).astype(np.float32),
                 m2=((h % 3 + 1) / 4 * (-1.0) ** h).reshape(out).astype(np.float32),
                 a2=((h % 5 - 2) / 8).reshape(out).astype(np.float32))
    nodes = [helper.make_node("Mul", ["X", "m1"], ["A"]),
             helper.make_node("Add", ["A", "a1"], ["B"])]
    if relu:
        nodes.append(helper.make_node("Relu", ["B"], ["C"]))
    reduce_nodes, extra = _reduce_nodes(op, axes, keepdims, "C" if relu else "B",
                                        "R", "r", noop=noop)
    inits.update(extra)
    nodes += reduce_nodes + [helper.make_node("Mul", ["R", "m2"], ["U"]),
                             helper.make_node("Add", ["U", "a2"], ["Y"])]
    return _twin(tmp_path, nodes, list(shape), list(out), inits, name, _opset([op]))


_SHAPED_REDUCTIONS = [
    ("ReduceMean", [2], 0, 0), ("ReduceSum", [1], 1, 0), ("ReduceSum11", [1, 2], 0, 0),
    ("ReduceMean", None, 1, 0), ("ReduceSum", [-1], 0, 0), ("ReduceMean", [0], 1, 0),
    ("ReduceSum11", [0, 2], 0, 0), ("ReduceSum", [], 0, 1),
]


# The box of the model cells: its entries straddle zero, are positive and are
# negative (``mixed``); the same box moved above zero (``positive``: every
# entry of the Reduce's input is then at least 0, so a mean taken with a
# wrong divisor misses the outputs instead of hiding inside the range of the
# right one); and the box scaled by 8 (``wide``).
_MLO, _MHI = flat_box(12, seed=19)
_MODEL_DATA = {"mixed": (_MLO, _MHI),
               "positive": (_MLO - _MLO.min() + 0.5, _MHI - _MLO.min() + 0.5),
               "wide": (_MLO * 8, _MHI * 8)}
# The sums and the batch-axis mean on the mixed box; the means over one axis
# and over every axis (lengths 1, 3, 4: their sum is not their product) on
# the positive box; a sum over two axes on the wide box. The mean without
# axes runs with keepdims 1 and 0: with keepdims=1 the onnx2torch module
# writes every axis into its ``axes`` during the shape-recording forward, so
# only keepdims=0 reaches the handler's no-axes branch.
_POSITIVE_SHAPED = (("ReduceMean", [2], 0, 0), ("ReduceMean", None, 1, 0),
                    ("ReduceMean", None, 0, 0))
_SHAPED_CELLS = (
    [pytest.param(spec, "mixed", id=_rid(spec)) for spec in _SHAPED_REDUCTIONS
     if spec not in _POSITIVE_SHAPED]
    + [pytest.param(spec, "positive", id=f"{_rid(spec)}-positive")
       for spec in _POSITIVE_SHAPED]
    + [pytest.param(("ReduceSum11", [1, 2], 0, 0), "wide",
                    id="[ReduceSum11,[1, 2],0,0]-wide")])


@pytest.mark.parametrize("spec,data", _SHAPED_CELLS)
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_reduce_of_a_flat_set_with_a_known_shape_is_sound(tmp_path, kind, spec,
                                                          data):
    """Mul, Add, Reduce, Mul, Add on a flat set standing for a (1, 3, 4) tensor,
    with ``input_shape``: the model entry builds the reduction matrix from the
    tracked shape (``_handle_onnx_reduce_flat``). Over one axis, both axes, a
    negative axis, the batch axis, no axes, and no axes with
    ``noop_with_empty_axes``, on a box straddling zero, above zero (where a
    mean divided by the wrong count misses: the divisor taken from the
    output shape, summed instead of multiplied over several axes, or the
    number of axes) and scaled by 8. Catches a
    matrix built from the wrong shape or axis, mean and sum swapped, a
    wrong divisor, a no-op flag read wrongly, and the constants behind it
    applied to permuted entries."""
    path32, path64 = _flat_model(tmp_path, (1, 3, 4), spec)
    lo, hi = _MODEL_DATA[data]
    what = f"Mul, Add, {spec}, Mul, Add on {kind} (tensor (1, 3, 4), {data})"
    sets = reach_model(load_onnx(path32), flat_set(kind, lo, hi), "approx",
                       what, input_shape=(3, 4))
    assert_covered(sets, replay_onnx(path64, samples(lo, hi), [1, 3, 4]), what)


_HEXOCT_MODEL_CELLS = [("Hexatope", ("ReduceMean", [2], 0, 0)),
                       ("Octatope", ("ReduceSum", [1], 1, 0))]


@pytest.mark.parametrize(
    "kind,spec", _HEXOCT_MODEL_CELLS,
    ids=[f"{kind}-{_rid(spec)}" for kind, spec in _HEXOCT_MODEL_CELLS])
def test_reduce_of_a_hexatope_or_octatope_through_the_model(tmp_path, kind, spec):
    """A graph that is a single Reduce node, on a Hexatope or Octatope standing
    for a (1, 3, 4) tensor, with ``input_shape``: neither type has a Reduce
    kernel, so the call must refuse, or (should one be added) hold every
    onnxruntime output. Catches a Reduce answered as the identity, or as a
    reduction over every entry, for these two types. The strict refusal is
    ``test_reduce_on_a_hexatope_or_octatope_is_refused``; this cell is the
    promise for the day a Reduce rule is added."""
    op, axes, keepdims, noop = spec
    nodes, inits = _reduce_nodes(op, axes, keepdims, "X", "Y", "r", noop=noop)
    path = save_onnx(tmp_path, nodes, [1, 3, 4],
                     list(_reduced_shape((1, 3, 4), axes, keepdims)),
                     inits=inits, name="r", opset=_opset([op]))
    lo, hi = flat_box(12, seed=19)
    what = f"{spec} on {kind} (tensor (1, 3, 4))"
    try:
        sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                           what, input_shape=(3, 4))
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, 3, 4]), what)


def test_exact_relu_hands_several_stars_to_a_flat_reduce(tmp_path):
    """Mul, Add, ReLU, ReduceSum over axis 2, Mul, Add on a Star with
    ``method='exact'`` and a known shape: the exact ReLU splits the Star, so
    the shape-aware Reduce maps a list of several Stars. Catches a reduction
    applied to the first Star only."""
    path32, path64 = _flat_model(tmp_path, (1, 3, 4), ("ReduceSum", [2], 0, 0),
                                 relu=True, name="mrrm")
    lo, hi = flat_box(12, seed=23)
    what = "Mul, Add, ReLU, ReduceSum(axes=[2]), Mul, Add on Star (exact)"
    sets = reach_model(load_onnx(path32), flat_set("Star", lo, hi), "exact", what,
                       input_shape=(3, 4))
    if len(sets) < 2:
        pytest.fail(f"{what}: the exact ReLU returned {len(sets)} set; the cell "
                    f"needs several sets at the Reduce")
    assert_covered(sets, replay_onnx(path64, samples(lo, hi), [1, 3, 4]), what)


@pytest.mark.parametrize("chain", ["mean3-sum1", "mean3-mean1-sum1"])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_reduce_chain_on_an_image_set_through_the_model_is_sound(tmp_path, kind,
                                                                 chain):
    """The chains of the ``reach_layer`` cell above as one ONNX graph through
    ``reach_pytorch_model`` with ``input_shape``: the walker tracks the
    tensor shape of the flat set between the nodes, so the second and third
    Reduce act on the right axis. The neighbour that passes of the marked
    ``reach_layer`` cells."""
    nodes, inits, shape = [], {}, (1, _RC, _RH, _RW)
    names = ["X"] + [f"R{k}" for k in range(len(_CHAINS[chain]) - 1)] + ["Y"]
    for k, (op, axes, keepdims) in enumerate(_CHAINS[chain]):
        more, extra = _reduce_nodes(op, axes, keepdims, names[k], names[k + 1],
                                    str(k))
        nodes += more
        inits.update(extra)
        shape = _reduced_shape(shape, axes, keepdims)
    ops = [node[0] for node in _CHAINS[chain]]
    if "ReduceSum11" in ops:   # one opset per graph: the input form of ReduceSum
        nodes, inits = [], {}  # is not in opset 11, so use ReduceSum11 throughout
        for k, (op, axes, keepdims) in enumerate(_CHAINS[chain]):
            op = "ReduceSum11" if op == "ReduceSum" else op
            more, extra = _reduce_nodes(op, axes, keepdims, names[k],
                                        names[k + 1], str(k))
            nodes += more
    path32, path64 = _twin(tmp_path, nodes, [1, _RC, _RH, _RW], list(shape), inits,
                           "rr", _opset(ops))
    what = f"{chain} on {kind} through the model"
    sets = reach_model(load_onnx(path32), image_set(kind, _RLO, _RHI), "approx",
                       what, input_shape=(_RC, _RH, _RW))
    assert_covered(sets, replay_onnx(path64, samples(_RLO, _RHI),
                                     [1, _RC, _RH, _RW]), what)


_CHAIN_MODEL_CELLS = [(kind, data) for data in ("mixed", "positive")
                      for kind in ("Star", "Zono", "Box")]


@pytest.mark.parametrize(
    "kind,data", _CHAIN_MODEL_CELLS,
    ids=[kind if data == "mixed" else f"{kind}-{data}"
         for kind, data in _CHAIN_MODEL_CELLS])
def test_reduce_chain_on_a_flat_set_of_a_matrix_through_the_model_is_sound(
        tmp_path, kind, data):
    """ReduceMean over axis 1, then ReduceSum over axis 1, of a flat set
    standing for a (1, 3, 4) tensor, as one graph with ``input_shape``: the
    neighbour that passes of the marked flat ``reach_layer`` cell. The walker
    must track the (1, 4) shape between the two nodes. On a box straddling
    zero and on the same box above zero, where the mean's divisor shows.
    Catches a shape recorded for the wrong
    node, a mean taken as a sum, and a sum divided like a mean."""
    nodes = (_reduce_nodes("ReduceMean", [1], 0, "X", "A", "0")[0]
             + _reduce_nodes("ReduceSum", [1], 0, "A", "Y", "1")[0])
    inits = _reduce_nodes("ReduceSum", [1], 0, "A", "Y", "1")[1]
    path32, path64 = _twin(tmp_path, nodes, [1, 3, 4], [1], inits, "fr", 13)
    lo, hi = flat_box(12, seed=17)
    if data == "positive":
        lo, hi = lo - lo.min() + 0.5, hi - lo.min() + 0.5
    what = f"ReduceMean(1), ReduceSum(1) on {kind} through the model ({data})"
    sets = reach_model(load_onnx(path32), flat_set(kind, lo, hi), "approx", what,
                       input_shape=(3, 4))
    assert_covered(sets, replay_onnx(path64, samples(lo, hi), [1, 3, 4]), what)


# --------------------------------------------------------------------------- #
# Flat sets: the refusals of the rows the library does not answer               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean", "ReduceMax"])
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_reduce_on_a_hexatope_or_octatope_is_refused(tmp_path, kind, op):
    """The rows (approx and exact) x (Hexatope, Octatope) of ReduceSum,
    ReduceMean and the other Reduce ops: neither type has a Reduce rule.
    ``reach_layer`` must raise ``NotImplementedError`` for either method (the
    else-branch of the type's dispatcher lane), the model entry
    ``NotImplementedError`` for ``approx`` (the graph handler takes only
    Star, Zono and Box and hands these to the dispatcher) and a refusal for
    ``exact`` (today ``ValueError``, before the walk: the type has no exact
    method). Catches a Reduce answered for these types, for example passed
    through as the identity (which returns the input's entries where one is
    due)."""
    lo, hi = flat_box(12, seed=19)
    what = f"{op} on {kind}"
    for method in ("approx", "exact"):
        with pytest.raises(NotImplementedError):
            reach_one_layer(_reducer(op, [1], 0), [flat_set(kind, lo, hi)],
                            method, f"{what} ({method}) through reach_layer")
    nodes, inits = _reduce_nodes(op, [2], 0, "X", "Y", "r")
    path = save_onnx(tmp_path, nodes, [1, 3, 4], [1, 3], inits=inits,
                     name="ho", opset=_opset([op]))
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                    f"{what} (approx) through the model", input_shape=(3, 4))
    with pytest.raises((ValueError, NotImplementedError)):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "exact",
                    f"{what} (exact) through the model", input_shape=(3, 4))


@pytest.mark.parametrize("kind", ["Zono", "Box"])
def test_exact_reduce_on_a_zono_or_box_through_the_model_is_refused(tmp_path,
                                                                    kind):
    """The rows (exact, Zono) and (exact, Box): the model entry has no exact
    method for these types and refuses ``method='exact'`` before the walk
    (today ``ValueError``; whether that type is right is decided in
    ``tests/contracts/test_reach_coverage.py``). ``reach_layer`` does not
    read the method and answers: the image of the set for a Zono
    (``test_exact_reduce_on_a_flat_set_is_the_exact_image``), the outward
    rounded interval for a Box (row REACH-8 leaves to the owner whether that lane
    may answer). Catches an exact request answered with the approx result
    through the model."""
    nodes, inits = _reduce_nodes("ReduceSum", [1], 0, "X", "Y", "r")
    path = save_onnx(tmp_path, nodes, [1, 3, 4], [1, 4], inits=inits,
                     name="xzb", opset=13)
    lo, hi = flat_box(12, seed=19)
    with pytest.raises((ValueError, NotImplementedError)):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "exact",
                    f"exact ReduceSum on {kind}", input_shape=(3, 4))


@pytest.mark.parametrize("kind,op", [("Star", "ReduceMax"), ("Zono", "ReduceMax"),
                                     ("Box", "ReduceMax"), ("Star", "ReduceL2")])
def test_non_affine_reduce_of_a_flat_set_through_the_model_is_refused(
        tmp_path, kind, op):
    """The rows of the other Reduce ops (ReduceMax and the like, which are
    the identity over size-1 axes; ReduceL2 and the like, which never are)
    on Star, Zono and Box through the model, over an axis of length 4 of a
    (1, 3, 4) tensor with ``input_shape``: the graph handler must hand them
    to the kernel, which raises ``NotImplementedError``, for ``approx`` and,
    on a Star (the one flat type the model entry lets ``exact`` through
    for), for ``exact``. Catches the handler letting a non-affine op through
    to its sum matrix (a maximum answered as a sum). The ``reach_layer``
    refusals are
    ``tests/soundness/test_leftover_reduce.py::test_non_affine_reduce_is_refused``."""
    nodes, _ = _reduce_nodes(op, [2], 0, "X", "Y", "r")
    path = save_onnx(tmp_path, nodes, [1, 3, 4], [1, 3], name="na", opset=13)
    lo, hi = flat_box(12, seed=19)
    for method in (("approx", "exact") if kind == "Star" else ("approx",)):
        with pytest.raises(NotImplementedError):
            reach_model(load_onnx(path), flat_set(kind, lo, hi), method,
                        f"{op}(axes=[2]) on {kind} through the model ({method})",
                        input_shape=(3, 4))


# --------------------------------------------------------------------------- #
# Flat sets: the exact method                                                  #
# --------------------------------------------------------------------------- #
def _low_rank_flat(kind, n, seed=3):
    """``(set, c, G, A, b)``: a flat set of ``n`` entries ``c + G a`` over
    three predicate variables (``c``, ``G`` dyadic) and the predicate
    polytope ``A a <= b``: the cube [-1, 1]^3, cut for a Star by the row
    a0 + a1 - a2 <= 1/2, which binds (the corner (1, 1, -1) is cut off)."""
    rng = np.random.default_rng(seed)
    c = rng.integers(-8, 9, size=n) / 4.0
    G = rng.integers(-4, 5, size=(n, 3)) / 4.0
    if np.linalg.matrix_rank(G) < 3:
        pytest.fail("setup: the generators are not independent")
    A, b = np.vstack([np.eye(3), -np.eye(3)]), np.ones(6)
    if kind == "Zono":
        return Zono(c.reshape(-1, 1), G), c, G, A, b
    A, b = np.vstack([A, [1.0, 1.0, -1.0]]), np.append(b, 0.5)
    made = Star(np.hstack([c[:, None], G]), A, b.reshape(-1, 1),
                -np.ones((3, 1)), np.ones((3, 1)))
    return made, c, G, A, b


def _exact_flat_pieces(c, G, A, b, op, axes, keepdims, shape):
    """The exact image of ``{c + G a : A a <= b}`` (the tensor's row-major
    order) under the reduction of a tensor of ``shape``, in fractions. The
    reduction matrix comes from the ONNX definition applied to the unit
    vectors (0 or 1 for a sum, divided by the count for a mean)."""
    n = int(np.prod(shape))
    sum_op = "ReduceSum11" if op == "ReduceMean" else op
    S = np.stack([_true_outputs([(sum_op, axes, keepdims)], [np.eye(n)[i]],
                                shape)[0] for i in range(n)], axis=1)
    weight = (Fraction(1, int(S.sum(axis=1)[0])) if op == "ReduceMean"
              else Fraction(1))
    M = [[sum((weight * int(S[j, i]) * Fraction(float(G[i, k]))
               for i in range(n)), Fraction(0)) for k in range(G.shape[1])]
         for j in range(S.shape[0])]
    m = [sum((weight * int(S[j, i]) * Fraction(float(c[i])) for i in range(n)),
             Fraction(0)) for j in range(S.shape[0])]
    return rational.affine(rational.polytope(A.tolist(), b.tolist()), M, m)


# (kind, route, op, axes, keepdims, tensor): every mean is over 4 or 8
# entries, so the exact image is the float64 result with no rounding (a mean
# over 3 or 6 entries would show the affine-map rounding of PLAN step 7b,
# registry ROUND-8, marked in tests/soundness/test_soundness_graph_walk.py).
_FLAT_EXACT = [
    ("Star", "reach_layer", "ReduceMean", [1], 0, (1, 8)),
    ("Star", "reach_layer", "ReduceSum", None, 1, (1, 8)),
    ("Zono", "reach_layer", "ReduceMean", [1], 0, (1, 8)),
    ("Zono", "reach_layer", "ReduceSum11", [0, 1], 1, (1, 8)),
    ("Star", "model", "ReduceMean", [2], 0, (1, 2, 4)),
    ("Star", "model", "ReduceSum", [1], 1, (1, 2, 4)),
]


@pytest.mark.parametrize(
    "kind,route,op,axes,keepdims,shape", _FLAT_EXACT,
    ids=[f"{k}-{r}-{o}-{_rid(a)}-{kd}-{'x'.join(map(str, t))}"
         for k, r, o, a, kd, t in _FLAT_EXACT])
def test_exact_reduce_on_a_flat_set_is_the_exact_image(tmp_path, kind, route, op,
                                                       axes, keepdims, shape):
    """``method='exact'`` on a flat set whose entries are an affine image of
    three predicate variables (a Star cut by a binding row; a Zono), through
    ``reach_layer`` on a set standing for a (1, 8) row vector and, for the
    Star, through a one-node ONNX model on a set standing for a (1, 2, 4)
    tensor (``input_shape`` given; the Reduce is the first node, so the
    placeholder's shape is the one it reads). ``reach_layer`` does not read
    the method and answers on a Zono, with the image of the set (the model
    entry refuses: ``test_exact_reduce_on_a_zono_or_box_through_the_model_is_refused``).
    Both directions against the exact image in rationals with no slack:
    every vertex of the exact image is in the returned set, and every vertex
    of the returned set is in the exact image. Catches a reduction over the
    wrong entries or with the wrong weight, a placeholder shape one axis
    short, and a result that loses or widens the constraint rows or the
    predicate bounds (a set that is sound but not the image)."""
    made, c, G, A, b = _low_rank_flat(kind, int(np.prod(shape)))
    pieces = _exact_flat_pieces(c, G, A, b, op, axes, keepdims, shape)
    what = (f"exact {op}(axes={axes}, keepdims={keepdims}) on {kind} "
            f"standing for {shape} via {route}")
    if route == "model":
        nodes, inits = _reduce_nodes(op, axes, keepdims, "X", "Y", "r")
        path = save_onnx(tmp_path, nodes, list(shape),
                         list(_reduced_shape(shape, axes, keepdims)),
                         inits=inits, name="xf", opset=_opset([op]))
        sets = reach_model(load_onnx(path), made, "exact", what,
                           input_shape=shape[1:])
    else:
        sets = reach_one_layer(_reducer(op, axes, keepdims), [made], "exact",
                               what)
    assert len(sets) == 1, f"{what}: {len(sets)} sets for one linear map"
    computed = rational.star(sets[0])
    assert len(computed[0]) == len(pieces[0].c), (
        f"{what}: the result has {len(computed[0])} entries, the output "
        f"{len(pieces[0].c)}")
    if len(computed[4]) > 4:
        pytest.fail(f"{what}: the result has {len(computed[4])} predicate "
                    f"variables, more than the vertex enumeration takes")
    escaped = rational.escapes(pieces, computed)
    assert not escaped, (f"{what}: {len(escaped)} vertices of the exact image "
                         f"are outside the result, e.g. "
                         f"{[float(v) for v in escaped[0][1]]}")
    extra = rational.unreached(pieces, rational.set_vertices(computed))
    assert not extra, (f"{what}: {len(extra)} vertices of the result are not "
                       f"in the exact image, e.g. "
                       f"{[float(v) for v in extra[0][1]]}")


# --------------------------------------------------------------------------- #
# Flat sets: entry routes without a tracked shape, and routes that refuse      #
# --------------------------------------------------------------------------- #
_NO_TRACKED_SHAPE = known_bug(
    12, "a flat set reached through an ONNX model whose input has rank 3, "
        "without input_shape, gets no tracked tensor shape when the "
        "placeholder forward fails, and the Reduce then takes the flat "
        "kernel, which reads axis 1 as every entry: ReduceSum over axis 1 of "
        "a set standing for a (1, 3, 4) tensor returns one entry where the "
        "model has four (registry SHAPE-2, on the path where no shape is "
        "tracked; the kernel's reading of axis 1 is AFFINE-4)")

_K3 = ("Star", "Zono", "Box")
_NO_SHAPE_CELLS = (
    [(kind, "row-vector", axis) for kind in _K3 for axis in (0, 1)]
    + [(kind, "behind-a-mul", axis) for kind in _K3 for axis in (0, -1, 2)]
    + [pytest.param("Star", "behind-a-mul", 1, marks=_NO_TRACKED_SHAPE,
                    id="Star-behind-a-mul-1")])


@pytest.mark.parametrize("kind,tensor,axis", _NO_SHAPE_CELLS)
def test_reduce_through_a_model_without_input_shape_refuses_or_is_sound(
        tmp_path, kind, tensor, axis):
    """An ONNX model reached without ``input_shape``, the way a flat set is
    usually reached. ``row-vector``: ReduceMean (keepdims=1) over the batch
    axis or the feature axis of a (1, 6) input, on the box above zero; the
    placeholder (1, 6) is the model's input, so the handler reads the right
    shape. ``behind-a-mul``: a Mul by a (1, 3, 4) constant, then ReduceSum
    (keepdims=0) over axis 0, -1, 2 or 1 of a set standing for a (1, 3, 4)
    tensor; both placeholders, (1, 12) and (12,), fail at the Mul, so no
    shape is tracked and the Reduce takes the flat kernel, which knows no
    shape: it must refuse, or hold every output (axis 1 is wrong today,
    marked; the batch axis is the identity, -1 and 2 are refused). Catches
    the unbatched placeholder tried first (the batch-axis reduction of a
    row vector then sums every entry), and a handler that, with no shape,
    takes the set for a row vector (axis 2 then reads as the batch axis, -1
    as every entry)."""
    if tensor == "row-vector":
        shape, (lo, hi), inits, opset = (1, _FLAT_N), (_FLO_POS, _FHI_POS), {}, 13
        nodes = _reduce_nodes("ReduceMean", [axis], 1, "X", "Y", "r")[0]
        out_shape = _reduced_shape(shape, [axis], 1)
    else:
        shape, (lo, hi), opset = (1, 3, 4), flat_box(12, seed=19), 11
        inits = {"m1": ((np.arange(12) % 4 + 1) / 2).reshape(shape)
                 .astype(np.float32)}
        nodes = ([helper.make_node("Mul", ["X", "m1"], ["A"])]
                 + _reduce_nodes("ReduceSum11", [axis], 0, "A", "Y", "r")[0])
        out_shape = _reduced_shape(shape, [axis], 0)
    path32, path64 = _twin(tmp_path, nodes, list(shape), list(out_shape), inits,
                           "ns", opset)
    what = f"Reduce over axis {axis} of a {tensor} ({shape}) on {kind}, no input_shape"
    try:
        sets = reach_model(load_onnx(path32), flat_set(kind, lo, hi), "approx",
                           what)
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path64, samples(lo, hi), list(shape)), what)


class _SumOverFeatures(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=1, keepdim=True)


_FLAT_TORCH_REDUCERS = {
    "onnx2torch-module": (lambda: torch.nn.Sequential(
        OnnxReduceStaticAxes("ReduceMean", [2], keepdims=0)), (1, 2, 3)),
    "tensor-method": (lambda: _SumOverFeatures(), (1, _FLAT_N)),
}


@pytest.mark.parametrize("kind,model", [
    ("Star", "onnx2torch-module"), ("Zono", "onnx2torch-module"),
    ("Box", "onnx2torch-module"), ("Star", "tensor-method")])
def test_reduce_in_a_torch_model_on_a_flat_set_refuses_or_is_sound(kind, model):
    """A torch model (not an ONNX file) that reduces a flat set: an onnx2torch
    ReduceMean module over the last axis of a (1, 2, 3) input inside
    ``nn.Sequential`` (torch.fx traces into it, so the walk sees
    ``torch.mean``), and a forward that calls ``x.sum`` over the features of
    a (1, 6) input. The walk has no rule for these today and refuses; the
    promise is a refusal or a result that holds every torch float64 output.
    Catches a rule added for them that ignores the reduced axis."""
    build, shape = _FLAT_TORCH_REDUCERS[model]
    module = build()
    outputs = replay_torch(module, samples(_FLO_POS, _FHI_POS), shape)
    what = f"{model} on {kind} through the torch model entry"
    try:
        sets = reach_model(module, flat_set(kind, _FLO_POS, _FHI_POS), "approx",
                           what)
    except NotImplementedError:
        return
    assert_covered(sets, outputs, what)


@pytest.mark.parametrize("kind,keepdims,noop", [
    pytest.param("Star", 0, 0, id="Star-keepdims0"),
    pytest.param("Zono", 0, 0, id="Zono-keepdims0"),
    pytest.param("Box", 0, 0, id="Box-keepdims0"),
    pytest.param("Star", 0, 1, id="Star-noop")])
def test_reduce_sum_without_an_axes_input_on_a_flat_set_refuses_or_is_sound(
        tmp_path, kind, keepdims, noop):
    """An opset-13 ReduceSum with no axes input (every entry, or the input
    itself with ``noop_with_empty_axes``) on a flat set standing for a
    (1, 6) tensor: it loads as onnx2torch's ``OnnxReduceSumDynamicAxes``,
    which the reach does not take today. Through the model (with
    ``input_shape``) and through ``reach_layer`` (the module alone) the call
    must refuse, or hold every onnxruntime output. Catches the module taken
    for an identity (six entries where one is due) and a handler for it that
    reads its no-op flag wrongly."""
    from onnx2torch.node_converters.reduce import OnnxReduceSumDynamicAxes
    attrs = {"keepdims": keepdims}
    if noop:
        attrs["noop_with_empty_axes"] = 1
    in_shape = [1, _FLAT_N]
    out_shape = in_shape if noop else []
    path = save_onnx(tmp_path, [helper.make_node("ReduceSum", ["X"], ["Y"],
                                                 **attrs)],
                     in_shape, out_shape, name="rdf", opset=13)
    outputs = replay_onnx(path, samples(_FLO_POS, _FHI_POS), in_shape)
    what = f"ReduceSum without axes (keepdims={keepdims}, noop={noop}) on {kind}"
    for route in ("model", "reach_layer"):
        input_set = flat_set(kind, _FLO_POS, _FHI_POS)
        try:
            if route == "model":
                sets = reach_model(load_onnx(path), input_set, "approx", what,
                                   input_shape=(_FLAT_N,))
            else:
                sets = reach_one_layer(OnnxReduceSumDynamicAxes(keepdims, noop),
                                       [input_set], "approx", what)
        except NotImplementedError:
            continue
        assert_covered(sets, outputs, f"{what} via {route}")


_SHAPE_RECORD_LOST = known_bug(
    12, "an opset-13 ReduceSum with an empty constant axes input and "
        "keepdims=0 loads as an onnx2torch module whose forward raises "
        "(LOAD-12), so the shape-recording forward of the "
        "whole model fails and no node gets a tensor shape, input_shape given "
        "or not; another Reduce of the model then takes the flat kernel, "
        "which reads axis 1 as every entry: ReduceMean over axis 1 of a set "
        "standing for a (1, 3, 4) tensor, then that ReduceSum, returns a "
        "quarter of the true output (SHAPE-8; the kernel's reading of "
        "axis 1 is AFFINE-4)")


@pytest.mark.parametrize("keepdims", [
    pytest.param(0, marks=_SHAPE_RECORD_LOST, id="keepdims0"),
    pytest.param(1, id="keepdims1")])
def test_reduce_in_a_model_whose_shape_record_fails_refuses_or_is_sound(
        tmp_path, keepdims):
    """ReduceMean over axis 1 of a flat Star standing for a (1, 3, 4) tensor,
    then an opset-13 ReduceSum whose axes input is an empty constant (every
    axis), through the model with ``input_shape``. With keepdims=0 the
    converted ReduceSum's forward raises, the shape-recording forward fails,
    and the ReduceMean has no tracked shape; with keepdims=1 the forward runs
    (the passing neighbour, same builder and reference). The call must
    refuse, or hold every onnxruntime output. Catches a Reduce that answers
    without its tensor shape when the shape record of the model is lost to
    another node."""
    nodes = [helper.make_node("ReduceMean", ["X"], ["A"], axes=[1], keepdims=0),
             helper.make_node("ReduceSum", ["A", "e"], ["Y"], keepdims=keepdims)]
    path32, path64 = _twin(tmp_path, nodes, [1, 3, 4], [1, 1] if keepdims else [],
                           {"e": np.array([], np.int64)}, "lost", 13)
    lo, hi = _MODEL_DATA["positive"]
    what = f"ReduceMean(1), ReduceSum(empty axes, keepdims={keepdims}) on Star"
    try:
        sets = reach_model(load_onnx(path32), flat_set("Star", lo, hi), "approx",
                           what, input_shape=(3, 4))
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path64, samples(lo, hi), [1, 3, 4]), what)
