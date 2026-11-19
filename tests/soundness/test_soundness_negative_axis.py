"""ONNX Slice, Concat and Split on image sets with a negative axis, and their
neighbours: non-negative axes, the batch axis, a constant Concat operand next
to an image set, opset 9 Slice, and a chunk with zero entries.

The cells
---------
A cell is one graph node (Slice, Concat or Split) on one axis, reached from an
``ImageStar`` (methods ``approx`` and ``exact``) or an ``ImageZono``
(``approx``) through ``reach_pytorch_model`` on a model loaded with
``load_onnx`` (entry ``model``) or through ``NeuralNetwork(model).reach``
(entry ``network``, a direct call of ``reach_pytorch_model``, one cell per
operation). A ReLU in front of the node makes the input of the node a set
with extra predicate variables, and for ``exact`` several sets.

What is here and what is not: the operation files hold the cells of the
non-negative axes (``test_soundness_slice.py``, ``test_soundness_concat.py``,
``test_soundness_split.py``) and of negative axes on flat sets; this file
holds negative axes on image sets, the batch axis, the constant Concat
operand next to an image set, the opset 9 Slice with a negative axis (image
and flat sets) and the chunk with zero entries. Each defect keeps one marked
cell per failing code path (set-type branch and method), with a passing
neighbour on a non-negative axis built by the same code.

The promise of every cell (``tests/README.md``, "A soundness cell"): either
the call refuses with ``NotImplementedError``, or every returned set has the
entry count of the true output and every true output is a member of some
returned set. The model is loaded before the reach, so a file the loader
cannot read is never taken for a refusal, and any other exception of the
reach (a ``ValueError`` of a set constructor or of a numpy reshape, an
``IndexError``) is a crash, not a refusal. Concat on the batch axis must
refuse. A cell the library declares supported (a non-negative axis that is
not the batch axis) must answer: a refusal there fails. For an image output
the true output is read the way the returned set says it is laid out (its
own height, width and channels, stored height by width by channel), which is
how the next layer reads it, and then the set must also say the true shape.

References: the true outputs are onnxruntime replays of 128 member points of
the input box (both extreme corners, random corners and quarter-grid points),
the box dyadic so that float32 evaluation is exact; membership is the LP of
``tests/_oracles/membership`` on the rows each returned set stores, with
``LP_TOL`` as the only slack. Where a verdict is checked, the unsafe region
is the exact interval hull of the true outputs, computed from the input box
in exact (dyadic) arithmetic without n2v, so it holds every true output and
the verdict must not be UNSAT.

Input boxes: every pixel has its own dyadic interval (distinct ends, steps of
1/8), three straddle zero, one has zero width, the others keep one sign;
tensors have more than one channel and height different from width where the
defect allows it ((1, 3, 4, 5)); (1, 4, 4, 4) is used where equal axis lengths
turn a wrong-size answer into a right-size escape.

The exact method: ``test_exact_slice_and_split_return_the_exact_image``
checks Slice and Split under ``method='exact'`` in both directions against
the exact image in rationals (``tests/_oracles/rational``), on a (2, 3, 2)
image small enough for vertex enumeration.

The defects (registry ``.claude/research/34-API-Redesign/test-audit/DEFECTS.md``
and ``sweep/verify/negative-axis.md``): GRAPH-1 (Slice, negative axis, also
over several axes), GRAPH-2 (Concat, negative axis; on axis -1 a crash where
an answer is due), GRAPH-3 (Concat on the batch axis),
GRAPH-4 (Split, negative axis), GRAPH-5 (a chunk with zero entries), GRAPH-6
(a constant Concat operand next to an ImageZono; with the image before the
constant a crash, ``AttributeError``, with its own marker), GRAPH-7 (opset 9
Slice, negative axis). Three more cases of these rows have their own
markers: GRAPH-1 (Slice on axis -4 raises ``IndexError``), GRAPH-4 (Split on
the batch axis by sizes cuts the height), GRAPH-3 (Concat on axis -4 of an
ImageZono raises numpy's ``AxisError``). Also in GRAPH-4: Split on axis -1
crashes with ``ValueError``.
"""

from fractions import Fraction

import numpy as np
import pytest
from onnx import helper

from n2v.nn.neural_network import NeuralNetwork
from n2v.sets import Box, HalfSpace, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from n2v.utils.verify_specification import verify_specification
from tests._oracles import rational
from tests._oracles.membership import LP_TOL, hwc, rows, uncovered
from tests.cell_support import (flat_set, image_set, reach_model,
                                replay_onnx, samples, save_onnx)
from tests.known_bugs import DID_NOT_RAISE, known_bug

INT_MIN = -2 ** 63

# --------------------------------------------------------------------------- #
# Markers: one per defect and failure type                                      #
# --------------------------------------------------------------------------- #
GRAPH_1 = known_bug(
    7, "GRAPH-1: ONNX Slice (opset 10 and later) with a negative axis on an "
       "ImageStar or ImageZono slices the wrong stored axis (-3 the height, -2 "
       "the width, -1 the channels): the result has the wrong entry count or "
       "leaves out every true output")
GRAPH_1_AXES = known_bug(
    7, "GRAPH-1 (several axes): ONNX Slice over the axes [3, -2] of an image "
       "set keeps the two slices under the stored keys 1 and -3, which "
       "_slice_set both reads as the width: the height slice overwrites the "
       "width slice and the height is not cut, so the result has the wrong "
       "entry count")
SLICE_BATCH_CRASH = known_bug(
    7, "GRAPH-1: "
       "ONNX Slice on axis -4, the batch axis, of an image set "
       "crashes with IndexError in _slice_set (stored axis -5) instead of "
       "refusing or answering", raises=IndexError)
GRAPH_2 = known_bug(
    7, "GRAPH-2: ONNX Concat of image sets with a negative axis joins along "
       "the wrong stored axis (-3 the width, -2 the channels; for an ImageZono "
       "centre and generators along two different axes): every true output "
       "is outside the result")
CONCAT_AXIS_M1_CRASH = known_bug(
    7, "GRAPH-2 (axis -1, not a refusal by an accident "
       "of sizes): ONNX Concat on axis -1 of an ImageStar joins the bases "
       "along their predicate-column axis, and of an ImageZono the centres on "
       "the channels and the generators on the generator axis; the set "
       "constructor raises ValueError where an answer is due",
    raises=ValueError)
GRAPH_3 = known_bug(
    7, "GRAPH-3: ONNX Concat on the batch axis (0, or -4 on an ImageStar) of "
       "an image set is not refused: one image of doubled height stands for "
       "an output of batch 2", raises=DID_NOT_RAISE)
CONCAT_BATCH_CRASH = known_bug(
    7, "GRAPH-3: ONNX Concat on "
       "axis -4, the batch axis, of an ImageZono crashes with numpy's "
       "AxisError (the raw axis is handed to np.concatenate on the 3-d "
       "centre) instead of refusing with NotImplementedError",
    raises=np.exceptions.AxisError)
GRAPH_4 = known_bug(
    7, "GRAPH-4: ONNX Split of an image set with a negative axis cuts the "
       "wrong stored axis (-3 the width, -2 the channels) and an even split "
       "reads the chunk length from yet another axis: wrong entry count or "
       "every true output outside the chunk")
SPLIT_AXIS_M1_CRASH = known_bug(
    7, "GRAPH-4: ONNX Split on axis -1 of an ImageStar cuts "
       "the predicate columns of the basis, and of an ImageZono the "
       "generators; the ImageStar constructor or a numpy reshape raises "
       "ValueError where an answer is due", raises=ValueError)
SPLIT_BATCH = known_bug(
    7, "GRAPH-4: "
       "ONNX Split on the batch axis (0 or -4) of an image set with "
       "listed sizes is not refused: it cuts the height instead and returns "
       "a chunk of the wrong size. A Split whose axis attribute is left out "
       "reaches the same path (onnx2torch reads it as axis 0)")
GRAPH_5 = known_bug(
    7, "GRAPH-5: an even Split with axis -2 of an ImageStar of tensor "
       "(1, 2, 4, 6) gives a second chunk with zero channels (length 3 read "
       "from the width, cut on the channels) and the next ReLU returns no sets "
       "for a non-empty input")
GRAPH_6 = known_bug(
    7, "GRAPH-6: ONNX Concat of a constant operand first and an ImageZono "
       "returns a flat Zono whose entries leave out every true output "
       "(_const_set_like takes an ImageZono for a Zono)")
GRAPH_6_CRASH = known_bug(
    7, "GRAPH-6: ONNX Concat of an ImageZono followed by a constant operand "
       "crashes with AttributeError ('Zono' object has no attribute "
       "'height') instead of refusing or answering", raises=AttributeError)
GRAPH_7 = known_bug(
    7, "GRAPH-7: ONNX Slice of opset 9 with a negative axis is reached as the "
       "identity (only _pos_axes_slices is read): the result has the wrong "
       "entry count, and after a ReduceSum a one-entry set that leaves out "
       "every true output")

# --------------------------------------------------------------------------- #
# Inputs                                                                        #
# --------------------------------------------------------------------------- #
KINDS = {"IS-approx": ("ImageStar", "approx"),
         "IS-exact": ("ImageStar", "exact"),
         "IZ-approx": ("ImageZono", "approx")}


def _image_box(channels, height, width, seed=0, n_cross=3):
    """Dyadic (C, H, W) bounds ``(lo, hi)`` in steps of 1/8: every pixel its
    own interval, ``n_cross`` of them straddling zero, one of zero width (a
    positive value), about a third of the others negative and the rest
    positive. A result that reads the pixels in the wrong order or along the
    wrong axis cannot hold the true outputs."""
    rng = np.random.default_rng(seed)
    n = channels * height * width
    key = rng.permutation(n) + 1.0                     # distinct, 1 .. n
    span = rng.integers(1, 5, size=n) / 8.0
    negative = rng.random(n) < 0.3
    lo = np.where(negative, -key / 8.0 - span, key / 8.0)
    hi = lo + span
    picked = rng.choice(n, size=n_cross + 1, replace=False)
    cross, flat = picked[:n_cross], picked[n_cross]
    for rank, i in enumerate(cross):
        lo[i], hi[i] = -(rank + 1) / 8.0, (n_cross - rank) / 8.0
    lo[flat] = hi[flat] = key[flat] / 8.0
    return lo.reshape(channels, height, width), hi.reshape(channels, height,
                                                           width)


def _relu_box(lo, hi):
    return np.maximum(lo, 0.0), np.maximum(hi, 0.0)


# --------------------------------------------------------------------------- #
# Reach                                                                         #
# --------------------------------------------------------------------------- #
def _reach(model, input_set, method, entry, what, **options):
    """Reach sets of the loaded ``model``: entry ``model`` is
    ``reach_pytorch_model`` (through ``cell_support.reach_model``, which fails
    on an empty result and on a changed input set), entry ``network`` is
    ``NeuralNetwork(model).reach``. Nothing is caught."""
    if entry == "model":
        return reach_model(model, input_set, method, what, **options)
    sets = NeuralNetwork(model).reach(input_set, method=method, **options)
    if sets is None or len(sets) == 0:
        raise AssertionError(f"{what}: NeuralNetwork.reach returned no sets "
                             f"for a non-empty input set")
    return list(sets)


def _answer(path, input_set, method, entry, what, may_refuse, **options):
    """The reach result of the model at ``path``, or None when the reach
    refuses with ``NotImplementedError`` and ``may_refuse``. The model is
    loaded first and outside the refusal, so a load failure fails the test;
    any other exception of the reach is a crash and fails it too."""
    model = load_onnx(path)
    if not may_refuse:
        return _reach(model, input_set, method, entry, what, **options)
    try:
        return _reach(model, input_set, method, entry, what, **options)
    except NotImplementedError:
        return None


# --------------------------------------------------------------------------- #
# The coverage check                                                            #
# --------------------------------------------------------------------------- #
def _is_image(s):
    return isinstance(s, (ImageStar, ImageZono))


def _entries(s):
    """The number of entries a returned set stores, read from its own arrays
    (``membership.rows`` cannot read a set with zero entries, which a wrong
    chunk length produces)."""
    if isinstance(s, ImageStar):
        return int(np.prod(np.shape(s.V)[:3]))
    if isinstance(s, Zono):                       # ImageZono included
        return int(np.asarray(s.c).size)
    if isinstance(s, Star):
        return int(np.shape(s.V)[0])
    if isinstance(s, Box):
        return int(np.asarray(s.lb).size)
    raise TypeError(f"no entry count for a {type(s).__name__}")


def _describe(s):
    if _is_image(s):
        return (f"{type(s).__name__} stored {s.height}x{s.width}x"
                f"{s.num_channels} (height x width x channels)")
    return f"{type(s).__name__} of {_entries(s)} entries"


def _outer_box(s):
    """The box of ``c + G a`` over the predicate box ``lb <= a <= ub`` alone,
    read from the set's stored rows: it contains the set, so a point outside
    it by more than the slack is not a member and needs no LP."""
    c, G, _, _, lb, ub = rows(s)
    with np.errstate(invalid="ignore"):
        a, b = G * lb[None, :], G * ub[None, :]
    a, b = np.where(G == 0, 0.0, a), np.where(G == 0, 0.0, b)
    low = c + np.minimum(a, b).sum(axis=1)
    high = c + np.maximum(a, b).sum(axis=1)
    scale = np.abs(c) + np.abs(a).sum(axis=1) + np.abs(b).sum(axis=1)
    low = np.where(np.isnan(low), -np.inf, low)
    high = np.where(np.isnan(high), np.inf, high)
    return low, high, scale


def _assert_holds(sets, outputs, out_shape, what):
    """Every set has the true entry count, every true output (network order,
    flat) is a member of some set, and an image set says the true shape.

    An image output (``out_shape`` of rank 4, batch 1) is read the way each
    returned image set lays it out: converted with ``membership.hwc`` using
    the set's own channels, height and width (the reading of the next
    Flatten or affine layer); a flat set is compared in the tensor's own
    order. For a flat output an image set must be 1x1. Membership is
    ``membership.uncovered`` (slack ``LP_TOL``) after a first pass that drops
    the sets whose outer box (``_outer_box``) misses the point by more than
    that slack plus a float64 rounding pad."""
    sets = list(sets)
    outputs = [np.asarray(y, dtype=np.float64).reshape(-1) for y in outputs]
    if not sets:
        raise AssertionError(f"{what}: the result is an empty list of sets")
    size = int(np.prod(out_shape))
    image_out = len(out_shape) == 4 and out_shape[0] == 1
    for s in sets:
        n = _entries(s)
        if n != size:
            raise AssertionError(
                f"{what}: wrong entry count: the result holds a "
                f"{_describe(s)} ({n} entries), the true output "
                f"{tuple(out_shape)} has {size}")
        if _is_image(s) and not image_out and s.height * s.width > 1:
            raise AssertionError(
                f"{what}: the result holds a {_describe(s)} for a flat output "
                f"of shape {tuple(out_shape)}")
    best = {i: float("inf") for i in range(len(outputs))}
    for s in sets:
        if image_out and _is_image(s):
            view = [hwc(y, s.num_channels, s.height, s.width)
                    for y in outputs]
        else:
            view = outputs
        low, high, scale = _outer_box(s)
        candidates = []
        for i in sorted(best):
            y = view[i]
            slack = LP_TOL * max(1.0, float(np.max(np.abs(y), initial=0.0)))
            with np.errstate(invalid="ignore"):
                excess = np.maximum(low - y, y - high) - 1e-12 * scale
            excess = np.where(np.isnan(excess), -np.inf, excess)
            gap = float(max(0.0, np.max(excess, initial=0.0)))
            if gap > slack:
                best[i] = min(best[i], gap)
            else:
                candidates.append(i)
        if candidates:
            missed = dict(uncovered([s], [view[i] for i in candidates]))
            for position, i in enumerate(candidates):
                if position in missed:
                    best[i] = min(best[i], missed[position])
                else:
                    del best[i]
        if not best:
            break
    if best:
        names = [_describe(s) for s in sets]
        summary = "; ".join(f"{names.count(n)} x {n}"
                            for n in sorted(set(names)))
        raise AssertionError(
            f"{what}: {len(best)} of {len(outputs)} true outputs are in none "
            f"of the {len(sets)} returned sets (largest distance at least "
            f"{max(best.values()):.3g}; first missed output {min(best)}); "
            f"result: {summary}")
    if image_out:
        channels, height, width = (int(v) for v in out_shape[1:])
        for s in sets:
            if _is_image(s) and (s.height, s.width, s.num_channels) != (
                    height, width, channels):
                raise AssertionError(
                    f"{what}: the result holds a {_describe(s)}, the output "
                    f"is a (C, H, W) = {(channels, height, width)} tensor")


# --------------------------------------------------------------------------- #
# One cell                                                                      #
# --------------------------------------------------------------------------- #
def _replayed(tmp_path, nodes, in_shape, out_shape, inits, opset, lo, hi):
    """Save the graph, replay 128 member points of [lo, hi] on onnxruntime and
    check (as setup, not absorbed by a marker) that each output has the
    entry count of ``out_shape``, which is computed by the caller from the
    node's definition."""
    path = save_onnx(tmp_path, nodes, in_shape, out_shape, inits=inits,
                     opset=opset)
    points = samples(lo, hi)
    outputs = replay_onnx(path, points, in_shape)
    size = int(np.prod(out_shape))
    if len(outputs) != len(points) or any(y.size != size for y in outputs):
        pytest.fail(f"setup: onnxruntime outputs do not have the {size} "
                    f"entries of {tuple(out_shape)}")
    return path, outputs


def _image_cell(tmp_path, nodes, in_shape, out_shape, inits, opset, kind,
                entry, may_refuse, what):
    """Reach the model from the image box of ``in_shape`` and check the
    promise; returns the result (None for a refusal)."""
    set_kind, method = KINDS[kind]
    lo, hi = _image_box(*in_shape[1:])
    path, outputs = _replayed(tmp_path, nodes, in_shape, out_shape, inits,
                              opset, lo, hi)
    sets = _answer(path, image_set(set_kind, lo, hi), method, entry, what,
                   may_refuse)
    if sets is not None:
        _assert_holds(sets, outputs, out_shape, what)
    return sets


def _box_region(L, U):
    """The unsafe region ``L <= y <= U`` as one half-space system (rows ANDed),
    ``y`` in the network's order."""
    L, U = np.asarray(L, float).reshape(-1), np.asarray(U, float).reshape(-1)
    n = L.size
    return HalfSpace(np.vstack([np.eye(n), -np.eye(n)]),
                     np.concatenate([U, -L]).reshape(-1, 1))


# --------------------------------------------------------------------------- #
# Graphs                                                                        #
# --------------------------------------------------------------------------- #
def _slice(start, end, step):
    """Python slice of ONNX ``starts``/``ends``/``steps`` as the cells use
    them (``end`` INT_MIN with a negative step runs to the first entry)."""
    return slice(None if (step < 0 and start == -1) else start,
                 None if end == INT_MIN else end, step)


def _sliced_shape(shape, axis, start, end, step):
    idx = [slice(None)] * len(shape)
    idx[axis] = _slice(start, end, step)
    return list(np.zeros(shape)[tuple(idx)].shape)


def _slice_nodes(axis, start, end, step, opset, follow=None):
    """Relu -> Slice (-> Flatten -> Gemm when ``follow`` holds the weights)."""
    out = "S" if follow is not None else "Y"
    nodes = [helper.make_node("Relu", ["X"], ["R"])]
    if opset < 10:
        nodes.append(helper.make_node("Slice", ["R"], [out], starts=[start],
                                      ends=[end], axes=[axis]))
        inits = {}
    else:
        inits = {"st": np.array([start], np.int64),
                 "en": np.array([end], np.int64),
                 "ax": np.array([axis], np.int64),
                 "sp": np.array([step], np.int64)}
        nodes.append(helper.make_node("Slice", ["R", "st", "en", "ax", "sp"],
                                      [out]))
    if follow is not None:
        nodes += _flatten_gemm("S", follow, inits)
    return nodes, inits


def _gemm_weights(n_in, seed=1):
    """Dyadic Gemm weights (3 outputs) and bias, float32."""
    rng = np.random.default_rng(seed)
    return ((rng.integers(-4, 5, size=(3, n_in)) / 4.0).astype(np.float32),
            (rng.integers(-4, 5, size=3) / 4.0).astype(np.float32))


def _flatten_gemm(source, weights, inits):
    W, b = weights
    inits["W"], inits["b"] = W, b
    return [helper.make_node("Flatten", [source], ["F"]),
            helper.make_node("Gemm", ["F", "W", "b"], ["Y"], transB=1)]


def _concat_nodes(axis, operands):
    """``Concat`` of ``operands`` over A = Relu(X) and B = Neg(X)."""
    nodes = [helper.make_node("Relu", ["X"], ["A"]),
             helper.make_node("Neg", ["X"], ["B"]),
             helper.make_node("Concat", list(operands), ["Y"], axis=axis)]
    return nodes, {}


def _split_sizes(shape, axis, sizes, n_out):
    return list(sizes) if sizes is not None else [shape[axis] // n_out] * n_out


def _split_nodes(axis, sizes, n_out, chunk, opset=13, follow=None):
    """Relu -> Split -> Relu on output ``chunk`` (-> Flatten -> Gemm)."""
    outs = [f"P{i}" for i in range(n_out)]
    inits = {}
    if sizes is None:
        split = helper.make_node("Split", ["R"], outs, axis=axis)
    elif opset < 13:
        split = helper.make_node("Split", ["R"], outs, axis=axis,
                                 split=list(sizes))
    else:
        inits["sp"] = np.array(sizes, np.int64)
        split = helper.make_node("Split", ["R", "sp"], outs, axis=axis)
    out = "Z" if follow is not None else "Y"
    nodes = [helper.make_node("Relu", ["X"], ["R"]), split,
             helper.make_node("Relu", [outs[chunk]], [out])]
    if follow is not None:
        nodes += _flatten_gemm("Z", follow, inits)
    return nodes, inits


def _split_part(array, axis, sizes, chunk):
    return np.split(array, np.cumsum(sizes)[:-1], axis=axis)[chunk]


# --------------------------------------------------------------------------- #
# A1: Slice, opset 10 and later                                                 #
# --------------------------------------------------------------------------- #
# (name, tensor, starts, ends, steps); the reversing slice keeps every length,
# so a slice on the wrong axis is a right-size escape on (1, 3, 4, 5) too;
# [1:3] on (1, 4, 4, 4) is a right-size escape between two axes of length 4.
_SLICES = {"4x4x4[1:3]": ((1, 4, 4, 4), 1, 3, 1),
           "3x4x5[::-1]": ((1, 3, 4, 5), -1, INT_MIN, -1),
           "3x4x5[0:1]": ((1, 3, 4, 5), 0, 1, 1)}


def _slice_params():
    out = []

    def add(family, axis, kind, entry="model", opset=13, compose=False,
            marks=()):
        name = (f"{family}-axis{axis}-{kind}-{entry}-opset{opset}"
                + ("-gemm" if compose else ""))
        out.append(pytest.param(family, axis, kind, entry, opset, compose,
                                marks=marks, id=name))

    # GRAPH-1: one marked cell per set-type branch and method, the three
    # negative axes spread over them
    add("3x4x5[::-1]", -3, "IS-approx", marks=GRAPH_1)
    add("3x4x5[::-1]", -2, "IZ-approx", marks=GRAPH_1)
    add("4x4x4[1:3]", -1, "IS-exact", marks=GRAPH_1)
    # passing neighbours on the non-negative axes, same builder and check
    add("3x4x5[::-1]", 1, "IS-approx")
    add("3x4x5[::-1]", 2, "IZ-approx")
    add("3x4x5[::-1]", 2, "IS-exact")
    add("3x4x5[::-1]", 2, "IS-approx", "network")
    # composition: the sliced ImageZono rows consumed by Flatten and Gemm
    add("4x4x4[1:3]", 3, "IZ-approx", compose=True)
    # the batch axis: [0:1] of a batch of one is the identity; axis 0 is
    # refused, axis -4 crashes (one cell per set-type branch)
    for kind in ("IS-approx", "IZ-approx"):
        add("3x4x5[0:1]", 0, kind)
        add("3x4x5[0:1]", -4, kind, marks=SLICE_BATCH_CRASH)
    return out


@pytest.mark.parametrize("family, axis, kind, entry, opset, compose",
                         _slice_params())
def test_slice_on_an_image_set_holds_every_model_output(
        tmp_path, family, axis, kind, entry, opset, compose):
    """``Relu -> Slice`` (and ``-> Flatten -> Gemm``) on an image set, one
    axis. Catches a Slice that applies an ONNX axis to the wrong stored axis
    (a negative axis not reduced before the channel/height/width remap, the
    remap itself swapped), a reversed slice whose ImageZono generators keep
    their forward order, and on the batch axis an answer, or a crash of any
    type but ``NotImplementedError``, where the refusal is due."""
    shape, start, end, step = _SLICES[family]
    out_shape = _sliced_shape(shape, axis, start, end, step)
    follow = None
    if compose:
        follow = _gemm_weights(int(np.prod(out_shape)))
        out_shape = [1, 3]
    nodes, inits = _slice_nodes(axis, start, end, step, opset, follow)
    may_refuse = axis < 0 or axis == 0
    _image_cell(tmp_path, nodes, shape, out_shape, inits, opset, kind, entry,
                may_refuse, f"Slice {family} axis {axis} {kind} {entry}")


# Several axes in one node: (axes, starts, ends) on (1, 3, 4, 5); every
# selection is the crop of rows 1:3 and columns 1:4.
_AXES_SLICES = {"[2,3]": ([2, 3], [1, 1], [3, 4]),
                "[-2,-1]": ([-2, -1], [1, 1], [3, 4]),
                "[3,-2]": ([3, -2], [1, 1], [4, 3])}


@pytest.mark.parametrize("axes, kind", [
    pytest.param("[-2,-1]", "IS-approx", marks=GRAPH_1,
                 id="axes[-2,-1]-IS-approx"),
    pytest.param("[3,-2]", "IS-approx", marks=GRAPH_1_AXES,
                 id="axes[3,-2]-IS-approx"),
    pytest.param("[3,-2]", "IZ-approx", marks=GRAPH_1_AXES,
                 id="axes[3,-2]-IZ-approx"),
    pytest.param("[2,3]", "IS-approx", id="axes[2,3]-IS-approx"),
    pytest.param("[2,3]", "IZ-approx", id="axes[2,3]-IZ-approx")])
def test_slice_over_several_axes_of_an_image_set_holds_every_model_output(
        tmp_path, axes, kind):
    """``Relu -> Slice`` over two axes at once (the spatial crop rows 1:3,
    columns 1:4 of a (1, 3, 4, 5) image), written as the axes [2, 3], [-2, -1]
    and [3, -2]. Catches a Slice that keeps only one of the listed axes (the
    dictionary of slices rebuilt per axis, or two listed axes landing on one
    stored key so that one overwrites the other) and a negative axis in a
    list of axes applied to the wrong stored axis."""
    shape = (1, 3, 4, 5)
    ax, starts, ends = _AXES_SLICES[axes]
    idx = [slice(None)] * 4
    for a, s, e in zip(ax, starts, ends):
        idx[a] = slice(s, e)
    out_shape = list(np.zeros(shape)[tuple(idx)].shape)
    inits = {"st": np.array(starts, np.int64),
             "en": np.array(ends, np.int64), "ax": np.array(ax, np.int64)}
    nodes = [helper.make_node("Relu", ["X"], ["R"]),
             helper.make_node("Slice", ["R", "st", "en", "ax"], ["Y"])]
    _image_cell(tmp_path, nodes, shape, out_shape, inits, 13, kind, "model",
                min(ax) < 0, f"Slice axes {axes} {kind}")


def _slice_hull(axis):
    def hull(lo, hi):
        rl, rh = _relu_box(lo, hi)
        idx = [slice(None)] * 3
        idx[axis if axis < 0 else axis - 1] = slice(1, 3)
        return rl[tuple(idx)], rh[tuple(idx)]
    return hull


@pytest.mark.parametrize("axis, kind, entry",
                         [pytest.param(1, "IS-approx", "model",
                                       id="axis1-IS-approx-model")])
def test_slice_negative_axis_verdict_is_not_unsat(tmp_path, axis, kind,
                                                  entry):
    """End to end: ``Relu -> Slice[1:3]`` on the channels of tensor
    (1, 4, 4, 4), an ImageStar, reach, then ``verify_specification`` against
    the exact interval hull of the true outputs (the reach result holds every
    true output, so the verdict must not be UNSAT). Catches a
    ``verify_specification`` that compares an ImageStar with the unsafe
    region in the set's stored (height, width, channel) order instead of the
    network's order, a false UNSAT on a sound reach set (plant
    ``verify-imagestar-read-in-stored-order``; no other test of the suite
    fails under it, the ImageZono reading is held by
    ``tests/unit/utils/test_verify_specification_regressions.py``). A Slice
    on the wrong stored axis is caught by the coverage cells of the same
    model, which see every lost output the verdict can see."""
    set_kind, method = KINDS[kind]
    shape = (1, 4, 4, 4)
    nodes, inits = _slice_nodes(axis, 1, 3, 1, 13)
    out_shape = _sliced_shape(shape, axis, 1, 3, 1)
    lo, hi = _image_box(*shape[1:])
    L, U = _slice_hull(axis)(lo, hi)
    path, outputs = _replayed(tmp_path, nodes, shape, out_shape, inits, 13,
                              lo, hi)
    Lf, Uf = L.reshape(-1), U.reshape(-1)
    if not all(np.all(y >= Lf) and np.all(y <= Uf) for y in outputs):
        pytest.fail("setup: a true output is outside the unsafe region")
    what = f"Slice[1:3] axis {axis} {kind} {entry}"
    sets = _answer(path, image_set(set_kind, lo, hi), method, entry, what,
                   may_refuse=False)
    verdict = verify_specification(sets, _box_region(L, U)).verdict
    assert verdict != "UNSAT", (
        f"{what}: verify_specification answered UNSAT although all "
        f"{len(outputs)} true outputs are in the unsafe region (result: "
        f"{len(sets)} x {_describe(sets[0])})")


# --------------------------------------------------------------------------- #
# A2: Concat                                                                    #
# --------------------------------------------------------------------------- #
_CONCATS = {"3x4x5-AB": ((1, 3, 4, 5), "AB")}


def _concat_params():
    out = []

    def add(family, axis, kind, entry="model", opset=13, compose=False,
            marks=()):
        name = (f"{family}-axis{axis}-{kind}-{entry}-opset{opset}"
                + ("-gemm" if compose else ""))
        out.append(pytest.param(family, axis, kind, entry, opset, compose,
                                marks=marks, id=name))

    # GRAPH-2: one marked cell per set-type branch and method (-3 and -2 on
    # the ImageStar approx branch, whose two stored axes differ)
    add("3x4x5-AB", -3, "IS-approx", marks=GRAPH_2)
    add("3x4x5-AB", -2, "IS-approx", marks=GRAPH_2)
    add("3x4x5-AB", -3, "IS-exact", marks=GRAPH_2)
    add("3x4x5-AB", -3, "IZ-approx", marks=GRAPH_2)
    # axis -1: a crash in the set constructor of each branch
    add("3x4x5-AB", -1, "IS-approx", marks=CONCAT_AXIS_M1_CRASH)
    add("3x4x5-AB", -1, "IZ-approx", marks=CONCAT_AXIS_M1_CRASH)
    # passing neighbours: the same concatenations on the non-negative axes
    for axis in (1, 2, 3):
        add("3x4x5-AB", axis, "IS-approx")
    for axis in (1, 3):
        add("3x4x5-AB", axis, "IZ-approx")
    add("3x4x5-AB", 1, "IS-exact")
    add("3x4x5-AB", 3, "IS-approx", "network")
    return out


@pytest.mark.parametrize("family, axis, kind, entry, opset, compose",
                         _concat_params())
def test_concat_on_image_sets_holds_every_model_output(
        tmp_path, family, axis, kind, entry, opset, compose):
    """``Concat(Relu(X), Neg(X))`` on an image set, one axis. Catches a
    Concat that joins along the wrong stored axis (a negative axis passed
    through the channel/height/width remap unchanged, a swapped remap, axis
    -1 joined on another stored axis), an ImageZono whose centre and
    generators are joined along different axes, operands stacked in the
    wrong order, a dropped operand, and an exact Relu list of which only one
    piece is joined."""
    shape, operands = _CONCATS[family]
    out_shape = list(shape)
    out_shape[axis] = shape[axis] * len(operands)
    nodes, inits = _concat_nodes(axis, operands)
    _image_cell(tmp_path, nodes, shape, out_shape, inits, opset, kind, entry,
                axis < 0, f"Concat {family} axis {axis} {kind} {entry}")


def _batch_concat_params():
    out = []
    for axis, kind, marks in ((0, "IS-approx", GRAPH_3),
                              (0, "IS-exact", GRAPH_3),
                              (0, "IZ-approx", GRAPH_3),
                              (-4, "IS-approx", GRAPH_3),
                              (-4, "IZ-approx", CONCAT_BATCH_CRASH)):
        out.append(pytest.param(axis, kind, "model", marks=marks,
                                id=f"axis{axis}-{kind}-model"))
    return out


@pytest.mark.parametrize("axis, kind, entry", _batch_concat_params())
def test_concat_on_the_batch_axis_of_an_image_set_is_refused(tmp_path, axis,
                                                             kind, entry):
    """``Concat(Relu(X), Neg(X))`` on axis 0 or -4 of tensor (1, 3, 4, 5): the
    true output has a batch of two, which one image set cannot stand for, so
    the call must refuse with ``NotImplementedError`` (the refusal alone).
    Catches a Concat that joins the batch items along the height (or any
    other stored axis) and answers, and a crash of another type where the
    refusal is due. The passing neighbours are the axis-0 cells of
    ``test_slice_on_an_image_set_holds_every_model_output``
    (``3x4x5[0:1]-axis0-IS-approx-model-opset13``, ``...-IZ-approx-...``),
    refusals on the same input through the same ``_image_box``,
    ``image_set`` and ``_reach``."""
    set_kind, method = KINDS[kind]
    shape = (1, 3, 4, 5)
    nodes, _ = _concat_nodes(axis, "AB")
    lo, hi = _image_box(3, 4, 5)
    path = save_onnx(tmp_path, nodes, shape, [2, 3, 4, 5])
    model = load_onnx(path)
    input_set = image_set(set_kind, lo, hi)
    with pytest.raises(NotImplementedError):
        _reach(model, input_set, method, entry, f"Concat batch axis {axis}")


# --------------------------------------------------------------------------- #
# A4: a constant Concat operand next to an image set                            #
# --------------------------------------------------------------------------- #
def _const_concat(axis, order, shape):
    """``Concat`` over K (a constant of the operand shape with length 1 on the
    axis, dyadic), A = Relu(X), B = Neg(X), in ``order``; its output shape
    and the exact constant."""
    k_shape = list(shape)
    k_shape[axis] = 1
    K = (np.arange(int(np.prod(k_shape))).reshape(k_shape) / 4.0
         - 1.0).astype(np.float32)
    nodes = [helper.make_node("Relu", ["X"], ["A"]),
             helper.make_node("Neg", ["X"], ["B"]),
             helper.make_node("Concat", list(order), ["Y"], axis=axis)]
    out_shape = list(shape)
    out_shape[axis] = sum(1 if o == "K" else shape[axis] for o in order)
    return nodes, {"K": K}, out_shape


@pytest.mark.parametrize("order, axis, kind", [
    pytest.param("KA", 1, "ImageZono", marks=GRAPH_6, id="KA-axis1-ImageZono"),
    pytest.param("AK", 1, "ImageZono", marks=GRAPH_6_CRASH,
                 id="AK-axis1-ImageZono"),
    pytest.param("KA", 3, "ImageStar", id="KA-axis3-ImageStar"),
    pytest.param("AK", 2, "ImageStar", id="AK-axis2-ImageStar"),
    pytest.param("KA", 1, "Zono", id="KA-axis1-Zono-input_shape"),
    pytest.param("AK", 1, "Zono", id="AK-axis1-Zono-input_shape")])
def test_concat_with_a_constant_operand_holds_every_model_output(
        tmp_path, order, axis, kind):
    """``Concat`` of a constant and ``Relu(X)`` (constant first or last) on
    tensor (1, 2, 3, 2): an ImageZono and an ImageStar (``approx``), and the
    same model from a flat Zono given ``input_shape`` (supported today, so it
    must answer; the neighbour of the ImageZono cells). The ImageStar is
    refused today (``_const_set_like``); its cells put the constant along the
    width or the height, where it has two channels, so that a constant laid
    out in the tensor's (C, H, W) order next to image rows in stored
    (H, W, C) order cannot hold the true outputs (along the channels the
    constant has one channel and both orders coincide). Catches that layout
    confusion, a constant placed at the wrong offset, an ImageZono taken for
    a flat Zono, and a crash where the image set should be refused. The flat
    Star cells and the other axes are the concat file's
    (``test_concat_with_a_constant_operand_holds_every_output``)."""
    shape = (1, 2, 3, 2)
    nodes, inits, out_shape = _const_concat(axis, order, shape)
    lo, hi = _image_box(2, 3, 2)
    path, outputs = _replayed(tmp_path, nodes, shape, out_shape, inits, 13,
                              lo, hi)
    what = f"Concat {order} axis {axis} {kind}"
    if kind.startswith("Image"):
        sets = _answer(path, image_set(kind, lo, hi), "approx", "model", what,
                       may_refuse=True)
    else:
        sets = _reach(load_onnx(path), flat_set(kind, lo, hi), "approx",
                      "model", what, input_shape=shape[1:])
    if sets is not None:
        _assert_holds(sets, outputs, out_shape, what)


# --------------------------------------------------------------------------- #
# A3: Split                                                                     #
# --------------------------------------------------------------------------- #
# (tensor, sizes or None for an even split, number of outputs)
_SPLITS = {"4x4x4-[1,3]": ((1, 4, 4, 4), (1, 3), 2),
           "4x4x4-[1,1,2]": ((1, 4, 4, 4), (1, 1, 2), 3),
           "2x4x6-even": ((1, 2, 4, 6), None, 2),
           "3x4x5-[L,0]": ((1, 3, 4, 5), "L0", 2)}


def _split_params():
    out = []

    def add(family, axis, chunk, kind, entry="model", opset=13,
            compose=False, marks=()):
        name = (f"{family}-axis{axis}-chunk{chunk}-{kind}-{entry}-opset"
                f"{opset}" + ("-gemm" if compose else ""))
        out.append(pytest.param(family, axis, chunk, kind, entry, opset,
                                compose, marks=marks, id=name))

    # GRAPH-4, listed sizes: one marked cell per set-type branch and method
    for kind in KINDS:
        add("4x4x4-[1,3]", -3, 1, kind, marks=GRAPH_4)
    # GRAPH-4, even split: the chunk length read from another axis
    add("2x4x6-even", -3, 0, "IS-approx", marks=GRAPH_4)
    add("2x4x6-even", -3, 1, "IZ-approx", marks=GRAPH_4)
    # axis -1: a crash in each set-type branch
    add("4x4x4-[1,3]", -1, 0, "IS-approx", marks=SPLIT_AXIS_M1_CRASH)
    add("4x4x4-[1,3]", -1, 1, "IZ-approx", marks=SPLIT_AXIS_M1_CRASH)
    # passing neighbours: the same splits on the non-negative axes
    add("4x4x4-[1,3]", 1, 1, "IS-approx")
    add("4x4x4-[1,3]", 3, 1, "IS-approx")
    add("4x4x4-[1,1,2]", 1, 2, "IZ-approx")
    add("4x4x4-[1,1,2]", 2, 2, "IS-exact")
    add("2x4x6-even", 1, 1, "IS-approx")
    add("2x4x6-even", 1, 1, "IZ-approx")
    add("4x4x4-[1,3]", 3, 1, "IS-approx", "network")
    add("4x4x4-[1,3]", 1, 1, "IS-approx", compose=True)
    # the batch axis: sizes [1, 0] of a batch of one keep the whole tensor
    for kind in KINDS:
        add("3x4x5-[L,0]", 0, 0, kind, marks=SPLIT_BATCH)
    add("3x4x5-[L,0]", -4, 0, "IS-approx", marks=SPLIT_BATCH)
    add("3x4x5-[L,0]", 1, 0, "IS-approx")
    add("3x4x5-[L,0]", 1, 0, "IZ-approx")
    return out


@pytest.mark.parametrize("family, axis, chunk, kind, entry, opset, compose",
                         _split_params())
def test_split_of_an_image_set_holds_every_model_output(
        tmp_path, family, axis, chunk, kind, entry, opset, compose):
    """``Relu -> Split -> Relu`` on one chunk in turn (and ``-> Flatten ->
    Gemm``) on an image set. Catches a Split that cuts the wrong stored axis
    (a negative axis not reduced before the remap, a swapped remap, axis -1
    cut on another stored axis), an even split whose chunk length is read
    from another axis, chunks returned in the wrong order or at the wrong
    offset (an ImageZono offset never advanced), and the batch axis cut as
    the height."""
    shape, sizes, n_out = _SPLITS[family]
    if sizes == "L0":
        sizes = (shape[axis], 0)
    sizes = _split_sizes(shape, axis, sizes, n_out)
    out_shape = list(_split_part(np.zeros(shape), axis, sizes, chunk).shape)
    follow = None
    if compose:
        follow = _gemm_weights(int(np.prod(out_shape)))
        out_shape = [1, 3]
    given = None if _SPLITS[family][1] is None else sizes
    nodes, inits = _split_nodes(axis, given, n_out, chunk, opset, follow)
    _image_cell(tmp_path, nodes, shape, out_shape, inits, opset, kind, entry,
                axis <= 0,
                f"Split {family} axis {axis} chunk {chunk} {kind} {entry}")


# --------------------------------------------------------------------------- #
# N1: a chunk with zero entries                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("axis, kind, entry", [
    pytest.param(-2, "IS-approx", "model", marks=GRAPH_5,
                 id="axis-2-IS-approx-model"),
    pytest.param(-2, "IS-exact", "model", marks=GRAPH_5,
                 id="axis-2-IS-exact-model"),
    pytest.param(2, "IS-approx", "model", id="axis2-IS-approx-model"),
    pytest.param(2, "IS-exact", "model", id="axis2-IS-exact-model")])
def test_even_split_second_chunk_is_not_empty(tmp_path, axis, kind, entry):
    """``Relu -> Split(even) -> Relu`` on the second chunk of an ImageStar of
    tensor (1, 2, 4, 6), axis -2 (the height) and its neighbour axis 2: the
    reach result is a refusal or a non-empty list that holds every true
    output. Catches an even split whose chunk length is read from one axis
    and applied to another, or taken as the whole axis, which leaves a chunk
    with zero entries and a reach result with no sets (which
    ``verify_specification`` reads as UNSAT: EMPTY-1, marked in
    ``tests/contracts/test_verify_api.py``)."""
    shape = (1, 2, 4, 6)
    out_shape = list(_split_part(np.zeros(shape), axis, [2, 2], 1).shape)
    nodes, inits = _split_nodes(axis, None, 2, 1)
    _image_cell(tmp_path, nodes, shape, out_shape, inits, 13, kind, entry,
                axis < 0, f"Split even axis {axis} chunk 1 {kind} {entry}")


# --------------------------------------------------------------------------- #
# The exact method on Slice and Split: both directions, in rationals            #
# --------------------------------------------------------------------------- #
# A (2, 3, 2) image with three pixels of positive width, all straddling zero
# (pixel 1 in row 0 of channel 0, pixel 6 in row 0 and pixel 10 in row 2 of
# channel 1, so that every selection of the cells holds one) and nine of zero
# width: the exact image is a union of eight polytopes over three variables,
# so the vertices of the exact image and of every returned set can be
# enumerated in rationals. (Concat under
# ``exact`` joins different predicate systems block-diagonally and is not
# exact today: registry row REACH-4, tested in
# ``tests/soundness/test_soundness_concat.py``.)
_EXACT_CHW = (2, 3, 2)
_EXACT_LIVE = {1: (-0.5, 0.75), 6: (-1.0, 0.25), 10: (-0.5, 1.5)}
_EXACT_FIXED = (-0.75, 0.0, 0.5, -1.25, 1.5, 0.25, 0.0, -0.5, 1.0, 1.75, 0.0,
                -1.5)                     # zero-width pixels; live ones unused


def _exact_box():
    lo = np.array(_EXACT_FIXED, dtype=np.float64)
    hi = lo.copy()
    for i, (a, b) in _EXACT_LIVE.items():
        lo[i], hi[i] = a, b
    return lo.reshape(_EXACT_CHW), hi.reshape(_EXACT_CHW)


def _exact_relu_then(select, sign=1, stored=True):
    """The exact image, in rationals and without n2v, of ``Relu`` followed by
    ``select`` (a function of a (C, H, W) array: the Slice or the Split chunk)
    and multiplied by ``sign`` (-1 for the ``Neg`` after a Split chunk), over
    the box of ``_exact_box``, as pieces over the live pixels whose outputs
    are in the stored (height, width, channel) order of the selected image
    (``stored``) or in its own (channel, height, width) order, the order of a
    flat set; and the selected (C, H, W) shape."""
    live = sorted(_EXACT_LIVE)
    n = int(np.prod(_EXACT_CHW))
    embed = [[int(i == j) for j in live] for i in range(n)]
    fixed = [0 if i in _EXACT_LIVE else Fraction(_EXACT_FIXED[i])
             for i in range(n)]
    pieces = rational.relu(rational.affine(
        rational.box([_EXACT_LIVE[i][0] for i in live],
                     [_EXACT_LIVE[i][1] for i in live]), embed, fixed))
    index = select(np.arange(n).reshape(_EXACT_CHW))
    order = (index.transpose(1, 2, 0) if stored else index).reshape(-1)
    pick = [[sign * int(j == k) for j in range(n)] for k in order]
    return rational.affine(pieces, pick), index.shape


# (op, axis, ONNX starts/ends/steps or split sizes, chunk, numpy selection)
_EXACT_CELLS = {
    "slice-axis2-[1:3]": ("slice", 2, (1, 3, 1), None,
                          lambda t: t[:, 1:3, :]),
    "slice-axis3-[::-1]": ("slice", 3, (-1, INT_MIN, -1), None,
                           lambda t: t[:, :, ::-1]),
    "slice-axis1-[1:2]": ("slice", 1, (1, 2, 1), None, lambda t: t[1:2]),
    "split-axis2-[1,2]-chunk1": ("split", 2, (1, 2), 1,
                                 lambda t: t[:, 1:3, :]),
    "split-axis1-[1,1]-chunk0": ("split", 1, (1, 1), 0, lambda t: t[0:1]),
    "split-axis3-[1,1]-chunk1": ("split", 3, (1, 1), 1,
                                 lambda t: t[:, :, 1:2]),
    # the negative-axis forms
    "slice-axis-1-[::-1]": ("slice", -1, (-1, INT_MIN, -1), None,
                            lambda t: t[:, :, ::-1]),
    "split-axis-2-[1,2]-chunk1": ("split", -2, (1, 2), 1,
                                  lambda t: t[:, 1:3, :]),
}
_EXACT_MARKS = {"slice-axis-1-[::-1]": GRAPH_1,
                "split-axis-2-[1,2]-chunk1": GRAPH_4}


@pytest.mark.parametrize("cell", [
    pytest.param(cell, marks=_EXACT_MARKS.get(cell, ()), id=cell)
    for cell in _EXACT_CELLS])
def test_exact_slice_and_split_return_the_exact_image(tmp_path, cell):
    """``Relu -> Slice`` and ``Relu -> Split -> Neg`` (one chunk) on an
    ImageStar under ``method='exact'``, which claims the exact image: every
    vertex of every piece of the exact image (rationals, built without n2v) is
    in one returned set with zero slack, and every vertex of every returned
    set is a point of the exact image (``rational.unreached``). Catches a
    Slice or Split that drops, loosens or misplaces the predicate rows of an
    exact piece (the result then holds points no input reaches), that pairs a
    piece with another piece's rows (every selection holds a pixel that
    straddles zero, so the pieces differ on it), or that selects the wrong
    entries. The negative-axis cells (marked GRAPH-1 and GRAPH-4 today) keep
    the two-direction check for the day those axes answer. The rational
    model of each node is first checked against onnxruntime at 128 member
    points, exactly (dyadic data).

    The Split chunk is followed by ``Neg`` (linear, exact) rather than by a
    second ReLU: a ReLU after a ReLU meets neurons whose LP lower bound is
    exactly 0, which the exact ReLU keeps in one star with a sliver variable
    about 1e-7 wide by design (``relu_reach._phase_sliver_star``), exact only
    up to that sliver; that is the ReLU's cell, not this one."""
    op, axis, args, chunk, select = _EXACT_CELLS[cell]
    pieces, (c_out, h_out, w_out) = _exact_relu_then(
        select, -1 if op == "split" else 1)
    in_shape = (1,) + _EXACT_CHW
    out_shape = [1, c_out, h_out, w_out]
    if op == "slice":
        start, end, step = args
        nodes, inits = _slice_nodes(axis, start, end, step, 13)
    else:
        nodes, inits = _split_nodes(axis, args, len(args), chunk)
        nodes[-1] = helper.make_node("Neg", [f"P{chunk}"], ["Y"])
    lo, hi = _exact_box()
    path, outputs = _replayed(tmp_path, nodes, in_shape, out_shape, inits, 13,
                              lo, hi)
    live = sorted(_EXACT_LIVE)
    for x, y in zip(samples(lo, hi), outputs):
        value = rational.evaluate(pieces, [x.reshape(-1)[i] for i in live])
        if tuple(Fraction(v) for v in hwc(y, c_out, h_out, w_out)) != value:
            pytest.fail(f"setup: the rational model of {cell} differs from "
                        f"onnxruntime at a member point")
    sets = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                       "exact", f"exact {cell}")
    images = [_is_image(s) for s in sets]
    assert all(images) or not any(images), (
        f"exact {cell}: the result mixes image sets and flat sets")
    for s in sets:
        assert (not _is_image(s) and _entries(s) == c_out * h_out * w_out) or (
            (s.height, s.width, s.num_channels) == (h_out, w_out, c_out)), (
            f"exact {cell}: the result holds a {_describe(s)}, the output is "
            f"a (C, H, W) = {(c_out, h_out, w_out)} tensor")
    if not all(images):                 # flat sets: the tensor's own order
        pieces, _ = _exact_relu_then(select, -1 if op == "split" else 1,
                                     stored=False)
    stars = [rational.star(s) for s in sets]
    escaped = rational.escapes(pieces, *stars)
    assert not escaped, (
        f"exact {cell}: {len(escaped)} vertices of the exact image are in "
        f"none of the {len(sets)} returned sets (first: piece {escaped[0][0]})")
    # A piece spread over several returned sets leaves the vertex test
    # undecided; the exact split by activation pattern produces none. A finer
    # split would need the pieces cut along the returned rows first
    # (``rational.split``) before this check can pass.
    spread = rational.straddlers(pieces, *stars)
    assert not spread, (
        f"exact {cell}: pieces {spread} of the exact image are each spread "
        f"over several returned sets (one star per activation pattern "
        f"expected); the vertex test cannot decide their containment")
    for k, star in enumerate(stars):
        extra = rational.unreached(pieces, rational.set_vertices(star))
        assert not extra, (
            f"exact {cell}: returned set {k} of {len(sets)} has "
            f"{len(extra)} vertices that no input reaches (first: "
            f"{[float(v) for v in extra[0][1]]})")


# --------------------------------------------------------------------------- #
# A5: Slice of opset 9                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("follow, axis, kind", [
    pytest.param("alone", -1, "IZ-approx", marks=GRAPH_7,
                 id="alone-axis-1-IZ-approx"),
    pytest.param("reducesum", -3, "IS-approx", marks=GRAPH_7,
                 id="reducesum-axis-3-IS-approx"),
    pytest.param("alone", 2, "IZ-approx", id="alone-axis2-IZ-approx"),
    pytest.param("reducesum", 1, "IS-approx", id="reducesum-axis1-IS-approx"),
    pytest.param("reducesum", 3, "IS-exact", id="reducesum-axis3-IS-exact")])
def test_opset9_slice_on_an_image_set_holds_every_model_output(
        tmp_path, follow, axis, kind):
    """``Relu -> Slice`` (opset 9, attributes ``starts``/``ends``/``axes``,
    ``[1:3]``) on tensor (1, 4, 4, 4), alone and followed by
    ``ReduceSum(axes 1, 2, 3, keepdims 0)``, which accepts any input size.
    Catches an opset 9 Slice whose negative-axis part is skipped (the
    identity), which alone has the wrong size and after the ReduceSum is a
    one-entry set that leaves out every true output, and an opset 9 Slice
    whose non-negative axes are not shifted past the batch axis."""
    shape = (1, 4, 4, 4)
    out_shape = _sliced_shape(shape, axis, 1, 3, 1)
    nodes, inits = _slice_nodes(axis, 1, 3, 1, 9)
    if follow == "reducesum":
        nodes[-1].output[0] = "S"
        nodes.append(helper.make_node("ReduceSum", ["S"], ["Y"],
                                      axes=[1, 2, 3], keepdims=0))
        out_shape = [1]
    _image_cell(tmp_path, nodes, shape, out_shape, inits, 9, kind, "model",
                axis < 0, f"Slice opset 9 {follow} axis {axis} {kind}")


def _slice9_flat_params():
    out = []
    for kind, method in (("Star", "approx"), ("Star", "exact"),
                         ("Zono", "approx"), ("Box", "approx")):
        out.append(pytest.param(-1, kind, method, marks=GRAPH_7,
                                id=f"axis-1-{kind}-{method}"))
    for kind in ("Star", "Zono", "Box"):
        out.append(pytest.param(1, kind, "approx", id=f"axis1-{kind}-approx"))
    out.append(pytest.param(2, "Star", "exact", id="axis2-Star-exact"))
    return out


@pytest.mark.parametrize("axis, kind, method", _slice9_flat_params())
def test_opset9_slice_on_a_flat_set_behind_a_reshape(tmp_path, axis, kind,
                                                     method):
    """``Reshape(1, 4, 4) -> Relu -> Slice`` (opset 9, ``[1:3]``) from a flat
    set of 16 entries, the model input (1, 16): the true output (1, 4, 2) or
    (1, 2, 4) has 8 entries. Catches an opset 9 Slice whose negative-axis
    part is skipped on a flat set with tracked shapes, and a shape-aware
    opset 9 Slice that drops an axis or reads the selected rows in the wrong
    order."""
    lo, hi = _image_box(1, 4, 4)
    lo, hi = lo.reshape(16), hi.reshape(16)
    out_shape = _sliced_shape((1, 4, 4), axis, 1, 3, 1)
    nodes = [helper.make_node("Reshape", ["X", "shape"], ["Z"]),
             helper.make_node("Relu", ["Z"], ["R"]),
             helper.make_node("Slice", ["R"], ["Y"], starts=[1], ends=[3],
                              axes=[axis])]
    path, outputs = _replayed(tmp_path, nodes, (1, 16), out_shape,
                              {"shape": np.array([1, 4, 4], np.int64)}, 9,
                              lo, hi)
    what = f"Slice opset 9 flat axis {axis} {kind} {method}"
    sets = _answer(path, flat_set(kind, lo, hi), method, "model", what,
                   may_refuse=axis < 0)
    if sets is not None:
        _assert_holds(sets, outputs, out_shape, what)
