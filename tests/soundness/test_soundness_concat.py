"""Soundness tests for set concatenation: ONNX Concat through the model entry
point, the kernel ``_concat_sets`` on lists of sets, the other entry routes
(``reach_layer``, a traced torch model, a flat set without a tracked shape),
the refusals, and the marked defects of today's library (operands of two set
families, the exact method on two predicate systems).
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.nn.reach import _concat_sets
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Zono


# =========================================================================== #
# Cells: ONNX Concat through the model entry point, and the kernel             #
# =========================================================================== #
# What these cells check: every true output of the network on a member point
# of the input set is a member of a returned set. Oracles, none of them n2v:
# onnxruntime replay of the same ONNX file (dyadic data, so float32 is exact),
# membership by an LP on the rows of the returned sets
# (``tests._oracles.membership`` through ``cell_support.assert_covered``), the
# exact image in rationals (``tests._oracles.rational``) where the method
# claims exactness, and for direct kernel calls the concatenation of member
# points computed from the arrays the test built. Layout: a tensor is
# (C, H, W) row-major, a flat set keeps the tensor's own row-major order, an
# image set stores (H, W, C) and is compared through ``out_chw``. Negative
# axes on image sets, a constant operand next to an image set and the batch
# axis belong to ``test_soundness_negative_axis.py``.

import pytest  # noqa: E402
from onnx import helper  # noqa: E402

from n2v.sets import Hexatope, Octatope  # noqa: E402
from n2v.utils.model_loader import load_onnx  # noqa: E402
from tests._oracles import rational as Q  # noqa: E402
from tests.cell_support import (  # noqa: E402
    assert_covered, constrained_image_star, constrained_star, flat_set,
    hwc_index, image_set, image_split_cover, reach_model, replay_onnx,
    replay_torch, samples, save_onnx, split_cover)
from tests.known_bugs import known_bug  # noqa: E402
from onnx2torch.node_converters.concat import OnnxConcat  # noqa: E402


def _grid(cells):
    """Parameters ``(kind, method, *option)`` from ``{(kind, method): [option,
    ...]}``: the handler code shared by all kinds (axis normalisation, row
    map, constant placement) is swept on the cheap Box (and a Star approx),
    the kind-specific kernels on fewer options, to keep the file fast."""
    return [(k, m) + (o if isinstance(o, tuple) else (o,))
            for (k, m), options in cells.items() for o in options]


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _i64(*values):
    return np.array(values, dtype=np.int64)


def _tensor_box(shape, crossing=(0, 4), flat=None):
    """Dyadic bounds ``(lo, hi)`` of a tensor of ``shape``: every entry has its
    own interval (distinct centres, widths 1/4 to 3/4), the entries listed in
    ``crossing`` (row-major index) straddle zero, the entry ``flat`` has zero
    width, and the others lie strictly on one side of zero (two in three
    positive). A result that puts an entry at another position misses the
    outputs."""
    n = int(np.prod(shape))
    lo, hi = np.empty(n), np.empty(n)
    for i in range(n):
        half = (1 + i % 3) / 8.0
        centre = (i + 2) / 4.0 * (-1.0 if i % 3 == 2 else 1.0)
        if i in crossing:
            lo[i], hi[i] = -(2 + i % 5) / 8.0, (1 + i % 3) / 8.0
        elif i == flat:
            lo[i] = hi[i] = centre
        else:
            lo[i], hi[i] = centre - half, centre + half
    return lo.reshape(shape), hi.reshape(shape)


def _const(shape):
    """A constant tensor with a different dyadic value in every entry."""
    n = int(np.prod(shape))
    values = (np.arange(n) * 3 % 17 - 8) / 4.0 + np.arange(n) / 64.0
    return _f32(values.reshape(shape))


def _input_set(kind, lo, hi, constrained=False):
    """The input set of ``kind`` for the (C, H, W) bounds and its member
    points in network order (corners and grid points of the box, or for a
    constrained Star / ImageStar the points the builder certifies)."""
    if constrained:
        build = constrained_image_star if kind == "ImageStar" else \
            constrained_star
        return build(lo, hi)
    if kind in ("ImageStar", "ImageZono"):
        return image_set(kind, lo, hi), samples(lo, hi)
    return flat_set(kind, lo, hi), samples(lo, hi)


def _check_model(tmp_path, nodes, inits, in_shape, out_shape, kind, method,
                 what, opset=13, box=None, constrained=False):
    """Reach the ONNX graph from the input set of ``kind`` built on ``box``
    (default :func:`_tensor_box`) and assert that every onnxruntime output
    of a member point is in a returned set, compared in the layout each
    returned set stores."""
    lo, hi = box if box is not None else _tensor_box(tuple(in_shape[1:]))
    path = save_onnx(tmp_path, nodes, in_shape, out_shape, inits=inits,
                     opset=opset)
    input_set, points = _input_set(kind, lo, hi, constrained)
    outputs = replay_onnx(path, points, in_shape)
    sets = reach_model(load_onnx(path), input_set, method, what,
                       input_shape=tuple(in_shape[1:]))
    out_chw = tuple(out_shape[1:]) if len(out_shape) == 4 else None
    assert_covered(sets, outputs, what, out_chw=out_chw)
    return sets


def _cat_shape(shapes, axis):
    out = list(shapes[0])
    out[axis] = sum(s[axis] for s in shapes)
    return out


def _concat_graph(variant, shape, axis):
    """``(nodes, inits, out_shape)`` of a Concat graph on input X of
    ``shape`` (with batch axis) along ``axis``.

    - ``relu_neg``: Concat(Relu(X), Neg(X)), two operands of one input that
      differ everywhere, so the joint output is checked, not each half;
    - ``neg_relu``: Concat(Neg(X), Relu(X)), the same operands in the other
      order, so the operand with the input's own rows comes first;
    - ``two_relus``: Concat(Relu(X), Relu(Neg(X))), each branch with its own
      relaxation variables (independent noise symbols for a Zono);
    - ``three_systems``: Concat(Relu(X), X, Relu(Neg(X))), three operands
      with three different predicate systems of different sizes (the
      input's, and each ReLU branch's with its own relaxation variables);
    - ``two_relus_apart``: Concat(Relu(X), Relu(X - K)), where K moves the
      entries that straddle zero: the first branch relaxes entries 0 and 4,
      the second entries 1 and 5, so their new error symbols stand for
      different neurons;
    - ``shared_then_relu``: Concat(X, Neg(X), Relu(X)), two operands that
      hold the input's own predicate system, then one with its own
      relaxation variables (two systems for three operands);
    - ``three_unequal``: Concat(Relu(X), Neg(Slice(X)[1:2]), X), three
      operands, the middle one of length 1 along the axis;
    - ``const_first`` / ``const_last`` / ``const_middle``: a constant operand
      of length 1 along the axis (two non-singleton other axes, every entry
      different) before, after or between Relu(X) and Neg(X)."""
    a = axis % len(shape)
    one = list(shape)
    one[a] = 1
    relu = helper.make_node("Relu", ["X"], ["A"])
    neg = helper.make_node("Neg", ["X"], ["B"])
    if variant in ("relu_neg", "neg_relu"):
        order = ["A", "B"] if variant == "relu_neg" else ["B", "A"]
        return ([relu, neg, helper.make_node("Concat", order, ["Y"],
                                             axis=axis)],
                {}, _cat_shape([shape, shape], a))
    if variant == "two_relus":
        return ([relu, neg, helper.make_node("Relu", ["B"], ["R"]),
                 helper.make_node("Concat", ["A", "R"], ["Y"], axis=axis)],
                {}, _cat_shape([shape, shape], a))
    if variant == "three_systems":
        return ([relu, neg, helper.make_node("Relu", ["B"], ["R"]),
                 helper.make_node("Concat", ["A", "X", "R"], ["Y"],
                                  axis=axis)],
                {}, _cat_shape([shape, shape, shape], a))
    if variant == "shared_then_relu":
        return ([relu, neg, helper.make_node("Concat", ["X", "B", "A"], ["Y"],
                                             axis=axis)],
                {}, _cat_shape([shape, shape, shape], a))
    if variant == "two_relus_apart":
        k = np.zeros(int(np.prod(shape)))
        k[[0, 4]] = -1.0                  # entries 0 and 4 no longer cross
        k[1], k[5] = 0.75, -1.75          # entries 1 and 5 now cross
        return ([relu, helper.make_node("Sub", ["X", "K"], ["D"]),
                 helper.make_node("Relu", ["D"], ["R"]),
                 helper.make_node("Concat", ["A", "R"], ["Y"], axis=axis)],
                {"K": _f32(k.reshape(shape))}, _cat_shape([shape, shape], a))
    if variant == "three_unequal":
        nodes = [relu,
                 helper.make_node("Slice", ["X", "st", "en", "ax"], ["S"]),
                 helper.make_node("Neg", ["S"], ["N"]),
                 helper.make_node("Concat", ["A", "N", "X"], ["Y"],
                                  axis=axis)]
        inits = {"st": _i64(1), "en": _i64(2), "ax": _i64(axis)}
        return nodes, inits, _cat_shape([shape, one, shape], a)
    operands = {"const_first": ["K", "A"], "const_last": ["A", "K"],
                "const_middle": ["A", "K", "B"]}[variant]
    shapes = [one if o == "K" else shape for o in operands]
    return ([relu, neg, helper.make_node("Concat", operands, ["Y"],
                                         axis=axis)],
            {"K": _const(one)}, _cat_shape(shapes, a))


_SHAPE = [1, 2, 3, 5]


@pytest.mark.parametrize("kind,method,axis", _grid({
    ("Box", "approx"): [1, 2, 3, -1, -3], ("Star", "approx"): [1, 3, -1, -3],
    ("Star", "exact"): [1, 3, -1], ("Zono", "approx"): [3, -3]}))
def test_concat_of_two_branches_of_a_flat_set_holds_every_output(
        tmp_path, kind, method, axis):
    """Concat(Relu(X), Neg(X)) on a flat set standing for a (1, 2, 3, 5)
    tensor, along the first, a middle and the last non-batch axis and their
    negative forms (the handler code shared by every kind is swept on the
    Box, each kind's own row selection on its first, last and negative
    axes). Catches a row map built for the wrong axis (a negative axis read
    with abs() or off by one, an inner-axis concat returned in stacked
    order, a dropped block offset), operands stacked in the wrong order, a
    Box whose upper bounds are the lower bounds, and an exact Relu list that
    loses pieces when paired with the single Neg set."""
    nodes, inits, out = _concat_graph("relu_neg", _SHAPE, axis)
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Concat(Relu, Neg) axis {axis}, {kind} {method}")


@pytest.mark.parametrize("kind,method,shape,axis", [
    pytest.param("ImageStar", "approx", (1, 2, 3, 5), 1,
                 id="ImageStar-approx-shape0-1"),
    pytest.param("ImageStar", "approx", (1, 2, 3, 5), 2,
                 id="ImageStar-approx-shape1-2"),
    pytest.param("ImageStar", "approx", (1, 2, 3, 5), 3,
                 id="ImageStar-approx-shape2-3"),
    pytest.param("ImageStar", "exact", (1, 2, 3, 5), 2,
                 id="ImageStar-exact-shape6-2"),
    pytest.param("ImageStar", "exact", (1, 3, 4, 2), 3,
                 id="ImageStar-exact-shape7-3"),
    pytest.param("ImageZono", "approx", (1, 2, 3, 5), 1,
                 id="ImageZono-approx-shape8-1"),
    pytest.param("ImageZono", "approx", (1, 2, 3, 5), 2,
                 id="ImageZono-approx-shape9-2"),
    pytest.param("ImageZono", "approx", (1, 2, 3, 5), 3,
                 id="ImageZono-approx-shape10-3")])
def test_concat_of_two_branches_of_an_image_set_holds_every_output(
        tmp_path, shape, kind, method, axis):
    """Concat(Relu(X), Neg(X)) on an ImageStar or ImageZono of a (2, 3, 5) and
    a (3, 4, 2) image, along the channel, height and width axes. Catches a
    joint basis read as (C, H, W) when it is stored (H, W, C), operands
    concatenated in reverse order, the ONNX axis mapped to the wrong stored
    axis, an ImageZono whose centre and generators are joined along different
    axes or in different operand orders. (Overlapping ImageZono generator
    blocks do not show here: Relu(X) and Neg(X) share the input's noise
    symbols. The relaxed-branch cells and the image-list kernel cells catch
    them.)"""
    shape = list(shape)
    nodes, inits, out = _concat_graph("relu_neg", shape, axis)
    _check_model(tmp_path, nodes, inits, shape, out, kind, method,
                 f"Concat(Relu, Neg) axis {axis} on {shape}, {kind} {method}")


@pytest.mark.parametrize("kind,method,axis", [
    ("Box", "approx", 1), ("Box", "approx", 2), ("Box", "approx", -1),
    ("Box", "approx", -2), ("Star", "approx", -1)])
def test_concat_of_rank_three_tensors_holds_every_output(tmp_path, kind,
                                                         method, axis):
    """Concat(Relu(X), Neg(X)) on a flat set standing for a (1, 3, 4) tensor.
    On rank 3 a negative axis read with abs() lands on axis 1 instead of 2,
    and an axis taken one too low concatenates along the wrong axis; both
    give a set of the right size whose entries are in the wrong order."""
    shape = [1, 3, 4]
    nodes, inits, out = _concat_graph("relu_neg", shape, axis)
    _check_model(tmp_path, nodes, inits, shape, out, kind, method,
                 f"Concat rank 3 axis {axis}, {kind} {method}")


@pytest.mark.parametrize("kind,method,axis", _grid({
    ("Box", "approx"): [1, 2, 3], ("Star", "approx"): [3],
    ("Star", "exact"): [3], ("Zono", "approx"): [2],
    ("ImageStar", "approx"): [1, 2, 3], ("ImageZono", "approx"): [2, 3]}))
def test_concat_of_three_operands_of_unequal_length_holds_every_output(
        tmp_path, kind, method, axis):
    """Concat(Relu(X), Neg(X[1:2]), X): three operands, the middle one of
    length 1 along the joined axis. Catches a block offset that is not
    advanced past an operand of another length, a row map that assumes equal
    operands, the operands stacked in the wrong order, and an image concat
    that reads the second operand's shape from the first."""
    nodes, inits, out = _concat_graph("three_unequal", _SHAPE, axis)
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Concat of three operands axis {axis}, {kind} {method}")


_CONSTS = ["const_first", "const_last", "const_middle"]


@pytest.mark.parametrize("kind,method,variant,axis", _grid({
    ("Box", "approx"): [(v, a) for v in _CONSTS for a in (1, 3, -1)],
    ("Star", "approx"): [(v, a) for v in _CONSTS for a in (1, 3)],
    ("Star", "exact"): [("const_last", 2), ("const_middle", -1)],
    ("Zono", "approx"): [(v, 3) for v in _CONSTS] + [("const_last", 1)]}))
def test_concat_with_a_constant_operand_holds_every_output(
        tmp_path, kind, method, variant, axis):
    """Concat with a constant operand (every entry different, two
    non-singleton axes besides the joined one) before, after or between two
    set operands. Catches a constant that is not written to the centre, is
    written to a generator column, is flattened in column-major order, is
    replaced by zeros, or is put in front of the operand list whatever its
    position."""
    nodes, inits, out = _concat_graph(variant, _SHAPE, axis)
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Concat {variant} axis {axis}, {kind} {method}")


@pytest.mark.parametrize("kind,method,variant,axis", _grid({
    ("Star", "approx"): [("two_relus", -2), ("two_relus_apart", 2),
                         ("three_systems", 3), ("shared_then_relu", 2)],
    ("Star", "exact"): [("two_relus", 3)],
    ("Zono", "approx"): [("two_relus", 1), ("two_relus_apart", 3),
                         ("two_relus_apart", -1)],
    ("ImageStar", "approx"): [("two_relus_apart", 2), ("three_systems", 1),
                              ("shared_then_relu", 3)],
    ("ImageZono", "approx"): [("two_relus_apart", 1),
                              ("two_relus_apart", 3)]}))
def test_concat_of_two_relaxed_branches_holds_every_output(tmp_path, kind,
                                                           method, variant,
                                                           axis):
    """Concat(Relu(X), Relu(Neg(X))): each branch adds its own relaxation
    variables (Star) or error symbols (Zono), and the exact method splits
    each branch on its own rows; with X between the two ReLU branches, three
    predicate systems of different sizes. Catches generator columns of the
    two operands placed on the same noise symbols (seen when the two
    branches relax different neurons), two predicate systems with the same
    number of variables read as one, a block offset that is right for two
    systems only (the third system written over the first one's variables),
    an operand that shares the system of the one before it counted as a
    system of its own (X, Neg(X), then Relu(X): the operand after them is
    then read against the wrong block), and the exact pieces of the two
    branches paired by position instead of by lineage."""
    nodes, inits, out = _concat_graph(variant, _SHAPE, axis)
    box = None
    if variant == "shared_then_relu":
        # Eight times the usual box: a relaxation variable of a neuron that
        # reaches above 1 has a range beyond [-1, 1], the range of every
        # variable of the input's own system, so one of those read in its
        # place does not hold the outputs.
        lo, hi = _tensor_box(tuple(_SHAPE[1:]))
        box = (8 * lo, 8 * hi)
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Concat {variant} axis {axis}, {kind} {method}", box=box)


@pytest.mark.parametrize("kind,axis", [("Star", 3), ("ImageStar", 1)])
def test_exact_concat_of_three_split_branches_holds_every_output(tmp_path,
                                                                 kind, axis):
    """Concat(Relu(X), Relu(Neg(X)), Relu(X - K)), method 'exact', on a
    (1, 2, 1, 2) tensor: entry 0 of X straddles zero, so the first two
    branches split on it; K moves the straddle to entry 1 in the third
    branch, which splits there. Three operand lists of two pieces each, none
    descended from another, so every piece of each list must meet every
    piece of the others. Catches a pairing of three or more multi-piece
    lists that keeps only one piece of a later list (or pairs them by
    position): the outputs of the input points on the other side of entry
    1 then escape. The check is the onnxruntime replay with membership on
    the rows of the returned sets."""
    shape = [1, 2, 1, 2]
    lo = np.array([-0.5, 0.25, -0.75, 0.125]).reshape(2, 1, 2)
    hi = np.array([0.25, 0.75, -0.25, 0.5]).reshape(2, 1, 2)
    K = _f32(np.array([-1.0, 0.5, 0.0, 0.0]).reshape(shape))
    nodes = [helper.make_node("Relu", ["X"], ["A"]),
             helper.make_node("Neg", ["X"], ["N"]),
             helper.make_node("Relu", ["N"], ["B"]),
             helper.make_node("Sub", ["X", "K"], ["D"]),
             helper.make_node("Relu", ["D"], ["E"]),
             helper.make_node("Concat", ["A", "B", "E"], ["Y"], axis=axis)]
    out = _cat_shape([shape] * 3, axis)
    _check_model(tmp_path, nodes, {"K": K}, shape, out, kind, "exact",
                 f"exact Concat of three split branches, {kind} axis {axis}",
                 box=(lo, hi))


def _binding_row_params():
    """The cells of the binding-row test: ``(kind, method, axis, variant)``;
    the Concat(Relu, Neg) cells keep their ids ``kind-method-axis``."""
    out = [pytest.param(k, m, a, "relu_neg", id=f"{k}-{m}-{a}")
           for k, m, a in _grid({("Star", "approx"): [1, 3],
                                 ("Star", "exact"): [2],
                                 ("ImageStar", "approx"): [1, 2, 3],
                                 ("ImageStar", "exact"): [3]})]
    out += [pytest.param(k, "approx", a, "neg_relu",
                         id=f"{k}-approx-{a}-neg_relu")
            for k, a in (("Star", 3), ("ImageStar", 1))]
    return out


@pytest.mark.parametrize("kind,method,axis,variant", _binding_row_params())
def test_concat_of_a_star_with_a_binding_row_holds_every_output(
        tmp_path, kind, method, axis, variant):
    """Concat(Relu(X), Neg(X)) and Concat(Neg(X), Relu(X)) on a Star /
    ImageStar whose predicate is cut by a row that is not a box row (the
    bounding box is the full box, the set is not). Catches a joint
    constraint system that misplaces the operands' rows (a block written on
    the first operand's variables instead of its own: with Neg(X) second its
    rows are the input's own and the error does not show, with Neg(X) first
    the relaxed operand's rows land on the input's variables): a box check
    passes such a result, the membership of the builder's points on the cut
    face does not."""
    nodes, inits, out = _concat_graph(variant, _SHAPE, axis)
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Concat {variant} on a constrained {kind} axis {axis}, "
                 f"{method}", constrained=True)


def _flat_composition(axis):
    """X (1, 6) -> Gemm -> Reshape (1, 2, 3, 2) -> Concat(Relu, Neg) along
    ``axis`` -> Flatten -> Gemm (5 outputs): affine, the operation, affine.
    The first Gemm mixes every input; its bias puts exactly two of the twelve
    pre-activations across zero (decided by interval arithmetic on the
    dyadic box); the last Gemm has a few non-zero weights per row, so an
    entry at a wrong position changes its outputs."""
    rng = np.random.default_rng(11)
    lo, hi = _tensor_box((6,), crossing=())
    W1 = rng.integers(-2, 3, size=(12, 6)) / 2.0
    mid = (W1 @ (lo + hi)) / 2.0
    rad = np.abs(W1) @ ((hi - lo) / 2.0)
    target = np.where(np.arange(12) % 3 == 0, 1.0, -1.0) * (rad + 0.25)
    target[[0, 7]] = rad[[0, 7]] / 4.0          # these two straddle zero
    b1 = target - mid                           # dyadic: no rounding
    zlo, zhi = mid + b1 - rad, mid + b1 + rad
    crossing = int(np.sum((zlo < 0) & (zhi > 0)))
    if crossing != 2:
        pytest.fail(f"composition: {crossing} crossing pre-activations, "
                    f"expected 2")
    W2 = np.zeros((5, 24))
    for r in range(5):
        cols = rng.choice(24, size=4, replace=False)
        W2[r, cols] = rng.choice([-1.0, -0.5, 0.5, 1.0, 2.0], size=4)
    t_shape = [1, 2, 3, 2]
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["Z"], transB=1),
             helper.make_node("Reshape", ["Z", "shape"], ["T"]),
             helper.make_node("Relu", ["T"], ["A"]),
             helper.make_node("Neg", ["T"], ["B"]),
             helper.make_node("Concat", ["A", "B"], ["C"], axis=axis),
             helper.make_node("Flatten", ["C"], ["F"], axis=1),
             helper.make_node("Gemm", ["F", "W2"], ["Y"], transB=1)]
    inits = {"W1": _f32(W1), "b1": _f32(b1), "shape": _i64(*t_shape),
             "W2": _f32(W2)}
    return nodes, inits, (lo, hi)


@pytest.mark.parametrize("kind,method,axis", _grid({
    ("Box", "approx"): [1, 3, -1], ("Star", "approx"): [2, -1],
    ("Star", "exact"): [3], ("Zono", "approx"): [1, 3]}))
def test_concat_between_two_affine_layers_holds_every_output(
        tmp_path, kind, method, axis):
    """Gemm, Reshape, Concat(Relu, Neg), Flatten, Gemm on a flat set: the rows
    the Concat emits are consumed by a Gemm whose rows single out entries,
    so a result with the right entries in the wrong order, or a constraint
    row lost in the join, changes the five outputs. The input set has only
    two crossing neurons, so the exact method splits into a few pieces."""
    nodes, inits, box = _flat_composition(axis)
    _check_model(tmp_path, nodes, inits, [1, 6], [1, 5], kind, method,
                 f"Gemm > Concat axis {axis} > Gemm, {kind} {method}", box=box)


@pytest.mark.parametrize("kind,method,axis", _grid({
    ("ImageStar", "approx"): [1, 2, 3], ("ImageStar", "exact"): [2],
    ("ImageZono", "approx"): [1, 3]}))
def test_concat_of_images_between_conv_and_gemm_holds_every_output(
        tmp_path, kind, method, axis):
    """Conv 1x1 (mixes the two channels), Concat(Relu, Neg) along each axis,
    Flatten, Gemm on a (2, 3, 2) image set: the Concat result is read by
    Flatten, which converts the stored (H, W, C) order to the tensor's order,
    so a concat that stores the right entries at the wrong pixels or
    channels changes the outputs of the Gemm."""
    rng = np.random.default_rng(5)
    Wc = _f32([[[[1.0]], [[0.5]]], [[[-0.5]], [[1.0]]]])
    bc = _f32([0.25, -0.5])
    W2 = np.zeros((5, 24))
    for r in range(5):
        cols = rng.choice(24, size=4, replace=False)
        W2[r, cols] = rng.choice([-1.0, -0.5, 0.5, 1.0, 2.0], size=4)
    nodes = [helper.make_node("Conv", ["X", "Wc", "bc"], ["T"]),
             helper.make_node("Relu", ["T"], ["A"]),
             helper.make_node("Neg", ["T"], ["B"]),
             helper.make_node("Concat", ["A", "B"], ["C"], axis=axis),
             helper.make_node("Flatten", ["C"], ["F"], axis=1),
             helper.make_node("Gemm", ["F", "W2"], ["Y"], transB=1)]
    _check_model(tmp_path, nodes, {"Wc": Wc, "bc": bc, "W2": _f32(W2)},
                 [1, 2, 3, 2], [1, 5], kind, method,
                 f"Conv > Concat axis {axis} > Gemm, {kind} {method}",
                 box=_tensor_box((2, 3, 2), crossing=(0, 7)))


@pytest.mark.parametrize("kind", ["Box", "ImageStar"])
def test_concat_of_two_constants_added_to_the_input_holds_every_output(
        tmp_path, kind):
    """X + Concat(K1, K2) along the width: a Concat with no set operand (the
    handler finds no set list) is folded to a constant tensor, which must be
    added entry by entry in the tensor's order (K1's column then K2's).
    Catches a Concat of constants that is not left to the fold (an empty
    list of sets, or another node's sets, put in its place)."""
    shape = [1, 2, 3, 2]
    K1 = _const([1, 2, 3, 1])
    K2 = -_const([1, 2, 3, 1]) / 2 + 1
    nodes = [helper.make_node("Concat", ["K1", "K2"], ["KK"], axis=3),
             helper.make_node("Add", ["X", "KK"], ["Y"])]
    _check_model(tmp_path, nodes, {"K1": K1, "K2": _f32(K2)}, shape, shape,
                 kind, "approx", f"X + Concat(K1, K2), {kind}")


# --------------------------------------------------------------------------- #
# The exact method on a Concat of one lineage: the exact image, both ways      #
# --------------------------------------------------------------------------- #
def _vertices_of(computed):
    """The images of the vertices of a computed set's predicate polytope
    (``rational.set_vertices``), skipping its boundedness test when every
    predicate variable has two finite bounds (the polytope is then bounded)
    and dropping repeated rows: the same answer, several times faster."""
    c, G = computed[0], computed[1]
    if any(v is None for v in computed[4] + computed[5]) or not computed[4]:
        return Q.set_vertices(computed)
    A, b = Q._predicate_rows(computed)
    unique = sorted({(tuple(row), rhs) for row, rhs in zip(A, b)})
    corners = Q.vertices([list(r) for r, _ in unique], [r for _, r in unique])
    return sorted({tuple(Q.dot(row, a) + ci for row, ci in zip(G, c))
                   for a in corners})


def _linear_map(fn, n):
    """The matrix of the linear map ``fn`` on vectors of ``n`` entries,
    computed by applying it to the unit vectors (numpy only)."""
    return np.stack([fn(np.eye(n)[j]) for j in range(n)], axis=1)


@pytest.mark.parametrize("kind,axis,operands", [
    pytest.param("Star", 1, "RN", id="Star-1"),
    pytest.param("Star", 3, "RN", id="Star-3"),
    pytest.param("ImageStar", 2, "RN", id="ImageStar-2"),
    pytest.param("ImageStar", 3, "RN", id="ImageStar-3"),
    pytest.param("Star", 3, "RNR", id="Star-3-three_operands")])
def test_exact_concat_of_one_lineage_is_the_exact_image(tmp_path, kind, axis,
                                                        operands):
    """Concat(R, Neg(R)) and Concat(R, Neg(R), R) with R = Relu(X) on a
    (1, 2, 1, 2) tensor whose entries 0 and 3 straddle zero, method 'exact'.
    Every operand comes from the same exact pieces, so the joined set of each
    piece is exact. Checked in rationals both ways: every vertex of every
    exact piece is in one returned set (no piece dropped or paired with
    another piece's operand), and every vertex of every returned set is the
    image of an input point (no piece joined with a piece of another
    activation pattern; with three operand lists, no list combined with
    every piece of another)."""
    shape = [1, 2, 1, 2]
    lo, hi = _tensor_box(tuple(shape[1:]), crossing=(0, 3))
    names = {"R": "R", "N": "N"}
    nodes = [helper.make_node("Relu", ["X"], ["R"]),
             helper.make_node("Neg", ["R"], ["N"]),
             helper.make_node("Concat", [names[o] for o in operands], ["Y"],
                              axis=axis)]
    out = _cat_shape([shape] * len(operands), axis)
    sets = _check_model(tmp_path, nodes, {}, shape, out, kind, "exact",
                        f"exact Concat {operands} axis {axis}, {kind}",
                        box=(lo, hi))
    sign = {"R": 1.0, "N": -1.0}

    def network_after_relu(v):
        r = v.reshape(shape)
        return np.concatenate([sign[o] * r for o in operands],
                              axis=axis).reshape(-1)
    M = _linear_map(network_after_relu, 4)
    if kind == "ImageStar":
        M = M[hwc_index(*out[1:])]            # the order the set stores
    pieces = Q.affine(Q.relu(Q.box(lo.reshape(-1), hi.reshape(-1))), M)
    computed = [Q.star(s) for s in sets]
    # A piece with every vertex in one returned set lies in it (convexity);
    # no straddler therefore means no escape either.
    assert Q.straddlers(pieces, *computed) == [], \
        "an exact piece has a vertex outside, or is spread over several sets"
    # The check below enumerates the vertices of each returned set's
    # predicate polytope, which it can do for the input's four variables. A
    # join that shares the variables of one lineage adds none; a set with
    # more of them had an operand joined as a system of its own, and its
    # enumeration would not finish. (An exact join that adds variables of
    # its own would need another check here.)
    for k, c in enumerate(computed):
        assert len(c[4]) <= 4, (
            f"returned set {k} has {len(c[4])} predicate variables for an "
            f"input of 4: operands of one lineage were not joined on their "
            f"shared variables")
    for k, c in enumerate(computed):
        assert Q.unreached(pieces, _vertices_of(c)) == [], \
            f"returned set {k} has a vertex that no input point reaches"


# --------------------------------------------------------------------------- #
# The kernel _concat_sets on lists of several sets                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,parts", [
    ("Star", (3, 1)), ("Star", (2, 3)), ("Zono", (1, 3)), ("Zono", (2, 3)),
    ("Box", (3, 1)), ("Box", (1, 3)), ("Box", (2, 3))])
def test_concat_kernel_on_lists_of_several_sets_holds_the_product(kind,
                                                                  parts):
    """_concat_sets on two operand lists, each the union of ``parts`` boxes
    built independently (a 4-entry and a 3-entry box): the true set is the
    product of the two unions. Catches pieces paired by position (a longer
    list loses pieces), a single set not broadcast against the other list,
    and pieces of one list joined to the wrong pieces of the other."""
    loA, hiA = _tensor_box((4,), crossing=(1,))
    loB, hiB = _tensor_box((3,), crossing=(2,))
    loB, hiB = loB + 3.0, hiB + 3.0
    A = split_cover(kind, loA, hiA, parts[0]) if parts[0] > 1 else \
        [flat_set(kind, loA, hiA)]
    B = split_cover(kind, loB, hiB, parts[1]) if parts[1] > 1 else \
        [flat_set(kind, loB, hiB)]
    sets = _concat_sets([A, B], axis=0)
    points = samples(np.concatenate([loA, loB]), np.concatenate([hiA, hiB]))
    assert_covered(sets, points, f"_concat_sets {kind} parts {parts}")


def test_concat_kernel_of_independent_zonotopes_keeps_their_noise_apart():
    """_concat_sets of two zonotopes built from different arrays, each with
    generators that mix its entries: their noise symbols are independent, so
    the concatenation must hold every pair (a, b). Catches generator columns
    of the second operand written over the first operand's columns (the
    k-th generator of every operand read as one symbol) and rows written over
    each other."""
    cA = np.array([[1.0], [-2.0], [0.5]])
    GA = np.array([[1.0, 0.5], [-0.5, 1.0], [0.25, -1.0]])
    cB = np.array([[3.0], [-1.0]])
    GB = np.array([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]])
    out = _concat_sets([[Zono(cA, GA)], [Zono(cB, GB)]], axis=0)
    # Every pair of a symbol vector of A (the 3 x 3 grid of {-1, 0, 1}^2)
    # with a corner of B's cube ({-1, 1}^3): the two are chosen apart.
    points = []
    for ia in np.ndindex(3, 3):
        ea = np.array(ia, dtype=float) - 1.0
        for ib in np.ndindex(2, 2, 2):
            eb = 2.0 * np.array(ib, dtype=float) - 1.0
            points.append(np.concatenate([cA[:, 0] + GA @ ea,
                                          cB[:, 0] + GB @ eb]))
    assert_covered(out, points, "_concat_sets of independent Zonos")


def test_concat_kernel_of_two_constrained_stars_holds_every_pair():
    """_concat_sets of two Stars with different predicate systems, each cut
    by a row that is not a box row: the true set is the product of the two
    sets. Catches the two systems read as one (when their variable counts
    are equal) and a joint constraint system that misplaces a cut row (one
    operand's row written on the other operand's variables). Every member
    point of each Star is paired with a dozen of the other's, so the points
    on each cut face meet the far corners of the other set."""
    loA, hiA = _tensor_box((4,), crossing=(0, 3))
    loB, hiB = _tensor_box((4,), crossing=(1,))
    A, pa = constrained_star(loA, hiA, n=24, seed=1)
    B, pb = constrained_star(loB, hiB, n=24, seed=2)
    out = _concat_sets([[A], [B]], axis=0)
    pairs = sorted({(i, j) for i in range(24) for j in range(12)}
                   | {(i, j) for i in range(12) for j in range(24)})
    points = [np.concatenate([pa[i], pb[j]]) for i, j in pairs]
    assert_covered(out, points, "_concat_sets of two constrained Stars")


def test_concat_kernel_of_a_set_with_itself_keeps_the_shared_variables():
    """_concat_sets of a constrained Star s and an affine image M s + b of
    it: the two operands share one predicate system, so the result must hold
    (x, M x + b) for every member x, on the cut face too. Catches operands of
    one system joined on the wrong variables (the second basis written to
    other columns, the shared rows dropped or misplaced) and the operands
    stacked in the wrong order."""
    lo, hi = _tensor_box((4,), crossing=(0, 2))
    s, pts = constrained_star(lo, hi, n=48, seed=3)
    M = np.array([[1.0, -0.5, 0.0, 2.0], [0.0, 1.0, 1.0, -1.0],
                  [-2.0, 0.0, 0.5, 0.0]])
    b = np.array([0.25, -1.0, 3.0])
    image = s.affine_map(M, b.reshape(-1, 1))
    out = _concat_sets([[s], [image]], axis=0)
    assert_covered(out, [np.concatenate([p, M @ p + b]) for p in pts],
                   "_concat_sets of a Star and an affine image of it")


@pytest.mark.parametrize("kind,hwc_axis,chw_axis", [
    ("ImageStar", 0, 1), ("ImageStar", 1, 2), ("ImageStar", 2, 0),
    ("ImageZono", 0, 1), ("ImageZono", 2, 0)])
def test_concat_kernel_on_image_set_lists_holds_the_product(kind, hwc_axis,
                                                           chw_axis):
    """_concat_sets of an ImageStar / ImageZono list of three parts and one
    image of other bounds along each stored axis (height 0, width 1, channel
    2). The reference is the concatenation of the two (C, H, W) tensors along
    the matching tensor axis, compared in the stored order. Catches a stored
    basis read in the wrong order, operands in reverse order and a list
    piece dropped."""
    shape_a = (2, 3, 2)
    loA, hiA = _tensor_box(shape_a, crossing=(0, 5))
    # B has a zero-width pixel, so its predicate system differs from A's:
    # the kernel reads equal systems as the same variables (the graph walk
    # tags each node's variables to make that hold), and two boxes built
    # apart are independent.
    loB, hiB = _tensor_box(shape_a, crossing=(3,), flat=8)
    loB, hiB = loB + 4.0, hiB + 4.0
    A = image_split_cover(kind, loA, hiA, 3)
    B = [image_set(kind, loB, hiB)]
    sets = _concat_sets([A, B], axis=hwc_axis)
    lo = np.concatenate([loA, loB], axis=chw_axis)
    hi = np.concatenate([hiA, hiB], axis=chw_axis)
    assert_covered(sets, samples(lo, hi), f"_concat_sets {kind} axis "
                   f"{hwc_axis}", out_chw=lo.shape)


# --------------------------------------------------------------------------- #
# Refusals                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("with_constant", [False, True])
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_concat_of_a_hexatope_or_octatope_is_refused(tmp_path, kind,
                                                     with_constant):
    """Concat has no rule for a Hexatope or an Octatope, with a set operand or
    with a constant operand: the model entry point must raise
    NotImplementedError, not return a set."""
    shape = [1, 2, 3, 2]
    lo, hi = _tensor_box((2, 3, 2), crossing=())
    if with_constant:
        nodes = [helper.make_node("Neg", ["X"], ["B"]),
                 helper.make_node("Concat", ["B", "K"], ["Y"], axis=2)]
        inits, out = {"K": _const([1, 2, 1, 2])}, [1, 2, 4, 2]
    else:
        nodes = [helper.make_node("Neg", ["X"], ["B"]),
                 helper.make_node("Concat", ["X", "B"], ["Y"], axis=2)]
        inits, out = {}, [1, 2, 6, 2]
    path = save_onnx(tmp_path, nodes, shape, out, inits=inits)
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                    f"Concat on {kind}", input_shape=(2, 3, 2))


@pytest.mark.parametrize("cls", [Hexatope, Octatope])
def test_concat_kernel_refuses_an_unsupported_set_type(cls):
    """_concat_sets on Hexatope or Octatope operands raises
    NotImplementedError instead of returning the first operand or a stacked
    set without its constraints."""
    lo, hi = _tensor_box((3,), crossing=())
    s = cls.from_bounds(lo, hi)
    with pytest.raises(NotImplementedError):
        _concat_sets([[s], [s]], axis=0)


# --------------------------------------------------------------------------- #
# The other entry routes: reach_layer, a traced torch model, no tracked shape  #
# --------------------------------------------------------------------------- #
def _any_set(kind, lo, hi):
    return (image_set(kind, lo, hi) if kind in ("ImageStar", "ImageZono")
            else flat_set(kind, lo, hi))


@pytest.mark.parametrize("kind", ["Star", "ImageStar", "Zono", "ImageZono",
                                  "Box", "Hexatope", "Octatope"])
def test_concat_module_handed_to_reach_layer_is_refused(kind):
    """reach_layer takes one list of sets, which cannot hold the operands of a
    Concat: an OnnxConcat module handed to it must raise NotImplementedError
    for every set type. Catches a Concat module passed through (the one
    operand returned, a set with fewer entries than the output) or answered
    from that one operand."""
    lo, hi = _tensor_box((2, 3, 2), crossing=())
    with pytest.raises(NotImplementedError):
        reach_layer(OnnxConcat(axis=1), [_any_set(kind, lo, hi)], "approx")


class _TorchCat(nn.Module):
    """torch.cat([relu(x), x], dim=1): a channel concat in a torch model."""

    def forward(self, x):
        return torch.cat([torch.relu(x), x], dim=1)


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box", "ImageStar",
                                  "ImageZono"])
def test_torch_cat_in_a_traced_model_refuses_or_holds_every_output(kind):
    """torch.cat([relu(x), x], dim=1) in a torch model on a (2, 3, 2) input,
    through reach_pytorch_model (the model is traced by torch.fx). The walker
    has no rule for torch.cat, so the promise is a refusal with
    NotImplementedError, or every float64 torch output in a returned set.
    Catches a torch.cat answered without reading its dim (joined along the
    first stored axis, which for an image set is the height) and one
    answered with its first operand only."""
    lo, hi = _tensor_box((2, 3, 2), crossing=(0, 7))
    what = f"torch.cat on a {kind}"
    outputs = replay_torch(_TorchCat(), samples(lo, hi), [1, 2, 3, 2])
    try:
        sets = reach_model(_TorchCat(), _any_set(kind, lo, hi), "approx",
                           what, input_shape=(2, 3, 2))
    except NotImplementedError:
        return
    assert_covered(sets, outputs, what, out_chw=(4, 3, 2))


@pytest.mark.parametrize("kind", ["Star", "Box"])
def test_concat_of_a_flat_set_without_a_tracked_shape_refuses_or_holds(
        tmp_path, kind):
    """Concat(Relu(X), Neg(X)) along the height (axis 2) of a (1, 2, 3, 2)
    model, reached from a flat set given without input_shape: the walk's
    placeholders (1, 12) and (12,) cannot run the model, so no tensor shape
    is tracked and an inner-axis Concat cannot be placed. The promise: a
    refusal (NotImplementedError, or a ValueError that asks for the shape),
    or every onnxruntime output in a returned set. Catches an inner-axis
    Concat answered in the stacked (outermost-axis) layout when the shapes
    are unknown. (Axes -1 and -2 run on the placeholder and are answered
    wrongly today: registry SHAPE-2, marked in
    test_soundness_tensor_shape.py.)"""
    shape = [1, 2, 3, 2]
    nodes, inits, out = _concat_graph("relu_neg", shape, 2)
    lo, hi = _tensor_box((2, 3, 2), crossing=(0, 7))
    path = save_onnx(tmp_path, nodes, shape, out, inits=inits)
    outputs = replay_onnx(path, samples(lo, hi), shape)
    what = f"Concat axis 2 of a {kind} without input_shape"
    try:
        sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                           what)
    except (NotImplementedError, ValueError):
        return
    assert_covered(sets, outputs, what)


@pytest.mark.parametrize("kind,axis", [("ImageZono", -1), ("ImageZono", 2),
                                       ("ImageStar", -1)])
def test_concat_after_a_reshape_of_an_image_set_given_without_its_shape(
        tmp_path, kind, axis):
    """Conv (kernel 2 x 1), Reshape to (1, 2, 4), then Concat(Relu(R),
    Neg(R)) along an inner axis, reached from an image set given without
    input_shape. After the Reshape the walk holds flat sets. Where it tracked
    no tensor shape (an ImageZono today) an inner-axis Concat cannot be
    placed, and where it did (an ImageStar today) it can. The promise: a
    refusal (NotImplementedError, or a ValueError that asks for the shape),
    or every onnxruntime output in a returned set. Catches the fallback for
    unknown shapes answering the last axis as if every tensor were (1, n):
    the operands stacked end to end where they interleave row by row."""
    shape = [1, 2, 3, 2]
    weight = _f32(np.array([[2, -1], [-3, 1], [1, 4], [-2, 3]]).reshape(
        2, 2, 2, 1) / 4.0)
    nodes = [helper.make_node("Conv", ["X", "W"], ["C"], kernel_shape=[2, 1]),
             helper.make_node("Reshape", ["C", "shp"], ["R"]),
             helper.make_node("Relu", ["R"], ["A"]),
             helper.make_node("Neg", ["R"], ["B"]),
             helper.make_node("Concat", ["A", "B"], ["Y"], axis=axis)]
    out = _cat_shape([[1, 2, 4], [1, 2, 4]], axis % 3)
    lo, hi = _tensor_box((2, 3, 2), crossing=(0, 7))
    path = save_onnx(tmp_path, nodes, shape, out,
                     inits={"W": weight, "shp": _i64(1, 2, 4)})
    outputs = replay_onnx(path, samples(lo, hi), shape)
    what = f"Concat axis {axis} after a Reshape, {kind} without input_shape"
    try:
        sets = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                           what)
    except (NotImplementedError, ValueError):
        return
    assert_covered(sets, outputs, what)


# --------------------------------------------------------------------------- #
# An image-set operand next to an operand the walk holds as a flat set         #
# --------------------------------------------------------------------------- #
_MIXED_LAYOUT = known_bug(
    7, "GRAPH-22: ONNX Concat of an operand the walk holds flat (Flatten, "
       "Gemm, Reshape back to the image shape) followed by an image-set "
       "operand stacks the image operand's entries in their stored (H, W, C) "
       "order where the row map reads the tensor's order: every true output "
       "is outside the flat result")
_MIXED_CRASH = {
    "ImageStar": known_bug(
        7, "GRAPH-22: ONNX Concat of an ImageStar operand, then an operand "
           "the walk holds as a flat Star crashes with IndexError (the "
           "ImageStar branch of _concat_sets reshapes the flat basis as an "
           "image) instead of answering or refusing", raises=IndexError),
    "ImageZono": known_bug(
        7, "GRAPH-22: ONNX Concat of an ImageZono operand, then an operand "
           "the walk holds as a flat Zono crashes with AttributeError (the "
           "ImageZono branch of _concat_sets reads the flat Zono's height) "
           "instead of answering or refusing", raises=AttributeError)}


def _mixed_params():
    out = []
    for kind in ("ImageStar", "ImageZono"):
        out.append(pytest.param(kind, "gemm_reshape", "other_first",
                                marks=_MIXED_LAYOUT,
                                id=f"{kind}-gemm_reshape-other_first"))
        out.append(pytest.param(kind, "gemm_reshape", "image_first",
                                marks=_MIXED_CRASH[kind],
                                id=f"{kind}-gemm_reshape-image_first"))
        # the passing neighbour: the other operand is an image set too
        out.append(pytest.param(kind, "conv", "other_first",
                                id=f"{kind}-conv-other_first"))
    return out


@pytest.mark.parametrize("kind,branch,order", _mixed_params())
def test_concat_of_an_image_operand_and_a_reshaped_operand_holds_every_output(
        tmp_path, kind, branch, order):
    """Concat along the channels of A = Relu(X) and an operand G computed
    from the whole input, on a (2, 3, 2) ImageStar or ImageZono, approx: G =
    Reshape(Gemm(Flatten(X))) back to the image shape (``gemm_reshape``: the
    walk keeps a Reshape that feeds no spatial layer flat, so G is a flat
    set) or G = Conv 1x1 (X) (``conv``, an image set: the passing
    neighbour), G before or after A. The promise: the call refuses with
    NotImplementedError, or every onnxruntime output is in a returned set.
    Catches a Concat that joins operands of two set families without
    bringing them to one layout (the image operand's (H, W, C) entries read
    in the tensor's order, or a flat basis reshaped as an image)."""
    shape = [1, 2, 3, 2]
    lo, hi = _tensor_box((2, 3, 2), crossing=(0, 7))
    rng = np.random.default_rng(3)
    nodes = [helper.make_node("Relu", ["X"], ["A"])]
    if branch == "gemm_reshape":
        nodes += [helper.make_node("Flatten", ["X"], ["F"], axis=1),
                  helper.make_node("Gemm", ["F", "W"], ["H"], transB=1),
                  helper.make_node("Reshape", ["H", "shape"], ["G"])]
        inits = {"W": _f32(rng.integers(-2, 3, size=(12, 12)) / 2.0),
                 "shape": _i64(*shape)}
    else:
        nodes.append(helper.make_node("Conv", ["X", "Wc"], ["G"]))
        inits = {"Wc": _f32(rng.integers(-2, 3, size=(2, 2, 1, 1)) / 2.0)}
    operands = ["G", "A"] if order == "other_first" else ["A", "G"]
    nodes.append(helper.make_node("Concat", operands, ["Y"], axis=1))
    path = save_onnx(tmp_path, nodes, shape, [1, 4, 3, 2], inits=inits)
    outputs = replay_onnx(path, samples(lo, hi), shape)
    what = f"Concat({', '.join(operands)}) with G from {branch}, {kind}"
    try:
        sets = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                           what, input_shape=(2, 3, 2))
    except NotImplementedError:
        return
    assert_covered(sets, outputs, what, out_chw=(4, 3, 2))


# --------------------------------------------------------------------------- #
# The exact method on operands with different predicate systems               #
# --------------------------------------------------------------------------- #
@known_bug(18, "REACH-4: method='exact' joins Concat operands with different "
               "predicate systems block-diagonally (the input's variables "
               "copied for the second operand): the result holds every true "
               "output but is not the exact image, and is returned as exact")
def test_exact_concat_of_two_predicate_systems_is_exact_or_refused(tmp_path):
    """Concat(Relu(X), X), method 'exact', on a (1, 2) input whose entry 0
    straddles zero: each ReLU piece adds a row to the input's predicate
    system and X keeps the input's system, so the two operands have
    different systems. The promise of 'exact': the call refuses with
    NotImplementedError, or the result is the exact image, checked in
    rationals both ways (an escape fails the test outright; the marker
    covers only the second direction). Catches the block-diagonal join (sound,
    not exact) returned under the name 'exact': a returned set then holds a
    point (relu(x), x') with x' != x that no input reaches. The passing
    neighbour with one predicate system is
    test_exact_concat_of_one_lineage_is_the_exact_image."""
    lo, hi = np.array([-0.5, 0.25]), np.array([0.25, 0.75])
    nodes = [helper.make_node("Relu", ["X"], ["R"]),
             helper.make_node("Concat", ["R", "X"], ["Y"], axis=1)]
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 4])
    what = "exact Concat(Relu(X), X)"
    try:
        sets = reach_model(load_onnx(path), flat_set("Star", lo, hi), "exact",
                           what, input_shape=(2,))
    except NotImplementedError:
        return
    # (relu(x), x) = (relu(x), relu(x) - relu(-x)): the ReLU of the lifted
    # input (x, -x), then a linear map.
    lift = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    back = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, -1, 0], [0, 1, 0, -1]]
    pieces = Q.affine(Q.relu(Q.affine(Q.box(lo, hi), lift)), back)
    computed = [Q.star(s) for s in sets]
    if Q.straddlers(pieces, *computed):
        pytest.fail(f"{what}: an exact piece has a vertex outside, or is "
                    f"spread over several returned sets")
    for k, c in enumerate(computed):
        assert Q.unreached(pieces, _vertices_of(c)) == [], \
            f"{what}: returned set {k} has a vertex that no input reaches"
