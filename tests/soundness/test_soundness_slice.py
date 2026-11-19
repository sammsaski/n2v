"""Soundness cells for ONNX Slice, the kernel ``_slice_set`` and tensor
indexing (``n2v/nn/reach.py``: ``_handle_onnx_slice``, ``_select_rows``,
``_slice_set``).

How a Slice is answered
-----------------------
Only the graph walker answers a Slice. ``_handle_onnx_slice`` takes one of two
paths. A flat Star, Zono or Box whose tensor shape is tracked takes the
shape-aware path: the ONNX starts, ends, axes and steps (or the attribute
slices of the opset-9 module ``OnnxSliceV9``) index an index tensor of the
tracked shape, and ``_select_rows`` keeps those rows. An ImageStar or
ImageZono (and a flat set whose shape is not tracked) takes the legacy path:
the batch axis is removed, the ONNX channel, height and width axes are mapped
to the stored height, width and channel axes, and ``_slice_set`` cuts the
stored arrays. ``reach_layer`` handed a Slice module, and indexing a tensor
(``x[:, 1:3]``) in a torch model, are refused with ``NotImplementedError``; a
Hexatope or an Octatope is refused on both paths.

What a cell checks
------------------
Every true output of the network on a member point of the input set is a
member of a returned set, and each returned set has the entry count of the
true output. True outputs come from onnxruntime on the same ONNX file (dyadic
data, so float32 evaluation is exact); for a direct kernel call they are the
numpy slice of member points computed from the arrays the test built.
Membership is the LP of ``tests/_oracles/membership`` on the rows each
returned set stores (``cell_support.assert_covered``), slack ``LP_TOL``. The
exact method is checked both ways in rationals (``tests/_oracles/rational``).
Layout: a tensor is (C, H, W) row-major, a flat set keeps the tensor's
row-major order, an image set stores (H, W, C) and is compared through
``out_chw``. Slice does no arithmetic on values, so it has no rounding case.

Inputs: a (2, 3, 5) or (3, 4, 2) tensor with its own dyadic interval per
entry; the first and last selected entries straddle zero (so an exact ReLU in
front splits into pieces that differ on the selected entries) and the middle
selected entry has zero width; a Star or ImageStar cut by a row that couples
a selected entry with one the slice removes; several sets at once after an
exact ReLU.

Defects marked here (registry ``.claude/research/34-API-Redesign/
test-audit/DEFECTS.md``): GRAPH-20, a negative-step Slice starting before
the first entry; GRAPH-21, a Slice whose ``axes`` input is omitted by an empty
name while ``steps`` is given. Negative axes on image sets (GRAPH-1), the
opset-9 Slice with a negative axis (GRAPH-7) and the batch axis -4 belong to
``test_soundness_negative_axis.py``; a flat set without ``input_shape``
(SHAPE-1, SHAPE-2) to ``test_soundness_tensor_shape.py``.
"""

import numpy as np
import pytest
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper
from onnx2torch.node_converters.slice import OnnxSlice, OnnxSliceV9

from n2v.nn.reach import _slice_set
from n2v.sets import Box, Hexatope, Octatope, Zono
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as Q
from tests.cell_support import (
    assert_covered, constrained_image_star, constrained_star, coupled_star,
    flat_set, hwc_index, image_set, reach_model, reach_one_layer, replay_onnx,
    samples, save_onnx)
from tests.known_bugs import known_bug

INT_MAX = 2 ** 63 - 1
INT_MIN = -2 ** 63


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


def _selected(shape, slices):
    """Row-major indices of the entries of a tensor of ``shape`` that the
    Python ``slices`` select, in the order the slice returns them."""
    return np.arange(int(np.prod(shape))).reshape(shape)[slices].ravel()


def _selection_box(shape, slices):
    """``_tensor_box`` for an input tensor of ``shape`` (batch axis first)
    with its regimes inside the selection: the first and the last selected
    entry straddle zero and the middle selected entry has zero width. An
    exact ReLU in front of the Slice then splits into pieces that differ on
    the selected entries, so a Slice that handles only some of the pieces
    misses outputs."""
    kept = _selected(shape, slices)
    crossing = tuple(sorted({int(kept[0]), int(kept[-1])}))
    middle = int(kept[len(kept) // 2])
    return _tensor_box(tuple(shape[1:]), crossing=crossing,
                       flat=None if middle in crossing else middle)


def _grid(cells):
    """Parameters ``(kind, method, *option)`` from ``{(kind, method): [option,
    ...]}``: the handler code shared by all kinds (axis normalisation, row
    selection) is swept on the cheap Box and a Star approx, the
    kind-specific kernels on fewer options, to keep the file fast."""
    return [(k, m) + (o if isinstance(o, tuple) else (o,))
            for (k, m), options in cells.items() for o in options]


def _input_set(kind, lo, hi, coupled=None):
    if coupled is not None:
        return coupled_star(kind, lo, hi, coupled)
    if kind in ("ImageStar", "ImageZono"):
        return image_set(kind, lo, hi), samples(lo, hi)
    return flat_set(kind, lo, hi), samples(lo, hi)


def _check_model(tmp_path, nodes, inits, in_shape, out_shape, kind, method,
                 what, opset=13, box=None, coupled=None):
    """Reach the ONNX graph from the input set of ``kind`` and assert that
    every onnxruntime output of a member point is in a returned set, compared
    in the layout each returned set stores. ``coupled`` (the selected
    entries) asks for the Star or ImageStar of ``coupled_star``."""
    lo, hi = box if box is not None else _tensor_box(tuple(in_shape[1:]))
    path = save_onnx(tmp_path, nodes, in_shape, out_shape, inits=inits,
                     opset=opset)
    input_set, points = _input_set(kind, lo, hi, coupled)
    outputs = replay_onnx(path, points, in_shape)
    sets = reach_model(load_onnx(path), input_set, method, what,
                       input_shape=tuple(in_shape[1:]))
    out_chw = tuple(out_shape[1:]) if len(out_shape) == 4 else None
    assert_covered(sets, outputs, what, out_chw=out_chw)
    return sets


# Slices as ONNX gives them: (starts, ends, axes or None, steps or None).
_CASES = {
    "w_middle": ([1], [4], [3], None),              # not a prefix, last axis
    "h_middle": ([1], [3], [2], None),
    "c_second": ([1], [2], [1], None),
    "w_step2": ([0], [INT_MAX], [3], [2]),
    "w_reversed": ([-1], [INT_MIN], [3], [-1]),
    "h_reversed_part": ([2], [0], [2], [-1]),
    "w_back_step2": ([4], [0], [3], [-2]),
    "two_axes_descending": ([1, 0], [3, 2], [3, 2], None),
    "three_axes": ([2, 1, 0], [INT_MIN, 2, INT_MAX], [2, 1, 3], [-1, 1, 2]),
    "default_axes": ([0, 1, 0, 1], [INT_MAX, 2, 3, 4], None, None),
}


def _onnx_slices(shape, starts, ends, axes, steps):
    """Python slices per axis with ONNX's meaning (numpy clamps the large
    end markers the same way ONNX does)."""
    axes = list(range(len(starts))) if axes is None else axes
    steps = [1] * len(starts) if steps is None else steps
    full = [slice(None)] * len(shape)
    for st, en, ax, sp in zip(starts, ends, axes, steps):
        full[ax % len(shape)] = slice(st, en, sp)
    return tuple(full)


def _slice_graph(shape, case, before="Relu"):
    """``(nodes, inits, out_shape)`` of ``before(X) -> Slice -> Neg``; the
    output shape is the numpy slice of an array of ``shape``."""
    starts, ends, axes, steps = case
    names, inits = ["A", "st", "en"], {"st": _i64(*starts), "en": _i64(*ends)}
    if axes is not None:
        names.append("ax")
        inits["ax"] = _i64(*axes)
    if steps is not None:
        names.append("sp")
        inits["sp"] = _i64(*steps)
    out = list(np.empty(shape)[_onnx_slices(shape, starts, ends, axes,
                                            steps)].shape)
    nodes = [helper.make_node(before, ["X"], ["A"]),
             helper.make_node("Slice", names, ["S"]),
             helper.make_node("Neg", ["S"], ["Y"])]
    return nodes, inits, out


_SHAPE = [1, 2, 3, 5]
_ALL = list(_CASES)


@pytest.mark.parametrize("kind,method,case", _grid({
    ("Box", "approx"): _ALL, ("Star", "approx"): _ALL,
    ("Star", "exact"): ["w_reversed", "two_axes_descending", "three_axes"],
    ("Zono", "approx"): ["h_middle", "w_back_step2", "default_axes"]}))
def test_slice_of_a_flat_set_holds_every_output(tmp_path, kind, method, case):
    """Relu, Slice, Neg on a flat set standing for a (1, 2, 3, 5) tensor: a
    slice that is not a prefix, steps 2, -1 and -2, axes listed in
    descending order, three axes at once, and no axes input. Catches the
    slice applied to the axis before the requested one, the selected rows in
    column-major or sorted order (a reversed axis kept in its original
    order), axes sorted without their starts and ends, and an exact Relu list
    of which only some pieces are sliced (the first and last selected
    entries straddle zero, so the pieces differ on the selection)."""
    nodes, inits, out = _slice_graph(_SHAPE, _CASES[case])
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Slice {case}, {kind} {method}",
                 box=_selection_box(_SHAPE, _onnx_slices(_SHAPE,
                                                         *_CASES[case])))


@pytest.mark.parametrize("kind,method,shape,case", _grid({
    ("ImageStar", "approx"): [((1, 2, 3, 5), c) for c in _ALL]
    + [((1, 3, 4, 2), c) for c in ("h_middle", "c_second",
                                   "h_reversed_part")],
    ("ImageStar", "exact"): [((1, 2, 3, 5), "three_axes"),
                             ((1, 3, 4, 2), "h_middle")],
    ("ImageZono", "approx"): [((1, 2, 3, 5), c) for c in
                              ("w_middle", "h_middle", "c_second",
                               "w_back_step2", "three_axes",
                               "default_axes")]}))
def test_slice_of_an_image_set_holds_every_output(tmp_path, kind, method,
                                                  shape, case):
    """Relu, Slice, Neg on an ImageStar or ImageZono of a (2, 3, 5) and a
    (3, 4, 2) image, through the handler's image path: channel, height and
    width slices, steps, a reversed axis, three axes at once, and no axes
    input (axis 0 then takes the whole batch). Catches the ONNX height and
    width axes mapped to the wrong stored axis, a slice of the stored basis
    whose predicate bounds are swapped, an ImageZono generator tensor
    flattened with the generators as the leading axis, the "to the end"
    marker taken as an index, a step read without its sign, default axes
    that skip the batch axis, and an exact Relu list of which only the first
    piece is sliced (the first and last selected pixels straddle zero)."""
    shape = list(shape)
    nodes, inits, out = _slice_graph(shape, _CASES[case])
    _check_model(tmp_path, nodes, inits, shape, out, kind, method,
                 f"Slice {case} on {shape}, {kind} {method}",
                 box=_selection_box(shape, _onnx_slices(shape,
                                                        *_CASES[case])))


# Negative axes on flat sets with a known shape: the tensor (1, 2, 3, 2) has
# two axes of length 2 (C and W), the tensor (1, 3, 3) two of length 3, so a
# negative axis normalised to the wrong axis gives a set of the right size
# whose entries are the wrong ones.
_NEG_CASES = {
    "rank4_axis_m1": ([1, 2, 3, 2], ([1], [2], [-1], None)),
    "rank4_axis_m3": ([1, 2, 3, 2], ([0], [1], [-3], None)),
    "rank4_axis_m2_rev": ([1, 2, 3, 2], ([-1], [INT_MIN], [-2], [-1])),
    "rank3_axis_m1": ([1, 3, 3], ([0], [2], [-1], None)),
    "rank3_axis_m2": ([1, 3, 3], ([2], [3], [-2], None)),
    "rank4_descending_equal": ([1, 2, 3, 2], ([1, 0], [2, 1], [3, 1], None)),
}


@pytest.mark.parametrize("kind,method,case", _grid({
    ("Box", "approx"): list(_NEG_CASES), ("Star", "approx"): list(_NEG_CASES),
    ("Zono", "approx"): ["rank4_axis_m1", "rank3_axis_m2"],
    ("Star", "exact"): ["rank4_axis_m3"]}))
def test_slice_of_a_flat_set_on_a_negative_or_unsorted_axis(tmp_path, kind,
                                                            method, case):
    """Slice with negative axes, and with two axes of equal length listed in
    descending order, on flat sets whose tensor shape is known. Catches a
    negative axis normalised with abs() instead of modulo the rank (axis -1
    read as axis 1), and axes sorted without their starts and ends: both
    select a set of the right size with the wrong entries."""
    shape, spec = _NEG_CASES[case]
    nodes, inits, out = _slice_graph(shape, spec)
    _check_model(tmp_path, nodes, inits, shape, out, kind, method,
                 f"Slice {case}, {kind} {method}",
                 box=_selection_box(shape, _onnx_slices(shape, *spec)))


@pytest.mark.parametrize("kind,method,case", _grid({
    ("Box", "approx"): ["w_middle", "h_middle", "two_axes"],
    ("Star", "approx"): ["w_middle", "two_axes"],
    ("ImageStar", "approx"): ["w_middle", "h_middle", "c_second",
                              "two_axes"],
    ("ImageZono", "approx"): ["h_middle", "two_axes"]}))
def test_slice_of_opset_9_holds_every_output(tmp_path, kind, method, case):
    """The opset-9 Slice (starts, ends and axes as attributes, no steps),
    which the library reads from another module type in both the
    shape-aware path (flat sets) and the image path. Catches the attribute
    slices placed on the wrong axis (the batch shift applied in the
    shape-aware path, or not applied in the image path) and a Star or
    ImageStar that keeps only the variables of the selected entries."""
    spec = {"w_middle": ([1], [4], [3]), "h_middle": ([1], [3], [2]),
            "c_second": ([1], [2], [1]),
            "two_axes": ([0, 2], [2, 5], [2, 3])}[case]
    starts, ends, axes = spec
    slices = _onnx_slices(_SHAPE, starts, ends, axes, None)
    out = list(np.empty(_SHAPE)[slices].shape)
    nodes = [helper.make_node("Relu", ["X"], ["A"]),
             helper.make_node("Slice", ["A"], ["S"], starts=starts,
                              ends=ends, axes=axes),
             helper.make_node("Neg", ["S"], ["Y"])]
    _check_model(tmp_path, nodes, {}, _SHAPE, out, kind, method,
                 f"opset-9 Slice {case}, {kind} {method}", opset=9,
                 box=_selection_box(_SHAPE, slices))


@pytest.mark.parametrize("opset", [10])
@pytest.mark.parametrize("kind", ["Box", "ImageStar"])
def test_slice_at_other_opsets_holds_every_output(tmp_path, kind, opset):
    """The input form of Slice at opset 10, the first opset that gives
    starts, ends, axes and steps as inputs (onnx2torch converts opsets 10 to
    18 to the same module, so 10 stands for 11 to 18 and the other cells use
    13), three axes with a reversed one: the same row selection must come
    out as at opset 13."""
    nodes, inits, out = _slice_graph(_SHAPE, _CASES["three_axes"])
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, "approx",
                 f"Slice opset {opset}, {kind}", opset=opset,
                 box=_selection_box(_SHAPE, _onnx_slices(
                     _SHAPE, *_CASES["three_axes"])))


@pytest.mark.parametrize("kind,method,case", _grid({
    ("Star", "approx"): ["w_middle", "three_axes"],
    ("Star", "exact"): ["w_back_step2"],
    ("ImageStar", "approx"): ["h_middle", "three_axes"],
    ("ImageStar", "exact"): ["w_reversed"]}))
def test_slice_of_a_star_with_a_binding_row_holds_every_output(
        tmp_path, kind, method, case):
    """Relu, Slice, Neg on a Star / ImageStar cut by two rows that are not
    box rows (``coupled_star`` of the cell support): one couples a selected
    entry with an entry the slice removes, one couples two selected entries.
    Catches a slice that drops the predicate variables of the removed entries
    together with their columns of the rows (which fixes them at zero, so the
    selected entry loses half its range), and rows or bounds moved onto other
    variables; a box check passes both, the member points on the faces of
    the rows do not. A sliced set that loses rows is larger, not wrong, and
    is not what this cell is for."""
    nodes, inits, out = _slice_graph(_SHAPE, _CASES[case])
    slices = _onnx_slices(_SHAPE, *_CASES[case])
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Slice {case} on a constrained {kind}, {method}",
                 box=_selection_box(_SHAPE, slices),
                 coupled=_selected(_SHAPE, slices))


def _sparse_rows(rng, n_out, n_in):
    W = np.zeros((n_out, n_in))
    for r in range(n_out):
        cols = rng.choice(n_in, size=min(4, n_in), replace=False)
        W[r, cols] = rng.choice([-1.0, -0.5, 0.5, 1.0, 2.0], size=cols.size)
    return W


@pytest.mark.parametrize("kind,method,case", _grid({
    ("Box", "approx"): ["w_back_step2", "two_axes_descending"],
    ("Star", "approx"): ["w_back_step2", "two_axes_descending"],
    ("Star", "exact"): ["three_axes"], ("Zono", "approx"): ["w_middle"]}))
def test_slice_between_two_affine_layers_holds_every_output(
        tmp_path, kind, method, case):
    """Gemm (mixes all inputs), Reshape to (1, 2, 3, 5), Relu, Slice, Flatten,
    Gemm on a flat set: the rows the Slice emits are consumed by a Gemm whose
    rows single out entries, so a slice with the right entries in the wrong
    order changes the outputs. The bias puts two pre-activations across zero
    (interval arithmetic on the dyadic box)."""
    rng = np.random.default_rng(13)
    lo, hi = _tensor_box((6,), crossing=())
    W1 = rng.integers(-2, 3, size=(30, 6)) / 2.0
    mid, rad = (W1 @ (lo + hi)) / 2.0, np.abs(W1) @ ((hi - lo) / 2.0)
    target = np.where(np.arange(30) % 3 == 0, 1.0, -1.0) * (rad + 0.25)
    target[[2, 17]] = rad[[2, 17]] / 4.0        # these two straddle zero
    b1 = target - mid
    starts, ends, axes, steps = _CASES[case]
    n_out = int(np.prod(np.empty(_SHAPE)[_onnx_slices(
        _SHAPE, starts, ends, axes, steps)].shape))
    W2 = _sparse_rows(rng, 5, n_out)
    names, inits = ["A", "st", "en"], {"st": _i64(*starts),
                                       "en": _i64(*ends)}
    for name, values in (("ax", axes), ("sp", steps)):
        if values is not None:
            names.append(name)
            inits[name] = _i64(*values)
    inits.update({"W1": _f32(W1), "b1": _f32(b1), "shape": _i64(*_SHAPE),
                  "W2": _f32(W2)})
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["Z"], transB=1),
             helper.make_node("Reshape", ["Z", "shape"], ["T"]),
             helper.make_node("Relu", ["T"], ["A"]),
             helper.make_node("Slice", names, ["S"]),
             helper.make_node("Flatten", ["S"], ["F"], axis=1),
             helper.make_node("Gemm", ["F", "W2"], ["Y"], transB=1)]
    _check_model(tmp_path, nodes, inits, [1, 6], [1, 5], kind, method,
                 f"Gemm > Slice {case} > Gemm, {kind} {method}", box=(lo, hi))


@pytest.mark.parametrize("kind,method,case", _grid({
    ("ImageStar", "approx"): ["h_middle", "w_back_step2", "three_axes"],
    ("ImageStar", "exact"): ["c_second"],
    ("ImageZono", "approx"): ["h_reversed_part", "w_middle"]}))
def test_slice_of_images_between_conv_and_gemm_holds_every_output(
        tmp_path, kind, method, case):
    """Conv 1x1 (mixes the two channels), Relu, Slice, Flatten, Gemm on a
    (2, 3, 5) image set: the sliced image is read by Flatten, which converts
    the stored (H, W, C) order to the tensor's order, so a slice applied to
    the wrong stored axis changes the outputs of the Gemm."""
    rng = np.random.default_rng(7)
    starts, ends, axes, steps = _CASES[case]
    n_out = int(np.prod(np.empty(_SHAPE)[_onnx_slices(
        _SHAPE, starts, ends, axes, steps)].shape))
    names, inits = ["A", "st", "en"], {"st": _i64(*starts),
                                       "en": _i64(*ends)}
    for name, values in (("ax", axes), ("sp", steps)):
        if values is not None:
            names.append(name)
            inits[name] = _i64(*values)
    inits.update({"Wc": _f32([[[[1.0]], [[0.5]]], [[[-0.5]], [[1.0]]]]),
                  "bc": _f32([0.25, -0.5]),
                  "W2": _f32(_sparse_rows(rng, 5, n_out))})
    nodes = [helper.make_node("Conv", ["X", "Wc", "bc"], ["T"]),
             helper.make_node("Relu", ["T"], ["A"]),
             helper.make_node("Slice", names, ["S"]),
             helper.make_node("Flatten", ["S"], ["F"], axis=1),
             helper.make_node("Gemm", ["F", "W2"], ["Y"], transB=1)]
    _check_model(tmp_path, nodes, inits, _SHAPE, [1, 5], kind, method,
                 f"Conv > Slice {case} > Gemm, {kind} {method}",
                 box=_tensor_box((2, 3, 5), crossing=(0, 22)))


# --------------------------------------------------------------------------- #
# The exact method: the exact image, both ways                                 #
# --------------------------------------------------------------------------- #
def _vertices_of(computed):
    """The images of the vertices of a computed set's predicate polytope
    (``rational.set_vertices``), skipping its boundedness test when every
    predicate variable has two finite bounds and dropping repeated rows."""
    c, G = computed[0], computed[1]
    if any(v is None for v in computed[4] + computed[5]) or not computed[4]:
        return Q.set_vertices(computed)
    A, b = Q._predicate_rows(computed)
    unique = sorted({(tuple(row), rhs) for row, rhs in zip(A, b)})
    corners = Q.vertices([list(r) for r, _ in unique], [r for _, r in unique])
    return sorted({tuple(Q.dot(row, a) + ci for row, ci in zip(G, c))
                   for a in corners})


@pytest.mark.parametrize("kind,case", [
    ("Star", ([-1], [INT_MIN], [3], [-1])), ("Star", ([1], [2], [1], None)),
    ("ImageStar", ([-1], [INT_MIN], [3], [-1])),
    ("ImageStar", ([1, 0], [INT_MIN, 2], [1, 3], [-1, 1]))])
def test_exact_slice_is_the_exact_image(tmp_path, kind, case):
    """Relu then Slice on a (1, 2, 1, 2) tensor whose entries 0 and 3 straddle
    zero, method 'exact', checked in rationals both ways: every vertex of
    every exact piece is in one returned set, and every vertex of every
    returned set is the image of an input point (a reversed axis kept in its
    order, or entries taken from the wrong pixel, give vertices with no
    preimage)."""
    shape = [1, 2, 1, 2]
    lo, hi = _tensor_box(tuple(shape[1:]), crossing=(0, 3))
    starts, ends, axes, steps = case
    names, inits = ["R", "st", "en", "ax"], {"st": _i64(*starts),
                                             "en": _i64(*ends),
                                             "ax": _i64(*axes)}
    if steps is not None:
        names.append("sp")
        inits["sp"] = _i64(*steps)
    slices = _onnx_slices(shape, starts, ends, axes, steps)
    out = list(np.empty(shape)[slices].shape)
    nodes = [helper.make_node("Relu", ["X"], ["R"]),
             helper.make_node("Slice", names, ["Y"])]
    sets = _check_model(tmp_path, nodes, inits, shape, out, kind, "exact",
                        f"exact Slice {case}, {kind}", box=(lo, hi))
    M = np.stack([np.eye(4)[j].reshape(shape)[slices].reshape(-1)
                  for j in range(4)], axis=1)
    if kind == "ImageStar":
        M = M[hwc_index(*out[1:])]            # the order the set stores
    pieces = Q.affine(Q.relu(Q.box(lo.reshape(-1), hi.reshape(-1))), M)
    computed = [Q.star(s) for s in sets]
    assert Q.straddlers(pieces, *computed) == [], \
        "an exact piece has a vertex outside, or is spread over several sets"
    for k, c in enumerate(computed):
        assert Q.unreached(pieces, _vertices_of(c)) == [], \
            f"returned set {k} has a vertex that no input point reaches"


# --------------------------------------------------------------------------- #
# The kernel _slice_set                                                        #
# --------------------------------------------------------------------------- #
_FLAT_SLICES = [slice(1, 5, 2), slice(4, 0, -1)]


def _mixed_zono(n, k, seed):
    """A Zono with centre and generators that mix its entries (dyadic), and
    member points c + G e for e on the {-1, -1/2, 0, 1/2, 1} grid corners
    and faces, computed from the arrays."""
    rng = np.random.default_rng(seed)
    c = rng.integers(-8, 9, size=(n, 1)) / 4.0
    G = rng.integers(-4, 5, size=(n, k)) / 4.0
    G[np.arange(min(n, k)), np.arange(min(n, k))] += 1.0
    es = [rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=k) for _ in range(64)]
    es += [rng.choice([-1.0, 1.0], size=k) for _ in range(64)]
    return Zono(c, G), [(c + G @ e[:, None]).ravel() for e in es]


@pytest.mark.parametrize("sl", _FLAT_SLICES)
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_slice_kernel_on_a_flat_set_holds_every_sliced_point(kind, sl):
    """_slice_set on a flat Star cut by a non-box row, a Zono whose generators
    mix its entries (no generator is a multiple of the centre) and a Box,
    with a forward slice of step 2 and a reversed slice (each branch indexes
    its arrays with the slice as given): every member point, sliced, is in
    the result. Catches the generators sliced out of the centre, predicate
    bounds replaced by one bound, and upper bounds taken from the lower
    bounds. The model entry reaches these branches only for a flat set whose
    tensor shape is not tracked, so this direct call is their test."""
    lo, hi = _tensor_box((6,), crossing=(0, 3))
    if kind == "Star":
        s, points = constrained_star(lo, hi, n=96, seed=4)
    elif kind == "Zono":
        s, points = _mixed_zono(6, 4, seed=5)
    else:
        s, points = Box(lo, hi), samples(lo, hi)
    out = _slice_set(s, {0: sl})
    assert_covered([out], [np.asarray(p)[sl] for p in points],
                   f"_slice_set {kind} {sl}")


_IMAGE_SLICES = [{0: slice(1, 3)}, {1: slice(4, None, -2)}, {2: slice(1, 2)},
                 {0: slice(None, None, -1), 1: slice(1, 4), 2: slice(0, 1)}]


@pytest.mark.parametrize("which", range(len(_IMAGE_SLICES)))
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_slice_kernel_on_an_image_set_holds_every_sliced_point(kind, which):
    """_slice_set on a (2, 3, 5) ImageStar cut by a non-box row and on an
    ImageZono whose generators mix its pixels, along the stored height,
    width and channel axes and all three at once: every member point,
    sliced along the matching tensor axes, is in the result, compared in the
    stored order. Catches the stored axes confused, predicate bounds
    swapped, and an ImageZono generator tensor reshaped in the wrong
    order."""
    hwc_slices = _IMAGE_SLICES[which]
    lo, hi = _tensor_box((2, 3, 5), crossing=(0, 7))
    if kind == "ImageStar":
        s, points = constrained_image_star(lo, hi, n=96, seed=6)
    else:
        z, flat_points = _mixed_zono(30, 8, seed=7)
        s = ImageZono(z.c, z.V, 3, 5, 2)          # stored (H, W, C) = 3x5x2
        order = hwc_index(2, 3, 5)
        points = []
        for p in flat_points:                     # stored order -> tensor
            chw = np.empty(30)
            chw[order] = p
            points.append(chw.reshape(2, 3, 5))
    chw = [slice(None)] * 3
    for hwc_axis, sl in hwc_slices.items():
        chw[{0: 1, 1: 2, 2: 0}[hwc_axis]] = sl
    out = _slice_set(s, hwc_slices)
    sliced = [np.asarray(p)[tuple(chw)] for p in points]
    assert_covered([out], sliced, f"_slice_set {kind} {hwc_slices}",
                   out_chw=sliced[0].shape)


# --------------------------------------------------------------------------- #
# Refusals                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_slice_of_a_hexatope_or_octatope_is_refused(tmp_path, kind):
    """Slice has no rule for a Hexatope or an Octatope: the model entry point
    must raise NotImplementedError, not return a set."""
    nodes, inits, out = _slice_graph(_SHAPE, _CASES["w_middle"],
                                     before="Neg")
    path = save_onnx(tmp_path, nodes, _SHAPE, out, inits=inits)
    lo, hi = _tensor_box((2, 3, 5), crossing=())
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                    f"Slice on {kind}", input_shape=(2, 3, 5))


@pytest.mark.parametrize("cls", [Hexatope, Octatope])
def test_slice_kernel_refuses_an_unsupported_set_type(cls):
    """_slice_set on a Hexatope or Octatope raises NotImplementedError instead
    of returning the input unsliced."""
    lo, hi = _tensor_box((4,), crossing=())
    with pytest.raises(NotImplementedError):
        _slice_set(cls.from_bounds(lo, hi), {0: slice(1, 3)})


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_slice_of_the_batch_axis_of_an_image_set_refuses_or_holds_every_output(
        tmp_path, kind):
    """A Slice [0:1] on axis 0 of an image set: with no tracked shape the
    handler cannot tell the batch axis from a data axis. The promise is a
    refusal (NotImplementedError) or a result that holds every output, never
    an answer that slices a data axis instead."""
    nodes, inits, out = _slice_graph(_SHAPE, ([0, 1], [1, 3], [0, 2], None))
    try:
        _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, "approx",
                     f"Slice of the batch axis, {kind}")
    except NotImplementedError:
        pass


def _any_set(kind, lo, hi):
    """The box [lo, hi] of a (C, H, W) tensor as a set of ``kind``: an image
    set for ImageStar and ImageZono, a flat set (tensor order) otherwise."""
    if kind in ("ImageStar", "ImageZono"):
        return image_set(kind, lo, hi)
    return flat_set(kind, lo, hi)


_ALL_KINDS = ["Star", "Zono", "Box", "Hexatope", "Octatope", "ImageStar",
              "ImageZono"]


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_slice_module_handed_to_reach_layer_is_refused(kind):
    """``reach_layer`` has no rule for a Slice: its starts, ends, axes and
    steps live in the graph (``OnnxSlice``) or in attributes the dispatcher
    does not read (``OnnxSliceV9``). Both modules, under both methods, must
    raise NotImplementedError for every set type. Catches a Slice module
    answered as a shape no-op (the identity), whose result keeps every input
    entry and so holds none of the sliced outputs."""
    lo, hi = _tensor_box((2, 3, 5), crossing=(0, 7))
    v9 = OnnxSliceV9(starts=np.array([1]), ends=np.array([3]),
                     axes=np.array([3]))
    for module in (OnnxSlice(), v9):
        for method in ("approx", "exact"):
            with pytest.raises(NotImplementedError):
                reach_one_layer(module, _any_set(kind, lo, hi), method,
                                f"{type(module).__name__} on {kind}")


class _IndexLastAxis(nn.Module):
    """``x[..., 1:3]``: tensor indexing of the model input, which torch.fx
    traces to an ``operator.getitem`` node with a tuple of slices."""

    def forward(self, x):
        return x[..., 1:3]


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_tensor_indexing_in_a_torch_model_is_refused(kind):
    """Indexing a tensor that holds sets (``x[..., 1:3]``) in a torch model
    has no reach rule; the walker must raise NotImplementedError for every
    set type (a flat set of a (1, 30) input, an image set of a (1, 2, 3, 5)
    one). Catches a getitem on a set-valued node answered as the identity
    (read as picking the one batch element), whose result keeps every entry
    and so holds none of the indexed outputs."""
    lo, hi = _tensor_box((2, 3, 5), crossing=(0, 7))
    image = kind in ("ImageStar", "ImageZono")
    if image:
        input_set = image_set(kind, lo, hi)
    else:
        input_set = flat_set(kind, lo.reshape(-1), hi.reshape(-1))
    with pytest.raises(NotImplementedError):
        reach_model(_IndexLastAxis(), input_set, "approx",
                    f"x[..., 1:3] on {kind}",
                    input_shape=(2, 3, 5) if image else (30,))


# --------------------------------------------------------------------------- #
# Parameters that are not initializers                                         #
# --------------------------------------------------------------------------- #
def _constant_start_nodes(form):
    """Nodes and initializers that compute the Slice inputs ``st`` = [2] and
    ``en`` = [5] (axis 3 of a (1, 2, 3, 5) tensor) without initializers for
    them. ``shape_gather_sub``: st = Shape(X)[3] - 3, as exporters with
    dynamic axes write it; ``load_onnx`` folds it with onnx-simplifier.
    ``constant_subgraph``: st = Sub(5, 3) of two initializers and ``en`` from
    a Constant node, which the walker folds itself (``_const_eval``)."""
    if form == "shape_gather_sub":
        return ([helper.make_node("Shape", ["X"], ["sh"]),
                 helper.make_node("Gather", ["sh", "i3"], ["w"], axis=0),
                 helper.make_node("Sub", ["w", "k3"], ["st"])],
                {"i3": _i64(3), "k3": _i64(3), "en": _i64(5)})
    return ([helper.make_node("Sub", ["k5", "k3"], ["st"]),
             helper.make_node("Constant", [], ["en"],
                              value=numpy_helper.from_array(_i64(5), "c5"))],
            {"k5": _i64(5), "k3": _i64(3)})


@pytest.mark.parametrize("form", ["shape_gather_sub", "constant_subgraph"])
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box", "ImageStar"])
def test_slice_parameters_from_a_constant_subgraph_hold_every_output(
        tmp_path, kind, form):
    """Relu, Slice [2:5] on axis 3, Neg, with the starts and ends computed by
    a constant subgraph instead of given as initializers (both forms of
    ``_constant_start_nodes``). Catches a parameter read that takes only
    initializers: on a flat set the shape-aware path then falls back to the
    legacy path, which keeps the width axis whole."""
    pre, inits = _constant_start_nodes(form)
    inits["ax"] = _i64(3)
    nodes = pre + [helper.make_node("Relu", ["X"], ["A"]),
                   helper.make_node("Slice", ["A", "st", "en", "ax"], ["S"]),
                   helper.make_node("Neg", ["S"], ["Y"])]
    slices = (slice(None), slice(None), slice(None), slice(2, 5))
    _check_model(tmp_path, nodes, inits, _SHAPE, [1, 2, 3, 3], kind,
                 "approx", f"Slice from a {form}, {kind}",
                 box=_selection_box(_SHAPE, slices))


@pytest.mark.parametrize("kind", ["Star", "Box", "ImageStar"])
def test_slice_with_starts_computed_from_the_input_is_refused(tmp_path,
                                                               kind):
    """Slice whose starts depend on the input's values (``Cast`` to int64 of
    ``ReduceSum(X)``): a set has no single start, so reach must refuse
    (NotImplementedError, or ValueError for a parameter it cannot read),
    whichever node refuses. Today the Slice handler refuses: its parameter
    read fails, the shape-aware path falls back to the legacy path, whose
    read raises ValueError. Catches a fallback that answers with a default
    start instead."""
    lo = np.zeros((2, 3, 5))
    hi = np.full((2, 3, 5), 0.125)                   # the sum is in [0, 3.75]
    nodes = [helper.make_node("ReduceSum", ["X", "ra"], ["m"], keepdims=0),
             helper.make_node("Cast", ["m"], ["mi"], to=TensorProto.INT64),
             helper.make_node("Reshape", ["mi", "one"], ["st"]),
             helper.make_node("Slice", ["X", "st", "en", "ax"], ["Y"])]
    inits = {"ra": _i64(1, 2, 3), "one": _i64(1), "en": _i64(5),
             "ax": _i64(3)}
    path = save_onnx(tmp_path, nodes, _SHAPE, [1, 2, 3, "w"], inits=inits)
    model = load_onnx(path)
    with pytest.raises((NotImplementedError, ValueError)):
        reach_model(model, _any_set(kind, lo, hi), "approx",
                    f"Slice with computed starts, {kind}",
                    input_shape=(2, 3, 5))


# --------------------------------------------------------------------------- #
# Defects of today's library                                                   #
# --------------------------------------------------------------------------- #
def _onnx_axis_indices(dim, start, end, step):
    """The indices ONNX Slice selects on an axis of length ``dim``, by the
    operator's definition written out (no Python or numpy slice): negative
    starts and ends get ``dim`` added, then for a positive step both are
    clamped to [0, dim], for a negative step the start to [0, dim - 1] and
    the end to [-1, dim - 1]."""
    start = start + dim if start < 0 else start
    end = end + dim if end < 0 else end
    if step > 0:
        start, end = min(max(start, 0), dim), min(max(end, 0), dim)
    else:
        start, end = min(max(start, 0), dim - 1), min(max(end, -1), dim - 1)
    return list(range(start, end, step))


_NEG_STEP_START = known_bug(
    7, "GRAPH-20: ONNX Slice with negative step and a start before the first "
       "entry (start < -dim: -6 on an axis of 5, or INT_MIN) selects nothing "
       "(Python's clamping) or, for |start| >= 2**31 - 1 on a flat set, the "
       "whole reversed axis, where ONNX clamps the start to index 0: the "
       "result has the wrong number of entries (behind a ReduceSum, a "
       "one-entry set that leaves out every onnxruntime output)")


@pytest.mark.parametrize("kind,start", [
    ("Star", -5),
    pytest.param("Star", -6, marks=_NEG_STEP_START),
    pytest.param("Star", INT_MIN, marks=_NEG_STEP_START),
    ("ImageStar", -5),
    pytest.param("ImageStar", -6, marks=_NEG_STEP_START)])
def test_slice_with_a_negative_step_from_before_the_first_entry(
        tmp_path, kind, start):
    """Relu, then Slice of the width (length 5) of a (1, 2, 3, 5) tensor with
    step -1, end INT_MIN and a start at or before the first entry. ONNX
    clamps a start below -dim to index 0 for a negative step, so -5 and -6
    and INT_MIN all select entry 0 alone (``_onnx_axis_indices``); onnxruntime
    agrees. Start -5 (exactly -dim) is the passing neighbour. Catches a start
    clamped by Python's rule (an empty selection) or read as an open start
    (the whole axis reversed), on the shape-aware path (flat Star) and on
    the image path (ImageStar)."""
    width = len(_onnx_axis_indices(5, start, INT_MIN, -1))
    if width != 1:
        pytest.fail(f"setup: ONNX selects {width} entries, not 1")
    inits = {"st": _i64(start), "en": _i64(INT_MIN), "ax": _i64(3),
             "sp": _i64(-1)}
    nodes = [helper.make_node("Relu", ["X"], ["A"]),
             helper.make_node("Slice", ["A", "st", "en", "ax", "sp"], ["Y"])]
    _check_model(tmp_path, nodes, inits, _SHAPE, [1, 2, 3, width], kind,
                 "approx", f"Slice [{start}::-1] of the width, {kind}",
                 box=_tensor_box((2, 3, 5), crossing=(0, 15), flat=20))


_STEPS_AS_KEYWORD = known_bug(
    7, "GRAPH-21: ONNX Slice whose axes input is omitted (empty name) while "
       "steps is given: onnx2torch passes steps as a keyword argument, "
       "_handle_onnx_slice reads only the positional arguments and takes "
       "every step as 1, so the result has the wrong entries and the wrong "
       "number of them (every onnxruntime output outside)")


@pytest.mark.parametrize("kind,form", [
    ("Box", "axes_given"),
    pytest.param("Box", "axes_omitted", marks=_STEPS_AS_KEYWORD),
    ("ImageStar", "axes_given"),
    pytest.param("ImageStar", "axes_omitted", marks=_STEPS_AS_KEYWORD)])
def test_slice_with_steps_and_no_axes_input(tmp_path, kind, form):
    """Relu, Slice, Neg on (1, 2, 3, 5): starts [0, 1, 0, 0], ends [INT_MAX,
    2, 3, 5], steps [1, 1, 1, 2] (channel 1, every row, columns 0, 2, 4),
    with the axes input omitted by an empty name (legal: an optional input
    may be skipped) or given as [0, 1, 2, 3], the passing neighbour. Catches
    steps dropped when they arrive as a keyword argument, on the shape-aware
    path (Box) and on the image path (ImageStar)."""
    names = ["A", "st", "en", "" if form == "axes_omitted" else "ax", "sp"]
    inits = {"st": _i64(0, 1, 0, 0), "en": _i64(INT_MAX, 2, 3, 5),
             "sp": _i64(1, 1, 1, 2)}
    if form == "axes_given":
        inits["ax"] = _i64(0, 1, 2, 3)
    nodes = [helper.make_node("Relu", ["X"], ["A"]),
             helper.make_node("Slice", names, ["S"]),
             helper.make_node("Neg", ["S"], ["Y"])]
    slices = (slice(None), slice(1, 2), slice(None), slice(0, 5, 2))
    _check_model(tmp_path, nodes, inits, _SHAPE, [1, 1, 3, 3], kind,
                 "approx", f"Slice with steps, {form}, {kind}",
                 box=_selection_box(_SHAPE, slices))
