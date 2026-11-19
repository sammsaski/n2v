"""Soundness cells for ONNX Split: the walker's Split and getitem branches
in ``n2v/nn/reach.py`` (``_handle_graphmodule``), ``_handle_onnx_split``,
the kernel ``_split_set`` and the row selection ``_select_rows``.

The rows. Through an ONNX file, Split answers on a Star and an ImageStar
under ``approx`` and ``exact`` and on a Zono, an ImageZono and a Box under
``approx``; ``method='exact'`` on a Zono, ImageZono, Box, Hexatope or
Octatope is refused by the model entry before the walk (``ValueError``
today, shared by every operation and tested with it); a Hexatope or an
Octatope under ``approx`` is refused by the Split code
(``NotImplementedError``). ``reach_layer`` handed a Split module has no rule
for it and refuses for every set type, and a torch model that splits
(``torch.split``, ``Tensor.chunk``) is refused by the walker.

What the cells check: every true output of the network on a member point of
the input set is a member of a returned set. Oracles, none of them n2v:
onnxruntime replay of the same ONNX file (dyadic data, so float32 is exact),
membership by an LP on the rows of the returned sets
(``cell_support.assert_covered``), the exact image in rationals
(``tests._oracles.rational``) for the exact method, and for direct kernel
calls the numpy split of member points computed from the arrays the test
built. Layout: a tensor is (C, H, W) row-major, a flat set keeps the tensor's
row-major order, an image set stores (H, W, C) and is compared through
``out_chw``. Every chunk is used: the graphs put all chunks back together in
reverse order (each through its own Neg), or take one chunk.

Defects of today's library marked here (strict expected failures):

- GRAPH-27: an invalid Split of an image
  set (an even split of a length that the number of outputs does not
  divide, sizes that do not sum to the axis length) is answered where
  onnxruntime refuses the model;
- SHAPE-7: sizes that do not sum to the axis
  length are answered on a flat set too, through the batch-free placeholder
  of the shape pass, or through the path of SHAPE-8 (no tracked shape);
- LOAD-6, a Split with one output: onnx2torch hands the one-element tuple
  to the next
  node, and the reach crashes with ``TypeError``;
- on the route of SHAPE-8 (a model whose tensor-shape record is lost although
  ``input_shape`` is given), the Split of a flat set cuts its rows as if the
  split axis were the outermost one: an even split along an inner axis
  returns chunks of the right size that hold none of the true outputs.

Not here: image sets with a negative axis or split on the batch axis, and a
chunk with zero entries (``test_soundness_negative_axis.py``: GRAPH-4,
GRAPH-5); a Split output that is the graph output, models without a batch
axis and flat sets given without ``input_shape``
(``test_soundness_tensor_shape.py``: GRAPH-8, SHAPE-1, SHAPE-2); the pairing
of Split outputs with look-alike predicate systems
(``test_soundness_set_arith_cells.py``).
"""

import numpy as np
import pytest
import torch
from onnx import helper
from onnx2torch.node_converters.split import OnnxSplit, OnnxSplit13
from torch import nn

from n2v.nn.reach import _split_set
from n2v.sets import Box, Hexatope, Octatope, Zono
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as Q
from tests.cell_support import (
    assert_covered, constrained_image_star, constrained_star, coupled_star,
    flat_set, hwc_index, image_set, reach_model, reach_one_layer, replay_onnx,
    samples, save_onnx)
from tests.known_bugs import DID_NOT_RAISE, known_bug


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


def _grid(cells):
    """Parameters ``(kind, method, *option)`` from ``{(kind, method): [option,
    ...]}``: the handler code shared by all kinds is swept on the cheap Box
    and a Star approx, the kind-specific kernels on fewer options."""
    return [(k, m) + (o if isinstance(o, tuple) else (o,))
            for (k, m), options in cells.items() for o in options]


def _input_set(kind, lo, hi, coupled=None):
    if coupled is not None:
        return coupled_star(kind, lo, hi, coupled)
    if kind in ("ImageStar", "ImageZono"):
        return image_set(kind, lo, hi), samples(lo, hi)
    return flat_set(kind, lo, hi), samples(lo, hi)


def _any_set(kind, lo, hi):
    """The box [lo, hi] of a tensor as a set of ``kind``: an image set for
    ImageStar and ImageZono ((C, H, W) bounds), a flat set otherwise."""
    if kind in ("ImageStar", "ImageZono"):
        return image_set(kind, lo, hi)
    return flat_set(kind, lo, hi)


def _check_model(tmp_path, nodes, inits, in_shape, out_shape, kind, method,
                 what, opset=13, box=None, coupled=None):
    """Reach the ONNX graph from the input set of ``kind`` and assert that
    every onnxruntime output of a member point is in a returned set, compared
    in the layout each returned set stores. ``coupled`` (row-major indices of
    the entries one chunk keeps) asks for the Star or ImageStar of
    ``coupled_star``. Under ``approx`` every layer of these graphs returns
    one set per input set (the Split one chunk per input set and output), so
    more sets fail before the membership LPs, whose cost grows with them."""
    lo, hi = box if box is not None else _tensor_box(tuple(in_shape[1:]))
    path = save_onnx(tmp_path, nodes, in_shape, out_shape, inits=inits,
                     opset=opset)
    input_set, points = _input_set(kind, lo, hi, coupled)
    outputs = replay_onnx(path, points, in_shape)
    sets = reach_model(load_onnx(path), input_set, method, what,
                       input_shape=tuple(in_shape[1:]))
    assert method != "approx" or len(sets) == 1, (
        f"{what}: approx returned {len(sets)} sets for one input set")
    out_chw = tuple(out_shape[1:]) if len(out_shape) == 4 else None
    assert_covered(sets, outputs, what, out_chw=out_chw)
    return sets


# Splits: (axis, sizes or None for an even split, number of outputs, opset,
# sizes given as an input (opset 13) or as the attribute (opset 11)).
_CASES = {
    "h_sizes_1_1_2": (2, [1, 1, 2], 3, 13),
    "c_sizes_1_2": (1, [1, 2], 2, 13),
    "w_even_2": (3, None, 2, 13),
    "h_even_2": (2, None, 2, 13),
    "c_even_3": (1, None, 3, 13),
    "h_attr_3_1": (2, [3, 1], 2, 11),
    "c_attr_2_1": (1, [2, 1], 2, 11),
    "h_even_2_opset11": (2, None, 2, 11),
}
_SHAPE = [1, 3, 4, 2]


def _split_graph(shape, axis, sizes, n_out, opset, use="all", before="Relu"):
    """``(nodes, inits, out_shape)``: before(X), Split, then either every
    chunk through its own Neg, concatenated in reverse order along the split
    axis (``use="all"``), or chunk ``use`` through Neg."""
    a = axis % len(shape)
    length = shape[a]
    chunk = sizes if sizes is not None else [length // n_out] * n_out
    outs = [f"P{k}" for k in range(n_out)]
    names, inits, attrs = ["A"], {}, {"axis": axis}
    if sizes is not None and opset >= 13:
        names.append("sp")
        inits["sp"] = _i64(*sizes)
    elif sizes is not None:
        attrs["split"] = list(sizes)
    nodes = [helper.make_node(before, ["X"], ["A"]),
             helper.make_node("Split", names, outs, **attrs)]
    if use == "all":
        nodes += [helper.make_node("Neg", [o], [f"N{k}"])
                  for k, o in enumerate(outs)]
        nodes.append(helper.make_node(
            "Concat", [f"N{k}" for k in reversed(range(n_out))], ["Y"],
            axis=axis))
        return nodes, inits, list(shape)
    nodes.append(helper.make_node("Neg", [outs[use]], ["Y"]))
    out = list(shape)
    out[a] = chunk[use]
    return nodes, inits, out


_ALL = list(_CASES)


@pytest.mark.parametrize("kind,method,case", _grid({
    ("Box", "approx"): _ALL,
    ("Star", "approx"): ["h_sizes_1_1_2", "c_even_3", "c_attr_2_1"],
    ("Star", "exact"): ["h_sizes_1_1_2", "c_even_3"],
    ("Zono", "approx"): ["h_sizes_1_1_2", "w_even_2", "c_attr_2_1"]}))
def test_split_of_a_flat_set_holds_every_output(tmp_path, kind, method, case):
    """Relu, Split, a Neg on every chunk, Concat of the chunks in reverse
    order, on a flat set standing for a (1, 3, 4, 2) tensor: sizes 1, 1, 2,
    even splits into two and three, the attribute form of opset 11. Catches
    the split applied to the axis before the requested one, chunk indices
    flattened in column-major order, a running offset advanced wrongly, and
    an exact Relu list whose pieces are not all split."""
    axis, sizes, n_out, opset = _CASES[case]
    nodes, inits, out = _split_graph(_SHAPE, axis, sizes, n_out, opset)
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Split {case}, {kind} {method}", opset=opset)


@pytest.mark.parametrize("kind,method,case,use", _grid({
    ("Box", "approx"): [("h_sizes_1_1_2", k) for k in range(3)]
    + [("c_sizes_1_2", 1)],
    ("Star", "approx"): [("h_sizes_1_1_2", 2), ("c_even_3", 1)],
    ("ImageStar", "approx"): [("h_sizes_1_1_2", k) for k in range(3)]
    + [("c_sizes_1_2", 1), ("w_even_2", 1)],
    ("ImageZono", "approx"): [("h_sizes_1_1_2", 2), ("c_even_3", 2)]}))
def test_split_single_chunk_holds_every_output(tmp_path, kind, method, case,
                                               use):
    """One chunk of a Split as the graph's last set (through a Neg): each
    chunk on its own must hold the true chunk, so a chunk that repeats an
    earlier chunk's entries, or is read from offset 0, is caught without the
    Concat of the other cells."""
    axis, sizes, n_out, opset = _CASES[case]
    nodes, inits, out = _split_graph(_SHAPE, axis, sizes, n_out, opset,
                                     use=use)
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Split {case} chunk {use}, {kind} {method}", opset=opset)


@pytest.mark.parametrize("kind,method,shape,case", _grid({
    ("ImageStar", "approx"): [((1, 3, 4, 2), c) for c in
                              ("h_sizes_1_1_2", "c_sizes_1_2", "w_even_2",
                               "c_even_3", "h_attr_3_1", "h_even_2_opset11")]
    + [((1, 2, 3, 5), "h_sizes_2_1"), ((1, 2, 3, 5), "w_sizes_2_3")],
    ("ImageStar", "exact"): [((1, 3, 4, 2), "h_sizes_1_1_2"),
                             ((1, 3, 4, 2), "c_attr_2_1")],
    ("ImageZono", "approx"): [((1, 3, 4, 2), c) for c in
                              ("h_sizes_1_1_2", "w_even_2", "c_even_3")]
    + [((1, 2, 3, 5), "w_sizes_2_3")]}))
def test_split_of_an_image_set_holds_every_output(tmp_path, kind, method,
                                                  shape, case):
    """Relu, Split, Neg on every chunk, Concat in reverse order, on an
    ImageStar or ImageZono of a (3, 4, 2) and a (2, 3, 5) image, through the
    handler's image path. Catches the ONNX axis mapped to the wrong stored
    axis, every chunk sliced from offset 0, an even split whose chunk length
    is read from another axis, and an ImageZono chunk whose centre and
    generators are cut differently."""
    shape = list(shape)
    extra = {"h_sizes_2_1": (2, [2, 1], 2, 13),
             "w_sizes_2_3": (3, [2, 3], 2, 13)}
    axis, sizes, n_out, opset = {**_CASES, **extra}[case]
    nodes, inits, out = _split_graph(shape, axis, sizes, n_out, opset)
    _check_model(tmp_path, nodes, inits, shape, out, kind, method,
                 f"Split {case} on {shape}, {kind} {method}", opset=opset)


# Negative axes on flat sets with a known shape: the tensor (1, 2, 3, 3) has
# two axes of length 3 (H and W), the tensor (1, 4, 4) two of length 4, so a
# negative axis normalised to the wrong axis splits a set of the right size
# along the wrong axis.
_NEG_CASES = {
    "rank4_m1_sizes": ([1, 2, 3, 3], -1, [1, 2], 2),
    "rank4_m1_even": ([1, 2, 3, 3], -1, None, 3),
    "rank4_m2_sizes": ([1, 2, 3, 3], -2, [2, 1], 2),
    "rank4_m3_even": ([1, 2, 3, 3], -3, None, 2),
    "rank3_m1_sizes": ([1, 4, 4], -1, [1, 1, 2], 3),
    "rank3_m2_even": ([1, 4, 4], -2, None, 2),
}


@pytest.mark.parametrize("kind,method,case", _grid({
    ("Box", "approx"): list(_NEG_CASES),
    ("Star", "approx"): ["rank4_m2_sizes", "rank3_m2_even"],
    ("Zono", "approx"): ["rank4_m1_sizes", "rank3_m1_sizes"],
    ("Star", "exact"): ["rank4_m1_even"]}))
def test_split_of_a_flat_set_on_a_negative_axis(tmp_path, kind, method,
                                                case):
    """Split with a negative axis on flat sets whose tensor shape is known
    (the opset-13 form, even and with sizes). Catches a negative axis
    normalised off by one (rank - 1 + axis) or with abs(): the chunks are cut
    along the wrong axis of the same length, or the sizes stop matching."""
    shape, axis, sizes, n_out = _NEG_CASES[case]
    nodes, inits, out = _split_graph(shape, axis, sizes, n_out, 13)
    _check_model(tmp_path, nodes, inits, shape, out, kind, method,
                 f"Split {case}, {kind} {method}")


@pytest.mark.parametrize("kind,method,case", _grid({
    ("Star", "approx"): ["h_sizes_1_1_2", "c_even_3"],
    ("Star", "exact"): ["h_attr_3_1"],
    ("ImageStar", "approx"): ["h_sizes_1_1_2", "c_sizes_1_2"],
    ("ImageStar", "exact"): ["w_even_2"]}))
def test_split_of_a_star_with_a_binding_row_holds_every_output(
        tmp_path, kind, method, case):
    """Relu, Split, Neg, reversed Concat on a Star / ImageStar cut by two rows
    that are not box rows (``coupled_star``): one ties an entry of the first
    chunk to an entry of another chunk (a_i + a_j <= 0), one ties two entries
    of the first chunk. Catches a chunk that drops the predicate variables
    its own entries do not use together with their columns of the rows,
    which fixes them at zero, so the entry tied to a dropped one keeps only
    half its range, and rows or
    bounds moved onto other variables; a box check passes both, the member
    points with a_i = 1 and a_j = -1 and on the faces of the rows do not. A
    chunk that loses rows is larger, not wrong, and is not this cell's
    subject."""
    axis, sizes, n_out, opset = _CASES[case]
    nodes, inits, out = _split_graph(_SHAPE, axis, sizes, n_out, opset)
    lengths = sizes if sizes is not None else \
        [_SHAPE[axis] // n_out] * n_out
    first = np.split(np.arange(int(np.prod(_SHAPE))).reshape(_SHAPE),
                     np.cumsum(lengths)[:-1], axis=axis)[0].ravel()
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Split {case} on a coupled {kind}, {method}",
                 opset=opset, coupled=first)


def _sparse_rows(rng, n_out, n_in):
    W = np.zeros((n_out, n_in))
    for r in range(n_out):
        cols = rng.choice(n_in, size=min(4, n_in), replace=False)
        W[r, cols] = rng.choice([-1.0, -0.5, 0.5, 1.0, 2.0], size=cols.size)
    return W


def _composition_tail(case, rng):
    """Split, Relu on chunk 0 and Neg on the others, Concat in the original
    order, Flatten, Gemm with a few weights per row: the chunks' rows are
    consumed downstream, so a chunk with the right entries in the wrong order
    changes the five outputs."""
    axis, sizes, n_out, opset = _CASES[case]
    outs = [f"P{k}" for k in range(n_out)]
    names, inits, attrs = ["A"], {}, {"axis": axis}
    if sizes is not None and opset >= 13:
        names.append("sp")
        inits["sp"] = _i64(*sizes)
    elif sizes is not None:
        attrs["split"] = list(sizes)
    nodes = [helper.make_node("Split", names, outs, **attrs)]
    nodes += [helper.make_node("Relu" if k == 0 else "Neg", [o], [f"N{k}"])
              for k, o in enumerate(outs)]
    nodes += [helper.make_node("Concat", [f"N{k}" for k in range(n_out)],
                               ["C"], axis=axis),
              helper.make_node("Flatten", ["C"], ["F"], axis=1),
              helper.make_node("Gemm", ["F", "W2"], ["Y"], transB=1)]
    inits["W2"] = _f32(_sparse_rows(rng, 5, 24))
    return nodes, inits, opset


@pytest.mark.parametrize("kind,method,case", _grid({
    ("Box", "approx"): ["h_sizes_1_1_2", "w_even_2", "c_attr_2_1"],
    ("Star", "approx"): ["h_sizes_1_1_2", "c_even_3"],
    ("Star", "exact"): ["c_sizes_1_2"], ("Zono", "approx"): ["h_attr_3_1"]}))
def test_split_between_two_affine_layers_holds_every_output(
        tmp_path, kind, method, case):
    """Gemm (mixes all six inputs), Reshape to (1, 3, 4, 2), Split, Relu on the
    first chunk and Neg on the others, Concat, Flatten, Gemm on a flat set:
    affine, the operation, affine. The bias puts two pre-activations across
    zero (interval arithmetic on the dyadic box)."""
    rng = np.random.default_rng(17)
    lo, hi = _tensor_box((6,), crossing=())
    W1 = rng.integers(-2, 3, size=(24, 6)) / 2.0
    mid, rad = (W1 @ (lo + hi)) / 2.0, np.abs(W1) @ ((hi - lo) / 2.0)
    target = np.where(np.arange(24) % 3 == 0, 1.0, -1.0) * (rad + 0.25)
    target[[0, 1]] = rad[[0, 1]] / 4.0          # two entries of chunk 0
    b1 = target - mid
    tail, inits, opset = _composition_tail(case, rng)
    inits.update({"W1": _f32(W1), "b1": _f32(b1), "shape": _i64(*_SHAPE)})
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["Z"], transB=1),
             helper.make_node("Reshape", ["Z", "shape"], ["A"])] + tail
    _check_model(tmp_path, nodes, inits, [1, 6], [1, 5], kind, method,
                 f"Gemm > Split {case} > Gemm, {kind} {method}", opset=opset,
                 box=(lo, hi))


@pytest.mark.parametrize("kind,method,case", _grid({
    ("ImageStar", "approx"): ["h_sizes_1_1_2", "w_even_2", "c_sizes_1_2"],
    ("ImageStar", "exact"): ["h_attr_3_1"],
    ("ImageZono", "approx"): ["h_sizes_1_1_2", "c_even_3"]}))
def test_split_of_images_between_conv_and_gemm_holds_every_output(
        tmp_path, kind, method, case):
    """Conv 1x1 (mixes the three channels), Split, Relu on the first chunk and
    Neg on the others, Concat, Flatten, Gemm on a (3, 4, 2) image set: the
    chunks are read back by Concat and Flatten, which converts the stored
    (H, W, C) order to the tensor's order."""
    rng = np.random.default_rng(19)
    tail, inits, opset = _composition_tail(case, rng)
    inits.update({"Wc": _f32([[[[1.0]], [[0.5]], [[0.0]]],
                              [[[-0.5]], [[1.0]], [[0.25]]],
                              [[[0.0]], [[-0.25]], [[1.0]]]]),
                  "bc": _f32([0.25, -0.5, 0.5])})
    nodes = [helper.make_node("Conv", ["X", "Wc", "bc"], ["A"])] + tail
    _check_model(tmp_path, nodes, inits, _SHAPE, [1, 5], kind, method,
                 f"Conv > Split {case} > Gemm, {kind} {method}", opset=opset,
                 box=_tensor_box((3, 4, 2), crossing=(0, 9)))


# --------------------------------------------------------------------------- #
# The Split input read a second time, by a join with the chunks                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,method", [
    ("Star", "approx"), ("Star", "exact"), ("Zono", "approx"),
    ("Box", "approx"), ("ImageStar", "approx"), ("ImageStar", "exact"),
    ("ImageZono", "approx")])
def test_split_input_read_again_by_a_join_holds_every_output(tmp_path, kind,
                                                            method):
    """A = Relu(X) feeds the Split and, a second time, a Sub: Y = Concat of
    the Neg of the three chunks (sizes 1, 1, 2 along the height) in reverse
    order, minus A, on a (3, 4, 2) tensor whose box is eight times the usual
    one (ends up to 53, widths 2 to 9); one case per row that answers. The
    chunks and A share A's predicate variables, so the join must pair them as
    one system. Catches a Split that changes the arrays of its input set in
    place (the second reader of A then gets another set), chunks whose rows
    the join pairs with the wrong variables, and a chunk that reads one
    predicate or relaxation variable in place of another or clips the
    relaxation variables of the ReLU to [-1, 1]: on the usual box every
    variable already ranges over about [-1, 1] and such a chunk still holds
    the outputs."""
    axis, sizes, n_out, opset = _CASES["h_sizes_1_1_2"]
    nodes, inits, out = _split_graph(_SHAPE, axis, sizes, n_out, opset)
    nodes[-1].output[0] = "C"
    nodes.append(helper.make_node("Sub", ["C", "A"], ["Y"]))
    lo, hi = _tensor_box(tuple(_SHAPE[1:]))
    _check_model(tmp_path, nodes, inits, _SHAPE, out, kind, method,
                 f"Concat(Neg(Split(A)))[reversed] - A, {kind} {method}",
                 opset=opset, box=(8 * lo, 8 * hi))


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


@pytest.mark.parametrize("kind,axis,sizes,use", [
    ("Star", 3, [1, 1], 1), ("Star", 1, [1, 1], "all"),
    ("ImageStar", 3, [1, 1], "all"), ("ImageStar", 1, [1, 1], 1)])
def test_exact_split_is_the_exact_image(tmp_path, kind, axis, sizes, use):
    """Relu then Split on a (1, 2, 1, 2) tensor whose entries 0 and 3 straddle
    zero, method 'exact', one chunk or all chunks back in reverse order,
    checked in rationals both ways: every vertex of every exact piece is in
    one returned set, and every vertex of every returned set is the image of
    an input point."""
    shape = [1, 2, 1, 2]
    lo, hi = _tensor_box(tuple(shape[1:]), crossing=(0, 3))
    nodes, inits, out = _split_graph(shape, axis, sizes, len(sizes), 13,
                                     use=use)
    sets = _check_model(tmp_path, nodes, inits, shape, out, kind, "exact",
                        f"exact Split axis {axis} use {use}, {kind}",
                        box=(lo, hi))

    def network_after_relu(v):
        chunks = np.split(v.reshape(shape), np.cumsum(sizes)[:-1], axis=axis)
        chunks = [-c for c in chunks]
        if use == "all":
            return np.concatenate(chunks[::-1], axis=axis).reshape(-1)
        return chunks[use].reshape(-1)
    M = np.stack([network_after_relu(np.eye(4)[j]) for j in range(4)],
                 axis=1)
    if kind == "ImageStar":
        M = M[hwc_index(*out[1:])]            # the order the set stores
    pieces = Q.affine(Q.relu(Q.box(lo.reshape(-1), hi.reshape(-1))), M)
    # four input variables, and the exact ReLU adds none: a result with more
    # is wrong, and its vertex enumeration could run for hours
    assert len(sets) <= 4 and all(s.nVar <= 4 for s in sets), (
        f"exact Split axis {axis}: {len(sets)} sets with "
        f"{[s.nVar for s in sets]} predicate variables, where the input has "
        f"four variables and two entries that cross zero")
    computed = [Q.star(s) for s in sets]
    assert Q.straddlers(pieces, *computed) == [], \
        "an exact piece has a vertex outside, or is spread over several sets"
    for k, c in enumerate(computed):
        assert Q.unreached(pieces, _vertices_of(c)) == [], \
            f"returned set {k} has a vertex that no input point reaches"


# --------------------------------------------------------------------------- #
# An attribute left out, a Split of a constant                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_split_with_the_axis_attribute_left_out_holds_every_output(tmp_path,
                                                                   kind):
    """A model whose input (4, 4) has no batch axis, split with the axis
    attribute left out: ONNX then splits axis 0, the rows, into sizes 1 and
    3, and the second chunk goes through a Relu. The input is square, so a
    handler that reads the axis 0 as absent (``axis or 1``) or that takes
    another default cuts chunks of the same size along the columns, and the
    true outputs escape."""
    shape = [4, 4]
    nodes = [helper.make_node("Split", ["X", "sp"], ["P0", "P1"]),
             helper.make_node("Relu", ["P1"], ["Y"])]
    path = save_onnx(tmp_path, nodes, shape, [3, 4], inits={"sp": _i64(1, 3)})
    lo, hi = _tensor_box((4, 4), crossing=(4, 9))
    what = f"Split with no axis attribute on a (4, 4) input, {kind}"
    sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                       what, input_shape=(4, 4))
    assert_covered(sets, replay_onnx(path, samples(lo, hi), shape), what)


@pytest.mark.parametrize("kind", ["Star", "Box"])
def test_split_of_a_constant_added_to_the_set_holds_every_output(tmp_path,
                                                                 kind):
    """Y = X + Split(K)[1]: a constant (1, 12) tensor split into sizes 2, 4
    and 6, whose second chunk (four entries from offset 2) is added to a
    (1, 4) input set. No set rule sees a Split of a constant: the Add folds
    it with torch (``_const_eval``), which keeps the module's tuple of chunks
    for the getitem. Catches the constant chunk taken from another offset
    (the module run without its sizes input splits evenly into chunks of
    four, the same length from offset 4) or from another output."""
    K = (np.arange(12, dtype=np.float64) / 4.0 - 1.5).reshape(1, 12)
    nodes = [helper.make_node("Split", ["K", "sp"], ["K0", "K1", "K2"],
                              axis=1),
             helper.make_node("Add", ["X", "K1"], ["Y"])]
    _check_model(tmp_path, nodes, {"K": _f32(K), "sp": _i64(2, 4, 6)},
                 [1, 4], [1, 4], kind, "approx", f"X + Split(K)[1], {kind}",
                 box=_tensor_box((4,), crossing=(1,)))


# --------------------------------------------------------------------------- #
# The kernel _split_set                                                        #
# --------------------------------------------------------------------------- #
def _mixed_zono(n, k, seed):
    """A Zono with centre and generators that mix its entries (dyadic), and
    member points c + G e for e on the {-1, -1/2, 0, 1/2, 1} grid, computed
    from the arrays."""
    rng = np.random.default_rng(seed)
    c = rng.integers(-8, 9, size=(n, 1)) / 4.0
    G = rng.integers(-4, 5, size=(n, k)) / 4.0
    G[np.arange(min(n, k)), np.arange(min(n, k))] += 1.0
    es = [rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=k) for _ in range(64)]
    es += [rng.choice([-1.0, 1.0], size=k) for _ in range(64)]
    return Zono(c, G), [(c + G @ e[:, None]).ravel() for e in es]


# One split with unequal sizes and a first chunk of one entry, which shows
# every offset slip, and one more per type (equal sizes, or four chunks for
# the Star); the ids keep their numbers over the full grid.
@pytest.mark.parametrize("kind,sizes", [
    pytest.param(k, s, id=f"{k}-sizes{i}") for i, (k, s) in enumerate(
        (k, s) for k in ("Star", "Zono", "Box")
        for s in ([1, 3, 2], [2, 2, 2], [1, 1, 2, 2], [4, 1, 1])
        if not (k == "Star" and s == [2, 2, 2]))
    if (k, s[0], len(s)) in {("Star", 1, 3), ("Star", 1, 4), ("Zono", 1, 3),
                             ("Zono", 2, 3), ("Box", 1, 3), ("Box", 2, 3)}])
def test_split_kernel_on_a_flat_set_holds_every_chunk(kind, sizes):
    """_split_set on a flat Star cut by a non-box row, a Zono whose generators
    mix its entries and a Box, into three or four chunks of equal and
    unequal sizes: every member point's chunk is in the matching returned
    chunk, by the LP on the chunk's rows. Catches a running offset advanced
    by one, by the first chunk's size, or assigned instead of accumulated,
    and generator rows always taken from the top (a chunk that loses rows is
    larger, not wrong)."""
    lo, hi = _tensor_box((6,), crossing=(0, 3))
    if kind == "Star":
        s, points = constrained_star(lo, hi, n=96, seed=8)
    elif kind == "Zono":
        s, points = _mixed_zono(6, 4, seed=9)
    else:
        s, points = Box(lo, hi), samples(lo, hi)
    chunks = _split_set(s, sizes, axis=0)
    if len(chunks) != len(sizes):
        pytest.fail(f"_split_set returned {len(chunks)} chunks for sizes "
                    f"{sizes}")
    cuts = np.cumsum(sizes)[:-1]
    for k, chunk in enumerate(chunks):
        assert_covered([chunk], [np.split(np.asarray(p), cuts)[k]
                                 for p in points],
                       f"_split_set {kind} {sizes} chunk {k}")


@pytest.mark.parametrize("hwc_axis,sizes", [(0, [1, 2]), (1, [2, 1, 2]),
                                            (2, [1, 1])])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_split_kernel_on_an_image_set_holds_every_chunk(kind, hwc_axis,
                                                        sizes):
    """_split_set on a (2, 3, 5) ImageStar cut by a non-box row and an
    ImageZono whose generators mix its pixels, along the stored height,
    width and channel axes: every member point's chunk (split along the
    matching tensor axis) is in the matching returned chunk, compared in the
    stored order. Catches every chunk sliced from offset 0, the stored axes
    confused, and an ImageZono chunk whose centre and generators are cut at
    different places."""
    lo, hi = _tensor_box((2, 3, 5), crossing=(0, 7))
    if kind == "ImageStar":
        s, points = constrained_image_star(lo, hi, n=96, seed=10)
    else:
        z, flat_points = _mixed_zono(30, 8, seed=11)
        s = ImageZono(z.c, z.V, 3, 5, 2)          # stored (H, W, C) = 3x5x2
        order = hwc_index(2, 3, 5)
        points = []
        for p in flat_points:                     # stored order -> tensor
            chw = np.empty(30)
            chw[order] = p
            points.append(chw.reshape(2, 3, 5))
    chw_axis = {0: 1, 1: 2, 2: 0}[hwc_axis]
    chunks = _split_set(s, sizes, axis=hwc_axis)
    if len(chunks) != len(sizes):
        pytest.fail(f"_split_set returned {len(chunks)} chunks for sizes "
                    f"{sizes}")
    cuts = np.cumsum(sizes)[:-1]
    for k, chunk in enumerate(chunks):
        parts = [np.split(np.asarray(p), cuts, axis=chw_axis)[k]
                 for p in points]
        assert_covered([chunk], parts, f"_split_set {kind} axis {hwc_axis} "
                       f"chunk {k}", out_chw=parts[0].shape)


# --------------------------------------------------------------------------- #
# Refusals                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_split_of_a_hexatope_or_octatope_is_refused(tmp_path, kind):
    """Split has no rule for a Hexatope or an Octatope: the model entry point
    must raise NotImplementedError, not return a set."""
    nodes, inits, out = _split_graph(_SHAPE, 2, [1, 1, 2], 3, 13, use=2,
                                     before="Neg")
    path = save_onnx(tmp_path, nodes, _SHAPE, out, inits=inits)
    lo, hi = _tensor_box((3, 4, 2), crossing=())
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                    f"Split on {kind}", input_shape=(3, 4, 2))


@pytest.mark.parametrize("kind", ["Zono", "ImageZono", "Box", "Hexatope",
                                  "Octatope"])
def test_exact_split_model_on_a_set_with_no_exact_method_is_refused(tmp_path,
                                                                    kind):
    """The rows (exact, Zono / ImageZono / Box / Hexatope / Octatope): the
    model entry refuses ``method='exact'`` on these set types before any
    node (today ``ValueError``, '... only supports approx'; the planned
    ``UnsupportedError`` subclasses ``NotImplementedError``), here on Relu,
    Split, Neg. Catches the model entry passing 'exact' through to a lane
    that never reads the method (the Relu in front of the Split would then
    answer with its relaxation under the name exact)."""
    nodes, inits, out = _split_graph(_SHAPE, 2, [1, 1, 2], 3, 13, use=2)
    path = save_onnx(tmp_path, nodes, _SHAPE, out, inits=inits)
    lo, hi = _tensor_box(tuple(_SHAPE[1:]))
    with pytest.raises((ValueError, NotImplementedError)):
        reach_model(load_onnx(path), _any_set(kind, lo, hi), "exact",
                    f"exact Split on {kind}", input_shape=tuple(_SHAPE[1:]))


@pytest.mark.parametrize("cls", [Hexatope, Octatope])
def test_split_kernel_refuses_an_unsupported_set_type(cls):
    """_split_set on a Hexatope or Octatope raises NotImplementedError instead
    of returning chunks without their constraints."""
    lo, hi = _tensor_box((4,), crossing=())
    with pytest.raises(NotImplementedError):
        _split_set(cls.from_bounds(lo, hi), [1, 3], axis=0)


# Two invalid Splits of a height of 3 (opsets 11 and 13): an even split into
# two chunks ("The input is not evenly splittable") and the sizes 1 and 1,
# which do not sum to 3. onnxruntime refuses to load both models and onnx's
# full checker rejects them; the torch module raises on the second. The
# library must refuse them too, as its flat path does for the first.
_INDIVISIBLE = known_bug(
    7, "an even ONNX Split of an image axis whose length is not a multiple "
       "of the number of outputs (height 3 into 2) is answered by the image "
       "path with chunks of length 1 that drop the last row, where "
       "onnxruntime refuses the model and the flat path raises ValueError",
    raises=DID_NOT_RAISE)
_SIZES_MISMATCH = known_bug(
    7, "an ONNX Split whose listed sizes do not sum to the axis length "
       "(sizes 1, 1 on a height of 3) is answered by the image path, which "
       "drops the rows the sizes do not cover, where onnxruntime and torch "
       "refuse the model",
    raises=DID_NOT_RAISE)
_SIZES_MISMATCH_FLAT = known_bug(
    7, "SHAPE-7: a Split whose listed "
       "sizes do not sum to the axis length (1, 1 on a height of 3) is "
       "answered on a flat set given with input_shape, where onnxruntime "
       "refuses the model: the shape pass falls back to the batch-free "
       "placeholder, on which the split axis is another axis whose length "
       "the sizes match, or, where that forward fails too, to the path for a "
       "set with no tracked shape, which checks no sum (SHAPE-8)",
    raises=DID_NOT_RAISE)


def _invalid_split_graph(variant, opset):
    names, inits, attrs = ["A"], {}, {"axis": 2}
    if variant.startswith("sizes_mismatch") and opset >= 13:
        names.append("sp")
        inits["sp"] = _i64(1, 1)
    elif variant.startswith("sizes_mismatch"):
        attrs["split"] = [1, 1]
    nodes = [helper.make_node("Relu", ["X"], ["A"]),
             helper.make_node("Split", names, ["P", "Q"], **attrs),
             helper.make_node("Neg", ["Q"], ["Y"])]
    return nodes, inits


@pytest.mark.parametrize("variant,kind,opset", [
    ("indivisible", "Star", 13), ("indivisible", "Box", 13),
    pytest.param("indivisible", "ImageStar", 13, marks=_INDIVISIBLE),
    pytest.param("indivisible", "ImageZono", 13, marks=_INDIVISIBLE),
    pytest.param("sizes_mismatch", "ImageStar", 13, marks=_SIZES_MISMATCH),
    pytest.param("sizes_mismatch", "ImageZono", 13, marks=_SIZES_MISMATCH),
    pytest.param("sizes_mismatch", "ImageStar", 11, marks=_SIZES_MISMATCH),
    pytest.param("sizes_mismatch", "Star", 13, marks=_SIZES_MISMATCH_FLAT),
    pytest.param("sizes_mismatch_w4", "Box", 13,
                 marks=_SIZES_MISMATCH_FLAT)])
def test_an_invalid_split_is_refused(tmp_path, variant, kind, opset):
    """An even split of a height of 3 into two chunks, or sizes 1 and 1 of a
    height of 3: onnxruntime refuses the model, so the reach must raise
    (ValueError or NotImplementedError) instead of returning chunks of some
    other length. The flat cells of the even split pass today (the
    shape-aware path raises ValueError) and use the same builder and call.
    The flat cells of the mismatched sizes are marked (SHAPE-7): on the
    (1, 3, 3, 2) tensor the batch-free
    placeholder (3, 3, 2) runs, and its axis 2 is the width of length 2;
    with a width of 4 (``sizes_mismatch_w4``) no placeholder runs and the
    path for an untracked shape answers (SHAPE-8)."""
    shape = [1, 3, 3, 4] if variant.endswith("_w4") else [1, 3, 3, 2]
    nodes, inits = _invalid_split_graph(variant, opset)
    path = save_onnx(tmp_path, nodes, shape, [1, 3, 1, shape[3]],
                     inits=inits, opset=opset)
    lo, hi = _tensor_box(tuple(shape[1:]), crossing=())
    input_set = (image_set(kind, lo, hi) if kind.startswith("Image")
                 else flat_set(kind, lo, hi))
    with pytest.raises((ValueError, NotImplementedError)):
        reach_model(load_onnx(path), input_set, "approx",
                    f"{variant} Split, {kind} opset {opset}",
                    input_shape=tuple(shape[1:]))


# --------------------------------------------------------------------------- #
# Other entry routes and spellings                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box", "Hexatope",
                                  "Octatope"])
def test_split_module_handed_to_reach_layer_is_refused(kind):
    """``reach_layer`` has no rule for a Split: its sizes and its several
    outputs live in the graph, which only the walker reads. Both modules
    (opset 13 and opset 11), under both methods, must raise
    NotImplementedError in each of the five lanes of the dispatcher (an
    ImageStar takes the Star lane, an ImageZono the Zono lane). Catches a
    Split module answered as a shape no-op (the identity), whose one result
    keeps every input entry and holds none of the chunks."""
    lo, hi = _tensor_box((2, 3, 5), crossing=(0, 7))
    for module in (OnnxSplit13(num_splits=2, axis=1),
                   OnnxSplit(num_splits=2, axis=1, split=[1, 1])):
        for method in ("approx", "exact"):
            with pytest.raises(NotImplementedError):
                reach_one_layer(module, _any_set(kind, lo, hi), method,
                                f"{type(module).__name__} on {kind}")


class _TorchSplit(nn.Module):
    """Splits the channels of a (1, 2, 3, 5) tensor and returns the second
    part, so that the split is the only node without a reach rule."""

    def __init__(self, how):
        super().__init__()
        self.how = how

    def forward(self, x):
        if self.how == "torch.split":
            _, b = torch.split(x, [1, 1], dim=1)
        else:
            _, b = x.chunk(2, 1)
        return b


@pytest.mark.parametrize("kind,how", [("Star", "torch.split"),
                                      ("ImageZono", "torch.split"),
                                      ("Star", "Tensor.chunk")])
def test_a_torch_model_that_splits_is_refused(kind, how):
    """A torch model that splits a set-valued tensor (``torch.split``, a
    function node, and ``Tensor.chunk``, a method node) has no reach rule:
    the walker must raise NotImplementedError, on a flat set and on an image
    set (the walker refuses before any code that reads the set type).
    Catches a split node answered as the identity or handed on whole, whose
    result holds none of the parts."""
    lo, hi = _tensor_box((2, 3, 5), crossing=(0, 7))
    image = kind in ("ImageStar", "ImageZono")
    with pytest.raises(NotImplementedError):
        reach_model(_TorchSplit(how), _any_set(kind, lo, hi), "approx",
                    f"{how} in a torch model on {kind}",
                    input_shape=None if image else (2, 3, 5))


@pytest.mark.parametrize("kind", ["ImageStar", "Star"])
@pytest.mark.parametrize("variant", ["even", "uneven"])
def test_an_opset_18_split_refuses_or_holds_every_output(tmp_path, variant,
                                                         kind):
    """Opset 18 gives the number of outputs as the attribute ``num_outputs``
    and allows a last chunk shorter than the others (a height of 3 into two:
    2 and 1). The vendored onnx2torch has no converter for it, so
    ``load_onnx`` refuses today (NotImplementedError). The promise: a
    refusal (NotImplementedError or ValueError), or every onnxruntime output
    of both chunks (put back in reverse order) in a returned set. A converter
    added later meets the uneven last chunk, which the image path's even
    split [length // n] * n would drop: with the opset-13 module registered
    for version 18 the uneven ImageStar case fails."""
    shape = [1, 2, 4, 3] if variant == "even" else [1, 2, 3, 3]
    nodes = [helper.make_node("Relu", ["X"], ["A"]),
             helper.make_node("Split", ["A"], ["P0", "P1"], axis=2,
                              num_outputs=2),
             helper.make_node("Neg", ["P0"], ["N0"]),
             helper.make_node("Neg", ["P1"], ["N1"]),
             helper.make_node("Concat", ["N1", "N0"], ["Y"], axis=2)]
    path = save_onnx(tmp_path, nodes, shape, shape, opset=18)
    lo, hi = _tensor_box(tuple(shape[1:]))
    input_set = _any_set(kind, lo, hi)
    outputs = replay_onnx(path, samples(lo, hi), shape)
    what = f"opset-18 Split, {variant}, {kind}"
    try:
        sets = reach_model(load_onnx(path), input_set, "approx", what,
                           input_shape=tuple(shape[1:]))
    except (NotImplementedError, ValueError):
        return
    assert_covered(sets, outputs, what, out_chw=tuple(shape[1:]))


_ONE_OUTPUT = known_bug(
    23, "LOAD-6: an "
        "ONNX Split with a single output is loaded by onnx2torch without a "
        "getitem node, so the next node receives the module's one-element "
        "tuple (the loaded torch model's own forward raises TypeError) and "
        "the reach crashes with TypeError ('Unsupported set type: list') "
        "where an answer or a refusal is due", raises=TypeError)


@pytest.mark.parametrize("n_out", [
    pytest.param(1, marks=_ONE_OUTPUT, id="one-output"),
    pytest.param(2, id="two-outputs")])
def test_a_split_with_one_output_refuses_or_holds_every_output(tmp_path,
                                                               n_out):
    """Relu, a Split along the height of a (1, 3, 4, 2) tensor, then Neg of
    the output that holds all four rows: with one output (sizes 4) it is the
    only output, with two (sizes 4 and 0) the first. A Split with one output
    is legal ONNX and onnxruntime answers it. The promise: a refusal
    (NotImplementedError or ValueError), or every onnxruntime output in a
    returned set. Catches a reach that crashes on, or answers from, the
    tuple a one-output Split hands on; the two-output case is its passing
    neighbour (same handler, reference and check)."""
    outs = ["P0", "P1"][:n_out]
    nodes = [helper.make_node("Relu", ["X"], ["A"]),
             helper.make_node("Split", ["A", "sp"], outs, axis=2),
             helper.make_node("Neg", ["P0"], ["Y"])]
    sizes = _i64(4) if n_out == 1 else _i64(4, 0)
    path = save_onnx(tmp_path, nodes, _SHAPE, _SHAPE, inits={"sp": sizes})
    lo, hi = _tensor_box(tuple(_SHAPE[1:]))
    outputs = replay_onnx(path, samples(lo, hi), _SHAPE)
    what = f"Split with {n_out} output(s) on a Star"
    try:
        sets = reach_model(load_onnx(path), flat_set("Star", lo, hi),
                           "approx", what, input_shape=tuple(_SHAPE[1:]))
    except (NotImplementedError, ValueError):
        return
    assert_covered(sets, outputs, what)


_NO_SHAPE = known_bug(
    7, "registry SHAPE-8: an ONNX Flatten with axis 0 on a rank-4 "
       "input raises in the torch forward of the shape pass, so no tensor "
       "shape is recorded although input_shape is given, and "
       "_handle_onnx_split cuts a flat set's rows as if the split axis were "
       "the outermost one: an even split along an inner axis takes its "
       "chunk length from the set's size and returns chunks of the right "
       "size that hold none of the true outputs; listed sizes give chunks "
       "of the wrong size")


@pytest.mark.parametrize("flatten_axis,axis,sizes,kind", [
    pytest.param(0, 2, None, "Star", marks=_NO_SHAPE,
                 id="no-shape-even-inner-axis-Star"),
    pytest.param(0, 2, (2, 4), "Box", marks=_NO_SHAPE,
                 id="no-shape-sizes-inner-axis-Box"),
    pytest.param(0, 1, None, "Star", id="no-shape-even-outer-axis-Star"),
    pytest.param(1, 2, None, "Star", id="shape-even-inner-axis-Star")])
def test_split_with_no_tensor_shape_recorded_refuses_or_holds_every_output(
        tmp_path, flatten_axis, axis, sizes, kind):
    """X (1, 2, 3, 4), ONNX Flatten with ``flatten_axis``, Reshape to
    (1, 4, 6), Relu, Split along ``axis`` into two (evenly or into 2 and 4),
    Neg of the second chunk, on a flat set given with ``input_shape``. With
    Flatten axis 0 the shape pass records no shape (its torch forward
    raises: the route of SHAPE-8), and the Split of a flat set falls back to a
    cut of its rows. The promise: a refusal (NotImplementedError or
    ValueError), or every onnxruntime output in a returned set. The two
    marked cases split the inner axis; the outer axis, where the fallback's
    guess happens to hold, and the same model with Flatten axis 1 (shape
    recorded) are the passing neighbours. Catches the fallback reading a
    chunk length from the wrong quantity (the number of predicate variables,
    which the Relu raises above the number of entries)."""
    shape = [1, 2, 3, 4]
    outs = ["P0", "P1"]
    names, inits = ["R"], {"shp": _i64(1, 4, 6)}
    if sizes is not None:
        names.append("sp")
        inits["sp"] = _i64(*sizes)
    nodes = [helper.make_node("Flatten", ["X"], ["F"], axis=flatten_axis),
             helper.make_node("Reshape", ["F", "shp"], ["H"]),
             helper.make_node("Relu", ["H"], ["R"]),
             helper.make_node("Split", names, outs, axis=axis),
             helper.make_node("Neg", ["P1"], ["Y"])]
    chunk = sizes[1] if sizes is not None else (4, 6)[axis - 1] // 2
    out = [1, 4, 6]
    out[axis] = chunk
    path = save_onnx(tmp_path, nodes, shape, out, inits=inits)
    lo, hi = _tensor_box(tuple(shape[1:]))
    outputs = replay_onnx(path, samples(lo, hi), shape)
    what = (f"Flatten axis {flatten_axis}, Split axis {axis} sizes {sizes}, "
            f"{kind}")
    try:
        sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                           what, input_shape=tuple(shape[1:]))
    except (NotImplementedError, ValueError):
        return
    assert_covered(sets, outputs, what)
