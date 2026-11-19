"""Soundness cells for ONNX Gather with constant indices: the row selection
of ``_handle_onnx_gather`` and ``_select_rows`` in ``n2v/nn/reach.py``.

A Gather with constant indices selects entries of a tensor: output entry k is
input entry p[k], for an index array p that the walker computes from the
tensor shape it tracked for the Gather's input. The cells check that every
true output is in a returned set for

- indices that are unsorted, repeated, negative, a scalar (which removes the
  axis) and a 2-D index tensor, on each of the three non-batch axes of a
  (1, 2, 3, 5) tensor (two channels, height 3, width 5); the negative axes
  -1, -2 and -3; axis 0 (the batch axis) written out and left out (the ONNX
  default at every opset), and left out on the (6, 5) tensor a Reshape makes;
- ImageStar and ImageZono inputs, which store height, width, channel and
  must be put in the tensor's channel-first order before the selection, and
  flat Star, Zono and Box inputs that stand for the tensor (``input_shape``);
- a box; a Star with a binding row that is not a box row; a Star and an
  ImageStar whose row ties a kept entry to a dropped one (``coupled_star``);
  a Zono with more generators than entries and a Star with more predicate
  variables than entries; the pieces of an exact split, which reach the
  Gather together;
- the composition affine, Gather, affine (a 1x1 convolution in front of an
  image input, a Gemm and an in-graph Reshape in front of a flat input) on a
  box eight times the usual size, and with the exact method a ReLU behind or
  in front of the Gather;
- a model whose tensor-shape record is dropped as a whole (an ONNX Flatten
  with axis 0, registry SHAPE-8): the Gather has no shape to read and must
  refuse, or hold every output; a flat set given without ``input_shape`` is
  read as the tensor (1, dim) (registry SHAPE-2, marked).

Refusals: Hexatope and Octatope have no row selection; ``method='exact'`` on a
Zono, ImageZono, Box, Hexatope or Octatope is refused at the model entry;
GatherElements and GatherND must be refused or answered soundly, never read
as a Gather.

Elsewhere: a model input without a batch axis (registry SHAPE-1) and an
ImageZono without ``input_shape`` (SHAPE-3) are marked for Gather in
``test_soundness_tensor_shape.py``. ``reach_layer`` handed an ``OnnxGather``
module refuses for every set type (the indices live in the graph, so only
the graph walker answers a Gather).

Oracles: onnxruntime replays member points of the input set (dyadic bounds
and weights, so its float32 arithmetic is exact) and each true output must be
a member of a returned set by an LP on that set's own rows
(``tests/_oracles/membership.py``). For the exact method the reference is the
exact image in rationals (``tests/_oracles/rational.py``) of a Star with three
predicate variables, in both directions: no vertex of the true image outside
the result, and no vertex of a returned set without a preimage. The linear
parts of those networks are read off onnxruntime on unit vectors, never off
n2v.

Layout: bounds, member points and true outputs are in the network's order
(channel, height, width for the input tensor). A Gather returns a flat set in
the row-major order of the gathered tensor.
"""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper

from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (N_SAMPLES, assert_covered,
                                constrained_image_star, constrained_star,
                                coupled_star, flat_set, hwc_index, image_set,
                                pixel_box, reach_model, replay_onnx, samples,
                                save_onnx)
from tests.known_bugs import known_bug

C, H, W = 2, 3, 5
SHAPE = (1, C, H, W)
N = C * H * W

# (axis, indices) on the (1, 2, 3, 5) tensor. Every case has an index order
# that a sorted or deduplicated selection would change.
CASES = {
    "channel-reversed-repeat": (1, [1, 0, 1]),
    "height-unsorted-repeat": (2, [2, 0, 2, 1]),
    "width-unsorted-repeat": (3, [4, 0, 3, 3, 1]),
    "height-2d-index-tensor": (2, [[2, 0], [1, 1]]),
    "width-negative-scalar": (3, -2),
    "height-negative": (2, [-1, 0]),
}
KINDS = ("ImageStar", "ImageZono", "Star", "Zono", "Box")
# The one-node grid: every index form (1-D unsorted and repeated, 2-D, scalar,
# negative) and each of the three non-batch axes on every set kind. A 1-D
# repeated index on the inner axis (height) adds no code path to the height
# cases with a 2-D and a negative index and the 1-D cases on the first and
# last axes; it is kept for the chain below.
ONE_NODE_CASES = [case for case in CASES if case != "height-unsorted-repeat"]


def _dyadic(rng, *shape):
    """Quarters in [-1, 1] as float32: exact in float32 and float64."""
    return (rng.integers(-4, 5, size=shape) / 4.0).astype(np.float32)


def _gather_node(axis, out="G", data="X"):
    """A Gather of ``data`` by the initializer ``I``; ``axis=None`` leaves the
    attribute out."""
    if axis is None:
        return helper.make_node("Gather", [data, "I"], [out])
    return helper.make_node("Gather", [data, "I"], [out], axis=axis)


def _out_shape(shape, axis, indices):
    return list(np.take(np.zeros(shape), indices, axis=axis).shape)


def _inputs(kind, form, lo, hi):
    """``(sets, member points in (C, H, W) order, reach options)``.

    ``form`` is "box" (the box itself) or "row" (a Star or ImageStar with a
    binding row that is not a box row). Flat sets keep the tensor's row-major
    order and carry ``input_shape``; an ImageZono carries it too, since
    without it the walker tracks no tensor shape for it (registry SHAPE-3)."""
    image = kind.startswith("Image")
    options = {} if kind == "ImageStar" else {"input_shape": (C, H, W)}
    if form == "box":
        s = image_set(kind, lo, hi) if image else flat_set(kind, lo, hi)
        return [s], samples(lo, hi), options
    if form == "row":
        s, points = (constrained_image_star(lo, hi) if image
                     else constrained_star(lo, hi))
        return [s], points, options
    raise ValueError(form)


# --------------------------------------------------------------------------- #
# One Gather node                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", ONE_NODE_CASES)
def test_gather_selects_the_entries_onnxruntime_selects(tmp_path, case, kind):
    """A Gather alone on a (2, 3, 5) tensor, every index form, every set kind
    that has a row selection. Catches indices sorted or deduplicated before
    the selection (``np.unique``: wrong order, wrong entry count; a Box
    selected by a mask), an image set flattened in its stored height, width,
    channel order instead of channel-first (``to_star`` / ``to_zono`` before
    the selection), row numbers computed for the wrong axis or for a tracked
    shape with height and width exchanged or the batch axis missing, a
    negative, scalar or 2-D index read wrongly, and a selection written into
    its input set (``reach_model`` fails when the input set changed)."""
    axis, indices = CASES[case]
    indices = np.asarray(indices, dtype=np.int64)
    path = save_onnx(tmp_path, [_gather_node(axis, out="Y")], SHAPE,
                     _out_shape(SHAPE, axis, indices), inits={"I": indices})
    lo, hi = pixel_box(C, H, W)
    sets, points, options = _inputs(kind, "box", lo, hi)
    what = f"Gather {case} on {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what, **options)
    assert_covered(out, replay_onnx(path, points, SHAPE), what)


# id -> (axis, indices, kind, Reshape target or None). Axis None leaves the
# attribute out; the converter then stores 0, the ONNX default at every
# opset. The indices of the negative axes are legal on the axis to the left
# too (except axis -3, whose left neighbour is the batch axis), so an axis
# read one place off selects other entries instead of failing.
AXIS_CASES = {
    "minus-1-Star": (-1, [2, 0, 2], "Star", None),
    "minus-1-ImageZono": (-1, [2, 0, 2], "ImageZono", None),
    "minus-2-Zono": (-2, [1, 0], "Zono", None),
    "minus-2-ImageStar": (-2, [1, 0], "ImageStar", None),
    "minus-3-Box": (-3, [1, 0, 1], "Box", None),
    "0-Star": (0, [0, 0], "Star", None),
    "left-out-Box": (None, [0, 0], "Box", None),
    "left-out-after-reshape-ImageStar": (None, [4, 0, 3], "ImageStar",
                                         (6, 5)),
}


@pytest.mark.parametrize("case", list(AXIS_CASES))
def test_gather_on_a_negative_axis_or_axis_0_selects_the_entries_onnxruntime_selects(  # noqa: E501
        tmp_path, case):
    """A Gather over the negative axes -1, -2 and -3 of a (1, 2, 3, 5) tensor,
    over axis 0 written out and left out (indices [0, 0] repeat the batch
    entry: an output of 60 entries), and with the axis left out on the
    (6, 5) tensor that a Reshape of an image makes, whose axis 0 has six
    entries. Catches a negative axis normalised to another axis (one place
    to the left), axis 0 replaced by a default because Python reads 0 as
    false (``axis or -1``), and a default for the left-out attribute other
    than the ONNX default 0."""
    axis, indices, kind, target = AXIS_CASES[case]
    indices = np.asarray(indices, dtype=np.int64)
    inits = {"I": indices}
    taken = 0 if axis is None else axis
    if target is None:
        nodes = [_gather_node(axis, out="Y")]
        out_shape = _out_shape(SHAPE, taken, indices)
    else:
        inits["S"] = np.asarray(target, dtype=np.int64)
        nodes = [helper.make_node("Reshape", ["X", "S"], ["R"]),
                 _gather_node(axis, out="Y", data="R")]
        out_shape = _out_shape(target, taken, indices)
    path = save_onnx(tmp_path, nodes, SHAPE, out_shape, inits=inits)
    lo, hi = pixel_box(C, H, W)
    sets, points, options = _inputs(kind, "box", lo, hi)
    what = f"Gather axis {axis} indices {indices.tolist()} on {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what, **options)
    assert_covered(out, replay_onnx(path, points, SHAPE), what)


# --------------------------------------------------------------------------- #
# Input sets whose inner shape is not the one from_bounds gives                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,method", [("Star", "approx"),
                                         ("Star", "exact"),
                                         ("ImageStar", "approx"),
                                         ("ImageStar", "exact")])
def test_gather_keeps_a_row_that_ties_a_kept_entry_to_a_dropped_one(
        tmp_path, kind, method):
    """A Gather over the width (indices [4, 0, 3]: widths 1 and 2 dropped) of
    a Star or ImageStar from ``coupled_star``, whose row a_i + a_j <= 0 ties a
    kept entry i to a dropped entry j, on a box eight times the usual size.
    The selection must keep every predicate variable and every row. Catches
    a selection that drops the predicate variables no kept entry reads by
    deleting their columns of the rows: the tie becomes the cut a_i <= 0 and
    the kept entry loses the upper half of its range. The Gather reads no
    method, so the same code answers under both; both rows are checked."""
    axis, indices = 3, np.array([4, 0, 3], dtype=np.int64)
    kept = np.take(np.arange(N).reshape(C, H, W), indices, axis=2).reshape(-1)
    lo, hi = pixel_box(C, H, W)
    star, points = coupled_star(kind, 8 * lo, 8 * hi, kept)
    path = save_onnx(tmp_path, [_gather_node(axis, out="Y")], SHAPE,
                     _out_shape(SHAPE, axis, indices), inits={"I": indices})
    options = {} if kind == "ImageStar" else {"input_shape": (C, H, W)}
    what = f"Gather over the width of a coupled {kind}, {method}"
    out = reach_model(load_onnx(path), star, method, what, **options)
    assert_covered(out, replay_onnx(path, points, SHAPE), what)


@pytest.mark.parametrize("kind", ["Zono", "Star"])
def test_gather_of_a_set_with_more_generators_than_entries(tmp_path, kind):
    """A Gather over the height (indices [2, 0, 2, 1]) of a Zono with 60
    dense generators for its 30 entries, and of a Star with 60 predicate
    variables (the shapes a ReLU relaxation leaves), centres up to 16 and
    generator entries up to 2. Catches a selection that reads the generator
    matrix by columns instead of rows (``V[:, rows].T``), which is the same
    set for the diagonal generators of a box and a different one here."""
    rng = np.random.default_rng(17)
    k = 2 * N
    centre = rng.integers(-16, 17, size=N).astype(np.float64)
    G = rng.integers(-8, 9, size=(N, k)) / 4.0
    if kind == "Zono":
        input_set = Zono(centre.reshape(-1, 1), G)
    else:
        input_set = Star(np.hstack([centre[:, None], G]), np.zeros((0, k)),
                         np.zeros((0, 1)), -np.ones((k, 1)), np.ones((k, 1)))
    alphas = [rng.choice([-1.0, 1.0], size=k) if j % 2
              else rng.integers(-2, 3, size=k) / 2.0
              for j in range(N_SAMPLES)]
    points = [centre + G @ a for a in alphas]
    indices = np.array([2, 0, 2, 1], dtype=np.int64)
    path = save_onnx(tmp_path, [_gather_node(2, out="Y")], SHAPE,
                     _out_shape(SHAPE, 2, indices), inits={"I": indices})
    what = f"Gather over the height of a {kind} with {k} generators"
    out = reach_model(load_onnx(path), input_set, "approx", what,
                      input_shape=(C, H, W))
    assert_covered(out, replay_onnx(path, points, SHAPE), what)


# --------------------------------------------------------------------------- #
# affine -> Gather -> affine                                                    #
# --------------------------------------------------------------------------- #
def _chain_model(tmp_path, kind, case, rng):
    """``(path, in_shape)`` of affine -> Gather -> Reshape(1, -1) -> Gemm.

    An image input passes a 1x1 convolution that mixes the two channels; a
    flat input of 30 entries passes a 30 x 30 Gemm and a Reshape to
    (1, 2, 3, 5) that feeds no convolution (so the set stays flat and its
    shape comes from the graph)."""
    axis, indices = CASES[case]
    indices = np.asarray(indices, dtype=np.int64)
    gathered = int(np.prod(_out_shape(SHAPE, axis, indices)))
    inits = {"I": indices, "F": np.array([1, -1], dtype=np.int64),
             "W2": _dyadic(rng, 3, gathered), "b2": _dyadic(rng, 3)}
    if kind.startswith("Image"):
        in_shape = SHAPE
        inits.update({"K": _dyadic(rng, C, C, 1, 1), "kb": _dyadic(rng, C)})
        front = [helper.make_node("Conv", ["X", "K", "kb"], ["A"])]
    else:
        in_shape = (1, N)
        inits.update({"W1": _dyadic(rng, N, N), "b1": _dyadic(rng, N),
                      "S": np.array(SHAPE, dtype=np.int64)})
        front = [helper.make_node("Gemm", ["X", "W1", "b1"], ["L"], transB=1),
                 helper.make_node("Reshape", ["L", "S"], ["A"])]
    nodes = front + [
        _gather_node(axis, data="A"),
        helper.make_node("Reshape", ["G", "F"], ["R"]),
        helper.make_node("Gemm", ["R", "W2", "b2"], ["Y"], transB=1)]
    return save_onnx(tmp_path, nodes, in_shape, [1, 3], inits=inits), in_shape


CHAIN_CELLS = (
    [(case, kind, "box") for case in ("channel-reversed-repeat",
                                      "width-unsorted-repeat")
     for kind in KINDS]
    + [(case, kind, "row") for case in ("height-unsorted-repeat",
                                        "width-unsorted-repeat")
       for kind in ("ImageStar", "Star")])


@pytest.mark.parametrize("case,kind,form", CHAIN_CELLS)
def test_gather_between_two_affine_layers_holds_every_output(tmp_path, case,
                                                             kind, form):
    """affine -> Gather -> affine, on a box and on a Star with a binding
    non-box row, the box eight times the usual size (bounds up to 16,
    intervals up to 10 wide), so a result that reads a predicate variable in
    place of an entry cannot stay inside [-1, 1]. Catches the bugs of the
    one-node cell where the downstream Gemm consumes the selected rows of a
    set whose generators the affine layer in front has mixed, a selection
    that drops the binding row's correlation, and a Gather whose tracked
    input shape is wrong because the shape came from an in-graph Reshape or
    a convolution (an ImageStar placeholder with height and width
    exchanged)."""
    rng = np.random.default_rng(7)
    path, in_shape = _chain_model(tmp_path, kind, case, rng)
    lo, hi = pixel_box(C, H, W, seed=3)
    lo, hi = 8 * lo, 8 * hi
    sets, points, options = _inputs(kind, form, lo, hi)
    if not kind.startswith("Image"):
        options = {}          # the in-graph Reshape gives the tensor shape
    what = f"affine/Gather {case}/Gemm on {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what, **options)
    assert_covered(out, replay_onnx(path, points, in_shape), what)


# --------------------------------------------------------------------------- #
# The exact method: exact image in rationals, both directions                   #
# --------------------------------------------------------------------------- #
def _linear_part(tmp_path, nodes, in_shape, out_shape, inits, name):
    """``(A, a)`` with ``f(x) = A x + a`` for the affine sub-graph ``nodes``,
    read off onnxruntime on 0 and on the unit vectors (exact for dyadic
    weights). The map is checked on one more dyadic point."""
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


def _low_rank(rng, n, k=3):
    """``(x0, E)``: centre and k generators in the network's order, dyadic,
    for the input set {x0 + E t : t in [-1, 1]^k}."""
    return (rng.integers(-8, 9, size=n) / 8.0,
            rng.integers(-4, 5, size=(n, k)) / 8.0)


def _assert_exact(out, pieces, what):
    computed = [rational.star(s) for s in out]
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


def test_exact_gather_on_an_image_star_then_relu_is_the_exact_image(tmp_path):
    """ImageStar, exact: Gather (width, unsorted) -> Gemm -> ReLU -> Gemm.
    Catches a Gather that reads the stored height, width, channel order or
    sorts its indices (the exact image is missed), and an exact reach that
    does not reach the ReLU after the Gather with ``method='exact'`` (an
    over-approximation has vertices without a preimage)."""
    rng = np.random.default_rng(11)
    axis, indices = 3, np.array([4, 0, 3], dtype=np.int64)
    gathered = C * H * 3
    W1, b1 = _dyadic(rng, 4, gathered), _dyadic(rng, 4)
    W2, b2 = _dyadic(rng, 2, 4), _dyadic(rng, 2)

    def front(out):
        return [_gather_node(axis),
                helper.make_node("Reshape", ["G", "F"], ["R"]),
                helper.make_node("Gemm", ["R", "W1", "b1"], [out], transB=1)]
    inits = {"I": indices, "F": np.array([1, -1], dtype=np.int64),
             "W1": W1, "b1": b1}
    A, a = _linear_part(tmp_path, front("Y"), SHAPE, [1, 4], inits, "prefix")
    nodes = front("P") + [
        helper.make_node("Relu", ["P"], ["Q"]),
        helper.make_node("Gemm", ["Q", "W2", "b2"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, SHAPE, [1, 2],
                     inits=dict(inits, W2=W2, b2=b2))
    x0, E = _low_rank(rng, N)
    order = hwc_index(C, H, W)
    V = np.hstack([x0[order, None], E[order]])
    k = E.shape[1]
    image = ImageStar(V, np.zeros((0, k)), np.zeros((0, 1)),
                      -np.ones((k, 1)), np.ones((k, 1)),
                      height=H, width=W, num_channels=C)
    what = "exact Gather/Gemm/ReLU/Gemm on a low-rank ImageStar"
    out = reach_model(load_onnx(path), image, "exact", what)
    pieces = rational.box([-1] * k, [1] * k)
    pieces = rational.affine(pieces, E.tolist(), x0.tolist())
    pieces = rational.affine(pieces, A.tolist(), a.tolist())
    pieces = rational.affine(rational.relu(pieces), W2.astype(float).tolist(),
                             b2.astype(float).tolist())
    _assert_exact(out, pieces, what)
    t = np.random.default_rng(1).integers(-4, 5, size=(64, k)) / 4.0
    assert_covered(out, replay_onnx(path, [x0 + E @ ti for ti in t],
                                    SHAPE), what)


def test_exact_gather_of_the_pieces_of_an_exact_split(tmp_path):
    """Star, exact: Gemm -> ReLU (split into pieces) -> Reshape (1, 2, 3) ->
    Gather (axis 2, [2, 0, 2]) -> Gemm. The Gather receives every piece of
    the split; catches a selection applied to the first piece only, rows in
    sorted order, and pieces dropped or duplicated on the way."""
    rng = np.random.default_rng(5)
    n_in, n_mid = 6, 6
    W1, b1 = _dyadic(rng, n_mid, n_in), _dyadic(rng, n_mid)
    W2, b2 = _dyadic(rng, 2, 6), _dyadic(rng, 2)
    tail_inits = {"S": np.array([1, 2, 3], dtype=np.int64),
                  "I": np.array([2, 0, 2], dtype=np.int64),
                  "F": np.array([1, -1], dtype=np.int64), "W2": W2, "b2": b2}
    tail = [helper.make_node("Reshape", ["Q", "S"], ["T"]),
            _gather_node(2, data="T"),
            helper.make_node("Reshape", ["G", "F"], ["R"]),
            helper.make_node("Gemm", ["R", "W2", "b2"], ["Y"], transB=1)]
    A, a = _linear_part(
        tmp_path, [helper.make_node("Identity", ["X"], ["Q"])] + tail,
        (1, n_mid), [1, 2], tail_inits, "tail")
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["P"], transB=1),
             helper.make_node("Relu", ["P"], ["Q"])] + tail
    path = save_onnx(tmp_path, nodes, (1, n_in), [1, 2],
                     inits=dict(tail_inits, W1=W1, b1=b1))
    x0, E = _low_rank(rng, n_in)
    k = E.shape[1]
    star = Star(np.hstack([x0[:, None], E]), np.zeros((0, k)),
                np.zeros((0, 1)), -np.ones((k, 1)), np.ones((k, 1)))
    what = "exact Gemm/ReLU/Gather/Gemm on a low-rank Star"
    out = reach_model(load_onnx(path), star, "exact", what)
    pieces = rational.affine(rational.box([-1] * k, [1] * k), E.tolist(),
                             x0.tolist())
    pieces = rational.relu(rational.affine(pieces, W1.astype(float).tolist(),
                                           b1.astype(float).tolist()))
    pieces = rational.affine(pieces, A.tolist(), a.tolist())
    assert len(out) > 1, f"{what}: the ReLU did not split ({len(out)} set)"
    _assert_exact(out, pieces, what)


# --------------------------------------------------------------------------- #
# No tracked tensor shape                                                       #
# --------------------------------------------------------------------------- #
_SHAPE_2 = known_bug(
    12, "a flat set on a model whose input has rank 4, given without "
        "input_shape, is analysed as the tensor (1, dim) with no error: a "
        "Gather over the channels then selects single entries and the result "
        "has 3 entries where the output has 45 (registry SHAPE-2)")

# indices per axis, legal on that axis of the (1, 2, 3, 5) tensor
_ROUTE_INDICES = {0: [0, 0], 1: [1, 0, 1], -1: [4, 0, 3], 2: [2, 0, 2, 1]}


@pytest.mark.parametrize("route,axis,kind", [
    pytest.param("record-dropped", 0, "Star", id="record-dropped-axis0-Star"),
    pytest.param("record-dropped", 1, "Star", id="record-dropped-axis1-Star"),
    pytest.param("record-dropped", -1, "Star",
                 id="record-dropped-axis-1-Star"),
    pytest.param("record-dropped", 2, "Star", id="record-dropped-axis2-Star"),
    pytest.param("record-dropped", 1, "ImageZono",
                 id="record-dropped-axis1-ImageZono"),
    pytest.param("record-kept", 1, "Star", id="record-kept-axis1-Star"),
    pytest.param("no-input-shape", 1, "Star", marks=_SHAPE_2,
                 id="no-input-shape-axis1-Star"),
])
def test_gather_with_no_tracked_shape_refuses_or_holds_every_output(
        tmp_path, route, axis, kind):
    """Gather, then an ONNX Flatten, on a (1, 2, 3, 5) input. With the
    Flatten's axis 0 the converted module's forward raises on the rank-4
    tensor, so the shape pass records no shape for any node (registry row
    SHAPE-8) although ``input_shape`` is given, and the Gather has no shape to
    read. For every spelling of its axis (0, 1, -1 and the inner axis 2) the
    call must be refused, or every onnxruntime output must be in a returned
    set of the true size. Catches a Gather that guesses a shape there (the
    tensor (1, dim)) instead of refusing. With the Flatten's axis 1 the
    record is kept and the Gather must answer (the passing neighbour);
    without ``input_shape`` a flat set is read as the tensor (1, dim),
    registry SHAPE-2, marked."""
    indices = np.asarray(_ROUTE_INDICES[axis], dtype=np.int64)
    gathered = _out_shape(SHAPE, axis, indices)
    flat_axis = 0 if route == "record-dropped" else 1
    out_shape = [int(np.prod(gathered[:flat_axis])),
                 int(np.prod(gathered[flat_axis:]))]
    path = save_onnx(tmp_path, [_gather_node(axis),
                                helper.make_node("Flatten", ["G"], ["Y"],
                                                 axis=flat_axis)],
                     SHAPE, out_shape, inits={"I": indices})
    lo, hi = pixel_box(C, H, W)
    sets, points, options = _inputs(kind, "box", lo, hi)
    if route == "no-input-shape":
        options = {}
    what = f"Gather axis {axis}, Flatten axis {flat_axis} on {kind} ({route})"
    outputs = replay_onnx(path, points, SHAPE)
    try:
        out = reach_model(load_onnx(path), sets, "approx", what, **options)
    except (NotImplementedError, ValueError):
        if route == "record-kept":
            raise               # a tracked shape must be answered
        return
    assert_covered(out, outputs, what)


# --------------------------------------------------------------------------- #
# Refusal                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_gather_on_a_hexatope_or_octatope_is_refused(tmp_path, kind):
    """Hexatope and Octatope have no row selection; a Gather on one must
    raise NotImplementedError, not return the set unchanged or a set of the
    wrong entries."""
    indices = np.array([2, 0, 2, 1], dtype=np.int64)
    path = save_onnx(tmp_path, [_gather_node(2, out="Y")], SHAPE,
                     _out_shape(SHAPE, 2, indices), inits={"I": indices})
    lo, hi = pixel_box(C, H, W)
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                    f"Gather on {kind}", input_shape=(C, H, W))


@pytest.mark.parametrize("kind", ["Zono", "ImageZono", "Box", "Hexatope",
                                  "Octatope"])
def test_exact_gather_on_a_set_with_no_exact_method_is_refused(tmp_path,
                                                              kind):
    """The rows (exact, Zono / ImageZono / Box / Hexatope / Octatope): the
    model entry refuses ``method='exact'`` on these set types before any node
    (today ``ValueError`` naming 'approx'; the planned ``UnsupportedError``
    subclasses ``NotImplementedError``). Catches the model entry passing
    'exact' through to the Gather handler, which reads no method and would
    answer under the name exact."""
    indices = np.array([2, 0, 2, 1], dtype=np.int64)
    path = save_onnx(tmp_path, [_gather_node(2, out="Y")], SHAPE,
                     _out_shape(SHAPE, 2, indices), inits={"I": indices})
    lo, hi = pixel_box(C, H, W)
    input_set = (image_set(kind, lo, hi) if kind == "ImageZono"
                 else flat_set(kind, lo, hi))
    with pytest.raises((NotImplementedError, ValueError), match="'approx'"):
        reach_model(load_onnx(path), input_set, "exact",
                    f"exact Gather on {kind}", input_shape=(C, H, W))


@pytest.mark.parametrize("op", ["GatherElements", "GatherND"])
def test_gather_elements_and_gather_nd_refuse_or_hold_every_output(tmp_path,
                                                                  op):
    """GatherElements (an index for every output entry, along the height) and
    GatherND (index tuples (batch, channel)) with constant indices on a Star
    that stands for the (1, 2, 3, 5) input. Neither has a reach rule today;
    each must be refused, or answered with every onnxruntime output in a
    returned set of the true size. Catches the walker sending them to the
    Gather handler (a type test that matches every name starting with
    OnnxGather), which reads their indices as Gather indices: the
    GatherElements output of 30 entries comes back as a set of 300."""
    rng = np.random.default_rng(3)
    if op == "GatherElements":
        indices = rng.integers(0, H, size=SHAPE).astype(np.int64)
        node = helper.make_node(op, ["X", "I"], ["Y"], axis=2)
        out_shape = list(SHAPE)
    else:
        indices = np.array([[0, 1], [0, 0]], dtype=np.int64)
        node = helper.make_node(op, ["X", "I"], ["Y"])
        out_shape = [2, H, W]
    path = save_onnx(tmp_path, [node], SHAPE, out_shape, inits={"I": indices})
    lo, hi = pixel_box(C, H, W)
    sets, points, options = _inputs("Star", "box", lo, hi)
    what = f"{op} on a Star"
    outputs = replay_onnx(path, points, SHAPE)
    try:
        out = reach_model(load_onnx(path), sets, "approx", what, **options)
    except NotImplementedError:
        return
    assert_covered(out, outputs, what)
