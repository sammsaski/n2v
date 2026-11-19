"""Soundness cells for ONNX Cast (``OnnxCast``) on every set type, and for the
casts a torch model spells as tensor methods.

The promise (owner decision of 2026-11-13, DESIGN section 17, registry rows
AFFINE-1, AFFINE-2 and AFFINE-3): a Cast to float32 or float64 is the
identity on values, which is what n2v returns; a Cast to any other target
(an integer type, BOOL, FLOAT16) changes the value and must be refused with
``NotImplementedError`` where the kernel meets the ``OnnxCast`` module, after
loading. Each cell passes when the call raises ``NotImplementedError`` or
every true output is a member of a returned set. Today every lane of the
dispatcher returns its input whatever the target (the Star, Zono and Box
lanes, AFFINE-1; the Hexatope and Octatope lanes and the image sets through
the Star and Zono lanes, AFFINE-2), and so does the interval pre-pass
(AFFINE-3, marked in ``test_soundness_bounds_prepass.py``).

Expected failures: one per set type at the model entry, the route that
decides what a user gets (``test_cast_graph_refuses_or_holds_every_
onnxruntime_output``), each beside a passing FLOAT neighbour on the same data.
The target is not read by today's code, so one target per set type marks the
one failing line of its lane; the targets differ from set type to set type
so that every kind of value change has a cell for the fix to pass (UINT8,
BOOL, INT64, FLOAT16). Star, Zono and Box under INT32, INT64, BOOL and
FLOAT16 are marked in ``tests/contracts/test_reach_coverage.py::
test_cast_refuses_or_is_sound``; only UINT8, which that test does not have, is
marked here.

Cells:

- Cast to FLOAT through ``reach_layer`` on one set of every type (an
  ImageStar under ``method='exact'``, with the reverse direction), on three
  sets of different sizes whose union is the box and on a Star cut by a
  binding row (both methods for the Star, whose lane is the only one with an
  exact method; the others under ``approx``), and through the model behind an
  exact ReLU, so that the Cast node is handed several Stars.
- The graph Cast through ``reach_pytorch_model`` on a model loaded with
  ``load_onnx`` (loaded before the call that may refuse): a single
  ``Cast(FLOAT)`` (so a Cast that moves or negates the set shows), or
  ``Cast(target)`` then ``Cast(FLOAT)`` for a value-changing target.
- The bound pre-pass in front of a ReLU that reads the recorded bounds: Gemm,
  Cast, ReLU, Gemm on a Star with ``precompute_bounds='ibp'`` and ``'zono'``,
  and Conv, Cast, ReLU, Conv on an ImageStar with ``'ibp'``; every ReLU input
  crosses zero, so recorded bounds that are moved, or laid out in the wrong
  order, overlap the true ones and narrow the ReLU's triangle instead of
  being dropped.
- A float64 graph input cast to FLOAT (the identity under the real-valued
  reading; exact membership of the input points) and a float32 input cast to
  DOUBLE.
- Tensor-method casts in a torch model (``x.to(dtype)``, ``x.type(dtype)``,
  ``x.double()``), which the graph walker decides in ``_is_identity_method``.

Oracle: the true outputs of sample points (corners included) by onnxruntime
or by torch in float64; each is a member of a returned set by the LP on the
set's own rows, or, for the float64 input, by exact rational membership. The
inputs are dyadic where the cast does not have to round them, cross zero,
hold a coordinate of zero width and values strictly between two integers and
between two neighbouring half-precision numbers; the UINT8 inputs are
non-negative and below 256, where the ONNX cast is defined. No reference
comes from n2v.

Rows that refuse: ``method='exact'`` on a Zono, ImageZono, Box, Hexatope or
Octatope at the model entry (``ValueError`` before any layer; section 11 of
``test_soundness_graph_walk.py``).

Layout: flat sets are in the tensor's row-major order; an image set stores
height, width, channel, and image outputs are compared with ``out_chw``, which
also checks the stored height, width and channel count.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from onnx2torch.node_converters.cast import OnnxCast

from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests._oracles.membership import rows, uncovered
from tests.cell_support import (FLAT_KINDS, IMAGE_KINDS, assert_covered,
                                constrained_star, flat_box, flat_set,
                                image_set, pixel_box, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx, split_cover)
from tests.known_bugs import known_bug

_AFFINE_1 = known_bug(
    7, "AFFINE-1: a Cast is reached as the identity whatever the target type "
       "on a Star, Zono or Box; a Cast to an integer, boolean or "
       "half-precision type changes the value, so the true outputs are left "
       "out of the returned set instead of the Cast being refused")
_AFFINE_2 = known_bug(
    7, "AFFINE-2: a Cast is reached as the identity whatever the target type "
       "on a Hexatope, Octatope, ImageStar or ImageZono; a Cast to an "
       "integer, boolean or half-precision type changes the value, so the "
       "true outputs are left out of the returned set instead of the Cast "
       "being refused")

_KEEPING = ("FLOAT", "DOUBLE")

# A flat input of five entries: x0 crosses zero, x1 lies strictly between 0
# and 1, x2 strictly between two neighbouring half-precision numbers
# (1 + 2^-12 to 1 + 2^-11, below the half spacing 2^-10 at 1), x3 is
# negative and reaches an integer at its lower end, x4 has zero width at a
# value that is not an integer.
_SIGNED_LO = np.array([-1.5, 0.25, 1 + 2.0 ** -12, -3.0, 1.5])
_SIGNED_HI = np.array([1.25, 0.75, 1 + 2.0 ** -11, -2.5, 1.5])
# For UINT8 (defined only on [0, 256)): the same kinds of entries, shifted
# above zero, one of them near the top of the range.
_UNSIGNED_LO = np.array([0.25, 3.25, 1 + 2.0 ** -12, 250.0, 1.5])
_UNSIGNED_HI = np.array([1.75, 3.75, 1 + 2.0 ** -11, 255.5, 1.5])

# An image with two channels and height different from width: half steps in
# [-4, 5.5] plus 2^-12, widths 1/2 to 5/2, so no pixel end is an integer or a
# half-precision number and the box reaches well beyond plus and minus 1.
_C, _H, _W = 2, 3, 5
_ILO, _IHI = (2 * v + 2.0 ** -12 for v in pixel_box(_C, _H, _W, seed=21))


def _flat_bounds(to):
    return (_UNSIGNED_LO, _UNSIGNED_HI) if to == "UINT8" else (_SIGNED_LO,
                                                               _SIGNED_HI)


def _cast_graph(tmp_path, to, in_shape, name="cast"):
    """``X -> Cast(to) -> Y`` for a float target (one Cast, so a Cast that
    moves the set shows), ``X -> Cast(to) -> Cast(FLOAT) -> Y`` for a
    value-changing one, whose output is then the value of ``Cast(to)``."""
    nodes = [helper.make_node("Cast", ["X"], ["A" if to not in _KEEPING
                                              else "Y"],
                              to=getattr(TensorProto, to))]
    if to not in _KEEPING:
        nodes.append(helper.make_node("Cast", ["A"], ["Y"],
                                      to=TensorProto.FLOAT))
    return save_onnx(tmp_path, nodes, in_shape, in_shape, name=f"{name}_{to}",
                     out_type=to if to in _KEEPING else "FLOAT")


def _corners(s, n=24, seed=0):
    """Up to ``n`` corners of the predicate box of ``s`` that keep its rows,
    mapped through its stored basis (read with ``membership.rows``, no method
    of the set). Stored order."""
    c, G, C, d, lb, ub = rows(s)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(20 * n):
        if len(out) >= n:
            break
        a = np.where(rng.integers(0, 2, size=lb.size) == 1, ub, lb)
        if C.shape[0] == 0 or np.all(C @ a <= d):
            out.append(c + G @ a)
    return out


# --------------------------------------------------------------------------- #
# Cast to FLOAT through reach_layer, every set type                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,to", [(kind, "FLOAT") for kind in
                                     FLAT_KINDS + IMAGE_KINDS])
def test_cast_layer_refuses_or_holds_every_output(tmp_path, kind, to):
    """``OnnxCast(FLOAT)`` through ``reach_layer`` on one set of each kind
    (the ImageStar under ``method='exact'``): the call refuses, or every
    onnxruntime output of the cast is a member of the result, and for the
    exact ImageStar every corner of the result is a point of the input.
    Catches a Cast branch that negates, shifts or replaces its input (a call
    to the neighbouring Neg helper, say) or returns an image set of another
    shape, in each of the five lanes and on both image types."""
    layer = OnnxCast(getattr(TensorProto, to))
    method = "exact" if kind == "ImageStar" else "approx"
    what = f"Cast to {to} on {kind} through reach_layer ({method})"
    if kind in IMAGE_KINDS:
        lo, hi = _ILO, _IHI
        shape = [1, _C, _H, _W]
        input_set = image_set(kind, lo, hi)
    else:
        lo, hi = _flat_bounds(to)
        shape = [1, lo.size]
        input_set = flat_set(kind, lo, hi)
    try:
        result = reach_one_layer(layer, [input_set], method, what)
    except NotImplementedError:
        return
    outputs = replay_onnx(_cast_graph(tmp_path, to, shape), samples(lo, hi),
                          shape)
    if kind in IMAGE_KINDS:
        assert_covered(result, outputs, what, out_chw=(_C, _H, _W))
    else:
        assert_covered(result, outputs, what)
    if method == "exact":
        back = [y for k, s in enumerate(result) for y in _corners(s, seed=k)]
        missed = uncovered([input_set], back)
        assert not missed, (
            f"{what}: {len(missed)} of {len(back)} corners of the result are "
            f"not points of the input (largest distance "
            f"{max(m[1] for m in missed):.3g})")


def _polytope_of(star):
    """The exact pieces of a Star built by ``cell_support`` (not by n2v): its
    predicate polytope ``C a <= d, lb <= a <= ub`` mapped by ``a -> c + G a``,
    read from the stored arrays in rationals."""
    c, G, C, d, lb, ub = R.star(star)
    k = len(lb)
    A, b = [list(row) for row in C], list(d)
    for j in range(k):
        unit = [int(i == j) for i in range(k)]
        A.append(unit)
        b.append(ub[j])
        A.append([-v for v in unit])
        b.append(-lb[j])
    return R.affine(R.polytope(A, b), G, c)


@pytest.mark.parametrize("kind,inputs,method", [
    ("Star", "several sets", "approx"), ("Star", "several sets", "exact"),
    ("Star", "binding row", "approx"), ("Star", "binding row", "exact"),
    ("Zono", "several sets", "approx"), ("Box", "several sets", "approx"),
    ("Hexatope", "several sets", "approx"),
    ("Octatope", "several sets", "approx")])
def test_float_cast_layer_on_several_or_cut_sets(tmp_path, kind, inputs, method):
    """``OnnxCast(FLOAT)`` (the identity) through ``reach_layer`` on three sets
    of different sizes whose union is the box, and on a Star cut by a binding
    row that is not a box row. Catches a Cast branch that returns only the
    first set of the list or answers the first set for every set. For a Star
    with ``method='exact'`` the result must be the exact image in rationals,
    both directions: every vertex of each input set is in a returned set, and
    no vertex of a returned set lies outside the input (a Cast that rebuilt
    the Star from its bounds would return the cut-off corner). The branch
    never reads the method; the model entry refuses ``'exact'`` for the other
    set types, so they run under ``'approx'`` only."""
    layer = OnnxCast(TensorProto.FLOAT)
    lo, hi = _SIGNED_LO, _SIGNED_HI
    what = f"Cast to FLOAT on {inputs} of {kind} ({method})"
    if inputs == "binding row":
        star, points = constrained_star(lo, hi, seed=3)
        sets = [star]
    else:
        sets = split_cover(kind, lo, hi, parts=3)
        points = samples(lo, hi)
    result = reach_one_layer(layer, sets, method, what)
    if len(result) != len(sets):
        raise AssertionError(f"{what}: {len(result)} sets returned for "
                             f"{len(sets)}; the Cast hands each set on")
    assert_covered(result, replay_onnx(_cast_graph(tmp_path, "FLOAT",
                                                   [1, lo.size]),
                                       points, [1, lo.size]), what)
    if kind == "Star" and method == "exact":
        image = [p for s in sets for p in _polytope_of(s)]
        computed = [R.star(s) for s in result]
        if any(len(s[4]) > 8 for s in computed):
            raise AssertionError(
                f"{what}: a returned set has more than 8 predicate variables, "
                f"more than the input; its vertices are not enumerated")
        escaped = R.escapes(image, *computed)
        assert not escaped, (
            f"{what}: {len(escaped)} vertices of the input are in none of the "
            f"returned sets, first {escaped[0]}")
        for k, s in enumerate(computed):
            far = R.unreached(image, R.set_vertices(s))
            assert not far, (
                f"{what}: returned set {k} has vertices outside the input, "
                f"first {far[0][1]}")


def test_float_cast_behind_an_exact_relu_holds_every_output(tmp_path):
    """Gemm, ReLU, Cast(FLOAT), Gemm with ``method='exact'`` through the model:
    the exact ReLU splits the input Star, so the Cast node is handed a list of
    several Stars. Catches a Cast that returns only the first Star of the
    list (outputs of the other pieces are then missing)."""
    w1 = np.array([[1, -2, 1, 0, 1], [2, 1, 0, -1, 0], [-1, 1, 2, 1, -1]],
                  np.float32) / 4
    w2 = np.array([[1, 0, -1], [-2, 1, 1]], np.float32) / 2
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["A"], transB=1),
             helper.make_node("Relu", ["A"], ["R"]),
             helper.make_node("Cast", ["R"], ["C"], to=TensorProto.FLOAT),
             helper.make_node("Gemm", ["C", "W2", "b2"], ["Y"], transB=1)]
    inits = dict(W1=w1, b1=np.array([0.25, -0.5, 0.125], np.float32),
                 W2=w2, b2=np.array([0.5, -0.25], np.float32))
    path = save_onnx(tmp_path, nodes, [1, 5], [1, 2], inits=inits, name="grcg")
    what = "Gemm, ReLU, Cast(FLOAT), Gemm on Star (exact)"
    sets = reach_model(load_onnx(path), flat_set("Star", _SIGNED_LO, _SIGNED_HI),
                       "exact", what)
    if len(sets) < 2:
        pytest.fail(f"{what}: the exact ReLU returned {len(sets)} set; the cell "
                    f"needs several sets at the Cast node")
    assert_covered(sets, replay_onnx(path, samples(_SIGNED_LO, _SIGNED_HI),
                                     [1, 5]), what)


# --------------------------------------------------------------------------- #
# The graph Cast through the model entry point                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,to", [
    ("Hexatope", "FLOAT"),
    pytest.param("Hexatope", "BOOL", marks=_AFFINE_2),
    ("Octatope", "FLOAT"),
    pytest.param("Octatope", "INT64", marks=_AFFINE_2),
    ("ImageStar", "FLOAT"),
    pytest.param("ImageStar", "FLOAT16", marks=_AFFINE_2),
    ("ImageZono", "FLOAT"),
    pytest.param("ImageZono", "FLOAT16", marks=_AFFINE_2),
    pytest.param("Star", "UINT8", marks=_AFFINE_1)])
def test_cast_graph_refuses_or_holds_every_onnxruntime_output(tmp_path, kind,
                                                              to):
    """The graph Cast loaded with ``load_onnx`` and reached with
    ``reach_pytorch_model`` (an image set with ``input_shape``): the reach
    refuses, or every onnxruntime output is a member. The model is loaded
    before the call that may refuse: the refusal belongs where the reach
    meets the ``OnnxCast`` module, not at load (the loader removes the INT64
    casts of some benchmark files). Catches a lane, reached from the graph
    walker, that answers a value-changing Cast as the identity (today's
    defect, one marked target per set type) and, on FLOAT, one that moves,
    negates or reshapes the set."""
    if kind in IMAGE_KINDS:
        lo, hi = _ILO, _IHI
        shape = [1, _C, _H, _W]
        input_set = image_set(kind, lo, hi)
        options = dict(input_shape=(_C, _H, _W))
    else:
        lo, hi = _flat_bounds(to)
        shape = [1, lo.size]
        input_set = flat_set(kind, lo, hi)
        options = {}
    path = _cast_graph(tmp_path, to, shape)
    what = f"Cast to {to} on {kind} through the model"
    model = load_onnx(path)
    try:
        sets = reach_model(model, input_set, "approx", what, **options)
    except NotImplementedError:
        return
    outputs = replay_onnx(path, samples(lo, hi), shape)
    if kind in IMAGE_KINDS:
        assert_covered(sets, outputs, what, out_chw=(_C, _H, _W))
    else:
        assert_covered(sets, outputs, what)


# --------------------------------------------------------------------------- #
# The bound pre-pass in front of a ReLU that reads the recorded bounds          #
# --------------------------------------------------------------------------- #
# Every ReLU input of the flat chain crosses zero with both ends farther than
# 1/2 from 0 (bounds -1.125 to -1.8125 and 0.9375 to 2); rows 0 and 1 of W1
# are negatives of each other, and W2 is invertible, so an output can pin a
# crossing neuron to either end of its interval.
_PRE_LO = np.array([-1.5, -1.25, -1.0, -1.75])
_PRE_HI = np.array([1.25, 1.5, 1.75, 1.0])
_PRE_INITS = dict(
    W1=np.array([[1, -2, 1, 0], [-1, 2, -1, 0], [2, 1, 0, -1], [-1, 1, 2, 1]],
                np.float32) / 4,
    b1=np.array([0.25, -0.25, -0.5, 0.125], np.float32),
    W2=np.array([[1, 0, -1, 2], [-2, 1, 1, 0], [0, 2, 1, -1], [1, 1, 0, 1]],
                np.float32) / 2,
    b2=np.array([0.5, -0.25, 0.0, 0.25], np.float32))


@pytest.mark.parametrize("precompute", ["ibp", "zono"])
@pytest.mark.parametrize("to", ["FLOAT", "DOUBLE"])
def test_cast_before_a_relu_with_the_bound_prepass(tmp_path, to, precompute):
    """Gemm, Cast(to), ReLU (then a Cast back to FLOAT for DOUBLE), Gemm on a
    Star reached with ``precompute_bounds``: the interval pre-pass has its own
    Cast rule (the bounds unchanged), the zonotope pre-pass runs the reach's
    own Zono Cast, and the Star ReLU reads the recorded bounds. Every ReLU
    input crosses zero, so moved bounds overlap the true ones and are used.
    Catches an interval Cast rule that moves the bounds down (the upper end
    the triangle uses is then too low) or up (the lower end too high, seen
    where the paired rows of W1 and the invertible W2 pin a neuron to its
    lower end), and a Zono Cast that moves the set (the zonotope pass then
    records wrong bounds). A value-changing target fails through the router
    of AFFINE-1, marked elsewhere; the pre-pass rule for it is AFFINE-3, whose
    cell on the recorded bounds is in ``test_soundness_bounds_prepass.py``."""
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["A"], transB=1),
             helper.make_node("Cast", ["A"], ["B"], to=getattr(TensorProto, to)),
             helper.make_node("Relu", ["B"], ["R" if to == "DOUBLE" else "Z"])]
    if to == "DOUBLE":
        nodes.append(helper.make_node("Cast", ["R"], ["Z"],
                                      to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Gemm", ["Z", "W2", "b2"], ["Y"], transB=1))
    path = save_onnx(tmp_path, nodes, [1, 4], [1, 4], inits=_PRE_INITS,
                     name=f"gcrg_{to}")
    what = f"Gemm, Cast({to}), ReLU, Gemm on Star, precompute={precompute!r}"
    sets = reach_model(load_onnx(path), flat_set("Star", _PRE_LO, _PRE_HI),
                       "approx", what, precompute_bounds=precompute)
    assert_covered(sets, replay_onnx(path, samples(_PRE_LO, _PRE_HI), [1, 4]),
                   what)


# Conv weights: dyadic, asymmetric kernels, mixed signs, mixing the channels.
_K1 = np.array([[[[1], [-2]], [[1], [1]]],
                [[[2], [1]], [[-1], [0]]]], np.float32) / 4          # (2,2,2,1)
_C1 = np.array([0.25, -0.5], np.float32)
_K2 = np.array([[[[1, -1, 2]], [[0, 1, -1]]],
                [[[-2, 1, 0]], [[1, 1, 1]]]], np.float32) / 4        # (2,2,1,3)
_C2 = np.array([-0.25, 0.5], np.float32)


def test_cast_before_a_relu_on_an_image_with_the_ibp_prepass(tmp_path):
    """Conv, Cast(FLOAT), ReLU, Conv on a (2, 3, 5) ImageStar with
    ``precompute_bounds='ibp'``: the interval pre-pass keeps the image shape
    through the Cast, so the ReLU's bounds are recorded in the stored
    (height, width, channel) order. Every ReLU input crosses zero with ends of
    different size, so bounds laid out in the channel-first order land on
    other neurons, overlap their true intervals and are used. Catches a Cast
    rule that drops the image shape (as the Flatten rule beside it does)."""
    rng = np.random.default_rng(1)
    lo = -(4 + rng.integers(0, 9, size=(_C, _H, _W))) / 4.0
    hi = (4 + rng.integers(0, 9, size=(_C, _H, _W))) / 4.0
    k = _K1[:, :, :, 0].astype(np.float64)
    centre, radius = (lo + hi) / 2, (hi - lo) / 2
    mid = np.zeros((2, _H - 1, _W)) + _C1.astype(np.float64)[:, None, None]
    rad = np.zeros((2, _H - 1, _W))
    for kh in range(2):
        mid += np.einsum("oi,ihw->ohw", k[:, :, kh], centre[:, kh:kh + _H - 1])
        rad += np.einsum("oi,ihw->ohw", np.abs(k[:, :, kh]),
                         radius[:, kh:kh + _H - 1])
    if not np.all((mid - rad < 0) & (mid + rad > 0) & (mid != 0)):
        pytest.fail("test data: every ReLU input must cross zero with ends "
                    "of different size")
    out_chw = (2, _H - 1, _W - 2)
    nodes = [helper.make_node("Conv", ["X", "K1", "c1"], ["A"]),
             helper.make_node("Cast", ["A"], ["B"], to=TensorProto.FLOAT),
             helper.make_node("Relu", ["B"], ["Z"]),
             helper.make_node("Conv", ["Z", "K2", "c2"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, _C, _H, _W], [1, *out_chw],
                     inits=dict(K1=_K1, c1=_C1, K2=_K2, c2=_C2), name="ccrc")
    what = "Conv, Cast(FLOAT), ReLU, Conv on ImageStar, precompute='ibp'"
    sets = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                       "approx", what, input_shape=(_C, _H, _W),
                       precompute_bounds="ibp")
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, _C, _H, _W]),
                   what, out_chw=out_chw)


# --------------------------------------------------------------------------- #
# Casts between the two float types                                            #
# --------------------------------------------------------------------------- #
# Not dyadic on purpose, and narrower than the float32 spacing in x0: a
# float64 value cast to float32 is rounded, which the real-valued reading of
# DESIGN section 1 treats as the identity.
_F64_LO = np.array([0.1, -1.5, 0.3])
_F64_HI = np.array([0.1 + 1e-9, 0.7, 0.3])


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_double_input_cast_to_float_is_the_identity_or_refused(tmp_path, kind):
    """A float64 graph input cast to FLOAT (narrowing): under the owner's
    real-valued reading (decision of 2026-11-13; DESIGN section 1) the network
    is the identity on values, so every input point itself, in float64, must
    be a member of the result, decided exactly in rationals with no slack, or
    the reach refuses. The points are clipped into the box (``samples`` on
    non-dyadic bounds can land a rounding step outside it). Catches a
    narrowing Cast answered by rounding the set to float32 (an escape of
    about 1e-8, which the LP check with its tolerance would forgive) or by
    moving the set; the onnxruntime value, rounded to float32, is not the
    reference here by that decision."""
    path = save_onnx(tmp_path, [helper.make_node("Cast", ["X"], ["Y"],
                                                 to=TensorProto.FLOAT)],
                     [1, 3], [1, 3], in_type="DOUBLE", name="d2f")
    what = f"Cast DOUBLE to FLOAT on {kind}"
    model = load_onnx(path)
    try:
        sets = reach_model(model, flat_set(kind, _F64_LO, _F64_HI), "approx",
                           what)
    except NotImplementedError:
        return
    members = [R.Membership(R.star(s)) for s in sets]
    points = [np.clip(x, _F64_LO, _F64_HI) for x in samples(_F64_LO, _F64_HI)]
    missed = [x for x in points if not any(m(x) for m in members)]
    assert not missed, (
        f"{what}: {len(missed)} of {len(points)} input points are not exact "
        f"members of the result, first {missed[0].tolist()}")


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box", "Hexatope", "Octatope"])
def test_float_input_cast_to_double_holds_every_onnxruntime_output(tmp_path,
                                                                   kind):
    """A float32 graph input cast to DOUBLE (widening) is the identity on
    values in every reading: every onnxruntime output (float64) must be a
    member of the result. Catches a widening Cast that moves or rescales the
    set, in each of the five flat lanes, through the model entry."""
    lo, hi = _SIGNED_LO, _SIGNED_HI
    path = save_onnx(tmp_path, [helper.make_node("Cast", ["X"], ["Y"],
                                                 to=TensorProto.DOUBLE)],
                     [1, lo.size], [1, lo.size], out_type="DOUBLE", name="f2d")
    what = f"Cast FLOAT to DOUBLE on {kind}"
    sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx", what)
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, lo.size]), what)


# --------------------------------------------------------------------------- #
# Casts spelled as tensor methods in a torch model                              #
# --------------------------------------------------------------------------- #
_METHOD_CASTS = {
    "to(int32)": lambda h: h.to(torch.int32).double(),
    "type(int64)": lambda h: h.type(torch.int64).double(),
}


class _CastBetweenLinears(nn.Module):
    """fc1 (4 -> 4), a cast to an integer type spelled as a tensor method and
    ``.double()`` (so that the network is legal), fc2 (4 -> 3); dyadic
    weights, so fc1's outputs have fractional parts that the cast truncates.
    The replay runs the module in float64."""

    def __init__(self, spelling):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(4, 4), nn.Linear(4, 3)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(
                [[1, -2, 1, 0], [2, 1, 0, -1], [-1, 1, 2, 1], [0, -1, 1, 2]]) / 4)
            self.fc1.bias.copy_(torch.tensor([0.25, -0.5, 0.125, 0.375]))
            self.fc2.weight.copy_(torch.tensor(
                [[1, 0, -1, 2], [-2, 1, 1, 0], [0, 2, 1, -1]]) / 2)
            self.fc2.bias.copy_(torch.tensor([0.5, -0.25, 0.0]))
        self.cast = _METHOD_CASTS[spelling]

    def forward(self, x):
        return self.fc2(self.cast(self.fc1(x)))


@pytest.mark.parametrize("kind,spelling", [
    ("Zono", "to(int32)"), ("Octatope", "type(int64)")])
def test_tensor_method_cast_in_a_torch_model_refuses_or_holds_every_output(
        kind, spelling):
    """A torch model whose cast to an integer type is a tensor method,
    reached with ``reach_pytorch_model``: ``_is_identity_method`` passes
    ``to``/``type`` with a float32 or float64 dtype as the identity and must
    refuse every other dtype. The reach refuses, or every float64 torch output
    is a member. Catches a dtype test that accepts every dtype for ``to`` or
    ``type`` (a truncating cast then passes as the identity), on set types
    whose affine maps are exact (a Box's interval hull after two layers is
    wide enough to hold the truncated outputs). The value-keeping spellings
    are cells of ``test_soundness_graph_walk.py``."""
    lo, hi = (4 * v for v in flat_box(4, seed=5))
    model = _CastBetweenLinears(spelling).eval()
    what = f"x.{spelling} between two Linear layers on {kind}"
    try:
        sets = reach_model(model, flat_set(kind, lo, hi), "approx", what)
    except NotImplementedError:
        return
    assert_covered(sets, replay_torch(model, samples(lo, hi), [1, 4]), what)
