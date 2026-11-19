"""Soundness cells for ONNX Neg (``OnnxNeg``) on every set type.

Cells:

- Neg through ``reach_layer``, the dispatcher's ``_neg_sets_star`` (Star and
  ImageStar), ``_neg_sets_zono`` (Zono and ImageZono), ``_neg_sets_box``,
  ``_neg_sets_hexatope`` and ``_neg_sets_octatope``: one input set, several
  input sets at once (the box cut into three parts of different sizes) and a
  Star or ImageStar with a binding constraint row that is not a box row, on
  data scaled by 8 (bounds well beyond plus and minus 1, widths above 2).
  The kernel never reads ``method``: a flat Star runs under ``'exact'`` (the
  stronger check) for one and several sets, under both methods for the
  binding row; Zono, Box, Hexatope and Octatope run under ``'approx'``, the
  only method the model entry lets them use.
- A ``reach_layer`` list whose sets are of two families (registry REACH-6):
  the dispatcher picks the Hexatope lane by the first set, and the Hexatope
  helper maps an ImageZono through its own ``affine_map``, which returns a
  flat Zono in the stored order (marked), while an ImageStar comes back as an
  ImageStar (its passing neighbour). The orders that crash are not tested
  here.
- The three-node chain affine, Neg, affine through ``reach_pytorch_model``
  on an ONNX graph: Gemm, Neg, Gemm for the five flat kinds; Conv, Neg, Conv
  with asymmetric kernels for the two image kinds; an exact ReLU in front of
  the Neg so that several Stars reach it at once; and the bound pre-pass in
  front of a ReLU that reads the Neg's output (only the Star and ImageStar
  ReLU read the recorded bounds), which exercises the pre-pass rule for
  ``OnnxNeg``: ``precompute_bounds='ibp'`` and ``'zono'`` on a flat Star, and
  ``'ibp'`` on an ImageStar whose ReLU inputs all cross zero.

Rows that refuse, with their cells in other files: ``method='exact'`` on a
Zono, ImageZono, Box, Hexatope or Octatope at the model entry (``ValueError``
before any layer; section 11 of ``test_soundness_graph_walk.py``), and a
negation spelled in a torch model (``x.neg()``, ``-x``, ``torch.neg``), which
the graph walker refuses (the unsupported-node cells of
``test_graph_ops_regressions.py``).

Oracle. For flat sets the exact image in rationals
(``tests/_oracles/rational.py``): the input box, or the input polytope in
predicate space, mapped by ``x -> -x``; every vertex of the image of each
input part is a member of one returned set, with zero slack (``escapes``,
``straddlers``), and for ``method='exact'`` every vertex of every returned set
is a point of the image (``unreached``). On every cell, the true outputs of
sample points (torch in float64 for a layer, onnxruntime for a model; the
data are dyadic, so both are exact) are members of a returned set by the LP
on the set's own rows (``assert_covered``). For an image set and
``method='exact'``, corners of every returned set mapped back by ``-y`` must
lie in an input set (the other direction, by the same LP).

Layout. A flat set is in the tensor's row-major order. An image set stores
height, width, channel; image outputs are compared with ``out_chw``, which
also checks the stored height, width and channel count of every result.
"""

import numpy as np
import pytest
from onnx import helper
from onnx2torch.node_converters.neg import OnnxNeg

from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests._oracles.membership import rows, uncovered
from tests.cell_support import (FLAT_KINDS, assert_covered,
                                constrained_image_star, constrained_star,
                                flat_box, flat_set, image_set, pixel_box,
                                reach_model, reach_one_layer, replay_onnx,
                                replay_torch, samples, save_onnx)
from tests.known_bugs import known_bug

# A flat input of four entries: entry 0 straddles zero, entry 1 is positive,
# entry 2 has zero width at a nonzero value, entry 3 is negative. The box is
# not symmetric about the origin, so a Neg that returns its input, or shifts
# it, leaves the true outputs out.
_N = 4
_LO, _HI = flat_box(_N, seed=3, zero_width=True)

# An image with two channels and height different from width, a different
# dyadic interval per pixel.
_C, _H, _W = 2, 3, 5
_ILO, _IHI = pixel_box(_C, _H, _W, seed=5)

# The layer cells read the same boxes scaled by 8: bounds well beyond plus and
# minus 1 and widths above 2, so a result that read one predicate variable in
# place of another would no longer hold the outputs by accident.
_SCALE = 8.0


def _parts(lo, hi, shares=(1, 2, 5)):
    """The box [lo, hi] cut along its first coordinate of positive width
    into parts of the given shares (of their sum, a power of two), as
    ``(lo, hi)`` pairs: dyadic when ``lo`` and ``hi`` are. Several input sets
    of different sizes, so a kernel that answers only the first, or answers
    the first set for every set, misses most outputs."""
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


def _negation(n):
    return [[-int(i == j) for j in range(n)] for i in range(n)]


def _polytope_of(star):
    """The exact pieces of a Star built by ``cell_support`` (not by n2v):
    its predicate polytope ``C a <= d, lb <= a <= ub`` mapped by
    ``a -> c + G a``, read from the stored arrays in rationals."""
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


# Neg keeps the predicate variables of its input (at most 4 here), so a result
# with more is wrong, and enumerating its vertices could run for hours.
_MAX_ENUMERATED = 8


def _assert_exact_negation(input_pieces, result, exact, what):
    """Every vertex of the image of each input piece under ``x -> -x`` is in
    one returned set (zero slack), and with ``exact`` every vertex of every
    returned set is a point of that image."""
    image = R.affine(input_pieces, _negation(len(input_pieces[0].W)))
    computed = [R.star(s) for s in result]
    if exact and any(len(s[4]) > _MAX_ENUMERATED for s in computed):
        raise AssertionError(
            f"{what}: a returned set has more than {_MAX_ENUMERATED} predicate "
            f"variables, more than the input; its vertices are not enumerated")
    escaped = R.escapes(image, *computed)
    assert not escaped, (
        f"{what}: {len(escaped)} vertices of the exact image of Neg are in "
        f"none of the {len(result)} returned sets, first {escaped[0]}")
    spread = R.straddlers(image, *computed)
    assert not spread, (
        f"{what}: the image of input part(s) {spread} is spread over several "
        f"returned sets, while Neg maps each input set to one set")
    if exact:
        for k, s in enumerate(computed):
            far = R.unreached(image, R.set_vertices(s))
            assert not far, (
                f"{what}: returned set {k} has vertices that no input point "
                f"maps to, first {far[0][1]}")


def _corners(s, n, seed):
    """Up to ``n`` points of the set ``s``: corners of its predicate box
    that satisfy its rows, mapped through its stored basis (read with
    ``membership.rows``, no method of the set). Stored order."""
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
# Neg through reach_layer, flat sets                                            #
# --------------------------------------------------------------------------- #
_FLAT_CELLS = (
    [(kind, "approx", inputs) for kind in FLAT_KINDS if kind != "Star"
     for inputs in ("one set", "several sets")]
    + [("Star", "exact", "one set"), ("Star", "exact", "several sets"),
       ("Star", "approx", "binding row"), ("Star", "exact", "binding row")])


@pytest.mark.parametrize("kind,method,inputs", _FLAT_CELLS)
def test_neg_layer_on_flat_sets_is_the_exact_negation(kind, method, inputs):
    """Neg through ``reach_layer`` on every flat kind. Catches a Neg that
    returns its input (a weight of +I in the Hexatope helper, an unused
    negated basis), a shift (a helper linear layer built with a random bias),
    a box built from one bound, a loop that returns after the first input set
    or answers the first set for every set, and a negated constraint matrix
    (which mirrors the predicate polytope: a box Star does not show it, the
    binding row does). With ``method='exact'`` the result must also hold
    nothing beyond the image."""
    what = f"Neg on {inputs} of {kind} ({method})"
    lo, hi = _SCALE * _LO, _SCALE * _HI
    if inputs == "binding row":
        star, points = constrained_star(lo, hi, seed=2)
        sets, pieces = [star], _polytope_of(star)
    else:
        boxes = [(lo, hi)] if inputs == "one set" else _parts(lo, hi)
        sets = [flat_set(kind, a, b) for a, b in boxes]
        pieces = [p for a, b in boxes for p in R.box(a, b)]
        points = samples(lo, hi)
    result = reach_one_layer(OnnxNeg(), sets, method, what)
    if len(result) != len(sets):
        raise AssertionError(f"{what}: {len(result)} sets returned for "
                             f"{len(sets)}; Neg maps each set to one set")
    _assert_exact_negation(pieces, result, method == "exact", what)
    assert_covered(result, replay_torch(OnnxNeg(), points, [1, _N]), what)


# --------------------------------------------------------------------------- #
# Neg through reach_layer, image sets                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,method,inputs", [
    ("ImageStar", "exact", "one set"),
    ("ImageStar", "approx", "several sets"),
    ("ImageStar", "exact", "binding row"),
    ("ImageZono", "approx", "one set"),
    ("ImageZono", "approx", "several sets"),
])
def test_neg_layer_on_image_sets_holds_every_output(kind, method, inputs):
    """Neg through ``reach_layer`` on a (2, 3, 5) ImageStar or ImageZono with
    a different interval per pixel. Catches an image branch that returns its
    input (the basis kept, or a zonotope's centre kept), an ImageStar branch
    that negates only the centre (right on a box, wrong with the binding
    row), an ImageZono branch that returns a flat Zono in the stored order, a
    loop over the input sets that returns early, and a result whose stored
    height, width or channel count is not the input's. With
    ``method='exact'``, corners of every returned set must come from an input
    point."""
    what = f"Neg on {inputs} of {kind} ({method})"
    lo, hi = _SCALE * _ILO, _SCALE * _IHI
    if inputs == "binding row":
        star, points = constrained_image_star(lo, hi, seed=4)
        sets = [star]
    else:
        boxes = [(lo, hi)] if inputs == "one set" else _parts(lo, hi)
        sets = [image_set(kind, a, b) for a, b in boxes]
        points = samples(lo, hi)
    result = reach_one_layer(OnnxNeg(), sets, method, what)
    if len(result) != len(sets):
        raise AssertionError(f"{what}: {len(result)} sets returned for "
                             f"{len(sets)}; Neg maps each set to one set")
    assert_covered(result, replay_torch(OnnxNeg(), points, [1, _C, _H, _W]),
                   what, out_chw=(_C, _H, _W))
    if method == "exact":
        back = [-y for k, s in enumerate(result) for y in _corners(s, 24, k)]
        missed = uncovered(sets, back)
        assert not missed, (
            f"{what}: {len(missed)} of {len(back)} corners of the result have "
            f"no preimage in the input (largest distance "
            f"{max(m[1] for m in missed):.3g})")


# --------------------------------------------------------------------------- #
# The chain affine, Neg, affine through the model entry point                   #
# --------------------------------------------------------------------------- #
_W1 = np.array([[1, -2, 1, 0], [2, 1, 0, -1], [-1, 1, 2, 1], [0, -1, 1, 2]],
               np.float32) / 4
_B1 = np.array([0.25, -0.5, 0.125, 0.75], np.float32)
_W2 = np.array([[1, 0, -1, 2], [-2, 1, 1, 0], [0, 2, 1, -1]], np.float32) / 2
_B2 = np.array([0.5, -0.25, 0.0], np.float32)
_FLAT_INITS = dict(W1=_W1, b1=_B1, W2=_W2, b2=_B2)


def _gemm(x, w, b, y):
    return helper.make_node("Gemm", [x, w, b], [y], transB=1)


def _flat_chain(tmp_path, middle, name):
    """X (1, 4) -> Gemm -> ``middle`` ops -> Gemm -> Y (1, 3)."""
    nodes = [_gemm("X", "W1", "b1", "A")] + middle + [_gemm("Z", "W2", "b2", "Y")]
    return save_onnx(tmp_path, nodes, [1, _N], [1, 3], inits=_FLAT_INITS,
                     name=name)


@pytest.mark.parametrize("kind,method", [
    ("Box", "approx"), ("Star", "approx"), ("Star", "exact"),
    ("Zono", "approx"), ("Hexatope", "approx"), ("Octatope", "approx")])
def test_gemm_neg_gemm_holds_every_onnxruntime_output(tmp_path, kind, method):
    """Gemm, Neg, Gemm through ``reach_pytorch_model``: the next Gemm consumes
    the rows the Neg emits, so a wrong row or a wrong sign of the negated set
    shows in the model output. Catches the same Neg bugs as the layer cell,
    through the graph walker."""
    path = _flat_chain(tmp_path, [helper.make_node("Neg", ["A"], ["Z"])], "gnn")
    what = f"Gemm, Neg, Gemm on {kind} ({method})"
    sets = reach_model(load_onnx(path), flat_set(kind, _LO, _HI), method, what)
    assert_covered(sets, replay_onnx(path, samples(_LO, _HI), [1, _N]), what)


def test_exact_relu_hands_several_stars_to_neg(tmp_path):
    """Gemm, ReLU, Neg, Gemm with ``method='exact'``: the exact ReLU splits the
    input Star, so the Neg node receives a list of several Stars. Catches a
    Neg loop that returns after the first Star or negates one Star for all."""
    path = _flat_chain(tmp_path, [helper.make_node("Relu", ["A"], ["R"]),
                                  helper.make_node("Neg", ["R"], ["Z"])], "grnn")
    what = "Gemm, ReLU, Neg, Gemm on Star (exact)"
    sets = reach_model(load_onnx(path), flat_set("Star", _LO, _HI), "exact", what)
    if len(sets) < 2:
        pytest.fail(f"{what}: the exact ReLU returned {len(sets)} set; the cell "
                    f"needs several sets at the Neg node")
    assert_covered(sets, replay_onnx(path, samples(_LO, _HI), [1, _N]), what)


@pytest.mark.parametrize("kind,precompute", [("Star", "ibp"), ("Star", "zono")])
def test_neg_before_relu_with_the_bound_prepass(tmp_path, kind, precompute):
    """Gemm, Neg, ReLU, Gemm on a Star with ``precompute_bounds``: the
    interval pre-pass carries its own rule for ``OnnxNeg`` (``-ub, -lb``),
    the zonotope pre-pass runs the reach's own Zono Neg, and the Star ReLU
    reads the recorded bounds. Catches an interval rule that keeps the
    bounds where they are (they then overlap the true ones and narrow a
    crossing neuron's triangle) and a Zono Neg that keeps its centre (the
    zonotope pass then records the bounds of x for -x). A rule that swaps the
    ends without negating them, or negates them without swapping, gives
    crossed intervals that the ReLU always drops, so it has no effect. The
    Box and Zono ReLU never read recorded bounds; their chains through the
    Neg are the Gemm, Neg, Gemm cells."""
    path = _flat_chain(tmp_path, [helper.make_node("Neg", ["A"], ["N"]),
                                  helper.make_node("Relu", ["N"], ["Z"])], "gnrg")
    what = f"Gemm, Neg, ReLU, Gemm on {kind} with precompute_bounds={precompute!r}"
    sets = reach_model(load_onnx(path), flat_set(kind, _LO, _HI), "approx",
                       what, precompute_bounds=precompute)
    assert_covered(sets, replay_onnx(path, samples(_LO, _HI), [1, _N]), what)


# Conv weights: dyadic, asymmetric kernels, mixed signs, mixing the channels.
_K1 = np.array([[[[1], [-2]], [[1], [1]]],
                [[[2], [1]], [[-1], [0]]]], np.float32) / 4          # (2,2,2,1)
_C1 = np.array([0.25, -0.5], np.float32)
_K2 = np.array([[[[1, -1, 2]], [[0, 1, -1]]],
                [[[-2, 1, 0]], [[1, 1, 1]]]], np.float32) / 4        # (2,2,1,3)
_C2 = np.array([-0.25, 0.5], np.float32)
_IMAGE_INITS = dict(K1=_K1, c1=_C1, K2=_K2, c2=_C2)
_OUT_CHW = (2, _H - 1, _W - 2)


def _image_chain(tmp_path, middle, name):
    """X (1, 2, 3, 5) -> Conv (2x1 kernel) -> ``middle`` -> Conv (1x3
    kernel) -> Y (1, 2, 2, 3)."""
    nodes = ([helper.make_node("Conv", ["X", "K1", "c1"], ["A"])] + middle
             + [helper.make_node("Conv", ["Z", "K2", "c2"], ["Y"])])
    return save_onnx(tmp_path, nodes, [1, _C, _H, _W], [1, *_OUT_CHW],
                     inits=_IMAGE_INITS, name=name)


@pytest.mark.parametrize("kind,method,inputs", [
    ("ImageStar", "approx", "box"), ("ImageStar", "exact", "box"),
    ("ImageStar", "approx", "binding row"), ("ImageZono", "approx", "box")])
def test_conv_neg_conv_holds_every_onnxruntime_output(tmp_path, kind, method,
                                                      inputs):
    """Conv, Neg, Conv on a (2, 3, 5) image through ``reach_pytorch_model``
    (with ``input_shape``, as the competition runner calls it). The second
    Conv reads the negated set with the stored image shape, so a Neg that
    returns an image of another height, width or channel count, or negates
    the centre and not the generators, misses outputs."""
    path = _image_chain(tmp_path, [helper.make_node("Neg", ["A"], ["Z"])], "cnc")
    what = f"Conv, Neg, Conv on {kind} {inputs} ({method})"
    if inputs == "binding row":
        input_set, points = constrained_image_star(_ILO, _IHI, seed=6)
    else:
        input_set, points = image_set(kind, _ILO, _IHI), samples(_ILO, _IHI)
    sets = reach_model(load_onnx(path), input_set, method, what,
                       input_shape=(_C, _H, _W))
    assert_covered(sets, replay_onnx(path, points, [1, _C, _H, _W]), what,
                   out_chw=_OUT_CHW)


def _crossing_pixel_box(seed):
    """(C, H, W) bounds whose every pixel crosses zero: lower ends in
    [-3, -1], upper ends in [1, 3], quarter steps, each pixel its own."""
    rng = np.random.default_rng(seed)
    return (-(4 + rng.integers(0, 9, size=(_C, _H, _W))) / 4.0,
            (4 + rng.integers(0, 9, size=(_C, _H, _W))) / 4.0)


def _conv1_bounds(lo, hi):
    """The exact interval of every output of the first Conv (2x1 kernel) over
    the box, in (C, H, W) order, by numpy on dyadic data."""
    centre, radius = (lo + hi) / 2, (hi - lo) / 2
    k = _K1[:, :, :, 0].astype(np.float64)
    mid = np.zeros((2, _H - 1, _W)) + _C1.astype(np.float64)[:, None, None]
    rad = np.zeros((2, _H - 1, _W))
    for kh in range(2):
        mid += np.einsum("oi,ihw->ohw", k[:, :, kh], centre[:, kh:kh + _H - 1])
        rad += np.einsum("oi,ihw->ohw", np.abs(k[:, :, kh]),
                         radius[:, kh:kh + _H - 1])
    return mid - rad, mid + rad


def test_conv_neg_relu_conv_with_the_ibp_prepass_on_an_image(tmp_path):
    """Conv, Neg, ReLU, Conv on a (2, 3, 5) ImageStar with
    ``precompute_bounds='ibp'``: the pre-pass negates the bounds of an image
    in its own channel-first order, keeps the image shape, and hands them to
    the ReLU in the stored (height, width, channel) order. Every ReLU input
    crosses zero with ends of different size, so wrong bounds overlap the
    true ones, are not dropped, and narrow the ReLU's triangle. Catches an
    interval rule for ``OnnxNeg`` that keeps the bounds where they are, and
    one that drops the image shape (the bounds are then recorded in the
    channel-first order and land on other neurons)."""
    lo, hi = _crossing_pixel_box(seed=1)
    pre_lo, pre_hi = _conv1_bounds(lo, hi)
    if not np.all((-pre_hi < 0) & (-pre_lo > 0) & (pre_lo != -pre_hi)):
        pytest.fail("test data: every ReLU input must cross zero with ends "
                    "of different size")
    path = _image_chain(tmp_path, [helper.make_node("Neg", ["A"], ["N"]),
                                   helper.make_node("Relu", ["N"], ["Z"])],
                        "cnrc")
    what = "Conv, Neg, ReLU, Conv on ImageStar with precompute_bounds='ibp'"
    sets = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                       "approx", what, input_shape=(_C, _H, _W),
                       precompute_bounds="ibp")
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, _C, _H, _W]),
                   what, out_chw=_OUT_CHW)


# --------------------------------------------------------------------------- #
# A reach_layer list of two set families                                        #
# --------------------------------------------------------------------------- #
_LANE_BY_FIRST_SET = known_bug(
    7, "REACH-6: reach_layer picks the kernel by the first set of the list; "
       "Neg on a Hexatope followed by an ImageZono maps the ImageZono "
       "through the Hexatope helper's affine map, which returns a flat Zono "
       "in the stored (height, width, channel) order, so every true output "
       "of that set lies outside it in the tensor's order")


@pytest.mark.parametrize("first,second", [
    ("Hexatope", "ImageStar"),
    pytest.param("Hexatope", "ImageZono", marks=_LANE_BY_FIRST_SET)])
def test_neg_layer_on_a_list_mixing_set_families_refuses_or_holds_every_output(
        first, second):
    """``reach_layer`` of Neg on a Hexatope of a (2, 3, 5) tensor followed by
    an image set of the same tensor: the call refuses (``NotImplementedError``
    or ``ValueError``), or each returned set holds every true output in the
    tensor's own layout. Catches a lane that hands a later set of another
    family to a kernel written for the first one and returns it in the wrong
    layout (the ImageZono order, marked) or un-negated."""
    lo, hi = _SCALE * _ILO, _SCALE * _IHI
    sets = [flat_set(first, lo.reshape(-1), hi.reshape(-1)),
            image_set(second, lo, hi)]
    what = f"Neg on [{first}, {second}]"
    try:
        result = reach_one_layer(OnnxNeg(), sets, "approx", what)
    except (NotImplementedError, ValueError):
        return
    if len(result) != len(sets):
        raise AssertionError(f"{what}: {len(result)} sets returned for "
                             f"{len(sets)}")
    outputs = replay_torch(OnnxNeg(), samples(lo, hi), [1, _C, _H, _W])
    for k, s in enumerate(result):
        assert_covered([s], outputs, f"{what}, set {k}", out_chw=(_C, _H, _W))
