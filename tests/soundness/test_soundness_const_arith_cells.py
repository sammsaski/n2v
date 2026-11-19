"""Soundness cells for ONNX Add, Sub, Mul and Div with one constant operand.

The code under test is case 2 of ``_handle_onnx_binary_op`` in
``n2v/nn/reach.py`` (a computed set and a parameter, in either order, with
ONNX broadcasting through the recorded tensor shape, the row expansion of a
broadcast that adds entries, and the tile fallback when no shape is
recorded), ``_mul_sets_by_constant`` and ``_div_sets_by_constant``, for
Star, Zono, Box, Hexatope, Octatope, ImageStar and ImageZono. Every model is
an ONNX graph built with ``cell_support.save_onnx`` and loaded with
``load_onnx``; every set goes through the model entry point. A torch model
whose forward adds or multiplies by a constant has no reach rule today and
is refused (the last cell); the ``OnnxBinaryMathOperation`` handed to
``reach_layer`` alone is refused for every set type
(``test_soundness_residual_add.py``, ``test_soundness_mul_div.py``).

The grids keep one cell per code path: the broadcast of the constant runs
before the set type and the operation are looked at, so its layouts run on
the Star; each type's kernel runs with a full-size constant whose entries
differ and have both signs; Sub negates the constant (or, with the constant
first, the set) and Div takes the reciprocal before the Add and Mul branches,
so they run where their own lines matter. A known defect keeps one marked
cell per failing code path beside a passing neighbour.

Oracle. The image of a set under an operation with a constant is affine, so
the true outputs are computed without n2v in two ways. First in exact
rationals (``fractions.Fraction`` with numpy broadcasting on object arrays,
and a direct sum for the convolutions of the composition cells), decided by
exact membership with no slack (``rational.Membership``). For a flat input
every vertex of the input set is checked this way, which proves that the
returned set contains the whole image (the image of a polytope under an
affine map is the hull of the images of its vertices); the exact-method
cells check both directions against the exact pieces. Second, onnxruntime
replay of the float32 model on 64 member points (128 on the exact cells),
corners first, decided by one LP on the rows of every returned set
(``cell_support.assert_covered``); the torch cell replays the torch model in
float64. The data are dyadic (inputs and weights multiples of small powers
of two, divisors powers of two), so float32, float64 and the rationals agree
and no slack beyond ``LP_TOL`` is used. The rounding cells and the cells of
a float64 constant that float32 cannot hold use float64 models and exact
fractions only (every corner of the box).

Layout. A network tensor of an image is (C, H, W); an image set stores
(H, W, C); a flat set keeps the tensor's own order. Exact outputs are
computed in the network's order and permuted to the stored order with
``hwc_index``; replayed outputs are converted by ``assert_covered`` with
``out_chw``. Images are (2, 3, 5) with a different interval per pixel and
constants whose entries are all different, so an entry added to or scaled
by the wrong position is seen.
"""

import itertools
import warnings
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn.neural_network import NeuralNetwork
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (FLAT_KINDS, IMAGE_KINDS, assert_covered,
                                constrained_image_star, constrained_star,
                                flat_box, flat_set, hwc_index, image_set,
                                pixel_box, reach_model, replay_onnx,
                                replay_torch, samples, save_onnx)
from tests.known_bugs import DID_NOT_RAISE, known_bug


def _reach(model, input_set, method, what, **options):
    """``cell_support.reach_model``, with its empty-result check reported by
    ``pytest.fail``: an ``AssertionError`` there is not the failure a strict
    expected-failure marker of these cells documents, so no marker may
    absorb it. A refusal (``NotImplementedError``) and any other exception
    pass through."""
    try:
        return reach_model(model, input_set, method, what, **options)
    except AssertionError as error:
        pytest.fail(str(error))


def _one_set(sets, what):
    """The handler translates or scales each input set into one output set,
    so ``approx`` on one input set returns one set. More sets would only make
    the coverage check slow (one LP per point and set) until the gate's time
    limit, where a failure is due: reported by ``pytest.fail``, which no
    marker of these cells absorbs."""
    if len(sets) != 1:
        pytest.fail(f"{what}: {len(sets)} sets returned for one input set")
    return sets


OPS = ("Add", "Sub", "Mul", "Div")
_NUMPY = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply,
          "Div": np.true_divide}
_IMAGE = (2, 3, 5)


# --------------------------------------------------------------------------- #
# Exact references                                                           #
# --------------------------------------------------------------------------- #
def _rationals(a):
    """The exact rationals the float entries of ``a`` are, as an object array
    of ``a``'s shape."""
    a = np.asarray(a, dtype=np.float64)
    out = np.empty(a.shape, dtype=object)
    for index in np.ndindex(a.shape):
        out[index] = Fraction(float(a[index]))
    return out


def _exact(op, x, k, constant_first=False):
    """``op(x, k)``, or ``op(k, x)`` when ``constant_first``, in exact
    rationals with ONNX (numpy) broadcasting of the two tensor shapes, as a
    flat list in row-major order. ``x`` and ``k`` are arrays (or object
    arrays of rationals) of their tensor shapes."""
    a = x if np.asarray(x).dtype == object else _rationals(x)
    b = k if np.asarray(k).dtype == object else _rationals(k)
    if constant_first:
        a, b = b, a
    return list(np.asarray(_NUMPY[op](a, b), dtype=object).reshape(-1))


def _matmul_exact(x, W, b=None):
    """``x @ W + b`` in exact rationals for a flat ``x`` (ONNX MatMul of a
    (1, n) tensor with an (n, m) weight)."""
    x, W = _rationals(x).reshape(-1), _rationals(W)
    out = [sum((x[i] * W[i, j] for i in range(len(x))), Fraction(0))
           for j in range(W.shape[1])]
    if b is not None:
        out = [v + bj for v, bj in zip(out, _rationals(b).reshape(-1))]
    return np.array(out, dtype=object)


def _conv_exact(x, K, b, pads):
    """A stride-1 ONNX Conv of a (C, H, W) tensor in exact rationals.
    ``pads`` is ONNX's [top, left, bottom, right]; the result is
    (O, H', W')."""
    x = x if np.asarray(x).dtype == object else _rationals(x)
    K, b = _rationals(K), _rationals(b)
    top, left, bottom, right = pads
    channels, height, width = x.shape
    padded = np.full((channels, height + top + bottom, width + left + right),
                     Fraction(0), dtype=object)
    padded[:, top:top + height, left:left + width] = x
    outs, _, kh, kw = K.shape
    oh, ow = padded.shape[1] - kh + 1, padded.shape[2] - kw + 1
    y = np.empty((outs, oh, ow), dtype=object)
    for o, r, c in itertools.product(range(outs), range(oh), range(ow)):
        y[o, r, c] = b[o] + sum(
            (K[o, ch, i, j] * padded[ch, r + i, c + j]
             for ch in range(channels) for i in range(kh) for j in range(kw)),
            Fraction(0))
    return y


def _is_image(s):
    return isinstance(s, (ImageStar, ImageZono))


def _assert_exactly_covered(sets, outputs, what, out_chw=None):
    """Every exact output (a flat sequence of rationals in the network's
    order) is a member of at least one of ``sets``, decided in exact
    arithmetic with no slack. With ``out_chw=(C, H, W)`` an image set must
    store exactly that image and is read in its (H, W, C) order; a flat set
    is read in the tensor's own order. A set with the wrong number of entries
    fails."""
    order = None if out_chw is None else hwc_index(*out_chw)
    holders = []
    for s in sets:
        if _is_image(s):
            if out_chw is None:
                raise AssertionError(
                    f"{what}: an image set came back for a flat output")
            channels, height, width = out_chw
            if (s.height, s.width, s.num_channels) != (height, width,
                                                       channels):
                raise AssertionError(
                    f"{what}: the result stores {s.height}x{s.width}x"
                    f"{s.num_channels}, the output is (C, H, W) = {out_chw}")
            size = height * width * channels
        else:
            size = s.dim
        holders.append((rational.Membership(rational.star(s)), _is_image(s),
                        size))
    for i, y in enumerate(outputs):
        y = list(y)
        hit = False
        for member, image, size in holders:
            if size != len(y):
                raise AssertionError(
                    f"{what}: a returned set has {size} entries, the true "
                    f"output has {len(y)}")
            if member([y[j] for j in order] if image else y):
                hit = True
                break
        if not hit:
            shown = ", ".join(f"{float(v):g}" for v in y[:6])
            raise AssertionError(
                f"{what}: the exact output of point {i} ({shown}, ...) is in "
                f"none of the {len(sets)} returned sets")


def _corners(lo, hi):
    """Every corner of the box [lo, hi] as a flat array, each once."""
    lo, hi = np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)
    return [np.array(c) for c in sorted(set(itertools.product(*zip(lo, hi))))]


def _constant(op, shape, seed=0):
    """A float32 constant of ``shape`` with dyadic entries of both signs and
    magnitudes below and above 1. For Add, Sub and Mul the entries are all
    different (odd multiples of 1/4 from 3/4 up); a divisor holds powers of
    two only, so every quotient of dyadic data is dyadic and the reciprocal a
    Star or Zono is scaled by is exact. A single entry is chosen so that a
    scale applied the wrong way round (multiplied where it divides, or the
    reverse) changes the result: |c| > 1 for Mul, |c| < 1 for Div."""
    n = int(np.prod(shape, dtype=int))
    if n == 1:
        value = {"Add": 1.25, "Sub": -0.75, "Mul": -2.5, "Div": -0.5}[op]
        return np.full(shape, value, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if op == "Div":
        pool = np.array([0.25, -0.5, 2.0, -4.0, 0.5, -0.25, 4.0, -2.0])
        values = np.resize(rng.permutation(pool), n)
    else:
        values = (2 * np.arange(n) + 3) / 4.0 * np.resize([1.0, -1.0, -1.0,
                                                           1.0], n)
        values = rng.permutation(values)
    return values.reshape(shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# A flat set and a constant                                                  #
# --------------------------------------------------------------------------- #
# name: (input tensor shape, constant shape). "middle" broadcasts along an
# axis that is not the last; "expanding" adds entries to the set.
_FLAT_LAYOUTS = {
    "scalar": ((1, 2, 3), ()),
    "trailing": ((1, 2, 3), (3,)),
    "middle": ((1, 2, 3), (2, 1)),
    "full": ((1, 2, 3), (1, 2, 3)),
    "expanding": ((1, 2, 1), (3,)),
}


def _refused(kind, op, expanding, constant_first):
    """The cells the library refuses by design (``NotImplementedError``):
    division by a computed set; Mul and Div, and the negation that Sub(K, X)
    needs, on a Hexatope or Octatope (no scaling rule); a broadcast that adds
    entries to a Hexatope or Octatope (no row selection)."""
    if constant_first and op == "Div":
        return True
    if kind in ("Hexatope", "Octatope"):
        return (op in ("Mul", "Div") or (constant_first and op == "Sub")
                or expanding)
    return False


def _flat_cell(tmp_path, kind, op, in_shape, k_shape, constant_first=False,
               seed=0):
    """One flat cell: ``op`` between a flat ``kind`` box set of tensor shape
    ``in_shape`` (with ``input_shape`` given) and a constant of ``k_shape``;
    every corner of the box is checked exactly and 64 replayed samples by
    LP."""
    k = _constant(op, k_shape, seed)
    out_shape = tuple(np.broadcast_shapes(tuple(in_shape), tuple(k_shape)))
    inputs = ["K", "X"] if constant_first else ["X", "K"]
    path = save_onnx(tmp_path, [helper.make_node(op, inputs, ["Y"])],
                     in_shape, out_shape, inits={"K": k})
    n = int(np.prod(in_shape))
    lo, hi = flat_box(n, seed=seed, zero_width=n >= 4)
    what = (f"{op}({'K, X' if constant_first else 'X, K'}) with K of shape "
            f"{tuple(k_shape)} on a {kind} of tensor shape {tuple(in_shape)}")
    model = load_onnx(path)
    options = {"input_shape": tuple(in_shape[1:])}
    expanding = int(np.prod(out_shape)) != n
    if _refused(kind, op, expanding, constant_first):
        with pytest.raises(NotImplementedError):
            _reach(model, flat_set(kind, lo, hi), "approx", what,
                        **options)
        return
    sets = _one_set(_reach(model, flat_set(kind, lo, hi), "approx", what,
                           **options), what)
    _assert_exactly_covered(
        sets, [_exact(op, c.reshape(in_shape), k, constant_first)
               for c in _corners(lo, hi)], what)
    assert_covered(sets, replay_onnx(path, samples(lo, hi, 64), in_shape),
                   what)


def _flat_grid():
    """The (kind, op, layout) cells of the flat test below. The broadcast of
    the constant (scalar tile, trailing and middle axis through the recorded
    shape) runs before the set type is looked at and hands every kernel a
    full-size constant, so it runs on the Star with every operation. Every
    type runs its kernel with the full-size constant, whose entries differ
    and have both signs, and its own row expansion ("expanding",
    ``_select_rows``: a branch per type, a refusal for a Hexatope or
    Octatope). Mul and Div on a Hexatope or Octatope are refused by the
    scaling (one cell each)."""
    cells = []
    for kind in FLAT_KINDS:
        for op in OPS:
            for layout in sorted(_FLAT_LAYOUTS):
                keep = (kind == "Star" or layout == "full"
                        or (layout == "expanding"
                            and (kind in ("Zono", "Box") or op == "Add")))
                if keep:
                    cells.append((kind, op, layout))
    return cells


@pytest.mark.parametrize("kind, op, layout", _flat_grid())
def test_constant_on_a_flat_set_holds_the_exact_image(tmp_path, kind, op,
                                                      layout):
    """X op K on a flat set with a straddling, a positive, a negative and a
    zero-width coordinate, for a scalar, a trailing-axis, a middle-axis, a
    full-size and an entry-adding constant. Catches a translation by the
    wrong sign or from the wrong centre, a dropped constant, a scale applied
    to the generators only or to the centre only, a scale or quotient bound
    that ignores the sign of a negative entry, a divisor used as it is
    instead of as its reciprocal, a constant broadcast along the wrong axis
    (repeated where it is tiled, flattened before it is broadcast), a row
    expansion that misplaces the set's entries, a kernel that writes into its
    input set (through ``reach_model``'s check), and a refusal of a supported
    cell (or an answer where the table above refuses)."""
    in_shape, k_shape = _FLAT_LAYOUTS[layout]
    _flat_cell(tmp_path, kind, op, in_shape, k_shape)


@pytest.mark.parametrize("kind, op", [
    (kind, op) for kind in FLAT_KINDS for op in OPS
    if (kind in ("Box", "Star", "Zono") and op != "Div")
    or (kind == "Star" and op == "Div")
    or (kind in ("Hexatope", "Octatope") and op == "Sub")])
def test_constant_first_operand_on_a_flat_set(tmp_path, kind, op):
    """K op X: Add and Mul commute, Sub(K, X) is -X + K (the negation is a
    scaling by -1, so a Hexatope or Octatope refuses it), and Div(K, X), a
    division by a computed set, is refused before the set type is looked at
    (one cell). Catches a mirrored Sub evaluated as X - K, a mirrored node
    whose operands are not swapped back, and a division by a set that
    answers."""
    _flat_cell(tmp_path, kind, op, (1, 2, 3), (3,), constant_first=True,
               seed=3)


_SAME_SIZE_EXPANDING = known_bug(
    7, "a constant with as many entries as the set whose broadcast adds "
       "entries ((1, 3, 1) op (1, 1, 3) gives (1, 3, 3)) is applied entry by "
       "entry: the result has the set's 3 entries where the output has 9, "
       "with or without the recorded tensor shape, for every flat set type "
       "and operation (GRAPH-9)")


@pytest.mark.parametrize("op", ["Add"])
@pytest.mark.parametrize("kind", ["Star"])
@_SAME_SIZE_EXPANDING
def test_constant_of_the_sets_size_that_adds_entries(tmp_path, kind, op):
    """(1, 3, 1) op (1, 1, 3) = (1, 3, 3). The handler compares only the
    number of entries of the constant with the set's, before the set type or
    the operation is looked at, so this broadcast is taken for an
    element-wise one: one marked cell. The same reference and layout code
    passes on the "expanding" cells of the flat test above, whose constant
    has a size of its own."""
    _flat_cell(tmp_path, kind, op, (1, 3, 1), (1, 1, 3), seed=5)


# --------------------------------------------------------------------------- #
# An image set and a constant                                                #
# --------------------------------------------------------------------------- #
_IMAGE_LAYOUTS = {"scalar": (), "channel": (1, 2, 1, 1), "full": (2, 3, 5)}


def _image_cell(tmp_path, kind, op, k_shape, chw=_IMAGE,
                constant_first=False, seed=0):
    """``op`` between an image box set of (C, H, W) = ``chw`` (an
    ``ImageStar``, an ``ImageZono``, or a flat ``Star`` with ``input_shape``
    as the tensor-order neighbour) and a constant of ``k_shape``: 12 sample
    points checked exactly, 64 replayed samples by LP."""
    channels, height, width = chw
    shape = [1, channels, height, width]
    k = _constant(op, k_shape, seed)
    inputs = ["K", "X"] if constant_first else ["X", "K"]
    path = save_onnx(tmp_path, [helper.make_node(op, inputs, ["Y"])], shape,
                     shape, inits={"K": k})
    lo, hi = pixel_box(channels, height, width, seed=seed + 1)
    if kind == "Star":
        input_set, options = flat_set("Star", lo, hi), {"input_shape": chw}
    else:
        input_set, options = image_set(kind, lo, hi), {}
    what = (f"{op}({'K, X' if constant_first else 'X, K'}) with K of shape "
            f"{tuple(k_shape)} on a {kind} of (C, H, W) = {chw}")
    sets = _one_set(_reach(load_onnx(path), input_set, "approx", what,
                           **options), what)
    points = samples(lo, hi, 64)
    _assert_exactly_covered(
        sets, [_exact(op, p.reshape(shape), k, constant_first)
               for p in points[:12]], what, out_chw=chw)
    assert_covered(sets, replay_onnx(path, points, shape), what, out_chw=chw)


@pytest.mark.parametrize("kind, op, layout", [
    (kind, op, layout) for kind in IMAGE_KINDS for op in OPS
    for layout in sorted(_IMAGE_LAYOUTS)
    if op in ("Add", "Mul") or (op, layout) in (("Sub", "channel"),
                                                 ("Div", "full"))])
def test_constant_on_an_image_set(tmp_path, kind, op, layout):
    """X op K on a (2, 3, 5) image with a different interval per pixel, for
    a scalar, a per-channel (1, 2, 1, 1) and a full (2, 3, 5) constant whose
    entries are all different. Add and Mul run every layout branch of each
    image type; Sub only negates the constant before the Add branches and
    Div only takes its reciprocal before the Mul branches, one layout each.
    Catches a constant laid out in (C, H, W) order on the stored (H, W, C)
    entries (channel blocks where the channel is innermost, a full constant
    not transposed: plants add-imagestar-full-layout,
    mul-imagezono-full-layout), a translation applied to a generator instead
    of the centre or with the wrong sign, a scale that divides, and a centre
    left unscaled."""
    _image_cell(tmp_path, kind, op, _IMAGE_LAYOUTS[layout])


@pytest.mark.parametrize("op", ["Sub"])
@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_constant_first_operand_on_an_image_set(tmp_path, kind, op):
    """Sub(K, X) with a per-channel constant negates the image (the scalar
    branch of the image scaling) before the layout-aware translation.
    Catches a mirrored Sub evaluated as X - K on an image set, and a
    negation that scales the image wrongly. The swap of a mirrored Add or Mul
    and the refusal of Div(K, X) are done before the set type is looked at
    (flat cells above)."""
    channels, _, _ = _IMAGE
    _image_cell(tmp_path, kind, op, (1, channels, 1, 1), constant_first=True,
                seed=2)


_SIZE_OF_C = known_bug(
    7, "a constant with as many entries as the image has channels is applied "
       "per channel whatever its shape: a (1, 1, W) constant with W = C, or a "
       "(1, H, W) constant with H * W = C, scales or shifts each pixel by "
       "another entry, and true outputs fall outside the result; Sub and Div "
       "go through the same Add and Mul branches (registry GRAPH-10)")

# name: ((C, H, W), constant shape): the constant has C entries but runs along
# the width, or over the whole plane.
_SIZE_C_CASES = {"width-is-C": ((2, 3, 2), (1, 1, 2)),
                 "plane-is-C": ((6, 2, 3), (1, 2, 3))}


@pytest.mark.parametrize("kind, op, case", [
    ("Star", "Add", "plane-is-C"), ("Star", "Add", "width-is-C"),
    ("Star", "Mul", "plane-is-C"), ("Star", "Mul", "width-is-C"),
    pytest.param("ImageStar", "Add", "width-is-C", marks=_SIZE_OF_C),
    pytest.param("ImageStar", "Mul", "plane-is-C", marks=_SIZE_OF_C),
    pytest.param("ImageZono", "Add", "plane-is-C", marks=_SIZE_OF_C),
    pytest.param("ImageZono", "Mul", "width-is-C", marks=_SIZE_OF_C),
])
def test_constant_with_as_many_entries_as_channels(tmp_path, kind, op, case):
    """The image branches tell a per-channel constant from others by its
    number of entries only: one marked cell per branch that does so (the Add
    and the Mul branch of each image type; Sub and Div reuse them), the two
    shapes shared out between them. The flat Star with the recorded tensor
    shape broadcasts by
    the constant's shape and is the passing neighbour, with the same
    reference and layout code. The cell demands an answer; whether a refusal
    of this broadcast would be enough is an open owner question (a
    refusal would turn the marked cells into failures, not passes)."""
    chw, k_shape = _SIZE_C_CASES[case]
    _image_cell(tmp_path, kind, op, k_shape, chw=chw, seed=4)


_PLANE_CRASH = known_bug(
    23, "Mul and Div of an ImageStar or ImageZono by a (1, H, W) constant "
        "raise ValueError ('Scale size ... does not match') from "
        "_mul_sets_by_constant where an answer or a NotImplementedError "
        "refusal is due, as Add and Sub refuse (registry GRAPH-11)",
    raises=ValueError)


@pytest.mark.parametrize("kind, op", [
    (kind, op) if kind == "Star" or op == "Add"
    else pytest.param(kind, op, marks=_PLANE_CRASH)
    for kind in ("Star",) + IMAGE_KINDS for op in ("Add", "Mul")])
def test_constant_over_the_image_plane_is_answered_or_refused(tmp_path, kind,
                                                              op):
    """A (1, 3, 5) constant on a (2, 3, 5) image is a legal ONNX broadcast.
    The flat Star with the recorded shape answers it; the image branches have
    no rule for it and must either answer soundly or refuse with
    ``NotImplementedError``. Add stands for Sub (the same branches after the
    sign) and Mul for Div (the same branches after the reciprocal). Catches
    an image branch that answers this constant by another rule."""
    try:
        _image_cell(tmp_path, kind, op, (1, 3, 5), seed=6)
    except NotImplementedError:
        if kind == "Star":
            raise


# --------------------------------------------------------------------------- #
# Composition: affine, the operation, affine                                 #
# --------------------------------------------------------------------------- #
_W1 = np.array([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5], [-0.25, 0.75, 1.0],
                [0.5, 0.25, 0.5]], dtype=np.float32)
_B1 = np.array([0.25, -0.5, 0.75], dtype=np.float32)
_W2 = np.array([[1.0, -0.5], [0.25, 1.0], [-0.75, 0.5]], dtype=np.float32)


@pytest.mark.parametrize("kind, op, constant_first", [
    (kind, op, first) for kind in FLAT_KINDS
    for op, first in [("Add", False), ("Sub", False), ("Sub", True),
                      ("Mul", False), ("Div", False)]
    if not _refused(kind, op, False, first)])
def test_constant_between_two_affine_layers_flat(tmp_path, kind, op,
                                                 constant_first):
    """MatMul + bias, then op with a (3,) constant, then MatMul: the rows the
    operation leaves are read by the next layer. The Star input has a binding
    row that is not a box row, so a wrong row of the result shows even where
    its bounding box is right. The set reaching the operation has more
    predicate variables or generators (4) than entries (3), each shared by
    several entries, so a generator row scaled without the sign of its
    constant entry is seen. The box
    is eight times ``flat_box``: entries well beyond plus and minus 1 and
    intervals wider than 2, so a predicate variable read in place of another
    moves outputs out of the result. Every vertex of the input set is checked
    exactly. The cells the library refuses (Mul, Div and a mirrored Sub on a
    Hexatope or Octatope) are asserted by the one-node cells above; Mul with
    the constant first is the same scaling after the swap of the operands,
    which the one-node cells check."""
    k = _constant(op, (3,), seed=7)
    a, b = ("K", "H") if constant_first else ("H", "K")
    nodes = [helper.make_node("MatMul", ["X", "W1"], ["P"]),
             helper.make_node("Add", ["P", "B1"], ["H"]),
             helper.make_node(op, [a, b], ["Q"]),
             helper.make_node("MatMul", ["Q", "W2"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 4], [1, 2],
                     inits={"W1": _W1, "B1": _B1, "K": k, "W2": _W2})
    lo, hi = flat_box(4, seed=8)
    lo, hi = 8 * lo, 8 * hi
    what = f"MatMul, {op} with a constant, MatMul on a {kind}"
    if kind == "Star":
        input_set, points = constrained_star(lo, hi, n=64, seed=8)
        vertices = [np.array([float(v) for v in vertex]) for vertex in
                    rational.set_vertices(rational.star(input_set))]
    else:
        input_set, points = flat_set(kind, lo, hi), samples(lo, hi, 64)
        vertices = _corners(lo, hi)
    sets = _one_set(_reach(load_onnx(path), input_set, "approx", what), what)
    exact = []
    for x in vertices:
        h = _matmul_exact(x, _W1, _B1)
        q = np.array(_exact(op, h, k, constant_first), dtype=object)
        exact.append(list(_matmul_exact_objects(q, _W2)))
    _assert_exactly_covered(sets, exact, what)
    assert_covered(sets, replay_onnx(path, points, [1, 4]), what)


def _matmul_exact_objects(x, W):
    """``x @ W`` for an object array ``x`` of rationals."""
    W = _rationals(W)
    return [sum((x[i] * W[i, j] for i in range(len(x))), Fraction(0))
            for j in range(W.shape[1])]


# Conv weights: (out, in, kh, kw), dyadic, mixing the two channels.
_KA = np.array([[[[0.5, -0.25, 1.0]], [[0.25, 0.5, -0.5]]],
                [[[-0.5, 1.0, 0.25]], [[0.75, -0.25, 0.5]]]],
               dtype=np.float32)                       # kernel (1, 3)
_BA = np.array([0.25, -0.5], dtype=np.float32)
_KB = np.array([[[[1.0], [-0.5], [0.25]], [[0.5], [0.25], [-1.0]]],
                [[[-0.25], [0.5], [0.75]], [[1.0], [-0.5], [0.5]]]],
               dtype=np.float32)                       # kernel (3, 1)
_BB = np.array([-0.25, 0.5], dtype=np.float32)


# The pixels (channel, row, column) that vary in the composition cells; every
# other pixel is fixed at its own value. Three live pixels in different
# channels, rows and columns keep the exact image to eight vertices (fewer
# under the binding row) while every pixel keeps a value of its own.
_LIVE = [(0, 0, 1), (1, 2, 3), (0, 1, 4)]


def _live_box(seed):
    """(C, H, W) bounds of ``pixel_box`` times eight (values well beyond plus
    and minus 1, intervals up to 10 wide) with a positive width at the pixels
    of ``_LIVE`` only."""
    lo, hi = pixel_box(*_IMAGE, seed=seed)
    lo, hi = 8 * lo, 8 * hi
    fixed = np.ones(_IMAGE, dtype=bool)
    for p in _LIVE:
        fixed[p] = False
    return lo, np.where(fixed, lo, hi)


def _input_vertices(input_set, chw):
    """The vertices of an image input set as (C, H, W) object arrays of
    rationals: the vertices of its predicate polytope mapped through its
    stored (H, W, C) basis (``rational.set_vertices``), put back in tensor
    order. A variable with a zero generator column that no row mentions
    (an ImageZono's fixed pixel) is pinned to 0 first, which leaves the set
    unchanged."""
    c, G, C, d, lb, ub = rational.star(input_set)
    keep = [j for j in range(len(lb))
            if any(row[j] != 0 for row in G) or any(row[j] != 0 for row in C)]
    reduced = (c, [[row[j] for j in keep] for row in G],
               [[row[j] for j in keep] for row in C], d,
               [lb[j] for j in keep], [ub[j] for j in keep])
    order = hwc_index(*chw)
    out = []
    for v in rational.set_vertices(reduced):
        x = np.empty(len(order), dtype=object)
        x[order] = list(v)
        out.append(x.reshape(chw))
    return out


@pytest.mark.parametrize("layout", ["channel", "full"])
@pytest.mark.parametrize("op, constant_first", [
    ("Add", False), ("Sub", True), ("Mul", False), ("Div", False)])
@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_constant_between_two_convolutions(tmp_path, kind, op,
                                           constant_first, layout):
    """Conv with a (1, 3) kernel, op with a per-channel or full constant,
    Conv with a (3, 1) kernel, on a (2, 3, 5) image with a value of its own
    at every pixel and three live pixels; the ImageStar input has a binding
    row that is not a box row. The convolutions read the operation's output
    in the stored layout, so a constant placed on the wrong pixel or channel
    moves outputs that the next layer mixes. Every vertex of the input set is
    mapped exactly and must land in the result, and 64 replayed samples
    too."""
    channels, height, width = _IMAGE
    shape = [1, channels, height, width]
    k_shape = {"channel": (1, channels, 1, 1), "full": _IMAGE}[layout]
    k = _constant(op, k_shape, seed=9)
    a, b = ("K", "A") if constant_first else ("A", "K")
    nodes = [helper.make_node("Conv", ["X", "KA", "BA"], ["A"],
                              pads=[0, 1, 0, 1]),
             helper.make_node(op, [a, b], ["Q"]),
             helper.make_node("Conv", ["Q", "KB", "BB"], ["Y"],
                              pads=[1, 0, 1, 0])]
    path = save_onnx(tmp_path, nodes, shape, shape,
                     inits={"KA": _KA, "BA": _BA, "K": k, "KB": _KB,
                            "BB": _BB})
    lo, hi = _live_box(seed=10)
    if kind == "ImageStar":
        input_set, points = constrained_image_star(lo, hi, n=64, seed=10)
    else:
        input_set, points = image_set(kind, lo, hi), samples(lo, hi, 64)
    what = f"Conv, {op} with a {layout} constant, Conv on an {kind}"
    sets = _one_set(_reach(load_onnx(path), input_set, "approx", what), what)
    kk = _rationals(k).reshape(k_shape)
    exact = []
    for x in _input_vertices(input_set, _IMAGE):
        a_val = _conv_exact(x, _KA, _BA, [0, 1, 0, 1])
        q = np.asarray(_exact(op, a_val[None], kk, constant_first),
                       dtype=object).reshape(_IMAGE)
        exact.append(list(_conv_exact(q, _KB, _BB, [1, 0, 1, 0]).reshape(-1)))
    if len(exact) < 4:
        pytest.fail(f"{what}: the input set has {len(exact)} vertices")
    _assert_exactly_covered(sets, exact, what, out_chw=_IMAGE)
    assert_covered(sets, replay_onnx(path, points, shape), what,
                   out_chw=_IMAGE)


# --------------------------------------------------------------------------- #
# Several input sets: the pieces of an exact split                           #
# --------------------------------------------------------------------------- #
_WS = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=np.float32)   # X @ _WS
_BS = np.array([0.0, 0.25], dtype=np.float32)


@pytest.mark.parametrize("op, constant_first", [
    ("Add", False), ("Sub", False), ("Sub", True), ("Mul", False),
    ("Div", False)])
def test_constant_after_an_exact_split_is_exact(tmp_path, op,
                                                constant_first):
    """Exact ReLU splits the input into several Stars; the operation must
    translate or scale every piece. Both directions in exact rationals: every
    vertex of every exact piece is in a returned set, and every vertex of
    every returned set is the true output of some input. Catches a handler
    that passes on only the first piece, or uses one piece's centre for
    another."""
    k = _constant(op, (2,), seed=11)
    a, b = ("K", "R") if constant_first else ("R", "K")
    nodes = [helper.make_node("MatMul", ["X", "WS"], ["P"]),
             helper.make_node("Add", ["P", "BS"], ["H"]),
             helper.make_node("Relu", ["H"], ["R"]),
             helper.make_node(op, [a, b], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 2],
                     inits={"WS": _WS, "BS": _BS, "K": k})
    lo, hi = np.array([-1.0, -0.5]), np.array([1.0, 1.0])
    what = f"exact ReLU, then {op} with a constant, on a Star"
    sets = _reach(load_onnx(path), flat_set("Star", lo, hi), "exact",
                       what)
    _enumerable(sets, 4, 2, what)
    kk = _rationals(k)
    if op in ("Add", "Sub"):
        sign = -1 if (op == "Sub" and constant_first) else 1
        shift = (kk if op == "Add" or constant_first else -kk)
        D, e = [[sign, 0], [0, sign]], list(shift)
    else:
        scale = kk if op == "Mul" else 1 / kk
        D, e = [[scale[0], 0], [0, scale[1]]], [0, 0]
    pieces = rational.affine(
        rational.relu(rational.affine(rational.box(lo, hi), _WS.T.tolist(),
                                      _BS.tolist())), D, e)
    computed = [rational.star(s) for s in sets]
    escaped = rational.escapes(pieces, *computed)
    assert not escaped, f"{what}: exact vertices outside: {escaped[:3]}"
    assert not rational.straddlers(pieces, *computed), (
        f"{what}: an exact piece is spread over several returned sets")
    vertices = [v for c in computed for v in rational.set_vertices(c)]
    unreached = rational.unreached(pieces, vertices)
    assert not unreached, (
        f"{what}: vertices of the result with no preimage: {unreached[:3]}")
    assert_covered(sets, replay_onnx(path, samples(lo, hi, 128), [1, 2]),
                   what)


def _enumerable(sets, most_sets, most_variables, what):
    """The vertex enumeration of the reverse direction grows fast with the
    number of returned sets and of their predicate variables; a bug that
    returns more of either would make the cell run until the gate's time
    limit instead of failing. An exact split of this input gives at most
    ``most_sets`` pieces over its ``most_variables`` predicate variables."""
    if len(sets) > most_sets:
        raise AssertionError(f"{what}: {len(sets)} sets returned, the exact "
                             f"split has at most {most_sets} pieces")
    for s in sets:
        n = s.nVar
        if n > most_variables:
            raise AssertionError(
                f"{what}: a returned set has {n} predicate variables, the "
                f"input has {most_variables}")


# A (2, 2, 3) image: three live pixels (flat (C, H, W) index: interval), each
# crossing zero, in both channels, both rows and all three columns; every
# other pixel is fixed at a nonzero value of its own, so the exact split has
# at most eight pieces over three predicate variables.
_SPLIT_CHW = (2, 2, 3)
_SPLIT_FIXED = np.array([0.0, 0.25, -1.0, 0.5, 1.0, 0.0, -0.25, 0.0, 0.75,
                         -1.0, 1.0, -0.5])
_SPLIT_LIVE = {0: (-0.5, 0.5), 5: (-2.0, 1.0), 7: (-1.0, 0.25)}


def _image_split_pieces(op, k, constant_first):
    """The exact image of Relu then ``op`` with the per-channel constant
    ``k`` on the input box, as ``rational`` pieces over the three live
    pixels' values, each output in the stored (H, W, C) order of an
    ImageStar."""
    channels, height, width = _SPLIT_CHW
    n = channels * height * width
    live = sorted(_SPLIT_LIVE)
    pieces = rational.box([_SPLIT_LIVE[i][0] for i in live],
                          [_SPLIT_LIVE[i][1] for i in live])
    select = [[int(i == j) for j in live] for i in range(n)]
    fixed = [0.0 if i in _SPLIT_LIVE else _SPLIT_FIXED[i] for i in range(n)]
    pieces = rational.relu(rational.affine(pieces, select, fixed))
    kk = [Fraction(float(v)) for v in np.asarray(k).reshape(-1)]
    per_entry = [kk[i // (height * width)] for i in range(n)]
    if op == "Add":
        scale, shift = [1] * n, per_entry
    elif op == "Sub" and constant_first:
        scale, shift = [-1] * n, per_entry
    elif op == "Mul":
        scale, shift = per_entry, [0] * n
    else:
        scale, shift = [1 / v for v in per_entry], [0] * n
    diagonal = [[scale[i] if i == j else 0 for j in range(n)]
                for i in range(n)]
    pieces = rational.affine(pieces, diagonal, shift)
    order = hwc_index(channels, height, width)
    stored = [[int(order[r] == j) for j in range(n)] for r in range(n)]
    return rational.affine(pieces, stored)


@pytest.mark.parametrize("op, constant_first", [
    pytest.param("Add", False, id="Add"), pytest.param("Mul", False, id="Mul"),
    pytest.param("Div", False, id="Div"),
    pytest.param("Sub", True, id="Sub-constant-first")])
def test_constant_after_an_exact_split_of_an_image(tmp_path, op,
                                                   constant_first):
    """Exact ReLU on a (2, 2, 3) ImageStar with three crossing pixels gives
    several ImageStars; a per-channel constant must reach every one of them
    in the stored layout, and the result must be the exact image. Both
    directions in exact rationals against the pieces of the network computed
    in tensor order and permuted to the stored order: every vertex of every
    exact piece is in a returned set, and every vertex of every returned set
    is the true output of some input. Catches a handler that passes on only
    the first piece, a constant on the
    wrong channel, and a result larger than the exact image (a generator
    scaled or shifted where it should not be)."""
    shape = [1, *_SPLIT_CHW]
    k = _constant(op, (1, 2, 1, 1), seed=12)
    a, b = ("K", "R") if constant_first else ("R", "K")
    nodes = [helper.make_node("Relu", ["X"], ["R"]),
             helper.make_node(op, [a, b], ["Y"])]
    path = save_onnx(tmp_path, nodes, shape, shape, inits={"K": k})
    lo, hi = _SPLIT_FIXED.copy(), _SPLIT_FIXED.copy()
    for i, (low, high) in _SPLIT_LIVE.items():
        lo[i], hi[i] = low, high
    lo, hi = lo.reshape(_SPLIT_CHW), hi.reshape(_SPLIT_CHW)
    what = f"exact ReLU, then {op} by a per-channel constant, on an ImageStar"
    sets = _reach(load_onnx(path), image_set("ImageStar", lo, hi),
                       "exact", what)
    if len(sets) < 2:
        pytest.fail(f"{what}: the split gave {len(sets)} set; the cell needs "
                    f"several")
    _enumerable(sets, 8, len(_SPLIT_LIVE), what)
    for s in sets:
        if (s.num_channels, s.height, s.width) != _SPLIT_CHW:
            raise AssertionError(f"{what}: a returned ImageStar stores "
                                 f"{s.height}x{s.width}x{s.num_channels}")
    pieces = _image_split_pieces(op, k, constant_first)
    computed = [rational.star(s) for s in sets]
    escaped = rational.escapes(pieces, *computed)
    assert not escaped, f"{what}: exact vertices outside: {escaped[:3]}"
    assert not rational.straddlers(pieces, *computed), (
        f"{what}: an exact piece is spread over several returned sets")
    vertices = [v for c in computed for v in rational.set_vertices(c)]
    unreached = rational.unreached(pieces, vertices)
    assert not unreached, (
        f"{what}: vertices of the result with no preimage: {unreached[:3]}")
    assert_covered(sets, replay_onnx(path, samples(lo, hi, 128), shape), what,
                   out_chw=_SPLIT_CHW)


# --------------------------------------------------------------------------- #
# No recorded tensor shape: the tile fallback                                #
# --------------------------------------------------------------------------- #
_K1 = np.array([[[[0.5]], [[-0.25]]], [[[0.75]], [[1.0]]]], dtype=np.float32)
_B1x1 = np.array([0.25, -0.5], dtype=np.float32)


def _after_reshape(tmp_path, kind, op, k_shape, target=(1, 6, 5)):
    """Conv 1x1, Reshape (1, 2, 3, 5) to ``target``, op with a constant of
    ``k_shape``. For an ImageZono reached without ``input_shape`` no tensor
    shape is recorded today (the dummy forward that records shapes is given
    the flat size and the Conv rejects it: SHAPE-3), so the constant reaches
    the tile fallback; an ImageStar gets the recorded shapes. The reshaped
    set is flat, in the tensor's row-major order."""
    channels, height, width = _IMAGE
    shape = [1, channels, height, width]
    k = _constant(op, k_shape, seed=13)
    nodes = [helper.make_node("Conv", ["X", "K1", "B1"], ["A"]),
             helper.make_node("Reshape", ["A", "S"], ["R"]),
             helper.make_node(op, ["R", "K"], ["Y"])]
    out_shape = tuple(np.broadcast_shapes(tuple(target), tuple(k_shape)))
    path = save_onnx(tmp_path, nodes, shape, out_shape,
                     inits={"K1": _K1, "B1": _B1x1,
                            "S": np.array(target, dtype=np.int64),
                            "K": k})
    lo, hi = pixel_box(channels, height, width, seed=14)
    what = (f"Conv, Reshape to {tuple(target)}, {op} with K of shape "
            f"{tuple(k_shape)} on an {kind}")
    sets = _one_set(_reach(load_onnx(path), image_set(kind, lo, hi), "approx",
                           what), what)
    points = samples(lo, hi, 64)
    exact = []
    for x in points[:12]:
        a = _conv_exact(np.asarray(x).reshape(_IMAGE), _K1, _B1x1,
                        [0, 0, 0, 0]).reshape(target)
        exact.append(_exact(op, a, k))
    _assert_exactly_covered(sets, exact, what)
    assert_covered(sets, replay_onnx(path, points, shape), what)


@pytest.mark.parametrize("op", ["Add", "Mul"])
@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_trailing_constant_after_a_reshape(tmp_path, kind, op):
    """A (5,) constant on a (1, 6, 5) tensor is the whole vector repeated six
    times along the flat entries. Catches a tile fallback that repeats each
    entry instead (np.repeat for np.tile), and a recorded-shape broadcast
    along the wrong axis. The broadcast is done before the operation is
    looked at, so Add (the translation of a flat Star or Zono) and Mul (the
    scaling) stand for Sub and Div."""
    _after_reshape(tmp_path, kind, op, (5,))


_UNTRACKED_EXPANSION = known_bug(
    7, "the tile fallback of a constant operand (no tracked tensor shape) "
       "assumes the broadcast adds no entries: an ImageZono reached without "
       "input_shape has no tracked shape after an image layer (SHAPE-3), so "
       "after Conv and a Reshape to (1, 30, 1) a (2,) constant, whose "
       "broadcast gives (1, 30, 2), is tiled over the 30 entries and the "
       "result has 30 entries where the output has 60")


@pytest.mark.parametrize("op", ["Add"])
@pytest.mark.parametrize("kind", [
    "ImageStar", pytest.param("ImageZono", marks=_UNTRACKED_EXPANSION)])
def test_constant_that_adds_entries_after_a_reshape(tmp_path, kind, op):
    """Conv, Reshape to (1, 30, 1), Add of a (2,) constant: the output is
    (1, 30, 2), each of the 30 entries repeated with the two constants. The
    ImageStar has the recorded shapes and expands its rows (the passing
    neighbour, same model, reference and checks); the ImageZono reaches the
    tile fallback without a shape. Catches a fallback or an expansion that
    applies an entry-adding constant entry by entry, and an expansion that
    misplaces the entries."""
    _after_reshape(tmp_path, kind, op, (2,), target=(1, 30, 1))


@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_constant_with_an_inner_unit_axis_is_answered_or_refused(tmp_path,
                                                                 kind):
    """A (6, 1) constant on (1, 6, 5) repeats each entry five times. Without
    a recorded shape a tile would be wrong, so the fallback must refuse
    (``NotImplementedError``); with the shape the broadcast must be right."""
    try:
        _after_reshape(tmp_path, kind, "Add", (6, 1))
    except NotImplementedError:
        if kind == "ImageStar":
            raise


# --------------------------------------------------------------------------- #
# The opset 6 broadcast attribute                                            #
# --------------------------------------------------------------------------- #
_OLD_BROADCAST = known_bug(
    7, "the opset 6 'broadcast'/'axis' attributes of Add, Sub, Mul and Div "
       "are ignored: a (2,) constant at axis 1 of a (1, 2, 3, 2) tensor is "
       "broadcast along the last axis, and true outputs fall outside the "
       "result (registry GRAPH-12; "
       "the row also holds two computed operands)")


@pytest.mark.parametrize("op", ["Add"])
@pytest.mark.parametrize("axis, k_shape", [
    pytest.param(1, (2,), marks=_OLD_BROADCAST, id="axis1"),
    pytest.param(2, (3, 2), id="trailing")])
def test_opset6_broadcast_axis_is_followed(tmp_path, op, axis, k_shape):
    """Opset 6 Add(X, K, broadcast=1, axis=a) places K at axis a. With K
    aligned to the trailing axes this is numpy broadcasting and passes; with
    K at axis 1 it is not. The handler reads no attribute of the node before
    the operation is looked at, so Add stands for Sub, Mul and Div.
    onnxruntime has no opset 6 kernel, so the replay is the loaded torch
    module in float64, checked against the exact value first."""
    shape = (1, 2, 3, 2)
    k = _constant(op, k_shape, seed=15)
    path = save_onnx(tmp_path, [helper.make_node(op, ["X", "K"], ["Y"],
                                                 broadcast=1, axis=axis)],
                     shape, shape, inits={"K": k}, opset=6)
    placed = k.reshape((1,) * axis + k_shape
                       + (1,) * (len(shape) - axis - len(k_shape)))
    lo, hi = flat_box(12, seed=16)
    model = load_onnx(path)
    points = samples(lo, hi, 64)
    exact = [_exact(op, p.reshape(shape), placed) for p in points]
    replayed = replay_torch(model, points, shape)
    if any(np.any(np.asarray([float(v) for v in e]) != r)
           for e, r in zip(exact, replayed)):
        pytest.fail("the loaded module does not compute the opset 6 "
                    "broadcast; the reference is wrong")
    what = f"opset 6 {op} with K of shape {k_shape} at axis {axis}"
    sets = _one_set(_reach(model, flat_set("Star", lo, hi), "approx", what,
                           input_shape=shape[1:]), what)
    _assert_exactly_covered(sets, exact[:16], what)
    assert_covered(sets, replayed, what)


# --------------------------------------------------------------------------- #
# Division by a constant with a zero entry                                   #
# --------------------------------------------------------------------------- #
_ZERO_DIVISOR = known_bug(
    7, "Div of a Star, Zono, ImageStar or ImageZono by a constant with a zero "
       "entry is answered with a non-finite basis (1 / 0, and 0 times "
       "infinity) instead of refused as the Box path does: the zero check of "
       "_div_sets_by_constant comes after the early return for a list "
       "without a Box (GRAPH-13)",
    raises=DID_NOT_RAISE)


@pytest.mark.parametrize("kind", [
    "Box",
    pytest.param("Star", marks=_ZERO_DIVISOR)])
def test_division_by_a_zero_constant_is_refused(tmp_path, kind):
    """X / K with K = (2, 0) per channel: the true output is infinite where
    K is 0, so no set holds it and the call must refuse. The numpy warning
    of 1 / 0 is silenced so that the refusal, not the warning, decides. The
    reach goes through ``reach_model`` itself: an empty result is an
    ``AssertionError`` there, which the marker (``DID_NOT_RAISE``) does not
    absorb. Every set type but the
    Box leaves ``_div_sets_by_constant`` by the same early return before the
    zero check, so the Star is the one marked cell."""
    channels, height, width = _IMAGE
    k = np.array([2.0, 0.0], dtype=np.float32).reshape(1, channels, 1, 1)
    shape = [1, channels, height, width]
    path = save_onnx(tmp_path, [helper.make_node("Div", ["X", "K"], ["Y"])],
                     shape, shape, inits={"K": k})
    lo, hi = pixel_box(channels, height, width)
    if kind in IMAGE_KINDS:
        input_set, options = image_set(kind, lo, hi), {}
    else:
        input_set, options = flat_set(kind, lo, hi), {"input_shape": _IMAGE}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(NotImplementedError):
            reach_model(load_onnx(path), input_set, "approx",
                        f"Div by a zero entry on a {kind}", **options)


# --------------------------------------------------------------------------- #
# Rounding of Box arithmetic with a constant                                 #
# --------------------------------------------------------------------------- #
# One cell per arithmetic path: the outward translation (Add; Add(K, X) and
# Sub(X, K) differ only by a swap of the operands and an exact negation of
# the constant before it), the negation of the set followed by the
# translation (Sub(K, X)), the outward product (Mul, either order) and the
# outward quotient (Div).
_ROUNDING_CASES = [("Add", False), ("Sub", True), ("Mul", False),
                   ("Div", False)]


def _rounding_cell(tmp_path, kind, op, constant_first, width, n=64):
    """A float64 model on ``n`` coordinates with non-dyadic inputs and
    constants; every exact value (for an interval input, both ends) must lie
    in the returned set, compared in exact fractions with no slack."""
    rng = np.random.default_rng(17)
    k = rng.uniform(-3.0, 3.0, n)
    if op == "Div":
        k = np.where(np.abs(k) < 0.25, 0.7, k)
    lo = rng.uniform(-3.0, 3.0, n)
    hi = lo if width == "point" else lo + rng.uniform(0.1, 2.0, n)
    a, b = ("K", "X") if constant_first else ("X", "K")
    path = save_onnx(tmp_path, [helper.make_node(op, [a, b], ["Y"])], [1, n],
                     [1, n], inits={"K": k}, in_type="DOUBLE",
                     out_type="DOUBLE")
    what = f"{op}({a}, {b}) on a float64 {kind} ({width} input)"
    sets = _one_set(_reach(load_onnx(path), flat_set(kind, lo, hi), "approx",
                           what), what)
    ends = [_exact(op, lo, k, constant_first),
            _exact(op, hi, k, constant_first)]
    _assert_exactly_covered(sets, ends, what)


@pytest.mark.parametrize("op, constant_first", _ROUNDING_CASES)
@pytest.mark.parametrize("width", ["point", "interval"])
def test_box_constant_arithmetic_is_rounded_outward(tmp_path, op,
                                                    constant_first, width):
    """A float64 model on 64 coordinates with non-dyadic inputs and
    constants (none of them a float32 number): every exact value (for an
    interval input, both ends) must lie in the returned Box, compared in
    exact fractions with no slack. Catches a bound translated, scaled or
    divided with plain float64 arithmetic, which lands one rounding error
    inside the exact value on about half the coordinates (plant
    mul-box-round-to-nearest), and a constant read through float32 (plant
    constant-read-through-float32)."""
    _rounding_cell(tmp_path, "Box", op, constant_first, width)


_ROUND_8 = known_bug(
    "7b", "ROUND-8: a Star or Zono is translated, scaled or divided by a "
          "constant with round-to-nearest arithmetic (the divisor through a "
          "rounded reciprocal), so the set can leave out the exact value by "
          "one unit in the last place")


@pytest.mark.parametrize("op, constant_first", [
    ("Add", False), ("Mul", False), ("Div", False)])
@pytest.mark.parametrize("width", ["point"])
@pytest.mark.parametrize("kind", ["Star", "Zono"])
@_ROUND_8
def test_star_and_zono_constant_arithmetic_holds_the_exact_value(
        tmp_path, kind, op, constant_first, width):
    """The Box rounding cell above (the passing neighbour, same model, data
    and exact reference) on a Star and a Zono, on 16 coordinates. Today the
    centre and generators are computed with round-to-nearest (ROUND-8, PLAN
    step 7b): one marked cell per rounded line, the translation, the scaling
    and the reciprocal of the divisor, per type; the cell goes green when the
    error term lands."""
    _rounding_cell(tmp_path, kind, op, constant_first, width, n=16)


# --------------------------------------------------------------------------- #
# A float64 constant that float32 cannot hold                                 #
# --------------------------------------------------------------------------- #
# Constants of a float64 model that are not float32 numbers. The sum of each
# with a multiple of 1/8 below 32 in size stays in the constant's binade, and
# the product of each with one has at most 40 significant bits, so float64
# computes the exact image of these dyadic boxes without rounding (no ROUND-8
# escape) and any rounding of the constant itself shows.
_F64_K = {"Add": (12345.678901234, -9876.54321012345, 4321.0987654321),
          "Mul": (1 + 2.0 ** -30, -(3 + 2.0 ** -28), 0.5 - 2.0 ** -33)}
_F64_IMAGE = (2, 1, 3)


@pytest.mark.parametrize("kind, op", [
    (kind, op) for kind in FLAT_KINDS + IMAGE_KINDS for op in ("Add", "Mul")
    if not (op == "Mul" and kind in ("Hexatope", "Octatope"))])
def test_a_float64_constant_that_is_no_float32_number_is_kept_exact(
        tmp_path, kind, op):
    """X op K in a float64 model whose constant float32 cannot hold (twelve
    digits near 1e4 for Add, 1 + 2**-30 and its kind for Mul), on a box eight
    times ``flat_box`` or ``pixel_box`` (entries up to 16 in size, intervals
    up to 10 wide), per entry on a flat set and per channel on an image of
    (C, H, W) = (2, 1, 3). Every corner of the box is mapped in exact
    rationals and must be in the result with no slack. Catches a constant
    read or stored through float32 anywhere on the way (plants
    constant-read-through-float32 and, on the Hexatope and Octatope
    translation, hexoct-bias-float32: the outputs move by about 1e-4), which
    a float32 model cannot show. Div is left out: a Star or Zono divides
    through a rounded reciprocal (ROUND-8), and the Box quotient is the
    rounding cell above, whose constants are not float32 numbers either."""
    if kind in IMAGE_KINDS:
        channels, height, width = _F64_IMAGE
        shape = [1, channels, height, width]
        k = np.array(_F64_K[op][:channels]).reshape(1, channels, 1, 1)
        lo, hi = pixel_box(channels, height, width, seed=21)
        lo, hi = 8 * lo, 8 * hi
        input_set, out_chw = image_set(kind, lo, hi), _F64_IMAGE
    else:
        shape = [1, 3]
        k = np.array(_F64_K[op])
        lo, hi = flat_box(3, seed=21)
        lo, hi = 8 * lo, 8 * hi
        input_set, out_chw = flat_set(kind, lo, hi), None
    path = save_onnx(tmp_path, [helper.make_node(op, ["X", "K"], ["Y"])],
                     shape, shape, inits={"K": k}, in_type="DOUBLE",
                     out_type="DOUBLE")
    what = f"{op} with a float64 constant that is no float32 number, {kind}"
    sets = _one_set(_reach(load_onnx(path), input_set, "approx", what), what)
    _assert_exactly_covered(
        sets, [_exact(op, c.reshape(shape), k) for c in _corners(lo, hi)],
        what, out_chw=out_chw)


# --------------------------------------------------------------------------- #
# A flat set reached without input_shape                                      #
# --------------------------------------------------------------------------- #
_NO_INPUT_SHAPE = known_bug(
    12, "SHAPE-2: a flat set reached without input_shape on a model whose "
        "input is (1, 2, 3) is analysed as if the input were (1, 6): a (2, "
        "1) constant is broadcast against (1, 6) and the set is expanded to "
        "12 entries where the output has 6")
_UNTRACKED_EXPANSION_CRASH = known_bug(
    23, "the tile fallback of a constant operand (no tracked tensor shape) "
        "hands a constant whose size does not divide the set's size to the "
        "kernels unchanged: (1, 6, 1) + (5,) on a flat set reached without "
        "input_shape makes numpy raise ValueError in the Zono, Box, Hexatope "
        "and Octatope translation and in the Star, Zono and Box scaling "
        "where a refusal is due (the Star translation refuses)",
    raises=ValueError)


@pytest.mark.parametrize("kind, in_shape, k_shape", [
    pytest.param("Star", (1, 2, 3), (3,), id="trailing"),
    pytest.param("Star", (1, 2, 3), (2, 1), id="inner-axis",
                 marks=_NO_INPUT_SHAPE),
    pytest.param("Star", (1, 2, 3), (1, 2, 3), id="full"),
    pytest.param("Star", (1, 6, 1), (5,), id="adds-entries-Star"),
    pytest.param("Zono", (1, 6, 1), (5,), id="adds-entries-Zono",
                 marks=_UNTRACKED_EXPANSION_CRASH)])
def test_flat_set_without_input_shape_refuses_or_holds_every_output(
        tmp_path, kind, in_shape, k_shape):
    """Add(X, K) on a flat set that stands for a tensor of rank 3, reached
    without ``input_shape``. The graph walk then guesses the input shape
    (``(1, dim)``, then ``(dim,)``) and keeps the first guess whose forward
    does not raise; when none does it has no shape and tiles the constant.
    The promise for every spelling of the constant: a refusal
    (``NotImplementedError``, or a ``ValueError`` that asks for
    ``input_shape``) or a result that holds every output. A trailing (3,)
    constant reaches the tile fallback (catches a fallback that repeats each
    entry, the listed bug add-const-star-tile-layout), a full-size one is
    applied entry by entry, an inner-axis (2, 1) one is broadcast against
    the wrong guess today (registry SHAPE-2, PLAN step 12), and an
    entry-adding (5,) constant on a (1, 6, 1) tensor, whose guesses both
    fail, is refused by the Star translation and crashes the Zono one
    today."""
    k = _constant("Add", k_shape, seed=19)
    out_shape = tuple(np.broadcast_shapes(in_shape, k_shape))
    path = save_onnx(tmp_path, [helper.make_node("Add", ["X", "K"], ["Y"])],
                     in_shape, out_shape, inits={"K": k})
    n = int(np.prod(in_shape))
    lo, hi = flat_box(n, seed=19, zero_width=True)
    what = (f"Add with K of shape {k_shape} on a flat {kind} of tensor shape "
            f"{in_shape} without input_shape")
    try:
        sets = _reach(load_onnx(path), flat_set(kind, lo, hi), "approx",
                      what)
    except NotImplementedError:
        return
    except ValueError as error:
        if "input_shape" not in str(error):
            raise
        return
    _assert_exactly_covered(
        sets, [_exact("Add", c.reshape(in_shape), k)
               for c in _corners(lo, hi)], what)
    assert_covered(_one_set(sets, what),
                   replay_onnx(path, samples(lo, hi, 64), in_shape), what)


# --------------------------------------------------------------------------- #
# A torch model with a constant operand                                       #
# --------------------------------------------------------------------------- #
class _ConstantJoin(nn.Module):
    """``x op k`` with ``k`` a frozen buffer, the way a normalisation is
    written in a torch model, or ``x`` times a Python number."""

    def __init__(self, form):
        super().__init__()
        self.register_buffer("k", torch.tensor([0.5, -1.25, 2.0]))
        self.form = form

    def forward(self, x):
        if self.form == "x+k":
            return x + self.k
        if self.form == "x*k":
            return x * self.k
        if self.form == "x/k":
            return x / self.k
        if self.form == "torch.sub":
            return torch.sub(x, self.k)
        return x * 2.5


@pytest.mark.parametrize("form", ["x+k", "x*k", "x/k", "torch.sub",
                                  "x*2.5"])
def test_torch_model_with_a_constant_operand_refuses_or_holds_every_output(
        form):
    """A torch model whose forward adds, subtracts, multiplies or divides
    by a constant, reached on a Star, a Zono and a Box through
    ``reach_pytorch_model`` and on a Star through ``NeuralNetwork.reach``.
    ``torch.fx`` makes these ``call_function`` (or ``call_method``) nodes,
    which have no reach rule today and are refused with
    ``NotImplementedError``; that passes. An answer must hold the torch
    float64 output of 128 member points of a box eight times ``flat_box``,
    decided by one LP per point on the returned rows. Catches a constant
    operation that is answered as the identity or as another operation.
    ``k - x`` traces to the
    method call ``k.sub(x)`` on the constant, which the walker refuses as an
    unsupported method; no small
    change to the walker makes it answer, so it has no cell here."""
    lo, hi = flat_box(3, seed=20)
    lo, hi = 8 * lo, 8 * hi
    points = samples(lo, hi)
    outputs = replay_torch(_ConstantJoin(form).eval(), points, [1, 3])
    routes = [("reach_pytorch_model", kind)
              for kind in ("Star", "Zono", "Box")]
    routes.append(("NeuralNetwork.reach", "Star"))
    for route, kind in routes:
        what = f"{route} of a torch model {form}, {kind}"
        model = _ConstantJoin(form).eval()
        try:
            if route == "NeuralNetwork.reach":
                sets = NeuralNetwork(model).reach(flat_set(kind, lo, hi),
                                                  method="approx")
            else:
                sets = _reach(model, flat_set(kind, lo, hi), "approx", what)
        except NotImplementedError:
            continue
        if not sets:
            pytest.fail(f"{what}: no set returned for a non-empty input")
        assert_covered(_one_set(sets, what), outputs, what)
