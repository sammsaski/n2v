"""Soundness cells for ONNX Transpose (``OnnxTranspose``) on every set type.

Cells:

- Transpose of an ImageStar or ImageZono through ``reach_layer``
  (``_transpose_sets_star``, ``_transpose_sets_zono``): the identity, a swap
  of two axes and the two rotations of channel, height and width (the
  rotations are not their own inverse), on one set, on several sets at once,
  on an ImageStar with a binding row and on sets with more generators than
  entries; and the permutations an image set cannot answer (the default
  perm, which reverses all axes; a perm that moves the batch axis; a perm of
  the wrong rank or an empty one), which must be refused or answered soundly.
- Transpose of a flat Star, Zono, Box, Hexatope or Octatope through
  ``reach_layer`` (``_flat_perm_is_identity``): a flat set carries no tensor
  shape, so only a perm that moves nothing but the size-1 batch axis leaves
  its entries in place; every other perm, the default one included, must be
  refused (or answered soundly for the tensor the set stands for).
- The chain affine, Transpose, affine through ``reach_pytorch_model``: Conv,
  Transpose, Conv on image sets, where the second Conv reads the stored image
  shape; Mul, Add, Transpose, Mul on a flat set with a known tensor shape
  (``input_shape``, the shape-aware row permutation in ``reach.py``), with an
  exact ReLU in front so that several Stars reach the Transpose; both chains
  under ``exact`` against the exact image in rationals; a single
  Transpose node on a Hexatope or Octatope, which must be refused; the bound
  pre-pass (``precompute_bounds``) in front of a ReLU that reads the
  transposed image (``_ibp_transpose``); the routes on which no tensor shape
  is tracked; the refusal of ``method='exact'`` on the set types that have no
  exact method; and the torch spellings of a transpose (``permute``,
  ``transpose``), which the walker must refuse or answer soundly.

Oracle: the true outputs of sample points (corners included), by
onnxruntime for a model and by torch in float64 for a layer; every input and
weight is dyadic, so both are exact. Each is a member of a returned set by the
LP on the set's own rows (``assert_covered``). Where the method claims the
exact image (an ImageStar, ``method='exact'``), corners of every returned set
mapped back by the inverse permutation must lie in an input set, and through
the model both directions are decided in rationals (``tests/_oracles``).

Layout: a flat set is in the tensor's row-major order; an image set stores
height, width, channel. An image output is compared with ``out_chw`` set to
the permuted (C, H, W) shape, which also requires every returned image set to
store exactly that height, width and channel count: a result whose rows are
right and whose stored height and width are swapped fails here, as it would
for the next spatial layer.

Defects marked here: a flat set with no tracked tensor shape whose Transpose
falls to the dispatcher's identity test, which takes axis 0 for a size-1
batch axis and accepts a perm of any length (registry GRAPH-29); the
default perm on a flat set reached without ``input_shape`` (registry
SHAPE-2); a model object whose Transpose module kept the default perm that
an earlier reach resolved for the wrong rank (SHAPE-9).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper
from onnx2torch.node_converters.transpose import OnnxTranspose

from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests._oracles.membership import hwc, rows, uncovered
from tests.cell_support import (assert_covered, constrained_image_star,
                                constrained_star,
                                flat_box, flat_set, hwc_index, image_set,
                                pixel_box, reach_model, reach_one_layer,
                                replay_onnx, replay_torch, samples, save_onnx)
from tests.known_bugs import DID_NOT_RAISE, known_bug

_C, _H, _W = 2, 3, 5
_ILO, _IHI = pixel_box(_C, _H, _W, seed=7)
# The same pixels scaled by 8: bounds well beyond [-1, 1] and intervals wider
# than 2, so a result that reads one predicate variable in place of another
# cannot hide inside the unit range every box variable has.
_SLO, _SHI = 8 * _ILO, 8 * _IHI

# A missing tensor shape is refused by the kernels, a set the model entry
# cannot take under 'exact' by the entry itself (ValueError today).
_REFUSALS = (NotImplementedError, ValueError)


def _permuted(chw, perm):
    """The (C, H, W) shape after a batch-preserving rank-4 ``perm``."""
    return tuple(chw[p - 1] for p in perm[1:])


def _parts(lo, hi, shares=(1, 2, 5)):
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


def _corners(s, n, seed):
    """Up to ``n`` points of ``s`` in its stored order: corners of its
    predicate box that keep its rows, through its stored basis."""
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


def _assert_every_point_has_a_preimage(result, input_sets, perm, what):
    """Corners of every returned image set, put back in tensor order, moved by
    the inverse permutation and stored again, lie in an input set."""
    inverse = list(np.argsort(perm))
    back = []
    for k, s in enumerate(result):
        out_chw = (s.num_channels, s.height, s.width)
        order = hwc_index(*out_chw)
        for y in _corners(s, 24, k):
            chw = np.empty(y.size)
            chw[order] = y
            x = np.transpose(chw.reshape(1, *out_chw), inverse)
            back.append(hwc(x, _C, _H, _W))
    missed = uncovered(input_sets, back)
    assert not missed, (
        f"{what}: {len(missed)} of {len(back)} corners of the result have no "
        f"preimage in the input (largest distance {max(m[1] for m in missed):.3g})")


def _single_transpose(tmp_path, perm, in_shape, name="t"):
    """An ONNX graph that is one Transpose node (no ``perm`` attribute when
    ``perm`` is None)."""
    out_shape = (np.transpose(np.zeros(in_shape), perm).shape
                 if perm is not None else tuple(reversed(in_shape)))
    attrs = {} if perm is None else {"perm": list(perm)}
    return save_onnx(tmp_path, [helper.make_node("Transpose", ["X"], ["Y"],
                                                 **attrs)],
                     list(in_shape), list(out_shape), name=name)


# --------------------------------------------------------------------------- #
# Image sets through reach_layer                                                #
# --------------------------------------------------------------------------- #
# The identity, a swap of height and width, and the two rotations of
# (C, H, W); the other swaps run the same code with the same kind of index
# change as the swap kept here.
_PERMS4 = ([0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [0, 3, 1, 2])


@pytest.mark.parametrize("perm", _PERMS4, ids=lambda p: "".join(map(str, p)))
@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
def test_transpose_layer_on_an_image_set_holds_every_output(kind, method, perm):
    """Batch-preserving perms of a (2, 3, 5) image with a different interval
    per pixel. Catches generators permuted along the stored axes instead of
    the tensor axes, the inverse perm applied in place of the perm (a
    three-axis rotation), the centre left in the old pixel order, rows
    flattened in channel-first order, and a result that stores its new
    height and width swapped. For the ImageStar, the result must also hold
    nothing beyond the image."""
    what = f"Transpose{perm} on {kind}"
    input_set = image_set(kind, _ILO, _IHI)
    result = reach_one_layer(OnnxTranspose(perm=list(perm)), [input_set],
                             method, what)
    assert_covered(result, replay_torch(OnnxTranspose(perm=list(perm)),
                                        samples(_ILO, _IHI), [1, _C, _H, _W]),
                   what, out_chw=_permuted((_C, _H, _W), perm))
    if method == "exact":
        _assert_every_point_has_a_preimage(result, [input_set], perm, what)


@pytest.mark.parametrize("perm", ([0, 2, 3, 1],),
                         ids=lambda p: "".join(map(str, p)))
@pytest.mark.parametrize("kind,inputs", [("ImageStar", "several sets"),
                                         ("ImageZono", "several sets"),
                                         ("ImageStar", "binding row")])
def test_transpose_layer_on_several_or_cut_image_sets(kind, inputs, perm):
    """Transpose of three image sets of different sizes at once, and of an
    ImageStar cut by a binding row that is not a box row. Catches a loop over
    the input sets that returns after the first set or transposes one set
    for all, and a transpose that touches the constraint rows."""
    what = f"Transpose{perm} on {inputs} of {kind}"
    if inputs == "binding row":
        star, points = constrained_image_star(_ILO, _IHI, seed=3)
        sets = [star]
    else:
        sets = [image_set(kind, a, b) for a, b in _parts(_ILO, _IHI)]
        points = samples(_ILO, _IHI)
    result = reach_one_layer(OnnxTranspose(perm=list(perm)), sets, "approx",
                             what)
    assert_covered(result, replay_torch(OnnxTranspose(perm=list(perm)), points,
                                        [1, _C, _H, _W]),
                   what, out_chw=_permuted((_C, _H, _W), perm))


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("perm", [None, [1, 0, 2, 3]],
                         ids=["default", "batch-moved"])
def test_transpose_an_image_set_cannot_hold_is_refused_or_sound(kind, perm):
    """The default perm (all axes reversed, so the batch axis goes last) and
    a perm that moves the batch axis: the output has a leading axis of size 2
    or 5, which no image set of batch 1 stands for. The call must refuse, or
    return sets that hold every true output flattened in the output tensor's
    own order (an image set larger than 1x1 fails that check, since nothing
    says how its entries line up). Catches a missing perm read as the
    identity, and a batch-moving perm let into the image arithmetic."""
    what = f"Transpose({perm}) on {kind}"
    try:
        result = reach_one_layer(OnnxTranspose(perm=perm), [image_set(kind, _ILO, _IHI)],
                                 "approx", what)
    except NotImplementedError:
        return
    assert_covered(result, replay_torch(OnnxTranspose(perm=perm),
                                        samples(_ILO, _IHI), [1, _C, _H, _W]),
                   what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("perm", [[0, 2, 1], [0, 1, 2, 3, 4]],
                         ids=["rank3", "rank5"])
def test_transpose_of_the_wrong_rank_on_an_image_set_is_refused(kind, perm):
    """A perm whose length is not the rank of the image tensor: torch raises
    on the true network, so the reach must raise too."""
    with pytest.raises(NotImplementedError):
        reach_one_layer(OnnxTranspose(perm=perm), [image_set(kind, _ILO, _IHI)],
                        "approx", f"Transpose({perm}) on {kind}")


@pytest.mark.parametrize("kind", [
    "ImageStar",
    pytest.param("Star", marks=known_bug(
        7, "A Transpose with an empty perm, which torch and onnxruntime "
        "reject for a tensor of any rank, is answered on a flat set: the "
        "dispatcher's identity test (_flat_perm_is_identity) finds no "
        "non-batch axis out of order and returns the set as it is",
        raises=DID_NOT_RAISE)),
])
def test_transpose_with_an_empty_perm_is_refused(kind):
    """``perm=[]`` is a value, not an absent perm: torch's ``permute([])``
    and onnxruntime raise for a tensor of rank 1 or more, while the default
    (no perm) reverses all axes. So the reach must raise, on an image set and
    on a flat set (whatever tensor the flat set stands for, its rank is at
    least 1). Catches an empty perm read as the default or as the identity.
    The flat set is answered today (strict expected failure); the image set
    is the passing neighbour."""
    input_set = (image_set(kind, _ILO, _IHI) if kind == "ImageStar"
                 else flat_set(kind, _ILO.reshape(-1), _IHI.reshape(-1)))
    with pytest.raises(_REFUSALS):
        reach_one_layer(OnnxTranspose(perm=[]), [input_set], "approx",
                        f"Transpose([]) on {kind}")


# --------------------------------------------------------------------------- #
# Flat sets through reach_layer                                                 #
# --------------------------------------------------------------------------- #
# The tensor a flat set of 6 or 12 entries stands for. A perm that only moves
# the size-1 batch axis leaves the entries in place; the identity test is
# perm-independent on that path, so one such perm is kept. Of the perms that
# reorder, kept are: no perm; the batch axis in the middle with the others
# swapped; an ascending pair before a swapped one; a swapped pair before an
# ascending one; all axes reversed with the batch axis last.
_FLAT_SHAPES = {3: (1, 2, 3), 4: (1, 2, 2, 3)}
_KEEPS = [1, 2, 3, 0]
_REORDERS = ((None, 3), ([2, 0, 1], 3), ([0, 1, 3, 2], 4), ([0, 2, 1, 3], 4),
             ([3, 2, 1, 0], 4))


def _flat_case(perm, rank=None):
    shape = _FLAT_SHAPES[len(perm) if perm is not None else rank]
    n = int(np.prod(shape))
    lo, hi = flat_box(n, seed=n)
    return shape, n, lo, hi


def _flat_ids(v):
    return "".join(map(str, v)) if isinstance(v, list) else str(v)


@pytest.mark.parametrize("kind,inputs,perm",
                         [(k, inputs, _KEEPS) for inputs in ("one set",
                                                             "several sets")
                          for k in ("Star", "Zono", "Box")]
                         + [("Star", "binding row", _KEEPS)],
                         ids=_flat_ids)
def test_transpose_that_moves_only_the_batch_axis_keeps_a_flat_set(kind, inputs,
                                                                   perm):
    """A perm that moves only the size-1 batch axis leaves the entries of the
    tensor in place, so a flat set may be returned as it is; refusing is
    also sound. Catches a flat answer that reorders the entries or drops a set
    of several, through the same check the refusal cells below rely on."""
    shape, n, lo, hi = _flat_case(perm)
    what = f"Transpose{perm} on {inputs} of {kind} (tensor {shape})"
    if inputs == "binding row":
        star, points = constrained_star(lo, hi, seed=1)
        sets = [star]
    else:
        boxes = [(lo, hi)] if inputs == "one set" else _parts(lo, hi)
        sets = [flat_set(kind, a, b) for a, b in boxes]
        points = samples(lo, hi)
    try:
        result = reach_one_layer(OnnxTranspose(perm=list(perm)), sets, "approx",
                                 what)
    except NotImplementedError:
        return
    assert_covered(result, replay_torch(OnnxTranspose(perm=list(perm)), points,
                                        list(shape)), what)


@pytest.mark.parametrize("kind,perm,rank",
                         [(k, p, r) for k in ("Box", "Star", "Zono")
                          for p, r in _REORDERS]
                         + [(k, [0, 2, 1], 3) for k in ("Hexatope",
                                                        "Octatope")],
                         ids=_flat_ids)
def test_transpose_that_reorders_a_flat_set_is_refused(kind, perm, rank):
    """A perm that reorders the entries of the tensor (the default one
    reverses all axes; ``[2, 0, 1]`` moves the batch axis and swaps the other
    two; ``[0, 1, 3, 2]`` keeps an ascending pair in front of a swapped one,
    ``[0, 2, 1, 3]`` the other way round). A flat set does not say what
    tensor it stands for, so the only right answer is a refusal; the cell
    also checks that a returned set would hold the outputs of the tensor of
    shape (1, 2, 3) or (1, 2, 2, 3). Catches an identity test that reads a
    missing perm as the identity, drops the first perm entry instead of the
    batch axis, asks for any ascending pair of neighbours instead of all of
    them, looks at the first or the last pair only, or asks for descending
    axes; and a Hexatope or Octatope lane that passes the set through."""
    shape, n, lo, hi = _flat_case(perm, rank)
    what = f"Transpose({perm}) on {kind} (tensor {shape})"
    with pytest.raises(NotImplementedError):
        result = reach_one_layer(OnnxTranspose(perm=perm), [flat_set(kind, lo, hi)],
                                 "approx", what)
        assert_covered(result, replay_torch(OnnxTranspose(perm=perm),
                                            samples(lo, hi), list(shape)), what)


# --------------------------------------------------------------------------- #
# The chain affine, Transpose, affine through the model entry point             #
# --------------------------------------------------------------------------- #
# Conv 1: (2, 3, 5) -> (3, 2, 5), a 2x1 kernel; every axis of its output has
# its own length, so every perm gives a different shape.
_K1 = np.array([[[[1], [-2]], [[1], [1]]],
                [[[2], [1]], [[-1], [0]]],
                [[[0], [1]], [[2], [-1]]]], np.float32) / 4           # (3,2,2,1)
_B1 = np.array([0.25, -0.5, 0.125], np.float32)
_MID = (3, _H - 1, _W)


def _k2(channels):
    """A dyadic 1x2 kernel from ``channels`` to 2 channels, mixed signs."""
    k = (np.arange(2 * channels * 2).reshape(2, channels, 1, 2) % 5 - 2)
    return k.astype(np.float32) / 4


def _image_chain(tmp_path, perm, middle_relu=False, name="ctc"):
    """X (1, 2, 3, 5) -> Conv -> Transpose(perm) [-> Relu] -> Conv (1x2)."""
    mid = _permuted(_MID, perm)
    out = (2, mid[1], mid[2] - 1)
    nodes = [helper.make_node("Conv", ["X", "K1", "b1"], ["A"]),
             helper.make_node("Transpose", ["A"], ["T" if middle_relu else "Z"],
                              perm=list(perm))]
    if middle_relu:
        nodes.append(helper.make_node("Relu", ["T"], ["Z"]))
    nodes.append(helper.make_node("Conv", ["Z", "K2", "b2"], ["Y"]))
    inits = dict(K1=_K1, b1=_B1, K2=_k2(mid[0]),
                 b2=np.array([0.5, -0.25], np.float32))
    path = save_onnx(tmp_path, nodes, [1, _C, _H, _W], [1, *out], inits=inits,
                     name=name)
    return path, out


@pytest.mark.parametrize("kind,inputs,perm",
                         [("ImageStar", "box", [0, 1, 3, 2]),
                          ("ImageStar", "box", [0, 2, 3, 1]),
                          ("ImageZono", "box", [0, 1, 3, 2]),
                          ("ImageZono", "box", [0, 2, 3, 1]),
                          ("ImageStar", "binding row", [0, 2, 3, 1])],
                         ids=_flat_ids)
def test_conv_transpose_conv_holds_every_onnxruntime_output(tmp_path, kind,
                                                            inputs, perm):
    """Conv, Transpose, Conv through ``reach_pytorch_model`` (with
    ``input_shape``, as the runner calls it), on the pixel box scaled by 8
    (bounds well beyond [-1, 1], intervals wider than 2). The second Conv
    reads the transposed set with its stored height, width and channel
    count, so a result whose rows are right and whose shape is recorded
    wrongly is caught here as well as by the layer cells."""
    path, out = _image_chain(tmp_path, perm)
    what = f"Conv, Transpose{perm}, Conv on {kind} {inputs}"
    if inputs == "binding row":
        input_set, points = constrained_image_star(_SLO, _SHI, seed=8)
    else:
        input_set, points = image_set(kind, _SLO, _SHI), samples(_SLO, _SHI)
    sets = reach_model(load_onnx(path), input_set, "approx", what,
                       input_shape=(_C, _H, _W))
    assert_covered(sets, replay_onnx(path, points, [1, _C, _H, _W]), what,
                   out_chw=out)


def _degenerate_pixels(scale=8, seed=11):
    """The (2, 3, 5) pixel box with three pixels of positive width (one per
    row of the image, both channels used) and every other pixel pinned to a
    value of its own, times ``scale``: few predicate variables for an exact
    vertex enumeration, and zero-width coordinates."""
    lo, hi = pixel_box(_C, _H, _W, seed=seed)
    live = np.zeros((_C, _H, _W), bool)
    for c, h, w in ((0, 0, 1), (1, 1, 3), (0, 2, 4)):
        live[c, h, w] = True
    return scale * lo, scale * np.where(live, hi, lo)


@pytest.mark.parametrize("kind", ["ImageStar", "Star"])
def test_exact_transpose_in_a_model_is_the_exact_image(tmp_path, kind):
    """``method='exact'`` through ``reach_pytorch_model`` with
    ``input_shape``, on a set of three free coordinates (the others of zero
    width) cut by a binding row, bounds scaled by 8: Conv, Transpose
    (0, 2, 3, 1), Conv on an ImageStar (the dispatcher's image path), and
    Mul, Add, Transpose (0, 3, 1, 2), Mul, Add on a flat Star of a
    (1, 2, 2, 3) tensor (the shape-aware row selection). Each network is
    affine, so its exact image is the affine image of the predicate
    polytope; the map is read from onnxruntime on the unit inputs (exact for
    these dyadic weights) and every comparison is in rationals: every vertex
    of the true image is in the result, and every vertex of the result has a
    preimage. Catches the exact lane answering with a set larger or smaller
    than the image, rows or generators permuted by the wrong (inverse) perm
    or along the wrong axes, and a result whose stored shape does not match
    its rows."""
    if kind == "ImageStar":
        perm, in_shape = [0, 2, 3, 1], (1, _C, _H, _W)
        path, out = _image_chain(tmp_path, perm, name="ctc-exact")
        lo, hi = _degenerate_pixels()
        star, points = constrained_image_star(lo, hi, seed=4)
        order = hwc_index(_C, _H, _W)            # stored = tensor[order]
        out_order, out_chw = hwc_index(*out), out
    else:
        perm, in_shape = [0, 3, 1, 2], (1, 2, 2, 3)
        path = _flat_chain(tmp_path, in_shape, perm, relu=False,
                           name="mat-exact")
        lo, hi = flat_box(12, seed=5)
        lo, hi = 8 * lo, 8 * np.where(np.isin(np.arange(12), (1, 6, 10)),
                                      hi, lo)
        star, points = constrained_star(lo, hi, seed=4)
        order = np.arange(12)
        out_order, out_chw = np.arange(12), None
    what = f"exact Transpose{perm} in a model on a {kind}"
    sets = reach_model(load_onnx(path), star, "exact", what,
                       input_shape=in_shape[1:])
    assert_covered(sets, replay_onnx(path, points, list(in_shape)), what,
                   out_chw=out_chw)
    n, k = order.size, star.V.shape[-1] - 1
    computed = [rational.star(s) for s in sets]
    if len(sets) != 1 or len(computed[0][4]) != k:
        pytest.fail(f"{what}: an affine chain must return one set over the "
                    f"input's {k} predicate variables; got {len(sets)} "
                    f"set(s) over {[len(s[4]) for s in computed]}")
    V = star.V.reshape(n, k + 1)
    centre, basis = np.empty(n), np.empty((n, k))
    centre[order], basis[order] = V[:, 0], V[:, 1:]       # tensor order
    y = replay_onnx(path, [np.zeros(n)] + list(np.eye(n)), list(in_shape))
    offset = y[0]
    M = np.stack([yi - offset for yi in y[1:]], axis=1)
    to_stored = np.eye(offset.size)[out_order]
    pieces = rational.polytope(star.C.tolist(), star.d.reshape(-1).tolist())
    pieces = rational.affine(pieces, basis.tolist(), centre.tolist())
    pieces = rational.affine(pieces, M.tolist(), offset.tolist())
    pieces = rational.affine(pieces, to_stored.tolist())
    escaped = rational.escapes(pieces, *computed)
    assert not escaped, (f"{what}: {len(escaped)} vertices of the true image "
                         f"are in no returned set, first {escaped[0]}")
    extra = rational.unreached(pieces, rational.set_vertices(computed[0]))
    assert not extra, (f"{what}: {len(extra)} vertices of the result have no "
                       f"preimage, first {extra[0][1]}")


def _straddling_pixels(seed=5):
    """Every pixel's interval holds [-1, 1] and runs to its own ends in
    [-2, -1] and [1, 2] (dyadic); with the first Conv's kernel and bias every
    pre-activation then straddles zero, with a range of its own."""
    rng = np.random.default_rng(seed)
    lo = -(1 + rng.integers(0, 5, size=(_C, _H, _W)) / 4)
    hi = 1 + rng.integers(0, 5, size=(_C, _H, _W)) / 4
    return lo, hi


def _first_conv_range(lo, hi):
    """Exact range of every output of the first Conv over the box, by
    interval arithmetic in float64 (exact for these dyadic numbers)."""
    K, b = _K1.astype(np.float64), _B1.astype(np.float64)
    lower, upper = np.empty(_MID), np.empty(_MID)
    for o in range(_MID[0]):
        for h in range(_MID[1]):
            for w in range(_MID[2]):
                terms = [(K[o, c, dh, 0] * lo[c, h + dh, w],
                          K[o, c, dh, 0] * hi[c, h + dh, w])
                         for c in range(_C) for dh in range(2)]
                lower[o, h, w] = b[o] + sum(min(t) for t in terms)
                upper[o, h, w] = b[o] + sum(max(t) for t in terms)
    return lower, upper


@pytest.mark.parametrize("precompute", ["ibp", "zono"])
def test_transpose_before_relu_with_the_bound_prepass(tmp_path, precompute):
    """Conv, Transpose (a three-axis rotation), ReLU, Conv on an ImageStar
    with ``precompute_bounds``: the pre-pass permutes the bounds itself and
    hands them to the ReLU in the stored order of the transposed image.
    Every pre-activation straddles zero with a range of its own, so bounds
    of another neuron still meet the star's own enclosure (the ReLU keeps
    them instead of dropping the layer's bounds) and narrow it: catches a
    pre-pass that leaves the bounds in the old order or applies the inverse
    rotation, which cuts the ReLU's triangle short of the true outputs."""
    perm = [0, 2, 3, 1]
    path, out = _image_chain(tmp_path, perm, middle_relu=True, name="ctrc")
    lo, hi = _straddling_pixels()
    lower, upper = _first_conv_range(lo, hi)
    if not (np.all(lower < 0) and np.all(upper > 0)):
        pytest.fail("the cell needs every pre-activation to straddle zero")
    what = f"Conv, Transpose{perm}, ReLU, Conv with precompute_bounds={precompute!r}"
    sets = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                       "approx", what, input_shape=(_C, _H, _W),
                       precompute_bounds=precompute)
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, _C, _H, _W]),
                   what, out_chw=out)


def _flat_chain(tmp_path, shape, perm, relu, name):
    """X ``shape`` -> Mul -> Add [-> Relu] -> Transpose(perm) -> Mul -> Add:
    elementwise dyadic constants in front of and behind the Transpose, so
    the constants behind it are applied in the permuted order."""
    n = int(np.prod(shape))
    out_shape = (np.transpose(np.zeros(shape), perm).shape if perm is not None
                 else tuple(reversed(shape)))
    grid = np.arange(n)
    inits = dict(m1=((grid % 4 + 1) / 2).reshape(shape).astype(np.float32),
                 a1=((grid % 3 - 1) / 4).reshape(shape).astype(np.float32),
                 m2=((grid % 3 + 1) / 4 * (-1) ** grid).reshape(out_shape)
                 .astype(np.float32),
                 a2=((grid % 5 - 2) / 8).reshape(out_shape).astype(np.float32))
    nodes = [helper.make_node("Mul", ["X", "m1"], ["A"]),
             helper.make_node("Add", ["A", "a1"], ["B"])]
    if relu:
        nodes.append(helper.make_node("Relu", ["B"], ["R"]))
    attrs = {} if perm is None else {"perm": list(perm)}
    nodes += [helper.make_node("Transpose", ["R" if relu else "B"], ["T"], **attrs),
              helper.make_node("Mul", ["T", "m2"], ["U"]),
              helper.make_node("Add", ["U", "a2"], ["Y"])]
    return save_onnx(tmp_path, nodes, list(shape), list(out_shape), inits=inits,
                     name=name)


@pytest.mark.parametrize(
    "shape,perm", [((1, 2, 3), [0, 2, 1]), ((1, 2, 3), None),
                   ((1, 2, 2, 3), [0, 3, 1, 2])],
    ids=["021", "default", "0312"])
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_transpose_of_a_flat_set_with_a_known_shape_is_sound(tmp_path, kind,
                                                             shape, perm):
    """Mul, Add, Transpose, Mul, Add on a flat set with ``input_shape``: the
    model entry knows the tensor shape and permutes the rows exactly
    (``_handle_onnx_transpose_flat``; with no perm attribute the network
    reverses all axes). The box is scaled by 8 and has one coordinate of
    zero width. Catches a row permutation built from the inverse perm (the
    rotation 0312), from the wrong shape, or not applied, for each of the
    three set types the row selection serves."""
    path = _flat_chain(tmp_path, shape, perm, relu=False, name="mat")
    n = int(np.prod(shape))
    lo, hi = flat_box(n, seed=n + 1, zero_width=True)
    lo, hi = 8 * lo, 8 * hi
    what = f"Mul, Add, Transpose({perm}), Mul, Add on {kind} (tensor {shape})"
    sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                       what, input_shape=shape[1:])
    assert_covered(sets, replay_onnx(path, samples(lo, hi), list(shape)), what)


@pytest.mark.parametrize("shape,perm", [((1, 2, 3), [0, 2, 1])], ids=["021"])
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_transpose_of_a_hexatope_or_octatope_through_the_model(tmp_path, kind,
                                                               shape, perm):
    """A graph that is a single Transpose node, on a Hexatope or Octatope
    with ``input_shape``: the shape-aware row selection has no rule for
    these types and must refuse (``NotImplementedError``); should it answer,
    the result is also checked against every onnxruntime output, so that a
    reordering Transpose answered as the identity fails with its own
    message. Catches a row selection or a dispatcher lane that passes these
    two types through."""
    path = _single_transpose(tmp_path, perm, shape)
    n = int(np.prod(shape))
    lo, hi = flat_box(n, seed=n + 1)
    what = f"Transpose({perm}) on {kind} (tensor {shape})"
    with pytest.raises(NotImplementedError):
        sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                           what, input_shape=shape[1:])
        assert_covered(sets, replay_onnx(path, samples(lo, hi), list(shape)),
                       what)


def test_exact_relu_hands_several_stars_to_a_flat_transpose(tmp_path):
    """Mul, Add, ReLU, Transpose, Mul, Add on a Star with ``method='exact'``
    and a known shape, the box scaled by 8: the exact ReLU splits the Star,
    so the shape-aware Transpose permutes a list of several Stars. Catches a
    permutation applied to the first Star only."""
    shape, perm = (1, 2, 3), [0, 2, 1]
    path = _flat_chain(tmp_path, shape, perm, relu=True, name="marat")
    lo, hi = flat_box(6, seed=7)
    lo, hi = 8 * lo, 8 * hi
    what = "Mul, Add, ReLU, Transpose[0, 2, 1], Mul, Add on Star (exact)"
    sets = reach_model(load_onnx(path), flat_set("Star", lo, hi), "exact", what,
                       input_shape=shape[1:])
    if len(sets) < 2:
        pytest.fail(f"{what}: the exact ReLU returned {len(sets)} set; the cell "
                    f"needs several sets at the Transpose")
    assert_covered(sets, replay_onnx(path, samples(lo, hi), list(shape)), what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono", "Star", "Zono"])
def test_transpose_of_a_set_with_more_generators_than_entries_holds_every_output(
        tmp_path, kind):
    """A Transpose (0, 2, 3, 1) of a (2, 3, 5) tensor in a model, on a set
    whose inner shape is not the one ``from_bounds`` gives: 34 generators or
    predicate variables (dense, mixed signs, dyadic) for 30 entries, the
    shape a relaxation leaves. The image sets take the dispatcher's index
    arithmetic, the flat sets the shape-aware row permutation with
    ``input_shape``. Catches a kernel that takes the number of generators
    for the number of entries (indexes, reshapes or truncates the generator
    columns by the entry count) and one that permutes generator columns
    instead of rows."""
    perm = [0, 2, 3, 1]
    path = _single_transpose(tmp_path, perm, (1, _C, _H, _W), name="wide")
    rng = np.random.default_rng(23)
    n = _C * _H * _W
    k = n + 4
    c = rng.integers(-24, 25, size=n) / 2.0
    G = rng.integers(-8, 9, size=(n, k)) / 4.0
    preds = [rng.choice([-1.0, 1.0], size=k) for _ in range(64)]
    preds += [rng.integers(-2, 3, size=k) / 2.0 for _ in range(64)]
    stored = [c + G @ a for a in preds]
    if kind.startswith("Image"):
        order = hwc_index(_C, _H, _W)
        points = []
        for y in stored:                     # c, G are in stored HWC order
            x = np.empty(n)
            x[order] = y
            points.append(x.reshape(_C, _H, _W))
        if kind == "ImageStar":
            input_set = ImageStar(
                np.hstack([c[:, None], G]).reshape(_H, _W, _C, k + 1),
                np.zeros((0, k)), np.zeros((0, 1)), -np.ones((k, 1)),
                np.ones((k, 1)), _H, _W, _C)
        else:
            input_set = ImageZono(c[:, None], G, _H, _W, _C)
        options = {}
    else:
        points = stored                      # c, G are in tensor order
        input_set = (Star(np.hstack([c[:, None], G]), np.zeros((0, k)),
                          np.zeros((0, 1)), -np.ones((k, 1)), np.ones((k, 1)))
                     if kind == "Star" else Zono(c[:, None], G))
        options = {"input_shape": (_C, _H, _W)}
    what = f"Transpose{perm} on a {kind} of {n} entries and {k} generators"
    sets = reach_model(load_onnx(path), input_set, "approx", what, **options)
    assert_covered(sets, replay_onnx(path, points, [1, _C, _H, _W]), what,
                   out_chw=_permuted((_C, _H, _W), perm))


@pytest.mark.parametrize("kind", ["Zono", "ImageZono", "Box", "Hexatope",
                                  "Octatope"])
def test_exact_transpose_on_a_set_without_an_exact_method_is_refused(tmp_path,
                                                                      kind):
    """The rows (exact, Zono / ImageZono / Box / Hexatope / Octatope) of
    Transpose: the model entry refuses ``method='exact'`` on these set types
    before any layer (today ``ValueError``, '... only supports approx'), on
    the image set and on a flat set given with ``input_shape``. Catches the
    model entry passing 'exact' through to a lane that never reads the method
    and answering under the name exact."""
    path = _single_transpose(tmp_path, [0, 2, 3, 1], (1, _C, _H, _W))
    if kind == "ImageZono":
        input_set, options = image_set(kind, _ILO, _IHI), {}
    else:
        input_set = flat_set(kind, _ILO.reshape(-1), _IHI.reshape(-1))
        options = {"input_shape": (_C, _H, _W)}
    with pytest.raises(_REFUSALS):
        reach_model(load_onnx(path), input_set, "exact",
                    f"exact Transpose on {kind}", **options)


# --------------------------------------------------------------------------- #
# Routes on which no tensor shape is tracked                                    #
# --------------------------------------------------------------------------- #
def _no_shape_model(tmp_path, case):
    """``(path, out_shape, input_shape)`` for a flat set of a (1, 2, 3) tensor.

    - ``reorder``: Transpose (0, 2, 1) alone, reached without ``input_shape``:
      the shape pass's (1, 6) placeholder cannot run a rank-3 perm, so no
      shape is tracked and the dispatcher must refuse.
    - ``squeeze-swap-flatten-axis{0,1}``: Squeeze of the batch axis to
      (2, 3), Transpose (1, 0) to (3, 2), Mul by a (3, 2) constant, Unsqueeze
      to (1, 1, 3, 2), ONNX Flatten with that axis to (1, 6); reached with
      ``input_shape``. The converted Flatten with axis 0 raises in torch on a
      rank-4 input, so the shape pass records nothing (SHAPE-8) and the
      Transpose falls to the dispatcher; with axis 1 every shape is tracked.
    - ``default-perm-{no-,}input-shape``: one Transpose with no perm, which
      reverses (1, 2, 3) to (3, 2, 1), with and without ``input_shape``."""
    i64 = lambda v: np.asarray(v, dtype=np.int64)                 # noqa: E731
    if case == "reorder":
        return (_single_transpose(tmp_path, [0, 2, 1], (1, 2, 3), name=case),
                None)
    if case.startswith("squeeze-swap"):
        axis = int(case[-1])
        m = ((np.arange(6) % 5 - 2) / 4).reshape(3, 2).astype(np.float32)
        nodes = [helper.make_node("Squeeze", ["X", "ax"], ["S"]),
                 helper.make_node("Transpose", ["S"], ["T"], perm=[1, 0]),
                 helper.make_node("Mul", ["T", "m"], ["U"]),
                 helper.make_node("Unsqueeze", ["U", "ax2"], ["V"]),
                 helper.make_node("Flatten", ["V"], ["Y"], axis=axis)]
        path = save_onnx(tmp_path, nodes, [1, 2, 3], [1, 6],
                         inits={"ax": i64([0]), "ax2": i64([0, 1]), "m": m},
                         name=f"sq{axis}")
        return path, (2, 3)
    path = _single_transpose(tmp_path, None, (1, 2, 3), name="default")
    return path, (None if case == "default-perm-no-input-shape" else (2, 3))


@pytest.mark.parametrize("case", [
    "reorder",
    "squeeze-swap-flatten-axis1",
    pytest.param("squeeze-swap-flatten-axis0", marks=known_bug(
        7, "With no tensor shape tracked (an ONNX Flatten with axis 0 makes "
        "the shape pass fail, SHAPE-8), the Transpose of a flat set falls to "
        "the dispatcher's identity test, which takes axis 0 for a size-1 "
        "batch axis: after a Squeeze of the batch axis, perm [1, 0] swaps a "
        "(2, 3) tensor and the set is returned unchanged; every output "
        "escapes")),
    "default-perm-input-shape",
    pytest.param("default-perm-no-input-shape", marks=known_bug(
        12, "SHAPE-2: a flat set reached without input_shape on a model "
        "whose input is (1, 2, 3) is analysed as the (1, 6) placeholder; the "
        "Transpose with no perm is resolved for rank 2 and permutes nothing, "
        "where the network reverses all three axes; every output escapes")),
])
def test_transpose_of_a_flat_set_with_no_tracked_shape_refuses_or_holds_every_output(
        tmp_path, case):
    """The routes on which the Transpose handler has no tensor shape, or
    only a guessed one: a flat Star of a (1, 2, 3) tensor,
    box scaled by 8. Promise: refuse (``NotImplementedError``), or hold
    every onnxruntime output with the output's number of entries. Catches a
    handler that answers a reordering perm without a shape (returns the set,
    or guesses the shape) and the dispatcher's identity test taking a
    non-batch axis 0 for the batch axis; ``reorder``, the Flatten with axis 1
    and the default perm with ``input_shape`` are the passing neighbours of
    the two marked cases."""
    path, input_shape = _no_shape_model(tmp_path, case)
    lo, hi = flat_box(6, seed=7)
    lo, hi = 8 * lo, 8 * hi
    options = {} if input_shape is None else {"input_shape": input_shape}
    what = f"Transpose case {case} on a flat Star"
    try:
        sets = reach_model(load_onnx(path), flat_set("Star", lo, hi), "approx",
                           what, **options)
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, 2, 3]), what)


@pytest.mark.parametrize("history", [
    "fresh-model",
    pytest.param("reached-before-without-input-shape", marks=known_bug(
        12, "The vendored OnnxTranspose.forward stores the default perm it "
        "resolves in the module; a first reach of a flat set without "
        "input_shape runs the shape pass's forward on the (1, 6) placeholder "
        "and leaves perm [1, 0] in the model. A later reach of the same model "
        "object with input_shape then records no shape (the rank-2 perm "
        "cannot run on the (1, 2, 3) placeholder, the (2, 3) one fails at the "
        "Mul), and the dispatcher's identity test accepts the stored [1, 0]: "
        "the set is returned unpermuted and every output escapes")),
])
def test_a_model_reached_again_with_input_shape_holds_every_output(tmp_path,
                                                                   history):
    """Transpose with no perm, then Mul by a (3, 2, 1) constant, on a flat
    Star of a (1, 2, 3) tensor reached with ``input_shape``, box scaled by 8:
    on a freshly loaded model, and on the same model object after one reach
    without ``input_shape`` (whose own answer is not checked here: that is
    the identity test marked above, on SHAPE-2's route). Catches a reach that
    changes the model it is given so that a later, correct reach of it cannot
    track shapes and reads another network; the fresh model is the passing
    neighbour. A fix of the identity test alone turns the marked case into a
    refusal, which this case still reports (the promise is the answer a fresh
    model gets)."""
    c = ((np.arange(6) % 3 + 1) / 4).reshape(3, 2, 1).astype(np.float32)
    path = save_onnx(tmp_path, [helper.make_node("Transpose", ["X"], ["T"]),
                                helper.make_node("Mul", ["T", "c"], ["Y"])],
                     [1, 2, 3], [3, 2, 1], inits={"c": c}, name="again")
    model = load_onnx(path)
    lo, hi = flat_box(6, seed=7)
    lo, hi = 8 * lo, 8 * hi
    if history != "fresh-model":
        try:
            reach_pytorch_model(model, flat_set("Star", lo, hi), method="approx")
        except NotImplementedError:
            pass
    what = f"Transpose with no perm, Mul on a flat Star ({history})"
    sets = reach_model(model, flat_set("Star", lo, hi), "approx", what,
                       input_shape=(2, 3))
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, 2, 3]), what)


# --------------------------------------------------------------------------- #
# The torch spellings of a transpose                                            #
# --------------------------------------------------------------------------- #
class _TorchTranspose(nn.Module):
    """A torch model that is one native transpose, as a user writes it."""

    def __init__(self, spelling):
        super().__init__()
        self.spelling = spelling

    def forward(self, x):
        if self.spelling == "x.permute":
            return x.permute(0, 2, 3, 1)
        if self.spelling == "torch.permute":
            return torch.permute(x, (0, 3, 1, 2))
        if self.spelling == "x.transpose":
            return x.transpose(2, 3)
        return torch.transpose(x, 1, 3)


@pytest.mark.parametrize("spelling", ["x.permute", "torch.permute",
                                      "x.transpose", "torch.transpose"])
def test_a_torch_permute_or_transpose_refuses_or_holds_every_output(spelling):
    """A torch model whose one operation is ``x.permute``, ``torch.permute``,
    ``x.transpose`` or ``torch.transpose`` of a (2, 3, 5) image, reached on
    an ImageStar through ``reach_pytorch_model``: the traced method or
    function node must be refused (``NotImplementedError``, today's answer:
    the walker has no rule for it), or answered with sets that hold every
    true output (torch, float64) in the transposed shape. Catches a walker
    that takes these nodes for layout methods that keep the values (like
    ``contiguous``) and passes the set through."""
    model = _TorchTranspose(spelling)
    points = samples(_SLO, _SHI)
    outputs = replay_torch(model, points, [1, _C, _H, _W])
    out_chw = tuple(model(torch.zeros(1, _C, _H, _W)).shape[1:])
    what = f"{spelling} in a torch model on an ImageStar"
    try:
        sets = reach_model(model, image_set("ImageStar", _SLO, _SHI), "approx",
                           what)
    except NotImplementedError:
        return
    assert_covered(sets, outputs, what, out_chw=out_chw)
