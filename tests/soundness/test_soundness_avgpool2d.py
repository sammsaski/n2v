"""Soundness cells for AvgPool2d (``n2v/nn/layer_ops/avgpool2d_reach.py``).

Average pooling is linear, so the ImageStar and ImageZono kernels claim the
exact image for both methods. Every cell here checks one of:

- the result holds every true output (torch in float64, onnxruntime for ONNX
  files) of sample points, by one LP on the set's own rows;
- the result is the exact image in rationals, both directions, on a set of
  two predicate variables;
- the call refuses with ``NotImplementedError`` where the library has no rule
  or the layer's border divisor is ambiguous between torch and ONNX.

The routes are ``reach_layer``, a torch model through ``reach_pytorch_model``
and an ONNX file through ``load_onnx`` (including a flat set reshaped to an
image in the graph). Box, Hexatope and Octatope have no AvgPool2d rule in the
dispatcher; their kernels exist in the file but no public entry reaches them
(registry HEXOCT-6, HEXOCT-7).
"""

from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn.layer_ops.avgpool2d_reach import (avgpool2d_box,
                                              avgpool2d_hexatope,
                                              avgpool2d_octatope)
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_image_star,
                                constrained_star, flat_set, hwc_index,
                                image_set, image_split_cover, pixel_box,
                                reach_model, reach_one_layer, replay_onnx,
                                replay_torch, samples, save_onnx, split_cover)
from tests.known_bugs import known_bug

# =========================================================================== #
# AvgPool2d on ImageStar and ImageZono through reach_layer                      #
# =========================================================================== #
# Two oracles, neither from n2v:
#
# - the exact image in rationals (tests/_oracles/rational.py) of a set of two
#   predicate variables, under the pool's matrix read from torch in float64
#   on the unit vectors (exact where every divisor is a power of two, which
#   ``_AP_DYADIC`` lists); containment with zero slack and, as the kernel
#   claims exactness, the other direction (no vertex of the result outside
#   the image);
# - torch in float64 (onnxruntime for ONNX files) on sample points, each a
#   member of some returned set by one LP on the set's own rows
#   (tests/cell_support.assert_covered).
#
# Layout: true outputs are computed in the network's (C, H, W) order and
# converted to the stored (H, W, C) order explicitly (``out_chw=``,
# ``hwc_index``); every image has two channels and height unlike width.

# name: ((C, H, W), AvgPool2d keywords). Kernels, strides and padding differ
# between the axes; border windows of a padded pool with
# count_include_pad=False hold fewer entries; ceil_mode adds a window that
# overhangs the input; divisor_override replaces every divisor (and, with
# ceil_mode and count_include_pad, exempts an overhanging window from the
# refusal of ``_check_ceil_mode_divisor``, because torch alone defines it).
_AP_CONFIGS = {
    "k1x2-s1x2": ((2, 3, 5), dict(kernel_size=(1, 2), stride=(1, 2))),
    "k2x1-s1x2-p1x0": ((2, 3, 4), dict(kernel_size=(2, 1), stride=(1, 2),
                                       padding=(1, 0))),
    "k2x1-s1x2-p1x0-nocip": ((2, 3, 4), dict(kernel_size=(2, 1),
                                             stride=(1, 2), padding=(1, 0),
                                             count_include_pad=False)),
    "k2-s1-p1-nocip": ((2, 3, 5), dict(kernel_size=2, stride=1, padding=1,
                                       count_include_pad=False)),
    "k1x2-s1x2-ceil-nocip": ((2, 3, 5), dict(kernel_size=(1, 2),
                                             stride=(1, 2), ceil_mode=True,
                                             count_include_pad=False)),
    "k2-s2x1-div8": ((2, 4, 3), dict(kernel_size=2, stride=(2, 1),
                                     divisor_override=8)),
    "k3-s1-p1": ((2, 3, 5), dict(kernel_size=3, stride=1, padding=1)),
    "k2x1-s2x1-ceil-div4": ((2, 3, 4), dict(kernel_size=(2, 1),
                                            stride=(2, 1), ceil_mode=True,
                                            divisor_override=4)),
}
# The configurations whose every divisor is a power of two (the rational
# image is then the float64 one).
_AP_DYADIC = [n for n in _AP_CONFIGS if n != "k3-s1-p1"]


def _ap_out_chw(layer, chw):
    """The (C, H, W) shape of the layer's output on a (C, H, W) input, by
    torch."""
    with torch.no_grad():
        y = layer(torch.zeros((1,) + tuple(chw), dtype=torch.float64))
    return tuple(int(t) for t in y.shape[1:])


def _ap_matrix(layer, chw):
    """The pool as a matrix from the flat (C, H, W) input to the flat
    (C, H_out, W_out) output, read from torch in float64 on the unit
    vectors."""
    n = int(np.prod(chw))
    with torch.no_grad():
        y = layer(torch.eye(n, dtype=torch.float64).reshape((n,) + tuple(chw)))
    return y.reshape(n, -1).T.numpy()


def _ap_inputs(kind, variant, chw, seed):
    """``(input sets, sample points)``, points in the network's order."""
    lo, hi = pixel_box(*chw, seed=seed)
    if variant == "box":
        return [image_set(kind, lo, hi)], samples(lo, hi)
    if variant == "degenerate":
        # every third pixel (in (C, H, W) order) has zero width: an ImageStar
        # gives it no predicate variable, an ImageZono a zero generator
        flat_lo, flat_hi = lo.reshape(-1).copy(), hi.reshape(-1)
        flat_lo[::3] = flat_hi[::3]
        lo = flat_lo.reshape(chw)
        return [image_set(kind, lo, hi)], samples(lo, hi)
    if variant == "split":
        return image_split_cover(kind, lo, hi, 3), samples(lo, hi)
    if variant == "rows":
        star, points = constrained_image_star(lo, hi, seed=seed)
        return [star], points
    raise ValueError(variant)


# Every layer configuration on a pixel box; each other input shape on one
# configuration, because the loop over the input list, the carried rows and
# a zero-width pixel take the same code whatever the pool: several sets and
# a binding row on the configuration that ``test_avgpool_is_the_exact_image``
# cannot take (divisor 9), zero-width pixels where they fall into border
# windows with fewer entries.
_AP_SHAPE_CONFIG = {"split": "k3-s1-p1", "rows": "k3-s1-p1",
                    "degenerate": "k2-s1-p1-nocip"}


def _ap_cells():
    cells = []
    for name in _AP_CONFIGS:
        for kind, variants in (("ImageStar",
                                ("box", "split", "rows", "degenerate")),
                               ("ImageZono", ("box", "split", "degenerate"))):
            for variant in variants:
                if variant != "box" and _AP_SHAPE_CONFIG[variant] != name:
                    continue
                cells.append(pytest.param(kind, variant, name,
                                          id=f"{kind}-{name}-{variant}"))
    return cells


@pytest.mark.parametrize("kind,variant,name", _ap_cells())
def test_avgpool_holds_every_torch_output(kind, variant, name):
    """AvgPool2d through ``reach_layer`` on an ImageStar or ImageZono: a
    pixel box, three sets covering it, a star with a binding row, or a box
    with zero-width pixels. Every torch output of a sample point is in a
    returned set, by LP on the set's rows. Catches stride, padding,
    count_include_pad, ceil_mode or divisor_override not passed to the
    pooling (also with ceil_mode, count_include_pad and divisor_override
    together, which must answer), height, width and channel confused in the
    basis or the centre, the predicate bounds of the input replaced, and
    only one of several input sets answered."""
    chw, kw = _AP_CONFIGS[name]
    layer = nn.AvgPool2d(**kw)
    what = f"AvgPool2d({kw}) {kind} {variant}"
    sets, points = _ap_inputs(kind, variant, chw, seed=sum(chw))
    out = reach_one_layer(layer, sets, "approx", what)
    assert_covered(out, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=_ap_out_chw(layer, chw))


# --------------------------------------------------------------------------- #
# The exact image in rationals, on a set of two predicate variables            #
# --------------------------------------------------------------------------- #
# The predicate polygon: |a1| <= 1, |a2| <= 1 and the binding row
# a1 + a2 / 2 <= 1, which cuts off the corner (1, 1).
_AP_POLY_A = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, Fraction(1, 2)]]
_AP_POLY_B = [1, 1, 1, 1, 1]
_AP_SQUARE_A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
_AP_SQUARE_B = [1, 1, 1, 1]


def _ap_pixels(chw, seed=3):
    """``(centre, gen)`` in (C, H, W) order: pixel p is
    ``centre[p] + gen[p] . a``. Dyadic: centres in [-2, 2] by 1/4, every
    generator entry in [-1/2, 1/2] by 1/8, so pixels are all different."""
    n = int(np.prod(chw))
    rng = np.random.default_rng(seed)
    return (rng.integers(-8, 9, size=n) / 4.0,
            rng.integers(-4, 5, size=(n, 2)) / 8.0)


def _ap_low_set(chw, centre, gen, kind, extra=None):
    """``centre + gen . a`` as an ImageStar over the polygon (cut by the row
    ``extra = (row, rhs)``), or as an ImageZono over the square, in stored
    (H, W, C) order."""
    C, H, W = chw
    order = hwc_index(C, H, W)
    c, G = centre[order], gen[order]
    if kind == "ImageZono":
        return ImageZono(c.reshape(-1, 1), G.copy(), H, W, C)
    rows = _AP_POLY_A + ([extra[0]] if extra else [])
    rhs = _AP_POLY_B + ([extra[1]] if extra else [])
    A = np.array([[float(v) for v in r] for r in rows])
    b = np.array([float(v) for v in rhs]).reshape(-1, 1)
    return ImageStar(np.hstack([c[:, None], G]), A, b, -np.ones((2, 1)),
                     np.ones((2, 1)), H, W, C)


def _ap_stored(pieces, out_chw):
    """``pieces`` with their outputs permuted to the stored (H, W, C)
    order."""
    order = hwc_index(*out_chw)
    m = len(order)
    return rational.affine(pieces, [[int(order[k] == j) for j in range(m)]
                                    for k in range(m)])


def _ap_check_exact(out, pieces, what):
    """Every vertex of every exact piece is in one returned set (zero slack),
    and every vertex of every returned set is in the exact image."""
    computed = [rational.star(s) for s in out]
    bad = rational.escapes(pieces, *computed)
    assert not bad, (f"{what}: {len(bad)} vertices of the exact image are "
                     f"in no returned set (first "
                     f"{[float(v) for v in bad[0][1]]})")
    spread = rational.straddlers(pieces, *computed)
    assert not spread, f"{what}: exact pieces {spread} straddle the result"
    for k, s in enumerate(computed):
        far = rational.unreached(pieces, rational.set_vertices(s))
        assert not far, (f"{what}: returned set {k} has a vertex that is no "
                         f"output of the pool: "
                         f"{[float(v) for v in far[0][1]]}")


def _ap_exact_cells():
    """One set on every dyadic configuration; the two halves (a list of two
    sets, each cut by a row) on one configuration, since the loop over the
    list does not depend on the pool."""
    cells = []
    for name in _AP_DYADIC:
        for kind in ("ImageStar", "ImageZono"):
            for split in (False, True):
                if split and name != "k2-s1-p1-nocip":
                    continue
                cells.append(pytest.param(
                    name, kind, split,
                    id=f"{name}-{kind}-{'two-halves' if split else 'one-set'}"))
    return cells


@pytest.mark.parametrize("name,kind,split", _ap_exact_cells())
def test_avgpool_is_the_exact_image(name, kind, split):
    """AvgPool2d on a set of two predicate variables (an ImageStar over a
    polygon with a binding row, an ImageZono over the square), alone or as
    the two halves a1 <= 0 and a1 >= 0: the result holds every vertex of the
    exact image with zero slack, and no vertex of the result lies outside the
    image. Catches a wrong divisor or window (an output scaled or shifted),
    pixels read in the wrong order, the centre pooled differently from the
    generators, and a second input set not answered."""
    chw, kw = _AP_CONFIGS[name]
    layer = nn.AvgPool2d(**kw)
    out_chw = _ap_out_chw(layer, chw)
    centre, gen = _ap_pixels(chw)
    square = kind == "ImageZono"
    A, b = (_AP_SQUARE_A, _AP_SQUARE_B) if square else (_AP_POLY_A, _AP_POLY_B)
    pieces = rational.affine(rational.polytope(A, b), gen.tolist(),
                             centre.tolist())
    pieces = rational.affine(pieces, _ap_matrix(layer, chw).tolist())
    if split:
        if square:
            # a zonotope has no rows: its halves a1 in [-1, 0] and [0, 1]
            # are the zonotopes with a1 = +-1/2 + a1' / 2
            sets = []
            for sign in (-1.0, 1.0):
                g = gen.copy()
                g[:, 0] *= 0.5
                sets.append(_ap_low_set(chw, centre + sign * g[:, 0], g,
                                        kind))
        else:
            sets = [_ap_low_set(chw, centre, gen, kind, ([1, 0], 0)),
                    _ap_low_set(chw, centre, gen, kind, ([-1, 0], 0))]
        pieces = rational.split(pieces, [1, 0], 0)
    else:
        sets = [_ap_low_set(chw, centre, gen, kind)]
    what = f"AvgPool2d {name} on {kind}, {'two halves' if split else 'one set'}"
    out = reach_one_layer(layer, sets, "exact", what)
    _ap_check_exact(out, _ap_stored(pieces, out_chw), what)


# --------------------------------------------------------------------------- #
# Composition: affine, AvgPool2d, affine through the model entry points        #
# --------------------------------------------------------------------------- #
_AP_COMP_CHW = (2, 3, 5)
_AP_COMP_CONV = np.array([[[[1.0, -0.5]], [[0.25, 0.5]]],
                          [[[-0.5, 0.25]], [[1.0, 0.75]]]])
_AP_COMP_CONV_BIAS = np.array([0.25, -0.5])
# kernel 2x1, stride 1x2, padding on the height only, border windows divided
# by the entries they hold (ONNX's default count_include_pad=0)
_AP_COMP_POOL = dict(kernel_size=(2, 1), stride=(1, 2), padding=(1, 0),
                     count_include_pad=False)


def _ap_comp_head(n_in, seed=5):
    rng = np.random.default_rng(seed)
    return (rng.integers(-4, 5, size=(3, n_in)) / 4.0,
            rng.integers(-4, 5, size=3) / 8.0)


def _ap_comp_torch():
    """Conv (1x2, mixing the channels) > AvgPool2d > Flatten > Linear in
    float64, dyadic weights; with the Linear weight and bias."""
    conv = nn.Conv2d(2, 2, (1, 2)).double()
    pool = nn.AvgPool2d(**_AP_COMP_POOL)
    pooled = _ap_out_chw(pool, (2, 3, 4))
    G, h = _ap_comp_head(int(np.prod(pooled)))
    head = nn.Linear(G.shape[1], 3).double()
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(_AP_COMP_CONV))
        conv.bias.copy_(torch.from_numpy(_AP_COMP_CONV_BIAS))
        head.weight.copy_(torch.from_numpy(G))
        head.bias.copy_(torch.from_numpy(h))
    return nn.Sequential(conv, pool, nn.Flatten(), head).eval(), G, h


def _ap_comp_onnx(tmp_path):
    """The same chain as an ONNX file: Conv > AveragePool (pads on the height
    only, count_include_pad=0) > Flatten > Gemm."""
    C, H, W = _AP_COMP_CHW
    _, G, h = _ap_comp_torch()
    nodes = [helper.make_node("Conv", ["X", "w", "b"], ["c"],
                              kernel_shape=[1, 2]),
             helper.make_node("AveragePool", ["c"], ["p"],
                              kernel_shape=[2, 1], strides=[1, 2],
                              pads=[1, 0, 1, 0], count_include_pad=0),
             helper.make_node("Flatten", ["p"], ["f"], axis=1),
             helper.make_node("Gemm", ["f", "g", "h"], ["Y"], transB=1)]
    inits = {"w": _AP_COMP_CONV.astype(np.float32),
             "b": _AP_COMP_CONV_BIAS.astype(np.float32),
             "g": G.astype(np.float32), "h": h.astype(np.float32)}
    return save_onnx(tmp_path, nodes, [1, C, H, W], [1, 3], inits,
                     name="conv_avgpool_gemm")


@pytest.mark.parametrize("kind,method", [("ImageStar", "approx"),
                                         ("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
@pytest.mark.parametrize("variant", ["box"])
def test_avgpool_in_an_onnx_chain(tmp_path, kind, method, variant):
    """Conv > AveragePool > Flatten > Gemm loaded from ONNX (the pool becomes
    an AvgPool2d with list-valued kernel and stride): every onnxruntime
    output of a sample point is in the result, by LP on its rows. Catches
    the pool's rows or centre read wrongly by the next layers, the layout
    change at Flatten, and ONNX's count_include_pad=0 divided as torch's
    default. (The model entry takes one input set per call, so several sets
    reach the layer at once only after a split inside the model:
    ``test_avgpool_after_an_exact_relu_split_in_a_model``.)"""
    path = _ap_comp_onnx(tmp_path)
    lo, hi = pixel_box(*_AP_COMP_CHW, seed=2)
    sets = image_set(kind, lo, hi)
    what = f"Conv > AveragePool > Flatten > Gemm, {method}, {kind} {variant}"
    out = reach_model(load_onnx(path), sets, method, what)
    assert_covered(out, replay_onnx(path, samples(lo, hi),
                                    [1] + list(_AP_COMP_CHW)), what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_avgpool_in_a_torch_chain_is_the_exact_image(kind):
    """Conv > AvgPool2d > Flatten > Linear (torch, float64, dyadic weights)
    through ``reach_pytorch_model`` on a two-variable set: the result holds
    the exact image (rationals, the chain's affine map read from torch on the
    unit vectors) with zero slack and has no vertex outside it. Catches the
    rows the pool emits consumed wrongly downstream and the layout change at
    Flatten."""
    model, _, _ = _ap_comp_torch()
    chw = _AP_COMP_CHW
    n = int(np.prod(chw))
    with torch.no_grad():
        offset = model(torch.zeros((1,) + chw, dtype=torch.float64))
        cols = model(torch.eye(n, dtype=torch.float64).reshape((n,) + chw))
    M = (cols - offset).numpy().T
    centre, gen = _ap_pixels(chw, seed=4)
    A, b = ((_AP_SQUARE_A, _AP_SQUARE_B) if kind == "ImageZono"
            else (_AP_POLY_A, _AP_POLY_B))
    pieces = rational.affine(rational.polytope(A, b), gen.tolist(),
                             centre.tolist())
    pieces = rational.affine(pieces, M.tolist(), offset.numpy()[0].tolist())
    what = f"Conv > AvgPool2d > Flatten > Linear on a two-variable {kind}"
    out = reach_model(model, _ap_low_set(chw, centre, gen, kind), "approx",
                      what)
    _ap_check_exact(out, pieces, what)


# --------------------------------------------------------------------------- #
# Refusals                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("entry", ["layer", "onnx"])
def test_avgpool_refuses_an_overhanging_ceil_window_with_padding_counted(
        tmp_path, entry, kind):
    """ceil_mode=True with count_include_pad=True and a last window that
    overhangs the input by exactly one entry: torch divides that window by
    its part inside the input, onnxruntime by the kernel area, so n2v cannot
    know which function is meant and must raise. Catches an off-by-one in
    the overhang test that answers with torch's divisor."""
    chw = (2, 3, 5)
    lo, hi = pixel_box(*chw)
    if entry == "layer":
        layer = nn.AvgPool2d((1, 2), stride=(1, 2), ceil_mode=True,
                             count_include_pad=True)
        with pytest.raises(NotImplementedError):
            reach_layer(layer, [image_set(kind, lo, hi)], "approx")
        return
    path = save_onnx(tmp_path, [helper.make_node(
        "AveragePool", ["X"], ["Y"], kernel_shape=[1, 2], strides=[1, 2],
        ceil_mode=1, count_include_pad=1)], [1, 2, 3, 5], [1, 2, 3, 3],
        name="avgpool_ceil_overhang")
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                    f"overhanging ceil AveragePool on {kind}")


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_avgpool_refuses_an_overhang_with_a_kernel_smaller_than_its_stride(
        tmp_path, kind):
    """The ceil_mode, count_include_pad refusal on the other geometry: an
    ONNX AveragePool with kernel 2 and stride 3 along the height of a
    (2, 7, 3) image, whose third window starts on the last row and overhangs
    by one entry. torch divides that window by 1, onnxruntime by 2, so the
    reach must raise. Catches an overhang test that confuses the kernel with
    the stride (``(o - 1) * k + s``), which agrees with the right one when
    the two are equal, and an overhang looked for on the width only."""
    path = save_onnx(tmp_path, [helper.make_node(
        "AveragePool", ["X"], ["Y"], kernel_shape=[2, 1], strides=[3, 1],
        ceil_mode=1, count_include_pad=1)], [1, 2, 7, 3], [1, 2, 3, 3],
        name="avgpool_ceil_overhang_height")
    lo, hi = pixel_box(2, 7, 3)
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                    f"overhanging ceil AveragePool (kernel < stride) on {kind}")


def test_avgpool_with_one_sided_onnx_pads_is_refused_at_load(tmp_path):
    """An ONNX AveragePool padded on one side only has no AvgPool2d
    equivalent; loading must refuse it, not pad both sides."""
    path = save_onnx(tmp_path, [helper.make_node(
        "AveragePool", ["X"], ["Y"], kernel_shape=[2, 2], strides=[1, 2],
        pads=[1, 0, 0, 0])], [1, 2, 3, 4], [1, 2, 3, 2], name="avgpool_asym")
    with pytest.raises(NotImplementedError):
        load_onnx(path)


@pytest.mark.parametrize("method", ["approx", "exact"])
@pytest.mark.parametrize("kind", ["Box", "Hexatope", "Octatope"])
def test_avgpool_on_a_flat_set_type_without_a_rule_is_refused(kind, method):
    """The dispatcher has no AvgPool2d rule for Box, Hexatope or Octatope; it
    must raise, never return the input or a placeholder."""
    lo, hi = pixel_box(2, 2, 3)
    with pytest.raises(NotImplementedError):
        reach_layer(nn.AvgPool2d(2), [flat_set(kind, lo, hi)], method)


_AP_FLAT_STAR = known_bug(
    23, "AvgPool2d on a flat Star raises TypeError ('AvgPool2D expects "
        "ImageStar input') instead of the declared refusal "
        "(NotImplementedError, later UnsupportedError)", raises=TypeError)
_AP_FLAT_ZONO = known_bug(
    23, "AvgPool2d on a flat Zono crashes with AttributeError ('Zono' object "
        "has no attribute 'height') instead of refusing",
    raises=AttributeError)


@pytest.mark.parametrize("entry", ["layer"])
@pytest.mark.parametrize("kind", [
    pytest.param("Star", marks=_AP_FLAT_STAR),
    pytest.param("Zono", marks=_AP_FLAT_ZONO)])
def test_avgpool_on_a_flat_star_or_zono_is_refused(kind, entry):
    """A flat Star or Zono carries no image shape, so AvgPool2d cannot place
    its windows: the call must refuse with NotImplementedError. One marked
    case per kernel: the model entry (``reach_pytorch_model`` with an input
    shape) and the exact method reach the same raise in ``avgpool2d_star``
    and ``avgpool2d_zono``. The Box, Hexatope and Octatope neighbours of
    ``test_avgpool_on_a_flat_set_type_without_a_rule_is_refused`` pass with
    the same inputs; a flat set reshaped to an image inside the model is
    answered (``test_avgpool_after_an_onnx_reshape_of_a_flat_set``)."""
    chw = (2, 2, 3)
    lo, hi = pixel_box(*chw)
    s = flat_set(kind, lo, hi)
    with pytest.raises(NotImplementedError):
        reach_layer(nn.AvgPool2d(2), [s], "approx")


# --------------------------------------------------------------------------- #
# The Box, Hexatope and Octatope kernels (reachable by a direct call only)     #
# --------------------------------------------------------------------------- #
_AP_HEXOCT_INPUT_BOX = known_bug(
    9, "HEXOCT-6: avgpool2d_hexatope and avgpool2d_octatope return the "
       "input's bounding box whatever the layer: outputs of a size-preserving "
       "pool escape, and a shrinking pool gets a set of the input's dimension")
_AP_BOX_PLACEHOLDER = known_bug(
    9, "HEXOCT-7: avgpool2d_box is a placeholder that returns the input box "
       "unchanged: outputs of a size-preserving pool escape, and a shrinking "
       "pool gets a box of the input's dimension")

_AP_KERNELS = {"Hexatope": avgpool2d_hexatope, "Octatope": avgpool2d_octatope,
               "Box": avgpool2d_box}
_AP_KERNEL_POOLS = {"k1": dict(kernel_size=1),
                    "k3-s1-p1": dict(kernel_size=3, stride=1, padding=1)}


def _ap_kernel_cells():
    """Per kernel: the identity pool, on which the input's bounding box is
    still right (one set, and three sets), and one marked size-preserving
    pool, on which it is wrong. A shrinking pool fails by the same code (the
    layer is never read), so it carries no marker of its own."""
    cells = []
    for kind in ("Hexatope", "Octatope", "Box"):
        mark = _AP_BOX_PLACEHOLDER if kind == "Box" else _AP_HEXOCT_INPUT_BOX
        cells += [pytest.param(kind, "k1", 1, id=f"{kind}-k1-one-set"),
                  pytest.param(kind, "k1", 3, id=f"{kind}-k1-three-sets"),
                  pytest.param(kind, "k3-s1-p1", 1, marks=mark,
                               id=f"{kind}-k3-s1-p1")]
    return cells


@pytest.mark.parametrize("kind,pool,parts", _ap_kernel_cells())
def test_avgpool_box_hexatope_octatope_kernel_is_sound_or_refuses(kind, pool,
                                                                  parts):
    """The Box, Hexatope and Octatope AvgPool kernels, called directly on the
    flattened (C, H, W) image (dyadic bounds, which ``from_bounds`` keeps
    exactly), one set or three sets covering the box: the result holds every
    torch output, or the call refuses (NotImplementedError). A 1x1 window is
    the identity, so the result must hold every member of every input set.
    Catches a bound taken from the wrong end and only one of several sets
    answered."""
    chw = (2, 3, 4)
    layer = nn.AvgPool2d(**_AP_KERNEL_POOLS[pool])
    lo, hi = pixel_box(*chw, seed=1)
    sets = (split_cover(kind, lo, hi, parts) if parts > 1
            else [flat_set(kind, lo, hi)])
    try:
        out = _AP_KERNELS[kind](layer, sets)
    except NotImplementedError:
        return
    what = f"{kind} AvgPool2d {pool}, {parts} set(s)"
    assert_covered(out, replay_torch(layer, samples(lo, hi), (1,) + chw), what)


# --------------------------------------------------------------------------- #
# After an exact split inside the model: several sets reach the layer          #
# --------------------------------------------------------------------------- #
def _ap_after_split_box(chw=(2, 2, 4), seed=12):
    """A pixel box with every pixel positive except three that straddle 0,
    so an exact ReLU splits it into at most eight stars."""
    lo, hi = pixel_box(*chw, seed=seed)
    lo, hi = lo + 3.0, hi + 3.0
    flat_lo, flat_hi = lo.reshape(-1), hi.reshape(-1)
    for p, (a, b) in zip((1, 6, 12), ((-0.5, 0.75), (-0.25, 0.5),
                                      (-1.0, 0.25))):
        flat_lo[p], flat_hi[p] = a, b
    return flat_lo.reshape(chw), flat_hi.reshape(chw)


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageStar", "approx"),
                                         ("ImageZono", "approx")])
def test_avgpool_after_an_exact_relu_split_in_a_model(kind, method):
    """ReLU > AvgPool2d((2, 1), stride (1, 2), padding (1, 0)) > Flatten > Linear through ``reach_pytorch_model``
    on a (2, 2, 4) box with three pixels straddling zero: by the exact method
    the ReLU hands up to eight stars to the layer in one call. Every torch
    output of a sample point is in a returned set, by LP on the rows.
    Catches only the first or the last of several sets answered inside the
    model, and a result of one set reused for another."""
    chw = (2, 2, 4)
    layer = nn.AvgPool2d((2, 1), stride=(1, 2), padding=(1, 0))
    with torch.no_grad():
        n_mid = int(np.prod(layer(torch.zeros((1,) + chw)).shape[1:]))
    head = nn.Linear(n_mid, 3).double()
    rng = np.random.default_rng(13)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(rng.integers(-4, 5, (3, n_mid))
                                           / 4.0))
        head.bias.copy_(torch.from_numpy(rng.integers(-4, 5, 3) / 8.0))
    model = nn.Sequential(nn.ReLU(), layer.double(), nn.Flatten(),
                          head).eval()
    lo, hi = _ap_after_split_box(chw)
    what = f"ReLU > AvgPool2d((2, 1), stride (1, 2), padding (1, 0)) > Flatten > Linear, {method}, {kind}"
    out = reach_model(model, image_set(kind, lo, hi), method, what)
    if method == "exact" and len(out) < 2:
        pytest.fail(f"test setup: the exact ReLU gave {len(out)} set(s)")
    assert_covered(out, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what)


# --------------------------------------------------------------------------- #
# ONNX AveragePool attributes, one node, checked against onnxruntime           #
# --------------------------------------------------------------------------- #
_AP_ONNX_DILATIONS = known_bug(
    23, "an ONNX AveragePool of opset 19 or later with dilations is loaded as "
        "an AvgPool2d without dilation: onnx2torch has no converter for the "
        "opset, load_onnx downgrades the model to opset 13, the dilations "
        "attribute stays on the node and the version-11 converter ignores "
        "it, so reach answers for another pooling and every onnxruntime "
        "output escapes when the two output shapes agree (registry row "
        "LOAD-7)")

# name: (opset, AveragePool attributes, input (C, H, W), output (C, H, W)).
_AP_ONNX_REGIMES = {
    # the last window overhangs the input; ONNX's default
    # count_include_pad=0 divides it by its one in-image entry, as torch does
    "ceil-nocip": (13, dict(kernel_shape=[1, 2], strides=[1, 2], ceil_mode=1,
                            count_include_pad=0), (2, 3, 5), (2, 3, 3)),
    # border windows counted with their padding (divided by 6 everywhere)
    "pads-cip": (13, dict(kernel_shape=[3, 2], strides=[1, 2],
                          pads=[1, 0, 1, 0], count_include_pad=1),
                 (2, 3, 5), (2, 3, 2)),
    # no converter for opset 19: load_onnx downgrades the model first
    "opset19": (19, dict(kernel_shape=[2, 2], strides=[2, 2]), (2, 4, 5),
                (2, 2, 2)),
    "opset19-dilations": (19, dict(kernel_shape=[2, 2], strides=[2, 2],
                                   dilations=[1, 2]), (2, 4, 5), (2, 2, 2)),
}


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=_AP_ONNX_DILATIONS) if n == "opset19-dilations"
    else n for n in _AP_ONNX_REGIMES])
def test_avgpool_onnx_attributes_hold_every_onnxruntime_output(tmp_path,
                                                               name):
    """One ONNX AveragePool loaded with ``load_onnx`` and reached on an
    ImageStar with a binding row: every onnxruntime output of a member point
    is in the result by LP on its rows, or the load or the reach refuses
    (NotImplementedError). Catches an attribute of the node lost or misread
    on the way to ``nn.AvgPool2d``: ``ceil_mode`` (the overhanging window
    dropped), ``count_include_pad=1`` read as ONNX's default 0, and an
    opset the converter does not know turned into another pooling (the
    dilated node, whose output has the same shape as the undilated one; its
    undilated neighbour takes the same downgrade and passes)."""
    opset, attrs, chw, out_chw = _AP_ONNX_REGIMES[name]
    path = save_onnx(tmp_path, [helper.make_node("AveragePool", ["X"], ["Y"],
                                                 **attrs)],
                     [1] + list(chw), [1] + list(out_chw),
                     name=f"avgpool_{name.replace('-', '_')}", opset=opset)
    lo, hi = pixel_box(*chw, seed=4)
    star, points = constrained_image_star(lo, hi, seed=4)
    outputs = replay_onnx(path, points, [1] + list(chw))
    what = f"ONNX AveragePool {name} on an ImageStar with a binding row"
    try:
        out = reach_model(load_onnx(path), star, "approx", what)
    except NotImplementedError:
        return
    assert_covered(out, outputs, what, out_chw=out_chw)


# --------------------------------------------------------------------------- #
# A flat set reshaped to an image in the ONNX graph, then pooled              #
# --------------------------------------------------------------------------- #
_AP_RESHAPE_CHW = (2, 3, 5)


def _ap_reshape_onnx(tmp_path):
    """X (1, 30) > Reshape (1, 2, 3, 5) > AveragePool (kernel 2x1, strides
    1x2, pads on the height, count_include_pad=0) > Flatten > Gemm, dyadic
    weights."""
    C, H, W = _AP_RESHAPE_CHW
    pool = nn.AvgPool2d((2, 1), stride=(1, 2), padding=(1, 0),
                        count_include_pad=False)
    n_mid = int(np.prod(_ap_out_chw(pool, _AP_RESHAPE_CHW)))
    rng = np.random.default_rng(7)
    G = rng.integers(-4, 5, size=(3, n_mid)) / 4.0
    h = rng.integers(-4, 5, size=3) / 8.0
    nodes = [helper.make_node("Reshape", ["X", "shape"], ["r"]),
             helper.make_node("AveragePool", ["r"], ["p"],
                              kernel_shape=[2, 1], strides=[1, 2],
                              pads=[1, 0, 1, 0]),
             helper.make_node("Flatten", ["p"], ["f"], axis=1),
             helper.make_node("Gemm", ["f", "g", "h"], ["Y"], transB=1)]
    inits = {"shape": np.array([1, C, H, W], dtype=np.int64),
             "g": G.astype(np.float32), "h": h.astype(np.float32)}
    return save_onnx(tmp_path, nodes, [1, C * H * W], [1, 3], inits,
                     name="reshape_avgpool_gemm")


@pytest.mark.parametrize("kind,method", [("Star", "exact"), ("Star", "approx"),
                                         ("Zono", "approx")])
def test_avgpool_after_an_onnx_reshape_of_a_flat_set(tmp_path, kind, method):
    """A flat Star (with a binding row) or Zono over a (1, 30) input, which
    the graph reshapes to (1, 2, 3, 5) before an AveragePool: the walker must
    see that the reshape feeds a pool (``AvgPool2d`` in
    ``_SPATIAL_CONSUMERS``), build an ImageStar or ImageZono in the stored
    (H, W, C) order and pool it; every onnxruntime output of a member point
    is in the result by LP on its rows. Catches the reshape kept flat in
    front of the pool (which then crashes) and the flat (C, H, W) order taken
    as the stored one."""
    path = _ap_reshape_onnx(tmp_path)
    lo, hi = pixel_box(*_AP_RESHAPE_CHW, seed=6)
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    if kind == "Star":
        s, points = constrained_star(lo, hi, seed=6)
    else:
        s, points = flat_set(kind, lo, hi), samples(lo, hi)
    what = f"Reshape > AveragePool > Flatten > Gemm, {method}, flat {kind}"
    out = reach_model(load_onnx(path), s, method, what)
    assert_covered(out, replay_onnx(path, points, [1, lo.size]), what)
