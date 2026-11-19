"""Soundness cells of global average pooling: ``nn.AdaptiveAvgPool2d(1)``,
the ONNX ``GlobalAveragePool`` (``OnnxGlobalAveragePoolWithKnownInputShape``,
the class ``load_onnx`` builds) and the shape-free ``OnnxGlobalAveragePool``,
on every set type and both methods.

The ImageStar and ImageZono rows answer with the exact image (the mean is
linear); the oracles are the exact image in rationals and onnxruntime or torch
float64 replay with membership by LP on the returned rows. The flat Star and
Zono rows crash today (marked), the Box, Hexatope and Octatope rows refuse,
and the model entry refuses ``exact`` on every set type without an exact
method.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import ImageStar, ImageZono


# =========================================================================== #
# Soundness cells: global average pooling on every set type                   #
# =========================================================================== #
# A cell is (set type, layer spelling, option). nn.AdaptiveAvgPool2d(1),
# the ONNX GlobalAveragePool and the shape-free OnnxGlobalAveragePool all
# reach the same kernels (nn.AdaptiveAvgPool2d((1, 1)) is the cell
# 4x6-to-1x1 of tests/contracts/test_reach_coverage.py). The operation is
# linear. Two oracles, neither from n2v:
#
# - the exact image in rationals (tests/_oracles/rational.py) of a set of two
#   predicate variables on images of 8 pixels per channel (the mean is then
#   exact in float64), containment with zero slack and, as the kernel claims
#   exactness, no vertex of the result outside the image;
# - torch in float64 or onnxruntime on sample points, each a member of some
#   returned set by one LP on the set's own rows (assert_covered).
#
# Layout: inputs are built in (C, H, W) order and converted to the stored
# (H, W, C) order by the builders; the output is a 1x1 image whose stored
# order is the channel order. Every image has several channels and height
# unlike width.
from fractions import Fraction

import pytest
from onnx import helper
from onnx2torch.node_converters.global_average_pool import \
    OnnxGlobalAveragePool

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_image_star,
                                flat_set, hwc_index, image_set,
                                image_split_cover, pixel_box, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx)
from tests.known_bugs import known_bug

_GP_SHAPES = [(2, 3, 5)]


def _gp_layer(spelling, chw, tmp_path):
    """``(layer, path)``: the global pool as ``spelling`` for a (C, H, W)
    input; ``path`` is the ONNX file for the ONNX spelling, else None. The
    ONNX spelling loads as ``OnnxGlobalAveragePoolWithKnownInputShape``
    (the converter always finds the input shape); ``onnx-shape-free`` is the
    converter's other class, which only a direct ``reach_layer`` call
    reaches."""
    if spelling == "adaptive-1":
        return nn.AdaptiveAvgPool2d(1), None
    if spelling == "onnx-shape-free":
        return OnnxGlobalAveragePool(), None
    C = chw[0]
    path = save_onnx(tmp_path, [helper.make_node(
        "GlobalAveragePool", ["X"], ["Y"])], [1] + list(chw), [1, C, 1, 1],
        name="gap")
    layers = [m for m in load_onnx(path).modules()
              if type(m).__name__.startswith("OnnxGlobalAveragePool")]
    if len(layers) != 1:
        pytest.fail(f"test setup: the ONNX GlobalAveragePool loaded as "
                    f"{[type(m).__name__ for m in layers]}")
    return layers[0], path


def _gp_inputs(kind, variant, chw, seed):
    """``(input sets, sample points)``, points in the network's order. The
    shifted box lies away from zero, where a wrong divisor scales the result
    away from the true outputs instead of around them. The wide box is the
    pixel box times 8 (bounds up to 24 away from zero, widths 2 to 10) with
    one pixel of zero width, which gets no predicate variable in an
    ImageStar and a zero generator in an ImageZono."""
    lo, hi = pixel_box(*chw, seed=seed)
    if variant == "shifted":
        lo, hi = lo + 4.0, hi + 4.0
    if variant == "wide":
        lo, hi = 8.0 * lo, 8.0 * hi
        hi[0, 1, 2] = lo[0, 1, 2]
    if variant in ("box", "shifted", "wide"):
        return [image_set(kind, lo, hi)], samples(lo, hi)
    if variant == "split":
        return image_split_cover(kind, lo, hi, 3), samples(lo, hi)
    if variant == "rows":
        star, points = constrained_image_star(lo + 2.0, hi + 2.0, seed=seed)
        return [star], points
    raise ValueError(variant)


def _gp_cells():
    """Every input variant through the torch and the ONNX spelling, and the
    pixel box through the shape-free class. One image shape: the kernel has
    no branch on the shape (the exact-image cells below hold (3, 4, 2), with
    height above width and three channels)."""
    cells = []
    for chw in _GP_SHAPES:
        for spelling in ("adaptive-1", "onnx", "onnx-shape-free"):
            for kind, variants in (
                    ("ImageStar", ("box", "shifted", "split", "rows",
                                   "wide")),
                    ("ImageZono", ("box", "shifted", "split", "wide"))):
                for variant in variants:
                    if spelling == "onnx-shape-free" and variant != "box":
                        continue
                    cells.append(pytest.param(
                        chw, spelling, kind, variant,
                        id=f"{'x'.join(map(str, chw))}-{spelling}-{kind}-"
                           f"{variant}"))
    return cells


@pytest.mark.parametrize("chw,spelling,kind,variant", _gp_cells())
def test_global_avgpool_holds_every_true_output(tmp_path, chw, spelling, kind,
                                                variant):
    """Global average pooling through ``reach_layer`` on a (2, 3, 5)
    ImageStar or ImageZono: a pixel box, a box away from zero, three sets
    covering a box, a star with a binding row, or a wide box with a pixel of
    zero width. Every true output (torch in float64, or onnxruntime for the
    ONNX spelling) of a sample point is in a returned set, by LP on the
    set's rows. Catches a divisor of height squared, of the height alone or
    applied twice, entries averaged over the wrong axes (channel taken for
    height or width, centre or generators in another layout), the centre
    averaged differently from the generators, an input set changed in place,
    only one of several input sets answered, and a layer class of the ONNX
    converter that is not routed to the kernel (a refusal where the kernel
    answers)."""
    layer, path = _gp_layer(spelling, chw, tmp_path)
    what = f"global average pool ({spelling}) on {chw} {kind} {variant}"
    sets, points = _gp_inputs(kind, variant, chw, seed=sum(chw))
    out = reach_one_layer(layer, sets, "approx", what)
    outputs = (replay_onnx(path, points, [1] + list(chw)) if path
               else replay_torch(layer, points, (1,) + chw))
    assert_covered(out, outputs, what, out_chw=(chw[0], 1, 1))


# --------------------------------------------------------------------------- #
# The exact image in rationals, on a set of two predicate variables            #
# --------------------------------------------------------------------------- #
_GP_POLY_A = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, Fraction(1, 2)]]
_GP_POLY_B = [1, 1, 1, 1, 1]
_GP_SQUARE_A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
_GP_SQUARE_B = [1, 1, 1, 1]


def _gp_pixels(chw, seed=3, shift=0.0):
    """``(centre, gen)`` in (C, H, W) order, dyadic: centres in
    [-2, 2] + shift by 1/4, generator entries in [-1/2, 1/2] by 1/8."""
    n = int(np.prod(chw))
    rng = np.random.default_rng(seed)
    return (rng.integers(-8, 9, size=n) / 4.0 + shift,
            rng.integers(-4, 5, size=(n, 2)) / 8.0)


def _gp_low_set(chw, centre, gen, kind, extra=None):
    """``centre + gen . a`` as an ImageStar over the polygon (cut by the row
    ``extra = (row, rhs)``), or as an ImageZono over the square, stored in
    (H, W, C) order."""
    C, H, W = chw
    order = hwc_index(C, H, W)
    c, G = centre[order], gen[order]
    if kind == "ImageZono":
        return ImageZono(c.reshape(-1, 1), G.copy(), H, W, C)
    rows = _GP_POLY_A + ([extra[0]] if extra else [])
    rhs = _GP_POLY_B + ([extra[1]] if extra else [])
    A = np.array([[float(v) for v in r] for r in rows])
    b = np.array([float(v) for v in rhs]).reshape(-1, 1)
    return ImageStar(np.hstack([c[:, None], G]), A, b, -np.ones((2, 1)),
                     np.ones((2, 1)), H, W, C)


def _gp_mean_matrix(chw):
    """The global mean as a rational matrix from the flat (C, H, W) input to
    the C outputs: entry 1 / (H W) on the pixels of each channel."""
    C, H, W = chw
    n = H * W
    return [[Fraction(1, n) if p // n == c else Fraction(0)
             for p in range(C * n)] for c in range(C)]


def _gp_check_exact(out, pieces, what):
    computed = [rational.star(s) for s in out]
    bad = rational.escapes(pieces, *computed)
    assert not bad, (f"{what}: {len(bad)} vertices of the exact image are in "
                     f"no returned set (first {[float(v) for v in bad[0][1]]})")
    spread = rational.straddlers(pieces, *computed)
    assert not spread, f"{what}: exact pieces {spread} straddle the result"
    for k, s in enumerate(computed):
        far = rational.unreached(pieces, rational.set_vertices(s))
        assert not far, (f"{what}: returned set {k} has a vertex that is no "
                         f"output of the pool: "
                         f"{[float(v) for v in far[0][1]]}")


@pytest.mark.parametrize("split", [False, True], ids=["one-set", "two-halves"])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("chw", [(2, 2, 4), (3, 4, 2)],
                         ids=["2x2x4", "3x4x2"])
def test_global_avgpool_is_the_exact_image(chw, kind, split):
    """``nn.AdaptiveAvgPool2d(1)`` on a set of two predicate variables (an
    ImageStar over a polygon with a binding row, an ImageZono over the
    square), alone or as the halves a1 <= 0 and a1 >= 0, centres away from
    zero: the result holds every vertex of the exact image with zero slack
    and has no vertex outside it. Catches a wrong divisor (height squared,
    height alone, twice), the mean taken over the wrong axes, the centre
    averaged differently from the generators, and a second set not
    answered."""
    centre, gen = _gp_pixels(chw, shift=1.5)
    square = kind == "ImageZono"
    A, b = (_GP_SQUARE_A, _GP_SQUARE_B) if square else (_GP_POLY_A, _GP_POLY_B)
    pieces = rational.affine(rational.polytope(A, b), gen.tolist(),
                             centre.tolist())
    pieces = rational.affine(pieces, _gp_mean_matrix(chw))
    if split and square:
        sets = []
        for sign in (-1.0, 1.0):
            g = gen.copy()
            g[:, 0] *= 0.5
            sets.append(_gp_low_set(chw, centre + sign * g[:, 0], g, kind))
    elif split:
        sets = [_gp_low_set(chw, centre, gen, kind, ([1, 0], 0)),
                _gp_low_set(chw, centre, gen, kind, ([-1, 0], 0))]
    else:
        sets = [_gp_low_set(chw, centre, gen, kind)]
    if split:
        pieces = rational.split(pieces, [1, 0], 0)
    what = f"global average pool on {chw} {kind}, " \
           f"{'two halves' if split else 'one set'}"
    out = reach_one_layer(nn.AdaptiveAvgPool2d(1), sets, "exact", what)
    _gp_check_exact(out, pieces, what)


# The predicate polytope of the next cells: a1 in [-3, 5], a2 in [-1/2, 2] and
# the binding row a1 + a2 <= 5, which cuts the corner (5, 2) off.
_GP_WIDE_A = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]]
_GP_WIDE_B = [5, 3, 2, Fraction(1, 2), 5]


def _gp_wide_star(chw, centre, gen, box):
    """``centre + gen . a`` (centre and gen in (C, H, W) order) as an
    ImageStar over that polytope, stored (H, W, C), the polytope given as
    ``own`` (the predicate box [-3, 5] x [-1/2, 2] and the cut row),
    ``one-infinite-end`` (the lower end of a1 at -inf and the row -a1 <= 3
    in its place) or ``none`` (no predicate box, every bound a row)."""
    C, H, W = chw
    order = hwc_index(C, H, W)
    V = np.hstack([centre[order][:, None], gen[order]])
    if box == "own":
        rows, rhs, lb, ub = [[1, 1]], [5], [-3, -0.5], [5, 2]
    elif box == "one-infinite-end":
        rows, rhs, lb, ub = [[1, 1], [-1, 0]], [5, 3], [-np.inf, -0.5], [5, 2]
    elif box == "none":
        rows, rhs, lb, ub = _GP_WIDE_A, _GP_WIDE_B, None, None
    else:
        raise ValueError(box)
    pred = ((None, None) if lb is None else
            (np.array(lb, float).reshape(-1, 1),
             np.array(ub, float).reshape(-1, 1)))
    return ImageStar(V, np.array([[float(v) for v in r] for r in rows]),
                     np.array([float(v) for v in rhs]).reshape(-1, 1), *pred,
                     H, W, C)


@pytest.mark.parametrize("box", ["own", "one-infinite-end", "none"])
def test_global_avgpool_keeps_the_predicate_box_of_its_input(box):
    """``AdaptiveAvgPool2d(1) > Flatten`` through ``reach_pytorch_model``
    under ``exact`` on a (2, 2, 4) ImageStar of two predicate variables whose
    predicate box is not the unit box: its own box [-3, 5] x [-1/2, 2] with
    a binding row, the same with one infinite end held by a row, and no
    predicate box at all (an ImageStar the model entry leaves as given).
    Pixels reach 4.875 away from zero with widths from 1 to 4.625. The
    result holds every vertex of the exact image (rationals, zero slack) and
    has no vertex outside it. Catches a result given the unit predicate box
    (or any box other than its input's) and rows handed on wrongly (negated;
    a dropped row shows as a vertex outside the image)."""
    chw = (2, 2, 4)
    centre, gen = _gp_pixels(chw, seed=5, shift=1.5)
    pieces = rational.affine(rational.polytope(_GP_WIDE_A, _GP_WIDE_B),
                             gen.tolist(), centre.tolist())
    pieces = rational.affine(pieces, _gp_mean_matrix(chw))
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    what = (f"AdaptiveAvgPool2d(1) > Flatten, exact, on an ImageStar with "
            f"predicate box {box}")
    out = reach_model(model, _gp_wide_star(chw, centre, gen, box), "exact",
                      what)
    _gp_check_exact(out, pieces, what)


_GP_FLOAT32 = known_bug(
    7, "ROUND-12: an ImageZono whose arrays are float32 is pooled by numpy in "
       "float32, so the mean of its centre and generators is rounded to "
       "float32 and exact outputs of the stored set lie outside the result "
       "by up to about 1e-7; the same numbers held as float64 are pooled "
       "exactly")


@pytest.mark.parametrize("dtype", [
    "float64", pytest.param("float32", marks=_GP_FLOAT32)])
def test_global_avgpool_holds_the_exact_image_of_the_stored_image_zono(dtype):
    """An ImageZono of a (2, 2, 4) image whose centre entries have a full
    float32 mantissa (1 + k / 2^23) and whose two generators are powers of
    two, held as float64 and as float32 (an ImageZono keeps the float type
    it is given; the numbers are the same). The mean of eight such numbers
    is exact in float64, so every vertex of the exact image of the stored
    set (rationals) must be in the result with zero slack. Catches a pooled
    zonotope rounded to its input's float type, and a mean that is not the
    mean of the stored numbers."""
    chw = (2, 2, 4)
    C, H, W = chw
    rng = np.random.default_rng(0)
    centre = 1.0 + rng.integers(1, 2 ** 23, size=C * H * W) / 2.0 ** 23
    gen = np.stack([2.0 ** -rng.integers(1, 6, size=C * H * W),
                    -(2.0 ** -rng.integers(1, 6, size=C * H * W))], axis=1)
    order = hwc_index(C, H, W)
    zono = ImageZono(centre[order].reshape(-1, 1).astype(dtype),
                     gen[order].astype(dtype), H, W, C)
    if zono.c.dtype != np.dtype(dtype) or not np.array_equal(
            zono.V.astype(np.float64), gen[order]):
        pytest.fail("test setup: the ImageZono does not hold the numbers "
                    f"as {dtype}")
    pieces = rational.affine(rational.polytope(_GP_SQUARE_A, _GP_SQUARE_B),
                             gen.tolist(), centre.tolist())
    pieces = rational.affine(pieces, _gp_mean_matrix(chw))
    what = f"global average pool on a {dtype} ImageZono"
    out = reach_one_layer(nn.AdaptiveAvgPool2d(1), [zono], "approx", what)
    bad = rational.escapes(pieces, *[rational.star(s) for s in out])
    assert not bad, (f"{what}: {len(bad)} of 4 vertices of the exact image "
                     f"are outside the result (first "
                     f"{[float(v) for v in bad[0][1]]})")


# --------------------------------------------------------------------------- #
# Composition: affine, global pool, affine through the model entry points      #
# --------------------------------------------------------------------------- #
_GP_COMP_CHW = (2, 2, 5)          # a 1x2 convolution makes it (2, 2, 4)
_GP_COMP_CONV = np.array([[[[1.0, -0.5]], [[0.25, 0.5]]],
                          [[[-0.5, 0.25]], [[1.0, 0.75]]]])
_GP_COMP_CONV_BIAS = np.array([0.25, -0.5])
_GP_COMP_HEAD = np.array([[1.0, -0.5], [0.25, 0.75], [-1.0, 0.5]])
_GP_COMP_HEAD_BIAS = np.array([0.125, -0.25, 0.5])


def _gp_comp_torch():
    conv = nn.Conv2d(2, 2, (1, 2)).double()
    head = nn.Linear(2, 3).double()
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(_GP_COMP_CONV))
        conv.bias.copy_(torch.from_numpy(_GP_COMP_CONV_BIAS))
        head.weight.copy_(torch.from_numpy(_GP_COMP_HEAD))
        head.bias.copy_(torch.from_numpy(_GP_COMP_HEAD_BIAS))
    return nn.Sequential(conv, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         head).eval()


def _gp_comp_onnx(tmp_path):
    C, H, W = _GP_COMP_CHW
    nodes = [helper.make_node("Conv", ["X", "w", "b"], ["c"],
                              kernel_shape=[1, 2]),
             helper.make_node("GlobalAveragePool", ["c"], ["p"]),
             helper.make_node("Flatten", ["p"], ["f"], axis=1),
             helper.make_node("Gemm", ["f", "g", "h"], ["Y"], transB=1)]
    inits = {"w": _GP_COMP_CONV.astype(np.float32),
             "b": _GP_COMP_CONV_BIAS.astype(np.float32),
             "g": _GP_COMP_HEAD.astype(np.float32),
             "h": _GP_COMP_HEAD_BIAS.astype(np.float32)}
    return save_onnx(tmp_path, nodes, [1, C, H, W], [1, 3], inits,
                     name="conv_gap_gemm")


@pytest.mark.parametrize("kind,method", [("ImageStar", "approx"),
                                         ("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
@pytest.mark.parametrize("variant", ["box"])
def test_global_avgpool_in_an_onnx_chain(tmp_path, kind, method, variant):
    """Conv > GlobalAveragePool > Flatten > Gemm loaded from ONNX, the
    ImageZono given without ``input_shape``: every onnxruntime output of a
    sample point is in the result, by LP on its rows. Catches the pooled set
    read wrongly by the next layers, a wrong mean or layout in the kernel
    behind a convolution, and the ONNX layer class not routed to the
    kernel. (Several sets reach the pool inside a model only after an exact
    split: ``test_global_avgpool_after_an_exact_relu_split_in_a_model``.)"""
    path = _gp_comp_onnx(tmp_path)
    lo, hi = pixel_box(*_GP_COMP_CHW, seed=2)
    sets = image_set(kind, lo, hi)
    what = f"Conv > GlobalAveragePool > Flatten > Gemm, {method}, {kind}"
    out = reach_model(load_onnx(path), sets, method, what)
    assert_covered(out, replay_onnx(path, samples(lo, hi),
                                    [1] + list(_GP_COMP_CHW)), what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_global_avgpool_in_a_torch_chain_is_the_exact_image(kind):
    """Conv > AdaptiveAvgPool2d(1) > Flatten > Linear (torch, float64,
    dyadic weights, 8 pixels per channel after the convolution) through
    ``reach_pytorch_model`` on a two-variable set: the result holds the exact
    image (rationals, the chain's affine map read from torch on the unit
    vectors) with zero slack and has no vertex outside it. Catches a wrong
    mean or layout behind a convolution in a torch model, and the pooled
    rows handed on wrongly (negated: the binding row of the polygon shows
    it, which a box or a 30-variable star cannot)."""
    model = _gp_comp_torch()
    chw = _GP_COMP_CHW
    n = int(np.prod(chw))
    with torch.no_grad():
        offset = model(torch.zeros((1,) + chw, dtype=torch.float64))
        cols = model(torch.eye(n, dtype=torch.float64).reshape((n,) + chw))
    M = (cols - offset).numpy().T
    centre, gen = _gp_pixels(chw, seed=4, shift=1.0)
    A, b = ((_GP_SQUARE_A, _GP_SQUARE_B) if kind == "ImageZono"
            else (_GP_POLY_A, _GP_POLY_B))
    pieces = rational.affine(rational.polytope(A, b), gen.tolist(),
                             centre.tolist())
    pieces = rational.affine(pieces, M.tolist(), offset.numpy()[0].tolist())
    what = f"Conv > AdaptiveAvgPool2d(1) > Flatten > Linear on a {kind}"
    out = reach_model(model, _gp_low_set(chw, centre, gen, kind), "approx",
                      what)
    _gp_check_exact(out, pieces, what)


# --------------------------------------------------------------------------- #
# POOL-3: a wrong-sized AdaptiveAvgPool2d hidden by a later global pool         #
# --------------------------------------------------------------------------- #
_GP_POOL3 = known_bug(
    7, "POOL-3: AdaptiveAvgPool2d(2) is computed as a global average pool "
       "(POOL-1); behind ReLU and AdaptiveAvgPool2d(1) the result has the "
       "right size and leaves out the true outputs, because the ReLU sees "
       "the mean of the whole image instead of the mean of each block")


def _gp_halves_box(chw):
    """An image box whose left half is near -1 and right half near +1 (a
    different interval per pixel, dyadic)."""
    lo, hi = pixel_box(*chw, seed=6)
    width = (hi - lo) / 8.0
    C, H, W = chw
    sign = np.where(np.arange(W) < W // 2, -1.0, 1.0)[None, None, :]
    lo = sign - width
    return lo, sign + width


@pytest.mark.parametrize("kind,method", [("ImageStar", "approx"),
                                         ("ImageZono", "approx")])
@pytest.mark.parametrize("model_name", [
    "gap-flatten",
    pytest.param("ada2-relu-gap-flatten", marks=_GP_POOL3)])
def test_global_avgpool_model_after_a_block_pool_holds_every_output(
        model_name, kind, method):
    """``AdaptiveAvgPool2d(2) > ReLU > AdaptiveAvgPool2d(1) > Flatten`` on a
    (2, 4, 6) image box whose left half is near -1 and right half near +1:
    every torch output must be in the result. The neighbour
    ``AdaptiveAvgPool2d(1) > Flatten`` on the same box, with the same replay
    and check, passes. Catches a block pool answered as a global one when a
    later global pool hides the wrong size. One case per lane of the
    dispatcher (Star, Zono): the method is not read on the failing path."""
    chw = (2, 4, 6)
    if model_name == "gap-flatten":
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    else:
        model = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.ReLU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten())
    lo, hi = _gp_halves_box(chw)
    what = f"{model_name} on a {kind}, {method}"
    out = reach_model(model.eval(), image_set(kind, lo, hi), method, what)
    assert_covered(out, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what)


# --------------------------------------------------------------------------- #
# Refusals                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["approx", "exact"])
@pytest.mark.parametrize("kind", ["Box", "Hexatope", "Octatope"])
def test_global_avgpool_on_a_flat_set_type_without_a_rule_is_refused(kind,
                                                                     method):
    """The dispatcher has no global-pool rule for Box, Hexatope or Octatope;
    it must raise, never return the input."""
    lo, hi = pixel_box(2, 2, 3)
    with pytest.raises(NotImplementedError):
        reach_layer(nn.AdaptiveAvgPool2d(1), [flat_set(kind, lo, hi)], method)


@pytest.mark.parametrize("kind,method", [
    ("Box", "approx"), ("Hexatope", "approx"), ("Octatope", "approx"),
    ("Zono", "exact"), ("ImageZono", "exact"), ("Box", "exact"),
    ("Hexatope", "exact"), ("Octatope", "exact")])
def test_global_avgpool_model_on_a_set_type_without_a_rule_is_refused(kind,
                                                                     method):
    """The rows that refuse, through the model entry, which decides the main
    answer: under approx the walk reaches the dispatcher's Box, Hexatope or
    Octatope lane, which has no global-pool rule (``NotImplementedError``);
    under exact the model entry refuses every set type without an exact
    method before any layer (today ``ValueError``, '... only supports
    approx'; the planned ``UnsupportedError`` subclasses
    ``NotImplementedError``). A flat set is given with ``input_shape``.
    Catches a lane that answers the pool with its input or a placeholder,
    and the model entry passing 'exact' on to a lane that never reads the
    method."""
    chw = (2, 2, 3)
    lo, hi = pixel_box(*chw)
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    if kind == "ImageZono":
        input_set, options = image_set(kind, lo, hi), {}
    else:
        input_set, options = flat_set(kind, lo, hi), {"input_shape": (1,) + chw}
    refusals = (NotImplementedError,) if method == "approx" else (
        NotImplementedError, ValueError)
    with pytest.raises(refusals):
        reach_model(model, input_set, method,
                    f"global pool model on {kind}, {method}", **options)


_GP_FLAT = known_bug(
    23, "global average pooling on a flat Star or Zono raises TypeError "
        "('GlobalAvgPool requires ImageStar/ImageZono input') instead of the "
        "declared refusal (NotImplementedError, later UnsupportedError); "
        "POOL-6",
    raises=TypeError)


@pytest.mark.parametrize("kind,entry", [
    pytest.param("Star", "layer", marks=_GP_FLAT),
    pytest.param("Star", "model", marks=_GP_FLAT),
    pytest.param("Zono", "layer", marks=_GP_FLAT),
    pytest.param("Zono", "model", marks=_GP_FLAT)])
def test_global_avgpool_on_a_flat_star_or_zono_is_refused(kind, entry):
    """A flat Star or Zono has no image shape for the kernel: a direct
    ``reach_layer`` call must refuse with NotImplementedError. Through the
    model entry with ``input_shape`` the tensor shape is known, so the call
    must refuse with NotImplementedError or answer with every torch output
    in the result (by LP on its rows). The Box, Hexatope and Octatope
    neighbours of
    ``test_global_avgpool_on_a_flat_set_type_without_a_rule_is_refused``
    pass with the same inputs. Catches the kernel's accidental exception
    and, once that is a refusal, a flat set pooled as if it were an image
    set."""
    chw = (2, 2, 3)
    lo, hi = pixel_box(*chw)
    s = flat_set(kind, lo, hi)
    if entry == "layer":
        with pytest.raises(NotImplementedError):
            reach_layer(nn.AdaptiveAvgPool2d(1), [s], "approx")
        return
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    what = f"global pool on a flat {kind} with input_shape"
    try:
        out = reach_model(model, s, "approx", what, input_shape=(1,) + chw)
    except NotImplementedError:
        return
    assert_covered(out, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what)


# --------------------------------------------------------------------------- #
# A Reshape of a flat set in front of the pool                                  #
# --------------------------------------------------------------------------- #
_GP_RESHAPED = known_bug(
    12, "an ONNX Reshape that feeds a GlobalAveragePool keeps a flat set "
        "flat: the walker's list of spatial consumers (_SPATIAL_CONSUMERS) "
        "names OnnxGlobalAveragePool, which load_onnx never builds, and not "
        "OnnxGlobalAveragePoolWithKnownInputShape, which it always builds, "
        "so the pool gets a flat Star and raises TypeError where the torch "
        "network (view, then AdaptiveAvgPool2d) is answered", raises=TypeError)


class _ViewThenPool(nn.Module):
    """``x.view(1, 2, 3, 5)`` then ``AdaptiveAvgPool2d(1)``: the torch twin of
    the ONNX Reshape then GlobalAveragePool."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(x.view(1, 2, 3, 5))


@pytest.mark.parametrize("route", [
    pytest.param("onnx-reshape", marks=_GP_RESHAPED), "torch-view"])
def test_global_avgpool_after_a_reshape_of_a_flat_set(tmp_path, route):
    """A flat Star of 30 entries given with ``input_shape`` (1, 30), reshaped
    to (1, 2, 3, 5) and pooled, as an ONNX Reshape then GlobalAveragePool
    and as a torch ``view`` then ``AdaptiveAvgPool2d(1)``: the walker turns
    the reshaped set into an image set for a pooling consumer, so the reach
    must answer, and every true output (onnxruntime, torch float64) must be
    in the result by LP on its rows. Catches a pooling layer class the
    walker does not count as a spatial consumer, and the image set built in
    the wrong (height, width, channel) order."""
    chw = (2, 3, 5)
    n = int(np.prod(chw))
    lo, hi = pixel_box(*chw, seed=1)
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    if route == "onnx-reshape":
        path = save_onnx(tmp_path, [
            helper.make_node("Reshape", ["X", "s"], ["r"]),
            helper.make_node("GlobalAveragePool", ["r"], ["Y"])],
            [1, n], [1, chw[0], 1, 1],
            {"s": np.array([1, *chw], dtype=np.int64)}, name="reshape_gap")
        model = load_onnx(path)
        outputs = replay_onnx(path, samples(lo, hi), [1, n])
    else:
        model = _ViewThenPool().eval()
        outputs = replay_torch(model, samples(lo, hi), (1, n))
    what = f"Reshape to (1, 2, 3, 5) then global pool ({route}) on a flat Star"
    out = reach_model(model, flat_set("Star", lo, hi), "approx", what,
                      input_shape=(1, n))
    assert_covered(out, outputs, what)


# --------------------------------------------------------------------------- #
# After an exact split inside the model: several sets reach the layer          #
# --------------------------------------------------------------------------- #
def _gp_after_split_box(chw=(2, 2, 4), seed=12):
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
def test_global_avgpool_after_an_exact_relu_split_in_a_model(kind, method):
    """ReLU > AdaptiveAvgPool2d(1) > Flatten > Linear through ``reach_pytorch_model``
    on a (2, 2, 4) box with three pixels straddling zero: by the exact method
    the ReLU hands up to eight stars to the layer in one call. Every torch
    output of a sample point is in a returned set, by LP on the rows.
    Catches only the first or the last of several sets answered inside the
    model, and a result of one set reused for another."""
    chw = (2, 2, 4)
    layer = nn.AdaptiveAvgPool2d(1)
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
    lo, hi = _gp_after_split_box(chw)
    what = f"ReLU > AdaptiveAvgPool2d(1) > Flatten > Linear, {method}, {kind}"
    out = reach_model(model, image_set(kind, lo, hi), method, what)
    if method == "exact" and len(out) < 2:
        pytest.fail(f"test setup: the exact ReLU gave {len(out)} set(s)")
    assert_covered(out, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what)
