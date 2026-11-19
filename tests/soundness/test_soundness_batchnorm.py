"""Soundness cells for BatchNorm1d and BatchNorm2d in evaluation mode.

In evaluation mode a BatchNorm is the per-channel affine map y = s x + t with
s = gamma / sqrt(var + eps) and t = beta - s mean. Three code paths answer it:

- the kernels of ``n2v/nn/layer_ops/batchnorm_reach.py`` through the
  dispatcher, one per set type (the Hexatope and Octatope lanes map by a
  diagonal float64 ``nn.Linear``); they read no option and not the method;
- the graph handler ``n2v.nn.reach._handle_batchnorm_flat`` for a flat set
  that stands for an (N, C, ...) tensor (through ONNX also in
  ``test_soundness_batchnorm_flat_cells.py``);
- the fusion of a Conv2d or Linear with the BatchNorm after it at the model
  entry (``n2v/utils/model_preprocessing.py``), never for a Box, whose rule
  rounds its own fold outward (the fold's own rounding for the other set
  types is PLAN step 7b, marked in ``test_soundness_graph_walk.py``).

Every layer here has eps = 1/4 (or 2^-10 in an ONNX file) and running
variances with var + eps a perfect square, so the real-number layer on the
stored parameter values is rational and computed here in fractions, never by
n2v. Oracles:

- the exact image in rationals (``tests/_oracles/rational.py``): the input
  set's own predicate polytope (or its box) mapped by the exact s and t,
  every vertex a member of one returned set with zero slack; for the Star
  and Zono kernels, which claim exactness, and for ``method='exact'``, no
  vertex of the result outside the image; for the Box rule, which rounds
  outward, the bounds against the exact endpoint images with non-dyadic
  parameters;
- torch in float64 (onnxruntime for ONNX files) on sample points, each a
  member of some returned set by one LP on the set's own rows.

Layout: images are (C, H, W) in the network and (H, W, C) in an image set;
true outputs are converted with ``out_chw=``. Images have two channels and
height unlike width.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_image_star,
                                constrained_star, flat_box, flat_set,
                                hwc_index, image_set, image_split_cover,
                                pixel_box, reach_model, reach_one_layer,
                                replay_onnx, replay_torch, samples, save_onnx,
                                split_cover)
from tests.known_bugs import known_bug


_BN_EPS = 0.25
# name: (gamma, beta, running mean, running var) per feature, or None for
# gamma and beta when the layer has affine=False. var + eps is 1/4, 1, 4,
# 16: the scale is dyadic, so float64 is exact on dyadic data.
_BN1D_PARAMS = {
    # positive, negative and zero scales; zero and non-zero shifts
    "mixed": ([1.5, -0.75, 0.0, 2.0, -2.5], [0.25, 0.0, -1.0, 0.5, 0.0],
              [0.5, -1.0, 0.25, 0.0, 1.5], [0.0, 0.75, 3.75, 15.75, 0.75]),
    "affine-off": (None, None, [0.5, -1.0, 0.25, 0.0, 1.5],
                   [0.0, 0.75, 3.75, 15.75, 0.75]),
}
_BN2D_PARAMS = {
    # |scale| 3 in the first channel, a negative scale in the second
    "mixed": ([1.5, -0.75], [0.25, -1.0], [0.5, -1.0], [0.0, 3.75]),
    # a zero scale: the channel is the constant beta
    "zero-scale": ([0.0, 2.0], [0.5, 0.25], [1.0, -0.5], [0.75, 0.0]),
    "affine-off": (None, None, [0.5, -1.0], [3.75, 0.0]),
}


def _bn_layer(cls, params, eps=_BN_EPS):
    """An eval-mode BatchNorm of ``cls`` with the given (gamma, beta, mean,
    var); affine=False when gamma is None."""
    gamma, beta, mean, var = params
    bn = cls(len(mean), eps=eps, affine=gamma is not None)
    with torch.no_grad():
        if gamma is not None:
            bn.weight.copy_(torch.tensor(gamma))
            bn.bias.copy_(torch.tensor(beta))
        bn.running_mean.copy_(torch.tensor(mean))
        bn.running_var.copy_(torch.tensor(var))
    return bn.eval()


def _exact_sqrt(q):
    """The rational square root of ``q``; the setup fails when there is
    none."""
    num, den = math.isqrt(q.numerator), math.isqrt(q.denominator)
    if Fraction(num * num, den * den) != q:
        pytest.fail(f"test setup: var + eps = {q} is not a rational square")
    return Fraction(num, den)


def _bn_exact(bn):
    """``(s, t)``: the real-number layer y = s x + t per feature, from the
    stored parameter values (float32, read exactly) in fractions."""
    n = bn.running_mean.numel()
    exact = lambda t: [Fraction(float(v)) for v in                 # noqa: E731
                       t.detach().cpu().double().numpy()]
    mean, var = exact(bn.running_mean), exact(bn.running_var)
    gamma = exact(bn.weight) if bn.weight is not None else [Fraction(1)] * n
    beta = exact(bn.bias) if bn.bias is not None else [Fraction(0)] * n
    eps = Fraction(bn.eps)
    s = [g / _exact_sqrt(v + eps) for g, v in zip(gamma, var)]
    t = [b - si * m for b, si, m in zip(beta, s, mean)]
    return s, t


def _bn_input_pieces(input_set):
    """The input set as rational pieces: the identity on its predicate
    polytope, mapped by its own centre and generators (read through
    ``rational.star``; the input is built by the test, not computed)."""
    c, G, C, d, lb, ub = rational.star(input_set)
    n = len(lb)
    A, b = [list(r) for r in C], list(d)
    for j in range(n):
        unit = [Fraction(int(k == j)) for k in range(n)]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    return rational.affine(rational.polytope(A, b), G, c)


def _bn_box_piece(lo, hi):
    """The identity on the box [lo, hi] as one rational piece. Built directly
    (``rational.box`` first proves the polytope bounded, which for six
    variables costs seconds; a box is bounded)."""
    lo, hi = rational.vector(lo), rational.vector(hi)
    n = len(lo)
    unit = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    A = unit + [[-v for v in row] for row in unit]
    return [rational.Piece(A, hi + [-v for v in lo], unit, [Fraction(0)] * n)]


def _bn_check(out, pieces, what, exact):
    """Every exact piece inside one returned set (zero slack); with
    ``exact``, no vertex of a returned set outside the image."""
    computed = [rational.star(s) for s in out]
    tests = [rational.Membership(s) for s in computed]
    for i, piece in enumerate(pieces):
        image = piece.image()
        if not any(all(t(y) for y in image) for t in tests):
            missing = [y for y in image if not any(t(y) for t in tests)]
            raise AssertionError(
                f"{what}: exact piece {i} of {len(pieces)} is not inside one "
                f"returned set; {len(missing)} of its {len(image)} vertices "
                f"are in none of the {len(computed)} sets (first "
                f"{[float(v) for v in missing[0]] if missing else None})")
    if exact:
        for k, s in enumerate(computed):
            far = rational.unreached(pieces, rational.set_vertices(s))
            assert not far, (f"{what}: returned set {k} has a vertex that is "
                             f"no output of the layer: "
                             f"{[float(v) for v in far[0][1]]}")


def _bn_diag(s, t):
    """The rational matrix and offset of the per-feature map."""
    n = len(s)
    return ([[s[i] if i == j else Fraction(0) for j in range(n)]
             for i in range(n)], t)


# --------------------------------------------------------------------------- #
# BatchNorm1d on the flat set types                                            #
# --------------------------------------------------------------------------- #
_BN_FLAT_KINDS = ("Box", "Star", "Zono", "Hexatope", "Octatope")


def _bn1d_cells():
    """Every input shape with the mixed parameters; affine=False changes only
    the fold of ``_get_bn_params``, which every lane shares, so it runs on the
    box of each set type."""
    cells = []
    for params in _BN1D_PARAMS:
        for kind in _BN_FLAT_KINDS:
            variants = ("box", "split") + (("rows",) if kind == "Star" else ())
            if params == "affine-off":
                variants = ("box",)
            for variant in variants:
                cells.append(pytest.param(params, kind, variant,
                                          id=f"{params}-{kind}-{variant}"))
    return cells


@pytest.mark.parametrize("params,kind,variant", _bn1d_cells())
def test_batchnorm1d_on_flat_sets_holds_the_exact_image(params, kind, variant):
    """BatchNorm1d through ``reach_layer`` on Box, Star, Zono, Hexatope and
    Octatope: a dyadic box whose coordinates straddle zero, are positive,
    negative or of zero width, three sets covering it, or a Star with a
    binding row. Every torch output of a sample point is in a returned set
    (LP on the rows); for one set the exact image (rationals) is inside the
    result with zero slack, and for Star and Zono no vertex of the result is
    outside it. Catches the shift dropped or applied before the scale, eps
    left out, mean not multiplied by the scale, a negative scale taken by its
    absolute value, a bias defaulting to ones when affine=False, the plain
    Zono path, and only one of several sets answered."""
    bn = _bn_layer(nn.BatchNorm1d, _BN1D_PARAMS[params])
    n = bn.running_mean.numel()
    lo, hi = flat_box(n, seed=3, zero_width=True)
    if variant == "box":
        sets, points = [flat_set(kind, lo, hi)], samples(lo, hi)
    elif variant == "split":
        sets, points = split_cover(kind, lo, hi, 3), samples(lo, hi)
    else:
        star, points = constrained_star(lo, hi, seed=3)
        sets = [star]
    what = f"BatchNorm1d {params} on {kind} {variant}"
    out = reach_one_layer(bn, sets, "approx", what)
    assert_covered(out, replay_torch(bn, points, (1, n)), what)
    if variant == "split":
        return
    if variant == "box":
        pieces = _bn_box_piece(lo.tolist(), hi.tolist())
    else:
        pieces = _bn_input_pieces(sets[0])
    pieces = rational.affine(pieces, *_bn_diag(*_bn_exact(bn)))
    _bn_check(out, pieces, what, exact=kind in ("Star", "Zono"))


# The Box kernel rounds its folding outward (_bn_fold_mag, _widen_sum). Its
# bounds are checked against the exact endpoint images of the real-number
# layer with non-dyadic float32 parameters, where float64 rounds. One
# feature for every combination of: the sign of the scale (positive,
# negative, zero); the input interval (lower end exactly 0, upper end
# exactly 0, the point 0, straddling, positive, negative); the folded terms
# (beta and mean both zero, only the mean non-zero, only beta non-zero, both
# non-zero); then the features below.
_BN_ROOTS = (Fraction(3, 2), Fraction(5, 2))   # sqrt(var + eps), var 2, 6
_BN_SIGNS = (1.0, -1.0, 0.0)
_BN_INTERVALS = ("lo0", "hi0", "point0", "straddle", "pos", "neg")
_BN_FOLDS = ("none", "mean", "beta", "both")
_BN_ROUNDING_FEATURES = [(g, x, f) for g in _BN_SIGNS for x in _BN_INTERVALS
                         for f in _BN_FOLDS]
# Features where float64 rounds inward: zero folded terms, one endpoint
# exactly 0, and gamma and the other endpoint drawn so that the plain
# float64 product fl(fl(gamma / root) * x) lies inside the exact image
# (``_bn_inward_draw``). Only a margin charged for the right endpoint covers
# them.
_BN_INWARD_FEATURES = [(g, x, "inward") for g in (1.0, -1.0)
                       for x in ("lo0", "hi0") for _ in range(2)]
# Features whose folded terms cancel: beta is the float32 nearest to
# scale * mean, so the folded shift beta - scale * mean is a rounding residue
# far smaller than either term, while the rounding of its computation scales
# with |beta| + |scale * mean|. A margin taken from |shift| misses the exact
# image at an input endpoint 0.
_BN_CANCEL_FEATURES = [(g, x, "cancel") for g in (1.0, -1.0)
                       for x in _BN_INTERVALS]


def _bn_box_rounding_features(case):
    """The features of the Box rounding layer of ``case``; the inward and the
    cancelling ones need a beta, so "affine-off" has neither."""
    return _BN_ROUNDING_FEATURES + (
        _BN_INWARD_FEATURES + _BN_CANCEL_FEATURES if case != "affine-off"
        else [])


def _bn_inward_draw(rng, sign, interval, root):
    """``(gamma, x)``: a float32 gamma of the given sign and the non-zero
    endpoint x of an interval with the other endpoint 0, such that the float64
    product of the rounded scale and x is on the inner side of the exact
    image s x, s = gamma / root. Deterministic for a seeded ``rng``."""
    for _ in range(10000):
        gamma = float(np.float32(sign * rng.uniform(0.5, 2.0)))
        x = rng.uniform(0.1, 1.0) * (1.0 if interval == "lo0" else -1.0)
        exact = Fraction(gamma) / root * Fraction(x)
        computed = Fraction((gamma / float(root)) * x)
        # the image is [min(0, s x), max(0, s x)]; s x is its lower end when
        # negative and its upper end when positive
        if (exact < 0 and computed > exact) or (exact > 0 and computed < exact):
            return gamma, x
    pytest.fail("test setup: no inward-rounding draw found")


def _bn_box_rounding_layer(case):
    """A BatchNorm1d over ``_BN_ROUNDING_FEATURES``; ``case`` "affine-off"
    builds it with affine=False (gamma 1, beta 0)."""
    features = _bn_box_rounding_features(case)
    n = len(features)
    rng = np.random.default_rng(114)
    weight = np.zeros(n, dtype=np.float32)
    bias = np.zeros(n, dtype=np.float32)
    mean = np.zeros(n, dtype=np.float32)
    # var + eps = (3/2)^2 or (5/2)^2: the scale gamma / root is not dyadic
    var = np.array([2.0, 6.0] * (n // 2) + [2.0] * (n % 2), dtype=np.float32)
    for i, (sign, interval, fold) in enumerate(features):
        if fold == "inward":
            weight[i] = _bn_inward_draw(np.random.default_rng(1000 + i), sign,
                                        interval, _BN_ROOTS[i % 2])[0]
            continue
        weight[i] = sign * rng.uniform(0.5, 2.0)
        if fold in ("mean", "both", "cancel"):
            mean[i] = rng.uniform(-1.0, 1.0)
        if fold in ("beta", "both"):
            bias[i] = rng.uniform(-1.0, 1.0)
        if fold == "cancel":
            bias[i] = float(weight[i]) / float(_BN_ROOTS[i % 2]) * float(
                mean[i])
    bn = nn.BatchNorm1d(n, eps=_BN_EPS, affine=case != "affine-off")
    with torch.no_grad():
        if case != "affine-off":
            bn.weight.copy_(torch.from_numpy(weight))
            bn.bias.copy_(torch.from_numpy(bias))
        bn.running_mean.copy_(torch.from_numpy(mean))
        bn.running_var.copy_(torch.from_numpy(var))
    return bn.eval()


def _bn_box_rounding_input(case):
    """The input interval of every feature of the layer of ``case``,
    non-dyadic float64 values where not 0."""
    features = _bn_box_rounding_features(case)
    rng = np.random.default_rng(115)
    lo, hi = [], []
    for i, (sign, kind, fold) in enumerate(features):
        if fold == "inward":
            x = _bn_inward_draw(np.random.default_rng(1000 + i), sign, kind,
                                _BN_ROOTS[i % 2])[1]
            lo.append(0.0 if kind == "lo0" else x)
            hi.append(x if kind == "lo0" else 0.0)
            continue
        a, b = rng.uniform(0.1, 1.0), rng.uniform(0.1, 1.0)
        lo.append({"lo0": 0.0, "hi0": -a, "point0": 0.0, "straddle": -a,
                   "pos": a, "neg": -a - b}[kind])
        hi.append({"lo0": b, "hi0": 0.0, "point0": 0.0, "straddle": b,
                   "pos": a + b, "neg": -a}[kind])
    return np.array(lo), np.array(hi)


@pytest.mark.parametrize("case", ["affine", "affine-off"])
def test_batchnorm1d_box_holds_the_exact_endpoint_images(case):
    """BatchNorm1d on a Box with non-dyadic float32 parameters, one feature
    for every combination of scale sign, input interval (endpoints exactly 0
    among them) and zero or non-zero folded terms, with affine=True and
    affine=False: each output bound is compared in fractions with the images
    of the input endpoints under the real-number layer (eps chosen so that
    var + eps is a rational square), no slack. Catches the rounding margin
    dropped where only one of the folded terms is zero, a margin charged for
    the wrong input endpoint under a negative scale, a margin taken from the
    folded shift where beta and scale * mean cancel, an upper bound rounded
    down, and the endpoints swapped for a negative scale."""
    bn = _bn_box_rounding_layer(case)
    lo, hi = _bn_box_rounding_input(case)
    what = f"BatchNorm1d on a Box ({case})"
    out = reach_one_layer(bn, Box(lo.reshape(-1, 1), hi.reshape(-1, 1)),
                          "approx", what)
    assert len(out) == 1, f"{what}: {len(out)} boxes returned for one"
    s, t = _bn_exact(bn)
    lb = np.asarray(out[0].lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(out[0].ub, dtype=np.float64).reshape(-1)
    worst = []
    for i in range(lo.size):
        images = [s[i] * Fraction(float(x)) + t[i] for x in (lo[i], hi[i])]
        if Fraction(float(lb[i])) > min(images):
            worst.append((i, "lower", float(Fraction(float(lb[i]))
                                             - min(images))))
        if Fraction(float(ub[i])) < max(images):
            worst.append((i, "upper", float(max(images)
                                             - Fraction(float(ub[i])))))
    assert not worst, (f"{what}: bounds inside the exact image (feature, "
                       f"side, amount): {worst}")


# --------------------------------------------------------------------------- #
# BatchNorm2d on the image set types                                           #
# --------------------------------------------------------------------------- #
_BN_CHW = (2, 3, 5)


def _bn2d_cells():
    """Every input shape with the mixed parameters; the zero scale and
    affine=False change no code path of the image kernels and are checked,
    with the exact oracle, by ``test_batchnorm2d_is_the_exact_image``."""
    cells = []
    for kind, variants in (("ImageStar", ("box", "split", "rows")),
                           ("ImageZono", ("box", "split"))):
        for variant in variants:
            cells.append(pytest.param("mixed", kind, variant,
                                      id=f"mixed-{kind}-{variant}"))
    return cells


@pytest.mark.parametrize("params,kind,variant", _bn2d_cells())
def test_batchnorm2d_holds_every_torch_output(params, kind, variant):
    """BatchNorm2d through ``reach_layer`` on an ImageStar or ImageZono: a
    pixel box, three sets covering it, or a star with a binding row; scale
    above 1 in the first channel, negative in the second. Every torch
    output of a sample point is in a returned set, by LP on the rows.
    Catches the channel index applied to the height or width axis, the
    generators scaled in one channel only or by |scale| (the binding row),
    the centre shifted before scaling or not scaled, and only one of several
    sets answered."""
    bn = _bn_layer(nn.BatchNorm2d, _BN2D_PARAMS[params])
    chw = _BN_CHW
    lo, hi = pixel_box(*chw, seed=9)
    if variant == "box":
        sets, points = [image_set(kind, lo, hi)], samples(lo, hi)
    elif variant == "split":
        sets, points = image_split_cover(kind, lo, hi, 3), samples(lo, hi)
    else:
        star, points = constrained_image_star(lo, hi, seed=9)
        sets = [star]
    what = f"BatchNorm2d {params} on {kind} {variant}"
    out = reach_one_layer(bn, sets, "approx", what)
    assert_covered(out, replay_torch(bn, points, (1,) + chw), what,
                   out_chw=chw)


_BN_POLY_A = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, Fraction(1, 2)]]
_BN_POLY_B = [1, 1, 1, 1, 1]
_BN_SQUARE_A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
_BN_SQUARE_B = [1, 1, 1, 1]


def _bn_low_image(chw, kind, seed=3):
    """``(set, centre, gen)``: ``centre + gen . a`` in (C, H, W) order, as an
    ImageStar over the polygon with a binding row or an ImageZono over the
    square, stored in (H, W, C) order. Dyadic."""
    C, H, W = chw
    n = C * H * W
    rng = np.random.default_rng(seed)
    centre = rng.integers(-8, 9, size=n) / 4.0
    gen = rng.integers(-4, 5, size=(n, 2)) / 8.0
    order = hwc_index(C, H, W)
    c, G = centre[order], gen[order]
    if kind == "ImageZono":
        return ImageZono(c.reshape(-1, 1), G.copy(), H, W, C), centre, gen
    A = np.array([[float(v) for v in r] for r in _BN_POLY_A])
    b = np.array([float(v) for v in _BN_POLY_B]).reshape(-1, 1)
    return (ImageStar(np.hstack([c[:, None], G]), A, b, -np.ones((2, 1)),
                      np.ones((2, 1)), H, W, C), centre, gen)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("params", sorted(_BN2D_PARAMS))
def test_batchnorm2d_is_the_exact_image(params, kind):
    """BatchNorm2d on a set of two predicate variables: the result holds
    every vertex of the exact image (rationals, in the stored (H, W, C)
    order) with zero slack and has no vertex outside it. Catches a scale or
    shift applied to the wrong channel or entry, and the centre mapped
    differently from the generators."""
    bn = _bn_layer(nn.BatchNorm2d, _BN2D_PARAMS[params])
    chw = _BN_CHW
    C, H, W = chw
    s_ch, t_ch = _bn_exact(bn)
    per = H * W
    s = [s_ch[p // per] for p in range(C * per)]
    t = [t_ch[p // per] for p in range(C * per)]
    low, centre, gen = _bn_low_image(chw, kind)
    A, b = ((_BN_SQUARE_A, _BN_SQUARE_B) if kind == "ImageZono"
            else (_BN_POLY_A, _BN_POLY_B))
    pieces = rational.affine(rational.polytope(A, b), gen.tolist(),
                             centre.tolist())
    pieces = rational.affine(pieces, *_bn_diag(s, t))
    order = hwc_index(C, H, W)
    m = len(order)
    pieces = rational.affine(pieces, [[int(order[k] == j) for j in range(m)]
                                      for k in range(m)])
    what = f"BatchNorm2d {params} on a two-variable {kind}"
    out = reach_one_layer(bn, [low], "approx", what)
    _bn_check(out, pieces, what, exact=True)


# --------------------------------------------------------------------------- #
# A frozen layer on wide sets with more variables than entries                 #
# --------------------------------------------------------------------------- #
def _bn_wide_points(centre, G, row, rhs, n=128, seed=0):
    """At least ``n`` member points ``centre + G a`` (flat, network order) of
    the set over a in [-1, 1]^k with ``row . a <= rhs`` (no row when ``row``
    is None), computed without n2v: corners, half-grid points and, with a
    row, points on its face (``row[0]`` is 1)."""
    rng = np.random.default_rng(seed)
    k = G.shape[1]
    points = []
    for attempt in range(200 * n):
        if len(points) >= n:
            break
        a = (rng.choice([-1.0, 1.0], size=k) if attempt % 3 == 0
             else rng.integers(-2, 3, size=k) / 2.0)
        if row is not None and attempt % 3 == 2:
            a[0] = 0.0
            a[0] = rhs - row @ a
            if abs(a[0]) > 1:
                continue
        if row is None or row @ a <= rhs:
            points.append(centre + G @ a)
    if len(points) < n:
        pytest.fail("test setup: too few member points")
    return points


def _bn_wide_input(kind, lo, hi, chw=None, seed=0):
    """``(set, points)``: the box [lo, hi] as a set of ``kind``, points in
    the network's order. A Zono or ImageZono gets two generators more than
    entries, a Star or ImageStar two predicate variables more than entries,
    tied to the first box variable by the binding row
    ``a_0 + a_n - a_(n+1) / 2 <= 1/2``; their sets are the box plus the two
    extra directions. Image sets store (H, W, C): ``chw`` is the image
    shape. Dyadic."""
    if kind in ("Box", "Hexatope", "Octatope"):
        return flat_set(kind, lo, hi), samples(lo, hi)
    flat_lo, flat_hi = lo.reshape(-1), hi.reshape(-1)
    n = flat_lo.size
    rng = np.random.default_rng(seed)
    centre = (flat_lo + flat_hi) / 2.0
    G = np.hstack([np.diag((flat_hi - flat_lo) / 2.0),
                   rng.integers(-4, 5, size=(n, 2)) / 2.0])
    row, rhs = None, None
    if kind in ("Star", "ImageStar"):
        row = np.zeros(n + 2)
        row[0], row[n], row[n + 1] = 1.0, 1.0, -0.5
        rhs = 0.5
    points = [p.reshape(lo.shape)
              for p in _bn_wide_points(centre, G, row, rhs, seed=seed)]
    order = hwc_index(*chw) if chw is not None else np.arange(n)
    c, V = centre[order].reshape(-1, 1), G[order]
    if kind in ("Zono", "ImageZono"):
        if kind == "Zono":
            return Zono(c, V), points
        C, H, W = chw
        return ImageZono(c, V, H, W, C), points
    k = n + 2
    rows = np.vstack([np.eye(k), -np.eye(k), row[None, :]])
    d = np.concatenate([np.ones(2 * k), [rhs]]).reshape(-1, 1)
    pred = (-np.ones((k, 1)), np.ones((k, 1)))
    if kind == "Star":
        return Star(np.hstack([c, V]), rows, d, *pred), points
    C, H, W = chw
    return ImageStar(np.hstack([c, V]), rows, d, *pred, H, W, C), points


@pytest.mark.parametrize("kind", _BN_FLAT_KINDS + ("ImageStar", "ImageZono"))
def test_batchnorm_frozen_on_a_wide_set_holds_every_torch_output(kind):
    """BatchNorm1d (flat sets) and BatchNorm2d (image sets) as a user hands
    them over for inference, the parameters frozen by
    ``requires_grad_(False)``, through ``reach_layer`` on a box eight times
    the usual one (bounds well beyond +-1, intervals wider than 2), given for
    Zono and ImageZono with two generators more than entries and for Star and
    ImageStar with two predicate variables more than entries tied by a
    binding row (the shapes a ReLU relaxation leaves). Every torch output of
    a member point is in the result (LP on the rows). Catches a parameter
    read only when it requires a gradient, a kernel that assumes one
    generator or predicate variable per entry, and a result that reads one
    variable in place of another, which data inside [-1, 1] would hide."""
    if kind.startswith("Image"):
        bn = _bn_layer(nn.BatchNorm2d, _BN2D_PARAMS["mixed"])
        lo, hi = pixel_box(*_BN_CHW, seed=11)
        chw, in_shape = _BN_CHW, (1,) + _BN_CHW
    else:
        bn = _bn_layer(nn.BatchNorm1d, _BN1D_PARAMS["mixed"])
        lo, hi = flat_box(5, seed=11)
        chw, in_shape = None, (1, 5)
    bn.requires_grad_(False)
    input_set, points = _bn_wide_input(kind, 8.0 * lo, 8.0 * hi, chw, seed=11)
    what = f"frozen {type(bn).__name__} on a wide {kind}"
    out = reach_one_layer(bn, [input_set], "approx", what)
    assert_covered(out, replay_torch(bn, points, in_shape), what,
                   out_chw=chw)


# --------------------------------------------------------------------------- #
# Composition through the model entry points (fused and unfused)               #
# --------------------------------------------------------------------------- #
def _bn_affine_of(model, in_shape):
    """``(M, offset)``: the affine map of a model built from affine layers,
    read from torch in float64 at 0 and on the unit vectors (exact for the
    dyadic weights here)."""
    n = int(np.prod(in_shape))
    with torch.no_grad():
        offset = model(torch.zeros((1,) + tuple(in_shape),
                                   dtype=torch.float64))
        cols = model(torch.eye(n, dtype=torch.float64).reshape(
            (n,) + tuple(in_shape)))
    return (cols - offset).numpy().T, offset.numpy()[0]


# How a chain reaches its BatchNorm: "fused", folded into the Conv2d or
# Linear before it by the model entry; "kernel", kept (a Dropout in between)
# so that the BatchNorm rule runs; "bare", folded, the Conv2d or Linear
# without bias and the BatchNorm with affine=False (the fold's -mean and its
# gamma 1, beta 0). A Box is never folded, so it has only "kernel".
_BN_CHAIN_VARIANTS = ("fused", "kernel", "bare")


def _bn_chain_cells(kinds):
    return [pytest.param(kind, variant, id=f"{kind}-{variant}")
            for kind in kinds for variant in _BN_CHAIN_VARIANTS
            if kind != "Box" or variant == "kernel"]


def _bn_flat_chain(variant):
    """Linear (4 -> 5) > [Dropout >] BatchNorm1d > Linear (5 -> 3), float64,
    dyadic, for a variant of ``_BN_CHAIN_VARIANTS``."""
    rng = np.random.default_rng(21)
    first = nn.Linear(4, 5, bias=variant != "bare").double()
    last = nn.Linear(5, 3).double()
    with torch.no_grad():
        first.weight.copy_(torch.from_numpy(rng.integers(-4, 5, (5, 4)) / 4.0))
        bias = torch.from_numpy(rng.integers(-4, 5, 5) / 8.0)
        if first.bias is not None:
            first.bias.copy_(bias)
        last.weight.copy_(torch.from_numpy(rng.integers(-4, 5, (3, 5)) / 4.0))
        last.bias.copy_(torch.from_numpy(rng.integers(-4, 5, 3) / 8.0))
    bn = _bn_layer(nn.BatchNorm1d, _BN1D_PARAMS[
        "affine-off" if variant == "bare" else "mixed"]).double()
    layers = [first] + ([nn.Dropout(0.5)] if variant == "kernel" else []) + [
        bn, last]
    return nn.Sequential(*layers).eval()


@pytest.mark.parametrize("kind,variant", _bn_chain_cells(_BN_FLAT_KINDS))
def test_batchnorm1d_in_a_chain_holds_the_exact_image(kind, variant):
    """Linear > BatchNorm1d > Linear through ``reach_pytorch_model``, with
    the BatchNorm folded into the Linear by the model entry (with and without
    the Linear's bias and the BatchNorm's affine parameters) or kept: the
    result holds the exact image of the input box (rationals) with zero
    slack, and every torch output of a sample point (LP on the rows).
    Catches a wrong fold (the mean folded with the wrong sign when the Linear
    has no bias, beta not 0 for affine=False, the scale on the wrong axis)
    and a wrong standalone kernel whose rows the last Linear consumes."""
    model = _bn_flat_chain(variant)
    lo, hi = flat_box(4, seed=5)
    what = f"Linear > BatchNorm1d > Linear ({variant}) on {kind}"
    out = reach_model(model, flat_set(kind, lo, hi), "approx", what)
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, 4)), what)
    M, offset = _bn_affine_of(model, (4,))
    pieces = rational.affine(_bn_box_piece(lo.tolist(), hi.tolist()),
                             M.tolist(), offset.tolist())
    _bn_check(out, pieces, what, exact=False)


def _bn_image_chain(variant):
    """Conv (1x2, mixing the channels) > [Dropout2d >] BatchNorm2d > Flatten
    > Linear, float64, dyadic, for a variant of ``_BN_CHAIN_VARIANTS``."""
    conv = nn.Conv2d(2, 2, (1, 2), bias=variant != "bare").double()
    head = nn.Linear(2 * 3 * 4, 3).double()
    rng = np.random.default_rng(22)
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(
            rng.integers(-4, 5, (2, 2, 1, 2)) / 4.0))
        bias = torch.from_numpy(rng.integers(-4, 5, 2) / 8.0)
        if conv.bias is not None:
            conv.bias.copy_(bias)
        head.weight.copy_(torch.from_numpy(rng.integers(-4, 5, (3, 24)) / 4.0))
        head.bias.copy_(torch.from_numpy(rng.integers(-4, 5, 3) / 8.0))
    bn = _bn_layer(nn.BatchNorm2d, _BN2D_PARAMS[
        "affine-off" if variant == "bare" else "mixed"]).double()
    layers = [conv] + ([nn.Dropout2d(0.5)] if variant == "kernel" else []) + [
        bn, nn.Flatten(), head]
    return nn.Sequential(*layers).eval()


@pytest.mark.parametrize("kind,variant",
                         _bn_chain_cells(["ImageStar", "ImageZono"]))
def test_batchnorm2d_in_a_torch_chain_is_the_exact_image(kind, variant):
    """Conv > BatchNorm2d > Flatten > Linear through ``reach_pytorch_model``
    on a two-variable image set, the BatchNorm folded into the Conv (with and
    without the Conv's bias and the BatchNorm's affine parameters) or kept:
    the result holds the exact image (rationals, the chain's affine map read
    from torch on the unit vectors) with zero slack and has no vertex outside
    it. Catches the BatchNorm's channel layout read wrongly after a Conv and
    at the Flatten, and a wrong fold (the mean's sign without a Conv bias,
    beta not 0 for affine=False, the scale on the input-channel axis)."""
    model = _bn_image_chain(variant)
    chw = _BN_CHW
    M, offset = _bn_affine_of(model, chw)
    low, centre, gen = _bn_low_image(chw, kind, seed=4)
    A, b = ((_BN_SQUARE_A, _BN_SQUARE_B) if kind == "ImageZono"
            else (_BN_POLY_A, _BN_POLY_B))
    pieces = rational.affine(rational.polytope(A, b), gen.tolist(),
                             centre.tolist())
    pieces = rational.affine(pieces, M.tolist(), offset.tolist())
    what = f"Conv > BatchNorm2d > Flatten > Linear ({variant}) on {kind}"
    out = reach_model(model, low, "approx", what)
    _bn_check(out, pieces, what, exact=True)


def _bn_tensor_chain(cls, shape):
    """BatchNorm of ``cls`` (mixed parameters, first channels) on a tensor
    of ``shape`` (batch axis left out) > Flatten > Linear (-> 3), float64,
    dyadic: the graph handler maps the flat set."""
    params = _BN1D_PARAMS["mixed"] if cls is nn.BatchNorm1d else \
        _BN2D_PARAMS["mixed"]
    bn = _bn_layer(cls, [v[:shape[0]] for v in params]).double()
    n = int(np.prod(shape))
    head = nn.Linear(n, 3).double()
    rng = np.random.default_rng(24)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(rng.integers(-4, 5, (3, n)) / 4.0))
        head.bias.copy_(torch.from_numpy(rng.integers(-4, 5, 3) / 8.0))
    return nn.Sequential(bn, nn.Flatten(), head).eval()


@pytest.mark.parametrize("operation,variant", [
    pytest.param(op, variant, id=f"{op}-{variant}")
    for op in ("BatchNorm1d", "BatchNorm2d")
    for variant in ("fused", "kernel", "tensor")])
def test_batchnorm_in_a_torch_chain_by_the_exact_method_is_the_exact_image(
        operation, variant):
    """``method='exact'`` through ``reach_pytorch_model``: Linear >
    BatchNorm1d > Linear on a Star with a binding row, and Conv > BatchNorm2d
    > Flatten > Linear on a two-variable ImageStar polygon, the BatchNorm
    folded or kept; and BatchNorm1d on a (1, 2, 2) and BatchNorm2d on a
    (1, 2, 1, 2) tensor > Flatten > Linear on a Star with a binding row (the
    graph handler, ``input_shape``). The exact method claims the image: the
    result holds every vertex of the exact image (rationals) with zero slack
    and has no vertex outside it. The other cells reach the same code by the
    approx method, which the kernels do not read, and check the handler for
    containment only; this one catches an exact walk that routes the
    BatchNorm or its fold differently, and a result that is not the
    image."""
    if variant == "tensor":
        shape = (2, 2) if operation == "BatchNorm1d" else (2, 1, 2)
        model = _bn_tensor_chain(getattr(nn, operation), shape)
        M, offset = _bn_affine_of(model, shape)
        lo, hi = flat_box(4, seed=6)
        star, _ = constrained_star(lo, hi, seed=6)
        pieces = _bn_input_pieces(star)
        options = {"input_shape": (1,) + shape}
    elif operation == "BatchNorm1d":
        model = _bn_flat_chain(variant)
        M, offset = _bn_affine_of(model, (4,))
        lo, hi = flat_box(4, seed=5)
        star, _ = constrained_star(lo, hi, seed=5)
        pieces = _bn_input_pieces(star)
        options = {}
    else:
        model = _bn_image_chain(variant)
        M, offset = _bn_affine_of(model, _BN_CHW)
        star, centre, gen = _bn_low_image(_BN_CHW, "ImageStar", seed=4)
        pieces = rational.affine(rational.polytope(_BN_POLY_A, _BN_POLY_B),
                                 gen.tolist(), centre.tolist())
        options = {}
    pieces = rational.affine(pieces, M.tolist(), offset.tolist())
    what = f"{operation} in a torch chain ({variant}), exact method"
    out = reach_model(model, star, "exact", what, **options)
    _bn_check(out, pieces, what, exact=True)


@pytest.mark.parametrize("kind,method", [("ImageStar", "approx"),
                                         ("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
def test_batchnorm_in_an_onnx_chain(tmp_path, kind, method):
    """Conv > BatchNormalization (eps 1/4, a negative scale) > Flatten >
    Gemm loaded from ONNX, where the model entry folds the BatchNorm into the
    Conv along the loaded graph (not a Sequential): every onnxruntime output
    of a sample point is in the result, by LP on its rows. Catches a wrong
    fold or a BatchNorm dropped on the graph route."""
    C, H, W = _BN_CHW
    model = _bn_image_chain("fused")
    conv, bn, head = model[0], model[1], model[3]
    f32 = lambda t: t.detach().numpy().astype(np.float32)   # noqa: E731
    nodes = [helper.make_node("Conv", ["X", "w", "b"], ["c"],
                              kernel_shape=[1, 2]),
             helper.make_node("BatchNormalization", ["c", "s", "o", "m", "v"],
                              ["n"], epsilon=_BN_EPS),
             helper.make_node("Flatten", ["n"], ["f"], axis=1),
             helper.make_node("Gemm", ["f", "g", "h"], ["Y"], transB=1)]
    inits = {"w": f32(conv.weight), "b": f32(conv.bias), "s": f32(bn.weight),
             "o": f32(bn.bias), "m": f32(bn.running_mean),
             "v": f32(bn.running_var), "g": f32(head.weight),
             "h": f32(head.bias)}
    path = save_onnx(tmp_path, nodes, [1, C, H, W], [1, 3], inits,
                     name="conv_bn_gemm")
    lo, hi = pixel_box(C, H, W, seed=2)
    what = f"Conv > BatchNormalization > Flatten > Gemm, {method}, {kind}"
    out = reach_model(load_onnx(path), image_set(kind, lo, hi), method, what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), [1, C, H, W]),
                   what)


@pytest.mark.parametrize("kind", ["Star", "Zono"])
def test_batchnorm_after_a_gemm_with_a_second_consumer(tmp_path, kind):
    """Gemm > BatchNormalization, and the Gemm output also feeds the Add that
    joins it with the BatchNorm output (a residual), loaded from ONNX. The
    model entry may fold the BatchNorm into the Gemm only when the BatchNorm
    is the Gemm output's single consumer; folded anyway, the other consumer
    reads the normalised value. Every onnxruntime output of a
    sample point is in the result, by LP on its rows. Catches a graph fusion
    that drops that check (a Box is never folded, so Star and Zono)."""
    inits = {"W": np.array([[1.0, 0.5, 0.0], [-0.5, 1.0, 0.25],
                            [0.0, -1.0, 1.0]], dtype=np.float32),
             "c": np.array([0.25, -0.5, 0.0], dtype=np.float32)}
    for key, value in zip("gbmv", _BN1D_PARAMS["mixed"]):
        inits[key] = np.array(value[:3], dtype=np.float32)
    nodes = [helper.make_node("Gemm", ["X", "W", "c"], ["T"], transB=1),
             helper.make_node("BatchNormalization", ["T", "g", "b", "m", "v"],
                              ["N"], epsilon=_BN_EPS),
             helper.make_node("Add", ["N", "T"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 3], [1, 3], inits,
                     name="gemm_bn_add")
    lo, hi = flat_box(3, seed=2)
    what = f"Gemm > BatchNormalization > Add with the Gemm, {kind}"
    out = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx", what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), [1, 3]), what)


# --------------------------------------------------------------------------- #
# Refusal: a BatchNorm without running statistics                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,route", [
    pytest.param(kind, "reach_layer", id=kind)
    for kind in ("Box", "Star", "Zono", "Hexatope", "Octatope", "ImageStar",
                 "ImageZono")] + [
    pytest.param("Star", "after-a-linear", id="Star-after-a-linear-model"),
    pytest.param("Box", "tensor-shape", id="Box-tensor-shape-model")])
def test_batchnorm_without_running_statistics_is_refused(kind, route):
    """With track_running_stats=False a BatchNorm normalizes each batch by
    its own statistics even in evaluation mode, so it is not an affine map:
    reach must raise ``NotImplementedError``, never answer with mean 0 and
    variance 1. Through ``reach_layer`` on every set type, and through the
    model entry behind a Linear (where it must not be folded into the
    Linear) and on a flat Box standing for a (1, 2, 3) tensor (the graph
    handler)."""
    if route == "after-a-linear":
        model = nn.Sequential(nn.Linear(4, 4).double(), nn.BatchNorm1d(
            4, track_running_stats=False).double()).eval()
        lo, hi = flat_box(4)
        with pytest.raises(NotImplementedError):
            reach_model(model, flat_set(kind, lo, hi), "approx", route)
        return
    if route == "tensor-shape":
        model = nn.Sequential(nn.BatchNorm1d(2, track_running_stats=False),
                              nn.Flatten()).eval()
        lo, hi = flat_box(6)
        with pytest.raises(NotImplementedError):
            reach_model(model, flat_set(kind, lo, hi), "approx", route,
                        input_shape=(1, 2, 3))
        return
    if kind.startswith("Image"):
        bn = nn.BatchNorm2d(2, track_running_stats=False).eval()
        lo, hi = pixel_box(*_BN_CHW)
        s = image_set(kind, lo, hi)
    else:
        bn = nn.BatchNorm1d(4, track_running_stats=False).eval()
        lo, hi = flat_box(4)
        s = flat_set(kind, lo, hi)
    with pytest.raises(NotImplementedError):
        reach_layer(bn, [s], "approx")


# --------------------------------------------------------------------------- #
# BatchNorm on a flat set that stands for an (N, C, ...) tensor                #
# --------------------------------------------------------------------------- #
# Through the model entry with an input shape, a BatchNorm on a flat set of
# C x (other axes) entries expands its per-channel scale, shift and rounding
# margin over the other axes (n2v/nn/reach.py, _handle_batchnorm_flat). The
# flat set is in the tensor's row-major order, channel first.
_BN_TENSOR_SHAPES = {
    "bn1d-3x2": (nn.BatchNorm1d, (3, 2),
                 ([1.5, -0.75, 2.0], [0.25, 0.0, -1.0], [0.5, -1.0, 0.25],
                  [0.0, 0.75, 3.75])),
    "bn2d-2x1x3": (nn.BatchNorm2d, (2, 1, 3), _BN2D_PARAMS["mixed"]),
}
# (six entries each: the exact check enumerates the 2^6 corners of the box)


def _bn_tensor_cells():
    """Every set type on one box; several sets at once for a Star and a Box
    (the handler's loop: the Box branch and the branch every other set type
    shares)."""
    return [pytest.param(name, kind, variant, id=f"{name}-{kind}-{variant}")
            for name in sorted(_BN_TENSOR_SHAPES) for kind in _BN_FLAT_KINDS
            for variant in ("box", "split")
            if variant == "box" or kind in ("Star", "Box")]


@pytest.mark.parametrize("name,kind,variant", _bn_tensor_cells())
def test_batchnorm_on_a_flat_set_with_a_tensor_shape(name, kind, variant):
    """BatchNorm1d on a (1, 3, 2) tensor and BatchNorm2d on a (1, 2, 1, 3)
    tensor, given as a flat set of every type with ``input_shape`` through
    ``reach_pytorch_model`` (the graph handler), on a box eight times the
    usual one (bounds well beyond +-1, intervals wider than 2): every torch
    output of a sample point is in a returned set (LP on the rows), and for
    one set the exact image of the box (rationals, the layer's map read from
    torch on the unit vectors) is inside the result with zero slack. Catches
    the per-channel parameters expanded along the wrong axis (channel taken
    for the last axis), the shift not expanded with the scale, a result that
    reads a predicate variable in place of an entry, and later input sets
    not answered."""
    cls, shape, params = _BN_TENSOR_SHAPES[name]
    model = nn.Sequential(_bn_layer(cls, params).double(), nn.Flatten()).eval()
    n = int(np.prod(shape))
    lo, hi = flat_box(n, seed=7)
    lo, hi = 8.0 * lo, 8.0 * hi
    sets = (split_cover(kind, lo, hi, 3) if variant == "split"
            else [flat_set(kind, lo, hi)])
    what = f"{name} on a flat {kind} ({variant})"
    out = reach_model(model, sets, "approx", what, input_shape=(1,) + shape)
    assert_covered(out, replay_torch(model, samples(lo, hi), (1,) + shape),
                   what)
    if variant == "split":
        return
    M, offset = _bn_affine_of(model, shape)
    pieces = rational.affine(_bn_box_piece(lo.tolist(), hi.tolist()),
                             M.tolist(), offset.tolist())
    # containment only: the other direction of the same maps is checked on
    # the kernels by test_batchnorm1d_on_flat_sets_holds_the_exact_image, and
    # here would enumerate a six-variable polytope stored with duplicate rows
    _bn_check(out, pieces, what, exact=False)


def test_batchnorm1d_box_on_a_channel_by_length_tensor_holds_the_exact_endpoint_images():
    """The Box rounding cell of
    ``test_batchnorm1d_box_holds_the_exact_endpoint_images`` through the
    model entry on a (1, C, 2) tensor: every feature's interval appears at
    both positions, and each output bound is compared in fractions with the
    exact endpoint images, no slack. Catches the rounding margin of the
    expanded map dropped, taken from the expanded shift (where beta and
    scale * mean cancel), or charged for the wrong endpoint or channel."""
    bn = _bn_box_rounding_layer("affine")
    lo1, hi1 = _bn_box_rounding_input("affine")
    C = lo1.size
    lo, hi = np.repeat(lo1, 2), np.repeat(hi1, 2)      # (C, 2) row-major
    model = nn.Sequential(bn, nn.Flatten()).eval()
    what = "BatchNorm1d on a (1, C, 2) Box"
    out = reach_model(model, Box(lo.reshape(-1, 1), hi.reshape(-1, 1)),
                      "approx", what, input_shape=(1, C, 2))
    if len(out) != 1 or not isinstance(out[0], Box):
        pytest.fail(f"{what}: expected one Box, got "
                    f"{[type(s).__name__ for s in out]}")
    s, t = _bn_exact(bn)
    lb = np.asarray(out[0].lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(out[0].ub, dtype=np.float64).reshape(-1)
    worst = []
    for i in range(lo.size):
        c = i // 2
        images = [s[c] * Fraction(float(x)) + t[c] for x in (lo[i], hi[i])]
        if Fraction(float(lb[i])) > min(images):
            worst.append((i, "lower"))
        if Fraction(float(ub[i])) < max(images):
            worst.append((i, "upper"))
    assert not worst, (f"{what}: bounds inside the exact image (entry, "
                       f"side): {worst}")


def _bn_mixed_list_crash(kernel, exc, why):
    return known_bug(
        23, f"{kernel} handed a list whose second set is of another family "
            f"raises {exc.__name__} ({why}) instead of the refusal of an "
            f"unsupported combination (NotImplementedError) or of an input "
            f"it cannot read (ValueError)", raises=exc)


@pytest.mark.parametrize("kinds", [
    pytest.param(("Star", "Zono"), id="star-then-zono",
                 marks=_bn_mixed_list_crash(
                     "batchnorm_star", TypeError,
                     "'batchnorm_star expects Star or ImageStar'")),
    pytest.param(("Zono", "Star"), id="zono-then-star",
                 marks=_bn_mixed_list_crash(
                     "batchnorm_zono", TypeError,
                     "'batchnorm_zono expects Zono or ImageZono'")),
    pytest.param(("Box", "Star"), id="box-then-star",
                 marks=_bn_mixed_list_crash(
                     "batchnorm_box", AttributeError,
                     "the Box rule reads the Star's lb")),
    pytest.param(("Hexatope", "Star"), id="hexatope-then-star")])
def test_batchnorm_on_a_list_mixing_set_types_refuses_or_holds_every_output(
        kinds):
    """Two sets of different families in one ``reach_layer`` call (only a
    direct call can build such a list). The dispatcher routes by the first
    set; the kernel must refuse the other (``ValueError`` for an input it
    cannot read, ``NotImplementedError`` for an unsupported combination) or
    answer with sets that hold the torch outputs of both (LP on the rows;
    the second box is moved by 4, so a result reused from the first set
    misses them). Catches a kernel that maps the second set as if it were of
    the first family, skips it, or repeats the first result. The Star, Zono
    and Box rules crash today (one marked case each); the Hexatope lane maps
    each set by its own affine map and is the passing neighbour."""
    bn = _bn_layer(nn.BatchNorm1d, _BN1D_PARAMS["mixed"])
    lo, hi = flat_box(5, seed=3)
    try:
        out = reach_layer(bn, [flat_set(kinds[0], lo, hi),
                               flat_set(kinds[1], lo + 4, hi + 4)], "approx")
    except (NotImplementedError, ValueError):
        return
    points = samples(lo, hi, 64) + samples(lo + 4, hi + 4, 64)
    assert_covered(out, replay_torch(bn, points, (1, 5)),
                   f"BatchNorm1d on a list of a {kinds[0]} and a {kinds[1]}")


# --------------------------------------------------------------------------- #
# After an exact split inside the model: several sets reach the layer          #
# --------------------------------------------------------------------------- #
def _bn_after_split_box(chw=(2, 2, 4), seed=12):
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
def test_batchnorm2d_after_an_exact_relu_split_in_a_model(kind, method):
    """ReLU > BatchNorm2d > Flatten > Linear through ``reach_pytorch_model``
    on a (2, 2, 4) box with three pixels straddling zero: by the exact method
    the ReLU hands up to eight stars to the layer in one call. Every torch
    output of a sample point is in a returned set, by LP on the rows.
    Catches only the first or the last of several sets answered inside the
    model, and a result of one set reused for another."""
    chw = (2, 2, 4)
    layer = _bn_layer(nn.BatchNorm2d, _BN2D_PARAMS["mixed"])
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
    lo, hi = _bn_after_split_box(chw)
    what = f"ReLU > BatchNorm2d > Flatten > Linear, {method}, {kind}"
    out = reach_model(model, image_set(kind, lo, hi), method, what)
    if method == "exact" and len(out) < 2:
        pytest.fail(f"test setup: the exact ReLU gave {len(out)} set(s)")
    assert_covered(out, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what)


# gamma, beta, running mean, running var per feature: means of the size of
# 1e4 with digits float32 does not hold (its spacing there is about 1e-3).
_BN_FLOAT64_ONLY = ([1.0000000123, -0.5, 2.0], [0.1, 0.2, -0.3],
                    [12345.678901234, -9876.54321012, 4321.0123456789],
                    [1.0, 4.0, 0.25])


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box", "ImageStar",
                                  "ImageZono"])
def test_batchnorm_float64_layer_with_a_mean_that_float32_cannot_hold(kind):
    """A float64 BatchNorm whose running means are near 1e4 and are not
    float32 numbers, on a box of half-width 1 around the means: the layer a
    user gets from ``.double()`` training or from a float64 file. Every
    float64 torch output of a sample point (corners among them) is in the
    result, by LP on the rows. Catches a parameter read through float32 on
    its way to the folded scale and shift: the mean moves by up to 5e-4 and
    the set with it, so the outputs at one face of the box fall outside.
    Every other float64 layer of this file is a float32 layer converted, so
    its parameters survive that read unchanged."""
    image = kind.startswith("Image")
    params = [np.array(v) for v in _BN_FLOAT64_ONLY]
    cls = nn.BatchNorm2d if image else nn.BatchNorm1d
    bn = cls(3, eps=0.0).double()
    with torch.no_grad():
        for target, value in zip((bn.weight, bn.bias, bn.running_mean,
                                  bn.running_var), params):
            target.copy_(torch.from_numpy(value))
    bn.eval()
    mean = params[2]
    if image:
        chw = (3, 2, 1)
        centre = np.broadcast_to(mean.reshape(3, 1, 1), chw)
        lo, hi = centre - 1.0, centre + 1.0
        input_set, shape = image_set(kind, lo, hi), (1,) + chw
    else:
        lo, hi = mean - 1.0, mean + 1.0
        input_set, shape = flat_set(kind, lo, hi), (1, 3)
    what = f"float64 {cls.__name__} with means near 1e4 on a {kind}"
    out = reach_one_layer(bn, input_set, "approx", what)
    assert_covered(out, replay_torch(bn, samples(lo, hi), shape), what,
                   out_chw=chw if image else None)
