"""
Soundness tests for LeakyReLU layer reachability.

Every set type and method: the exact image in fractions
(tests/_oracles/rational.py) and float64 or onnxruntime replay checked by an LP
on the returned set's own rows; one layer through ``reach_layer`` and Linear,
LeakyReLU, Linear (or Conv, LeakyRelu, Conv) through the model entry. The
cells are described in the banner below.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.dispatcher import reach_layer


# =========================================================================== #
# Soundness cells                                                              #
#                                                                              #
# Reference: the exact image in rationals (tests/_oracles/rational.py), read   #
# from the input set's stored arrays, never from n2v. Two shapes of input:     #
#                                                                              #
# - a set that is a product of intervals (Box, Star/Zono/ImageStar/ImageZono  #
#   from bounds, and the parts of a split box). LeakyReLU acts coordinate by   #
#   coordinate, so the exact image is the product of the coordinates' images, #
#   each computed by the oracle in one dimension; a piece of the image is one #
#   choice of a piece per coordinate, and its vertices are the products of    #
#   the coordinates' end values;                                               #
# - a set whose coordinates share at most three predicate variables (a Star   #
#   with a binding row that is not a box row, two halves of a split): the     #
#   oracle's polytope pipeline on the predicate space (its vertex enumeration #
#   by row subsets costs 0.2 s with three variables and 30 s with five, which #
#   is why no cell here has more).                                             #
#                                                                              #
# Soundness: every piece of the exact image has all its vertices in one        #
# returned set, decided in fractions with no slack. Exactness                  #
# (method='exact'): every vertex of every returned set has a preimage. Sample  #
# points replayed in float64 (torch) or float32 (onnxruntime; exact on these   #
# dyadic data) are checked by one LP on the set's own rows as well. A coupled  #
# Zono is checked by that LP alone (see its cell). Layout: a network tensor    #
# is (C, H, W); an image set stores (H, W, C).                                 #
# =========================================================================== #
from fractions import Fraction
from itertools import product as _cartesian

import pytest
from onnx import helper

from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests.cell_support import (assert_covered, flat_set, image_set,
                                image_split_cover, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx, split_cover)
from tests.known_bugs import known_bug
from torch.nn import functional

# Dyadic slopes: 0 (ReLU), a slope in (0, 1), 1 (identity), a slope above 1
# (the concave branch of the Star and Zono rules) and a negative one (the
# function is then not monotone).
_SLOPES = [0.0, 0.25, 1.0, 2.0, -0.5]
# Coordinates: crossing, lower bound exactly 0, upper bound exactly 0, zero
# width, strictly negative, crossing again. The boundary values 0 are where
# the kernels compare (``ub <= 0``, ``lb >= 0``, ``lb < 0 < ub``). A crossing
# interval is a power of two wide, so the Zono band's slope
# (u - slope l) / (u - l) is dyadic and the cell does not hinge on its last
# bit (that rounding is marked by
# test_leakyrelu_zono_holds_the_exact_image_of_its_stored_data).
_LO = np.array([-0.75, 0.0, -1.25, 0.375, -1.5, -0.5])
_HI = np.array([0.25, 0.75, 0.0, 0.375, -0.25, 1.5])


def _leaky(slope):
    return lambda pieces: R.leaky_relu(pieces, slope)


# --------------------------------------------------------------------------- #
# The exact image of a product of intervals                                    #
# --------------------------------------------------------------------------- #
def _box_of(input_set):
    """``(lo, hi)``: the exact interval of every entry of a set that is a
    product of intervals, in the order the set stores its entries, read from
    its stored arrays (``rational.star``). Raises ``ValueError`` when the set
    is not such a product (a row or a predicate variable couples entries)."""
    c, G, C, d, lb, ub = R.star(input_set)
    low, high = list(lb), list(ub)
    for row, rhs in zip(C, d):
        used = [j for j, v in enumerate(row) if v != 0]
        if not used:
            if rhs < 0:
                raise ValueError("the input set is empty")
            continue
        if len(used) > 1:
            raise ValueError("a constraint row couples predicate variables")
        j = used[0]
        bound = rhs / row[j]
        if row[j] > 0:
            high[j] = bound if high[j] is None else min(high[j], bound)
        else:
            low[j] = bound if low[j] is None else max(low[j], bound)
    owner = {}
    lo, hi = [], []
    for i, (ci, g) in enumerate(zip(c, G)):
        used = [j for j, v in enumerate(g) if v != 0]
        if not used:
            lo.append(ci)
            hi.append(ci)
            continue
        if len(used) > 1 or used[0] in owner:
            raise ValueError("a predicate variable drives two entries")
        j = used[0]
        owner[j] = i
        if low[j] is None or high[j] is None:
            raise ValueError("an entry is unbounded")
        ends = (ci + g[j] * low[j], ci + g[j] * high[j])
        lo.append(min(ends))
        hi.append(max(ends))
    return lo, hi


def _coordinate_images(op, lo, hi):
    """The exact image of each interval ``[lo_i, hi_i]`` under the scalar
    operation, as the oracle's one-dimensional pieces."""
    return [op(R.box([l], [h])) for l, h in zip(lo, hi)]


def _assert_holds_product_image(op, input_sets, sets, what, pieces=True,
                                order=None):
    """Every input set is a product of intervals; its exact image is the
    product of the coordinates' images. With ``pieces`` each product of one
    piece per coordinate (a piece of the exact image) must have all its
    vertices in one returned set; without, the hull of the image (the box of
    the coordinates' end values) must lie in one returned set, which is what
    a method returning one convex set per input promises. ``order`` maps a
    vertex in the input's stored order to the order the results store."""
    assert sets, f"{what}: no set returned"
    holders = [R.Membership(R.star(s)) for s in sets]
    for k, s in enumerate(input_sets):
        images = _coordinate_images(op, *_box_of(s))
        if pieces:
            groups = _cartesian(*[[sorted({y[0] for y in p.image()})
                                   for p in coord] for coord in images])
        else:
            hull = [sorted({y[0] for p in coord for y in p.image()})
                    for coord in images]
            groups = [[[v[0], v[-1]] if len(v) > 1 else v for v in hull]]
        for group in groups:
            corners = [tuple(y) for y in _cartesian(*group)]
            if order is not None:
                corners = [tuple(y[i] for i in order) for y in corners]
            if any(all(h(y) for y in corners) for h in holders):
                continue
            outside = [y for y in corners if not any(h(y) for h in holders)]
            shown = ([float(v) for v in outside[0]] if outside else
                     "none; its vertices are spread over several sets")
            raise AssertionError(
                f"{what}: input set {k}: a piece of the exact image is in no "
                f"one returned set ({len(sets)} sets); {len(outside)} of its "
                f"{len(corners)} vertices are in none, first {shown}")


def _set_corners(computed):
    """The vertices of a computed set whose rows each bound one predicate
    variable (its predicate polytope is a box), else ``None``."""
    c, G, C, d, lb, ub = computed
    low, high = list(lb), list(ub)
    for row, rhs in zip(C, d):
        used = [j for j, v in enumerate(row) if v != 0]
        if len(used) > 1:
            return None
        if not used:
            if rhs < 0:
                return []
            continue
        j = used[0]
        bound = rhs / row[j]
        if row[j] > 0:
            high[j] = bound if high[j] is None else min(high[j], bound)
        else:
            low[j] = bound if low[j] is None else max(low[j], bound)
    if any(v is None for v in low + high):
        return None
    if any(a > b for a, b in zip(low, high)):
        return []
    points = set()
    for a in _cartesian(*[sorted({x, y}) for x, y in zip(low, high)]):
        points.add(tuple(R.dot(g, a) + ci for g, ci in zip(G, c)))
    return sorted(points)


def _assert_product_exact(op, input_sets, sets, what):
    """method='exact' on products of intervals: every vertex of every
    returned set is in the closure of the image of some input set, which is
    the product of the closures of the coordinates' images."""
    images = [_coordinate_images(op, *_box_of(s)) for s in input_sets]
    for k, s in enumerate(sets):
        computed = R.star(s)
        corners = _set_corners(computed)
        if corners is None:
            corners = R.set_vertices(computed)
        for y in corners:
            if not any(all(R.image_contains(coord, [v])
                           for coord, v in zip(image, y))
                       for image in images):
                raise AssertionError(
                    f"{what}: returned set {k} has the vertex "
                    f"{[float(v) for v in y]}, which no input point reaches")


# --------------------------------------------------------------------------- #
# The exact image of a set on a few predicate variables                        #
# --------------------------------------------------------------------------- #
def _input_pieces(input_set):
    """The identity on ``input_set`` as rational pieces: its predicate
    polytope (rows and bounds, read exactly from the stored arrays) mapped
    through its centre and generators. No n2v method is called."""
    c, G, C, d, lb, ub = R.star(input_set)
    n = len(lb)
    A, b = [list(row) for row in C], list(d)
    for j in range(n):
        unit = [Fraction(int(i == j)) for i in range(n)]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    return R.affine(R.polytope(A, b), G, c)


def _assert_holds_exact_image(pieces, sets, what):
    """Every piece of the exact image has all its vertices in one returned
    set, in fractions (a convex set that holds the vertices holds the
    piece)."""
    assert sets, f"{what}: no set returned"
    holders = [R.Membership(R.star(s)) for s in sets]
    bad = []
    for i, piece in enumerate(pieces):
        image = piece.image()
        if any(all(h(y) for y in image) for h in holders):
            continue
        outside = [y for y in image if not any(h(y) for h in holders)]
        bad.append((i, len(image), outside))
    if bad:
        i, total, outside = bad[0]
        shown = ([float(v) for v in outside[0]] if outside
                 else "none; its vertices are spread over several sets")
        raise AssertionError(
            f"{what}: {len(bad)} of {len(pieces)} pieces of the exact image "
            f"are not held by any one returned set ({len(sets)} sets); piece "
            f"{i}: {len(outside)} of its {total} vertices in no set, first "
            f"{shown}")


def _assert_no_point_without_preimage(pieces, sets, what):
    """method='exact': every vertex of every returned set is a value of the
    network at some input (in the closure of the exact image). Stops at the
    first vertex that is not."""
    for k, s in enumerate(sets):
        for y in R.set_vertices(R.star(s)):
            if not R.image_contains(pieces, y):
                raise AssertionError(
                    f"{what}: returned set {k} of {len(sets)} has the vertex "
                    f"{[float(v) for v in y]}, which no input reaches")


# Five entries on three predicate variables a in [-1, 1]^3: entry 0 crosses 0
# on [-3/4, 5/4], entry 1 crosses 0 on [-3/2, 1/2], entry 2 has interval
# lower bound exactly 0, entry 3 has interval upper bound exactly 0, entry 4
# is the constant 0.375. Crossing ranges are two wide (a dyadic band slope).
_ROW_C = np.array([0.25, -0.5, 0.75, -1.0, 0.375])
_ROW_G = np.array([[0.5, 0.25, 0.25], [0.5, -0.25, 0.25], [0.25, 0.0, 0.5],
                   [0.25, 0.5, 0.25], [0.0, 0.0, 0.0]])
# The binding row a0 + a1 - a2/2 <= 5/4: it cuts off the corner (1, 1, -1)
# and lowers the true maximum of entries 0 and 3, so the interval bounds of
# the star are not its true ranges.
_ROW = np.array([1.0, 1.0, -0.5])
_ROW_RHS = 1.25


def _grid(k, steps=5):
    values = np.linspace(-1.0, 1.0, steps)
    return [np.array(a) for a in _cartesian(*[values] * k)]


def _row_star(extra=None):
    """``(star, points)``: the five entries above as a ``Star`` with the box
    rows, the binding row and ``extra`` rows ``(row, rhs)``; ``points`` are
    the member points of the half grid of the predicate box, exact."""
    rows = [(_ROW, _ROW_RHS)] + list(extra or [])
    C = np.vstack([np.eye(3), -np.eye(3)] + [r for r, _ in rows])
    d = np.concatenate([np.ones(6), [v for _, v in rows]]).reshape(-1, 1)
    star = Star(np.hstack([_ROW_C[:, None], _ROW_G]), C, d,
                -np.ones((3, 1)), np.ones((3, 1)))
    points = [_ROW_C + _ROW_G @ a for a in _grid(3)
              if all(r @ a <= v for r, v in rows)]
    return star, points


def _coupled_star():
    """Five coordinates on two predicate variables, so the coordinates are
    related: a split of one neuron decides others in a child (x3 is stably
    negative where x0 <= 0, x4 stably positive where x0 >= 0), and a row
    that relates an output to the wrong variable moves outputs. No range
    ends exactly at 0, here or in a child of a split."""
    c = np.array([0.25, -0.25, 0.625, -0.75, 0.75])
    G = np.array([[0.5, 0.25], [0.75, -0.5], [0.25, 0.25], [0.5, 0.5],
                  [0.5, 0.375]])
    k = G.shape[1]
    star = Star(np.hstack([c[:, None], G]),
                np.vstack([np.eye(k), -np.eye(k)]), np.ones((2 * k, 1)),
                -np.ones((k, 1)), np.ones((k, 1)))
    return star, [c + G @ a for a in _grid(2, 9)]


def _coupled_family(name):
    """``(input sets, member points)`` of a family whose entries share
    predicate variables."""
    if name == "coupled Star":
        star, points = _coupled_star()
        return [star], points
    if name == "Star with a binding row":
        star, points = _row_star()
        return [star], points
    if name == "two halves of a split":
        # The star cut at entry 0 = 0, as an earlier exact split leaves it:
        # two input sets at once, each with one more binding row.
        g, c = _ROW_G[0], _ROW_C[0]
        low, p1 = _row_star([(g, -c)])
        high, p2 = _row_star([(-g, c)])
        return [low, high], p1 + p2
    if name == "Star with no predicate bounds":
        # The coupled Star with its box as constraint rows only: the kernels
        # take their ranges from padded LP enclosures, the approx rows get no
        # outward slack and the new variables no predicate bounds.
        star, points = _coupled_star()
        return [Star(star.V, star.C, star.d, None, None)], points
    if name == "coupled Zono":
        zono = Zono(_ROW_C.reshape(-1, 1), _ROW_G.copy())
        return [zono], [_ROW_C + _ROW_G @ a for a in _grid(3)]
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# One layer through reach_layer                                                 #
# --------------------------------------------------------------------------- #
# The exact image is the verdict of every cell below; the replayed samples
# are the uniform extra check, 32 per cell to keep the file's run time down.
_N = 32
# Box and Zono: every slope on one set, and the loop over three sets at one
# slope (the loop does not depend on the slope).
_BOX_CELLS = (
    [pytest.param(k, "approx", p, g, id=f"{k}-approx-{p}set-{g}")
     for k in ("Box", "Star", "Zono") for p in (1, 3) for g in _SLOPES
     if k == "Star" or p == 1 or g == 0.25]
    + [pytest.param("Star", "exact", p, g, id=f"Star-exact-{p}set-{g}")
       for p, g in ((1, 0.25), (1, 2.0), (1, -0.5), (3, 0.25))])


@pytest.mark.parametrize("kind,method,parts,slope", _BOX_CELLS)
def test_leakyrelu_on_a_product_of_intervals_holds_the_exact_image(
        kind, method, parts, slope):
    """LeakyReLU through ``reach_layer`` on a Box, a Star and a Zono built
    from bounds, one set and three parts of a split box at once, with
    coordinates on every side of 0 and on it (crossing, lower bound exactly
    0, upper bound exactly 0, zero width, negative). Exact image in
    fractions, both directions under ``method='exact'``, and float64
    samples by LP. Catches a bound or slope on the wrong side (range), a
    regime decided by the wrong comparison at a bound of exactly 0 or a
    point interval, a crossing neuron's band or triangle at the wrong
    height, a slope above 1 or below 0 handled as one in [0, 1], a dropped
    split piece or one given the wrong phase, and the outputs of all but one
    input set dropped (plumbing)."""
    sets = ([flat_set(kind, _LO, _HI)] if parts == 1
            else split_cover(kind, _LO, _HI, parts))
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"LeakyReLU({slope}) on {kind} ({len(sets)} sets), {method}"
    out = reach_one_layer(layer, sets, method, what)
    _assert_holds_product_image(_leaky(slope), sets, out, what,
                                pieces=method == "exact")
    if method == "exact":
        _assert_product_exact(_leaky(slope), sets, out, what)
    assert_covered(out, replay_torch(layer, samples(_LO, _HI, n=_N),
                                     (1, _LO.size)), what)


_COUPLED_SLOPES = {"coupled Star": (0.0, 0.25, 2.0, -0.5),
                   "Star with a binding row": (0.0, 0.25, 2.0, -0.5),
                   "two halves of a split": (0.25, 2.0),
                   "Star with no predicate bounds": (0.25, 2.0),
                   "coupled Zono": (0.0, 0.25, 2.0, -0.5)}
_COUPLED_CELLS = [
    pytest.param(f, m, g, id=f"{f}-{m}-{g}")
    for f in ("coupled Star", "Star with a binding row",
              "two halves of a split", "Star with no predicate bounds",
              "coupled Zono")
    for m in (("approx",) if f == "coupled Zono" else ("approx", "exact"))
    for g in _COUPLED_SLOPES[f]]


@pytest.mark.parametrize("family,method,slope", _COUPLED_CELLS)
def test_leakyrelu_on_a_coupled_set_holds_the_exact_image(family, method,
                                                          slope):
    """LeakyReLU on sets whose entries share predicate variables: a coupled
    Star, a Star with a binding row that is not a box row (its interval
    bounds are not its true ranges, so the LP ranges decide), the two halves
    of that Star cut at entry 0 = 0 (several input sets, each the result of
    an earlier split), the coupled Star with its box as rows and no
    predicate bounds (the branch of the Star kernels without a predicate box:
    LP enclosures, rows without outward slack, new variables bounded by rows
    only) and a coupled Zono. The relaxation rows bind here, so a row that
    relates an output to the wrong variable, with the wrong sign or at the
    wrong height fails, as does a neuron that the LP shows stable inside a
    split child handled as crossing, and the outputs of one input set
    dropped. For the Zono it is the regression test of issue 16 (the band
    without its error generator collapses onto the secant and leaves out the
    outputs of correlated entries; the reverse of that fix fails all four
    coupled Zono items).

    The Stars are checked against the exact image in fractions. The coupled
    Zono is checked by LP at ``LP_TOL`` only: its bounds come padded by
    about 1e-15, the band slope is then rounded to nearest and the exact
    image is outside by about 2e-15, the rounding marked by
    ``test_leakyrelu_zono_holds_the_exact_image_of_its_stored_data``."""
    sets, points = _coupled_family(family)
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"LeakyReLU({slope}) on {family}, {method}"
    out = reach_one_layer(layer, sets, method, what)
    if family != "coupled Zono":
        pieces = [p for s in sets
                  for p in R.leaky_relu(_input_pieces(s), slope)]
        _assert_holds_exact_image(pieces, out, what)
    assert_covered(out, replay_torch(layer, points[::3], (1, 5)), what)


# Every entry decided by its interval bound: positive, negative, lower bound
# exactly 0, upper bound exactly 0, a point.
_DECIDED_LO = np.array([0.25, -1.5, 0.0, -1.25, 0.375])
_DECIDED_HI = np.array([1.0, -0.25, 0.75, 0.0, 0.375])


def _decided_by_lp(side):
    """A Star on two predicate variables whose two entries both straddle 0
    by their interval bounds, while a binding row decides them: "inactive"
    (a0 + a1 <= -1/2 keeps both below 0) or "active" (a0 + a1 >= 1/2 keeps
    both above 0)."""
    sign = -1.0 if side == "inactive" else 1.0
    V = np.array([[0.25 * sign, 0.5, 0.25], [0.5 * sign, 0.5, 0.5]])
    C = np.vstack([np.eye(2), -np.eye(2), -sign * np.ones((1, 2))])
    d = np.array([[1.0], [1.0], [1.0], [1.0], [-0.5]])
    return Star(V, C, d, -np.ones((2, 1)), np.ones((2, 1)))


@pytest.mark.parametrize("slope", [0.25, 2.0, -0.5])
@pytest.mark.parametrize("case,method", [
    ("Star", "approx"), ("Star", "exact"), ("decided inactive by its LP range", "approx"),
    ("decided inactive by its LP range", "exact"),
    ("decided active by its LP range", "approx"),
    ("decided active by its LP range", "exact")])
def test_leakyrelu_with_no_crossing_neuron_holds_the_exact_image(
        case, method, slope):
    """LeakyReLU where no neuron ends up crossing 0: a Star from bounds
    whose entries are all decided by their bounds, and Stars
    whose entries straddle 0 by their interval bounds but are decided by
    the LP range (all inactive, or all active). These take the early
    returns of the approx kernel (no crossing neuron by the estimate; none
    left after the LP maximum; none after the LP minimum) and the decided
    branches of the exact split step. Exact image in fractions, both
    directions under 'exact'. Catches an early return that hands back the
    input instead of the star with its inactive rows scaled, and a neuron
    the LP decides given the other phase."""
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"LeakyReLU({slope}) on {case}, {method}"
    if case == "Star":
        sets = [flat_set(case, _DECIDED_LO, _DECIDED_HI)]
        out = reach_one_layer(layer, sets, method, what)
        _assert_holds_product_image(_leaky(slope), sets, out, what,
                                    pieces=method == "exact")
        if method == "exact":
            _assert_product_exact(_leaky(slope), sets, out, what)
        return
    star = _decided_by_lp(case.split()[1])
    out = reach_one_layer(layer, [star], method, what)
    pieces = R.leaky_relu(_input_pieces(star), slope)
    _assert_holds_exact_image(pieces, out, what)
    if method == "exact":
        _assert_no_point_without_preimage(pieces, out, what)


_BAND_NOT_EXACT = known_bug(
    18, "exact LeakyReLU answers a neuron whose LP range ends within the 1e-7 "
        "pad of 0 with one star and a band variable of width (1 - slope) "
        "times the pad; the interval bounds of a Star are padded outward, so "
        "every neuron whose true range ends exactly at 0 (behind a binding "
        "row, or in a child of an earlier split) takes that path, and the "
        "exact method returns points no input reaches")


@pytest.mark.parametrize("family,slope", [
    pytest.param("coupled Star", g, id=f"coupled Star-{g}")
    for g in (0.0, 0.25, 2.0, -0.5)] + [
    pytest.param("Star with no predicate bounds", 0.25,
                 id="Star with no predicate bounds-0.25"),
    pytest.param("Star with a binding row", 0.25, marks=_BAND_NOT_EXACT,
                 id="Star with a binding row-0.25"),
])
def test_leakyrelu_exact_on_a_coupled_star_is_exact(family, slope):
    """The other direction of the coupled cells for ``method='exact'``: every
    vertex of every returned star has a preimage (DESIGN principle 4: exact
    up to float64 rounding). The coupled Star, whose ranges end away from 0
    in every child, passes, with and without predicate bounds; the Star with
    a binding row (entry 2 has true minimum exactly 0, so its LP minimum is
    within the pad of 0 and the one-star band answer is taken) does not
    today. Its two halves (entry 0 ends at 0 in each) take the same branch
    and are checked for soundness only, in the cell above. Catches a split
    child that is wider than its activation pattern (a cut row dropped or
    with the wrong sign) and a slope applied to a neuron the child has on
    the other side."""
    sets, _ = _coupled_family(family)
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"exact LeakyReLU({slope}) on {family}"
    out = reach_one_layer(layer, sets, "exact", what)
    pieces = [p for s in sets
              for p in R.leaky_relu(_input_pieces(s), slope)]
    if R.escapes(pieces, *[R.star(s) for s in out]):
        pytest.fail(f"{what}: the result is not even sound; see "
                    f"test_leakyrelu_on_a_coupled_set_holds_the_exact_image")
    _assert_no_point_without_preimage(pieces, out, what)


@pytest.mark.parametrize("method", ["approx", "exact"])
@pytest.mark.parametrize("lp_solver", ["linprog", "SCS"])
def test_leakyrelu_with_another_lp_solver_holds_the_exact_image(
        request, lp_solver, method):
    """LeakyReLU(1/4) on the Star with a binding row, with ``lp_solver``
    named; for SCS the ``inexact_scs`` fixture makes every SCS optimum 2e-5
    inward, as SCS is at its default tolerance. A phase decided, or a
    triangle built, from an inward range leaves outputs out, so the ranges
    must come from HiGHS, padded, whatever the caller names. Exact image in
    fractions."""
    if lp_solver == "SCS":
        request.getfixturevalue("inexact_scs")
    sets, _ = _coupled_family("Star with a binding row")
    layer = nn.LeakyReLU(negative_slope=0.25)
    what = f"LeakyReLU(0.25) on a Star with a binding row, {method}, " \
           f"{lp_solver}"
    out = reach_one_layer(layer, sets, method, what, lp_solver=lp_solver)
    pieces = [p for s in sets for p in R.leaky_relu(_input_pieces(s), 0.25)]
    _assert_holds_exact_image(pieces, out, what)


# A (2, 2, 3) image with its own dyadic value or interval per pixel: three
# pixels cross 0 (on intervals a power of two wide) and one has upper bound
# exactly 0; the other eight are points of distinct values of both signs, so
# a pixel read in the wrong place moves an output while the exact method
# gives only sixteen pieces of sixteen vertices each.
_IMG_LO = np.array([[[-0.75, 0.375, 0.25], [-1.0, 0.5, -0.25]],
                    [[0.125, -1.25, -0.25], [0.75, -1.5, -0.625]]])
_IMG_HI = np.array([[[0.25, 0.375, 0.25], [-1.0, 0.5, 0.25]],
                    [[0.125, 0.0, 0.75], [0.75, -1.5, -0.625]]])
_CHW = _IMG_LO.shape


@pytest.mark.parametrize("parts", [1, 2])
@pytest.mark.parametrize("kind,method", [("ImageStar", "approx"),
                                         ("ImageStar", "exact"),
                                         ("ImageZono", "approx")])
def test_leakyrelu_on_an_image_set_keeps_the_image(kind, method, parts):
    """LeakyReLU on an ImageStar and an ImageZono of shape (2, 2, 3), one
    set and two parts at once: every result is an image set of the input's
    height, width and channels, and holds the exact image in the stored
    (height, width, channel) order. Catches a result converted back with
    height and width exchanged or left flat (layout), and the ImageStar and
    ImageZono paths of the kernels that no other cell reaches."""
    slope = 0.25
    sets = ([image_set(kind, _IMG_LO, _IMG_HI)] if parts == 1
            else image_split_cover(kind, _IMG_LO, _IMG_HI, parts))
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"LeakyReLU({slope}) on {kind} ({parts} sets), {method}"
    out = reach_one_layer(layer, sets, method, what)
    wanted = ImageStar if kind == "ImageStar" else ImageZono
    shapes = {(type(s), s.height, s.width, s.num_channels) for s in out}
    assert shapes == {(wanted, 2, 3, 2)}, (
        f"{what}: results {shapes}, expected {wanted.__name__} 2x3x2")
    _assert_holds_product_image(_leaky(slope), sets, out, what,
                                pieces=method == "exact")
    if method == "exact":
        _assert_product_exact(_leaky(slope), sets, out, what)
    assert_covered(out, replay_torch(layer, samples(_IMG_LO, _IMG_HI, n=_N),
                                     (1,) + _CHW), what, out_chw=_CHW)


# --------------------------------------------------------------------------- #
# A neuron whose range lies within the LP pad of 0 (method='exact')             #
# --------------------------------------------------------------------------- #
_E = 2.0 ** -30          # about 9.3e-10, far inside the 1e-7 pad
_D = 2.0 ** -26          # about 1.5e-8


def _thin_neuron_star(case):
    """A Star on one or two predicate variables whose entry 0 has a range
    within the LP pad of 0 while entry 1 shares a predicate variable with
    it:

    - "max in the pad": entry 0 in [-1, 1e-8] (LP maximum inside the pad);
    - "max in the pad, no predicate bounds": the same with its box as
      constraint rows only, so the band variable's bounds become rows;
    - "min in the pad": entry 0 in [-1e-8, 1];
    - "thin and split, positive": entry 0 in [2^-30, 2^-30 + 2^-25] by a
      row that its interval bounds do not see (they straddle 0); the split
      step cannot certify the side x <= 0 empty and splits;
    - "thin with a certified side, positive": entry 0 = 2^-25 (1/4 + a0)
      with the row a0 >= 1/2, so entry 0 lies in [3 2^-27, 5 2^-27] while
      its interval bound (the predicate box alone) straddles 0; the side
      x <= 0 needs a0 <= -1/4, far from the row, so the split step
      certifies it empty and keeps the identity;
    - the "negative" cases: the mirror images, below 0.
    """
    if case.startswith("max in the pad"):
        star = Star.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        star = star.affine_map(np.array([[0.5 + 0.5e-8], [0.5]]),
                               np.array([[-0.5 + 0.5e-8], [1.0]]))
        if case.endswith("no predicate bounds"):
            return Star(star.V, star.C, star.d, None, None)
        return star
    if case == "min in the pad":
        star = Star.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        return star.affine_map(np.array([[0.5 + 0.5e-8], [0.5]]),
                               np.array([[0.5 - 0.5e-8], [1.0]]))
    sign = 1.0 if case.endswith("positive") else -1.0
    if "split" in case:
        V = np.array([[sign * _E, _D, _D], [0.5, 0.25, -0.5]])
        # a0 + a1 >= 0 (positive case) or <= 0 (negative case)
        extra, rhs = -sign * np.ones((1, 2)), 0.0
    else:
        g = 2.0 ** -25
        V = np.array([[sign * g / 4, sign * g, 0.0], [0.5, 0.25, -0.5]])
        extra, rhs = np.array([[-1.0, 0.0]]), -0.5      # a0 >= 1/2
    C = np.vstack([np.eye(2), -np.eye(2), extra])
    d = np.array([[1.0], [1.0], [1.0], [1.0], [rhs]])
    return Star(V, C, d, -np.ones((2, 1)), np.ones((2, 1)))


_THIN_CASES = ["max in the pad", "max in the pad, no predicate bounds",
               "min in the pad",
               "thin and split, positive", "thin and split, negative",
               "thin with a certified side, positive",
               "thin with a certified side, negative"]


@pytest.mark.parametrize("slope", [0.0, 0.25, 2.0, -0.5])
@pytest.mark.parametrize("case", _THIN_CASES)
def test_leakyrelu_exact_on_a_neuron_within_the_lp_pad_is_sound(case, slope):
    """Exact LeakyReLU on a neuron whose LP range lies within the LP pad of 0
    (the one-star band answers, with and without predicate bounds, and the
    certified-side answers of the split step), with a second neuron on the
    same predicate variable: every piece of the exact image is in one
    returned set, in fractions. Catches the band star's row kept at scale 1
    instead of the slope (or the slope applied to the wrong phase), which
    moves the joint output by (1 - slope) |x| while each entry's interval
    still looks right, a band on the wrong side of 0 for a slope above 1 or
    with a wrong bound row when the Star has no predicate bounds, and a thin
    neuron given the wrong phase, which no float tolerance would see."""
    star = _thin_neuron_star(case)
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"exact LeakyReLU({slope}), entry 0 {case}"
    out = reach_one_layer(layer, [star], "exact", what)
    pieces = R.leaky_relu(_input_pieces(star), slope)
    _assert_holds_exact_image(pieces, out, what)


@pytest.mark.parametrize("case", [
    pytest.param(c, marks=_BAND_NOT_EXACT) if c == "max in the pad" else c
    for c in _THIN_CASES if c.startswith("thin") or c == "max in the pad"])
def test_leakyrelu_exact_on_a_neuron_within_the_lp_pad_is_exact(case):
    """The other direction of the cell above: method='exact' promises the
    exact image up to float64 rounding (DESIGN principle 4), so every
    vertex of every returned set must have a preimage. The four "thin"
    cases pass (a split child whose side is empty has no vertex; a certified
    side keeps one exact phase); the band case "max in the pad" does not
    today. The band of "min in the pad" is the other branch of the band
    answer, whose expected failure is
    ``test_leakyrelu_exact_on_a_coupled_star_is_exact[Star with a binding
    row-0.25]``."""
    slope = 0.25
    star = _thin_neuron_star(case)
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"exact LeakyReLU({slope}), entry 0 {case}"
    out = reach_one_layer(layer, [star], "exact", what)
    pieces = R.leaky_relu(_input_pieces(star), slope)
    if R.escapes(pieces, *[R.star(s) for s in out]):
        pytest.fail(f"{what}: the result is not even sound; see the cell "
                    f"above")
    _assert_no_point_without_preimage(pieces, out, what)


# The width the one-star band answer is documented to have: "at most about
# 2e-7 * (1 + |bound|)" (relu_reach.relu_star_exact, which
# leakyrelu_reach._step_leakyrelu follows), times |1 - slope| for LeakyReLU
# (the band is (1 - slope) times the padded bound). The bound is within the
# pad of 0 here, so 1 + |bound| is below 1 + 2^-20.
_BAND_WIDTH = Fraction(2e-7) * (1 + Fraction(1, 2 ** 20))


def _near_the_image(piece, y, width):
    """Is ``y`` within ``width``, in every coordinate, of the value of the
    piece's map at a point of its closed region? An exact feasibility
    problem in fractions."""
    A, b = [list(row) for row in piece.A], list(piece.b)
    for row, ci, yi in zip(piece.W, piece.c, y):
        A += [list(row), [-v for v in row]]
        b += [yi - ci + width, ci - yi + width]
    return R.feasible((), (), A, b) is not None


@pytest.mark.parametrize("slope", [0.25, 2.0])
@pytest.mark.parametrize("case", ["max in the pad", "min in the pad",
                                  "max in the pad, no predicate bounds"])
def test_leakyrelu_exact_band_of_a_neuron_within_the_lp_pad_is_as_documented(
        case, slope):
    """Exact LeakyReLU answers a neuron whose LP maximum (or minimum) lies
    within the LP pad of 0 with one star and a band variable, which the
    library documents (DESIGN section 18; the strict exactness cells above
    mark it as not exact, an open owner question). The documented promise
    still holds and is checked here: every vertex of every returned set is
    within |1 - slope| 2e-7 (1 + 2^-20) of a value of the network at an
    input, in every coordinate, in fractions. Both branches of the band
    answer (row scaled, band (1 - slope) [0, max]; row kept, band
    (1 - slope) [0, -min]), a Star without predicate bounds (the band's
    bounds are then constraint rows) and a slope above 1, where the band is
    negative. Catches a band wider than documented, which the soundness
    cells pass and the marked exactness cells cannot see."""
    star = _thin_neuron_star(case)
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"exact LeakyReLU({slope}), entry 0 {case}"
    out = reach_one_layer(layer, [star], "exact", what)
    pieces = R.leaky_relu(_input_pieces(star), slope)
    width = abs(1 - Fraction(slope)) * _BAND_WIDTH
    for k, s in enumerate(out):
        for y in R.set_vertices(R.star(s)):
            if not any(_near_the_image(p, y, width) for p in pieces):
                raise AssertionError(
                    f"{what}: returned set {k} of {len(out)} has the vertex "
                    f"{[float(v) for v in y]}, farther than the documented "
                    f"band {float(width):.3g} from every value of the "
                    f"network")


# --------------------------------------------------------------------------- #
# Between two affine layers, through the model entry points                     #
# --------------------------------------------------------------------------- #
_W1 = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, -0.5], [0.5, -1.0, 1.0],
                [0.25, 0.0, 1.0]])
_B1 = np.array([0.25, -0.5, 0.0, 0.125])
_W2 = np.array([[1.0, -1.0, 0.5, 0.25], [0.5, 0.5, -1.0, 1.0]])
_B2 = np.array([0.0, 0.5])
_IN_LO = np.array([-1.0, -0.5, 0.25])
_IN_HI = np.array([0.5, 1.0, 0.75])


def _torch_chain(slope):
    model = nn.Sequential(nn.Linear(3, 4), nn.LeakyReLU(slope),
                          nn.Linear(4, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(_W1))
        model[0].bias.copy_(torch.tensor(_B1))
        model[2].weight.copy_(torch.tensor(_W2))
        model[2].bias.copy_(torch.tensor(_B2))
    return model


class _FunctionalChain(nn.Module):
    """Linear, ``F.leaky_relu`` with the slope as a keyword, Linear. torch.fx
    records the call as a function node whose keyword arguments hold the
    slope (it does so for a positional slope as well), and the graph walker
    maps the node to ``nn.LeakyReLU`` (``_function_node_to_module``)."""

    def __init__(self, slope):
        super().__init__()
        self.first, self.second = nn.Linear(3, 4), nn.Linear(4, 2)
        self.slope = slope

    def forward(self, x):
        return self.second(functional.leaky_relu(self.first(x),
                                                 negative_slope=self.slope))


def _torch_model(entry, slope):
    if entry == "torch":
        return _torch_chain(slope)
    model = _FunctionalChain(slope)
    with torch.no_grad():
        model.first.weight.copy_(torch.tensor(_W1))
        model.first.bias.copy_(torch.tensor(_B1))
        model.second.weight.copy_(torch.tensor(_W2))
        model.second.bias.copy_(torch.tensor(_B2))
    return model


def _onnx_chain(tmp_path, slope):
    f32 = lambda a: np.asarray(a, dtype=np.float32)      # noqa: E731
    return save_onnx(
        tmp_path,
        [helper.make_node("Gemm", ["X", "W1", "B1"], ["H"], transB=1),
         helper.make_node("LeakyRelu", ["H"], ["A"], alpha=slope),
         helper.make_node("Gemm", ["A", "W2", "B2"], ["Y"], transB=1)],
        [1, 3], [1, 2], {"W1": f32(_W1), "B1": f32(_B1), "W2": f32(_W2),
                         "B2": f32(_B2)}, name="leaky_chain")


def _chain_input(kind):
    if kind == "Star with a binding row":
        star, points = _row_star()
        # the first three entries of the row star, a 3-entry input
        return Star(star.V[:3], star.C, star.d, star.predicate_lb,
                    star.predicate_ub), [p[:3] for p in points[::3]]
    return flat_set(kind, _IN_LO, _IN_HI), samples(_IN_LO, _IN_HI, n=_N)


_CHAIN_CELLS = (
    [pytest.param(e, "Star", "approx", None, g, id=f"{e}-Star-approx-{g}")
     for e, g in (("torch", 0.25), ("torch", 2.0), ("torch", -0.5),
                  ("onnx", -0.5))]
    + [pytest.param(e, k, "approx", None, 0.25, id=f"{e}-{k}-approx-0.25")
       for e in ("torch", "onnx") for k in ("Zono", "Box")]
    + [pytest.param("torch", "Star with a binding row", m, None, g,
                    id=f"torch-Star with a binding row-{m}-{g}")
       for m, g in (("approx", 2.0), ("exact", 0.25))]
    + [pytest.param("torch", "Star", m, "ibp", g, id=f"torch-Star-{m}-ibp-{g}")
       for m, g in (("approx", -0.5), ("exact", 0.25))]
    + [pytest.param(e, "Star", "exact", None, g, id=f"{e}-Star-exact-{g}")
       for e, g in (("torch", -0.5), ("onnx", 2.0))]
    + [pytest.param("torch-functional", "Star", "approx", None, 2.0,
                    id="torch-functional-Star-approx-2.0")])


@pytest.mark.parametrize("entry,kind,method,prepass,slope", _CHAIN_CELLS)
def test_leakyrelu_between_two_affine_layers_holds_the_exact_image(
        tmp_path, entry, kind, method, prepass, slope):
    """Linear, LeakyReLU, Linear through ``reach_pytorch_model``, as a torch
    Sequential, as a traced module that calls ``F.leaky_relu`` (a function
    node the graph walker maps to ``nn.LeakyReLU``) and as an ONNX graph
    (Gemm, LeakyRelu, Gemm) loaded with ``load_onnx``, so the rows the
    activation emits are consumed by the next layer and the slope is read
    from the module the loader builds or from the function node's keyword. Exact
    image in fractions (both directions for 'exact'), and float64 or
    onnxruntime samples by LP. Catches a relaxation row that keeps the right
    box but the wrong relation, a slope read from the wrong attribute, a
    pre-pass bound used for the wrong neuron or without its pad (``ibp``),
    and a function node's slope read from the wrong argument."""
    input_set, points = _chain_input(kind)
    what = f"Linear, LeakyReLU({slope}), Linear ({entry}) on {kind}, {method}"
    options = {} if prepass is None else {"precompute_bounds": prepass}
    if entry.startswith("torch"):
        model = _torch_model(entry, slope)
        out = reach_model(model, input_set, method, what, **options)
        truth = replay_torch(model, points, (1, 3))
    else:
        path = _onnx_chain(tmp_path, slope)
        out = reach_model(load_onnx(path), input_set, method, what, **options)
        truth = replay_onnx(path, points, [1, 3])
    pieces = R.affine(R.leaky_relu(R.affine(_input_pieces(input_set), _W1,
                                            _B1),
                                   slope), _W2, _B2)
    _assert_holds_exact_image(pieces, out, what)
    if method == "exact":
        _assert_no_point_without_preimage(pieces, out, what)
    assert_covered(out, truth, what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_leakyrelu_between_two_convolutions_covers_onnxruntime(tmp_path,
                                                                kind):
    """Conv (1x1, mixing the channels), LeakyRelu, Conv (2x1 kernel) on a
    (2, 3, 5) image with its own interval per pixel, loaded with
    ``load_onnx``: every onnxruntime output is in a returned set, compared
    in the stored order of the (1, 2, 5) result. Catches an activation
    result whose pixels the next convolution reads in the wrong place."""
    shape = [1, 2, 3, 5]
    path = save_onnx(
        tmp_path,
        [helper.make_node("Conv", ["X", "K1"], ["H"], kernel_shape=[1, 1]),
         helper.make_node("LeakyRelu", ["H"], ["A"], alpha=0.25),
         helper.make_node("Conv", ["A", "K2"], ["Y"], kernel_shape=[2, 1])],
        shape, [1, 1, 2, 5],
        {"K1": np.array([[[[1.0]], [[0.5]]], [[[-0.5]], [[1.0]]]],
                        dtype=np.float32),
         "K2": np.array([[[[1.0], [0.5]], [[-0.25], [1.0]]]],
                        dtype=np.float32)}, name="leaky_conv")
    rng = np.random.default_rng(3)
    lo = rng.integers(-8, 8, size=(2, 3, 5)) / 4.0
    hi = lo + rng.integers(1, 6, size=(2, 3, 5)) / 4.0
    what = f"Conv, LeakyRelu, Conv on {kind}"
    out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                      what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), shape), what,
                   out_chw=(1, 2, 5))


def _conv_matrix(kernel, height, width):
    """The (O Ho Wo, C H W) matrix of a convolution with no padding and
    stride 1 on a (C, H, W) tensor, both in row-major order."""
    out_c, in_c, kh, kw = kernel.shape
    oh, ow = height - kh + 1, width - kw + 1
    M = np.zeros((out_c * oh * ow, in_c * height * width))
    for o, y, x, c, i, j in _cartesian(range(out_c), range(oh), range(ow),
                                       range(in_c), range(kh), range(kw)):
        M[(o * oh + y) * ow + x, (c * height + y + i) * width + x + j] += \
            float(kernel[o, c, i, j])
    return M


def test_leakyrelu_exact_between_two_convolutions_is_the_exact_image(
        tmp_path):
    """Exact reach of Conv (1x1, mixing the channels), LeakyRelu (1/4), Conv
    (2x1 kernel) loaded with ``load_onnx``, on a (2, 3, 5) ImageStar with
    three free pixels in different channels, rows and columns (each on
    [-1/2, 3/4], so both of its channels cross 0 after the first Conv) and
    the other pixels fixed at values of both signs. Exact image in
    fractions, both directions, and onnxruntime samples by LP; the (1, 2, 5)
    output has one channel, so its stored order is the tensor's. Catches
    split pieces of an ImageStar that the next layer reads in the wrong
    place or without their cut rows, and a dropped piece."""
    shape = [1, 2, 3, 5]
    k1 = np.array([[[[1.0]], [[0.5]]], [[[-0.5]], [[1.0]]]], dtype=np.float32)
    k2 = np.array([[[[1.0], [0.5]], [[-0.25], [1.0]]]], dtype=np.float32)
    path = save_onnx(
        tmp_path,
        [helper.make_node("Conv", ["X", "K1"], ["H"], kernel_shape=[1, 1]),
         helper.make_node("LeakyRelu", ["H"], ["A"], alpha=0.25),
         helper.make_node("Conv", ["A", "K2"], ["Y"], kernel_shape=[2, 1])],
        shape, [1, 1, 2, 5], {"K1": k1, "K2": k2}, name="leaky_conv_exact")
    lo = np.random.default_rng(3).integers(-8, 8, size=(2, 3, 5)) / 4.0
    hi = lo.copy()
    free = [(0, 0, 1), (1, 2, 3), (0, 1, 4)]
    x0, E = lo.reshape(-1).copy(), np.zeros((lo.size, len(free)))
    for k, (c, y, x) in enumerate(free):
        lo[c, y, x], hi[c, y, x] = -0.5, 0.75
        x0[(c * 3 + y) * 5 + x] = 0.0
        E[(c * 3 + y) * 5 + x, k] = 1.0
    what = "exact Conv, LeakyRelu, Conv on an ImageStar"
    out = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                      "exact", what)
    pieces = R.affine(R.leaky_relu(R.affine(R.affine(R.box(
        [-0.5] * len(free), [0.75] * len(free)), E, x0),
        _conv_matrix(k1, 3, 5)), 0.25), _conv_matrix(k2, 3, 5))
    _assert_holds_exact_image(pieces, out, what)
    _assert_no_point_without_preimage(pieces, out, what)
    assert_covered(out, replay_onnx(path, samples(lo, hi, n=_N), shape),
                   what, out_chw=(1, 2, 5))


@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_leakyrelu_on_a_hexatope_or_octatope_is_refused(kind):
    """No LeakyReLU rule exists for these set types: ``reach_layer`` must
    raise ``NotImplementedError`` from the dispatcher's lane, never return
    the input or another rule's answer (the lane does not read the method, so
    'approx' stands for both). The set is built before the call, so a builder
    that raises cannot pass the cell. The model entry is
    ``test_leakyrelu_model_on_a_set_without_a_rule_is_refused``."""
    input_set = flat_set(kind, _LO, _HI)
    with pytest.raises(NotImplementedError,
                       match="not implemented for layer type: LeakyReLU"):
        reach_layer(nn.LeakyReLU(0.25), [input_set], "approx")


@pytest.mark.parametrize("slope", [0.1, 0.3])
def test_leakyrelu_star_bounds_its_new_variables_outward(slope):
    """Approx LeakyReLU on a Star from bounds whose eight entries all cross
    0, with a slope that is not dyadic: for these lower bounds (-k/16) the
    float product slope * lb lies above the exact one, so a bound of the new
    output variable taken from the float product without rounding it
    outward leaves the exact output slope * lb out. Exact fractions: the
    image corner with every entry at its lower bound, the one with every
    entry at its upper bound, and the eight with one entry at its lower
    bound and the rest at the upper must be in the returned set. No entry is
    inactive, so the round-to-nearest scaling of inactive rows (registry
    ROUND-9) does not enter."""
    k = np.array([5, 9, 10, 13, 3, 6, 11, 18])
    lo, hi = -k / 16.0, np.full(k.size, 0.5)
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"LeakyReLU({slope}) on a crossing Star"
    star = flat_set("Star", lo, hi)
    out = reach_one_layer(layer, [star], "approx", what)
    g = Fraction(float(layer.negative_slope))
    low = [g * R.frac(v) for v in lo]
    high = [R.frac(v) for v in hi]
    corners = [low, high] + [high[:i] + [low[i]] + high[i + 1:]
                             for i in range(k.size)]
    held = R.Membership(R.star(out[0]))
    missed = [i for i, y in enumerate(corners) if not held(y)]
    assert len(out) == 1 and not missed, (
        f"{what}: {len(out)} sets; image corners {missed} are outside, "
        f"the first {[float(v) for v in corners[missed[0]]]}"
        if missed else f"{what}: expected one set, got {len(out)}")


@known_bug(
    16, "approx LeakyReLU on a Star without predicate bounds stores the row "
        "y >= slope x of each crossing neuron as the float product slope * V "
        "with no outward correction (the correction is computed from the "
        "predicate box, and there is none), so with a slope that is not "
        "dyadic the exact output slope * lb at the input's lower corner "
        "violates the row by about 2e-18; giving the Star a finite box or "
        "refusing it, the fix of registry REACH-5, removes it")
def test_leakyrelu_star_without_predicate_bounds_bounds_its_new_variables_outward():
    """The cell above on the same eight crossing entries, the box given as
    constraint rows and no predicate bounds, slope 0.1: the branch of the
    approx kernel without a predicate box adds no outward slack to the rows
    it emits. Exact fractions, the same ten image corners. Neighbour that
    passes today: ``test_leakyrelu_star_bounds_its_new_variables_outward[0.1]``
    (the same input with predicate bounds)."""
    k = np.array([5, 9, 10, 13, 3, 6, 11, 18])
    lo, hi = -k / 16.0, np.full(k.size, 0.5)
    layer = nn.LeakyReLU(negative_slope=0.1)
    what = "LeakyReLU(0.1) on a crossing Star without predicate bounds"
    boxed = flat_set("Star", lo, hi)
    star = Star(boxed.V, boxed.C, boxed.d, None, None)
    if star.C.shape[0] != 2 * k.size:
        pytest.fail(f"{what}: the box rows are missing, {star.C.shape}")
    out = reach_one_layer(layer, [star], "approx", what)
    g = Fraction(float(layer.negative_slope))
    low = [g * R.frac(v) for v in lo]
    high = [R.frac(v) for v in hi]
    corners = [low, high] + [high[:i] + [low[i]] + high[i + 1:]
                             for i in range(k.size)]
    held = R.Membership(R.star(out[0]))
    missed = [i for i, y in enumerate(corners) if not held(y)]
    assert len(out) == 1 and not missed, (
        f"{what}: {len(out)} sets; image corners {missed} are outside, "
        f"the first {[float(v) for v in corners[missed[0]]]}"
        if missed else f"{what}: expected one set, got {len(out)}")


_DIMENSION_ZERO_DROPPED = known_bug(
    7, "the LeakyReLU Star kernels return no set for a Star of dimension 0 "
       "(the exact kernel returns an empty list, the approx kernel None, "
       "which its wrapper drops), although its one member, the empty "
       "vector, has the empty output: the result says the input is empty "
       "(registry EMPTY-6)")


@pytest.mark.parametrize("method", [
    pytest.param(m, marks=_DIMENSION_ZERO_DROPPED) for m in ("exact",
                                                             "approx")])
def test_leakyrelu_on_a_star_of_dimension_zero_is_not_dropped(method):
    """A Star of dimension 0 over a non-empty predicate polytope is not
    empty, so ``reach_layer`` must return a set for it, or refuse it with
    ``ValueError``; no sets at all is the answer "empty", which a
    specification check reads as a proof. Catches a kernel that drops an
    input set it has not certified empty. Neighbour that passes today:
    ``test_leakyrelu_on_a_product_of_intervals_holds_the_exact_image``
    (``[Star-exact-1set-0.25]``, ``[Star-approx-1set-0.25]``)."""
    star = Star(np.zeros((0, 2)), np.array([[1.0], [-1.0]]), np.ones((2, 1)),
                -np.ones((1, 1)), np.ones((1, 1)))
    if star.dim != 0:
        pytest.fail(f"the input Star has dimension {star.dim}, not 0")
    try:
        out = reach_layer(nn.LeakyReLU(0.25), [star], method)
    except ValueError:
        return
    assert len(out) >= 1, (
        f"LeakyReLU {method} on a Star of dimension 0: no set returned for "
        f"a non-empty input")


@pytest.mark.parametrize("slope", [0.3, -0.3, 3.0])
def test_leakyrelu_box_rounds_the_slope_product_outward(slope):
    """The Box rule promises f(lb), f(ub) and 0 rounded outward. With a slope
    that is not dyadic the float product of the slope and a negative bound is
    rounded; the exact product (fractions of the stored slope and bounds)
    must still be inside the returned box. Checked one coordinate at a time
    (a box is the product of its coordinates' intervals). Catches a product
    rounded to nearest or inward instead of outward, at the lower and at the
    upper end. Bounds: negative multiples of 1/8 and of 1/3 and 1/7, so that
    both directions of rounding occur. A positive slope (the lower end from
    slope * lb, the upper from slope * ub of a negative coordinate) and a
    negative one (where slope * lb is the upper end); 0.3 is not a float32
    number, so a slope read through float32 fails here too. A slope above
    one (3.0), where the product is larger in magnitude than the bound and
    is still rounded for the bounds that are not dyadic: a grid of slopes
    below one alone would pass a rule that is outward only when the product
    shrinks."""
    lo = np.array([-0.75, -1.25, -1 / 3, -2 / 3, -0.875, -5 / 3, -1.0])
    hi = np.array([0.5, -0.25, 0.25, -1 / 3, 0.0, 0.75, -1 / 7])
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"LeakyReLU({slope}) on Box"
    out = reach_one_layer(layer, [Box(lo, hi)], "approx", what)
    assert len(out) == 1 and isinstance(out[0], Box), (
        f"{what}: expected one Box, got {[type(s).__name__ for s in out]}")
    out_lb = np.asarray(out[0].lb, dtype=np.float64).reshape(-1)
    out_ub = np.asarray(out[0].ub, dtype=np.float64).reshape(-1)
    missed = []
    for i in range(lo.size):
        pieces = R.leaky_relu(R.box([lo[i]], [hi[i]]), slope)
        held = R.Membership(R.interval([out_lb[i]], [out_ub[i]]))
        missed += [(i, y[0]) for p in pieces for y in p.image()
                   if not held(y)]
    assert not missed, (
        f"{what}: {len(missed)} exact end values outside the returned box, "
        f"first coordinate {missed[0][0]}: exact {missed[0][1]} "
        f"({float(missed[0][1])!r}), box [{out_lb[missed[0][0]]!r}, "
        f"{out_ub[missed[0][0]]!r}]")


# =========================================================================== #
# Zono, ImageZono and Box: the shape a relaxation leaves, wide data, the       #
# kernels' own arithmetic, infinite bounds, the exact method and refusals      #
# =========================================================================== #
# A zonotope of four entries on seven generators, the shape an earlier
# relaxation leaves: three generators shared by the entries, then one of each
# entry's own beyond the first four columns. Entries 0 to 2 cross 0 on ranges
# up to 18 wide ([-8, 10], [-11, 7], [-4, 10]); entry 3 is stably negative on
# [-17, -7]. Integer data, so every member point below is exact in float64.
_WIDE_C = np.array([1.0, -2.0, 3.0, -12.0])
_WIDE_G = np.array([[4.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                    [2.0, -4.0, 1.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 2.0, 4.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, -2.0, 0.0, 0.0, 0.0, 2.0]])


def _wide_set(kind):
    """``(input set, member points in the order the set stores)``: the
    zonotope above as a ``Zono``, or as an ``ImageZono`` of height 1, width
    2 and 2 channels (its four entries in stored order), with the 128
    vertices of its predicate cube as points; or the box of ``_LO``, ``_HI``
    times 8 (ends from -12 to 12, crossing intervals 8 and 16 wide) as a
    ``Box``, with 128 points of ``samples``."""
    if kind == "Box":
        lo, hi = 8 * _LO, 8 * _HI
        return Box(lo, hi), samples(lo, hi, n=128)
    points = [_WIDE_C + _WIDE_G @ np.array(a)
              for a in _cartesian(*[(-1.0, 1.0)] * _WIDE_G.shape[1])]
    c = _WIDE_C.reshape(-1, 1)
    if kind == "Zono":
        return Zono(c, _WIDE_G.copy()), points
    return ImageZono(c, _WIDE_G.copy(), 1, 2, 2), points


@pytest.mark.parametrize("kind,slope", [("Zono", 0.25), ("Zono", 2.0),
                                        ("ImageZono", -0.5), ("Box", -0.5)])
def test_leakyrelu_on_a_wide_zono_or_box_holds_every_output(kind, slope):
    """Approx LeakyReLU on a Zono with more generators than entries (the
    shape a relaxation leaves: the bounds of a neuron must come from every
    generator column, not from the first ``dim`` ones), the same as an
    ImageZono (whose result must keep height, width and channels), and a Box,
    all with data well beyond [-1, 1] and intervals up to 18 wide. One set
    per input set; 128 points (every vertex of the predicate cube, or the box
    corners and grid points) replayed in float64, each a member of the result
    by the LP on its own rows; for the Box also the exact image, coordinate
    by coordinate, in fractions. Catches a neuron's range read from part of
    the generators (a band built for too narrow a range misses the outputs
    near the true ends), slopes below 1, above 1 (the concave band) and below
    0 (where f(lb) can be the largest value), an ImageZono rebuilt with height
    and width exchanged, and a result that holds the outputs only while the
    data stay inside [-1, 1]."""
    input_set, points = _wide_set(kind)
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"LeakyReLU({slope}) on a wide {kind}"
    out = reach_one_layer(layer, [input_set], "approx", what)
    assert len(out) == 1, f"{what}: {len(out)} sets for one input set"
    if kind == "ImageZono":
        shape = (type(out[0]), out[0].height, out[0].width, out[0].num_channels)
        assert shape == (ImageZono, 1, 2, 2), (
            f"{what}: result {shape}, expected an ImageZono 1x2x2")
    if kind == "Box":
        _assert_holds_product_image(_leaky(slope), [input_set], out, what,
                                    pieces=False)
    truth = [np.where(x >= 0, x, slope * x) for x in points]
    assert_covered(out, truth, what, stored_order=kind == "ImageZono")


# The arithmetic of the Zono kernel, in exact fractions. Each zonotope has two
# entries: entry 0 = c + v a is the neuron, entry 1 = 2 + a is stably positive
# and pins the predicate value a in the output, so the exact output
# (f(c + v a), 2 + a) must be matched by the neuron's own row.
_ZONO_ARITHMETIC_TO_NEAREST = known_bug(
    "7b", "approx LeakyReLU on a Zono computes in round-to-nearest with no "
          "outward correction: the slope times the centre and generator row "
          "of a stably negative entry (the slope product of ROUND-9), and the "
          "band of a crossing entry (its slope a = (u - slope l) / (u - l), "
          "centre a c + b/2 and error generator |b|/2, the relaxation "
          "rounding of ROUND-10), so the exact image of the stored zonotope "
          "under the stored slope lies outside the returned zonotope by units "
          "in the last place (ROUND-16)")


def _pinned_zonos(path, data):
    """Eight zonotopes of the shape above, passed at once.

    - "inactive": entry 0 is negative on its whole range. With data "powers
      of two", c = -2^(k+3) and v = 2^(k+1) (k = 0 to 7, ranges up to
      [-1536, -512]), so the slope times c and times v is a float for any
      slope; with "other", seeded floats with full mantissas.
    - "crossing": entry 0 crosses 0. With "powers of two", c = +-2^k / 4 and
      v = 2^k (k = -2 to 5), so the range is 2^(k+1) wide and, for a dyadic
      slope, the band's slope, centre and error generator are floats; with
      "other", seeded floats with |c| < v.

    Data "powers of two, float32": the same numbers stored as float32 arrays
    (a Zono keeps the float type it is given; the numbers are exact in
    float32)."""
    rng = np.random.default_rng(29 if path == "inactive" else 31)
    dtype = np.float32 if data.endswith("float32") else np.float64
    zonos = []
    for k in range(8):
        if data.startswith("powers of two"):
            if path == "inactive":
                c, v = -2.0 ** (k + 3), 2.0 ** (k + 1)
            else:
                c, v = (-1) ** k * 2.0 ** (k - 4), 2.0 ** (k - 2)
        elif path == "inactive":
            c = -rng.uniform(1.0, 100.0)
            v = rng.uniform(0.05, 0.95) * -c
        else:
            c = rng.uniform(-4.0, 4.0)
            v = abs(c) + rng.uniform(0.5, 4.0)
        zonos.append(Zono(np.array([[c], [2.0]], dtype=dtype),
                          np.array([[v], [1.0]], dtype=dtype)))
    return zonos


_FLOAT32_ZONO_COMPUTED_IN_FLOAT32 = known_bug(
    7, "approx LeakyReLU on a Zono whose arrays are float32 (the Zono "
       "constructor keeps a float type) writes the new centre and generator "
       "rows into float32 copies of the input's arrays, so they are rounded "
       "to float32: with slope 0.1 on data exact in float32 the exact output "
       "lies outside the returned zonotope by up to 1.49e-07 (LP distance, "
       "32 of 60 exact outputs), against 0 for the same numbers stored as "
       "float64")


@pytest.mark.parametrize("path,data,slope", [
    pytest.param("inactive", "powers of two", 0.1,
                 id="inactive-powers of two-0.1"),
    pytest.param("inactive", "powers of two, float32", 0.1,
                 marks=_FLOAT32_ZONO_COMPUTED_IN_FLOAT32,
                 id="inactive-powers of two, float32-0.1"),
    pytest.param("inactive", "other", 0.1, marks=_ZONO_ARITHMETIC_TO_NEAREST,
                 id="inactive-other-0.1"),
    pytest.param("crossing", "powers of two", 0.25,
                 id="crossing-powers of two-0.25"),
    pytest.param("crossing", "other", 0.25, marks=_ZONO_ARITHMETIC_TO_NEAREST,
                 id="crossing-other-0.25")])
def test_leakyrelu_zono_holds_the_exact_image_of_its_stored_data(path, data,
                                                                 slope):
    """Approx LeakyReLU on eight pinned zonotopes at once: the exact output of
    every end point of entry 0 (a = -1 and 1) and, for a crossing entry, of
    the kink (a = -c / v), in fractions of the stored centre, generators and
    slope, must be a member of a returned set, decided by the exact simplex
    with no slack. The data "powers of two" make every product the kernel
    forms a float, so these cases pass and catch a slope read through float32
    (0.1 is not a float32 number; the outputs, up to 154 in size, move by
    about 2e-7), a band of the wrong slope or height, and an entry given the
    wrong phase; the data "other" show the round-to-nearest arithmetic of the
    kernel (strict expected failures, one per path: the slope product of a
    negative entry, and the band of a crossing one), and the same powers of
    two stored as float32 show the kernel computing in the input's float32
    (strict expected failure; its neighbour is the float64 item)."""
    zonos = _pinned_zonos(path, data)
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"LeakyReLU({slope}) on eight {path} Zonos ({data})"
    out = reach_one_layer(layer, zonos, "approx", what)
    if len(out) != len(zonos):
        pytest.fail(f"{what}: {len(out)} sets for {len(zonos)} input sets")
    holders = [R.Membership(R.star(s)) for s in out]
    s = Fraction(float(layer.negative_slope))
    missed = []
    for k, z in enumerate(zonos):
        c, v = R.frac(z.c[0, 0]), R.frac(z.V[0, 0])
        ends = [Fraction(-1), Fraction(1)] + ([-c / v] if path == "crossing"
                                              else [])
        for a in ends:
            x = c + v * a
            y = [x if x >= 0 else s * x, 2 + a]
            if not any(h(y) for h in holders):
                missed.append((k, float(a)))
    assert not missed, (
        f"{what}: the exact output is in no returned set at {len(missed)} "
        f"(zonotope, a), the first {missed[:4]}")


_ZERO_SLOPE_TIMES_INFINITY = known_bug(
    7, "LeakyReLU of slope 0 on a Box with an infinite lower bound multiplies "
       "the bound by the slope: 0 * -inf is NaN, and the returned box has NaN "
       "bounds, which hold no point, where the true outputs of a coordinate "
       "[-inf, u] fill [0, max(u, 0)] (ReLU on the same box answers it)")


@pytest.mark.parametrize("slope", [
    pytest.param(0.0, marks=_ZERO_SLOPE_TIMES_INFINITY), 0.25, -0.5])
def test_leakyrelu_on_a_box_with_an_infinite_bound_holds_its_image(slope):
    """A Box may have infinite bounds. LeakyReLU on the coordinates
    [-inf, 1], [-1, inf], [-inf, -2] and [1/2, inf]: the closure of the image
    of each coordinate, from the end values f(-inf) (-inf for a positive
    slope, 0 for slope 0, +inf for a negative one), f(u) and 0 when the
    coordinate crosses, must lie inside the returned interval, with no NaN
    bound. NumPy's warning for an invalid product is silenced around the
    call, as it would only be printed in ordinary use, so the case sees the
    box a user gets. Catches an infinite end multiplied by a slope of 0,
    and a negative slope whose f(-inf) = +inf is not taken as the upper
    bound. Neighbours that pass today: slopes 1/4 and -1/2."""
    lo = np.array([-np.inf, -1.0, -np.inf, 0.5])
    hi = np.array([1.0, np.inf, -2.0, np.inf])
    layer = nn.LeakyReLU(negative_slope=slope)
    what = f"LeakyReLU({slope}) on a Box with infinite bounds"
    with np.errstate(invalid="ignore"):
        out = reach_one_layer(layer, [Box(lo, hi)], "approx", what)
    if len(out) != 1 or not isinstance(out[0], Box):
        pytest.fail(f"{what}: expected one Box, got "
                    f"{[type(s).__name__ for s in out]}")

    def end(v):
        return v if v >= 0 else (0.0 if slope == 0 else slope * v)

    out_lb = np.asarray(out[0].lb, dtype=np.float64).reshape(-1)
    out_ub = np.asarray(out[0].ub, dtype=np.float64).reshape(-1)
    bad = []
    for i, (l, u) in enumerate(zip(lo, hi)):
        values = [end(l), end(u)] + ([0.0] if l < 0 < u else [])
        if not (out_lb[i] <= min(values) and out_ub[i] >= max(values)):
            bad.append((i, (min(values), max(values)),
                        (out_lb[i], out_ub[i])))
    assert not bad, (
        f"{what}: {len(bad)} coordinates whose image is not inside the "
        f"returned interval, (index, image, returned): {bad}")


_EXACT_ZONO_IS_THE_RELAXATION = known_bug(
    18, "reach_layer with method='exact' on a Zono or an ImageZono runs the "
        "LeakyReLU relaxation and returns it under the name exact, with no "
        "warning: the Zono lane of the dispatcher does not read the method "
        "(REACH-8, as for Sigmoid and Tanh); the model entry refuses "
        "'exact' on a Zono")


@pytest.mark.parametrize("kind", [
    pytest.param("Zono", marks=_EXACT_ZONO_IS_THE_RELAXATION), "Box"])
def test_exact_leakyrelu_on_a_zono_or_a_box_refuses_or_is_the_image(kind):
    """DESIGN: 'exact' is exact up to float64 rounding or raises.
    ``reach_layer(LeakyReLU, sets, 'exact')``: a refusal
    (``NotImplementedError`` or ``ValueError``, as the model entry gives)
    keeps the promise; an answer must hold the exact image and have no point
    farther from it than float64 rounding. Zono: LeakyReLU(1/4) on the
    correlated zonotope {(a, -a)}, whose image is two segments meeting at 0
    and is no zonotope, so only a refusal is right; today the relaxation is
    returned (6 of its 8 vertices have no preimage, in fractions). The same
    lane serves an ImageZono, so this one marked case stands for both. Box:
    LeakyReLU(-1/2) on a box with crossing, negative, zero-width and
    bound-0 coordinates: the image of a box under a coordinate-wise map is
    the box of the coordinates' images, and every returned end must be the
    exact end moved outward by at most two units in the last place (the
    rounded product and its outward step). Catches the exact method
    returning an over-approximation under its name."""
    what = f"exact LeakyReLU on a {kind}"
    try:
        if kind == "Zono":
            sets = [Zono(np.zeros((2, 1)), np.array([[1.0], [-1.0]]))]
            out = reach_one_layer(nn.LeakyReLU(0.25), sets, "exact", what)
        else:
            sets = [Box(_LO, _HI)]
            out = reach_one_layer(nn.LeakyReLU(-0.5), sets, "exact", what)
    except (NotImplementedError, ValueError):
        return
    if kind == "Zono":
        pieces = R.leaky_relu(R.affine(R.box([-1], [1]), [[1], [-1]],
                                       [0, 0]), 0.25)
        _assert_holds_exact_image(pieces, out, what)
        vertices = [y for s in out for y in R.set_vertices(R.star(s))]
        extra = R.unreached(pieces, vertices)
        assert not extra, (
            f"{what}: {len(extra)} of the {len(vertices)} vertices of the "
            f"result have no preimage, the first "
            f"{[float(v) for v in extra[0][1]]}")
        return
    images = _coordinate_images(_leaky(-0.5), *_box_of(sets[0]))
    out_lb = np.asarray(out[0].lb, dtype=np.float64).reshape(-1)
    out_ub = np.asarray(out[0].ub, dtype=np.float64).reshape(-1)
    bad = []
    for i, coord in enumerate(images):
        values = [y[0] for p in coord for y in p.image()]
        low, high = min(values), max(values)
        slack_low = 2 * R.frac(np.spacing(abs(float(low))))
        slack_high = 2 * R.frac(np.spacing(abs(float(high))))
        if not (low - slack_low <= R.frac(out_lb[i]) <= low
                and high <= R.frac(out_ub[i]) <= high + slack_high):
            bad.append((i, (float(low), float(high)), (out_lb[i], out_ub[i])))
    assert len(out) == 1 and not bad, (
        f"{what}: {len(out)} sets; coordinates whose returned interval is "
        f"not the image up to rounding, (index, image, returned): {bad}")


@pytest.mark.parametrize("kind,method", [
    ("Zono", "exact"), ("ImageZono", "exact"), ("Box", "exact"),
    ("Hexatope", "exact"), ("Octatope", "exact"),
    ("Hexatope", "approx"), ("Octatope", "approx")])
def test_leakyrelu_model_on_a_set_without_a_rule_is_refused(kind, method):
    """Linear, LeakyReLU(1/4), Linear through ``reach_pytorch_model``.
    ``exact`` on a set type with no exact method is refused before any layer
    (today ``ValueError``, "... only supports 'approx'"; the planned
    ``UnsupportedError`` subclasses ``NotImplementedError``); ``approx`` on a
    Hexatope or an Octatope passes the first Linear and reaches the
    LeakyReLU node, for which these lanes have no rule:
    ``NotImplementedError`` naming the layer. Catches the model entry handing
    'exact' to a lane that does not read it (the relaxation returned under
    the name exact, cell above) and a Hexatope or Octatope lane that answers
    LeakyReLU with another rule (the ReLU rule leaves out every negative
    output)."""
    if kind == "ImageZono":
        input_set = image_set(kind, _IN_LO.reshape(1, 1, 3),
                              _IN_HI.reshape(1, 1, 3))
    else:
        input_set = flat_set(kind, _IN_LO, _IN_HI)
    what = f"Linear, LeakyReLU, Linear on a {kind}, {method}"
    if method == "exact":
        with pytest.raises((NotImplementedError, ValueError)):
            reach_model(_torch_chain(0.25), input_set, method, what)
    else:
        with pytest.raises(NotImplementedError,
                           match="not implemented for layer type: LeakyReLU"):
            reach_model(_torch_chain(0.25), input_set, method, what)
