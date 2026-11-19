"""Soundness tests for Sign activation reachability.

Sign is reached as the module ``load_onnx`` builds for an ONNX Sign node
(onnx2torch's ``OnnxFunction`` wrapping ``torch.sign``): the Star lane of the
dispatcher calls ``sign_star`` (exact split or approx relaxation, an
ImageStar processed flat and converted back), the Zono lane ``sign_zono``
and the Box lane ``sign_box`` (interval rules that do not read the method);
the Hexatope and Octatope lanes have no Sign rule and refuse. A torch model
that calls ``torch.sign`` is refused by the graph walk.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono


# =========================================================================== #
# Soundness cells                                                              #
#                                                                              #
# Sign is reached as the module ``load_onnx`` builds for an ONNX Sign node     #
# (onnx2torch's ``OnnxFunction`` wrapping ``torch.sign``), through             #
# ``reach_layer`` and, inside a graph, through ``reach_pytorch_model``.         #
#                                                                              #
# Reference: the exact image in rationals (tests/_oracles/rational.py, its     #
# ``sign`` keeps sign(0) = 0 as a branch of its own), read from the input      #
# set's stored arrays, never from n2v. Two shapes of input:                    #
#                                                                              #
# - a set that is a product of intervals (Box, Star/Zono/ImageStar from        #
#   bounds, the parts of a split box). Sign acts coordinate by coordinate, so  #
#   the exact image is the product of the coordinates' images, each computed  #
#   by the oracle in one dimension;                                            #
# - a set whose coordinates share predicate variables (a coupled Star, one    #
#   with a binding row that is not a box row, the halves of a split, a Star   #
#   with no predicate bounds, a Star with more variables than entries, a      #
#   coupled Zono, a Zono with more generators than entries): the oracle's     #
#   polytope pipeline on the predicate space.                                  #
#                                                                              #
# Soundness: every piece of the exact image has all its vertices in one        #
# returned set, decided in fractions with no slack. Exactness                  #
# (method='exact'): every vertex of every returned set has a preimage, and so  #
# has the point a quarter of the way between two of its vertices (the image   #
# of Sign is a finite set of points). Sample points (dyadic, with 0 on the     #
# grid) replayed in float64 are checked by one LP on the set's own rows as     #
# well. Layout: a network tensor is (C, H, W); an image set stores (H, W, C).  #
# Sign on an ImageZono is left to tests/contracts/test_reach_coverage.py,      #
# which marks it (registry ACT-3).                                             #
# =========================================================================== #
from fractions import Fraction
from itertools import product as _cartesian

import pytest
from onnx import helper
from onnx2torch.node_converters.functions import OnnxFunction

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets.image_star import ImageStar
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests.cell_support import (N_SAMPLES, assert_covered, flat_set,
                                image_set, image_split_cover, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx, split_cover)
from tests.known_bugs import known_bug

# Coordinates: crossing, lower bound exactly 0, upper bound exactly 0, the
# point 0, a nonzero point, negative, positive. The values 0 are where the
# kernels compare (``lb > 0``, ``ub < 0``, ``lb == ub == 0``, ``lb == 0``,
# ``ub == 0``); the sample grid holds 0 in every interval that has it. The
# first coordinate is the widest, so a split box is cut along it into a
# negative part, a crossing part and a positive last part: an approx result
# for the last part alone (sign +1 there) leaves out the first part's -1.
_LO = np.array([-0.5, 0.0, -1.25, 0.0, 0.375, -1.5, 0.25])
_HI = np.array([1.5, 0.75, 0.0, 0.0, 0.375, -0.25, 1.0])
# The exact image is the verdict of every cell; 32 replayed samples a cell are
# the uniform extra check (corners and the quarter grid, which holds 0).
_N = 32


def _sign_layer():
    return OnnxFunction("Sign")



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
    first vertex that is not. The exact image of a network whose last
    nonlinear layer is Sign is a finite set of points, so a returned set with
    two distinct vertices holds the point a quarter of the way between them,
    which must have a preimage as well (a box [-1, 1] has the preimages -1
    and 1 at its vertices and none at -1/2)."""
    for k, s in enumerate(sets):
        computed = R.star(s)
        if len(computed[4]) > 4:
            raise AssertionError(
                f"{what}: returned set {k} has {len(computed[4])} predicate "
                f"variables; an exact result of these inputs keeps the "
                f"input's (at most 3)")
        corners = R.set_vertices(computed)
        for y in corners:
            if not R.image_contains(pieces, y):
                raise AssertionError(
                    f"{what}: returned set {k} of {len(sets)} has the vertex "
                    f"{[float(v) for v in y]}, which no input reaches")
        if len(corners) > 1:
            y = tuple(a + (b - a) / 4 for a, b in zip(corners[0], corners[1]))
            if not R.image_contains(pieces, y):
                raise AssertionError(
                    f"{what}: returned set {k} of {len(sets)} holds the point "
                    f"{[float(v) for v in y]} between two of its vertices, "
                    f"which no input reaches")



def _grid(k, steps=5):
    values = np.linspace(-1.0, 1.0, steps)
    return [np.array(a) for a in _cartesian(*[values] * k)]


# Seven entries on two predicate variables a in [-1, 1]^2: entry 0 crosses
# 0, entry 1 crosses 0, entry 2 is positive, entry 3 negative, entry 4 the
# constant 0, entry 5 is entry 0 + 1/8 and entry 6 is entry 0 - 1/8. Entry 5
# is decided positive where entry 0 >= 0 and entry 6 negative where entry 0
# <= 0, so the split step decides them by their LP range in some children
# (its early returns) and splits them in others.
_SC = np.array([0.25, -0.25, 1.0, -0.75, 0.0, 0.375, 0.125])
_SG = np.array([[0.5, 0.25], [0.5, -0.5], [0.25, 0.25], [0.25, 0.0],
                [0.0, 0.0], [0.5, 0.25], [0.5, 0.25]])


def _sign_star(extra=(), box=True):
    """``(star, points)``: the seven entries as a ``Star`` with the box rows
    and ``extra`` rows ``(row, rhs)``, with the predicate box [-1, 1]^2 or
    (``box=False``) with no predicate bounds, so that the rows alone bound
    the variables; ``points`` are the member points of the half grid of the
    predicate box (exact)."""
    rows = list(extra)
    C = np.vstack([np.eye(2), -np.eye(2)] + [r for r, _ in rows])
    d = np.concatenate([np.ones(4), [v for _, v in rows]]).reshape(-1, 1)
    bounds = (-np.ones((2, 1)), np.ones((2, 1))) if box else ()
    star = Star(np.hstack([_SC[:, None], _SG]), C, d, *bounds)
    points = [_SC + _SG @ a for a in _grid(2, 9)
              if all(r @ a <= v for r, v in rows)]
    return star, points


# Two entries on three predicate variables (more variables than entries, the
# shape a relaxation leaves), well beyond [-1, 1]: entry 0 in [-3.5, 4.5],
# entry 1 in [-5, 3]. The row a0 + a1 + a2 <= 1 cuts a corner off the box.
_WC = np.array([0.5, -1.0])
_WG = np.array([[2.0, 1.0, -1.0], [1.0, -2.0, 1.0]])
# Two entries on three generators (more generators than entries), entry 0 in
# [-1.25, 4.25] (its third generator alone takes it across 0), entry 1 in
# [-4, 3].
_ZC = np.array([1.5, -0.5])
_ZG = np.array([[0.5, -0.25, 2.0], [1.0, 2.0, -0.5]])


def _family(name):
    """``(input sets, member points)`` of a family whose entries share
    predicate variables."""
    if name == "coupled Star":
        star, points = _sign_star()
        return [star], points
    if name == "Star whose entry 0 has true minimum 0":
        # the binding row -a0/2 - a1/4 <= 1/4 keeps entry 0 >= 0; its
        # interval bound is still -1/2
        star, points = _sign_star([(-_SG[0], _SC[0])])
        return [star], points
    if name == "two halves of a split":
        # cut at entry 1 = 0, as an earlier exact split leaves the star
        low, p1 = _sign_star([(_SG[1], -_SC[1])])
        high, p2 = _sign_star([(-_SG[1], _SC[1])])
        return [low, high], p1 + p2
    if name == "Star whose entry 0 has true maximum 0":
        # the binding row a0/2 + a1/4 <= -1/4 keeps entry 0 <= 0; its
        # interval bound is still 1
        star, points = _sign_star([(_SG[0], -_SC[0])])
        return [star], points
    if name == "Star with no predicate bounds":
        star, points = _sign_star(box=False)
        return [star], points
    if name == "Star with more predicate variables than entries":
        row = np.ones(3)
        C = np.vstack([np.eye(3), -np.eye(3), row])
        d = np.concatenate([np.ones(6), [1.0]]).reshape(-1, 1)
        star = Star(np.hstack([_WC[:, None], _WG]), C, d, -np.ones((3, 1)),
                    np.ones((3, 1)))
        return [star], [_WC + _WG @ a for a in _grid(3, 5) if row @ a <= 1]
    if name == "Star with a row that decides entry 0":
        # x0 = a0 is held in [1/4, 1] by the row -a0 <= -1/4, where the
        # predicate box gives [-1, 1]: only the LP range decides its sign;
        # x1 = 1/4 + a0/2 - a1/2 and x2 = (a0 + a1)/2 cross 0
        V = np.array([[0.0, 1.0, 0.0], [0.25, 0.5, -0.5], [0.0, 0.5, 0.5]])
        C = np.vstack([np.eye(2), -np.eye(2), [[-1.0, 0.0]]])
        d = np.array([[1.0], [1.0], [1.0], [1.0], [-0.25]])
        star = Star(V, C, d, -np.ones((2, 1)), np.ones((2, 1)))
        return [star], [V[:, 0] + V[:, 1:] @ a for a in _grid(2, 9)
                        if a[0] >= 0.25]
    if name == "coupled Zono":
        zono = Zono(_SC.reshape(-1, 1), _SG.copy())
        return [zono], [_SC + _SG @ a for a in _grid(2, 9)]
    if name == "Zono with more generators than entries":
        zono = Zono(_ZC.reshape(-1, 1), _ZG.copy())
        return [zono], [_ZC + _ZG @ a for a in _grid(3, 5)]
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# One layer through reach_layer                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,method,parts", [
    ("Box", "approx", 1), ("Box", "approx", 3), ("Zono", "approx", 1),
    ("Zono", "approx", 3), ("Star", "approx", 1), ("Star", "approx", 3),
    ("Star", "exact", 1), ("Star", "exact", 3)])
def test_sign_on_a_product_of_intervals_holds_the_exact_image(kind, method,
                                                              parts):
    """Sign through ``reach_layer`` on a Box, a Zono and a Star built from
    bounds, one set and three parts of a split box at once, with intervals
    on every side of 0 and on it: crossing, lower bound exactly 0 (the point
    0 has sign 0), upper bound exactly 0, the point 0 itself, a nonzero
    point, negative, positive. Exact image in fractions (both directions
    under 'exact') and float64 samples by LP. Catches a bound of exactly 0
    classed as the strict side (sign(0) = 0 left out, as `lb >= 0` instead
    of `lb > 0` would), a crossing output narrowed or given the wrong sign,
    a decided coordinate read from another's bounds, a dropped split child
    (the sign-0 child above all), and the outputs of all but one input set
    dropped."""
    sets = ([flat_set(kind, _LO, _HI)] if parts == 1
            else split_cover(kind, _LO, _HI, parts))
    what = f"Sign on {kind} ({len(sets)} sets), {method}"
    out = reach_one_layer(_sign_layer(), sets, method, what)
    _assert_holds_product_image(R.sign, sets, out, what,
                                pieces=method == "exact")
    if method == "exact":
        _assert_product_exact(R.sign, sets, out, what)
    assert_covered(out, replay_torch(_sign_layer(), samples(_LO, _HI, n=_N),
                                     (1, _LO.size)), what)


# Every entry decided: positive, negative, the point 0, a nonzero point.
_DECIDED_LO = np.array([0.25, -1.5, 0.0, 0.375, -0.5])
_DECIDED_HI = np.array([1.0, -0.25, 0.0, 0.375, -0.5])


@pytest.mark.parametrize("kind,method", [
    ("Box", "approx"), ("Zono", "approx"), ("Star", "approx"),
    ("Star", "exact")])
def test_sign_with_every_entry_decided_holds_the_exact_image(kind, method):
    """Sign on a set whose every entry has a decided sign (no free output
    variable at all): the branch that writes the constant signs straight
    into the result. Exact image in fractions. Catches a constant written
    with the wrong sign or to the wrong entry, the point 0 given sign +1 or
    -1, and the Zono built with no generator in the wrong shape."""
    sets = [flat_set(kind, _DECIDED_LO, _DECIDED_HI)]
    what = f"Sign on {kind} with every entry decided, {method}"
    out = reach_one_layer(_sign_layer(), sets, method, what)
    _assert_holds_product_image(R.sign, sets, out, what,
                                pieces=method == "exact")
    if method == "exact":
        _assert_product_exact(R.sign, sets, out, what)
    assert_covered(out, replay_torch(_sign_layer(),
                                     samples(_DECIDED_LO, _DECIDED_HI, n=8),
                                     (1, _DECIDED_LO.size)), what)


@pytest.mark.parametrize("family,method", [
    pytest.param(f, m, id=f"{f}-{m}")
    for f in ("coupled Star", "Star whose entry 0 has true minimum 0",
              "two halves of a split", "Star with no predicate bounds",
              "Star with more predicate variables than entries")
    for m in ("approx", "exact")]
    + [pytest.param(f, "approx", id=f"{f}-approx")
       for f in ("coupled Zono", "Zono with more generators than entries")])
def test_sign_on_a_coupled_set_holds_the_exact_image(family, method):
    """Sign on sets whose entries share predicate variables: a coupled Star,
    one with a binding row (entry 0's interval bound is -1/2, its true
    minimum 0, so the LP range decides), the two halves of the Star cut at
    entry 1 = 0 (two input sets at once, each the result of an earlier
    split), the coupled Star with no predicate bounds (its ranges come from
    the padded LP enclosure, and the approx result's new variables are
    bounded by rows alone), a Star with more predicate variables than
    entries and a binding row, a coupled Zono and a Zono with more
    generators than entries; the last two families of two entries range
    well beyond [-1, 1]. Exact image in fractions. Catches a split child
    given the wrong sign or the wrong half-space, a neuron that the LP range
    decides inside a child split anyway or given the wrong side, a row that
    ties a free output variable to the input (the secant coupling of the
    old C2 bug), a Zono bound that leaves out generator columns, and the
    outputs of one input set dropped."""
    sets, points = _family(family)
    what = f"Sign on {family}, {method}"
    out = reach_one_layer(_sign_layer(), sets, method, what)
    pieces = [p for s in sets for p in R.sign(_input_pieces(s))]
    _assert_holds_exact_image(pieces, out, what)
    assert_covered(out, replay_torch(_sign_layer(), points[::2],
                                     (1, points[0].size)), what)


@pytest.mark.parametrize("method", ["approx", "exact"])
@pytest.mark.parametrize("lp_solver", ["SCS"])
def test_sign_with_another_lp_solver_holds_the_exact_image(request,
                                                           lp_solver, method):
    """Sign with ``lp_solver`` named on two Stars whose rows make a range the
    kernel asks for strictly narrower than the predicate box gives: the Star
    whose entry 0 has true minimum 0 (range [0, 1], box [-1/2, 1]) and the
    Star whose row holds entry 0 in [1/4, 1] (box [-1, 1]). The
    ``inexact_scs`` fixture makes every SCS optimum 2e-5 inward, as SCS is
    at its default tolerance. A sign decided from an inward range misses an
    output (sign 0 or the other side), so the ranges must come from HiGHS,
    padded, whatever the caller names (``_bound_lp_solver``). The scipy
    solvers (``linprog``, ``highs``) take the path of the default solver,
    which the other Star cells run. Exact image in fractions, read at the
    vertices of every piece. Catches a range taken from the solver the
    caller names, unpadded (first Star), and an LP that reads the rows with
    the wrong sign, which puts entry 0 of the second Star in [-1, -1/4] and
    decides its sign -1."""
    request.getfixturevalue("inexact_scs")
    for family in ("Star whose entry 0 has true minimum 0",
                   "Star with a row that decides entry 0"):
        sets, _ = _family(family)
        what = f"Sign on {family}, {method}, {lp_solver}"
        out = reach_one_layer(_sign_layer(), sets, method, what,
                              lp_solver=lp_solver)
        pieces = [p for s in sets for p in R.sign(_input_pieces(s))]
        _assert_holds_exact_image(pieces, out, what)


_SPLIT_ON_THE_FACE = known_bug(
    18, "exact Sign splits a neuron whose true range ends exactly at 0 (the "
        "LP range, padded outward, reaches past 0) into a -1 or +1 child "
        "that holds only the face x = 0, where the sign is 0, so the exact "
        "method returns sign vectors that no input has (ACT-9)")


@pytest.mark.parametrize("family", [
    "coupled Star",
    "two halves of a split",
    "Star with no predicate bounds",
    pytest.param("Star whose entry 0 has true minimum 0",
                 marks=_SPLIT_ON_THE_FACE),
    pytest.param("Star whose entry 0 has true maximum 0",
                 marks=_SPLIT_ON_THE_FACE),
])
def test_sign_exact_on_a_coupled_star_is_exact(family):
    """method='exact' promises the exact image up to float64 rounding
    (DESIGN principle 4): every vertex of every returned star has a
    preimage. The coupled Star, the two halves and the coupled Star with no
    predicate bounds pass (in the halves the spurious +1 child on the cut
    face x1 = 0 is matched by the other half, so the union is still exact).
    Where a binding row makes a neuron's true minimum exactly 0 while its
    interval bound is past 0, the split step makes a -1 child on the face
    x = 0 alone, whose output no input has; with the true maximum exactly 0
    the +1 child (the other branch of the split step). Catches a child kept
    for a side the input does not reach, and an output sign fixed for the
    wrong child."""
    sets, _ = _family(family)
    what = f"exact Sign on {family}"
    out = reach_one_layer(_sign_layer(), sets, "exact", what)
    pieces = [p for s in sets for p in R.sign(_input_pieces(s))]
    if R.escapes(pieces, *[R.star(s) for s in out]):
        pytest.fail(f"{what}: the result is not even sound; see "
                    f"test_sign_on_a_coupled_set_holds_the_exact_image")
    _assert_no_point_without_preimage(pieces, out, what)


# A (2, 2, 3) image: three pixels cross 0, one has upper bound exactly 0, the
# other eight are points of distinct values (two of them 0), so a pixel read
# in the wrong place moves an output.
_IMG_LO = np.array([[[-0.75, 0.375, 0.0], [-1.0, 0.5, -0.25]],
                    [[0.125, -1.25, -0.25], [0.0, -1.5, -0.625]]])
_IMG_HI = np.array([[[0.25, 0.375, 0.0], [-1.0, 0.5, 0.25]],
                    [[0.125, 0.0, 0.75], [0.0, -1.5, -0.625]]])
_CHW = _IMG_LO.shape


@pytest.mark.parametrize("parts", [1, 2])
@pytest.mark.parametrize("method", ["approx", "exact"])
def test_sign_on_an_image_star_keeps_the_image(method, parts):
    """Sign on an ImageStar of shape (2, 2, 3), one set and two parts at
    once: every result is an ImageStar of the input's height, width and
    channels and holds the exact image in the stored (height, width,
    channel) order. Catches a result converted back with height and width
    exchanged (a flat star of the same entries, so only the shape and a
    later spatial layer see it) or left flat."""
    sets = ([image_set("ImageStar", _IMG_LO, _IMG_HI)] if parts == 1
            else image_split_cover("ImageStar", _IMG_LO, _IMG_HI, parts))
    what = f"Sign on ImageStar ({parts} sets), {method}"
    out = reach_one_layer(_sign_layer(), sets, method, what)
    shapes = {(type(s), s.height, s.width, s.num_channels) for s in out}
    assert shapes == {(ImageStar, 2, 3, 2)}, (
        f"{what}: results {shapes}, expected ImageStar 2x3x2")
    _assert_holds_product_image(R.sign, sets, out, what,
                                pieces=method == "exact")
    if method == "exact":
        _assert_product_exact(R.sign, sets, out, what)
    assert_covered(out, replay_torch(_sign_layer(),
                                     samples(_IMG_LO, _IMG_HI, n=_N),
                                     (1,) + _CHW), what, out_chw=_CHW)


@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_sign_on_a_hexatope_or_octatope_is_refused(kind):
    """No Sign rule exists for these set types: the call must raise
    ``NotImplementedError``, never return the input or a box."""
    with pytest.raises(NotImplementedError):
        reach_layer(_sign_layer(), [flat_set(kind, _LO, _HI)], "approx")


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_a_function_node_other_than_sign_is_refused_by_the_sign_router(kind):
    """onnx2torch builds the same module class, ``OnnxFunction``, for Abs,
    Exp, Log and the other function nodes; the dispatcher must send only
    the one that wraps ``torch.sign`` to the Sign kernels. Abs has no rule
    and must be refused with ``NotImplementedError`` in the Star, Zono and
    Box lanes, each of which asks ``_is_sign_layer``. Catches a router that
    accepts any wrapped function: Abs answered as Sign gives [-1, 1] where
    |x| reaches 3/2."""
    with pytest.raises(NotImplementedError):
        reach_layer(OnnxFunction("Abs"), [flat_set(kind, _LO, _HI)],
                    "approx")


_EXACT_IS_THE_INTERVAL_RULE = known_bug(
    18, "reach_layer with method='exact' on a Zono, an ImageZono or a Box "
        "answers Sign with the interval rule of the approx method under the "
        "name exact, with no warning: the Zono and Box lanes of the "
        "dispatcher do not read the method (REACH-8 on the Zono lane and "
        "the Box lane); the model entry refuses 'exact' on these sets")

# Entry 0 crosses 0, entry 1 is positive: the image is {-1, 0, 1} x {1}.
_X_LO, _X_HI = np.array([-0.5, 0.25]), np.array([0.5, 0.75])


@pytest.mark.parametrize("kind,entries", [
    pytest.param("Zono", "crossing", marks=_EXACT_IS_THE_INTERVAL_RULE),
    pytest.param("Box", "crossing", marks=_EXACT_IS_THE_INTERVAL_RULE),
    ("Zono", "decided"), ("Box", "decided")])
def test_sign_exact_on_a_zono_or_a_box_refuses_or_is_the_image(kind,
                                                                entries):
    """DESIGN: 'exact' is exact up to float64 rounding or raises.
    ``reach_layer(Sign, sets, 'exact')`` on a Zono or a Box: a refusal
    (``NotImplementedError`` or ``ValueError``, as the model entry gives)
    keeps the promise; an answer must hold the exact image and no point
    without a preimage. With a crossing entry the image is three points,
    which no Zono or Box is, and today the interval [-1, 1] is returned; the
    same Zono lane serves an ImageZono, so this marked case stands for it.
    With every entry decided (positive, negative, the point 0) the image is
    one point, which the interval rule returns exactly: the passing
    neighbour. Catches the exact method returning an over-approximation
    under its name, and a decided entry given the wrong constant."""
    lo, hi = ((_X_LO, _X_HI) if entries == "crossing"
              else (_DECIDED_LO[:3], _DECIDED_HI[:3]))
    sets = [flat_set(kind, lo, hi)]
    what = f"exact Sign on a {kind} ({entries})"
    try:
        out = reach_one_layer(_sign_layer(), sets, "exact", what)
    except (NotImplementedError, ValueError):
        return
    pieces = R.sign(R.box(lo, hi))
    _assert_holds_exact_image(pieces, out, what)
    _assert_no_point_without_preimage(pieces, out, what)


_DIMENSION_ZERO_DROPPED = known_bug(
    7, "the Sign Star kernels skip a Star of dimension 0 and return no set "
       "for it (sign_reach._sign_star_approx and _sign_star_exact), although "
       "its one member, the empty vector, has the empty output: the result "
       "says the input is empty (registry EMPTY-6)")


@pytest.mark.parametrize("method", [
    pytest.param(m, marks=_DIMENSION_ZERO_DROPPED) for m in ("approx",
                                                             "exact")])
def test_sign_on_a_star_of_dimension_zero_is_not_dropped(method):
    """A Star of dimension 0 over a non-empty predicate polytope is not
    empty (torch.sign maps a tensor of 0 elements to a tensor of 0
    elements), so ``reach_layer`` must return a set for it, or refuse it
    with ``ValueError``; no sets at all is the answer "empty", which a
    specification check reads as a proof. Catches a kernel that drops an
    input set it has not certified empty. Neighbours that pass today:
    ``test_sign_on_a_product_of_intervals_holds_the_exact_image``
    (``[Star-approx-1]``, ``[Star-exact-1]``)."""
    star = Star(np.zeros((0, 2)), np.array([[1.0], [-1.0]]), np.ones((2, 1)),
                -np.ones((1, 1)), np.ones((1, 1)))
    if star.dim != 0:
        pytest.fail(f"the input Star has dimension {star.dim}, not 0")
    try:
        out = reach_layer(_sign_layer(), [star], method)
    except ValueError:
        return
    assert len(out) >= 1, (
        f"Sign {method} on a Star of dimension 0: no set returned for a "
        f"non-empty input")


# --------------------------------------------------------------------------- #
# Inside a graph, through the model entry point                                 #
# --------------------------------------------------------------------------- #
_W1 = np.array([[1.0, 0.5], [0.5, -1.0], [-0.25, 1.0]])
_B1 = np.array([0.25, 0.0, -0.5])
_W2 = np.array([[1.0, -0.5, 0.25], [0.5, 1.0, -1.0]])
_B2 = np.array([0.0, 0.125])
_IN_LO = np.array([-1.0, -0.5])
_IN_HI = np.array([0.5, 1.0])


def _sign_chain(tmp_path):
    f32 = lambda a: np.asarray(a, dtype=np.float32)      # noqa: E731
    return save_onnx(
        tmp_path,
        [helper.make_node("Gemm", ["X", "W1", "B1"], ["H"], transB=1),
         helper.make_node("Sign", ["H"], ["A"]),
         helper.make_node("Gemm", ["A", "W2", "B2"], ["Y"], transB=1)],
        [1, 2], [1, 2], {"W1": f32(_W1), "B1": f32(_B1), "W2": f32(_W2),
                         "B2": f32(_B2)}, name="sign_chain")


@pytest.mark.parametrize("kind,method", [
    ("Star", "approx"), ("Star", "exact"), ("Zono", "approx"),
    ("Box", "approx"), ("Star with a binding row", "approx"),
    ("Star with a binding row", "exact")])
def test_sign_between_two_affine_layers_holds_the_exact_image(tmp_path,
                                                              kind, method):
    """Gemm, Sign, Gemm loaded with ``load_onnx`` and reached through
    ``reach_pytorch_model``: the free output variables Sign adds are consumed
    by the next layer. Hidden neuron 1 is 0 on a line through the input box
    and the grid of samples meets it. Exact image in fractions (both
    directions for 'exact'; no hidden range here ends exactly at 0) and
    onnxruntime samples by LP. Catches a relaxation
    that ties the sign to its input, a constant output written to the wrong
    row, and a split child dropped inside a graph."""
    path = _sign_chain(tmp_path)
    if kind == "Star with a binding row":
        rows = [(np.array([1.0, 1.0]), 1.5)]
        C = np.vstack([np.eye(2), -np.eye(2)] + [r for r, _ in rows])
        d = np.concatenate([np.ones(4), [1.5]]).reshape(-1, 1)
        centre, half = (_IN_LO + _IN_HI) / 2, (_IN_HI - _IN_LO) / 2
        input_set = Star(np.hstack([centre[:, None], np.diag(half)]), C, d,
                         -np.ones((2, 1)), np.ones((2, 1)))
        points = [centre + half * a for a in _grid(2, 9)
                  if a.sum() <= 1.5]
    else:
        input_set = flat_set(kind, _IN_LO, _IN_HI)
        points = samples(_IN_LO, _IN_HI,
                         n=N_SAMPLES if method == "approx" else _N)
    what = f"Gemm, Sign, Gemm on {kind}, {method}"
    out = reach_model(load_onnx(path), input_set, method, what)
    pieces = R.affine(R.sign(R.affine(_input_pieces(input_set), _W1, _B1)),
                      _W2, _B2)
    _assert_holds_exact_image(pieces, out, what)
    if method == "exact":
        _assert_no_point_without_preimage(pieces, out, what)
    assert_covered(out, replay_onnx(path, points, [1, 2]), what)


def test_sign_between_two_convolutions_covers_onnxruntime(tmp_path):
    """Conv (1x1, mixing the channels), Sign, Conv (2x1 kernel) on a
    (2, 3, 5) ImageStar with its own interval per pixel, loaded with
    ``load_onnx``: every onnxruntime output is in a returned set, compared
    in the stored order of the (1, 2, 5) result. Catches a Sign result whose
    height and width are exchanged, which the next convolution reads in the
    wrong place."""
    shape = [1, 2, 3, 5]
    path = save_onnx(
        tmp_path,
        [helper.make_node("Conv", ["X", "K1"], ["H"], kernel_shape=[1, 1]),
         helper.make_node("Sign", ["H"], ["A"]),
         helper.make_node("Conv", ["A", "K2"], ["Y"], kernel_shape=[2, 1])],
        shape, [1, 1, 2, 5],
        {"K1": np.array([[[[1.0]], [[0.5]]], [[[-0.5]], [[1.0]]]],
                        dtype=np.float32),
         "K2": np.array([[[[1.0], [0.5]], [[-0.25], [1.0]]]],
                        dtype=np.float32)}, name="sign_conv")
    rng = np.random.default_rng(3)
    lo = rng.integers(-8, 8, size=(2, 3, 5)) / 4.0
    hi = lo + rng.integers(1, 6, size=(2, 3, 5)) / 4.0
    what = "Conv, Sign, Conv on ImageStar"
    out = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                      "approx", what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), shape), what,
                   out_chw=(1, 2, 5))


# A (2, 2, 3) image whose pixels are points except two intervals: behind the
# 1x1 convolution, hidden pixel (0, 0) of channel 0 crosses 0 (it is 0 at
# input value -1/4, on the sample grid) and hidden pixel (1, 0) of channel 1
# has upper bound exactly 0 (reached at input value 1/2); hidden pixel (0, 2)
# of channel 1 is the point 0. Six exact pieces at most.
_EX_LO = np.array([[[-0.5, 0.25, 1.0], [0.5, -0.75, 0.0]],
                   [[0.5, -1.0, 0.5], [0.25, 0.5, 1.0]]])
_EX_HI = np.array([[[0.5, 0.25, 1.0], [0.75, -0.75, 0.0]],
                   [[0.5, -1.0, 0.5], [0.25, 0.5, 1.0]]])


def test_sign_exact_between_two_convolutions_covers_onnxruntime(tmp_path):
    """Conv (1x1, mixing the channels), Sign, Conv (2x1 kernel) on a
    (2, 2, 3) ImageStar under ``method='exact'``, loaded with ``load_onnx``:
    every onnxruntime output (128 samples, the zeros of the crossing hidden
    pixels on the grid) is in a returned set, compared in the stored order
    of the (1, 1, 3) result. Catches a split child converted back to an
    ImageStar with height and width exchanged or left flat, which the next
    convolution reads in the wrong place, and a child dropped inside a
    graph."""
    shape = [1, 2, 2, 3]
    path = save_onnx(
        tmp_path,
        [helper.make_node("Conv", ["X", "K1"], ["H"], kernel_shape=[1, 1]),
         helper.make_node("Sign", ["H"], ["A"]),
         helper.make_node("Conv", ["A", "K2"], ["Y"], kernel_shape=[2, 1])],
        shape, [1, 1, 1, 3],
        {"K1": np.array([[[[1.0]], [[0.5]]], [[[-0.5]], [[1.0]]]],
                        dtype=np.float32),
         "K2": np.array([[[[1.0], [0.5]], [[-0.25], [1.0]]]],
                        dtype=np.float32)}, name="sign_conv_exact")
    what = "Conv, Sign, Conv on ImageStar, exact"
    out = reach_model(load_onnx(path), image_set("ImageStar", _EX_LO, _EX_HI),
                      "exact", what)
    if len(out) > 6:
        pytest.fail(f"{what}: {len(out)} sets; the input has at most 6 "
                    f"exact pieces")
    assert_covered(out, replay_onnx(path, samples(_EX_LO, _EX_HI), shape),
                   what, out_chw=(1, 1, 3))


class _TorchSign(nn.Module):
    """Gemm, then Sign spelled as a torch function or a tensor method."""

    def __init__(self, spelling):
        super().__init__()
        self.spelling = spelling
        self.lin = nn.Linear(2, 3).double()
        with torch.no_grad():
            self.lin.weight.copy_(torch.as_tensor(_W1))
            self.lin.bias.copy_(torch.as_tensor(_B1))

    def forward(self, x):
        h = self.lin(x)
        return torch.sign(h) if self.spelling == "torch.sign" else h.sign()


@pytest.mark.parametrize("spelling", ["torch.sign", "Tensor.sign"])
def test_sign_in_a_torch_model_refuses_or_holds_every_output(spelling):
    """A torch model whose forward calls ``torch.sign`` or ``x.sign()`` (an
    ``OnnxFunction`` module inside a torch model is traced to the same
    call): today the graph walk has no rule for the call and refuses with
    ``NotImplementedError``. An answer must hold every true output of the
    model (torch in float64, dyadic data, the samples meet the line where
    hidden neuron 1 is 0), under each method. Catches the call mapped to a
    module of another function, or answered by the sets of the node before
    it."""
    model = _TorchSign(spelling)
    outputs = replay_torch(model, samples(_IN_LO, _IN_HI, n=_N), (1, 2))
    for method in ("approx", "exact"):
        what = f"Sign as {spelling} in a torch model, {method}"
        try:
            out = reach_model(model, flat_set("Star", _IN_LO, _IN_HI),
                              method, what)
        except NotImplementedError:
            continue
        assert_covered(out, outputs, what)
