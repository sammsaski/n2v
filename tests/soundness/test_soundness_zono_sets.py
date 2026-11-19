"""Soundness cells for the Zono and ImageZono set operations.

What the file covers: the constructor and ``from_bounds``;
``minkowski_sum``; ``convex_hull``; ``convex_hull_with_linear_transform``;
``get_oriented_box``; ``order_reduction_box`` and ``reduce_order``;
``change_vars_precision``; ``intersect_half_space``; ``contains``; the bound
queries (``get_bounds``, ``get_ranges``, ``estimate_ranges``, ``get_range``,
``get_box``, ``get_interval_hull``), also on float32 arrays and on a box with
an infinite end; ``to_star``; ``to_image_zono`` and ``to_image_star``
followed by spatial layers; ``ImageZono.from_bounds``, ``to_zono``,
``get_bounds``; and the public route, a Zono and an ImageZono through
``reach_pytorch_model`` and ``verify_specification``. On a general (sheared)
zonotope ``Zono.contains`` tests the bounding box: that is the marked case of
``contracts/test_set_protocol.py`` (registry SET-1).

Oracle. A zonotope ``c + V a, a in [-1, 1]^m`` is read from its stored
arrays (``rational.star`` through ``membership.rows``), never through a
method of the set. Its vertices are the points ``c + V s`` for the sign
vectors ``s``, computed here in ``fractions.Fraction`` from the stored floats;
membership of a rational point in a computed set is decided exactly
(``rational.Membership``, zero slack). Ranges are compared with the exact
``c +- sum |V|``. Where a cell runs a layer, the true outputs come from torch
in float64 on dyadic inputs and weights (exact), and membership in the
returned image sets is decided by the LP oracle (``assert_covered``, slack
``LP_TOL``) with the outputs converted to the stored order.

Data. Most cells use dyadic data, on which the library's float arithmetic is
exact, so a set that misses an exact point is wrong in its formula, not in
its rounding. The cells marked as known defects use float64 data on purpose:
there the operation must still hold the exact set and does not.

Layout. A flat Zono that stands for an image is in the image set's stored
order (height, width, channel), which is how ``to_image_zono`` and
``to_image_star`` read it; true outputs are computed in the network's
(channel, height, width) order and converted with ``membership.hwc``.
"""

import itertools
import warnings
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import HalfSpace, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.verify_specification import verify_specification
from tests._oracles import rational as R
from tests._oracles.membership import LP_TOL, distance, hwc
from tests.cell_support import (assert_covered, image_set, pixel_box,
                                reach_model, reach_one_layer, replay_torch,
                                samples, split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug


# --------------------------------------------------------------------------- #
# The exact reference                                                           #
# --------------------------------------------------------------------------- #
def _vertices(z):
    """``c + V s`` for every sign vector ``s``, exact, from the stored
    arrays: points whose hull is the zonotope."""
    c = [Fraction(float(v)) for v in np.asarray(z.c, dtype=np.float64).ravel()]
    V = np.asarray(z.V, dtype=np.float64).reshape(len(c), -1)
    G = [[Fraction(float(v)) for v in row] for row in V]
    return [tuple(ci + sum((g * s for g, s in zip(row, signs)), Fraction(0))
                  for ci, row in zip(c, G))
            for signs in itertools.product((-1, 1), repeat=V.shape[1])]


def _exact_bounds(z):
    """``c - sum |V|`` and ``c + sum |V|`` per entry, exact."""
    c = np.asarray(z.c, dtype=np.float64).ravel()
    V = np.asarray(z.V, dtype=np.float64).reshape(c.size, -1)
    radius = [sum((abs(Fraction(float(v))) for v in row), Fraction(0))
              for row in V]
    centre = [Fraction(float(v)) for v in c]
    return ([a - r for a, r in zip(centre, radius)],
            [a + r for a, r in zip(centre, radius)])


def _assert_holds(computed, points, what):
    """Every rational point is a member of the computed set, exactly."""
    member = R.Membership(R.star(computed))
    missed = [p for p in points if not member(p)]
    assert not missed, (
        f"{what}: {len(missed)} of {len(points)} points of the true set are "
        f"not members of the result; first {[float(v) for v in missed[0]]}")


def _assert_outward(lo, hi, exact_lo, exact_hi, what):
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    for i, (a, b, ea, eb) in enumerate(zip(lo, hi, exact_lo, exact_hi)):
        assert Fraction(float(a)) <= ea, (
            f"{what}: lower bound {a!r} of entry {i} is above the exact "
            f"minimum {float(ea)!r}")
        assert Fraction(float(b)) >= eb, (
            f"{what}: upper bound {b!r} of entry {i} is below the exact "
            f"maximum {float(eb)!r}")


def _dyadic(rng, shape, scale=8):
    return rng.integers(-2 * scale, 2 * scale + 1, size=shape) / scale


# --------------------------------------------------------------------------- #
# Constructor                                                                   #
# --------------------------------------------------------------------------- #
def test_constructor_reads_integer_arrays_and_a_single_generator():
    """``Zono`` built from integer arrays and a one-dimensional generator:
    the set is the segment ``c + t v``, ``t in [-1, 1]``, with float entries,
    so both ends are members and the affine image of both ends by a
    non-integer map is in the mapped set. Catches an integer dtype kept (the
    map then truncates) and a one-dimensional V read as a row."""
    z = Zono(np.array([1, -2]), np.array([3, 1]))
    assert z.c.dtype.kind == "f" and z.V.dtype.kind == "f"
    _assert_holds(z, [(4, -1), (-2, -3)], "Zono(int c, 1-D int V)")
    W = np.array([[0.5, -0.25], [0.125, 1.0], [-1.0, 0.75]])
    b = np.array([0.25, -0.5, 0.0])
    ends = [(Fraction(4), Fraction(-1)), (Fraction(-2), Fraction(-3))]
    images = [tuple(R.dot([Fraction(v) for v in row], e) + Fraction(bi)
                    for row, bi in zip(W, b)) for e in ends]
    _assert_holds(z.affine_map(W, b), images, "affine_map of the segment")


_ARRAY_OF_THREE_AXES = known_bug(
    15, "the Zono constructor checks the height and the column count of its "
        "arrays only, so a generator array or a centre with a third axis is "
        "accepted (with V of shape (2, 2, 2) get_bounds returns four bounds "
        "for a set of two entries)", raises=DID_NOT_RAISE)


@pytest.mark.parametrize("c, V", [
    (np.zeros((3, 1)), np.zeros((2, 2))),
    (np.zeros((3, 2)), np.zeros((3, 2))),
    (np.zeros((3, 1)), None),
    pytest.param(np.zeros((2, 1)), np.ones((2, 2, 2)), id="V-of-three-axes",
                 marks=_ARRAY_OF_THREE_AXES),
    pytest.param(np.zeros((2, 1, 3)), np.ones((2, 4)), id="c-of-three-axes",
                 marks=_ARRAY_OF_THREE_AXES)])
def test_constructor_refuses_inconsistent_arrays(c, V):
    """A centre and generators of different heights, a centre with two
    columns, only one of the two arrays, or an array with a third axis (the
    docstring reads the centre as ``(n,)`` or ``(n, 1)`` and the generators
    as ``(n, m)``): ``ValueError``. The arrays with a third axis are accepted
    today and are the marked cases; the first three are their passing
    neighbours."""
    with pytest.raises(ValueError):
        Zono(c, V)


# --------------------------------------------------------------------------- #
# from_bounds                                                                   #
# --------------------------------------------------------------------------- #
def _box_corners(lo, hi):
    """Every corner of the box ``[lo, hi]``, exact (a zero-width coordinate
    has one value)."""
    return list(itertools.product(*[
        sorted({Fraction(float(a)), Fraction(float(b))})
        for a, b in zip(lo, hi)]))


def test_from_bounds_is_exactly_the_box():
    """``Zono.from_bounds`` of a dyadic box whose intervals reach beyond plus
    and minus 1 and are wider than 2, of both signs, with one zero-width
    coordinate: every corner of the box is a member of the zonotope and every
    vertex of the zonotope lies in the box, both exactly, so the zonotope is
    the box (its docstring: the midpoint as centre and one generator per
    entry of the half-width). Catches a half-width halved (the zonotope is
    the middle of the box) or doubled, a centre off the midpoint, one entry's
    interval read for another, and a width given to the zero-width
    coordinate."""
    lo = np.array([-6.5, 2.25, -3.0, -0.75, -9.0])
    hi = np.array([1.5, 7.0, -3.0, 4.25, -2.5])
    z = Zono.from_bounds(lo, hi)
    _assert_holds(z, _box_corners(lo, hi), "Zono.from_bounds")
    box = list(zip((Fraction(float(v)) for v in lo),
                   (Fraction(float(v)) for v in hi)))
    outside = [v for v in _vertices(z)
               if not all(a <= x <= b for (a, b), x in zip(box, v))]
    assert not outside, (
        f"Zono.from_bounds: {len(outside)} of {2 ** z.V.shape[1]} vertices of "
        f"the zonotope lie outside the box; first "
        f"{[float(x) for x in outside[0]]}")


def test_from_bounds_holds_every_corner_of_a_non_dyadic_box():
    """The exact-fraction rounding case of ``from_bounds``: on the intervals
    [-2.7, 0.3], [1/3, 2.9], [-123.456, -0.001] and [0.1, 0.2] a midpoint and
    a half-width computed by round to nearest leave out one end of each.
    Every corner of the box, as the exact rationals of the float bounds, is a
    member of ``Zono.from_bounds`` and of ``ImageZono.from_bounds`` (the same
    numbers as a 1 x 2 x 2 image), exactly. Catches a centre or a half-width
    rounded to nearest instead of outward."""
    lo = np.array([-2.7, 1 / 3, -123.456, 0.1])
    hi = np.array([0.3, 2.9, -0.001, 0.2])
    corners = _box_corners(lo, hi)
    _assert_holds(Zono.from_bounds(lo, hi), corners,
                  "Zono.from_bounds of a non-dyadic box")
    _assert_holds(ImageZono.from_bounds(lo.reshape(1, 2, 2),
                                        hi.reshape(1, 2, 2), 1, 2, 2),
                  corners, "ImageZono.from_bounds of a non-dyadic box")


@pytest.mark.parametrize("bounds", [
    "shapes-differ",
    pytest.param("lower-above-upper", marks=known_bug(
        8, "Zono.from_bounds (and ImageZono.from_bounds through it) accepts "
           "a lower bound above the upper bound, where Box and "
           "Star.from_bounds raise ValueError: the box is empty and the "
           "zonotope returned is not (for [1, 0] it is [0, 1], whose centre "
           "lies outside the box)", raises=DID_NOT_RAISE))])
def test_from_bounds_refuses_bounds_that_are_not_a_box(bounds):
    """Bounds of different shapes, or a lower bound above the upper bound in
    one entry (an empty box, which no zonotope is): ``ValueError``, as
    ``Box`` and ``Star.from_bounds`` raise for the same bounds. The second is
    accepted today and is the marked case; the first is its passing
    neighbour."""
    lo, hi = {"shapes-differ": (np.zeros(3), np.ones(2)),
              "lower-above-upper": (np.array([1.0, -2.0, 0.5]),
                                    np.array([0.0, 2.0, 0.75]))}[bounds]
    with pytest.raises(ValueError):
        Zono.from_bounds(lo, hi)


# --------------------------------------------------------------------------- #
# affine_map                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("W", [
    pytest.param(np.array([[0.5, -1.0]]), id="row-matrix"),
    pytest.param(np.array([0.5, -1.0]), id="one-dimensional", marks=known_bug(
        23, "Zono.affine_map reads W.shape[1] without checking that W has two "
            "axes, so a one-dimensional weight raises IndexError ('tuple index "
            "out of range') where a ValueError, or the answer as one row that "
            "Box.affine_map gives, is due", raises=IndexError))])
def test_affine_map_by_a_single_row_holds_every_image(W):
    """``affine_map`` by a weight of one row, given as a (1, 2) matrix or as
    a one-dimensional array (which ``Box.affine_map`` reads as that row):
    the one-dimensional form may be refused with ``ValueError`` (the
    docstring asks for an (m, n) matrix); otherwise the result holds the
    image of every vertex of the zonotope, exactly. The one-dimensional form
    raises ``IndexError`` today and is the marked case; the (1, 2) matrix is
    its passing neighbour and catches the generators mapped by the absolute
    value of W (the sign relation between generators lost)."""
    z = Zono(np.array([[1.5], [-2.0]]),
             np.array([[0.5, 0.25, -1.0], [-1.0, 0.75, 0.5]]))
    try:
        out = z.affine_map(W)
    except ValueError:
        if W.ndim == 2:
            raise
        return
    _assert_holds(out, _image(np.atleast_2d(W), _vertices(z)),
                  f"affine_map by a weight of shape {W.shape}")


# --------------------------------------------------------------------------- #
# minkowski_sum                                                                 #
# --------------------------------------------------------------------------- #
def _two_zonos(seed, m1=3, m2=3):
    rng = np.random.default_rng(seed)
    return (Zono(_dyadic(rng, (3, 1)), _dyadic(rng, (3, m1), 4)),
            Zono(_dyadic(rng, (3, 1)), _dyadic(rng, (3, m2), 4)))


@pytest.mark.parametrize("seed", [0])
def test_minkowski_sum_holds_every_sum_of_points(seed):
    """Two 3-dimensional zonotopes with the same number of generators in
    different directions (dyadic): every sum of a vertex of the first and a
    vertex of the second is a member of the sum, exactly; and every vertex
    of the result is such a sum (decided by an exact feasibility problem on
    the operands' own arrays). Catches generator matrices added entry by
    entry instead of concatenated (the coefficients of the two operands tied
    together), a dropped operand and a wrong centre."""
    z1, z2 = _two_zonos(seed)
    s = z1.minkowski_sum(z2)
    sums = [tuple(a + b for a, b in zip(u, w))
            for u in _vertices(z1) for w in _vertices(z2)]
    _assert_holds(s, sums, "minkowski_sum")
    c1, G1 = R.star(z1)[:2]
    c2, G2 = R.star(z2)[:2]
    G = [r1 + r2 for r1, r2 in zip(G1, G2)]
    m = len(G[0])
    unit = [[Fraction(int(i == j)) for j in range(m)] for i in range(m)]
    box = unit + [[-v for v in row] for row in unit]
    for y in _vertices(s):
        rhs = [yi - a - b for yi, a, b in zip(y, c1, c2)]
        assert R.feasible(G, rhs, box, [Fraction(1)] * (2 * m)) is not None, (
            f"minkowski_sum: the vertex {[float(v) for v in y]} of the result "
            f"is not a sum of points of the operands")


def test_minkowski_sum_refuses_a_wrong_operand():
    """A non-Zono operand is a ``TypeError``, a dimension mismatch a
    ``ValueError``."""
    z = Zono(np.zeros((2, 1)), np.eye(2))
    with pytest.raises(TypeError):
        z.minkowski_sum(np.zeros((2, 1)))
    with pytest.raises(ValueError):
        z.minkowski_sum(Zono(np.zeros((3, 1)), np.eye(3)))


# --------------------------------------------------------------------------- #
# convex_hull                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0])
def test_convex_hull_holds_both_operands(seed):
    """Two zonotopes with different centres and different numbers of
    generators (dyadic): every vertex of each operand is a member of the
    hull, exactly. Catches the generator that spans the difference of the
    centres left out (the hull is then the Minkowski sum of the generators
    around the midpoint and misses both operands), a wrong midpoint, and a
    dropped operand."""
    z1, z2 = _two_zonos(seed, 2, 3)
    h = z1.convex_hull(z2)
    _assert_holds(h, _vertices(z1) + _vertices(z2), "convex_hull")


@known_bug(8, "Zono.convex_hull computes the midpoint 0.5 (c1 + c2) and "
              "the half-difference 0.5 (c1 - c2) by round to nearest, so "
              "the hull can leave out an operand (centres -1 and 2^-60, no "
              "generators: the hull is [-1, 0] and misses 2^-60)")
def test_convex_hull_holds_an_operand_whose_centre_rounds_away():
    """The promise of ``convex_hull`` (an over-approximation of the hull)
    for any float operands: both operands are held, exactly. The neighbour
    with dyadic centres above passes with the same membership code."""
    z1 = Zono(np.array([[-1.0]]), np.zeros((1, 1)))
    z2 = Zono(np.array([[2.0 ** -60]]), np.zeros((1, 1)))
    _assert_holds(z1.convex_hull(z2), _vertices(z1) + _vertices(z2),
                  "convex_hull of the points -1 and 2^-60")


def test_convex_hull_refuses_a_wrong_operand():
    """A non-Zono operand is a ``TypeError``, a dimension mismatch a
    ``ValueError``."""
    z = Zono(np.zeros((2, 1)), np.eye(2))
    with pytest.raises(TypeError):
        z.convex_hull(np.zeros((2, 1)))
    with pytest.raises(ValueError):
        z.convex_hull(Zono(np.zeros((3, 1)), np.eye(3)))


# --------------------------------------------------------------------------- #
# convex_hull_with_linear_transform                                             #
# --------------------------------------------------------------------------- #
_L = np.array([[0.5, -1.0, 0.25], [0.75, 0.0, -0.5], [-0.25, 1.25, 1.0]])


def _hull_operands(centre):
    rng = np.random.default_rng(5)
    c = np.zeros((3, 1)) if centre == "origin" else _dyadic(rng, (3, 1))
    return Zono(c, _dyadic(rng, (3, 3), 4))


def _image(L, points):
    L = [[Fraction(float(v)) for v in row] for row in L]
    return [tuple(R.dot(row, p) for row in L) for p in points]


@pytest.mark.parametrize("centre", ["away-from-origin", "origin"])
def test_hull_with_linear_transform_holds_the_set_and_its_image(centre):
    """``convex_hull_with_linear_transform`` with a square, non-symmetric
    dyadic L that has negative entries: every vertex of Z and of L Z is a
    member of the result, exactly. Catches a second mixing matrix I + L
    instead of I - L (the result then holds neither Z nor L Z) and, for a
    centre away from the origin, a missing generator (I - L) c."""
    z = _hull_operands(centre)
    verts = _vertices(z)
    _assert_holds(z.convex_hull_with_linear_transform(_L),
                  verts + _image(_L, verts),
                  f"convex_hull_with_linear_transform, centre {centre}")


@known_bug(8, "Zono.convex_hull_with_linear_transform computes I + L, "
              "I - L, the products and the halves by round to nearest, so "
              "on float64 data the result leaves out vertices of Z or of "
              "L Z (47 of 50 random cases)")
def test_hull_with_linear_transform_holds_the_set_on_float64_data():
    """The same promise with a float64 zonotope and float64 L (five random
    cases, seed 3): every vertex of Z and of L Z is held, exactly. The
    dyadic neighbours above pass with the same membership code."""
    rng = np.random.default_rng(3)
    for k in range(5):
        z = Zono(rng.uniform(-2, 2, (2, 1)), rng.uniform(-1, 1, (2, 2)))
        L = rng.uniform(-1, 1, (2, 2))
        verts = _vertices(z)
        _assert_holds(z.convex_hull_with_linear_transform(L),
                      verts + _image(L, verts),
                      f"convex_hull_with_linear_transform, case {k}")


@pytest.mark.parametrize("L", [np.ones((3, 2)), np.eye(2)])
def test_hull_with_linear_transform_refuses_a_wrong_matrix(L):
    """A non-square L, or a square L of the wrong size: ``ValueError``."""
    with pytest.raises(ValueError):
        _hull_operands("origin").convex_hull_with_linear_transform(L)


# --------------------------------------------------------------------------- #
# get_oriented_box                                                              #
# --------------------------------------------------------------------------- #
def test_oriented_box_holds_the_zonotope_when_the_frame_is_exact():
    """A zonotope with five mixed-sign generators whose rows use disjoint
    generators, so the principal directions are the axes and the SVD's Q is
    exactly the identity: every vertex is a member of the oriented box,
    exactly. Catches half-widths taken as the absolute value of the row sums
    instead of the sums of absolute values (generators of opposite sign
    cancel)."""
    V = np.array([[2.0, -1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, -0.75, 0.5, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.25]])
    z = Zono(np.array([[0.25], [-0.5], [1.0]]), V)
    _assert_holds(z.get_oriented_box(), _vertices(z), "get_oriented_box")


def test_oriented_box_holds_a_general_zonotope_by_the_lp_oracle():
    """Three dimensions, five generators of mixed signs in general position
    (dyadic): every vertex is in the oriented box by the LP oracle
    (``LP_TOL``; the exact check is the marked cell below). Catches the
    cancelling half-widths and a frame that is not the SVD's."""
    rng = np.random.default_rng(11)
    z = Zono(_dyadic(rng, (3, 1)), _dyadic(rng, (3, 5), 4))
    box = z.get_oriented_box()
    far = [v for v in _vertices(z)
           if distance(box, [float(x) for x in v]) > LP_TOL * max(
               1.0, max(abs(float(x)) for x in v))]
    assert not far, (f"get_oriented_box: {len(far)} vertices of the zonotope "
                     f"are outside it; first {[float(x) for x in far[0]]}")


@known_bug(8, "Zono.get_oriented_box takes the frame Q from a float SVD and "
              "forms Q^T V, the half-widths and Q D by round to nearest, so "
              "the box can leave out vertices of the zonotope (38 of 50 random "
              "zonotopes, decided exactly; registry SET-8)")
def test_oriented_box_holds_a_general_zonotope_exactly():
    """Ten random float64 zonotopes (seed 0): every vertex is a member of the
    oriented box, exactly. The neighbour with an exact frame above passes
    with the same membership code."""
    rng = np.random.default_rng(0)
    for k in range(10):
        z = Zono(rng.uniform(-2, 2, (3, 1)), rng.uniform(-1, 1, (3, 5)))
        _assert_holds(z.get_oriented_box(), _vertices(z),
                      f"get_oriented_box, case {k}")


# --------------------------------------------------------------------------- #
# Order reduction                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("data", ["float64"])
def test_order_reduction_holds_the_zonotope(data):
    """``order_reduction_box`` to 3, 4, 6 and 7 generators and
    ``reduce_order`` to order 1, 2 and 3 of a 3-dimensional zonotope with 6
    generators of float64 data: every vertex of the original is a member of
    the reduced zonotope, exactly (the interval hull of the removed
    generators is rounded outward). Catches the largest removed generator
    neither kept nor covered, a reduction that drops generators without
    covering them (``reduce_order(1)`` reduces here), and a no-op that
    returns another set. A radius summed by round to nearest is caught by
    ``test_order_reduction_rounds_the_removed_radius_outward``."""
    rng = np.random.default_rng(2)
    z = Zono(rng.uniform(-2, 2, (3, 1)), rng.uniform(-1, 1, (3, 6)))
    verts = _vertices(z)
    for n_max in (3, 4, 6, 7):
        _assert_holds(z.order_reduction_box(n_max), verts,
                      f"order_reduction_box({n_max}), {data}")
    for order in (1, 2, 3):
        _assert_holds(z.reduce_order(order), verts,
                      f"reduce_order({order}), {data}")


@pytest.mark.parametrize("small", [2.0 ** -54, 2.0 ** -60])
def test_order_reduction_rounds_the_removed_radius_outward(small):
    """Removed generators whose absolute row sums are not float64 numbers:
    each row of the three small generators sums to 1 + 2 small exactly
    (1 + 2^-53 or 1 + 2^-59), which a round-to-nearest sum returns as 1. The
    reduced zonotope (``order_reduction_box(3)`` and ``reduce_order(2)``)
    must still hold every vertex of the original, exactly. Catches a radius
    of the interval hull summed by round to nearest instead of rounded
    outward."""
    t = small
    V = np.array([[4.0, 0.5, 0.25, 1.0, t, t],
                  [-0.5, 3.0, 0.5, t, 1.0, t],
                  [0.25, -0.25, 2.0, t, t, -1.0]])
    z = Zono(np.array([[0.25], [-0.5], [0.125]]), V)
    verts = _vertices(z)
    _assert_holds(z.order_reduction_box(3), verts,
                  f"order_reduction_box(3), small entries {t!r}")
    _assert_holds(z.reduce_order(2), verts,
                  f"reduce_order(2), small entries {t!r}")


@pytest.mark.parametrize("order", [
    pytest.param(1, marks=known_bug(
        8, "Zono.reduce_order(1) keeps max(dim, 0) = dim generators and adds "
           "up to dim interval-hull generators, so it returns up to 2 dim "
           "generators although its docstring promises at most "
           "target_order * dim")),
    2, 3])
def test_reduce_order_keeps_its_generator_budget(order):
    """``reduce_order(k)`` of a 3-dimensional zonotope with 10 generators
    returns at most ``k * 3`` generators, the bound its docstring states.
    A unit check, not a soundness one (more generators are still sound)."""
    rng = np.random.default_rng(12)
    z = Zono(_dyadic(rng, (3, 1)), _dyadic(rng, (3, 10), 4))
    out = z.reduce_order(order)
    assert out.V.shape[1] <= order * z.dim, (
        f"reduce_order({order}) returned {out.V.shape[1]} generators for a "
        f"{z.dim}-dimensional zonotope")


@pytest.mark.parametrize("removed", [
    "zero",
    pytest.param("general", marks=known_bug(
        8, "Zono.order_reduction_box(n_max) keeps the n_max largest "
           "generators and adds up to dim interval-hull generators, so it "
           "returns up to n_max + dim generators although its docstring "
           "promises at most n_max (a 3 x 10 zonotope reduced to 9 comes "
           "back with 12, more than it had)"))])
def test_order_reduction_box_keeps_its_generator_budget(removed):
    """``order_reduction_box(4)`` of a 3-dimensional zonotope with 10
    generators returns at most 4 generators, the bound its docstring states.
    A unit check, not a soundness one (more generators are still sound; the
    soundness of the reduction is ``test_order_reduction_holds_the_zonotope``).
    With ``removed='zero'`` the six smallest generators are zero, so the
    interval hull of the removed ones is zero and its columns must be
    dropped: the passing neighbour, which catches zero hull columns kept."""
    rng = np.random.default_rng(12)
    V = _dyadic(rng, (3, 10), 4)
    if removed == "zero":
        V[:, 4:] = 0.0
        V[:, :4] += np.sign(V[:, :4]) + (V[:, :4] == 0)    # no zero kept column
    z = Zono(_dyadic(rng, (3, 1)), V)
    out = z.order_reduction_box(4)
    assert out.V.shape[1] <= 4, (
        f"order_reduction_box(4) returned {out.V.shape[1]} generators for a "
        f"zonotope with {z.V.shape[1]}")


def test_order_reduction_refuses_impossible_targets():
    """Fewer generators than dimensions, or an order below 1: ``ValueError``."""
    z = Zono(np.zeros((3, 1)), np.ones((3, 6)))
    with pytest.raises(ValueError):
        z.order_reduction_box(2)
    with pytest.raises(ValueError):
        z.reduce_order(0)


# --------------------------------------------------------------------------- #
# change_vars_precision                                                         #
# --------------------------------------------------------------------------- #
_FLOAT32_CAST = known_bug(
    8, "Zono.change_vars_precision('float32') casts the centre and the "
       "generators to float32 by round to nearest, so the float32 zonotope "
       "can leave out points of the original (every one of 50 random cases; "
       "registry SET-8)")


@pytest.mark.parametrize("precision, data", [
    ("float32", "dyadic"), ("float64", "float64"),
    pytest.param("float32", "float64", marks=_FLOAT32_CAST)])
def test_change_vars_precision_holds_the_original(precision, data):
    """The zonotope with its numbers cast to ``precision`` holds every vertex
    of the original, exactly. The float32 cast of float64 data is the marked
    cell; dyadic data (exact in float32) and the float64 cast pass with the
    same membership code. Catches a cast that changes values beyond the
    rounding (a lost generator, a truncation)."""
    rng = np.random.default_rng(4)
    if data == "dyadic":
        z = Zono(_dyadic(rng, (3, 1)), _dyadic(rng, (3, 4), 4))
    else:
        z = Zono(rng.uniform(-2, 2, (3, 1)), rng.uniform(-1, 1, (3, 4)))
    out = z.change_vars_precision(precision)
    assert out.c.dtype == np.dtype(precision)
    _assert_holds(out, _vertices(z),
                  f"change_vars_precision({precision!r}) of {data} data")


def test_change_vars_precision_refuses_an_unknown_precision():
    """An unknown precision name is a ``ValueError``."""
    with pytest.raises(ValueError):
        Zono(np.zeros((2, 1)), np.eye(2)).change_vars_precision("float16")


# --------------------------------------------------------------------------- #
# get_vertices                                                                  #
# --------------------------------------------------------------------------- #
def test_get_vertices_lists_every_vertex():
    """``get_vertices`` of a 3-dimensional zonotope with three mixed-sign
    generators (dyadic): its columns are exactly the points ``c + V s`` for
    the eight sign vectors, each a member of the set; more than 20 generators
    are refused with ``ValueError``. With four generators, one more than
    the zonotope has entries, two of the sixteen sign points are not
    vertices: every listed point must be a member, and the listed points
    must reach the exact lower and upper end of every coordinate. Catches a
    sign vector built wrongly (a bit read as 0 and +1 instead of -1 and +1),
    a generator skipped, the sign vectors counted by the entries instead of
    the generators, and the centre left out."""
    rng = np.random.default_rng(13)
    z = Zono(_dyadic(rng, (3, 1)), _dyadic(rng, (3, 3), 4))
    got = {tuple(Fraction(float(v)) for v in column)
           for column in np.asarray(z.get_vertices()).T}
    assert got == set(_vertices(z)), (
        f"get_vertices returned {len(got)} distinct points; the exact "
        f"vertices are {len(set(_vertices(z)))} and differ")
    _assert_holds(z, sorted(got), "get_vertices")
    wide = Zono(np.array([[0.5], [-1.0], [0.25]]),
                np.array([[1.0, -0.5, 0.25, 2.0], [0.5, 1.0, -1.0, -0.75],
                          [-0.25, 0.5, 1.0, 0.5]]))
    listed = {tuple(Fraction(float(v)) for v in column)
              for column in np.asarray(wide.get_vertices()).T}
    _assert_holds(wide, sorted(listed), "get_vertices, four generators")
    exact = _vertices(wide)
    for k in range(3):
        ends = (min(p[k] for p in listed), max(p[k] for p in listed))
        assert ends == (min(p[k] for p in exact), max(p[k] for p in exact)), (
            f"get_vertices of four generators: the listed points span "
            f"{[float(v) for v in ends]} in coordinate {k}, not the "
            f"zonotope's exact range")
    with pytest.raises(ValueError):
        Zono(np.zeros((2, 1)), np.ones((2, 21))).get_vertices()


# --------------------------------------------------------------------------- #
# Bound queries                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("data", ["dyadic", "float64", "float32"])
def test_bound_queries_hold_the_exact_range(data):
    """``get_bounds``, ``get_ranges``, ``estimate_ranges``, ``get_box``,
    ``get_range(i)`` and ``ImageZono.get_bounds`` of zonotopes with a full,
    mixed-sign generator matrix hold the exact range ``c +- sum |V|`` with
    zero slack, also on float64 data, where the sums are rounded outward, and
    on float32 arrays (the constructor keeps them; whether a Zono may hold
    them is an open question of the owner), where the bounds must hold the
    exact range of the stored float32 numbers; ``get_interval_hull`` holds
    every vertex, exactly. Catches the sum of a column instead of a row,
    coefficients taken in [0, 1], a signed sum, a round-to-nearest sum, and
    bounds of a float32 set rounded to nearest float32."""
    rng = np.random.default_rng(6)
    for k in range(6):
        if data == "dyadic":
            z = Zono(_dyadic(rng, (6, 1)), _dyadic(rng, (6, 5), 4))
        elif data == "float64":
            z = Zono(rng.uniform(-2, 2, (6, 1)), rng.uniform(-1, 1, (6, 5)))
        else:
            z = Zono(rng.uniform(-2, 2, (6, 1)).astype(np.float32),
                     rng.uniform(-1, 1, (6, 5)).astype(np.float32))
            if z.c.dtype != np.float32:
                pytest.fail("test setup: the Zono did not keep float32")
        exact_lo, exact_hi = _exact_bounds(z)
        what = f"Zono case {k} ({data})"
        for name in ("get_bounds", "get_ranges", "estimate_ranges"):
            lo, hi = getattr(z, name)()
            _assert_outward(lo, hi, exact_lo, exact_hi, f"{what}: {name}")
        box = z.get_box()
        _assert_outward(box.lb, box.ub, exact_lo, exact_hi, f"{what}: get_box")
        for i in range(z.dim):
            lo, hi = z.get_range(i)
            _assert_outward([lo], [hi], exact_lo[i:i + 1], exact_hi[i:i + 1],
                            f"{what}: get_range({i})")
        _assert_holds(z.get_interval_hull(), _vertices(z),
                      f"{what}: get_interval_hull")
        image = ImageZono(z.c, z.V, 1, 3, 2)
        lo, hi = image.get_bounds()
        _assert_outward(lo, hi, exact_lo, exact_hi,
                        f"{what}: ImageZono.get_bounds")


def test_get_range_refuses_an_index_out_of_range():
    """An index below 0 or at the dimension: ``ValueError``."""
    z = Zono(np.zeros((2, 1)), np.eye(2))
    for index in (-1, 2):
        with pytest.raises(ValueError):
            z.get_range(index)


def _exact_range_over_a_box(W, b, lo, hi):
    """Exact range of each entry of ``W x + b`` over the box ``[lo, hi]``,
    whose ends may be infinite: a zero weight adds 0 (for every finite x), a
    nonzero weight on an infinite end makes that side unbounded (None)."""
    lows, highs = [], []
    for row, bias in zip(W, b):
        low = high = Fraction(float(bias))
        for w, a, z in zip(row, lo, hi):
            if w == 0:
                continue
            down, up = (a, z) if w > 0 else (z, a)
            w = Fraction(float(w))
            low = (None if low is None or np.isinf(down)
                   else low + w * Fraction(float(down)))
            high = (None if high is None or np.isinf(up)
                    else high + w * Fraction(float(up)))
        lows.append(low)
        highs.append(high)
    return lows, highs


def test_bound_queries_on_a_box_with_an_infinite_end_hold_the_range():
    """A box with an infinite lower end, an infinite upper end and both,
    beside a finite coordinate, made a Zono by ``from_bounds`` (centre 0 and
    an infinite half-width there), and its image under a map whose zero
    entries meet the infinite coordinates (the image's generators then hold
    0 * inf = NaN): ``get_bounds``, ``get_ranges``, ``estimate_ranges``,
    ``get_range`` and ``get_box`` either refuse (``ValueError`` or
    ``NotImplementedError``) or return bounds with no NaN that hold the exact
    range of every entry, infinite on a side where the entry is unbounded
    (dyadic data, zero slack elsewhere). Whether a Zono may hold an infinite
    generator is an open question of the owner;
    this promise is right under either answer. numpy's warning about the
    NaN is silenced, as it only warns outside the test suite. Catches a NaN
    bound (a NaN compares false with everything, so a kernel that reads it
    can take either branch) and a finite bound for an unbounded entry."""
    lo = np.array([-np.inf, -1.5, -np.inf, 2.25])
    hi = np.array([0.5, np.inf, np.inf, 6.0])
    W = np.array([[0.0, 0.0, 0.0, 2.0],       # the finite coordinate only
                  [1.0, 0.0, 0.0, -0.5],      # unbounded below
                  [0.0, -2.0, 0.0, 1.0],      # unbounded below
                  [0.0, 0.0, 0.0, 0.0],       # the point -3
                  [0.5, 0.25, 0.0, 0.0]])     # unbounded on both sides
    b = np.array([0.25, 0.0, 1.0, -3.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z = Zono.from_bounds(lo, hi)
        sets = [("the box", z, np.eye(4), np.zeros(4)),
                ("its affine image", z.affine_map(W, b), W, b)]
        for what, s, M, m in sets:
            exact_lo, exact_hi = _exact_range_over_a_box(M, m, lo, hi)
            queries = {
                "get_bounds": s.get_bounds,
                "get_ranges": s.get_ranges,
                "estimate_ranges": s.estimate_ranges,
                "get_box": lambda s=s: (lambda box: (box.lb, box.ub))(
                    s.get_box()),
                "get_range": lambda s=s: tuple(
                    np.array(v) for v in zip(*(s.get_range(i)
                                              for i in range(s.dim))))}
            for name, query in queries.items():
                try:
                    got_lo, got_hi = query()
                except (ValueError, NotImplementedError):
                    continue
                got_lo = np.asarray(got_lo, dtype=np.float64).reshape(-1)
                got_hi = np.asarray(got_hi, dtype=np.float64).reshape(-1)
                for i, (a, c, ea, ec) in enumerate(zip(got_lo, got_hi,
                                                       exact_lo, exact_hi)):
                    tag = f"{name} of {what}, entry {i}: [{a!r}, {c!r}]"
                    assert not (np.isnan(a) or np.isnan(c)), f"{tag} is NaN"
                    assert (a == -np.inf if ea is None
                            else a == -np.inf or Fraction(float(a)) <= ea), (
                        f"{tag} does not hold the lower end "
                        f"{'-inf' if ea is None else float(ea)}")
                    assert (c == np.inf if ec is None
                            else c == np.inf or Fraction(float(c)) >= ec), (
                        f"{tag} does not hold the upper end "
                        f"{'inf' if ec is None else float(ec)}")


# --------------------------------------------------------------------------- #
# intersect_half_space                                                          #
# --------------------------------------------------------------------------- #
def test_intersect_half_space_is_the_exact_intersection():
    """A 3-dimensional zonotope with four generators cut by two half-spaces
    (one at a time and both at once): every vertex of the exact intersection
    (rationals: the generator box with the cut rows, mapped) is a member of
    the returned Star, exactly; and every vertex of the result is a point of
    the zonotope (the cut itself is inflated outward on purpose, see
    ``Star.intersect_half_space``, so the result may exceed it). Catches the
    cut imposed with the wrong sign, against the wrong side of the centre,
    and a conversion to a Star whose predicate box is wrong."""
    rng = np.random.default_rng(8)
    z = Zono(_dyadic(rng, (3, 1)), _dyadic(rng, (3, 4), 4))
    c, G = R.star(z)[:2]
    m = len(G[0])
    unit = [[Fraction(int(i == j)) for j in range(m)] for i in range(m)]
    box_A = unit + [[-v for v in row] for row in unit]
    box_b = [Fraction(1)] * (2 * m)
    for H, g in (([[1.0, -0.5, 0.25]], [0.125]),
                 ([[-0.25, 1.0, 1.0]], [0.0]),
                 ([[1.0, -0.5, 0.25], [-0.25, 1.0, 1.0]], [0.125, 0.0])):
        H_rows = [[Fraction(v) for v in row] for row in H]
        A, b = list(box_A), list(box_b)
        for row, rhs in zip(H_rows, g):
            A.append([R.dot(row, [G[i][j] for i in range(3)])
                      for j in range(m)])
            b.append(Fraction(rhs) - R.dot(row, c))
        verts = [tuple(R.dot(row, a) + ci for row, ci in zip(G, c))
                 for a in R.vertices(A, b)]
        if not verts:
            pytest.fail("test setup: the intersection is empty")
        out = z.intersect_half_space(np.array(H), np.array(g).reshape(-1, 1))
        _assert_holds(out, verts, f"intersect_half_space({H})")
        zono_member = R.Membership(R.star(z))
        for y in R.set_vertices(R.star(out)):
            assert zono_member(y), (
                f"intersect_half_space({H}): the result has the vertex "
                f"{[float(v) for v in y]}, which is not in the zonotope")


# --------------------------------------------------------------------------- #
# contains                                                                      #
# --------------------------------------------------------------------------- #
def test_contains_answers_exactly_on_a_box_shaped_zonotope():
    """``contains`` on a zonotope whose generators lie on the axes (two on
    the first, so more generators than entries): the set is its own bounding
    box, so today's bounding-box test is exact there, and the answer must be
    True for every member and False for every other point, membership
    decided exactly on the stored arrays, for a flat point and a column. The
    points: the centre, every vertex and the middle of every face (members);
    a point 1/8 beyond each face with the other entries at the centre, and a
    point outside in the first entry only with the others at a corner (not
    members). Catches a radius scaled up (a point within twice the radius
    accepted) and the entries combined by 'any' instead of 'all'. On a
    general zonotope a bounding-box test accepts points that are not
    members: that is the marked case
    ``contracts/test_set_protocol.py::test_zono_contains_rejects_points_between_the_set_and_its_bounding_box``
    (registry SET-1)."""
    c = np.array([[1.5], [-4.0], [0.25]])
    V = np.array([[2.0, 0.5, 0.0, 0.0],
                  [0.0, 0.0, 3.0, 0.0],
                  [0.0, 0.0, 0.0, -0.75]])
    z = Zono(c, V)
    centre = [Fraction(float(v)) for v in c.ravel()]
    radius = [Fraction(5, 2), Fraction(3), Fraction(3, 4)]
    points = [tuple(centre)] + _vertices(z)
    for i in range(3):
        for side in (-1, 1):
            for beyond in (Fraction(0), Fraction(1, 8)):
                p = list(centre)
                p[i] += side * (radius[i] + beyond)
                points.append(tuple(p))
    points.append((centre[0] + radius[0] + Fraction(1, 8),
                   centre[1] + radius[1], centre[2] - radius[2]))
    member = R.Membership(R.star(z))
    expected = [member(p) for p in points]
    if all(expected) or not any(expected):
        pytest.fail("test setup: the points must include members and others")
    wrong = []
    for p, inside in zip(points, expected):
        x = np.array([float(v) for v in p])
        for form in (x, x.reshape(-1, 1)):
            if bool(z.contains(form)) != inside:
                wrong.append((x.tolist(), inside))
    assert not wrong, (
        f"Zono.contains answered wrongly for {len(wrong)} of "
        f"{2 * len(points)} queries; first point {wrong[0][0]}, a member: "
        f"{wrong[0][1]}")


# --------------------------------------------------------------------------- #
# to_star                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("generators", [4, 0])
def test_to_star_is_the_same_set(generators):
    """``to_star`` of a 3-dimensional zonotope with four mixed-sign
    generators (dyadic, entries beyond plus and minus 1), and of one with no
    generator: every vertex of the zonotope is a member of the Star, and
    every vertex of the Star (its predicate polytope, mapped) is a member of
    the zonotope, both decided exactly on the stored arrays (the Star's rows
    and predicate bounds, the zonotope's cube). With no generator the Star
    has no predicate variable and its only point is the centre. Catches a
    predicate bound of 0 in place of -1 (the Star is the half of the
    zonotope with non-negative coefficients), a generator column dropped or
    repeated, and the centre left out of the basis. ``verify_specification``
    reads a Zono through this conversion: see the public route below."""
    rng = np.random.default_rng(14)
    z = Zono(_dyadic(rng, (3, 1), 2), _dyadic(rng, (3, generators), 2))
    star = z.to_star()
    _assert_holds(star, _vertices(z), f"Zono.to_star, {generators} generators")
    zono_member = R.Membership(R.star(z))
    for y in R.set_vertices(R.star(star)):
        assert zono_member(y), (
            f"Zono.to_star, {generators} generators: the Star has the vertex "
            f"{[float(v) for v in y]}, which is not in the zonotope")


# --------------------------------------------------------------------------- #
# The public route: reach_pytorch_model, then verify_specification              #
# --------------------------------------------------------------------------- #
def _dyadic_linear(rng, n_in, n_out):
    layer = nn.Linear(n_in, n_out)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(_dyadic(rng, (n_out, n_in), 4)))
        layer.bias.copy_(torch.tensor(_dyadic(rng, (n_out,), 4)))
    return layer


def _zono_points(z, n, seed):
    """``n`` points of a zonotope: its vertices, then ``c + V a`` for random
    dyadic ``a`` in the cube (exact in float64)."""
    rng = np.random.default_rng(seed)
    points = [np.array([float(v) for v in p]) for p in _vertices(z)][:n]
    while len(points) < n:
        a = rng.integers(-4, 5, size=(z.V.shape[1], 1)) / 4.0
        points.append((z.c + z.V @ a).ravel())
    return points


def _assert_no_false_unsat(sets, outputs, what):
    """``verify_specification`` of the reach sets against the unsafe
    half-space ``g . y <= t``, with ``t`` the smallest value of ``g . y``
    over the true outputs, so that one true output lies in it: the answer
    must not be UNSAT. ``g`` is a fixed dyadic row over the outputs in the
    tensor's own order."""
    size = outputs[0].size
    g = np.array([[(-1.0) ** k * (1 + k % 3) / 4 for k in range(size)]])
    t = min(float(g[0] @ y) for y in outputs)
    verdict = verify_specification(sets, HalfSpace(g, np.array([t]))).verdict
    assert verdict != "UNSAT", (
        f"{what}: verify_specification answers UNSAT although a true output "
        f"lies in the unsafe half-space")


@pytest.mark.parametrize("input_set", ["from-bounds",
                                       "more-generators-than-entries"])
def test_reach_and_verify_of_a_zono_hold_every_output(input_set):
    """The public route of a Zono: ``reach_pytorch_model`` of Linear(3, 4),
    ReLU, Linear(4, 2) (dyadic weights) with method 'approx' on a Zono built
    by ``from_bounds`` (intervals beyond plus and minus 1 and wider than 2,
    one of zero width) or given directly with five generators for three
    entries (the shape a ReLU relaxation leaves). Every true output of 128
    member points (torch, float64) is in the result by the LP oracle on its
    rows, and ``verify_specification``, which reads the result through
    ``to_star``, does not answer UNSAT for an unsafe half-space that holds a
    true output. The Linear kernel calls ``affine_map`` and the ReLU kernel
    decides each neuron's phase from ``get_bounds``, so this catches, at the
    level of a verification answer, a wrong affine map, bounds that stop
    short (a crossing neuron taken as inactive), a box from ``from_bounds``
    that is too small and a ``to_star`` that loses part of the set."""
    rng = np.random.default_rng(15)
    model = nn.Sequential(_dyadic_linear(rng, 3, 4), nn.ReLU(),
                          _dyadic_linear(rng, 4, 2))
    if input_set == "from-bounds":
        lo, hi = np.array([-3.5, 0.75, 2.0]), np.array([1.25, 4.0, 2.0])
        z, points = Zono.from_bounds(lo, hi), samples(lo, hi)
    else:
        z = Zono(np.array([[1.5], [-2.0], [0.5]]), _dyadic(rng, (3, 5), 2))
        points = _zono_points(z, 128, seed=15)
    what = f"Linear, ReLU, Linear on a Zono ({input_set})"
    out = reach_model(model, z, "approx", what)
    outputs = replay_torch(model, points, (1, 3))
    assert_covered(out, outputs, what)
    _assert_no_false_unsat(out, outputs, what)


def test_reach_and_verify_of_an_image_zono_hold_every_output():
    """The public route of an ImageZono: ``image_set('ImageZono', ...)``
    (``ImageZono.from_bounds`` on the stored layout) of a (2, 3, 5) image
    with its own interval per pixel, then Conv2d(2, 3, (2, 3), stride (1, 2),
    padding (1, 0)) and ReLU through ``reach_pytorch_model``, approx. Every
    true output of 128 points (torch, float64) is in the result (LP oracle,
    outputs converted to the stored order), and ``verify_specification``,
    which reads an ImageZono through its tensor-order flattening and
    ``to_star``, does not answer UNSAT for an unsafe half-space over the
    (channel, height, width) output that holds a true output. Catches height
    and width exchanged in the constructor or in ``from_bounds``, bounds read
    in another order, and ``get_bounds`` of an image set returned in another
    order than the one it stores (the ReLU kernel then reads another pixel's
    range)."""
    lo, hi = pixel_box(C_, H_, W_, seed=3)
    model = nn.Sequential(_conv(), nn.ReLU())
    what = "Conv2d, ReLU on an ImageZono"
    out = reach_model(model, image_set("ImageZono", lo, hi), "approx", what)
    outputs = replay_torch(model, samples(lo, hi), (1, C_, H_, W_))
    shape = tuple(model(torch.zeros(1, C_, H_, W_)).shape[1:])
    assert_covered(out, outputs, what, out_chw=shape)
    _assert_no_false_unsat(out, outputs, what)


# --------------------------------------------------------------------------- #
# to_image_zono and to_image_star, then spatial layers                          #
# --------------------------------------------------------------------------- #
C_, H_, W_ = 2, 3, 5


def _image_zonos(parts):
    """``parts`` flat Zonos whose union is the box of ``pixel_box(2, 3, 5)``,
    each in the stored (height, width, channel) order."""
    lo, hi = pixel_box(C_, H_, W_, seed=2)
    return lo, hi, split_cover("Zono", hwc(lo, C_, H_, W_),
                               hwc(hi, C_, H_, W_), parts=parts)


def _conv():
    conv = nn.Conv2d(C_, 3, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
    rng = np.random.default_rng(9)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor(_dyadic(rng, conv.weight.shape, 4),
                                       dtype=torch.float32))
        conv.bias.copy_(torch.tensor(_dyadic(rng, conv.bias.shape, 4),
                                     dtype=torch.float32))
    return conv


_LAYERS = {
    "avgpool": lambda: nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)),
    "conv": _conv,
}


@pytest.mark.parametrize("target, layer, parts", [
    ("ImageZono", "avgpool", 1), ("ImageZono", "conv", 3),
    ("ImageStar", "avgpool", 1), ("ImageStar", "conv", 3)])
def test_conversion_to_an_image_set_keeps_the_pixel_order(target, layer,
                                                          parts):
    """A flat Zono in the stored (height, width, channel) order of a (2, 3, 5)
    image, with a different interval per pixel, converted by
    ``to_image_zono(3, 5, 2)`` or ``to_image_star(3, 5, 2)``, alone or as
    three sets that cover the box, then AvgPool2d (kernel 1 x 2) or Conv2d
    (kernel 2 x 3, stride (1, 2), padding (1, 0)) through ``reach_layer``:
    the converted sets hold the image's own points and the layer's results
    hold every true output (torch float64, converted to the stored order).
    Catches height and width exchanged in the conversion (the pixels are
    regrouped for every later spatial layer)."""
    lo, hi, zonos = _image_zonos(parts)
    convert = {"ImageZono": lambda z: z.to_image_zono(H_, W_, C_),
               "ImageStar": lambda z: z.to_image_star(H_, W_, C_)}[target]
    images = [convert(z) for z in zonos]
    for im in images:
        assert isinstance(im, {"ImageZono": ImageZono,
                               "ImageStar": ImageStar}[target])
    points = samples(lo, hi, n=48)
    what = f"Zono.to_{target}(3, 5, 2) of {parts} set(s)"
    assert_covered(images, points, what, out_chw=(C_, H_, W_))
    module = _LAYERS[layer]()
    out = reach_one_layer(module, images, "approx", f"{what}, {layer}")
    outputs = replay_torch(module, points, (1, C_, H_, W_))
    shape = tuple(module(torch.zeros(1, C_, H_, W_)).shape[1:])
    assert_covered(out, outputs, f"{what}, then {layer}", out_chw=shape)


@pytest.mark.parametrize("target", ["to_image_zono", "to_image_star"])
def test_conversion_to_an_image_set_refuses_a_wrong_size(target):
    """Height x width x channels different from the dimension:
    ``ValueError``."""
    z = Zono(np.zeros((6, 1)), np.eye(6))
    with pytest.raises(ValueError):
        getattr(z, target)(2, 2, 2)


# --------------------------------------------------------------------------- #
# ImageZono                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(3, 5, 2), (30,)])
def test_image_zono_from_bounds_reads_its_own_layouts(shape):
    """A (height, width, channel) array, or a flat one in that order: entry
    (h, w, c) of the set is entry (h, w, c) of the array, so the centre of
    every pixel's interval is a member and a point one unit above it is
    1/2 away (LP oracle). ``to_zono`` and ``get_image_shape`` keep the set
    and its shape. Catches the bounds read in another order."""
    lo = np.arange(30.0).reshape(3, 5, 2)
    z = ImageZono.from_bounds(lo.reshape(shape), (lo + 1.0).reshape(shape),
                              height=3, width=5, num_channels=2)
    assert z.get_image_shape() == (3, 5, 2)
    assert distance(z, lo.reshape(-1) + 0.5) <= LP_TOL
    assert distance(z, lo.reshape(-1) + 1.5) == pytest.approx(0.5)
    flat = z.to_zono()
    assert type(flat) is Zono
    assert distance(flat, lo.reshape(-1) + 0.5) <= LP_TOL
    assert distance(flat, lo.reshape(-1) + 1.5) == pytest.approx(0.5)


@known_bug(7, "ImageZono.from_bounds checks only the number of entries: a "
              "(C, H, W) array is read as (H, W, C) and the pixels are "
              "permuted (registry SHAPE-6)", raises=DID_NOT_RAISE)
def test_image_zono_from_bounds_refuses_a_channels_first_array():
    """Bounds given as a (C, H, W) = (2, 3, 5) array for an image declared
    3 x 5 x 2 must be refused, as for ``ImageStar.from_bounds`` (SHAPE-5);
    the neighbour above reads the image's own layouts."""
    lo = np.zeros((2, 3, 5))
    with pytest.raises(ValueError):
        ImageZono.from_bounds(lo, lo + 1.0, height=3, width=5, num_channels=2)


def test_image_zono_constructor_refuses_a_wrong_size():
    """Height x width x channels different from the number of entries:
    ``ValueError``."""
    with pytest.raises(ValueError):
        ImageZono(np.zeros((30, 1)), np.eye(30), 5, 3, 3)
