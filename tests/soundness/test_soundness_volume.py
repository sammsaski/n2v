"""Volume of stars and star unions (``n2v/sets/volume.py``), checked against
volumes known in closed form.

Components: ``star_volume`` (``method='exact'`` and ``method='mc'``),
``star_union_volume_sound_lower``, ``star_union_volume_mc`` (the serial loop
and the process pool), ``star_union_bbox``, ``compute_mc_bbox`` and
``exact_volume_2d``, each also reached from the exact reach set of a small
ReLU network.

The promise: a volume documented as exact is the volume of the set in output
space, in both directions; a volume given as a lower bound is never above the
true volume; a Monte Carlo estimate comes with an interval that holds the
true volume and is at least as wide as the Hoeffding bound for the requested
confidence, the number of samples and the range of one draw; a union box
holds every point of every member; the box of ``compute_mc_bbox`` holds the
image of its input ball; the raster area of ``exact_volume_2d`` lies within a
bound derived from its raster and its sampling.

The oracle is never n2v: volumes of cut parallelograms, simplex corners,
rotated squares and unions (inclusion and exclusion) in
``fractions.Fraction``, polygon vertices from ``tests/_oracles/rational.py``.
Monte Carlo cases fix their seed and request confidence ``1 - 1e-7``, so the
probability that a correct implementation fails one interval check is at most
1e-7 (Hoeffding); a case that rests on another bound states its own figure.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from n2v.sets import Star
from n2v.sets.volume import (
    compute_mc_bbox,
    exact_volume_2d,
    star_union_bbox,
    star_union_volume_mc,
    star_union_volume_sound_lower,
    star_volume,
)
from tests._oracles import rational
from tests.cell_support import reach_model
from tests.known_bugs import known_bug

_CONFIDENCE = 1 - 1e-7


# --------------------------------------------------------------------------- #
# Oracle helpers (no n2v)                                                       #
# --------------------------------------------------------------------------- #
def _hoeffding(scale, n, confidence):
    """Half-width of the two-sided Hoeffding interval of a mean of ``n``
    draws in [0, scale]: 2 exp(-2 n h^2 / scale^2) = 1 - confidence."""
    return float(scale) * math.sqrt(math.log(2 / (1 - confidence)) / (2 * n))


def _assert_interval(estimate, true_volume, scale, n, confidence=_CONFIDENCE):
    """The interval holds the true volume, and on each side it is at least
    the Hoeffding half-width for the requested confidence (``ci_low`` may be
    clipped at 0). ``scale`` is a lower bound on the range of one draw, so
    the check is one-sided: a wider interval is loose, a narrower one is not
    backed by the bound."""
    half = _hoeffding(scale, n, confidence)
    assert estimate.n_samples == n, estimate
    assert (Fraction(float(estimate.ci_low)) <= Fraction(true_volume)
            <= Fraction(float(estimate.ci_high))), (
        f"true volume {float(true_volume)} outside "
        f"[{estimate.ci_low}, {estimate.ci_high}]")
    assert estimate.ci_high - estimate.mean >= half * (1 - 1e-12), (
        f"upper half-width {estimate.ci_high - estimate.mean} below the "
        f"Hoeffding half-width {half}")
    if estimate.mean - half > 0:
        assert estimate.mean - estimate.ci_low >= half * (1 - 1e-12), (
            f"lower half-width {estimate.mean - estimate.ci_low} below the "
            f"Hoeffding half-width {half}")


def _polygon_area(A, b):
    """Area of the bounded polygon ``{x : A x <= b}`` in fractions: vertices
    by exact enumeration, ordered around their centroid, shoelace."""
    verts = rational.vertices(rational.matrix(A), rational.vector(b))
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    verts.sort(key=lambda v: math.atan2(float(v[1] - cy), float(v[0] - cx)))
    twice = sum(p[0] * q[1] - q[0] * p[1]
                for p, q in zip(verts, verts[1:] + verts[:1]))
    return abs(twice) / 2


def _box_rows(plb, pub):
    n = len(plb)
    eye = np.eye(n)
    return np.vstack([eye, -eye]), np.concatenate([pub, -np.asarray(plb)])


def _det(B):
    """Determinant in fractions (Laplace expansion; at most 4 x 4 here)."""
    B = [[rational.frac(x) for x in row] for row in B]
    if len(B) == 1:
        return B[0][0]
    return sum((-1) ** j * B[0][j] * _det([row[:j] + row[j + 1:]
                                           for row in B[1:]])
               for j in range(len(B)))


# --------------------------------------------------------------------------- #
# Shapes whose volume is known in closed form                                   #
# --------------------------------------------------------------------------- #
def _cut_parallelogram():
    """x = (5, -7) + [[2, 3], [1, -1]] a, determinant -5 (the basis
    reflects), a in [0.5, 2] x [1, 3] (a predicate box away from [-1, 1]),
    cut by a0 + a1 <= 3.5 and a0 - a1 <= 0.5 (rows with offsets, neither a
    box row). The predicate polygon has vertices (0.5, 1), (1.5, 1),
    (2, 1.5), (0.5, 3), area 7/4, so the volume is 35/4."""
    V = np.array([[5.0, 2.0, 3.0], [-7.0, 1.0, -1.0]])
    C = np.array([[1.0, 1.0], [1.0, -1.0]])
    d = np.array([3.5, 0.5])
    plb, pub = np.array([0.5, 1.0]), np.array([2.0, 3.0])
    A, b = _box_rows(plb, pub)
    area = _polygon_area(np.vstack([A, C]), np.concatenate([b, d]))
    true = abs(_det(V[:, 1:])) * area
    if true != Fraction(35, 4):
        pytest.fail(f"the oracle gives {true}, the docstring 35/4")
    return Star(V, C, d, plb, pub), true, 5 * Fraction(3)


def _simplex_corner_3d():
    """Basis [[1, 1, 0], [0, 2, 0], [0, 0, -1]] (determinant -2), predicate
    box [1, 2] x [-3, -2] x [0.5, 1.5], cut by a0 - a1 + a2 <= 4.5: the
    corner simplex at (1, -2, 0.5) with three legs of length 1, volume 1/6
    in predicate space, 1/3 in output space. The second row a2 <= 1.5 is
    the box row again, so a sample is inside only when it meets both."""
    V = np.array([[3.0, 1.0, 1.0, 0.0], [-2.0, 0.0, 2.0, 0.0],
                  [1.0, 0.0, 0.0, -1.0]])
    C = np.array([[1.0, -1.0, 1.0], [0.0, 0.0, 1.0]])
    d = np.array([4.5, 1.5])
    plb, pub = np.array([1.0, -3.0, 0.5]), np.array([2.0, -2.0, 1.5])
    return Star(V, C, d, plb, pub), Fraction(1, 3), Fraction(2)


def _simplex_corner_4d():
    """Basis [[2, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.5]]
    (determinant 3), predicate box [-1, 0] x [0, 1] x [2, 3] x [0, 1], cut by
    a0 + a1 + a2 + a3 <= 2: the corner simplex with four legs of length 1,
    volume 1/24 in predicate space, 1/8 in output space. The second row
    a0 <= 0 is the box row again."""
    V = np.array([[0.0, 2.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 1.5]])
    C = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    d = np.array([2.0, 0.0])
    plb, pub = np.array([-1.0, 0.0, 2.0, 0.0]), np.array([0.0, 1.0, 3.0, 1.0])
    return Star(V, C, d, plb, pub), Fraction(1, 8), Fraction(3)


_SHAPES = {
    "2d-cut-parallelogram": _cut_parallelogram,
    "3d-simplex-corner": _simplex_corner_3d,
    "4d-simplex-corner": _simplex_corner_4d,
}


def _diamond():
    """|x| + |y| <= 1: x = [[0.5, 0.5], [0.5, -0.5]] a, a in [-1, 1]^2;
    area 2, bounding box [-1, 1]^2 of area 4."""
    V = np.array([[0.0, 0.5, 0.5], [0.0, 0.5, -0.5]])
    return Star(V, None, None, -np.ones(2), np.ones(2))


def _bar():
    """The box [0, 2] x [-0.5, 0.5], area 2. It meets the diamond in
    {0 <= x, |y| <= 0.5, |x| + |y| <= 1}, area 1/2 + 1/4."""
    return Star.from_bounds(np.array([[0.0], [-0.5]]), np.array([[2.0], [0.5]]))


_DIAMOND_BAR_UNION = Fraction(2) + 2 - Fraction(3, 4)      # 13/4
_DIAMOND_BAR_BOX = (np.array([-1.0, -1.0]), np.array([2.0, 1.0]))   # area 6


# --------------------------------------------------------------------------- #
# star_volume                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", list(_SHAPES))
def test_exact_volume_is_the_closed_form_volume(shape):
    """``method='exact'`` claims the volume: compared both ways with the
    closed form (relative 1e-9 for the rounding of vertex enumeration and
    triangulation). Catches a cut row whose offset changes sign, a lower
    predicate bound read with the wrong sign, a reflecting basis that gives a
    negative volume, a simplex volume divided by 2 instead of d! (the 3-D and
    4-D shapes), and an unused-variable test that drops or keeps the wrong
    variables."""
    star, true, _ = _SHAPES[shape]()
    estimate = star_volume(star, method="exact")
    assert estimate.mean == pytest.approx(float(true), rel=1e-9), estimate
    assert estimate.ci_low == estimate.mean == estimate.ci_high


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_mc_interval_holds_the_closed_form_volume(shape):
    """``method='mc'``: the interval holds the closed form and is no
    narrower than the Hoeffding half-width whose range is the predicate box
    volume times |det| (a draw is 0 or that product). Catches a rejection
    step that keeps a sample meeting any row instead of every row (the
    second row of each shape is a box row, met by every sample), a box
    volume summed instead of multiplied, a half-width without the Jacobian
    or without the factor 2 of the two-sided bound. 20000 draws, one numpy
    pass each."""
    star, true, scale = _SHAPES[shape]()
    estimate = star_volume(star, method="mc", n_samples=20_000, seed=3,
                           confidence=_CONFIDENCE)
    _assert_interval(estimate, true, scale, 20_000)


@pytest.mark.parametrize("method", ["exact", "mc"])
def test_volume_of_a_star_whose_predicate_box_is_empty_in_one_coordinate(
        method):
    """a1 in [2, 1.5] is empty while a0 in [-1, 1] is not, so the star is
    empty. Catches an emptiness test of the predicate box that requires the
    order in any coordinate instead of in all of them (the Monte Carlo branch
    then samples the inverted interval and returns a negative volume)."""
    star = Star(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), None, None,
                np.array([-1.0, 2.0]), np.array([1.0, 1.5]))
    estimate = star_volume(star, method=method, n_samples=1_000)
    assert estimate.mean == 0.0, estimate
    assert estimate.ci_low <= 0.0 <= estimate.ci_high, estimate


@pytest.mark.parametrize("rows, method", [
    pytest.param("empty", "exact", id="empty-exact"),
    pytest.param("empty", "mc", id="empty-mc"),
    pytest.param("flat", "exact", id="flat-exact"),
])
def test_volume_of_a_star_emptied_or_flattened_by_its_rows_is_zero(rows,
                                                                   method):
    """Rows a0 <= -1/2 and -a0 <= -1/2 leave no point of [-1, 1]^2; rows
    a0 <= 0 and -a0 <= 0 leave the segment a0 = 0, of area 0 under a square
    basis. Catches the offsets of the cut rows read with the wrong sign (the
    rows then keep |a0| <= 1/2), a rejection step that keeps a sample meeting
    any row, and an emptiness check of the vertex enumeration that reads the
    LP's point without looking at its status."""
    offset = -0.5 if rows == "empty" else 0.0
    star = Star(np.array([[1.0, 2.0, 1.0], [-1.0, 0.0, 3.0]]),
                np.array([[1.0, 0.0], [-1.0, 0.0]]),
                np.array([offset, offset]), -np.ones(2), np.ones(2))
    estimate = star_volume(star, method=method, n_samples=2_000,
                           confidence=_CONFIDENCE)
    assert estimate.mean == 0.0, estimate
    assert estimate.ci_low <= 0.0 <= estimate.ci_high, estimate


@pytest.mark.parametrize("method", [
    pytest.param("exact", marks=known_bug(
        8, "star_volume(method='exact') returns 0 for a star with one "
           "predicate variable: scipy's HalfspaceIntersection needs "
           "dimension 2, and its error is read as an empty polytope")),
    "mc",
])
def test_volume_of_a_one_dimensional_star_is_its_length(method):
    """x = 3 - 1.5 a with a in [1, 2.5]: the segment [-0.75, 1.5], length
    9/4. Dimension 1 is a legal input (the reach set of a network with one
    output); ``method='mc'`` is the passing neighbour with the same
    reference."""
    star = Star(np.array([[3.0, -1.5]]), None, None, np.array([1.0]),
                np.array([2.5]))
    true = Fraction(9, 4)
    if method == "exact":
        assert star_volume(star, method="exact").mean == pytest.approx(
            float(true), rel=1e-9)
    else:
        estimate = star_volume(star, method="mc", n_samples=2_000,
                               confidence=_CONFIDENCE)
        _assert_interval(estimate, true, Fraction(9, 4), 2_000)


@pytest.mark.parametrize("method, half_width", [
    pytest.param("exact", np.inf, id="exact", marks=known_bug(
        8, "star_volume(method='exact') of a star with infinite predicate "
           "bounds and bounding rows raises scipy's ValueError ('b_ub must "
           "not contain values inf') instead of answering",
        raises=ValueError)),
    pytest.param("mc", np.inf, id="mc", marks=known_bug(
        8, "star_volume(method='mc') of a star with infinite predicate "
           "bounds and bounding rows samples the infinite box and raises "
           "OverflowError instead of answering", raises=OverflowError)),
    pytest.param("exact", 3.0, id="exact-finite-box"),
    pytest.param("mc", 3.0, id="mc-finite-box"),
])
def test_volume_of_a_star_with_infinite_predicate_bounds(method, half_width):
    """x = [[1, 0.5], [0, 2]] a with |a_i| <= 3 stated by rows and the
    predicate box (-inf, inf)^2, the form the reach of Mul, Pow and the
    set-by-set operations gives a star without a predicate box: the
    parallelogram of area 36 x 2. The volume is due, or a refusal
    (``NotImplementedError``); any other exception is a crash. The same set
    with the finite predicate box [-3, 3]^2 is the passing neighbour of each
    method."""
    rows = np.vstack([np.eye(2), -np.eye(2)])
    star = Star(np.array([[0.0, 1.0, 0.5], [0.0, 0.0, 2.0]]), rows,
                3.0 * np.ones(4), -half_width * np.ones(2),
                half_width * np.ones(2))
    try:
        estimate = star_volume(star, method=method, n_samples=2_000,
                               confidence=_CONFIDENCE)
    except NotImplementedError:
        return
    if method == "exact":
        assert estimate.mean == pytest.approx(72.0, rel=1e-9)
    else:
        assert estimate.ci_low <= 72.0 <= estimate.ci_high, estimate


def test_volume_of_the_reach_set_of_a_layer_with_more_outputs_than_inputs():
    """A Linear layer from 2 to 3 with weight columns u = (2, 0, 0) and
    v = (1, 3, 4) maps the box [-1, 1] x [0, 2] onto a parallelogram in a
    plane of R^3. Two readings of its volume are accepted, because the
    module states both: 0, its volume in R^3 (the docstring's measure-zero
    clause, and what ``star_union_volume_mc`` returns for a flat set), or
    its area in its own plane, which the comment on the Jacobian in
    ``star_volume`` documents for a basis with fewer columns than rows:
    sqrt(det(B^T B)) = |u x v| = 10 (in fractions) times the box area 4, so
    40. Any other number is wrong. Catches the square root of the Gram
    determinant dropped (400). Reached through ``reach_pytorch_model``, the
    route a user takes."""
    W = [[2, 1], [0, 3], [0, 4]]
    u, v = [Fraction(row[0]) for row in W], [Fraction(row[1]) for row in W]
    cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
    if sum(c * c for c in cross) != 100:
        pytest.fail("the oracle's |u x v| should be 10")
    plane_area = 10 * Fraction(2 * 2)
    layer = torch.nn.Linear(2, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(W, dtype=torch.float32))
        layer.bias.copy_(torch.tensor([1.0, -1.0, 0.5]))
    sets = reach_model(torch.nn.Sequential(layer),
                       Star.from_bounds(np.array([[-1.0], [0.0]]),
                                        np.array([[1.0], [2.0]])),
                       "approx", "Linear 2 -> 3")
    if len(sets) != 1 or sets[0].dim != 3:
        pytest.fail(f"expected one star of dimension 3, got {sets}")
    got = star_volume(sets[0], method="exact").mean
    assert got == 0.0 or got == pytest.approx(float(plane_area), rel=1e-9), (
        f"volume {got}: neither 0 nor the area {float(plane_area)} in the "
        f"image plane")


# --------------------------------------------------------------------------- #
# star_union_volume_sound_lower                                                 #
# --------------------------------------------------------------------------- #
def test_sound_lower_volume_is_the_largest_members_exact_volume():
    """The documented value is the largest member's exact volume: the cut
    parallelogram (35/4), listed first, then a box of area 6 inside the
    parallelogram's plane but disjoint from it. Compared both ways (relative
    1e-9) and named by ``meta['argmax_star']``. Catches member volumes taken
    by Monte Carlo instead of ``method='exact'`` (sampling error on a cut
    star) and an argmax that names the last member."""
    big, true, _ = _cut_parallelogram()
    small = Star.from_bounds(np.array([[0.0], [0.0]]), np.array([[2.0], [3.0]]))
    estimate = star_union_volume_sound_lower([big, small])
    assert estimate.mean == pytest.approx(float(true), rel=1e-9), estimate
    assert estimate.meta["argmax_star"] == 0, estimate.meta


_FLAT = Star.from_bounds(np.array([[0.0], [2.0], [0.0]]),
                         np.array([[1.0], [2.0], [3.0]]))


@pytest.mark.parametrize("members, union_volume", [
    pytest.param("two-boxes", Fraction(3, 16), id="two-boxes"),
    pytest.param("flat-and-box", Fraction(1, 8), id="flat-and-box",
                 marks=known_bug(
                     8, "star_union_volume_sound_lower returns the area of a "
                        "flat member (fewer predicate variables than "
                        "outputs) as a lower bound on the volume of the "
                        "union: 3 for a union of volume 1/8")),
])
def test_sound_lower_volume_is_not_above_the_union_volume(members,
                                                          union_volume):
    """A lower bound may be loose, never above the volume of the union.
    ``two-boxes``: [0, 1/2]^3 and [0, 1/4] x [0, 1] x [0, 1/2], volume
    1/8 + 1/8 - 1/16 by inclusion and exclusion (the passing neighbour).
    ``flat-and-box``: the box [0, 1] x {2} x [0, 3], which has volume 0 in
    R^3, and [0, 1/2]^3: the union's volume in R^3 is 1/8 (the measure of
    ``star_union_volume_mc``), whatever ``star_volume`` reports for the flat
    member alone. Catches a bound that takes the maximum over member
    volumes of different dimensions (the flat member's area 3)."""
    cube = Star.from_bounds(np.zeros((3, 1)), 0.5 * np.ones((3, 1)))
    other = (_FLAT if members == "flat-and-box" else
             Star.from_bounds(np.zeros((3, 1)),
                              np.array([[0.25], [1.0], [0.5]])))
    lower = star_union_volume_sound_lower([other, cube]).mean
    assert Fraction(lower) <= union_volume * (1 + Fraction(1, 10**12)), (
        f"lower bound {lower} on a volume of {float(union_volume)}")


@pytest.mark.parametrize("hi", [
    pytest.param((0.5, 0.75), id="dyadic"),
    pytest.param((0.1, 0.3), id="non-dyadic", marks=known_bug(
        8, "star_union_volume_sound_lower computes the member volume by "
           "rounding to nearest, so the lower bound lands above the exact "
           "volume of the stored set (0.030000000000000016 for the box "
           "[0, 0.1] x [0, 0.3], whose stored set has volume "
           "0.02999999999999999889 in fractions)")),
])
def test_sound_lower_volume_is_not_above_the_exact_volume_of_the_stored_set(
        hi):
    """The exact volume of the set the star stores, |det| of its basis times
    its predicate box volume, in fractions from the stored floats, and the
    bound compared with zero slack. ``dyadic`` is the passing neighbour."""
    star = Star.from_bounds(np.zeros((2, 1)), np.array(hi).reshape(2, 1))
    if star.C.shape[0] != 4:
        pytest.fail("from_bounds no longer gives the four box rows alone")
    box = Fraction(1)
    for lo_k, hi_k in zip(star.predicate_lb.ravel(), star.predicate_ub.ravel()):
        box *= Fraction(hi_k) - Fraction(lo_k)
    exact = abs(_det(star.V[:, 1:])) * box
    lower = star_union_volume_sound_lower([star]).mean
    assert Fraction(lower) <= exact, (
        f"lower bound {lower!r} above the exact volume {float(exact)!r}")


# --------------------------------------------------------------------------- #
# star_union_volume_mc and star_union_bbox                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("contains_method, bbox", [
    pytest.param("lp", None, id="lp-union-box"),
    pytest.param("algebraic", (np.array([-2.0, -1.5]), np.array([2.5, 1.5])),
                 id="algebraic-given-box"),
])
def test_union_mc_interval_holds_the_union_volume(contains_method, bbox):
    """The diamond and the bar overlap (union 13/4 by inclusion and
    exclusion); the diamond fills half of its own bounding box, so the
    pre-filter box is not the set. The interval must hold 13/4 and be no
    narrower than Hoeffding with the volume of the sampling box as range
    (the exact union box, area 6, or the given box, area 13.5). Catches a
    containment test that is skipped (every pre-filter candidate counted),
    a sample in the overlap counted twice, a given box read as (hi, lo), a
    half-width without the box volume or at the default confidence instead
    of the requested one. 3000 draws: about 2 s of LPs on the ``lp`` path."""
    estimate = star_union_volume_mc(
        [_diamond(), _bar()], bbox=bbox, n_samples=3_000, batch_size=1_000,
        seed=1, confidence=_CONFIDENCE, contains_method=contains_method)
    scale = Fraction(6) if bbox is None else Fraction(27, 2)
    _assert_interval(estimate, _DIAMOND_BAR_UNION, scale, 3_000)


def test_union_mc_in_worker_processes_holds_the_union_volume():
    """The process-pool path (``n_workers=2``, two chunks of 1500 draws):
    the same promise as the serial path. Catches a worker that counts a
    sample once per containing star, a star rebuilt in the worker with its
    predicate upper bound taken from the lower one, and a pool loop that
    keeps one chunk's count instead of the sum. Spawning two interpreters
    costs about 5 s; there is no cheaper route into ``_mc_worker_chunk``."""
    estimate = star_union_volume_mc(
        [_diamond(), _bar()], n_samples=3_000, batch_size=1_500, seed=2,
        confidence=_CONFIDENCE, contains_method="algebraic", n_workers=2)
    assert estimate.meta["n_workers"] == 2, estimate.meta
    _assert_interval(estimate, _DIAMOND_BAR_UNION, Fraction(6), 3_000)


def test_union_mc_with_algebraic_containment_answers_or_refuses_on_a_flat_member():
    """A union of the diamond and a flat star (the diagonal segment from
    (-1, -1) to (1, 1), whose bounding box has area 4) has volume 2. The
    algebraic test cannot decide membership in a rank-deficient star; the
    function may refuse (today ``ValueError`` from ``Star.contains``, or
    ``NotImplementedError``), and an answer must hold 2."""
    segment = Star(np.array([[0.0, 1.0], [0.0, 1.0]]), None, None,
                   -np.ones(1), np.ones(1))
    try:
        estimate = star_union_volume_mc(
            [_diamond(), segment], n_samples=300, batch_size=300, seed=0,
            confidence=_CONFIDENCE, contains_method="algebraic")
    except (NotImplementedError, ValueError):
        return
    assert estimate.ci_low <= 2.0 <= estimate.ci_high, estimate


def test_union_bbox_holds_every_vertex_of_every_member():
    """Members that are not boxes, in an order where the last member is
    extreme in no direction: the cut parallelogram (x in [9, 15], y in
    [-9.5, -6.5]), the diamond and the bar. Every vertex, in fractions,
    lies in the returned box with zero slack. Catches a lower corner
    overwritten by each member instead of merged by minimum, and an upper
    corner merged from the members' lower bounds."""
    para, _, _ = _cut_parallelogram()
    lo, hi = star_union_bbox([para, _diamond(), _bar()])
    A, b = _box_rows(np.array([0.5, 1.0]), np.array([2.0, 3.0]))
    alphas = rational.vertices(
        rational.matrix(np.vstack([A, para.C])),
        rational.vector(np.concatenate([b, para.d.ravel()])))
    points = [tuple(rational.frac(para.V[i, 0])
                    + rational.dot(rational.vector(para.V[i, 1:]), a)
                    for i in range(2)) for a in alphas]
    points += [(1, 0), (-1, 0), (0, 1), (0, -1),              # diamond
               (0, Fraction(-1, 2)), (2, Fraction(1, 2))]      # bar
    for p in points:
        for k in range(2):
            assert Fraction(lo[k]) <= p[k] <= Fraction(hi[k]), (
                f"vertex {tuple(map(float, p))} outside [{lo}, {hi}]")


# --------------------------------------------------------------------------- #
# compute_mc_bbox                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("output_dim", [3, 2])
def test_mc_bbox_holds_the_image_of_the_input_ball(output_dim):
    """y = W x + b with W = [[1, -2], [0.5, 0.25], [-3, 1]], b = (1, -2,
    0.5), on the ball of radius 3 around (1, -1), i.e. [-2, 4] x [-4, 2]:
    the image box, by interval arithmetic in fractions, lies in the returned
    box (first ``output_dim`` outputs, float32 tensors). With 5000 draws and
    pad 1, a correct implementation misses an extreme by more than the pad
    only when no draw lands in the corner region of the input box within pad
    of it; the smallest such region is 1/216 of the box, so the probability
    is at most 6 (1 - 1/216)^5000 < 1e-9. Catches draws from one orthant of
    the ball (U[0, 1) instead of U[-1, 1)), a pad subtracted from the upper
    corner, and the wrong outputs kept."""
    W = [[1, -2], [Fraction(1, 2), Fraction(1, 4)], [-3, 1]]
    b = [1, -2, Fraction(1, 2)]
    net = torch.nn.Linear(2, 3)
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[float(w) for w in row] for row in W]))
        net.bias.copy_(torch.tensor([float(v) for v in b]))
    lo, hi = compute_mc_bbox(net, np.array([1.0, -1.0]), radius=3.0,
                             output_dim=output_dim)
    assert lo.dtype == hi.dtype == torch.float32
    assert tuple(lo.shape) == tuple(hi.shape) == (output_dim,)
    x_lo, x_hi = (-2, -4), (4, 2)
    for k in range(output_dim):
        low = b[k] + sum(min(w * x_lo[j], w * x_hi[j])
                         for j, w in enumerate(W[k]))
        high = b[k] + sum(max(w * x_lo[j], w * x_hi[j])
                          for j, w in enumerate(W[k]))
        assert Fraction(float(lo[k])) <= low, (k, float(lo[k]), low)
        assert Fraction(float(hi[k])) >= high, (k, float(hi[k]), high)


# --------------------------------------------------------------------------- #
# exact_volume_2d                                                               #
# --------------------------------------------------------------------------- #
def _raster_area(rects, bbox, resolution, grow):
    """Area that ``exact_volume_2d``'s raster gives the union of axis-aligned
    rectangles ``(x0, x1, y0, y1)``: the grid points of ``np.linspace`` over
    each axis, endpoints included, times the cell area. ``grow`` > 0 takes
    the closed rectangles grown by ``grow`` grid steps on each side; ``grow``
    = 0 takes the open rectangles."""
    (xmin, ymin), (xmax, ymax) = bbox
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    xx, yy = np.meshgrid(xs, ys)
    inside = np.zeros(xx.shape, dtype=bool)
    for x0, x1, y0, y1 in rects:
        if grow:
            inside |= ((xx >= x0 - grow * dx) & (xx <= x1 + grow * dx)
                       & (yy >= y0 - grow * dy) & (yy <= y1 + grow * dy))
        else:
            inside |= (xx > x0) & (xx < x1) & (yy > y0) & (yy < y1)
    return inside.sum() * (xmax - xmin) * (ymax - ymin) / resolution ** 2


def _shrink(rect, t):
    x0, x1, y0, y1 = rect
    return (x0 + t * (x1 - x0), x1 - t * (x1 - x0),
            y0 + t * (y1 - y0), y1 - t * (y1 - y0))


def test_raster_area_of_a_star_union_is_within_its_derived_bound():
    """Two disjoint members in a box of 8 x 5 (not square):
    B = (5, 17) + diag(0.5, 0.25) a with a in [10, 16] x [12, 20], the
    rectangle [10, 13] x [20, 22] (its predicate box is far from [-1, 1]);
    A = the box [14, 16] x [20, 23] cut by its own rows a0 <= 0.5 and
    a1 <= 0, the rectangle [14, 15.5] x [20, 21.5] (box rows stay in C).
    Upper bound: the raster of the members grown by the two dilation steps
    of the closing. Lower bound: each member's polygon is the hull of its
    accepted predicate samples (2400 for B, at least 750 for A except with
    probability 2e-10); it holds the member shrunk by t on every side unless
    one of four corner squares of relative side t got no sample, with t
    chosen so that each corner fails with probability at most 1e-7. Total
    probability that a correct implementation fails: below 1e-6. Catches a
    sample kept when it meets any row instead of every row, the centre left
    out of the image points, the predicate box ignored, raster rows spread
    over the x range, and an occupancy mask overwritten by each polygon."""
    b_star = Star(np.array([[5.0, 0.5, 0.0], [17.0, 0.0, 0.25]]), None, None,
                  np.array([10.0, 12.0]), np.array([16.0, 20.0]))
    box = Star.from_bounds(np.array([[14.0], [20.0]]), np.array([[16.0], [23.0]]))
    a_star = box.intersect_half_space(np.array([[1.0, 0.0], [0.0, 1.0]]),
                                      np.array([[15.5], [21.5]]))
    if a_star.C.shape[0] < 2:
        pytest.fail("the cut star should carry several rows")
    rects = [(10.0, 13.0, 20.0, 22.0), (14.0, 15.5, 20.0, 21.5)]
    bbox = ((9.0, 19.0), (17.0, 24.0))
    t_b = math.sqrt(math.log(1e7) / 2400)
    t_a = math.sqrt(math.log(1e7) / 750)
    area = exact_volume_2d([b_star, a_star], (np.array(bbox[0]),
                                              np.array(bbox[1])))
    lower = _raster_area([_shrink(rects[0], t_b), _shrink(rects[1], t_a)],
                         bbox, 500, 0)
    upper = _raster_area(rects, bbox, 500, 2)
    assert lower <= area <= upper, (lower, area, upper)
    assert exact_volume_2d([], (np.zeros(2), np.ones(2))) == 0.0


# --------------------------------------------------------------------------- #
# The public route: the exact reach set of a ReLU network                       #
# --------------------------------------------------------------------------- #
def test_volume_of_the_exact_reach_set_of_a_relu_network():
    """y = W2 relu(W1 x + b1) + b2 on the box [-1, 2] x [-1.5, 1], reached
    with ``method='exact'``. Pieces with an inactive neuron map onto
    segments (volume 0), so the image has area |det W2| |det W1| times the
    area of {x in the box : W1 x + b1 >= 0}, a polygon whose area the oracle
    computes in fractions. The exact volumes of the pieces sum to it (the
    pieces partition the box), the sound lower bound is not above it, and
    the Monte Carlo interval on the union holds it. Catches any volume
    function wrong on the stars a reach returns: rows with offsets, flat
    pieces, several members."""
    W1, b1 = [[1.0, 0.5], [-0.5, 1.0]], [0.25, -0.5]
    W2, b2 = [[2.0, -1.0], [0.5, 1.0]], [1.0, -3.0]
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU(),
                                torch.nn.Linear(2, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(W1))
        model[0].bias.copy_(torch.tensor(b1))
        model[2].weight.copy_(torch.tensor(W2))
        model[2].bias.copy_(torch.tensor(b2))
    lo, hi = np.array([-1.0, -1.5]), np.array([2.0, 1.0])
    stars = reach_model(model, Star.from_bounds(lo.reshape(2, 1),
                                                hi.reshape(2, 1)),
                        "exact", "affine-ReLU-affine")
    A, b = _box_rows(lo, hi)
    active = _polygon_area(np.vstack([A, -np.array(W1)]),
                           np.concatenate([b, b1]))
    true = abs(_det(W2)) * abs(_det(W1)) * active
    total = sum(star_volume(s, method="exact").mean for s in stars)
    assert total == pytest.approx(float(true), rel=1e-9), (total, true)
    lower = star_union_volume_sound_lower(stars).mean
    assert Fraction(lower) <= true * (1 + Fraction(1, 10**12)), (lower, true)
    estimate = star_union_volume_mc(stars, n_samples=1_500, batch_size=1_500,
                                    seed=0, confidence=_CONFIDENCE)
    assert estimate.ci_low <= true <= estimate.ci_high, (estimate, true)
