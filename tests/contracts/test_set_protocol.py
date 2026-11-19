"""Sets: what membership, construction, conversion, sampling and volume
promise (DESIGN sections 8, 11, 12 and 17).

``contains`` is exact membership, never a bounding-box test. ``to_star``
returns the same set. ``sample`` returns only points of the set. A constructor
that is told a layout checks the array against it. A volume is the volume of
the set's image, whatever the number of predicate variables.

Membership of a point in a computed Star is decided by the LP of
``tests/_oracles/membership.py``, not by the Star's own ``contains``.
"""

import numpy as np
import pytest

from n2v.sets import Hexatope, Octatope, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.volume import star_union_volume_sound_lower, star_volume
from tests._oracles.membership import LP_TOL, distance
from tests.known_bugs import DID_NOT_RAISE, known_bug


# --------------------------------------------------------------------------- #
# Zono.contains                                                                 #
# --------------------------------------------------------------------------- #
# The square with corners (2, 0), (0, 2), (-2, 0), (0, -2): the unit box
# sheared by 45 degrees. Its bounding box is [-2, 2]^2, twice its area.
_DIAMOND = Zono(np.zeros((2, 1)), np.array([[1.0, 1.0], [1.0, -1.0]]))


@pytest.mark.parametrize("point", [[0.0, 0.0], [1.9, 0.0], [1.0, 1.0],
                                   [2.0, 0.0], [-0.5, 1.5]])
def test_zono_contains_accepts_points_of_the_set(point):
    assert _DIAMOND.contains(np.array(point))


@pytest.mark.parametrize("point", [[2.5, 0.0], [0.0, -2.25], [3.0, 3.0]])
def test_zono_contains_rejects_points_outside_its_bounding_box(point):
    assert not _DIAMOND.contains(np.array(point))


# One marked case: every point between the set and its box fails on the one
# bounding-box test today. (1.25, 1.0) lies 0.25 outside the set in the sum
# of its coordinates, the nearest of the points once kept here; the passing
# neighbours are the two tests above.
@pytest.mark.parametrize("point", [pytest.param([1.25, 1.0], id="point1")])
@known_bug(8, "Zono.contains tests the bounding box, not the zonotope")
def test_zono_contains_rejects_points_between_the_set_and_its_bounding_box(
        point):
    assert not _DIAMOND.contains(np.array(point))


# --------------------------------------------------------------------------- #
# ImageStar.contains                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lp_solver", [
    "default", "highs",
    pytest.param("linprog", marks=known_bug(
        8, "ImageStar.contains hands the solver name to cvxpy, which has no "
           "'linprog'; the error is swallowed and reported as 'not a member'")),
])
def test_image_star_contains_its_centre_with_every_solver(lp_solver):
    lo = np.arange(30.0).reshape(3, 5, 2) / 4.0
    star = ImageStar.from_bounds(lo, lo + 0.5, 3, 5, 2)
    assert star.contains(lo + 0.25, lp_solver=lp_solver)
    assert not star.contains(lo + 1.0, lp_solver=lp_solver)


# --------------------------------------------------------------------------- #
# ImageStar.from_bounds                                                         #
# --------------------------------------------------------------------------- #
@known_bug(7, "ImageStar.from_bounds checks only the number of entries: a "
              "(C, H, W) array is read as (H, W, C) and the pixels are "
              "permuted", raises=DID_NOT_RAISE)
def test_image_star_from_bounds_rejects_a_channels_first_array():
    lo = np.zeros((2, 3, 5))                                # (C, H, W)
    with pytest.raises(ValueError):
        ImageStar.from_bounds(lo, lo + 1.0, height=3, width=5, num_channels=2)


@pytest.mark.parametrize("shape", [(3, 5, 2), (30,), (30, 1)])
def test_image_star_from_bounds_accepts_its_own_layouts(shape):
    """A (H, W, C) array, or a flat one in that order: entry (h, w, c) of
    the set is entry (h, w, c) of the array."""
    lo = np.arange(30.0).reshape(3, 5, 2)
    star = ImageStar.from_bounds(lo.reshape(shape), (lo + 1.0).reshape(shape),
                                 height=3, width=5, num_channels=2)
    assert distance(star, lo.reshape(-1) + 0.5) <= LP_TOL
    assert distance(star, lo.reshape(-1) + 1.5) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Hexatope and Octatope after a cut                                             #
# --------------------------------------------------------------------------- #
def _unit_square_cut(cls, bound):
    """[0, 1]^2 cut by y0 <= bound: the rectangle [0, bound] x [0, 1]."""
    square = cls.from_bounds(np.zeros((2, 1)), np.ones((2, 1)))
    return square.intersect_half_space(np.array([[1.0, 0.0]]),
                                       np.array([[bound]]))


@pytest.mark.parametrize("cls", [
    pytest.param(Hexatope, marks=known_bug(
        9, "Hexatope.to_star leaves the anchor variable free in [-1, 1], so "
           "a cut stated against the anchor no longer binds")),
    Octatope,
])
def test_to_star_after_a_cut_is_the_same_set(cls):
    """Replaces the assertion of ``test_hexatope_to_star_basic`` that every
    predicate variable, the anchor included, ranges over [-1, 1]. On a set
    built by ``from_bounds`` alone the free anchor cannot be seen."""
    star = _unit_square_cut(cls, 0.6).to_star()
    for inside in ([0.0, 0.0], [0.6, 1.0], [0.3, 0.5], [0.6, 0.0], [0.0, 1.0]):
        assert distance(star, inside) <= LP_TOL, inside
    escaped = [p for p in ([0.7, 0.5], [0.9, 0.5], [1.0, 1.0])
               if distance(star, p) <= LP_TOL]
    assert not escaped, (
        f"{cls.__name__}.to_star() contains {escaped}, which the set "
        f"[0, 0.6] x [0, 1] does not")


@pytest.mark.parametrize("cls", [
    pytest.param(Hexatope, marks=known_bug(
        9, "Hexatope.sample draws the anchor variable from [-1, 1] instead of "
           "fixing it at 0, so it returns points outside the set")),
    Octatope,
])
def test_sample_after_a_cut_returns_points_of_the_set(cls):
    points = _unit_square_cut(cls, 0.6).sample(200)
    assert points.shape[1] == 2 and len(points) >= 1
    # 1e-9: the sampler's own rejection test admits 1e-10 in the variables.
    outside = points[points[:, 0] > 0.6 + 1e-9]
    assert len(outside) == 0, (
        f"{len(outside)} of {len(points)} samples have y0 > 0.6 "
        f"(largest {points[:, 0].max():.3f})")


@pytest.mark.parametrize("cls", [Hexatope, Octatope])
@known_bug(9, "sample() fills the samples it could not draw with the centre "
              "of the generators, whether or not the centre is in the set")
def test_sample_of_a_thin_set_returns_points_of_the_set(cls):
    """[0, 0.001] x [0, 1]: rejection accepts one draw in a thousand. Fewer
    points than asked for is allowed; a point outside the set is not."""
    points = _unit_square_cut(cls, 0.001).sample(50)
    outside = points[points[:, 0] > 0.001 + 1e-9]
    assert len(outside) == 0, (
        f"{len(outside)} of {len(points)} samples have y0 > 0.001")


# --------------------------------------------------------------------------- #
# star_volume                                                                   #
# --------------------------------------------------------------------------- #
def _assert_volume(estimate, method, true_volume):
    if method == "mc":
        assert estimate.ci_low <= true_volume <= estimate.ci_high, estimate
    else:
        assert estimate.mean == pytest.approx(true_volume, rel=1e-9), estimate


@pytest.mark.parametrize("method", ["exact", "mc"])
def test_star_volume_of_a_cut_box(method):
    """[0, 2] x [1, 4] cut by a0 + a1 <= 0, half of it: volume 3."""
    box = Star.from_bounds(np.array([[0.0], [1.0]]), np.array([[2.0], [4.0]]))
    half = Star(box.V, np.array([[1.0, 1.0]]), np.array([[0.0]]),
                box.predicate_lb, box.predicate_ub)
    _assert_volume(star_volume(half, method=method), method, 3.0)


# x = a0 + 0.01 a1 with a in [-1, 1]^2: the segment [-1.01, 1.01].
_SEGMENT = Star(np.array([[0.0, 1.0, 0.01]]), np.zeros((0, 2)),
                np.zeros((0, 1)), -np.ones((2, 1)), np.ones((2, 1)))
_WIDE_BASIS = known_bug(
    8, "the volume of a star with more predicate variables than outputs is "
       "the predicate volume times a row-space factor: 4.0002 for a segment "
       "of length 2.02")


@pytest.mark.parametrize("method", ["exact", "mc"])
@_WIDE_BASIS
def test_star_volume_with_more_variables_than_outputs(method):
    """Refusing is correct (vertex enumeration in predicate space does not
    give the image's volume); a number must be the length of the segment."""
    try:
        estimate = star_volume(_SEGMENT, method=method)
    except NotImplementedError:
        return
    _assert_volume(estimate, method, 2.02)


@_WIDE_BASIS
def test_sound_lower_volume_with_more_variables_than_outputs():
    try:
        estimate = star_union_volume_sound_lower([_SEGMENT])
    except NotImplementedError:
        return
    assert estimate.mean <= 2.02, (
        f"a lower bound of {estimate.mean} on a volume of 2.02")


@pytest.mark.parametrize("method", ["exact", "mc"])
@known_bug(8, "star_volume assumes the predicate box [-1, 1]^n when the star "
              "has none: 4 for the square [-3, 3]^2")
def test_star_volume_of_a_star_without_a_predicate_box(method):
    """x = a with |a_i| <= 3 given as rows only: the square of area 36."""
    rows = np.vstack([np.eye(2), -np.eye(2)])
    star = Star(np.hstack([np.zeros((2, 1)), np.eye(2)]), rows,
                3.0 * np.ones((4, 1)))
    if star.predicate_lb is not None or star.predicate_ub is not None:
        # The case's own setup: not absorbed by the marker.
        pytest.fail("Star() now gives a star built from rows a predicate box")
    _assert_volume(star_volume(star, method=method), method, 36.0)
