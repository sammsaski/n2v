"""Self-test of the float membership oracle (``membership.py``).

The oracle answers "how far is this point from this set" with one LP, and the
point-replay tests rest on the answer. Every distance here is known by hand
or confirmed in exact rationals by ``rational.meets_box``, which shares no
code with the LP: the distance from ``y`` to a set is at most ``t`` exactly
when the set meets the box of half-width ``t`` around ``y``. The sets are
built from arrays written out in the test, so :func:`membership.rows`, the one
place either oracle reads an n2v object, is compared with what was stored and
never with an n2v method.

The planted bugs are the ones a bounding-box check cannot see: a set whose box
is right and whose rows are wrong, a point given in the wrong image layout, an
empty set, a constraint row or a centre that is dropped.

Hexatopes and octatopes are built from constraint lists written out in the
test, and their exact counterparts from the meaning written out beside them
(one row per constraint, the hexatope's anchor pinned to 0), never through
``membership.rows``. Their planted bugs: a binding two-variable row that is
dropped or read the wrong way round, an anchor left free, an unbounded set
read as bounded. A predicate box as narrow as the solver tolerance (where
the unscaled LP called 210 of 300 members non-members) checks the scaled
LP, and planted solver points check that an
answer is recomputed at the solver's point, never read from its objective.
"""

from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.sets.halfspace import HalfSpace
from n2v.sets.hexatope import DifferenceConstraintSystem
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.sets.octatope import UTVPIConstraintSystem

from tests._oracles import membership as M
from tests._oracles import rational as R

NEAR = 1e-9         # what a solved LP on a handful of variables is good to


def _column(values):
    return None if values is None else np.array(values, dtype=float).reshape(-1, 1)


def _star(c, G, C=(), d=(), lb=None, ub=None):
    """A Star from its centre, generators, rows and predicate bounds."""
    G = np.array(G, dtype=float).reshape(len(c), -1)
    n_var = G.shape[1]
    V = np.hstack([np.array(c, dtype=float).reshape(-1, 1), G])
    return Star(V, np.array(C, dtype=float).reshape(-1, n_var), _column(d),
                _column(lb), _column(ub))


def _exact(c, G, C=(), d=(), lb=None, ub=None):
    """The same set for the rational oracle, from the same lists."""
    n_var = len(G[0])
    bound = lambda values: ([None] * n_var if values is None       # noqa: E731
                            else [R.frac(v) for v in values])
    return (R.vector(c), R.matrix(G), [R.vector(row) for row in C],
            R.vector(d), bound(lb), bound(ub))


# The square with corners (2, 0), (0, 2), (-2, 0), (0, -2): its bounding box
# is [-2, 2] x [-2, 2] and holds four corners the set does not.
DIAMOND = dict(c=[0, 0], G=[[1, 1], [1, -1]], lb=[-1, -1], ub=[1, 1])
# The lower-left half of the unit square of half-width 1: a1 + a2 <= 0.
TRIANGLE = dict(c=[0, 0], G=[[1, 0], [0, 1]], C=[[1, 1]], d=[0],
                lb=[-1, -1], ub=[1, 1])


# --------------------------------------------------------------------------- #
# Reading a set                                                                 #
# --------------------------------------------------------------------------- #
def test_a_star_is_read_as_it_is_stored():
    s = _star([1, 2], [[1, 2], [3, 4]], C=[[1, 1], [-1, 2]], d=[5, 6],
              lb=[-1, -2], ub=[3, 4])
    c, G, C, d, lb, ub = M.rows(s)
    assert c.tolist() == [1, 2]
    assert G.tolist() == [[1, 2], [3, 4]]
    assert C.tolist() == [[1, 1], [-1, 2]]
    assert d.tolist() == [5, 6]
    assert lb.tolist() == [-1, -2]
    assert ub.tolist() == [3, 4]


def test_a_star_without_rows_or_bounds_is_read_as_unconstrained():
    c, G, C, d, lb, ub = M.rows(_star([1, 2], [[1, 0, 2], [0, 1, 3]]))
    assert C.shape == (0, 3) and d.shape == (0,)
    assert lb.tolist() == [-np.inf] * 3 and ub.tolist() == [np.inf] * 3
    assert c.tolist() == [1, 2] and G.tolist() == [[1, 0, 2], [0, 1, 3]]
    # One side of the bounds only.
    _, _, _, _, lb, ub = M.rows(_star([0], [[1, 1]], ub=[2, 3]))
    assert lb.tolist() == [-np.inf] * 2 and ub.tolist() == [2, 3]
    _, _, _, _, lb, ub = M.rows(_star([0], [[1, 1]], lb=[-2, -3]))
    assert lb.tolist() == [-2, -3] and ub.tolist() == [np.inf] * 2


def test_a_zonotope_is_read_as_generators_on_the_unit_cube():
    z = Zono(_column([1, 2]), np.array([[1., 0, 5], [0, 1, 7]]))
    c, G, C, d, lb, ub = M.rows(z)
    assert c.tolist() == [1, 2]
    assert G.tolist() == [[1, 0, 5], [0, 1, 7]]
    assert C.shape == (0, 3) and d.shape == (0,)
    assert lb.tolist() == [-1] * 3 and ub.tolist() == [1] * 3


# Three different sizes: with two alike, reading the tensor in the wrong
# shape can give the right vector.
HEIGHT, WIDTH, CHANNELS = 2, 3, 4


def _pixel(h, w, k):
    """A number that names the pixel: no two layouts give the same vector."""
    return 100.0 * h + 10.0 * w + k


def _stored_order(entry):
    """The flat vector an image set stores, written as three loops."""
    return [entry(h, w, k) for h in range(HEIGHT) for w in range(WIDTH)
            for k in range(CHANNELS)]


def _image_star():
    """Every pixel has its own centre; the one generator moves the pixel at
    height 1, width 2, channel 0 by up to 3 each way."""
    V = np.zeros((HEIGHT, WIDTH, CHANNELS, 2))
    for h in range(HEIGHT):
        for w in range(WIDTH):
            for k in range(CHANNELS):
                V[h, w, k, 0] = _pixel(h, w, k)
    V[1, 2, 0, 1] = 3.0
    return ImageStar(V, np.zeros((0, 1)), np.zeros((0, 1)), _column([-1]),
                     _column([1]), HEIGHT, WIDTH, CHANNELS)


def _image_zono():
    centre = _column(_stored_order(_pixel))
    generator = _column(_stored_order(
        lambda h, w, k: 3.0 * ((h, w, k) == (1, 2, 0))))
    return ImageZono(centre, generator, HEIGHT, WIDTH, CHANNELS)


@pytest.mark.parametrize("build", [_image_star, _image_zono])
def test_an_image_set_is_read_height_by_width_by_channel(build):
    c, G, C, d, lb, ub = M.rows(build())
    assert c.tolist() == _stored_order(_pixel)
    assert G.shape == (HEIGHT * WIDTH * CHANNELS, 1)
    assert G[:, 0].tolist() == _stored_order(
        lambda h, w, k: 3.0 * ((h, w, k) == (1, 2, 0)))
    assert C.shape == (0, 1) and d.shape == (0,)
    assert lb.tolist() == [-1] and ub.tolist() == [1]


@pytest.mark.parametrize("thing", [
    HalfSpace(np.array([[1., 0]]), np.array([[1.]])), np.zeros(3), None,
    "a star"])
def test_a_set_with_no_reader_is_refused(thing):
    with pytest.raises(TypeError, match="no row reader"):
        M.rows(thing)


# --------------------------------------------------------------------------- #
# Distances known by hand                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("point, expected", [
    ([0.5, 1.5], 0.0), ([0, 1], 0.0), ([1, 2], 0.0),         # inside, corners
    ([1.25, 1.5], 0.25), ([-0.5, 1.5], 0.5),                 # off one side
    ([0.5, 2.75], 0.75), ([0.5, 0.875], 0.125),
    ([3, 2.5], 2.0), ([-1, -2], 3.0),                        # off two: the larger
    ([-0.5, 2.75], 0.75), ([1.5, 0.75], 0.5),                # below one, above one
])
def test_the_distance_to_a_box_is_its_largest_overshoot(point, expected):
    assert M.distance(Box(np.array([0., 1]), np.array([1., 2])), point) \
        == expected


@pytest.mark.parametrize("point, expected", [
    ([0, 0], 0.0), ([1, 1], 0.0), ([2, 0], 0.0), ([0, -2], 0.0),
    ([1.5, 1.5], 0.5), ([2, 2], 1.0), ([-2, 2], 1.0), ([2, -2], 1.0),
    ([-1.25, -1.25], 0.25), ([3, 0], 1.0), ([0, -2.5], 0.5),
])
def test_the_distance_to_a_star_is_to_its_rows_not_its_box(point, expected):
    """Every point here is inside the diamond's bounding box or beside it;
    the corners of the box are a whole unit away from the set."""
    assert M.distance(_star(**DIAMOND), point) == pytest.approx(
        expected, abs=NEAR)


@pytest.mark.parametrize("point, expected", [
    ([-0.5, -0.5], 0.0), ([1, -1], 0.0), ([0.25, -0.25], 0.0),
    ([0.5, 0.5], 0.5), ([1, 1], 1.0), ([0.75, -0.25], 0.25),
    ([-1.5, 0], 0.5), ([2, 2], 2.0),
])
def test_a_constraint_row_cuts_the_set(point, expected):
    assert M.distance(_star(**TRIANGLE), point) == pytest.approx(
        expected, abs=NEAR)


def test_a_star_without_predicate_bounds_is_unbounded():
    """The half-plane ``x1 <= 0``: the row is the only limit."""
    half_plane = _star([0, 0], [[1, 0], [0, 1]], C=[[1, 0]], d=[0])
    assert M.distance(half_plane, [-1e6, 1e6]) == pytest.approx(0, abs=NEAR)
    assert M.distance(half_plane, [3, 7e3]) == pytest.approx(3, abs=NEAR)
    # Bounded on one side of one variable only.
    quadrant = _star([0, 0], [[1, 0], [0, 1]], lb=[0, 0])
    assert M.distance(quadrant, [5, 9]) == pytest.approx(0, abs=NEAR)
    assert M.distance(quadrant, [-2, 9]) == pytest.approx(2, abs=NEAR)
    assert M.distance(quadrant, [4, -0.5]) == pytest.approx(0.5, abs=NEAR)
    ceiling = _star([0, 0], [[1, 0], [0, 1]], ub=[0, 0])
    assert M.distance(ceiling, [-5, -9]) == pytest.approx(0, abs=NEAR)
    assert M.distance(ceiling, [-5, 1.5]) == pytest.approx(1.5, abs=NEAR)


def test_the_centre_moves_the_set():
    moved = _star([10, -20], DIAMOND["G"], lb=[-1, -1], ub=[1, 1])
    assert M.distance(moved, [11, -19]) == pytest.approx(0, abs=NEAR)
    assert M.distance(moved, [12, -18]) == pytest.approx(1, abs=NEAR)
    # From the origin the nearest point is the corner (10, -18): 18 away in
    # the second coordinate, which is the larger.
    assert M.distance(moved, [0, 0]) == pytest.approx(18, abs=NEAR)
    assert M.distance(moved, [10, 0]) == pytest.approx(18, abs=NEAR)
    assert M.distance(moved, [14, -20]) == pytest.approx(2, abs=NEAR)


def test_a_generator_is_read_with_its_own_variable():
    """Two generators of different lengths, and bounds that are not
    symmetric: swapping the columns or the bounds changes the set."""
    s = _star([0, 0], [[4, 0], [0, 1]], lb=[0, -1], ub=[1, 3])
    for point, expected in [([4, 3], 0), ([0, -1], 0), ([2, 1], 0),
                            ([5, 0], 1), ([-0.5, 0], 0.5), ([2, 3.5], 0.5),
                            ([2, -1.25], 0.25), ([4.5, 4], 1)]:
        assert M.distance(s, point) == pytest.approx(expected, abs=NEAR), point


def test_the_distance_to_a_zonotope_is_to_the_set_not_its_box():
    """``c + (a1 + a3, a2 + a3)``: a hexagon, the square of half-width 2 with
    two opposite corners cut off."""
    z = Zono(_column([1, 2]), np.array([[1., 0, 1], [0, 1, 1]]))
    for point, expected in [([1, 2], 0), ([3, 4], 0), ([-1, 0], 0),
                            ([3, 2], 0), ([1, 4], 0),
                            ([3, 0], 1), ([-1, 4], 1), ([2.5, 0.5], 0.5),
                            ([4, 5], 1), ([1, 5], 1), ([-2, 2], 1)]:
        assert M.distance(z, point) == pytest.approx(expected, abs=NEAR), point


def test_an_empty_set_is_infinitely_far_from_every_point():
    empty = _star([0, 0], [[1, 0], [0, 1]], C=[[1, 0], [-1, 0]], d=[-1, -1],
                  lb=[-5, -5], ub=[5, 5])                # a1 <= -1, a1 >= 1
    for point in ([0, 0], [1, 0], [-1, 0], [100, 100]):
        assert M.distance(empty, point) == float("inf")
    crossed = _star([0], [[1]], lb=[1], ub=[-1])         # bounds that cross
    assert M.distance(crossed, [0]) == float("inf")


def test_a_point_of_the_wrong_size_is_an_error():
    """A point with one entry would otherwise be spread over every row."""
    box = Box(np.zeros(2), np.ones(2))
    for s in (box, _star(**DIAMOND),
              Zono(_column([0, 0]), np.array([[1., 0], [0, 1]]))):
        for point in ([0.5], [0.5, 0.5, 0.5], []):
            with pytest.raises(AssertionError, match="entries"):
                M.distance(s, point)
        assert M.distance(s, [[0.5], [0.5]]) == pytest.approx(0, abs=NEAR)


class _Solved:
    def __init__(self, status):
        self.status, self.fun, self.message = status, 0.0, "stopped early"


def test_a_solve_that_fails_is_an_error_and_never_an_answer(monkeypatch):
    for status in (1, 3, 4):
        monkeypatch.setattr(M, "linprog", lambda *a, s=status, **k: _Solved(s))
        with pytest.raises(RuntimeError, match="stopped early"):
            M.distance(_star(**DIAMOND), [0, 0])
    monkeypatch.setattr(M, "linprog", lambda *a, **k: _Solved(2))
    assert M.distance(_star(**DIAMOND), [0, 0]) == float("inf")


# --------------------------------------------------------------------------- #
# Image layout                                                                  #
# --------------------------------------------------------------------------- #
def _tensor(entry):
    """A (channels, height, width) tensor, the layout a network returns."""
    t = np.zeros((CHANNELS, HEIGHT, WIDTH))
    for h in range(HEIGHT):
        for w in range(WIDTH):
            for k in range(CHANNELS):
                t[k, h, w] = entry(h, w, k)
    return t


def test_hwc_puts_a_tensor_in_the_stored_order():
    assert M.hwc(_tensor(_pixel), CHANNELS, HEIGHT, WIDTH).tolist() \
        == _stored_order(_pixel)
    # A flat tensor or a batch of one is the same tensor.
    flat = _tensor(_pixel).reshape(-1)
    assert M.hwc(flat, CHANNELS, HEIGHT, WIDTH).tolist() \
        == _stored_order(_pixel)
    assert M.hwc(_tensor(_pixel)[None], CHANNELS, HEIGHT, WIDTH).tolist() \
        == _stored_order(_pixel)
    # A square image with as many channels as rows: every wrong order of the
    # three axes still has the right shape.
    cube = np.arange(27.0).reshape(3, 3, 3)
    assert M.hwc(cube, 3, 3, 3).tolist() == [
        cube[k, h, w] for h in range(3) for w in range(3) for k in range(3)]


@pytest.mark.parametrize("build", [_image_star, _image_zono])
def test_an_image_point_is_compared_in_the_stored_layout(build):
    s = build()
    centre = _tensor(_pixel)
    assert M.distance(s, M.hwc(centre, CHANNELS, HEIGHT, WIDTH)) \
        == pytest.approx(0, abs=NEAR)
    # The same tensor flattened as the network returns it is another point.
    assert M.distance(s, centre.reshape(-1)) > 1
    # The generator moves one pixel, and only that one.
    moved = centre.copy()
    moved[0, 1, 2] += 2.5                  # channel 0, height 1, width 2
    assert M.distance(s, M.hwc(moved, CHANNELS, HEIGHT, WIDTH)) \
        == pytest.approx(0, abs=NEAR)
    moved[0, 1, 2] += 1.0                  # 3.5: half a unit past its reach
    assert M.distance(s, M.hwc(moved, CHANNELS, HEIGHT, WIDTH)) \
        == pytest.approx(0.5, abs=NEAR)
    for other in [(1, 1, 2), (3, 1, 2), (0, 0, 2), (0, 1, 1), (2, 0, 1)]:
        elsewhere = centre.copy()
        elsewhere[other] += 2.5
        assert M.distance(s, M.hwc(elsewhere, CHANNELS, HEIGHT, WIDTH)) \
            == pytest.approx(2.5, abs=NEAR), other


# --------------------------------------------------------------------------- #
# Points that no set holds                                                      #
# --------------------------------------------------------------------------- #
def test_a_point_is_covered_when_one_set_holds_it():
    left = Box(np.array([0., 0]), np.array([1., 1]))
    right = Box(np.array([2., 0]), np.array([3., 1]))
    points = [[0.5, 0.5], [2.5, 0.5], [1.5, 0.5], [0.5, 0.5], [4, 0.5]]
    assert M.uncovered([left, right], points) == [(2, 0.5), (4, 1.0)]
    assert M.uncovered([right, left], points) == [(2, 0.5), (4, 1.0)]
    assert M.uncovered([left], points) == [(1, 1.5), (2, 0.5), (4, 3.0)]
    assert M.uncovered([left, right], []) == []
    # The distance reported is to the nearest set, whichever comes first.
    near = Box(np.array([1.25, 0]), np.array([1.375, 1]))
    assert M.uncovered([left, near, right], points) == [(2, 0.125), (4, 1.0)]
    assert M.uncovered([near, right, left], points) == [(2, 0.125), (4, 1.0)]


def test_no_set_and_an_empty_set_cover_nothing():
    points = [[0, 0], [1, 1]]
    assert M.uncovered([], points) == [(0, float("inf")), (1, float("inf"))]
    empty = _star([0, 0], [[1, 0], [0, 1]], C=[[1, 0], [-1, 0]], d=[-1, -1],
                  lb=[-5, -5], ub=[5, 5])
    assert M.uncovered([empty], points) \
        == [(0, float("inf")), (1, float("inf"))]
    # An empty set beside a real one changes nothing.
    assert M.uncovered([empty, Box(np.zeros(2), np.ones(2))], points) == []


def test_the_only_slack_is_the_solver_tolerance():
    unit = Box(np.zeros(1), np.ones(1))
    assert M.LP_TOL == 1e-7
    # Below the tolerance a point is a member; above it, not.
    assert M.uncovered([unit], [[1 + 5e-8], [-5e-8]]) == []
    (missed,) = M.uncovered([unit], [[1 + 2e-7]])
    assert missed[0] == 0 and missed[1] == pytest.approx(2e-7, rel=1e-6)
    # The tolerance is relative for large points and absolute for small ones.
    big = Box(np.array([0.]), np.array([1000.]))
    assert M.uncovered([big], [[1000 + 5e-5]]) == []
    assert [i for i, _ in M.uncovered([big], [[1000 + 2e-4]])] == [0]
    small = Box(np.array([0.]), np.array([1e-3]))
    assert M.uncovered([small], [[1e-3 + 5e-8]]) == []
    assert [i for i, _ in M.uncovered([small], [[1e-3 + 2e-7]])] == [0]
    # A caller's own tolerance replaces it.
    assert M.uncovered([unit], [[1.25]], tol=0.5) == []
    assert M.uncovered([unit], [[1 + 5e-8]], tol=0.0) == [
        (0, pytest.approx(5e-8, rel=1e-6))]
    # The same through the LP of a star.
    diamond = _star(**DIAMOND)
    assert M.uncovered([diamond], [[1 + 2e-8, 1 + 2e-8], [1, 1]]) == []
    assert [i for i, _ in M.uncovered(
        [diamond], [[1, 1], [1 + 1e-6, 1 + 1e-6]])] == [1]


# --------------------------------------------------------------------------- #
# The planted bug a bounding box cannot see                                     #
# --------------------------------------------------------------------------- #
def _relu_relaxation(lower_row, lower_rhs):
    """A set in the plane of (x, y) for x in [-1, 1], below the chord
    ``y <= (x + 1) / 2`` and above ``y >= 0`` and one more row."""
    return _star([0, 0], [[1, 0], [0, 1]],
                 C=[[0, -1], [-0.5, 1], lower_row], d=[0, 0.5, lower_rhs],
                 lb=[-1, 0], ub=[1, 1])


def _bounding_box(s):
    """The box of a planar set, from the exact vertices of its rows."""
    c, G, C, d, lb, ub = M.rows(s)
    corners = R.set_vertices(_exact(c.tolist(), G.tolist(), C.tolist(),
                                    d.tolist(), lb.tolist(), ub.tolist()))
    return [(min(p[i] for p in corners), max(p[i] for p in corners))
            for i in range(2)]


def test_a_relaxation_with_the_right_box_and_a_wrong_row_is_caught():
    xs = np.linspace(-1, 1, 41)
    graph = [[x, max(x, 0.0)] for x in xs]
    sound = _relu_relaxation([1, -1], 0)                     # y >= x
    assert M.uncovered([sound], graph) == []
    # The planted bug: y >= (x + 1) / 4 in place of y >= x. The set still
    # reaches both ends of the graph, so its bounding box is the same.
    planted = _relu_relaxation([0.25, -1], -0.25)
    assert _bounding_box(planted) == _bounding_box(sound) == [(-1, 1), (0, 1)]
    missed = dict(M.uncovered([planted], graph))
    assert 0 not in missed and 40 not in missed              # the two ends
    assert sorted(missed) == list(range(1, 27))              # -1 < x < 1/3
    assert missed[20] == pytest.approx(0.2, abs=NEAR)        # the kink (0, 0)
    assert max(missed.values()) == pytest.approx(0.2, abs=NEAR)
    # Beside the sound set the planted one costs nothing.
    assert M.uncovered([planted, sound], graph) == []


# --------------------------------------------------------------------------- #
# Every distance confirmed in rationals                                         #
# --------------------------------------------------------------------------- #
def _random_set(rng):
    """Lists for a star with dyadic entries and finite predicate bounds."""
    dim, n_var = int(rng.integers(1, 4)), int(rng.integers(1, 5))
    n_rows = int(rng.integers(0, 4))
    dyadic = lambda *shape: (rng.integers(-16, 17, shape) / 8).tolist()  # noqa: E731
    lb = (-rng.integers(0, 9, n_var) / 4).tolist()
    ub = (rng.integers(0, 9, n_var) / 4).tolist()
    return dict(c=dyadic(dim), G=dyadic(dim, n_var), C=dyadic(n_rows, n_var),
                d=(rng.integers(-12, 13, n_rows) / 8).tolist(), lb=lb, ub=ub)


@pytest.mark.parametrize("seed", range(40))
def test_the_distance_is_confirmed_in_rationals(seed):
    rng = np.random.default_rng(1000 + seed)
    lists = _random_set(rng)
    s, exact = _star(**lists), _exact(**lists)
    dim = len(lists["c"])
    tol = F(1, 10 ** 7)
    everywhere = [F(10 ** 6)] * dim
    is_empty = not R.meets_box(exact, [-v for v in everywhere], everywhere)
    points = (rng.integers(-40, 41, (6, dim)) / 8).tolist()
    # Members by construction: images of predicate points that keep the rows.
    for _ in range(40):
        a = [lo + (hi - lo) * t for lo, hi, t in
             zip(lists["lb"], lists["ub"], rng.integers(0, 9, len(lists["lb"])) / 8)]
        if all(np.dot(row, a) <= rhs for row, rhs in zip(lists["C"], lists["d"])):
            points.append((np.array(lists["c"]) + np.array(lists["G"]) @ a).tolist())
    for y in points:
        found = M.distance(s, y)
        if is_empty:
            assert found == float("inf")
            continue
        y = R.vector(y)
        reach = R.frac(found) + tol
        assert R.meets_box(exact, [v - reach for v in y],
                           [v + reach for v in y]), (y, found)
        short = R.frac(found) - tol
        if short >= 0:
            assert not R.meets_box(exact, [v - short for v in y],
                                   [v + short for v in y]), (y, found)
        if R.member(exact, y):
            assert found <= M.LP_TOL
        assert M.uncovered([s], [[float(v) for v in y]]) == (
            [] if R.member(exact, y) or found <= M.LP_TOL else [(0, found)])


def test_the_random_sets_cover_every_case():
    """The confirmation above is only worth what its sets are: some empty,
    some with rows that cut, points inside, outside and on the edge."""
    empty = cut = inside = outside = 0
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        lists = _random_set(rng)
        exact, dim = _exact(**lists), len(lists["c"])
        everywhere = [F(10 ** 6)] * dim
        if not R.meets_box(exact, [-v for v in everywhere], everywhere):
            empty += 1
            continue
        uncut = _exact(**dict(lists, C=[], d=[]))
        for y in (rng.integers(-40, 41, (6, dim)) / 8).tolist():
            if R.member(exact, y):
                inside += 1
            else:
                outside += 1
                cut += R.member(uncut, y)
    assert empty >= 4 and cut >= 10 and inside >= 15 and outside >= 100, \
        (empty, cut, inside, outside)


# --------------------------------------------------------------------------- #
# A box, a hexatope and an octatope are read as they are stored                #
# --------------------------------------------------------------------------- #
def test_a_box_is_read_as_identity_generators_on_its_bounds():
    c, G, C, d, lb, ub = M.rows(Box(np.array([-1., 0.5, 2]),
                                    np.array([0.25, 0.5, 3])))
    assert c.tolist() == [0, 0, 0]
    assert G.tolist() == np.eye(3).tolist()
    assert C.shape == (0, 3) and d.shape == (0,)
    assert lb.tolist() == [-1, 0.5, 2] and ub.tolist() == [0.25, 0.5, 3]


def _hexatope(c, G, constraints):
    """A Hexatope from its centre, generators and ``(i, j, b)`` rows
    ``x_i - x_j <= b``; variable 0 is the anchor."""
    G = np.array(G, dtype=float).reshape(len(c), -1)
    dcs = DifferenceConstraintSystem(G.shape[1])
    for i, j, b in constraints:
        dcs.add_constraint(i, j, b)
    return Hexatope(np.array(c, dtype=float), G, dcs)


def _octatope(c, G, constraints):
    """An Octatope from its centre, generators and ``(i, j, ai, aj, b)`` rows
    ``ai x_i + aj x_j <= b``."""
    G = np.array(G, dtype=float).reshape(len(c), -1)
    utvpi = UTVPIConstraintSystem(G.shape[1])
    for i, j, ai, aj, b in constraints:
        utvpi.add_constraint(i, j, ai, aj, b)
    return Octatope(np.array(c, dtype=float), G, utvpi)


def _hexatope_lists(c, G, constraints):
    """The same hexatope as lists for :func:`_star` and :func:`_exact`: the
    meaning written out, one row per constraint and the anchor pinned."""
    p = len(G[0])
    C = []
    for i, j, _ in constraints:
        row = [0.0] * p
        row[i] += 1.0
        row[j] -= 1.0
        C.append(row)
    lb = [0.0] + [-np.inf] * (p - 1)
    ub = [0.0] + [np.inf] * (p - 1)
    return dict(c=c, G=G, C=C, d=[b for _, _, b in constraints], lb=lb, ub=ub)


def _octatope_lists(c, G, constraints):
    p = len(G[0])
    C = []
    for i, j, ai, aj, _ in constraints:
        row = [0.0] * p
        row[i] += ai
        row[j] += aj
        C.append(row)
    return dict(c=c, G=G, C=C, d=[b for *_, b in constraints],
                lb=[-np.inf] * p, ub=[np.inf] * p)


def _exact_open(c, G, C=(), d=(), lb=None, ub=None):
    """:func:`_exact` with an infinite bound read as no bound."""
    open_ = lambda values: None if values is None else [            # noqa: E731
        None if np.isinf(v) else v for v in values]
    n_var = len(G[0])
    bound = lambda values: ([None] * n_var if values is None       # noqa: E731
                            else [None if v is None else R.frac(v)
                                  for v in values])
    return (R.vector(c), R.matrix(G), [R.vector(row) for row in C],
            R.vector(d), bound(open_(lb)), bound(open_(ub)))


def test_a_hexatope_is_read_as_its_difference_rows_with_the_anchor_pinned():
    """Every stored constraint is one row, read with its own two variables in
    its own direction; the anchor is pinned whatever its generator column
    holds, and no other variable gets a bound."""
    h = _hexatope([1, 2], [[5, 1, 2, 0], [7, 3, 4, 1]],
                  [(1, 0, 0.5), (0, 2, 0.25), (3, 1, -2), (2, 3, 4)])
    c, G, C, d, lb, ub = M.rows(h)
    assert c.tolist() == [1, 2]
    assert G.tolist() == [[5, 1, 2, 0], [7, 3, 4, 1]]
    assert C.tolist() == [[-1, 1, 0, 0], [1, 0, -1, 0], [0, -1, 0, 1],
                          [0, 0, 1, -1]]
    assert d.tolist() == [0.5, 0.25, -2, 4]
    assert lb.tolist() == [0, -np.inf, -np.inf, -np.inf]
    assert ub.tolist() == [0, np.inf, np.inf, np.inf]


def test_an_octatope_is_read_as_its_rows_with_both_coefficients():
    """Two variables in either order and either sign; on one variable the two
    coefficients add (``x_i <= b``, ``2 x_i <= b``, ``0 <= b``)."""
    o = _octatope([0, 1], [[1, 2, 3], [4, 5, 6]],
                  [(0, 2, 1, -1, 1.5), (2, 1, -1, -1, 2), (1, 1, 1, 0, 3),
                   (2, 2, 0, -1, 0.5), (0, 0, 1, 1, 1), (1, 1, 1, -1, 7)])
    c, G, C, d, lb, ub = M.rows(o)
    assert c.tolist() == [0, 1] and G.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert C.tolist() == [[1, 0, -1], [0, -1, -1], [0, 1, 0], [0, 0, -1],
                          [2, 0, 0], [0, 0, 0]]
    assert d.tolist() == [1.5, 2, 3, 0.5, 1, 7]
    assert lb.tolist() == [-np.inf] * 3 and ub.tolist() == [np.inf] * 3


def test_a_star_on_no_variable_keeps_its_rows():
    """A point star stores rows on no variable; ``0 <= d`` is the whole set
    condition, and a negative entry empties the set. Dropping those rows
    read an empty set as the point."""
    point = np.array([[1.0], [2.0]])
    empty = Star(point, np.zeros((1, 0)), np.array([[-1.0]]))
    c, G, C, d, lb, ub = M.rows(empty)
    assert C.shape == (1, 0) and d.tolist() == [-1]
    assert M.distance(empty, [1, 2]) == float("inf")
    assert R.member(R.star(empty), [1, 2]) is False
    kept = Star(point, np.zeros((2, 0)), np.array([[0.0], [3.0]]))
    assert M.distance(kept, [1, 2]) == 0
    assert M.distance(kept, [1.5, 2]) == 0.5
    assert R.member(R.star(kept), [1, 2])


# --------------------------------------------------------------------------- #
# Hexatope and octatope membership                                              #
# --------------------------------------------------------------------------- #
# x in [-1, 1]^2 with x1 - x2 <= 1/2, identity generators: the square with the
# corner (1, -1) cut off along x1 - x2 = 1/2. Its bounding box is the square.
HEX_CUT = [(1, 0, 1), (0, 1, 1), (2, 0, 1), (0, 2, 1), (1, 2, 0.5)]
HEX_CUT_SET = dict(c=[0, 0], G=[[0, 1, 0], [0, 0, 1]], constraints=HEX_CUT)
# x in [-1, 1]^2 with x0 + x1 <= 1/2: the corner (1, 1) cut off.
OCT_CUT = [(0, 0, 1, 0, 1), (0, 0, -1, 0, 1), (1, 1, 1, 0, 1),
           (1, 1, -1, 0, 1), (0, 1, 1, 1, 0.5)]
OCT_CUT_SET = dict(c=[0, 0], G=[[1, 0], [0, 1]], constraints=OCT_CUT)


@pytest.mark.parametrize("point, expected", [
    ([0, 0], 0.0), ([1, 0.5], 0.0), ([0.75, 0.25], 0.0), ([1, 1], 0.0),
    ([-1, -1], 0.0), ([-0.5, -1], 0.0), ([-1, 1], 0.0),           # members
    ([1, -1], 0.75), ([1, 0], 0.25), ([0.5, -0.5], 0.25),         # cut corner
    ([1.5, 0.5], 0.5), ([0, -2], 1.0), ([3, 3], 2.0),             # outside box
])
def test_a_difference_row_between_two_variables_cuts_the_hexatope(point,
                                                                  expected):
    """The row x1 - x2 <= 1/2 binds: the box corner (1, -1) and the points
    near it are in the bounding box and not in the set. A reader that drops
    the row, or reads it as x2 - x1, calls them members."""
    h = _hexatope(**HEX_CUT_SET)
    assert M.distance(h, point) == pytest.approx(expected, abs=NEAR)
    assert R.member(R.star(h), point) == (expected == 0)


@pytest.mark.parametrize("point, expected", [
    ([0, 0], 0.0), ([1, -0.5], 0.0), ([-1, 1], 0.0), ([0.25, 0.25], 0.0),
    ([-1, -1], 0.0), ([1, -1], 0.0),                              # members
    ([1, 1], 0.75), ([0.5, 0.5], 0.25), ([1, 0], 0.25),           # cut corner
    ([2, -1], 1.0), ([-1, 1.25], 0.25),                           # outside box
])
def test_a_sum_row_cuts_the_octatope(point, expected):
    """The row x0 + x1 <= 1/2 binds: the corner (1, 1) is in the bounding box
    and not in the set. A reader that drops it, or flips a sign, misses it."""
    o = _octatope(**OCT_CUT_SET)
    assert M.distance(o, point) == pytest.approx(expected, abs=NEAR)
    assert R.member(R.star(o), point) == (expected == 0)


def test_the_anchor_of_a_hexatope_is_pinned_to_zero():
    """The anchor stands for 0, whatever its generator column holds. A
    reader that leaves it free (or bounds it to [-1, 1], as ``to_star``
    does) takes points that need x0 != 0 as members."""
    # y = x0 + x1 with -1 <= x1 - x0 <= 1: with x0 = 0 the segment [-1, 1];
    # with x0 in [-1, 1] the segment [-3, 3]; with x0 free the whole line.
    h = _hexatope([0], [[1, 1]], [(1, 0, 1), (0, 1, 1)])
    for y, expected in [([0], 0), ([1], 0), ([-1], 0), ([2], 1), ([-3], 2),
                        ([100], 99)]:
        assert M.distance(h, y) == pytest.approx(expected, abs=NEAR), y
    unpinned = _star([0], [[1, 1]], C=[[-1, 1], [1, -1]], d=[1, 1],
                     lb=[-1, -1], ub=[1, 1])          # the to_star reading
    assert M.distance(unpinned, [2]) == pytest.approx(0, abs=NEAR)
    # After a split, x1 - x0 <= -1/2: pinned, x1 in [-1, -1/2].
    split = _hexatope([0], [[0, 1]], [(1, 0, -0.5), (0, 1, 1)])
    assert M.distance(split, [-0.75]) == pytest.approx(0, abs=NEAR)
    assert M.distance(split, [0]) == pytest.approx(0.5, abs=NEAR)
    assert not R.member(R.star(split), [0])
    assert R.member(R.star(split), [-0.5])


def test_an_unbounded_hexatope_and_octatope_are_read_as_unbounded():
    """No row bounds x1 from below: every point below 1 is a member. The
    distance LP has an open direction and must still answer."""
    h = _hexatope([0, 0], [[0, 1, 0], [0, 0, 1]], [(1, 0, 1), (2, 1, 0)])
    # x1 <= 1 and x2 <= x1: the wedge below the diagonal and left of x = 1.
    for y, expected in [([-1e6, -2e6], 0), ([1, 1], 0), ([1, -5], 0),
                        ([3, 0], 2), ([0, 1], 0.5), ([-1e6, 0], 5e5)]:
        assert M.distance(h, y) == pytest.approx(expected, abs=1e-6), y
    with pytest.raises(ValueError, match="unbounded"):
        R.set_vertices(R.star(h))
    nothing = _octatope([1, 2], [[1, 0], [0, 1]], [])
    assert M.distance(nothing, [1e9, -1e9]) == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError, match="unbounded"):
        R.set_vertices(R.star(nothing))


def test_an_infeasible_hexatope_or_octatope_is_empty():
    """A negative cycle of differences, and ``0 <= -1`` on one variable."""
    cycle = _hexatope([0], [[0, 1, 1]], [(1, 2, -1), (2, 1, -1), (1, 0, 5)])
    assert M.distance(cycle, [0]) == float("inf")
    assert M.uncovered([cycle], [[0]]) == [(0, float("inf"))]
    void = _octatope([0], [[1]], [(0, 0, 1, -1, -1), (0, 0, 1, 0, 1)])
    assert M.distance(void, [0]) == float("inf")
    assert not R.member(R.star(void), [0])


# A different interval per coordinate, one of zero width, all dyadic.
FROM_LO = np.array([-0.75, 0.25, -2.0, 0.375])
FROM_HI = np.array([0.5, 1.75, -1.25, 0.375])


@pytest.mark.parametrize("cls", [Hexatope, Octatope, Box])
def test_from_bounds_holds_its_box_and_no_more(cls):
    """Every corner is a member and the exact vertex list is the box's own
    corners; a point 2^-10 outside the middle of any face is not a member
    and is 2^-10 away. The degenerate coordinate is held at its value."""
    s = cls.from_bounds(FROM_LO, FROM_HI) if cls is not Box \
        else Box(FROM_LO, FROM_HI)
    exact = R.star(s)
    corners = sorted(set(product(*zip(FROM_LO.tolist(), FROM_HI.tolist()))))
    assert R.set_vertices(exact) == [tuple(R.vector(p)) for p in corners]
    for p in corners:
        assert M.distance(s, p) == pytest.approx(0, abs=NEAR)
        assert R.member(exact, p)
    step = 2.0 ** -10
    middle = (FROM_LO + FROM_HI) / 2
    for i in range(FROM_LO.size):
        for end, sign in ((FROM_LO[i], -1), (FROM_HI[i], 1)):
            y = middle.copy()
            y[i] = end + sign * step
            assert M.distance(s, y) == pytest.approx(step, abs=NEAR), (i, sign)
            assert not R.member(exact, y), (i, sign)
            y[i] = end
            assert M.distance(s, y) == pytest.approx(0, abs=NEAR), (i, sign)


def _random_hexatope(rng):
    """Lists for a hand-built hexatope: an anchor with a generator column of
    its own (which must not count), interval rows to the anchor on most
    variables, and a few random difference rows."""
    dim, p = int(rng.integers(1, 4)), int(rng.integers(3, 5))
    dyadic = lambda *shape: (rng.integers(-16, 17, shape) / 8).tolist()  # noqa: E731
    constraints = []
    for i in range(1, p):
        if rng.random() < 0.85:
            constraints.append((i, 0, float(rng.integers(0, 9) / 4)))
        if rng.random() < 0.85:
            constraints.append((0, i, float(rng.integers(0, 9) / 4)))
    for _ in range(int(rng.integers(1, 4))):
        i, j = rng.choice(p, size=2, replace=False)
        constraints.append((int(i), int(j), float(rng.integers(-6, 9) / 8)))
    return dict(c=dyadic(dim), G=dyadic(dim, p), constraints=constraints)


def _random_octatope(rng):
    dim, p = int(rng.integers(1, 4)), int(rng.integers(2, 4))
    dyadic = lambda *shape: (rng.integers(-16, 17, shape) / 8).tolist()  # noqa: E731
    constraints = []
    for i in range(p):
        if rng.random() < 0.85:
            constraints.append((i, i, 1, 0, float(rng.integers(0, 9) / 4)))
        if rng.random() < 0.85:
            constraints.append((i, i, -1, 0, float(rng.integers(0, 9) / 4)))
    for _ in range(int(rng.integers(1, 4))):
        i, j = (int(v) for v in rng.choice(p, size=2, replace=False))
        ai, aj = (int(v) for v in rng.choice([-1, 1], size=2))
        constraints.append((i, j, ai, aj, float(rng.integers(-6, 9) / 8)))
    return dict(c=dyadic(dim), G=dyadic(dim, p), constraints=constraints)


def _members_by_construction(rng, lists, count):
    """Images of predicate points on a dyadic grid that keep every row."""
    c, G = np.array(lists["c"]), np.array(lists["G"])
    found = []
    for _ in range(40 * count):
        if len(found) == count:
            break
        a = rng.integers(-16, 17, G.shape[1]) / 8
        a = np.where(np.array(lists["lb"]) == 0, 0.0, a)       # the anchor
        if all(np.dot(row, a) <= rhs
               for row, rhs in zip(lists["C"], lists["d"])):
            found.append((c + G @ a).tolist())
    return found


def _interval_rows_only(kind, spec):
    """The constraints of a random set without its random two-variable rows:
    the rows to the anchor (hexatope) or on one variable (octatope)."""
    keep = ((lambda con: 0 in con[:2]) if kind == "Hexatope"
            else (lambda con: con[0] == con[1]))
    return dict(spec, constraints=[con for con in spec["constraints"]
                                   if keep(con)])


@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_the_distance_to_a_hexatope_or_octatope_is_confirmed_in_rationals(
        kind):
    """The LP oracle against the exact one, on hand-built sets with rows of
    every sign, some unbounded and some empty, at points inside, on the
    boundary and outside. The exact set is built from the meaning written out
    in the test, not through ``membership.rows``."""
    build, lists_of, random_set = {
        "Hexatope": (_hexatope, _hexatope_lists, _random_hexatope),
        "Octatope": (_octatope, _octatope_lists, _random_octatope)}[kind]
    tol = F(1, 10 ** 7)
    seen = dict(empty=0, inside=0, outside=0, cut=0, unbounded=0)
    for seed in range(12):
        rng = np.random.default_rng(2000 + seed)
        spec = random_set(rng)
        lists = lists_of(**spec)
        s, exact = build(**spec), _exact_open(**lists)
        uncut = _exact_open(**lists_of(**_interval_rows_only(kind, spec)))
        dim = len(spec["c"])
        everywhere = [F(10 ** 6)] * dim
        if not R.meets_box(exact, [-v for v in everywhere], everywhere):
            seen["empty"] += 1
            assert M.distance(s, [0.0] * dim) == float("inf")
            continue
        try:
            R.set_vertices(exact)
        except ValueError:
            seen["unbounded"] += 1
        points = (rng.integers(-40, 41, (12, dim)) / 8).tolist()
        points += _members_by_construction(rng, lists, 12)
        for y in points:
            found = M.distance(s, y)
            y = R.vector(y)
            reach = R.frac(found) + tol
            assert R.meets_box(exact, [v - reach for v in y],
                               [v + reach for v in y]), (seed, y, found)
            short = R.frac(found) - tol
            if short >= 0:
                assert not R.meets_box(exact, [v - short for v in y],
                                       [v + short for v in y]), (seed, y, found)
            if R.member(exact, y):
                assert found <= M.LP_TOL, (seed, y, found)
                seen["inside"] += 1
            else:
                assert M.uncovered([s], [[float(v) for v in y]]) == [
                    (0, found)] or found <= M.LP_TOL * max(
                        1, max(abs(float(v)) for v in y)), (seed, y, found)
                seen["outside"] += 1
                seen["cut"] += R.member(uncut, y)
    assert seen["empty"] >= 1 and seen["inside"] >= 60 \
        and seen["outside"] >= 60 and seen["cut"] >= 5, seen


@pytest.mark.parametrize("build", [
    lambda: _hexatope(**HEX_CUT_SET), lambda: _octatope(**OCT_CUT_SET),
    lambda: Hexatope.from_bounds(FROM_LO, FROM_HI),
    lambda: Octatope.from_bounds(FROM_LO, FROM_HI)],
    ids=["hexatope", "octatope", "hexatope-from-bounds",
         "octatope-from-bounds"])
def test_a_point_of_the_wrong_size_is_an_error_for_both_types(build):
    s = build()
    dim = s.dim
    for point in ([0.5] * (dim - 1), [0.5] * (dim + 1), []):
        with pytest.raises(AssertionError, match="entries"):
            M.distance(s, point)
        with pytest.raises(ValueError):
            R.member(R.star(s), point)


# --------------------------------------------------------------------------- #
# A predicate box as narrow as the solver tolerance                             #
# --------------------------------------------------------------------------- #
# The star: predicate
# box [-w, w]^2 with w about 1e-7, generators about 1e6 and one cut row;
# neuron 0 crosses zero, neurons 1 and 2 are 3 + 1e6 a_j and stay positive.
# Its exact ReLU is two stars, the active and the inactive piece of neuron 0,
# written out here by hand (n2v's exact ReLU returns these same arrays). The
# reproducer's own numbers are what made the unscaled LP fail; the dyadic copy
# of the same sizes has float points that are exact members.
NARROW = {
    "reproducer": dict(
        V=[[0.0, 0.26 * 1e6, -0.3 * 1e6], [3.0, 1e6, 0.0], [3.0, 0.0, 1e6]],
        cut=[0.91, 0.33], rhs=-3.844e-8, w=1e-7),
    "dyadic": dict(
        V=[[0.0, 17 / 64 * 2.0 ** 20, -19 / 64 * 2.0 ** 20],
           [3.0, 2.0 ** 20, 0.0], [3.0, 0.0, 2.0 ** 20]],
        cut=[29 / 32, 21 / 64], rhs=-5 / 16 * 2.0 ** -23, w=2.0 ** -23),
}


def _narrow_pieces(V, cut, rhs, w):
    box = dict(lb=[-w, -w], ub=[w, w])
    active = dict(c=[row[0] for row in V], G=[row[1:] for row in V],
                  C=[cut, [-V[0][1], -V[0][2]]], d=[rhs, 0.0], **box)
    inactive = dict(active, c=[0.0, 3.0, 3.0],
                    G=[[0.0, 0.0]] + [row[1:] for row in V[1:]],
                    C=[cut, [V[0][1], V[0][2]]])
    return [active, inactive]


def _narrow_output(V, a):
    """The exact ReLU output at the rational predicate point ``a``, rounded to
    float64 once at the end."""
    x = [F(row[0]) + F(row[1]) * a[0] + F(row[2]) * a[1] for row in V]
    return [float(max(F(0), v)) for v in x]


@pytest.mark.parametrize("name", sorted(NARROW))
def test_a_narrow_predicate_box_with_large_generators_is_read_exactly(name):
    """At this scale HiGHS's default tolerance, 1e-7 on the raw predicate
    variables, is half the box, and the unscaled LP called most members of
    the ReLU result non-members (210 of 300 in the reproducer, at distances up
    to 0.13). Every point within 2^-40 of the set (decided exactly; for the
    dyadic data such points are exact members) must be reported a member, and
    every point whose exact distance is above the tolerance must be reported
    outside: predicate points pushed out of the box, points a hair beyond the
    cut row, and points off the kink of neuron 0."""
    V, cut, rhs, w = (NARROW[name][k] for k in ("V", "cut", "rhs", "w"))
    pieces = _narrow_pieces(V, cut, rhs, w)
    stars, exact = [_star(**p) for p in pieces], [_exact(**p) for p in pieces]
    row, bound, half = R.vector(cut), R.frac(rhs), R.frac(w)
    rng = np.random.default_rng(5)

    def predicate_point(outside=0):
        if name == "dyadic":                        # images exact in float64
            a = [half * F(int(k), 1024) for k in rng.integers(-1024, 1025, 2)]
        else:
            a = [R.frac(v) for v in rng.uniform(-w, w, 2)]
        j = int(rng.integers(0, 2))
        if outside:                                 # off the box
            a[j] = half * F(outside) * (1 if a[j] >= 0 else -1)
        elif rng.random() < 0.25:                   # on a face of the box
            a[j] = half * (1 if a[j] >= 0 else -1)
        return a

    members = []
    while len(members) < 150:
        a = predicate_point()
        if R.dot(row, a) <= bound:
            members.append(_narrow_output(V, a))
    while len(members) < 200:                       # on the cut face
        a1 = R.frac(rng.uniform(-w, w))
        a0 = (bound - row[1] * a1) / row[0]
        if abs(a0) <= half:
            members.append(_narrow_output(V, [a0, a1]))
    pad = F(1, 2 ** 40)
    for y in members:
        if not any(R.meets_box(e, [R.frac(v) - pad for v in y],
                               [R.frac(v) + pad for v in y]) for e in exact):
            pytest.fail(f"a constructed member is farther than 2^-40: {y}")
    if name == "dyadic" and not all(any(R.member(e, y) for e in exact)
                                    for y in members[:150:5]):
        pytest.fail("a dyadic member is not an exact member")
    assert M.uncovered(stars, members) == []

    outside = []
    while len(outside) < 60:                        # off the predicate box
        a = predicate_point(outside=rng.choice([F(101, 100), F(5, 4), 2]))
        if R.dot(row, a) <= bound:
            outside.append(_narrow_output(V, a))
    while len(outside) < 120:                       # a hair beyond the cut
        a1 = R.frac(rng.uniform(-w, w))
        a0 = (bound + F(1, 10 ** 11) * int(rng.integers(1, 4))
              - row[1] * a1) / row[0]
        if abs(a0) <= half:
            outside.append(_narrow_output(V, [a0, a1]))
    for y in members[:60]:                          # off the kink of neuron 0
        outside.append([y[0] + 2.0 ** -18, y[1], y[2]])
    far = []
    for y in outside:
        ball = R.frac(M.LP_TOL * max(1.0, max(abs(v) for v in y)))
        lo, hi = [R.frac(v) - ball for v in y], [R.frac(v) + ball for v in y]
        if not any(R.meets_box(e, lo, hi) for e in exact):
            far.append(y)
    if len(far) < 100:
        pytest.fail(f"only {len(far)} points are farther than the tolerance")
    assert [i for i, _ in M.uncovered(stars, far)] == list(range(len(far)))


class _Point:
    """What ``linprog`` returns: a status, an objective and a point."""

    def __init__(self, x, fun=0.0, status=0):
        self.x, self.fun, self.status = np.array(x, dtype=float), fun, status
        self.message = "planted"


def test_the_answer_is_the_distance_at_the_solver_point(monkeypatch):
    """The returned distance is recomputed at the solver's point, never taken
    from its objective. TRIANGLE has a in [-1, 1]^2, scaled to s in [0, 1]^2:
    s = (1/2, 1/2) is the predicate point (0, 0), the set's point (0, 0)."""
    s = _star(**TRIANGLE)
    monkeypatch.setattr(M, "linprog",
                        lambda *a, **k: _Point([0.5, 0.5, 0.0], fun=0.0))
    # A solver that claims distance 0 for a point 5 away: the answer is 5.
    assert M.distance(s, [5, 5]) == 5
    assert M.uncovered([s], [[5, 5]]) == [(0, 5.0)]
    # A suboptimal solver point for a member: too large, never too small.
    assert M.distance(s, [-0.5, -0.75]) == 0.75


@pytest.mark.parametrize("x, row", [
    ([1.0, 1.0, 0.0], 0),                 # a = (1, 1): a1 + a2 <= 0 broken by 2
    ([0.5 + 1e-6, 0.5, 0.0], 0),          # broken by 2e-6
])
def test_a_solver_point_that_breaks_a_row_is_an_error(monkeypatch, x, row):
    """A point outside the rows is not a point of the set: its distance is
    no answer, so the oracle raises rather than report it."""
    monkeypatch.setattr(M, "linprog", lambda *a, **k: _Point(x))
    with pytest.raises(RuntimeError, match=f"breaks row {row}"):
        M.distance(_star(**TRIANGLE), [0.0, 0.0])


def test_a_solver_point_outside_the_bounds_is_clipped(monkeypatch):
    """Bounds are made exact by clipping; the rows are then checked at the
    clipped point. s = (-1e-12, 1/2) is a1 = -1 - 2e-12, clipped to -1."""
    monkeypatch.setattr(M, "linprog",
                        lambda *a, **k: _Point([-1e-12, 0.5, 0.0]))
    assert M.distance(_star(**TRIANGLE), [-1.0, 0.0]) == 0.0


def test_the_bounds_hold_exactly_where_float64_overshoots_them(monkeypatch):
    """With lb = -0.3 and ub = 0.1, ``lb + (ub - lb)`` is one unit in the
    last place above ``ub``: the predicate point at the top of the box is
    clipped back to ``ub``, so its image is the corner and not one ulp past."""
    lb, ub = -0.3, 0.1
    if not lb + (ub - lb) > ub:
        pytest.fail("the bounds no longer overshoot in float64")
    s = _star([0.0], [[1.0]], lb=[lb], ub=[ub])
    monkeypatch.setattr(M, "linprog", lambda *a, **k: _Point([1.0, 0.0]))
    assert M.distance(s, [ub]) == 0.0
    assert M.distance(s, [lb + (ub - lb)]) == (lb + (ub - lb)) - ub


def test_a_set_or_point_with_nan_is_an_error_not_an_answer():
    s = _star(**DIAMOND)
    with pytest.raises(ValueError, match="NaN"):
        M.distance(s, [np.nan, 0])
    with pytest.raises(ValueError, match="NaN"):
        M.distance(Box(np.zeros(2), np.ones(2)), [0.5, np.nan])
    broken = _star([0, 0], [[1, 0], [0, 1]], C=[[np.nan, 1]], d=[0],
                   lb=[-1, -1], ub=[1, 1])
    with pytest.raises(ValueError, match="NaN"):
        M.distance(broken, [0, 0])
    no_rhs = _star([0, 0], [[1, 0], [0, 1]], C=[[1, 1]], d=[np.nan],
                   lb=[-1, -1], ub=[1, 1])
    with pytest.raises(ValueError, match="NaN"):
        M.distance(no_rhs, [0, 0])


def test_an_infinite_right_hand_side_is_no_limit_and_minus_infinity_empties():
    open_row = _star([0, 0], [[1, 0], [0, 1]], C=[[1, 1]], d=[np.inf],
                     lb=[-1, -1], ub=[1, 1])
    assert M.distance(open_row, [1, 1]) == 0
    closed = _star([0, 0], [[1, 0], [0, 1]], C=[[1, 1]], d=[-np.inf],
                   lb=[-1, -1], ub=[1, 1])
    assert M.distance(closed, [0, 0]) == float("inf")


def test_a_pinned_predicate_variable_is_held_at_its_value():
    """lb == ub: the variable is its value, wherever its column points."""
    s = _star([0, 0], [[1, 4], [0, 1]], C=[[1, 1]], d=[1.25],
              lb=[-1, 0.5], ub=[1, 0.5])
    # x = (a1 + 2, 0.5) with a1 in [-1, 0.75].
    for y, expected in [([1, 0.5], 0), ([2.75, 0.5], 0), ([3, 0.5], 0.25),
                        ([2, 0.25], 0.25), ([0, 0.5], 1)]:
        assert M.distance(s, y) == pytest.approx(expected, abs=NEAR), y
