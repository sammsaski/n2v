"""Self-test of the exact rational oracle (``tests/_oracles/rational.py``).

An oracle that accepts everything proves nothing. Each group below checks the
oracle against something worked out another way (known polytopes, a direct
forward pass in rationals, brute-force enumeration, HiGHS), and then plants a
bug in a correct enclosure and requires the oracle to report it: a bound moved
in by one unit in the last place, a flipped sign, a missing piece, a region
that is too large.
"""

import math
import random
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
from scipy.optimize import linprog

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.sets.hexatope import DifferenceConstraintSystem
from n2v.sets.octatope import UTVPIConstraintSystem
from tests._oracles import rational as R

ULP = F(1, 2 ** 53)                     # the gap below 1.0 in float64
SQUARE = ([-1, -1], [1, 1])


# --------------------------------------------------------------------------- #
# Exact linear algebra                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("A, b, expected", [
    pytest.param([[1, 0], [0, 1], [-1, 0], [0, -1]], [1, 1, 0, 0],
                 [(0, 0), (0, 1), (1, 0), (1, 1)], id="unit-square"),
    pytest.param([[-1, 0], [0, -1], [1, 1]], [0, 0, 1],
                 [(0, 0), (0, 1), (1, 0)], id="triangle"),
    pytest.param([[-1, 0], [0, -1], [1, 1], [1, 1], [1, 0]], [0, 0, 1, 2, 5],
                 [(0, 0), (0, 1), (1, 0)], id="triangle-with-redundant-rows"),
    pytest.param([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 1, 1]], [0, 0, 0, 1],
                 [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)], id="simplex"),
    pytest.param([[1, 0], [-1, 0], [0, 1], [0, -1]], [1, -2, 1, 0],
                 [], id="empty"),
    pytest.param([[1, 0], [-1, 0], [0, 1], [0, -1]], [1, -1, 1, 0],
                 [(1, 0), (1, 1)], id="segment"),
])
def test_vertices_of_known_polytopes(A, b, expected):
    found = R.vertices(R.matrix(A), R.vector(b))
    assert found == [tuple(F(v) for v in point) for point in expected]


def test_solve_is_exact_where_floats_are_not():
    # 0.1 + 0.2 != 0.3 in floats; the rational system has the exact answer.
    x = R.solve(R.matrix([[3, 0], [1, 10]]), R.vector([1, 1]))
    assert x == [F(1, 3), F(1, 15)]
    assert R.solve(R.matrix([[1, 2], [2, 4]]), R.vector([1, 2])) is None


def test_boundedness_is_decided_from_the_rows():
    square = R.matrix([[1, 0], [-1, 0], [0, 1], [0, -1]])
    assert R.is_bounded(square)
    assert not R.is_bounded(square[:3])              # open towards y = -inf
    assert not R.is_bounded(R.matrix([[1, -1], [-1, 1]]))        # a strip
    with pytest.raises(ValueError, match="unbounded"):
        R.polytope([[1, 0], [-1, 0], [0, 1]], [1, 1, 1])


def _random_system(rng, n):
    """Integer rows on n variables inside the box [-2, 2]^n."""
    unit = [[int(i == j) for j in range(n)] for i in range(n)]
    A = unit + [[-v for v in row] for row in unit]
    b = [2] * (2 * n)
    for _ in range(rng.randint(1, 4)):
        A.append([rng.randint(-3, 3) for _ in range(n)])
        b.append(rng.randint(-4, 4))
    return A, b


@pytest.mark.parametrize("n", [1, 2, 3])
def test_feasibility_agrees_with_vertex_enumeration(n):
    rng = random.Random(n)
    seen = set()
    for _ in range(150):
        A, b = _random_system(rng, n)
        point = R.feasible((), (), R.matrix(A), R.vector(b))
        corners = R.vertices(R.matrix(A), R.vector(b))
        assert (point is not None) == bool(corners), (A, b)
        seen.add(point is not None)
    assert seen == {True, False}        # the comparison saw both answers


def test_feasibility_with_equalities_agrees_with_highs():
    rng = random.Random(7)
    seen = set()
    for _ in range(150):
        A, b = _random_system(rng, 3)
        E = [[rng.randint(-2, 2) for _ in range(3)]]
        e = [rng.randint(-2, 2)]
        exact = R.feasible(R.matrix(E), R.vector(e), R.matrix(A), R.vector(b))
        res = linprog(np.zeros(3), A_ub=np.array(A, float), b_ub=np.array(b, float),
                      A_eq=np.array(E, float), b_eq=np.array(e, float),
                      bounds=[(None, None)] * 3, method="highs")
        assert res.status in (0, 2)
        assert (exact is not None) == (res.status == 0), (E, e, A, b)
        seen.add(exact is not None)
    assert seen == {True, False}


def test_feasibility_sees_a_gap_no_float_solver_can():
    rows = R.matrix([[1], [-1]])
    assert R.feasible((), (), rows, [F(1), F(-1)]) == [F(1)]
    assert R.feasible((), (), rows, [F(1) - F(1, 2 ** 200), F(-1)]) is None


@pytest.mark.parametrize("n", [1, 2, 3])
def test_no_point_comes_with_multipliers_that_prove_it(n):
    """Each answer of ``decide`` is checked here by hand, not by the oracle's
    own check: a point against the rows, multipliers against the three
    conditions of a proof."""
    rng = random.Random(40 + n)
    proofs = 0
    for _ in range(150):
        A, b = _random_system(rng, n)
        E = [[rng.randint(-2, 2) for _ in range(n)]
             for _ in range(rng.randint(0, 1))]
        e = [rng.randint(-2, 2) for _ in E]
        A, b, E, e = R.matrix(A), R.vector(b), R.matrix(E), R.vector(e)
        point, proof = R.decide(E, e, A, b)
        assert (point is None) != (proof is None)
        if proof is None:
            assert all(R.dot(row, point) == r for row, r in zip(E, e))
            assert all(R.dot(row, point) <= r for row, r in zip(A, b))
            continue
        u, v = proof
        assert len(u) == len(E) and len(v) == len(A)
        assert all(m >= 0 for m in v)
        for j in range(n):
            assert (R.dot(u, [row[j] for row in E])
                    + R.dot(v, [row[j] for row in A])) == 0
        assert R.dot(u, e) + R.dot(v, b) < 0
        proofs += 1
    assert 15 < proofs < 135            # the comparison saw both answers


def test_multipliers_that_prove_nothing_are_rejected():
    # x <= 1 and -x <= -2 have no point: the two rows add up to 0 <= -1.
    A, b = R.matrix([[1], [-1]]), R.vector([1, -2])
    assert R.is_refutation(([], [F(1), F(1)]), (), (), A, b)
    assert not R.is_refutation(([], [F(1), F(2)]), (), (), A, b)   # x is left
    assert not R.is_refutation(([], [F(0), F(0)]), (), (), A, b)   # 0 <= 0
    # A proof names every row. Cut short, this one would still add up.
    C, d = R.matrix([[0], [1]]), R.vector([-1, 5])
    assert R.is_refutation(([], [F(1), F(0)]), (), (), C, d)
    assert not R.is_refutation(([], [F(1)]), (), (), C, d)
    # x <= 1 and -x <= -1 have the point 1, and the rows add up to 0 <= 0.
    assert not R.is_refutation(([], [F(1), F(1)]), (), (), A, R.vector([1, -1]))
    # x <= 1 and x <= 3 have points. A negative multiplier turns the second
    # row around and "proves" 0 <= -2.
    assert not R.is_refutation(([], [F(1), F(-1)]), (), (),
                               R.matrix([[1], [1]]), R.vector([1, 3]))
    # The multiplier of an equality may have either sign: x = 3 with x <= 1.
    E, e = R.matrix([[1]]), R.vector([3])
    assert R.is_refutation(([F(-1)], [F(1), F(0)]), E, e, A, b)
    assert not R.is_refutation(([F(1)], [F(1), F(0)]), E, e, A, b)


def test_rows_on_no_variable_are_decided_by_their_right_hand_sides():
    """A set with no predicate variable is one point, and asking whether
    another point is in it gives rows with no columns."""
    assert R.decide([[]], [F(0)], [[]], [F(0)]) == ([], None)
    for rows in [([[]], [F(1)], [], []), ([[]], [F(-1)], [], []),
                 ([], [], [[]], [F(-1)]), ([[]], [F(0)], [[], []], [F(2), F(-1)])]:
        point, proof = R.decide(*rows)
        assert point is None and R.is_refutation(proof, *rows)
    point_set = ([F(1), F(2)], [[], []], [], [], [], [])
    for ask in (R.member, R.Membership(point_set)):
        assert ask(point_set, [1, 2]) if ask is R.member else ask([1, 2])
        assert not (ask(point_set, [1, 3]) if ask is R.member else ask([1, 3]))


# --------------------------------------------------------------------------- #
# Pieces against a direct forward pass                                          #
# --------------------------------------------------------------------------- #
W1 = [[1, 1], [1, -1], [F(-1, 2), 1]]
C1 = [0, F(1, 2), F(1, 4)]
W2 = [[1, -1, F(1, 2)], [F(1, 2), 1, -1]]
C2 = [F(1, 4), 0]


def _affine(W, c, x):
    return [R.dot(R.vector(row), x) + F(ci) for row, ci in zip(W, c)]


def _sign(t):
    return (t > 0) - (t < 0)


# name -> (oracle operation, the same function on one rational, hidden scale).
# The hidden layer is scaled so that the step functions meet several jumps.
ACTIVATIONS = {
    "relu": (R.relu, lambda t: max(t, F(0)), 1),
    "leaky_relu": (lambda p: R.leaky_relu(p, F(1, 8)),
                   lambda t: t if t >= 0 else t / 8, 1),
    "sign": (R.sign, _sign, 1),
    "round": (R.round_, round, 2),      # Python rounds a half to even, exactly
    "floor": (R.floor, math.floor, 2),
    "ceil": (R.ceil, math.ceil, 2),
}


def _network(name):
    """(pieces of the two-layer network on the square, its direct forward)."""
    operation, scalar, scale = ACTIVATIONS[name]
    W = [[F(w) * scale for w in row] for row in W1]
    c = [F(v) * scale for v in C1]
    pieces = R.affine(operation(R.affine(R.box(*SQUARE), W, c)), W2, C2)

    def forward(x):
        hidden = [F(scalar(t)) for t in _affine(W, c, x)]
        return tuple(_affine(W2, C2, hidden))
    return pieces, forward


def _grid(step):
    """The points of the square on a grid: on it, pre-activations are multiples
    of ``step / 2``, so the grid lands on kinks and jumps as well as off them."""
    ticks = [F(-1) + step * i for i in range(int(2 / step) + 1)]
    return [list(x) for x in product(ticks, repeat=2)]


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_pieces_evaluate_to_the_direct_forward_pass(name):
    pieces, forward = _network(name)
    for x in _grid(F(1, 8)):
        assert R.evaluate(pieces, x) == forward(x), x
    for x in _grid(F(1, 2)):
        assert R.image_contains(pieces, forward(x)), x


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_every_vertex_of_the_image_is_a_limit_of_values(name):
    """No piece reports a value the function does not approach: each vertex of
    each piece is the limit of the forward pass along the segment from a
    strictly interior point of that piece's region."""
    pieces, forward = _network(name)
    for piece in pieces:
        corners = piece.corners()
        centre = [sum(col) / len(corners) for col in zip(*corners)]
        assert piece.holds(centre)
        for corner in corners:
            near = [c + (v - c) * (1 - F(1, 2 ** 40))
                    for c, v in zip(centre, corner)]
            gap = max(abs(a - b) for a, b in
                      zip(forward(near), piece.value(corner)))
            assert gap <= F(1, 2 ** 30), (corner, gap)


def test_relu_pieces_are_the_feasible_activation_patterns():
    pieces, _ = _network("relu")
    patterns = set()
    for x in _grid(F(1, 16)):
        z = _affine(W1, C1, x)
        if all(t != 0 for t in z):
            patterns.add(tuple(t > 0 for t in z))
    full = [p for p in pieces if p.is_full_dimensional()]
    assert len(full) == len(patterns) == 6
    assert len(pieces) == len(full)     # no kink touches the square's boundary


def test_a_region_that_only_touches_a_kink_is_not_a_pattern():
    """x0 + x1 + 2 is zero at one corner of the square and positive elsewhere.
    The inactive region is that corner: a piece (its value belongs to the
    image) and not a feasible activation pattern."""
    pieces = R.relu(R.affine(R.box(*SQUARE), [[1, 1]], [2]))
    assert sorted(p.is_full_dimensional() for p in pieces) == [False, True]
    corner = next(p for p in pieces if not p.is_full_dimensional())
    assert corner.corners() == [(F(-1), F(-1))]
    assert [p.is_full_dimensional() for p in R.sign(R.box([-1], [1]))] == [
        True, False, True]


@pytest.mark.parametrize("operation, lo, hi, values", [
    pytest.param(R.floor, 0, 1, [0, 1], id="floor-ends-on-a-jump"),
    pytest.param(R.ceil, 0, 1, [0, 1], id="ceil-starts-on-a-jump"),
    pytest.param(R.ceil, -1, 0, [-1, 0], id="ceil-ends-on-a-jump"),
    pytest.param(R.sign, 0, 1, [0, 1], id="sign-starts-at-zero"),
    pytest.param(R.sign, -1, 0, [-1, 0], id="sign-ends-at-zero"),
    pytest.param(R.sign, -1, 1, [-1, 0, 1], id="sign-across-zero"),
    pytest.param(R.round_, 0, F(1, 2), [0], id="round-half-goes-to-even-0"),
    pytest.param(R.round_, 1, F(3, 2), [1, 2], id="round-half-goes-to-even-2"),
    pytest.param(R.round_, F(1, 2), F(3, 2), [0, 1, 2], id="round-between-halves"),
])
def test_a_range_that_ends_on_a_jump_gains_no_value(operation, lo, hi, values):
    pieces = operation(R.box([lo], [hi]))
    assert sorted({y[0] for p in pieces for y in p.image()}) == values


def test_maxpool_is_the_largest_of_each_group():
    pieces = R.maxpool(R.affine(R.box(*SQUARE), W1, C1), [[0, 1], [1, 2]])
    assert len(pieces) == 4
    for x in _grid(F(1, 4)):
        z = _affine(W1, C1, x)
        assert R.evaluate(pieces, x) == (max(z[0], z[1]), max(z[1], z[2]))


def test_split_keeps_the_image():
    pieces, forward = _network("relu")
    halves = R.split(pieces, [1, 0], F(1, 8))
    assert len(halves) > len(pieces)
    for x in _grid(F(1, 4)):
        assert R.evaluate(halves, x) == forward(x)


# --------------------------------------------------------------------------- #
# Membership                                                                    #
# --------------------------------------------------------------------------- #
def test_membership_has_no_slack():
    unit = R.interval([0, 0], [1, 1])
    assert R.member(unit, [1, F(1, 2)])
    assert not R.member(unit, [1 + F(1, 2 ** 200), F(1, 2)])
    triangle = R.halfspaces([[-1, 0], [0, -1], [1, 1]], [0, 0, 1])
    assert R.member(triangle, [F(1, 2), F(1, 2)])
    assert not R.member(triangle, [F(1, 2), F(1, 2) + F(1, 2 ** 200)])


def test_membership_in_a_zonotope_with_more_generators_than_dimensions():
    # c = 0, generators (1,0), (0,1), (1,1): a hexagon.
    zono = R.star(Zono(np.zeros((2, 1)),
                       np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])))
    hexagon = [(2, 2), (2, 0), (0, 2), (-2, -2), (-2, 0), (0, -2)]
    assert R.set_vertices(zono) == sorted(
        {(F(a + c), F(b + c)) for a, b, c in product([-1, 1], repeat=3)})
    for x, y in hexagon:
        assert R.member(zono, [x, y])
        assert not R.member(zono, [x * (1 + ULP), y * (1 + ULP)])
    assert not R.member(zono, [2, -1])       # in the bounding box, not the set
    assert R.member(zono, [1, -1])


def test_a_computed_set_is_read_as_the_floats_it_stores():
    """A box star stores a centre and a half-width per axis. Its lower end is
    the rational difference of those two floats, which is neither the float
    0.1 the box was built from nor one tenth."""
    built = Star.from_bounds(np.array([[0.1], [0.0]]),
                             np.array([[1.0], [1.0]]))
    centre = [F(float(v)) for v in built.V[:, 0]]
    half = [F(float(built.V[i, i + 1])) for i in range(2)]
    low, high = centre[0] - half[0], centre[0] + half[0]
    assert low not in (F(0.1), F(1, 10))
    star = R.star(built)
    assert R.set_vertices(star) == sorted(
        (x, y) for x in (low, high)
        for y in (centre[1] - half[1], centre[1] + half[1]))
    assert R.member(star, [low, F(1, 2)])
    assert not R.member(star, [low - F(1, 2 ** 200), F(1, 2)])


def test_a_star_without_predicate_bounds_is_bounded_by_its_rows_alone():
    """A star given by rows only stores no predicate bounds. Reading an absent
    bound as a number would shrink the set."""
    built = Star(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                 np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]),
                 np.array([[0.0], [0.0], [1.0]]))
    assert built.predicate_lb is None and built.predicate_ub is None
    triangle = R.star(built)
    assert triangle[4] == [None, None] and triangle[5] == [None, None]
    assert R.set_vertices(triangle) == [(F(0), F(0)), (F(0), F(1)), (F(1), F(0))]
    assert R.member(triangle, [F(1, 2), F(1, 2)])
    assert not R.member(triangle, [F(1, 2), F(1, 2) + ULP])


def test_membership_agrees_with_the_float_oracle_away_from_the_boundary():
    from tests._oracles.membership import distance
    star = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1))).affine_map(
        np.array([[1.0, 0.5], [-0.25, 1.0]]), np.array([[0.5], [0.0]]))
    exact = R.star(star)
    inside = 0
    for x in _grid(F(1, 4)):
        y = [2 * x[0] + F(1, 16), 2 * x[1] + F(1, 16)]
        is_member = R.member(exact, y)
        assert is_member == (distance(star, [float(v) for v in y]) < 1e-9), y
        inside += is_member
    assert 0 < inside < 81


def test_meets_box_is_exact():
    triangle = R.halfspaces([[-1, 0], [0, -1], [1, 1]], [0, 0, 1])
    assert R.meets_box(triangle, [F(1, 2), F(1, 2)], [1, 1])       # one point
    assert not R.meets_box(triangle, [F(1, 2) + ULP, F(1, 2)], [1, 1])


# --------------------------------------------------------------------------- #
# Membership of many points, with remembered certificates                       #
# --------------------------------------------------------------------------- #
def _sets_to_ask():
    hexagon = R.star(Zono(np.zeros((2, 1)),
                          np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])))
    slanted = R.star(
        Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1))).affine_map(
            np.array([[1.0, 0.5], [-0.25, 1.0]]), np.array([[0.5], [0.0]])))
    triangle = R.halfspaces([[-1, 0], [0, -1], [1, 1]], [0, 0, 1])
    return {"hexagon": hexagon, "slanted": slanted, "triangle": triangle,
            "relaxation": tuple(_triangle_relaxation())}


SETS_TO_ASK = ["hexagon", "slanted", "triangle", "relaxation"]


def _hull(points):
    """The corners of the convex hull of points of the plane, by the monotone
    chain: ``set_vertices`` also returns points inside the set."""
    points = sorted(set(map(tuple, points)))

    def chain(sequence):
        out = []
        for p in sequence:
            while len(out) >= 2 and (
                    (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                    - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return chain(points) + chain(points[::-1])


def test_the_hull_helper_drops_inner_and_collinear_points():
    hexagon = _sets_to_ask()["hexagon"]
    assert sorted(_hull(R.set_vertices(hexagon))) == sorted(
        [(2, 2), (2, 0), (0, 2), (-2, -2), (-2, 0), (0, -2)])
    assert sorted(_hull([(0, 0), (1, 0), (2, 0), (0, 2), (1, 1)])) == [
        (0, 0), (0, 2), (2, 0)]


@pytest.mark.parametrize("keep", [16, 1])
@pytest.mark.parametrize("name", SETS_TO_ASK)
def test_remembered_certificates_give_the_answers_of_member(
        name, keep, monkeypatch):
    computed = _sets_to_ask()[name]
    corners = R.set_vertices(computed)
    low = [min(v[k] for v in corners) for k in range(2)]
    high = [max(v[k] for v in corners) for k in range(2)]
    rng = random.Random(5)
    points = [[lo + (hi - lo) * F(rng.randint(-6, 38), 32)
               for lo, hi in zip(low, high)] for _ in range(300)]

    solved = []
    decide = R.decide
    monkeypatch.setattr(
        R, "decide", lambda *rows: solved.append(1) or decide(*rows))
    holder = R.Membership(computed, keep=keep)
    answers = [holder(y) for y in points]
    by_simplex = len(solved)
    assert answers == [R.member(computed, y) for y in points]
    assert 60 < sum(answers) < 240          # the comparison saw both answers
    if keep == 16:
        assert by_simplex < 60, "the certificates were hardly ever reused"


def test_a_remembered_member_is_found_again_without_the_simplex(monkeypatch):
    """The rows kept for a member must pin its predicate point down. Here the
    first row that holds with equality repeats the generator row and pins
    nothing, so it cannot stand in for a row that does."""
    line = ([F(0)], [[F(1), F(1)]], [[F(1), F(1)]], [F(3, 2)],
            [F(-1), F(-1)], [F(1), F(1)])            # y = a1 + a2 <= 3/2
    solved = []
    decide = R.decide
    monkeypatch.setattr(
        R, "decide", lambda *rows: solved.append(1) or decide(*rows))
    holder = R.Membership(line)
    assert holder([F(3, 2)]) and len(solved) == 1
    assert holder([F(3, 2)]) and len(solved) == 1
    assert holder([F(5, 4)]) and len(solved) == 1    # the same rows, moved
    assert not holder([F(7, 4)]) and len(solved) == 2
    assert not holder([F(2)]) and len(solved) == 2   # the same proof


@pytest.mark.parametrize("name", SETS_TO_ASK)
def test_a_remembered_certificate_is_checked_for_each_point(name):
    """A point just across a face from a member is not a member, whatever was
    remembered for its neighbour, and the reverse."""
    computed = _sets_to_ask()[name]
    tiny = F(1, 2 ** 200)
    corners = _hull(R.set_vertices(computed))
    assert len(corners) >= 3
    centre = [sum(v[k] for v in corners) / len(corners) for k in range(2)]
    for corner in corners:
        outward = [c - m for c, m in zip(corner, centre)]
        inside = [c - tiny * o for c, o in zip(corner, outward)]
        outside = [c + tiny * o for c, o in zip(corner, outward)]
        holder = R.Membership(computed)
        for _ in range(2):              # the second round is all remembered
            assert holder(corner) and holder(inside)
            assert not holder(outside)
            assert holder(centre)
        assert not R.member(computed, outside)


def test_a_point_of_the_wrong_length_is_an_error_not_an_answer():
    unit = R.interval([0, 0], [1, 1])
    with pytest.raises(ValueError, match="2 entries, the point 3"):
        R.member(unit, [0, 0, 0])
    with pytest.raises(ValueError, match="2 entries, the point 3"):
        R.Membership(unit)([0, 0, 0])


# --------------------------------------------------------------------------- #
# Planted bugs: a relaxation of one ReLU layer                                  #
# --------------------------------------------------------------------------- #
# Two neurons with ranges [-1, 1] and [-1/2, 3/2] on the square, so both cross
# zero and both triangle slopes (1/2 and 3/4) are dyadic. The output is the
# hidden layer itself, so every row of the relaxation is a face of the set.
LAYER_W = [[F(1, 2), F(1, 2)], [F(1, 2), F(-1, 2)]]
LAYER_C = [F(0), F(1, 2)]
RANGES = [(F(-1), F(1)), (F(-1, 2), F(3, 2))]


def _triangle_relaxation():
    """The triangle relaxation of the layer as ``(c, G, C, d, lb, ub)`` on the
    predicate ``(x0, x1, a0, a1)``, written out here in rationals. Row 2k is
    ``z_k <= a_k`` and row 2k + 1 is ``a_k <= u (z_k - l) / (u - l)``."""
    C, d = [], []
    for k, (lo, hi) in enumerate(RANGES):
        slope = hi / (hi - lo)
        unit = [F(int(j == k)) for j in range(2)]
        C.append(LAYER_W[k] + [-v for v in unit])
        d.append(-LAYER_C[k])
        C.append([-slope * w for w in LAYER_W[k]] + unit)
        d.append(slope * (LAYER_C[k] - lo))
    G = [[F(0), F(0), F(1), F(0)], [F(0), F(0), F(0), F(1)]]
    lb = [F(-1), F(-1), F(0), F(0)]
    ub = [F(1), F(1)] + [hi for _, hi in RANGES]
    return [F(0), F(0)], G, C, d, lb, ub


def _layer_pieces():
    return R.relu(R.affine(R.box(*SQUARE), LAYER_W, LAYER_C))


def test_a_correct_relaxation_is_accepted():
    relaxation = _triangle_relaxation()
    assert R.escapes(_layer_pieces(), relaxation) == []
    assert R.straddlers(_layer_pieces(), relaxation) == []


def _upper_row_moved_in(s):
    s[3][1] -= ULP


def _lower_row_moved_in(s):
    s[3][0] -= ULP


def _input_bound_moved_in(s):
    s[5][0] -= ULP


def _activation_bound_moved_in(s):
    s[5][2] -= ULP


def _centre_moved(s):
    s[0][0] += ULP


def _generator_sign_flipped(s):
    s[1][0][2] = -s[1][0][2]


def _slope_from_half_the_upper_bound(s):
    lo, hi = RANGES[0][0], RANGES[0][1] / 2
    slope = hi / (hi - lo)
    s[2][1] = [-slope * w for w in LAYER_W[0]] + [F(1), F(0)]
    s[3][1] = slope * (LAYER_C[0] - lo)


def _lower_row_dropped_and_activation_capped(s):
    # The bug of a relaxation that keeps a_0 <= 1/2 (a stale bound).
    s[5][2] = F(1, 2)


@pytest.mark.parametrize("plant", [
    _upper_row_moved_in, _lower_row_moved_in, _input_bound_moved_in,
    _activation_bound_moved_in, _centre_moved, _generator_sign_flipped,
    _slope_from_half_the_upper_bound, _lower_row_dropped_and_activation_capped,
], ids=lambda f: f.__name__.strip("_").replace("_", "-"))
def test_a_planted_bug_in_the_relaxation_is_reported(plant):
    relaxation = _triangle_relaxation()
    plant(relaxation)
    escaped = R.escapes(_layer_pieces(), relaxation)
    assert escaped, "the oracle accepted an enclosure that misses the image"
    # The report is checkable: each escaped point is a value of the layer.
    pieces = _layer_pieces()
    for index, y in escaped:
        assert y in pieces[index].image()
        assert R.image_contains(pieces, y)


def test_a_looser_relaxation_is_still_accepted():
    """Moving a row outward is not a bug, and must not be reported as one."""
    relaxation = _triangle_relaxation()
    relaxation[3][1] += F(1, 8)
    relaxation[5][2] += F(1, 8)
    assert R.escapes(_layer_pieces(), relaxation) == []


# --------------------------------------------------------------------------- #
# Planted bugs: a method that claims the exact image                            #
# --------------------------------------------------------------------------- #
def _hidden_pieces():
    """The three-neuron ReLU layer: a surface of six flat parts in 3-D."""
    return R.relu(R.affine(R.box(*SQUARE), W1, C1))


def _as_set(piece, rows=None):
    """A piece written as the set an exact method would return for it."""
    keep = range(len(piece.A)) if rows is None else rows
    n = len(piece.A[0])
    return (list(piece.c), [list(row) for row in piece.W],
            [piece.A[i] for i in keep], [piece.b[i] for i in keep],
            [None] * n, [None] * n)


def _extra(pieces, computed):
    """Vertices of the computed sets that are outside the exact image."""
    return [y for s in computed
            for _, y in R.unreached(pieces, R.set_vertices(s))]


def test_the_exact_image_is_accepted_in_both_directions():
    pieces = _hidden_pieces()
    computed = [_as_set(p) for p in pieces]
    assert R.escapes(pieces, *computed) == []
    assert R.straddlers(pieces, *computed) == []
    assert _extra(pieces, computed) == []


@pytest.mark.parametrize("missing", range(6))
def test_a_missing_piece_is_reported(missing):
    pieces = _hidden_pieces()
    computed = [_as_set(p) for i, p in enumerate(pieces) if i != missing]
    assert R.escapes(pieces, *computed) or R.straddlers(pieces, *computed)


def test_a_piece_that_is_too_large_is_reported_as_not_exact():
    """Drop the row that keeps a neuron active on its piece: the set still
    contains the image, and now holds points the network never outputs."""
    pieces = _hidden_pieces()
    grown = None
    for index, piece in enumerate(pieces):
        for row in range(4, len(piece.A)):          # rows 0-3 are the square
            if any(piece.W[k] == [-a for a in piece.A[row]] for k in range(3)):
                grown = (index, row)
    assert grown is not None
    index, row = grown
    computed = [_as_set(p) for p in pieces]
    computed[index] = _as_set(
        pieces[index], [i for i in range(len(pieces[index].A)) if i != row])
    assert R.escapes(pieces, *computed) == []
    assert _extra(pieces, computed), "the oracle accepted an over-approximation"


def test_vertices_spread_over_two_sets_are_not_a_proof():
    """[0, 1] is not inside [0, 1/4] with [3/4, 1], though both its vertices
    are. The vertex test alone would pass; the piece is reported as spread
    over two sets, and cutting it shows the hole."""
    pieces = R.box([0], [1])
    ends = (R.interval([0], [F(1, 4)]), R.interval([F(3, 4)], [1]))
    assert R.escapes(pieces, *ends) == []
    assert R.straddlers(pieces, *ends) == [0]
    assert R.escapes(R.split(pieces, [1], F(1, 2)), *ends) == [
        (0, (F(1, 2),)), (1, (F(1, 2),))]
    halves = (R.interval([0], [F(1, 2)]), R.interval([F(1, 2)], [1]))
    cut = R.split(pieces, [1], F(1, 2))
    assert R.escapes(cut, *halves) == [] and R.straddlers(cut, *halves) == []


# --------------------------------------------------------------------------- #
# End to end on a set n2v computed                                              #
# --------------------------------------------------------------------------- #
def test_one_ulp_in_a_stored_array_is_reported():
    """An affine map with dyadic data is exact in float64, so the computed star
    passes with no slack. Moving one stored number by one unit in the last
    place is enough to fail."""
    W, c = [[1, F(1, 2)], [F(-1, 4), 1]], [F(1, 2), 0]
    pieces = R.affine(R.box(*SQUARE), W, c)
    star = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1))).affine_map(
        np.array(W, dtype=float), np.array(c, dtype=float).reshape(2, 1))
    assert R.escapes(pieces, R.star(star)) == []
    assert _extra(pieces, [R.star(star)]) == []

    star.V[0, 1] = np.nextafter(star.V[0, 1], 0.0)      # a generator shrinks
    assert R.escapes(pieces, R.star(star))


# --------------------------------------------------------------------------- #
# A box, a hexatope and an octatope                                             #
# --------------------------------------------------------------------------- #
def _cut_hexatope(b):
    """x in [-1, 1]^2 (rows to the anchor x0) with x1 - x2 <= b."""
    dcs = DifferenceConstraintSystem(3)
    for i in (1, 2):
        dcs.add_constraint(i, 0, 1.0)
        dcs.add_constraint(0, i, 1.0)
    dcs.add_constraint(1, 2, b)
    return Hexatope(np.zeros(2), np.array([[0.0, 1, 0], [0, 0, 1]]), dcs)


def _cut_octatope(b):
    """x in [-1, 1]^2 (rows on one variable) with x0 + x1 <= b."""
    utvpi = UTVPIConstraintSystem(2)
    for i in (0, 1):
        utvpi.add_constraint(i, i, 1, 0, 1.0)
        utvpi.add_constraint(i, i, -1, 0, 1.0)
    utvpi.add_constraint(0, 1, 1, 1, b)
    return Octatope(np.zeros(2), np.eye(2), utvpi)


def test_a_box_a_hexatope_and_an_octatope_are_read_as_the_sets_they_are():
    """``star`` reads them through ``membership.rows``: a box is the identity
    on its bounds, a hexatope pins its anchor (``lb == ub == 0``) and bounds
    nothing else, an octatope bounds nothing. The vertex lists are exact: the
    box's own corners, and the square with one corner cut off."""
    box = R.star(Box(np.array([-1.0, 0.5]), np.array([0.25, 2.0])))
    assert R.set_vertices(box) == [(F(-1), F(1, 2)), (F(-1), F(2)),
                                   (F(1, 4), F(1, 2)), (F(1, 4), F(2))]
    assert R.member(box, [F(1, 4), 2]) and not R.member(box, [F(1, 4) + ULP, 2])
    from_bounds = R.star(Hexatope.from_bounds(np.array([-1.0, 0.5]),
                                              np.array([0.25, 2.0])))
    assert from_bounds[4][0] == from_bounds[5][0] == 0
    assert from_bounds[4][1:] == [None, None] == from_bounds[5][1:]
    assert R.set_vertices(from_bounds) == R.set_vertices(box)
    corner_cut = sorted([(F(-1), F(-1)), (F(-1), F(1)), (F(1), F(1)),
                         (F(1), F(1, 2)), (F(-1, 2), F(-1))])
    hexagon = R.star(_cut_hexatope(0.5))
    assert R.set_vertices(hexagon) == corner_cut
    octagon = R.star(_cut_octatope(0.5))
    assert octagon[4] == [None, None] and octagon[5] == [None, None]
    assert R.set_vertices(octagon) == sorted(
        [(F(-1), F(-1)), (F(1), F(-1)), (F(-1), F(1)), (F(1), F(-1, 2)),
         (F(-1, 2), F(1))])


def test_the_exact_image_escapes_a_hexatope_or_octatope_that_cuts_it():
    """The planted bug is a set one corner too small: the box's image escapes
    it at that corner and nowhere else, to the last bit."""
    pieces = R.box(*SQUARE)
    assert R.escapes(pieces, R.star(_cut_hexatope(2.0))) == []
    assert R.escapes(pieces, R.star(_cut_hexatope(0.5))) == [
        (0, (F(1), F(-1)))]
    assert R.escapes(pieces, R.star(_cut_hexatope(2 - 2.0 ** -40))) == [
        (0, (F(1), F(-1)))]
    assert R.escapes(pieces, R.star(_cut_octatope(2.0))) == []
    assert R.escapes(pieces, R.star(_cut_octatope(0.5))) == [(0, (F(1), F(1)))]


def test_an_unbounded_or_point_set_has_its_own_vertex_answer():
    """No vertex list exists for an unbounded predicate polytope, including
    one with no row at all; a set on no predicate variable is its centre, or
    nothing when a row on no variable fails."""
    open_dcs = DifferenceConstraintSystem(2)
    open_dcs.add_constraint(1, 0, 1.0)                 # x1 <= 1 only
    with pytest.raises(ValueError, match="unbounded"):
        R.set_vertices(R.star(Hexatope(np.zeros(1), np.array([[0.0, 1]]),
                                       open_dcs)))
    with pytest.raises(ValueError, match="unbounded"):
        R.set_vertices(R.star(Octatope(np.zeros(2), np.eye(2),
                                       UTVPIConstraintSystem(2))))
    point = Star(np.array([[1.0], [2.0]]), np.zeros((1, 0)), np.array([[0.0]]))
    assert R.set_vertices(R.star(point)) == [(F(1), F(2))]
    empty = Star(np.array([[1.0], [2.0]]), np.zeros((1, 0)), np.array([[-1.0]]))
    assert R.set_vertices(R.star(empty)) == []
