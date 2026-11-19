"""Soundness cells for the operations of ``Star`` other than its range
queries, and for ``n2v/sets/predicate_box.py``.

Covered, each with the promise checked:

- ``fresh_predicate_tag``: every call returns a different float in [1, 2), so
  two stars re-encoded from bounds are never read as one predicate system by
  the graph reach (direct, and through a graph that joins Sin and Cos).
- ``Star.__init__``: the star stores exactly the set it is given (rows,
  predicate box with missing or infinite entries, float32 data).
- ``Star.fresh_from_bounds``: the box exactly, over one new unused variable
  (direct, and through a Softmax reach of a model).
- ``Star.affine_map``, ``Star.minkowski_sum``, ``Star.intersect_half_space``:
  the exact image, sum and cut, both directions (every point of the true set
  is in the result, and every vertex of the result is a point of the true
  set; a cut row may be padded outward by the documented rounding term).
- ``Star.convex_hull``: an outer hull, so every convex combination of member
  points of the two operands is in the result.
- ``Star.is_empty_set``: True exactly when the predicate polytope is empty
  (decided here in rationals), with the default and a named LP solver, and
  through ``verify_specification`` where a wrong True is a false UNSAT.
- ``Star.contains`` (both the LP and the algebraic path): True exactly for
  member points, one point or a batch.
- ``Star.to_image_star``: the flat order read as height, width, channel
  (direct, and through ReLU then pooling on a tall two-channel image).
- ``Star.sample``: member points only, also when the star stores wrong state
  bounds; ``Star.get_vertices`` and ``Star._fibonacci_sphere``: points of
  the star, and every vertex of a polygon or a box.
- ``ensure_finite_predicate_box``: a finite box that holds every feasible
  predicate point and the same set; empty reported only when empty; a
  refusal (``ValueError``) where no trustworthy box exists (direct, and
  through ``reach`` of a star without a predicate box, exact and approx).

Oracle. Every input set is written down here as the exact rationals
``{c + G a : C a <= d, lb <= a <= ub}`` of the same dyadic arrays the n2v
star is built from (:class:`Exact`), so the reference never comes from n2v.
Its vertices are the images of the vertices of its predicate polytope
(``tests/_oracles/rational.py``). A computed set is read from its stored
arrays (``rational.star``) and judged in exact arithmetic with no slack:
membership by the exact simplex (``rational.Membership``), and the vertices
of the computed set enumerated from its own rows. Reach results are judged
by the exact image of the network (``rational.relu`` of the input polytope)
and by 128 member points replayed in torch float64 or onnxruntime, decided
by the LP oracle on the result's rows (``assert_covered``).

Inputs. Coordinates well beyond plus and minus 1 with intervals wider than 2,
predicate boxes that are not [-1, 1] (one excludes 0), binding rows that are
not box rows (one excludes the lower predicate corner), a star without a
predicate box, one with a half-open box, a point star, a star with more
entries than predicate variables, and operands whose centres, rows,
right-hand sides and boxes differ, in both orders.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn.reach import reach_pytorch_model
from n2v.sets import HalfSpace, Star
from n2v.sets.predicate_box import ensure_finite_predicate_box
from n2v.sets.star import fresh_predicate_tag
from n2v.utils.lpsolver import LPFailure
from n2v.utils.model_loader import load_onnx
from n2v.utils.verify_specification import verify_specification
from tests._oracles import rational as Q
from tests._oracles.membership import distance
from tests.cell_support import (assert_covered, image_set, pixel_box,
                                reach_model, replay_torch, samples,
                                save_onnx)
from tests.known_bugs import known_bug

INF = float("inf")


# --------------------------------------------------------------------------- #
# The exact set                                                                 #
# --------------------------------------------------------------------------- #
def _frac_or_none(values, k):
    if values is None or math.isinf(values[k]):
        return None
    return Fraction(float(values[k]))


class Exact:
    """``{c + G a : C a <= d, lb <= a <= ub}`` in Fractions (``q``, the tuple
    ``rational.star`` returns; an absent bound is None) together with the
    float arrays it was written from (``arrays``), from which :meth:`star`
    builds the n2v star with the constructor under test."""

    def __init__(self, q, arrays=None):
        self.q = q
        self.arrays = arrays
        self.dim = len(q[0])
        self.nvar = len(q[4])
        self._vertices = None
        self._member = None

    @classmethod
    def of(cls, c, G, C=None, d=None, lb=None, ub=None):
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        G = np.asarray(G, dtype=np.float64).reshape(c.size, -1)
        n = G.shape[1]
        if C is None:
            C = np.zeros((0, n))
        C = np.asarray(C, dtype=np.float64)
        C = C.reshape(C.shape[0] if C.ndim == 2 else -1, n)
        d = (np.zeros(C.shape[0]) if d is None
             else np.asarray(d, dtype=np.float64).reshape(-1))
        lb = None if lb is None else np.asarray(lb, np.float64).reshape(-1)
        ub = None if ub is None else np.asarray(ub, np.float64).reshape(-1)
        live = ~np.isposinf(d)          # a row ``<= +inf`` always holds
        q = (Q.vector(c), [Q.vector(row) for row in G],
             [Q.vector(row) for row in C[live]], Q.vector(d[live]),
             [_frac_or_none(lb, k) for k in range(n)],
             [_frac_or_none(ub, k) for k in range(n)])
        return cls(q, (c, G, C, d, lb, ub))

    def star(self, dtype=np.float64):
        c, G, C, d, lb, ub = self.arrays
        cast = (lambda x: None if x is None else np.asarray(x, dtype=dtype))
        return Star(cast(np.hstack([c[:, None], G])), cast(C),
                    cast(d.reshape(-1, 1)), cast(lb), cast(ub))

    def vertices(self):
        if self._vertices is None:
            self._vertices = Q.set_vertices(self.q)
            if not self._vertices:
                pytest.fail("test data: the exact set is empty")
        return self._vertices

    def holds(self, y):
        if self._member is None:
            self._member = Q.Membership(self.q)
        return self._member(y)

    def midpoints(self, k=6):
        V = self.vertices()
        return [tuple((x + y) / 2 for x, y in zip(V[i], V[(3 * i + 1) % len(V)]))
                for i in range(min(k, len(V)))]

    def is_empty(self):
        A, b = self.predicate_rows()
        if self.nvar == 0:
            return any(v < 0 for v in b)
        return Q.feasible((), (), A, b) is None

    def predicate_rows(self):
        n = self.nvar
        A, b = [list(r) for r in self.q[2]], list(self.q[3])
        for j in range(n):
            unit = [Fraction(int(k == j)) for k in range(n)]
            if self.q[5][j] is not None:
                A.append(unit)
                b.append(self.q[5][j])
            if self.q[4][j] is not None:
                A.append([-v for v in unit])
                b.append(-self.q[4][j])
        return A, b

    def predicate_ranges(self):
        """Exact least and greatest value of every predicate variable."""
        A, b = self.predicate_rows()
        P = Q.vertices(A, b)
        return ([min(p[j] for p in P) for j in range(self.nvar)],
                [max(p[j] for p in P) for j in range(self.nvar)])

    # exact images -------------------------------------------------------- #
    def affine(self, W, b=None):
        W = [Q.vector(row) for row in np.asarray(W, dtype=np.float64)]
        b = ([Fraction(0)] * len(W) if b is None
             else Q.vector(np.asarray(b, dtype=np.float64).reshape(-1)))
        c, G, C, d, lb, ub = self.q
        cols = list(zip(*G)) if self.nvar else []
        return Exact(([Q.dot(row, c) + bi for row, bi in zip(W, b)],
                      [[Q.dot(row, col) for col in cols] for row in W],
                      C, d, lb, ub))

    def cut(self, H, g):
        H = [Q.vector(row) for row in np.asarray(H, dtype=np.float64)]
        g = Q.vector(np.asarray(g, dtype=np.float64).reshape(-1))
        c, G, C, d, lb, ub = self.q
        cols = list(zip(*G)) if self.nvar else []
        return Exact((c, G, C + [[Q.dot(h, col) for col in cols] for h in H],
                      d + [gi - Q.dot(h, c) for h, gi in zip(H, g)], lb, ub))

    def plus(self, other):
        """The Minkowski sum as one exact set (block-diagonal rows)."""
        c1, G1, C1, d1, lb1, ub1 = self.q
        c2, G2, C2, d2, lb2, ub2 = other.q
        z = Fraction(0)
        C = ([list(r) + [z] * other.nvar for r in C1]
             + [[z] * self.nvar + list(r) for r in C2])
        return Exact(([x + y for x, y in zip(c1, c2)],
                      [list(r1) + list(r2) for r1, r2 in zip(G1, G2)],
                      C, list(d1) + list(d2), lb1 + lb2, ub1 + ub2))


def _vec(y):
    return tuple(Q.vector(np.asarray(y, dtype=np.float64).reshape(-1)))


def assert_holds(result, points, what):
    """Every rational point is a member of the computed set ``result``, by
    the exact simplex on the set's stored rows, with no slack."""
    member = Q.Membership(Q.star(result))
    missed = [p for p in points if not member(p)]
    assert not missed, (
        f"{what}: {len(missed)} of {len(points)} points of the true set are "
        f"not in the result, e.g. {[float(x) for x in missed[0]]}")


def computed_vertices(s):
    """Points whose hull is the computed set ``s``, in exact arithmetic from
    its stored arrays (``rational.set_vertices`` of ``rational.star``)."""
    try:
        return Q.set_vertices(Q.star(s))
    except ValueError as exc:
        raise AssertionError(f"the computed {type(s).__name__} is unbounded "
                             f"({exc}); the true set is bounded") from None


def assert_no_extra(result, true_set, what):
    """The second direction of an exact operation: every vertex of the
    computed set is a point of the true set (exact, no slack)."""
    verts = computed_vertices(result)
    extra = [v for v in verts if not true_set.holds(v)]
    assert not extra, (
        f"{what}: {len(extra)} of {len(verts)} vertices of the result lie "
        f"outside the true set, e.g. {[float(x) for x in extra[0]]}")


# --------------------------------------------------------------------------- #
# The input stars (dyadic)                                                      #
# --------------------------------------------------------------------------- #
_SETS = {
    # two binding rows that are not box rows; the second excludes the lower
    # predicate corner; predicate box not [-1, 1]; generators not diagonal
    "rows": ([0.5, -1.25, 2.0],
             [[2, 0.5, 0], [0, 1.5, -1], [1, 0, 2.5]],
             [[1, 1, 0.5], [-1, 0.5, 0]], [1, 0.75],
             [-1.5, -0.5, -1], [1, 2, 0.5]),
    # predicate box excluding 0 on two variables (a0 >= 1/2, a1 <= -1/4)
    "offset-box": ([-0.5, 1.0, 0.25],
                   [[1, -0.5, 0.25], [0.5, 1, 0], [0, 0.25, 1]],
                   [[1, 0, 1]], [2.5], [0.5, -2, 0.25], [1.5, -0.25, 2]),
    # no predicate box; the rows give the variables different ranges
    # (a0 in [-3, 1], a1 in [-1/2, 2], a2 in [-1, 1]) and cut a corner
    "no-box": ([1.0, -0.25, 0.5],
               [[0.5, 1, 0], [-1, 0.25, 0.5], [0, 0.5, -1]],
               np.vstack([np.eye(3), -np.eye(3), [[1, 1, 1]]]),
               [1, 2, 1, 3, 0.5, 1, 1.5], None, None),
    # a half-open box: lower side all finite, upper open on a1 (rows bound it)
    "half-open": ([0.25, -0.5, 0.0],
                  [[1, 0.5, 0], [0, 1, -0.5], [0.25, 0, 1]],
                  [[1, 1, 1], [0, 1, 0]], [2, 1.75],
                  [-1, -0.5, -2], [1, INF, 0.5]),
    # more entries than predicate variables
    "tall": ([3.0, -5.0, 7.0], [[1, 0.5], [0.25, -1], [0.5, 0.75]],
             [[1, -1]], [1], [-1, -1], [1, 1]),
}


def make(name):
    exact = Exact.of(*_SETS[name])
    return exact, exact.star()


# --------------------------------------------------------------------------- #
# fresh_predicate_tag                                                           #
# --------------------------------------------------------------------------- #
def test_fresh_predicate_tags_differ_between_consecutive_calls():
    """``fresh_predicate_tag`` returns, call after call, a float in [1, 2)
    that no other call returns, exactly representable (an integer multiple
    of 2^-52 above 1). Catches a counter that contributes nothing (every tag
    the same) and an integer scaled below the spacing of floats in [1, 2),
    which rounds odd counter values onto their neighbours; either makes the
    graph reach read two unrelated stars as one predicate system."""
    tags = [fresh_predicate_tag() for _ in range(65)]
    assert all(1.0 <= t < 2.0 for t in tags)
    assert all(((Fraction(t) - 1) * 2 ** 52).denominator == 1 for t in tags)
    assert len(set(tags)) == len(tags), "two calls returned the same tag"


def test_join_of_sin_and_cos_tells_their_new_variables_apart(tmp_path):
    """Gemm(Sin(x) - Cos(x)) on a Star whose range holds extrema of both:
    each node appends a box-only predicate variable that looks like the
    other's and takes the next tag, so the join must tell them apart
    (``n2v.nn.reach._tag_new_predicates``). Float64 graph; the reference is
    numpy in float64 on 128 points with the corners. Catches equal tags of
    consecutive calls, through a model reach."""
    W = np.array([[1.0, -0.5], [0.25, 2.0]])
    b = np.array([0.5, -1.0])
    nodes = [helper.make_node("Sin", ["X"], ["S"]),
             helper.make_node("Cos", ["X"], ["C"]),
             helper.make_node("Sub", ["S", "C"], ["T"]),
             helper.make_node("Gemm", ["T", "W", "b"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 2], inits={"W": W, "b": b},
                     in_type="DOUBLE", out_type="DOUBLE", name="sin-cos")
    lo, hi = np.array([-4.0, -5.0]), np.array([4.0, 3.0])
    star = Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    out = reach_model(load_onnx(path), star, "approx", "Sin - Cos join")
    points = samples(lo, hi)
    assert_covered(out, [W @ (np.sin(p) - np.cos(p)) + b for p in points],
                   "Gemm(Sin(x) - Cos(x)) on a Star")


# --------------------------------------------------------------------------- #
# The constructor and fresh_from_bounds                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, dtype", [
    ("rows", np.float64), ("rows", np.float32), ("no-box", np.float64),
    ("half-open", np.float64), ("tall", np.float64)])
def test_constructed_star_is_the_given_set(name, dtype):
    """``Star.__init__`` stores the set it is given, with a predicate box
    that is not [-1, 1], one with an infinite entry, none at all, more
    entries than variables, and float32 arrays (dyadic, so the float64 star
    is the same set): every vertex of the true set is a member and every
    vertex of the stored set is a point of the true set. Catches one
    predicate bound stored for the other and rows or bounds lost or
    reordered."""
    exact = Exact.of(*_SETS[name])
    star = exact.star(dtype)
    assert star.V.dtype == np.float64 and star.C.dtype == np.float64
    assert_holds(star, exact.vertices(), f"{name}: constructed star")
    assert_no_extra(star, exact, f"{name}: constructed star")


def test_fresh_from_bounds_is_the_box_over_one_new_unused_variable():
    """``Star.fresh_from_bounds`` of a box with a zero-width coordinate,
    intervals beyond plus and minus 1 and wider than 2 is the box exactly
    (every corner is in it, every vertex of it is in the box), and its first
    predicate variable has no generator and the box [0, t] with t in [1, 2),
    t different for the next call. Catches the upper bounds taken from the
    lower ones (the star collapses to a corner) and the tag's bound landing
    on a real variable."""
    lo = np.array([-3.0, 0.5, 1.25, -2.5])
    hi = np.array([1.0, 3.75, 1.25, -0.25])
    star = Star.fresh_from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    box = Exact.of(np.zeros(4), np.eye(4), lb=lo, ub=hi)
    assert_holds(star, box.vertices(), "fresh_from_bounds")
    assert_no_extra(star, box, "fresh_from_bounds")
    assert np.all(star.V[:, 1] == 0), "the tag variable has a generator"
    t = float(star.predicate_ub[0, 0])
    assert star.predicate_lb[0, 0] == 0.0 and 1.0 <= t < 2.0
    again = Star.fresh_from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    assert again.predicate_ub[0, 0] != t


@pytest.mark.parametrize("low", [
    -2.0 ** 20,
    pytest.param(-INF, marks=known_bug(
        8, "Star.from_bounds (Box.to_star) of a box with an infinite bound "
           "returns a star whose generator for that entry is infinite and "
           "whose centre is 0 (a NaN bound is read the same way), which is "
           "not a set, where a refusal or an unbounded predicate variable is "
           "due; Star.fresh_from_bounds does the same (the route of registry "
           "ACT-7)")),
], ids=["far-finite", "infinite"])
def test_from_bounds_of_a_box_with_a_far_lower_bound(low):
    """``Star.from_bounds`` of the box [low, 1] x [0, 1] is refused with
    ``ValueError``, or is a star of finite arrays holding the points
    (-3, 1/2), (1, 1) and (max(low, -2^20), 0) of the box, in exact
    arithmetic. The far but finite box is the passing neighbour."""
    lo, hi = np.array([low, 0.0]), np.array([1.0, 1.0])
    try:
        star = Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    except ValueError:
        return
    assert np.all(np.isfinite(star.V)), f"the basis is not finite: {star.V}"
    points = [(-3.0, 0.5), (1.0, 1.0), (max(low, -2.0 ** 20), 0.0)]
    assert_holds(star, [_vec(p) for p in points], "from_bounds of a far box")


def _dyadic_linear(n_in, n_out, seed):
    rng = np.random.default_rng(seed)
    layer = nn.Linear(n_in, n_out).double()
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(
            rng.integers(-8, 9, size=(n_out, n_in)) / 4.0))
        layer.bias.copy_(torch.as_tensor(rng.integers(-8, 9, size=n_out) / 8.0))
    return layer


def test_softmax_reach_of_a_star_holds_every_output():
    """A model Linear then Softmax on a Star: the Softmax kernel re-encodes
    its output with ``Star.fresh_from_bounds``. Every float64 output of 128
    input points (with the corners) is in the result. Catches the
    re-encoded box collapsed to its lower corner and the last coordinate
    losing the lower part of its interval, through a model reach."""
    lo, hi = np.array([-2.0, -0.5, 0.0]), np.array([0.5, 1.5, 2.5])
    model = nn.Sequential(_dyadic_linear(3, 4, seed=5), nn.Softmax(dim=1))
    star = Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    out = reach_model(model, star, "approx", "Linear then Softmax")
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, 3)),
                   "Linear then Softmax on a Star")


# --------------------------------------------------------------------------- #
# affine_map                                                                    #
# --------------------------------------------------------------------------- #
_W = np.array([[1.5, -0.5, 0.25], [-2.0, 1.0, 0.75], [0.0, 0.5, -1.25],
               [3.0, -0.25, 0.5]])
_B = np.array([2.5, -0.75, 1.0, -3.0])


@pytest.mark.parametrize("name, rows, bias", [
    ("rows", 4, True), ("rows", 2, False), ("offset-box", 4, True),
    ("no-box", 2, True), ("tall", 4, True)])
def test_affine_map_is_the_exact_image(name, rows, bias):
    """``Star.affine_map(W, b)`` with a non-square W whose entries exceed 1
    in magnitude is the exact image, both directions. Catches a bias
    subtracted or dropped, a transposed weight, and rows or predicate bounds
    replaced in the result."""
    exact, star = make(name)
    W, b = _W[:rows], (_B[:rows] if bias else None)
    image = exact.affine(W, b)
    result = star.affine_map(W, b)
    what = f"{name}: affine_map {W.shape} bias={bias}"
    assert_holds(result, image.vertices(), what)
    assert_no_extra(result, image, what)


# --------------------------------------------------------------------------- #
# minkowski_sum                                                                 #
# --------------------------------------------------------------------------- #
# Two stars of dimension 3 with one row each, different right-hand sides and
# different predicate boxes; B's row binds at 3/2 and would cut B at 1/2.
_SUM_A = ([0.5, 0.0, -0.25], [[1, 0.5], [0, 1], [0.5, -0.5]], [[1, 1]], [0.5],
          [-1, -1.5], [1, 1])
_SUM_B = ([-1.0, 0.75, 2.0], [[0.5, 0], [1, 1], [0, -1]], [[1, 1]], [1.5],
          [-0.5, -1], [2, 3])
_SUM_ONE = ([1.5, -2.0, 0.25], [[1], [-0.5], [0]], None, None, [-0.5], [2.5])
# The predicate system of A (rows, right-hand sides, box) with other
# generators and centre: the two share one system, as two stars derived
# from one input do, and their sum must still use independent variables.
_SUM_A_OTHER = ([-1.0, 0.5, 1.0], [[-1, 0.5], [0.5, 0], [0, 1]], [[1, 1]],
                [0.5], [-1, -1.5], [1, 1])
_SUM_NAMES = {"A": _SUM_A, "B": _SUM_B, "one": _SUM_ONE,
              "A-other": _SUM_A_OTHER}


@pytest.mark.parametrize("left, right", [
    ("A", "B"), ("B", "A"), ("one", "A"), ("B", "one"), ("A", "A"),
    ("A", "A-other")])
def test_minkowski_sum_is_the_exact_sum(left, right):
    """``Star.minkowski_sum`` of two stars whose centres, rows, right-hand
    sides, predicate boxes (both sides) and numbers of variables differ, in
    both orders, of a star with itself (the same object), and of two stars
    that share one predicate system with different generators, is the sum
    exactly, both directions (the sum of stars that share a system is not
    the correlated set c1 + c2 + (G1 + G2) a). Catches a centre taken twice from one operand,
    one operand's right-hand sides, lower or upper predicate bounds, or
    generators given to the other's variables."""
    ea, eb = Exact.of(*_SUM_NAMES[left]), Exact.of(*_SUM_NAMES[right])
    sa = ea.star()
    sb = sa if left == right else eb.star()
    result = sa.minkowski_sum(sb)
    true_sum = ea.plus(eb)
    what = f"{left} + {right}: minkowski_sum"
    sums = sorted({tuple(x + y for x, y in zip(u, v))
                   for u in ea.vertices() for v in eb.vertices()})
    assert_holds(result, sums, what)
    assert_no_extra(result, true_sum, what)


# --------------------------------------------------------------------------- #
# intersect_half_space                                                          #
# --------------------------------------------------------------------------- #
def _cut_pad(H, g, star):
    """The outward pad ``intersect_half_space`` may add to a row: the 1e-12
    class of the magnitudes in the row."""
    from tests._oracles.membership import rows as stored_rows
    c, G, C, d, lb, ub = stored_rows(star)
    amax = (np.max(np.abs(np.concatenate([lb, ub])), initial=1.0)
            if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)) else 1.0)
    mag = (np.abs(H) @ np.abs(c) + np.abs(g)
           + (np.abs(H) @ np.abs(G)).sum(1) * amax)
    return 1e-12 * (1.0 + mag)


def assert_inside_cut(result, exact, H, g, pad, what):
    """Every vertex of the computed cut (exact, from its own rows) is a point
    of the old set (no slack) and keeps every new row up to its pad."""
    verts = computed_vertices(result)
    Hq = [Q.vector(h) for h in np.asarray(H, dtype=np.float64)]
    gq = Q.vector(np.asarray(g, dtype=np.float64).reshape(-1))
    outside = [v for v in verts if not exact.holds(v)]
    assert not outside, (f"{what}: a vertex of the result is outside the set "
                         f"that was cut: {[float(x) for x in outside[0]]}")
    crossing = [k for v in verts for k in range(len(Hq))
                if Q.dot(Hq[k], v) > gq[k] + Fraction(float(pad[k]))]
    assert not crossing, f"{what}: a vertex of the result crosses a new row"


def _cutting_rows(exact):
    """Two rows that each cut the set halfway along their range, with the
    set's centre off both hyperplanes."""
    H = np.array([[1.0, -0.5, 0.25], [-0.5, 0.0, 1.0]])[:, :exact.dim]
    g = []
    for h in H:
        values = [Q.dot(Q.vector(h), v) for v in exact.vertices()]
        g.append(float((min(values) + max(values)) / 2))
    return H, np.array(g)


@pytest.mark.parametrize("name", ["rows", "offset-box", "no-box", "tall"])
def test_intersect_half_space_is_the_exact_cut(name):
    """``Star.intersect_half_space(H, g)`` with two rows at once on a star
    that already has rows, generators that are not the identity and a
    centre off both hyperplanes is the intersection: every vertex of the
    true cut is a member (no slack) and every vertex of the result is in
    the old set and keeps the new rows up to the documented rounding pad.
    Catches a right-hand side without ``- H c``, a row in state space, and
    new right-hand sides paired with the old rows."""
    exact, star = make(name)
    H, g = _cutting_rows(exact)
    result = star.intersect_half_space(H, g)
    cut = exact.cut(H, g)
    what = f"{name}: intersect_half_space"
    assert_holds(result, cut.vertices(), what)
    assert_inside_cut(result, exact, H, g, _cut_pad(H, g, star), what)


def _point_star():
    c = np.array([1.5, -2.0, 0.25])
    return Exact.of(c, np.zeros((3, 0))), Star(c.reshape(-1, 1))


@pytest.mark.parametrize("case", [
    "misses", "holds-all", "zero-row-holds", "zero-row-empties",
    "point-touching", "point-beyond"])
def test_intersect_half_space_regimes_then_is_empty_set(case):
    """The cut-then-emptiness check that ``verify_specification`` makes, on
    the regimes of a cut: a half-space that misses the star (empty), one
    that holds all of it (the star itself, both directions), a row of zeros
    with a right-hand side of either sign, and a point star cut through its
    point (not empty: a wrong 'empty' is a false UNSAT) or a quarter beyond
    it (empty). The truth is decided in rationals."""
    if case.startswith("point"):
        exact, star = _point_star()
        H = np.array([[1.0, -0.5, 2.0]])
        at = float(Q.dot(Q.vector(H[0]), exact.q[0]))
        g = np.array([at if case == "point-touching" else at - 0.25])
    else:
        exact, star = make("rows")
        h = np.array([0.5, 1.0, -0.25])
        values = [Q.dot(Q.vector(h), v) for v in exact.vertices()]
        H, g = {
            "misses": (h[None], [float(min(values)) - 0.25]),
            "holds-all": (h[None], [float(max(values)) + 0.25]),
            "zero-row-holds": (np.zeros((1, 3)), [0.5]),
            "zero-row-empties": (np.zeros((1, 3)), [-0.5]),
        }[case]
        g = np.array(g)
    result = star.intersect_half_space(H, g)
    cut = exact.cut(H, g)
    assert result.is_empty_set() is cut.is_empty(), case
    if not cut.is_empty():
        assert_holds(result, cut.vertices(), case)
        if case == "holds-all":
            assert_no_extra(result, exact, case)


# --------------------------------------------------------------------------- #
# convex_hull                                                                   #
# --------------------------------------------------------------------------- #
# The mirror of offset-box on the other side: a0 <= -1/2 (box excludes 0).
_NEG_BOX = ([2.0, -1.0, 0.5], [[1, 0.5, 0], [0, 1, -0.5], [0.25, 0, 1]],
            [[1, 1, 0]], [0.5], [-2, -1, -1], [-0.5, 1, 1])
# Point stars whose centre difference is not a float: 1 - (-2^-60).
_POINT_ONE = ([1.0, 0.5], np.zeros((2, 0)))
_POINT_TINY = ([-2.0 ** -60, 0.5], np.zeros((2, 0)))
# A star with a row whose right-hand side is infinite (dropped by the hull).
_INF_ROW = ([0.0, 1.0, -0.5], [[1, 0, 0.5], [0, 2, 0], [-0.5, 0, 1]],
            [[1, 0, 0], [0, 1, 1]], [INF, 0.75], [-1, -1, -1], [1, 1, 1])
# The predicate system of 'rows' with other generators and centre.
_ROWS_OTHER = ([-1.0, 2.5, 0.5], [[-1, 0.5, 1], [0.5, -2, 0], [0, 1, -0.5]],
               _SETS["rows"][2], _SETS["rows"][3], _SETS["rows"][4],
               _SETS["rows"][5])
_HULL = {"rows": _SETS["rows"], "offset-box": _SETS["offset-box"],
         "rows-other": _ROWS_OTHER, "tall": _SETS["tall"],
         "no-box": _SETS["no-box"], "neg-box": _NEG_BOX,
         "point-one": _POINT_ONE, "point-tiny": _POINT_TINY,
         "inf-row": _INF_ROW}


@pytest.mark.parametrize("left, right", [
    ("offset-box", "rows"), ("rows", "offset-box"), ("rows", "neg-box"),
    ("no-box", "offset-box"), ("offset-box", "no-box"),
    ("point-one", "point-tiny"), ("point-tiny", "point-one"),
    ("inf-row", "rows"), ("rows", "rows"), ("rows", "rows-other"),
    ("tall", "offset-box")])
def test_convex_hull_holds_every_convex_combination(left, right):
    """``Star.convex_hull`` holds both operands and every convex combination
    of their points, in exact arithmetic with no slack: every vertex of each
    operand and midpoints of pairs of vertices. The operands have different
    centres (point stars whose difference 1 + 2^-60 rounds), predicate boxes
    that exclude 0 on either side, negative lower bounds, rows (one with an
    infinite right-hand side), no box at all, different numbers of
    predicate variables, the same star twice, and two stars that share one
    predicate system.
    Catches a hull anchored at the wrong centre, a sign lost on a
    predicate-box row, a result box that keeps a bound excluding 0 (the
    operand at lambda = 0 or 1 is cut off), the second operand's box rows
    unscaled, and a dropped rounding slack of the centre difference."""
    ea, eb = Exact.of(*_HULL[left]), Exact.of(*_HULL[right])
    sa = ea.star()
    sb = sa if left == right else eb.star()
    result = sa.convex_hull(sb)
    what = f"{left} hull {right}"
    assert_holds(result, ea.vertices(), f"{what}: first operand")
    assert_holds(result, eb.vertices(), f"{what}: second operand")
    mids = [tuple((x + y) / 2 for x, y in zip(u, v))
            for u in ea.vertices()[:4] for v in eb.vertices()[:4]]
    assert_holds(result, mids, f"{what}: midpoints")


def test_convex_hull_refuses_what_it_cannot_build():
    """A non-Star operand is refused with ``TypeError``, operands of
    different dimensions and centres whose difference overflows with
    ``ValueError``; never a hull."""
    _, star = make("rows")
    with pytest.raises(TypeError):
        star.convex_hull(np.zeros(3))
    with pytest.raises(ValueError):
        star.convex_hull(Star.from_bounds(np.zeros((2, 1)), np.ones((2, 1))))
    with pytest.raises(ValueError):
        Star(np.array([[1e308]])).convex_hull(Star(np.array([[-1e308]])))


# --------------------------------------------------------------------------- #
# is_empty_set                                                                  #
# --------------------------------------------------------------------------- #
_EIGHTH = 0.125


def _two_var(C, d, lb=(-1, -1), ub=(1, 1), scale=1.0):
    return (np.zeros(2), scale * np.eye(2), C, d,
            None if lb is None else list(lb), None if ub is None else list(ub))


_EMPTINESS = {
    "rows-excluding-lower-corner": _SETS["rows"],
    "offset-box": _SETS["offset-box"],
    "no-box": _SETS["no-box"],
    "empty-by-rows": _two_var([[1, 1]], [-2.25]),
    "empty-by-rows-no-box": _two_var(
        [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]], [1, 1, 1, 1, -2.25],
        lb=None, ub=None),
    "empty-by-box": _two_var(None, None, lb=(1, 0), ub=(0, 1)),
    "sliver": _two_var([[1, 1]], [-2 + 2.0 ** -17], scale=2.0 ** 20),
    "sliver-2^-30": _two_var([[1, 1]], [-2 + 2.0 ** -30], scale=2.0 ** 20),
    "sliver-2^-43": _two_var([[1, 1]], [-2 + 2.0 ** -43]),
    "thin-band": _two_var([[1, -1], [-1, 1]], [-1 + _EIGHTH, 1]),
    "point-on-boundary": (np.array([1.0, -2.0]), np.zeros((2, 0)),
                          np.zeros((2, 0)), [0.0, 0.25], None, None),
}


@pytest.mark.parametrize("solver", ["default", "linprog"])
@pytest.mark.parametrize("case", list(_EMPTINESS))
def test_is_empty_set_answers_as_the_exact_feasibility(case, solver):
    """``Star.is_empty_set`` is True exactly when the predicate polytope has
    no point, decided here in rationals, with the default solver (the
    certified path) and a named one (a default solve confirmed by
    ``confirm_infeasible``). A wrong True is a false UNSAT: a sliver of
    width 2^-17 or 2^-30 with generators of 2^20, one of width 2^-43, a thin
    band off the origin, a point on the boundary of its row and a star whose
    row removes the lower predicate corner are not empty. Catches the feasibility answer returned
    without negation, the LP given the lower bounds as upper bounds, and the
    point-star rule with the tolerance of the wrong sign."""
    exact = Exact.of(*_EMPTINESS[case])
    star = exact.star()
    truth = exact.is_empty()
    assert star.is_empty_set(lp_solver=solver) is truth, (case, truth)


def _linear_reach_and_touching_spec(margin):
    """A Linear layer reached on the 'rows' star (exact image), and the
    unsafe half-space ``G y <= g`` that touches the image at its least value
    of ``G y`` plus ``margin``."""
    exact, star = make("rows")
    layer = _dyadic_linear(3, 2, seed=11)
    W = layer.weight.detach().numpy()
    b = layer.bias.detach().numpy()
    image = exact.affine(W, b)
    G = np.array([[1.0, -0.5]])
    low = min(Q.dot(Q.vector(G[0]), v) for v in image.vertices())
    out = reach_pytorch_model(nn.Sequential(layer), star, method="approx")
    return out, HalfSpace(G, np.array([[float(low) + margin]]))


@pytest.mark.parametrize("margin, verdict", [(0.0, "UNKNOWN"),
                                             (-0.25, "UNSAT")],
                         ids=["touching", "a-quarter-beyond"])
def test_verify_specification_decides_the_cut_of_a_reach_set(margin, verdict):
    """``verify_specification`` cuts each reach star by the unsafe
    half-space (``intersect_half_space``) and calls the cut empty
    (``is_empty_set``). An unsafe region that touches the exact image of a
    Linear layer at one vertex is not disjoint from it (UNSAT would be
    false); one a quarter beyond it is (the cut and the emptiness check are
    exact). The input star's row removes its lower predicate corner.
    Catches a non-empty cut reported empty, by a feasibility answer without
    negation or by a test of the lower predicate corner alone."""
    out, spec = _linear_reach_and_touching_spec(margin)
    result = verify_specification(out, spec)
    assert result.verdict == verdict


# --------------------------------------------------------------------------- #
# contains                                                                      #
# --------------------------------------------------------------------------- #
def _probe(name):
    """``(exact, star, inside, outside)``: vertices and midpoints (members);
    points beyond a vertex, points just outside the predicate box by 2^-16
    while every row holds, and the reflection of a vertex through the centre
    (non-members, each checked here in rationals)."""
    exact, star = make(name)
    V = exact.vertices()
    centre = [sum(v[i] for v in V) / len(V) for i in range(exact.dim)]
    inside = [np.array([float(x) for x in v]) for v in V[:6] + exact.midpoints(3)]
    candidates = [[float(x + (x - m) / 4) for x, m in zip(v, centre)]
                  for v in V[:6]]
    c, G = exact.arrays[0], exact.arrays[1]
    candidates += [list(2 * c - y) for y in inside[:3]]
    if name == "rows":     # a1 = -1/2 - 2^-16 < lb1, both rows hold
        candidates.append(list(c + G @ np.array([-0.25, -0.5 - 2.0 ** -16, 0])))
    if name == "tall":     # off the plane the two generators span
        candidates.append(list(c + np.array([0.0, 0.0, 2.0 ** -16])))
    outside = [np.array(y) for y in candidates
               if not exact.holds(_vec(y))]
    if len(outside) < 4:
        pytest.fail("test data: too few points outside the set")
    return exact, star, inside, outside


@pytest.mark.parametrize("method", ["lp", "algebraic"])
@pytest.mark.parametrize("name", ["rows", "tall", "offset-box", "no-box"])
def test_contains_answers_by_the_rows_and_the_box(name, method):
    """``Star.contains`` accepts the vertices and midpoints of the set and
    rejects points beyond a vertex (inside the bounding box where a row cuts
    it), points outside the predicate box by 2^-16 while every row holds,
    reflections of members through the centre, and, for a star with more
    entries than variables, a point off the plane of its generators; as one
    point (both shapes) and as a batch. Catches the rows ignored, the
    predicate box padded, the point taken relative to the centre with the
    wrong sign, and the residual test of the algebraic path dropped."""
    exact, star, inside, outside = _probe(name)
    for y in outside:
        if distance(star, y) <= 1e-9:
            pytest.fail(f"test data: {y} is a non-member within 1e-9")
    for y in inside:
        assert star.contains(y, method=method) is True, y
    for y in outside:
        assert star.contains(y.reshape(-1, 1), method=method) is False, y
    batch = star.contains(np.vstack(inside + outside), method=method)
    assert batch.tolist() == [True] * len(inside) + [False] * len(outside)


_POINT = np.array([1.5, -2.0, 0.25])


@pytest.mark.parametrize("method, rows", [
    pytest.param("lp", None, marks=known_bug(
        8, "Star.contains(method='lp') of a star with no predicate variable "
           "hands the LP solver a problem with no variable, which raises, "
           "and _contains_lp reads the exception as 'not a member': the "
           "point star rejects its own point")),
    ("algebraic", None),
    ("lp", -0.25),
    pytest.param("algebraic", -0.25, marks=known_bug(
        8, "Star._contains_algebraic skips the rows of a star with no "
           "predicate variable (its C has no column), so a point star "
           "emptied by a row 0 <= d with d < 0 still contains its point")),
], ids=["lp-point", "algebraic-point", "lp-empty-point",
        "algebraic-empty-point"])
def test_contains_of_a_point_star(method, rows):
    """A star with no predicate variable is its centre, or nothing when a row
    ``0 <= d`` has ``d < 0`` (the form a point cut off by a half-space
    takes): ``contains`` accepts the centre exactly when the set is not
    empty, and never accepts another point."""
    C = None if rows is None else np.zeros((1, 0))
    d = None if rows is None else np.array([[rows]])
    star = Star(_POINT.reshape(-1, 1), C, d)
    empty = rows is not None and rows < 0
    assert star.contains(_POINT + np.array([0.0, 0.25, 0.0]),
                         method=method) is False
    assert star.contains(_POINT, method=method) is (not empty)


@pytest.mark.parametrize("undecided", [
    False,
    pytest.param(True, marks=known_bug(
        8, "Star._contains_lp catches every exception of the feasibility "
           "LP (an LPFailure: no solve decided, a solver that is not "
           "installed or whose licence expired) and reports 'not a member', "
           "so a member is rejected where a refusal is due")),
], ids=["decided", "undecided"])
def test_contains_never_reads_an_undecided_lp_as_not_a_member(undecided,
                                                              monkeypatch):
    """When the feasibility LP of ``contains`` decides nothing
    (``check_feasibility`` raises ``LPFailure``, injected here), a member
    must not be reported as a non-member: the call raises or answers True.
    The 'decided' case is the passing neighbour with the real solver."""
    import n2v.sets.star as star_module
    exact, star = make("rows")
    y = np.array([float(x) for x in exact.vertices()[0]])
    if undecided:
        def undecided_lp(*args, **kwargs):
            raise LPFailure("feasibility LP did not decide (injected)")
        monkeypatch.setattr(star_module, "check_feasibility", undecided_lp)
    try:
        answer = star.contains(y, method="lp")
    except (LPFailure, ValueError, NotImplementedError):
        return
    assert answer is True


@pytest.mark.parametrize("scale", [
    0,
    pytest.param(30, marks=known_bug(
        8, "Star._contains_algebraic checks the least-squares residual of a "
           "star with more entries than variables against an absolute 1e-9, "
           "so a member of a star whose entries are near 1e9 is rejected by "
           "the float64 rounding of the residual alone")),
], ids=["scale-1", "scale-2^30"])
def test_contains_algebraic_keeps_the_members_of_a_large_tall_star(scale):
    """The algebraic path on a star with more entries than variables, all
    data dyadic and scaled by 2^scale (so the member is exact): the member
    is accepted. The tolerance of the residual must scale with the data."""
    c, G = (np.array(v, dtype=np.float64) * 2.0 ** scale
            for v in (_SETS["tall"][0], _SETS["tall"][1]))
    star = Star(np.hstack([c[:, None], G]), None, None, -np.ones(2),
                np.ones(2))
    y = c + G @ np.array([0.75, -0.5])
    assert star.contains(y, method="lp") is True
    assert star.contains(y, method="algebraic") is True


# --------------------------------------------------------------------------- #
# to_image_star                                                                 #
# --------------------------------------------------------------------------- #
def test_to_image_star_reads_the_flat_order_as_height_width_channel():
    """``Star.to_image_star(H, W, C)`` with H != W and two channels stores
    height H and width W and puts flat entry (h W + w) C + c at pixel
    (h, w, c), with the rows and the predicate box of the star. A dimension
    that does not match is refused with ``ValueError``."""
    H, W, C = 3, 5, 2
    rng = np.random.default_rng(1)
    V = rng.integers(-8, 9, size=(H * W * C, 4)) / 8.0
    star = Star(V, np.array([[1.0, -1.0, 0.5]]), np.array([[0.25]]),
                -np.ones(3), 2 * np.ones(3))
    image = star.to_image_star(H, W, C)
    assert (image.height, image.width, image.num_channels) == (H, W, C)
    for h, w, c in np.ndindex(H, W, C):
        assert np.array_equal(image.V[h, w, c], V[(h * W + w) * C + c])
    assert np.array_equal(image.C, star.C) and np.array_equal(image.d, star.d)
    assert np.array_equal(image.predicate_lb, star.predicate_lb)
    assert np.array_equal(image.predicate_ub, star.predicate_ub)
    with pytest.raises(ValueError):
        star.to_image_star(H, W, C + 1)


def test_relu_then_pooling_on_a_tall_two_channel_image_star():
    """Approx ReLU on a (2, 3, 5) ImageStar works on the flat star and comes
    back through ``Star.to_image_star``; an asymmetric AvgPool2d after it
    pools the pixels the conversion recorded. Every float64 output of the
    model on 128 points is in the result, whose stored shape is checked.
    Catches height and width swapped by the conversion, through a model
    reach."""
    lo, hi = pixel_box(2, 3, 5, seed=2)
    model = nn.Sequential(nn.ReLU(), nn.AvgPool2d((2, 3), stride=1))
    out = reach_model(model, image_set("ImageStar", lo, hi), "approx",
                      "relu then avgpool")
    ys = replay_torch(model, samples(lo, hi), (1, 2, 3, 5))
    assert_covered(out, ys, "relu then avgpool", out_chw=(2, 2, 3))


# --------------------------------------------------------------------------- #
# sample                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stored", ["none", "wider-and-shifted", "missing"])
def test_sample_returns_member_points_only(stored):
    """``Star.sample(N)`` returns at most N points as a (dim, k) array, each
    a point of the set (exactly, up to the LP containment tolerance of
    1e-9): without stored state bounds (it computes them), and with stored
    state bounds that are wrong on purpose, wider than the set and shifted
    (the sampler must still reject by the rows), or missing the set
    altogether (no point at all). Catches a sampler that skips the
    rejection and an empty result of the wrong shape."""
    exact, star = make("rows")
    lo = np.array([float(min(v[i] for v in exact.vertices()))
                   for i in range(3)])
    hi = np.array([float(max(v[i] for v in exact.vertices()))
                   for i in range(3)])
    if stored == "wider-and-shifted":
        star.state_lb = (lo - 0.25).reshape(-1, 1)
        star.state_ub = (hi + 0.75).reshape(-1, 1)
    elif stored == "missing":
        star.state_lb = (hi + 1.0).reshape(-1, 1)
        star.state_ub = (hi + 2.0).reshape(-1, 1)
    pts = star.sample(30)
    assert pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] <= 30
    if stored == "missing":
        assert pts.shape[1] == 0
        return
    assert pts.shape[1] >= 1
    far = [k for k in range(pts.shape[1])
           if not exact.holds(_vec(pts[:, k]))
           and distance(star, pts[:, k]) > 1e-9]
    assert not far, f"{len(far)} of {pts.shape[1]} samples are not in the set"


# --------------------------------------------------------------------------- #
# get_vertices and _fibonacci_sphere                                            #
# --------------------------------------------------------------------------- #
def _close_to(points, targets, tol=1e-9):
    return all(any(np.max(np.abs(p - t)) <= tol * (1 + np.max(np.abs(t)))
                   for t in targets) for p in points)


def test_get_vertices_of_a_polygon_are_its_vertices():
    """``get_vertices`` of a 2-D star with a cut corner and an off-origin
    centre returns exactly its vertices (each has a normal cone wider than
    the spacing of the 64 directions). Catches the centre or the rows left
    out of the LPs and points outside the set."""
    exact = Exact.of([2.5, -1.25], [[1, 0.5], [0, 1.5]], [[1, 1]], [1],
                     [-1, -1], [1, 1])
    got = exact.star().get_vertices()
    true = [np.array([float(x) for x in v]) for v in exact.vertices()]
    assert got is not None and got.shape[1] == 2
    assert _close_to(got, true), "a returned point is not a vertex"
    assert _close_to(true, got), "a vertex was not returned"


def test_get_vertices_of_a_box_in_three_dimensions_and_a_projection():
    """``get_vertices`` of a 3-D box star finds all 8 corners (the directions
    of ``_fibonacci_sphere`` reach both hemispheres), and the projection of
    the 'rows' star onto coordinates (0, 2) returns points of the projected
    set only. Catches directions on one hemisphere and generators projected
    onto the wrong coordinates."""
    lo, hi = np.array([-3.0, 0.5, -0.25]), np.array([0.5, 1.5, 2.0])
    box = Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    got = box.get_vertices()
    corners = [np.where(bits, hi, lo) for bits in np.ndindex(2, 2, 2)]
    assert got is not None and _close_to(got, corners)
    assert _close_to(corners, got), "a corner of the box was not returned"
    exact, star = make("rows")
    proj = star.get_vertices(projection=[0, 2])
    flat = [np.array([float(v[0]), float(v[2])]) for v in exact.vertices()]
    assert proj is not None and proj.shape[1] == 2
    assert _close_to(proj, flat), "a returned point is not a projected vertex"


def test_get_vertices_degenerate_cases_and_refusals():
    """A point star returns its centre, an empty star returns None, a star
    of dimension 4 without a projection and a projection onto one coordinate
    are refused with ``ValueError``."""
    c = np.array([1.5, -2.0])
    got = Star(c.reshape(-1, 1)).get_vertices()
    assert got.shape == (1, 2) and np.array_equal(got[0], c)
    assert Exact.of(*_two_var([[1, 1]], [-2.25])).star().get_vertices() is None
    four = Star.from_bounds(np.zeros((4, 1)), np.ones((4, 1)))
    with pytest.raises(ValueError):
        four.get_vertices()
    with pytest.raises(ValueError):
        four.get_vertices(projection=[1])


@pytest.mark.parametrize("dim", [
    2,
    pytest.param(3, marks=known_bug(
        23, "Star.get_vertices(n_directions=1) of a 3-D star raises "
            "ZeroDivisionError in _fibonacci_sphere (i / (n - 1) with n = 1) "
            "where it must answer or refuse with ValueError",
        raises=ZeroDivisionError)),
])
def test_get_vertices_with_one_direction_answers_or_refuses(dim):
    """``n_directions=1`` is a number of directions like any other: the call
    returns points of the star (or None), or refuses with ``ValueError``.
    The 2-D case is the passing neighbour."""
    star = Star.from_bounds(np.zeros((dim, 1)), np.ones((dim, 1)))
    try:
        got = star.get_vertices(n_directions=1)
    except ValueError:
        return
    if got is not None:
        assert all(np.all(p >= -1e-9) and np.all(p <= 1 + 1e-9) for p in got)


# --------------------------------------------------------------------------- #
# ensure_finite_predicate_box                                                   #
# --------------------------------------------------------------------------- #
_LOWER_MISSING = ([0.5, -0.25], [[1, 0.5], [0.25, -1]],
                  [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]],
                  [1, 3, 2, 0.5, 1.5], None, [1, 2])


def _finite_box_case(name):
    if name == "lower-missing":
        # a predicate_ub is given and predicate_lb is None: a0 reaches -3
        exact = Exact.of(*_LOWER_MISSING)
        c, G, C, d, _, ub = exact.arrays
        return exact, Star(np.hstack([c[:, None], G]), C, d.reshape(-1, 1),
                           None, ub)
    return make(name)


@pytest.mark.parametrize("solver", [None, "linprog"])
@pytest.mark.parametrize("name", ["no-box", "half-open", "lower-missing"])
def test_ensure_finite_predicate_box_holds_every_feasible_predicate_point(
        name, solver):
    """``ensure_finite_predicate_box`` of a star without a predicate box
    (variables of different ranges, one reaching -3), of a half-open box
    (lower side finite, one upper entry infinite) and of a star with only
    upper bounds returns a fully finite box that holds the exact range of
    every predicate variable (no slack), keeps the finite entries it was
    given, and is the same set, both directions. Catches every variable
    solved with the objective of variable 0, a missing lower bound read as
    -1, the last variable skipped, the early exit taken when only one side
    is finite, finite upper entries replaced by lower ones, and the LP pad
    applied inward."""
    exact, star = _finite_box_case(name)
    star2, empty = ensure_finite_predicate_box(star, lp_solver=solver)
    assert empty is False
    lb = star2.predicate_lb.reshape(-1)
    ub = star2.predicate_ub.reshape(-1)
    assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)), (lb, ub)
    a_lo, a_hi = exact.predicate_ranges()
    bad = [j for j in range(exact.nvar)
           if not (Fraction(lb[j]) <= a_lo[j] and Fraction(ub[j]) >= a_hi[j])]
    assert not bad, f"variables {bad}: box {lb}, {ub} cuts the polytope"
    given_lb, given_ub = exact.arrays[4], exact.arrays[5]
    for given, got in ((given_lb, lb), (given_ub, ub)):
        if given is not None:
            keep = np.isfinite(given)
            assert np.array_equal(got[keep], given[keep]), (given, got)
    assert_holds(star2, exact.vertices(), f"{name}: finite box star")
    assert_no_extra(star2, exact, f"{name}: finite box star")


def test_ensure_finite_predicate_box_reports_empty_only_when_empty():
    """A star without a box whose rows contradict each other by a quarter is
    reported empty (the same star and True); slivers of width 2^-17 and
    2^-43 are not, and their boxes hold them; a fully finite box is returned as it is
    (the same object, no LP), even when its rows empty it. A wrong 'empty'
    makes reach return nothing, a false UNSAT."""
    empty = Exact.of(*_EMPTINESS["empty-by-rows-no-box"]).star()
    star2, is_empty = ensure_finite_predicate_box(empty)
    assert is_empty is True and star2 is empty
    for width in (2.0 ** -17, 2.0 ** -43):
        sliver = Exact.of(*_two_var([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]],
                                    [1, 1, 1, 1, -2 + width], lb=None,
                                    ub=None))
        star2, is_empty = ensure_finite_predicate_box(sliver.star())
        assert is_empty is False, width
        assert_holds(star2, sliver.vertices(), f"sliver {width} after finite box")
    finite = Exact.of(*_EMPTINESS["empty-by-rows"]).star()
    star2, is_empty = ensure_finite_predicate_box(finite)
    assert star2 is finite and is_empty is False


def test_ensure_finite_predicate_box_refuses_what_it_cannot_bound():
    """A predicate variable unbounded over the polytope, a NaN in the box, a
    NaN or an infinity in the rows are refused with ``ValueError``."""
    open_star = Star(np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0]]),
                     np.array([[1.0]]))
    nan_box = Star(np.array([[0.0, 1.0]]), None, None, np.array([np.nan]),
                   np.array([1.0]))
    nan_rows = Star(np.array([[0.0, 1.0]]), np.array([[np.nan]]),
                    np.array([[1.0]]), None, None)
    inf_rows = Star(np.array([[0.0, 1.0]]), np.array([[1.0], [-1.0]]),
                    np.array([[1.0], [np.inf]]), None, None)
    for star in (open_star, nan_box, nan_rows, inf_rows):
        with pytest.raises(ValueError):
            ensure_finite_predicate_box(star)


def _stub(minimize_value, maximize_value, status="optimal"):
    """A bound LP that answers ``minimize_value`` or ``maximize_value`` with
    ``status`` and no solution point."""
    def solve(f, *args, minimize=True, **kwargs):
        return None, (minimize_value if minimize else maximize_value), \
            status, {}
    return solve


@pytest.mark.parametrize("injected", ["contradicting", "one-inward",
                                      "finite-for-unbounded"])
def test_ensure_finite_predicate_box_under_injected_bound_lps(injected,
                                                              monkeypatch):
    """Bound LPs that go wrong in a way a solver can, injected: two solves
    whose minimum lies above their maximum must be refused (an empty box
    would read as a vacuous UNSAT); when one solve of a bound stops inside
    the true range, the more outward optimum is kept, so the box still holds
    the exact range [-3, 1] of a0; when the careful solve reports a finite
    optimum for a variable that is unbounded and the other reports
    'unbounded', the helper must refuse and not return a finite box."""
    import n2v.utils.lpsolver as lpsolver
    if injected == "contradicting":
        monkeypatch.setattr(lpsolver, "_solve_lp_careful", _stub(2.0, -2.0))
        monkeypatch.setattr(lpsolver, "solve_lp", _stub(2.0, -2.0))
        with pytest.raises(ValueError):
            ensure_finite_predicate_box(make("no-box")[1])
    elif injected == "one-inward":
        # Both solves return their optimum without a solution point (so the
        # widening by solution points cannot hide the choice); the careful
        # one is exact, the other stops 1/2 inside.
        real, careful = lpsolver.solve_lp, lpsolver._solve_lp_careful

        def inward(f, *args, minimize=True, **kwargs):
            x, val, status, info = real(f, *args, minimize=minimize, **kwargs)
            if val is not None:
                val = val + 0.5 if minimize else val - 0.5
            return None, val, status, info

        def pointless(f, *args, minimize=True, **kwargs):
            x, val, status, info = careful(f, *args, minimize=minimize,
                                           **kwargs)
            return None, val, status, info
        monkeypatch.setattr(lpsolver, "solve_lp", inward)
        monkeypatch.setattr(lpsolver, "_solve_lp_careful", pointless)
        exact, star = make("no-box")
        star2, _ = ensure_finite_predicate_box(star)
        a_lo, a_hi = exact.predicate_ranges()
        for j in range(3):
            assert Fraction(float(star2.predicate_lb[j, 0])) <= a_lo[j]
            assert Fraction(float(star2.predicate_ub[j, 0])) >= a_hi[j]
    else:
        real = lpsolver._solve_lp_careful

        def finite(f, *args, minimize=True, **kwargs):
            x, val, status, info = real(f, *args, minimize=minimize, **kwargs)
            if status == "unbounded":
                return None, 1e6, "optimal", info
            return x, val, status, info
        monkeypatch.setattr(lpsolver, "_solve_lp_careful", finite)
        open_star = Star(np.array([[0.0, 1.0, 0.0]]),
                         np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
                         np.array([[1.0], [1.0], [0.0]]))
        with pytest.raises(ValueError):
            ensure_finite_predicate_box(open_star)


def _relu_net(seed):
    l1, l2 = _dyadic_linear(3, 3, seed), _dyadic_linear(3, 2, seed + 1)
    return nn.Sequential(l1, nn.ReLU(), l2), (l1, l2)


@pytest.mark.parametrize("method", ["approx", "exact"])
def test_reach_of_a_star_without_a_predicate_box_holds_the_exact_image(
        method):
    """``reach`` of a flat Star with no predicate box finitizes the box first
    (``ensure_finite_predicate_box``). Through Linear, ReLU, Linear, every
    vertex of every piece of the exact image (rationals, the input polytope
    mapped piece by piece) is in a returned set, and 128 member points
    replayed in torch float64 are covered. The input's variables have
    different ranges and one reaches -3. Catches a finitized box cut to
    another variable's range or to [-1, ...], through a model reach."""
    exact, star = make("no-box")
    model, (l1, l2) = _relu_net(seed=21)
    out = reach_model(model, star, method, f"{method} reach of a no-box star")
    A, b = exact.predicate_rows()
    pieces = Q.affine(Q.polytope(A, b), exact.arrays[1], exact.arrays[0])
    pieces = Q.relu(Q.affine(pieces, l1.weight.detach().numpy(),
                             l1.bias.detach().numpy()))
    pieces = Q.affine(pieces, l2.weight.detach().numpy(),
                      l2.bias.detach().numpy())
    computed = [Q.star(s) for s in out]
    missed = Q.escapes(pieces, *computed)
    assert not missed, f"{len(missed)} vertices of the exact image escape"
    if method == "exact":
        assert not Q.straddlers(pieces, *computed)
    # 128 member points: every vertex of the predicate polytope, then
    # random convex combinations of them, mapped through the basis.
    P = np.array([[float(x) for x in p] for p in Q.vertices(A, b)])
    rng = np.random.default_rng(3)
    alphas = list(P)
    while len(alphas) < 128:
        w = rng.integers(0, 4, size=len(P)).astype(float) + 0.25
        alphas.append((w / w.sum()) @ P)
    c, G = exact.arrays[0], exact.arrays[1]
    points = [c + G @ a for a in alphas]
    assert_covered(out, replay_torch(model, points, (1, 3)),
                   f"{method} reach of a no-box star")


def test_reach_of_an_empty_star_without_a_predicate_box_returns_no_set():
    """An input star without a predicate box whose rows are contradictory
    (exactly empty) reaches nothing: ``reach`` returns no set, and does not
    raise. Catches the emptiness confirmation read the wrong way round."""
    star = Exact.of(*_EMPTINESS["empty-by-rows-no-box"]).star()
    model = nn.Sequential(nn.Linear(2, 2).double())
    assert reach_pytorch_model(model, star, method="approx") == []
