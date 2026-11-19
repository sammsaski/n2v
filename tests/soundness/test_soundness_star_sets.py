"""Soundness cells for the methods of ``Star`` that reach and verification
rely on, and for ``n2v/sets/predicate_box.py``.

Covered: the constructor, ``from_bounds``, ``fresh_from_bounds`` and
``fresh_predicate_tag``; the set operations ``affine_map``,
``minkowski_sum``, ``intersect_half_space`` and ``convex_hull`` (and its
refusal of centres whose difference overflows); the range queries
``get_range``, ``get_ranges`` (batch path, parallel path, and the fallback of
each when an LP gives no optimum), ``get_min``, ``get_max``, ``get_box``,
``estimate_range`` and ``estimate_ranges``, with their refusal of an index
outside the star, every HiGHS value and spelling of ``lp_solver`` (handed to
every LP; the CVXPY solvers stop inside the range and are marked in
``test_soundness_range_rounding.py``, registry SET-6), the global solver
setting, stored state bounds that are stale, a star unbounded in one
direction, an empty point star, constant entries, and the approx and exact
ReLU that read them through ``reach_layer`` and the model entry;
``is_empty_set``, ``contains``, ``sample``, ``get_vertices``,
``to_image_star`` (alone and consumed by ReLU then a pooling layer) and
``__repr__``; and ``ensure_finite_predicate_box``. The input stars cover a
box star with a zero-width coordinate, a star with binding rows that are not
box rows, a predicate box that excludes 0, a predicate box open on one side,
a predicate box given on one side only (the other bound None), no predicate
box at all (also with a constant entry, and with predicate columns scaled by
2^-30 and 2^30), more predicate variables than dimensions, one predicate
variable, and a point star of twelve dimensions. The parallel range path
reached through the global settings (``set_parallel``, the automatic
threshold, the worker counts) is checked in ``test_soundness_parallel.py``.

Oracle. Every input star is written down here as exact rationals, the set
``{c + G a : C a <= d, lb <= a <= ub}`` of the same dyadic arrays the n2v star
is built from (:class:`Exact`), so the reference never comes from n2v. Its
vertices are the images of the vertices of its predicate polytope
(``tests/_oracles/rational.py``: ``set_vertices``), and the exact range of a
coordinate is the least and greatest value over them. The image of an
operation is computed in rationals from those vertices (an affine map maps
vertices to vertices, a Minkowski sum is spanned by sums of vertices, a cut
is a new predicate row). A computed set is read from its stored arrays
(``rational.star``, ``membership.rows``) and judged three ways: every exact
vertex is a member (exact simplex, no slack); for a method that claims the
exact set, every vertex of the computed set, enumerated exactly from its
own rows, is a member of the true set (:func:`assert_no_extra`,
:func:`assert_inside_cut`; no slack beyond the documented outward pad of a
cut row); and float points of the true set are members by the LP oracle
(``membership.distance``, slack ``LP_TOL``).

Slack. None against the exact references for every set-returning method and
for ``estimate_range(s)``, whose bounds are documented as outward-rounded
enclosures. The LP range queries (``get_range(s)``, ``get_min``, ``get_max``,
``get_box``) add the centre in round-to-nearest (registry row SET-5, owned by
the rounding author), so they get a pad of ``1e-12`` relative, and, because
they claim the exact range, a tightness check at ``1e-9`` relative that sees
a query that silently returns the interval estimate instead.

Layout. A flat star is in the order of its own entries. ``to_image_star``
reads the flat order as the height, width, channel order an ImageStar stores;
image outputs of the composition cells are compared with
``assert_covered(..., out_chw=...)``.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.optimize import linprog

from n2v.config import config
from n2v.sets import Star
from n2v.sets.image_star import ImageStar
from n2v.sets.predicate_box import ensure_finite_predicate_box
from n2v.sets.star import fresh_predicate_tag
from n2v.utils.lp_solver_enum import LPSolver
from n2v.utils.lp_solver_enum import resolve as resolve_solver
from tests._oracles import rational as Q
from tests._oracles.membership import LP_TOL, distance, uncovered
from tests._oracles.membership import rows as stored_rows
from tests.cell_support import (assert_covered, flat_box, image_set,
                                image_split_cover, pixel_box, reach_model,
                                reach_one_layer, replay_torch, samples)
from tests.known_bugs import known_bug

INF = float("inf")
LP_REL = 1e-12        # pad of the LP range queries (float64 class)
TIGHT_REL = 1e-9      # how far an LP range query may lie outside the range


# --------------------------------------------------------------------------- #
# The exact set                                                                 #
# --------------------------------------------------------------------------- #
def _bound(values, k):
    if values is None or math.isinf(values[k]):
        return None
    return Fraction(float(values[k]))


def membership(q):
    """Exact membership in ``{c + G a : C a <= d, lb <= a <= ub}`` (``q`` in
    the form ``rational.star`` returns), as a callable on rational points.

    When ``G`` has full column rank the predicate point of a member is
    unique: it is the solution of ``G a = y - c`` on a set of independent
    rows, and ``y`` is a member exactly when that ``a`` solves every row of
    ``G a = y - c`` and keeps every row of ``C a <= d`` and every finite
    bound. That is one small exact solve per point instead of a simplex on
    all the entries of an image. Otherwise ``rational.Membership`` decides."""
    c, G, C, d, lb, ub = q
    n = len(lb)
    chosen, rows = [], []
    for i, row in enumerate(G):
        if len(rows) == n:
            break
        if Q.rank(rows + [row]) > len(rows):
            rows.append(row)
            chosen.append(i)
    if n == 0 or len(rows) < n:
        return Q.Membership(q)

    def member(y):
        y = Q.vector(y)
        if len(y) != len(c):
            raise ValueError(f"the set has {len(c)} entries, the point "
                             f"{len(y)}")
        rhs = [yi - ci for yi, ci in zip(y, c)]
        a = Q.solve(rows, [rhs[i] for i in chosen])
        return (all(Q.dot(row, a) == r for row, r in zip(G, rhs))
                and all(Q.dot(row, a) <= r for row, r in zip(C, d))
                and all(lo is None or v >= lo for v, lo in zip(a, lb))
                and all(hi is None or v <= hi for v, hi in zip(a, ub)))
    return member


class Exact:
    """``{c + G a : C a <= d, lb <= a <= ub}`` in Fractions, as written in
    this file. ``q`` is the tuple ``rational.member`` reads; an absent bound
    is None. ``arrays`` (when the set was given by float arrays) builds the
    n2v star of the same numbers with the constructor under test."""

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
        C = np.zeros((0, n)) if C is None else np.asarray(C, dtype=np.float64)
        C = C.reshape(C.shape[0] if C.ndim == 2 else -1, n)
        d = (np.zeros(C.shape[0]) if d is None
             else np.asarray(d, dtype=np.float64).reshape(-1))
        lb = None if lb is None else np.asarray(lb, np.float64).reshape(-1)
        ub = None if ub is None else np.asarray(ub, np.float64).reshape(-1)
        q = (Q.vector(c), [Q.vector(row) for row in G],
             [Q.vector(row) for row in C], Q.vector(d),
             [_bound(lb, k) for k in range(n)],
             [_bound(ub, k) for k in range(n)])
        return cls(q, (c, G, C, d, lb, ub))

    def star(self):
        """The n2v Star of the arrays (``Star.__init__`` under test)."""
        c, G, C, d, lb, ub = self.arrays
        return Star(np.hstack([c[:, None], G]), C, d.reshape(-1, 1), lb, ub)

    def vertices(self):
        if self._vertices is None:
            self._vertices = Q.set_vertices(self.q)
            if not self._vertices:
                pytest.fail("test data: the exact set is empty")
        return self._vertices

    def holds(self, y):
        if self._member is None:
            self._member = membership(self.q)
        return self._member(y)

    def ranges(self):
        V = self.vertices()
        return ([min(v[i] for v in V) for i in range(self.dim)],
                [max(v[i] for v in V) for i in range(self.dim)])

    def points(self, extra=8):
        """Float points of the set: every vertex and midpoints of vertex
        pairs (rounded to float64; the LP oracle's slack covers that)."""
        V = self.vertices()
        out = [np.array([float(x) for x in v]) for v in V]
        for k in range(min(extra, len(V) - 1)):
            a, b = V[k], V[(3 * k + 1) % len(V)]
            out.append(np.array([float((x + y) / 2) for x, y in zip(a, b)]))
        return out

    # exact images ---------------------------------------------------------
    def affine(self, W, b=None):
        W = [Q.vector(row) for row in np.asarray(W, dtype=np.float64)]
        b = [Fraction(0)] * len(W) if b is None else Q.vector(
            np.asarray(b, dtype=np.float64).reshape(-1))
        c, G, C, d, lb, ub = self.q
        columns = list(zip(*G)) if self.nvar else []
        G2 = [[Q.dot(row, col) for col in columns] for row in W]
        c2 = [Q.dot(row, c) + bi for row, bi in zip(W, b)]
        return Exact((c2, G2, C, d, lb, ub))

    def permuted(self, order):
        """The same set with its entries reordered: entry ``k`` of the new
        set is entry ``order[k]`` of this one."""
        c, G, C, d, lb, ub = self.q
        return Exact(([c[i] for i in order], [G[i] for i in order],
                      C, d, lb, ub))

    def cut(self, H, g):
        """The set intersected with ``H y <= g`` (rows in predicate space)."""
        H = [Q.vector(row) for row in np.asarray(H, dtype=np.float64)]
        g = Q.vector(np.asarray(g, dtype=np.float64).reshape(-1))
        c, G, C, d, lb, ub = self.q
        columns = list(zip(*G)) if self.nvar else []
        rows = [[Q.dot(h, col) for col in columns] for h in H]
        rhs = [gi - Q.dot(h, c) for h, gi in zip(H, g)]
        return Exact((c, G, C + rows, d + rhs, lb, ub))


def sum_vertices(a, b):
    """Points whose hull is the Minkowski sum of two exact sets."""
    return sorted({tuple(x + y for x, y in zip(u, v))
                   for u in a.vertices() for v in b.vertices()})


def exact_sum(a, b):
    """The Minkowski sum as one exact set (block-diagonal rows)."""
    c1, G1, C1, d1, lb1, ub1 = a.q
    c2, G2, C2, d2, lb2, ub2 = b.q
    z = Fraction(0)
    C = ([list(r) + [z] * b.nvar for r in C1]
         + [[z] * a.nvar + list(r) for r in C2])
    return Exact(([x + y for x, y in zip(c1, c2)],
                  [list(r1) + list(r2) for r1, r2 in zip(G1, G2)],
                  C, list(d1) + list(d2), lb1 + lb2, ub1 + ub2))


# --------------------------------------------------------------------------- #
# Judging a computed set                                                        #
# --------------------------------------------------------------------------- #
def support(s, h):
    """``max h . y`` over the computed set ``s``, by scipy's HiGHS on the
    set's own stored rows (``membership.rows``): +inf when unbounded, -inf
    when empty. No n2v method is called."""
    c, G, C, d, lb, ub = stored_rows(s)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    if G.shape[1] == 0:
        return float(h @ c) if np.all(d >= 0) else -INF
    bounds = [(None if math.isinf(lo) else lo, None if math.isinf(hi) else hi)
              for lo, hi in zip(lb, ub)]
    res = linprog(-(h @ G), A_ub=C if C.shape[0] else None,
                  b_ub=d if C.shape[0] else None, bounds=bounds,
                  method="highs")
    if res.status == 3:
        return INF
    if res.status == 2:
        return -INF
    if res.status != 0:
        pytest.fail(f"support LP did not solve: {res.message}")
    return float(h @ c) - float(res.fun)


def assert_holds_exact(result, exact_vertices, what):
    """Every exact vertex is a member of one of the computed sets, in
    rationals with no slack."""
    sets = result if isinstance(result, list) else [result]
    members = [membership(Q.star(s)) for s in sets]
    missed = [v for v in exact_vertices if not any(m(v) for m in members)]
    assert not missed, (
        f"{what}: {len(missed)} of {len(exact_vertices)} vertices of the true "
        f"set are not in the result, e.g. {[float(x) for x in missed[0]]}")


def computed_vertices(s):
    """Points whose hull is the computed set ``s``, in exact arithmetic from
    its stored arrays (``rational.star``): the images of the vertices of its
    predicate polytope. ``rational.set_vertices`` does the same, but its
    boundedness test enumerates a cone with two more rows per variable, which
    took seconds for six variables; here the polytope is bounded exactly when
    no direction ``r`` with ``A r <= 0`` has a coordinate of size 1 (2 n small
    exact feasibility problems). An unbounded predicate polytope is an
    ``AssertionError``: every true set of these cells is bounded."""
    c, G, C, d, lb, ub = Q.star(s)
    n = len(lb)
    A, b = [list(row) for row in C], list(d)
    for j in range(n):
        unit = [Fraction(int(k == j)) for k in range(n)]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    if n == 0:
        return [tuple(c)] if all(v >= 0 for v in b) else []
    zeros = [Fraction(0)] * len(A)
    for j in range(n):
        for sign in (1, -1):
            row = [Fraction(-sign * int(k == j)) for k in range(n)]
            if Q.feasible((), (), A + [row], zeros + [Fraction(-1)]) is not None:
                raise AssertionError(
                    f"the computed {type(s).__name__} is unbounded along "
                    f"predicate variable {j}; the true set is bounded")
    return sorted({tuple(Q.dot(row, a) + ci for row, ci in zip(G, c))
                   for a in Q.vertices(A, b)})


def assert_no_extra(result, true_set, what):
    """A method that claims the exact set (PROTOCOL 3.1, the second
    direction): every vertex of the computed set, enumerated in exact
    arithmetic from its stored rows (:func:`computed_vertices`), is a member
    of the true set ``true_set`` (an :class:`Exact`, decided on its own rows
    by ``rational.Membership``), with no slack. A computed set whose
    predicate polytope is unbounded holds points beyond any bounded true
    set."""
    verts = computed_vertices(result)
    extra = [v for v in verts if not true_set.holds(v)]
    assert not extra, (
        f"{what}: {len(extra)} of {len(verts)} vertices of the result lie "
        f"outside the true set, e.g. {[float(x) for x in extra[0]]}")


def assert_replay(result, points, what):
    """Float points of the true set are members by the LP oracle."""
    sets = result if isinstance(result, list) else [result]
    best = {}
    for k, y in enumerate(points):
        best[k] = min(distance(s, y) for s in sets)
    missed = {k: v for k, v in best.items()
              if v > LP_TOL * max(1.0, float(np.max(np.abs(points[k]))))}
    assert not missed, (
        f"{what}: {len(missed)} of {len(points)} points of the true set are "
        f"outside the result (largest distance {max(missed.values()):.3g})")


def _le(v, x, rel):
    """Is the float ``v`` at most the rational ``x`` (plus ``rel`` of
    max(1, |x|))? NaN is never; -inf always; +inf never."""
    if v is None or math.isnan(v):
        return False
    if math.isinf(v):
        return v < 0
    return Fraction(v) <= x + Fraction(rel) * max(1, abs(x))


def _ge(v, x, rel):
    if v is None or math.isnan(v):
        return False
    if math.isinf(v):
        return v > 0
    return Fraction(v) >= x - Fraction(rel) * max(1, abs(x))


def assert_encloses(lo, hi, exact_lo, exact_hi, what, rel=0.0, tight=None,
                    rows=None):
    """``[lo_i, hi_i]`` contains the exact range of every coordinate (with
    ``rel`` slack); with ``tight`` it also lies within ``tight`` of it.
    ``rows`` limits the check to some coordinates."""
    lo = [None if v is None else float(v)
          for v in np.asarray(lo, dtype=object).reshape(-1)]
    hi = [None if v is None else float(v)
          for v in np.asarray(hi, dtype=object).reshape(-1)]
    if len(lo) != len(exact_lo) or len(hi) != len(exact_hi):
        pytest.fail(f"{what}: {len(lo)} bounds for {len(exact_lo)} entries")
    bad = []
    for i in (range(len(lo)) if rows is None else rows):
        if not (_le(lo[i], exact_lo[i], rel) and _ge(hi[i], exact_hi[i], rel)):
            bad.append((i, "inside", lo[i], hi[i], exact_lo[i], exact_hi[i]))
        elif tight is not None and not (_ge(lo[i], exact_lo[i], tight)
                                        and _le(hi[i], exact_hi[i], tight)):
            bad.append((i, "wider", lo[i], hi[i], exact_lo[i], exact_hi[i]))
    assert not bad, (
        f"{what}: {len(bad)} entries wrong, first: entry {bad[0][0]} "
        f"returned [{bad[0][2]}, {bad[0][3]}], exact range "
        f"[{float(bad[0][4])!r}, {float(bad[0][5])!r}] ({bad[0][1]})")


# --------------------------------------------------------------------------- #
# The input stars                                                               #
# --------------------------------------------------------------------------- #
_BOX_LO, _BOX_HI = flat_box(5, seed=3, zero_width=True)

_ARRAYS = {
    # two binding rows that are not box rows, predicate box not [-1, 1]
    "rows": ([0.5, -0.75, 1.25, 0.0],
             [[1, 0.5, 0], [0, 0.25, -1], [0.5, 0, 0.75], [-0.25, 1, 0.5]],
             [[1, 1, 0.5], [-1, 0.5, 0]], [1, 0.5],
             [-1, -0.5, -1], [1, 1, 0.5]),
    # predicate box excluding 0 on two variables, one binding row
    "offset-box": ([0, 0.25, -0.5],
                   [[1, -0.5, 0.25], [0.5, 1, 0], [0, 0.25, 1]],
                   [[1, 0, 1]], [2], [0.5, -1, 0.25], [1.5, -0.25, 2]),
    # predicate box open above on a0 and below on a2, bounded by the rows
    "half-open-box": ([0.25, -0.5, 0],
                      [[1, 0.5, 0], [0, 1, -0.5], [0.25, 0, 1]],
                      [[1, 1, 1], [1, 0, -1]], [1, 2],
                      [-1, -1, -INF], [INF, 1, 1]),
    # no predicate box: rows only
    "no-box": ([1, -0.25, 0.5],
               [[0.5, 1, 0], [-1, 0.25, 0.5], [0, 0.5, -1]],
               np.vstack([np.eye(3), -np.eye(3), [[1, 1, 1]]]),
               [1, 1, 1.5, 1, 0.5, 0.5, 1], None, None),
    # more predicate variables than dimensions
    "wide": ([0.25, -0.5],
             [[1, 0.5, -0.25, 1], [0, 1, 0.5, -0.5]],
             [[1, 1, 0, 1]], [1.5], [-1] * 4, [1] * 4),
    # one predicate variable and a constant coordinate
    "one-variable": ([0.5, -1, 0.75], [[1], [-0.5], [0]], None, None,
                     [-0.5], [1]),
    # a point star of twelve dimensions (the parallel range path)
    "point": (np.array([3, -5, 7, 1, -2, 9, -11, 4, 6, -8, 10, -3]) / 8.0,
              np.zeros((12, 0)), None, None, None, None),
}


def make(name):
    """``(exact, star)`` of a named case, built fresh (a star caches state
    bounds)."""
    if name == "box-zero-width":
        n = _BOX_LO.size
        exact = Exact.of(np.zeros(n), np.eye(n), lb=_BOX_LO, ub=_BOX_HI)
        return exact, Star.from_bounds(_BOX_LO.reshape(-1, 1),
                                       _BOX_HI.reshape(-1, 1))
    exact = Exact.of(*(_ARRAYS[name] if name in _ARRAYS else _EXTRA[name]))
    return exact, exact.star()


CASES = ["box-zero-width"] + list(_ARRAYS)
SET_CASES = [c for c in CASES if c != "point"]


# --------------------------------------------------------------------------- #
# Construction                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", CASES)
def test_constructed_star_is_the_given_set(name):
    """``Star.__init__`` (and ``Star.from_bounds`` for the box) stores the set
    it is given: every vertex of the true set is a member and nothing beyond
    it is. Catches a constructor that stores one predicate bound for the
    other, drops a row, or loses the generator of a coordinate that follows a
    zero-width one."""
    exact, star = make(name)
    V = exact.vertices()
    assert_holds_exact(star, V, f"{name}: constructed star")
    assert_no_extra(star, exact, f"{name}: constructed star")
    assert_replay(star, exact.points(), f"{name}: constructed star")


def test_from_bounds_holds_every_corner_of_an_inexact_box():
    """``Star.from_bounds`` of a box whose midpoints and half-widths are not
    floats: every corner of the user's box is in the star in exact
    arithmetic, and the star reaches past the box by rounding only. Catches
    a centre or radius rounded inward."""
    lo = np.array([0.1, -0.3, 1 / 3, 7.0, -2.2])
    hi = np.array([0.7, 0.2, 2 / 3, 7.0, 1e-3])
    star = Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    corners = [tuple(Fraction(float(v)) for v in np.where(bits, hi, lo))
               for bits in np.ndindex(*(2,) * lo.size)]
    assert_holds_exact(star, corners, "from_bounds of an inexact box")
    for i in range(lo.size):
        e = np.eye(lo.size)[i]
        assert support(star, e) <= hi[i] + 4e-16 * max(1, abs(hi[i]))
        assert -support(star, -e) >= lo[i] - 4e-16 * max(1, abs(lo[i]))


def test_fresh_from_bounds_is_the_box_with_a_new_unused_variable():
    """``Star.fresh_from_bounds`` is the box [lb, ub] exactly, with one extra
    predicate variable that moves nothing and whose box is [0, t] with t in
    [1, 2). Catches the box collapsed to a corner, the tag bound landing on a
    real variable, and a tag variable that moves the set."""
    lo, hi = _BOX_LO, _BOX_HI
    star = Star.fresh_from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    exact = Exact.of(np.zeros(lo.size), np.eye(lo.size), lb=lo, ub=hi)
    assert_holds_exact(star, exact.vertices(), "fresh_from_bounds")
    assert_no_extra(star, exact, "fresh_from_bounds")
    assert np.all(star.V[:, 1] == 0), "the tag variable has a generator"
    t = float(star.predicate_ub[0, 0])
    assert star.predicate_lb[0, 0] == 0.0 and 1.0 <= t < 2.0


def test_fresh_predicate_tags_differ_between_calls():
    """``fresh_predicate_tag`` returns a value in [1, 2) that no other call
    returns: 64 consecutive calls give 64 distinct tags, and two consecutive
    ``fresh_from_bounds`` stars carry different tags. Catches a tag counter
    that is lost or rounded onto a neighbouring call's tag, which makes the
    graph reach read two unrelated stars as correlated."""
    tags = [fresh_predicate_tag() for _ in range(64)]
    assert all(1.0 <= t < 2.0 for t in tags)
    assert len(set(tags)) == len(tags), "two calls returned the same tag"
    lo, hi = np.zeros((2, 1)), np.ones((2, 1))
    a = Star.fresh_from_bounds(lo, hi)
    b = Star.fresh_from_bounds(lo, hi)
    assert a.predicate_ub[0, 0] != b.predicate_ub[0, 0]


# --------------------------------------------------------------------------- #
# affine_map                                                                    #
# --------------------------------------------------------------------------- #
def _weights(dim, rows, seed):
    rng = np.random.default_rng(seed)
    W = rng.integers(-4, 5, size=(rows, dim)) / 4.0
    W[0, 0] = 1.0                       # never an all-zero first row
    b = rng.integers(-8, 9, size=rows) / 8.0
    return W, b


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("out_rows", [3, 6])
@pytest.mark.parametrize("name", CASES)
def test_affine_map_is_the_exact_image(name, out_rows, bias):
    """``Star.affine_map(W, b)`` with a non-square dyadic W is the image of
    the star exactly: every image of a true vertex is a member and the result
    reaches no further than the image. Catches a bias subtracted, dropped or
    added twice, rows of the constraint system or predicate box replaced,
    and a transposed weight."""
    exact, star = make(name)
    W, b = _weights(exact.dim, out_rows, seed=out_rows)
    image = exact.affine(W, b if bias else None)
    result = star.affine_map(W, b if bias else None)
    V = image.vertices()
    what = f"{name}: affine_map {W.shape}"
    assert_holds_exact(result, V, what)
    assert_no_extra(result, image, what)
    ys = [W @ p + (b if bias else 0) for p in exact.points()]
    assert_replay(result, ys, what)


# --------------------------------------------------------------------------- #
# minkowski_sum                                                                 #
# --------------------------------------------------------------------------- #
# Two stars of dimension 3 with one row each and different right-hand sides:
# the second's row binds at 3/2 and would cut it at 1/2.
_ROWS_A = ([0.5, 0, -0.25], [[1, 0.5], [0, 1], [0.5, -0.5]], [[1, 1]], [0.5],
           [-1, -1], [1, 1])
_ROWS_B = ([-1, 0.75, 0], [[0.5, 0], [1, 1], [0, -1]], [[1, 1]], [1.5],
           [-1, -0.5], [2, 1])


def _pair(left, right):
    def build(spec):
        if isinstance(spec, tuple):
            e = Exact.of(*spec)
            return e, e.star()
        return make(spec)
    return build(left) + build(right)


SUM_PAIRS = {
    "offset-box+half-open-box": ("offset-box", "half-open-box"),
    "half-open-box+offset-box": ("half-open-box", "offset-box"),
    "offset-box+one-variable": ("offset-box", "one-variable"),
    "one-variable+offset-box": ("one-variable", "offset-box"),
    "rows-tight+rows-loose": (_ROWS_A, _ROWS_B),
    "rows-loose+rows-tight": (_ROWS_B, _ROWS_A),
    "no-box+no-box": ("no-box", "no-box"),
}


@pytest.mark.parametrize("pair", list(SUM_PAIRS))
def test_minkowski_sum_is_the_exact_sum(pair):
    """``Star.minkowski_sum`` of two stars with different centres, rows,
    right-hand sides and predicate boxes is their sum exactly. Catches a
    centre taken twice from one operand, the right-hand sides or predicate
    bounds of one operand given to the other's variables (visible only when
    the two differ), and dropped rows."""
    ea, sa, eb, sb = _pair(*SUM_PAIRS[pair])
    result = sa.minkowski_sum(sb)
    V = sum_vertices(ea, eb)
    assert_holds_exact(result, V, f"{pair}: minkowski_sum")
    assert_no_extra(result, exact_sum(ea, eb), f"{pair}: minkowski_sum")
    ys = [p + q for p in ea.points(4) for q in eb.points(4)]
    assert_replay(result, ys, f"{pair}: minkowski_sum")


_HALF_BOXED = ([0.25, 0, -0.5], [[1, 0], [0.5, 1], [0, 0.5]],
               [[1, 1]], [1], [-1, -1], None)


@pytest.mark.parametrize("pair", [
    pytest.param(("offset-box", "no-box"), id="box+no-box", marks=known_bug(
        8, "Star.minkowski_sum drops both predicate boxes when one operand "
           "has none, so the other operand's variables are bounded by its "
           "rows alone and the result is unbounded instead of the sum")),
    pytest.param(("offset-box", _HALF_BOXED), id="box+lower-bounds-only",
                 marks=known_bug(
        8, "Star.minkowski_sum drops both upper predicate bounds when one "
           "operand has none, so the other operand's upper bounds are lost "
           "and the result is larger than the sum")),
    pytest.param(("offset-box", "half-open-box"), id="box+box"),
])
def test_minkowski_sum_keeps_each_operands_predicate_box(pair):
    """The sum of a star with a predicate box and one without (or with lower
    bounds only) is still the sum: each operand keeps its own bounds. The
    last parameter, two stars with full boxes, is the passing neighbour that
    shows the reference right."""
    ea, sa, eb, sb = _pair(*pair)
    result = sa.minkowski_sum(sb)
    V = sum_vertices(ea, eb)
    assert_holds_exact(result, V, "minkowski_sum")
    assert_no_extra(result, exact_sum(ea, eb), "minkowski_sum")


# --------------------------------------------------------------------------- #
# intersect_half_space                                                          #
# --------------------------------------------------------------------------- #
def _cutting_rows(exact, seed):
    """Two dyadic rows ``H`` and right-hand sides ``g`` halfway along the
    exact range of ``H y``, so both rows cut the set."""
    rng = np.random.default_rng(seed)
    H = rng.integers(-4, 5, size=(2, exact.dim)) / 4.0
    H[:, 0] = [1.0, -0.5]
    g = []
    for h in H:
        values = [Q.dot(Q.vector(h), v) for v in exact.vertices()]
        g.append(float((min(values) + max(values)) / 2))
    return H, np.array(g)


def _cut_slack(H, g, star):
    """The outward pad ``intersect_half_space`` may add to a row: the 1e-12
    class of the magnitudes in the row."""
    c, G, C, d, lb, ub = stored_rows(star)
    amax = np.max(np.abs(np.concatenate([lb, ub])), initial=1.0) \
        if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)) else 1.0
    mag = np.abs(H) @ np.abs(c) + np.abs(g) + (np.abs(H) @ np.abs(G)).sum(1) * amax
    return 1e-12 * (1.0 + mag)


def assert_inside_cut(result, exact, H, g, pad, what):
    """The other direction of a cut that claims the exact intersection:
    every vertex of the computed set (:func:`computed_vertices` of its stored
    rows, in exact arithmetic) is a point of the old set (exact membership,
    no slack) and keeps each new row ``H_k y <= g_k`` up to ``pad_k``, the
    outward rounding pad of the new row."""
    verts = computed_vertices(result)
    H = [Q.vector(h) for h in np.asarray(H, dtype=np.float64)]
    g = Q.vector(np.asarray(g, dtype=np.float64).reshape(-1))
    pad = [Fraction(float(p)) for p in pad]
    outside = [v for v in verts if not exact.holds(v)]
    assert not outside, (
        f"{what}: {len(outside)} of {len(verts)} vertices of the result are "
        f"outside the set that was cut, e.g. {[float(x) for x in outside[0]]}")
    crossing = [(k, v) for v in verts for k in range(len(H))
                if Q.dot(H[k], v) > g[k] + pad[k]]
    assert not crossing, (
        f"{what}: a vertex of the result crosses new row {crossing[0][0]} "
        f"by {float(Q.dot(H[crossing[0][0]], crossing[0][1]) - g[crossing[0][0]]):.3g}"
        f" (pad {float(pad[crossing[0][0]]):.3g})")


@pytest.mark.parametrize("name", SET_CASES)
def test_intersect_half_space_is_the_exact_cut(name):
    """``Star.intersect_half_space(H, g)`` with two rows that cut the set and
    a centre off the hyperplane is the intersection: every vertex of the true
    intersection is a member (no slack), and every vertex of the result
    keeps the old rows exactly and the new ones up to the documented
    rounding pad (:func:`assert_inside_cut`). Catches a right-hand side
    without ``- H c``, a row in state space instead of predicate space (seen
    when the generators are not the identity), and new right-hand sides
    paired with old rows."""
    exact, star = make(name)
    H, g = _cutting_rows(exact, seed=len(name))
    result = star.intersect_half_space(H, g)
    cut = exact.cut(H, g)
    V = cut.vertices()
    assert_holds_exact(result, V, f"{name}: intersect_half_space")
    assert_replay(result, cut.points(), f"{name}: intersect_half_space")
    assert_inside_cut(result, exact, H, g, _cut_slack(H, g, star),
                      f"{name}: intersect_half_space")


@pytest.mark.parametrize("offset, empty", [(0.0, False), (-0.25, True)],
                         ids=["touching", "beyond"])
def test_intersect_half_space_of_a_point_star_then_is_empty_set(offset,
                                                                 empty):
    """A point star cut by a half-space through the point (margin 0) is the
    point, not empty; cut by one a quarter beyond it, it is empty. This is
    the cut-then-emptiness check ``verify_specification`` makes; a wrong
    'empty' is a false UNSAT. Catches a pad of the wrong sign and the
    point-star emptiness rule on the boundary."""
    exact, star = make("point")
    H = np.zeros((1, exact.dim))
    H[0, 2], H[0, 5] = 1.0, -0.5
    g = np.array([float(Q.dot(Q.vector(H[0]), exact.q[0])) + offset])
    result = star.intersect_half_space(H, g)
    assert result.is_empty_set() is empty
    if not empty:
        assert_holds_exact(result, exact.vertices(), "touching cut")


# --------------------------------------------------------------------------- #
# convex_hull                                                                   #
# --------------------------------------------------------------------------- #
_SHIFTED = ([0.1, -0.3, 0.7], [[1, -0.5, 0.25], [0.5, 1, 0], [0, 0.25, 1]],
            [[1, 0, 1]], [2], [0.5, -1, 0.25], [1.5, -0.25, 2])

HULL_PAIRS = {
    "offset-box+half-open-box": ("offset-box", "half-open-box"),
    "half-open-box+offset-box": ("half-open-box", "offset-box"),
    "offset-box+no-box": ("offset-box", "no-box"),
    "no-box+offset-box": ("no-box", "offset-box"),
    "one-variable+offset-box": ("one-variable", "offset-box"),
    "inexact-centre+offset-box": (_SHIFTED, "offset-box"),
    "rows-tight+rows-loose": (_ROWS_A, _ROWS_B),
}


@pytest.mark.parametrize("pair", list(HULL_PAIRS))
def test_convex_hull_holds_both_operands(pair):
    """``Star.convex_hull`` holds both operands, in exact arithmetic with no
    slack, for operands whose centres differ (one of them not dyadic) and
    whose predicate boxes are not [-1, 1] (one excludes 0). The result is
    convex, so holding every vertex of both holds the hull. Catches a hull
    anchored at the wrong centre, a lost sign on a predicate-box row, a
    predicate box that keeps a positive lower bound (the operand at
    lambda = 0 cut off), and a dropped rounding slack of the centre
    difference."""
    ea, sa, eb, sb = _pair(*HULL_PAIRS[pair])
    result = sa.convex_hull(sb)
    assert_holds_exact(result, ea.vertices(), f"{pair}: hull, first operand")
    assert_holds_exact(result, eb.vertices(), f"{pair}: hull, second operand")
    mids = [(p + q) / 2 for p in ea.points(3) for q in eb.points(3)]
    assert_replay(result, ea.points() + eb.points() + mids, f"{pair}: hull")


def test_convex_hull_refuses_centres_whose_difference_overflows():
    """``Star.convex_hull`` and ``ImageStar.convex_hull`` of two stars with
    centres 1.7e308 and -1.7e308 refuse with ``ValueError``: the hull is
    built on the difference of the centres, which is not a float, and a hull
    with an infinite generator holds no information. Catches the refusal
    removed (the hull answered with infinite or NaN generators)."""
    def star(c):
        return Star(np.array([[c, 1.0]]), None, None, -np.ones(1), np.ones(1))

    def image(c):
        return ImageStar(np.array([c, 1.0]).reshape(1, 1, 1, 2), None, None,
                         -np.ones(1), np.ones(1), 1, 1, 1)
    with pytest.raises(ValueError):
        star(1.7e308).convex_hull(star(-1.7e308))
    with pytest.raises(ValueError):
        image(1.7e308).convex_hull(image(-1.7e308))


# --------------------------------------------------------------------------- #
# Range queries                                                                 #
# --------------------------------------------------------------------------- #
def _lp_query(star, query):
    n = star.dim
    if query == "get_ranges(parallel=False)":
        return star.get_ranges(parallel=False)
    if query == "get_ranges(parallel=True)":
        return star.get_ranges(parallel=True, n_workers=2)
    if query == "get_ranges()":
        return star.get_ranges()
    if query == "get_range":
        pairs = [star.get_range(i) for i in range(n)]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    if query == "get_min/get_max":
        return ([star.get_min(i) for i in range(n)],
                [star.get_max(i) for i in range(n)])
    if query == "get_box":
        box = star.get_box()
        return box.lb, box.ub
    raise ValueError(query)


LP_QUERIES = ["get_ranges(parallel=False)", "get_ranges(parallel=True)",
              "get_ranges()", "get_range", "get_min/get_max", "get_box"]


@pytest.mark.parametrize("query", LP_QUERIES)
@pytest.mark.parametrize("name", CASES)
def test_lp_range_query_is_the_exact_range(name, query):
    """The LP range queries return the exact range of every coordinate: not
    inside it (pad 1e-12 relative) and not wider than LP error (1e-9
    relative). The star with binding non-box rows has a range strictly
    narrower than its interval estimate, so a query that returns the
    estimate fails the second check. Catches a centre left out of a bound,
    a maximum solved as a minimum, a result read from the wrong LP, the
    last dimension of the parallel path dropped, a one-variable star treated
    as a point, and a point star read from the centre of dimension 0."""
    exact, star = make(name)
    lo, hi = _lp_query(star, query)
    e_lo, e_hi = exact.ranges()
    assert_encloses(lo, hi, e_lo, e_hi, f"{name}: {query}", rel=LP_REL,
                    tight=TIGHT_REL)


_BIG = ([3 * 2.0 ** 39 + 1, -(5 * 2.0 ** 38) - 3, 7 * 2.0 ** 37 + 0.5],
        [[2.0 ** -14, 2.0 ** -15], [-(2.0 ** -15), 2.0 ** -16],
         [2.0 ** -16, -(2.0 ** -17)]],
        np.vstack([np.eye(2), -np.eye(2), [[1, 1]]]), [1, 1, 1, 1, 1],
        None, None)


# A box star (no rows) on non-dyadic data: its exact range is the interval
# estimate itself, so a bound summed in round-to-nearest lands an ulp inside
# it on about half of the twenty bounds.
_INEXACT_RNG = np.random.default_rng(17)
_INEXACT_BOX = (_INEXACT_RNG.standard_normal(10),
                _INEXACT_RNG.standard_normal((10, 4)), None, None,
                [-1.0, 0.1, -0.3, 1 / 3], [1.0, 0.7, 0.2, 2 / 3])

_COLUMN_SCALE = np.array([2.0 ** -30, 1.0, 2.0 ** 30])

# Stars of the range-query cells only (``make`` reads them; they are not in
# CASES, which drives the set-operation cells).
_EXTRA = {
    "no-box-large-centre": _BIG,
    "inexact-box": _INEXACT_BOX,
    # a predicate box given on one side only (the other bound None): that
    # side of every variable is bounded by the rows alone, and variables
    # reach beyond [-1, 1] (a0 up to 4, a2 down to -5)
    "lower-bounds-only": ([-0.5, 1.25, 3],
                          [[2, -0.5, 0], [0.25, 1, -1.5], [0, 0.75, 1]],
                          [[1, 1, 1], [0.5, -1, 0.25]], [1.5, 0.75],
                          [-1, -0.5, -2], None),
    "upper-bounds-only": ([-0.5, 1.25, 3],
                          [[2, -0.5, 0], [0.25, 1, -1.5], [0, 0.75, 1]],
                          [[-1, -1, -1], [1, 0.5, -1]], [2, 1],
                          None, [1, 2, 0.5]),
    # no predicate box and one constant entry (a zero generator row)
    "no-box-constant": ([1, -0.25, 0.75, 0.5],
                        [[0.5, 1, 0], [-1, 0.25, 0.5], [0, 0, 0],
                         [0, 0.5, -1]],
                        np.vstack([np.eye(3), -np.eye(3), [[1, 1, 1]]]),
                        [1, 1, 1.5, 1, 0.5, 0.5, 1], None, None),
    # no predicate box, predicate columns scaled by 2^-30, 1 and 2^30 (the
    # column-scaled family of registry SET-6, here dyadic): a first-order
    # solver stops far inside the range of this star
    "no-box-scaled-columns": (
        [0.5, -0.25, 1.0, 0.0],
        np.array([[0.5, -0.75, 0.25], [1, 0.5, -0.5], [-0.25, 1, 0.75],
                  [0.75, -0.5, 1]]) / _COLUMN_SCALE,
        np.vstack([np.eye(3), -np.eye(3),
                   [[0.5, 0.75, -0.5], [-0.75, 0.25, 0.5]]]) / _COLUMN_SCALE,
        [1, 1, 1, 1, 1, 1, 0.5, 0.625], None, None),
}

ESTIMATE_CASES = CASES + ["no-box-large-centre", "inexact-box",
                          "lower-bounds-only", "upper-bounds-only",
                          "no-box-constant", "no-box-scaled-columns"]
# a star without both predicate bounds returns its LP enclosure and stores
# nothing; a point star has no predicate variable to store bounds over
_NO_BOX = ("no-box", "no-box-large-centre", "point", "lower-bounds-only",
           "upper-bounds-only", "no-box-constant", "no-box-scaled-columns")
_BOUNDED = ("no-box", "no-box-large-centre", "rows", "wide",
            "lower-bounds-only", "upper-bounds-only", "no-box-constant",
            "no-box-scaled-columns")


@pytest.mark.parametrize("name, query", [
    (name, query) for name in ESTIMATE_CASES
    for query in ("estimate_ranges", "estimate_range", "stored state bounds")
    if not (query == "stored state bounds" and name in _NO_BOX)])
def test_estimate_query_encloses_the_exact_range(name, query):
    """``estimate_range`` and ``estimate_ranges`` (and the state bounds
    ``estimate_ranges`` stores) contain the exact range with no slack. On a
    star without a predicate box (or with one side of it missing) they come
    from the padded LP enclosure; with a centre of about 2^40 and generators
    of about 2^-14 a centre added in round-to-nearest lands back inside the
    range. On a box star with non-dyadic centre, generators and predicate box
    the exact range is the interval estimate, so an interval sum rounded to
    nearest lands inside it. An entry whose generator row is all zero is its
    centre exactly, on the interval path and on the LP enclosure alike (both
    document it). Catches an upper bound built from lower bounds, a predicate
    box assumed symmetric, an LP objective taken by position, a centre or
    interval sum added without outward rounding, a star with only one of its
    predicate bounds sent down the interval path (a crash), and a constant
    entry given a padded interval."""
    exact, star = make(name)
    if query == "estimate_ranges":
        lo, hi = star.estimate_ranges()
    elif query == "estimate_range":
        pairs = [star.estimate_range(i) for i in range(star.dim)]
        lo, hi = [p[0] for p in pairs], [p[1] for p in pairs]
    else:
        star.estimate_ranges()
        if star.state_lb is None:
            pytest.fail(f"{name}: estimate_ranges stored no state bounds")
        lo, hi = star.state_lb, star.state_ub
    e_lo, e_hi = exact.ranges()
    assert_encloses(lo, hi, e_lo, e_hi, f"{name}: {query}")
    if name in _BOUNDED:
        finite = np.isfinite(np.asarray(lo, dtype=np.float64)).all() and \
            np.isfinite(np.asarray(hi, dtype=np.float64)).all()
        assert finite, f"{name}: {query} returned an infinite bound for a " \
                       f"bounded star (the check above would be vacuous)"
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    for i in np.flatnonzero(~np.any(star.V[:, 1:] != 0, axis=1)):
        assert lo[i] == hi[i] == star.V[i, 0], (
            f"{name}: {query} gave the constant entry {i} = {star.V[i, 0]!r} "
            f"the interval [{lo[i]!r}, {hi[i]!r}]")


def _poisoning(real, poison, mode):
    """A stand-in for ``solve_lp_batch`` that answers every objective equal
    to one of the rows in ``poison`` with no optimum (``mode='none'``), or
    raises in the single-coordinate call a worker of the parallel path makes
    (``mode='raise'``), and the others with the real solver."""
    def stub(objectives, *args, **kwargs):
        hit = [any(np.array_equal(np.asarray(f, dtype=np.float64).ravel(), r)
                   for r in poison) for f in objectives]
        if mode == "raise" and any(hit) and len(objectives) == 2:
            raise RuntimeError("injected LP failure")
        values = real(objectives, *args, **kwargs)
        return [None if h else v for v, h in zip(values, hit)]
    return stub


@pytest.mark.parametrize("path, mode", [
    ("parallel=False", "none"), ("parallel=True", "none"),
    ("parallel=True", "raise")])
@pytest.mark.parametrize("name", ["rows", "offset-box", "no-box",
                                  "half-open-box"])
def test_get_ranges_falls_back_to_an_enclosure_when_an_lp_fails(
        name, path, mode, monkeypatch):
    """When the LP of some coordinates returns no optimum (or raises in a
    worker of the parallel path), ``get_ranges`` falls back to the interval
    estimate for those coordinates: the fallback must still contain the exact
    range, and the coordinates whose LP worked stay exact. The failure is
    injected for coordinates 1 and 2 only. Catches a fallback that returns
    the lower bounds twice, a failed coordinate left at [0, 0], and a
    fallback written to the wrong coordinates."""
    import n2v.sets.star as star_module
    exact, star = make(name)
    poison = [np.asarray(star.V[i, 1:], dtype=np.float64) for i in (1, 2)]
    monkeypatch.setattr(star_module, "solve_lp_batch", _poisoning(
        star_module.solve_lp_batch, poison, mode))
    if path == "parallel=False":
        lo, hi = star.get_ranges(parallel=False)
    else:
        lo, hi = star.get_ranges(parallel=True, n_workers=2)
    e_lo, e_hi = exact.ranges()
    what = f"{name}: get_ranges({path}) with failed LPs ({mode})"
    assert_encloses(lo, hi, e_lo, e_hi, what, rel=LP_REL)
    others = [i for i in range(exact.dim) if i not in (1, 2)]
    assert_encloses(lo, hi, e_lo, e_hi, what, rel=LP_REL, tight=TIGHT_REL,
                    rows=others)


def test_range_queries_of_an_empty_star_return_no_bound():
    """A star whose rows contradict each other has no range: ``get_min``,
    ``get_max`` and ``get_range`` return None, never a number a caller
    would read as a bound (star.py, the None returns of get_min and
    get_max). Its ``get_ranges`` may return any enclosure."""
    star = Star(np.array([[0.5, 1.0, 0.0], [0.0, 0.5, 1.0]]),
                np.array([[1.0, 1.0]]), np.array([[-3.0]]),
                -np.ones(2), np.ones(2))
    assert star.get_min(0) is None and star.get_max(1) is None
    assert star.get_range(1) == (None, None)


def _unbounded_below():
    """``y0 = a0`` and ``y1 = 1/2 + a0/4 + a1`` over ``a0 <= 1``,
    ``-1 <= a1 <= 1`` (rows only, no predicate box): neither entry has a
    minimum; the maxima are 1 and 7/4."""
    return Star(np.array([[0.0, 1.0, 0.0], [0.5, 0.25, 1.0]]),
                np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
                np.array([[1.0], [1.0], [1.0]]))


_UNBOUNDED_MAX = [Fraction(1), Fraction(7, 4)]


def test_range_queries_of_a_star_unbounded_below_give_no_finite_minimum():
    """The star of :func:`_unbounded_below`: ``get_min`` returns None (or
    -inf) and never a finite number, the maximum of y0 is at least 1, and
    every query that returns bounds for all entries (``get_ranges`` on both
    paths, ``get_box``, ``estimate_ranges``, ``estimate_range``) gives -inf
    as the lower bound of both entries and an upper bound at or above their
    maxima. The star has no predicate box, so the interval queries and the
    fallback of ``get_ranges`` take the LP enclosure, which finds no finite
    predicate box. Catches an enclosure that bounds the star over the box
    [-1, 1] (or any finite box) when its predicate polytope is unbounded."""
    star = _unbounded_below()
    low = star.get_min(0)
    assert low is None or low == -INF, f"finite minimum {low}"
    assert _ge(star.get_max(0), Fraction(1), LP_REL)
    lo, _ = star.get_range(0)
    assert lo is None or lo == -INF
    pairs = lambda: [star.estimate_range(i) for i in range(2)]  # noqa: E731
    queries = {
        "estimate_ranges": star.estimate_ranges,
        "estimate_range": lambda: ([p[0] for p in pairs()],
                                   [p[1] for p in pairs()]),
        "get_ranges(parallel=False)": lambda: _lp_query(
            star, "get_ranges(parallel=False)"),
        "get_ranges(parallel=True)": lambda: _lp_query(
            star, "get_ranges(parallel=True)"),
        "get_box": lambda: _lp_query(star, "get_box"),
    }
    for query, run in queries.items():
        lo, hi = (np.asarray(v, dtype=np.float64).reshape(-1) for v in run())
        assert np.all(lo == -INF), (
            f"{query}: finite lower bound {lo.tolist()} for entries that are "
            f"unbounded below")
        assert all(_ge(h, m, LP_REL) for h, m in zip(hi, _UNBOUNDED_MAX)), (
            f"{query}: upper bounds {hi.tolist()} below the maxima 1 and 7/4")


_UNBOUNDED_FALLBACK = (
    "on a star unbounded below, get_range returns (None, None) although the "
    "maximum exists, and get_ranges replaces the whole entry by its interval "
    "estimate, so the documented exact range has an infinite upper bound "
    "where the maximum is finite")


@pytest.mark.parametrize("query", [
    "get_max",
    pytest.param("get_ranges(parallel=False)", marks=known_bug(
        18, f"{_UNBOUNDED_FALLBACK} (batch path)")),
    pytest.param("get_ranges(parallel=True)", marks=known_bug(
        18, f"{_UNBOUNDED_FALLBACK} (parallel path)")),
])
def test_range_queries_of_a_star_unbounded_below_keep_its_finite_maximum(
        query):
    """``get_ranges`` is documented as the exact range of every entry. On the
    star of :func:`_unbounded_below` the maxima 1 and 7/4 are attained, so
    the upper bounds are those maxima (within the LP error of the exactness
    checks above). ``get_max`` is the passing neighbour on the same star.
    Catches a query that throws away the decided side of an entry when the
    other side has no optimum."""
    star = _unbounded_below()
    if query == "get_max":
        hi = [star.get_max(i) for i in range(2)]
    else:
        _, hi = _lp_query(star, query)
    hi = [None if v is None else float(v)
          for v in np.asarray(hi, dtype=object).reshape(-1)]
    assert all(_ge(h, m, LP_REL) and _le(h, m, TIGHT_REL)
               for h, m in zip(hi, _UNBOUNDED_MAX)), (
        f"{query}: upper bounds {hi}, exact maxima 1 and 7/4")


def _point_cut(offset):
    """The point (1, -2) as a star with no predicate variable, cut by the
    half-space ``y0 + y1/2 <= offset``: through the point for ``offset`` 0,
    a quarter beyond it for -1/4."""
    point = np.array([1.0, -2.0])
    star = Star(point.reshape(-1, 1)).intersect_half_space(
        np.array([[1.0, 0.5]]), np.array([offset]))
    return point, star


_EMPTY_POINT = (
    "{q} of a star with no predicate variable answers its centre without "
    "reading its rows, so an empty point star (a point cut off by a "
    "half-space) gets a number as its range instead of None")


@pytest.mark.parametrize("query, cut", [
    pytest.param("get_range", "beyond", marks=known_bug(
        8, _EMPTY_POINT.format(q="get_range"))),
    pytest.param("get_min/get_max", "beyond", marks=known_bug(
        8, _EMPTY_POINT.format(q="get_min and get_max (via _solve_lp)"))),
    ("get_range", "touching"),
    ("get_min/get_max", "touching"),
])
def test_range_queries_of_a_point_star_cut_by_a_half_space(query, cut):
    """A point star cut by a half-space a quarter beyond the point (what
    ``intersect_half_space`` returns in ``verify_specification``) is empty:
    its one row reads ``0 <= d`` with ``d < 0``, decided here in Fractions.
    It has no range, so ``get_min`` and ``get_max`` return None (their
    docstrings: "None if infeasible") and ``get_range`` returns
    ``(None, None)``, as for every other empty star
    (:func:`test_range_queries_of_an_empty_star_return_no_bound`). Cut through
    the point it is the point: each query returns the centre exactly (the
    passing neighbours). Catches a query that answers a point star from its
    centre without reading its rows."""
    point, star = _point_cut(0.0 if cut == "touching" else -0.25)
    d = [Fraction(float(v)) for v in np.asarray(star.d).reshape(-1)]
    if star.nVar != 0 or any(v < 0 for v in d) is not (cut == "beyond"):
        pytest.fail(f"test data: the {cut} cut is not the expected point "
                    f"star (nVar {star.nVar}, rows 0 <= {d})")
    if query == "get_range":
        got = [star.get_range(i) for i in range(2)]
    else:
        got = [(star.get_min(i), star.get_max(i)) for i in range(2)]
    expected = ([(None, None)] * 2 if cut == "beyond"
                else [(point[i], point[i]) for i in range(2)])
    assert got == expected, f"{cut} cut: {query} returned {got}"


@pytest.mark.parametrize("query", ["get_range", "get_min", "get_max"])
def test_range_query_refuses_an_index_outside_the_star(query):
    """``get_range``, ``get_min`` and ``get_max`` refuse an index below 0 and
    one at the dimension with ``ValueError`` (the index check of each).
    Catches a check that lets a negative index through, where numpy wraps it
    and the query answers with the range of the last entry, and an
    off-by-one that lets the index ``dim`` through to an ``IndexError``."""
    _, star = make("rows")
    for index in (-1, star.dim):
        with pytest.raises(ValueError):
            getattr(star, query)(index)


def test_range_queries_refuse_an_unknown_solver_name():
    """An ``lp_solver`` name that names no solver is refused with
    ``ValueError`` by every LP range query, on both paths of ``get_ranges``
    and by ``get_box``, before any LP is solved. Catches a query that hands
    the name on unchecked: the worker of the parallel path swallows every
    exception and the batch path reads a failed solve as "no optimum", so an
    unread name would silently give interval estimates (or a crash) instead
    of the refusal."""
    _, star = make("rows")
    queries = {
        "get_range": lambda: star.get_range(0, "no-such-solver"),
        "get_min": lambda: star.get_min(0, "no-such-solver"),
        "get_max": lambda: star.get_max(0, "no-such-solver"),
        "get_ranges(parallel=False)": lambda: star.get_ranges(
            lp_solver="no-such-solver", parallel=False),
        "get_ranges(parallel=True)": lambda: star.get_ranges(
            lp_solver="no-such-solver", parallel=True, n_workers=2),
        "get_box": lambda: star.get_box(lp_solver="no-such-solver"),
    }
    for query, run in queries.items():
        with pytest.raises(ValueError):
            run()
            pytest.fail(f"{query} answered for an unknown solver name")


# Every HiGHS value of ``lp_solver`` the range queries accept, in several
# spellings: the highspy batch and single-LP paths ('linprog', 'HIGHS'), the
# scipy paths ('highs-ds', LPSolver.HIGHS_IPM), None, and the global setting
# (``config.lp_solver``) with the argument left out.
_HIGHS_VALUES = {"linprog": "linprog", "HIGHS": "HIGHS",
                 "highs-ds": "highs-ds", "LPSolver.HIGHS_IPM":
                 LPSolver.HIGHS_IPM, "None": None,
                 "global-highs-ipm": "global"}


def _recording(real, seen):
    """``real`` (``solve_lp`` or ``solve_lp_batch``), recording the solver
    each call is given, resolved as the solver resolves it."""
    def wrapper(*args, **kwargs):
        seen.append(resolve_solver(kwargs.get("lp_solver"),
                                   allow_sentinel=False))
        return real(*args, **kwargs)
    return wrapper


@pytest.mark.parametrize("solver", list(_HIGHS_VALUES))
@pytest.mark.parametrize("name", ["rows", "half-open-box"])
def test_lp_range_query_with_each_highs_solver_is_the_exact_range(
        name, solver, monkeypatch):
    """Every range query (``get_range``, ``get_min``/``get_max``,
    ``get_ranges`` on the batch and the parallel path, ``get_box``) with each
    HiGHS value of ``lp_solver`` returns the exact range (pad 1e-12, LP error
    1e-9 relative, as above), and every LP it solves is given the solver the
    caller named: each call of ``solve_lp`` and ``solve_lp_batch`` is
    recorded. The stars: binding rows over a finite box (the certified
    path) and a box open on one side (no certificate). The CVXPY solvers stop
    inside the range and are marked in ``test_soundness_range_rounding.py``
    (registry SET-6). Catches a query that drops the caller's solver on one
    path (``get_box``, the batch path, a worker of the parallel path) and
    solves with the global default instead."""
    import n2v.sets.star as star_module
    seen = []
    monkeypatch.setattr(star_module, "solve_lp_batch",
                        _recording(star_module.solve_lp_batch, seen))
    monkeypatch.setattr(star_module, "solve_lp",
                        _recording(star_module.solve_lp, seen))
    exact, star = make(name)
    e_lo, e_hi = exact.ranges()
    n = star.dim
    value = _HIGHS_VALUES[solver]
    try:
        if value == "global":
            config.lp_solver = "highs-ipm"
            value = None
        expected = resolve_solver(value, allow_sentinel=False)
        queries = {
            "get_range": lambda: tuple(zip(*[star.get_range(i, value)
                                             for i in range(n)])),
            "get_min/get_max": lambda: ([star.get_min(i, value)
                                         for i in range(n)],
                                        [star.get_max(i, value)
                                         for i in range(n)]),
            "get_ranges(parallel=False)": lambda: star.get_ranges(
                lp_solver=value, parallel=False),
            "get_ranges(parallel=True)": lambda: star.get_ranges(
                lp_solver=value, parallel=True, n_workers=2),
            "get_box": lambda: (lambda box: (box.lb, box.ub))(
                star.get_box(lp_solver=value)),
        }
        for query, run in queries.items():
            seen.clear()
            lo, hi = run()
            what = f"{name}: {query} with lp_solver={solver}"
            if not seen:
                pytest.fail(f"{what}: no LP was solved")
            other = sorted({s.value for s in seen if s is not expected})
            assert not other, (f"{what}: LPs solved with {other} instead of "
                               f"{expected.value}")
            assert_encloses(lo, hi, e_lo, e_hi, what, rel=LP_REL,
                            tight=TIGHT_REL)
    finally:
        config.reset()


@pytest.mark.parametrize("name", ["rows", "no-box"])
def test_range_queries_ignore_stale_stored_state_bounds(name):
    """A star can carry state bounds that do not hold its entries: given to
    the constructor, kept by an operation that changed the set, or handed on
    in the wrong order (``ImageStar.flatten_to_star``, registry SET-9). No
    range query reads them: on a star whose stored bounds are its centre
    alone, ``estimate_ranges``, ``estimate_range``, ``get_ranges`` (both
    paths) and ``get_box`` still contain the exact range, and
    ``estimate_ranges`` of a star with a predicate box replaces the stored
    bounds by bounds that hold. Catches a query that returns stored state
    bounds as a cache."""
    exact, fresh = make(name)
    centre = fresh.V[:, :1].copy()
    star = Star(fresh.V, fresh.C, fresh.d, fresh.predicate_lb,
                fresh.predicate_ub, state_lb=centre, state_ub=centre)
    e_lo, e_hi = exact.ranges()
    if all(lo == hi for lo, hi in zip(e_lo, e_hi)):
        pytest.fail("test data: the stored centre would be the exact range")
    lo, hi = star.estimate_ranges()
    assert_encloses(lo, hi, e_lo, e_hi, f"{name}: estimate_ranges")
    pairs = [star.estimate_range(i) for i in range(star.dim)]
    assert_encloses([p[0] for p in pairs], [p[1] for p in pairs], e_lo, e_hi,
                    f"{name}: estimate_range")
    if star.predicate_lb is not None:
        assert_encloses(star.state_lb, star.state_ub, e_lo, e_hi,
                        f"{name}: state bounds after estimate_ranges")
    for query in ("get_ranges(parallel=False)", "get_ranges(parallel=True)",
                  "get_box"):
        lo, hi = _lp_query(Star(fresh.V, fresh.C, fresh.d, fresh.predicate_lb,
                                fresh.predicate_ub, state_lb=centre,
                                state_ub=centre), query)
        assert_encloses(lo, hi, e_lo, e_hi, f"{name}: {query}", rel=LP_REL)


@pytest.mark.parametrize("query", ["estimate_ranges", "estimate_range"])
def test_estimate_query_without_a_box_solves_on_highs_under_a_global_cvxpy_solver(
        query):
    """``estimate_range(s)`` of a star without a predicate box is an
    enclosure, so its LPs run on HiGHS and are padded whatever the global
    solver is (``_lp_enclosure`` through ``_bound_lp_solver``). With the
    global solver set to SCS, the bounds of a star whose predicate columns
    are scaled by 2^-30 and 2^30 still contain the exact range with no slack.
    Catches the enclosure LPs handed to the global solver: SCS stops inside
    the range of this star by up to about 1.2."""
    exact, star = make("no-box-scaled-columns")
    try:
        config.lp_solver = "SCS"
        if query == "estimate_ranges":
            lo, hi = star.estimate_ranges()
        else:
            pairs = [star.estimate_range(i) for i in range(star.dim)]
            lo, hi = [p[0] for p in pairs], [p[1] for p in pairs]
    finally:
        config.reset()
    e_lo, e_hi = exact.ranges()
    assert_encloses(lo, hi, e_lo, e_hi, f"{query} under lp_solver SCS")


# --------------------------------------------------------------------------- #
# is_empty_set                                                                  #
# --------------------------------------------------------------------------- #
_EIGHTH = 2.0 ** -19


def _two_var(C, d, lb=(-1, -1), ub=(1, 1), scale=1.0):
    return (np.zeros(2), scale * np.eye(2), C, d,
            None if lb is None else list(lb), None if ub is None else list(ub))


EMPTINESS = {
    "rows-not-empty": ("rows", False),
    "empty-by-rows": (_two_var([[1, 1]], [-2.25]), True),
    "empty-by-rows-no-box": (_two_var(
        [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]], [1, 1, 1, 1, -2.25],
        lb=None, ub=None), True),
    "empty-by-predicate-box": (_two_var(None, None, lb=(1, 0), ub=(0, 1)),
                               True),
    "sliver-not-empty": (_two_var([[1, 1]], [-2 + _EIGHTH], scale=2.0 ** 20),
                         False),
    "thin-band-not-empty": (_two_var([[1, -1], [-1, 1]],
                                     [-1 + _EIGHTH, 1]), False),
    "row-excludes-lower-corner": (_two_var([[-1, -1]], [1]), False),
    "point-on-boundary": ((np.array([1.0, -2.0]), np.zeros((2, 0)),
                           np.zeros((2, 0)), [0.0, 0.25], None, None), False),
    "point-cut-off": ((np.array([1.0, -2.0]), np.zeros((2, 0)),
                       np.zeros((1, 0)), [-0.25], None, None), True),
    "point-no-rows": ((np.array([1.0, -2.0]), np.zeros((2, 0)), None, None,
                       None, None), False),
}


@pytest.mark.parametrize("case", list(EMPTINESS))
def test_is_empty_set_answers_as_the_exact_feasibility(case):
    """``Star.is_empty_set`` is True exactly when the predicate polytope has
    no point, decided here in rationals. A wrong True is a false UNSAT: a
    sliver of width about 2e-6 (with generators of 2^20) and a thin band
    away from the origin are not empty, a point on the boundary of its row
    is not empty, a row that only removes the lower predicate corner leaves
    the set non-empty. Catches a feasibility answer returned without
    negation, the LP given the lower bounds as upper bounds, and a tolerance
    of the wrong sign on the point-star rule."""
    spec, expected = EMPTINESS[case]
    exact = make(spec)[0] if isinstance(spec, str) else Exact.of(*spec)
    star = exact.star()
    A = [list(r) for r in exact.q[2]]
    b = list(exact.q[3])
    for j in range(exact.nvar):
        unit = [Fraction(int(k == j)) for k in range(exact.nvar)]
        if exact.q[5][j] is not None:
            A.append(unit)
            b.append(exact.q[5][j])
        if exact.q[4][j] is not None:
            A.append([-v for v in unit])
            b.append(-exact.q[4][j])
    if exact.nvar:
        truth = Q.feasible((), (), A, b) is None
    else:
        truth = any(v < 0 for v in b)
    if truth is not expected:
        pytest.fail(f"test data: case {case} is labelled empty={expected}")
    assert star.is_empty_set() is expected, case


# --------------------------------------------------------------------------- #
# contains                                                                      #
# --------------------------------------------------------------------------- #
def _probe_points(exact):
    """Points with a known answer: vertices and midpoints (members) and
    points pushed out of the set from a vertex (non-members by a margin)."""
    V = exact.vertices()
    centre = [sum(v[i] for v in V) / len(V) for i in range(exact.dim)]
    inside = [np.array([float(x) for x in v]) for v in V[:6]]
    inside += [np.array([float((a + b) / 2) for a, b in
                         zip(V[k % len(V)], V[-1 - k % len(V)])])
               for k in range(3)]
    outside = []
    for v in V[:6]:
        y = [x + (x - m) / 4 for x, m in zip(v, centre)]
        y = np.array([float(t) for t in y])
        if not exact.holds(Q.vector(y)):
            outside.append(y)
    return inside, outside


def _row_probes(exact):
    """Points that break a constraint row and nothing else: for each row the
    corner of the predicate box that maximises it, kept when it breaks the
    row, mapped through the basis. Each is checked, in exact arithmetic, to
    lie outside the set and inside the image of the predicate box alone, so
    only the rows can reject it. Empty for a star without rows or without a
    finite predicate box."""
    c, G, C, d, lb, ub = exact.q
    if not C or any(v is None for v in list(lb) + list(ub)):
        return []
    box_only = Exact((c, G, [], [], lb, ub))
    out = []
    for row, rhs in zip(C, d):
        a = [hi if w > 0 else lo for w, lo, hi in zip(row, lb, ub)]
        if Q.dot(row, a) <= rhs:
            continue
        y = np.array([float(Q.dot(g, a) + ci) for g, ci in zip(G, c)])
        if exact.holds(Q.vector(y)) or not box_only.holds(Q.vector(y)):
            pytest.fail("test data: a row probe is not outside the rows only")
        out.append(y)
    return out


@pytest.mark.parametrize("method", ["lp", "algebraic"])
@pytest.mark.parametrize("name", ["rows", "offset-box", "no-box",
                                  "one-variable"])
def test_contains_answers_by_the_rows(name, method):
    """``Star.contains`` accepts points of the set, including vertices, and
    rejects points beyond a vertex and, for the stars with a predicate box
    and constraint rows ('rows', 'offset-box'), points that break one
    constraint row while lying inside the image of the predicate box
    (:func:`_row_probes`); one point at a time and as a batch. Catches a
    containment test against the bounding box instead of the rows, and one
    that drops the constraint rows when the star has a predicate box (the
    points beyond a vertex alone do not see it: none of them lies in the
    image of the predicate box)."""
    exact, star = make(name)
    inside, outside = _probe_points(exact)
    if not outside:
        pytest.fail("test data: no point outside the set")
    row_only = _row_probes(exact)
    if name in ("rows", "offset-box") and not row_only:
        pytest.fail("test data: no point that breaks a row only")
    outside = outside + row_only
    for y in inside:
        assert star.contains(y, method=method), y
    for y in outside:
        assert distance(star, y) > 1e-6
        assert not star.contains(y, method=method), y
    batch = star.contains(np.vstack(inside + outside), method=method)
    assert batch.tolist() == [True] * len(inside) + [False] * len(outside)


def test_contains_algebraic_refuses_a_wide_basis():
    """The algebraic containment test needs a basis of full column rank; a
    star with more predicate variables than dimensions must refuse it."""
    _, star = make("wide")
    with pytest.raises(ValueError):
        star.contains(np.zeros(2), method="algebraic")
    assert star.contains(np.array([0.25, -0.5]), method="lp")


# --------------------------------------------------------------------------- #
# sample                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["offset-box", "no-box", "wide",
                                  "box-zero-width"])
def test_sample_returns_points_of_the_set(name):
    """``Star.sample(N)`` returns at most N points, each a member of the set
    by the LP oracle and, in exact arithmetic, within 1e-9 of the true set.
    The stars here fill a large part of their bounding box, so rejection
    sampling from the box finds at least one point. Catches a sampler that
    skips the rejection by the rows (it returns points of the bounding box
    that the rows cut off) or draws from the wrong box. A star of lower
    dimension than its space (the 'rows' star) gets no point from rejection
    sampling, which the docstring allows (``k <= N``), so it has no cell."""
    exact, star = make(name)
    pts = star.sample(40)
    assert pts.shape[0] == star.dim and 1 <= pts.shape[1] <= 40
    bad = uncovered([star], list(pts.T))
    assert not bad, f"{name}: {len(bad)} samples outside the star"
    outside_exact = [k for k, y in enumerate(pts.T)
                     if not exact.holds(Q.vector(y))]
    far = [k for k in outside_exact if distance(star, pts[:, k]) > 1e-9]
    assert not far


# --------------------------------------------------------------------------- #
# get_vertices                                                                  #
# --------------------------------------------------------------------------- #
def _extreme_points_2d(points):
    """The vertices of the convex hull of rational 2-D points (Andrew's
    monotone chain in exact arithmetic; collinear points are dropped)."""
    pts = sorted(set(tuple(p) for p in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def chain(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out
    lower, upper = chain(pts), chain(reversed(pts))
    return lower[:-1] + upper[:-1]


def _close_to(points, targets, tol=1e-9):
    return all(any(np.max(np.abs(p - t)) <= tol * (1 + np.max(np.abs(t)))
                   for t in targets) for p in points)


def test_get_vertices_of_a_polygon_are_its_vertices():
    """``get_vertices`` of a 2-D star with a cut corner returns points that are
    vertices of the set, and every vertex (each has a normal cone wider than
    the 64 directions' spacing). Catches a projection or an LP direction of
    the wrong sign, and returned points outside the set."""
    exact = Exact.of([0.5, 0.25], [[1, 0.5], [0, 1]], [[1, 1]], [1],
                     [-1, -1], [1, 1])
    star = exact.star()
    got = star.get_vertices()
    true = [np.array([float(x) for x in v]) for v in exact.vertices()]
    assert got is not None and got.shape[1] == 2
    assert _close_to(got, true), "a returned point is not a vertex"
    assert _close_to(true, got), "a vertex was not returned"


def test_get_vertices_of_a_box_in_three_dimensions_and_a_projection():
    """``get_vertices`` of a 3-D box star finds its 8 corners (directions on
    the sphere reach every octant); the projection onto coordinates (0, 2)
    of the star with rows returns vertices of the projected polygon only.
    Catches a projection onto the wrong coordinates."""
    lo, hi = np.array([-1.0, 0.5, -0.25]), np.array([0.5, 1.5, 2.0])
    box = Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    got = box.get_vertices()
    corners = [np.where(bits, hi, lo) for bits in np.ndindex(2, 2, 2)]
    assert got is not None and _close_to(got, corners) and \
        _close_to(corners, got)
    exact, star = make("offset-box")
    proj = star.get_vertices(projection=[0, 2])
    # The vertices of the projected polygon are among the projections of the
    # true vertices, so a returned point close to one of them is in the
    # projection and is a vertex of it.
    flat = [np.array([float(v[0]), float(v[2])]) for v in exact.vertices()]
    assert proj is not None and proj.shape[1] == 2
    assert _close_to(proj, flat), "a returned point is not a projected vertex"
    hull = _extreme_points_2d(exact.affine(np.eye(3)[[0, 2]]).vertices())
    assert _close_to([np.array([float(x) for x in v]) for v in hull], proj), \
        "a vertex of the projected polygon was not returned"


def test_get_vertices_degenerate_cases():
    """A point star returns its centre; an empty star returns None; a star of
    dimension 4 without a projection, or a projection onto one coordinate,
    is refused."""
    exact, star = make("point")
    two = Star(star.V[:2])
    got = two.get_vertices()
    assert got.shape == (1, 2) and np.array_equal(got[0], star.V[:2, 0])
    empty = Exact.of(*_two_var([[1, 1]], [-2.25])).star()
    assert empty.get_vertices() is None
    _, rows = make("rows")
    with pytest.raises(ValueError):
        rows.get_vertices()
    with pytest.raises(ValueError):
        rows.get_vertices(projection=[1])


# --------------------------------------------------------------------------- #
# to_image_star                                                                 #
# --------------------------------------------------------------------------- #
def test_to_image_star_reads_the_flat_order_as_height_width_channel():
    """``Star.to_image_star(H, W, C)`` of a flat star in height, width,
    channel order stores height H and width W and puts flat entry
    (h W + w) C + c at pixel (h, w, c), for H != W and two channels. Catches
    height and width passed in the wrong order. A dimension that does not
    match is refused."""
    H, W, C = 3, 5, 2
    rng = np.random.default_rng(1)
    V = rng.integers(-8, 9, size=(H * W * C, 4)) / 8.0
    star = Star(V, None, None, -np.ones(3), np.ones(3))
    image = star.to_image_star(H, W, C)
    assert (image.height, image.width, image.num_channels) == (H, W, C)
    for h, w, c in np.ndindex(H, W, C):
        assert np.array_equal(image.V[h, w, c], V[(h * W + w) * C + c])
    assert np.array_equal(image.predicate_ub, star.predicate_ub)
    with pytest.raises(ValueError):
        star.to_image_star(H, W, C + 1)


def _relu_pool():
    return nn.Sequential(nn.ReLU(), nn.AvgPool2d((2, 3), stride=1))


@pytest.mark.parametrize("entry", ["reach_layer", "reach_model"])
def test_relu_then_pooling_on_a_wide_two_channel_image(entry):
    """Approx ReLU on a (2, 3, 5) ImageStar (height 3, width 5) works on the
    flat star and converts back with ``Star.to_image_star``; an asymmetric
    AvgPool2d after it pools the pixels the conversion recorded. Three input
    sets cut along one pixel. Every float64 output of the true model is in a
    result, with the result's stored shape checked. Catches height and width
    swapped in the conversion."""
    lo, hi = pixel_box(2, 3, 5, seed=2)
    sets = image_split_cover("ImageStar", lo, hi, parts=3)
    model = _relu_pool()
    if entry == "reach_layer":
        mid = reach_one_layer(nn.ReLU(), sets, "approx", "relu")
        out = reach_one_layer(nn.AvgPool2d((2, 3), stride=1), mid, "approx",
                              "avgpool")
    else:
        out = reach_model(model, sets, "approx", "relu then avgpool")
    ys = replay_torch(model, samples(lo, hi), (1, 2, 3, 5))
    assert_covered(out, ys, f"relu then avgpool via {entry}",
                   out_chw=(2, 2, 3))


def test_exact_relu_then_pooling_on_a_small_tall_image():
    """Exact ReLU on a (2, 3, 2) ImageStar with a few pixels straddling zero,
    then AvgPool2d((2, 1)): every output is in some returned image set of
    stored shape (2, 2, 2) channels first. Catches the conversion of each
    exact piece back to an image with height and width swapped."""
    lo, hi = pixel_box(2, 3, 2, seed=5)
    lo, hi = lo + 0.75, hi + 0.75
    model = nn.Sequential(nn.ReLU(), nn.AvgPool2d((2, 1), stride=1))
    out = reach_model(model, image_set("ImageStar", lo, hi), "exact",
                      "exact relu then avgpool")
    ys = replay_torch(model, samples(lo, hi), (1, 2, 3, 2))
    assert_covered(out, ys, "exact relu then avgpool", out_chw=(2, 2, 2))


# --------------------------------------------------------------------------- #
# The range queries read by the ReLU kernels (the public route)                 #
# --------------------------------------------------------------------------- #
_RELU_W1 = np.array([[1, -0.5, 0.25, 0.5], [-0.75, 1, 0, 0.25],
                     [0.5, 0.5, -1, 0]])
_RELU_B1 = np.array([0.25, -0.5, 0.125])
_RELU_W2 = np.array([[1, -1, 0.5], [0.25, 0.5, 1]])
_RELU_B2 = np.array([-0.25, 0.5])


def _member_points(exact, n, seed):
    """``n`` points of the exact set: its vertices, then convex combinations
    of three vertices with weights in eighths (dyadic when the set is)."""
    V = exact.vertices()
    rng = np.random.default_rng(seed)
    points = [np.array([float(x) for x in v]) for v in V]
    while len(points) < n:
        k = rng.choice(len(V), size=3)
        w = rng.integers(0, 8, size=2)
        w = [Fraction(int(w[0]), 8), Fraction(int(w[1]), 8)]
        w.append(1 - w[0] - w[1])
        if w[2] < 0:
            continue
        points.append(np.array([float(sum(wj * V[kj][i]
                                          for wj, kj in zip(w, k)))
                                for i in range(exact.dim)]))
    return points[:n]


@pytest.mark.parametrize("method", ["approx-range", "exact"])
@pytest.mark.parametrize("name, entry", [("rows", "reach_model"),
                                         ("no-box", "reach_layer")])
def test_relu_on_a_star_with_binding_rows_holds_the_exact_image(name, entry,
                                                                method):
    """Linear, ReLU, Linear (dyadic) on a star whose rows bind. The approx
    ReLU with ``relax_method='range'`` and ``relax_factor=0.1`` classifies
    the neurons by ``estimate_ranges`` and then takes the phase and the
    triangle of every crossing neuron from ``get_max`` and ``get_min`` (the
    default relaxation reads ``estimate_ranges`` alone); the exact ReLU
    decides each phase by ``get_range``. On the 'rows' star the LP range of
    neuron 1 is [-47/16, -13/64] while its interval estimate reaches 3/16,
    so the kernel's answer rests on the LP ranges; the other two neurons
    cross zero. The 'rows' star goes through the model entry (which first
    makes the predicate box finite), the 'no-box' star through
    ``reach_layer`` (the kernel reads a star without a predicate box: the LP
    enclosure and uncertified LP ranges). Every vertex of the exact image, in
    rationals, is a member of a returned set (no slack), and 128 member
    points replayed in float64 are members by the LP oracle. Catches a range
    query, read through the kernel, that returns a maximum below the true
    one (a crossing neuron zeroed) or a minimum above it."""
    exact, star = make(name)
    W1 = _RELU_W1[:, :exact.dim]
    what = f"{name}: Linear-ReLU-Linear ({method}) via {entry}"
    options = ({"relax_method": "range", "relax_factor": 0.1}
               if method == "approx-range" else {})
    method = "approx" if method == "approx-range" else method
    if entry == "reach_model":
        first, second = nn.Linear(exact.dim, 3), nn.Linear(3, 2)
        with torch.no_grad():
            for layer, (W, b) in ((first, (W1, _RELU_B1)),
                                  (second, (_RELU_W2, _RELU_B2))):
                layer.weight.copy_(torch.tensor(W, dtype=torch.float32))
                layer.bias.copy_(torch.tensor(b, dtype=torch.float32))
        model = nn.Sequential(first, nn.ReLU(), second).requires_grad_(False)
        sets = reach_model(model, star, method, what, **options)
    else:
        mid = reach_one_layer(nn.ReLU(), star.affine_map(W1, _RELU_B1),
                              method, what, **options)
        sets = [s.affine_map(_RELU_W2, _RELU_B2) for s in mid]
    A, b = Q._predicate_rows(exact.q)
    pieces = Q.affine(Q.polytope(A, b), exact.q[1], exact.q[0])
    pieces = Q.affine(Q.relu(Q.affine(pieces, W1, _RELU_B1)),
                      _RELU_W2, _RELU_B2)
    vertices = sorted({v for piece in pieces for v in piece.image()})
    assert_holds_exact(sets, vertices, what)
    xs = _member_points(exact, 128, seed=len(name))
    ys = [_RELU_W2 @ np.maximum(W1 @ x + _RELU_B1, 0) + _RELU_B2 for x in xs]
    assert_covered(sets, ys, what)


# --------------------------------------------------------------------------- #
# __repr__                                                                      #
# --------------------------------------------------------------------------- #
def test_repr_names_dimension_variables_and_rows():
    """``repr`` of a star states its dimension, number of predicate variables
    and number of constraint rows."""
    _, star = make("rows")
    text = repr(star)
    assert "dim=4" in text and "nVar=3" in text and "nConstraints=2" in text


# --------------------------------------------------------------------------- #
# ensure_finite_predicate_box                                                   #
# --------------------------------------------------------------------------- #
def _predicate_ranges(exact):
    """Exact least and greatest value of every predicate variable over the
    predicate polytope."""
    n = exact.nvar
    unit = Exact((
        [Fraction(0)] * n,
        [[Fraction(int(i == j)) for j in range(n)] for i in range(n)],
        exact.q[2], exact.q[3], exact.q[4], exact.q[5]))
    return unit.ranges()


@pytest.mark.parametrize("name", ["no-box", "half-open-box", "rows",
                                  "low-variable", "lower-bounds-only",
                                  "upper-bounds-only"])
def test_ensure_finite_predicate_box_keeps_the_set(name):
    """``ensure_finite_predicate_box`` returns a star with a finite predicate
    box that holds every feasible predicate point (no slack) and is the same
    set, both directions. A finite box is returned unchanged. The
    'low-variable' star has no predicate box and a variable that reaches -3,
    below the -1 a normalised star would assume; 'lower-bounds-only' and
    'upper-bounds-only' have finite bounds on one side and None on the other,
    which the rows bound. Catches a missing lower bound read as -1, the bound
    LP of every variable solved with the objective of variable 0, finite
    entries overwritten, and an early return when the bounds of one side
    alone are finite."""
    if name == "low-variable":
        exact = Exact.of([0.5, -0.25], [[1, 0.5], [0.25, -1]],
                         [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]],
                         [1, 3, 2, 0.5, 1.5], None, None)
    else:
        exact = make(name)[0]
    star = exact.star()
    star2, empty = ensure_finite_predicate_box(star)
    assert empty is False
    if name == "rows":
        assert star2 is star
        return
    lb = star2.predicate_lb.reshape(-1)
    ub = star2.predicate_ub.reshape(-1)
    assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))
    a_lo, a_hi = _predicate_ranges(exact)
    assert_encloses(lb, ub, a_lo, a_hi, f"{name}: finite predicate box")
    assert_holds_exact(star2, exact.vertices(), f"{name}: finite box star")
    assert_no_extra(star2, exact, f"{name}: finite box star")


def test_ensure_finite_predicate_box_reports_empty_only_when_empty():
    """A star without a box whose rows contradict each other by a quarter is
    reported empty (the same star and True); a sliver of width about 2e-6 is
    not, and its box holds the sliver. A wrong 'empty' makes reach return
    nothing, a false UNSAT."""
    empty = Exact.of(*_two_var([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]],
                               [1, 1, 1, 1, -2.25], lb=None, ub=None)).star()
    star2, is_empty = ensure_finite_predicate_box(empty)
    assert is_empty is True and star2 is empty
    sliver = Exact.of(*_two_var([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]],
                                [1, 1, 1, 1, -2 + _EIGHTH], lb=None, ub=None))
    star2, is_empty = ensure_finite_predicate_box(sliver.star())
    assert is_empty is False
    assert_holds_exact(star2, sliver.vertices(), "sliver after finite box")


def test_ensure_finite_predicate_box_refuses_what_it_cannot_bound():
    """A predicate variable unbounded over the polytope, a NaN in the box and
    a NaN in the rows are refused with ``ValueError``, never a box."""
    open_star = Star(np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0]]),
                     np.array([[1.0]]))
    with pytest.raises(ValueError):
        ensure_finite_predicate_box(open_star)
    nan_box = Star(np.array([[0.0, 1.0]]), None, None, np.array([np.nan]),
                   np.array([1.0]))
    with pytest.raises(ValueError):
        ensure_finite_predicate_box(nan_box)
    nan_rows = Star(np.array([[0.0, 1.0]]), np.array([[np.nan]]),
                    np.array([[1.0]]), None, None)
    with pytest.raises(ValueError):
        ensure_finite_predicate_box(nan_rows)


@pytest.mark.parametrize("status", ["error", "infeasible"])
def test_ensure_finite_predicate_box_refuses_when_both_solves_fail(
        status, monkeypatch):
    """When both bound LPs of a variable fail (an error status), or both
    report the polytope infeasible while it is not, no trustworthy box exists:
    the helper must raise ``ValueError`` and never return a box or an
    emptiness report (predicate_box.py, the error after two failed solves;
    the 'infeasible' that ``confirm_infeasible`` does not confirm). Both
    solves are replaced by a stub; the star is the feasible 'no-box' star."""
    import n2v.utils.lpsolver as lpsolver

    def failed(*args, **kwargs):
        return None, None, status, {}
    monkeypatch.setattr(lpsolver, "_solve_lp_careful", failed)
    monkeypatch.setattr(lpsolver, "solve_lp", failed)
    _, star = make("no-box")
    with pytest.raises(ValueError):
        ensure_finite_predicate_box(star)
