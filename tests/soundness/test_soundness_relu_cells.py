"""Soundness cells of ReLU on Star, ImageStar, Zono, ImageZono and Box.

A cell is one (set type, method, option): the input sets go through ReLU by
the dispatcher (``reach_layer``) for the kernel alone, and by the model entry
(``reach_pytorch_model``, or an ONNX file loaded with ``load_onnx``) for a
chain affine, ReLU, affine, so that the rows the kernel emits are consumed
downstream. Every cell checks that every true output lies in a returned set;
cells of ``method='exact'`` also check that every returned point is a true
output. The process-pool path of the exact ReLU has its cells in
``test_soundness_parallel.py``, Hexatope and Octatope in
``test_soundness_hexoct_relu.py``; both reuse the reference code here.

Oracle. Every number is dyadic (a multiple of a small power of two), so the
float32 and float64 evaluations and the exact image coincide. The verdict is
the exact image in rationals (``tests/_oracles/rational.py``): the input set is
read from its stored arrays (``rational.star``), its predicate polytope is
mapped through the network piece by piece, and all vertices of each piece of
the image must lie in one returned set, decided by the simplex method in
fractions with zero slack. For ``method='exact'`` the other direction is
checked as well: every vertex of every returned set is the image of a point
of the input set (``rational.unreached``), except on the families with a
neuron whose LP bound lies within the LP pad of 0, which the exact method
keeps in one star with a slack variable about 1e-7 wide (the behaviour
DESIGN section 18 approves for ``_step_relu``). On every cell, member points of the
input set (the vertices of its predicate polytope and points of a grid) are
also replayed on torch in float64, or on onnxruntime for an ONNX model, and
their outputs are checked against the returned sets' own rows by the LP
oracle (``assert_covered``, slack ``LP_TOL``). Image cells check the vertices
of the exact image (a box's ReLU image is the product of its coordinates'
images) and replayed samples with the LP oracle, in the stored layout.

Inputs. The flat families share three predicate variables, so neurons are
coupled: neurons that cross zero (of different widths), neurons whose upper
bound is exactly 0 or whose lower bound is exactly 0 (one generator each, so
the interval estimate is exact), stably active neurons that pin every
predicate variable (an output point then fixes the input point, so a wrong
row cannot hide), a stably inactive neuron and constant neurons. Some Stars
carry a binding row that is not a box row, and neurons whose interval
estimate crosses zero while the LP proves them stable. Lists of several input
sets are pieces of one set cut along a predicate variable. The Zono families
give each crossing neuron one generator and a slope that is a power of two,
so the relaxation is computed without rounding (the rounding of other slopes
is ROUND-10, marked below); one of them has more generators than entries.
One Box cell has bounds that float32 cannot hold and infinite ends, compared
in exact arithmetic.

Layout. A tensor is (channels, height, width); an image set stores (height,
width, channels); a flat set keeps the tensor's own order. Image cells use the
image (C, H, W) = (2, 3, 5) and compare outputs through
``assert_covered(..., out_chw=(2, 3, 5))``, which also checks the stored
height, width and channel count.
"""

import copy
import functools
import itertools
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.lpsolver import LPFailure
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests._oracles.membership import LP_TOL, distance
from tests.cell_support import (assert_covered, constrained_image_star,
                                constrained_star, flat_set, hwc_index,
                                image_set, pixel_box, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx, split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug

_C, _H, _W = 2, 3, 5            # the image of every image cell: 30 entries
_TINY = 2.0 ** -27              # below the LP pad (1e-7) of a phase decision


# --------------------------------------------------------------------------- #
# The exact reference                                                           #
# --------------------------------------------------------------------------- #
def _predicate_rows(computed):
    """``A a <= b`` of a set read by ``rational.star``: its rows ``C a <= d``
    and its finite predicate bounds."""
    _, _, C, d, lb, ub = computed
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
    return A, b


def _read(computed_set):
    """``(c, G, A, b)``: the set ``{c + G a : A a <= b}`` read from its stored
    arrays by ``rational.star``, in the fewest variables and rows that describe
    it exactly. A pinned predicate variable (``lb == ub``, such as the anchor
    of a Hexatope) is replaced by its value; a variable that neither a
    generator nor a row mentions (a zonotope's zero generator for a pixel of
    zero width) is dropped; of rows with the same coefficients only the
    tightest is kept (a Star from ``from_bounds`` states its box both as rows
    and as predicate bounds); a row ``0 <= r`` with ``r >= 0`` is dropped (one
    with ``r < 0`` is kept: the set is empty). None of these changes the set;
    each makes the exact vertex enumeration cheaper."""
    c, G, C, d, lb, ub = R.star(computed_set)
    n = len(lb)
    pinned = {j: lb[j] for j in range(n)
              if lb[j] is not None and ub[j] is not None and lb[j] == ub[j]}
    c = [ci + sum((row[j] * v for j, v in pinned.items()), Fraction(0))
         for ci, row in zip(c, G)]
    d = [di - sum((row[j] * v for j, v in pinned.items()), Fraction(0))
         for di, row in zip(d, C)]
    keep = [j for j in range(n) if j not in pinned and (
        any(row[j] != 0 for row in G) or any(row[j] != 0 for row in C))]
    computed = (c, [[row[j] for j in keep] for row in G],
                [[row[j] for j in keep] for row in C], d,
                [lb[j] for j in keep], [ub[j] for j in keep])
    tightest, infeasible = {}, []
    for row, rhs in zip(*_predicate_rows(computed)):
        if not any(row):
            if rhs < 0:
                infeasible.append((row, rhs))
            continue
        key = tuple(row)
        tightest[key] = min(rhs, tightest.get(key, rhs))
    A = [list(k) for k in tightest] + [r for r, _ in infeasible]
    b = list(tightest.values()) + [r for _, r in infeasible]
    return computed[0], computed[1], A, b


def _pieces(input_set):
    """The input set as one exact piece: its predicate polytope mapped by its
    own centre and generators, read from the stored arrays (no n2v call)."""
    c, G, A, b = _read(input_set)
    return R.affine(R.polytope(A, b), G, c)


def _points(input_set, n):
    """Member points of ``input_set`` as float vectors, exact: the images of
    the vertices of its predicate polytope, then images of points of the
    quarter grid of the polytope's bounding box that keep every row, until
    there are ``n``."""
    c, G, A, b = _read(input_set)
    verts = R.vertices(A, b)
    k = len(verts[0])
    lo = [min(v[j] for v in verts) for j in range(k)]
    hi = [max(v[j] for v in verts) for j in range(k)]
    chosen = list(verts)
    grid = itertools.product(*[[lo[j] + (hi[j] - lo[j]) * Fraction(s, 4)
                                for s in range(5)] for j in range(k)])
    inside = [a for a in grid
              if all(R.dot(row, a) <= rhs for row, rhs in zip(A, b))]
    rng = np.random.default_rng(len(inside))
    for i in rng.permutation(len(inside)):
        if len(chosen) >= n:
            break
        chosen.append(inside[i])
    return [np.array([float(R.dot(row, a) + ci) for row, ci in zip(G, c)])
            for a in chosen[:max(n, len(verts))]]


def _vertices(computed_set, what):
    """The vertices of a returned set, exact: the images of the vertices of
    its predicate polytope (read by :func:`_read`); an empty list for an
    empty set."""
    c, G, A, b = _read(computed_set)
    if not G or not G[0]:
        return [] if any(rhs < 0 for rhs in b) else [tuple(c)]
    if not A or not R.is_bounded(A):
        pytest.fail(f"{what}: a returned set has an unbounded predicate "
                    f"polytope, so its vertices do not describe it")
    return sorted({tuple(R.dot(row, a) + ci for row, ci in zip(G, c))
                   for a in R.vertices(A, b)})


def _member(computed_set):
    """Exact membership in a returned set (``rational.Membership``), on the
    reading of :func:`_read`: the same set in fewer variables and rows, so
    the simplex in fractions is faster."""
    c, G, A, b = _read(computed_set)
    if not G or not G[0]:
        return R.Membership(R.star(computed_set))
    k = len(G[0])
    return R.Membership((c, G, A, b, [None] * k, [None] * k))


def _maximal(pieces):
    """The pieces whose image is not inside another piece's image, with the
    vertices of their images: ``(pieces, images)``.

    A piece whose image vertices are all vertices of another piece's image
    (a region on a kink, which is a face of its neighbour, or a copy) adds
    nothing to the union, so the checks below skip it."""
    images = [frozenset(p.image()) for p in pieces]
    keep = [i for i, im in enumerate(images)
            if not any(im < other or (im == other and j < i)
                       for j, other in enumerate(images) if j != i)]
    return [pieces[i] for i in keep], [sorted(images[i]) for i in keep]


def _assert_holds_image(pieces, sets, what):
    """All vertices of each exact piece lie in one returned set, exactly.

    A convex set holding the vertices of a polytope holds the polytope, so
    the sets then contain the whole exact image. A piece whose vertices are
    spread over several sets proves nothing and fails as well: every rule
    tested here returns one set per activation pattern or per input set."""
    readers = [_member(s) for s in sets]
    escaped, spread = [], []
    pieces, images = _maximal(pieces)
    for i, image in enumerate(images):
        if any(all(member(y) for y in image) for member in readers):
            continue
        outside = [y for y in image if not any(m(y) for m in readers)]
        if outside:
            escaped.append((i, [float(v) for v in outside[0]]))
        else:
            spread.append(i)
    assert not escaped, (
        f"{what}: {len(escaped)} of {len(pieces)} pieces of the exact image "
        f"have a vertex in none of the {len(sets)} returned sets; first: "
        f"piece {escaped[0][0]}, vertex {escaped[0][1]}")
    assert not spread, (
        f"{what}: pieces {spread} of the exact image have their vertices "
        f"spread over several returned sets and in no single one")


def _assert_no_extra_point(pieces, sets, what):
    """Every vertex of every returned set is a true output (exact method)."""
    pieces = _maximal(pieces)[0]
    for k, s in enumerate(sets):
        extra = R.unreached(pieces, _vertices(s, what))
        assert not extra, (
            f"{what}: returned set {k} of {len(sets)} has {len(extra)} "
            f"vertices that are not outputs of any input point; first "
            f"{[float(v) for v in extra[0][1]]}")


def _assert_holds_image_lp(pieces, sets, what):
    """All vertices of each exact piece lie in one returned set, by the LP
    oracle with the slack ``LP_TOL`` relative to the size of the vertex. For
    the Hexatope and Octatope cells (here the Clip cell, the others in
    ``test_soundness_hexoct_relu.py``): their ReLU split pieces carry bounds
    that a solver computed with no outward pad, which the exact check would
    report at the scale of 1e-8 (registry HEXOCT-4, marked by
    ``test_soundness_range_rounding.py::test_hexoct_relu_pieces_hold_the_exact_image``).
    For each piece the sets are tried in turn and a set is left at its first
    vertex outside, so a passing check solves few LPs; a failing one then
    names a vertex in no set, or the piece whose vertices are spread."""
    pieces, images = _maximal(pieces)

    def inside(s, y):
        return distance(s, y) <= LP_TOL * max(1.0, float(np.max(np.abs(y))))
    escaped, spread = [], []
    for i, image in enumerate(images):
        points = [np.array([float(v) for v in y]) for y in image]
        if any(all(inside(s, y) for y in points) for s in sets):
            continue
        outside = [y for y in points if not any(inside(s, y) for s in sets)]
        if outside:
            escaped.append((i, outside[0].tolist()))
        else:
            spread.append(i)
    assert not escaped, (
        f"{what}: {len(escaped)} of {len(images)} pieces of the exact image "
        f"have a vertex in none of the {len(sets)} returned sets; first: "
        f"piece {escaped[0][0]}, vertex {escaped[0][1]}")
    assert not spread, (
        f"{what}: pieces {spread} of the exact image have their vertices "
        f"spread over several returned sets and in no single one")


def _relu_points(points):
    return replay_torch(nn.ReLU(), points, (points[0].size,))


# --------------------------------------------------------------------------- #
# Flat input families                                                           #
# --------------------------------------------------------------------------- #
# Three predicate variables a0, a1, a2 in [-1, 1]; neuron = c + G a.
_MIXED = (
    np.array([-0.5, 0.25, -1.0, 1.0, 3.0, 3.0, 3.0, -2.5, 0.5, -0.25, 0.0]),
    np.array([[2.0, 1.0, 0.0],      # crosses zero, wide: [-3.5, 2.5]
              [0.0, 0.0, 0.5],      # crosses zero, narrow: [-0.25, 0.75]
              [1.0, 0.0, 0.0],      # upper bound exactly 0: [-2, 0]
              [0.0, 1.0, 0.0],      # lower bound exactly 0: [0, 2]
              [1.0, 0.0, 0.0],      # stably active, pins a0
              [0.0, 1.0, 0.0],      # stably active, pins a1
              [0.0, 0.0, 1.0],      # stably active, pins a2
              [-1.0, -1.0, 0.0],    # stably inactive: [-4.5, -0.5]
              [0.0, 0.0, 0.0],      # constant 0.5
              [0.0, 0.0, 0.0],      # constant -0.25
              [0.0, 0.0, 0.0]]))    # constant 0
# Neurons whose estimate crosses zero while the row a0 + a1 <= 3/2 makes them
# stable: the LP maximum of the first is -1/4, the LP minimum of the second 1/4.
_RESOLVED = (np.array([-1.75, 1.75]),
             np.array([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]]))
_ROW = (np.array([[1.0, 1.0, 0.0]]), np.array([1.5]))
# Every neuron stable by its interval estimate, bounds of exactly 0 included.
_STABLE = (
    np.array([3.0, 3.0, 3.0, -1.0, 1.0, -2.5, 1.5, 0.5, -0.25, 0.0]),
    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
              [1.0, 0.0, 0.0],      # [-2, 0]
              [0.0, 1.0, 0.0],      # [0, 2]
              [-1.0, -1.0, 0.0],    # [-4.5, -0.5]
              [0.0, 0.0, 1.0],      # [0.5, 2.5]
              [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
# LP bounds within the LP pad of zero, over four predicate variables, with
# the row a3 >= 1/8 (a row of C, not a predicate bound):
_TINY_FAMILY = (
    np.array([0.5 - _TINY, -0.5 + _TINY, -0.5 + 2.0 ** -20, 0.0, 0.0,
              3.0, 3.0, 3.0, 3.0, 0.0, 0.0]),
    np.array([[0.5, 0.0, 0.0, 0.0],     # LP min -2^-27: kept in one star
              [0.0, 0.5, 0.0, 0.0],     # LP max 2^-27: kept in one star
              [0.0, 0.0, 0.5, 0.0],     # max 2^-20, below an SCS shortfall
              [0.0, 0.0, 0.0, _TINY],   # LP range [2^-30, 2^-27]: one side
              [1.0, 1.0, 0.0, 0.0],     # crosses zero: [-2, 2]
              [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
              [0.0, 0.0, 0.0, -_TINY],  # LP range [-2^-27, -2^-30]: the other
              [0.0, 0.0, _TINY / 2, 0.0]]))  # [-2^-28, 2^-28]: thin, split
_TINY_ROW = (np.array([[0.0, 0.0, 0.0, -1.0]]), np.array([-0.125]))
# Cuts of a0 for a list of input sets: shares 3, 4 and 9 of 16, placed so
# that no neuron of a part has an LP bound at or near 0 unless its interval
# estimate is exact (a bound the LP cannot sign is kept as a sliver).
_CUTS = ((-1.0, -0.625), (-0.625, -0.125), (-0.125, 1.0))
# The Zono version of _MIXED: the wide crossing neuron is -1 + 2 a0, on
# [-3, 1] with slope 1/4 (the narrow one has slope 3/4). One generator per
# crossing neuron keeps ``Zono.get_bounds`` exact, so the slope, error term
# and scaled row are exact in float64. The Zono cuts keep every slope a power
# of two: neuron 0 is inactive on the first two parts and has slope 1/2 on
# the third, where neuron 2 has an upper bound of exactly 0.
_ZMIXED = (np.concatenate([[-1.0], _MIXED[0][1:]]),
           np.vstack([[2.0, 0.0, 0.0], _MIXED[1][1:]]))
_ZCUTS = ((-1.0, -0.5), (-0.5, 0.0), (0.0, 1.0))
# A Zono with more generators (5) than entries (4), the shape a relaxation
# leaves behind for the next ReLU: generators 3 and 4 each appear in one
# entry only, as an error generator does. Two crossing neurons with one
# generator each, -1 + 2 a0 on [-3, 1] (slope 1/4) and 1/2 + a1 on
# [-1/2, 3/2] (slope 3/4), a stably active neuron over four generators,
# [5/2, 19/2], and a stably inactive one, [-29/4, -19/4]. On the zonotopes of
# ``from_bounds`` (square and diagonal) a row and a column of a neuron hold
# the same generator; here they do not.
_ZWIDE = (np.array([-1.0, 0.5, 6.0, -6.0]),
          np.array([[2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.5, 0.0],
                    [0.0, 0.0, 1.0, 0.0, -0.25]]))


def _stack(*parts):
    return (np.concatenate([p[0] for p in parts]),
            np.vstack([p[1] for p in parts]))


def _star(c, G, rows=None, a_lo=None, a_hi=None, predicate_box=True,
          outer_zono=None):
    """``{c + G a : -1 <= a <= 1 (rows), rows}`` as a Star whose ``C`` holds
    the box rows ``+-I``; ``a_lo``/``a_hi`` narrow the predicate bounds."""
    k = G.shape[1]
    C = np.vstack([np.eye(k), -np.eye(k)])
    d = np.ones((2 * k, 1))
    if rows is not None:
        C = np.vstack([C, rows[0]])
        d = np.vstack([d, rows[1].reshape(-1, 1)])
    V = np.hstack([c.reshape(-1, 1), G])
    if not predicate_box:
        return Star(V, C, d)
    a_lo = -np.ones(k) if a_lo is None else np.asarray(a_lo, float)
    a_hi = np.ones(k) if a_hi is None else np.asarray(a_hi, float)
    return Star(V, C, d, a_lo.reshape(-1, 1), a_hi.reshape(-1, 1),
                outer_zono=outer_zono)


def _zono(c, G, a_lo=None, a_hi=None):
    """``{c + G a : a_lo <= a <= a_hi}`` as a Zono (a = m + r t, t in
    [-1, 1]; exact for dyadic data)."""
    k = G.shape[1]
    a_lo = -np.ones(k) if a_lo is None else np.asarray(a_lo, float)
    a_hi = np.ones(k) if a_hi is None else np.asarray(a_hi, float)
    m, r = (a_lo + a_hi) / 2, (a_hi - a_lo) / 2
    return Zono((c + G @ m).reshape(-1, 1), G * r)


def _cut(build, c, G, cuts=_CUTS, **kw):
    out = []
    for lo0, hi0 in cuts:
        a_lo, a_hi = -np.ones(G.shape[1]), np.ones(G.shape[1])
        a_lo[0], a_hi[0] = lo0, hi0
        out.append(build(c, G, a_lo=a_lo, a_hi=a_hi, **kw))
    return out


# Crossing neurons with a neuron that only the LP proves inactive between them
# in the order the relaxed rules choose neurons for the LP (by width, by
# |l| u, by the larger of u and |l|): [-7/2, 5/2], then [-15/4, 1/4] (LP
# maximum -1/4 under the row a0 + a1 <= 3/2), then [-5/8, 3/8], and the
# narrow [-1/16, 3/16] last; three neurons pin a0, a1, a2. With the factor
# 1/4 the first three go to the LP, and the bounds of the third are read
# after the second has left the list.
_INTERLEAVED = (np.array([-0.5, -1.75, -0.125, 0.0625, 3.0, 3.0, 3.0]),
                np.array([[2.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.5],
                          [0.0, 0.0, 0.125], [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
_MIXED_ROW = _stack(_MIXED, _RESOLVED)
_RESOLVED_ONLY = _stack(_RESOLVED, (np.array([3.0, 3.0, 3.0, -1.0, -0.5]),
                                    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                              [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                                              [0.0, 0.0, 0.0]])))

# family: (builder of the input sets of a kind, kinds, exact method is exact)
_FAMILIES = {
    "mixed": (lambda kind: [_star(*_MIXED) if kind == "Star"
                            else _zono(*_ZMIXED)], ("Star", "Zono"), True),
    "mixed-row": (lambda kind: [_star(*_MIXED_ROW, rows=_ROW)], ("Star",),
                  True),
    "resolved": (lambda kind: [_star(*_RESOLVED_ONLY, rows=_ROW)], ("Star",),
                 True),
    "interleaved": (lambda kind: [_star(*_INTERLEAVED, rows=_ROW)],
                    ("Star",), True),
    "stable": (lambda kind: [_star(*_STABLE) if kind == "Star"
                             else _zono(*_STABLE)], ("Star", "Zono"), True),
    "several": (lambda kind: (_cut(_star, *_MIXED_ROW, rows=_ROW)
                              if kind == "Star"
                              else _cut(_zono, *_ZMIXED, cuts=_ZCUTS)),
                ("Star", "Zono"), True),
    "wide": (lambda kind: [_zono(*_ZWIDE)], ("Zono",), True),
    # LP bounds within the pad of 0 are kept in one star with a sliver
    # variable of width about 1e-7: exact only up to that sliver.
    "tiny": (lambda kind: [_star(*_TINY_FAMILY, rows=_TINY_ROW)], ("Star",),
             False),
    "no-predicate-box": (lambda kind: [_star(
        *_stack(_MIXED_ROW, (np.array([0.5 - _TINY]),
                             np.array([[0.5, 0.0, 0.0]]))),
        rows=_ROW, predicate_box=False)], ("Star",), False),
}


@functools.lru_cache(maxsize=None)
def _family_reference(name, kind):
    sets = _FAMILIES[name][0](kind)
    pieces = _maximal([p for s in sets for p in R.relu(_pieces(s))])[0]
    points = [x for s in sets for x in _points(s, 24)]
    return sets, pieces, points


def _family(name, kind):
    """``(input sets, exact pieces of their ReLU image, member points)``; the
    reference is computed once per family, the sets are fresh copies, so a
    kernel that changed its input in place could not spoil another cell."""
    sets, pieces, points = _family_reference(name, kind)
    return copy.deepcopy(sets), pieces, points


def test_exact_reader_describes_the_same_set():
    """The reader :func:`_read` (pinned variables substituted, unused
    variables and repeated rows dropped) describes the same set as the
    oracle's own reading: same vertices, on a Star with a binding row and on
    a Hexatope (pinned anchor). A reader that lost a row or a variable would
    make every exact reference here wrong; this is its planted-bug guard."""
    from n2v.sets import Hexatope
    for s in (_star(*_MIXED_ROW, rows=_ROW),
              Hexatope.from_bounds(np.array([-1.0, 0.25]),
                                   np.array([0.5, 0.75])).affine_map(
                  np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 2.0]]),
                  np.array([[0.5], [0.0], [-0.25]]))):
        assert _vertices(s, "reader") == R.set_vertices(R.star(s)), (
            f"the reduced reading of a {type(s).__name__} has other vertices "
            f"than the oracle's reading")
        centre = R.star(s)[0]
        outside = [ci + 4 for ci in centre]
        member = _member(s)
        assert member(centre) == R.member(R.star(s), centre), (
            f"the reduced member test of a {type(s).__name__} disagrees with "
            f"the oracle on its centre")
        assert not member(outside), (
            f"the reduced member test of a {type(s).__name__} accepts a point "
            f"far outside the set")


_STAR_OPTIONS = [("exact", {}), ("approx", {})] + [
    ("approx", {"relax_method": m, "relax_factor": f})
    for m in ("range", "area", "bound") for f in (0.25, 0.5, 1.0)]


def _option_id(method, opts):
    if "relax_method" in opts:
        return f"{method}-{opts['relax_method']}-{opts['relax_factor']}"
    if "precompute_bounds" in opts:
        return f"{method}-pre-{opts['precompute_bounds']}"
    return method if method == "exact" else "approx-standard"


# Relaxation factors per family where fewer suffice: on the stable family every
# relax method returns before it reads the factor; on the resolved family the
# factor 1 (no LP) is the estimate-only path every other family runs too; on
# the several family (a list of three stars) the factor 1/4 runs each rule's
# LP path on every star, and the other factors add no path (the factors 1/2
# and 1 run on the single stars of the other families); the interleaved
# family exists for the LP bookkeeping of the factor 1/4, and the standard
# rule and the exact method do not run on it.
_FAMILY_FACTORS = {"stable": (0.5,), "resolved": (0.25, 0.5),
                   "several": (0.25,), "no-predicate-box": (0.5, 1.0),
                   "interleaved": (0.25,)}
_RELAX_ONLY = {"interleaved"}


def _family_runs(name, opts):
    if "relax_factor" not in opts:
        return name not in _RELAX_ONLY
    return opts["relax_factor"] in _FAMILY_FACTORS.get(name, (0.25, 0.5, 1.0))


_STAR_CELLS = [pytest.param(name, method, opts,
                            id=f"{name}-{_option_id(method, opts)}")
               for name, (_, kinds, _) in _FAMILIES.items() if "Star" in kinds
               for method, opts in _STAR_OPTIONS if _family_runs(name, opts)]


@pytest.mark.parametrize("name,method,opts", _STAR_CELLS)
def test_relu_star_holds_the_exact_image(name, method, opts):
    """ReLU on Stars through ``reach_layer``, every method and relaxation.

    Catches: a crossing neuron's relaxation or split cut below the graph
    (range); a wrong triangle or split row whose bounding box is still right,
    seen because stably active neurons pin every predicate variable (row); a
    neuron with an upper bound of exactly 0, a lower bound of exactly 0, or a
    phase that only the LP decides, handled as another phase (regime); the
    neurons to relax given another neuron's bounds (plumbing); a split piece
    dropped or replaced (piece); an LP bound padded inward or a sliver
    variable of the wrong sign, 1e-8 to 1e-7 wide, seen in exact arithmetic
    (rounding); a list of input sets of which only one is processed; the
    approximate method with a relaxation factor above 0 routed to the
    splitting kernel, which holds the image and returns several stars per
    input star (count). For ``exact``, also a returned point that is no
    output."""
    sets, pieces, points = _family(name, "Star")
    what = f"ReLU {method} {opts} on the Star family {name!r}"
    out = reach_one_layer(nn.ReLU(), sets, method, what, **opts)
    if method == "approx":
        assert len(out) == len(sets), (
            f"{what}: {len(out)} stars for {len(sets)} input star(s); the "
            f"approximate method relaxes each star and does not split")
    _assert_holds_image(pieces, out, what)
    if method == "exact" and _FAMILIES[name][2]:
        _assert_no_extra_point(pieces, out, what)
    assert_covered(out, _relu_points(points), what)


@pytest.mark.parametrize("name", [n for n, (_, kinds, _) in _FAMILIES.items()
                                  if "Zono" in kinds])
def test_relu_zono_holds_the_exact_image(name):
    """ReLU on Zonos through ``reach_layer``. Catches: the error generators
    of several crossing neurons merged into one noise symbol (the box stays
    right, combinations of outputs are lost; row); a centre shifted the wrong
    way (range); a neuron with an upper bound of exactly 0 passed through, or
    one with a lower bound of exactly 0 relaxed (regime); a stably inactive
    neuron whose generator row is kept, or which is relaxed as a crossing
    one (its true output 0 escapes once the stably active neurons fix the
    predicate point; regime); a slope applied to a generator column instead
    of the neuron's row (row); a list of zonotopes of which one is dropped.
    The families: generators fewer than entries (``mixed``, ``stable``,
    ``several``, a list of three) and more generators than entries
    (``wide``, the shape a relaxation leaves for the next ReLU)."""
    sets, pieces, points = _family(name, "Zono")
    what = f"ReLU approx on the Zono family {name!r}"
    out = reach_one_layer(nn.ReLU(), sets, "approx", what)
    _assert_holds_image(pieces, out, what)
    assert_covered(out, _relu_points(points), what)


def test_relu_zono_keeps_the_error_of_a_neuron_that_barely_crosses_zero():
    """Neuron 0 crosses zero by 2^-40 and neuron 1 by 2^-30 on both sides;
    their relaxation errors (about 4.5e-13 and 2.3e-10) are exact in float64
    (the slopes are 2^-40 and 1/2) and must be kept: dropping a small error
    generator loses the outputs near the top, seen only in exact arithmetic
    (rounding)."""
    c = np.array([-0.5 + 2.0 ** -40, 0.0, 3.0, 3.0])
    G = np.array([[0.5, 0.0], [0.0, 2.0 ** -30], [1.0, 0.0], [0.0, 1.0]])
    zono = Zono(c.reshape(-1, 1), G)
    what = "ReLU approx on a Zono with neurons that barely cross zero"
    out = reach_one_layer(nn.ReLU(), [zono], "approx", what)
    _assert_holds_image(R.relu(_pieces(zono)), out, what)


# A box with every regime: straddling zero, positive, negative, of zero width,
# an upper bound of exactly 0, a lower bound of exactly 0, and the widest
# coordinate positive, so that the parts a list cuts along it have different
# images (none of them holds another's).
_BOX_LO = np.array([-0.75, 0.375, -1.25, 0.625, -1.0, 0.0, 0.25])
_BOX_HI = np.array([1.0, 0.875, -0.5, 0.625, 0.0, 0.75, 2.25])


def _clipped(v):
    return [max(Fraction(float(x)), Fraction(0)) for x in np.ravel(v)]


@pytest.mark.parametrize("parts", [1, 3], ids=["one-box", "three-boxes"])
@pytest.mark.parametrize("method", ["approx", "exact"])
def test_relu_box_list_and_boundary_values_hold_the_exact_image(method, parts):
    """ReLU on Boxes, one box and a list of three cut from it. The exact image
    of a box is the box of its clipped bounds (ReLU acts on each coordinate
    and a box is a product); a Box's rows are its bounds, so comparing bounds
    in exact arithmetic is membership against the rows. Every image box must
    lie in a returned box, and under ``exact`` (``reach_layer`` answers it
    with the same kernel, whose result is the image) every returned box in an
    image box; ``approx`` promises only the first direction. Catches: an
    upper bound taken from the lower bound, or a lower
    bound from the upper bound (range); only the last box of a list returned
    (regime: the others' outputs are lost); a bound of exactly 0 or a
    zero-width coordinate handled wrongly; an input box clipped in place
    (``reach_one_layer`` fails when the reach changes its input)."""
    sets = ([Box(_BOX_LO, _BOX_HI)] if parts == 1
            else split_cover("Box", _BOX_LO, _BOX_HI, parts=3))
    what = f"ReLU {method} on {len(sets)} Box(es)"
    out = reach_one_layer(nn.ReLU(), sets, method, what)
    images = [(_clipped(s.lb), _clipped(s.ub)) for s in sets]
    boxes = [([Fraction(float(v)) for v in np.ravel(s.lb)],
              [Fraction(float(v)) for v in np.ravel(s.ub)]) for s in out]

    def inside(inner, outer):
        return all(ol <= il and iu <= ou for il, iu, ol, ou in
                   zip(inner[0], inner[1], outer[0], outer[1]))
    missed = [k for k, image in enumerate(images)
              if not any(inside(image, box) for box in boxes)]
    assert not missed, (
        f"{what}: the exact images of input boxes {missed} lie in none of the "
        f"{len(out)} returned boxes")
    if method == "exact":
        extra = [k for k, box in enumerate(boxes)
                 if not any(inside(box, image) for image in images)]
        assert not extra, (
            f"{what}: returned boxes {extra} hold points that are no output")
    assert_covered(out, _relu_points(samples(_BOX_LO, _BOX_HI, 64)), what)


def _box_beyond_float32():
    """Bounds ``(lo, hi)`` of 24 coordinates that float32 cannot hold (random
    float64 numbers, so neither dyadic nor short), from -8 to 13, intervals up
    to 16 wide: crossing zero, positive, negative, of zero width, with an upper
    bound of exactly 0 and a lower bound of exactly 0, and one coordinate with
    a lower bound of -inf and one with an upper bound of +inf (a Box allows
    them)."""
    rng = np.random.default_rng(31)
    lo = rng.uniform(-8.0, 4.0, size=24)
    hi = lo + rng.uniform(0.5, 9.0, size=24)
    lo[0], hi[0] = -rng.uniform(1.0, 8.0), rng.uniform(1.0, 8.0)
    lo[1], hi[1] = rng.uniform(0.1, 1.0), rng.uniform(2.0, 9.0)
    lo[2], hi[2] = -rng.uniform(4.0, 8.0), -rng.uniform(0.1, 1.0)
    hi[3] = lo[3] = rng.uniform(0.1, 1.0)
    lo[4], hi[4] = -rng.uniform(1.0, 8.0), 0.0
    lo[5], hi[5] = 0.0, rng.uniform(1.0, 8.0)
    lo[6], hi[6] = -np.inf, rng.uniform(1.0, 8.0)
    lo[7], hi[7] = -rng.uniform(1.0, 8.0), np.inf
    return lo, hi


def _exact_end(v):
    """A float bound as the rational it is, an infinite one as itself
    (``Fraction`` compares with ``inf``)."""
    v = float(v)
    return v if np.isinf(v) else Fraction(v)


def test_relu_box_holds_the_exact_image_of_non_float32_bounds():
    """Approx ReLU through ``reach_layer`` on one Box of
    :func:`_box_beyond_float32`. Its exact image is the box
    [max(lb, 0), max(ub, 0)]; the ends of that box, as the rationals the
    stored floats are (an infinite end as itself), must lie in the returned
    box with no slack, and 128 member points (an infinite end replaced by a
    finite one 16 beyond the other end) replayed on torch in float64 must be
    members by the LP oracle. Catches a bound stored through float32 or
    computed by a rounded operation, such as a centre plus a radius
    (rounding: the dyadic bounds of the cell above have so few bits that
    float32 holds them), and an infinite end turned into a finite number or
    NaN."""
    lo, hi = _box_beyond_float32()
    what = "ReLU approx on a Box whose bounds are not float32 numbers"
    out = reach_one_layer(nn.ReLU(), [Box(lo, hi)], "approx", what)
    image = [(_exact_end(max(a, 0.0)), _exact_end(max(b, 0.0)))
             for a, b in zip(lo, hi)]
    holding = [s for s in out if all(
        _exact_end(l) <= a and b <= _exact_end(u) for (a, b), l, u in
        zip(image, np.ravel(s.lb), np.ravel(s.ub)))]
    assert holding, (
        f"{what}: the exact image box lies in none of the {len(out)} "
        f"returned boxes")
    finite_lo = np.where(np.isinf(lo), np.where(np.isinf(hi), -16.0,
                                                 hi - 16.0), lo)
    finite_hi = np.where(np.isinf(hi), finite_lo + 16.0, hi)
    assert_covered(out, _relu_points(samples(finite_lo, finite_hi)), what)


# --------------------------------------------------------------------------- #
# Options and injected faults of the Star rules                                 #
# --------------------------------------------------------------------------- #
# The standard approx rule makes no LP (its phases come from the interval
# estimate), so it has no item here: with any solver it is
# test_relu_star_holds_the_exact_image[tiny-approx-standard].
@pytest.mark.parametrize("method,opts", [
    pytest.param(m, o, id=_option_id(m, o)) for m, o in _STAR_OPTIONS
    if o.get("relax_factor", 0.25) == 0.25 and (m == "exact" or o)])
def test_relu_star_ignores_an_inexact_solver_choice(inexact_scs, method, opts):
    """The caller asks for SCS, whose optima are made 2e-5 too small (a
    maximum) or too large (a minimum) by the ``inexact_scs`` fixture. The
    phase decisions must use HiGHS whatever is asked: neuron 2 of the tiny
    family has a maximum of 2^-20, which an SCS bound would call inactive and
    zero, and the LP minima of the relaxed rules would read too high
    (plumbing)."""
    sets, pieces, points = _family("tiny", "Star")
    what = f"ReLU {method} {opts} with lp_solver='SCS'"
    out = reach_one_layer(nn.ReLU(), sets, method, what, lp_solver="SCS",
                          **opts)
    _assert_holds_image(pieces, out, what)


# A HiGHS variant named by the caller ('linprog', 'highs') has no cell of its
# own: the phase LPs keep any solver HiGHS runs (``_bound_lp_solver``), and
# 'linprog', 'highs' and the default (linprog) reach the same call,
# ``scipy.optimize.linprog(method='highs')``, which every cell of
# test_relu_star_holds_the_exact_image runs; the replaced solver is checked by
# test_relu_star_ignores_an_inexact_solver_choice.


@pytest.fixture
def lp_fails(monkeypatch):
    """Every Star range LP returns no optimum (a solver failure or a false
    'infeasible'); the certified emptiness check still works."""
    monkeypatch.setattr(Star, "get_range", lambda self, i, s=None: (None, None))
    monkeypatch.setattr(Star, "get_max", lambda self, i, s=None: None)
    monkeypatch.setattr(Star, "get_min", lambda self, i, s=None: None)


@pytest.fixture
def emptiness_check_fails(monkeypatch, lp_fails):
    """As ``lp_fails``, and the emptiness check raises ``LPFailure``."""
    def raise_failure(self, *args, **kwargs):
        raise LPFailure("injected: the emptiness check is undecided")
    monkeypatch.setattr(Star, "is_empty_set", raise_failure)


# The standard approx rule calls neither a range LP nor the emptiness check,
# so it has no item here (it is test_relu_star_holds_the_exact_image
# [mixed-row-approx-standard] and [no-predicate-box-approx-standard]).
@pytest.mark.parametrize("name,fault", [
    ("mixed-row", "lp_fails"), ("mixed-row", "emptiness_check_fails"),
    ("no-predicate-box", "lp_fails")])
@pytest.mark.parametrize("method,opts", [
    pytest.param(m, o, id=_option_id(m, o)) for m, o in _STAR_OPTIONS
    if o.get("relax_factor", 0.25) == 0.25 and (m == "exact" or o)])
def test_relu_star_keeps_a_star_whose_lps_fail(request, fault, method, opts,
                                               name):
    """No range LP returns an optimum, and in the second case the emptiness
    check is undecided too. A solver failure is not a proof: every crossing
    neuron must be kept with its interval bounds (or, on a Star without a
    predicate box, as unbounded) and the star must not be dropped as empty
    (plumbing)."""
    request.getfixturevalue(fault)
    sets, pieces, points = _family(name, "Star")
    what = f"ReLU {method} {opts} with {fault} on the family {name!r}"
    out = reach_one_layer(nn.ReLU(), sets, method, what, **opts)
    _assert_holds_image(pieces, out, what)


_SLIVER_NEURON = (np.array([0.5 - _TINY]), np.array([[0.5, 0.0, 0.0]]))


@pytest.mark.parametrize("method,opts", [
    pytest.param(m, o, id=_option_id(m, o)) for m, o in _STAR_OPTIONS
    if o.get("relax_factor", 0.5) == 0.5])
def test_relu_star_with_an_outer_zonotope(method, opts):
    """A Star that carries an outer zonotope, with a neuron whose LP minimum
    lies within the LP pad (kept in one star with a sliver variable): the
    star part of the result holds the exact image (the branches that update
    the outer zonotope run beside the star's own update and must not change
    it)."""
    tiny = _stack(_MIXED, _SLIVER_NEURON)
    star = _star(*tiny, outer_zono=Zono(tiny[0].reshape(-1, 1), tiny[1]))
    pieces = R.relu(_pieces(star))
    what = f"ReLU {method} {opts} on a Star with an outer zonotope"
    out = reach_one_layer(nn.ReLU(), [star], method, what, **opts)
    _assert_holds_image(pieces, out, what)


_OUTER_ZONO_ROUNDED = known_bug(
    "7b", "the exact ReLU keeps a neuron whose LP minimum lies within the LP "
          "pad in one star with a sliver variable t in [0, T]; the outer "
          "zonotope's centre c + T/2 is rounded to nearest, so the star's own "
          "vertex lies 1e-17 outside its outer zonotope (nothing in n2v reads "
          "the outer zonotope today)")
# variant: (centre, generators, extra rows); "resolved" has neurons that only
# the LP decides (one proven inactive, one proven active), so the rows of the
# first are reset in the star and its zonotope by ``_reset_star_rows``.
_Z_VARIANTS = {"split": (*_MIXED, None), "stable": (*_STABLE, None),
               "resolved": (*_RESOLVED_ONLY, _ROW),
               "sliver": (*_stack(_MIXED, _SLIVER_NEURON), None)}


@pytest.mark.parametrize("variant,method,opts", [
    pytest.param("split", "exact", {}, id="split-exact"),
    pytest.param("stable", "exact", {}, id="stable-exact"),
    *[pytest.param("stable", m, o, id=f"stable-{_option_id(m, o)}")
      for m, o in _STAR_OPTIONS
      if m == "approx" and o.get("relax_factor", 0.5) == 0.5],
    pytest.param("resolved", "exact", {}, id="resolved-exact"),
    *[pytest.param("resolved", m, o, id=f"resolved-{_option_id(m, o)}")
      for m, o in _STAR_OPTIONS if o.get("relax_factor") == 0.25],
    pytest.param("sliver", "exact", {}, id="sliver-exact",
                 marks=_OUTER_ZONO_ROUNDED)])
def test_relu_keeps_each_star_inside_its_outer_zonotope(variant, method, opts):
    """The outer zonotope a Star carries covers the star (its documented
    meaning). ReLU updates it beside the star: a row zeroed in the star and
    not in the zonotope, a split piece given the other piece's zonotope, or a
    sliver variable without its generator leaves star points outside. Every
    vertex of every returned star that carries a zonotope must be a member
    of that zonotope, exactly. The split and stable variants pass; the sliver
    variant is the defect named in its marker."""
    c, G, rows = _Z_VARIANTS[variant]
    star = _star(c, G, rows=rows, outer_zono=Zono(c.reshape(-1, 1), G))
    what = f"ReLU {method} {opts} on the {variant} Star with an outer zonotope"
    out = reach_one_layer(nn.ReLU(), [star], method, what, **opts)
    carrying = [(k, s) for k, s in enumerate(out) if s.Z is not None]
    if not carrying:
        pytest.fail(f"{what}: no returned star carries an outer zonotope, so "
                    f"the cell checks nothing")
    for k, s in carrying:
        zono = R.Membership(R.star(s.Z))
        outside = [v for v in _vertices(s, what) if not zono(v)]
        assert not outside, (
            f"{what}: returned star {k} has {len(outside)} vertices outside "
            f"its own outer zonotope; first {[float(x) for x in outside[0]]}")


@pytest.mark.parametrize("method", ["range", "area", "bound"])
@pytest.mark.parametrize("factor", [-0.25, 1.25])
def test_relu_relax_factor_outside_zero_one_is_refused(method, factor):
    """A relaxation factor outside [0, 1] has no meaning and raises."""
    star = _star(*_MIXED)
    with pytest.raises(ValueError):
        reach_layer(nn.ReLU(), [star], "approx", relax_method=method,
                    relax_factor=factor)


def test_relu_relax_factor_zero_is_the_exact_image_or_refused():
    """``relax_factor=0`` runs the exact reach today (DESIGN section 7 makes it
    a ``ValueError`` later, contract ``REACH-3``). While it answers, the
    answer is the exact image (the delegation passes the solver and list of
    stars through): it holds every piece, adds no predicate variable (a
    relaxation would, and the count is checked before the vertices are
    enumerated) and has no vertex that is not an output. Catches the factor
    0 sent through a relaxation kernel, which holds the image and is not
    exact."""
    sets, pieces, points = _family("several", "Star")
    for method in ("range", "area", "bound", "standard"):
        what = f"ReLU approx relax_factor=0 relax_method={method!r}"
        try:
            out = reach_layer(nn.ReLU(), sets, "approx", relax_factor=0.0,
                              relax_method=method)
        except ValueError:
            continue
        assert out, f"{what}: no set returned for non-empty input stars"
        _assert_holds_image(pieces, out, what)
        added = sorted({s.nVar for s in out} - {s.nVar for s in sets})
        assert not added, (
            f"{what}: returned stars with {added} predicate variables for "
            f"input stars with {sorted({s.nVar for s in sets})}: a "
            f"relaxation, not the exact reach")
        _assert_no_extra_point(pieces, out, what)


def _exact_ranges(input_set):
    """The exact range of every entry of ``input_set`` over its predicate
    polytope (read by :func:`_read`, extremes over the polytope's vertices in
    rationals), as float arrays ``(lb, ub)`` of shape (dim, 1). Fails when an
    end is not a float, so the bounds are the ranges themselves."""
    c, G, A, b = _read(input_set)
    verts = R.vertices(A, b)
    ends = [(min(ci + R.dot(row, v) for v in verts),
             max(ci + R.dot(row, v) for v in verts)) for ci, row in zip(c, G)]
    if any(Fraction(float(e)) != e for pair in ends for e in pair):
        pytest.fail("an end of an exact range is not a float")
    return tuple(np.array([float(pair[k]) for pair in ends]).reshape(-1, 1)
                 for k in (0, 1))


_APPROX_HALF = [o for m, o in _STAR_OPTIONS
                if m == "approx" and o.get("relax_factor", 0.5) == 0.5]


@pytest.mark.parametrize("opts", [pytest.param(o, id=_option_id("approx", o))
                                  for o in _APPROX_HALF])
def test_relu_approx_with_given_bounds_holds_the_exact_image(opts):
    """``reach_layer`` with ``precomputed_bounds`` (the bound pre-pass's, or a
    caller's such as branch and bound) equal to the exact range of every
    neuron of the mixed-row family, computed in rationals. The approx rules
    use given bounds as they are: here they decide the phase of the two
    neurons whose interval estimate crosses zero while the row a0 + a1 <= 3/2
    makes them stable (one inactive, one active), and they are the ends of
    the triangle of each crossing neuron, tighter than its estimate, so the
    upper face passes through the true extremes and the outward correction
    of its rounding is needed. Catches given bounds intersected from the
    wrong end or with the star's own pair exchanged, at the intersection or
    at a rule's own call (a crossing neuron then reads stable and is zeroed
    or kept whole: range, plumbing), and a triangle row rounded inward
    (rounding). Verdict: the exact image in rationals, and member points
    replayed on torch and checked against the rows (LP oracle)."""
    sets, pieces, points = _family("mixed-row", "Star")
    bounds = _exact_ranges(sets[0])
    what = f"ReLU approx {opts} with the exact ranges as given bounds"
    out = reach_one_layer(nn.ReLU(), sets, "approx", what,
                          precomputed_bounds=bounds, **opts)
    _assert_holds_image(pieces, out, what)
    assert_covered(out, _relu_points(points), what)


def _pinned_crossing_stars(predicate_box, count=24, seed=1):
    """``count`` Stars over two predicate variables a in [-1, 1], stated as
    rows of ``C`` (and, with ``predicate_box``, as predicate bounds too):
    entry 0 is c + g a with dyadic c and g and crosses zero, entries 1 and 2
    are 3 + a0 and 3 + a1 (stably active), which pin the predicate point in
    the output, so a row that cuts off one output cannot be made up for by
    another predicate point. Each comes with the exact range of every entry
    (dyadic ends, exact floats)."""
    rng = np.random.default_rng(seed)
    C = np.vstack([np.eye(2), -np.eye(2)])
    d = np.ones((4, 1))
    out = []
    while len(out) < count:
        c0, g = rng.integers(-8, 9) / 16.0, rng.integers(-16, 17, 2) / 16.0
        lo, hi = c0 - np.abs(g).sum(), c0 + np.abs(g).sum()
        if not lo < 0 < hi:
            continue
        V = np.array([[c0, g[0], g[1]], [3.0, 1.0, 0.0], [3.0, 0.0, 1.0]])
        star = (Star(V, C, d, -np.ones((2, 1)), np.ones((2, 1)))
                if predicate_box else Star(V, C, d))
        out.append((star, (np.array([[lo], [2.0], [2.0]]),
                           np.array([[hi], [4.0], [4.0]]))))
    return out


_ROWS_ONLY_TRIANGLE_ROUNDED = known_bug(
    16, "the approx ReLU moves the right-hand side of the triangle's upper "
        "face out by a bound on its rounding only when the Star has a "
        "predicate box; on a Star given by rows only, with given bounds equal "
        "to a crossing neuron's real range, the row stays rounded to nearest "
        "and the true output 0 at the neuron's minimum violates it by up to "
        "1.1e-16 (ROUND-14, its ReLU path; the fix of REACH-5 removes it)")


@pytest.mark.parametrize("box", [
    "predicate-box",
    pytest.param("rows-only", marks=_ROWS_ONLY_TRIANGLE_ROUNDED)])
def test_relu_approx_with_tight_given_bounds_holds_the_ends_exactly(box):
    """Standard approx ReLU through ``reach_layer`` on 24 Stars with one
    crossing neuron whose given bounds (``precomputed_bounds``) are its exact
    range, so the triangle's faces pass through the true outputs at both ends
    of the range with no margin: every vertex of the exact image must be a
    member of the returned star in exact arithmetic. Catches a triangle row
    whose rounding is not bounded outward (rounding). The Stars with a
    predicate box pass; the same Stars given by rows only are the defect in
    the marker."""
    missed = []
    for k, (star, bounds) in enumerate(_pinned_crossing_stars(
            box == "predicate-box")):
        what = f"approx ReLU on pinned Star {k} ({box}), exact given bounds"
        out = reach_one_layer(nn.ReLU(), [star], "approx", what,
                              precomputed_bounds=bounds)
        member = _member(out[0]) if len(out) == 1 else None
        if member is None:
            pytest.fail(f"{what}: {len(out)} sets returned for one star")
        image = {y for p in R.relu(_pieces(star)) for y in p.image()}
        missed += [(k, [float(v) for v in y]) for y in sorted(image)
                   if not member(y)]
    assert not missed, (
        f"{len(missed)} vertices of the exact images lie outside the returned "
        f"stars ({box}); first: star {missed[0][0]}, output {missed[0][1]}")


# --------------------------------------------------------------------------- #
# Composition through the model entry: affine, ReLU, affine                     #
# --------------------------------------------------------------------------- #
_W1 = np.array([[1.0, 1.0, 0.0],        # crosses zero: [-1.75, 1.25]
                [0.5, -1.0, 0.5],       # crosses zero: [-0.875, 1.625]
                [-1.0, 0.0, -1.0],      # stably inactive
                [0.0, 1.0, 1.0],        # stably active
                [1.0, 0.0, 0.0]])       # upper bound exactly 0: [-2, 0]
_B1 = np.array([-0.25, 0.0, -1.5, 0.5, -1.0])
_W2 = np.array([[1.0, -0.5, 0.25, 1.0, 2.0],
                [-0.5, 1.0, 1.0, -0.25, -1.0]])
_B2 = np.array([0.125, -0.25])
_W3 = np.array([[1.0, -1.0], [0.5, 1.0], [-1.0, 0.25]])     # second ReLU
_B3 = np.array([0.25, -0.5, 0.125])
_LO3, _HI3 = np.array([-1.0, -0.5, 0.25]), np.array([1.0, 0.5, 1.25])


def _linear(W, b):
    layer = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
        layer.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    return layer


def _chain(two_relus=False):
    """Linear(3, 5), ReLU, Linear(5, 2), and with ``two_relus`` a further
    ReLU and Linear(2, 3), whose ReLU receives the pieces of the first."""
    layers = [_linear(_W1, _B1), nn.ReLU(), _linear(_W2, _B2)]
    if two_relus:
        layers += [nn.ReLU(), _linear(_W3, _B3)]
    return nn.Sequential(*layers).eval()


def _chain_pieces(input_set, two_relus=False):
    pieces = R.affine(R.relu(R.affine(_pieces(input_set), _W1, _B1)),
                      _W2, _B2)
    if two_relus:
        pieces = R.affine(R.relu(pieces), _W3, _B3)
    return pieces


_CHAIN_CELLS = [
    ("Star", "exact", {}), ("Star", "approx", {}),
    ("Star", "approx", {"relax_method": "range", "relax_factor": 0.5}),
    ("Star", "approx", {"relax_method": "area", "relax_factor": 0.5}),
    ("Star", "approx", {"relax_method": "bound", "relax_factor": 0.5}),
    ("Zono", "approx", {}), ("Box", "approx", {}),
]


# The Zono chain has one ReLU: behind the first ReLU its generators are no
# longer dyadic, and the next Linear's affine map rounds them to nearest
# (ROUND-8, the affine kernels' cell), which leaves exact-image points 1e-16
# outside (measured stage by stage: only the Linear stages lose points). A
# second ReLU adds nothing for a Zono, which stays one set.
# The exact cell with the bound pre-pass runs two ReLUs only: the first ReLU
# gets the recorded bounds of the input box, the second those of the pieces
# the first returns, so the one-ReLU chain adds nothing to it.
@pytest.mark.parametrize("kind,method,opts,two_relus", [
    pytest.param(k, m, o, t,
                 id=f"{k}-{_option_id(m, o)}-{'two' if t else 'one'}-relu")
    for k, m, o in _CHAIN_CELLS for t in (False, True)
    if not (k == "Zono" and t)] + [
    pytest.param("Star", "exact", {"precompute_bounds": "ibp"}, True,
                 id="Star-exact-pre-ibp-two-relu")])
def test_relu_chain_through_the_model_entry(kind, method, opts, two_relus):
    """Linear, ReLU, Linear (and a second ReLU and Linear, which then receives
    the several pieces of the first) through ``reach_pytorch_model``. The
    rows the ReLU emits are consumed by the next layer, so a row that is wrong
    while the bounding box is right shows downstream; the options travel
    through the model entry to the kernel (plumbing). For ``exact``, every
    returned point is an output too.

    With ``precompute_bounds`` the graph walker hands each ReLU node the
    bounds the pre-pass recorded for it, and the exact kernel widens them by
    the LP pad and intersects them with each star's own enclosure before
    they decide a phase (``_intersect_precomputed``). Catches given bounds
    that are widened inward or from the wrong end (a crossing neuron then
    reads as stably inactive and is zeroed: range), at the first ReLU and at
    the second, which receives several pieces and one set of bounds for all
    of them; the exact image must come out unchanged, in both directions.

    What this cell cannot see: a ``relax_*`` option that is dropped on the
    way. Without it the kernel applies the standard rule, which is sound, so
    containment holds either way."""
    model = _chain(two_relus)
    inputs = [flat_set(kind, _LO3, _HI3)]
    if kind == "Star":
        inputs.append(constrained_star(_LO3, _HI3, seed=2)[0])
    for k, input_set in enumerate(inputs):
        what = (f"Linear-ReLU-Linear{'-ReLU-Linear' * two_relus} {method} "
                f"{opts} on {kind} input {k} of {len(inputs)}")
        out = reach_model(model, input_set, method, what, **opts)
        pieces = _chain_pieces(input_set, two_relus)
        _assert_holds_image(pieces, out, what)
        if method == "exact":
            _assert_no_extra_point(pieces, out, what)
        points = _points(input_set, 64)
        assert_covered(out, replay_torch(model, points, (3,)), what)


# One activation called twice behind a fan-out: x in [-4, 4] x [-2, 6] (a box
# star cut by one row). h = fc1(x) is in [1/2, 7/2] x [1/2, 5/2] x [1/2, 7/2]
# x [1/2, 5/2]: the first call is stably active. fc2 of it is in [-11/4, 15/4]
# x [-3/4, 5/4] x [3/2, 9/2] x [3/2, 7/2]: the second call crosses zero on its
# first two neurons, and its last two are stably active and fix x, so an
# output point fixes the input point and a relaxation that holds outputs of
# other input points cannot hide a wrong one. Both calls have four neurons,
# so the bounds of one fit the other. aux(h) is negative everywhere.
_RE_W1 = np.array([[0.25, 0.125], [0.125, -0.125], [-0.125, 0.25],
                   [0.125, 0.125]])
_RE_B1 = np.array([1.75, 1.75, 1.5, 1.25])
_RE_W2 = np.array([[2.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_RE_B2 = np.array([-4.5, 0.25, 1.0, 1.0])
_RE_W3 = np.array([[1.0, -0.5, 0.0, 0.0], [0.0, 1.0, 0.25, 0.0],
                   [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0]])
_RE_B3 = np.array([0.125, -0.25, 0.0, 0.5])
_RE_LO, _RE_HI = np.array([-4.0, -2.0]), np.array([4.0, 6.0])


class _ReusedActivation(nn.Module):
    """x -> h = fc1(x); an auxiliary head aux(h) runs before the activation
    reads h (its value is not returned); y = fc3(act(fc2(act(h)))), where
    both calls of ``act`` are one ``nn.ReLU`` object or one function
    (``F.relu``, ``torch.relu``), so they are two graph nodes of one module
    or one function."""

    def __init__(self, how):
        super().__init__()
        self.fc1, self.fc2 = _linear(_RE_W1, _RE_B1), _linear(_RE_W2, _RE_B2)
        self.fc3 = _linear(_RE_W3, _RE_B3)
        self.aux = _linear(-np.eye(4), np.array([0.0, 0.25, -0.25, 0.0]))
        self.relu = nn.ReLU()
        self.how = how

    def act(self, h):
        if self.how == "nn.ReLU":
            return self.relu(h)
        if self.how == "F.relu":
            return nn.functional.relu(h)
        return torch.relu(h)

    def forward(self, x):
        h = self.fc1(x)
        early = self.aux(h)                                   # noqa: F841
        return self.fc3(self.act(self.fc2(self.act(h))))


# (activation, approx options, pre-pass): every approx rule once with the
# pre-pass, each spelling of the activation and of the pre-pass at least once.
_REUSED_CELLS = [("nn.ReLU", {}, "ibp"), ("F.relu", {}, "zono"),
                 ("torch.relu", _APPROX_HALF[1], "ibp"),
                 ("nn.ReLU", _APPROX_HALF[2], "zono"),
                 ("F.relu", _APPROX_HALF[3], True)]


@pytest.mark.parametrize("how,opts,pre", [
    pytest.param(h, o, p, id=f"{h}-{_option_id('approx', o)}-pre-{p}")
    for h, o, p in _REUSED_CELLS])
def test_relu_approx_called_twice_behind_a_fan_out_with_the_pre_pass(
        how, opts, pre):
    """Approx reach through ``reach_pytorch_model`` with the bound pre-pass
    (``precompute_bounds``) of a model in which one activation is called at
    two places and a second consumer of its input runs before it. The walker
    must hand each call the sets of its own input node and the bounds the
    pre-pass recorded for that node (a module node and a function node each),
    and each approx rule must use those bounds on the crossing neurons of the
    second call. Catches an activation applied to the sets computed last
    (the auxiliary head's, negative everywhere), bounds looked up by the
    module's attribute name or the function's name (the second call gets the
    first call's [1/2, ...] and its crossing neurons read stably active), a
    function node mapped to another layer, and given bounds intersected or
    unpacked in the wrong order by a rule (its crossing neurons zeroed).
    Verdict: the exact image in rationals and 128 member points of the input
    (box corners, the cut face) replayed on torch in float64, checked against
    the rows."""
    model = _ReusedActivation(how).eval()
    star, points = constrained_star(_RE_LO, _RE_HI, seed=3)
    what = (f"{how} called twice behind a fan-out, approx {opts}, "
            f"precompute_bounds={pre!r}")
    out = reach_model(model, star, "approx", what, precompute_bounds=pre,
                      **opts)
    pieces = R.relu(R.affine(_pieces(star), _RE_W1, _RE_B1))
    pieces = R.relu(R.affine(pieces, _RE_W2, _RE_B2))
    _assert_holds_image(R.affine(pieces, _RE_W3, _RE_B3), out, what)
    assert_covered(out, replay_torch(model, points, (2,)), what)


_FLAT_CLIP_KINDS = ("Star", "Zono", "Box", "Hexatope", "Octatope")


def _clip_model(tmp_path, kind, max_input):
    """An ONNX Clip with min 0 and no max, which ``load_onnx`` turns into
    ``nn.ReLU``: Gemm(3 -> 5), Clip, Gemm(5 -> 2) with the weights of
    :func:`_chain` for a flat input (Star, Zono, Box, Hexatope, Octatope),
    Clip, Flatten,
    Gemm(30 -> 3) with the weights of :func:`_image_model` for a (1, 2, 3, 5)
    image. The max input is left out by a shorter input list or by an empty
    name."""
    clip_in = ["mn"] + ([""] if max_input == "empty-name" else [])
    inits = {"mn": np.array(0.0, dtype=np.float32)}
    if kind in _FLAT_CLIP_KINDS:
        nodes = [helper.make_node("Gemm", ["X", "w1", "b1"], ["H"], transB=1),
                 helper.make_node("Clip", ["H"] + clip_in, ["R"]),
                 helper.make_node("Gemm", ["R", "w2", "b2"], ["Y"], transB=1)]
        inits.update({"w1": _W1.astype(np.float32),
                      "b1": _B1.astype(np.float32),
                      "w2": _W2.astype(np.float32),
                      "b2": _B2.astype(np.float32)})
        return save_onnx(tmp_path, nodes, [1, 3], [1, 2], inits,
                         name="gemm_clip_gemm")
    weight, bias = _gemm_weights()
    nodes = [helper.make_node("Clip", ["X"] + clip_in, ["R"]),
             helper.make_node("Flatten", ["R"], ["F"], axis=1),
             helper.make_node("Gemm", ["F", "w", "b"], ["Y"], transB=1)]
    inits.update({"w": weight.astype(np.float32),
                  "b": bias.astype(np.float32)})
    return save_onnx(tmp_path, nodes, [1, _C, _H, _W], [1, 3], inits,
                     name="clip_flatten_gemm")


# The Box input of the Clip cell: every neuron of the first Gemm is stable
# on it (n0 in [1/4, 9/4] and n3 in [7/4, 15/4] active; n1 in [-17/8, -1/8],
# n2 in [-13/4, -5/4] and n4 in [-3/2, -1/2] inactive), so a Clip loaded as
# the identity moves the Box's upper bound of the first output below the true
# maximum 49/8. On a box where neurons cross zero the interval arithmetic of
# the identity is looser than the ReLU's and holds every output.
_CLIP_BOX_LO = np.array([-0.5, 1.0, 0.25])
_CLIP_BOX_HI = np.array([0.5, 2.0, 1.25])


@pytest.mark.parametrize("kind,method,max_input", [
    ("Star", "exact", "left-out"), ("Star", "approx", "empty-name"),
    ("ImageStar", "approx", "left-out"), ("Zono", "approx", "left-out"),
    ("Box", "approx", "empty-name"), ("ImageZono", "approx", "empty-name"),
    ("Hexatope", "approx", "left-out"), ("Octatope", "approx", "empty-name")])
def test_onnx_clip_with_min_zero_through_the_model_entry(tmp_path, kind,
                                                         method, max_input):
    """An ONNX Clip with min 0 and no max through ``load_onnx`` and the model
    entry, on a flat Star and Zono (for ``approx`` a box beyond [-1, 1], four
    times the chain's; for ``exact`` the chain's, which splits into fewer
    pieces), a flat Box (``_CLIP_BOX_LO``, ``_CLIP_BOX_HI``, on which every
    neuron is stable, so that a Box result can tell the identity from the
    ReLU), a flat Hexatope and Octatope on the chain's box (their split
    pieces carry solver bounds, so the exact image is checked by the LP
    oracle, ``LP_TOL``; the identity misses all 128 onnxruntime outputs on
    that box) and a (2, 3, 5) ImageStar and ImageZono.
    Catches a load path that makes another module of such a Clip
    (the identity, or a Clip that ignores its min): the result must hold the
    exact image in rationals (for ``exact`` it must also return no point
    that is not an output) and every onnxruntime output of the member
    points, checked against the rows. A Clip with other bounds is refused
    today (``ReLU6``, ``OnnxClip``); that refusal is not this cell's
    subject."""
    path = _clip_model(tmp_path, kind, max_input)
    model = load_onnx(path)
    what = f"Clip(min=0) {method} on a {kind}, max input {max_input}"
    if kind in _FLAT_CLIP_KINDS:
        scale = (4.0 if method == "approx" and kind not in ("Hexatope",
                                                            "Octatope")
                 else 1.0)
        lo, hi = ((_CLIP_BOX_LO, _CLIP_BOX_HI) if kind == "Box"
                  else (scale * _LO3, scale * _HI3))
        input_set = flat_set(kind, lo, hi)
        pieces = _chain_pieces(input_set)
        shape = [1, 3]
    else:
        lo, hi = _image_box_few_live()
        input_set = image_set(kind, lo, hi)
        pieces = _gemm_pieces()
        shape = [1, _C, _H, _W]
    out = reach_model(model, input_set, method, what)
    if kind in ("Hexatope", "Octatope"):
        _assert_holds_image_lp(pieces, out, what)
    else:
        _assert_holds_image(pieces, out, what)
    if method == "exact":
        _assert_no_extra_point(pieces, out, what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), shape), what)


# --------------------------------------------------------------------------- #
# Image sets: (C, H, W) = (2, 3, 5)                                             #
# --------------------------------------------------------------------------- #
# The live pixels of the image box (every other pixel is a constant): two
# cross zero (slopes 1/4 and 3/4), one has a lower bound of exactly 0, one an
# upper bound of exactly 0, and the widest is positive, so that the parts a
# list cuts along it have different images.
_LIVE = {(0, 2, 1): (-1.5, 0.5), (1, 0, 3): (-0.25, 0.75),
         (1, 1, 4): (0.25, 2.75), (0, 1, 2): (0.0, 0.5),
         (1, 2, 0): (-1.0, 0.0)}
_IMAGE_CUT = ((1, 1, 4), (0.25, 0.5, 1.0, 2.75))


def _image_box_few_live():
    """A (2, 3, 5) box in which the pixels of ``_LIVE`` have positive width
    and every other pixel is a constant, positive, negative or 0: the image
    set then has five predicate variables, and every pixel holds a different
    value, so a result that reads the pixels in another order misses."""
    rng = np.random.default_rng(5)
    lo = rng.integers(-8, 9, size=(_C, _H, _W)) / 4.0
    hi = lo.copy()
    for pixel, (a, b) in _LIVE.items():
        lo[pixel], hi[pixel] = a, b
    lo[0, 0, 0] = hi[0, 0, 0] = 0.0
    return lo, hi


def _image_parts(several):
    """The box of :func:`_image_box_few_live`, or three parts of it cut along
    its widest pixel at dyadic points, as ``(lo, hi)`` pairs."""
    lo, hi = _image_box_few_live()
    if not several:
        return [(lo, hi)]
    pixel, ends = _IMAGE_CUT
    parts = []
    for a, b in zip(ends[:-1], ends[1:]):
        part_lo, part_hi = lo.copy(), hi.copy()
        part_lo[pixel], part_hi[pixel] = a, b
        parts.append((part_lo, part_hi))
    return parts


def _relu_box_vertices(lo, hi):
    """The vertices of the pieces of the exact ReLU image of the box
    [lo, hi], as (C, H, W) arrays in the network's order. The image of a box
    is the product of the coordinates' images [relu(lo), relu(hi)], and each
    piece of the exact split is a product too, with corners among those end
    values (0 is the lower end of a crossing pixel). Exact for dyadic
    bounds."""
    choices = [sorted({max(a, 0.0), max(b, 0.0)})
               for a, b in zip(lo.ravel(), hi.ravel())]
    return [np.array(p).reshape(lo.shape) for p in itertools.product(*choices)]


def _assert_inside_image_boxes(parts, sets, what):
    """Every vertex of every returned image set lies in the exact ReLU image
    of some input part, the box [relu(lo), relu(hi)], compared in the stored
    (height, width, channels) order of the set (exact method)."""
    boxes = [(_clipped(np.transpose(a, (1, 2, 0))),
              _clipped(np.transpose(b, (1, 2, 0)))) for a, b in parts]
    for k, s in enumerate(sets):
        for v in _vertices(s, what):
            if not any(all(l <= x <= u for x, l, u in zip(v, lb, ub))
                       for lb, ub in boxes):
                raise AssertionError(
                    f"{what}: returned set {k} of {len(sets)} has a vertex "
                    f"that is no output (read in the stored order)")


@pytest.mark.parametrize("several", [False, True], ids=["one-set",
                                                        "three-sets"])
@pytest.mark.parametrize("kind,method,opts", [
    ("ImageStar", "exact", {}), ("ImageStar", "approx", {}),
    ("ImageStar", "approx", {"relax_method": "area", "relax_factor": 0.5}),
    ("ImageZono", "approx", {})])
def test_relu_image_set_holds_the_exact_image(kind, method, opts, several):
    """ReLU on a (2, 3, 5) image set with five live pixels through
    ``reach_layer``: the result is an image set of the input's type, height,
    width and channels (a result rebuilt with height and width swapped, with
    (C, H, W) passed for (H, W, C), or returned as a plain set in the stored
    order, fails here), and holds the vertices of the exact image and the
    replayed outputs, read in its stored order (layout). For ``exact`` every
    vertex of every returned set is an output (checked on one set; the
    kernel's exactness is the Star cells' business). Three sets cut from the
    box check that each is processed and keeps its type."""
    parts = _image_parts(several)
    sets = [image_set(kind, a, b) for a, b in parts]
    what = f"ReLU {method} {opts} on {len(sets)} {kind}(s)"
    out = reach_one_layer(nn.ReLU(), sets, method, what, **opts)
    for s in out:
        assert type(s).__name__ == kind, (
            f"{what}: returned a {type(s).__name__}, not a {kind}")
    vertices = [v for a, b in parts for v in _relu_box_vertices(a, b)]
    assert_covered(out, vertices, what + " (vertices of the exact image)",
                   out_chw=(_C, _H, _W))
    lo, hi = _image_box_few_live()
    assert_covered(out, replay_torch(nn.ReLU(), samples(lo, hi, 64),
                                     (_C, _H, _W)),
                   what, out_chw=(_C, _H, _W))
    if method == "exact" and not several:
        _assert_inside_image_boxes(parts, out, what)


def _tied_image_star():
    """The box of :func:`_image_box_few_live` with only its two crossing
    pixels live (the other three fixed at their lower ends, so 28 constants
    of different values), as an ImageStar cut by the row ``a_q - a_p <= 1/4``
    between the predicate variables of p = (0, 2, 1), x_p = -1/2 + a_p, and
    q = (1, 0, 3), x_q = 1/4 + a_q / 2 (each pixel's variable is read from
    the stored basis). The upper end of q is reached only where p is active,
    so on the piece where p is inactive the row binds on q: the output
    (x_p, x_q) = (0, 3/4) is no true output."""
    lo, hi = _image_box_few_live()
    for pixel in ((1, 1, 4), (0, 1, 2), (1, 2, 0)):
        hi[pixel] = lo[pixel]
    image = image_set("ImageStar", lo, hi)

    def variable(c, y, x):
        return int(np.flatnonzero(image.V[y, x, c, 1:])[0])

    row = np.zeros((1, image.nVar))
    row[0, variable(0, 2, 1)] = -1.0
    row[0, variable(1, 0, 3)] = 1.0
    return ImageStar(image.V, np.vstack([image.C, row]),
                     np.vstack([image.d, [[0.25]]]), image.predicate_lb,
                     image.predicate_ub, _H, _W, _C)


def test_exact_relu_on_an_image_star_with_a_binding_row():
    """Exact ReLU on a (2, 3, 5) ImageStar whose two crossing pixels are tied
    by a row that is not a box row, through ``reach_layer``: both directions
    against the exact image in rationals, read in the stored (height, width,
    channels) order, and member points replayed on torch in the network's
    order and checked with ``out_chw``. The box cells above give every pixel
    its own variable, so a split row or input row lost or put on the wrong
    variable on the way through the flattened Star and back changes only the
    pixel it belongs to and stays exact there; here the inactive piece of
    p must keep its row x_p <= 0, or it returns points that are no output
    (not exact), and a row read against the wrong pixels loses outputs."""
    image = _tied_image_star()
    pieces = _maximal(R.relu(_pieces(image)))[0]
    what = "exact ReLU on an ImageStar with two crossing pixels tied by a row"
    out = reach_one_layer(nn.ReLU(), [image], "exact", what)
    for s in out:
        assert type(s).__name__ == "ImageStar", (
            f"{what}: returned a {type(s).__name__}, not an ImageStar")
    _assert_holds_image(pieces, out, what)
    _assert_no_extra_point(pieces, out, what)
    order = hwc_index(_C, _H, _W)
    tensors = []
    for p in _points(image, 32):
        chw = np.empty(p.size)
        chw[order] = p
        tensors.append(chw.reshape(_C, _H, _W))
    assert_covered(out, replay_torch(nn.ReLU(), tensors, (_C, _H, _W)), what,
                   out_chw=(_C, _H, _W))


@pytest.mark.parametrize("kind,method,opts", [
    ("ImageStar", "approx", {}),
    ("ImageStar", "approx", {"relax_method": "range", "relax_factor": 0.5}),
    ("ImageStar", "approx", {"relax_method": "bound", "relax_factor": 0.5}),
    ("ImageZono", "approx", {})])
def test_relu_image_set_with_every_pixel_live(kind, method, opts):
    """ReLU on a (2, 3, 5) box with its own interval per pixel, and on an
    ImageStar with a binding row that is not a box row, through
    ``reach_layer``; outputs replayed and checked against the rows in the
    stored (height, width, channels) order."""
    lo, hi = pixel_box(_C, _H, _W, seed=1)
    cases = [(image_set(kind, lo, hi), samples(lo, hi))]
    if kind == "ImageStar":
        cases.append(constrained_image_star(lo, hi, seed=1))
    for image, points in cases:
        what = f"ReLU {method} {opts} on a {kind} with every pixel live"
        out = reach_one_layer(nn.ReLU(), [image], method, what, **opts)
        assert_covered(out, replay_torch(nn.ReLU(), points, (_C, _H, _W)),
                       what, out_chw=(_C, _H, _W))


def _gemm_weights():
    rng = np.random.default_rng(11)
    weight = (rng.integers(-4, 5, size=(3, _C * _H * _W)) / 4.0)
    return weight, rng.integers(-4, 5, size=3) / 8.0


def _image_model(tmp_path):
    """Relu, Flatten, Gemm(3 x 30) as an ONNX file with dyadic weights."""
    weight, bias = _gemm_weights()
    nodes = [helper.make_node("Relu", ["X"], ["R"]),
             helper.make_node("Flatten", ["R"], ["F"], axis=1),
             helper.make_node("Gemm", ["F", "w", "b"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, [1, _C, _H, _W], [1, 3],
                     {"w": weight.astype(np.float32),
                      "b": bias.astype(np.float32)}, name="relu_flatten_gemm")
    return path, weight, bias


@functools.lru_cache(maxsize=None)
def _gemm_pieces():
    """The exact image of the box of :func:`_image_box_few_live` under Relu,
    Flatten and the Gemm of :func:`_image_model`, as rational pieces over the
    live pixels: the constant pixels' ReLU values are folded into the offset
    in fractions, in the network's (C, H, W) order."""
    lo, hi = _image_box_few_live()
    weight, bias = _gemm_weights()
    flat_lo, flat_hi = lo.reshape(-1), hi.reshape(-1)
    live = np.flatnonzero(flat_hi > flat_lo)
    const = np.flatnonzero(flat_hi == flat_lo)
    offset = [Fraction(float(bias[i])) + sum(
        (Fraction(float(weight[i, j])) * max(Fraction(float(flat_lo[j])),
                                             Fraction(0)) for j in const),
        Fraction(0)) for i in range(weight.shape[0])]
    return R.affine(R.relu(R.box(flat_lo[live], flat_hi[live])),
                    weight[:, live], offset)


@pytest.mark.parametrize("kind,method,opts", [
    ("ImageStar", "exact", {}), ("ImageStar", "approx", {}),
    ("ImageStar", "approx", {"relax_method": "range", "relax_factor": 0.5}),
    ("ImageZono", "approx", {})])
def test_relu_flatten_gemm_on_an_image_set(tmp_path, kind, method, opts):
    """ONNX Relu, Flatten, Gemm on a (1, 2, 3, 5) input through the model
    entry. Flatten reorders the stored (height, width, channels) entries to
    (channels, height, width) using the shape the ReLU result carries, so a
    result with a wrong height, width or channel count, or a plain set in the
    stored order, makes Gemm read the wrong pixels (layout). Reference:
    onnxruntime on the box's samples, and the exact image computed in the
    network's own order from the live pixels."""
    path = _image_model(tmp_path)[0]
    model = load_onnx(path)
    shape = [1, _C, _H, _W]
    lo, hi = _image_box_few_live()
    what = f"Relu-Flatten-Gemm {method} {opts} on a {kind}"
    out = reach_model(model, image_set(kind, lo, hi), method, what, **opts)
    assert_covered(out, replay_onnx(path, samples(lo, hi), shape), what)
    pieces = _gemm_pieces()
    _assert_holds_image(pieces, out, what)
    if method == "exact":
        _assert_no_extra_point(pieces, out, what)
        return
    lo_all, hi_all = pixel_box(_C, _H, _W, seed=2)
    out = reach_model(model, image_set(kind, lo_all, hi_all), method, what,
                      **opts)
    assert_covered(out, replay_onnx(path, samples(lo_all, hi_all), shape),
                   what + " (every pixel live)")


# --------------------------------------------------------------------------- #
# Refusals and defects of the unmodified library                                #
# --------------------------------------------------------------------------- #
_EXACT_ON_AN_ENCLOSURE = known_bug(
    18, "reach_layer with method='exact' on a Zono or ImageZono runs the "
        "over-approximating ReLU relaxation and returns it as the exact "
        "result; the name 'exact' states a guarantee the result does not have",
    raises=DID_NOT_RAISE)


@pytest.mark.parametrize("kind", [
    pytest.param("Zono", marks=_EXACT_ON_AN_ENCLOSURE),
    pytest.param("ImageZono", marks=_EXACT_ON_AN_ENCLOSURE)])
def test_exact_relu_on_a_zonotope_refuses(kind):
    """A zonotope cannot hold the union the exact ReLU image is (the result
    holds -1/2 for the input [-1, 1], whose image is [0, 1]), so
    ``method='exact'`` must refuse and name ``"approx"`` (DESIGN section 7).
    The neighbour cells that pass are Box and Star under ``exact``
    (``test_relu_box_list_and_boundary_values_hold_the_exact_image`` and the
    exact Star cells), where the result is checked exact in both
    directions, and the model entry below, which refuses."""
    if kind == "Zono":
        input_set = Zono(np.zeros((1, 1)), np.ones((1, 1)))
    else:
        lo, hi = pixel_box(_C, _H, _W)
        input_set = image_set("ImageZono", lo, hi)
    with pytest.raises(NotImplementedError):
        reach_layer(nn.ReLU(), [input_set], "exact")


@pytest.mark.parametrize("kind", ["Zono", "Box", "ImageZono"])
def test_exact_through_the_model_entry_refuses_a_set_without_exact_rules(kind):
    """The model entry refuses ``method='exact'`` for a Zono, a Box and a
    (2, 3, 5) ImageZono input instead of answering with a relaxation
    (``ValueError`` today, ``UnsupportedError`` later; DESIGN section 7: the
    refusal names ``"approx"``, which the match checks, so that an unrelated
    ``ValueError`` raised on the way does not pass).
    Catches a model entry that runs the walk under ``exact`` for these sets
    (the Zono kernel would return its relaxation as the exact result)."""
    if kind == "ImageZono":
        model = nn.Sequential(nn.ReLU(), nn.Flatten()).eval()
        input_set = image_set(kind, *pixel_box(_C, _H, _W))
    else:
        model, input_set = _chain(), flat_set(kind, _LO3, _HI3)
    with pytest.raises((ValueError, NotImplementedError), match="'approx'"):
        reach_pytorch_model(model, input_set, method="exact")


_MIXED_LIST_CRASH = known_bug(
    23, "reach_layer with a ReLU on a list whose sets are of two families "
        "(a Box after a Zono or an ImageZono, a Zono after a Box, a Zono or "
        "a Box after a Star) crashes with AttributeError in the kernel of "
        "the first set's family (_relu_single_zono reads get_bounds, "
        "relu_box reads lb, _relu_single_star_approx and "
        "_relu_single_star_exact read V and Z) where a refusal is due",
    raises=AttributeError)


_HEXOCT_LIST_CRASH = (
    "reach_layer with a ReLU on a list led by {} whose second set is of "
    "another family and has a neuron that crosses zero: the {} kernel "
    "splits that set with its own call {}.intersect_half_space(H, g, "
    "solver=...), which {}, where a refusal is due (registry REACH-6)")


@pytest.mark.parametrize("first,second,method", [
    pytest.param("Zono", "ImageZono", "approx", id="zono-then-imagezono"),
    pytest.param("Zono", "Box", "approx", marks=_MIXED_LIST_CRASH,
                 id="zono-then-box"),
    pytest.param("Box", "Zono", "approx", marks=_MIXED_LIST_CRASH,
                 id="box-then-zono"),
    pytest.param("Star", "ImageStar", "approx", id="star-then-imagestar"),
    pytest.param("Star", "Zono", "approx", marks=_MIXED_LIST_CRASH,
                 id="star-then-zono"),
    pytest.param("Star", "Box", "exact", marks=_MIXED_LIST_CRASH,
                 id="star-then-box-exact"),
    pytest.param("Hexatope", "Octatope", "approx",
                 id="hexatope-then-octatope"),
    pytest.param("Hexatope", "Star", "approx", marks=known_bug(
        23, _HEXOCT_LIST_CRASH.format(
            "a Hexatope", "Hexatope", "s",
            "a Star does not take (TypeError; a Zono or ImageZono the same)"),
        raises=TypeError), id="hexatope-then-star"),
    pytest.param("Octatope", "Box", "approx", marks=known_bug(
        23, _HEXOCT_LIST_CRASH.format(
            "an Octatope", "Octatope", "s",
            "a Box does not have (AttributeError; a Star, ImageStar, Zono or "
            "ImageZono raises TypeError)"),
        raises=AttributeError), id="octatope-then-box")])
def test_relu_on_a_list_mixing_set_families_refuses_or_holds_every_output(
        first, second, method):
    """One ``reach_layer`` call with two input sets of two families, the
    first over the box [-1, 1] x [1/4, 3/2] x [-2, -1/2] x [-1/2, 3/4] and
    the second over that box moved by (3/2, 1, 2, 1), whose outputs are all
    different, as (1, 2, 2) tensors: the call must refuse
    (``NotImplementedError`` or ``ValueError``) or return sets that hold
    every true output of 128 samples of each box replayed on torch in
    float64 (LP oracle against the rows), so a set dropped from the list
    fails as well. The dispatcher picks the lane by the first set; the Zono
    lane answers an ImageZono after a Zono and the Star lane an ImageStar
    after a Star (the passing neighbours, the same reference); a Box after a
    Zono, a Zono after a Box, a Zono after a Star under ``approx`` and a Box
    after a Star under ``exact`` are the defect in the marker, one per
    kernel that is handed a set of another family.

    The Hexatope and Octatope kernels hand back unchanged a set with no
    crossing neuron, which every family survives, so a list led by one of
    them gets a second box that crosses zero (the first moved by (1/2, -1,
    2, 0)): the kernel then splits the second set by calling its
    ``intersect_half_space`` with the keyword ``solver``. A Hexatope then an
    Octatope is answered (the passing neighbour); a Hexatope then a Star and
    an Octatope then a Box crash, one marked order per kernel (REACH-6). The
    method is not read by these kernels, so ``exact`` runs the same code.
    The orders with a Hexatope or Octatope second crash in the Star, Zono
    and Box kernels already marked here."""
    lo = np.array([-1.0, 0.25, -2.0, -0.5]).reshape(1, 2, 2)
    hi = np.array([1.0, 1.5, -0.5, 0.75]).reshape(1, 2, 2)
    shift = np.array([1.5, 1.0, 2.0, 1.0]).reshape(1, 2, 2)
    if first in ("Hexatope", "Octatope"):
        shift = np.array([0.5, -1.0, 2.0, 0.0]).reshape(1, 2, 2)
    boxes = [(lo, hi), (lo + shift, hi + shift)]
    sets = [image_set(k, a, b) if k.startswith("Image") else flat_set(k, a, b)
            for k, (a, b) in zip((first, second), boxes)]
    what = f"ReLU {method} on a list of a {first} and a {second}"
    try:
        out = reach_one_layer(nn.ReLU(), sets, method, what)
    except (NotImplementedError, ValueError):
        return
    points = [x for a, b in boxes for x in samples(a, b)]
    assert_covered(out, replay_torch(nn.ReLU(), points, (1, 2, 2)), what,
                   out_chw=(1, 2, 2))


def _zero_dimensional_star():
    """A Star with no output entry over one predicate variable in [-1, 1]:
    not empty (it is the one point of a zero-dimensional space), and what a
    tensor chunk with zero entries becomes (``GRAPH-5``)."""
    return Star(np.zeros((0, 2)), np.array([[1.0], [-1.0]]), np.ones((2, 1)),
                -np.ones((1, 1)), np.ones((1, 1)))


_ZERO_DIM_DROPPED = known_bug(
    7, "ReLU returns no set for a non-empty Star with zero entries (the "
       "kernels skip a star of dimension 0), an empty result that no "
       "certificate backs and that a verdict reads as a proof (EMPTY-6)")


@pytest.mark.parametrize("dim", [
    pytest.param(0, marks=_ZERO_DIM_DROPPED, id="zero-entries"),
    pytest.param(1, id="one-entry")])
@pytest.mark.parametrize("method,opts", [
    ("exact", {}), ("approx", {}),
    ("approx", {"relax_method": "range", "relax_factor": 0.5}),
    ("approx", {"relax_method": "area", "relax_factor": 0.5}),
    ("approx", {"relax_method": "bound", "relax_factor": 0.5})])
def test_relu_never_drops_a_non_empty_star(method, opts, dim):
    """A Star that is not empty is never answered with no set: ReLU returns a
    set for it, or refuses (``ValueError`` is the proposed fix of
    ``EMPTY-6``). The neighbour is the same predicate with one entry, which
    is answered with a set holding its outputs, under every method; it may
    not refuse (a legal one-entry Star refused with ``ValueError`` would pass
    here and check nothing)."""
    if dim == 0:
        star = _zero_dimensional_star()
    else:
        star = Star(np.array([[0.25, 1.0]]), np.array([[1.0], [-1.0]]),
                    np.ones((2, 1)), -np.ones((1, 1)), np.ones((1, 1)))
    try:
        out = reach_layer(nn.ReLU(), [star], method, **opts)
    except ValueError:
        if dim == 1:
            raise
        return
    assert len(out) >= 1, (
        f"ReLU {method} {opts} returned no set for a non-empty Star with "
        f"{dim} entries")
    if dim == 1:
        _assert_holds_image(R.relu(_pieces(star)), out, f"ReLU {method}")


_ZONO_RELU_ROUNDED_TO_NEAREST = known_bug(
    "7b", "the Zono ReLU relaxation computes its slope, error term and scaled "
          "row in round-to-nearest, so the true image of an end point of a "
          "crossing neuron lies outside the returned zonotope by up to "
          "2.8e-16 (ROUND-10)")


_FLOAT32_ZONO_RELU_IN_FLOAT32 = known_bug(
    7, "approx ReLU on a Zono whose arrays are float32 (the Zono constructor "
       "keeps a float type) writes the new centre and the scaled row into "
       "float32 copies of the input's arrays, so they are rounded to "
       "float32: on numbers for which the kernel's float64 arithmetic is "
       "exact the true image of an end point lies outside the returned "
       "zonotope by up to 1.5e-7, against 0 for the same numbers stored as "
       "float64")


@pytest.mark.parametrize("data", [
    pytest.param("float", marks=_ZONO_RELU_ROUNDED_TO_NEAREST),
    pytest.param("dyadic"),
    pytest.param("float32 numbers"),
    pytest.param("float32 numbers, float32",
                 marks=_FLOAT32_ZONO_RELU_IN_FLOAT32)])
def test_relu_zono_holds_the_end_points_in_exact_arithmetic(data):
    """One-dimensional zonotopes ``c + r a`` that cross zero. The true outputs
    of ``a = -1`` and ``a = 1`` must be members of the returned zonotope in
    exact arithmetic. With non-dyadic ``c`` and ``r`` the slope and error are
    rounded, and an outward rounding is needed; the dyadic neighbour has
    exact slopes and passes with the same reference. "float32 numbers" are
    centres with a full float32 mantissa and ``|c|`` between ``r / 2`` and
    ``r``, with ``r`` 1, 2 or 4: every product the kernel forms is a float64
    number, so the item passes when the arrays are float64; stored as float32
    arrays (a Zono keeps the float type it is given) the same numbers show
    the kernel writing its result into float32 copies (strict expected
    failure)."""
    rng = np.random.default_rng(12)
    dtype = np.float32 if data.endswith(", float32") else np.float64
    cases = []
    while len(cases) < 40:
        if data == "float":
            c, r = float(rng.uniform(-1, 1)), float(rng.uniform(0.1, 2))
        elif data.startswith("float32 numbers"):
            r = float(2.0 ** rng.integers(0, 3))
            c = float(np.float32(
                rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0]) * r))
        else:
            r = float(2.0 ** rng.integers(-3, 2))
            c = float(rng.integers(-15, 16)) * r / 16
        if c - r < 0 < c + r:
            cases.append((c, r))
    missed = []
    for c, r in cases:
        zono = Zono(np.array([[c]], dtype=dtype), np.array([[r]], dtype=dtype))
        out = reach_layer(nn.ReLU(), [zono], "approx")
        if len(out) != 1:
            pytest.fail(f"ReLU on one Zono returned {len(out)} sets")
        member = R.Membership(R.star(out[0]))
        for a in (-1, 1):
            y = max(Fraction(0), Fraction(c) + Fraction(r) * a)
            if not member([y]):
                missed.append((c, r, a))
    assert not missed, (
        f"{len(missed)} of {2 * len(cases)} end-point outputs are outside "
        f"the returned zonotope; first (c, r, a) = {missed[0]}")
