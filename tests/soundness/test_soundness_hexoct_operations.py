"""Soundness and exactness cells for the Hexatope and Octatope operations that
every layer shares: ``from_bounds``, ``affine_map`` (directly and through the
Linear rule of ``reach_layer`` and of the model entry), ``is_empty_set``,
``contains`` and ``sample``. ``to_star`` and its verdict route
(``verify_specification``) are in ``test_soundness_to_star.py``; the range
queries, ``intersect_half_space`` and the ReLU split belong to
``test_soundness_hexoct_sets.py`` and ``test_soundness_hexoct_relu.py``.

The promise checked here: ``from_bounds`` builds exactly the box it is given,
or refuses; ``affine_map`` returns exactly the image of the set under the map
(no image point outside, no returned point without a preimage); a refusal is
a ``ValueError`` for an argument that cannot be read; ``is_empty_set`` says
empty exactly when the stored rows have no solution; ``contains`` says True
exactly for the points of the set (up to its tolerance); ``sample`` returns
points of the set only, as many as asked for, as an (n, dim) array.

Oracle. A set means what its stored arrays say, read by ``tests/_oracles``
(``membership.rows`` through ``rational.star``) and never by a method of the
set: a Hexatope is ``center + generators x`` for every ``x`` that keeps each
stored difference row ``x_i - x_j <= b`` with the anchor ``x_0`` pinned to 0;
an Octatope is ``center + generators x`` for every ``x`` that keeps each
stored UTVPI row. Vertices are enumerated exactly (``rational.vertices``),
images are computed in Fractions (``rational.affine``), membership and
emptiness are decided by the exact simplex of ``rational`` with zero slack.
Every number is dyadic, and the long-mantissa coordinate (1 + 2^-30, a value
float32 cannot hold) still leaves the library's float64 arithmetic exact, so
no slack is needed anywhere except for ``sample``, whose own rejection test
admits 1e-10 in the kernel variables (there ``membership.uncovered`` with
``LP_TOL``).

Inputs: five entries with bounds well beyond plus and minus 1 and intervals
wider than 2, one straddling zero, one positive, one negative, one of zero
width and one with a long mantissa; a kernel cut by a difference row of two
kernel variables (never the Hexatope anchor, which is registry HEXOCT-1 and
HEXOCT-2 in ``contracts/test_set_protocol.py``); kernels with extra rows
built with the public constructors (the Octatope ones with sum rows and a
row with a zero first coefficient); a set with more kernel variables than
entries and one with more entries than kernel variables.
"""

from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Hexatope, Octatope
from n2v.sets.hexatope import DifferenceConstraintSystem
from n2v.sets.octatope import UTVPIConstraintSystem
from tests._oracles import rational as R
from tests._oracles.membership import LP_TOL, distance, uncovered
from tests.cell_support import assert_covered, reach_model, reach_one_layer, replay_torch
from tests.known_bugs import known_bug

KINDS = ("Hexatope", "Octatope")
CLASSES = {"Hexatope": Hexatope, "Octatope": Octatope}

# Straddling, positive, negative, zero width, long mantissa (not a float32).
_LO = np.array([-3.0, 0.5, -6.25, 1.5, 1.0 + 2.0 ** -30])
_HI = np.array([5.0, 4.5, -2.25, 1.5, 3.0 + 2.0 ** -29])

# Maps with negative entries, magnitudes above and below 1, and one entry with
# a long mantissa in the column of a short coordinate, so the float64 products
# stay exact while a float32 copy of W would not.
_W_DOWN = np.array([[1.0 + 2.0 ** -30, -1.5, 0.25, 2.0, -0.5],
                    [-2.0, 0.5, 1.25, -0.75, 1.0],
                    [0.5, 1.0, -0.5, 0.25, -1.25]])
_B_DOWN = np.array([0.375, -2.5, 1.125])
_W_FLAT = np.array([[1.0, 3.0, -0.5, 0.0, 0.25],
                    [0.25, -1.0, 2.0, 0.5, -1.5]])
_B_FLAT = np.array([0.5, -1.0])
# Six outputs from five entries; the last row is the sum of the first two, so
# a point that breaks that sum lies off the set's affine hull.
_W_UP = np.array([[1.0, 0.0, 0.5, 0.0, 0.0],
                  [0.0, -1.0, 0.0, 0.0, 0.25],
                  [0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0],
                  [-0.5, 0.0, 0.0, 0.0, 1.0],
                  [1.0, -1.0, 0.5, 0.0, 0.25]])

# The cut 0.25 y0 - 0.5 y1 <= -0.25 is, on the half-widths 4 and 2, the
# difference row x_a - x_b <= 3/4 of the two kernel variables of y0 and y1.
_CUT_H = np.array([[0.25, -0.5, 0.0, 0.0, 0.0]])
_CUT_G = np.array([[-0.25]])


# --------------------------------------------------------------------------- #
# Reading a set exactly                                                         #
# --------------------------------------------------------------------------- #
def _kernel(s):
    """``(c, G, A, b)`` of ``{c + G a : A a <= b}`` in Fractions, read by
    ``rational.star`` from the stored arrays, with a pinned predicate variable
    (the Hexatope anchor) replaced by its value and every other finite bound
    written as a row."""
    c, G, C, d, lb, ub = R.star(s)
    n = len(lb)
    pinned = {j: lb[j] for j in range(n)
              if lb[j] is not None and ub[j] is not None and lb[j] == ub[j]}
    free = [j for j in range(n) if j not in pinned]
    zero = Fraction(0)
    c = [ci + sum((row[j] * v for j, v in pinned.items()), zero)
         for ci, row in zip(c, G)]
    G = [[row[j] for j in free] for row in G]
    A = [[row[j] for j in free] for row in C]
    b = [di - sum((row[j] * v for j, v in pinned.items()), zero)
         for di, row in zip(d, C)]
    for k, j in enumerate(free):
        unit = [Fraction(int(m == k)) for m in range(len(free))]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    return c, G, A, b


def _bounded(A):
    """Is ``{a : A a <= b}`` bounded? At once when every variable has a row
    ``a_j <= .`` and a row ``-a_j <= .`` (every kernel built here); otherwise
    by ``rational.is_bounded``, which enumerates a cone and is slow."""
    n = len(A[0])
    rows = {tuple(row) for row in A}
    unit = [tuple(Fraction(int(k == j)) for k in range(n)) for j in range(n)]
    if all(u in rows and tuple(-v for v in u) in rows for u in unit):
        return True
    return R.is_bounded(A)


def _pieces(s):
    """The set as one piece of ``rational``: its kernel polytope mapped by
    its centre and generators. A setup failure when the kernel is unbounded."""
    c, G, A, b = _kernel(s)
    if not A or not _bounded(A):
        pytest.fail("test setup: the kernel is unbounded")
    n = len(A[0])
    identity = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    return R.affine([R.Piece(A, b, identity, [Fraction(0)] * n)], G, c)


def _vertices(s):
    """The exact vertices of the set; a setup failure when it is empty."""
    found = sorted({y for piece in _pieces(s) for y in piece.image()})
    if not found:
        pytest.fail("test setup: the set is empty")
    return found


def _floats(point):
    return np.array([float(v) for v in point])


def _snapshot(s):
    rows = s.dcs.constraints if isinstance(s, Hexatope) else s.utvpi.constraints
    return (s.center.copy(), s.generators.copy(), [tuple(vars(r).values()) for r in rows])


def _assert_unchanged(s, before, what):
    after = _snapshot(s)
    if not (np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
            and before[2] == after[2]):
        pytest.fail(f"{what}: the input set was changed")


# --------------------------------------------------------------------------- #
# The families of sets                                                          #
# --------------------------------------------------------------------------- #
def _hexatope(center, generators, rows):
    generators = np.asarray(generators, dtype=np.float64)
    dcs = DifferenceConstraintSystem(generators.shape[1])
    for i, j, b in rows:
        dcs.add_constraint(i, j, b)
    return Hexatope(np.asarray(center, dtype=np.float64), generators, dcs)


def _octatope(center, generators, rows):
    generators = np.asarray(generators, dtype=np.float64)
    utvpi = UTVPIConstraintSystem(generators.shape[1])
    for row in rows:
        utvpi.add_constraint(*row)
    return Octatope(np.asarray(center, dtype=np.float64), generators, utvpi)


_ROWS_CENTER = [0.625, -1.5, 3.25]


def _rows_set(kind):
    """A kernel built with the public constructor: the box rows and extra
    rows that bind (Hexatope: difference rows of kernel variables, never the
    anchor; Octatope: sum rows, a difference row and a one-variable row with
    a zero first coefficient). The kernel point 0 keeps every row."""
    if kind == "Hexatope":
        box = [r for i in (1, 2, 3) for r in ((i, 0, 1.0), (0, i, 1.0))]
        return _hexatope(_ROWS_CENTER,
                         [[0.0, 2.0, -0.5, 1.5], [0.0, -1.0, 3.0, 0.25],
                          [0.0, 0.75, 0.5, -2.0]],
                         box + [(1, 2, 0.5), (3, 1, 0.25), (2, 3, 1.0)])
    box = [r for i in range(3) for r in ((i, i, 1, 0, 1.0), (i, i, -1, 0, 1.0))]
    return _octatope(_ROWS_CENTER,
                     [[2.0, -0.5, 1.5], [-1.0, 3.0, 0.25], [0.75, 0.5, -2.0]],
                     box + [(0, 1, 1, 1, 0.5), (1, 2, -1, 1, 0.75),
                            (0, 2, -1, -1, 1.25), (2, 2, 0, -1, 0.5)])


def _family(kind, name):
    cls = CLASSES[kind]
    if name == "box":
        return cls.from_bounds(_LO, _HI)
    if name == "cut":
        return cls.from_bounds(_LO, _HI).intersect_half_space(_CUT_H, _CUT_G)
    if name == "rows":
        return _rows_set(kind)
    if name == "flat":       # more kernel variables than entries
        return cls.from_bounds(_LO, _HI).affine_map(_W_FLAT, _B_FLAT)
    if name == "raised":     # more entries than kernel variables
        return cls.from_bounds(_LO, _HI).affine_map(_W_UP)
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# from_bounds                                                                   #
# --------------------------------------------------------------------------- #
def _corners(lo, hi):
    grids = [sorted({Fraction(float(a)), Fraction(float(b))}) for a, b in zip(lo, hi)]
    corners = [[]]
    for grid in grids:
        corners = [c + [v] for c in corners for v in grid]
    return corners


@pytest.mark.parametrize("layout", ["flat", "column"])
@pytest.mark.parametrize("kind", KINDS)
def test_from_bounds_is_exactly_the_box(kind, layout):
    """``from_bounds`` on five dyadic coordinates (straddling, positive,
    negative, zero width, and one with a long mantissa) given as flat or as
    column vectors is the box itself: every corner is a member and every
    vertex of the set lies in the box, exactly; the state bounds are the
    given bounds. Catches a generator written into the anchor column (the
    first coordinate loses its width), a centre computed as (ub - lb) / 2 or
    as lb + ub / 2, a row bound other than 1, the data stored in float32 (the
    long-mantissa coordinate moves), and state bounds swapped."""
    lo, hi = _LO, _HI
    if layout == "column":
        lo, hi = lo.reshape(-1, 1), hi.reshape(-1, 1)
    s = CLASSES[kind].from_bounds(lo, hi)
    assert s.dim == 5
    member = R.Membership(R.star(s))
    missed = [p for p in _corners(_LO, _HI) if not member(p)]
    assert not missed, (f"{kind}.from_bounds: {len(missed)} corners of its box "
                        f"are not members; first {[float(v) for v in missed[0]]}")
    low = [Fraction(float(v)) for v in _LO]
    high = [Fraction(float(v)) for v in _HI]
    outside = [v for v in _vertices(s)
               if not all(a <= x <= b for a, x, b in zip(low, v, high))]
    assert not outside, (f"{kind}.from_bounds: the set has the vertex "
                         f"{[float(x) for x in outside[0]]} outside its box")
    assert np.array_equal(np.ravel(s.state_lb), _LO)
    assert np.array_equal(np.ravel(s.state_ub), _HI)


@pytest.mark.parametrize("lengths", [(2, 3), (3, 1)])
@pytest.mark.parametrize("kind", KINDS)
def test_from_bounds_refuses_bounds_of_different_lengths(kind, lengths):
    """Bounds of different lengths cannot be read as a box: ``ValueError``,
    whichever is the longer. Catches a constructor that reads the length of
    ``lb`` only and pads or cuts ``ub``."""
    with pytest.raises(ValueError):
        CLASSES[kind].from_bounds(np.zeros(lengths[0]), np.ones(lengths[1]))


_INVERTED = known_bug(
    9, "Hexatope.from_bounds and Octatope.from_bounds accept a lower bound "
       "above the upper bound and build the box with the two ends exchanged, "
       "a set with points, where Star.from_bounds and Box refuse with "
       "ValueError")


@pytest.mark.parametrize("order", ["ordered", pytest.param("inverted", marks=_INVERTED)])
@pytest.mark.parametrize("kind", KINDS)
def test_from_bounds_of_crossed_bounds_refuses_or_is_empty(kind, order):
    """A lower bound above the upper bound describes no box. ``from_bounds``
    must refuse it (``ValueError``, as ``Star.from_bounds`` and ``Box`` do) or
    return an empty set; on ordered bounds it is the box (the passing
    neighbour, same reference). Catches the ends exchanged silently."""
    lo, hi = np.array([-2.0, 0.5]), np.array([3.0, 1.5])
    if order == "inverted":
        lo, hi = np.array([3.0, 0.5]), np.array([-2.0, 1.5])
    try:
        s = CLASSES[kind].from_bounds(lo, hi)
    except ValueError:
        return
    c, G, A, b = _kernel(s)
    if order == "inverted":
        assert R.feasible((), (), A, b) is None, (
            f"{kind}.from_bounds({lo}, {hi}) returned a set with points")
    else:
        assert sorted(_vertices(s)) == sorted(tuple(p) for p in _corners(lo, hi))


_INFINITE_END = known_bug(
    9, "Hexatope.from_bounds and Octatope.from_bounds take an infinite (or "
       "NaN) bound into (lb + ub) / 2 and (ub - lb) / 2 and return a set whose "
       "centre is infinite and whose generator is infinite or NaN: it holds "
       "no finite point of the box and its ranges are NaN; no refusal")


@pytest.mark.parametrize("end", ["finite", pytest.param("infinite", marks=_INFINITE_END)])
@pytest.mark.parametrize("kind", KINDS)
def test_from_bounds_with_an_infinite_end_refuses_or_holds_the_box(kind, end):
    """The box (-inf, 2] x [0.5, 1.5] has the finite points below. A Hexatope
    or Octatope cannot hold a box with an infinite end in its [-1, 1] kernel,
    so ``from_bounds`` must refuse (``ValueError`` or ``NotImplementedError``)
    or return a finite set that holds those points; the finite box [-4, 2] x
    [0.5, 1.5] is the passing neighbour. Catches a NaN or infinite set
    returned silently (the reach of such a set is NaN throughout)."""
    lo = np.array([-np.inf if end == "infinite" else -4.0, 0.5])
    hi = np.array([2.0, 1.5])
    try:
        s = CLASSES[kind].from_bounds(lo, hi)
    except (ValueError, NotImplementedError):
        return
    assert np.all(np.isfinite(s.center)) and np.all(np.isfinite(s.generators)), (
        f"{kind}.from_bounds({lo}, {hi}): centre {s.center}, generators "
        f"{s.generators.tolist()}")
    for p in ([-4.0, 0.5], [2.0, 1.5], [0.0, 1.0], [-4.0, 1.5]):
        assert distance(s, np.array(p)) <= LP_TOL, p


# --------------------------------------------------------------------------- #
# affine_map                                                                    #
# --------------------------------------------------------------------------- #
def _maps(family):
    if family == "box":
        return _W_DOWN, _B_DOWN                   # 5 -> 3
    if family == "cut":
        return _W_UP, np.arange(6) * 0.75 - 2.0   # 5 -> 6
    if family == "rows":
        return np.array([[0.5, -2.0, 1.0], [1.5, 0.25, -1.0],
                         [-1.0, 0.0, 0.75]]), np.array([-0.5, 2.25, 0.125])
    raise ValueError(family)


def _assert_exact_image(s, out, W, b, what):
    """``out`` is exactly the image of ``s`` under ``y -> W y + b``: every
    vertex of the image is in ``out`` and every vertex of ``out`` has a
    preimage, both decided in Fractions with zero slack."""
    image = R.affine(_pieces(s), W, b)
    member = R.Membership(R.star(out))
    escaped = [y for piece in image for y in piece.image() if not member(y)]
    assert not escaped, (f"{what}: {len(escaped)} vertices of the exact image "
                         f"are not in the result; first "
                         f"{[float(v) for v in escaped[0]]}")
    extra = R.unreached(image, _vertices(out))
    assert not extra, (f"{what}: the result has the vertex "
                       f"{[float(v) for v in extra[0][1]]}, which is the image "
                       f"of no point of the set")


@pytest.mark.parametrize("family,bias", [("box", "none"), ("box", "column"),
                                         ("cut", "flat"), ("rows", "column")])
@pytest.mark.parametrize("kind", KINDS)
def test_affine_map_is_the_exact_image(kind, family, bias):
    """``affine_map`` of a box with the long-mantissa coordinate (5 -> 3),
    of a cut box (5 -> 6) and of a kernel with extra rows (3 -> 3), with no
    bias, a flat bias and a column bias of distinct entries of both signs
    (no bias and a bias are the two paths; flat and column the two shapes the
    docstring names): the result is exactly the image (both directions, zero
    slack) and the input set is unchanged. Catches the bias overwriting the mapped centre or
    left out, the bias read from its first entry only, generators stored as
    absolute values or mapped by |W| (a mirrored set with the same per-entry
    ranges), W transposed, W or the data rounded to float32, and the kernel
    rows dropped or shared with the input."""
    s = _family(kind, family)
    W, b = _maps(family)
    before = _snapshot(s)
    if bias == "none":
        out = s.affine_map(W)
        b = None
    else:
        out = s.affine_map(W, b if bias == "flat" else b.reshape(-1, 1))
    _assert_unchanged(s, before, f"{kind}.affine_map")
    assert out.dim == W.shape[0], f"{kind}.affine_map: {out.dim} entries"
    _assert_exact_image(s, out, W, b, f"{kind}.affine_map of the {family!r} set, bias {bias}")


_ONE_DIMENSIONAL_W = known_bug(
    23, "Hexatope.affine_map and Octatope.affine_map read W.shape[1] before any "
        "check, so a one-dimensional W raises IndexError ('tuple index out of "
        "range') where a ValueError (or the map by the row W) is due",
    raises=IndexError)


@pytest.mark.parametrize("case", ["bias-of-the-wrong-length",
                                  pytest.param("one-dimensional-W", marks=_ONE_DIMENSIONAL_W)])
@pytest.mark.parametrize("kind", KINDS)
def test_affine_map_refuses_what_it_cannot_read(kind, case):
    """A bias of two entries for three outputs cannot be read: ``ValueError``.
    A one-dimensional W is either refused with ``ValueError`` or read as the
    one-row matrix it spells, whose image is then exact. Catches a bias that
    is silently cut or padded, and a crash where a refusal is due."""
    s = CLASSES[kind].from_bounds(np.array([-1.0, 2.0]), np.array([3.0, 5.0]))
    if case == "bias-of-the-wrong-length":
        with pytest.raises(ValueError):
            s.affine_map(np.ones((3, 2)), np.array([1.0, 2.0]))
        return
    try:
        out = s.affine_map(np.array([0.5, -1.5]))
    except ValueError:
        return
    _assert_exact_image(s, out, np.array([[0.5, -1.5]]), None,
                        f"{kind}.affine_map of a one-dimensional W")


_W1 = np.array([[0.5, -1.0, 0.25, 0.0, 1.5], [-2.0, 0.75, 0.0, 1.0, -0.5],
                [1.0, 0.5, -1.25, -0.5, 0.25], [0.0, -0.25, 2.0, 0.5, -1.0]])
_B1 = np.array([0.25, -1.5, 0.75, 2.0])
_W2 = np.array([[1.0, -0.5, 0.25, -2.0], [-0.75, 1.5, 1.0, 0.5]])
_B2 = np.array([-0.125, 1.0])


def _linear(W, b):
    layer = nn.Linear(W.shape[1], W.shape[0], dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(W))
        layer.bias.copy_(torch.from_numpy(b))
    return layer


def _cut_members(n=160):
    """Points of the cut box, computed without n2v: box corners, quarter-grid
    points and points on the cut face, kept when 0.25 y0 - 0.5 y1 <= -0.25."""
    rng = np.random.default_rng(5)
    points = []
    while len(points) < n:
        k = len(points) % 3
        steps = (4 * rng.integers(0, 2, size=5) if k == 0
                 else rng.integers(0, 5, size=5))
        y = _LO + (_HI - _LO) * steps / 4.0
        if k == 2:   # move y1 onto the cut face where it stays in its interval
            y1 = (0.25 * y[0] + 0.25) / 0.5
            if _LO[1] <= y1 <= _HI[1]:
                y[1] = y1
        if 0.25 * y[0] - 0.5 * y[1] <= -0.25:
            points.append(y)
    return points


@pytest.mark.parametrize("entry", ["layer", "model"])
@pytest.mark.parametrize("kind", KINDS)
def test_linear_reach_holds_the_exact_image(kind, entry):
    """The public routes of ``affine_map``: a float64 ``nn.Linear`` (5 -> 4,
    bias) through ``reach_layer``, and Linear then Linear (4 -> 2) through
    ``reach_pytorch_model``, on the cut box. The result is exactly the image
    (both directions, zero slack), 160 member points replayed on torch in
    float64 are members by the result's own rows, and the input set is
    unchanged. Catches what ``test_affine_map_is_the_exact_image`` catches,
    on the route that hands the bias over as a column."""
    s = _family(kind, "cut")
    what = f"Linear on a cut {kind} through the {entry}"
    if entry == "layer":
        model = _linear(_W1, _B1)
        out = reach_one_layer(model, [s], "approx", what)
        maps = [(_W1, _B1)]
    else:
        model = nn.Sequential(_linear(_W1, _B1), _linear(_W2, _B2))
        out = reach_model(model, s, "approx", what)
        maps = [(_W1, _B1), (_W2, _B2)]
    assert len(out) == 1, f"{what}: {len(out)} sets returned for one"
    W, b = maps[0]
    for W2, b2 in maps[1:]:
        W, b = W2 @ W, W2 @ b + b2
    _assert_exact_image(s, out[0], W, b, what)
    points = _cut_members()
    assert_covered(out, replay_torch(model, points, (1, 5)), what)


# --------------------------------------------------------------------------- #
# is_empty_set                                                                  #
# --------------------------------------------------------------------------- #
def _hexatope_kernel(extra):
    """Two kernel variables x1, x2 in [-1, 1] against the anchor, generators
    of half-width 4 and 2, and the extra difference rows."""
    box = [r for i in (1, 2) for r in ((i, 0, 1.0), (0, i, 1.0))]
    return _hexatope([1.0, -0.5], [[0.0, 4.0, 0.0], [0.0, 0.0, 2.0]], box + extra)


def _octatope_kernel(extra):
    box = [r for i in (0, 1) for r in ((i, i, 1, 0, 1.0), (i, i, -1, 0, 1.0))]
    return _octatope([1.0, -0.5], [[4.0, 0.0], [0.0, 2.0]], box + extra)


_EMPTINESS = {
    # name: (Hexatope extra rows, Octatope extra rows, empty?)
    "box": ([], [], False),
    "one-point": ([(1, 2, -2.0)], [(0, 1, 1, 1, -2.0)], False),
    "cycle": ([(1, 2, -0.5), (2, 1, 0.25)], [(0, 1, 1, -1, -0.5), (1, 0, 1, -1, 0.25)], True),
    "bound": ([(1, 0, -1.5)], [(0, 1, 1, 1, -2.5)], True),
    "one-variable": ([(1, 1, -0.5)], [(0, 0, 1, 1, -3.0)], True),
}


@pytest.mark.parametrize("case", list(_EMPTINESS))
@pytest.mark.parametrize("kind", KINDS)
def test_is_empty_set_is_exact(kind, case):
    """``is_empty_set`` answers whether the stored rows have a solution, as an
    exact simplex in Fractions decides it: a box; a set of one point (x1 - x2
    <= -2, or x0 + x1 <= -2, on the box: nearly empty); a cycle of two
    difference rows of negative sum that does not touch the anchor; a row that
    contradicts a box row (Hexatope: against the anchor; Octatope: a sum row
    below -2); and a row on one variable (Hexatope: x1 - x1 <= -1/2, which
    reads 0 <= -1/2; Octatope: 2 x0 <= -3). Catches an emptiness test that is
    never run or always says non-empty, one that ignores the sum rows of an
    Octatope or a row on one variable, and one that calls a one-point set
    empty."""
    hex_rows, oct_rows, empty = _EMPTINESS[case]
    s = _hexatope_kernel(hex_rows) if kind == "Hexatope" else _octatope_kernel(oct_rows)
    c, G, A, b = _kernel(s)
    truth = R.feasible((), (), A, b) is None
    if truth != empty:
        pytest.fail(f"test setup: the {case!r} set is {'' if truth else 'not '}empty")
    assert bool(s.is_empty_set()) == empty, (
        f"{kind}.is_empty_set() of the {case!r} set says "
        f"{'empty' if not empty else 'not empty'}")


# --------------------------------------------------------------------------- #
# contains                                                                      #
# --------------------------------------------------------------------------- #
def _probe_points(s, family):
    """``(point, label)`` pairs: the vertices; points 1/8 of the way from each
    vertex to the centre (members, by convexity: the centre is the kernel
    point 0, which keeps every row strictly); points beyond each vertex on
    the ray from the centre (not members); for the raised set, members moved
    off the affine hull."""
    centre = [Fraction(float(v)) for v in s.center]
    points = []
    for v in _vertices(s)[:16]:
        points.append((v, "vertex"))
        points.append((tuple(x + (cx - x) / 8 for x, cx in zip(v, centre)), "inside"))
        points.append((tuple(x + (x - cx) / 4 for x, cx in zip(v, centre)), "beyond"))
        if family == "raised":
            lifted = list(x + (cx - x) / 8 for x, cx in zip(v, centre))
            lifted[-1] += Fraction(1, 4)
            points.append((tuple(lifted), "off the hull"))
    return points


_CONTAINS_LP = known_bug(
    9, "Hexatope.contains and Octatope.contains reject points of the set when "
       "the least-squares preimage breaks a row and the feasibility LP decides: "
       "CVXPY OSQP at 1e-7 returns a preimage whose residual is 1.1e-7, or stops "
       "at its iteration limit, and the recheck at the same 1e-7 rejects it; on "
       "a 5-entry box mapped to 2 entries, vertices and points 1/32 deep inside "
       "are answered False")


@pytest.mark.parametrize("family", ["box", "cut", "rows", "raised",
                                    pytest.param("flat", marks=_CONTAINS_LP)])
@pytest.mark.parametrize("kind", KINDS)
def test_contains_is_exact_membership(kind, family):
    """``contains`` agrees with exact membership on every vertex of the set,
    on points inside near each vertex, on points beyond each vertex and, for
    the set with more entries than kernel variables, on points off its affine
    hull. The families take each path of ``contains``: least squares that
    finds the preimage (box, cut, rows), a residual that rules a point out
    (raised), and the LP for a set with more kernel variables than entries
    (flat, marked). Catches a ``contains`` that always says True or False, one
    that skips the rows after a cut, one that reads the anchor as a free
    variable, and one that ignores the residual."""
    s = _family(kind, family)
    member = R.Membership(R.star(s))
    wrong = []
    for point, label in _probe_points(s, family):
        exact = member(point)
        y = _floats(point)
        if not exact and distance(s, y) < 2.0 ** -6:
            pytest.fail(f"test setup: the {label} point {y} is too near the set")
        if bool(s.contains(y)) != exact:
            wrong.append((label, y.tolist(), exact))
    assert not wrong, (f"{kind}.contains on the {family!r} set: {len(wrong)} "
                       f"wrong answers; first {wrong[0]} (label, point, exact "
                       f"membership)")


@pytest.mark.parametrize("kind", KINDS)
def test_contains_accepts_members_the_feasibility_lp_decides(kind):
    """The set ``{1/2 + 2 a + b : -1 <= a, b <= 1}`` = [-5/2, 7/2] has two
    kernel variables for one entry. For a point further than 5/2 from the
    centre the least-squares preimage has ``|a| > 1`` and breaks a row, so
    the feasibility LP decides. The six points at distance 21/8, 11/4 and
    23/8 from the centre are members at least 1/8 inside the set, which the
    LP answers today (the vertices and the points near them are the marked
    ``flat`` family above). Catches a ``contains`` that says False as soon
    as the least-squares preimage breaks a row, without asking the LP."""
    if kind == "Hexatope":
        s = _hexatope([0.5], [[0.0, 2.0, 1.0]],
                      [r for i in (1, 2) for r in ((i, 0, 1.0), (0, i, 1.0))])
    else:
        s = _octatope([0.5], [[2.0, 1.0]],
                      [r for i in range(2)
                       for r in ((i, i, 1, 0, 1.0), (i, i, -1, 0, 1.0))])
    member = R.Membership(R.star(s))
    for eighths in (21, 22, 23, -21, -22, -23):
        point = (Fraction(1, 2) + Fraction(eighths, 8),)
        if not member(point):
            pytest.fail(f"test setup: {point} is not a member")
        assert s.contains(_floats(point)), (
            f"{kind}.contains rejects the member {float(point[0])}, which "
            f"is 1/8 or more inside [-5/2, 7/2]")


@pytest.mark.parametrize("kind", KINDS)
def test_contains_refuses_a_point_of_the_wrong_length(kind):
    """A point with one entry more than the set cannot be read:
    ``ValueError``. Catches a point cut or padded to the set's length."""
    s = CLASSES[kind].from_bounds(_LO, _HI)
    with pytest.raises(ValueError):
        s.contains(np.zeros(6))


# --------------------------------------------------------------------------- #
# sample                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", ["box", "cut", "rows", "flat"])
@pytest.mark.parametrize("kind", KINDS)
def test_sample_returns_points_of_the_set(kind, family):
    """``sample(64)`` returns 64 points of the set as a (64, dim) array, each
    a member by the set's own rows (``membership.uncovered``, slack
    ``LP_TOL``; the sampler admits 1e-10 in the kernel). The cut is a
    difference row of two kernel variables, where today's sampler is right
    for both types (a cut against the Hexatope anchor is registry HEXOCT-2).
    Catches a rejection test that accepts every draw, skips a row or reads
    the rows reversed, a point without its centre, and a wrong count or
    layout."""
    s = _family(kind, family)
    points = s.sample(64)
    assert points.shape == (64, s.dim), f"{kind}.sample(64): shape {points.shape}"
    missed = uncovered([s], list(points))
    assert not missed, (f"{kind}.sample on the {family!r} set: {len(missed)} of "
                        f"64 points are not in the set (largest distance "
                        f"{max(d for _, d in missed):.3g})")
