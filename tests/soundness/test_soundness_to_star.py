"""
Soundness tests for to_star() conversion methods.

These tests verify the conversion of a Hexatope, an Octatope and a Zono to a
Star, which the verdict route (``verify_specification``) and the Star lane
rely on:
- every point of the set is in the Star (a Star smaller than the set is a
  false UNSAT on the verdict route);
- where the conversion is exact (an Octatope, a Zono, a Hexatope whose rows do
  not cut against its anchor) every point of the Star is in the set;
- ``verify_specification`` on a list of Hexatopes or Octatopes answers UNSAT
  only when no true output lies in the unsafe region.
The oracle is exact (vertices and membership in Fractions, ``tests/_oracles``).
"""

import pytest
import numpy as np
from n2v.sets import Star, Hexatope, Octatope


class TestHexatopeToStarSoundness:
    """``Star.contains`` on the Star of a Hexatope box."""

    def test_soundness_point_containment_corners(self):
        """The four corners of [0, 1]^2 are members of the Hexatope and of its
        Star, by ``Hexatope.contains`` and ``Star.contains`` (expected by
        construction). Three corners lie below the centre in some coordinate,
        so the membership LP of ``Star.contains`` has an equality row with a
        negative right-hand side. Catches the certificate path of
        ``check_feasibility`` writing the negated copy of an equality row with
        the wrong sign of its right-hand side (a member then reads as empty),
        a generator written into the anchor column, and predicate lower bounds
        0 instead of -1. The other old point checks of this class were
        removed: ``test_to_star_is_the_same_set_without_an_anchor_cut``
        replaces them with an exact oracle."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)
        star = hexatope.to_star()

        # Test all corner points
        corners = [
            np.array([[0.0], [0.0]]),
            np.array([[1.0], [0.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0], [1.0]]),
        ]

        for corner in corners:
            assert hexatope.contains(corner), f"Hexatope should contain corner {corner.T}"
            assert star.contains(corner), f"Star should contain corner {corner.T}"


class TestOctatopeToStarSoundness:
    """``Star.contains`` on the Star of an Octatope box."""

    def test_soundness_point_containment_corners(self):
        """The four corners of [0, 1]^2 are members of the Octatope and of its
        Star, by ``Octatope.contains`` and ``Star.contains`` (expected by
        construction). Catches the certificate path of ``check_feasibility``
        writing the negated copy of an equality row with the wrong sign of its
        right-hand side (the corners below the centre then read as empty), and
        the basis stacked as [G, c]. The other old point checks of this class
        were removed: ``test_to_star_is_the_same_set_without_an_anchor_cut``
        and ``test_octatope_to_star_adds_no_point`` replace them with an exact
        oracle."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)
        star = octatope.to_star()

        # Test all corner points
        corners = [
            np.array([[0.0], [0.0]]),
            np.array([[1.0], [0.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0], [1.0]]),
        ]

        for corner in corners:
            assert octatope.contains(corner), f"Octatope should contain corner {corner.T}"
            assert star.contains(corner), f"Star should contain corner {corner.T}"


# =========================================================================== #
# Soundness cells: to_star of Zono, Hexatope and Octatope                      #
# =========================================================================== #
# Oracle: each set read from its stored arrays by ``tests/_oracles`` (a
# Hexatope with its anchor pinned to 0, see ``membership.rows``), its vertices
# and the vertices of the Star computed in rationals, membership decided
# exactly (``rational.Membership``, zero slack). Inputs are dyadic. A Star is
# compared with the set it came from, never with another n2v path. The
# direction "the Star of a Hexatope holds no extra point" is marked in
# ``contracts/test_set_protocol.py`` (registry HEXOCT-1) and is not repeated.
from fractions import Fraction as _Fraction

import torch.nn as _nn

from n2v.sets import Zono
from n2v.sets.hexatope import DifferenceConstraintSystem
from n2v.sets.octatope import UTVPIConstraintSystem
from tests._oracles import rational as _R
from tests.cell_support import reach_one_layer
from tests.known_bugs import known_bug


def _set_polytope(s):
    """``(c, G, A, b)`` of ``{c + G a : A a <= b}`` read by ``rational.star``,
    with a pinned predicate variable (``lb == ub``) replaced by its value and
    every other finite predicate bound written as a row."""
    c, G, C, d, lb, ub = _R.star(s)
    n = len(lb)
    pinned = {j: lb[j] for j in range(n)
              if lb[j] is not None and ub[j] is not None and lb[j] == ub[j]}
    free = [j for j in range(n) if j not in pinned]
    zero = _Fraction(0)
    c = [ci + sum((row[j] * v for j, v in pinned.items()), zero)
         for ci, row in zip(c, G)]
    G = [[row[j] for j in free] for row in G]
    A = [[row[j] for j in free] for row in C]
    b = [di - sum((row[j] * v for j, v in pinned.items()), zero)
         for di, row in zip(d, C)]
    for k, j in enumerate(free):
        unit = [_Fraction(int(m == k)) for m in range(len(free))]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    return c, G, A, b


def _bounded(A):
    """Is ``{a : A a <= b}`` bounded? At once when every variable has a row
    ``a_j <= .`` and a row ``-a_j <= .``; otherwise by ``rational.is_bounded``
    (a vertex enumeration of a cone, slow beyond four variables)."""
    n = len(A[0])
    rows = {tuple(row) for row in A}
    unit = [tuple(_Fraction(int(k == j)) for k in range(n)) for j in range(n)]
    if all(u in rows and tuple(-v for v in u) in rows for u in unit):
        return True
    return _R.is_bounded(A)


def _set_vertices(s):
    """Points whose hull is the set, exact; a setup failure when the set is
    empty or unbounded."""
    c, G, A, b = _set_polytope(s)
    if not A or not _bounded(A):
        pytest.fail("test setup: the set is unbounded")
    found = sorted({tuple(_R.dot(row, a) + ci for row, ci in zip(G, c))
                    for a in _R.vertices(A, b)})
    if not found:
        pytest.fail("test setup: the set is empty")
    return found


def _set_pieces(s):
    c, G, A, b = _set_polytope(s)
    return _R.affine(_R.polytope(A, b), G, c)


def _assert_inside(points, computed, what):
    member = _R.Membership(_R.star(computed))
    missed = [p for p in points if not member(p)]
    assert not missed, (
        f"{what}: {len(missed)} of {len(points)} points are not members; "
        f"first {[float(v) for v in missed[0]]}")


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


_TS_CENTER = [-0.375, 0.5, -1.25]
_TS_HEX_G = [[0.0, 0.5, -0.25, 0.75], [0.0, -1.0, 0.5, 0.25],
             [0.0, 0.25, 0.25, -0.5]]
_TS_OCT_G = [[0.5, -0.25, 0.75], [-1.0, 0.5, 0.25], [0.25, 0.25, -0.5]]


def _to_star_family(kind, name):
    """Sets whose kernel lies in [-1, 1] (``from_bounds`` on a dyadic box,
    cut by a difference row, then mapped; or a kernel with extra rows), or the set
    ``(t, 2 t)`` whose rows give t the interval [-3, 3] ("wide")."""
    lo, hi = np.array([-0.5, -0.75, 0.0]), np.array([1.5, 1.25, 0.5])
    if name == "cut":
        # the cut x0 - x1 <= 1/4 on equal half-widths is one difference (or
        # UTVPI) row in generator space; the map then couples the entries
        s = {"Hexatope": Hexatope, "Octatope": Octatope}[kind].from_bounds(
            lo, hi).intersect_half_space(np.array([[1.0, -1.0, 0.0]]),
                                         np.array([[0.25]]))
        return s.affine_map(np.array([[1.0, -0.5, 0.25], [0.5, 1.0, -1.0],
                                      [0.0, 0.25, 2.0]]),
                            np.array([[0.125], [-0.25], [0.5]]))
    if name == "rows" and kind == "Hexatope":
        box = [r for i in (1, 2, 3) for r in ((i, 0, 1.0), (0, i, 1.0))]
        return _hexatope(_TS_CENTER, _TS_HEX_G,
                         box + [(1, 2, 0.25), (3, 0, 0.5), (0, 1, 0.75),
                                (2, 3, 0.5)])
    if name == "rows":
        box = [r for i in range(3) for r in ((i, i, 1, 0, 1.0),
                                             (i, i, -1, 0, 1.0))]
        return _octatope(_TS_CENTER, _TS_OCT_G,
                         box + [(0, 1, -1, -1, 0.5), (1, 2, 1, 1, 0.75),
                                (0, 2, 1, -1, 0.25), (2, 1, -1, 1, 1.0)])
    if name == "wide" and kind == "Hexatope":
        return _hexatope([0.0, 0.0], [[0.0, 1.0], [0.0, 2.0]],
                         [(1, 0, 3.0), (0, 1, 3.0)])
    if name == "wide":
        return _octatope([0.0, 0.0], [[1.0], [2.0]],
                         [(0, 0, 1, 0, 3.0), (0, 0, -1, 0, 3.0)])
    raise ValueError(name)


_WIDE_KERNEL = known_bug(
    9, "Hexatope.to_star and Octatope.to_star give every predicate variable "
       "the bounds [-1, 1]; for a set whose rows allow a kernel variable "
       "beyond 1 the Star is strictly smaller (t in [-3, 3]: the member "
       "(3, 6) is outside the Star; registry HEXOCT-9)")


@pytest.mark.parametrize("name", [
    "cut", "rows", pytest.param("wide", marks=_WIDE_KERNEL)])
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_to_star_holds_every_point_of_the_set(kind, name):
    """Every vertex of the Hexatope or Octatope (anchor pinned) is a member
    of its ``to_star()``, exactly: the Star is an over-approximation. Sets: a
    box cut by a difference row and then mapped, a kernel with one-sided (or
    two-variable) rows, and a kernel variable in [-3, 3] (marked). Catches
    predicate lower bounds 0 instead of -1, the basis stacked as [G, c], a
    lost or mirrored kernel row, and a two-variable row written into the
    wrong column."""
    s = _to_star_family(kind, name)
    _assert_inside(_set_vertices(s), s.to_star(),
                   f"{kind}.to_star() of the {name!r} set")


@pytest.mark.parametrize("name", ["cut", "rows", "wide"])
def test_octatope_to_star_adds_no_point(name):
    """The other direction for an Octatope, which has no anchor: every vertex
    of ``to_star()`` is a member of the Octatope, exactly (for the "wide"
    set the Star is smaller, so this direction holds there too). Catches a
    Star whose rows are looser than the UTVPI rows (a dropped or mirrored
    two-variable row)."""
    s = _to_star_family("Octatope", name)
    _assert_inside(_set_vertices(s.to_star()), s,
                   f"Octatope.to_star() of the {name!r} set")


def test_zono_to_star_is_the_same_set():
    """``Zono.to_star()`` of a 3-dimensional zonotope with four mixed-sign
    generators (dyadic) is the same set: every vertex of the zonotope is in
    the Star and every vertex of the Star is in the zonotope, exactly.
    Catches predicate bounds [0, 1] instead of [-1, 1], the basis stacked as
    [V, c], and a row of the box dropped."""
    z = Zono(np.array([[0.25], [-0.5], [1.0]]),
             np.array([[0.5, -1.0, 0.25, 0.0], [0.0, 0.75, -0.5, 1.0],
                       [-0.25, 0.0, 0.5, 0.5]]))
    star = z.to_star()
    _assert_inside(_set_vertices(z), star, "Zono.to_star()")
    _assert_inside(_set_vertices(star), z, "the vertices of Zono.to_star()")


@pytest.mark.parametrize("kind", ["Zono", "Octatope", "Hexatope"])
def test_to_star_then_exact_relu_holds_the_image(kind):
    """``to_star()`` followed by an exact ReLU on the Star (``reach_layer``):
    every piece of the exact image of the original set is held by one
    returned Star, exactly; for a Zono and an Octatope, whose Star is the
    same set, every vertex of every returned Star has a preimage too (the
    exact method's other direction). Catches a conversion whose basis, rows
    or predicate bounds the ReLU then reads wrongly."""
    if kind == "Zono":
        s = Zono(np.array([[0.25], [-0.5], [0.125]]),
                 np.array([[0.5, -1.0, 0.25], [0.0, 0.75, -0.5],
                           [-0.25, 0.0, 0.5]]))
    else:
        s = _to_star_family(kind, "cut")
    out = reach_one_layer(_nn.ReLU(), [s.to_star()], "exact",
                          f"exact ReLU on {kind}.to_star()")
    pieces = _R.relu(_set_pieces(s))
    readers = [_R.Membership(_R.star(o)) for o in out]
    images = [sorted(set(p.image())) for p in pieces if p.is_full_dimensional()]
    spread = [k for k, image in enumerate(images)
              if not any(all(m(y) for y in image) for m in readers)]
    assert not spread, (f"exact ReLU on {kind}.to_star(): pieces {spread} of "
                        f"the exact image are held by no single returned set")
    if kind != "Hexatope":
        extra = [(k, y) for k, o in enumerate(out)
                 for y in _set_vertices(o) if not _R.image_contains(pieces, y)]
        assert not extra, (f"exact ReLU on {kind}.to_star(): returned set "
                           f"{extra[0][0]} has the vertex "
                           f"{[float(v) for v in extra[0][1]]} outside the image")


# =========================================================================== #
# Soundness cells: to_star is the same set where no cut touches the            #
# Hexatope anchor, and the verdict route through it                            #
# =========================================================================== #
# Same oracle as the cells above. ``verify_specification`` reaches
# ``to_star`` for a list of Hexatopes or Octatopes (``_is_disjoint``) and
# decides UNSAT on the Star, so a Star smaller than the set is a false UNSAT.
import torch

from n2v.sets import HalfSpace
from n2v.utils.verify_specification import verify_specification
from tests.cell_support import reach_model

# Straddling, zero width, and a negative coordinate with a long mantissa (not
# a float32 number); every end beyond plus and minus 1.
_R6_LO = np.array([-3.0, 1.5, -(3.0 + 2.0 ** -29)])
_R6_HI = np.array([5.0, 1.5, -(1.0 + 2.0 ** -30)])
_R6_FLAT_W = np.array([[1.0, 3.0, -0.5], [0.25, -1.0, 2.0]])


def _r6_family(kind, name):
    """"box" (``from_bounds`` on the bounds above), "flat" (that box mapped
    to two entries: more kernel variables than entries) or the "cut"
    set (a difference row of two kernel variables, then a map)."""
    if name == "cut":
        return _to_star_family(kind, "cut")
    s = {"Hexatope": Hexatope, "Octatope": Octatope}[kind].from_bounds(_R6_LO, _R6_HI)
    if name == "box":
        return s
    if name == "flat":
        return s.affine_map(_R6_FLAT_W, np.array([0.5, -1.0]))
    raise ValueError(name)


@pytest.mark.parametrize("kind,name", [("Hexatope", "box"), ("Hexatope", "cut"),
                                       ("Hexatope", "flat"), ("Octatope", "box"),
                                       ("Octatope", "flat")])
def test_to_star_is_the_same_set_without_an_anchor_cut(kind, name):
    """On a set whose rows do not cut against the Hexatope anchor, ``to_star()``
    is the same set: every vertex of the set is in the Star and every vertex
    of the Star is in the set, exactly. Sets: a box with a zero-width and a
    long-mantissa coordinate, a difference cut of two kernel variables, and a
    box mapped to fewer entries than kernel variables. Catches the basis
    stacked as [G, c], predicate bounds other than [-1, 1] in either
    direction, a dropped or mirrored row, and the centre or generators
    rounded to float32 (the long-mantissa coordinate moves). An anchor cut is
    registry HEXOCT-1 (``contracts/test_set_protocol.py``)."""
    s = _r6_family(kind, name)
    star = s.to_star()
    _assert_inside(_set_vertices(s), star, f"{kind}.to_star() of the {name!r} set")
    _assert_inside(_set_vertices(star), s,
                   f"the vertices of {kind}.to_star() of the {name!r} set")


_WIDE_KERNEL_VERDICT = known_bug(
    9, "Hexatope.to_star and Octatope.to_star give every predicate variable "
       "the bounds [-1, 1], so verify_specification on the reach of "
       "{(t, 2 t) : t in [-3, 3]} answers UNSAT for the unsafe region y0 >= 3, "
       "which holds the true output (3, 6) (registry HEXOCT-9)")

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -2)]


@pytest.mark.parametrize("name", ["cut", pytest.param("wide", marks=_WIDE_KERNEL_VERDICT)])
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_verify_specification_through_to_star_is_unsat_only_when_safe(kind, name):
    """The verdict route of ``to_star``: the set (the "cut" set, or
    the "wide" set whose kernel variable ranges over [-3, 3]) is reached
    through a float64 Linear layer (to two outputs) by
    ``reach_pytorch_model``; for each of eight directions d the unsafe region
    d . y >= m, with m the exact maximum of d . y over the image (the region
    touches the image), holds a true output, so ``verify_specification`` must
    not answer UNSAT. Catches every to_star fault that makes the Star smaller
    than the set (predicate bounds [0, 1], the basis stacked as [G, c], a
    row tightened), which here is a false UNSAT."""
    s = _to_star_family(kind, name)
    if name == "cut":
        W, b = np.array([[1.0, -0.5, 0.25], [0.5, 1.0, -1.0]]), np.array([0.25, -0.5])
    else:
        W, b = np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 0.0])
    layer = _nn.Linear(W.shape[1], 2, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(W))
        layer.bias.copy_(torch.from_numpy(b))
    what = f"reach of the {name!r} {kind} through Linear"
    out = reach_model(_nn.Sequential(layer), s, "approx", what)
    image = sorted({y for piece in _R.affine(_set_pieces(s), W, b) for y in piece.image()})
    unsat = []
    for d in _DIRECTIONS:
        top = max(_R.dot([_Fraction(v) for v in d], y) for y in image)
        spec = HalfSpace(-np.array([d], dtype=np.float64), np.array([-float(top)]))
        if verify_specification(out, spec).verdict == "UNSAT":
            unsat.append((d, float(top)))
    assert not unsat, (f"{what}: UNSAT for the unsafe regions d . y >= m that "
                       f"touch the image, (d, m) = {unsat}")
