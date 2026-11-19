"""Soundness cells for the methods of ``Box`` that reach and verification
rely on.

Covered: the constructor through its zonotope form (centre and generators,
with a zero-width coordinate in front of a perturbed one, with infinite
bounds, and the column-by-column fallback after a ``MemoryError``), its
refusals of bounds that are no box, and its independence from the caller's
arrays; ``to_zono`` and ``to_star`` (``Star.from_bounds`` is
``Box.to_star``); the range queries ``get_range``, ``get_ranges`` and
``estimate_ranges``; ``affine_map`` with a non-square matrix and with a 1-D
weight row, with and without a bias, on a box with infinite ends (directly
and through a Linear layer), and its refusal of a weight or bias of another
size; ``minkowski_sum``, ``convex_hull``, ``union`` and ``box_hull``, and
their refusal of an operand of another dimension or type; ``intersect`` of
overlapping boxes, of boxes that share a fixed coordinate, of boxes that
touch, and of disjoint boxes; ``single_partition`` and ``partition`` over
one, two and three dimensions, of a width that overflows and of an
unbounded coordinate; ``contains``, ``sample``, ``get_vertices`` and
``__repr__``; and one composition, Linear then ReLU then Linear on several
input boxes at once, through ``reach_layer`` and through
``reach_pytorch_model``.

Oracle. Every reference is computed here in ``fractions.Fraction`` from the
float bounds the test passes (a float is the rational it is): the hull, the
intersection, the sum, the interval image of an affine map, the corners, and
the grid that decides whether partition pieces tile the box. A computed Box
is read from its stored ``lb`` and ``ub`` (for a box these are its rows); a
computed Zono or Star from its stored arrays (``tests/_oracles/rational.py``:
``star``, ``Membership``, ``set_vertices``). The composition replays the
torch model in float64 and checks membership with
``cell_support.assert_covered``.

Slack. None. Soundness is outward containment of the exact result. Where a
method claims the exact result (hull, union, box hull, intersection, sum,
partition, the range queries, the conversions) and the data are dyadic, the
result must also equal it. Non-dyadic data check the outward rounding: a
bound rounded to nearest can land inside the exact value by an ulp, and the
exact comparison sees that. An infinite bound stands for every real number
beyond the finite end: a zero weight times it is 0, a nonzero weight makes
that side infinite, and a NaN bound holds no point.

Layout. A Box is flat; its entries are in the order of its bounds.
"""

import itertools
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from tests._oracles import rational as Q
from tests.cell_support import (assert_covered, flat_box, reach_model,
                                reach_one_layer, replay_torch, samples,
                                split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug


# --------------------------------------------------------------------------- #
# Exact helpers                                                                 #
# --------------------------------------------------------------------------- #
def fr(values):
    """The rationals that the floats ``values`` are, as a flat list."""
    return [Fraction(float(v)) for v in np.asarray(values,
                                                  dtype=np.float64).ravel()]


def stored(box):
    """``(lb, ub)`` of a computed Box as Fractions, read from its arrays."""
    if not isinstance(box, Box):
        pytest.fail(f"expected a Box, got {type(box).__name__}")
    return fr(box.lb), fr(box.ub)


def assert_outward(box, lo, hi, what):
    """The computed box contains the exact box ``[lo, hi]`` (Fractions):
    every lower bound at or below, every upper bound at or above, checked
    one side at a time with no slack."""
    blo, bhi = stored(box)
    if len(blo) != len(lo):
        pytest.fail(f"{what}: {len(blo)} entries, expected {len(lo)}")
    low = [i for i in range(len(lo)) if blo[i] > lo[i]]
    high = [i for i in range(len(hi)) if bhi[i] < hi[i]]
    assert not low, (f"{what}: lower bound above the exact one at entries "
                     f"{low}, e.g. {float(blo[low[0]])!r} > "
                     f"{float(lo[low[0]])!r}")
    assert not high, (f"{what}: upper bound below the exact one at entries "
                      f"{high}, e.g. {float(bhi[high[0]])!r} < "
                      f"{float(hi[high[0]])!r}")


def assert_exact(box, lo, hi, what):
    """The computed box is exactly ``[lo, hi]``: outward (soundness) and not
    larger (the method claims the exact result)."""
    assert_outward(box, lo, hi, what)
    blo, bhi = stored(box)
    assert blo == lo and bhi == hi, (
        f"{what}: the result is larger than the exact box: lb "
        f"{[float(v) for v in blo]} vs {[float(v) for v in lo]}, ub "
        f"{[float(v) for v in bhi]} vs {[float(v) for v in hi]}")


def _as_box(lo, hi):
    """Returned bound arrays held as a Box without the constructor (which
    would refuse crossed bounds), so the one-sided checks report them."""
    out = Box.__new__(Box)
    out.lb = np.asarray(lo, dtype=np.float64).reshape(-1, 1)
    out.ub = np.asarray(hi, dtype=np.float64).reshape(-1, 1)
    return out


def corners(lo, hi):
    """Every corner of the box ``[lo, hi]`` (Fractions), without repeats."""
    return sorted(set(itertools.product(*[(a, b) for a, b in zip(lo, hi)])))


# --------------------------------------------------------------------------- #
# The input boxes                                                               #
# --------------------------------------------------------------------------- #
# Dyadic boxes, so every exact result is a float; coordinate 2 of A has zero
# width. A's upper bounds are below B's on coordinates 0 and 2 and above on 1
# and 3, its lower bounds below on 0 and above on 1 to 3, so a result that
# takes a bound from the wrong operand is seen.
_A = ([-1.0, 0.5, -0.25, 2.0], [0.5, 1.5, -0.25, 3.0])
_B = ([-0.5, -1.0, -2.0, 1.5], [1.25, 0.75, 0.5, 2.5])
# Non-dyadic boxes: sums, midpoints and products of their bounds round.
_C = ([0.1, -0.3, 1 / 3, 7.0], [0.7, 0.2, 2 / 3, 7.0])
_D = ([0.2, -1 / 3, 0.1, -2.2], [0.3, 0.1, 0.4, 1e-3])

PAIRS = {"dyadic": (_A, _B), "inexact": (_C, _D)}


def box_of(bounds):
    lo, hi = bounds
    return Box(np.array(lo), np.array(hi))


# --------------------------------------------------------------------------- #
# Constructor, to_zono, to_star                                                 #
# --------------------------------------------------------------------------- #
_ZONO_BOXES = {
    "dyadic": _A,
    "zero-width-first": ([0.5, -1.0, 0.25, -0.75], [0.5, 1.0, 0.75, -0.75]),
    "inexact": _C,
}


def _assert_zonotope_form(box, lo, hi, dyadic, what):
    """``to_zono``, ``to_star`` and ``Star.from_bounds`` hold every corner of
    the box in exact arithmetic; with dyadic bounds their vertices are
    exactly the corners (nothing beyond the box)."""
    lo_q, hi_q = fr(lo), fr(hi)
    box_corners = corners(lo_q, hi_q)
    built = {"to_zono": box.to_zono(), "to_star": box.to_star(),
             "Star.from_bounds": Star.from_bounds(
                 np.asarray(lo).reshape(-1, 1), np.asarray(hi).reshape(-1, 1))}
    for name, s in built.items():
        q = Q.star(s)
        member = Q.Membership(q)
        missed = [c for c in box_corners if not member(c)]
        assert not missed, (f"{what}: {name} leaves out {len(missed)} of "
                            f"{len(box_corners)} corners of the box, e.g. "
                            f"{[float(v) for v in missed[0]]}")
        if dyadic:
            verts = Q.set_vertices(q)
            assert sorted(verts) == box_corners, (
                f"{what}: {name} is not exactly the box: vertices "
                f"{[[float(v) for v in p] for p in verts[:4]]}...")
        if name != "to_zono":
            # The user's bounds ride along for witness code, which clips a
            # reconstructed corner into them (refine/witness.py).
            raw_lo, raw_hi = fr(s._raw_lb), fr(s._raw_ub)
            assert raw_lo == lo_q and raw_hi == hi_q, (
                f"{what}: {name} carries raw bounds {raw_lo}, {raw_hi} for "
                f"the box {lo_q}, {hi_q}")


@pytest.mark.parametrize("name", list(_ZONO_BOXES))
def test_box_zonotope_form_is_the_box(name):
    """The centre and generators a Box stores (its zonotope form, read by
    ``to_zono``, ``to_star`` and ``Star.from_bounds``) give back the box: every
    corner is a member, exactly; with dyadic bounds the converted set is the
    box and nothing more; the star carries the user's own bounds for witness
    code. Catches half-widths halved, a zonotope centred on the lower bound,
    the generator of a perturbed coordinate lost when a zero-width coordinate
    comes before it, a centre or half-width rounded inward on non-dyadic
    bounds, and the raw bounds attached swapped or rounded."""
    lo, hi = _ZONO_BOXES[name]
    box = Box(np.array(lo), np.array(hi))
    _assert_zonotope_form(box, lo, hi, name != "inexact", name)


def _box_without_diag(monkeypatch, lo, hi):
    """``Box(lo, hi)`` built while ``np.diag`` raises ``MemoryError``, so the
    constructor takes its column-by-column fallback (box.py, ``__init__``)."""
    def no_memory(*args, **kwargs):
        raise MemoryError("injected")
    with monkeypatch.context() as patch:
        patch.setattr(np, "diag", no_memory)
        return Box(np.array(lo), np.array(hi))


# The fallback assigns a one-element array to an array element, which numpy
# 2.3 accepts with a DeprecationWarning (an error in this suite and in a
# future numpy); the value cell lets that warning through so that it checks
# the set, and the next cell marks the warning itself.
@pytest.mark.filterwarnings(
    "ignore:Conversion of an array with ndim > 0 to a scalar:"
    "DeprecationWarning")
def test_box_zonotope_form_after_a_memory_error_is_the_box(monkeypatch):
    """When building the diagonal generator matrix raises ``MemoryError``,
    the constructor builds the generators column by column; the result must
    be the same box: every corner a member, nothing beyond it, one generator
    per coordinate of positive width. The bounds have zero-width coordinates
    in front of and behind perturbed ones. Catches a fallback that puts a
    half-width in the wrong row or column, or keeps a zero column."""
    lo, hi = _ZONO_BOXES["zero-width-first"]
    box = _box_without_diag(monkeypatch, lo, hi)
    assert box.generators.shape == (4, 2)
    _assert_zonotope_form(box, lo, hi, True, "MemoryError fallback")


@known_bug(8, "Box.__init__, on its fallback after a MemoryError, assigns "
              "vec[idx] (a one-element array) to one array element, which "
              "numpy deprecates; under this suite's warnings-as-errors policy "
              "(and in a future numpy) the constructor raises ValueError "
              "instead of building the box", raises=ValueError)
@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_box_constructor_fallback_raises_no_deprecation(monkeypatch):
    """The column-by-column fallback of the constructor builds the box without
    relying on behaviour numpy deprecates: with the suite's warnings as
    errors it must not raise. The passing neighbour is
    :func:`test_box_zonotope_form_after_a_memory_error_is_the_box`, the same
    construction with that one warning let through."""
    lo, hi = _ZONO_BOXES["zero-width-first"]
    box = _box_without_diag(monkeypatch, lo, hi)
    if box.generators.shape != (4, 2):
        pytest.fail(f"generators of shape {box.generators.shape}")


# Bounds that are no interval of the documented shape, (n,) or (n, 1). The
# first three are refused today; the marked two are built as a Box.
_UNREADABLE_BOUNDS = {
    # crossed in one coordinate only, so a check over all coordinates fails
    "lower above upper": ([0.5, 1.0, -2.0], [0.25, 2.0, 3.0]),
    "different lengths": (np.zeros(3), np.ones(4)),
    "two columns": (np.zeros((3, 2)), np.ones((3, 2))),
    "NaN bound": ([-1.0, np.nan, 0.25], [1.0, 2.0, 0.25]),
    "three axes": (np.zeros((2, 1, 3)), np.ones((2, 1, 3))),
}


@pytest.mark.parametrize("name", [
    "lower above upper", "different lengths", "two columns",
    pytest.param("NaN bound", marks=known_bug(
        8, "Box.__init__ refuses lb > ub only, which is false for a NaN "
           "bound, so a box with a NaN bound is built; it holds no point",
        raises=DID_NOT_RAISE)),
    pytest.param("three axes", marks=known_bug(
        8, "Box.__init__ checks that axis 1 of its bounds has length 1 and "
           "not that they have at most two axes, so bounds of shape "
           "(2, 1, 3) are built into a Box of dimension 2 holding six "
           "numbers per bound", raises=DID_NOT_RAISE)),
])
def test_box_refuses_bounds_it_cannot_read(name):
    """``Box(lb, ub)`` refuses with ``ValueError`` bounds that are no box:
    a lower bound above its upper bound in one coordinate, bounds of
    different lengths, two-column arrays, a NaN bound (no number, so no
    interval) and an array of three axes. Catches a crossed coordinate let
    through when the check needs every coordinate crossed, and the two
    marked inputs answered."""
    lo, hi = _UNREADABLE_BOUNDS[name]
    with pytest.raises(ValueError):
        Box(np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64))


@known_bug(23, "Box.__init__ reads axis 1 of its bounds without checking "
               "their rank, so scalar bounds (Box(-0.75, 2.5)) raise "
               "IndexError instead of a ValueError or the box [-0.75, 2.5]",
           raises=IndexError)
def test_box_of_scalar_bounds_is_the_interval_or_refused():
    """``Box(-0.75, 2.5)`` with two floats either refuses with ``ValueError``
    or is the one-dimensional box ``[-0.75, 2.5]``: its range queries give
    exactly those bounds and its zonotope form is exactly that interval. An
    accidental exception is neither. The passing neighbour is
    :func:`test_box_zonotope_form_is_the_box`."""
    try:
        box = Box(-0.75, 2.5)
    except ValueError:
        return
    got_lo, got_hi = box.get_ranges()
    assert_exact(_as_box(got_lo, got_hi), fr([-0.75]), fr([2.5]),
                 "scalar bounds")
    _assert_zonotope_form(box, [-0.75], [2.5], True, "scalar bounds")


@pytest.mark.parametrize("dtype", [
    pytest.param("float64", marks=known_bug(
        8, "Box.__init__ keeps float64 bound arrays without a copy "
           "(np.asarray), so a later change of the caller's arrays changes "
           "the box's lb and ub but not its centre and generators: the range "
           "queries and the zonotope form then describe different sets")),
    "float32",
])
def test_box_does_not_change_when_the_callers_arrays_change(dtype):
    """A Box keeps the set it was built from when the caller later writes
    into the arrays it passed (a loop that reuses one pair of bound arrays
    for several input regions). After the caller's column arrays are
    overwritten, ``get_range``, ``get_ranges`` and ``estimate_ranges`` still
    give the original bounds exactly, and the zonotope form still holds every
    corner of the original box and nothing beyond. float32 arrays are
    converted, and so copied, today: the passing neighbour. Catches bounds
    stored as views of the caller's arrays."""
    lo, hi = _ZONO_BOXES["zero-width-first"]
    lo_arr = np.array(lo, dtype=dtype).reshape(-1, 1)
    hi_arr = np.array(hi, dtype=dtype).reshape(-1, 1)
    box = Box(lo_arr, hi_arr)
    lo_arr[:] = 7.0
    hi_arr[:] = -7.0
    for query in ("get_range", "get_ranges", "estimate_ranges"):
        got_lo, got_hi = getattr(box, query)()
        assert_exact(_as_box(got_lo, got_hi), fr(lo), fr(hi),
                     f"{dtype}: {query} after the caller changed its arrays")
    _assert_zonotope_form(box, lo, hi, True,
                          f"{dtype}: after the caller changed its arrays")


def _row_interval(c, G):
    """Per row of ``c + G alpha`` with ``alpha`` in ``[-1, 1]^m``, the
    interval ``(c - sum_j |G_ij|, c + sum_j |G_ij|)``: Fractions where every
    number is finite, otherwise float arithmetic on the infinite values (an
    infinite centre against an infinite half-width gives NaN, which holds
    nothing)."""
    rows = np.atleast_2d(np.asarray(G, dtype=np.float64))
    out = []
    for ci, row in zip(np.asarray(c, dtype=np.float64).ravel(), rows):
        r = (np.inf if not np.isfinite(row).all()
             else sum(abs(Fraction(float(g))) for g in row))
        cq = Fraction(float(ci)) if np.isfinite(ci) else float(ci)
        out.append((cq - r, cq + r))
    return out


def test_box_zonotope_form_of_infinite_bounds_holds_the_box():
    """A Box may have infinite bounds (the constructor gives such a coordinate
    centre 0 and an infinite half-width). ``to_zono``, ``to_star`` and
    ``Star.from_bounds`` must still hold every point of the box: per
    coordinate, the interval of ``c + G alpha`` over the predicate box holds
    the box's interval (an infinite end needs an infinite half-width), and on
    the finite dyadic coordinate it is exactly that interval. The coordinates
    have an infinite lower end, an infinite upper end, both, and none; the
    star's predicate polytope must hold the whole cube ``[-1, 1]^m`` and the
    star carries the user's infinite bounds for witness code. Catches
    infinite bounds replaced by a finite number before the centre and
    half-width are computed, and an infinite half-width that leaks into the
    row of a finite coordinate."""
    lo = [-np.inf, -2.5, -np.inf, 0.25]
    hi = [1.5, np.inf, np.inf, 3.0]
    box = Box(np.array(lo), np.array(hi))
    built = {"to_zono": box.to_zono(), "to_star": box.to_star(),
             "Star.from_bounds": Star.from_bounds(
                 np.array(lo).reshape(-1, 1), np.array(hi).reshape(-1, 1))}
    for name, s in built.items():
        if name == "to_zono":
            c, G = s.c, s.V
        else:
            c, G = s.V[:, :1], s.V[:, 1:]
            m = G.shape[1]
            p_lo = (np.full(m, -1.0) if s.predicate_lb is None
                    else np.asarray(s.predicate_lb, dtype=np.float64).ravel())
            p_hi = (np.full(m, 1.0) if s.predicate_ub is None
                    else np.asarray(s.predicate_ub, dtype=np.float64).ravel())
            assert (p_lo <= -1.0).all() and (p_hi >= 1.0).all(), (
                f"{name}: predicate bounds {p_lo}, {p_hi} cut the cube")
            C = [fr(row) for row in np.atleast_2d(s.C)]
            d = fr(s.d)
            for v in itertools.product((-1, 1), repeat=m):
                assert all(sum(a * x for a, x in zip(row, v)) <= di
                           for row, di in zip(C, d)), (
                    f"{name}: the predicate rows cut the cube at {v}")
            assert np.array_equal(s._raw_lb, lo) \
                and np.array_equal(s._raw_ub, hi), (
                f"{name}: raw bounds {s._raw_lb}, {s._raw_ub}")
        rows = _row_interval(c, G)
        assert len(rows) == len(lo), f"{name}: {len(rows)} rows"
        for k, ((r_lo, r_hi), l, h) in enumerate(zip(rows, lo, hi)):
            lq = Fraction(l) if np.isfinite(l) else l
            hq = Fraction(h) if np.isfinite(h) else h
            assert r_lo <= lq and r_hi >= hq, (
                f"{name}: coordinate {k} is [{float(r_lo)}, {float(r_hi)}], "
                f"which does not hold the box's [{l}, {h}]")
        assert rows[3] == (Fraction(0.25), Fraction(3.0)), (
            f"{name}: the finite coordinate is {rows[3]}, not [0.25, 3]")


# --------------------------------------------------------------------------- #
# Range queries                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query", ["get_range", "get_ranges",
                                   "estimate_ranges"])
@pytest.mark.parametrize("data", ["inexact"])
def test_range_query_is_the_box(data, query):
    """``get_range``, ``get_ranges`` and ``estimate_ranges`` of a Box return
    ``(lower, upper)`` equal to its bounds, checked one side at a time (a box
    is its own exact range). The bounds are not dyadic, so a range computed
    from the centre and half-width shows; a centre different from either
    bound on three coordinates and one zero-width coordinate. Catches the
    bounds returned in swapped order, a lower bound replaced by the centre,
    and an upper bound replaced by the lower one."""
    lo, hi = PAIRS[data][0]
    box = Box(np.array(lo), np.array(hi))
    got_lo, got_hi = getattr(box, query)()
    assert_exact(_as_box(got_lo, got_hi), fr(lo), fr(hi), f"{data}: {query}")


# --------------------------------------------------------------------------- #
# affine_map                                                                    #
# --------------------------------------------------------------------------- #
_W = np.array([[0.5, -1.0, 0.25, 0.0],
               [-0.75, 0.0, 1.0, 0.5],
               [1.0, 0.5, -0.5, -0.25]])
_b = np.array([0.125, -1.5, 2.0])
_W_INEXACT = np.array([[0.1, -0.7, 1 / 3, 0.0],
                       [-0.3, 0.0, 1.1, 0.2],
                       [1e-3, 0.9, -0.6, -0.4]])
_b_INEXACT = np.array([0.1, -1 / 3, 2.2])


def exact_image(W, b, lo, hi):
    """Bounds of ``{W x + b : lo <= x <= hi}`` in Fractions."""
    W = [fr(row) for row in np.atleast_2d(W)]
    b = [Fraction(0)] * len(W) if b is None else fr(b)
    lo, hi = fr(lo), fr(hi)
    out_lo = [bi + sum(min(w * l, w * h) for w, l, h in zip(row, lo, hi))
              for row, bi in zip(W, b)]
    out_hi = [bi + sum(max(w * l, w * h) for w, l, h in zip(row, lo, hi))
              for row, bi in zip(W, b)]
    return out_lo, out_hi


@pytest.mark.parametrize("weight", ["matrix", "1-D row"])
@pytest.mark.parametrize("bias", ["bias", "column bias", "no bias"])
@pytest.mark.parametrize("data", list(PAIRS))
def test_affine_map_contains_the_exact_image(data, bias, weight):
    """``Box.affine_map(W, b)`` contains the exact interval image of the box
    (each output bound is ``b_i + sum_j min/max(W_ij lb_j, W_ij ub_j)``), in
    exact arithmetic with no slack, for a 3 x 4 weight and for a 1-D weight
    (read as one row), a bias given flat, as a column, or absent. Catches a
    bias dropped or subtracted, endpoints paired with the wrong sign of a
    weight, and a bound rounded to nearest where the exact value is not a
    float (non-dyadic data)."""
    lo, hi = PAIRS[data][0]
    W, b = (_W, _b) if data == "dyadic" else (_W_INEXACT, _b_INEXACT)
    if weight == "1-D row":
        W, b = W[1], b[1:2]
    if bias == "no bias":
        b = None
    elif bias == "column bias":
        b = np.asarray(b).reshape(-1, 1)
    box = Box(np.array(lo), np.array(hi))
    result = box.affine_map(W, b)
    e_lo, e_hi = exact_image(W, b, lo, hi)
    assert_outward(result, e_lo, e_hi, f"{data}: affine_map ({weight}, {bias})")


# A box with an infinite lower end (coordinate 0), an infinite upper end
# (coordinate 1) and two finite coordinates, one of them non-dyadic. Row 0
# meets the infinite lower end with a positive weight and the infinite upper
# end with a zero weight; row 1 the reverse with a negative weight; row 2
# has zero weights on both infinite coordinates (a finite image, rounded
# outward); row 3 is unbounded above only; row 4 is the bias alone.
_INF_LO = [-np.inf, -2.5, 0.1, -3.0]
_INF_HI = [1.5, np.inf, 0.7, 4.0]
_INF_W = np.array([[0.5, 0.0, 0.1, -1.0],
                   [0.0, -2.0, 1.0, 0.25],
                   [0.0, 0.0, 0.3, 2.0],
                   [-1.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
_INF_b = np.array([0.1, -1 / 3, 2.0, 0.0, -0.5])


def exact_image_extended(W, b, lo, hi):
    """Bounds of ``{W x + b : lo <= x <= hi}`` for a box with infinite ends:
    a zero weight contributes 0 whatever the end (0 times any real is 0),
    a nonzero weight on an infinite end makes that side infinite. Fractions,
    or the floats ``-inf`` and ``inf``."""
    out_lo, out_hi = [], []
    for row, bi in zip(W, b):
        low, up = [Fraction(float(bi))], [Fraction(float(bi))]
        for w, l, h in zip(row, lo, hi):
            if w == 0.0:
                continue
            a, c = (l, h) if w > 0 else (h, l)
            low.append(-np.inf if np.isinf(a)
                       else Fraction(float(w)) * Fraction(float(a)))
            up.append(np.inf if np.isinf(c)
                      else Fraction(float(w)) * Fraction(float(c)))
        out_lo.append(-np.inf if -np.inf in low else sum(low))
        out_hi.append(np.inf if np.inf in up else sum(up))
    return out_lo, out_hi


@pytest.mark.parametrize("entry", ["affine_map", "Linear via reach_layer"])
def test_affine_map_of_a_box_with_infinite_ends_contains_the_exact_image(
        entry):
    """``Box.affine_map`` of a box with infinite ends contains the exact
    image: every bound is a number (no NaN from zero times infinity), a side
    that a nonzero weight on an infinite end makes unbounded is infinite,
    and the finite sides hold the exact value with no slack (non-dyadic
    weights and bounds, so a bound rounded inward is seen). Called directly
    and through ``reach_layer`` of a float64 ``nn.Linear`` (``linear_box``).
    Catches infinite ends replaced by a finite number before the interval
    product, and a zero weight times an infinite end read as NaN."""
    box = Box(np.array(_INF_LO), np.array(_INF_HI))
    if entry == "affine_map":
        result = box.affine_map(_INF_W, _INF_b)
    else:
        layer = nn.Linear(4, 5).double()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(_INF_W, dtype=torch.float64))
            layer.bias.copy_(torch.tensor(_INF_b, dtype=torch.float64))
        out = reach_one_layer(layer, [box], "approx", "Linear on a box with "
                              "infinite ends")
        assert len(out) == 1, f"{len(out)} sets for one input box"
        result = out[0]
    e_lo, e_hi = exact_image_extended(_INF_W, _INF_b, _INF_LO, _INF_HI)
    if not isinstance(result, Box):
        pytest.fail(f"expected a Box, got {type(result).__name__}")
    got_lo, got_hi = result.lb.ravel(), result.ub.ravel()
    assert got_lo.shape == (5,) and got_hi.shape == (5,), (
        f"{entry}: bounds of shapes {got_lo.shape}, {got_hi.shape}")
    for i in range(5):
        lo_i, hi_i = float(got_lo[i]), float(got_hi[i])
        assert not (np.isnan(lo_i) or np.isnan(hi_i)), (
            f"{entry}: row {i} has the bounds [{lo_i}, {hi_i}]")
        low_ok = lo_i == -np.inf or (
            e_lo[i] != -np.inf and lo_i != np.inf and Fraction(lo_i) <= e_lo[i])
        up_ok = hi_i == np.inf or (
            e_hi[i] != np.inf and hi_i != -np.inf and Fraction(hi_i) >= e_hi[i])
        assert low_ok and up_ok, (
            f"{entry}: row {i} is [{lo_i}, {hi_i}], which does not hold the "
            f"exact image [{float(e_lo[i])}, {float(e_hi[i])}]")


def test_affine_map_refuses_a_weight_or_bias_of_another_size():
    """``Box.affine_map`` refuses with ``ValueError`` a weight whose column
    count is not the box's dimension (6 columns, and a 1-D weight of 8
    entries, on a box of dimension 4: both hold a multiple of 4 numbers) and
    a bias whose length is neither the weight's row count nor 1. Catches a
    weight reshaped to fit the box instead of refused."""
    box = box_of(_A)
    with pytest.raises(ValueError):
        box.affine_map(np.ones((2, 6)))
    with pytest.raises(ValueError):
        box.affine_map(np.ones(8))
    with pytest.raises(ValueError):
        box.affine_map(np.ones((3, 4)), np.ones(2))


# --------------------------------------------------------------------------- #
# minkowski_sum, convex_hull, union, box_hull                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", ["A+B", "B+A"])
@pytest.mark.parametrize("data", list(PAIRS))
def test_minkowski_sum_is_the_exact_sum(data, order):
    """``Box.minkowski_sum`` is ``[lb1 + lb2, ub1 + ub2]``: outward in exact
    arithmetic (the non-dyadic sums, such as 0.1 + 0.2, are not floats and
    must round down below and up above), and exactly that box for dyadic
    bounds. Catches an upper bound that adds the other box's lower bound and
    a lower bound rounded upward."""
    first, second = PAIRS[data]
    if order == "B+A":
        first, second = second, first
    result = box_of(first).minkowski_sum(box_of(second))
    lo = [a + b for a, b in zip(fr(first[0]), fr(second[0]))]
    hi = [a + b for a, b in zip(fr(first[1]), fr(second[1]))]
    if data == "dyadic":
        assert_exact(result, lo, hi, f"{data}: minkowski_sum {order}")
    else:
        assert_outward(result, lo, hi, f"{data}: minkowski_sum {order}")


@pytest.mark.parametrize("method", ["convex_hull", "union"])
@pytest.mark.parametrize("order", ["A,B", "B,A"])
@pytest.mark.parametrize("data", list(PAIRS))
def test_hull_and_union_are_the_bounding_box_of_both(data, order, method):
    """``convex_hull`` and ``union`` of two boxes are the smallest box that
    holds both: the elementwise least lower and greatest upper bound,
    exactly, in both argument orders. Each operand's bound is the extreme
    one on some coordinate, so a result that ignores the argument, or takes
    the larger lower bound, is seen."""
    first, second = PAIRS[data]
    if order == "B,A":
        first, second = second, first
    result = getattr(box_of(first), method)(box_of(second))
    lo = [min(a, b) for a, b in zip(fr(first[0]), fr(second[0]))]
    hi = [max(a, b) for a, b in zip(fr(first[1]), fr(second[1]))]
    assert_exact(result, lo, hi, f"{data}: {method} {order}")


@pytest.mark.parametrize("count", [1, 3])
def test_box_hull_is_the_bounding_box_of_all(count):
    """``Box.box_hull`` of one box is that box, of three boxes (dyadic and
    non-dyadic mixed) the smallest box holding all three, exactly. Catches an
    upper bound taken from the lower bounds and a box left out."""
    pool = [_A, _B, _C]
    chosen = pool[:count]
    result = Box.box_hull([box_of(b) for b in chosen])
    lo = [min(vals) for vals in zip(*[fr(b[0]) for b in chosen])]
    hi = [max(vals) for vals in zip(*[fr(b[1]) for b in chosen])]
    assert_exact(result, lo, hi, f"box_hull of {count}")
    with pytest.raises(ValueError):
        Box.box_hull([])


_A_ZONO_MARK = ("Box.{} reads lb and ub of its operand without checking its "
                "type, so a Zono operand raises AttributeError where the "
                "unsupported combination must be refused")


@pytest.mark.parametrize("method, operand, refusal", [
    ("minkowski_sum", "box of dimension 1", ValueError),
    ("convex_hull", "box of dimension 1", ValueError),
    ("union", "box of dimension 1", ValueError),
    ("intersect", "box of dimension 1", ValueError),
    ("box_hull", "box of dimension 1", ValueError),
    pytest.param("minkowski_sum", "zonotope", NotImplementedError,
                 marks=known_bug(23, _A_ZONO_MARK.format("minkowski_sum"),
                                 raises=AttributeError)),
    pytest.param("convex_hull", "zonotope", NotImplementedError,
                 marks=known_bug(23, _A_ZONO_MARK.format("convex_hull"),
                                 raises=AttributeError)),
    pytest.param("intersect", "zonotope", NotImplementedError,
                 marks=known_bug(23, _A_ZONO_MARK.format("intersect"),
                                 raises=AttributeError)),
    pytest.param("box_hull", "zonotope", NotImplementedError,
                 marks=known_bug(23, _A_ZONO_MARK.format("box_hull"),
                                 raises=AttributeError)),
])
def test_set_operations_refuse_an_operand_they_cannot_combine(
        method, operand, refusal):
    """``minkowski_sum``, ``convex_hull``, ``union``, ``intersect`` and
    ``box_hull`` of a box of dimension 4 refuse a box of dimension 1 with
    ``ValueError`` (numpy would broadcast a dimension of 1 against every
    coordinate, so only the dimension check refuses it; a dimension of 3
    numpy refuses by itself), and refuse a zonotope operand, a combination
    they do not support, with ``NotImplementedError``. Catches a dropped
    dimension check, which returns a box of dimension 4 built from the one
    coordinate of the other operand."""
    box = box_of(_A)
    other = (Box(np.array([-0.5]), np.array([0.25]))
             if operand == "box of dimension 1"
             else Zono(np.zeros((4, 1)), np.eye(4)))
    with pytest.raises(refusal):
        if method == "box_hull":
            Box.box_hull([box, other])
        else:
            getattr(box, method)(other)


# --------------------------------------------------------------------------- #
# intersect                                                                     #
# --------------------------------------------------------------------------- #
INTERSECTIONS = {
    # overlapping, the first box's upper bound the smaller on coordinate 0
    "overlap": (([-1.0, 0.0, -2.0], [0.5, 2.0, 1.0]),
                ([-0.5, -1.0, 0.0], [1.0, 1.0, 3.0])),
    # both fix coordinate 1 at the same value
    "shared-fixed-coordinate": (([-1.0, 0.25, -2.0], [0.5, 0.25, 1.0]),
                                ([-0.5, 0.25, 0.0], [1.0, 0.25, 3.0])),
    # the boxes touch along coordinate 2 (a face in common)
    "touching": (([-1.0, 0.0, -2.0], [0.5, 2.0, 0.75]),
                 ([-0.5, -1.0, 0.75], [1.0, 1.0, 3.0])),
    # non-dyadic bounds
    "inexact": (([0.1, -0.3, 1 / 3], [0.7, 0.2, 2 / 3]),
                ([0.2, -1 / 3, 0.1], [0.3, 0.1, 0.4])),
    # disjoint along coordinate 1 by a quarter
    "disjoint": (([-1.0, 0.0, -2.0], [0.5, 0.5, 1.0]),
                 ([-0.5, 0.75, 0.0], [1.0, 1.0, 3.0])),
}


@pytest.mark.parametrize("order", ["first,second", "second,first"])
@pytest.mark.parametrize("case", list(INTERSECTIONS))
def test_intersect_is_the_exact_intersection(case, order):
    """``Box.intersect`` is ``[max(lb1, lb2), min(ub1, ub2)]`` exactly when that
    is not empty, including a result of zero width in a coordinate (a fixed
    coordinate both boxes share, or a face where they touch), and None
    exactly when some coordinate has ``max(lb) > min(ub)``. A wrong None is
    a false 'disjoint'. Catches a lower bound taken from an upper bound, and
    an emptiness test that reads a zero-width result as empty."""
    first, second = INTERSECTIONS[case]
    if order == "second,first":
        first, second = second, first
    lo = [max(a, b) for a, b in zip(fr(first[0]), fr(second[0]))]
    hi = [min(a, b) for a, b in zip(fr(first[1]), fr(second[1]))]
    empty = any(a > b for a, b in zip(lo, hi))
    if empty is not (case == "disjoint"):
        pytest.fail(f"test data: case {case} has empty={empty}")
    result = box_of(first).intersect(box_of(second))
    if empty:
        assert result is None, f"{case}: disjoint boxes gave {result!r}"
    else:
        assert result is not None, f"{case}: intersecting boxes gave None"
        assert_exact(result, lo, hi, f"{case}: intersect {order}")


# --------------------------------------------------------------------------- #
# single_partition, partition                                                   #
# --------------------------------------------------------------------------- #
def assert_pieces_tile(pieces, lo, hi, count, what):
    """The boxes ``pieces`` are ``count`` boxes inside ``[lo, hi]`` whose union
    is all of it, decided exactly: every piece lies in the box, and every
    cell of the grid that the pieces' own faces cut the box into has its
    centre in some piece (a closed box whose faces lie on the grid holds a
    cell whole or misses its interior, so the closed pieces then cover the
    closed box)."""
    assert len(pieces) == count, f"{what}: {len(pieces)} pieces, " \
                                 f"expected {count}"
    lo, hi = fr(lo), fr(hi)
    parts = [stored(p) for p in pieces]
    outside = [k for k, (plo, phi) in enumerate(parts)
               if any(a < l or b > h or a > b
                      for a, b, l, h in zip(plo, phi, lo, hi))]
    assert not outside, f"{what}: pieces {outside} reach outside the box"
    centres = []
    for k in range(len(lo)):
        cuts = sorted({lo[k], hi[k]} | {p[0][k] for p in parts}
                      | {p[1][k] for p in parts})
        centres.append([(a + b) / 2 for a, b in zip(cuts, cuts[1:])]
                       or [cuts[0]])
    missed = [x for x in itertools.product(*centres)
              if not any(all(plo[k] <= x[k] <= phi[k] for k in range(len(lo)))
                         for plo, phi in parts)]
    assert not missed, (f"{what}: {len(missed)} cells of the box are in no "
                        f"piece, e.g. around {[float(v) for v in missed[0]]}")


_PARTITION_BOXES = {
    "dyadic": ([-1.0, 0.5, -0.25, 2.0], [0.5, 1.5, 1.75, 3.0]),
    # 0.1 + 3 * ((1.0 - 0.1) / 3) and -2.2 + 3 * ((1e-3 + 2.2) / 3) land one
    # ulp below the upper bound in round-to-nearest
    "inexact": ([0.1, -0.3, -2.2, 0.2], [1.0, 0.2, 1e-3, 0.9]),
    "zero-width": ([-1.0, 0.25, -0.25, 2.0], [0.5, 0.25, 1.75, 3.0]),
}


def assert_cut_along(pieces, lo, hi, dims, counts, what):
    """Each coordinate in ``dims`` is cut into its own count: along it the
    pieces take that many distinct intervals (one for a zero-width
    coordinate, which every piece keeps whole), each in the same number of
    pieces; every other coordinate is the box's own in every piece. Tiling
    alone cannot see a cut along the wrong coordinate or with another
    coordinate's count."""
    lo, hi = fr(lo), fr(hi)
    parts = [stored(p) for p in pieces]
    for k in range(len(lo)):
        seen = [(p[0][k], p[1][k]) for p in parts]
        if k not in dims:
            assert set(seen) == {(lo[k], hi[k])}, (
                f"{what}: coordinate {k}, not partitioned, takes the "
                f"intervals {sorted({(float(a), float(b)) for a, b in seen})}")
            continue
        n = counts[list(dims).index(k)] if lo[k] < hi[k] else 1
        distinct = set(seen)
        assert len(distinct) == n and all(
            seen.count(iv) == len(parts) // n for iv in distinct), (
            f"{what}: coordinate {k} is cut into "
            f"{sorted((float(a), float(b)) for a, b in distinct)}, "
            f"not into {n} intervals")


@pytest.mark.parametrize("name, dim, parts", [
    ("dyadic", 0, 2), ("dyadic", 0, 3), ("dyadic", 2, 2), ("dyadic", 2, 3),
    ("inexact", 0, 3), ("inexact", 2, 3)])
def test_single_partition_tiles_the_box(name, dim, parts):
    """``Box.single_partition(dim, parts)`` returns ``parts`` boxes inside the
    box whose union is the whole box, exactly, cut along ``dim`` only, on
    dyadic and non-dyadic bounds. The first and an inner coordinate, two
    pieces (no inner piece) and three (an inner piece); with non-dyadic
    bounds three parts make ``lo + 3 * ((hi - lo) / 3)`` land below ``hi``.
    Catches the last slice left out, an outer cut point that lands an ulp
    inside the box's own bound, and a cut along another coordinate."""
    lo, hi = _PARTITION_BOXES[name]
    pieces = box_of((lo, hi)).single_partition(dim, parts)
    what = f"{name}: single_partition({dim}, {parts})"
    assert_pieces_tile(pieces, lo, hi, parts, what)
    assert_cut_along(pieces, lo, hi, [dim], [parts], what)


_PARTITION_REQUESTS = [([0, 2], [3, 2]), ([2, 0], [2, 3]),
                       ([0, 2, 3], [2, 3, 2]), ([1], [4]), ([3], [1])]


@pytest.mark.parametrize("name, k", [
    pytest.param(name, k, id=f"{name}-dims{k}-counts{k}")
    for name, k in [("dyadic", 0), ("dyadic", 1), ("dyadic", 2),
                    ("dyadic", 3), ("dyadic", 4), ("inexact", 0),
                    ("inexact", 2), ("zero-width", 3)]])
def test_partition_tiles_the_box(name, k):
    """``Box.partition(dims, counts)`` over one, two and three dimensions, in
    increasing and in decreasing order, returns the product of the counts of
    boxes, inside the box, whose union is the whole box, exactly, each
    coordinate cut into its own count; a count of 1 gives the box; a
    zero-width coordinate partitioned gives pieces of zero width that still
    cover it. Catches the pieces of a later dimension overwriting the list
    instead of extending it, and a dimension partitioned with another
    dimension's count."""
    dims, counts = _PARTITION_REQUESTS[k]
    lo, hi = _PARTITION_BOXES[name]
    pieces = box_of((lo, hi)).partition(dims, counts)
    what = f"{name}: partition({dims}, {counts})"
    assert_pieces_tile(pieces, lo, hi, int(np.prod(counts)), what)
    assert_cut_along(pieces, lo, hi, dims, counts, what)


def test_partition_refuses_malformed_requests():
    """A partition index outside the box, a count below 1, and index and
    count lists of different lengths are refused."""
    box = box_of(_A)
    with pytest.raises(ValueError):
        box.single_partition(4, 2)
    with pytest.raises(ValueError):
        box.single_partition(0, 0)
    with pytest.raises(ValueError):
        box.partition([0, 1], [2])


# The middle coordinate of a 3-dimensional box: a width that overflows to
# infinity (each bound is finite), an infinite upper end, an infinite lower
# end.
_EXTREME_COORDINATES = {
    "overflowing width": (-1.5e308, 1.5e308),
    "infinite upper end": (-0.75, np.inf),
    "infinite lower end": (-np.inf, 0.75),
}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("name", [
    "overflowing width", "infinite upper end",
    pytest.param("infinite lower end", marks=known_bug(
        8, "Box.single_partition of a coordinate whose lower bound is -inf "
           "computes the cut points as -inf + k * inf, which is NaN, so the "
           "pieces have NaN bounds, hold no point, and the box is in no "
           "piece")),
])
def test_single_partition_of_a_huge_or_unbounded_coordinate_covers_it(name):
    """``single_partition(1, 3)`` of a coordinate whose width ``hi - lo``
    overflows returns three boxes inside the box whose union is the box;
    with an infinite end it either refuses with ``ValueError`` or does the
    same. Decided without arithmetic: along the partitioned coordinate the
    pieces, in order, start at the box's lower bound, end at its upper bound
    and share every cut point, no bound is NaN, and the other coordinates
    are the box's own. numpy's RuntimeWarnings (overflow, an invalid value)
    are let through, so the answer is the one a user gets. Catches cut
    points no longer clipped into the box (the overflowing width then gives
    a piece whose lower bound is above its upper bound) and NaN cut
    points."""
    end_lo, end_hi = _EXTREME_COORDINATES[name]
    lo = [-1.0, end_lo, 0.25]
    hi = [2.0, end_hi, 0.25]
    box = Box(np.array(lo), np.array(hi))
    try:
        pieces = box.single_partition(1, 3)
    except ValueError:
        if name == "overflowing width":
            raise
        return
    assert len(pieces) == 3, f"{name}: {len(pieces)} pieces"
    ends = [(float(p.lb[1, 0]), float(p.ub[1, 0])) for p in pieces]
    assert not np.isnan(ends).any(), f"{name}: pieces {ends}"
    for p in pieces:
        assert [p.lb[k, 0] for k in (0, 2)] == [lo[0], lo[2]] \
            and [p.ub[k, 0] for k in (0, 2)] == [hi[0], hi[2]], (
            f"{name}: a piece changed another coordinate: {p.lb.ravel()}, "
            f"{p.ub.ravel()}")
    ends.sort()
    assert ends[0][0] == end_lo and ends[-1][1] == end_hi, (
        f"{name}: the pieces {ends} do not reach both ends of "
        f"[{end_lo}, {end_hi}]")
    assert all(a[1] == b[0] for a, b in zip(ends, ends[1:])), (
        f"{name}: the pieces {ends} leave a gap or overlap past a cut")


# --------------------------------------------------------------------------- #
# contains, sample, get_vertices, __repr__                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("data", list(PAIRS))
def test_contains_answers_by_the_bounds(data):
    """``Box.contains`` accepts every corner of the box (closed bounds) and
    rejects the corner moved one ulp out along each coordinate of positive
    or zero width, as a flat vector and as a column; it rejects a point with
    a NaN coordinate (no number lies in an interval), or refuses it with
    ``ValueError``; it refuses with ``ValueError`` a point of another
    dimension, among them a point of one entry, which numpy would broadcast
    against every coordinate. Catches strict comparisons, bounds compared
    with the wrong side, a test for a coordinate outside that lets a NaN
    through, and a dropped dimension check."""
    lo, hi = PAIRS[data][0]
    box = box_of((lo, hi))
    for c in corners(fr(lo), fr(hi)):
        y = np.array([float(v) for v in c])
        assert box.contains(y) and box.contains(y.reshape(-1, 1)), y
    for k in range(len(lo)):
        for base, direction in ((np.array(lo), -np.inf), (np.array(hi), np.inf)):
            y = base.copy()
            y[k] = np.nextafter(y[k], direction)
            assert not box.contains(y), f"{data}: {y} is outside the box"
    y = np.array(lo, dtype=np.float64)
    y[1] = np.nan
    try:
        held = box.contains(y)
    except ValueError:
        held = False
    assert held is False, f"{data}: the point {y} is reported in the box"
    with pytest.raises(ValueError):
        box.contains(np.zeros(len(lo) + 1))
    with pytest.raises(ValueError):
        box.contains(np.array([float(lo[0])]))


@pytest.mark.parametrize("data", list(PAIRS))
def test_sample_returns_points_of_the_box(data):
    """``Box.sample(N)`` returns a (dim, N) array of points inside the box,
    decided in exact arithmetic (a zero-width coordinate is its value)."""
    lo, hi = PAIRS[data][0]
    pts = box_of((lo, hi)).sample(64)
    assert pts.shape == (len(lo), 64)
    lo_q, hi_q = fr(lo), fr(hi)
    bad = [j for j in range(64)
           if any(not (l <= Fraction(float(v)) <= h)
                  for v, l, h in zip(pts[:, j], lo_q, hi_q))]
    assert not bad, f"{data}: samples {bad[:5]} are outside the box"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@known_bug(8, "Box.sample of a box with an infinite bound computes "
              "(ub - lb) * r + lb, which is NaN or infinite, and returns "
              "those as sample points instead of refusing: none of them is a "
              "point of the box")
def test_sample_of_a_box_with_an_infinite_bound_refuses_or_gives_its_points():
    """``Box.sample(N)`` of a box with an infinite lower bound either refuses
    with ``ValueError`` (no uniform sample exists) or returns ``N`` points of
    the box: finite numbers inside every bound. numpy's RuntimeWarnings are
    let through, so the answer is the one a user gets. The passing neighbour
    is :func:`test_sample_returns_points_of_the_box`."""
    box = Box(np.array([-np.inf, 0.25]), np.array([0.5, 0.75]))
    try:
        pts = box.sample(16)
    except ValueError:
        return
    if pts.shape != (2, 16):
        pytest.fail(f"samples of shape {pts.shape}")
    ok = (np.isfinite(pts).all(axis=0) & (pts[0] <= 0.5)
          & (pts[1] >= 0.25) & (pts[1] <= 0.75))
    assert ok.all(), (f"{int((~ok).sum())} of 16 samples are no points of "
                      f"the box, e.g. {pts[:, ~ok][:, 0]}")


@pytest.mark.parametrize("data", list(PAIRS))
def test_get_vertices_are_the_corners(data):
    """``Box.get_vertices`` returns each corner of the box once, as columns
    (a zero-width coordinate halves the count)."""
    lo, hi = PAIRS[data][0]
    got = box_of((lo, hi)).get_vertices()
    expected = corners(fr(lo), fr(hi))
    assert got.shape == (len(lo), len(expected))
    assert sorted(tuple(fr(col)) for col in got.T) == expected


def test_repr_names_the_dimension_and_bound_ranges():
    """``repr`` of a box states its dimension and the range of its lower and
    upper bounds."""
    text = repr(box_of(_A))
    assert text.startswith("Box(dim=4,") and "lb_range=[-1.000, 2.000]" in text \
        and "ub_range=[-0.250, 3.000]" in text, text


@known_bug(8, "Box.__repr__ takes the minimum and maximum of the bound "
              "arrays, which numpy refuses for a box of dimension 0 "
              "(zero-size array to reduction operation), so printing or "
              "logging such a box raises ValueError", raises=ValueError)
def test_repr_of_a_box_of_dimension_zero_names_it():
    """The constructor builds a box of dimension 0 from empty bounds; its
    ``repr`` states that dimension. The passing neighbour is
    :func:`test_repr_names_the_dimension_and_bound_ranges`."""
    box = Box(np.zeros(0), np.zeros(0))
    if box.dim != 0:
        pytest.fail(f"a box of dimension {box.dim}")
    assert repr(box).startswith("Box(dim=0")


# --------------------------------------------------------------------------- #
# Composition: Linear, ReLU, Linear on several boxes                            #
# --------------------------------------------------------------------------- #
def _chain():
    torch.manual_seed(0)
    first = nn.Linear(4, 3)
    second = nn.Linear(3, 2)
    with torch.no_grad():
        first.weight.copy_(torch.tensor(_W, dtype=torch.float32))
        first.bias.copy_(torch.tensor(_b, dtype=torch.float32))
        second.weight.copy_(torch.tensor([[1.0, -0.5, 0.25],
                                          [-0.75, 1.0, 0.5]]))
        second.bias.copy_(torch.tensor([0.25, -0.125]))
    return nn.Sequential(first, nn.ReLU(), second)


@pytest.mark.parametrize("entry", ["reach_layer", "reach_model"])
def test_linear_relu_linear_on_several_boxes_holds_every_output(entry):
    """Three boxes that tile an input box (cut at dyadic points), each
    reached through Linear (``Box.affine_map``), ReLU and Linear, by the
    dispatcher layer by layer and by the model entry point: every float64
    output of the true network at 128 points of the box (corners and quarter
    grid) is in one of the returned boxes. Catches the bias of the affine map
    dropped and a box of one input set reused for another. The data are
    dyadic, so no rounding shows here; the outward rounding of the map is
    checked by the non-dyadic cases of
    :func:`test_affine_map_contains_the_exact_image`."""
    lo, hi = flat_box(4, seed=5)
    model = _chain()
    sets = split_cover("Box", lo, hi, parts=3)
    if entry == "reach_layer":
        out = sets
        for k, layer in enumerate(model):
            out = reach_one_layer(layer, out, "approx", f"layer {k}")
    else:
        out = reach_model(model, sets, "approx", "linear relu linear")
    assert len(out) == 3
    ys = replay_torch(model, samples(lo, hi), (1, 4))
    assert_covered(out, ys, f"linear relu linear via {entry}")
