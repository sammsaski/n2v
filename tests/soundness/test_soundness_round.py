"""Soundness cells for the rounding operations Round, Floor and Ceil.

The three are ONNX operators; ``load_onnx`` turns each node into onnx2torch's
``OnnxRound`` module, which is what the cells pass to ``reach_layer`` (one
layer) and what ``reach_pytorch_model`` meets inside a graph (Gemm or Conv
before and after). The kernels are ``round_star``, ``round_zono`` and
``round_box`` in ``n2v/nn/layer_ops/round_reach.py``; every one is an
over-approximation, so the soundness cells use ``method='approx'``. What
``'exact'`` does: through the model entry on a Star,
``tests/contracts/test_reach_coverage.py::test_exact_refuses_an_operation_with_no_exact_rule``;
on the other set types, and through ``reach_layer`` on a Zono and a Box, the
cells near the end of this file. An ImageZono input is
``test_rounding_op_on_an_image_set_keeps_the_pixel_order`` in the same
contracts file. A torch model that calls ``torch.round``, ``torch.floor`` or
``torch.ceil`` is refused today (one cell here holds the promise).

Oracle: the exact image in rationals (``tests/_oracles/rational.py``:
``round_`` rounds a half to the even neighbour as ``torch.round`` and ONNX do,
``floor`` and ``ceil`` keep the open ends of their branches), read from the
input set's stored arrays and never from n2v. A set that is a product of
intervals (Box, Star and Zono from bounds, an ImageStar from pixel bounds,
the parts of a split box) has as image the product of its coordinates'
images, each computed by the oracle in one dimension; a set whose entries
share two predicate variables goes through the oracle's polytope pipeline.
Every vertex of the image's hull must be in one returned set, decided in
fractions with no slack. Sample points (dyadic, with the ties and integers on
the grid) are replayed in float64 torch or float32 onnxruntime, both exact on
these data, and checked by one LP on the set's own rows.

Layout: a network tensor is (C, H, W) and the bounds and samples here are in
that order; an image set stores (H, W, C), and the image cells compare the
exact image in the stored order of the input after checking that every
result is an ImageStar of the input's height, width and channels.
"""
from fractions import Fraction
from itertools import product as _cartesian

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper
from onnx2torch.node_converters.roundings import OnnxRound

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests.cell_support import (assert_covered, flat_set, image_set,
                                image_split_cover, pixel_box, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx, split_cover)
from tests.known_bugs import known_bug

_OPS = {"Round": R.round_, "Floor": R.floor, "Ceil": R.ceil}

# Coordinates: across a tie (-0.5 rounds to 0), from a tie to a tie (0.5 and
# 1.5 round to 0 and 2), a point at a tie (2.5 rounds to 2), an interval
# whose rounded value is one constant, negative across integers, from an
# integer to an integer (the ends where Floor and Ceil jump) far beyond
# [-1, 1] and wider than 2 (the parts of a split cut this one), a point at a
# negative tie. The rounding errors at the ends differ in sign between the
# coordinates, so a shared error term is caught at a corner.
_LO = np.array([-0.75, 0.5, 2.5, 0.125, -2.25, -3.0, -1.5])
_HI = np.array([0.25, 1.5, 2.5, 0.375, -0.75, 5.0, -1.5])
# The exact image is the verdict of every cell; 32 replayed samples a cell are
# the uniform extra check.
_N = 32



# --------------------------------------------------------------------------- #
# The exact image of a product of intervals                                    #
# --------------------------------------------------------------------------- #
def _box_of(input_set):
    """``(lo, hi)``: the exact interval of every entry of a set that is a
    product of intervals, in the order the set stores its entries, read from
    its stored arrays (``rational.star``). Raises ``ValueError`` when the set
    is not such a product (a row or a predicate variable couples entries)."""
    c, G, C, d, lb, ub = R.star(input_set)
    low, high = list(lb), list(ub)
    for row, rhs in zip(C, d):
        used = [j for j, v in enumerate(row) if v != 0]
        if not used:
            if rhs < 0:
                raise ValueError("the input set is empty")
            continue
        if len(used) > 1:
            raise ValueError("a constraint row couples predicate variables")
        j = used[0]
        bound = rhs / row[j]
        if row[j] > 0:
            high[j] = bound if high[j] is None else min(high[j], bound)
        else:
            low[j] = bound if low[j] is None else max(low[j], bound)
    owner = {}
    lo, hi = [], []
    for i, (ci, g) in enumerate(zip(c, G)):
        used = [j for j, v in enumerate(g) if v != 0]
        if not used:
            lo.append(ci)
            hi.append(ci)
            continue
        if len(used) > 1 or used[0] in owner:
            raise ValueError("a predicate variable drives two entries")
        j = used[0]
        owner[j] = i
        if low[j] is None or high[j] is None:
            raise ValueError("an entry is unbounded")
        ends = (ci + g[j] * low[j], ci + g[j] * high[j])
        lo.append(min(ends))
        hi.append(max(ends))
    return lo, hi


def _coordinate_images(op, lo, hi):
    """The exact image of each interval ``[lo_i, hi_i]`` under the scalar
    operation, as the oracle's one-dimensional pieces."""
    return [op(R.box([l], [h])) for l, h in zip(lo, hi)]


def _assert_holds_product_image(op, input_sets, sets, what, pieces=True,
                                order=None):
    """Every input set is a product of intervals; its exact image is the
    product of the coordinates' images. With ``pieces`` each product of one
    piece per coordinate (a piece of the exact image) must have all its
    vertices in one returned set; without, the hull of the image (the box of
    the coordinates' end values) must lie in one returned set, which is what
    a method returning one convex set per input promises. ``order`` maps a
    vertex in the input's stored order to the order the results store."""
    assert sets, f"{what}: no set returned"
    holders = [R.Membership(R.star(s)) for s in sets]
    for k, s in enumerate(input_sets):
        images = _coordinate_images(op, *_box_of(s))
        if pieces:
            groups = _cartesian(*[[sorted({y[0] for y in p.image()})
                                   for p in coord] for coord in images])
        else:
            hull = [sorted({y[0] for p in coord for y in p.image()})
                    for coord in images]
            groups = [[[v[0], v[-1]] if len(v) > 1 else v for v in hull]]
        for group in groups:
            corners = [tuple(y) for y in _cartesian(*group)]
            if order is not None:
                corners = [tuple(y[i] for i in order) for y in corners]
            if any(all(h(y) for y in corners) for h in holders):
                continue
            outside = [y for y in corners if not any(h(y) for h in holders)]
            shown = ([float(v) for v in outside[0]] if outside else
                     "none; its vertices are spread over several sets")
            raise AssertionError(
                f"{what}: input set {k}: a piece of the exact image is in no "
                f"one returned set ({len(sets)} sets); {len(outside)} of its "
                f"{len(corners)} vertices are in none, first {shown}")


# --------------------------------------------------------------------------- #
# The exact image of a set on a few predicate variables                        #
# --------------------------------------------------------------------------- #
def _input_pieces(input_set):
    """The identity on ``input_set`` as rational pieces: its predicate
    polytope (rows and bounds, read exactly from the stored arrays) mapped
    through its centre and generators. No n2v method is called."""
    c, G, C, d, lb, ub = R.star(input_set)
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
    return R.affine(R.polytope(A, b), G, c)


def _assert_holds_exact_image(pieces, sets, what):
    """Every piece of the exact image has all its vertices in one returned
    set, in fractions (a convex set that holds the vertices holds the
    piece)."""
    assert sets, f"{what}: no set returned"
    holders = [R.Membership(R.star(s)) for s in sets]
    bad = []
    for i, piece in enumerate(pieces):
        image = piece.image()
        if any(all(h(y) for y in image) for h in holders):
            continue
        outside = [y for y in image if not any(h(y) for h in holders)]
        bad.append((i, len(image), outside))
    if bad:
        i, total, outside = bad[0]
        shown = ([float(v) for v in outside[0]] if outside
                 else "none; its vertices are spread over several sets")
        raise AssertionError(
            f"{what}: {len(bad)} of {len(pieces)} pieces of the exact image "
            f"are not held by any one returned set ({len(sets)} sets); piece "
            f"{i}: {len(outside)} of its {total} vertices in no set, first "
            f"{shown}")


def _grid(k, steps=5):
    values = np.linspace(-1.0, 1.0, steps)
    return [np.array(a) for a in _cartesian(*[values] * k)]


# Five entries on two predicate variables a in [-1, 1]^2, each over at most
# two rounding boundaries: entry 0 in [-1, 1/2], entry 1 in [0, 3/2] (from
# an integer, through a tie), entry 2 in [-1/4, 3/4], entry 3 in [3/4, 7/4],
# entry 4 the constant 2.5 (a tie). A rounded value tied to the wrong
# predicate variable, or an error term shared between entries, moves a
# joint output.
_RC = np.array([-0.25, 0.75, 0.25, 1.25, 2.5])
_RG = np.array([[0.5, 0.25], [0.5, -0.25], [0.25, 0.25], [0.0, 0.5],
                [0.0, 0.0]])


def _round_star(extra=(), box=True):
    """``(star, points)``: the five entries as a ``Star`` with the box rows
    and ``extra`` rows ``(row, rhs)``; without ``box`` the predicate bounds
    are left out and the box rows alone bound the variables. ``points`` are
    the member points of the quarter grid of the predicate box (exact)."""
    rows = list(extra)
    C = np.vstack([np.eye(2), -np.eye(2)] + [r for r, _ in rows])
    d = np.concatenate([np.ones(4), [v for _, v in rows]]).reshape(-1, 1)
    V = np.hstack([_RC[:, None], _RG])
    star = (Star(V, C, d, -np.ones((2, 1)), np.ones((2, 1))) if box
            else Star(V, C, d))
    points = [_RC + _RG @ a for a in _grid(2, 9)
              if all(r @ a <= v for r, v in rows)]
    return star, points


# Two entries on three predicate variables (more variables than entries,
# the shape a relaxation leaves), each variable over an interval other than
# [-1, 1]: a0 in [-2, 3/2], a1 in [-1, 2], a2 in [-3/2, 1], and the row
# a0 - a1 + a2 <= 1, which cuts entry 1's maximum from 7/2 to 23/8 (Round
# then gives 3, not 4). Entry 0 in [-5/2, 9/4], entry 1 in [-15/8, 23/8]:
# far beyond [-1, 1] and wider than 2.
_WC = np.array([0.25, 1.5])
_WG = np.array([[1.0, 0.0, 0.5], [0.5, -1.0, 0.25]])
_WLO = np.array([-2.0, -1.0, -1.5])
_WHI = np.array([1.5, 2.0, 1.0])
_WROW = np.array([1.0, -1.0, 1.0])

# Two entries on three generators (more generators than entries), entry 0 in
# [-3/2, 3], entry 1 in [-17/4, 5/4].
_ZC = np.array([0.75, -1.5])
_ZG = np.array([[1.5, -0.5, 0.25], [0.5, 1.0, -1.25]])


def _wide_star(box):
    """``(star, points)``: the three-variable Star above, with its variable
    intervals as the predicate box (``box``) or as rows only; ``points`` are
    the member points of the half-unit grid of the variables' box."""
    C = np.vstack([np.eye(3), -np.eye(3), _WROW[None, :]])
    d = np.concatenate([_WHI, -_WLO, [1.0]]).reshape(-1, 1)
    V = np.hstack([_WC[:, None], _WG])
    star = (Star(V, C, d, _WLO.reshape(-1, 1), _WHI.reshape(-1, 1)) if box
            else Star(V, C, d))
    grid = [np.array(a) for a in _cartesian(
        *[np.arange(lo, hi + 0.25, 0.5) for lo, hi in zip(_WLO, _WHI)])]
    return star, [_WC + _WG @ a for a in grid if _WROW @ a <= 1.0]


def _family(name):
    """``(input sets, member points)`` of a family whose entries share
    predicate variables."""
    if name == "coupled Star":
        star, points = _round_star()
        return [star], points
    if name == "Star with a binding row":
        # a0 + a1 <= 1/2 cuts off the corner (1, 1): entries 0, 2 and 3
        # lose their interval maximum
        star, points = _round_star([(np.array([1.0, 1.0]), 0.5)])
        return [star], points
    if name == "Star with no predicate bounds":
        star, points = _round_star(box=False)
        return [star], points
    if name == "two halves of a split":
        # cut at entry 0 = 0, as an earlier exact split leaves the star
        low, p1 = _round_star([(_RG[0], -_RC[0])])
        high, p2 = _round_star([(-_RG[0], _RC[0])])
        return [low, high], p1 + p2
    if name == "coupled Zono":
        zono = Zono(_RC.reshape(-1, 1), _RG.copy())
        return [zono], [_RC + _RG @ a for a in _grid(2, 9)]
    if name == "wide Star with more variables than entries":
        star, points = _wide_star(box=True)
        return [star], points
    if name == "the same Star by its rows alone":
        star, points = _wide_star(box=False)
        return [star], points
    if name == "wide Zono with more generators than entries":
        zono = Zono(_ZC.reshape(-1, 1), _ZG.copy())
        return [zono], [_ZC + _ZG @ a for a in _grid(3, 5)]
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# One layer through reach_layer                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("parts", [1, 3])
@pytest.mark.parametrize("kind", ["Box", "Star", "Zono"])
@pytest.mark.parametrize("op", sorted(_OPS))
def test_rounding_on_a_product_of_intervals_holds_the_exact_image(op, kind,
                                                                  parts):
    """Round, Floor and Ceil through ``reach_layer`` on a Box, a Star and a
    Zono built from bounds, one set and three parts of a split box at once.
    The coordinates cross ties and integers, start and end on them, sit on
    them as points, and include one whose rounded value is constant. Exact
    image in fractions and float64 samples by LP. Catches a rounded end on
    the wrong side (Floor and Ceil confused, a tie rounded away from even),
    an error interval of the wrong operation, a constant coordinate given
    the wrong value or a free variable in the wrong row, one error term
    shared by all coordinates (each keeps its range, the joint corners do
    not), and the outputs of all but one input set dropped (the three parts:
    on a Box nothing else sees it)."""
    sets = ([flat_set(kind, _LO, _HI)] if parts == 1
            else split_cover(kind, _LO, _HI, parts))
    layer = OnnxRound(op)
    what = f"{op} on {kind} ({len(sets)} sets)"
    out = reach_one_layer(layer, sets, "approx", what)
    _assert_holds_product_image(_OPS[op], sets, out, what, pieces=False)
    assert_covered(out, replay_torch(layer, samples(_LO, _HI, n=_N),
                                     (1, _LO.size)), what)


_FAMILIES = ["coupled Star", "Star with a binding row",
             "Star with no predicate bounds", "two halves of a split",
             "coupled Zono", "wide Star with more variables than entries",
             "the same Star by its rows alone",
             "wide Zono with more generators than entries"]


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("op", sorted(_OPS))
def test_rounding_on_a_coupled_set_holds_the_exact_image(op, family):
    """Round, Floor and Ceil on sets whose entries share predicate
    variables: a coupled Star, one with a binding row (its interval bounds
    are not its LP ranges), one with no predicate bounds (its box rows bound
    the variables), the two halves of the Star cut at entry 0 = 0 (two input
    sets at once), a coupled Zono, a Star with more predicate variables than
    entries whose variables range beyond [-1, 1] (with that box as its
    predicate bounds and, the same set, by its rows alone) and a Zono with
    more generators than entries, both with entries far beyond [-1, 1].
    Exact image in fractions. Catches a row ``z - x in [e_lo, e_hi]`` tied
    to another entry's generators or with the wrong sign, a rounded range
    taken from the interval bound instead of the LP range, the input's rows
    placed on the new variables, a missing predicate box filled with
    [-1, 1] (only the Star by its rows alone, whose variables leave that
    box, sees it), a rounding error added to the input's generators instead
    of a new one, and the outputs of one input set dropped."""
    sets, points = _family(family)
    layer = OnnxRound(op)
    what = f"{op} on {family}"
    out = reach_one_layer(layer, sets, "approx", what)
    pieces = [p for s in sets for p in _OPS[op](_input_pieces(s))]
    _assert_holds_exact_image(pieces, out, what)
    assert_covered(out, replay_torch(layer, points[::2],
                                     (1, sets[0].dim)), what)


# The Star of the LP-option cell: entry 0 = 5/4 + (a0 + a1) / 2, entry 1 =
# -1/2 + a0 - a1 / 4, a in [-1, 1]^2 cut by a0 + a1 <= 1/2. The row makes
# entry 0's range [1/4, 3/2] where the box gives [1/4, 9/4]. Its maximum 3/2
# is a tie (Round gives 2) that an optimum stopping inside the range rounds
# to 1; the row read with the wrong sign (a0 + a1 >= -1/2) moves the
# minimum to 1, which rounds to 1 and drops the output 0.
_LP_C = np.array([1.25, -0.5])
_LP_G = np.array([[0.5, 0.5], [1.0, -0.25]])


@pytest.mark.parametrize("lp_solver", ["SCS"])
def test_rounding_with_another_lp_solver_holds_the_exact_image(inexact_scs,
                                                               lp_solver):
    """Round with ``lp_solver='SCS'`` on a Star whose binding row decides a
    rounded range end; the ``inexact_scs`` fixture makes every SCS optimum
    2e-5 inward, as SCS is at its default tolerance, so the result holds the
    exact image only if the range LPs run on HiGHS whatever the caller names
    (``_bound_lp_solver``; ``'linprog'`` and ``'highs'`` take the same HiGHS
    path as the default solver of every other Star cell here). Exact image
    in fractions, from the vertices of the predicate polytope. Catches range
    LPs run on the caller's inexact solver (the tie 3/2 rounds to 1) and
    range LPs that read the rows with the wrong sign (the minimum 1/4 moves
    to 1)."""
    C = np.vstack([np.eye(2), -np.eye(2), [[1.0, 1.0]]])
    d = np.array([[1.0], [1.0], [1.0], [1.0], [0.5]])
    star = Star(np.hstack([_LP_C[:, None], _LP_G]), C, d, -np.ones((2, 1)),
                np.ones((2, 1)))
    what = f"Round on a Star with a binding row, lp_solver={lp_solver}"
    out = reach_one_layer(OnnxRound("Round"), [star], "approx", what,
                          lp_solver=lp_solver)
    _assert_holds_exact_image(R.round_(_input_pieces(star)), out, what)


# A (2, 2, 3) image: three pixels of positive width (across a tie, from an
# integer to a tie over more than four units, across integers) and nine
# points of distinct values, two of them ties, so a pixel read in the wrong
# place moves an output.
_IMG_LO = np.array([[[-0.75, 2.5, 0.25], [-1.0, 0.5, -2.0]],
                    [[0.125, -1.5, -0.25], [3.0, -2.5, -0.625]]])
_IMG_HI = np.array([[[0.25, 2.5, 0.25], [-1.0, 0.5, 2.5]],
                    [[0.125, -1.5, 1.75], [3.0, -2.5, -0.625]]])
_CHW = _IMG_LO.shape


@pytest.mark.parametrize("parts", [1, 2])
@pytest.mark.parametrize("op", sorted(_OPS))
def test_rounding_on_an_image_star_keeps_the_image(op, parts):
    """Round, Floor and Ceil on an ImageStar of shape (2, 2, 3), one set and
    two parts at once: every result is an ImageStar of the input's height,
    width and channels and holds the exact image in the stored (height,
    width, channel) order. Catches a result converted back with height and
    width exchanged (the flat entries are unchanged, so only the shape and a
    later spatial layer see it), left flat, or given another input's shape,
    and an image flattened in channel, height, width order and read back in
    the stored order (the pixels permuted)."""
    sets = ([image_set("ImageStar", _IMG_LO, _IMG_HI)] if parts == 1
            else image_split_cover("ImageStar", _IMG_LO, _IMG_HI, parts))
    layer = OnnxRound(op)
    what = f"{op} on ImageStar ({parts} sets)"
    out = reach_one_layer(layer, sets, "approx", what)
    shapes = {(type(s), s.height, s.width, s.num_channels) for s in out}
    assert shapes == {(ImageStar, 2, 3, 2)}, (
        f"{what}: results {shapes}, expected ImageStar 2x3x2")
    _assert_holds_product_image(_OPS[op], sets, out, what, pieces=False)
    assert_covered(out, replay_torch(layer, samples(_IMG_LO, _IMG_HI, n=_N),
                                     (1,) + _CHW), what, out_chw=_CHW)


# Entries that each stay inside one step of all three operations: [1/8, 3/8]
# (Round 0, Floor 0, Ceil 1) and [-15/8, -13/8] (Round -2, Floor -2,
# Ceil -1), on two predicate variables; and the same as a point.
_ONE_STEP_C = np.array([0.25, -1.75])
_ONE_STEP_G = np.array([[0.125, 0.0], [0.0625, 0.0625]])


@pytest.mark.parametrize("kind", ["a Star", "a point"])
@pytest.mark.parametrize("op", sorted(_OPS))
def test_rounding_of_entries_that_each_round_to_one_value(tmp_path, op,
                                                          kind):
    """Every entry of the Star lies inside one step of the operation, so the
    image is one point and ``round_star`` appends no variable and no row
    (its branch for no varying entry): a Star on two predicate variables
    through ``reach_layer``, and "a point", a Star with no predicate
    variable at all (``from_bounds`` of a zero-width box), through an ONNX
    file of the one node and ``reach_pytorch_model``. The exact image, in
    fractions, must be held. Catches the branch for no varying entry
    returning its input or the unrounded values."""
    what = f"{op} on {kind} whose entries each round to one value"
    if kind == "a Star":
        input_set = Star(np.hstack([_ONE_STEP_C[:, None], _ONE_STEP_G]),
                         np.vstack([np.eye(2), -np.eye(2)]), np.ones((4, 1)),
                         -np.ones((2, 1)), np.ones((2, 1)))
        out = reach_one_layer(OnnxRound(op), [input_set], "approx", what)
    else:
        input_set = flat_set("Star", _ONE_STEP_C, _ONE_STEP_C)
        path = save_onnx(tmp_path, [helper.make_node(op, ["X"], ["Y"])],
                         [1, 2], [1, 2], name=f"{op}_point")
        out = reach_model(load_onnx(path), input_set, "approx", what)
    if kind == "a Star":
        _assert_holds_exact_image(_OPS[op](_input_pieces(input_set)), out,
                                  what)
    else:
        _assert_holds_product_image(_OPS[op], [input_set], out, what,
                                    pieces=False)


@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_rounding_on_a_hexatope_or_octatope_is_refused(kind):
    """No rounding rule exists for these set types: ``reach_layer`` must
    raise ``NotImplementedError``, never return the input (a pass-through
    branch, as Cast has, leaves every output that differs from its input
    outside). The model entry is
    ``test_rounding_model_on_a_set_without_a_rule_is_refused``."""
    with pytest.raises(NotImplementedError):
        reach_layer(OnnxRound("Round"), [flat_set(kind, _LO, _HI)], "approx")


# --------------------------------------------------------------------------- #
# Inside a graph, through the model entry point                                 #
# --------------------------------------------------------------------------- #
_W1 = np.array([[1.0, 0.5], [0.5, -1.0], [-0.25, 1.0]])
_B1 = np.array([0.25, 0.5, -0.5])
_W2 = np.array([[1.0, -0.5, 0.25], [0.5, 1.0, -1.0]])
_B2 = np.array([0.0, 0.125])
_IN_LO = np.array([-1.0, -0.5])
_IN_HI = np.array([0.5, 1.0])


def _chain(tmp_path, op):
    f32 = lambda a: np.asarray(a, dtype=np.float32)      # noqa: E731
    return save_onnx(
        tmp_path,
        [helper.make_node("Gemm", ["X", "W1", "B1"], ["H"], transB=1),
         helper.make_node(op, ["H"], ["A"]),
         helper.make_node("Gemm", ["A", "W2", "B2"], ["Y"], transB=1)],
        [1, 2], [1, 2], {"W1": f32(_W1), "B1": f32(_B1), "W2": f32(_W2),
                         "B2": f32(_B2)}, name=f"{op}_chain")


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box",
                                  "Star with a binding row"])
@pytest.mark.parametrize("op", sorted(_OPS))
def test_rounding_between_two_affine_layers_holds_the_exact_image(tmp_path,
                                                                  op, kind):
    """Gemm, op, Gemm loaded with ``load_onnx`` and reached through
    ``reach_pytorch_model``: the rows the rounding emits are consumed by the
    next layer. Exact image in fractions and onnxruntime samples by LP.
    Catches a rounding relaxation whose rows keep each entry's range but
    tie it to the wrong input, and an error term or constant written to the
    wrong hidden neuron."""
    path = _chain(tmp_path, op)
    if kind == "Star with a binding row":
        C = np.vstack([np.eye(2), -np.eye(2), [[1.0, 1.0]]])
        d = np.array([[1.0], [1.0], [1.0], [1.0], [1.5]])
        centre, half = (_IN_LO + _IN_HI) / 2, (_IN_HI - _IN_LO) / 2
        input_set = Star(np.hstack([centre[:, None], np.diag(half)]), C, d,
                         -np.ones((2, 1)), np.ones((2, 1)))
        points = [centre + half * a for a in _grid(2, 9) if a.sum() <= 1.5]
    else:
        input_set = flat_set(kind, _IN_LO, _IN_HI)
        points = samples(_IN_LO, _IN_HI, n=_N)
    what = f"Gemm, {op}, Gemm on {kind}"
    out = reach_model(load_onnx(path), input_set, "approx", what)
    pieces = R.affine(_OPS[op](R.affine(_input_pieces(input_set), _W1, _B1)),
                      _W2, _B2)
    _assert_holds_exact_image(pieces, out, what)
    assert_covered(out, replay_onnx(path, points, [1, 2]), what)


@pytest.mark.parametrize("op", sorted(_OPS))
def test_rounding_between_two_convolutions_covers_onnxruntime(tmp_path, op):
    """Conv (1x1, mixing the channels), op, Conv (2x1 kernel) on a (2, 3, 5)
    ImageStar with its own interval per pixel, loaded with ``load_onnx``:
    every onnxruntime output is in a returned set, compared in the stored
    order of the (1, 2, 5) result. Catches a rounded result whose height and
    width are exchanged, which the next convolution reads in the wrong
    place."""
    shape = [1, 2, 3, 5]
    path = save_onnx(
        tmp_path,
        [helper.make_node("Conv", ["X", "K1"], ["H"], kernel_shape=[1, 1]),
         helper.make_node(op, ["H"], ["A"]),
         helper.make_node("Conv", ["A", "K2"], ["Y"], kernel_shape=[2, 1])],
        shape, [1, 1, 2, 5],
        {"K1": np.array([[[[1.0]], [[0.5]]], [[[-0.5]], [[1.0]]]],
                        dtype=np.float32),
         "K2": np.array([[[[1.0], [0.5]], [[-0.25], [1.0]]]],
                        dtype=np.float32)}, name=f"{op}_conv")
    rng = np.random.default_rng(3)
    lo = rng.integers(-8, 8, size=(2, 3, 5)) / 4.0
    hi = lo + rng.integers(1, 6, size=(2, 3, 5)) / 4.0
    what = f"Conv, {op}, Conv on ImageStar"
    out = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                      "approx", what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), shape), what,
                   out_chw=(1, 2, 5))


# --------------------------------------------------------------------------- #
# The exact method through reach_layer                                          #
# --------------------------------------------------------------------------- #
_EXACT_ZONO_IS_THE_RELAXATION = known_bug(
    18, "reach_layer with method='exact' on a Zono or an ImageZono runs the "
        "rounding relaxation (one error generator per entry) and returns it "
        "under the name exact: the Zono lane of the dispatcher does not read "
        "the method (REACH-8); "
        "the model entry refuses 'exact' on a Zono")
_EXACT_BOX_IS_ITS_BOUNDING_BOX = known_bug(
    18, "reach_layer with method='exact' on a Box does not read the method "
        "and returns the box of the rounded ends, which holds the points "
        "between the integers that no input reaches (REACH-8, as for Linear "
        "and Softmax); the model entry refuses 'exact' on a Box")


@pytest.mark.parametrize("kind", [
    pytest.param("Zono", marks=_EXACT_ZONO_IS_THE_RELAXATION),
    pytest.param("Box", marks=_EXACT_BOX_IS_ITS_BOUNDING_BOX),
    "Box of one value per coordinate"])
def test_exact_rounding_through_reach_layer_refuses_or_is_the_image(kind):
    """DESIGN: 'exact' is exact up to float64 rounding or raises.
    ``reach_layer(Round, sets, 'exact')``: a refusal (``NotImplementedError``
    or ``ValueError``, as the model entry gives) keeps the promise; an answer
    must hold the exact image and nothing else. Round takes finitely many
    values on a bounded set, so every returned set must be a single point
    that some input reaches (its one vertex, decided in fractions). On the
    box [-5/4, 7/4] x [1/4, 3/8] (images {-1, 0, 1, 2} and {0}) the Zono
    lane returns the relaxation and the Box lane the box from (-1, 0) to
    (2, 0), which holds (1/2, 0). The box [1/8, 3/8] x [5/2, 5/2], whose
    every coordinate rounds to one value, is answered exactly today: the
    neighbour that runs the same reference. The Zono lane serves an
    ImageZono too and the Star lane is marked through the model entry in
    ``tests/contracts/test_reach_coverage.py``. Catches the exact method
    returning an over-approximation under its name."""
    if kind == "Box of one value per coordinate":
        lo, hi = np.array([0.125, 2.5]), np.array([0.375, 2.5])
    else:
        lo, hi = np.array([-1.25, 0.25]), np.array([1.75, 0.375])
    input_set = flat_set("Zono" if kind == "Zono" else "Box", lo, hi)
    what = f"exact Round on a {kind}"
    try:
        out = reach_one_layer(OnnxRound("Round"), [input_set], "exact", what)
    except (NotImplementedError, ValueError):
        return
    pieces = R.round_(_input_pieces(input_set))
    _assert_holds_exact_image(pieces, out, what)
    for k, result in enumerate(out):
        vertices = R.set_vertices(R.star(result))
        unreached = R.unreached(pieces, vertices)
        assert len(vertices) == 1 and not unreached, (
            f"{what}: returned set {k} is not one point that an input "
            f"reaches: it has {len(vertices)} vertices, {len(unreached)} of "
            f"them reached by no input; the image is {len(pieces)} points")


# --------------------------------------------------------------------------- #
# Set types and methods without a rule, through the model entry                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,method", [
    ("Zono", "exact"), ("ImageZono", "exact"), ("Box", "exact"),
    ("Hexatope", "exact"), ("Octatope", "exact"),
    ("Hexatope", "approx"), ("Octatope", "approx")])
def test_rounding_model_on_a_set_without_a_rule_is_refused(tmp_path, kind,
                                                           method):
    """The rows of Round, Floor and Ceil that refuse, through
    ``reach_pytorch_model`` on an ONNX file: ``exact`` on a set type with no
    exact method is refused before any layer (today ``ValueError``, "...
    only supports 'approx'"; the planned ``UnsupportedError`` subclasses
    ``NotImplementedError``); ``approx`` on a Hexatope or an Octatope passes
    the first Gemm and reaches the rounding node, for which these lanes have
    no rule: ``NotImplementedError`` naming the layer. The three operations
    share every branch on the way, so Round stands for them. Catches the
    model entry handing 'exact' to a lane that does not read it (the
    relaxation returned under the name exact, cell above) and a Hexatope or
    Octatope lane that answers the rounding node with a rule of another
    layer or as a pass-through."""
    if kind == "ImageZono":
        shape = [1, 2, 3, 5]
        path = save_onnx(tmp_path, [helper.make_node("Round", ["X"], ["Y"])],
                         shape, shape, name="round_image")
        input_set = image_set(kind, *pixel_box(2, 3, 5))
        what = f"Round on a {kind}, {method}"
    else:
        path = _chain(tmp_path, "Round")
        input_set = flat_set(kind, _IN_LO, _IN_HI)
        what = f"Gemm, Round, Gemm on a {kind}, {method}"
    if method == "exact":
        with pytest.raises((NotImplementedError, ValueError)):
            reach_model(load_onnx(path), input_set, method, what)
    else:
        with pytest.raises(NotImplementedError,
                           match="not implemented for layer type: OnnxRound"):
            reach_model(load_onnx(path), input_set, method, what)


# --------------------------------------------------------------------------- #
# A torch model that calls the rounding functions                               #
# --------------------------------------------------------------------------- #
class _TorchRounding(nn.Module):
    """Linear, ``fn`` (torch.round, torch.floor or torch.ceil) called as a
    function, Linear, with the weights of the ONNX chain above."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.first, self.last = nn.Linear(2, 3), nn.Linear(3, 2)
        with torch.no_grad():
            for layer, W, b in ((self.first, _W1, _B1),
                                (self.last, _W2, _B2)):
                layer.weight.copy_(torch.as_tensor(W))
                layer.bias.copy_(torch.as_tensor(b))

    def forward(self, x):
        return self.last(self.fn(self.first(x)))


_TORCH_FN = {"Round": torch.round, "Floor": torch.floor, "Ceil": torch.ceil}


@pytest.mark.parametrize("op", sorted(_OPS))
def test_rounding_in_a_torch_model_refuses_or_holds_every_output(op):
    """A torch model whose forward calls ``torch.round``, ``torch.floor`` or
    ``torch.ceil``, traced by ``reach_pytorch_model`` (an ``OnnxRound``
    module inside a torch model is traced to the same function call). Today
    the graph walker has no rule for these functions and refuses
    (``NotImplementedError``) on a Star, a Zono and a Box. The promise:
    refuse, or hold the exact image (fractions) and every torch float64
    output. Catches a rule added for these functions that answers one of
    them with another's (Floor answered as Round leaves 48 of 128 outputs
    outside on a Star)."""
    model = _TorchRounding(_TORCH_FN[op])
    pieces = R.affine(_OPS[op](R.affine(R.box(_IN_LO.tolist(),
                                              _IN_HI.tolist()), _W1, _B1)),
                      _W2, _B2)
    for kind in ("Star", "Zono", "Box"):
        what = f"torch Linear, torch.{op.lower()}, Linear on {kind}"
        try:
            out = reach_model(model, flat_set(kind, _IN_LO, _IN_HI),
                              "approx", what)
        except NotImplementedError:
            continue
        _assert_holds_exact_image(pieces, out, what)
        assert_covered(out, replay_torch(model, samples(_IN_LO, _IN_HI),
                                         (1, 2)), what)


# --------------------------------------------------------------------------- #
# Data whose rounding error the kernels' own float arithmetic rounds            #
# --------------------------------------------------------------------------- #
_ZONO_CENTRE_TO_NEAREST = known_bug(
    "7b", "round_zono adds the midpoint of the error interval to the centre "
          "in round to nearest (Floor: c - 1/2, Ceil: c + 1/2): with "
          "c = 2^-60 the stored centre -1/2 is 2^-60 below the exact one, "
          "and the true output floor(1) = 1 lies 2^-60 above the returned "
          "zonotope (at c = 2^53 + 2 the gap is 1/4)")
_STAR_ROW_BOUND_TO_NEAREST = known_bug(
    7, "round_star writes the bounds of its rows z - y <= e_hi and "
       "y - z <= -e_lo as e_hi + c and -e_lo - c in round to nearest "
       "(Round: 1/2 + c and 1/2 - c; Ceil: 1 + c; Floor: 1 - c): with "
       "c = 2^-60 the bound 1/2 + 2^-60 is stored as 1/2, and the true output "
       "round(3/2) = 2 lies 2^-60 outside the returned star (at "
       "c = 2^53 + 2 the gap is 1/2)")

# One entry y = c + g0 a0 + g1 a1, a in [-1, 1]^2, with c = 2^-60 cancelled
# by the last generator, so that the largest value of y is the integer 1
# (Zono) or the tie 3/2 (Star) while c +- 1/2 and c +- 1 are not floats.
_TINY = 2.0 ** -60
_PINNED = {"Zono": (_TINY, [1.0 - 2.0 ** -53, 2.0 ** -53 - _TINY]),
           "Star": (_TINY, [1.5 - 2.0 ** -52, 2.0 ** -52 - _TINY])}


@pytest.mark.parametrize("kind,op", [
    pytest.param("Zono", "Floor", marks=_ZONO_CENTRE_TO_NEAREST),
    ("Zono", "Round"),
    pytest.param("Star", "Round", marks=_STAR_ROW_BOUND_TO_NEAREST),
    ("Star", "Floor")])
def test_rounding_holds_the_exact_image_of_data_its_arithmetic_rounds(kind,
                                                                      op):
    """The result must hold the exact image of the stored set in fractions,
    with no slack, also where the kernel's own float arithmetic rounds. On
    a Zono the kernel stores the centre c + (e_lo + e_hi) / 2, on a Star the
    row bounds e_hi + c and -e_lo - c. Floor on the Zono and Round on the
    Star round them inward today (strict expected failures, one per kernel;
    the class of ``ROUND-10`` for the Zono, of ``ROUND-1`` for the Star).
    Round on the same Zono adds 0 and Floor on the same Star rounds its one
    rounded bound outward: they pass and check the reference on the same
    data. Catches a centre or a row bound moved inward by its rounding."""
    c, g = _PINNED[kind]
    if kind == "Zono":
        input_set = Zono(np.array([[c]]), np.array([g]))
    else:
        input_set = Star(np.array([[c] + g]),
                         np.vstack([np.eye(2), -np.eye(2)]), np.ones((4, 1)),
                         -np.ones((2, 1)), np.ones((2, 1)))
    what = f"{op} on a pinned {kind}"
    out = reach_one_layer(OnnxRound(op), [input_set], "approx", what)
    _assert_holds_exact_image(_OPS[op](_input_pieces(input_set)), out, what)
