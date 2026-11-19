"""Soundness of the MatMul of two computed sets (the bilinear reach).

A MatMul whose two operands both depend on the input (the attention
products Q @ K^T and A @ V) is answered by the graph handler
``_handle_bilinear_matmul`` (``n2v/nn/reach.py``), which pairs the operand
pieces and calls one of the three kernels of
``n2v/nn/layer_ops/bilinear_matmul_reach.py``: each concretises both
operands to per-entry intervals, multiplies them with an outward interval
matmul and re-encodes the product as a box over new variables. Every
result must hold every true output.

The cells:

- the three kernels on box operands at the operand corners where every
  output entry takes its least and its greatest value, and on Stars known by
  their member points, products in fractions, exact membership on the rows
  of the returned set (``rational.Membership``); several operand pairs at
  once, Stars whose predicate box is not [-1, 1] or that have no predicate
  box, Stars cut by a binding row, a negative scale and a scale of
  magnitude above one, non-dyadic data and
  points whose products are not float64 numbers, plain, batched and
  broadcast shapes;
- the Box kernel on operand ranges whose products overflow or underflow;
- refusals of unknown options and wrong shapes;
- ``bounds='lp'`` on Stars whose cut rows bind the operand ranges, with
  every vertex pair of the two predicate polytopes, each solver;
- the graph handler through ``reach_pytorch_model`` with onnxruntime replay
  of member points and LP membership (dyadic data, so the float32 replay is
  exact): each kernel, the three ways of pairing pieces under ``exact``,
  operands split by two different ReLUs, batched heads, a residual Sub of
  the input after the product, image-set operands, a set set for which no
  kernel exists, a flat set without a tensor shape, and a torch model;
- the A @ V envelope on a value Star that has no predicate box.
"""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn.layer_ops.bilinear_matmul_reach import (
    av_envelope_star,
    bilinear_matmul_box,
    bilinear_matmul_star,
    bilinear_matmul_zono,
)
from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_image_star,
                                constrained_star, flat_box, flat_set,
                                image_set, pixel_box, reach_model,
                                replay_onnx, replay_torch, samples, save_onnx)
from tests.known_bugs import known_bug

_KERNEL = {"Box": bilinear_matmul_box, "Zono": bilinear_matmul_zono,
           "Star": bilinear_matmul_star}

# (left shape, right shape): a plain product with a non-square result, a
# batched product with a head axis on both sides, and a broadcast one.
_PAIR_SHAPES = {
    "(2,3)@(3,4)": ((2, 3), (3, 4)),
    "(2,2,3)@(2,3,2)": ((2, 2, 3), (2, 3, 2)),
    "(2,3)@(1,3,2)": ((2, 3), (1, 3, 2)),
}
# the shape of every family other than the dyadic one (the layouts are one
# code path each, run on the dyadic family)
_ONE_SHAPE = "(2,3)@(3,4)"


def _fractions(values):
    return [Fraction(float(v)) for v in np.asarray(values).reshape(-1)]


def _box_operand(kind, lo, hi):
    """The box [lo, hi] as an operand of ``kind``: ``(set, lo, hi, None)``
    with the ends as fractions. A Box is the box; ``Zono.from_bounds`` and
    ``Star.from_bounds`` widen the half-widths outward, so they hold it."""
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    if kind == "Box":
        s = Box(lo, hi)
    else:
        s = {"Zono": Zono, "Star": Star}[kind].from_bounds(
            lo.reshape(-1, 1), hi.reshape(-1, 1))
    return s, _fractions(lo), _fractions(hi), None


def _wide_predicate_operand(n, seed, predicate_box=True):
    """A Star ``x = c + diag(g) a`` whose predicate box is not [-1, 1]: the
    boxes [0, 3] and [0, 1.5] (one-sided, as an approx ReLU makes them),
    [-2.5, 0.5], [1, 2.5] and [-4, -1], generators of both signs. The set is
    exactly the box with ends ``c + g pl`` and ``c + g pu``, all dyadic.
    With ``predicate_box=False`` the same rows ``pl <= a <= pu`` are the
    only bounds of the variables (``predicate_lb`` and ``predicate_ub`` are
    None), so the Star's ranges come from its LP enclosure."""
    rng = np.random.default_rng(seed)
    c = rng.integers(-8, 9, size=n) / 8.0
    g = rng.choice([-1.5, -0.75, 0.5, 1.25, 2.0], size=n)
    boxes = [(0.0, 3.0), (-2.5, 0.5), (1.0, 2.5), (-4.0, -1.0), (0.0, 1.5)]
    order = rng.permutation(n)
    pl = np.array([boxes[i % len(boxes)][0] for i in order])
    pu = np.array([boxes[i % len(boxes)][1] for i in order])
    V = np.hstack([c[:, None], np.diag(g)])
    C = np.vstack([np.eye(n), -np.eye(n)])
    d = np.concatenate([pu, -pl]).reshape(-1, 1)
    if predicate_box:
        star = Star(V, C, d, pl.reshape(-1, 1), pu.reshape(-1, 1))
    else:
        star = Star(V, C, d)
    ends = np.stack([c + g * pl, c + g * pu])
    return star, _fractions(ends.min(axis=0)), _fractions(ends.max(axis=0)), \
        None


def _row_operand(lo, hi, seed):
    """A Star cut by a binding row that is not a box row
    (``constrained_star``), known by its member points."""
    star, points = constrained_star(lo, hi, n=24, seed=seed)
    return star, None, None, [_fractions(p) for p in points]


def _corner(lo, hi, rng):
    return [l if bit else h for l, h, bit in
            zip(lo, hi, rng.integers(0, 2, size=len(lo)))]


def _exact_product(a, b, lshape, rshape, scale):
    """``scale * numpy.matmul(a, b)`` in fractions, flattened row-major."""
    A = np.array(a, dtype=object).reshape(lshape)
    B = np.array(b, dtype=object).reshape(rshape)
    return [Fraction(scale) * v for v in np.matmul(A, B).reshape(-1)]


def _true_products(left, right, lshape, rshape, scale, rng):
    """Exact true outputs of one operand pair.

    Two box operands: for every output entry, the operand point at which it
    is least and the one at which it is greatest (each of its product terms
    at the corner of its own two intervals that minimises or maximises
    ``scale * a * b``; the other coordinates at their lower ends), the two
    extreme corners, and random corners. An operand known by member points
    contributes those points, each paired with points of the other."""
    _, llo, lhi, lpts = left
    _, rlo, rhi, rpts = right
    exact_scale = Fraction(scale)
    out = []
    if lpts is None and rpts is None:
        batch = np.broadcast_shapes(lshape[:-2], rshape[:-2])
        a_idx = np.broadcast_to(np.arange(len(llo)).reshape(lshape),
                                batch + tuple(lshape[-2:]))
        b_idx = np.broadcast_to(np.arange(len(rlo)).reshape(rshape),
                                batch + tuple(rshape[-2:]))
        rows, cols = lshape[-2], rshape[-1]
        for index in np.ndindex(*(batch + (rows, cols))):
            *lead, i, j = index
            a_ids = a_idx[tuple(lead) + (i, slice(None))]
            b_ids = b_idx[tuple(lead) + (slice(None), j)]
            for pick in (min, max):
                a, b = list(llo), list(rlo)
                for p, q in zip(a_ids, b_ids):
                    corners = [(x, y) for x in (llo[p], lhi[p])
                               for y in (rlo[q], rhi[q])]
                    a[p], b[q] = pick(
                        corners, key=lambda xy: exact_scale * xy[0] * xy[1])
                out.append(_exact_product(a, b, lshape, rshape, scale))
        out.append(_exact_product(llo, rlo, lshape, rshape, scale))
        out.append(_exact_product(lhi, rhi, lshape, rshape, scale))
        for _ in range(12):
            out.append(_exact_product(_corner(llo, lhi, rng),
                                      _corner(rlo, rhi, rng),
                                      lshape, rshape, scale))
        return out
    lefts = lpts if lpts is not None else [
        llo, lhi] + [_corner(llo, lhi, rng) for _ in range(6)]
    rights = rpts if rpts is not None else [
        rlo, rhi] + [_corner(rlo, rhi, rng) for _ in range(6)]
    for k in range(24):
        a = lefts[k % len(lefts)]
        b = rights[(7 * k + 3) % len(rights)]
        out.append(_exact_product(a, b, lshape, rshape, scale))
    return out


def _operand_pairs(kind, family, lshape, rshape):
    """``(pairs, options)``: the operand pairs a family hands the kernel at
    once, and the kernel's keyword options."""
    nl, nr = int(np.prod(lshape)), int(np.prod(rshape))
    if family == "pieces":
        # three pairs with different ranges: a kernel that keeps the first
        # pair's product only misses the others
        pairs = []
        for p in range(3):
            llo, lhi = flat_box(nl, seed=20 + p)
            rlo, rhi = flat_box(nr, seed=30 + p)
            pairs.append((_box_operand(kind, llo + 2 * p, lhi + 2 * p),
                          _box_operand(kind, (p + 1) * rlo - p,
                                       (p + 1) * rhi - p)))
        return pairs, {}
    if family == "non-dyadic":
        # three boxes with non-dyadic ends (the first is the draw this family
        # always had), and a pair of points whose exact products and sums are
        # not float64 numbers: a product rounded to nearest with no outward
        # term leaves one end of such an entry inside on every draw, where a
        # random box shows it only now and then
        rng = np.random.default_rng(7)
        pairs = []
        for _ in range(3):
            llo = rng.uniform(-1.0, 1.0, nl) / 3
            rlo = rng.uniform(-1.0, 1.0, nr) / 3
            pairs.append((_box_operand(kind, llo,
                                       llo + rng.uniform(0, 0.7, nl)),
                          _box_operand(kind, rlo,
                                       rlo + rng.uniform(0, 0.7, nr))))
        lpt = (np.arange(nl) + 1) / 3.0
        rpt = -(np.arange(nr) + 2) / 7.0
        pairs.append((_box_operand(kind, lpt, lpt),
                      _box_operand(kind, rpt, rpt)))
        return pairs, {}
    if family == "wide-predicate":
        return [(_wide_predicate_operand(nl, 1),
                 _wide_predicate_operand(nr, 2))], {}
    if family == "no-predicate-box":
        return [(_wide_predicate_operand(nl, 1, predicate_box=False),
                 _wide_predicate_operand(nr, 2, predicate_box=False))], {}
    if family == "binding-row":
        llo, lhi = flat_box(nl, seed=3, zero_width=True)
        rlo, rhi = flat_box(nr, seed=4, zero_width=True)
        return [(_row_operand(llo, lhi, seed=5),
                 _row_operand(rlo, rhi, seed=6))], {}
    llo, lhi = flat_box(nl, seed=1, zero_width=True)
    rlo, rhi = flat_box(nr, seed=2, zero_width=True)
    # two scales: a negative one of magnitude below one, and one of magnitude
    # above one. A scale applied the wrong way round (divided where it
    # multiplies) widens the result when the magnitude is below one, so it
    # shows only where a product range lies on one side of zero; above one it
    # narrows the result and every extreme product is outside.
    options = {"negative-scale": {"scale": -0.75},
               "scale-above-one": {"scale": 2.5}}.get(family, {})
    return [(_box_operand(kind, llo, lhi), _box_operand(kind, rlo, rhi))], \
        options


def _kernel_cells():
    cells = []
    for kind in ("Box", "Zono", "Star"):
        families = ["dyadic", "pieces", "non-dyadic", "negative-scale",
                    "scale-above-one"]
        if kind == "Star":
            families += ["wide-predicate", "binding-row", "no-predicate-box"]
        for family in families:
            shapes = list(_PAIR_SHAPES) if family == "dyadic" else [_ONE_SHAPE]
            for shape in shapes:
                cells.append(pytest.param(kind, family, shape,
                                          id=f"{kind}-{family}-{shape}"))
    return cells


@pytest.mark.parametrize("kind,family,shape", _kernel_cells())
def test_bilinear_kernel_holds_the_exact_products(kind, family, shape):
    """Every exact true product ``scale * (left @ right)`` of member points of
    each operand pair is a member of a returned set, decided exactly on its
    rows with zero slack, including the points where an output entry is
    least or greatest (where the interval product is tight).

    Catches: a bound computed from the wrong end of an operand (lower used
    for both ends, the signed midpoint in the radius, a dropped
    radius-times-radius term), a missing outward rounding of the interval
    product (the non-dyadic family, whose point operands have products that
    are not float64 numbers), an operand or the product reshaped in the
    wrong order (non-square, batched and broadcast shapes), a negative scale
    applied to one end only, a scale applied as its reciprocal
    (``scale-above-one``), a kernel that returns after the first operand
    pair or reuses one pair's bounds for another (``pieces``), Star ranges
    that assume a [-1, 1] predicate box (``wide-predicate``), and Star
    ranges taken wrongly from the LP enclosure of a Star that has no
    predicate box (``no-predicate-box``)."""
    lshape, rshape = _PAIR_SHAPES[shape]
    pairs, options = _operand_pairs(kind, family, lshape, rshape)
    scale = options.get("scale", 1.0)
    out = _KERNEL[kind]([p[0][0] for p in pairs], [p[1][0] for p in pairs],
                        lshape, rshape, **options)
    if not out:
        raise AssertionError(f"{kind} {family} {shape}: no output set")
    holders = [rational.Membership(rational.star(s)) for s in out]
    rng = np.random.default_rng(11)
    missed = []
    for p, (left, right) in enumerate(pairs):
        for y in _true_products(left, right, lshape, rshape, scale, rng):
            if not any(holder(y) for holder in holders):
                missed.append((p, [float(v) for v in y]))
    assert not missed, (
        f"{kind} {family} {shape}: {len(missed)} exact products are in none "
        f"of the {len(out)} returned sets, e.g. pair {missed[0][0]}: "
        f"{missed[0][1]}")


# --------------------------------------------------------------------------- #
# Overflowing and underflowing products (Box kernel), exact arithmetic          #
# --------------------------------------------------------------------------- #
_OVERFLOW = {
    # left (1, 2) @ right (2, 1); bounds as (lo, hi) pairs per coordinate.
    # The two products overflow with opposite signs: the midpoint product
    # is inf - inf, so the upper bound is computed as NaN.
    "opposite-signs": ([(-1e200, -1e200), (1e200, 1e200)],
                       [(1e200, 2e200), (1e200, 2e200)]),
    "opposite-signs-mirrored": ([(1e200, 2e200), (1e200, 2e200)],
                                [(1e200, 1e200), (-1e200, -1e200)]),
    # both products overflow upward: the lower bound is computed as NaN
    "same-sign": ([(1e200, 2e200), (1e200, 2e200)],
                  [(1e200, 2e200), (1e200, 2e200)]),
    # both overflow downward
    "same-sign-negative": ([(-2e200, -1e200), (1e200, 2e200)],
                           [(1e200, 2e200), (-2e200, -1e200)]),
    # an infinite operand bound
    "infinite-bound": ([(-np.inf, 1.0), (0.5, 2.0)],
                       [(1.0, 2.0), (-1.0, 1.0)]),
    # every product is far below the least subnormal (about 1e-400): the
    # float products are 0, and only the outward term for underflow keeps
    # the nonzero true values
    "underflow": ([(1e-200, 3e-200), (-2e-200, 1e-200)],
                  [(1e-200, 2e-200), (2e-200, 5e-200)]),
}


def _within(lb, ub, v):
    """Is the exact value ``v`` in [lb, ub] (float ends, possibly infinite)?
    A NaN end holds nothing."""
    if np.isnan(lb) or np.isnan(ub):
        return False
    low = lb == -np.inf or (lb != np.inf and Fraction(lb) <= v)
    high = ub == np.inf or (ub != -np.inf and v <= Fraction(ub))
    return low and high


@pytest.mark.parametrize("case", list(_OVERFLOW))
def test_bilinear_box_kernel_with_overflowing_products_keeps_every_value(case):
    """Operand ranges whose products overflow float64 (or are infinite), or
    underflow below the least subnormal: the true product of every member
    point is a finite real, so the returned Box must hold it; the kernel's
    documented answer is an infinite bound for an overflow and an outward
    term of half a subnormal per product for an underflow. Catches a NaN
    bound (from inf - inf) replaced by the wrong infinity on either side, a
    NaN left in the result, an overflow that drops a term, and a missing
    underflow term (the Box [0, 0] then holds none of the nonzero products).
    Member points: every corner and the midpoint of each interval (an
    infinite end is replaced by the largest float64 number), products in
    exact fractions. The Zono and Star kernels encode an unbounded entry as
    an infinite generator, whose members no oracle can decide (c + inf * a
    is not a real number for any a other than 0), so the Box kernel, which
    runs the same interval product, is the cell."""
    left, right = _OVERFLOW[case]
    llo, lhi = (np.array(v, dtype=np.float64) for v in zip(*left))
    rlo, rhi = (np.array(v, dtype=np.float64) for v in zip(*right))
    out = bilinear_matmul_box([Box(llo, lhi)], [Box(rlo, rhi)], (1, 2), (2, 1))
    if len(out) != 1:
        raise AssertionError(f"{case}: {len(out)} output sets for one pair")
    lb = float(np.asarray(out[0].lb).reshape(-1)[0])
    ub = float(np.asarray(out[0].ub).reshape(-1)[0])
    largest = float(np.finfo(np.float64).max)

    def choices(lo, hi):
        lo = -largest if lo == -np.inf else lo
        hi = largest if hi == np.inf else hi
        return sorted({Fraction(lo), Fraction(hi),
                       (Fraction(lo) + Fraction(hi)) / 2})

    left_points = list(product(*[choices(a, b) for a, b in zip(llo, lhi)]))
    right_points = list(product(*[choices(a, b) for a, b in zip(rlo, rhi)]))
    missed = [(a, b) for a in left_points for b in right_points
              if not _within(lb, ub, a[0] * b[0] + a[1] * b[1])]
    assert not missed, (
        f"{case}: {len(missed)} exact products are outside the returned Box "
        f"[{lb}, {ub}], e.g. left {[float(v) for v in missed[0][0]]}, right "
        f"{[float(v) for v in missed[0][1]]}")


# --------------------------------------------------------------------------- #
# Refusals of the kernels                                                       #
# --------------------------------------------------------------------------- #
_REFUSALS = {
    "unknown mode": (dict(mode="interval"), ValueError),
    "mode mccormick": (dict(mode="mccormick"), NotImplementedError),
    "shape of another size": (dict(left_shape=(2, 2)), ValueError),
    "one-axis shape": (dict(left_shape=(6,), right_shape=(6,)), ValueError),
    "operand lists of different lengths": (dict(lists=(2, 1)), ValueError),
    "unknown bounds mode": (dict(bounds="interval"), ValueError),
}


_REFUSAL_CELLS = [pytest.param(kind, case, id=f"{kind}-{case}")
                  for case in _REFUSALS
                  for kind in (("Star",) if case == "unknown bounds mode"
                               else ("Box", "Zono", "Star"))]


@pytest.mark.parametrize("kind,case", _REFUSAL_CELLS)
def test_bilinear_kernel_refuses_unknown_options_and_wrong_shapes(kind, case):
    """An option the kernel does not know, a tensor shape that does not fit
    the set, a shape with fewer than two axes, or operand lists that cannot
    be paired must raise, never return a set computed for something else
    (``bounds`` exists only on the Star kernel; ``mode='mccormick'`` is
    documented as not implemented and raises ``NotImplementedError``)."""
    options, error = _REFUSALS[case]
    options = dict(options)
    lo, hi = flat_box(6, seed=8)
    s = _box_operand(kind, lo, hi)[0]
    counts = options.pop("lists", (1, 1))
    lshape = options.pop("left_shape", (2, 3))
    rshape = options.pop("right_shape", (3, 2))
    with pytest.raises(error):
        _KERNEL[kind]([s] * counts[0], [s] * counts[1], lshape, rshape,
                      **options)


# --------------------------------------------------------------------------- #
# LP operand ranges on Stars whose cut rows bind the ranges, each solver        #
# --------------------------------------------------------------------------- #
# (basis V = [centre | generators], cut rows H, right-hand sides g) over a
# in [-1, 1]^2. Left x (2, 2) = [[50 a0 + 1/2, 1], [50 a1 - 1/4, -2]] cut by
# a0 + a1 <= -1/2: x00 ranges over [-49.5, 25.5] and x10 over
# [-50.25, 24.75], strictly inside the predicate-box ranges; x01 and x11 are
# constants. Right y (2, 2) = [[b0 + 1, 2 b1 - 2], [3/4, b0 / 2 + 1/4]] cut
# by -b0 - b1 <= -1/2: the least b0 and b1 are -1/2.
_LP_LEFT = (np.array([[0.5, 50.0, 0.0], [1.0, 0.0, 0.0],
                      [-0.25, 0.0, 50.0], [-2.0, 0.0, 0.0]]),
            [[1.0, 1.0]], [-0.5])
_LP_RIGHT = (np.array([[1.0, 1.0, 0.0], [-2.0, 0.0, 2.0],
                       [0.75, 0.0, 0.0], [0.25, 0.5, 0.0]]),
             [[-1.0, -1.0]], [-0.5])


def _cut_operand(V, H, g):
    """The Star ``V [1; a]`` over ``a`` in [-1, 1]^2 cut by ``H a <= g``,
    and the images of the vertices of its predicate polytope, enumerated
    exactly (``rational.vertices``) without n2v."""
    C = np.vstack([np.eye(2), -np.eye(2), np.asarray(H, dtype=np.float64)])
    d = np.concatenate([np.ones(4), g]).reshape(-1, 1)
    star = Star(V, C, d, -np.ones((2, 1)), np.ones((2, 1)))
    G, centre = rational.matrix(V[:, 1:]), rational.vector(V[:, 0])
    points = [[rational.dot(row, a) + c for row, c in zip(G, centre)]
              for a in rational.vertices(rational.matrix(C),
                                         rational.vector(d.ravel()))]
    return star, points


@pytest.mark.parametrize("lp_solver", ["default", "linprog", "SCS"])
def test_bilinear_star_kernel_lp_ranges_hold_the_extreme_members(lp_solver):
    """``bounds='lp'`` on two Stars whose cut rows bind the ranges of their
    entries strictly inside the predicate-box ranges. Every output entry of
    ``x @ y`` is bilinear in the two predicate points, so its least and
    greatest value over the two polytopes are taken at a pair of vertices;
    every vertex pair's exact product (vertices enumerated in fractions from
    the rows) must be a member of the result, decided exactly with zero
    slack. As the result is a box, this proves that it holds the whole
    image.

    Catches: the LP ranges taken over the wrong rows (the cut with the wrong
    sign keeps x00 >= -24.5 and drops the member at -49.5), the caller's
    CVXPY solver used for the range LPs instead of HiGHS (SCS stops short of
    the extremes at this scale), the upper range read from the lower
    optimum, and a constant coordinate detected from the wrong column of the
    basis (an entry that depends only on the first predicate variable is
    then taken for a constant). A range not padded outward is not shown:
    HiGHS returns these dyadic vertices exactly, and a one-ulp shortfall
    would be inside the interval product's own outward rounding term."""
    left, left_points = _cut_operand(*_LP_LEFT)
    right, right_points = _cut_operand(*_LP_RIGHT)
    if max(p[0] for p in left_points) != Fraction(51, 2):
        pytest.fail("the left cut does not bind the range of x00 as built")
    out = bilinear_matmul_star([left], [right], (2, 2), (2, 2),
                               bounds="lp", lp_solver=lp_solver)
    holders = [rational.Membership(rational.star(s)) for s in out]
    missed = []
    for x in left_points:
        for y in right_points:
            z = _exact_product(x, y, (2, 2), (2, 2), 1.0)
            if not any(holder(z) for holder in holders):
                missed.append([float(v) for v in z])
    assert not missed, (
        f"bounds='lp', lp_solver={lp_solver}: {len(missed)} of "
        f"{len(left_points) * len(right_points)} exact products of vertex "
        f"pairs are outside the result, e.g. {missed[0]}")


# --------------------------------------------------------------------------- #
# The graph handler: a product of two sets computed from one input             #
# --------------------------------------------------------------------------- #
_GA = np.array([[0.5, -1.0, 0.25], [-0.75, 0.5, 1.0], [1.0, 0.25, -0.5]],
               dtype=np.float32)
_GC = np.array([[1.0, -0.5], [0.25, 0.75], [-1.0, 0.5]], dtype=np.float32)
_GD = np.array([[0.5, -0.25, 1.0, -1.0], [0.25, 0.5, -0.75, 0.5],
                [-1.0, 0.75, 0.25, 0.5]], dtype=np.float32)
_GR = np.array([[0.75, -1.0], [-0.5, 0.25], [1.0, 0.5]], dtype=np.float32)
_GO2 = np.array([[0.5, -0.25], [-1.0, 0.75]], dtype=np.float32)
_GO4 = np.array([[0.5, -0.25], [-1.0, 0.75], [0.25, 1.0], [-0.5, -0.5]],
                dtype=np.float32)
_GB = np.array([[0.25, -0.5], [1.0, 0.0]], dtype=np.float32)
# constant offsets of the operand that is not split in the ``-tight``
# variants, different in every entry so that no sum of them cancels
_KL = np.array([[4.0, -6.0, 2.0], [-2.0, 5.0, -4.0]], dtype=np.float32)
_KR2 = np.array([[4.0, -4.0], [-6.0, 2.0], [3.0, 5.0]], dtype=np.float32)
_KR4 = np.array([[4.0, -4.0, 2.0, -3.0], [-6.0, 2.0, 5.0, 1.0],
                 [3.0, 5.0, -2.0, 6.0]], dtype=np.float32)


def _setset_model(tmp_path, variant, wide):
    """X (2, 3); left operand L = S @ Ga + Kl (2, 3); right operand
    R = Gr (3, 2) @ (S' @ Gc) + Kr (3, 2), or (3, 4) with Gd when ``wide``;
    Y = (L @ R) @ Go + B (2, 2). S and S' are X or Relu(X): ``left-split``
    reads the ReLU on the left path only, ``right-split`` on the right path
    only, ``both-split`` on both. The two operands have the same number of
    entries in the narrow model and different shapes in the wide one.

    Kl and Kr are zero except in the ``-tight`` variants of the one-sided
    splits, where the operand that is not split has its weights divided by
    16 and a constant offset added: it is then nearly constant, the interval
    product is nearly exact in the split operand, and an output of a piece
    that is left out cannot hide in the product's slack (with the plain
    weights the first ReLU piece, the narrowest, is still covered by the
    slack of a product with a right operand that straddles zero)."""
    nodes, left_src, right_src = [], "X", "X"
    split = variant.replace("-tight", "")
    if variant != "plain":
        nodes.append(helper.make_node("Relu", ["X"], ["H"]))
        left_src = "H" if split in ("left-split", "both-split") else "X"
        right_src = "H" if split in ("right-split", "both-split") else "X"
    nodes += [helper.make_node("MatMul", [left_src, "Ga"], ["L0"]),
              helper.make_node("Add", ["L0", "Kl"], ["L"]),
              helper.make_node("MatMul", [right_src, "Gc"], ["T"]),
              helper.make_node("MatMul", ["Gr", "T"], ["R0"]),
              helper.make_node("Add", ["R0", "Kr"], ["R"]),
              helper.make_node("MatMul", ["L", "R"], ["P"]),
              helper.make_node("MatMul", ["P", "Go"], ["Q"]),
              helper.make_node("Add", ["Q", "B"], ["Y"])]
    ga, gc = _GA, (_GD if wide else _GC)
    kl = np.zeros((2, 3), dtype=np.float32)
    kr = np.zeros((3, 4) if wide else (3, 2), dtype=np.float32)
    if variant == "left-split-tight":
        gc, kr = gc / 16, (_KR4 if wide else _KR2)
    if variant == "right-split-tight":
        ga, kl = ga / 16, _KL
    inits = {"Ga": ga, "Gc": gc, "Gr": _GR, "Go": _GO4 if wide else _GO2,
             "B": _GB, "Kl": kl, "Kr": kr}
    return save_onnx(tmp_path, nodes, (2, 3), (2, 2), inits,
                     name=f"setset_{variant}_{int(wide)}")


def _setset_box():
    """Dyadic bounds of X with two coordinates straddling zero, one up to
    3.5 (so an approx ReLU adds a predicate variable with box [0, 3.5]), two
    positive and two negative."""
    lo, hi = flat_box(6, seed=11)
    return 4 * lo, 4 * hi


# (variant, wide, kind, method): each kernel on operands of the same size
# (narrow) and of different shapes (wide); a ReLU on a path gives a Zono
# operand more generators than entries and a Star operand predicate
# variables with box [0, u]; under exact, the three ways the handler pairs
# pieces: several left pieces with one right set, one left set with several
# right pieces, and the pieces of one split on both operands
_SETSET_CELLS = [
    ("right-split-tight", False, "Box", "approx"),
    ("right-split-tight", True, "Box", "approx"),
    ("left-split-tight", False, "Zono", "approx"),
    ("right-split-tight", True, "Zono", "approx"),
    ("right-split-tight", False, "Star", "approx"),
    ("left-split-tight", True, "Star", "approx"),
    ("left-split-tight", False, "Star-row", "approx"),
    ("left-split-tight", False, "Star", "exact"),
    ("right-split-tight", True, "Star", "exact"),
    ("both-split", False, "Star", "exact"),
    ("left-split-tight", True, "Star-row", "exact"),
]


@pytest.mark.parametrize("variant,wide,kind,method", [
    pytest.param(v, w, k, m, id=f"{v}-{'wide' if w else 'narrow'}-{k}-{m}")
    for v, w, k, m in _SETSET_CELLS])
def test_set_matmul_in_a_graph_holds_every_model_output(tmp_path, variant,
                                                        wide, kind, method):
    """A MatMul whose two operands are sets computed from the same input by
    different paths, then an affine node, through ``reach_pytorch_model``.
    Every onnxruntime output of 128 member points (the two extreme corners,
    random corners, where the operands reach the ends of their ranges, the
    quarter grid, and points on the cut face of the constrained Star) is a
    member of a returned set by LP on its rows.

    Catches: the right operand's sets read under the left node's name (the
    narrow model's operands have the same number of entries), the operands
    swapped, the wrong tensor shape handed to the kernel ((2,3) @ (3,4)),
    the dispatch to the wrong kernel for Box and Zono, operand lists of
    different lengths combined by position instead of paired (``exact``
    with a ReLU split on one path only; the ``-tight`` variants show it
    whichever side is split), a kernel that returns after the
    first pair of pieces (split on both paths), Star ranges that assume a
    [-1, 1] predicate box (``approx`` after the ReLU adds [0, 3.5]
    variables), and a product whose rows the affine node after it does not
    receive."""
    path = _setset_model(tmp_path, variant, wide)
    lo, hi = _setset_box()
    if kind == "Star-row":
        input_set, points = constrained_star(lo, hi, seed=3)
    else:
        input_set, points = flat_set(kind, lo, hi), samples(lo, hi)
    what = (f"set @ set MatMul, {variant}, {'(2,3)@(3,4)' if wide else '(2,3)@(3,2)'}"
            f", {kind}, {method}")
    sets = reach_model(load_onnx(path), input_set, method, what,
                       input_shape=(2, 3))
    assert_covered(sets, replay_onnx(path, points, (2, 3)), what)


def _separate_splits_model(tmp_path):
    """X (1, 2) = (x1, x2), both crossing zero. L = Relu(x1, -x1) + 1 and
    R = Relu(x2, -x2) + 1, each (1, 2); Y = Transpose(L) @ R, the (2, 2)
    outer product. On x1 >= 0 the second entry of L is 1, on x1 <= 0 the
    first is, and R likewise in x2, so no piece's box holds another's."""
    nodes = [helper.make_node("MatMul", ["X", "Ma"], ["A"]),
             helper.make_node("Relu", ["A"], ["Ha"]),
             helper.make_node("Add", ["Ha", "One"], ["L"]),
             helper.make_node("MatMul", ["X", "Mb"], ["B"]),
             helper.make_node("Relu", ["B"], ["Hb"]),
             helper.make_node("Add", ["Hb", "One"], ["R"]),
             helper.make_node("Transpose", ["L"], ["Lt"], perm=[1, 0]),
             helper.make_node("MatMul", ["Lt", "R"], ["Y"])]
    inits = {"Ma": np.array([[1.0, -1.0], [0.0, 0.0]], dtype=np.float32),
             "Mb": np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32),
             "One": np.ones((1, 2), dtype=np.float32)}
    return save_onnx(tmp_path, nodes, (1, 2), (2, 2), inits,
                     name="separate_splits")


@pytest.mark.parametrize("kind", ["Star", "Star-row"])
def test_set_matmul_of_two_separately_split_operands_holds_every_model_output(
        tmp_path, kind):
    """``exact`` on a product whose two operands are split by two different
    ReLUs into the same number of pieces (two each). A left piece and a
    right piece of different splits do not descend from one another, so the
    handler must combine every left piece with every right piece; each
    combination holds the outputs of the input points in both. Every
    onnxruntime output of 128 member points (bounds beyond [-1, 1], intervals
    wider than 2) must be a member of a returned set by LP. Catches operand
    pieces paired by position when the two lists have the same length (65 of
    128 outputs are then outside; the ``both-split`` cell splits one ReLU on
    both paths, where position and lineage agree)."""
    path = _separate_splits_model(tmp_path)
    lo, hi = np.array([-3.0, -1.5]), np.array([2.0, 2.5])
    if kind == "Star-row":
        input_set, points = constrained_star(lo, hi, seed=7)
    else:
        input_set, points = flat_set(kind, lo, hi), samples(lo, hi)
    what = f"outer product of two separately split operands, {kind}, exact"
    sets = reach_model(load_onnx(path), input_set, "exact", what,
                       input_shape=(1, 2))
    assert_covered(sets, replay_onnx(path, points, (1, 2)), what)


_WQ = np.array([[0.5, -1.0, 0.25], [-0.75, 0.5, 1.0], [1.0, 0.25, -0.5]],
               dtype=np.float32)
_WK = np.array([[1.0, 0.5, -0.25], [0.25, -1.0, 0.5], [-0.5, 0.75, 1.0]],
               dtype=np.float32)


@pytest.mark.parametrize("kind", ["Box", "Zono", "Star"])
def test_set_matmul_of_two_heads_holds_every_model_output(tmp_path, kind):
    """The attention scores of two heads: X (1, 2, 2, 3), Q = X @ Wq,
    K = X @ Wk, S = Q @ Transpose(K) of shape (1, 2, 2, 2), through the
    graph. Catches a product that mixes the heads (batched shapes flattened
    or broadcast wrongly) and a wrong shape handed to the kernel."""
    nodes = [helper.make_node("MatMul", ["X", "Wq"], ["Q"]),
             helper.make_node("MatMul", ["X", "Wk"], ["K"]),
             helper.make_node("Transpose", ["K"], ["Kt"], perm=[0, 1, 3, 2]),
             helper.make_node("MatMul", ["Q", "Kt"], ["Y"])]
    path = save_onnx(tmp_path, nodes, (1, 2, 2, 3), (1, 2, 2, 2),
                     {"Wq": _WQ, "Wk": _WK}, name="heads")
    lo, hi = flat_box(12, seed=2, zero_width=True)
    input_set, points = flat_set(kind, lo, hi), samples(lo, hi)
    what = f"Q @ K^T of two heads on {kind}"
    sets = reach_model(load_onnx(path), input_set, "approx", what,
                       input_shape=(2, 2, 3))
    assert_covered(sets, replay_onnx(path, points, (1, 2, 2, 3)), what)


@pytest.mark.parametrize("kind,method", [("Box", "approx"), ("Zono", "approx"),
                                         ("Star", "approx"), ("Star", "exact")],
                         ids=["Box-approx", "Zono-approx", "Star-approx",
                              "Star-exact"])
def test_set_matmul_with_its_input_subtracted_after_it_holds_every_model_output(
        tmp_path, kind, method):
    """Y = MatMul(X, X) - X, X (2, 2) with bounds beyond [-1, 1]: the input
    set is read by the product and again by the Sub, a residual connection
    around the product. Every onnxruntime output of 128 member points must
    be a member of a returned set by LP. Catches a product re-encoded over
    the input's own predicate system (``Star.from_bounds`` in place of
    ``Star.fresh_from_bounds``): the box of four entries then has the same
    rows as the input Star, the Sub reads the two as the same variables and
    31 of 128 outputs are outside; and a kernel that changes its input's
    arrays (``reach_model`` fails then)."""
    path = save_onnx(tmp_path, [helper.make_node("MatMul", ["X", "X"], ["P"]),
                                helper.make_node("Sub", ["P", "X"], ["Y"])],
                     (2, 2), (2, 2), name="residual")
    lo, hi = flat_box(4, seed=3)
    lo, hi = 4 * lo, 4 * hi
    what = f"MatMul(X, X) - X on {kind}, {method}"
    sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), method, what,
                       input_shape=(2, 2))
    assert_covered(sets, replay_onnx(path, samples(lo, hi), (2, 2)), what)


# --------------------------------------------------------------------------- #
# Image-set operands                                                            #
# --------------------------------------------------------------------------- #
_IMAGE_ZONO_OPERAND = known_bug(
    7, "a set @ set MatMul hands an ImageZono operand to the Zono kernel, "
       "which reads its entries in the stored (height, width, channel) order "
       "as the tensor's (channel, height, width) order: every true output of "
       "X @ X on a (2, 3, 3) ImageZono lies outside the result")
_IMAGE_STAR_OPERAND = known_bug(
    7, "a set @ set MatMul whose left operand is a flat Star and right "
       "operand an ImageStar runs the Star kernel, which reads the "
       "ImageStar's entries in the stored (height, width, channel) order as "
       "the tensor's (channel, height, width) order: every true output lies "
       "outside the result")


@pytest.mark.parametrize("model,kind", [
    pytest.param("X@X", "ImageZono", id="X@X-ImageZono",
                 marks=_IMAGE_ZONO_OPERAND),
    pytest.param("X@X", "Zono", id="X@X-Zono"),
    pytest.param("Reshape(X)@X", "ImageStar", id="Reshape(X)@X-ImageStar",
                 marks=_IMAGE_STAR_OPERAND),
    pytest.param("Reshape(X)@Reshape(X)", "ImageStar",
                 id="Reshape(X)@Reshape(X)-ImageStar"),
    pytest.param("Reshape(X)@Reshape(X)", "ImageStar-rows-only",
                 id="Reshape(X)@Reshape(X)-ImageStar-rows-only"),
])
def test_set_matmul_of_an_image_operand_holds_every_model_output(tmp_path,
                                                                  model,
                                                                  kind):
    """X (1, 2, 3, 3), a different dyadic interval per pixel, given as an
    image set (stored height, width, channel) or as the flat Zono of the
    same box (tensor order), with ``input_shape``. ``X@X``: both operands
    are the input. ``Reshape(X)@X``: the left operand is X reshaped to
    (2, 3, 3), a flat set in tensor order, the right is the image set
    itself, so the handler picks the kernel by the flat left operand.
    ``Reshape(X)@Reshape(X)``: both operands flat; on an ImageStar given by
    its rows alone (``ImageStar-rows-only``: a binding cut, no predicate
    box, which the model entry leaves as it is) the reshaped operands are
    flat Stars without a predicate box, whose ranges come from the LP
    enclosure. Every onnxruntime output of 128 member points must be a
    member of a returned set by LP, compared in the order each returned set
    stores (``out_chw``). The promise is a sound answer or a refusal with
    ``NotImplementedError``. Catches an image operand read in its stored
    order as the tensor's order (marked: the ImageZono reaches the Zono
    kernel, the ImageStar after a flat Star the Star kernel), and operand
    ranges of a Star without a predicate box taken wrongly from its LP
    enclosure; the flat Zono and the two flat operands are the passing
    neighbours on the same reference and layout code."""
    if model == "X@X":
        nodes, inits, out_shape = [
            helper.make_node("MatMul", ["X", "X"], ["Y"])], {}, (1, 2, 3, 3)
    else:
        right = "R" if model == "Reshape(X)@Reshape(X)" else "X"
        nodes = [helper.make_node("Reshape", ["X", "s"], ["R"]),
                 helper.make_node("MatMul", ["R", right], ["Y"])]
        inits = {"s": np.array([2, 3, 3], dtype=np.int64)}
        out_shape = (2, 3, 3) if right == "R" else (1, 2, 3, 3)
    path = save_onnx(tmp_path, nodes, (1, 2, 3, 3), out_shape, inits,
                     name="image_operand")
    lo, hi = pixel_box(2, 3, 3, seed=0)
    if kind == "ImageStar-rows-only":
        full, points = constrained_image_star(lo, hi, seed=2)
        input_set = ImageStar(full.V, full.C, full.d, height=3, width=3,
                              num_channels=2)
    else:
        input_set = (image_set(kind, lo, hi) if kind.startswith("Image")
                     else flat_set(kind, lo, hi))
        points = samples(lo, hi)
    what = f"{model} on {kind}"
    try:
        sets = reach_model(load_onnx(path), input_set, "approx", what,
                           input_shape=(2, 3, 3))
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path, points, (1, 2, 3, 3)), what,
                   out_chw=(2, 3, 3))


@pytest.mark.parametrize("kind,method", [
    pytest.param("Hexatope", "approx", id="Hexatope"),
    pytest.param("Octatope", "approx", id="Octatope"),
    pytest.param("ImageStar", "approx", id="ImageStar"),
    pytest.param("ImageStar", "exact", id="ImageStar-exact"),
])
def test_set_matmul_in_a_graph_refuses_a_set_type_it_has_no_kernel_for(
        tmp_path, kind, method):
    """Y = X @ X on X (1, 2, 3, 3), the MatMul the only node, so a refusal
    can come from nowhere else. The product of two Hexatopes, two Octatopes
    or two ImageStars (under either method; the model entry refuses exact on
    a Hexatope or Octatope before the walk) has no kernel and must be
    refused with ``NotImplementedError``, not answered by another set type's
    rule or by the operand passed through."""
    path = save_onnx(tmp_path, [helper.make_node("MatMul", ["X", "X"], ["Y"])],
                     (1, 2, 3, 3), (1, 2, 3, 3), name="xx")
    lo, hi = pixel_box(2, 3, 3, seed=6)
    input_set = (image_set(kind, lo, hi) if kind == "ImageStar"
                 else flat_set(kind, lo, hi))
    with pytest.raises(NotImplementedError):
        reach_pytorch_model(load_onnx(path), input_set, method=method,
                            input_shape=(2, 3, 3))


# --------------------------------------------------------------------------- #
# Routes without a tracked tensor shape, and a torch model                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("input_shape", [
    pytest.param(None, id="no-input-shape", marks=known_bug(
        12, "SHAPE-2: a flat set on a model whose input is (2, 3), with no "
            "input_shape, is analysed as a (1, 6) tensor; Transpose then "
            "gives (6, 1), and X @ Transpose(X) is answered with one entry "
            "where the output has four")),
    pytest.param((2, 3), id="input-shape"),
])
def test_set_matmul_of_a_flat_set_without_a_tensor_shape_holds_every_output(
        tmp_path, input_shape):
    """Y = X @ Transpose(X), X (2, 3), on a flat Star with and without
    ``input_shape``. The handler reads the tensor shapes the shape pass
    recorded; the promise is a refusal with ``NotImplementedError`` or a
    result that holds every onnxruntime output of 128 member points (LP
    membership; a result with another number of entries holds none).
    Catches a product taken over shapes the model does not have."""
    path = save_onnx(tmp_path,
                     [helper.make_node("Transpose", ["X"], ["T"], perm=[1, 0]),
                      helper.make_node("MatMul", ["X", "T"], ["Y"])],
                     (2, 3), (2, 2), name="xxt")
    lo, hi = flat_box(6, seed=3)
    options = {} if input_shape is None else {"input_shape": input_shape}
    what = f"X @ X^T on a flat Star, input_shape={input_shape}"
    try:
        sets = reach_model(load_onnx(path), flat_set("Star", lo, hi),
                           "approx", what, **options)
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path, samples(lo, hi), (2, 3)), what)


class _SelfProduct(nn.Module):
    def forward(self, x):
        return torch.matmul(x, x)


def test_set_matmul_in_a_torch_model_is_refused_or_holds_every_output():
    """``torch.matmul(x, x)`` of a (2, 2) input in a torch model: the walk
    has no reach rule for a ``call_function`` matmul. The promise is a
    refusal with ``NotImplementedError`` or a result that holds every torch
    float64 output of 128 member points (LP membership). Catches a rule
    registered for ``torch.matmul`` that answers another function (the
    operand passed through)."""
    lo, hi = flat_box(4, seed=5)
    what = "torch.matmul(x, x) on a flat Star"
    try:
        sets = reach_model(_SelfProduct(), flat_set("Star", lo, hi), "approx",
                           what, input_shape=(2, 2))
    except NotImplementedError:
        return
    assert_covered(sets, replay_torch(_SelfProduct(), samples(lo, hi), (2, 2)),
                   what)


# --------------------------------------------------------------------------- #
# The A @ V envelope on a value Star with no predicate box                      #
# --------------------------------------------------------------------------- #
def test_av_envelope_with_a_value_star_without_a_predicate_box():
    """``av_envelope_star`` on a value Star whose variables are bounded by
    its rows only (``predicate_lb`` and ``predicate_ub`` are None), cut by a
    binding row. The old variables get the bounds (-inf, inf) in the output
    Star; a wrong fill (0, or the upper bound for the lower one) would cut
    the variables' negative half and leave true outputs out.
    Reference: ``A @ V`` in fractions at member points of V (corners kept
    by the cut and points on the cut face) and at the corners of the A box
    where an output is least or greatest, LP membership against the rows
    (the envelope's slopes are divisions, so the reference is float64 and
    the slack is the LP tolerance)."""
    H, M, K, D = 1, 2, 3, 2
    lo, hi = flat_box(H * K * D, seed=13, zero_width=True)
    full, v_points = constrained_star(lo, hi, n=24, seed=5)
    v_star = Star(full.V, full.C, full.d)
    rng = np.random.default_rng(14)
    a_lb = rng.integers(0, 4, size=(H, M, K)) / 8.0
    a_ub = a_lb + rng.integers(1, 5, size=(H, M, K)) / 8.0
    out = av_envelope_star(a_lb, a_ub, v_star, H, M, K, D)
    outputs = []
    for v in v_points:
        V = np.array(_fractions(v), dtype=object).reshape(H, K, D)
        signs = np.asarray(v, dtype=np.float64).reshape(H, K, D) >= 0
        for m in range(M):
            for d in range(D):
                # entry (m, d) is least with a_lb where V >= 0 and a_ub
                # where V < 0 in its row, greatest the other way round
                for low in (True, False):
                    A = np.where(rng.integers(0, 2, size=(H, M, K)) > 0,
                                 a_lb, a_ub)
                    A[:, m, :] = np.where(signs[:, :, d] == low,
                                          a_lb[:, m, :], a_ub[:, m, :])
                    A = np.array(_fractions(A), dtype=object).reshape(H, M, K)
                    outputs.append([float(x) for x in
                                    np.matmul(A, V).reshape(-1)])
    assert_covered([out], outputs, "A @ V envelope, V without a predicate box")
