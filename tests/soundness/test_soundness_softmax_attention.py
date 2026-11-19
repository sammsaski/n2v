"""Soundness of the attention Softmax: the correlated row bound.

What the file covers. ``n2v/nn/layer_ops/softmax_attention_reach.py`` bounds
a softmax over one axis of a tensor (the attention weights) per entry: entry
j of a row is greatest with its own logit at the top of its range and the
rest of its row at the bottom, least the other way round, and those corners
are jointly attainable on a box of logits. ``softmax_attn_box``,
``softmax_attn_zono`` and ``softmax_attn_star`` take the logit ranges of
their input set (``box.lb``/``box.ub``, ``Zono.get_bounds``,
``Star.estimate_ranges`` or, with ``bounds='lp'``, padded LP ranges), apply
``correlated_row_softmax_bounds`` and return one Box, Zono or Star (on new
predicate variables, ``Star.fresh_from_bounds``) per input set.

No public route reaches these kernels except the graph walker's Softmax
handler (``_handle_softmax_flat`` in ``n2v/nn/reach.py``), for a flat Star,
Zono or Box whose tensor shape is tracked: ONNX Softmax at opset 13 (one
axis) and at opset 11 (the flattened tail from the axis), and ``nn.Softmax``
in a traced torch model. ``reach_layer`` with ``nn.Softmax`` goes to
``softmax_reach.py`` (``test_soundness_softmax.py``). Cells here:

- the kernels called directly: tensors with two axes after the softmax axis,
  negative axes, rows of one key, several input sets, Stars whose predicate
  box is not [-1, 1], Stars cut by a binding row, a Star without a predicate
  box (bounded by its rows alone), a Zono and a Star with more
  generators or predicate variables than entries, rows of logits far apart,
  ``bounds='lp'`` on a Star whose cut rows bind the logit ranges (checked in
  exact arithmetic at every vertex), wrong arguments, lists that mix set
  families;
- the model entry: ONNX Softmax on a flat set at opsets 11 and 13, between
  two affine nodes, after an exact ReLU split, inside a residual connection,
  ``nn.Softmax`` in a torch model through ``reach_pytorch_model`` and
  ``NeuralNetwork.reach``, an input set with an infinite bound, and the rows
  that refuse;
- the rounding of the bounds (registry ROUND-6), one marked cell per kernel.

Not here: Softmax on an image set of more than one pixel (``ACT-1``,
``tests/contracts/test_reach_coverage.py``), ``method='exact'`` on a flat
Star (``REACH-4``, same file), and, in ``test_soundness_softmax.py``, the
graph handler's own cells: its grid of shapes, axes and opsets,
``nn.Softmax()`` without ``dim``, the routes on which no tensor shape is
tracked (``SHAPE-2``, ``SHAPE-8``), the refusal of Hexatope and Octatope under
approx, and a flattened ImageStar without a predicate box.

Oracle. True outputs never come from n2v: a float64 softmax computed here or
onnxruntime / torch in float64 on float64 graphs, on dyadic logits.
Membership is decided by LP on the returned set's own rows
(``assert_covered``, slack ``LP_TOL``) and, in the rounding and LP-range
cells, exactly (``rational.meets_box``) against the softmax enclosed at 100
bits by ``tests/_oracles/interval_proof.py``. The member points include, for
every entry, the logits where it is greatest and least, where a returned box
is tight.

Layout. A flat set holds the tensor in its row-major order; image sets store
(height, width, channel) and are compared through ``out_chw``.
"""

from fractions import Fraction

import numpy as np
import pytest
import torch.nn as nn
from onnx import helper

from n2v.nn.layer_ops.softmax_attention_reach import (
    correlated_row_softmax_bounds,
    softmax_attn_box,
    softmax_attn_star,
    softmax_attn_zono,
)
from n2v.nn.neural_network import NeuralNetwork
from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.utils.model_loader import load_onnx
from tests._oracles import interval_proof as proof
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_image_star,
                                constrained_star, flat_box, flat_set,
                                image_set, pixel_box, reach_model,
                                replay_onnx, replay_torch, samples, save_onnx)
from tests.known_bugs import known_bug

_ATTN_KERNEL = {"Box": softmax_attn_box, "Zono": softmax_attn_zono,
                "Star": softmax_attn_star}


def _softmax_np(x, axis):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# =========================================================================== #
# The kernels called directly                                                 #
# =========================================================================== #
# (tensor shape, softmax axis)
_ATTN_SHAPES = {
    "(2,3,4)/0": ((2, 3, 4), 0),          # two axes after it, lengths 3 and 4
    "(2,3,4)/-3": ((2, 3, 4), -3),        # the same axis, negative
    "(2,3,4)/1": ((2, 3, 4), 1),
    "(2,3,4)/-1": ((2, 3, 4), -1),
    "(2,2,3,2)/1": ((2, 2, 3, 2), 1),     # four axes, two after it
    "(3,1)/-1": ((3, 1), -1),             # one key per row: softmax is 1
}


def _logit_bounds(n, seed, scale=2.0):
    """Dyadic logit bounds of ``n`` entries: straddling, positive and
    negative ranges and one of zero width, times ``scale``."""
    lo, hi = flat_box(n, seed=seed, zero_width=True)
    return scale * lo, scale * hi


def _rows(shape, axis):
    """The flat indices of the entries normalised together, one row each."""
    n = int(np.prod(shape))
    return np.moveaxis(np.arange(n).reshape(shape), axis, -1).reshape(
        -1, shape[axis])


def _attention_points(lo, hi, shape, axis, rng):
    """Logit tensors of the box: for every entry, the logits where it is
    greatest (its own at ``hi``, the rest of its row at ``lo``) and least
    (the other way round), the other rows at a random corner; the two
    extreme corners; random corners and quarter-grid points."""
    lo_m = np.moveaxis(lo.reshape(shape), axis, -1)
    hi_m = np.moveaxis(hi.reshape(shape), axis, -1)
    points = []
    for index in np.ndindex(*lo_m.shape):
        row = index[:-1]
        for greatest in (True, False):
            x = np.where(rng.integers(0, 2, size=lo_m.shape) > 0, lo_m, hi_m)
            x[row] = lo_m[row] if greatest else hi_m[row]
            x[index] = hi_m[index] if greatest else lo_m[index]
            points.append(np.moveaxis(x, -1, axis).reshape(-1))
    return points + samples(lo, hi, n=24, seed=int(rng.integers(1000)))


def _wide_predicate_logits(n, seed):
    """A Star ``x = c + diag(g) a`` whose predicate box is not [-1, 1] (one
    sided [0, 3] and [0, 1.5], [-2.5, 0.5], [1, 2.5], [-4, -1]); it is
    exactly the box returned as ``(star, lo, hi)``."""
    rng = np.random.default_rng(seed)
    c = rng.integers(-8, 9, size=n) / 8.0
    g = rng.choice([-1.5, -0.75, 0.5, 1.25, 2.0], size=n)
    boxes = [(0.0, 3.0), (-2.5, 0.5), (1.0, 2.5), (-4.0, -1.0), (0.0, 1.5)]
    order = rng.permutation(n)
    pl = np.array([boxes[i % len(boxes)][0] for i in order])
    pu = np.array([boxes[i % len(boxes)][1] for i in order])
    star = Star(np.hstack([c[:, None], np.diag(g)]),
                np.vstack([np.eye(n), -np.eye(n)]),
                np.concatenate([pu, -pl]).reshape(-1, 1),
                pl.reshape(-1, 1), pu.reshape(-1, 1))
    ends = np.stack([c + g * pl, c + g * pu])
    return star, ends.min(axis=0), ends.max(axis=0)


def _cut_logit_star(lo, hi, shape, axis, rng):
    """``(star, points)``: the box [lo, hi] as ``x = c + h a`` (one predicate
    variable in [-1, 1] per entry of positive width) cut by
    ``a_i + a_j <= -1/2``, where i and j are the first entries of positive
    width of the first two rows. The cut lowers the top of x_i and of x_j to
    ``c + h/2``: their LP ranges are strictly inside the predicate-box
    ranges, so LP ranges taken from the wrong rows show. The two lie in
    different rows, so every row's extremes
    stay jointly attainable. ``points``: :func:`_attention_points` of the
    box, each moved into the cut (the partner of an entry pushed to its top
    goes to its bottom), dyadic; among them the greatest value of x_i and of
    x_j with the rest of their rows at the bottom."""
    n = lo.size
    live = hi > lo
    rows = _rows(shape, axis)
    i = next(int(k) for k in rows[0] if live[k])
    j = next(int(k) for k in rows[1] if live[k])
    centre, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    var = {int(p): v for v, p in enumerate(np.flatnonzero(live))}
    k = len(var)
    G = np.zeros((n, k))
    for p, v in var.items():
        G[p, v] = half[p]
    cut = np.zeros(k)
    cut[var[i]] = cut[var[j]] = 1.0
    star = Star(np.hstack([centre[:, None], G]),
                np.vstack([np.eye(k), -np.eye(k), cut[None, :]]),
                np.concatenate([np.ones(2 * k), [-0.5]]).reshape(-1, 1),
                -np.ones((k, 1)), np.ones((k, 1)))
    points = []
    for x in _attention_points(lo, hi, shape, axis, rng):
        a = np.where(live, (x - centre) / np.where(live, half, 1.0), 0.0)
        if a[i] + a[j] > -0.5:
            if a[i] >= a[j]:
                a[i], a[j] = min(a[i], 0.5), -1.0
            else:
                a[i], a[j] = -1.0, min(a[j], 0.5)
        points.append(centre + half * a)
    return star, points


def _extra_variable_set(kind, lo, hi, shape, axis, rng):
    """``(set, points)``: the box [lo, hi] as a Zono or a Star with twice as
    many generators or predicate variables as entries (the
    shape a relaxation leaves): ``x_p = c_p + h_p/2 a_p + h_p/4 (b_p +
    b_{p+1})`` with ``h`` the half-widths, so neighbouring entries share a
    variable and the generator matrix is not diagonal; the Star also has the
    row ``a_0 - b_1 <= 1/2``. ``points``: for every entry, the variables of
    its row at -1 (or +1) and its own at +1 (or -1), the rest at a random
    corner, then random corners; all members, dyadic."""
    n = lo.size
    centre, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    G = np.zeros((n, 2 * n))
    G[np.arange(n), np.arange(n)] = half / 2.0
    G[np.arange(n), n + np.arange(n)] += half / 4.0
    G[np.arange(n), n + (np.arange(n) + 1) % n] += half / 4.0
    own = {p: [p, n + p, n + (p + 1) % n] for p in range(n)}
    zs = []
    for row in _rows(shape, axis):
        for p in row:
            for greatest in (True, False):
                z = rng.choice([-1.0, 1.0], size=2 * n)
                for q in row:
                    z[own[int(q)]] = -1.0 if greatest else 1.0
                z[own[int(p)]] = 1.0 if greatest else -1.0
                zs.append(z)
    zs += [rng.choice([-1.0, 1.0], size=2 * n) for _ in range(32)]
    if kind == "Zono":
        return Zono(centre.reshape(-1, 1), G), [centre + G @ z for z in zs]
    cut = np.zeros(2 * n)
    cut[0], cut[n + 1] = 1.0, -1.0
    star = Star(np.hstack([centre[:, None], G]),
                np.vstack([np.eye(2 * n), -np.eye(2 * n), cut[None, :]]),
                np.concatenate([np.ones(4 * n), [0.5]]).reshape(-1, 1),
                -np.ones((2 * n, 1)), np.ones((2 * n, 1)))
    points = []
    for z in zs:
        z = z.copy()
        z[0] = min(z[0], z[n + 1] + 0.5)        # keep the cut row
        points.append(centre + G @ z)
    return star, points


def _far_rows_bounds(n, shape, axis):
    """Logit bounds whose odd rows are 1024 above the even rows: a softmax
    that subtracts the maximum of the whole tensor instead of each row's
    underflows to 0 / 0 on the even rows."""
    lo, hi = _logit_bounds(n, seed=4)
    offset = np.zeros(n)
    rows = _rows(shape, axis)
    for r in range(1, rows.shape[0], 2):
        offset[rows[r]] = 1024.0
    return lo + offset, hi + offset


def _attention_inputs(kind, family, shape, axis):
    """``(input sets, options, member points per set)`` of a family."""
    n = int(np.prod(shape))
    rng = np.random.default_rng(17)
    if family == "pieces":
        # three sets with different logit ranges, the narrowest first
        sets, points = [], []
        for p in range(3):
            lo, hi = _logit_bounds(n, seed=40 + p, scale=1.0 + p)
            lo, hi = lo + p, hi + p
            sets.append(flat_set(kind, lo, hi))
            points.append(_attention_points(lo, hi, shape, axis, rng))
        return sets, {}, points
    if family == "wide-predicate":
        star, lo, hi = _wide_predicate_logits(n, seed=3)
        return [star], {}, [_attention_points(lo, hi, shape, axis, rng)]
    if family in ("binding-row", "no-predicate-box"):
        lo, hi = _logit_bounds(n, seed=5)
        star, pts = constrained_star(lo, hi, n=48, seed=6)
        if family == "no-predicate-box":
            # the same set, bounded by its rows alone (they hold +-I): its
            # logit ranges come from the LP enclosure of estimate_ranges
            star = Star(star.V, star.C, star.d, None, None)
        return [star], {}, [[p.reshape(-1) for p in pts]]
    if family == "lp-default":
        lo, hi = _logit_bounds(n, seed=5)
        star, pts = _cut_logit_star(lo, hi, shape, axis, rng)
        return [star], {"bounds": "lp", "lp_solver": "default"}, [pts]
    if family == "extra-variables":
        lo, hi = _logit_bounds(n, seed=7, scale=4.0)
        s, pts = _extra_variable_set(kind, lo, hi, shape, axis, rng)
        return [s], {}, [pts]
    if family == "far-rows":
        lo, hi = _far_rows_bounds(n, shape, axis)
        return [flat_set(kind, lo, hi)], {}, [
            _attention_points(lo, hi, shape, axis, rng)]
    lo, hi = _logit_bounds(n, seed=4)
    return [flat_set(kind, lo, hi)], {}, [
        _attention_points(lo, hi, shape, axis, rng)]


def _attention_cells():
    cells = []
    for kind in ("Box", "Zono", "Star"):
        for shape in _ATTN_SHAPES:
            cells.append(pytest.param(kind, "box", shape,
                                      id=f"{kind}-box-{shape}"))
        cells.append(pytest.param(kind, "pieces", "(2,3,4)/-3",
                                  id=f"{kind}-pieces-(2,3,4)/-3"))
        cells.append(pytest.param(kind, "far-rows", "(2,3,4)/-1",
                                  id=f"{kind}-far-rows-(2,3,4)/-1"))
    for family, shape in (("wide-predicate", "(2,3,4)/-1"),
                          ("binding-row", "(2,2,3,2)/1"),
                          ("no-predicate-box", "(2,3,4)/-1"),
                          ("lp-default", "(2,3,4)/0"),
                          ("extra-variables", "(2,3,4)/0")):
        cells.append(pytest.param("Star", family, shape,
                                  id=f"Star-{family}-{shape}"))
    cells.append(pytest.param("Zono", "extra-variables", "(2,3,4)/0",
                              id="Zono-extra-variables-(2,3,4)/0"))
    return cells


@pytest.mark.parametrize("kind,family,shape", _attention_cells())
def test_attention_softmax_kernel_holds_every_softmax(kind, family, shape):
    """Every softmax (float64, computed here) of member logits of every input
    set, including the logits where each entry is greatest and least, is a
    member of a returned set by LP on its rows.

    Catches: the axis put back by a swap instead of a move, or the output
    flattened in another order (two axes after the softmax axis), a negative
    axis resolved to the wrong axis, the axis not forwarded, a normalising
    sum over the whole tensor instead of each row, a bound evaluated with the
    own or the competing logits at the wrong end, a working row that writes
    into the caller's bounds, an input range read from the wrong end, a
    kernel that returns after the first input set (``pieces``), Star ranges
    that assume a [-1, 1] predicate box (``wide-predicate``), a predicate
    box at all (``no-predicate-box``) or one variable per entry
    (``extra-variables``, also for a Zono), LP ranges taken from
    the wrong rows (``lp-default``, whose cut binds two ranges) and a
    softmax shifted by the maximum of the whole tensor, which turns the rows
    far below it into 0 / 0 (``far-rows``)."""
    tensor_shape, axis = _ATTN_SHAPES[shape]
    sets, options, points = _attention_inputs(kind, family, tensor_shape,
                                              axis)
    out = _ATTN_KERNEL[kind](sets, tensor_shape, axis, **options)
    what = f"attention softmax, {kind}, {family}, shape {shape}"
    if not out:
        raise AssertionError(f"{what}: no output set")
    outputs = [_softmax_np(np.asarray(x, dtype=np.float64).reshape(
        tensor_shape), axis).reshape(-1) for pts in points for x in pts]
    assert_covered(out, outputs, what)


# --------------------------------------------------------------------------- #
# bounds='lp' on a Star whose cut rows bind the logit ranges, exact           #
# --------------------------------------------------------------------------- #
# Logits (2, 2), softmax over the last axis: rows (x0, x1) and (x2, x3),
# x = c + diag(g) a over a in [-1, 1]^4 cut by two rows that bind ranges:
# a0 + a2 <= -1/2 lowers the top of x0 and of x2 to 25 (the two are in
# different rows, so each row's extremes stay jointly attainable), and
# a3 - a1 <= -1/4 raises the bottom of x1 to -49.5 and lowers the top of x3
# to 25.5. x0 in [-50, 25] and x2 in [-50, 25] have scale 50, so a range LP
# that stops short (SCS, by 1.1e-4 on their bottom ends) misses by far more
# than the pad. x1 in [-49.5, -46] sits near the bottom of x0 and x3 in
# [22, 25.5] near the top of x2, so the softmax is not saturated where the
# least value of x0 and the greatest value of x2, the end its cut binds,
# decide a bound (0.018 and 0.95).
_LP_CENTRE = [0.0, -48.0, 0.0, 24.0]
_LP_SCALE = [50.0, 2.0, 50.0, 2.0]
_LP_C = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
         [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1],
         [1, 0, 1, 0], [0, -1, 0, 1]]
_LP_D = [1, 1, 1, 1, 1, 1, 1, 1, Fraction(-1, 2), Fraction(-1, 4)]


def _softmax_over_rows(shape, axis):
    """The softmax over ``axis`` of a tensor of ``shape`` as an enclosure of
    ``interval_proof``: entry j of a row is 1 / sum_k exp(x_k - x_j), in
    the proof's interval arithmetic (100 bits, rounded outward)."""
    order = np.moveaxis(np.arange(int(np.prod(shape))).reshape(shape),
                        axis, -1)

    def softmax(xs):
        out = [None] * len(xs)
        for index in np.ndindex(*order.shape):
            row = [xs[int(k)] for k in order[index[:-1]]]
            mine = xs[int(order[index])]
            out[int(order[index])] = 1 / sum(proof.IV.exp(xk - mine)
                                             for xk in row)
        return out
    return proof.chain(proof.function(softmax))


@pytest.mark.parametrize("lp_solver", ["default", "linprog", "SCS"])
def test_attention_softmax_star_lp_ranges_hold_the_extreme_members(lp_solver):
    """``bounds='lp'`` on the cut logit Star above: at every vertex of the
    predicate polytope (exact, from its rows), the softmax enclosed at 100
    bits must meet the returned set, decided exactly on its rows. Among the
    vertices are, for every entry, the logits where it is greatest and least
    (for x2: a2 = 1/2 with a0 = -1, the top the cut leaves, and x3 at its
    bottom), where the correlated row bound is attained.
    Catches LP ranges taken from the wrong rows or from the predicate box
    alone, the caller's solver used for
    the range LPs (SCS stops short by up to 1e-4 here) and a range of the
    centre only, with no slack at all. An LP range left
    unpadded is not shown unsound here: the HiGHS ranges ``get_ranges``
    returns already lie 1e-14 to 1e-12 outside the exact ones."""
    C = np.array(_LP_C, dtype=np.float64)
    d = np.array([float(v) for v in _LP_D]).reshape(-1, 1)
    V = np.hstack([np.array(_LP_CENTRE)[:, None], np.diag(_LP_SCALE)])
    star = Star(V, C, d, -np.ones((4, 1)), np.ones((4, 1)))
    out = softmax_attn_star([star], (2, 2), -1, lp_solver=lp_solver,
                            bounds="lp")
    if len(out) != 1:
        pytest.fail(f"{len(out)} output sets for one input set")
    computed = rational.star(out[0])
    enclose = _softmax_over_rows((2, 2), -1)
    corners = rational.vertices(rational.matrix(_LP_C),
                                rational.vector(_LP_D))
    if len(corners) < 8:
        pytest.fail(f"the cut polytope has only {len(corners)} vertices")
    missed = []
    for a in corners:
        x = [Fraction(c) + Fraction(g) * ai
             for c, g, ai in zip(_LP_CENTRE, _LP_SCALE, a)]
        low, high = proof.value(enclose, x)
        if not rational.meets_box(computed, low, high):
            missed.append([float(v) for v in a])
    assert not missed, (
        f"bounds='lp', lp_solver={lp_solver}: the exact softmax at "
        f"{len(missed)} of {len(corners)} vertices of the predicate polytope "
        f"is outside the returned set, e.g. at a = {missed[0]}")


# --------------------------------------------------------------------------- #
# Rounding of the bounds (registry ROUND-6), exact                            #
# --------------------------------------------------------------------------- #
def _exact_softmax(x, shape, axis):
    """The softmax of the logits ``x`` (floats, read as the rationals they
    are), each entry enclosed: ``(low, high)``, lists of fractions in the
    tensor's order (``interval_proof.value``)."""
    return proof.value(_softmax_over_rows(shape, axis),
                       [Fraction(float(v)) for v in
                        np.asarray(x, dtype=np.float64).reshape(-1)])


def _rounding_box(family, shape, axis):
    n = int(np.prod(shape))
    if family == "random":
        # widths from 1e-6 to 1: a bound computed by round-to-nearest lands
        # inside the exact value about half the time
        rng = np.random.default_rng(20261113)
        centre = rng.uniform(-4.0, 4.0, n)
        radius = 10.0 ** rng.uniform(-6.0, 0.0, n)
        return centre - radius, centre + radius
    # equal logits in every row, one point: every value is 1/4, a float64
    # number, so a correct kernel returns it exactly
    return np.full(n, 0.75), np.full(n, 0.75)


_ROUNDING_DEFECT = known_bug(
    7, "the attention softmax bounds (correlated_row_softmax_bounds, used by "
       "softmax_attn_box, softmax_attn_zono and softmax_attn_star) are "
       "computed by subtraction of the row maximum, exp, a sum and a "
       "division in round-to-nearest with no outward step, so a bound can "
       "lie inside the exact least or greatest value of its entry by several "
       "units in the last place (measured up to 10.6 ulps, 2.75e-16 "
       "absolute; a one-ulp step does not fix it): the returned set leaves "
       "out the true softmax at the corner where the bound is attained "
       "(registry ROUND-6)")


def _rounding_cells():
    cells = []
    for kind in ("Box", "Zono", "Star"):
        for family in ("equal", "random"):
            marks = () if family == "equal" else (_ROUNDING_DEFECT,)
            cells.append(pytest.param(kind, "(2,4)/-1", family, marks=marks,
                                      id=f"{kind}-(2,4)/-1-{family}"))
    return cells


_ROUNDING_SHAPES = {"(2,4)/-1": ((2, 4), -1)}


@pytest.mark.parametrize("kind,shape,family", _rounding_cells())
def test_attention_softmax_bounds_hold_the_exact_extremes(kind, shape, family):
    """At the logits where each entry is greatest and least (corners of the
    logit box), the exact softmax, enclosed at 100 bits, must meet the
    returned set, decided exactly on its rows (``rational.meets_box``): a
    set that misses the enclosure leaves out a true output, by however
    little. Catches a bound that is not moved outward after round-to-nearest
    exp and division (registry ROUND-6; one marked cell per kernel). The
    ``equal`` family (every value 1/4, exact in float64) is the passing
    neighbour with the same reference; it also catches a sum over the whole
    tensor (1/8) and a softmax over the other axis (1/2)."""
    tensor_shape, axis = _ROUNDING_SHAPES[shape]
    n = int(np.prod(tensor_shape))
    lo, hi = _rounding_box(family, tensor_shape, axis)
    if kind == "Box":
        input_set = Box(lo, hi)
    else:
        input_set = {"Zono": Zono, "Star": Star}[kind].from_bounds(
            lo.reshape(-1, 1), hi.reshape(-1, 1))
    out = _ATTN_KERNEL[kind]([input_set], tensor_shape, axis)
    if len(out) != 1:
        pytest.fail(f"{kind}: {len(out)} output sets for one input set")
    computed = rational.star(out[0])
    rng = np.random.default_rng(5)
    lo_m = np.moveaxis(lo.reshape(tensor_shape), axis, -1)
    hi_m = np.moveaxis(hi.reshape(tensor_shape), axis, -1)
    missed = []
    for index in np.ndindex(*lo_m.shape):
        for greatest in (True, False):
            x = np.where(rng.integers(0, 2, size=lo_m.shape) > 0, lo_m, hi_m)
            x[index[:-1]] = lo_m[index[:-1]] if greatest else hi_m[index[:-1]]
            x[index] = hi_m[index] if greatest else lo_m[index]
            x = np.moveaxis(x, -1, axis)
            low, high = _exact_softmax(x, tensor_shape, axis)
            if not rational.meets_box(computed, low, high):
                missed.append((index, greatest))
    assert not missed, (
        f"{kind}, {shape}, {family}: the exact softmax is outside the "
        f"returned set at {len(missed)} of {2 * n} extreme corners, e.g. the "
        f"{'greatest' if missed[0][1] else 'least'} value of entry "
        f"{missed[0][0]} (row-moved index)")


# --------------------------------------------------------------------------- #
# Wrong arguments and mixed lists, called directly                            #
# --------------------------------------------------------------------------- #
_SM_REFUSALS = {
    # a logit box with an upper end below its lower end is not a set
    "upper below lower": "bounds",
    # an axis the tensor does not have, positive and negative
    "axis 2 of a 2-D tensor": "axis",
    "axis -3 of a 2-D tensor": "axis-negative",
    # a tensor shape with another number of entries than the set
    "shape of another size": "shape",
    # a bounds mode the Star kernel does not know
    "unknown bounds mode": "mode",
}


_SM_REFUSAL_CELLS = [
    pytest.param(kind, case, id=f"{kind}-{case}")
    for case, what in _SM_REFUSALS.items()
    for kind in (("Box",) if what == "bounds" else
                 ("Star",) if what == "mode" else ("Box", "Zono", "Star"))]


@pytest.mark.parametrize("kind,case", _SM_REFUSAL_CELLS)
def test_attention_softmax_kernel_refuses_wrong_arguments(kind, case):
    """A logit box that is not a box, an axis the tensor does not have, a
    shape that does not fit the set, or an unknown bounds mode must raise
    ``ValueError``, never return a softmax over some other axis or rows
    (the bounds mode exists only on the Star kernel; the inverted box is
    checked on the shared row bound, which every kernel calls)."""
    what = _SM_REFUSALS[case]
    lo, hi = _logit_bounds(6, seed=12)
    if what == "bounds":
        with pytest.raises(ValueError):
            correlated_row_softmax_bounds(hi.reshape(2, 3) + 1.0,
                                          lo.reshape(2, 3), axis=-1)
        return
    s = flat_set(kind, lo, hi)
    args = {"axis": ((2, 3), 2), "axis-negative": ((2, 3), -3),
            "shape": ((2, 2), -1), "mode": ((2, 3), -1)}[what]
    options = {"bounds": "interval"} if what == "mode" else {}
    with pytest.raises(ValueError):
        _ATTN_KERNEL[kind]([s], args[0], args[1], **options)


_MIXED_LIST_READ_IN_STORED_ORDER = known_bug(
    7, "softmax_attn_star and softmax_attn_zono check no set of their "
       "list: an ImageStar or ImageZono after a flat Star or Zono is read as "
       "a flat set, its (height, width, channel) entries taken for the "
       "tensor's row-major order, so its softmax is bounded over the wrong "
       "groups and none of its outputs is in the result (only a direct call "
       "builds such a list; registry ACT-4, the kind of REACH-6)")


@pytest.mark.parametrize("first,second,channels", [
    pytest.param("Star", "ImageStar", 2,
                 marks=_MIXED_LIST_READ_IN_STORED_ORDER),
    ("Star", "ImageStar", 1),
    pytest.param("Zono", "ImageZono", 2,
                 marks=_MIXED_LIST_READ_IN_STORED_ORDER),
    ("Zono", "ImageZono", 1),
], ids=["Star-then-ImageStar", "Star-then-one-channel-ImageStar",
        "Zono-then-ImageZono", "Zono-then-one-channel-ImageZono"])
def test_attention_softmax_kernel_on_a_list_mixing_set_families_refuses_or_holds_every_output(  # noqa: E501
        first, second, channels):
    """A flat set followed by an image set of a (C, 2, 3) tensor, softmax
    over the width, in one direct call: the kernel must
    refuse (``NotImplementedError`` or ``ValueError``) or hold every output
    of both sets, the image set's in the layout of the set returned for it.
    Today the answers are sound for the flat set and, with one channel
    (stored order equal to tensor order, the passing neighbour), for the
    image set; with two channels the image set is read in its stored order.
    The other orders of the probe (a Box, Hexatope or Octatope after a Zono,
    any other family after a Box, an ImageStar after a Zono) crash with
    ``AttributeError`` or ``TypeError`` (registry ACT-4, no case here)."""
    shape = (channels, 2, 3)
    n = int(np.prod(shape))
    lo_f, hi_f = _logit_bounds(n, seed=21)
    lo3, hi3 = pixel_box(*shape, seed=2)
    lo3, hi3 = 2.0 * lo3, 2.0 * hi3
    sets = [flat_set(first, lo_f, hi_f), image_set(second, lo3, hi3)]
    rng = np.random.default_rng(4)
    try:
        out = _ATTN_KERNEL[first](sets, shape, -1)
    except (NotImplementedError, ValueError):
        return
    if len(out) != 2:
        pytest.fail(f"{len(out)} output sets for two input sets")
    what = f"{first} then {second} of {shape}"
    assert_covered([out[0]], [
        _softmax_np(x.reshape(shape), -1).reshape(-1)
        for x in _attention_points(lo_f, hi_f, shape, -1, rng)], what)
    assert_covered([out[1]], [
        _softmax_np(x.reshape(shape), -1).reshape(-1)
        for x in _attention_points(lo3.reshape(-1), hi3.reshape(-1), shape,
                                   -1, rng)],
        f"{what}, the image set", out_chw=shape)


# =========================================================================== #
# The model entry: the Softmax node of a graph                                #
# =========================================================================== #
def _softmax_graph(tmp_path, shape, axis, opset, before=None, after=None,
                   name=None):
    """A float64 graph X -> [before] -> Softmax(axis) -> [after] -> Y on a
    tensor with a leading batch axis of one; ``before`` is
    ``(nodes, inits)``, ``after`` is ``(nodes, inits, output shape)``."""
    nodes, inits, src = [], {}, "X"
    if before is not None:
        nodes += before[0]
        inits.update(before[1])
        src = before[0][-1].output[0]
    nodes.append(helper.make_node("Softmax", [src], ["S"], axis=axis))
    out_shape = shape
    if after is not None:
        nodes += after[0]
        inits.update(after[1])
        out_shape = after[2]
    else:
        nodes[-1].output[0] = "Y"
    return save_onnx(tmp_path, nodes, shape, out_shape, inits,
                     name=name or f"softmax_{opset}_{axis}", opset=opset,
                     in_type="DOUBLE", out_type="DOUBLE")


_GRAPH_AXES = [(13, 1), (13, 2), (13, -1), (13, -3), (11, 1), (11, 2)]


@pytest.mark.parametrize("kind", ["Box", "Zono", "Star", "Star-row"])
@pytest.mark.parametrize("opset,axis", _GRAPH_AXES,
                         ids=[f"opset{o}-axis{a}" for o, a in _GRAPH_AXES])
def test_onnx_softmax_on_a_flat_set_holds_every_model_output(tmp_path, kind,
                                                             opset, axis):
    """ONNX Softmax over each axis of X (1, 2, 3, 2) on a flat set, through
    ``reach_pytorch_model`` with the batch-stripped ``input_shape``: at
    opset 13 one axis (axes 1 and -3 have two axes after them), at opset 11
    the flattened tail from the axis. Every onnxruntime output (float64) of
    the member points is a member of a returned set by LP.

    Catches: the axis handed to the kernel wrong (resolved, forwarded or put
    back wrongly), the opset 11 tail coerced to the wrong rows, a wrong row
    of a Star cut by a binding row, a result whose entries are stored in
    another order than the model's output, and a kernel that writes into
    the input set's arrays (``reach_model`` fails on a changed input set)."""
    shape = (1, 2, 3, 2)
    path = _softmax_graph(tmp_path, shape, axis, opset)
    lo, hi = _logit_bounds(12, seed=9)
    if kind == "Star-row":
        input_set, points = constrained_star(lo, hi, seed=7)
    else:
        input_set = flat_set(kind, lo, hi)
        # the rows the model normalises: one axis at opset 13, the flattened
        # tail from the axis at opset 11
        if opset >= 13:
            rows_shape, rows_axis = (2, 3, 2), (axis - 1 if axis > 0 else axis)
        else:
            rows_shape, rows_axis = (int(np.prod(shape[:axis])),
                                     int(np.prod(shape[axis:]))), -1
        points = _attention_points(lo, hi, rows_shape, rows_axis,
                                   np.random.default_rng(3))
    what = f"ONNX Softmax opset {opset} axis {axis} on {kind}"
    sets = reach_model(load_onnx(path), input_set, "approx", what,
                       input_shape=(2, 3, 2))
    assert_covered(sets, replay_onnx(path, points, shape), what)


@pytest.mark.parametrize("kind", ["Star", "Star-row"])
@pytest.mark.parametrize("shape,axis", [((1, 2, 2, 2), 1)],
                         ids=["(1,2,2,2)-axis1"])
def test_onnx_softmax_after_an_exact_split_holds_every_piece(tmp_path, kind,
                                                             shape, axis):
    """Relu (exact) then Softmax over axis 1 of X (1, 2, 2, 2), which has two
    axes after it: the Softmax receives one Star per activation pattern, with
    different logit ranges, and must answer for every one (under ``exact`` it
    answers with the over-approximation, registry REACH-4, which this cell
    does not judge). Catches a kernel or a handler that keeps only the first
    piece, or uses one piece's ranges for all."""
    n = int(np.prod(shape))
    relu = ([helper.make_node("Relu", ["X"], ["H"])], {})
    path = _softmax_graph(tmp_path, shape, axis, 13, before=relu)
    lo, hi = _logit_bounds(n, seed=10)
    if kind == "Star-row":
        input_set, points = constrained_star(lo, hi, seed=8)
    else:
        input_set, points = flat_set("Star", lo, hi), samples(lo, hi)
    what = f"Relu (exact) then Softmax axis {axis} of {shape} on {kind}"
    sets = reach_model(load_onnx(path), input_set, "exact", what,
                       input_shape=shape[1:])
    if len(sets) < 2:
        pytest.fail(f"{what}: the ReLU must split the input")
    assert_covered(sets, replay_onnx(path, points, shape), what)


_SM_IN = np.array([[0.5, -1.0, 0.25], [-0.75, 0.5, 1.0], [1.0, 0.25, -0.5]])
_SM_OUT = np.array([[1.0, -0.5], [0.25, 0.75], [-1.0, 0.5]])
_SM_BIAS = np.array([[[0.25, -0.5], [1.0, 0.0]]])


@pytest.mark.parametrize("kind", ["Box", "Zono", "Star", "Star-row"])
@pytest.mark.parametrize("axis", [-1, 1])
def test_onnx_softmax_between_two_affine_nodes(tmp_path, kind, axis):
    """X (1, 2, 3) @ W, Softmax over the keys (axis -1) or over the two rows
    (axis 1), then @ W2 + B: the box the Softmax returns is mapped by the
    next nodes, so a wrong entry or order of the Softmax result shows in
    the model output. Catches the Softmax reading the wrong node's sets or
    shape and a result the next node does not receive."""
    before = ([helper.make_node("MatMul", ["X", "Win"], ["P"])],
              {"Win": _SM_IN})
    after = ([helper.make_node("MatMul", ["S", "Wout"], ["Q"]),
              helper.make_node("Add", ["Q", "B"], ["Y"])],
             {"Wout": _SM_OUT, "B": _SM_BIAS}, (1, 2, 2))
    path = _softmax_graph(tmp_path, (1, 2, 3), axis, 13, before, after)
    lo, hi = _logit_bounds(6, seed=11, scale=1.0)
    if kind == "Star-row":
        input_set, points = constrained_star(lo, hi, seed=9)
    else:
        input_set, points = flat_set(kind, lo, hi), samples(lo, hi)
    what = f"MatMul, Softmax axis {axis}, MatMul, Add on {kind}"
    sets = reach_model(load_onnx(path), input_set, "approx", what,
                       input_shape=(2, 3))
    assert_covered(sets, replay_onnx(path, points, (1, 2, 3)), what)


@pytest.mark.parametrize("kind", ["Box", "Zono", "Star"])
def test_onnx_softmax_inside_a_residual_connection_holds_every_output(
        tmp_path, kind):
    """Y = X - Softmax(X) over the rows of X (1, 2, 3): the input set is read
    twice, by the Softmax and by the Sub, and the Sub joins
    the Softmax result with the input. Every entry has positive width, so a
    Star re-encoded with ``Star.from_bounds`` would carry the predicate
    system of the input (alpha in [-1, 1]^6, rows +-I) and the Sub would
    read the two as the same variables. Catches that (the output must be on
    new predicate variables, ``Star.fresh_from_bounds``) and a kernel that
    changes its input set in place."""
    shape = (1, 2, 3)
    path = _softmax_graph(tmp_path, shape, -1, 13,
                          after=([helper.make_node("Sub", ["X", "S"],
                                                   ["Y"])], {}, shape),
                          name="residual")
    lo, hi = flat_box(6, seed=2)
    lo, hi = 4.0 * lo, 4.0 * hi
    what = f"X - Softmax(X) on {kind}"
    sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                       what, input_shape=shape[1:])
    assert_covered(sets, replay_onnx(path, samples(lo, hi), shape), what)


@pytest.mark.parametrize("kind", ["Box", "Zono", "Star-row"])
@pytest.mark.parametrize("entry", ["reach_pytorch_model", "NeuralNetwork"])
def test_torch_softmax_in_a_model_holds_every_output(kind, entry):
    """``nn.Softmax(dim=1)`` in a traced torch model on a flat set that
    stands for a (1, 2, 3, 2) tensor (``input_shape``), through
    ``reach_pytorch_model`` and ``NeuralNetwork.reach``: the handler reads
    the torch module's ``dim`` (an ONNX node has ``axis``), and axis 1 has
    two axes after it. Every torch float64 output of the member points is a
    member of a returned set by LP. Catches the dimension of a torch module
    read wrongly or not at all and a result in another order than the
    model's output."""
    shape = (1, 2, 3, 2)
    model = nn.Sequential(nn.Softmax(dim=1)).double().eval()
    lo, hi = _logit_bounds(12, seed=13)
    if kind == "Star-row":
        input_set, points = constrained_star(lo, hi, seed=5)
    else:
        input_set = flat_set(kind, lo, hi)
        points = _attention_points(lo, hi, (2, 3, 2), 0,
                                   np.random.default_rng(6))
    what = f"torch Softmax(dim=1) through {entry} on {kind}"
    if entry == "NeuralNetwork":
        sets = NeuralNetwork(model).reach(input_set, method="approx",
                                          input_shape=shape[1:])
        if not sets:
            raise AssertionError(f"{what}: no output set")
    else:
        sets = reach_model(model, input_set, "approx", what,
                           input_shape=shape[1:])
    assert_covered(sets, replay_torch(model, points, shape), what)


_INFINITE_LOGIT_GIVES_NAN = known_bug(
    7, "correlated_row_softmax_bounds evaluates the softmax at an infinite "
       "logit bound (inf minus the row maximum inf is NaN), so a Box with an "
       "infinite upper bound comes back from a Softmax node with NaN bounds, "
       "which hold no point, although every member has a softmax in [0, 1]")
_INFINITE_LOGIT_CRASHES_THE_STAR = known_bug(
    7, "a Star whose predicate bounds are infinite (an ImageStar bounded by "
       "its rows, flattened) gets infinite logit ranges from "
       "estimate_ranges; correlated_row_softmax_bounds turns them into NaN "
       "and the reach returns a Star whose generators hold NaN or inf, which "
       "is no set, where an answer is due; the membership oracle of the tests "
       "refuses that Star with ValueError ('the set's generators hold NaN or "
       "inf')", raises=ValueError)


def _flattened_image_softmax(tmp_path):
    """X (1, 2, 3, 2) -> Flatten -> Reshape (1, 2, 6) -> Softmax(axis 2)
    -> Flatten, float64."""
    nodes = [helper.make_node("Flatten", ["X"], ["F"], axis=1),
             helper.make_node("Reshape", ["F", "s"], ["R"]),
             helper.make_node("Softmax", ["R"], ["S"], axis=2),
             helper.make_node("Flatten", ["S"], ["Y"], axis=1)]
    return save_onnx(tmp_path, nodes, [1, 2, 3, 2], [1, 12],
                     {"s": np.array([1, 2, 6], dtype=np.int64)},
                     name="image_softmax", in_type="DOUBLE",
                     out_type="DOUBLE")


# numpy warns on inf - inf; in ordinary use that is a warning, not an error,
# and the cell judges what the reach returns
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("case", [
    pytest.param("Box, an infinite upper bound",
                 marks=_INFINITE_LOGIT_GIVES_NAN),
    "Box, an infinite lower bound",
    pytest.param("ImageStar, infinite predicate bounds",
                 marks=_INFINITE_LOGIT_CRASHES_THE_STAR),
    "ImageStar, one infinite predicate bound",
])
def test_onnx_softmax_of_a_set_with_an_infinite_bound_holds_every_output(
        tmp_path, case):
    """Input sets that hold an infinite bound where the type allows one, on
    the model entry: a Box with +inf or -inf ends into ONNX Softmax over the
    rows of X (1, 2, 3); an ImageStar bounded by its rows whose predicate
    bounds are all infinite, or only one, flattened and passed to a Softmax
    over the six entries of each channel (the model entry finitizes a flat
    Star's predicate box, not an ImageStar's). Every finite member has a
    softmax in [0, 1], so the result must hold the onnxruntime outputs (a
    refusal with ``NotImplementedError`` keeps the promise too). Catches
    NaN bounds from an infinite logit and a crash on one. The cells with
    -inf only and with one infinite predicate bound pass today."""
    if case.startswith("Box"):
        path = _softmax_graph(tmp_path, (1, 2, 3), -1, 13, name="box_inf")
        lo = np.array([-1.0, 0.0, -2.0, -0.5, 0.5, -1.0])
        hi = np.array([1.0, 1.5, -1.0, 0.0, 1.5, 2.0])
        if "upper" in case:
            hi[1] = np.inf
        else:
            lo[3] = -np.inf
        finite_lo, finite_hi = (np.where(np.isfinite(lo), lo, -64.0),
                                np.where(np.isfinite(hi), hi, 64.0))
        input_set, options = Box(lo, hi), {"input_shape": (2, 3)}
        points, in_shape = samples(finite_lo, finite_hi), (1, 2, 3)
    else:
        path = _flattened_image_softmax(tmp_path)
        lo, hi = pixel_box(2, 3, 2, seed=3)
        image, points = constrained_image_star(lo, hi, n=64, seed=1)
        k = image.C.shape[1]
        if case.endswith("one infinite predicate bound"):
            lower = np.vstack([[[-np.inf]], -np.ones((k - 1, 1))])
            upper = np.ones((k, 1))
        else:
            lower, upper = np.full((k, 1), -np.inf), np.full((k, 1), np.inf)
        try:
            input_set = ImageStar(image.V, image.C, image.d, lower, upper,
                                  3, 2, 2)
        except Exception as error:     # a broken setup is not the defect
            pytest.fail(f"building the ImageStar failed: {error!r}")
        options, in_shape = {}, (1, 2, 3, 2)
    what = f"Softmax on {case}"
    try:
        sets = reach_model(load_onnx(path), input_set, "approx", what,
                           **options)
    except NotImplementedError:
        return
    for s in sets:
        if isinstance(s, Box) and (np.isnan(np.asarray(s.lb)).any()
                                   or np.isnan(np.asarray(s.ub)).any()):
            raise AssertionError(f"{what}: the returned Box has NaN bounds, "
                                 f"which hold no point")
    assert_covered(sets, replay_onnx(path, points, in_shape), what)


@pytest.mark.parametrize("method,kind", [
    ("exact", "Zono"), ("exact", "ImageZono"), ("exact", "Box"),
    ("exact", "Hexatope"), ("exact", "Octatope"), ("approx", "ImageZono"),
])
def test_softmax_node_refuses_the_rows_it_has_no_rule_for(tmp_path, method,
                                                          kind):
    """The rows of the attention Softmax that refuse: ``method='exact'`` on
    a set type with no exact method is refused by the model entry before
    any node (today ``ValueError``, "... only supports 'approx'"; the
    planned ``UnsupportedError`` subclasses ``NotImplementedError``), and a
    Softmax node on an ImageZono under approx is refused by the Zono lane
    of the dispatcher (the handler defers on image sets;
    ``NotImplementedError``). Catches the model entry passing 'exact' on to
    a kernel that never reads the method, and an image set answered by a
    kernel that takes it for a flat set. Hexatope and Octatope under approx
    are refused in ``test_soundness_softmax.py``
    (``test_onnx_softmax_refuses_a_hexatope_or_octatope``)."""
    path = _softmax_graph(tmp_path, (1, 2, 3, 2), 1, 13, name="refusal")
    if kind == "ImageZono":
        lo, hi = pixel_box(2, 3, 2)
        input_set, options = image_set(kind, lo, hi), {}
    else:
        lo, hi = flat_box(12, seed=1)
        input_set, options = flat_set(kind, lo, hi), {"input_shape": (2, 3, 2)}
    refusal = (NotImplementedError, ValueError) if method == "exact" \
        else NotImplementedError
    with pytest.raises(refusal):
        reach_model(load_onnx(path), input_set, method,
                    f"Softmax on {kind}, {method}", **options)
