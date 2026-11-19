"""Soundness cells for ONNX Add, Sub, Mul and Div between two computed sets,
and for Concat of operands with different predicate systems.

The code under test is case 1 of ``_handle_onnx_binary_op`` in
``n2v/nn/reach.py`` (both operands are computed sets; ONNX broadcasting
between their tensor shapes by row selection) and what it calls:
``_add_sets``; ``_mul_sets`` with ``_mul_boxes``, ``_mul_zonos`` and the
McCormick product ``_mul_stars_mccormick``; ``_div_sets`` with the Box
interval quotient and the McCormick quotient ``_div_stars_mccormick`` (both
McCormick kernels with their operand ranges ``_mccormick_operand_ranges``
and the ``lp_solver`` option); the
joining of predicate systems (``_same_predicate_system``,
``_join_predicates``, ``_join_star_systems``, the last one also for Concat);
the pairing of operand lists after an exact split (``_pair_set_lists``,
``_lineage_groups``); and the coercion of an image set and a flat set
(``_coerce_set_types``). Set types: Star (plain, and with a binding row that
is not a box row), Zono, Box, ImageStar (also one given by its rows alone,
with no predicate box), ImageZono; Hexatope and Octatope are refused. The
other entry routes of Add and Sub (a torch model, ``reach_layer`` on the
layer alone) are in ``test_soundness_residual_add.py``, those of Mul and Div
in ``test_soundness_mul_div.py``.

Models. Every model is an ONNX graph built with ``cell_support.save_onnx``,
loaded with ``load_onnx`` and reached through the model entry point. The two
operands are computed from the same input (two affine branches, a ReLU on
either, one operand computed from the other), so they are correlated and the
set types' joins and pairings are all exercised. Over the input box the
branch weights put each regime of every neuron in place: in branch A neuron
0 crosses zero, neuron 1 is positive and neuron 2 negative; in branch B
neuron 0 is positive, neuron 1 negative and neuron 2 crosses zero; the
divisor branch is positive, negative and positive. Further biases give the
products and quotients the remaining sign patterns of their operands
(``_BB_SAME``, ``_BA_NEG``), and a ReLU branch shifted by a constant gives
an operand on a system of its own that stays away from zero.

Oracles, all computed without n2v.

- Add, Sub and Concat are piecewise linear: the exact image in rationals
  (``tests/_oracles/rational.py``), as pieces over the input set's own
  predicate polytope read from its stored arrays, so a Star's binding row is
  part of the input. Every vertex of every piece must be a member of one
  returned set that holds the whole piece, decided in exact arithmetic with
  no slack (a convex set that holds the vertices holds the piece). Under
  method='exact' also the other direction: every vertex of every returned
  set has a preimage (``rational.unreached``).
- Mul and Div are smooth: the containment proof by interval branch and
  bound (``tests/_oracles/interval_proof.py``) over the same predicate
  polytope, with a product stage and a reciprocal stage written here; where
  the proof is too slow (four predicate variables, the exact line a
  constant operand entry gives) the exact outputs of sample points, decided
  in exact arithmetic; for the image products and quotients (30 entries)
  the replay alone.
- Every cell: onnxruntime (or float64 torch) replay of member points
  (corners first) decided by one LP on the rows of the returned sets
  (``cell_support.assert_covered``).

Data are dyadic (multiples of small powers of two), so float32 onnxruntime,
float64 and the rationals agree; the Div models are float64, so their replay
rounds at the 1e-16 scale, inside the oracle's ``LP_TOL``. The rounding cells
use float64 models with non-dyadic data and exact fractions only.

Layout. A network tensor of an image is (C, H, W); an image set stores
(H, W, C); a flat set keeps the tensor's own order. Exact outputs are
computed in the network's order and permuted to the stored order with
``hwc_index``; replayed outputs are converted by ``assert_covered`` with
``out_chw``. Images are (2, 3, 5) with asymmetric (1, 3) and (3, 1)
kernels, and (2, 2, 4) for the reduced-copy cells. Some cells scale the box
by 8, so that predicate variables do not all range over about [-1, 1].
"""

import itertools
from fractions import Fraction

import numpy as np
import pytest
from onnx import helper

from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests._oracles import interval_proof, rational
from tests.cell_support import (N_SAMPLES, assert_covered,
                                constrained_image_star,
                                constrained_star, flat_set, hwc_index,
                                image_set, pixel_box, reach_model,
                                replay_onnx, replay_torch, samples, save_onnx)
from tests.known_bugs import DID_NOT_RAISE, known_bug


def _reach(model, input_set, method, what, **options):
    """``cell_support.reach_model``, with its empty-result check reported by
    ``pytest.fail``: an ``AssertionError`` there is not the failure a strict
    expected-failure marker of these cells documents, so no marker may
    absorb it. A refusal (``NotImplementedError``) and any other exception
    pass through."""
    try:
        return reach_model(model, input_set, method, what, **options)
    except AssertionError as error:
        pytest.fail(str(error))


N_POINTS = 48

# --------------------------------------------------------------------------- #
# The two-branch flat model                                                  #
# --------------------------------------------------------------------------- #
_LO, _HI = np.array([-1.0, -0.5]), np.array([1.0, 0.75])

# X @ W + B for X of shape (1, 2). Over [_LO, _HI]: A0 in [-21/16, 21/16]
# crosses zero, A1 in [1.3125, 2.625] is positive, A2 in [-2, -0.875]
# negative; B0 in [1.5, 3.75] positive, B1 in [-3.125, -1] negative, B2 in
# [-13/8, 13/8] crosses. A crossing range symmetric about zero gives the
# approximate ReLU the slope 1/2, so every relaxation coefficient is dyadic
# and no cell depends on the rounding of the ReLU kernels or of the affine
# map that follows (ROUND-8, ROUND-10), which belong to other kernels.
_WA = np.array([[1.0, 0.5, -0.25], [0.5, -0.25, 0.5]])
_BA = np.array([-0.0625, 2.0, -1.5])
_WB = np.array([[-0.5, 0.75, 1.0], [1.0, -0.5, -1.0]])
_BB = np.array([2.5, -2.0, 0.125])
# The bias of the divisor branch: with it B0 is in [2, 4.25], B1 in
# [-4.125, -2] and B2 in [0.75, 4], away from zero with both signs.
_BB_DIV = np.array([3.0, -3.0, 2.5])
# The bias of a numerator branch away from zero: with it A0 is in
# [1.75, 4.375], A1 in [1.3125, 2.625] and A2 in [-3.5, -2.375].
_BA_AWAY = np.array([3.0, 2.0, -3.0])
# A third branch for Concat: every neuron crosses zero, on a range symmetric
# about zero ([-9/8, 9/8], [-37/32, 37/32], [-7/8, 7/8]).
_WC = np.array([[0.5, -1.0, 0.25], [-1.0, 0.25, 1.0]])
_BC = np.array([0.125, -0.03125, -0.125])
# Q @ _TAIL after the operation: y = (2 q2, -q0, q1 / 2). Invertible, so a
# wrong operation result stays wrong after it, and exact on every float, so
# the cells do not depend on the round-to-nearest affine map of a Zono or a
# Star (ROUND-8): the Zono ReLU's error generator is not dyadic (its bounds
# are padded), and a mixing tail would leave its image out by 1e-16.
_TAIL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.5], [2.0, 0.0, 0.0]])
# The same for two outputs: y = (2 q1, -q0).
_TAIL2 = np.array([[0.0, -1.0], [2.0, 0.0]])

_REGIMES = {"affine-affine": (False, False), "relu-affine": (True, False),
            "affine-relu": (False, True), "relu-relu": (True, True)}
_FLAT_KINDS = ("Star", "Star+row", "Zono", "Box")


def _input(kind, n=N_POINTS, scale=1):
    """``(input set, member points)`` on the box [_LO, _HI] times ``scale``.
    ``Star+row`` is a Star with one binding row that is not a box row (its
    bounding box is still the box); the others are the box itself."""
    lo, hi = scale * _LO, scale * _HI
    if kind == "Star+row":
        return constrained_star(lo, hi, n=n, seed=3)
    return flat_set(kind, lo, hi), samples(lo, hi, n)


def _branch(nodes, inits, tag, source, W, B, relu, dtype):
    """``source @ W + B``, then a ReLU when ``relu``; the output name."""
    inits[f"W{tag}"] = np.asarray(W, dtype=dtype)
    inits[f"B{tag}"] = np.asarray(B, dtype=dtype)
    nodes += [helper.make_node("MatMul", [source, f"W{tag}"], [f"P{tag}"]),
              helper.make_node("Add", [f"P{tag}", f"B{tag}"], [f"H{tag}"])]
    if not relu:
        return f"H{tag}"
    nodes.append(helper.make_node("Relu", [f"H{tag}"], [f"R{tag}"]))
    return f"R{tag}"


def _two_branch_model(tmp_path, op, relu_a, relu_b, bb=_BB, tail=True,
                      swap=False, ba=_BA, cols=(0, 1, 2), scale=1,
                      shift_b=None):
    """X -> A = [Relu](X @ _WA + ba), B = [Relu](X @ _WB + bb) [+ shift_b],
    both restricted to the neurons ``cols`` -> op(A, B) (op(B, A) when
    ``swap``) -> [@ the tail]. Float64 for Div, so the replay of a quotient
    is exact to 1e-16; float32 (exact on dyadic data) otherwise. ``scale``
    multiplies the biases (and the shift), for an input box multiplied by
    the same power of two: every branch value is then scaled too and each
    crossing range stays symmetric about zero (dyadic ReLU slopes).
    ``shift_b`` adds a constant to every entry of B after its ReLU (an ONNX
    Add of a constant), which moves a ReLU branch away from zero."""
    cols = list(cols)
    dtype = np.float64 if op == "Div" else np.float32
    nodes, inits = [], {}
    a = _branch(nodes, inits, "A", "X", _WA[:, cols], scale * ba[cols],
                relu_a, dtype)
    b = _branch(nodes, inits, "B", "X", _WB[:, cols], scale * bb[cols],
                relu_b, dtype)
    if shift_b is not None:
        inits["SB"] = np.full(len(cols), scale * shift_b, dtype=dtype)
        nodes.append(helper.make_node("Add", [b, "SB"], ["HS"]))
        b = "HS"
    first, second = (b, a) if swap else (a, b)
    nodes.append(helper.make_node(op, [first, second],
                                  ["Q" if tail else "Y"]))
    if tail:
        inits["T"] = _tail(len(cols)).astype(dtype)
        nodes.append(helper.make_node("MatMul", ["Q", "T"], ["Y"]))
    kind = "DOUBLE" if dtype == np.float64 else "FLOAT"
    return save_onnx(tmp_path, nodes, [1, 2], [1, len(cols)], inits=inits,
                     in_type=kind, out_type=kind)


def _tail(n):
    return {3: _TAIL, 2: _TAIL2}[n]


# --------------------------------------------------------------------------- #
# Exact references for piecewise-linear graphs                               #
# --------------------------------------------------------------------------- #
def _eye(n):
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def _predicate_pieces(input_set):
    """The input set as one rational piece over its own predicate polytope:
    ``x = c + G a`` on ``{C a <= d, lb <= a <= ub}``, read from the stored
    arrays (``rational.star``). A variable with a zero generator column that
    no row mentions (an ImageZono keeps one for every pixel of zero width)
    does not move ``x`` and is left out, which leaves the set unchanged. A
    star without a predicate box (an ImageStar given by its rows alone,
    :func:`_rows_only`) has no bound rows: its rows ``C a <= d`` bound it."""
    c, G, C, d, lb, ub = rational.star(input_set)
    keep = [j for j in range(len(lb))
            if any(row[j] != 0 for row in G) or any(row[j] != 0 for row in C)]
    G = [[row[j] for j in keep] for row in G]
    C = [[row[j] for j in keep] for row in C]
    lb, ub = [lb[j] for j in keep], [ub[j] for j in keep]
    unit = _eye(len(keep))
    upper = [(r, u) for r, u in zip(unit, ub) if u is not None]
    lower = [([-v for v in r], -l) for r, l in zip(unit, lb) if l is not None]
    rows = C + [r for r, _ in upper] + [r for r, _ in lower]
    rhs = list(d) + [v for _, v in upper] + [v for _, v in lower]
    return rational.affine(rational.polytope(rows, rhs), G, c)


def _relu_on(pieces, coords):
    """ReLU on the output coordinates ``coords`` only. The others are moved
    above zero by a constant larger than any value they take, through the
    ReLU (the identity there, with no cut) and back."""
    n = len(pieces[0].W)
    big = 1 + max(abs(v) for piece in pieces for y in piece.image()
                  for v in y)
    shift = [Fraction(0) if i in coords else big for i in range(n)]
    pieces = rational.relu(rational.affine(pieces, _eye(n), shift))
    return rational.affine(pieces, _eye(n), [-s for s in shift])


def _combine(op, n=3):
    """The rows of ``A op B`` on the stacked vector [A; B]."""
    s = 1 if op == "Add" else -1
    return [[int(j == i) + s * int(j == n + i) for j in range(2 * n)]
            for i in range(n)]


def _two_branch_pieces(input_set, op, relu_a, relu_b, tail=True, swap=False,
                       scale=1):
    """The exact image of the model of :func:`_two_branch_model` (Add or
    Sub, biases times ``scale``) on the input set, as rational pieces."""
    pieces = _predicate_pieces(input_set)
    W = np.vstack([_WA.T, _WB.T])
    pieces = rational.affine(pieces, W.tolist(),
                             (scale * np.concatenate([_BA, _BB])).tolist())
    pieces = _relu_on(pieces, ([0, 1, 2] if relu_a else [])
                      + ([3, 4, 5] if relu_b else []))
    rows = _combine(op)
    if swap:                                    # op(B, A)
        rows = [r[3:] + r[:3] for r in rows]
    pieces = rational.affine(pieces, rows)
    if tail:
        pieces = rational.affine(pieces, _TAIL.T.tolist())
    return pieces


def _holders(sets, what, out_chw):
    """Exact membership tests of the returned sets, each with the
    permutation that puts a tensor-order output in its stored order."""
    order = None if out_chw is None else list(hwc_index(*out_chw))
    out = []
    for s in sets:
        image = isinstance(s, (ImageStar, ImageZono))
        if image:
            if out_chw is None:
                raise AssertionError(f"{what}: an image set came back for a "
                                     f"flat output")
            channels, height, width = out_chw
            if (s.height, s.width, s.num_channels) != (height, width,
                                                       channels):
                raise AssertionError(
                    f"{what}: the result stores {s.height}x{s.width}x"
                    f"{s.num_channels}, the output is (C, H, W) = {out_chw}")
        out.append((rational.Membership(rational.star(s)),
                    order if image else None))
    return out


def _assert_pieces_held(pieces, sets, what, out_chw=None):
    """Every exact piece has all its vertices in one returned set, decided
    in exact arithmetic with no slack: then the returned sets contain the
    exact image. A vertex in no returned set is a soundness failure. A piece
    whose vertices are all held but spread over several sets is
    inconclusive, which is a fault of the test, not a verdict."""
    holders = _holders(sets, what, out_chw)
    size = len(pieces[0].W)

    def member(holder, y):
        test, order = holder
        if len(test.c) != size:
            raise AssertionError(f"{what}: a returned set has {len(test.c)} "
                                 f"entries, the true output has {size}")
        return test([y[j] for j in order] if order else list(y))

    for i, piece in enumerate(pieces):
        image = piece.image()
        if any(all(member(h, y) for y in image) for h in holders):
            continue
        outside = [y for y in image if not any(member(h, y) for h in holders)]
        if outside:
            shown = ", ".join(f"{float(v):g}" for v in outside[0][:6])
            raise AssertionError(
                f"{what}: {len(outside)} exact vertices of piece {i} of "
                f"{len(pieces)} are in none of the {len(sets)} returned sets, "
                f"for example ({shown})")
        pytest.fail(f"{what}: piece {i} is spread over several returned "
                    f"sets; cut it before the vertex test can decide")


def _assert_points_held(sets, outputs, what, out_chw=None):
    """Exact outputs (rational vectors in the network's order) are members
    of the returned sets, decided in exact arithmetic with no slack."""
    holders = _holders(sets, what, out_chw)
    for i, y in enumerate(outputs):
        y = list(y)
        if not any(test([y[j] for j in order] if order else y)
                   for test, order in holders):
            shown = ", ".join(f"{float(v):g}" for v in y[:6])
            raise AssertionError(f"{what}: the exact output of point {i} "
                                 f"({shown}, ...) is in none of the "
                                 f"{len(sets)} returned sets")


# --------------------------------------------------------------------------- #
# Add and Sub of two flat branches                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime", sorted(_REGIMES))
@pytest.mark.parametrize("op", ["Add", "Sub"])
@pytest.mark.parametrize("kind", _FLAT_KINDS)
def test_add_or_sub_of_two_branches_holds_the_exact_image(tmp_path, kind, op,
                                                          regime):
    """A op B with A = [Relu](x WA + bA), B = [Relu](x WB + bB), then an
    invertible affine tail, on a Star (plain and with a binding row), a Zono
    and a Box. With two affine branches a Star keeps one predicate system
    (an exact sum); a ReLU on either branch appends relaxation variables and
    the systems are joined block-diagonally; on a Zono each ReLU appends its
    own error generator at a different neuron. Catches a dropped or wrongly
    signed centre or generator block, a subtraction computed as an addition,
    a Box bound taken from the wrong end, the second operand's relaxation
    rows written on the first operand's predicate columns, a subtracted
    ReLU branch whose generators are not negated, two Zonos' error
    generators merged as if they were one noise symbol, and a join decided
    by the number of variables instead of the systems."""
    relu_a, relu_b = _REGIMES[regime]
    path = _two_branch_model(tmp_path, op, relu_a, relu_b)
    input_set, points = _input(kind)
    what = f"{op} of two branches ({regime}) on a {kind}"
    sets = _reach(load_onnx(path), input_set, "approx", what)
    _assert_pieces_held(_two_branch_pieces(input_set, op, relu_a, relu_b),
                        sets, what)
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)


_WIDE = 8


def _skewed_input(kind, scale, n=N_POINTS):
    """``(input set, member points)`` whose inner shape is not the one
    ``from_bounds`` gives: the box [_LO, _HI] times ``scale`` as a Zono with
    four generators on its two entries (``Zono+gens``), or as a Star with
    four predicate variables on two entries cut by one binding row
    (``Star+vars``). Each entry's generators sum in absolute value to its
    half-width, so the Zono's bounding box is the box (the Star's row cuts
    off the corners where a0 + a1 + a2 / 2 > 3 / 2). Member points are
    images of dyadic predicate points (corners and the half grid) that keep
    the row."""
    lo, hi = scale * _LO, scale * _HI
    c, r = (lo + hi) / 2, (hi - lo) / 2
    G = np.array([[r[0] / 2, r[0] / 4, r[0] / 4, 0.0],
                  [0.0, r[1] / 2, -r[1] / 4, r[1] / 4]])
    row, rhs = np.array([1.0, 1.0, 0.5, 0.0]), 1.5
    rng = np.random.default_rng(7)
    alphas = [a for a in itertools.product((-1.0, 1.0), repeat=4)]
    alphas += [rng.integers(-2, 3, size=4) / 2.0 for _ in range(8 * n)]
    if kind == "Star+vars":
        alphas = [a for a in alphas if row @ np.asarray(a) <= rhs]
        input_set = Star(np.hstack([c[:, None], G]), row[None, :],
                         np.array([[rhs]]), -np.ones((4, 1)), np.ones((4, 1)))
    else:
        input_set = Zono(c.reshape(-1, 1), G)
    points = [c + G @ np.asarray(a) for a in alphas[:n]]
    return input_set, points


@pytest.mark.parametrize("op", ["Add", "Sub"])
@pytest.mark.parametrize("kind", ["Star+vars", "Zono+gens", "Box"])
def test_add_or_sub_of_two_branches_on_a_wide_box(tmp_path, kind, op):
    """The relu-relu cell above on the box times 8 (x0 in [-8, 8], x1 in
    [-4, 6]) with the biases times 8, so every value is scaled and each
    crossing range stays symmetric about zero. On the unit-scale box every
    input predicate variable, and most relaxation variables, range over
    about [-1, 1] or [0, 2]; here the input variables have generators 8 and
    5 and the relaxation variables reach 13. ``Star+vars`` (with a binding
    row) and ``Zono+gens`` give the box four predicate variables or
    generators on two entries (:func:`_skewed_input`), so each operand has
    more variables than entries. Catches a join or a sum that reads one
    operand's variable, generator or predicate bound in place of the
    other's, or counts them by the number of entries, which data inside
    [-1, 1] and square bases can hide."""
    path = _two_branch_model(tmp_path, op, True, True, scale=_WIDE)
    if kind in ("Star+vars", "Zono+gens"):
        input_set, points = _skewed_input(kind, _WIDE)
    else:
        input_set, points = _input(kind, scale=_WIDE)
    what = f"{op} of two ReLU branches on a {kind} times {_WIDE}"
    sets = _reach(load_onnx(path), input_set, "approx", what)
    _assert_pieces_held(_two_branch_pieces(input_set, op, True, True,
                                           scale=_WIDE), sets, what)
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)


@pytest.mark.parametrize("op", ["Add", "Sub"])
@pytest.mark.parametrize("kind", _FLAT_KINDS)
def test_operands_in_the_other_order(tmp_path, kind, op):
    """B op A with a ReLU on A only: the joined system has the relaxation
    variables second, and Sub negates the first operand's own block. Catches
    an operand order lost between the node and the join (the sets of one
    input used for the other)."""
    path = _two_branch_model(tmp_path, op, True, False, swap=True)
    input_set, points = _input(kind)
    what = f"{op}(B, relu A) on a {kind}"
    sets = _reach(load_onnx(path), input_set, "approx", what)
    _assert_pieces_held(_two_branch_pieces(input_set, op, True, False,
                                           swap=True), sets, what)
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)


# One operand computed from the other: H and Relu(H) on one node.
_CORRELATED = {
    "h-relu(h)": ("Sub", "H", "R"),       # min(h, 0)
    "relu(h)-h": ("Sub", "R", "H"),       # max(-h, 0)
    "h+relu(h)": ("Add", "H", "R"),
    "relu(h)+relu(h)": ("Add", "R", "R"),  # one set used twice
    "h-h": ("Sub", "H", "H"),             # x - x = 0
}


@pytest.mark.parametrize("case", sorted(_CORRELATED))
@pytest.mark.parametrize("kind", _FLAT_KINDS)
def test_operand_computed_from_the_other(tmp_path, kind, case):
    """Operands that are one node and its ReLU, or the same node twice. The
    true image (min(h, 0), 2 relu(h), ...) is far smaller than the sum of
    the operands' boxes, and a result that ignores the correlation is still
    sound; one that pairs the wrong variables, or reads one system's rows as
    the other's, is not. Catches the same faults as the two-branch cell on a
    graph where the second operand's system extends the first's, and with
    "h-h" (the same node on both sides of a Sub, true output 0) a node fed
    twice by one value that is answered as that value or as its double."""
    op, first, second = _CORRELATED[case]
    nodes, inits = [], {}
    _branch(nodes, inits, "", "X", _WA, _BA, True, np.float32)  # H and R
    nodes += [helper.make_node(op, [first, second], ["Q"]),
              helper.make_node("MatMul", ["Q", "T"], ["Y"])]
    inits["T"] = _TAIL.astype(np.float32)
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 3], inits=inits)
    input_set, points = _input(kind)
    what = f"{case} on a {kind}"
    sets = _reach(load_onnx(path), input_set, "approx", what)
    pieces = rational.affine(_predicate_pieces(input_set), _WA.T.tolist(),
                             _BA.tolist())
    # [h; h] with the ReLU on the copies that stand for R.
    pieces = rational.affine(pieces, _eye(3) + _eye(3))
    pieces = _relu_on(pieces, [3 * k + i for k, name in
                               enumerate((first, second)) if name == "R"
                               for i in range(3)])
    pieces = rational.affine(rational.affine(pieces, _combine(op)),
                             _TAIL.T.tolist())
    _assert_pieces_held(pieces, sets, what)
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)


def _sin_then_cos(box, forms):
    """The stage [x; x] -> [sin x; cos x]."""
    return interval_proof.SIN(box, forms[:1]) + interval_proof.COS(box,
                                                                   forms[1:])


_LOOK_ALIKE = {
    # variant: (input shape, output shape, input kind)
    "split-outputs": ([1, 4], [1, 2], "Star"),
    "image": ([1, 1, 1, 2], [1, 1, 1, 2], "ImageStar"),
}


@pytest.mark.parametrize("variant", sorted(_LOOK_ALIKE))
def test_operands_whose_systems_differ_only_in_their_tags_are_kept_apart(
        tmp_path, variant):
    """Sin(X) - Cos(X) where the Sub's operands are not the Sin and Cos
    nodes' own Stars of one entry (that item is
    ``test_operands_whose_systems_differ_only_in_their_tags``): the first
    halves of a Split of each (``split-outputs``, the join's operands are
    then outputs of multi-output nodes, read by index), and ImageStars
    (``image``). Every entry ranges over [0, 7], so Sin and Cos each append
    look-alike box-only variables. Every float64 output of a sample point
    must be a member. Catches a walk that does not see a join whose operands
    come from multi-output nodes, and new variables tagged on a Star and not
    on an ImageStar: either reads Sin(x) - Cos(x) as the single point 0,
    while the true values reach 1.41.
    Replay only: the proof of the flat item covers the rule itself."""
    in_shape, out_shape, kind = _LOOK_ALIKE[variant]
    nodes = [helper.make_node("Sin", ["X"], ["A"]),
             helper.make_node("Cos", ["X"], ["B"])]
    inits = None
    if variant == "split-outputs":
        nodes += [helper.make_node("Split", ["A", "sp"], ["A0", "A1"], axis=1),
                  helper.make_node("Split", ["B", "sp"], ["B0", "B1"], axis=1),
                  helper.make_node("Sub", ["A0", "B0"], ["Y"])]
        inits = {"sp": np.array([2, 2], dtype=np.int64)}
    else:
        nodes.append(helper.make_node("Sub", ["A", "B"], ["Y"]))
    path = save_onnx(tmp_path, nodes, in_shape, out_shape, inits=inits)
    lo, hi = np.zeros(in_shape[1:]), np.full(in_shape[1:], 7.0)
    input_set = (image_set(kind, lo, hi) if kind == "ImageStar"
                 else flat_set(kind, lo, hi))
    what = f"Sin(X) - Cos(X), {variant}, on an {kind}"
    model = load_onnx(path)
    options = {} if kind == "ImageStar" else {"input_shape": tuple(
        in_shape[1:])}
    sets = _reach(model, input_set, "approx", what, **options)
    assert_covered(sets, replay_torch(model, samples(lo, hi, N_SAMPLES),
                                      in_shape), what,
                   out_chw=tuple(out_shape[1:]) if kind == "ImageStar"
                   else None)


@pytest.mark.parametrize("op", ["Sub"])
def test_operands_whose_systems_differ_only_in_their_tags(tmp_path, op):
    """Sin(x) - Cos(x) on a Star over x in [0, 7]. Over a range that holds an
    extremum of both, each operation appends one box-only predicate variable
    in [-1, 1], so the two operands' systems have equal rows and lower bounds
    and differ only in the upper bound of each node's tag variable; they are
    different functions and must not share variables. Proved by interval
    branch and bound and replayed in float64 (onnxruntime has no float64
    Cos). Catches a same-system test that compares only part of the two
    systems (the rows, the lower bounds, one operand's bounds with
    themselves), and a graph walk that tags no node's new variables, either
    of which reads Sin(x) - Cos(x) as a single point. Only the difference
    sees that merge: on shared variables Sin(x) + Cos(x) reads as 2 z with z
    in [-1, 1], which still holds the true sum, so the Add item was removed.
    Sin and Cos are other
    kernels, used here only to build the two operands."""
    nodes = [helper.make_node("Sin", ["X"], ["A"]),
             helper.make_node("Cos", ["X"], ["B"]),
             helper.make_node(op, ["A", "B"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 1], [1, 1])
    lo, hi = np.array([0.0]), np.array([7.0])
    input_set = flat_set("Star", lo, hi)
    what = f"Sin(x) {op} Cos(x) on a Star"
    model = load_onnx(path)
    sets = _reach(model, input_set, "approx", what)
    c, G, C, d, lb, ub = rational.star(input_set)
    sign = 1 if op == "Add" else -1
    result = interval_proof.prove(
        interval_proof.chain(interval_proof.linear(G, c),
                             interval_proof.linear([[1], [1]]),
                             _sin_then_cos,
                             interval_proof.linear([[1, sign]])),
        lb, ub, *[rational.star(s) for s in sets],
        domain=(C, d) if C else None)
    assert result.proved, f"{what}: {result}"
    assert_covered(sets, replay_torch(model, samples(lo, hi, N_POINTS),
                                      [1, 1]), what)


# --------------------------------------------------------------------------- #
# The exact method: operands that are unions of pieces                       #
# --------------------------------------------------------------------------- #
# X @ _WL + _BL on R1 = Relu(A): each neuron crosses zero on the range of R1.
_WL = np.array([[1.0, -1.0, 0.5], [-0.5, 0.25, 1.0], [0.0, 0.0, 0.0]])
_BL = np.array([0.75, 0.5, -2.0])


@pytest.mark.parametrize("pairing", ["one-split", "two-splits", "lineage"])
@pytest.mark.parametrize("op", ["Add", "Sub"])
def test_exact_splits_are_paired_to_cover_every_input(tmp_path, op, pairing):
    """method='exact' splits each ReLU into pieces, so an operand is a list
    of Stars. "one-split": Relu(A) op B, several pieces against one set;
    "two-splits": Relu(A) op Relu(B), two lists split independently, every
    combination needed; "lineage": R1 op Relu(R1 WL + bL), the pieces of the
    second operand descend from those of the first and must meet their own
    ancestor. Checked against every exact piece of the network (each must lie
    in one returned set) and replayed. Catches a pairing that combines
    pieces index by index, takes the shortest list, or indexes the lineage
    groups with the wrong position, all of which drop the inputs of the
    combinations they never form. The joins under method='exact' are not
    the exact image (REACH-4, marked in
    :func:`test_exact_add_or_sub_is_the_exact_image`); only enclosure is
    checked here. When PLAN step 18 makes method='exact' refuse these joins,
    this cell must move to the approximate method with
    split_piecewise_linear=True, which keeps today's pairing; left as it is
    it goes red and the pairing code loses its only cell."""
    nodes, inits = [], {}
    r1 = _branch(nodes, inits, "A", "X", _WA, _BA, True, np.float32)
    pieces = rational.affine(_predicate_pieces(flat_set("Star", _LO, _HI)),
                             _WA.T.tolist(), _BA.tolist())
    if pairing == "lineage":
        second = _branch(nodes, inits, "L", r1, _WL, _BL, True, np.float32)
        pieces = rational.relu(pieces)
        pieces = rational.affine(pieces, _eye(3) + _WL.T.tolist(),
                                 [0, 0, 0] + _BL.tolist())
        pieces = _relu_on(pieces, [3, 4, 5])
    else:
        relu_b = pairing == "two-splits"
        second = _branch(nodes, inits, "B", "X", _WB, _BB, relu_b,
                         np.float32)
        pieces = rational.affine(_predicate_pieces(flat_set("Star", _LO,
                                                            _HI)),
                                 np.vstack([_WA.T, _WB.T]).tolist(),
                                 np.concatenate([_BA, _BB]).tolist())
        pieces = _relu_on(pieces, [0, 1, 2] + ([3, 4, 5] if relu_b else []))
    nodes.append(helper.make_node(op, [r1, second], ["Y"]))
    pieces = rational.affine(pieces, _combine(op))
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 3], inits=inits)
    what = f"exact {op} of operands with {pairing}"
    sets = _reach(load_onnx(path), flat_set("Star", _LO, _HI), "exact",
                       what)
    if len(sets) < 2:
        pytest.fail(f"{what}: {len(sets)} set came back; the cell needs an "
                    f"operand split into several pieces")
    _assert_pieces_held(pieces, sets, what)
    assert_covered(sets, replay_onnx(path, samples(_LO, _HI, N_POINTS),
                                     [1, 2]), what)


# --------------------------------------------------------------------------- #
# Concat of operands with three different predicate systems                  #
# --------------------------------------------------------------------------- #
_CONCAT_ORDERS = {"affine-first": "ABC", "affine-middle": "CAB",
                  "affine-last": "BCA"}


@pytest.mark.parametrize("kind, method, order", [
    ("Star+row", "approx", "affine-first"), ("Star+row", "approx",
                                             "affine-last"),
    ("Star", "exact", "affine-middle")])
def test_concat_of_three_predicate_systems(tmp_path, kind, method, order):
    """Concat(A, Relu(B), Relu(C)) in three orders: under the approximate
    method A keeps the input's predicate system and each ReLU branch has one
    of its own, so three systems are joined block-diagonally; under the
    exact method two operands are lists of pieces and one is a single set.
    Catches a block offset that is right for two systems only (the third
    system's basis and rows written over the second's variables), a basis
    written to the leading columns instead of its own block, and a pairing
    that drops the single operand or a combination of pieces."""
    nodes, inits = [], {}
    names = {"A": _branch(nodes, inits, "A", "X", _WA, _BA, False,
                          np.float32),
             "B": _branch(nodes, inits, "B", "X", _WB, _BB, True,
                          np.float32),
             "C": _branch(nodes, inits, "C", "X", _WC, _BC, True,
                          np.float32)}
    sequence = _CONCAT_ORDERS[order]
    nodes.append(helper.make_node("Concat", [names[k] for k in sequence],
                                  ["Y"], axis=1))
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 9], inits=inits)
    input_set, points = _input(kind)
    what = f"{method} Concat in the order {sequence} on a {kind}"
    sets = _reach(load_onnx(path), input_set, method, what)
    block = {"A": 0, "B": 3, "C": 6}
    pieces = rational.affine(_predicate_pieces(input_set),
                             np.vstack([_WA.T, _WB.T, _WC.T]).tolist(),
                             np.concatenate([_BA, _BB, _BC]).tolist())
    pieces = _relu_on(pieces, list(range(3, 9)))
    pieces = rational.affine(pieces, [
        [int(j == block[k] + i) for j in range(9)]
        for k in sequence for i in range(3)])
    _assert_pieces_held(pieces, sets, what)
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)


# --------------------------------------------------------------------------- #
# Mul and Div of two flat branches: the containment proof                    #
# --------------------------------------------------------------------------- #
def _spread(form, radii):
    """The interval of ``form - centre`` over the box."""
    reach = sum(abs(s) * r for s, r in zip(form.slopes, radii))
    return -reach + form.low, reach + form.high


def _times(a, b):
    corners = [x * y for x in a for y in b]
    return min(corners), max(corners)


def _product(box, f, g):
    """The form of ``f * g``: the linear part ``cf cg + (cf sg + cg sf) dx``
    and the remainder ``cf eg + cg ef + (sf dx + ef)(sg dx + eg)``, enclosed
    in rationals; the interval is the product of the two intervals."""
    radii = [(hi - lo) / 2 for lo, hi in box]
    parts = [_times((f.centre, f.centre), (g.low, g.high)),
             _times((g.centre, g.centre), (f.low, f.high)),
             _times(_spread(f, radii), _spread(g, radii))]
    return interval_proof.Form(
        f.centre * g.centre,
        [f.centre * sg + g.centre * sf for sf, sg in zip(f.slopes, g.slopes)],
        sum(p[0] for p in parts), sum(p[1] for p in parts),
        _times(f.span, g.span))


_RECIPROCAL = interval_proof.smooth(lambda t: 1 / t, lambda t: -1 / (t * t))


def _relu_forms(box, forms):
    """ReLU of each form: unchanged when it is non-negative on the box, zero
    when non-positive, otherwise the interval [0, its upper end]."""
    radii = [(hi - lo) / 2 for lo, hi in box]
    flat = [Fraction(0)] * len(box)
    out = []
    for f in forms:
        lo, hi = f.bounds(radii)
        if lo >= 0:
            out.append(f)
        elif hi <= 0:
            out.append(interval_proof.Form(Fraction(0), flat, Fraction(0),
                                           Fraction(0), (Fraction(0),
                                                         Fraction(0))))
        else:
            out.append(interval_proof.Form(Fraction(0), flat, Fraction(0), hi,
                                           (Fraction(0), hi)))
    return out


def _shifted(forms, shift):
    """Each form plus the constant ``shift``."""
    return [interval_proof.Form(f.centre + shift, f.slopes, f.low, f.high,
                                (f.span[0] + shift, f.span[1] + shift))
            for f in forms]


def _bilinear(op, relu_a, relu_b, swap=False, shift_b=None):
    """The stage [A; B] -> A * B (or A / B) element by element, with the
    ReLUs and the shift of B of :func:`_two_branch_model` applied first."""
    def stage(box, forms):
        half = len(forms) // 2
        a, b = forms[:half], forms[half:]
        if relu_a:
            a = _relu_forms(box, a)
        if relu_b:
            b = _relu_forms(box, b)
        if shift_b is not None:
            b = _shifted(b, rational.frac(shift_b))
        if swap:
            a, b = b, a
        if op == "Div":
            b = _RECIPROCAL(box, b)
        return [_product(box, f, g) for f, g in zip(a, b)]
    return stage


def _bilinear_exact(points, op, relu_a, relu_b, ba, bb, cols, swap,
                    shift_b=None):
    """The exact outputs (rationals) of the bilinear model at ``points``
    (``ba``, ``bb`` and ``shift_b`` as the model holds them, scaled)."""
    cols = list(cols)
    W = np.vstack([_WA[:, cols].T, _WB[:, cols].T])
    bias = np.concatenate([ba[cols], bb[cols]])
    tail = _tail(len(cols)).T
    exact = []
    for x in points:
        h = [sum((rational.frac(w) * rational.frac(v)
                  for w, v in zip(row, x)), Fraction(0)) + rational.frac(c)
             for row, c in zip(W, bias)]
        a, b = h[:len(cols)], h[len(cols):]
        if relu_a:
            a = [max(v, Fraction(0)) for v in a]
        if relu_b:
            b = [max(v, Fraction(0)) for v in b]
        if shift_b is not None:
            b = [v + rational.frac(shift_b) for v in b]
        if swap:
            a, b = b, a
        q = [u * v if op == "Mul" else u / v for u, v in zip(a, b)]
        exact.append([sum((rational.frac(t) * v for t, v in zip(row, q)),
                          Fraction(0)) for row in tail])
    return exact


def _bilinear_cell(tmp_path, kind, op, relu_a, relu_b, ba, bb, cols, swap,
                   what, scale=1, shift_b=None, options=None,
                   exact_points=None):
    """Reach the bilinear model on ``kind`` and check it: the containment
    proof over the input's predicate polytope for the Star with a binding
    row, the Zonos and the Box; for the plain Star, which takes the same
    code path with box rows only, the exact output of every sample point
    decided by exact membership (``exact_points`` chooses the second check
    for any kind). Then the onnxruntime replay. ``scale`` multiplies the
    input box and the model's biases (:func:`_two_branch_model`),
    ``shift_b`` moves the B branch after its ReLU, ``options`` go to the
    reach."""
    cols = list(cols)
    path = _two_branch_model(tmp_path, op, relu_a, relu_b, bb=bb, swap=swap,
                             ba=ba, cols=cols, scale=scale, shift_b=shift_b)
    if kind in ("Star+vars", "Zono+gens"):
        input_set, points = _skewed_input(kind, scale)
    else:
        input_set, points = _input(kind, scale=scale)
    sets = _reach(load_onnx(path), input_set, "approx", what,
                  **(options or {}))
    ba, bb = scale * ba, scale * bb
    shift_b = None if shift_b is None else scale * shift_b
    W = np.vstack([_WA[:, cols].T, _WB[:, cols].T])
    bias = np.concatenate([ba[cols], bb[cols]])
    tail = _tail(len(cols)).T
    if exact_points is None:
        exact_points = kind == "Star"
    if exact_points:
        exact = _bilinear_exact(points, op, relu_a, relu_b, ba, bb, cols,
                                swap, shift_b)
        _assert_points_held(sets, exact, what)
    else:
        c, G, C, d, lb, ub = rational.star(input_set)
        result = interval_proof.prove(
            interval_proof.chain(interval_proof.linear(G, c),
                                 interval_proof.linear(W.tolist(),
                                                       bias.tolist()),
                                 _bilinear(op, relu_a, relu_b, swap, shift_b),
                                 interval_proof.linear(tail.tolist())),
            lb, ub, *[rational.star(s) for s in sets],
            domain=(C, d) if C else None)
        assert result.proved, f"{what}: {result}"
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)


_MUL_REGIMES = ("affine-affine", "relu-affine", "affine-relu")


@pytest.mark.parametrize("regime", _MUL_REGIMES)
@pytest.mark.parametrize("kind", _FLAT_KINDS)
def test_mul_of_two_branches_is_proved(tmp_path, kind, regime):
    """A * B on neurons 0 and 2 (crossing times positive, negative times
    crossing), then the affine tail. Both products need all four corner
    products; on a Star the McCormick rows couple the product with the
    operands through one shared predicate system or, with a ReLU on either
    branch, two joined ones. Catches an interval product from two corners
    only, a maximum taken as a minimum, the second operand's bounds read
    from the first, an envelope row with a wrong sign (the z box unchanged),
    the second operand's basis padded on the wrong side of the joined
    system, and the first operand's ranges used for the second."""
    relu_a, relu_b = _REGIMES[regime]
    _bilinear_cell(tmp_path, kind, "Mul", relu_a, relu_b, _BA, _BB, (0, 2),
                   False, f"Mul of two branches ({regime}) on a {kind}")


# numerator: (ReLU on it, its bias); the divisor is the B branch.
_NUMERATORS = {"crossing": (False, _BA), "relu": (True, _BA),
               "away": (False, _BA_AWAY)}


@pytest.mark.parametrize("order", ["A/B", "B/A"])
@pytest.mark.parametrize("numerator", sorted(_NUMERATORS))
@pytest.mark.parametrize("kind", ["Star", "Star+row", "Box"])
def test_div_of_two_branches_is_proved_or_refused(tmp_path, kind, numerator,
                                                  order):
    """A / B on neurons 0 and 1, then the affine tail, with the divisor B
    positive in one entry and negative in the other, and A crossing zero, a
    ReLU of it, or away from zero; "B/A" divides by A. A quotient whose
    divisor range contains zero (A crossing, a ReLU of A) must be refused;
    every other quotient is checked. Catches a quotient box from two corners
    only, the operands passed to the quotient in swapped order, the joined
    system built with the operands swapped while the bases stay
    numerator-first, a divisor basis padded without its centre, an envelope
    row with a wrong sign, and an answer where the divisor reaches zero."""
    relu_a, ba = _NUMERATORS[numerator]
    what = f"Div {order} with A {numerator} on a {kind}"
    swap = order == "B/A"
    if swap and numerator != "away":
        path = _two_branch_model(tmp_path, "Div", relu_a, False, bb=_BB_DIV,
                                 swap=True, ba=ba, cols=(0, 1))
        with pytest.raises(NotImplementedError, match="straddles zero"):
            _reach(load_onnx(path), _input(kind)[0], "approx", what)
        return
    _bilinear_cell(tmp_path, kind, "Div", relu_a, False, ba, _BB_DIV, (0, 1),
                   swap, what)


# A second operand whose neurons have the signs of the first's, so that A * B
# has both operands crossing zero (B0 in [-9/8, 9/8]), both positive (B1 in
# [7/8, 3]) and both negative (B2 in [-15/4, -1/2]); with _BB the pairs are
# (crossing, positive), (positive, negative) and (negative, crossing).
_BB_SAME = np.array([-0.125, 2.0, -2.0])
# A numerator below zero on every neuron: A0 in [-17/4, -13/8], A1 in
# [-43/16, -11/8], A2 in [-7/2, -19/8]; over _BB_DIV the quotients on
# neurons 0 and 1 are negative over positive and negative over negative.
_BA_NEG = np.array([-3.0, -2.0, -3.0])
# signs: (bias of A, bias of B, neurons)
_SIGNS = {"mixed": (_BA, _BB, (0, 1, 2)), "same": (_BA, _BB_SAME, (0, 1, 2)),
          "negative": (_BA_NEG, _BB_DIV, (0, 1))}


@pytest.mark.parametrize("op, kind, signs, regime", [
    ("Mul", "Star", "mixed", "affine-affine"),
    ("Mul", "Star", "same", "relu-relu"),
    ("Mul", "Star+vars", "same", "affine-affine"),
    ("Mul", "Zono+gens", "mixed", "affine-affine"),
    ("Mul", "Zono+gens", "same", "affine-affine"),
    ("Mul", "Box", "mixed", "affine-affine"),
    ("Mul", "Box", "same", "affine-affine"),
    ("Div", "Star", "negative", "affine-affine"),
    ("Div", "Star+vars", "negative", "affine-affine"),
    ("Div", "Box", "negative", "affine-affine")])
def test_product_or_quotient_on_a_wide_box(tmp_path, op, kind, signs, regime):
    """A * B and A / B on the box times 8 (x0 in [-8, 8], x1 in [-4, 6]) with
    the biases times 8, then the affine tail, with operands of every sign
    pattern the cells above lack: "same" pairs neurons that both cross
    zero, are both positive and both negative, "mixed" adds a positive
    times a negative operand (with the cells above: all nine pairs),
    "negative" divides a negative numerator by a positive and by a negative
    divisor. ``Star+vars`` and ``Zono+gens`` give the box four predicate
    variables or generators on two entries (the Star cut by a binding row,
    :func:`_skewed_input`), so every operand has more variables than
    entries and its values reach 26 and its products 130. The relu-relu item
    puts a ReLU on both branches: each appends one relaxation variable for
    its crossing neuron 0, so the two systems have equal variable counts and
    differ (with neuron 1 positive in both, the second operand's basis
    carries the input variables too, which a merged system misreads). The
    Box is proved by interval branch and bound; on the Stars and Zonos the
    exact output of every sample point (corners of the predicate box first)
    is decided by exact membership, because the proof over four predicate
    variables takes from seconds to over a minute an item (the unit-box
    cells above prove the Zono product). All are replayed. Catches a
    product or quotient box built from the corners that are right only when
    an operand crosses zero (or only for one sign), two systems shared
    because their variable counts agree, and a kernel that reads one
    variable, generator or bound in place of another, which data inside
    [-1, 1] and square bases hide."""
    ba, bb, cols = _SIGNS[signs]
    relu_a, relu_b = _REGIMES[regime]
    what = f"{op} ({signs} signs, {regime}) on a {kind} times {_WIDE}"
    _bilinear_cell(tmp_path, kind, op, relu_a, relu_b, ba, bb, cols, False,
                   what, scale=_WIDE, exact_points=kind != "Box")


@pytest.mark.parametrize("op, kind", [("Mul", "Star"), ("Mul", "Zono"),
                                      ("Mul", "Box"), ("Div", "Star"),
                                      ("Div", "Box")])
def test_product_or_quotient_by_a_shifted_relu_branch(tmp_path, op, kind):
    """A op (Relu(B) + 1/2) on neurons 1 and 2, then the affine tail. B1 is
    negative, so its ReLU is the constant 0 and the second operand's entry 1
    the constant 1/2, an entry with no generator whose operand range the
    kernel keeps exact; B2 crosses zero, so its ReLU appends a relaxation
    variable and the second operand (the divisor of the quotient, in
    [1/2, 17/8]) is on a system of its own, joined behind the first
    operand's. The Zono and the Box are proved by interval branch and bound;
    the Star is checked by the exact outputs of its sample points, because
    the McCormick rows of the constant entry are the exact line z = 2 x
    (z = x / 2 for the product), which no interval enclosure fits. All are
    replayed. Catches operand ranges padded or kept exact by the other
    operand's basis (a first-operand entry taken as constant because its
    partner is), and a divisor basis padded on the wrong side of the joined
    system, which only a second operand with variables of its own can show
    (the ReLU numerator items above join the other way round)."""
    what = f"{op}(A, Relu(B) + 1/2) on a {kind}"
    _bilinear_cell(tmp_path, kind, op, False, True, _BA, _BB, (1, 2), False,
                   what, shift_b=0.5)


# One node read twice, or one operand computed from the other: H = X WA + b
# on neurons 0 and 2 and R = Relu(H) (H0 crosses zero, H2 is negative; with
# _BA_AWAY, for the quotient, H0 is positive and H2 negative).
_SELF = {"h*h": ("Mul", "H", "H", _BA), "h*relu(h)": ("Mul", "H", "R", _BA),
         "h/h": ("Div", "H", "H", _BA_AWAY)}


_SQUARE = interval_proof.power(2)


def _self_stage(case):
    """The stage H -> H * H, H * Relu(H) or H / H element by element. The
    square is enclosed as a power (the product of two forms would treat the
    factors as independent) and H / H, whose divisor is away from zero, as
    the constant 1: a product of H and 1 / H is too loose near the line
    z = 1 for a sound result that hugs it to be proved."""
    def stage(box, forms):
        if case == "h*h":
            return _SQUARE(box, forms)
        if case == "h/h":
            one = Fraction(1)
            return [interval_proof.Form(one, [Fraction(0)] * len(box),
                                        Fraction(0), Fraction(0), (one, one))
                    for _ in forms]
        others = _relu_forms(box, forms)
        return [_product(box, f, g) for f, g in zip(forms, others)]
    return stage


@pytest.mark.parametrize("case, kind", [
    ("h*h", "Star+row"), ("h*h", "Zono"), ("h*h", "Box"),
    ("h*relu(h)", "Star+row"), ("h/h", "Star+row"), ("h/h", "Box")])
def test_product_or_quotient_of_a_node_with_itself(tmp_path, case, kind):
    """H * H (``x * x``: one node feeds both operands), H * Relu(H) (the
    second operand's system extends the first's by the ReLU's relaxation
    variable, so they are joined block-diagonally) and H / H (true output 1)
    on neurons 0 and 2, then the affine tail. Proved by interval branch and
    bound over the input's predicate polytope and replayed. Catches a
    product of one set with itself that is answered as that set or its
    double, operands of one node paired with the wrong variables, and a
    kernel that changes the arrays of an operand it reads twice (the reach
    helper checks that the input set comes back unchanged)."""
    op, first, second, bias = _SELF[case]
    cols = [0, 2]
    dtype = np.float64 if op == "Div" else np.float32
    nodes, inits = [], {}
    _branch(nodes, inits, "", "X", _WA[:, cols], bias[cols], True, dtype)
    nodes += [helper.make_node(op, [first, second], ["Q"]),
              helper.make_node("MatMul", ["Q", "T"], ["Y"])]
    inits["T"] = _TAIL2.astype(dtype)
    kind_name = "DOUBLE" if op == "Div" else "FLOAT"
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 2], inits=inits,
                     in_type=kind_name, out_type=kind_name)
    input_set, points = _input(kind)
    what = f"{case} on a {kind}"
    sets = _reach(load_onnx(path), input_set, "approx", what)
    c, G, C, d, lb, ub = rational.star(input_set)
    result = interval_proof.prove(
        interval_proof.chain(interval_proof.linear(G, c),
                             interval_proof.linear(_WA[:, cols].T.tolist(),
                                                   bias[cols].tolist()),
                             _self_stage(case),
                             interval_proof.linear(_TAIL2.T.tolist())),
        lb, ub, *[rational.star(s) for s in sets],
        domain=(C, d) if C else None)
    assert result.proved, f"{what}: {result}"
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)


@pytest.mark.parametrize("op", ["Mul", "Div"])
def test_mccormick_ranges_come_from_highs_whatever_the_lp_solver(tmp_path,
                                                                 op):
    """A * Relu(B) on neurons 0 and 2 and Relu(A) / B on neurons 0 and 1 (the
    affine-relu product and the ReLU-numerator quotient above) reached with
    ``lp_solver='SCS'``. The operand ranges
    of the McCormick envelopes are LP optima used as bounds, and their
    outward pad is sized for HiGHS; SCS stops up to about 1e-5 inside the
    true optimum. The exact outputs of the sample points of a Star must be
    members of the result, and replayed outputs too. Catches operand ranges
    taken with the caller's solver instead of HiGHS
    (``_bound_lp_solver``), which leaves out true products by about 1e-5."""
    what = f"{op} with lp_solver='SCS' on a Star"
    options = {"lp_solver": "SCS"}
    if op == "Mul":
        _bilinear_cell(tmp_path, "Star", op, False, True, _BA, _BB, (0, 2),
                       False, what, options=options)
    else:
        _bilinear_cell(tmp_path, "Star", op, True, False, _BA, _BB_DIV,
                       (0, 1), False, what, options=options)


_REACH_4_PRODUCT = known_bug(
    18, "REACH-4: method='exact' answers the product or quotient of two "
        "computed sets with the McCormick relaxation and returns it as "
        "exact, where the exact method must refuse an operation it has no "
        "exact rule for", raises=DID_NOT_RAISE)


@_REACH_4_PRODUCT
@pytest.mark.parametrize("op", ["Mul", "Div"])
def test_exact_product_or_quotient_is_refused(tmp_path, op):
    """method='exact' on A * B or A / B of two affine branches of a Star (B
    away from zero for the quotient). A product of two computed values has
    no exact rule in n2v (the image is curved; the kernel is a McCormick
    relaxation), so the exact method must raise ``NotImplementedError``
    (PLAN step 18); today it answers with the relaxation, the strict
    expected failure REACH-4. An answer must still hold every true output:
    the exact outputs of the sample points and the replay are checked
    first with plain assertions, which the marker (it absorbs only the
    missing refusal) does not absorb, and so is the reach helper's check
    that the input set came back unchanged. Catches an exact reach that
    answers a product or quotient where it must refuse, and one that
    answers unsoundly."""
    cols, bb = ((0, 2), _BB) if op == "Mul" else ((0, 1), _BB_DIV)
    path = _two_branch_model(tmp_path, op, False, False, bb=bb, cols=cols)
    points = samples(_LO, _HI, N_POINTS)
    what = f"exact {op} of two affine branches on a Star"
    try:
        sets = reach_model(load_onnx(path), flat_set("Star", _LO, _HI),
                           "exact", what)
    except NotImplementedError:
        return
    except pytest.fail.Exception as error:     # the input set was changed
        raise AssertionError(str(error)) from None
    _assert_points_held(sets, _bilinear_exact(points, op, False, False, _BA,
                                              bb, cols, False), what)
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)
    pytest.fail(f"{what}: answered with {len(sets)} set(s) where the exact "
                f"method must refuse")


@pytest.mark.parametrize("op", ["Mul", "Div"])
def test_exact_split_operand_of_a_product_or_quotient(tmp_path, op):
    """method='exact' with a ReLU on the first operand: Relu(A) * B or
    Relu(A) / B (B away from zero for the quotient), then the affine tail.
    The first operand is a list of pieces and each must be combined with the
    single second operand. The exact outputs of the sample points are decided
    by exact membership in the union of the returned Stars, and replayed.
    Catches a product or quotient that combines the lists index by index or
    passes on only the first piece. The result is not the exact image under
    method='exact' (REACH-4, marked by
    :func:`test_exact_product_or_quotient_is_refused`); only enclosure is
    checked. When PLAN step 18 makes method='exact' refuse a product or
    quotient, this cell must move to the approximate method with
    split_piecewise_linear=True, which keeps today's pairing; left as it is
    it goes red and the pairing of split operands loses its only Mul and
    Div cell."""
    cols, bb = ((0, 2), _BB) if op == "Mul" else ((0, 1), _BB_DIV)
    path = _two_branch_model(tmp_path, op, True, False, bb=bb, cols=cols)
    points = samples(_LO, _HI, N_POINTS)
    what = f"exact {op} of a split operand and a single set"
    sets = _reach(load_onnx(path), flat_set("Star", _LO, _HI), "exact",
                       what)
    if len(sets) < 2:
        pytest.fail(f"{what}: {len(sets)} set came back; the cell needs an "
                    f"operand split into several pieces")
    _assert_points_held(sets, _bilinear_exact(points, op, True, False, _BA,
                                              bb, cols, False), what)
    assert_covered(sets, replay_onnx(path, points, [1, 2]), what)


# --------------------------------------------------------------------------- #
# Broadcasting between two computed sets of different tensor shapes          #
# --------------------------------------------------------------------------- #
def _corner_images(input_set, W, b):
    """The images ``W x + b`` (rational) of the vertices of a box-shaped
    input set (a Box, a Zono, or a Star whose rows are box rows only): the
    corners of its predicate box mapped through its basis. The image of the
    set under an affine map is the hull of these points."""
    c, G, C, d, lb, ub = rational.star(input_set)
    for row, rhs in zip(C, d):
        nonzero = [j for j, v in enumerate(row) if v != 0]
        j = nonzero[0] if len(nonzero) == 1 else None
        if j is None or not (rhs == (ub[j] if row[j] == 1 else -lb[j])
                             and abs(row[j]) == 1):
            pytest.fail("the input set has a row that is not a box row")
    W = rational.matrix(W)
    b = rational.vector(b)
    out = []
    for a in itertools.product(*zip(lb, ub)):
        x = [ci + rational.dot(g, a) for ci, g in zip(c, G)]
        out.append([rational.dot(row, x) + bi for row, bi in zip(W, b)])
    return out


def _sum_keepdims_matrix(shape, axis):
    """The 0/1 matrix of ReduceSum over ``axis`` with keepdims followed by the
    ONNX broadcast back to ``shape``: entry (out, in) is 1 when the input
    entry is summed into the output entry's broadcast position."""
    index = np.arange(int(np.prod(shape))).reshape(shape)
    M = np.zeros((index.size, index.size))
    for out in np.ndindex(*shape):
        line = list(out)
        line[axis] = slice(None)
        M[index[out], index[tuple(line)].reshape(-1)] = 1
    return M


_BROADCAST_SHAPE = (1, 2, 3)


@pytest.mark.parametrize("operands", ["X op S", "S op X"])
@pytest.mark.parametrize("axis", [1, 2])
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_sub_of_sets_of_different_tensor_shapes(tmp_path, kind, axis,
                                                operands):
    """X - S and S - X with S = ReduceSum(X, axis, keepdims) on a (1, 2, 3)
    tensor given with its shape: S is (1, 1, 3) or (1, 2, 1) and is broadcast
    to (1, 2, 3) by selecting its rows. Every vertex of the input maps to an
    exact output that must be a member. Catches an operand expanded by
    tiling its flat rows, which is right only when the broadcast axis is the
    leading one (axis 1 here) and repeats the wrong entries along axis 2."""
    first, second = ("X", "S") if operands == "X op S" else ("S", "X")
    nodes = [helper.make_node("ReduceSum", ["X"], ["S"], axes=[axis],
                              keepdims=1),
             helper.make_node("Sub", [first, second], ["Y"])]
    path = save_onnx(tmp_path, nodes, _BROADCAST_SHAPE, _BROADCAST_SHAPE,
                     opset=11)
    n = int(np.prod(_BROADCAST_SHAPE))
    # A different interval per entry: straddling, positive and negative.
    lo = np.array([-0.5, 0.25, -1.0, 0.75, -0.25, 1.0])
    hi = np.array([0.5, 0.75, -0.5, 1.5, 0.5, 1.25])
    input_set = flat_set(kind, lo, hi)
    what = f"Sub({first}, {second}) with S summed over axis {axis} on a {kind}"
    sets = _reach(load_onnx(path), input_set, "approx", what,
                       input_shape=_BROADCAST_SHAPE[1:])
    S = _sum_keepdims_matrix(_BROADCAST_SHAPE, axis)
    W = np.eye(n) - S if operands == "X op S" else S - np.eye(n)
    _assert_points_held(sets, _corner_images(input_set, W, np.zeros(n)), what)
    assert_covered(sets, replay_onnx(path, samples(lo, hi, N_POINTS),
                                     _BROADCAST_SHAPE), what)


@pytest.mark.parametrize("kind, op", [("Star", "Add"), ("Zono", "Sub"),
                                      ("Box", "Add")])
def test_add_or_sub_of_two_broadcast_operands(tmp_path, kind, op):
    """ReduceSum(X, axis 2) op ReduceSum(X, axis 1) on a (1, 2, 3) tensor
    given with its shape: (1, 2, 1) op (1, 1, 3), so both operands are
    expanded to (1, 2, 3), which neither has. The box is the one of the Sub
    cell above times 8.
    Every vertex of the input maps to an exact output that must be a
    member, and replayed outputs too. Catches a broadcast that expands only
    one operand or expands one against the other's shape instead of the
    output's, and a row map built from the wrong operand's shape."""
    nodes = [helper.make_node("ReduceSum", ["X"], ["S2"], axes=[2],
                              keepdims=1),
             helper.make_node("ReduceSum", ["X"], ["S1"], axes=[1],
                              keepdims=1),
             helper.make_node(op, ["S2", "S1"], ["Y"])]
    path = save_onnx(tmp_path, nodes, _BROADCAST_SHAPE, _BROADCAST_SHAPE,
                     opset=11)
    n = int(np.prod(_BROADCAST_SHAPE))
    lo = _WIDE * np.array([-0.5, 0.25, -1.0, 0.75, -0.25, 1.0])
    hi = _WIDE * np.array([0.5, 0.75, -0.5, 1.5, 0.5, 1.25])
    input_set = flat_set(kind, lo, hi)
    what = f"{op}(S2, S1) of two broadcast sums on a {kind}"
    sets = _reach(load_onnx(path), input_set, "approx", what,
                  input_shape=_BROADCAST_SHAPE[1:])
    S2 = _sum_keepdims_matrix(_BROADCAST_SHAPE, 2)
    S1 = _sum_keepdims_matrix(_BROADCAST_SHAPE, 1)
    W = S2 + S1 if op == "Add" else S2 - S1
    _assert_points_held(sets, _corner_images(input_set, W, np.zeros(n)), what)
    assert_covered(sets, replay_onnx(path, samples(lo, hi, N_POINTS),
                                     _BROADCAST_SHAPE), what)


@pytest.mark.parametrize("op, axis", [("Mul", 2), ("Div", 1), ("Div", 2),
                                      ("Div", "both")])
def test_mul_and_div_of_sets_of_different_tensor_shapes(tmp_path, op, axis):
    """X * S and X / S on a Star with S = ReduceSum(X, axis, keepdims) on a
    positive (1, 2, 3) box: S is (1, 1, 3) for axis 1 and (1, 2, 1) for
    axis 2, and the broadcast operand enters the McCormick product or
    quotient. "both" is ReduceSum(X, axis 2) op ReduceSum(X, axis 1),
    (1, 2, 1) op (1, 1, 3), so both operands are expanded to (1, 2, 3)
    (the quotient, whose operands are not
    interchangeable). The exact outputs of the first 12 sample points (the
    two extreme corners only for "both", whose result has twice the
    variables) are decided by exact membership, and 48 are replayed. Catches an operand expanded by tiling
    its flat rows, which is right only along the leading axis (axis 1) and
    repeats the wrong entries along axis 2, and a broadcast that expands one
    operand
    only, on the path through the bilinear kernels."""
    if axis == "both":
        nodes = [helper.make_node("ReduceSum", ["X"], ["S2"], axes=[2],
                                  keepdims=1),
                 helper.make_node("ReduceSum", ["X"], ["S1"], axes=[1],
                                  keepdims=1),
                 helper.make_node(op, ["S2", "S1"], ["Y"])]
        first = _sum_keepdims_matrix(_BROADCAST_SHAPE, 2)
        second = _sum_keepdims_matrix(_BROADCAST_SHAPE, 1)
    else:
        nodes = [helper.make_node("ReduceSum", ["X"], ["S"], axes=[axis],
                                  keepdims=1),
                 helper.make_node(op, ["X", "S"], ["Y"])]
        first = np.eye(int(np.prod(_BROADCAST_SHAPE)))
        second = _sum_keepdims_matrix(_BROADCAST_SHAPE, axis)
    in_type = "DOUBLE" if op == "Div" else "FLOAT"
    path = save_onnx(tmp_path, nodes, _BROADCAST_SHAPE, _BROADCAST_SHAPE,
                     opset=11, in_type=in_type, out_type=in_type)
    lo = np.array([0.5, 0.25, 1.0, 0.75, 0.25, 1.0])
    hi = lo + np.array([0.5, 0.25, 0.5, 0.75, 0.25, 0.5])
    what = f"{op} with the operands broadcast along axis {axis} on a Star"
    sets = _reach(load_onnx(path), flat_set("Star", lo, hi), "approx",
                  what, input_shape=_BROADCAST_SHAPE[1:])
    points = samples(lo, hi, N_POINTS)
    exact = []
    for x in points[:2 if axis == "both" else 12]:
        x = rational.vector(x)
        a = [rational.dot(row, x) for row in rational.matrix(first)]
        s = [rational.dot(row, x) for row in rational.matrix(second)]
        exact.append([u * v if op == "Mul" else u / v for u, v in zip(a, s)])
    _assert_points_held(sets, exact, what)
    assert_covered(sets, replay_onnx(path, points, _BROADCAST_SHAPE), what)


# --------------------------------------------------------------------------- #
# Image sets: two convolution branches                                       #
# --------------------------------------------------------------------------- #
_IMAGE = (2, 3, 5)
# Conv weights (out, in, kh, kw), dyadic: A has a (1, 3) kernel, B a (3, 1)
# kernel, both padded to keep the (3, 5) plane.
_KA = np.array([[[[0.5, -0.25, 1.0]], [[0.25, 0.5, -0.5]]],
                [[[-0.5, 1.0, 0.25]], [[0.75, -0.25, 0.5]]]])
_CA = np.array([0.25, -0.5])
_KB = np.array([[[[1.0], [-0.5], [0.25]], [[0.5], [0.25], [-1.0]]],
                [[[-0.25], [0.5], [0.75]], [[1.0], [-0.5], [0.5]]]])
_CB = np.array([-0.25, 0.5])
_PADS_A, _PADS_B = [0, 1, 0, 1], [1, 0, 1, 0]
# A 1x1 tail that maps channel 0 to -1 times channel 1 and channel 1 to 2
# times channel 0: invertible and exact on every float.
_KT = np.array([[[[0.0]], [[2.0]]], [[[-1.0]], [[0.0]]]])
_CT = np.zeros(2)
# The pixels (channel, row, column) that vary; every other pixel is fixed at
# its own value. Three live pixels in different channels, rows and columns
# keep the exact image small and every layout confusion visible.
_LIVE = [(0, 0, 1), (1, 2, 3), (0, 1, 4)]
# Two live pixels, for the cells whose exact check grows fast with the
# predicate variables (the vertex enumeration of the exact method, the ReLU
# of an image on a box times 8).
_TWO_LIVE = _LIVE[:2]


def _live_box(seed=0, live=_LIVE, scale=1):
    """(C, H, W) bounds with a different value per pixel (``pixel_box``),
    times ``scale``, and positive width at the pixels of ``live`` only."""
    lo, hi = pixel_box(*_IMAGE, seed=seed)
    lo, hi = scale * lo, scale * hi
    fixed = np.ones(_IMAGE, dtype=bool)
    for p in live:
        fixed[p] = False
    return lo, np.where(fixed, lo, hi)


def _rows_only(image_star):
    """The same ImageStar given by its rows alone: its predicate box written
    as rows of ``C a <= d`` and ``predicate_lb``, ``predicate_ub`` None, the
    ImageStar a user builds from a polytope. The model entry gives a flat
    Star a finite predicate box and leaves an ImageStar as it is, so this
    input reaches the branches for stars without a box: the same-system test
    and the join with both boxes absent, and the tag rows of the predicate
    tagging."""
    n = image_star.nVar
    C = np.asarray(image_star.C, dtype=np.float64).reshape(-1, n)
    d = np.asarray(image_star.d, dtype=np.float64).reshape(-1, 1)
    lb = np.asarray(image_star.predicate_lb, dtype=np.float64).reshape(-1, 1)
    ub = np.asarray(image_star.predicate_ub, dtype=np.float64).reshape(-1, 1)
    return ImageStar(image_star.V, np.vstack([C, np.eye(n), -np.eye(n)]),
                     np.vstack([d, ub, -lb]), None, None, image_star.height,
                     image_star.width, image_star.num_channels)


def _image_input(kind, lo, hi, n=N_POINTS):
    """``(input set, member points)`` of the (C, H, W) box [lo, hi]: an
    ImageStar or ImageZono of the box, or for ``ImageStar+row-rows`` the
    ImageStar cut by one binding row that is not a box row
    (``constrained_image_star``) and given by its rows alone
    (:func:`_rows_only`), with member points that keep the row."""
    if kind == "ImageStar+row-rows":
        star, points = constrained_image_star(lo, hi, n=n, seed=5)
        return _rows_only(star), points
    return image_set(kind, lo, hi), samples(lo, hi, n)


def _conv_matrix(K, bias, pads, chw=_IMAGE):
    """``(W, b)``: an ONNX Conv with stride 1 on a (C, H, W) tensor as the
    affine map ``y = W x + b`` of the flattened tensors, in rationals."""
    channels, height, width = chw
    top, left, bottom, right = pads
    outs, _, kh, kw = K.shape
    oh, ow = height + top + bottom - kh + 1, width + left + right - kw + 1
    K = [[[[rational.frac(v) for v in row] for row in plane] for plane in out]
         for out in K]
    W = [[Fraction(0)] * (channels * height * width)
         for _ in range(outs * oh * ow)]
    for o, r, c in itertools.product(range(outs), range(oh), range(ow)):
        for ch, i, j in itertools.product(range(channels), range(kh),
                                          range(kw)):
            rr, cc = r + i - top, c + j - left
            if 0 <= rr < height and 0 <= cc < width:
                W[(o * oh + r) * ow + c][(ch * height + rr) * width + cc] += \
                    K[o][ch][i][j]
    b = [rational.frac(bias[o]) for o in range(outs) for _ in range(oh * ow)]
    return W, b


def _image_pieces(input_set, chw=_IMAGE):
    """The input image set as rational pieces in the network's (C, H, W)
    order: its stored (H, W, C) basis rows are put back in tensor order."""
    pieces = _predicate_pieces(input_set)
    order = hwc_index(*chw)
    n = len(order)
    P = [[Fraction(0)] * n for _ in range(n)]
    for k, p in enumerate(order):
        P[p][k] = Fraction(1)
    return rational.affine(pieces, P)


def _image_branch_model(tmp_path, op, relu_a, tail=True):
    """X -> A = [Relu](Conv_(1,3)(X)), B = Conv_(3,1)(X) -> op(A, B) ->
    [the 1x1 tail], on (1, 2, 3, 5)."""
    shape = [1, *_IMAGE]
    nodes = [helper.make_node("Conv", ["X", "KA", "CA"], ["A"],
                              pads=_PADS_A),
             helper.make_node("Conv", ["X", "KB", "CB"], ["B"],
                              pads=_PADS_B)]
    a = "A"
    if relu_a:
        nodes.append(helper.make_node("Relu", ["A"], ["RA"]))
        a = "RA"
    nodes.append(helper.make_node(op, [a, "B"], ["Q" if tail else "Y"]))
    inits = {"KA": _KA, "CA": _CA, "KB": _KB, "CB": _CB}
    if tail:
        nodes.append(helper.make_node("Conv", ["Q", "KT", "CT"], ["Y"]))
        inits.update(KT=_KT, CT=_CT)
    return save_onnx(tmp_path, nodes, shape, shape,
                     inits={k: v.astype(np.float32) for k, v in inits.items()})


def _image_branch_pieces(input_set, op, relu_a, tail=True):
    """The exact image of :func:`_image_branch_model` (Add or Sub, with the
    tail or not) on an image set, in (C, H, W) order."""
    WA, bA = _conv_matrix(_KA, _CA, _PADS_A)
    WB, bB = _conv_matrix(_KB, _CB, _PADS_B)
    n = len(WA)
    pieces = rational.affine(_image_pieces(input_set), WA + WB, bA + bB)
    if relu_a:
        pieces = _relu_on(pieces, list(range(n)))
    pieces = rational.affine(pieces, _combine(op, n))
    if not tail:
        return pieces
    WT, bT = _conv_matrix(_KT, _CT, [0, 0, 0, 0])
    return rational.affine(pieces, WT, bT)


# The ReLU regime runs on the ImageStar only: the ImageZono ReLU relaxation
# rounds to nearest and leaves its own exact image out by 7e-18 on this
# input (ROUND-10, a ReLU-kernel row), which this cell must not depend on.
@pytest.mark.parametrize("kind, relu", [("ImageStar", False),
                                        ("ImageStar", True),
                                        ("ImageZono", False),
                                        ("ImageStar+row-rows", False),
                                        ("ImageStar+row-rows", True)],
                         ids=["ImageStar-affine", "ImageStar-relu",
                              "ImageZono-affine", "ImageStar+row-rows-affine",
                              "ImageStar+row-rows-relu"])
@pytest.mark.parametrize("op", ["Add", "Sub"])
def test_add_or_sub_of_two_convolution_branches(tmp_path, op, kind, relu):
    """Conv_(1,3)(X) op Conv_(3,1)(X), with a ReLU on the first branch or
    not, then a 1x1 tail, on a (2, 3, 5) image whose pixels each have their
    own value and three of them a positive width. Every vertex of the exact
    image (in tensor order, permuted to the stored order) must be a member,
    and replayed outputs too. Catches an ImageStar or ImageZono sum that
    keeps one operand's centre or generators only, a subtraction of two
    ImageZonos computed as an addition, a join of two ImageStar predicate
    systems built with the operands swapped, and a sum in the wrong layout.
    ``ImageStar+row-rows`` is an ImageStar cut by a binding row and given by
    its rows alone: it runs the same-system
    test and the join with no predicate box, and the tag rows a ReLU branch
    then gets; the binding row makes the subtracted operand's polytope
    asymmetric, so a difference whose second block is not negated is caught
    (a box polytope is symmetric and hides it). It is on the box times 8
    with two live pixels (three would give the ReLU item 12 exact pieces
    and 18 s of rational arithmetic)."""
    path = _image_branch_model(tmp_path, op, relu)
    if kind.endswith("rows"):
        lo, hi = _live_box(seed=1, live=_TWO_LIVE, scale=_WIDE)
    else:
        lo, hi = _live_box(seed=1)
    input_set, points = _image_input(kind, lo, hi)
    what = f"{op} of two convolution branches ({'relu' if relu else 'affine'})"
    what += f" on an {kind}"
    sets = _reach(load_onnx(path), input_set, "approx", what)
    _assert_pieces_held(_image_branch_pieces(input_set, op, relu), sets, what,
                        out_chw=_IMAGE)
    assert_covered(sets, replay_onnx(path, points, [1, *_IMAGE]), what,
                   out_chw=_IMAGE)


@pytest.mark.parametrize("op", ["Add"])
def test_exact_split_of_an_image_branch(tmp_path, op):
    """method='exact': Relu(Conv_(1,3)(X)) op Conv_(3,1)(X) on a (2, 3, 5)
    ImageStar with three live pixels. The ReLU branch is a list of ImageStar
    pieces, each to be combined with the single convolution branch; every
    exact piece of the network must lie in one returned ImageStar (permuted
    to the stored order), and replayed outputs too. Catches a pairing of
    image pieces that drops pieces or combinations, and a sum of a piece and
    the single operand in the wrong layout. Only enclosure is checked: the
    joins are not the exact image (REACH-4); when PLAN step 18 makes them a
    refusal, this cell moves to the approximate method with
    split_piecewise_linear=True."""
    path = _image_branch_model(tmp_path, op, True)
    lo, hi = _live_box(seed=1)
    input_set = image_set("ImageStar", lo, hi)
    what = f"exact {op} of a split image branch and a single one"
    sets = _reach(load_onnx(path), input_set, "exact", what)
    if len(sets) < 2:
        pytest.fail(f"{what}: {len(sets)} set came back; the cell needs an "
                    f"operand split into several pieces")
    _assert_pieces_held(_image_branch_pieces(input_set, op, True), sets, what,
                        out_chw=_IMAGE)
    assert_covered(sets, replay_onnx(path, samples(lo, hi, N_POINTS),
                                     [1, *_IMAGE]), what, out_chw=_IMAGE)


# --------------------------------------------------------------------------- #
# Concat of a system without a predicate box and one with                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", ["flat+softmax", "softmax+flat"])
def test_concat_of_a_system_without_a_predicate_box_and_one_with(tmp_path,
                                                                 order):
    """Concat(Flatten(X), Softmax(Flatten(X))) in both orders, on an
    ImageStar cut by a binding row and given by its rows alone, on the box
    times 8. The flattened branch keeps the input's predicate system, which
    has no predicate box; Softmax rebuilds its star from bounds, with a
    finite box. The join of the two systems must leave the variables of the
    first unbounded, so that the input's rows alone bound them. Catches a
    join that gives a system without a box a finite one (zeros, or the other
    system's box): member points are then lost. The other rows-only cells of
    this file never reach that branch of ``_join_star_systems``: all their
    operands lack a box (an affine map, a ReLU and a Sigmoid keep the
    input's absence of one). The half of each replayed output that is the
    input itself, in tensor order, also holds the flattened branch to the
    layout."""
    names = ["F", "S"] if order == "flat+softmax" else ["S", "F"]
    nodes = [helper.make_node("Flatten", ["X"], ["F"], axis=1),
             helper.make_node("Softmax", ["F"], ["S"], axis=1),
             helper.make_node("Concat", names, ["Y"], axis=1)]
    size = int(np.prod(_IMAGE))
    path = save_onnx(tmp_path, nodes, [1, *_IMAGE], [1, 2 * size])
    lo, hi = _live_box(seed=1, live=_TWO_LIVE, scale=_WIDE)
    input_set, points = _image_input("ImageStar+row-rows", lo, hi)
    what = f"approx Concat ({order}) on an ImageStar given by its rows"
    sets = _reach(load_onnx(path), input_set, "approx", what)
    if len(sets) != 1:
        pytest.fail(f"{what}: {len(sets)} sets came back; the cell needs "
                    f"the one joined star")
    assert_covered(sets, replay_onnx(path, points, [1, *_IMAGE]), what)


# --------------------------------------------------------------------------- #
# The exact method: the result must be the exact image                       #
# --------------------------------------------------------------------------- #
_REACH_4 = known_bug(
    18, "REACH-4: method='exact' joins the pieces of a split operand with an "
        "operand on another predicate system block-diagonally, which drops "
        "their correlation: the result holds every output but is not the "
        "exact image (returned vertices have no preimage), where the exact "
        "method must return the image or refuse")
# With two live pixels the joined ImageStar has four predicate variables.
_EXACT_VARIABLES = 6


def _assert_no_vertex_without_preimage(sets, pieces, what, out_chw=None):
    """The other direction of an exact result: every vertex of every returned
    set lies in the closure of the exact image (``rational.unreached``),
    decided in exact arithmetic with no slack. An image set's vertices are
    put back in tensor order first. The vertices are enumerated over the
    set's predicate polytope, which grows fast with its variables, so a set
    with more than ``_EXACT_VARIABLES`` stops the test first with
    ``pytest.fail`` (a fault the test cannot decide, which no marker
    absorbs)."""
    order = None if out_chw is None else list(hwc_index(*out_chw))
    for i, s in enumerate(sets):
        if s.nVar > _EXACT_VARIABLES:
            pytest.fail(f"{what}: returned set {i} has {s.nVar} predicate "
                        f"variables; the vertex check enumerates at most "
                        f"{_EXACT_VARIABLES}")
        vertices = rational.set_vertices(rational.star(s))
        if order is not None and isinstance(s, (ImageStar, ImageZono)):
            back = []
            for v in vertices:
                y = [None] * len(v)
                for k, p in enumerate(order):
                    y[p] = v[k]
                back.append(y)
            vertices = back
        missing = rational.unreached(pieces, vertices)
        if missing:
            shown = ", ".join(f"{float(v):g}" for v in missing[0][1][:6])
            raise AssertionError(
                f"{what}: {len(missing)} of {len(vertices)} vertices of "
                f"returned set {i} have no preimage, for example ({shown})")


@pytest.mark.parametrize("kind, op, split", [
    ("Star", "Add", False), ("Star", "Sub", False),
    pytest.param("Star", "Sub", True, marks=_REACH_4),
    ("ImageStar", "Sub", False),
    pytest.param("ImageStar", "Add", True, marks=_REACH_4)],
    ids=["Star-Add-shared", "Star-Sub-shared", "Star-Sub-split",
         "ImageStar-Sub-shared", "ImageStar-Add-split"])
def test_exact_add_or_sub_is_the_exact_image(tmp_path, kind, op, split):
    """method='exact' claims the exact image, so both directions are
    checked: every exact piece lies in one returned set, and every vertex of
    every returned set has a preimage. "shared": two affine branches of the
    input, one predicate system, summed exactly; catches an exact sum that
    takes the join path or counts a generator twice (a larger set, sound but
    not exact). "split": a ReLU on the first branch splits it into pieces
    whose systems extend the input's; today each piece is joined
    block-diagonally with the other branch, so the result is larger than the
    image (REACH-4, strict expected failures); the promise is the image or a
    refusal (PLAN step 18), and a refusal passes. A true output outside the
    result is a worse defect than a result that is too large, so the
    enclosure checks report with ``pytest.fail``, which the not-exact marker
    does not absorb. The flat cells use the box and biases times 8 (x0 in
    [-8, 8], x1 in [-4, 6]); the image cells a (2, 3, 5) ImageStar with two
    live pixels, its box times 8. 128 points are replayed."""
    if kind == "Star":
        path = _two_branch_model(tmp_path, op, split, False, tail=False,
                                 scale=_WIDE)
        input_set, points = _input("Star", n=N_SAMPLES, scale=_WIDE)
        pieces = _two_branch_pieces(input_set, op, split, False, tail=False,
                                    scale=_WIDE)
        out_chw, in_shape = None, [1, 2]
    else:
        path = _image_branch_model(tmp_path, op, split, tail=False)
        lo, hi = _live_box(seed=1, live=_TWO_LIVE, scale=_WIDE)
        input_set, points = image_set(kind, lo, hi), samples(lo, hi)
        pieces = _image_branch_pieces(input_set, op, split, tail=False)
        out_chw, in_shape = _IMAGE, [1, *_IMAGE]
    what = (f"exact {op} of two branches "
            f"({'split' if split else 'shared'}) on a {kind}")
    try:
        sets = _reach(load_onnx(path), input_set, "exact", what)
    except NotImplementedError:
        if split:
            return
        raise
    try:
        _assert_pieces_held(pieces, sets, what, out_chw=out_chw)
        assert_covered(sets, replay_onnx(path, points, in_shape), what,
                       out_chw=out_chw)
    except AssertionError as error:
        pytest.fail(f"true outputs outside the result: {error}")
    _assert_no_vertex_without_preimage(sets, pieces, what, out_chw=out_chw)


_IMAGEZONO_PRODUCT = known_bug(
    7, "Mul of two ImageZonos returns a flat Zono whose entries are in the "
       "stored (height, width, channel) order, while a flat set stands for "
       "the tensor in its own (channel, height, width) order: read as the "
       "output tensor, the true outputs fall outside it")


@pytest.mark.parametrize("op, kind, method", [
    pytest.param("Mul", "ImageZono", "approx", marks=_IMAGEZONO_PRODUCT,
                 id="Mul-ImageZono"),
    pytest.param("Mul", "ImageStar", "approx", id="Mul-ImageStar"),
    pytest.param("Div", "ImageStar", "approx", id="Div-ImageStar"),
    pytest.param("Div", "ImageZono", "approx", id="Div-ImageZono"),
    pytest.param("Mul", "ImageStar", "exact", id="Mul-ImageStar-exact"),
    pytest.param("Div", "ImageStar", "exact", id="Div-ImageStar-exact")])
def test_product_or_quotient_of_two_image_branches(tmp_path, op, kind,
                                                   method):
    """Conv_(1,3)(X) op Conv_(3,1)(X) on a (2, 3, 5) image. The ImageStar
    product and both quotients are refused (``NotImplementedError``), under
    both methods for the ImageStar (an exact reach of an ImageZono is
    refused at the entry); the ImageZono product is answered and must hold
    every replayed output in the tensor's order (a flat result) or the
    stored order (an image result). The passing neighbour with the same
    model, input and replay is the ImageZono cell of the Add test above.
    Catches a product or quotient of two image sets that is answered where
    the library has no rule for it, and a product computed in the stored
    layout and returned as a flat set."""
    path = _image_branch_model(tmp_path, op, False, tail=False)
    lo, hi = _live_box(seed=2)
    what = f"{method} {op} of two convolution branches on an {kind}"
    model = load_onnx(path)
    if not (op == "Mul" and kind == "ImageZono"):
        with pytest.raises(NotImplementedError, match="not supported"):
            _reach(model, image_set(kind, lo, hi), method, what)
        return
    sets = _reach(model, image_set(kind, lo, hi), method, what)
    assert_covered(sets, replay_onnx(path, samples(lo, hi, N_POINTS),
                                     [1, *_IMAGE]), what, out_chw=_IMAGE)


_GD = np.random.default_rng(11).integers(-4, 5, size=(30, 30)) / 8.0
_BD = np.random.default_rng(12).integers(-4, 5, size=30) / 4.0
_K1 = np.array([[[[0.5]], [[-1.0]]], [[[0.25]], [[0.75]]]])
_C1 = np.array([0.25, -0.5])


@pytest.mark.parametrize("op, kind, order", [
    (op, kind, order)
    for op, kinds in (("Add", ("ImageStar", "ImageZono",
                               "ImageStar+row-rows")),
                      ("Sub", ("ImageStar", "ImageZono")))
    for kind in kinds for order in ("image+flat", "flat+image")])
def test_add_or_sub_of_an_image_set_and_a_flat_set(tmp_path, op, kind, order):
    """Conv_1x1(X) op Reshape(Flatten(X) @ G + b): the convolution branch is
    an image set, the dense branch a flat set in the tensor's (C, H, W)
    order, and the image operand is flattened before the sum or difference,
    whichever operand comes first. Every vertex of the exact image must be a
    member of the result (a flat set, compared in tensor order) and replayed
    outputs too. Catches an image operand flattened in its stored (H, W, C)
    order, for either operand order, and, by the Sub items, the two operands
    handed back in the other order by the step that brings them to one set
    type (a difference read as flat minus image where the graph says image
    minus flat; the sum cannot see it).
    ``ImageStar+row-rows`` (an ImageStar given by
    its rows alone, cut by a binding row, box times 8) makes both flat
    operands stars without a predicate box
    on one system, the exact sum's branch for that case."""
    shape = [1, *_IMAGE]
    a, b = ("A", "D") if order == "image+flat" else ("D", "A")
    nodes = [helper.make_node("Conv", ["X", "K1", "C1"], ["A"]),
             helper.make_node("Flatten", ["X"], ["F"], axis=1),
             helper.make_node("MatMul", ["F", "G"], ["P"]),
             helper.make_node("Add", ["P", "BD"], ["E"]),
             helper.make_node("Reshape", ["E", "S"], ["D"]),
             helper.make_node(op, [a, b], ["Y"])]
    inits = {"K1": _K1.astype(np.float32), "C1": _C1.astype(np.float32),
             "G": _GD.astype(np.float32), "BD": _BD.astype(np.float32),
             "S": np.array(shape, dtype=np.int64)}
    path = save_onnx(tmp_path, nodes, shape, shape, inits=inits)
    lo, hi = _live_box(seed=3, scale=_WIDE if kind.endswith("rows") else 1)
    input_set, points = _image_input(kind, lo, hi)
    what = f"{op} ({order}) on an {kind}"
    sets = _reach(load_onnx(path), input_set, "approx", what)
    W1, b1 = _conv_matrix(_K1, _C1, [0, 0, 0, 0])
    G = rational.matrix(_GD.T)
    # the coefficients of the image branch and of the flat branch in Y
    si, sf = ((1, 1) if op == "Add" else
              (1, -1) if order == "image+flat" else (-1, 1))
    W = [[si * u + sf * v for u, v in zip(r1, r2)] for r1, r2 in zip(W1, G)]
    b = [si * u + sf * rational.frac(v) for u, v in zip(b1, _BD)]
    _assert_pieces_held(rational.affine(_image_pieces(input_set), W, b), sets,
                        what, out_chw=_IMAGE)
    assert_covered(sets, replay_onnx(path, points, shape), what,
                   out_chw=_IMAGE)


# For the quotients of the image and the flat branch: with these the
# convolution branch is in [27/4, 23/2] (channel 0) and [-173/16, -115/16]
# (channel 1) and the dense branch in [9/8, 471/32] on the box of
# pixel_box(seed=3) (so on _live_box(seed=3) too), so either can be the
# divisor.
_C1_DIV = np.array([9.0, -9.0])
_BD_DIV = np.full(30, 8.0)


@pytest.mark.parametrize("op, kind, order", [
    ("Mul", "ImageStar", "image+flat"), ("Mul", "ImageZono", "image+flat"),
    ("Mul", "ImageStar+row-rows", "image+flat"),
    ("Div", "ImageStar", "image+flat"), ("Div", "ImageStar", "flat+image"),
    ("Div", "ImageStar+row-rows", "flat+image")])
def test_product_or_quotient_of_an_image_set_and_a_flat_set(tmp_path, op,
                                                            kind, order):
    """Conv_1x1(X) op Reshape(Flatten(X) @ G + b) on a (2, 3, 5) image with
    its own interval per pixel: the convolution branch is an image set, the
    dense branch a flat set in the tensor's (C, H, W) order, and the image
    operand is flattened before the product or quotient (the quotient's
    operands are
    not interchangeable, so it runs in both orders; the product commutes,
    and the flattening of an image operand in either position is shared
    with the Add and Sub items above, which run both orders). Three pixels
    have a positive width, so the input has three predicate variables
    against 30 output entries (bounds of the result stacked in the wrong
    order then land on the input variables). ``ImageStar+row-rows`` is an
    ImageStar cut by a binding row and given by its rows alone, so both flat
    operands are stars without a predicate box on one system (the branch of
    the kernels for operands without a box; the box times 8 for the
    product). 128 outputs of the float64 torch model (the extreme corners
    first) must be members, decided by one LP per point on the result's
    rows in tensor order (the result is flat). Not proved: the interval
    proof needs the vertices of a 30-entry enclosure, and an exact
    membership in rationals takes seconds to minutes a point; the product's
    known rounding on operand entries of zero width (ROUND-20; the
    point item of the rounding cells) stays inside the LP's slack. Catches an image operand flattened in its stored (H, W, C)
    order, the two operands handed back swapped by the step that brings
    them to one set type (a quotient flat / image read as image / flat), and
    the bounds of a product or quotient of operands without a predicate box
    stacked in the wrong order. The ImageZono quotient is refused
    (:func:`test_product_or_quotient_of_two_image_branches`)."""
    shape = [1, *_IMAGE]
    a, b = ("A", "D") if order == "image+flat" else ("D", "A")
    c1, bd = (_C1, _BD) if op == "Mul" else (_C1_DIV, _BD_DIV)
    nodes = [helper.make_node("Conv", ["X", "K1", "C1"], ["A"]),
             helper.make_node("Flatten", ["X"], ["F"], axis=1),
             helper.make_node("MatMul", ["F", "G"], ["P"]),
             helper.make_node("Add", ["P", "BD"], ["E"]),
             helper.make_node("Reshape", ["E", "S"], ["D"]),
             helper.make_node(op, [a, b], ["Y"])]
    inits = {"K1": _K1.astype(np.float32), "C1": c1.astype(np.float32),
             "G": _GD.astype(np.float32), "BD": bd.astype(np.float32),
             "S": np.array(shape, dtype=np.int64)}
    path = save_onnx(tmp_path, nodes, shape, shape, inits=inits)
    wide = op == "Mul" and kind.endswith("rows")
    lo, hi = _live_box(seed=3, scale=_WIDE if wide else 1)
    input_set, points = _image_input(kind, lo, hi, n=N_SAMPLES)
    what = f"{op} ({order}) on an {kind}"
    model = load_onnx(path)
    sets = _reach(model, input_set, "approx", what)
    assert_covered(sets, replay_torch(model, points, shape), what,
                   out_chw=_IMAGE)


# --------------------------------------------------------------------------- #
# Image sets of different tensor shapes: X op Reduce(X)                      #
# --------------------------------------------------------------------------- #
_SE_IMAGE = (2, 2, 4)
_SE_LIVE = [(0, 0, 1), (1, 1, 3), (0, 1, 2)]
# reducer: (ONNX node attributes, reduced axes of (1, C, H, W), scale)
_REDUCERS = {"GlobalAveragePool": (("GlobalAveragePool", {}), (2, 3), 1 / 8),
             "ReduceSum-H": (("ReduceSum", {"axes": [2], "keepdims": 1}),
                             (2,), 1),
             "ReduceSum-W": (("ReduceSum", {"axes": [3], "keepdims": 1}),
                             (3,), 1)}
_SHAPE_3 = known_bug(
    12, "SHAPE-3: an ImageZono gets no tracked tensor shape without "
        "input_shape, so Add or Sub of an image and a reduced copy of it is "
        "not broadcast and crashes with a numpy ValueError (with operands of "
        "equal entry count and different shapes it adds them entry by entry "
        "into a set of the wrong size)", raises=ValueError)


def _reduce_broadcast_matrix(chw, axes, scale):
    """The matrix of ``scale`` times the sum over ``axes`` of a (1, C, H, W)
    tensor, broadcast back to (1, C, H, W): entry (out, in) is ``scale``
    when the input entry is reduced into the output entry's broadcast
    position (both in row-major tensor order)."""
    shape = (1, *chw)
    index = np.arange(int(np.prod(shape))).reshape(shape)
    M = np.zeros((index.size, index.size))
    for out in np.ndindex(*shape):
        line = list(out)
        for ax in axes:
            line[ax] = slice(None)
        M[index[out], index[tuple(line)].reshape(-1)] = scale
    return M


@pytest.mark.parametrize("op, order, reducer, kind, shape_given", [
    ("Add", "X op R", "GlobalAveragePool", "ImageStar", False),
    ("Sub", "R op X", "ReduceSum-W", "ImageStar", False),
    ("Sub", "X op R", "ReduceSum-H", "ImageZono", True),
    ("Add", "R op X", "GlobalAveragePool", "ImageZono", True),
    pytest.param("Sub", "X op R", "GlobalAveragePool", "ImageZono", False,
                 marks=_SHAPE_3)],
    ids=["Add-X-gap-ImageStar", "Sub-R-sumW-ImageStar",
         "Sub-X-sumH-ImageZono-shape", "Add-R-gap-ImageZono-shape",
         "Sub-X-gap-ImageZono-noshape"])
def test_add_or_sub_of_an_image_and_a_reduced_copy(tmp_path, op, order,
                                                   reducer, kind,
                                                   shape_given):
    """X op R and R op X with R = GlobalAveragePool(X) (C, 1, 1) or a
    ReduceSum over the height (C, 1, W) or the width (C, H, 1), the
    squeeze-and-excitation pattern, on a (2, 2, 4) image with its own value
    per pixel and three live pixels, box times 8. R is broadcast to
    (C, H, W) by row selection: the
    small operand is flattened in tensor order and expanded, the image
    operand passes and is flattened when it meets the flat one, whichever
    comes first. Every vertex of the input maps to an exact output that
    must be a member (the result is flat, in tensor order), and replayed
    outputs too. Catches a reduced image operand of several pixels
    flattened in its stored (H, W, C) order before the row map, and either
    operand order lost. 128 points are replayed. The ImageZono gets its
    shape from ``input_shape``; without it the walk tracks no shape for an
    ImageZono (SHAPE-3) and the marked item crashes."""
    (node_type, attrs), axes, scale = _REDUCERS[reducer]
    first, second = ("X", "R") if order == "X op R" else ("R", "X")
    nodes = [helper.make_node(node_type, ["X"], ["R"], **attrs),
             helper.make_node(op, [first, second], ["Y"])]
    shape = [1, *_SE_IMAGE]
    path = save_onnx(tmp_path, nodes, shape, shape, opset=11)
    lo, hi = pixel_box(*_SE_IMAGE, seed=6)
    lo, hi = _WIDE * lo, _WIDE * hi
    fixed = np.ones(_SE_IMAGE, dtype=bool)
    for p in _SE_LIVE:
        fixed[p] = False
    hi = np.where(fixed, lo, hi)
    what = f"{op}({first}, {second}) with R = {reducer} on an {kind}"
    options = {"input_shape": _SE_IMAGE} if shape_given else {}
    sets = _reach(load_onnx(path), image_set(kind, lo, hi), "approx", what,
                  **options)
    R = _reduce_broadcast_matrix(_SE_IMAGE, axes, scale)
    M = np.eye(R.shape[0]) + (1 if op == "Add" else -1) * R
    if order == "R op X":
        M = R + (1 if op == "Add" else -1) * np.eye(R.shape[0])
    M = rational.matrix(M)
    corners = []
    for ends in itertools.product(*[(lo[p], hi[p]) for p in _SE_LIVE]):
        x = lo.copy()
        for p, v in zip(_SE_LIVE, ends):
            x[p] = v
        x = rational.vector(x.reshape(-1))
        corners.append([rational.dot(row, x) for row in M])
    _assert_points_held(sets, corners, what, out_chw=_SE_IMAGE)
    assert_covered(sets, replay_onnx(path, samples(lo, hi), shape), what,
                   out_chw=_SE_IMAGE)


_SHAPE_3_PRODUCT = known_bug(
    12, "SHAPE-3: an ImageZono gets no tracked tensor shape without "
        "input_shape, so the product of an image and a reduced copy of it "
        "is not broadcast and _mul_zonos crashes with a numpy ValueError "
        "where an answer or a refusal is due", raises=ValueError)


@pytest.mark.parametrize("op, order, reducer, kind, shape_given", [
    ("Mul", "X op R", "GlobalAveragePool", "ImageStar", False),
    ("Div", "R op X", "ReduceSum-W", "ImageStar", False),
    ("Mul", "R op X", "ReduceSum-H", "ImageZono", True),
    pytest.param("Mul", "X op R", "GlobalAveragePool", "ImageZono", False,
                 marks=_SHAPE_3_PRODUCT)],
    ids=["Mul-X-gap-ImageStar", "Div-R-sumW-ImageStar",
         "Mul-R-sumH-ImageZono-shape", "Mul-X-gap-ImageZono-noshape"])
def test_product_or_quotient_of_an_image_and_a_reduced_copy(
        tmp_path, op, order, reducer, kind, shape_given):
    """X op R and R op X with R = GlobalAveragePool(X) (C, 1, 1) or a
    ReduceSum over the height (C, 1, W) or the width (C, H, 1), the
    squeeze-and-excitation pattern, on a (2, 2, 4) image with its own
    interval per pixel, moved above zero by
    2 for the quotient. Every pixel has a positive width: with fixed pixels
    a reduced row with no live pixel meets a fixed pixel, both operands are
    constants there, the result's variable has a box one rounding step wide,
    and the LP of the membership oracle leaves its valid regime on it. The
    box is not scaled: on the box times 8 the LP of
    the membership oracle leaves its valid regime (rows broken by 1e-9 of
    their scale); the flat cells on the wide box hold that property. R is broadcast to (C, H, W) by row
    selection and the image operand is flattened when it meets it, so the
    result is flat, in tensor order. 128 outputs of the float64 torch model
    (the extreme corners first; onnxruntime has no float64
    GlobalAveragePool) must be members, decided by one LP per point on the
    result's rows (an exact membership over 32 variables is too slow, see
    the image and flat cell). Catches a reduced
    operand of several pixels flattened in its stored (H, W, C) order before
    the row map, a quotient R / X read as X / R, and an ImageZono product
    computed in the stored layout. The ImageZono gets its shape from
    ``input_shape``; without it the walk tracks no shape for an ImageZono
    (SHAPE-3) and the marked item crashes in ``_mul_zonos``. The ImageZono
    quotient is refused (``test_product_or_quotient_of_two_image_branches``)."""
    (node_type, attrs), axes, scale = _REDUCERS[reducer]
    first, second = ("X", "R") if order == "X op R" else ("R", "X")
    nodes = [helper.make_node(node_type, ["X"], ["R"], **attrs),
             helper.make_node(op, [first, second], ["Y"])]
    shape = [1, *_SE_IMAGE]
    path = save_onnx(tmp_path, nodes, shape, shape, opset=11)
    lo, hi = pixel_box(*_SE_IMAGE, seed=6)
    shift = 2.0 if op == "Div" else 0.0
    lo, hi = lo + shift, hi + shift
    what = f"{op}({first}, {second}) with R = {reducer} on an {kind}"
    options = {"input_shape": _SE_IMAGE} if shape_given else {}
    model = load_onnx(path)
    sets = _reach(model, image_set(kind, lo, hi), "approx", what, **options)
    assert_covered(sets, replay_torch(model, samples(lo, hi), shape), what,
                   out_chw=_SE_IMAGE)


_OPSET6_IMAGE = (3, 2, 3)
_OPSET6_LIVE = [(0, 0, 1), (2, 1, 0), (1, 0, 2)]
_OPSET6_AXIS_IGNORED = known_bug(
    7, "Add or Sub of two computed sets at opset 6 with broadcast=1 and an "
       "inner axis: the graph walk ignores the broadcast and axis attributes "
       "and broadcasts the second operand along the trailing axes, so the "
       "result is for another network and every true output falls outside "
       "(registry GRAPH-12; the same attributes are ignored for a "
       "constant operand)")


@pytest.mark.parametrize("axis", [pytest.param(1, marks=_OPSET6_AXIS_IGNORED),
                                  3], ids=["axis1", "axis3"])
def test_opset6_broadcast_of_a_computed_operand_follows_its_axis(tmp_path,
                                                                 axis):
    """Add(X, R, broadcast=1, axis) at opset 6 on a (3, 2, 3) ImageStar, R
    the ReduceSum of X over every axis but ``axis`` (keepdims 0), so R has
    the length of that axis and the model adds it along that axis. C = W =
    3, so R also fits the trailing axis: a broadcast that ignores ``axis``
    answers silently. onnxruntime has no opset-6 kernels, so the loaded
    torch module is the replay; it is checked first against the exact
    value (a setup check, ``pytest.fail``). Then every vertex of the input
    maps to an exact output that must be a member, and 48 torch outputs are
    replayed. Catches a broadcast of a computed operand that follows numpy's
    trailing alignment instead of the node's axis; axis 3 is the passing
    neighbour, where the two agree."""
    keep = [a for a in (0, 1, 2, 3) if a != axis]
    nodes = [helper.make_node("ReduceSum", ["X"], ["R"], axes=keep,
                              keepdims=0),
             helper.make_node("Add", ["X", "R"], ["Y"], broadcast=1,
                              axis=axis)]
    shape = [1, *_OPSET6_IMAGE]
    path = save_onnx(tmp_path, nodes, shape, shape, opset=6)
    model = load_onnx(path)
    lo, hi = pixel_box(*_OPSET6_IMAGE, seed=2)
    fixed = np.ones(_OPSET6_IMAGE, dtype=bool)
    for p in _OPSET6_LIVE:
        fixed[p] = False
    hi = np.where(fixed, lo, hi)
    corners, exact = [], []
    for ends in itertools.product(*[(lo[p], hi[p]) for p in _OPSET6_LIVE]):
        x = lo.copy()
        for p, v in zip(_OPSET6_LIVE, ends):
            x[p] = v
        r = x.sum(axis=tuple(a - 1 for a in keep if a > 0))
        y = x + r.reshape([-1 if a == axis - 1 else 1 for a in range(3)])
        corners.append(x)
        exact.append([rational.frac(v) for v in y.reshape(-1)])
    replayed = replay_torch(model, corners, shape)
    if any(not np.array_equal(np.array([float(v) for v in e]), t)
           for e, t in zip(exact, replayed)):
        pytest.fail("the loaded opset-6 model does not compute the axis-"
                    f"{axis} broadcast the cell's reference assumes")
    what = f"opset-6 Add(X, R, broadcast=1, axis={axis}) on an ImageStar"
    sets = _reach(model, image_set("ImageStar", lo, hi), "approx", what)
    _assert_points_held(sets, exact, what, out_chw=_OPSET6_IMAGE)
    assert_covered(sets, replay_torch(model, samples(lo, hi, N_POINTS),
                                      shape), what, out_chw=_OPSET6_IMAGE)


# --------------------------------------------------------------------------- #
# Refusals                                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind, op", [
    (kind, op) for kind in ("Hexatope", "Octatope", "Zono")
    for op in ("Add", "Sub", "Mul", "Div")
    if not (kind == "Zono" and op != "Div")])
def test_unsupported_two_set_operations_are_refused(tmp_path, kind, op):
    """Every operation between two computed Hexatopes or Octatopes, and the
    quotient of two Zonos, has no rule and must raise
    ``NotImplementedError``. The Zono's Add, Sub and Mul items, which
    answered and were replayed on 16 points, were removed: the
    affine-affine Zono items of
    :func:`test_add_or_sub_of_two_branches_holds_the_exact_image` hold the
    exact image of the same model, and :func:`test_mul_of_two_branches_is_proved`
    and :func:`test_product_or_quotient_on_a_wide_box` (the same three
    neurons) prove the product. Catches an operation that answers with one
    operand, or with a set of a type nobody handles, where it must refuse."""
    path = _two_branch_model(tmp_path, op, False, False,
                             bb=_BB_DIV if op == "Div" else _BB)
    lo, hi = np.array([-1.0, -0.5]), np.array([1.0, 0.75])
    what = f"{op} of two branches on a {kind}"
    model = load_onnx(path)
    with pytest.raises(NotImplementedError, match="not supported"):
        _reach(model, flat_set(kind, lo, hi), "approx", what)


# --------------------------------------------------------------------------- #
# Rounding: two-set arithmetic on float64 data that is not dyadic            #
# --------------------------------------------------------------------------- #
_ROUNDING_N = 12
_ROUND_8 = known_bug(
    "7b", "ROUND-8: the sum or difference of two Stars or two Zonos adds "
          "centres and generators with round-to-nearest, so the set can leave "
          "out the exact sum of its operands by one unit in the last place")
_ZONO_PRODUCT_ROUNDING = known_bug(
    7, "Mul of two Zonos multiplies the operands' bounds with "
       "round-to-nearest and builds the Zono from those bounds, so the "
       "exact product can fall outside it")
_STAR_PRODUCT_ROUNDING = known_bug(
    7, "Mul of two Stars on an operand entry with no generator pins the "
       "product to its round-to-nearest value through the McCormick rows, "
       "whose right-hand sides are not moved outward, so the exact product "
       "falls outside")


def _rounding_model(tmp_path, op, n=_ROUNDING_N, split=False):
    """A float64 model X (1, 2n) -> A = X[:n], B = X[n:] -> op(A, B). The
    operands are taken by two 0/1 MatMuls (exact for a Star and a Zono) or,
    with ``split``, by an ONNX Split, an exact row selection for every set
    type: the affine map of a Box rounds outward and so widens each operand
    by one rounding step, which hides a sum rounded to nearest."""
    if split:
        nodes = [helper.make_node("Split", ["X"], ["A", "B"], axis=1),
                 helper.make_node(op, ["A", "B"], ["Y"])]
        return save_onnx(tmp_path, nodes, [1, 2 * n], [1, n],
                         in_type="DOUBLE", out_type="DOUBLE")
    pick_a = np.vstack([np.eye(n), np.zeros((n, n))])
    pick_b = np.vstack([np.zeros((n, n)), np.eye(n)])
    nodes = [helper.make_node("MatMul", ["X", "PA"], ["A"]),
             helper.make_node("MatMul", ["X", "PB"], ["B"]),
             helper.make_node(op, ["A", "B"], ["Y"])]
    return save_onnx(tmp_path, nodes, [1, 2 * n], [1, n],
                     inits={"PA": pick_a, "PB": pick_b}, in_type="DOUBLE",
                     out_type="DOUBLE")


def _exact_extremes(op, lo, hi, n=_ROUNDING_N):
    """The two output vectors of the per-coordinate minima and maxima of
    ``op(a, b)`` over the box, in exact fractions; the coordinates are
    independent, so each vector is a true output."""
    f = {"Add": lambda u, v: u + v, "Sub": lambda u, v: u - v,
         "Mul": lambda u, v: u * v, "Div": lambda u, v: u / v}[op]
    low, high = [], []
    for i in range(n):
        values = [f(rational.frac(u), rational.frac(v))
                  for u in (lo[i], hi[i]) for v in (lo[n + i], hi[n + i])]
        low.append(min(values))
        high.append(max(values))
    return [low, high]


def _rounding_cell(tmp_path, kind, op, width, n=_ROUNDING_N, split=False):
    rng = np.random.default_rng(17)
    lo = rng.uniform(0.3, 3.0, 2 * n)
    hi = lo if width == "point" else lo + rng.uniform(0.01, 0.5, lo.size)
    path = _rounding_model(tmp_path, op, n, split)
    what = f"{op} of two {kind} operands ({width} input), float64"
    sets = _reach(load_onnx(path), flat_set(kind, lo, hi), "approx",
                       what)
    _assert_points_held(sets, _exact_extremes(op, lo, hi, n), what)


@pytest.mark.parametrize("width", ["point", "interval"])
@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div"])
def test_box_two_set_arithmetic_is_rounded_outward(tmp_path, op, width):
    """A op B for two Box operands with non-dyadic float64 bounds: the exact
    sum, difference, product or quotient of the bounds (both extremes) must
    lie in the returned Box, compared in exact fractions with no slack.
    Catches a bound computed with plain float64 arithmetic, which lands one
    rounding error inside the exact value on about half the entries. The
    operands are taken by an ONNX Split over 64 coordinates: through the 0/1
    MatMuls of the other cells the Box
    affine map widens each operand by one rounding step, which hides a sum,
    difference, product or quotient rounded to nearest."""
    _rounding_cell(tmp_path, "Box", op, width, n=64, split=True)


@pytest.mark.parametrize("width", ["point", "interval"])
@pytest.mark.parametrize("op, kind", [
    pytest.param("Add", "Star", marks=_ROUND_8),
    pytest.param("Add", "Zono", marks=_ROUND_8),
    pytest.param("Sub", "Star", marks=_ROUND_8),
    pytest.param("Sub", "Zono", marks=_ROUND_8),
    pytest.param("Mul", "Zono", marks=_ZONO_PRODUCT_ROUNDING),
    ("Div", "Star")])
def test_star_and_zono_two_set_arithmetic_holds_the_exact_value(
        tmp_path, op, kind, width):
    """The rounding cell of the Box test above on a Star and a Zono (same
    model, data and exact reference). The Star quotient passes; the sums and
    differences are ROUND-8; the Zono product rounds its interval products
    to nearest."""
    _rounding_cell(tmp_path, kind, op, width)


@pytest.mark.parametrize("width", [
    pytest.param("point", marks=_STAR_PRODUCT_ROUNDING), "interval"])
def test_star_product_holds_the_exact_value(tmp_path, width):
    """The rounding cell on the McCormick product of two Stars. On an
    interval input the padded operand ranges leave room and the cell passes
    (the neighbour); on a point input the envelope rows are tight and their
    round-to-nearest right-hand sides exclude the exact product."""
    _rounding_cell(tmp_path, "Star", "Mul", width)
