"""Soundness cells for ONNX MatMul with one constant operand
(``_handle_onnx_matmul`` in ``n2v/nn/reach.py``), through the model entry
point ``reach_pytorch_model``.

What the cells cover. The constant on the right of the data (``x @ W``) and on
the left (``W @ x``); data with one row or column, with several rows or
columns, and with a leading axis (of length one, and of length two: one
matrix per head, a known defect marked once); a constant with one axis (ONNX
promotes it to a row or a column and drops that axis again) and a constant
with three axes whose leading axis has length one. Every form runs under
``approx`` on a Star with a binding constraint row that is not a box row;
the forms that differ in the set-type branch (each side, square, several
rows) also run on Box, Zono, Hexatope and Octatope, and under ``exact`` on
the plain and the constrained Star, where the result must be the exact image
in both directions. Further: the pieces of an exact ReLU split ahead of the
MatMul; MatMul between two affine nodes, an approx ReLU between two MatMuls
(the rows the first one emits are read downstream), and a residual Add that
reads the MatMul's input again; a float64 file whose constant is not a
float32 number, on a box eight times the usual one, on every set type; a Box
with non-dyadic data (outward rounding); the model reached without a tensor
shape; a torch model that spells the product (refused today); and
``reach_layer`` handed the MatMul module (refused). Refusals: a constant with
three axes whose leading axis is longer than one, an image set fed to MatMul,
and an input set of the wrong size. The handler reads no option
(``lp_solver``, relaxation, pre-computed bounds), so none is varied.

Oracles, neither of them n2v. (1) The exact image in rationals
(``tests/_oracles/rational.py``): the input polytope (the box, or the Star's
own predicate polytope mapped through its basis) followed by the MatMul's flat
matrix, whose column ``j`` is ``numpy.matmul`` of the constant with the
``j``-th unit tensor (ONNX defines MatMul as ``numpy.matmul``; the matrix is
checked against onnxruntime by ``test_reference_flat_matrix_is_onnxruntime``);
every vertex of every exact piece must lie in one returned set, decided
exactly with zero slack. (2) onnxruntime replay of 128 member points (box
corners, the quarter grid, points on the cut face of the constrained Star)
with LP membership against the rows of the result (torch in float64 for the
torch model). Weights and bounds are dyadic (in the float64 file, dyadic
plus multiples of 2**-30, still exact in float64), so the replay and the
reach are exact and no slack beyond the LP tolerance is used; the one cell
with non-dyadic data checks the Box only, against the rationals of its
stored numbers.

The input polytopes all carry the rows ``+-e_i`` for every coordinate, so
they are bounded; they are built as ``rational.Piece`` directly instead of
through ``rational.polytope``, whose boundedness proof enumerates
``C(4n, n)`` row subsets (seconds per call at six coordinates).

Layout: every tensor is flattened row-major in the network's own order and
every set is flat (an image set is refused by this operation).
"""

import itertools
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper
from onnx2torch.node_converters.matmul import OnnxMatMul

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Star
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_star, flat_box,
                                flat_set, image_set, pixel_box, reach_model,
                                replay_onnx, replay_torch, samples, save_onnx)
from tests.known_bugs import known_bug

# --------------------------------------------------------------------------- #
# Constants and forms                                                           #
# --------------------------------------------------------------------------- #
# Dyadic, mixed signs, no two columns alike, square ones not symmetric.
_W43 = np.array([[0.5, -1.0, 0.25],
                 [-0.75, 0.5, 1.0],
                 [1.0, 0.25, -0.5],
                 [0.25, -0.5, -1.0]], dtype=np.float32)
_W32 = np.array([[1.0, -0.5],
                 [0.25, 0.75],
                 [-1.0, 0.5]], dtype=np.float32)
_W33 = np.array([[0.5, -2.0, 1.0],
                 [3.0, 1.0, 0.0],
                 [-1.0, 2.5, -0.5]], dtype=np.float32)
_W23 = np.array([[0.5, -2.0, 1.0],
                 [3.0, 1.0, -0.25]], dtype=np.float32)
_W22 = np.array([[1.0, -0.5],
                 [0.75, 0.25]], dtype=np.float32)
_V3 = np.array([0.5, -1.0, 0.25], dtype=np.float32)

# name: (data shape, constant, side of the constant, output shape)
_FORMS = {
    "x(1,4)@W(4,3)": ((1, 4), _W43, "right", (1, 3)),
    "x(1,3)@W(3,3)": ((1, 3), _W33, "right", (1, 3)),
    "x(2,3)@W(3,2)": ((2, 3), _W32, "right", (2, 2)),
    "x(1,2,3)@W(3,2)": ((1, 2, 3), _W32, "right", (1, 2, 2)),
    "x(2,3)@w(3)": ((2, 3), _V3, "right", (2,)),
    "x(2,3)@W(1,3,2)": ((2, 3), _W32[None], "right", (1, 2, 2)),
    "W(3,3)@x(3,1)": ((3, 1), _W33, "left", (3, 1)),
    "W(2,3)@x(3,2)": ((3, 2), _W23, "left", (2, 2)),
    "W(2,3)@x(1,3,2)": ((1, 3, 2), _W23, "left", (1, 2, 2)),
    "w(3)@x(3,2)": ((3, 2), _V3, "left", (2,)),
    "W(1,2,3)@x(3,2)": ((3, 2), _W23[None], "left", (1, 2, 2)),
    "W(2,3)@x(1,2,3,1)": ((1, 2, 3, 1), _W23, "left", (1, 2, 2, 1)),
    # four-entry forms for the exact ReLU split (few pieces, cheap oracle)
    "x(2,2)@W(2,3)": ((2, 2), _W23, "right", (2, 3)),
    "W(3,2)@x(2,2)": ((2, 2), _W32, "left", (3, 2)),
}

_LEFT_WITH_HEADS = known_bug(
    7, "MatMul with a constant on the left and data with a leading axis longer "
       "than one (W @ x, x of shape (1, heads, K, N), heads > 1) is mapped as "
       "kron(W, I_{heads*N}) on the flat set instead of one copy of "
       "kron(W, I_N) per head, which mixes the heads: model outputs escape "
       "the result")

_KINDS = ("Box", "Zono", "Star", "Star-row", "Hexatope", "Octatope")
_ONE_MATMUL = [name for name in _FORMS
               if name not in ("x(2,2)@W(2,3)", "W(3,2)@x(2,2)")]
# The forms every set-type branch runs: the constant on each side, square
# and not symmetric (a dropped transpose changes the values) and with several
# rows or columns (the Kronecker expansion). The other forms differ only in
# code that runs before the set-type branch (1-D and 3-D constants, a leading
# axis of the data), so they run on the constrained Star alone.
_BRANCH_FORMS = ("x(1,3)@W(3,3)", "x(2,3)@W(3,2)", "W(3,3)@x(3,1)",
                 "W(2,3)@x(3,2)")


def _form_params(names):
    out = []
    for name in names:
        marks = (_LEFT_WITH_HEADS,) if name == "W(2,3)@x(1,2,3,1)" else ()
        out.append(pytest.param(name, marks=marks, id=name))
    return out


def _holds_params():
    """(form, kind) of the approx cells: every form on the constrained Star,
    the branch forms on every other flat kind. The head-axis defect is
    marked once, on the constrained Star (the same matrix reaches every
    set-type branch)."""
    out = []
    for name in _ONE_MATMUL:
        kinds = (("Box", "Zono", "Star-row", "Hexatope", "Octatope")
                 if name in _BRANCH_FORMS else ("Star-row",))
        for kind in kinds:
            marks = ((_LEFT_WITH_HEADS,) if name == "W(2,3)@x(1,2,3,1)"
                     else ())
            out.append(pytest.param(name, kind, marks=marks,
                                    id=f"{name}-{kind}"))
    return out


# --------------------------------------------------------------------------- #
# Reference helpers (no n2v)                                                    #
# --------------------------------------------------------------------------- #
def _flat_matrix(const, side, x_shape):
    """The matrix ``M`` with ``y.ravel() = M @ x.ravel()`` for
    ``y = numpy.matmul(const, x)`` (side left) or ``numpy.matmul(x, const)``
    (side right): column ``j`` is the product with the ``j``-th unit tensor.
    Exact for dyadic constants."""
    n = int(np.prod(x_shape))
    const = np.asarray(const, dtype=np.float64)
    columns = []
    for j in range(n):
        unit = np.zeros(n)
        unit[j] = 1.0
        unit = unit.reshape(x_shape)
        y = np.matmul(const, unit) if side == "left" else np.matmul(unit, const)
        columns.append(np.asarray(y, dtype=np.float64).reshape(-1))
    return np.stack(columns, axis=1)


def _identity_on(A, b):
    """The identity on ``{x : A x <= b}`` as one exact piece. The rows must
    hold ``+e_i`` and ``-e_i`` for every coordinate (checked here), which
    bounds the polytope, so the brute-force boundedness proof of
    ``rational.polytope`` is not needed."""
    A, b = rational.matrix(A), rational.vector(b)
    n = len(A[0])
    eye = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for row in eye:
        if row not in A or [-v for v in row] not in A:
            raise ValueError("the rows do not bound every coordinate")
    return [rational.Piece(A, b, eye, [Fraction(0)] * n)]


def _box_pieces(lo, hi):
    """The identity on the box [lo, hi], exact."""
    lo, hi = rational.vector(lo), rational.vector(hi)
    n = len(lo)
    eye = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    return _identity_on(eye + [[-v for v in row] for row in eye],
                        hi + [-v for v in lo])


def _input(kind, lo, hi, seed=0):
    """``(input set, member points, exact input pieces)`` for ``kind``.

    ``Star-row`` is a Star whose predicate box gives exactly [lo, hi], cut by
    one binding row (``constrained_star``); its exact pieces are its own
    predicate polytope mapped through its basis. Every other kind is the box
    [lo, hi] built by that kind's constructor."""
    if kind == "Star-row":
        star, points = constrained_star(lo, hi, seed=seed)
        V = np.asarray(star.V, dtype=np.float64)
        pieces = rational.affine(
            _identity_on(np.asarray(star.C, dtype=np.float64),
                         np.asarray(star.d, dtype=np.float64).reshape(-1)),
            V[:, 1:], V[:, 0])
        return star, points, pieces
    return flat_set(kind, lo, hi), samples(lo, hi), _box_pieces(lo, hi)


def _coupled_star(lo, hi):
    """``(star, member points, exact pieces)``: the box [lo, hi] as a Star
    ``x = c + diag(g) a``, ``a`` in [-1, 1], cut by one row per coordinate
    ``i`` that straddles zero, ``x_j + x_i <= c_j``, with ``j`` a coordinate
    that does not straddle. Where ``x_i`` is negative, ``x_j`` reaches
    beyond its value anywhere ``x_i`` is positive, so after a ReLU the piece
    in which every coordinate is active does not hold the images of the
    other pieces (on a plain box it does, and a MatMul that maps only that
    piece would go unseen). Member points: the predicate half grid, kept
    where every row holds exactly; all data dyadic."""
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    n = lo.size
    c, g = (lo + hi) / 2, (hi - lo) / 2
    straddle = [i for i in range(n) if lo[i] < 0 < hi[i]]
    others = [j for j in range(n) if j not in straddle]
    if not straddle or not others or np.any(g == 0):
        raise ValueError("needs straddling and other coordinates of width")
    cuts, rhs = [], []
    for t, i in enumerate(straddle):
        j = others[t % len(others)]
        row = np.zeros(n)
        row[i], row[j] = g[i], g[j]
        cuts.append(row)
        rhs.append(-c[i])            # g_i a_i + g_j a_j <= -c_i
    C = np.vstack([np.eye(n), -np.eye(n), np.array(cuts)])
    d = np.concatenate([np.ones(2 * n), rhs])
    V = np.hstack([c[:, None], np.diag(g)])
    star = Star(V, C, d.reshape(-1, 1), -np.ones((n, 1)), np.ones((n, 1)))
    Cq, dq = rational.matrix(C), rational.vector(d)
    points = []
    for a in itertools.product(*[(-1.0, -0.5, 0.0, 0.5, 1.0)] * n):
        aq = rational.vector(a)
        if all(rational.dot(row, aq) <= r for row, r in zip(Cq, dq)):
            points.append(c + g * np.array(a))
    pieces = rational.affine(_identity_on(C, d), V[:, 1:], V[:, 0])
    return star, points, pieces


def _vertices_of(computed):
    """Points whose hull is a computed set (``rational.star`` form): the
    images of the vertices of its predicate polytope. When every predicate
    variable has two finite bounds the polytope is bounded and its vertices
    are enumerated directly; otherwise ``rational.set_vertices`` decides.
    A row that repeats another (a Star keeps its box rows in ``C`` and in
    its predicate bounds) is kept once with the smaller right-hand side and
    a zero row that every point meets is dropped: the same polytope, with
    far fewer row subsets for the brute-force enumeration."""
    c, G, C, d, lb, ub = computed
    if not lb or any(v is None for v in list(lb) + list(ub)):
        return rational.set_vertices(computed)
    n = len(lb)
    eye = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    tightest = {}
    for row, rhs in zip([list(row) for row in C] + eye
                        + [[-v for v in row] for row in eye],
                        list(d) + list(ub) + [-v for v in lb]):
        key = tuple(row)
        if not any(key) and rhs >= 0:
            continue
        if key not in tightest or rhs < tightest[key]:
            tightest[key] = rhs
    A = [list(key) for key in tightest]
    b = [tightest[key] for key in tightest]
    return sorted({tuple(rational.dot(row, a) + ci for row, ci in zip(G, c))
                   for a in rational.vertices(A, b)})


# The inputs below have at most 5 predicate variables and give at most 4
# returned sets; a result with more predicate variables or many more sets is
# wrong, and its vertex enumeration or membership checks would run for a very
# long time instead of failing.
_MAX_VARS = 6
_MAX_SETS = 16


def _assert_exact_image_inside(pieces, sets, what):
    """Every exact piece has all its vertices in one returned set (exact, zero
    slack): the returned sets contain the image. A piece whose vertices lie
    in different returned sets proves nothing and fails too. Each piece's
    image vertices are listed once, and each returned set is decided by one
    ``rational.Membership``, which reuses its certificates and checks each
    one again, exactly, for the point at hand."""
    if len(sets) > _MAX_SETS:
        raise AssertionError(f"{what}: {len(sets)} returned sets, more than "
                             f"any input of this file can need")
    members = [rational.Membership(rational.star(s)) for s in sets]
    for i, piece in enumerate(pieces):
        image = piece.image()
        if any(all(member(y) for y in image) for member in members):
            continue
        outside = [y for y in image if not any(m(y) for m in members)]
        if outside:
            raise AssertionError(
                f"{what}: {len(outside)} vertices of exact piece {i} are in no "
                f"returned set, e.g. {[float(v) for v in outside[0]]}")
        raise AssertionError(f"{what}: exact piece {i} has its vertices in "
                             f"different returned sets, so the check proves "
                             f"nothing")


def _assert_nothing_unreached(pieces, sets, what):
    """Every vertex of every returned set has a preimage (the exact method
    claims the exact image). Fails before the enumeration when a returned
    set has more predicate variables than an affine map of these inputs
    can give."""
    for k, s in enumerate(sets):
        computed = rational.star(s)
        if len(computed[4]) > _MAX_VARS:
            raise AssertionError(
                f"{what}: returned set {k} has {len(computed[4])} predicate "
                f"variables; an affine map of these inputs keeps at most "
                f"{_MAX_VARS}")
        far = rational.unreached(pieces, _vertices_of(computed))
        assert not far, (f"{what}: returned set {k} has {len(far)} vertices "
                         f"outside the exact image, e.g. "
                         f"{[float(v) for v in far[0][1]]}")


def _assert_one_set(sets, what):
    """MatMul with a constant is an affine map: one input set gives one
    output set (checked before the membership work, whose cost grows with
    the number of sets)."""
    if len(sets) != 1:
        raise AssertionError(f"{what}: {len(sets)} returned sets for one "
                             f"input set of an affine map")


def _matmul_model(tmp_path, name):
    x_shape, const, side, y_shape = _FORMS[name]
    inputs = ["W", "X"] if side == "left" else ["X", "W"]
    path = save_onnx(tmp_path, [helper.make_node("MatMul", inputs, ["Y"])],
                     x_shape, y_shape, {"W": const}, name="matmul")
    return path, x_shape, _flat_matrix(const, side, x_shape)


# --------------------------------------------------------------------------- #
# One MatMul, every set kind, approx                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("form,kind", _holds_params())
def test_constant_matmul_holds_the_exact_image(tmp_path, form, kind):
    """MatMul with a constant operand under approx on a box with all three
    sign regimes and a zero-width entry (or a Star cut by a binding row).
    Every form runs on the constrained Star; the four branch forms run on
    the Box, Zono, Hexatope and Octatope too (``_BRANCH_FORMS``).

    Catches: the transpose of the weight dropped on either side (a square
    non-symmetric W on the left and on the right, a non-square W on the
    right), the Kronecker factors in the wrong order for data with several
    rows or columns or a head axis (a layout error), a 1-D constant promoted
    to the wrong side, a leading axis of length one not removed, a Box mapped
    through its two corners only (mixed-sign weights), the Hexatope and
    Octatope branch handing the wrong matrix to their linear rule, and a
    wrong row of the Star (the cut row binds, so a bounding-box check would
    not see it)."""
    path, x_shape, M = _matmul_model(tmp_path, form)
    n = int(np.prod(x_shape))
    lo, hi = flat_box(n, seed=1, zero_width=n >= 4)
    input_set, points, pieces = _input(kind, lo, hi)
    what = f"MatMul {form} on {kind} (approx)"
    sets = reach_model(load_onnx(path), input_set, "approx", what,
                       input_shape=x_shape)
    _assert_one_set(sets, what)
    _assert_exact_image_inside(rational.affine(pieces, M), sets, what)
    assert_covered(sets, replay_onnx(path, points, x_shape), what)


# --------------------------------------------------------------------------- #
# exact on a Star: the image in both directions                                 #
# --------------------------------------------------------------------------- #
# The handler has no branch of its own for ``exact``; these forms run the
# constant on each side with one and with several rows or columns. The 1-D
# constant and the head axis differ only before the set-type branch and run
# under approx (``test_constant_matmul_holds_the_exact_image``).
_EXACT_FORMS = ["x(1,3)@W(3,3)", "x(2,3)@W(3,2)", "W(3,3)@x(3,1)",
                "W(2,3)@x(3,2)"]


@pytest.mark.parametrize("kind", ["Star", "Star-row"])
@pytest.mark.parametrize("form", _form_params(_EXACT_FORMS))
def test_constant_matmul_exact_is_the_exact_image(tmp_path, form, kind):
    """Under ``exact`` a MatMul with a constant is an affine map and must
    return its exact image: no true output outside, and no vertex of the
    result without a preimage (a result that is too large, say the image
    under the transposed weight joined with the right one, fails here)."""
    path, x_shape, M = _matmul_model(tmp_path, form)
    n = int(np.prod(x_shape))
    lo, hi = flat_box(n, seed=2, zero_width=n >= 4)
    input_set, points, pieces = _input(kind, lo, hi, seed=1)
    what = f"MatMul {form} on {kind} (exact)"
    sets = reach_model(load_onnx(path), input_set, "exact", what,
                       input_shape=x_shape)
    _assert_one_set(sets, what)
    image = rational.affine(pieces, M)
    _assert_exact_image_inside(image, sets, what)
    _assert_nothing_unreached(image, sets, what)
    assert_covered(sets, replay_onnx(path, points, x_shape), what)


# --------------------------------------------------------------------------- #
# Several input pieces: an exact ReLU split ahead of the MatMul                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Star", "Star-row", "Star-coupled"])
@pytest.mark.parametrize("form", ["x(2,2)@W(2,3)", "W(3,2)@x(2,2)"])
def test_constant_matmul_after_an_exact_split_maps_every_piece(tmp_path, form,
                                                               kind):
    """Relu then MatMul under ``exact`` on a (2, 2) input with two
    coordinates straddling zero: the MatMul receives one Star per activation
    pattern (four) and must map every one of them. Catches a loop over the
    input pieces that keeps only the first piece (any input) or only the
    last one, the piece in which every coordinate is active (the
    ``Star-coupled`` input, whose cuts make the pieces not nested), a piece
    mapped by the wrong matrix, and, on both sides of the constant, a result
    that is not the exact image in either direction."""
    x_shape, const, side, y_shape = _FORMS[form]
    inputs = ["W", "H"] if side == "left" else ["H", "W"]
    path = save_onnx(tmp_path, [helper.make_node("Relu", ["X"], ["H"]),
                                helper.make_node("MatMul", inputs, ["Y"])],
                     x_shape, y_shape, {"W": const}, name="relu_matmul")
    n = int(np.prod(x_shape))
    lo, hi = flat_box(n, seed=3)
    if kind == "Star-coupled":
        input_set, points, pieces = _coupled_star(lo, hi)
    else:
        input_set, points, pieces = _input(kind, lo, hi, seed=2)
    what = f"Relu then MatMul {form} on {kind} (exact)"
    sets = reach_model(load_onnx(path), input_set, "exact", what,
                       input_shape=x_shape)
    image = rational.affine(rational.relu(pieces),
                            _flat_matrix(const, side, x_shape))
    if len(image) < 2:
        pytest.fail(f"{what}: the input must split into several pieces")
    _assert_exact_image_inside(image, sets, what)
    _assert_nothing_unreached(image, sets, what)
    assert_covered(sets, replay_onnx(path, points, x_shape), what)


# --------------------------------------------------------------------------- #
# Composition                                                                   #
# --------------------------------------------------------------------------- #
_BIAS_IN = (np.arange(6).reshape(2, 3) % 4 / 4.0 - 0.25).astype(np.float32)
_BIAS_OUT = np.array([[0.5, -0.25], [-1.0, 0.75]], dtype=np.float32)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("side", ["right", "left"])
def test_constant_matmul_between_two_affine_nodes(tmp_path, kind, side):
    """Add, MatMul, Add on a (2, 3) tensor (right: x @ W(3,2); left:
    W(2,2) @ x): the rows the MatMul emits are carried through the second
    Add. Catches a MatMul that reads the wrong node's sets (the Add before
    it) or whose result the next node does not receive."""
    if side == "right":
        nodes = [helper.make_node("Add", ["X", "B"], ["A"]),
                 helper.make_node("MatMul", ["A", "W"], ["M"]),
                 helper.make_node("Add", ["M", "C"], ["Y"])]
        inits = {"B": _BIAS_IN, "W": _W32, "C": _BIAS_OUT}
        y_shape, M = (2, 2), _flat_matrix(_W32, "right", (2, 3))
        bias_out = _BIAS_OUT
    else:
        nodes = [helper.make_node("Add", ["X", "B"], ["A"]),
                 helper.make_node("MatMul", ["W", "A"], ["M"]),
                 helper.make_node("Add", ["M", "C"], ["Y"])]
        bias_out = (np.arange(6).reshape(2, 3) % 3 / 2.0 - 0.5).astype(
            np.float32)
        inits = {"B": _BIAS_IN, "W": _W22, "C": bias_out}
        y_shape, M = (2, 3), _flat_matrix(_W22, "left", (2, 3))
    path = save_onnx(tmp_path, nodes, (2, 3), y_shape, inits, name="chain")
    lo, hi = flat_box(6, seed=4, zero_width=True)
    input_set, points, pieces = _input(kind, lo, hi, seed=3)
    what = f"Add, MatMul ({side} constant), Add on {kind}"
    sets = reach_model(load_onnx(path), input_set, "approx", what,
                       input_shape=(2, 3))
    image = rational.affine(rational.affine(pieces, np.eye(6),
                                            _BIAS_IN.reshape(-1)),
                            M, bias_out.reshape(-1))
    _assert_exact_image_inside(image, sets, what)
    assert_covered(sets, replay_onnx(path, points, (2, 3)), what)


@pytest.mark.parametrize("kind", ["Box", "Zono", "Star", "Star-row",
                                  "Hexatope", "Octatope"])
def test_relu_between_two_constant_matmuls_reads_the_rows(tmp_path, kind):
    """x(2,2) @ W(2,2), approx ReLU, then W(3,2) @ h. The ReLU relaxation is
    built from the ranges of the first MatMul's output, and its rows are
    mapped by the second, so a wrong row or range of the first MatMul shows
    downstream. The second MatMul receives a Star with more predicate
    variables and a Zono or Hexatope with more generators than entries, and
    on a Hexatope or Octatope two sets (their approx ReLU returns two here),
    so a branch that maps only one of several sets fails too. Reference: the
    exact image (ReLU pieces in rationals), each piece inside one returned
    set."""
    path = save_onnx(tmp_path, [helper.make_node("MatMul", ["X", "W1"], ["P"]),
                                helper.make_node("Relu", ["P"], ["H"]),
                                helper.make_node("MatMul", ["W2", "H"], ["Y"])],
                     (2, 2), (3, 2), {"W1": _W22, "W2": _W32}, name="mlp")
    lo, hi = flat_box(4, seed=5)
    input_set, points, pieces = _input(kind, lo, hi, seed=4)
    what = f"MatMul, Relu, MatMul on {kind} (approx)"
    sets = reach_model(load_onnx(path), input_set, "approx", what,
                       input_shape=(2, 2))
    image = rational.affine(
        rational.relu(rational.affine(pieces,
                                      _flat_matrix(_W22, "right", (2, 2)))),
        _flat_matrix(_W32, "left", (2, 2)))
    if len(image) < 2:
        pytest.fail(f"{what}: the hidden layer must have a crossing neuron")
    _assert_exact_image_inside(image, sets, what)
    assert_covered(sets, replay_onnx(path, points, (2, 2)), what)


@pytest.mark.parametrize("kind", ["Box", "Zono", "Star-row"])
def test_constant_matmul_whose_input_also_feeds_a_residual_add(tmp_path, kind):
    """X @ W + X on a (2, 3) tensor, W square and not symmetric: the input's
    sets are read by the MatMul and read again by the Add. Catches a handler
    that writes its result into the list or the sets it was handed, so that
    the Add reads the product where the input belongs. The Hexatope and the
    Octatope are left out: the Add of two computed sets refuses them."""
    nodes = [helper.make_node("MatMul", ["X", "W"], ["M"]),
             helper.make_node("Add", ["M", "X"], ["Y"])]
    path = save_onnx(tmp_path, nodes, (2, 3), (2, 3), {"W": _W33},
                     name="residual")
    lo, hi = flat_box(6, seed=9, zero_width=True)
    input_set, points, pieces = _input(kind, lo, hi, seed=7)
    what = f"X @ W + X on {kind}"
    sets = reach_model(load_onnx(path), input_set, "approx", what,
                       input_shape=(2, 3))
    _assert_one_set(sets, what)
    M = _flat_matrix(_W33, "right", (2, 3)) + np.eye(6)
    _assert_exact_image_inside(rational.affine(pieces, M), sets, what)
    assert_covered(sets, replay_onnx(path, points, (2, 3)), what)


# --------------------------------------------------------------------------- #
# A float64 file whose constant is not a float32 number, on a wide box          #
# --------------------------------------------------------------------------- #
# Large dyadic entries plus multiples of 2**-30: not float32 numbers, yet every
# float64 product and sum of these cells is exact, so the rational image is the
# reference with zero slack.
_TINY = 2.0 ** -30
_W64 = {
    "right": (np.array([[1536.25, -0.5], [-3.75, 2047.5], [0.125, -1024.0]])
              + _TINY * np.array([[1, -3], [5, 1], [-7, 2]])),
    "left": (np.array([[-1024.5, 3.25, 0.75], [0.5, -2048.0, 1536.125]])
             + _TINY * np.array([[3, -1, 2], [-5, 7, 1]])),
}
_W64_SHAPES = {"right": ((2, 3), (2, 2)), "left": ((3, 2), (2, 2))}


@pytest.mark.parametrize("side,kind,method", [
    ("right", "Box", "approx"), ("right", "Zono", "approx"),
    ("right", "Star-row", "approx"), ("right", "Star-row", "exact"),
    ("right", "Hexatope", "approx"), ("right", "Octatope", "approx"),
    ("left", "Star-row", "approx")])
def test_float64_constant_that_is_not_a_float32_number_on_a_wide_box(
        tmp_path, side, kind, method):
    """A float64 ONNX file whose MatMul constant has entries up to about 2e3
    that are not float32 numbers (a dyadic value plus a multiple of 2**-30),
    on the usual box times eight (bounds up to 12, widths up to 9). Catches
    the constant read through float32 on either side (in the handler, or in
    the stand-in Linear layer of the Hexatope and Octatope branch), which
    moves the image by about 1e-8 of its size, and a result that holds the
    outputs only while every coordinate stays inside [-1, 1]. The reference
    is the rational image with zero slack (every float64 product and sum is
    exact here), and the onnxruntime replay is in float64."""
    x_shape, y_shape = _W64_SHAPES[side]
    const = _W64[side]
    inputs = ["W", "X"] if side == "left" else ["X", "W"]
    path = save_onnx(tmp_path, [helper.make_node("MatMul", inputs, ["Y"])],
                     x_shape, y_shape, {"W": const}, name="matmul64",
                     in_type="DOUBLE", out_type="DOUBLE")
    lo, hi = flat_box(6, seed=11, zero_width=True)
    input_set, points, pieces = _input(kind, 8 * lo, 8 * hi, seed=5)
    what = f"float64 MatMul ({side} constant) on {kind} ({method})"
    sets = reach_model(load_onnx(path), input_set, method, what,
                       input_shape=x_shape)
    _assert_one_set(sets, what)
    image = rational.affine(pieces, _flat_matrix(const, side, x_shape))
    _assert_exact_image_inside(image, sets, what)
    if method == "exact":
        _assert_nothing_unreached(image, sets, what)
    assert_covered(sets, replay_onnx(path, points, x_shape), what)


def test_constant_matmul_on_a_box_holds_the_exact_image_of_non_dyadic_data(
        tmp_path):
    """The Box branch maps through ``Box.affine_map``, which rounds its
    bounds outward, so it must hold the exact image of the stored box under
    the stored float64 constant even when every product and sum rounds
    (thirds, sevenths, tenths). The reference is the image in rationals of
    exactly those numbers, with no slack. Catches a Box branch that rounds to
    nearest (a centre and radius form, or the interval sums without the
    outward move), which leaves about half of the bounds one unit inside."""
    const = np.array([[0.1, -1.0 / 3.0, 0.7], [2.0 / 3.0, 0.3, -1.0 / 7.0],
                      [-0.7, 1.0 / 7.0, 0.9], [1.0 / 3.0, -0.6, 0.2]])
    path = save_onnx(tmp_path, [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                     (1, 4), (1, 3), {"W": const}, name="matmul_box",
                     in_type="DOUBLE", out_type="DOUBLE")
    lo = np.array([-1.0 / 3.0, 0.2, -5.0 / 7.0, 0.1])
    hi = lo + np.array([1.0 / 3.0, 0.3, 2.0 / 7.0, 1.0 / 7.0])
    what = "float64 MatMul on a Box with non-dyadic data"
    sets = reach_model(load_onnx(path), flat_set("Box", lo, hi), "approx",
                       what, input_shape=(1, 4))
    _assert_one_set(sets, what)
    _assert_exact_image_inside(
        rational.affine(_box_pieces(lo, hi), _flat_matrix(const, "right",
                                                          (1, 4))),
        sets, what)
    assert_covered(sets, replay_onnx(path, samples(lo, hi), (1, 4)), what)


# --------------------------------------------------------------------------- #
# No tensor shape given                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Box", "Star-row"])
@pytest.mark.parametrize("form", ["x(1,4)@W(4,3)", "x(2,3)@W(3,2)",
                                  "W(3,3)@x(3,1)", "W(2,3)@x(3,2)"])
def test_constant_matmul_without_a_tensor_shape_refuses_or_holds_the_image(
        tmp_path, form, kind):
    """The model reached without ``input_shape``, so the tensor shape is
    known only where the forward runs on ``(1, n)`` or ``(n,)``: data with
    one row (known), data with several rows and the constant on the right
    (unknown: the handler falls back to the plain constant, which today fails
    the set's dimension check with ``ValueError``), and the constant on the
    left, whose column count is read from the set's size. The promise is a
    refusal (``NotImplementedError`` or ``ValueError``) or the exact image.
    Catches a fallback that guesses the missing shape and lays the rows out
    wrongly."""
    path, x_shape, M = _matmul_model(tmp_path, form)
    n = int(np.prod(x_shape))
    lo, hi = flat_box(n, seed=8, zero_width=n >= 4)
    input_set, points, pieces = _input(kind, lo, hi, seed=6)
    what = f"MatMul {form} on {kind} without input_shape"
    try:
        sets = reach_model(load_onnx(path), input_set, "approx", what)
    except (NotImplementedError, ValueError):
        return
    _assert_one_set(sets, what)
    _assert_exact_image_inside(rational.affine(pieces, M), sets, what)
    assert_covered(sets, replay_onnx(path, points, x_shape), what)


# --------------------------------------------------------------------------- #
# The other entry routes: a torch model, and reach_layer                        #
# --------------------------------------------------------------------------- #
class _TorchProduct(nn.Module):
    """``x @ W`` or ``W @ x`` with a constant buffer: the product as a torch
    model spells it, which fx traces to a call_function (an ``OnnxMatMul``
    inside a user module is traced the same way)."""

    def __init__(self, side):
        super().__init__()
        self.side = side
        self.register_buffer("W", torch.tensor(_W33))

    def forward(self, x):
        return x @ self.W if self.side == "right" else self.W @ x


@pytest.mark.parametrize("side", ["right", "left"])
def test_matmul_in_a_torch_model_refuses_or_holds_the_image(side):
    """A torch model that multiplies by a constant has no reach rule today:
    the walker's guard for function nodes refuses it with
    ``NotImplementedError``. The promise is that refusal, or the exact image
    and every torch output held. Catches a rule for the torch spelling that
    maps through the constant untransposed or on the wrong side."""
    model = _TorchProduct(side)
    x_shape = (1, 3) if side == "right" else (3, 1)
    lo, hi = flat_box(3, seed=10)
    input_set, points, pieces = _input("Star-row", lo, hi, seed=8)
    what = f"torch {'x @ W' if side == 'right' else 'W @ x'} on a Star"
    try:
        sets = reach_model(model, input_set, "approx", what,
                           input_shape=x_shape)
    except NotImplementedError:
        return
    _assert_exact_image_inside(
        rational.affine(pieces, _flat_matrix(_W33, side, x_shape)), sets, what)
    assert_covered(sets, replay_torch(model, points, x_shape), what)


@pytest.mark.parametrize("kind", ["Star", "ImageStar", "Zono", "ImageZono",
                                  "Box", "Hexatope", "Octatope"])
def test_reach_layer_refuses_the_matmul_module(kind):
    """``reach_layer`` handed the ``OnnxMatMul`` module itself, which takes
    two operands and so has no meaning on one list of sets: it must refuse
    with ``NotImplementedError`` under both methods (the graph walker answers
    the operation before the dispatcher could see it). Catches a dispatcher
    that passes the module through as the identity or answers it with
    another layer's rule."""
    if kind.startswith("Image"):
        lo, hi = pixel_box(2, 3, 5)
        input_set = image_set(kind, lo, hi)
    else:
        lo, hi = flat_box(4, seed=12)
        input_set = flat_set(kind, lo, hi)
    for method in ("approx", "exact"):
        with pytest.raises(NotImplementedError):
            reach_layer(OnnxMatMul(), [input_set], method)


# --------------------------------------------------------------------------- #
# Refusals                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Star"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_constant_with_a_long_leading_axis_is_refused(tmp_path, kind, side):
    """A constant of shape (2, 3, 3) broadcasts a different matrix per
    leading index; the handler has no rule for it and must refuse rather
    than answer with one of the matrices. The check runs before the
    set-type branch, so one set type stands for all."""
    const = np.stack([_W33, _W33[::-1]]).astype(np.float32)
    inputs = ["W", "X"] if side == "left" else ["X", "W"]
    path = save_onnx(tmp_path, [helper.make_node("MatMul", inputs, ["Y"])],
                     (3, 3), (2, 3, 3), {"W": const}, name="matmul3d")
    lo, hi = flat_box(9, seed=6)
    with pytest.raises(NotImplementedError):
        reach_pytorch_model(load_onnx(path), flat_set(kind, lo, hi),
                            method="approx", input_shape=(3, 3))


@pytest.mark.parametrize("kind,method", [
    pytest.param("ImageStar", "approx", id="ImageStar"),
    pytest.param("ImageZono", "approx", id="ImageZono"),
    pytest.param("ImageStar", "exact", id="ImageStar-exact")])
def test_matmul_on_an_image_set_is_refused(tmp_path, kind, method):
    """x (1, 2, 3, 5) @ W (5, 4) on an image set: the handler stores no rule
    for an image layout and must refuse (an answer computed on the stored
    height-width-channel order would mix pixels), under either method (an
    exact reach of an ImageZono is refused by the model entry before the
    walk, ``test_entry_refuses_an_exact_reach_of_an_image_zono`` in
    ``test_soundness_graph_walk.py``). The tensor shape is given, as a runner
    gives it: without it an ImageZono has no tracked shape and a missing
    refusal would only show as a dimension error, while with it the Zono
    branch answers in the stored order and misses every output."""
    W = (np.arange(20).reshape(5, 4) % 3 - 1).astype(np.float32) / 2
    path = save_onnx(tmp_path, [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                     (1, 2, 3, 5), (1, 2, 3, 4), {"W": W}, name="image")
    lo, hi = pixel_box(2, 3, 5)
    with pytest.raises(NotImplementedError):
        reach_pytorch_model(load_onnx(path), image_set(kind, lo, hi),
                            method=method, input_shape=(2, 3, 5))


@pytest.mark.parametrize("given_shape,form", [
    pytest.param(True, "x(2,3)@W(3,2)", id="True-x(2,3)@W(3,2)"),
    pytest.param(True, "W(2,3)@x(3,2)", id="True-W(2,3)@x(3,2)"),
    pytest.param(False, "x(2,3)@W(3,2)", id="False-x(2,3)@W(3,2)")])
def test_input_set_of_the_wrong_size_is_refused(tmp_path, form, given_shape):
    """A Star of 5 entries on a model whose input has 6: the MatMul must not
    answer (a result of some size for a set that is not the model's input
    would be read as a verdict). The constant on the right meets the tracked
    shape (given) and the set's own dimension (not given); the constant on
    the left never reads the tracked shape, so one spelling stands for it."""
    path, x_shape, _ = _matmul_model(tmp_path, form)
    lo, hi = flat_box(5, seed=7)
    options = {"input_shape": x_shape} if given_shape else {}
    with pytest.raises(ValueError):
        reach_pytorch_model(load_onnx(path), flat_set("Star", lo, hi),
                            method="approx", **options)


# --------------------------------------------------------------------------- #
# The flat matrix of the reference itself                                       #
# --------------------------------------------------------------------------- #
def test_reference_flat_matrix_is_onnxruntime(tmp_path):
    """The reference matrix of every form reproduces onnxruntime on random
    dyadic points, so the exact image above is the model's own map."""
    rng = np.random.default_rng(0)
    for name in _FORMS:
        path, x_shape, M = _matmul_model(tmp_path, name)
        n = int(np.prod(x_shape))
        points = [rng.integers(-8, 9, size=n) / 8.0 for _ in range(4)]
        for x, y in zip(points, replay_onnx(path, points, x_shape)):
            exact = [sum((Fraction(float(m)) * Fraction(float(v))
                          for m, v in zip(row, x)), Fraction(0)) for row in M]
            assert [Fraction(float(v)) for v in y] == exact, name
