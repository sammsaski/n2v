"""Soundness of the check of nonlinear VNN-LIB 2.0 specifications
(``n2v/utils/verify_nonlinear.py``).

What the file promises
----------------------
- ``verify_nonlinear_reach`` answers ``'UNSAT'`` only when no pair ``(x, y)``
  that the input set and the reach sets allow makes every assertion true. In
  the default model the pairs are every ``x`` of the input set with every
  ``y`` of every reach set; with ``input_predicates_shared=True`` they are the
  ``y`` that keep the input's own predicate values in the reach star's
  prefix. So every interval rule must enclose its true range, every
  three-valued comparison and every ``and``/``or`` must answer "surely false"
  only when it is, and a set it cannot read must give UNKNOWN.
- ``network_keeps_input_predicates`` answers True only for a network whose
  approx and exact Star reach keeps the input predicates as the prefix of
  every output star (the premise of the shared model).
- ``falsify_nonlinear`` returns only a point of ``[lb, ub]`` whose output, on
  the forward it was given, makes every assertion true, and never a point
  whose output is not finite.
- Through the competition runner, the only public entry of these functions,
  a nonlinear instance is ``unsat`` only when no counterexample exists and
  ``sat`` only with a witness that holds on the onnxruntime replay.

The oracle
----------
The assertion text is read by a reader of its own (``_read``, ``_truth``):
s-expressions evaluated in exact Fractions, every number read as the exact
decimal it is written as. The tests write only dyadic numbers or the exact
decimal expansion of a float (``_lit``), so the n2v reading of a number and
the exact one agree. ``+`` and ``*`` are n-ary, ``-`` is unary or n-ary and
``/`` is n-ary, both read left to right. Every case that demands "not UNSAT"
carries a witness: predicate values of the reach set, checked against the
set's own bounds and rows, and an input point; the reader evaluates the
assertions there to true. The witnesses sit at the extreme of the asserted
expression wherever one exists, so an enclosure that is too narrow on that
side by any amount turns the case into UNSAT. Network outputs come from an
exact forward pass in Fractions on dyadic weights (``tests/_oracles/rational``
pieces for the ReLU networks, whose vertices also give the exact extremes),
from the forward the falsifier was given, or from onnxruntime for the runner.
"""

import os
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn import NeuralNetwork
from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.vnnlib2 import load_vnnlib_v2
from n2v.utils.verify_nonlinear import (
    falsify_nonlinear, network_keeps_input_predicates, verify_nonlinear_reach,
)
from tests._oracles import rational
from tests._oracles.membership import LP_TOL, distance
from tests.cell_support import replay_onnx, save_onnx
from tests.known_bugs import known_bug


# --------------------------------------------------------------------------- #
# The reader of the text (the oracle)                                           #
# --------------------------------------------------------------------------- #
def _read(text):
    """The text as nested lists of tokens."""
    stack, top = [], []
    for token in text.replace("(", " ( ").replace(")", " ) ").split():
        if token == "(":
            stack.append(top)
            top = []
        elif token == ")":
            done, top = top, stack.pop()
            top.append(done)
        else:
            top.append(token)
    return top


def _assertions(text):
    return [node[1] for node in _read(text) if node and node[0] == "assert"]


def _value(node, x, y):
    if isinstance(node, str):
        if node.startswith("X["):
            return x[int(node[2:-1])]
        if node.startswith("Y["):
            return y[int(node[2:-1])]
        return Fraction(node)
    op, args = node[0], [_value(a, x, y) for a in node[1:]]
    if op == "+":
        return sum(args, Fraction(0))
    if op == "-":
        return -args[0] if len(args) == 1 else args[0] - sum(args[1:], Fraction(0))
    if op in ("*", "/"):
        out = args[0] if op == "/" else Fraction(1)
        for a in (args[1:] if op == "/" else args):
            out = out / a if op == "/" else out * a
        return out
    raise ValueError(f"the reader has no arithmetic {op!r}")


def _truth(node, x, y):
    op = node[0]
    if op == "and":
        return all(_truth(c, x, y) for c in node[1:])
    if op == "or":
        return any(_truth(c, x, y) for c in node[1:])
    if len(node) != 3:
        raise ValueError(f"the reader compares two operands: {node}")
    a, b = _value(node[1], x, y), _value(node[2], x, y)
    return {"<=": a <= b, "<": a < b, ">=": a >= b, ">": a > b,
            "==": a == b, "!=": a != b}[op]


def _holds(text, x, y):
    """Do all assertions of ``text`` hold at the rational point (x, y)?"""
    x = [Fraction(v) for v in x]
    y = [Fraction(v) for v in y]
    return all(_truth(a, x, y) for a in _assertions(text))


def _lit(v):
    """The exact decimal of the float ``v`` (a dyadic number prints short)."""
    s = str(Decimal(float(v)))
    return s if ("." in s or "E" in s) else s + ".0"


def _text(box, n_out, *asserts):
    """A VNN-LIB 2.0 text with one input X (the box, one pair of bounds per
    entry) and one output Y of ``n_out`` entries, then ``asserts``."""
    head = [
        "(vnnlib-version <2.0>)",
        f"(declare-network N (declare-input X float32 [{len(box)}])"
        f" (declare-output Y float32 [{n_out}]))",
        "(assert (and " + " ".join(
            f"(>= X[{i}] {_lit(lo)}) (<= X[{i}] {_lit(hi)})"
            for i, (lo, hi) in enumerate(box)) + "))",
    ]
    return "\n".join(head + [f"(assert {a})" for a in asserts]) + "\n"


def _load(tmp_path, text, name="spec"):
    path = tmp_path / f"{name}.vnnlib"
    path.write_text(text)
    spec = load_vnnlib(str(path))
    if spec.get("format") != "nonlinear":
        pytest.fail(f"setup: the text was not read as nonlinear: {spec.get('format')}")
    return spec


def _spec_from_text(text):
    spec = load_vnnlib_v2("mem.vnnlib", text=text)
    if spec.get("format") != "nonlinear":
        pytest.fail(f"setup: the text was not read as nonlinear: {spec.get('format')}")
    return spec


def _input_star(spec):
    """The input set the runner builds from the parsed box."""
    lb = np.asarray(spec["lb"], dtype=np.float64).reshape(-1, 1)
    ub = np.asarray(spec["ub"], dtype=np.float64).reshape(-1, 1)
    return Star.from_bounds(lb, ub)


# --------------------------------------------------------------------------- #
# A hand-built reach set with known ranges                                      #
# --------------------------------------------------------------------------- #
# Predicate boxes away from [-1, 1] and of different widths, one predicate
# that no output reads with infinite bounds, and one binding row a2 + a3 <= 4.5.
_PLB = [0.0, -3.0, -1.0, 1.0, -2.0, -1.0, -2.0, 0.0, -np.inf]
_PUB = [2.0, 1.0, 3.0, 2.0, 2.0, 1.0, 2.0, 4.0, np.inf]
_V = np.zeros((8, 10))
for _row, _c, _coef in (
        (0, -2.0, {0: 1.5}),               # Y0 = -2 + 1.5 a0         in [-2, 1]
        (1, 1.0, {1: -1.0}),               # Y1 = 1 - a1              in [0, 4]
        (2, 1.0, {2: 0.5, 3: -0.25}),      # Y2 = 1 + a2/2 - a3/4     in [0, 2.25]
        (3, -3.0, {4: 0.5}),               # Y3 = -3 + a4/2           in [-4, -2]
        (4, 3.0, {5: 1.0}),                # Y4 = 3 + a5              in [2, 4]
        (5, 0.0, {6: 0.5}),                # Y5 = a6/2                in [-1, 1]
        (6, 0.0, {7: 0.25}),               # Y6 = a7/4                in [0, 1]
        (7, 2.0, {0: -1.5, 6: 0.5})):      # Y7 = -Y0 + Y5            in [-2, 3]
    _V[_row, 0] = _c
    for _j, _g in _coef.items():
        _V[_row, _j + 1] = _g
_C = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=np.float64)
_D = np.array([[4.5]])
_A_BASE = [1, 0, 0, 1, 0, 0, 0, 2, 0]
_X_BOX = [(-3.0, 5.0), (0.5, 0.5), (1.0, 2.0)]   # X1 pinned
_X_BASE = [0, Fraction(1, 2), Fraction(3, 2)]
_X_SQUARE = "(>= (* X[2] X[2]) 1.0)"   # a product: the text is nonlinear; true


def _output_star(shift_y0=0.0):
    V = _V.copy()
    V[0, 0] += shift_y0
    return Star(V=V, C=_C, d=_D, pred_lb=np.array(_PLB), pred_ub=np.array(_PUB))


def _point_of_star(alpha):
    """The exact output y = V [1; alpha] of predicate values ``alpha``, after
    checking that alpha lies in the predicate box and satisfies the row."""
    a = [Fraction(v) for v in alpha]
    for v, lo, hi in zip(a, _PLB, _PUB):
        if (np.isfinite(lo) and v < Fraction(lo)) or (np.isfinite(hi) and v > Fraction(hi)):
            pytest.fail(f"setup: the witness {alpha} leaves the predicate box")
    for row, rhs in zip(_C, _D[:, 0]):
        if sum(Fraction(c) * v for c, v in zip(row, a)) > Fraction(rhs):
            pytest.fail(f"setup: the witness {alpha} breaks a row of the star")
    return [Fraction(r[0]) + sum(Fraction(g) * v for g, v in zip(r[1:], a))
            for r in _V]


def _y_at(alpha_over):
    alpha = list(_A_BASE)
    for j, v in alpha_over.items():
        alpha[j] = Fraction(v)
    return _point_of_star(alpha)


def _witness(alpha_over, x_over):
    alpha = list(_A_BASE)
    for j, v in alpha_over.items():
        alpha[j] = Fraction(v)
    x = list(_X_BASE)
    for i, v in x_over.items():
        x[i] = Fraction(v)
    for v, (lo, hi) in zip(x, _X_BOX):
        if not Fraction(lo) <= v <= Fraction(hi):
            pytest.fail(f"setup: the witness input {x} leaves the box")
    return x, _point_of_star(alpha)


# (id, assertion, predicate values, input values): every assertion holds at
# its witness, which sits at the extreme of the asserted expression.
_FAMILY = [
    ("affine-negative-coefficient-max", "(>= Y[1] 4.0)", {1: -3}, {}),
    ("affine-negative-coefficient-min", "(<= Y[1] 0.0)", {1: 1}, {}),
    ("affine-two-predicates-max", "(>= Y[2] 2.25)", {2: 3, 3: 1}, {}),
    ("affine-two-predicates-min", "(<= Y[2] 0.0)", {2: -1, 3: 2}, {}),
    ("affine-shared-predicates-difference-max", "(>= (- Y[0] Y[7]) 3.0)",
     {0: 2, 6: -2}, {}),
    ("affine-shared-predicates-difference-min", "(<= (- Y[0] Y[7]) -5.0)",
     {0: 0, 6: 2}, {}),
    ("constant-left-times-affine", "(>= (* 4.0 Y[4]) 16.0)", {5: 1}, {}),
    ("affine-times-negative-constant-right", "(<= (* Y[4] -0.5) -2.0)",
     {5: 1}, {}),
    ("product-mixed-signs-min", "(<= (* Y[0] Y[1]) -8.0)", {0: 0, 1: -3}, {}),
    ("product-mixed-signs-max", "(>= (* Y[0] Y[1]) 4.0)", {0: 2, 1: -3}, {}),
    ("product-of-positives-max", "(>= (* Y[4] Y[4]) 16.0)", {5: 1}, {}),
    ("product-negative-by-positive-min", "(<= (* Y[3] Y[4]) -16.0)",
     {4: -2, 5: 1}, {}),
    ("square-of-a-straddling-output-min", "(<= (* Y[5] Y[5]) 0.0)", {6: 0}, {}),
    ("three-factors-constant-first", "(>= (* 2.0 Y[4] Y[1]) 32.0)",
     {5: 1, 1: -3}, {}),
    ("sum-of-three-terms-with-a-product", "(>= (+ (* Y[0] Y[1]) Y[2] Y[4]) 10.25)",
     {0: 2, 1: -3, 2: 3, 3: 1, 5: 1}, {}),
    ("difference-of-three-terms", "(>= (- (* Y[0] Y[1]) Y[3] Y[5]) 9.0)",
     {0: 2, 1: -3, 4: -2, 6: -2}, {}),
    ("unary-minus-of-a-product", "(<= (- (* Y[4] Y[4])) -16.0)", {5: 1}, {}),
    ("negative-constant-after-a-product",
     "(>= (+ (* Y[0] Y[1] -2.0) Y[2]) 18.25)", {0: 0, 1: -3, 2: 3, 3: 1}, {}),
    ("product-on-the-right-of-a-comparison", "(<= Y[6] (* Y[0] Y[1]))",
     {7: 0, 0: 2, 1: -3}, {}),
    ("product-divided-by-a-constant", "(>= (/ (* Y[0] Y[1]) 0.5) 8.0)",
     {0: 2, 1: -3}, {}),
    ("affine-divided-by-a-constant", "(>= (/ Y[4] 0.25) 16.0)", {5: 1}, {}),
    ("affine-divided-by-a-negative-constant", "(<= (/ Y[4] -0.5) -8.0)", {5: 1}, {}),
    ("divided-by-a-positive-output-plus-an-output",
     "(>= (+ (/ 1.0 Y[4]) Y[6]) 1.5)", {5: -1, 7: 4}, {}),
    ("divided-by-a-negative-output-min", "(<= (/ 1.0 Y[3]) -0.5)", {4: 2}, {}),
    ("divided-by-a-negative-output-max", "(>= (/ 1.0 Y[3]) -0.25)", {4: -2}, {}),
    ("divided-by-a-straddling-output-times-an-output",
     "(>= (* (/ Y[0] Y[5]) Y[6]) 2.0)", {0: 2, 6: Fraction(1, 2), 7: 4}, {}),
    ("divided-by-an-output-touching-zero", "(>= (/ 1.0 Y[6]) 4.0)", {7: 1}, {}),
    ("ge-at-the-maximum", "(>= Y[0] 1.0)", {0: 2}, {}),
    ("gt-below-the-maximum", "(> (* Y[0] Y[1]) 3.5)", {0: 2, 1: -3}, {}),
    ("lt-above-the-minimum", "(< (* Y[0] Y[1]) -7.5)", {0: 0, 1: -3}, {}),
    ("eq-inside-the-range", "(== (* Y[0] Y[1]) 2.0)", {0: 2, 1: -1}, {}),
    ("ne-everywhere", "(!= (* Y[4] Y[4]) 2.0)", {}, {}),
    ("and-of-two-surely-true-comparisons",
     "(and (>= (* Y[4] Y[4]) 3.0) (<= Y[0] 1.5))", {}, {}),
    ("or-of-an-undecided-and-a-surely-false-comparison",
     "(or (>= (* Y[0] Y[1]) 4.0) (>= (* Y[4] Y[4]) 17.0))", {0: 2, 1: -3}, {}),
    ("or-of-an-and", "(or (and (>= Y[0] 1.0) (<= Y[1] 0.0)) (>= Y[3] 5.0))",
     {0: 2, 1: 1}, {}),
    ("output-minus-input-min", "(<= (- Y[1] X[0]) -5.0)", {1: 1}, {0: 5}),
    ("input-plus-output-min", "(<= (+ X[0] Y[1]) -3.0)", {1: 1}, {0: -3}),
    ("input-times-output-max", "(>= (* X[0] Y[0]) 6.0)", {0: 0}, {0: -3}),
    ("output-at-the-index-of-a-pinned-input", "(<= (* Y[1] Y[1]) 0.0)",
     {1: 1}, {}),
    ("pinned-input-times-output", "(>= (* X[1] Y[4]) 2.0)", {5: 1}, {}),
]


# The NaN of 0 * inf is how _ival_mul detects an unbounded product; numpy
# warns on it, which in ordinary use is a warning, not an error, and the case
# judges the verdict.
_NAN_PRODUCT = pytest.mark.filterwarnings(
    "ignore:invalid value encountered in scalar multiply:RuntimeWarning")


# Every case runs with the default disjoint predicates; the cases that read X
# run again with input_predicates_shared=True, which this star (prefix bounds
# [0, 2] x [-3, 1], not the input's [-1, 1]^2) must answer by the fallback.
_FAMILY_RUNS = [
    pytest.param(*c[1:], shared, id=f"{c[0]}-{'shared-asked' if shared else 'disjoint'}",
                 marks=[_NAN_PRODUCT] if "straddling-output-times" in c[0] else [])
    for c in _FAMILY for shared in (False, True) if not shared or "X[" in c[1]]


@pytest.mark.parametrize("assertion, alpha, x, shared", _FAMILY_RUNS)
def test_unsat_only_when_no_pair_of_the_sets_satisfies_the_assertion(
        tmp_path, assertion, alpha, x, shared):
    """Each case asserts an expression at its true extreme over a hand-built
    reach star (or inside its range for == and !=), with a witness that the
    reader of the text confirms in Fractions, so ``verify_nonlinear_reach``
    must not answer UNSAT. The family covers every interval rule (each sign
    of an affine coefficient, outputs correlated through shared predicates, a
    constant on either side and of either sign, products of every sign
    pattern, n-ary sums, differences and products, unary minus, a product on
    the right of a comparison, division by a constant and by an output that
    is positive, negative, straddling or touching zero), every comparison at
    its boundary, the three-valued ``and``/``or``, an input with a pinned
    entry and outputs read beside inputs of the same index. It catches an
    enclosure narrowed on the witness's side by any amount (a missing min/max,
    a dropped operand, an inverted interval, a constant not applied to the
    coefficients, an unsorted reciprocal, a NaN product taken as zero), a
    comparison or connective that answers "surely false" too early, and a
    cached form of X[i] returned for Y[i]. The ``shared-asked`` runs of the
    cases that read X pass ``input_predicates_shared=True`` with a star whose
    prefix bounds differ from the input's, so the verifier must fall back to
    disjoint predicates; read from the star's prefix, X0 would range over
    [1, 9] and miss its minimum -3."""
    text = _text(_X_BOX, 8, _X_SQUARE, assertion)
    spec = _load(tmp_path, text)
    wx, wy = _witness(alpha, x)
    if not _holds(text, wx, wy):
        pytest.fail(f"setup: the assertions do not hold at the witness {wx}, {wy}")
    verdict = verify_nonlinear_reach([_output_star()], _input_star(spec), spec,
                                     input_predicates_shared=shared)
    assert verdict != "UNSAT", (
        f"UNSAT although x={[str(v) for v in wx]}, y={[str(v) for v in wy]} "
        f"of the sets satisfies {assertion}")


@pytest.mark.parametrize("position", [0, 1, 2])
def test_several_reach_sets_are_unsat_only_when_every_set_is_free(tmp_path, position):
    """An exact reach returns several stars. Two of three are provably free
    of ``Y0 * Y1 >= 4`` (Y0 shifted to [-3.5, -0.5], so the product is at
    most 0); the one at ``position`` holds the witness Y0 = 1, Y1 = 4. Catches
    a verdict taken from one star only (the last, or the first), and a
    verifier that keeps state from an earlier call: the free stars are
    checked alone first and must give UNSAT, then the list must not."""
    assertion = "(>= (* Y[0] Y[1]) 4.0)"
    text = _text(_X_BOX, 8, _X_SQUARE, assertion)
    spec = _load(tmp_path, text)
    wx, wy = _witness({0: 2, 1: -3}, {})
    if not _holds(text, wx, wy):
        pytest.fail("setup: the witness does not satisfy the assertions")
    free = _output_star(shift_y0=-1.5)
    iset = _input_star(spec)
    if verify_nonlinear_reach([free, free], iset, spec) != "UNSAT":
        pytest.fail("setup: the free stars are not proved free")
    sets = [free, free, free]
    sets[position] = _output_star()
    assert verify_nonlinear_reach(sets, iset, spec) != "UNSAT"
    assert verify_nonlinear_reach([_output_star()], iset, spec) != "UNSAT"


def _y_box():
    """The bounds of Y0..Y7 of the hand-built star."""
    lo = np.array([-2, 0, 0, -4, 2, -1, 0, -2], dtype=np.float64)
    hi = np.array([1, 4, 2.25, -2, 4, 1, 1, 3], dtype=np.float64)
    return lo, hi


def _unreadable_reach(kind):
    lo, hi = _y_box()
    if kind == "Box":
        return [Box(lo, hi)]
    if kind == "Zono":
        return [Zono.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))]
    if kind == "ImageStar":
        return [ImageStar.from_bounds(lo.reshape(2, 4, 1), hi.reshape(2, 4, 1),
                                      height=2, width=4, num_channels=1)]
    if kind == "Star-without-predicate-bounds":
        s = _output_star()
        rows = np.vstack([s.C, np.eye(9)[:8], -np.eye(9)[:8]])
        rhs = np.vstack([s.d, np.array(_PUB[:8]).reshape(-1, 1),
                         -np.array(_PLB[:8]).reshape(-1, 1)])
        return [Star(V=s.V, C=rows, d=rhs)]
    if kind == "free-Star-then-Box":
        return [_output_star(shift_y0=-1.5), Box(lo, hi)]
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["Box", "Zono", "ImageStar",
                                  "Star-without-predicate-bounds",
                                  "free-Star-then-Box"])
def test_a_reach_set_the_verifier_cannot_read_is_never_unsat(tmp_path, kind):
    """The verifier reads only Stars with predicate bounds. A Box, a Zono, an
    ImageStar or a Star whose predicates are bounded only by its rows, each
    holding the witness Y0 = 1, Y1 = 4 of ``Y0 * Y1 >= 4``, must not give
    UNSAT, also after a provably free Star in the same list. Catches a set
    that is skipped instead of answered UNKNOWN."""
    assertion = "(>= (* Y[0] Y[1]) 4.0)"
    text = _text(_X_BOX, 8, _X_SQUARE, assertion)
    spec = _load(tmp_path, text)
    if not _holds(text, _X_BASE, _y_at({0: 2, 1: -3})):
        pytest.fail("setup: the witness does not satisfy the assertions")
    assert verify_nonlinear_reach(_unreadable_reach(kind), _input_star(spec),
                                  spec) != "UNSAT"


# The input box in the network's (channel, height, width) order for a
# (2, 1, 2) image: X0 = (c0, w0), X1 = (c0, w1), X2 = (c1, w0), X3 = (c1, w1).
_IMG_LO = np.array([-1.0, -1.0, -5.0, 0.0])
_IMG_HI = np.array([2.0, 1.0, -4.0, 0.5])


def _input_set(kind):
    """Input sets ``_input_affine`` does not read as a flat coordinate box;
    each holds the point x = (1, 1, -4.5, 0.25) of the box above."""
    if kind == "Box":
        return Box(_IMG_LO, _IMG_HI)
    if kind == "two-channel-ImageStar":
        hwc = lambda v: v.reshape(2, 1, 2).transpose(1, 2, 0)      # noqa: E731
        return ImageStar.from_bounds(hwc(_IMG_LO), hwc(_IMG_HI),
                                     height=1, width=2, num_channels=2)
    if kind == "rotated-Star":        # x0 = a0 + a1, x1 = a0 - a1
        return Star(V=np.array([[0.0, 1.0, 1.0], [0.0, 1.0, -1.0],
                                [-4.5, 0.0, 0.0], [0.25, 0.0, 0.0]]),
                    pred_lb=-np.ones(2), pred_ub=np.ones(2))
    if kind == "Star-with-more-predicates-than-entries":   # x1 = (a1 + a4) / 2
        V = np.zeros((4, 6))
        V[:, 0] = [0.5, 0.0, -4.5, 0.25]
        V[0, 1], V[1, 2], V[2, 3], V[3, 4], V[1, 5] = 1.5, 0.5, 0.5, 0.25, 0.5
        return Star(V=V, pred_lb=-np.ones(5), pred_ub=np.ones(5))
    if kind == "Star-without-predicate-bounds":
        box = Star.from_bounds(_IMG_LO.reshape(-1, 1), _IMG_HI.reshape(-1, 1))
        return Star(V=box.V, C=np.vstack([np.eye(4), -np.eye(4)]), d=np.ones((8, 1)))
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["Box", "two-channel-ImageStar", "rotated-Star",
                                  "Star-with-more-predicates-than-entries",
                                  "Star-without-predicate-bounds"])
def test_an_input_set_that_is_not_a_coordinate_box_is_never_unsat(tmp_path, kind):
    """``(>= (+ X[1] Y[0]) 2.0)`` holds at x1 = 1 with Y0 = 1, and the point
    x = (1, 1, -4.5, 0.25) lies in each of these input sets. The verifier
    reads x only from a flat Star whose generators each scale one coordinate
    and which has predicate bounds; for any other input it must not answer
    UNSAT. The image set has two channels and entries of different ranges,
    so reading it in its stored (height, width, channel) order puts X2's
    range [-5, -4] in place of X1's. Catches an input set read as a box it
    is not: an image set flattened in the wrong order, the predicate columns
    of a coupled star taken as coordinates, missing predicate bounds read as
    zero."""
    box = list(zip(_IMG_LO.tolist(), _IMG_HI.tolist()))
    text = _text(box, 8, "(>= (* X[3] X[3]) 0.0)", "(>= (+ X[1] Y[0]) 2.0)")
    spec = _load(tmp_path, text)
    if not _holds(text, [1, 1, Fraction(-9, 2), Fraction(1, 4)], _y_at({0: 2, 1: -3})):
        pytest.fail("setup: the witness does not satisfy the assertions")
    assert verify_nonlinear_reach([_output_star()], _input_set(kind), spec) != "UNSAT"
    assert verify_nonlinear_reach([_output_star()], _input_set(kind), spec,
                                  input_predicates_shared=True) != "UNSAT"


def test_sharing_asked_for_a_star_with_fewer_predicates_than_the_input_falls_back(tmp_path):
    """``input_predicates_shared=True`` with an output star of one predicate
    while the input box has two: there is no prefix to share, so the
    verifier must fall back to disjoint predicates and, since
    ``(>= (+ X[0] Y[0]) 3.0)`` holds at x0 = 2, Y0 = 1, must not answer
    UNSAT. Catches a prefix compared or sliced past the star's own
    predicates (a crash or a misread x)."""
    text = _text([(-1.0, 2.0), (-1.0, 1.0)], 1, "(>= (* X[1] X[1]) 0.0)",
                 "(>= (+ X[0] Y[0]) 3.0)")
    spec = _load(tmp_path, text)
    if not _holds(text, [2, 0], [1]):
        pytest.fail("setup: the witness does not satisfy the assertions")
    one = Star(V=np.array([[0.0, 1.0]]), pred_lb=-np.ones(1), pred_ub=np.ones(1))
    iset = _input_star(spec)
    if iset.nVar != 2:
        pytest.fail("setup: the input set does not have two predicates")
    assert verify_nonlinear_reach([one], iset, spec,
                                  input_predicates_shared=True) != "UNSAT"


# --------------------------------------------------------------------------- #
# The shared model on real reach sets, and its premise                          #
# --------------------------------------------------------------------------- #
def _linear(W, b):
    W = np.asarray(W, dtype=np.float64)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
        lin.bias.copy_(torch.tensor(b, dtype=torch.float32))
    return lin


_W1 = [[1.5, -0.5, 0.25], [-1.0, 2.0, 0.5], [0.5, 1.0, -1.0]]
_B1 = [0.25, -0.5, 1.0]
_W2 = [[1.0, -0.75, 0.5], [-0.5, 1.25, 1.0]]
_B2 = [-0.25, 0.5]
_IN_BOX = [(-1.5, 2.5), (-3.0, 1.0), (0.5, 0.5)]   # X2 pinned


def _network(name):
    """(torch module, rational pieces of the same function) for dyadic
    networks that ``network_keeps_input_predicates`` accepts: an affine one
    (its output star has exactly the input's predicates) and two with
    unstable ReLUs (appended predicates), one of them with an Identity."""
    box = rational.box([lo for lo, _ in _IN_BOX], [hi for _, hi in _IN_BOX])
    if name == "affine":
        return nn.Sequential(_linear(_W2[:1] + [[0.75, -1.0, 2.0]], [0.5, -1.0])), \
            rational.affine(box, _W2[:1] + [[0.75, -1.0, 2.0]], [0.5, -1.0])
    if name == "linear-relu-linear":
        return nn.Sequential(_linear(_W1, _B1), nn.ReLU(), _linear(_W2, _B2)), \
            rational.affine(rational.relu(rational.affine(box, _W1, _B1)), _W2, _B2)
    if name == "linear-identity-relu":
        return nn.Sequential(_linear(_W1[:2], _B1[:2]), nn.Identity(), nn.ReLU()), \
            rational.relu(rational.affine(box, _W1[:2], _B1[:2]))
    raise ValueError(name)


def _extreme(pieces, coef_x, coef_y, want_max):
    """The exact extreme of coef_x . x + coef_y . f(x) over the input box and
    a point where it is attained: a linear function on each piece, so the
    extreme is at a vertex of a piece."""
    best = None
    for piece in pieces:
        for v in piece.corners():
            y = piece.value(v)
            val = sum(Fraction(c) * xi for c, xi in zip(coef_x, v)) + \
                sum(Fraction(c) * yi for c, yi in zip(coef_y, y))
            if best is None or (val > best[0] if want_max else val < best[0]):
                best = (val, list(v), list(y))
    return best


def _float_outward(q, up):
    f = float(q)
    while (Fraction(f) < q) if up else (Fraction(f) > q):
        f = float(np.nextafter(f, np.inf if up else -np.inf))
    return f


# (relation, coefficients on x, coefficients on y, side): the threshold is
# the exact extreme rounded outward to a float, so the witness stays one.
_RELATIONS = {
    "X0-plus-Y1-min": ("(<= (+ X[0] Y[1]) {t})", [1, 0, 0], [0, 1], "min"),
    "Y0-minus-X1-max": ("(>= (- Y[0] X[1]) {t})", [0, -1, 0], [1, 0], "max"),
    "Y0-minus-twice-X0-min": ("(<= (- Y[0] (* 2.0 X[0])) {t})", [-2, 0, 0], [1, 0], "min"),
}


@pytest.mark.parametrize("method, prepass", [("approx", "ibp"), ("exact", None)])
@pytest.mark.parametrize("relation", list(_RELATIONS))
@pytest.mark.parametrize("name", ["affine", "linear-relu-linear",
                                  "linear-identity-relu"])
def test_the_shared_model_on_a_real_reach_is_unsat_only_without_a_violation(
        tmp_path, name, relation, method, prepass):
    """A relation between X and Y asserted at its exact extreme over the
    input box (from the vertices of the network's linear pieces in Fractions)
    is violated at that vertex, so the verifier must not answer UNSAT with
    ``input_predicates_shared=True`` on an approx or exact reach of a network
    that ``network_keeps_input_predicates`` accepts. The input box has widths
    and centres away from [-1, 1] and a pinned entry, so the input has fewer
    predicates than entries. Catches x read from the wrong predicate columns
    (the appended ReLU ones, or the output's own), a prefix of the wrong
    length, and a coefficient row of x built from the wrong basis row."""
    model, pieces = _network(name)
    pattern, cx, cy, side = _RELATIONS[relation]
    val, vx, vy = _extreme(pieces, cx, cy, side == "max")
    threshold = _float_outward(val, up=(side == "min"))
    text = _text(_IN_BOX, len(pieces[0].c), "(>= (* X[0] X[0]) 0.0)",
                 pattern.format(t=_lit(threshold)))
    spec = _load(tmp_path, text)
    if not _holds(text, vx, vy):
        pytest.fail("setup: the extreme vertex does not violate the text")
    net = NeuralNetwork(model.eval())
    if network_keeps_input_predicates(net) is not True:
        pytest.fail("setup: the network is not accepted for sharing")
    iset = _input_star(spec)
    options = {"input_shape": (3,)}
    if prepass:
        options["precompute_bounds"] = prepass
    sets = net.reach(iset, method=method, **options)
    assert verify_nonlinear_reach(sets, iset, spec,
                                  input_predicates_shared=True) != "UNSAT"


def _predicate_values(box_star, x):
    """The predicate values of the point x in a box Star whose generator
    columns each scale one coordinate (``Star.from_bounds``)."""
    c, G = box_star.V[:, 0], box_star.V[:, 1:]
    values = []
    for j in range(G.shape[1]):
        i = int(np.nonzero(G[:, j])[0][0])
        values.append((x[i] - c[i]) / G[i, j])
    return values


def _pinned(s, prefix):
    """The star ``s`` with its first predicates fixed to ``prefix``, or None
    when ``prefix`` leaves the star's own bounds for them."""
    n = len(prefix)
    V = np.asarray(s.V, dtype=np.float64)
    plb = np.asarray(s.predicate_lb, dtype=np.float64).ravel()
    pub = np.asarray(s.predicate_ub, dtype=np.float64).ravel()
    p = np.asarray(prefix, dtype=np.float64)
    if np.any(p < plb[:n]) or np.any(p > pub[:n]):
        return None
    c = V[:, 0] + V[:, 1:n + 1] @ p
    C = np.asarray(s.C, dtype=np.float64).reshape(-1, s.nVar)
    d = np.asarray(s.d, dtype=np.float64).reshape(-1)
    return Star(V=np.hstack([c.reshape(-1, 1), V[:, n + 1:]]),
                C=C[:, n:], d=(d - C[:, :n] @ p).reshape(-1, 1),
                pred_lb=plb[n:], pred_ub=pub[n:])


@pytest.mark.parametrize("method, prepass", [("approx", None), ("approx", "ibp"),
                                             ("exact", "ibp")])
@pytest.mark.parametrize("name", ["affine", "linear-relu-linear",
                                  "linear-identity-relu"])
def test_a_network_accepted_for_sharing_keeps_the_input_predicates_as_prefix(
        name, method, prepass):
    """The premise of ``input_predicates_shared=True``, which the runner
    takes from ``network_keeps_input_predicates``: for every member point x
    of the input set, the true output f(x) lies in some output star with the
    star's first predicates fixed to x's own predicate values in the input
    set. Checked on the box corners and 120 dyadic points (exact outputs from
    the network's pieces, membership by one LP on the pinned star's rows).
    Catches a network wrongly refused (the Identity network) and a Linear,
    ReLU or Identity kernel that stops keeping the input predicates as the
    prefix: placing the relaxation predicates before the old ones keeps the
    reach set exact but fails here (measured), while only the prefix-bound
    guard of the verifier keeps that from a false UNSAT. The settings are the
    library default (approx) and the runner's (approx and exact with the
    interval pre-pass)."""
    model, pieces = _network(name)
    net = NeuralNetwork(model.eval())
    assert network_keeps_input_predicates(net) is True
    lo = np.array([a for a, _ in _IN_BOX])
    hi = np.array([b for _, b in _IN_BOX])
    iset = Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    n_pred = iset.nVar
    options = {"input_shape": (3,)}
    if prepass:
        options["precompute_bounds"] = prepass
    sets = net.reach(iset, method=method, **options)
    if any(s.nVar < n_pred for s in sets):
        pytest.fail("a reach star has fewer predicates than the input")
    rng = np.random.default_rng(5)
    points = [np.array([a, b, 0.5]) for a in (lo[0], hi[0]) for b in (lo[1], hi[1])]
    points += [np.array([lo[0] + (hi[0] - lo[0]) * k / 64, lo[1] + (hi[1] - lo[1]) * m / 64, 0.5])
               for k, m in rng.integers(0, 65, size=(124, 2))]
    missed = []
    for x in points:
        prefix = _predicate_values(iset, x)
        y = np.array([float(v) for v in rational.evaluate(pieces, [Fraction(v) for v in x])])
        stars = [p for p in (_pinned(s, prefix) for s in sets) if p is not None]
        best = min((distance(p, y) for p in stars), default=np.inf)
        if best > LP_TOL * max(1.0, float(np.max(np.abs(y)))):
            missed.append((x.tolist(), best))
    assert not missed, f"{len(missed)} outputs outside every pinned star: {missed[:3]}"


class _DataDependent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        if x.sum() > 0:                       # torch.fx cannot trace this
            return torch.softmax(self.fc(x), dim=-1)
        return self.fc(x)


class _ModelAttribute(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        self.head = nn.Sigmoid()

    def forward(self, x):
        return self.head(self.model(x))


class _MethodCall(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x).softmax(dim=-1)


@pytest.mark.parametrize("make", [
    lambda: nn.Sequential(nn.Linear(2, 2), nn.Sigmoid()),
    lambda: _DataDependent(),
    lambda: _ModelAttribute(),
    lambda: _MethodCall(),
    lambda: torch.fx.symbolic_trace(_MethodCall()),
], ids=["sigmoid-module", "untraceable", "submodule-named-model", "method-call",
        "GraphModule-method-call"])
def test_a_network_whose_reach_re_encodes_is_not_accepted_for_sharing(make):
    """Each network has a node whose star kernel does not keep the input
    predicates as the prefix (Softmax and Sigmoid re-encode from bounds; an
    untraceable forward cannot be vetted), so the helper must answer False:
    a True makes the runner and the relational join read fresh predicates as
    the input (a false UNSAT). Catches an untraceable model reported as
    keeping them, a plain module unwrapped by its ``model`` attribute, a
    method call or a module type outside Linear/ReLU/Identity let through,
    and a GraphModule (every ONNX-loaded model) read differently from a
    traced module. The Softmax module and a NeuralNetwork around a module
    with a ``model`` submodule are checked by
    ``test_verify_nonlinear_regressions.py::test_network_keeps_input_predicates_allowlist``."""
    assert network_keeps_input_predicates(make()) is False


# --------------------------------------------------------------------------- #
# falsify_nonlinear                                                             #
# --------------------------------------------------------------------------- #
def _sum_model(n, w=1.0):
    return _linear(np.full((1, n), w), [0.0])


def _overflow_model():
    """y0 = 1e20 (1e20 x), y1 = x: the real y0 is at most about 1e40, its
    float32 forward is inf for x above about 3.4e-2, while y1 stays finite."""
    return nn.Sequential(_linear([[1e20], [1.0]], [0.0, 0.0]),
                         _linear([[1e20, 0.0], [0.0, 1.0]], [0.0, 0.0]))


def _nan_model():
    """y = 1e20 h0 - 1e20 h1 with h0 = h1 = 1e20 x: the real output is 0, its
    float32 forward is inf - inf = NaN."""
    return nn.Sequential(_linear([[1e20], [1e20]], [0.0, 0.0]),
                         _linear([[1e20, -1e20]], [0.0]))


def _forward_flat(X):
    """A caller's batched forward that returns a flat (N,) array: y = 2 x0 - x1."""
    X = np.asarray(X, dtype=np.float64)
    return 2.0 * X[:, 0] - X[:, 1]


def _replay(model, x, shape):
    """The model's float32 output at x, as the default forward computes it."""
    with torch.no_grad():
        t = torch.tensor(np.asarray(x, np.float32).reshape((1,) + tuple(shape)))
        return model(t).numpy().astype(np.float64).reshape(-1)


# (id, model maker, forward or None, input shape, spec box, search box, assertion)
_FALSIFY = [
    ("corner-only-12-dims", lambda: _sum_model(12), None, (12,),
     [(0.0, 1.0)] * 12, None, "(>= (* Y[0] Y[0]) 144.0)"),
    ("sub-box-without-a-witness", lambda: _sum_model(1), None, (1,),
     [(0.0, 5.0)], [(1.0, 2.0)], "(>= (* Y[0] Y[0]) 6.25)"),
    ("float32-overflow-to-inf-in-one-entry", _overflow_model, None, (1,),
     [(0.5, 1.0)], None, "(>= (* Y[0] Y[0]) 1e90)"),
    ("float32-nan", _nan_model, None, (1,),
     [(0.5, 1.0)], None, "(!= (* Y[0] Y[0]) 0.0)"),
    ("image-shaped-input", lambda: nn.Sequential(nn.Flatten(), _sum_model(4, 0.5)),
     None, (1, 2, 2), [(-1.0, 1.0)] * 4, None, "(>= (* Y[0] Y[0]) 3.0)"),
    ("caller-forward-flat-output", None, _forward_flat, (2,),
     [(-1.0, 3.0), (-2.0, 0.5)], None, "(>= (* Y[0] Y[0]) 64.0)"),
]


@pytest.mark.parametrize("make, forward, shape, box, search, assertion",
                         [c[1:] for c in _FALSIFY], ids=[c[0] for c in _FALSIFY])
def test_falsify_returns_only_points_of_the_box_that_satisfy_the_text(
        make, forward, shape, box, search, assertion):
    """A returned point must lie in [lb, ub] (the box the caller asked for,
    narrower than the spec's in ``sub-box``) and its output on the forward
    the falsifier used (the caller's, or the model's float32 forward) must be
    finite and satisfy every assertion as the reader of the text evaluates
    it. Where a witness exists it is at a corner (``corner-only-12-dims``,
    where only the all-upper corner reaches the threshold, and the flat-output
    forward, where only (3, -2) does). Catches draws outside the box, a non-finite
    output accepted (inf passes ``>= 1e90`` while the other entry is finite,
    NaN passes ``!=``), a flat
    output read with the wrong layout, and a point returned with another
    point's output."""
    model = make() if make is not None else None
    text = _text(box, 2 if make is _overflow_model else 1,
                 "(>= (* X[0] X[0]) 0.0)", assertion)
    spec = _spec_from_text(text)
    lo = np.array([a for a, _ in (search or box)])
    hi = np.array([b for _, b in (search or box)])
    x = falsify_nonlinear(model, lo, hi, spec, shape, n_samples=200, seed=3,
                          forward=forward)
    if x is None:
        return
    x = np.asarray(x, dtype=np.float64)
    assert np.all(x >= lo) and np.all(x <= hi), f"the point {x} leaves [{lo}, {hi}]"
    y = (np.asarray(forward(x.reshape(1, -1)), dtype=np.float64).reshape(-1)
         if forward is not None else _replay(model, x, shape))
    assert np.all(np.isfinite(y)), f"the output {y} of the point {x} is not finite"
    assert _holds(text, x, y), f"the point {x} with output {y} does not satisfy the text"


# --------------------------------------------------------------------------- #
# The competition runner (the public route)                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def runner(monkeypatch):
    """The runner module with approx and exact reach as its sound methods,
    and the list of verdicts ``verify_nonlinear_reach`` returned during the
    test. The runner's own ``set_parallel``/``set_lp_solver`` calls run for
    real (one worker); it restores the configuration on exit."""
    monkeypatch.syspath_prepend(os.path.join(
        os.path.dirname(__file__), "..", "..", "examples", "Submission",
        "VNN_COMP2026"))
    import vnncomp_runner as R
    monkeypatch.setattr(R, "get_config", lambda *a, **k: {
        "reach_methods": [("approx", {}), ("exact", {})], "n_rand": 64})
    calls = []
    import n2v.utils.verify_nonlinear as vn
    real = vn.verify_nonlinear_reach

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out)
        return out
    monkeypatch.setattr(vn, "verify_nonlinear_reach", spy)
    return R, calls


def _gemm_net(tmp_path, layers, n_in, name):
    """An ONNX chain of Gemm (and Relu after each but the last) layers."""
    nodes, inits, src = [], {}, "X"
    for k, (W, b) in enumerate(layers):
        inits[f"W{k}"] = np.asarray(W, np.float32)
        inits[f"B{k}"] = np.asarray(b, np.float32)
        out = "Y" if k == len(layers) - 1 else f"h{k}"
        nodes.append(helper.make_node("Gemm", [src, f"W{k}", f"B{k}"], [out], transB=1))
        if k < len(layers) - 1:
            nodes.append(helper.make_node("Relu", [out], [f"r{k}"]))
            out = f"r{k}"
        src = out
    return save_onnx(tmp_path, nodes, [1, n_in], [1, len(layers[-1][1])],
                     inits=inits, name=name)


def _ce_blocks(witness):
    """The 2.0 witness text as {name: values}."""
    lines = [ln.strip() for ln in witness.splitlines() if ln.strip()]
    out, i = {}, 0
    while i < len(lines):
        name, _dtype, dims = lines[i].split(" ", 2)
        n = int(np.prod([int(d) for d in dims.strip("[]").split(",") if d.strip()]))
        out[name] = [float(v) for v in lines[i + 1:i + 1 + n]]
        i += 1 + n
    return out


def _check_runner_answer(res, text, onnx_path, box, what):
    """A ``sat`` must carry an input inside the box (zero tolerance, after
    the float32 reading the grader does) whose onnxruntime output satisfies
    the text; ``unsat`` is the caller's to judge."""
    if res["result"] == "sat":
        x = np.asarray(_ce_blocks(res["counterexample"])["X"], dtype=np.float32)
        for v, (lo, hi) in zip(x, box):
            assert lo <= float(v) <= hi, f"{what}: the witness {x} leaves the box"
        y = replay_onnx(onnx_path, [x.astype(np.float64)], [1, len(box)])[0]
        assert _holds(text, x.astype(np.float64), y), (
            f"{what}: the witness {x} with replayed output {y} does not satisfy the text")


_RUNNER_LAYERS = [(_W1, _B1), (_W2, _B2)]


@pytest.mark.parametrize("case", ["violated-on-a-level-set", "violated-on-a-region"])
def test_the_runner_answers_a_violated_nonlinear_instance_soundly(tmp_path, runner, case):
    """End to end through ``verify_instance``: a Gemm-Relu-Gemm ONNX network
    (which the runner shares predicates for) and a nonlinear text that is
    violated, on a measure-zero level set the sampler cannot hit (so the
    sound reach must run and must not prove it), or on a region of positive
    measure (so the falsifier finds it). The answer must not be ``unsat``,
    and a ``sat`` must hold on the onnxruntime replay. Catches the runner
    handing the verifier the wrong input set or a sharing flag its network
    does not earn, and a falsifier witness the runner emits unchecked."""
    onnx_path = _gemm_net(tmp_path, _RUNNER_LAYERS, 3, "gemm_relu_gemm")
    model, pieces = _network("linear-relu-linear")
    xs = [Fraction(1, 2), Fraction(-1), Fraction(1, 2)]
    y_at = rational.evaluate(pieces, xs)
    if case == "violated-on-a-level-set":
        relation = f"(== (- Y[0] (* X[0] X[1])) {_lit(float(y_at[0] - xs[0] * xs[1]))})"
    else:
        val, vx, vy = _extreme(pieces, [0, 0, 0], [1, 0], True)
        relation = f"(>= (* Y[0] Y[0]) {_lit(float(val * val) * 0.25)})"
    text = _text(_IN_BOX, 2, relation)
    if case == "violated-on-a-level-set" and not _holds(text, xs, y_at):
        pytest.fail("setup: the level-set point does not violate the text")
    spec_path = tmp_path / "spec.vnnlib"
    spec_path.write_text(text)
    R, reach_verdicts = runner
    res = R.verify_instance(onnx_path, str(spec_path), "nonlinear_demo", workers=1)
    assert res["result"] != "unsat", f"{case}: unsat for a violated instance"
    _check_runner_answer(res, text, onnx_path, _IN_BOX, case)
    if case == "violated-on-a-level-set" and res["result"] != "sat" \
            and not reach_verdicts:
        pytest.fail("setup: the sound reach stage never ran")


def test_the_runner_proves_a_relation_only_the_shared_model_decides(tmp_path, runner):
    """An affine Gemm network y0 = W0 . x + 0.25 and the text
    ``y0 - W0 . x >= 0.5``, which no input reaches: the difference is the
    constant 0.25. Only the shared model, which the runner earns for a Gemm
    network, sees the constant; the disjoint model sees y0 and x over
    separate predicates and an interval around it. So the runner must answer
    ``unsat``. Catches the sharing flag not handed on, a
    network wrongly refused for sharing, and the shared model falling back
    to disjoint predicates on an affine network."""
    W = [[1.5, -0.5, 0.25]]
    onnx_path = _gemm_net(tmp_path, [(W, [0.25])], 3, "affine")
    rel = "(>= (- Y[0] (+ (* 1.5 X[0]) (* -0.5 X[1]) (* 0.25 X[2]))) 0.5)"
    text = _text(_IN_BOX, 1, "(>= (* X[0] X[0]) 0.0)", rel)
    spec_path = tmp_path / "spec.vnnlib"
    spec_path.write_text(text)
    R, _ = runner
    res = R.verify_instance(onnx_path, str(spec_path), "nonlinear_demo", workers=1)
    assert res["result"] == "unsat", f"expected unsat, got {res['result']}"


# --------------------------------------------------------------------------- #
# Refusals                                                                      #
# --------------------------------------------------------------------------- #
def test_a_comparison_inside_arithmetic_is_refused(tmp_path):
    """``(+ (>= Y[0] 1.0) 2.0)`` mixes a comparison into arithmetic; the
    parser keeps it as written and the verifier must refuse it with
    ``ValueError`` rather than evaluate a truth value as a number."""
    text = _text(_X_BOX, 8, _X_SQUARE, "(>= (+ (>= Y[0] 1.0) 2.0) 0.0)")
    spec = _load(tmp_path, text)
    with pytest.raises(ValueError):
        verify_nonlinear_reach([_output_star()], _input_star(spec), spec)


def test_an_assertion_that_is_not_a_comparison_or_connective_is_refused():
    """The parser produces only ``and``, ``or`` and comparisons at the top of
    an assertion, but ``verify_nonlinear_reach`` takes any result dict: an
    assertion headed by anything else (here ``not``, which VNN-LIB 2.0 does
    not have) must be refused with ``ValueError``, not read as undecided or
    false."""
    spec = {"assertions": [("not", (">=", ("var", "out", 0), ("const", 1.0)))]}
    iset = Star.from_bounds(np.array([[0.0]]), np.array([[1.0]]))
    with pytest.raises(ValueError):
        verify_nonlinear_reach([_output_star()], iset, spec)


@known_bug(23, "verify_nonlinear_reach reads a spec output index beyond the "
               "reach set's dimension with a raw array index and crashes with "
               "IndexError instead of refusing the mismatched spec with ValueError",
           raises=IndexError)
def test_a_spec_reading_an_output_the_reach_set_does_not_have_is_refused(tmp_path):
    """The text declares 9 outputs and reads Y[8]; the reach set has 8
    entries. A spec whose shape does not fit the sets is an input that cannot
    be read: ``ValueError``, not an index error."""
    text = _text(_X_BOX, 9, _X_SQUARE, "(>= (* Y[8] Y[8]) 0.5)")
    spec = _load(tmp_path, text)
    with pytest.raises(ValueError):
        verify_nonlinear_reach([_output_star()], _input_star(spec), spec)


# --------------------------------------------------------------------------- #
# Defects of today's library                                                    #
# --------------------------------------------------------------------------- #
@known_bug(6, "verify_nonlinear_reach evaluates '/' on its first two operands "
              "only, while '+', '-' and '*' are n-ary left to right: "
              "(/ Y0 0.5 0.5) >= 3 is read as Y0 / 0.5 >= 3 and answered UNSAT "
              "although Y0 = 1 gives (1 / 0.5) / 0.5 = 4")
def test_a_division_with_more_than_two_operands_is_read_left_to_right_or_refused(tmp_path):
    """The parser keeps every operand of ``(/ Y[0] 0.5 0.5)``; read left to
    right like the n-ary ``-`` the verifier implements, it is 4 Y0, which
    reaches 4 at Y0 = 1. The verifier must refuse the text (``ValueError``)
    or not answer UNSAT."""
    text = _text(_X_BOX, 8, _X_SQUARE, "(>= (/ Y[0] 0.5 0.5) 3.0)")
    spec = _load(tmp_path, text)
    wx, wy = _witness({0: 2}, {})
    if not _holds(text, wx, wy):
        pytest.fail("setup: the witness does not satisfy the text")
    try:
        verdict = verify_nonlinear_reach([_output_star()], _input_star(spec), spec)
    except ValueError:
        return
    assert verdict != "UNSAT"


def _three_output_text(assertion):
    return _text([(0.0, 1.0)], 3, "(>= (* X[0] X[0]) 0.0)", assertion)


@known_bug(6, "verify_nonlinear_reach sums the interval of an affine output in "
              "round to nearest and takes the reciprocal of the computed lower "
              "end: a denominator whose exact range starts at 0 is computed as "
              "starting at 2^-54, so (>= (/ 1.0 Y0) 2^60) is answered UNSAT "
              "although every y in (0, 2^-60] is reachable")
def test_a_denominator_whose_exact_range_starts_at_zero_is_not_decided_by_rounding(tmp_path):
    """Y0 = 1 + 2^-54 a0 + (1 - 2^-53) a1 + 2^-54 a2 over [-1, 1]^3 (dyadic,
    exact minimum 0). At a = (-1, -1 + d, -1) with d = 2^-61 / (1 - 2^-53),
    Y0 = 2^-61 and 1 / Y0 = 2^61 >= 2^60."""
    g = [2.0 ** -54, 1.0 - 2.0 ** -53, 2.0 ** -54]
    star = Star(V=np.array([[1.0] + g, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
                pred_lb=-np.ones(3), pred_ub=np.ones(3))
    text = _three_output_text(f"(>= (/ 1.0 Y[0]) {_lit(2.0 ** 60)})")
    spec = _load(tmp_path, text)
    d = Fraction(1, 2 ** 61) / (1 - Fraction(1, 2 ** 53))
    a = [Fraction(-1), Fraction(-1) + d, Fraction(-1)]
    y0 = 1 + sum(Fraction(gi) * ai for gi, ai in zip(g, a))
    if not (y0 == Fraction(1, 2 ** 61) and _holds(text, [0], [y0, 0, 0])):
        pytest.fail("setup: the witness does not satisfy the text")
    assert verify_nonlinear_reach([star], _input_star(spec), spec) != "UNSAT"


@known_bug(6, "verify_nonlinear_reach multiplies interval ends in round to "
              "nearest and its outward margin is relative to the final "
              "interval: the product of three outputs 384 + 2^-45 a_i is "
              "computed as exactly 384^3, and (>= (* Y0 Y1 Y2) K) with K the "
              "largest float below the exact maximum is answered UNSAT")
def test_a_product_compared_with_the_float_below_its_exact_maximum_is_not_unsat(tmp_path):
    """Y_i = 384 + 2^-45 a_i (dyadic): the interval of each rounds to the
    point 384, the product to 384^3, while the exact maximum
    (384 + 2^-45)^3 lies above 384^3 + 2^-27 = K. The all-upper corner
    reaches K."""
    w = 2.0 ** -45
    V = np.zeros((3, 4))
    V[:, 0] = 384.0
    for i in range(3):
        V[i, i + 1] = w
    star = Star(V=V, pred_lb=-np.ones(3), pred_ub=np.ones(3))
    exact_max = (Fraction(384) + Fraction(w)) ** 3
    K = _float_outward(exact_max, up=False)
    text = _three_output_text(f"(>= (* Y[0] Y[1] Y[2]) {_lit(K)})")
    spec = _load(tmp_path, text)
    corner = [Fraction(384) + Fraction(w)] * 3
    if not (Fraction(K) > 384 ** 3 and _holds(text, [0], corner)):
        pytest.fail("setup: the corner does not reach K above 384^3")
    assert verify_nonlinear_reach([star], _input_star(spec), spec) != "UNSAT"


def test_an_affine_output_at_its_exact_maximum_is_kept_by_the_rounding_margin(tmp_path):
    """Y0 = 0.3 + 0.2 a0 + 0.2 a1 + 0.2 a2 (float data): the exact maximum
    is float(0.9) exactly, while the interval summed in round to nearest
    ends at 0.8999999999999999. Only the outward margin of the comparison
    keeps ``(>= Y[0] 0.9)`` from UNSAT; it is the passing neighbour of the
    two rounding defects above and catches a margin removed or made
    relative only."""
    star = Star(V=np.array([[0.3, 0.2, 0.2, 0.2], [0.0] * 4, [0.0] * 4]),
                pred_lb=-np.ones(3), pred_ub=np.ones(3))
    text = _three_output_text(f"(>= Y[0] {_lit(0.9)})")
    spec = _load(tmp_path, text)
    top = Fraction(0.3) + 3 * Fraction(0.2)
    if not (top == Fraction(0.9) and _holds(text, [0], [top, 0, 0])):
        pytest.fail("setup: the corner does not reach 0.9")
    assert verify_nonlinear_reach([star], _input_star(spec), spec) != "UNSAT"


@known_bug(5, "falsify_nonlinear's default forward casts the candidates to "
              "float32 and calls the model on them, so a float64 torch model "
              "raises RuntimeError (mat1 and mat2 must have the same dtype) "
              "instead of answering", raises=RuntimeError)
def test_falsify_answers_on_a_float64_model():
    """y = 2 x on [1, 2] as a float64 Linear; (* Y Y) >= 12.25 holds at the
    corner x = 2 (y = 4, 16 >= 12.25). The falsifier must answer, and the
    answer must be a witness."""
    model = _linear([[2.0]], [0.0]).double()
    text = _text([(1.0, 2.0)], 1, "(>= (* Y[0] Y[0]) 12.25)")
    spec = _spec_from_text(text)
    x = falsify_nonlinear(model, [1.0], [2.0], spec, (1,), n_samples=16, seed=0)
    assert x is not None and _holds(text, x, [2 * Fraction(float(x[0]))])
