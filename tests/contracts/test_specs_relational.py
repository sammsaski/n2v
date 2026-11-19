"""Relational and nonlinear VNN-LIB 2.0 specifications: the parse means what the text says.

Component: the relational and nonlinear half of ``n2v/utils/vnnlib2.py``.

* A **relational** specification names two networks, their inputs, the coupling
  between the inputs and a property over both outputs (``_load_relational``,
  ``_parse_relational_declarations``, ``_coupling_rows``). Promise: the stored
  joint box and coupling rows accept exactly the joint inputs the text allows,
  the stored property groups accept exactly the joint outputs the text calls
  unsafe, and each network's offsets and sizes place its tensors where the
  joint spaces have them. A region read too small or a property read too weak
  can turn a real counterexample into UNSAT; a region read too large or a
  property read too strong can give an invalid SAT.
* A **nonlinear** specification is kept as an assertion tree
  (``_load_nonlinear``, ``_resolve_ast``) together with a best-effort input box.
  ``evaluate_nonlinear`` / ``evaluate_ast`` decide whether a point satisfies the
  assertions: they are what a SAT answer is checked with, so a wrong True is a
  false witness accepted and a wrong False a real one thrown away. With
  ``dtype=np.float32`` they must evaluate as the VNN-COMP 2026 checker does:
  every value and every operation result in float32, left to right. The box is
  what the runner reaches over for UNSAT, so it must hold the input of every
  counterexample of the text.
* A **linear** 2.0 specification whose rows can accept a point the checker
  rejects (a strict comparison read as closed, arithmetic the checker does in
  float32) keeps its source assertions (``_keep_assertions_for_sat_checks``);
  rows and kept assertions together must accept exactly the counterexamples of
  the text.

The reference is ``_Text`` below: it reads the assertions from the text alone,
reads every number as the exact decimal it spells (as SMT-LIB does: ``0.1`` is
1/10) and evaluates in fractions, so a point on a bound is decided exactly. In
its float32 mode every variable and constant is rounded to float32 first and
every operation result after it, left to right. The parse's side is read in
fractions on the stored float64 numbers, never through n2v's own helpers.

Data are binary fractions with few bits wherever a case is not about rounding,
so float64 evaluation, float32 evaluation and the exact meaning coincide there.
"""

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pytest
import torch

from n2v.nn.relational import verify_relational
from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.vnnlib2 import evaluate_nonlinear, load_vnnlib_v2
from tests.known_bugs import DID_NOT_RAISE, known_bug


# =========================================================================== #
# The reference: what the text says                                            #
# =========================================================================== #
class _Undefined(Exception):
    """The text has no meaning here (an unknown name, a term of the wrong kind,
    a comparison with one operand). Only a refusal keeps the promise then."""


_COMPARISONS = {"<=", "<", ">=", ">", "==", "!="}
_ARITHMETIC = {"+", "-", "*", "/"}


def _sexprs(text):
    """Every top-level s-expression of ``text``, ``;`` comments removed."""
    tokens = " ".join(line.split(";")[0] for line in text.splitlines())
    tokens = tokens.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def read():
        nonlocal pos
        token = tokens[pos]
        pos += 1
        if token != "(":
            return token
        items = []
        while tokens[pos] != ")":
            items.append(read())
        pos += 1
        return items

    out = []
    while pos < len(tokens):
        out.append(read())
    return out


def _to_f32(q):
    """The float32 number nearest to ``q`` (correct for the sums, differences
    and products of two float32 numbers used here: they are exact in float64)."""
    return Fraction(float(np.float32(float(q))))


class _Text:
    """The meaning of a VNN-LIB 2.0 text, read from the text alone.

    A point is a counterexample exactly when every ``(assert F)`` holds. ``F``
    is built from ``and``, ``or``, ``not`` and the comparisons ``<= < >= > ==
    !=`` in prefix form (chained over more than two operands, as SMT-LIB
    chains them) or in infix form ``(a op b)``, over the arithmetic ``+ - * /``
    (left-associative over any number of operands, ``(- a)`` is the negation)
    of variables and numbers. Variables are looked up by their spelling
    (``XF[0,1]``); a name not given has no meaning, and so has a number where a
    formula is due or a formula where a number is due.

    ``mode="exact"``: numbers are exact decimals and every operation is exact.
    ``mode="f32"``: the checker's float32 evaluation, every variable and
    constant rounded to float32 and every operation result after it, left to
    right (the documented semantics of ``evaluate_ast(..., dtype=np.float32)``).
    """

    def __init__(self, text):
        self.asserts = [e[1] for e in _sexprs(text)
                        if isinstance(e, list) and e and e[0] == "assert"]

    def accepts(self, values, mode="exact", asserts=None):
        asserts = self.asserts if asserts is None else asserts
        return all(self._holds(a, values, mode) for a in asserts)

    def _holds(self, node, values, mode):
        if not isinstance(node, list) or not node:
            raise _Undefined(f"not a formula: {node!r}")
        head = node[0]
        if head == "and":
            return all(self._holds(c, values, mode) for c in node[1:])
        if head == "or":
            return any(self._holds(c, values, mode) for c in node[1:])
        if head == "not" and len(node) == 2:
            return not self._holds(node[1], values, mode)
        if isinstance(head, str) and head in _COMPARISONS:
            op, operands = head, node[1:]
        elif (len(node) == 3 and isinstance(node[1], str)
              and node[1] in _COMPARISONS):
            op, operands = node[1], [node[0], node[2]]
        else:
            raise _Undefined(f"no meaning for {node!r}")
        if len(operands) < 2:
            raise _Undefined(f"comparison with one operand: {node!r}")
        vals = [self._value(c, values, mode) for c in operands]
        if op == "!=":
            return all(a != b for a, b in itertools.combinations(vals, 2))
        rel = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
               ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
               "==": lambda a, b: a == b}[op]
        return all(rel(a, b) for a, b in zip(vals, vals[1:]))

    def _value(self, node, values, mode):
        rnd = _to_f32 if mode == "f32" else (lambda q: q)
        if isinstance(node, str):
            if node in values:
                return rnd(Fraction(values[node]))
            try:
                q = Fraction(node)
            except ValueError:
                raise _Undefined(f"unknown name {node!r}") from None
            return _to_f32(float(node)) if mode == "f32" else q
        if (not node or not isinstance(node[0], str)
                or node[0] not in _ARITHMETIC or len(node) < 2):
            raise _Undefined(f"not a number: {node!r}")
        head = node[0]
        args = [self._value(c, values, mode) for c in node[1:]]
        if head == "-" and len(args) == 1:
            return -args[0]
        acc = args[0]
        for a in args[1:]:
            if head == "/" and a == 0:
                raise _Undefined(f"division by zero in {node!r}")
            acc = rnd(acc + a if head == "+" else acc - a if head == "-"
                      else acc * a if head == "*" else acc / a)
        return acc


def _names(node):
    """Every atom of ``node`` that is not a number or an operator."""
    if isinstance(node, list):
        return set().union(*(_names(c) for c in node)) if node else set()
    if node in _COMPARISONS or node in _ARITHMETIC or node in (
            "and", "or", "not"):
        return set()
    try:
        Fraction(node)
        return set()
    except ValueError:
        return {node}


_F32_01 = float(np.float32(0.1))

_REFERENCE_SELF_TEST = [
    # (text, values, mode, accepted)
    ("(assert (<= Y[0] 0.5))", {"Y[0]": 0.5}, "exact", True),
    ("(assert (< Y[0] 0.5))", {"Y[0]": 0.5}, "exact", False),
    ("(assert (Y[0] > X[0]))", {"Y[0]": 0.25, "X[0]": 0.25}, "exact", False),
    ("(assert (Y[0] <= X[0]))", {"Y[0]": 0.5, "X[0]": 0.25}, "exact", False),
    ("(assert (>= (- XG[0] XF[0]) 0.25))", {"XG[0]": 0.5, "XF[0]": 0.25},
     "exact", True),
    ("(assert (== XG[0] XF[0]))", {"XG[0]": 0.25, "XF[0]": 0.5}, "exact",
     False),
    ("(assert (!= Y[0] Y[1]))", {"Y[0]": 0.5, "Y[1]": 0.5}, "exact", False),
    ("(assert (and (>= X[0] -1) (<= X[0] 1)))", {"X[0]": -1.25}, "exact",
     False),
    ("(assert (or (<= Y[0] -0.5) (>= Y[1] 1)))", {"Y[0]": 0, "Y[1]": 1},
     "exact", True),
    ("(assert (not (<= Y[0] 0)))", {"Y[0]": 0}, "exact", False),
    ("(assert (<= (* (- Y[0]) Y[1]) 0.25))", {"Y[0]": -0.5, "Y[1]": 1},
     "exact", False),
    ("(assert (<= (+ Y[0] Y[1] X[0]) 0.5))",
     {"Y[0]": 0.25, "Y[1]": 0.25, "X[0]": 0.25}, "exact", False),
    ("(assert (>= (- Y[0] Y[1] 0.5) -1))", {"Y[0]": 0, "Y[1]": 0.5}, "exact",
     True),
    ("(assert (<= (/ Y[0] Y[1] Y[2]) 1))",
     {"Y[0]": 1, "Y[1]": 1, "Y[2]": 0.5}, "exact", False),
    ("(assert (<= Y[0] Y[1] Y[2]))", {"Y[0]": 0.5, "Y[1]": 1, "Y[2]": 0.25},
     "exact", False),
    # decimals are exact: the double nearest to 0.1 is above 1/10
    ("(assert (<= XG[0] (+ XF[0] 0.1)))", {"XG[0]": 0.1, "XF[0]": 0.0},
     "exact", False),
    # float32: (1 + 2^-23) + 2^-24 rounds to 1 + 2^-22, so the sum is 2^-22
    ("(assert (<= (+ Y[0] Y[1] Y[2]) 0.000000178813934326171875))",
     {"Y[0]": 1 + 2.0**-23, "Y[1]": 2.0**-24, "Y[2]": -1.0}, "f32", False),
    ("(assert (<= (+ Y[0] Y[1] Y[2]) 0.000000178813934326171875))",
     {"Y[0]": 1 + 2.0**-23, "Y[1]": 2.0**-24, "Y[2]": -1.0}, "exact", True),
    ("(assert (> Y[0] 0.1))", {"Y[0]": _F32_01}, "f32", False),
    ("(assert (> Y[0] 0.1))", {"Y[0]": _F32_01}, "exact", True),
    # left to right: (1 - 2^-25) rounds to 1, then 1 - 2^-25 rounds to 1
    ("(assert (>= (- Y[0] Y[1] Y[2]) 1))",
     {"Y[0]": 1.0, "Y[1]": 2.0**-25, "Y[2]": 2.0**-25}, "f32", True),
]


@pytest.mark.parametrize("text, values, mode, accepted", _REFERENCE_SELF_TEST)
def test_reference_reads_hand_written_cases(text, values, mode, accepted):
    """The reference of every cell below, on cases decided by hand. A bug in
    it would make every cell check the wrong promise."""
    assert _Text(text).accepts(values, mode) is accepted


@pytest.mark.parametrize("text, values", [
    ("(assert (<= Y[0] foo))", {"Y[0]": 0.0}),
    ("(assert Y[0])", {"Y[0]": 1.0}),
    ("(assert (<= Y[0]))", {"Y[0]": 0.0}),
    ("(assert (<= (and (<= Y[0] 0)) 1))", {"Y[0]": 0.0}),
])
def test_reference_gives_no_meaning_to_ill_formed_text(text, values):
    with pytest.raises(_Undefined):
        _Text(text).accepts(values)


# =========================================================================== #
# Helpers on the parse's side (exact arithmetic on the stored float64 numbers) #
# =========================================================================== #
def _q(value):
    value = float(value)
    return value if math.isinf(value) else Fraction(value)


def _rows_hold(G, g, point):
    G = np.asarray(G, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64).ravel()
    pq = [Fraction(v) for v in point]
    for i in range(G.shape[0]):
        lhs = sum((Fraction(float(G[i, j])) * pq[j]
                   for j in range(G.shape[1]) if G[i, j] != 0.0), Fraction(0))
        if lhs > _q(g[i]):
            return False
    return True


def _in_box(lb, ub, point):
    return all(_q(lo) <= Fraction(v) <= _q(hi)
               for lo, v, hi in zip(np.ravel(lb), point, np.ravel(ub)))


def _in_property(prop, y):
    """``y`` in every group's union of polytopes ``G y <= g``."""
    for group in prop:
        hss = group["Hg"] if isinstance(group["Hg"], list) else [group["Hg"]]
        if not any(_rows_hold(h.G, h.g, y) for h in hss):
            return False
    return True


def _load(tmp_path, text):
    path = tmp_path / "spec.vnnlib"
    path.write_text(text)
    return load_vnnlib(str(path))


def _grid(axes):
    return list(itertools.product(*axes))


# =========================================================================== #
# Relational specifications                                                    #
# =========================================================================== #
@dataclass
class _Rel:
    text: str
    inputs: dict          # joint input order: name -> values to try
    outputs: dict         # joint output order: name -> values to try
    extra_x: tuple = field(default_factory=tuple)

    def x_points(self):
        return _grid(list(self.inputs.values())) + list(self.extra_x)

    def y_points(self):
        return _grid(list(self.outputs.values()))


def _two_networks(f_in, f_out, g_in, g_out, relation="(equal-to F)"):
    """Header of networks F and G; ``*_in``/``*_out`` are declaration lines."""
    return ("(vnnlib-version <2.0>)\n(declare-network F\n"
            + "".join(f"    {d}\n" for d in f_in + f_out) + ")\n"
            + "(declare-network G\n" + (f"    {relation}\n" if relation else "")
            + "".join(f"    {d}\n" for d in g_in + g_out) + ")\n")


_SCALAR_PAIR = _two_networks(["(declare-input XF real [1])"],
                             ["(declare-output YF real [1])"],
                             ["(declare-input XG real [1])"],
                             ["(declare-output YG real [1])"])

_REL = {
    # The monotonic shape, with a network whose input and output sizes
    # differ, negative and wide bounds, a dimension of G bounded on one side
    # and coupled by >=, and an equality coupling.
    "box-ge-and-equality-couplings": _Rel(
        _two_networks(["(declare-input XF real [1,2])"],
                      ["(declare-output YF real [1,1])"],
                      ["(declare-input XG real [1,2])"],
                      ["(declare-output YG real [1,1])"])
        + "(assert (and (>= XF[0,0] -1.5) (<= XF[0,0] 2.5)))\n"
          "(assert (and (<= XF[0,1] 0.25) (>= XF[0,1] -0.75)))\n"
          "(assert (and (>= XF[0,0] XG[0,0]) (>= XG[0,0] -1.0)))\n"
          "(assert (== XF[0,1] XG[0,1]))\n"
          "(assert (YF[0,0] <= YG[0,0]))\n",
        {"XF[0,0]": (-1.75, -1.5, -1.0, 0.25, 2.5, 2.75),
         "XF[0,1]": (-1.0, -0.75, 0.0, 0.25, 0.5),
         "XG[0,0]": (-1.25, -1.0, -0.5, 0.25, 2.5, 2.75),
         "XG[0,1]": (-1.0, -0.75, 0.0, 0.25, 0.5)},
        {"YF[0,0]": (-1.5, -0.5, 0.0, 0.5, 1.5),
         "YG[0,0]": (-1.5, -0.5, 0.0, 0.5, 1.5)}),
    # Couplings with constants on both sides, a coefficient of 2 over three
    # variables, a coupling inside one network with a coefficient of 1/2, G
    # bounded through the couplings only; an output (or ...) whose second
    # disjunct is an (and ...) of rows, and a second output assert.
    "constants-coefficients-and-output-or": _Rel(
        _two_networks(["(declare-input XF real [2])"],
                      ["(declare-output YF real [2])"],
                      ["(declare-input XG real [2])"],
                      ["(declare-output YG real [2])"],
                      relation="(isomorphic-to F)")
        + "(assert (and (>= XF[0] -2) (<= XF[0] 2)))\n"
          "(assert (and (>= XF[1] -0.5) (<= XF[1] 1.5)))\n"
          "(assert (<= XG[0] (+ XF[0] 0.25)))\n"
          "(assert (>= XG[0] (- XF[0] 1.5)))\n"
          "(assert (>= XG[1] (- XF[1] 0.25)))\n"
          "(assert (<= (* 2 XG[1]) (+ XF[0] XF[1] 1)))\n"
          "(assert (>= XF[1] (* 0.5 XF[0])))\n"
          "(assert (or (<= YF[0] -0.5) "
          "(and (>= (- YG[1] YF[1]) 1.25) (<= YG[1] 2))))\n"
          "(assert (and (<= YG[0] 2) (>= YF[1] -2)))\n",
        {"XF[0]": (-2.25, -2.0, -0.5, 0.0, 1.0, 2.0),
         "XF[1]": (-0.75, -0.5, 0.0, 0.5, 1.5),
         "XG[0]": (-3.5, -1.5, -0.5, 0.0, 0.25, 2.25),
         "XG[1]": (-0.75, -0.25, 0.25, 0.5, 1.0, 1.25, 1.5)},
        {"YF[0]": (-1.0, -0.5, 0.0),
         "YF[1]": (-2.25, -2.0, 0.5),
         "YG[0]": (2.0, 2.25),
         "YG[1]": (1.25, 1.75, 2.0, 2.25)}),
    # G declares two input tensors; F's inputs are a [1,3] tensor. Every
    # tensor's entries must land at their own joint positions.
    "several-input-tensors": _Rel(
        _two_networks(["(declare-input XF real [1,3])"],
                      ["(declare-output YF real [1,2])"],
                      ["(declare-input XA real [1,1])",
                       "(declare-input XB real [1,2])"],
                      ["(declare-output YG real [2])"])
        + "(assert (and (>= XF[0,0] -1) (<= XF[0,0] 1)))\n"
          "(assert (and (>= XF[0,1] 0.5) (<= XF[0,1] 1.5)))\n"
          "(assert (and (>= XF[0,2] -3) (<= XF[0,2] -2)))\n"
          "(assert (== XA[0,0] XF[0,0]))\n"
          "(assert (== XB[0,0] XF[0,1]))\n"
          "(assert (and (>= XB[0,1] -2.5) (<= XB[0,1] 0)))\n"
          "(assert (<= (+ YF[0,1] YG[1]) 0.5))\n",
        {"XF[0,0]": (-1.25, -1.0, 1.0),
         "XF[0,1]": (0.25, 0.5, 1.5),
         "XF[0,2]": (-3.25, -3.0, -2.0, -1.75),
         "XA[0,0]": (-1.0, 0.5, 1.0),
         "XB[0,0]": (0.5, 1.0, 1.5),
         "XB[0,1]": (-2.75, -2.5, 0.0, 0.25)},
        {"YF[0,0]": (0.0,), "YF[0,1]": (-0.5, 0.0, 0.5),
         "YG[0]": (3.0,), "YG[1]": (0.0, 0.5, 1.0)}),
    # The strict forms of the relational benchmarks: a strict output
    # comparison in infix form, and a strict coupling.
    "strict-output": _Rel(
        _SCALAR_PAIR
        + "(assert (and (>= XF[0] -1) (<= XF[0] 1)))\n"
          "(assert (== XG[0] XF[0]))\n"
          "(assert (YF[0] < YG[0]))\n",
        {"XF[0]": (-1.25, -1.0, 0.0, 1.0, 1.25),
         "XG[0]": (-1.0, 0.0, 1.0)},
        {"YF[0]": (-0.5, 0.0, 0.5), "YG[0]": (-0.5, 0.0, 0.5)}),
    "strict-coupling": _Rel(
        _SCALAR_PAIR
        + "(assert (and (>= XF[0] -1) (<= XF[0] 1)))\n"
          "(assert (< XG[0] XF[0]))\n"
          "(assert (and (>= XG[0] -1.5) (<= XG[0] 1.5)))\n"
          "(assert (<= YF[0] YG[0]))\n",
        {"XF[0]": (-1.25, -1.0, 0.0, 0.5, 1.0),
         "XG[0]": (-1.75, -1.0, 0.0, 0.5, 1.0, 1.5)},
        {"YF[0]": (-0.5, 0.0, 0.5), "YG[0]": (-0.5, 0.0, 0.5)}),
}

_REL_NEQ_INPUT = _Rel(
    _SCALAR_PAIR
    + "(assert (and (>= XF[0] -1) (<= XF[0] 1)))\n"
      "(assert (!= XF[0] 0.5))\n"
      "(assert (== XG[0] XF[0]))\n"
      "(assert (<= YF[0] YG[0]))\n",
    {"XF[0]": (-1.0, 0.0, 0.5, 1.0), "XG[0]": (-1.0, 0.0, 0.5, 1.0)},
    {"YF[0]": (0.0,), "YG[0]": (0.0,)})


def _coupling_with_tenth(op):
    """XG op XF + 0.1, with joint inputs on both sides of the exact boundary
    xg - xf = 1/10 (decimals read exactly)."""
    tenth = Fraction(1, 10)
    extra = []
    for xf in (0.0, 2.0**-60, -(2.0**-60), 2.0**-56, 0.5):
        xg = float(Fraction(xf) + tenth)
        for step in (-2, -1, 0, 1, 2):
            v = xg
            for _ in range(abs(step)):
                v = math.nextafter(v, math.inf if step > 0 else -math.inf)
            extra.append((xf, v))
    return _Rel(
        _SCALAR_PAIR
        + "(assert (and (>= XF[0] -1) (<= XF[0] 1)))\n"
          "(assert (and (>= XG[0] -1) (<= XG[0] 2)))\n"
          f"(assert ({op} XG[0] (+ XF[0] 0.1)))\n"
          "(assert (<= YF[0] YG[0]))\n",
        {"XF[0]": (0.0,), "XG[0]": (0.0,)},
        {"YF[0]": (0.0,), "YG[0]": (0.0,)},
        extra_x=tuple(extra))


_RELATIONAL_STRICT = known_bug(
    6, "a relational 2.0 spec keeps no source assertions for the witness "
       "check: a strict comparison (on the outputs or in an input coupling) "
       "is stored as the closed row, so the parsed region accepts a point on "
       "the strict boundary that the text rejects")


def _split_asserts(case):
    """The text's input-only and output-only asserts. A relational text that
    mixes the two in one assert is outside these cells (it must be refused,
    see the refusal test)."""
    text = _Text(case.text)
    ins, outs = [], []
    for a in text.asserts:
        names = _names(a)
        if names <= set(case.inputs):
            ins.append(a)
        elif names <= set(case.outputs):
            outs.append(a)
        else:
            pytest.fail(f"setup: assert mixes inputs and outputs: {a!r}")
    return text, ins, outs


def _relational_disagreements(tmp_path, case):
    """Points where the parse and the text disagree:
    ``(kind, point, parse_accepts, text_accepts)``."""
    parsed = _load(tmp_path, case.text)
    if parsed.get("format") != "relational":
        pytest.fail(f"setup: not read as relational: {parsed.get('format')}")
    n_in, n_out = len(case.inputs), len(case.outputs)
    if np.ravel(parsed["lb"]).size != n_in:
        pytest.fail(f"setup: {np.ravel(parsed['lb']).size} joint inputs, "
                    f"the text declares {n_in}")
    text, ins, outs = _split_asserts(case)
    coupling = parsed["input_coupling"]
    out = []
    for x in case.x_points():
        values = dict(zip(case.inputs, x))
        p = _in_box(parsed["lb"], parsed["ub"], x) and (
            coupling is None or _rows_hold(coupling.G, coupling.g, x))
        t = text.accepts(values, asserts=ins)
        if p != t:
            out.append(("input", x, p, t))
    for y in case.y_points():
        values = dict(zip(case.outputs, y))
        p = _in_property(parsed["prop"], y)
        t = text.accepts(values, asserts=outs)
        if p != t:
            out.append(("output", y, p, t))
    return out


@pytest.mark.parametrize("case", [
    *[pytest.param(c, id=k) for k, c in _REL.items()],
    pytest.param(_REL_NEQ_INPUT, id="neq-on-one-input", marks=known_bug(
        6, "registry SPEC-9 on the relational route: (!= XF c) on an input "
           "is stored as the lower bound c, so the joint inputs below c are "
           "left out of the region (false UNSAT)")),
    pytest.param(_coupling_with_tenth(">="), id="coupling-ge-tenth",
                 marks=known_bug(
        6, "a relational coupling row with a constant that is not a binary "
           "fraction is stored rounded to nearest with nothing moved outward: "
           "(>= XG (+ XF 0.1)) gives xf - xg <= -fl(0.1), which leaves out "
           "xf = 2^-60, xg = fl(0.1), a joint input the text allows")),
    pytest.param(_coupling_with_tenth("<="), id="coupling-le-tenth"),
])
def test_relational_parse_keeps_every_counterexample_of_the_text(
        tmp_path, case):
    """Every joint input the text allows is in the stored box and satisfies
    the stored coupling rows, and every joint output the text calls unsafe is
    in the stored property. Catches a coupling row with the wrong sign or
    right-hand side (``<=``, ``>=``, ``==``), a lower bound clipped at 0 by the
    initial box, an output disjunct dropped, a bound of one tensor written on
    another's entries; each would leave out a real counterexample (false
    UNSAT)."""
    missed = [d for d in _relational_disagreements(tmp_path, case)
              if d[3] and not d[2]]
    assert not missed, (
        f"{len(missed)} points the text allows and the parse leaves out, "
        f"e.g. {missed[:3]}")


@pytest.mark.parametrize("case", [
    pytest.param(c, id=k, marks=_RELATIONAL_STRICT if k.startswith("strict")
                 else ()) for k, c in _REL.items()])
def test_relational_parse_accepts_nothing_the_text_rejects(tmp_path, case):
    """No joint input outside the text's region and no joint output outside
    its unsafe property is accepted by the parse. Catches an equality coupling
    read as one inequality, a coupling or output row read on the wrong
    side, a coefficient dropped, a later output assert overwriting an earlier
    one; each would let the falsifier report an invalid SAT."""
    extra = [d for d in _relational_disagreements(tmp_path, case)
             if d[2] and not d[3]]
    assert not extra, (
        f"{len(extra)} points the parse accepts and the text rejects, "
        f"e.g. {extra[:3]}")


@pytest.mark.parametrize("case, expected", [
    pytest.param(_REL["several-input-tensors"], [
        ("F", None, 0, 3, 0, 2), ("G", ("equal-to", "F"), 3, 3, 2, 2)],
        id="several-input-tensors"),
    pytest.param(_REL["box-ge-and-equality-couplings"], [
        ("F", None, 0, 2, 0, 1), ("G", ("equal-to", "F"), 2, 2, 1, 1)],
        id="inputs-and-outputs-of-other-sizes"),
])
def test_relational_networks_are_numbered_in_declaration_order(
        tmp_path, case, expected):
    """Each network's ``(name, relation, input_offset, input_size,
    output_offset, output_size)``, by hand: its tensors follow those of the
    networks declared before it. ``relational_reach`` slices each network's
    inputs and outputs by these numbers, so a wrong offset makes a network read
    the other network's inputs (false UNSAT, see the verdict test below).
    Catches an offset summed over the network's own tensors or over the other
    kind, a size of the first tensor only, the relation pair reversed."""
    parsed = _load(tmp_path, case.text)
    got = [(n["name"], n["relation"], n["input_offset"], n["input_size"],
            n["output_offset"], n["output_size"]) for n in parsed["networks"]]
    assert got == expected


# --------------------------------------------------------------------------- #
# The public route: load_vnnlib -> verify_relational                           #
# --------------------------------------------------------------------------- #
def _first_input():
    """y = x[0] for a two-entry input (weights 1 and 0, bias 0)."""
    net = torch.nn.Sequential(torch.nn.Linear(2, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        net[0].bias.zero_()
    return net.eval()


_TWO_BOXES = _two_networks(["(declare-input XF real [2])"],
                           ["(declare-output YF real [1])"],
                           ["(declare-input XG real [2])"],
                           ["(declare-output YG real [1])"]) + (
    "(assert (and (>= XF[0] 0) (<= XF[0] 1)))\n"
    "(assert (and (>= XF[1] 0) (<= XF[1] 1)))\n"
    "(assert (and (>= XG[0] 2) (<= XG[0] 3)))\n"
    "(assert (and (>= XG[1] 2) (<= XG[1] 3)))\n")


@pytest.mark.parametrize("text, witness", [
    # only the second disjunct of the output (or ...) is reachable, and only
    # when F reads its own inputs (YG - YF >= 1.5 needs XG - XF >= 1.5)
    pytest.param(_TWO_BOXES + "(assert (or (<= YF[0] -1) "
                 "(>= (- YG[0] YF[0]) 1.5)))\n",
                 (0.0, 0.0, 2.0, 2.0), id="second-disjunct-of-an-output-or"),
])
def test_relational_verdict_is_not_unsat_while_the_text_has_a_counterexample(
        tmp_path, text, witness):
    """Through the public route (``load_vnnlib`` then ``verify_relational``):
    the joint input ``witness`` is allowed by the text and its outputs, y =
    x[0] for both networks computed by hand, are unsafe by the text, so the
    verdict must not be UNSAT. Catches a network offset that makes F read G's
    inputs and an output disjunct dropped by the relational loader."""
    t = _Text(text)
    xf, xg = witness[:2], witness[2:]
    values = {"XF[0]": xf[0], "XF[1]": xf[1], "XG[0]": xg[0], "XG[1]": xg[1],
              "YF[0]": xf[0], "YG[0]": xg[0]}
    if not t.accepts(values):
        pytest.fail("setup: the witness is not a counterexample of the text")
    parsed = _load(tmp_path, text)
    result = verify_relational(_first_input(), _first_input(), parsed)
    assert result.verdict != "UNSAT"


# --------------------------------------------------------------------------- #
# Refusals                                                                     #
# --------------------------------------------------------------------------- #
_BOXED_PAIR = _SCALAR_PAIR + (
    "(assert (and (>= XF[0] 0) (<= XF[0] 1)))\n"
    "(assert (and (>= XG[0] 0) (<= XG[0] 1)))\n")


@pytest.mark.parametrize("text", [
    pytest.param(_BOXED_PAIR + "(assert (<= YF[0] YG[0]))\n"
                 "(assert (<= (- XG[0] XF[0]) YF[0]))\n",
                 id="inputs-and-an-output-in-one-comparison"),
    pytest.param(_BOXED_PAIR + "(assert (or (<= XF[0] 0.25) "
                 "(>= XF[0] 0.75)))\n(assert (<= YF[0] YG[0]))\n",
                 id="disjunctive-input-assert"),
    pytest.param(_BOXED_PAIR + "(assert (!= XF[0] XG[0]))\n"
                 "(assert (<= YF[0] YG[0]))\n", id="disequality-coupling"),
    pytest.param(_BOXED_PAIR + "(assert (<= YF[0] YG[0]))\n"
                 "(assert (!= YF[0] YG[0]))\n",
                 id="disequality-on-the-outputs"),
    pytest.param(_BOXED_PAIR + "(assert (<= (* XF[0] XG[0]) 0.5))\n"
                 "(assert (<= YF[0] YG[0]))\n", id="product-of-inputs"),
    pytest.param(_BOXED_PAIR, id="no-output-property"),
    pytest.param(_two_networks(["(declare-input XF real [1])"],
                               ["(declare-output YF real [1])"],
                               ["(declare-input XG real [1])"], [])
                 + "(assert (and (>= XF[0] 0) (<= XF[0] 1)))\n"
                   "(assert (== XG[0] XF[0]))\n(assert (<= YF[0] 0))\n",
                 id="network-without-an-output"),
])
def test_relational_text_outside_the_supported_fragment_is_refused(
        tmp_path, text):
    """A relational text the lowering cannot represent raises ``ValueError``
    (``VNNLibParseError``) instead of being read as something else, through
    both public entries (``load_vnnlib`` adds its own check of an empty
    property, ``load_vnnlib_v2`` does not; an empty property list would be
    answered UNSAT by ``verify_specification``). Catches the loss of the
    refusals of a mixed input/output assert (it would become an input coupling
    row with the output's coefficient on an input), of a disjunctive input
    assert (only its first disjunct would be kept), of ``!=`` in a coupling, of
    a spec without an output property or a network without an output, and an
    assert that cannot be read being skipped instead of refused."""
    with pytest.raises(ValueError):
        _load(tmp_path, text)
    with pytest.raises(ValueError):
        load_vnnlib_v2(str(tmp_path / "spec.vnnlib"))


# =========================================================================== #
# Nonlinear specifications: the assertion tree and the point check             #
# =========================================================================== #
def _one_network(inputs, outputs):
    return ("(vnnlib-version <2.0>)\n(declare-network N\n"
            + "".join(f"    {d}\n" for d in inputs + outputs) + ")\n")


_NL_HEADER = _one_network(["(declare-input X real [1,2])"],
                          ["(declare-output Y real [1,2])"])
_NL_BOX = ("(assert (and (>= X[0,0] -1) (<= X[0,0] 1)))\n"
           "(assert (and (>= X[0,1] -1) (<= X[0,1] 1)))\n")
_XV = (-1.25, -1.0, 0.0, 0.5, 1.0)
_YV = (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0)


@dataclass
class _Nl:
    text: str
    axes: dict            # name -> values; inputs first, in joint order
    n_in: int = 2

    def points(self):
        names = list(self.axes)
        for p in _grid(list(self.axes.values())):
            yield dict(zip(names, p)), np.array(p[:self.n_in]), \
                np.array(p[self.n_in:])


def _xy(xv=_XV, yv=_YV):
    return {"X[0,0]": xv, "X[0,1]": xv, "Y[0,0]": yv, "Y[0,1]": yv}


_NL = {
    "product-with-box-and-infix": _Nl(
        _NL_HEADER
        + "(assert (and (>= X[0,0] -1.5) (<= X[0,0] 2)))\n"
          "(assert (and (<= X[0,1] 0.5) (>= X[0,1] -0.25)))\n"
          "(assert (<= (* Y[0,0] Y[0,1]) 0.25))\n"
          "(assert ((* Y[0,0] X[0,0]) >= -1))\n",
        _xy(xv=(-1.75, -1.5, -0.25, 0.5, 2.0, 2.25))),
    "strict-comparisons": _Nl(
        _NL_HEADER + _NL_BOX
        + "(assert (< (* Y[0,0] Y[0,1]) 0.5))\n"
          "(assert (> Y[0,0] X[0,0]))\n"
          "(assert (Y[0,1] > -1))\n",
        _xy()),
    "and-or-inside-one-assert": _Nl(
        _NL_HEADER
        + "(assert (and (>= X[0,0] -1) (<= X[0,0] 1) (>= X[0,1] 0) "
          "(<= X[0,1] 1)))\n"
          "(assert (or (and (<= Y[0,0] 0.5) (>= Y[0,1] 0.5)) "
          "(< (* Y[0,0] Y[0,1]) -0.25) (== Y[0,0] X[0,1])))\n",
        _xy()),
    "sums-differences-negation-products": _Nl(
        _NL_HEADER + _NL_BOX
        + "(assert (<= (+ Y[0,0] Y[0,1] X[0,0]) 0.5))\n"
          "(assert (>= (- Y[0,0] Y[0,1] 0.5) -1.5))\n"
          "(assert (<= (* (- Y[0,0]) Y[0,1]) 0.25))\n"
          "(assert (<= (* 2 Y[0,0] Y[0,1] X[0,1]) 1))\n",
        _xy()),
    "disequality-and-equality": _Nl(
        _NL_HEADER + _NL_BOX
        + "(assert (!= Y[0,0] Y[0,1]))\n"
          "(assert (or (== Y[0,0] X[0,0]) (<= Y[0,1] -0.5)))\n",
        _xy()),
    "division": _Nl(
        _NL_HEADER + _NL_BOX
        + "(assert (<= (/ Y[0,0] Y[0,1]) 0.5))\n"
          "(assert (>= (/ Y[0,0] 4) -0.125))\n",
        _xy(yv=(-2.0, -0.5, 0.25, 0.5, 1.0, 2.0))),
    "several-tensors": _Nl(
        _one_network(["(declare-input A real [1,1])",
                      "(declare-input B real [1,2])"],
                     ["(declare-output Y real [1,2])",
                      "(declare-output Z real [1])"])
        + "(assert (and (>= A[0,0] -1) (<= A[0,0] 1)))\n"
          "(assert (and (>= B[0,1] 0) (<= B[0,1] 2)))\n"
          "(assert (<= (* Y[0,1] Z[0]) 0.5))\n"
          "(assert (<= (* B[0,1] Z[0]) 0.25))\n"
          "(assert (>= (- Y[0,1] A[0,0]) Z[0]))\n",
        {"A[0,0]": (-1.25, -1.0, 1.0), "B[0,0]": (0.0,),
         "B[0,1]": (-0.5, 0.0, 0.5, 2.0), "Y[0,0]": (0.0,),
         "Y[0,1]": (-1.0, 0.0, 0.5, 1.0), "Z[0]": (-0.5, 0.0, 0.5, 1.0)},
        n_in=3),
}


def _nonlinear(tmp_path, text):
    parsed = _load(tmp_path, text)
    if parsed.get("format") != "nonlinear":
        pytest.fail(f"setup: not read as nonlinear: {parsed.get('format')}")
    return parsed


def _nonlinear_disagreements(parsed, case):
    text = _Text(case.text)
    out = []
    for values, x, y in case.points():
        got = bool(evaluate_nonlinear(parsed, x, y))
        want = text.accepts(values)
        if got != want:
            out.append((values, got, want))
    return out


@pytest.mark.parametrize("case", [pytest.param(c, id=k)
                                  for k, c in _NL.items()])
def test_nonlinear_assertions_classify_every_point_as_the_text_does(
        tmp_path, case):
    """``load_vnnlib`` then ``evaluate_nonlinear`` at every grid point, inside,
    outside and on every comparison (both sides equal, a product on its
    bound), equals the text's verdict. A wrong True is a false witness
    accepted, a wrong False a real one thrown away. Catches ``and`` read as
    ``or`` and the reverse, a child of ``and``/``or`` skipped, the asserts
    combined with any(), an infix comparison with its operands swapped, a
    strict comparison read as closed, ``!=`` read as ``==``, ``==`` read as
    ``<=``, a sum or a difference of three terms cut to two, a unary minus
    dropped, a factor of a product skipped, a quotient reversed, input and
    output read from each other's vector, all but the last assert dropped."""
    parsed = _nonlinear(tmp_path, case.text)
    bad = _nonlinear_disagreements(parsed, case)
    assert not bad, (f"{len(bad)} points where the assertions and the text "
                     f"disagree (point, parse, text), e.g. {bad[:3]}")


_NL_FIRST = _NL_HEADER + _NL_BOX + "(assert (<= (* Y[0,0] Y[0,1]) 4))\n"
_NL3_HEADER = _one_network(["(declare-input X real [1,1])"],
                           ["(declare-output Y real [1,3])"])
_NL3_FIRST = (_NL3_HEADER + "(assert (and (>= X[0,0] -1) (<= X[0,0] 1)))\n"
              "(assert (<= (* Y[0,0] Y[0,1]) 4))\n")
_NL3_AXES = {"X[0,0]": (0.0,), "Y[0,0]": (-0.5, 0.5, 1.0),
             "Y[0,1]": (0.25, 0.5, 1.0), "Y[0,2]": (-0.5, 0.25, 0.5, 1.0)}
_NOT_CHECKED = (
    "a nonlinear 2.0 assertion is not checked for its form: _resolve_ast "
    "accepts {what} and evaluate_ast {how}; the linear lowering refuses the "
    "same text when this assert comes first")


@pytest.mark.parametrize("case", [
    pytest.param(_Nl(_NL_FIRST + "(assert (not (<= Y[0,0] 0)))\n",
                     _xy()), id="negation"),
    pytest.param(_Nl(_NL_FIRST + "(assert (<= (/ Y[0,0] Y[0,1]) 1))\n",
                     _xy(yv=(-1.0, 0.5, 1.0, 2.0))), id="binary-division"),
    pytest.param(_Nl(_NL3_FIRST + "(assert (<= Y[0,0] Y[0,1] Y[0,2]))\n",
                     _NL3_AXES, n_in=1),
                 id="chained-comparison", marks=known_bug(
        6, _NOT_CHECKED.format(
            what="a comparison with three operands",
            how="reads only the first two, so a point the text rejects is "
                "accepted as a counterexample"))),
    pytest.param(_Nl(_NL3_FIRST + "(assert (<= (/ Y[0,0] Y[0,1] Y[0,2]) "
                     "1))\n", _NL3_AXES, n_in=1),
                 id="division-of-three-operands", marks=known_bug(
        6, _NOT_CHECKED.format(
            what="a division with three operands",
            how="divides the first by the second only, so a point the text "
                "rejects is accepted as a counterexample"))),
    pytest.param(_Nl(_NL_FIRST + "(assert (<= Y[0,0]))\n", _xy()),
                 id="comparison-with-one-operand", marks=known_bug(
        6, _NOT_CHECKED.format(
            what="a comparison with one operand",
            how="crashes with IndexError on it"), raises=IndexError)),
])
def test_nonlinear_form_is_read_exactly_or_refused(tmp_path, case):
    """A form the reference knows either raises ``ValueError`` at load or is
    read exactly: the assertions classify every grid point as the text does
    (a chained comparison as SMT-LIB chains it, a division of three operands
    left to right, ``not`` as the negation); a form without meaning (one
    operand) must be refused. Catches a form read as another one."""
    try:
        parsed = _load(tmp_path, case.text)
    except ValueError:
        return
    if parsed.get("format") != "nonlinear":
        pytest.fail(f"setup: not read as nonlinear: {parsed.get('format')}")
    text = _Text(case.text)
    bad = []
    for values, x, y in case.points():
        got = bool(evaluate_nonlinear(parsed, x, y))
        try:
            want = text.accepts(values)
        except _Undefined:
            pytest.fail("the text has no meaning here and was not refused")
        if got != want:
            bad.append((values, got, want))
    assert not bad, (f"{len(bad)} points read otherwise than the text, "
                     f"e.g. {bad[:3]}")


@pytest.mark.parametrize("text", [
    pytest.param(_NL_FIRST + "(assert (<= Y[0,0] foo))\n", id="unknown-name"),
    pytest.param(_NL_FIRST + "(assert ())\n", id="empty-clause"),
    pytest.param(_NL_FIRST + "(assert (=> (<= Y[0,0] 0) (<= Y[0,1] 0)))\n",
                 id="operator-outside-the-language"),
    pytest.param(_NL_FIRST + "(assert Y[0,0])\n", id="number-as-a-formula",
                 marks=known_bug(6, _NOT_CHECKED.format(
                     what="a variable where a formula is due",
                     how="takes the variable's value as true when it is not "
                         "zero, so a malformed spec is answered"),
                     raises=DID_NOT_RAISE)),
])
def test_nonlinear_text_without_meaning_is_refused(tmp_path, text):
    """An assertion with no meaning, after a nonlinear assert (so that the
    nonlinear loader reads it), raises ``ValueError``. Catches an unknown name
    read as a number and an unknown operator accepted."""
    with pytest.raises(ValueError):
        _load(tmp_path, text)


# --------------------------------------------------------------------------- #
# The float32 check of a witness                                               #
# --------------------------------------------------------------------------- #
_NL_Y3 = _one_network(["(declare-input X real [1,1])"],
                      ["(declare-output Y real [1,3])"]) + (
    "(assert (and (>= X[0,0] -1) (<= X[0,0] 1)))\n"
    "(assert (<= (* Y[0,0] Y[0,0]) 4))\n")
_ORDINARY = ((0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.25, -0.5), (0.5, 1.0, 0.0, 0.0))


@pytest.mark.parametrize("text, points", [
    # float32: (1 + 2^-23) + 2^-24 is 1 + 2^-22, minus 1 is 2^-22 > c
    pytest.param(_NL_Y3 + "(assert (<= (+ Y[0,0] Y[0,1] Y[0,2]) "
                 "0.000000178813934326171875))\n",
                 ((0.0, 1 + 2.0**-23, 2.0**-24, -1.0),) + _ORDINARY,
                 id="sum-of-three"),
    # float32(0.1) > 0.1 holds in float64, not in float32
    pytest.param(_NL_Y3 + "(assert (> Y[0,1] 0.1))\n",
                 ((0.0, 1.0, _F32_01, 0.0), (0.0, 1.0, 0.125, 0.0)),
                 id="constant-rounded-to-float32"),
    # (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11 in float32
    pytest.param(_NL_Y3 + "(assert (<= (* Y[0,1] Y[0,2]) 1.00048828125))\n",
                 ((0.0, 1.0, 1 + 2.0**-12, 1 + 2.0**-12),) + _ORDINARY,
                 id="product-rounded-to-float32"),
    pytest.param(_NL_Y3 + "(assert (<= (- Y[0,0] Y[0,1] Y[0,2]) "
                 "0.999999940395355224609375))\n",
                 ((0.0, 1.0, 2.0**-25, 2.0**-25),) + _ORDINARY,
                 id="difference-of-three", marks=known_bug(
        6, "evaluate_ast in float32 computes (- a b c) as a - (b + c), not "
           "left to right as documented: at y = (1, 2^-25, 2^-25) it gives "
           "1 - 2^-24 <= c where the checker's (1 - 2^-25) - 2^-25 is 1 > c, "
           "so the witness check accepts a point the checker rejects")),
])
def test_float32_check_evaluates_as_the_checker_does(tmp_path, text, points):
    """``evaluate_nonlinear(..., dtype=np.float32)`` evaluates as the VNN-COMP
    2026 checker does (float32, left to right, one rounding per operation),
    and without ``dtype`` as the exact text (the points are chosen so that
    float64 is exact). The points are where float32 and the exact value
    decide differently. Catches the float32 mode lost in the recursion or not
    handed on by ``evaluate_nonlinear`` (a witness the checker rejects would
    pass the runner's float32 check)."""
    parsed = _nonlinear(tmp_path, text)
    t = _Text(text)
    names = ["X[0,0]", "Y[0,0]", "Y[0,1]", "Y[0,2]"]
    for p in points:
        values = dict(zip(names, p))
        x, y = np.array(p[:1]), np.array(p[1:])
        assert bool(evaluate_nonlinear(parsed, x, y)) is t.accepts(values), p
        assert bool(evaluate_nonlinear(parsed, x, y, dtype=np.float32)) is \
            t.accepts(values, mode="f32"), p


# --------------------------------------------------------------------------- #
# The best-effort input box of a nonlinear spec                                #
# --------------------------------------------------------------------------- #
_BOX_AXES = {"X[0,0]": (-1.75, -1.5, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5),
             "X[0,1]": (-2.0, -1.5, 0.0, 0.25, 1.5, 2.0),
             "Y[0,0]": (-1.0, 0.0, 1.0), "Y[0,1]": (-1.0, 0.0, 1.0)}
_PRODUCT = "(assert (<= (* Y[0,0] Y[0,1]) 0.5))\n"


@pytest.mark.parametrize("case, stated", [
    pytest.param(_Nl(_NL_HEADER + "(assert (and (>= X[0,0] -1.5) "
                     "(<= X[0,0] 1)))\n(assert (and (<= X[0,1] 1.5) "
                     "(>= X[0,1] -2)))\n" + _PRODUCT, _BOX_AXES),
                 ([-1.5, -2.0], [1.0, 1.5]), id="plain-bounds"),
    pytest.param(_Nl(_NL_HEADER + "(assert (and (and (>= 0.5 X[0,0]) "
                     "(>= X[0,0] -0.5)) (<= (* 2 X[0,1]) 3) "
                     "(<= (- X[0,1]) 1.5)))\n" + _PRODUCT, _BOX_AXES),
                 ([-0.5, -1.5], [0.5, 1.5]),
                 id="nested-and-number-first-and-scaled"),
    pytest.param(_Nl(_NL_HEADER + "(assert (or (and (>= X[0,0] -1.5) "
                     "(<= X[0,0] -0.5)) (and (>= X[0,0] 0.5) "
                     "(<= X[0,0] 1.5))))\n(assert (and (>= X[0,1] -2) "
                     "(<= X[0,1] 2)))\n" + _PRODUCT, _BOX_AXES),
                 None, id="disjunction-of-input-boxes"),
    pytest.param(_Nl(_NL_HEADER + "(assert (and (>= X[0,0] -1.5) "
                     "(<= X[0,0] 1.5) (<= X[0,0] Y[0,0])))\n(assert (or "
                     "(<= X[0,1] -1.5) (< (* Y[0,0] Y[0,1]) 0)))\n"
                     + _PRODUCT, _BOX_AXES),
                 None, id="input-atoms-mixed-with-outputs"),
    pytest.param(_Nl(_NL_HEADER + "(assert (and (>= X[0,0] -1.5) "
                     "(<= X[0,0] 1.5)))\n(assert (!= X[0,0] 0.5))\n"
                     "(assert (and (>= X[0,1] -2) (<= X[0,1] 2)))\n"
                     + _PRODUCT, _BOX_AXES),
                 None, id="disequality-on-one-input", marks=known_bug(
        6, "registry SPEC-9 on the nonlinear route: (!= X c) on an input "
           "gives the box the lower bound c, so the inputs below c of real "
           "counterexamples lie outside the box the runner reaches over "
           "(false UNSAT)")),
])
def test_nonlinear_input_box_holds_every_counterexample_input(
        tmp_path, case, stated):
    """The runner reaches over the nonlinear spec's box for UNSAT, so the box
    must hold the input of every point the text accepts; where the text
    states a plain box of single-variable bounds at top level, the box is that
    box, exactly. Catches the box collector descending into an ``(or ...)``
    (the bounds of the disjuncts intersected), a lower bound clipped at 0 by
    the initial box, a bound taken from an atom that also names an output."""
    parsed = _nonlinear(tmp_path, case.text)
    lb, ub = np.ravel(parsed["lb"]), np.ravel(parsed["ub"])
    text = _Text(case.text)
    outside = [values for values, x, _ in case.points()
               if text.accepts(values) and not _in_box(lb, ub, x)]
    assert not outside, (f"{len(outside)} counterexamples of the text with "
                         f"their input outside the box [{lb}, {ub}], "
                         f"e.g. {outside[:2]}")
    if stated is not None:
        assert lb.tolist() == stated[0] and ub.tolist() == stated[1]


# =========================================================================== #
# Linear specs: the source assertions kept for the witness check              #
# =========================================================================== #
_LIN = _one_network(["(declare-input X real [1,2])"],
                    ["(declare-output Y real [1,3])"])
_LIN_X = (-1.25, -1.0, -0.5, -0.25, 0.0, 0.25, 0.75, 1.0)
_LIN_Y = (-0.75, -0.5, 0.0, 0.5, 0.75)


def _lin_points(extra=()):
    pts = [(x0, 0.0, y0, y1, 0.0) for x0 in _LIN_X for y0 in _LIN_Y
           for y1 in (0.0, 0.5, 0.75)]
    return pts + list(extra)


def _witness_check(parsed, x, y):
    """What the parse accepts as a witness: ``(x, y)`` in some stored pair
    (box, then every group's rows, decided exactly) and, when the parse keeps
    source assertions, accepted by them in float64 and in float32, as the
    runner's final SAT gate checks them (``_keep_assertions_for_sat_checks``
    and ``_witness_satisfies_spec``)."""
    if not any(_in_box(pair["lb"], pair["ub"], x)
               and _in_property(pair["prop"], y) for pair in parsed["pairs"]):
        return False
    if "assertions" not in parsed:
        return True
    return bool(evaluate_nonlinear(parsed, np.array(x), np.array(y))
                and evaluate_nonlinear(parsed, np.array(x), np.array(y),
                                       dtype=np.float32))


@pytest.mark.parametrize("text, points", [
    pytest.param(_LIN + "(assert (and (>= X[0,0] -1) (<= X[0,0] 1)))\n"
                 "(assert (and (>= X[0,1] -1) (<= X[0,1] 1)))\n"
                 "(assert (<= Y[0,1] Y[0,0]))\n(assert (> Y[0,0] 0.5))\n",
                 _lin_points(), id="strict-output-no-arithmetic"),
    pytest.param(_LIN + "(assert (and (>= X[0,1] -1) (<= X[0,1] 1)))\n"
                 "(assert (>= X[0,0] -0.5))\n(assert (< X[0,0] 0.75))\n"
                 "(assert (>= Y[0,0] 0.5))\n",
                 _lin_points(), id="strict-input-no-arithmetic"),
    pytest.param(_LIN + "(assert (and (>= X[0,0] -1) (<= X[0,0] 1)))\n"
                 "(assert (and (>= X[0,1] -1) (<= X[0,1] 1)))\n"
                 "(assert (or (and (<= X[0,0] -0.25) (> Y[0,0] 0.5)) "
                 "(and (>= X[0,0] 0.25) (< Y[0,0] -0.5))))\n",
                 _lin_points(), id="strict-in-a-combined-assert"),
    pytest.param(_LIN + "(assert (and (>= X[0,0] -1) (<= X[0,0] 1)))\n"
                 "(assert (and (>= X[0,1] -1) (<= X[0,1] 1)))\n"
                 "(assert (<= (+ Y[0,0] Y[0,1] Y[0,2]) "
                 "0.000000178813934326171875))\n",
                 [(0.0, 0.0, 1 + 2.0**-23, 2.0**-24, -1.0),
                  (0.0, 0.0, 2.0**-24, 1 + 2.0**-23, -1.0),
                  (0.5, 0.0, 0.0, 0.0, 0.0), (0.5, 0.0, 0.5, 0.0, 0.0)],
                 id="sum-of-three-no-strict"),
])
def test_linear_witness_check_accepts_exactly_the_counterexamples_of_the_text(
        tmp_path, text, points):
    """A linear 2.0 spec whose rows can accept a point the checker rejects (a
    strict comparison read as its closure, a sum the checker does in float32)
    keeps its source assertions, and rows plus assertions accept exactly the
    points the text accepts both exactly and in the checker's float32
    evaluation; on a strict boundary or at the float32 sum the rows alone
    accept, so without the assertions an invalid witness passes. Catches the
    assertions kept only when strictness and arithmetic both occur, and only
    a part of the asserts kept."""
    parsed = _load(tmp_path, text)
    t = _Text(text)
    names = ["X[0,0]", "X[0,1]", "Y[0,0]", "Y[0,1]", "Y[0,2]"]
    bad = []
    for p in points:
        values = dict(zip(names, p))
        want = t.accepts(values) and t.accepts(values, mode="f32")
        got = _witness_check(parsed, p[:2], p[2:])
        if got != want:
            bad.append((p, got, want))
    assert not bad, (f"{len(bad)} points (point, parse, text), "
                     f"e.g. {bad[:3]}")
