"""Specification parsing contract (DESIGN sections 5 and 17).

A parsed VNN-LIB file means what the file says. Each test parses a small
hand-built file and compares n2v's reading with ``naive_eval``, an evaluator
that works on the file's text and shares no code with the parser, at points
chosen by hand on both sides of every bound. A point is a counterexample of
the file exactly when every assert holds there.

The promise for every file: ``load_vnnlib`` raises ``ValueError`` (the planned
``SpecError`` subclasses it), or the stored regions accept exactly the points
the text accepts. A refusal keeps the promise only for a form that DESIGN
section 5 does not list as read, or whose registry row accepts a refusal as
the fix (``_Case.may_refuse``); a form the parsers read (a box, input regions,
a paired ``or``, output rows, an ``or`` of ``and`` over outputs, a 2.0 strict
or number-first comparison, 2.0 arithmetic) must be read, and a refusal of it
fails the test. The two directions are two different harms and are tested
separately:

* a stored region that leaves out a point the text accepts can turn a
  counterexample into a false UNSAT
  (``test_stored_regions_hold_every_counterexample_of_the_text``);
* a parse that accepts a point the text rejects can report an invalid SAT
  (``test_parse_accepts_no_point_the_text_rejects``). For a 2.0 file the
  parse's acceptance includes the source assertions it keeps for the SAT check.

The older tests at the top compare with ``tests/differential/naive_eval`` at
points away from every bound. The newer ones use ``_Text`` below, which reads
numbers as exact decimals (as SMT-LIB does: ``0.1`` is 1/10, not the double
nearest to it) and evaluates in fractions, so a bound rounded the wrong way by
one unit in the last place is seen. Points are float64 numbers.

Convention: SAT = unsafe. The parsed ``Hg`` rows describe the unsafe region
``G y <= g`` and the parsed ``lb``/``ub`` the input region.
"""

import itertools
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pytest
import torch

from n2v.nn.neural_network import NeuralNetwork
from n2v.sets.star import Star
from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.verify_specification import verify_specification
from n2v.utils.vnnlib2 import evaluate_nonlinear
from tests.known_bugs import known_bug
from tests.known_bugs import DID_NOT_RAISE
from tests.differential import naive_eval
from tests.differential.point_agreement import pairs_satisfied

_HEADER = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n"


class _SetupError(Exception):
    """A case that cannot test what it claims: no counterexample among its
    points, no point the text rejects, or a reference that gives a text a
    meaning the test says it has not. Its own type, so that no strict
    expected-failure marker of this file absorbs it (the markers absorb
    ``AssertionError``, pytest's ``Failed``, ``IndexError``, ``TypeError``,
    ``AttributeError`` or ``ValueError``)."""


def _parse(tmp_path, body):
    path = tmp_path / "spec.vnnlib"
    path.write_text(_HEADER + body)
    return load_vnnlib(str(path))


def _disagreements(result, body, points):
    """The points where n2v's parse and the text disagree, with both
    verdicts: ``(x, y, parsed, text)``."""
    spec = naive_eval.NaiveSpec(_HEADER + body)
    out = []
    for x, y in points:
        parsed = pairs_satisfied(result["pairs"], np.array([x]), np.array([y]))
        text = spec.satisfied({"X_0": x, "Y_0": y})
        if parsed != text:
            out.append((x, y, parsed, text))
    return out


# Inputs on both sides of 0.25, 0.5 and 0.75, each with an output on both
# sides of 0.9. Every bound used below is a dyadic number away from the points.
_POINTS = [(x, y) for x in (0.125, 0.375, 0.625, 0.875) for y in (0.5, 0.95)]


# --------------------------------------------------------------------------- #
# VNN-LIB 1.0, one assert per line                                              #
# --------------------------------------------------------------------------- #
@known_bug(6, "a 1.0 input bound written after an output assert is parsed as "
              "an output row: the unsafe region comes out empty (false UNSAT)")
def test_v1_input_bound_after_an_output_assert_is_an_input_bound(tmp_path):
    body = ("(assert (>= X_0 0.0))\n"
            "(assert (<= X_0 1.0))\n"
            "(assert (>= Y_0 0.9))\n"
            "(assert (<= X_0 0.5))\n")
    result = _parse(tmp_path, body)
    # x = 0.375 with y = 0.95 is a counterexample of the file; x = 0.625 is
    # outside the input region.
    assert _disagreements(result, body, _POINTS) == []
    assert np.asarray(result["ub"]).tolist() == [0.5]


@known_bug(6, "a 1.0 single-line strict '>' on an input is stored as an "
              "upper bound: the input region is the complement of the stated "
              "one (false UNSAT)")
def test_v1_strict_lower_bound_is_a_lower_bound(tmp_path):
    body = ("(assert (>= X_0 0.0))\n"
            "(assert (<= X_0 1.0))\n"
            "(assert (> X_0 0.25))\n"
            "(assert (>= Y_0 0.9))\n")
    result = _parse(tmp_path, body)
    assert _disagreements(result, body, _POINTS) == []


def test_v1_strict_upper_bound_is_an_upper_bound(tmp_path):
    body = ("(assert (>= X_0 0.0))\n"
            "(assert (< X_0 0.75))\n"
            "(assert (>= Y_0 0.9))\n")
    result = _parse(tmp_path, body)
    assert _disagreements(result, body, _POINTS) == []


@pytest.mark.parametrize("body", [
    pytest.param("(assert (>= X_0 0.0))\n"
                 "(assert (<= X_0 0.5))\n"
                 "(assert (<= X_0 1.0))\n"
                 "(assert (>= Y_0 0.9))\n", id="upper"),
    pytest.param("(assert (>= X_0 0.5))\n"
                 "(assert (>= X_0 0.0))\n"
                 "(assert (<= X_0 1.0))\n"
                 "(assert (>= Y_0 0.9))\n", id="lower"),
])
@known_bug(6, "a 1.0 input bound stated twice keeps the last value instead "
              "of the tighter one: points outside the stated region are "
              "accepted as counterexamples (invalid SAT)")
def test_v1_repeated_input_bound_keeps_the_tighter_value(tmp_path, body):
    result = _parse(tmp_path, body)
    assert _disagreements(result, body, _POINTS) == []


@pytest.mark.parametrize("body", [
    pytest.param("(assert (>= X_0 0.0))\n"
                 "(assert (<= X_0 1.0))\n"
                 "(assert (<= X_0 0.5))\n"
                 "(assert (>= Y_0 0.9))\n", id="upper"),
    pytest.param("(assert (>= X_0 0.0))\n"
                 "(assert (>= X_0 0.5))\n"
                 "(assert (<= X_0 1.0))\n"
                 "(assert (>= Y_0 0.9))\n", id="lower"),
])
def test_v1_repeated_input_bound_stated_tighter_last_keeps_the_tighter_value(
        tmp_path, body):
    """The tighter of two values stated for the same bound is the stated
    region, here with the tighter value stated last. The test above states it
    first and fails today because the parser keeps the last value; these two
    cases pass today and fail for a fix that keeps the first value stated, so
    the four cases together accept only "keep the tighter". Same reference
    and points as the test above."""
    result = _parse(tmp_path, body)
    assert _disagreements(result, body, _POINTS) == []


@pytest.mark.parametrize("body", [
    pytest.param("(assert (>= X_0 0.0))\n"
                 "(assert (<= X_0 1.0))\n", id="bounds-only"),
    pytest.param("(assert (or (and (>= X_0 0.0) (<= X_0 0.25)) "
                 "(and (>= X_0 0.75) (<= X_0 1.0))))\n", id="regions-only"),
])
def test_v1_file_without_an_output_assert_is_refused(tmp_path, body):
    """A file that states no unsafe region has no verdict. The parser must
    refuse it; an empty ``prop`` handed on would verify as UNSAT."""
    with pytest.raises(ValueError, match="empty"):
        _parse(tmp_path, body)


# --------------------------------------------------------------------------- #
# Paired files: region i goes with unsafe region i                              #
# --------------------------------------------------------------------------- #
_PAIRED = ("(assert (or "
           "(and (>= X_0 0.0) (<= X_0 0.25) (>= Y_0 0.9)) "
           "(and (>= X_0 0.75) (<= X_0 1.0) (<= Y_0 0.6))))\n")


def test_paired_file_pairs_each_region_with_its_own_unsafe_region(tmp_path):
    result = _parse(tmp_path, _PAIRED)
    points = [(x, y) for x in (0.125, 0.5, 0.875) for y in (0.5, 0.75, 0.95)]
    assert _disagreements(result, _PAIRED, points) == []


@known_bug(6, "a paired file still carries a top-level 'prop' listing every "
              "region's unsafe set; read as one specification it is their "
              "conjunction, which is empty here (false UNSAT)")
def test_paired_file_has_no_top_level_prop(tmp_path):
    """The per-region reading lives in ``pairs``. A top-level ``prop`` would
    be read by ``verify_specification`` as the AND of its entries, here
    ``y >= 0.9`` and ``y <= 0.6``, so a caller looping over regions with it
    gets UNSAT for outputs that region 0 calls unsafe."""
    result = _parse(tmp_path, _PAIRED)
    assert "prop" not in result


# =========================================================================== #
# The reference: what the text says                                            #
# =========================================================================== #
class _Undefined(Exception):
    """The text names something that has no meaning (an undeclared variable,
    an operator outside VNN-LIB). Only a refusal keeps the promise then."""


_COMPARISONS = {"<=", "<", ">=", ">", "=", "==", "!=", "distinct"}


def _sexprs(text):
    """Every top-level s-expression of ``text``, comments removed."""
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


class _Text:
    """The meaning of a VNN-LIB file, read from its text alone.

    A point is a counterexample exactly when every ``(assert F)`` holds.
    ``F`` is built from ``and``, ``or``, the comparisons ``<= < >= > = ==
    != distinct`` in prefix form (or infix ``(a op b)``) and the arithmetic
    ``+ - * /`` over variables and numbers. Numbers are exact decimals
    (``Fraction("0.1") == 1/10``) and every operation is exact, so the
    evaluator decides each comparison without rounding. Variables are looked
    up by their spelling in the file (``X_0``, ``X[0,0]``); a name it is not
    given is undefined. Declarations are ignored: they give no constraint."""

    def __init__(self, text):
        self.asserts = [e[1] for e in _sexprs(text)
                        if isinstance(e, list) and e and e[0] == "assert"]

    def accepts(self, values):
        values = {name: Fraction(v) for name, v in values.items()}
        return all(self._holds(a, values) for a in self.asserts)

    def _holds(self, node, values):
        if not isinstance(node, list) or not node:
            raise _Undefined(f"not a formula: {node!r}")
        head = node[0]
        if head == "and":
            return all(self._holds(c, values) for c in node[1:])
        if head == "or":
            return any(self._holds(c, values) for c in node[1:])
        if isinstance(head, str) and head in _COMPARISONS and len(node) == 3:
            op, a, b = node
        elif (len(node) == 3 and isinstance(node[1], str)
              and node[1] in _COMPARISONS):
            a, op, b = node
        else:
            raise _Undefined(f"no meaning for {node!r}")
        a, b = self._value(a, values), self._value(b, values)
        return {"<=": a <= b, "<": a < b, ">=": a >= b, ">": a > b,
                "=": a == b, "==": a == b, "!=": a != b,
                "distinct": a != b}[op]

    def _value(self, node, values):
        if isinstance(node, str):
            if node in values:
                return values[node]
            try:
                return Fraction(node)
            except ValueError:
                raise _Undefined(f"unknown name {node!r}") from None
        args = [self._value(c, values) for c in node[1:]]
        head = node[0]
        if head == "+":
            return sum(args, Fraction(0))
        if head == "-":
            return -args[0] if len(args) == 1 else args[0] - sum(args[1:],
                                                                 Fraction(0))
        if head == "*":
            out = Fraction(1)
            for a in args:
                out *= a
            return out
        if head == "/" and len(args) == 2:
            return args[0] / args[1]
        raise _Undefined(f"no meaning for {node!r}")


_SELF_TEST = [
    # (text, values, accepted)
    ("(assert (<= X_0 0.5))", {"X_0": 0.5}, True),
    ("(assert (< X_0 0.5))", {"X_0": 0.5}, False),
    ("(assert (> X_0 0.25))", {"X_0": 0.25}, False),
    ("(assert (>= X_0 0.25))", {"X_0": 0.25}, True),
    ("(assert (<= 0.25 X_0))", {"X_0": 0.125}, False),
    ("(assert (< 0.25 X_0))", {"X_0": 0.375}, True),
    ("(assert (>= X_0 0.0))\n(assert (<= X_0 1.0))", {"X_0": 1.125}, False),
    ("(assert (and (>= X_0 0.25) (<= X_0 0.5)))", {"X_0": 0.375}, True),
    ("(assert (and (>= X_0 0.25) (<= X_0 0.5)))", {"X_0": 0.625}, False),
    ("(assert (or (<= X_0 0.25) (>= X_0 0.75)))", {"X_0": 0.5}, False),
    ("(assert (or (<= X_0 0.25) (>= X_0 0.75)))", {"X_0": 0.75}, True),
    ("(assert (= X_0 0.5))", {"X_0": 0.5}, True),
    ("(assert (== X[0,0] 0.5))", {"X[0,0]": 0.625}, False),
    ("(assert (!= X[0,0] 0.5))", {"X[0,0]": 0.5}, False),
    ("(assert (distinct X_0 0.5))", {"X_0": 0.25}, True),
    ("(assert (<= Y_0 Y_1))", {"Y_0": 0.5, "Y_1": 0.25}, False),
    ("(assert (>= X_0 (- 0.5)))", {"X_0": -0.5}, True),
    ("(assert (>= X_0 -0.5))", {"X_0": -0.625}, False),
    ("(assert (<= X_0 5e-1))", {"X_0": 0.5}, True),
    # numbers are exact decimals: 0.1 * 3 is 0.3, and the double next to 3
    # from above is out
    ("(assert (<= (* 0.1 X[0,0]) 0.3))", {"X[0,0]": 3.0}, True),
    ("(assert (<= (* 0.1 X[0,0]) 0.3))",
     {"X[0,0]": math.nextafter(3.0, 4.0)}, False),
    ("(assert (<= (+ X[0,0] 0.2) 0.3))", {"X[0,0]": 0.09999999999999999},
     True),
    ("(assert (<= (/ X[0,0] 0.1) 0.3))", {"X[0,0]": 0.03}, True),
    ("(assert (<= (/ X[0,0] 0.1) 0.3))",
     {"X[0,0]": math.nextafter(0.03, 1.0)}, False),
    ("(assert (X[0,0] <= 0.5))", {"X[0,0]": 0.5}, True),
    ("; (assert (<= X_0 0.0))\n(assert (<= X_0 1.0))", {"X_0": 0.5}, True),
]


@pytest.mark.parametrize("text, values, accepted", _SELF_TEST)
def test_text_evaluator_reads_hand_written_cases(text, values, accepted):
    """The reference of the cells below, checked on cases whose meaning is
    read off by hand. A bug in the evaluator would make every cell that uses
    it check the wrong promise."""
    assert _Text(text).accepts(values) is accepted


@pytest.mark.parametrize("text, values", [
    ("(assert (<= X_3 0.5))", {"X_0": 0.5}),
    ("(assert (=> X_0 0.5))", {"X_0": 0.5}),
    ("(assert (not (<= X_0 0.5)))", {"X_0": 0.5}),
])
def test_text_evaluator_gives_no_meaning_to_unknown_names(text, values):
    with pytest.raises(_Undefined):
        _Text(text).accepts(values)


# --------------------------------------------------------------------------- #
# The parse's side                                                             #
# --------------------------------------------------------------------------- #
def _f(value):
    """A stored float64 as an exact fraction (infinite bounds kept as floats)."""
    value = float(value)
    return value if math.isinf(value) else Fraction(value)


def _in_stored_region(result, x, y):
    """Is (x, y) in some stored pair: x in [lb, ub] and y in every group's
    union of polytopes ``G y <= g``? Decided in exact arithmetic on the
    stored float64 numbers."""
    for pair in result["pairs"]:
        lb = np.asarray(pair["lb"], dtype=np.float64).ravel()
        ub = np.asarray(pair["ub"], dtype=np.float64).ravel()
        assert lb.size == len(x) and ub.size == len(x), (
            f"the parse has {lb.size} input entries, the text {len(x)}")
        if not all(_f(lo) <= Fraction(xi) <= _f(hi)
                   for lo, xi, hi in zip(lb, x, ub)):
            continue
        if all(_in_union(group["Hg"], y) for group in pair["prop"]):
            return True
    return False


def _in_union(hg, y):
    for h in (hg if isinstance(hg, list) else [hg]):
        G = np.asarray(h.G, dtype=np.float64)
        g = np.asarray(h.g, dtype=np.float64).ravel()
        assert G.shape[1] == len(y), (
            f"the parse has {G.shape[1]} output entries, the text {len(y)}")
        if all(sum((_f(G[i, j]) * Fraction(y[j]) for j in range(len(y))),
                   Fraction(0)) <= _f(g[i]) for i in range(G.shape[0])):
            return True
    return False


def _parse_accepts(result, x, y):
    """What the parse calls a counterexample: in a stored region, and, when
    the parse keeps the source assertions for the SAT check (2.0 files with
    a strict comparison or arithmetic), accepted by them."""
    if not _in_stored_region(result, x, y):
        return False
    if "assertions" in result:
        return bool(evaluate_nonlinear(result, np.array(x, dtype=np.float64),
                                       np.array(y, dtype=np.float64)))
    return True


# --------------------------------------------------------------------------- #
# The cases                                                                    #
# --------------------------------------------------------------------------- #
def _v1_header(n_in=1, n_out=1):
    return ("".join(f"(declare-const X_{i} Real)\n" for i in range(n_in))
            + "".join(f"(declare-const Y_{j} Real)\n" for j in range(n_out)))


def _v2_header(n_in=1, n_out=1):
    return ("(vnnlib-version <2.0>)\n(declare-network N\n"
            f"  (declare-input X float32 [1,{n_in}])\n"
            f"  (declare-output Y float32 [1,{n_out}])\n)\n")


# Inputs on and around every bound used below (multiples of 1/8 from -1/8 to
# 9/8) and outputs on both sides of 0.875 and on it, and one above 1, so an
# input bound misread as an output row is seen in both directions.
_XS = tuple(k / 8 for k in range(-1, 10))
_YS = (0.5, 0.75, 0.875, 0.9375, 1.0, 1.125)


@dataclass
class _Case:
    version: str
    body: str
    n_in: int = 1
    n_out: int = 1
    xs: tuple = _XS
    ys: tuple = _YS
    header: str | None = None
    region_mark: object = None
    sat_mark: object = None
    extra_x: tuple = field(default_factory=tuple)
    # The scaled cases are checked in the region direction only: their SAT
    # side is decided by the kept source assertions, which the witness check
    # evaluates in float32 as the grader does, and an exact decimal reading
    # and a float reading disagree within rounding of the boundary. That SAT
    # side is checked in float32 by
    # tests/unit/vnncomp/test_runner_witness_regressions.py::
    # test_three_term_sum_checked_in_float32_is_not_sat and
    # ::test_coefficient_checked_in_float32.
    sat_direction: bool = True
    # True for a form DESIGN section 5 does not list as read, or whose
    # registry row accepts a refusal as the fix: a ValueError keeps the
    # promise then. Otherwise the form is read, and a refusal is a failure
    # (a supported form refused: the instance gets no verdict).
    may_refuse: bool = False
    # Explicit (x, y) points, used instead of the product of xs and ys when
    # the case has too many variables for a product (two-digit indices).
    point_list: tuple = ()

    @property
    def text(self):
        if self.header is not None:
            return self.header + self.body
        make = _v1_header if self.version == "1.0" else _v2_header
        return make(self.n_in, self.n_out) + self.body

    def names(self):
        if self.version == "1.0":
            return ([f"X_{i}" for i in range(self.n_in)],
                    [f"Y_{j}" for j in range(self.n_out)])
        return ([f"X[0,{i}]" for i in range(self.n_in)],
                [f"Y[0,{j}]" for j in range(self.n_out)])

    def points(self):
        if self.point_list:
            yield from self.point_list
            return
        xs = sorted(set(self.xs) | set(self.extra_x))
        for x in itertools.product(xs, repeat=self.n_in):
            for y in itertools.product(self.ys, repeat=self.n_out):
                yield x, y

    def values(self, x, y):
        x_names, y_names = self.names()
        return dict(zip(x_names, x)) | dict(zip(y_names, y))


def _doubles_around(q, k=3):
    """The ``k`` float64 numbers on each side of the rational ``q`` and the
    nearest one."""
    centre = float(q)
    out = {centre}
    up = down = centre
    for _ in range(k):
        up, down = math.nextafter(up, math.inf), math.nextafter(down, -math.inf)
        out |= {up, down}
    return tuple(sorted(out))


_B1 = "(assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n"
_O1 = "(assert (>= Y_0 0.875))\n"
_X2, _Y2 = "X[0,0]", "Y[0,0]"
_B2 = f"(assert (>= {_X2} 0.0))\n(assert (<= {_X2} 1.0))\n"
_O2 = f"(assert (>= {_Y2} 0.875))\n"

_SPEC2 = known_bug(
    6, "a 1.0 input bound written after an input-only (or ...) assert, or "
       "after an input line whose text (a trailing comment included) "
       "contains the letters 'or' anywhere (here inside 'for'), is parsed as "
       "an output row: an upper bound read so makes the unsafe region empty "
       "or smaller (false UNSAT), a lower bound or a second input-only "
       "(or ...) leaves the input region wider than stated (invalid SAT)")
_SPEC5 = known_bug(
    6, "the 1.0 parser stores a strict comparison as the closed one and "
       "keeps no source assertion, so a point exactly on a strict input or "
       "output bound is accepted although the file rejects it (invalid SAT)")
_SPEC6 = known_bug(
    6, "a 1.0 input line is read by its last comparison only: a top-level "
       "(assert (and ...)) over inputs keeps only its last atom, and of two "
       "asserts written on one line the first is lost; the stored region is "
       "wider than the stated one (invalid SAT)")
_SPEC7 = known_bug(
    6, "a 1.0 assert with no '<' or '>' in it ((= X_i c), (distinct X_i c), "
       "or (= Y_j c) before the first output inequality) is dropped without "
       "a message: the stored region is wider than the stated one "
       "(invalid SAT)")
_SPEC9 = known_bug(
    6, "a 2.0 (!= X c) on an input is stored as a one-sided bound with no "
       "source assertion kept: the stored region leaves out one side of the "
       "stated one (false UNSAT) and takes in x = c (invalid SAT)")
_SPEC10 = known_bug(
    6, "a 2.0 input bound with constant arithmetic ((* a X), (+ X b)) is "
       "solved for X with round to nearest and can be stored inward: "
       "(<= (* 0.1 X) 0.3) gives the upper bound 2.9999999999999996 and "
       "leaves out x = 3.0 (false UNSAT kind)")
_OUTPUT_ROW_ROUNDED = known_bug(
    6, "a 2.0 output comparison with a non-unit coefficient is stored with "
       "its constants rounded to nearest and nothing outward: "
       "(<= (* 0.1 Y) 0.3) becomes 0.1 y <= 0.3 in float64, which leaves "
       "out y = 3.0 (false UNSAT kind)")
_ASSERT_BEFORE_OUTPUT_DECLARATIONS = known_bug(
    6, "a 1.0 assert written before the output declarations is dropped "
       "without a message: the stored region is wider than the stated one "
       "(invalid SAT)")
_AFTER_COMBINED = known_bug(
    6, "a 1.0 assert written after a combined input/output (or ...) assert "
       "is misread: an output assert replaces the regions and their paired "
       "output rows by the declared box and itself, and an input bound is "
       "dropped; the stored region is wider than the stated one "
       "(invalid SAT)")
_SAME_OUTPUT_TWICE = known_bug(
    6, "a 1.0 output comparison of an output with itself, (<= Y_0 Y_0) or "
       "(>= Y_0 Y_0), is stored as y0 >= 0 or y0 <= 0 instead of a row that "
       "always holds: the unsafe region leaves out half the outputs "
       "(false UNSAT)")
_SAME_OUTPUT_TWICE_IN_OR = known_bug(
    6, "inside an output-only (or ...) or (or (and ...)) assert of 1.0, an "
       "output compared with itself, (<= Y_0 Y_0), is stored as the "
       "disjunct y0 >= 0 instead of one that always holds (the same "
       "overwrite in _process_constraint, reached through _process_or and "
       "_process_and): the unsafe region leaves out outputs (false UNSAT)")
_OUTPUT_ATOM_READ_AS_INPUT = known_bug(
    6, "a 1.0 top-level (assert (and ...)) on one line that mixes an input "
       "and an output comparison is read as an input bound by its last "
       "comparison, and an output name there is read as the input of the "
       "same index: (and (<= X_0 0.5) (>= Y_0 0.875)) is stored as "
       "x >= 0.875 with no output row, so counterexamples are left out "
       "(false UNSAT) and rejected points are accepted (invalid SAT)")


def _scaled_input(lhs, op, c, q, mark=None):
    """A 2.0 input bound ``(op lhs c)`` whose exact solution for X is ``q``,
    inside the box [-100, 100], checked at the float64 numbers around ``q``."""
    return _Case("2.0", (f"(assert (>= {_X2} -100.0))\n"
                         f"(assert (<= {_X2} 100.0))\n"
                         f"(assert ({op} {lhs} {c}))\n" + _O2),
                 xs=_doubles_around(q), ys=(0.5, 0.875), region_mark=mark,
                 sat_direction=False)


def _scaled_output(lhs, op, c, q, mark=None):
    """A 2.0 output row ``(op lhs c)`` whose exact solution for Y is ``q``,
    checked at the float64 numbers around ``q``."""
    return _Case("2.0", (f"(assert (>= {_X2} 0.0))\n(assert (<= {_X2} 4.0))\n"
                         f"(assert ({op} {lhs} {c}))\n"),
                 xs=(0.5,), ys=_doubles_around(q), region_mark=mark,
                 sat_direction=False)


def _around(*bounds):
    """Each bound, and the multiples of 1/8 next to it on both sides."""
    return tuple(sorted({b + d for b in bounds for d in (-0.125, 0.0, 0.125)}))


def _one_coordinate_at_a_time(bases, x_values, y_values):
    """Each base point (x, y), and each base point with one input or one
    output replaced by each value listed for it: points around every bound of
    every variable, without the product over all of them (twelve variables
    with five values each would be 244 million points)."""
    out = []
    for x, y in bases:
        out.append((tuple(x), tuple(y)))
        for i, values in enumerate(x_values):
            out += [(x[:i] + (v,) + x[i + 1:], tuple(y)) for v in values]
        for j, values in enumerate(y_values):
            out += [(tuple(x), y[:j] + (v,) + y[j + 1:]) for v in values]
    return tuple(dict.fromkeys(out))


# Input i in [i, i + 1/2]; X_10 in [10, 10.5] or [12, 12.5] in the blocks.
_TWELVE_BOXES = "".join(f"(assert (>= X_{i} {i}.0))\n(assert (<= X_{i} {i}.5))\n"
                        for i in range(12))
_TEN_BOXES = "".join(f"(assert (>= X_{i} {i}.0))\n(assert (<= X_{i} {i}.5))\n"
                     for i in range(10))
_X10_X11_LOW = "(>= X_10 10.0) (<= X_10 10.5) (>= X_11 11.0) (<= X_11 11.5)"
_X10_X11_HIGH = "(>= X_10 12.0) (<= X_10 12.5) (>= X_11 11.0) (<= X_11 11.5)"
_TWELVE_CENTRES = tuple(i + 0.25 for i in range(12))
# y3 = 1 >= y11 = 0.5, y2 = 0.25 <= y10 = 0.75, y10 >= 0.5: every row holds.
_TWELVE_OUTPUTS = (0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.5)
_Y_STEPS = (-1.5, -1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
_TWO_DIGIT_BLOCK_POINTS = _one_coordinate_at_a_time(
    [(_TWELVE_CENTRES[:10] + (x10, 11.25), y)
     for x10 in (10.25, 12.25)
     # the first output block holds (y10 >= 0.5, y11 <= y1); the second
     # (y0 <= -1) only
     for y in ((0.0, 0.5) + (0.0,) * 8 + (0.75, 0.25),
               (-1.5, 0.0) + (0.0,) * 8 + (0.0, 1.0))],
    [_around(i, i + 0.5) for i in range(10)]
    + [_around(10.0, 10.5, 12.0, 12.5), _around(11.0, 11.5)],
    [_Y_STEPS] * 12)

_F = Fraction
_CASES = {
    # ---- VNN-LIB 1.0 -------------------------------------------------------
    "v1-box": _Case("1.0", _B1 + _O1),
    "v1-strict-lt": _Case("1.0", _B1 + "(assert (< X_0 0.75))\n" + _O1,
                          sat_mark=_SPEC5, may_refuse=True),
    "v1-strict-gt-in-or": _Case(
        "1.0", "(assert (or (and (>= X_0 0.0) (> X_0 0.25) "
               "(<= X_0 1.0))))\n" + _O1,
        sat_mark=_SPEC5, may_refuse=True),
    "v1-strict-lt-in-or": _Case(
        "1.0", "(assert (or (and (>= X_0 0.0) (< X_0 0.75) "
               "(<= X_0 1.0))))\n" + _O1,
        sat_mark=_SPEC5, may_refuse=True),
    "v1-strict-gt-output": _Case("1.0", _B1 + "(assert (> Y_0 0.875))\n",
                                 sat_mark=_SPEC5, may_refuse=True),
    "v1-strict-lt-output": _Case("1.0", _B1 + "(assert (< Y_0 0.875))\n",
                                 sat_mark=_SPEC5, may_refuse=True),
    "v1-strict-output-in-paired-or": _Case(
        "1.0", "(assert (or (and (>= X_0 0.0) (<= X_0 1.0) (> Y_0 0.875))))\n",
        sat_mark=_SPEC5, may_refuse=True),
    "v1-and-tighter-last": _Case(
        "1.0", _B1 + "(assert (and (<= X_0 1.0) (<= X_0 0.5)))\n" + _O1),
    "v1-and-tighter-first": _Case(
        "1.0", _B1 + "(assert (and (<= X_0 0.5) (<= X_0 1.0)))\n" + _O1,
        sat_mark=_SPEC6),
    "v1-and-lower-last": _Case(
        "1.0", _B1 + "(assert (and (<= X_0 0.5) (>= X_0 0.25)))\n" + _O1,
        sat_mark=_SPEC6),
    "v1-and-three-atoms": _Case(
        "1.0", _B1 + "(assert (and (>= X_0 0.25) (<= X_0 0.75) "
                     "(<= X_0 0.5)))\n" + _O1,
        sat_mark=_SPEC6),
    "v1-and-two-variables": _Case(
        "1.0", (_B1 + "(assert (>= X_1 0.0))\n(assert (<= X_1 1.0))\n"
                "(assert (and (<= X_0 0.5) (>= X_1 0.5)))\n" + _O1),
        n_in=2, xs=(0.0, 0.25, 0.5, 0.75, 1.0), sat_mark=_SPEC6),
    "v1-repeated-bound-in-one-and-group": _Case(
        "1.0", "(assert (or (and (>= X_0 0.0) (<= X_0 0.5) (<= X_0 1.0))))\n"
               + _O1),
    "v1-or-regions": _Case(
        "1.0", "(assert (or (and (>= X_0 0.0) (<= X_0 0.25)) "
               "(and (>= X_0 0.75) (<= X_0 1.0))))\n" + _O1),
    "v1-equality-input": _Case("1.0", _B1 + "(assert (= X_0 0.5))\n" + _O1,
                               sat_mark=_SPEC7, may_refuse=True),
    "v1-distinct-input": _Case(
        "1.0", _B1 + "(assert (distinct X_0 0.5))\n" + _O1, sat_mark=_SPEC7,
        may_refuse=True),
    "v1-equality-output-before-inequality": _Case(
        "1.0", _B1 + "(assert (= Y_0 0.875))\n" + _O1, sat_mark=_SPEC7,
        may_refuse=True),
    "v1-equality-output-after-inequality": _Case(
        "1.0", _B1 + _O1 + "(assert (= Y_0 0.9375))\n", may_refuse=True),
    "v1-upper-bound-after-input-or": _Case(
        "1.0", "(assert (or (and (>= X_0 0.0) (<= X_0 1.0))))\n"
               "(assert (<= X_0 0.5))\n" + _O1, region_mark=_SPEC2),
    "v1-lower-bound-after-input-or": _Case(
        "1.0", "(assert (or (and (>= X_0 0.0) (<= X_0 1.0))))\n"
               "(assert (>= X_0 0.5))\n" + _O1, sat_mark=_SPEC2),
    "v1-second-input-or": _Case(
        "1.0", "(assert (or (and (>= X_0 0.0) (<= X_0 1.0))))\n"
               "(assert (or (and (>= X_0 0.0) (<= X_0 0.5)) "
               "(and (>= X_0 0.75) (<= X_0 1.0))))\n" + _O1,
        region_mark=_SPEC2, sat_mark=_SPEC2),
    "v1-upper-bound-after-a-comment-with-or": _Case(
        "1.0", "(assert (>= X_0 0.0))\n"
               "(assert (<= X_0 1.0)) ; upper bound for x\n"
               "(assert (<= X_0 0.5))\n" + _O1, region_mark=_SPEC2),
    "v1-lower-bound-after-a-comment-with-or": _Case(
        "1.0", "(assert (>= X_0 0.0))\n"
               "(assert (<= X_0 1.0)) ; upper bound for x\n"
               "(assert (>= X_0 0.5))\n" + _O1, sat_mark=_SPEC2),
    "v1-bound-after-a-comment": _Case(
        "1.0", "(assert (>= X_0 0.0))\n"
               "(assert (<= X_0 1.0)) ; upper bound\n"
               "(assert (<= X_0 0.5))\n(assert (>= X_0 0.25))\n" + _O1),
    "v1-two-asserts-on-one-line": _Case(
        "1.0", _B1 + "(assert (<= X_0 0.5)) (assert (>= X_0 0.25))\n" + _O1,
        sat_mark=_SPEC6),
    "v1-paired-or": _Case(
        "1.0", _B1 + "(assert (or (and (<= X_0 0.5) (>= Y_0 0.875)) "
                     "(and (>= X_0 0.75) (<= Y_0 0.25))))\n",
        ys=(0.125, 0.25, 0.5, 0.875, 1.0, 1.125)),
    "v1-paired-or-with-an-input-only-disjunct": _Case(
        "1.0", _B1 + "(assert (or (and (<= X_0 0.25) (>= Y_0 0.875)) "
                     "(and (>= X_0 0.75))))\n",
        ys=(0.25, 0.5, 0.875, 1.0, 1.125)),
    "v1-multi-line-or": _Case(
        "1.0", _B1 + "(assert\n  (or (and (>= X_0 0.0) (<= X_0 0.25))\n"
               "    (and (>= X_0 0.75) (<= X_0 1.0))))\n" + _O1),
    "v1-paired-or-with-y-vs-y": _Case(
        "1.0", _B1 + "(assert (or (and (<= X_0 0.5) (<= Y_0 Y_1)) "
                     "(and (>= X_0 0.75) (>= Y_0 Y_1))))\n",
        n_out=2, ys=(0.25, 0.5, 0.75)),
    "v1-output-assert-after-paired-or": _Case(
        "1.0", _B1 + "(assert (or (and (<= X_0 0.5) (>= Y_0 0.875)) "
                     "(and (>= X_0 0.75) (<= Y_0 0.25))))\n"
                     "(assert (>= Y_0 0.5))\n",
        ys=(0.125, 0.25, 0.5, 0.875, 1.0, 1.125), sat_mark=_AFTER_COMBINED,
        may_refuse=True),
    "v1-input-bound-after-paired-or": _Case(
        "1.0", _B1 + "(assert (or (and (<= X_0 0.5) (>= Y_0 0.875)) "
                     "(and (>= X_0 0.75) (<= Y_0 0.25))))\n"
                     "(assert (>= X_0 0.25))\n",
        ys=(0.125, 0.25, 0.5, 0.875, 1.0, 1.125), sat_mark=_AFTER_COMBINED,
        may_refuse=True),
    "v1-output-y0-le-y0": _Case("1.0", _B1 + "(assert (<= Y_0 Y_0))\n",
                                ys=(-0.5, 0.0, 0.5),
                                region_mark=_SAME_OUTPUT_TWICE),
    "v1-output-y0-ge-y0": _Case("1.0", _B1 + "(assert (>= Y_0 Y_0))\n",
                                ys=(-0.5, 0.0, 0.5),
                                region_mark=_SAME_OUTPUT_TWICE),
    "v1-bound-before-output-declarations": _Case(
        "1.0", "(assert (or (and (>= X_0 0.0) (<= X_0 1.0))))\n" + _O1,
        header="(declare-const X_0 Real)\n(assert (<= X_0 0.5))\n"
               "(declare-const Y_0 Real)\n",
        sat_mark=_ASSERT_BEFORE_OUTPUT_DECLARATIONS, may_refuse=True),
    "v1-scientific-notation": _Case(
        "1.0", "(assert (>= X_0 0.0))\n(assert (<= X_0 5e-1))\n" + _O1),
    "v1-negative-literal": _Case(
        "1.0", "(assert (>= X_0 -0.5))\n(assert (<= X_0 1.0))\n" + _O1,
        extra_x=(-0.75, -0.5, -0.25)),
    "v1-output-y-le-y": _Case("1.0", _B1 + "(assert (<= Y_0 Y_1))\n",
                              n_out=2, ys=(0.25, 0.5, 0.75)),
    "v1-output-y-ge-y": _Case("1.0", _B1 + "(assert (>= Y_0 Y_1))\n",
                              n_out=2, ys=(0.25, 0.5, 0.75)),
    "v1-output-or-of-and": _Case(
        "1.0", _B1 + "(assert (or (and (<= Y_0 Y_1)) (and (>= Y_0 0.875))))\n",
        n_out=2, ys=(0.25, 0.5, 0.875, 1.0)),
    "v1-output-rows-in-two-asserts": _Case(
        "1.0", _B1 + "(assert (>= Y_0 0.5))\n(assert (<= Y_0 0.875))\n",
        ys=(0.25, 0.5, 0.75, 0.875, 1.0)),
    "v1-output-or-then-row": _Case(
        "1.0", _B1 + "(assert (or (<= Y_0 0.625) (>= Y_0 0.9375)))\n"
                     "(assert (>= Y_0 0.5))\n",
        ys=(0.25, 0.5, 0.625, 0.75, 0.9375, 1.0)),
    "v1-output-two-ors": _Case(
        "1.0", _B1 + "(assert (or (<= Y_0 0.625) (>= Y_0 0.9375)))\n"
                     "(assert (or (>= Y_0 0.5) (<= Y_0 0.25)))\n",
        ys=(0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.9375, 1.0)),
    # An (or (and ...)) over outputs with two comparisons in each block: every
    # comparison of a block is a row with its own constant (_process_and).
    "v1-output-or-of-and-with-two-atoms": _Case(
        "1.0", _B1 + "(assert (or (and (>= Y_0 Y_1) (>= Y_0 Y_2)) "
                     "(and (<= Y_1 0.25) (>= Y_2 -0.5))))\n",
        n_out=3, xs=(-0.125, 0.0, 0.5, 1.0, 1.125),
        ys=(-0.75, 0.0, 0.25, 0.5, 0.75)),
    "v1-output-or-with-y0-vs-y0": _Case(
        "1.0", _B1 + "(assert (or (<= Y_0 Y_0) (>= Y_1 0.875)))\n",
        n_out=2, ys=(-0.5, 0.0, 0.5, 0.875, 1.0),
        region_mark=_SAME_OUTPUT_TWICE_IN_OR),
    "v1-output-or-of-and-with-y0-vs-y0": _Case(
        "1.0", _B1 + "(assert (or (and (<= Y_0 Y_0)) (and (>= Y_1 0.875))))\n",
        n_out=2, ys=(-0.5, 0.0, 0.5, 0.875, 1.0),
        region_mark=_SAME_OUTPUT_TWICE_IN_OR),
    # The combined reader adds the two coefficients, so the always-true row
    # is stored right there: the passing neighbour of the two cases above.
    "v1-paired-or-with-y0-vs-y0": _Case(
        "1.0", _B1 + "(assert (or (and (<= X_0 0.5) (<= Y_0 Y_0)) "
                     "(and (>= X_0 0.75) (>= Y_0 Y_0))))\n",
        ys=(-0.5, 0.0, 0.5)),
    "v1-input-or-of-single-comparisons": _Case(
        "1.0", _B1 + "(assert (or (<= X_0 0.25) (>= X_0 0.75)))\n" + _O1),
    "v1-paired-or-with-a-single-comparison-disjunct": _Case(
        "1.0", _B1 + "(assert (or (and (<= X_0 0.5) (>= Y_0 0.875)) "
                     "(<= Y_0 0.25)))\n",
        ys=(0.125, 0.25, 0.5, 0.875, 1.0)),
    # The bound split over lines ends its last token with ')' (0.75 must not
    # lose a digit), and the statement right after a multi-line assert is
    # read (the count of merged lines must be exact).
    "v1-multi-line-input-bound": _Case(
        "1.0", "(assert (>= X_0 0.0))\n(assert\n  (<= X_0 0.75)\n)\n" + _O1),
    "v1-multi-line-output-then-row": _Case(
        "1.0", _B1 + "(assert\n  (>= Y_0 0.5))\n(assert (<= Y_0 0.875))\n",
        ys=(0.25, 0.5, 0.75, 0.875, 1.0)),
    # Decimal bounds whose nearest double lies inside the stated interval
    # (0.1 rounds up, 0.7 rounds down): read in float64 the stored box holds
    # exactly the doubles the text accepts; rounded to float32 it does not.
    "v1-box-with-decimal-bounds": _Case(
        "1.0", "(assert (>= X_0 0.1))\n(assert (<= X_0 0.7))\n" + _O1,
        xs=_doubles_around(_F(1, 10)) + _doubles_around(_F(7, 10)) + (0.5,),
        ys=(0.5, 0.875)),
    # lb == ub is a region of one point in that coordinate, not an empty one.
    "v1-point-dimension": _Case(
        "1.0", (_B1 + "(assert (>= X_1 0.5))\n(assert (<= X_1 0.5))\n"
                + _O1),
        n_in=2, xs=(-0.125, 0.0, 0.375, 0.5, 0.625, 1.0, 1.125)),
    "v1-mixed-and-with-an-output-row": _Case(
        "1.0", _B1 + "(assert (and (<= X_0 0.5) (>= Y_0 0.875)))\n"
                     "(assert (<= Y_0 0.9375))\n",
        region_mark=_OUTPUT_ATOM_READ_AS_INPUT,
        sat_mark=_OUTPUT_ATOM_READ_AS_INPUT, may_refuse=True),
    # Variable names with two-digit indices, single-line and inside blocks,
    # on inputs and on outputs (12 of each): checked one coordinate at a time.
    "v1-two-digit-input-index": _Case(
        "1.0", _TWELVE_BOXES + _O1, n_in=12,
        point_list=_one_coordinate_at_a_time(
            [(_TWELVE_CENTRES, (0.875,))],
            [_around(i, i + 0.5) for i in range(12)], [(0.5, 0.875, 1.0)])),
    "v1-two-digit-output-index": _Case(
        "1.0", _B1 + "(assert (<= Y_11 Y_3))\n(assert (<= Y_2 Y_10))\n"
                     "(assert (>= Y_10 0.5))\n", n_out=12,
        point_list=_one_coordinate_at_a_time(
            [((0.5,), _TWELVE_OUTPUTS)], [_around(0.0, 1.0)],
            [_Y_STEPS] * 12)),
    "v1-two-digit-index-in-an-input-or": _Case(
        "1.0", (_TEN_BOXES + "(assert (or (and " + _X10_X11_LOW + ") (and "
                + _X10_X11_HIGH + ")))\n(assert (or (and (>= Y_10 0.5) "
                "(<= Y_11 Y_1)) (and (<= Y_0 -1.0))))\n"),
        n_in=12, n_out=12, point_list=_TWO_DIGIT_BLOCK_POINTS),
    "v1-two-digit-index-in-a-paired-or": _Case(
        "1.0", (_TEN_BOXES + "(assert (or (and " + _X10_X11_LOW
                + " (>= Y_10 0.5) (<= Y_11 Y_1)) (and " + _X10_X11_HIGH
                + " (<= Y_0 -1.0))))\n"),
        n_in=12, n_out=12, point_list=_TWO_DIGIT_BLOCK_POINTS),
    # ---- VNN-LIB 2.0 -------------------------------------------------------
    "v2-box": _Case("2.0", _B2 + _O2),
    "v2-strict-gt": _Case("2.0", _B2 + f"(assert (> {_X2} 0.25))\n" + _O2),
    "v2-strict-lt-number-first": _Case(
        "2.0", _B2 + f"(assert (> 0.75 {_X2}))\n" + _O2),
    "v2-strict-in-or": _Case(
        "2.0", f"(assert (or (and (> {_X2} 0.25) (<= {_X2} 0.5)) "
               f"(and (>= {_X2} 0.625) (< {_X2} 0.875))))\n" + _O2),
    "v2-strict-output": _Case("2.0", _B2 + f"(assert (> {_Y2} 0.875))\n"),
    "v2-repeated-upper-looser-last": _Case(
        "2.0", f"(assert (>= {_X2} 0.0))\n(assert (<= {_X2} 0.5))\n"
               f"(assert (<= {_X2} 1.0))\n" + _O2),
    "v2-repeated-lower-looser-last": _Case(
        "2.0", f"(assert (>= {_X2} 0.5))\n(assert (>= {_X2} 0.0))\n"
               f"(assert (<= {_X2} 1.0))\n" + _O2),
    "v2-top-level-and": _Case(
        "2.0", _B2 + f"(assert (and (<= {_X2} 0.5) (>= {_X2} 0.25)))\n"
               + _O2),
    "v2-top-level-and-upper-last": _Case(
        "2.0", _B2 + f"(assert (and (>= {_X2} 0.25) (<= {_X2} 0.5)))\n"
               + _O2),
    "v2-output-and-of-or": _Case(
        "2.0", _B2 + f"(assert (and (or (<= {_Y2} 0.625) (>= {_Y2} 0.9375)) "
               f"(>= {_Y2} 0.5)))\n",
        ys=(0.25, 0.5, 0.625, 0.75, 0.9375, 1.0)),
    "v2-equality": _Case(
        "2.0", f"(assert (>= {_X2} -1.0))\n(assert (<= {_X2} 1.0))\n"
               f"(assert (== {_X2} 0.5))\n" + _O2,
        extra_x=(-0.75, -0.5, -0.25)),
    "v2-negative-number": _Case(
        "2.0", f"(assert (>= {_X2} (- 0.5)))\n(assert (<= {_X2} 1.0))\n" + _O2,
        extra_x=(-0.75, -0.5, -0.25)),
    "v2-paired-or-with-an-output-assert": _Case(
        "2.0", f"(assert (or (and (>= {_X2} 0.0) (<= {_X2} 0.5) "
               f"(>= {_Y2} 0.875)) (and (>= {_X2} 0.75) (<= {_X2} 1.0) "
               f"(<= {_Y2} 0.625))))\n(assert (<= {_Y2} 0.9375))\n",
        ys=(0.5, 0.625, 0.75, 0.875, 0.9375, 1.0)),
    "v2-input-or-boxed": _Case(
        "2.0", _B2 + f"(assert (or (<= {_X2} 0.25) (>= {_X2} 0.75)))\n" + _O2),
    "v2-paired-or-boxed": _Case(
        "2.0", _B2 + f"(assert (or (and (<= {_X2} 0.5) (>= {_Y2} 0.875)) "
               f"(and (>= {_X2} 0.75) (<= {_Y2} 0.25))))\n",
        ys=(0.125, 0.25, 0.5, 0.875, 1.0, 1.125)),
    "v2-disequality": _Case(
        "2.0", _B2 + f"(assert (!= {_X2} 0.5))\n" + _O2,
        region_mark=_SPEC9, sat_mark=_SPEC9, may_refuse=True),
    "v2-disequality-number-first": _Case(
        "2.0", _B2 + f"(assert (!= 0.5 {_X2}))\n" + _O2,
        region_mark=_SPEC9, sat_mark=_SPEC9, may_refuse=True),
    "v2-disequality-in-or": _Case(
        "2.0", f"(assert (or (and (>= {_X2} 0.0) (<= {_X2} 1.0) "
               f"(!= {_X2} 0.5))))\n" + _O2,
        region_mark=_SPEC9, sat_mark=_SPEC9, may_refuse=True),
    "v2-disequality-in-paired-or": _Case(
        "2.0", f"(assert (or (and (>= {_X2} 0.0) (<= {_X2} 1.0) "
               f"(!= {_X2} 0.5) (>= {_Y2} 0.875))))\n",
        region_mark=_SPEC9, sat_mark=_SPEC9, may_refuse=True),
    # Constant arithmetic in an input bound: the exact solution for X is a
    # float64 number, and the stored bound must not exclude it. The pairs
    # marked are those the parser rounds inward today.
    "v2-scaled-0.1x-le-0.3": _scaled_input(f"(* 0.1 {_X2})", "<=", "0.3",
                                           _F(3), _SPEC10),
    "v2-scaled-0.1x-le-0.7": _scaled_input(f"(* 0.1 {_X2})", "<=", "0.7",
                                           _F(7), _SPEC10),
    "v2-scaled-0.2x-le-0.6": _scaled_input(f"(* 0.2 {_X2})", "<=", "0.6",
                                           _F(3), _SPEC10),
    "v2-scaled-1.1x-le-3.3": _scaled_input(f"(* 1.1 {_X2})", "<=", "3.3",
                                           _F(3), _SPEC10),
    "v2-scaled-0.7x-ge-2.1": _scaled_input(f"(* 0.7 {_X2})", ">=", "2.1",
                                           _F(3), _SPEC10),
    "v2-shifted-x+0.2-le-0.3": _scaled_input(f"(+ {_X2} 0.2)", "<=", "0.3",
                                             _F(1, 10), _SPEC10),
    "v2-shifted-0.2-x-le-0.3": _scaled_input(f"(- 0.2 {_X2})", "<=", "0.3",
                                             _F(-1, 10), _SPEC10),
    "v2-scaled-0.3x-le-0.9": _scaled_input(f"(* 0.3 {_X2})", "<=", "0.9",
                                           _F(3)),
    "v2-scaled-0.1x-ge-0.3": _scaled_input(f"(* 0.1 {_X2})", ">=", "0.3",
                                           _F(3)),
    "v2-scaled-0.7x-le-2.1": _scaled_input(f"(* 0.7 {_X2})", "<=", "2.1",
                                           _F(3)),
    "v2-scaled-3x-ge-1": _scaled_input(f"(* 3 {_X2})", ">=", "1", _F(1, 3)),
    "v2-scaled-0.3x-le-0.1": _scaled_input(f"(* 0.3 {_X2})", "<=", "0.1",
                                           _F(1, 3)),
    "v2-divided-x/0.1-le-0.3": _scaled_input(f"(/ {_X2} 0.1)", "<=", "0.3",
                                             _F(3, 100)),
    "v2-shifted-x+0.1-le-0.3": _scaled_input(f"(+ {_X2} 0.1)", "<=", "0.3",
                                             _F(1, 5)),
    "v2-output-0.1y-le-0.3": _scaled_output(f"(* 0.1 {_Y2})", "<=", "0.3",
                                            _F(3), _OUTPUT_ROW_ROUNDED),
    "v2-output-0.2y-le-0.6": _scaled_output(f"(* 0.2 {_Y2})", "<=", "0.6",
                                            _F(3), _OUTPUT_ROW_ROUNDED),
    "v2-output-0.7y-ge-2.1": _scaled_output(f"(* 0.7 {_Y2})", ">=", "2.1",
                                            _F(3), _OUTPUT_ROW_ROUNDED),
    "v2-output-0.3y-le-0.9": _scaled_output(f"(* 0.3 {_Y2})", "<=", "0.9",
                                            _F(3)),
    "v2-output-2y-le-1": _scaled_output(f"(* 2.0 {_Y2})", "<=", "1.0",
                                        _F(1, 2)),
}


def _params(mark_of, sat=False):
    out = []
    for name, case in _CASES.items():
        if sat and not case.sat_direction:
            continue
        mark = mark_of(case)
        out.append(pytest.param(case, id=name,
                                marks=[mark] if mark is not None else []))
    return out


def _load(tmp_path, text):
    path = tmp_path / "spec.vnnlib"
    path.write_text(text)
    return load_vnnlib(str(path))


def _load_or_refuse(tmp_path, case):
    """The parse of the case's text, or None when the parser refuses it with
    ``ValueError`` and the form may be refused. A refusal of a form the
    parsers read fails through ``pytest.fail``, which no ``AssertionError``
    marker absorbs: a supported file refused gets no verdict at all."""
    try:
        return _load(tmp_path, case.text)
    except ValueError as err:
        if case.may_refuse:
            return None
        pytest.fail(f"the parser refused a form it reads: {err}")


def _verdicts(case):
    """The text's verdict on every point of the case."""
    text = _Text(case.text)
    return [(x, y, text.accepts(case.values(x, y))) for x, y in case.points()]


@pytest.mark.parametrize("case", _params(lambda c: c.region_mark))
def test_stored_regions_hold_every_counterexample_of_the_text(tmp_path, case):
    """The false UNSAT direction. Every point the text calls a counterexample
    must lie in a stored region: an input region narrower than stated, or an
    unsafe region smaller than stated, lets reach prove safe an instance that
    has a counterexample. Catches a lost or inverted bound, a strict
    comparison read on the wrong side, an assert read as the wrong kind, a
    one-sided reading of ``!=``, and a bound rounded inward."""
    verdicts = _verdicts(case)
    if not any(accepted for _, _, accepted in verdicts):
        raise _SetupError("the case has no counterexample among its points")
    result = _load_or_refuse(tmp_path, case)
    if result is None:
        return                     # a refusal keeps the promise
    left_out = [(x, y) for x, y, accepted in verdicts
                if accepted and not _in_stored_region(result, x, y)]
    assert left_out == [], (
        f"{len(left_out)} counterexamples of the text are outside every "
        f"stored region, e.g. {left_out[:3]}")


@pytest.mark.parametrize("case", _params(lambda c: c.sat_mark, sat=True))
def test_parse_accepts_no_point_the_text_rejects(tmp_path, case):
    """The invalid SAT direction. A point the parse accepts (in a stored
    region, and accepted by the source assertions the parse keeps) must be a
    counterexample of the text: otherwise a witness the grader rejects can be
    reported as SAT. Catches a dropped assert, an ``and`` that keeps one
    atom, a repeated bound that keeps the looser value, and a strict
    comparison stored closed with nothing to reject a point on it."""
    verdicts = _verdicts(case)
    if all(accepted for _, _, accepted in verdicts):
        raise _SetupError("the case has no point the text rejects")
    result = _load_or_refuse(tmp_path, case)
    if result is None:
        return                     # a refusal keeps the promise
    wrongly = [(x, y) for x, y, accepted in verdicts
               if not accepted and _parse_accepts(result, x, y)]
    assert wrongly == [], (
        f"{len(wrongly)} points the text rejects are accepted by the parse, "
        f"e.g. {wrongly[:3]}")


# --------------------------------------------------------------------------- #
# Read or refused: forms the 1.0 parser does not handle                        #
# --------------------------------------------------------------------------- #
_SPEC8 = known_bug(
    6, "a 1.0 comparison with the number first, (<= 0.0 X_0), crashes with "
       "IndexError instead of being read or refused with ValueError",
    raises=IndexError)
_SPEC8_OPERAND_FORM = known_bug(
    6, "the 1.0 parser crashes with IndexError on a comparison whose operand "
       "is not a plain variable or number ((- 0.5) as the number, a product "
       "(* 2.0 Y_0)) instead of reading it or refusing with ValueError",
    raises=IndexError)
_SPEC8_IN_A_BLOCK = known_bug(
    6, "a number written (- 0.5) inside a 1.0 (or (and ...)) block crashes "
       "with TypeError (float() of a list in _apply_block_atom) instead of "
       "being read or refused with ValueError", raises=TypeError)
_SPEC19_UNDECLARED = known_bug(
    6, "a 1.0 comparison that names a variable beyond the declared count "
       "(X_3 or Y_3 with one of each declared) crashes with IndexError from "
       "inside the parser instead of being refused with ValueError",
    raises=IndexError)
_SPEC19_IN_A_BLOCK = known_bug(
    6, "a 1.0 name beyond the declared count inside an (or (and ...)) block "
       "(X_3 on an input, Y_3 on an output, one of each declared) crashes "
       "with IndexError in _apply_block_atom instead of being refused with "
       "ValueError", raises=IndexError)
_SPEC19_NEGATIVE_INDEX = known_bug(
    6, "a 1.0 name with a negative number is not refused: X_-1 is read as "
       "the last declared input (with two inputs, (<= X_-1 0.5) bounds X_1) "
       "and Y_-1 inside an (or (and ...)) block as the last declared output",
    raises=DID_NOT_RAISE)
_NOT_OR_UNKNOWN_OPERATOR_ON_AN_INPUT = known_bug(
    6, "a 1.0 input assert with an operator VNN-LIB does not define, "
       "(not (<= X_0 0.5)) or (=> X_0 0.5), is not refused: its last "
       "comparison is read as an upper bound, so (not (<= X_0 0.5)) is stored "
       "as x <= 0.5, the complement of its meaning in SMT-LIB (false UNSAT "
       "for a file meant that way); on an output both are refused",
    raises=DID_NOT_RAISE)
_V1_OUTPUT_OR_CRASH = known_bug(
    6, "a 1.0 output assert followed by an output (or ...) assert, or a "
       "one-disjunct (or ...) followed by another output assert, crashes with "
       "AttributeError instead of being read as their conjunction or refused "
       "with ValueError", raises=AttributeError)


def _read_or_refused_cases():
    out = []
    for op in ("<=", "<", ">=", ">"):
        # 1.0: one marked case per failing code path (the operand read at
        # load_vnnlib.py:516 for an input, :382 for an output), plus both
        # strict input forms, which also decide whether the first input line
        # is recognised as an input bound at all. The 2.0 reader has a branch
        # per operator and keeps all four on both sides.
        if op in ("<", ">"):
            out.append(pytest.param(
                _Case("1.0", _B1 + f"(assert ({op} 0.25 X_0))\n" + _O1,
                      may_refuse=True),
                id=f"v1-number-first-{op}-input", marks=_SPEC8))
        if op == "<=":
            out.append(pytest.param(
                _Case("1.0", _B1 + f"(assert ({op} 0.875 Y_0))\n",
                      may_refuse=True),
                id=f"v1-number-first-{op}-output", marks=_SPEC8))
        out.append(pytest.param(
            _Case("2.0", _B2 + f"(assert ({op} 0.25 {_X2}))\n" + _O2),
            id=f"v2-number-first-{op}-input"))
        out.append(pytest.param(
            _Case("2.0", _B2 + f"(assert ({op} 0.875 {_Y2}))\n"),
            id=f"v2-number-first-{op}-output"))
    out += [
        pytest.param(_Case("1.0", "(assert (>= X_0 (- 0.5)))\n"
                                  "(assert (<= X_0 1.0))\n" + _O1,
                           extra_x=(-0.75, -0.5, -0.25), may_refuse=True),
                     id="v1-negative-number-in-parentheses",
                     marks=_SPEC8_OPERAND_FORM),
        pytest.param(_Case("1.0", "(assert (or (and (>= X_0 (- 0.5)) "
                                  "(<= X_0 1.0))))\n" + _O1,
                           extra_x=(-0.75, -0.5, -0.25), may_refuse=True),
                     id="v1-negative-number-in-parentheses-in-an-or-block",
                     marks=_SPEC8_IN_A_BLOCK),
        pytest.param(_Case("1.0", _B1 + "(assert (<= (* 2.0 Y_0) 1.0))\n",
                           ys=(0.25, 0.5, 0.625), may_refuse=True),
                     id="v1-scaled-output", marks=_SPEC8_OPERAND_FORM),
        pytest.param(_Case("2.0", _B2 + f"(assert (<= (* 2.0 {_Y2}) 1.0))\n",
                           ys=(0.25, 0.5, 0.625)),
                     id="v2-scaled-output"),
        pytest.param(_Case("1.0", _B1 + "(assert (>= Y_0 0.5))\n"
                                  "(assert (or (<= Y_0 0.625) "
                                  "(>= Y_0 0.9375)))\n",
                           ys=(0.25, 0.5, 0.625, 0.75, 0.9375, 1.0),
                           may_refuse=True),
                     id="v1-output-row-then-or", marks=_V1_OUTPUT_OR_CRASH),
        pytest.param(_Case("1.0", _B1 + "(assert (or (and (>= Y_0 0.5))))\n"
                                  "(assert (<= Y_0 0.9375))\n",
                           ys=(0.25, 0.5, 0.75, 0.9375, 1.0), may_refuse=True),
                     id="v1-one-disjunct-or-then-row",
                     marks=_V1_OUTPUT_OR_CRASH),
        pytest.param(_Case("2.0", _B2 + f"(assert (>= {_Y2} 0.5))\n"
                                  f"(assert (or (<= {_Y2} 0.625) "
                                  f"(>= {_Y2} 0.9375)))\n",
                           ys=(0.25, 0.5, 0.625, 0.75, 0.9375, 1.0)),
                     id="v2-output-row-then-or"),
        pytest.param(_Case("1.0", _B1 + "(assert (and (>= Y_0 0.5) "
                                  "(<= Y_0 0.875)))\n",
                           ys=(0.25, 0.5, 0.75, 0.875, 1.0), may_refuse=True),
                     id="v1-output-and"),
        pytest.param(_Case("1.0", "(assert (or (and (>= X_0 0.0) "
                                  "(or (<= X_0 0.25) (>= X_0 0.75)) "
                                  "(<= X_0 1.0))))\n" + _O1, may_refuse=True),
                     id="v1-input-or-inside-and"),
        pytest.param(_Case("1.0", _B1 + "(assert (or (and (or (<= Y_0 0.25) "
                                  "(>= Y_0 0.875)) (>= Y_0 0.0))))\n",
                           ys=(0.125, 0.25, 0.5, 0.875, 1.0), may_refuse=True),
                     id="v1-output-or-inside-and"),
        pytest.param(_Case("1.0", _B1 + "(assert (or (< Y_0 0.25) "
                                  "(> Y_0 0.875)))\n",
                           ys=(0.125, 0.25, 0.5, 0.875, 1.0), may_refuse=True),
                     id="v1-output-strict-in-or"),
        pytest.param(_Case("1.0", _B1 + _O1,
                           header="(declare-const Y_0 Real)\n"
                                  "(declare-const X_0 Real)\n",
                           may_refuse=True),
                     id="v1-outputs-declared-first"),
        pytest.param(_Case("2.0", f"(assert (or (and (>= {_X2} 0.0) "
                                  f"(or (<= {_X2} 0.25) (>= {_X2} 0.75)) "
                                  f"(<= {_X2} 1.0))))\n" + _O2,
                           may_refuse=True),
                     id="v2-input-or-inside-and"),
    ]
    return out


@pytest.mark.parametrize("case", _read_or_refused_cases())
def test_comparison_form_is_read_or_refused(tmp_path, case):
    """A form the parser may not support (a number written first, a negative
    number written ``(- 0.5)``, a product, an output row followed by an
    ``or``) must be read as the text says or refused with ``ValueError``
    naming the file. Any other exception is a crash: the competition runner
    turns it into ``unknown`` without saying which file and construct it
    could not read, and other callers get an ``IndexError`` from inside the
    parser. The 2.0 cells read the same forms correctly today, and a refusal
    of them fails (DESIGN section 5 lists a number written first, 2.0
    arithmetic and an output row followed by an ``or`` as read)."""
    verdicts = _verdicts(case)
    result = _load_or_refuse(tmp_path, case)
    if result is None:
        return
    left_out = [(x, y) for x, y, accepted in verdicts
                if accepted and not _in_stored_region(result, x, y)]
    wrongly = [(x, y) for x, y, accepted in verdicts
               if not accepted and _parse_accepts(result, x, y)]
    assert left_out == [], f"counterexamples left out: {left_out[:3]}"
    assert wrongly == [], f"rejected points accepted: {wrongly[:3]}"


# --------------------------------------------------------------------------- #
# Refused: a name outside the declarations                                     #
# --------------------------------------------------------------------------- #
_V2_INDEX_UNCHECKED = ("a 2.0 index outside the declared shape is not "
                       "checked: X[0,2] of a [2,2] tensor is read as X[1,0] "
                       "without a message, and X[0,5] of a [1,1] tensor "
                       "crashes with IndexError instead of being refused "
                       "with ValueError")
_V2_SQUARE = ("(vnnlib-version <2.0>)\n(declare-network N\n"
              "  (declare-input X float32 [2,2])\n"
              "  (declare-output Y float32 [2,2])\n)\n")
_V2_SQUARE_BOX = "".join(f"(assert (>= X[{i},{j}] 0.0))\n"
                         f"(assert (<= X[{i},{j}] 1.0))\n"
                         for i in range(2) for j in range(2))


@pytest.mark.parametrize("text", [
    pytest.param(_v1_header() + _B1 + "(assert (<= X_3 0.5))\n" + _O1,
                 id="v1-input-beyond-the-declared", marks=_SPEC19_UNDECLARED),
    pytest.param(_v1_header() + _B1 + "(assert (>= Y_3 0.875))\n",
                 id="v1-output-beyond-the-declared", marks=_SPEC19_UNDECLARED),
    pytest.param(_v1_header() + "(assert (or (and (>= X_0 0.0) (<= X_0 1.0) "
                 "(<= X_3 0.5))))\n" + _O1,
                 id="v1-input-beyond-the-declared-in-an-or-block",
                 marks=_SPEC19_IN_A_BLOCK),
    pytest.param(_v1_header() + _B1 + "(assert (or (and (<= X_0 0.5) "
                 "(>= Y_3 0.875))))\n",
                 id="v1-output-beyond-the-declared-in-a-paired-or",
                 marks=_SPEC19_IN_A_BLOCK),
    pytest.param(_v1_header(2) + _B1 + "(assert (>= X_1 0.0))\n"
                 "(assert (<= X_1 1.0))\n(assert (<= X_-1 0.5))\n" + _O1,
                 id="v1-negative-input-index", marks=_SPEC19_NEGATIVE_INDEX),
    pytest.param(_v1_header(1, 2) + _B1 + "(assert (or (and (<= X_0 0.5) "
                 "(>= Y_-1 0.875))))\n",
                 id="v1-negative-output-index-in-a-paired-or",
                 marks=_SPEC19_NEGATIVE_INDEX),
    pytest.param(_v1_header() + _B1 + "(assert (not (<= X_0 0.5)))\n" + _O1,
                 id="v1-not-on-an-input",
                 marks=_NOT_OR_UNKNOWN_OPERATOR_ON_AN_INPUT),
    pytest.param(_v1_header() + _B1 + "(assert (=> X_0 0.5))\n" + _O1,
                 id="v1-unknown-operator-on-an-input",
                 marks=_NOT_OR_UNKNOWN_OPERATOR_ON_AN_INPUT),
    pytest.param(_v1_header() + _B1 + "(assert (not (<= Y_0 0.5)))\n",
                 id="v1-not-on-an-output"),
    pytest.param(_v1_header() + _B1 + "(assert (=> Y_0 0.5))\n",
                 id="v1-unknown-operator-on-an-output"),
    pytest.param(_v2_header() + _B2 + "(assert (<= X[0,5] 0.5))\n" + _O2,
                 id="v2-input-index-beyond-the-tensor",
                 marks=known_bug(6, _V2_INDEX_UNCHECKED, raises=IndexError)),
    pytest.param(_V2_SQUARE + _V2_SQUARE_BOX + "(assert (<= X[0,2] 0.5))\n"
                 "(assert (>= Y[0,0] 0.875))\n",
                 id="v2-input-index-beyond-its-axis",
                 marks=known_bug(6, _V2_INDEX_UNCHECKED,
                                 raises=DID_NOT_RAISE)),
    pytest.param(_V2_SQUARE + _V2_SQUARE_BOX + "(assert (>= Y[0,2] 0.875))\n",
                 id="v2-output-index-beyond-its-axis",
                 marks=known_bug(6, _V2_INDEX_UNCHECKED,
                                 raises=DID_NOT_RAISE)),
    pytest.param(_v2_header() + _B2 + "(assert (<= Z[0,0] 0.5))\n" + _O2,
                 id="v2-undeclared-tensor"),
    pytest.param(_v2_header() + _B2 + "(assert (<= X[0,0,0] 0.5))\n" + _O2,
                 id="v2-index-of-the-wrong-rank"),
    pytest.param(_v2_header(2) + _B2 + "(assert (>= X[0,1] 0.0))\n"
                 "(assert (<= X[0,1] 1.0))\n(assert (<= X 0.5))\n" + _O2,
                 id="v2-bare-name-of-a-tensor"),
    pytest.param(_v2_header() + _B2 + f"(assert (=> {_X2} 0.5))\n" + _O2,
                 id="v2-unknown-operator"),
    pytest.param(_v2_header() + _B2 + f"(assert (not (<= {_X2} 0.5)))\n"
                 + _O2, id="v2-not"),
])
def test_name_or_operator_outside_the_language_is_refused(tmp_path, text):
    """A file that names a variable its declarations do not have, or uses an
    operator VNN-LIB does not define, has no meaning (``_Text`` gives it
    none); the parser must refuse it with ``ValueError``. Reading
    ``X[0,2]`` of a [2,2] tensor as the flat index 2 constrains ``X[1,0]``,
    which the file never mentions. The reference check raises its own type,
    so that a marker absorbing pytest's ``Failed`` (``DID_NOT_RAISE``) cannot
    absorb it."""
    try:
        _Text(text).accepts(_declared_values(text))
    except _Undefined:
        pass
    else:
        raise _SetupError("the reference gives the text a meaning")
    with pytest.raises(ValueError):
        _load(tmp_path, text)


def _declared_values(text, value=0.5):
    """``value`` for every variable the declarations of ``text`` name."""
    names = re.findall(r"\(declare-const\s+(\S+)\s+Real\)", text)
    for name, shape in re.findall(
            r"\(declare-(?:input|output)\s+(\S+)\s+\S+\s+\[([^\]]*)\]",
            text):
        dims = [int(d) for d in shape.replace(",", " ").split()]
        names += [f"{name}[{','.join(map(str, idx))}]"
                  for idx in itertools.product(*map(range, dims))]
    return {name: value for name in names}


def test_declared_values_names_every_declared_variable():
    assert set(_declared_values(_v1_header(2, 1))) == {"X_0", "X_1", "Y_0"}
    assert set(_declared_values(_V2_SQUARE)) == {
        f"{t}[{i},{j}]" for t in "XY" for i in range(2) for j in range(2)}


_IN_SHAPE, _OUT_SHAPE = (1, 2, 3), (2, 3)


def test_v2_tensor_index_is_stored_in_row_major_order(tmp_path):
    """A flat set that stands for a tensor is in the tensor's own row-major
    order (the order of ``numpy.ravel``, which is also the order of a VNN-COMP
    witness). Each input entry of a (1, 2, 3) tensor gets its own interval
    and one output row names ``Y[1,0]`` of a (2, 3) tensor; the reference
    index is ``numpy.ravel_multi_index``, which knows nothing of the parser.
    Catches a flattening in column-major order or with the axes swapped,
    which would bound each input entry by the interval of another and put
    the output row on the wrong output. The stated intervals are 1/16 wide
    and 1/16 apart, so a stored interval is identified as the right one when
    it holds the stated one and lies within 1/32 of it: a bound rounded
    outward passes, the interval of another entry does not."""
    header = ("(vnnlib-version <2.0>)\n(declare-network N\n"
              "  (declare-input X float32 [1,2,3])\n"
              "  (declare-output Y float32 [2,3])\n)\n")
    stated = {}
    body = ""
    for idx in itertools.product(*map(range, _IN_SHAPE)):
        k = len(stated)
        lo, hi = k / 8, k / 8 + 1 / 16
        stated[idx] = (lo, hi)
        name = "X[" + ",".join(map(str, idx)) + "]"
        body += f"(assert (>= {name} {lo}))\n(assert (<= {name} {hi}))\n"
    body += "(assert (>= Y[1,0] 0.875))\n"
    result = _load(tmp_path, header + body)
    (pair,) = result["pairs"]
    lb = np.asarray(pair["lb"], dtype=np.float64).ravel()
    ub = np.asarray(pair["ub"], dtype=np.float64).ravel()
    assert lb.size == ub.size == int(np.prod(_IN_SHAPE))
    window = 1 / 32
    for idx, (lo, hi) in stated.items():
        flat = int(np.ravel_multi_index(idx, _IN_SHAPE))
        assert (lo - window <= lb[flat] <= lo
                and hi <= ub[flat] <= hi + window), (
            f"X{list(idx)} is stated in [{lo}, {hi}] and stored at flat "
            f"index {flat} as [{lb[flat]}, {ub[flat]}]")
    (group,) = pair["prop"]
    G = np.asarray(group["Hg"].G, dtype=np.float64)
    g = np.asarray(group["Hg"].g, dtype=np.float64).ravel()
    flat = int(np.ravel_multi_index((1, 0), _OUT_SHAPE))
    assert G.shape == (1, int(np.prod(_OUT_SHAPE)))
    assert np.flatnonzero(G[0]).tolist() == [flat] and G[0, flat] < 0, (
        f"the row on Y[1,0] is stored as {G[0].tolist()}")
    threshold = g[0] / G[0, flat]           # the row reads y >= threshold
    assert 0.875 - window <= threshold <= 0.875


@pytest.mark.parametrize("text", [
    pytest.param(_v1_header() + "(assert (>= X_0 0.75))\n"
                 "(assert (<= X_0 0.25))\n" + _O1, id="v1"),
    pytest.param(_v2_header() + f"(assert (>= {_X2} 0.75))\n"
                 f"(assert (<= {_X2} 0.25))\n" + _O2, id="v2"),
    pytest.param(_v2_header() + _B2 + f"(assert (== {_X2} 2.0))\n" + _O2,
                 id="v2-equality-outside-the-box"),
    pytest.param(_v1_header(2) + _B1 + "(assert (>= X_1 0.75))\n"
                 "(assert (<= X_1 0.25))\n" + _O1, id="v1-second-input-empty"),
    pytest.param(_v2_header(2) + _B2 + "(assert (>= X[0,1] 0.75))\n"
                 "(assert (<= X[0,1] 0.25))\n" + _O2, id="v2-second-input-empty"),
    pytest.param(_v1_header() + "(assert (or (and (>= X_0 0.0) (<= X_0 0.25)) "
                 "(and (>= X_0 0.75) (<= X_0 0.5))))\n" + _O1,
                 id="v1-second-region-empty"),
])
def test_empty_input_region_is_refused(tmp_path, text):
    """A lower bound above the upper bound states an empty input region
    (DESIGN section 5: ``SpecError``). Handed on, it would verify as UNSAT or
    fail far downstream. The emptiness of one entry decides it, whichever
    entry and whichever region it is (a check of the first entry or region
    only, or of every entry at once, passes the one-input cases)."""
    with pytest.raises(ValueError):
        _load(tmp_path, text)


@pytest.mark.parametrize("text", [
    pytest.param(_v1_header(2) + _B1 + "(assert (>= X_1 0.0))\n" + _O1,
                 id="v1-second-input-without-an-upper-bound"),
    pytest.param(_v1_header() + "(assert (<= X_0 1.0))\n" + _O1,
                 id="v1-input-without-a-lower-bound"),
    pytest.param(_v1_header() + "(assert (or (and (>= X_0 0.0) (<= X_0 0.25)) "
                 "(and (>= X_0 0.75))))\n" + _O1,
                 id="v1-second-region-without-an-upper-bound"),
    pytest.param(_v2_header() + f"(assert (>= {_X2} 0.0))\n" + _O2,
                 id="v2-input-without-an-upper-bound"),
])
def test_unbounded_input_region_is_refused(tmp_path, text):
    """An input the text leaves unbounded on one side (DESIGN section 5:
    ``SpecError``). Handed on with an infinite bound, reach and sampling
    either fail far downstream or work on a box the file does not state.
    Catches a finiteness check that combines the two sides with ``or``, that
    looks at one entry or one region only, or a default bound put in place
    of the missing one."""
    with pytest.raises(ValueError):
        _load(tmp_path, text)


@pytest.mark.parametrize("text", [
    pytest.param(_v1_header() + _B1 + _O1 + "(assert (<= Y_0 X_0))\n",
                 id="v1-output-compared-with-an-input"),
    pytest.param(_v1_header() + _B1 + "(assert (<= X_0 Y_0))\n" + _O1,
                 id="v1-input-compared-with-an-output"),
    pytest.param(_v1_header() + _B1 + "(assert (or (and (<= X_0 0.5) "
                 "(<= Y_0 X_0))))\n",
                 id="v1-output-compared-with-an-input-in-a-paired-or"),
])
def test_comparison_of_an_input_with_an_output_is_refused(tmp_path, text):
    """DESIGN section 5 refuses a comparison that mixes an input and an
    output variable (``SpecError``): it is neither an input bound nor an
    output row, so any reading of it states a property the file does not.
    The 1.0 reader refuses these today only because ``float()`` of the other
    name fails; a reader that took the other name for an output index, or
    for the number 0, would answer instead. (The 2.0 reader refuses the same
    form at two places, so no single wrong change shows a 2.0 case failing;
    its cells belong to the 2.0 reader's own file.)"""
    with pytest.raises(ValueError):
        _load(tmp_path, text)


_COMMENT_AFTER_AN_OUTPUT_ASSERT = known_bug(
    6, "a 1.0 output assert followed by a trailing comment, "
       "(assert (>= Y_0 0.875)) ; unsafe, is refused with ValueError: the "
       "comment is not removed and the output reader meets it as an unknown "
       "statement, so a legal file gets no verdict", raises=ValueError)
_COMMENT_WITH_A_PARENTHESIS = known_bug(
    6, "a 1.0 input bound followed by a trailing comment that holds a "
       "parenthesis, (assert (<= X_0 1.0)) ; (inclusive), crashes with "
       "IndexError: the bound is read from the text after the last '(' of "
       "the line, which is the comment", raises=IndexError)


@pytest.mark.parametrize("case", [
    pytest.param(_Case("1.0", "(assert (>= X_0 0.0)) ; lower bound\n"
                              "(assert (<= X_0 1.0))\n" + _O1),
                 id="after-an-input-bound"),
    pytest.param(_Case("1.0", "(assert (or (and (>= X_0 0.0) (<= X_0 0.25)) "
                              "(and (>= X_0 0.75) (<= X_0 1.0)))) ; two "
                              "regions\n" + _O1),
                 id="after-input-regions"),
    pytest.param(_Case("1.0", "(assert (or (and (>= X_0 0.0) (<= X_0 0.5) "
                              "(>= Y_0 0.875)))) ; one pair\n"),
                 id="after-a-combined-assert"),
    pytest.param(_Case("1.0", _B1 + "(assert (>= Y_0 0.875)) ; unsafe\n"),
                 id="after-an-output-assert",
                 marks=_COMMENT_AFTER_AN_OUTPUT_ASSERT),
    pytest.param(_Case("1.0", "(assert (>= X_0 0.0))\n(assert (<= X_0 1.0)) "
                              "; (inclusive)\n" + _O1),
                 id="with-a-parenthesis-after-an-input-bound",
                 marks=_COMMENT_WITH_A_PARENTHESIS),
])
def test_v1_trailing_comment_is_not_part_of_the_statement(tmp_path, case):
    """A comment runs from ``;`` to the end of the line and has no meaning
    (``_Text`` drops it). A legal file with a trailing comment must be read
    as the same file without it, in both directions: no refusal, no crash,
    no other reading. The comments here hold no letters 'or' (that route is
    SPEC-2, marked above). Catches a reader that classifies or splits a line
    by text the comment holds."""
    verdicts = _verdicts(case)
    if all(a for _, _, a in verdicts) or not any(a for _, _, a in verdicts):
        raise _SetupError("the case needs points on both sides of the text")
    result = _load(tmp_path, case.text)
    left_out = [(x, y) for x, y, accepted in verdicts
                if accepted and not _in_stored_region(result, x, y)]
    wrongly = [(x, y) for x, y, accepted in verdicts
               if not accepted and _parse_accepts(result, x, y)]
    assert left_out == [], f"counterexamples left out: {left_out[:3]}"
    assert wrongly == [], f"rejected points accepted: {wrongly[:3]}"


@pytest.mark.parametrize("text", [
    pytest.param(_v1_header() + _B1 + "(assert (or (<= Y_0 0.25) "
                 "(>= Y_0 0.875)\n", id="v1",
                 marks=known_bug(
                     6, "a 1.0 file that ends before its last assert is "
                        "closed is read without a message (the statement is "
                        "merged up to the end of the file and read as far as "
                        "it goes), where a refusal with ValueError is due: a "
                        "file cut short is verified as the part that is left",
                     raises=DID_NOT_RAISE)),
    pytest.param(_v2_header() + _B2 + f"(assert (or (<= {_Y2} 0.25) "
                 f"(>= {_Y2} 0.875)\n", id="v2"),
])
def test_file_whose_last_assert_is_not_closed_is_refused(tmp_path, text):
    """A file whose parentheses do not balance (a download cut short) has no
    meaning; the parser must refuse it with ``ValueError`` instead of reading
    the part that is there."""
    with pytest.raises(ValueError):
        _load(tmp_path, text)


# --------------------------------------------------------------------------- #
# End to end: the verdict on a misread file                                    #
# --------------------------------------------------------------------------- #
def _line(weight, bias):
    """The network y = weight * x + bias, in float64."""
    model = torch.nn.Sequential(torch.nn.Linear(1, 1)).double()
    with torch.no_grad():
        model[0].weight.fill_(weight)
        model[0].bias.fill_(bias)
    return model.eval()


_E2E = {
    "v2-disequality-number-first-y=x": (
        _CASES["v2-disequality-number-first"], (1.0, 0.0), _SPEC9),
    "v2-disequality-y=1-x": (_CASES["v2-disequality"], (-1.0, 1.0), _SPEC9),
    "v2-disequality-y=x": (_CASES["v2-disequality"], (1.0, 0.0), None),
    "v1-upper-bound-after-input-or-y=1-x": (
        _CASES["v1-upper-bound-after-input-or"], (-1.0, 1.0), _SPEC2),
    "v1-input-or-y=1-x": (
        _Case("1.0", "(assert (or (and (>= X_0 0.0) (<= X_0 1.0))))\n" + _O1),
        (-1.0, 1.0), None),
    # x in [0.375, 0.4375] gives y in [0.875, 0.9375]; the parse keeps
    # x >= 0.875, where y >= 1.375 lies above the row y <= 0.9375.
    "v1-mixed-and-y=x+0.5": (
        _CASES["v1-mixed-and-with-an-output-row"], (1.0, 0.5),
        _OUTPUT_ATOM_READ_AS_INPUT),
    # Every output is unsafe; the parse keeps y >= 0, and y = x - 2 < 0.
    "v1-output-y0-le-y0-y=x-2": (
        _CASES["v1-output-y0-le-y0"], (1.0, -2.0), _SAME_OUTPUT_TWICE),
}


@pytest.mark.parametrize("method", ["exact", "approx"])
@pytest.mark.parametrize("case, line", [
    pytest.param(case, line, id=name,
                 marks=[mark] if mark is not None else [])
    for name, (case, line, mark) in _E2E.items()])
def test_verdict_is_not_unsat_while_the_text_has_a_counterexample(
        tmp_path, case, line, method):
    """The harm of a misread file, end to end: the network is the line
    ``y = w x + b`` (torch, float64), each stored region is reached with
    ``NeuralNetwork.reach`` and checked with ``verify_specification``
    against its own unsafe region. The reference is the text and torch: the
    inputs k/1024 in [-1/8, 9/8] whose output the text calls a
    counterexample. While one exists, the verdict must not be UNSAT (a
    refusal by the parser keeps the promise only for a form that may be
    refused, ``_Case.may_refuse``)."""
    model = _line(*line)
    xs = [k / 1024 for k in range(-128, 1153)]
    with torch.no_grad():
        ys = model(torch.tensor(xs, dtype=torch.float64).reshape(-1, 1))
    ys = ys.reshape(-1).tolist()
    text = _Text(case.text)
    witnesses = [(x, y) for x, y in zip(xs, ys)
                 if text.accepts(case.values((x,), (y,)))]
    if not witnesses:
        raise _SetupError("the case must have a counterexample on the network")
    result = _load_or_refuse(tmp_path, case)
    if result is None:
        return
    verdicts = []
    for pair in result["pairs"]:
        lb = np.asarray(pair["lb"], dtype=np.float64).reshape(-1, 1)
        ub = np.asarray(pair["ub"], dtype=np.float64).reshape(-1, 1)
        reach = NeuralNetwork(model).reach(Star.from_bounds(lb, ub),
                                           method=method)
        verdicts.append(verify_specification(reach, pair["prop"]).verdict)
    assert not all(v == "UNSAT" for v in verdicts), (
        f"UNSAT in every region {verdicts}, and x = {witnesses[0][0]} gives "
        f"y = {witnesses[0][1]}, a counterexample of the text "
        f"({len(witnesses)} of {len(xs)} grid inputs are)")
