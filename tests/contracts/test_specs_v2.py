"""VNN-LIB 2.0 linear reader contract (``n2v/utils/vnnlib2.py``, the single-network linear lowering).

Component: ``load_vnnlib_v2`` and the helpers of its linear fragment (the text layer, the declarations and the variable resolver, affine expressions and comparisons, the output rows and their disjunctive normal form, the input box, the disjunctive input regions, the combined input and output asserts, and the kept source assertions), reached through the public loader ``load_vnnlib`` and, where it matters, through ``load_vnnlib_v2`` itself. The relational and nonlinear loaders belong to other tests; only the dispatch to the nonlinear loader is checked here.

The promise: a parsed file means what its text says. A point (x, y) is a counterexample of the text exactly when every ``(assert F)`` holds there. The parse calls (x, y) a counterexample when x lies in the box of some stored pair and y in every unsafe group of that pair (``G y <= g``, a union of polytopes per group) and, when the parse keeps the source assertions for the SAT check, those accept (x, y) too. Three things are checked at every point, chosen on, inside and outside every bound of the text:

* no counterexample of the text is missed (a region read too small or an unsafe region read too small turns a real violation into a false UNSAT; a kept assertion that rejects a true counterexample throws a valid witness away);
* no point the text rejects is accepted (a region read too large or an unsafe region read too large reports an invalid SAT);
* the stored rows alone accept nothing beyond the text read with every strict comparison closed (the docstring of ``_compare_to_rows``: a strict comparison becomes the closed row). Rows wider than that read a stronger unsafe region than the file states, which the kept assertions hide from the SAT check but which reach and the verdict see as UNKNOWN.

The reference is ``_Text`` below: a reader of the 2.0 text that shares no code with n2v. It reads numbers as exact decimals and evaluates in fractions, finds a variable's position with ``numpy.ravel_multi_index`` over the declared shape and offsets computed from the declarations, and gives no meaning to a name or an index the declarations do not have. All numbers of the cases are dyadic, so the float64 numbers the parser stores and the float64 evaluation of the kept assertions are exact and every comparison below is decided without slack.

Forms outside the fragment must be refused with ``ValueError`` (``VNNLibParseError`` is one); a crash with another exception type is a defect of its own. ``tests/contracts/test_specs.py`` holds the cases of the 2.0 reader that this file does not repeat: one variable of shape [1, 1] with strict comparisons, repeated and looser-last top-level bounds, ``!=`` on an input (registry SPEC-9), decimal coefficients rounded to nearest (SPEC-10 and SPEC-16), an index beyond its axis (SPEC-21) and the number written first.
"""

import itertools
import re
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest

from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.vnnlib2 import evaluate_nonlinear, load_vnnlib_v2
from tests.known_bugs import DID_NOT_RAISE, known_bug


# =========================================================================== #
# The reference: what a 2.0 text says                                          #
# =========================================================================== #
class _Undefined(Exception):
    """The text names something the 2.0 language gives no meaning (an undeclared name, an index outside the declared shape, an operator it does not have)."""


_ORDER = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
          ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
          "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
_CLOSED = {"<": "<=", ">": ">="}


def _top_level(text):
    """Every top-level s-expression of ``text``. A ``;`` starts a comment that runs to the end of its line."""
    body = " ".join(line.split(";", 1)[0] for line in text.splitlines())
    tokens = body.replace("(", " ( ").replace(")", " ) ").split()
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
    """The meaning of a single-network VNN-LIB 2.0 file, read from its text alone.

    The declarations give each tensor a shape; the input tensors, then the output tensors, are laid end to end in the order they are declared, each in row-major order (``numpy.ravel_multi_index``), which is the joint vector a point is given in. ``NAME[i,j]`` is the entry at that multi-index and must have one index per axis inside the axis; a bare ``NAME`` is a rank-0 tensor. A point is a counterexample when every assert holds, with ``and``, ``or``, the comparisons ``<= < >= > == !=`` in prefix form or infix ``(a op b)`` and the arithmetic ``+ - * /`` over exact decimals. ``closed=True`` reads every strict comparison as the non-strict one."""

    def __init__(self, text):
        self.inputs, self.outputs, self.asserts = [], [], []
        for item in _top_level(text):
            if not isinstance(item, list) or not item:
                continue
            if item[0] == "declare-network":
                for decl in item[2:]:
                    if isinstance(decl, list) and decl[0] in ("declare-input", "declare-output"):
                        dims = "".join(decl[3:]).strip("[]")
                        shape = tuple(int(d) for d in dims.split(",") if d)
                        (self.inputs if decl[0] == "declare-input" else self.outputs).append((decl[1], shape))
            elif item[0] == "assert":
                self.asserts.append(item[1])
        self._where = {}
        for kind, decls in (("in", self.inputs), ("out", self.outputs)):
            offset = 0
            for name, shape in decls:
                self._where[name] = (kind, offset, shape)
                offset += int(np.prod(shape, dtype=np.int64))
        self.n_in = sum(int(np.prod(s, dtype=np.int64)) for _, s in self.inputs)
        self.n_out = sum(int(np.prod(s, dtype=np.int64)) for _, s in self.outputs)

    def accepts(self, x, y, closed=False):
        x = [Fraction(v) for v in x]
        y = [Fraction(v) for v in y]
        assert len(x) == self.n_in and len(y) == self.n_out, "a point of the wrong size"
        return all(self._holds(a, x, y, closed) for a in self.asserts)

    def position(self, atom):
        """``('in' | 'out', joint index)`` of a variable atom, or ``_Undefined``."""
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[([0-9,]*)\]", atom)
        name = m.group(1) if m else atom
        if name not in self._where:
            raise _Undefined(f"unknown name {atom!r}")
        kind, offset, shape = self._where[name]
        if m is None:
            if shape != ():
                raise _Undefined(f"bare name of a tensor of shape {shape}")
            return kind, offset
        index = tuple(int(i) for i in m.group(2).split(",") if i)
        if len(index) != len(shape) or any(not 0 <= i < d for i, d in zip(index, shape)):
            raise _Undefined(f"index {index} outside the shape {shape}")
        flat = int(np.ravel_multi_index(index, shape)) if shape else 0
        return kind, offset + flat

    def _holds(self, node, x, y, closed):
        if not isinstance(node, list) or not node:
            raise _Undefined(f"not a formula: {node!r}")
        if node[0] == "and":
            return all(self._holds(c, x, y, closed) for c in node[1:])
        if node[0] == "or":
            return any(self._holds(c, x, y, closed) for c in node[1:])
        if len(node) == 3 and isinstance(node[0], str) and node[0] in _ORDER:
            op, a, b = node
        elif len(node) == 3 and isinstance(node[1], str) and node[1] in _ORDER:
            a, op, b = node
        else:
            raise _Undefined(f"no meaning for {node!r}")
        if closed:
            op = _CLOSED.get(op, op)
        return _ORDER[op](self._value(a, x, y), self._value(b, x, y))

    def _value(self, node, x, y):
        if isinstance(node, str):
            try:
                return Fraction(node)
            except ValueError:
                kind, index = self.position(node)
                return x[index] if kind == "in" else y[index]
        head, args = node[0], [self._value(c, x, y) for c in node[1:]]
        if head == "+":
            return sum(args, Fraction(0))
        if head == "-" and len(args) == 1:
            return -args[0]
        if head == "-" and len(args) > 1:
            return args[0] - sum(args[1:], Fraction(0))
        if head == "*":
            out = Fraction(1)
            for a in args:
                out *= a
            return out
        if head == "/" and len(args) == 2 and args[1] != 0:
            return args[0] / args[1]
        raise _Undefined(f"no meaning for {node!r}")


_SELF_TEST_HEADER = ("(vnnlib-version <2.0>)\n(declare-network N\n"
                     "  (declare-input A float32 [2, 3])\n  (declare-input B float32 [1,2])\n"
                     "  (declare-output Z float32 [2,2])\n  (declare-output S float32 [])\n)\n")


@pytest.mark.parametrize("body, x_at, y_at, accepted, closed", [
    # A[1,0] is joint input 3 (row-major in a [2,3] tensor), B[0,1] is joint input 7.
    pytest.param("(assert (<= A[1,0] 0.5))", {3: 0.5}, {}, True, False, id="entry-of-a-rank-2-tensor-on-its-bound"),
    pytest.param("(assert (<= A[1,0] 0.5))", {3: 0.625}, {}, False, False, id="entry-of-a-rank-2-tensor-above-its-bound"),
    pytest.param("(assert (<= A[1,0] 0.5))", {1: 0.625}, {}, True, False, id="another-entry-above-the-bound"),
    pytest.param("(assert (>= B[0,1] 0.25))", {7: 0.25}, {}, True, False, id="entry-of-a-second-tensor-on-its-bound"),
    pytest.param("(assert (>= B[0,1] 0.25))", {6: 0.25, 7: 0.0}, {}, False, False, id="entry-before-a-second-tensor"),
    # Z[1,0] is joint output 2, the bare S joint output 4.
    pytest.param("(assert (<= Z[1,0] S))", {}, {2: 0.5, 4: 0.25}, False, False, id="rank-0-tensor-named-bare"),
    pytest.param("(assert (<= Z[1,0] S))", {}, {1: 0.5, 4: 0.25}, True, False, id="other-output-than-the-named-one"),
    pytest.param("(assert (< S 0.5))", {}, {4: 0.5}, False, False, id="strict-on-its-bound"),
    pytest.param("(assert (< S 0.5))", {}, {4: 0.5}, True, True, id="strict-read-closed-on-its-bound"),
    pytest.param("(assert (S > 0.5))", {}, {4: 0.75}, True, False, id="infix-strict"),
    pytest.param("(assert (== Z[0,1] S))", {}, {1: 0.25, 4: 0.25}, True, False, id="equality-of-two-outputs"),
    pytest.param("(assert (!= Z[0,1] S))", {}, {1: 0.25, 4: 0.25}, False, False, id="disequality-of-two-equal-outputs"),
    pytest.param("(assert (or (<= S -1.0) (and (>= S 1.0) (<= S 2.0))))", {}, {4: 1.5}, True, False, id="or-of-and-inside"),
    pytest.param("(assert (or (<= S -1.0) (and (>= S 1.0) (<= S 2.0))))", {}, {4: 0.5}, False, False, id="or-of-and-outside"),
    # Arithmetic over exact decimals: 0.1 * 3 is 0.3.
    pytest.param("(assert (<= (* 0.1 S) 0.3))", {}, {4: 3.0}, True, False, id="decimal-product-is-exact"),
    pytest.param("(assert (<= (- S) -0.5))", {}, {4: 0.25}, False, False, id="negation"),
    pytest.param("(assert (<= (- S Z[0,0] Z[0,1]) 0.0))", {}, {0: 0.5, 1: 0.25, 4: 0.75}, True, False, id="difference-of-three"),
    pytest.param("(assert (<= (/ (+ S 1.0) 2.0) 0.5))", {}, {4: 0.25}, False, False, id="quotient-of-a-sum"),
    pytest.param("(assert (<= (* 2.0 (+ S 0.5) 0.5) 0.75))", {}, {4: 0.25}, True, False, id="product-of-three-factors"),
    pytest.param("; (assert (<= S 0.0))\n(assert (<= S 1.0)) ; (assert (<= S 0.0))", {}, {4: 0.5}, True, False, id="comments"),
])
def test_reference_reads_hand_written_cases(body, x_at, y_at, accepted, closed):
    """The reference of every cell below, checked on cases whose meaning is read off by hand: the position of an entry of a second tensor and of a rank-0 tensor, strict and closed reading, infix, ``==`` and ``!=``, nesting, exact decimal arithmetic and comments. A bug in the reference would make every cell check the wrong promise."""
    text = _Text(_SELF_TEST_HEADER + body)
    x = [0.0] * text.n_in
    y = [0.0] * text.n_out
    for i, v in x_at.items():
        x[i] = v
    for j, v in y_at.items():
        y[j] = v
    assert text.accepts(x, y, closed=closed) is accepted


@pytest.mark.parametrize("body", [
    pytest.param("(assert (<= A[0,3] 0.5))", id="index-beyond-its-axis"),
    pytest.param("(assert (<= A[3] 0.5))", id="one-index-into-a-rank-2-tensor"),
    pytest.param("(assert (<= A 0.5))", id="bare-name-of-a-rank-2-tensor"),
    pytest.param("(assert (<= W[0] 0.5))", id="undeclared-name"),
    pytest.param("(assert (<= S nan))", id="nan"),
    pytest.param("(assert (not (<= S 0.5)))", id="not"),
])
def test_reference_gives_no_meaning_to_what_the_declarations_do_not_have(body):
    """The reference refuses what the 2.0 language does not define, so a cell built on it cannot give such a text a meaning of its own."""
    text = _Text(_SELF_TEST_HEADER + body)
    with pytest.raises(_Undefined):
        text.accepts([0.0] * text.n_in, [0.0] * text.n_out)


# =========================================================================== #
# The parse's side                                                             #
# =========================================================================== #
def _q(value):
    """A stored float64 as an exact fraction; an infinite bound stays a float (a fraction compares with it)."""
    value = float(value)
    return Fraction(value) if np.isfinite(value) else value


def _stored_pairs(result):
    """The (region, unsafe groups) pairs of a parse, read as ``load_vnnlib_v2`` documents them: ``pairs`` when the parse gives them, else one region per entry of a list ``lb`` (or the one region of an array ``lb``), each with every group of ``prop``."""
    if result.get("pairs"):
        return result["pairs"]
    lb, ub, prop = result["lb"], result["ub"], result["prop"]
    if isinstance(lb, list):
        return [{"lb": l, "ub": u, "prop": prop} for l, u in zip(lb, ub)]
    return [{"lb": lb, "ub": ub, "prop": prop}]


def _in_polytopes(hg, y):
    """Is ``y`` in the union ``hg`` (a HalfSpace or a list of them) of polytopes ``G y <= g``, decided exactly on the stored numbers?"""
    for h in hg if isinstance(hg, list) else [hg]:
        G = np.asarray(h.G, dtype=np.float64)
        g = np.asarray(h.g, dtype=np.float64).ravel()
        assert G.shape == (g.size, len(y)), f"a stored row block of shape {G.shape} for {len(y)} outputs"
        if all(sum((_q(G[i, j]) * y[j] for j in range(len(y)) if G[i, j] != 0), Fraction(0)) <= _q(g[i])
               for i in range(g.size)):
            return True
    return False


def _rows_accept(pairs, x, y):
    """Is (x, y) in some stored pair: x in its box and y in every unsafe group of it?"""
    for pair in pairs:
        lb = np.asarray(pair["lb"], dtype=np.float64).ravel()
        ub = np.asarray(pair["ub"], dtype=np.float64).ravel()
        assert lb.size == ub.size == len(x), f"a stored region of {lb.size} entries for {len(x)} inputs"
        if all(_q(lo) <= xi <= _q(hi) for lo, xi, hi in zip(lb, x, ub)) and \
                all(_in_polytopes(group["Hg"], y) for group in pair["prop"]):
            return True
    return False


def _disagreements(result, text, points):
    """The points where the parse and the text disagree, by direction: counterexamples of the text the parse does not accept (``missed``), points the text rejects that the parse accepts (``accepted``), and points the stored rows accept that the text read closed rejects (``beyond``)."""
    pairs = _stored_pairs(result)
    out = {"missed": [], "accepted": [], "beyond": []}
    for x, y in points:
        xq = [Fraction(v) for v in x]
        yq = [Fraction(v) for v in y]
        truth = text.accepts(x, y)
        rows = _rows_accept(pairs, xq, yq)
        parse = rows and ("assertions" not in result or bool(evaluate_nonlinear(
            result, np.array(x, dtype=np.float64), np.array(y, dtype=np.float64))))
        if truth and not parse:
            out["missed"].append((x, y))
        if parse and not truth:
            out["accepted"].append((x, y))
        if rows and not text.accepts(x, y, closed=True):
            out["beyond"].append((x, y))
    return out


def _assert_read_as_stated(result, text, points):
    found = _disagreements(result, text, points)
    msg = []
    if found["missed"]:
        msg.append(f"{len(found['missed'])} counterexamples of the text are not accepted by the parse, "
                   f"e.g. {found['missed'][:3]}")
    if found["accepted"]:
        msg.append(f"{len(found['accepted'])} points the text rejects are accepted by the parse, "
                   f"e.g. {found['accepted'][:3]}")
    if found["beyond"]:
        msg.append(f"{len(found['beyond'])} points the stored rows accept are rejected by the text read "
                   f"closed, e.g. {found['beyond'][:3]}")
    assert not msg, "; ".join(msg)


# =========================================================================== #
# The cases                                                                    #
# =========================================================================== #
def _header(inputs=(("X", "[1,2]"),), outputs=(("Y", "[1,2]"),)):
    return ("(vnnlib-version <2.0>)\n(declare-network N\n"
            + "".join(f"  (declare-input {n} float32 {s})\n" for n, s in inputs)
            + "".join(f"  (declare-output {n} float32 {s})\n" for n, s in outputs) + ")\n")


def _grid(*axes):
    """Every combination of one value per coordinate."""
    return list(itertools.product(*axes))


def _around(centre, per_coordinate):
    """``centre`` and every point that differs from it in one coordinate, set to each of that coordinate's values."""
    points = {tuple(centre)}
    for i, values in enumerate(per_coordinate):
        for v in values:
            p = list(centre)
            p[i] = v
            points.add(tuple(p))
    return sorted(points)


def _edges(lo, hi, step=0.125):
    """Values on and on both sides of the bounds ``lo`` and ``hi``, and between them."""
    return (lo - step, lo, (lo + hi) / 2, hi, hi + step)


@dataclass
class _Case:
    text: str
    xs: list
    ys: list

    def points(self):
        return [(x, y) for x in self.xs for y in self.ys]


_B01 = "(assert (>= X[0,0] 0.0))\n(assert (<= X[0,0] 1.0))\n(assert (>= X[0,1] 0.0))\n(assert (<= X[0,1] 1.0))\n"
_X01 = _grid((0.5,), (0.5,))
_STEPS = tuple(k / 4 for k in range(-6, 9))          # -1.5 to 2 in steps of 1/4

_CASES = {
    # Bounds wider than [-1, 1] and on both sides of zero, the upper bound written first for X[0,1],
    # a zero-width coordinate, and an output compared with an output of a lower index.
    "box-wide-asymmetric-and-degenerate": _Case(
        _header(outputs=(("Y", "[1,2]"),), inputs=(("X", "[1,3]"),))
        + "(assert (>= X[0,0] -2.0))\n(assert (<= X[0,0] 0.75))\n"
          "(assert (<= X[0,1] 3.0))\n(assert (>= X[0,1] 1.5))\n"
          "(assert (>= X[0,2] 0.5))\n(assert (<= X[0,2] 0.5))\n"
          "(assert (>= Y[0,1] Y[0,0]))\n",
        _grid((-2.25, -2.0, 0.75, 1.0), (1.25, 1.5, 3.0, 3.25), (0.25, 0.5, 0.75)),
        _grid((-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0))),
    # One index per axis of a [2,3] input (height different from width) and a [2,2] output: every
    # entry has its own interval, so an entry read at another entry's position is seen.
    "multi-index-row-major": _Case(
        _header(inputs=(("X", "[2,3]"),), outputs=(("Y", "[2, 2]"),))
        + "".join(f"(assert (>= X[{i},{j}] {k / 2}))\n(assert (<= X[{i},{j}] {k / 2 + 0.25}))\n"
                  for k, (i, j) in enumerate(itertools.product(range(2), range(3))))
        + "(assert (<= Y[1,0] Y[0,1]))\n(assert (>= Y[1,1] 0.5))\n",
        _around([k / 2 + 0.125 for k in range(6)],
                [(k / 2 - 0.125, k / 2, k / 2 + 0.25, k / 2 + 0.375) for k in range(6)]),
        _grid((0.0,), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0.25, 0.5, 0.75))),
    # Two input tensors and two output tensors, the second a rank-0 tensor named bare: the joint
    # positions of B (after A) and of S (after Z) depend on the offsets.
    "two-input-tensors-and-a-scalar-output": _Case(
        _header(inputs=(("A", "[2]"), ("B", "[1, 2]")), outputs=(("Z", "[1,2]"), ("S", "[]")))
        + "(assert (>= A[0] -1.5))\n(assert (<= A[0] -1.0))\n(assert (>= A[1] 0.25))\n(assert (<= A[1] 0.5))\n"
          "(assert (>= B[0,0] 2.0))\n(assert (<= B[0,0] 2.5))\n(assert (>= B[0,1] -0.5))\n(assert (<= B[0,1] 0.75))\n"
          "(assert (>= S Z[0,1]))\n(assert (<= Z[0,0] 0.5))\n",
        _around([-1.25, 0.375, 2.25, 0.125],
                [_edges(-1.5, -1.0), _edges(0.25, 0.5), _edges(2.0, 2.5), _edges(-0.5, 0.75)]),
        _grid((0.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0))),
    # Input bounds through a coefficient: magnitude above and below one, both signs, so the bound is
    # -const / c and its side depends on the sign of c.
    "scaled-input-bounds": _Case(
        _header(outputs=(("Y", "[1,1]"),))
        + "(assert (<= (* 2.0 X[0,0]) 1.0))\n(assert (>= (* 0.5 X[0,0]) -1.0))\n"
          "(assert (<= (* -4.0 X[0,1]) 2.0))\n(assert (>= (* -0.25 X[0,1]) -0.5))\n"
          "(assert (>= Y[0,0] 0.5))\n",
        _grid(_edges(-2.0, 0.5), _edges(-0.5, 2.0)),
        _grid((0.25, 0.5, 0.75))),
    # Input bounds through negation, a difference with the number first and a shifted sum.
    "negated-and-shifted-input-bounds": _Case(
        _header(outputs=(("Y", "[1,1]"),))
        + "(assert (<= (- X[0,0]) 0.5))\n(assert (>= (- X[0,0]) -1.5))\n"
          "(assert (<= (- 0.25 X[0,1]) 1.0))\n(assert (<= (+ X[0,1] 0.5) 2.0))\n"
          "(assert (<= Y[0,0] -0.25))\n",
        _grid(_edges(-0.5, 1.5), _edges(-0.75, 1.5)),
        _grid((-0.5, -0.25, 0.0))),
    # Output rows through every arithmetic form: a difference, a negation, a product of two
    # constants and a variable, a product with an offset, a quotient with an offset, a sum of three.
    "affine-output-rows": _Case(
        _header() + _B01
        + "(assert (<= (- Y[0,0] Y[0,1]) 0.5))\n"
          "(assert (<= (- Y[0,0]) 0.5))\n"
          "(assert (<= (* 0.5 4.0 Y[0,1]) 3.0))\n"
          "(assert (>= (* 2.0 (+ Y[0,1] 0.5)) 1.0))\n"
          "(assert (<= (/ (+ Y[0,0] 1.0) 2.0) 1.25))\n"
          "(assert (<= (+ Y[0,0] Y[0,1] 0.25) 2.75))\n",
        _X01, _grid(_STEPS, _STEPS)),
    # Infix comparisons, one with the number first and one strict between two outputs.
    "infix-comparisons": _Case(
        _header()
        + "(assert (X[0,0] >= -1.0))\n(assert (X[0,0] <= 0.5))\n(assert (0.25 <= X[0,1]))\n(assert (X[0,1] <= 1.5))\n"
          "(assert (Y[0,0] < Y[0,1]))\n(assert (Y[0,1] <= 2.0))\n",
        _grid(_edges(-1.0, 0.5), _edges(0.25, 1.5)),
        _grid((0.0, 1.0, 2.0), (0.0, 1.0, 2.0, 2.25))),
    # Strict comparisons in prefix form, '>' only: on the bound the text rejects the point and only
    # the kept assertions can say so.
    "strict-prefix-greater-only": _Case(
        _header()
        + "(assert (> X[0,0] -0.5))\n(assert (<= X[0,0] 0.5))\n(assert (>= X[0,1] 0.0))\n(assert (<= X[0,1] 1.0))\n"
          "(assert (> Y[0,0] Y[0,1]))\n(assert (> Y[0,1] 0.25))\n",
        _grid(_edges(-0.5, 0.5), (0.5,)),
        _grid((0.0, 0.25, 0.5, 0.75), (0.0, 0.25, 0.5, 0.75))),
    # Strict comparisons in prefix form, '<' only.
    "strict-prefix-less-only": _Case(
        _header()
        + "(assert (>= X[0,0] 0.0))\n(assert (<= X[0,0] 1.0))\n(assert (>= X[0,1] -1.0))\n(assert (< X[0,1] 0.75))\n"
          "(assert (< Y[0,0] -0.25))\n(assert (<= Y[0,1] Y[0,0]))\n",
        _grid((0.5,), _edges(-1.0, 0.75)),
        _grid((-0.5, -0.25, 0.0), (-0.75, -0.5, -0.25))),
    # Bounds stated twice inside one conjunction (the tighter first), an equality inside a box, an
    # equality between two outputs.
    "repeated-bounds-and-equalities": _Case(
        _header()
        + "(assert (and (<= X[0,0] 0.5) (<= X[0,0] 1.5) (>= X[0,0] -0.5) (>= X[0,0] -2.0)))\n"
          "(assert (and (>= X[0,1] -1.0) (<= X[0,1] 1.0) (== X[0,1] 0.25)))\n"
          "(assert (== Y[0,0] Y[0,1]))\n(assert (<= Y[0,1] 1.0))\n",
        _grid((-2.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.5), (-1.0, 0.0, 0.25, 0.5, 1.0)),
        _grid((-0.5, 0.0, 0.5, 1.0, 1.5), (-0.5, 0.0, 0.5, 1.0, 1.5))),
    # An output 'or' of 'and's holding an 'or' (nesting depth three), ANDed with two more output asserts.
    "output-or-and-or-with-more-asserts": _Case(
        _header() + _B01
        + "(assert (or (and (<= Y[0,0] -0.5) (or (>= Y[0,1] 1.0) (<= Y[0,1] -1.0)))\n"
          "            (and (>= Y[0,0] 0.5) (>= Y[0,1] 0.5))))\n"
          "(assert (<= Y[0,0] 1.25))\n"
          "(assert (or (<= Y[0,1] 1.5) (>= Y[0,0] 1.0)))\n",
        _X01, _grid(_STEPS, _STEPS)),
    # Disjunctive input regions: each region is the base box of the separate asserts intersected with
    # its disjunct; the regions differ in both bounds, a bound is repeated inside a disjunct (the tighter
    # first), one bound is strict, and the last region is cut by the base box.
    "input-or-regions-with-a-base-box": _Case(
        _header(outputs=(("Y", "[1,1]"),))
        + "(assert (>= X[0,1] -1.0))\n(assert (<= X[0,1] 1.0))\n(assert (>= X[0,0] -3.0))\n(assert (<= X[0,0] 3.0))\n"
          "(assert (or (and (>= X[0,0] -2.0) (<= X[0,0] -1.0) (<= X[0,1] 0.5) (<= X[0,1] 2.0))\n"
          "            (and (> X[0,0] 0.5) (<= X[0,0] 1.5) (>= X[0,1] -0.5) (>= X[0,1] -2.0))\n"
          "            (and (>= X[0,0] 2.5) (<= X[0,0] 4.0))))\n"
          "(assert (>= Y[0,0] 0.5))\n",
        _grid((-3.25, -3.0, -2.25, -2.0, -1.5, -1.0, -0.75, 0.25, 0.5, 0.75, 1.5, 1.75, 2.25, 2.5, 3.0, 3.25),
              (-1.25, -1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0, 1.25)),
        _grid((0.25, 0.5, 0.75))),
    # A combined input and output 'or': each disjunct pairs its own region (the base box intersected
    # with its input atoms; the regions differ in both bounds) with its own output rows (two in the
    # first disjunct, none in the input-only third), and every pair carries the separate output assert.
    "combined-or-with-separate-asserts": _Case(
        _header()
        + "(assert (>= X[0,0] -2.0))\n(assert (<= X[0,0] 2.0))\n(assert (>= X[0,1] -1.0))\n(assert (<= X[0,1] 1.0))\n"
          "(assert (or (and (<= X[0,0] -0.5) (>= X[0,1] 0.25) (>= Y[0,0] Y[0,1]) (<= Y[0,1] 0.5))\n"
          "            (and (>= X[0,0] 0.5) (<= X[0,0] 1.5) (<= X[0,1] 0.0) (>= Y[0,1] 1.0))\n"
          "            (and (>= X[0,0] 1.75) (<= X[0,1] -0.75))))\n"
          "(assert (<= Y[0,0] 1.5))\n",
        _grid((-2.25, -2.0, -0.75, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 1.75, 2.0, 2.25),
              (-1.25, -1.0, -0.75, -0.5, 0.0, 0.25, 0.5, 1.0, 1.25)),
        _grid((-0.5, 0.5, 1.5, 1.75), (0.0, 0.5, 0.75, 1.0))),
    # A combined conjunction (one pair) beside a strict input bound and a strict output assert that are
    # asserts of their own: the kept assertions must hold all of them.
    "combined-and-with-separate-strict-asserts": _Case(
        _header()
        + "(assert (>= X[0,0] -1.0))\n(assert (<= X[0,0] 1.0))\n(assert (> X[0,1] -0.5))\n(assert (<= X[0,1] 0.5))\n"
          "(assert (and (<= X[0,0] 0.5) (>= Y[0,0] 0.25)))\n"
          "(assert (< Y[0,1] 0.75))\n",
        _grid((-1.25, -1.0, 0.5, 0.75), (-0.75, -0.5, -0.25, 0.5, 0.75)),
        _grid((0.0, 0.25, 0.5), (0.5, 0.75, 1.0))),
    # A combined 'and' of two 'or's (nesting depth three): four (region, rows) pairs.
    "combined-and-of-ors": _Case(
        _header()
        + "(assert (>= X[0,0] -1.0))\n(assert (<= X[0,0] 1.0))\n(assert (>= X[0,1] -1.0))\n(assert (<= X[0,1] 1.0))\n"
          "(assert (and (or (and (<= X[0,0] 0.0) (>= Y[0,0] 0.5)) (and (>= X[0,0] 0.5) (<= Y[0,0] -0.5)))\n"
          "             (or (>= Y[0,1] 1.0) (and (<= X[0,1] 0.0) (<= Y[0,1] -1.0)))))\n",
        _grid((-1.25, -1.0, 0.0, 0.25, 0.5, 1.0, 1.25), (-1.25, -1.0, 0.0, 0.25, 1.0, 1.25)),
        _grid((-0.75, -0.5, 0.0, 0.5, 0.75), (-1.25, -1.0, 0.0, 1.0, 1.25))),
    # Comments: a whole line that holds an assert, a comment after an assert that holds another; and an
    # assert broken over lines with no indentation between two atoms.
    "comments-and-line-breaks": _Case(
        _header(inputs=(("X", "[1]"),), outputs=(("Y", "[1]"),))
        + "; (assert (<= X[0] 0.25))\n"
          "(assert (>= X[0] -1.0)) ; (assert (>= X[0] 0.5))\n"
          "(assert (<= X[0]\n1.0))\n"
          "(assert (>=\nY[0] 0.5))\n",
        _grid(tuple(k / 4 for k in range(-5, 6))),
        _grid((0.25, 0.5, 0.75))),
}


def _write(tmp_path, text):
    path = tmp_path / "spec.vnnlib"
    path.write_text(text)
    return str(path)


def _check_case(case):
    """Setup checks of a case, outside every marker: the points have a counterexample and a point the text rejects."""
    text = _Text(case.text)
    verdicts = [text.accepts(x, y) for x, y in case.points()]
    if not any(verdicts) or all(verdicts):
        pytest.fail("the case needs a counterexample and a rejected point among its points")
    return text


@pytest.mark.parametrize("name", list(_CASES))
def test_linear_file_is_read_as_the_text_states(tmp_path, name):
    """Through ``load_vnnlib``, a file of the supported linear fragment is read (not refused) and its parse accepts exactly the counterexamples of the text, at points on, inside and outside every bound; the stored rows accept nothing beyond the text read closed. Catches a lost, flipped, loosened or misplaced bound or row (a sign or an offset of the arithmetic, a coefficient's sign deciding the side of a bound, a scaled bound computed as a product, a repeated bound overwriting the tighter one, an equality read one-sided), an index flattened in another order or without its tensor's offset, a conjunct or disjunct dropped, an 'and' of 'or's distributed by pairs instead of every combination, an 'or' region sharing its bound arrays with the others, a combined disjunct losing a row or the separate output asserts, a strict comparison with no kept assertion to reject its bound, a comment read as an assert, and a supported form refused."""
    case = _CASES[name]
    text = _check_case(case)
    result = load_vnnlib(_write(tmp_path, case.text))
    _assert_read_as_stated(result, text, case.points())


@pytest.mark.parametrize("name", ["input-or-regions-with-a-base-box", "combined-or-with-separate-asserts",
                                  "two-input-tensors-and-a-scalar-output"])
def test_file_read_by_load_vnnlib_v2_itself_is_read_as_the_text_states(tmp_path, name):
    """The same promise through ``load_vnnlib_v2(path)`` with no text given, which reads the file itself and hands back the structure the loader documents (no ``pairs`` for disjunctive input regions, which ``_stored_pairs`` builds as documented). Catches a lowering that only the dispatcher's own post-processing makes right, and a direct call that does not read the file it is given."""
    case = _CASES[name]
    text = _check_case(case)
    result = load_vnnlib_v2(_write(tmp_path, case.text))
    _assert_read_as_stated(result, text, case.points())


# =========================================================================== #
# Refused: forms outside the linear fragment                                   #
# =========================================================================== #
_O = "(assert (>= Y[0,0] 0.5))\n"
_NEW_CRASH = ("a malformed 2.0 arithmetic term ({form}) crashes the reader with {exc} instead of being "
              "refused with ValueError")
_REFUSED = {
    # The linear lowering handles one combined input and output assert; a second one would be dropped.
    "two-combined-asserts": _header() + _B01
    + "(assert (or (and (<= X[0,0] 0.5) (>= Y[0,0] 0.5))))\n(assert (or (and (>= X[0,1] 0.5) (<= Y[0,1] 0.25))))\n",
    # A combined assert beside an input-only 'or': the 'or' regions would be dropped.
    "combined-beside-an-input-or": _header() + _B01
    + "(assert (or (<= X[0,0] 0.25) (>= X[0,0] 0.75)))\n(assert (or (and (<= X[0,1] 0.5) (>= Y[0,0] 0.5))))\n",
    # Two input-only 'or's: the regions are their intersection, which the lowering does not build.
    "two-input-ors": _header() + _B01
    + "(assert (or (<= X[0,0] 0.25) (>= X[0,0] 0.75)))\n(assert (or (<= X[0,1] 0.25) (>= X[0,1] 0.75)))\n" + _O,
    # An input constraint over two inputs is not a box bound.
    "input-sum-of-two-inputs": _header() + _B01 + "(assert (<= (+ X[0,0] X[0,1]) 1.0))\n" + _O,
    "input-compared-with-an-input": _header() + _B01 + "(assert (<= X[0,0] X[0,1]))\n" + _O,
    # One comparison of an input with an output.
    "input-compared-with-an-output": _header() + _B01 + "(assert (<= X[0,0] Y[0,0]))\n",
    # A comparison of two numbers inside a combined disjunct.
    "number-comparison-in-a-combined-disjunct": _header() + _B01
    + "(assert (or (and (<= X[0,0] 0.5) (>= Y[0,0] 0.5) (<= 1.0 0.5))))\n",
    "no-output-assert": _header() + _B01,
    "no-output-declaration": ("(vnnlib-version <2.0>)\n(declare-network N\n  (declare-input X float32 [1,2])\n)\n"
                              + _B01 + _O),
    "unbalanced-parentheses": _header() + _B01 + "(assert (>= Y[0,0] 0.5)\n",
    "empty-assert": _header() + _B01 + "(assert)\n" + _O,
    # A token that is neither a declared name nor a number, and an operator the language does not have
    # inside an arithmetic term.
    "unknown-atom": _header() + _B01 + "(assert (<= Y[0,0] abc))\n",
    "unknown-operator-in-a-term": _header() + _B01 + "(assert (<= (^ Y[0,0] 2.0) 0.5))\n",
    # An empty input region: an equality below an earlier lower bound, and one above an earlier upper
    # bound inside one conjunction.
    "equality-below-an-earlier-lower-bound": _header()
    + "(assert (>= X[0,0] 0.75))\n(assert (<= X[0,0] 1.0))\n(assert (== X[0,0] 0.5))\n"
      "(assert (>= X[0,1] 0.0))\n(assert (<= X[0,1] 1.0))\n" + _O,
    "equality-above-an-earlier-upper-bound": _header()
    + "(assert (and (>= X[0,0] 0.0) (<= X[0,0] 0.25) (== X[0,0] 0.5)))\n"
      "(assert (>= X[0,1] 0.0))\n(assert (<= X[0,1] 1.0))\n" + _O,
    "input-left-without-an-upper-bound": _header()
    + "(assert (>= X[0,0] 0.0))\n(assert (<= X[0,0] 1.0))\n(assert (>= X[0,1] 0.0))\n" + _O,
    # Malformed arithmetic.
    "difference-of-nothing": pytest.param(
        _header() + _B01 + "(assert (<= (-) Y[0,0]))\n",
        marks=known_bug(23, _NEW_CRASH.format(form="(-)", exc="IndexError"), raises=IndexError)),
    "quotient-of-one-term": pytest.param(
        _header() + _B01 + "(assert (<= (/ Y[0,0]) 0.5))\n",
        marks=known_bug(23, _NEW_CRASH.format(form="(/ Y[0,0])", exc="IndexError"), raises=IndexError)),
    "quotient-by-zero": pytest.param(
        _header() + _B01 + "(assert (<= (/ Y[0,0] 0.0) 0.5))\n",
        marks=known_bug(23, _NEW_CRASH.format(form="(/ Y[0,0] 0.0)", exc="ZeroDivisionError"),
                        raises=ZeroDivisionError)),
    # Names the language does not have: 'nan' is neither a number of the language nor a declared name
    # (the official parser reports an undeclared variable); Python's float() reads it.
    "nan-read-as-a-number": pytest.param(
        _header() + _B01 + "(assert (>= Y[0,0] nan))\n",
        marks=known_bug(6, "a 2.0 atom that is neither a declared name nor a number of the language "
                           "(nan, inf, 1_0) is read by Python's float() instead of refused: (>= Y nan) "
                           "is stored as a row whose right-hand side is nan, and the file is verified "
                           "instead of refused",
                        raises=DID_NOT_RAISE)),
}


@pytest.mark.parametrize("text", [v if not isinstance(v, str) else pytest.param(v) for v in _REFUSED.values()],
                         ids=list(_REFUSED))
def test_form_outside_the_linear_fragment_is_refused(tmp_path, text):
    """A form the linear lowering does not support (two combined asserts, a combined assert beside an input 'or', two input 'or's, a constraint over two inputs, a comparison of an input with an output, a comparison of two numbers inside a combined disjunct), a file with no output property, a malformed file (unbalanced parentheses, an empty assert, malformed arithmetic, a name or form outside the language), and a file whose input region is empty or unbounded must raise ``ValueError`` through ``load_vnnlib``. Catches a refusal that is removed (the unsupported part is then dropped or overwritten without a message and the stored region is wider or narrower than the text), a parse error that falls back to the faithful nonlinear loader or is swallowed, an equality that overwrites an earlier bound instead of intersecting with it, and a crash of another exception type."""
    with pytest.raises(ValueError):
        load_vnnlib(_write(tmp_path, text))


@known_bug(6, "a 2.0 assert written '( assert ...)', with a space after its parenthesis, is dropped without "
              "a message (the reader looks for the text '(assert'), instead of being read as the assert it "
              "is as an s-expression or refused (the official parser rejects it): the stored unsafe region "
              "leaves out its rows (invalid SAT kind)")
def test_assert_with_a_space_after_its_parenthesis_is_read_or_refused(tmp_path):
    """``( assert F)`` is the s-expression ``(assert F)`` with a space after the parenthesis; the official grammar rejects the space. Either reading keeps the promise: the parse means what the reference reads (an assert), or the file is refused with ``ValueError``. Dropping it silently reads a weaker file: here the output row y1 <= 0.25 is lost and outputs with y1 above 0.25 are accepted."""
    body = _header() + _B01 + "( assert (<= Y[0,1] 0.25))\n" + _O
    text = _Text(body)
    try:
        result = load_vnnlib(_write(tmp_path, body))
    except ValueError:
        return
    _assert_read_as_stated(result, text, [(x, y) for x in _X01 for y in _grid((0.25, 0.5, 0.75),
                                                                                 (0.0, 0.25, 0.5))])


@known_bug(6, "a 2.0 division of three operands in a linear text is lowered on its first two operands only: "
              "(>= (/ Y[0,0] 0.5 0.5) 3.0) is stored as the row y >= 1.5 where the text, read left to right "
              "like the n-ary '-', says y >= 0.75, so the counterexamples with y in [0.75, 1.5) are left out "
              "of the unsafe region, a false UNSAT kind (registry row SPEC-38)")
def test_division_of_three_operands_is_read_left_to_right_or_refused(tmp_path):
    """``(/ a b c)`` read left to right, as the reader reads the n-ary ``-``, is ``(a / b) / c``. The reference of this file gives a division of three operands no meaning, so the truth is computed here in fractions. Either the file is refused with ``ValueError``, or its parse accepts exactly the outputs with ``(y / 0.5) / 0.5 >= 3``, at outputs on and on both sides of 0.75 and of 1.5 (the bound of a reading that drops the third operand). Catches a third operand that is dropped, and one applied as a factor."""
    body = _header(outputs=(("Y", "[1,1]"),)) + _B01 + "(assert (>= (/ Y[0,0] 0.5 0.5) 3.0))\n"
    try:
        result = load_vnnlib(_write(tmp_path, body))
    except ValueError:
        return
    pairs = _stored_pairs(result)
    x = (0.5, 0.5)
    wrong = []
    for y in (0.5, 0.625, 0.75, 1.0, 1.5, 2.0):
        truth = Fraction(y) / Fraction(1, 2) / Fraction(1, 2) >= 3
        parse = _rows_accept(pairs, [Fraction(v) for v in x], [Fraction(y)]) and (
            "assertions" not in result or bool(evaluate_nonlinear(result, np.array(x), np.array([y]))))
        if parse != truth:
            wrong.append((y, truth))
    assert not wrong, f"(output, is it a counterexample of the text) where the parse says the opposite: {wrong}"


@known_bug(6, "SPEC-21 by the one-index route: a single index into a 2.0 tensor of rank above one is read "
              "as a flat index with no check against the tensor's size, so A[2] of a [1,2] tensor names "
              "B[0,0] of the next tensor and the bound lands there without a message", raises=DID_NOT_RAISE)
def test_one_index_beyond_its_tensor_is_refused(tmp_path):
    """``A[2]`` of a [1, 2] tensor names no entry (the official parser rejects it); the reader must refuse it with ``ValueError``. Today ``_ravel`` passes a single index through unchecked and the resolver adds A's offset, so the bound is stored on ``B[0,0]``, an entry the text never bounds that way. The full-rank route (``X[0,2]`` of a [2,2] tensor) is marked in ``tests/contracts/test_specs.py``."""
    text = (_header(inputs=(("A", "[1,2]"), ("B", "[1,2]")), outputs=(("Y", "[1,1]"),))
            + "".join(f"(assert (>= {v} 0.0))\n(assert (<= {v} 1.0))\n"
                      for v in ("A[0,0]", "A[0,1]", "B[0,0]", "B[0,1]"))
            + "(assert (<= A[2] 0.25))\n" + _O)
    reference = _Text(text)
    try:
        reference.accepts([0.5] * 4, [0.5])
        reference_gives_meaning = True
    except _Undefined:
        reference_gives_meaning = False
    assert not reference_gives_meaning, "the reference gives A[2] a meaning"
    with pytest.raises(ValueError):
        load_vnnlib(_write(tmp_path, text))


# =========================================================================== #
# Tensor metadata                                                              #
# =========================================================================== #
_TWO_TENSORS = _header(inputs=(("A", "[2]"), ("B", "[1, 2]")), outputs=(("Z", "[1,2]"), ("S", "[]")))
_TWO_TENSOR_BOX = ("(assert (>= A[0] 0.0))\n(assert (<= A[0] 1.0))\n(assert (>= A[1] 0.0))\n(assert (<= A[1] 1.0))\n"
                   "(assert (>= B[0,0] 0.0))\n(assert (<= B[0,0] 1.0))\n(assert (>= B[0,1] 0.0))\n"
                   "(assert (<= B[0,1] 1.0))\n")


@pytest.mark.parametrize("body", [
    pytest.param(_TWO_TENSOR_BOX + "(assert (>= S Z[0,1]))\n", id="output-asserts"),
    pytest.param(_TWO_TENSOR_BOX + "(assert (or (and (<= B[0,1] 0.5) (>= S 0.5))))\n", id="combined-assert"),
])
def test_tensor_metadata_names_each_tensor_with_its_joint_offset(tmp_path, body):
    """``input_tensors`` and ``output_tensors`` list every declared tensor in declaration order with its shape, its joint offset and its size, written here by hand: A [2] at 0, B [1,2] at 2; Z [1,2] at 0, the rank-0 S at 2. The runner reads them to tell a multi-input specification from a single-input one, and a caller splits a joint vector back into tensors with them. Catches an offset recorded after it is advanced (each tensor's end instead of its start), a wrong size or shape, and metadata missing from the combined lowering."""
    result = load_vnnlib(_write(tmp_path, _TWO_TENSORS + body))
    assert result["input_tensors"] == [{"name": "A", "shape": (2,), "offset": 0, "size": 2},
                                       {"name": "B", "shape": (1, 2), "offset": 2, "size": 2}]
    assert result["output_tensors"] == [{"name": "Z", "shape": (1, 2), "offset": 0, "size": 2},
                                        {"name": "S", "shape": (), "offset": 2, "size": 1}]


# =========================================================================== #
# Outside the linear fragment: the faithful loader                             #
# =========================================================================== #
@pytest.mark.parametrize("body", [
    pytest.param("(assert (!= Y[0,0] Y[0,1]))\n", id="output-disequality"),
    pytest.param("(assert (<= (* Y[0,0] Y[0,1]) 0.25))\n", id="product-of-two-outputs"),
    pytest.param("(assert (or (and (>= Y[0,0] 0.5) (!= Y[0,1] 0.25)) (<= Y[0,0] -0.5)))\n",
                 id="disequality-inside-an-or"),
])
def test_file_outside_the_linear_fragment_loads_as_its_assertions(tmp_path, body):
    """An output ``!=`` (non-convex) or a product of two variables (nonlinear) is not lowered to rows; ``load_vnnlib_v2`` hands the file to the faithful loader, whose result has ``format='nonlinear'`` and whose kept assertions accept exactly the counterexamples of the text, while its best-effort input box holds the input of every counterexample. Catches a ``!=`` lowered to the two rows of an equality (the plane the text excludes), a dispatch that refuses instead of loading (the fallback condition requiring both words of the two error messages), and an assertion or box that misreads the text."""
    text_body = _header() + _B01 + body
    text = _Text(text_body)
    result = load_vnnlib(_write(tmp_path, text_body))
    assert result.get("format") == "nonlinear", f"loaded as format {result.get('format')!r}"
    points = _grid(_edges(0.0, 1.0), _edges(0.0, 1.0), (-0.75, -0.5, 0.0, 0.25, 0.5, 1.0),
                   (-0.5, 0.0, 0.25, 0.5, 1.0))
    wrong = []
    for x0, x1, y0, y1 in points:
        truth = text.accepts((x0, x1), (y0, y1))
        if bool(evaluate_nonlinear(result, np.array([x0, x1]), np.array([y0, y1]))) != truth:
            wrong.append(((x0, x1), (y0, y1), truth))
        elif truth and not all(_q(lo) <= Fraction(v) <= _q(hi)
                               for lo, v, hi in zip(np.ravel(result["lb"]), (x0, x1), np.ravel(result["ub"]))):
            wrong.append(((x0, x1), (y0, y1), "outside the box"))
    assert not wrong, f"{len(wrong)} points read otherwise than the text, e.g. {wrong[:3]}"


# =========================================================================== #
# Kept assertions and the checker's float32 reading                            #
# =========================================================================== #
def _float32_value(text, node, x, y):
    """The value of an arithmetic term as the VNN-COMP 2026 checker computes it for a float32 network: every variable and every constant (read as a Python float) in float32, left to right, rounding after each operation."""
    f32 = np.float32
    if isinstance(node, str):
        try:
            return f32(float(node))
        except ValueError:
            kind, index = text.position(node)
            return f32(x[index] if kind == "in" else y[index])
    args = [_float32_value(text, c, x, y) for c in node[1:]]
    head = node[0]
    if head == "-" and len(args) == 1:
        return -args[0]
    out = args[0]
    for a in args[1:]:
        out = {"+": lambda p, q: p + q, "-": lambda p, q: p - q, "*": lambda p, q: p * q,
               "/": lambda p, q: p / q}[head](out, a)
    return f32(out)


def _float32_accepts(text, x, y):
    """Does the checker's float32 reading of the text accept (x, y)? Every assert of the cases below is a conjunction of comparisons."""
    def holds(node):
        if node[0] == "and":
            return all(holds(c) for c in node[1:])
        op, a, b = node
        return bool(_ORDER[op](_float32_value(text, a, x, y), _float32_value(text, b, x, y)))
    return all(holds(a) for a in text.asserts)


def _floats32_around(value, k=3):
    out = {np.float32(value)}
    up = down = np.float32(value)
    for _ in range(k):
        up = np.nextafter(up, np.float32(np.inf))
        down = np.nextafter(down, np.float32(-np.inf))
        out |= {up, down}
    return sorted(float(v) for v in out)


_E23 = 2.0 ** -23
_F32_CASES = {
    # A difference of a difference, no '+' anywhere in the text: at y = (1 + 2^-23, -2^-24, 1) the
    # float64 row y0 - y1 - y2 <= 3 * 2^-24 holds with equality, while the checker's float32 evaluation
    # rounds y0 - y1 up to 1 + 2^-22 and gets 2^-22 > 3 * 2^-24. (The nested binary form fixes the order
    # of the two subtractions; the n-ary (- a b c) is not used, since the order in which the checker
    # evaluates it is not stated by the official parser.)
    "difference-of-three-outputs": (
        _header(inputs=(("X", "[1,1]"),), outputs=(("Y", "[1,3]"),))
        + "(assert (>= X[0,0] 0.0))\n(assert (<= X[0,0] 1.0))\n"
          "(assert (<= (- (- Y[0,0] Y[0,1]) Y[0,2]) 0.000000178813934326171875))\n",
        [(0.5,)],
        _grid(_floats32_around(1 + _E23, 1), _floats32_around(-_E23 / 2, 1), (1.0,))),
    # A coefficient the checker rounds to float32: at y = 3 the float64 row 0.3 y <= 0.9 holds, while
    # float32(0.3) * 3 rounds to the float32 number above float32(0.9).
    "coefficient-of-an-output": (
        _header(inputs=(("X", "[1,1]"),), outputs=(("Y", "[1,1]"),))
        + "(assert (>= X[0,0] 0.0))\n(assert (<= X[0,0] 1.0))\n(assert (<= (* 0.3 Y[0,0]) 0.9))\n",
        [(0.5,)],
        _grid(_floats32_around(3.0))),
}


@pytest.mark.parametrize("name", list(_F32_CASES))
def test_kept_assertions_reject_what_the_float32_checker_rejects(tmp_path, name):
    """The rows are float64 and the competition's checker evaluates a witness of a float32 network in float32, left to right; with arithmetic in the text the rows can accept a point the checker rejects, which is why the reader keeps the source assertions (``_lower_linear``, ``_keep_assertions_for_sat_checks``). The parse's SAT check as the runner applies it (the rows, then the kept assertions in float64 and in float32) must accept no point the checker's float32 reading rejects; the reference is a float32 evaluator written here. Catches arithmetic left undetected (an operator scan that stops early or misses an operator), so that no assertion is kept and an invalid witness is reported."""
    text_body, xs, ys = _F32_CASES[name]
    text = _Text(text_body)
    points = [(x, y) for x in xs for y in ys]
    result = load_vnnlib(_write(tmp_path, text_body))
    pairs = _stored_pairs(result)
    rows = {(x, y): _rows_accept(pairs, [Fraction(v) for v in x], [Fraction(v) for v in y]) for x, y in points}
    if not any(rows[p] and not _float32_accepts(text, *p) for p in points):
        pytest.fail("the case has no point the rows accept and the checker rejects")
    wrongly = []
    for x, y in points:
        if not rows[(x, y)]:
            continue
        kept = "assertions" not in result or (
            bool(evaluate_nonlinear(result, np.array(x), np.array(y)))
            and bool(evaluate_nonlinear(result, np.array(x), np.array(y), dtype=np.float32)))
        if kept and not _float32_accepts(text, x, y):
            wrongly.append((x, y))
    assert not wrongly, (f"{len(wrongly)} points the float32 checker rejects pass the parse's SAT check, "
                         f"e.g. {wrongly[:3]}")


# =========================================================================== #
# The top-level 'prop' of a combined file                                      #
# =========================================================================== #
@known_bug(6, "SPEC-11 by the 2.0 route: a combined input and output file (paired) still carries a top-level "
              "'prop' with one group per disjunct; read as one specification over a region it is the "
              "conjunction of every disjunct's rows (empty here: a false UNSAT), and read paired (region i "
              "with entry i) it leaves out the separate output asserts (an invalid SAT)")
def test_combined_file_has_no_top_level_prop_that_misstates_it(tmp_path):
    """``load_vnnlib`` documents ``prop`` as the output property to verify, and for a paired parse ``_build_pairs`` documents region i with entry i of ``prop``. A combined 2.0 file has a correct ``pairs``, but its top-level ``prop`` (one group per disjunct, without the separate output asserts) misstates the file under either reading. The promise: the top-level ``prop`` is absent, or both readings keep every counterexample and accept no rejected point."""
    case = _CASES["combined-or-with-separate-asserts"]
    text = _check_case(case)
    result = load_vnnlib(_write(tmp_path, case.text))
    if "prop" not in result:
        return
    regions = list(zip(result["lb"], result["ub"])) if isinstance(result["lb"], list) \
        else [(result["lb"], result["ub"])]
    as_one = [{"lb": lo, "ub": hi, "prop": result["prop"]} for lo, hi in regions]
    paired = [{"lb": lo, "ub": hi, "prop": [p]} for (lo, hi), p in zip(regions, result["prop"])]
    missed, accepted = [], []
    for x, y in case.points():
        xq, yq = [Fraction(v) for v in x], [Fraction(v) for v in y]
        truth = text.accepts(x, y)
        if truth and not _rows_accept(as_one, xq, yq):
            missed.append((x, y))
        if not truth and _rows_accept(paired, xq, yq):
            accepted.append((x, y))
    assert not missed and not accepted, (
        f"read as one specification, {len(missed)} counterexamples are left out (e.g. {missed[:2]}); "
        f"read paired, {len(accepted)} rejected points are accepted (e.g. {accepted[:2]})")
