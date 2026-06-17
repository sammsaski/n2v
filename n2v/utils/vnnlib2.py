"""VNNLIB 2.0 parser for n2v.

VNN-COMP 2026 migrated its specifications to the VNNLIB 2.0 format::

    (vnnlib-version <2.0>)
    (declare-network N
        (declare-input  X float32 [1, 1, 1, 5])
        (declare-output Y float32 [1, 5])
    )
    (assert (<= X[0,0,0,0] 0.68))
    (assert (>= X[0,0,0,0] 0.60))
    ...
    (assert (or (<= Y[0,0] Y[0,1]) (<= Y[0,0] Y[0,2])))

Variables are *tensor-indexed* (``X[0,0,0,0]``) instead of the flat
``(declare-const X_0 Real)`` / ``X_0`` of VNNLIB 1.0. This module parses
the **single-network linear** fragment into the SAME dict structure the
1.0 parser produces (``{'lb', 'ub', 'prop'}``), so everything downstream
(:func:`n2v.utils.verify_specification.verify_specification`, the
VNN-COMP runner) is unchanged.

Semantics (VNN-COMP convention): the conjunction of all top-level asserts
describes the *unsafe* region — the (input, output) pair that, if
realizable, is a counterexample to safety. Input-only asserts define the
input box (a disjunction of boxes when an ``or`` is present); asserts that
mention an output variable define the output property. Each output assert
is converted to disjunctive normal form (an OR of AND-of-linear-rows) and
emitted as one property group; groups are ANDed together — exactly the
shape ``verify_specification`` consumes.

Anything outside that fragment raises :class:`VNNLibParseError` with a
concrete reason rather than silently mis-parsing:

  * multiple ``declare-network`` blocks (relational / equivalence specs,
    e.g. monotonic_acasxu, isomorphic_acasxu),
  * nonlinear terms (a product of two variables, e.g. the
    ``adaptive_cruise_control_non_linear`` specs),
  * ``!=`` in the unsafe region (non-convex),
  * a single boolean clause that mixes input and output variables, or a
    non-box linear input constraint.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from n2v.sets.halfspace import HalfSpace

logger = logging.getLogger(__name__)

_COMPARE = {"<=", "<", ">=", ">", "==", "!="}
_ARITH = {"+", "-", "*", "/"}


class VNNLibParseError(ValueError):
    """Raised when a VNNLIB 2.0 spec falls outside the supported fragment."""


# ---------------------------------------------------------------------------
# Tokenizer / reader for s-expressions
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove ``;`` line comments and flatten to a single whitespace string."""
    out = []
    for line in text.splitlines():
        idx = line.find(";")
        if idx >= 0:
            line = line[:idx]
        out.append(line)
    return " ".join(out)


def _extract_sexprs(text: str, head: str) -> List[str]:
    """Return every balanced-paren substring starting with ``head``."""
    res = []
    i, n = 0, len(text)
    while True:
        j = text.find(head, i)
        if j < 0:
            break
        depth, k = 0, j
        while k < n:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if depth != 0:
            raise VNNLibParseError(f"unbalanced parentheses after position {j}")
        res.append(text[j:k + 1])
        i = k + 1
    return res


def _read(sexpr: str):
    """Parse one balanced s-expression string into nested lists/atoms."""
    tokens = sexpr.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def read_from():
        nonlocal pos
        if pos >= len(tokens):
            raise VNNLibParseError("unexpected end of s-expression")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            lst = []
            while tokens[pos] != ")":
                lst.append(read_from())
            pos += 1  # consume ')'
            return lst
        if tok == ")":
            raise VNNLibParseError("unexpected ')'")
        return tok

    tree = read_from()
    return tree


# ---------------------------------------------------------------------------
# Network / variable declarations
# ---------------------------------------------------------------------------

def _parse_shape(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.replace(",", " ").split())


def _flat_size(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _ravel(idx: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    """Row-major flat index of a multi-index, with single-index passthrough."""
    if len(idx) == len(shape):
        flat = 0
        for i, ix in enumerate(idx):
            flat = flat * shape[i] + ix
        return flat
    if len(idx) == 1:
        return idx[0]
    raise VNNLibParseError(
        f"index {idx} rank does not match declared shape {shape}"
    )


class _Resolver:
    """Maps a variable atom (``X[0,0,0,3]``, ``A[1]``, bare ``Y``) to
    ``('in'|'out', joint_flat_index)``.

    A network may declare several input (or output) tensors; they occupy
    a single joint index space per kind, concatenated in declaration
    order (row-major within each tensor).
    """

    def __init__(self, inputs, outputs):
        # inputs/outputs: list of (name, shape) in declaration order.
        self._tensors: Dict[str, Tuple[str, int, Tuple[int, ...]]] = {}
        self.input_tensors = []
        self.output_tensors = []
        sizes = {"in": 0, "out": 0}
        for kind, decls, meta in (("in", inputs, self.input_tensors),
                                  ("out", outputs, self.output_tensors)):
            for name, shape in decls:
                size = _flat_size(shape)
                self._tensors[name] = (kind, sizes[kind], shape)
                meta.append({"name": name, "shape": shape,
                             "offset": sizes[kind], "size": size})
                sizes[kind] += size
        self.in_size = sizes["in"]
        self.out_size = sizes["out"]
        self._var = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[([0-9,\s]*)\]$")

    def is_var(self, atom: str) -> bool:
        if not isinstance(atom, str):
            return False
        return bool(self._var.match(atom)) or atom in self._tensors

    def resolve(self, atom: str) -> Tuple[str, int]:
        # Bare tensor reference (rank-0 / scalar declarations, e.g.
        # ``declare-output Y float32 []`` referenced as ``Y``).
        if atom in self._tensors:
            kind, offset, shape = self._tensors[atom]
            if _flat_size(shape) != 1:
                raise VNNLibParseError(
                    f"bare reference {atom!r} to non-scalar tensor "
                    f"(shape {shape})"
                )
            return (kind, offset)
        m = self._var.match(atom)
        if not m:
            raise VNNLibParseError(f"not a variable reference: {atom!r}")
        name, idx_s = m.group(1), m.group(2)
        if name not in self._tensors:
            raise VNNLibParseError(f"unknown variable network/name in {atom!r}")
        kind, offset, shape = self._tensors[name]
        idx = tuple(int(x) for x in idx_s.replace(",", " ").split())
        return (kind, offset + _ravel(idx, shape))


def _parse_declarations(text: str) -> _Resolver:
    ins = re.findall(
        r"\(declare-input\s+(\S+)\s+\S+\s+\[([^\]]*)\]", text)
    outs = re.findall(
        r"\(declare-output\s+(\S+)\s+\S+\s+\[([^\]]*)\]", text)
    if not ins or not outs:
        raise VNNLibParseError(
            "could not find declare-input/declare-output declarations"
        )
    return _Resolver(
        [(name, _parse_shape(shape)) for name, shape in ins],
        [(name, _parse_shape(shape)) for name, shape in outs],
    )


def _parse_relational_declarations(text: str):
    """Parse N ``declare-network`` blocks into a joint resolver + per-
    network metadata (name, relation, joint offsets/sizes). All networks'
    tensors share one joint index space per kind, concatenated in
    declaration order."""
    networks = []
    inputs, outputs = [], []
    for block in _extract_sexprs(text, "(declare-network"):
        m = re.match(r"\(declare-network\s+(\S+)", block)
        name = m.group(1)
        rel = re.search(r"\((equal-to|isomorphic-to)\s+(\S+?)\s*\)", block)
        ins = [(n, _parse_shape(s)) for n, s in re.findall(
            r"\(declare-input\s+(\S+)\s+\S+\s+\[([^\]]*)\]", block)]
        outs = [(n, _parse_shape(s)) for n, s in re.findall(
            r"\(declare-output\s+(\S+)\s+\S+\s+\[([^\]]*)\]", block)]
        if not ins or not outs:
            raise VNNLibParseError(
                f"network {name!r} lacks input/output declarations")
        networks.append({
            "name": name,
            "relation": (rel.group(1), rel.group(2)) if rel else None,
            "input_offset": sum(_flat_size(s) for _, s in inputs),
            "input_size": sum(_flat_size(s) for _, s in ins),
            "output_offset": sum(_flat_size(s) for _, s in outputs),
            "output_size": sum(_flat_size(s) for _, s in outs),
        })
        inputs.extend(ins)
        outputs.extend(outs)
    return _Resolver(inputs, outputs), networks


def _coupling_rows(op, coeffs, const, in_size: int):
    """Lower a multi-variable INPUT comparison to rows over the joint
    input space (``row . x <= rhs``). ``==`` produces both directions."""
    dense = np.zeros(in_size, dtype=np.float64)
    for (kind, idx), v in coeffs.items():
        dense[idx] = v
    if op in ("<=", "<"):
        return [(dense, -const)]
    if op in (">=", ">"):
        return [(-dense, const)]
    if op == "==":
        return [(dense, -const), (-dense, const)]
    raise VNNLibParseError(f"unsupported coupling operator {op!r}")


def _resolve_ast(node, resolver: _Resolver):
    """Resolve a formula/arith tree into a faithful nested-tuple AST:
    ``('and'|'or', subtrees...)``, ``(op, lhs, rhs)`` for comparisons
    (infix normalized to prefix), arithmetic ``('+','-','*','/', ...)``,
    leaves ``('var', 'in'|'out', flat_index)`` and ``('const', float)``."""
    if isinstance(node, str):
        if resolver.is_var(node):
            kind, idx = resolver.resolve(node)
            return ("var", kind, idx)
        try:
            return ("const", float(node))
        except ValueError:
            raise VNNLibParseError(f"unrecognized atom: {node!r}")
    if not node:
        raise VNNLibParseError("empty expression")
    op = node[0]
    if isinstance(op, str) and op in ("and", "or"):
        return (op,) + tuple(_resolve_ast(c, resolver) for c in node[1:])
    if isinstance(op, str) and op in _COMPARE | _ARITH:
        return (op,) + tuple(_resolve_ast(c, resolver) for c in node[1:])
    # infix comparison: (a op b)
    if len(node) == 3 and isinstance(node[1], str) and node[1] in _COMPARE:
        return (node[1], _resolve_ast(node[0], resolver),
                _resolve_ast(node[2], resolver))
    raise VNNLibParseError(f"unsupported clause: {node!r}")


def evaluate_ast(node, x, y):
    """Evaluate a resolved AST at concrete input/output vectors."""
    op = node[0]
    if op == "var":
        return float(x[node[2]] if node[1] == "in" else y[node[2]])
    if op == "const":
        return node[1]
    if op == "and":
        return all(evaluate_ast(c, x, y) for c in node[1:])
    if op == "or":
        return any(evaluate_ast(c, x, y) for c in node[1:])
    args = [evaluate_ast(c, x, y) for c in node[1:]]
    if op == "+":
        return sum(args)
    if op == "-":
        return -args[0] if len(args) == 1 else args[0] - sum(args[1:])
    if op == "*":
        r = 1.0
        for a in args:
            r *= a
        return r
    if op == "/":
        return args[0] / args[1]
    return {"<=": args[0] <= args[1], "<": args[0] < args[1],
            ">=": args[0] >= args[1], ">": args[0] > args[1],
            "==": args[0] == args[1], "!=": args[0] != args[1]}[op]


def evaluate_nonlinear(result: Dict, x, y) -> bool:
    """Does (x, y) satisfy a ``format='nonlinear'`` spec (i.e. witness a
    violation)? The assertion conjunction, evaluated on the faithful AST."""
    return all(evaluate_ast(a, x, y) for a in result["assertions"])


def _load_nonlinear(text: str, resolver: _Resolver) -> Dict:
    """Faithful loading for specs outside the linear fragment: every
    assert as a resolved AST (the authoritative representation), plus a
    best-effort input box from the single-variable affine atoms (a
    convenience for samplers; NOT a complete encoding of the spec)."""
    assertions = []
    lb = np.full(resolver.in_size, -np.inf, dtype=np.float64)
    ub = np.full(resolver.in_size, np.inf, dtype=np.float64)

    for sexpr in _extract_sexprs(text, "(assert"):
        tree = _read(sexpr)
        if not (isinstance(tree, list) and tree and tree[0] == "assert"):
            continue
        formula = tree[1]
        assertions.append(_resolve_ast(formula, resolver))

        # best-effort box: top-level conjunctions of single-input-var
        # affine comparisons only
        stack = [formula]
        while stack:
            node = stack.pop()
            if isinstance(node, list) and node and node[0] == "and":
                stack.extend(node[1:])
                continue
            try:
                cmp = _parse_compare(node, resolver)
                if cmp is None:
                    continue
                op, coeffs, const = cmp
                in_vars = [k for k in coeffs if k[0] == "in"]
                if len(in_vars) == 1 and len(coeffs) == 1:
                    _apply_bounds(lb, ub,
                                  [_atomic_input_bound(op, coeffs, const)])
            except VNNLibParseError:
                continue  # nonlinear / mixed atom: AST is authoritative

    if not assertions:
        raise VNNLibParseError("no assertions found in nonlinear spec")
    return {
        "format": "nonlinear",
        "assertions": assertions,
        "lb": lb, "ub": ub,
        "prop": None,
        "input_tensors": resolver.input_tensors,
        "output_tensors": resolver.output_tensors,
    }


def _load_relational(text: str, resolver: _Resolver, networks) -> Dict:
    """Lower a multi-network spec: joint box + input-coupling HalfSpace +
    output property over the joint output space, all as asserted."""
    lb = np.full(resolver.in_size, -np.inf, dtype=np.float64)
    ub = np.full(resolver.in_size, np.inf, dtype=np.float64)
    coupling = []  # (dense row, rhs)
    out_groups: List[Dict] = []

    for sexpr in _extract_sexprs(text, "(assert"):
        tree = _read(sexpr)
        if not (isinstance(tree, list) and tree and tree[0] == "assert"):
            continue
        formula = tree[1]
        roles: set = set()
        _collect_roles(formula, resolver, roles)

        if roles == {"out"}:
            dnf = _to_dnf(formula, resolver)
            halfspaces = _dnf_to_halfspaces(dnf, resolver.out_size)
            hg = halfspaces[0] if len(halfspaces) == 1 else halfspaces
            out_groups.append({"Hg": hg})
            continue
        if "out" in roles:
            raise VNNLibParseError(
                "relational specs with combined input/output asserts are "
                "not supported"
            )

        # input-only: conjunction of box bounds and/or coupling rows
        dnf = _to_dnf_comparisons(formula, resolver)
        if len(dnf) != 1:
            raise VNNLibParseError(
                "disjunctive input asserts are not supported in relational "
                "specs"
            )
        for (op, coeffs, const) in dnf[0]:
            in_vars = [k for k in coeffs if k[0] == "in"]
            if len(in_vars) == 1:
                _apply_bounds(lb, ub,
                              [_atomic_input_bound(op, coeffs, const)])
            else:
                coupling.extend(
                    _coupling_rows(op, coeffs, const, resolver.in_size))

    if not out_groups:
        raise VNNLibParseError(
            "no output property parsed in relational spec")

    # Every joint input dim must be constrained: directly boxed, or
    # appearing in some coupling row (e.g. bounded via equality with a
    # boxed dim). Anything else is an unconstrained verification input.
    coupled_dims = set()
    for row, _ in coupling:
        coupled_dims.update(np.nonzero(row)[0].tolist())
    for i in range(resolver.in_size):
        if not (np.isfinite(lb[i]) or np.isfinite(ub[i])) \
                and i not in coupled_dims:
            raise VNNLibParseError(
                f"joint input dim {i} is neither bounded nor coupled")

    if coupling:
        G = np.stack([r for r, _ in coupling])
        g = np.array([[rhs] for _, rhs in coupling], dtype=np.float64)
        coupling_hs = HalfSpace(G, g)
    else:
        coupling_hs = None

    return {
        "format": "relational",
        "networks": networks,
        "lb": lb, "ub": ub,
        "input_coupling": coupling_hs,
        "prop": out_groups,
        "input_tensors": resolver.input_tensors,
        "output_tensors": resolver.output_tensors,
    }


# ---------------------------------------------------------------------------
# Affine-expression and comparison parsing
# ---------------------------------------------------------------------------

Affine = Tuple[Dict[Tuple[str, int], float], float]  # (coeffs, const)


def _aff_add(a: Affine, b: Affine, sign: float = 1.0) -> Affine:
    coeffs = dict(a[0])
    for k, v in b[0].items():
        coeffs[k] = coeffs.get(k, 0.0) + sign * v
    return coeffs, a[1] + sign * b[1]


def _parse_affine(node, resolver: _Resolver) -> Affine:
    """Parse a (necessarily affine) arithmetic expression. Raise on nonlinear."""
    if isinstance(node, list):
        if not node:
            raise VNNLibParseError("empty expression")
        op = node[0]
        args = node[1:]
        if op == "+":
            acc: Affine = ({}, 0.0)
            for a in args:
                acc = _aff_add(acc, _parse_affine(a, resolver))
            return acc
        if op == "-":
            first = _parse_affine(args[0], resolver)
            if len(args) == 1:
                return {k: -v for k, v in first[0].items()}, -first[1]
            acc = first
            for a in args[1:]:
                acc = _aff_add(acc, _parse_affine(a, resolver), sign=-1.0)
            return acc
        if op == "*":
            const = 1.0
            var_factor: Optional[Affine] = None
            for a in args:
                ca = _parse_affine(a, resolver)
                if ca[0]:  # has variables
                    if var_factor is not None:
                        raise VNNLibParseError(
                            "nonlinear term: product of two variable "
                            "expressions"
                        )
                    var_factor = ca
                else:
                    const *= ca[1]
            if var_factor is None:
                return {}, const
            return {k: v * const for k, v in var_factor[0].items()}, \
                var_factor[1] * const
        if op == "/":
            num = _parse_affine(args[0], resolver)
            den = _parse_affine(args[1], resolver)
            if den[0]:
                raise VNNLibParseError("nonlinear term: division by a variable")
            d = den[1]
            return {k: v / d for k, v in num[0].items()}, num[1] / d
        raise VNNLibParseError(f"unsupported operator in expression: {op!r}")
    # atom
    if resolver.is_var(node):
        return {resolver.resolve(node): 1.0}, 0.0
    try:
        return {}, float(node)
    except ValueError:
        raise VNNLibParseError(f"unrecognized atom: {node!r}")


def _parse_compare(node, resolver: _Resolver):
    """Return ``(op, coeffs, const)`` for ``lhs - rhs`` or ``None`` if not a
    comparison. Accepts both prefix ``(<= a b)`` and infix ``(a < b)``."""
    if not isinstance(node, list) or len(node) != 3:
        return None
    if isinstance(node[0], str) and node[0] in _COMPARE:
        op, a, b = node[0], node[1], node[2]
    elif isinstance(node[1], str) and node[1] in _COMPARE:
        a, op, b = node[0], node[1], node[2]
    else:
        return None
    ca = _parse_affine(a, resolver)
    cb = _parse_affine(b, resolver)
    coeffs, const = _aff_add(ca, cb, sign=-1.0)  # lhs - rhs
    coeffs = {k: v for k, v in coeffs.items() if v != 0.0}
    return op, coeffs, const


# ---------------------------------------------------------------------------
# Output property: comparison -> rows, formula -> DNF -> HalfSpaces
# ---------------------------------------------------------------------------

Row = Tuple[Dict[int, float], float]  # (out-index -> coeff, rhs) : row . y <= rhs


def _compare_to_rows(op, coeffs, const) -> List[Row]:
    """Canonicalize an output comparison into rows of ``row . y <= rhs``."""
    if any(k[0] == "in" for k in coeffs):
        raise VNNLibParseError(
            "constraint mixes input and output variables (combined / "
            "relational form not supported)"
        )
    out = {k[1]: v for k, v in coeffs.items()}
    if op in ("<=", "<"):
        return [(out, -const)]
    if op in (">=", ">"):
        return [({i: -v for i, v in out.items()}, const)]
    if op == "==":
        return [(out, -const), ({i: -v for i, v in out.items()}, const)]
    raise VNNLibParseError(
        "'!=' in the unsafe region is non-convex and not supported"
    )


def _to_dnf(node, resolver: _Resolver) -> List[List[Row]]:
    """Disjunctive normal form: a list of disjuncts, each a list of rows (AND)."""
    cmp = _parse_compare(node, resolver)
    if cmp is not None:
        return [_compare_to_rows(*cmp)]
    if isinstance(node, list) and node:
        op = node[0]
        if op == "and":
            acc: List[List[Row]] = [[]]
            for child in node[1:]:
                child_dnf = _to_dnf(child, resolver)
                acc = [d + cd for d in acc for cd in child_dnf]
            return acc
        if op == "or":
            res: List[List[Row]] = []
            for child in node[1:]:
                res.extend(_to_dnf(child, resolver))
            return res
    raise VNNLibParseError(f"unsupported output clause: {node!r}")


def _dnf_to_halfspaces(dnf: List[List[Row]], out_size: int) -> List[HalfSpace]:
    halfspaces = []
    for disjunct in dnf:
        G = np.zeros((len(disjunct), out_size), dtype=np.float64)
        g = np.zeros((len(disjunct), 1), dtype=np.float64)
        for r, (coeffs, rhs) in enumerate(disjunct):
            for idx, v in coeffs.items():
                G[r, idx] = v
            g[r, 0] = rhs
        halfspaces.append(HalfSpace(G, g))
    return halfspaces


# ---------------------------------------------------------------------------
# Input box: comparison -> single-variable bound
# ---------------------------------------------------------------------------

def _atomic_input_bound(op, coeffs, const) -> Tuple[int, str, float]:
    """Reduce a single-variable input comparison to ``(idx, kind, value)``."""
    if any(k[0] == "out" for k in coeffs):
        raise VNNLibParseError(
            "constraint mixes input and output variables (combined form "
            "not supported)"
        )
    in_keys = [k for k in coeffs if k[0] == "in"]
    if len(in_keys) != 1:
        raise VNNLibParseError(
            "input constraint is not a single-variable box bound "
            f"(references {len(in_keys)} input variables)"
        )
    key = in_keys[0]
    c = coeffs[key]
    val = -const / c
    if op == "==":
        kind = "both"
    elif op in ("<=", "<"):
        kind = "upper" if c > 0 else "lower"
    else:  # >=, >
        kind = "lower" if c > 0 else "upper"
    return key[1], kind, val


def _conj_bounds(node, resolver: _Resolver) -> List[Tuple[int, str, float]]:
    """Collect single-variable bounds from a conjunction (and / atoms)."""
    bounds: List[Tuple[int, str, float]] = []

    def rec(n):
        if isinstance(n, list) and n and n[0] == "and":
            for c in n[1:]:
                rec(c)
            return
        cmp = _parse_compare(n, resolver)
        if cmp is None:
            raise VNNLibParseError(f"unsupported input clause: {n!r}")
        bounds.append(_atomic_input_bound(*cmp))

    rec(node)
    return bounds


def _apply_bounds(lb, ub, bounds):
    for idx, kind, val in bounds:
        if kind == "upper":
            ub[idx] = min(ub[idx], val)
        elif kind == "lower":
            lb[idx] = max(lb[idx], val)
        else:  # both (equality)
            lb[idx] = val
            ub[idx] = val


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _collect_roles(node, resolver: _Resolver, roles: set):
    if isinstance(node, list):
        for c in node:
            _collect_roles(c, resolver, roles)
    elif isinstance(node, str) and resolver.is_var(node):
        roles.add(resolver.resolve(node)[0])


def _to_dnf_comparisons(node, resolver: _Resolver):
    """DNF over raw comparisons: list of disjuncts, each a list of
    ``(op, coeffs, const)`` tuples. Unlike :func:`_to_dnf`, atoms are not
    yet lowered to output rows — combined input/output asserts need to
    route each atom to either the region bounds or the output polytope."""
    cmp = _parse_compare(node, resolver)
    if cmp is not None:
        return [[cmp]]
    if isinstance(node, list) and node:
        op = node[0]
        if op == "and":
            acc = [[]]
            for child in node[1:]:
                child_dnf = _to_dnf_comparisons(child, resolver)
                acc = [d + cd for d in acc for cd in child_dnf]
            return acc
        if op == "or":
            res = []
            for child in node[1:]:
                res.extend(_to_dnf_comparisons(child, resolver))
            return res
    raise VNNLibParseError(f"unsupported clause in combined assert: {node!r}")


def _rows_to_halfspace(rows, out_size: int) -> HalfSpace:
    """Stack ``(coeffs, rhs)`` output rows into one AND-of-rows HalfSpace.
    No rows -> trivially-true ``0 . y <= 0`` (input-only disjunct)."""
    if not rows:
        return HalfSpace(np.zeros((1, out_size), dtype=np.float64),
                         np.zeros((1, 1), dtype=np.float64))
    G = np.zeros((len(rows), out_size), dtype=np.float64)
    g = np.zeros((len(rows), 1), dtype=np.float64)
    for r, (coeffs, rhs) in enumerate(rows):
        for idx, v in coeffs.items():
            G[r, idx] = v
        g[r, 0] = rhs
    return HalfSpace(G, g)


def _lower_combined(formula, resolver: _Resolver, base_lb, base_ub,
                    out_groups: List[Dict]) -> Dict:
    """Lower one combined input/output assert into per-disjunct
    (region, prop) pairs, conjoined with any global output asserts."""
    lb_list, ub_list, prop_list, pairs = [], [], [], []
    for disjunct in _to_dnf_comparisons(formula, resolver):
        lb_r = base_lb.copy()
        ub_r = base_ub.copy()
        rows = []
        for (op, coeffs, const) in disjunct:
            kinds = {k[0] for k in coeffs}
            if kinds == {"in"}:
                _apply_bounds(lb_r, ub_r,
                              [_atomic_input_bound(op, coeffs, const)])
            elif kinds == {"out"}:
                rows.extend(_compare_to_rows(op, coeffs, const))
            elif not kinds:
                raise VNNLibParseError(
                    f"constant comparison in combined assert: {disjunct!r}")
            else:
                raise VNNLibParseError(
                    "a single constraint mixes input and output variables"
                )
        group = {"Hg": _rows_to_halfspace(rows, resolver.out_size)}
        pairs.append({"lb": lb_r, "ub": ub_r, "prop": [group] + out_groups})
        lb_list.append(lb_r)
        ub_list.append(ub_r)
        prop_list.append(group)
    return {"lb": lb_list, "ub": ub_list, "prop": prop_list,
            "paired": True, "pairs": pairs}


def load_vnnlib_v2(property_file: str, text: Optional[str] = None) -> Dict:
    """Parse a VNNLIB 2.0 property file into ``{'lb', 'ub', 'prop'}``.

    Produces the same structure as the 1.0 parser:
      - ``lb`` / ``ub``: ``np.ndarray`` (single input region) or a list of
        arrays (disjunctive input regions).
      - ``prop``: list of ``{'Hg': HalfSpace | list[HalfSpace]}`` groups.

    Raises :class:`VNNLibParseError` for specs outside the supported
    single-network linear fragment.
    """
    if text is None:
        with open(property_file, "r") as f:
            text = f.read()
    text = _strip_comments(text)

    if len(re.findall(r"\(declare-network\b", text)) > 1:
        resolver, networks = _parse_relational_declarations(text)
        return _load_relational(text, resolver, networks)

    resolver = _parse_declarations(text)
    try:
        return _lower_linear(text, resolver)
    except VNNLibParseError as e:
        # Outside the linear fragment (nonlinear terms / non-convex !=):
        # load faithfully as resolved ASTs instead of rejecting. Other
        # parse errors (unknown atoms, malformed clauses) stay loud.
        if "nonlinear" in str(e) or "non-convex" in str(e):
            return _load_nonlinear(text, resolver)
        raise


def _lower_linear(text: str, resolver: _Resolver) -> Dict:
    """The single-network linear lowering (box/pairs/halfspaces)."""
    base_lb = np.full(resolver.in_size, -np.inf, dtype=np.float64)
    base_ub = np.full(resolver.in_size, np.inf, dtype=np.float64)
    or_regions: Optional[List[List[Tuple[int, str, float]]]] = None
    out_groups: List[Dict] = []
    combined: List = []

    for sexpr in _extract_sexprs(text, "(assert"):
        tree = _read(sexpr)
        if not (isinstance(tree, list) and tree and tree[0] == "assert"):
            continue
        if len(tree) < 2:
            raise VNNLibParseError(f"empty assert: {sexpr!r}")
        formula = tree[1]

        roles: set = set()
        _collect_roles(formula, resolver, roles)

        if "out" in roles and "in" in roles:
            combined.append(formula)
        elif "out" in roles:
            dnf = _to_dnf(formula, resolver)
            halfspaces = _dnf_to_halfspaces(dnf, resolver.out_size)
            hg = halfspaces[0] if len(halfspaces) == 1 else halfspaces
            out_groups.append({"Hg": hg})
        else:
            if isinstance(formula, list) and formula and formula[0] == "or":
                if or_regions is not None:
                    raise VNNLibParseError(
                        "more than one disjunctive (or) input assert is not "
                        "supported"
                    )
                or_regions = [_conj_bounds(c, resolver) for c in formula[1:]]
            else:
                _apply_bounds(base_lb, base_ub, _conj_bounds(formula, resolver))

    if combined:
        # Combined input/output assert(s): each DNF disjunct pairs an
        # input region with its own output constraint.
        if len(combined) > 1:
            raise VNNLibParseError(
                "multiple combined input/output asserts are not supported"
            )
        if or_regions is not None:
            raise VNNLibParseError(
                "a combined input/output assert together with a separate "
                "disjunctive input assert is not supported"
            )
        result = _lower_combined(combined[0], resolver, base_lb, base_ub,
                                 out_groups)
        result["input_tensors"] = resolver.input_tensors
        result["output_tensors"] = resolver.output_tensors
        return result

    if not out_groups:
        raise VNNLibParseError(
            "no output property parsed (no assert references an output "
            "variable)"
        )

    if or_regions is not None:
        lb_list, ub_list = [], []
        for region in or_regions:
            lb_r = base_lb.copy()
            ub_r = base_ub.copy()
            _apply_bounds(lb_r, ub_r, region)
            lb_list.append(lb_r)
            ub_list.append(ub_r)
        lb_out, ub_out = lb_list, ub_list
    else:
        lb_out, ub_out = base_lb, base_ub

    return {"lb": lb_out, "ub": ub_out, "prop": out_groups,
            "input_tensors": resolver.input_tensors,
            "output_tensors": resolver.output_tensors}
