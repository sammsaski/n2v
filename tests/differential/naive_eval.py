"""A deliberately naive, independent VNNLIB evaluator — the trusted base.

Given the raw spec text and concrete values for every variable atom, this
decides whether the conjunction of asserts holds. It knows NOTHING about
n2v's parser: variables are looked up by their literal spelling in the
file (``X_0``, ``X[0,3]``, ``X_f[2]``, bare ``Y``), and the s-expression
is evaluated directly. ~60 lines — read it once, trust it forever.

Used by the point-agreement tests: n2v's parsed representation and this
evaluator must agree, on every sampled point, for every spec file.
"""


def _tokenize(text):
    out = []
    for line in text.splitlines():
        i = line.find(";")
        if i >= 0:
            line = line[:i]
        out.append(line)
    return " ".join(out).replace("(", " ( ").replace(")", " ) ").split()


def _read_all(tokens):
    """Parse a token stream into a list of top-level s-expressions."""
    pos = 0

    def read():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            lst = []
            while tokens[pos] != ")":
                lst.append(read())
            pos += 1
            return lst
        return tok

    exprs = []
    while pos < len(tokens):
        exprs.append(read())
    return exprs


def _precompile(node):
    """One-time pass: numeric literal strings -> floats (so evaluation is
    pure lookups/arithmetic; matters for the 100MB+ specs). List heads
    are operators and stay untouched."""
    if isinstance(node, list):
        return [node[0]] + [_precompile(c) for c in node[1:]]
    try:
        return float(node)
    except (TypeError, ValueError):
        return node


def _arith(node, values):
    """Evaluate an arithmetic expression to a float."""
    if isinstance(node, float):
        return node  # precompiled literal
    if isinstance(node, str):
        if node in values:
            return float(values[node])
        return float(node)  # numeric literal; unknown atoms raise here
    op, args = node[0], [_arith(a, values) for a in node[1:]]
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
    raise ValueError(f"unknown arithmetic operator {op!r}")


def _formula(node, values):
    """Evaluate a boolean formula to True/False."""
    op = node[0]
    if op == "and":
        return all(_formula(c, values) for c in node[1:])
    if op == "or":
        return any(_formula(c, values) for c in node[1:])
    # comparison: prefix (op a b) or infix (a op b)
    if op in ("<=", "<", ">=", ">", "==", "!="):
        a, b = _arith(node[1], values), _arith(node[2], values)
    elif len(node) == 3 and node[1] in ("<=", "<", ">=", ">", "==", "!="):
        op = node[1]
        a, b = _arith(node[0], values), _arith(node[2], values)
    else:
        raise ValueError(f"unknown formula {node!r}")
    return {"<=": a <= b, "<": a < b, ">=": a >= b, ">": a > b,
            "==": a == b, "!=": a != b}[op]


def _iter_assert_blocks(text):
    """Yield each balanced ``(assert ...)`` substring. Linear scan with a
    paren counter — avoids tokenizing 100MB+ files into one giant list."""
    i = 0
    while True:
        j = text.find("(assert", i)
        if j < 0:
            return
        depth, k = 0, j
        while k < len(text):
            ch = text[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        yield text[j:k + 1]
        i = k + 1


class NaiveSpec:
    """Parse once, evaluate many (the monster specs are 100+ MB):
    asserts are extracted and tokenized one at a time."""

    def __init__(self, text):
        # comments are stripped block-wise to keep memory bounded
        self.asserts = []
        for block in _iter_assert_blocks(text):
            exprs = _read_all(_tokenize(block))
            for e in exprs:
                if isinstance(e, list) and e and e[0] == "assert":
                    self.asserts.append(_precompile(e[1]))

    def satisfied(self, values):
        return all(_formula(a, values) for a in self.asserts)

    def variable_atoms(self):
        """Distinct variable spellings across the (precompiled) asserts."""
        ops = {"and", "or", "<=", "<", ">=", ">", "==", "!=",
               "+", "-", "*", "/"}
        found, seen = [], set()

        def walk(node):
            if isinstance(node, list):
                for c in node:
                    walk(c)
            elif isinstance(node, str) and node not in ops \
                    and node not in seen:
                seen.add(node)
                found.append(node)

        for a in self.asserts:
            walk(a)
        return found


def satisfied(text, values):
    """True iff the conjunction of all (assert ...) statements holds under
    ``values`` (dict: variable atom spelling -> number). Declarations and
    version markers are ignored — only asserts have runtime meaning."""
    return NaiveSpec(text).satisfied(values)


def variable_atoms(text):
    """All distinct variable spellings appearing inside asserts (atoms that
    are not numbers and not operators)."""
    ops = {"assert", "and", "or", "<=", "<", ">=", ">", "==", "!=",
           "+", "-", "*", "/"}
    found = []

    def walk(node):
        if isinstance(node, list):
            for c in node:
                walk(c)
            return
        if node in ops:
            return
        try:
            float(node)
        except ValueError:
            if node not in found:
                found.append(node)

    for expr in _read_all(_tokenize(text)):
        if isinstance(expr, list) and expr and expr[0] == "assert":
            walk(expr[1])
    return found
