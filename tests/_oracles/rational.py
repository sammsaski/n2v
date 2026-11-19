"""The exact image of a polytope under a piecewise-linear network, and exact
membership in a computed set, in ``fractions.Fraction`` (PROTOCOL.md 3.1).

A piecewise-linear network on a bounded polytope is affine on each of finitely
many pieces. A :class:`Piece` is one of them: the part ``A x <= b`` of the
input set and the map ``x -> W x + c`` the network is there. The image of the
piece is the convex hull of the images of its vertices, so the exact reach set
is a finite union of vertex-described polytopes, with no rounding anywhere.

    pieces = relu(affine(box([-1, -1], [1, 1]), W1, c1))
    pieces = affine(pieces, W2, c2)
    assert escapes(pieces, star(computed)) == []

A convex set that holds the vertices of a polytope holds the polytope. So the
computed sets contain the exact image when :func:`escapes` and
:func:`straddlers` are both empty: every piece then has all its vertices in
one computed set. A piece whose vertices are spread over several computed sets
proves nothing; cut it with :func:`split` along the boundary between them. The
other direction, for a method that claims the exact image, is
:func:`unreached` on the vertices of each computed set (:func:`set_vertices`).

Nothing here calls n2v. Vertices come from brute-force enumeration over row
subsets solved exactly, which is enough for the two to four variables of a
test. Membership of a rational point in ``{c + G a : C a <= d, lb <= a <= ub}``
is decided by phase one of the simplex method in exact arithmetic
(:func:`decide`). Either answer is checked before it is returned: a member
has a predicate point that satisfies every row, a non-member has multipliers
that prove no such point exists (:func:`is_refutation`). :class:`Membership`
answers the same question for many points of one set and reuses those
certificates, each checked again for the point at hand.
:func:`star` reads a computed set's stored arrays through
``membership.rows`` and converts every float to the rational it is.

Discontinuous operations (Sign, Round, Floor, Ceil) have branches that are
open at an end. A piece records those ends as strict rows, so
:func:`evaluate` is the function itself at a jump, and a region that only
touches a branch at its open end is dropped. The vertices of a piece are those
of its closed region, so the union of the piece images is exactly the closure
of the image. A computed set is closed, so it contains the image exactly when
it contains the closure.
"""

import math
from fractions import Fraction
from itertools import combinations, product


# --------------------------------------------------------------------------- #
# Exact linear algebra                                                          #
# --------------------------------------------------------------------------- #
def frac(x):
    """The rational a number is: exact for ints, floats and numpy scalars."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, str)):
        return Fraction(x)
    return Fraction(float(x))


def vector(values):
    return [frac(v) for v in values]


def matrix(rows):
    return [vector(row) for row in rows]


def dot(a, x):
    return sum((ai * xi for ai, xi in zip(a, x)), Fraction(0))


def solve(A, b):
    """The solution of the square system ``A x = b``, or None when ``A`` is
    singular."""
    n = len(A)
    M = [list(row) + [rhs] for row, rhs in zip(A, b)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if M[r][col] != 0), None)
        if pivot is None:
            return None
        M[col], M[pivot] = M[pivot], M[col]
        M[col] = [v / M[col][col] for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [v - factor * p for v, p in zip(M[r], M[col])]
    return [M[r][n] for r in range(n)]


def rank(rows):
    """Rank of a list of rational vectors."""
    M = [list(row) for row in rows]
    r = 0
    for col in range(len(M[0]) if M else 0):
        pivot = next((i for i in range(r, len(M)) if M[i][col] != 0), None)
        if pivot is None:
            continue
        M[r], M[pivot] = M[pivot], M[r]
        for i in range(r + 1, len(M)):
            if M[i][col] != 0:
                factor = M[i][col] / M[r][col]
                M[i] = [v - factor * p for v, p in zip(M[i], M[r])]
        r += 1
    return r


def vertices(A, b):
    """The vertices of ``{x : A x <= b}``, sorted; an empty list when the set
    is empty. Every choice of n rows is solved as equalities and kept when it
    satisfies the rest. The set must be bounded (:func:`is_bounded`): an
    unbounded set is not the hull of its vertices."""
    if not A:
        raise ValueError("a polytope needs at least one row")
    n = len(A[0])
    found = set()
    for subset in combinations(range(len(A)), n):
        x = solve([A[i] for i in subset], [b[i] for i in subset])
        if x is not None and all(dot(row, x) <= rhs for row, rhs in zip(A, b)):
            found.add(tuple(x))
    return sorted(found)


def is_bounded(A):
    """Is ``{x : A x <= b}`` bounded for every ``b``? Its recession cone
    ``{r : A r <= 0}`` is the origin alone exactly when the cone cut by the
    unit box has no other vertex."""
    n = len(A[0])
    unit = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    cone = list(A) + unit + [[-v for v in row] for row in unit]
    rhs = [Fraction(0)] * len(A) + [Fraction(1)] * (2 * n)
    return vertices(cone, rhs) == [tuple([Fraction(0)] * n)]


def feasible(A_eq=(), b_eq=(), A_ub=(), b_ub=()):
    """A point ``x`` with ``A_eq x = b_eq`` and ``A_ub x <= b_ub``, or None
    when there is none. The variables are free. Exact; see :func:`decide`."""
    return decide(A_eq, b_eq, A_ub, b_ub)[0]


def decide(A_eq=(), b_eq=(), A_ub=(), b_ub=()):
    """``(x, None)`` for a point ``x`` with ``A_eq x = b_eq`` and
    ``A_ub x <= b_ub``, or ``(None, (u, v))`` for a proof that there is none:
    multipliers that :func:`is_refutation` accepts.

    Phase one of the simplex method on ``x = p - q`` with slack variables, in
    Fractions, with Bland's rule (lowest index enters, lowest index leaves), so
    it terminates. Either answer is checked against the rows before it is
    returned, so a wrong pivot cannot produce a wrong answer, only an
    ``ArithmeticError``."""
    A_eq, A_ub = [list(r) for r in A_eq], [list(r) for r in A_ub]
    b_eq, b_ub = list(b_eq), list(b_ub)
    zero, one = Fraction(0), Fraction(1)
    if not A_eq and not A_ub:
        return [], None
    n = len((A_eq or A_ub)[0])
    if n == 0:
        for i, r in enumerate(b_eq):
            if r != 0:
                u = [zero] * len(b_eq)
                u[i] = -one if r > 0 else one
                return None, (u, [zero] * len(b_ub))
        for k, r in enumerate(b_ub):
            if r < 0:
                v = [zero] * len(b_ub)
                v[k] = one
                return None, ([zero] * len(b_eq), v)
        return [], None
    # Columns: p, q, one slack per inequality, one artificial per equality.
    # The artificial of an inequality whose slack cannot start in the basis is
    # not stored: nothing reads it.
    slack0 = 2 * n
    art0 = slack0 + len(A_ub)
    width = art0 + len(A_eq)
    tableau, basis, sign = [], [], []
    unstored = width
    for i, (a, r) in enumerate(zip(A_eq, b_eq)):
        row = list(a) + [-v for v in a] + [zero] * (width - 2 * n)
        if r < 0:
            row, r = [-v for v in row], -r
            sign.append(-one)
        else:
            sign.append(one)
        row[art0 + i] = one
        basis.append(art0 + i)
        tableau.append(row + [r])
    for k, (a, r) in enumerate(zip(A_ub, b_ub)):
        row = list(a) + [-v for v in a] + [zero] * (width - 2 * n)
        row[slack0 + k] = one
        if r < 0:
            row, r = [-v for v in row], -r
            basis.append(unstored)
            unstored += 1
        else:
            basis.append(slack0 + k)
        tableau.append(row + [r])
    # Reduced costs of "minimise the sum of the artificial variables".
    cost = [(one if art0 <= j < width else zero)
            - sum((tableau[i][j] for i in range(len(tableau))
                   if basis[i] >= art0), zero) for j in range(width + 1)]
    while True:
        enter = next((j for j in range(art0) if cost[j] < 0), None)
        if enter is None:
            break
        ratios = [(tableau[i][-1] / tableau[i][enter], basis[i], i)
                  for i in range(len(tableau)) if tableau[i][enter] > 0]
        if not ratios:                      # phase one is bounded below by 0
            raise ArithmeticError("phase one of the simplex is unbounded")
        leave = min(ratios)[2]
        pivot = tableau[leave][enter]
        tableau[leave] = [v / pivot for v in tableau[leave]]
        for i in range(len(tableau)):
            if i != leave and tableau[i][enter] != 0:
                factor = tableau[i][enter]
                tableau[i] = [v - factor * p
                              for v, p in zip(tableau[i], tableau[leave])]
        factor = cost[enter]
        cost = [v - factor * p for v, p in zip(cost, tableau[leave])]
        basis[leave] = enter
    if cost[-1] != 0:                       # some artificial stays positive
        u = [s * (cost[art0 + i] - one) for i, s in enumerate(sign)]
        v = [cost[slack0 + k] for k in range(len(A_ub))]
        if not is_refutation((u, v), A_eq, b_eq, A_ub, b_ub):
            raise ArithmeticError("the simplex found no point and no proof "
                                  "that there is none")
        return None, (u, v)
    z = [zero] * art0
    for i, var in enumerate(basis):
        if var < art0:
            z[var] = tableau[i][-1]
    x = [z[j] - z[n + j] for j in range(n)]
    if not (all(dot(a, x) == r for a, r in zip(A_eq, b_eq))
            and all(dot(a, x) <= r for a, r in zip(A_ub, b_ub))):
        raise ArithmeticError("the simplex returned a point that violates "
                              "its own rows")
    return x, None


def is_refutation(proof, A_eq=(), b_eq=(), A_ub=(), b_ub=()):
    """Do the multipliers ``proof = (u, v)`` prove that no ``x`` has
    ``A_eq x = b_eq`` and ``A_ub x <= b_ub``? They do when ``v >= 0``,
    ``u A_eq + v A_ub = 0`` and ``u b_eq + v b_ub < 0``: a point would give
    ``0 = (u A_eq + v A_ub) x <= u b_eq + v b_ub < 0``."""
    u, v = proof
    if len(u) != len(A_eq) or len(v) != len(A_ub):
        return False
    rows = [list(r) for r in A_eq] + [list(r) for r in A_ub]
    weights = list(u) + list(v)
    columns = range(len(rows[0])) if rows else ()
    return (all(m >= 0 for m in v)
            and all(dot(weights, [row[j] for row in rows]) == 0
                    for j in columns)
            and dot(u, b_eq) + dot(v, b_ub) < 0)


# --------------------------------------------------------------------------- #
# The exact image: pieces                                                       #
# --------------------------------------------------------------------------- #
class Piece:
    """The network on one linear region: ``x -> W x + c`` on ``A x <= b``.

    ``strict`` lists the rows ``(a, r)`` that hold strictly on the region,
    ``a x < r``: the open ends of the branches of a discontinuous operation.
    Vertices and images are those of the closed region ``A x <= b``."""

    def __init__(self, A, b, W, c, strict=()):
        self.A, self.b, self.W, self.c = A, b, W, c
        self.strict = list(strict)
        self._corners = None

    def corners(self):
        """Vertices of the closed region, in input space."""
        if self._corners is None:
            self._corners = vertices(self.A, self.b)
        return self._corners

    def value(self, x):
        return tuple(dot(row, x) + ci for row, ci in zip(self.W, self.c))

    def image(self):
        """Vertices whose hull is the image of the region (not minimal)."""
        return sorted({self.value(v) for v in self.corners()})

    def holds(self, x):
        """Is the input ``x`` in the region, strict rows included?"""
        return (all(dot(row, x) <= rhs for row, rhs in zip(self.A, self.b))
                and all(dot(row, x) < rhs for row, rhs in self.strict))

    def is_inhabited(self):
        """Does the region have a point? Each strict row must leave room at
        some vertex; the average of such vertices then satisfies them all, and
        the region is dense in its closure."""
        corners = self.corners()
        return bool(corners) and all(
            any(dot(row, v) < rhs for v in corners) for row, rhs in self.strict)

    def is_full_dimensional(self):
        corners = self.corners()
        if not corners:
            return False
        base = corners[0]
        return rank([[vi - bi for vi, bi in zip(v, base)]
                     for v in corners[1:]]) == len(base)


def polytope(A, b):
    """The identity on the bounded polytope ``A x <= b``."""
    A, b = matrix(A), vector(b)
    if not is_bounded(A):
        raise ValueError("the input polytope is unbounded")
    n = len(A[0])
    identity = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    return [Piece(A, b, identity, [Fraction(0)] * n)]


def box(lb, ub):
    """The identity on the box ``lb <= x <= ub``."""
    lb, ub = vector(lb), vector(ub)
    n = len(lb)
    unit = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    return polytope(unit + [[-v for v in row] for row in unit],
                    ub + [-v for v in lb])


def affine(pieces, W, c=None):
    """Follow every piece by ``y -> W y + c``."""
    W = matrix(W)
    c = [Fraction(0)] * len(W) if c is None else vector(c)
    out = []
    for piece in pieces:
        columns = list(zip(*piece.W))
        out.append(Piece(piece.A, piece.b,
                         [[dot(row, col) for col in columns] for row in W],
                         [dot(row, piece.c) + ci for row, ci in zip(W, c)],
                         piece.strict))
    return out


def split(pieces, row, rhs):
    """Cut every region by the input-space hyperplane ``row x = rhs`` and keep
    both closed sides. The union of the images is unchanged."""
    row, rhs = vector(row), frac(rhs)
    out = []
    for piece in pieces:
        for a, r in ((row, rhs), ([-v for v in row], -rhs)):
            child = Piece(piece.A + [a], piece.b + [r], piece.W, piece.c,
                          piece.strict)
            if child.is_inhabited():
                out.append(child)
    return out


def piecewise(pieces, branches):
    """Apply a scalar piecewise-linear function to every output coordinate.

    ``branches(lo, hi)`` lists the function's branches that a coordinate with
    range ``[lo, hi]`` can meet, each as ``(lower, upper, slope, intercept)``:
    on ``lower <= t <= upper`` the function is ``slope * t + intercept``. A
    bound of None is infinite. Two more entries, ``lower_open`` and
    ``upper_open``, say that the branch does not hold at that end (a jump).

    Regions with no point are dropped. Regions of lower dimension are kept (a
    point branch such as Sign at zero, or a region that touches a kink of a
    continuous function); :meth:`Piece.is_full_dimensional` tells them
    apart."""
    out = []
    for piece in pieces:
        parts = [piece]
        for i in range(len(piece.W)):
            children = []
            for part in parts:
                row, offset = part.W[i], part.c[i]
                values = [dot(row, v) + offset for v in part.corners()]
                lo, hi = min(values), max(values)
                for branch in branches(lo, hi):
                    lower, upper, slope, intercept = branch[:4]
                    lower_open, upper_open = branch[4:] or (False, False)
                    if (lower is not None and lower > hi) or \
                            (upper is not None and upper < lo):
                        continue
                    A, b, strict = list(part.A), list(part.b), list(part.strict)
                    if lower is not None:
                        above_lower = ([-w for w in row], offset - lower)
                        if lower > lo:
                            A.append(above_lower[0])
                            b.append(above_lower[1])
                        if lower_open:
                            strict.append(above_lower)
                    if upper is not None:
                        below_upper = (list(row), upper - offset)
                        if upper < hi:
                            A.append(below_upper[0])
                            b.append(below_upper[1])
                        if upper_open:
                            strict.append(below_upper)
                    W, c = list(part.W), list(part.c)
                    W[i] = [slope * w for w in row]
                    c[i] = slope * offset + intercept
                    child = Piece(A, b, W, c, strict)
                    if child.is_inhabited():
                        children.append(child)
            parts = children
        out.extend(parts)
    return out


def relu(pieces):
    return piecewise(pieces, lambda lo, hi: [(None, 0, 0, 0), (0, None, 1, 0)])


def leaky_relu(pieces, slope):
    slope = frac(slope)
    return piecewise(pieces, lambda lo, hi: [(None, 0, slope, 0),
                                             (0, None, 1, 0)])


def sign(pieces):
    """-1 below zero, 0 at zero, 1 above."""
    return piecewise(pieces, lambda lo, hi: [(None, 0, 0, -1, False, True),
                                             (0, 0, 0, 0),
                                             (0, None, 0, 1, True, False)])


def round_(pieces):
    """Round to the nearest integer, a half to the even neighbour (the rule of
    ``torch.round``, ``numpy.round`` and ONNX ``Round``): the branch of an
    even k is closed at both ends, that of an odd k open at both."""
    half = Fraction(1, 2)

    def branches(lo, hi):
        return [(k - half, k + half, 0, k, k % 2 == 1, k % 2 == 1)
                for k in range(math.ceil(lo - half), math.floor(hi + half) + 1)]
    return piecewise(pieces, branches)


def floor(pieces):
    """The integer k on ``k <= t < k + 1``."""
    return piecewise(pieces, lambda lo, hi: [
        (k, k + 1, 0, k, False, True)
        for k in range(math.ceil(lo - 1), math.floor(hi) + 1)])


def ceil(pieces):
    """The integer k on ``k - 1 < t <= k``."""
    return piecewise(pieces, lambda lo, hi: [
        (k - 1, k, 0, k, True, False)
        for k in range(math.ceil(lo), math.floor(hi + 1) + 1)])


def maxpool(pieces, groups):
    """Output j is the largest of the coordinates listed in ``groups[j]``. A
    region is split by which coordinate attains each maximum."""
    out = []
    for piece in pieces:
        parts = [(list(piece.A), list(piece.b), [])]
        for group in groups:
            children = []
            for A, b, chosen in parts:
                for k in group:
                    A2, b2 = list(A), list(b)
                    for other in group:
                        if other != k:
                            A2.append([wo - wk for wo, wk in
                                       zip(piece.W[other], piece.W[k])])
                            b2.append(piece.c[k] - piece.c[other])
                    if Piece(A2, b2, [], [], piece.strict).is_inhabited():
                        children.append((A2, b2, chosen + [k]))
            parts = children
        out.extend(Piece(A, b, [piece.W[k] for k in chosen],
                         [piece.c[k] for k in chosen], piece.strict)
                   for A, b, chosen in parts)
    return out


def evaluate(pieces, x):
    """The network's value at the input ``x``, read from the piece that holds
    it (strict rows included, so a jump gets the function's own value)."""
    x = vector(x)
    for piece in pieces:
        if piece.holds(x):
            return piece.value(x)
    raise ValueError("the point is in no piece")


# --------------------------------------------------------------------------- #
# Membership in a computed set                                                  #
# --------------------------------------------------------------------------- #
def star(computed):
    """``(c, G, C, d, lb, ub)`` of a computed set as Fractions, each the exact
    value of the stored float, read by ``membership.rows``: a Star, ImageStar,
    Zono, ImageZono, Box, Hexatope or Octatope. An infinite predicate bound
    becomes None; a pinned variable (the anchor of a Hexatope) has
    ``lb == ub``."""
    from tests._oracles.membership import rows
    c, G, C, d, lb, ub = rows(computed)
    finite = lambda v: None if math.isinf(v) else frac(v)       # noqa: E731
    return (vector(c), matrix(G), matrix(C), vector(d),
            [finite(v) for v in lb], [finite(v) for v in ub])


def interval(lb, ub):
    """The box ``lb <= y <= ub`` in the same form as :func:`star`."""
    lb, ub = vector(lb), vector(ub)
    n = len(lb)
    identity = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    return ([Fraction(0)] * n, identity, [], [], lb, ub)


def halfspaces(A, b):
    """The set ``{y : A y <= b}`` in the same form as :func:`star`: rows given
    directly on the output, as a relaxation polygon is."""
    A, b = matrix(A), vector(b)
    n = len(A[0])
    identity = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    return ([Fraction(0)] * n, identity, A, b, [None] * n, [None] * n)


def _predicate_rows(computed):
    """``C a <= d`` together with the finite bounds, as inequality rows."""
    _, G, C, d, lb, ub = computed
    n = len(lb)
    A, b = [list(row) for row in C], list(d)
    for j in range(n):
        unit = [Fraction(int(k == j)) for k in range(n)]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    return A, b


def member(computed, y):
    """Is the rational point ``y`` in ``{c + G a : C a <= d, lb <= a <= ub}``?
    Exact, with no slack."""
    c, G = computed[0], computed[1]
    y = vector(y)
    if len(y) != len(c):
        raise ValueError(f"the set has {len(c)} entries, the point {len(y)}")
    A, b = _predicate_rows(computed)
    return feasible(G, [yi - ci for yi, ci in zip(y, c)], A, b) is not None


class Membership:
    """:func:`member` for many points of one computed set, faster and as
    exact.

    Either answer has a certificate that carries over to nearby points. A
    member ``y`` has a predicate point ``a`` with ``c + G a = y`` inside every
    row; the simplex returns a vertex, where some rows hold with equality, and
    the same rows solved for another ``y`` often give a point that is inside
    again. A point that is not a member has multipliers ``(u, v)`` with
    ``v >= 0`` and ``u G + v A = 0`` (:func:`decide`); any ``y`` with
    ``u (y - c) + v b < 0`` is then not a member either. Remembered
    certificates are tried first and checked exactly for the point at hand;
    when none applies the simplex decides."""

    def __init__(self, computed, keep=16):
        self.c, self.G = computed[0], [list(row) for row in computed[1]]
        self.A, self.b = _predicate_rows(computed)
        self._keep = keep
        self._inside = []           # row choices that gave a certifying point
        self._outside = []          # (u, v b) of a proof of non-membership

    def __call__(self, y):
        y = vector(y)
        if len(y) != len(self.c):
            raise ValueError(
                f"the set has {len(self.c)} entries, the point {len(y)}")
        rhs = [yi - ci for yi, ci in zip(y, self.c)]
        for k, (u, offset) in enumerate(self._outside):
            if dot(u, rhs) + offset < 0:
                self._outside.insert(0, self._outside.pop(k))
                return False
        for k, tight in enumerate(self._inside):
            a = solve(self.G + [self.A[i] for i in tight],
                      rhs + [self.b[i] for i in tight])
            if a is not None and self._certifies(a, rhs):
                self._inside.insert(0, self._inside.pop(k))
                return True
        a, proof = decide(self.G, rhs, self.A, self.b)
        if a is None:
            u, v = proof
            self._outside.insert(0, (u, dot(v, self.b)))
            del self._outside[self._keep:]
            return False
        tight = self._vertex_rows(a)
        if tight is not None and tight not in self._inside:
            self._inside.insert(0, tight)
            del self._inside[self._keep:]
        return True

    def _certifies(self, a, rhs):
        return (all(dot(row, a) == r for row, r in zip(self.G, rhs))
                and all(dot(row, a) <= r for row, r in zip(self.A, self.b)))

    def _vertex_rows(self, a):
        """Rows that hold with equality at ``a`` and, with ``G``, determine
        it; None when ``a`` is not a vertex."""
        n = len(a)
        chosen, rows = [], list(self.G)
        if rank(rows) < len(rows):
            return None
        for i, (row, r) in enumerate(zip(self.A, self.b)):
            if len(rows) == n:
                break
            if dot(row, a) == r and rank(rows + [row]) > len(rows):
                rows.append(row)
                chosen.append(i)
        return tuple(chosen) if len(rows) == n else None


def meets_box(computed, lo, hi):
    """Does the set have a point in the box ``lo <= y <= hi``? Exact."""
    c, G = computed[0], computed[1]
    lo, hi = vector(lo), vector(hi)
    A, b = _predicate_rows(computed)
    A = A + [list(row) for row in G] + [[-v for v in row] for row in G]
    b = b + [h - ci for h, ci in zip(hi, c)] + [ci - l for l, ci in zip(lo, c)]
    return feasible((), (), A, b) is not None


def set_vertices(computed):
    """Points whose hull is the computed set: the images of the vertices of
    its predicate polytope; an empty list when the set is empty. For a handful
    of predicate variables only. An unbounded predicate polytope (a Hexatope
    or Octatope whose rows leave a variable open, a Star without bounds) has
    no such list: ``ValueError``. A set on no predicate variable is its centre,
    or empty when a row ``0 <= d`` fails."""
    c, G = computed[0], computed[1]
    A, b = _predicate_rows(computed)
    if not computed[4]:
        return [tuple(c)] if all(rhs >= 0 for rhs in b) else []
    if not A or not is_bounded(A):
        raise ValueError("the predicate polytope is unbounded")
    return sorted({tuple(dot(row, a) + ci for row, ci in zip(G, c))
                   for a in vertices(A, b)})


# --------------------------------------------------------------------------- #
# The two directions of a comparison                                            #
# --------------------------------------------------------------------------- #
def escapes(pieces, *computed):
    """Vertices of the exact image that are in none of the computed sets, as
    ``(piece index, vertex)``."""
    return [(i, y) for i, piece in enumerate(pieces) for y in piece.image()
            if not any(member(s, y) for s in computed)]


def straddlers(pieces, *computed):
    """Indices of the pieces that no single computed set holds all the
    vertices of. With :func:`escapes` empty these are the pieces spread over
    several computed sets, for which the vertex test proves nothing."""
    return [i for i, piece in enumerate(pieces)
            if not any(all(member(s, y) for y in piece.image())
                       for s in computed)]


def image_contains(pieces, y):
    """Is the rational point ``y`` in the closure of the image: the value of
    some piece's map at a point of its closed region?"""
    y = vector(y)
    return any(feasible(piece.W, [yi - ci for yi, ci in zip(y, piece.c)],
                        piece.A, piece.b) is not None for piece in pieces)


def unreached(pieces, points):
    """The points outside the closure of the image: what a method that claims
    the exact image must not return. A list of ``(index, point)``."""
    return [(i, tuple(vector(y))) for i, y in enumerate(points)
            if not image_contains(pieces, y)]
