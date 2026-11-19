"""Soundness cells for the Hexatope and Octatope set operations.

What the file covers: ``from_bounds``; ``affine_map`` through the Linear
kernel (``reach_layer`` and the model entry); the range queries
(``get_range``, ``get_ranges`` sequential and parallel, ``get_bounds``,
``get_box``, ``optimize_linear``, ``estimate_range(s)``) with the solvers
'lp' (CVXPY OSQP) and 'mcf' (min-cost flow on the constraint graph, through
``UTVPIConstraintSystem.to_dcs`` for an Octatope), each against the exact
range in both directions where the docstring says exact, on kernels with
and without a negative edge, with two rows on one pair of variables, with no
row, and with a row the flow cannot take; their refusals (an unknown solver,
an index outside the set, a constraint system given an index outside it or
a UTVPI row it cannot hold); ``intersect_half_space`` in each branch of the
bounding-box functions (a row that is already a difference or UTVPI row, a
scaled one, a general one, two rows, coefficients equal only up to a
tolerance); the ReLU split those functions serve (``reach_layer`` with and
without a phase solver, and the model entry); ``contains``,
``is_empty_set`` (``DifferenceConstraintSystem.is_feasible``) and
``Octatope.sample``. ``to_star`` is in ``test_soundness_to_star.py``. ReLU
cells of other shapes (coupled families, several input sets, a kernel
variable beyond one) belong to ``test_soundness_hexoct_relu.py``.

Oracle. A set means what its stored arrays say, read by ``tests/_oracles``
(``membership.rows``, ``rational.star``) and never by a method of the set:
a Hexatope is ``center + generators x`` for every ``x`` that satisfies each
stored difference row ``x_i - x_j <= b`` with the anchor ``x_0`` pinned to 0;
an Octatope is ``center + generators x`` for every ``x`` that satisfies each
stored UTVPI row. Exact vertices come from ``rational.set_vertices``, exact
ranges are the extremes over them, exact images of piecewise-linear maps from
``rational.affine`` and ``rational.relu``, exact membership from
``rational.Membership`` with zero slack. Every input is dyadic, so the float
arithmetic of the library is exact wherever it has no solver in it.

Where the library's answer carries a number computed by CVXPY OSQP at 1e-7
(the 'lp' solver of the range queries, the bounding box of a split by a
general half-space) membership is decided by the LP oracle
(``membership.distance``) with the slack ``LP_TOL`` relative to the size of
the point, and a range by the same slack: that solver's missing outward pad
is registry row HEXOCT-4, marked elsewhere. The other side of
an exactness check (a range no wider than the exact one) allows
``_OUTWARD``: 1e-12 relative for 'mcf', 1e-6 for 'lp'. Every other check
here has zero slack.

Layout: all sets are flat; entries are in the network's own order.
"""

import itertools
import multiprocessing
import queue as queue_module
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Hexatope, Octatope
from n2v.sets.hexatope import DifferenceConstraintSystem
from n2v.sets.octatope import UTVPIConstraint, UTVPIConstraintSystem
from tests._oracles import rational as R
from tests._oracles.membership import LP_TOL, distance
from tests.cell_support import (assert_covered, flat_box, flat_set,
                                reach_model, reach_one_layer, replay_torch,
                                samples, split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug

KINDS = ("Hexatope", "Octatope")
SOLVERS = ("lp", "mcf")


# --------------------------------------------------------------------------- #
# Building sets with the public constructors                                    #
# --------------------------------------------------------------------------- #
def _hexatope(center, generators, rows):
    """A Hexatope from its centre, generators (anchor column first) and
    difference rows ``(i, j, b)`` meaning ``x_i - x_j <= b``."""
    generators = np.asarray(generators, dtype=np.float64)
    dcs = DifferenceConstraintSystem(generators.shape[1])
    for i, j, b in rows:
        dcs.add_constraint(i, j, b)
    return Hexatope(np.asarray(center, dtype=np.float64), generators, dcs)


def _octatope(center, generators, rows):
    """An Octatope from its centre, generators and UTVPI rows
    ``(i, j, ai, aj, b)`` meaning ``ai x_i + aj x_j <= b`` (both
    coefficients on ``x_i`` when ``i == j``)."""
    generators = np.asarray(generators, dtype=np.float64)
    utvpi = UTVPIConstraintSystem(generators.shape[1])
    for row in rows:
        utvpi.add_constraint(*row)
    return Octatope(np.asarray(center, dtype=np.float64), generators, utvpi)


def _box_rows(n):
    """Difference rows that give each of the kernel variables 1..n the
    interval [-1, 1] against the anchor."""
    return [r for i in range(1, n + 1) for r in ((i, 0, 1.0), (0, i, 1.0))]


# The Hexatope kernel of the range cells: the box plus rows that are not
# mirror-symmetric (x1 - x2 <= 1/4, x3 <= 1/2, x1 >= -3/4, x2 - x3 <= 1/2).
# Every bound is non-negative: a negative difference bound with these
# generators makes the min-cost-flow query hang (HEXOCT-10, cell below).
_HEX_ROWS = _box_rows(3) + [(1, 2, 0.25), (3, 0, 0.5), (0, 1, 0.75),
                            (2, 3, 0.5)]
_CENTER = [-0.375, 0.5, -1.25]
_HEX_G = [[0.0, 0.5, -0.25, 0.75],
          [0.0, -1.0, 0.5, 0.25],
          [0.0, 0.25, 0.25, -0.5]]
# The Octatope kernel: the box plus two-variable rows of every sign pattern
# (-x0 - x1 <= 1/2, x1 + x2 <= 3/4, x0 - x2 <= 1/4, -x2 + x1 <= 1) and
# one-variable rows written with ai == 0 (-x2 <= 3/4, x0 <= 1/2).
_OCT_ROWS = ([r for i in range(3) for r in ((i, i, 1, 0, 1.0),
                                            (i, i, -1, 0, 1.0))]
             + [(0, 1, -1, -1, 0.5), (1, 2, 1, 1, 0.75), (0, 2, 1, -1, 0.25),
                (2, 1, -1, 1, 1.0), (0, 2, 0, -1, 0.75), (1, 0, 0, 1, 0.5)])
_OCT_G = [[0.5, -0.25, 0.75],
          [-1.0, 0.5, 0.25],
          [0.25, 0.25, -0.5]]


def _kernel_set(kind):
    """The cut set of ``kind`` used by the range and membership cells."""
    if kind == "Hexatope":
        return _hexatope(_CENTER, _HEX_G, _HEX_ROWS)
    return _octatope(_CENTER, _OCT_G, _OCT_ROWS)


# --------------------------------------------------------------------------- #
# The exact reference                                                           #
# --------------------------------------------------------------------------- #
def _polytope(s):
    """``(c, G, A, b)``: the set ``{c + G a : A a <= b}`` read from its stored
    arrays by ``rational.star``. A pinned predicate variable (the anchor of a
    Hexatope, ``lb == ub``) is replaced by its value; ``A`` holds the stored
    rows and every other finite predicate bound."""
    c, G, C, d, lb, ub = R.star(s)
    n = len(lb)
    pinned = {j: lb[j] for j in range(n)
              if lb[j] is not None and ub[j] is not None and lb[j] == ub[j]}
    free = [j for j in range(n) if j not in pinned]
    c = [ci + sum((row[j] * v for j, v in pinned.items()), Fraction(0))
         for ci, row in zip(c, G)]
    G = [[row[j] for j in free] for row in G]
    A = [[row[j] for j in free] for row in C]
    b = [di - sum((row[j] * v for j, v in pinned.items()), Fraction(0))
         for di, row in zip(d, C)]
    for k, j in enumerate(free):
        unit = [Fraction(int(m == k)) for m in range(len(free))]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    return c, G, A, b


def _vertices(s):
    """Points whose hull is the set, exact; a test-setup failure when the set
    is empty or unbounded."""
    c, G, A, b = _polytope(s)
    if not A or not R.is_bounded(A):
        pytest.fail("test setup: the set is unbounded")
    found = sorted({tuple(R.dot(row, a) + ci for row, ci in zip(G, c))
                    for a in R.vertices(A, b)})
    if not found:
        pytest.fail("test setup: the set is empty")
    return found


def _exact_ranges(s):
    verts = _vertices(s)
    return ([min(v[i] for v in verts) for i in range(len(verts[0]))],
            [max(v[i] for v in verts) for i in range(len(verts[0]))])


def _pieces(s):
    """The set as one exact piece of ``rational`` (its predicate polytope
    mapped by its own centre and generators)."""
    c, G, A, b = _polytope(s)
    return R.affine(R.polytope(A, b), G, c)


def _assert_range(lo, hi, exact_lo, exact_hi, slack, what):
    """``[lo, hi]`` holds ``[exact_lo, exact_hi]`` entry by entry, one-sided,
    with ``slack`` relative to the size of the exact value (0: exact)."""
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    assert lo.size == len(exact_lo) and hi.size == len(exact_hi), (
        f"{what}: {lo.size} and {hi.size} bounds for {len(exact_lo)} entries")
    for i, (a, b, ea, eb) in enumerate(zip(lo, hi, exact_lo, exact_hi)):
        pad_lo = Fraction(slack) * max(1, abs(ea))
        pad_hi = Fraction(slack) * max(1, abs(eb))
        assert np.isfinite(a) and Fraction(float(a)) <= ea + pad_lo, (
            f"{what}: lower bound {a!r} of entry {i} is above the exact "
            f"minimum {float(ea)!r}")
        assert np.isfinite(b) and Fraction(float(b)) >= eb - pad_hi, (
            f"{what}: upper bound {b!r} of entry {i} is below the exact "
            f"maximum {float(eb)!r}")


# How far a range that its docstring calls exact ('Compute exact range',
# 'Compute exact ranges', 'Compute exact bounding box') may lie outside the
# exact range. 'mcf' is network simplex on dyadic data: exact up to the float64
# class (1e-12 relative, room for a value recomputed and rounded outward by one
# unit in the last place). 'lp' is an LP optimum, which DESIGN section 12 pads
# outward by the library's LP pad (1e-7 relative, ``_LP_PAD`` of
# ``n2v/utils/lpsolver.py``): ten times that. Neither forgives a bound inside
# the exact range; that side keeps the slack of ``_assert_range``.
_OUTWARD = {"mcf": 1e-12, "lp": 1e-6}


def _assert_exact_range(lo, hi, exact_lo, exact_hi, solver, what):
    """``[lo, hi]`` is the exact range: it holds it (slack ``LP_TOL`` for 'lp',
    none for 'mcf') and lies outside it by no more than ``_OUTWARD``."""
    _assert_range(lo, hi, exact_lo, exact_hi,
                  LP_TOL if solver == "lp" else 0.0, what)
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    out = Fraction(_OUTWARD[solver])
    for i, (a, b, ea, eb) in enumerate(zip(lo, hi, exact_lo, exact_hi)):
        assert Fraction(float(a)) >= ea - out * max(1, abs(ea)), (
            f"{what}: lower bound {a!r} of entry {i} is below the exact "
            f"minimum {float(ea)!r}: the range is not the exact range")
        assert Fraction(float(b)) <= eb + out * max(1, abs(eb)), (
            f"{what}: upper bound {b!r} of entry {i} is above the exact "
            f"maximum {float(eb)!r}: the range is not the exact range")


def _maximal(pieces):
    """The pieces whose image is not inside another piece's image, with the
    vertices of those images (a piece on a kink is a face of a neighbour)."""
    images = [frozenset(p.image()) for p in pieces]
    keep = [i for i, im in enumerate(images)
            if not any(im < other or (im == other and j < i)
                       for j, other in enumerate(images) if j != i)]
    return [sorted(images[i]) for i in keep]


def _assert_image_held(pieces, sets, what, exact):
    """All vertices of each exact piece lie in one returned set: exactly
    (``rational.Membership``, zero slack) or by the LP oracle with
    ``LP_TOL``. A convex set holding the vertices of a polytope holds the
    polytope; a piece spread over several sets proves nothing and fails."""
    if exact:
        readers = [R.Membership(R.star(s)) for s in sets]
    else:
        def reader(s):
            def member(y):
                y = np.array([float(v) for v in y])
                return distance(s, y) <= LP_TOL * max(1.0, np.max(np.abs(y)))
            return member
        readers = [reader(s) for s in sets]
    images = _maximal(pieces)
    escaped, spread = [], []
    for k, image in enumerate(images):
        if any(all(member(y) for y in image) for member in readers):
            continue
        outside = [y for y in image if not any(m(y) for m in readers)]
        if outside:
            escaped.append((k, [float(v) for v in outside[0]]))
        else:
            spread.append(k)
    assert not escaped, (
        f"{what}: {len(escaped)} of {len(images)} pieces of the exact image "
        f"have a vertex in none of the {len(sets)} returned sets; first: "
        f"piece {escaped[0][0]}, vertex {escaped[0][1]}")
    assert not spread, (
        f"{what}: pieces {spread} of the exact image are spread over several "
        f"returned sets and held by no single one")


def _linear(W, b):
    layer = nn.Linear(len(W[0]), len(W))
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(W, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(b, dtype=torch.float32))
    return layer


def _in_child(fn, limit):
    """``fn()`` computed in a forked child process. ``TimeoutError`` when it
    gives no answer within ``limit`` seconds (the child is killed); a
    ``RuntimeError`` naming the child's exception when it raised."""
    ctx = multiprocessing.get_context("fork")
    answers = ctx.Queue()

    def target():
        try:
            answers.put(("value", fn()))
        except BaseException as error:              # reported to the parent
            answers.put(("error", f"{type(error).__name__}: {error}"))

    process = ctx.Process(target=target, daemon=True)
    process.start()
    try:
        kind, value = answers.get(timeout=limit)
    except queue_module.Empty:
        process.kill()
        process.join()
        raise TimeoutError(f"no answer within {limit} s") from None
    process.join()
    if kind == "error":
        raise RuntimeError(f"the call raised in the child: {value}")
    return value


# --------------------------------------------------------------------------- #
# from_bounds                                                                   #
# --------------------------------------------------------------------------- #
def _corners(lo, hi):
    return [list(p) for p in itertools.product(
        *[sorted({Fraction(float(a)), Fraction(float(b))})
          for a, b in zip(lo, hi)])]


@pytest.mark.parametrize("kind", KINDS)
def test_from_bounds_is_the_box_of_dyadic_bounds(kind):
    """``from_bounds`` on dyadic bounds (four coordinates: straddling,
    positive, negative and one of zero width) is the box itself: every corner
    is a member, exactly, and every vertex of the set lies in the box.
    Catches a generator placed on the anchor column (that coordinate loses
    its width), a centre computed as (ub - lb) / 2, a half-width halved, and
    a lost or doubled kernel row."""
    lo, hi = flat_box(4, seed=3, zero_width=True)
    s = flat_set(kind, lo, hi)
    member = R.Membership(R.star(s))
    missed = [p for p in _corners(lo, hi) if not member(p)]
    assert not missed, (f"{kind}.from_bounds: {len(missed)} corners of its own "
                        f"box are not members; first {[float(v) for v in missed[0]]}")
    low = [Fraction(float(v)) for v in lo]
    high = [Fraction(float(v)) for v in hi]
    outside = [v for v in _vertices(s)
               if not all(a <= x <= b for a, x, b in zip(low, v, high))]
    assert not outside, (f"{kind}.from_bounds: the set has the vertex "
                         f"{[float(x) for x in outside[0]]} outside the box")


@pytest.mark.parametrize("kind", KINDS)
@known_bug(9, "Hexatope.from_bounds and Octatope.from_bounds compute the "
              "centre (lb + ub) / 2 and the half-widths (ub - lb) / 2 by "
              "round to nearest, so the set can leave out a corner of the box "
              "it was built from (registry HEXOCT-8)")
def test_from_bounds_holds_every_corner_of_a_non_dyadic_box(kind):
    """The promise of ``from_bounds`` for any finite bounds: the set holds the
    box, so every corner is a member, decided exactly. Forty random 3-dimensional
    boxes with float64 ends (seed 0). The neighbour with dyadic bounds above
    passes with the same membership code."""
    rng = np.random.default_rng(0)
    missed = []
    for k in range(40):
        lo = rng.uniform(-3, 3, 3)
        hi = lo + rng.uniform(0.01, 2, 3)
        s = flat_set(kind, lo, hi)
        member = R.Membership(R.star(s))
        missed += [(k, p) for p in _corners(lo, hi) if not member(p)]
    assert not missed, (f"{kind}.from_bounds: {len(missed)} corners of 40 "
                        f"non-dyadic boxes are not members of the set built "
                        f"from them (first in box {missed[0][0]})")


# --------------------------------------------------------------------------- #
# estimate_range(s) on a kernel in [-1, 1]                                      #
# --------------------------------------------------------------------------- #
def _generator_sets(kind, data):
    """Sets on the kernel box [-1, 1]^3 with coupled generators: dyadic
    (multiples of 1/8) or random float64."""
    rng = np.random.default_rng(1)
    out = []
    for _ in range(12):
        if data == "dyadic":
            c = rng.integers(-16, 17, 3) / 8.0
            G = rng.integers(-8, 9, (3, 3)) / 8.0
        else:
            c = rng.uniform(-2, 2, 3)
            G = rng.uniform(-1, 1, (3, 3))
        if kind == "Hexatope":
            out.append(_hexatope(c, np.hstack([np.zeros((3, 1)), G]),
                                 _box_rows(3)))
        else:
            out.append(_octatope(c, G, [r for i in range(3) for r in
                                        ((i, i, 1, 0, 1.0),
                                         (i, i, -1, 0, 1.0))]))
    return out


@pytest.mark.parametrize("data", [
    "dyadic",
    pytest.param("float64", marks=known_bug(
        9, "Hexatope.estimate_range and Octatope.estimate_range sum the "
           "centre and the generator magnitudes by round to nearest, so a "
           "bound can lie inside the exact range (registry HEXOCT-8)"))])
@pytest.mark.parametrize("kind", KINDS)
def test_estimate_ranges_hold_the_exact_range(kind, data):
    """``estimate_range`` and ``estimate_ranges`` (the bounds the ReLU split
    uses when no solver is given) hold the exact range of a set whose kernel
    is the box [-1, 1]^3, zero slack. Catches a sum of signed generator
    entries instead of magnitudes (a neuron whose generators are mostly
    negative gets an upper bound below its maximum), an absolute value of the
    sum, a centre left out; with float64 data, a sum rounded inward."""
    for k, s in enumerate(_generator_sets(kind, data)):
        exact_lo, exact_hi = _exact_ranges(s)
        lo, hi = s.estimate_ranges()
        _assert_range(lo, hi, exact_lo, exact_hi, 0,
                      f"{kind}.estimate_ranges set {k} ({data})")
        for i in range(s.dim):
            a, b = s.estimate_range(i)
            _assert_range([a], [b], exact_lo[i:i + 1], exact_hi[i:i + 1], 0,
                          f"{kind}.estimate_range({i}) set {k} ({data})")


# --------------------------------------------------------------------------- #
# A kernel variable beyond [-1, 1] (HEXOCT-11)                                #
# --------------------------------------------------------------------------- #
def _single_variable(kind, low, high):
    """The set ``(t, 2 t)`` for the kernel variable t in [low, high], built
    with the public constructor (a Hexatope keeps the anchor x_0 with a zero
    generator and bounds x_1 against it)."""
    if kind == "Hexatope":
        rows = []
        if high is not None:
            rows.append((1, 0, high))
        if low is not None:
            rows.append((0, 1, -low))
        return _hexatope([0.0, 0.0], [[0.0, 1.0], [0.0, 2.0]], rows)
    rows = []
    if high is not None:
        rows.append((0, 0, 1, 0, high))
    if low is not None:
        rows.append((0, 0, -1, 0, -low))
    return _octatope([0.0, 0.0], [[1.0], [2.0]], rows)


_BEYOND_ONE = known_bug(
    9, "Hexatope.estimate_range and Octatope.estimate_range assume every "
       "kernel variable lies in [-1, 1]; for a set whose rows allow more the "
       "range leaves members out (t in [-3, 3]: (-1, 1) instead of (-3, 3))")


@pytest.mark.parametrize("bounds", [
    pytest.param((-0.5, 1.0), id="t-in-[-1/2,1]"),
    pytest.param((-3.0, 3.0), id="t-in-[-3,3]", marks=_BEYOND_ONE)])
@pytest.mark.parametrize("kind", KINDS)
def test_estimate_ranges_of_a_kernel_variable_beyond_one(kind, bounds):
    """A legal set ``(t, 2 t)`` whose rows give the kernel variable t the
    interval ``bounds``: ``estimate_ranges`` and ``estimate_range`` hold its
    exact range. The neighbour with t in [-1/2, 1] passes with the same
    reference. What the ReLU kernel does with these bounds is marked in
    ``test_soundness_hexoct_relu.py``."""
    s = _single_variable(kind, *bounds)
    exact_lo, exact_hi = _exact_ranges(s)
    lo, hi = s.estimate_ranges()
    _assert_range(lo, hi, exact_lo, exact_hi, 0,
                  f"{kind}.estimate_ranges, t in {bounds}")
    for i in range(2):
        a, b = s.estimate_range(i)
        _assert_range([a], [b], exact_lo[i:i + 1], exact_hi[i:i + 1], 0,
                      f"{kind}.estimate_range({i}), t in {bounds}")


_OPEN_BELOW = known_bug(
    9, "get_ranges of a Hexatope or Octatope whose rows leave a kernel "
       "variable unbounded falls back to estimate_range, which assumes "
       "[-1, 1]: the set {t <= 3} gets the range [-1, 1] although t = -100 "
       "and t = 3 are members")


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("case", [
    pytest.param("t-in-[-3,3]", id="bounded"),
    pytest.param("t<=3", id="open-below", marks=_OPEN_BELOW)])
@pytest.mark.parametrize("kind", KINDS)
def test_get_ranges_hold_every_member(kind, case, solver):
    """``get_ranges`` returns a range that holds every member of the set, or
    refuses (``NotImplementedError``), the refusal only for the set open
    below. Members: t = -100 (only for the set open below), -3, 0 and 3, with
    outputs ``(t, 2 t)``. Catches the fallback to the [-1, 1] estimate when
    the solver finds no optimum (an unbounded objective), and for the bounded
    neighbour a solver that assumes the kernel box [-1, 1] or refuses a
    bounded set."""
    s = _single_variable(kind, -3.0 if case == "t-in-[-3,3]" else None, 3.0)
    members = [-3.0, 0.0, 3.0] + ([-100.0] if case == "t<=3" else [])
    try:
        lo, hi = s.get_ranges(solver=solver)
    except NotImplementedError:
        if case != "t<=3":
            raise                   # a bounded set is supported: no refusal
        return
    lo, hi = np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)
    slack = LP_TOL if solver == "lp" else 0.0
    for t in members:
        y = np.array([t, 2 * t])
        pad = slack * max(1.0, float(np.max(np.abs(y))))
        assert np.all(lo <= y + pad) and np.all(y - pad <= hi), (
            f"{kind}.get_ranges(solver={solver!r}) on the set with {case} "
            f"returns [{lo.tolist()}, {hi.tolist()}], which leaves out the "
            f"member {y.tolist()}")


# --------------------------------------------------------------------------- #
# affine_map through the Linear kernel                                          #
# --------------------------------------------------------------------------- #
_W1 = [[0.5, -1.0, 0.25, 0.0],
       [-0.75, 0.0, 1.0, -0.5],
       [0.0, 0.25, -0.5, 1.0],
       [1.0, 0.5, 0.0, -0.25],
       [-0.25, -0.5, -0.75, 0.5]]
_B1 = [0.25, -0.5, 0.0, 0.75, -0.125]
_W2 = [[1.0, -0.5, 0.0, 0.25, -1.0],
       [-0.25, 0.75, 0.5, 0.0, 0.5]]
_B2 = [-0.25, 0.5]


def _linear_family(kind, family):
    """Input sets of four entries: the box of ``flat_box`` (with a zero-width
    coordinate), the cut kernel sets mapped to four entries, or three sets
    that cover a box."""
    lo, hi = flat_box(4, seed=7, zero_width=True)
    if family == "box":
        return [flat_set(kind, lo, hi)]
    if family == "rows":
        s = _kernel_set(kind)
        return [s.affine_map(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                       [0.0, 0.0, 1.0], [0.5, -0.5, 0.25]]),
                             np.array([[0.0], [0.0], [0.0], [0.125]]))]
    if family == "several":
        return split_cover(kind, lo, hi, parts=3)
    raise ValueError(family)


@pytest.mark.parametrize("entry", ["layer", "model"])
@pytest.mark.parametrize("family", ["box", "rows", "several"])
@pytest.mark.parametrize("kind", KINDS)
def test_linear_is_the_exact_affine_image(kind, family, entry):
    """Linear (4 -> 5, negative weights, bias) through ``reach_layer``, or
    Linear then Linear (5 -> 2) through ``reach_pytorch_model``: the result
    is exactly the affine image of each input set. Every vertex of the image
    of each input set is in the set returned for it (zero slack), and every
    vertex of every returned set has a preimage. Catches generators stored as
    absolute values (a mirrored parallelotope with the same per-entry
    ranges), the bias overwriting or missing from the centre, the kernel rows
    of a cut set dropped, and one set of a list used for another."""
    sets = _linear_family(kind, family)
    what = f"Linear on the {kind} family {family!r} through the {entry}"
    first = _linear(_W1, _B1)
    if entry == "layer":
        out = reach_one_layer(first, sets, "approx", what)
        maps = [(_W1, _B1)]
    else:
        model = nn.Sequential(first, _linear(_W2, _B2))
        out = reach_model(model, sets, "approx", what)
        maps = [(_W1, _B1), (_W2, _B2)]
    assert len(out) == len(sets), (
        f"{what}: {len(out)} sets returned for {len(sets)} input sets")
    for k, (s, o) in enumerate(zip(sets, out)):
        pieces = _pieces(s)
        for W, b in maps:
            pieces = R.affine(pieces, W, b)
        _assert_image_held(pieces, [o], f"{what}, input set {k}", exact=True)
        extra = R.unreached(pieces, _vertices(o))
        assert not extra, (f"{what}, input set {k}: the returned set has the "
                           f"vertex {[float(v) for v in extra[0][1]]}, which "
                           f"is not the image of an input point")


# --------------------------------------------------------------------------- #
# Range queries with the 'lp' and 'mcf' solvers                                 #
# --------------------------------------------------------------------------- #
_OBJECTIVES = [[1.0, -2.0, 0.5], [-0.25, 0.75, 1.0]]


def _query(s, query, solver):
    """``(lo, hi)`` per entry from one of the range queries."""
    if query == "get_ranges":
        return s.get_ranges(solver=solver)
    if query == "get_ranges-parallel":
        return s.get_ranges(solver=solver, parallel=True, n_workers=2)
    if query == "get_bounds":
        return s.get_bounds(solver=solver)
    if query == "get_box":
        box = s.get_box(solver=solver)
        return box.lb, box.ub
    if query == "get_range":
        pairs = [s.get_range(i, solver=solver) for i in range(s.dim)]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    raise ValueError(query)


@pytest.mark.parametrize("query", ["get_ranges", "get_ranges-parallel",
                                   "get_bounds", "get_box", "get_range"])
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("kind", KINDS)
def test_range_queries_hold_the_exact_range(kind, solver, query):
    """The range of every entry of a set with non-zero centres and a kernel
    cut by rows that are not mirror-symmetric (Hexatope: one-sided
    difference rows; Octatope: two-variable rows of every sign pattern and
    one-variable rows written with a zero first coefficient), against the
    exact range from the vertices, in both directions: the docstrings call
    each of these queries exact. Holding it: zero slack for 'mcf', ``LP_TOL``
    for 'lp' (HEXOCT-4). Being it: at most ``_OUTWARD`` outside. Every edge of
    these kernels has a non-negative cost, so the min-cost flow takes no cost
    shift. Catches a constraint-graph edge pointing the wrong way (the
    min-cost flow then optimises over the mirror image of the kernel), a
    minimisation that negates the objective and not its constant, a
    single-variable row x_i <= b converted to x+ - x- <= b instead of 2 b,
    the row -x_i - x_j <= b converted as x_i + x_j <= b, an LP objective
    without its constant, lower and upper bounds exchanged; and, by the
    second direction, a range that is sound and too wide: an Octatope flow
    objective not halved for the split variables, a row dropped on the way
    to the solver, an entry answered by the [-1, 1] estimate."""
    s = _kernel_set(kind)
    exact_lo, exact_hi = _exact_ranges(s)
    lo, hi = _query(s, query, solver)
    _assert_exact_range(lo, hi, exact_lo, exact_hi, solver,
                        f"{kind}.{query}(solver={solver!r})")


# An Octatope whose rows put both coefficients on one variable: (0, 0, 1, 1)
# is 2 x0 <= 1, (0, 0, -1, -1) is -2 x0 <= 1, and (1, 1, 1, -1) is 0 <= 1/2,
# which binds nothing; x1 has its box rows. Coupled generators.
_SAME_VARIABLE_ROWS = [(0, 0, 1, 1, 1.0), (0, 0, -1, -1, 1.0),
                       (1, 1, 1, 0, 1.0), (1, 1, -1, 0, 1.0),
                       (1, 1, 1, -1, 0.5)]


@pytest.mark.parametrize("solver", SOLVERS)
def test_octatope_rows_with_both_coefficients_on_one_variable(solver):
    """``get_ranges`` of an Octatope with rows ``(i, i, ai, aj, b)`` whose two
    coefficients are both nonzero, which ``UTVPIConstraintSystem`` reads as
    ``(ai + aj) x_i <= b`` (``to_matrix_form`` adds them; ``to_dcs`` emits
    both difference rows, or self-loops): the exact range in both directions
    (``_assert_exact_range``). Catches a matrix row that keeps only the first
    coefficient (2 x0 <= 1 read as x0 <= 1, and the vacuous 0 <= 1/2 read as
    x1 <= 1/2, which cuts the set) and a conversion to difference rows that
    drops the second coefficient."""
    s = _octatope([0.25, -0.5], [[1.0, 0.5], [-0.5, 1.0]], _SAME_VARIABLE_ROWS)
    exact_lo, exact_hi = _exact_ranges(s)
    lo, hi = _in_child(lambda: s.get_ranges(solver=solver), _MCF_LIMIT)
    _assert_exact_range(lo, hi, exact_lo, exact_hi, solver,
                        f"Octatope with two coefficients on one variable: "
                        f"get_ranges({solver!r})")


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("kind", KINDS)
def test_optimize_linear_is_outward_of_the_exact_optimum(kind, solver):
    """``optimize_linear`` of two objectives that mix every entry, both
    senses: a maximum not below the exact maximum and a minimum not above the
    exact minimum (zero slack for 'mcf', ``LP_TOL`` for 'lp'). Catches the
    objective composed with the generators in the wrong order, the sense
    ignored, and the centre term dropped."""
    s = _kernel_set(kind)
    verts = _vertices(s)
    slack = Fraction(LP_TOL) if solver == "lp" else Fraction(0)
    for f in _OBJECTIVES:
        values = [R.dot([Fraction(v) for v in f], y) for y in verts]
        for maximize, exact in ((True, max(values)), (False, min(values))):
            got = s.optimize_linear(np.array(f), maximize=maximize,
                                    solver=solver)
            what = (f"{kind}.optimize_linear({f}, maximize={maximize}, "
                    f"solver={solver!r})")
            assert got is not None and np.isfinite(got), f"{what}: {got!r}"
            pad = slack * max(1, abs(exact))
            if maximize:
                assert Fraction(float(got)) >= exact - pad, (
                    f"{what} = {got!r} is below the exact {float(exact)!r}")
            else:
                assert Fraction(float(got)) <= exact + pad, (
                    f"{what} = {got!r} is above the exact {float(exact)!r}")


# --------------------------------------------------------------------------- #
# The min-cost-flow query that does not terminate (HEXOCT-10)                   #
# --------------------------------------------------------------------------- #
# Seconds. The answering neighbour of the range cell needs milliseconds, that
# of the ReLU cell about a second (its split runs OSQP in the child).
_MCF_LIMIT = 3.0
_MCF_RELU_LIMIT = 6.0


def _difference_bound_set(bound):
    """The box kernel of three variables with generators that couple them,
    cut by the difference row x1 - x2 <= ``bound``."""
    return _hexatope(_CENTER, _HEX_G, _box_rows(3) + [(1, 2, bound)])


_MCF_HANG = known_bug(
    9, "the min-cost-flow optimisation of a Hexatope (networkx network_simplex "
       "on float costs and demands, after a cost shift of |min cost| + 1e-10) "
       "does not terminate when the kernel has a negative difference bound "
       "(x1 - x2 <= -1/4 with coupled generators); get_ranges and a ReLU "
       "with solver='mcf' never answer (registry HEXOCT-10)",
    raises=TimeoutError)


@pytest.mark.parametrize("bound", [
    pytest.param(0.25, id="x1-x2<=1/4"),
    pytest.param(-0.25, id="x1-x2<=-1/4", marks=_MCF_HANG)])
def test_min_cost_flow_ranges_answer_and_hold_the_exact_range(bound):
    """``get_ranges(solver='mcf')`` run in a child process with a time limit
    of 3 s: it answers, and the range holds the exact range (zero slack).
    The neighbour with a non-negative bound answers in milliseconds with the
    same harness and reference."""
    s = _difference_bound_set(bound)
    exact_lo, exact_hi = _exact_ranges(s)
    lo, hi = _in_child(lambda: s.get_ranges(solver="mcf"), _MCF_LIMIT)
    _assert_range(lo, hi, exact_lo, exact_hi, 0,
                  f"Hexatope.get_ranges(solver='mcf'), x1 - x2 <= {bound}")


_MCF_HANG_OCTATOPE = known_bug(
    9, "the min-cost-flow optimisation of an Octatope (to_dcs, then networkx "
       "network_simplex on float costs and demands after the cost shift of "
       "Hexatope._optimize_dcs_mcf) does not terminate on the box [-1, 1]^2 "
       "cut by y0 - y1 <= -1/4: get_ranges(solver='mcf') never answers "
       "(registry HEXOCT-10, which names the Hexatope; the Hexatope of the "
       "same box and cut hangs as well)",
    raises=TimeoutError)


@pytest.mark.parametrize("bound", [
    pytest.param(0.25, id="y0-y1<=1/4"),
    pytest.param(-0.25, id="y0-y1<=-1/4", marks=_MCF_HANG_OCTATOPE)])
def test_octatope_min_cost_flow_ranges_answer_and_hold_the_exact_range(bound):
    """The Octatope [-1, 1]^2 cut by y0 - y1 <= ``bound`` through
    ``intersect_half_space``: ``get_ranges(solver='mcf')``, run in a child
    process with a time limit of 3 s, answers and holds the exact range
    (zero slack). The smallest input I found on which the flow does not
    terminate: two variables and one negative difference row. The neighbour
    with a positive bound answers in milliseconds with the same harness and
    reference."""
    s = Octatope.from_bounds(np.full(2, -1.0), np.full(2, 1.0))
    s = s.intersect_half_space(np.array([[1.0, -1.0]]), np.array([[bound]]))
    exact_lo, exact_hi = _exact_ranges(s)
    lo, hi = _in_child(lambda: s.get_ranges(solver="mcf"), _MCF_LIMIT)
    _assert_range(lo, hi, exact_lo, exact_hi, 0,
                  f"Octatope.get_ranges(solver='mcf'), y0 - y1 <= {bound}")


@pytest.mark.parametrize("bound", [
    pytest.param(0.25, id="x1-x2<=1/4"),
    pytest.param(-0.25, id="x1-x2<=-1/4", marks=_MCF_HANG)])
def test_relu_with_min_cost_flow_phases_answers_and_holds_the_image(bound):
    """ReLU on the same sets through ``reach_layer`` with ``solver='mcf'``,
    in a child process with a time limit of 6 s: it answers, and every piece
    of the exact image is in one returned set (LP oracle, the split pieces
    carry OSQP bounds)."""
    s = _difference_bound_set(bound)
    out = _in_child(lambda: reach_one_layer(
        nn.ReLU(), [s], "approx", "ReLU with solver='mcf'", solver="mcf"),
        _MCF_RELU_LIMIT)
    _assert_image_held(R.relu(_pieces(s)), out,
                       f"ReLU(solver='mcf') on x1 - x2 <= {bound}", exact=False)


_OCT_BOX_ROWS = [r for i in range(3) for r in ((i, i, 1, 0, 1.0),
                                                (i, i, -1, 0, 1.0))]
_COUPLED = [[0.5, -0.25, 0.75], [-1.0, 0.5, 0.25], [0.25, 0.25, -0.5]]
_IDENTITY = [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
_SMALL_C = [0.25, -0.5, 0.125]


def _negative_bound_set(case):
    """Sets whose constraint graph has an edge of negative cost, on which the
    min-cost flow shifts every cost by |min cost| + 1e-10 and corrects the
    total afterwards."""
    if case == "hex-identity":
        return _hexatope(_SMALL_C, _IDENTITY, _box_rows(3) + [(1, 2, -0.25)])
    if case == "hex-identity-two-rows":
        return _hexatope(_SMALL_C, _IDENTITY,
                         _box_rows(3) + [(1, 2, -0.25), (3, 1, -0.5)])
    if case == "oct-identity-sum":
        return _octatope(_SMALL_C, np.eye(3),
                         _OCT_BOX_ROWS + [(0, 1, 1, 1, -0.25)])
    if case == "oct-coupled-sum":
        return _octatope(_SMALL_C, _COUPLED,
                         _OCT_BOX_ROWS + [(0, 1, 1, 1, -0.25)])
    if case == "oct-coupled-difference":
        return _octatope(_SMALL_C, _COUPLED,
                         _OCT_BOX_ROWS + [(0, 2, 1, -1, -0.5)])
    raise ValueError(case)


@pytest.mark.parametrize("case", [
    "hex-identity", "hex-identity-two-rows", "oct-identity-sum",
    "oct-coupled-sum",
    pytest.param("oct-coupled-difference", marks=known_bug(
        9, "the min-cost-flow optimum of a set whose constraint graph has a "
           "negative edge is computed with every cost shifted by |min cost| + "
           "1e-10 and corrected by shift * total flow in round to nearest, so "
           "a bound can land inside the range (Octatope cut by x0 - x2 <= "
           "-1/2: upper bound 1.2499999999999998 of entry 1, while the member "
           "kernel point (-1, 1, 1) gives 5/4)"))])
def test_min_cost_flow_with_a_negative_edge_holds_the_exact_range(case):
    """``get_ranges(solver='mcf')`` on sets with a negative difference bound
    or a negative UTVPI bound (the cost-shift branch of
    ``_optimize_dcs_mcf``), run in a child process with a time limit of 3 s:
    the range holds the exact range, zero slack. These sets answer; the
    hanging ones are the HEXOCT-10 cells above. Catches a shift correction
    with the wrong flow total (a bound moved inward), and the negative-bound
    row of an Octatope converted with the wrong pair of split variables."""
    s = _negative_bound_set(case)
    exact_lo, exact_hi = _exact_ranges(s)
    lo, hi = _in_child(lambda: s.get_ranges(solver="mcf"), _MCF_LIMIT)
    _assert_range(lo, hi, exact_lo, exact_hi, 0,
                  f"get_ranges(solver='mcf') on the {case} set")


# --------------------------------------------------------------------------- #
# The min-cost-flow range is documented as exact                                #
# --------------------------------------------------------------------------- #
_COST_SHIFT_NOT_EXACT = known_bug(
    9, "get_ranges(solver='mcf'), documented as 'Compute exact ranges', shifts "
       "every edge cost by |min cost| + 1e-10 when the constraint graph has a "
       "negative edge, which changes which flow is cheapest (a path of fewer "
       "edges gains); the value is that of a worse flow, sound and up to 0.25 "
       "wider than the exact range here (Hexatope box cut by x1 - x2 <= -1/4: "
       "upper bound 1.25 of entry 0 for the maximum 1). The comment at "
       "hexatope.py:565 says the optimal flow is unchanged")


def _identity_cut_set(kind, bound):
    """The kernel box [-1, 1]^3 with identity generators and centre
    ``_SMALL_C``, cut by one two-variable row of bound ``bound``: Hexatope
    x1 - x2 <= bound, Octatope x0 + x1 <= bound. A negative bound gives the
    constraint graph a negative edge (the cost-shift branch of
    ``_optimize_dcs_mcf``); a positive one does not."""
    if kind == "Hexatope":
        return _hexatope(_SMALL_C, _IDENTITY, _box_rows(3) + [(1, 2, bound)])
    return _octatope(_SMALL_C, np.eye(3),
                     _OCT_BOX_ROWS + [(0, 1, 1, 1, bound)])


@pytest.mark.parametrize("bound", [
    pytest.param(0.25, id="positive-bound"),
    pytest.param(-0.25, id="negative-bound", marks=_COST_SHIFT_NOT_EXACT)])
@pytest.mark.parametrize("kind", KINDS)
def test_min_cost_flow_range_is_the_exact_range(kind, bound):
    """``get_ranges(solver='mcf')``, run in a child process with a time limit
    of 3 s (HEXOCT-10), is the exact range of the set in both directions
    (``_assert_exact_range``: no slack inside, 1e-12 outside). The neighbour
    with a positive bound has no negative edge and passes with the same
    reference. Catches a min-cost-flow value that is not the optimum: the
    value of the flow found after the cost shift (marked), a flow objective
    not halved for an Octatope's split variables, a centre term dropped."""
    s = _identity_cut_set(kind, bound)
    exact_lo, exact_hi = _exact_ranges(s)
    lo, hi = _in_child(lambda: s.get_ranges(solver="mcf"), _MCF_LIMIT)
    _assert_exact_range(lo, hi, exact_lo, exact_hi, "mcf",
                        f"{kind}.get_ranges(solver='mcf'), cut bound {bound}")


# --------------------------------------------------------------------------- #
# Two rows on the same ordered pair of variables                                #
# --------------------------------------------------------------------------- #
_LAST_ROW_ON_A_PAIR = known_bug(
    9, "DifferenceConstraintSystem.to_constraint_graph adds one networkx DiGraph "
       "edge per row, and add_edge on a pair that has an edge overwrites its "
       "cost, so of several rows on one ordered pair only the last is read: "
       "after the cuts y0 <= 1/2 and then y0 <= 3/2 of one difference, "
       "get_ranges(solver='mcf') gives y0 the upper bound 3/2 for the maximum "
       "1/2, and after y0 <= -3 and then y0 <= 3 is_empty_set calls the empty "
       "set non-empty; the same for an Octatope through to_dcs")


def _two_cuts_on_one_difference(kind, bounds):
    """The box [-1, 1]^2 as ``kind`` mapped to (x1 - x2, x1 + x2), then cut by
    y0 <= b for each b of ``bounds`` in turn. Each cut is the generator-space
    row x_i - x_j <= b, which the direct branch of the bounding box stores as
    it is: the set holds one row per cut on the same ordered pair of
    variables (an Octatope's ``to_dcs`` makes each the same two difference
    rows)."""
    s = flat_set(kind, [-1.0, -1.0], [1.0, 1.0]).affine_map(
        np.array([[1.0, -1.0], [1.0, 1.0]]))
    for b in bounds:
        s = s.intersect_half_space(np.array([[1.0, 0.0]]), np.array([[b]]))
    return s


@pytest.mark.parametrize("order", [
    "tighter-last", pytest.param("tighter-first", marks=_LAST_ROW_ON_A_PAIR)])
@pytest.mark.parametrize("consumer", ["get_ranges-mcf", "is_empty_set"])
@pytest.mark.parametrize("kind", KINDS)
def test_two_cuts_on_one_difference_are_both_read(kind, consumer, order):
    """Two cuts on the same difference, a tight one and a loose one, in either
    order: the set is the intersection of both, whatever the order.
    ``get_ranges-mcf`` (cuts 1/2 and 3/2): the min-cost-flow range is the
    exact range (``_assert_exact_range``, in a child process with the 3 s
    limit). ``is_empty_set`` (cuts -3 and 3, empty by the exact feasibility
    of the rows): it says True. The order with the tight cut last passes
    with the same reference. Catches a constraint graph that keeps one row
    per ordered pair: the last (marked) or the first."""
    tight, loose = (0.5, 1.5) if consumer == "get_ranges-mcf" else (-3.0, 3.0)
    bounds = (loose, tight) if order == "tighter-last" else (tight, loose)
    s = _two_cuts_on_one_difference(kind, bounds)
    what = f"{kind} cut by y0 <= {bounds[0]} and then by y0 <= {bounds[1]}"
    if consumer == "get_ranges-mcf":
        exact_lo, exact_hi = _exact_ranges(s)
        lo, hi = _in_child(lambda: s.get_ranges(solver="mcf"), _MCF_LIMIT)
        _assert_exact_range(lo, hi, exact_lo, exact_hi, "mcf",
                            f"{what}: get_ranges(solver='mcf')")
    else:
        _, _, A, b = _polytope(s)
        if R.feasible((), (), A, b) is not None:
            pytest.fail("test setup: the cut set is not empty")
        assert s.is_empty_set() is True, f"{what}: is_empty_set() is not True"


# --------------------------------------------------------------------------- #
# A flow that cannot be built, and a set without rows                           #
# --------------------------------------------------------------------------- #
_PARALLEL_FALLBACK = known_bug(
    9, "the parallel get_ranges (_get_ranges_parallel) gives an entry whose "
       "solver finds no optimum the value of estimate_range at once, with no "
       "'lp' retry, and estimate_range assumes the kernel box [-1, 1] "
       "(registry HEXOCT-11): (t, 2 t) for t in [-3, 3] with a vacuous row of "
       "infinite bound, which the min-cost flow cannot take, gets the range "
       "[-1, 1] x [-2, 2]")


@pytest.mark.parametrize("parallel", [
    False, pytest.param(True, marks=_PARALLEL_FALLBACK)])
@pytest.mark.parametrize("kind", KINDS)
def test_get_ranges_answers_when_the_min_cost_flow_cannot_be_built(kind,
                                                                   parallel):
    """The set ``(t, 2 t)`` for t in [-3, 3] with one more row of infinite
    bound (Hexatope x_1 - x_0 <= inf, Octatope x_0 <= inf), which changes
    nothing; networkx refuses an edge of infinite weight, so 'mcf' finds no
    optimum. ``get_ranges(solver='mcf')`` must still hold the members
    t = -3, 0, 3 (outputs ``(t, 2 t)``, slack ``LP_TOL``: the answer comes
    from the 'lp' retry). The sequential path retries with 'lp' and passes.
    Catches a sequential path that skips the 'lp' retry and answers with the
    [-1, 1] estimate."""
    s = _single_variable(kind, -3.0, 3.0)
    if kind == "Hexatope":
        s.dcs.add_constraint(1, 0, np.inf)
    else:
        s.utvpi.add_constraint(0, 0, 1, 0, np.inf)
    lo, hi = s.get_ranges(solver="mcf", parallel=parallel, n_workers=2)
    lo, hi = np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)
    for t in (-3.0, 0.0, 3.0):
        y = np.array([t, 2 * t])
        pad = LP_TOL * max(1.0, float(np.max(np.abs(y))))
        assert np.all(lo <= y + pad) and np.all(y - pad <= hi), (
            f"{kind}.get_ranges(solver='mcf', parallel={parallel}) returns "
            f"[{lo.tolist()}, {hi.tolist()}], which leaves out the member "
            f"{y.tolist()}")


_NO_EDGE = known_bug(
    9, "Hexatope._optimize_dcs_mcf takes min() over the edges of the "
       "constraint graph before its try, so for a set with no rows (a point: "
       "its one kernel variable has a zero generator) get_ranges(solver='mcf') "
       "raises 'ValueError: min() iterable argument is empty' where 'lp' "
       "answers the point; an Octatope reaches the same line through to_dcs",
    raises=ValueError)


@pytest.mark.parametrize("solver", ["lp", pytest.param("mcf", marks=_NO_EDGE)])
@pytest.mark.parametrize("kind", KINDS)
def test_ranges_of_a_point_set_without_rows(kind, solver):
    """The point (3/2, -1/2) built with the public constructor: one kernel
    variable (for a Hexatope the anchor) with a zero generator and no row.
    ``get_ranges`` is [3/2, 3/2] x [-1/2, -1/2] in both directions
    (``_assert_exact_range``); the 'lp' neighbour passes. Catches a crash of
    the min-cost flow on a graph with no edge (marked) and a centre term
    dropped by either solver."""
    build = _hexatope if kind == "Hexatope" else _octatope
    s = build([1.5, -0.5], [[0.0], [0.0]], [])
    lo, hi = s.get_ranges(solver=solver)
    point = [Fraction(3, 2), Fraction(-1, 2)]
    _assert_exact_range(lo, hi, point, point, solver,
                        f"{kind} point without rows: get_ranges({solver!r})")


# --------------------------------------------------------------------------- #
# Refusals                                                                       #
# --------------------------------------------------------------------------- #
_PARALLEL_SWALLOWS_REFUSAL = known_bug(
    9, "Hexatope._get_ranges_parallel and Octatope._get_ranges_parallel run "
       "each entry's get_range inside 'except Exception', so the ValueError "
       "of an unknown solver name becomes the [-1, 1] estimate: "
       "get_ranges(solver='highs', parallel=True) answers where the "
       "sequential get_ranges and every other range query refuse",
    raises=DID_NOT_RAISE)

_ESTIMATE_INDEX_CRASH = known_bug(
    9, "Hexatope.estimate_range and Octatope.estimate_range do not check "
       "their index: estimate_range(dim) raises IndexError from numpy where "
       "get_range refuses the same index with ValueError",
    raises=IndexError)

_ESTIMATE_INDEX_WRAPS = known_bug(
    9, "Hexatope.estimate_range and Octatope.estimate_range do not check "
       "their index: estimate_range(-1) answers the range of the last entry "
       "(a numpy negative index) where get_range(-1) refuses with ValueError",
    raises=DID_NOT_RAISE)

_BOUND_OF_ANOTHER_LENGTH = known_bug(
    9, "Hexatope.intersect_half_space and Octatope.intersect_half_space "
       "compute g - H c by numpy broadcasting and cut one row per row of H, "
       "so a bound vector g with more entries than H has rows is answered "
       "with its first entries and the rest dropped, where "
       "Star.intersect_half_space refuses the same arrays with ValueError",
    raises=DID_NOT_RAISE)

_REFUSAL_MARKS = {
    "get_ranges-parallel-unknown-solver": _PARALLEL_SWALLOWS_REFUSAL,
    "estimate_range-index-dim": _ESTIMATE_INDEX_CRASH,
    "estimate_range-index-minus-1": _ESTIMATE_INDEX_WRAPS,
    "intersect_half_space-bound-of-another-length": _BOUND_OF_ANOTHER_LENGTH,
}


def _refused_call(kind, call):
    """Run the call named ``call`` on the cut kernel set of ``kind``, on the
    box [-1, 1]^3 of ``kind`` (the cuts: the row y0 - y1 is a difference row
    there, so no solver runs) or on an empty constraint system of three
    variables."""
    s = _kernel_set(kind)
    n = s.dim
    one_entry = s.affine_map(np.array([[1.0, -0.5, 0.25]]))
    cube = flat_set(kind, -np.ones(3), np.ones(3))
    queries = {
        "get_ranges-unknown-solver": lambda: s.get_ranges(solver="highs"),
        "get_ranges-parallel-unknown-solver": lambda: s.get_ranges(
            solver="highs", parallel=True, n_workers=2),
        "get_ranges-parallel-unknown-solver-one-entry": lambda: (
            one_entry.get_ranges(solver="highs", parallel=True, n_workers=2)),
        "get_range-unknown-solver": lambda: s.get_range(0, solver="highs"),
        "get_bounds-unknown-solver": lambda: s.get_bounds(solver="highs"),
        "get_box-unknown-solver": lambda: s.get_box(solver="highs"),
        "optimize_linear-unknown-solver": lambda: s.optimize_linear(
            np.ones(n), solver="highs"),
        "get_range-index-dim": lambda: s.get_range(n, solver="lp"),
        "get_range-index-minus-1": lambda: s.get_range(-1, solver="lp"),
        "optimize_linear-objective-too-long": lambda: s.optimize_linear(
            np.ones(n + 1), solver="lp"),
        "estimate_range-index-dim": lambda: s.estimate_range(n),
        "estimate_range-index-minus-1": lambda: s.estimate_range(-1),
        "intersect_half_space-row-of-another-width": lambda: (
            cube.intersect_half_space(np.array([[1.0, -1.0, 0.0, 0.0]]),
                                      np.array([[0.5]]))),
        "intersect_half_space-bound-of-another-length": lambda: (
            cube.intersect_half_space(np.array([[1.0, -1.0, 0.0]]),
                                      np.array([[0.5], [-0.5]]))),
    }
    if kind == "Hexatope":
        system = DifferenceConstraintSystem(3)
        rows = {"first-index-3": (3, 0, 1.0), "second-index-3": (0, 3, 1.0),
                "first-index-minus-1": (-1, 0, 1.0)}
    else:
        system = UTVPIConstraintSystem(3)
        rows = {"first-index-3": (3, 0, 1, -1, 1.0),
                "second-index-3": (0, 3, 1, -1, 1.0),
                "first-index-minus-1": (-1, 0, 1, -1, 1.0)}
    for name, row in rows.items():
        queries[f"add_constraint-{name}"] = (
            lambda row=row: system.add_constraint(*row))
    queries["utvpi-coefficient-2"] = lambda: UTVPIConstraint(0, 1, 2, 0, 1.0)
    queries["utvpi-both-coefficients-0"] = lambda: UTVPIConstraint(
        0, 1, 0, 0, 1.0)
    return queries[call]()


def _refusal_cells():
    cells = []
    calls = ["get_ranges-unknown-solver", "get_ranges-parallel-unknown-solver",
             "get_ranges-parallel-unknown-solver-one-entry",
             "get_range-unknown-solver", "get_bounds-unknown-solver",
             "get_box-unknown-solver", "optimize_linear-unknown-solver",
             "get_range-index-dim", "get_range-index-minus-1",
             "optimize_linear-objective-too-long",
             "estimate_range-index-dim", "estimate_range-index-minus-1",
             "intersect_half_space-row-of-another-width",
             "intersect_half_space-bound-of-another-length",
             "add_constraint-first-index-3", "add_constraint-second-index-3",
             "add_constraint-first-index-minus-1"]
    for kind in KINDS:
        for call in calls + (["utvpi-coefficient-2", "utvpi-both-coefficients-0"]
                             if kind == "Octatope" else []):
            marks = _REFUSAL_MARKS.get(call, ())
            cells.append(pytest.param(kind, call, marks=marks,
                                      id=f"{kind}-{call}"))
    return cells


@pytest.mark.parametrize("kind,call", _refusal_cells())
def test_queries_and_constraint_systems_refuse_what_they_cannot_read(kind,
                                                                     call):
    """Every range query given a solver name other than 'lp' and 'mcf', an
    index outside the set, or an objective of the wrong size,
    ``intersect_half_space`` given a row of another width or a bound vector
    of another length than its rows, and every constraint system given a
    variable index outside it or a UTVPI row with a coefficient outside
    {-1, 0, 1} or with both coefficients 0, raises ``ValueError``.
    Marked where today's library answers or crashes instead:
    the parallel ``get_ranges`` of a set of several entries (that of a
    one-entry set takes the sequential path and refuses, its passing
    neighbour), ``estimate_range`` at the index dim (``IndexError``) and -1
    (an answer; ``get_range-index-*`` are the passing neighbours), and a
    bound vector longer than the rows (the row of another width is the
    passing neighbour). Catches an
    index check off by one (an ``IndexError`` later, or a stored row on no
    variable), a query that drops its solver argument, a one-entry set sent
    to the parallel path, and a UTVPI row validation that lets a row
    through."""
    with pytest.raises(ValueError):
        _refused_call(kind, call)


# --------------------------------------------------------------------------- #
# A difference row on one variable (HEXOCT-13)                                #
# --------------------------------------------------------------------------- #
def _one_variable_row_set(b):
    """t in [-1, 1] against the anchor, generator 2, plus the hand-built row
    (1, 1, b), which on one variable reads x_1 - x_1 <= b, that is 0 <= b
    (the constraint graph's self-loop, and the oracle's reading). The library
    never builds such a row itself. ``None`` when the constructor refuses
    it, which also keeps the promise."""
    dcs = DifferenceConstraintSystem(2)
    dcs.add_constraint(1, 0, 1.0)
    dcs.add_constraint(0, 1, 1.0)
    try:
        dcs.add_constraint(1, 1, b)
    except ValueError:
        return None
    return Hexatope(np.array([0.0]), np.array([[0.0, 2.0]]), dcs)


_ONE_VARIABLE_ROW = known_bug(
    9, "DifferenceConstraintSystem.to_matrix_form writes a row with i == j "
       "as -x_i <= b (the +1 is overwritten by -1) while the constraint graph "
       "reads it as 0 <= b; the LP ranges, contains and to_star of such a "
       "Hexatope drop members (row (1, 1, 1/2) on t in [-1, 1], generator 2: "
       "get_ranges('lp') = [-1, 2] for the set [-2, 2])")


@pytest.mark.parametrize("consumer", [
    "get_ranges-mcf", "is_empty_set",
    pytest.param("get_ranges-lp", marks=_ONE_VARIABLE_ROW),
    pytest.param("contains", marks=_ONE_VARIABLE_ROW),
    pytest.param("to_star", marks=_ONE_VARIABLE_ROW)])
def test_a_one_variable_difference_row_keeps_every_member(consumer):
    """The set [-2, 2] carrying the vacuous row (1, 1, 1/2): every consumer
    of the stored rows must keep the members -2, -3/2, 0 and 2. The
    min-cost-flow ranges and ``is_empty_set`` read the row as the constraint
    graph does and pass."""
    s = _one_variable_row_set(0.5)
    if s is None:
        return
    members = [-2.0, -1.5, 0.0, 2.0]
    what = f"Hexatope with the row (1, 1, 1/2): {consumer}"
    if consumer.startswith("get_ranges"):
        solver = consumer.split("-")[1]
        lo, hi = s.get_ranges(solver=solver)
        _assert_range(lo, hi, [Fraction(-2)], [Fraction(2)],
                      LP_TOL if solver == "lp" else 0.0, what)
    elif consumer == "is_empty_set":
        assert s.is_empty_set() is False, f"{what}: a non-empty set is empty"
    elif consumer == "contains":
        missed = [t for t in members if not s.contains(np.array([t]))]
        assert not missed, f"{what}: members {missed} reported outside"
    else:
        star = s.to_star()
        missed = [t for t in members if distance(star, [t]) > LP_TOL]
        assert not missed, f"{what}: members {missed} are not in to_star()"


_EMPTY_BUT_CONTAINS = known_bug(
    9, "a Hexatope whose hand-built row (1, 1, -1) reads 0 <= -1 is empty "
       "(is_empty_set says so), yet contains([2.0]) is True, because "
       "to_matrix_form reads the row as -x_1 <= -1")


@pytest.mark.parametrize("consumer", [
    "is_empty_set", pytest.param("contains", marks=_EMPTY_BUT_CONTAINS)])
def test_an_infeasible_one_variable_difference_row_empties_the_set(consumer):
    """Rows t in [-5, 5] and (1, 1, -1): the oracle reads the last as
    0 <= -1, so the set is empty. ``is_empty_set`` must say True and
    ``contains`` must reject every point; the two answers agree."""
    dcs = DifferenceConstraintSystem(2)
    dcs.add_constraint(1, 0, 5.0)
    dcs.add_constraint(0, 1, 5.0)
    try:
        dcs.add_constraint(1, 1, -1.0)
    except ValueError:
        return
    s = Hexatope(np.array([0.0]), np.array([[0.0, 1.0]]), dcs)
    if distance(s, [2.0]) != float("inf"):
        pytest.fail("test setup: the oracle does not read the set as empty")
    if consumer == "is_empty_set":
        assert s.is_empty_set() is True, "an empty Hexatope is not empty"
    else:
        accepted = [t for t in (-5.0, 0.0, 2.0, 5.0)
                    if s.contains(np.array([t]))]
        assert not accepted, (f"an empty Hexatope contains {accepted}")


# --------------------------------------------------------------------------- #
# intersect_half_space, each branch of the bounding box                         #
# --------------------------------------------------------------------------- #
def _cut_reference(s, H, g):
    """Vertices of the exact intersection of ``s`` with ``H y <= g``: the
    stored rows, the finite bounds and ``H (c + G a) <= g`` on the predicate
    variables, solved in rationals and mapped."""
    c, G, A, b = _polytope(s)
    for row, rhs in zip(H, g):
        row = [Fraction(float(v)) for v in row]
        A = A + [[R.dot(row, [G[i][j] for i in range(len(G))])
                  for j in range(len(G[0]))]]
        b = b + [Fraction(float(rhs)) - R.dot(row, c)]
    verts = R.vertices(A, b)
    if not verts:
        pytest.fail("test setup: the intersection is empty")
    return [tuple(R.dot(row, a) + ci for row, ci in zip(G, c)) for a in verts]


# Half-spaces on the box [-1/2, 3/2] x [-3/4, 5/4] x [0, 1/2] (half-widths 1,
# 1, 1/4): the generator-space row of each is named in the id.
_LO_CUT, _HI_CUT = np.array([-0.5, -0.75, 0.0]), np.array([1.5, 1.25, 0.5])
_CUTS = {
    "difference": ([[1.0, -1.0, 0.0]], [0.25]),            # x1 - x2
    "scaled-difference": ([[-2.0, 2.0, 0.0]], [0.5]),       # 2 (x2 - x1)
    # Scale 1/2 with a positive bound and scale 2 with a negative bound: a
    # bound not divided by its scale is then below the true one.
    "difference-half-scale": ([[0.5, -0.5, 0.0]], [0.25]),  # (x1 - x2) / 2
    "difference-double-scale-negative": ([[-2.0, 2.0, 0.0]], [-1.5]),
    "one-variable-scaled": ([[0.0, 0.0, 4.0]], [1.5]),      # x3
    "sum-scaled": ([[-0.5, -0.5, 0.0]], [-0.125]),          # -(x1 + x2) / 2
    "general": ([[1.0, 0.5, -2.0]], [0.75]),
    "two-rows": ([[1.0, -1.0, 0.0], [0.5, 0.0, 4.0]], [0.25, 1.0]),
    # The same two rows in the other order: the later row's own bound in
    # generator space (0) is now above the first one's (-1/4), so a later
    # row read with the first row's bound drops members.
    "two-rows-reversed": ([[0.5, 0.0, 4.0], [1.0, -1.0, 0.0]], [1.0, 0.25]),
}
# Exact branches: the row is added as it is, no solver.
_EXACT_CUT = {
    "Hexatope": {"difference", "scaled-difference", "difference-half-scale",
                 "difference-double-scale-negative"},
    "Octatope": {"difference", "scaled-difference", "difference-half-scale",
                 "difference-double-scale-negative", "one-variable-scaled",
                 "sum-scaled"},
}


@pytest.mark.parametrize("cut", list(_CUTS))
@pytest.mark.parametrize("kind", KINDS)
def test_intersect_half_space_holds_the_exact_intersection(kind, cut):
    """``intersect_half_space`` of a box with one or two half-spaces: every
    vertex of the exact intersection is a member of the result. Exactly for a
    row the bounding box adds as it is (a difference row, scaled or not;
    for an Octatope also a scaled one-variable row and a scaled sum); by the
    LP oracle for a general row (its bounds come from OSQP). Catches a
    difference row added with its variables exchanged (the piece keeps the
    wrong side), a scaled row not divided by its scale (seen where the
    division loosens nothing: scale 1/2 with a positive bound, scale 2 with
    a negative one), H c added instead of subtracted, a bounding-box LP that
    minimises when asked to maximise, a two-variable bound stored on the
    exchanged pair or with its two signs exchanged, a half-space imposed as
    >=, and a later row cut with the bound of the first (``two-rows-reversed``:
    the order matters). A later row that is ignored only loosens the result
    (an outer approximation, as documented) and is not caught here."""
    H, g = _CUTS[cut]
    s = flat_set(kind, _LO_CUT, _HI_CUT)
    out = s.intersect_half_space(np.array(H), np.array(g).reshape(-1, 1))
    verts = _cut_reference(s, H, g)
    exact = cut in _EXACT_CUT[kind]
    if exact:
        member = R.Membership(R.star(out))
        missed = [v for v in verts if not member(v)]
    else:
        missed = [v for v in verts if distance(out, [float(x) for x in v])
                  > LP_TOL * max(1.0, max(abs(float(x)) for x in v))]
    assert not missed, (
        f"{kind}.intersect_half_space({cut}): {len(missed)} of {len(verts)} "
        f"vertices of the exact intersection are not in the result; first "
        f"{[float(x) for x in missed[0]]}")


_NEARLY_EQUAL = known_bug(
    9, "Hexatope._dcs_bounding_box and Octatope._utvpi_bounding_box take a "
       "half-space whose two generator coefficients have magnitudes equal "
       "only up to np.isclose / np.allclose (relative 1e-5) for an exact "
       "difference or UTVPI row and divide by one of them, which drops part "
       "of the intersection (y1 - (1 + 8e-6) y2 <= 0 on [-8, 8]^2: the member "
       "(8, 8 / (1 + 8e-6)) is 3.2e-5 away)")


_NEARLY_ONE_FLOW = known_bug(
    9, "Hexatope._optimize_with_constraint and Octatope._optimize_with_constraint "
       "take a cut row whose two generator coefficients are -1 and 1 only up to "
       "np.allclose / np.isclose (relative 1e-5) for the exact row and solve "
       "the bounding box of another set by min-cost flow; this branch is "
       "reached when the direct branch of the bounding box refuses the row "
       "(coefficients 1 - 8e-6 and -(1 + 8e-6) on [-8, 8]^2: the member "
       "vertex near (8, 8) is 6.4e-5 away), and the fix proposed for "
       "HEXOCT-14 sends more rows here")


@pytest.mark.parametrize("row", [
    pytest.param([1.0, -1.0], id="equal"),
    pytest.param([1.0, -(1.0 + 8e-6)], id="relative-8e-6",
                 marks=_NEARLY_EQUAL),
    pytest.param([(1.0 - 8e-6) / 8, -(1.0 + 8e-6) / 8],
                 id="both-8e-6-off-one", marks=_NEARLY_ONE_FLOW)])
@pytest.mark.parametrize("kind", KINDS)
def test_intersect_half_space_with_nearly_equal_coefficients(kind, row):
    """The square [-8, 8]^2 (half-width 8) cut by ``row y <= 0``: every vertex
    of the exact intersection (rationals) is in the result, by the LP oracle.
    ``equal`` is a difference row and the cut is exact. ``relative-8e-6``
    reaches the direct branch of the bounding box (HEXOCT-14).
    ``both-8e-6-off-one`` has generator-space coefficients 1 - 8e-6 and
    -(1 + 8e-6): the direct branch refuses it (their ratio is 1.6e-5 from 1)
    and the general branch hands it to ``_optimize_with_constraint``, whose
    min-cost-flow branch takes it for x_i - x_j. The cut runs in a child
    process with a time limit of 6 s: a min-cost flow with a negative edge
    can hang (HEXOCT-10), and a case must fail rather than block the run."""
    s = flat_set(kind, np.full(2, -8.0), np.full(2, 8.0))
    H, g = [row], [0.0]
    out = _in_child(lambda: s.intersect_half_space(np.array(H),
                                                   np.array([[0.0]])),
                    _MCF_RELU_LIMIT)
    verts = _cut_reference(s, H, g)
    missed = [v for v in verts if distance(out, [float(x) for x in v])
              > LP_TOL * max(1.0, max(abs(float(x)) for x in v))]
    assert not missed, (
        f"{kind}: {len(missed)} vertices of the square cut by "
        f"{row} y <= 0 are not in the result; first "
        f"{[float(x) for x in missed[0]]}")


# A neuron y0 = a x0 - b x1 on the box [-1, 1]^2 (half-width 1, so its
# generator-space row is (a, -b)) and y1 = x0 + 2, which pins x0.
_NEARLY_DIFFERENCE = {"equal": (1.0, 1.0), "relative-8e-6": (1.0, 1.0 + 8e-6),
                      "both-8e-6-off-one": (1.0 - 8e-6, 1.0 + 8e-6)}


@pytest.mark.parametrize("coefficients", [
    "equal", pytest.param("relative-8e-6", marks=_NEARLY_EQUAL),
    pytest.param("both-8e-6-off-one", marks=_NEARLY_ONE_FLOW)])
def test_relu_of_a_nearly_difference_neuron_keeps_every_model_output(
        coefficients):
    """Linear (y0 = a x0 - b x1, y1 = x0 + 2, float32 weights) then ReLU on
    the Hexatope [-1, 1]^2 through ``reach_pytorch_model``: 128 sampled
    member points, and three points just on the active side of y0 = 0 where
    x0 - x1 < 0 (x1 = -3/4, -1/2, -1/4 and x0 halfway between b x1 / a and
    x1), replayed on torch in float64, are covered. With ``equal`` the cut
    is exact. With the others the split takes the cut for x0 - x1 >= 0
    (HEXOCT-14 in the direct branch of the bounding box; the min-cost-flow
    branch of ``_optimize_with_constraint`` for ``both-8e-6-off-one``) and
    the outputs of those points, which pin the input, are lost by about 2e-6
    and 4e-6: the consequence on the model entry of the two set-level cells
    above. The reach runs in a child process with a time limit of 6 s (the
    min-cost-flow branch can hang, HEXOCT-10)."""
    a, b = (float(np.float32(v)) for v in _NEARLY_DIFFERENCE[coefficients])
    W1, b1 = [[a, -b], [1.0, 0.0]], [0.0, 2.0]
    model = nn.Sequential(_linear(W1, b1), nn.ReLU())
    lo, hi = np.full(2, -1.0), np.full(2, 1.0)
    what = f"Linear-ReLU of a nearly-difference neuron ({coefficients})"
    out = _in_child(lambda: reach_model(
        model, Hexatope.from_bounds(lo, hi), "approx", what), _MCF_RELU_LIMIT)
    edge = [np.array([(b * x1 / a + x1) / 2, x1]) for x1 in (-0.75, -0.5, -0.25)]
    assert_covered(out, replay_torch(model, samples(lo, hi) + edge, (2,)), what)


# --------------------------------------------------------------------------- #
# ReLU: the split by difference rows (Hexatope) and by general rows             #
# --------------------------------------------------------------------------- #
# Hexatope input [-1/2, 1/2] x [1/4, 5/4] x [-1, 0], half-widths 1/2 each.
# Neuron 0 = x0 - x1 + 1/2 in [-5/4, 3/4] and neuron 3 = -x0 + x2 + 1/4 in
# [-5/4, 3/4] cross zero with generator rows of two opposite equal entries,
# so each split adds a difference row as it is; neuron 1 is stably active,
# neuron 2 stably inactive.
_DIFF_LO, _DIFF_HI = np.array([-0.5, 0.25, -1.0]), np.array([0.5, 1.25, 0.0])
_DIFF_W1 = [[1.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0]]
_DIFF_B1 = [0.5, 0.0, -1.5, 0.25]
_DIFF_W2 = [[1.0, -0.5, 0.25, -1.0], [-0.5, 1.0, 0.5, 0.75]]
_DIFF_B2 = [0.125, -0.25]


def _relu_reference(lo, hi, W1, b1, W2=None, b2=None):
    pieces = R.relu(R.affine(R.box(lo, hi), W1, b1))
    return pieces if W2 is None else R.affine(pieces, W2, b2)


@pytest.mark.parametrize("entry", ["layer", "model"])
def test_relu_split_by_difference_rows_holds_the_exact_image(entry):
    """Linear then ReLU on a Hexatope through ``reach_layer``, and Linear,
    ReLU, Linear through ``reach_pytorch_model``: two crossing neurons whose
    generator rows are +w and -w on inputs of equal width, so both splits add
    a difference row directly, with no solver. Every piece of the exact image
    is in one returned set, exactly (zero slack); member points replayed on
    torch in float64 are covered. Catches the difference row added with its
    two variables exchanged (the active piece keeps the negative side), the
    half-space moved by +H c, a cut moved inward, and a second split that
    reads the first split's row in mirror image. A bound not divided by its
    scale is not caught here (the outputs the inactive
    piece loses lie in the active one); the cell that pins the outputs,
    ``test_relu_split_by_a_scaled_difference_row_keeps_every_pinned_output``,
    catches it."""
    lo, hi = _DIFF_LO, _DIFF_HI
    s = Hexatope.from_bounds(lo, hi)
    what = f"Linear-ReLU on a Hexatope split by difference rows ({entry})"
    if entry == "layer":
        mid = reach_one_layer(_linear(_DIFF_W1, _DIFF_B1), [s], "approx", what)
        out = reach_one_layer(nn.ReLU(), mid, "approx", what)
        pieces = _relu_reference(lo, hi, _DIFF_W1, _DIFF_B1)
        model = nn.Sequential(_linear(_DIFF_W1, _DIFF_B1), nn.ReLU())
    else:
        model = nn.Sequential(_linear(_DIFF_W1, _DIFF_B1), nn.ReLU(),
                              _linear(_DIFF_W2, _DIFF_B2))
        out = reach_model(model, s, "approx", what)
        pieces = _relu_reference(lo, hi, _DIFF_W1, _DIFF_B1, _DIFF_W2,
                                 _DIFF_B2)
    _assert_image_held(pieces, out, what, exact=True)
    assert_covered(out, replay_torch(model, samples(lo, hi, n=48), (3,)), what)


# Neuron 0 = x0 - x1 + 1/4 on the box of the cell above crosses zero with the
# generator row (1/2, -1/2), a difference row of scale 1/2 whose inactive cut
# has a positive bound (1/2, stored as 1 after the division). Neurons 1 to 3
# are the inputs moved to be positive, stably active: the outputs pin the
# input, so an output of the inactive piece that a cut loses is in no other
# piece.
_PIN_W1 = [[1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
           [0.0, 0.0, 1.0]]
_PIN_B1 = [0.25, 1.0, 0.0, 2.0]


@pytest.mark.parametrize("entry", ["layer", "model"])
@pytest.mark.parametrize("kind", KINDS)
def test_relu_split_by_a_scaled_difference_row_keeps_every_pinned_output(
        kind, entry):
    """Linear then ReLU through ``reach_layer`` (one layer at a time) and
    through ``reach_pytorch_model``: one crossing neuron that is a difference
    of two inputs with generator coefficients 1/2 and -1/2, and three outputs
    that are the inputs. Every piece of the exact image is in one returned
    set, exactly (zero slack: the split adds the row with no solver), and 128
    member points replayed on torch in float64 are covered. Catches the bound
    of a scaled difference row not divided by its scale (without outputs that
    pin the input, the outputs the inactive piece loses lie in the active
    one), the row's variables exchanged, and H c added instead of
    subtracted."""
    lo, hi = _DIFF_LO, _DIFF_HI
    s = flat_set(kind, lo, hi)
    model = nn.Sequential(_linear(_PIN_W1, _PIN_B1), nn.ReLU())
    what = (f"Linear-ReLU on a {kind} split by a scaled difference row, "
            f"outputs pinned ({entry})")
    if entry == "layer":
        mid = reach_one_layer(model[0], [s], "approx", what)
        out = reach_one_layer(model[1], mid, "approx", what)
    else:
        out = reach_model(model, s, "approx", what)
    _assert_image_held(_relu_reference(lo, hi, _PIN_W1, _PIN_B1), out, what,
                       exact=True)
    assert_covered(out, replay_torch(model, samples(lo, hi), (3,)), what)


# Input [-1, 1] x [-1/2, 3/2] x [1/4, 3/4]. Neurons 0 = x0 + x1/2 + 1/4 and
# 1 = x0/2 - x1 cross zero with generator rows of two entries of different
# size (the general bounding box, OSQP); the second split reads the first
# split's two-variable rows. Neuron 2 = x2 is stably active; neuron 3 =
# -x0/2 + 2 x2 - 5/4 crosses with a scaled UTVPI row.
_GEN_LO, _GEN_HI = np.array([-1.0, -0.5, 0.25]), np.array([1.0, 1.5, 0.75])
_GEN_W1 = [[1.0, 0.5, 0.0], [0.5, -1.0, 0.0], [0.0, 0.0, 1.0],
           [-0.5, 0.0, 2.0]]
_GEN_B1 = [0.25, 0.0, 0.0, -1.25]
_GEN_W2 = [[1.0, -1.0, 0.5, 0.25], [-0.5, 0.5, 0.0, 1.0]]
_GEN_B2 = [0.0, 0.125]


@pytest.mark.parametrize("entry", ["layer", "model"])
@pytest.mark.parametrize("kind", KINDS)
def test_relu_split_by_general_rows_holds_the_exact_image(kind, entry):
    """Linear then ReLU (``reach_layer``) and Linear, ReLU, Linear (model
    entry) with two crossing neurons that mix two inputs with different
    weights and one that is a scaled UTVPI row: every piece of the exact
    image is in one returned set (LP oracle, ``LP_TOL``), and member points
    replayed on torch in float64 are covered. Catches a bounding-box LP that
    minimises when asked to maximise (pieces far too small), the two-variable
    bounds stored on the exchanged pair of variables, the matrix row of a
    two-variable constraint written into the wrong column (the second split
    then works on a wrong kernel), the cut imposed as >=, and the half-space
    moved by +H c."""
    lo, hi = _GEN_LO, _GEN_HI
    s = flat_set(kind, lo, hi)
    what = f"Linear-ReLU on a {kind} split by general rows ({entry})"
    if entry == "layer":
        mid = reach_one_layer(_linear(_GEN_W1, _GEN_B1), [s], "approx", what)
        out = reach_one_layer(nn.ReLU(), mid, "approx", what)
        pieces = _relu_reference(lo, hi, _GEN_W1, _GEN_B1)
        model = nn.Sequential(_linear(_GEN_W1, _GEN_B1), nn.ReLU())
    else:
        model = nn.Sequential(_linear(_GEN_W1, _GEN_B1), nn.ReLU(),
                              _linear(_GEN_W2, _GEN_B2))
        out = reach_model(model, s, "approx", what)
        pieces = _relu_reference(lo, hi, _GEN_W1, _GEN_B1, _GEN_W2, _GEN_B2)
    _assert_image_held(pieces, out, what, exact=False)
    assert_covered(out, replay_torch(model, samples(lo, hi, n=48), (3,)), what)


@pytest.mark.parametrize("kind", KINDS)
def test_relu_with_lp_phases_holds_the_image(kind):
    """Linear then ReLU of the cell above through ``reach_layer``, the ReLU
    with ``solver='lp'``: each neuron's phase then comes from
    ``get_ranges(solver='lp')`` (padded, in ``_kernel_phase_ranges``) and
    not from the [-1, 1] estimate. Every piece of the exact image is in one
    returned set (LP oracle, ``LP_TOL``) and 128 member points replayed on
    torch in float64 are covered. Catches an 'lp' range that is too narrow
    on the public route that uses it: a crossing neuron taken for a stable
    one loses the outputs on its other side."""
    lo, hi = _GEN_LO, _GEN_HI
    s = flat_set(kind, lo, hi)
    what = f"Linear-ReLU on a {kind}, ReLU phases from get_ranges('lp')"
    mid = reach_one_layer(_linear(_GEN_W1, _GEN_B1), [s], "approx", what)
    out = reach_one_layer(nn.ReLU(), mid, "approx", what, solver="lp")
    model = nn.Sequential(_linear(_GEN_W1, _GEN_B1), nn.ReLU())
    _assert_image_held(_relu_reference(lo, hi, _GEN_W1, _GEN_B1), out, what,
                       exact=False)
    assert_covered(out, replay_torch(model, samples(lo, hi), (3,)), what)


# --------------------------------------------------------------------------- #
# contains, is_empty_set, Octatope.sample                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", KINDS)
def test_contains_agrees_with_the_rows(kind):
    """``contains`` on the cut kernel set: True for every vertex and for the
    midpoints of pairs of vertices (members, exactly), False for points
    pushed 1/4 beyond a vertex along the direction away from the set's
    centroid that the exact oracle puts outside by at least 1/16. Catches a
    membership test that ignores a cut row (a bounding-box test) or reads a
    row with the wrong sign (the matrix form of a difference or UTVPI row
    written into the wrong columns). A membership test that leaves the
    Hexatope anchor free accepts none of these points;
    ``test_hexatope_contains_rejects_a_point_off_the_anchor`` catches it."""
    s = _kernel_set(kind)
    verts = _vertices(s)
    member = R.Membership(R.star(s))
    inside = [list(v) for v in verts] + [
        [(a + b) / 2 for a, b in zip(u, w)]
        for u, w in itertools.combinations(verts, 2)][:20]
    for p in inside:
        if not member(p):
            pytest.fail("test setup: a midpoint of vertices is not a member")
    missed = [p for p in inside
              if not s.contains(np.array([float(v) for v in p]))]
    assert not missed, (f"{kind}.contains rejects {len(missed)} members; "
                        f"first {[float(v) for v in missed[0]]}")
    centroid = [sum(v[i] for v in verts) / len(verts) for i in range(s.dim)]
    outside = []
    for v in verts:
        step = [x - m for x, m in zip(v, centroid)]
        size = max(abs(x) for x in step)
        p = [float(x + Fraction(1, 4) * d / size) for x, d in zip(v, step)]
        if distance(s, p) >= 1 / 16:
            outside.append(p)
    if len(outside) < 3:
        pytest.fail("test setup: too few points outside the set")
    accepted = [p for p in outside if s.contains(np.array(p))]
    assert not accepted, (f"{kind}.contains accepts {len(accepted)} points "
                          f"at least 1/16 outside the set; first {accepted[0]}")


def test_hexatope_contains_rejects_a_point_off_the_anchor():
    """``Hexatope.contains`` on the cut kernel set at the points
    c + G (a + t 1), for every vertex a of the kernel polytope (anchor 0) and
    t = 1/4 and -1/4: the kernel point moved by t in every variable, the
    anchor included, still satisfies every difference row, so only the
    anchor x_0 = 0 puts it outside. Each such point that the exact oracle
    puts at least 1/16 outside must be rejected. Catches a membership LP
    that leaves the anchor free."""
    s = _kernel_set("Hexatope")
    c, G, A, b = _polytope(s)
    outside = []
    for a in R.vertices(A, b):
        for t in (Fraction(1, 4), Fraction(-1, 4)):
            y = [float(ci + R.dot(row, [v + t for v in a]))
                 for ci, row in zip(c, G)]
            if distance(s, y) >= 1 / 16:
                outside.append(y)
    if len(outside) < 3:
        pytest.fail("test setup: too few moved points outside the set")
    accepted = [y for y in outside if s.contains(np.array(y))]
    assert not accepted, (
        f"Hexatope.contains accepts {len(accepted)} of {len(outside)} points "
        f"that only the anchor puts outside; first {accepted[0]}")


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_is_empty_set_agrees_with_the_rows(kind, empty):
    """``is_empty_set`` on the cut kernel set, and on the same set with one
    more row that contradicts the others (Hexatope x1 - x0 <= -7/8 against
    x1 >= -3/4; Octatope -x0 - x1 <= -2 against the box): the exact
    feasibility of the rows decides. Catches a test that never finds a
    negative cycle or always finds one, and the row -x_i - x_j <= b converted
    to the constraint graph as x_i + x_j <= b (the Octatope's contradiction
    then disappears). A contradiction that no edge from node 0 reaches, and a
    cycle of cost 0, are in
    ``test_is_empty_set_away_from_variable_zero_and_on_a_flat_set``."""
    s = _kernel_set(kind)
    if empty:
        if kind == "Hexatope":
            s.dcs.add_constraint(1, 0, -0.875)
        else:
            s.utvpi.add_constraint(0, 1, -1, -1, -2.0)
    _, _, A, b = _polytope(s)
    truth = R.feasible((), (), A, b) is None
    if truth != empty:
        pytest.fail("test setup: the exact feasibility is not as intended")
    assert s.is_empty_set() is truth, (
        f"{kind}.is_empty_set() is {s.is_empty_set()} for a set that is "
        f"{'empty' if truth else 'not empty'}")


def _empty_away_from_variable_zero(kind):
    """An empty set whose contradicting rows involve only variables that no
    edge of the constraint graph leaves node 0 for. Hexatope: x1 >= -1 and
    x2 >= -1 against the anchor (edges into node 0 only), x1 - x2 <= -1 and
    x2 - x1 <= 0. Octatope: x0 in [-1, 1], x1 <= -1 and -x1 <= 0."""
    if kind == "Hexatope":
        return _hexatope([0.0, 0.0], [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                         [(0, 1, 1.0), (0, 2, 1.0), (1, 2, -1.0),
                          (2, 1, 0.0)])
    return _octatope([0.0, 0.0], np.eye(2),
                     [(0, 0, 1, 0, 1.0), (0, 0, -1, 0, 1.0),
                      (1, 1, 1, 0, -1.0), (1, 1, -1, 0, 0.0)])


def _set_on_a_line(kind):
    """The box [-1, 1]^2 cut by y0 - y1 <= 0 and by y1 - y0 <= 0: the
    diagonal, not empty, with a cycle of cost 0 in its constraint graph."""
    s = flat_set(kind, [-1.0, -1.0], [1.0, 1.0])
    for H in ([[1.0, -1.0]], [[-1.0, 1.0]]):
        s = s.intersect_half_space(np.array(H), np.array([[0.0]]))
    return s


@pytest.mark.parametrize("case", ["contradiction-away-from-variable-0",
                                  "two-variables-forced-equal"])
@pytest.mark.parametrize("kind", KINDS)
def test_is_empty_set_away_from_variable_zero_and_on_a_flat_set(kind, case):
    """``is_empty_set`` (``DifferenceConstraintSystem.is_feasible``, for an
    Octatope through ``UTVPIConstraintSystem.is_feasible`` and ``to_dcs``)
    against the exact feasibility of the rows: an empty set whose
    contradiction no edge from node 0 reaches, and a
    set flattened onto a line by two cuts, whose rows form a cycle of cost
    0. Catches a negative-cycle search from node 0 only (an empty set called
    non-empty), and one that counts a cycle of cost 0 as negative (a set
    that is not empty called empty: the soundness direction)."""
    empty = case == "contradiction-away-from-variable-0"
    s = _empty_away_from_variable_zero(kind) if empty else _set_on_a_line(kind)
    _, _, A, b = _polytope(s)
    truth = R.feasible((), (), A, b) is None
    if truth != empty:
        pytest.fail("test setup: the exact feasibility is not as intended")
    assert s.is_empty_set() is truth, (
        f"{kind}.is_empty_set() is {s.is_empty_set()} for a set that is "
        f"{'empty' if truth else 'not empty'} ({case})")


_INFINITE_NEGATIVE_CYCLE = known_bug(
    9, "DifferenceConstraintSystem.is_feasible asks networkx.negative_edge_cycle, "
       "which does not report a cycle through an edge of cost -inf, so a "
       "Hexatope or an Octatope cut by a row of bound -inf (the box [-1, 1]^2 "
       "cut by y0 - y1 <= -inf, stored as x_i - x_j <= -inf, which no point "
       "satisfies) is called non-empty")


@pytest.mark.parametrize("bound", [
    pytest.param(np.inf, id="plus-infinity"),
    pytest.param(-np.inf, id="minus-infinity", marks=_INFINITE_NEGATIVE_CYCLE)])
@pytest.mark.parametrize("kind", KINDS)
def test_is_empty_set_of_a_cut_with_an_infinite_bound(kind, bound):
    """``is_empty_set`` after ``intersect_half_space`` of the box [-1, 1]^2
    with y0 - y1 <= ``bound``, which the direct branch stores as the row
    x_i - x_j <= ``bound``. The truth is read off the row by hand (the
    rational oracle reads no infinite bound): with +inf it binds nothing and
    the set is the box, not empty (the passing neighbour); with -inf no point
    satisfies it and the set is empty. Catches an emptiness test that drops
    or misreads a row of infinite bound (the input regime of a missing or
    infinite bound)."""
    s = flat_set(kind, np.full(2, -1.0), np.full(2, 1.0))
    s = s.intersect_half_space(np.array([[1.0, -1.0]]), np.array([[bound]]))
    rows = s.dcs.constraints if kind == "Hexatope" else s.utvpi.constraints
    if rows[-1].b != bound:
        pytest.fail("test setup: the cut is not stored as one row of that bound")
    empty = bound < 0
    assert s.is_empty_set() is empty, (
        f"{kind} cut by y0 - y1 <= {bound}: is_empty_set() is "
        f"{s.is_empty_set()} for a set that is {'empty' if empty else 'the box'}")


def test_octatope_sample_returns_members():
    """``Octatope.sample`` on the cut kernel set (two-variable rows of every
    sign pattern): every sample is a member by the LP oracle (``LP_TOL``; the
    sampler's own rejection test admits 1e-10 in the variables). Catches a
    rejection test that ignores two-variable rows or reads them with the
    wrong sign. The Hexatope sampler is marked in
    ``contracts/test_set_protocol.py`` (registry HEXOCT-2)."""
    s = _kernel_set("Octatope")
    points = s.sample(64)
    if len(points) == 0:
        pytest.fail("test setup: no samples")
    outside = [p for p in points if distance(s, p) > LP_TOL]
    assert not outside, (f"{len(outside)} of {len(points)} samples are not in "
                         f"the Octatope; first {np.asarray(outside[0]).tolist()}")
