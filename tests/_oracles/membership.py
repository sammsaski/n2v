"""Is a float point a member of a computed set? One LP on the set's own rows.

Point replay (PROTOCOL.md section 3.3) needs membership, not a bounding box: a
relaxation whose box is right and whose rows are wrong passes every box check.

Reading a set
-------------
:func:`rows` reads a computed set as ``{c + G a : C a <= d, lb <= a <= ub}``
from the arrays it stores, for every set type n2v returns. No method of the
set is called.

``Star``, ``ImageStar``
    ``V = [c, G]``, the rows ``C a <= d`` and the predicate bounds as stored;
    an absent bound is infinite. A star with no predicate variable keeps its
    rows on no variable (``0 <= d``), so a negative entry of ``d`` empties it.
``Zono``, ``ImageZono``
    ``c + G a`` with ``a`` in the cube ``[-1, 1]``.
``Box``
    ``c = 0``, ``G = I`` and ``a`` between the stored bounds, so the exact
    oracle (``rational.star``) reads a box too; :func:`distance` keeps a direct
    branch for it.
``Hexatope``
    ``center + generators x`` with one row ``x_i - x_j <= b`` per stored
    ``DifferenceConstraint(i, j, b)`` and the anchor variable ``x_0`` pinned
    to 0 (``lb[0] = ub[0] = 0``, whatever its generator column holds); every
    other variable is bounded by the rows alone, so a hexatope whose rows do
    not bound a variable is an unbounded set. This is the meaning of
    ``from_bounds`` (the anchor stands for 0), of the LP path
    ``_optimize_dcs_lp`` and of ``contains``. It is not the meaning of
    ``to_star`` or ``sample``, which leave ``x_0`` free in ``[-1, 1]``. A
    constraint with ``i == j`` is ``0 <= b``.
``Octatope``
    ``center + generators x`` with one row ``a_i x_i + a_j x_j <= b`` per
    stored ``UTVPIConstraint(i, j, a_i, a_j, b)``, coefficients in
    ``{-1, 0, 1}``. A constraint with ``i == j`` adds both coefficients to the
    one variable, ``(a_i + a_j) x_i <= b``: ``(i, i, 1, 0, b)`` is ``x_i <= b``,
    ``(i, i, 1, 1, b)`` is ``2 x_i <= b`` and ``(i, i, 1, -1, b)`` is
    ``0 <= b``. That is the reading of ``UTVPIConstraintSystem.to_matrix_form``
    and of ``to_dcs`` (which turns ``x_i <= b`` into ``x+_i - x-_i <= 2 b``
    with ``x_i = (x+_i - x-_i) / 2``). There is no anchor and no variable is
    bounded except by the rows.

Image sets are read in the layout they store, height by width by channel, so
the caller passes the point in that layout (:func:`hwc`).

The distance
------------
For a point ``y``, :func:`distance` returns the infinity-norm distance to the
set, the optimum of

    min t   subject to   -t <= c + G a - y <= t,  C a <= d,  lb <= a <= ub,

solved by scipy's HiGHS called directly, so no n2v solver wrapper, range
method or ``contains`` is trusted; ``inf`` when the set is empty.

The LP is solved in scaled variables. A predicate variable with two finite
bounds is mapped to ``[0, 1]``, a pinned one (``lb == ub``) is replaced by its
value, a variable with an open side keeps its finite bound as origin and is
scaled by a power of two that brings its largest coefficient near 1; every
row of ``C`` and the distance rows are scaled by powers of two to a largest
coefficient near 1. The solver runs at feasibility tolerance
:data:`SOLVER_TOL` in those units, so its error is small against the set
however narrow the predicate box is. (The unscaled LP, at HiGHS's default
1e-7 on the raw variables, called 210 of 300 member points of a star with a
predicate box of width 2e-7 and generators of size 1e6 non-members, at
reported distances up to 0.13.)

The answer is not the solver's objective. From the solver's point the oracle
takes a predicate point ``a``, clips it into ``[lb, ub]`` so that the bounds
hold exactly, and checks every row itself: ``C_k a - d_k``, plus a bound on
the float64 rounding of that evaluation, must be at most :data:`ROW_TOL` times
the row's scale times ``max(1, |s|)``. The row's scale is the largest change
one variable can make in the row over its range (``max_j |C_kj| w_j``, with
``w_j`` the width of the variable's box, or its scale when the box is open),
and ``s`` is the point in the scaled units, so ``|s| <= 1`` unless the point
lies far out along an open direction, where the check becomes relative as
float64 itself is. The distance returned is ``max_i |c_i + (G a)_i - y_i|`` at
that ``a``. So a returned distance belongs to a point of the set loosened in
every row by a billionth of the row's scale; an inaccurate solve can only
make it larger than the true distance, which is the direction "not a member".

Where the oracle is valid: wherever loosening each row by ``ROW_TOL`` of its
scale moves the set, in the output, by less than the caller's tolerance. That
fails only for a predicate polytope thinner than about ``ROW_TOL`` of its own
box in some direction, magnified past the tolerance by the generators; no set
the tests build is of that kind. Outside the regime the oracle raises rather
than answer: a solver point that breaks a row by more than ``ROW_TOL`` (or
whose rows cannot be evaluated to that precision, a predicate box far
narrower than its distance from zero) and a solve that stops without an
optimum raise ``RuntimeError``; a set or point holding NaN, or an infinite
centre, generator or row coefficient, raises ``ValueError``.

The only slack :func:`uncovered` adds is :data:`LP_TOL`, relative to the size
of the point.
"""

import math
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.sets.image_star import ImageStar

# The slack ``uncovered`` allows by default: a point is a member when its
# distance is at most LP_TOL * max(1, |y|). It was HiGHS's default primal
# feasibility tolerance; the distance itself now carries far less solver error
# (SOLVER_TOL in scaled units, then rechecked), so LP_TOL is the pad for the
# float64 rounding of reference outputs and of the set's own arrays.
LP_TOL = 1e-7

# HiGHS primal and dual feasibility tolerance, in the scaled units of the LP.
SOLVER_TOL = 1e-10

# The largest violation of a row, in units of the row's scale, that the
# predicate point behind an answer may have (module docstring).
ROW_TOL = 1e-9

_EPS = float(np.finfo(np.float64).eps)


def hwc(y, channels, height, width):
    """A (channels, height, width) tensor as the flat vector an image set
    stores (height, width, channels)."""
    y = np.asarray(y, dtype=np.float64).reshape(channels, height, width)
    return y.transpose(1, 2, 0).reshape(-1)


# --------------------------------------------------------------------------- #
# Reading a set                                                                 #
# --------------------------------------------------------------------------- #
def rows(s):
    """``(c, G, C, d, lb, ub)`` of a computed set, read from its stored
    arrays: ``x = c + G a``, ``C a <= d``, ``lb <= a <= ub`` (module
    docstring for each set type). Raises ``TypeError`` for any other object."""
    if isinstance(s, (Star, ImageStar)):
        return _star_rows(s)
    if isinstance(s, Zono):
        G = np.asarray(s.V, dtype=np.float64)
        G = G.reshape(G.shape[0], -1)
        n_var = G.shape[1]
        return (np.asarray(s.c, dtype=np.float64).reshape(-1), G,
                np.zeros((0, n_var)), np.zeros(0),
                -np.ones(n_var), np.ones(n_var))
    if isinstance(s, Box):
        lb = np.asarray(s.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(s.ub, dtype=np.float64).reshape(-1)
        n = lb.size
        return np.zeros(n), np.eye(n), np.zeros((0, n)), np.zeros(0), lb, ub
    if isinstance(s, Hexatope):
        c, G = _centre_and_generators(s, s.dcs.num_vars)
        p = G.shape[1]
        C = np.zeros((len(s.dcs.constraints), p))
        d = np.zeros(len(s.dcs.constraints))
        for k, dc in enumerate(s.dcs.constraints):
            i, j = _index(dc.i, p), _index(dc.j, p)
            C[k, i] += 1.0
            C[k, j] -= 1.0
            d[k] = float(dc.b)
        lb, ub = np.full(p, -np.inf), np.full(p, np.inf)
        if p:
            lb[0] = ub[0] = 0.0                       # the anchor stands for 0
        return c, G, C, d, lb, ub
    if isinstance(s, Octatope):
        c, G = _centre_and_generators(s, s.utvpi.num_vars)
        p = G.shape[1]
        C = np.zeros((len(s.utvpi.constraints), p))
        d = np.zeros(len(s.utvpi.constraints))
        for k, uc in enumerate(s.utvpi.constraints):
            i, j = _index(uc.i, p), _index(uc.j, p)
            for coefficient in (uc.ai, uc.aj):
                if coefficient not in (-1, 0, 1):
                    raise ValueError(
                        f"UTVPI coefficient {coefficient!r} is not -1, 0 or 1")
            C[k, i] += uc.ai
            C[k, j] += uc.aj
            d[k] = float(uc.b)
        return c, G, C, d, np.full(p, -np.inf), np.full(p, np.inf)
    raise TypeError(f"no row reader for {type(s).__name__}")


def _star_rows(s):
    n_var = int(s.nVar)
    V = np.asarray(s.V, dtype=np.float64).reshape(-1, n_var + 1)
    c, G = V[:, 0], V[:, 1:]
    stored = None if s.C is None else np.asarray(s.C, dtype=np.float64)
    n_rows = stored.shape[0] if stored is not None and stored.ndim == 2 else 0
    if n_rows == 0:
        C, d = np.zeros((0, n_var)), np.zeros(0)
    else:
        C = stored.reshape(n_rows, n_var)
        d = np.asarray(s.d, dtype=np.float64).reshape(-1)
        if d.size != n_rows:
            raise ValueError(f"{type(s).__name__} stores {n_rows} constraint "
                             f"rows and {d.size} right-hand sides")
    lb = (np.full(n_var, -np.inf) if s.predicate_lb is None
          else np.asarray(s.predicate_lb, dtype=np.float64).reshape(-1))
    ub = (np.full(n_var, np.inf) if s.predicate_ub is None
          else np.asarray(s.predicate_ub, dtype=np.float64).reshape(-1))
    return c, G, C, d, lb, ub


def _centre_and_generators(s, num_vars):
    c = np.asarray(s.center, dtype=np.float64).reshape(-1)
    G = np.asarray(s.generators, dtype=np.float64).reshape(c.size, -1)
    if G.shape[1] != num_vars:
        raise ValueError(f"{type(s).__name__} has {G.shape[1]} generators and "
                         f"a constraint system on {num_vars} variables")
    return c, G


def _index(i, p):
    i = int(i)
    if not 0 <= i < p:
        raise ValueError(f"constraint on variable {i} of {p}")
    return i


# --------------------------------------------------------------------------- #
# The distance                                                                  #
# --------------------------------------------------------------------------- #
def distance(s, y):
    """Infinity-norm distance from the point ``y`` to the set ``s``; ``inf``
    when the set is empty. The distance is attained by a predicate point the
    oracle checked itself (module docstring); a point with the wrong number of
    entries is an ``AssertionError``."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(y)):
        raise ValueError("the point holds NaN or inf")
    if isinstance(s, Box):
        lb = np.asarray(s.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(s.ub, dtype=np.float64).reshape(-1)
        _same_dim(lb.size, y.size, s)
        if np.isnan(lb).any() or np.isnan(ub).any():
            raise ValueError("the box holds NaN")
        return float(max(0.0, np.max(lb - y, initial=0.0),
                         np.max(y - ub, initial=0.0)))
    c, G, C, d, lb, ub = rows(s)
    _same_dim(c.size, y.size, s)
    return _distance(c, G, C, d, lb, ub, y)


def _pow2_scale(x):
    """A power of two ``p`` with ``p * x`` in ``[0.5, 1)``; 1 when ``x`` is 0.
    Multiplying by it is exact."""
    if not x > 0 or not math.isfinite(x):
        return 1.0
    return math.ldexp(1.0, -math.frexp(x)[1])


def _distance(c, G, C, d, lb, ub, y):
    for name, values in (("centre", c), ("generators", G), ("rows", C)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"the set's {name} hold NaN or inf")
    if np.isnan(d).any() or np.isnan(lb).any() or np.isnan(ub).any():
        raise ValueError("the set's right-hand sides or bounds hold NaN")
    if np.isposinf(lb).any() or np.isneginf(ub).any() or np.any(lb > ub):
        return float("inf")                           # crossed bounds: empty
    holds = ~np.isposinf(d)                           # a row ``<= inf`` holds
    C, d = C[holds], d[holds]
    if np.isneginf(d).any():
        return float("inf")

    # a = origin + scale * s, with s in [0, 1] for a variable boxed on both
    # sides; a pinned variable is its value and leaves the LP.
    n = G.shape[1]
    low, high = np.isfinite(lb), np.isfinite(ub)
    pinned = low & high & (lb == ub)
    boxed = low & high & ~pinned
    origin = np.where(low, lb, np.where(high, ub, 0.0))
    scale = np.ones(n)
    with np.errstate(over="ignore"):
        scale[boxed] = (ub - lb)[boxed]
    if not np.all(np.isfinite(scale)):
        raise ValueError("a predicate box too wide for float64")
    for j in np.flatnonzero(~boxed & ~pinned):
        scale[j] = _pow2_scale(max(np.max(np.abs(G[:, j]), initial=0.0),
                                   np.max(np.abs(C[:, j]), initial=0.0)))
    s_low = np.where(boxed | low, 0.0, -np.inf)
    s_high = np.where(boxed, 1.0, np.where(high & ~low, 0.0, np.inf))
    free = ~pinned
    Gs = G[:, free] * scale[free]
    cs = c + G @ origin
    Cs = C[:, free] * scale[free]
    ds = d - C @ origin

    # Rows left with no free variable are decided exactly at the origin.
    empty_rows = ~np.any(Cs != 0, axis=1)
    for k in np.flatnonzero(empty_rows):
        value = sum((Fraction(float(C[k, j])) * Fraction(float(origin[j]))
                     for j in range(n) if C[k, j] != 0), Fraction(0))
        if value > Fraction(float(d[k])):
            return float("inf")
    C, d, Cs, ds = C[~empty_rows], d[~empty_rows], Cs[~empty_rows], ds[~empty_rows]
    row_scale = np.max(np.abs(Cs), axis=1, initial=0.0)

    a = origin.copy()
    size = 1.0                       # max(1, |s|) of the scaled point
    m = int(free.sum())
    if m:
        s = _solve(Gs, cs, Cs, ds, row_scale, s_low[free], s_high[free], y)
        if s is None:
            return float("inf")
        s = np.clip(s, s_low[free], s_high[free])
        size = max(1.0, float(np.max(np.abs(s), initial=0.0)))
        a[free] = origin[free] + scale[free] * s
        a = np.clip(a, lb, ub)
    if C.shape[0]:
        excess = C @ a - d
        rounding = (n + 2) * _EPS * (np.abs(C) @ np.abs(a) + np.abs(d))
        share = (excess + rounding) / (row_scale * size)
        k = int(np.argmax(share))
        if share[k] > ROW_TOL:
            raise RuntimeError(
                f"the membership LP returned a predicate point that breaks "
                f"row {k} by {excess[k] / (row_scale[k] * size):.3g} of the "
                f"row's scale (float64 rounding of the row at that point "
                f"{rounding[k] / (row_scale[k] * size):.3g}; allowed "
                f"{ROW_TOL:g}): the set is outside the oracle's valid regime")
    return float(np.max(np.abs(c + G @ a - y), initial=0.0))


def _solve(Gs, cs, Cs, ds, row_scale, s_low, s_high, y):
    """The scaled LP; the predicate point ``s`` of its optimum, or None when
    it is infeasible."""
    m = Gs.shape[1]
    q = _pow2_scale(np.max(np.abs(Gs), initial=0.0))
    one = np.ones((Gs.shape[0], 1))
    A = [np.hstack([q * Gs, -one]), np.hstack([-q * Gs, -one])]
    b = [q * (y - cs), q * (cs - y)]
    if Cs.shape[0]:
        r = np.array([_pow2_scale(v) for v in row_scale])
        A.append(np.hstack([r[:, None] * Cs, np.zeros((Cs.shape[0], 1))]))
        b.append(r * ds)
    bounds = [(None if math.isinf(lo) else lo, None if math.isinf(hi) else hi)
              for lo, hi in zip(s_low, s_high)] + [(0, None)]
    cost = np.zeros(m + 1)
    cost[-1] = 1.0
    A, b = np.vstack(A), np.concatenate(b)
    res = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs",
                  options={"primal_feasibility_tolerance": SOLVER_TOL,
                           "dual_feasibility_tolerance": SOLVER_TOL})
    if res.status not in (0, 2):                     # retry at HiGHS defaults
        res = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if res.status == 2:                              # infeasible: empty set
        return None
    if res.status != 0:
        raise RuntimeError(f"membership LP did not solve: {res.message}")
    return np.asarray(res.x[:m], dtype=np.float64)


def _same_dim(set_dim, point_dim, s):
    if set_dim != point_dim:
        raise AssertionError(
            f"{type(s).__name__} has {set_dim} entries, the point has "
            f"{point_dim}")


def uncovered(sets, points, tol=LP_TOL):
    """The points that are members of none of ``sets``: a list of
    ``(index, distance to the nearest set)``. A point counts as a member when
    its distance is at most ``tol * max(1, |y|)``."""
    missed = []
    for i, y in enumerate(points):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        best = min((distance(s, y) for s in sets), default=float("inf"))
        if best > tol * max(1.0, float(np.max(np.abs(y), initial=0.0))):
            missed.append((i, best))
    return missed
