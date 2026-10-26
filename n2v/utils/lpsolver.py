"""
LP solver interface with multiple backends.

Provides a unified interface for linear programming, replacing MATLAB's linprog.
Supports:
- highspy: Direct HiGHS C++ API (fastest, builds model once for batch solves)
- scipy.optimize.linprog with HiGHS (fast, recommended for Star set operations)
- CVXPY with various solvers (CLARABEL, ECOS, etc.)
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog as scipy_linprog
from scipy.sparse import issparse

# Profiler hook (no-op when profiling is disabled)
from n2v.profiling import count

from n2v.utils.lp_solver_enum import Backend, LPSolver, resolve


class LPFailure(RuntimeError):
    """The LP solver returned without deciding the problem.

    Raised where a *failure* status would otherwise be indistinguishable from
    a decided answer. The distinction is verdict-bearing: ``infeasible`` is a
    certificate of emptiness (and an empty set is vacuously safe), while
    ``unbounded`` / ``failed: *`` / ``error: *`` mean the solver gave up. A
    caller that reads the latter as emptiness turns a numerical failure into a
    false UNSAT, so those statuses raise and each caller decides explicitly --
    in the refine engine, by degrading that branch to UNKNOWN.
    """


# ============================================================================
# LP optima used as bounds
# ============================================================================

# SOUNDNESS: an LP optimum is exact only to the solver's tolerance (HiGHS
# reported a MAX up to 5e-10 below the true max on thin wedges). Wherever an
# optimum is used as a bound (a neuron's phase, the box of a relaxation, the
# range of a McCormick operand) it is padded outward by _LP_PAD * (1 + |v|).
# The pad is sized for HiGHS, so those LPs run on the solver _bound_lp_solver
# picks.
_LP_PAD = 1e-7


def _lp_pad(v):
    """Outward pad of an LP optimum ``v`` (scalar or array)."""
    return _LP_PAD * (1.0 + np.abs(v))


def _bound_lp_solver(lp_solver: Union[LPSolver, str, None]) -> LPSolver:
    """The solver for the LPs whose optimum is used as a bound.

    ``_LP_PAD`` is sized for HiGHS (scipy linprog or highspy). The CVXPY
    solvers stop at their own tolerances, which can be far larger: SCS
    reported maxima up to 3.6e-5 below the true max and OSQP up to 1.6e-5,
    both with status 'optimal', and 'optimal_inaccurate' is accepted as well.
    A bound taken from such a value can leave out reachable points (a false
    UNSAT), so these LPs are solved with HiGHS whatever solver the caller
    chose. A HiGHS variant the caller picked is kept.
    """
    solver = resolve(lp_solver, allow_sentinel=False)
    return solver if solver.is_scipy() else LPSolver.LINPROG


def _outward_lp_ranges(lo, hi, V) -> Tuple[np.ndarray, np.ndarray]:
    """Pad the LP ranges ``[lo, hi]`` of the dimensions of a set with basis
    ``V = [center | generators]`` outward by ``_lp_pad``.

    A dimension whose generator row is all zero is the constant ``V[i, 0]``;
    its range is set to that value exactly, without padding, so an op can
    still treat it as a constant.

    Args:
        lo, hi: LP lower/upper bounds, one per row of ``V``.
        V: the set's basis, shape ``(N, 1 + n)`` (an image basis
            ``(H, W, C, 1 + n)`` is read in its flat HWC row order).

    Returns:
        ``(lo, hi)`` as 1-D float64 arrays of length N.
    """
    V = np.asarray(V, dtype=np.float64)
    V = V.reshape(-1, V.shape[-1])
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    exact = ~np.any(V[:, 1:] != 0, axis=1)
    lo = np.where(exact, V[:, 0], lo - _lp_pad(lo))
    hi = np.where(exact, V[:, 0], hi + _lp_pad(hi))
    return lo, hi


# Try to import highspy for direct HiGHS API access
try:
    import highspy
    _HAS_HIGHSPY = True
except ImportError:
    _HAS_HIGHSPY = False

# Back-compat aliases: kept so any external importer still sees the sets.
# Derived from the enum; prefer LPSolver methods in new code.
SCIPY_SOLVERS = {m.value for m in LPSolver if m.is_scipy()}
_HIGHSPY_BATCH_SOLVERS = {m.value for m in LPSolver if m.is_highspy_batch_eligible()}


# HiGHS drops every constraint-matrix entry with |a| <= small_matrix_value
# (default 1e-9). Each highspy model built here lowers that to 1e-12, the
# smallest value HiGHS accepts; scipy is asked for the same, but older scipy
# versions ignore the option, so scipy rows are scaled against the default.
_HIGHS_SMALL_MATRIX_VALUE = 1e-12
_HIGHS_DEFAULT_SMALL_MATRIX_VALUE = 1e-9


def _new_highs(presolve: bool = True):
    """A silent ``highspy.Highs`` with the lowered small-matrix threshold,
    optionally with presolve off. Options must be set before rows are added:
    HiGHS drops small entries as the rows come in."""
    h = highspy.Highs()
    h.silent()
    h.setOptionValue("small_matrix_value", _HIGHS_SMALL_MATRIX_VALUE)
    if not presolve:
        h.setOptionValue("presolve", "off")
    return h


# Model statuses that decide an LP. With presolve on, the lowered threshold
# and row-scaled rows, HiGHS was measured to end in kUnknown on feasible,
# well-posed LPs (a 220-row star from ml4acopf) that a presolve-off solve
# decides at once. An LP left undecided by a presolve-on model is therefore
# re-solved once by solve_lp_no_presolve, as _scipy_linprog does for scipy.
_HIGHS_DECIDED = (
    frozenset({highspy.HighsModelStatus.kOptimal,
               highspy.HighsModelStatus.kInfeasible,
               highspy.HighsModelStatus.kUnbounded})
    if _HAS_HIGHSPY else frozenset())


def _add_le_rows(h, A, b):
    """Add the rows ``A x <= b`` (dense ``A``) to the HiGHS model ``h`` in
    one call. An all-zero row is kept, so ``0 <= b_i`` still constrains.

    HiGHS rejects the whole call when one entry exceeds its
    large_matrix_value (1e15); the rows are then added one by one, so only
    the offending rows are lost. A lost row relaxes the system: bounds only
    move outward and an 'infeasible' stays a proof, so this stays sound."""
    m = A.shape[0]
    if m == 0:
        return
    b = np.asarray(b, dtype=np.float64)
    r, c = np.nonzero(A)
    starts = np.searchsorted(r, np.arange(m)).astype(np.int32)
    inf = highspy.kHighsInf
    if h.addRows(m, np.full(m, -inf), b, c.size, starts, c.astype(np.int32),
                 A[r, c]) != highspy.HighsStatus.kError:
        return
    for i in range(m):
        nz = np.nonzero(A[i])[0]
        h.addRow(-inf, float(b[i]), len(nz), nz.astype(np.int32), A[i, nz])


def _scipy_linprog(c, options=None, **kwargs):
    """``scipy.optimize.linprog`` with the lowered small-matrix threshold
    requested for its HiGHS methods. scipy forwards options it does not know
    to HiGHS verbatim (older versions ignore them); either way it warns, and
    the warning is silenced here. A solve that ends in numerical trouble
    (scipy status 4) is retried once with presolve off, which decides most
    of them."""
    options = dict(options or {})
    highs = str(kwargs.get('method', 'highs')).startswith('highs')
    if highs:
        options.setdefault('small_matrix_value', _HIGHS_SMALL_MATRIX_VALUE)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options')
        res = scipy_linprog(c, options=options, **kwargs)
        if (res.status == 4 and highs
                and options.get('presolve', True) is not False):
            retry = scipy_linprog(c, options=dict(options, presolve=False),
                                  **kwargs)
            if retry.status != 4:
                res = retry
    return res


def _scipy_status(res) -> str:
    """``'optimal'`` / ``'infeasible'`` / ``'unbounded'`` / ``'failed: ...'``
    for a scipy ``linprog`` result. scipy also files solver failures under
    its infeasible code 2 (HiGHS kModelError, e.g. a matrix entry above
    1e15), and its failure messages name HiGHS's primal status ("model_status
    is Unknown; primal_status is Infeasible" under code 4), so a verdict is
    read only from code 2 / 3 together with the matching HiGHS text."""
    msg = str(getattr(res, 'message', '')).lower()
    if res.status == 0:
        return 'optimal'
    if res.status == 2 and 'problem is infeasible' in msg:
        return 'infeasible'
    if res.status == 3 and 'problem is unbounded' in msg:
        return 'unbounded'
    return f'failed: {res.message}'


def _row_scaled(A, b, keep_above: float = _HIGHS_SMALL_MATRIX_VALUE):
    """``(A / s, b / s, s)`` with ``s_i`` a power of two per row (1 for an
    all-zero or non-finite row).

    ``s_i`` brings the largest ``|A_ij|`` of row ``i`` into [0.5, 1), except
    that a row is never scaled DOWN so far that one of its nonzero entries
    falls to ``keep_above`` or below (pass the solver's small-matrix
    threshold): HiGHS would drop that entry and solve a different system.
    The one exception to the exception: a row is always scaled down until
    its largest entry is below 2^49 (HiGHS rejects the model above 1e15);
    its entries more than ~1e24 times smaller are then lost, as they would
    be to rounding anyway. Dividing by a power of two is exact in binary floating point (short of
    underflow), so the solutions are unchanged in exact arithmetic; what
    changes is the scale HiGHS applies its 1e-7 tolerances to. Measured on
    known-feasible polytopes: unscaled rows with ~1e-7..1e-9 coefficients
    were dropped or distorted (optima cutting off the known point, feasible
    systems reported infeasible); scaling every row down to [0.5, 1) pushed
    the small entries of rows with a wide coefficient range under the
    threshold instead (optima up to 0.5 inside the true range); with the
    guard and the lowered threshold, neither.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).flatten()
    absA = np.abs(A)
    # rowmax = m * 2^e with m in [0.5, 1); frexp gives e = 0 for an all-zero,
    # infinite or NaN row, which is left unscaled
    _, e = np.frexp(np.max(absA, axis=1, initial=0.0))
    down = e > 0
    if down.any():
        absA[absA == 0.0] = np.inf
        # rowmin >= 2^(e_min - 1) and keep_above < 2^e_keep, so dividing by
        # at most 2^(e_min - e_keep - 2) leaves rowmin >= 2 * keep_above
        _, e_min = np.frexp(np.min(absA, axis=1))
        e_keep = int(np.frexp(keep_above)[1])
        e_down = np.maximum(np.minimum(e, np.maximum(e_min - e_keep - 2, 0)),
                            e - 49)
        e = np.where(down, e_down, e)
    if not e.any():
        return A, b, np.ones(A.shape[0])
    scale = np.ldexp(1.0, e)
    return A / scale[:, None], b / scale, scale


def _finite_box(lb, ub, n):
    """``(lb, ub)`` as flat float arrays if both are given and finite, else
    ``None``."""
    if lb is None or ub is None:
        return None
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    if lb.size != n or ub.size != n:
        return None
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        return None
    return lb, ub


def _highs_row_multipliers(h, scale, minimize):
    """Row multipliers ``y >= 0`` (up to clamping) for the unscaled rows from
    a solved highspy model's row duals, or ``None`` if HiGHS has none. HiGHS
    reports ``d(objective)/d(rhs)``: nonpositive on a binding row of a
    minimization, nonnegative for a maximization. A row of the scaled system
    ``A_i / s_i`` with dual ``y_i`` is the unscaled row with ``y_i / s_i``."""
    sol = h.getSolution()
    if not getattr(sol, 'dual_valid', True):
        return None
    y = np.asarray(sol.row_dual, dtype=np.float64)
    if scale is not None and y.size != scale.size:
        return None      # HiGHS rejected rows (see _add_le_rows)
    if minimize:
        y = -y
    return y if scale is None else y / scale


def _scipy_row_multipliers(res, scale):
    """As ``_highs_row_multipliers`` for a scipy result (scipy minimizes, and
    reports the ``A_ub`` marginals, ``d(objective)/d(b_ub)``), or ``None``
    when the method gives no marginals."""
    marg = getattr(getattr(res, 'ineqlin', None), 'marginals', None)
    if marg is None:
        return None
    y = -np.asarray(marg, dtype=np.float64).flatten()
    return y if scale is None else y / scale


def _outward_optimum(fval, f, y, A, b, box, minimize):
    """The LP optimum ``fval`` of ``min``/``max f'x`` s.t. ``A x <= b`` over
    the finite ``box``, pushed outward to a rigorous bound where needed.

    For any multipliers ``y >= 0`` on the rows, the NS bound
    (``ns_certificate.ns_bound``) encloses the true optimum with float64
    rounding accounted for. Built from the solver's own row duals (``y`` for
    the rows as given, i.e. unscaled) it matches a well-conditioned optimum
    to ~1e-11 relative, while on ill-conditioned rows, where HiGHS's value
    can land inside the true range (measured ~1e-5 relative on column-scaled
    polytopes: a neuron's phase decided wrongly, a false UNSAT through exact
    reach), it stays on the safe side. The more outward of the two is
    returned, so the result is never tighter than the solver's value.
    """
    if box is None or fval is None:
        return fval
    from n2v.utils.ns_certificate import ns_bound
    if A is None or np.size(A) == 0:
        A, b, y = None, None, np.zeros(0)
    elif y is None:
        # no duals: y = 0 would give the bare box bound, valid but far too
        # loose to replace the optimum with
        return fval
    r = ns_bound(f, y, A=A, b=b, lb=box[0], ub=box[1], minimize=minimize)
    if not np.isfinite(r):
        return fval
    return min(fval, r) if minimize else max(fval, r)


def _outward_optima(results, certify, A, b, box, chunk: int = 256):
    """``_outward_optimum`` for several objectives over the same rows, in
    place on ``results``. ``certify`` lists ``(i, f, y, minimize)`` for each
    optimal ``results[i]`` whose row multipliers ``y`` are known; the NS
    bounds are computed in vectorized chunks (``ns_bounds_population``)."""
    if box is None or not certify:
        return
    from n2v.utils.ns_certificate import ns_bounds_population
    n = box[0].size
    if A is None or b is None or np.size(A) == 0:
        A, b = np.zeros((0, n)), np.zeros(0)
    m = A.shape[0]
    for s in range(0, len(certify), chunk):
        part = certify[s:s + chunk]
        F = np.stack([f for _, f, _, _ in part], axis=1)
        Y = (np.stack([y for _, _, y, _ in part], axis=1) if m
             else np.zeros((0, len(part))))
        flags = [mn for _, _, _, mn in part]
        r = ns_bounds_population(F[None], Y[None], A[None], b[None],
                                 box[0][None], box[1][None], flags)[0]
        for (i, _, _, mn), ri in zip(part, r):
            if np.isfinite(ri):
                results[i] = min(results[i], ri) if mn else max(results[i], ri)


def solve_lp_batch(
    objectives: List[np.ndarray],
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    minimize_flags: Optional[List[bool]] = None,
    lp_solver: Union[LPSolver, str, None] = LPSolver.DEFAULT,
) -> List[Optional[float]]:
    """
    Solve multiple LPs sharing the same constraints but different objectives.

    Builds the HiGHS model ONCE, then solves each objective by only changing
    the cost vector. Falls back to scipy.linprog if highspy is unavailable.

    Args:
        objectives: List of objective vectors, each shape (n,)
        A: Inequality constraint matrix (m, n), shared across all LPs
        b: Inequality constraint vector (m,), shared across all LPs
        lb: Lower bounds (n,), shared
        ub: Upper bounds (n,), shared
        minimize_flags: List of booleans (True=minimize, False=maximize).
                        If None, all minimize.
        lp_solver: Solver to use. 'default' resolves via global config.

    Returns:
        List of optimal objective values (None for infeasible/failed).
        Returns only optimal objective values (not solution vectors), which
        is sufficient for bound computation in reachability analysis. With
        a finite box the HiGHS paths return each value pushed outward to a
        rigorous bound, as ``solve_lp`` does.
    """
    if not objectives:
        return []

    count("n_lp_solves", len(objectives))

    # Resolve solver at the public boundary. ``allow_sentinel=False`` drops
    # ``LPSolver.DEFAULT`` down to whatever ``config.lp_solver`` currently is.
    solver = resolve(lp_solver, allow_sentinel=False)

    # Normalize and validate objectives
    objectives = [np.asarray(obj, dtype=np.float64).flatten() for obj in objectives]
    n = len(objectives[0])
    if not all(len(obj) == n for obj in objectives):
        raise ValueError("All objectives must have the same length")

    if minimize_flags is None:
        minimize_flags = [True] * len(objectives)
    elif len(minimize_flags) != len(objectives):
        raise ValueError(
            f"minimize_flags length ({len(minimize_flags)}) must match "
            f"objectives length ({len(objectives)})"
        )

    # Direct HiGHS path (fast)
    if _HAS_HIGHSPY and solver.is_highspy_batch_eligible():
        return _solve_batch_highspy(objectives, A, b, lb, ub, minimize_flags, n)

    # SciPy linprog fallback (for highs, highs-ds, highs-ipm, linprog)
    if solver.is_scipy():
        if lb is not None:
            lb_arr = np.asarray(lb, dtype=np.float64).flatten()
        else:
            lb_arr = np.full(n, -np.inf)
        if ub is not None:
            ub_arr = np.asarray(ub, dtype=np.float64).flatten()
        else:
            ub_arr = np.full(n, np.inf)
        bounds = list(zip(lb_arr, ub_arr))
        A_in = np.asarray(A, dtype=np.float64) if A is not None else None
        b_in = np.asarray(b, dtype=np.float64).flatten() if b is not None else None
        A_ub, b_ub, scale = A_in, b_in, None
        if A_ub is not None and b_ub is not None:
            A_ub, b_ub, scale = _row_scaled(    # see _row_scaled
                A_ub, b_ub, _HIGHS_DEFAULT_SMALL_MATRIX_VALUE)
        box = _finite_box(lb, ub, n)

        method = solver.scipy_method or 'highs'
        results = []
        certify = []    # optima to push outward (see _outward_optimum)
        for i, (f_obj, do_min) in enumerate(zip(objectives, minimize_flags)):
            f = f_obj if do_min else -f_obj

            try:
                res = _scipy_linprog(c=f, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
                if res.success:
                    fval = -res.fun if not do_min else res.fun
                    y = _scipy_row_multipliers(res, scale)
                    if box is not None and (y is not None or A_in is None):
                        certify.append((len(results), f_obj, y, do_min))
                    results.append(fval)
                else:
                    results.append(None)
            except Exception as e:
                warnings.warn(
                    f"LP solve failed for objective {i}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                results.append(None)

        _outward_optima(results, certify, A_in, b_in, box)
        return results

    # CVXPY / other solvers: sequential via solve_lp() which routes properly
    results = []
    for f_obj, do_min in zip(objectives, minimize_flags):
        f_col = f_obj.reshape(-1, 1)
        _, fval, status, _ = solve_lp(
            f=f_col, A=A, b=b, lb=lb, ub=ub,
            lp_solver=solver, minimize=do_min,
        )
        if status in ('optimal', 'optimal_inaccurate'):
            results.append(fval)
        else:
            results.append(None)

    return results


def _solve_batch_highspy(
    objectives: List[np.ndarray],
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
    minimize_flags: List[bool],
    n: int,
) -> List[Optional[float]]:
    """Solve batch LPs using direct highspy API. Builds model once."""
    h = _new_highs()

    # Use HiGHS infinity constant instead of hardcoded values
    inf = highspy.kHighsInf

    # Set column bounds
    col_lb = np.asarray(lb, dtype=np.float64).flatten() if lb is not None else np.full(n, -inf)
    col_ub = np.asarray(ub, dtype=np.float64).flatten() if ub is not None else np.full(n, inf)

    # Initial dummy cost (will be overwritten per solve)
    h.addVars(n, col_lb, col_ub)

    # Add inequality constraints: A @ x <= b (row-scaled, see _row_scaled)
    A_in = b_in = scale = None
    if A is not None and b is not None:
        A_in = np.asarray(A, dtype=np.float64)
        b_in = np.asarray(b, dtype=np.float64).flatten()
        A_dense, b_flat, scale = _row_scaled(A_in, b_in)
        _add_le_rows(h, A_dense, b_flat)
    box = _finite_box(lb, ub, n)

    results = []
    certify = []    # optima to push outward (see _outward_optimum)
    col_indices = np.arange(n, dtype=np.int32)

    for f_obj, do_min in zip(objectives, minimize_flags):
        f = np.asarray(f_obj, dtype=np.float64).flatten()

        # Set objective direction
        if do_min:
            h.changeObjectiveSense(highspy.ObjSense.kMinimize)
        else:
            h.changeObjectiveSense(highspy.ObjSense.kMaximize)

        # Update cost vector (the only thing that changes between solves)
        h.changeColsCost(n, col_indices, f)

        # Solve
        h.run()

        model_status = h.getModelStatus()
        if model_status == highspy.HighsModelStatus.kOptimal:
            if box is not None:
                y = _highs_row_multipliers(h, scale, do_min)
                if y is not None or A_in is None:
                    certify.append((len(results), f, y, do_min))
            results.append(h.getInfoValue("objective_function_value")[1])
        elif model_status in _HIGHS_DECIDED:
            results.append(None)
        else:
            # Not decided: retry with presolve off (see _HIGHS_DECIDED). Its
            # optimum is already pushed outward, so it is not certified again.
            _, fval, status, _ = solve_lp_no_presolve(
                f, A_in, b_in, lb, ub, minimize=do_min)
            results.append(fval if status == 'optimal' else None)

        # Clear solver state for next solve but keep model
        h.clearSolver()

    _outward_optima(results, certify, A_in, b_in, box)
    return results


def _solve_lp_highspy(
    f: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    minimize: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[float], str, Dict[str, Any]]:
    """Solve single LP using direct highspy API.

    Faster than scipy.linprog for repeated calls because it avoids
    the Python wrapper overhead. Does not support equality constraints;
    callers with Aeq/beq should use _solve_lp_scipy instead.
    """
    f = np.asarray(f, dtype=np.float64).flatten()
    n = f.shape[0]

    h = _new_highs()
    inf = highspy.kHighsInf

    # Column bounds
    col_lb = (
        np.asarray(lb, dtype=np.float64).flatten()
        if lb is not None
        else np.full(n, -inf)
    )
    col_ub = (
        np.asarray(ub, dtype=np.float64).flatten()
        if ub is not None
        else np.full(n, inf)
    )

    # Set objective direction
    if minimize:
        h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    else:
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)

    # Add variables with cost and bounds
    h.addVars(n, col_lb, col_ub)
    col_indices = np.arange(n, dtype=np.int32)
    h.changeColsCost(n, col_indices, f)

    # Add inequality constraints: A @ x <= b (row-scaled, see _row_scaled)
    A_in = b_in = scale = None
    if A is not None and b is not None:
        A_in = np.asarray(A, dtype=np.float64)
        b_in = np.asarray(b, dtype=np.float64).flatten()
        A_dense, b_flat, scale = _row_scaled(A_in, b_in)
        _add_le_rows(h, A_dense, b_flat)

    h.run()

    model_status = h.getModelStatus()
    if model_status == highspy.HighsModelStatus.kOptimal:
        fval = h.getInfoValue("objective_function_value")[1]
        x_opt = np.array(h.getSolution().col_value)
        box = _finite_box(lb, ub, n)
        if box is not None:
            fval = _outward_optimum(
                fval, f, _highs_row_multipliers(h, scale, minimize),
                A_in, b_in, box, minimize)
        return x_opt, fval, 'optimal', {'solver': 'highspy'}

    if model_status == highspy.HighsModelStatus.kUnbounded:
        return None, None, 'unbounded', {'solver': 'highspy'}

    if model_status == highspy.HighsModelStatus.kInfeasible:
        return None, None, 'infeasible', {'solver': 'highspy'}

    # Anything else (kUnknown, kModelEmpty from rejected inputs, limits, solve
    # errors, kUnboundedOrInfeasible, ...) is a FAILURE, not a verdict: callers
    # that treat 'infeasible' as an emptiness certificate (e.g. vacuous-UNSAT
    # paths) must never see solver failures collapsed into it. It is retried
    # once with presolve off (see _HIGHS_DECIDED); that solve's 'infeasible'
    # is no stronger than this one's would have been, and every verdict path
    # confirms an 'infeasible' before reading it as empty.
    retry = solve_lp_no_presolve(f, A, b, lb, ub, minimize=minimize)
    if not retry[2].startswith('failed'):
        return retry
    return None, None, f'failed: {model_status.name}', {'solver': 'highspy'}


def solve_lp(
    f: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    Aeq: Optional[np.ndarray] = None,
    beq: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    lp_solver: Union[LPSolver, str, None] = LPSolver.DEFAULT,
    minimize: bool = True,
    **solver_kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], str, Dict[str, Any]]:
    """
    Solve linear programming problem.

    Solves:
        min/max f^T * x
        subject to:
            A * x <= b       (inequality constraints)
            Aeq * x = beq    (equality constraints)
            lb <= x <= ub    (bounds)

    Args:
        f: Objective coefficient vector (n,) or (n, 1)
        A: Inequality constraint matrix (m, n)
        b: Inequality constraint vector (m,) or (m, 1)
        Aeq: Equality constraint matrix (p, n)
        beq: Equality constraint vector (p,) or (p, 1)
        lb: Lower bounds (n,) or (n, 1)
        ub: Upper bounds (n,) or (n, 1)
        lp_solver: Solver to use:
            - 'default': Use global config (n2v.set_lp_solver), falls back to CVXPY
            - 'linprog' or 'highs': scipy.optimize.linprog with HiGHS (fast)
            - 'highs-ds': HiGHS dual simplex
            - 'highs-ipm': HiGHS interior point method
            - Any CVXPY solver name ('ECOS', 'SCS', 'OSQP', 'GLPK', etc.)
        minimize: If True, minimize; otherwise maximize
        **solver_kwargs: Additional keyword arguments for solver

    Returns:
        Tuple of (x, fval, status, info):
            x: Optimal solution vector (or None if infeasible)
            fval: Optimal objective value (or None). With a finite box and
                no equality rows, the HiGHS paths (highspy and scipy) return
                the more outward of the solver's value and a rigorous bound
                built from the solver's duals (see ``_outward_optimum``), so
                it never lies inside the true optimum.
            status: Solution status string
            info: Dictionary with solver information
    """
    # Coerce once at the public boundary; sentinel is resolved against the
    # global config so the remainder of this function speaks enum only.
    solver = resolve(lp_solver, allow_sentinel=False)

    # Route to appropriate solver backend
    # Direct highspy path: fastest, no equality constraint support
    if (
        _HAS_HIGHSPY
        and solver.is_highspy_batch_eligible()
        and Aeq is None
        and beq is None
    ):
        return _solve_lp_highspy(
            f, A, b, lb, ub, minimize,
        )

    if solver.is_scipy():
        return _solve_lp_scipy(
            f, A, b, Aeq, beq, lb, ub,
            solver, minimize, **solver_kwargs,
        )

    return _solve_lp_cvxpy(
        f, A, b, Aeq, beq, lb, ub,
        solver, minimize, **solver_kwargs,
    )


def _solve_lp_scipy(
    f: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    Aeq: Optional[np.ndarray] = None,
    beq: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    lp_solver: Union[LPSolver, str] = LPSolver.HIGHS,
    minimize: bool = True,
    **solver_kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], str, Dict[str, Any]]:
    """
    Solve LP using scipy.optimize.linprog with HiGHS solver.

    HiGHS is a high-performance LP solver that handles sparse matrices efficiently.
    This is significantly faster than CVXPY for the sparse constraint structures
    typical in Star set reachability analysis.
    """
    # Convert to numpy arrays
    f = np.asarray(f, dtype=np.float64).flatten()
    f_in = f
    n = f.shape[0]

    # Handle maximization by negating objective
    if not minimize:
        f = -f

    # Prepare inequality constraints (dense rows row-scaled, see _row_scaled)
    A_ub = None
    b_ub = None
    A_in = b_in = scale = None
    if A is not None and b is not None:
        if issparse(A):
            A_ub, b_ub = A, np.asarray(b, dtype=np.float64).flatten()
        else:
            A_in = np.asarray(A, dtype=np.float64)
            b_in = np.asarray(b, dtype=np.float64).flatten()
            A_ub, b_ub, scale = _row_scaled(A_in, b_in,
                                            _HIGHS_DEFAULT_SMALL_MATRIX_VALUE)

    # Prepare equality constraints
    A_eq = None
    b_eq = None
    if Aeq is not None and beq is not None:
        if issparse(Aeq):
            A_eq, b_eq = Aeq, np.asarray(beq, dtype=np.float64).flatten()
        else:
            A_eq, b_eq, _ = _row_scaled(Aeq, beq,
                                        _HIGHS_DEFAULT_SMALL_MATRIX_VALUE)

    # Prepare bounds as list of (lb, ub) tuples
    if lb is not None:
        lb = np.asarray(lb, dtype=np.float64).flatten()
    else:
        lb = np.full(n, -np.inf)

    if ub is not None:
        ub = np.asarray(ub, dtype=np.float64).flatten()
    else:
        ub = np.full(n, np.inf)

    bounds = list(zip(lb, ub))

    # Map solver name to scipy method via enum metadata.
    solver = resolve(lp_solver) if not isinstance(lp_solver, LPSolver) else lp_solver
    method = solver.scipy_method or 'highs'

    try:
        result = _scipy_linprog(
            c=f,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=method,
            **solver_kwargs
        )

        # Extract results. SOUNDNESS: 'infeasible' is verdict-bearing, so it
        # is read from the status code and the HiGHS text together, never
        # from a failure message that merely mentions a primal status.
        status = _scipy_status(result)
        if status == 'optimal':
            x_opt = result.x
            fval = result.fun
            if not minimize:
                fval = -fval  # Undo negation for maximization
            # rigorous outward value (dense rows, no equalities; see
            # _outward_optimum)
            if A_eq is None and (A_ub is None or A_in is not None):
                fval = _outward_optimum(
                    fval, f_in, _scipy_row_multipliers(result, scale),
                    A_in, b_in, _finite_box(lb, ub, n), minimize)
        else:
            x_opt = None
            fval = None

        info = {
            'solver': f'scipy_{method}',
            'num_iters': result.nit if hasattr(result, 'nit') else None,
            'message': result.message,
        }

        return x_opt, fval, status, info

    except Exception as e:
        return None, None, f'error: {str(e)}', {}


def _solve_lp_cvxpy(
    f: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    Aeq: Optional[np.ndarray] = None,
    beq: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    lp_solver: Union[LPSolver, str] = LPSolver.DEFAULT,
    minimize: bool = True,
    **solver_kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], str, Dict[str, Any]]:
    """
    Solve LP using CVXPY with specified solver.
    """
    # Convert to numpy arrays
    f = np.asarray(f, dtype=np.float64).flatten()
    n = f.shape[0]

    # Define optimization variable
    x = cp.Variable(n)

    # Define objective
    if minimize:
        objective = cp.Minimize(f @ x)
    else:
        objective = cp.Maximize(f @ x)

    # Build constraints
    constraints = []

    # Inequality constraints A * x <= b
    if A is not None and b is not None:
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).flatten()
        constraints.append(A @ x <= b)

    # Equality constraints Aeq * x = beq
    if Aeq is not None and beq is not None:
        Aeq = np.asarray(Aeq, dtype=np.float64)
        beq = np.asarray(beq, dtype=np.float64).flatten()
        constraints.append(Aeq @ x == beq)

    # Lower bounds
    if lb is not None:
        lb = np.asarray(lb, dtype=np.float64).flatten()
        constraints.append(x >= lb)

    # Upper bounds
    if ub is not None:
        ub = np.asarray(ub, dtype=np.float64).flatten()
        constraints.append(x <= ub)

    # Create and solve problem
    prob = cp.Problem(objective, constraints)

    solver = (
        lp_solver if isinstance(lp_solver, LPSolver) else resolve(lp_solver)
    )
    cvxpy_name = solver.cvxpy_name  # None for SENTINEL/SCIPY-family

    try:
        if cvxpy_name is None:
            prob.solve(**solver_kwargs)
        else:
            prob.solve(solver=cvxpy_name, **solver_kwargs)

        # Extract results
        if prob.status in ['optimal', 'optimal_inaccurate']:
            x_opt = x.value
            fval = prob.value
            status = prob.status
        elif prob.status == 'infeasible':
            x_opt = None
            fval = None
            status = 'infeasible'
        elif prob.status == 'unbounded':
            x_opt = None
            fval = None
            status = 'unbounded'
        elif prob.status in ['infeasible_inaccurate', 'unbounded_inaccurate']:
            # First-order solvers' *_inaccurate statuses are explicitly
            # uncertified claims -- never surface them as verdicts.
            x_opt = None
            fval = None
            status = f'failed: {prob.status}'
        else:
            x_opt = None
            fval = None
            status = prob.status

        # Prepare info dictionary
        info = {
            'solver': prob.solver_stats.solver_name if hasattr(prob, 'solver_stats') else cvxpy_name,
            'num_iters': prob.solver_stats.num_iters if hasattr(prob, 'solver_stats') else None,
            'setup_time': prob.solver_stats.setup_time if hasattr(prob, 'solver_stats') else None,
            'solve_time': prob.solver_stats.solve_time if hasattr(prob, 'solver_stats') else None,
        }

        return x_opt, fval, status, info

    except Exception as e:
        # Solver failed
        return None, None, f'error: {str(e)}', {}


def _feasibility_robust(A, b, lb, ub):
    """Feasibility of ``{A x <= b, lb <= x <= ub}`` solved the careful way:
    HiGHS with presolve OFF on a row-scaled system (scipy with presolve off as
    the fallback). Returns ``(status, ray)`` with status ``'optimal'`` /
    ``'infeasible'`` / ``'failed: ...'`` and ``ray`` the solver's dual ray
    mapped back to the UNSCALED rows (``None`` when unavailable).

    Rationale: HiGHS's default presolve declares FEASIBLE systems infeasible
    when coefficient scales fall near its 1e-7 tolerances (measured 2026-09-16:
    false 'infeasible' on 5-75 of 300 known-feasible systems at coefficient
    scales 1e-7..1e-9, and on triangle-relaxed neurons with a variable range
    ~1e-8 -- 0 of 300 with presolve off). The plain status is therefore not a
    certificate of emptiness on its own.
    """
    n = None
    scale = None
    if A is not None:
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).flatten()
        n = A.shape[1]
        if A.size:
            A, b, scale = _row_scaled(
                A, b, _HIGHS_SMALL_MATRIX_VALUE if _HAS_HIGHSPY
                else _HIGHS_DEFAULT_SMALL_MATRIX_VALUE)
    if lb is not None:
        lb = np.asarray(lb, dtype=np.float64).flatten()
        n = lb.size if n is None else n
    if ub is not None:
        ub = np.asarray(ub, dtype=np.float64).flatten()
        n = ub.size if n is None else n
    if n is None:
        return 'failed: no dimension', None
    f = np.zeros(n)

    if _HAS_HIGHSPY:
        h = _new_highs(presolve=False)
        inf = highspy.kHighsInf
        col_lb = lb if lb is not None else np.full(n, -inf)
        col_ub = ub if ub is not None else np.full(n, inf)
        h.addVars(n, col_lb, col_ub)
        h.changeColsCost(n, np.arange(n, dtype=np.int32), f)
        if A is not None and A.size:
            _add_le_rows(h, A, b)
        h.run()
        st = h.getModelStatus()
        if st == highspy.HighsModelStatus.kOptimal:
            return 'optimal', None
        if st == highspy.HighsModelStatus.kInfeasible:
            ray = None
            try:
                _, has_ray, ray_s = h.getDualRay()
                # (a ray shorter than A means HiGHS rejected rows; unusable)
                if (has_ray and A is not None and A.size
                        and len(ray_s) == A.shape[0]):
                    ray = np.asarray(ray_s, dtype=np.float64).flatten()
                    # a ray of the scaled rows A_i/s_i is a ray y_i/s_i of
                    # the unscaled rows
                    ray = ray / scale
            except Exception:  # pragma: no cover - API drift
                ray = None
            return 'infeasible', ray
        return f'failed: {st.name}', None

    bounds = list(zip(lb if lb is not None else [None] * n,
                      ub if ub is not None else [None] * n))
    res = _scipy_linprog(f, A_ub=A if (A is not None and A.size) else None,
                         b_ub=b if (A is not None and A.size) else None,
                         bounds=bounds, method='highs',
                         options={'presolve': False})
    status = _scipy_status(res)
    return (status if status in ('optimal', 'infeasible')
            else f'failed: scipy status {res.status}'), None


def _farkas_certified_empty(A, b, lb, ub, ray) -> bool:
    """Rigorous emptiness from a dual ray, via the NS certificate.

    For any ``y >= 0`` and every feasible ``x``: ``0 = 0'x >= min_box (A'y)'x
    - y'b``. The NS bound evaluates the right-hand side with outward rounding,
    so a strictly positive value is a machine-checked proof that no feasible
    ``x`` exists -- exactly the certificate the batched contraction uses.
    Either sign of the ray is tried (solver conventions differ); a wrong sign
    only yields a vacuous bound, never a false certificate.
    """
    from n2v.utils.ns_certificate import ns_bound
    if A is None or not np.size(A) or ray is None:
        return False
    if lb is None or ub is None:
        return False
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        return False
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).flatten()
    n = A.shape[1]
    for sign in (-1.0, 1.0):
        y = np.nan_to_num(np.clip(sign * ray, 0.0, None), nan=0.0,
                          posinf=0.0, neginf=0.0)
        if not np.any(y):
            continue
        # certified upper bound on max 0'x over the system; < 0 => empty
        u0 = ns_bound(np.zeros(n), y, A=A, b=b, lb=lb, ub=ub)
        if np.isfinite(u0) and u0 < 0.0:
            return True
    return False


# Row slack used when no machine-checked certificate can be formed (see
# ``confirm_infeasible``): the size of HiGHS's own 1e-7 primal feasibility
# tolerance, so a presolve-off 'infeasible' is accepted only if the system
# stays infeasible after every row is loosened by about that much.
_UNCERTIFIED_ROW_SLACK = 1e-7


def _stacked_rows(A, b, Aeq=None, beq=None):
    """``A x <= b`` and ``Aeq x = beq`` as one inequality system, each
    equality written as the two rows ``+-Aeq x <= +-beq``. Returns
    ``(None, None)`` when there are no rows."""
    blocks, rhs = [], []
    if A is not None:
        blocks.append(np.asarray(A, dtype=np.float64))
        rhs.append(np.asarray(b, dtype=np.float64).flatten())
    if Aeq is not None:
        Aeq = np.asarray(Aeq, dtype=np.float64)
        beq = np.asarray(beq, dtype=np.float64).flatten()
        blocks += [Aeq, -Aeq]
        rhs += [beq, -beq]
    if not blocks:
        return None, None
    return np.vstack(blocks), np.concatenate(rhs)


def _can_certify(A, lb, ub) -> bool:
    """Whether a machine-checked emptiness certificate can be formed: it needs
    the solver's dual ray (highspy), at least one row, and a finite box (the
    NS bound multiplies each column's reduced cost by its box endpoint, which
    is infinite for an unbounded column)."""
    if not _HAS_HIGHSPY or A is None or np.size(A) == 0:
        return False
    if lb is None or ub is None:
        return False
    return bool(np.all(np.isfinite(np.asarray(lb, dtype=np.float64)))
                and np.all(np.isfinite(np.asarray(ub, dtype=np.float64))))


def _careful_emptiness(A, b, lb, ub, tol: float = _UNCERTIFIED_ROW_SLACK):
    """Shared core of ``confirm_infeasible`` and ``check_feasibility`` for
    ``{A x <= b, lb <= x <= ub}`` (equalities already stacked as rows).

    Returns ``(status, evidence)``: ``status`` is the deciding careful solve's
    status (``'optimal'`` / ``'infeasible'`` / ``'failed: ...'``);
    ``evidence`` is ``'certificate'`` when emptiness is machine-checked,
    ``'status'`` when only a presolve-off 'infeasible' supports it (no
    certificate could be formed), and ``None`` when emptiness is not shown.
    """
    if A is not None:
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).flatten()
    if _can_certify(A, lb, ub):
        from n2v.utils.ns_certificate import outward_interval_slack
        lb = np.asarray(lb, dtype=np.float64).flatten()
        ub = np.asarray(ub, dtype=np.float64).flatten()
        # rigorous bound on the round-to-nearest error of each row's residual
        # A_i x - b_i over the box (a (k_i + 1)-term sum, k_i the nonzeros)
        b_s = b + outward_interval_slack(np.abs(b), A, lb, ub)
        status, ray = _feasibility_robust(A, b_s, lb, ub)
        if status != 'infeasible':
            return status, None
        if ray is not None:
            certified = _farkas_certified_empty(A, b_s, lb, ub, ray)
            return status, ('certificate' if certified else None)
        # 'infeasible' without a dual ray: decide as without a certificate
    b_t = None if A is None else b + tol * (1.0 + np.abs(b))
    status, _ = _feasibility_robust(A, b_t, lb, ub)
    return status, ('status' if status == 'infeasible' else None)


def confirm_infeasible(A, b, lb, ub, Aeq=None, beq=None,
                       tol: float = _UNCERTIFIED_ROW_SLACK) -> bool:
    """Return True only if ``{A x <= b, Aeq x = beq, lb <= x <= ub}`` is
    infeasible ROBUSTLY. A plain 'infeasible' from the default solve is not
    evidence (presolve artefacts on tiny-scale systems, measured). Each
    equality is handled as a pair of opposite rows.

    With highspy and a finite box (the case on verdict paths: reach keeps
    finite predicate boxes) emptiness is machine-checked:

      1. every row is slackened by a rigorous bound on the round-to-nearest
         error of evaluating its residual over the box,
         ``SAFETY * gamma_{k_i+1} * (|A_i| max(|lb|, |ub|) + |b_i|)`` with
         ``k_i`` the row's nonzeros (``outward_interval_slack``). A box point
         whose rows pass a float64 check is therefore never certified away,
         while the slack stays a few ulps per term. (A flat
         ``1e-7 * (1 + |b_i|)`` on every star row was amplified by the output
         weights into ~1e-6 relative and turned certified-disjoint stars into
         UNKNOWN; rounding of the spec rows themselves is covered by
         ``outward_spec_rows``.)
      2. re-solve with presolve OFF on row-scaled data; feasible -> False;
      3. on 'infeasible' require the solver's dual ray to pass the
         outward-rounded Farkas/NS check (``_farkas_certified_empty``): a
         machine-checked proof that the slackened system, and hence the
         given one, is empty.

    The certificate decides emptiness in real arithmetic. A system whose only
    "feasible" points pass a float64 check through rounding alone (a spec
    threshold equal to a float64-evaluated output, say) can be certified
    empty; it is empty over the reals.

    Without a finite box (the NS bound is infinite on an unbounded column),
    without highspy (scipy returns no dual ray) or without a ray, no
    certificate can be formed. The presolve-off 'infeasible' status is then
    accepted on rows slackened by ``tol * (1 + |b_i|)``, a margin the size
    of the solver's own primal tolerance, so a system the solver can only
    just call infeasible is not declared empty. That is evidence, not a
    proof (it is the trust main placed in these statuses): on badly
    conditioned rows a feasible system can still come back empty. Every
    caller reaches this only after another solve already reported the
    system infeasible, so two differently configured solves must agree.

    This is the single emptiness certificate used on verdict paths; an
    unconfirmed 'infeasible' is treated as "not decided" (never as empty).
    """
    A, b = _stacked_rows(A, b, Aeq, beq)
    _, evidence = _careful_emptiness(A, b, lb, ub, tol)
    return evidence is not None


def solve_lp_no_presolve(f, A, b, lb, ub, minimize: bool = True):
    """Re-solve an LP the careful way (HiGHS, presolve off, row-scaled rows).
    Used when the default solve reported 'infeasible' but the robust check
    did not confirm it. Returns ``(x, fval, status, info)`` like ``solve_lp``.
    """
    if not _HAS_HIGHSPY:
        return None, None, 'failed: highspy unavailable', {}
    f = np.asarray(f, dtype=np.float64).flatten()
    n = f.size
    h = _new_highs(presolve=False)
    inf = highspy.kHighsInf
    col_lb = (np.asarray(lb, dtype=np.float64).flatten() if lb is not None
              else np.full(n, -inf))
    col_ub = (np.asarray(ub, dtype=np.float64).flatten() if ub is not None
              else np.full(n, inf))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize if minimize
                           else highspy.ObjSense.kMaximize)
    h.addVars(n, col_lb, col_ub)
    h.changeColsCost(n, np.arange(n, dtype=np.int32), f)
    A_in = b_in = scale = None
    if A is not None and b is not None and np.size(A):
        A_in = np.asarray(A, dtype=np.float64)
        b_in = np.asarray(b, dtype=np.float64).flatten()
        A_s, b_s, scale = _row_scaled(A_in, b_in)
        _add_le_rows(h, A_s, b_s)
    h.run()
    st = h.getModelStatus()
    if st == highspy.HighsModelStatus.kOptimal:
        fval = h.getInfoValue("objective_function_value")[1]
        box = _finite_box(lb, ub, n)
        if box is not None:
            fval = _outward_optimum(
                fval, f, _highs_row_multipliers(h, scale, minimize),
                A_in, b_in, box, minimize)
        return (np.array(h.getSolution().col_value), fval,
                'optimal', {'solver': 'highspy-nopresolve'})
    if st == highspy.HighsModelStatus.kInfeasible:
        return None, None, 'infeasible', {'solver': 'highspy-nopresolve'}
    if st == highspy.HighsModelStatus.kUnbounded:
        return None, None, 'unbounded', {'solver': 'highspy-nopresolve'}
    return None, None, f'failed: {st.name}', {'solver': 'highspy-nopresolve'}


def _solve_lp_careful(f, A, b, lb, ub, minimize: bool = True):
    """``solve_lp_no_presolve``, with scipy's HiGHS (presolve off, row-scaled
    rows) standing in when highspy is absent. For LP optima used as bounds
    without a certificate (``ensure_finite_predicate_box``): the default
    presolve was measured to move such optima far inside the true range on
    tiny-coefficient systems."""
    if _HAS_HIGHSPY:
        return solve_lp_no_presolve(f, A, b, lb, ub, minimize=minimize)
    # _solve_lp_scipy row-scales dense rows itself
    return _solve_lp_scipy(f, A, b, None, None, lb, ub, LPSolver.HIGHS,
                           minimize, options={'presolve': False})


def check_feasibility(
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    Aeq: Optional[np.ndarray] = None,
    beq: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    lp_solver: Union[LPSolver, str, None] = LPSolver.DEFAULT,
) -> bool:
    """
    Check whether ``{A x <= b, Aeq x = beq, lb <= x <= ub}`` may be feasible.

    The answer is one-sided: False only when ``confirm_infeasible``'s
    emptiness test accepts (a machine-checked certificate when highspy is
    available and the box is finite). True means "not shown empty": feasible,
    or infeasible by less than the test can resolve. That is the safe reading
    on verdict paths, where an empty set is vacuously safe.

    Args:
        Same as solve_lp (excluding f and minimize)

    Returns:
        False if the system is shown empty, True otherwise.

    Raises:
        LPFailure: no solve decided the question (a solver limit or numerical
            failure); that is not evidence of emptiness.
    """
    # Determine dimension from constraints
    if A is not None:
        n = A.shape[1]
    elif Aeq is not None:
        n = Aeq.shape[1]
    elif lb is not None:
        n = len(np.asarray(lb).flatten())
    elif ub is not None:
        n = len(np.asarray(ub).flatten())
    else:
        raise ValueError("Cannot determine problem dimension")

    # Coerce once, resolving the DEFAULT sentinel against the global config.
    solver = resolve(lp_solver, allow_sentinel=False)

    A_all, b_all = None, None
    if _HAS_HIGHSPY and solver.is_highspy_batch_eligible():
        A_all, b_all = _stacked_rows(A, b, Aeq, beq)
    if _can_certify(A_all, lb, ub):
        # One careful solve decides here. Running the default solve first
        # would only duplicate it: that solve's 'infeasible' is not a
        # certificate and has to be confirmed by this same careful solve.
        status, evidence = _careful_emptiness(A_all, b_all, lb, ub)
        if evidence == 'certificate':
            return False
        if evidence is None and status in ('optimal', 'infeasible'):
            return True
        # The careful solve failed, or reported 'infeasible' without a dual
        # ray: the default solve below must agree before anything is empty.

    # Use zero objective (feasibility problem)
    f = np.zeros(n)

    x_opt, _, status, _ = solve_lp(
        f, A, b, Aeq, beq, lb, ub, lp_solver=solver, minimize=True
    )

    # SOUNDNESS: only an explicit ``infeasible`` status is evidence of
    # emptiness. Every other non-optimal status ('unbounded', 'failed: ...',
    # 'error: ...') means the solver did not decide the question, and callers
    # that read a plain False as "provably empty" would turn a solver failure
    # into a false UNSAT (empty set => vacuously safe). Raise instead, so the
    # caller must decide explicitly (the refine engine degrades to UNKNOWN).
    if status in ('optimal', 'optimal_inaccurate'):
        return True
    if status == 'infeasible':
        # SOUNDNESS: the default solve's 'infeasible' is not a certificate on
        # tiny-scale or near-degenerate systems (presolve artefact, measured).
        # Confirm robustly; an unconfirmed infeasibility is reported as
        # FEASIBLE, i.e. "not provably empty", which is the safe direction.
        return not confirm_infeasible(A, b, lb, ub, Aeq=Aeq, beq=beq)
    raise LPFailure(
        f"feasibility LP did not decide: solver status {status!r}; "
        f"this is not evidence of emptiness"
    )
