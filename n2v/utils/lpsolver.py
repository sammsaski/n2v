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

from n2v.utils.lp_solver_enum import Backend, LPSolver, resolve

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
        is sufficient for bound computation in reachability analysis.
    """
    if not objectives:
        return []

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
        A_ub = np.asarray(A, dtype=np.float64) if A is not None else None
        b_ub = np.asarray(b, dtype=np.float64).flatten() if b is not None else None

        method = solver.scipy_method or 'highs'
        results = []
        for i, (f_obj, do_min) in enumerate(zip(objectives, minimize_flags)):
            f = f_obj if do_min else -f_obj

            try:
                res = scipy_linprog(c=f, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
                if res.success:
                    fval = -res.fun if not do_min else res.fun
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
    h = highspy.Highs()
    h.silent()

    # Use HiGHS infinity constant instead of hardcoded values
    inf = highspy.kHighsInf

    # Set column bounds
    col_lb = np.asarray(lb, dtype=np.float64).flatten() if lb is not None else np.full(n, -inf)
    col_ub = np.asarray(ub, dtype=np.float64).flatten() if ub is not None else np.full(n, inf)

    # Initial dummy cost (will be overwritten per solve)
    h.addVars(n, col_lb, col_ub)

    # Add inequality constraints: A @ x <= b
    if A is not None and b is not None:
        A_dense = np.asarray(A, dtype=np.float64)
        b_flat = np.asarray(b, dtype=np.float64).flatten()
        m = A_dense.shape[0]

        # Build sparse representation row by row
        for i in range(m):
            row = A_dense[i, :]
            nz_idx = np.nonzero(row)[0]
            if len(nz_idx) > 0:
                h.addRow(-inf, b_flat[i],
                         len(nz_idx),
                         nz_idx.astype(np.int32),
                         row[nz_idx])
            else:
                h.addRow(-inf, b_flat[i], 0, np.array([], dtype=np.int32), np.array([]))

    results = []
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
            results.append(
                h.getInfoValue("objective_function_value")[1]
            )
        else:
            results.append(None)

        # Clear solver state for next solve but keep model
        h.clearSolver()

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

    h = highspy.Highs()
    h.silent()
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

    # Add inequality constraints: A @ x <= b
    if A is not None and b is not None:
        A_dense = np.asarray(A, dtype=np.float64)
        b_flat = np.asarray(b, dtype=np.float64).flatten()
        m = A_dense.shape[0]

        for i in range(m):
            row = A_dense[i, :]
            nz_idx = np.nonzero(row)[0]
            if len(nz_idx) > 0:
                h.addRow(
                    -inf, b_flat[i],
                    len(nz_idx),
                    nz_idx.astype(np.int32),
                    row[nz_idx],
                )
            else:
                h.addRow(
                    -inf, b_flat[i], 0,
                    np.array([], dtype=np.int32),
                    np.array([]),
                )

    h.run()

    model_status = h.getModelStatus()
    if model_status == highspy.HighsModelStatus.kOptimal:
        fval = h.getInfoValue("objective_function_value")[1]
        x_opt = np.array(h.getSolution().col_value)
        return x_opt, fval, 'optimal', {'solver': 'highspy'}

    if model_status == highspy.HighsModelStatus.kUnbounded:
        return None, None, 'unbounded', {'solver': 'highspy'}

    return None, None, 'infeasible', {'solver': 'highspy'}


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
            fval: Optimal objective value (or None)
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
    n = f.shape[0]

    # Handle maximization by negating objective
    if not minimize:
        f = -f

    # Prepare inequality constraints
    A_ub = None
    b_ub = None
    if A is not None and b is not None:
        A_ub = np.asarray(A, dtype=np.float64) if not issparse(A) else A
        b_ub = np.asarray(b, dtype=np.float64).flatten()

    # Prepare equality constraints
    A_eq = None
    b_eq = None
    if Aeq is not None and beq is not None:
        A_eq = np.asarray(Aeq, dtype=np.float64) if not issparse(Aeq) else Aeq
        b_eq = np.asarray(beq, dtype=np.float64).flatten()

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
        result = scipy_linprog(
            c=f,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=method,
            **solver_kwargs
        )

        # Extract results
        if result.success:
            x_opt = result.x
            fval = result.fun
            if not minimize:
                fval = -fval  # Undo negation for maximization
            status = 'optimal'
        else:
            x_opt = None
            fval = None
            if 'infeasible' in result.message.lower():
                status = 'infeasible'
            elif 'unbounded' in result.message.lower():
                status = 'unbounded'
            else:
                status = f'failed: {result.message}'

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
        elif prob.status in ['infeasible', 'infeasible_inaccurate']:
            x_opt = None
            fval = None
            status = 'infeasible'
        elif prob.status in ['unbounded', 'unbounded_inaccurate']:
            x_opt = None
            fval = None
            status = 'unbounded'
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
    Check if a system of linear constraints is feasible.

    Args:
        Same as solve_lp (excluding f and minimize)

    Returns:
        True if feasible, False otherwise
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

    # Coerce once; solve_lp will also coerce defensively.
    solver = resolve(lp_solver)

    # Use zero objective (feasibility problem)
    f = np.zeros(n)

    x_opt, _, status, _ = solve_lp(
        f, A, b, Aeq, beq, lb, ub, lp_solver=solver, minimize=True
    )

    return status in ['optimal', 'optimal_inaccurate']
