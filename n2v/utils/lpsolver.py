"""
LP solver interface with multiple backends.

Provides a unified interface for linear programming, replacing MATLAB's linprog.
Supports:
- scipy.optimize.linprog with HiGHS (fast, recommended for Star set operations)
- CVXPY with various solvers (CLARABEL, ECOS, etc.)
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog as scipy_linprog
from scipy.sparse import issparse, csr_matrix
from typing import Optional, Tuple, Dict, Any


# Solvers that use scipy linprog backend
SCIPY_SOLVERS = {'linprog', 'highs', 'highs-ds', 'highs-ipm'}


def solve_lp(
    f: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    Aeq: Optional[np.ndarray] = None,
    beq: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    lp_solver: str = 'default',
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
    # Check global config for 'default' solver
    if lp_solver == 'default':
        from n2v.config import config
        lp_solver = config.lp_solver

    # Route to appropriate solver backend
    if lp_solver in SCIPY_SOLVERS:
        return _solve_lp_scipy(f, A, b, Aeq, beq, lb, ub, lp_solver, minimize, **solver_kwargs)
    else:
        return _solve_lp_cvxpy(f, A, b, Aeq, beq, lb, ub, lp_solver, minimize, **solver_kwargs)


def _solve_lp_scipy(
    f: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    Aeq: Optional[np.ndarray] = None,
    beq: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    lp_solver: str = 'highs',
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

    # Map solver name to scipy method
    if lp_solver == 'linprog':
        method = 'highs'  # Default to HiGHS
    else:
        method = lp_solver

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
    lp_solver: str = 'default',
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

    try:
        if lp_solver == 'default' or lp_solver is None:
            prob.solve(**solver_kwargs)
        else:
            prob.solve(solver=lp_solver, **solver_kwargs)

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
            'solver': prob.solver_stats.solver_name if hasattr(prob, 'solver_stats') else lp_solver,
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
    lp_solver: str = 'default',
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

    # Use zero objective (feasibility problem)
    f = np.zeros(n)

    x_opt, _, status, _ = solve_lp(
        f, A, b, Aeq, beq, lb, ub, lp_solver=lp_solver, minimize=True
    )

    return status in ['optimal', 'optimal_inaccurate']
