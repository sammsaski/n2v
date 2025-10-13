"""
LP solver interface using CVXPY.

Provides a unified interface for linear programming, replacing MATLAB's linprog.
Includes differentiable solver option based on Gumbel-Softmax for discrete optimization.
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple, Dict, Any, List
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def solve_lp(
    f: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    Aeq: Optional[np.ndarray] = None,
    beq: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    solver: str = 'default',
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
        solver: CVXPY solver name ('ECOS', 'SCS', 'OSQP', 'GLPK', etc.)
        minimize: If True, minimize; otherwise maximize
        **solver_kwargs: Additional keyword arguments for solver

    Returns:
        Tuple of (x, fval, status, info):
            x: Optimal solution vector (or None if infeasible)
            fval: Optimal objective value (or None)
            status: Solution status string
            info: Dictionary with solver information
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
        if solver == 'default' or solver is None:
            prob.solve(**solver_kwargs)
        else:
            prob.solve(solver=solver, **solver_kwargs)

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
            'solver': prob.solver_stats.solver_name if hasattr(prob, 'solver_stats') else solver,
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
    solver: str = 'default',
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
        f, A, b, Aeq, beq, lb, ub, solver=solver, minimize=True
    )

    return status in ['optimal', 'optimal_inaccurate']


# ======================== Differentiable Solver ========================

def solve_lp_differentiable(
    f: np.ndarray,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    Aeq: Optional[np.ndarray] = None,
    beq: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    minimize: bool = True,
    num_epochs: int = 100,
    batch_size: int = 32,
    init_temp: float = 10.0,
    final_temp: float = 0.1,
    learning_rate: float = 0.01,
    device: str = 'cpu',
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[np.ndarray], Optional[float], str, Dict[str, Any]]:
    """
    Solve LP problem using differentiable optimization with Gumbel-Softmax.

    This method is inspired by "Differentiable Combinatorial Scheduling at Scale" (ICML'24).
    It uses Gumbel-Softmax to make discrete optimization differentiable, enabling
    gradient-based optimization for problems that would otherwise require discrete solvers.

    The approach discretizes the continuous LP problem into a grid and uses
    Gumbel-Softmax sampling to select values differentiably.

    Args:
        f: Objective coefficient vector (n,) or (n, 1)
        A: Inequality constraint matrix (m, n)
        b: Inequality constraint vector (m,) or (m, 1)
        Aeq: Equality constraint matrix (p, n)
        beq: Equality constraint vector (p,) or (p, 1)
        lb: Lower bounds (n,) or (n, 1)
        ub: Upper bounds (n,) or (n, 1)
        minimize: If True, minimize; otherwise maximize
        num_epochs: Number of optimization epochs
        batch_size: Batch size for sampling
        init_temp: Initial Gumbel-Softmax temperature
        final_temp: Final Gumbel-Softmax temperature
        learning_rate: Learning rate for optimizer
        device: 'cpu' or 'cuda'
        verbose: If True, print progress
        **kwargs: Additional arguments (grid_size, constraint_penalty_weight, etc.)

    Returns:
        Tuple of (x, fval, status, info):
            x: Optimal solution vector (or None if infeasible)
            fval: Optimal objective value (or None)
            status: Solution status string
            info: Dictionary with solver information
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for differentiable solver. Install with: pip install torch")

    # Convert to numpy arrays
    f = np.asarray(f, dtype=np.float64).flatten()
    n = f.shape[0]

    # Determine bounds for each variable
    if lb is None:
        lb = np.full(n, -1e6)
    else:
        lb = np.asarray(lb, dtype=np.float64).flatten()

    if ub is None:
        ub = np.full(n, 1e6)
    else:
        ub = np.asarray(ub, dtype=np.float64).flatten()

    # Grid size for discretization
    grid_size = kwargs.get('grid_size', 50)
    constraint_penalty = kwargs.get('constraint_penalty_weight', 100.0)

    # Create the differentiable LP solver
    solver = _DifferentiableLPSolver(
        n_vars=n,
        objective_coef=f,
        A_ineq=A,
        b_ineq=b,
        A_eq=Aeq,
        b_eq=beq,
        lb=lb,
        ub=ub,
        minimize=minimize,
        grid_size=grid_size,
        constraint_penalty=constraint_penalty,
        device=device
    )

    # Optimize
    best_x, best_fval, info = solver.optimize(
        num_epochs=num_epochs,
        batch_size=batch_size,
        init_temp=init_temp,
        final_temp=final_temp,
        learning_rate=learning_rate,
        verbose=verbose
    )

    if best_x is not None:
        status = 'optimal'
    else:
        status = 'failed'

    return best_x, best_fval, status, info


class _DifferentiableLPSolver:
    """
    Internal class for differentiable LP solving using Gumbel-Softmax.

    Implements the core algorithm from "Differentiable Combinatorial Scheduling at Scale".
    """

    def __init__(
        self,
        n_vars: int,
        objective_coef: np.ndarray,
        A_ineq: Optional[np.ndarray],
        b_ineq: Optional[np.ndarray],
        A_eq: Optional[np.ndarray],
        b_eq: Optional[np.ndarray],
        lb: np.ndarray,
        ub: np.ndarray,
        minimize: bool,
        grid_size: int,
        constraint_penalty: float,
        device: str
    ):
        self.n_vars = n_vars
        self.grid_size = grid_size
        self.minimize = minimize
        self.constraint_penalty = constraint_penalty
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        # Convert to torch tensors
        self.objective = torch.tensor(objective_coef, dtype=torch.float32, device=self.device)

        # Store constraints
        self.A_ineq = torch.tensor(A_ineq, dtype=torch.float32, device=self.device) if A_ineq is not None else None
        self.b_ineq = torch.tensor(b_ineq.flatten(), dtype=torch.float32, device=self.device) if b_ineq is not None else None
        self.A_eq = torch.tensor(A_eq, dtype=torch.float32, device=self.device) if A_eq is not None else None
        self.b_eq = torch.tensor(b_eq.flatten(), dtype=torch.float32, device=self.device) if b_eq is not None else None

        # Create discretization grids for each variable
        self.grids = []
        for i in range(n_vars):
            grid = torch.linspace(lb[i], ub[i], grid_size, device=self.device)
            self.grids.append(grid)

        # Initialize learnable parameters (logits for Gumbel-Softmax)
        self.logits = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(grid_size, device=self.device))
            for _ in range(n_vars)
        ])

    def gumbel_softmax_sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        hard: bool = True
    ) -> torch.Tensor:
        """
        Sample from Gumbel-Softmax distribution.

        Args:
            logits: Logits tensor
            temperature: Temperature parameter
            hard: If True, return one-hot; otherwise return soft probabilities

        Returns:
            Sampled tensor
        """
        return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)

    def sample_solution(self, temperature: float, batch_size: int) -> torch.Tensor:
        """
        Sample a batch of solutions using Gumbel-Softmax.

        Args:
            temperature: Gumbel-Softmax temperature
            batch_size: Number of samples

        Returns:
            Tensor of shape (batch_size, n_vars) containing sampled solutions
        """
        solutions = []

        for var_idx in range(self.n_vars):
            # Get logits for this variable
            logits = self.logits[var_idx].unsqueeze(0).expand(batch_size, -1)

            # Sample from Gumbel-Softmax
            probs = self.gumbel_softmax_sample(logits, temperature, hard=False)

            # Convert probabilities to continuous values
            grid = self.grids[var_idx]
            value = torch.sum(probs * grid, dim=-1)

            solutions.append(value)

        return torch.stack(solutions, dim=1)

    def compute_objective(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute objective value for batch of solutions.

        Args:
            x: Solutions tensor of shape (batch_size, n_vars)

        Returns:
            Objective values of shape (batch_size,)
        """
        obj = torch.matmul(x, self.objective)
        if not self.minimize:
            obj = -obj
        return obj

    def compute_constraint_violation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute constraint violation penalty.

        Args:
            x: Solutions tensor of shape (batch_size, n_vars)

        Returns:
            Penalty values of shape (batch_size,)
        """
        penalty = torch.zeros(x.shape[0], device=self.device)

        # Inequality constraints: A * x <= b
        if self.A_ineq is not None:
            violations = torch.matmul(x, self.A_ineq.T) - self.b_ineq
            # Only penalize violations (positive values)
            penalty += torch.sum(F.relu(violations), dim=1)

        # Equality constraints: A * x = b
        if self.A_eq is not None:
            violations = torch.abs(torch.matmul(x, self.A_eq.T) - self.b_eq)
            penalty += torch.sum(violations, dim=1)

        return penalty * self.constraint_penalty

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss (objective + constraint violations).

        Args:
            x: Solutions tensor of shape (batch_size, n_vars)

        Returns:
            Loss values of shape (batch_size,)
        """
        obj = self.compute_objective(x)
        penalty = self.compute_constraint_violation(x)
        return obj + penalty

    def optimize(
        self,
        num_epochs: int,
        batch_size: int,
        init_temp: float,
        final_temp: float,
        learning_rate: float,
        verbose: bool
    ) -> Tuple[Optional[np.ndarray], Optional[float], Dict[str, Any]]:
        """
        Run the differentiable optimization.

        Returns:
            Tuple of (best_x, best_fval, info)
        """
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.logits, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-7
        )

        best_loss = float('inf')
        best_x = None
        best_fval = None

        # Temperature schedule
        temps = torch.linspace(init_temp, final_temp, num_epochs)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Sample solutions
            temperature = temps[epoch].item()
            x = self.sample_solution(temperature, batch_size)

            # Compute loss
            loss = self.compute_loss(x)
            loss_mean = loss.mean()

            # Backpropagation
            loss_mean.backward()
            optimizer.step()
            scheduler.step()

            # Track best solution
            min_loss, min_idx = loss.min(dim=0)
            if min_loss.item() < best_loss:
                best_loss = min_loss.item()
                best_x = x[min_idx].detach().cpu().numpy()
                best_fval = self.compute_objective(x[min_idx:min_idx+1]).item()
                if not self.minimize:
                    best_fval = -best_fval

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss_mean.item():.6f}, "
                      f"Best Loss = {best_loss:.6f}, Temp = {temperature:.3f}")

        info = {
            'solver': 'differentiable_gumbel',
            'num_epochs': num_epochs,
            'final_loss': best_loss,
            'device': str(self.device)
        }

        return best_x, best_fval, info
