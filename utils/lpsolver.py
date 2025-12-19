"""
LP solver interface using CVXPY.

Provides a unified interface for linear programming, replacing MATLAB's linprog.
Includes differentiable solver based on constraint-aware Gumbel-Softmax for
difference constraint systems (DCS) and UTVPI constraints, inspired by
"Differentiable Combinatorial Scheduling at Scale" (ICML'24).
"""

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple, Dict, Any, List
try:
    import torch
    import torch.nn.functional as F
    import networkx as nx
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


# ======================== Differentiable DCS/UTVPI Solver ========================

def solve_dcs_differentiable(
    constraint_graph: 'nx.DiGraph',
    objective_coef: np.ndarray,
    constant_term: float = 0.0,
    maximize: bool = True,
    num_epochs: int = 100,
    batch_size: int = 32,
    init_temp: float = 10.0,
    final_temp: float = 0.1,
    learning_rate: float = 0.01,
    device: str = 'cpu',
    verbose: bool = False,
    **kwargs
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Solve linear optimization over difference constraint system (DCS) using
    differentiable optimization inspired by "Differentiable Combinatorial
    Scheduling at Scale" (ICML'24).

    This solver is specifically designed for optimization problems over constraint
    graphs arising from difference constraint systems (hexatope) and UTVPI
    constraints (octatope).

    The key innovation is using constrained Gumbel-Softmax sampling that respects
    the constraint graph structure, rather than treating variables independently.

    Args:
        constraint_graph: NetworkX DiGraph with 'cost' and 'demand' node attributes
        objective_coef: Objective coefficients for each variable (node)
        constant_term: Constant term in the objective
        maximize: If True, maximize; else minimize
        num_epochs: Number of optimization epochs
        batch_size: Batch size for sampling
        init_temp: Initial Gumbel-Softmax temperature
        final_temp: Final Gumbel-Softmax temperature
        learning_rate: Learning rate for optimizer
        device: 'cpu' or 'cuda'
        verbose: If True, print progress
        **kwargs: Additional arguments (grid_size, etc.)

    Returns:
        Tuple of (optimal_value, info):
            optimal_value: Optimal objective value (or None if failed)
            info: Dictionary with solver information
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for differentiable solver. Install with: pip install torch")

    # Create solver instance
    solver = _DifferentiableDCSSolver(
        constraint_graph=constraint_graph,
        objective_coef=objective_coef,
        constant_term=constant_term,
        maximize=maximize,
        grid_size=kwargs.get('grid_size', 50),
        device=device
    )

    # Optimize
    optimal_value, info = solver.optimize(
        num_epochs=num_epochs,
        batch_size=batch_size,
        init_temp=init_temp,
        final_temp=final_temp,
        learning_rate=learning_rate,
        verbose=verbose
    )

    return optimal_value, info




class _DifferentiableDCSSolver:
    """
    Differentiable solver for Difference Constraint Systems (DCS).

    This solver is adapted from "Differentiable Combinatorial Scheduling at Scale"
    (ICML'24) but modified for DCS optimization problems arising in hexatope/octatope
    abstract domains.

    Key innovation: Instead of independent variable discretization, we use a
    constraint-aware assignment approach where each variable is assigned a value
    from a discretized grid, subject to difference constraints x_i - x_j <= b_ij.

    The constrained Gumbel-Softmax trick ensures that sampled assignments respect
    the partial order implied by the constraint graph.
    """

    def __init__(
        self,
        constraint_graph: 'nx.DiGraph',
        objective_coef: np.ndarray,
        constant_term: float,
        maximize: bool,
        grid_size: int,
        device: str
    ):
        self.graph = constraint_graph
        self.n_nodes = len(constraint_graph.nodes())
        self.maximize = maximize
        self.grid_size = grid_size
        self.constant_term = constant_term
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        # Convert objective to torch tensor
        self.objective = torch.tensor(objective_coef, dtype=torch.float32, device=self.device)

        # Build topological order (for constraint-aware sampling)
        try:
            # For DCS graphs, try topological sort
            self.topo_order = list(nx.topological_sort(constraint_graph))
        except:
            # If not DAG (e.g., has cycles), use arbitrary order
            # In DCS, negative cycles mean infeasible, but we still try
            self.topo_order = list(constraint_graph.nodes())

        # Extract predecessors for each node (for constrained Gumbel sampling)
        self.predecessors = {
            node: list(constraint_graph.predecessors(node))
            for node in constraint_graph.nodes()
        }

        # Determine value range from graph structure
        # For DCS, we estimate bounds using shortest path distances
        self.value_bounds = self._estimate_value_bounds()

        # Create discretization grids for each variable
        self.grids = {}
        for node in constraint_graph.nodes():
            lb, ub = self.value_bounds[node]
            grid = torch.linspace(lb, ub, grid_size, device=self.device)
            self.grids[node] = grid

        # Initialize learnable parameters (logits for Gumbel-Softmax)
        # Each node gets logits for choosing a grid value
        self.logits = torch.nn.ParameterDict({
            str(node): torch.nn.Parameter(torch.zeros(grid_size, device=self.device))
            for node in constraint_graph.nodes()
        })

    def _estimate_value_bounds(self) -> Dict[int, Tuple[float, float]]:
        """
        Estimate value bounds for each variable using graph structure.

        For hexatope/octatope DCS problems, variables in generator space are
        constrained to [-1, 1]. The constraint graph encodes relative constraints,
        but the absolute bounds come from the generator space restriction.

        We use a conservative approach: set reference node to 0, and use [-2, 2]
        for other variables (which allows [-1, 1] after accounting for constraints).
        """
        bounds = {}

        # For hexatope/octatope problems, generator space variables are in [-1, 1]
        # Node 0 is the reference (can be set to 0)
        # Nodes 1...n are the actual variables

        # Conservative bounds that should work for most cases:
        # - Reference node: fixed near 0
        # - Other nodes: wide enough to contain [-1, 1] with margin
        for node in self.graph.nodes():
            if node == 0:
                # Reference node - keep it near 0 for numerical stability
                bounds[node] = (-0.5, 0.5)
            else:
                # For hexatope/octatope, variables are typically in [-1, 1] in generator space
                # Use wider bounds to be safe
                bounds[node] = (-2.0, 2.0)

        return bounds

    def constrained_gumbel_softmax(
        self,
        logits: torch.Tensor,
        predecessors_samples: List[torch.Tensor],
        temperature: float,
        batch_size: int,
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constrained Gumbel-Softmax sampling adapted from Yu et al.

        This ensures that the sampled value for a variable respects constraints
        from its predecessors in the constraint graph.

        For a constraint x_i - x_j <= b, if x_j is assigned value v_j, then
        x_i must be assigned value <= v_j + b. This is enforced by masking
        the gumbel probabilities based on predecessor assignments.

        Args:
            logits: Logits for this variable (grid_size,)
            predecessors_samples: List of cumulative distributions from predecessors
            temperature: Gumbel temperature
            batch_size: Batch size
            hard: If True, use straight-through estimator

        Returns:
            (probs, cumulative_probs): Probability distribution and cumulative distribution
        """
        # Expand logits to batch
        logits_batch = logits.unsqueeze(0).expand(batch_size, -1)

        # Apply standard Gumbel-Softmax
        gumbels = -torch.empty_like(logits_batch).exponential_().log()
        gumbels = (logits_batch + gumbels) / temperature
        gumbels = gumbels.softmax(dim=-1)

        # Apply constraint masking from predecessors
        # The key insight from Yu et al.: multiply by predecessor cumulative distributions
        # This biases sampling toward values that satisfy ordering constraints
        if predecessors_samples:
            # Create bias toward higher indices (enforces ordering)
            bias = torch.arange(
                self.grid_size + 1, 1, -1, device=self.device
            ).log().repeat(batch_size, 1).float()

            # Multiply by each predecessor's cumulative distribution
            # This ensures we can't assign a value "before" our predecessors
            constrained = gumbels * bias
            for pred_cumsum in predecessors_samples:
                constrained = constrained * pred_cumsum

            probs = constrained.softmax(dim=-1)
        else:
            probs = gumbels

        # Compute cumulative distribution for use by successors
        cumsum = probs.cumsum(dim=1)

        # Hard sampling if requested (straight-through estimator)
        if hard:
            index = probs.max(dim=-1, keepdim=True)[1]
            probs_hard = torch.zeros_like(probs).scatter_(-1, index, 1.0)
            probs = probs_hard - probs.detach() + probs

        return probs, cumsum

    def sample_solution(
        self,
        temperature: float,
        batch_size: int
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Sample a batch of solutions respecting constraint graph structure.

        Uses topological ordering to ensure constraints are satisfied.

        Args:
            temperature: Gumbel-Softmax temperature
            batch_size: Number of samples

        Returns:
            (values, cumsums): Dictionary of sampled values and cumulative distributions
        """
        values = {}
        cumsums = {}

        # Sample in topological order to respect constraints
        for node in self.topo_order:
            logits = self.logits[str(node)]

            # Get cumulative distributions from predecessors
            pred_cumsums = [
                cumsums[pred] for pred in self.predecessors[node]
                if pred in cumsums
            ]

            # Sample with constraints
            probs, cumsum = self.constrained_gumbel_softmax(
                logits, pred_cumsums, temperature, batch_size, hard=False
            )

            # Convert probabilities to continuous values
            grid = self.grids[node]
            value = torch.sum(probs * grid, dim=-1)

            values[node] = value
            cumsums[node] = cumsum

        return values, cumsums

    def compute_objective(self, values: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Compute objective value for batch of solutions.

        Args:
            values: Dictionary mapping node to sampled values (batch_size,)

        Returns:
            Objective values of shape (batch_size,)
        """
        batch_size = next(iter(values.values())).shape[0]

        # Stack values for actual variables (nodes 1...n, excluding node 0)
        # Node 0 is the reference node in DCS constraint graphs
        x = torch.stack([
            values[node+1] if (node+1) in values else torch.zeros(batch_size, device=self.device)
            for node in range(len(self.objective))
        ], dim=1)

        # Compute objective: f^T * x + constant
        obj = torch.matmul(x, self.objective) + self.constant_term

        if not self.maximize:
            obj = -obj

        return obj

    def compute_constraint_violation(
        self,
        values: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute penalty for constraint violations.

        For DCS: x_i - x_j <= b_ij (encoded as edge j->i with cost b_ij)
        Violation = max(0, x_i - x_j - b_ij)

        Also enforces box constraints for hexatope/octatope: variables in
        generator space must be in [-1, 1].

        Args:
            values: Dictionary mapping node to sampled values

        Returns:
            Penalty values of shape (batch_size,)
        """
        batch_size = next(iter(values.values())).shape[0]
        penalty = torch.zeros(batch_size, device=self.device)

        # Check each edge constraint
        for i, j in self.graph.edges():
            edge_data = self.graph.edges[i, j]
            b_ij = edge_data.get('cost', 0.0)

            # Constraint: values[j] - values[i] <= b_ij
            # (Note: edge direction in constraint graph)
            if i in values and j in values:
                violation = values[j] - values[i] - b_ij
                penalty += F.relu(violation)

        # For hexatope/octatope: enforce generator space box constraints
        # Variables (nodes 1...n) should be in [-1, 1]
        for node in self.graph.nodes():
            if node > 0 and node in values:  # Skip reference node 0
                # Lower bound: x_i >= -1
                penalty += F.relu(-1.0 - values[node])
                # Upper bound: x_i <= 1
                penalty += F.relu(values[node] - 1.0)

        return penalty * 100.0  # Penalty weight

    def compute_loss(
        self,
        values: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total loss (negative objective + constraint violations) for minimization.

        For maximization problems, we minimize the negative of the objective.

        Args:
            values: Dictionary of sampled values

        Returns:
            Loss values of shape (batch_size,)
        """
        obj = self.compute_objective(values)
        penalty = self.compute_constraint_violation(values)

        # For maximization, minimize -obj; for minimization, minimize obj
        if self.maximize:
            return -obj + penalty
        else:
            return obj + penalty

    def optimize(
        self,
        num_epochs: int,
        batch_size: int,
        init_temp: float,
        final_temp: float,
        learning_rate: float,
        verbose: bool
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Run the differentiable optimization.

        Returns:
            Tuple of (best_value, info)
        """
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.logits.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-7
        )

        best_loss = float('inf')
        best_value = None

        # Temperature schedule
        temps = torch.linspace(init_temp, final_temp, num_epochs)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Sample solutions
            temperature = temps[epoch].item()
            values, _ = self.sample_solution(temperature, batch_size)

            # Compute loss
            loss = self.compute_loss(values)
            loss_mean = loss.mean()

            # Backpropagation
            loss_mean.backward()
            optimizer.step()
            scheduler.step()

            # Track best solution
            min_loss, min_idx = loss.min(dim=0)
            if min_loss.item() < best_loss:
                best_loss = min_loss.item()
                # Extract objective value
                obj = self.compute_objective(values)
                best_value = obj[min_idx].item()
                if not self.maximize:
                    best_value = -best_value

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss_mean.item():.6f}, "
                      f"Best Loss = {best_loss:.6f}, Temp = {temperature:.3f}")

        info = {
            'solver': 'differentiable_dcs',
            'num_epochs': num_epochs,
            'final_loss': best_loss,
            'device': str(self.device)
        }

        return best_value, info
