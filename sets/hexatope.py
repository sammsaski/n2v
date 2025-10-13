"""
Hexatope Abstract Domain for Neural Network Verification

A hexatope is an affine transformation of a higher-dimensional hexagon,
defined by difference constraint systems (DCS).

Definition: H = <c, G, A, b> where:
- c ∈ ℝⁿ is the center
- G ∈ ℝⁿˣᵖ is the generator matrix
- Ax ≤ b defines a difference constraint system (DCS)

Semantics: [[H]] = {Gx + c : Ax ≤ b}

Reference: Bak et al., "The hexatope and octatope abstract domains for neural
network verification", Formal Methods in System Design (2024) 64:178–199
"""

import numpy as np
import networkx as nx
from typing import Tuple, Optional, List
from dataclasses import dataclass
import cvxpy as cp


@dataclass
class DifferenceConstraint:
    """Represents a difference constraint: x_i - x_j ≤ b"""
    i: int  # Index of variable x_i
    j: int  # Index of variable x_j
    b: float  # Bound


class DifferenceConstraintSystem:
    """
    Difference Constraint System (DCS)

    A conjunction of constraints of the form x_i - x_j ≤ b
    """

    def __init__(self, num_vars: int):
        self.num_vars = num_vars
        self.constraints: List[DifferenceConstraint] = []

    def add_constraint(self, i: int, j: int, b: float):
        """Add constraint x_i - x_j ≤ b"""
        if i < 0 or i >= self.num_vars or j < 0 or j >= self.num_vars:
            raise ValueError(f"Invalid variable indices: i={i}, j={j}")
        self.constraints.append(DifferenceConstraint(i, j, b))

    def to_constraint_graph(self) -> nx.DiGraph:
        """
        Convert DCS to constraint graph for minimum cost flow

        For each constraint x_i - x_j ≤ b, add edge (v_j, v_i) with cost b.
        Add extra vertex v_0 with edges (v_0, v_i) with cost 0 for all i > 0.
        """
        G = nx.DiGraph()

        # Add vertices
        G.add_node(0)  # Extra vertex v_0
        for i in range(1, self.num_vars + 1):
            G.add_node(i)

        # Add edges for each difference constraint
        for dc in self.constraints:
            # Constraint x_i - x_j ≤ b becomes edge (v_j, v_i) with cost b
            # Note: using 1-indexed for non-zero vertices
            G.add_edge(dc.j + 1, dc.i + 1, cost=dc.b, capacity=float('inf'))

        # Add edges from v_0 to all other vertices
        for i in range(1, self.num_vars + 1):
            G.add_edge(0, i, cost=0, capacity=float('inf'))

        return G

    def is_feasible(self) -> bool:
        """Check if the DCS is feasible (no negative cycles)"""
        G = self.to_constraint_graph()

        # Check for negative cycles using Bellman-Ford
        try:
            # Try to find shortest paths from v_0
            nx.single_source_bellman_ford_path_length(G, 0, weight='cost')
            return True
        except nx.NetworkXUnbounded:
            # Negative cycle detected
            return False

    def to_matrix_form(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DCS to matrix form Ax ≤ b

        Each constraint x_i - x_j ≤ b_k becomes row: [... 1 ... -1 ...] ≤ b_k
        """
        m = len(self.constraints)
        A = np.zeros((m, self.num_vars))
        b = np.zeros(m)

        for k, dc in enumerate(self.constraints):
            A[k, dc.i] = 1
            A[k, dc.j] = -1
            b[k] = dc.b

        return A, b

    def copy(self) -> 'DifferenceConstraintSystem':
        """Create a deep copy of the DCS"""
        new_dcs = DifferenceConstraintSystem(self.num_vars)
        new_dcs.constraints = [
            DifferenceConstraint(dc.i, dc.j, dc.b)
            for dc in self.constraints
        ]
        return new_dcs


class Hexatope:
    """
    Hexatope Abstract Domain

    A hexatope H = <c, G, A, b> is a special type of linear star set
    where the kernel Ax ≤ b is defined by a difference constraint system.

    Semantics: [[H]] = {Gx + c : Ax ≤ b where Ax ≤ b is a DCS}
    """

    def __init__(self, center: np.ndarray, generators: np.ndarray,
                 dcs: DifferenceConstraintSystem,
                 state_lb: Optional[np.ndarray] = None,
                 state_ub: Optional[np.ndarray] = None,
                 extra_A: Optional[np.ndarray] = None,
                 extra_b: Optional[np.ndarray] = None):
        """
        Initialize a hexatope

        Args:
            center: Center vector c ∈ ℝⁿ
            generators: Generator matrix G ∈ ℝⁿˣᵖ
            dcs: Difference constraint system defining the kernel
            state_lb: Lower bounds for state variables (optional)
            state_ub: Upper bounds for state variables (optional)
            extra_A: Additional linear constraint matrix (optional, for constraints beyond DCS)
            extra_b: Additional linear constraint bounds (optional)
        """
        self.center = np.asarray(center, dtype=np.float64).reshape(-1)
        self.generators = np.asarray(generators, dtype=np.float64)
        self.dcs = dcs

        # Validate dimensions
        if len(self.generators.shape) == 1:
            self.generators = self.generators.reshape(-1, 1)

        n, p = self.generators.shape
        if self.center.shape[0] != n:
            raise ValueError(f"Center dimension {self.center.shape[0]} != {n}")
        if self.dcs.num_vars != p:
            raise ValueError(f"DCS variables {self.dcs.num_vars} != generators {p}")

        # Store state bounds
        if state_lb is not None:
            state_lb = np.asarray(state_lb, dtype=np.float64).reshape(-1, 1)
            if state_lb.shape[0] != n:
                raise ValueError(f"State lb size doesn't match dimension {n}")
        if state_ub is not None:
            state_ub = np.asarray(state_ub, dtype=np.float64).reshape(-1, 1)
            if state_ub.shape[0] != n:
                raise ValueError(f"State ub size doesn't match dimension {n}")

        self.state_lb = state_lb
        self.state_ub = state_ub

        # Store extra linear constraints (for constraints that can't be expressed in DCS)
        if extra_A is not None:
            extra_A = np.asarray(extra_A, dtype=np.float64)
            if extra_A.ndim == 1:
                extra_A = extra_A.reshape(1, -1)
            if extra_A.shape[1] != p:
                raise ValueError(f"Extra constraint matrix has {extra_A.shape[1]} columns, expected {p}")
        if extra_b is not None:
            extra_b = np.asarray(extra_b, dtype=np.float64).reshape(-1, 1)
            if extra_A is not None and extra_A.shape[0] != extra_b.shape[0]:
                raise ValueError(f"Extra constraint matrix and vector size mismatch")

        self.extra_A = extra_A
        self.extra_b = extra_b

    @property
    def dim(self) -> int:
        """Dimension of the hexatope (output dimension)"""
        return self.center.shape[0]

    @property
    def nVar(self) -> int:
        """Number of generator vectors (kernel dimension)"""
        return self.generators.shape[1]

    def __repr__(self) -> str:
        return (f"Hexatope(dim={self.dim}, nVar={self.nVar}, "
                f"nConstraints={len(self.dcs.constraints)})")

    @classmethod
    def from_bounds(cls, lb: np.ndarray, ub: np.ndarray) -> 'Hexatope':
        """
        Create a hexatope representing a hyperrectangle [lower, upper]

        Args:
            lb: Lower bounds
            ub: Upper bounds

        Returns:
            Hexatope representing the box
        """
        lb = np.asarray(lb, dtype=np.float64).flatten()
        ub = np.asarray(ub, dtype=np.float64).flatten()
        n = lb.shape[0]

        # Center: midpoint of box
        center = (lb + ub) / 2

        # Generators: diagonal matrix with half-widths
        half_widths = (ub - lb) / 2
        generators = np.diag(half_widths)

        # DCS: -1 ≤ x_i ≤ 1 for each i in generator space
        #
        # Standard DCS representation for box [-1, 1]^n uses implicit reference x_0 = 0:
        # For the constraint graph, vertex v_0 represents x_0 = 0 (implicit)
        # Vertices v_1, ..., v_n represent x_1, ..., x_n
        #
        # However, in the DCS class, we only track n variables (x_1, ..., x_n).
        # The reference x_0 = 0 is handled implicitly in the constraint graph.
        #
        # To represent x_i ≤ 1, we need to anchor to a known value.
        # Since all vars are in [-1, 1], we can use relative constraints.
        # But DCS alone without reference point cannot bound absolute values.
        #
        # Solution: Use both upper and lower bound constraints relative to other variables
        # For box [-1, 1]^n, add constraints to bound each variable:
        dcs = DifferenceConstraintSystem(n)

        if n == 1:
            # For 1D, we can't use difference constraints alone
            # The constraint graph will handle this with v_0 reference
            # No explicit DCS constraints needed - bounds come from optimization
            pass
        else:
            # For multi-dimensional boxes, add constraints to maintain box structure
            # x_i - x_j ≤ 2 for all i,j ensures max spread of 2
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # x_i - x_j ≤ 2 (when x_i=1, x_j=-1: 1-(-1)=2)
                        dcs.add_constraint(i, j, 2.0)

        return cls(center, generators, dcs, state_lb=lb.reshape(-1, 1),
                   state_ub=ub.reshape(-1, 1))

    # ======================== Affine Transformations ========================

    def affine_map(self, W: np.ndarray, b: Optional[np.ndarray] = None) -> 'Hexatope':
        """
        Apply affine transformation: W*x + b

        Theorem 4: Hexatopes are closed under affine transformation.

        For hexatope H = <c, G, A, b> and affine map f(x) = Wx + d,
        the result is H' = <c', G', A, b> where:
        - c' = Wc + d
        - G' = WG
        - A, b remain unchanged (kernel unchanged)

        Args:
            W: Mapping matrix (m, n)
            b: Mapping vector (m,) or (m, 1), optional

        Returns:
            New Hexatope object
        """
        W = np.asarray(W, dtype=np.float64)

        if W.shape[1] != self.dim:
            raise ValueError(f"Matrix W has {W.shape[1]} columns, expected {self.dim}")

        # New center: c' = Wc + d
        new_center = W @ self.center
        if b is not None:
            b_arr = np.asarray(b, dtype=np.float64).flatten()
            new_center = new_center + b_arr

        # New generators: G' = WG
        new_generators = W @ self.generators

        # Kernel constraints remain unchanged (both DCS and extra constraints)
        return Hexatope(new_center, new_generators, self.dcs,
                       extra_A=self.extra_A, extra_b=self.extra_b)

    # ======================== Bounds Computation ========================

    def get_range(self, index: int, use_mcf: bool = True) -> Tuple[float, float]:
        """
        Compute exact range at specific dimension

        Theorem 5: Linear optimization over hexatopes can be solved in
        strongly polynomial time via reduction to minimum cost flow.

        Args:
            index: Dimension index (0-based)
            use_mcf: If True, use min-cost flow; else use LP

        Returns:
            Tuple of (min, max) values
        """
        if index < 0 or index >= self.dim:
            raise ValueError(f"Invalid index {index}, dimension is {self.dim}")

        # Create objective vector with 1 at position index
        objective = np.zeros(self.dim)
        objective[index] = 1.0

        # Minimize and maximize
        xmin = self.optimize_linear(objective, maximize=False, use_mcf=use_mcf)
        xmax = self.optimize_linear(objective, maximize=True, use_mcf=use_mcf)

        if xmin is None or xmax is None:
            return None, None

        return xmin, xmax

    def get_ranges(self, use_mcf: bool = True, parallel: bool = False,
                   n_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute exact ranges for all dimensions

        Args:
            use_mcf: If True, use min-cost flow; else use LP (LP is more reliable)
            parallel: If True, use parallel computation
            n_workers: Number of parallel workers

        Returns:
            Tuple of (lb, ub) arrays
        """
        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        if parallel and self.dim > 1:
            return self._get_ranges_parallel(use_mcf, n_workers)

        # Sequential version - use LP for reliability
        for i in range(self.dim):
            lb_i, ub_i = self.get_range(i, use_mcf=False)  # Force LP for now
            if lb_i is None or ub_i is None:
                # Fallback to estimation if exact fails
                lb_i, ub_i = self.estimate_range(i)
            lb[i] = lb_i
            ub[i] = ub_i

        return lb, ub

    def get_bounds(self, use_mcf: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alias for get_ranges() for API consistency with other set types.

        Args:
            use_mcf: If True, use min-cost flow; else use LP

        Returns:
            Tuple of (lb, ub) arrays
        """
        return self.get_ranges(use_mcf=use_mcf)

    def _get_ranges_parallel(self, use_mcf: bool = True,
                            n_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ranges in parallel"""
        from concurrent.futures import ThreadPoolExecutor

        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        def compute_range(i):
            try:
                return i, self.get_range(i, use_mcf)
            except Exception:
                return i, (None, None)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(compute_range, i) for i in range(self.dim)]

            for future in futures:
                i, (lb_i, ub_i) = future.result()
                if lb_i is not None and ub_i is not None:
                    lb[i] = lb_i
                    ub[i] = ub_i
                else:
                    # Fall back to estimation
                    lb[i], ub[i] = self.estimate_range(i)

        return lb, ub

    def estimate_range(self, index: int) -> Tuple[float, float]:
        """
        Fast over-approximate range estimation

        Args:
            index: Dimension index

        Returns:
            Tuple of (min_estimate, max_estimate)
        """
        # Use interval arithmetic on generators
        # Assume generators are bounded by [-1, 1] (standard for DCS)
        c = self.center[index]
        generators = self.generators[index, :]

        lb_contrib = -np.sum(np.abs(generators))
        ub_contrib = np.sum(np.abs(generators))

        return c + lb_contrib, c + ub_contrib

    def estimate_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fast over-approximate ranges for all dimensions"""
        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        for i in range(self.dim):
            lb[i], ub[i] = self.estimate_range(i)

        self.state_lb = lb
        self.state_ub = ub

        return lb, ub

    def get_box(self, use_mcf: bool = True) -> 'Box':
        """
        Compute exact bounding box

        Args:
            use_mcf: If True, use min-cost flow; else use LP

        Returns:
            Box object
        """
        from n2v.sets.box import Box

        lb, ub = self.get_ranges(use_mcf=use_mcf)
        return Box(lb, ub)

    def optimize_linear(self, objective: np.ndarray, maximize: bool = True,
                       use_mcf: bool = True) -> Optional[float]:
        """
        Optimize linear objective over hexatope

        Theorem 5: Linear optimization over hexatopes can be solved in
        strongly polynomial time via reduction to minimum cost flow.

        To optimize f(y) = f^T y over [[H]], we optimize f^T(Gx + c) over Ax ≤ b.
        This reduces to optimizing (f^T G)x + f^T c over the DCS.

        Args:
            objective: Objective vector f ∈ ℝⁿ
            maximize: If True, maximize; else minimize
            use_mcf: If True, use min-cost flow; else use LP

        Returns:
            Optimal value, or None if infeasible
        """
        objective = np.asarray(objective, dtype=np.float64).flatten()

        if objective.shape[0] != self.dim:
            raise ValueError(f"Objective size {objective.shape[0]} != dim {self.dim}")

        # Compose objective with affine mapping
        # f^T(Gx + c) = (f^T G)x + f^T c
        composed_obj = objective @ self.generators  # w = f^T G
        constant_term = objective @ self.center  # f^T c

        # Now optimize w^T x over DCS
        if use_mcf:
            result = self._optimize_dcs_mcf(composed_obj, constant_term, maximize)
        else:
            result = self._optimize_dcs_lp(composed_obj, constant_term, maximize)

        return result

    def _optimize_dcs_mcf(self, w: np.ndarray, constant: float,
                          maximize: bool) -> Optional[float]:
        """
        Optimize linear objective w^T x + constant over DCS using min-cost flow

        Reduces to minimum cost flow problem on constraint graph.
        """
        if maximize:
            w = -w
            constant = -constant

        # Build constraint graph
        G = self.dcs.to_constraint_graph()

        # Set demands as node attributes based on objective coefficients
        # d(v_i) = w_i for i > 0, d(v_0) = -sum(w)
        nx.set_node_attributes(G, 0, 'demand')  # Default all to 0
        G.nodes[0]['demand'] = float(-np.sum(w))
        for i in range(self.dcs.num_vars):
            G.nodes[i + 1]['demand'] = float(w[i])

        # Check if total demand is zero (required for feasibility)
        total_demand = sum(G.nodes[n]['demand'] for n in G.nodes())
        if not np.isclose(total_demand, 0):
            return None

        try:
            # Solve minimum cost flow using network simplex
            # demand parameter is the attribute name (default 'demand')
            flow_cost, flow_dict = nx.network_simplex(G, demand='demand', weight='cost')

            # The optimal value is flow_cost + constant
            optimal_value = flow_cost + constant

            if maximize:
                optimal_value = -optimal_value

            return optimal_value

        except (nx.NetworkXUnfeasible, nx.NetworkXUnbounded):
            return None

    def _optimize_dcs_lp(self, w: np.ndarray, constant: float,
                         maximize: bool, use_differentiable: bool = False) -> Optional[float]:
        """
        Optimize linear objective w^T x + constant over DCS using LP

        Fallback method using standard LP solver.
        For hexatopes created from bounds, we assume x ∈ [-1, 1]^n in generator space.

        Args:
            w: Objective coefficients
            constant: Constant term
            maximize: If True, maximize; else minimize
            use_differentiable: If True, use differentiable Gumbel-Softmax solver
        """
        A, b = self.dcs.to_matrix_form()

        # Combine DCS constraints with extra constraints
        if self.extra_A is not None and self.extra_A.shape[0] > 0:
            if A.shape[0] > 0:
                A = np.vstack([A, self.extra_A])
                b = np.concatenate([b, self.extra_b.flatten()])
            else:
                A = self.extra_A
                b = self.extra_b.flatten()

        # Use differentiable solver if requested
        if use_differentiable:
            try:
                from n2v.utils.lpsolver import solve_lp_differentiable

                # Prepare bounds
                lb = np.full(self.dcs.num_vars, -1.0)
                ub = np.full(self.dcs.num_vars, 1.0)

                # Solve
                x_opt, fval, status, _ = solve_lp_differentiable(
                    f=w,
                    A=A if A.shape[0] > 0 else None,
                    b=b if A.shape[0] > 0 else None,
                    lb=lb,
                    ub=ub,
                    minimize=not maximize,
                    num_epochs=50,
                    batch_size=16,
                    verbose=False
                )

                if status == 'optimal' and fval is not None:
                    return fval + constant
                else:
                    return None
            except Exception as e:
                # Fall back to standard LP if differentiable solver fails
                print(f"Differentiable solver failed: {e}, falling back to CVXPY")
                pass

        # Standard CVXPY solver
        x = cp.Variable(self.dcs.num_vars)

        if maximize:
            objective = cp.Maximize(w @ x + constant)
        else:
            objective = cp.Minimize(w @ x + constant)

        constraints = []

        # Add combined constraints if any
        if A.shape[0] > 0:
            constraints.append(A @ x <= b)

        # Add box constraints [-1, 1]^n for generator space
        # This is the key: DCS alone doesn't bound individual variables
        constraints.append(x >= -1)
        constraints.append(x <= 1)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()

            if prob.status in ['optimal', 'optimal_inaccurate']:
                return prob.value
            else:
                return None
        except:
            return None

    # ======================== Set Operations ========================

    def intersect_half_space(self, H: np.ndarray, g: np.ndarray) -> 'Hexatope':
        """
        Intersect hexatope with half-space: H*x <= g

        Section 5.2: Intersection with half-spaces.

        For hexatope H = <c, G, A, b> and halfspace {y | Hy ≤ g},
        the result is H' = <c, G, A', b'> where A'x ≤ b' comprises:
        - Original constraints: Ax ≤ b (DCS)
        - New constraints: HGx ≤ g - Hc (may not be expressible in DCS)

        Uses Algorithm 5.1 (DCSBoundingBox) to compute DCS bounding box.
        Additional linear constraints are stored separately since DCS cannot
        express all linear constraints.

        Args:
            H: Half-space matrix
            g: Half-space vector

        Returns:
            New Hexatope (over-approximation of intersection)
        """
        H = np.asarray(H, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64).reshape(-1, 1)

        # New constraint in generator space: HGx ≤ g - Hc
        constraint_coef = H @ self.generators  # Shape: (m, p) where m = #half-spaces, p = #generators
        constraint_bound = g - H @ self.center.reshape(-1, 1)  # Shape: (m, 1)

        # Combine with existing extra constraints
        if self.extra_A is not None:
            new_extra_A = np.vstack([self.extra_A, constraint_coef])
            new_extra_b = np.vstack([self.extra_b, constraint_bound])
        else:
            new_extra_A = constraint_coef
            new_extra_b = constraint_bound

        # Create new DCS with bounding box (over-approximation)
        new_dcs = self._dcs_bounding_box(self.dcs, constraint_coef, constraint_bound)

        return Hexatope(self.center, self.generators, new_dcs,
                       state_lb=None, state_ub=None,
                       extra_A=new_extra_A, extra_b=new_extra_b)

    def _dcs_bounding_box(self, D: DifferenceConstraintSystem,
                         constraint_coef: np.ndarray,
                         constraint_bound: np.ndarray) -> DifferenceConstraintSystem:
        """
        Algorithm 5.1: DCSBoundingBox (adapted for DCS)

        Compute DCS bounding box of D ∪ {constraint_coef * x ≤ constraint_bound}

        This computes an over-approximation of the intersection by finding
        the tightest difference constraints that bound the intersection.

        Note: DCS constraints alone cannot express absolute bounds on individual
        variables. The LP solver adds box constraints [-1, 1]^n when optimizing,
        which provides the necessary bounding.

        Args:
            D: Original DCS system
            constraint_coef: Coefficients of new constraint
            constraint_bound: Bound of new constraint

        Returns:
            New DCS system over-approximating the intersection
        """
        # Start with a copy of the original DCS to preserve existing constraints
        new_dcs = D.copy()

        # For all pairs of variables x_i, x_j, compute tighter bounds on x_i - x_j
        # by optimizing over D ∪ {new constraint}
        for i in range(D.num_vars):
            for j in range(D.num_vars):
                if i == j:
                    continue

                # Optimize x_i - x_j over D ∪ {constraint}
                # This gives us the tightest bound for the difference constraint x_i - x_j ≤ b_ij
                obj = np.zeros(D.num_vars)
                obj[i] = 1.0   # Coefficient for x_i
                obj[j] = -1.0  # Coefficient for x_j

                # Maximize x_i - x_j to get upper bound
                u_ij = self._optimize_with_constraint(D, obj, constraint_coef,
                                                     constraint_bound, maximize=True)

                if u_ij is not None:
                    # Only add constraint if it's tighter than what we have
                    # Check if this constraint already exists in the original DCS
                    existing_bound = self._find_dcs_constraint(D, i, j)
                    if existing_bound is None or u_ij < existing_bound:
                        # Add tighter constraint x_i - x_j ≤ u_ij
                        new_dcs.add_constraint(i, j, u_ij)

        return new_dcs

    def _find_dcs_constraint(self, D: DifferenceConstraintSystem, i: int, j: int) -> Optional[float]:
        """Find existing bound for x_i - x_j in DCS, returns None if not found"""
        for c in D.constraints:
            if c.i == i and c.j == j:
                return c.b
        return None

    def _optimize_with_constraint(self, D: DifferenceConstraintSystem,
                                  obj: np.ndarray,
                                  constraint_coef: np.ndarray,
                                  constraint_bound: np.ndarray,
                                  maximize: bool) -> Optional[float]:
        """
        Helper: optimize objective over DCS with additional linear constraint

        Solves: max/min obj^T x subject to:
                - Dx ≤ d (original DCS constraints)
                - constraint_coef^T x ≤ constraint_bound (new constraint)
                - x ∈ [-1, 1]^n (box constraints for generator space)

        Args:
            D: DCS system
            obj: Objective vector
            constraint_coef: Coefficients of additional constraint
            constraint_bound: Bound of additional constraint
            maximize: If True, maximize; else minimize

        Returns:
            Optimal value, or None if infeasible
        """
        A, b = D.to_matrix_form()

        x = cp.Variable(D.num_vars)

        if maximize:
            objective = cp.Maximize(obj @ x)
        else:
            objective = cp.Minimize(obj @ x)

        constraints = []

        # Add original DCS constraints
        if A.shape[0] > 0:
            constraints.append(A @ x <= b)

        # Add new constraint
        constraints.append(constraint_coef.flatten() @ x <= constraint_bound.flatten())

        # Add box constraints for generator space
        constraints.append(x >= -1)
        constraints.append(x <= 1)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()

            if prob.status in ['optimal', 'optimal_inaccurate']:
                return prob.value
            else:
                return None
        except:
            return None

    def is_empty_set(self) -> bool:
        """
        Check if Hexatope is empty (constraints are infeasible)

        Returns:
            True if empty, False otherwise
        """
        return not self.dcs.is_feasible()

    # ======================== Utility Methods ========================

    def contains(self, x: np.ndarray) -> bool:
        """
        Check if point x is in the Hexatope

        Args:
            x: Point to check (dim,) or (dim, 1)

        Returns:
            True if x is in the Hexatope
        """
        x = np.asarray(x, dtype=np.float64).flatten()

        if x.shape[0] != self.dim:
            raise ValueError(f"Point dimension {x.shape[0]} doesn't match dim {self.dim}")

        # Solve: find alpha such that G * alpha + c = x and DCS constraints hold
        # This is: G * alpha = x - c

        A, b = self.dcs.to_matrix_form()

        alpha = cp.Variable(self.dcs.num_vars)

        # For overdetermined systems (dim > nVar), use tolerance-based constraints
        # instead of exact equality to handle numerical errors
        tolerance = 1e-6
        diff = self.generators @ alpha - (x - self.center)

        # Use soft constraints: ||G*alpha - (x-c)||_inf <= tolerance
        constraints = [
            diff <= tolerance,
            diff >= -tolerance
        ]

        if A.shape[0] > 0:
            constraints.append(A @ alpha <= b)

        # Add box constraints [-1, 1]^n for generator space
        # This is critical: hexatopes from bounds assume alpha ∈ [-1, 1]^n
        constraints.append(alpha >= -1)
        constraints.append(alpha <= 1)

        # Also check extra constraints if present
        if self.extra_A is not None and self.extra_A.shape[0] > 0:
            constraints.append(self.extra_A @ alpha <= self.extra_b.flatten())

        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            prob.solve()
            return prob.status in ['optimal', 'optimal_inaccurate']
        except:
            return False

    # ======================== Conversion Methods ========================

    def to_star(self) -> 'Star':
        """
        Convert Hexatope to Star set representation

        A Hexatope H = <c, G, DCS> represents {Gx + c : Ax ≤ b, x ∈ [-1, 1]^n}
        where Ax ≤ b is the DCS constraint system.

        The corresponding Star set is:
        - V = [c, G] where c is the center and G columns are generators
        - C = [A; -I; I] where A is DCS matrix, -I/I enforce box bounds
        - d = [b; 1...1; 1...1] where b is DCS bounds, 1s are box bounds

        This conversion is sound: if a point is in the Hexatope, it is also
        in the resulting Star set.

        Returns:
            Star object representing this Hexatope
        """
        from n2v.sets.star import Star

        # Get DCS constraints in matrix form
        A_dcs, b_dcs = self.dcs.to_matrix_form()

        # Build constraint matrix C and bound vector d
        # C includes:
        # 1. DCS constraints: A_dcs * x <= b_dcs
        # 2. Extra constraints (if any): extra_A * x <= extra_b
        # 3. Box constraints: -1 <= x <= 1 (i.e., -x <= 1 and x <= 1)

        n_vars = self.dcs.num_vars

        # Start with DCS constraints
        if A_dcs.shape[0] > 0:
            C_list = [A_dcs]
            d_list = [b_dcs]
        else:
            C_list = []
            d_list = []

        # Add extra constraints if present
        if self.extra_A is not None and self.extra_A.shape[0] > 0:
            C_list.append(self.extra_A)
            d_list.append(self.extra_b.flatten())

        # Add box constraints: x <= 1 and -x <= 1 (i.e., x >= -1)
        C_list.append(np.eye(n_vars))  # x <= 1
        d_list.append(np.ones(n_vars))

        C_list.append(-np.eye(n_vars))  # -x <= 1 (x >= -1)
        d_list.append(np.ones(n_vars))

        # Combine all constraints
        C = np.vstack(C_list)
        d = np.concatenate(d_list).reshape(-1, 1)

        # Build basis matrix V = [c, G]
        # c is the center (dim,), G is the generator matrix (dim, n_vars)
        V = np.hstack([self.center.reshape(-1, 1), self.generators])

        # Create predicate bounds (all generators are in [-1, 1])
        pred_lb = np.full((n_vars, 1), -1.0)
        pred_ub = np.full((n_vars, 1), 1.0)

        # Create Star set
        star = Star(V, C, d, pred_lb=pred_lb, pred_ub=pred_ub,
                   state_lb=self.state_lb, state_ub=self.state_ub)

        return star

    # ======================== Reachability Analysis ========================

    def reach(
        self,
        model: 'nn.Module',
        method: str = 'exact',
        **kwargs
    ) -> List['Hexatope']:
        """
        Perform reachability analysis through a neural network model.

        Args:
            model: PyTorch neural network model
            method: Reachability method to use:
                - 'exact': Exact reachability with exact ReLU handling using CVXPY
                - 'exact-differentiable': Exact reachability using differentiable LP solver
                - 'approx': Over-approximate reachability
            **kwargs: Additional arguments:
                - dis_opt: 'display' to show progress

        Returns:
            List of output Hexatope sets

        Example:
            >>> from n2v.sets import Hexatope
            >>> import torch.nn as nn
            >>> model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 1))
            >>> input_hex = Hexatope.from_bounds(lb, ub)
            >>> # Standard exact method with CVXPY
            >>> output_hexes = input_hex.reach(model, method='exact')
            >>> # Exact method with differentiable solver
            >>> output_hexes = input_hex.reach(model, method='exact-differentiable')
        """
        import torch.nn as nn
        from n2v.nn.reach.reach_hexatope import reach_hexatope_exact, reach_hexatope_approx

        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be a PyTorch nn.Module, got {type(model)}")

        # Determine if we should use differentiable solver
        use_differentiable = (method == 'exact-differentiable')

        if method in ('exact', 'exact-differentiable'):
            return reach_hexatope_exact(
                model, [self],
                use_differentiable=use_differentiable,
                **kwargs
            )
        elif method == 'approx':
            return reach_hexatope_approx(model, [self], **kwargs)
        else:
            raise ValueError(
                f"Unknown method '{method}' for Hexatope reachability. "
                f"Supported methods: 'exact', 'exact-differentiable', 'approx'"
            )
