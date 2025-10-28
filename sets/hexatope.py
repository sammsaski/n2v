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

        Graph nodes correspond directly to DCS variable indices (x_i → node i).

        Only edges corresponding to actual DCS constraints are added. There are no
        unconditional zero-cost edges, as those would create a short-circuit that
        allows MCF to satisfy demands at zero cost, breaking the dual objective.
        """
        G = nx.DiGraph()

        # Add vertices for DCS variables (direct indexing: x_i → node i)
        for i in range(self.num_vars):
            G.add_node(i)

        # Add edges for each difference constraint ONLY
        for dc in self.constraints:
            # Constraint x_i - x_j ≤ b becomes edge (v_j, v_i) with cost b
            G.add_edge(dc.j, dc.i, cost=dc.b, capacity=float('inf'))

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
                 state_ub: Optional[np.ndarray] = None):
        """
        Initialize a hexatope

        Args:
            center: Center vector c ∈ ℝⁿ
            generators: Generator matrix G ∈ ℝⁿˣᵖ
            dcs: Difference constraint system defining the kernel
            state_lb: Lower bounds for state variables (optional)
            state_ub: Upper bounds for state variables (optional)
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

        Uses the anchor variable technique to represent absolute bounds in DCS.

        Implementation Detail:
        ----------------------
        A pure DCS cannot express absolute bounds on variables without a reference point.
        We introduce an "anchor variable" x_0 that serves as the reference (conceptually x_0 = 0).

        For each generator variable x_i ∈ [-1, 1], we encode:
        - x_i - x_0 ≤ 1   (upper bound: x_i ≤ 1 when x_0 = 0)
        - x_0 - x_i ≤ 1   (lower bound: x_i ≥ -1 when x_0 = 0)

        The anchor variable has a ZERO generator column, so it doesn't affect
        the affine transformation y = Gx + c. All affine operations work correctly
        because the anchor column contributes 0 to all output dimensions.

        Kernel structure:
        - Index 0: anchor variable x_0 (generator column is zero)
        - Index 1..n: actual generator variables corresponding to box dimensions

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

        # Generators: (n × n+1) matrix
        # - Column 0 (anchor): all zeros (doesn't affect affine map)
        # - Columns 1..n: diagonal with half-widths
        half_widths = (ub - lb) / 2
        generators = np.zeros((n, n + 1))
        generators[:, 0] = 0  # Anchor column: zero vector
        generators[:, 1:] = np.diag(half_widths)  # Box generators

        # DCS with n+1 variables (index 0 = anchor, indices 1..n = box dimensions)
        dcs = DifferenceConstraintSystem(n + 1)

        # Encode bounds [-1, 1] for each generator variable using anchor
        # x_i - x_0 ≤ 1 and x_0 - x_i ≤ 1 means -1 ≤ x_i - x_0 ≤ 1
        # When x_0 = 0 (fixed in LP), this gives -1 ≤ x_i ≤ 1
        for i in range(1, n + 1):
            # x_i - x_0 ≤ 1   (upper bound)
            dcs.add_constraint(i, 0, 1.0)
            # x_0 - x_i ≤ 1   (lower bound: x_i ≥ -1)
            dcs.add_constraint(0, i, 1.0)

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

        # Kernel constraints remain unchanged - deep copy to avoid aliasing bugs
        return Hexatope(new_center, new_generators, self.dcs.copy())

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

        # Sequential version - use MCF by default with LP fallback
        for i in range(self.dim):
            lb_i, ub_i = self.get_range(i, use_mcf=use_mcf)
            if lb_i is None or ub_i is None:
                # Try LP fallback if MCF failed
                if use_mcf:
                    lb_i, ub_i = self.get_range(i, use_mcf=False)
                # Final fallback to estimation if both failed
                if lb_i is None or ub_i is None:
                    lb_i, ub_i = self.estimate_range(i)
            lb[i] = lb_i
            ub[i] = ub_i

        return lb, ub

    def get_bounds(self, use_mcf: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alias for get_ranges() for API consistency with other set types.

        Args:
            use_mcf: If True, use min-cost flow; else use LP (default: MCF)

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
                       use_mcf: bool = True, use_differentiable: bool = False) -> Optional[float]:
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
            use_differentiable: If True, use differentiable solver (MCF or LP)

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
            result = self._optimize_dcs_mcf(composed_obj, constant_term, maximize, use_differentiable)
        else:
            result = self._optimize_dcs_lp(composed_obj, constant_term, maximize, use_differentiable)

        return result

    def _optimize_dcs_mcf(self, w: np.ndarray, constant: float,
                          maximize: bool, use_differentiable: bool = False) -> Optional[float]:
        """
        Optimize linear objective w^T x + constant over DCS using min-cost flow

        Reduces to minimum cost flow problem on constraint graph.
        Can use either NetworkX's network_simplex (traditional) or
        differentiable DCS solver (gradient-based).

        Args:
            w: Objective coefficients
            constant: Constant term
            maximize: If True, maximize; else minimize
            use_differentiable: If True, use differentiable solver

        Returns:
            Optimal value, or None if infeasible
        """
        # Build constraint graph
        G = self.dcs.to_constraint_graph()

        if use_differentiable:
            # Use differentiable DCS solver
            try:
                from n2v.utils.lpsolver import solve_dcs_differentiable

                optimal_value, info = solve_dcs_differentiable(
                    constraint_graph=G,
                    objective_coef=w,
                    constant_term=constant,
                    maximize=maximize,
                    num_epochs=100,
                    batch_size=32,
                    verbose=False
                )

                return optimal_value

            except Exception as e:
                # Fall back to NetworkX if differentiable solver fails
                print(f"Differentiable DCS solver failed: {e}, falling back to network_simplex")
                use_differentiable = False

        if not use_differentiable:
            # Use traditional network simplex (NetworkX)
            # MCF with demands=w computes max w^T x (dual objective = primal max)
            # To compute min: feed -w and negate result
            # To compute max: feed +w directly
            if maximize:
                w_adj = w
                constant_adj = constant
                sgn = +1
            else:
                w_adj = -w
                constant_adj = -constant
                sgn = -1

            # Set demands as node attributes based on objective coefficients
            # Balance total demand at the anchor node (x_0, node 0)
            # This is the dual analogue of the primal constraint x_0 = 0
            d = w_adj.astype(float).copy()
            d[0] -= d.sum()  # Soak imbalance into anchor; now sum(d) = 0

            for i in range(self.dcs.num_vars):
                G.nodes[i]['demand'] = float(d[i])

            # Verify total demand is zero (required for MCF feasibility)
            total_demand = sum(G.nodes[n]['demand'] for n in G.nodes())
            if not np.isclose(total_demand, 0):
                return None

            try:
                # Solve minimum cost flow using network simplex
                flow_cost, flow_dict = nx.network_simplex(G, demand='demand', weight='cost')

                # The optimal value with correct sign
                optimal_value = sgn * (flow_cost + constant_adj)

                return optimal_value

            except nx.NetworkXUnfeasible:
                # DCS constraints are infeasible
                return None
            except nx.NetworkXUnbounded:
                # DCS is unbounded in objective direction
                # This should not happen with proper anchor constraints
                # but handle gracefully by returning None
                return None
            except Exception as e:
                # Catch any other MCF solver errors and fall back
                print(f"MCF solver error: {e}, returning None")
                return None

    def _optimize_dcs_lp(self, w: np.ndarray, constant: float,
                         maximize: bool, use_differentiable: bool = False) -> Optional[float]:
        """
        Optimize linear objective w^T x + constant over DCS using LP

        Fallback method using standard LP solver.
        Bounds on variables come from the DCS constraints themselves (via anchor variable).

        Args:
            w: Objective coefficients
            constant: Constant term
            maximize: If True, maximize; else minimize
            use_differentiable: If True, use differentiable Gumbel-Softmax solver
        """
        A, b = self.dcs.to_matrix_form()

        # Use differentiable DCS solver if requested
        if use_differentiable:
            try:
                from n2v.utils.lpsolver import solve_dcs_differentiable

                # Build constraint graph
                G = self.dcs.to_constraint_graph()

                # Solve using differentiable DCS solver
                optimal_value, info = solve_dcs_differentiable(
                    constraint_graph=G,
                    objective_coef=w,
                    constant_term=constant,
                    maximize=maximize,
                    num_epochs=50,
                    batch_size=16,
                    verbose=False
                )

                return optimal_value
            except Exception as e:
                # Fall back to standard LP if differentiable solver fails
                print(f"Differentiable DCS solver failed: {e}, falling back to CVXPY")
                pass

        # Standard CVXPY solver
        x = cp.Variable(self.dcs.num_vars)

        if maximize:
            objective = cp.Maximize(w @ x + constant)
        else:
            objective = cp.Minimize(w @ x + constant)

        constraints = []

        # Add DCS constraints
        if A.shape[0] > 0:
            constraints.append(A @ x <= b)

        # Fix anchor variable (index 0) to zero for LP
        # Note: MCF handles this differently through demand structure,
        # but LP needs explicit bound to avoid unbounded problem
        constraints.append(x[0] == 0)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7)

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
        we compute the tightest DCS bounding box that over-approximates
        the intersection. This keeps the result in the DCS template domain.

        Uses Algorithm 5.1 (DCSBoundingBox) to compute DCS bounding box.

        Following ChatGPT's recommendation: process constraints incrementally row-by-row
        for empirically tighter and simpler results.

        Note: This is an OVER-APPROXIMATION. The true intersection may include
        points outside the DCS template. To get exactness, use star sets with
        hexatope/octatope prefilters (as described in Section 5.3).

        Args:
            H: Half-space matrix (m × n)
            g: Half-space vector (m × 1)

        Returns:
            New Hexatope (over-approximation of intersection, DCS-only)
        """
        H = np.asarray(H, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64).reshape(-1, 1)

        # Ensure H is 2D
        if len(H.shape) == 1:
            H = H.reshape(1, -1)

        # New constraint in generator space: HGx ≤ g - Hc
        constraint_coef = H @ self.generators  # (m × p) where m = #half-spaces, p = #generators
        constraint_bound = g - H @ self.center.reshape(-1, 1)  # (m × 1)

        # Process constraints incrementally (row-by-row) for tighter results
        current_dcs = self.dcs
        for k in range(constraint_coef.shape[0]):
            # Extract single row
            row_coef = constraint_coef[k:k+1, :]  # Keep 2D: (1 × p)
            row_bound = constraint_bound[k:k+1, :]  # Keep 2D: (1 × 1)

            # Compute bounding box with this constraint
            current_dcs = self._dcs_bounding_box(current_dcs, row_coef, row_bound)

        return Hexatope(self.center, self.generators, current_dcs,
                       state_lb=None, state_ub=None)

    def _dcs_bounding_box(self, D: DifferenceConstraintSystem,
                         constraint_coef: np.ndarray,
                         constraint_bound: np.ndarray,
                         use_mcf: bool = True) -> DifferenceConstraintSystem:
        """
        Algorithm 5.1: DCSBoundingBox (adapted for DCS)

        Compute DCS bounding box of D ∪ {constraint_coef * x ≤ constraint_bound}

        This computes an over-approximation of the intersection by finding
        the tightest difference constraints that bound the intersection.

        According to the paper (Section 5.2), these inner optimizations can and
        should use MCF instead of LP for better performance.

        Args:
            D: Original DCS system
            constraint_coef: Coefficients of new constraint (should be single row)
            constraint_bound: Bound of new constraint (should be single value)
            use_mcf: If True, use MCF for inner optimizations; else use LP

        Returns:
            New DCS system over-approximating the intersection
        """
        # Ensure constraint_coef is 2D for processing
        constraint_coef = np.atleast_2d(constraint_coef)
        constraint_bound = np.atleast_1d(constraint_bound.flatten())

        # Fast-path: If constraint is already DCS-expressible, add it directly
        # Also handle normalized constraints (positive scalar multiples)
        if constraint_coef.shape[0] == 1:
            row = constraint_coef[0, :]
            nonzero_indices = np.nonzero(row)[0]

            # Check if DCS-expressible (exactly 2 nonzeros)
            if len(nonzero_indices) == 2:
                vals = row[nonzero_indices]
                # Check if it's a scaled version of [+1, -1] or [-1, +1]
                # i.e., vals = k * [+1, -1] for some k > 0
                sorted_vals = sorted(vals)
                if sorted_vals[0] < 0 and sorted_vals[1] > 0:
                    # Positive scalar multiple check: vals[1] / vals[0] should be -1
                    if np.isclose(sorted_vals[1] / (-sorted_vals[0]), 1.0):
                        # This is a normalized DCS constraint: k*(x_i - x_j) <= b
                        # Normalize to: x_i - x_j <= b/k
                        scale = sorted_vals[1]  # The positive coefficient
                        normalized_bound = constraint_bound[0] / scale

                        new_dcs = D.copy()
                        i_idx = nonzero_indices[np.argmax(vals)]  # Index with positive coef
                        j_idx = nonzero_indices[np.argmin(vals)]  # Index with negative coef
                        new_dcs.add_constraint(i_idx, j_idx, normalized_bound)
                        return new_dcs

        # Fall back to full bounding box algorithm for non-DCS constraints
        # This is expensive: O(n²) optimizations
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
                # Paper suggests using MCF here for performance
                u_ij = self._optimize_with_constraint(D, obj, constraint_coef,
                                                     constraint_bound, maximize=True,
                                                     use_mcf=use_mcf)

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
                                  maximize: bool,
                                  use_mcf: bool = True) -> Optional[float]:
        """
        Helper: optimize objective over DCS with additional linear constraints

        Solves: max/min obj^T x subject to:
                - Dx ≤ d (original DCS constraints, including anchor bounds)
                - constraint_coef @ x ≤ constraint_bound (new constraints, may be multiple rows)

        According to the paper, when the new constraint can be expressed in DCS,
        we can use MCF. Otherwise, we fall back to LP.

        Args:
            D: DCS system
            obj: Objective vector
            constraint_coef: Coefficients of additional constraints (m × p matrix, m >= 1)
            constraint_bound: Bounds of additional constraints (m × 1 vector)
            maximize: If True, maximize; else minimize
            use_mcf: If True, try to use MCF fast-path when possible

        Returns:
            Optimal value, or None if infeasible
        """
        constraint_coef = np.atleast_2d(constraint_coef)  # Ensure 2D
        constraint_bound = np.atleast_1d(constraint_bound.flatten())  # Ensure 1D

        # MCF fast-path: Check if ALL added constraints are DCS-expressible
        # A constraint is DCS-expressible if it has exactly two nonzeros: +1 and -1
        # (representing x_i - x_j ≤ b)
        all_dcs_expressible = True
        dcs_constraints = []  # List of (i, j, b) for each DCS-expressible constraint

        if use_mcf:
            for k in range(constraint_coef.shape[0]):
                row = constraint_coef[k, :]
                nonzero_indices = np.nonzero(row)[0]

                # Check: exactly 2 nonzeros with values +1 and -1
                if len(nonzero_indices) == 2:
                    vals = row[nonzero_indices]
                    if np.allclose(sorted(vals), [-1.0, 1.0]):
                        # DCS-expressible: x_i - x_j ≤ b
                        i_idx = nonzero_indices[np.argmax(vals)]  # Index with +1
                        j_idx = nonzero_indices[np.argmin(vals)]  # Index with -1
                        dcs_constraints.append((i_idx, j_idx, constraint_bound[k]))
                    else:
                        all_dcs_expressible = False
                        break
                else:
                    all_dcs_expressible = False
                    break

        # If all constraints are DCS-expressible, use MCF fast-path
        if use_mcf and all_dcs_expressible and len(dcs_constraints) > 0:
            # Create augmented DCS with additional constraints
            D_aug = D.copy()
            for i, j, b in dcs_constraints:
                D_aug.add_constraint(i, j, b)

            # Create temporary hexatope for MCF optimization
            temp_center = np.zeros(D_aug.num_vars)
            temp_generators = np.eye(D_aug.num_vars)
            temp_hex = Hexatope(temp_center, temp_generators, D_aug)

            # Use MCF to optimize
            try:
                result = temp_hex._optimize_dcs_mcf(obj, 0.0, maximize, use_differentiable=False)
                if result is not None:
                    return result
            except Exception:
                # Fall back to LP if MCF fails
                pass

        # Fall back to LP for non-DCS constraints or if MCF failed
        A, b = D.to_matrix_form()

        x = cp.Variable(D.num_vars)

        if maximize:
            objective = cp.Maximize(obj @ x)
        else:
            objective = cp.Minimize(obj @ x)

        constraints = []

        # Add original DCS constraints (includes bounds via anchor)
        if A.shape[0] > 0:
            constraints.append(A @ x <= b)

        # Fix anchor variable (index 0) to zero for LP
        constraints.append(x[0] == 0)

        # Add new constraints (iterate over all rows if multiple)
        for k in range(constraint_coef.shape[0]):
            constraints.append(constraint_coef[k, :] @ x <= constraint_bound[k])

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7)

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

    def contains(self, x: np.ndarray, tolerance: float = 1e-7) -> bool:
        """
        Check if point x is in the Hexatope

        Uses two-phase approach per ChatGPT V3 feedback:
        1. Fast-path: Least-squares solve to propose alpha
        2. Explicit verification: Check residuals and constraints
        3. Fallback: CVXPY feasibility LP if fast-path fails

        This prevents false positives from solver inaccuracies.

        Args:
            x: Point to check (dim,) or (dim, 1)
            tolerance: Numerical tolerance for feasibility checks

        Returns:
            True if x is in the Hexatope
        """
        x = np.asarray(x, dtype=np.float64).flatten()

        if x.shape[0] != self.dim:
            raise ValueError(f"Point dimension {x.shape[0]} doesn't match dim {self.dim}")

        # Target: G * alpha = x - c
        target = x - self.center
        A, b = self.dcs.to_matrix_form()

        # Phase 1: Fast-path least-squares solve
        # For hexatope, enforce alpha[0] = 0 (anchor variable)
        try:
            # Use least-squares to propose alpha
            # Note: Need to handle anchor constraint alpha[0] = 0
            # Approach: Solve only for alpha[1:], set alpha[0] = 0

            G_reduced = self.generators[:, 1:]  # Remove anchor column (all zeros anyway)
            if G_reduced.shape[1] > 0:
                alpha_reduced, residuals, rank, s = np.linalg.lstsq(G_reduced, target, rcond=None)
                # Reconstruct full alpha with anchor = 0
                alpha_proposed = np.zeros(self.dcs.num_vars)
                alpha_proposed[1:] = alpha_reduced
            else:
                # Edge case: only anchor variable
                alpha_proposed = np.zeros(self.dcs.num_vars)

            # Verify feasibility explicitly
            # 1. Check residual: ||G*alpha - target||_inf <= tol
            residual = self.generators @ alpha_proposed - target
            if np.linalg.norm(residual, ord=np.inf) > tolerance:
                # Residual too large, try LP fallback
                pass
            else:
                # 2. Check anchor: alpha[0] == 0
                if not np.isclose(alpha_proposed[0], 0.0, atol=tolerance):
                    pass  # Anchor violated, try LP fallback
                else:
                    # 3. Check DCS constraints: A*alpha <= b + tol
                    if A.shape[0] > 0:
                        constraint_violations = A @ alpha_proposed - b
                        if np.max(constraint_violations) > tolerance:
                            pass  # Constraints violated, try LP fallback
                        else:
                            # All checks passed - fast-path success
                            return True
                    else:
                        # No DCS constraints, residual check sufficient
                        return True

        except np.linalg.LinAlgError:
            # Least-squares failed, fall through to LP
            pass

        # Phase 2: Fallback to CVXPY feasibility LP with OSQP solver
        # Use OSQP (not SCS) for better accuracy in feasibility contexts
        alpha = cp.Variable(self.dcs.num_vars)

        diff = self.generators @ alpha - target

        # Use soft constraints: ||G*alpha - target||_inf <= tolerance
        constraints = [
            diff <= tolerance,
            diff >= -tolerance
        ]

        # Add DCS constraints (includes bounds via anchor)
        if A.shape[0] > 0:
            constraints.append(A @ alpha <= b)

        # Fix anchor variable (index 0) to zero
        constraints.append(alpha[0] == 0)

        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            # Use OSQP solver with tight tolerances
            prob.solve(solver=cp.OSQP, eps_abs=tolerance, eps_rel=tolerance, verbose=False)

            if prob.status not in ['optimal', 'optimal_inaccurate']:
                return False

            # Explicit post-solve verification (prevents false positives)
            alpha_val = alpha.value
            if alpha_val is None:
                return False

            # Recheck residual
            residual = self.generators @ alpha_val - target
            if np.linalg.norm(residual, ord=np.inf) > tolerance:
                return False

            # Recheck anchor
            if not np.isclose(alpha_val[0], 0.0, atol=tolerance):
                return False

            # Recheck DCS constraints
            if A.shape[0] > 0:
                constraint_violations = A @ alpha_val - b
                if np.max(constraint_violations) > tolerance:
                    return False

            return True

        except Exception:
            return False

    # ======================== Conversion Methods ========================

    def to_star(self) -> 'Star':
        """
        Convert Hexatope to Star set representation

        A Hexatope H = <c, G, DCS> represents {Gx + c : Ax ≤ b}
        where Ax ≤ b is the DCS constraint system (including anchor variable bounds).

        The corresponding Star set is:
        - V = [c, G] where c is the center and G columns are generators
        - C = A where A is the DCS matrix
        - d = b where b is the DCS bounds

        Note: The anchor variable (index 0) has a zero generator column,
        so it doesn't affect the affine transformation but does provide
        bounds for the other variables through DCS constraints.

        This conversion is sound: if a point is in the Hexatope, it is also
        in the resulting Star set.

        Returns:
            Star object representing this Hexatope
        """
        from n2v.sets.star import Star

        # Get DCS constraints in matrix form (includes anchor bounds)
        A_dcs, b_dcs = self.dcs.to_matrix_form()

        n_vars = self.dcs.num_vars

        # Build constraint matrix C and bound vector d
        if A_dcs.shape[0] > 0:
            C = A_dcs
            d = b_dcs.reshape(-1, 1)
        else:
            # Empty constraints (shouldn't happen with anchor, but handle gracefully)
            C = np.zeros((0, n_vars))
            d = np.zeros((0, 1))

        # Build basis matrix V = [c, G]
        # c is the center (dim,), G is the generator matrix (dim, n_vars)
        V = np.hstack([self.center.reshape(-1, 1), self.generators])

        # Predicate bounds: Hexatope generator variables are bounded in [-1, 1]
        # The DCS constraints enforce this through the anchor variable
        # Star expects predicate_lb and predicate_ub to be (nVar, 1) arrays
        pred_lb = -np.ones((n_vars, 1))
        pred_ub = np.ones((n_vars, 1))

        # Create Star set
        star = Star(V, C, d, pred_lb=pred_lb, pred_ub=pred_ub,
                   state_lb=self.state_lb, state_ub=self.state_ub)

        return star

    # ======================== Reachability Analysis ========================
    # Note: Reachability analysis should be performed through NeuralNetwork.reach()
    # instead of calling reach() on set objects directly. This maintains proper
    # separation of concerns where sets represent geometric objects and reachability
    # is a neural network operation.
    #
    # Example usage:
    #     from n2v.nn import NeuralNetwork
    #     from n2v.sets import Hexatope
    #     import torch.nn as nn
    #
    #     model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 1))
    #     net = NeuralNetwork(model)
    #     input_hex = Hexatope.from_bounds(lb, ub)
    #     # Standard exact method with CVXPY
    #     output_hexes = net.reach(input_hex, method='exact')
    #     # Exact method with differentiable solver
    #     output_hexes = net.reach(input_hex, method='exact-differentiable')
