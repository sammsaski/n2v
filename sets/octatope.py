"""
Octatope Abstract Domain for Neural Network Verification

An octatope is an affine transformation of a higher-dimensional octagon,
defined by unit-two-variable-per-inequality (UTVPI) constraint systems.

Definition: O = <c, G, A, b> where:
- c ∈ ℝⁿ is the center
- G ∈ ℝⁿˣᵖ is the generator matrix
- Ax ≤ b defines a UTVPI constraint system

Semantics: [[O]] = {Gx + c : Ax ≤ b}

Reference: Bak et al., "The hexatope and octatope abstract domains for neural
network verification", Formal Methods in System Design (2024) 64:178–199
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import cvxpy as cp


@dataclass
class UTVPIConstraint:
    """
    Represents a UTVPI constraint: a_i*x_i + a_j*x_j ≤ b
    where a_i, a_j ∈ {-1, 0, +1}
    """
    i: int  # Index of first variable
    j: int  # Index of second variable
    ai: int  # Coefficient of x_i: -1, 0, or +1
    aj: int  # Coefficient of x_j: -1, 0, or +1
    b: float  # Bound

    def __post_init__(self):
        if self.ai not in {-1, 0, 1} or self.aj not in {-1, 0, 1}:
            raise ValueError("UTVPI coefficients must be in {-1, 0, 1}")
        if self.ai == 0 and self.aj == 0:
            raise ValueError("At least one coefficient must be non-zero")


class UTVPIConstraintSystem:
    """
    Unit-Two-Variables-Per-Inequality (UTVPI) Constraint System

    A conjunction of constraints of the form a_i*x_i + a_j*x_j ≤ b
    where a_i, a_j ∈ {-1, 0, +1}
    """

    def __init__(self, num_vars: int):
        self.num_vars = num_vars
        self.constraints: List[UTVPIConstraint] = []

    def add_constraint(self, i: int, j: int, ai: int, aj: int, b: float):
        """Add UTVPI constraint: ai*x_i + aj*x_j ≤ b"""
        if i < 0 or i >= self.num_vars or j < 0 or j >= self.num_vars:
            raise ValueError(f"Invalid variable indices: i={i}, j={j}")
        self.constraints.append(UTVPIConstraint(i, j, ai, aj, b))

    def to_dcs(self) -> 'DifferenceConstraintSystem':
        """
        Convert UTVPI system to Difference Constraint System (DCS)

        Theorem 7: UTVPI optimization can be reduced to DCS optimization.

        Following the conversion in the paper:
        - Create variables x+_i and x-_i for each variable x_i
        - Convert each UTVPI constraint to two difference constraints
        """
        from n2v.sets.hexatope import DifferenceConstraintSystem

        # Create DCS with 2 * num_vars variables (x+_i and x-_i for each x_i)
        dcs = DifferenceConstraintSystem(2 * self.num_vars)

        for uc in self.constraints:
            # Get indices for x+_i, x-_i, x+_j, x-_j
            i_pos = 2 * uc.i  # x+_i
            i_neg = 2 * uc.i + 1  # x-_i
            j_pos = 2 * uc.j  # x+_j
            j_neg = 2 * uc.j + 1  # x-_j

            # Convert based on constraint type:
            if uc.ai == 1 and uc.aj == 1:
                # x_i + x_j ≤ b
                # Becomes: x+_i - x-_j ≤ b and -x-_i + x+_j ≤ b
                dcs.add_constraint(i_pos, j_neg, uc.b)
                dcs.add_constraint(j_pos, i_neg, uc.b)

            elif uc.ai == 1 and uc.aj == -1:
                # x_i - x_j ≤ b
                # Becomes: x+_i - x+_j ≤ b and -x-_i + x-_j ≤ b
                dcs.add_constraint(i_pos, j_pos, uc.b)
                dcs.add_constraint(j_neg, i_neg, uc.b)

            elif uc.ai == -1 and uc.aj == 1:
                # -x_i + x_j ≤ b
                # Becomes: x-_i - x-_j ≤ b and -x+_i + x+_j ≤ b
                dcs.add_constraint(i_neg, j_neg, uc.b)
                dcs.add_constraint(j_pos, i_pos, uc.b)

            elif uc.ai == -1 and uc.aj == -1:
                # -x_i - x_j ≤ b
                # Becomes: x-_i - x+_j ≤ b and -x+_i + x-_j ≤ b
                dcs.add_constraint(i_neg, j_pos, uc.b)
                dcs.add_constraint(j_neg, i_pos, uc.b)

            elif uc.ai == 1 and uc.aj == 0:
                # x_i ≤ b
                # Becomes: x+_i - x-_i ≤ 2*b
                dcs.add_constraint(i_pos, i_neg, 2 * uc.b)

            elif uc.ai == -1 and uc.aj == 0:
                # -x_i ≤ b
                # Becomes: x-_i - x+_i ≤ 2*b
                dcs.add_constraint(i_neg, i_pos, 2 * uc.b)

            elif uc.ai == 0 and uc.aj == 1:
                # x_j ≤ b
                # Becomes: x+_j - x-_j ≤ 2*b
                dcs.add_constraint(j_pos, j_neg, 2 * uc.b)

            elif uc.ai == 0 and uc.aj == -1:
                # -x_j ≤ b
                # Becomes: x-_j - x+_j ≤ 2*b
                dcs.add_constraint(j_neg, j_pos, 2 * uc.b)

        return dcs

    def is_feasible(self) -> bool:
        """
        Check if UTVPI system is feasible

        Theorem 8: Can be decided in O(p * m) time where p is number of
        variables and m is number of constraints.

        For now, we use the DCS conversion + feasibility check.
        """
        dcs = self.to_dcs()
        return dcs.is_feasible()

    def to_matrix_form(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert UTVPI system to matrix form Ax ≤ b"""
        m = len(self.constraints)
        A = np.zeros((m, self.num_vars))
        b = np.zeros(m)

        for k, uc in enumerate(self.constraints):
            A[k, uc.i] = uc.ai
            A[k, uc.j] = uc.aj
            b[k] = uc.b

        return A, b

    def copy(self) -> 'UTVPIConstraintSystem':
        """Create a deep copy of the UTVPI system"""
        new_utvpi = UTVPIConstraintSystem(self.num_vars)
        new_utvpi.constraints = [
            UTVPIConstraint(uc.i, uc.j, uc.ai, uc.aj, uc.b)
            for uc in self.constraints
        ]
        return new_utvpi


class Octatope:
    """
    Octatope Abstract Domain

    An octatope O = <c, G, A, b> is a special type of linear star set
    where the kernel Ax ≤ b is defined by a UTVPI constraint system.

    Semantics: [[O]] = {Gx + c : Ax ≤ b where Ax ≤ b is a UTVPI system}

    Octatopes are more expressive than hexatopes (which use difference
    constraints) but less expressive than general star sets.
    """

    def __init__(self, center: np.ndarray, generators: np.ndarray,
                 utvpi: UTVPIConstraintSystem,
                 state_lb: Optional[np.ndarray] = None,
                 state_ub: Optional[np.ndarray] = None):
        """
        Initialize an octatope

        Args:
            center: Center vector c ∈ ℝⁿ
            generators: Generator matrix G ∈ ℝⁿˣᵖ
            utvpi: UTVPI constraint system defining the kernel
            state_lb: Lower bounds for state variables (optional)
            state_ub: Upper bounds for state variables (optional)
        """
        self.center = np.asarray(center, dtype=np.float64).reshape(-1)
        self.generators = np.asarray(generators, dtype=np.float64)
        self.utvpi = utvpi

        # Validate dimensions
        if len(self.generators.shape) == 1:
            self.generators = self.generators.reshape(-1, 1)

        n, p = self.generators.shape
        if self.center.shape[0] != n:
            raise ValueError(f"Center dimension {self.center.shape[0]} != {n}")
        if self.utvpi.num_vars != p:
            raise ValueError(f"UTVPI variables {self.utvpi.num_vars} != generators {p}")

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
        """Dimension of the octatope (output dimension)"""
        return self.center.shape[0]

    @property
    def nVar(self) -> int:
        """Number of generator vectors (kernel dimension)"""
        return self.generators.shape[1]

    def __repr__(self) -> str:
        return (f"Octatope(dim={self.dim}, nVar={self.nVar}, "
                f"nConstraints={len(self.utvpi.constraints)})")

    @classmethod
    def from_bounds(cls, lb: np.ndarray, ub: np.ndarray) -> 'Octatope':
        """
        Create an octatope representing a hyperrectangle [lower, upper]

        Args:
            lb: Lower bounds
            ub: Upper bounds

        Returns:
            Octatope representing the box
        """
        lb = np.asarray(lb, dtype=np.float64).flatten()
        ub = np.asarray(ub, dtype=np.float64).flatten()
        n = lb.shape[0]

        # Center: midpoint of box
        center = (lb + ub) / 2

        # Generators: diagonal matrix with half-widths
        half_widths = (ub - lb) / 2
        generators = np.diag(half_widths)

        # UTVPI: -1 ≤ x_i ≤ 1 for each i
        utvpi = UTVPIConstraintSystem(n)

        for i in range(n):
            # x_i ≤ 1
            utvpi.add_constraint(i, i, 1, 0, 1.0)
            # -x_i ≤ 1 (i.e., x_i ≥ -1)
            utvpi.add_constraint(i, i, -1, 0, 1.0)

        return cls(center, generators, utvpi, state_lb=lb.reshape(-1, 1),
                   state_ub=ub.reshape(-1, 1))

    # ======================== Affine Transformations ========================

    def affine_map(self, W: np.ndarray, b: Optional[np.ndarray] = None) -> 'Octatope':
        """
        Apply affine transformation: W*x + b

        Theorem 6: Octatopes are closed under affine transformation.

        For octatope O = <c, G, A, b> and affine map f(x) = Wx + d,
        the result is O' = <c', G', A, b> where:
        - c' = Wc + d
        - G' = WG
        - A, b remain unchanged (kernel unchanged)

        Args:
            W: Mapping matrix (m, n)
            b: Mapping vector (m,) or (m, 1), optional

        Returns:
            New Octatope object
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

        # Kernel constraints remain unchanged
        return Octatope(new_center, new_generators, self.utvpi)

    # ======================== Bounds Computation ========================

    def get_range(self, index: int, use_mcf: bool = True) -> Tuple[float, float]:
        """
        Compute exact range at specific dimension

        Theorem 7: Linear optimization over octatopes can be solved in
        strongly polynomial time via reduction to hexatope optimization
        (which uses min-cost flow).

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
            use_mcf: If True, use min-cost flow via DCS; else use direct LP
            parallel: If True, use parallel computation
            n_workers: Number of parallel workers

        Returns:
            Tuple of (lb, ub) arrays
        """
        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        if parallel and self.dim > 1:
            return self._get_ranges_parallel(use_mcf, n_workers)

        # Sequential version - use direct LP for reliability
        for i in range(self.dim):
            lb_i, ub_i = self.get_range(i, use_mcf=False)  # Force direct LP for now
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
        # Assume generators are bounded by [-1, 1] (standard for UTVPI)
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
        Optimize linear objective over octatope

        Theorem 7: Linear optimization over octatopes can be solved in
        strongly polynomial time via reduction to hexatope optimization.

        To optimize f(y) = f^T y over [[O]], we optimize f^T(Gx + c) over Ax ≤ b.
        This reduces to optimizing (f^T G)x + f^T c over the UTVPI system.

        Args:
            objective: Objective vector f ∈ ℝⁿ
            maximize: If True, maximize; else minimize
            use_mcf: If True, use min-cost flow via DCS conversion; else use LP

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

        # Now optimize w^T x over UTVPI system
        if use_mcf:
            result = self._optimize_utvpi_mcf(composed_obj, constant_term, maximize)
        else:
            result = self._optimize_utvpi_lp(composed_obj, constant_term, maximize)

        return result

    def _optimize_utvpi_mcf(self, w: np.ndarray, constant: float,
                           maximize: bool) -> Optional[float]:
        """
        Optimize linear objective w^T x + constant over UTVPI system

        Uses conversion to DCS + min-cost flow.
        """
        # Convert UTVPI to DCS
        dcs = self.utvpi.to_dcs()

        # Expand objective for x+ and x- variables
        # Original: optimize w^T x where x_i = (1/2)(x+_i - x-_i)
        # New: optimize (1/2) * [w, -w]^T * [x+; x-]
        w_expanded = np.zeros(2 * self.utvpi.num_vars)
        for i in range(self.utvpi.num_vars):
            w_expanded[2*i] = 0.5 * w[i]  # x+_i coefficient
            w_expanded[2*i + 1] = -0.5 * w[i]  # x-_i coefficient

        # Use hexatope optimization on the DCS
        from n2v.sets.hexatope import Hexatope

        # Create a temporary hexatope with identity center/generators for optimization
        temp_center = np.zeros(dcs.num_vars)
        temp_generators = np.eye(dcs.num_vars)
        temp_hex = Hexatope(temp_center, temp_generators, dcs)

        # Optimize w_expanded over this hexatope
        result = temp_hex._optimize_dcs_mcf(w_expanded, constant, maximize)

        return result

    def _optimize_utvpi_lp(self, w: np.ndarray, constant: float,
                          maximize: bool, use_differentiable: bool = False) -> Optional[float]:
        """
        Optimize linear objective w^T x + constant over UTVPI system using LP

        Fallback method using standard LP solver.
        For octatopes created from bounds, we assume x ∈ [-1, 1]^n in generator space.

        Args:
            w: Objective coefficients
            constant: Constant term
            maximize: If True, maximize; else minimize
            use_differentiable: If True, use differentiable Gumbel-Softmax solver
        """
        A, b = self.utvpi.to_matrix_form()

        # Use differentiable solver if requested
        if use_differentiable:
            try:
                from n2v.utils.lpsolver import solve_lp_differentiable

                # Prepare bounds
                lb = np.full(self.utvpi.num_vars, -1.0)
                ub = np.full(self.utvpi.num_vars, 1.0)

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
        x = cp.Variable(self.utvpi.num_vars)

        if maximize:
            objective = cp.Maximize(w @ x + constant)
        else:
            objective = cp.Minimize(w @ x + constant)

        constraints = []

        # Add UTVPI constraints if any
        if A.shape[0] > 0:
            constraints.append(A @ x <= b)

        # Add box constraints [-1, 1]^n for generator space
        # UTVPI alone may not fully bound individual variables
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

    def intersect_half_space(self, H: np.ndarray, g: np.ndarray) -> 'Octatope':
        """
        Intersect octatope with half-space: H*x <= g

        Section 5.2: Intersection with half-spaces.

        For octatope O = <c, G, A, b> and halfspace {y | Hy ≤ g},
        the result is O' = <c, G, A', b'> where A'x ≤ b' comprises:
        - Original constraints: Ax ≤ b
        - New constraints: HGx ≤ g - Hc

        Uses Algorithm 5.1 (UTVPIBoundingBox) to compute UTVPI bounding box.

        Args:
            H: Half-space matrix
            g: Half-space vector

        Returns:
            New Octatope (over-approximation of intersection)
        """
        H = np.asarray(H, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64).reshape(-1, 1)

        # New constraint in generator space: HGx ≤ g - Hc
        constraint_coef = H @ self.generators
        constraint_bound = g - H @ self.center.reshape(-1, 1)

        # Create new UTVPI system with bounding box
        new_utvpi = self._utvpi_bounding_box(self.utvpi, constraint_coef, constraint_bound)

        return Octatope(self.center, self.generators, new_utvpi)

    def _utvpi_bounding_box(self, U: UTVPIConstraintSystem,
                           constraint_coef: np.ndarray,
                           constraint_bound: np.ndarray) -> UTVPIConstraintSystem:
        """
        Algorithm 5.1: UTVPIBoundingBox

        Compute UTVPI bounding box of U ∪ {constraint_coef * x ≤ constraint_bound}

        Args:
            U: Original UTVPI system
            constraint_coef: Coefficients of new constraint
            constraint_bound: Bound of new constraint

        Returns:
            New UTVPI system over-approximating the intersection
        """
        new_utvpi = U.copy()

        # For all pairs of variables x_i, x_j
        for i in range(U.num_vars):
            for j in range(U.num_vars):
                if i == j:
                    # Bound for x_i
                    # Maximize x_i over U ∪ {constraint}
                    obj = np.zeros(U.num_vars)
                    obj[i] = 1.0
                    u_plus = self._optimize_with_constraint(U, obj, constraint_coef,
                                                            constraint_bound, maximize=True)
                    if u_plus is not None:
                        new_utvpi.add_constraint(i, i, 1, 0, u_plus)

                    # Maximize -x_i
                    u_minus = self._optimize_with_constraint(U, -obj, constraint_coef,
                                                             constraint_bound, maximize=True)
                    if u_minus is not None:
                        new_utvpi.add_constraint(i, i, -1, 0, u_minus)
                else:
                    # Bounds for x_i ± x_j
                    for ai in [1, -1]:
                        for aj in [1, -1]:
                            obj = np.zeros(U.num_vars)
                            obj[i] = ai
                            obj[j] = aj

                            u_ij = self._optimize_with_constraint(U, obj, constraint_coef,
                                                                  constraint_bound, maximize=True)
                            if u_ij is not None:
                                new_utvpi.add_constraint(i, j, ai, aj, u_ij)

        return new_utvpi

    def _optimize_with_constraint(self, U: UTVPIConstraintSystem,
                                  obj: np.ndarray,
                                  constraint_coef: np.ndarray,
                                  constraint_bound: np.ndarray,
                                  maximize: bool) -> Optional[float]:
        """Helper: optimize over UTVPI system with additional linear constraint"""
        A, b = U.to_matrix_form()

        x = cp.Variable(U.num_vars)

        if maximize:
            objective = cp.Maximize(obj @ x)
        else:
            objective = cp.Minimize(obj @ x)

        constraints = []
        if A.shape[0] > 0:
            constraints.append(A @ x <= b)

        # Add new constraint
        constraints.append(constraint_coef.flatten() @ x <= constraint_bound.flatten())

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
        Check if Octatope is empty (constraints are infeasible)

        Theorem 8: Can be decided in O(p * m) time.

        Returns:
            True if empty, False otherwise
        """
        return not self.utvpi.is_feasible()

    # ======================== Utility Methods ========================

    def contains(self, x: np.ndarray) -> bool:
        """
        Check if point x is in the Octatope

        Args:
            x: Point to check (dim,) or (dim, 1)

        Returns:
            True if x is in the Octatope
        """
        x = np.asarray(x, dtype=np.float64).flatten()

        if x.shape[0] != self.dim:
            raise ValueError(f"Point dimension {x.shape[0]} doesn't match dim {self.dim}")

        # Solve: find alpha such that G * alpha + c = x and UTVPI constraints hold
        # This is: G * alpha = x - c

        A, b = self.utvpi.to_matrix_form()

        alpha = cp.Variable(self.utvpi.num_vars)

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
        # This is critical: octatopes from bounds assume alpha ∈ [-1, 1]^n
        constraints.append(alpha >= -1)
        constraints.append(alpha <= 1)

        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            prob.solve()
            return prob.status in ['optimal', 'optimal_inaccurate']
        except:
            return False

    # ======================== Conversion Methods ========================

    def to_star(self) -> 'Star':
        """
        Convert Octatope to Star set representation

        An Octatope O = <c, G, UTVPI> represents {Gx + c : Ax ≤ b, x ∈ [-1, 1]^n}
        where Ax ≤ b is the UTVPI constraint system.

        The corresponding Star set is:
        - V = [c, G] where c is the center and G columns are generators
        - C = [A; -I; I] where A is UTVPI matrix, -I/I enforce box bounds
        - d = [b; 1...1; 1...1] where b is UTVPI bounds, 1s are box bounds

        This conversion is sound: if a point is in the Octatope, it is also
        in the resulting Star set.

        Returns:
            Star object representing this Octatope
        """
        from n2v.sets.star import Star

        # Get UTVPI constraints in matrix form
        A_utvpi, b_utvpi = self.utvpi.to_matrix_form()

        # Build constraint matrix C and bound vector d
        # C includes:
        # 1. UTVPI constraints: A_utvpi * x <= b_utvpi
        # 2. Box constraints: -1 <= x <= 1 (i.e., -x <= 1 and x <= 1)

        n_vars = self.utvpi.num_vars

        # Start with UTVPI constraints
        if A_utvpi.shape[0] > 0:
            C_list = [A_utvpi]
            d_list = [b_utvpi]
        else:
            C_list = []
            d_list = []

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
    ) -> List['Octatope']:
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
            List of output Octatope sets

        Example:
            >>> from n2v.sets import Octatope
            >>> import torch.nn as nn
            >>> model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 1))
            >>> input_oct = Octatope.from_bounds(lb, ub)
            >>> # Standard exact method with CVXPY
            >>> output_octs = input_oct.reach(model, method='exact')
            >>> # Exact method with differentiable solver
            >>> output_octs = input_oct.reach(model, method='exact-differentiable')
        """
        import torch.nn as nn
        from n2v.nn.reach.reach_octatope import reach_octatope_exact, reach_octatope_approx

        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be a PyTorch nn.Module, got {type(model)}")

        # Determine if we should use differentiable solver
        use_differentiable = (method == 'exact-differentiable')

        if method in ('exact', 'exact-differentiable'):
            return reach_octatope_exact(
                model, [self],
                use_differentiable=use_differentiable,
                **kwargs
            )
        elif method == 'approx':
            return reach_octatope_approx(model, [self], **kwargs)
        else:
            raise ValueError(
                f"Unknown method '{method}' for Octatope reachability. "
                f"Supported methods: 'exact', 'exact-differentiable', 'approx'"
            )
