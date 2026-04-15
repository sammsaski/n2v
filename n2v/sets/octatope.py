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
from typing import Tuple, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
import cvxpy as cp
from concurrent.futures import ThreadPoolExecutor

from n2v.utils.lp_solver_enum import LPSolver, resolve as _resolve_lp

# TYPE_CHECKING imports for type hints (avoid circular import at runtime)
if TYPE_CHECKING:
    from n2v.sets.hexatope import DifferenceConstraintSystem, Hexatope
    from n2v.sets.box import Box
    from n2v.sets.star import Star

# NOTE: Runtime imports of n2v.sets.* modules are kept inline in methods
# to avoid circular dependencies (octatope <-> hexatope <-> star <-> box)


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
        """Validate UTVPI coefficient constraints.

        Raises:
            ValueError: If coefficients are not in {-1, 0, 1}
                or both are zero.
        """
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
        """Initialize a UTVPI Constraint System.

        Creates an empty UTVPI system over the given number
        of variables. Constraints of the form
        a_i*x_i + a_j*x_j <= b (with a_i, a_j in {-1, 0, 1})
        can be added via add_constraint().

        See Section 4 of [Bak et al., FMSD 2024].

        Args:
            num_vars: Number of variables in the system.
        """
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
            # Handle case where i == j (absolute constraints like ±x_i ≤ b)
            if uc.i == uc.j:
                # Constraint is ai*x_i + aj*x_i = (ai+aj)*x_i ≤ b
                A[k, uc.i] = uc.ai + uc.aj
            else:
                # Different variables: ai*x_i + aj*x_j ≤ b
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
        """Return string representation of the Octatope."""
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

        # Kernel constraints remain unchanged - deep copy to avoid aliasing bugs
        return Octatope(new_center, new_generators, self.utvpi.copy())

    # ======================== Bounds Computation ========================

    def get_range(self, index: int, solver: str) -> Tuple[float, float]:
        """
        Compute exact range at specific dimension

        Theorem 7: Linear optimization over octatopes can be solved in
        strongly polynomial time via reduction to hexatope optimization
        (which uses min-cost flow).

        Args:
            index: Dimension index (0-based)
            solver: Solver to use: 'lp' or 'mcf'

        Returns:
            Tuple of (min, max) values
        """
        if index < 0 or index >= self.dim:
            raise ValueError(f"Invalid index {index}, dimension is {self.dim}")

        # Create objective vector with 1 at position index
        objective = np.zeros(self.dim)
        objective[index] = 1.0

        # Minimize and maximize
        xmin = self.optimize_linear(objective, maximize=False, solver=solver)
        xmax = self.optimize_linear(objective, maximize=True, solver=solver)

        if xmin is None or xmax is None:
            return None, None

        return xmin, xmax

    def get_ranges(self, solver: str, parallel: bool = False,
                   n_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute exact ranges for all dimensions

        Args:
            solver: Solver to use: 'lp' or 'mcf'
            parallel: If True, use parallel computation
            n_workers: Number of parallel workers

        Returns:
            Tuple of (lb, ub) arrays
        """
        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        if parallel and self.dim > 1:
            return self._get_ranges_parallel(solver=solver, n_workers=n_workers)

        # Sequential version with LP fallback
        for i in range(self.dim):
            lb_i, ub_i = self.get_range(i, solver=solver)
            if (lb_i is None or ub_i is None
                    or not (np.isfinite(lb_i) and np.isfinite(ub_i))
                    or lb_i > ub_i + 1e-10):
                # Try LP fallback if solver failed, returned non-finite, or lb > ub
                if solver != 'lp':
                    lb_i, ub_i = self.get_range(i, solver='lp')
                # Final fallback to estimation if both failed
                if (lb_i is None or ub_i is None
                        or not (np.isfinite(lb_i) and np.isfinite(ub_i))
                        or lb_i > ub_i + 1e-10):
                    lb_i, ub_i = self.estimate_range(i)
            lb[i] = lb_i
            ub[i] = ub_i

        return lb, ub

    def get_bounds(self, solver: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alias for get_ranges() for API consistency with other set types.

        Args:
            solver: Solver to use: 'lp' or 'mcf'

        Returns:
            Tuple of (lb, ub) arrays
        """
        return self.get_ranges(solver=solver)

    def _get_ranges_parallel(self, solver: str,
                            n_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ranges in parallel"""
        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        def compute_range(i):
            """Compute range for dimension i, returning (i, (lb, ub))."""
            try:
                return i, self.get_range(i, solver=solver)
            except Exception:
                return i, (None, None)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(compute_range, i) for i in range(self.dim)]

            for future in futures:
                i, (lb_i, ub_i) = future.result()
                if (lb_i is not None and ub_i is not None
                        and np.isfinite(lb_i) and np.isfinite(ub_i)):
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

    def get_box(self, solver: str) -> 'Box':
        """
        Compute exact bounding box

        Args:
            solver: Solver to use: 'lp' or 'mcf'

        Returns:
            Box object
        """
        from n2v.sets.box import Box

        lb, ub = self.get_ranges(solver=solver)
        return Box(lb, ub)

    def optimize_linear(self, objective: np.ndarray, maximize: bool = True,
                       solver: str = None) -> Optional[float]:
        """
        Optimize linear objective over octatope

        Theorem 7: Linear optimization over octatopes can be solved in
        strongly polynomial time via reduction to hexatope optimization.

        To optimize f(y) = f^T y over [[O]], we optimize f^T(Gx + c) over Ax ≤ b.
        This reduces to optimizing (f^T G)x + f^T c over the UTVPI system.

        Args:
            objective: Objective vector f ∈ ℝⁿ
            maximize: If True, maximize; else minimize
            solver: Solver to use: 'lp' or 'mcf'

        Returns:
            Optimal value, or None if infeasible
        """
        if solver not in ('lp', 'mcf'):
            raise ValueError(f"Unknown solver '{solver}'. Must be 'lp' or 'mcf'.")

        objective = np.asarray(objective, dtype=np.float64).flatten()

        if objective.shape[0] != self.dim:
            raise ValueError(f"Objective size {objective.shape[0]} != dim {self.dim}")

        # Compose objective with affine mapping
        # f^T(Gx + c) = (f^T G)x + f^T c
        composed_obj = objective @ self.generators  # w = f^T G
        constant_term = objective @ self.center  # f^T c

        # Now optimize w^T x over UTVPI system
        if solver == 'mcf':
            result = self._optimize_utvpi_mcf(composed_obj, constant_term, maximize)
        else:  # solver == 'lp': cross into enum-land (hex_oct alias maps 'lp' -> DEFAULT)
            _lp_enum = _resolve_lp(solver, hex_oct_alias=True)  # noqa: F841 - reserved for future use
            result = self._optimize_utvpi_lp(composed_obj, constant_term, maximize)

        return result

    def _optimize_utvpi_mcf(self, w: np.ndarray, constant: float,
                           maximize: bool) -> Optional[float]:
        """
        Optimize linear objective w^T x + constant over UTVPI system via MCF

        Implements Theorem 7 from Bak et al. (FMSD 2024):
        UTVPI optimization reduces to DCS optimization via variable splitting.

        For each UTVPI variable x_i, we create split variables x+_i and x-_i where:
            x_i = (1/2)(x+_i - x-_i)

        The objective w^T x becomes:
            w^T x = w^T (1/2)(x+ - x-) = (1/2)[w, -w]^T [x+; x-]

        So the DCS objective coefficients are [0.5*w, -0.5*w].

        Args:
            w: Objective coefficients in UTVPI/generator space
            constant: Constant term
            maximize: If True, maximize; else minimize

        Returns:
            Optimal value, or None if infeasible
        """
        from n2v.sets.hexatope import Hexatope, DifferenceConstraintSystem

        # Step 1: Convert UTVPI to DCS (Theorem 7)
        # This creates split variables x+_i, x-_i where x_i = 0.5(x+_i - x-_i)
        dcs = self.utvpi.to_dcs()

        # Step 2: Expand objective for split variables
        # w^T x = w^T (1/2)(x+ - x-) = (1/2)[w, -w]^T [x+; x-]
        p = self.utvpi.num_vars
        w_expanded = np.empty(2 * p)
        w_expanded[0::2] = 0.5 * w      # x+_i coefficients (even indices: 0, 2, 4, ...)
        w_expanded[1::2] = -0.5 * w     # x-_i coefficients (odd indices: 1, 3, 5, ...)

        # Step 3: Create temporary hexatope over the DCS (no anchor)
        # The gauge is fixed in the dual by demand balance and the super-node.
        # For MCF optimization, we just need a hexatope that holds the DCS.
        # Use identity generators (the affine map doesn't matter for optimization).
        temp_hex = Hexatope(center=np.zeros(dcs.num_vars),
                           generators=np.eye(dcs.num_vars),
                           dcs=dcs)

        # Step 4: Optimize using hexatope's MCF solver (now with corrected sign handling)
        return temp_hex._optimize_dcs_mcf(w_expanded, constant, maximize)

    def _optimize_utvpi_lp(self, w: np.ndarray, constant: float,
                          maximize: bool) -> Optional[float]:
        """
        Optimize linear objective w^T x + constant over UTVPI system using LP

        Fallback method using standard LP solver.
        For octatopes created from bounds, we assume x ∈ [-1, 1]^n in generator space.

        Args:
            w: Objective coefficients
            constant: Constant term
            maximize: If True, maximize; else minimize
        """
        A, b = self.utvpi.to_matrix_form()

        # Standard CVXPY solver
        x = cp.Variable(self.utvpi.num_vars)

        if maximize:
            objective = cp.Maximize(w @ x + constant)
        else:
            objective = cp.Minimize(w @ x + constant)

        constraints = []

        # Add UTVPI constraints (includes absolute bounds ±x_i ≤ 1)
        if A.shape[0] > 0:
            constraints.append(A @ x <= b)

        # Note: For octatopes from from_bounds(), the UTVPI system already includes
        # ±x_i ≤ 1 constraints, so no additional box constraints are needed

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

    def intersect_half_space(self, H: np.ndarray, g: np.ndarray, solver=None) -> 'Octatope':
        """
        Intersect octatope with half-space: H*x <= g

        Section 5.2: Intersection with half-spaces.

        For octatope O = <c, G, A, b> and halfspace {y | Hy ≤ g},
        the result is O' = <c, G, A', b'> where A'x ≤ b' comprises:
        - Original constraints: Ax ≤ b
        - New constraints: HGx ≤ g - Hc

        Uses Algorithm 5.1 (UTVPIBoundingBox) to compute UTVPI bounding box.

        Following ChatGPT's recommendation: process constraints incrementally row-by-row
        for empirically tighter and simpler results.

        Args:
            H: Half-space matrix (m × n)
            g: Half-space vector (m × 1)
            solver: Optional solver method (currently unused, reserved for future use).

        Returns:
            New Octatope (over-approximation of intersection)
        """
        H = np.asarray(H, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64).reshape(-1, 1)

        # Ensure H is 2D
        if len(H.shape) == 1:
            H = H.reshape(1, -1)

        # New constraint in generator space: HGx ≤ g - Hc
        constraint_coef = H @ self.generators  # (m × p)
        constraint_bound = g - H @ self.center.reshape(-1, 1)  # (m × 1)

        # Process constraints incrementally (row-by-row) for tighter results
        current_utvpi = self.utvpi
        for k in range(constraint_coef.shape[0]):
            # Extract single row
            row_coef = constraint_coef[k:k+1, :]  # Keep 2D: (1 × p)
            row_bound = constraint_bound[k:k+1, :]  # Keep 2D: (1 × 1)

            # Compute bounding box with this constraint
            current_utvpi = self._utvpi_bounding_box(current_utvpi, row_coef, row_bound, solver=solver)

        return Octatope(self.center, self.generators, current_utvpi)

    def _utvpi_bounding_box(self, U: UTVPIConstraintSystem,
                           constraint_coef: np.ndarray,
                           constraint_bound: np.ndarray,
                           solver=None) -> UTVPIConstraintSystem:
        """
        Algorithm 5.1: UTVPIBoundingBox

        Compute UTVPI bounding box of U ∪ {constraint_coef * x ≤ constraint_bound}

        Args:
            U: Original UTVPI system
            constraint_coef: Coefficients of new constraint (should be single row)
            constraint_bound: Bound of new constraint (should be single value)
            solver: Optional solver method (currently unused, reserved for future use).

        Returns:
            New UTVPI system over-approximating the intersection
        """
        # Ensure constraint_coef is a 1D array for single-row case
        constraint_coef = np.atleast_2d(constraint_coef)
        constraint_bound = np.atleast_1d(constraint_bound.flatten())

        # Fast-path: If constraint is already UTVPI-expressible, add it directly
        # Also handle normalized constraints (positive scalar multiples)
        if constraint_coef.shape[0] == 1:
            row = constraint_coef[0, :]
            nonzero_indices = np.nonzero(row)[0]

            # Check if UTVPI-expressible (at most 2 nonzeros)
            if len(nonzero_indices) <= 2 and len(nonzero_indices) >= 1:
                vals = row[nonzero_indices]

                # Check if all coefficients have the same absolute magnitude (normalized UTVPI)
                abs_vals = np.abs(vals)
                if np.allclose(abs_vals, abs_vals[0]):
                    # This is a normalized UTVPI constraint: k*(±x_i ± x_j) <= b
                    # Normalize to: ±x_i ± x_j <= b/k
                    scale = abs_vals[0]
                    normalized_bound = constraint_bound[0] / scale

                    new_utvpi = U.copy()

                    if len(nonzero_indices) == 1:
                        # Absolute constraint: ±x_i ≤ b/k
                        i = nonzero_indices[0]
                        ai = int(np.sign(vals[0]))
                        new_utvpi.add_constraint(i, i, ai, 0, normalized_bound)
                    else:  # len == 2
                        # Two-variable: ai*x_i + aj*x_j ≤ b/k
                        i, j = nonzero_indices[0], nonzero_indices[1]
                        ai = int(np.sign(vals[0]))
                        aj = int(np.sign(vals[1]))
                        new_utvpi.add_constraint(i, j, ai, aj, normalized_bound)

                    return new_utvpi

        # Fall back to full bounding box algorithm for non-UTVPI constraints
        # This is expensive: O(n²) optimizations
        new_utvpi = U.copy()

        # Single-variable bounds: +x_i and -x_i for each variable
        for i in range(U.num_vars):
            for ai in [1, -1]:
                obj = np.zeros(U.num_vars)
                obj[i] = ai
                u = self._optimize_with_constraint(U, obj, constraint_coef,
                                                   constraint_bound, maximize=True,
                                                   use_mcf=True)
                if u is not None:
                    new_utvpi.add_constraint(i, i, ai, 0, u)

        # Two-variable bounds: ai*x_i + aj*x_j for all i != j pairs
        for i in range(U.num_vars):
            for j in range(U.num_vars):
                if i == j:
                    continue
                for ai in [1, -1]:
                    for aj in [1, -1]:
                        obj = np.zeros(U.num_vars)
                        obj[i] = ai
                        obj[j] = aj
                        u_ij = self._optimize_with_constraint(U, obj, constraint_coef,
                                                              constraint_bound, maximize=True,
                                                              use_mcf=True)
                        if u_ij is not None:
                            new_utvpi.add_constraint(i, j, ai, aj, u_ij)

        return new_utvpi

    def _optimize_with_constraint(self, U: UTVPIConstraintSystem,
                                  obj: np.ndarray,
                                  constraint_coef: np.ndarray,
                                  constraint_bound: np.ndarray,
                                  maximize: bool,
                                  use_mcf: bool = True) -> Optional[float]:
        """
        Helper: optimize over UTVPI system with additional linear constraints

        Solves: max/min obj^T x subject to:
                - Ux ≤ u (original UTVPI constraints)
                - constraint_coef @ x ≤ constraint_bound (new constraints, may be multiple rows)

        Args:
            U: UTVPI system
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

        # MCF fast-path: Check if ALL added constraints are UTVPI-expressible
        # A constraint is UTVPI-expressible if it has at most 2 nonzeros with coefficients in {-1, 0, +1}
        all_utvpi_expressible = True
        utvpi_constraints = []  # List of (i, j, ai, aj, b) for each UTVPI constraint

        if use_mcf:
            for k in range(constraint_coef.shape[0]):
                row = constraint_coef[k, :]
                nonzero_indices = np.nonzero(row)[0]

                # Check: at most 2 nonzeros with values in {-1, +1}
                if len(nonzero_indices) <= 2 and len(nonzero_indices) >= 1:
                    vals = row[nonzero_indices]
                    # Check all values are ±1
                    if all(np.isclose(abs(v), 1.0) for v in vals):
                        if len(nonzero_indices) == 1:
                            # Absolute constraint: ±x_i ≤ b
                            i = nonzero_indices[0]
                            ai = int(np.sign(vals[0]))
                            utvpi_constraints.append((i, i, ai, 0, constraint_bound[k]))
                        else:  # len == 2
                            # Two-variable: ai*x_i + aj*x_j ≤ b
                            i, j = nonzero_indices[0], nonzero_indices[1]
                            ai = int(np.sign(vals[0]))
                            aj = int(np.sign(vals[1]))
                            utvpi_constraints.append((i, j, ai, aj, constraint_bound[k]))
                    else:
                        all_utvpi_expressible = False
                        break
                else:
                    all_utvpi_expressible = False
                    break

        # If all constraints are UTVPI-expressible, use MCF fast-path via DCS conversion
        if use_mcf and all_utvpi_expressible and len(utvpi_constraints) > 0:
            # Create augmented UTVPI system with additional constraints
            U_aug = U.copy()
            for i, j, ai, aj, b in utvpi_constraints:
                U_aug.add_constraint(i, j, ai, aj, b)

            # Convert to DCS and use MCF
            try:
                dcs = U_aug.to_dcs()

                # Expand objective for x+ and x- variables (with 0.5 scaling)
                w_expanded = np.zeros(2 * U_aug.num_vars)
                for i in range(U_aug.num_vars):
                    w_expanded[2*i] = 0.5 * obj[i]       # x+_i coefficient
                    w_expanded[2*i + 1] = -0.5 * obj[i]  # x-_i coefficient

                # Create temporary hexatope for MCF optimization
                from n2v.sets.hexatope import Hexatope
                temp_center = np.zeros(dcs.num_vars)
                temp_generators = np.eye(dcs.num_vars)
                temp_hex = Hexatope(temp_center, temp_generators, dcs)

                # Use MCF to optimize
                result = temp_hex._optimize_dcs_mcf(w_expanded, 0.0, maximize)
                if result is not None:
                    return result
            except Exception:
                # Fall back to LP if MCF fails
                pass

        # Fall back to LP for non-UTVPI constraints or if MCF failed
        A, b = U.to_matrix_form()

        x = cp.Variable(U.num_vars)

        if maximize:
            objective = cp.Maximize(obj @ x)
        else:
            objective = cp.Minimize(obj @ x)

        constraints = []
        if A.shape[0] > 0:
            constraints.append(A @ x <= b)

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
        Check if Octatope is empty (constraints are infeasible)

        Theorem 8: Can be decided in O(p * m) time.

        Returns:
            True if empty, False otherwise
        """
        return not self.utvpi.is_feasible()

    # ======================== Utility Methods ========================

    def contains(self, x: np.ndarray, tolerance: float = 1e-7) -> bool:
        """
        Check if point x is in the Octatope

        Uses two-phase approach per ChatGPT V3 feedback:
        1. Fast-path: Least-squares solve to propose alpha
        2. Explicit verification: Check residuals and constraints
        3. Fallback: CVXPY feasibility LP if fast-path fails

        This prevents false positives from solver inaccuracies.

        Args:
            x: Point to check (dim,) or (dim, 1)
            tolerance: Numerical tolerance for feasibility checks

        Returns:
            True if x is in the Octatope
        """
        x = np.asarray(x, dtype=np.float64).flatten()

        if x.shape[0] != self.dim:
            raise ValueError(f"Point dimension {x.shape[0]} doesn't match dim {self.dim}")

        # Target: G * alpha = x - c
        target = x - self.center
        A, b = self.utvpi.to_matrix_form()

        # Phase 1: Fast-path least-squares solve
        try:
            # Use least-squares to propose alpha
            alpha_proposed, residuals, rank, s = np.linalg.lstsq(self.generators, target, rcond=None)

            # Verify feasibility explicitly
            # 1. Check residual: ||G*alpha - target||_inf <= tol
            residual = self.generators @ alpha_proposed - target
            if np.linalg.norm(residual, ord=np.inf) > tolerance:
                # Residual too large, try LP fallback
                pass
            else:
                # 2. Check UTVPI constraints: A*alpha <= b + tol
                if A.shape[0] > 0:
                    constraint_violations = A @ alpha_proposed - b
                    if np.max(constraint_violations) > tolerance:
                        pass  # Constraints violated, try LP fallback
                    else:
                        # All checks passed - fast-path success
                        return True
                else:
                    # No UTVPI constraints, residual check sufficient
                    return True

        except np.linalg.LinAlgError:
            # Least-squares failed, fall through to LP
            pass

        # Phase 2: Fallback to CVXPY feasibility LP with OSQP solver
        # Use OSQP (not SCS) for better accuracy in feasibility contexts
        alpha = cp.Variable(self.utvpi.num_vars)

        diff = self.generators @ alpha - target

        # Use soft constraints: ||G*alpha - target||_inf <= tolerance
        constraints = [
            diff <= tolerance,
            diff >= -tolerance
        ]

        # Add UTVPI constraints (includes bounds ±alpha_i ≤ 1 from from_bounds())
        if A.shape[0] > 0:
            constraints.append(A @ alpha <= b)

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

            # Recheck UTVPI constraints
            if A.shape[0] > 0:
                constraint_violations = A @ alpha_val - b
                if np.max(constraint_violations) > tolerance:
                    return False

            return True

        except Exception:
            return False

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample points from the Octatope using rejection sampling.

        Samples alpha vectors uniformly from [-1, 1]^nVar and rejects those
        that violate UTVPI constraints. Maps accepted alphas to state space.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, dim) with points in state space
        """
        A, b_vec = self.utvpi.to_matrix_form()
        samples = []
        max_attempts = n_samples * 200

        for _ in range(max_attempts):
            if len(samples) >= n_samples:
                break
            alpha = np.random.uniform(-1, 1, self.nVar)
            # Check UTVPI constraints
            if A.shape[0] == 0 or np.all(A @ alpha <= b_vec + 1e-10):
                point = self.generators @ alpha + self.center.flatten()
                samples.append(point)

        while len(samples) < n_samples:
            samples.append(self.center.flatten())

        return np.array(samples)

    # ======================== Conversion Methods ========================

    def to_star(self) -> 'Star':
        """
        Convert Octatope to Star set representation

        An Octatope O = <c, G, UTVPI> represents {Gx + c : Ax ≤ b}
        where Ax ≤ b is the UTVPI constraint system (includes absolute bounds).

        The corresponding Star set is:
        - V = [c, G] where c is the center and G columns are generators
        - C = A where A is the UTVPI matrix
        - d = b where b is the UTVPI bounds

        Note: For octatopes from from_bounds(), the UTVPI system already includes
        ±x_i ≤ 1 constraints, so no additional box constraints are needed.

        This conversion is sound: if a point is in the Octatope, it is also
        in the resulting Star set.

        Returns:
            Star object representing this Octatope
        """
        from n2v.sets.star import Star

        # Get UTVPI constraints in matrix form (includes absolute bounds)
        A_utvpi, b_utvpi = self.utvpi.to_matrix_form()

        n_vars = self.utvpi.num_vars

        # Build constraint matrix C and bound vector d from UTVPI only
        if A_utvpi.shape[0] > 0:
            C = A_utvpi
            d = b_utvpi.reshape(-1, 1)
        else:
            # Empty constraints (shouldn't happen, but handle gracefully)
            C = np.zeros((0, n_vars))
            d = np.zeros((0, 1))

        # Build basis matrix V = [c, G]
        # c is the center (dim,), G is the generator matrix (dim, n_vars)
        V = np.hstack([self.center.reshape(-1, 1), self.generators])

        # Predicate bounds: Octatope generator variables are bounded in [-1, 1]
        # The UTVPI constraints enforce this (±x_i ≤ 1 added by from_bounds())
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
    #     from n2v.sets import Octatope
    #     import torch.nn as nn
    #
    #     model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 1))
    #     net = NeuralNetwork(model)
    #     input_oct = Octatope.from_bounds(lb, ub)
    #     # Standard exact method with CVXPY
    #     output_octs = net.reach(input_oct, method='exact')
