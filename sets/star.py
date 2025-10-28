"""
Star set representation.

Represents a star set: x = c + sum_{i=1}^n alpha_i * v_i
                       = V * [1, alpha_1, ..., alpha_n]^T
                       subject to C*alpha <= d

Translated from MATLAB NNV Star.m
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import cvxpy as cp


class Star:
    """
    Star set class.

    A Star set is defined by:
        x = V * [1, alpha]^T, where V = [c, v1, v2, ..., vn]
        subject to: C * alpha <= d

    Attributes:
        V: Basic matrix [c v1 v2 ... vn] (dim, nVar+1)
        C: Constraint matrix (nConstr, nVar)
        d: Constraint vector (nConstr, 1)
        dim: Dimension of the star set
        nVar: Number of predicate variables
        predicate_lb: Lower bounds of predicate variables
        predicate_ub: Upper bounds of predicate variables
        state_lb: Lower bounds of state variables
        state_ub: Upper bounds of state variables
        Z: Outer zonotope covering this star (optional)
    """

    def __init__(
        self,
        V: Optional[np.ndarray] = None,
        C: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        pred_lb: Optional[np.ndarray] = None,
        pred_ub: Optional[np.ndarray] = None,
        state_lb: Optional[np.ndarray] = None,
        state_ub: Optional[np.ndarray] = None,
        outer_zono: Optional['Zono'] = None,
    ):
        """
        Initialize a Star set.

        Args:
            V: Basic matrix [c v1 v2 ... vn]
            C: Constraint matrix
            d: Constraint vector
            pred_lb: Predicate variable lower bounds
            pred_ub: Predicate variable upper bounds
            state_lb: State variable lower bounds
            state_ub: State variable upper bounds
            outer_zono: Outer zonotope approximation
        """
        if V is None:
            # Empty constructor
            self.V = np.array([]).reshape(0, 0)
            self.C = np.array([]).reshape(0, 0)
            self.d = np.array([]).reshape(0, 1)
            self.dim = 0
            self.nVar = 0
            self.predicate_lb = None
            self.predicate_ub = None
            self.state_lb = None
            self.state_ub = None
            self.Z = None
            return

        # Convert to numpy arrays
        V = np.asarray(V, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64) if C is not None else np.array([]).reshape(0, 0)
        d = np.asarray(d, dtype=np.float64) if d is not None else np.array([]).reshape(0, 1)

        # Ensure d is column vector
        if d.ndim == 1:
            d = d.reshape(-1, 1)

        # Validate dimensions
        nV, mV = V.shape
        nC, mC = C.shape if C.size > 0 else (0, mV - 1)
        nd, md = d.shape if d.size > 0 else (0, 1)

        if mV != mC + 1:
            raise ValueError(
                f"Inconsistency between basic matrix (cols={mV}) "
                f"and constraint matrix (cols={mC}). Expected mV = mC + 1"
            )

        if C.size > 0 and nC != nd:
            raise ValueError(
                f"Inconsistency between constraint matrix (rows={nC}) "
                f"and constraint vector (rows={nd})"
            )

        if md != 1:
            raise ValueError("Constraint vector should have one column")

        # Set basic properties
        self.V = V
        self.C = C
        self.d = d
        self.dim = nV
        self.nVar = mC

        # Handle predicate bounds
        if pred_lb is not None:
            pred_lb = np.asarray(pred_lb, dtype=np.float64).reshape(-1, 1)
            if pred_lb.shape[0] != mC:
                raise ValueError(f"Predicate lb size {pred_lb.shape[0]} doesn't match nVar {mC}")

        if pred_ub is not None:
            pred_ub = np.asarray(pred_ub, dtype=np.float64).reshape(-1, 1)
            if pred_ub.shape[0] != mC:
                raise ValueError(f"Predicate ub size {pred_ub.shape[0]} doesn't match nVar {mC}")

        self.predicate_lb = pred_lb
        self.predicate_ub = pred_ub

        # Handle state bounds
        if state_lb is not None:
            state_lb = np.asarray(state_lb, dtype=np.float64).reshape(-1, 1)
            if state_lb.shape[0] != nV:
                raise ValueError(f"State lb size doesn't match dimension {nV}")

        if state_ub is not None:
            state_ub = np.asarray(state_ub, dtype=np.float64).reshape(-1, 1)
            if state_ub.shape[0] != nV:
                raise ValueError(f"State ub size doesn't match dimension {nV}")

        self.state_lb = state_lb
        self.state_ub = state_ub

        # Outer zonotope
        self.Z = outer_zono

    def __repr__(self) -> str:
        return f"Star(dim={self.dim}, nVar={self.nVar}, nConstraints={self.C.shape[0]})"

    @classmethod
    def from_bounds(cls, lb: np.ndarray, ub: np.ndarray) -> 'Star':
        """
        Create Star from lower and upper bounds.

        Args:
            lb: Lower bound vector
            ub: Upper bound vector

        Returns:
            Star object
        """
        from .box import Box

        box = Box(lb, ub)
        return box.to_star()

    # ======================== Affine Transformations ========================

    def affine_map(self, W: np.ndarray, b: Optional[np.ndarray] = None) -> 'Star':
        """
        Apply affine transformation: W*x + b.

        Args:
            W: Mapping matrix (m, n)
            b: Mapping vector (m,) or (m, 1), optional

        Returns:
            New Star object
        """
        W = np.asarray(W, dtype=np.float64)

        if W.shape[1] != self.dim:
            raise ValueError(f"Matrix W has {W.shape[1]} columns, expected {self.dim}")

        # Transform V: new_V = W * V
        new_V = W @ self.V

        # Add bias to center if provided
        if b is not None:
            b = np.asarray(b, dtype=np.float64).reshape(-1, 1)
            new_V[:, 0:1] = new_V[:, 0:1] + b

        # Constraints remain the same
        new_pred_lb = self.predicate_lb
        new_pred_ub = self.predicate_ub

        return Star(new_V, self.C, self.d, new_pred_lb, new_pred_ub)

    # ======================== Set Operations ========================

    def minkowski_sum(self, other: 'Star') -> 'Star':
        """
        Compute Minkowski sum with another Star.

        Args:
            other: Another Star object

        Returns:
            New Star representing the Minkowski sum
        """
        if not isinstance(other, Star):
            raise TypeError("Can only compute Minkowski sum with another Star")
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")

        # Combine basis vectors: new_V = [c1+c2, V1, V2]
        new_c = self.V[:, 0:1] + other.V[:, 0:1]
        new_V = np.hstack([new_c, self.V[:, 1:], other.V[:, 1:]])

        # Combine constraints in block-diagonal form
        from scipy.linalg import block_diag

        new_C = block_diag(self.C, other.C)
        new_d = np.vstack([self.d, other.d])

        # Combine predicate bounds
        new_pred_lb = None
        new_pred_ub = None
        if self.predicate_lb is not None and other.predicate_lb is not None:
            new_pred_lb = np.vstack([self.predicate_lb, other.predicate_lb])
        if self.predicate_ub is not None and other.predicate_ub is not None:
            new_pred_ub = np.vstack([self.predicate_ub, other.predicate_ub])

        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)

    def intersect_half_space(self, H: np.ndarray, g: np.ndarray) -> 'Star':
        """
        Intersect star with half-space: H*x <= g.

        Args:
            H: Half-space matrix
            g: Half-space vector

        Returns:
            New Star object
        """
        H = np.asarray(H, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64).reshape(-1, 1)

        # Transform constraint to predicate space: H*(V*[1;alpha]) <= g
        # H*V*[1;alpha] <= g
        # H*V[:, 0] + H*V[:, 1:]*alpha <= g
        # H*V[:, 1:]*alpha <= g - H*V[:, 0]

        H_alpha = H @ self.V[:, 1:]  # Coefficient matrix for alpha
        g_alpha = g - H @ self.V[:, 0:1]  # New constraint bound

        # Add new constraints
        new_C = np.vstack([self.C, H_alpha])
        new_d = np.vstack([self.d, g_alpha])

        return Star(self.V, new_C, new_d, self.predicate_lb, self.predicate_ub)

    def convex_hull(self, other: 'Star') -> 'Star':
        """
        Compute over-approximation of convex hull with another Star.

        Args:
            other: Another Star object

        Returns:
            New Star over-approximating the convex hull
        """
        if not isinstance(other, Star):
            raise TypeError("Can only compute convex hull with another Star")
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")

        # Convex hull approximation using scalar parameter
        # Similar to zonotope convex hull
        new_c = 0.5 * (self.V[:, 0:1] + other.V[:, 0:1])
        new_V = np.hstack(
            [new_c, self.V[:, 1:], other.V[:, 1:], 0.5 * (self.V[:, 0:1] - other.V[:, 0:1])]
        )

        # Combine constraints
        from scipy.linalg import block_diag

        new_C = block_diag(self.C, other.C)
        new_d = np.vstack([self.d, other.d])

        # Add constraint for convex combination parameter
        # This is an over-approximation
        n_new_var = new_V.shape[1] - 1
        C_extra = np.zeros((2, n_new_var))
        C_extra[0, -1] = 1  # lambda <= 1
        C_extra[1, -1] = -1  # lambda >= -1
        d_extra = np.ones((2, 1))

        new_C = np.vstack([new_C, C_extra])
        new_d = np.vstack([new_d, d_extra])

        return Star(new_V, new_C, new_d)

    # ======================== Bounds Computation ========================

    def get_box(self, lp_solver: str = 'default') -> 'Box':
        """
        Compute exact bounding box using LP.

        Args:
            lp_solver: LP solver to use ('default', 'ECOS', 'SCS', etc.)

        Returns:
            Box object
        """
        from .box import Box

        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        for i in range(self.dim):
            lb[i], ub[i] = self.get_range(i, lp_solver)

        return Box(lb, ub)

    def get_range(self, index: int, lp_solver: str = 'default') -> Tuple[float, float]:
        """
        Compute exact range at specific dimension using LP.

        Args:
            index: Dimension index (0-based)
            lp_solver: LP solver to use

        Returns:
            Tuple of (min, max) values
        """
        if index < 0 or index >= self.dim:
            raise ValueError(f"Invalid index {index}, dimension is {self.dim}")

        # Minimize and maximize V[index, :] * [1; alpha]
        # This is equivalent to: V[index, 0] + V[index, 1:] * alpha

        # Define LP: min/max f^T * alpha subject to C * alpha <= d
        f = self.V[index, 1:].reshape(-1, 1)

        xmin = self._solve_lp(f, minimize=True, lp_solver=lp_solver)
        xmax = self._solve_lp(f, minimize=False, lp_solver=lp_solver)

        if xmin is None or xmax is None:
            # Infeasible or unbounded
            return None, None

        # Add constant term
        xmin = xmin + self.V[index, 0]
        xmax = xmax + self.V[index, 0]

        return xmin, xmax

    def get_ranges(self, lp_solver: str = 'default', parallel: bool = None,
                   n_workers: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute exact ranges for all dimensions.

        Args:
            lp_solver: LP solver to use
            parallel: If True, use parallel LP solving. If None, use global config (default: None)
            n_workers: Number of parallel workers. If None, use global config (default: None)

        Returns:
            Tuple of (lb, ub) arrays

        Note:
            Parallel solving is beneficial for high-dimensional outputs (dim > 10).
            For small dimensions, sequential solving is faster due to overhead.

        Example:
            >>> # Use global configuration
            >>> lb, ub = star.get_ranges()

            >>> # Force parallel with 8 workers
            >>> lb, ub = star.get_ranges(parallel=True, n_workers=8)

            >>> # Force sequential
            >>> lb, ub = star.get_ranges(parallel=False)
        """
        # Import config here to avoid circular dependency
        try:
            from n2v.config import config as global_config
        except ImportError:
            # If config not available, use defaults
            use_parallel = parallel if parallel is not None else False
            n_workers = n_workers if n_workers is not None else 4
        else:
            # Determine if we should use parallel
            if parallel is None:
                use_parallel = global_config.should_use_parallel(self.dim)
            else:
                use_parallel = parallel and self.dim > 1

            # Determine number of workers
            if n_workers is None:
                n_workers = global_config.get_n_workers(self.dim)

        if use_parallel:
            return self._get_ranges_parallel(lp_solver, n_workers)

        # Sequential version
        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        for i in range(self.dim):
            lb[i], ub[i] = self.get_range(i, lp_solver)

        return lb, ub

    def _get_ranges_parallel(self, lp_solver: str = 'default',
                            n_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ranges for all dimensions in parallel using ThreadPoolExecutor.

        Args:
            lp_solver: LP solver to use
            n_workers: Number of parallel workers

        Returns:
            Tuple of (lb, ub) arrays
        """
        from concurrent.futures import ThreadPoolExecutor

        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        def compute_range(i):
            """Compute range for dimension i."""
            try:
                return i, self.get_range(i, lp_solver)
            except Exception as e:
                # If LP fails, return None to indicate failure
                return i, (None, None)

        # Use ThreadPoolExecutor for IO-bound LP solving
        # (LP solvers release GIL when calling external libraries)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(compute_range, i) for i in range(self.dim)]

            # Collect results
            for future in futures:
                i, (lb_i, ub_i) = future.result()
                if lb_i is not None and ub_i is not None:
                    lb[i] = lb_i
                    ub[i] = ub_i
                else:
                    # LP failed for this dimension, use estimate
                    lb[i], ub[i] = self.estimate_range(i)

        return lb, ub

    def estimate_range(self, index: int) -> Tuple[float, float]:
        """
        Fast over-approximate range estimation using predicate bounds.

        Args:
            index: Dimension index (0-based)

        Returns:
            Tuple of (min_estimate, max_estimate)
        """
        if self.predicate_lb is None or self.predicate_ub is None:
            # Fall back to LP
            return self.get_range(index)

        # Estimate using interval arithmetic on predicate bounds
        c = self.V[index, 0]
        generators = self.V[index, 1:]

        # For each generator, compute its contribution range
        lb_contrib = 0.0
        ub_contrib = 0.0

        for i, g in enumerate(generators):
            alpha_min = self.predicate_lb[i, 0]
            alpha_max = self.predicate_ub[i, 0]

            if g >= 0:
                lb_contrib += g * alpha_min
                ub_contrib += g * alpha_max
            else:
                lb_contrib += g * alpha_max
                ub_contrib += g * alpha_min

        return c + lb_contrib, c + ub_contrib

    def estimate_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast over-approximate ranges for all dimensions.

        Also stores results in state_lb and state_ub for convenience.

        Returns:
            Tuple of (lb, ub) arrays
        """
        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        for i in range(self.dim):
            lb[i], ub[i] = self.estimate_range(i)

        # Store in state attributes for later use
        self.state_lb = lb
        self.state_ub = ub

        return lb, ub

    def _solve_lp(
        self, f: np.ndarray, minimize: bool = True, lp_solver: str = 'default'
    ) -> Optional[float]:
        """
        Solve LP: min/max f^T * alpha subject to C * alpha <= d and bounds.

        Args:
            f: Objective coefficient vector
            minimize: If True, minimize; else maximize
            lp_solver: Solver to use

        Returns:
            Optimal objective value, or None if infeasible
        """
        if self.nVar == 0:
            return 0.0

        # Import locally to avoid circular dependency
        from n2v.utils.lpsolver import solve_lp

        # Prepare constraints for solve_lp
        A = self.C if self.C.size > 0 else None
        b = self.d if self.C.size > 0 else None
        lb = self.predicate_lb if self.predicate_lb is not None else None
        ub = self.predicate_ub if self.predicate_ub is not None else None

        # Call centralized LP solver
        x_opt, fval, status, info = solve_lp(
            f=f,
            A=A,
            b=b,
            lb=lb,
            ub=ub,
            solver=lp_solver,
            minimize=minimize
        )

        # Return objective value or None if infeasible
        if status in ['optimal', 'optimal_inaccurate']:
            return fval
        else:
            return None

    def is_empty_set(self, lp_solver: str = 'default') -> bool:
        """
        Check if Star is empty (constraints are infeasible).

        Returns:
            True if empty, False otherwise
        """
        # Import locally to avoid circular dependency
        from n2v.utils.lpsolver import check_feasibility

        # Use centralized feasibility checker
        A = self.C if self.C.size > 0 else None
        b = self.d if self.C.size > 0 else None
        lb = self.predicate_lb if self.predicate_lb is not None else None
        ub = self.predicate_ub if self.predicate_ub is not None else None

        return not check_feasibility(A=A, b=b, lb=lb, ub=ub, solver=lp_solver)

    def contains(self, x: np.ndarray, lp_solver: str = 'default') -> bool:
        """
        Check if point x is in the Star.

        Args:
            x: Point to check (dim,) or (dim, 1)

        Returns:
            True if x is in the Star
        """
        x = np.asarray(x).reshape(-1, 1)

        if x.shape[0] != self.dim:
            raise ValueError(f"Point dimension {x.shape[0]} doesn't match Star dim {self.dim}")

        # Solve: find alpha such that V * [1; alpha] = x and C * alpha <= d
        # This is: V[:, 0] + V[:, 1:] * alpha = x
        # So: V[:, 1:] * alpha = x - V[:, 0]

        # This is a feasibility problem
        alpha = cp.Variable(self.nVar)

        constraints = [self.V[:, 1:] @ alpha == (x - self.V[:, 0:1]).flatten()]

        if self.C.size > 0:
            constraints.append(self.C @ alpha <= self.d.flatten())
        if self.predicate_lb is not None:
            constraints.append(alpha >= self.predicate_lb.flatten())
        if self.predicate_ub is not None:
            constraints.append(alpha <= self.predicate_ub.flatten())

        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            if lp_solver == 'default':
                prob.solve()
            else:
                prob.solve(solver=lp_solver)

            return prob.status in ['optimal', 'optimal_inaccurate']
        except:
            return False

    # ======================== Conversion Methods ========================

    def to_image_star(self, height: int, width: int, num_channels: int) -> 'ImageStar':
        """
        Convert Star to ImageStar format.

        Args:
            height: Image height
            width: Image width
            num_channels: Number of channels

        Returns:
            ImageStar object
        """
        from .image_star import ImageStar

        if height * width * num_channels != self.dim:
            raise ValueError(
                f"Image dimensions {height}x{width}x{num_channels} = "
                f"{height * width * num_channels} don't match Star dim {self.dim}"
            )

        return ImageStar(self.V, self.C, self.d, self.predicate_lb, self.predicate_ub,
                         height, width, num_channels)

    # ======================== Utility Methods ========================

    def sample(self, N: int) -> np.ndarray:
        """
        Sample points from the Star (using rejection sampling).

        Args:
            N: Number of samples to attempt

        Returns:
            Array of sampled points (dim, k) where k <= N
        """
        # Get bounding box
        if self.state_lb is not None and self.state_ub is not None:
            lb = self.state_lb
            ub = self.state_ub
        else:
            lb, ub = self.estimate_ranges()

        # Sample from box and check constraints
        samples = []
        from .box import Box
        box = Box(lb, ub)

        candidates = box.sample(2 * N)  # Over-sample

        for i in range(candidates.shape[1]):
            if self.contains(candidates[:, i:i+1]):
                samples.append(candidates[:, i:i+1])
                if len(samples) >= N:
                    break

        if samples:
            return np.hstack(samples)
        else:
            return np.array([]).reshape(self.dim, 0)

    # ======================== Reachability Analysis ========================
    # Note: Reachability analysis should be performed through NeuralNetwork.reach()
    # instead of calling reach() on set objects directly. This maintains proper
    # separation of concerns where sets represent geometric objects and reachability
    # is a neural network operation.
    #
    # Example usage:
    #     from n2v.nn import NeuralNetwork
    #     from n2v.sets import Star
    #     import torch.nn as nn
    #
    #     model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 1))
    #     net = NeuralNetwork(model)
    #     input_star = Star.from_bounds(lb, ub)
    #     output_stars = net.reach(input_star, method='exact')
