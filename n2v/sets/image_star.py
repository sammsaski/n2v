"""
ImageStar set representation for image inputs.

Represents a star set in image format with native 4D storage (H, W, C, nVar+1).
This design aligns with the ImageStar paper (arXiv:2004.05511) and MATLAB NNV implementation.

Key design decisions:
- V is stored as 4D array (H, W, C, nVar+1) for efficient spatial operations
- to_star() converts to 2D Star using HWC flattening order
- Flatten layer uses CHW ordering to match PyTorch's nn.Flatten()
- Set operations (convex_hull, minkowski_sum, etc.) delegate to Star

Translated from MATLAB NNV ImageStar.m
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

# TYPE_CHECKING imports to avoid circular imports at runtime
if TYPE_CHECKING:
    from n2v.sets.star import Star

# Import LP solver utilities
from n2v.utils.lpsolver import solve_lp, check_feasibility, solve_lp_batch

from n2v.config import config as global_config


class ImageStar:
    """
    ImageStar class with native 4D V representation.

    A 4D Star set designed for image data, where V is stored as (H, W, C, nVar+1).
    This enables efficient spatial operations (convolution, pooling) without reshaping.

    Representation:
        x(h, w, c) = V[h, w, c, 0] + sum_{i=1}^{nVar} alpha_i * V[h, w, c, i]
        subject to: C * alpha <= d
                    predicate_lb <= alpha <= predicate_ub

    Attributes:
        V: Basis tensor (H, W, C, nVar+1) where V[:,:,:,0] is center, V[:,:,:,1:] are generators
        C: Constraint matrix (nConstr, nVar)
        d: Constraint vector (nConstr, 1)
        height: Image height (H)
        width: Image width (W)
        num_channels: Number of channels (C)
        dim: Total dimension (H * W * C)
        nVar: Number of predicate variables
        predicate_lb: Lower bounds of predicate variables (nVar, 1)
        predicate_ub: Upper bounds of predicate variables (nVar, 1)
        state_lb: Lower bounds of state variables (dim, 1) - cached
        state_ub: Upper bounds of state variables (dim, 1) - cached
    """

    def __init__(
        self,
        V: np.ndarray,
        C: np.ndarray,
        d: np.ndarray,
        pred_lb: Optional[np.ndarray] = None,
        pred_ub: Optional[np.ndarray] = None,
        height: int = 0,
        width: int = 0,
        num_channels: int = 0,
    ) -> None:
        """
        Initialize an ImageStar.

        Args:
            V: Basis matrix/tensor. Can be:
               - 2D (H*W*C, nVar+1): Will be reshaped to 4D
               - 4D (H, W, C, nVar+1): Used directly
            C: Constraint matrix (nConstr, nVar)
            d: Constraint vector (nConstr, 1) or (nConstr,)
            pred_lb: Predicate lower bounds (nVar, 1) or (nVar,)
            pred_ub: Predicate upper bounds (nVar, 1) or (nVar,)
            height: Image height (required if V is 2D)
            width: Image width (required if V is 2D)
            num_channels: Number of channels (required if V is 2D)
        """
        # Convert inputs to numpy arrays
        V = np.asarray(V, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64) if C is not None else np.array([]).reshape(0, 0)
        d = np.asarray(d, dtype=np.float64) if d is not None else np.array([]).reshape(0, 1)

        # Ensure d is column vector
        if d.ndim == 1:
            d = d.reshape(-1, 1)

        # Handle V dimensionality
        if V.ndim == 2:
            # 2D input - need image dimensions to reshape
            if height <= 0 or width <= 0 or num_channels <= 0:
                raise ValueError(
                    "When V is 2D, height, width, and num_channels must be provided"
                )
            expected_dim = height * width * num_channels
            if V.shape[0] != expected_dim:
                raise ValueError(
                    f"V has {V.shape[0]} rows but image dimensions "
                    f"{height}x{width}x{num_channels} = {expected_dim}"
                )
            # Reshape to 4D (H, W, C, nVar+1)
            self.V = V.reshape(height, width, num_channels, -1)
            self.height = height
            self.width = width
            self.num_channels = num_channels

        elif V.ndim == 4:
            # Already 4D
            self.V = V
            self.height = V.shape[0]
            self.width = V.shape[1]
            self.num_channels = V.shape[2]
            # If dimensions provided, validate them
            if height > 0 and height != self.height:
                raise ValueError(f"height={height} doesn't match V shape {V.shape}")
            if width > 0 and width != self.width:
                raise ValueError(f"width={width} doesn't match V shape {V.shape}")
            if num_channels > 0 and num_channels != self.num_channels:
                raise ValueError(f"num_channels={num_channels} doesn't match V shape {V.shape}")
        else:
            raise ValueError(f"V must be 2D or 4D, got {V.ndim}D")

        # Compute derived properties
        self.dim = self.height * self.width * self.num_channels
        self.nVar = self.V.shape[3] - 1  # Last dimension is nVar+1

        # Validate constraint dimensions
        nC = C.shape[0] if C.size > 0 else 0
        mC = C.shape[1] if C.size > 0 else self.nVar
        nd = d.shape[0] if d.size > 0 else 0

        if C.size > 0 and mC != self.nVar:
            raise ValueError(
                f"C has {mC} columns but V implies {self.nVar} predicate variables"
            )
        if C.size > 0 and nC != nd:
            raise ValueError(
                f"C has {nC} rows but d has {nd} rows"
            )

        # Store constraints
        self.C = C
        self.d = d

        # Handle predicate bounds
        if pred_lb is not None:
            pred_lb = np.asarray(pred_lb, dtype=np.float64).reshape(-1, 1)
            if pred_lb.shape[0] != self.nVar:
                raise ValueError(f"pred_lb size {pred_lb.shape[0]} != nVar {self.nVar}")
        if pred_ub is not None:
            pred_ub = np.asarray(pred_ub, dtype=np.float64).reshape(-1, 1)
            if pred_ub.shape[0] != self.nVar:
                raise ValueError(f"pred_ub size {pred_ub.shape[0]} != nVar {self.nVar}")

        self.predicate_lb = pred_lb
        self.predicate_ub = pred_ub

        # State bounds (computed lazily)
        self.state_lb = None
        self.state_ub = None

    def __repr__(self) -> str:
        """Return string representation of the ImageStar."""
        return (
            f"ImageStar(height={self.height}, width={self.width}, "
            f"channels={self.num_channels}, nVar={self.nVar})"
        )

    # ======================== Conversion Methods ========================

    def to_star(self) -> 'Star':
        """
        Convert ImageStar to regular Star (flattens V in HWC order).

        The flattening uses row-major (C-order) through H, W, C dimensions.
        This is compatible with Star.to_image_star() for round-trip conversion.

        Returns:
            Star object with 2D V matrix (H*W*C, nVar+1)
        """
        from n2v.sets.star import Star

        # Flatten V from (H, W, C, nVar+1) to (H*W*C, nVar+1)
        V_2d = self.V.reshape(-1, self.V.shape[3])

        return Star(
            V_2d,
            self.C,
            self.d,
            self.predicate_lb,
            self.predicate_ub,
            state_lb=self.state_lb,
            state_ub=self.state_ub
        )

    def flatten_to_star(self) -> 'Star':
        """
        Convert ImageStar to Star with CHW ordering (PyTorch Flatten convention).

        This method reorders V from HWC to CHW order to match PyTorch's nn.Flatten().
        Use this when transitioning from convolutional to fully-connected layers.

        Note: This is different from to_star() which uses HWC ordering.

        Returns:
            Star object with CHW-ordered flattening
        """
        from n2v.sets.star import Star

        # Transpose from (H, W, C, nVar+1) to (C, H, W, nVar+1)
        V_chw = np.transpose(self.V, (2, 0, 1, 3))

        # Flatten to (C*H*W, nVar+1)
        V_flat = V_chw.reshape(-1, self.V.shape[3])

        return Star(
            V_flat,
            self.C,
            self.d,
            self.predicate_lb,
            self.predicate_ub,
            state_lb=self.state_lb,
            state_ub=self.state_ub
        )

    def get_image_shape(self) -> tuple:
        """Get image shape (height, width, channels)."""
        return (self.height, self.width, self.num_channels)

    # ======================== Factory Methods ========================

    @classmethod
    def from_bounds(
        cls,
        lb: np.ndarray,
        ub: np.ndarray,
        height: int,
        width: int,
        num_channels: int
    ) -> 'ImageStar':
        """
        Create ImageStar from image bounds.

        Each pixel becomes an independent predicate variable.

        Args:
            lb: Lower bound image (H, W, C) or flattened (H*W*C,)
            ub: Upper bound image (H, W, C) or flattened (H*W*C,)
            height: Image height
            width: Image width
            num_channels: Number of channels

        Returns:
            ImageStar object
        """
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)

        # Flatten if needed
        lb_flat = lb.reshape(-1)
        ub_flat = ub.reshape(-1)

        dim = height * width * num_channels
        if lb_flat.shape[0] != dim:
            raise ValueError(
                f"lb has {lb_flat.shape[0]} elements but expected {dim} "
                f"for image {height}x{width}x{num_channels}"
            )
        if ub_flat.shape[0] != dim:
            raise ValueError(f"ub has {ub_flat.shape[0]} elements but expected {dim}")

        # Center and generators (only for dimensions with nonzero range)
        center = 0.5 * (lb_flat + ub_flat)
        ranges = ub_flat - lb_flat
        nonzero_mask = ranges != 0
        nVar = int(nonzero_mask.sum())

        if nVar == 0:
            # Point input — no predicates
            V_2d = center.reshape(-1, 1)
            C = np.zeros((0, 0))
            d = np.zeros((0, 1))
            pred_lb = np.zeros((0, 1))
            pred_ub = np.zeros((0, 1))
        else:
            # Build generator columns only for perturbed dimensions
            generators = np.zeros((dim, nVar))
            nonzero_idx = np.where(nonzero_mask)[0]
            for j, idx in enumerate(nonzero_idx):
                generators[idx, j] = 0.5 * ranges[idx]

            V_2d = np.hstack([center.reshape(-1, 1), generators])

            # Constraints: each predicate variable in [-1, 1]
            C = np.vstack([np.eye(nVar), -np.eye(nVar)])
            d = np.ones((2 * nVar, 1))
            pred_lb = -np.ones((nVar, 1))
            pred_ub = np.ones((nVar, 1))

        return cls(V_2d, C, d, pred_lb, pred_ub, height, width, num_channels)

    # ======================== Range Estimation ========================

    def estimate_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast over-approximate range estimation using interval arithmetic.

        Uses vectorized 4D operations with einsum for efficiency.
        This is an over-approximation; for exact bounds use get_ranges().

        Returns:
            Tuple of (lb, ub) arrays, each shape (dim, 1)
        """
        if self.predicate_lb is None or self.predicate_ub is None:
            # Fall back to LP-based computation
            return self.get_ranges()

        # Extract center and generators from 4D V
        center = self.V[:, :, :, 0]        # (H, W, C)
        generators = self.V[:, :, :, 1:]   # (H, W, C, nVar)

        # Separate positive and negative parts
        pos_gens = np.maximum(generators, 0)
        neg_gens = np.minimum(generators, 0)

        pred_lb_flat = self.predicate_lb.flatten()
        pred_ub_flat = self.predicate_ub.flatten()

        # Vectorized computation using einsum
        # lb = center + pos_gens @ pred_lb + neg_gens @ pred_ub
        # ub = center + pos_gens @ pred_ub + neg_gens @ pred_lb
        lb_3d = center + np.einsum('hwcn,n->hwc', pos_gens, pred_lb_flat) + \
                         np.einsum('hwcn,n->hwc', neg_gens, pred_ub_flat)
        ub_3d = center + np.einsum('hwcn,n->hwc', pos_gens, pred_ub_flat) + \
                         np.einsum('hwcn,n->hwc', neg_gens, pred_lb_flat)

        # Flatten to (dim, 1)
        self.state_lb = lb_3d.reshape(-1, 1)
        self.state_ub = ub_3d.reshape(-1, 1)

        return self.state_lb, self.state_ub

    def estimate_range(self, index: int) -> Tuple[float, float]:
        """
        Fast range estimation for a single dimension using interval arithmetic.

        Args:
            index: Flattened dimension index (0 to dim-1)

        Returns:
            Tuple of (min_estimate, max_estimate)
        """
        if self.predicate_lb is None or self.predicate_ub is None:
            return self.get_range_flat(index)

        # Convert flat index to (h, w, c) coordinates
        h = index // (self.width * self.num_channels)
        remainder = index % (self.width * self.num_channels)
        w = remainder // self.num_channels
        c = remainder % self.num_channels

        # Get center and generators for this pixel
        center_val = self.V[h, w, c, 0]
        generators = self.V[h, w, c, 1:]  # (nVar,)

        pred_lb_flat = self.predicate_lb.flatten()
        pred_ub_flat = self.predicate_ub.flatten()

        # Interval arithmetic
        lb_contrib = 0.0
        ub_contrib = 0.0
        for i, g in enumerate(generators):
            if g >= 0:
                lb_contrib += g * pred_lb_flat[i]
                ub_contrib += g * pred_ub_flat[i]
            else:
                lb_contrib += g * pred_ub_flat[i]
                ub_contrib += g * pred_lb_flat[i]

        return center_val + lb_contrib, center_val + ub_contrib

    def get_ranges(self, lp_solver: str = 'default', parallel: bool = None,
                   n_workers: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute exact ranges for all pixels using LP.

        Args:
            lp_solver: LP solver to use
            parallel: If True, use parallel LP solving (None = use global config)
            n_workers: Number of workers (None = use global config)

        Returns:
            Tuple of (lb, ub) arrays, each shape (dim, 1)
        """
        # Determine parallelism settings
        if parallel is None:
            use_parallel = global_config.should_use_parallel(self.dim)
        else:
            use_parallel = parallel and self.dim > 1
        if n_workers is None:
            n_workers = global_config.get_n_workers(self.dim)

        if use_parallel:
            return self._get_ranges_parallel(lp_solver, n_workers)

        # Batch path
        return self._get_ranges_batch(lp_solver)

    def _get_ranges_parallel(self, lp_solver: str = 'default',
                             n_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ranges in parallel using ThreadPoolExecutor."""
        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        def compute_range(i):
            """Compute range for flat index i, returning (i, (lb, ub))."""
            try:
                return i, self.get_range_flat(i, lp_solver)
            except Exception:
                return i, self.estimate_range(i)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(compute_range, i) for i in range(self.dim)]
            for future in futures:
                i, (lb_i, ub_i) = future.result()
                lb[i, 0] = lb_i
                ub[i, 0] = ub_i

        self.state_lb = lb
        self.state_ub = ub
        return lb, ub

    def _get_ranges_batch(
        self, lp_solver: str = 'default',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ranges for all pixels using a single batched LP call.

        Args:
            lp_solver: LP solver to use

        Returns:
            Tuple of (lb, ub) arrays, each shape (dim, 1)
        """
        if self.nVar == 0:
            flat_center = self.V[:, :, :, 0].reshape(-1, 1)
            return flat_center.copy(), flat_center.copy()

        # Flatten generators: (H, W, C, nVar) -> (dim, nVar)
        generators_flat = self.V[:, :, :, 1:].reshape(-1, self.nVar)
        centers_flat = self.V[:, :, :, 0].flatten()

        # Build all objectives: min and max for each pixel
        objectives = []
        minimize_flags = []
        for i in range(self.dim):
            f = generators_flat[i, :]
            objectives.extend([f, f])
            minimize_flags.extend([True, False])

        results = self._solve_lp_batch(
            objectives, minimize_flags, lp_solver,
        )

        lb = np.zeros((self.dim, 1))
        ub = np.zeros((self.dim, 1))

        for i in range(self.dim):
            xmin_val = results[2 * i]
            xmax_val = results[2 * i + 1]
            if xmin_val is None or xmax_val is None:
                lb[i, 0], ub[i, 0] = self.estimate_range(i)
            else:
                lb[i, 0] = xmin_val + centers_flat[i]
                ub[i, 0] = xmax_val + centers_flat[i]

        self.state_lb = lb
        self.state_ub = ub
        return lb, ub

    def get_range(self, h: int, w: int, c: int, lp_solver: str = 'default') -> Tuple[float, float]:
        """
        Get exact bounds for a single pixel using LP.

        Args:
            h: Height index
            w: Width index
            c: Channel index
            lp_solver: LP solver to use

        Returns:
            Tuple of (min, max) values
        """
        if h < 0 or h >= self.height:
            raise ValueError(f"h={h} out of range [0, {self.height})")
        if w < 0 or w >= self.width:
            raise ValueError(f"w={w} out of range [0, {self.width})")
        if c < 0 or c >= self.num_channels:
            raise ValueError(f"c={c} out of range [0, {self.num_channels})")

        center = self.V[h, w, c, 0]
        f = self.V[h, w, c, 1:].flatten()

        results = self._solve_lp_batch(
            [f, f], [True, False], lp_solver,
        )

        xmin_val, xmax_val = results[0], results[1]
        if xmin_val is None or xmax_val is None:
            return None, None

        return center + xmin_val, center + xmax_val

    def get_range_flat(self, index: int, lp_solver: str = 'default') -> Tuple[float, float]:
        """
        Get exact bounds for a pixel by flat index.

        Args:
            index: Flattened index (0 to dim-1)
            lp_solver: LP solver to use

        Returns:
            Tuple of (min, max) values
        """
        # Convert flat index to (h, w, c)
        h = index // (self.width * self.num_channels)
        remainder = index % (self.width * self.num_channels)
        w = remainder // self.num_channels
        c = remainder % self.num_channels
        return self.get_range(h, w, c, lp_solver)

    def _solve_lp_batch(
        self,
        objectives: List[np.ndarray],
        minimize_flags: List[bool],
        lp_solver: str = 'default',
    ) -> List[Optional[float]]:
        """
        Batch solve LPs sharing this ImageStar's constraints.

        Args:
            objectives: List of objective vectors
            minimize_flags: List of booleans (True=minimize)
            lp_solver: Solver to use

        Returns:
            List of optimal objective values (None if infeasible)
        """
        if self.nVar == 0:
            return [0.0] * len(objectives)

        A = self.C if self.C.size > 0 else None
        b = self.d if self.C.size > 0 else None
        lb = self.predicate_lb
        ub = self.predicate_ub

        return solve_lp_batch(
            objectives=objectives, A=A, b=b,
            lb=lb, ub=ub,
            minimize_flags=minimize_flags,
            lp_solver=lp_solver,
        )

    def _solve_lp(
        self, f: np.ndarray, minimize: bool = True, lp_solver: str = 'default'
    ) -> Optional[float]:
        """
        Solve LP: min/max f^T * alpha subject to C * alpha <= d and bounds.
        """
        if self.nVar == 0:
            return 0.0

        A = self.C if self.C.size > 0 else None
        b = self.d if self.C.size > 0 else None
        lb = self.predicate_lb if self.predicate_lb is not None else None
        ub = self.predicate_ub if self.predicate_ub is not None else None

        x_opt, fval, status, info = solve_lp(
            f=f, A=A, b=b, lb=lb, ub=ub,
            lp_solver=lp_solver, minimize=minimize
        )

        if status in ['optimal', 'optimal_inaccurate']:
            return fval
        return None

    # ======================== Evaluation Methods ========================

    def evaluate(self, pred_val: np.ndarray) -> np.ndarray:
        """
        Evaluate the ImageStar at specific predicate values.

        Args:
            pred_val: Predicate values, shape (nVar,) or (nVar, 1)

        Returns:
            Image of shape (height, width, num_channels)

        Raises:
            ValueError: If pred_val has wrong dimension
        """
        pred_val = np.asarray(pred_val, dtype=np.float64).flatten()
        if pred_val.shape[0] != self.nVar:
            raise ValueError(
                f"Predicate vector has {pred_val.shape[0]} elements, expected {self.nVar}"
            )

        # Vectorized evaluation using einsum
        center = self.V[:, :, :, 0]        # (H, W, C)
        generators = self.V[:, :, :, 1:]   # (H, W, C, nVar)

        image = center + np.einsum('hwcn,n->hwc', generators, pred_val)
        return image

    def contains(self, image: np.ndarray, lp_solver: str = 'default') -> bool:
        """
        Check if an image is contained in the ImageStar set.

        Args:
            image: Image to check, shape (H, W, C) or flattened (H*W*C,)
            lp_solver: LP solver to use

        Returns:
            True if the image is in the set

        Raises:
            ValueError: If image dimensions don't match
        """
        import cvxpy as cp

        image = np.asarray(image, dtype=np.float64)

        # Handle both image-shaped and flattened inputs
        if image.shape == (self.height, self.width, self.num_channels):
            y = image.reshape(-1, 1)
        elif image.size == self.dim:
            y = image.reshape(-1, 1)
        else:
            raise ValueError(
                f"Image shape {image.shape} doesn't match ImageStar dimensions "
                f"({self.height}, {self.width}, {self.num_channels})"
            )

        # Flatten V for the feasibility check
        V_2d = self.V.reshape(-1, self.V.shape[3])

        # Solve: find alpha such that V[:, 0] + V[:, 1:] * alpha = y and C * alpha <= d
        alpha = cp.Variable(self.nVar)

        constraints = [V_2d[:, 1:] @ alpha == (y - V_2d[:, 0:1]).flatten()]

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
        except Exception:
            return False

    # ======================== Sampling Methods ========================

    def sample(self, N: int) -> List[np.ndarray]:
        """
        Generate random sample images from the ImageStar set.

        Args:
            N: Number of images to sample

        Returns:
            List of N images, each of shape (height, width, num_channels)
        """
        if self.nVar == 0:
            # No predicate variables - only one image possible
            center = self.V[:, :, :, 0]
            return [center.copy() for _ in range(N)]

        # Sample predicate values within constraints
        pred_samples = self._sample_predicates(N)

        images = []
        for i in range(pred_samples.shape[1]):
            image = self.evaluate(pred_samples[:, i])
            images.append(image)

        return images

    def _sample_predicates(self, N: int) -> np.ndarray:
        """
        Sample N valid predicate vectors satisfying constraints.

        Uses rejection sampling within predicate bounds.

        Returns:
            Array of shape (nVar, N) with sampled predicate values
        """
        samples = []
        max_attempts = N * 100
        attempts = 0

        while len(samples) < N and attempts < max_attempts:
            # Sample uniformly within predicate bounds
            alpha = (
                self.predicate_lb.flatten()
                + (self.predicate_ub.flatten() - self.predicate_lb.flatten())
                * np.random.rand(self.nVar)
            )

            # Check if it satisfies constraints C * alpha <= d
            if np.all(self.C @ alpha.reshape(-1, 1) <= self.d + 1e-10):
                samples.append(alpha)

            attempts += 1

        if len(samples) < N:
            # Duplicate samples if we couldn't find enough
            while len(samples) < N:
                samples.append(samples[np.random.randint(len(samples))])

        return np.column_stack(samples)

    # ======================== Set Operations ========================

    def is_empty_set(self, lp_solver: str = 'default') -> bool:
        """
        Check if ImageStar is empty (constraints are infeasible).

        Returns:
            True if empty, False otherwise
        """
        A = self.C if self.C.size > 0 else None
        b = self.d if self.C.size > 0 else None
        lb = self.predicate_lb if self.predicate_lb is not None else None
        ub = self.predicate_ub if self.predicate_ub is not None else None

        return not check_feasibility(A=A, b=b, lb=lb, ub=ub, lp_solver=lp_solver)

    def affine_map(self, W: np.ndarray, b: Optional[np.ndarray] = None) -> 'ImageStar':
        """
        Apply element-wise affine transformation: W * x + b.

        For spatial affine maps (convolution), use conv2d layer operations instead.
        This method applies a per-pixel affine transformation.

        Args:
            W: Weight matrix (dim, dim) for full transformation or (dim,) for diagonal
            b: Bias vector (dim,) or (H, W, C), optional

        Returns:
            New ImageStar after transformation
        """
        W = np.asarray(W, dtype=np.float64)

        # Handle diagonal case
        if W.ndim == 1:
            if W.shape[0] != self.dim:
                raise ValueError(f"W has {W.shape[0]} elements but dim={self.dim}")
            # Apply diagonal scaling to 4D V
            W_4d = W.reshape(self.height, self.width, self.num_channels)
            new_V = self.V * W_4d[:, :, :, np.newaxis]
        elif W.ndim == 2:
            if W.shape[1] != self.dim:
                raise ValueError(f"W has {W.shape[1]} columns but dim={self.dim}")
            # Full matrix multiplication - convert to Star, transform, convert back
            star = self.to_star()
            new_star = star.affine_map(W, b.flatten() if b is not None else None)
            # Note: Output may have different dimensions
            if W.shape[0] == self.dim:
                return ImageStar(
                    new_star.V, new_star.C, new_star.d,
                    new_star.predicate_lb, new_star.predicate_ub,
                    self.height, self.width, self.num_channels
                )
            else:
                raise ValueError("Non-square affine map changes dimensions")
        else:
            raise ValueError(f"W must be 1D or 2D, got {W.ndim}D")

        # Add bias if provided
        if b is not None:
            b = np.asarray(b, dtype=np.float64)
            if b.size != self.dim:
                raise ValueError(f"b has {b.size} elements but dim={self.dim}")
            b_4d = b.reshape(self.height, self.width, self.num_channels)
            new_V[:, :, :, 0] = new_V[:, :, :, 0] + b_4d

        return ImageStar(
            new_V, self.C, self.d,
            self.predicate_lb, self.predicate_ub,
            self.height, self.width, self.num_channels
        )

    def minkowski_sum(self, other: 'ImageStar') -> 'ImageStar':
        """
        Compute Minkowski sum with another ImageStar.

        Delegates to Star.minkowski_sum() and converts back.

        Args:
            other: Another ImageStar with same dimensions

        Returns:
            New ImageStar representing the Minkowski sum
        """
        if not isinstance(other, ImageStar):
            raise TypeError("Can only compute Minkowski sum with another ImageStar")
        if self.get_image_shape() != other.get_image_shape():
            raise ValueError(
                f"Shape mismatch: {self.get_image_shape()} vs {other.get_image_shape()}"
            )

        # Convert to Stars, compute, convert back
        s1 = self.to_star()
        s2 = other.to_star()
        result = s1.minkowski_sum(s2)

        # Convert back to ImageStar
        return ImageStar(
            result.V, result.C, result.d,
            result.predicate_lb, result.predicate_ub,
            self.height, self.width, self.num_channels
        )

    def convex_hull(self, other: 'ImageStar') -> 'ImageStar':
        """
        Compute over-approximation of convex hull with another ImageStar.

        Delegates to Star.convex_hull() and converts back.

        Args:
            other: Another ImageStar with same dimensions

        Returns:
            New ImageStar over-approximating the convex hull
        """
        if not isinstance(other, ImageStar):
            raise TypeError("Can only compute convex hull with another ImageStar")
        if self.get_image_shape() != other.get_image_shape():
            raise ValueError(
                f"Shape mismatch: {self.get_image_shape()} vs {other.get_image_shape()}"
            )

        s1 = self.to_star()
        s2 = other.to_star()
        result = s1.convex_hull(s2)

        return ImageStar(
            result.V, result.C, result.d,
            result.predicate_lb, result.predicate_ub,
            self.height, self.width, self.num_channels
        )

    def intersect_half_space(self, H: np.ndarray, g: np.ndarray) -> 'ImageStar':
        """
        Intersect ImageStar with half-space: H*x <= g.

        Delegates to Star.intersect_half_space() and converts back.

        Args:
            H: Half-space matrix
            g: Half-space vector

        Returns:
            New ImageStar representing the intersection
        """
        star = self.to_star()
        result = star.intersect_half_space(H, g)

        return ImageStar(
            result.V, result.C, result.d,
            result.predicate_lb, result.predicate_ub,
            self.height, self.width, self.num_channels
        )
