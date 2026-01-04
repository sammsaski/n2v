"""
Clipping Block surrogate: projects outputs onto convex hull of training data.

This surrogate constructs the convex hull of training outputs and projects
calibration outputs onto this hull using linear programming. This produces
tighter bounds than the naive approach by exploiting correlation structure.
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import warnings

from n2v.probabilistic.surrogates.base import Surrogate


class ClippingBlockSurrogate(Surrogate):
    """
    Clipping Block surrogate using convex hull projection.

    For each calibration output y, finds the closest point in the convex hull
    of training outputs by solving:

        min_α ||y - Σ_j α_j y_train_j||_∞
        s.t.  Σ_j α_j = 1
              α_j >= 0

    This is reformulated as a linear program:

        min  z
        s.t. y - Y @ α <=  z * 1
             y - Y @ α >= -z * 1
             1^T @ α = 1
             α >= 0

    Where Y is the matrix of training outputs (each column is a sample).

    Example:
        >>> surrogate = ClippingBlockSurrogate()
        >>> surrogate.fit(training_outputs)  # Shape: (t, n)
        >>>
        >>> # Project calibration outputs onto convex hull
        >>> projections = surrogate.predict(calibration_outputs)
        >>> errors = calibration_outputs - projections
        >>>
        >>> # Bounds come from extreme points of convex hull
        >>> lb, ub = surrogate.get_bounds()
    """

    def __init__(self, n_workers: int = 4, verbose: bool = False):
        """
        Initialize ClippingBlockSurrogate.

        Args:
            n_workers: Number of parallel workers for LP solving
            verbose: Print progress during projection
        """
        self.vertices = None  # Training outputs (vertices of convex hull)
        self.n_samples = None  # Number of training samples (t)
        self.n_dim = None  # Output dimension (n)
        self.lb = None  # Lower bounds of convex hull
        self.ub = None  # Upper bounds of convex hull
        self.n_workers = n_workers
        self.verbose = verbose
        self._is_fitted = False

    def fit(self, training_outputs: np.ndarray) -> None:
        """
        Store training outputs as vertices of convex hull.

        Args:
            training_outputs: Array of shape (t, n)
        """
        if training_outputs.ndim == 1:
            training_outputs = training_outputs.reshape(1, -1)

        self.vertices = training_outputs.copy()  # Shape: (t, n)
        self.n_samples, self.n_dim = self.vertices.shape

        # Compute bounds of convex hull (element-wise min/max of vertices)
        self.lb = np.min(self.vertices, axis=0)  # Shape: (n,)
        self.ub = np.max(self.vertices, axis=0)  # Shape: (n,)

        self._is_fitted = True

    def predict(self, outputs: np.ndarray) -> np.ndarray:
        """
        Project outputs onto convex hull of training data.

        Args:
            outputs: Array of shape (m, n)

        Returns:
            Projected outputs of shape (m, n)
        """
        if not self._is_fitted:
            raise RuntimeError("Surrogate must be fitted before predicting")

        if outputs.ndim == 1:
            outputs = outputs.reshape(1, -1)

        m, n = outputs.shape
        if n != self.n_dim:
            raise ValueError(f"Output dimension {n} doesn't match fitted dimension {self.n_dim}")

        # Project each output in parallel
        projections = np.zeros_like(outputs)

        if self.n_workers > 1 and m > 1:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(self._project_single, outputs[i])
                    for i in range(m)
                ]
                for i, future in enumerate(futures):
                    projections[i] = future.result()
                    if self.verbose and (i + 1) % 1000 == 0:
                        print(f"Projected {i + 1}/{m} samples")
        else:
            for i in range(m):
                projections[i] = self._project_single(outputs[i])
                if self.verbose and (i + 1) % 1000 == 0:
                    print(f"Projected {i + 1}/{m} samples")

        return projections

    def _project_single(self, y: np.ndarray) -> np.ndarray:
        """
        Project a single output onto the convex hull.

        Solves:
            min  z
            s.t. y - Y^T @ α <=  z * 1
                 y - Y^T @ α >= -z * 1
                 1^T @ α = 1
                 α >= 0

        Variables: [z, α_1, α_2, ..., α_t]

        Args:
            y: Output vector of shape (n,)

        Returns:
            Projected output of shape (n,)
        """
        n = self.n_dim
        t = self.n_samples

        # Decision variables: [z, α_1, ..., α_t]
        # Objective: minimize z
        c = np.zeros(1 + t)
        c[0] = 1.0  # Minimize z

        # Inequality constraints: A_ub @ x <= b_ub
        #
        # Constraint 1: y[k] - Σ_j α_j * vertices[j,k] <= z for each k
        #   Rewrite: -z + Σ_j (-vertices[j,k]) * α_j <= -y[k]
        #   So row k of A_ub is: [-1, -vertices[:,k]]
        #   And b_ub[k] = -y[k]
        #
        # Constraint 2: -(y[k] - Σ_j α_j * vertices[j,k]) <= z for each k
        #   => -y[k] + Σ_j α_j * vertices[j,k] <= z
        #   Rewrite: -z + Σ_j vertices[j,k] * α_j <= y[k]
        #   So row n+k of A_ub is: [-1, vertices[:,k]]
        #   And b_ub[n+k] = y[k]

        A_ub = np.zeros((2 * n, 1 + t))
        b_ub = np.zeros(2 * n)

        for k in range(n):
            # Constraint: y[k] - Σ_j α_j * vertices[j,k] <= z
            A_ub[k, 0] = -1  # coefficient of z
            A_ub[k, 1:] = -self.vertices[:, k]  # coefficients of α
            b_ub[k] = -y[k]

            # Constraint: -y[k] + Σ_j α_j * vertices[j,k] <= z
            A_ub[n + k, 0] = -1  # coefficient of z
            A_ub[n + k, 1:] = self.vertices[:, k]  # coefficients of α
            b_ub[n + k] = y[k]

        # Equality constraint: Σ_j α_j = 1
        A_eq = np.zeros((1, 1 + t))
        A_eq[0, 0] = 0  # z not involved
        A_eq[0, 1:] = 1  # sum of α = 1
        b_eq = np.array([1.0])

        # Bounds: z >= 0, α >= 0
        bounds = [(0, None)] + [(0, None)] * t  # z >= 0, α_j >= 0

        # Solve LP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )

        if not result.success:
            # Fallback: return closest vertex
            distances = np.max(np.abs(self.vertices - y), axis=1)
            closest_idx = np.argmin(distances)
            return self.vertices[closest_idx].copy()

        # Extract α and compute projection
        alpha = result.x[1:]  # Shape: (t,)
        projection = self.vertices.T @ alpha  # Shape: (n,)

        return projection

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds of the convex hull (before inflation).

        Returns:
            Tuple of (lower_bounds, upper_bounds), each of shape (n,)
        """
        if not self._is_fitted:
            raise RuntimeError("Surrogate must be fitted before getting bounds")

        return (self.lb.copy(), self.ub.copy())


class BatchedClippingBlockSurrogate(ClippingBlockSurrogate):
    """
    Memory-efficient batched version for very large calibration sets.

    Processes calibration samples in batches to avoid memory issues.
    """

    def __init__(self, batch_size: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def predict(self, outputs: np.ndarray) -> np.ndarray:
        """
        Project outputs in batches.

        Args:
            outputs: Array of shape (m, n)

        Returns:
            Projected outputs of shape (m, n)
        """
        if not self._is_fitted:
            raise RuntimeError("Surrogate must be fitted before predicting")

        if outputs.ndim == 1:
            outputs = outputs.reshape(1, -1)

        m, n = outputs.shape
        projections = np.zeros_like(outputs)

        n_batches = (m + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, m)

            if self.verbose:
                print(f"Processing batch {batch_idx + 1}/{n_batches} (samples {start}-{end})")

            batch_outputs = outputs[start:end]
            # Use parent's predict for this batch
            projections[start:end] = super().predict(batch_outputs)

        return projections
