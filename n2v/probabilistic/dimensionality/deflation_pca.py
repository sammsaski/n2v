"""
Deflation-based PCA for high-dimensional output reduction.

This module implements Algorithm 1 from the paper: a variant of PCA that
works well when the number of samples t is much smaller than the dimension n.
Instead of computing eigenvectors of the n×n covariance matrix (which is
expensive and potentially singular), we iteratively find principal directions
using gradient ascent and deflation.
"""

import numpy as np
from typing import Optional
import warnings


class DeflationPCA:
    """
    Deflation-based PCA for high-dimensional data.

    Standard PCA computes eigenvectors of the n×n covariance matrix, which:
    - Requires O(n²) memory
    - Is singular when t < n (more dimensions than samples)

    Deflation PCA instead:
    - Iteratively finds one principal direction at a time
    - Uses gradient ascent on the variance objective
    - Removes (deflates) each found direction before finding the next

    This is efficient when t << n (few samples, high dimension).

    Attributes:
        n_components: Number of principal components to extract
        max_iter: Maximum iterations for gradient ascent per component
        tol: Convergence tolerance
        components_: Fitted principal directions, shape (n_components, n)
        mean_: Mean of training data, shape (n,)

    Example:
        >>> pca = DeflationPCA(n_components=100)
        >>> pca.fit(training_outputs)  # Shape: (t, n) where n >> t
        >>>
        >>> # Project to lower dimension
        >>> reduced = pca.transform(outputs)  # Shape: (m, 100)
        >>>
        >>> # Project back to full dimension
        >>> reconstructed = pca.inverse_transform(reduced)  # Shape: (m, n)
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 1000,
        tol: float = 1e-6,
        learning_rate: float = 0.1,
        verbose: bool = False
    ):
        """
        Initialize DeflationPCA.

        Args:
            n_components: Number of components to extract (N in the paper)
            max_iter: Maximum gradient ascent iterations per component
            tol: Convergence tolerance (stop when change < tol)
            learning_rate: Step size for gradient ascent
            verbose: Print progress during fitting
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.components_ = None  # Shape: (n_components, n)
        self.mean_ = None  # Shape: (n,)
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> 'DeflationPCA':
        """
        Fit PCA using deflation algorithm.

        Algorithm:
        1. Center the data
        2. For each component:
           a. Initialize random direction
           b. Gradient ascent to maximize variance
           c. Deflate: remove projection onto found direction

        Args:
            X: Data matrix of shape (t, n) where t is samples, n is dimension

        Returns:
            self (for method chaining)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        t, n = X.shape

        actual_components = self.n_components
        if actual_components > min(t, n):
            warnings.warn(
                f"n_components={self.n_components} > min(t={t}, n={n}). "
                f"Setting n_components={min(t, n)}"
            )
            actual_components = min(t, n)

        # Center data
        self.mean_ = np.mean(X, axis=0)  # Shape: (n,)
        Z = X - self.mean_  # Centered data, shape: (t, n)

        # Store principal directions
        self.components_ = np.zeros((actual_components, n))

        for idx in range(actual_components):
            if self.verbose:
                print(f"Finding component {idx + 1}/{actual_components}")

            # Find principal direction for current (deflated) data
            a = self._find_principal_direction(Z)
            self.components_[idx] = a

            # Deflate: remove component along this direction
            # z_j = z_j - (a^T z_j) * a
            projections = Z @ a  # Shape: (t,), projection coefficients
            Z = Z - np.outer(projections, a)  # Remove projections

        self._is_fitted = True
        return self

    def _find_principal_direction(self, Z: np.ndarray) -> np.ndarray:
        """
        Find the principal direction maximizing variance.

        Objective: max_a J(a) = (1/t) * Σ_j (a^T z_j)²
        Constraint: ||a||₂ = 1

        Gradient: ∂J/∂a = (2/t) * Σ_j (a^T z_j) * z_j = (2/t) * Z^T @ Z @ a

        Use projected gradient ascent: update a, then normalize.

        Args:
            Z: Centered data matrix of shape (t, n)

        Returns:
            Principal direction of shape (n,)
        """
        t, n = Z.shape

        # Initialize with random unit vector
        a = np.random.randn(n)
        a = a / np.linalg.norm(a)

        # Precompute Z^T @ Z for efficiency (only if t < n)
        # Otherwise compute Z @ a and Z^T @ (Z @ a) separately
        use_gram = (t < n)
        if use_gram and t * n < 1e8:  # Memory limit check
            try:
                ZtZ = Z.T @ Z  # Shape: (n, n) - may be large but saves time
            except MemoryError:
                use_gram = False

        prev_objective = -np.inf

        for iteration in range(self.max_iter):
            # Compute gradient
            if use_gram:
                gradient = (2 / t) * ZtZ @ a
            else:
                Za = Z @ a  # Shape: (t,)
                gradient = (2 / t) * Z.T @ Za  # Shape: (n,)

            # Gradient ascent step
            a_new = a + self.learning_rate * gradient

            # Project onto unit sphere (normalize)
            norm = np.linalg.norm(a_new)
            if norm < 1e-10:
                # Degenerate case - restart with random direction
                a_new = np.random.randn(n)
                norm = np.linalg.norm(a_new)
            a_new = a_new / norm

            # Check convergence
            objective = np.mean((Z @ a_new) ** 2)
            if abs(objective - prev_objective) < self.tol:
                if self.verbose:
                    print(f"  Converged at iteration {iteration + 1}")
                return a_new

            a = a_new
            prev_objective = objective

        if self.verbose:
            print(f"  Max iterations reached")

        return a

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto principal components.

        Args:
            X: Data matrix of shape (m, n)

        Returns:
            Projected data of shape (m, n_components)
        """
        if not self._is_fitted:
            raise RuntimeError("PCA must be fitted before transform")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Center and project
        X_centered = X - self.mean_
        return X_centered @ self.components_.T  # Shape: (m, n_components)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Project back from reduced space to original space.

        Args:
            X_reduced: Reduced data of shape (m, n_components)

        Returns:
            Reconstructed data of shape (m, n)
        """
        if not self._is_fitted:
            raise RuntimeError("PCA must be fitted before inverse_transform")

        if X_reduced.ndim == 1:
            X_reduced = X_reduced.reshape(1, -1)

        # Project back and add mean
        return X_reduced @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.

        Args:
            X: Data matrix of shape (t, n)

        Returns:
            Projected data of shape (t, n_components)
        """
        self.fit(X)
        return self.transform(X)
