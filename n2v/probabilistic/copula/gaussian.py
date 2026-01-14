"""
Gaussian copula implementation for copula-based conformal prediction.

The Gaussian copula captures linear correlation structure between dimensions
while allowing arbitrary marginal distributions.
"""

import numpy as np
from scipy.stats import norm
from typing import Optional


class GaussianCopula:
    """
    Gaussian copula for modeling dependency structure.

    The Gaussian copula is defined as:
        C_Sigma(u) = Phi_Sigma(Phi^{-1}(u_1), ..., Phi^{-1}(u_d))

    where Phi is the standard normal CDF, Phi^{-1} is its inverse,
    and Phi_Sigma is the multivariate normal CDF with correlation matrix Sigma.

    The copula density is:
        c_Sigma(u) = |Sigma|^{-1/2} * exp(-1/2 * xi^T (Sigma^{-1} - I) xi)

    where xi_j = Phi^{-1}(u_j).

    Attributes:
        correlation: Fitted correlation matrix (d x d)
        correlation_inv: Inverse of correlation matrix
        log_det: Log determinant of correlation matrix
        d: Dimensionality
    """

    def __init__(self):
        """Initialize an unfitted Gaussian copula."""
        self.correlation: Optional[np.ndarray] = None
        self.correlation_inv: Optional[np.ndarray] = None
        self.log_det: Optional[float] = None
        self.d: Optional[int] = None
        self._cholesky: Optional[np.ndarray] = None

    def fit(self, U: np.ndarray, min_eigenvalue: float = 1e-6) -> 'GaussianCopula':
        """
        Fit the Gaussian copula from pseudo-observations.

        Args:
            U: Pseudo-observations, shape (n, d), values in [0, 1]
            min_eigenvalue: Minimum eigenvalue for regularization

        Returns:
            self
        """
        U = np.asarray(U)
        if U.ndim == 1:
            U = U.reshape(-1, 1)

        n, d = U.shape
        self.d = d

        if n < 2:
            raise ValueError("Need at least 2 samples to fit copula")

        # Clip pseudo-observations to avoid infinite values at boundaries
        U_clipped = np.clip(U, 1e-10, 1 - 1e-10)

        # Transform to normal space: Z = Phi^{-1}(U)
        Z = norm.ppf(U_clipped)

        # Compute correlation matrix
        if d == 1:
            self.correlation = np.array([[1.0]])
        else:
            self.correlation = np.corrcoef(Z.T)

        # Regularize for positive definiteness
        self.correlation = self._regularize(self.correlation, min_eigenvalue)

        # Precompute inverse and log determinant
        self.correlation_inv = np.linalg.inv(self.correlation)
        sign, self.log_det = np.linalg.slogdet(self.correlation)

        if sign <= 0:
            raise ValueError("Correlation matrix has non-positive determinant")

        # Precompute Cholesky decomposition for sampling
        self._cholesky = np.linalg.cholesky(self.correlation)

        return self

    def _regularize(
        self,
        matrix: np.ndarray,
        min_eigenvalue: float = 1e-6
    ) -> np.ndarray:
        """
        Regularize matrix to ensure positive definiteness.

        Clips eigenvalues to be at least min_eigenvalue.

        Args:
            matrix: Matrix to regularize
            min_eigenvalue: Minimum eigenvalue threshold

        Returns:
            Regularized positive definite matrix
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Clip eigenvalues
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

        # Reconstruct matrix
        regularized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Ensure diagonal is exactly 1 (correlation matrix property)
        d = np.sqrt(np.diag(regularized))
        regularized = regularized / np.outer(d, d)

        return regularized

    def log_density(self, u: np.ndarray) -> float:
        """
        Compute the log of the copula density at point u.

        log c_Sigma(u) = -1/2 * log|Sigma| - 1/2 * xi^T (Sigma^{-1} - I) xi

        Args:
            u: Pseudo-observation, shape (d,), values in [0, 1]

        Returns:
            Log density value
        """
        if self.correlation is None:
            raise ValueError("Copula must be fitted before computing density")

        u = np.asarray(u).flatten()
        if len(u) != self.d:
            raise ValueError(f"Expected {self.d} dimensions, got {len(u)}")

        # Clip and transform to normal space
        u_clipped = np.clip(u, 1e-10, 1 - 1e-10)
        xi = norm.ppf(u_clipped)

        # Compute quadratic form: xi^T (Sigma^{-1} - I) xi
        I = np.eye(self.d)
        diff = self.correlation_inv - I
        quadratic = xi @ diff @ xi

        # Log density
        log_dens = -0.5 * self.log_det - 0.5 * quadratic

        return log_dens

    def log_density_batch(self, U: np.ndarray) -> np.ndarray:
        """
        Compute log density for multiple points (vectorized).

        Args:
            U: Pseudo-observations, shape (n, d), values in [0, 1]

        Returns:
            Log density values, shape (n,)
        """
        if self.correlation is None:
            raise ValueError("Copula must be fitted before computing density")

        U = np.asarray(U)
        if U.ndim == 1:
            U = U.reshape(1, -1)

        n = U.shape[0]

        # Clip and transform to normal space
        U_clipped = np.clip(U, 1e-10, 1 - 1e-10)
        Xi = norm.ppf(U_clipped)  # Shape: (n, d)

        # Compute quadratic forms: xi^T (Sigma^{-1} - I) xi for each row
        I = np.eye(self.d)
        diff = self.correlation_inv - I

        # Vectorized quadratic form computation
        # (Xi @ diff) has shape (n, d), then element-wise multiply with Xi and sum
        quadratics = np.sum((Xi @ diff) * Xi, axis=1)

        # Log densities
        log_densities = -0.5 * self.log_det - 0.5 * quadratics

        return log_densities

    def density(self, u: np.ndarray) -> float:
        """
        Compute the copula density at point u.

        Args:
            u: Pseudo-observation, shape (d,), values in [0, 1]

        Returns:
            Density value (non-negative)
        """
        return np.exp(self.log_density(u))

    def sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample from the Gaussian copula.

        Args:
            n: Number of samples
            seed: Random seed for reproducibility

        Returns:
            Samples, shape (n, d), values in [0, 1]
        """
        if self.correlation is None:
            raise ValueError("Copula must be fitted before sampling")

        rng = np.random.default_rng(seed)

        # Sample from standard multivariate normal
        Z = rng.standard_normal((n, self.d))

        # Apply correlation structure via Cholesky
        Z_correlated = Z @ self._cholesky.T

        # Transform to uniform [0, 1] via normal CDF
        U = norm.cdf(Z_correlated)

        return U

    def get_correlation(self) -> np.ndarray:
        """
        Get the fitted correlation matrix.

        Returns:
            Correlation matrix, shape (d, d)
        """
        if self.correlation is None:
            raise ValueError("Copula must be fitted first")
        return self.correlation.copy()
