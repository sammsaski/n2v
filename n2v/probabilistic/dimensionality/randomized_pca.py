"""Randomized-SVD PCA for high-dimensional output reduction.

Drop-in replacement for :class:`DeflationPCA` (same public interface:
``components_`` of shape ``(k, n)``, ``mean_`` of shape ``(n,)``,
``fit`` / ``fit_transform`` / ``transform`` / ``inverse_transform``), but using
the randomized range-finder + small dense SVD of Halko, Martinsson & Tropp
(2011). Cost is ``O(t·n·l)`` with ``l = k + oversamples`` (plus a few power
iterations), versus the per-component gradient ascent of the deflation variant —
much faster and more accurate when the target rank ``k`` is modest, which is the
regime for reducing a (down-sampled) dense depth / segmentation output before
flow-matching reachability.

Components are the top-``k`` right singular vectors of the centred data, i.e.
orthonormal rows spanning the principal subspace, so
``transform(X) = (X - mean_) @ components_.T`` and
``inverse_transform(Z) = Z @ components_ + mean_`` — identical conventions to
:class:`DeflationPCA`, so both ``conformal_reach`` and ``flow_reach`` consume
either interchangeably.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RandomizedSVDPCA:
    """PCA via randomized SVD (Halko-Martinsson-Tropp).

    Attributes (set after :meth:`fit`):
        components_: Principal directions, shape ``(n_components_fitted_, n)``
            (orthonormal rows).
        mean_: Column-wise mean of the training data, shape ``(n,)``.
        singular_values_: Singular values of the centred data, shape ``(k,)``.
        explained_variance_: Per-component variance ``s**2 / (t - 1)``.
        n_components_fitted_: Actual number of components (``<= n_components``,
            capped at ``min(t, n)``).
    """

    def __init__(
        self,
        n_components: int,
        n_oversamples: int = 10,
        n_iter: int = 4,
        random_state: int = 0,
        verbose: bool = False,
    ):
        """
        Args:
            n_components: Target number of components ``k``.
            n_oversamples: Extra random dimensions for the range finder
                (accuracy vs cost; 10 is a good default).
            n_iter: Power iterations (with QR re-orthonormalisation) — improves
                accuracy when the spectrum decays slowly. 4 is a safe default.
            random_state: Seed for the random test matrix (reproducibility).
            verbose: Emit debug logging.
        """
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = int(n_components)
        self.n_oversamples = int(n_oversamples)
        self.n_iter = int(n_iter)
        self.random_state = random_state
        self.verbose = verbose

        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.n_components_fitted_: Optional[int] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "RandomizedSVDPCA":
        """Fit on ``X`` of shape ``(t, n)`` (t samples, n features)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (t, n), got shape {X.shape}")
        t, n = X.shape

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        k = min(self.n_components, t, n)
        ell = min(k + self.n_oversamples, min(t, n))

        rng = np.random.default_rng(self.random_state)
        omega = rng.standard_normal((n, ell))

        # Randomized range finder for the column space of Xc, with power
        # iterations (QR-stabilised) to sharpen the captured subspace.
        Y = Xc @ omega                       # (t, ell)
        Q, _ = np.linalg.qr(Y)               # (t, ell)
        for _ in range(self.n_iter):
            Z, _ = np.linalg.qr(Xc.T @ Q)    # (n, ell)
            Q, _ = np.linalg.qr(Xc @ Z)      # (t, ell)

        # Project and take the SVD of the small (ell, n) matrix.
        Bmat = Q.T @ Xc                      # (ell, n)
        _U, s, Vt = np.linalg.svd(Bmat, full_matrices=False)  # Vt: (ell, n)

        self.components_ = Vt[:k]            # (k, n), orthonormal rows
        self.singular_values_ = s[:k]
        self.explained_variance_ = (s[:k] ** 2) / max(t - 1, 1)
        self.n_components_fitted_ = k
        self._is_fitted = True
        if self.verbose:
            logger.debug("RandomizedSVDPCA: fit t=%d n=%d -> k=%d (ell=%d)",
                         t, n, k, ell)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project ``X`` ``(m, n)`` to the reduced space ``(m, k)``."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct full-space outputs ``(m, n)`` from reduced ``(m, k)``."""
        self._check_fitted()
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("RandomizedSVDPCA must be fit before use")
