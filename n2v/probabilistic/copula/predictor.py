"""
Copula-based conformal predictor for probabilistic verification.

This module implements the CopulaConformalPredictor class which uses
copula density as the nonconformity score to produce tighter prediction
regions that account for correlation structure.
"""

import numpy as np
from scipy.stats import norm, beta
from typing import Callable, List, Optional, Tuple

from .marginal import MarginalCDF, KernelCDF
from .gaussian import GaussianCopula


class CopulaConformalPredictor:
    """
    Copula-based conformal predictor.

    Uses copula density as the conformity score:
        s(y) = -log(c_Sigma(F_1(r_1), ..., F_d(r_d)))

    where r = y - center is the residual and F_j are marginal CDFs.

    This approach produces tighter prediction regions than hyperrectangles
    by modeling the correlation structure between output dimensions.

    Attributes:
        center: Output space center (mean of training outputs)
        marginals: Per-dimension marginal CDF estimators
        copula: Fitted Gaussian copula
        threshold: Calibrated score threshold tau
        m: Calibration set size
        ell: Rank parameter
        epsilon: Miscoverage level
        coverage: delta_1 = 1 - epsilon
        confidence: delta_2 = 1 - beta.cdf(1-epsilon, ell, m+1-ell)
    """

    def __init__(self):
        """Initialize an uncalibrated predictor."""
        self.center: Optional[np.ndarray] = None
        self.marginals: Optional[List[MarginalCDF]] = None
        self.copula: Optional[GaussianCopula] = None
        self.threshold: Optional[float] = None
        self.m: Optional[int] = None
        self.ell: Optional[int] = None
        self.epsilon: Optional[float] = None
        self.coverage: Optional[float] = None
        self.confidence: Optional[float] = None
        self.d: Optional[int] = None

    def calibrate(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        input_lb: np.ndarray,
        input_ub: np.ndarray,
        training_samples: int = 4000,
        m: int = 8000,
        ell: Optional[int] = None,
        epsilon: float = 0.001,
        batch_size: int = 100,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> 'CopulaConformalPredictor':
        """
        Calibrate the predictor using training and calibration samples.

        Algorithm:
        1. Generate training samples, compute outputs, center them
        2. Fit marginal CDFs per dimension
        3. Transform residuals to pseudo-observations U in [0,1]^d
        4. Fit Gaussian copula (correlation matrix)
        5. Generate calibration samples
        6. Compute scores: s = -log(copula_density(u))
        7. Set threshold: tau = sorted_scores[ell-1]

        Args:
            model: Function mapping input arrays to output arrays
            input_lb: Lower bounds of input region
            input_ub: Upper bounds of input region
            training_samples: Number of training samples (t)
            m: Calibration set size
            ell: Rank parameter (default: m-1)
            epsilon: Miscoverage level
            batch_size: Batch size for model evaluation
            seed: Random seed for reproducibility
            verbose: Print progress information

        Returns:
            self
        """
        rng = np.random.default_rng(seed)

        input_lb = np.asarray(input_lb).flatten()
        input_ub = np.asarray(input_ub).flatten()
        input_dim = len(input_lb)

        if ell is None:
            ell = m - 1

        self.m = m
        self.ell = ell
        self.epsilon = epsilon
        self.coverage = 1 - epsilon
        self.confidence = self._compute_confidence(m, ell, epsilon)

        if verbose:
            print(f"Calibrating copula predictor with t={training_samples}, m={m}")

        # Step 1: Generate training samples
        if verbose:
            print("  Generating training samples...")

        train_inputs = rng.uniform(
            input_lb, input_ub,
            size=(training_samples, input_dim)
        )
        train_outputs = self._evaluate_model(model, train_inputs, batch_size)

        # Compute center and residuals
        self.center = np.mean(train_outputs, axis=0)
        self.d = len(self.center)
        train_residuals = train_outputs - self.center

        if verbose:
            print(f"  Output dimension: {self.d}")

        # Step 2: Fit marginal CDFs per dimension
        if verbose:
            print("  Fitting marginal CDFs...")

        self.marginals = []
        for j in range(self.d):
            marginal = KernelCDF(train_residuals[:, j])
            self.marginals.append(marginal)

        # Step 3: Transform residuals to pseudo-observations
        U_train = self._to_pseudo_observations(train_residuals)

        # Step 4: Fit Gaussian copula
        if verbose:
            print("  Fitting Gaussian copula...")

        self.copula = GaussianCopula()
        self.copula.fit(U_train)

        if verbose and self.d > 1:
            corr = self.copula.get_correlation()
            off_diag = corr[np.triu_indices(self.d, k=1)]
            if len(off_diag) > 0:
                print(f"  Mean off-diagonal correlation: {np.mean(np.abs(off_diag)):.4f}")

        # Step 5: Generate calibration samples
        if verbose:
            print("  Generating calibration samples...")

        calib_inputs = rng.uniform(
            input_lb, input_ub,
            size=(m, input_dim)
        )
        calib_outputs = self._evaluate_model(model, calib_inputs, batch_size)
        calib_residuals = calib_outputs - self.center

        # Step 6: Compute conformity scores (full likelihood: copula + marginals)
        if verbose:
            print("  Computing conformity scores...")

        U_calib = self._to_pseudo_observations(calib_residuals)

        # Compute marginal log-densities for calibration samples
        marginal_log_densities = np.zeros(m)
        for j in range(self.d):
            marginal_log_densities += self.marginals[j].log_density(calib_residuals[:, j])

        # Full score: -log(copula_density) - log(marginal_densities)
        copula_log_densities = self.copula.log_density_batch(U_calib)
        scores = -copula_log_densities - marginal_log_densities

        # Clip scores for numerical stability
        scores = np.clip(scores, -1e10, 1e10)

        # Step 7: Set threshold (ell-th smallest score)
        sorted_scores = np.sort(scores)
        self.threshold = sorted_scores[ell - 1]

        if verbose:
            print(f"  Threshold tau: {self.threshold:.4f}")
            print(f"  Coverage: {self.coverage:.4f}, Confidence: {self.confidence:.4f}")

        return self

    def _evaluate_model(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        inputs: np.ndarray,
        batch_size: int
    ) -> np.ndarray:
        """Evaluate model in batches."""
        n = len(inputs)
        outputs = []

        for i in range(0, n, batch_size):
            batch = inputs[i:min(i + batch_size, n)]
            batch_out = model(batch)
            if batch_out.ndim == 1:
                batch_out = batch_out.reshape(-1, 1)
            outputs.append(batch_out)

        return np.vstack(outputs)

    def _to_pseudo_observations(self, residuals: np.ndarray) -> np.ndarray:
        """
        Transform residuals to pseudo-observations in [0, 1]^d.

        Args:
            residuals: Residuals, shape (n, d)

        Returns:
            Pseudo-observations, shape (n, d), values in [0, 1]
        """
        n = residuals.shape[0]
        U = np.zeros((n, self.d))

        for j in range(self.d):
            U[:, j] = self.marginals[j].cdf(residuals[:, j])

        # Clip to avoid numerical issues at boundaries
        U = np.clip(U, 1e-10, 1 - 1e-10)

        return U

    def _compute_confidence(self, m: int, ell: int, epsilon: float) -> float:
        """Compute confidence level delta_2."""
        return 1 - beta.cdf(1 - epsilon, ell, m + 1 - ell)

    def score(self, y: np.ndarray) -> float:
        """
        Compute the conformity score for a point y.

        Uses the full negative log-likelihood:
            s(y) = -log(c_Sigma(u)) - sum_j log(f_j(r_j))

        where u_j = F_j(r_j) are pseudo-observations and f_j are marginal densities.
        This ensures points outside the marginal support get high scores.

        Args:
            y: Output point, shape (d,)

        Returns:
            Conformity score (higher = more unusual)
        """
        if self.center is None:
            raise ValueError("Predictor must be calibrated before scoring")

        y = np.asarray(y).flatten()
        if len(y) != self.d:
            raise ValueError(f"Expected {self.d} dimensions, got {len(y)}")

        # Compute residual
        r = y - self.center

        # Transform to pseudo-observations and compute marginal log-densities
        u = np.zeros(self.d)
        marginal_log_density = 0.0
        for j in range(self.d):
            u[j] = self.marginals[j].cdf(r[j])
            # Compute marginal log-density (kernel density)
            marginal_log_density += self.marginals[j].log_density(r[j])

        u = np.clip(u, 1e-10, 1 - 1e-10)

        # Compute full score: -log(copula_density) - log(marginal_densities)
        copula_log_density = self.copula.log_density(u)
        score = -copula_log_density - marginal_log_density

        return float(np.clip(score, -1e10, 1e10))

    def contains(self, y: np.ndarray) -> bool:
        """
        Test if a point y is in the prediction region.

        Args:
            y: Output point, shape (d,)

        Returns:
            True if y is in the region (score <= threshold)
        """
        return bool(self.score(y) <= self.threshold)

    def contains_batch(self, Y: np.ndarray) -> np.ndarray:
        """
        Test if multiple points are in the prediction region.

        Args:
            Y: Output points, shape (n, d)

        Returns:
            Boolean array, shape (n,)
        """
        if self.center is None:
            raise ValueError("Predictor must be calibrated before testing")

        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        n = Y.shape[0]

        # Compute residuals
        R = Y - self.center

        # Transform to pseudo-observations and compute marginal log-densities
        U = np.zeros((n, self.d))
        marginal_log_densities = np.zeros(n)

        for j in range(self.d):
            U[:, j] = self.marginals[j].cdf(R[:, j])
            marginal_log_densities += self.marginals[j].log_density(R[:, j])

        U = np.clip(U, 1e-10, 1 - 1e-10)

        # Compute full scores: -log(copula_density) - log(marginal_densities)
        copula_log_densities = self.copula.log_density_batch(U)
        scores = -copula_log_densities - marginal_log_densities
        scores = np.clip(scores, -1e10, 1e10)

        return scores <= self.threshold

    def sample(
        self,
        n: int,
        max_attempts: int = 100000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample points from the prediction region using rejection sampling.

        Algorithm:
        1. Sample z ~ N(0, Sigma) to get correlated pseudo-observations
        2. Transform u = Phi(z) to uniform marginals
        3. Transform to original space: r = F^{-1}(u), y = center + r
        4. Accept if score(y) <= threshold (full likelihood test)

        Args:
            n: Number of samples to generate
            max_attempts: Maximum number of attempts before giving up
            seed: Random seed for reproducibility

        Returns:
            Samples, shape (n, d)
        """
        if self.center is None:
            raise ValueError("Predictor must be calibrated before sampling")

        rng = np.random.default_rng(seed)
        samples = []
        attempts = 0

        while len(samples) < n and attempts < max_attempts:
            # Sample from copula (accounts for correlation)
            batch_size = min(1000, (n - len(samples)) * 5)  # oversample due to rejection
            U = self.copula.sample(batch_size, seed=rng.integers(0, 2**31))

            # Transform back to original space
            Y = np.zeros((batch_size, self.d))
            for j in range(self.d):
                Y[:, j] = self.marginals[j].inverse(U[:, j])
            Y = Y + self.center

            # Accept samples that are in the region
            accepted = self.contains_batch(Y)

            for y in Y[accepted]:
                if len(samples) >= n:
                    break
                samples.append(y)

            attempts += batch_size

        if len(samples) < n:
            print(f"Warning: Only generated {len(samples)}/{n} samples after {attempts} attempts")

        return np.array(samples[:n])

    def get_bounding_box(
        self,
        n_samples: int = 10000,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the bounding box of the prediction region.

        Uses sampling to estimate min/max bounds.

        Args:
            n_samples: Number of samples for estimation
            seed: Random seed

        Returns:
            Tuple of (lower_bounds, upper_bounds), each shape (d,)
        """
        samples = self.sample(n_samples, seed=seed)

        lb = np.min(samples, axis=0)
        ub = np.max(samples, axis=0)

        return lb, ub

    def get_guarantee(self) -> Tuple[float, float]:
        """
        Get the probabilistic guarantee parameters.

        Returns:
            Tuple of (coverage, confidence)
        """
        return (self.coverage, self.confidence)

    def get_guarantee_string(self) -> str:
        """
        Get a human-readable description of the guarantee.

        Returns:
            Description string
        """
        return (
            f"Probabilistic Guarantee: "
            f"Coverage >= {100 * self.coverage:.2f}% with "
            f"{100 * self.confidence:.2f}% confidence "
            f"(m={self.m}, ell={self.ell}, epsilon={self.epsilon})"
        )
