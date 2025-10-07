"""
Conformal inference primitives for probabilistic verification.

This module implements the core conformal inference computations:
- Nonconformity score calculation
- Threshold selection
- Guarantee computation
"""

import numpy as np
from scipy.stats import beta
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConformalGuarantee:
    """
    Probabilistic guarantee parameters and values.

    Attributes:
        m: Calibration set size
        ell: Rank parameter
        epsilon: Miscoverage level
        coverage: δ₁ = 1 - ε
        confidence: δ₂ = 1 - betacdf_{1-ε}(ℓ, m+1-ℓ)
        threshold: R_ℓ value used for inflation
        inflation: Per-dimension inflation σ = τ * R_ℓ
    """
    m: int
    ell: int
    epsilon: float
    coverage: float
    confidence: float
    threshold: float
    inflation: np.ndarray


def compute_confidence(m: int, ell: int, epsilon: float) -> float:
    """
    Compute confidence level δ₂ for given parameters.

    δ₂ = 1 - betacdf_{1-ε}(ℓ, m+1-ℓ)

    Args:
        m: Calibration set size
        ell: Rank parameter (1-indexed, typically m-1 or m)
        epsilon: Miscoverage level

    Returns:
        Confidence level δ₂

    Example:
        >>> compute_confidence(m=8000, ell=7999, epsilon=0.001)
        0.9970...
    """
    return 1 - beta.cdf(1 - epsilon, ell, m + 1 - ell)


def compute_normalization(
    training_errors: np.ndarray,
    tau_star_factor: float = 1e-5
) -> np.ndarray:
    """
    Compute per-dimension normalization factors τ.

    τ[k] = max(τ*, max_j |training_errors[j, k]|)

    Where τ* is a small value to prevent division by zero.

    Args:
        training_errors: Array of shape (t, n) where t is number of training
                         samples and n is output dimension
        tau_star_factor: Factor for computing τ* (default: 1e-5)

    Returns:
        Normalization vector τ of shape (n,)
    """
    # Handle edge case of empty or single-sample training errors
    if training_errors.size == 0:
        raise ValueError("training_errors cannot be empty")

    if training_errors.ndim == 1:
        training_errors = training_errors.reshape(1, -1)

    # Compute τ* to prevent division by zero
    # τ* = tau_star_factor * mean(|training_errors|)
    mean_abs = np.mean(np.abs(training_errors))
    tau_star = tau_star_factor * mean_abs if mean_abs > 0 else 1e-10
    tau_star = max(tau_star, 1e-10)  # Absolute minimum

    # τ[k] = max(τ*, max_j |training_errors[j, k]|)
    max_abs_errors = np.max(np.abs(training_errors), axis=0)  # Shape: (n,)
    tau = np.maximum(tau_star, max_abs_errors)

    return tau


def compute_nonconformity_scores(
    prediction_errors: np.ndarray,
    tau: np.ndarray,
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute nonconformity scores for calibration samples.

    R_i = max_k(|q_i[k] - c[k]| / τ[k])

    Args:
        prediction_errors: Array of shape (m, n) where m is number of
                          calibration samples and n is output dimension
        tau: Normalization vector of shape (n,)
        center: Center vector of shape (n,). If None, uses zero.

    Returns:
        Nonconformity scores of shape (m,)
    """
    if prediction_errors.ndim == 1:
        prediction_errors = prediction_errors.reshape(1, -1)

    m, n = prediction_errors.shape

    if center is None:
        center = np.zeros(n)

    # Compute |q_i[k] - c[k]| / τ[k] for all i, k
    centered_errors = np.abs(prediction_errors - center)  # Shape: (m, n)
    normalized_errors = centered_errors / tau  # Shape: (m, n)

    # R_i = max over k
    scores = np.max(normalized_errors, axis=1)  # Shape: (m,)

    return scores


def compute_threshold(
    scores: np.ndarray,
    ell: int
) -> float:
    """
    Select the ℓ-th smallest nonconformity score as threshold.

    Args:
        scores: Nonconformity scores of shape (m,)
        ell: Rank parameter (1-indexed). ell=m means largest score.

    Returns:
        R_ℓ threshold value
    """
    sorted_scores = np.sort(scores)
    # ell is 1-indexed, so ell-1 gives the ℓ-th smallest
    return sorted_scores[ell - 1]


def compute_inflation(
    tau: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Compute per-dimension inflation.

    σ[k] = τ[k] * R_ℓ

    Args:
        tau: Normalization vector of shape (n,)
        threshold: R_ℓ threshold value

    Returns:
        Inflation vector σ of shape (n,)
    """
    return tau * threshold


def conformal_inference(
    training_errors: np.ndarray,
    calibration_errors: np.ndarray,
    m: int,
    ell: int,
    epsilon: float
) -> ConformalGuarantee:
    """
    Complete conformal inference computation.

    Given training and calibration prediction errors, computes the
    inflation required for the ⟨ε, ℓ, m⟩ guarantee.

    Args:
        training_errors: Training set prediction errors, shape (t, n)
        calibration_errors: Calibration set prediction errors, shape (m, n)
        m: Calibration set size (should match calibration_errors.shape[0])
        ell: Rank parameter
        epsilon: Miscoverage level

    Returns:
        ConformalGuarantee with all computed values

    Example:
        >>> training_errors = compute_errors(training_outputs, surrogate_predictions)
        >>> calib_errors = compute_errors(calib_outputs, surrogate_predictions)
        >>> guarantee = conformal_inference(
        ...     training_errors, calib_errors,
        ...     m=8000, ell=7999, epsilon=0.001
        ... )
        >>> print(f"Confidence: {guarantee.confidence:.4f}")
        >>> bounds = surrogate_bounds +/- guarantee.inflation
    """
    # Ensure 2D arrays
    if training_errors.ndim == 1:
        training_errors = training_errors.reshape(1, -1)
    if calibration_errors.ndim == 1:
        calibration_errors = calibration_errors.reshape(1, -1)

    # Validate inputs
    if calibration_errors.shape[0] != m:
        raise ValueError(
            f"calibration_errors has {calibration_errors.shape[0]} samples, expected m={m}"
        )
    if ell < 1 or ell > m:
        raise ValueError(f"ell must be in [1, m], got ell={ell}, m={m}")
    if not 0 < epsilon < 1:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")

    # Step 1: Compute normalization
    tau = compute_normalization(training_errors)

    # Step 2: Compute nonconformity scores
    scores = compute_nonconformity_scores(calibration_errors, tau)

    # Step 3: Select threshold
    threshold = compute_threshold(scores, ell)

    # Step 4: Compute inflation
    inflation = compute_inflation(tau, threshold)

    # Step 5: Compute guarantee
    coverage = 1 - epsilon
    confidence = compute_confidence(m, ell, epsilon)

    return ConformalGuarantee(
        m=m,
        ell=ell,
        epsilon=epsilon,
        coverage=coverage,
        confidence=confidence,
        threshold=threshold,
        inflation=inflation
    )
