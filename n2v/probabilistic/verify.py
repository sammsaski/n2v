"""
Model-agnostic probabilistic verification.

This module provides the main entry point for probabilistic verification,
which works with any callable model (PyTorch, TensorFlow, ONNX, APIs, etc.).
"""

import logging
import numpy as np
from typing import Callable, Optional, Tuple, Union

from n2v.sets import Box
from n2v.sets.probabilistic_box import ProbabilisticBox
from n2v.probabilistic.conformal import conformal_inference
from n2v.probabilistic.surrogates.naive import NaiveSurrogate
from n2v.probabilistic.surrogates.clipping_block import (
    ClippingBlockSurrogate,
    BatchedClippingBlockSurrogate
)

logger = logging.getLogger(__name__)


def _inverse_transform_bounds(pca: object, lb_reduced: np.ndarray, ub_reduced: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map reduced-space box bounds to original space using interval arithmetic.

    For the linear mapping y[k] = mean[k] + Σ_j x[j] * A[j,k]:
        lb[k] = mean[k] + Σ_j min(lb_reduced[j] * A[j,k], ub_reduced[j] * A[j,k])
        ub[k] = mean[k] + Σ_j max(lb_reduced[j] * A[j,k], ub_reduced[j] * A[j,k])

    Args:
        pca: Fitted DeflationPCA with components_ (N, n) and mean_ (n,)
        lb_reduced: Lower bounds in reduced space, shape (N,)
        ub_reduced: Upper bounds in reduced space, shape (N,)

    Returns:
        Tuple of (lb_original, ub_original), each of shape (n,)
    """
    A = pca.components_  # Shape: (N, n)

    products1 = lb_reduced[:, np.newaxis] * A  # (N, n)
    products2 = ub_reduced[:, np.newaxis] * A  # (N, n)

    mins = np.minimum(products1, products2)  # (N, n)
    maxs = np.maximum(products1, products2)  # (N, n)

    lb = pca.mean_ + np.sum(mins, axis=0)
    ub = pca.mean_ + np.sum(maxs, axis=0)

    return lb, ub


def verify(
    model: Callable[[np.ndarray], np.ndarray],
    input_set: Box,
    m: int = 8000,
    ell: Optional[int] = None,
    epsilon: float = 0.001,
    surrogate: str = 'clipping_block',
    training_samples: Optional[int] = None,
    pca_components: Optional[int] = None,
    batch_size: int = 100,
    seed: Optional[int] = None,
    verbose: bool = False
) -> ProbabilisticBox:
    """
    Model-agnostic probabilistic reachability verification.

    Computes a reachable set with probabilistic coverage guarantees using
    conformal inference. Works with ANY model that can be called as y = model(x).

    The guarantee is:
        Pr[Pr[f(x) ∈ R] > 1-ε] > δ₂

    Where:
        - ε is the miscoverage level (probability output falls outside R)
        - δ₂ is the confidence that this guarantee holds

    Args:
        model: Any callable that maps input -> output.
               Should accept inputs of shape (batch_size, *input_dims)
               and return outputs of shape (batch_size, *output_dims).
               Examples:
               - PyTorch: lambda x: model(torch.tensor(x)).numpy()
               - TensorFlow: lambda x: model.predict(x)
               - ONNX: lambda x: session.run(None, {'input': x})[0]
               - API: lambda x: requests.post(url, json=x.tolist()).json()

        input_set: Box defining the input region to verify over.
                   Samples are drawn uniformly from this region.

        m: Calibration set size. Larger m gives tighter confidence.
           Typical values: 1000-100000. Default: 8000.

        ell: Rank parameter. The ℓ-th smallest nonconformity score is used.
             Default: m - 1 (second largest score).
             Using m gives the largest score (more conservative).

        epsilon: Miscoverage level. The guarantee is that at least (1-ε)
                 of outputs are contained in the reachset.
                 Typical values: 0.001-0.01. Default: 0.001.

        surrogate: Surrogate method to use:
                   - 'naive': Simple center-based surrogate. Fast but may be
                              conservative in high dimensions.
                   - 'clipping_block': Projects onto convex hull of training
                                       outputs. Tighter bounds. Default.

        training_samples: Number of samples for surrogate construction.
                         Only used for 'clipping_block'. Default: m // 2.

        pca_components: Number of PCA components for dimensionality reduction.
                       If None, no reduction is performed.
                       Use when output dimension is very large (e.g., > 10000).

        batch_size: Batch size for model inference. Default: 100.

        seed: Random seed for reproducibility. Default: None.

        verbose: Print progress. Default: False.

    Returns:
        ProbabilisticBox containing:
        - Lower and upper bounds on the reachable set
        - Guarantee parameters (m, ℓ, ε)
        - Computed coverage and confidence levels

    Example:
        >>> import torch
        >>> from n2v.probabilistic import verify
        >>> from n2v.sets import Box
        >>>
        >>> # Load any model
        >>> model = torch.load('model.pt')
        >>> model.eval()
        >>> model_fn = lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        >>>
        >>> # Define input region (e.g., L∞ perturbation around an image)
        >>> image = np.random.rand(784)  # Flattened 28x28
        >>> epsilon_input = 0.1
        >>> input_set = Box(image - epsilon_input, image + epsilon_input)
        >>>
        >>> # Verify
        >>> result = verify(
        ...     model=model_fn,
        ...     input_set=input_set,
        ...     m=8000,
        ...     epsilon=0.001,
        ...     verbose=True
        ... )
        >>>
        >>> print(f"Coverage: {result.coverage:.4f}")
        >>> print(f"Confidence: {result.confidence:.4f}")
        >>> print(result.get_guarantee_string())

    Notes:
        - The input_set Box should have bounds that represent the full
          perturbation region you want to verify over.
        - For semantic segmentation, you may want to use pca_components
          to reduce the very high-dimensional output space.
        - The 'clipping_block' surrogate is generally better than 'naive'
          but requires more computation (LP solving per calibration sample).
    """

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Set defaults
    if ell is None:
        ell = m - 1
    if training_samples is None:
        training_samples = m // 2

    # Validate inputs
    if not isinstance(input_set, Box):
        raise TypeError(f"input_set must be a Box, got {type(input_set)}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if ell < 1 or ell > m:
        raise ValueError(f"ell must be in [1, m], got ell={ell}, m={m}")
    if not 0 < epsilon < 1:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    if surrogate not in ('naive', 'clipping_block'):
        raise ValueError(f"surrogate must be 'naive' or 'clipping_block', got {surrogate}")

    input_dim = input_set.dim

    if verbose:
        logger.info("Probabilistic Verification")
        logger.debug(f"  Input dimension: {input_dim}")
        logger.info(f"  Calibration size m: {m}")
        logger.info(f"  Rank ell: {ell}")
        logger.info(f"  Miscoverage epsilon: {epsilon}")
        logger.info(f"  Surrogate: {surrogate}")

    # =========================================
    # Step 1: Generate training samples
    # =========================================
    if verbose:
        logger.info(f"Step 1: Generating {training_samples} training samples...")

    training_inputs = _sample_from_box(input_set, training_samples)  # Shape: (t, input_dim)
    training_outputs = _batched_inference(model, training_inputs, batch_size)  # Shape: (t, output_dim)

    output_dim = training_outputs.shape[1]

    if verbose:
        logger.debug(f"  Output dimension: {output_dim}")

    # =========================================
    # Step 2: Dimensionality reduction (optional)
    # =========================================
    pca = None
    if pca_components is not None and pca_components < output_dim:
        if verbose:
            logger.info(f"Step 2: Reducing dimension {output_dim} -> {pca_components} via PCA...")

        # Import DeflationPCA only if needed
        from n2v.probabilistic.dimensionality.deflation_pca import DeflationPCA
        pca = DeflationPCA(n_components=pca_components, verbose=verbose)
        training_outputs_reduced = pca.fit_transform(training_outputs)
        output_dim_reduced = pca_components
    else:
        training_outputs_reduced = training_outputs
        output_dim_reduced = output_dim

    # =========================================
    # Step 3: Fit surrogate
    # =========================================
    if verbose:
        logger.info(f"Step 3: Fitting {surrogate} surrogate...")

    if surrogate == 'naive':
        surr = NaiveSurrogate()
    else:
        surr = BatchedClippingBlockSurrogate(batch_size=1000, verbose=verbose)

    surr.fit(training_outputs_reduced)
    surrogate_lb, surrogate_ub = surr.get_bounds()

    # Compute training errors for normalization.
    # For clipping block, training outputs are the convex hull vertices,
    # so predict() returns them unchanged — training errors are zero.
    # Skip the expensive LP projections in that case.
    if surrogate == 'clipping_block':
        training_errors = np.zeros_like(training_outputs_reduced)
    else:
        training_projections = surr.predict(training_outputs_reduced)
        training_errors = training_outputs_reduced - training_projections

    if verbose:
        logger.debug("  Surrogate bounds computed")

    # =========================================
    # Step 4: Generate calibration samples
    # =========================================
    if verbose:
        logger.info(f"Step 4: Generating {m} calibration samples...")

    calibration_inputs = _sample_from_box(input_set, m)  # Shape: (m, input_dim)
    calibration_outputs = _batched_inference(model, calibration_inputs, batch_size)  # Shape: (m, output_dim)

    # Reduce if PCA was used
    if pca is not None:
        calibration_outputs_reduced = pca.transform(calibration_outputs)
    else:
        calibration_outputs_reduced = calibration_outputs

    # =========================================
    # Step 5: Compute calibration errors
    # =========================================
    if verbose:
        logger.info("Step 5: Computing calibration errors...")

    calibration_projections = surr.predict(calibration_outputs_reduced)
    calibration_errors = calibration_outputs_reduced - calibration_projections

    # =========================================
    # Step 6: Conformal inference
    # =========================================
    if verbose:
        logger.info("Step 6: Running conformal inference...")

    guarantee = conformal_inference(
        training_errors=training_errors,
        calibration_errors=calibration_errors,
        m=m,
        ell=ell,
        epsilon=epsilon
    )

    if verbose:
        logger.info(f"  Coverage: {guarantee.coverage:.4f}")
        logger.info(f"  Confidence: {guarantee.confidence:.4f}")
        logger.info(f"  Threshold R_ell: {guarantee.threshold:.4f}")

    # =========================================
    # Step 7: Compute final bounds
    # =========================================
    if verbose:
        logger.info("Step 7: Computing final bounds...")

    final_lb_reduced = surrogate_lb - guarantee.inflation
    final_ub_reduced = surrogate_ub + guarantee.inflation

    # Transform back to full dimension if PCA was used
    if pca is not None:
        final_lb, final_ub = _inverse_transform_bounds(
            pca, final_lb_reduced, final_ub_reduced
        )
    else:
        final_lb = final_lb_reduced
        final_ub = final_ub_reduced

    # =========================================
    # Step 8: Create ProbabilisticBox
    # =========================================
    result = ProbabilisticBox(
        lb=final_lb,
        ub=final_ub,
        m=m,
        ell=ell,
        epsilon=epsilon
    )

    if verbose:
        logger.info(f"Result: {result}")
        logger.info(result.get_guarantee_string())

    return result


def _sample_from_box(box: Box, n_samples: int) -> np.ndarray:
    """
    Sample uniformly from a Box.

    Args:
        box: Box to sample from
        n_samples: Number of samples

    Returns:
        Samples of shape (n_samples, dim)
    """
    lb = box.lb.flatten()
    ub = box.ub.flatten()
    dim = box.dim

    samples = np.random.uniform(lb, ub, size=(n_samples, dim))
    return samples.astype(np.float32)


def _batched_inference(
    model: Callable,
    inputs: np.ndarray,
    batch_size: int
) -> np.ndarray:
    """
    Run model inference in batches.

    Args:
        model: Model callable
        inputs: Input array of shape (n, input_dim)
        batch_size: Batch size

    Returns:
        Outputs of shape (n, output_dim)
    """
    n = inputs.shape[0]
    outputs = []

    for i in range(0, n, batch_size):
        batch = inputs[i:i+batch_size]
        batch_output = model(batch)

        # Flatten if needed (e.g., for images)
        if batch_output.ndim > 2:
            batch_output = batch_output.reshape(batch_output.shape[0], -1)

        outputs.append(batch_output)

    return np.vstack(outputs)
