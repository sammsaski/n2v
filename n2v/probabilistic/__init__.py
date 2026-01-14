"""
Probabilistic verification module for n2v.

This module provides model-agnostic probabilistic reachability verification
using conformal inference. It works with any callable model and provides
formal coverage guarantees on the computed reachable sets.

Main entry point:
    verify(model, input_set, ...) -> ProbabilisticBox or CopulaRegion

Example:
    >>> from n2v.probabilistic import verify
    >>> from n2v.sets import Box
    >>>
    >>> # Hyperrectangle method (default)
    >>> result = verify(
    ...     model=lambda x: my_model(x),
    ...     input_set=Box(lb, ub),
    ...     m=8000,
    ...     epsilon=0.001
    ... )
    >>> print(f"Coverage: {result.coverage}, Confidence: {result.confidence}")
    >>>
    >>> # Copula method (tighter for correlated outputs)
    >>> result = verify(
    ...     model=lambda x: my_model(x),
    ...     input_set=Box(lb, ub),
    ...     surrogate='copula',
    ...     m=8000,
    ...     epsilon=0.001
    ... )
    >>> result.contains(y)        # Membership test
    >>> result.volume_ratio()     # Compare to bounding box
"""

from n2v.probabilistic.verify import verify
from n2v.sets.probabilistic_box import ProbabilisticBox
from n2v.sets.copula_region import CopulaRegion
from n2v.probabilistic.conformal import (
    ConformalGuarantee,
    compute_confidence,
    compute_normalization,
    compute_nonconformity_scores,
    compute_threshold,
    compute_inflation,
    conformal_inference
)
from n2v.probabilistic.copula import (
    CopulaConformalPredictor,
    GaussianCopula,
    KernelCDF,
    EmpiricalCDF,
    MarginalCDF
)

__all__ = [
    # Main entry point
    'verify',
    # Set types
    'ProbabilisticBox',
    'CopulaRegion',
    # Conformal inference
    'ConformalGuarantee',
    'compute_confidence',
    'compute_normalization',
    'compute_nonconformity_scores',
    'compute_threshold',
    'compute_inflation',
    'conformal_inference',
    # Copula components
    'CopulaConformalPredictor',
    'GaussianCopula',
    'KernelCDF',
    'EmpiricalCDF',
    'MarginalCDF',
]
