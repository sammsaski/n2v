"""
Probabilistic verification module for n2v.

This module provides model-agnostic probabilistic reachability verification
using conformal inference. It works with any callable model and provides
formal coverage guarantees on the computed reachable sets.

Main entry point:
    verify(model, input_set, ...) -> ProbabilisticBox

Example:
    >>> from n2v.probabilistic import verify
    >>> from n2v.sets import Box
    >>>
    >>> result = verify(
    ...     model=lambda x: my_model(x),
    ...     input_set=Box(lb, ub),
    ...     m=8000,
    ...     epsilon=0.001
    ... )
    >>> print(f"Coverage: {result.coverage}, Confidence: {result.confidence}")
"""

from n2v.probabilistic.verify import verify
from n2v.sets.probabilistic_box import ProbabilisticBox
from n2v.probabilistic.conformal import (
    ConformalGuarantee,
    compute_confidence,
    compute_normalization,
    compute_nonconformity_scores,
    compute_threshold,
    compute_inflation,
    conformal_inference
)

__all__ = [
    'verify',
    'ProbabilisticBox',
    'ConformalGuarantee',
    'compute_confidence',
    'compute_normalization',
    'compute_nonconformity_scores',
    'compute_threshold',
    'compute_inflation',
    'conformal_inference'
]
