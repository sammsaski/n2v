"""Shared utilities for the FlowConformal benchmarks.

These are helpers that are benchmark-specific (wrappers over n2v reach,
output-space bbox construction tuned for these experiments). Truly
library-worthy code lives under ``n2v/`` instead.
"""

from examples.FlowConformal.utils.reach import compute_exact_reach

__all__ = ['compute_exact_reach']
