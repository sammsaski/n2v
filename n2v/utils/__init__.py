"""
Utility functions for n2v.

This module provides helper functions for model loading, LP solving,
conversions, verification, falsification, and other utilities.
"""

from n2v.utils.lpsolver import solve_lp, solve_lp_batch
from n2v.utils.model_loader import load_onnx, load_pytorch
from n2v.utils.load_vnnlib import load_vnnlib
from n2v.utils.falsify import falsify
from n2v.utils.model_preprocessing import fuse_batchnorm, strip_final_softmax

# NOTE: ``verify_specification`` (and ``spec_summary``) are intentionally
# NOT re-exported here. ``verify_specification.py`` imports from
# ``n2v.sets`` at module level, and ``n2v.sets.star`` imports from
# ``n2v.utils.lpsolver``. Re-exporting through this ``__init__`` triggers
# a circular import during ``n2v.sets`` initialisation. Callers should
# use the explicit ``from n2v.utils.verify_specification import ...`` path
# (which is what every existing caller in the tree does anyway).

__all__ = [
    "solve_lp",
    "solve_lp_batch",
    "load_onnx",
    "load_pytorch",
    "load_vnnlib",
    "falsify",
    "fuse_batchnorm",
    "strip_final_softmax",
]
