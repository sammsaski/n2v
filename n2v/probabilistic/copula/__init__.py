"""
Copula-based conformal prediction for probabilistic verification.

This module implements copula-based conformal inference which produces
tighter prediction regions by modeling output correlations.
"""

from .marginal import MarginalCDF, KernelCDF, EmpiricalCDF
from .gaussian import GaussianCopula
from .predictor import CopulaConformalPredictor

__all__ = [
    'MarginalCDF',
    'KernelCDF',
    'EmpiricalCDF',
    'GaussianCopula',
    'CopulaConformalPredictor',
]
