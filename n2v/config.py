"""
Global configuration for N2V (Neural Network Verification).

This module provides global settings for parallelization, LP solvers, and other options.
"""

import os
import multiprocessing
from typing import Optional


class Config:
    """Global configuration for N2V."""

    def __init__(self):
        # Parallelization settings
        self._parallel_lp = False
        self._n_workers = 4
        self._auto_parallel = True  # Automatically use parallel for dim > threshold
        self._parallel_threshold = 10  # Minimum dimension to use parallel

        # LP solver settings
        self._default_lp_solver = 'default'

        # Try to auto-detect optimal settings
        self._detect_optimal_settings()

    def _detect_optimal_settings(self):
        """Auto-detect optimal settings based on system."""
        try:
            cpu_count = multiprocessing.cpu_count()
            # Use half of available cores, min 2, max 8
            self._n_workers = max(2, min(cpu_count // 2, 8))
        except:
            self._n_workers = 4

    @property
    def parallel_lp(self) -> bool:
        """Whether to use parallel LP solving by default."""
        return self._parallel_lp

    @parallel_lp.setter
    def parallel_lp(self, value: bool):
        """Set parallel LP solving."""
        self._parallel_lp = bool(value)

    @property
    def n_workers(self) -> int:
        """Number of parallel workers for LP solving."""
        return self._n_workers

    @n_workers.setter
    def n_workers(self, value: int):
        """Set number of parallel workers."""
        if value < 1:
            raise ValueError("n_workers must be at least 1")
        self._n_workers = value

    @property
    def auto_parallel(self) -> bool:
        """Whether to automatically use parallel for high-dimensional problems."""
        return self._auto_parallel

    @auto_parallel.setter
    def auto_parallel(self, value: bool):
        """Set auto parallel mode."""
        self._auto_parallel = bool(value)

    @property
    def parallel_threshold(self) -> int:
        """Minimum dimension to automatically use parallel LP."""
        return self._parallel_threshold

    @parallel_threshold.setter
    def parallel_threshold(self, value: int):
        """Set parallel threshold."""
        if value < 1:
            raise ValueError("parallel_threshold must be at least 1")
        self._parallel_threshold = value

    @property
    def lp_solver(self) -> str:
        """Default LP solver to use."""
        return self._default_lp_solver

    @lp_solver.setter
    def lp_solver(self, value: str):
        """Set default LP solver."""
        self._default_lp_solver = str(value)

    @property
    def default_lp_solver(self) -> str:
        """Default LP solver to use (alias for lp_solver)."""
        return self._default_lp_solver

    @default_lp_solver.setter
    def default_lp_solver(self, value: str):
        """Set default LP solver (alias for lp_solver)."""
        self._default_lp_solver = str(value)

    def should_use_parallel(self, dim: int) -> bool:
        """
        Determine if parallel LP solving should be used for given dimension.

        Args:
            dim: Problem dimension

        Returns:
            True if parallel should be used
        """
        if self.parallel_lp:
            return True
        if self.auto_parallel and dim >= self.parallel_threshold:
            return True
        return False

    def get_n_workers(self, dim: int) -> int:
        """
        Get optimal number of workers for given dimension.

        Args:
            dim: Problem dimension

        Returns:
            Number of workers to use
        """
        # For small problems, limit workers to avoid overhead
        if dim < 10:
            return min(2, self.n_workers)
        elif dim < 20:
            return min(4, self.n_workers)
        else:
            return self.n_workers

    def reset(self):
        """Reset configuration to defaults."""
        self._parallel_lp = False
        self._auto_parallel = True
        self._parallel_threshold = 10
        self._default_lp_solver = 'default'
        self._detect_optimal_settings()

    def __repr__(self):
        return (f"Config(parallel_lp={self.parallel_lp}, "
                f"n_workers={self.n_workers}, "
                f"auto_parallel={self.auto_parallel}, "
                f"parallel_threshold={self.parallel_threshold}, "
                f"default_lp_solver='{self.default_lp_solver}')")


# Global configuration instance
config = Config()


def set_parallel(enabled = True, n_workers: Optional[int] = None, threshold: Optional[int] = None):
    """
    Enable or disable parallel LP solving globally.

    Args:
        enabled: Whether to enable parallel LP solving (True/False/'auto')
        n_workers: Number of parallel workers (None = auto-detect)
        threshold: Dimension threshold for auto-parallel mode

    Example:
        >>> import n2v
        >>> n2v.set_parallel(True, n_workers=8)
        >>> # Now all get_ranges() calls will use parallel solving

        >>> n2v.set_parallel('auto', threshold=10)
        >>> # Automatically use parallel for dim >= 10
    """
    if enabled == 'auto':
        config.parallel_lp = False
        config.auto_parallel = True
        if threshold is not None:
            config.parallel_threshold = threshold
    else:
        config.parallel_lp = bool(enabled)
        # When setting explicit mode, disable auto
        if not enabled:
            config.auto_parallel = False
        if n_workers is not None:
            config.n_workers = n_workers


def set_lp_solver(solver: str):
    """
    Set default LP solver globally.

    Args:
        solver: Solver name ('default', 'GUROBI', 'MOSEK', 'SCIPY', etc.)

    Example:
        >>> import n2v
        >>> n2v.set_lp_solver('GUROBI')
        >>> # Now all LP solves will use Gurobi
    """
    config.lp_solver = solver


def get_config() -> dict:
    """Get the global configuration as a dictionary."""
    return {
        'parallel_lp': config.parallel_lp,
        'n_workers': config.n_workers,
        'auto_parallel': config.auto_parallel,
        'parallel_threshold': config.parallel_threshold,
        'lp_solver': config.lp_solver
    }
