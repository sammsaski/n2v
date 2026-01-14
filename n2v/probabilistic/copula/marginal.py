"""
Marginal CDF classes for copula-based conformal prediction.

This module provides classes for estimating marginal cumulative distribution
functions (CDFs) from data, which are used to transform residuals to
pseudo-observations in the copula space.
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Union


class MarginalCDF(ABC):
    """
    Abstract base class for marginal CDF estimation.

    Marginal CDFs are used to transform residuals to pseudo-observations
    in [0, 1] for copula fitting.
    """

    @abstractmethod
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the CDF at point(s) x.

        Args:
            x: Point(s) at which to evaluate the CDF

        Returns:
            CDF value(s) F(x) in [0, 1]
        """
        pass

    @abstractmethod
    def inverse(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the inverse CDF (quantile function) at u.

        Args:
            u: Probability value(s) in [0, 1]

        Returns:
            Quantile value(s) F^{-1}(u)
        """
        pass

    @abstractmethod
    def log_density(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the log density (log PDF) at point(s) x.

        Args:
            x: Point(s) at which to evaluate the density

        Returns:
            Log density value(s) log(f(x))
        """
        pass


class KernelCDF(MarginalCDF):
    """
    Kernel-smoothed CDF estimator using Gaussian kernel.

    Uses Silverman's rule of thumb for bandwidth selection:
        h = 1.06 * std(data) * n^{-0.2}

    The kernel CDF is defined as:
        F_hat(x) = (1/n) * sum_i Phi((x - x_i) / h)

    where Phi is the standard normal CDF.

    Attributes:
        data: The training data points
        bandwidth: Kernel bandwidth (Silverman's rule)
        n: Number of data points
    """

    def __init__(self, data: np.ndarray, bandwidth: float = None):
        """
        Initialize the kernel CDF estimator.

        Args:
            data: 1D array of training data points
            bandwidth: Kernel bandwidth (if None, uses Silverman's rule)
        """
        self.data = np.asarray(data).flatten()
        self.n = len(self.data)

        if self.n < 2:
            raise ValueError("KernelCDF requires at least 2 data points")

        # Silverman's rule of thumb
        if bandwidth is None:
            std = np.std(self.data, ddof=1)
            # Handle case of zero or near-zero standard deviation
            if std < 1e-10:
                std = 1e-10
            self.bandwidth = 1.06 * std * self.n ** (-0.2)
        else:
            self.bandwidth = bandwidth

        # Precompute bounds for inverse
        self._data_min = np.min(self.data)
        self._data_max = np.max(self.data)
        # Extend bounds for inverse search
        self._search_min = self._data_min - 10 * self.bandwidth
        self._search_max = self._data_max + 10 * self.bandwidth

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the kernel CDF at point(s) x.

        F_hat(x) = (1/n) * sum_i Phi((x - x_i) / h)

        Args:
            x: Point(s) at which to evaluate the CDF

        Returns:
            CDF value(s) in [0, 1]
        """
        x = np.asarray(x)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)

        # Compute (x - data) / h for all combinations
        # Shape: (len(x), n)
        z = (x[:, np.newaxis] - self.data[np.newaxis, :]) / self.bandwidth

        # Apply standard normal CDF and average
        result = np.mean(norm.cdf(z), axis=1)

        if scalar_input:
            return float(result[0])
        return result

    def inverse(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the inverse CDF (quantile function) at u.

        Uses Brent's method to solve F_hat(x) = u.

        Args:
            u: Probability value(s) in [0, 1]

        Returns:
            Quantile value(s) F^{-1}(u)
        """
        u = np.asarray(u)
        scalar_input = u.ndim == 0
        u = np.atleast_1d(u)

        # Clip to avoid numerical issues at boundaries
        u = np.clip(u, 1e-10, 1 - 1e-10)

        result = np.zeros_like(u, dtype=float)

        for i, ui in enumerate(u):
            # Solve F_hat(x) = ui using Brent's method
            try:
                result[i] = brentq(
                    lambda x: self.cdf(x) - ui,
                    self._search_min,
                    self._search_max,
                    xtol=1e-10
                )
            except ValueError:
                # If Brent's method fails, use linear interpolation fallback
                if ui <= 0.5:
                    result[i] = self._search_min
                else:
                    result[i] = self._search_max

        if scalar_input:
            return float(result[0])
        return result

    def log_density(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the log of the kernel density at point(s) x.

        The kernel density is:
            f_hat(x) = (1 / (n * h)) * sum_i phi((x - x_i) / h)

        where phi is the standard normal PDF.

        Args:
            x: Point(s) at which to evaluate the density

        Returns:
            Log density value(s)
        """
        x = np.asarray(x)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)

        # Compute (x - data) / h for all combinations
        # Shape: (len(x), n)
        z = (x[:, np.newaxis] - self.data[np.newaxis, :]) / self.bandwidth

        # Compute kernel density: mean of normal PDFs
        # Use log-sum-exp for numerical stability
        log_phi = norm.logpdf(z)  # Shape: (len(x), n)

        # log(mean(exp(log_phi))) = log(sum(exp(log_phi))) - log(n)
        # Use scipy.special.logsumexp for stability
        from scipy.special import logsumexp
        log_density = logsumexp(log_phi, axis=1) - np.log(self.n) - np.log(self.bandwidth)

        if scalar_input:
            return float(log_density[0])
        return log_density


class EmpiricalCDF(MarginalCDF):
    """
    Empirical CDF estimator.

    The empirical CDF is defined as:
        F_hat(x) = (1/n) * sum_i I(x_i <= x)

    This is a step function that jumps at each data point.

    Attributes:
        data: The sorted training data points
        n: Number of data points
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize the empirical CDF estimator.

        Args:
            data: 1D array of training data points
        """
        self.data = np.sort(np.asarray(data).flatten())
        self.n = len(self.data)

        if self.n < 1:
            raise ValueError("EmpiricalCDF requires at least 1 data point")

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the empirical CDF at point(s) x.

        F_hat(x) = (1/n) * sum_i I(x_i <= x)

        Args:
            x: Point(s) at which to evaluate the CDF

        Returns:
            CDF value(s) in [0, 1]
        """
        x = np.asarray(x)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)

        # Use searchsorted for efficient computation
        # This gives the number of data points <= x
        result = np.searchsorted(self.data, x, side='right') / self.n

        if scalar_input:
            return float(result[0])
        return result

    def inverse(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the inverse CDF (quantile function) at u.

        Returns the smallest x such that F_hat(x) >= u.

        Args:
            u: Probability value(s) in [0, 1]

        Returns:
            Quantile value(s) F^{-1}(u)
        """
        u = np.asarray(u)
        scalar_input = u.ndim == 0
        u = np.atleast_1d(u)

        # Clip to valid range
        u = np.clip(u, 1e-10, 1 - 1e-10)

        # Compute indices: ceil(u * n) - 1, clipped to valid range
        indices = np.clip(
            np.ceil(u * self.n).astype(int) - 1,
            0,
            self.n - 1
        )

        result = self.data[indices]

        if scalar_input:
            return float(result[0])
        return result

    def log_density(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the log density at point(s) x.

        For empirical CDF, we use a simple histogram-like density with
        Silverman bandwidth for smoothing.

        Args:
            x: Point(s) at which to evaluate the density

        Returns:
            Log density value(s)
        """
        x = np.asarray(x)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)

        # Use Silverman bandwidth
        std = np.std(self.data, ddof=1) if self.n > 1 else 1.0
        if std < 1e-10:
            std = 1e-10
        bandwidth = 1.06 * std * self.n ** (-0.2)

        # Compute kernel density estimate
        z = (x[:, np.newaxis] - self.data[np.newaxis, :]) / bandwidth
        log_phi = norm.logpdf(z)

        from scipy.special import logsumexp
        log_density = logsumexp(log_phi, axis=1) - np.log(self.n) - np.log(bandwidth)

        if scalar_input:
            return float(log_density[0])
        return log_density
