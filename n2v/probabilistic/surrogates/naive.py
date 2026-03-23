"""
Naive surrogate: uses center of training outputs.

This is the simplest surrogate method, which approximates the reachset
as the center of the training outputs. Produces axis-aligned hyper-rectangular
bounds but can be overly conservative in high dimensions.
"""

import numpy as np
from typing import Tuple

from n2v.probabilistic.surrogates.base import Surrogate


class NaiveSurrogate(Surrogate):
    """
    Naive surrogate using center of training outputs.

    The surrogate reachset is a single point (the mean), so:
    - predict() always returns the center
    - get_bounds() returns (center, center) before inflation

    This is equivalent to directly applying conformal inference
    to the neural network outputs without any surrogate.

    Example:
        >>> surrogate = NaiveSurrogate()
        >>> surrogate.fit(training_outputs)  # Shape: (t, n)
        >>>
        >>> # All predictions return the center
        >>> projections = surrogate.predict(calibration_outputs)
        >>> errors = calibration_outputs - projections
        >>>
        >>> # Bounds before inflation are just the center
        >>> lb, ub = surrogate.get_bounds()
        >>> assert np.allclose(lb, ub)
    """

    def __init__(self):
        """Initialize NaiveSurrogate with empty state."""
        self.center = None
        self.n_dim = None
        self._is_fitted = False

    def fit(self, training_outputs: np.ndarray) -> None:
        """
        Fit by computing mean of training outputs.

        Args:
            training_outputs: Array of shape (t, n)
        """
        if training_outputs.ndim == 1:
            training_outputs = training_outputs.reshape(1, -1)

        self.center = np.mean(training_outputs, axis=0)  # Shape: (n,)
        self.n_dim = self.center.shape[0]
        self._is_fitted = True

    def predict(self, outputs: np.ndarray) -> np.ndarray:
        """
        Return center for all outputs (naive prediction).

        Args:
            outputs: Array of shape (m, n)

        Returns:
            Array of shape (m, n) where each row is the center
        """
        if not self._is_fitted:
            raise RuntimeError("Surrogate must be fitted before predicting")

        if outputs.ndim == 1:
            outputs = outputs.reshape(1, -1)

        m = outputs.shape[0]
        return np.tile(self.center, (m, 1))  # Shape: (m, n)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return center as both lower and upper bound.

        The actual bounds come from inflation, not from the surrogate itself.

        Returns:
            Tuple of (center, center), each of shape (n,)
        """
        if not self._is_fitted:
            raise RuntimeError("Surrogate must be fitted before getting bounds")

        return (self.center.copy(), self.center.copy())
