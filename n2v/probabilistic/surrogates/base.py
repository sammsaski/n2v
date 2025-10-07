"""
Abstract base class for surrogate reachset methods.

Surrogates provide a deterministic approximation of the reachable set,
which is then inflated using conformal inference.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class Surrogate(ABC):
    """
    Abstract base class for surrogate reachset methods.

    A surrogate takes training outputs and provides:
    1. A method to "project" or "predict" outputs (for computing errors)
    2. Bounds on the surrogate reachset itself

    The final probabilistic reachset is:
        surrogate_bounds +/- conformal_inflation
    """

    @abstractmethod
    def fit(self, training_outputs: np.ndarray) -> None:
        """
        Fit the surrogate to training outputs.

        Args:
            training_outputs: Array of shape (t, n) where t is number of
                            training samples and n is output dimension
        """
        pass

    @abstractmethod
    def predict(self, outputs: np.ndarray) -> np.ndarray:
        """
        Project/predict outputs using the surrogate.

        For naive surrogate: returns the center (same for all inputs)
        For clipping block: projects each output onto convex hull

        Args:
            outputs: Array of shape (m, n) to project

        Returns:
            Projected outputs of shape (m, n)
        """
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds of the surrogate reachset.

        Returns:
            Tuple of (lower_bounds, upper_bounds), each of shape (n,)
        """
        pass

    def compute_errors(
        self,
        outputs: np.ndarray,
        projections: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute prediction errors.

        Error = actual_output - surrogate_prediction

        Args:
            outputs: Actual outputs of shape (m, n)
            projections: Pre-computed projections. If None, calls predict().

        Returns:
            Prediction errors of shape (m, n)
        """
        if projections is None:
            projections = self.predict(outputs)
        return outputs - projections
