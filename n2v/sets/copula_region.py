"""
Copula-based prediction region for probabilistic verification.

A non-rectangular set representation that captures correlation structure
between output dimensions, providing tighter bounds than hyperrectangles.
"""

import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

from n2v.sets.probabilistic_box import ProbabilisticBox

if TYPE_CHECKING:
    from n2v.probabilistic.copula.predictor import CopulaConformalPredictor


class CopulaRegion:
    """
    Copula-based prediction region with probabilistic coverage guarantee.

    Unlike ProbabilisticBox which is axis-aligned, CopulaRegion uses
    copula density contours to define the prediction region, producing
    tighter bounds for correlated outputs.

    The region is defined as:
        R = {y : -log(c_Sigma(F_1(r_1), ..., F_d(r_d))) <= tau}

    where c_Sigma is the Gaussian copula density, F_j are marginal CDFs,
    r = y - center, and tau is the calibrated threshold.

    Attributes:
        predictor: The underlying CopulaConformalPredictor
        d: Output dimensionality
        m: Calibration set size
        ell: Rank parameter
        epsilon: Miscoverage level
        coverage: delta_1 = 1 - epsilon
        confidence: delta_2

    Example:
        >>> from n2v.probabilistic import verify
        >>> result = verify(model, input_set, surrogate='copula')
        >>> result.contains(y)            # Membership test
        >>> result.sample(1000)           # Sample from region
        >>> result.volume_ratio()         # Compare to hyperrectangle
        >>> result.to_box()               # Convert to ProbabilisticBox
    """

    def __init__(self, predictor: 'CopulaConformalPredictor'):
        """
        Initialize a CopulaRegion from a calibrated predictor.

        Args:
            predictor: A calibrated CopulaConformalPredictor

        Raises:
            ValueError: If predictor is not calibrated
        """
        if predictor.center is None:
            raise ValueError("Predictor must be calibrated before creating CopulaRegion")

        self.predictor = predictor
        self.d = predictor.d
        self.m = predictor.m
        self.ell = predictor.ell
        self.epsilon = predictor.epsilon
        self.coverage = predictor.coverage
        self.confidence = predictor.confidence

        # Cached bounding box (computed lazily)
        self._bounding_box: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __repr__(self) -> str:
        return (
            f"CopulaRegion(dim={self.d}, "
            f"coverage={self.coverage:.4f}, confidence={self.confidence:.4f}, "
            f"m={self.m}, ell={self.ell}, epsilon={self.epsilon})"
        )

    # ======================== Core Operations ========================

    def contains(self, y: np.ndarray) -> bool:
        """
        Test if a point is in the prediction region.

        Args:
            y: Point to test, shape (d,) or (d, 1)

        Returns:
            True if y is in the region
        """
        return self.predictor.contains(y)

    def contains_batch(self, Y: np.ndarray) -> np.ndarray:
        """
        Test if multiple points are in the prediction region.

        Args:
            Y: Points to test, shape (n, d)

        Returns:
            Boolean array, shape (n,)
        """
        return self.predictor.contains_batch(Y)

    def sample(
        self,
        n: int,
        max_attempts: int = 100000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample points uniformly from the prediction region.

        Uses rejection sampling from the copula distribution.

        Args:
            n: Number of samples to generate
            max_attempts: Maximum attempts before giving up
            seed: Random seed for reproducibility

        Returns:
            Samples, shape (n, d)
        """
        return self.predictor.sample(n, max_attempts=max_attempts, seed=seed)

    def score(self, y: np.ndarray) -> float:
        """
        Compute the conformity score for a point.

        Higher scores indicate points that are more "unusual" or
        further from the center of the distribution.

        Args:
            y: Point to score, shape (d,)

        Returns:
            Conformity score (threshold is at self.predictor.threshold)
        """
        return self.predictor.score(y)

    # ======================== Conversion Operations ========================

    def get_bounding_box(
        self,
        n_samples: int = 10000,
        seed: Optional[int] = None,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the axis-aligned bounding box of the region.

        Uses Monte Carlo sampling to estimate bounds.

        Args:
            n_samples: Number of samples for estimation
            seed: Random seed
            use_cache: If True, reuse cached result if available

        Returns:
            Tuple of (lower_bounds, upper_bounds), each shape (d,)
        """
        if use_cache and self._bounding_box is not None:
            return self._bounding_box

        lb, ub = self.predictor.get_bounding_box(n_samples, seed)
        self._bounding_box = (lb, ub)
        return (lb, ub)

    def to_box(
        self,
        n_samples: int = 10000,
        seed: Optional[int] = None
    ) -> ProbabilisticBox:
        """
        Convert to a ProbabilisticBox (axis-aligned bounding box).

        The resulting box is an over-approximation of the copula region
        and preserves the probabilistic guarantee.

        Args:
            n_samples: Number of samples for bound estimation
            seed: Random seed

        Returns:
            ProbabilisticBox containing this region
        """
        lb, ub = self.get_bounding_box(n_samples, seed, use_cache=False)

        return ProbabilisticBox(
            lb=lb,
            ub=ub,
            m=self.m,
            ell=self.ell,
            epsilon=self.epsilon
        )

    # Alias for compatibility
    to_hyperrectangle = to_box

    # ======================== Volume Analysis ========================

    def volume_ratio(
        self,
        n_samples: int = 50000,
        seed: Optional[int] = None
    ) -> float:
        """
        Estimate the volume ratio compared to bounding hyperrectangle.

        volume_ratio = Volume(CopulaRegion) / Volume(BoundingBox)

        A ratio < 1 indicates the copula region is tighter than the box.

        Args:
            n_samples: Number of samples for Monte Carlo estimation
            seed: Random seed

        Returns:
            Volume ratio in [0, 1]
        """
        rng = np.random.default_rng(seed)

        # Get bounding box
        lb, ub = self.get_bounding_box(seed=seed)

        # Sample uniformly from bounding box
        samples = rng.uniform(lb, ub, size=(n_samples, self.d))

        # Count how many are in the copula region
        in_region = self.contains_batch(samples)
        ratio = np.mean(in_region)

        return float(ratio)

    def volume_reduction(
        self,
        n_samples: int = 50000,
        seed: Optional[int] = None
    ) -> float:
        """
        Estimate the volume reduction compared to bounding hyperrectangle.

        volume_reduction = 1 - volume_ratio = 1 - V(Copula)/V(Box)

        A higher value indicates more volume savings.

        Args:
            n_samples: Number of samples for Monte Carlo estimation
            seed: Random seed

        Returns:
            Volume reduction in [0, 1]
        """
        return 1 - self.volume_ratio(n_samples, seed)

    # ======================== Guarantee Utilities ========================

    def get_guarantee(self) -> Tuple[float, float]:
        """
        Get the probabilistic guarantee as (coverage, confidence).

        Returns:
            Tuple of (coverage delta_1, confidence delta_2)
        """
        return (self.coverage, self.confidence)

    def get_guarantee_string(self) -> str:
        """
        Get human-readable description of the guarantee.

        Returns:
            String describing the probabilistic guarantee
        """
        return self.predictor.get_guarantee_string()

    # ======================== Properties ========================

    @property
    def center(self) -> np.ndarray:
        """Get the center of the prediction region."""
        return self.predictor.center.copy()

    @property
    def threshold(self) -> float:
        """Get the conformity score threshold."""
        return self.predictor.threshold

    @property
    def correlation(self) -> np.ndarray:
        """Get the fitted correlation matrix."""
        return self.predictor.copula.get_correlation()

    @property
    def dim(self) -> int:
        """Get the dimensionality."""
        return self.d

    # ======================== Ranges ========================

    def get_ranges(
        self,
        n_samples: int = 10000,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the per-dimension ranges (alias for get_bounding_box).

        This method is provided for compatibility with other set types.

        Args:
            n_samples: Number of samples for estimation
            seed: Random seed

        Returns:
            Tuple of (lower_bounds, upper_bounds), each shape (d, 1)
        """
        lb, ub = self.get_bounding_box(n_samples, seed)
        return lb.reshape(-1, 1), ub.reshape(-1, 1)

    def estimate_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast range estimation (uses cached bounding box if available).

        Returns:
            Tuple of (lower_bounds, upper_bounds), each shape (d, 1)
        """
        lb, ub = self.get_bounding_box(n_samples=5000, use_cache=True)
        return lb.reshape(-1, 1), ub.reshape(-1, 1)
