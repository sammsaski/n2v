"""
Probabilistic Box set representation.

A Box with probabilistic coverage guarantees from conformal inference.
Inherits from Box and can be used anywhere a Box is used.
"""

import numpy as np
from scipy.stats import beta
from typing import Optional, Tuple, TYPE_CHECKING

from n2v.sets.box import Box

if TYPE_CHECKING:
    from n2v.sets.star import Star


class ProbabilisticBox(Box):
    """
    Hyper-rectangle with probabilistic coverage guarantee.

    Represents a set R such that:
        Pr[Pr[f(x) ∈ R] > 1-ε] > δ₂

    Where:
        - ε is the miscoverage level
        - δ₂ is the confidence level
        - The inner probability is over random inputs x
        - The outer probability is over the randomness in constructing R

    Attributes:
        lb: Lower-bound vector (inherited from Box)
        ub: Upper-bound vector (inherited from Box)
        m: Calibration set size
        ell: Rank parameter (ℓ)
        epsilon: Miscoverage level
        coverage: δ₁ = 1 - ε (probability of containing a random output)
        confidence: δ₂ (probability that coverage guarantee holds)

    Example:
        >>> from n2v.sets import ProbabilisticBox
        >>> pbox = ProbabilisticBox(lb, ub, m=8000, ell=7999, epsilon=0.001)
        >>> print(f"Coverage: {pbox.coverage:.4f}")  # 0.999
        >>> print(f"Confidence: {pbox.confidence:.4f}")  # ~0.997
        >>>
        >>> # Works with verify_specification
        >>> from n2v.utils import verify_specification
        >>> result = verify_specification([pbox], property)
    """

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        m: int,
        ell: int,
        epsilon: float
    ):
        """
        Initialize a ProbabilisticBox.

        Args:
            lb: Lower-bound vector (n,) or (n, 1)
            ub: Upper-bound vector (n,) or (n, 1)
            m: Calibration set size (number of samples used)
            ell: Rank parameter (which order statistic, typically m-1)
            epsilon: Miscoverage level (e.g., 0.001 for 99.9% coverage)

        Raises:
            ValueError: If m < 1, ell < 1, ell > m, or epsilon not in (0, 1)
        """
        # Validate guarantee parameters
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if ell < 1 or ell > m:
            raise ValueError(f"ell must be in [1, m], got ell={ell}, m={m}")
        if not 0 < epsilon < 1:
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")

        # Initialize parent Box
        super().__init__(lb, ub)

        # Store guarantee parameters
        self.m = m
        self.ell = ell
        self.epsilon = epsilon

        # Compute derived guarantees
        self.coverage = 1 - epsilon
        self.confidence = self._compute_confidence()

    def _compute_confidence(self) -> float:
        """
        Compute confidence level δ₂ using beta CDF.

        δ₂ = 1 - betacdf_{1-ε}(ℓ, m+1-ℓ)

        This is the probability that the coverage guarantee holds.
        """
        # beta.cdf(x, a, b) computes P(X <= x) for X ~ Beta(a, b)
        # We want 1 - P(X <= 1-ε) where X ~ Beta(ℓ, m+1-ℓ)
        return 1 - beta.cdf(1 - self.epsilon, self.ell, self.m + 1 - self.ell)

    def __repr__(self) -> str:
        """Return string representation with guarantee metadata."""
        return (
            f"ProbabilisticBox(dim={self.dim}, "
            f"coverage={self.coverage:.4f}, confidence={self.confidence:.4f}, "
            f"m={self.m}, ℓ={self.ell}, ε={self.epsilon})"
        )

    # ======================== Set Operations ========================
    # These override Box methods to preserve probabilistic metadata

    def minkowski_sum(self, other: 'Box') -> 'ProbabilisticBox':
        """
        Compute Minkowski sum, preserving probabilistic guarantee.

        When adding a deterministic Box, the guarantee is preserved.
        When adding another ProbabilisticBox, we use conservative combination.

        Args:
            other: Box or ProbabilisticBox of same dimension

        Returns:
            ProbabilisticBox with combined bounds and guarantee
        """
        # Compute Box result
        result = super().minkowski_sum(other)

        if isinstance(other, ProbabilisticBox):
            # Conservative combination: take worse parameters
            # This is sound but may be overly conservative
            new_m = min(self.m, other.m)
            new_ell = min(self.ell, other.ell)
            new_epsilon = max(self.epsilon, other.epsilon)
        else:
            # Deterministic box doesn't affect guarantee
            new_m = self.m
            new_ell = self.ell
            new_epsilon = self.epsilon

        return ProbabilisticBox(
            result.lb, result.ub,
            m=new_m, ell=new_ell, epsilon=new_epsilon
        )

    def affine_map(self, W: np.ndarray, b: Optional[np.ndarray] = None) -> 'ProbabilisticBox':
        """
        Apply affine transformation, preserving probabilistic guarantee.

        Affine maps preserve the probabilistic coverage guarantee since
        if y ∈ R with probability 1-ε, then W@y+b ∈ W@R+b with the same probability.

        Args:
            W: Mapping matrix (m, n)
            b: Mapping vector (m,) or (m, 1), optional

        Returns:
            Transformed ProbabilisticBox
        """
        result = super().affine_map(W, b)
        return ProbabilisticBox(
            result.lb, result.ub,
            m=self.m, ell=self.ell, epsilon=self.epsilon
        )

    def to_star(self) -> 'Star':
        """
        Convert to Star representation.

        WARNING: This loses the probabilistic guarantee metadata.
        The resulting Star is a valid over-approximation but no longer
        carries the coverage/confidence information.

        Consider using to_star_with_metadata() if you need to preserve
        guarantee information.

        Returns:
            Star representation of this box
        """
        import warnings
        warnings.warn(
            "Converting ProbabilisticBox to Star loses guarantee metadata. "
            "The Star is still a valid over-approximation.",
            UserWarning
        )
        return super().to_star()

    # ======================== Guarantee Utilities ========================

    def get_guarantee(self) -> Tuple[float, float]:
        """
        Get the probabilistic guarantee as (coverage, confidence).

        Returns:
            Tuple of (coverage δ₁, confidence δ₂)
        """
        return (self.coverage, self.confidence)

    def get_guarantee_string(self) -> str:
        """
        Get human-readable description of the guarantee.

        Returns:
            String describing the probabilistic guarantee
        """
        return (
            f"With {self.confidence:.2%} confidence, "
            f"at least {self.coverage:.2%} of outputs are contained in this set"
        )

    @staticmethod
    def compute_parameters(
        target_coverage: float,
        target_confidence: float,
        max_m: int = 1000000
    ) -> Tuple[int, int, float]:
        """
        Compute (m, ℓ, ε) to achieve target coverage and confidence.

        Finds the smallest m such that with ℓ = m-1 and ε = 1-target_coverage,
        the confidence level meets or exceeds target_confidence.

        Args:
            target_coverage: Desired coverage level (e.g., 0.999)
            target_confidence: Desired confidence level (e.g., 0.99)
            max_m: Maximum m to try

        Returns:
            Tuple of (m, ℓ, ε) that achieves the targets

        Raises:
            ValueError: If targets cannot be achieved with m <= max_m
        """
        epsilon = 1 - target_coverage

        # Binary search for minimum m
        for m in range(10, max_m + 1):
            ell = m - 1
            confidence = 1 - beta.cdf(1 - epsilon, ell, m + 1 - ell)
            if confidence >= target_confidence:
                return (m, ell, epsilon)

        raise ValueError(
            f"Cannot achieve coverage={target_coverage}, confidence={target_confidence} "
            f"with m <= {max_m}"
        )
