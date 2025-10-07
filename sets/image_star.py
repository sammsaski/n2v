"""
ImageStar set representation for image inputs.

Represents a star set in image format (height x width x channels).
Translated from MATLAB NNV ImageStar.m
"""

import numpy as np
from typing import Optional
from n2v.sets.star import Star


class ImageStar(Star):
    """
    ImageStar class for representing star sets in image format.

    Extends Star with image-specific dimensions and operations.

    Attributes:
        height: Image height
        width: Image width
        num_channels: Number of channels
        (Inherits V, C, d, etc. from Star)
    """

    def __init__(
        self,
        V: np.ndarray,
        C: np.ndarray,
        d: np.ndarray,
        pred_lb: Optional[np.ndarray] = None,
        pred_ub: Optional[np.ndarray] = None,
        height: int = 0,
        width: int = 0,
        num_channels: int = 0,
    ):
        """
        Initialize an ImageStar.

        Args:
            V: Basic matrix
            C: Constraint matrix
            d: Constraint vector
            pred_lb: Predicate lower bounds
            pred_ub: Predicate upper bounds
            height: Image height
            width: Image width
            num_channels: Number of channels
        """
        super().__init__(V, C, d, pred_lb, pred_ub)

        self.height = height
        self.width = width
        self.num_channels = num_channels

        # Validate dimensions
        expected_dim = height * width * num_channels
        if self.dim != expected_dim:
            raise ValueError(
                f"Image dimensions {height}x{width}x{num_channels} = {expected_dim} "
                f"don't match Star dimension {self.dim}"
            )

    def __repr__(self) -> str:
        return (
            f"ImageStar(height={self.height}, width={self.width}, "
            f"channels={self.num_channels}, nVar={self.nVar})"
        )

    def to_star(self) -> Star:
        """Convert ImageStar back to regular Star."""
        return Star(self.V, self.C, self.d, self.predicate_lb, self.predicate_ub)

    def get_image_shape(self) -> tuple:
        """Get image shape (height, width, channels)."""
        return (self.height, self.width, self.num_channels)

    @classmethod
    def from_bounds(
        cls, lb: np.ndarray, ub: np.ndarray, height: int, width: int, num_channels: int
    ) -> 'ImageStar':
        """
        Create ImageStar from image bounds.

        Args:
            lb: Lower bound image (height, width, channels) or flattened
            ub: Upper bound image (height, width, channels) or flattened
            height: Image height
            width: Image width
            num_channels: Number of channels

        Returns:
            ImageStar object
        """
        # Flatten if needed
        lb = np.asarray(lb).reshape(-1, 1)
        ub = np.asarray(ub).reshape(-1, 1)

        # Create Star from bounds
        star = Star.from_bounds(lb, ub)

        return cls(
            star.V, star.C, star.d, star.predicate_lb, star.predicate_ub,
            height, width, num_channels
        )

    def flatten_to_star(self) -> Star:
        """
        Flatten ImageStar to regular Star (for fully-connected layers).

        Returns:
            Star object with flattened representation
        """
        # V is already in flattened form (dim, nVar+1)
        return Star(
            self.V,
            self.C,
            self.d,
            self.predicate_lb,
            self.predicate_ub,
            state_lb=self.state_lb,
            state_ub=self.state_ub,
            outer_zono=self.Z
        )

    def estimate_ranges(self, lp_solver: str = 'default'):
        """
        Estimate lower and upper bounds for all pixels in the ImageStar.

        This is inherited from Star class and will populate state_lb and state_ub.

        Args:
            lp_solver: LP solver option
        """
        # Call parent class method to estimate ranges
        super().estimate_ranges(lp_solver)
