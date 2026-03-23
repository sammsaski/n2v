"""
ImageZono set representation for image inputs.

Represents a zonotope in image format (height x width x channels).
Translated from MATLAB NNV ImageZono.m
"""

import numpy as np
from n2v.sets.zono import Zono


class ImageZono(Zono):
    """
    ImageZono class for representing zonotopes in image format.

    Extends Zono with image-specific dimensions.

    Attributes:
        height: Image height
        width: Image width
        num_channels: Number of channels
        (Inherits c, V from Zono)
    """

    def __init__(
        self,
        c: np.ndarray,
        V: np.ndarray,
        height: int,
        width: int,
        num_channels: int,
    ):
        """
        Initialize an ImageZono.

        Args:
            c: Center vector
            V: Generator matrix
            height: Image height
            width: Image width
            num_channels: Number of channels
        """
        super().__init__(c, V)

        self.height = height
        self.width = width
        self.num_channels = num_channels

        # Validate dimensions
        expected_dim = height * width * num_channels
        if self.dim != expected_dim:
            raise ValueError(
                f"Image dimensions {height}x{width}x{num_channels} = {expected_dim} "
                f"don't match Zono dimension {self.dim}"
            )

    def __repr__(self) -> str:
        """Return string representation of the ImageZono."""
        return (
            f"ImageZono(height={self.height}, width={self.width}, "
            f"channels={self.num_channels}, n_generators={self.V.shape[1]})"
        )

    def to_zono(self) -> Zono:
        """Convert ImageZono back to regular Zono."""
        return Zono(self.c, self.V)

    def get_image_shape(self) -> tuple:
        """Get image shape (height, width, channels)."""
        return (self.height, self.width, self.num_channels)

    @classmethod
    def from_bounds(
        cls, lb: np.ndarray, ub: np.ndarray, height: int, width: int, num_channels: int
    ) -> 'ImageZono':
        """
        Create ImageZono from image bounds.

        Args:
            lb: Lower bound image (height, width, channels) or flattened
            ub: Upper bound image (height, width, channels) or flattened
            height: Image height
            width: Image width
            num_channels: Number of channels

        Returns:
            ImageZono object
        """
        # Flatten if needed
        lb = np.asarray(lb).reshape(-1, 1)
        ub = np.asarray(ub).reshape(-1, 1)

        # Create Zono from bounds
        zono = Zono.from_bounds(lb, ub)

        return cls(zono.c, zono.V, height, width, num_channels)

    def get_bounds(self):
        """Get lower and upper bounds for the ImageZono."""
        return super().get_bounds()
