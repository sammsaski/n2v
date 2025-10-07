"""
HalfSpace class for representing linear constraints.

A HalfSpace represents the constraint: G @ x <= g
where G is a matrix and g is a vector.
"""

import numpy as np
from typing import Union


class HalfSpace:
    """
    HalfSpace class defining G @ x <= g

    Attributes:
        G: Half-space matrix (n x m) where n is number of constraints, m is dimension
        g: Half-space vector (n x 1)
        dim: Dimension of the half-space (m)
    """

    def __init__(self, G: np.ndarray, g: np.ndarray):
        """
        Constructor for HalfSpace.

        Args:
            G: Half-space matrix (n x m)
            g: Half-space vector (n x 1)

        Raises:
            ValueError: If dimensions are inconsistent
        """
        G = np.asarray(G, dtype=np.float32)
        g = np.asarray(g, dtype=np.float32)

        # Ensure g is 2D
        if g.ndim == 1:
            g = g.reshape(-1, 1)

        if G.ndim == 1:
            G = G.reshape(1, -1)

        n1, m1 = G.shape
        n2, m2 = g.shape

        if n1 != n2:
            raise ValueError(
                f'Inconsistent dimension between half-space matrix and half-space vector: '
                f'G has {n1} rows but g has {n2} rows'
            )

        if m2 != 1:
            raise ValueError(f'Half-space vector should have one column, got {m2}')

        self.G = G
        self.g = g
        self.dim = m1

    def contains(self, x: np.ndarray) -> bool:
        """
        Check if the half-space contains point x.

        Args:
            x: Input vector (dim x 1) or (dim,)

        Returns:
            True if G @ x <= g, False otherwise

        Raises:
            ValueError: If dimensions are inconsistent
        """
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n, m = x.shape

        if n != self.dim:
            raise ValueError(
                f'Inconsistent dimension between the vector x and the half-space object: '
                f'x has dimension {n} but half-space has dimension {self.dim}'
            )

        if m != 1:
            raise ValueError(f'Input vector x should have one column, got {m}')

        # Check if G @ x <= g
        result = self.G @ x
        return np.all(result <= self.g + 1e-8)  # Small tolerance for numerical errors

    def __repr__(self) -> str:
        """String representation of HalfSpace."""
        return f"HalfSpace(dim={self.dim}, constraints={self.G.shape[0]})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"HalfSpace: G @ x <= g\n  G shape: {self.G.shape}\n  g shape: {self.g.shape}"
