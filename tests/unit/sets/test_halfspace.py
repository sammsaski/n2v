"""Tests for set representations."""

import pytest
import numpy as np
from n2v.sets import Star, Zono, Box, HalfSpace, Hexatope, Octatope

class TestHalfSpace:
    """Tests for HalfSpace."""

    def test_creation_basic(self):
        """Test basic HalfSpace creation."""
        # Create halfspace: x1 <= 5 (in 2D space)
        G = np.array([[1, 0]], dtype=np.float32)
        g = np.array([[5]], dtype=np.float32)

        hs = HalfSpace(G, g)

        assert hs.dim == 2
        assert hs.G.shape == (1, 2)
        assert hs.g.shape == (1, 1)
        np.testing.assert_array_equal(hs.G, G)
        np.testing.assert_array_equal(hs.g, g)

    def test_creation_multiple_constraints(self):
        """Test HalfSpace with multiple constraints."""
        # x1 <= 5, x2 <= 3
        G = np.array([[1, 0], [0, 1]], dtype=np.float32)
        g = np.array([[5], [3]], dtype=np.float32)

        hs = HalfSpace(G, g)

        assert hs.dim == 2
        assert hs.G.shape == (2, 2)
        assert hs.g.shape == (2, 1)

    def test_creation_from_1d(self):
        """Test HalfSpace creation from 1D arrays."""
        G = np.array([1, 0], dtype=np.float32)  # 1D
        g = np.array([5], dtype=np.float32)     # 1D

        hs = HalfSpace(G, g)

        assert hs.dim == 2
        assert hs.G.shape == (1, 2)
        assert hs.g.shape == (1, 1)

    def test_invalid_dimensions(self):
        """Test that inconsistent dimensions raise error."""
        G = np.array([[1, 0]], dtype=np.float32)  # 1 row
        g = np.array([[5], [3]], dtype=np.float32)  # 2 rows

        with pytest.raises(ValueError, match="Inconsistent dimension"):
            HalfSpace(G, g)

    def test_invalid_g_columns(self):
        """Test that g with multiple columns raises error."""
        G = np.array([[1, 0]], dtype=np.float32)
        g = np.array([[5, 3]], dtype=np.float32)  # 2 columns

        with pytest.raises(ValueError, match="should have one column"):
            HalfSpace(G, g)

    def test_contains_point_inside(self):
        """Test contains for point inside halfspace."""
        # x1 <= 5, x2 <= 3
        G = np.array([[1, 0], [0, 1]], dtype=np.float32)
        g = np.array([[5], [3]], dtype=np.float32)
        hs = HalfSpace(G, g)

        # Point clearly inside
        x = np.array([[2], [1]], dtype=np.float32)
        assert hs.contains(x)

        # Point on boundary (within tolerance)
        x_boundary = np.array([[5], [3]], dtype=np.float32)
        assert hs.contains(x_boundary)

    def test_contains_point_outside(self):
        """Test contains for point outside halfspace."""
        # x1 <= 5, x2 <= 3
        G = np.array([[1, 0], [0, 1]], dtype=np.float32)
        g = np.array([[5], [3]], dtype=np.float32)
        hs = HalfSpace(G, g)

        # Point clearly outside (violates first constraint)
        x = np.array([[7], [1]], dtype=np.float32)
        assert not hs.contains(x)

        # Point outside (violates second constraint)
        x2 = np.array([[2], [5]], dtype=np.float32)
        assert not hs.contains(x2)

    def test_contains_1d_input(self):
        """Test contains with 1D input vector."""
        G = np.array([[1, 0]], dtype=np.float32)
        g = np.array([[5]], dtype=np.float32)
        hs = HalfSpace(G, g)

        # 1D input should work
        x = np.array([3, 0], dtype=np.float32)
        assert hs.contains(x)

    def test_contains_invalid_dimension(self):
        """Test that contains raises error for wrong dimension."""
        G = np.array([[1, 0]], dtype=np.float32)
        g = np.array([[5]], dtype=np.float32)
        hs = HalfSpace(G, g)

        # Wrong dimension
        x = np.array([[1], [2], [3]], dtype=np.float32)
        with pytest.raises(ValueError, match="Inconsistent dimension"):
            hs.contains(x)

    def test_repr_str(self):
        """Test string representations."""
        G = np.array([[1, 0]], dtype=np.float32)
        g = np.array([[5]], dtype=np.float32)
        hs = HalfSpace(G, g)

        repr_str = repr(hs)
        assert "HalfSpace" in repr_str
        assert "dim=2" in repr_str

        str_str = str(hs)
        assert "HalfSpace" in str_str
        assert "G @ x <= g" in str_str


