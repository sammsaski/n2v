"""
Tests for set representations: Star, Zono, Box, HalfSpace.
"""

import pytest
import numpy as np
from n2v.sets import Star, Zono, Box, HalfSpace


class TestStar:
    """Tests for Star set."""

    def test_creation(self, simple_star):
        """Test Star creation."""
        assert simple_star.dim == 3
        assert simple_star.nVar == 2
        pytest.assert_star_valid(simple_star)

    def test_from_bounds(self):
        """Test Star creation from bounds."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        assert star.dim == 3
        assert star.nVar == 3  # One per dimension
        pytest.assert_star_valid(star)

    def test_affine_map(self, simple_star):
        """Test affine transformation."""
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0]])
        b = np.array([[0.5], [0.5]])

        result = simple_star.affine_map(W, b)

        assert result.dim == 2
        assert result.nVar == simple_star.nVar
        pytest.assert_star_valid(result)

    def test_intersect_half_space(self, simple_star):
        """Test intersection with half-space."""
        G = np.array([[1.0, 0.0]])
        g = np.array([[0.5]])

        result = simple_star.intersect_half_space(G, g)

        assert result.dim == simple_star.dim
        assert result.C.shape[0] == simple_star.C.shape[0] + 1
        pytest.assert_star_valid(result)

    def test_get_bounds(self, simple_star):
        """Test bounds computation."""
        lb, ub = simple_star.get_bounds()

        assert lb.shape == (simple_star.dim, 1)
        assert ub.shape == (simple_star.dim, 1)
        assert np.all(lb <= ub)

    def test_contains_point(self):
        """Test point containment."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Point inside
        point_in = np.array([[0.5], [0.5]])
        assert star.contains(point_in)

        # Point outside
        point_out = np.array([[1.5], [0.5]])
        assert not star.contains(point_out)

    def test_estimate_ranges(self, simple_star):
        """Test range estimation."""
        simple_star.estimate_ranges()

        assert simple_star.state_lb is not None
        assert simple_star.state_ub is not None
        assert simple_star.state_lb.shape == (simple_star.dim, 1)
        assert np.all(simple_star.state_lb <= simple_star.state_ub)


class TestZono:
    """Tests for Zonotope."""

    def test_creation(self, simple_zono):
        """Test Zono creation."""
        assert simple_zono.dim == 3
        pytest.assert_zono_valid(simple_zono)

    def test_from_bounds(self):
        """Test Zono creation from bounds."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        zono = Zono.from_bounds(lb, ub)

        assert zono.dim == 3
        pytest.assert_zono_valid(zono)

        # Check bounds are preserved
        computed_lb, computed_ub = zono.get_bounds()
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_affine_map(self, simple_zono):
        """Test affine transformation."""
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0]])
        b = np.array([[0.5], [0.5]])

        result = simple_zono.affine_map(W, b)

        assert result.dim == 2
        pytest.assert_zono_valid(result)

    def test_minkowski_sum(self, simple_zono):
        """Test Minkowski sum."""
        zono2 = Zono(
            np.array([[0.1], [0.1], [0.1]]),
            np.array([[0.05, 0.0], [0.0, 0.05], [0.0, 0.0]])
        )

        result = simple_zono.minkowski_sum(zono2)

        assert result.dim == simple_zono.dim
        pytest.assert_zono_valid(result)

    def test_get_bounds(self, simple_zono):
        """Test bounds computation."""
        lb, ub = simple_zono.get_bounds()

        assert lb.shape == (simple_zono.dim, 1)
        assert ub.shape == (simple_zono.dim, 1)
        assert np.all(lb <= ub)

    def test_order_reduction(self):
        """Test order reduction."""
        # Create high-order zonotope
        c = np.zeros((3, 1))
        V = np.random.rand(3, 20)  # 20 generators
        zono = Zono(c, V)

        # Reduce to order 3
        reduced = zono.reduce_order(target_order=3)

        assert reduced.dim == zono.dim
        assert reduced.V.shape[1] <= 3 * zono.dim
        pytest.assert_zono_valid(reduced)


class TestBox:
    """Tests for Box set."""

    def test_creation(self, simple_box):
        """Test Box creation."""
        assert simple_box.dim == 3
        assert simple_box.lb.shape == (3, 1)
        assert simple_box.ub.shape == (3, 1)

    def test_from_bounds(self):
        """Test Box creation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        box = Box(lb, ub)

        assert box.dim == 2
        np.testing.assert_array_equal(box.lb, lb)
        np.testing.assert_array_equal(box.ub, ub)

    def test_invalid_bounds(self):
        """Test that invalid bounds raise error."""
        lb = np.array([[1.0], [0.0]])
        ub = np.array([[0.0], [1.0]])  # ub < lb for first dim

        with pytest.raises(ValueError):
            Box(lb, ub)

    def test_affine_map(self, simple_box):
        """Test affine transformation."""
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0]])
        b = np.array([[0.5], [0.5]])

        result = simple_box.affine_map(W, b)

        assert result.dim == 2
        assert np.all(result.lb <= result.ub)

    def test_get_bounds(self, simple_box):
        """Test bounds computation."""
        lb, ub = simple_box.get_bounds()

        np.testing.assert_array_equal(lb, simple_box.lb)
        np.testing.assert_array_equal(ub, simple_box.ub)

    def test_contains_point(self, simple_box):
        """Test point containment."""
        point_in = np.array([[0.5], [0.5], [0.5]])
        point_out = np.array([[1.5], [0.5], [0.5]])

        assert simple_box.contains(point_in)
        assert not simple_box.contains(point_out)

    def test_intersection(self):
        """Test box intersection."""
        box1 = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))
        box2 = Box(np.array([[0.5], [0.5]]), np.array([[1.5], [1.5]]))

        result = box1.intersect(box2)

        assert result.dim == 2
        np.testing.assert_array_equal(result.lb, np.array([[0.5], [0.5]]))
        np.testing.assert_array_equal(result.ub, np.array([[1.0], [1.0]]))

    def test_union(self):
        """Test box union (overapproximation)."""
        box1 = Box(np.array([[0.0], [0.0]]), np.array([[0.5], [0.5]]))
        box2 = Box(np.array([[0.5], [0.5]]), np.array([[1.0], [1.0]]))

        result = box1.union(box2)

        assert result.dim == 2
        np.testing.assert_array_equal(result.lb, np.array([[0.0], [0.0]]))
        np.testing.assert_array_equal(result.ub, np.array([[1.0], [1.0]]))


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


class TestSetConversions:
    """Test conversions between set types."""

    def test_star_to_box(self, simple_star):
        """Test Star to Box conversion."""
        box = simple_star.to_box()

        assert box.dim == simple_star.dim
        assert np.all(box.lb <= box.ub)

    def test_zono_to_box(self, simple_zono):
        """Test Zono to Box conversion."""
        box = simple_zono.to_box()

        assert box.dim == simple_zono.dim

        # Box should contain zonotope
        zono_lb, zono_ub = simple_zono.get_bounds()
        np.testing.assert_allclose(box.lb, zono_lb, atol=1e-6)
        np.testing.assert_allclose(box.ub, zono_ub, atol=1e-6)

    def test_box_to_zono(self, simple_box):
        """Test Box to Zono conversion."""
        zono = simple_box.to_zono()

        assert zono.dim == simple_box.dim

        # Zono should represent same box
        zono_lb, zono_ub = zono.get_bounds()
        np.testing.assert_allclose(zono_lb, simple_box.lb, atol=1e-6)
        np.testing.assert_allclose(zono_ub, simple_box.ub, atol=1e-6)
