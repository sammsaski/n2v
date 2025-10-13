"""
Tests for set representations: Star, Zono, Box, HalfSpace, Hexatope, Octatope.
"""

import pytest
import numpy as np
from n2v.sets import Star, Zono, Box, HalfSpace, Hexatope, Octatope


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
        G = np.array([[1.0, 0.0, 0.0]])
        g = np.array([[0.5]])

        result = simple_star.intersect_half_space(G, g)

        assert result.dim == simple_star.dim
        assert result.C.shape[0] == simple_star.C.shape[0] + 1
        pytest.assert_star_valid(result)

    def test_get_bounds(self, simple_star):
        """Test bounds computation."""
        lb, ub = simple_star.get_ranges()

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

    @pytest.mark.skip(reason="Zono.reduce_order() not implemented yet")
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
        lb, ub = simple_box.get_range()

        np.testing.assert_array_equal(lb, simple_box.lb)
        np.testing.assert_array_equal(ub, simple_box.ub)

    @pytest.mark.skip(reason="Box.contains() not implemented yet")
    def test_contains_point(self, simple_box):
        """Test point containment."""
        point_in = np.array([[0.5], [0.5], [0.5]])
        point_out = np.array([[1.5], [0.5], [0.5]])

        assert simple_box.contains(point_in)
        assert not simple_box.contains(point_out)

    @pytest.mark.skip(reason="Box.intersect() not implemented yet")
    def test_intersection(self):
        """Test box intersection."""
        box1 = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))
        box2 = Box(np.array([[0.5], [0.5]]), np.array([[1.5], [1.5]]))

        result = box1.intersect(box2)

        assert result.dim == 2
        np.testing.assert_array_equal(result.lb, np.array([[0.5], [0.5]]))
        np.testing.assert_array_equal(result.ub, np.array([[1.0], [1.0]]))

    @pytest.mark.skip(reason="Box.union() not implemented yet")
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
        box = simple_star.get_box()

        assert box.dim == simple_star.dim
        assert np.all(box.lb <= box.ub)

    def test_zono_to_box(self, simple_zono):
        """Test Zono to Box conversion."""
        box = simple_zono.get_box()

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


class TestHexatope:
    """Tests for Hexatope set."""

    def test_creation(self, simple_hexatope):
        """Test Hexatope creation."""
        assert simple_hexatope.dim == 3
        assert simple_hexatope.nVar == 3
        pytest.assert_hexatope_valid(simple_hexatope)

    def test_from_bounds(self):
        """Test Hexatope creation from bounds."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        assert hexatope.dim == 3
        assert hexatope.nVar == 3
        pytest.assert_hexatope_valid(hexatope)

        # Check bounds are preserved
        assert hexatope.state_lb is not None
        assert hexatope.state_ub is not None
        np.testing.assert_allclose(hexatope.state_lb, lb, atol=1e-6)
        np.testing.assert_allclose(hexatope.state_ub, ub, atol=1e-6)

    def test_affine_map(self, simple_hexatope):
        """Test affine transformation."""
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0]])
        b = np.array([[0.5], [0.5]])

        result = simple_hexatope.affine_map(W, b)

        assert result.dim == 2
        assert result.nVar == simple_hexatope.nVar
        pytest.assert_hexatope_valid(result)

    def test_estimate_ranges(self, simple_hexatope):
        """Test range estimation."""
        lb, ub = simple_hexatope.estimate_ranges()

        assert lb.shape == (simple_hexatope.dim, 1)
        assert ub.shape == (simple_hexatope.dim, 1)
        assert np.all(lb <= ub)

        # Check that state bounds are updated
        assert simple_hexatope.state_lb is not None
        assert simple_hexatope.state_ub is not None

    def test_get_bounds(self, simple_hexatope):
        """Test bounds computation."""
        lb, ub = simple_hexatope.get_bounds()

        assert lb.shape == (simple_hexatope.dim, 1)
        assert ub.shape == (simple_hexatope.dim, 1)
        assert np.all(lb <= ub)

    def test_identity_transformation(self):
        """Test identity transformation preserves bounds."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply identity transformation
        W = np.eye(2)
        b = np.zeros((2, 1))
        result = hexatope.affine_map(W, b)

        result_lb, result_ub = result.estimate_ranges()
        np.testing.assert_allclose(result_lb, lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, ub, atol=1e-6)

    def test_translation(self):
        """Test pure translation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Translate by [2, 3]
        W = np.eye(2)
        b = np.array([[2.0], [3.0]])
        result = hexatope.affine_map(W, b)

        result_lb, result_ub = result.estimate_ranges()
        expected_lb = np.array([[2.0], [3.0]])
        expected_ub = np.array([[3.0], [4.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_scaling(self):
        """Test scaling transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Scale by 2
        W = np.eye(2) * 2
        result = hexatope.affine_map(W)

        result_lb, result_ub = result.estimate_ranges()
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.0], [2.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_dimension_reduction(self):
        """Test dimension reduction."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Project to 2D and sum third dimension
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0]])
        result = hexatope.affine_map(W)

        assert result.dim == 2
        result_lb, result_ub = result.estimate_ranges()

        # First dimension: [0, 1]
        # Second dimension: [0, 2] (sum of two [0, 1] ranges)
        assert result_lb[0] <= 0.0 + 1e-6
        assert result_ub[0] >= 1.0 - 1e-6
        assert result_lb[1] <= 0.0 + 1e-6
        assert result_ub[1] >= 2.0 - 1e-6

    def test_is_empty_set(self, simple_hexatope):
        """Test emptiness checking."""
        # Simple hexatope from bounds should not be empty
        assert not simple_hexatope.is_empty_set()

    def test_contains_point_inside(self):
        """Test point containment for point inside."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        point_in = np.array([[0.5], [0.5]])
        assert hexatope.contains(point_in)

    def test_hexatope_to_box_conversion(self, simple_hexatope):
        """Test conversion to Box."""
        box = simple_hexatope.get_box(use_mcf=False)

        assert box.dim == simple_hexatope.dim
        assert np.all(box.lb <= box.ub)


    # Exact reachability tests for Hexatope
    def test_exact_simple_box_2d(self):
        """Test exact bounds for simple 2D box."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        lb_computed, ub_computed = hexatope.get_bounds()

        assert np.allclose(lb_computed, lb, atol=1e-6)
        assert np.allclose(ub_computed, ub, atol=1e-6)

    def test_exact_affine_transformed(self):
        """Test exact bounds after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply scaling: y = 2x
        W = np.eye(2) * 2
        hexatope_transformed = hexatope.affine_map(W)

        lb_computed, ub_computed = hexatope_transformed.get_bounds()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.0], [2.0]])

        assert np.allclose(lb_computed, expected_lb, atol=1e-6)
        assert np.allclose(ub_computed, expected_ub, atol=1e-6)

    def test_exact_dimension_reduction(self):
        """Test exact bounds after dimension reduction."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Project to 2D: y = [x_0, x_1 + x_2]
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0]])
        hexatope_projected = hexatope.affine_map(W)

        lb_computed, ub_computed = hexatope_projected.get_bounds()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1.0], [2.0]])

        assert np.allclose(lb_computed, expected_lb, atol=1e-6)
        assert np.allclose(ub_computed, expected_ub, atol=1e-6)

    def test_exact_vs_estimate(self):
        """Verify exact bounds are tighter or equal to estimates."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply complex transformation
        W = np.array([[1.0, 0.5, 0.0],
                      [0.0, 1.0, 0.5]])
        hexatope_transformed = hexatope.affine_map(W)

        lb_exact, ub_exact = hexatope_transformed.get_bounds()
        lb_estimate, ub_estimate = hexatope_transformed.estimate_ranges()

        # Exact should be contained in estimate
        assert np.all(lb_exact >= lb_estimate - 1e-6)
        assert np.all(ub_exact <= ub_estimate + 1e-6)


class TestOctatope:
    """Tests for Octatope set."""

    def test_creation(self, simple_octatope):
        """Test Octatope creation."""
        assert simple_octatope.dim == 3
        assert simple_octatope.nVar == 3
        pytest.assert_octatope_valid(simple_octatope)

    def test_from_bounds(self):
        """Test Octatope creation from bounds."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        assert octatope.dim == 3
        assert octatope.nVar == 3
        pytest.assert_octatope_valid(octatope)

        # Check bounds are preserved
        assert octatope.state_lb is not None
        assert octatope.state_ub is not None
        np.testing.assert_allclose(octatope.state_lb, lb, atol=1e-6)
        np.testing.assert_allclose(octatope.state_ub, ub, atol=1e-6)

    def test_affine_map(self, simple_octatope):
        """Test affine transformation."""
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0]])
        b = np.array([[0.5], [0.5]])

        result = simple_octatope.affine_map(W, b)

        assert result.dim == 2
        assert result.nVar == simple_octatope.nVar
        pytest.assert_octatope_valid(result)

    def test_estimate_ranges(self, simple_octatope):
        """Test range estimation."""
        lb, ub = simple_octatope.estimate_ranges()

        assert lb.shape == (simple_octatope.dim, 1)
        assert ub.shape == (simple_octatope.dim, 1)
        assert np.all(lb <= ub)

        # Check that state bounds are updated
        assert simple_octatope.state_lb is not None
        assert simple_octatope.state_ub is not None

    def test_get_bounds(self, simple_octatope):
        """Test bounds computation."""
        lb, ub = simple_octatope.get_bounds()

        assert lb.shape == (simple_octatope.dim, 1)
        assert ub.shape == (simple_octatope.dim, 1)
        assert np.all(lb <= ub)

    def test_identity_transformation(self):
        """Test identity transformation preserves bounds."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Apply identity transformation
        W = np.eye(2)
        b = np.zeros((2, 1))
        result = octatope.affine_map(W, b)

        result_lb, result_ub = result.estimate_ranges()
        np.testing.assert_allclose(result_lb, lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, ub, atol=1e-6)

    def test_translation(self):
        """Test pure translation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Translate by [2, 3]
        W = np.eye(2)
        b = np.array([[2.0], [3.0]])
        result = octatope.affine_map(W, b)

        result_lb, result_ub = result.estimate_ranges()
        expected_lb = np.array([[2.0], [3.0]])
        expected_ub = np.array([[3.0], [4.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_scaling(self):
        """Test scaling transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Scale by 2
        W = np.eye(2) * 2
        result = octatope.affine_map(W)

        result_lb, result_ub = result.estimate_ranges()
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.0], [2.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_dimension_reduction(self):
        """Test dimension reduction."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Project to 2D and sum third dimension
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0]])
        result = octatope.affine_map(W)

        assert result.dim == 2
        result_lb, result_ub = result.estimate_ranges()

        # First dimension: [0, 1]
        # Second dimension: [0, 2] (sum of two [0, 1] ranges)
        assert result_lb[0] <= 0.0 + 1e-6
        assert result_ub[0] >= 1.0 - 1e-6
        assert result_lb[1] <= 0.0 + 1e-6
        assert result_ub[1] >= 2.0 - 1e-6

    def test_is_empty_set(self, simple_octatope):
        """Test emptiness checking."""
        # Simple octatope from bounds should not be empty
        assert not simple_octatope.is_empty_set()

    def test_contains_point_inside(self):
        """Test point containment for point inside."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        point_in = np.array([[0.5], [0.5]])
        assert octatope.contains(point_in)

    def test_octatope_to_box_conversion(self, simple_octatope):
        """Test conversion to Box."""
        box = simple_octatope.get_box(use_mcf=False)

        assert box.dim == simple_octatope.dim
        assert np.all(box.lb <= box.ub)

    def test_utvpi_feasibility(self):
        """Test UTVPI constraint system feasibility."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Should be feasible
        assert octatope.utvpi.is_feasible()
        assert not octatope.is_empty_set()

    """Exact reachability tests for Octatope."""

    def test_exact_simple_box_2d(self):
        """Test exact bounds for simple 2D box."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        octatope = Octatope.from_bounds(lb, ub)

        lb_computed, ub_computed = octatope.get_bounds()

        assert np.allclose(lb_computed, lb, atol=1e-6)
        assert np.allclose(ub_computed, ub, atol=1e-6)

    def test_exact_affine_transformed(self):
        """Test exact bounds after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Apply scaling: y = 2x
        W = np.eye(2) * 2
        octatope_transformed = octatope.affine_map(W)

        lb_computed, ub_computed = octatope_transformed.get_bounds()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.0], [2.0]])

        assert np.allclose(lb_computed, expected_lb, atol=1e-6)
        assert np.allclose(ub_computed, expected_ub, atol=1e-6)

    def test_exact_dimension_reduction(self):
        """Test exact bounds after dimension reduction."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Project to 2D: y = [x_0, x_1 + x_2]
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0]])
        octatope_projected = octatope.affine_map(W)

        lb_computed, ub_computed = octatope_projected.get_bounds()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1.0], [2.0]])

        assert np.allclose(lb_computed, expected_lb, atol=1e-6)
        assert np.allclose(ub_computed, expected_ub, atol=1e-6)

    def test_exact_vs_estimate(self):
        """Verify exact bounds are tighter or equal to estimates."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Apply complex transformation
        W = np.array([[1.0, 0.5, 0.0],
                      [0.0, 1.0, 0.5]])
        octatope_transformed = octatope.affine_map(W)

        lb_exact, ub_exact = octatope_transformed.get_bounds()
        lb_estimate, ub_estimate = octatope_transformed.estimate_ranges()

        # Exact should be contained in estimate
        assert np.all(lb_exact >= lb_estimate - 1e-6)
        assert np.all(ub_exact <= ub_estimate + 1e-6)

    def test_hexatope_octatope_consistency(self):
        """Hexatope and Octatope should give same results for boxes."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])

        hexatope = Hexatope.from_bounds(lb, ub)
        octatope = Octatope.from_bounds(lb, ub)

        lb_hex, ub_hex = hexatope.get_bounds()
        lb_oct, ub_oct = octatope.get_bounds()

        assert np.allclose(lb_hex, lb_oct, atol=1e-6)
        assert np.allclose(ub_hex, ub_oct, atol=1e-6)


class TestToStarConversion:
    """Tests for to_star() conversion methods."""

    def test_hexatope_to_star_basic(self):
        """Test basic Hexatope to Star conversion."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # Check dimensions
        assert star.dim == hexatope.dim
        assert star.nVar == hexatope.nVar

        # Check that Star is valid
        pytest.assert_star_valid(star)

        # Check that predicate bounds are [-1, 1]
        np.testing.assert_array_equal(star.predicate_lb, np.full((hexatope.nVar, 1), -1.0))
        np.testing.assert_array_equal(star.predicate_ub, np.full((hexatope.nVar, 1), 1.0))

    def test_octatope_to_star_basic(self):
        """Test basic Octatope to Star conversion."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # Check dimensions
        assert star.dim == octatope.dim
        assert star.nVar == octatope.nVar

        # Check that Star is valid
        pytest.assert_star_valid(star)

        # Check that predicate bounds are [-1, 1]
        np.testing.assert_array_equal(star.predicate_lb, np.full((octatope.nVar, 1), -1.0))
        np.testing.assert_array_equal(star.predicate_ub, np.full((octatope.nVar, 1), 1.0))

    def test_hexatope_to_star_preserves_bounds(self):
        """Test that Hexatope to Star conversion preserves bounds."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # Compute bounds on both
        hex_lb, hex_ub = hexatope.get_bounds()
        star_lb, star_ub = star.get_ranges()

        # Star bounds should match Hexatope bounds (within tolerance)
        np.testing.assert_allclose(star_lb, hex_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, hex_ub, atol=1e-5)

    def test_octatope_to_star_preserves_bounds(self):
        """Test that Octatope to Star conversion preserves bounds."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # Compute bounds on both
        oct_lb, oct_ub = octatope.get_bounds()
        star_lb, star_ub = star.get_ranges()

        # Star bounds should match Octatope bounds (within tolerance)
        np.testing.assert_allclose(star_lb, oct_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, oct_ub, atol=1e-5)

    def test_hexatope_to_star_center_preservation(self):
        """Test that center is preserved in conversion."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # Star's center should be hexatope's center
        np.testing.assert_allclose(star.V[:, 0], hexatope.center, atol=1e-9)

    def test_octatope_to_star_center_preservation(self):
        """Test that center is preserved in conversion."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # Star's center should be octatope's center
        np.testing.assert_allclose(star.V[:, 0], octatope.center, atol=1e-9)

    def test_hexatope_to_star_generators_preservation(self):
        """Test that generators are preserved in conversion."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # Star's generators should match hexatope's generators
        np.testing.assert_allclose(star.V[:, 1:], hexatope.generators, atol=1e-9)

    def test_octatope_to_star_generators_preservation(self):
        """Test that generators are preserved in conversion."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # Star's generators should match octatope's generators
        np.testing.assert_allclose(star.V[:, 1:], octatope.generators, atol=1e-9)

    def test_hexatope_to_star_constraints_include_box(self):
        """Test that converted Star includes box constraints."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        star = hexatope.to_star()

        # Star should have constraints for DCS + box bounds
        # Box bounds add 2*nVar constraints (x <= 1 and x >= -1)
        n_vars = hexatope.nVar
        n_dcs_constraints = len(hexatope.dcs.constraints)

        expected_min_constraints = n_dcs_constraints + 2 * n_vars

        assert star.C.shape[0] >= expected_min_constraints

    def test_octatope_to_star_constraints_include_box(self):
        """Test that converted Star includes box constraints."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        star = octatope.to_star()

        # Star should have constraints for UTVPI + box bounds
        # Box bounds add 2*nVar constraints (x <= 1 and x >= -1)
        n_vars = octatope.nVar
        n_utvpi_constraints = len(octatope.utvpi.constraints)

        expected_min_constraints = n_utvpi_constraints + 2 * n_vars

        assert star.C.shape[0] >= expected_min_constraints

    def test_hexatope_to_star_after_affine_map(self):
        """Test conversion works after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply affine map
        W = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([[1.0], [2.0]])
        hexatope_transformed = hexatope.affine_map(W, b)

        star = hexatope_transformed.to_star()

        # Check that bounds are preserved
        hex_lb, hex_ub = hexatope_transformed.get_bounds()
        star_lb, star_ub = star.get_ranges()

        np.testing.assert_allclose(star_lb, hex_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, hex_ub, atol=1e-5)

    def test_octatope_to_star_after_affine_map(self):
        """Test conversion works after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Apply affine map
        W = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([[1.0], [2.0]])
        octatope_transformed = octatope.affine_map(W, b)

        star = octatope_transformed.to_star()

        # Check that bounds are preserved
        oct_lb, oct_ub = octatope_transformed.get_bounds()
        star_lb, star_ub = star.get_ranges()

        np.testing.assert_allclose(star_lb, oct_lb, atol=1e-5)
        np.testing.assert_allclose(star_ub, oct_ub, atol=1e-5)

    def test_hexatope_to_star_empty_dcs(self):
        """Test conversion with empty DCS (only box constraints)."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        # Create hexatope with no DCS constraints
        center = np.array([0.5, 0.5])
        generators = np.eye(2)
        dcs = DifferenceConstraintSystem(2)  # Empty DCS

        hexatope = Hexatope(center, generators, dcs)
        star = hexatope.to_star()

        # Should still work - star will have only box constraints
        assert star.dim == 2
        assert star.nVar == 2
        pytest.assert_star_valid(star)

    def test_hexatope_to_star_with_extra_constraints(self):
        """Test conversion when hexatope has extra constraints."""
        from n2v.sets.hexatope import DifferenceConstraintSystem

        # Create hexatope
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Add extra constraints via intersect_half_space
        H = np.array([[1.0, 1.0]])  # x + y <= 1.5
        g = np.array([[1.5]])
        hexatope_constrained = hexatope.intersect_half_space(H, g)

        star = hexatope_constrained.to_star()

        # Should include extra constraints
        assert star.C.shape[0] > len(hexatope.dcs.constraints) + 2 * hexatope.nVar
        pytest.assert_star_valid(star)

    def test_hexatope_octatope_star_consistency(self):
        """Test that Hexatope and Octatope give similar Star representations for boxes."""
        lb = np.array([[0.0], [1.0]])
        ub = np.array([[1.0], [3.0]])

        hexatope = Hexatope.from_bounds(lb, ub)
        octatope = Octatope.from_bounds(lb, ub)

        star_from_hex = hexatope.to_star()
        star_from_oct = octatope.to_star()

        # Both should have same dimensions
        assert star_from_hex.dim == star_from_oct.dim
        assert star_from_hex.nVar == star_from_oct.nVar

        # Both should give same bounds
        hex_star_lb, hex_star_ub = star_from_hex.get_ranges()
        oct_star_lb, oct_star_ub = star_from_oct.get_ranges()

        np.testing.assert_allclose(hex_star_lb, oct_star_lb, atol=1e-5)
        np.testing.assert_allclose(hex_star_ub, oct_star_ub, atol=1e-5)
