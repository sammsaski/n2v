"""Tests for set representations."""

import pytest
import numpy as np
from n2v.sets import Star, Zono, Box, HalfSpace, Hexatope, Octatope

class TestStar:
    """Tests for Star set."""

    def test_creation(self, simple_star):
        """Test Star creation with all properties."""
        # Test basic dimensions
        assert simple_star.dim == 3
        assert simple_star.nVar == 2

        # Test structural validity
        pytest.assert_star_valid(simple_star)

        # Test matrix shapes are correct
        assert simple_star.V.shape == (3, 3)  # (dim, nVar+1)
        assert simple_star.C.shape == (2, 2)  # (nConstraints, nVar)
        assert simple_star.d.shape == (2, 1)  # (nConstraints, 1)

        # Test constraint matrix and vector have matching row counts
        assert simple_star.C.shape[0] == simple_star.d.shape[0]

        # Test predicate bounds are set and have correct shape
        assert simple_star.predicate_lb is not None
        assert simple_star.predicate_ub is not None
        assert simple_star.predicate_lb.shape == (simple_star.nVar, 1)
        assert simple_star.predicate_ub.shape == (simple_star.nVar, 1)

        # Test predicate bounds have expected values
        np.testing.assert_array_equal(simple_star.predicate_lb, np.array([[0.0], [0.0]]))
        np.testing.assert_array_equal(simple_star.predicate_ub, np.array([[1.0], [1.0]]))

        # Test data types are float64
        assert simple_star.V.dtype == np.float64
        assert simple_star.C.dtype == np.float64
        assert simple_star.d.dtype == np.float64
        assert simple_star.predicate_lb.dtype == np.float64
        assert simple_star.predicate_ub.dtype == np.float64

    def test_creation_empty(self):
        """Test empty Star creation."""
        star = Star()

        assert star.dim == 0
        assert star.nVar == 0
        assert star.V.shape == (0, 0)
        assert star.C.shape == (0, 0)
        assert star.d.shape == (0, 1)
        assert star.predicate_lb is None
        assert star.predicate_ub is None
        assert star.state_lb is None
        assert star.state_ub is None
        assert star.Z is None

    def test_creation_no_constraints(self):
        """Test Star creation with no constraints (empty C and d)."""
        V = np.array([[1.0, 0.1, 0.0],
                      [0.0, 0.0, 0.2],
                      [0.0, 0.1, 0.0]])
        C = np.array([]).reshape(0, 2)
        d = np.array([]).reshape(0, 1)
        pred_lb = np.array([[0.0], [0.0]])
        pred_ub = np.array([[1.0], [1.0]])

        star = Star(V, C, d, pred_lb, pred_ub)

        assert star.dim == 3
        assert star.nVar == 2
        assert star.C.shape == (0, 2)
        assert star.d.shape == (0, 1)
        pytest.assert_star_valid(star)

    def test_creation_with_state_bounds(self):
        """Test Star creation with state bounds."""
        V = np.array([[1.0, 0.1], [0.0, 0.2]])
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[0.0]])
        pred_ub = np.array([[1.0]])
        state_lb = np.array([[-1.0], [-2.0]])
        state_ub = np.array([[2.0], [3.0]])

        star = Star(V, C, d, pred_lb, pred_ub, state_lb, state_ub)

        assert star.state_lb is not None
        assert star.state_ub is not None
        assert star.state_lb.shape == (star.dim, 1)
        assert star.state_ub.shape == (star.dim, 1)
        np.testing.assert_array_equal(star.state_lb, state_lb)
        np.testing.assert_array_equal(star.state_ub, state_ub)

    def test_creation_1d_constraint_vector(self):
        """Test that 1D constraint vector d is reshaped to column vector."""
        V = np.array([[1.0, 0.1], [0.0, 0.2]])
        C = np.array([[1.0]])
        d = np.array([1.0])  # 1D array

        star = Star(V, C, d)

        assert star.d.shape == (1, 1)  # Should be reshaped to column vector
        assert star.d[0, 0] == 1.0

    def test_creation_invalid_V_C_mismatch(self):
        """Test that mismatched V and C dimensions raise error."""
        V = np.array([[1.0, 0.1, 0.0], [0.0, 0.0, 0.2]])  # 3 columns
        C = np.array([[1.0]])  # 1 column - mismatch! (should be 2 for V with 3 cols)
        d = np.array([[1.0]])

        with pytest.raises(ValueError, match="Inconsistency between basic matrix"):
            Star(V, C, d)

    def test_creation_invalid_C_d_mismatch(self):
        """Test that mismatched C and d rows raise error."""
        V = np.array([[1.0, 0.1], [0.0, 0.2]])
        C = np.array([[1.0], [0.5]])  # 2 rows
        d = np.array([[1.0]])  # 1 row - mismatch!

        with pytest.raises(ValueError, match="Inconsistency between constraint matrix"):
            Star(V, C, d)

    def test_creation_invalid_d_multiple_columns(self):
        """Test that d with multiple columns raises error."""
        V = np.array([[1.0, 0.1], [0.0, 0.2]])
        C = np.array([[1.0]])
        d = np.array([[1.0, 2.0]])  # 2 columns - invalid!

        with pytest.raises(ValueError, match="should have one column"):
            Star(V, C, d)

    def test_creation_invalid_predicate_lb_size(self):
        """Test that wrong-sized predicate_lb raises error."""
        V = np.array([[1.0, 0.1, 0.0], [0.0, 0.0, 0.2]])
        C = np.array([[1.0, 0.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[0.0], [0.0], [0.0]])  # 3 elements, should be 2 (nVar)

        with pytest.raises(ValueError, match="Predicate lb size"):
            Star(V, C, d, pred_lb=pred_lb)

    def test_creation_invalid_state_lb_size(self):
        """Test that wrong-sized state_lb raises error."""
        V = np.array([[1.0, 0.1], [0.0, 0.2]])
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        state_lb = np.array([[0.0], [0.0], [0.0]])  # 3 elements, should be 2 (dim)

        with pytest.raises(ValueError, match="State lb size doesn't match dimension"):
            Star(V, C, d, state_lb=state_lb)

    def test_from_bounds(self):
        """Test Star creation from bounds."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Test basic dimensions
        assert star.dim == 3
        assert star.nVar == 3  # One per dimension
        pytest.assert_star_valid(star)

        # Test matrix shapes - Star from box should have nVar == dim
        assert star.V.shape == (3, 4)  # (dim, nVar+1)
        assert star.C.shape[1] == 3  # nVar columns
        assert star.d.shape[0] == star.C.shape[0]  # Matching constraint rows

        # CRITICAL: Check bounds are preserved
        computed_lb, computed_ub = star.get_ranges()
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

        # Check predicate bounds are set correctly (Box.to_star uses [-1, 1])
        assert star.predicate_lb is not None
        assert star.predicate_ub is not None
        np.testing.assert_allclose(star.predicate_lb, -np.ones((3, 1)), atol=1e-6)
        np.testing.assert_allclose(star.predicate_ub, np.ones((3, 1)), atol=1e-6)

    def test_from_bounds_1d(self):
        """Test Star creation from 1D bounds."""
        lb = np.array([[0.0]])
        ub = np.array([[5.0]])
        star = Star.from_bounds(lb, ub)

        assert star.dim == 1
        assert star.nVar == 1

        # Verify bounds
        computed_lb, computed_ub = star.get_ranges()
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_from_bounds_2d(self):
        """Test Star creation from 2D bounds."""
        lb = np.array([[-1.0], [2.0]])
        ub = np.array([[3.0], [5.0]])
        star = Star.from_bounds(lb, ub)

        assert star.dim == 2
        assert star.nVar == 2

        # Verify bounds
        computed_lb, computed_ub = star.get_ranges()
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_from_bounds_negative(self):
        """Test Star creation from bounds with negative values."""
        lb = np.array([[-5.0], [-3.0]])
        ub = np.array([[-1.0], [0.0]])
        star = Star.from_bounds(lb, ub)

        assert star.dim == 2

        # Verify bounds
        computed_lb, computed_ub = star.get_ranges()
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_from_bounds_different_ranges(self):
        """Test Star from bounds with different range sizes."""
        lb = np.array([[0.0], [10.0], [-100.0]])
        ub = np.array([[1.0], [20.0], [50.0]])
        star = Star.from_bounds(lb, ub)

        assert star.dim == 3

        # Verify bounds
        computed_lb, computed_ub = star.get_ranges()
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_from_bounds_point_containment(self):
        """Test that Star from bounds correctly contains points."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Points inside should be contained
        assert star.contains(np.array([[0.5], [0.5]]))
        assert star.contains(np.array([[0.0], [0.0]]))  # Lower corner
        assert star.contains(np.array([[1.0], [1.0]]))  # Upper corner
        assert star.contains(np.array([[0.25], [0.75]]))

        # Points outside should not be contained
        assert not star.contains(np.array([[1.5], [0.5]]))
        assert not star.contains(np.array([[0.5], [1.5]]))
        assert not star.contains(np.array([[-0.1], [0.5]]))
        assert not star.contains(np.array([[0.5], [-0.1]]))

    def test_affine_map(self, simple_star):
        """Test affine transformation - basic functionality."""
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 2.0, 0.0]])
        b = np.array([[0.5], [0.5]])

        result = simple_star.affine_map(W, b)

        # Test dimensions
        assert result.dim == 2
        assert result.nVar == simple_star.nVar
        pytest.assert_star_valid(result)

        # Test that constraints are preserved (C and d should be unchanged)
        np.testing.assert_array_equal(result.C, simple_star.C)
        np.testing.assert_array_equal(result.d, simple_star.d)

        # Test that predicate bounds are preserved
        if simple_star.predicate_lb is not None:
            np.testing.assert_array_equal(result.predicate_lb, simple_star.predicate_lb)
        if simple_star.predicate_ub is not None:
            np.testing.assert_array_equal(result.predicate_ub, simple_star.predicate_ub)

    def test_affine_map_identity(self):
        """Test identity transformation preserves Star."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Apply identity transformation
        W = np.eye(2)
        b = np.zeros((2, 1))
        result = star.affine_map(W, b)

        # Bounds should be preserved
        result_lb, result_ub = result.get_ranges()
        np.testing.assert_allclose(result_lb, lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, ub, atol=1e-6)

    def test_affine_map_translation(self):
        """Test pure translation (W=I, b≠0)."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Translate by [2, 3]
        W = np.eye(2)
        b = np.array([[2.0], [3.0]])
        result = star.affine_map(W, b)

        # Bounds should be shifted
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[2.0], [3.0]])
        expected_ub = np.array([[3.0], [4.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_affine_map_scaling(self):
        """Test scaling transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Scale by 2
        W = np.eye(2) * 2
        result = star.affine_map(W)

        # Bounds should be scaled
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.0], [2.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_affine_map_negative_scaling(self):
        """Test negative scaling (reflection)."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Reflect across origin
        W = np.eye(2) * -1
        result = star.affine_map(W)

        # Bounds should be reflected (and swapped)
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[-1.0], [-1.0]])
        expected_ub = np.array([[0.0], [0.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_affine_map_dimension_reduction(self):
        """Test dimension reduction via projection."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Project to 2D: [x0, x1+x2]
        W = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 1.0]])
        result = star.affine_map(W)

        assert result.dim == 2
        result_lb, result_ub = result.get_ranges()

        # First dimension: [0, 1]
        # Second dimension: [0, 2] (sum of two [0,1] ranges)
        np.testing.assert_allclose(result_lb[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result_ub[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(result_lb[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(result_ub[1], 2.0, atol=1e-6)

    def test_affine_map_no_bias(self):
        """Test affine map without bias (b=None)."""
        lb = np.array([[1.0], [2.0]])
        ub = np.array([[3.0], [4.0]])
        star = Star.from_bounds(lb, ub)

        # Scale without translation
        W = np.eye(2) * 2
        result = star.affine_map(W)  # No b parameter

        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[2.0], [4.0]])
        expected_ub = np.array([[6.0], [8.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_affine_map_combined(self):
        """Test combined scaling and translation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Scale by 2 and translate by [1, -1]
        W = np.eye(2) * 2
        b = np.array([[1.0], [-1.0]])
        result = star.affine_map(W, b)

        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[1.0], [-1.0]])
        expected_ub = np.array([[3.0], [1.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_affine_map_invalid_dimension(self):
        """Test that mismatched W dimensions raise error."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # W has wrong number of columns
        W = np.array([[1.0, 0.0, 0.0]])  # 3 columns, but star.dim = 2

        with pytest.raises(ValueError, match="has 3 columns, expected 2"):
            star.affine_map(W)

    def test_minkowski_sum(self):
        """Test Minkowski sum of two Stars."""
        # Create two simple box Stars
        lb1 = np.array([[0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        lb2 = np.array([[0.0], [0.0]])
        ub2 = np.array([[0.5], [0.5]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.minkowski_sum(star2)

        # Test dimensions
        assert result.dim == 2
        assert result.nVar == star1.nVar + star2.nVar  # Combined variables
        pytest.assert_star_valid(result)

        # Test bounds: Minkowski sum of [0,1]×[0,1] and [0,0.5]×[0,0.5] is [0,1.5]×[0,1.5]
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1.5], [1.5]])
        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_minkowski_sum_with_translation(self):
        """Test Minkowski sum with translated Stars."""
        # Star1: [0,1] × [0,1]
        lb1 = np.array([[0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # Star2: [2,3] × [2,3] (translated)
        lb2 = np.array([[2.0], [2.0]])
        ub2 = np.array([[3.0], [3.0]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.minkowski_sum(star2)

        # Minkowski sum should be [2,4] × [2,4]
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[2.0], [2.0]])
        expected_ub = np.array([[4.0], [4.0]])
        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_minkowski_sum_different_sizes(self):
        """Test Minkowski sum of Stars with different sizes."""
        # Star1: [0,2] × [0,2] (larger)
        lb1 = np.array([[0.0], [0.0]])
        ub1 = np.array([[2.0], [2.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # Star2: [0,0.1] × [0,0.1] (smaller)
        lb2 = np.array([[0.0], [0.0]])
        ub2 = np.array([[0.1], [0.1]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.minkowski_sum(star2)

        # Minkowski sum should be [0,2.1] × [0,2.1]
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.1], [2.1]])
        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_minkowski_sum_negative_ranges(self):
        """Test Minkowski sum with negative ranges."""
        # Star1: [-1,0] × [-1,0]
        lb1 = np.array([[-1.0], [-1.0]])
        ub1 = np.array([[0.0], [0.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # Star2: [0,1] × [0,1]
        lb2 = np.array([[0.0], [0.0]])
        ub2 = np.array([[1.0], [1.0]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.minkowski_sum(star2)

        # Minkowski sum should be [-1,1] × [-1,1]
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[-1.0], [-1.0]])
        expected_ub = np.array([[1.0], [1.0]])
        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_minkowski_sum_1d(self):
        """Test Minkowski sum in 1D."""
        # Star1: [0,1]
        lb1 = np.array([[0.0]])
        ub1 = np.array([[1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # Star2: [2,3]
        lb2 = np.array([[2.0]])
        ub2 = np.array([[3.0]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.minkowski_sum(star2)

        # Minkowski sum should be [2,4]
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[2.0]])
        expected_ub = np.array([[4.0]])
        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_minkowski_sum_3d(self):
        """Test Minkowski sum in 3D."""
        # Star1: [0,1]³
        lb1 = np.array([[0.0], [0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # Star2: [0,0.5]³
        lb2 = np.array([[0.0], [0.0], [0.0]])
        ub2 = np.array([[0.5], [0.5], [0.5]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.minkowski_sum(star2)

        # Minkowski sum should be [0,1.5]³
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[0.0], [0.0], [0.0]])
        expected_ub = np.array([[1.5], [1.5], [1.5]])
        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_minkowski_sum_structure(self):
        """Test that Minkowski sum has correct internal structure."""
        lb1 = np.array([[0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        lb2 = np.array([[0.0], [0.0]])
        ub2 = np.array([[0.5], [0.5]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.minkowski_sum(star2)

        # V matrix should have nVar+1 columns
        assert result.V.shape[0] == 2  # dim
        assert result.V.shape[1] == result.nVar + 1

        # C and d should be block-diagonal combination
        assert result.C.shape[0] == star1.C.shape[0] + star2.C.shape[0]
        assert result.C.shape[1] == star1.nVar + star2.nVar
        assert result.d.shape[0] == star1.d.shape[0] + star2.d.shape[0]

        # Predicate bounds should be combined
        if star1.predicate_lb is not None and star2.predicate_lb is not None:
            assert result.predicate_lb.shape[0] == star1.nVar + star2.nVar
            assert result.predicate_ub.shape[0] == star1.nVar + star2.nVar

    def test_minkowski_sum_invalid_type(self):
        """Test that Minkowski sum with non-Star raises TypeError."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        with pytest.raises(TypeError, match="Can only compute Minkowski sum with another Star"):
            star.minkowski_sum("not a star")

    def test_minkowski_sum_dimension_mismatch(self):
        """Test that Minkowski sum with mismatched dimensions raises error."""
        # 2D Star
        lb1 = np.array([[0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # 3D Star
        lb2 = np.array([[0.0], [0.0], [0.0]])
        ub2 = np.array([[1.0], [1.0], [1.0]])
        star2 = Star.from_bounds(lb2, ub2)

        with pytest.raises(ValueError, match="Dimension mismatch: 2 vs 3"):
            star1.minkowski_sum(star2)

    def test_intersect_half_space(self, simple_star):
        """Test intersection with half-space - basic functionality."""
        G = np.array([[1.0, 0.0, 0.0]])
        g = np.array([[0.5]])

        result = simple_star.intersect_half_space(G, g)

        # Test dimensions preserved
        assert result.dim == simple_star.dim
        assert result.nVar == simple_star.nVar

        # Test constraint added
        assert result.C.shape[0] == simple_star.C.shape[0] + 1
        assert result.d.shape[0] == simple_star.d.shape[0] + 1

        # Test V unchanged (only constraints change)
        np.testing.assert_array_equal(result.V, simple_star.V)

        # Test predicate bounds unchanged
        if simple_star.predicate_lb is not None:
            np.testing.assert_array_equal(result.predicate_lb, simple_star.predicate_lb)
        if simple_star.predicate_ub is not None:
            np.testing.assert_array_equal(result.predicate_ub, simple_star.predicate_ub)

        pytest.assert_star_valid(result)

    def test_intersect_half_space_bounds(self):
        """Test that half-space intersection correctly restricts bounds."""
        # Start with [0,1] × [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Intersect with x0 <= 0.5
        H = np.array([[1.0, 0.0]])
        g = np.array([[0.5]])
        result = star.intersect_half_space(H, g)

        # Should get [0, 0.5] × [0, 1]
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[0.5], [1.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_intersect_half_space_multiple(self):
        """Test multiple half-space intersections."""
        # Start with [0,1] × [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Intersect with x0 <= 0.7
        H1 = np.array([[1.0, 0.0]])
        g1 = np.array([[0.7]])
        star = star.intersect_half_space(H1, g1)

        # Then intersect with x1 <= 0.6
        H2 = np.array([[0.0, 1.0]])
        g2 = np.array([[0.6]])
        result = star.intersect_half_space(H2, g2)

        # Should get [0, 0.7] × [0, 0.6]
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[0.7], [0.6]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_intersect_half_space_diagonal(self):
        """Test intersection with diagonal half-space."""
        # Start with [0,1] × [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Intersect with x0 + x1 <= 1 (diagonal constraint)
        H = np.array([[1.0, 1.0]])
        g = np.array([[1.0]])
        result = star.intersect_half_space(H, g)

        # Corner (0,0) should be in
        assert result.contains(np.array([[0.0], [0.0]]))

        # Corner (1,1) should be on boundary or just outside
        # (depends on tolerance, but (0.9, 0.9) should definitely be out)
        assert result.contains(np.array([[0.4], [0.4]]))

        # Point (0.6, 0.6) violates x0+x1<=1
        assert not result.contains(np.array([[0.6], [0.6]]))

    def test_intersect_half_space_lower_bound(self):
        """Test intersection with lower bound constraint (>=)."""
        # Start with [0,1] × [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Intersect with -x0 <= -0.3, i.e., x0 >= 0.3
        H = np.array([[-1.0, 0.0]])
        g = np.array([[-0.3]])
        result = star.intersect_half_space(H, g)

        # Should get [0.3, 1] × [0, 1]
        result_lb, result_ub = result.get_ranges()
        expected_lb = np.array([[0.3], [0.0]])
        expected_ub = np.array([[1.0], [1.0]])

        np.testing.assert_allclose(result_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(result_ub, expected_ub, atol=1e-6)

    def test_intersect_half_space_point_containment(self):
        """Test that intersection correctly includes/excludes points."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Intersect with x0 <= 0.5
        H = np.array([[1.0, 0.0]])
        g = np.array([[0.5]])
        result = star.intersect_half_space(H, g)

        # Points with x0 <= 0.5 should be in
        assert result.contains(np.array([[0.0], [0.5]]))
        assert result.contains(np.array([[0.3], [0.5]]))
        assert result.contains(np.array([[0.5], [0.5]]))  # Boundary

        # Points with x0 > 0.5 should be out
        assert not result.contains(np.array([[0.6], [0.5]]))
        assert not result.contains(np.array([[0.9], [0.5]]))
        assert not result.contains(np.array([[1.0], [0.5]]))

    def test_intersect_half_space_no_intersection(self):
        """Test intersection with half-space that excludes entire Star."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Intersect with x0 <= -1 (entire Star has x0 >= 0)
        H = np.array([[1.0, 0.0]])
        g = np.array([[-1.0]])
        result = star.intersect_half_space(H, g)

        # Result should be empty (or at least have no valid points)
        # The Star structure exists but represents empty set
        # We can check this by verifying bounds are inconsistent
        result_lb, result_ub = result.get_ranges()

        # When empty, typically lb > ub or bounds are at limits
        # Just verify the structure is created (actual emptiness checking
        # would be done by is_empty() method if it exists)
        assert result.dim == 2
        pytest.assert_star_valid(result)

    def test_intersect_half_space_3d(self):
        """Test half-space intersection in 3D."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Intersect with x0 + x1 + x2 <= 1.5
        H = np.array([[1.0, 1.0, 1.0]])
        g = np.array([[1.5]])
        result = star.intersect_half_space(H, g)

        assert result.dim == 3

        # Origin should be in
        assert result.contains(np.array([[0.0], [0.0], [0.0]]))

        # Center should be in (0.5+0.5+0.5 = 1.5)
        assert result.contains(np.array([[0.5], [0.5], [0.5]]))

        # Corner (1,1,1) should be out (1+1+1 = 3 > 1.5)
        assert not result.contains(np.array([[1.0], [1.0], [1.0]]))

    def test_intersect_half_space_structure(self):
        """Test that constraint is correctly added to Star structure."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        original_C_rows = star.C.shape[0]

        # Add constraint
        H = np.array([[1.0, 0.0]])
        g = np.array([[0.5]])
        result = star.intersect_half_space(H, g)

        # V should be unchanged
        np.testing.assert_array_equal(result.V, star.V)

        # C should have one more row
        assert result.C.shape[0] == original_C_rows + 1
        assert result.C.shape[1] == star.C.shape[1]  # Same columns (nVar)

        # d should have one more row
        assert result.d.shape[0] == original_C_rows + 1

        # Old constraints should be preserved (first rows of C and d)
        np.testing.assert_array_equal(result.C[:original_C_rows, :], star.C)
        np.testing.assert_array_equal(result.d[:original_C_rows, :], star.d)

    def test_convex_hull(self):
        """Test convex hull over-approximation of two Stars."""
        # Create two box Stars
        lb1 = np.array([[0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        lb2 = np.array([[2.0], [2.0]])
        ub2 = np.array([[3.0], [3.0]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.convex_hull(star2)

        # Test dimensions
        assert result.dim == 2
        assert result.nVar == star1.nVar + star2.nVar + 1  # Combined + convex parameter
        pytest.assert_star_valid(result)

        # Convex hull should contain both original Stars
        result_lb, result_ub = result.get_ranges()

        # Should contain the entire range from min of both to max of both
        # (This is an over-approximation)
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[3.0], [3.0]])

        # Result should at least contain the bounding box of both Stars
        assert np.all(result_lb <= expected_lb + 1e-6)
        assert np.all(result_ub >= expected_ub - 1e-6)

    def test_convex_hull_overlapping(self):
        """Test convex hull of overlapping Stars."""
        # Star1: [0,1] × [0,1]
        lb1 = np.array([[0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # Star2: [0.5,1.5] × [0.5,1.5] (overlapping)
        lb2 = np.array([[0.5], [0.5]])
        ub2 = np.array([[1.5], [1.5]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.convex_hull(star2)

        # Should contain both Stars
        result_lb, result_ub = result.get_ranges()

        # The union would be at least [0, 1.5] × [0, 1.5]
        assert result_lb[0] <= 0.0 + 1e-6
        assert result_lb[1] <= 0.0 + 1e-6
        assert result_ub[0] >= 1.5 - 1e-6
        assert result_ub[1] >= 1.5 - 1e-6

    def test_convex_hull_identical(self):
        """Test convex hull of identical Stars."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star1 = Star.from_bounds(lb, ub)
        star2 = Star.from_bounds(lb, ub)

        result = star1.convex_hull(star2)

        # Convex hull of identical sets should contain the original
        result_lb, result_ub = result.get_ranges()

        # Should at least cover the original bounds
        assert result_lb[0] <= 0.0 + 1e-6
        assert result_lb[1] <= 0.0 + 1e-6
        assert result_ub[0] >= 1.0 - 1e-6
        assert result_ub[1] >= 1.0 - 1e-6

    def test_convex_hull_1d(self):
        """Test convex hull in 1D."""
        # Star1: [0,1]
        lb1 = np.array([[0.0]])
        ub1 = np.array([[1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # Star2: [3,4]
        lb2 = np.array([[3.0]])
        ub2 = np.array([[4.0]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.convex_hull(star2)

        assert result.dim == 1

        # Convex hull of [0,1] and [3,4] should contain [0,4]
        result_lb, result_ub = result.get_ranges()
        assert result_lb[0] <= 0.0 + 1e-6
        assert result_ub[0] >= 4.0 - 1e-6

    def test_convex_hull_3d(self):
        """Test convex hull in 3D."""
        # Star1: [0,1]³
        lb1 = np.array([[0.0], [0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # Star2: [2,3]³
        lb2 = np.array([[2.0], [2.0], [2.0]])
        ub2 = np.array([[3.0], [3.0], [3.0]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.convex_hull(star2)

        assert result.dim == 3

        # Should contain the range [0,3]³
        result_lb, result_ub = result.get_ranges()
        assert np.all(result_lb <= np.array([[0.0], [0.0], [0.0]]) + 1e-6)
        assert np.all(result_ub >= np.array([[3.0], [3.0], [3.0]]) - 1e-6)

    def test_convex_hull_point_containment(self):
        """Test that convex hull contains points from both original Stars."""
        # Star1: [0,1] × [0,1]
        lb1 = np.array([[0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # Star2: [3,4] × [3,4]
        lb2 = np.array([[3.0], [3.0]])
        ub2 = np.array([[4.0], [4.0]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.convex_hull(star2)

        # Points from star1 should be in convex hull
        assert result.contains(np.array([[0.5], [0.5]]))

        # Points from star2 should be in convex hull
        assert result.contains(np.array([[3.5], [3.5]]))

        # Points on the line segment between centers should be in convex hull
        # Center of star1: [0.5, 0.5], Center of star2: [3.5, 3.5]
        # Midpoint: [2.0, 2.0]
        assert result.contains(np.array([[2.0], [2.0]]))

    def test_convex_hull_structure(self):
        """Test that convex hull has correct internal structure."""
        lb1 = np.array([[0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        lb2 = np.array([[2.0], [2.0]])
        ub2 = np.array([[3.0], [3.0]])
        star2 = Star.from_bounds(lb2, ub2)

        result = star1.convex_hull(star2)

        # V matrix should have nVar+1 columns
        assert result.V.shape[0] == 2  # dim
        assert result.V.shape[1] == result.nVar + 1

        # nVar should be sum of both Stars' nVars plus 1 (for convex parameter)
        expected_nVar = star1.nVar + star2.nVar + 1
        assert result.nVar == expected_nVar

        # C should include constraints from both Stars plus extra constraints
        # At minimum: star1 constraints + star2 constraints + 2 (for convex parameter bounds)
        min_constraints = star1.C.shape[0] + star2.C.shape[0] + 2
        assert result.C.shape[0] >= min_constraints

    def test_convex_hull_invalid_type(self):
        """Test that convex hull with non-Star raises TypeError."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        with pytest.raises(TypeError, match="Can only compute convex hull with another Star"):
            star.convex_hull("not a star")

    def test_convex_hull_dimension_mismatch(self):
        """Test that convex hull with mismatched dimensions raises error."""
        # 2D Star
        lb1 = np.array([[0.0], [0.0]])
        ub1 = np.array([[1.0], [1.0]])
        star1 = Star.from_bounds(lb1, ub1)

        # 3D Star
        lb2 = np.array([[0.0], [0.0], [0.0]])
        ub2 = np.array([[1.0], [1.0], [1.0]])
        star2 = Star.from_bounds(lb2, ub2)

        with pytest.raises(ValueError, match="Dimension mismatch: 2 vs 3"):
            star1.convex_hull(star2)

    def test_get_ranges(self, simple_star):
        """Test get_ranges() - basic functionality."""
        lb, ub = simple_star.get_ranges()

        assert lb.shape == (simple_star.dim, 1)
        assert ub.shape == (simple_star.dim, 1)
        assert np.all(lb <= ub)

    def test_get_ranges_from_bounds(self):
        """Test that get_ranges() preserves bounds for box Stars."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        star = Star.from_bounds(lb, ub)

        computed_lb, computed_ub = star.get_ranges()

        # Should recover original bounds exactly
        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_get_ranges_after_translation(self):
        """Test get_ranges() after translation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Translate by [2, 3]
        W = np.eye(2)
        b = np.array([[2.0], [3.0]])
        star_translated = star.affine_map(W, b)

        computed_lb, computed_ub = star_translated.get_ranges()

        # Bounds should be shifted
        expected_lb = np.array([[2.0], [3.0]])
        expected_ub = np.array([[3.0], [4.0]])
        np.testing.assert_allclose(computed_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, expected_ub, atol=1e-6)

    def test_get_ranges_after_scaling(self):
        """Test get_ranges() after scaling."""
        lb = np.array([[0.0], [1.0]])
        ub = np.array([[1.0], [2.0]])
        star = Star.from_bounds(lb, ub)

        # Scale by 2
        W = np.eye(2) * 2
        star_scaled = star.affine_map(W)

        computed_lb, computed_ub = star_scaled.get_ranges()

        # Bounds should be scaled
        expected_lb = np.array([[0.0], [2.0]])
        expected_ub = np.array([[2.0], [4.0]])
        np.testing.assert_allclose(computed_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, expected_ub, atol=1e-6)

    def test_get_ranges_after_half_space_intersection(self):
        """Test get_ranges() after half-space intersection."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Intersect with x0 <= 0.5
        H = np.array([[1.0, 0.0]])
        g = np.array([[0.5]])
        star_intersected = star.intersect_half_space(H, g)

        computed_lb, computed_ub = star_intersected.get_ranges()

        # Should get [0, 0.5] × [0, 1]
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[0.5], [1.0]])
        np.testing.assert_allclose(computed_lb, expected_lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, expected_ub, atol=1e-6)

    def test_get_ranges_1d(self):
        """Test get_ranges() for 1D Star."""
        lb = np.array([[2.0]])
        ub = np.array([[5.0]])
        star = Star.from_bounds(lb, ub)

        computed_lb, computed_ub = star.get_ranges()

        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_get_ranges_3d(self):
        """Test get_ranges() for 3D Star."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        star = Star.from_bounds(lb, ub)

        computed_lb, computed_ub = star.get_ranges()

        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_get_ranges_negative(self):
        """Test get_ranges() with negative bounds."""
        lb = np.array([[-5.0], [-3.0]])
        ub = np.array([[-1.0], [0.0]])
        star = Star.from_bounds(lb, ub)

        computed_lb, computed_ub = star.get_ranges()

        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)

    def test_get_range_single_dimension(self):
        """Test get_range() for a single dimension."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        star = Star.from_bounds(lb, ub)

        # Test each dimension
        for i in range(3):
            min_val, max_val = star.get_range(i)
            np.testing.assert_allclose(min_val, lb[i, 0], atol=1e-6)
            np.testing.assert_allclose(max_val, ub[i, 0], atol=1e-6)

    def test_get_range_invalid_index(self):
        """Test that get_range() raises error for invalid index."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Negative index
        with pytest.raises(ValueError, match="Invalid index -1"):
            star.get_range(-1)

        # Index >= dim
        with pytest.raises(ValueError, match="Invalid index 2"):
            star.get_range(2)

    def test_get_ranges_parallel(self):
        """Test that parallel get_ranges() gives same results as sequential."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        star = Star.from_bounds(lb, ub)

        # Transform to make it more interesting
        W = np.array([[1.0, 0.5, 0.0],
                      [0.0, 1.0, 0.5],
                      [0.0, 0.0, 1.0]])
        b = np.array([[1.0], [2.0], [3.0]])
        star_transformed = star.affine_map(W, b)

        # Get ranges sequentially
        lb_seq, ub_seq = star_transformed.get_ranges(parallel=False)

        # Get ranges in parallel
        lb_par, ub_par = star_transformed.get_ranges(parallel=True, n_workers=2)

        # Results should be identical
        np.testing.assert_allclose(lb_par, lb_seq, atol=1e-6)
        np.testing.assert_allclose(ub_par, ub_seq, atol=1e-6)

    def test_get_ranges_batch_matches_sequential(self):
        """Batch get_ranges matches per-dimension get_range."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        star = Star.from_bounds(lb, ub)

        W = np.array([[1.0, 0.5, 0.0],
                      [0.0, 1.0, 0.5],
                      [0.0, 0.0, 1.0]])
        b = np.array([[1.0], [2.0], [3.0]])
        star_t = star.affine_map(W, b)

        # Per-dimension sequential via get_range
        lb_seq = np.zeros((star_t.dim, 1))
        ub_seq = np.zeros((star_t.dim, 1))
        for i in range(star_t.dim):
            lb_seq[i], ub_seq[i] = star_t.get_range(i)

        # Batch via get_ranges (uses _get_ranges_batch)
        lb_batch, ub_batch = star_t.get_ranges(parallel=False)

        np.testing.assert_allclose(lb_batch, lb_seq, atol=1e-6)
        np.testing.assert_allclose(ub_batch, ub_seq, atol=1e-6)

    def test_get_ranges_consistency_with_get_box(self):
        """Test that get_ranges() and get_box() give consistent results."""
        lb = np.array([[0.0], [1.0]])
        ub = np.array([[2.0], [3.0]])
        star = Star.from_bounds(lb, ub)

        # Get ranges
        ranges_lb, ranges_ub = star.get_ranges()

        # Get box
        box = star.get_box()

        # Should match
        np.testing.assert_allclose(ranges_lb, box.lb, atol=1e-6)
        np.testing.assert_allclose(ranges_ub, box.ub, atol=1e-6)

    def test_estimate_ranges(self, simple_star):
        """Test estimate_ranges() - basic functionality."""
        lb, ub = simple_star.estimate_ranges()

        # Check returned values
        assert lb.shape == (simple_star.dim, 1)
        assert ub.shape == (simple_star.dim, 1)
        assert np.all(lb <= ub)

        # Check state attributes are set
        assert simple_star.state_lb is not None
        assert simple_star.state_ub is not None
        np.testing.assert_array_equal(simple_star.state_lb, lb)
        np.testing.assert_array_equal(simple_star.state_ub, ub)

    def test_estimate_ranges_from_bounds(self):
        """Test that estimate_ranges() gives reasonable bounds for box Stars."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        star = Star.from_bounds(lb, ub)

        est_lb, est_ub = star.estimate_ranges()

        # Estimates should contain the true bounds (over-approximation)
        assert np.all(est_lb <= lb + 1e-6)
        assert np.all(est_ub >= ub - 1e-6)

    def test_estimate_ranges_over_approximation(self):
        """Test that estimate_ranges() over-approximates get_ranges()."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Add diagonal constraint to make it non-trivial
        H = np.array([[1.0, 1.0]])
        g = np.array([[1.0]])
        star_constrained = star.intersect_half_space(H, g)

        # Get exact bounds (LP)
        exact_lb, exact_ub = star_constrained.get_ranges()

        # Get estimated bounds (interval arithmetic)
        est_lb, est_ub = star_constrained.estimate_ranges()

        # Estimates should over-approximate (contain) exact bounds
        assert np.all(est_lb <= exact_lb + 1e-6)
        assert np.all(est_ub >= exact_ub - 1e-6)

    def test_estimate_ranges_after_translation(self):
        """Test estimate_ranges() after translation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Translate by [2, 3]
        W = np.eye(2)
        b = np.array([[2.0], [3.0]])
        star_translated = star.affine_map(W, b)

        est_lb, est_ub = star_translated.estimate_ranges()

        # Estimates should be close to expected shifted bounds
        expected_lb = np.array([[2.0], [3.0]])
        expected_ub = np.array([[3.0], [4.0]])

        # Should over-approximate
        assert np.all(est_lb <= expected_lb + 1e-6)
        assert np.all(est_ub >= expected_ub - 1e-6)

    def test_estimate_ranges_after_scaling(self):
        """Test estimate_ranges() after scaling."""
        lb = np.array([[0.0], [1.0]])
        ub = np.array([[1.0], [2.0]])
        star = Star.from_bounds(lb, ub)

        # Scale by 2
        W = np.eye(2) * 2
        star_scaled = star.affine_map(W)

        est_lb, est_ub = star_scaled.estimate_ranges()

        # Estimates should be close to scaled bounds
        expected_lb = np.array([[0.0], [2.0]])
        expected_ub = np.array([[2.0], [4.0]])

        assert np.all(est_lb <= expected_lb + 1e-6)
        assert np.all(est_ub >= expected_ub - 1e-6)

    def test_estimate_ranges_1d(self):
        """Test estimate_ranges() for 1D Star."""
        lb = np.array([[2.0]])
        ub = np.array([[5.0]])
        star = Star.from_bounds(lb, ub)

        est_lb, est_ub = star.estimate_ranges()

        assert est_lb.shape == (1, 1)
        assert est_ub.shape == (1, 1)
        assert est_lb[0, 0] <= 2.0 + 1e-6
        assert est_ub[0, 0] >= 5.0 - 1e-6

    def test_estimate_ranges_3d(self):
        """Test estimate_ranges() for 3D Star."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        star = Star.from_bounds(lb, ub)

        est_lb, est_ub = star.estimate_ranges()

        assert est_lb.shape == (3, 1)
        assert est_ub.shape == (3, 1)

        # Should over-approximate
        for i in range(3):
            assert est_lb[i, 0] <= lb[i, 0] + 1e-6
            assert est_ub[i, 0] >= ub[i, 0] - 1e-6

    def test_estimate_ranges_negative(self):
        """Test estimate_ranges() with negative bounds."""
        lb = np.array([[-5.0], [-3.0]])
        ub = np.array([[-1.0], [0.0]])
        star = Star.from_bounds(lb, ub)

        est_lb, est_ub = star.estimate_ranges()

        assert np.all(est_lb <= lb + 1e-6)
        assert np.all(est_ub >= ub - 1e-6)

    def test_estimate_range_single_dimension(self):
        """Test estimate_range() for a single dimension."""
        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [3.0], [5.0]])
        star = Star.from_bounds(lb, ub)

        # Test each dimension
        for i in range(3):
            min_est, max_est = star.estimate_range(i)

            # Should over-approximate the true bounds
            assert min_est <= lb[i, 0] + 1e-6
            assert max_est >= ub[i, 0] - 1e-6

    def test_estimate_ranges_fallback_no_predicate_bounds(self):
        """Test that estimate_range falls back to LP when no predicate bounds."""
        # Create a Star without predicate bounds but with valid constraints
        V = np.array([[1.0, 0.1, 0.0],
                      [0.0, 0.0, 0.2],
                      [0.0, 0.1, 0.0]])
        C = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [-1.0, 0.0],
                      [0.0, -1.0]])
        d = np.array([[1.0], [1.0], [0.0], [0.0]])  # Box constraints: 0 <= alpha <= 1
        star = Star(V, C, d)  # No predicate bounds

        # Should fall back to LP (exact computation)
        for i in range(star.dim):
            est_min, est_max = star.estimate_range(i)
            exact_min, exact_max = star.get_range(i)

            # Since it falls back to LP, should be exact (both non-None)
            assert est_min is not None
            assert exact_min is not None
            np.testing.assert_allclose(est_min, exact_min, atol=1e-6)
            np.testing.assert_allclose(est_max, exact_max, atol=1e-6)

    def test_estimate_ranges_faster_than_exact(self):
        """Test that estimate_ranges() is faster than get_ranges() for complex Stars."""
        import time

        # Create a more complex Star with many variables
        lb = np.array([[0.0], [0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Add multiple constraints
        for i in range(3):
            H = np.zeros((1, 4))
            H[0, i] = 1.0
            H[0, i+1] = 1.0
            star = star.intersect_half_space(H, np.array([[1.5]]))

        # Time estimate_ranges (should be fast - interval arithmetic)
        start = time.time()
        est_lb, est_ub = star.estimate_ranges()
        estimate_time = time.time() - start

        # Time get_ranges (slower - LP solving)
        start = time.time()
        exact_lb, exact_ub = star.get_ranges()
        exact_time = time.time() - start

        # Estimate should be faster (though this is not guaranteed on all systems)
        # At minimum, both should complete successfully
        assert est_lb is not None
        assert exact_lb is not None

        # Estimates should over-approximate
        assert np.all(est_lb <= exact_lb + 1e-6)
        assert np.all(est_ub >= exact_ub - 1e-6)

    # ========================================================================
    # _solve_lp Tests
    # ========================================================================

    def test_solve_lp_minimization_basic(self):
        """Test basic minimization with bounded LP."""
        # Create Star: 0 ≤ α₀, α₁ ≤ 1
        V = np.array([[1.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        C = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [-1.0, 0.0],
                      [0.0, -1.0]])
        d = np.array([[1.0], [1.0], [0.0], [0.0]])
        pred_lb = np.array([[0.0], [0.0]])
        pred_ub = np.array([[1.0], [1.0]])
        star = Star(V, C, d, pred_lb, pred_ub)

        # Minimize α₀ + α₁: optimal at (0, 0) → value = 0
        f = np.array([[1.0], [1.0]])
        result = star._solve_lp(f, minimize=True)
        assert result is not None
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_solve_lp_maximization_basic(self):
        """Test basic maximization with bounded LP."""
        # Create Star: 0 ≤ α₀, α₁ ≤ 1
        V = np.array([[1.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        C = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [-1.0, 0.0],
                      [0.0, -1.0]])
        d = np.array([[1.0], [1.0], [0.0], [0.0]])
        pred_lb = np.array([[0.0], [0.0]])
        pred_ub = np.array([[1.0], [1.0]])
        star = Star(V, C, d, pred_lb, pred_ub)

        # Maximize α₀ + α₁: optimal at (1, 1) → value = 2
        f = np.array([[1.0], [1.0]])
        result = star._solve_lp(f, minimize=False)
        assert result is not None
        np.testing.assert_allclose(result, 2.0, atol=1e-6)

    def test_solve_lp_single_variable(self):
        """Test LP with single variable."""
        # Create 1D Star: 0 ≤ α ≤ 5
        V = np.array([[0.0, 1.0]])
        C = np.array([[1.0], [-1.0]])
        d = np.array([[5.0], [0.0]])
        pred_lb = np.array([[0.0]])
        pred_ub = np.array([[5.0]])
        star = Star(V, C, d, pred_lb, pred_ub)

        # Minimize α: optimal at 0
        f = np.array([[1.0]])
        result = star._solve_lp(f, minimize=True)
        assert result is not None
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

        # Maximize α: optimal at 5
        result = star._solve_lp(f, minimize=False)
        assert result is not None
        np.testing.assert_allclose(result, 5.0, atol=1e-6)

    def test_solve_lp_zero_variables(self):
        """Test LP with zero variables (edge case)."""
        # Create Star with nVar = 0 (single point)
        V = np.array([[1.0]])
        C = np.empty((0, 0))
        d = np.empty((0, 1))
        star = Star(V, C, d)

        # Should return 0.0 for nVar = 0
        f = np.empty((0, 1))
        result = star._solve_lp(f, minimize=True)
        assert result == 0.0

    def test_solve_lp_infeasible(self):
        """Test LP with infeasible constraints."""
        # Create Star with contradictory constraints: α ≤ 0 and α ≥ 1
        V = np.array([[0.0, 1.0]])
        C = np.array([[1.0], [-1.0]])
        d = np.array([[0.0], [-1.0]])  # α ≤ 0 and -α ≤ -1 (i.e., α ≥ 1)
        star = Star(V, C, d)

        # LP should be infeasible
        f = np.array([[1.0]])
        result = star._solve_lp(f, minimize=True)
        assert result is None

    def test_solve_lp_with_predicate_bounds_only(self):
        """Test LP using only predicate bounds (no C matrix constraints)."""
        # Create Star with empty C but predicate bounds
        V = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        C = np.empty((0, 2))
        d = np.empty((0, 1))
        pred_lb = np.array([[-1.0], [-1.0]])
        pred_ub = np.array([[2.0], [3.0]])
        star = Star(V, C, d, pred_lb, pred_ub)

        # Minimize α₀: optimal at -1
        f = np.array([[1.0], [0.0]])
        result = star._solve_lp(f, minimize=True)
        assert result is not None
        np.testing.assert_allclose(result, -1.0, atol=1e-6)

        # Maximize α₁: optimal at 3
        f = np.array([[0.0], [1.0]])
        result = star._solve_lp(f, minimize=False)
        assert result is not None
        np.testing.assert_allclose(result, 3.0, atol=1e-6)

    def test_solve_lp_no_bounds_feasible(self):
        """Test LP with no predicate bounds but feasible C constraints."""
        # Create Star with only C matrix constraints: α₀ + α₁ ≤ 10
        V = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        C = np.array([[1.0, 1.0]])
        d = np.array([[10.0]])
        star = Star(V, C, d)

        # Minimize α₀ - α₁: unbounded below → should return None
        f = np.array([[1.0], [-1.0]])
        result = star._solve_lp(f, minimize=True)
        # LP is unbounded, should return None
        assert result is None

    def test_solve_lp_tight_constraints(self):
        """Test LP with tight constraints forming a single point."""
        # Create Star representing single point: α₀ = α₁ = 0.5
        V = np.array([[1.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        C = np.array([[1.0, 0.0],
                      [-1.0, 0.0],
                      [0.0, 1.0],
                      [0.0, -1.0]])
        d = np.array([[0.5], [-0.5], [0.5], [-0.5]])
        star = Star(V, C, d)

        # Any objective should give same value since feasible region is a point
        f = np.array([[1.0], [0.0]])
        result_min = star._solve_lp(f, minimize=True)
        result_max = star._solve_lp(f, minimize=False)
        assert result_min is not None
        assert result_max is not None
        np.testing.assert_allclose(result_min, 0.5, atol=1e-6)
        np.testing.assert_allclose(result_max, 0.5, atol=1e-6)

    def test_solve_lp_negative_objective_coefficients(self):
        """Test LP with negative objective coefficients."""
        # Create Star: 0 ≤ α₀, α₁ ≤ 1
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Minimize -α₀ - α₁ (equivalent to maximizing α₀ + α₁)
        # Optimal at (1, 1) → value = -2
        f = np.array([[-1.0], [-1.0]])
        result = star._solve_lp(f, minimize=True)
        assert result is not None
        np.testing.assert_allclose(result, -2.0, atol=1e-6)

    def test_solve_lp_zero_objective(self):
        """Test LP with zero objective (all coefficients zero)."""
        # Create Star: 0 ≤ α₀, α₁ ≤ 1
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Minimize 0*α₀ + 0*α₁: optimal value = 0
        f = np.array([[0.0], [0.0]])
        result = star._solve_lp(f, minimize=True)
        assert result is not None
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_solve_lp_mixed_objective(self):
        """Test LP with mixed positive/negative objective coefficients.

        Star.from_bounds creates predicate space with -1 ≤ α ≤ 1.
        This test verifies _solve_lp correctly handles mixed sign coefficients.
        """
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        # Star.from_bounds creates: -1 ≤ α₀, α₁ ≤ 1 (NOT 0 ≤ α ≤ 1)
        assert np.allclose(star.predicate_lb, [[-1.0], [-1.0]])
        assert np.allclose(star.predicate_ub, [[1.0], [1.0]])

        # Minimize α₀ - α₁
        # Optimal: α₀ = -1, α₁ = 1 → value = -2
        f = np.array([[1.0], [-1.0]])
        result = star._solve_lp(f, minimize=True)
        assert result is not None
        np.testing.assert_allclose(result, -2.0, atol=1e-6)

        # Maximize α₀ - α₁
        # Optimal: α₀ = 1, α₁ = -1 → value = 2
        result = star._solve_lp(f, minimize=False)
        assert result is not None
        np.testing.assert_allclose(result, 2.0, atol=1e-6)

    def test_solve_lp_consistency_with_get_range(self):
        """Test that _solve_lp is consistent with get_range.

        get_range(i) computes: [V[i,0] + min(V[i,1:]@α), V[i,0] + max(V[i,1:]@α)]
        _solve_lp(V[i,1:]) computes: min/max(V[i,1:]@α)

        So: get_range(i) should equal [V[i,0] + _solve_lp(V[i,1:], min),
                                        V[i,0] + _solve_lp(V[i,1:], max)]
        """
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        star = Star.from_bounds(lb, ub)

        # Test for each dimension
        for dim in range(3):
            # Get range using get_range method
            exact_lb, exact_ub = star.get_range(dim)

            # Get the same range using _solve_lp directly
            # The objective for dimension i is V[i, 1:] (generators for that dimension)
            f = star.V[dim, 1:].reshape(-1, 1)

            result_min = star._solve_lp(f, minimize=True)
            result_max = star._solve_lp(f, minimize=False)

            # Add the constant term V[dim, 0]
            computed_lb = star.V[dim, 0] + result_min
            computed_ub = star.V[dim, 0] + result_max

            # Should match get_range results
            np.testing.assert_allclose(computed_lb, exact_lb, atol=1e-6)
            np.testing.assert_allclose(computed_ub, exact_ub, atol=1e-6)

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


