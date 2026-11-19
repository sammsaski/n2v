"""Tests for set representations."""

import pytest
import numpy as np
from n2v.sets import Star, Zono, Box, HalfSpace, Hexatope, Octatope
from tests.known_bugs import known_bug

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




class TestHalfSpaceReading:
    """HalfSpace reads exactly the region its arrays state: rows of G with
    the threshold g of the same row, over as many outputs as G has columns,
    in float64 without changing a stored number."""

    def test_a_1d_threshold_vector_holds_one_threshold_per_row(self):
        """G of 2 rows over 3 outputs with g given as a 1-D array of 2: g is a
        column, row k keeps threshold k, and dim is the number of columns.
        Catches a 1-D g turned into one row (refused as several columns) and
        dim read from the rows."""
        hs = HalfSpace(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                       np.array([0.25, -0.5]))
        assert hs.g.shape == (2, 1)
        assert hs.g[:, 0].tolist() == [0.25, -0.5]
        assert hs.dim == 3
        assert hs.contains(np.array([0.25, -0.5, 7.0]))
        assert not hs.contains(np.array([0.25, -0.25, 7.0]))

    def test_inputs_are_stored_as_float64_without_rounding(self):
        """An int G and a float32 g are stored as float64 holding the same
        numbers; a float64 threshold that float32 cannot hold (0.7) is kept
        as given. Catches g stored in float32, which moves 0.7 down to
        0.69999999 and shrinks the unsafe region."""
        hs = HalfSpace(np.array([[1, -2]]), np.array([[np.float32(0.7)]]))
        assert hs.G.dtype == np.float64 and hs.g.dtype == np.float64
        assert hs.G.tolist() == [[1.0, -2.0]]
        assert hs.g[0, 0] == float(np.float32(0.7))
        hs64 = HalfSpace(np.array([[1.0, -2.0]]), np.array([[0.7]]))
        assert hs64.g[0, 0] == 0.7
        assert hs64.contains(np.array([0.7, 0.0]))

    def test_a_matrix_of_another_rank_is_refused(self):
        """A 3-D G is not a list of rows: refused with ValueError."""
        with pytest.raises(ValueError):
            HalfSpace(np.ones((1, 1, 2)), np.array([[0.5]]))

    def test_no_rows_is_the_whole_space(self):
        """G with no row over 3 outputs: an AND of nothing holds at every
        point, and the dimension is still 3."""
        hs = HalfSpace(np.zeros((0, 3)), np.zeros((0, 1)))
        assert hs.dim == 3
        assert hs.contains(np.array([1e6, -1e6, 0.0]))

    def test_contains_refuses_several_columns(self):
        """contains takes one point: an x with two columns is refused, not
        answered for both."""
        hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.5]]))
        with pytest.raises(ValueError, match="one column"):
            hs.contains(np.zeros((2, 2)))

    def test_repr_and_str_state_the_dimension_and_the_shapes(self):
        """With 2 rows over 5 outputs the dimension, the row count and the
        shapes are each visible. Catches rows and columns swapped in the
        text."""
        hs = HalfSpace(np.ones((2, 5)), np.ones((2, 1)))
        assert "dim=5" in repr(hs) and "constraints=2" in repr(hs)
        assert "(2, 5)" in str(hs) and "(2, 1)" in str(hs)


class TestHalfSpaceContainsAtTheBoundary:
    """``contains`` answers ``G @ x <= g`` (its docstring). The unsafe region
    of a specification is this closed set, and a witness is graded at zero
    tolerance, so a point on the boundary is inside and a point outside by
    any margin is not."""

    def test_a_point_on_the_boundary_is_inside(self):
        """Exact equality in every row (dyadic numbers, so G @ x is exact)."""
        hs = HalfSpace(np.array([[0.5, -0.25], [1.0, 1.0]]),
                       np.array([[0.125], [1.5]]))
        x = np.array([0.5, 0.5])            # 0.25 - 0.125 = 0.125; 1.0 <= 1.5
        assert hs.contains(x)
        assert hs.contains(np.array([0.5, 1.0]))        # 0 <= 0.125; 1.5 = 1.5

    @known_bug(15, "HalfSpace.contains adds an absolute tolerance of 1e-8 "
                   "to g, so a point outside the half-space by less than that "
                   "is reported inside, against its docstring 'True if G @ x "
                   "<= g'; a witness checked with it can be one a "
                   "zero-tolerance replay rejects")
    def test_a_point_outside_by_a_small_margin_is_outside(self):
        """y0 <= 1/2 and the point y0 = 1/2 + 2^-30 (about 9.3e-10, exactly
        representable): outside, in exact arithmetic."""
        hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.5]]))
        x = np.array([0.5 + 2.0 ** -30, 0.0])
        from fractions import Fraction
        if not Fraction(x[0]) > Fraction(1, 2):
            pytest.fail("the point is not outside")
        assert not hs.contains(x)
