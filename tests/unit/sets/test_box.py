"""Tests for set representations."""

import pytest
import numpy as np
from n2v.sets import Star, Zono, Box, HalfSpace, Hexatope, Octatope

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

    def test_get_ranges(self, simple_box):
        """Test get_ranges() for API consistency with Star."""
        lb, ub = simple_box.get_ranges()

        np.testing.assert_array_equal(lb, simple_box.lb)
        np.testing.assert_array_equal(ub, simple_box.ub)

    def test_get_ranges_matches_estimate_ranges(self, simple_box):
        """Test that get_ranges() and estimate_ranges() return same values for Box."""
        lb_get, ub_get = simple_box.get_ranges()
        lb_est, ub_est = simple_box.estimate_ranges()

        np.testing.assert_array_equal(lb_get, lb_est)
        np.testing.assert_array_equal(ub_get, ub_est)

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

    # ========================================================================
    # Additional Creation & Validation Tests
    # ========================================================================

    def test_creation_1d(self):
        """Test creation of 1D Box."""
        lb = np.array([[0.0]])
        ub = np.array([[5.0]])
        box = Box(lb, ub)

        assert box.dim == 1
        assert box.lb.shape == (1, 1)
        assert box.ub.shape == (1, 1)
        np.testing.assert_array_equal(box.lb, lb)
        np.testing.assert_array_equal(box.ub, ub)

    def test_creation_from_1d_arrays(self):
        """Test that 1D arrays are converted to column vectors."""
        lb = np.array([0.0, 1.0, 2.0])  # 1D array
        ub = np.array([1.0, 2.0, 3.0])  # 1D array
        box = Box(lb, ub)

        assert box.dim == 3
        assert box.lb.shape == (3, 1)
        assert box.ub.shape == (3, 1)

    def test_creation_center_computed(self):
        """Test that center is correctly computed."""
        lb = np.array([[0.0], [2.0]])
        ub = np.array([[4.0], [6.0]])
        box = Box(lb, ub)

        expected_center = np.array([[2.0], [4.0]])
        np.testing.assert_array_equal(box.center, expected_center)

    def test_creation_generators_computed(self):
        """Test that generators are correctly computed."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[2.0], [4.0]])
        box = Box(lb, ub)

        # Generators should be diagonal with half-widths
        # Half-widths: [1.0, 2.0]
        expected = np.array([[1.0, 0.0],
                            [0.0, 2.0]])
        np.testing.assert_array_equal(box.generators, expected)

    def test_creation_with_zero_width_dimension(self):
        """Test Box with zero width in one dimension (degenerate)."""
        lb = np.array([[0.0], [1.0]])
        ub = np.array([[1.0], [1.0]])  # Second dimension has zero width
        box = Box(lb, ub)

        assert box.dim == 2
        # Generators should have only non-zero columns
        assert box.generators.shape[1] == 1  # Only one non-zero dimension

    def test_creation_invalid_dimension_mismatch(self):
        """Test that mismatched dimensions raise error."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])  # Wrong dimension

        with pytest.raises(ValueError, match="same dimension"):
            Box(lb, ub)

    def test_creation_invalid_non_column_vector(self):
        """Test that non-column vectors raise error."""
        lb = np.array([[0.0, 0.0]])  # Row vector (1, 2) instead of (2, 1)
        ub = np.array([[1.0, 1.0]])

        with pytest.raises(ValueError, match="column vectors"):
            Box(lb, ub)

    def test_creation_negative_bounds(self):
        """Test Box with negative bounds."""
        lb = np.array([[-5.0], [-3.0]])
        ub = np.array([[-1.0], [0.0]])
        box = Box(lb, ub)

        assert box.dim == 2
        np.testing.assert_array_equal(box.lb, lb)
        np.testing.assert_array_equal(box.ub, ub)

    # ========================================================================
    # Partitioning Tests
    # ========================================================================

    def test_single_partition_basic(self):
        """Test single dimension partitioning."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[4.0], [2.0]])
        box = Box(lb, ub)

        # Partition first dimension into 2 parts
        parts = box.single_partition(part_id=0, part_num=2)

        assert len(parts) == 2
        # First partition: [0, 2] × [0, 2]
        np.testing.assert_array_almost_equal(parts[0].lb, [[0.0], [0.0]])
        np.testing.assert_array_almost_equal(parts[0].ub, [[2.0], [2.0]])
        # Second partition: [2, 4] × [0, 2]
        np.testing.assert_array_almost_equal(parts[1].lb, [[2.0], [0.0]])
        np.testing.assert_array_almost_equal(parts[1].ub, [[4.0], [2.0]])

    def test_single_partition_into_one(self):
        """Test partition with part_num=1 returns original box."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        box = Box(lb, ub)

        parts = box.single_partition(part_id=0, part_num=1)

        assert len(parts) == 1
        assert parts[0] is box  # Should return self

    def test_single_partition_many_parts(self):
        """Test partitioning into many parts."""
        lb = np.array([[0.0]])
        ub = np.array([[10.0]])
        box = Box(lb, ub)

        parts = box.single_partition(part_id=0, part_num=5)

        assert len(parts) == 5
        # Each part should be width 2.0
        for i, part in enumerate(parts):
            expected_lb = i * 2.0
            expected_ub = (i + 1) * 2.0
            np.testing.assert_array_almost_equal(part.lb, [[expected_lb]])
            np.testing.assert_array_almost_equal(part.ub, [[expected_ub]])

    def test_single_partition_invalid_index(self):
        """Test that invalid partition index raises error."""
        box = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        with pytest.raises(ValueError, match="Invalid partition index"):
            box.single_partition(part_id=5, part_num=2)

        with pytest.raises(ValueError, match="Invalid partition index"):
            box.single_partition(part_id=-1, part_num=2)

    def test_single_partition_invalid_part_num(self):
        """Test that invalid part_num raises error."""
        box = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        with pytest.raises(ValueError, match="Number of partitions must be"):
            box.single_partition(part_id=0, part_num=0)

    def test_partition_multiple_dimensions(self):
        """Test partitioning multiple dimensions."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[4.0], [6.0]])
        box = Box(lb, ub)

        # Partition: dim 0 into 2, dim 1 into 3
        parts = box.partition(part_indexes=[0, 1], part_numbers=[2, 3])

        # Should get 2 × 3 = 6 boxes
        assert len(parts) == 6

        # Verify coverage: each box should have volume (4/2) × (6/3) = 2 × 2 = 4
        for part in parts:
            volume = np.prod(part.ub - part.lb)
            np.testing.assert_almost_equal(volume, 4.0, decimal=6)

    def test_partition_invalid_mismatch(self):
        """Test that mismatched indexes and numbers raise error."""
        box = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        with pytest.raises(ValueError, match="same length"):
            box.partition(part_indexes=[0, 1], part_numbers=[2])

    # ========================================================================
    # Affine Transformation Tests
    # ========================================================================

    def test_affine_map_identity(self):
        """Test identity transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        box = Box(lb, ub)

        W = np.eye(2)
        result = box.affine_map(W)

        # Should be unchanged
        np.testing.assert_array_almost_equal(result.lb, lb)
        np.testing.assert_array_almost_equal(result.ub, ub)

    def test_affine_map_translation(self):
        """Test pure translation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        box = Box(lb, ub)

        W = np.eye(2)
        b = np.array([[2.0], [3.0]])
        result = box.affine_map(W, b)

        # Bounds should be shifted by b
        expected_lb = np.array([[2.0], [3.0]])
        expected_ub = np.array([[3.0], [4.0]])
        np.testing.assert_array_almost_equal(result.lb, expected_lb)
        np.testing.assert_array_almost_equal(result.ub, expected_ub)

    def test_affine_map_scaling(self):
        """Test scaling transformation."""
        lb = np.array([[0.0], [1.0]])
        ub = np.array([[1.0], [2.0]])
        box = Box(lb, ub)

        W = np.eye(2) * 3  # Scale by 3
        result = box.affine_map(W)

        expected_lb = np.array([[0.0], [3.0]])
        expected_ub = np.array([[3.0], [6.0]])
        np.testing.assert_array_almost_equal(result.lb, expected_lb)
        np.testing.assert_array_almost_equal(result.ub, expected_ub)

    def test_affine_map_dimension_reduction(self):
        """Test projection to lower dimension."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        box = Box(lb, ub)

        # Project to 2D: [x0, x1+x2]
        W = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 1.0]])
        result = box.affine_map(W)

        assert result.dim == 2
        # x1+x2 has range [0, 2]
        np.testing.assert_array_almost_equal(result.lb, [[0.0], [0.0]])
        np.testing.assert_array_almost_equal(result.ub, [[1.0], [2.0]])

    def test_affine_map_rotation(self):
        """Test rotation transformation produces valid bounding box."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        box = Box(lb, ub)

        # 45-degree rotation matrix
        theta = np.pi / 4
        W = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
        result = box.affine_map(W)

        assert result.dim == 2
        # Result should be a valid box (lb <= ub)
        assert np.all(result.lb <= result.ub)
        # The resulting box should be an over-approximation of rotated box
        # which has bounds approximately [-0.707, 0.707] × [0, 1.414]
        # Check that result contains the origin (which is in original box)
        assert np.all(result.lb <= np.array([[0.0], [0.0]]))
        assert np.all(result.ub >= np.array([[0.0], [0.0]]))

    # ========================================================================
    # Conversion Tests
    # ========================================================================

    def test_to_zono(self):
        """Test conversion to Zonotope."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[2.0], [4.0]])
        box = Box(lb, ub)

        zono = box.to_zono()

        # Center should match
        np.testing.assert_array_equal(zono.c, box.center)
        # Generators should match
        np.testing.assert_array_equal(zono.V, box.generators)

    def test_to_star(self):
        """Test conversion to Star."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        box = Box(lb, ub)

        star = box.to_star()

        # Star should represent same region
        star_lb, star_ub = star.get_ranges()
        np.testing.assert_array_almost_equal(star_lb, lb, decimal=5)
        np.testing.assert_array_almost_equal(star_ub, ub, decimal=5)

    # ========================================================================
    # Sampling Tests
    # ========================================================================

    def test_sample_basic(self):
        """Test random sampling."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        box = Box(lb, ub)

        samples = box.sample(N=100)

        assert samples.shape == (2, 100)
        # All samples should be within bounds
        assert np.all(samples >= lb)
        assert np.all(samples <= ub)

    def test_sample_1d(self):
        """Test sampling from 1D box."""
        lb = np.array([[2.0]])
        ub = np.array([[5.0]])
        box = Box(lb, ub)

        samples = box.sample(N=50)

        assert samples.shape == (1, 50)
        assert np.all(samples >= 2.0)
        assert np.all(samples <= 5.0)

    def test_sample_single_point(self):
        """Test sampling with N=1."""
        box = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        sample = box.sample(N=1)

        assert sample.shape == (2, 1)

    # ========================================================================
    # Getter Tests
    # ========================================================================

    def test_get_vertices_2d(self):
        """Test vertex enumeration for 2D box."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        box = Box(lb, ub)

        vertices = box.get_vertices()

        # 2D box has 4 vertices
        assert vertices.shape[0] == 2
        assert vertices.shape[1] == 4

        # Verify all corners are present
        expected_vertices = np.array([[0.0, 1.0, 0.0, 1.0],
                                     [0.0, 0.0, 1.0, 1.0]])
        # Sort both for comparison
        vertices_sorted = vertices[:, np.lexsort(vertices)]
        expected_sorted = expected_vertices[:, np.lexsort(expected_vertices)]
        np.testing.assert_array_almost_equal(vertices_sorted, expected_sorted)

    def test_get_vertices_1d(self):
        """Test vertex enumeration for 1D box (interval)."""
        lb = np.array([[2.0]])
        ub = np.array([[5.0]])
        box = Box(lb, ub)

        vertices = box.get_vertices()

        # 1D box has 2 vertices
        assert vertices.shape == (1, 2)
        assert 2.0 in vertices
        assert 5.0 in vertices

    def test_get_vertices_3d(self):
        """Test vertex enumeration for 3D box."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        box = Box(lb, ub)

        vertices = box.get_vertices()

        # 3D box has 8 vertices
        assert vertices.shape[0] == 3
        assert vertices.shape[1] == 8

    def test_get_vertices_degenerate(self):
        """Test vertex enumeration for degenerate box (point)."""
        lb = np.array([[1.0], [2.0]])
        ub = np.array([[1.0], [2.0]])  # Point, not a box
        box = Box(lb, ub)

        vertices = box.get_vertices()

        # Should have only 1 unique vertex
        assert vertices.shape == (2, 1)
        np.testing.assert_array_equal(vertices, [[1.0], [2.0]])

    # ========================================================================
    # Static Method Tests
    # ========================================================================

    def test_box_hull_two_boxes(self):
        """Test hull of two boxes."""
        box1 = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))
        box2 = Box(np.array([[2.0], [2.0]]), np.array([[3.0], [3.0]]))

        hull = Box.box_hull([box1, box2])

        # Hull should contain both boxes
        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[3.0], [3.0]])
        np.testing.assert_array_equal(hull.lb, expected_lb)
        np.testing.assert_array_equal(hull.ub, expected_ub)

    def test_box_hull_overlapping(self):
        """Test hull of overlapping boxes."""
        box1 = Box(np.array([[0.0], [0.0]]), np.array([[2.0], [2.0]]))
        box2 = Box(np.array([[1.0], [1.0]]), np.array([[3.0], [3.0]]))

        hull = Box.box_hull([box1, box2])

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[3.0], [3.0]])
        np.testing.assert_array_equal(hull.lb, expected_lb)
        np.testing.assert_array_equal(hull.ub, expected_ub)

    def test_box_hull_single_box(self):
        """Test hull of single box."""
        box = Box(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))

        hull = Box.box_hull([box])

        # Hull of single box is itself
        np.testing.assert_array_equal(hull.lb, box.lb)
        np.testing.assert_array_equal(hull.ub, box.ub)

    def test_box_hull_many_boxes(self):
        """Test hull of many boxes."""
        boxes = [
            Box(np.array([[0.0]]), np.array([[1.0]])),
            Box(np.array([[2.0]]), np.array([[3.0]])),
            Box(np.array([[4.0]]), np.array([[5.0]])),
            Box(np.array([[-1.0]]), np.array([[0.5]]))
        ]

        hull = Box.box_hull(boxes)

        # Hull should span from -1.0 to 5.0
        np.testing.assert_array_equal(hull.lb, [[-1.0]])
        np.testing.assert_array_equal(hull.ub, [[5.0]])

    def test_box_hull_empty_list(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="empty box list"):
            Box.box_hull([])


