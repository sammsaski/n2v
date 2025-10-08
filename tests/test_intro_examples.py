"""
Tests for introductory NNV examples.

These tests validate the Python implementation against MATLAB NNV by comparing
computed results with saved MATLAB outputs.
"""

import pytest
import numpy as np
from pathlib import Path
from scipy.io import loadmat

from n2v.sets import Zono


# Path to saved MATLAB results
INTRO_EXAMPLES_DIR = Path(__file__).parent / "intro_examples"
SAVED_RESULTS_DIR = INTRO_EXAMPLES_DIR / "saved_results"


@pytest.fixture
def matlab_example_9d_results():
    """Load MATLAB results for Example 9.D."""
    results_path = SAVED_RESULTS_DIR / "example_9d_results.mat"

    if not results_path.exists():
        pytest.skip(
            f"MATLAB results not found at {results_path}. "
            "Please run example_9d_parallelogram.m first."
        )

    data = loadmat(str(results_path))
    results = data['results']

    # Helper to extract fields from MATLAB struct
    def get_field(struct, field_name):
        return struct[field_name][0, 0]

    return {
        'c': get_field(results, 'c'),
        'V': get_field(results, 'V'),
        'dim': int(get_field(results, 'dim')),
        'num_generators': int(get_field(results, 'num_generators')),
        'lb': get_field(results, 'lb'),
        'ub': get_field(results, 'ub'),
        'vertices': get_field(results, 'vertices'),
        'num_vertices': int(get_field(results, 'num_vertices')),
        'affine_W': get_field(results, 'affine_W'),
        'affine_b': get_field(results, 'affine_b'),
        'affine_c': get_field(results, 'affine_c'),
        'affine_V': get_field(results, 'affine_V'),
        'affine_lb': get_field(results, 'affine_lb'),
        'affine_ub': get_field(results, 'affine_ub'),
        'mink_c2': get_field(results, 'mink_c2'),
        'mink_V2': get_field(results, 'mink_V2'),
        'mink_sum_c': get_field(results, 'mink_sum_c'),
        'mink_sum_V': get_field(results, 'mink_sum_V'),
        'mink_sum_lb': get_field(results, 'mink_sum_lb'),
        'mink_sum_ub': get_field(results, 'mink_sum_ub'),
        'test_points': get_field(results, 'test_points'),
        'contains_results': get_field(results, 'contains_results').flatten(),
    }


class TestExample9DParallelogram:
    """Tests for Example 9.D: Parallelogram Zonotope."""

    def test_zonotope_creation(self, matlab_example_9d_results):
        """Test zonotope creation matches MATLAB."""
        matlab = matlab_example_9d_results

        # Create Python zonotope
        c = np.array([[2.0], [3.0]])
        V = np.array([[1.0, 0.0], [1.0, 1.0]])
        Z = Zono(c, V)

        # Validate against MATLAB
        assert Z.dim == matlab['dim']
        assert V.shape[1] == matlab['num_generators']
        np.testing.assert_allclose(Z.c, matlab['c'], atol=1e-10)
        np.testing.assert_allclose(Z.V, matlab['V'], atol=1e-10)

    def test_bounds_computation(self, matlab_example_9d_results):
        """Test bounds computation matches MATLAB."""
        matlab = matlab_example_9d_results

        # Create zonotope
        Z = Zono(matlab['c'], matlab['V'])

        # Compute bounds
        lb, ub = Z.get_bounds()

        # Validate against MATLAB
        np.testing.assert_allclose(lb, matlab['lb'], atol=1e-10)
        np.testing.assert_allclose(ub, matlab['ub'], atol=1e-10)

        # Verify the specific property from Example 9.D:
        # Upper bound in vertical dimension should be 3 + 1 + 1 = 5
        assert abs(ub[1, 0] - 5.0) < 1e-10, \
            f"Expected upper bound y = 5.0, got {ub[1, 0]}"

    def test_vertices_computation(self, matlab_example_9d_results):
        """Test vertex enumeration matches MATLAB."""
        matlab = matlab_example_9d_results

        # Create zonotope
        Z = Zono(matlab['c'], matlab['V'])

        # Compute vertices
        vertices = Z.get_vertices()

        # Validate shape
        assert vertices.shape[1] == matlab['num_vertices']

        # MATLAB and Python may return vertices in different orders
        # Check that all Python vertices exist in MATLAB vertices
        for i in range(vertices.shape[1]):
            python_vertex = vertices[:, i:i+1]

            # Find matching MATLAB vertex
            found = False
            for j in range(matlab['vertices'].shape[1]):
                matlab_vertex = matlab['vertices'][:, j:j+1]
                if np.allclose(python_vertex, matlab_vertex, atol=1e-10):
                    found = True
                    break

            assert found, f"Python vertex {python_vertex.flatten()} not found in MATLAB vertices"

    def test_affine_transformation(self, matlab_example_9d_results):
        """Test affine transformation matches MATLAB."""
        matlab = matlab_example_9d_results

        # Create zonotope
        Z = Zono(matlab['c'], matlab['V'])

        # Apply affine transformation
        W = matlab['affine_W']
        b = matlab['affine_b']
        Z_affine = Z.affine_map(W, b)

        # Validate transformed zonotope
        np.testing.assert_allclose(Z_affine.c, matlab['affine_c'], atol=1e-10)
        np.testing.assert_allclose(Z_affine.V, matlab['affine_V'], atol=1e-10)

        # Validate bounds
        lb_affine, ub_affine = Z_affine.get_bounds()
        np.testing.assert_allclose(lb_affine, matlab['affine_lb'], atol=1e-10)
        np.testing.assert_allclose(ub_affine, matlab['affine_ub'], atol=1e-10)

    def test_minkowski_sum(self, matlab_example_9d_results):
        """Test Minkowski sum matches MATLAB."""
        matlab = matlab_example_9d_results

        # Create first zonotope
        Z1 = Zono(matlab['c'], matlab['V'])

        # Create second zonotope
        Z2 = Zono(matlab['mink_c2'], matlab['mink_V2'])

        # Compute Minkowski sum
        Z_sum = Z1.minkowski_sum(Z2)

        # Validate result
        np.testing.assert_allclose(Z_sum.c, matlab['mink_sum_c'], atol=1e-10)
        np.testing.assert_allclose(Z_sum.V, matlab['mink_sum_V'], atol=1e-10)

        # Validate bounds
        lb_sum, ub_sum = Z_sum.get_bounds()
        np.testing.assert_allclose(lb_sum, matlab['mink_sum_lb'], atol=1e-10)
        np.testing.assert_allclose(ub_sum, matlab['mink_sum_ub'], atol=1e-10)

    def test_point_containment(self, matlab_example_9d_results):
        """Test point containment matches MATLAB."""
        matlab = matlab_example_9d_results

        # Create zonotope
        Z = Zono(matlab['c'], matlab['V'])

        # Test points
        test_points = matlab['test_points']
        matlab_contains = matlab['contains_results']

        # Test each point
        for i in range(test_points.shape[0]):
            point = test_points[i, :].reshape(-1, 1)
            python_result = Z.contains(point)
            matlab_result = bool(matlab_contains[i])

            assert python_result == matlab_result, \
                f"Point {point.flatten()}: Python={python_result}, MATLAB={matlab_result}"

    def test_compact_notation(self, matlab_example_9d_results):
        """Test that the parallelogram is correctly represented in compact notation."""
        matlab = matlab_example_9d_results

        # The parallelogram in compact notation: (⟨2, 1, 0⟩, ⟨3, 1, 1⟩)
        # This means:
        #   - Dimension 1: center=2, generators=[1, 0]
        #   - Dimension 2: center=3, generators=[1, 1]

        c = matlab['c']
        V = matlab['V']

        # Check center
        assert c[0, 0] == 2.0, "First dimension center should be 2"
        assert c[1, 0] == 3.0, "Second dimension center should be 3"

        # Check generators
        assert V[0, 0] == 1.0 and V[0, 1] == 0.0, \
            "First dimension generators should be [1, 0]"
        assert V[1, 0] == 1.0 and V[1, 1] == 1.0, \
            "Second dimension generators should be [1, 1]"

    def test_parallelogram_corners(self, matlab_example_9d_results):
        """Test that the corners of the parallelogram are correct."""
        matlab = matlab_example_9d_results

        # Create zonotope
        Z = Zono(matlab['c'], matlab['V'])

        # The parallelogram corners occur when both epsilons are at their extremes
        # Corner 1: ε₁=-1, ε₂=-1 → (2-1+0, 3-1-1) = (1, 1)
        # Corner 2: ε₁=+1, ε₂=-1 → (2+1+0, 3+1-1) = (3, 3)
        # Corner 3: ε₁=+1, ε₂=+1 → (2+1+0, 3+1+1) = (3, 5)
        # Corner 4: ε₁=-1, ε₂=+1 → (2-1+0, 3-1+1) = (1, 3)

        expected_corners = np.array([
            [1.0, 1.0],
            [3.0, 3.0],
            [3.0, 5.0],
            [1.0, 3.0]
        ])

        vertices = matlab['vertices']

        # Check that all expected corners are in the vertex set
        for corner in expected_corners:
            found = False
            for j in range(vertices.shape[1]):
                vertex = vertices[:, j]
                if np.allclose(corner, vertex, atol=1e-10):
                    found = True
                    break
            assert found, f"Expected corner {corner} not found in vertices"


class TestExample9DWithoutMATLAB:
    """Tests for Example 9.D that don't require MATLAB results."""

    def test_basic_zonotope_creation(self):
        """Test basic zonotope creation without MATLAB."""
        c = np.array([[2.0], [3.0]])
        V = np.array([[1.0, 0.0], [1.0, 1.0]])
        Z = Zono(c, V)

        assert Z.dim == 2
        assert Z.V.shape[1] == 2
        np.testing.assert_array_equal(Z.c, c)
        np.testing.assert_array_equal(Z.V, V)

    def test_upper_bound_verification(self):
        """Test that upper bound in y-dimension equals 5."""
        c = np.array([[2.0], [3.0]])
        V = np.array([[1.0, 0.0], [1.0, 1.0]])
        Z = Zono(c, V)

        lb, ub = Z.get_bounds()

        # From Example 9.D: 3 + ε₁ + ε₂ with ε₁=1, ε₂=1 gives 5
        assert abs(ub[1, 0] - 5.0) < 1e-10

    def test_bounds_formula(self):
        """Test bounds computation using the formula."""
        c = np.array([[2.0], [3.0]])
        V = np.array([[1.0, 0.0], [1.0, 1.0]])
        Z = Zono(c, V)

        lb, ub = Z.get_bounds()

        # Manual calculation:
        # x: center=2, generators=[1, 0], so x ∈ [2-1-0, 2+1+0] = [1, 3]
        # y: center=3, generators=[1, 1], so y ∈ [3-1-1, 3+1+1] = [1, 5]

        np.testing.assert_allclose(lb, np.array([[1.0], [1.0]]), atol=1e-10)
        np.testing.assert_allclose(ub, np.array([[3.0], [5.0]]), atol=1e-10)

    def test_center_is_contained(self):
        """Test that center point is always contained in zonotope."""
        c = np.array([[2.0], [3.0]])
        V = np.array([[1.0, 0.0], [1.0, 1.0]])
        Z = Zono(c, V)

        # Center should always be in the zonotope (all alphas = 0)
        assert Z.contains(c)

    def test_corner_points(self):
        """Test that corner points are contained in zonotope."""
        c = np.array([[2.0], [3.0]])
        V = np.array([[1.0, 0.0], [1.0, 1.0]])
        Z = Zono(c, V)

        # Test corners (from vertex enumeration)
        corners = [
            np.array([[1.0], [1.0]]),  # ε₁=-1, ε₂=-1
            np.array([[3.0], [3.0]]),  # ε₁=+1, ε₂=-1
            np.array([[3.0], [5.0]]),  # ε₁=+1, ε₂=+1
            np.array([[1.0], [3.0]]),  # ε₁=-1, ε₂=+1
        ]

        for corner in corners:
            assert Z.contains(corner), f"Corner {corner.flatten()} should be in zonotope"

    def test_outside_points(self):
        """Test that points outside the zonotope are correctly identified."""
        c = np.array([[2.0], [3.0]])
        V = np.array([[1.0, 0.0], [1.0, 1.0]])
        Z = Zono(c, V)

        # Points clearly outside
        outside_points = [
            np.array([[0.0], [0.0]]),
            np.array([[5.0], [5.0]]),
            np.array([[0.0], [5.0]]),
            np.array([[5.0], [0.0]]),
        ]

        for point in outside_points:
            assert not Z.contains(point), f"Point {point.flatten()} should be outside zonotope"
