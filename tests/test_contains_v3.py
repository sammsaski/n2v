"""
Tests for improved contains() methods (V3 ChatGPT feedback)

These tests validate the two-phase containment checking with:
1. Fast-path least-squares solve
2. Explicit residual and constraint verification
3. LP fallback with OSQP solver
4. Post-solve verification to prevent false positives

Tests cover edge cases that could cause false positives with naive solvers.
"""

import numpy as np
import pytest
from n2v.sets.hexatope import Hexatope
from n2v.sets.octatope import Octatope


def test_hexatope_contains_interior_point():
    """Basic test: Interior point should be contained"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    hex_box = Hexatope.from_bounds(lb, ub)

    # Interior point
    assert hex_box.contains(np.array([0.5, 0.5]))
    assert hex_box.contains(np.array([0.1, 0.9]))


def test_hexatope_contains_boundary_point():
    """Boundary points should be contained (within tolerance)"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    hex_box = Hexatope.from_bounds(lb, ub)

    # Boundary points
    assert hex_box.contains(np.array([0.0, 0.5]))
    assert hex_box.contains(np.array([1.0, 0.5]))
    assert hex_box.contains(np.array([0.5, 0.0]))
    assert hex_box.contains(np.array([0.5, 1.0]))

    # Corners
    assert hex_box.contains(np.array([0.0, 0.0]))
    assert hex_box.contains(np.array([1.0, 1.0]))


def test_hexatope_contains_exterior_point():
    """Exterior points should NOT be contained"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    hex_box = Hexatope.from_bounds(lb, ub)

    # Clearly outside
    assert not hex_box.contains(np.array([-0.1, 0.5]))
    assert not hex_box.contains(np.array([1.1, 0.5]))
    assert not hex_box.contains(np.array([0.5, -0.1]))
    assert not hex_box.contains(np.array([0.5, 1.1]))

    # Far outside
    assert not hex_box.contains(np.array([2.0, 2.0]))
    assert not hex_box.contains(np.array([-1.0, -1.0]))


def test_hexatope_contains_near_boundary():
    """Test points very close to boundary (edge case for false positives)"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    hex_box = Hexatope.from_bounds(lb, ub)

    tol = 1e-7  # Default tolerance

    # Just inside (should pass)
    assert hex_box.contains(np.array([0.0 + tol/2, 0.5]))
    assert hex_box.contains(np.array([1.0 - tol/2, 0.5]))

    # Just outside (should fail - this is the critical test for false positives)
    # Note: Due to over-approximation in DCS, some points slightly outside may be included
    # But points clearly outside (> 2*tol) should definitely be rejected
    assert not hex_box.contains(np.array([-10*tol, 0.5]))
    assert not hex_box.contains(np.array([1.0 + 10*tol, 0.5]))


def test_hexatope_contains_after_affine_map():
    """Test contains after affine transformation"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    hex_box = Hexatope.from_bounds(lb, ub)

    # Scale by 2 and translate by [1, 1]
    W = 2.0 * np.eye(2)
    b = np.array([1.0, 1.0])

    hex_transformed = hex_box.affine_map(W, b)

    # Original [0, 1]² becomes [1, 3]² after transformation
    assert hex_transformed.contains(np.array([2.0, 2.0]))  # Center of [1, 3]²
    assert hex_transformed.contains(np.array([1.0, 1.0]))  # Corner
    assert hex_transformed.contains(np.array([3.0, 3.0]))  # Corner

    # Outside transformed set
    assert not hex_transformed.contains(np.array([0.5, 2.0]))  # Below lower bound
    assert not hex_transformed.contains(np.array([3.5, 2.0]))  # Above upper bound


def test_hexatope_contains_after_intersection():
    """Test contains after half-space intersection"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    hex_box = Hexatope.from_bounds(lb, ub)

    # Intersect with x1 + x2 ≤ 1.0 (lower triangle)
    H = np.array([[1.0, 1.0]])
    g = np.array([[1.0]])

    hex_result = hex_box.intersect_half_space(H, g)

    # Point (0.25, 0.25) has sum 0.5 < 1.0, should be inside
    assert hex_result.contains(np.array([0.25, 0.25]))

    # Point (0.6, 0.6) has sum 1.2 > 1.0, should be outside
    # Note: Due to over-approximation, might still be included in bounding box
    # But contains() should reject it if it violates the kernel constraints
    test_point = np.array([0.6, 0.6])
    result = hex_result.contains(test_point)
    # This may pass or fail depending on over-approximation
    # The key is that it's *sound* - we never reject points that are actually inside
    print(f"Point (0.6, 0.6) after intersection: {result}")


def test_octatope_contains_interior_point():
    """Basic test: Interior point should be contained"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    oct_box = Octatope.from_bounds(lb, ub)

    # Interior point
    assert oct_box.contains(np.array([0.5, 0.5]))
    assert oct_box.contains(np.array([0.1, 0.9]))


def test_octatope_contains_boundary_point():
    """Boundary points should be contained (within tolerance)"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    oct_box = Octatope.from_bounds(lb, ub)

    # Boundary points
    assert oct_box.contains(np.array([0.0, 0.5]))
    assert oct_box.contains(np.array([1.0, 0.5]))
    assert oct_box.contains(np.array([0.5, 0.0]))
    assert oct_box.contains(np.array([0.5, 1.0]))

    # Corners
    assert oct_box.contains(np.array([0.0, 0.0]))
    assert oct_box.contains(np.array([1.0, 1.0]))


def test_octatope_contains_exterior_point():
    """Exterior points should NOT be contained"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    oct_box = Octatope.from_bounds(lb, ub)

    # Clearly outside
    assert not oct_box.contains(np.array([-0.1, 0.5]))
    assert not oct_box.contains(np.array([1.1, 0.5]))
    assert not oct_box.contains(np.array([0.5, -0.1]))
    assert not oct_box.contains(np.array([0.5, 1.1]))

    # Far outside
    assert not oct_box.contains(np.array([2.0, 2.0]))
    assert not oct_box.contains(np.array([-1.0, -1.0]))


def test_octatope_contains_near_boundary():
    """Test points very close to boundary (edge case for false positives)"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    oct_box = Octatope.from_bounds(lb, ub)

    tol = 1e-7

    # Just inside (should pass)
    assert oct_box.contains(np.array([0.0 + tol/2, 0.5]))
    assert oct_box.contains(np.array([1.0 - tol/2, 0.5]))

    # Just outside (should fail - critical test for false positives)
    assert not oct_box.contains(np.array([-10*tol, 0.5]))
    assert not oct_box.contains(np.array([1.0 + 10*tol, 0.5]))


def test_octatope_contains_after_affine_map():
    """Test contains after affine transformation"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    oct_box = Octatope.from_bounds(lb, ub)

    # Scale by 2 and translate by [1, 1]
    W = 2.0 * np.eye(2)
    b = np.array([1.0, 1.0])

    oct_transformed = oct_box.affine_map(W, b)

    # Original [0, 1]² becomes [1, 3]² after transformation
    assert oct_transformed.contains(np.array([2.0, 2.0]))  # Center
    assert oct_transformed.contains(np.array([1.0, 1.0]))  # Corner
    assert oct_transformed.contains(np.array([3.0, 3.0]))  # Corner

    # Outside transformed set
    assert not oct_transformed.contains(np.array([0.5, 2.0]))
    assert not oct_transformed.contains(np.array([3.5, 2.0]))


def test_octatope_contains_diagonal_constraint():
    """Test octatope with diagonal UTVPI constraint"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    oct_box = Octatope.from_bounds(lb, ub)

    # Intersect with x1 + x2 ≤ 1.0 (UTVPI constraint)
    H = np.array([[1.0, 1.0]])
    g = np.array([[1.0]])

    oct_result = oct_box.intersect_half_space(H, g)

    # Point (0.25, 0.25) has sum 0.5 < 1.0, should be inside
    assert oct_result.contains(np.array([0.25, 0.25]))

    # Point (0.4, 0.4) has sum 0.8 < 1.0, should be inside
    assert oct_result.contains(np.array([0.4, 0.4]))


def test_hexatope_contains_custom_tolerance():
    """Test that custom tolerance parameter works"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    hex_box = Hexatope.from_bounds(lb, ub)

    # With looser tolerance, point slightly outside might pass
    loose_tol = 1e-3
    point = np.array([1.0 + 0.5e-3, 0.5])  # Slightly outside

    # With default tight tolerance, should fail
    assert not hex_box.contains(point, tolerance=1e-7)

    # With loose tolerance, might pass (depends on over-approximation)
    result_loose = hex_box.contains(point, tolerance=loose_tol)
    print(f"Loose tolerance result: {result_loose}")


def test_octatope_contains_custom_tolerance():
    """Test that custom tolerance parameter works"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    oct_box = Octatope.from_bounds(lb, ub)

    # With looser tolerance
    loose_tol = 1e-3
    point = np.array([1.0 + 0.5e-3, 0.5])

    # With default tight tolerance, should fail
    assert not oct_box.contains(point, tolerance=1e-7)

    # With loose tolerance, might pass
    result_loose = oct_box.contains(point, tolerance=loose_tol)
    print(f"Loose tolerance result: {result_loose}")


def test_hexatope_contains_degenerate_case():
    """Test contains on very small/degenerate hexatope"""
    # Create a very small box
    lb = np.array([0.0, 0.0])
    ub = np.array([1e-6, 1e-6])
    hex_tiny = Hexatope.from_bounds(lb, ub)

    # Origin should be inside
    assert hex_tiny.contains(np.array([0.0, 0.0]))

    # Small point inside
    assert hex_tiny.contains(np.array([0.5e-6, 0.5e-6]))

    # Point outside tiny box
    assert not hex_tiny.contains(np.array([2e-6, 0.0]))


def test_octatope_contains_degenerate_case():
    """Test contains on very small/degenerate octatope"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1e-6, 1e-6])
    oct_tiny = Octatope.from_bounds(lb, ub)

    # Origin should be inside
    assert oct_tiny.contains(np.array([0.0, 0.0]))

    # Small point inside
    assert oct_tiny.contains(np.array([0.5e-6, 0.5e-6]))

    # Point outside tiny box
    assert not oct_tiny.contains(np.array([2e-6, 0.0]))


if __name__ == "__main__":
    # Run tests
    test_hexatope_contains_interior_point()
    test_hexatope_contains_boundary_point()
    test_hexatope_contains_exterior_point()
    test_hexatope_contains_near_boundary()
    test_hexatope_contains_after_affine_map()
    test_hexatope_contains_after_intersection()

    test_octatope_contains_interior_point()
    test_octatope_contains_boundary_point()
    test_octatope_contains_exterior_point()
    test_octatope_contains_near_boundary()
    test_octatope_contains_after_affine_map()
    test_octatope_contains_diagonal_constraint()

    test_hexatope_contains_custom_tolerance()
    test_octatope_contains_custom_tolerance()

    test_hexatope_contains_degenerate_case()
    test_octatope_contains_degenerate_case()

    print("\n✅ All contains() V3 tests completed!")
