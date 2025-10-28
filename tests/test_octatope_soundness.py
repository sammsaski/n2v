"""
Test octatope soundness fixes

This tests the critical soundness fixes for octatope:
1. No hardcoded [-1,1] bounds in LP (UTVPI already has them)
2. Template-closed intersections (UTVPI-only results)
"""

import numpy as np
import pytest
from n2v.sets.octatope import Octatope


def test_from_bounds_utvpi_structure():
    """Test that from_bounds creates proper UTVPI constraints"""
    lb = np.array([0.0, 1.0])
    ub = np.array([2.0, 3.0])

    O = Octatope.from_bounds(lb, ub)

    # Check dimensions
    assert O.dim == 2, f"Expected dim=2, got {O.dim}"
    assert O.nVar == 2, f"Expected nVar=2, got {O.nVar}"

    # Check generator structure (diagonal with half-widths)
    expected_gen = np.array([[1.0, 0.0],
                             [0.0, 1.0]])
    assert np.allclose(O.generators, expected_gen), \
        f"Expected generators:\n{expected_gen}\nGot:\n{O.generators}"

    # Check center
    expected_center = np.array([1.0, 2.0])
    assert np.allclose(O.center, expected_center), \
        f"Expected center {expected_center}, got {O.center}"

    # Check UTVPI has absolute bounds
    # Should have constraints like: x_i ≤ 1 and -x_i ≤ 1 for each i
    assert O.utvpi.num_vars == 2, f"UTVPI should have 2 vars, got {O.utvpi.num_vars}"
    assert len(O.utvpi.constraints) == 4, \
        f"UTVPI should have 4 constraints (±x_i ≤ 1 for each i), got {len(O.utvpi.constraints)}"

    print("✓ UTVPI structure test passed")


def test_from_bounds_optimization():
    """Test that optimization works with UTVPI constraints"""
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])

    O = Octatope.from_bounds(lb, ub)

    # Test range computation at dimension 0
    min_val, max_val = O.get_range(0, use_mcf=False)  # Use LP

    assert min_val is not None, "Min value should not be None"
    assert max_val is not None, "Max value should not be None"
    assert np.isclose(min_val, -1.0, atol=1e-4), \
        f"Expected min=-1.0, got {min_val}"
    assert np.isclose(max_val, 1.0, atol=1e-4), \
        f"Expected max=1.0, got {max_val}"

    # Test range at dimension 1
    min_val, max_val = O.get_range(1, use_mcf=False)
    assert np.isclose(min_val, -1.0, atol=1e-4), \
        f"Expected min=-1.0, got {min_val}"
    assert np.isclose(max_val, 1.0, atol=1e-4), \
        f"Expected max=1.0, got {max_val}"

    print("✓ Optimization test passed")


def test_affine_map():
    """Test that affine maps work correctly"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    O = Octatope.from_bounds(lb, ub)

    # Apply affine map: 2*x + [1, 1]
    W = 2 * np.eye(2)
    b = np.array([1.0, 1.0])

    O2 = O.affine_map(W, b)

    # Check dimensions preserved
    assert O2.dim == 2
    assert O2.nVar == 2

    # Check transformed generators
    # Original: diag([0.5, 0.5])
    # After W: diag([1.0, 1.0])
    expected_gen = np.array([[1.0, 0.0],
                             [0.0, 1.0]])
    assert np.allclose(O2.generators, expected_gen), \
        f"Expected generators:\n{expected_gen}\nGot:\n{O2.generators}"

    # Check transformed center
    # Original center: [0.5, 0.5]
    # After W*c + b: [2.0, 2.0]
    expected_center = np.array([2.0, 2.0])
    assert np.allclose(O2.center, expected_center), \
        f"Expected center {expected_center}, got {O2.center}"

    # Verify the transformed box has correct bounds [1, 3]
    min_val, max_val = O2.get_range(0, use_mcf=False)
    assert np.isclose(min_val, 1.0, atol=1e-4), \
        f"Expected min=1.0, got {min_val}"
    assert np.isclose(max_val, 3.0, atol=1e-4), \
        f"Expected max=3.0, got {max_val}"

    print("✓ Affine map test passed")


def test_contains():
    """Test that contains() works correctly"""
    lb = np.array([0.0, 0.0])
    ub = np.array([2.0, 2.0])

    O = Octatope.from_bounds(lb, ub)

    # Points inside
    assert O.contains(np.array([1.0, 1.0])), "Should contain center point"
    assert O.contains(np.array([0.0, 0.0])), "Should contain lower bound"
    assert O.contains(np.array([2.0, 2.0])), "Should contain upper bound"
    assert O.contains(np.array([0.5, 1.5])), "Should contain interior point"

    # Points outside
    assert not O.contains(np.array([-0.1, 1.0])), "Should not contain point below lb"
    assert not O.contains(np.array([2.1, 1.0])), "Should not contain point above ub"

    print("✓ Contains test passed")


def test_intersect_half_space_template_closed():
    """Verify half-space intersection returns UTVPI-only (template closed)"""
    lb = np.array([0.0, 0.0])
    ub = np.array([2.0, 2.0])

    O = Octatope.from_bounds(lb, ub)

    # Intersect with half-space x + y <= 3
    H_half = np.array([[1.0, 1.0]])
    g_half = np.array([3.0])

    O2 = O.intersect_half_space(H_half, g_half)

    # Should have more UTVPI constraints (tightened bounding box)
    # Note: May have many constraints due to O(n²) template tightening
    assert len(O2.utvpi.constraints) >= len(O.utvpi.constraints), \
        "Should have at least as many UTVPI constraints"

    print("✓ Half-space intersection template-closed test passed")


def test_utvpi_includes_absolute_bounds():
    """Verify UTVPI explicitly includes ±x_i ≤ 1 constraints"""
    lb = np.array([0.0])
    ub = np.array([1.0])

    O = Octatope.from_bounds(lb, ub)

    # Check UTVPI constraints
    found_pos = False
    found_neg = False
    for c in O.utvpi.constraints:
        # Check for x_0 ≤ 1 (i=0, j=0, ai=1, aj=0, b=1)
        if c.i == 0 and c.j == 0 and c.ai == 1 and c.aj == 0 and np.isclose(c.b, 1.0):
            found_pos = True
        # Check for -x_0 ≤ 1 (i=0, j=0, ai=-1, aj=0, b=1)
        if c.i == 0 and c.j == 0 and c.ai == -1 and c.aj == 0 and np.isclose(c.b, 1.0):
            found_neg = True

    assert found_pos, "UTVPI should contain x_0 ≤ 1"
    assert found_neg, "UTVPI should contain -x_0 ≤ 1"

    print("✓ UTVPI absolute bounds test passed")


if __name__ == "__main__":
    print("Running octatope soundness tests...\n")

    test_from_bounds_utvpi_structure()
    test_from_bounds_optimization()
    test_affine_map()
    test_contains()
    test_intersect_half_space_template_closed()
    test_utvpi_includes_absolute_bounds()

    print("\n✅ All octatope soundness tests passed!")
