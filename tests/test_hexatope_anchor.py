"""
Test hexatope anchor variable implementation

This tests the critical soundness fixes:
1. Anchor variable for absolute bounds
2. No extra_A/extra_b
3. No hardcoded [-1,1] bounds in LP
"""

import numpy as np
import pytest
from n2v.sets.hexatope import Hexatope


def test_from_bounds_anchor_structure():
    """Test that from_bounds creates proper anchor variable structure"""
    lb = np.array([0.0, 1.0])
    ub = np.array([2.0, 3.0])

    H = Hexatope.from_bounds(lb, ub)

    # Check dimensions
    assert H.dim == 2, f"Expected dim=2, got {H.dim}"
    assert H.nVar == 3, f"Expected nVar=3 (anchor + 2 vars), got {H.nVar}"

    # Check generator structure
    # Column 0 should be zero (anchor)
    assert np.allclose(H.generators[:, 0], 0), "Anchor column should be zero"

    # Columns 1-2 should be diagonal half-widths
    expected_gen = np.array([[1.0, 0.0],
                             [0.0, 1.0]])
    assert np.allclose(H.generators[:, 1:], expected_gen), \
        f"Expected generators:\n{expected_gen}\nGot:\n{H.generators[:, 1:]}"

    # Check center
    expected_center = np.array([1.0, 2.0])
    assert np.allclose(H.center, expected_center), \
        f"Expected center {expected_center}, got {H.center}"

    # Check DCS has anchor bounds
    # Should have constraints like: x_1 - x_0 <= 1, x_0 - x_1 <= 1, etc.
    assert H.dcs.num_vars == 3, f"DCS should have 3 vars, got {H.dcs.num_vars}"
    assert len(H.dcs.constraints) == 4, \
        f"DCS should have 4 constraints (2 per var), got {len(H.dcs.constraints)}"

    print("✓ Anchor structure test passed")


def test_from_bounds_optimization():
    """Test that optimization works with anchor variable"""
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])

    H = Hexatope.from_bounds(lb, ub)

    # Test range computation at dimension 0
    min_val, max_val = H.get_range(0, use_mcf=False)  # Use LP for now

    assert min_val is not None, "Min value should not be None"
    assert max_val is not None, "Max value should not be None"
    assert np.isclose(min_val, -1.0, atol=1e-4), \
        f"Expected min=-1.0, got {min_val}"
    assert np.isclose(max_val, 1.0, atol=1e-4), \
        f"Expected max=1.0, got {max_val}"

    # Test range at dimension 1
    min_val, max_val = H.get_range(1, use_mcf=False)
    assert np.isclose(min_val, -1.0, atol=1e-4), \
        f"Expected min=-1.0, got {min_val}"
    assert np.isclose(max_val, 1.0, atol=1e-4), \
        f"Expected max=1.0, got {max_val}"

    print("✓ Optimization test passed")


def test_affine_map_preserves_anchor():
    """Test that affine maps preserve anchor structure"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    H = Hexatope.from_bounds(lb, ub)

    # Apply affine map: 2*x + [1, 1]
    W = 2 * np.eye(2)
    b = np.array([1.0, 1.0])

    H2 = H.affine_map(W, b)

    # Check dimensions preserved
    assert H2.dim == 2
    assert H2.nVar == 3, "Affine map should preserve nVar (including anchor)"

    # Check anchor column still zero
    assert np.allclose(H2.generators[:, 0], 0), \
        "Affine map should preserve zero anchor column"

    # Check transformed generators
    # Original: diag([0.5, 0.5])
    # After W: diag([1.0, 1.0])
    expected_gen = np.array([[1.0, 0.0],
                             [0.0, 1.0]])
    assert np.allclose(H2.generators[:, 1:], expected_gen), \
        f"Expected generators:\n{expected_gen}\nGot:\n{H2.generators[:, 1:]}"

    # Check transformed center
    # Original center: [0.5, 0.5]
    # After W*c + b: [2.0, 2.0]
    expected_center = np.array([2.0, 2.0])
    assert np.allclose(H2.center, expected_center), \
        f"Expected center {expected_center}, got {H2.center}"

    # Verify the transformed box has correct bounds [1, 3]
    min_val, max_val = H2.get_range(0, use_mcf=False)
    assert np.isclose(min_val, 1.0, atol=1e-4), \
        f"Expected min=1.0, got {min_val}"
    assert np.isclose(max_val, 3.0, atol=1e-4), \
        f"Expected max=3.0, got {max_val}"

    print("✓ Affine map test passed")


def test_contains_with_anchor():
    """Test that contains() works correctly with anchor"""
    lb = np.array([0.0, 0.0])
    ub = np.array([2.0, 2.0])

    H = Hexatope.from_bounds(lb, ub)

    # Point inside
    assert H.contains(np.array([1.0, 1.0])), "Should contain center point"
    assert H.contains(np.array([0.0, 0.0])), "Should contain lower bound"
    assert H.contains(np.array([2.0, 2.0])), "Should contain upper bound"
    assert H.contains(np.array([0.5, 1.5])), "Should contain interior point"

    # Points outside
    assert not H.contains(np.array([-0.1, 1.0])), "Should not contain point below lb"
    assert not H.contains(np.array([2.1, 1.0])), "Should not contain point above ub"

    print("✓ Contains test passed")


def test_no_extra_constraints():
    """Verify extra_A and extra_b are gone"""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    H = Hexatope.from_bounds(lb, ub)

    # These attributes should not exist
    assert not hasattr(H, 'extra_A'), "extra_A should be removed"
    assert not hasattr(H, 'extra_b'), "extra_b should be removed"

    print("✓ No extra constraints test passed")


def test_intersect_half_space_template_closed():
    """Verify half-space intersection returns DCS-only (no extra constraints)"""
    lb = np.array([0.0, 0.0])
    ub = np.array([2.0, 2.0])

    H = Hexatope.from_bounds(lb, ub)

    # Intersect with half-space x + y <= 3
    H_half = np.array([[1.0, 1.0]])
    g_half = np.array([3.0])

    H2 = H.intersect_half_space(H_half, g_half)

    # Should still have no extra constraints
    assert not hasattr(H2, 'extra_A'), "Result should have no extra_A"
    assert not hasattr(H2, 'extra_b'), "Result should have no extra_b"

    # Should have more DCS constraints (tightened bounding box)
    assert len(H2.dcs.constraints) >= len(H.dcs.constraints), \
        "Should have at least as many DCS constraints"

    print("✓ Half-space intersection template-closed test passed")


if __name__ == "__main__":
    print("Running hexatope anchor variable tests...\n")

    test_from_bounds_anchor_structure()
    test_from_bounds_optimization()
    test_affine_map_preserves_anchor()
    test_contains_with_anchor()
    test_no_extra_constraints()
    test_intersect_half_space_template_closed()

    print("\n✅ All hexatope anchor tests passed!")
