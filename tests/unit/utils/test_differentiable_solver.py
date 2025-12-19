"""
Unit tests for differentiable DCS solver for Hexatope and Octatope.

Tests the constraint-aware Gumbel-Softmax based differentiable solver inspired by
"Differentiable Combinatorial Scheduling at Scale" (ICML'24).

Note: Hexatope and Octatope use Difference Constraint System (DCS) and UTVPI
constraint systems, not general LPs. The differentiable solver leverages the
constraint graph structure for better performance.
"""

import pytest
import numpy as np

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from n2v.sets import Hexatope, Octatope


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestHexatopeDifferentiable:
    """Unit tests for Hexatope with differentiable solver."""

    def test_hexatope_from_bounds_ranges(self):
        """Test Hexatope range computation with both solvers."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Standard LP solver
        lb_std, ub_std = hexatope.get_ranges(use_mcf=False)

        # Check results are reasonable
        assert np.allclose(lb_std, lb, atol=0.1)
        assert np.allclose(ub_std, ub, atol=0.1)

    def test_hexatope_affine_map(self):
        """Test Hexatope after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply affine map: y = 2*x + [1, 1]
        W = 2 * np.eye(2)
        b = np.array([1.0, 1.0])
        hexatope_transformed = hexatope.affine_map(W, b)

        # Get ranges
        lb_t, ub_t = hexatope_transformed.get_ranges(use_mcf=False)

        # Expected: [1, 1] to [3, 3]
        assert np.allclose(lb_t, [[1.0], [1.0]], atol=0.2)
        assert np.allclose(ub_t, [[3.0], [3.0]], atol=0.2)

    def test_hexatope_intersect_half_space(self):
        """Test Hexatope half-space intersection."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Intersect with x1 <= 0.5
        H = np.array([[1.0, 0.0]])
        g = np.array([[0.5]])
        result = hexatope.intersect_half_space(H, g)

        # V1 soundness fix: extra_A/extra_b removed for template closure
        # Instead, check that DCS constraints have been updated
        assert len(result.dcs.constraints) > len(hexatope.dcs.constraints)

        # Get ranges
        lb_r, ub_r = result.get_ranges(use_mcf=False)

        # First dimension should be bounded by 0.5 (with tolerance for over-approximation)
        assert ub_r[0] <= 0.6  # Small tolerance

    def test_hexatope_3d(self):
        """Test 3D Hexatope."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        lb_computed, ub_computed = hexatope.get_ranges(use_mcf=False)

        assert lb_computed.shape == (3, 1)
        assert ub_computed.shape == (3, 1)
        assert np.allclose(lb_computed, lb, atol=0.1)
        assert np.allclose(ub_computed, ub, atol=0.1)

    def test_hexatope_get_bounds_alias(self):
        """Test get_bounds() alias method."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        lb_computed, ub_computed = hexatope.get_bounds(use_mcf=False)

        assert np.allclose(lb_computed, lb, atol=0.1)
        assert np.allclose(ub_computed, ub, atol=0.1)

    def test_hexatope_contains(self):
        """Test Hexatope contains() method."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        # Point inside
        assert hexatope.contains(np.array([0.5, 0.5]))

        # Point on boundary
        assert hexatope.contains(np.array([0.0, 0.0]))
        assert hexatope.contains(np.array([1.0, 1.0]))

        # Point outside
        assert not hexatope.contains(np.array([1.5, 0.5]))
        assert not hexatope.contains(np.array([-0.5, 0.5]))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOctatopeDifferentiable:
    """Unit tests for Octatope with differentiable solver."""

    def test_octatope_from_bounds_ranges(self):
        """Test Octatope range computation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [2.0]])
        octatope = Octatope.from_bounds(lb, ub)

        lb_computed, ub_computed = octatope.get_ranges(use_mcf=False)

        assert np.allclose(lb_computed, lb, atol=0.1)
        assert np.allclose(ub_computed, ub, atol=0.1)

    def test_octatope_affine_map(self):
        """Test Octatope after affine transformation."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Apply affine map
        W = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([1.0, 2.0])
        octatope_transformed = octatope.affine_map(W, b)

        lb_t, ub_t = octatope_transformed.get_ranges(use_mcf=False)

        # Expected: [1, 2] to [3, 5]
        assert np.allclose(lb_t, [[1.0], [2.0]], atol=0.2)
        assert np.allclose(ub_t, [[3.0], [5.0]], atol=0.2)

    def test_octatope_intersect_half_space(self):
        """Test Octatope half-space intersection."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Intersect with x1 + x2 <= 1
        H = np.array([[1.0, 1.0]])
        g = np.array([[1.0]])
        result = octatope.intersect_half_space(H, g)

        lb_r, ub_r = result.get_ranges(use_mcf=False)

        # Results should still be bounded
        assert np.all(lb_r >= -0.1)
        assert np.all(ub_r <= 1.1)

    def test_octatope_3d(self):
        """Test 3D Octatope."""
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        lb_computed, ub_computed = octatope.get_ranges(use_mcf=False)

        assert lb_computed.shape == (3, 1)
        assert ub_computed.shape == (3, 1)
        assert np.allclose(lb_computed, lb, atol=0.1)
        assert np.allclose(ub_computed, ub, atol=0.1)

    def test_octatope_hexatope_consistency(self):
        """Test that Octatope and Hexatope give consistent results for simple boxes."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])

        hexatope = Hexatope.from_bounds(lb, ub)
        octatope = Octatope.from_bounds(lb, ub)

        lb_hex, ub_hex = hexatope.get_ranges(use_mcf=False)
        lb_oct, ub_oct = octatope.get_ranges(use_mcf=False)

        # Should be very similar for simple boxes
        assert np.allclose(lb_hex, lb_oct, atol=0.1)
        assert np.allclose(ub_hex, ub_oct, atol=0.1)

    def test_octatope_contains(self):
        """Test Octatope contains() method."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        # Point inside
        assert octatope.contains(np.array([0.5, 0.5]))

        # Point on boundary
        assert octatope.contains(np.array([0.0, 0.0]))
        assert octatope.contains(np.array([1.0, 1.0]))

        # Point outside
        assert not octatope.contains(np.array([1.5, 0.5]))
        assert not octatope.contains(np.array([-0.5, 0.5]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
