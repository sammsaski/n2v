"""Tests for layer operations."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box, ImageStar, ImageZono, Hexatope, Octatope
from n2v.nn.layer_ops import (
    linear_reach, relu_reach, conv2d_reach,
    maxpool2d_reach, avgpool2d_reach, flatten_reach
)

class TestReLUReach:
    """Tests for ReLU activation reachability."""

    def test_relu_star_exact_always_active(self):
        """Test ReLU with always active neuron."""
        # Star with all positive bounds
        lb = np.array([[1.0], [2.0]])
        ub = np.array([[2.0], [3.0]])
        star = Star.from_bounds(lb, ub)

        result = relu_reach.relu_star_exact([star])

        # Should not split
        assert len(result) == 1
        pytest.assert_star_valid(result[0])

    def test_relu_star_exact_always_inactive(self):
        """Test ReLU with always inactive neuron."""
        # Star with all negative bounds
        lb = np.array([[-2.0], [-3.0]])
        ub = np.array([[-1.0], [-2.0]])
        star = Star.from_bounds(lb, ub)

        result = relu_reach.relu_star_exact([star])

        # Should not split, but zero out
        assert len(result) == 1
        result[0].estimate_ranges()
        np.testing.assert_allclose(result[0].state_lb, np.zeros((2, 1)), atol=1e-6)
        np.testing.assert_allclose(result[0].state_ub, np.zeros((2, 1)), atol=1e-6)

    def test_relu_star_exact_splitting(self):
        """Test ReLU with uncertain neuron (causes splitting)."""
        # Star with crossing zero
        lb = np.array([[-1.0], [1.0]])
        ub = np.array([[1.0], [2.0]])
        star = Star.from_bounds(lb, ub)

        result = relu_reach.relu_star_exact([star])

        # Should split on first neuron
        assert len(result) == 2  # One for active, one for inactive
        for s in result:
            pytest.assert_star_valid(s)

    def test_relu_zono_approx(self, simple_zono):
        """Test ReLU with Zonotope (over-approximation)."""
        result = relu_reach.relu_zono_approx([simple_zono])

        assert len(result) == 1
        pytest.assert_zono_valid(result[0])

    def test_relu_box(self):
        """Test ReLU with Box."""
        lb = np.array([[-1.0], [-2.0], [1.0]])
        ub = np.array([[1.0], [0.5], [2.0]])
        box = Box(lb, ub)

        result = relu_reach.relu_box([box])

        assert len(result) == 1
        # Check ReLU applied correctly (Box has .lb and .ub attributes)
        result_lb = result[0].lb
        result_ub = result[0].ub
        np.testing.assert_allclose(result_lb, np.array([[0.0], [0.0], [1.0]]), atol=1e-6)
        np.testing.assert_allclose(result_ub, np.array([[1.0], [0.5], [2.0]]), atol=1e-6)

    def test_relu_hexatope_approx_always_active(self):
        """Test ReLU approximation with Hexatope for always active neurons."""
        lb = np.array([[1.0], [2.0], [3.0]])
        ub = np.array([[2.0], [3.0], [4.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        result = relu_reach.relu_hexatope_approx([hexatope])

        assert len(result) == 1
        pytest.assert_hexatope_valid(result[0])
        # Bounds should be preserved (all positive)
        result_lb, result_ub = result[0].get_bounds()
        np.testing.assert_allclose(result_lb, lb, atol=1e-5)
        np.testing.assert_allclose(result_ub, ub, atol=1e-5)

    def test_relu_hexatope_approx_always_inactive(self):
        """Test ReLU approximation with Hexatope for always inactive neurons."""
        lb = np.array([[-2.0], [-3.0], [-4.0]])
        ub = np.array([[-1.0], [-2.0], [-3.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        result = relu_reach.relu_hexatope_approx([hexatope])

        assert len(result) == 1
        pytest.assert_hexatope_valid(result[0])
        # All outputs should be zero
        result_lb, result_ub = result[0].get_bounds()
        np.testing.assert_allclose(result_lb, np.zeros((3, 1)), atol=1e-6)
        np.testing.assert_allclose(result_ub, np.zeros((3, 1)), atol=1e-6)

    def test_relu_hexatope_approx_mixed(self):
        """Test ReLU approximation with Hexatope for mixed active/inactive/uncertain neurons."""
        lb = np.array([[-1.0], [-2.0], [1.0]])
        ub = np.array([[1.0], [0.5], [2.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        result = relu_reach.relu_hexatope_approx([hexatope])

        assert len(result) == 1
        pytest.assert_hexatope_valid(result[0])
        # First two uncertain/inactive -> approximated, third always active
        result_lb, result_ub = result[0].get_bounds()
        # First dimension: [-1, 1] -> approx [0, 1]
        assert result_lb[0, 0] >= -1e-6
        assert result_ub[0, 0] <= 1.0 + 1e-6
        # Second dimension: [-2, 0.5] -> approx [0, 0.5]
        assert result_lb[1, 0] >= -1e-6
        assert result_ub[1, 0] <= 0.5 + 1e-6
        # Third dimension: [1, 2] -> exact [1, 2]
        np.testing.assert_allclose(result_lb[2, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(result_ub[2, 0], 2.0, atol=1e-5)

    def test_relu_hexatope_exact_always_active(self):
        """Test ReLU exact with Hexatope for always active neurons."""
        lb = np.array([[1.0], [2.0]])
        ub = np.array([[2.0], [3.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        result = relu_reach.relu_hexatope_exact([hexatope])

        # Should not split
        assert len(result) == 1
        pytest.assert_hexatope_valid(result[0])
        result_lb, result_ub = result[0].get_bounds()
        np.testing.assert_allclose(result_lb, lb, atol=1e-5)
        np.testing.assert_allclose(result_ub, ub, atol=1e-5)

    def test_relu_hexatope_exact_always_inactive(self):
        """Test ReLU exact with Hexatope for always inactive neurons."""
        lb = np.array([[-2.0], [-3.0]])
        ub = np.array([[-1.0], [-2.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        result = relu_reach.relu_hexatope_exact([hexatope])

        # Should not split, but zero out
        assert len(result) == 1
        pytest.assert_hexatope_valid(result[0])
        result_lb, result_ub = result[0].get_bounds()
        np.testing.assert_allclose(result_lb, np.zeros((2, 1)), atol=1e-6)
        np.testing.assert_allclose(result_ub, np.zeros((2, 1)), atol=1e-6)

    def test_relu_hexatope_exact_splitting(self):
        """Test ReLU exact with Hexatope for uncertain neuron (causes splitting)."""
        lb = np.array([[-1.0], [1.0]])
        ub = np.array([[1.0], [2.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        result = relu_reach.relu_hexatope_exact([hexatope])

        # Should split on first neuron
        assert len(result) == 2
        for h in result:
            pytest.assert_hexatope_valid(h)

    def test_relu_hexatope_exact_vs_approx(self):
        """Test that exact ReLU contains approximate ReLU (soundness)."""
        lb = np.array([[-1.0], [-0.5]])
        ub = np.array([[1.0], [0.5]])
        hexatope = Hexatope.from_bounds(lb, ub)

        exact_result = relu_reach.relu_hexatope_exact([hexatope])
        approx_result = relu_reach.relu_hexatope_approx([hexatope])

        # Get bounds for all exact results
        exact_lb_all = []
        exact_ub_all = []
        for h in exact_result:
            h_lb, h_ub = h.get_bounds()
            exact_lb_all.append(h_lb)
            exact_ub_all.append(h_ub)

        # Compute convex hull of exact results
        exact_lb = np.min(exact_lb_all, axis=0)
        exact_ub = np.max(exact_ub_all, axis=0)

        # Get approx bounds
        approx_lb, approx_ub = approx_result[0].get_bounds()

        # Exact should contain approx (approx should be within exact bounds)
        assert np.all(exact_lb <= approx_lb + 1e-5)
        assert np.all(exact_ub >= approx_ub - 1e-5)

    def test_relu_octatope_approx_always_active(self):
        """Test ReLU approximation with Octatope for always active neurons."""
        lb = np.array([[1.0], [2.0], [3.0]])
        ub = np.array([[2.0], [3.0], [4.0]])
        octatope = Octatope.from_bounds(lb, ub)

        result = relu_reach.relu_octatope_approx([octatope])

        assert len(result) == 1
        pytest.assert_octatope_valid(result[0])
        # Bounds should be preserved (all positive)
        result_lb, result_ub = result[0].get_bounds()
        np.testing.assert_allclose(result_lb, lb, atol=1e-5)
        np.testing.assert_allclose(result_ub, ub, atol=1e-5)

    def test_relu_octatope_approx_always_inactive(self):
        """Test ReLU approximation with Octatope for always inactive neurons."""
        lb = np.array([[-2.0], [-3.0], [-4.0]])
        ub = np.array([[-1.0], [-2.0], [-3.0]])
        octatope = Octatope.from_bounds(lb, ub)

        result = relu_reach.relu_octatope_approx([octatope])

        assert len(result) == 1
        pytest.assert_octatope_valid(result[0])
        # All outputs should be zero
        result_lb, result_ub = result[0].get_bounds()
        np.testing.assert_allclose(result_lb, np.zeros((3, 1)), atol=1e-6)
        np.testing.assert_allclose(result_ub, np.zeros((3, 1)), atol=1e-6)

    def test_relu_octatope_approx_mixed(self):
        """Test ReLU approximation with Octatope for mixed active/inactive/uncertain neurons."""
        lb = np.array([[-1.0], [-2.0], [1.0]])
        ub = np.array([[1.0], [0.5], [2.0]])
        octatope = Octatope.from_bounds(lb, ub)

        result = relu_reach.relu_octatope_approx([octatope])

        assert len(result) == 1
        pytest.assert_octatope_valid(result[0])
        # First two uncertain/inactive -> approximated, third always active
        result_lb, result_ub = result[0].get_bounds()
        # First dimension: [-1, 1] -> approx [0, 1]
        assert result_lb[0, 0] >= -1e-6
        assert result_ub[0, 0] <= 1.0 + 1e-6
        # Second dimension: [-2, 0.5] -> approx [0, 0.5]
        assert result_lb[1, 0] >= -1e-6
        assert result_ub[1, 0] <= 0.5 + 1e-6
        # Third dimension: [1, 2] -> exact [1, 2]
        np.testing.assert_allclose(result_lb[2, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(result_ub[2, 0], 2.0, atol=1e-5)

    def test_relu_octatope_exact_always_active(self):
        """Test ReLU exact with Octatope for always active neurons."""
        lb = np.array([[1.0], [2.0]])
        ub = np.array([[2.0], [3.0]])
        octatope = Octatope.from_bounds(lb, ub)

        result = relu_reach.relu_octatope_exact([octatope])

        # Should not split
        assert len(result) == 1
        pytest.assert_octatope_valid(result[0])
        result_lb, result_ub = result[0].get_bounds()
        np.testing.assert_allclose(result_lb, lb, atol=1e-5)
        np.testing.assert_allclose(result_ub, ub, atol=1e-5)

    def test_relu_octatope_exact_always_inactive(self):
        """Test ReLU exact with Octatope for always inactive neurons."""
        lb = np.array([[-2.0], [-3.0]])
        ub = np.array([[-1.0], [-2.0]])
        octatope = Octatope.from_bounds(lb, ub)

        result = relu_reach.relu_octatope_exact([octatope])

        # Should not split, but zero out
        assert len(result) == 1
        pytest.assert_octatope_valid(result[0])
        result_lb, result_ub = result[0].get_bounds()
        np.testing.assert_allclose(result_lb, np.zeros((2, 1)), atol=1e-6)
        np.testing.assert_allclose(result_ub, np.zeros((2, 1)), atol=1e-6)

    def test_relu_octatope_exact_splitting(self):
        """Test ReLU exact with Octatope for uncertain neuron (causes splitting)."""
        lb = np.array([[-1.0], [1.0]])
        ub = np.array([[1.0], [2.0]])
        octatope = Octatope.from_bounds(lb, ub)

        result = relu_reach.relu_octatope_exact([octatope])

        # Should split on first neuron
        assert len(result) == 2
        for o in result:
            pytest.assert_octatope_valid(o)

    def test_relu_octatope_exact_vs_approx(self):
        """Test that exact ReLU contains approximate ReLU (soundness)."""
        lb = np.array([[-1.0], [-0.5]])
        ub = np.array([[1.0], [0.5]])
        octatope = Octatope.from_bounds(lb, ub)

        exact_result = relu_reach.relu_octatope_exact([octatope])
        approx_result = relu_reach.relu_octatope_approx([octatope])

        # Get bounds for all exact results
        exact_lb_all = []
        exact_ub_all = []
        for o in exact_result:
            o_lb, o_ub = o.get_bounds()
            exact_lb_all.append(o_lb)
            exact_ub_all.append(o_ub)

        # Compute convex hull of exact results
        exact_lb = np.min(exact_lb_all, axis=0)
        exact_ub = np.max(exact_ub_all, axis=0)

        # Get approx bounds
        approx_lb, approx_ub = approx_result[0].get_bounds()

        # Exact should contain approx (approx should be within exact bounds)
        assert np.all(exact_lb <= approx_lb + 1e-5)
        assert np.all(exact_ub >= approx_ub - 1e-5)

    def test_relu_hexatope_vs_octatope_consistency(self):
        """Test that Hexatope and Octatope ReLU produce consistent results."""
        lb = np.array([[0.5], [1.5]])
        ub = np.array([[1.5], [2.5]])
        hexatope = Hexatope.from_bounds(lb, ub)
        octatope = Octatope.from_bounds(lb, ub)

        # Both exact methods should produce same bounds (all active)
        hexa_result = relu_reach.relu_hexatope_exact([hexatope])[0]
        octa_result = relu_reach.relu_octatope_exact([octatope])[0]

        hexa_lb, hexa_ub = hexa_result.get_bounds()
        octa_lb, octa_ub = octa_result.get_bounds()

        np.testing.assert_allclose(hexa_lb, octa_lb, atol=1e-5)
        np.testing.assert_allclose(hexa_ub, octa_ub, atol=1e-5)

        # Both approx methods should also be consistent
        hexa_approx = relu_reach.relu_hexatope_approx([hexatope])[0]
        octa_approx = relu_reach.relu_octatope_approx([octatope])[0]

        hexa_approx_lb, hexa_approx_ub = hexa_approx.get_bounds()
        octa_approx_lb, octa_approx_ub = octa_approx.get_bounds()

        np.testing.assert_allclose(hexa_approx_lb, octa_approx_lb, atol=1e-5)
        np.testing.assert_allclose(hexa_approx_ub, octa_approx_ub, atol=1e-5)


