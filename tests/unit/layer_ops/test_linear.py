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

class TestLinearReach:
    """Tests for Linear layer reachability."""

    def test_linear_star(self, simple_star):
        """Test Linear layer with Star."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = linear_reach.linear_star(layer, [simple_star])

        assert len(result) == 1
        assert result[0].dim == 2
        pytest.assert_star_valid(result[0])

    def test_linear_zono(self, simple_zono):
        """Test Linear layer with Zono."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = linear_reach.linear_zono(layer, [simple_zono])

        assert len(result) == 1
        assert result[0].dim == 2
        pytest.assert_zono_valid(result[0])

    def test_linear_box(self, simple_box):
        """Test Linear layer with Box."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = linear_reach.linear_box(layer, [simple_box])

        assert len(result) == 1
        assert result[0].dim == 2

    def test_linear_preserves_exactness(self):
        """Test that Linear is exact (no over-approximation)."""
        layer = nn.Linear(2, 2)
        # Set to identity + constant
        with torch.no_grad():
            layer.weight.data = torch.eye(2)
            layer.bias.data = torch.ones(2)

        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        star = Star.from_bounds(lb, ub)

        result = linear_reach.linear_star(layer, [star])[0]

        # Expected: [0, 1] + 1 = [1, 2]
        result.estimate_ranges()
        np.testing.assert_allclose(result.state_lb, np.array([[1.0], [1.0]]), atol=1e-5)
        np.testing.assert_allclose(result.state_ub, np.array([[2.0], [2.0]]), atol=1e-5)

    def test_linear_hexatope(self, simple_hexatope):
        """Test Linear layer with Hexatope."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = linear_reach.linear_hexatope(layer, [simple_hexatope])

        assert len(result) == 1
        assert result[0].dim == 2
        pytest.assert_hexatope_valid(result[0])

    def test_linear_hexatope_identity(self):
        """Test Linear layer with Hexatope using identity transformation."""
        layer = nn.Linear(3, 3)
        with torch.no_grad():
            layer.weight.data = torch.eye(3)
            layer.bias.data = torch.zeros(3)

        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        result = linear_reach.linear_hexatope(layer, [hexatope])[0]

        # Identity should preserve bounds
        result_lb, result_ub = result.get_bounds()
        np.testing.assert_allclose(result_lb, lb, atol=1e-5)
        np.testing.assert_allclose(result_ub, ub, atol=1e-5)

    def test_linear_hexatope_dimension_change(self):
        """Test Linear layer with Hexatope changing dimensions."""
        layer = nn.Linear(3, 2)
        with torch.no_grad():
            # Simple projection: output = [x0, x1]
            layer.weight.data = torch.tensor([[1.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0]])
            layer.bias.data = torch.zeros(2)

        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        hexatope = Hexatope.from_bounds(lb, ub)

        result = linear_reach.linear_hexatope(layer, [hexatope])[0]

        assert result.dim == 2
        result_lb, result_ub = result.get_bounds()
        np.testing.assert_allclose(result_lb, np.array([[0.0], [1.0]]), atol=1e-5)
        np.testing.assert_allclose(result_ub, np.array([[1.0], [2.0]]), atol=1e-5)

    def test_linear_octatope(self, simple_octatope):
        """Test Linear layer with Octatope."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = linear_reach.linear_octatope(layer, [simple_octatope])

        assert len(result) == 1
        assert result[0].dim == 2
        pytest.assert_octatope_valid(result[0])

    def test_linear_octatope_identity(self):
        """Test Linear layer with Octatope using identity transformation."""
        layer = nn.Linear(3, 3)
        with torch.no_grad():
            layer.weight.data = torch.eye(3)
            layer.bias.data = torch.zeros(3)

        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        result = linear_reach.linear_octatope(layer, [octatope])[0]

        # Identity should preserve bounds
        result_lb, result_ub = result.get_bounds()
        np.testing.assert_allclose(result_lb, lb, atol=1e-5)
        np.testing.assert_allclose(result_ub, ub, atol=1e-5)

    def test_linear_octatope_dimension_change(self):
        """Test Linear layer with Octatope changing dimensions."""
        layer = nn.Linear(3, 2)
        with torch.no_grad():
            # Simple projection: output = [x0, x1]
            layer.weight.data = torch.tensor([[1.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0]])
            layer.bias.data = torch.zeros(2)

        lb = np.array([[0.0], [1.0], [2.0]])
        ub = np.array([[1.0], [2.0], [3.0]])
        octatope = Octatope.from_bounds(lb, ub)

        result = linear_reach.linear_octatope(layer, [octatope])[0]

        assert result.dim == 2
        result_lb, result_ub = result.get_bounds()
        np.testing.assert_allclose(result_lb, np.array([[0.0], [1.0]]), atol=1e-5)
        np.testing.assert_allclose(result_ub, np.array([[1.0], [2.0]]), atol=1e-5)

    def test_linear_octatope_with_bias(self):
        """Test Linear layer with Octatope including bias translation."""
        layer = nn.Linear(2, 2)
        with torch.no_grad():
            layer.weight.data = torch.eye(2)
            layer.bias.data = torch.tensor([1.0, 2.0])

        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        octatope = Octatope.from_bounds(lb, ub)

        result = linear_reach.linear_octatope(layer, [octatope])[0]

        # Expected: [0, 1] + [1, 2] = [1, 2] to [2, 3]
        result_lb, result_ub = result.get_bounds()
        np.testing.assert_allclose(result_lb, np.array([[1.0], [2.0]]), atol=1e-5)
        np.testing.assert_allclose(result_ub, np.array([[2.0], [3.0]]), atol=1e-5)


