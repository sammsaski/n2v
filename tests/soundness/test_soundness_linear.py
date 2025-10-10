"""
Soundness tests for Linear layer reachability.

These tests verify that Linear layer reachability produces mathematically correct results
for all set representations (Star, Zono, Box) by comparing against known ground truth.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.linear_reach import linear_star, linear_zono, linear_box


class TestLinearStarSoundness:
    """Soundness tests for Linear layer with Star sets."""

    def test_identity_transformation(self):
        """Test identity transformation: y = x."""
        # Input: [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Layer: y = x (identity, no bias)
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: output should be identical to input
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        assert np.allclose(output_lb, lb, atol=1e-6)
        assert np.allclose(output_ub, ub, atol=1e-6)

    def test_translation(self):
        """Test pure translation: y = x + b."""
        # Input: [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Layer: y = x + [2, 3]
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.tensor([2.0, 3.0])

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: [0,1] + 2 = [2,3], [0,1] + 3 = [3,4]
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected_lb = np.array([[2.0], [3.0]])
        expected_ub = np.array([[3.0], [4.0]])

        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_scaling(self):
        """Test scaling transformation: y = 2x."""
        # Input: [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Layer: y = 2x (scale by 2)
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2) * 2
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: [0,1] * 2 = [0,2]
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[2.0], [2.0]])

        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_rotation_90deg(self):
        """Test 90-degree rotation in 2D."""
        # Input: [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Layer: 90-degree counterclockwise rotation
        # [0, -1]   [x]   [-y]
        # [1,  0] @ [y] = [x]
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: rotation of [0,1]x[0,1] -> [-1,0]x[0,1]
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected_lb = np.array([[-1.0], [0.0]])
        expected_ub = np.array([[0.0], [1.0]])

        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_dimension_reduction(self):
        """Test dimension reduction: R^3 -> R^2."""
        # Input: [0,1]^3
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Layer: project to first two dimensions and sum third
        # y1 = x1, y2 = x2 + x3
        layer = nn.Linear(3, 2)
        layer.weight.data = torch.tensor([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 1.0]])
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: y1 in [0,1], y2 in [0,2]
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1.0], [2.0]])

        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_dimension_expansion(self):
        """Test dimension expansion: R^2 -> R^3."""
        # Input: [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Layer: expand to 3D
        # y1 = x1, y2 = x2, y3 = x1 + x2
        layer = nn.Linear(2, 3)
        layer.weight.data = torch.tensor([[1.0, 0.0],
                                          [0.0, 1.0],
                                          [1.0, 1.0]])
        layer.bias.data = torch.zeros(3)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: y1 in [0,1], y2 in [0,1], y3 in [0,2]
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected_lb = np.array([[0.0], [0.0], [0.0]])
        expected_ub = np.array([[1.0], [1.0], [2.0]])

        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_negative_weights(self):
        """Test transformation with negative weights."""
        # Input: [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Layer: y = -x
        layer = nn.Linear(2, 2)
        layer.weight.data = -torch.eye(2)
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: -[0,1] = [-1,0]
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected_lb = np.array([[-1.0], [-1.0]])
        expected_ub = np.array([[0.0], [0.0]])

        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)

    def test_no_bias_layer(self):
        """Test layer with no bias term."""
        # Input: [1,2] x [1,2]
        lb = np.array([[1.0], [1.0]])
        ub = np.array([[2.0], [2.0]])
        input_star = Star.from_bounds(lb, ub)

        # Create layer without bias
        layer = nn.Linear(2, 2, bias=False)
        layer.weight.data = torch.eye(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: output should equal input (identity with no bias)
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        assert np.allclose(output_lb, lb, atol=1e-6)
        assert np.allclose(output_ub, ub, atol=1e-6)

    def test_zero_input_range(self):
        """Test with zero-width input (single point)."""
        # Input: single point [0.5, 0.5]
        lb = np.array([[0.5], [0.5]])
        ub = np.array([[0.5], [0.5]])
        input_star = Star.from_bounds(lb, ub)

        # Layer: y = 2x + 1
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2) * 2
        layer.bias.data = torch.ones(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: 2*0.5 + 1 = 2
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected = np.array([[2.0], [2.0]])
        assert np.allclose(output_lb, expected, atol=1e-6)
        assert np.allclose(output_ub, expected, atol=1e-6)

    def test_constrained_input_star(self):
        """Test with constrained Star (not just a box)."""
        # Create a Star representing x1 + x2 <= 1, x1,x2 >= 0, x1,x2 <= 1
        # This is a triangle, not a box
        V = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])  # center at origin, basis vectors
        C = np.array([[1.0, 1.0]])  # constraint: alpha1 + alpha2 <= 1
        d = np.array([[1.0]])
        pred_lb = np.array([0.0, 0.0])
        pred_ub = np.array([1.0, 1.0])

        input_star = Star(V, C, d, pred_lb, pred_ub)

        # Layer: y = x (identity)
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: output should have same constraints
        assert len(output_stars) == 1
        output_star = output_stars[0]

        # Check that the constraint is preserved (affine map doesn't change it for identity)
        assert output_star.C.shape == input_star.C.shape
        assert output_star.nVar == input_star.nVar


class TestLinearZonoSoundness:
    """Soundness tests for Linear layer with Zonotope sets."""

    def test_identity_transformation(self):
        """Test identity transformation with zonotope."""
        # Create zonotope: center + generators
        # [0.5, 0.5] + [-0.5, 0.5] * alpha1 + [0, 0] (alpha1 in [-1, 1])
        # Represents [0,1] x [0.5, 0.5] (interval in x, point in y)
        c = np.array([[0.5], [0.5]])
        V = np.array([[0.5], [0.0]])

        input_zono = Zono(c, V)

        # Layer: identity
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_zonos = linear_zono(layer, [input_zono])

        # Ground truth: should be identical
        assert len(output_zonos) == 1
        assert np.allclose(output_zonos[0].c, c, atol=1e-6)
        assert np.allclose(output_zonos[0].V, V, atol=1e-6)

    def test_translation(self):
        """Test translation with zonotope."""
        c = np.array([[0.5], [0.5]])
        V = np.array([[0.5], [0.0]])
        input_zono = Zono(c, V)

        # Layer: y = x + [1, 2]
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.tensor([1.0, 2.0])

        # Compute reachability
        output_zonos = linear_zono(layer, [input_zono])

        # Ground truth: center shifts by bias
        expected_c = np.array([[1.5], [2.5]])
        assert len(output_zonos) == 1
        assert np.allclose(output_zonos[0].c, expected_c, atol=1e-6)
        assert np.allclose(output_zonos[0].V, V, atol=1e-6)  # generators unchanged

    def test_scaling(self):
        """Test scaling with zonotope."""
        c = np.array([[1.0], [1.0]])
        V = np.array([[0.5, 0.0],
                      [0.0, 0.5]])
        input_zono = Zono(c, V)

        # Layer: y = 2x
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2) * 2
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_zonos = linear_zono(layer, [input_zono])

        # Ground truth: center and generators scale by 2
        expected_c = np.array([[2.0], [2.0]])
        expected_V = np.array([[1.0, 0.0],
                               [0.0, 1.0]])

        assert len(output_zonos) == 1
        assert np.allclose(output_zonos[0].c, expected_c, atol=1e-6)
        assert np.allclose(output_zonos[0].V, expected_V, atol=1e-6)


class TestLinearBoxSoundness:
    """Soundness tests for Linear layer with Box sets."""

    def test_identity_transformation(self):
        """Test identity transformation with box."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_box = Box(lb, ub)

        # Layer: identity
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_boxes = linear_box(layer, [input_box])

        # Ground truth: identical bounds
        assert len(output_boxes) == 1
        assert np.allclose(output_boxes[0].lb, lb, atol=1e-6)
        assert np.allclose(output_boxes[0].ub, ub, atol=1e-6)

    def test_translation(self):
        """Test translation with box."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_box = Box(lb, ub)

        # Layer: y = x + [3, 4]
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.tensor([3.0, 4.0])

        # Compute reachability
        output_boxes = linear_box(layer, [input_box])

        # Ground truth: bounds shift by bias
        expected_lb = np.array([[3.0], [4.0]])
        expected_ub = np.array([[4.0], [5.0]])

        assert len(output_boxes) == 1
        assert np.allclose(output_boxes[0].lb, expected_lb, atol=1e-6)
        assert np.allclose(output_boxes[0].ub, expected_ub, atol=1e-6)

    def test_negative_bounds(self):
        """Test with negative input bounds."""
        lb = np.array([[-2.0], [-1.0]])
        ub = np.array([[-1.0], [0.0]])
        input_box = Box(lb, ub)

        # Layer: y = x
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2)
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_boxes = linear_box(layer, [input_box])

        # Ground truth: bounds unchanged
        assert len(output_boxes) == 1
        assert np.allclose(output_boxes[0].lb, lb, atol=1e-6)
        assert np.allclose(output_boxes[0].ub, ub, atol=1e-6)


class TestLinearEdgeCases:
    """Edge case soundness tests for Linear layer."""

    def test_very_large_weights(self):
        """Test with very large weight values."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Layer with large weights
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2) * 1e6
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: [0,1] * 1e6 = [0, 1e6]
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1e6], [1e6]])

        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, rtol=1e-6)

    def test_very_small_weights(self):
        """Test with very small weight values."""
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Layer with small weights
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.eye(2) * 1e-6
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth: [0,1] * 1e-6 = [0, 1e-6]
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected_lb = np.array([[0.0], [0.0]])
        expected_ub = np.array([[1e-6], [1e-6]])

        assert np.allclose(output_lb, expected_lb, atol=1e-9)
        assert np.allclose(output_ub, expected_ub, atol=1e-9)

    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative weights."""
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Layer: y1 = x1 - x2, y2 = x1 + x2
        layer = nn.Linear(2, 2)
        layer.weight.data = torch.tensor([[1.0, -1.0],
                                          [1.0, 1.0]])
        layer.bias.data = torch.zeros(2)

        # Compute reachability
        output_stars = linear_star(layer, [input_star])

        # Ground truth:
        # y1 = x1 - x2 in [-1-1, 1-(-1)] = [-2, 2]
        # y2 = x1 + x2 in [-1-1, 1+1] = [-2, 2]
        assert len(output_stars) == 1
        output_lb, output_ub = output_stars[0].estimate_ranges()

        expected_lb = np.array([[-2.0], [-2.0]])
        expected_ub = np.array([[2.0], [2.0]])

        assert np.allclose(output_lb, expected_lb, atol=1e-6)
        assert np.allclose(output_ub, expected_ub, atol=1e-6)
