"""Tests for ONNX MatMul operation with Zono and Box set types."""

import numpy as np
import pytest
from n2v.sets import Zono, Box


class TestZonoMatMul:
    """Test MatMul (y = x @ W, i.e., y = W^T @ x) for Zono sets."""

    def test_identity_weight_preserves_bounds(self, simple_zono):
        """MatMul with identity weight matrix should preserve the zonotope."""
        n = simple_zono.dim
        W_identity = np.eye(n)  # x @ I = x, so W^T = I^T = I
        result = simple_zono.affine_map(W_identity.T)

        np.testing.assert_array_almost_equal(result.c, simple_zono.c)
        np.testing.assert_array_almost_equal(result.V, simple_zono.V)

    def test_projection_weight_reduces_dimension(self, simple_zono):
        """MatMul with a projection weight should reduce output dimension."""
        # W is (3, 2): x (1x3) @ W (3x2) -> y (1x2), so W^T is (2x3)
        W = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [0.0, 0.0]])
        result = simple_zono.affine_map(W.T)

        assert result.dim == 2
        assert result.c.shape == (2, 1)
        assert result.V.shape[0] == 2

    def test_soundness_samples_within_bounds(self):
        """Random samples through matmul should stay within Zono output bounds."""
        np.random.seed(42)
        n_in, n_out = 4, 3
        c = np.random.randn(n_in, 1)
        V = np.random.randn(n_in, 3) * 0.1
        zono = Zono(c, V)

        W = np.random.randn(n_in, n_out)  # x @ W
        result = zono.affine_map(W.T)

        # Get bounds of the output zonotope
        result_lb, result_ub = result.get_ranges()

        # Sample from input zonotope and transform
        n_samples = 500
        for _ in range(n_samples):
            alpha = np.random.uniform(-1, 1, size=(V.shape[1], 1))
            x = c + V @ alpha  # point in input zono
            y = W.T @ x  # apply W^T (column-vector matmul)
            assert np.all(y >= result_lb - 1e-10), f"Sample below lb: {y.T} < {result_lb.T}"
            assert np.all(y <= result_ub + 1e-10), f"Sample above ub: {y.T} > {result_ub.T}"

    def test_scaling_weight(self):
        """MatMul with a diagonal scaling weight should scale center and generators."""
        c = np.array([[1.0], [2.0]])
        V = np.array([[0.1, 0.0],
                      [0.0, 0.2]])
        zono = Zono(c, V)

        # W = diag([2, 3]) means x @ W = [2*x1, 3*x2], W^T = diag([2, 3])
        W = np.diag([2.0, 3.0])
        result = zono.affine_map(W.T)

        np.testing.assert_array_almost_equal(result.c, np.array([[2.0], [6.0]]))
        expected_V = np.array([[0.2, 0.0],
                               [0.0, 0.6]])
        np.testing.assert_array_almost_equal(result.V, expected_V)


class TestBoxMatMul:
    """Test MatMul (y = x @ W, i.e., y = W^T @ x) for Box sets."""

    def test_identity_weight_preserves_bounds(self, simple_box):
        """MatMul with identity weight matrix should preserve the box."""
        n = simple_box.lb.shape[0]
        W_identity = np.eye(n)
        W_T = W_identity.T.astype(np.float64)
        pos = np.maximum(W_T, 0)
        neg = np.minimum(W_T, 0)
        new_lb = pos @ simple_box.lb + neg @ simple_box.ub
        new_ub = pos @ simple_box.ub + neg @ simple_box.lb
        result = Box(new_lb, new_ub)

        np.testing.assert_array_almost_equal(result.lb, simple_box.lb)
        np.testing.assert_array_almost_equal(result.ub, simple_box.ub)

    def test_positive_weight_preserves_order(self):
        """Positive weight matrix should preserve bound ordering."""
        lb = np.array([[0.0], [1.0]])
        ub = np.array([[2.0], [3.0]])
        box = Box(lb, ub)

        W = np.array([[1.0, 0.5],
                      [0.0, 1.0]])  # all non-negative, W^T applied
        W_T = W.T.astype(np.float64)
        pos = np.maximum(W_T, 0)
        neg = np.minimum(W_T, 0)
        new_lb = pos @ lb + neg @ ub
        new_ub = pos @ ub + neg @ lb
        result = Box(new_lb, new_ub)

        assert np.all(result.lb <= result.ub), "Lower bounds must not exceed upper bounds"

    def test_negative_weight_flips_bounds(self):
        """Negative weight entries should swap lb/ub contributions."""
        lb = np.array([[1.0]])
        ub = np.array([[3.0]])
        box = Box(lb, ub)

        # W = [[-2]], so W^T = [[-2]]
        W_T = np.array([[-2.0]])
        pos = np.maximum(W_T, 0)
        neg = np.minimum(W_T, 0)
        new_lb = pos @ lb + neg @ ub  # 0*1 + (-2)*3 = -6
        new_ub = pos @ ub + neg @ lb  # 0*3 + (-2)*1 = -2
        result = Box(new_lb, new_ub)

        np.testing.assert_array_almost_equal(result.lb, np.array([[-6.0]]))
        np.testing.assert_array_almost_equal(result.ub, np.array([[-2.0]]))

    def test_soundness_samples_within_bounds(self):
        """Random samples through matmul should stay within Box output bounds."""
        np.random.seed(42)
        n_in, n_out = 5, 3
        lb = np.random.randn(n_in, 1)
        ub = lb + np.abs(np.random.randn(n_in, 1)) + 0.1
        box = Box(lb, ub)

        W = np.random.randn(n_in, n_out)
        W_T = W.T.astype(np.float64)
        pos = np.maximum(W_T, 0)
        neg = np.minimum(W_T, 0)
        new_lb = pos @ lb + neg @ ub
        new_ub = pos @ ub + neg @ lb
        result = Box(new_lb, new_ub)

        # Sample random points from the input box and transform
        n_samples = 500
        for _ in range(n_samples):
            x = lb + np.random.rand(n_in, 1) * (ub - lb)
            y = W_T @ x
            assert np.all(y >= result.lb - 1e-10), f"Sample below lb"
            assert np.all(y <= result.ub + 1e-10), f"Sample above ub"

    def test_dimension_reduction(self):
        """MatMul reducing dimensions should produce correct output shape."""
        lb = np.array([[0.0], [0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0], [1.0]])
        box = Box(lb, ub)

        # W is (4, 2), W^T is (2, 4)
        W = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0],
                      [0.0, 0.0]])
        W_T = W.T.astype(np.float64)
        pos = np.maximum(W_T, 0)
        neg = np.minimum(W_T, 0)
        new_lb = pos @ lb + neg @ ub
        new_ub = pos @ ub + neg @ lb
        result = Box(new_lb, new_ub)

        assert result.lb.shape == (2, 1)
        assert result.ub.shape == (2, 1)
