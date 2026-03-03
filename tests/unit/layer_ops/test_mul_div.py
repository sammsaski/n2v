"""Tests for element-wise Mul/Div by constant in graph module reachability."""

import numpy as np
import pytest
from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.reach import _mul_sets_by_constant


class TestMulStarPositive:
    """Test Star multiplication by positive constant."""

    def test_star_mul_positive_scalar(self):
        """Multiply Star V by a positive scalar: V_out = scale * V."""
        V = np.array([[1.0, 0.5, 0.0],
                       [2.0, 0.0, 0.3]])
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([[1.0], [1.0]])
        pred_lb = np.array([[-1.0], [-1.0]])
        pred_ub = np.array([[1.0], [1.0]])

        s = Star(V, C, d, pred_lb, pred_ub)
        scale = np.array([3.0, 3.0])

        result = _mul_sets_by_constant([s], scale)
        out = result[0]

        expected_V = 3.0 * V
        assert np.allclose(out.V, expected_V)

    def test_star_mul_channel_wise_positive(self):
        """Multiply Star by per-dimension positive scale factors."""
        V = np.array([[1.0, 0.5],
                       [2.0, 0.3],
                       [3.0, 0.1]])
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[-1.0]])
        pred_ub = np.array([[1.0]])

        s = Star(V, C, d, pred_lb, pred_ub)
        scale = np.array([2.0, 0.5, 3.0])

        result = _mul_sets_by_constant([s], scale)
        out = result[0]

        # Each row of V should be scaled by corresponding scale factor
        expected_V = V.copy()
        expected_V[0, :] *= 2.0
        expected_V[1, :] *= 0.5
        expected_V[2, :] *= 3.0
        assert np.allclose(out.V, expected_V)


class TestMulStarNegative:
    """Test Star multiplication by negative constant."""

    def test_star_mul_negative_scalar(self):
        """Multiply Star V by a negative scalar: V_out = scale * V."""
        V = np.array([[1.0, 0.5, 0.0],
                       [2.0, 0.0, 0.3]])
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([[1.0], [1.0]])
        pred_lb = np.array([[-1.0], [-1.0]])
        pred_ub = np.array([[1.0], [1.0]])

        s = Star(V, C, d, pred_lb, pred_ub)
        scale = np.array([-2.0, -2.0])

        result = _mul_sets_by_constant([s], scale)
        out = result[0]

        expected_V = -2.0 * V
        assert np.allclose(out.V, expected_V)

    def test_star_mul_mixed_signs(self):
        """Multiply Star by mixed positive/negative per-dimension scale."""
        V = np.array([[1.0, 0.5],
                       [2.0, 0.3]])
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[-1.0]])
        pred_ub = np.array([[1.0]])

        s = Star(V, C, d, pred_lb, pred_ub)
        scale = np.array([2.0, -3.0])

        result = _mul_sets_by_constant([s], scale)
        out = result[0]

        expected_V = V.copy()
        expected_V[0, :] *= 2.0
        expected_V[1, :] *= -3.0
        assert np.allclose(out.V, expected_V)


class TestMulStarPreservesConstraints:
    """Verify that Mul preserves constraints (C, d, pred_lb, pred_ub)."""

    def test_constraints_unchanged(self):
        """C, d, predicate_lb, predicate_ub must not change after Mul."""
        V = np.array([[1.0, 0.5, 0.2],
                       [2.0, 0.3, 0.1]])
        C = np.array([[1.0, -1.0], [0.5, 0.5]])
        d = np.array([[2.0], [3.0]])
        pred_lb = np.array([[-2.0], [-1.0]])
        pred_ub = np.array([[2.0], [1.0]])

        s = Star(V, C, d, pred_lb, pred_ub)
        scale = np.array([5.0, -0.5])

        result = _mul_sets_by_constant([s], scale)
        out = result[0]

        assert np.array_equal(out.C, C)
        assert np.array_equal(out.d, d)
        assert np.array_equal(out.predicate_lb, pred_lb)
        assert np.array_equal(out.predicate_ub, pred_ub)

    def test_multiple_sets_constraints_preserved(self):
        """Constraints are preserved across multiple sets in the list."""
        V1 = np.array([[1.0, 0.5], [2.0, 0.3]])
        V2 = np.array([[3.0, 0.1], [4.0, 0.2]])
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[-1.0]])
        pred_ub = np.array([[1.0]])

        s1 = Star(V1, C, d, pred_lb, pred_ub)
        s2 = Star(V2, C, d, pred_lb, pred_ub)
        scale = np.array([2.0, 3.0])

        result = _mul_sets_by_constant([s1, s2], scale)

        for out in result:
            assert np.array_equal(out.C, C)
            assert np.array_equal(out.d, d)
            assert np.array_equal(out.predicate_lb, pred_lb)
            assert np.array_equal(out.predicate_ub, pred_ub)


class TestMulImageStar:
    """Test ImageStar multiplication channel-wise."""

    def test_imagestar_mul_channel_wise(self):
        """Channel-wise scale applied to V tensor (H, W, C, nVar+1)."""
        lb = np.zeros((2, 3, 2))
        ub = np.ones((2, 3, 2))
        istar = ImageStar.from_bounds(lb, ub, height=2, width=3, num_channels=2)

        # Scale per channel: [2.0, -0.5]
        scale = np.array([2.0, -0.5])

        result = _mul_sets_by_constant([istar], scale)
        out = result[0]

        assert isinstance(out, ImageStar)
        assert out.height == 2
        assert out.width == 3
        assert out.num_channels == 2

        # V has shape (H, W, C, nVar+1)
        # Channel 0 should be scaled by 2.0
        assert np.allclose(out.V[:, :, 0, :], 2.0 * istar.V[:, :, 0, :])
        # Channel 1 should be scaled by -0.5
        assert np.allclose(out.V[:, :, 1, :], -0.5 * istar.V[:, :, 1, :])

    def test_imagestar_mul_uniform_scale(self):
        """Uniform scale across all channels."""
        lb = np.zeros((3, 3, 1))
        ub = np.ones((3, 3, 1))
        istar = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        scale = np.array([3.0])

        result = _mul_sets_by_constant([istar], scale)
        out = result[0]

        assert isinstance(out, ImageStar)
        assert np.allclose(out.V, 3.0 * istar.V)

    def test_imagestar_mul_preserves_constraints(self):
        """ImageStar Mul preserves constraints."""
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))
        istar = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        scale = np.array([5.0])
        result = _mul_sets_by_constant([istar], scale)
        out = result[0]

        assert np.array_equal(out.C, istar.C)
        assert np.array_equal(out.d, istar.d)
        assert np.array_equal(out.predicate_lb, istar.predicate_lb)
        assert np.array_equal(out.predicate_ub, istar.predicate_ub)

    def test_imagestar_mul_full_spatial_scale(self):
        """Full H*W*C scale applied element-wise."""
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))
        istar = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Scale each pixel independently (H*W*C = 4)
        scale = np.array([1.0, 2.0, 3.0, 4.0])

        result = _mul_sets_by_constant([istar], scale)
        out = result[0]

        assert isinstance(out, ImageStar)
        # Reshape scale to (H, W, C, 1) = (2, 2, 1, 1) and check
        scale_4d = scale.reshape(2, 2, 1, 1)
        assert np.allclose(out.V, scale_4d * istar.V)


class TestMulZono:
    """Test Zono multiplication."""

    def test_zono_mul_positive(self):
        """Zono Mul: c and V both scaled."""
        c = np.array([[1.0], [2.0]])
        V = np.array([[0.5, 0.0], [0.0, 0.3]])
        z = Zono(c, V)

        scale = np.array([3.0, 2.0])

        result = _mul_sets_by_constant([z], scale)
        out = result[0]

        assert isinstance(out, Zono)
        scale_col = scale.reshape(-1, 1)
        assert np.allclose(out.c, scale_col * c)
        assert np.allclose(out.V, scale_col * V)

    def test_zono_mul_negative(self):
        """Zono Mul with negative scale: c and V both scaled."""
        c = np.array([[1.0], [2.0]])
        V = np.array([[0.5, 0.0], [0.0, 0.3]])
        z = Zono(c, V)

        scale = np.array([-1.0, -2.0])

        result = _mul_sets_by_constant([z], scale)
        out = result[0]

        scale_col = scale.reshape(-1, 1)
        assert np.allclose(out.c, scale_col * c)
        assert np.allclose(out.V, scale_col * V)

    def test_zono_mul_mixed(self):
        """Zono Mul with mixed positive/negative scale."""
        c = np.array([[1.0], [2.0], [3.0]])
        V = np.array([[0.5], [0.3], [0.1]])
        z = Zono(c, V)

        scale = np.array([2.0, -1.0, 0.5])

        result = _mul_sets_by_constant([z], scale)
        out = result[0]

        scale_col = scale.reshape(-1, 1)
        assert np.allclose(out.c, scale_col * c)
        assert np.allclose(out.V, scale_col * V)


class TestMulImageZono:
    """Test ImageZono multiplication."""

    def test_imagezono_mul_channel_wise(self):
        """ImageZono Mul: c and V scaled channel-wise."""
        c = np.array([[1.0], [2.0], [3.0], [4.0]])  # H*W*C = 2*1*2 = 4
        V = np.array([[0.1, 0.0], [0.2, 0.0], [0.0, 0.1], [0.0, 0.2]])
        iz = ImageZono(c, V, height=2, width=1, num_channels=2)

        # Scale per channel
        scale = np.array([3.0, -1.0])

        result = _mul_sets_by_constant([iz], scale)
        out = result[0]

        assert isinstance(out, ImageZono)
        assert out.height == 2
        assert out.width == 1
        assert out.num_channels == 2

        # For HWC layout: pixels are stored as (h0w0c0, h0w0c1, h1w0c0, h1w0c1)
        # scale_flat should be tiled: [3.0, -1.0, 3.0, -1.0]
        scale_flat = np.tile(scale, 2).reshape(-1, 1)  # h*w=2, tile by 2
        assert np.allclose(out.c, scale_flat * c)
        assert np.allclose(out.V, scale_flat * V)


class TestMulBox:
    """Test Box multiplication."""

    def test_box_mul_positive(self):
        """Box Mul positive: lb*s, ub*s."""
        b = Box(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        scale = np.array([2.0, 3.0])

        result = _mul_sets_by_constant([b], scale)
        out = result[0]

        assert isinstance(out, Box)
        assert np.allclose(out.lb, np.array([[2.0], [6.0]]))
        assert np.allclose(out.ub, np.array([[6.0], [12.0]]))

    def test_box_mul_negative_swaps_bounds(self):
        """Box Mul negative: bounds swap via min/max."""
        b = Box(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        scale = np.array([-2.0, -1.0])

        result = _mul_sets_by_constant([b], scale)
        out = result[0]

        assert isinstance(out, Box)
        # -2 * [1,3] -> [-6,-2], so lb=-6, ub=-2
        # -1 * [2,4] -> [-4,-2], so lb=-4, ub=-2
        assert np.allclose(out.lb, np.array([[-6.0], [-4.0]]))
        assert np.allclose(out.ub, np.array([[-2.0], [-2.0]]))

    def test_box_mul_mixed_signs(self):
        """Box Mul with mixed positive/negative scale."""
        b = Box(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        scale = np.array([2.0, -3.0])

        result = _mul_sets_by_constant([b], scale)
        out = result[0]

        # dim 0: 2*[1,3] -> [2,6]
        # dim 1: -3*[2,4] -> [-12,-6], so lb=-12, ub=-6
        assert np.allclose(out.lb, np.array([[2.0], [-12.0]]))
        assert np.allclose(out.ub, np.array([[6.0], [-6.0]]))

    def test_box_mul_zero_scale(self):
        """Box Mul by zero: produces zero-width box."""
        b = Box(np.array([[1.0]]), np.array([[3.0]]))
        scale = np.array([0.0])

        result = _mul_sets_by_constant([b], scale)
        out = result[0]

        assert np.allclose(out.lb, np.array([[0.0]]))
        assert np.allclose(out.ub, np.array([[0.0]]))


class TestDivByConstant:
    """Test that Div by constant is equivalent to Mul by 1/constant."""

    def test_div_star_equivalent_to_mul_reciprocal(self):
        """Div Star by constant = Mul Star by 1/constant."""
        V = np.array([[1.0, 0.5], [2.0, 0.3]])
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[-1.0]])
        pred_ub = np.array([[1.0]])

        s = Star(V, C, d, pred_lb, pred_ub)
        divisor = np.array([2.0, 4.0])

        result_div = _mul_sets_by_constant([s], 1.0 / divisor)
        result_mul = _mul_sets_by_constant([s], np.array([0.5, 0.25]))

        assert np.allclose(result_div[0].V, result_mul[0].V)

    def test_div_box_equivalent_to_mul_reciprocal(self):
        """Div Box by constant = Mul Box by 1/constant."""
        b = Box(np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]]))
        divisor = np.array([2.0, -4.0])

        result_div = _mul_sets_by_constant([b], 1.0 / divisor)
        result_mul = _mul_sets_by_constant([b], np.array([0.5, -0.25]))

        assert np.allclose(result_div[0].lb, result_mul[0].lb)
        assert np.allclose(result_div[0].ub, result_mul[0].ub)

    def test_div_zono_equivalent_to_mul_reciprocal(self):
        """Div Zono by constant = Mul Zono by 1/constant."""
        c = np.array([[1.0], [2.0]])
        V = np.array([[0.5], [0.3]])
        z = Zono(c, V)
        divisor = np.array([5.0, 2.0])

        result_div = _mul_sets_by_constant([z], 1.0 / divisor)
        result_mul = _mul_sets_by_constant([z], np.array([0.2, 0.5]))

        assert np.allclose(result_div[0].c, result_mul[0].c)
        assert np.allclose(result_div[0].V, result_mul[0].V)
