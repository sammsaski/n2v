"""Tests for ONNX binary operations (Add/Sub) with Zono and Box set types."""

import numpy as np
import pytest
from n2v.sets import Zono, Box
from n2v.sets.image_zono import ImageZono


class TestZonoBiasAdd:
    """Test bias addition for Zono sets in _handle_onnx_binary_op."""

    def test_positive_bias_shifts_center(self, simple_zono):
        """Adding a positive bias should shift the center without affecting generators."""
        bias = np.array([0.1, 0.2, 0.3])
        bias_reshaped = bias.reshape(-1, 1)

        original_c = simple_zono.c.copy()
        original_V = simple_zono.V.copy()

        new_c = simple_zono.c + bias_reshaped
        result = Zono(new_c, simple_zono.V)

        np.testing.assert_array_almost_equal(result.c, original_c + bias_reshaped)
        np.testing.assert_array_almost_equal(result.V, original_V)

    def test_negative_bias_shifts_center(self, simple_zono):
        """Adding a negative bias should shift the center in the negative direction."""
        bias = np.array([-0.5, -0.3, -0.1])
        bias_reshaped = bias.reshape(-1, 1)

        original_c = simple_zono.c.copy()
        original_V = simple_zono.V.copy()

        new_c = simple_zono.c + bias_reshaped
        result = Zono(new_c, simple_zono.V)

        np.testing.assert_array_almost_equal(result.c, original_c + bias_reshaped)
        np.testing.assert_array_almost_equal(result.V, original_V)

    def test_zero_bias_no_change(self, simple_zono):
        """Adding a zero bias should not change the zonotope."""
        bias = np.array([0.0, 0.0, 0.0])
        bias_reshaped = bias.reshape(-1, 1)

        original_c = simple_zono.c.copy()
        original_V = simple_zono.V.copy()

        new_c = simple_zono.c + bias_reshaped
        result = Zono(new_c, simple_zono.V)

        np.testing.assert_array_almost_equal(result.c, original_c)
        np.testing.assert_array_almost_equal(result.V, original_V)

    def test_generators_unchanged(self, simple_zono):
        """Generators must remain identical after bias addition."""
        bias = np.array([10.0, -5.0, 3.0])
        bias_reshaped = bias.reshape(-1, 1)

        original_V = simple_zono.V.copy()
        new_c = simple_zono.c + bias_reshaped
        result = Zono(new_c, simple_zono.V)

        np.testing.assert_array_equal(result.V, original_V)


class TestBoxBiasAdd:
    """Test bias addition for Box sets in _handle_onnx_binary_op."""

    def test_positive_bias_shifts_bounds(self, simple_box):
        """Adding a positive bias should shift both lb and ub."""
        bias = np.array([0.1, 0.2, 0.3])
        bias_reshaped = bias.reshape(-1, 1)

        original_lb = simple_box.lb.copy()
        original_ub = simple_box.ub.copy()

        result = Box(simple_box.lb + bias_reshaped, simple_box.ub + bias_reshaped)

        np.testing.assert_array_almost_equal(result.lb, original_lb + bias_reshaped)
        np.testing.assert_array_almost_equal(result.ub, original_ub + bias_reshaped)

    def test_negative_bias_shifts_bounds(self, simple_box):
        """Adding a negative bias should shift bounds in the negative direction."""
        bias = np.array([-0.5, -0.3, -0.1])
        bias_reshaped = bias.reshape(-1, 1)

        original_lb = simple_box.lb.copy()
        original_ub = simple_box.ub.copy()

        result = Box(simple_box.lb + bias_reshaped, simple_box.ub + bias_reshaped)

        np.testing.assert_array_almost_equal(result.lb, original_lb + bias_reshaped)
        np.testing.assert_array_almost_equal(result.ub, original_ub + bias_reshaped)

    def test_zero_bias_no_change(self, simple_box):
        """Adding zero bias should not change the box."""
        bias = np.array([0.0, 0.0, 0.0])
        bias_reshaped = bias.reshape(-1, 1)

        original_lb = simple_box.lb.copy()
        original_ub = simple_box.ub.copy()

        result = Box(simple_box.lb + bias_reshaped, simple_box.ub + bias_reshaped)

        np.testing.assert_array_almost_equal(result.lb, original_lb)
        np.testing.assert_array_almost_equal(result.ub, original_ub)

    def test_width_preserved(self, simple_box):
        """Box width (ub - lb) should be preserved after bias addition."""
        bias = np.array([5.0, -3.0, 1.0])
        bias_reshaped = bias.reshape(-1, 1)

        original_width = simple_box.ub - simple_box.lb
        result = Box(simple_box.lb + bias_reshaped, simple_box.ub + bias_reshaped)

        np.testing.assert_array_almost_equal(result.ub - result.lb, original_width)


class TestImageZonoBiasAdd:
    """Test bias addition for ImageZono sets."""

    def test_channel_bias_tiled(self, simple_image_zono):
        """A per-channel bias should be tiled across spatial dimensions."""
        # 1-channel ImageZono, so bias is a single value
        bias = np.array([0.5])
        bias_reshaped = bias.reshape(-1, 1)
        h, w, c_ch = simple_image_zono.height, simple_image_zono.width, simple_image_zono.num_channels

        if bias_reshaped.size == c_ch:
            bias_flat = np.tile(bias.flatten(), h * w).reshape(-1, 1)
        else:
            bias_flat = bias_reshaped

        original_V = simple_image_zono.V.copy()
        new_c = simple_image_zono.c + bias_flat
        result = ImageZono(new_c, simple_image_zono.V, h, w, c_ch)

        # Center should be shifted by bias_flat
        np.testing.assert_array_almost_equal(result.c - simple_image_zono.c, bias_flat)
        # Generators unchanged
        np.testing.assert_array_equal(result.V, original_V)
