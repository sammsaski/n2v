"""Tests for Reshape operations on sets."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box, ImageStar, ImageZono
from n2v.nn.reach import _handle_reshape


class TestReshapeImageStar:
    """Test reshape on ImageStar sets."""

    def test_flatten_imagestar(self):
        """Reshape to flat should produce Star (not ImageStar)."""
        lb = np.zeros((3, 3, 2))
        ub = np.ones((3, 3, 2))
        img_star = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        # Target shape (1, -1) means flatten (batch=1, rest flat)
        result = _handle_reshape([img_star], (1, -1))

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Star)
        assert not isinstance(out, ImageStar)
        assert out.dim == 3 * 3 * 2

    def test_reshape_to_1x1xC(self):
        """Reshape from (1,C,1,1) to (1,C) — common after GlobalAvgPool."""
        # Create a 1x1x4 ImageStar (like output of GlobalAvgPool with 4 channels)
        lb = np.zeros((1, 1, 4))
        ub = np.ones((1, 1, 4))
        img_star = ImageStar.from_bounds(lb, ub, height=1, width=1, num_channels=4)

        # Reshape to (1, 4) — flatten spatial dims
        result = _handle_reshape([img_star], (1, 4))

        assert len(result) == 1
        out = result[0]
        assert out.dim == 4

    def test_reshape_preserves_bounds(self):
        """Reshape should not change the bounds, just the layout."""
        lb = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        ub = np.array([[[5.0, 6.0], [7.0, 8.0]]])
        img_star = ImageStar.from_bounds(lb, ub, height=1, width=2, num_channels=2)

        result = _handle_reshape([img_star], (1, -1))

        assert len(result) == 1
        out = result[0]
        assert out.dim == 4
        lb_out, ub_out = out.estimate_ranges()
        # Bounds should be a permutation of the original bounds
        assert lb_out.min() >= 1.0 - 1e-6
        assert ub_out.max() <= 8.0 + 1e-6


class TestReshapeImageZono:
    """Test reshape on ImageZono sets."""

    def test_flatten_imagezono(self):
        """Reshape to flat should produce Zono (not ImageZono)."""
        lb = np.zeros((3, 3, 2))
        ub = np.ones((3, 3, 2))
        img_zono = ImageZono.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        result = _handle_reshape([img_zono], (1, -1))

        assert len(result) == 1
        out = result[0]
        assert isinstance(out, Zono)
        assert not isinstance(out, ImageZono)
        assert out.dim == 3 * 3 * 2

    def test_reshape_preserves_bounds(self):
        """Reshape should not change the bounds."""
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))
        img_zono = ImageZono.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        result = _handle_reshape([img_zono], (1, -1))

        out = result[0]
        lb_out, ub_out = out.get_bounds()
        assert np.all(lb_out >= -1e-6)
        assert np.all(ub_out <= 1.0 + 1e-6)


class TestReshapePlainStar:
    """Test reshape on plain Star/Zono (already flat)."""

    def test_reshape_star_noop(self):
        """Reshape on flat Star with matching dims is essentially a noop."""
        V = np.array([[1.0, 0.5], [0.0, 0.3], [0.0, 0.1], [0.0, 0.2]])
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[0.0]])
        pred_ub = np.array([[1.0]])
        star = Star(V, C, d, pred_lb, pred_ub)

        result = _handle_reshape([star], (1, 4))

        assert len(result) == 1
        assert result[0].dim == 4
