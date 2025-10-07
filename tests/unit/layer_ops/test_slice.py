"""Tests for set slicing in graph module reachability."""

import numpy as np
from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.nn.reach import _slice_set


class TestSliceStarSimple:
    """Test slicing a Star along dimension axis."""

    def test_slice_star_simple(self):
        """Slice Star [1:3] from dim=4 -> select rows 1,2 of V, dim=2, constraints preserved."""
        V = np.array([[1.0, 0.5, 0.0],
                      [2.0, 0.0, 0.3],
                      [3.0, 0.1, 0.0],
                      [4.0, 0.0, 0.2]])
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([[1.0], [1.0]])
        pred_lb = np.array([[-1.0], [-1.0]])
        pred_ub = np.array([[1.0], [1.0]])

        star = Star(V, C, d, pred_lb, pred_ub)

        result = _slice_set(star, {0: slice(1, 3)})

        assert isinstance(result, Star)
        # V should be rows 1 and 2 of original V
        expected_V = V[1:3, :]
        assert np.allclose(result.V, expected_V)
        assert result.dim == 2
        # Constraints preserved unchanged
        assert np.array_equal(result.C, C)
        assert np.array_equal(result.d, d)
        assert np.array_equal(result.predicate_lb, pred_lb)
        assert np.array_equal(result.predicate_ub, pred_ub)


class TestSliceStarWithStep:
    """Test slicing a Star with a step."""

    def test_slice_star_with_step(self):
        """Slice Star [0:4:2] -> every other element (rows 0, 2)."""
        V = np.array([[1.0, 0.5, 0.0],
                      [2.0, 0.0, 0.3],
                      [3.0, 0.1, 0.0],
                      [4.0, 0.0, 0.2]])
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([[1.0], [1.0]])
        pred_lb = np.array([[-1.0], [-1.0]])
        pred_ub = np.array([[1.0], [1.0]])

        star = Star(V, C, d, pred_lb, pred_ub)

        result = _slice_set(star, {0: slice(0, 4, 2)})

        assert isinstance(result, Star)
        expected_V = V[0:4:2, :]
        assert np.allclose(result.V, expected_V)
        assert result.dim == 2  # rows 0 and 2


class TestSliceZono:
    """Test slicing a Zono."""

    def test_slice_zono(self):
        """Slice Zono [1:3] -> select rows 1,2 of c and V."""
        c = np.array([[1.0], [2.0], [3.0], [4.0]])
        V = np.array([[0.5, 0.0],
                      [0.0, 0.3],
                      [0.1, 0.0],
                      [0.0, 0.2]])
        zono = Zono(c, V)

        result = _slice_set(zono, {0: slice(1, 3)})

        assert isinstance(result, Zono)
        assert np.allclose(result.c, c[1:3, :])
        assert np.allclose(result.V, V[1:3, :])
        assert result.dim == 2


class TestSliceBox:
    """Test slicing a Box."""

    def test_slice_box(self):
        """Slice Box [0:2] -> select rows 0,1 of lb and ub."""
        lb = np.array([[0.0], [1.0], [2.0], [3.0]])
        ub = np.array([[1.0], [2.0], [3.0], [4.0]])
        box = Box(lb, ub)

        result = _slice_set(box, {0: slice(0, 2)})

        assert isinstance(result, Box)
        assert np.allclose(result.lb, lb[0:2, :])
        assert np.allclose(result.ub, ub[0:2, :])
        assert result.dim == 2


class TestSliceImageStarChannel:
    """Test slicing an ImageStar along the channel dimension."""

    def test_slice_imagestar_channel(self):
        """Slice ImageStar channel dim [0:2] (axis=2 in HWC) -> 2 of 3 channels."""
        lb = np.zeros((4, 4, 3))
        ub = np.ones((4, 4, 3))
        istar = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=3)

        # Slice channels 0 and 1 (axis=2 in HWC)
        result = _slice_set(istar, {2: slice(0, 2)})

        assert isinstance(result, ImageStar)
        assert result.height == 4
        assert result.width == 4
        assert result.num_channels == 2
        # V shape: (H, W, C, nVar+1)
        assert result.V.shape[0] == 4
        assert result.V.shape[1] == 4
        assert result.V.shape[2] == 2


class TestSliceImageStarSpatial:
    """Test slicing an ImageStar along the H (spatial) dimension."""

    def test_slice_imagestar_spatial(self):
        """Slice ImageStar H dim [1:3] (axis=0 in HWC) -> rows 1,2."""
        lb = np.zeros((4, 4, 2))
        ub = np.ones((4, 4, 2))
        istar = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=2)

        # Slice H dimension [1:3] (axis=0 in HWC)
        result = _slice_set(istar, {0: slice(1, 3)})

        assert isinstance(result, ImageStar)
        assert result.height == 2
        assert result.width == 4
        assert result.num_channels == 2
        # V shape: (H, W, C, nVar+1)
        assert result.V.shape[0] == 2
        assert result.V.shape[1] == 4
        assert result.V.shape[2] == 2
