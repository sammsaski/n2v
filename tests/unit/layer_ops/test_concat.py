"""Tests for set concatenation in graph module reachability."""

import numpy as np
from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.nn.reach import _concat_sets


class TestConcatTwoStars:
    """Test concatenation of two Stars along feature dimension."""

    def test_concat_two_stars(self):
        """Concat two Stars: vstack V, dim = sum of dims, constraints from first."""
        V1 = np.array([[1.0, 0.5, 0.0],
                        [2.0, 0.0, 0.3]])
        V2 = np.array([[3.0, 0.1, 0.0],
                        [4.0, 0.0, 0.2],
                        [5.0, 0.4, 0.0]])
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([[1.0], [1.0]])
        pred_lb = np.array([[-1.0], [-1.0]])
        pred_ub = np.array([[1.0], [1.0]])

        s1 = Star(V1, C, d, pred_lb, pred_ub)
        s2 = Star(V2, C, d, pred_lb, pred_ub)

        result = _concat_sets([[s1], [s2]], axis=0)
        out = result[0]

        assert isinstance(out, Star)
        # V should be vstack of V1 and V2
        expected_V = np.vstack([V1, V2])
        assert np.allclose(out.V, expected_V)
        # Dimension should be sum
        assert out.dim == 5  # 2 + 3
        # Constraints from first set
        assert np.array_equal(out.C, C)
        assert np.array_equal(out.d, d)
        assert np.array_equal(out.predicate_lb, pred_lb)
        assert np.array_equal(out.predicate_ub, pred_ub)


class TestConcatThreeStars:
    """Test concatenation of three Stars."""

    def test_concat_three_stars(self):
        """Concat three Stars: vstack all V matrices."""
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[-1.0]])
        pred_ub = np.array([[1.0]])

        V1 = np.array([[1.0, 0.5],
                        [2.0, 0.3]])
        V2 = np.array([[3.0, 0.1]])
        V3 = np.array([[4.0, 0.2],
                        [5.0, 0.4],
                        [6.0, 0.6]])

        s1 = Star(V1, C, d, pred_lb, pred_ub)
        s2 = Star(V2, C, d, pred_lb, pred_ub)
        s3 = Star(V3, C, d, pred_lb, pred_ub)

        result = _concat_sets([[s1], [s2], [s3]], axis=0)
        out = result[0]

        assert isinstance(out, Star)
        expected_V = np.vstack([V1, V2, V3])
        assert np.allclose(out.V, expected_V)
        assert out.dim == 6  # 2 + 1 + 3
        assert np.array_equal(out.C, C)
        assert np.array_equal(out.d, d)


class TestConcatTwoZonos:
    """Test concatenation of two Zonos."""

    def test_concat_two_zonos(self):
        """Concat two Zonos: vstack c and V."""
        c1 = np.array([[1.0], [2.0]])
        V1 = np.array([[0.5, 0.0], [0.0, 0.3]])
        c2 = np.array([[3.0], [4.0], [5.0]])
        V2 = np.array([[0.1, 0.0], [0.0, 0.2], [0.4, 0.0]])

        z1 = Zono(c1, V1)
        z2 = Zono(c2, V2)

        result = _concat_sets([[z1], [z2]], axis=0)
        out = result[0]

        assert isinstance(out, Zono)
        assert np.allclose(out.c, np.vstack([c1, c2]))
        assert np.allclose(out.V, np.vstack([V1, V2]))
        assert out.dim == 5  # 2 + 3


class TestConcatTwoBoxes:
    """Test concatenation of two Boxes."""

    def test_concat_two_boxes(self):
        """Concat two Boxes: vstack lb and ub."""
        b1 = Box(np.array([[0.0], [1.0]]), np.array([[1.0], [2.0]]))
        b2 = Box(np.array([[3.0], [4.0], [5.0]]), np.array([[6.0], [7.0], [8.0]]))

        result = _concat_sets([[b1], [b2]], axis=0)
        out = result[0]

        assert isinstance(out, Box)
        assert np.allclose(out.lb, np.array([[0.0], [1.0], [3.0], [4.0], [5.0]]))
        assert np.allclose(out.ub, np.array([[1.0], [2.0], [6.0], [7.0], [8.0]]))
        assert out.dim == 5


class TestConcatImageStarsChannel:
    """Test concatenation of ImageStars along channel dimension."""

    def test_concat_imagestars_channel(self):
        """Concat ImageStars along channel dim (axis=2 in HWC) -- output has summed channels."""
        lb1 = np.zeros((2, 3, 1))
        ub1 = np.ones((2, 3, 1))
        istar1 = ImageStar.from_bounds(lb1, ub1, height=2, width=3, num_channels=1)

        lb2 = np.zeros((2, 3, 2))
        ub2 = np.ones((2, 3, 2)) * 0.5
        istar2 = ImageStar.from_bounds(lb2, ub2, height=2, width=3, num_channels=2)

        result = _concat_sets([[istar1], [istar2]], axis=2)
        out = result[0]

        assert isinstance(out, ImageStar)
        assert out.height == 2
        assert out.width == 3
        assert out.num_channels == 3  # 1 + 2
        # V shape: (H, W, C_total, nVar+1)
        assert out.V.shape[0] == 2
        assert out.V.shape[1] == 3
        assert out.V.shape[2] == 3


class TestConcatBroadcastSingle:
    """Test broadcasting when one input list has 1 set and another has multiple."""

    def test_concat_broadcast_single(self):
        """One input list has 1 set, another has 2: broadcast the single to match."""
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[-1.0]])
        pred_ub = np.array([[1.0]])

        V1 = np.array([[1.0, 0.5], [2.0, 0.3]])
        V2a = np.array([[3.0, 0.1]])
        V2b = np.array([[4.0, 0.2]])

        s1 = Star(V1, C, d, pred_lb, pred_ub)
        s2a = Star(V2a, C, d, pred_lb, pred_ub)
        s2b = Star(V2b, C, d, pred_lb, pred_ub)

        # First list has 1 set (should be broadcast), second has 2
        result = _concat_sets([[s1], [s2a, s2b]], axis=0)

        assert len(result) == 2

        # First output: concat s1 with s2a
        out0 = result[0]
        assert isinstance(out0, Star)
        assert np.allclose(out0.V, np.vstack([V1, V2a]))
        assert out0.dim == 3  # 2 + 1

        # Second output: concat s1 with s2b
        out1 = result[1]
        assert isinstance(out1, Star)
        assert np.allclose(out1.V, np.vstack([V1, V2b]))
        assert out1.dim == 3
