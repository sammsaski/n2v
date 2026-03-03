"""Tests for residual element-wise Add/Sub in graph module reachability."""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.reach import _add_sets


class TestAddSetsStar:
    """Test set addition with Star."""

    def test_star_addition_shared_predicates(self):
        """Adding two Stars with shared predicates: V_out = V1 + V2."""
        V1 = np.array([[1.0, 0.5, 0.0],
                        [2.0, 0.0, 0.3]])
        V2 = np.array([[0.5, 0.1, 0.0],
                        [1.0, 0.0, 0.2]])
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([[1.0], [1.0]])
        pred_lb = np.array([[-1.0], [-1.0]])
        pred_ub = np.array([[1.0], [1.0]])

        s1 = Star(V1, C, d, pred_lb, pred_ub)
        s2 = Star(V2, C, d, pred_lb, pred_ub)

        result = _add_sets([s1], [s2], 'add')
        out = result[0]

        expected_V = V1 + V2
        assert np.allclose(out.V, expected_V)
        assert np.array_equal(out.C, C)
        assert np.array_equal(out.d, d)

    def test_star_subtraction(self):
        """Subtracting Stars: V_out = V1 - V2."""
        V1 = np.array([[1.0, 0.5], [2.0, 0.3]])
        V2 = np.array([[0.5, 0.1], [1.0, 0.2]])
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[-1.0]])
        pred_ub = np.array([[1.0]])

        s1 = Star(V1, C, d, pred_lb, pred_ub)
        s2 = Star(V2, C, d, pred_lb, pred_ub)

        result = _add_sets([s1], [s2], 'sub')
        out = result[0]

        expected_V = V1 - V2
        assert np.allclose(out.V, expected_V)

    def test_star_addition_multiple_sets(self):
        """Adding lists of multiple Stars pair-wise."""
        V1 = np.array([[1.0, 0.5], [2.0, 0.3]])
        V2 = np.array([[0.5, 0.1], [1.0, 0.2]])
        C = np.array([[1.0]])
        d = np.array([[1.0]])
        pred_lb = np.array([[-1.0]])
        pred_ub = np.array([[1.0]])

        s1a = Star(V1, C, d, pred_lb, pred_ub)
        s1b = Star(V1 * 2, C, d, pred_lb, pred_ub)
        s2a = Star(V2, C, d, pred_lb, pred_ub)
        s2b = Star(V2 * 2, C, d, pred_lb, pred_ub)

        result = _add_sets([s1a, s1b], [s2a, s2b], 'add')
        assert len(result) == 2
        assert np.allclose(result[0].V, V1 + V2)
        assert np.allclose(result[1].V, V1 * 2 + V2 * 2)

    def test_star_preserves_constraints(self):
        """Constraints (C, d, pred_lb, pred_ub) are preserved from set a."""
        # V has 3 columns: 1 center + 2 predicate variables
        V1 = np.array([[1.0, 0.5, 0.2], [2.0, 0.3, 0.1]])
        V2 = np.array([[0.5, 0.1, 0.05], [1.0, 0.2, 0.1]])
        # C has 2 columns matching the 2 predicate variables
        C = np.array([[1.0, -1.0], [0.5, 0.5]])
        d = np.array([[2.0], [3.0]])
        pred_lb = np.array([[-2.0], [-1.0]])
        pred_ub = np.array([[2.0], [1.0]])

        s1 = Star(V1, C, d, pred_lb, pred_ub)
        s2 = Star(V2, C, d, pred_lb, pred_ub)

        result = _add_sets([s1], [s2], 'add')
        out = result[0]

        assert np.array_equal(out.C, C)
        assert np.array_equal(out.d, d)
        assert np.array_equal(out.predicate_lb, pred_lb)
        assert np.array_equal(out.predicate_ub, pred_ub)


class TestAddSetsImageStar:
    def test_imagestar_addition(self):
        """Adding two ImageStars: V_out = V1 + V2 element-wise."""
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))

        s1 = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)
        s2 = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        result = _add_sets([s1], [s2], 'add')
        out = result[0]

        assert isinstance(out, ImageStar)
        assert out.height == 2
        assert out.width == 2
        lb_out, ub_out = out.estimate_ranges()
        assert np.all(lb_out >= -1e-6)
        assert np.all(ub_out <= 2.0 + 1e-6)

    def test_imagestar_subtraction(self):
        """Subtracting two ImageStars: V_out = V1 - V2 element-wise."""
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))

        s1 = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)
        s2 = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        result = _add_sets([s1], [s2], 'sub')
        out = result[0]

        assert isinstance(out, ImageStar)
        # V_out = V1 - V2, so center = c1 - c2, generators = g1 - g2
        assert out.V.shape == s1.V.shape

    def test_imagestar_preserves_spatial_dimensions(self):
        """ImageStar addition preserves height, width, num_channels."""
        lb = np.zeros((3, 4, 2))
        ub = np.ones((3, 4, 2))

        s1 = ImageStar.from_bounds(lb, ub, height=3, width=4, num_channels=2)
        s2 = ImageStar.from_bounds(lb, ub, height=3, width=4, num_channels=2)

        result = _add_sets([s1], [s2], 'add')
        out = result[0]

        assert isinstance(out, ImageStar)
        assert out.height == 3
        assert out.width == 4
        assert out.num_channels == 2


class TestAddSetsZono:
    def test_zono_addition_concatenates_generators(self):
        """Adding two Zonos: concatenate generators."""
        c1 = np.array([[1.0], [2.0]])
        V1 = np.array([[0.5, 0.0], [0.0, 0.3]])
        c2 = np.array([[0.5], [1.0]])
        V2 = np.array([[0.1], [0.2]])

        z1 = Zono(c1, V1)
        z2 = Zono(c2, V2)

        result = _add_sets([z1], [z2], 'add')
        out = result[0]

        assert isinstance(out, Zono)
        assert np.allclose(out.c, c1 + c2)
        assert out.V.shape == (2, 3)  # concatenated generators
        assert np.allclose(out.V[:, :2], V1)
        assert np.allclose(out.V[:, 2:], V2)

    def test_zono_subtraction_negates_second_generators(self):
        """Subtracting Zonos: c1-c2, hstack(V1, -V2)."""
        c1 = np.array([[1.0], [2.0]])
        V1 = np.array([[0.5, 0.0], [0.0, 0.3]])
        c2 = np.array([[0.5], [1.0]])
        V2 = np.array([[0.1], [0.2]])

        z1 = Zono(c1, V1)
        z2 = Zono(c2, V2)

        result = _add_sets([z1], [z2], 'sub')
        out = result[0]

        assert isinstance(out, Zono)
        assert np.allclose(out.c, c1 - c2)
        assert out.V.shape == (2, 3)
        assert np.allclose(out.V[:, :2], V1)
        assert np.allclose(out.V[:, 2:], -V2)

    def test_zono_addition_multiple_sets(self):
        """Adding lists of Zonos pair-wise."""
        c1 = np.array([[1.0]])
        V1 = np.array([[0.5]])
        c2 = np.array([[2.0]])
        V2 = np.array([[0.3]])

        z1a = Zono(c1, V1)
        z1b = Zono(c1 * 2, V1)
        z2a = Zono(c2, V2)
        z2b = Zono(c2 * 2, V2)

        result = _add_sets([z1a, z1b], [z2a, z2b], 'add')
        assert len(result) == 2
        assert np.allclose(result[0].c, c1 + c2)
        assert np.allclose(result[1].c, c1 * 2 + c2 * 2)


class TestAddSetsImageZono:
    def test_imagezono_addition(self):
        """Adding two ImageZonos: concatenate generators, add centers."""
        c1 = np.array([[1.0], [2.0], [3.0], [4.0]])
        V1 = np.array([[0.1, 0.0], [0.0, 0.2], [0.1, 0.0], [0.0, 0.1]])
        c2 = np.array([[0.5], [0.5], [0.5], [0.5]])
        V2 = np.array([[0.05], [0.05], [0.05], [0.05]])

        iz1 = ImageZono(c1, V1, height=2, width=2, num_channels=1)
        iz2 = ImageZono(c2, V2, height=2, width=2, num_channels=1)

        result = _add_sets([iz1], [iz2], 'add')
        out = result[0]

        assert isinstance(out, ImageZono)
        assert np.allclose(out.c, c1 + c2)
        assert out.V.shape == (4, 3)  # 2 + 1 generators
        assert out.height == 2
        assert out.width == 2
        assert out.num_channels == 1


class TestAddSetsBox:
    def test_box_addition(self):
        """Adding two Boxes: lb1+lb2, ub1+ub2."""
        b1 = Box(np.array([[0.0], [1.0]]), np.array([[1.0], [2.0]]))
        b2 = Box(np.array([[0.5], [0.5]]), np.array([[1.5], [1.5]]))

        result = _add_sets([b1], [b2], 'add')
        out = result[0]

        assert isinstance(out, Box)
        assert np.allclose(out.lb, np.array([[0.5], [1.5]]))
        assert np.allclose(out.ub, np.array([[2.5], [3.5]]))

    def test_box_subtraction(self):
        """Subtracting Boxes: lb1-ub2, ub1-lb2."""
        b1 = Box(np.array([[1.0]]), np.array([[3.0]]))
        b2 = Box(np.array([[0.5]]), np.array([[1.5]]))

        result = _add_sets([b1], [b2], 'sub')
        out = result[0]

        assert np.allclose(out.lb, np.array([[-0.5]]))
        assert np.allclose(out.ub, np.array([[2.5]]))

    def test_box_addition_multiple_sets(self):
        """Adding lists of Boxes pair-wise."""
        b1a = Box(np.array([[0.0]]), np.array([[1.0]]))
        b1b = Box(np.array([[2.0]]), np.array([[3.0]]))
        b2a = Box(np.array([[0.5]]), np.array([[1.5]]))
        b2b = Box(np.array([[1.0]]), np.array([[2.0]]))

        result = _add_sets([b1a, b1b], [b2a, b2b], 'add')
        assert len(result) == 2
        assert np.allclose(result[0].lb, np.array([[0.5]]))
        assert np.allclose(result[0].ub, np.array([[2.5]]))
        assert np.allclose(result[1].lb, np.array([[3.0]]))
        assert np.allclose(result[1].ub, np.array([[5.0]]))


class TestCoerceSetTypes:
    """Test type coercion for mixed set types."""

    def test_imagestar_plus_star_flattens(self):
        """ImageStar + Star should flatten ImageStar to Star."""
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))
        istar = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        # Create a Star with same dimension (4)
        V2 = np.zeros((4, 5))
        V2[:, 0] = 0.5  # center
        for i in range(4):
            V2[i, i + 1] = 0.5  # generators
        C = np.eye(4)
        d = np.ones((4, 1))
        pred_lb = -np.ones((4, 1))
        pred_ub = np.ones((4, 1))
        star = Star(V2, C, d, pred_lb, pred_ub)

        result = _add_sets([istar], [star], 'add')
        out = result[0]

        # Should be a Star (not ImageStar) after coercion
        assert isinstance(out, Star)

    def test_imagezono_plus_zono_coerces(self):
        """ImageZono + Zono should convert ImageZono to Zono."""
        c1 = np.array([[1.0], [2.0], [3.0], [4.0]])
        V1 = np.array([[0.1], [0.2], [0.1], [0.1]])
        iz = ImageZono(c1, V1, height=2, width=2, num_channels=1)

        c2 = np.array([[0.5], [0.5], [0.5], [0.5]])
        V2 = np.array([[0.05], [0.05], [0.05], [0.05]])
        z = Zono(c2, V2)

        result = _add_sets([iz], [z], 'add')
        out = result[0]

        assert isinstance(out, Zono)
        assert np.allclose(out.c, c1 + c2)
