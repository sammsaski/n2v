"""Tests for set splitting in graph module reachability."""

import numpy as np
from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.nn.reach import _split_set


class TestSplitStarEven:
    """Test splitting a Star evenly into two chunks."""

    def test_split_star_even(self):
        """Split dim=4 Star into [2,2] -> 2 chunks, each dim=2, V rows match."""
        V = np.array([[1.0, 0.5, 0.0],
                      [2.0, 0.0, 0.3],
                      [3.0, 0.1, 0.0],
                      [4.0, 0.0, 0.2]])
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([[1.0], [1.0]])
        pred_lb = np.array([[-1.0], [-1.0]])
        pred_ub = np.array([[1.0], [1.0]])

        star = Star(V, C, d, pred_lb, pred_ub)

        chunks = _split_set(star, [2, 2], axis=0)

        assert len(chunks) == 2

        # First chunk: rows 0-1
        assert isinstance(chunks[0], Star)
        assert chunks[0].dim == 2
        assert np.allclose(chunks[0].V, V[0:2, :])
        assert np.array_equal(chunks[0].C, C)
        assert np.array_equal(chunks[0].d, d)
        assert np.array_equal(chunks[0].predicate_lb, pred_lb)
        assert np.array_equal(chunks[0].predicate_ub, pred_ub)

        # Second chunk: rows 2-3
        assert isinstance(chunks[1], Star)
        assert chunks[1].dim == 2
        assert np.allclose(chunks[1].V, V[2:4, :])
        assert np.array_equal(chunks[1].C, C)
        assert np.array_equal(chunks[1].d, d)


class TestSplitStarUneven:
    """Test splitting a Star unevenly."""

    def test_split_star_uneven(self):
        """Split dim=5 Star into [2,3] -> chunk dims are 2 and 3."""
        lb = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ub = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        star = Star.from_bounds(lb, ub)

        chunks = _split_set(star, [2, 3], axis=0)

        assert len(chunks) == 2
        assert chunks[0].dim == 2
        assert chunks[1].dim == 3

        # V rows should partition correctly
        assert np.allclose(chunks[0].V, star.V[0:2, :])
        assert np.allclose(chunks[1].V, star.V[2:5, :])


class TestSplitStarThreeChunks:
    """Test splitting a Star into three chunks."""

    def test_split_star_three_chunks(self):
        """Split dim=6 Star into [2,2,2] -> 3 chunks, each dim=2."""
        lb = np.arange(6, dtype=np.float64)
        ub = lb + 1.0
        star = Star.from_bounds(lb, ub)

        chunks = _split_set(star, [2, 2, 2], axis=0)

        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, Star)
            assert chunk.dim == 2
            assert np.allclose(chunk.V, star.V[i*2:(i+1)*2, :])


class TestSplitZono:
    """Test splitting a Zono."""

    def test_split_zono(self):
        """Split dim=4 Zono into [2,2] -> 2 chunks, c and V rows match."""
        c = np.array([[1.0], [2.0], [3.0], [4.0]])
        V = np.array([[0.5, 0.0],
                      [0.0, 0.3],
                      [0.1, 0.0],
                      [0.0, 0.2]])
        zono = Zono(c, V)

        chunks = _split_set(zono, [2, 2], axis=0)

        assert len(chunks) == 2

        assert isinstance(chunks[0], Zono)
        assert chunks[0].dim == 2
        assert np.allclose(chunks[0].c, c[0:2, :])
        assert np.allclose(chunks[0].V, V[0:2, :])

        assert isinstance(chunks[1], Zono)
        assert chunks[1].dim == 2
        assert np.allclose(chunks[1].c, c[2:4, :])
        assert np.allclose(chunks[1].V, V[2:4, :])


class TestSplitBox:
    """Test splitting a Box."""

    def test_split_box(self):
        """Split dim=4 Box into [1,3] -> chunks of dim=1 and dim=3."""
        lb = np.array([[0.0], [1.0], [2.0], [3.0]])
        ub = np.array([[1.0], [2.0], [3.0], [4.0]])
        box = Box(lb, ub)

        chunks = _split_set(box, [1, 3], axis=0)

        assert len(chunks) == 2

        assert isinstance(chunks[0], Box)
        assert chunks[0].dim == 1
        assert np.allclose(chunks[0].lb, lb[0:1, :])
        assert np.allclose(chunks[0].ub, ub[0:1, :])

        assert isinstance(chunks[1], Box)
        assert chunks[1].dim == 3
        assert np.allclose(chunks[1].lb, lb[1:4, :])
        assert np.allclose(chunks[1].ub, ub[1:4, :])


class TestSplitImageStarChannel:
    """Test splitting an ImageStar along the channel axis."""

    def test_split_imagestar_channel(self):
        """Split 4-channel ImageStar into [2,2] along axis=2 (C in HWC)."""
        lb = np.zeros((3, 3, 4))
        ub = np.ones((3, 3, 4))
        istar = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=4)

        # Split along channel axis (axis=2 in HWC)
        chunks = _split_set(istar, [2, 2], axis=2)

        assert len(chunks) == 2

        for chunk in chunks:
            assert isinstance(chunk, ImageStar)
            assert chunk.height == 3
            assert chunk.width == 3
            assert chunk.num_channels == 2
            # V shape: (H, W, C, nVar+1)
            assert chunk.V.shape[0] == 3
            assert chunk.V.shape[1] == 3
            assert chunk.V.shape[2] == 2

        # First chunk should have channels 0-1 of original V
        assert np.allclose(chunks[0].V, istar.V[:, :, 0:2, :])
        # Second chunk should have channels 2-3
        assert np.allclose(chunks[1].V, istar.V[:, :, 2:4, :])
