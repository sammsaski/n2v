"""
Soundness tests for set splitting.

Tests verify that for random points sampled from the input set,
splitting those points produces outputs contained in the split set bounds.
"""

import numpy as np
from n2v.sets import Star, Zono, Box
from n2v.nn.reach import _split_set


class TestSplitStarSoundness:
    """Soundness: split a Star and verify sampled points are within bounds."""

    def test_split_star_soundness(self):
        """Create Star from bounds, split into [2,3].
        Sample 200 points, verify each chunk bounds contain corresponding slice."""
        lb = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ub = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        star = Star.from_bounds(lb, ub)

        chunks = _split_set(star, [2, 3], axis=0)
        assert len(chunks) == 2
        assert chunks[0].dim == 2
        assert chunks[1].dim == 3

        # Get bounds from each chunk via LP
        lb0, ub0 = chunks[0].get_ranges()
        lb1, ub1 = chunks[1].get_ranges()

        np.random.seed(42)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(5,))
            point = lb + alpha * (ub - lb)

            # First chunk corresponds to indices [0:2]
            chunk0_point = point[0:2]
            assert np.all(chunk0_point >= lb0.flatten() - 1e-6), (
                f"Chunk 0 point {chunk0_point} below lower bound {lb0.flatten()}"
            )
            assert np.all(chunk0_point <= ub0.flatten() + 1e-6), (
                f"Chunk 0 point {chunk0_point} above upper bound {ub0.flatten()}"
            )

            # Second chunk corresponds to indices [2:5]
            chunk1_point = point[2:5]
            assert np.all(chunk1_point >= lb1.flatten() - 1e-6), (
                f"Chunk 1 point {chunk1_point} below lower bound {lb1.flatten()}"
            )
            assert np.all(chunk1_point <= ub1.flatten() + 1e-6), (
                f"Chunk 1 point {chunk1_point} above upper bound {ub1.flatten()}"
            )


class TestSplitZonoSoundness:
    """Soundness: split a Zono and verify sampled points are within bounds."""

    def test_split_zono_soundness(self):
        """Create Zono from bounds, split into [2,2].
        Sample 200 points, verify each chunk bounds contain corresponding slice."""
        lb = np.array([0.0, 1.0, 2.0, 3.0])
        ub = np.array([1.0, 2.0, 3.0, 4.0])
        zono = Zono.from_bounds(lb, ub)

        chunks = _split_set(zono, [2, 2], axis=0)
        assert len(chunks) == 2

        lb0, ub0 = chunks[0].get_bounds()
        lb1, ub1 = chunks[1].get_bounds()

        np.random.seed(99)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(4,))
            point = lb + alpha * (ub - lb)

            chunk0_point = point[0:2]
            assert np.all(chunk0_point >= lb0.flatten() - 1e-6), (
                f"Chunk 0 point {chunk0_point} below lower bound {lb0.flatten()}"
            )
            assert np.all(chunk0_point <= ub0.flatten() + 1e-6), (
                f"Chunk 0 point {chunk0_point} above upper bound {ub0.flatten()}"
            )

            chunk1_point = point[2:4]
            assert np.all(chunk1_point >= lb1.flatten() - 1e-6), (
                f"Chunk 1 point {chunk1_point} below lower bound {lb1.flatten()}"
            )
            assert np.all(chunk1_point <= ub1.flatten() + 1e-6), (
                f"Chunk 1 point {chunk1_point} above upper bound {ub1.flatten()}"
            )


class TestSplitBoxSoundness:
    """Soundness: split a Box and verify bounds are exact slices."""

    def test_split_box_soundness(self):
        """Create Box from bounds, split into [1,3].
        Verify bounds are exact slices of original and sampled points are within."""
        lb = np.array([[0.0], [1.0], [2.0], [3.0]])
        ub = np.array([[1.0], [2.0], [3.0], [4.0]])
        box = Box(lb, ub)

        chunks = _split_set(box, [1, 3], axis=0)
        assert len(chunks) == 2

        # Box bounds should be exact slices
        assert np.allclose(chunks[0].lb, lb[0:1, :])
        assert np.allclose(chunks[0].ub, ub[0:1, :])
        assert np.allclose(chunks[1].lb, lb[1:4, :])
        assert np.allclose(chunks[1].ub, ub[1:4, :])

        # Also verify sampled points
        np.random.seed(7)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(4,))
            point = lb.flatten() + alpha * (ub.flatten() - lb.flatten())

            chunk0_point = point[0:1]
            assert np.all(chunk0_point >= chunks[0].lb.flatten() - 1e-6)
            assert np.all(chunk0_point <= chunks[0].ub.flatten() + 1e-6)

            chunk1_point = point[1:4]
            assert np.all(chunk1_point >= chunks[1].lb.flatten() - 1e-6)
            assert np.all(chunk1_point <= chunks[1].ub.flatten() + 1e-6)
