"""
Soundness tests for set slicing.

Tests verify that for random points sampled from the input set,
slicing those points produces outputs contained in the sliced set bounds.
"""

import numpy as np
from n2v.sets import Star, Zono, Box
from n2v.nn.reach import _slice_set


class TestSliceStarSoundness:
    """Soundness: slice a Star and verify sampled points are within bounds."""

    def test_slice_star_soundness(self):
        """Create Star from bounds [0..4] to [1..5], slice [1:4].
        Sample 200 points, verify point[1:4] is within sliced set bounds."""
        lb = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ub = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        star = Star.from_bounds(lb, ub)

        sliced = _slice_set(star, {0: slice(1, 4)})
        assert sliced.dim == 3

        # Get bounds from the sliced Star via LP
        lb_out, ub_out = sliced.get_ranges()

        np.random.seed(42)
        for _ in range(200):
            # Sample a random point from the input bounds
            alpha = np.random.uniform(0.0, 1.0, size=(5,))
            point = lb + alpha * (ub - lb)

            # Slice the point
            sliced_point = point[1:4]

            assert np.all(sliced_point >= lb_out.flatten() - 1e-6), (
                f"Sliced point {sliced_point} below lower bound {lb_out.flatten()}"
            )
            assert np.all(sliced_point <= ub_out.flatten() + 1e-6), (
                f"Sliced point {sliced_point} above upper bound {ub_out.flatten()}"
            )


class TestSliceZonoSoundness:
    """Soundness: slice a Zono and verify sampled points are within bounds."""

    def test_slice_zono_soundness(self):
        """Create Zono from bounds [0..4] to [1..5], slice [1:4].
        Sample 200 points, verify point[1:4] is within sliced set bounds."""
        lb = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ub = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        zono = Zono.from_bounds(lb, ub)

        sliced = _slice_set(zono, {0: slice(1, 4)})
        assert sliced.dim == 3

        # Get bounds from the sliced Zono
        lb_out, ub_out = sliced.get_bounds()

        np.random.seed(99)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(5,))
            point = lb + alpha * (ub - lb)
            sliced_point = point[1:4]

            assert np.all(sliced_point >= lb_out.flatten() - 1e-6), (
                f"Sliced point {sliced_point} below lower bound {lb_out.flatten()}"
            )
            assert np.all(sliced_point <= ub_out.flatten() + 1e-6), (
                f"Sliced point {sliced_point} above upper bound {ub_out.flatten()}"
            )


class TestSliceBoxSoundness:
    """Soundness: slice a Box and verify bounds are exact slices."""

    def test_slice_box_soundness(self):
        """Create Box from bounds [0..4] to [1..5], slice [1:4].
        Verify bounds are exact slices of original, and sampled points are within."""
        lb = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        ub = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        box = Box(lb, ub)

        sliced = _slice_set(box, {0: slice(1, 4)})
        assert sliced.dim == 3

        # Box bounds should be exact slices
        assert np.allclose(sliced.lb, lb[1:4, :])
        assert np.allclose(sliced.ub, ub[1:4, :])

        # Also verify sampled points
        np.random.seed(7)
        for _ in range(200):
            alpha = np.random.uniform(0.0, 1.0, size=(5,))
            point = lb.flatten() + alpha * (ub.flatten() - lb.flatten())
            sliced_point = point[1:4]

            assert np.all(sliced_point >= sliced.lb.flatten() - 1e-6), (
                f"Sliced point {sliced_point} below lower bound {sliced.lb.flatten()}"
            )
            assert np.all(sliced_point <= sliced.ub.flatten() + 1e-6), (
                f"Sliced point {sliced_point} above upper bound {sliced.ub.flatten()}"
            )
