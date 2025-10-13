"""
Soundness tests for parallel LP solving.

These tests verify that parallel LP solving maintains mathematical correctness
and soundness properties.
"""

import pytest
import numpy as np

import n2v
from n2v.sets import Star, Box, Hexatope, Octatope
from n2v.config import set_parallel, config


class TestParallelLPSoundness:
    """Test soundness of parallel LP solving."""

    def setup_method(self):
        """Reset configuration before each test."""
        config.reset()

    def test_simple_bounds_soundness(self):
        """Test parallel LP on simple box-based Star sets."""
        # Create Star from bounds
        lb = np.array([[0], [0], [0]], dtype=np.float32)
        ub = np.array([[1], [1], [1]], dtype=np.float32)
        star = Star.from_bounds(lb, ub)

        # Get ranges with sequential
        set_parallel(False)
        lb_seq, ub_seq = star.get_ranges()

        # Get ranges with parallel
        set_parallel(True, n_workers=4)
        lb_par, ub_par = star.get_ranges()

        # Verify exact equality
        np.testing.assert_allclose(lb_seq, lb_par, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ub_seq, ub_par, rtol=1e-10, atol=1e-10)

    def test_box_overapproximation_soundness(self):
        """Test that Box overapproximation is sound with parallel LP."""
        # Create 10-dimensional Star
        lb = np.zeros((10, 1), dtype=np.float32)
        ub = np.ones((10, 1), dtype=np.float32)
        star = Star.from_bounds(lb, ub)

        # Get Box with sequential
        set_parallel(False)
        box_seq = star.get_box()

        # Get Box with parallel
        set_parallel(True, n_workers=4)
        box_par = star.get_box()

        # Boxes must be identical
        np.testing.assert_allclose(box_seq.lb, box_par.lb, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(box_seq.ub, box_par.ub, rtol=1e-10, atol=1e-10)

    def test_high_dimensional_soundness(self):
        """Test soundness on high-dimensional problems where parallel is beneficial."""
        np.random.seed(42)  # For reproducibility
        dim = 25
        lb = np.random.randn(dim, 1).astype(np.float32) - 1.0
        ub = np.random.randn(dim, 1).astype(np.float32) + 1.0
        # Ensure lb < ub
        lb, ub = np.minimum(lb, ub), np.maximum(lb, ub)

        star = Star.from_bounds(lb, ub)

        # Compare methods
        set_parallel(False)
        lb_seq, ub_seq = star.get_ranges()

        set_parallel(True, n_workers=4)
        lb_par, ub_par = star.get_ranges()

        np.testing.assert_allclose(lb_seq, lb_par, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ub_seq, ub_par, rtol=1e-5, atol=1e-6)

    def test_auto_parallel_soundness(self):
        """Test that auto-parallel mode maintains soundness."""
        # Small dimension (below threshold)
        star_small = Star.from_bounds(
            np.zeros((5, 1), dtype=np.float32),
            np.ones((5, 1), dtype=np.float32)
        )

        # Large dimension (above threshold)
        star_large = Star.from_bounds(
            np.zeros((15, 1), dtype=np.float32),
            np.ones((15, 1), dtype=np.float32)
        )

        # Get reference with explicit sequential
        set_parallel(False)
        lb_small_ref, ub_small_ref = star_small.get_ranges()
        lb_large_ref, ub_large_ref = star_large.get_ranges()

        # Test with auto mode
        set_parallel('auto', threshold=10)
        lb_small_auto, ub_small_auto = star_small.get_ranges()
        lb_large_auto, ub_large_auto = star_large.get_ranges()

        # Must be sound regardless of which mode is chosen
        np.testing.assert_allclose(lb_small_ref, lb_small_auto, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ub_small_ref, ub_small_auto, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(lb_large_ref, lb_large_auto, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ub_large_ref, ub_large_auto, rtol=1e-10, atol=1e-10)

    def test_repeated_calls_soundness(self):
        """Test that repeated parallel calls give consistent results."""
        dim = 12
        star = Star.from_bounds(
            np.zeros((dim, 1), dtype=np.float32),
            np.ones((dim, 1), dtype=np.float32)
        )

        set_parallel(True, n_workers=4)

        # Call multiple times
        results = [star.get_ranges() for _ in range(5)]

        # All results should be identical
        lb_ref, ub_ref = results[0]
        for lb, ub in results[1:]:
            np.testing.assert_allclose(lb, lb_ref, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(ub, ub_ref, rtol=1e-10, atol=1e-10)

    def test_different_worker_counts_soundness(self):
        """Test that different worker counts produce sound results."""
        dim = 16
        star = Star.from_bounds(
            np.zeros((dim, 1), dtype=np.float32),
            np.ones((dim, 1), dtype=np.float32)
        )

        # Get reference
        set_parallel(False)
        lb_ref, ub_ref = star.get_ranges()

        # Test with various worker counts
        for n_workers in [1, 2, 4, 8]:
            set_parallel(True, n_workers=n_workers)
            lb_par, ub_par = star.get_ranges()

            np.testing.assert_allclose(lb_par, lb_ref, rtol=1e-5, atol=1e-6,
                                       err_msg=f"Failed with {n_workers} workers")
            np.testing.assert_allclose(ub_par, ub_ref, rtol=1e-5, atol=1e-6,
                                       err_msg=f"Failed with {n_workers} workers")

    def test_mixed_bounds_soundness(self):
        """Test soundness with mixed positive/negative bounds."""
        np.random.seed(123)
        dim = 20
        lb = (np.random.randn(dim, 1).astype(np.float32) - 0.5) * 2
        ub = (np.random.randn(dim, 1).astype(np.float32) + 0.5) * 2
        # Ensure lb < ub
        lb, ub = np.minimum(lb, ub), np.maximum(lb, ub)
        # Ensure there's at least 0.1 gap
        ub = np.maximum(ub, lb + 0.1)

        star = Star.from_bounds(lb, ub)

        set_parallel(False)
        lb_seq, ub_seq = star.get_ranges()

        set_parallel(True, n_workers=4)
        lb_par, ub_par = star.get_ranges()

        np.testing.assert_allclose(lb_seq, lb_par, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ub_seq, ub_par, rtol=1e-5, atol=1e-6)


# ============================================================================
# Star-Level Parallelization Soundness Tests
# ============================================================================

class TestStarParallelSoundness:
    """Test soundness of Star-level parallel processing."""

    def setup_method(self):
        """Reset configuration before each test."""
        config.reset()

    def test_relu_soundness_multiple_stars(self):
        """Test that parallel ReLU maintains soundness for multiple Stars."""
        from n2v.nn.layer_ops.relu_reach import relu_star_exact

        # Create multiple input Stars
        input_stars = [
            Star.from_bounds(
                np.array([[-1], [0.5]], dtype=np.float32),
                np.array([[0.5], [1.5]], dtype=np.float32)
            ),
            Star.from_bounds(
                np.array([[0.2], [-0.8]], dtype=np.float32),
                np.array([[1.2], [0.3]], dtype=np.float32)
            ),
            Star.from_bounds(
                np.array([[-0.5], [-0.3]], dtype=np.float32),
                np.array([[0.5], [0.7]], dtype=np.float32)
            )
        ]

        # Sequential processing
        set_parallel(False)
        output_seq = relu_star_exact(input_stars)

        # Get overall bounds from sequential
        lb_seq_overall = np.ones((2, 1)) * np.inf
        ub_seq_overall = np.ones((2, 1)) * -np.inf
        for star in output_seq:
            lb, ub = star.get_ranges()
            lb_seq_overall = np.minimum(lb_seq_overall, lb)
            ub_seq_overall = np.maximum(ub_seq_overall, ub)

        # Parallel processing
        set_parallel(True, n_workers=4)
        output_par = relu_star_exact(input_stars)

        # Get overall bounds from parallel
        lb_par_overall = np.ones((2, 1)) * np.inf
        ub_par_overall = np.ones((2, 1)) * -np.inf
        for star in output_par:
            lb, ub = star.get_ranges()
            lb_par_overall = np.minimum(lb_par_overall, lb)
            ub_par_overall = np.maximum(ub_par_overall, ub)

        # Soundness: bounds must be identical
        np.testing.assert_allclose(lb_seq_overall, lb_par_overall, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ub_seq_overall, ub_par_overall, rtol=1e-5, atol=1e-6)

    def test_star_count_preservation(self):
        """Test that parallel processing produces same number of Stars."""
        from n2v.nn.layer_ops.relu_reach import relu_star_exact

        input_stars = [
            Star.from_bounds(
                np.array([[-1 + i*0.3], [0.5]], dtype=np.float32),
                np.array([[0.5 + i*0.3], [1.5]], dtype=np.float32)
            )
            for i in range(6)
        ]

        # Sequential
        output_seq = relu_star_exact(input_stars, parallel=False)

        # Parallel
        output_par = relu_star_exact(input_stars, parallel=True, n_workers=4)

        # Must produce same count
        assert len(output_seq) == len(output_par), \
            f"Star counts differ: sequential={len(output_seq)}, parallel={len(output_par)}"

    def test_individual_star_soundness(self):
        """Test soundness of each individual Star produced."""
        from n2v.nn.layer_ops.relu_reach import relu_star_exact

        input_stars = [
            Star.from_bounds(
                np.array([[-1], [-1]], dtype=np.float32),
                np.array([[1], [1]], dtype=np.float32)
            ),
            Star.from_bounds(
                np.array([[0], [0]], dtype=np.float32),
                np.array([[2], [2]], dtype=np.float32)
            )
        ]

        # Sequential
        output_seq = relu_star_exact(input_stars, parallel=False)

        # Parallel
        output_par = relu_star_exact(input_stars, parallel=True, n_workers=2)

        # Compare each Star's properties
        assert len(output_seq) == len(output_par)

        # Sort by bounds for consistent comparison
        seq_sorted = sorted(output_seq, key=lambda s: (s.get_ranges()[0].sum(), s.get_ranges()[1].sum()))
        par_sorted = sorted(output_par, key=lambda s: (s.get_ranges()[0].sum(), s.get_ranges()[1].sum()))

        for star_seq, star_par in zip(seq_sorted, par_sorted):
            # Compare dimensions
            assert star_seq.dim == star_par.dim

            # Compare ranges
            lb_seq, ub_seq = star_seq.get_ranges()
            lb_par, ub_par = star_par.get_ranges()

            np.testing.assert_allclose(lb_seq, lb_par, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(ub_seq, ub_par, rtol=1e-5, atol=1e-6)

    def test_relu_splitting_soundness(self):
        """Test soundness when ReLU causes Star splitting."""
        from n2v.nn.layer_ops.relu_reach import relu_star_exact

        # Create Stars that will split (cross zero boundary)
        input_stars = [
            Star.from_bounds(
                np.array([[-1], [-1], [-1]], dtype=np.float32),
                np.array([[1], [1], [1]], dtype=np.float32)
            ),
            Star.from_bounds(
                np.array([[-0.5], [-0.5], [-0.5]], dtype=np.float32),
                np.array([[0.5], [0.5], [0.5]], dtype=np.float32)
            )
        ]

        # Sequential
        output_seq = relu_star_exact(input_stars, parallel=False)

        # Parallel
        output_par = relu_star_exact(input_stars, parallel=True, n_workers=2)

        # Both should produce many Stars (due to splitting)
        assert len(output_seq) > len(input_stars)
        assert len(output_par) > len(input_stars)
        assert len(output_seq) == len(output_par)


class TestHexatopeSoundness:
    """Test soundness of Hexatope operations."""

    def setup_method(self):
        """Reset configuration before each test."""
        config.reset()

    def test_hexatope_bounds_computation(self):
        """Test that hexatope bounds computation is sound."""
        lb = np.array([[0], [0], [0]], dtype=np.float32)
        ub = np.array([[1], [1], [1]], dtype=np.float32)
        hexatope = Hexatope.from_bounds(lb, ub)

        # Get ranges
        lb_out, ub_out = hexatope.estimate_ranges()

        # Verify soundness
        assert np.all(lb_out >= lb - 1e-6)
        assert np.all(ub_out <= ub + 1e-6)

    def test_hexatope_affine_transformation(self):
        """Test that hexatope affine transformation is sound."""
        lb = np.zeros((3, 1), dtype=np.float32)
        ub = np.ones((3, 1), dtype=np.float32)
        hexatope = Hexatope.from_bounds(lb, ub)

        # Apply affine transformation: y = 2x + 1
        W = np.eye(3) * 2
        b = np.ones((3, 1))
        result = hexatope.affine_map(W, b)

        # Verify bounds
        lb_out, ub_out = result.estimate_ranges()
        expected_lb = 2 * lb + b
        expected_ub = 2 * ub + b

        np.testing.assert_allclose(lb_out, expected_lb, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ub_out, expected_ub, rtol=1e-5, atol=1e-6)

    def test_hexatope_negative_bounds(self):
        """Test hexatope with negative bounds."""
        lb = np.array([[-2], [-3], [-1]], dtype=np.float32)
        ub = np.array([[-1], [-1], [0]], dtype=np.float32)
        hexatope = Hexatope.from_bounds(lb, ub)

        lb_out, ub_out = hexatope.estimate_ranges()

        np.testing.assert_allclose(lb_out, lb, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ub_out, ub, rtol=1e-5, atol=1e-6)

    def test_hexatope_dimension_expansion(self):
        """Test hexatope dimension expansion."""
        lb = np.zeros((2, 1), dtype=np.float32)
        ub = np.ones((2, 1), dtype=np.float32)
        hexatope = Hexatope.from_bounds(lb, ub)

        # Expand to 3D
        W = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        b = np.zeros((3, 1), dtype=np.float32)
        result = hexatope.affine_map(W, b)

        # Verify output dimension
        assert result.dim == 3

    def test_hexatope_to_box_conversion(self):
        """Test hexatope estimate_ranges soundness."""
        lb = np.array([[1], [2], [3]], dtype=np.float32)
        ub = np.array([[4], [5], [6]], dtype=np.float32)
        hexatope = Hexatope.from_bounds(lb, ub)

        # Get ranges using estimation (exact get_ranges requires full DCS implementation)
        lb_out, ub_out = hexatope.estimate_ranges()

        # Estimated ranges should contain the original bounds
        np.testing.assert_allclose(lb_out, lb, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ub_out, ub, rtol=1e-5, atol=1e-6)


class TestOctatopeSoundness:
    """Test soundness of Octatope operations."""

    def setup_method(self):
        """Reset configuration before each test."""
        config.reset()

    def test_octatope_bounds_computation(self):
        """Test that octatope bounds computation is sound."""
        lb = np.array([[0], [0], [0]], dtype=np.float32)
        ub = np.array([[1], [1], [1]], dtype=np.float32)
        octatope = Octatope.from_bounds(lb, ub)

        # Get ranges
        lb_out, ub_out = octatope.estimate_ranges()

        # Verify soundness
        assert np.all(lb_out >= lb - 1e-6)
        assert np.all(ub_out <= ub + 1e-6)

    def test_octatope_affine_transformation(self):
        """Test that octatope affine transformation is sound."""
        lb = np.zeros((3, 1), dtype=np.float32)
        ub = np.ones((3, 1), dtype=np.float32)
        octatope = Octatope.from_bounds(lb, ub)

        # Apply affine transformation: y = 2x + 1
        W = np.eye(3) * 2
        b = np.ones((3, 1))
        result = octatope.affine_map(W, b)

        # Verify bounds
        lb_out, ub_out = result.estimate_ranges()
        expected_lb = 2 * lb + b
        expected_ub = 2 * ub + b

        np.testing.assert_allclose(lb_out, expected_lb, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ub_out, expected_ub, rtol=1e-5, atol=1e-6)

    def test_octatope_negative_bounds(self):
        """Test octatope with negative bounds."""
        lb = np.array([[-2], [-3], [-1]], dtype=np.float32)
        ub = np.array([[-1], [-1], [0]], dtype=np.float32)
        octatope = Octatope.from_bounds(lb, ub)

        lb_out, ub_out = octatope.estimate_ranges()

        np.testing.assert_allclose(lb_out, lb, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ub_out, ub, rtol=1e-5, atol=1e-6)

    def test_octatope_dimension_expansion(self):
        """Test octatope dimension expansion."""
        lb = np.zeros((2, 1), dtype=np.float32)
        ub = np.ones((2, 1), dtype=np.float32)
        octatope = Octatope.from_bounds(lb, ub)

        # Expand to 3D
        W = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        b = np.zeros((3, 1), dtype=np.float32)
        result = octatope.affine_map(W, b)

        # Verify output dimension
        assert result.dim == 3

    def test_octatope_to_box_conversion(self):
        """Test octatope estimate_ranges soundness."""
        lb = np.array([[1], [2], [3]], dtype=np.float32)
        ub = np.array([[4], [5], [6]], dtype=np.float32)
        octatope = Octatope.from_bounds(lb, ub)

        # Get ranges using estimation (exact get_ranges requires full UTVPI implementation)
        lb_out, ub_out = octatope.estimate_ranges()

        # Estimated ranges should contain the original bounds
        np.testing.assert_allclose(lb_out, lb, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ub_out, ub, rtol=1e-5, atol=1e-6)
