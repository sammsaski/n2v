"""
Tests for image set representations: ImageStar, ImageZono.
"""

import pytest
import numpy as np
from n2v.sets import ImageStar, ImageZono, Star, Zono


class TestImageStar:
    """Tests for ImageStar set."""

    def test_creation(self, simple_image_star):
        """Test ImageStar creation."""
        assert simple_image_star.height == 4
        assert simple_image_star.width == 4
        assert simple_image_star.num_channels == 1
        assert simple_image_star.dim == 16  # 4*4*1
        pytest.assert_image_star_valid(simple_image_star)

    def test_from_bounds(self):
        """Test ImageStar creation from bounds."""
        lb = np.zeros((28, 28, 3))
        ub = np.ones((28, 28, 3))
        img_star = ImageStar.from_bounds(lb, ub, height=28, width=28, num_channels=3)

        assert img_star.height == 28
        assert img_star.width == 28
        assert img_star.num_channels == 3
        assert img_star.dim == 28 * 28 * 3
        pytest.assert_image_star_valid(img_star)

    def test_flatten_to_star(self, simple_image_star):
        """Test flattening to regular Star."""
        star = simple_image_star.flatten_to_star()

        assert isinstance(star, Star)
        assert star.dim == simple_image_star.dim
        assert star.nVar == simple_image_star.nVar
        pytest.assert_star_valid(star)

    def test_get_image_shape(self, simple_image_star):
        """Test getting image shape."""
        shape = simple_image_star.get_image_shape()

        assert shape == (4, 4, 1)

    def test_to_star(self, simple_image_star):
        """Test conversion to regular Star."""
        star = simple_image_star.to_star()

        assert isinstance(star, Star)
        assert star.dim == simple_image_star.dim

    def test_dimension_validation(self):
        """Test that dimension mismatch raises error."""
        V = np.random.rand(10, 3)  # Wrong size for 4x4x1
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = np.array([[1.0], [1.0]])

        with pytest.raises(ValueError):
            ImageStar(V, C, d, height=4, width=4, num_channels=1)

    def test_estimate_ranges(self, simple_image_star):
        """Test range estimation for ImageStar."""
        simple_image_star.estimate_ranges()

        assert simple_image_star.state_lb is not None
        assert simple_image_star.state_ub is not None
        assert simple_image_star.state_lb.shape[0] == simple_image_star.dim

    def test_sample(self):
        """Test sampling images from ImageStar."""
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        samples = img_star.sample(5)

        assert len(samples) == 5
        for sample in samples:
            assert sample.shape == (4, 4, 1)
            # All samples should be within bounds
            assert np.all(sample >= lb - 1e-6)
            assert np.all(sample <= ub + 1e-6)

    def test_sample_single_point(self):
        """Test sampling from ImageStar with no uncertainty."""
        # Create ImageStar with lb == ub (single point)
        image = np.random.rand(4, 4, 1)
        lb = image.copy()
        ub = image.copy()
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        samples = img_star.sample(3)

        assert len(samples) == 3
        for sample in samples:
            np.testing.assert_allclose(sample, image, atol=1e-6)

    def test_evaluate(self):
        """Test evaluating ImageStar at specific predicate values."""
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        # Evaluate at predicate bounds (all -1 gives lower bound)
        pred_val_low = -np.ones(img_star.nVar)
        image_low = img_star.evaluate(pred_val_low)
        np.testing.assert_allclose(image_low, lb, atol=1e-6)

        # Evaluate at upper bound predicates (all +1 gives upper bound)
        pred_val_high = np.ones(img_star.nVar)
        image_high = img_star.evaluate(pred_val_high)
        np.testing.assert_allclose(image_high, ub, atol=1e-6)

    def test_evaluate_wrong_dimension(self):
        """Test that evaluate raises error for wrong predicate dimension."""
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        wrong_pred = np.ones(img_star.nVar + 5)

        with pytest.raises(ValueError):
            img_star.evaluate(wrong_pred)

    def test_contains_center(self):
        """Test that center image is contained in ImageStar."""
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        # Center image (at predicate = 0)
        center = img_star.evaluate(np.zeros(img_star.nVar))

        assert img_star.contains(center)

    def test_contains_bounds(self):
        """Test that bound images are contained in ImageStar."""
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        assert img_star.contains(lb)
        assert img_star.contains(ub)

    def test_contains_outside(self):
        """Test that images outside bounds are not contained."""
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        outside_image = np.ones((4, 4, 1)) * 2  # Above upper bound
        assert not img_star.contains(outside_image)

    def test_contains_flattened_input(self):
        """Test contains with flattened image input."""
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        center = img_star.evaluate(np.zeros(img_star.nVar))
        center_flat = center.flatten()

        assert img_star.contains(center_flat)

    def test_contains_wrong_shape(self):
        """Test that contains raises error for wrong image shape."""
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        wrong_shape = np.zeros((8, 8, 1))

        with pytest.raises(ValueError):
            img_star.contains(wrong_shape)


class TestImageZono:
    """Tests for ImageZono set."""

    def test_creation(self, simple_image_zono):
        """Test ImageZono creation."""
        assert simple_image_zono.height == 4
        assert simple_image_zono.width == 4
        assert simple_image_zono.num_channels == 1
        assert simple_image_zono.dim == 16

    def test_from_bounds(self):
        """Test ImageZono creation from bounds."""
        lb = np.zeros((8, 8, 3))
        ub = np.ones((8, 8, 3))
        img_zono = ImageZono.from_bounds(lb, ub, height=8, width=8, num_channels=3)

        assert img_zono.height == 8
        assert img_zono.width == 8
        assert img_zono.num_channels == 3
        assert img_zono.dim == 8 * 8 * 3

    def test_get_image_shape(self, simple_image_zono):
        """Test getting image shape."""
        shape = simple_image_zono.get_image_shape()

        assert shape == (4, 4, 1)

    def test_to_zono(self, simple_image_zono):
        """Test conversion to regular Zono."""
        zono = simple_image_zono.to_zono()

        assert isinstance(zono, Zono)
        assert zono.dim == simple_image_zono.dim

    def test_get_bounds(self, simple_image_zono):
        """Test bounds computation."""
        lb, ub = simple_image_zono.get_bounds()

        assert lb.shape[0] == simple_image_zono.dim
        assert ub.shape[0] == simple_image_zono.dim
        assert np.all(lb <= ub)

    def test_dimension_validation(self):
        """Test that dimension mismatch raises error."""
        c = np.random.rand(10, 1)  # Wrong size
        V = np.random.rand(10, 2)

        with pytest.raises(ValueError):
            ImageZono(c, V, height=4, width=4, num_channels=1)


class TestImageSetOperations:
    """Test operations specific to image sets."""

    def test_image_star_perturbation(self):
        """Test creating ImageStar from perturbed image."""
        image = np.random.rand(28, 28, 1) * 0.5 + 0.25  # [0.25, 0.75]
        epsilon = 0.1

        lb = np.maximum(image - epsilon, 0)
        ub = np.minimum(image + epsilon, 1)

        img_star = ImageStar.from_bounds(lb, ub, height=28, width=28, num_channels=1)

        # Check bounds are within valid range
        img_star.estimate_ranges()
        assert np.all(img_star.state_lb >= -0.01)  # Allow small numerical error
        assert np.all(img_star.state_ub <= 1.01)

    def test_multichannel_image_star(self):
        """Test ImageStar with multiple channels."""
        lb = np.zeros((16, 16, 3))
        ub = np.ones((16, 16, 3))

        img_star = ImageStar.from_bounds(lb, ub, height=16, width=16, num_channels=3)

        assert img_star.num_channels == 3
        assert img_star.dim == 16 * 16 * 3

    def test_image_zono_perturbation(self):
        """Test creating ImageZono from perturbed image."""
        image = np.random.rand(8, 8, 1) * 0.5 + 0.25
        epsilon = 0.05

        lb = np.maximum(image - epsilon, 0)
        ub = np.minimum(image + epsilon, 1)

        img_zono = ImageZono.from_bounds(lb, ub, height=8, width=8, num_channels=1)

        # Check bounds match
        computed_lb, computed_ub = img_zono.get_bounds()
        computed_lb = computed_lb.reshape(8, 8, 1)
        computed_ub = computed_ub.reshape(8, 8, 1)

        np.testing.assert_allclose(computed_lb, lb, atol=1e-6)
        np.testing.assert_allclose(computed_ub, ub, atol=1e-6)
