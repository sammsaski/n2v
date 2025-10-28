"""
Tests for layer dispatcher system.
"""

import pytest
import numpy as np
import torch.nn as nn
from n2v.sets import Star
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestDispatcherStar:
    """Tests for Star set dispatcher."""

    def test_dispatch_linear(self, simple_star):
        """Test dispatching Linear layer with Star."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = reach_layer(layer, [simple_star], method='exact')

        assert len(result) == 1
        assert result[0].dim == 2

    def test_dispatch_relu(self, simple_star):
        """Test dispatching ReLU layer with Star."""
        layer = nn.ReLU()

        result = reach_layer(layer, [simple_star], method='exact')

        assert len(result) >= 1
        for star in result:
            pytest.assert_star_valid(star)

    def test_dispatch_conv2d(self, simple_image_star):
        """Test dispatching Conv2D layer with ImageStar."""
        layer = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        layer.eval()

        result = reach_layer(layer, [simple_image_star], method='exact')

        assert len(result) == 1
        assert result[0].num_channels == 2

    def test_dispatch_maxpool2d(self, simple_image_star):
        """Test dispatching MaxPool2D layer with ImageStar."""
        layer = nn.MaxPool2d(2, 2)

        result = reach_layer(layer, [simple_image_star], method='exact')

        assert len(result) >= 1
        assert result[0].height == 2

    def test_dispatch_avgpool2d(self, simple_image_star):
        """Test dispatching AvgPool2D layer with ImageStar."""
        layer = nn.AvgPool2d(2, 2)

        result = reach_layer(layer, [simple_image_star], method='exact')

        # AvgPool should never split
        assert len(result) == 1
        assert result[0].height == 2

    def test_dispatch_flatten(self, simple_image_star):
        """Test dispatching Flatten layer with ImageStar."""
        layer = nn.Flatten()

        result = reach_layer(layer, [simple_image_star], method='exact')

        assert len(result) == 1
        assert result[0].dim == 16  # 4*4*1

    def test_dispatch_sequential(self, simple_star):
        """Test dispatching Sequential container."""
        layer = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU()
        )
        layer.eval()

        result = reach_layer(layer, [simple_star], method='exact')

        assert len(result) >= 1

    def test_unknown_layer_error(self, simple_star):
        """Test error on unknown layer type."""
        class CustomLayer(nn.Module):
            def forward(self, x):
                return x

        layer = CustomLayer()

        with pytest.raises((ValueError, NotImplementedError)):
            reach_layer(layer, [simple_star], method='exact')


class TestDispatcherZono:
    """Tests for Zonotope dispatcher."""

    def test_dispatch_linear(self, simple_zono):
        """Test dispatching Linear layer with Zono."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = reach_layer(layer, [simple_zono])

        assert len(result) == 1
        assert result[0].dim == 2

    def test_dispatch_relu(self, simple_zono):
        """Test dispatching ReLU layer with Zono."""
        layer = nn.ReLU()

        result = reach_layer(layer, [simple_zono])

        assert len(result) == 1
        pytest.assert_zono_valid(result[0])

    @pytest.mark.skip(reason="Conv2d not implemented for Zonotope")
    def test_dispatch_conv2d(self, simple_image_zono):
        """Test dispatching Conv2D layer with ImageZono."""
        layer = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        layer.eval()

        result = reach_layer(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].num_channels == 2

    def test_dispatch_maxpool2d(self, simple_image_zono):
        """Test dispatching MaxPool2D layer with ImageZono."""
        layer = nn.MaxPool2d(2, 2)

        result = reach_layer(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].height == 2

    def test_dispatch_avgpool2d(self, simple_image_zono):
        """Test dispatching AvgPool2D layer with ImageZono."""
        layer = nn.AvgPool2d(2, 2)

        result = reach_layer(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].height == 2

    def test_dispatch_flatten(self, simple_image_zono):
        """Test dispatching Flatten layer with ImageZono."""
        layer = nn.Flatten()

        result = reach_layer(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].dim == 16


class TestDispatcherBox:
    """Tests for Box dispatcher."""

    def test_dispatch_linear(self, simple_box):
        """Test dispatching Linear layer with Box."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = reach_layer(layer, [simple_box])

        assert len(result) == 1
        assert result[0].dim == 2

    def test_dispatch_relu(self, simple_box):
        """Test dispatching ReLU layer with Box."""
        layer = nn.ReLU()

        result = reach_layer(layer, [simple_box])

        assert len(result) == 1
        assert result[0].dim == simple_box.dim

    def test_dispatch_flatten(self):
        """Test dispatching Flatten layer with Box."""
        from n2v.sets import Box
        import numpy as np

        # Create a simple 2D image box
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))
        box = Box(lb.flatten().reshape(-1, 1), ub.flatten().reshape(-1, 1))

        layer = nn.Flatten()

        result = reach_layer(layer, [box])

        assert len(result) == 1


class TestDispatcherOptions:
    """Test dispatcher with different options."""

    def test_exact_vs_approx_relu(self, simple_star):
        """Test exact vs approx method for ReLU."""
        layer = nn.ReLU()

        exact_result = reach_layer(layer, [simple_star], method='exact')
        approx_result = reach_layer(layer, [simple_star], method='approx')

        # Approx should not split
        assert len(approx_result) <= len(exact_result)

    def test_maxpool_exact_vs_approx(self, simple_image_star):
        """Test exact vs approx method for MaxPool2D."""
        layer = nn.MaxPool2d(2, 2)

        exact_result = reach_layer(layer, [simple_image_star], method='exact')
        approx_result = reach_layer(layer, [simple_image_star], method='approx')

        # Approx should not split
        assert len(approx_result) <= len(exact_result)

    def test_display_option(self, capsys):
        """Test dis_opt='display' produces output."""
        layer = nn.ReLU()

        # Create a star that crosses zero to guarantee splitting (and thus output)
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        star_crossing_zero = Star.from_bounds(lb, ub)

        reach_layer(layer, [star_crossing_zero], method='exact', dis_opt='display')

        captured = capsys.readouterr()
        # Should print something about ReLU processing
        assert len(captured.out) > 0 or len(captured.err) > 0
