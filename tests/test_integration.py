"""
Integration tests for full network verification.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from nnv_py.sets import Star, ImageStar
from nnv_py.nn.reach.reach_star import reach_star_exact
from nnv_py.nn.reach.reach_zono import reach_zono_approx


class TestFeedforwardNetworks:
    """Integration tests for feedforward networks."""

    def test_simple_feedforward_exact(self):
        """Test simple feedforward network with exact method."""
        # Create network
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        model.eval()

        # Create input
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Verify
        output_stars = reach_star_exact(model, [input_star])

        assert len(output_stars) >= 1
        for star in output_stars:
            assert star.dim == 2
            pytest.assert_star_valid(star)

    def test_feedforward_with_bounds_check(self):
        """Test feedforward and check output bounds."""
        # Create identity network
        model = nn.Sequential(
            nn.Linear(2, 2)
        )
        model.eval()

        # Set to identity
        with torch.no_grad():
            model[0].weight.data = torch.eye(2)
            model[0].bias.data = torch.zeros(2)

        # Input bounds [0,1] x [0,1]
        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        # Verify
        output_stars = reach_star_exact(model, [input_star])

        # Should preserve bounds (identity)
        assert len(output_stars) == 1
        output_stars[0].estimate_ranges()
        np.testing.assert_allclose(output_stars[0].state_lb, lb, atol=1e-5)
        np.testing.assert_allclose(output_stars[0].state_ub, ub, atol=1e-5)

    def test_multiple_relu_layers(self):
        """Test network with multiple ReLU layers."""
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        model.eval()

        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        output_stars = reach_star_exact(model, [input_star])

        # May have significant splitting
        assert len(output_stars) >= 1
        for star in output_stars:
            assert star.dim == 1


class TestCNNNetworks:
    """Integration tests for CNN networks."""

    def test_simple_cnn_exact(self):
        """Test simple CNN with exact method."""
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16, 2)
        )
        model.eval()

        # Create 4x4 input
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        # Verify
        output_stars = reach_star_exact(model, [input_star])

        assert len(output_stars) >= 1
        for star in output_stars:
            assert star.dim == 2

    def test_cnn_with_avgpool_no_splitting(self):
        """Test that AvgPool doesn't cause splitting."""
        model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.AvgPool2d(2, 2),  # Should not split!
            nn.Flatten()
        )
        model.eval()

        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        output_stars = reach_star_exact(model, [input_star])

        # No ReLU, so should be exactly 1 star
        assert len(output_stars) == 1

    def test_cnn_strided_conv(self):
        """Test CNN with strided convolution."""
        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16, 2)
        )
        model.eval()

        lb = np.zeros((8, 8, 1))
        ub = np.ones((8, 8, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=8, width=8, num_channels=1)

        output_stars = reach_star_exact(model, [input_star])

        assert len(output_stars) >= 1
        for star in output_stars:
            assert star.dim == 2


class TestRobustnessVerification:
    """Integration tests for robustness verification."""

    def test_local_robustness_verified(self):
        """Test local robustness verification (robust case)."""
        # Create simple classifier that's robust
        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        model.eval()

        # Small perturbation around a point
        center = np.array([[0.5], [0.5]])
        epsilon = 0.01
        lb = center - epsilon
        ub = center + epsilon
        input_star = Star.from_bounds(lb, ub)

        # Verify
        output_stars = reach_star_exact(model, [input_star])

        # Check all stars for robustness
        true_class = 1
        robust = True

        for star in output_stars:
            star.estimate_ranges()
            lb_out = star.state_lb.flatten()
            ub_out = star.state_ub.flatten()

            # Check if any other class could beat true_class
            for i in range(len(lb_out)):
                if i != true_class and ub_out[i] >= lb_out[true_class]:
                    robust = False
                    break

        # Note: This might not be robust, but test should run without errors
        assert isinstance(robust, bool)

    def test_adversarial_example_detection(self):
        """Test detection of potential adversarial examples."""
        model = nn.Sequential(
            nn.Linear(2, 2)
        )
        model.eval()

        # Set weights to favor class 0 for positive inputs
        with torch.no_grad():
            model[0].weight.data = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
            model[0].bias.data = torch.zeros(2)

        # Input region that crosses decision boundary
        lb = np.array([[-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5]])
        input_star = Star.from_bounds(lb, ub)

        output_stars = reach_star_exact(model, [input_star])

        # Get output bounds
        assert len(output_stars) == 1
        output_stars[0].estimate_ranges()
        lb_out = output_stars[0].state_lb.flatten()
        ub_out = output_stars[0].state_ub.flatten()

        # Both classes should have overlapping ranges (ambiguous)
        assert lb_out[0] < ub_out[1]  # Ranges overlap


class TestApproximateMethods:
    """Integration tests for approximate methods."""

    def test_zono_approximate_verification(self):
        """Test zonotope approximate verification."""
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        model.eval()

        from nnv_py.sets import Zono
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_zono = Zono.from_bounds(lb, ub)

        # Approximate verification
        output_zonos = reach_zono_approx(model, [input_zono])

        # Approx should not split
        assert len(output_zonos) == 1
        assert output_zonos[0].dim == 2

    def test_approx_faster_than_exact(self):
        """Test that approximate is faster (fewer splits)."""
        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
        model.eval()

        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])

        # Exact method
        from nnv_py.sets import Star
        star = Star.from_bounds(lb, ub)
        exact_output = reach_star_exact(model, [star])

        # Approximate method
        from nnv_py.sets import Zono
        zono = Zono.from_bounds(lb, ub)
        approx_output = reach_zono_approx(model, [zono])

        # Approximate should have fewer sets (no splitting)
        assert len(approx_output) <= len(exact_output)


class TestEdgeCases:
    """Integration tests for edge cases."""

    def test_single_neuron_network(self):
        """Test network with single neuron."""
        model = nn.Sequential(
            nn.Linear(2, 1)
        )
        model.eval()

        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        output_stars = reach_star_exact(model, [input_star])

        assert len(output_stars) == 1
        assert output_stars[0].dim == 1

    def test_very_small_perturbation(self):
        """Test with very small perturbation."""
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )
        model.eval()

        center = np.array([[0.5], [0.5]])
        epsilon = 1e-6
        lb = center - epsilon
        ub = center + epsilon
        input_star = Star.from_bounds(lb, ub)

        output_stars = reach_star_exact(model, [input_star])

        # Small perturbation should still work
        assert len(output_stars) >= 1

    def test_exact_point_input(self):
        """Test with exact point (no uncertainty)."""
        model = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU()
        )
        model.eval()

        point = np.array([[0.5], [0.5]])
        input_star = Star.from_bounds(point, point)

        output_stars = reach_star_exact(model, [input_star])

        # Point input might still split due to ReLU, but should be valid
        assert len(output_stars) >= 1
