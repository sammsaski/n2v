"""Unit tests for synthetic test networks."""

import pytest
import sys
import os
import torch

# Add examples/ to path so we can import FlowConformal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'examples'))


class TestRotatedBananaNet:
    """Tests for RotatedBananaNet."""

    def test_output_shape(self):
        """Network should map R^2 -> R^2."""
        from FlowConformal.networks import RotatedBananaNet

        net = RotatedBananaNet()
        x = torch.rand(10, 2)
        y = net(x)
        assert y.shape == (10, 2)

    def test_produces_banana_shape(self):
        """Output of unit square should have parabolic structure.

        y1 ≈ x1, y2 ≈ x1^2 + 0.3*x2.
        So for inputs in [0,1]^2:
        - y1 range should be approximately [0, 1]
        - y2 should have a quadratic relationship with y1
        """
        from FlowConformal.networks import RotatedBananaNet

        torch.manual_seed(0)
        net = RotatedBananaNet()
        x = torch.rand(5000, 2)
        with torch.no_grad():
            y = net(x)

        # y1 should span roughly [0, 1]
        assert y[:, 0].min() < 0.2
        assert y[:, 0].max() > 0.8

        # y2 should correlate nonlinearly with y1 (not just linear)
        # Check that y2 at high y1 is significantly larger than at low y1
        low_y1 = y[y[:, 0] < 0.3]
        high_y1 = y[y[:, 0] > 0.7]
        assert high_y1[:, 1].mean() > low_y1[:, 1].mean()

    def test_output_is_deterministic(self):
        """Same input should give same output."""
        from FlowConformal.networks import RotatedBananaNet

        torch.manual_seed(42)
        net = RotatedBananaNet()
        x = torch.tensor([[0.5, 0.5]])
        y1 = net(x)
        y2 = net(x)
        torch.testing.assert_close(y1, y2)


class TestThreeBlobClassifier:
    """Tests for ThreeBlobClassifier."""

    def test_output_shape(self):
        """Network should map R^2 -> R^3 logits."""
        from FlowConformal.networks import ThreeBlobClassifier

        net = ThreeBlobClassifier()
        x = torch.randn(10, 2)
        logits = net(x)
        assert logits.shape == (10, 3)

    def test_accuracy_on_training_distribution(self):
        """Trained classifier should achieve >= 90% on its own distribution."""
        from FlowConformal.networks import ThreeBlobClassifier

        torch.manual_seed(0)
        net = ThreeBlobClassifier()
        x, y = net.sample_data(1000, seed=42)
        with torch.no_grad():
            preds = net(x).argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
        assert accuracy >= 0.9

    def test_sample_data_shapes(self):
        """sample_data should return (x, y) of matching shapes."""
        from FlowConformal.networks import ThreeBlobClassifier

        net = ThreeBlobClassifier()
        x, y = net.sample_data(100, seed=0)
        assert x.shape == (100, 2)
        assert y.shape == (100,)
