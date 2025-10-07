"""
Tests for model preprocessing utilities (BatchNorm fusion).
"""

import copy
import pytest
import torch
import torch.nn as nn

from n2v.utils.model_preprocessing import fuse_batchnorm, has_batchnorm


def _set_bn_nontrivial_stats(bn):
    """Set non-default running stats on a BatchNorm layer for meaningful testing."""
    with torch.no_grad():
        bn.running_mean.uniform_(-2.0, 2.0)
        bn.running_var.uniform_(0.5, 3.0)
        if bn.weight is not None:
            bn.weight.uniform_(0.5, 2.0)
        if bn.bias is not None:
            bn.bias.uniform_(-1.0, 1.0)


class TestFuseBatchNormConv2d:
    """Tests for Conv2d + BatchNorm2d fusion."""

    def test_conv_bn_fusion_matches_original(self):
        """Fused Conv2d+BN2d model produces identical outputs (atol=1e-5)."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        _set_bn_nontrivial_stats(model[1])
        model.eval()

        x = torch.randn(2, 3, 8, 8)
        expected = model(x)

        fused = fuse_batchnorm(model)
        actual = fused(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item()}"
        )

    def test_conv_bn_fusion_removes_batchnorm(self):
        """No BatchNorm layers remain after fusion."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        _set_bn_nontrivial_stats(model[1])
        model.eval()

        fused = fuse_batchnorm(model)
        assert not has_batchnorm(fused), "Fused model still contains BatchNorm layers"

    def test_conv_bn_no_bias(self):
        """Fusion works when Conv2d has bias=False."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        _set_bn_nontrivial_stats(model[1])
        model.eval()

        x = torch.randn(2, 3, 8, 8)
        expected = model(x)

        fused = fuse_batchnorm(model)
        actual = fused(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item()}"
        )
        # Fused conv should now have a bias
        fused_conv = list(fused.children()).__iter__().__next__()
        assert fused_conv.bias is not None, "Fused conv should have a bias after fusion"


class TestFuseBatchNormLinear:
    """Tests for Linear + BatchNorm1d fusion."""

    def test_linear_bn_fusion_matches_original(self):
        """Fused Linear+BN1d model produces identical outputs (atol=1e-5)."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(20, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
        _set_bn_nontrivial_stats(model[1])
        model.eval()

        x = torch.randn(4, 20)
        expected = model(x)

        fused = fuse_batchnorm(model)
        actual = fused(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item()}"
        )
        assert not has_batchnorm(fused), "Fused model still contains BatchNorm layers"


class TestFuseBatchNormEdgeCases:
    """Edge case tests for BatchNorm fusion."""

    def test_no_batchnorm_returns_copy(self):
        """Model without BN returns a deep copy (different object, same weights)."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        model.eval()

        fused = fuse_batchnorm(model)

        # Different object
        assert fused is not model, "fuse_batchnorm should return a new object"

        # Same weights
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), fused.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} differs"

    def test_standalone_batchnorm_kept(self):
        """BN without preceding Conv/Linear is NOT fused (remains in model)."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.BatchNorm1d(10),  # standalone, no preceding Linear
            nn.ReLU(),
            nn.Linear(10, 5),
        )
        _set_bn_nontrivial_stats(model[0])
        model.eval()

        fused = fuse_batchnorm(model)

        # The standalone BN should still be present
        assert has_batchnorm(fused), (
            "Standalone BatchNorm should remain (not fused)"
        )

        # Output should still match
        x = torch.randn(4, 10)
        expected = model(x)
        actual = fused(x)
        assert torch.allclose(expected, actual, atol=1e-5)

    def test_original_model_unchanged(self):
        """Original model weights are unchanged after fusion."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        _set_bn_nontrivial_stats(model[1])
        model.eval()

        # Save original weights
        orig_conv_weight = model[0].weight.clone()
        orig_conv_bias = model[0].bias.clone()
        orig_bn_weight = model[1].weight.clone()
        orig_bn_bias = model[1].bias.clone()
        orig_bn_mean = model[1].running_mean.clone()
        orig_bn_var = model[1].running_var.clone()

        _ = fuse_batchnorm(model)

        # Verify original is unchanged
        assert torch.equal(model[0].weight, orig_conv_weight)
        assert torch.equal(model[0].bias, orig_conv_bias)
        assert torch.equal(model[1].weight, orig_bn_weight)
        assert torch.equal(model[1].bias, orig_bn_bias)
        assert torch.equal(model[1].running_mean, orig_bn_mean)
        assert torch.equal(model[1].running_var, orig_bn_var)

    def test_nested_sequential(self):
        """Fusion works through nested Sequential modules."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            ),
        )
        _set_bn_nontrivial_stats(model[0][1])
        _set_bn_nontrivial_stats(model[1][1])
        model.eval()

        x = torch.randn(2, 3, 8, 8)
        expected = model(x)

        fused = fuse_batchnorm(model)
        actual = fused(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item()}"
        )
        assert not has_batchnorm(fused), "Fused model still contains BatchNorm layers"

    def test_multiple_conv_bn_pairs(self):
        """Multiple Conv+BN pairs all get fused."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
        )
        _set_bn_nontrivial_stats(model[1])
        _set_bn_nontrivial_stats(model[4])
        _set_bn_nontrivial_stats(model[7])
        model.eval()

        x = torch.randn(2, 3, 8, 8)
        expected = model(x)

        fused = fuse_batchnorm(model)
        actual = fused(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            f"Max diff: {(expected - actual).abs().max().item()}"
        )
        assert not has_batchnorm(fused), "Fused model still contains BatchNorm layers"

        # Count Identity layers (should have replaced 3 BN layers)
        identity_count = sum(
            1 for m in fused.modules() if isinstance(m, nn.Identity)
        )
        assert identity_count == 3, f"Expected 3 Identity layers, got {identity_count}"


class TestHasBatchNorm:
    """Tests for the has_batchnorm helper."""

    def test_model_with_batchnorm(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))
        assert has_batchnorm(model) is True

    def test_model_without_batchnorm(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        assert has_batchnorm(model) is False

    def test_nested_batchnorm(self):
        model = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8)),
            nn.ReLU(),
        )
        assert has_batchnorm(model) is True
