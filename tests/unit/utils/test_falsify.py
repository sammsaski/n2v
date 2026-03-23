"""
Unit tests for falsification techniques.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from n2v.utils import falsify
from n2v.sets import HalfSpace


class TestFalsifyRandom:
    """Tests for random sampling falsification (method='random')."""

    def test_finds_counterexample(self):
        """Test that random sampling finds a counterexample when one exists."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Property: output[0] >= 0.5 is unsafe
        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='random', n_samples=100, seed=42)

        assert result == 0, "Should find counterexample (SAT)"
        assert cex is not None, "Counterexample should be returned"
        inp, out = cex
        assert inp[0] >= 0.5, "Counterexample input should satisfy property"

    def test_no_counterexample(self):
        """Test that random sampling returns UNKNOWN when no counterexample exists."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([0.4, 0.4])

        # Property: output[0] >= 0.5 is unsafe (but inputs never reach 0.5)
        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='random', n_samples=100, seed=42)

        assert result == 2, "Should return UNKNOWN (no counterexample found)"
        assert cex is None, "No counterexample should be returned"

    def test_with_dict_property(self):
        """Test that falsify handles dict property format (from vnnlib)."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)
        prop = [{'Hg': hs}]

        result, cex = falsify(model, lb, ub, prop, method='random', n_samples=100, seed=42)

        assert result == 0, "Should find counterexample"

    def test_reproducible_with_seed(self):
        """Test that falsify is reproducible with the same seed."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result1, cex1 = falsify(model, lb, ub, hs, method='random', n_samples=10, seed=123)
        result2, cex2 = falsify(model, lb, ub, hs, method='random', n_samples=10, seed=123)

        assert result1 == result2
        if cex1 is not None and cex2 is not None:
            np.testing.assert_array_equal(cex1[0], cex2[0])
            np.testing.assert_array_equal(cex1[1], cex2[1])

    def test_default_method_is_random(self):
        """Test that the default method is random sampling."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        # No method specified - should default to random
        result, cex = falsify(model, lb, ub, hs, n_samples=100, seed=42)

        assert result == 0, "Default method should find counterexample"


class TestFalsifyPGD:
    """Tests for PGD falsification (method='pgd')."""

    def test_finds_counterexample(self):
        """Test that PGD finds a counterexample when one exists."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='pgd',
                              n_restarts=5, n_steps=20, seed=42)

        assert result == 0, "Should find counterexample (SAT)"
        assert cex is not None, "Counterexample should be returned"
        inp, out = cex
        assert inp[0] >= 0.5 - 1e-6, "Counterexample input should satisfy property"

    def test_no_counterexample(self):
        """Test that PGD returns UNKNOWN when no counterexample exists."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([0.4, 0.4])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='pgd',
                              n_restarts=5, n_steps=20, seed=42)

        assert result == 2, "Should return UNKNOWN (no counterexample found)"
        assert cex is None, "No counterexample should be returned"

    def test_with_relu_model(self):
        """Test PGD with a model containing ReLU (non-linear)."""
        model = nn.Sequential(
            nn.Linear(2, 2, bias=False),
            nn.ReLU(),
            nn.Linear(2, 1, bias=False)
        )
        model[0].weight.data = torch.eye(2)
        model[2].weight.data = torch.ones(1, 2)

        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Property: output >= 1.5 is unsafe
        G = np.array([[-1.0]])
        g = np.array([-1.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='pgd',
                              n_restarts=10, n_steps=50, seed=42)

        assert result == 0, "Should find counterexample"
        assert cex is not None
        inp, out = cex
        assert out[0] >= 1.5 - 1e-6, "Output should be >= 1.5"

    def test_reproducible_with_seed(self):
        """Test that PGD is reproducible with the same seed."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result1, cex1 = falsify(model, lb, ub, hs, method='pgd',
                                n_restarts=3, n_steps=10, seed=123)
        result2, cex2 = falsify(model, lb, ub, hs, method='pgd',
                                n_restarts=3, n_steps=10, seed=123)

        assert result1 == result2
        if cex1 is not None and cex2 is not None:
            np.testing.assert_allclose(cex1[0], cex2[0], rtol=1e-5)
            np.testing.assert_allclose(cex1[1], cex2[1], rtol=1e-5)

    def test_custom_step_size(self):
        """Test PGD with custom step size."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, method='pgd',
                              n_restarts=3, n_steps=30, step_size=0.05, seed=42)

        assert result == 0, "Should find counterexample with custom step size"


class TestFalsifyCombined:
    """Tests for combined falsification (method='random+pgd')."""

    def test_finds_counterexample_via_random(self):
        """Test that combined method finds counterexample via random (first)."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        # With enough random samples, should find via random first
        result, cex = falsify(model, lb, ub, hs, method='random+pgd',
                              n_samples=100, n_restarts=5, n_steps=20, seed=42)

        assert result == 0, "Should find counterexample"
        assert cex is not None

    def test_finds_counterexample_via_pgd_when_random_fails(self):
        """Test that combined method falls back to PGD when random fails."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        # With 0 random samples, PGD should find the counterexample
        result, cex = falsify(model, lb, ub, hs, method='random+pgd',
                              n_samples=0, n_restarts=5, n_steps=20, seed=42)

        assert result == 0, "PGD should find counterexample when random has 0 samples"
        assert cex is not None


class TestFalsifyMultiGroupProperty:
    """Tests for multi-group property handling (AND logic across groups).

    VNN-LIB properties can have multiple top-level asserts that are ANDed.
    A counterexample must satisfy ALL property groups simultaneously.
    """

    def test_rejects_false_counterexample_when_second_group_not_satisfied(self):
        """Counterexample satisfying only first property group should not be reported as SAT.

        Reproduces lsnc_relu bug: falsify only checked property[0], ignoring property[1].
        """
        # Identity model: output = input
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Group 0: x1 >= 0.3 (easy to satisfy in [0,1]^2)
        hs0 = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.3]]))

        # Group 1: x2 >= 2.0 (impossible in [0,1]^2)
        hs1 = HalfSpace(np.array([[0.0, -1.0]]), np.array([[-2.0]]))

        prop = [{'Hg': hs0}, {'Hg': hs1}]

        result, cex = falsify(model, lb, ub, prop, method='random', n_samples=500, seed=42)

        # No valid counterexample exists (group 1 is infeasible)
        assert result == 2, "Should return UNKNOWN — no input satisfies both groups"
        assert cex is None

    def test_finds_counterexample_when_all_groups_satisfied(self):
        """Counterexample satisfying all property groups should be reported as SAT."""
        # Identity model: output = input
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Group 0: x1 >= 0.3
        hs0 = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.3]]))

        # Group 1: x2 >= 0.3
        hs1 = HalfSpace(np.array([[0.0, -1.0]]), np.array([[-0.3]]))

        prop = [{'Hg': hs0}, {'Hg': hs1}]

        result, cex = falsify(model, lb, ub, prop, method='random', n_samples=500, seed=42)

        assert result == 0, "Should find counterexample satisfying both groups"
        assert cex is not None
        inp, out = cex
        assert inp[0] >= 0.3 and inp[1] >= 0.3, "Counterexample must satisfy both groups"

    def test_pgd_rejects_false_counterexample_multi_group(self):
        """PGD should also respect multi-group AND logic."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Group 0: x1 >= 0.3 (easy to satisfy)
        hs0 = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.3]]))

        # Group 1: x2 >= 2.0 (impossible)
        hs1 = HalfSpace(np.array([[0.0, -1.0]]), np.array([[-2.0]]))

        prop = [{'Hg': hs0}, {'Hg': hs1}]

        result, cex = falsify(model, lb, ub, prop, method='pgd',
                              n_restarts=5, n_steps=20, seed=42)

        assert result == 2, "PGD should return UNKNOWN — no input satisfies both groups"
        assert cex is None

    def test_multi_group_with_or_within_group(self):
        """Multi-group where one group has OR of halfspaces (list)."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Group 0: x1 >= 0.8 OR x1 <= 0.2 (OR of two halfspaces)
        hs0a = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.8]]))
        hs0b = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.2]]))

        # Group 1: x2 >= 2.0 (impossible)
        hs1 = HalfSpace(np.array([[0.0, -1.0]]), np.array([[-2.0]]))

        prop = [{'Hg': [hs0a, hs0b]}, {'Hg': hs1}]

        result, cex = falsify(model, lb, ub, prop, method='random', n_samples=500, seed=42)

        assert result == 2, "Should not find counterexample — group 1 is infeasible"
        assert cex is None


class TestFalsifyValidation:
    """Tests for input validation."""

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        with pytest.raises(ValueError, match="Unknown method"):
            falsify(model, lb, ub, hs, method='invalid_method')

    def test_mismatched_bounds(self):
        """Test that mismatched bounds raise ValueError."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0, 1.0])  # Wrong shape

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        with pytest.raises(ValueError, match="same shape"):
            falsify(model, lb, ub, hs)
