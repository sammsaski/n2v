"""
Unit tests for falsification via random sampling.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from n2v.utils import falsify
from n2v.sets import HalfSpace


class TestFalsify:
    """Tests for the falsify function."""

    def test_falsify_finds_counterexample(self):
        """Test that falsify finds a counterexample when one exists."""
        # Simple model: identity
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        # Input bounds: [0, 1] x [0, 1]
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Property: output[0] > 0.5 is unsafe (i.e., x[0] > 0.5 is unsafe)
        # HalfSpace: -x[0] <= -0.5 means x[0] >= 0.5
        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, n_samples=100, seed=42)

        assert result == 0, "Should find counterexample (SAT)"
        assert cex is not None, "Counterexample should be returned"
        inp, out = cex
        assert inp[0] >= 0.5, "Counterexample input should satisfy property"

    def test_falsify_no_counterexample(self):
        """Test that falsify returns UNKNOWN when no counterexample exists in samples."""
        # Simple model: identity
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        # Input bounds: [0, 0.4] x [0, 0.4]
        lb = np.array([0.0, 0.0])
        ub = np.array([0.4, 0.4])

        # Property: output[0] > 0.5 is unsafe
        # Since inputs are in [0, 0.4], outputs will be in [0, 0.4], never > 0.5
        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result, cex = falsify(model, lb, ub, hs, n_samples=100, seed=42)

        assert result == 2, "Should return UNKNOWN (no counterexample found)"
        assert cex is None, "No counterexample should be returned"

    def test_falsify_with_dict_property(self):
        """Test that falsify handles dict property format (from vnnlib)."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        # Property in dict format (like from load_vnnlib)
        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)
        prop = [{'Hg': hs}]

        result, cex = falsify(model, lb, ub, prop, n_samples=100, seed=42)

        assert result == 0, "Should find counterexample"

    def test_falsify_reproducible_with_seed(self):
        """Test that falsify is reproducible with the same seed."""
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        model[0].weight.data = torch.eye(2)

        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])

        G = np.array([[-1.0, 0.0]])
        g = np.array([-0.5])
        hs = HalfSpace(G, g)

        result1, cex1 = falsify(model, lb, ub, hs, n_samples=10, seed=123)
        result2, cex2 = falsify(model, lb, ub, hs, n_samples=10, seed=123)

        assert result1 == result2
        if cex1 is not None and cex2 is not None:
            np.testing.assert_array_equal(cex1[0], cex2[0])
            np.testing.assert_array_equal(cex1[1], cex2[1])
