"""Tests for dispatcher forwarding of precomputed_bounds."""

import numpy as np
import torch.nn as nn
from n2v.sets import Star
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestDispatcherPrecomputedBounds:

    def test_relu_receives_precomputed_bounds(self):
        """ReLU via dispatcher should accept precomputed_bounds kwarg."""
        lb = np.array([[-1.0], [0.5]])
        ub = np.array([[1.0], [1.5]])
        star = Star.from_bounds(lb, ub)
        layer = nn.ReLU()

        # Should not raise
        result = reach_layer(
            layer, [star], method='exact',
            precomputed_bounds=(lb, ub),
        )
        assert len(result) >= 1

    def test_relu_approx_receives_precomputed_bounds(self):
        """ReLU approx via dispatcher should accept precomputed_bounds kwarg."""
        lb = np.array([[-1.0], [0.5]])
        ub = np.array([[1.0], [1.5]])
        star = Star.from_bounds(lb, ub)
        layer = nn.ReLU()

        result = reach_layer(
            layer, [star], method='approx',
            precomputed_bounds=(lb, ub),
        )
        assert len(result) >= 1

    def test_leakyrelu_receives_precomputed_bounds(self):
        """LeakyReLU via dispatcher should accept precomputed_bounds kwarg."""
        lb = np.array([[-1.0], [0.5]])
        ub = np.array([[1.0], [1.5]])
        star = Star.from_bounds(lb, ub)
        layer = nn.LeakyReLU(0.1)

        result = reach_layer(
            layer, [star], method='exact',
            precomputed_bounds=(lb, ub),
        )
        assert len(result) >= 1

    def test_linear_ignores_precomputed_bounds(self):
        """Linear layer should work fine with precomputed_bounds in kwargs (ignored)."""
        lb = np.array([[-1.0], [0.5]])
        ub = np.array([[1.0], [1.5]])
        star = Star.from_bounds(lb, ub)
        layer = nn.Linear(2, 3)
        layer.eval()

        # Should not raise -- precomputed_bounds is just ignored for Linear
        result = reach_layer(
            layer, [star], method='exact',
            precomputed_bounds=(lb, ub),
        )
        assert len(result) >= 1
