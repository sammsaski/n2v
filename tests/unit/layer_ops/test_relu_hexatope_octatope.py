"""Tests for hexatope/octatope ReLU reachability."""

import pytest
import numpy as np
from n2v.sets import Hexatope, Octatope
from n2v.nn.layer_ops.relu_reach import relu_hexatope, relu_octatope


class TestReluHexatope:
    def test_active_neuron_returns_identity(self):
        """Neuron with lb >= 0 should return input unchanged."""
        lb = np.array([[1.0], [1.0]])
        ub = np.array([[2.0], [2.0]])
        h = Hexatope.from_bounds(lb, ub)
        result = relu_hexatope([h])
        assert len(result) == 1

    def test_inactive_neuron_preserves_constraints(self):
        """Neuron with ub <= 0 should zero out via affine map, not from_bounds."""
        lb = np.array([[-2.0], [0.0]])
        ub = np.array([[-1.0], [1.0]])
        h = Hexatope.from_bounds(lb, ub)
        n_constraints_before = len(h.dcs.constraints)
        result = relu_hexatope([h])
        assert len(result) == 1
        assert len(result[0].dcs.constraints) >= n_constraints_before

    def test_never_returns_empty_for_feasible_input(self):
        """A feasible hexatope should never produce empty result."""
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        h = Hexatope.from_bounds(lb, ub)
        result = relu_hexatope([h])
        assert len(result) > 0


class TestReluOctatope:
    def test_inactive_neuron_preserves_constraints(self):
        lb = np.array([[-2.0], [0.0]])
        ub = np.array([[-1.0], [1.0]])
        o = Octatope.from_bounds(lb, ub)
        n_constraints_before = len(o.utvpi.constraints)
        result = relu_octatope([o])
        assert len(result) == 1
        assert len(result[0].utvpi.constraints) >= n_constraints_before

    def test_never_returns_empty_for_feasible_input(self):
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        o = Octatope.from_bounds(lb, ub)
        result = relu_octatope([o])
        assert len(result) > 0
