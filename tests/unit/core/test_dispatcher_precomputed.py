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


# --------------------------------------------------------------------------- #
# The dispatcher hands the bounds to the kernel, which reads them               #
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402



def _rows_of_neuron_zero(sets):
    """Row 0 (centre and generators) of every returned star."""
    return [np.asarray(s.V)[0] for s in sets]


@pytest.mark.parametrize('method', ['exact', 'approx'])
@pytest.mark.parametrize('layer, slope', [(nn.ReLU(), 0.0),
                                          (nn.LeakyReLU(0.125), 0.125)])
def test_handed_bounds_decide_a_neurons_phase(layer, slope, method):
    """Star x0 in [-1, 1], x1 in [1/2, 3/2]. Handed bounds that put neuron 0
    in [-1, -1/2] (a deliberate fault: its true range crosses 0) make the
    kernel treat it as stably negative, so row 0 of every returned star is
    the input row times the slope (0 for ReLU), exactly; without them the
    neuron crosses and row 0 is not that. Catches a dispatcher that drops
    ``precomputed_bounds`` or passes them under another name for a branch
    (ReLU or LeakyReLU, exact or approx), and a kernel that ignores them."""
    star = Star.from_bounds(np.array([[-1.0], [0.5]]),
                            np.array([[1.0], [1.5]]))
    given = (np.array([[-1.0], [0.5]]), np.array([[-0.5], [1.5]]))
    expected = slope * np.asarray(star.V)[0]
    with_bounds = reach_layer(layer, [star], method=method,
                              precomputed_bounds=given)
    assert with_bounds
    for row in _rows_of_neuron_zero(with_bounds):
        assert np.array_equal(row[:expected.size], expected), row
        assert not np.any(row[expected.size:]), row
    without = reach_layer(layer, [star], method=method)
    assert any(not np.array_equal(row[:expected.size], expected)
               or np.any(row[expected.size:])
               for row in _rows_of_neuron_zero(without))
