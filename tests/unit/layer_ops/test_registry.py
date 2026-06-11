"""Tests for the declarative layer-reachability registry.

Verifies that a new layer can register a Star/Box reach handler via
``@register(LayerCls, SetCls)`` and the dispatcher will call it when no
``isinstance`` branch matches.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

from n2v.sets import Box, Star
from n2v.nn.layer_ops import register, lookup, reach_layer
from n2v.nn.layer_ops.registry import clear_registry, registered_pairs


class _Dummy(nn.Module):
    """A layer class the dispatcher's isinstance chains do NOT recognise."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1.0


@pytest.fixture
def cleared_registry():
    """Ensure each test sees an empty registry, then restore the state."""
    clear_registry()
    yield
    clear_registry()


def test_register_and_lookup_box(cleared_registry):
    @register(_Dummy, Box)
    def _dummy_box(layer, input_sets, method="exact", **kwargs):
        return [Box(b.lb + 1.0, b.ub + 1.0) for b in input_sets]

    pairs = list(registered_pairs())
    assert (_Dummy, Box) in pairs

    handler = lookup(_Dummy(), Box)
    assert handler is _dummy_box


def test_register_and_lookup_returns_none_for_unregistered(cleared_registry):
    """If a layer hasn't registered for a set type, lookup returns None."""
    @register(_Dummy, Box)
    def _dummy_box(layer, input_sets, method="exact", **kwargs):
        return input_sets

    # Star isn't registered for _Dummy.
    assert lookup(_Dummy(), Star) is None


def test_dispatcher_falls_back_to_registry(cleared_registry):
    """End-to-end: a layer with no isinstance branch routes via the registry."""
    @register(_Dummy, Box)
    def _dummy_box(layer, input_sets, method="exact", **kwargs):
        return [Box(b.lb + 7.0, b.ub + 7.0) for b in input_sets]

    inp = Box(np.zeros((2, 1)), np.ones((2, 1)))
    out = reach_layer(_Dummy(), [inp])
    assert len(out) == 1
    np.testing.assert_allclose(out[0].lb.flatten(), [7.0, 7.0])
    np.testing.assert_allclose(out[0].ub.flatten(), [8.0, 8.0])


def test_dispatcher_raises_when_neither_isinstance_nor_registry_match(cleared_registry):
    """No fallback means the dispatcher raises NotImplementedError."""
    inp = Box(np.zeros((2, 1)), np.ones((2, 1)))
    with pytest.raises(NotImplementedError, match="Box reachability"):
        reach_layer(_Dummy(), [inp])


def test_registry_precedence_first_registered_wins(cleared_registry):
    """First registered handler matching (layer, set) wins."""
    @register(_Dummy, Box)
    def _first(layer, input_sets, method="exact", **kwargs):
        return [Box(b.lb + 1.0, b.ub + 1.0) for b in input_sets]

    @register(_Dummy, Box)
    def _second(layer, input_sets, method="exact", **kwargs):
        return [Box(b.lb - 100.0, b.ub - 100.0) for b in input_sets]

    handler = lookup(_Dummy(), Box)
    assert handler is _first
