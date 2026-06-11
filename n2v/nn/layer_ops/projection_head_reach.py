"""ProjectionHead reachability: Linear -> activation -> Linear.

Decomposed via the dispatcher; the helper exposes a passthrough for
opaque-module dispatching.
"""

from __future__ import annotations

from typing import List


def projection_head_passthrough(layer, input_sets: List, method: str = "exact", **kwargs):
    from n2v.nn.layer_ops.dispatcher import reach_layer
    current = input_sets
    current = reach_layer(layer.fc1, current, method, **kwargs)
    current = reach_layer(layer.act, current, method, **kwargs)
    current = reach_layer(layer.fc2, current, method, **kwargs)
    return current
