"""Pooler reachability: take CLS row, apply Linear + Tanh.

``Pooler.forward`` extracts the first token (``x[:, 0]``) before the
Linear + Tanh head. The passthrough mirrors that: it slices the first
``hidden_size``-sized chunk of the flat input set, then dispatches the
inner submodules.

Inputs whose flat dim is not a multiple of ``hidden_size`` (i.e. don't
look like a flattened ``(L, hidden)`` sequence) are rejected with a
clear error rather than silently passing the whole flat vector into a
``hidden_size``-input Linear.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar


def _slice_first_token(input_sets: List, hidden_size: int) -> List:
    """Slice the first ``hidden_size``-sized chunk from each set."""
    out: List = []
    for s in input_sets:
        is_image_star = isinstance(s, ImageStar)
        base = s.to_star() if is_image_star else s
        total = getattr(base, "dim", None) or base.lb.size
        if total % hidden_size != 0:
            raise ValueError(
                f"Pooler flat input dim {total} is not a multiple of "
                f"hidden_size={hidden_size}; cannot identify the CLS token."
            )
        if isinstance(base, Star):
            sliced = Star(base.V[:hidden_size], base.C, base.d,
                          base.predicate_lb, base.predicate_ub)
            out.append(sliced)
        elif isinstance(base, Zono):
            out.append(Zono(base.c[:hidden_size], base.V[:hidden_size]))
        elif isinstance(base, Box):
            out.append(Box(base.lb[:hidden_size], base.ub[:hidden_size]))
        else:
            raise TypeError(f"Unsupported set type for Pooler: {type(base).__name__}")
    return out


def pooler_passthrough(layer, input_sets: List, method: str = "exact", **kwargs):
    from n2v.nn.layer_ops.dispatcher import reach_layer

    hidden_size = int(layer.hidden_size)
    current = _slice_first_token(input_sets, hidden_size)
    current = reach_layer(layer.dense, current, method, **kwargs)
    current = reach_layer(layer.activation, current, method, **kwargs)
    return current
