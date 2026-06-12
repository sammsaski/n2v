"""CausalMask reachability.

Adds a constant lower-triangular mask to attention logits — a pure
constant translation, applied directly to each set representation via
:mod:`_translate` in O(n) (Copilot review: the previous eye-Linear
surrogate was O(n^2) in the flattened logit count ``n = L*L``, which
explodes quadratically in sequence length).

Coverage matches nnVLA: Box, Star, Zono (+ Hex/Oct).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops._translate import translate_set


def _mask_vec(layer, input_dim: int) -> np.ndarray:
    """Flatten the layer's ``(L, L)`` mask into a length-``L*L`` vector.

    The reach is an affine translation by the mask, so the input set
    must be the flattened ``(L, L)`` attention logits. Non-square
    ``input_dim`` is rejected with a clear error rather than silently
    treated as no-mask (which would be unsound, since the concrete
    forward adds large negative values to upper-triangle entries).
    """
    full = layer.mask.detach().cpu().numpy().astype(np.float64)
    l = int(np.sqrt(input_dim))
    if l * l != input_dim:
        raise ValueError(
            f"CausalMask reach requires a square (L, L) flattened input, "
            f"got input_dim={input_dim}. A zero fallback would be unsound "
            f"since the concrete forward adds {layer.fill_value} to masked "
            f"entries."
        )
    return full[:l, :l].reshape(-1)


def _apply(layer, input_sets: List) -> List:
    return [translate_set(s, _mask_vec(layer, s.dim)) for s in input_sets]


def causal_mask_box(layer, input_boxes: List[Box]) -> List[Box]:
    return _apply(layer, input_boxes)


def causal_mask_star(layer, input_stars: List[Star]) -> List[Star]:
    return _apply(layer, input_stars)


def causal_mask_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    return _apply(layer, input_zonos)


def causal_mask_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    return _apply(layer, input_sets)


def causal_mask_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    return _apply(layer, input_sets)
