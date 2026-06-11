"""CausalMask reachability.

Adds a constant lower-triangular mask to attention logits. Affine
addition of a fixed matrix routes through the standard
translation-via-linear pattern.

Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _make_translation(mask_vec: np.ndarray) -> nn.Linear:
    n = mask_vec.size
    dummy = nn.Linear(n, n, bias=True)
    with torch.no_grad():
        dummy.weight.copy_(torch.eye(n).float())
        dummy.bias.copy_(torch.from_numpy(mask_vec).float())
    return dummy


def _mask_vec(layer, input_dim: int) -> np.ndarray:
    """Flatten the layer's ``(L, L)`` mask into a length-``L*L`` vector.

    The reach surrogate is an affine translation by the mask, so the
    input set must be the flattened ``(L, L)`` attention logits. Non-
    square ``input_dim`` is rejected with a clear error rather than
    silently treated as no-mask (which would be unsound, since the
    concrete forward adds large negative values to upper-triangle
    entries).
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


def causal_mask_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        mv = _mask_vec(layer, b.dim).reshape(-1, 1)
        out.append(Box(b.lb + mv, b.ub + mv))
    return out


def causal_mask_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        mv = _mask_vec(layer, s.dim)
        out.extend(linear_reach.linear_star(_make_translation(mv), [s]))
    return out


def causal_mask_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        mv = _mask_vec(layer, z.dim)
        out.extend(linear_reach.linear_zono(_make_translation(mv), [z]))
    return out


def causal_mask_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    out: List[Hexatope] = []
    for s in input_sets:
        mv = _mask_vec(layer, s.dim)
        out.extend(linear_reach.linear_hexatope(_make_translation(mv), [s]))
    return out


def causal_mask_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    out: List[Octatope] = []
    for s in input_sets:
        mv = _mask_vec(layer, s.dim)
        out.extend(linear_reach.linear_octatope(_make_translation(mv), [s]))
    return out
