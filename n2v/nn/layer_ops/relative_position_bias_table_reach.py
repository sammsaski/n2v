"""RelativePositionBiasTable (Swin) reachability.

The wrapper's forward is ``attn_logits + bias`` where the bias is a
constant ``(n_heads, W*W, W*W)`` tensor derived from the learnable
table (Copilot review: the wrapper now takes the logits tensor so the
fx node carries an input set and the module participates in
end-to-end reachability). The reach is therefore an EXACT constant
translation via :mod:`_translate` -- replacing the previous loose
``[min(table), max(table)]`` envelope, which discarded the input set
entirely.

Coverage matches nnVLA: Box, Star, Zono (+ Hex/Oct).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops._translate import translate_set


def _bias_vec(layer, input_dim: int) -> np.ndarray:
    """Flatten the layer's additive bias to match the flat logits set.

    The input set must be the flattened ``(n_heads, W*W, W*W)`` logits;
    any other dim is rejected loudly -- a zero or envelope fallback
    would silently verify a different function.
    """
    with torch.no_grad():
        bias = layer.bias().detach().cpu().numpy().astype(np.float64)
    if bias.size != input_dim:
        raise ValueError(
            f"RelativePositionBiasTable reach: input set dim {input_dim} "
            f"does not match the bias size {bias.size} "
            f"(= n_heads * W^2 * W^2 for window_size={layer.window_size}, "
            f"n_heads={layer.n_heads}). The input set must be the "
            f"flattened attention logits."
        )
    return bias.reshape(-1)


def _apply(layer, input_sets: List) -> List:
    return [translate_set(s, _bias_vec(layer, s.dim)) for s in input_sets]


def relative_position_bias_table_box(layer, input_boxes: List[Box]) -> List[Box]:
    return _apply(layer, input_boxes)


def relative_position_bias_table_star(layer, input_stars: List[Star]) -> List[Star]:
    return _apply(layer, input_stars)


def relative_position_bias_table_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    return _apply(layer, input_zonos)


def relative_position_bias_table_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    return _apply(layer, input_sets)


def relative_position_bias_table_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    return _apply(layer, input_sets)
