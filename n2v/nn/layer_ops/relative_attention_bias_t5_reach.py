"""RelativeAttentionBiasT5 reachability.

The wrapper's forward is ``attn_logits + bias`` where the bias is a
constant ``(1, n_heads, q_len, k_len)`` tensor computed from the
learnable bucket embedding (Copilot review: the wrapper now takes the
logits tensor so the fx node carries an input set and the module
participates in end-to-end reachability). The reach is therefore an
EXACT constant translation via :mod:`_translate` -- replacing the
previous loose ``[min(table), max(table)]`` envelope, which discarded
the input set entirely.

The flat input set must be the flattened ``(n_heads, L, L)`` logits of
a self-attention block (square ``q_len == k_len``); the sequence
length is recovered from the set dim and anything non-square is
rejected loudly.

Coverage matches nnVLA: Box, Star, Zono (+ Hex/Oct).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops._translate import translate_set


def _bias_vec(layer, input_dim: int) -> np.ndarray:
    """Flatten the layer's additive bias to match the flat logits set."""
    n_heads = int(layer.n_heads)
    if input_dim % n_heads != 0:
        raise ValueError(
            f"RelativeAttentionBiasT5 reach: input set dim {input_dim} "
            f"is not divisible by n_heads={n_heads}; expected the "
            f"flattened (n_heads, L, L) attention logits."
        )
    per_head = input_dim // n_heads
    l = int(np.sqrt(per_head))
    if l * l != per_head:
        raise ValueError(
            f"RelativeAttentionBiasT5 reach: per-head dim {per_head} is "
            f"not a square L*L; expected the flattened (n_heads, L, L) "
            f"self-attention logits. Cross-shaped (q_len != k_len) "
            f"logits are not supported via flat sets."
        )
    with torch.no_grad():
        bias = layer.bias(l, l).detach().cpu().numpy().astype(np.float64)
    return bias.reshape(-1)


def _apply(layer, input_sets: List) -> List:
    return [translate_set(s, _bias_vec(layer, s.dim)) for s in input_sets]


def relative_attention_bias_t5_box(layer, input_boxes: List[Box]) -> List[Box]:
    return _apply(layer, input_boxes)


def relative_attention_bias_t5_star(layer, input_stars: List[Star]) -> List[Star]:
    return _apply(layer, input_stars)


def relative_attention_bias_t5_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    return _apply(layer, input_zonos)


def relative_attention_bias_t5_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    return _apply(layer, input_sets)


def relative_attention_bias_t5_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    return _apply(layer, input_sets)
