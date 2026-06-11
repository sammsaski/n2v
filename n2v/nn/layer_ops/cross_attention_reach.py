"""CrossAttention reachability.

Cross-attention is softmax attention with K, V coming from a different
input stream than Q. The reachability functions accept three set lists
(one per Q/K/V port) and delegate to :mod:`softmax_attention_reach`.
"""

from __future__ import annotations

from typing import List

from n2v.sets import Box, Star
from n2v.nn.layer_ops import softmax_attention_reach


def cross_attention_box(q_box: List[Box], k_box: List[Box], v_box: List[Box], l_q: int, d_v: int):
    return softmax_attention_reach.softmax_attention_box(q_box, k_box, v_box, l_q, d_v)


def cross_attention_star_approx(q_star: List[Star], k_star: List[Star], v_star: List[Star], l_q: int, d_v: int):
    return softmax_attention_reach.softmax_attention_star_approx(q_star, k_star, v_star, l_q, d_v)
