"""MultiQueryAttention reachability.

MQA shares a single K/V across all Q heads. Reachability identical to
softmax attention; forwards to :mod:`softmax_attention_reach`.
"""

from __future__ import annotations

from n2v.nn.layer_ops import softmax_attention_reach


def multi_query_attention_box(q_box, k_box, v_box, l_q, d_v):
    return softmax_attention_reach.softmax_attention_box(q_box, k_box, v_box, l_q, d_v)


def multi_query_attention_star_approx(q_star, k_star, v_star, l_q, d_v):
    return softmax_attention_reach.softmax_attention_star_approx(q_star, k_star, v_star, l_q, d_v)
