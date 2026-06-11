"""GroupedQueryAttention reachability.

GQA shares K/V heads across groups of Q heads. The reachability bounds
are identical to softmax attention applied per group, so the helpers
just forward to :mod:`softmax_attention_reach`.
"""

from __future__ import annotations

from n2v.nn.layer_ops import softmax_attention_reach


def grouped_query_attention_box(q_box, k_box, v_box, l_q, d_v):
    return softmax_attention_reach.softmax_attention_box(q_box, k_box, v_box, l_q, d_v)


def grouped_query_attention_star_approx(q_star, k_star, v_star, l_q, d_v):
    return softmax_attention_reach.softmax_attention_star_approx(q_star, k_star, v_star, l_q, d_v)
