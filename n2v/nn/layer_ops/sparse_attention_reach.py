"""SparseAttention reachability.

SparseAttention zeroes positions outside a sparsity pattern then
performs softmax attention. The sound box reach is identical to the
dense softmax attention reach (zeroing only tightens the post-softmax
distribution).

Coverage matches nnVLA: Box, Star (box-lifted).
"""

from __future__ import annotations

from typing import List

from n2v.sets import Box, Star
from n2v.nn.layer_ops import softmax_attention_reach


def sparse_attention_box(q_box, k_box, v_box, l_q, d_v):
    return softmax_attention_reach.softmax_attention_box(q_box, k_box, v_box, l_q, d_v)


def sparse_attention_star_approx(q_star, k_star, v_star, l_q, d_v):
    return softmax_attention_reach.softmax_attention_star_approx(q_star, k_star, v_star, l_q, d_v)
