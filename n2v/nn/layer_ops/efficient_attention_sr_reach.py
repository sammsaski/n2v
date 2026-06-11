"""EfficientAttentionSR reachability — explicitly *not* implemented.

EfficientAttentionSR (Spatial-Reduction Attention) branches Q/K/V from
the same input and reduces K/V via a strided convolution before
applying softmax attention. Verifying it correctly requires the same
multi-port handling as :class:`SoftmaxAttention` plus the SR
convolution applied only to the K/V branches.

The previous passthrough composed ``sr → proj_q → proj_k → proj_v →
attention → proj_out`` *serially* on the same input set, which
verifies a fundamentally different network. To avoid silent
unsoundness this helper now raises; users should construct the model
from the underlying primitives (Conv2d + Linear + SoftmaxAttention)
so the multi-input dispatcher can handle the QKV branching correctly.
"""

from __future__ import annotations

from typing import List


def efficient_attention_sr_passthrough(
    layer, input_sets: List, method: str = "exact", **kwargs
):
    raise NotImplementedError(
        "EfficientAttentionSR reach is not implemented: serial sub-module "
        "dispatch (sr → q → k → v → attention) silently verifies a "
        "different network. Decompose the model into Conv2d + Linear + "
        "SoftmaxAttention primitives and use the multi-input dispatcher."
    )
