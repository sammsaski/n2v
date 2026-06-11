"""SegmentEmbedding reachability.

Adds a learned per-segment embedding to the current activation. The
segment IDs are constants from the reachability perspective, so the
operation is an affine translation routed through :mod:`linear_reach`.

Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _emb_translation(layer, dim: int, segment_ids: Optional[torch.Tensor]) -> np.ndarray:
    """Concrete embedding translation given fixed ``segment_ids``.

    ``segment_ids`` is required: the concrete forward always adds the
    learned per-segment embedding, so returning a zero translation
    when ``segment_ids is None`` would silently treat SegmentEmbedding
    as the identity (unsound for any nonzero training).
    """
    table = layer.embedding.weight.detach().cpu().numpy().astype(np.float64)
    if segment_ids is None:
        raise ValueError(
            "SegmentEmbedding reach requires explicit `segment_ids`. "
            "Treating them as zero would silently make this layer an "
            "identity and produce unsound reach for any trained segment "
            "embedding."
        )
    ids = segment_ids.detach().cpu().numpy().reshape(-1)
    embedded = table[ids]
    if embedded.size != dim:
        raise ValueError(
            f"SegmentEmbedding translation size {embedded.size} does not "
            f"match input set dim {dim}. Provide `segment_ids` whose "
            f"product with embedding_dim equals the flattened input dim."
        )
    return embedded.reshape(-1)


def _make_translation(bias: np.ndarray) -> nn.Linear:
    n = bias.size
    dummy = nn.Linear(n, n, bias=True)
    with torch.no_grad():
        dummy.weight.copy_(torch.eye(n).float())
        dummy.bias.copy_(torch.from_numpy(bias).float())
    return dummy


def segment_embedding_star(layer, input_stars: List[Star], segment_ids: Optional[torch.Tensor] = None) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        bias = _emb_translation(layer, s.dim, segment_ids)
        out.extend(linear_reach.linear_star(_make_translation(bias), [s]))
    return out


def segment_embedding_box(layer, input_boxes: List[Box], segment_ids: Optional[torch.Tensor] = None) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        bias = _emb_translation(layer, b.dim, segment_ids)
        out.extend(linear_reach.linear_box(_make_translation(bias), [b]))
    return out


def segment_embedding_zono(layer, input_zonos: List[Zono], segment_ids: Optional[torch.Tensor] = None) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        bias = _emb_translation(layer, z.dim, segment_ids)
        out.extend(linear_reach.linear_zono(_make_translation(bias), [z]))
    return out


def segment_embedding_hexatope(
    layer, input_sets: List[Hexatope], segment_ids: Optional[torch.Tensor] = None
) -> List[Hexatope]:
    out: List[Hexatope] = []
    for s in input_sets:
        bias = _emb_translation(layer, s.dim, segment_ids)
        out.extend(linear_reach.linear_hexatope(_make_translation(bias), [s]))
    return out


def segment_embedding_octatope(
    layer, input_sets: List[Octatope], segment_ids: Optional[torch.Tensor] = None
) -> List[Octatope]:
    out: List[Octatope] = []
    for s in input_sets:
        bias = _emb_translation(layer, s.dim, segment_ids)
        out.extend(linear_reach.linear_octatope(_make_translation(bias), [s]))
    return out
