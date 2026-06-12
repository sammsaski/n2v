"""SegmentEmbedding reachability.

Adds a learned per-segment embedding to the current activation. The
segment IDs are constants from the reachability perspective, so the
operation is a pure constant translation, applied directly to each set
representation via :mod:`_translate` -- O(n), no dense identity matrix
(Copilot review: the previous eye-Linear surrogate was O(n^2) and
OOMed at transformer-flattened sizes).

Coverage matches nnVLA: Box, Star, Zono (+ Hex/Oct).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops._translate import translate_set


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


def _apply(layer, input_sets: List, segment_ids: Optional[torch.Tensor]) -> List:
    return [
        translate_set(s, _emb_translation(layer, s.dim, segment_ids))
        for s in input_sets
    ]


def segment_embedding_star(layer, input_stars: List[Star], segment_ids: Optional[torch.Tensor] = None) -> List[Star]:
    return _apply(layer, input_stars, segment_ids)


def segment_embedding_box(layer, input_boxes: List[Box], segment_ids: Optional[torch.Tensor] = None) -> List[Box]:
    return _apply(layer, input_boxes, segment_ids)


def segment_embedding_zono(layer, input_zonos: List[Zono], segment_ids: Optional[torch.Tensor] = None) -> List[Zono]:
    return _apply(layer, input_zonos, segment_ids)


def segment_embedding_hexatope(
    layer, input_sets: List[Hexatope], segment_ids: Optional[torch.Tensor] = None
) -> List[Hexatope]:
    return _apply(layer, input_sets, segment_ids)


def segment_embedding_octatope(
    layer, input_sets: List[Octatope], segment_ids: Optional[torch.Tensor] = None
) -> List[Octatope]:
    return _apply(layer, input_sets, segment_ids)
