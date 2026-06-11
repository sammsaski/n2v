"""Embedding lookup reachability.

For reachability the embedding *input* carries integer token indices
and the *output* is the post-lookup dense vector. The input "set" is
treated as a length-``n_tokens`` collection of indices (one per
dimension), so the output has length ``n_tokens * embedding_dim`` and
each token's embedding lies in the per-column bounds of the embedding
table.

Index-space inputs are intentionally not represented as set-valued
(see module docstring of the wrapper); this reach function only
bounds the post-lookup region of the model.

Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from n2v.sets import Box, Star, Zono


def _table(layer) -> np.ndarray:
    return layer.weight.detach().cpu().numpy().astype(np.float64)


def _bounds_from_table(layer) -> Tuple[np.ndarray, np.ndarray]:
    w = _table(layer)
    return w.min(axis=0).reshape(-1, 1), w.max(axis=0).reshape(-1, 1)


def embedding_box(layer, input_boxes: List[Box]) -> List[Box]:
    lb, ub = _bounds_from_table(layer)
    out: List[Box] = []
    for b in input_boxes:
        # The input set is in token-index space — each dim is one token.
        n_tokens = b.dim
        out.append(Box(np.tile(lb, (n_tokens, 1)), np.tile(ub, (n_tokens, 1))))
    return out


def embedding_star(layer, input_stars: List[Star]) -> List[Star]:
    """Output dim = ``n_tokens * embed_dim``.

    ImageStar shape is intentionally not preserved: the input is in
    token-space, the output is in embedding-space, and dimensionality
    differs.
    """
    lb, ub = _bounds_from_table(layer)
    out: List[Star] = []
    for s in input_stars:
        n_tokens = max(1, s.dim)
        out.append(
            Star.from_bounds(np.tile(lb, (n_tokens, 1)), np.tile(ub, (n_tokens, 1)))
        )
    return out


def embedding_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    lb, ub = _bounds_from_table(layer)
    out: List[Zono] = []
    for z in input_zonos:
        n_tokens = max(1, z.dim)
        out.append(
            Zono.from_bounds(np.tile(lb, (n_tokens, 1)), np.tile(ub, (n_tokens, 1)))
        )
    return out
