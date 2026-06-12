"""AddWithFrozenSkip reachability: ``y = x + s`` for a constant skip ``s``.

Adding a constant to a set only moves its centre, so this routes
through :mod:`_translate` -- O(n), no dense identity matrix (Copilot
review: the previous eye-Linear surrogate was O(n^2) and OOMed at
transformer-flattened sizes).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops._translate import translate_set


def _skip_vec(layer) -> np.ndarray:
    return layer.skip.detach().cpu().numpy().astype(np.float64).reshape(-1)


def _tiled_skip(layer, input_dim: int) -> np.ndarray:
    """Tile the per-feature skip across ``L = input_dim / skip_dim`` tokens.

    ``AddWithFrozenSkip.forward`` broadcasts the skip across the last
    axis, so an ``L*D``-flat input must be matched with a length-``L*D``
    translation vector. Raises ``ValueError`` if ``input_dim`` is not a
    multiple of the skip length.
    """
    skip = _skip_vec(layer)
    if input_dim % skip.size != 0:
        raise ValueError(
            f"AddWithFrozenSkip flat input dim {input_dim} is not a multiple "
            f"of the skip length {skip.size}. The concrete forward broadcasts "
            f"the skip across the last axis."
        )
    L = input_dim // skip.size
    return np.tile(skip, L)


def add_with_frozen_skip_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        out.append(translate_set(s, _tiled_skip(layer, s.dim)))
    return out


def add_with_frozen_skip_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        out.append(translate_set(z, _tiled_skip(layer, z.dim)))
    return out


def add_with_frozen_skip_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        out.append(translate_set(b, _tiled_skip(layer, b.dim)))
    return out


def add_with_frozen_skip_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    out: List[Hexatope] = []
    for s in input_sets:
        out.append(translate_set(s, _tiled_skip(layer, s.dim)))
    return out


def add_with_frozen_skip_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    out: List[Octatope] = []
    for s in input_sets:
        out.append(translate_set(s, _tiled_skip(layer, s.dim)))
    return out
