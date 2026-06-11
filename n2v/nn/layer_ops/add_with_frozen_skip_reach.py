"""AddWithFrozenSkip reachability: ``y = x + s`` for a constant skip ``s``.

Adding a constant to a set is the same as an affine map with weight=I,
bias=skip, so this routes through :mod:`linear_reach`.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _skip_vec(layer) -> np.ndarray:
    return layer.skip.detach().cpu().numpy().astype(np.float64).reshape(-1)


def _make_translation(skip: np.ndarray) -> nn.Linear:
    n = skip.size
    dummy = nn.Linear(n, n, bias=True)
    with torch.no_grad():
        dummy.weight.copy_(torch.eye(n).float())
        dummy.bias.copy_(torch.from_numpy(skip).float())
    return dummy


def _surrogate_for(layer, input_dim: int) -> nn.Linear:
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
    return _make_translation(np.tile(skip, L))


def add_with_frozen_skip_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        out.extend(linear_reach.linear_star(_surrogate_for(layer, s.dim), [s]))
    return out


def add_with_frozen_skip_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        out.extend(linear_reach.linear_zono(_surrogate_for(layer, z.dim), [z]))
    return out


def add_with_frozen_skip_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        out.extend(linear_reach.linear_box(_surrogate_for(layer, b.dim), [b]))
    return out


def add_with_frozen_skip_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    out: List[Hexatope] = []
    for s in input_sets:
        out.extend(linear_reach.linear_hexatope(_surrogate_for(layer, s.dim), [s]))
    return out


def add_with_frozen_skip_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    out: List[Octatope] = []
    for s in input_sets:
        out.extend(linear_reach.linear_octatope(_surrogate_for(layer, s.dim), [s]))
    return out
