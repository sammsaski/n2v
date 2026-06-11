"""LayerScale reachability: ``y = gamma * x`` with learnable per-channel ``gamma``.

``LayerScale.forward`` broadcasts a length-``dim`` ``gamma`` across the
last axis of its input — i.e. for an ``(L, dim)`` sequence each of the
``L`` tokens is scaled by the same ``gamma``. The reach surrogate tiles
``gamma`` across ``L`` to construct a flat ``L*dim``-by-``L*dim``
diagonal linear operator, then routes through :mod:`linear_reach`.

Coverage matches nnVLA: Box, Star, Zono, Hexatope, Octatope.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _gamma_tiled(layer, input_dim: int) -> np.ndarray:
    """Return a length-``input_dim`` diagonal scale vector.

    ``layer.gamma`` has length ``dim``; we tile it across ``L = input_dim //
    dim`` tokens. Raises ``ValueError`` if ``input_dim`` is not a multiple
    of ``dim``.
    """
    gamma = layer.gamma.detach().cpu().numpy().astype(np.float64).reshape(-1)
    dim = gamma.size
    if input_dim % dim != 0:
        raise ValueError(
            f"LayerScale flat input dim {input_dim} is not a multiple of "
            f"dim={dim}. The concrete forward broadcasts gamma across the "
            f"last axis."
        )
    L = input_dim // dim
    return np.tile(gamma, L)


def _make_diag_linear(diag: np.ndarray) -> nn.Linear:
    n = diag.size
    dummy = nn.Linear(n, n, bias=False)
    with torch.no_grad():
        dummy.weight.copy_(torch.from_numpy(np.diag(diag)).float())
    return dummy


def _surrogate_for(layer, input_dim: int) -> nn.Linear:
    return _make_diag_linear(_gamma_tiled(layer, input_dim))


def layerscale_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        out.extend(linear_reach.linear_star(_surrogate_for(layer, s.dim), [s]))
    return out


def layerscale_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        out.extend(linear_reach.linear_zono(_surrogate_for(layer, z.dim), [z]))
    return out


def layerscale_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        out.extend(linear_reach.linear_box(_surrogate_for(layer, b.dim), [b]))
    return out


def layerscale_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    out: List[Hexatope] = []
    for s in input_sets:
        out.extend(linear_reach.linear_hexatope(_surrogate_for(layer, s.dim), [s]))
    return out


def layerscale_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    out: List[Octatope] = []
    for s in input_sets:
        out.extend(linear_reach.linear_octatope(_surrogate_for(layer, s.dim), [s]))
    return out
