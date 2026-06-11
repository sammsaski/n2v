"""Rotary Positional Embedding (RoPE) reachability.

For a fixed sequence position the rotation is an affine map per token:
``x[i] = x[i] * cos + rotate(x[i]) * sin``. The implementation builds a
block-diagonal rotation matrix and routes through :mod:`linear_reach`.

Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _rotation_matrix(layer, dim: int) -> np.ndarray:
    """Construct the linear operator that RoPE applies to a (L*D) vector.

    Raises ``ValueError`` if the flat input dim is not a multiple of
    ``layer.dim``, or if the implied sequence length exceeds the
    precomputed ``max_len``. Silently collapsing extra positions to
    zero would be unsound — the concrete forward would either index
    out of range or use unintended encodings.
    """
    cos = layer.cos.detach().cpu().numpy().astype(np.float64)
    sin = layer.sin.detach().cpu().numpy().astype(np.float64)
    d = layer.dim
    l_max = cos.shape[0]
    if dim % d != 0:
        raise ValueError(
            f"RoPE flat input dim {dim} is not a multiple of layer.dim={d}."
        )
    L = dim // d
    if L > l_max:
        raise ValueError(
            f"RoPE sequence length {L} exceeds layer.max_len={l_max}. The "
            f"concrete forward has no encodings for positions beyond max_len."
        )
    half = d // 2

    R = np.zeros((dim, dim), dtype=np.float64)
    for pos in range(L):
        c = cos[pos, :d]
        s = sin[pos, :d]
        start = pos * d
        for i in range(half):
            j = i + half
            ci = c[i]
            si = s[i]
            R[start + i, start + i] = ci
            R[start + i, start + j] = -si
            R[start + j, start + i] = si
            R[start + j, start + j] = ci
    return R


def _make_rotation_linear(R: np.ndarray) -> nn.Linear:
    n = R.shape[0]
    dummy = nn.Linear(n, n, bias=False)
    with torch.no_grad():
        dummy.weight.copy_(torch.from_numpy(R).float())
    return dummy


def rope_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        R = _rotation_matrix(layer, s.dim)
        out.extend(linear_reach.linear_star(_make_rotation_linear(R), [s]))
    return out


def rope_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        R = _rotation_matrix(layer, b.dim)
        out.extend(linear_reach.linear_box(_make_rotation_linear(R), [b]))
    return out


def rope_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        R = _rotation_matrix(layer, z.dim)
        out.extend(linear_reach.linear_zono(_make_rotation_linear(R), [z]))
    return out


def rope_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    out: List[Hexatope] = []
    for s in input_sets:
        R = _rotation_matrix(layer, s.dim)
        out.extend(linear_reach.linear_hexatope(_make_rotation_linear(R), [s]))
    return out


def rope_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    out: List[Octatope] = []
    for s in input_sets:
        R = _rotation_matrix(layer, s.dim)
        out.extend(linear_reach.linear_octatope(_make_rotation_linear(R), [s]))
    return out
