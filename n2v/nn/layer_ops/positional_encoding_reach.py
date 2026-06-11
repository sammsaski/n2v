"""Sinusoidal positional encoding reachability.

Adds a fixed per-position sinusoidal tensor to a sequence; pure
constant-bias affine map, routed through :mod:`linear_reach`.

Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _pe_vec(layer, dim: int) -> np.ndarray:
    """Slice the layer's precomputed positional table to the input dim.

    Rejects inputs longer than ``max_len`` because the concrete forward
    cannot extend beyond the precomputed table either — silently
    padding with zeros would hide a runtime error in the model.
    """
    pe = layer.pe.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if pe.size < dim:
        raise ValueError(
            f"PositionalEncoding flat-vector length is {pe.size} but the "
            f"input set dim is {dim}. The concrete forward would fail on "
            f"this input (positions beyond max_len={layer.max_len} have "
            f"no encoding)."
        )
    return pe[:dim]


def _make_translation(bias: np.ndarray) -> nn.Linear:
    n = bias.size
    dummy = nn.Linear(n, n, bias=True)
    with torch.no_grad():
        dummy.weight.copy_(torch.eye(n).float())
        dummy.bias.copy_(torch.from_numpy(bias).float())
    return dummy


def positional_encoding_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        bias = _pe_vec(layer, s.dim)
        out.extend(linear_reach.linear_star(_make_translation(bias), [s]))
    return out


def positional_encoding_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        bias = _pe_vec(layer, b.dim)
        out.extend(linear_reach.linear_box(_make_translation(bias), [b]))
    return out


def positional_encoding_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        bias = _pe_vec(layer, z.dim)
        out.extend(linear_reach.linear_zono(_make_translation(bias), [z]))
    return out


def positional_encoding_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    out: List[Hexatope] = []
    for s in input_sets:
        bias = _pe_vec(layer, s.dim)
        out.extend(linear_reach.linear_hexatope(_make_translation(bias), [s]))
    return out


def positional_encoding_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    out: List[Octatope] = []
    for s in input_sets:
        bias = _pe_vec(layer, s.dim)
        out.extend(linear_reach.linear_octatope(_make_translation(bias), [s]))
    return out
