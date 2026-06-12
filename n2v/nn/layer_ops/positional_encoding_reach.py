"""Sinusoidal positional encoding reachability.

Adds a fixed per-position sinusoidal tensor to a sequence; a pure
constant translation, applied directly to each set representation via
:mod:`_translate` -- O(n), no dense identity matrix (Copilot review:
the previous eye-Linear surrogate was O(n^2) and OOMed at
transformer-flattened sizes).

Coverage matches nnVLA: Box, Star, Zono (+ Hex/Oct).
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops._translate import translate_set


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


def _apply(layer, input_sets: List) -> List:
    return [translate_set(s, _pe_vec(layer, s.dim)) for s in input_sets]


def positional_encoding_star(layer, input_stars: List[Star]) -> List[Star]:
    return _apply(layer, input_stars)


def positional_encoding_box(layer, input_boxes: List[Box]) -> List[Box]:
    return _apply(layer, input_boxes)


def positional_encoding_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    return _apply(layer, input_zonos)


def positional_encoding_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    return _apply(layer, input_sets)


def positional_encoding_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    return _apply(layer, input_sets)
