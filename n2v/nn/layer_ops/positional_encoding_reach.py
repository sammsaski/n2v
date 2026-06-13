"""Sinusoidal positional encoding reachability.

Adds a fixed per-position sinusoidal tensor to a sequence; a pure
constant translation, applied directly to each set representation via
:mod:`_translate` -- O(n), no dense identity matrix (Copilot review:
the previous eye-Linear surrogate was O(n^2) and OOMed at
transformer-flattened sizes).

Coverage: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops._translate import translate_set


def _pe_vec(layer, dim: int) -> np.ndarray:
    """Slice the layer's precomputed positional table to the input dim.

    The flat input must be ``L * layer.dim`` for some sequence length
    ``L`` (the concrete forward adds the per-position encoding across
    the model-dim axis). Two failure modes are rejected loudly, matching
    what the concrete forward would do:

    * ``dim`` not a multiple of ``layer.dim`` -- the input cannot be a
      ``(L, layer.dim)`` sequence, so the per-token broadcast (and this
      reach's row slice) is undefined.
    * ``L > max_len`` (``dim`` exceeds the precomputed table) -- there is
      no encoding for positions beyond ``max_len``.
    """
    model_dim = int(layer.dim)
    if dim % model_dim != 0:
        raise ValueError(
            f"PositionalEncoding reach: input set dim {dim} is not a "
            f"multiple of the model dim {model_dim}; the input cannot be "
            f"an (L, {model_dim}) sequence and the concrete forward would "
            f"raise a shape mismatch."
        )
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


