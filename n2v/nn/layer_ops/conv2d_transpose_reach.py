"""Transposed Conv2d reachability.

A transposed convolution is the gradient operator of an ordinary
convolution and is itself affine in the input. The implementation
materialises the equivalent dense ``W^T`` matrix for the configured
kernel/stride/padding and forwards to :mod:`linear_reach`.

Because ``W^T`` can be large, this helper is intended for small feature
maps (test, verification of small ViT/U-Net decoders).

Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.nn.layer_ops import linear_reach


def _explicit_linear(layer: nn.ConvTranspose2d, input_shape: tuple) -> nn.Linear:
    """Materialise ConvTranspose2d as an ``nn.Linear`` operator.

    Constructs the equivalent dense ``W^T`` by enumerating one-hot
    inputs. The convolution's bias is added by the *original* layer to
    each one-hot response, so we subtract it before copying into
    ``dense.weight`` and then store it once on ``dense.bias`` — adding
    the bias twice (once in the weight, once again from dense.bias)
    would double-count it.
    """
    c_in, h_in, w_in = input_shape
    n_in = c_in * h_in * w_in
    with torch.no_grad():
        zero_in = torch.zeros(1, c_in, h_in, w_in)
        bias_response = layer(zero_in).reshape(-1).float()
        eye = torch.eye(n_in).reshape(n_in, c_in, h_in, w_in)
        out = layer(eye)  # (n_in, c_out, h_out, w_out) with bias added in.
        out_flat = out.reshape(n_in, -1).T  # (n_out, n_in)
        # Strip the bias contribution from each column to get pure W^T.
        weight_only = out_flat - bias_response.unsqueeze(1)
        n_out = weight_only.shape[0]
        dense = nn.Linear(n_in, n_out, bias=True)
        dense.weight.copy_(weight_only.float())
        dense.bias.copy_(bias_response)
    return dense


def _input_shape_from_layer(layer, input_dim: int, hint: tuple | None = None) -> tuple:
    """Recover ``(C, H, W)`` from a flat input dim.

    Prefers an explicit ``hint=(H, W)`` if supplied. Otherwise assumes a
    square feature map (``H == W``) and raises ``ValueError`` if the
    flat dim is incompatible with that assumption — non-square inputs
    must pass ``hint`` explicitly.
    """
    c_in = int(layer.in_channels)
    if hint is not None:
        h_in, w_in = int(hint[0]), int(hint[1])
        if c_in * h_in * w_in != input_dim:
            raise ValueError(
                f"ConvTranspose2d shape hint {(c_in, h_in, w_in)} does "
                f"not match input dim {input_dim}."
            )
        return c_in, h_in, w_in
    side_sq = input_dim // c_in
    side = int(np.sqrt(side_sq))
    if c_in * side * side != input_dim:
        raise ValueError(
            f"ConvTranspose2d input dim {input_dim} is not C * H * W with "
            f"C={c_in} and a square feature map. Pass an explicit "
            f"(H, W) hint via the **kwargs of the reach function."
        )
    return c_in, side, side


def conv2d_transpose_star(layer, input_stars: List[Star]) -> List[Star]:
    out: List[Star] = []
    for s in input_stars:
        shape = _input_shape_from_layer(layer, s.dim)
        out.extend(linear_reach.linear_star(_explicit_linear(layer, shape), [s]))
    return out


def conv2d_transpose_box(layer, input_boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        shape = _input_shape_from_layer(layer, b.dim)
        out.extend(linear_reach.linear_box(_explicit_linear(layer, shape), [b]))
    return out


def conv2d_transpose_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    out: List[Zono] = []
    for z in input_zonos:
        shape = _input_shape_from_layer(layer, z.dim)
        out.extend(linear_reach.linear_zono(_explicit_linear(layer, shape), [z]))
    return out


def conv2d_transpose_hexatope(layer, input_sets: List[Hexatope]) -> List[Hexatope]:
    out: List[Hexatope] = []
    for s in input_sets:
        shape = _input_shape_from_layer(layer, s.dim)
        out.extend(linear_reach.linear_hexatope(_explicit_linear(layer, shape), [s]))
    return out


def conv2d_transpose_octatope(layer, input_sets: List[Octatope]) -> List[Octatope]:
    out: List[Octatope] = []
    for s in input_sets:
        shape = _input_shape_from_layer(layer, s.dim)
        out.extend(linear_reach.linear_octatope(_explicit_linear(layer, shape), [s]))
    return out
