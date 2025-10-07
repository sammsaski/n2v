"""
Conv1D layer reachability operations.

1D convolution is a linear operation. We build the convolution matrix explicitly
and apply it as an affine map, reusing existing Star/Zono/Box affine_map methods.

This avoids needing a 1D analogue of ImageStar — the Conv1d weight matrix
is constructed via F.conv1d on identity columns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

from n2v.sets import Star, Zono, Box


def _build_conv1d_matrix(layer: nn.Conv1d, input_length: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Build the weight matrix and bias vector for a Conv1d layer.

    For Conv1d with C_in input channels and input length L, the flattened input
    has dimension C_in * L. We construct the (C_out * L_out, C_in * L) matrix
    by applying F.conv1d to each standard basis vector.

    Args:
        layer: PyTorch nn.Conv1d layer
        input_length: Length of the 1D input signal (L)

    Returns:
        W: Weight matrix of shape (C_out * L_out, C_in * L)
        b: Bias vector of shape (C_out * L_out,) or None
    """
    c_in = layer.in_channels
    flat_dim = c_in * input_length

    # Build identity matrix and reshape to (flat_dim, C_in, L)
    eye = torch.eye(flat_dim, dtype=torch.float32)
    eye_3d = eye.reshape(flat_dim, c_in, input_length)

    # Apply conv1d (without bias) to get the weight matrix rows
    with torch.no_grad():
        out = F.conv1d(
            eye_3d,
            layer.weight,
            bias=None,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
    # out shape: (flat_dim, C_out, L_out)
    # Flatten spatial dims and transpose to get (C_out * L_out, flat_dim)
    W = out.reshape(flat_dim, -1).T.numpy()

    # Build bias vector
    b = None
    if layer.bias is not None:
        bias = layer.bias.detach().cpu().numpy()  # (C_out,)
        l_out = out.shape[2]
        # Repeat each channel's bias L_out times to match (C_out, L_out) flattening
        b = np.repeat(bias, l_out)  # (C_out * L_out,)

    return W, b


def conv1d_star(layer: nn.Conv1d, input_stars: List[Star], **kwargs) -> List[Star]:
    """
    Exact reachability for Conv1D using Star sets.

    Builds the convolution matrix and applies as affine map.

    Args:
        layer: PyTorch nn.Conv1d layer
        input_stars: List of input Star sets

    Returns:
        List of output Star sets
    """
    output_stars = []

    for star in input_stars:
        flat_dim = star.dim
        c_in = layer.in_channels
        input_length = flat_dim // c_in

        if c_in * input_length != flat_dim:
            raise ValueError(
                f"Star dimension {flat_dim} is not divisible by "
                f"Conv1d in_channels {c_in}"
            )

        W, b = _build_conv1d_matrix(layer, input_length)

        if b is not None:
            output_star = star.affine_map(W, b.reshape(-1, 1))
        else:
            output_star = star.affine_map(W)
        output_stars.append(output_star)

    return output_stars


def conv1d_zono(layer: nn.Conv1d, input_zonos: List[Zono], **kwargs) -> List[Zono]:
    """
    Exact reachability for Conv1D using Zonotopes.

    Args:
        layer: PyTorch nn.Conv1d layer
        input_zonos: List of input Zonotopes

    Returns:
        List of output Zonotopes
    """
    output_zonos = []

    for zono in input_zonos:
        flat_dim = zono.dim
        c_in = layer.in_channels
        input_length = flat_dim // c_in

        if c_in * input_length != flat_dim:
            raise ValueError(
                f"Zono dimension {flat_dim} is not divisible by "
                f"Conv1d in_channels {c_in}"
            )

        W, b = _build_conv1d_matrix(layer, input_length)

        if b is not None:
            output_zono = zono.affine_map(W, b.reshape(-1, 1))
        else:
            output_zono = zono.affine_map(W)
        output_zonos.append(output_zono)

    return output_zonos


def conv1d_box(layer: nn.Conv1d, input_boxes: List[Box], **kwargs) -> List[Box]:
    """
    Exact reachability for Conv1D using Boxes (interval arithmetic).

    Args:
        layer: PyTorch nn.Conv1d layer
        input_boxes: List of input Boxes

    Returns:
        List of output Boxes
    """
    output_boxes = []

    for box in input_boxes:
        flat_dim = box.dim
        c_in = layer.in_channels
        input_length = flat_dim // c_in

        if c_in * input_length != flat_dim:
            raise ValueError(
                f"Box dimension {flat_dim} is not divisible by "
                f"Conv1d in_channels {c_in}"
            )

        W, b = _build_conv1d_matrix(layer, input_length)

        if b is not None:
            output_box = box.affine_map(W, b.reshape(-1, 1))
        else:
            output_box = box.affine_map(W)
        output_boxes.append(output_box)

    return output_boxes
