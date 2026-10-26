"""
Transposed Conv2D (deconvolution) layer reachability operations.

Transposed convolution is an affine transformation, so it is exact for
star/zonotope sets (cf. NNV TransposedConv2DLayer): the center column
maps through conv_transpose2d WITH bias, the generator/basis columns
WITHOUT bias.

Supports ImageStar (4D V) and ImageZono inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

from n2v.sets import Star, ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops.conv2d_reach import (
    float64_params, kernel_area, map_basis_columns)

# Profiler hook (no-op when profiling is disabled)
from n2v.profiling import region, count, OPERATION, is_enabled


def _ct2d_columns(layer: nn.ConvTranspose2d,
                  V: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply the layer's transposed convolution, without bias, to every
    column of a 4D (H, W, C, n) V in float64 (see
    :func:`~n2v.nn.layer_ops.conv2d_reach.map_basis_columns`).

    Returns the (H', W', C', n) result and the float64 bias (or None),
    which the caller adds to the center column only."""
    weight, bias = float64_params(layer)

    def conv_t(x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(
            x,
            weight,
            None,
            stride=layer.stride,
            padding=layer.padding,
            output_padding=layer.output_padding,
            groups=layer.groups,
            dilation=layer.dilation,
        )

    return (map_basis_columns(conv_t, V, kernel_area(layer)),
            bias.numpy() if bias is not None else None)


def conv_transpose2d_star(
    layer: nn.ConvTranspose2d,
    input_stars: List[Star],
    method: str = 'exact',
    **kwargs
) -> List[Star]:
    """
    Exact reachability for ConvTranspose2D using ImageStar sets.

    Args:
        layer: PyTorch nn.ConvTranspose2d layer
        input_stars: List of input sets (must be ImageStar)
        method: 'exact' or 'approx' (both exact — the map is affine)

    Returns:
        List of output ImageStars
    """
    output_stars = []
    for star in input_stars:
        if not isinstance(star, ImageStar):
            raise ValueError(
                "ConvTranspose2D requires ImageStar input. Convert Star "
                "to ImageStar with proper height/width/channels first."
            )
        with region("conv_transpose2d", OPERATION):
            out = _conv_transpose2d_imagestar(layer, star)
            # FLOPs (input-driven MAC count for transposed conv, x2 for
            # mul-add, applied to every basis column).
            if is_enabled():
                kh, kw = (layer.kernel_size
                          if isinstance(layer.kernel_size, tuple)
                          else (layer.kernel_size, layer.kernel_size))
                in_elems = star.height * star.width * star.num_channels
                n_cols = out.V.shape[-1]
                count("flops",
                      2 * in_elems * (layer.out_channels // layer.groups)
                      * kh * kw * n_cols)
        output_stars.append(out)
    return output_stars


def _conv_transpose2d_imagestar(
    layer: nn.ConvTranspose2d, input_star: ImageStar
) -> ImageStar:
    """Apply ConvTranspose2D to the 4D V tensor (H, W, C, nVar+1)."""
    V = input_star.V
    c_in = V.shape[2]

    if c_in != layer.in_channels:
        raise ValueError(
            f"Input has {c_in} channels but ConvTranspose2D expects "
            f"{layer.in_channels}"
        )

    # Center and generators together; the bias belongs to the center only
    V_out, bias = _ct2d_columns(layer, V)
    if bias is not None:
        V_out[:, :, :, 0] += bias
    h_out, w_out, c_out_channels = V_out.shape[:3]

    return ImageStar(
        V_out,
        input_star.C,
        input_star.d,
        input_star.predicate_lb,
        input_star.predicate_ub,
        h_out, w_out, c_out_channels,
    )


def conv_transpose2d_zono(
    layer: nn.ConvTranspose2d, input_zonos: List
) -> List:
    """Exact reachability for ConvTranspose2D using ImageZono sets."""
    output_zonos = []
    for zono in input_zonos:
        if not isinstance(zono, ImageZono):
            raise ValueError("ConvTranspose2D requires ImageZono input")
        output_zonos.append(_conv_transpose2d_imagezono(layer, zono))
    return output_zonos


def _conv_transpose2d_imagezono(
    layer: nn.ConvTranspose2d, input_zono: ImageZono
) -> ImageZono:
    h_in = input_zono.height
    w_in = input_zono.width
    c_in = input_zono.num_channels

    if c_in != layer.in_channels:
        raise ValueError(
            f"Input has {c_in} channels but ConvTranspose2D expects "
            f"{layer.in_channels}"
        )

    n_gen = input_zono.V.shape[1]

    c_out, bias = _ct2d_columns(
        layer, input_zono.c.reshape(h_in, w_in, c_in, 1))
    if bias is not None:
        c_out[:, :, :, 0] += bias
    h_out, w_out, c_out_channels = c_out.shape[:3]
    V_out, _ = _ct2d_columns(
        layer, input_zono.V.reshape(h_in, w_in, c_in, n_gen))

    return ImageZono(
        c_out.reshape(-1, 1),
        V_out.reshape(h_out * w_out * c_out_channels, n_gen),
        h_out, w_out, c_out_channels,
    )
