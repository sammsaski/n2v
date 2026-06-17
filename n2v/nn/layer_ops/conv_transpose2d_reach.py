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
from typing import List, Optional

from n2v.sets import Star, ImageStar
from n2v.sets.image_zono import ImageZono


def _ct2d(layer: nn.ConvTranspose2d, x: torch.Tensor,
          bias: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply the layer's transposed convolution to a batched NCHW tensor."""
    with torch.no_grad():
        return F.conv_transpose2d(
            x,
            layer.weight,
            bias,
            stride=layer.stride,
            padding=layer.padding,
            output_padding=layer.output_padding,
            groups=layer.groups,
            dilation=layer.dilation,
        )


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
        output_stars.append(_conv_transpose2d_imagestar(layer, star))
    return output_stars


def _conv_transpose2d_imagestar(
    layer: nn.ConvTranspose2d, input_star: ImageStar
) -> ImageStar:
    """Apply ConvTranspose2D to the 4D V tensor (H, W, C, nVar+1)."""
    V = input_star.V
    h_in, w_in, c_in, n_cols = V.shape
    n_pred = n_cols - 1

    if c_in != layer.in_channels:
        raise ValueError(
            f"Input has {c_in} channels but ConvTranspose2D expects "
            f"{layer.in_channels}"
        )

    center = V[:, :, :, 0]
    center_torch = torch.from_numpy(center).permute(2, 0, 1) \
        .unsqueeze(0).float()
    c_out = _ct2d(layer, center_torch, layer.bias)
    c_out_np = c_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    h_out, w_out, c_out_channels = c_out_np.shape

    if n_pred > 0:
        generators_torch = torch.from_numpy(V[:, :, :, 1:]) \
            .permute(3, 2, 0, 1).float()
        V_conv = _ct2d(layer, generators_torch, None)
        generators_out = V_conv.permute(2, 3, 1, 0).cpu().numpy()
    else:
        generators_out = np.zeros((h_out, w_out, c_out_channels, 0))

    V_out = np.zeros((h_out, w_out, c_out_channels, n_cols))
    V_out[:, :, :, 0] = c_out_np
    if n_pred > 0:
        V_out[:, :, :, 1:] = generators_out

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
    c_img = input_zono.c.reshape(h_in, w_in, c_in)
    V_img = input_zono.V.reshape(h_in, w_in, c_in, n_gen)

    c_torch = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).float()
    c_out = _ct2d(layer, c_torch, layer.bias)
    c_out = c_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    h_out, w_out, c_out_channels = c_out.shape

    if n_gen > 0:
        V_torch = torch.from_numpy(V_img).permute(3, 2, 0, 1).float()
        V_conv = _ct2d(layer, V_torch, None)
        V_out = V_conv.permute(2, 3, 1, 0).cpu().numpy()
    else:
        V_out = np.zeros((h_out, w_out, c_out_channels, 0))

    return ImageZono(
        c_out.reshape(-1, 1),
        V_out.reshape(h_out * w_out * c_out_channels, n_gen),
        h_out, w_out, c_out_channels,
    )
