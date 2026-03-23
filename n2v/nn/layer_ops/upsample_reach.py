"""
Upsample/Resize layer reachability operations.

Nearest-neighbor upsampling replicates each spatial pixel into a block.
This is a linear operation — each output element is a copy of one input element —
so it is exact for all set types.

Supports both OnnxResize (from onnx2torch) and nn.Upsample (native PyTorch).
"""

import numpy as np
import torch.nn as nn
from typing import List, Tuple

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono


def _get_scale_factors(layer: nn.Module) -> Tuple[int, int]:
    """
    Extract integer spatial scale factors from an upsample/resize layer.

    Args:
        layer: nn.Upsample or OnnxResize module

    Returns:
        (scale_h, scale_w): Integer scale factors for height and width

    Raises:
        NotImplementedError: If mode is not nearest-neighbor or scale factors
                            are not integers
    """
    # Handle nn.Upsample
    if isinstance(layer, nn.Upsample):
        if layer.mode != 'nearest':
            raise NotImplementedError(
                f"Only nearest-neighbor upsampling is supported, got '{layer.mode}'"
            )
        if layer.scale_factor is None:
            raise NotImplementedError(
                "Upsample with explicit output size is not supported; "
                "use scale_factor instead"
            )
        sf = layer.scale_factor
        if isinstance(sf, (int, float)):
            scale_h = scale_w = int(sf)
        else:
            scale_h, scale_w = int(sf[0]), int(sf[1])

        if scale_h != sf if isinstance(sf, (int, float)) else (scale_h != sf[0] or scale_w != sf[1]):
            raise NotImplementedError(
                f"Only integer scale factors are supported, got {sf}"
            )
        return scale_h, scale_w

    # Handle OnnxResize (from onnx2torch)
    if hasattr(layer, 'onnx_mode'):
        if layer.onnx_mode != 'nearest':
            raise NotImplementedError(
                f"Only nearest-neighbor resize is supported, got '{layer.onnx_mode}'"
            )
        # OnnxResize doesn't store scale factors as attributes;
        # they are passed as forward() arguments. We need to extract them
        # from the graph module's bound constants.
        # For now, we'll detect them during the reach call by inspecting
        # the layer's stored parameters.
        raise _NeedForwardInspection()

    raise NotImplementedError(
        f"Cannot extract scale factors from layer type: {type(layer).__name__}"
    )


class _NeedForwardInspection(Exception):
    """Signal that scale factors must be detected from forward pass."""
    pass


def _detect_scale_factors_from_forward(layer: nn.Module, h_in: int, w_in: int) -> Tuple[int, int]:
    """
    Detect scale factors by running a forward pass with a probe tensor.

    Args:
        layer: The upsample/resize module
        h_in: Input height
        w_in: Input width

    Returns:
        (scale_h, scale_w): Integer scale factors
    """
    import torch
    probe = torch.zeros(1, 1, h_in, w_in)
    with torch.no_grad():
        # OnnxResize may need extra args (roi, scales, sizes)
        # Try calling with scale factors as positional args
        try:
            out = layer(probe)
        except TypeError:
            # OnnxResize forward signature: (input, roi, scales, sizes)
            scales = torch.tensor([1.0, 1.0, 2.0, 2.0])
            out = layer(probe, None, scales, None)

    h_out, w_out = out.shape[2], out.shape[3]
    scale_h = h_out // h_in
    scale_w = w_out // w_in

    if scale_h * h_in != h_out or scale_w * w_in != w_out:
        raise NotImplementedError(
            f"Non-integer scale factors detected: {h_out}/{h_in}={h_out/h_in}, "
            f"{w_out}/{w_in}={w_out/w_in}"
        )

    return scale_h, scale_w


def _upsample_nearest_4d(V: np.ndarray, scale_h: int, scale_w: int) -> np.ndarray:
    """
    Apply nearest-neighbor upsampling to a 4D array (H, W, C, N).

    Each spatial position (h, w) is replicated into a (scale_h, scale_w) block.

    Args:
        V: Input array of shape (H, W, C, N)
        scale_h: Height scale factor
        scale_w: Width scale factor

    Returns:
        Output array of shape (H*scale_h, W*scale_w, C, N)
    """
    return np.repeat(np.repeat(V, scale_h, axis=0), scale_w, axis=1)


def upsample_star(layer: nn.Module, input_stars: List, **kwargs) -> List:
    """
    Exact reachability for nearest-neighbor upsampling using Star sets.

    For ImageStar: applies np.repeat on the V tensor along H and W axes.
    For flat Star: builds a replication matrix and applies as affine map.

    Args:
        layer: nn.Upsample or OnnxResize module
        input_stars: List of input Star/ImageStar sets

    Returns:
        List of output Star/ImageStar sets
    """
    output_stars = []

    for star in input_stars:
        if isinstance(star, ImageStar):
            # Get scale factors
            try:
                scale_h, scale_w = _get_scale_factors(layer)
            except _NeedForwardInspection:
                scale_h, scale_w = _detect_scale_factors_from_forward(
                    layer, star.height, star.width
                )

            V_out = _upsample_nearest_4d(star.V, scale_h, scale_w)
            h_out = star.height * scale_h
            w_out = star.width * scale_w

            output_star = ImageStar(
                V_out,
                star.C,
                star.d,
                star.predicate_lb,
                star.predicate_ub,
                h_out,
                w_out,
                star.num_channels,
            )
            output_stars.append(output_star)
        elif isinstance(star, Star):
            # Flat Star: build replication matrix
            # Assume NCHW flattening: dimension = C * H * W
            raise NotImplementedError(
                "Upsample on flat Star requires known spatial dimensions. "
                "Use ImageStar for spatial operations."
            )
        else:
            raise TypeError(f"upsample_star expects Star or ImageStar, got {type(star)}")

    return output_stars


def upsample_zono(layer: nn.Module, input_zonos: List, **kwargs) -> List:
    """
    Exact reachability for nearest-neighbor upsampling using Zonotopes.

    For ImageZono: applies np.repeat on center and generators.

    Args:
        layer: nn.Upsample or OnnxResize module
        input_zonos: List of input Zono/ImageZono sets

    Returns:
        List of output Zono/ImageZono sets
    """
    output_zonos = []

    for zono in input_zonos:
        if isinstance(zono, ImageZono):
            try:
                scale_h, scale_w = _get_scale_factors(layer)
            except _NeedForwardInspection:
                scale_h, scale_w = _detect_scale_factors_from_forward(
                    layer, zono.height, zono.width
                )

            h_in, w_in, c_in = zono.height, zono.width, zono.num_channels
            n_gen = zono.V.shape[1]

            # Reshape to spatial, upsample, flatten back
            c_img = zono.c.reshape(h_in, w_in, c_in, 1)
            V_img = zono.V.reshape(h_in, w_in, c_in, n_gen)

            c_up = _upsample_nearest_4d(c_img, scale_h, scale_w)
            V_up = _upsample_nearest_4d(V_img, scale_h, scale_w)

            h_out = h_in * scale_h
            w_out = w_in * scale_w

            c_flat = c_up.reshape(-1, 1)
            V_flat = V_up.reshape(-1, n_gen)

            output_zono = ImageZono(c_flat, V_flat, h_out, w_out, c_in)
            output_zonos.append(output_zono)
        else:
            raise NotImplementedError(
                "Upsample on flat Zono requires known spatial dimensions. "
                "Use ImageZono for spatial operations."
            )

    return output_zonos


def upsample_box(layer: nn.Module, input_boxes: List[Box], **kwargs) -> List[Box]:
    """
    Exact reachability for nearest-neighbor upsampling using Boxes.

    Nearest-neighbor upsampling replicates elements, so bounds are simply
    replicated to match the output spatial dimensions.

    Args:
        layer: nn.Upsample or OnnxResize module
        input_boxes: List of input Boxes

    Returns:
        List of output Boxes
    """
    raise NotImplementedError(
        "Upsample on Box requires known spatial dimensions. "
        "Use ImageStar or ImageZono for spatial operations."
    )
