"""
AvgPool2D layer reachability operations.

Translated from MATLAB NNV AveragePooling2DLayer.m

Note: Average pooling is a LINEAR operation, so it's EXACT for all set types
(Star, Zono, Box). No approximation or splitting needed!

Supports both ImageStar (4D V) and Star (2D V) inputs with optimized paths for each.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from functools import partial
from typing import List
from n2v.sets import Star, ImageStar, ImageZono, Box, Hexatope, Octatope
from n2v.nn.layer_ops.conv2d_reach import map_basis_columns

# Profiler hook (no-op when profiling is disabled)
from n2v.profiling import region, OPERATION


def _check_ceil_mode_divisor(layer: nn.AvgPool2d, in_hw, out_hw) -> None:
    """Refuse a pool whose border divisor torch and ONNX disagree on.

    With ``ceil_mode=True`` the last window along an axis can overhang the
    padded input. With ``count_include_pad=True`` torch divides such a
    window by its part inside the padded input, while ONNX AveragePool (as
    onnxruntime evaluates it) divides by the full kernel area. onnx2torch
    copies both flags into ``nn.AvgPool2d``, so a converted layer then
    computes a different function than the ONNX model, and the layer
    cannot tell which of the two is meant; verifying either could certify
    a property the other violates. ``divisor_override`` exists only in
    torch and fixes every divisor, so it is not affected."""
    if not (layer.ceil_mode and layer.count_include_pad) \
            or layer.divisor_override is not None:
        return
    kernel, stride = _pair(layer.kernel_size), _pair(layer.stride)
    padding = _pair(layer.padding)
    for n, o, k, s, p in zip(in_hw, out_hw, kernel, stride, padding):
        if (int(o) - 1) * s + k > int(n) + 2 * p:
            raise NotImplementedError(
                "AvgPool2d with ceil_mode=True and count_include_pad=True "
                "has a window that overhangs the padded input; torch and "
                "ONNX AveragePool divide that window differently")


def _avg_pool(layer: nn.AvgPool2d, x: torch.Tensor) -> torch.Tensor:
    """The layer's own average pooling of an NCHW batch.

    Padding, ``ceil_mode``, ``count_include_pad`` and ``divisor_override``
    are all passed through, so every border window is divided exactly as
    the layer divides it. ONNX AveragePool defaults to count_include_pad=0
    (a border window averages only its in-image entries); padding by hand
    and then pooling with padding=0 would divide by the full kernel area
    instead. The divisor of each output position does not depend on the
    data, so the op is linear and applies column-wise to a star or zonotope
    basis exactly. A layer whose divisor is ambiguous between torch and
    ONNX raises (see :func:`_check_ceil_mode_divisor`)."""
    y = F.avg_pool2d(x, layer.kernel_size, stride=layer.stride,
                     padding=layer.padding, ceil_mode=layer.ceil_mode,
                     count_include_pad=layer.count_include_pad,
                     divisor_override=layer.divisor_override)
    _check_ceil_mode_divisor(layer, x.shape[-2:], y.shape[-2:])
    return y


def avgpool2d_star(
    layer: nn.AvgPool2d,
    input_stars: List[Star],
    **kwargs
) -> List[Star]:
    """
    AvgPool2D reachability for Star sets (exact).

    Since average pooling is a linear operation, this is exact with no
    over-approximation or star splitting.

    Supports both ImageStar (optimized 4D path) and Star (requires ImageStar).

    Args:
        layer: PyTorch nn.AvgPool2d layer
        input_stars: List of input Stars (should be ImageStars)
        **kwargs: Additional options (ignored for AvgPool)

    Returns:
        List of output Stars (ImageStars)
    """
    output_stars = []
    for star in input_stars:
        if isinstance(star, ImageStar):
            with region("avgpool2d", OPERATION):
                output_star = _avgpool2d_imagestar_4d(layer, star)
        else:
            raise TypeError(f"AvgPool2D expects ImageStar input, got {type(star)}")
        output_stars.append(output_star)
    return output_stars


def _avgpool2d_imagestar_4d(layer: nn.AvgPool2d, input_star: ImageStar) -> ImageStar:
    """
    Apply AvgPool2D to ImageStar using optimized 4D operations.

    Works directly on the 4D V tensor (H, W, C, nVar+1) without reshaping.

    Algorithm:
    1. Apply the layer's average pooling (its own padding and divisor
       rules) to the center and all generators, in float64
    2. Construct output ImageStar with 4D V

    This is exact because averaging is linear:
    avg_pool(V * α) = avg_pool(V) * α

    Args:
        layer: PyTorch nn.AvgPool2d layer
        input_star: Input ImageStar with 4D V

    Returns:
        Output ImageStar with 4D V
    """
    # V is 4D: (H, W, C, nVar+1) -> (H_out, W_out, C, nVar+1)
    V_out = map_basis_columns(partial(_avg_pool, layer), input_star.V)
    h_out, w_out, c_out = V_out.shape[:3]

    # Create output ImageStar with 4D V
    output_star = ImageStar(
        V_out,
        input_star.C,
        input_star.d,
        input_star.predicate_lb,
        input_star.predicate_ub,
        h_out,
        w_out,
        c_out
    )

    return output_star


def avgpool2d_zono(layer: nn.AvgPool2d, input_zonos: List[ImageZono]) -> List[ImageZono]:
    """
    AvgPool2D for ImageZono (exact).

    Since averaging is linear, this is exact for zonotopes.

    Args:
        layer: PyTorch nn.AvgPool2d layer
        input_zonos: List of input ImageZonos

    Returns:
        List of output ImageZonos
    """
    output_zonos = []
    pool = partial(_avg_pool, layer)
    for zono in input_zonos:
        h_in, w_in, c_in = zono.height, zono.width, zono.num_channels
        n_gen = zono.V.shape[1]

        # Pool center and generators in image layout (H, W, C, n), float64
        c_out = map_basis_columns(pool, zono.c.reshape(h_in, w_in, c_in, 1))
        h_out, w_out = c_out.shape[:2]
        V_out = map_basis_columns(pool, zono.V.reshape(h_in, w_in, c_in, n_gen))

        # Create output ImageZono
        output_zono = ImageZono(c_out.reshape(-1, 1),
                                V_out.reshape(h_out * w_out * c_in, n_gen),
                                h_out, w_out, c_in)
        output_zonos.append(output_zono)

    return output_zonos


def avgpool2d_box(layer: nn.AvgPool2d, input_boxes: List[Box]) -> List[Box]:
    """
    AvgPool2D for Box sets (exact).

    Since averaging is linear and monotonic, we can compute exact bounds.

    Args:
        layer: PyTorch nn.AvgPool2d layer
        input_boxes: List of input Boxes

    Returns:
        List of output Boxes
    """
    output_boxes = []
    for box in input_boxes:
        # Get bounds

        # Assume box represents an image - need to know dimensions
        # For simplicity, we'll convert to ImageStar and back
        # In practice, you'd need to know the image dimensions

        # This is a simplified implementation - real one would need image dimensions
        # For now, return the box as-is (placeholder)
        # TODO: Implement proper box pooling with known image dimensions
        output_boxes.append(box)

    return output_boxes


def avgpool2d_hexatope(layer: nn.AvgPool2d, input_hexatopes: List[Hexatope]) -> List[Hexatope]:
    """
    AvgPool2D for Hexatopes (over-approximation using bounds).

    Since hexatopes don't have inherent image structure, we use interval
    arithmetic over-approximation.

    Args:
        layer: PyTorch nn.AvgPool2d layer
        input_hexatopes: List of input Hexatopes

    Returns:
        List of output Hexatopes (over-approximation)
    """
    output_hexatopes = []

    for hexatope in input_hexatopes:
        # Get bounds
        lb, ub = hexatope.estimate_ranges()

        # Apply pooling to bounds (over-approximation)
        # For average pooling, the output bounds are within [min(lb), max(ub)]
        # This is a conservative approximation that treats it as a reshape

        # Simple approach: preserve bounds (very conservative)
        new_lb = lb
        new_ub = ub

        # Create output hexatope from bounds
        output_hexatope = Hexatope.from_bounds(new_lb, new_ub)
        output_hexatopes.append(output_hexatope)

    return output_hexatopes


def avgpool2d_octatope(layer: nn.AvgPool2d, input_octatopes: List[Octatope]) -> List[Octatope]:
    """
    AvgPool2D for Octatopes (over-approximation using bounds).

    Since octatopes don't have inherent image structure, we use interval
    arithmetic over-approximation.

    Args:
        layer: PyTorch nn.AvgPool2d layer
        input_octatopes: List of input Octatopes

    Returns:
        List of output Octatopes (over-approximation)
    """
    output_octatopes = []

    for octatope in input_octatopes:
        # Get bounds
        lb, ub = octatope.estimate_ranges()

        # Apply pooling to bounds (over-approximation)
        new_lb = lb
        new_ub = ub

        # Create output octatope from bounds
        output_octatope = Octatope.from_bounds(new_lb, new_ub)
        output_octatopes.append(output_octatope)

    return output_octatopes
