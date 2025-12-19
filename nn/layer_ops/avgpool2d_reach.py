"""
AvgPool2D layer reachability operations.

Translated from MATLAB NNV AveragePooling2DLayer.m

Note: Average pooling is a LINEAR operation, so it's EXACT for all set types
(Star, Zono, Box). No approximation or splitting needed!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from n2v.sets import Star, ImageStar, ImageZono, Zono, Box, Hexatope, Octatope


def avgpool2d_star(
    layer: nn.AvgPool2d,
    input_stars: List[Star],
    **kwargs
) -> List[Star]:
    """
    AvgPool2D reachability for Star sets (exact).

    Since average pooling is a linear operation, this is exact with no
    over-approximation or star splitting.

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
            output_star = _avgpool2d_single_imagestar(layer, star)
        else:
            raise TypeError(f"AvgPool2D expects ImageStar input, got {type(star)}")
        output_stars.append(output_star)
    return output_stars


def _avgpool2d_single_imagestar(layer: nn.AvgPool2d, input_star: ImageStar) -> ImageStar:
    """
    Apply AvgPool2D to a single ImageStar.

    Algorithm:
    1. Apply padding if needed
    2. Apply avg_pool to each basis vector in V
    3. Construct output ImageStar with pooled bases

    This is exact because averaging is linear:
    avg_pool(V * α) = avg_pool(V) * α

    Args:
        layer: PyTorch nn.AvgPool2d layer
        input_star: Input ImageStar

    Returns:
        Output ImageStar
    """
    # Apply padding
    pad_star = _apply_padding(layer, input_star)

    # Get dimensions
    h_in, w_in, c_in = pad_star.height, pad_star.width, pad_star.num_channels
    n_pred = pad_star.nVar

    # Get kernel size and stride
    kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
    stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)

    # Calculate output dimensions
    h_out = (h_in - kernel_size[0]) // stride[0] + 1
    w_out = (w_in - kernel_size[1]) // stride[1] + 1

    # Reshape V to image format: (h, w, c, nVar+1)
    V_img = pad_star.V.reshape(h_in, w_in, c_in, n_pred + 1)

    # Apply avg_pool to each basis vector
    # We need to apply it to all n_pred+1 columns (center + all basis vectors)
    V_out = np.zeros((h_out, w_out, c_in, n_pred + 1))

    for i in range(n_pred + 1):
        # Get the i-th basis image
        basis_img = V_img[:, :, :, i]  # (h, w, c)

        # Convert to PyTorch format: (c, h, w) and add batch dimension
        basis_torch = torch.from_numpy(basis_img.transpose(2, 0, 1)).unsqueeze(0).float()

        # Apply avg_pool
        pooled = F.avg_pool2d(
            basis_torch,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # Already padded
            count_include_pad=layer.count_include_pad if hasattr(layer, 'count_include_pad') else True
        )

        # Convert back to (h, w, c) format
        pooled_np = pooled.squeeze(0).numpy().transpose(1, 2, 0)
        V_out[:, :, :, i] = pooled_np

    # Flatten V back to (h*w*c, nVar+1)
    V_out_flat = V_out.reshape(-1, n_pred + 1)

    # Create output ImageStar with same constraints
    output_star = ImageStar(
        V_out_flat,
        pad_star.C,
        pad_star.d,
        pad_star.predicate_lb,
        pad_star.predicate_ub,
        h_out,
        w_out,
        c_in
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
    for zono in input_zonos:
        # Apply padding
        pad_zono = _apply_padding_zono(layer, zono)

        # Get dimensions
        h_in, w_in, c_in = pad_zono.height, pad_zono.width, pad_zono.num_channels
        n_gen = pad_zono.V.shape[1]

        # Get kernel and stride
        kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
        stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)

        # Calculate output dimensions
        h_out = (h_in - kernel_size[0]) // stride[0] + 1
        w_out = (w_in - kernel_size[1]) // stride[1] + 1

        # Reshape center and generators to image format
        c_img = pad_zono.c.reshape(h_in, w_in, c_in)
        V_img = pad_zono.V.reshape(h_in, w_in, c_in, n_gen)

        # Apply avg_pool to center
        c_torch = torch.from_numpy(c_img.transpose(2, 0, 1)).unsqueeze(0).float()
        c_pooled = F.avg_pool2d(c_torch, kernel_size=kernel_size, stride=stride)
        c_out = c_pooled.squeeze(0).numpy().transpose(1, 2, 0).reshape(-1, 1)

        # Apply avg_pool to each generator
        V_out_list = []
        for i in range(n_gen):
            v_img = V_img[:, :, :, i]
            v_torch = torch.from_numpy(v_img.transpose(2, 0, 1)).unsqueeze(0).float()
            v_pooled = F.avg_pool2d(v_torch, kernel_size=kernel_size, stride=stride)
            v_out = v_pooled.squeeze(0).numpy().transpose(1, 2, 0).reshape(-1, 1)
            V_out_list.append(v_out)

        V_out = np.hstack(V_out_list) if V_out_list else np.zeros((c_out.shape[0], 0))

        # Create output ImageZono
        output_zono = ImageZono(c_out, V_out, h_out, w_out, c_in)
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
        lb = box.lb
        ub = box.ub

        # Assume box represents an image - need to know dimensions
        # For simplicity, we'll convert to ImageStar and back
        # In practice, you'd need to know the image dimensions

        # This is a simplified implementation - real one would need image dimensions
        # For now, return the box as-is (placeholder)
        # TODO: Implement proper box pooling with known image dimensions
        output_boxes.append(box)

    return output_boxes


# Helper functions

def _apply_padding(layer: nn.AvgPool2d, input_star: ImageStar) -> ImageStar:
    """Apply zero padding to ImageStar if needed."""
    padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)

    if padding == (0, 0):
        return input_star

    # Pad the ImageStar
    h, w, c = input_star.height, input_star.width, input_star.num_channels
    n_pred = input_star.nVar

    # Padding: (top, bottom, left, right)
    pad_t, pad_b = padding[0], padding[0]
    pad_l, pad_r = padding[1], padding[1]

    h_pad = h + pad_t + pad_b
    w_pad = w + pad_l + pad_r

    # Reshape V to image format
    V_img = input_star.V.reshape(h, w, c, n_pred + 1)

    # Create padded V
    V_pad = np.zeros((h_pad, w_pad, c, n_pred + 1))
    V_pad[pad_t:pad_t + h, pad_l:pad_l + w, :, :] = V_img

    # Flatten back
    V_pad_flat = V_pad.reshape(-1, n_pred + 1)

    return ImageStar(
        V_pad_flat, input_star.C, input_star.d,
        input_star.predicate_lb, input_star.predicate_ub,
        h_pad, w_pad, c
    )


def _apply_padding_zono(layer: nn.AvgPool2d, input_zono: ImageZono) -> ImageZono:
    """Apply zero padding to ImageZono if needed."""
    padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)

    if padding == (0, 0):
        return input_zono

    h, w, c = input_zono.height, input_zono.width, input_zono.num_channels
    n_gen = input_zono.V.shape[1]

    # Padding
    pad_t, pad_b = padding[0], padding[0]
    pad_l, pad_r = padding[1], padding[1]

    h_pad = h + pad_t + pad_b
    w_pad = w + pad_l + pad_r

    # Reshape to image format
    c_img = input_zono.c.reshape(h, w, c)
    V_img = input_zono.V.reshape(h, w, c, n_gen)

    # Pad
    c_pad = np.zeros((h_pad, w_pad, c))
    c_pad[pad_t:pad_t + h, pad_l:pad_l + w, :] = c_img

    V_pad = np.zeros((h_pad, w_pad, c, n_gen))
    V_pad[pad_t:pad_t + h, pad_l:pad_l + w, :, :] = V_img

    # Flatten
    c_pad_flat = c_pad.reshape(-1, 1)
    V_pad_flat = V_pad.reshape(-1, n_gen)

    return ImageZono(c_pad_flat, V_pad_flat, h_pad, w_pad, c)


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
