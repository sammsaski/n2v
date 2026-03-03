"""
BatchNorm layer reachability operations.

Handles standalone BatchNorm layers (nn.BatchNorm1d, nn.BatchNorm2d) that survive
fusion (no preceding Conv/Linear). Implements direct reachability as a channel-wise
affine transformation.

For a BatchNorm layer in eval mode:
    y = gamma / sqrt(sigma2 + eps) * x + (beta - gamma / sqrt(sigma2 + eps) * mu)
      = scale * x + shift   (per-channel)
"""

import torch.nn as nn
import numpy as np
from typing import List, Tuple, Union

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono


def _get_bn_params(layer: Union[nn.BatchNorm1d, nn.BatchNorm2d]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the affine transform parameters from a BatchNorm layer.

    Computes:
        scale = gamma / sqrt(sigma2 + eps)
        shift = beta - scale * mu

    Args:
        layer: PyTorch BatchNorm layer (must be in eval mode)

    Returns:
        Tuple of (scale, shift) as 1D numpy arrays of shape (num_features,)
    """
    gamma = layer.weight.detach().cpu().numpy()         # (num_features,)
    beta = layer.bias.detach().cpu().numpy()             # (num_features,)
    mu = layer.running_mean.detach().cpu().numpy()       # (num_features,)
    sigma2 = layer.running_var.detach().cpu().numpy()    # (num_features,)
    eps = layer.eps

    scale = gamma / np.sqrt(sigma2 + eps)
    shift = beta - scale * mu

    return scale, shift


def batchnorm_star(
    layer: Union[nn.BatchNorm1d, nn.BatchNorm2d],
    input_stars: List[Star]
) -> List[Star]:
    """
    Exact reachability for BatchNorm using Star/ImageStar sets.

    For ImageStar (V shape H, W, C, nVar+1): channel-wise scaling of V and shift of center.
    For plain Star: construct diag(scale) matrix and use star.affine_map(W, b).

    Args:
        layer: PyTorch BatchNorm layer (eval mode)
        input_stars: List of input Star or ImageStar sets

    Returns:
        List of output Star or ImageStar sets
    """
    scale, shift = _get_bn_params(layer)

    output_stars = []
    for star in input_stars:
        if isinstance(star, ImageStar):
            output_star = _batchnorm_imagestar(star, scale, shift)
        elif isinstance(star, Star):
            output_star = _batchnorm_plain_star(star, scale, shift)
        else:
            raise TypeError(f"batchnorm_star expects Star or ImageStar, got {type(star)}")
        output_stars.append(output_star)

    return output_stars


def _batchnorm_imagestar(
    star: ImageStar,
    scale: np.ndarray,
    shift: np.ndarray
) -> ImageStar:
    """
    Apply BatchNorm to an ImageStar via channel-wise affine transform.

    V is (H, W, C, nVar+1). For each channel c:
      V[:, :, c, :] *= scale[c]       (scale all columns including center)
      V[:, :, c, 0] += shift[c]       (shift center column only)

    Args:
        star: Input ImageStar
        scale: Per-channel scale factors (C,)
        shift: Per-channel shift values (C,)

    Returns:
        Output ImageStar
    """
    V = star.V.copy()  # (H, W, C, nVar+1)
    num_channels = V.shape[2]

    for c in range(num_channels):
        V[:, :, c, :] *= scale[c]
        V[:, :, c, 0] += shift[c]

    return ImageStar(
        V,
        star.C,
        star.d,
        star.predicate_lb,
        star.predicate_ub,
        star.height,
        star.width,
        star.num_channels,
    )


def _batchnorm_plain_star(
    star: Star,
    scale: np.ndarray,
    shift: np.ndarray
) -> Star:
    """
    Apply BatchNorm to a plain Star via diagonal affine map.

    Constructs W = diag(scale) and b = shift, then uses star.affine_map(W, b).

    Args:
        star: Input Star
        scale: Per-feature scale factors (dim,)
        shift: Per-feature shift values (dim,)

    Returns:
        Output Star
    """
    W = np.diag(scale)
    b = shift.reshape(-1, 1)
    return star.affine_map(W, b)


def batchnorm_zono(
    layer: Union[nn.BatchNorm1d, nn.BatchNorm2d],
    input_zonos: List[Zono]
) -> List[Zono]:
    """
    Exact reachability for BatchNorm using Zonotope/ImageZono sets.

    For ImageZono: reshape c/V to (H, W, C, ...), apply channel-wise scale/shift, reshape back.
    For plain Zono: use zono.affine_map(diag(scale), shift).

    Args:
        layer: PyTorch BatchNorm layer (eval mode)
        input_zonos: List of input Zono or ImageZono sets

    Returns:
        List of output Zono or ImageZono sets
    """
    scale, shift = _get_bn_params(layer)

    output_zonos = []
    for zono in input_zonos:
        if isinstance(zono, ImageZono):
            output_zono = _batchnorm_imagezono(zono, scale, shift)
        elif isinstance(zono, Zono):
            output_zono = _batchnorm_plain_zono(zono, scale, shift)
        else:
            raise TypeError(f"batchnorm_zono expects Zono or ImageZono, got {type(zono)}")
        output_zonos.append(output_zono)

    return output_zonos


def _batchnorm_imagezono(
    zono: ImageZono,
    scale: np.ndarray,
    shift: np.ndarray
) -> ImageZono:
    """
    Apply BatchNorm to an ImageZono via channel-wise affine transform.

    Reshapes c and V to image format (H, W, C, ...), applies scale/shift per channel,
    then flattens back.

    Args:
        zono: Input ImageZono
        scale: Per-channel scale factors (C,)
        shift: Per-channel shift values (C,)

    Returns:
        Output ImageZono
    """
    h, w, c_ch = zono.height, zono.width, zono.num_channels
    n_gen = zono.V.shape[1]

    # Reshape to image format
    c_img = zono.c.reshape(h, w, c_ch).copy()         # (H, W, C)
    V_img = zono.V.reshape(h, w, c_ch, n_gen).copy()   # (H, W, C, n_gen)

    # Apply channel-wise scale and shift
    for ch in range(c_ch):
        c_img[:, :, ch] = c_img[:, :, ch] * scale[ch] + shift[ch]
        V_img[:, :, ch, :] *= scale[ch]

    # Flatten back
    c_flat = c_img.reshape(-1, 1)
    V_flat = V_img.reshape(-1, n_gen)

    return ImageZono(c_flat, V_flat, h, w, c_ch)


def _batchnorm_plain_zono(
    zono: Zono,
    scale: np.ndarray,
    shift: np.ndarray
) -> Zono:
    """
    Apply BatchNorm to a plain Zono via diagonal affine map.

    Args:
        zono: Input Zono
        scale: Per-feature scale factors (dim,)
        shift: Per-feature shift values (dim,)

    Returns:
        Output Zono
    """
    W = np.diag(scale)
    b = shift.reshape(-1, 1)
    return zono.affine_map(W, b)


def batchnorm_box(
    layer: Union[nn.BatchNorm1d, nn.BatchNorm2d],
    input_boxes: List[Box]
) -> List[Box]:
    """
    Exact reachability for BatchNorm using Box sets.

    Applies y = scale * x + shift element-wise. If scale[i] < 0, swaps lb/ub.

    Args:
        layer: PyTorch BatchNorm layer (eval mode)
        input_boxes: List of input Box sets

    Returns:
        List of output Box sets
    """
    scale, shift = _get_bn_params(layer)

    output_boxes = []
    for box in input_boxes:
        output_box = _batchnorm_single_box(box, scale, shift)
        output_boxes.append(output_box)

    return output_boxes


def _batchnorm_single_box(
    box: Box,
    scale: np.ndarray,
    shift: np.ndarray
) -> Box:
    """
    Apply BatchNorm to a single Box.

    For each dimension i:
        new_lb[i] = scale[i] * box.lb[i] + shift[i]  if scale[i] >= 0
                   = scale[i] * box.ub[i] + shift[i]  if scale[i] < 0
        new_ub[i] = scale[i] * box.ub[i] + shift[i]  if scale[i] >= 0
                   = scale[i] * box.lb[i] + shift[i]  if scale[i] < 0

    Args:
        box: Input Box
        scale: Per-feature scale factors (dim,)
        shift: Per-feature shift values (dim,)

    Returns:
        Output Box
    """
    scale_col = scale.reshape(-1, 1)  # (dim, 1)
    shift_col = shift.reshape(-1, 1)  # (dim, 1)

    # Compute both products
    prod_lb = scale_col * box.lb  # (dim, 1)
    prod_ub = scale_col * box.ub  # (dim, 1)

    # Element-wise min/max to handle negative scales
    new_lb = np.minimum(prod_lb, prod_ub) + shift_col
    new_ub = np.maximum(prod_lb, prod_ub) + shift_col

    return Box(new_lb, new_ub)
