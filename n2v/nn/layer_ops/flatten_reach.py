"""
Flatten layer reachability operations.

Converts ImageStar (4D V) to Star (2D V) with CHW ordering to match PyTorch's nn.Flatten().
Note: This is different from ImageStar.to_star() which uses HWC ordering.
"""

import torch.nn as nn
import numpy as np
from typing import List
from n2v.sets import Star, Zono, Box, ImageStar, ImageZono, Hexatope, Octatope


def flatten_star(layer: nn.Flatten, input_stars: List[Star]) -> List[Star]:
    """
    Flatten converts ImageStar to regular Star with CHW ordering.

    PyTorch's nn.Flatten expects CHW (channel-first) ordering, so we transpose
    the 4D V from (H, W, C, nVar+1) to (C, H, W, nVar+1) before flattening.

    Note: This is different from ImageStar.to_star() which uses HWC ordering.
    Use this for transitioning from convolutional to fully-connected layers.

    Args:
        layer: PyTorch nn.Flatten layer
        input_stars: List of input Stars (may be ImageStars)

    Returns:
        List of output Stars (regular Stars, no image structure)
    """
    output_stars = []

    for star in input_stars:
        if isinstance(star, ImageStar):
            # Use CHW ordering for PyTorch compatibility
            output_star = star.flatten_to_star()
        else:
            # Already a regular Star, no change needed
            output_star = star

        output_stars.append(output_star)

    return output_stars


def flatten_zono(layer: nn.Flatten, input_zonos: List[Zono]) -> List[Zono]:
    """
    Flatten for Zonotopes.

    Args:
        layer: PyTorch nn.Flatten layer
        input_zonos: List of input Zonotopes

    Returns:
        List of output Zonotopes
    """
    output_zonos = []

    for zono in input_zonos:
        if isinstance(zono, ImageZono):
            # Convert to regular Zono with CHW ordering
            output_zono = _flatten_imagezono_chw(zono)
        else:
            output_zono = zono

        output_zonos.append(output_zono)

    return output_zonos


def _flatten_imagezono_chw(zono: ImageZono) -> Zono:
    """
    Flatten ImageZono with CHW ordering for PyTorch compatibility.

    Args:
        zono: Input ImageZono

    Returns:
        Regular Zono with CHW-ordered flattening
    """
    h, w, c = zono.height, zono.width, zono.num_channels
    n_gen = zono.V.shape[1]

    # Reshape center and generators to image format
    c_img = zono.c.reshape(h, w, c)  # (H, W, C)
    V_img = zono.V.reshape(h, w, c, n_gen)  # (H, W, C, n_gen)

    # Transpose to CHW ordering
    c_chw = np.transpose(c_img, (2, 0, 1))  # (C, H, W)
    V_chw = np.transpose(V_img, (2, 0, 1, 3))  # (C, H, W, n_gen)

    # Flatten
    c_flat = c_chw.reshape(-1, 1)
    V_flat = V_chw.reshape(-1, n_gen)

    return Zono(c_flat, V_flat)


def flatten_box(layer: nn.Flatten, input_boxes: List[Box]) -> List[Box]:
    """
    Flatten for Boxes.

    Args:
        layer: PyTorch nn.Flatten layer
        input_boxes: List of input Boxes

    Returns:
        List of output Boxes
    """
    # Boxes don't have image structure, so no change
    return input_boxes


def flatten_hexatope(layer: nn.Flatten, input_hexatopes: List[Hexatope]) -> List[Hexatope]:
    """
    Flatten for Hexatopes.

    Since hexatopes are already vector-based (not image-based), flatten is essentially
    a no-op. The hexatope structure is preserved.

    Args:
        layer: PyTorch nn.Flatten layer
        input_hexatopes: List of input Hexatopes

    Returns:
        List of output Hexatopes (unchanged)
    """
    # Hexatopes don't have image structure, so no change needed
    return input_hexatopes


def flatten_octatope(layer: nn.Flatten, input_octatopes: List[Octatope]) -> List[Octatope]:
    """
    Flatten for Octatopes.

    Since octatopes are already vector-based (not image-based), flatten is essentially
    a no-op. The octatope structure is preserved.

    Args:
        layer: PyTorch nn.Flatten layer
        input_octatopes: List of input Octatopes

    Returns:
        List of output Octatopes (unchanged)
    """
    # Octatopes don't have image structure, so no change needed
    return input_octatopes
