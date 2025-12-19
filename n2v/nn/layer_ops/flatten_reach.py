"""
Flatten layer reachability operations.
"""

import torch.nn as nn
from typing import List
from n2v.sets import Star, Zono, Box, ImageStar, ImageZono, Hexatope, Octatope


def flatten_star(layer: nn.Flatten, input_stars: List[Star]) -> List[Star]:
    """
    Flatten converts ImageStar to regular Star.

    Args:
        layer: PyTorch nn.Flatten layer
        input_stars: List of input Stars (may be ImageStars)

    Returns:
        List of output Stars (regular Stars, no image structure)
    """
    output_stars = []

    for star in input_stars:
        if isinstance(star, ImageStar):
            # Convert ImageStar to regular Star (already flattened internally)
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
            # Convert to regular Zono
            output_zono = zono.to_zono()
        else:
            output_zono = zono

        output_zonos.append(output_zono)

    return output_zonos


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
