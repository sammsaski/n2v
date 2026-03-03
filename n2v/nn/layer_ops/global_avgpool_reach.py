"""
GlobalAvgPool layer reachability operations.

Handles nn.AdaptiveAvgPool2d(1) and ONNX GlobalAveragePool.
This is a linear operation — mean over spatial dimensions.
No splitting or approximation needed.
"""

import numpy as np
from typing import List
from n2v.sets import Star, Zono, ImageStar, ImageZono


def global_avgpool_star(input_stars: List[Star]) -> List[Star]:
    """
    GlobalAvgPool reachability for Star sets (exact).

    Averages over spatial dimensions (H, W), keeping channels.
    Input must be ImageStar.

    Args:
        input_stars: List of ImageStar sets

    Returns:
        List of ImageStar sets with H=1, W=1
    """
    output_stars = []
    for star in input_stars:
        if not isinstance(star, ImageStar):
            raise TypeError(
                f"GlobalAvgPool requires ImageStar input, got {type(star).__name__}"
            )
        output_star = _global_avgpool_imagestar(star)
        output_stars.append(output_star)
    return output_stars


def global_avgpool_zono(input_zonos: List[Zono]) -> List[Zono]:
    """
    GlobalAvgPool reachability for Zonotope sets (exact).

    Averages over spatial dimensions. Input must be ImageZono.

    Args:
        input_zonos: List of ImageZono sets

    Returns:
        List of ImageZono sets with H=1, W=1
    """
    output_zonos = []
    for zono in input_zonos:
        if not isinstance(zono, ImageZono):
            raise TypeError(
                f"GlobalAvgPool requires ImageZono input, got {type(zono).__name__}"
            )
        output_zono = _global_avgpool_imagezono(zono)
        output_zonos.append(output_zono)
    return output_zonos


def _global_avgpool_imagestar(input_star: ImageStar) -> ImageStar:
    """
    Apply GlobalAvgPool to ImageStar.

    V is (H, W, C, nVar+1). Average over H and W dims:
    V_out[0, 0, c, k] = mean(V[:, :, c, k])

    This is exact because averaging is linear.
    """
    V = input_star.V  # (H, W, C, nVar+1)

    # Average over spatial dims (0, 1), keep (C, nVar+1)
    V_out = np.mean(V, axis=(0, 1), keepdims=True)  # (1, 1, C, nVar+1)

    return ImageStar(
        V_out,
        input_star.C,
        input_star.d,
        input_star.predicate_lb,
        input_star.predicate_ub,
        1,  # height
        1,  # width
        input_star.num_channels,
    )


def _global_avgpool_imagezono(input_zono: ImageZono) -> ImageZono:
    """
    Apply GlobalAvgPool to ImageZono.

    c is (H*W*C, 1), V is (H*W*C, n_gen).
    Reshape to (H, W, C, ...), average over H and W, flatten back.
    """
    h, w, c = input_zono.height, input_zono.width, input_zono.num_channels
    n_gen = input_zono.V.shape[1]

    # Reshape to image format
    c_img = input_zono.c.reshape(h, w, c)        # (H, W, C)
    V_img = input_zono.V.reshape(h, w, c, n_gen)  # (H, W, C, n_gen)

    # Average over spatial dims
    c_out = np.mean(c_img, axis=(0, 1), keepdims=True)    # (1, 1, C)
    V_out = np.mean(V_img, axis=(0, 1), keepdims=True)    # (1, 1, C, n_gen)

    # Flatten back
    c_flat = c_out.reshape(-1, 1)       # (C, 1)
    V_flat = V_out.reshape(-1, n_gen)   # (C, n_gen)

    return ImageZono(c_flat, V_flat, 1, 1, c)
