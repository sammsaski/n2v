"""
Tanh activation reachability operations.

Approximate reachability for Tanh using linear relaxation.
Translated from MATLAB NNV TanSig.m (multiStepTanSig_NoSplit).

Reuses the _s_curve_single_star_approx helper from sigmoid_reach.py,
parameterized with tanh function, derivative, and inflection point values.
"""

import numpy as np
from typing import List
from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.nn.layer_ops.sigmoid_reach import (
    _s_curve_single_star_approx,
    _s_curve_single_zono,
    _preserve_imagestar_type,
)


def _tanh(x: np.ndarray) -> np.ndarray:
    """Numpy tanh."""
    return np.tanh(x)

def _tanh_deriv(x: np.ndarray) -> np.ndarray:
    """Tanh derivative: 1 - tanh^2(x)."""
    t = np.tanh(x)
    return 1.0 - t * t


def tanh_star_approx(
    input_stars: List[Star],
    lp_solver: str = 'default',
) -> List[Star]:
    """
    Approximate Tanh reachability for Star sets.

    Uses same NNV algorithm as Sigmoid but with tanh function.
    f(0) = 0, f'(0) = 1.

    Args:
        input_stars: List of input Stars
        lp_solver: LP solver

    Returns:
        List of output Stars
    """
    output_stars = []
    for star in input_stars:
        star_2d = star.to_star() if isinstance(star, ImageStar) else star
        result = _s_curve_single_star_approx(
            star_2d, _tanh, _tanh_deriv, f0=0.0, df0=1.0, lp_solver=lp_solver
        )
        if result is not None:
            result = _preserve_imagestar_type(star, result)
            output_stars.append(result)
    return output_stars


def tanh_zono_approx(input_zonos: List[Zono]) -> List[Zono]:
    """Approximate Tanh for Zonotopes, preserving ImageZono type."""
    from n2v.sets.image_zono import ImageZono

    output = []
    for z in input_zonos:
        result = _s_curve_single_zono(z, _tanh)
        if isinstance(z, ImageZono) and not isinstance(result, ImageZono):
            result = ImageZono(result.c, result.V, z.height, z.width, z.num_channels)
        output.append(result)
    return output


def tanh_box(input_boxes: List) -> List:
    """Tanh for Boxes. Monotone, so just apply to bounds."""
    from n2v.sets import Box
    return [Box(_tanh(box.lb), _tanh(box.ub)) for box in input_boxes]
