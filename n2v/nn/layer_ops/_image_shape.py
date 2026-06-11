"""Shared helpers for preserving ImageStar / ImageZono spatial shape.

Many ports compute reachability on the flattened Star/Zono representation
and then need to restore the original ``(height, width, num_channels)``
metadata for downstream image-aware layers. This module centralises that
capture-then-restore pattern so per-layer ports don't repeat themselves.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono


def _star_shape(s: Star) -> Tuple[bool, int, int, int]:
    """Return ``(is_image, H, W, C)``; ``H/W/C`` are 0 when not ImageStar."""
    if isinstance(s, ImageStar):
        return True, s.height, s.width, s.num_channels
    return False, 0, 0, 0


def _zono_shape(z: Zono) -> Tuple[bool, int, int, int]:
    if isinstance(z, ImageZono):
        return True, z.height, z.width, z.num_channels
    return False, 0, 0, 0


def apply_box_lift_star(
    inputs: Sequence[Star],
    box_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
) -> List[Star]:
    """Apply a per-element box-lift to a list of Stars / ImageStars.

    ``box_fn`` maps an input range ``(lb, ub)`` (shape ``(dim, 1)``) to an
    output range. If the input was an ImageStar, the result is restored
    to ImageStar with the same ``(H, W, C)``.
    """
    out: List[Star] = []
    for s in inputs:
        is_image, H, W, C = _star_shape(s)
        base = s.to_star() if is_image else s
        lb, ub = base.estimate_ranges()
        out_lb, out_ub = box_fn(lb, ub)
        new_star = Star.from_bounds(out_lb, out_ub)
        if is_image:
            new_star = new_star.to_image_star(H, W, C)
        out.append(new_star)
    return out


def apply_box_lift_zono(
    inputs: Sequence[Zono],
    box_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
) -> List[Zono]:
    """Apply a per-element box-lift to a list of Zonos / ImageZonos."""
    out: List[Zono] = []
    for z in inputs:
        is_image, H, W, C = _zono_shape(z)
        base = z.to_zono() if is_image else z
        lb, ub = base.get_bounds()
        out_lb, out_ub = box_fn(lb, ub)
        new_z = Zono.from_bounds(out_lb, out_ub)
        if is_image:
            new_z = ImageZono(new_z.c, new_z.V, H, W, C)
        out.append(new_z)
    return out
