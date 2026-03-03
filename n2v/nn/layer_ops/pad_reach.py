"""
Pad layer reachability operations.

Zero-padding is a linear operation: it adds zeros around the spatial
dimensions of the image. The V tensor (and Zono center/generators) are
padded with zeros along the spatial axes.
"""

import numpy as np
import torch.nn as nn
from typing import List

from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

# ONNX Pad type (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.pad import OnnxPad
except ImportError:
    OnnxPad = None


def _extract_padding(layer) -> tuple:
    """
    Extract (left, right, top, bottom) padding from various layer types.

    Returns:
        Tuple (left, right, top, bottom)
    """
    if isinstance(layer, (nn.ZeroPad2d, nn.ConstantPad2d)):
        p = layer.padding
        if isinstance(p, int):
            return (p, p, p, p)
        # nn.ZeroPad2d stores (left, right, top, bottom)
        return tuple(p)

    if OnnxPad is not None and isinstance(layer, OnnxPad):
        # OnnxPad stores padding differently — extract from attributes
        if hasattr(layer, 'pads'):
            pads = layer.pads
            # ONNX pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            # For 4D (N,C,H,W): [0, 0, top, left, 0, 0, bottom, right]
            if len(pads) == 8:
                top, left = int(pads[2]), int(pads[3])
                bottom, right = int(pads[6]), int(pads[7])
                return (left, right, top, bottom)
        # Fallback
        return (0, 0, 0, 0)

    raise TypeError(f"Cannot extract padding from {type(layer).__name__}")


# Tuple of supported pad types for isinstance checks
_PAD_TYPES = (nn.ZeroPad2d, nn.ConstantPad2d)
if OnnxPad is not None:
    _PAD_TYPES = _PAD_TYPES + (OnnxPad,)


def pad_star(layer: nn.Module, input_sets: List) -> List:
    """
    Zero-padding reachability for ImageStar sets.

    Pads the V tensor with zeros along spatial dimensions.
    """
    left, right, top, bottom = _extract_padding(layer)

    output_sets = []
    for s in input_sets:
        if not isinstance(s, ImageStar):
            raise TypeError(f"pad_star requires ImageStar, got {type(s).__name__}")

        # V shape: (H, W, C, nVar+1)
        V_padded = np.pad(
            s.V,
            ((top, bottom), (left, right), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0.0
        )

        new_h = s.height + top + bottom
        new_w = s.width + left + right

        output_sets.append(ImageStar(
            V_padded, s.C, s.d, s.predicate_lb, s.predicate_ub,
            new_h, new_w, s.num_channels
        ))

    return output_sets


def pad_zono(layer: nn.Module, input_sets: List) -> List:
    """
    Zero-padding reachability for ImageZono sets.

    Pads center and generators with zeros along spatial dimensions.
    """
    left, right, top, bottom = _extract_padding(layer)

    output_sets = []
    for z in input_sets:
        if not isinstance(z, ImageZono):
            raise TypeError(f"pad_zono requires ImageZono, got {type(z).__name__}")

        h, w, c = z.height, z.width, z.num_channels
        n_gen = z.V.shape[1]

        # Reshape to image format
        c_img = z.c.reshape(h, w, c)
        V_img = z.V.reshape(h, w, c, n_gen)

        # Pad spatial dims
        c_padded = np.pad(
            c_img,
            ((top, bottom), (left, right), (0, 0)),
            mode='constant', constant_values=0.0
        )
        V_padded = np.pad(
            V_img,
            ((top, bottom), (left, right), (0, 0), (0, 0)),
            mode='constant', constant_values=0.0
        )

        new_h = h + top + bottom
        new_w = w + left + right

        output_sets.append(ImageZono(
            c_padded.reshape(-1, 1),
            V_padded.reshape(-1, n_gen),
            new_h, new_w, c
        ))

    return output_sets
