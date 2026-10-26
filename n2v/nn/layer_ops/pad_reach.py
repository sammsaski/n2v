"""
Pad layer reachability operations.

Zero-padding is a linear operation: it adds zeros around the spatial
dimensions of the image. The V tensor (and Zono center/generators) are
padded with zeros along the spatial axes.

Only zero padding of the two spatial axes is reached. A pad with another
constant (e.g. the -inf padding onnx2torch puts in front of an
asymmetrically padded MaxPool) or another mode (reflect, replicate), a
negative pad (cropping), and a pad of the batch or channel axis raise
NotImplementedError rather than being reached as zero padding.
"""

import numpy as np
import torch.nn as nn
from typing import List, Tuple

from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

# ONNX Pad types (onnx2torch is a required dependency)
from onnx2torch.node_converters.pad import OnnxPadDynamic, OnnxPadStatic


def _require_zero_padding(layer: nn.Module, mode: str, value: float) -> None:
    """Raise unless the layer pads with the constant 0."""
    if mode != 'constant':
        raise NotImplementedError(
            f"{type(layer).__name__} with mode '{mode}' is not supported "
            f"(only zero padding)")
    if value != 0:
        raise NotImplementedError(
            f"{type(layer).__name__} with constant value {value} is not "
            f"supported (only zero padding)")


def _extract_padding(layer: nn.Module) -> Tuple[int, int, int, int]:
    """
    Extract (left, right, top, bottom) padding from various layer types.

    Returns:
        Tuple (left, right, top, bottom)

    Raises:
        NotImplementedError: if the layer does not zero-pad only the two
            spatial axes by nonnegative amounts.
    """
    if isinstance(layer, (nn.ZeroPad2d, nn.ConstantPad2d)):
        _require_zero_padding(layer, 'constant', getattr(layer, 'value', 0))
        p = layer.padding
        # nn.ZeroPad2d / nn.ConstantPad2d store (left, right, top, bottom)
        pads = (p, p, p, p) if isinstance(p, int) else tuple(p)

    elif isinstance(layer, OnnxPadStatic):
        _require_zero_padding(layer, layer.mode, layer.constant_value)
        # onnx2torch stores these pads in torch.nn.functional.pad order, last
        # axis first: (left, right, top, bottom, channel begin, channel end,
        # batch begin, batch end). The module it puts in front of an
        # asymmetrically padded conv holds only the first four.
        pads = [int(v) for v in layer.pads]
        if len(pads) % 2 or len(pads) > 8 or any(pads[4:]):
            raise NotImplementedError(
                f"OnnxPadStatic pads {list(layer.pads)} pad more than the "
                f"two spatial axes of an (N, C, H, W) tensor")
        pads = tuple((pads + [0] * 4)[:4])

    elif isinstance(layer, OnnxPadDynamic):
        # The constant value is a runtime input; the graph reach checks it
        # when it resolves the pads (n2v.nn.reach).
        _require_zero_padding(layer, layer.mode, 0)
        # ONNX order, all begins then all ends: for (N, C, H, W)
        # [N_begin, C_begin, top, left, N_end, C_end, bottom, right].
        pads = getattr(layer, 'pads', None)
        if pads is None or len(pads) != 8:
            # No resolvable pads: fail loud rather than silently apply zero
            # padding (which would produce a wrong, smaller output).
            raise NotImplementedError(
                f"Cannot extract pad amounts from {type(layer).__name__}; "
                f"the pads input was not resolved to the 8 entries of an "
                f"(N, C, H, W) pad")
        pads = [int(v) for v in pads]
        if any(pads[i] for i in (0, 1, 4, 5)):
            raise NotImplementedError(
                f"OnnxPadDynamic pads {pads} pad the batch or channel axis")
        pads = (pads[3], pads[7], pads[2], pads[6])

    else:
        raise TypeError(f"Cannot extract padding from {type(layer).__name__}")

    if any(v < 0 for v in pads):
        raise NotImplementedError(
            f"Negative padding (cropping) is not supported: {pads}")
    return tuple(int(v) for v in pads)


# Tuple of supported pad types for isinstance checks
_PAD_TYPES = (nn.ZeroPad2d, nn.ConstantPad2d, OnnxPadDynamic, OnnxPadStatic)


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
