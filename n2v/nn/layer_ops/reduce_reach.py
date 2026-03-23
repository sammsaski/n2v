"""
ReduceSum / ReduceMean layer reachability operations.

Handles ONNX ReduceSum and ReduceMean operations, which reduce (sum or average)
tensor values along specified axes. Both are linear operations, so reachability
is exact for all set types — no splitting or approximation needed.
"""

import numpy as np
from typing import List, Optional, Union

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono


# ---------------------------------------------------------------------------
# Helpers for extracting layer configuration
# ---------------------------------------------------------------------------

def _get_axes(layer: object) -> Optional[List[int]]:
    """
    Extract reduction axes from an ONNX reduce layer, removing batch dim.

    The layer stores ONNX axes which include the batch dimension (axis 0).
    We remove axis 0 if present and shift the remaining axes by -1.

    Returns:
        List of int — axes in the set's coordinate system (no batch dim)
    """
    if hasattr(layer, 'axes') and layer.axes is not None:
        raw = list(layer.axes)
    elif hasattr(layer, '_axes') and layer._axes is not None:
        raw = list(layer._axes)
    else:
        # If axes is None, reduce all dims (will be handled per-set)
        return None

    # Remove batch dimension (axis 0) and shift remaining
    axes = []
    for a in raw:
        if a == 0:
            continue  # skip batch dim
        axes.append(a - 1)
    return axes if axes else None


def _get_reduce_op(layer: object) -> str:
    """
    Determine whether the layer performs 'mean' or 'sum'.

    Returns:
        'mean' or 'sum'
    """
    if hasattr(layer, 'operation_type'):
        op = layer.operation_type
        if 'Mean' in op:
            return 'mean'
        if 'Sum' in op:
            return 'sum'
    # OnnxReduceSumStaticAxes is always sum
    return 'sum'


def _get_keepdims(layer: object) -> bool:
    """
    Return whether the layer keeps reduced dimensions.

    Returns:
        bool
    """
    if hasattr(layer, 'keepdims'):
        return bool(layer.keepdims)
    if hasattr(layer, '_keepdims'):
        return bool(layer._keepdims)
    return True


# ---------------------------------------------------------------------------
# Star reachability
# ---------------------------------------------------------------------------

def reduce_star(layer, input_sets: List) -> List:
    """
    ReduceSum / ReduceMean reachability for Star and ImageStar sets.

    Both operations are linear, so reachability is exact.

    Args:
        layer: ONNX reduce layer (OnnxReduceStaticAxes or OnnxReduceSumStaticAxes)
        input_sets: List of Star or ImageStar sets

    Returns:
        List of output sets
    """
    axes = _get_axes(layer)
    op = _get_reduce_op(layer)
    keepdims = _get_keepdims(layer)

    output_sets = []
    for s in input_sets:
        if isinstance(s, ImageStar):
            output_sets.append(_reduce_imagestar(s, axes, op, keepdims))
        elif isinstance(s, Star):
            output_sets.append(_reduce_flat_star(s, axes, op, keepdims))
        else:
            raise TypeError(
                f"reduce_star requires Star or ImageStar, got {type(s).__name__}"
            )
    return output_sets


def _reduce_imagestar(star: ImageStar, axes: Optional[List[int]], op: str, keepdims: bool) -> Union[ImageStar, Star]:
    """
    Apply reduce to an ImageStar.

    V is (H, W, C, nVar+1).  ONNX axes (after batch removal) are in CHW order:
    0=C, 1=H, 2=W.  Map to HWC for ImageStar: {0: 2, 1: 0, 2: 1}.
    """
    V = star.V  # (H, W, C, nVar+1)

    # Map ONNX axes (after batch removal, CHW) to HWC
    chw_to_hwc = {0: 2, 1: 0, 2: 1}

    if axes is None:
        # Reduce all spatial dims
        hwc_axes = (0, 1, 2)
    else:
        hwc_axes = tuple(chw_to_hwc.get(a, a) for a in axes)

    reduce_fn = np.mean if op == 'mean' else np.sum

    V_out = reduce_fn(V, axis=hwc_axes, keepdims=keepdims)

    if keepdims:
        h_out = V_out.shape[0]
        w_out = V_out.shape[1]
        c_out = V_out.shape[2]
        return ImageStar(
            V_out, star.C, star.d, star.predicate_lb, star.predicate_ub,
            h_out, w_out, c_out
        )
    else:
        # Reduced dims are removed — result is lower-dimensional.
        # Flatten to Star.
        n_cols = V_out.shape[-1] if V_out.ndim > 1 else 1
        V_flat = V_out.reshape(-1, n_cols)
        return Star(V_flat, star.C, star.d, star.predicate_lb, star.predicate_ub)


def _reduce_flat_star(star: Star, axes: Optional[List[int]], op: str, keepdims: bool) -> Star:
    """
    Apply reduce to a flat Star.

    V is (dim, nVar+1).  For a flat (1D) set, axes after batch removal
    is [0] meaning reduce along the single feature dimension.
    """
    dim = star.dim
    n_cols = star.V.shape[1]

    if axes is None or axes == [0] or (len(axes) == 1 and axes[0] == 0):
        # Reduce the entire feature dimension -> scalar output
        if op == 'mean':
            W = np.ones((1, dim), dtype=np.float64) / dim
        else:
            W = np.ones((1, dim), dtype=np.float64)
        return star.affine_map(W)
    else:
        raise NotImplementedError(
            f"reduce_star: axes={axes} not supported for flat Star (dim={dim})"
        )


# ---------------------------------------------------------------------------
# Zonotope reachability
# ---------------------------------------------------------------------------

def reduce_zono(layer, input_sets: List) -> List:
    """
    ReduceSum / ReduceMean reachability for Zono and ImageZono sets.

    Args:
        layer: ONNX reduce layer
        input_sets: List of Zono or ImageZono sets

    Returns:
        List of output sets
    """
    axes = _get_axes(layer)
    op = _get_reduce_op(layer)
    keepdims = _get_keepdims(layer)

    output_sets = []
    for s in input_sets:
        if isinstance(s, ImageZono):
            output_sets.append(_reduce_imagezono(s, axes, op, keepdims))
        elif isinstance(s, Zono):
            output_sets.append(_reduce_flat_zono(s, axes, op, keepdims))
        else:
            raise TypeError(
                f"reduce_zono requires Zono or ImageZono, got {type(s).__name__}"
            )
    return output_sets


def _reduce_imagezono(zono: ImageZono, axes: Optional[List[int]], op: str, keepdims: bool) -> Union[ImageZono, Zono]:
    """Apply reduce to an ImageZono."""
    h, w, c = zono.height, zono.width, zono.num_channels
    n_gen = zono.V.shape[1]

    c_img = zono.c.reshape(h, w, c)
    V_img = zono.V.reshape(h, w, c, n_gen)

    # Map ONNX axes (after batch removal, CHW) to HWC
    chw_to_hwc = {0: 2, 1: 0, 2: 1}

    if axes is None:
        hwc_axes = (0, 1, 2)
    else:
        hwc_axes = tuple(chw_to_hwc.get(a, a) for a in axes)

    reduce_fn = np.mean if op == 'mean' else np.sum

    c_out = reduce_fn(c_img, axis=hwc_axes, keepdims=keepdims)
    # For V, reduce along the same spatial axes (not the generator axis)
    V_out = reduce_fn(V_img, axis=hwc_axes, keepdims=keepdims)

    if keepdims:
        h_out = c_out.shape[0]
        w_out = c_out.shape[1]
        c_ch_out = c_out.shape[2]
        return ImageZono(
            c_out.reshape(-1, 1),
            V_out.reshape(-1, n_gen),
            h_out, w_out, c_ch_out
        )
    else:
        # Flatten to plain Zono
        c_flat = c_out.reshape(-1, 1)
        V_flat = V_out.reshape(-1, n_gen)
        return Zono(c_flat, V_flat)


def _reduce_flat_zono(zono: Zono, axes: Optional[List[int]], op: str, keepdims: bool) -> Zono:
    """Apply reduce to a flat Zono."""
    dim = zono.dim

    if axes is None or axes == [0] or (len(axes) == 1 and axes[0] == 0):
        if op == 'mean':
            W = np.ones((1, dim), dtype=np.float64) / dim
        else:
            W = np.ones((1, dim), dtype=np.float64)
        return zono.affine_map(W)
    else:
        raise NotImplementedError(
            f"reduce_zono: axes={axes} not supported for flat Zono (dim={dim})"
        )


# ---------------------------------------------------------------------------
# Box reachability
# ---------------------------------------------------------------------------

def reduce_box(layer, input_sets: List) -> List:
    """
    ReduceSum / ReduceMean reachability for Box sets.

    Uses interval arithmetic: for mean, new bounds are mean of old bounds.
    For sum, new bounds are sum of old bounds.

    Args:
        layer: ONNX reduce layer
        input_sets: List of Box sets

    Returns:
        List of output Box sets
    """
    axes = _get_axes(layer)
    op = _get_reduce_op(layer)
    keepdims = _get_keepdims(layer)

    output_sets = []
    for s in input_sets:
        if not isinstance(s, Box):
            raise TypeError(
                f"reduce_box requires Box, got {type(s).__name__}"
            )
        output_sets.append(_reduce_flat_box(s, axes, op, keepdims))
    return output_sets


def _reduce_flat_box(box: Box, axes: Optional[List[int]], op: str, keepdims: bool) -> Box:
    """Apply reduce to a flat Box using interval arithmetic."""
    if axes is None or axes == [0] or (len(axes) == 1 and axes[0] == 0):
        if op == 'mean':
            lb_out = np.mean(box.lb, axis=0, keepdims=True)
            ub_out = np.mean(box.ub, axis=0, keepdims=True)
        else:
            lb_out = np.sum(box.lb, axis=0, keepdims=True)
            ub_out = np.sum(box.ub, axis=0, keepdims=True)
        return Box(lb_out, ub_out)
    else:
        raise NotImplementedError(
            f"reduce_box: axes={axes} not supported for flat Box (dim={box.dim})"
        )
