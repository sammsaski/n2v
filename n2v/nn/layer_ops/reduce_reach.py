"""
ReduceSum / ReduceMean layer reachability operations.

Handles ONNX ReduceSum and ReduceMean operations, which reduce (sum or average)
tensor values along specified axes. Both are linear operations, so reachability
is exact for all set types — no splitting or approximation needed.
ReduceMax, ReduceMin, ReduceProd and ReduceLogSumExp are handled only where
they are the identity (over size-1 axes of an image set); the other reduce
operations are refused.
"""

import numpy as np
from typing import List, Optional, Union

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.ns_certificate import _widen_sum


# ---------------------------------------------------------------------------
# Helpers for extracting layer configuration
# ---------------------------------------------------------------------------

def _raw_axes(layer: object) -> Optional[List[int]]:
    """The ONNX axes stored on the layer (batch axis included), or None."""
    if getattr(layer, 'axes', None) is not None:
        return [int(a) for a in layer.axes]
    if getattr(layer, '_axes', None) is not None:
        return [int(a) for a in layer._axes]
    return None


def _is_noop(layer: object) -> bool:
    """True for a ReduceSum with no axes and noop_with_empty_axes set, which
    returns its input unchanged (whatever keepdims says)."""
    return (not _raw_axes(layer)
            and bool(getattr(layer, '_noop_with_empty_axes', 0)))


def _get_axes(layer: object, rank: Optional[int]) -> Optional[List[int]]:
    """
    Extract reduction axes from an ONNX reduce layer, removing batch dim.

    The layer stores ONNX axes which include the batch dimension (axis 0)
    and may be negative (counted from the end of the tensor's ``rank``
    axes). Negative axes are resolved against ``rank``, axis 0 is dropped
    and the remaining axes are shifted by -1.

    Args:
        layer: ONNX reduce layer
        rank: Rank of the input tensor including the batch axis (4 for an
            image set), or None when it is unknown (flat sets): a negative
            axis then cannot be resolved and raises NotImplementedError.

    Returns:
        None to reduce every non-batch axis (no axes given), otherwise the
        sorted axes in the set's coordinate system (no batch dim). The list
        is empty when only the batch axis is reduced, which leaves the
        values unchanged.
    """
    raw = _raw_axes(layer)
    if not raw:
        return None

    axes = set()
    for a in raw:
        if a < 0 and rank is None:
            raise NotImplementedError(
                f"reduce: negative axis {a} needs the input rank, which a "
                f"flat set does not carry")
        b = a + rank if a < 0 else a
        if b < 0 or (rank is not None and b >= rank):
            raise ValueError(
                f"reduce: axis {a} is out of range for a rank-{rank} input")
        if b != 0:  # axis 0 is the batch dim, of size 1
            axes.add(b - 1)
    return sorted(axes)


# Reductions whose value over a single element is that element
_IDENTITY_ON_ONE_ELEMENT = ('ReduceMax', 'ReduceMin', 'ReduceProd',
                            'ReduceLogSumExp')


def _get_reduce_op(layer: object,
                   reduced_sizes: Optional[List[int]] = None) -> str:
    """
    Determine whether the layer performs 'mean' or 'sum'.

    OnnxReduceStaticAxes also carries ReduceMax, ReduceMin, ReduceProd,
    ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp and ReduceSumSquare,
    which are not affine and have no handler here. ReduceMax, ReduceMin,
    ReduceProd and ReduceLogSumExp over axes that all have size 1 are the
    identity, the same as a sum over those axes, so they are reached as a
    sum. Every other case of these ops raises NotImplementedError.
    ReduceL1, ReduceL2, ReduceLogSum and ReduceSumSquare change the value
    even over a single element (|x|, log x, x^2), so they always raise.

    Args:
        layer: ONNX reduce layer
        reduced_sizes: Sizes of the reduced tensor axes, or None when they
            are not known (flat sets carry no tensor shape), in which case
            only ReduceSum and ReduceMean are accepted.

    Returns:
        'mean' or 'sum'
    """
    # OnnxReduceSumStaticAxes has no operation_type and is always a sum
    op = getattr(layer, 'operation_type', 'ReduceSum')
    if op == 'ReduceMean':
        return 'mean'
    if op == 'ReduceSum':
        return 'sum'
    if (op in _IDENTITY_ON_ONE_ELEMENT and reduced_sizes is not None
            and all(n == 1 for n in reduced_sizes)):
        return 'sum'
    raise NotImplementedError(
        f"{op} reachability is not supported (only ReduceSum and "
        f"ReduceMean, which are affine, and ReduceMax, ReduceMin, "
        f"ReduceProd and ReduceLogSumExp over axes of size 1 on an image "
        f"set)")


def _image_reduced_sizes(s: Union[ImageStar, ImageZono],
                         axes: Optional[List[int]]) -> List[int]:
    """Sizes of the image set's (C, H, W) axes named by ``axes`` (as
    returned by ``_get_axes``; None means all of them)."""
    sizes = (s.num_channels, s.height, s.width)
    return [sizes[a] for a in (range(3) if axes is None else axes)]


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


def _reduce_hwc(T: np.ndarray, axes: Optional[List[int]], op: str,
                keepdims: bool) -> np.ndarray:
    """
    Reduce an (H, W, C, k) array over the given tensor axes.

    ``axes`` are the non-batch axes of the (C, H, W) tensor, as returned by
    ``_get_axes``, and are mapped onto the (H, W, C) layout. With keepdims
    the result stays (H', W', C', k). Without, the reduced axes leave the
    tensor, and the result is (n, k) with its rows in the row-major order of
    the remaining (C, H, W) axes, which is how the following layers read a
    flat set. The reduction is done with keepdims first, so the reduced axes
    have size 1 there and do not change that order.
    """
    chw_to_hwc = (2, 0, 1)
    hwc_axes = (0, 1, 2) if axes is None else tuple(chw_to_hwc[a] for a in axes)
    if hwc_axes:
        reduce_fn = np.mean if op == 'mean' else np.sum
        T = reduce_fn(T, axis=hwc_axes, keepdims=True)
    if keepdims:
        return T
    return T.transpose(2, 0, 1, 3).reshape(-1, T.shape[-1])


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
    if _is_noop(layer):
        return list(input_sets)
    keepdims = _get_keepdims(layer)

    output_sets = []
    for s in input_sets:
        if isinstance(s, ImageStar):
            axes = _get_axes(layer, rank=4)
            op = _get_reduce_op(layer, _image_reduced_sizes(s, axes))
            output_sets.append(_reduce_imagestar(s, axes, op, keepdims))
        elif isinstance(s, Star):
            op = _get_reduce_op(layer)
            axes = _get_axes(layer, rank=None)
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
    0=C, 1=H, 2=W.  With keepdims the result is an ImageStar; without, it is
    a flat Star in the (C, H, W) order of the remaining axes.
    """
    if axes == [] and keepdims:
        return star
    V_out = _reduce_hwc(star.V, axes, op, keepdims)

    if keepdims:
        h_out, w_out, c_out = V_out.shape[:3]
        return ImageStar(
            V_out, star.C, star.d, star.predicate_lb, star.predicate_ub,
            h_out, w_out, c_out
        )
    return Star(V_out, star.C, star.d, star.predicate_lb, star.predicate_ub)


def _reduce_flat_star(star: Star, axes: Optional[List[int]], op: str, keepdims: bool) -> Star:
    """
    Apply reduce to a flat Star.

    V is (dim, nVar+1).  For a flat (1D) set, axes after batch removal
    is [0] meaning reduce along the single feature dimension. An empty list
    (only the batch axis) leaves the set unchanged.
    """
    dim = star.dim

    if axes == []:
        return star
    if axes is None or axes == [0]:
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
    if _is_noop(layer):
        return list(input_sets)
    keepdims = _get_keepdims(layer)

    output_sets = []
    for s in input_sets:
        if isinstance(s, ImageZono):
            axes = _get_axes(layer, rank=4)
            op = _get_reduce_op(layer, _image_reduced_sizes(s, axes))
            output_sets.append(_reduce_imagezono(s, axes, op, keepdims))
        elif isinstance(s, Zono):
            op = _get_reduce_op(layer)
            axes = _get_axes(layer, rank=None)
            output_sets.append(_reduce_flat_zono(s, axes, op, keepdims))
        else:
            raise TypeError(
                f"reduce_zono requires Zono or ImageZono, got {type(s).__name__}"
            )
    return output_sets


def _reduce_imagezono(zono: ImageZono, axes: Optional[List[int]], op: str, keepdims: bool) -> Union[ImageZono, Zono]:
    """Apply reduce to an ImageZono (rows in HWC order). With keepdims the
    result is an ImageZono; without, it is a flat Zono in the (C, H, W)
    order of the remaining axes."""
    if axes == [] and keepdims:
        return zono
    h, w, c = zono.height, zono.width, zono.num_channels
    n_gen = zono.V.shape[1]

    # The center and the generators go through the same helper, so they are
    # reduced over the same axes and flattened in the same order
    c_out = _reduce_hwc(zono.c.reshape(h, w, c, 1), axes, op, keepdims)
    V_out = _reduce_hwc(zono.V.reshape(h, w, c, n_gen), axes, op, keepdims)

    if keepdims:
        h_out, w_out, c_ch_out = c_out.shape[:3]
        return ImageZono(c_out.reshape(-1, 1), V_out.reshape(-1, n_gen),
                         h_out, w_out, c_ch_out)
    return Zono(c_out, V_out)


def _reduce_flat_zono(zono: Zono, axes: Optional[List[int]], op: str, keepdims: bool) -> Zono:
    """Apply reduce to a flat Zono. An empty ``axes`` (only the batch axis)
    leaves it unchanged."""
    dim = zono.dim

    if axes == []:
        return zono
    if axes is None or axes == [0]:
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
    if _is_noop(layer):
        return list(input_sets)
    axes = _get_axes(layer, rank=None)
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
    """Apply reduce to a flat Box using interval arithmetic. An empty
    ``axes`` (only the batch axis) leaves it unchanged."""
    if axes == []:
        return box
    if axes is None or axes == [0]:
        if op == 'mean':
            lb_out = np.mean(box.lb, axis=0, keepdims=True)
            ub_out = np.mean(box.ub, axis=0, keepdims=True)
        else:
            lb_out = np.sum(box.lb, axis=0, keepdims=True)
            ub_out = np.sum(box.ub, axis=0, keepdims=True)
        # Rounded outward: a sum of the nonzero entries (and one division
        # for the mean) over the magnitudes of the entries. Each direction is
        # charged for the endpoints it sums, so an infinite entry on one side
        # leaves the other bound finite, and a sum of exact zeros stays exact.
        reduce_abs = np.mean if op == 'mean' else np.sum

        def _mag_count(x):
            n = np.count_nonzero(x, axis=0, keepdims=True)
            count = n + ((op == 'mean') & (n > 0))
            return reduce_abs(np.abs(x), axis=0, keepdims=True), count

        lb_out = _widen_sum(lb_out, lb_out, *_mag_count(box.lb))[0]
        ub_out = _widen_sum(ub_out, ub_out, *_mag_count(box.ub))[1]
        return Box(lb_out, ub_out)
    else:
        raise NotImplementedError(
            f"reduce_box: axes={axes} not supported for flat Box (dim={box.dim})"
        )
