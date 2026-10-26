"""
Pre-pass for computing intermediate bounds before nonlinear layers.

Two strategies:
- 'ibp': Interval Bound Propagation using PyTorch forward passes. O(n) per layer,
         works for any model. Fast but loose bounds.
- 'zono': Zonotope propagation. O(n*g) per layer where g = generators.
          Tighter bounds but expensive for high-dimensional inputs.

These bounds are passed to Star reachability to skip LP calls for stable neurons
(provably always-active or always-inactive).

SOUNDNESS: the IBP bounds are rounded outward at every operation, so each one
encloses the real-number range of its node. The approximate ReLU/LeakyReLU
reach uses them as given (nn/bab.py relies on that to pin a forced neuron's
bound to exactly 0), and a round-to-nearest pass measurably gave an upper
bound of 0.0 for a true max of +2.8e-17, which zeroed the neuron. A bound the
pass computes exactly (e.g. a zero row, or an exact-zero lower bound after a
ReLU) stays exact.

The zonotope bounds are the outward-rounded ranges of the zonotopes the pass
computes, so they enclose those sets as stored. The pass itself runs in
round-to-nearest (the affine maps and the ReLU relaxation of the zonotope
reach), which can leave a computed zonotope a few ulps inside the
real-number range of its node. Along a path of Linear, Conv2d, ReLU, Flatten
and pass-through layers from the input, the pass carries a rigorous bound on
that distance and moves the recorded bounds out by it, so they enclose the
real-number range like the IBP bounds (see ``_zono_error_step``). Past any
other layer or graph op (BatchNorm, pooling, Sigmoid/Tanh, an ONNX MatMul,
Add or Reshape, ...) the distance is not bounded and the recorded bounds
enclose the computed zonotope only, the guarantee the star reach gives for
its own sets.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union, Optional, Any

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.ns_certificate import (
    _ETA, _SAFETY, _add_dir, _div_dir, _gamma, _mul_dir, _row_blocks,
    _widen_relative, _widen_sum, outward_interval_bounds,
)


logger = logging.getLogger(__name__)

# Layer types that are nonlinear and benefit from pre-computed bounds
NONLINEAR_TYPES = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)


def compute_intermediate_bounds(
    model: nn.Module,
    input_set: Union[Star, Zono, Box, ImageStar, ImageZono],
    method: str = 'ibp',
    input_shape: Optional[tuple] = None,
) -> Dict[Union[int, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Compute bounds before each nonlinear layer via a fast pre-pass.

    Args:
        model: PyTorch model (Sequential or GraphModule).
        input_set: Input set (Star, Zono, Box, ImageStar, or ImageZono).
        method: 'ibp' (interval bound propagation) or 'zono' (zonotope).
        input_shape: Batch-stripped model input shape, if known. The
            zonotope pass on a GraphModule uses it for the tensor shapes of
            shape-dependent ops (see ``n2v.nn.reach._propagate_shapes``).

    Returns:
        Dictionary mapping layer_id -> (lb, ub) numpy arrays.
        lb and ub have shape (dim, 1).
    """
    if method == 'ibp':
        return _compute_bounds_ibp(model, input_set)
    elif method == 'zono':
        return _compute_bounds_zono(model, input_set, input_shape)
    else:
        raise ValueError(f"Unknown precompute method: {method!r}. Use 'ibp' or 'zono'.")


# ============================================================================
# IBP (Interval Bound Propagation) — fast, works for any model
# ============================================================================

def _compute_bounds_ibp(
    model: nn.Module,
    input_set: Union[Star, Zono, Box, ImageStar, ImageZono],
) -> Dict[Union[int, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Compute bounds via Interval Bound Propagation.

    Hooks into PyTorch forward pass to record pre-activation bounds at every
    nonlinear layer. Uses two forward passes with lb and ub tensors.
    """
    import torch.fx as fx

    lb, ub = _extract_bounds(input_set)

    # Determine initial spatial shape from input set
    spatial_shape = None
    if isinstance(input_set, (ImageStar, ImageZono)):
        spatial_shape = (input_set.num_channels, input_set.height, input_set.width)

    if isinstance(model, fx.GraphModule):
        return _ibp_graphmodule(model, lb, ub, spatial_shape)
    else:
        return _ibp_sequential(model, lb, ub, spatial_shape)


def _hwc_flat_to_chw_flat(v: np.ndarray, height: int, width: int,
                          channels: int) -> np.ndarray:
    """Reorder a spatial vector from the ImageStar/ImageZono HWC ravel to the
    NCHW (channel-major) ravel the IBP forward pass assumes.

    ImageStar/ImageZono store their data (H, W, C) and ``estimate_ranges`` /
    ``get_bounds`` ravel it in that HWC order. The IBP conv/pool/batchnorm
    helpers, however, ``reshape`` their flat input to ``(C, H, W)`` (torch's
    NCHW layout). Feeding an HWC-flat vector into a ``(C, H, W)`` reshape
    silently scrambles the image (channels read as rows), so multi-channel
    CNNs produce wrong — and unsound — precomputed bounds. Reorder up front."""
    return v.reshape(height, width, channels).transpose(2, 0, 1).reshape(-1)


def _record_spatial_bounds(lb: np.ndarray, ub: np.ndarray,
                           shape: Optional[tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """Package (lb, ub) for a nonlinear layer so their row order matches the
    Star the reach consumer builds.

    The IBP pass carries NCHW (channel-major) flat vectors. A spatial ReLU in
    the reach path runs on an ImageStar, which ``to_star()`` flattens in HWC
    order; the precomputed bounds are matched positionally against those rows,
    so a spatial (rank-3 ``(C, H, W)``) layer's bounds must be reordered
    CHW->HWC. Flat layers (``shape is None``, e.g. after a flatten/Linear)
    already agree and are passed through unchanged."""
    if shape is not None and len(shape) == 3:
        c, h, w = shape
        lb = lb.reshape(c, h, w).transpose(1, 2, 0).reshape(-1)
        ub = ub.reshape(c, h, w).transpose(1, 2, 0).reshape(-1)
    return lb.reshape(-1, 1), ub.reshape(-1, 1)


def _extract_bounds(input_set: Union[Star, Zono, Box, ImageStar, ImageZono]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (lb, ub) numpy arrays from any input set type, in the NCHW
    (channel-major) order the IBP forward pass reshapes to. Spatial sets store
    HWC, so their ranges are reordered to CHW here (see
    :func:`_hwc_flat_to_chw_flat`)."""
    if isinstance(input_set, Box):
        return input_set.lb.flatten(), input_set.ub.flatten()
    elif isinstance(input_set, ImageStar):
        lb, ub = input_set.estimate_ranges()
        h, w, c = input_set.height, input_set.width, input_set.num_channels
        return (_hwc_flat_to_chw_flat(lb.flatten(), h, w, c),
                _hwc_flat_to_chw_flat(ub.flatten(), h, w, c))
    elif isinstance(input_set, Star):
        lb, ub = input_set.estimate_ranges()
        return lb.flatten(), ub.flatten()
    elif isinstance(input_set, ImageZono):
        lb, ub = input_set.get_bounds()
        h, w, c = input_set.height, input_set.width, input_set.num_channels
        return (_hwc_flat_to_chw_flat(lb.flatten(), h, w, c),
                _hwc_flat_to_chw_flat(ub.flatten(), h, w, c))
    elif isinstance(input_set, Zono):
        lb, ub = input_set.get_bounds()
        return lb.flatten(), ub.flatten()
    else:
        raise TypeError(f"Unsupported input_set type: {type(input_set).__name__}")


def _ibp_linear(lb: np.ndarray, ub: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """IBP through a linear layer: y = W*x + b, rounded outward
    (``outward_interval_bounds``: c = b, generators W, box [lb, ub])."""
    weight = np.asarray(weight, dtype=np.float64)
    center = (np.zeros(weight.shape[0]) if bias is None
              else np.asarray(bias, dtype=np.float64).reshape(-1))
    return outward_interval_bounds(center, weight, lb, ub)


def _ibp_conv(
    lb: np.ndarray,
    ub: np.ndarray,
    layer: Union[nn.Conv2d, nn.Conv1d],
    input_shape: tuple,
) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """IBP through Conv2d/Conv1d using PyTorch (handles padding, stride, dilation)."""
    # Reshape to NCHW/NCW
    lb_t = torch.tensor(lb.reshape(input_shape), dtype=torch.float64).unsqueeze(0)
    ub_t = torch.tensor(ub.reshape(input_shape), dtype=torch.float64).unsqueeze(0)

    weight = layer.weight.detach().double()
    W_pos = torch.clamp(weight, min=0)
    W_neg = torch.clamp(weight, max=0)

    # Convolve with split positive/negative weights. ``_conv_forward``
    # applies the layer's own padding mode: a mode other than 'zeros' copies
    # border entries, and a copy's bounds are those of the entry it copies.
    def conv_fn(x, w):
        return layer._conv_forward(x, w, None)

    new_lb = conv_fn(lb_t, W_pos) + conv_fn(ub_t, W_neg)
    new_ub = conv_fn(ub_t, W_pos) + conv_fn(lb_t, W_neg)

    # Rounding: each output is a sum of the nonzero weight x endpoint
    # products of its window plus the bias. The same convolution of the
    # magnitudes bounds the terms, and of the nonzero indicators counts them
    # (small integers, exact).
    x_abs = torch.maximum(lb_t.abs(), ub_t.abs())
    mag = conv_fn(x_abs, weight.abs())
    count = conv_fn((x_abs != 0).double(), (weight != 0).double())

    if layer.bias is not None:
        bias = layer.bias.detach().double()
        # Reshape bias for broadcasting: (1, C, 1, 1) or (1, C, 1)
        shape = [1, -1] + [1] * (lb_t.dim() - 2)
        new_lb = new_lb + bias.view(*shape)
        new_ub = new_ub + bias.view(*shape)
        mag = mag + bias.abs().view(*shape)

    out_shape = new_lb.squeeze(0).shape
    lo, hi = _widen_sum(new_lb.numpy().ravel(), new_ub.numpy().ravel(),
                        mag.numpy().ravel(), count.numpy().ravel())
    return lo, hi, out_shape


def _ibp_relu(lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """IBP through ReLU."""
    return np.maximum(lb, 0), np.maximum(ub, 0)


def _ibp_leakyrelu(lb: np.ndarray, ub: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """IBP through LeakyReLU, f(x) = x for x >= 0 and gamma * x otherwise.

    f is linear on each side of 0, so its extremes over [lb, ub] are among
    f(lb), f(ub) and, for a crossing interval, f(0) = 0. Taking the hull of
    those values holds for any slope: a negative gamma (allowed by ONNX
    LeakyRelu) makes f non-monotone, so mapping lb to lb and ub to ub
    would drop the gamma * lb maximum of a crossing neuron and invert the
    bounds of an all-negative one. The products gamma * x are rounded
    outward."""
    g = np.float64(gamma)
    # f at each endpoint, rounded down (for the lower bound) and up.
    f_lb_dn = np.where(lb >= 0, lb, _mul_dir(g, lb, -1.0))
    f_lb_up = np.where(lb >= 0, lb, _mul_dir(g, lb, 1.0))
    f_ub_dn = np.where(ub >= 0, ub, _mul_dir(g, ub, -1.0))
    f_ub_up = np.where(ub >= 0, ub, _mul_dir(g, ub, 1.0))
    new_lb = np.minimum(f_lb_dn, f_ub_dn)
    new_ub = np.maximum(f_lb_up, f_ub_up)
    crossing = (lb < 0) & (ub > 0)
    new_lb = np.where(crossing, np.minimum(new_lb, 0.0), new_lb)
    new_ub = np.where(crossing, np.maximum(new_ub, 0.0), new_ub)
    return new_lb, new_ub


def _ibp_sigmoid(lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """IBP through Sigmoid (monotone). ``expit`` is accurate to a few ulps,
    not correctly rounded, so the bounds are moved outward by a relative
    ``gamma_4`` (times the safety factor) and clipped to the range [0, 1]."""
    from scipy.special import expit
    s_ub = expit(ub)
    lo, hi = _widen_relative(expit(lb), s_ub, lb, ub, 4)
    # expit returns exactly 0 once exp(-x) overflows (x below about -709.78),
    # where the true value is positive, so the relative widening leaves the
    # upper bound below it. For such x, sigmoid(x) < e^x < 2^-1021, so an
    # upper bound computed from a value under 2^-1021 is raised to 2^-1021
    # (a widened bound above it is kept).
    floor = 2.0 ** -1021
    hi = np.where(s_ub < floor, np.maximum(hi, floor), hi)
    return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)


def _ibp_tanh(lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """IBP through Tanh (monotone), moved outward the same way as
    :func:`_ibp_sigmoid` and clipped to the range [-1, 1]."""
    lo, hi = _widen_relative(np.tanh(lb), np.tanh(ub), lb, ub, 4)
    return np.clip(lo, -1.0, 1.0), np.clip(hi, -1.0, 1.0)


def _ibp_pool(
    lb: np.ndarray,
    ub: np.ndarray,
    layer: Union[nn.MaxPool2d, nn.AvgPool2d],
    spatial_shape: tuple,
) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """IBP through MaxPool2d/AvgPool2d."""
    lb_t = torch.tensor(lb.reshape(spatial_shape), dtype=torch.float64).unsqueeze(0)
    ub_t = torch.tensor(ub.reshape(spatial_shape), dtype=torch.float64).unsqueeze(0)

    if isinstance(layer, nn.MaxPool2d):
        # MaxPool is monotone
        new_lb = nn.functional.max_pool2d(lb_t, layer.kernel_size, layer.stride,
                                           layer.padding, layer.dilation,
                                           layer.ceil_mode).squeeze(0)
        new_ub = nn.functional.max_pool2d(ub_t, layer.kernel_size, layer.stride,
                                           layer.padding, layer.dilation,
                                           layer.ceil_mode).squeeze(0)
    elif isinstance(layer, nn.AvgPool2d):
        # AvgPool is linear with nonnegative weights, so it maps lb to lb and
        # ub to ub. The reach's own pooling divides border windows the way
        # the layer does (count_include_pad=False, the ONNX default, averages
        # only the in-image entries) and raises for a divisor torch and ONNX
        # disagree on, which leaves the graph pass untrusted downstream.
        from n2v.nn.layer_ops.avgpool2d_reach import _avg_pool
        from n2v.nn.layer_ops.conv2d_reach import kernel_area
        new_lb = _avg_pool(layer, lb_t).squeeze(0)
        new_ub = _avg_pool(layer, ub_t).squeeze(0)
        # Rounding: a window sum and one division. The pooled magnitudes
        # bound the terms; a window with no nonzero entry is an exact 0.
        x_abs = torch.maximum(lb_t.abs(), ub_t.abs())
        mag = _avg_pool(layer, x_abs).squeeze(0).numpy().ravel()
        live = _avg_pool(layer, (x_abs != 0).double()).squeeze(0).numpy().ravel()
        k = kernel_area(layer)
        lo, hi = _widen_sum(new_lb.numpy().ravel(), new_ub.numpy().ravel(),
                            mag, np.where(live > 0, k + 1, 0))
        return lo, hi, new_lb.shape
    else:
        raise NotImplementedError(f"Pool IBP not implemented for {type(layer)}")

    # MaxPool selects entries: exact.
    return new_lb.numpy().flatten(), new_ub.numpy().flatten(), new_lb.shape


def _ibp_transpose(
    layer: nn.Module,
    lb: np.ndarray,
    ub: np.ndarray,
    spatial_shape: Optional[tuple],
) -> Tuple[np.ndarray, np.ndarray, Optional[tuple]]:
    """IBP through OnnxTranspose.

    A transpose moves entries, so the bounds must move with them. The pass
    knows the tensor shape only for an image, (C, H, W) behind a size-1
    batch axis: a batch-preserving rank-4 perm of such an image is applied
    exactly, the same way the reach permutes an ImageStar, and the result is
    again an image. A flat vector carries no shape here, and only the
    identity perm provably keeps its order, so any other perm raises and
    the graph pass stops trusting the bounds downstream.
    """
    perm = getattr(layer, 'perm', None)
    if perm is not None:
        perm = [int(p) for p in perm]
        if perm == list(range(len(perm))):
            return lb, ub, spatial_shape
        if (spatial_shape is not None and len(spatial_shape) == 3
                and len(perm) == 4 and perm[0] == 0):
            axes = [p - 1 for p in perm[1:]]  # over (C, H, W)
            new_lb = lb.reshape(spatial_shape).transpose(axes)
            new_ub = ub.reshape(spatial_shape).transpose(axes)
            return new_lb.reshape(-1), new_ub.reshape(-1), new_lb.shape
    raise NotImplementedError(
        f"IBP through Transpose perm={perm} needs the tensor shape")


def _ibp_batchnorm(
    lb: np.ndarray,
    ub: np.ndarray,
    layer: Union[nn.BatchNorm1d, nn.BatchNorm2d],
    spatial_shape: tuple,
) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """IBP through BatchNorm (affine: y = gamma * (x - mean) / std + beta)."""
    # BN in eval mode is just an affine transform per channel
    mean = layer.running_mean.detach().double().numpy()
    var = layer.running_var.detach().double().numpy()
    eps = layer.eps
    gamma = layer.weight.detach().double().numpy() if layer.weight is not None else np.ones_like(mean)
    beta = layer.bias.detach().double().numpy() if layer.bias is not None else np.zeros_like(mean)

    scale = gamma / np.sqrt(var + eps)
    shift = beta - scale * mean

    # Reshape for broadcasting over spatial dims
    lb_r = lb.reshape(spatial_shape)
    ub_r = ub.reshape(spatial_shape)
    n_spatial = len(spatial_shape) - 1  # dims after channel
    shape = [-1] + [1] * n_spatial
    scale_r = scale.reshape(shape)
    shift_r = shift.reshape(shape)

    # Affine: handle negative scale
    scale_pos = np.maximum(scale_r, 0)
    scale_neg = np.minimum(scale_r, 0)
    new_lb = scale_pos * lb_r + scale_neg * ub_r + shift_r
    new_ub = scale_pos * ub_r + scale_neg * lb_r + shift_r

    # Rounding: scale and shift carry a few roundings each (add, sqrt,
    # divide, multiply, subtract) against the real-number layer, and the map
    # two more. gamma_8 over the magnitudes of every term covers all of them.
    x_abs = np.maximum(np.abs(lb_r), np.abs(ub_r))
    mag = (np.abs(scale_r) * x_abs + np.abs(beta).reshape(shape)
           + np.abs(scale_r * mean.reshape(shape)))
    lo, hi = _widen_sum(new_lb.ravel(), new_ub.ravel(),
                        np.broadcast_to(mag, new_lb.shape).ravel(), 6)
    return lo, hi, new_lb.shape


def _ibp_sequential(
    model: nn.Module,
    lb: np.ndarray,
    ub: np.ndarray,
    spatial_shape: Optional[tuple] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """IBP for Sequential models."""
    layer_bounds = {}

    layers = list(model.children())
    if not layers:
        layers = [model]

    for i, layer in enumerate(layers):
        # Record bounds BEFORE nonlinear layers (reordered CHW->HWC for a
        # spatial layer so they line up with the reach set's row order).
        if isinstance(layer, NONLINEAR_TYPES):
            layer_bounds[i] = _record_spatial_bounds(lb, ub, spatial_shape)

        # Propagate
        lb, ub, spatial_shape = _ibp_propagate_layer(layer, lb, ub, spatial_shape)

    return layer_bounds


def _get_param_np(graph_module: Any, node: Any) -> np.ndarray:
    """Value of a constant node as a numpy array: a get_attr, an OnnxConstant,
    or a constant subgraph (ops applied to constants only), folded the way
    the reach folds it."""
    from n2v.nn.reach import _get_parameter
    return _get_parameter(graph_module, node).detach().cpu().numpy()


def _ibp_onnx_matmul(graph_module, node, node_bounds, const_names):
    """IBP through OnnxMatMul ``y = x @ W`` (W a constant node)."""
    if len(node.args) != 2:
        return None
    a, w = node.args
    if not (hasattr(a, 'name') and a.name in node_bounds):
        return None
    if getattr(w, 'name', None) not in const_names:
        return None
    W = _get_param_np(graph_module, w)
    la, ua, _ = node_bounds[a.name]
    la_flat = la.flatten()
    # Soundness/robustness guard: only the plain flat ``x @ W`` form (W a 2-D matrix whose
    # contraction dim matches the flattened activation) is handled here. For any other shape
    # (multi-dim left operand, weight-on-the-left, conv-flattened/batched matmul) return None
    # so the caller falls back to the untrusted/LP path -- mirrors ``_ibp_onnx_binary``. IBP is
    # only a precompute speedup, so skipping it stays sound and avoids a hard ValueError crash
    # on these matmuls (see status repo CRASH_ROOTCAUSE.md, Bug A).
    if W.ndim != 2 or la_flat.shape[0] != W.shape[0]:
        return None
    new_lb, new_ub = _ibp_linear(la_flat, ua.flatten(), W.T, None)
    return new_lb, new_ub, None


def _ibp_onnx_binary(graph_module, node, module, node_bounds, const_names):
    """IBP through OnnxBinaryMathOperation (Add/Sub/Mul/Div).

    Handles both the residual case (two computed branches) and the
    constant case (second operand a constant node, see
    :func:`_get_param_np`). Returns None when
    the operation cannot be soundly propagated (e.g. a per-channel
    constant that does not broadcast onto a flattened image), so the
    caller marks the result untrusted and falls back to LP.
    """
    if len(node.args) != 2 or not hasattr(module, 'math_op_function'):
        return None
    a, b = node.args
    op = module.math_op_function.__name__
    a_known = hasattr(a, 'name') and a.name in node_bounds
    b_known = hasattr(b, 'name') and b.name in node_bounds

    if a_known and b_known:  # residual: both operands are computed sets
        la, ua, sa = node_bounds[a.name]
        lb2, ub2, _ = node_bounds[b.name]
        la, ua, lb2, ub2 = (x.flatten() for x in (la, ua, lb2, ub2))
        if la.shape != lb2.shape:
            return None
        # Every result below is rounded outward (see _add_dir / _mul_dir).
        if 'add' in op:
            return _add_dir(la, lb2, -1.0), _add_dir(ua, ub2, 1.0), sa
        if 'sub' in op:
            return _add_dir(la, -ub2, -1.0), _add_dir(ua, -lb2, 1.0), sa
        if op == 'mul':
            pairs = [(la, lb2), (la, ub2), (ua, lb2), (ua, ub2)]
            lo = np.min([_mul_dir(x, y, -1.0) for x, y in pairs], axis=0)
            hi = np.max([_mul_dir(x, y, 1.0) for x, y in pairs], axis=0)
            return lo, hi, sa
        return None

    if not a_known or getattr(b, 'name', None) not in const_names:
        return None
    param = _get_param_np(graph_module, b).flatten()
    la, ua, sa = node_bounds[a.name]
    la, ua = la.flatten(), ua.flatten()
    if param.size not in (1, la.size):
        return None  # non-broadcastable (e.g. per-channel on flat image)
    param = param.astype(np.float64)
    if op == 'mul':
        lo = np.minimum(_mul_dir(la, param, -1.0), _mul_dir(ua, param, -1.0))
        hi = np.maximum(_mul_dir(la, param, 1.0), _mul_dir(ua, param, 1.0))
        return lo, hi, sa
    if op == '_onnx_div':
        if np.any(param == 0):
            return None
        lo = np.minimum(_div_dir(la, param, -1.0), _div_dir(ua, param, -1.0))
        hi = np.maximum(_div_dir(la, param, 1.0), _div_dir(ua, param, 1.0))
        return lo, hi, sa
    if 'add' in op:
        return _add_dir(la, param, -1.0), _add_dir(ua, param, 1.0), sa
    if 'sub' in op:
        return _add_dir(la, -param, -1.0), _add_dir(ua, -param, 1.0), sa
    return None


def _ibp_graphmodule(
    graph_module: Any,
    lb: np.ndarray,
    ub: np.ndarray,
    spatial_shape: Optional[tuple] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """IBP for GraphModule (ONNX-converted or traced) models.

    Tracks, per node, whether its bounds are *trusted* (every op on the
    path was soundly propagated). Bounds are recorded for a nonlinear
    layer only when its input is trusted — so an unhandled op makes the
    pre-pass skip downstream layers (reach falls back to LP for them)
    rather than attach stale / wrong-dimension bounds. A node the pass does
    not propagate (an unhandled call_function or call_method such as a
    ``view``) gets no entry, and a consumer of a node without an entry is
    untrusted: it never falls back to the network input box.
    """
    import operator
    from n2v.nn.reach import _constant_node_names, _function_node_to_module

    named_modules = dict(graph_module.named_modules())
    # Nodes that depend on no input (parameters, OnnxConstant, ops on
    # constants only). They carry no bounds: consumers read their value.
    const_names = _constant_node_names(graph_module)
    node_bounds = {}    # node_name -> (lb, ub, spatial_shape)
    node_trusted = {}   # node_name -> bool
    layer_bounds = {}

    def _propagate(node, module):
        """Record bounds before a nonlinear layer and propagate through a
        standard layer; a failure leaves the output untrusted."""
        arg = node.args[0] if node.args else None
        name = getattr(arg, 'name', None)
        if name in node_bounds:
            cur_lb, cur_ub, cur_shape = node_bounds[name]
            cur_trusted = node_trusted.get(name, False)
        else:
            # The producer was not propagated: nothing is known about the
            # input, so nothing downstream may be trusted.
            cur_lb, cur_ub, cur_shape = lb, ub, spatial_shape
            cur_trusted = False

        # Record bounds before nonlinear layers (only if trusted),
        # reordered CHW->HWC for a spatial layer so they line up with the
        # reach set's row order (ImageStar.to_star() is HWC).
        if isinstance(module, NONLINEAR_TYPES) and cur_trusted:
            layer_bounds[node.name] = _record_spatial_bounds(
                cur_lb.reshape(-1), cur_ub.reshape(-1), cur_shape)
        try:
            new_lb, new_ub, new_shape = _ibp_propagate_layer(
                module, cur_lb, cur_ub, cur_shape)
            node_bounds[node.name] = (new_lb, new_ub, new_shape)
            node_trusted[node.name] = cur_trusted
        except Exception:  # noqa: BLE001 - unknown op: stop trusting
            node_bounds[node.name] = (cur_lb, cur_ub, cur_shape)
            node_trusted[node.name] = False

    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            node_bounds[node.name] = (lb, ub, spatial_shape)
            node_trusted[node.name] = True

        elif node.name in const_names:
            pass

        elif node.op == 'call_module':
            module = named_modules.get(node.target)
            if module is None:
                continue

            # Propagate. Try ONNX graph ops first, then standard layers.
            module_type = type(module).__name__
            result = None
            try:
                if module_type == 'OnnxMatMul':
                    result = _ibp_onnx_matmul(
                        graph_module, node, node_bounds, const_names)
                elif module_type == 'OnnxBinaryMathOperation':
                    result = _ibp_onnx_binary(
                        graph_module, node, module, node_bounds, const_names)
            except Exception:  # noqa: BLE001 - IBP precompute is an optimization; never crash reach
                result = None

            if result is not None:
                # Trusted only if every operand with bounds is: a residual
                # add must not inherit trust from its first operand alone.
                # Constant operands carry no entry.
                node_bounds[node.name] = result
                node_trusted[node.name] = all(
                    node_trusted.get(arg.name, False) for arg in node.args
                    if hasattr(arg, 'name') and arg.name in node_bounds)
            else:
                _propagate(node, module)

        elif node.op == 'call_function':
            if node.target is operator.getitem:
                src_node = node.args[0]
                if hasattr(src_node, 'name') and src_node.name in node_bounds:
                    node_bounds[node.name] = node_bounds[src_node.name]
                    node_trusted[node.name] = node_trusted.get(
                        src_node.name, False)
            else:
                # Functional activations and torch.flatten, converted the
                # way the reach converts them (it looks up
                # layer_bounds[node.name] for a functional activation too).
                # Any other function gets no entry.
                try:
                    equiv_module = _function_node_to_module(node)
                except Exception:  # noqa: BLE001 - unhandled: no entry
                    equiv_module = None
                if equiv_module is not None:
                    _propagate(node, equiv_module)

        elif node.op == 'call_method':
            # x.flatten() keeps the flat order, like nn.Flatten, and
            # x.contiguous() only changes the memory layout. Other methods
            # (view, reshape, size, ...) get no entry.
            if node.target == 'flatten':
                _propagate(node, nn.Flatten())
            elif node.target == 'contiguous':
                _propagate(node, nn.Identity())

        elif node.op == 'output':
            pass

    return layer_bounds


def _ibp_propagate_layer(
    layer: nn.Module,
    lb: np.ndarray,
    ub: np.ndarray,
    spatial_shape: Optional[tuple],
) -> Tuple[np.ndarray, np.ndarray, Optional[tuple]]:
    """Propagate IBP bounds through a single layer. Returns (lb, ub, spatial_shape)."""

    if isinstance(layer, nn.Linear):
        W = layer.weight.detach().double().numpy()
        b = layer.bias.detach().double().numpy() if layer.bias is not None else None
        new_lb, new_ub = _ibp_linear(lb, ub, W, b)
        return new_lb, new_ub, None

    elif isinstance(layer, nn.ReLU):
        new_lb, new_ub = _ibp_relu(lb, ub)
        return new_lb, new_ub, spatial_shape

    elif isinstance(layer, nn.LeakyReLU):
        new_lb, new_ub = _ibp_leakyrelu(lb, ub, layer.negative_slope)
        return new_lb, new_ub, spatial_shape

    elif isinstance(layer, nn.Sigmoid):
        new_lb, new_ub = _ibp_sigmoid(lb, ub)
        return new_lb, new_ub, spatial_shape

    elif isinstance(layer, nn.Tanh):
        new_lb, new_ub = _ibp_tanh(lb, ub)
        return new_lb, new_ub, spatial_shape

    elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
        if spatial_shape is None:
            # Infer spatial shape from layer
            if isinstance(layer, nn.Conv2d):
                c_in = layer.in_channels
                spatial_size = len(lb) // c_in
                h = w = int(np.sqrt(spatial_size))
                if h * w != spatial_size:
                    # Non-square — can't infer, use torch to figure it out
                    # Try common aspect ratios
                    for candidate_h in range(1, spatial_size + 1):
                        if spatial_size % candidate_h == 0:
                            candidate_w = spatial_size // candidate_h
                            if abs(candidate_h - candidate_w) < max(candidate_h, candidate_w):
                                h, w = candidate_h, candidate_w
                                break
                spatial_shape = (c_in, h, w)
            else:
                c_in = layer.in_channels
                seq_len = len(lb) // c_in
                spatial_shape = (c_in, seq_len)
        new_lb, new_ub, out_shape = _ibp_conv(lb, ub, layer, spatial_shape)
        return new_lb, new_ub, tuple(out_shape)

    elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
        if spatial_shape is None:
            # Pooling changes the size; without the image shape it cannot be
            # propagated, and passing the bounds through would misplace them.
            raise NotImplementedError(
                f"IBP through {type(layer).__name__} needs the image shape")
        new_lb, new_ub, out_shape = _ibp_pool(lb, ub, layer, spatial_shape)
        return new_lb, new_ub, tuple(out_shape)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if spatial_shape is not None:
            new_lb, new_ub, out_shape = _ibp_batchnorm(lb, ub, layer, spatial_shape)
            return new_lb, new_ub, tuple(out_shape)
        else:
            # 1D case
            new_lb, new_ub, out_shape = _ibp_batchnorm(lb, ub, layer, (len(lb),))
            return new_lb, new_ub, None

    elif isinstance(layer, (nn.Flatten,)):
        return lb, ub, None

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return lb, ub, spatial_shape

    else:
        # Check for onnx2torch types
        module_type = type(layer).__name__
        if module_type == 'OnnxTranspose':
            return _ibp_transpose(layer, lb, ub, spatial_shape)
        if module_type in ('OnnxFlatten', 'OnnxCast'):
            return lb, ub, None if module_type == 'OnnxFlatten' else spatial_shape
        elif module_type == 'OnnxNeg':
            return -ub, -lb, spatial_shape

        raise NotImplementedError(f"IBP not implemented for {type(layer).__name__}")


# ============================================================================
# Zonotope pre-pass — tighter bounds, expensive for large inputs
# ============================================================================

def _compute_bounds_zono(
    model: nn.Module,
    input_set: Union[Star, Zono, Box, ImageStar, ImageZono],
    input_shape: Optional[tuple] = None,
) -> Dict[Union[int, str], Tuple[np.ndarray, np.ndarray]]:
    """Compute bounds using Zonotope propagation."""
    import torch.fx as fx

    zono_set, err = _convert_to_zono(input_set)

    if isinstance(model, fx.GraphModule):
        return _zono_graphmodule(model, zono_set, err, input_shape)
    else:
        return _zono_sequential(model, zono_set, err)


def _convert_to_zono(
    input_set: Union[Star, Zono, Box, ImageStar, ImageZono],
) -> Tuple[Union[Zono, ImageZono], np.ndarray]:
    """Convert any supported input set to Zono or ImageZono.

    Returns ``(zono, err)``: every point of the input set is ``z + delta``
    with ``z`` in the zonotope and ``|delta_i| <= err[i]`` (see
    :func:`_zono_error_step`). A zonotope input is used as given (``err`` is
    0). Any other set becomes the zonotope of its interval enclosure, whose
    center and radii ``from_bounds`` rounds to nearest, so the zonotope can
    miss the enclosure by an ulp; ``err`` is that miss, rounded up.
    """
    if isinstance(input_set, ImageZono):
        return input_set, np.zeros(input_set.dim)
    elif isinstance(input_set, ImageStar):
        lb, ub = input_set.estimate_ranges()
        zono = ImageZono.from_bounds(
            lb.reshape(input_set.height, input_set.width, input_set.num_channels),
            ub.reshape(input_set.height, input_set.width, input_set.num_channels),
            input_set.height, input_set.width, input_set.num_channels,
        )
    elif isinstance(input_set, Zono):
        return input_set, np.zeros(input_set.dim)
    elif isinstance(input_set, Star):
        lb, ub = input_set.estimate_ranges()
        zono = Zono.from_bounds(lb, ub)
    elif isinstance(input_set, Box):
        lb, ub = input_set.lb, input_set.ub
        zono = Zono.from_bounds(lb, ub)
    else:
        raise TypeError(f"Unsupported input_set type: {type(input_set).__name__}")
    # from_bounds builds the diagonal zonotope [c - v, c + v] per row, and
    # (H, W, C) image bounds flatten to the same rows as the ImageZono.
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    c = zono.c.reshape(-1)
    v = np.abs(np.diagonal(zono.V))
    over = _add_dir(_add_dir(ub, -c, 1.0), -v, 1.0)     # ub - (c + v)
    under = _add_dir(_add_dir(c, -v, 1.0), -lb, 1.0)    # (c - v) - lb
    return zono, np.maximum(np.maximum(over, under), 0.0)


def _zono_row_bounds(s: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Flat interval bounds of a set of the zonotope pass. A zonotope's are
    the outward-rounded range of ``c + V alpha`` over ``[-1, 1]^m``, which
    encloses it exactly as stored; any other set (not produced by the
    zonotope reach today) gives its own ``estimate_ranges``."""
    if isinstance(s, Zono):
        m = s.V.shape[1]
        return outward_interval_bounds(np.asarray(s.c).reshape(-1), s.V,
                                       -np.ones(m), np.ones(m))
    lo, hi = s.estimate_ranges()
    return np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)


def _zono_layer_bounds(bounds: list, err: Optional[np.ndarray],
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Bounds recorded for a nonlinear layer from the :func:`_zono_row_bounds`
    of the sets entering it: their hull, moved outward by ``err`` (per row)
    when the distance between the computed sets and the real-number ones is
    known."""
    lbs, ubs = zip(*bounds)
    lb = np.min(lbs, axis=0)
    ub = np.max(ubs, axis=0)
    if err is not None and err.shape == lb.shape:
        lb = _add_dir(lb, -err, -1.0)
        ub = _add_dir(ub, err, 1.0)
    return lb.reshape(-1, 1), ub.reshape(-1, 1)


def _zono_magnitude(sets: list) -> np.ndarray:
    """Per row, the largest ``|c_i| + sum_k |V_ik|`` over the sets: a bound on
    the magnitude of the terms an affine map of a set sums. Computed in
    round-to-nearest, a sum of nonnegative terms that is off by a relative
    ``gamma_m`` at most, which the ``_SAFETY`` factor of the caller covers."""
    mag = None
    for s in sets:
        r = np.empty(s.dim)
        for sl, blk in _row_blocks(s.V, s.V.shape[1]):
            r[sl] = np.abs(blk).sum(axis=1)
        r += np.abs(s.c.reshape(-1))
        mag = r if mag is None else np.maximum(mag, r)
    return mag


def _zono_error_step(
    module: nn.Module,
    sets_in: list,
    sets_out: list,
    err: Optional[np.ndarray],
    bounds_in: Optional[list] = None,
) -> Optional[np.ndarray]:
    """Carry the rounding error of the zonotope pass through one layer.

    ``err`` bounds, per row, how far the real-number network can lie from
    the computed sets at the layer's input: every point it reaches there is
    ``z + delta`` for a point ``z`` of one of ``sets_in`` and
    ``|delta_i| <= err[i]``. Returns the same bound for ``sets_out`` (the
    layer's output sets, in order), or None when this layer's rounding is
    not bounded here; the caller then records the computed sets' own bounds.
    ``bounds_in`` are the :func:`_zono_row_bounds` of ``sets_in`` if the
    caller has them.

    - Pass-through (the output sets are the input sets): ``err`` is kept.
    - Linear: the zonotope reach computes ``W c + b`` and ``W V`` in
      float64. The rounding of a row is at most ``gamma_{n+1}`` times
      ``|W| (|c| + sum_k |V_k|) + |b|`` (Higham's inner-product bound, in
      any summation order; ``n`` the fan-in), plus half a subnormal for each
      product of two nonzero factors, which can underflow (``m + 1`` inner
      products of at most as many such products as the row has nonzero
      weights on nonzero input rows; ``m`` the generator count). ``delta``
      maps to ``|W| err``.
    - Conv2d: the same, with the fan-in of one output (input channels per
      group times kernel area) and ``|weight|`` convolved in the layer's own
      padding mode.
    - ReLU: ReLU is 1-Lipschitz, so ``delta`` stays within ``err``. The
      reach (``relu_zono_approx``) copies a stably active row and zeroes a
      stably inactive one exactly, by the outward bounds ``l, u`` of the
      zonotope (those of :func:`_zono_row_bounds`). A zeroed row keeps only
      ``max(u + err, 0)``, the most the real input can exceed 0 by. A row
      that is inactive in the real network too thus carries no error into
      the next layer; otherwise ``|W|`` multiplies it at every layer and the
      error grows geometrically with depth on rows that are exactly 0. A
      crossing row becomes ``lam z + mu + mu beta`` with ``lam = u / (u -
      l)`` and ``mu = (1 - lam) u / 2``, both rounded: ``lam (u - l)`` is
      within ``gamma_2 u`` of ``u`` and ``2 mu`` within ``gamma_2 u`` of
      ``(1 - lam) u``, so ``[0, 2 mu]`` misses ``ReLU(z) - lam z`` by at most
      ``2 gamma_2 u``, and forming the row rounds by at most
      ``gamma_2 (lam a + mu)`` plus half a subnormal per product. All of it
      stays under ``4 gamma_2 a`` plus ``m + 3`` subnormals, ``a =
      max(|l|, |u|)`` and ``m`` the generator count.
    - Flatten: the entries move, so the largest ``err`` bounds every row.

    Every term is multiplied by ``_SAFETY`` like the other rounding bounds.
    Other layers (BatchNorm, pooling, Sigmoid/Tanh, ...) and sets that are
    not float64 zonotopes return None.
    """
    if err is None or not sets_in or len(sets_in) != len(sets_out):
        return None
    if all(o is i for o, i in zip(sets_out, sets_in)):
        return err
    if not all(isinstance(s, Zono) and s.c.dtype == np.float64
               and s.V.dtype == np.float64
               for s in list(sets_in) + list(sets_out)):
        return None
    n_out = sets_out[0].dim
    if (any(s.dim != err.size for s in sets_in)
            or any(s.dim != n_out for s in sets_out)):
        return None

    n_gens = max(s.V.shape[1] for s in sets_in)
    if isinstance(module, nn.Linear):
        W = module.weight.detach().cpu().double().numpy()
        g = _SAFETY * _gamma(W.shape[1] + 1)
        x = _add_dir(err, _mul_dir(g, _zono_magnitude(sets_in), 1.0), 1.0)
        cb = _underflow_error(n_gens, _nonzero_products(W, x != 0))
        if module.bias is not None:
            b = np.abs(module.bias.detach().cpu().double().numpy())
            cb = _add_dir(_mul_dir(g, b, 1.0), cb, 1.0)
        out = outward_interval_bounds(cb, np.abs(W), x, x)[1]
    elif isinstance(module, nn.Conv2d):
        if not (isinstance(sets_in[0], ImageZono)
                and all(isinstance(s, ImageZono) for s in sets_out)):
            return None
        out = _conv2d_error(module, sets_in[0], sets_out[0], err,
                            _zono_magnitude(sets_in), n_gens)
    elif isinstance(module, nn.ReLU):
        if bounds_in is None:
            bounds_in = [_zono_row_bounds(s) for s in sets_in]
        out = np.zeros(err.size)
        for (lo, hi), s in zip(bounds_in, sets_in):
            inactive = hi <= 0
            crossing = ~inactive & ~(lo >= 0)
            a = np.maximum(np.abs(lo), np.abs(hi))
            step = _SAFETY * (4.0 * _gamma(2) * a
                              + (s.V.shape[1] + 3) * _ETA)
            # A zeroed row computes exactly 0 and its real value is
            # ReLU(z + delta) with z <= hi: at most max(hi + err, 0), which
            # is 0 when the row is inactive in the real network too.
            e = np.where(inactive, np.maximum(_add_dir(hi, err, 1.0), 0.0),
                         _add_dir(err, np.where(crossing, step, 0.0), 1.0))
            out = np.maximum(out, e)
    elif (isinstance(module, nn.Flatten)
          or type(module).__name__ == 'OnnxFlatten'):
        out = np.full(n_out, err.max() if err.size else 0.0)
    else:
        return None
    return out if out.shape == (n_out,) else None


def _underflow_error(n_gens: int, count: np.ndarray) -> np.ndarray:
    """Per row, ``_SAFETY`` times the underflow an affine map of a zonotope
    with ``n_gens`` generators can add beyond Higham's relative bound: the
    row is ``n_gens + 1`` inner products (center and generators), each
    product with two nonzero factors can land in the subnormal range with an
    absolute error of half a subnormal, and ``count`` bounds the number of
    such products per inner product. Exact (integer multiples of ``_ETA``);
    0 on a row with no such product."""
    return _SAFETY * (n_gens + 1) * np.asarray(count, dtype=np.float64) * _ETA


def _nonzero_products(W: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Per row of ``W``, the number of nonzero weights on the ``live``
    columns (the input rows that are not identically zero)."""
    count = np.zeros(W.shape[0])
    if live.any():
        for sl, blk in _row_blocks(W, int(live.sum())):
            count[sl] = np.count_nonzero(blk[:, live], axis=1)
    return count


def _conv2d_error(layer: nn.Conv2d, z_in: ImageZono, z_out: ImageZono,
                  err: np.ndarray, mag: np.ndarray,
                  n_gens: int) -> Optional[np.ndarray]:
    """Conv2d case of :func:`_zono_error_step`: ``|weight|`` convolved with
    ``err + g mag``, plus ``g |bias|`` and the underflow term, rounded up
    (the rows are the ImageZono's (H, W, C) order, the convolution's
    (C, H, W)). ``n_gens`` is the largest generator count of the input
    sets."""
    from n2v.nn.layer_ops.conv2d_reach import float64_params
    weight, bias = float64_params(layer)
    fan_in = int(np.prod(weight.shape[1:]))
    g = _SAFETY * _gamma(fan_in + 1)
    x = _add_dir(err, _mul_dir(g, mag, 1.0), 1.0)
    h, w, c = z_in.height, z_in.width, z_in.num_channels
    xt = torch.from_numpy(np.ascontiguousarray(
        x.reshape(h, w, c).transpose(2, 0, 1)))[None]
    with torch.no_grad():
        s = layer._conv_forward(xt, weight.abs(), None)
        # Nonzero products per output. x is nonzero on every input row that
        # is not identically zero, so this also bounds the products of the
        # reach's own convolution that can underflow.
        count = layer._conv_forward((xt != 0).double(),
                                    (weight != 0).double(), None)
    if tuple(s.shape[1:]) != (z_out.num_channels, z_out.height, z_out.width):
        return None
    s = s[0].permute(1, 2, 0).numpy().ravel()
    count = count[0].permute(1, 2, 0).numpy().ravel()
    # A sum of nonnegative terms: it is its own magnitude.
    hi = _widen_sum(s, s, s, count)[1]
    const = _underflow_error(n_gens, count)
    if bias is not None:
        gb = _mul_dir(g, bias.abs().numpy(), 1.0)
        const = _add_dir(np.tile(gb, z_out.height * z_out.width), const, 1.0)
    return _add_dir(hi, const, 1.0)


def _zono_sequential(model: nn.Module, zono_set: Union[Zono, ImageZono],
                     err: Optional[np.ndarray],
                     ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Zonotope pre-pass for Sequential models. ``err`` is the input's
    distance bound (see :func:`_zono_error_step`)."""
    from n2v.nn.layer_ops.dispatcher import reach_layer

    layer_bounds = {}
    current_sets = [zono_set]

    layers = list(model.children())
    if not layers:
        layers = [model]

    for i, layer in enumerate(layers):
        bounds = None
        if isinstance(layer, NONLINEAR_TYPES) and current_sets:
            bounds = [_zono_row_bounds(s) for s in current_sets]
            layer_bounds[i] = _zono_layer_bounds(bounds, err)

        next_sets = reach_layer(layer, current_sets, method='approx')
        err = _zono_error_step(layer, current_sets, next_sets, err, bounds)
        current_sets = next_sets

    return layer_bounds


def _zono_graphmodule(graph_module: Any, zono_set: Union[Zono, ImageZono],
                      err: Optional[np.ndarray],
                      input_shape: Optional[tuple] = None,
                      ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Zonotope pre-pass for GraphModule models.

    Runs the reach executor itself (``n2v.nn.reach._handle_graphmodule``,
    approx, on the zonotope) and records the bounds of the sets entering
    each nonlinear layer. The executor resolves every node's input from its
    own producer, folds constant subgraphs (e.g. a Reshape of a constant),
    broadcasts constants by the tensor shapes, and raises on a node it has
    no rule for, instead of computing on the sets of an unrelated node. If
    it raises, the pass stops there: the bounds recorded up to that node
    came from correctly propagated sets and are kept, and later layers get
    none (the reach computes its own).

    The executor reports each layer it runs through the dispatcher, with its
    input sets. The distance bound ``err`` (see :func:`_zono_error_step`) of
    such a layer's input is carried from its producer when the producer is
    the graph input or another reported layer. Any other producer (a graph
    op the executor runs itself, e.g. MatMul, Add or Reshape) leaves it
    unknown for everything downstream.
    """
    from n2v.nn.reach import _handle_graphmodule

    layer_bounds = {}
    nodes = {n.name: n for n in graph_module.graph.nodes}
    # node name -> (module, input sets, their err, their bounds or None)
    reported = {}
    stepped = {}    # node name -> err of its output sets

    def input_error(name, sets):
        args = nodes[name].args
        src = args[0] if args else None
        if getattr(src, 'op', None) == 'placeholder':
            return err if len(sets) == 1 and sets[0] is zono_set else None
        key = getattr(src, 'name', None)
        if key not in reported:
            return None
        if key not in stepped:
            module, sets_in, e, bounds = reported[key]
            stepped[key] = _zono_error_step(module, sets_in, sets, e, bounds)
        return stepped[key]

    def record(name, module, sets):
        e = input_error(name, sets)
        bounds = None
        if isinstance(module, NONLINEAR_TYPES) and sets:
            bounds = [_zono_row_bounds(s) for s in sets]
            layer_bounds[name] = _zono_layer_bounds(bounds, e)
        reported[name] = (module, sets, e, bounds)

    try:
        _handle_graphmodule(graph_module, [zono_set], 'approx',
                            input_shape=input_shape, _on_layer_input=record)
    except Exception as e:  # noqa: BLE001 - the pre-pass only saves LPs
        logger.debug(
            f'zonotope bound pre-pass stopped ({type(e).__name__}: {e}); '
            f'{len(layer_bounds)} layer(s) have bounds')
    return layer_bounds
