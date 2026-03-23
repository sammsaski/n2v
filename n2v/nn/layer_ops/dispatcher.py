"""
Layer reachability dispatcher - routes computation based on layer type and set type.

Dispatches reachability computation for a single layer based on PyTorch layer type
and input set type, without requiring custom layer wrapper classes.
"""

import warnings
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union

# Import set types
from n2v.sets import Star, Zono, Box, Hexatope, Octatope
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

# Import layer-specific reach functions
from . import linear_reach, relu_reach, conv2d_reach, flatten_reach
from . import maxpool2d_reach, avgpool2d_reach, global_avgpool_reach
from . import batchnorm_reach
from . import pad_reach
from .pad_reach import _PAD_TYPES
from . import reduce_reach
from . import leakyrelu_reach
from . import sigmoid_reach
from . import tanh_reach
from . import conv1d_reach
from . import upsample_reach
from . import sign_reach

# ONNX GlobalAveragePool types (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.global_average_pool import (
        OnnxGlobalAveragePool,
        OnnxGlobalAveragePoolWithKnownInputShape,
    )
    _ONNX_GAP_TYPES = (nn.AdaptiveAvgPool2d, OnnxGlobalAveragePool, OnnxGlobalAveragePoolWithKnownInputShape)
except ImportError:
    _ONNX_GAP_TYPES = (nn.AdaptiveAvgPool2d,)

# ONNX Reduce types (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.reduce import (
        OnnxReduceStaticAxes,
        OnnxReduceSumStaticAxes,
    )
    _ONNX_REDUCE_TYPES = (OnnxReduceStaticAxes, OnnxReduceSumStaticAxes)
except ImportError:
    _ONNX_REDUCE_TYPES = ()

# ONNX Resize types (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.resize import OnnxResize
    _ONNX_RESIZE_TYPES = (nn.Upsample, OnnxResize)
except ImportError:
    _ONNX_RESIZE_TYPES = (nn.Upsample,)

# ONNX Neg type (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.neg import OnnxNeg
    _ONNX_NEG_TYPES = (OnnxNeg,)
except ImportError:
    _ONNX_NEG_TYPES = ()

# ONNX Cast type (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.cast import OnnxCast
    _ONNX_CAST_TYPES = (OnnxCast,)
except ImportError:
    _ONNX_CAST_TYPES = ()

# ONNX Function type (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.functions import OnnxFunction
    _ONNX_FUNCTION_TYPES = (OnnxFunction,)
except ImportError:
    _ONNX_FUNCTION_TYPES = ()

# ONNX Transpose type (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.transpose import OnnxTranspose
    _ONNX_TRANSPOSE_TYPES = (OnnxTranspose,)
except ImportError:
    _ONNX_TRANSPOSE_TYPES = ()

# ONNX Flatten type (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.flatten import OnnxFlatten
    _ONNX_FLATTEN_TYPES = (nn.Flatten, OnnxFlatten)
except ImportError:
    _ONNX_FLATTEN_TYPES = (nn.Flatten,)


def reach_layer(
    layer: nn.Module,
    input_sets: List,
    method: str = 'exact',
    **kwargs
) -> List:
    """
    Compute reachable sets through a PyTorch layer.

    Automatically detects the input set type and dispatches to the appropriate
    layer-specific implementation.

    Args:
        layer: PyTorch layer (nn.Linear, nn.ReLU, nn.Conv2d, etc.)
        input_sets: List of input sets (Star, Zono, Box, Hexatope, or Octatope)
        method: 'exact' or 'approx' (not all combinations supported)
        **kwargs: Additional options:
            - lp_solver: LP solver to use
            - verbose: Display option
            - parallel: Enable parallel processing
            - n_workers: Number of workers
            - relax_factor: Relaxation factor for approx methods
            - relax_method: Relaxation strategy

    Returns:
        List of output sets (same type as input)

    Raises:
        NotImplementedError: If layer/set combination is not supported
    """

    if not input_sets:
        return []

    # Detect set type from first input
    first_set = input_sets[0]

    # Route based on set type (including ImageStar/ImageZono as subclasses)
    if isinstance(first_set, (Star, ImageStar)):
        return _reach_layer_star(layer, input_sets, method, **kwargs)
    elif isinstance(first_set, (Zono, ImageZono)):
        return _reach_layer_zono(layer, input_sets, method, **kwargs)
    elif isinstance(first_set, Box):
        return _reach_layer_box(layer, input_sets, method, **kwargs)
    elif isinstance(first_set, Hexatope):
        return _reach_layer_hexatope(layer, input_sets, method, **kwargs)
    elif isinstance(first_set, Octatope):
        return _reach_layer_octatope(layer, input_sets, method, **kwargs)
    else:
        raise TypeError(
            f"Unsupported set type: {type(first_set).__name__}. "
            f"Supported: Star, ImageStar, Zono, ImageZono, Box, Hexatope, Octatope"
        )


def _reach_layer_star(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Star set reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_star(layer, input_sets)

    elif isinstance(layer, nn.ReLU):
        lp_solver = kwargs.get('lp_solver', 'default')
        verbose = kwargs.get('verbose', False)
        parallel = kwargs.get('parallel', None)
        n_workers = kwargs.get('n_workers', None)
        precomputed_bounds = kwargs.get('precomputed_bounds', None)

        if method == 'exact':
            return relu_reach.relu_star_exact(
                input_sets, lp_solver=lp_solver, verbose=verbose,
                parallel=parallel, n_workers=n_workers,
                precomputed_bounds=precomputed_bounds,
            )
        else:  # approx
            relax_factor = kwargs.get('relax_factor', 0.5)
            relax_method = kwargs.get('relax_method', 'standard')
            return relu_reach.relu_star_approx(
                input_sets, relax_factor, lp_solver, relax_method,
                precomputed_bounds=precomputed_bounds,
            )

    elif isinstance(layer, nn.LeakyReLU):
        gamma = layer.negative_slope
        lp_solver = kwargs.get('lp_solver', 'default')
        verbose = kwargs.get('verbose', False)
        precomputed_bounds = kwargs.get('precomputed_bounds', None)
        if method == 'exact':
            return leakyrelu_reach.leakyrelu_star_exact(
                input_sets, gamma=gamma, lp_solver=lp_solver, verbose=verbose,
                precomputed_bounds=precomputed_bounds,
            )
        else:
            return leakyrelu_reach.leakyrelu_star_approx(
                input_sets, gamma=gamma, lp_solver=lp_solver,
                precomputed_bounds=precomputed_bounds,
            )

    elif isinstance(layer, nn.Sigmoid):
        lp_solver = kwargs.get('lp_solver', 'default')
        if method == 'exact':
            warnings.warn("Sigmoid does not support exact method; using approx.")
        return sigmoid_reach.sigmoid_star_approx(input_sets, lp_solver=lp_solver)

    elif isinstance(layer, nn.Tanh):
        lp_solver = kwargs.get('lp_solver', 'default')
        if method == 'exact':
            warnings.warn("Tanh does not support exact method; using approx.")
        return tanh_reach.tanh_star_approx(input_sets, lp_solver=lp_solver)

    elif isinstance(layer, nn.Conv2d):
        return conv2d_reach.conv2d_star(layer, input_sets, method=method, **kwargs)

    elif isinstance(layer, nn.Conv1d):
        return conv1d_reach.conv1d_star(layer, input_sets, **kwargs)

    elif isinstance(layer, nn.MaxPool2d):
        lp_solver = kwargs.get('lp_solver', 'default')
        verbose = kwargs.get('verbose', False)
        return maxpool2d_reach.maxpool2d_star(
            layer, input_sets, method=method, lp_solver=lp_solver,
            verbose=verbose, **kwargs
        )

    elif isinstance(layer, nn.AvgPool2d):
        return avgpool2d_reach.avgpool2d_star(layer, input_sets, **kwargs)

    elif isinstance(layer, _ONNX_GAP_TYPES):
        return global_avgpool_reach.global_avgpool_star(input_sets)

    elif isinstance(layer, _ONNX_FLATTEN_TYPES):
        return flatten_reach.flatten_star(layer, input_sets)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return batchnorm_reach.batchnorm_star(layer, input_sets)

    elif isinstance(layer, _PAD_TYPES):
        return pad_reach.pad_star(layer, input_sets)

    elif _ONNX_REDUCE_TYPES and isinstance(layer, _ONNX_REDUCE_TYPES):
        return reduce_reach.reduce_star(layer, input_sets)

    elif isinstance(layer, _ONNX_RESIZE_TYPES):
        return upsample_reach.upsample_star(layer, input_sets, **kwargs)

    elif _ONNX_NEG_TYPES and isinstance(layer, _ONNX_NEG_TYPES):
        return _neg_sets_star(input_sets)

    elif _ONNX_CAST_TYPES and isinstance(layer, _ONNX_CAST_TYPES):
        return input_sets

    elif _ONNX_TRANSPOSE_TYPES and isinstance(layer, _ONNX_TRANSPOSE_TYPES):
        return _transpose_sets_star(layer, input_sets)

    elif _is_sign_layer(layer):
        return sign_reach.sign_star(layer, input_sets, method, **kwargs)

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        # Recursively handle Sequential
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        raise NotImplementedError(
            f"Star reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_zono(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Zonotope reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_zono(layer, input_sets)

    elif isinstance(layer, nn.ReLU):
        return relu_reach.relu_zono_approx(input_sets)

    elif isinstance(layer, nn.LeakyReLU):
        return leakyrelu_reach.leakyrelu_zono_approx(input_sets, gamma=layer.negative_slope)

    elif isinstance(layer, nn.Sigmoid):
        return sigmoid_reach.sigmoid_zono_approx(input_sets)

    elif isinstance(layer, nn.Tanh):
        return tanh_reach.tanh_zono_approx(input_sets)

    elif isinstance(layer, nn.Conv2d):
        return conv2d_reach.conv2d_zono(layer, input_sets)

    elif isinstance(layer, nn.Conv1d):
        return conv1d_reach.conv1d_zono(layer, input_sets, **kwargs)

    elif isinstance(layer, nn.MaxPool2d):
        return maxpool2d_reach.maxpool2d_zono(layer, input_sets)

    elif isinstance(layer, nn.AvgPool2d):
        return avgpool2d_reach.avgpool2d_zono(layer, input_sets)

    elif isinstance(layer, _ONNX_GAP_TYPES):
        return global_avgpool_reach.global_avgpool_zono(input_sets)

    elif isinstance(layer, _ONNX_FLATTEN_TYPES):
        return flatten_reach.flatten_zono(layer, input_sets)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return batchnorm_reach.batchnorm_zono(layer, input_sets)

    elif isinstance(layer, _PAD_TYPES):
        return pad_reach.pad_zono(layer, input_sets)

    elif _ONNX_REDUCE_TYPES and isinstance(layer, _ONNX_REDUCE_TYPES):
        return reduce_reach.reduce_zono(layer, input_sets)

    elif isinstance(layer, _ONNX_RESIZE_TYPES):
        return upsample_reach.upsample_zono(layer, input_sets, **kwargs)

    elif _ONNX_NEG_TYPES and isinstance(layer, _ONNX_NEG_TYPES):
        return _neg_sets_zono(input_sets)

    elif _ONNX_CAST_TYPES and isinstance(layer, _ONNX_CAST_TYPES):
        return input_sets

    elif _ONNX_TRANSPOSE_TYPES and isinstance(layer, _ONNX_TRANSPOSE_TYPES):
        return _transpose_sets_zono(layer, input_sets)

    elif _is_sign_layer(layer):
        return sign_reach.sign_zono(input_sets)

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        raise NotImplementedError(
            f"Zono reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_box(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Box reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_box(layer, input_sets)

    elif isinstance(layer, nn.ReLU):
        return relu_reach.relu_box(input_sets)

    elif isinstance(layer, nn.LeakyReLU):
        return leakyrelu_reach.leakyrelu_box(input_sets, gamma=layer.negative_slope)

    elif isinstance(layer, nn.Sigmoid):
        return sigmoid_reach.sigmoid_box(input_sets)

    elif isinstance(layer, nn.Tanh):
        return tanh_reach.tanh_box(input_sets)

    elif isinstance(layer, nn.Conv1d):
        return conv1d_reach.conv1d_box(layer, input_sets, **kwargs)

    elif isinstance(layer, _ONNX_FLATTEN_TYPES):
        return flatten_reach.flatten_box(layer, input_sets)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return batchnorm_reach.batchnorm_box(layer, input_sets)

    elif _ONNX_REDUCE_TYPES and isinstance(layer, _ONNX_REDUCE_TYPES):
        return reduce_reach.reduce_box(layer, input_sets)

    elif _ONNX_NEG_TYPES and isinstance(layer, _ONNX_NEG_TYPES):
        return _neg_sets_box(input_sets)

    elif _ONNX_CAST_TYPES and isinstance(layer, _ONNX_CAST_TYPES):
        return input_sets

    elif _ONNX_TRANSPOSE_TYPES and isinstance(layer, _ONNX_TRANSPOSE_TYPES):
        return _transpose_sets_box(layer, input_sets)

    elif _is_sign_layer(layer):
        return sign_reach.sign_box(input_sets)

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        raise NotImplementedError(
            f"Box reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_hexatope(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Hexatope reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_hexatope(layer, input_sets)

    elif isinstance(layer, nn.ReLU):
        verbose = kwargs.get('verbose', False)
        solver = kwargs.get('solver', None)
        return relu_reach.relu_hexatope_approx(input_sets, verbose=verbose, solver=solver)

    elif isinstance(layer, _ONNX_FLATTEN_TYPES):
        return flatten_reach.flatten_hexatope(layer, input_sets)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        scale, shift = batchnorm_reach._get_bn_params(layer)
        dummy = nn.Linear(len(scale), len(scale), bias=True)
        with torch.no_grad():
            dummy.weight.copy_(torch.from_numpy(np.diag(scale)).float())
            dummy.bias.copy_(torch.from_numpy(shift).float())
        return linear_reach.linear_hexatope(dummy, input_sets)

    elif _ONNX_NEG_TYPES and isinstance(layer, _ONNX_NEG_TYPES):
        return _neg_sets_hexatope(input_sets)

    elif _ONNX_CAST_TYPES and isinstance(layer, _ONNX_CAST_TYPES):
        return input_sets

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        raise NotImplementedError(
            f"Hexatope reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_octatope(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Octatope reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_octatope(layer, input_sets)

    elif isinstance(layer, nn.ReLU):
        verbose = kwargs.get('verbose', False)
        solver = kwargs.get('solver', None)
        return relu_reach.relu_octatope_approx(input_sets, verbose=verbose, solver=solver)

    elif isinstance(layer, _ONNX_FLATTEN_TYPES):
        return flatten_reach.flatten_octatope(layer, input_sets)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        scale, shift = batchnorm_reach._get_bn_params(layer)
        dummy = nn.Linear(len(scale), len(scale), bias=True)
        with torch.no_grad():
            dummy.weight.copy_(torch.from_numpy(np.diag(scale)).float())
            dummy.bias.copy_(torch.from_numpy(shift).float())
        return linear_reach.linear_octatope(dummy, input_sets)

    elif _ONNX_NEG_TYPES and isinstance(layer, _ONNX_NEG_TYPES):
        return _neg_sets_octatope(input_sets)

    elif _ONNX_CAST_TYPES and isinstance(layer, _ONNX_CAST_TYPES):
        return input_sets

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        raise NotImplementedError(
            f"Octatope reachability not implemented for layer type: {type(layer).__name__}"
        )


# ===========================================================================
# OnnxFunction helpers — detect Sign activation
# ===========================================================================

def _is_sign_layer(layer: nn.Module) -> bool:
    """Check if layer is a Sign activation (OnnxFunction wrapping torch.sign)."""
    if _ONNX_FUNCTION_TYPES and isinstance(layer, _ONNX_FUNCTION_TYPES):
        return getattr(layer, 'function', None) is torch.sign
    return False


# ===========================================================================
# OnnxNeg helpers — negate sets (multiply by -1)
# ===========================================================================

def _neg_sets_star(input_sets: List) -> List:
    """Negate Star/ImageStar sets: multiply V matrix by -1."""
    output = []
    for s in input_sets:
        if isinstance(s, ImageStar):
            new_V = -s.V
            output.append(ImageStar(
                new_V, s.C, s.d, s.predicate_lb, s.predicate_ub,
                s.height, s.width, s.num_channels
            ))
        else:
            new_V = -s.V
            output.append(Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub))
    return output


def _neg_sets_zono(input_sets: List) -> List:
    """Negate Zono/ImageZono sets: negate center and generators."""
    output = []
    for s in input_sets:
        if isinstance(s, ImageZono):
            output.append(ImageZono(-s.c, -s.V, s.height, s.width, s.num_channels))
        else:
            output.append(Zono(-s.c, -s.V))
    return output


def _neg_sets_box(input_sets: List) -> List:
    """Negate Box sets: swap and negate bounds."""
    output = []
    for s in input_sets:
        output.append(Box(-s.ub, -s.lb))
    return output


def _neg_sets_hexatope(input_sets: List) -> List:
    """Negate Hexatope sets via affine map with -I."""
    import torch
    output = []
    for s in input_sets:
        dummy = nn.Linear(s.dim, s.dim, bias=False)
        with torch.no_grad():
            dummy.weight.copy_(torch.from_numpy(-np.eye(s.dim)).float())
        result = linear_reach.linear_hexatope(dummy, [s])
        output.extend(result)
    return output


def _neg_sets_octatope(input_sets: List) -> List:
    """Negate Octatope sets via affine map with -I."""
    import torch
    output = []
    for s in input_sets:
        dummy = nn.Linear(s.dim, s.dim, bias=False)
        with torch.no_grad():
            dummy.weight.copy_(torch.from_numpy(-np.eye(s.dim)).float())
        result = linear_reach.linear_octatope(dummy, [s])
        output.extend(result)
    return output


# ===========================================================================
# OnnxTranspose helpers — permute dimensions of sets
# ===========================================================================

def _transpose_sets_star(layer: nn.Module, input_sets: List) -> List:
    """Permute rows of Star V matrix."""
    perm = layer.perm
    output_sets = []
    for s in input_sets:
        new_V = s.V[perm, :]
        output_sets.append(Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub))
    return output_sets


def _transpose_sets_zono(layer: nn.Module, input_sets: List) -> List:
    """Permute rows of Zono center and generators."""
    perm = layer.perm
    output_sets = []
    for s in input_sets:
        new_c = s.c[perm, :]
        new_V = s.V[perm, :]
        output_sets.append(Zono(new_c, new_V))
    return output_sets


def _transpose_sets_box(layer: nn.Module, input_sets: List) -> List:
    """Permute rows of Box bounds."""
    perm = layer.perm
    output_sets = []
    for s in input_sets:
        output_sets.append(Box(s.lb[perm, :], s.ub[perm, :]))
    return output_sets
