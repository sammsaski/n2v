"""
Pre-pass for computing intermediate bounds before nonlinear layers.

Two strategies:
- 'ibp': Interval Bound Propagation using PyTorch forward passes. O(n) per layer,
         works for any model. Fast but loose bounds.
- 'zono': Zonotope propagation. O(n*g) per layer where g = generators.
          Tighter bounds but expensive for high-dimensional inputs.

These bounds are passed to Star reachability to skip LP calls for stable neurons
(provably always-active or always-inactive).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional, Any

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono


# Layer types that are nonlinear and benefit from pre-computed bounds
NONLINEAR_TYPES = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)


def compute_intermediate_bounds(
    model: nn.Module,
    input_set: Union[Star, Zono, Box, ImageStar, ImageZono],
    method: str = 'ibp',
) -> Dict[Union[int, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Compute bounds before each nonlinear layer via a fast pre-pass.

    Args:
        model: PyTorch model (Sequential or GraphModule).
        input_set: Input set (Star, Zono, Box, ImageStar, or ImageZono).
        method: 'ibp' (interval bound propagation) or 'zono' (zonotope).

    Returns:
        Dictionary mapping layer_id -> (lb, ub) numpy arrays.
        lb and ub have shape (dim, 1).
    """
    if method == 'ibp':
        return _compute_bounds_ibp(model, input_set)
    elif method == 'zono':
        return _compute_bounds_zono(model, input_set)
    else:
        raise ValueError(f"Unknown precompute method: {method!r}. Use 'ibp' or 'zono'.")


# ============================================================================
# IBP (Interval Bound Propagation) — fast, works for any model
# ============================================================================

def _compute_bounds_ibp(model: nn.Module, input_set: Union[Star, Zono, Box, ImageStar, ImageZono]) -> Dict[Union[int, str], Tuple[np.ndarray, np.ndarray]]:
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


def _extract_bounds(input_set: Union[Star, Zono, Box, ImageStar, ImageZono]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (lb, ub) numpy arrays from any input set type."""
    if isinstance(input_set, Box):
        return input_set.lb.flatten(), input_set.ub.flatten()
    elif isinstance(input_set, (Star, ImageStar)):
        lb, ub = input_set.estimate_ranges()
        return lb.flatten(), ub.flatten()
    elif isinstance(input_set, (Zono, ImageZono)):
        lb, ub = input_set.get_bounds()
        return lb.flatten(), ub.flatten()
    else:
        raise TypeError(f"Unsupported input_set type: {type(input_set).__name__}")


def _ibp_linear(lb: np.ndarray, ub: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """IBP through a linear layer: y = W*x + b."""
    W_pos = np.maximum(weight, 0)
    W_neg = np.minimum(weight, 0)
    new_lb = W_pos @ lb + W_neg @ ub
    new_ub = W_pos @ ub + W_neg @ lb
    if bias is not None:
        new_lb = new_lb + bias
        new_ub = new_ub + bias
    return new_lb, new_ub


def _ibp_conv(lb: np.ndarray, ub: np.ndarray, layer: Union[nn.Conv2d, nn.Conv1d], input_shape: tuple) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """IBP through Conv2d/Conv1d using PyTorch (handles padding, stride, dilation)."""
    # Reshape to NCHW/NCW
    lb_t = torch.tensor(lb.reshape(input_shape), dtype=torch.float64).unsqueeze(0)
    ub_t = torch.tensor(ub.reshape(input_shape), dtype=torch.float64).unsqueeze(0)

    weight = layer.weight.detach().double()
    W_pos = torch.clamp(weight, min=0)
    W_neg = torch.clamp(weight, max=0)

    # Use functional conv with split positive/negative weights
    conv_fn = torch.nn.functional.conv2d if isinstance(layer, nn.Conv2d) else torch.nn.functional.conv1d
    conv_kwargs = dict(stride=layer.stride, padding=layer.padding,
                       dilation=layer.dilation, groups=layer.groups)

    new_lb = conv_fn(lb_t, W_pos, None, **conv_kwargs) + conv_fn(ub_t, W_neg, None, **conv_kwargs)
    new_ub = conv_fn(ub_t, W_pos, None, **conv_kwargs) + conv_fn(lb_t, W_neg, None, **conv_kwargs)

    if layer.bias is not None:
        bias = layer.bias.detach().double()
        # Reshape bias for broadcasting: (1, C, 1, 1) or (1, C, 1)
        shape = [1, -1] + [1] * (lb_t.dim() - 2)
        new_lb = new_lb + bias.view(*shape)
        new_ub = new_ub + bias.view(*shape)

    return new_lb.squeeze(0).numpy().flatten(), new_ub.squeeze(0).numpy().flatten(), new_lb.squeeze(0).shape


def _ibp_relu(lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """IBP through ReLU."""
    return np.maximum(lb, 0), np.maximum(ub, 0)


def _ibp_leakyrelu(lb: np.ndarray, ub: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """IBP through LeakyReLU."""
    new_lb = np.where(lb >= 0, lb, gamma * lb)
    new_ub = np.where(ub >= 0, ub, gamma * ub)
    # Handle crossing: lb < 0, ub > 0
    crossing = (lb < 0) & (ub > 0)
    new_lb[crossing] = np.minimum(gamma * lb[crossing], 0)
    return new_lb, new_ub


def _ibp_sigmoid(lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """IBP through Sigmoid (monotone)."""
    from scipy.special import expit
    return expit(lb), expit(ub)


def _ibp_tanh(lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """IBP through Tanh (monotone)."""
    return np.tanh(lb), np.tanh(ub)


def _ibp_pool(lb: np.ndarray, ub: np.ndarray, layer: Union[nn.MaxPool2d, nn.AvgPool2d], spatial_shape: tuple) -> Tuple[np.ndarray, np.ndarray, tuple]:
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
        # AvgPool is linear — same IBP as conv with uniform weights
        new_lb = nn.functional.avg_pool2d(lb_t, layer.kernel_size, layer.stride,
                                           layer.padding, layer.ceil_mode).squeeze(0)
        new_ub = nn.functional.avg_pool2d(ub_t, layer.kernel_size, layer.stride,
                                           layer.padding, layer.ceil_mode).squeeze(0)
    else:
        raise NotImplementedError(f"Pool IBP not implemented for {type(layer)}")

    return new_lb.numpy().flatten(), new_ub.numpy().flatten(), new_lb.shape


def _ibp_batchnorm(lb: np.ndarray, ub: np.ndarray, layer: Union[nn.BatchNorm1d, nn.BatchNorm2d], spatial_shape: tuple) -> Tuple[np.ndarray, np.ndarray, tuple]:
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

    return new_lb.flatten(), new_ub.flatten(), new_lb.shape


def _ibp_sequential(model: nn.Module, lb: np.ndarray, ub: np.ndarray, spatial_shape: Optional[tuple] = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """IBP for Sequential models."""
    layer_bounds = {}

    layers = list(model.children())
    if not layers:
        layers = [model]

    for i, layer in enumerate(layers):
        # Record bounds BEFORE nonlinear layers
        if isinstance(layer, NONLINEAR_TYPES):
            layer_bounds[i] = (lb.reshape(-1, 1), ub.reshape(-1, 1))

        # Propagate
        lb, ub, spatial_shape = _ibp_propagate_layer(layer, lb, ub, spatial_shape)

    return layer_bounds


def _ibp_graphmodule(graph_module: Any, lb: np.ndarray, ub: np.ndarray, spatial_shape: Optional[tuple] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """IBP for GraphModule (ONNX-converted) models."""
    import operator

    named_modules = dict(graph_module.named_modules())
    node_bounds = {}  # node_name -> (lb, ub, spatial_shape)
    layer_bounds = {}

    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            node_bounds[node.name] = (lb, ub, spatial_shape)

        elif node.op == 'get_attr':
            pass

        elif node.op == 'call_module':
            module = named_modules.get(node.target)
            if module is None:
                continue

            # Get input bounds
            cur_lb, cur_ub, cur_shape = lb, ub, spatial_shape
            if node.args and hasattr(node.args[0], 'name') and node.args[0].name in node_bounds:
                cur_lb, cur_ub, cur_shape = node_bounds[node.args[0].name]

            # Record bounds before nonlinear layers
            if isinstance(module, NONLINEAR_TYPES):
                layer_bounds[node.name] = (cur_lb.reshape(-1, 1), cur_ub.reshape(-1, 1))

            # Propagate
            try:
                new_lb, new_ub, new_shape = _ibp_propagate_layer(module, cur_lb, cur_ub, cur_shape)
                node_bounds[node.name] = (new_lb, new_ub, new_shape)
                lb, ub, spatial_shape = new_lb, new_ub, new_shape
            except (NotImplementedError, Exception):
                # For unsupported layers, pass through with infinite bounds
                node_bounds[node.name] = (cur_lb, cur_ub, cur_shape)
                lb, ub, spatial_shape = cur_lb, cur_ub, cur_shape

        elif node.op == 'call_function':
            if node.target is operator.getitem:
                src_node = node.args[0]
                if hasattr(src_node, 'name') and src_node.name in node_bounds:
                    node_bounds[node.name] = node_bounds[src_node.name]

        elif node.op == 'output':
            pass

    return layer_bounds


def _ibp_propagate_layer(layer: nn.Module, lb: np.ndarray, ub: np.ndarray, spatial_shape: Optional[tuple]) -> Tuple[np.ndarray, np.ndarray, Optional[tuple]]:
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
        if spatial_shape is not None:
            new_lb, new_ub, out_shape = _ibp_pool(lb, ub, layer, spatial_shape)
            return new_lb, new_ub, tuple(out_shape)
        return lb, ub, spatial_shape

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
        if module_type in ('OnnxFlatten', 'OnnxTranspose', 'OnnxCast'):
            return lb, ub, None if module_type == 'OnnxFlatten' else spatial_shape
        elif module_type == 'OnnxNeg':
            return -ub, -lb, spatial_shape

        raise NotImplementedError(f"IBP not implemented for {type(layer).__name__}")


# ============================================================================
# Zonotope pre-pass — tighter bounds, expensive for large inputs
# ============================================================================

def _compute_bounds_zono(model: nn.Module, input_set: Union[Star, Zono, Box, ImageStar, ImageZono]) -> Dict[Union[int, str], Tuple[np.ndarray, np.ndarray]]:
    """Compute bounds using Zonotope propagation."""
    import torch.fx as fx
    from n2v.nn.layer_ops.dispatcher import reach_layer

    zono_set = _convert_to_zono(input_set)

    if isinstance(model, fx.GraphModule):
        return _zono_graphmodule(model, zono_set)
    else:
        return _zono_sequential(model, zono_set)


def _convert_to_zono(input_set: Union[Star, Zono, Box, ImageStar, ImageZono]) -> Union[Zono, ImageZono]:
    """Convert any supported input set to Zono or ImageZono."""
    if isinstance(input_set, ImageZono):
        return input_set
    elif isinstance(input_set, ImageStar):
        lb, ub = input_set.estimate_ranges()
        return ImageZono.from_bounds(
            lb.reshape(input_set.height, input_set.width, input_set.num_channels),
            ub.reshape(input_set.height, input_set.width, input_set.num_channels),
            input_set.height, input_set.width, input_set.num_channels,
        )
    elif isinstance(input_set, Zono):
        return input_set
    elif isinstance(input_set, Star):
        lb, ub = input_set.estimate_ranges()
        return Zono.from_bounds(lb, ub)
    elif isinstance(input_set, Box):
        return Zono.from_bounds(input_set.lb, input_set.ub)
    else:
        raise TypeError(f"Unsupported input_set type: {type(input_set).__name__}")


def _zono_sequential(model: nn.Module, zono_set: Union[Zono, ImageZono]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Zonotope pre-pass for Sequential models."""
    from n2v.nn.layer_ops.dispatcher import reach_layer

    layer_bounds = {}
    current_sets = [zono_set]

    layers = list(model.children())
    if not layers:
        layers = [model]

    for i, layer in enumerate(layers):
        if isinstance(layer, NONLINEAR_TYPES):
            lb, ub = current_sets[0].estimate_ranges()
            layer_bounds[i] = (lb.reshape(-1, 1), ub.reshape(-1, 1))

        current_sets = reach_layer(layer, current_sets, method='approx')

    return layer_bounds


def _zono_graphmodule(graph_module: Any, zono_set: Union[Zono, ImageZono]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Zonotope pre-pass for GraphModule models."""
    import operator
    from n2v.nn.layer_ops.dispatcher import reach_layer
    from n2v.nn.reach import (
        _handle_reshape,
        _handle_onnx_concat,
        _handle_onnx_slice,
        _handle_onnx_split,
        _handle_onnx_binary_op,
        _handle_onnx_matmul,
        _get_parameter,
    )

    try:
        from onnx2torch.node_converters.reshape import OnnxReshape
    except ImportError:
        OnnxReshape = None
    try:
        from onnx2torch.node_converters.concat import OnnxConcat
    except ImportError:
        OnnxConcat = None
    try:
        from onnx2torch.node_converters.slice import OnnxSlice
    except ImportError:
        OnnxSlice = None
    try:
        from onnx2torch.node_converters.slice import OnnxSliceV9
    except ImportError:
        OnnxSliceV9 = None
    try:
        from onnx2torch.node_converters.split import OnnxSplit, OnnxSplit13
    except ImportError:
        OnnxSplit = None
        OnnxSplit13 = None

    named_modules = dict(graph_module.named_modules())
    node_values = {}
    current_sets = [zono_set]
    layer_bounds = {}

    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            node_values[node.name] = current_sets

        elif node.op == 'get_attr':
            pass

        elif node.op == 'call_module':
            module = named_modules.get(node.target)
            if module is None:
                continue

            module_type = type(module).__name__

            if OnnxReshape is not None and isinstance(module, OnnxReshape):
                first_arg = node.args[0]
                if hasattr(first_arg, 'name') and first_arg.name in node_values:
                    input_sets_op = node_values[first_arg.name]
                else:
                    input_sets_op = current_sets
                shape_node = node.args[1]
                shape_tensor = _get_parameter(graph_module, shape_node)
                target_shape = tuple(shape_tensor.numpy().astype(int))
                result_sets = _handle_reshape(input_sets_op, target_shape)
                node_values[node.name] = result_sets
                current_sets = result_sets
                continue

            if OnnxConcat is not None and isinstance(module, OnnxConcat):
                result_sets = _handle_onnx_concat(module, node, node_values)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            if ((OnnxSlice is not None and isinstance(module, OnnxSlice)) or
                    (OnnxSliceV9 is not None and isinstance(module, OnnxSliceV9))):
                result_sets = _handle_onnx_slice(module, node, node_values, graph_module)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            if ((OnnxSplit is not None and isinstance(module, OnnxSplit)) or
                    (OnnxSplit13 is not None and isinstance(module, OnnxSplit13))):
                result = _handle_onnx_split(module, node, node_values, graph_module)
                if result is not None:
                    node_values[node.name] = result
                    continue

            if module_type == 'OnnxBinaryMathOperation':
                set_type = type(current_sets[0])
                result = _handle_onnx_binary_op(
                    module, node, node_values, graph_module, set_type
                )
                if result is not None:
                    node_values[node.name] = result
                    current_sets = result
                    continue

            elif module_type == 'OnnxMatMul':
                set_type = type(current_sets[0])
                result = _handle_onnx_matmul(
                    module, node, node_values, graph_module, set_type
                )
                if result is not None:
                    node_values[node.name] = result
                    current_sets = result
                    continue

            # Standard layer
            if node.args and len(node.args) > 0:
                first_arg = node.args[0]
                if hasattr(first_arg, 'name') and first_arg.name in node_values:
                    input_sets_op = node_values[first_arg.name]

                    if isinstance(module, NONLINEAR_TYPES):
                        lb, ub = input_sets_op[0].estimate_ranges()
                        layer_bounds[node.name] = (lb.reshape(-1, 1), ub.reshape(-1, 1))

                    output_sets = reach_layer(module, input_sets_op, method='approx')
                    node_values[node.name] = output_sets
                    current_sets = output_sets

        elif node.op == 'call_function':
            if node.target is operator.getitem:
                args = node.args
                if len(args) >= 2:
                    src_node = args[0]
                    index = args[1]
                    if hasattr(src_node, 'name') and src_node.name in node_values:
                        src_val = node_values[src_node.name]
                        if (isinstance(src_val, list) and len(src_val) > 0
                                and isinstance(src_val[0], list)):
                            node_values[node.name] = src_val[index]
                            current_sets = src_val[index]

        elif node.op == 'output':
            if node.args and len(node.args) > 0:
                output_node = node.args[0]
                if hasattr(output_node, 'name') and output_node.name in node_values:
                    current_sets = node_values[output_node.name]

    return layer_bounds
