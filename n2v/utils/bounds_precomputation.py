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
from typing import Dict, Tuple, Union, Optional, Any

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
        # AvgPool is linear — same IBP as conv with uniform weights
        new_lb = nn.functional.avg_pool2d(lb_t, layer.kernel_size, layer.stride,
                                           layer.padding, layer.ceil_mode).squeeze(0)
        new_ub = nn.functional.avg_pool2d(ub_t, layer.kernel_size, layer.stride,
                                           layer.padding, layer.ceil_mode).squeeze(0)
    else:
        raise NotImplementedError(f"Pool IBP not implemented for {type(layer)}")

    return new_lb.numpy().flatten(), new_ub.numpy().flatten(), new_lb.shape


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

    return new_lb.flatten(), new_ub.flatten(), new_lb.shape


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
        # Record bounds BEFORE nonlinear layers
        if isinstance(layer, NONLINEAR_TYPES):
            layer_bounds[i] = (lb.reshape(-1, 1), ub.reshape(-1, 1))

        # Propagate
        lb, ub, spatial_shape = _ibp_propagate_layer(layer, lb, ub, spatial_shape)

    return layer_bounds


def _get_param_np(graph_module: Any, node: Any) -> np.ndarray:
    """Extract a parameter tensor (get_attr or OnnxConstant) as a numpy array."""
    if getattr(node, 'op', None) == 'call_module':
        module = dict(graph_module.named_modules()).get(node.target)
        if module is not None and hasattr(module, 'value'):
            return module.value.detach().cpu().numpy()
    obj = graph_module
    for attr in node.target.split('.'):
        obj = getattr(obj, attr)
    return obj.detach().cpu().numpy()


def _ibp_onnx_matmul(graph_module, node, node_bounds):
    """IBP through OnnxMatMul ``y = x @ W`` (W a frozen parameter)."""
    if len(node.args) != 2:
        return None
    a, w = node.args
    if not (hasattr(a, 'name') and a.name in node_bounds):
        return None
    if getattr(w, 'op', None) != 'get_attr':
        return None
    W = _get_param_np(graph_module, w)
    la, ua, _ = node_bounds[a.name]
    new_lb, new_ub = _ibp_linear(la.flatten(), ua.flatten(), W.T, None)
    return new_lb, new_ub, None


def _ibp_onnx_binary(graph_module, node, module, node_bounds):
    """IBP through OnnxBinaryMathOperation (Add/Sub/Mul/Div).

    Handles both the residual case (two computed branches) and the
    constant case (second operand a frozen parameter). Returns None when
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
        if 'add' in op:
            return la + lb2, ua + ub2, sa
        if 'sub' in op:
            return la - ub2, ua - lb2, sa
        if op == 'mul':
            p = np.stack([la * lb2, la * ub2, ua * lb2, ua * ub2])
            return p.min(0), p.max(0), sa
        return None

    if not a_known or getattr(b, 'op', None) != 'get_attr':
        return None
    param = _get_param_np(graph_module, b).flatten()
    la, ua, sa = node_bounds[a.name]
    la, ua = la.flatten(), ua.flatten()
    if param.size not in (1, la.size):
        return None  # non-broadcastable (e.g. per-channel on flat image)
    if op == 'mul':
        p = np.stack([la * param, ua * param])
        return p.min(0), p.max(0), sa
    if op == '_onnx_div':
        inv = 1.0 / param
        p = np.stack([la * inv, ua * inv])
        return p.min(0), p.max(0), sa
    if 'add' in op:
        return la + param, ua + param, sa
    if 'sub' in op:
        return la - param, ua - param, sa
    return None


def _ibp_graphmodule(
    graph_module: Any,
    lb: np.ndarray,
    ub: np.ndarray,
    spatial_shape: Optional[tuple] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """IBP for GraphModule (ONNX-converted) models.

    Tracks, per node, whether its bounds are *trusted* (every op on the
    path was soundly propagated). Bounds are recorded for a nonlinear
    layer only when its input is trusted — so an unhandled op makes the
    pre-pass skip downstream layers (reach falls back to LP for them)
    rather than attach stale / wrong-dimension bounds.
    """
    import operator

    named_modules = dict(graph_module.named_modules())
    node_bounds = {}    # node_name -> (lb, ub, spatial_shape)
    node_trusted = {}   # node_name -> bool
    layer_bounds = {}

    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            node_bounds[node.name] = (lb, ub, spatial_shape)
            node_trusted[node.name] = True

        elif node.op == 'get_attr':
            pass

        elif node.op == 'call_module':
            module = named_modules.get(node.target)
            if module is None:
                continue

            in_name = (node.args[0].name
                       if node.args and hasattr(node.args[0], 'name')
                       else None)
            cur_lb, cur_ub, cur_shape = node_bounds.get(
                in_name, (lb, ub, spatial_shape))
            cur_trusted = node_trusted.get(in_name, True)

            # Record bounds before nonlinear layers (only if trusted).
            if isinstance(module, NONLINEAR_TYPES) and cur_trusted:
                layer_bounds[node.name] = (
                    cur_lb.reshape(-1, 1), cur_ub.reshape(-1, 1))

            # Propagate. Try ONNX graph ops first, then standard layers.
            module_type = type(module).__name__
            result = None
            if module_type == 'OnnxMatMul':
                result = _ibp_onnx_matmul(graph_module, node, node_bounds)
            elif module_type == 'OnnxBinaryMathOperation':
                result = _ibp_onnx_binary(
                    graph_module, node, module, node_bounds)

            if result is not None:
                new_lb, new_ub, new_shape = result
                node_bounds[node.name] = (new_lb, new_ub, new_shape)
                node_trusted[node.name] = cur_trusted
            else:
                try:
                    new_lb, new_ub, new_shape = _ibp_propagate_layer(
                        module, cur_lb, cur_ub, cur_shape)
                    node_bounds[node.name] = (new_lb, new_ub, new_shape)
                    node_trusted[node.name] = cur_trusted
                except Exception:  # noqa: BLE001 - unknown op: stop trusting
                    node_bounds[node.name] = (cur_lb, cur_ub, cur_shape)
                    node_trusted[node.name] = False

        elif node.op == 'call_function':
            if node.target is operator.getitem:
                src_node = node.args[0]
                if hasattr(src_node, 'name') and src_node.name in node_bounds:
                    node_bounds[node.name] = node_bounds[src_node.name]
                    node_trusted[node.name] = node_trusted.get(
                        src_node.name, False)

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

def _compute_bounds_zono(
    model: nn.Module,
    input_set: Union[Star, Zono, Box, ImageStar, ImageZono],
) -> Dict[Union[int, str], Tuple[np.ndarray, np.ndarray]]:
    """Compute bounds using Zonotope propagation."""
    import torch.fx as fx

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

    from onnx2torch.node_converters.reshape import OnnxReshape
    from onnx2torch.node_converters.concat import OnnxConcat
    from onnx2torch.node_converters.slice import OnnxSlice, OnnxSliceV9
    from onnx2torch.node_converters.split import OnnxSplit, OnnxSplit13

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

            if isinstance(module, OnnxReshape):
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

            if isinstance(module, OnnxConcat):
                result_sets = _handle_onnx_concat(module, node, node_values)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            if isinstance(module, (OnnxSlice, OnnxSliceV9)):
                result_sets = _handle_onnx_slice(module, node, node_values, graph_module)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            if isinstance(module, (OnnxSplit, OnnxSplit13)):
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
