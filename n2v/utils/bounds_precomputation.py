"""
Zonotope pre-pass for computing intermediate bounds before nonlinear layers.

Propagates a Zonotope (fast, no LP) through the network to identify provably
stable neurons (always active or always inactive) at each nonlinear layer.
These bounds can be passed to Star reachability functions to skip LP calls
for stable neurons.
"""

import numpy as np
import torch.nn as nn
from typing import Dict, Tuple, Union

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops.dispatcher import reach_layer


# Layer types that are nonlinear and benefit from pre-computed bounds
NONLINEAR_TYPES = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)


def compute_intermediate_bounds(
    model: nn.Module,
    input_set: Union[Star, Zono, Box, ImageStar, ImageZono],
) -> Dict[Union[int, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Run a fast Zonotope pre-pass to compute bounds before each nonlinear layer.

    Converts the input set to a Zono/ImageZono and propagates through the model
    using approx method. Before each nonlinear layer (ReLU, LeakyReLU, Sigmoid,
    Tanh), stores the estimate_ranges() bounds.

    Args:
        model: PyTorch model (nn.Sequential or similar).
            For GraphModule, use compute_intermediate_bounds_graph().
        input_set: Input set (Star, Zono, Box, ImageStar, or ImageZono).

    Returns:
        Dictionary mapping layer_id -> (lb, ub) numpy arrays.
        For Sequential models, layer_id is the integer layer index.
        lb and ub have shape (dim, 1).
    """
    import torch.fx as fx

    zono_set = _convert_to_zono(input_set)

    if isinstance(model, fx.GraphModule):
        return _compute_bounds_graphmodule(model, zono_set)
    else:
        return _compute_bounds_sequential(model, zono_set)


def _convert_to_zono(input_set):
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


def _compute_bounds_sequential(model, zono_set):
    """Compute bounds for Sequential models by iterating layers."""
    layer_bounds = {}
    current_sets = [zono_set]

    layers = list(model.children())
    if not layers:
        layers = [model]

    for i, layer in enumerate(layers):
        if isinstance(layer, NONLINEAR_TYPES):
            # Store bounds BEFORE the nonlinearity
            lb, ub = current_sets[0].estimate_ranges()
            layer_bounds[i] = (lb.reshape(-1, 1), ub.reshape(-1, 1))

        # Propagate through layer using Zono approx
        current_sets = reach_layer(layer, current_sets, method='approx')

    return layer_bounds


def _compute_bounds_graphmodule(graph_module, zono_set):
    """Compute bounds for GraphModule models by walking the FX graph."""
    import operator
    import torch.fx as fx

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

    # Reuse the existing graph handler but with Zono sets, recording bounds
    from n2v.nn.reach import (
        _handle_reshape,
        _handle_onnx_concat,
        _handle_onnx_slice,
        _handle_onnx_split,
        _handle_onnx_binary_op,
        _handle_onnx_matmul,
        _get_parameter,
    )

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

            # Handle OnnxReshape
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

            # Handle OnnxConcat
            if OnnxConcat is not None and isinstance(module, OnnxConcat):
                result_sets = _handle_onnx_concat(module, node, node_values)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            # Handle OnnxSlice
            if ((OnnxSlice is not None and isinstance(module, OnnxSlice)) or
                    (OnnxSliceV9 is not None and isinstance(module, OnnxSliceV9))):
                result_sets = _handle_onnx_slice(module, node, node_values, graph_module)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            # Handle OnnxSplit
            if ((OnnxSplit is not None and isinstance(module, OnnxSplit)) or
                    (OnnxSplit13 is not None and isinstance(module, OnnxSplit13))):
                result = _handle_onnx_split(module, node, node_values, graph_module)
                if result is not None:
                    node_values[node.name] = result
                    continue

            # Handle ONNX binary ops
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

            # Standard layer: check if nonlinear, store bounds, then propagate
            if node.args and len(node.args) > 0:
                first_arg = node.args[0]
                if hasattr(first_arg, 'name') and first_arg.name in node_values:
                    input_sets_op = node_values[first_arg.name]

                    # Record bounds before nonlinear layers
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
