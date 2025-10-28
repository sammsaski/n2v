"""
Unified reachability analysis for neural networks.

This module provides the core reachability computation engine that routes
computation based on set type and handles both standard PyTorch models
and ONNX GraphModules.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Optional


def reach_pytorch_model(
    model: nn.Module,
    input_set: Union['Star', 'Zono', 'Box', 'Hexatope', 'Octatope'],
    method: str = 'exact',
    **kwargs
) -> List:
    """
    Compute reachability through a PyTorch model.

    This is the main dispatcher that routes to appropriate handlers based on
    the input set type and model structure.

    Args:
        model: PyTorch model (nn.Module or torch.fx.GraphModule)
        input_set: Input specification (Star, Zono, Box, Hexatope, or Octatope)
        method: Reachability method:
            - 'exact': Exact reachability (Star, Hexatope, Octatope)
            - 'exact-differentiable': Exact with differentiable solver (Hexatope, Octatope)
            - 'approx': Over-approximate reachability (all set types)
        **kwargs: Method-specific arguments:
            - lp_solver: LP solver to use
            - dis_opt: 'display' to show progress
            - parallel: Enable parallel processing (Star)
            - n_workers: Number of parallel workers
            - relax_factor: Relaxation factor for approx methods
            - relax_method: Relaxation strategy

    Returns:
        List of output sets (same type as input)

    Raises:
        TypeError: If input_set type is not supported
        ValueError: If method is not valid for the given set type
    """
    from n2v.sets import Star, Zono, Box, Hexatope, Octatope

    # Validate method for set type
    if isinstance(input_set, Star):
        if method not in ('exact', 'approx'):
            raise ValueError(f"Star supports 'exact' or 'approx', got '{method}'")
    elif isinstance(input_set, Zono):
        if method != 'approx':
            raise ValueError(f"Zono only supports 'approx', got '{method}'")
    elif isinstance(input_set, Box):
        if method != 'approx':
            raise ValueError(f"Box only supports 'approx', got '{method}'")
    elif isinstance(input_set, (Hexatope, Octatope)):
        if method not in ('exact', 'exact-differentiable', 'approx'):
            raise ValueError(
                f"{type(input_set).__name__} supports 'exact', 'exact-differentiable', or 'approx', got '{method}'"
            )
    else:
        raise TypeError(
            f"Unsupported input set type: {type(input_set).__name__}. "
            f"Supported types: Star, Zono, Box, Hexatope, Octatope"
        )

    # Check if model is a GraphModule (from torch.fx / onnx2torch)
    try:
        import torch.fx as fx
        if isinstance(model, fx.GraphModule):
            return _handle_graphmodule(model, [input_set], method, **kwargs)
    except ImportError:
        pass

    # Standard sequential model processing
    return _reach_sequential(model, [input_set], method, **kwargs)


def _reach_sequential(
    model: nn.Module,
    input_sets: List,
    method: str,
    **kwargs
) -> List:
    """
    Propagate sets through a sequential model layer by layer.

    Args:
        model: PyTorch model
        input_sets: List of input sets (all same type)
        method: Reachability method
        **kwargs: Additional arguments

    Returns:
        List of output sets
    """
    from n2v.nn.layer_ops.dispatcher import reach_layer

    current_sets = input_sets
    dis_opt = kwargs.get('dis_opt', None)

    # Get all layers from the model
    layers = list(model.children())
    if not layers:
        # Model might be a single layer
        layers = [model]

    # Propagate through each layer
    for i, layer in enumerate(layers):
        if dis_opt == 'display':
            set_type = type(current_sets[0]).__name__
            print(f'Layer {i+1}/{len(layers)}: {type(layer).__name__}')

        current_sets = reach_layer(layer, current_sets, method, **kwargs)

        if dis_opt == 'display':
            print(f'  Output: {len(current_sets)} {set_type} sets')

    return current_sets


def _handle_graphmodule(
    graph_module,
    input_sets: List,
    method: str,
    **kwargs
) -> List:
    """
    Handle reachability for torch.fx.GraphModule (e.g., from onnx2torch).

    Processes the computational graph node by node, handling ONNX operations
    and standard PyTorch layers.

    Args:
        graph_module: torch.fx.GraphModule to analyze
        input_sets: Input sets (all same type)
        method: Reachability method
        **kwargs: Additional arguments

    Returns:
        List of output sets
    """
    from n2v.nn.layer_ops.dispatcher import reach_layer
    from n2v.sets import Star, Hexatope, Octatope

    # Get the set type from the first input
    set_type = type(input_sets[0])

    # Get named modules
    named_modules = dict(graph_module.named_modules())

    # Store intermediate values for each node
    node_values = {}
    dis_opt = kwargs.get('dis_opt', None)
    current_sets = input_sets

    # Process each node in the computational graph
    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            # Input node
            node_values[node.name] = current_sets

        elif node.op == 'get_attr':
            # Parameter retrieval - handled when used by operations
            pass

        elif node.op == 'call_module':
            module = named_modules.get(node.target)
            if module is None:
                continue

            module_type = type(module).__name__

            if dis_opt == 'display':
                print(f'  Processing: {node.target} ({module_type})')

            # Handle ONNX-specific operations
            if module_type == 'OnnxBinaryMathOperation':
                current_sets = _handle_onnx_binary_op(
                    module, node, node_values, graph_module, set_type
                )
                if current_sets is not None:
                    node_values[node.name] = current_sets
                    continue

            elif module_type == 'OnnxMatMul':
                current_sets = _handle_onnx_matmul(
                    module, node, node_values, graph_module, set_type
                )
                if current_sets is not None:
                    node_values[node.name] = current_sets
                    continue

            # Standard PyTorch layer - use dispatcher
            if node.args and len(node.args) > 0:
                first_arg = node.args[0]
                if hasattr(first_arg, 'name') and first_arg.name in node_values:
                    input_sets_op = node_values[first_arg.name]
                    output_sets = reach_layer(module, input_sets_op, method, **kwargs)
                    node_values[node.name] = output_sets
                    current_sets = output_sets

        elif node.op == 'output':
            # Output node - extract final result
            if node.args and len(node.args) > 0:
                output_node = node.args[0]
                if hasattr(output_node, 'name') and output_node.name in node_values:
                    current_sets = node_values[output_node.name]

    return current_sets


def _handle_onnx_binary_op(module, node, node_values, graph_module, set_type):
    """Handle ONNX binary math operations (Add, Sub, etc.)."""
    from n2v.sets import Star, Hexatope, Octatope

    input_nodes = node.args
    if len(input_nodes) != 2:
        return None

    first_input, second_input = input_nodes

    # Check if second input is a parameter (bias term)
    if second_input.op != 'get_attr':
        return None

    # Get the parameter value
    param_path = second_input.target.split('.')
    param_module = graph_module
    for attr in param_path:
        param_module = getattr(param_module, attr)
    param_value = param_module.detach().cpu().numpy()

    # Get the sets from the first input
    if first_input.name not in node_values:
        return None

    input_sets_op = node_values[first_input.name]

    # Determine operation type
    if not hasattr(module, 'math_op_function'):
        return None

    op_name = module.math_op_function.__name__

    if 'add' in op_name:
        bias = param_value
    elif 'sub' in op_name:
        bias = -param_value
    else:
        raise NotImplementedError(
            f"Binary operation {op_name} not supported for {set_type.__name__}"
        )

    # Apply translation based on set type
    if set_type == Star:
        # Optimized: directly modify center
        output_sets = []
        bias_reshaped = bias.reshape(-1, 1)
        for s in input_sets_op:
            new_V = s.V.copy()
            new_V[:, 0:1] = new_V[:, 0:1] + bias_reshaped
            output_set = Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub)
            output_sets.append(output_set)
        return output_sets

    elif set_type in (Hexatope, Octatope):
        # Apply via dummy linear layer with identity + bias
        from n2v.nn.layer_ops.linear_reach import linear_hexatope, linear_octatope
        linear_fn = linear_hexatope if set_type == Hexatope else linear_octatope

        output_sets = []
        bias_tensor = torch.tensor(bias, dtype=torch.float32).reshape(-1)

        for s in input_sets_op:
            dummy_linear = nn.Linear(s.dim, s.dim, bias=True)
            dummy_linear.weight.data = torch.eye(s.dim)
            dummy_linear.bias.data = bias_tensor
            result = linear_fn(dummy_linear, [s])
            output_sets.extend(result)

        return output_sets

    else:
        raise NotImplementedError(
            f"ONNX binary operations not supported for {set_type.__name__}"
        )


def _handle_onnx_matmul(module, node, node_values, graph_module, set_type):
    """Handle ONNX MatMul operations."""
    from n2v.sets import Star, Hexatope, Octatope

    input_nodes = node.args
    if len(input_nodes) != 2:
        return None

    first_input, second_input = input_nodes

    # Check if second input is a weight matrix
    if second_input.op != 'get_attr':
        return None

    # Get the weight matrix
    param_path = second_input.target.split('.')
    param_module = graph_module
    for attr in param_path:
        param_module = getattr(param_module, attr)
    weight_matrix = param_module.detach().cpu().numpy()

    # Get the sets from the first input
    if first_input.name not in node_values:
        return None

    input_sets_op = node_values[first_input.name]

    # Apply linear transformation: y = x @ W means y = W^T @ x (in column vector form)
    if set_type == Star:
        output_sets = []
        for s in input_sets_op:
            output_set = s.affine_map(weight_matrix.T)
            output_sets.append(output_set)
        return output_sets

    elif set_type in (Hexatope, Octatope):
        from n2v.nn.layer_ops.linear_reach import linear_hexatope, linear_octatope
        linear_fn = linear_hexatope if set_type == Hexatope else linear_octatope

        output_sets = []
        for s in input_sets_op:
            dummy_linear = nn.Linear(s.dim, weight_matrix.shape[0], bias=False)
            dummy_linear.weight.data = torch.tensor(weight_matrix.T, dtype=torch.float32)
            result = linear_fn(dummy_linear, [s])
            output_sets.extend(result)

        return output_sets

    else:
        raise NotImplementedError(
            f"ONNX MatMul not supported for {set_type.__name__}"
        )
