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

# Import set types
from n2v.sets import Star, Zono, Box, Hexatope, Octatope
from n2v.sets.image_star import ImageStar

# Import layer ops
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.layer_ops.linear_reach import linear_hexatope, linear_octatope

# Optional torch.fx import for ONNX support
try:
    import torch.fx as fx
    HAS_TORCH_FX = True
except ImportError:
    HAS_TORCH_FX = False


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
            - 'probabilistic': Model-agnostic probabilistic verification (any input set -> ProbabilisticBox)
            - 'hybrid': Deterministic until threshold, then probabilistic
        **kwargs: Method-specific arguments:
            - lp_solver: LP solver to use
            - verbose: 'display' to show progress
            - parallel: Enable parallel processing (Star)
            - n_workers: Number of parallel workers
            - relax_factor: Relaxation factor for approx methods
            - relax_method: Relaxation strategy

            For 'probabilistic' and 'hybrid' methods:
            - m: int - Calibration set size (default: 8000)
            - ell: int - Rank parameter (default: m-1)
            - epsilon: float - Miscoverage level (default: 0.001)
            - surrogate: str - 'naive' or 'clipping_block' (default: 'clipping_block')
            - training_samples: int - For clipping_block surrogate (default: m//2)
            - pca_components: int - Dimensionality reduction (default: None)

            For 'hybrid' method additionally:
            - max_stars: int - Switch to probabilistic if exceeded (default: 1000)
            - timeout_per_layer: float - Seconds before switching (default: 30.0)

    Returns:
        List of output sets (same type as input for deterministic methods,
        ProbabilisticBox for probabilistic method)

    Raises:
        TypeError: If input_set type is not supported
        ValueError: If method is not valid for the given set type
    """
    # Handle probabilistic and hybrid methods
    if method == 'probabilistic':
        return _reach_probabilistic(model, input_set, **kwargs)

    if method == 'hybrid':
        return _reach_hybrid(model, input_set, **kwargs)
    # Validate method for set type
    # Note: ImageStar is checked first since it's more specific than Star
    if isinstance(input_set, ImageStar):
        if method not in ('exact', 'approx'):
            raise ValueError(f"ImageStar supports 'exact' or 'approx', got '{method}'")
    elif isinstance(input_set, Star):
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
            f"Supported types: Star, ImageStar, Zono, Box, Hexatope, Octatope"
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
    current_sets = input_sets
    verbose = kwargs.get('verbose', False)

    # Get all layers from the model
    layers = list(model.children())
    if not layers:
        # Model might be a single layer
        layers = [model]

    # Propagate through each layer
    for i, layer in enumerate(layers):
        if verbose:
            set_type = type(current_sets[0]).__name__
            print(f'Layer {i+1}/{len(layers)}: {type(layer).__name__}')

        current_sets = reach_layer(layer, current_sets, method, **kwargs)

        if verbose:
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
    # Get the set type from the first input
    set_type = type(input_sets[0])

    # Get named modules
    named_modules = dict(graph_module.named_modules())

    # Store intermediate values for each node
    node_values = {}
    verbose = kwargs.get('verbose', None)
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

            if verbose:
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


def _reach_probabilistic(model, input_set, **kwargs):
    """
    Probabilistic reachability using conformal inference.

    This is a model-agnostic approach that works with any PyTorch model.
    """
    from n2v.probabilistic import verify

    # Convert input_set to Box if needed
    if isinstance(input_set, Box):
        box = input_set
    elif hasattr(input_set, 'estimate_ranges'):
        lb, ub = input_set.estimate_ranges()
        box = Box(lb, ub)
    elif hasattr(input_set, 'get_ranges'):
        lb, ub = input_set.get_ranges()
        box = Box(lb, ub)
    else:
        raise TypeError(f"Cannot convert {type(input_set)} to Box for probabilistic verification")

    # Create model wrapper for numpy interface
    def model_fn(x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            output = model(x_tensor)
            return output.numpy()

    # Run probabilistic verification
    result = verify(
        model=model_fn,
        input_set=box,
        m=kwargs.get('m', 8000),
        ell=kwargs.get('ell', None),
        epsilon=kwargs.get('epsilon', 0.001),
        surrogate=kwargs.get('surrogate', 'clipping_block'),
        training_samples=kwargs.get('training_samples', None),
        pca_components=kwargs.get('pca_components', None),
        batch_size=kwargs.get('batch_size', 100),
        seed=kwargs.get('seed', None),
        verbose=kwargs.get('verbose', False)
    )

    return [result]


def _reach_hybrid(model, input_set, **kwargs):
    """
    Hybrid reachability: deterministic until threshold, then probabilistic.

    Attempts exact reachability layer by layer. If the number of stars exceeds
    max_stars or time exceeds timeout_per_layer, switches to probabilistic
    verification for the remaining layers.
    """
    import time

    max_stars = kwargs.get('max_stars', 1000)
    timeout_per_layer = kwargs.get('timeout_per_layer', 30.0)
    verbose = kwargs.get('verbose', False)

    # Get layers
    layers = list(model.children())
    if not layers:
        layers = [model]

    current_sets = [input_set]

    for i, layer in enumerate(layers):
        if verbose:
            print(f"Layer {i+1}/{len(layers)}: {type(layer).__name__}")

        start_time = time.time()

        try:
            # Try deterministic reachability
            next_sets = reach_layer(layer, current_sets, 'exact', **kwargs)
            elapsed = time.time() - start_time

            # Check thresholds
            if len(next_sets) > max_stars:
                if verbose:
                    print(f"  Exceeded {max_stars} stars, switching to probabilistic")
                raise _SwitchToProbabilistic()

            if elapsed > timeout_per_layer:
                if verbose:
                    print(f"  Exceeded {timeout_per_layer}s timeout, switching to probabilistic")
                raise _SwitchToProbabilistic()

            current_sets = next_sets

        except (_SwitchToProbabilistic, MemoryError):
            # Switch to probabilistic for remaining layers
            remaining_model = nn.Sequential(*layers[i:])

            # Get bounds from current sets
            all_lb = []
            all_ub = []
            for s in current_sets:
                if hasattr(s, 'estimate_ranges'):
                    lb, ub = s.estimate_ranges()
                elif hasattr(s, 'get_ranges'):
                    lb, ub = s.get_ranges()
                else:
                    lb, ub = s.lb, s.ub
                all_lb.append(lb.flatten())
                all_ub.append(ub.flatten())

            combined_lb = np.min(np.stack(all_lb), axis=0)
            combined_ub = np.max(np.stack(all_ub), axis=0)

            # Run probabilistic on remaining network
            return _reach_probabilistic(
                remaining_model,
                Box(combined_lb, combined_ub),
                **kwargs
            )

    return current_sets


class _SwitchToProbabilistic(Exception):
    """Signal to switch from deterministic to probabilistic."""
    # TODO:
    pass
