"""
Star-based reachability analysis for neural networks.
"""

import torch.nn as nn
from typing import List, Optional
from n2v.nn.layer_ops.dispatcher import reach_layer_star


def reach_star_exact(
    model: nn.Module,
    input_stars: List,
    num_cores: int = 1,
    lp_solver: str = 'default',
    dis_opt: Optional[str] = None
) -> List:
    """
    Exact reachability using Star sets.

    Propagates Star sets through the network layer by layer using exact methods.

    Args:
        model: PyTorch model
        input_stars: List of input Star sets
        num_cores: Number of cores for parallel computation (TODO)
        lp_solver: LP solver to use
        dis_opt: 'display' to show progress

    Returns:
        List of output Star sets
    """
    current_stars = input_stars

    # Check if model is a GraphModule (from torch.fx / onnx2torch)
    try:
        import torch.fx as fx
        if isinstance(model, fx.GraphModule):
            return _reach_graphmodule_star(
                model, input_stars, method='exact',
                lp_solver=lp_solver, dis_opt=dis_opt
            )
    except ImportError:
        pass

    # Get all layers from the model
    layers = list(model.children())

    if not layers:
        # Model might be a single layer
        layers = [model]

    # Propagate through each layer
    for i, layer in enumerate(layers):
        if dis_opt == 'display':
            print(f'Layer {i+1}/{len(layers)}: {type(layer).__name__}')

        current_stars = reach_layer_star(
            layer,
            current_stars,
            method='exact',
            lp_solver=lp_solver,
            dis_opt=dis_opt
        )

        if dis_opt == 'display':
            print(f'  Output: {len(current_stars)} Star sets')

    return current_stars


def reach_star_approx(
    model: nn.Module,
    input_stars: List,
    num_cores: int = 1,
    relax_factor: float = 0.5,
    relax_method: str = 'standard',
    lp_solver: str = 'default',
    dis_opt: Optional[str] = None
) -> List:
    """
    Approximate reachability using Star sets with relaxation.

    Args:
        model: PyTorch model
        input_stars: List of input Star sets
        num_cores: Number of cores
        relax_factor: Relaxation factor for over-approximation (0=exact, 1=max)
        relax_method: Relaxation strategy - 'standard', 'range', 'area', 'bound'
        lp_solver: LP solver
        dis_opt: Display option

    Returns:
        List of output Star sets
    """
    current_stars = input_stars

    # Check if model is a GraphModule (from torch.fx / onnx2torch)
    try:
        import torch.fx as fx
        if isinstance(model, fx.GraphModule):
            return _reach_graphmodule_star(
                model, input_stars, method='approx',
                relax_factor=relax_factor, relax_method=relax_method,
                lp_solver=lp_solver, dis_opt=dis_opt
            )
    except ImportError:
        pass

    layers = list(model.children())
    if not layers:
        layers = [model]

    for i, layer in enumerate(layers):
        if dis_opt == 'display':
            print(f'Layer {i+1}/{len(layers)}: {type(layer).__name__}')

        current_stars = reach_layer_star(
            layer,
            current_stars,
            method='approx',
            relax_factor=relax_factor,
            relax_method=relax_method,
            lp_solver=lp_solver,
            dis_opt=dis_opt
        )

        if dis_opt == 'display':
            print(f'  Output: {len(current_stars)} Star sets')

    return current_stars


def _reach_graphmodule_star(
    graph_module,
    input_stars: List,
    method: str = 'exact',
    **kwargs
) -> List:
    """
    Handle reachability for torch.fx.GraphModule (e.g., from onnx2torch).

    Args:
        graph_module: GraphModule to analyze
        input_stars: Input star sets
        method: 'exact' or 'approx'
        **kwargs: Additional arguments (lp_solver, relax_factor, etc.)

    Returns:
        List of output Star sets
    """
    import torch
    import numpy as np

    # Get named modules
    named_modules = dict(graph_module.named_modules())

    # Store intermediate values
    node_values = {}

    dis_opt = kwargs.get('dis_opt', None)

    # Get the current stars (mapped to the input node)
    current_stars = input_stars

    # Process each node in the graph
    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            # This is the input
            node_values[node.name] = current_stars

        elif node.op == 'get_attr':
            # This retrieves a parameter/weight - store it for later use
            # We'll handle it when an operation uses it
            pass

        elif node.op == 'call_module':
            module = named_modules.get(node.target)
            if module is None:
                continue

            module_type = type(module).__name__

            if dis_opt == 'display':
                print(f'  Processing: {node.target} ({module_type})')

            # Check if it's an ONNX operation that needs special handling
            if module_type == 'OnnxBinaryMathOperation':
                # Handle Add/Sub/Mul/Div - these typically add bias or normalize
                input_nodes = node.args
                if len(input_nodes) == 2:
                    # Get the first input (should be from previous layer)
                    first_input = input_nodes[0]
                    second_input = input_nodes[1]

                    # Check if second input is a parameter
                    if second_input.op == 'get_attr':
                        # Get the parameter value
                        param_path = second_input.target.split('.')
                        param_module = graph_module
                        for attr in param_path:
                            param_module = getattr(param_module, attr)
                        param_value = param_module.detach().cpu().numpy()

                        # Get the stars from the first input
                        if first_input.name in node_values:
                            input_stars_op = node_values[first_input.name]

                            # Apply the operation
                            # For Add with a constant bias, this is a translation
                            if hasattr(module, 'math_op_function'):
                                op_name = module.math_op_function.__name__
                                if 'add' in op_name:
                                    # Translation: y = x + b
                                    # Optimized: directly modify center instead of matrix multiply
                                    from n2v.sets import Star
                                    output_stars = []
                                    bias_reshaped = param_value.reshape(-1, 1)
                                    for star in input_stars_op:
                                        # Create new star with translated center
                                        new_V = star.V.copy()
                                        new_V[:, 0:1] = new_V[:, 0:1] + bias_reshaped
                                        output_star = Star(new_V, star.C, star.d,
                                                         star.predicate_lb, star.predicate_ub)
                                        output_stars.append(output_star)
                                    node_values[node.name] = output_stars
                                    current_stars = output_stars
                                elif 'sub' in op_name:
                                    # Translation: y = x - b (equivalent to x + (-b))
                                    from n2v.sets import Star
                                    output_stars = []
                                    bias_reshaped = (-param_value).reshape(-1, 1)
                                    for star in input_stars_op:
                                        # Create new star with translated center
                                        new_V = star.V.copy()
                                        new_V[:, 0:1] = new_V[:, 0:1] + bias_reshaped
                                        output_star = Star(new_V, star.C, star.d,
                                                         star.predicate_lb, star.predicate_ub)
                                        output_stars.append(output_star)
                                    node_values[node.name] = output_stars
                                    current_stars = output_stars
                                else:
                                    # Other operations not yet supported
                                    raise NotImplementedError(
                                        f"Binary operation {op_name} not yet supported for reachability"
                                    )
                        continue

            elif module_type == 'OnnxMatMul':
                # Handle MatMul - this is a linear transformation
                input_nodes = node.args
                if len(input_nodes) == 2:
                    first_input = input_nodes[0]
                    second_input = input_nodes[1]

                    # Check if second input is a weight matrix
                    if second_input.op == 'get_attr':
                        # Get the weight matrix
                        param_path = second_input.target.split('.')
                        param_module = graph_module
                        for attr in param_path:
                            param_module = getattr(param_module, attr)
                        weight_matrix = param_module.detach().cpu().numpy()

                        # Get the stars from the first input
                        if first_input.name in node_values:
                            input_stars_op = node_values[first_input.name]

                            # Apply linear transformation: y = x @ W
                            # Note: PyTorch matmul is x @ W, so result is (batch, out_features)
                            # For Star affine_map, we need W' where y = W' @ x
                            # If x is (n,1) and we want y = x^T @ W, then we need W^T
                            output_stars = []
                            for star in input_stars_op:
                                # y = x @ W  means  y = W^T @ x (in column vector form)
                                output_star = star.affine_map(weight_matrix.T)
                                output_stars.append(output_star)
                            node_values[node.name] = output_stars
                            current_stars = output_stars
                        continue

            # For standard PyTorch layers, use the regular dispatcher
            if node.args and len(node.args) > 0:
                first_arg = node.args[0]
                if hasattr(first_arg, 'name') and first_arg.name in node_values:
                    input_stars_op = node_values[first_arg.name]
                    output_stars = reach_layer_star(
                        module,
                        input_stars_op,
                        method=method,
                        **kwargs
                    )
                    node_values[node.name] = output_stars
                    current_stars = output_stars

        elif node.op == 'output':
            # This is the output node
            if node.args and len(node.args) > 0:
                output_node = node.args[0]
                if hasattr(output_node, 'name') and output_node.name in node_values:
                    current_stars = node_values[output_node.name]

    return current_stars
