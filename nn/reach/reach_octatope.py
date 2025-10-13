"""
Octatope-based reachability analysis for neural networks.
"""

import torch.nn as nn
from typing import List, Optional
from n2v.sets.octatope import Octatope
from n2v.nn.layer_ops.dispatcher import reach_layer_octatope


def reach_octatope_approx(
    model: nn.Module,
    input_octatopes: List[Octatope],
    use_differentiable: bool = False,
    dis_opt: Optional[str] = None
) -> List[Octatope]:
    """
    Approximate reachability using Octatope sets (UTVPI-based representation).

    Propagates Octatope sets through the network layer by layer.

    Args:
        model: PyTorch model
        input_octatopes: List of input Octatope sets
        use_differentiable: Use differentiable LP solver instead of CVXPY
        dis_opt: 'display' to show progress

    Returns:
        List of output Octatope sets
    """
    current_octatopes = input_octatopes

    # Check if model is a GraphModule (from torch.fx / onnx2torch)
    try:
        import torch.fx as fx
        if isinstance(model, fx.GraphModule):
            return _reach_graphmodule_octatope(
                model, input_octatopes,
                method='approx',
                use_differentiable=use_differentiable,
                dis_opt=dis_opt
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

        current_octatopes = reach_layer_octatope(
            layer,
            current_octatopes,
            method='approx',
            use_differentiable=use_differentiable,
            dis_opt=dis_opt
        )

        if dis_opt == 'display':
            print(f'  Output: {len(current_octatopes)} Octatope sets')

    return current_octatopes


def reach_octatope_exact(
    model: nn.Module,
    input_octatopes: List[Octatope],
    use_differentiable: bool = False,
    dis_opt: Optional[str] = None
) -> List[Octatope]:
    """
    Exact reachability using Octatope sets with exact ReLU splitting.

    Propagates Octatope sets through the network layer by layer using exact
    ReLU handling. Splits on neurons crossing zero for tighter bounds.

    Args:
        model: PyTorch model
        input_octatopes: List of input Octatope sets
        use_differentiable: Use differentiable LP solver instead of CVXPY
        dis_opt: 'display' to show progress

    Returns:
        List of output Octatope sets (may be more due to splitting)
    """
    current_octatopes = input_octatopes

    # Check if model is a GraphModule (from torch.fx / onnx2torch)
    try:
        import torch.fx as fx
        if isinstance(model, fx.GraphModule):
            return _reach_graphmodule_octatope(
                model, input_octatopes,
                method='exact',
                use_differentiable=use_differentiable,
                dis_opt=dis_opt
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

        current_octatopes = reach_layer_octatope(
            layer,
            current_octatopes,
            method='exact',
            use_differentiable=use_differentiable,
            dis_opt=dis_opt
        )

        if dis_opt == 'display':
            print(f'  Output: {len(current_octatopes)} Octatope sets')

    return current_octatopes


def _reach_graphmodule_octatope(
    graph_module,
    input_octatopes: List[Octatope],
    method: str = 'exact',
    **kwargs
) -> List[Octatope]:
    """
    Handle reachability for torch.fx.GraphModule (e.g., from onnx2torch).

    Args:
        graph_module: GraphModule to analyze
        input_octatopes: Input octatope sets
        method: 'exact' or 'approx'
        **kwargs: Additional arguments (use_differentiable, dis_opt, etc.)

    Returns:
        List of output Octatope sets
    """
    import torch
    import numpy as np
    from n2v.nn.layer_ops.linear_reach import linear_octatope

    # Get named modules
    named_modules = dict(graph_module.named_modules())

    # Store intermediate values
    node_values = {}

    dis_opt = kwargs.get('dis_opt', None)

    # Get the current octatopes (mapped to the input node)
    current_octatopes = input_octatopes

    # Process each node in the graph
    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            # This is the input
            node_values[node.name] = current_octatopes

        elif node.op == 'get_attr':
            # This retrieves a parameter/weight - store it for later use
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
                # Handle Add/Sub - these typically add bias
                input_nodes = node.args
                if len(input_nodes) == 2:
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

                        # Get the octatopes from the first input
                        if first_input.name in node_values:
                            input_octatopes_op = node_values[first_input.name]

                            # Apply the operation
                            if hasattr(module, 'math_op_function'):
                                op_name = module.math_op_function.__name__
                                if 'add' in op_name or 'sub' in op_name:
                                    # Translation: y = x ± b
                                    # Apply via dummy linear layer with identity + bias
                                    output_octatopes = []
                                    bias = param_value if 'add' in op_name else -param_value
                                    bias_tensor = torch.tensor(bias, dtype=torch.float32).reshape(-1)

                                    for oct_set in input_octatopes_op:
                                        dummy_linear = nn.Linear(oct_set.dim, oct_set.dim, bias=True)
                                        dummy_linear.weight.data = torch.eye(oct_set.dim)
                                        dummy_linear.bias.data = bias_tensor
                                        result = linear_octatope(dummy_linear, [oct_set])
                                        output_octatopes.extend(result)

                                    node_values[node.name] = output_octatopes
                                    current_octatopes = output_octatopes
                                else:
                                    raise NotImplementedError(
                                        f"Binary operation {op_name} not yet supported for Octatopes"
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

                        # Get the octatopes from the first input
                        if first_input.name in node_values:
                            input_octatopes_op = node_values[first_input.name]

                            # Apply linear transformation: y = x @ W means y = W^T @ x
                            output_octatopes = []
                            for oct_set in input_octatopes_op:
                                dummy_linear = nn.Linear(oct_set.dim, weight_matrix.shape[0], bias=False)
                                dummy_linear.weight.data = torch.tensor(weight_matrix.T, dtype=torch.float32)
                                result = linear_octatope(dummy_linear, [oct_set])
                                output_octatopes.extend(result)

                            node_values[node.name] = output_octatopes
                            current_octatopes = output_octatopes
                        continue

            # For standard PyTorch layers, use the dispatcher
            if node.args and len(node.args) > 0:
                first_arg = node.args[0]
                if hasattr(first_arg, 'name') and first_arg.name in node_values:
                    input_octatopes_op = node_values[first_arg.name]
                    output_octatopes = reach_layer_octatope(
                        module,
                        input_octatopes_op,
                        method=method,
                        **kwargs
                    )
                    node_values[node.name] = output_octatopes
                    current_octatopes = output_octatopes

        elif node.op == 'output':
            # This is the output node
            if node.args and len(node.args) > 0:
                output_node = node.args[0]
                if hasattr(output_node, 'name') and output_node.name in node_values:
                    current_octatopes = node_values[output_node.name]

    return current_octatopes
