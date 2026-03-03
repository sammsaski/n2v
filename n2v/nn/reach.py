"""
Unified reachability analysis for neural networks.

This module provides the core reachability computation engine that routes
computation based on set type and handles both standard PyTorch models
and ONNX GraphModules.
"""

import operator

import torch
import torch.nn as nn
import numpy as np
from typing import List, Union, Optional

# Import set types
from n2v.sets import Star, Zono, Box, Hexatope, Octatope
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

# Import layer ops
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.layer_ops.linear_reach import linear_hexatope, linear_octatope

# Import model preprocessing
from n2v.utils.model_preprocessing import fuse_batchnorm, has_batchnorm

import torch.fx as fx

# ONNX Reshape type (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.reshape import OnnxReshape
except ImportError:
    OnnxReshape = None

# ONNX Concat type (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.concat import OnnxConcat
except ImportError:
    OnnxConcat = None

# ONNX Slice types (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.slice import OnnxSlice
except ImportError:
    OnnxSlice = None

try:
    from onnx2torch.node_converters.slice import OnnxSliceV9
except ImportError:
    OnnxSliceV9 = None

# ONNX Split types (optional — onnx2torch may not be installed)
try:
    from onnx2torch.node_converters.split import OnnxSplit, OnnxSplit13
except ImportError:
    OnnxSplit = None
    OnnxSplit13 = None


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

    # Auto-fuse BatchNorm layers if present
    if has_batchnorm(model):
        model = fuse_batchnorm(model)

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
    if isinstance(model, fx.GraphModule):
        return _handle_graphmodule(model, [input_set], method, **kwargs)

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

            # Handle OnnxReshape
            if OnnxReshape is not None and isinstance(module, OnnxReshape):
                first_arg = node.args[0]
                if hasattr(first_arg, 'name') and first_arg.name in node_values:
                    input_sets_op = node_values[first_arg.name]
                else:
                    input_sets_op = current_sets

                # Get target shape from second argument (frozen parameter)
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

            # Handle OnnxSplit / OnnxSplit13
            if ((OnnxSplit is not None and isinstance(module, OnnxSplit)) or
                    (OnnxSplit13 is not None and isinstance(module, OnnxSplit13))):
                result = _handle_onnx_split(module, node, node_values, graph_module)
                if result is not None:
                    node_values[node.name] = result
                    # Don't set current_sets — outputs are extracted by getitem
                    continue

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

        elif node.op == 'call_function':
            # Handle operator.getitem for multi-output ops (e.g., Split)
            if node.target is operator.getitem:
                args = node.args
                if len(args) >= 2:
                    src_node = args[0]
                    index = args[1]
                    if hasattr(src_node, 'name') and src_node.name in node_values:
                        src_val = node_values[src_node.name]
                        # Multi-output: list-of-lists from Split
                        if (isinstance(src_val, list) and len(src_val) > 0
                                and isinstance(src_val[0], list)):
                            node_values[node.name] = src_val[index]
                            current_sets = src_val[index]

        elif node.op == 'output':
            # Output node - extract final result
            if node.args and len(node.args) > 0:
                output_node = node.args[0]
                if hasattr(output_node, 'name') and output_node.name in node_values:
                    current_sets = node_values[output_node.name]

    return current_sets


def _get_parameter(graph_module, node):
    """Extract a parameter tensor from a get_attr node."""
    param_path = node.target.split('.')
    param_module = graph_module
    for attr in param_path:
        param_module = getattr(param_module, attr)
    return param_module.detach().cpu()


def _handle_reshape(input_sets: List, target_shape: tuple) -> List:
    """
    Reshape sets to a new shape.

    Handles ONNX Reshape operations by reordering the V matrix.
    ONNX uses NCHW format; ImageStar stores HWC internally.

    Args:
        input_sets: List of input sets
        target_shape: Target shape tuple from ONNX (includes batch dim)

    Returns:
        List of reshaped sets
    """
    output_sets = []

    # Strip batch dimension from target shape
    if len(target_shape) >= 2:
        spatial_shape = target_shape[1:]  # Remove batch dim
    else:
        spatial_shape = target_shape

    for s in input_sets:
        if isinstance(s, ImageStar):
            output_sets.append(_reshape_imagestar(s, spatial_shape))
        elif isinstance(s, ImageZono):
            output_sets.append(_reshape_imagezono(s, spatial_shape))
        elif isinstance(s, Star):
            output_sets.append(_reshape_star(s, spatial_shape))
        elif isinstance(s, Zono):
            output_sets.append(_reshape_zono(s, spatial_shape))
        elif isinstance(s, Box):
            output_sets.append(_reshape_box(s, spatial_shape))
        else:
            # For Hexatope/Octatope, reshape is just reinterpretation
            output_sets.append(s)

    return output_sets


def _resolve_shape(shape: tuple, total_size: int) -> tuple:
    """Resolve -1 in a shape tuple given total element count."""
    if -1 not in shape:
        return shape
    known = 1
    neg_idx = -1
    for i, s in enumerate(shape):
        if s == -1:
            neg_idx = i
        else:
            known *= s
    resolved = list(shape)
    resolved[neg_idx] = total_size // known
    return tuple(resolved)


def _reshape_imagestar(star: 'ImageStar', spatial_shape: tuple) -> 'Star':
    """
    Reshape ImageStar. ONNX target is in CHW format.

    If result is flat (1D), return plain Star.
    If result is spatial (3D like C,H,W), return ImageStar.
    """
    V = star.V  # (H, W, C, nVar+1)
    h, w, c, n_cols = V.shape
    total = h * w * c

    resolved = _resolve_shape(spatial_shape, total)

    # Transpose V from HWC to CHW for ONNX compatibility
    V_chw = np.transpose(V, (2, 0, 1, 3))  # (C, H, W, nVar+1)
    V_flat = V_chw.reshape(total, n_cols)   # (C*H*W, nVar+1)

    if len(resolved) == 1 or (len(resolved) == 1 and resolved[0] == total):
        # Flat output -> plain Star
        return Star(V_flat, star.C, star.d, star.predicate_lb, star.predicate_ub)

    if len(resolved) == 3:
        # 3D output (C, H', W') -> ImageStar
        c_out, h_out, w_out = resolved
        V_chw_new = V_flat.reshape(c_out, h_out, w_out, n_cols)
        V_hwc = np.transpose(V_chw_new, (1, 2, 0, 3))  # (H', W', C, nVar+1)
        return ImageStar(
            V_hwc, star.C, star.d, star.predicate_lb, star.predicate_ub,
            h_out, w_out, c_out
        )

    # Default: flatten
    return Star(V_flat, star.C, star.d, star.predicate_lb, star.predicate_ub)


def _reshape_imagezono(zono: 'ImageZono', spatial_shape: tuple) -> 'Zono':
    """Reshape ImageZono. Returns plain Zono if flattened."""
    h, w, c_ch = zono.height, zono.width, zono.num_channels
    n_gen = zono.V.shape[1]
    total = h * w * c_ch

    resolved = _resolve_shape(spatial_shape, total)

    # Reshape center and generators to image format, then CHW
    c_img = zono.c.reshape(h, w, c_ch)
    V_img = zono.V.reshape(h, w, c_ch, n_gen)

    c_chw = np.transpose(c_img, (2, 0, 1))       # (C, H, W)
    V_chw = np.transpose(V_img, (2, 0, 1, 3))     # (C, H, W, n_gen)

    c_flat = c_chw.reshape(-1, 1)
    V_flat = V_chw.reshape(-1, n_gen)

    if len(resolved) == 3:
        c_out, h_out, w_out = resolved
        c_new = c_flat.reshape(c_out, h_out, w_out).transpose(1, 2, 0).reshape(-1, 1)
        V_new = V_flat.reshape(c_out, h_out, w_out, n_gen).transpose(1, 2, 0, 3).reshape(-1, n_gen)
        return ImageZono(c_new, V_new, h_out, w_out, c_out)

    # Flat output
    return Zono(c_flat, V_flat)


def _reshape_star(star: 'Star', spatial_shape: tuple) -> 'Star':
    """Reshape plain Star — V is already flat, just validate dims."""
    total = star.dim
    resolved = _resolve_shape(spatial_shape, total)
    product = 1
    for s in resolved:
        product *= s
    if product != total:
        raise ValueError(
            f"Cannot reshape Star of dim {total} to shape {resolved}"
        )
    # Plain Star has no spatial structure, reshape is a noop
    return star


def _reshape_zono(zono: 'Zono', spatial_shape: tuple) -> 'Zono':
    """Reshape plain Zono — already flat, validate dims."""
    total = zono.dim
    resolved = _resolve_shape(spatial_shape, total)
    product = 1
    for s in resolved:
        product *= s
    if product != total:
        raise ValueError(
            f"Cannot reshape Zono of dim {total} to shape {resolved}"
        )
    return zono


def _reshape_box(box: 'Box', spatial_shape: tuple) -> 'Box':
    """Reshape Box — already flat, validate dims."""
    total = box.dim
    resolved = _resolve_shape(spatial_shape, total)
    product = 1
    for s in resolved:
        product *= s
    if product != total:
        raise ValueError(
            f"Cannot reshape Box of dim {total} to shape {resolved}"
        )
    return box


def _handle_onnx_binary_op(module, node, node_values, graph_module, set_type):
    """Handle ONNX binary math operations (Add, Sub, etc.)."""
    input_nodes = node.args
    if len(input_nodes) != 2:
        return None

    first_input, second_input = input_nodes

    # Case 1: Both inputs are computed sets (residual connection)
    if (first_input.name in node_values and
            hasattr(second_input, 'name') and second_input.name in node_values):
        sets_a = node_values[first_input.name]
        sets_b = node_values[second_input.name]

        if not hasattr(module, 'math_op_function'):
            return None
        op_name = module.math_op_function.__name__

        if op_name in ('mul', '_onnx_div'):
            raise NotImplementedError(
                f"Element-wise {op_name} of two computed sets is not supported. "
                f"Only Mul/Div by constant is implemented."
            )

        return _add_sets(sets_a, sets_b, op_name)

    # Case 2: Second input is a parameter (bias/constant)
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

    if op_name == 'mul':
        return _mul_sets_by_constant(input_sets_op, param_value)
    elif op_name == '_onnx_div':
        return _mul_sets_by_constant(input_sets_op, 1.0 / param_value)
    elif 'add' in op_name:
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


def _coerce_set_types(sa, sb):
    """
    Coerce mismatched set types for element-wise operations.

    Handles cases where one branch produces an ImageStar/ImageZono and
    the other produces a Star/Zono (e.g., after flatten in one branch).

    Returns:
        Tuple (sa_coerced, sb_coerced) with compatible types.
    """
    # ImageStar + Star -> flatten ImageStar to Star
    if isinstance(sa, ImageStar) and isinstance(sb, Star) and not isinstance(sb, ImageStar):
        return sa.flatten_to_star(), sb
    if isinstance(sb, ImageStar) and isinstance(sa, Star) and not isinstance(sa, ImageStar):
        return sa, sb.flatten_to_star()

    # ImageZono + Zono -> convert ImageZono to plain Zono
    if isinstance(sa, ImageZono) and isinstance(sb, Zono) and not isinstance(sb, ImageZono):
        return Zono(sa.c, sa.V), sb
    if isinstance(sb, ImageZono) and isinstance(sa, Zono) and not isinstance(sa, ImageZono):
        return sa, Zono(sb.c, sb.V)

    return sa, sb


def _add_sets(sets_a, sets_b, op_name):
    """
    Element-wise addition or subtraction of two lists of sets.

    Used for residual connections where both inputs are computed sets
    (not constant parameters).

    For Star/ImageStar: Both sets share the same predicate variables alpha
    because they originated from the same input set. V_out = V1 +/- V2.
    Constraints (C, d, predicate_lb, predicate_ub) are preserved unchanged.

    For Zono/ImageZono: Generator tracking is not available, so we use
    Minkowski sum (generator concatenation) which is sound but over-approximate.
    c_out = c1 +/- c2, V_out = hstack(V1, +/-V2).

    For Box: Interval arithmetic. Add: lb1+lb2, ub1+ub2. Sub: lb1-ub2, ub1-lb2.

    Args:
        sets_a: List of sets from the first operand
        sets_b: List of sets from the second operand
        op_name: 'add' or 'sub'

    Returns:
        List of output sets
    """
    if len(sets_a) != len(sets_b):
        raise ValueError(
            f"Cannot {op_name} set lists of different lengths: "
            f"{len(sets_a)} vs {len(sets_b)}"
        )

    output_sets = []

    for sa, sb in zip(sets_a, sets_b):
        # Coerce mismatched types (e.g., ImageStar + Star)
        sa, sb = _coerce_set_types(sa, sb)

        if isinstance(sa, ImageStar) and isinstance(sb, ImageStar):
            # ImageStar: element-wise V addition (shared predicates)
            if op_name == 'add' or 'add' in op_name:
                V_out = sa.V + sb.V
            else:
                V_out = sa.V - sb.V

            out = ImageStar(
                V_out, sa.C, sa.d, sa.predicate_lb, sa.predicate_ub,
                sa.height, sa.width, sa.num_channels
            )
            output_sets.append(out)

        elif isinstance(sa, Star) and isinstance(sb, Star):
            # Star: element-wise V addition (shared predicates)
            if op_name == 'add' or 'add' in op_name:
                V_out = sa.V + sb.V
            else:
                V_out = sa.V - sb.V

            out = Star(V_out, sa.C, sa.d, sa.predicate_lb, sa.predicate_ub)
            output_sets.append(out)

        elif isinstance(sa, ImageZono) and isinstance(sb, ImageZono):
            # ImageZono: Minkowski sum via generator concatenation
            if op_name == 'add' or 'add' in op_name:
                c_out = sa.c + sb.c
                V_out = np.hstack([sa.V, sb.V])
            else:
                c_out = sa.c - sb.c
                V_out = np.hstack([sa.V, -sb.V])

            out = ImageZono(c_out, V_out, sa.height, sa.width, sa.num_channels)
            output_sets.append(out)

        elif isinstance(sa, Zono) and isinstance(sb, Zono):
            # Zono: Minkowski sum via generator concatenation
            if op_name == 'add' or 'add' in op_name:
                c_out = sa.c + sb.c
                V_out = np.hstack([sa.V, sb.V])
            else:
                c_out = sa.c - sb.c
                V_out = np.hstack([sa.V, -sb.V])

            out = Zono(c_out, V_out)
            output_sets.append(out)

        elif isinstance(sa, Box) and isinstance(sb, Box):
            # Box: interval arithmetic
            if op_name == 'add' or 'add' in op_name:
                lb_out = sa.lb + sb.lb
                ub_out = sa.ub + sb.ub
            else:
                lb_out = sa.lb - sb.ub
                ub_out = sa.ub - sb.lb

            out = Box(lb_out, ub_out)
            output_sets.append(out)

        else:
            raise NotImplementedError(
                f"Residual {op_name} not supported for "
                f"{type(sa).__name__} and {type(sb).__name__}"
            )

    return output_sets


def _mul_sets_by_constant(input_sets, scale):
    """
    Element-wise multiplication of sets by a constant scale vector.

    Used for ONNX Mul/Div operations where one operand is a frozen parameter
    (constant). Division is handled by the caller passing 1/constant.

    For Star/ImageStar: V_out = diag(scale) * V. Constraints (C, d,
    predicate_lb, predicate_ub) are unchanged because the predicate
    variables alpha are not affected by scaling the output space.

    For Zono/ImageZono: c_out = scale * c, V_out = scale * V (element-wise
    across the dimension axis). Zonotope alpha variables satisfy -1 <= alpha_i <= 1
    regardless of scaling.

    For Box: new_lb = min(scale*lb, scale*ub), new_ub = max(scale*lb, scale*ub).
    This correctly handles negative scale factors that swap bounds.

    Args:
        input_sets: List of input sets
        scale: Scale vector as numpy array (will be broadcast appropriately)

    Returns:
        List of scaled output sets
    """
    scale = np.asarray(scale, dtype=np.float64).flatten()

    output_sets = []

    for s in input_sets:
        if isinstance(s, ImageStar):
            # ImageStar V shape: (H, W, C, nVar+1)
            h, w, c, n_cols = s.V.shape
            total = h * w * c

            if scale.size == c:
                # Channel-wise scale: reshape to (1, 1, C, 1)
                scale_4d = scale.reshape(1, 1, c, 1)
            elif scale.size == total:
                # Full spatial scale: reshape to (H, W, C, 1)
                scale_4d = scale.reshape(h, w, c, 1)
            else:
                raise ValueError(
                    f"Scale size {scale.size} does not match ImageStar "
                    f"channels ({c}) or total dims ({total})"
                )

            new_V = s.V * scale_4d
            out = ImageStar(
                new_V, s.C, s.d, s.predicate_lb, s.predicate_ub,
                h, w, c
            )
            output_sets.append(out)

        elif isinstance(s, Star):
            # Star V shape: (dim, nVar+1)
            scale_col = scale.reshape(-1, 1)
            new_V = s.V * scale_col
            out = Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub)
            output_sets.append(out)

        elif isinstance(s, ImageZono):
            # ImageZono stores flat: c (H*W*C, 1), V (H*W*C, n_gen)
            # Scale is per-channel; tile across H*W pixels
            h, w, c_ch = s.height, s.width, s.num_channels
            total = h * w * c_ch

            if scale.size == c_ch:
                # Channel-wise: tile to H*W*C in HWC order
                scale_flat = np.tile(scale, h * w).reshape(-1, 1)
            elif scale.size == total:
                scale_flat = scale.reshape(-1, 1)
            else:
                raise ValueError(
                    f"Scale size {scale.size} does not match ImageZono "
                    f"channels ({c_ch}) or total dims ({total})"
                )

            new_c = s.c * scale_flat
            new_V = s.V * scale_flat
            out = ImageZono(new_c, new_V, h, w, c_ch)
            output_sets.append(out)

        elif isinstance(s, Zono):
            # Zono: c (dim, 1), V (dim, n_gen)
            scale_col = scale.reshape(-1, 1)
            new_c = s.c * scale_col
            new_V = s.V * scale_col
            out = Zono(new_c, new_V)
            output_sets.append(out)

        elif isinstance(s, Box):
            # Box: handle negative scale by taking min/max
            scale_col = scale.reshape(-1, 1)
            prod_lb = scale_col * s.lb
            prod_ub = scale_col * s.ub
            new_lb = np.minimum(prod_lb, prod_ub)
            new_ub = np.maximum(prod_lb, prod_ub)
            out = Box(new_lb, new_ub)
            output_sets.append(out)

        else:
            raise NotImplementedError(
                f"Mul by constant not supported for {type(s).__name__}"
            )

    return output_sets


def _concat_sets(set_lists, axis):
    """
    Concatenate multiple lists of sets along a specified axis.

    All branches in a single reachability pass share the same predicate
    variables (same C, d, pred bounds). Concat is just vertical stacking
    of V matrices -- no Minkowski sum needed.

    Args:
        set_lists: List of lists of sets, one list per concat input,
                   e.g. [[s1_a, s1_b], [s2_a, s2_b]]
        axis: Concatenation axis (0 for feature dim of flat sets,
              2 for channel dim of ImageStar HWC)

    Returns:
        List of concatenated output sets
    """
    if not set_lists or not any(set_lists):
        return []

    # Determine max length across input lists for broadcasting
    max_len = max(len(sl) for sl in set_lists)

    # Broadcast: if a list has length 1 and others have length N, repeat it
    broadcast_lists = []
    for sl in set_lists:
        if len(sl) == 1 and max_len > 1:
            broadcast_lists.append(sl * max_len)
        else:
            broadcast_lists.append(sl)

    output_sets = []

    for idx in range(max_len):
        # Collect one set from each broadcast list at this index
        sets_to_concat = [bl[idx] for bl in broadcast_lists]

        first = sets_to_concat[0]

        if isinstance(first, ImageStar):
            # ImageStar: np.concatenate V tensors along the specified axis
            # V shape: (H, W, C, nVar+1)
            # Pad V matrices to match on nVar+1 dimension if they differ
            V_list = [s.V for s in sets_to_concat]
            max_cols = max(v.shape[3] for v in V_list)
            padded_V = []
            for v in V_list:
                if v.shape[3] < max_cols:
                    pad_width = [(0, 0)] * 3 + [(0, max_cols - v.shape[3])]
                    v = np.pad(v, pad_width, mode='constant', constant_values=0.0)
                padded_V.append(v)

            V_out = np.concatenate(padded_V, axis=axis)

            # Update spatial dims from result shape
            h_out = V_out.shape[0]
            w_out = V_out.shape[1]
            c_out = V_out.shape[2]

            # Merge constraints: use the set with the most predicates
            ref = max(sets_to_concat, key=lambda s: s.V.shape[3])
            out = ImageStar(
                V_out, ref.C, ref.d,
                ref.predicate_lb, ref.predicate_ub,
                h_out, w_out, c_out
            )
            output_sets.append(out)

        elif isinstance(first, Star):
            # Star: vstack V matrices (only axis=0 makes sense for flat sets)
            # Pad V matrices to match on nVar+1 dimension if they differ
            V_list = [s.V for s in sets_to_concat]
            max_cols = max(v.shape[1] for v in V_list)
            padded_V = []
            for v in V_list:
                if v.shape[1] < max_cols:
                    pad_width = [(0, 0), (0, max_cols - v.shape[1])]
                    v = np.pad(v, pad_width, mode='constant', constant_values=0.0)
                padded_V.append(v)

            V_out = np.vstack(padded_V)

            # Merge constraints: use the set with the most predicates
            ref = max(sets_to_concat, key=lambda s: s.V.shape[1])
            out = Star(V_out, ref.C, ref.d,
                       ref.predicate_lb, ref.predicate_ub)
            output_sets.append(out)

        elif isinstance(first, ImageZono):
            # ImageZono: reshape to image, concatenate along axis, flatten back
            # ImageZono stores flat: c (H*W*C, 1), V (H*W*C, n_gen)
            img_c_list = []
            img_V_list = []
            total_channels = 0

            # Find max generators for padding
            max_gen = max(s.V.shape[1] for s in sets_to_concat)

            for s in sets_to_concat:
                h, w, c_ch = s.height, s.width, s.num_channels
                n_gen = s.V.shape[1]

                V_padded = s.V
                if n_gen < max_gen:
                    V_padded = np.pad(s.V, [(0, 0), (0, max_gen - n_gen)],
                                      mode='constant', constant_values=0.0)

                c_img = s.c.reshape(h, w, c_ch)
                V_img = V_padded.reshape(h, w, c_ch, max_gen)

                img_c_list.append(c_img)
                img_V_list.append(V_img)
                total_channels += c_ch

            c_cat = np.concatenate(img_c_list, axis=axis)
            V_cat = np.concatenate(img_V_list, axis=axis)

            h_out = c_cat.shape[0]
            w_out = c_cat.shape[1]
            c_out = c_cat.shape[2]

            c_flat = c_cat.reshape(-1, 1)
            V_flat = V_cat.reshape(-1, V_cat.shape[-1])

            out = ImageZono(c_flat, V_flat, h_out, w_out, c_out)
            output_sets.append(out)

        elif isinstance(first, Zono):
            # Zono: vstack c and V, pad generators if they differ
            c_list = [s.c for s in sets_to_concat]
            V_list = [s.V for s in sets_to_concat]

            max_gen = max(v.shape[1] for v in V_list)
            padded_V = []
            for v in V_list:
                if v.shape[1] < max_gen:
                    v = np.pad(v, [(0, 0), (0, max_gen - v.shape[1])],
                               mode='constant', constant_values=0.0)
                padded_V.append(v)

            c_out = np.vstack(c_list)
            V_out = np.vstack(padded_V)

            out = Zono(c_out, V_out)
            output_sets.append(out)

        elif isinstance(first, Box):
            # Box: vstack lb and ub
            lb_list = [s.lb for s in sets_to_concat]
            ub_list = [s.ub for s in sets_to_concat]
            lb_out = np.vstack(lb_list)
            ub_out = np.vstack(ub_list)

            out = Box(lb_out, ub_out)
            output_sets.append(out)

        else:
            raise NotImplementedError(
                f"Concat not supported for {type(first).__name__}"
            )

    return output_sets


def _handle_onnx_concat(module, node, node_values):
    """
    Handle ONNX Concat operations.

    Collects sets from all input nodes, maps the ONNX axis (which includes
    the batch dimension) to the set axis, and calls _concat_sets.

    Args:
        module: OnnxConcat module (has .axis attribute)
        node: Graph node
        node_values: Dict mapping node names to lists of sets

    Returns:
        List of concatenated output sets, or None if inputs not found
    """
    onnx_axis = module.axis

    # Collect set lists from all input arguments
    set_lists = []
    first_set = None
    for arg in node.args:
        if hasattr(arg, 'name') and arg.name in node_values:
            sl = node_values[arg.name]
            set_lists.append(sl)
            if first_set is None and len(sl) > 0:
                first_set = sl[0]

    if not set_lists or first_set is None:
        return None

    # Map ONNX axis (with batch dim) to set axis
    if isinstance(first_set, (ImageStar, ImageZono)):
        # ONNX uses NCHW: axis 0=N, 1=C, 2=H, 3=W
        # ImageStar uses HWC: axis 0=H, 1=W, 2=C
        onnx_to_hwc = {1: 2, 2: 0, 3: 1}
        set_axis = onnx_to_hwc.get(onnx_axis, onnx_axis)
    else:
        # Flat sets: strip batch dimension
        set_axis = onnx_axis - 1

    return _concat_sets(set_lists, set_axis)


def _slice_set(s, slices_by_axis):
    """
    Slice a set along specified axes.

    Slicing is a linear operation — it selects rows/elements from the
    basis matrix V. Constraints are unchanged because the predicate
    variables alpha are not affected.

    Args:
        s: Input set (Star, ImageStar, Zono, Box)
        slices_by_axis: Dict mapping axis (int) to Python slice object.
            For flat sets (Star, Zono, Box), axis 0 is the dimension axis.
            For ImageStar, axes are in HWC format: 0=H, 1=W, 2=C.

    Returns:
        New set of the same type with sliced dimensions
    """
    if isinstance(s, ImageStar):
        # V is (H, W, C, nVar+1) — build index for first 3 dims
        idx = [slice(None)] * 4  # default: select all along each axis
        for ax, sl in slices_by_axis.items():
            if ax < 3:
                idx[ax] = sl
        V_out = s.V[tuple(idx)]
        h_out = V_out.shape[0]
        w_out = V_out.shape[1]
        c_out = V_out.shape[2]
        return ImageStar(
            V_out, s.C, s.d, s.predicate_lb, s.predicate_ub,
            h_out, w_out, c_out
        )

    elif isinstance(s, Star):
        # V is (dim, nVar+1) — slice along dimension axis (axis 0)
        sl = slices_by_axis.get(0, slice(None))
        V_out = s.V[sl, :]
        return Star(V_out, s.C, s.d, s.predicate_lb, s.predicate_ub)

    elif isinstance(s, Zono):
        # c is (dim, 1), V is (dim, n_gen)
        sl = slices_by_axis.get(0, slice(None))
        c_out = s.c[sl, :]
        V_out = s.V[sl, :]
        return Zono(c_out, V_out)

    elif isinstance(s, Box):
        # lb, ub are (dim, 1)
        sl = slices_by_axis.get(0, slice(None))
        lb_out = s.lb[sl, :]
        ub_out = s.ub[sl, :]
        return Box(lb_out, ub_out)

    else:
        raise NotImplementedError(
            f"Slice not supported for {type(s).__name__}"
        )


def _handle_onnx_slice(module, node, node_values, graph_module):
    """
    Handle ONNX Slice operations.

    Supports two ONNX opset versions:
    - OnnxSlice (v10+): starts, ends, axes, steps from node args
    - OnnxSliceV9: slice info stored in module._pos_axes_slices

    Args:
        module: OnnxSlice or OnnxSliceV9 module
        node: Graph node
        node_values: Dict mapping node names to lists of sets
        graph_module: Parent graph module (for parameter extraction)

    Returns:
        List of sliced output sets, or None if inputs not found
    """
    # Get input sets
    first_arg = node.args[0]
    if hasattr(first_arg, 'name') and first_arg.name in node_values:
        input_sets = node_values[first_arg.name]
    else:
        return None

    first_set = input_sets[0] if input_sets else None
    if first_set is None:
        return None

    slices_by_axis = {}

    if OnnxSliceV9 is not None and isinstance(module, OnnxSliceV9):
        # V9: module._pos_axes_slices is a list of slice objects per axis
        # Skip axis 0 (batch dim), shift remaining by -1
        for ax, sl in enumerate(module._pos_axes_slices):
            if ax == 0:
                continue  # skip batch dimension
            set_ax = ax - 1
            slices_by_axis[set_ax] = sl

    elif OnnxSlice is not None and isinstance(module, OnnxSlice):
        # V10+: extract starts, ends, axes, steps from node args
        # node.args = [input, starts, ends, axes, steps] (axes and steps optional)
        starts_tensor = _get_parameter(graph_module, node.args[1])
        ends_tensor = _get_parameter(graph_module, node.args[2])
        starts = starts_tensor.numpy().astype(int).flatten()
        ends = ends_tensor.numpy().astype(int).flatten()

        if len(node.args) > 3:
            axes_tensor = _get_parameter(graph_module, node.args[3])
            axes = axes_tensor.numpy().astype(int).flatten()
        else:
            axes = np.arange(len(starts))

        if len(node.args) > 4:
            steps_tensor = _get_parameter(graph_module, node.args[4])
            steps = steps_tensor.numpy().astype(int).flatten()
        else:
            steps = np.ones(len(starts), dtype=int)

        for i in range(len(starts)):
            ax = int(axes[i])
            if ax == 0:
                continue  # skip batch dimension
            set_ax = ax - 1  # remove batch dim

            start_val = int(starts[i])
            end_val = int(ends[i])
            step_val = int(steps[i])

            # ONNX uses very large numbers (e.g., 2^63-1) for "to end"
            if end_val > 2**30:
                end_val = None

            slices_by_axis[set_ax] = slice(start_val, end_val, step_val)

    else:
        return None

    # For ImageStar/ImageZono inputs, map ONNX axes (after batch removal)
    # from NCHW (0=C, 1=H, 2=W) to HWC (0=H, 1=W, 2=C)
    if isinstance(first_set, (ImageStar, ImageZono)):
        nchw_to_hwc = {0: 2, 1: 0, 2: 1}
        remapped = {}
        for ax, sl in slices_by_axis.items():
            hwc_ax = nchw_to_hwc.get(ax, ax)
            remapped[hwc_ax] = sl
        slices_by_axis = remapped

    # Apply slice to each set
    output_sets = []
    for s in input_sets:
        output_sets.append(_slice_set(s, slices_by_axis))

    return output_sets


def _split_set(s, split_sizes, axis):
    """
    Split a set into chunks along a given axis.

    Splitting is a linear operation — it partitions rows/elements of the
    basis matrix. Constraints are unchanged because the predicate variables
    alpha are not affected.

    Args:
        s: Input set (Star, ImageStar, Zono, ImageZono, Box)
        split_sizes: List of ints giving the size of each chunk along axis
        axis: Axis to split along.
            For flat sets (Star, Zono, Box), axis 0 is the dimension axis.
            For ImageStar/ImageZono, axes are in HWC format: 0=H, 1=W, 2=C.

    Returns:
        List of sets, one per chunk
    """
    chunks = []
    offset = 0

    if isinstance(s, ImageStar):
        # V is (H, W, C, nVar+1) — split along the specified HWC axis
        for size in split_sizes:
            idx = [slice(None)] * 4
            idx[axis] = slice(offset, offset + size)
            V_chunk = s.V[tuple(idx)]
            h_out = V_chunk.shape[0]
            w_out = V_chunk.shape[1]
            c_out = V_chunk.shape[2]
            chunk = ImageStar(
                V_chunk, s.C, s.d, s.predicate_lb, s.predicate_ub,
                h_out, w_out, c_out
            )
            chunks.append(chunk)
            offset += size

    elif isinstance(s, Star):
        # V is (dim, nVar+1) — split along rows (axis 0)
        for size in split_sizes:
            V_chunk = s.V[offset:offset + size, :]
            chunk = Star(V_chunk, s.C, s.d, s.predicate_lb, s.predicate_ub)
            chunks.append(chunk)
            offset += size

    elif isinstance(s, ImageZono):
        # Reshape to image, split, flatten back
        h, w, c_ch = s.height, s.width, s.num_channels
        n_gen = s.V.shape[1]
        c_img = s.c.reshape(h, w, c_ch)
        V_img = s.V.reshape(h, w, c_ch, n_gen)

        for size in split_sizes:
            idx = [slice(None)] * 4
            idx[axis] = slice(offset, offset + size)
            c_chunk = c_img[tuple(idx[:3])]
            V_chunk = V_img[tuple(idx)]
            h_out = c_chunk.shape[0]
            w_out = c_chunk.shape[1]
            c_out = c_chunk.shape[2]
            chunk = ImageZono(
                c_chunk.reshape(-1, 1),
                V_chunk.reshape(-1, n_gen),
                h_out, w_out, c_out
            )
            chunks.append(chunk)
            offset += size

    elif isinstance(s, Zono):
        # c is (dim, 1), V is (dim, n_gen) — split along rows
        for size in split_sizes:
            c_chunk = s.c[offset:offset + size, :]
            V_chunk = s.V[offset:offset + size, :]
            chunk = Zono(c_chunk, V_chunk)
            chunks.append(chunk)
            offset += size

    elif isinstance(s, Box):
        # lb, ub are (dim, 1) — split along rows
        for size in split_sizes:
            lb_chunk = s.lb[offset:offset + size, :]
            ub_chunk = s.ub[offset:offset + size, :]
            chunk = Box(lb_chunk, ub_chunk)
            chunks.append(chunk)
            offset += size

    else:
        raise NotImplementedError(
            f"Split not supported for {type(s).__name__}"
        )

    return chunks


def _handle_onnx_split(module, node, node_values, graph_module):
    """
    Handle ONNX Split operations.

    Supports OnnxSplit (v2/v11) and OnnxSplit13.

    Args:
        module: OnnxSplit or OnnxSplit13 module
        node: Graph node
        node_values: Dict mapping node names to lists of sets
        graph_module: Parent graph module (for parameter extraction)

    Returns:
        List of lists: [[chunk0_sets], [chunk1_sets], ...] — one list per
        split output — or None if inputs not found
    """
    # Get input sets
    first_arg = node.args[0]
    if hasattr(first_arg, 'name') and first_arg.name in node_values:
        input_sets = node_values[first_arg.name]
    else:
        return None

    first_set = input_sets[0] if input_sets else None
    if first_set is None:
        return None

    # Determine split sizes
    split_sizes = None
    onnx_axis = module.axis

    if OnnxSplit13 is not None and isinstance(module, OnnxSplit13):
        # OnnxSplit13: split sizes from second argument (dynamic) or even division
        if len(node.args) > 1:
            split_tensor = _get_parameter(graph_module, node.args[1])
            split_sizes = split_tensor.numpy().astype(int).tolist()
        else:
            # Even division by num_splits
            split_sizes = None  # handled below

    elif OnnxSplit is not None and isinstance(module, OnnxSplit):
        # OnnxSplit: split sizes from module.split attribute
        if module.split is not None:
            split_sizes = list(module.split)
        else:
            split_sizes = None  # handled below

    # Map ONNX axis (with batch dim) to set axis
    if isinstance(first_set, (ImageStar, ImageZono)):
        # ONNX uses NCHW: axis 0=N, 1=C, 2=H, 3=W
        # ImageStar uses HWC: axis 0=H, 1=W, 2=C
        onnx_to_hwc = {1: 2, 2: 0, 3: 1}
        set_axis = onnx_to_hwc.get(onnx_axis, onnx_axis)
    else:
        # Flat sets: strip batch dimension
        set_axis = onnx_axis - 1

    # If split_sizes not specified, use even division
    if split_sizes is None:
        num_splits = module.num_splits
        # Determine dimension size along set_axis
        if isinstance(first_set, ImageStar):
            axis_sizes = [first_set.V.shape[0], first_set.V.shape[1],
                          first_set.V.shape[2]]
            dim_size = axis_sizes[set_axis]
        elif isinstance(first_set, ImageZono):
            axis_sizes = [first_set.height, first_set.width,
                          first_set.num_channels]
            dim_size = axis_sizes[set_axis]
        elif isinstance(first_set, Star):
            dim_size = first_set.dim
        elif isinstance(first_set, Zono):
            dim_size = first_set.dim
        elif isinstance(first_set, Box):
            dim_size = first_set.dim
        else:
            return None
        split_sizes = [dim_size // num_splits] * num_splits

    # Apply split to each set, producing list-of-lists
    num_chunks = len(split_sizes)
    # result[i] = list of chunk-i sets across all input sets
    result = [[] for _ in range(num_chunks)]

    for s in input_sets:
        chunks = _split_set(s, split_sizes, set_axis)
        for i, chunk in enumerate(chunks):
            result[i].append(chunk)

    return result


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
