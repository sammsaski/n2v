"""
Unified reachability analysis for neural networks.

This module provides the core reachability computation engine that routes
computation based on set type and handles both standard PyTorch models
and ONNX GraphModules.
"""

import logging
import operator
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from n2v.sets import Star, Zono, Box, Hexatope, Octatope
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.layer_ops.linear_reach import linear_hexatope, linear_octatope
from n2v.utils.model_preprocessing import fuse_batchnorm, has_batchnorm
from n2v.utils.bounds_precomputation import compute_intermediate_bounds

# Profiler hooks (no-op when profiling is disabled)
from n2v.profiling import region, PHASE
from n2v.probabilistic.conformal_reach import ConformalReachConfig, conformal_reach
from n2v.probabilistic.flow.reach import FlowReachConfig, flow_reach
from onnx2torch.node_converters.reshape import OnnxReshape
from onnx2torch.node_converters.concat import OnnxConcat
from onnx2torch.node_converters.slice import OnnxSlice, OnnxSliceV9
from onnx2torch.node_converters.split import OnnxSplit, OnnxSplit13
from onnx2torch.node_converters.reduce import (
    OnnxReduceStaticAxes,
    OnnxReduceSumStaticAxes,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses used by :func:`reach_pytorch_model` to bundle
# per-method kwargs (validated via :func:`_validate_reach_config`).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReachConfig:
    """Configuration for sound reachability via :meth:`NeuralNetwork.reach`.

    Used with sound methods ``'exact'`` (Star-only) and ``'approx'``
    (Star, Box, Zono, Hexatope, Octatope). Probabilistic methods use
    their own config classes: ``'flow_matching'`` -> ``FlowReachConfig``
    and ``'conformal'`` -> ``ConformalReachConfig``.

    Attributes:
        method: ``'exact'`` for exact star reachability with splitting
            (sound, complete on piecewise-linear nets) or ``'approx'``
            for over-approximate relaxation reachability (sound, sound-
            but-incomplete). Default ``'exact'``.
        lp_solver: LP solver identifier passed to the underlying reach
            kernels. ``'default'`` defers to the kernel's choice.
        verbose: ``'display'`` to print progress, ``None`` to stay quiet.
        parallel: Enable parallel Star processing (only relevant for
            ``method='exact'`` on Star sets).
        n_workers: Number of parallel workers when ``parallel=True``.
            Must be >= 1.
        relax_factor: Relaxation factor in ``[0, 1]`` for ``method='approx'``;
            ``0`` is exact (no relaxation), ``1`` is maximal relaxation.
            Default ``0.5``. Ignored for ``method='exact'`` (a warning is
            emitted if set to a non-default with ``method='exact'``).
        relax_method: Relaxation strategy for ``method='approx'``. Default
            ``'standard'``. Ignored for ``method='exact'``.

    Example:
        >>> # Pass directly to .reach():
        >>> output_stars = net.reach(input_star, config=ReachConfig(
        ...     method='exact', parallel=True, n_workers=8,
        ... ))
        >>> # Or equivalently via bare kwargs (validator builds the config):
        >>> output_stars = net.reach(input_star, method='exact',
        ...                          parallel=True, n_workers=8)
    """

    method: Literal['exact', 'approx'] = 'exact'
    lp_solver: str = 'default'
    verbose: Optional[str] = None
    parallel: bool = False
    n_workers: int = 1
    relax_factor: float = 0.5
    relax_method: str = 'standard'

    def __post_init__(self):
        if self.method not in ('exact', 'approx'):
            raise ValueError(
                f"ReachConfig.method must be 'exact' or 'approx', "
                f"got {self.method!r}"
            )
        if self.method == 'exact' and (
            self.relax_factor != 0.5 or self.relax_method != 'standard'
        ):
            warnings.warn(
                "ReachConfig.relax_factor / relax_method are ignored for "
                "method='exact'",
                stacklevel=2,
            )
        if self.n_workers < 1:
            raise ValueError(
                f"ReachConfig.n_workers must be >= 1, got {self.n_workers}"
            )
        if not 0.0 <= self.relax_factor <= 1.0:
            raise ValueError(
                f"ReachConfig.relax_factor must be in [0, 1], "
                f"got {self.relax_factor}"
            )


def _validate_reach_config(method, config, **kwargs):
    """Reconcile ``config=`` and bare kwargs for ``.reach()`` dispatch.

    The dispatch surface allows two styles equivalently:

      * ``net.reach(box, method='exact', parallel=True, n_workers=8)``
      * ``net.reach(box, config=ReachConfig(method='exact', parallel=True,
                                            n_workers=8))``

    This helper produces a validated config from either form; the two
    forms are mutually exclusive (passing both raises ``TypeError``).

    Args:
        method: The ``method=`` argument to ``.reach()``.
        config: An optional dataclass instance — ``ReachConfig`` for
            sound methods (``'exact'``, ``'approx'``); ``FlowReachConfig``
            for ``'flow_matching'``.
        **kwargs: Method-specific keyword arguments. Empty when the
            caller uses ``config=``.

    Returns:
        A validated config instance of the type appropriate for ``method``.

    Raises:
        TypeError: If both ``config=`` and ``kwargs`` are passed.
        TypeError: If ``config``'s type doesn't match ``method``.
        TypeError: If ``config.method`` disagrees with the ``method`` arg.
        TypeError: If kwargs contain unknown keys for the chosen config.
        ValueError: If ``method`` is not a known reach method.
    """
    if config is not None and kwargs:
        raise TypeError(
            "pass either config= or method-specific kwargs, not both "
            f"(got config={type(config).__name__} and kwargs={list(kwargs)})"
        )

    if method in ('exact', 'approx'):
        expected_cls = ReachConfig
    elif method == 'flow_matching':
        expected_cls = FlowReachConfig
    elif method == 'conformal':
        expected_cls = ConformalReachConfig
    else:
        raise ValueError(
            f"unknown reach method: {method!r}. "
            f"Known: 'exact', 'approx', 'flow_matching', 'conformal'."
        )

    if config is None:
        # Build from kwargs. Unknown keys raise ``TypeError`` via the
        # dataclass constructor — desirable: typo'd kwargs fail loudly.
        if 'method' in expected_cls.__dataclass_fields__:
            kwargs.setdefault('method', method)
            if kwargs['method'] != method:
                raise TypeError(
                    f"method in kwargs ({kwargs['method']!r}) disagrees "
                    f"with method= argument ({method!r})"
                )
        return expected_cls(**kwargs)

    if not isinstance(config, expected_cls):
        raise TypeError(
            f"method={method!r} expects {expected_cls.__name__}, "
            f"got {type(config).__name__}"
        )

    if hasattr(config, 'method') and config.method != method:
        raise TypeError(
            f"config.method ({config.method!r}) disagrees with "
            f"method= argument ({method!r})"
        )

    return config

# Maps torch functional ops to their nn.Module equivalents.
#   Used by _function_node_to_module to convert call_function
#   fx nodes into modules for the reachability dispatcher.
FUNCTION_TO_MODULE_CLS: dict[type, type[nn.Module]] = {
    F.relu: nn.ReLU,
    torch.relu: nn.ReLU,
    torch.sigmoid: nn.Sigmoid,
    torch.tanh: nn.Tanh,
}


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
            - 'exact': Exact reachability (Star only)
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

    if method == 'flow_matching':
        # Flow-matching probabilistic reach: trains a flow on outputs
        # sampled from the input box, calibrates a conformal threshold,
        # and returns a ProbabilisticSet. ``flow_reach`` accepts both
        # ``nn.Module`` and numpy-callable inputs; we pass the module
        # directly so its forward pass stays device-aware (preserves
        # GPU acceleration where applicable).
        if not isinstance(input_set, Box):
            raise TypeError(
                f"method='flow_matching' requires Box input, "
                f"got {type(input_set).__name__}"
            )
        config = kwargs.pop('config', None)
        config = _validate_reach_config(method, config, **kwargs)
        return flow_reach(model, input_set, config)

    if method == 'conformal':
        # Surrogate-based conformal reach. ``conformal_reach`` accepts
        # both ``nn.Module`` and numpy-callable inputs; we pass the
        # module directly so its internal device-aware wrapping kicks in.
        if not isinstance(input_set, Box):
            raise TypeError(
                f"method='conformal' requires Box input, "
                f"got {type(input_set).__name__}"
            )
        config = kwargs.pop('config', None)
        config = _validate_reach_config(method, config, **kwargs)
        return conformal_reach(model, input_set, config)

    # Sound dispatch ('exact' / 'approx'): if the caller passed
    # ``config=ReachConfig(...)`` (the documented form), validate it and
    # unpack its fields back into ``kwargs`` so the downstream
    # ``_handle_graphmodule`` -> ``reach_layer`` chain consumes them as
    # before. Without this branch the ``config=`` form silently no-op'd
    # on sound methods while the documented public API claimed it worked.
    if 'config' in kwargs:
        config = kwargs.pop('config')
        # Pass remaining kwargs so ``_validate_reach_config`` enforces
        # the documented "config XOR kwargs" rule.
        config = _validate_reach_config(method, config, **kwargs)
        kwargs = {
            'lp_solver': config.lp_solver,
            'verbose': config.verbose,
            'parallel': config.parallel,
            'n_workers': config.n_workers,
            'relax_factor': config.relax_factor,
            'relax_method': config.relax_method,
        }

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
        if method != 'approx':
            raise ValueError(
                f"{type(input_set).__name__} only supports 'approx', got '{method}'"
            )
    else:
        raise TypeError(
            f"Unsupported input set type: {type(input_set).__name__}. "
            f"Supported types: Star, ImageStar, Zono, Box, Hexatope, Octatope"
        )

    # Trace non-GraphModule models with torch.fx
    if not isinstance(model, fx.GraphModule):
        try:
            model = torch.fx.symbolic_trace(model)
        except Exception as e:
            raise TypeError(
                f"n2v requires models to be traceable by torch.fx. "
                f"Models with data-dependent control flow (e.g., "
                f"'if x.sum() > 0') or inline module instantiation "
                f"(e.g., 'nn.ReLU()(x)') are not supported. "
                f"For inline activations, use functional equivalents "
                f"(e.g., F.relu(x) instead of nn.ReLU()(x)). "
                f"Tracing failed with: {e}"
            ) from e

    with region("reach", PHASE):
        return _handle_graphmodule(model, [input_set], method, **kwargs)


def _function_node_to_module(
    node: fx.Node,
) -> Optional[nn.Module]:
    """Convert a call_function fx node to an nn.Module equivalent.

    Args:
        node: torch.fx.Node with op == 'call_function'.

    Returns:
        Equivalent nn.Module, or None if the function is not
        a dispatchable activation or operation.
    """
    fn = node.target

    if fn in FUNCTION_TO_MODULE_CLS:
        return FUNCTION_TO_MODULE_CLS[fn]()

    # Parameterized functions
    if fn is F.leaky_relu:
        slope = node.kwargs.get('negative_slope', 0.01)
        if (len(node.args) > 1
                and not isinstance(node.args[1], fx.Node)):
            slope = node.args[1]
        return nn.LeakyReLU(negative_slope=slope)

    if fn is torch.flatten:
        start_dim = node.kwargs.get('start_dim', 1)
        end_dim = node.kwargs.get('end_dim', -1)
        if (len(node.args) > 1
                and not isinstance(node.args[1], fx.Node)):
            start_dim = node.args[1]
        if (len(node.args) > 2
                and not isinstance(node.args[2], fx.Node)):
            end_dim = node.args[2]
        return nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    return None


def _handle_graphmodule(
    graph_module: fx.GraphModule,
    input_sets: List,
    method: str,
    **kwargs: Any
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
    # Handle precompute_bounds option ('ibp', 'zono', True, or False)
    precompute = kwargs.pop('precompute_bounds', False)
    layer_bounds = kwargs.pop('_precomputed_layer_bounds', None)

    # Best-effort tensor-shape annotation (for shape-dependent exact ops:
    # Gather, Slice, Transpose on flat sets). Absence of a shape makes
    # those ops raise loudly — never guess.
    input_shape = kwargs.pop('input_shape', None)
    node_shapes = _propagate_shapes(graph_module, input_sets, input_shape)
    const_names = _constant_node_names(graph_module)

    if precompute and layer_bounds is None:
        precompute_method = precompute if isinstance(precompute, str) else 'ibp'
        layer_bounds = compute_intermediate_bounds(graph_module, input_sets[0], method=precompute_method)

    # Get the set type from the first input
    set_type = type(input_sets[0])

    # Get named modules
    named_modules = dict(graph_module.named_modules())

    # Store intermediate values for each node
    node_values = _new_node_values()
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
                logger.debug(f'  Processing: {node.target} ({module_type})')

            # Handle OnnxReshape
            if isinstance(module, OnnxReshape):
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
            if isinstance(module, OnnxConcat):
                result_sets = _handle_onnx_concat(
                    module, node, node_values, node_shapes)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            # Handle OnnxSlice
            if isinstance(module, (OnnxSlice, OnnxSliceV9)):
                result_sets = _handle_onnx_slice(module, node, node_values,
                                                 graph_module, node_shapes)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            # Handle OnnxGather (constant indices: exact row selection)
            if module_type == 'OnnxGather':
                result_sets = _handle_onnx_gather(
                    module, node, node_values, graph_module, node_shapes)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            # Handle OnnxTranspose on flat sets (shape-aware exact path;
            # ImageStar/ImageZono fall through to the dispatcher)
            if module_type == 'OnnxTranspose':
                result_sets = _handle_onnx_transpose_flat(
                    module, node, node_values, node_shapes)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            # OnnxPadDynamic carries its pads as a runtime input (not an
            # attribute); resolve them from node.args[1] and stash on the
            # module so the pad handler can read them. Without this the
            # handler would silently apply zero padding (wrong output).
            if module_type == 'OnnxPadDynamic' and not hasattr(
                    module, 'pads'):
                if len(node.args) > 1:
                    module.pads = _get_parameter(
                        graph_module, node.args[1]).numpy().astype(
                            int).tolist()
                # a 3rd input (constant pad value) other than 0 is not a
                # zero-pad and would be unsound to treat as one
                if len(node.args) > 2:
                    try:
                        pv = _get_parameter(
                            graph_module, node.args[2]).numpy()
                        if np.any(np.asarray(pv) != 0):
                            raise NotImplementedError(
                                "Pad with non-zero constant value is not "
                                "supported")
                    except (KeyError, AttributeError):
                        pass

            # Handle OnnxSplit / OnnxSplit13
            if isinstance(module, (OnnxSplit, OnnxSplit13)):
                result = _handle_onnx_split(
                    module, node, node_values, graph_module, node_shapes)
                if result is not None:
                    node_values[node.name] = result
                    # Don't set current_sets — outputs are extracted by getitem
                    continue

            # Exact shape-aware ReduceSum/ReduceMean on flat sets;
            # ImageStar/ImageZono and non-affine reduces (max/min/prod)
            # fall through to the dispatcher
            if isinstance(module, (OnnxReduceStaticAxes,
                                   OnnxReduceSumStaticAxes)):
                result_sets = _handle_onnx_reduce_flat(
                    module, node, node_values, node_shapes)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            # Pow(x, c): constant-exponent power. The exponent lives in
            # node.args[1], so it must be read here rather than in the
            # generic dispatcher.
            if module_type == 'OnnxPow':
                current_sets = _handle_onnx_pow(
                    module, node, node_values, graph_module, kwargs)
                if current_sets is not None:
                    node_values[node.name] = current_sets
                    continue

            # Handle ONNX-specific operations
            if module_type == 'OnnxBinaryMathOperation':
                current_sets = _handle_onnx_binary_op(
                    module, node, node_values, graph_module, set_type,
                    node_shapes,
                )
                if current_sets is not None:
                    node_values[node.name] = current_sets
                    continue

            elif module_type == 'OnnxMatMul':
                current_sets = _handle_onnx_matmul(
                    module, node, node_values, graph_module, set_type,
                    node_shapes,
                )
                if current_sets is not None:
                    node_values[node.name] = current_sets
                    continue

            # Standard PyTorch layer - use dispatcher
            if node.args and len(node.args) > 0:
                first_arg = node.args[0]
                if hasattr(first_arg, 'name') and first_arg.name in node_values:
                    input_sets_op = node_values[first_arg.name]
                    layer_kwargs = dict(kwargs)
                    if layer_bounds is not None and node.name in layer_bounds:
                        layer_kwargs['precomputed_bounds'] = layer_bounds[node.name]
                    else:
                        layer_kwargs.pop('precomputed_bounds', None)
                    output_sets = reach_layer(module, input_sets_op, method, **layer_kwargs)
                    node_values[node.name] = output_sets
                    current_sets = output_sets

            # No silent skips: a call_module node that produced no sets
            # would poison every downstream consumer with STALE sets
            # (current_sets from an unrelated branch) — the recipe for a
            # silently wrong reach result. Constants carry no set
            # semantics and are consumed via _get_parameter.
            if node.name not in node_values and node.name not in const_names:
                raise NotImplementedError(
                    f"graph node {node.name!r} ({module_type}) was not "
                    f"handled by any reachability path"
                )

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
            else:
                # Try to convert function to module equivalent
                equiv_module = _function_node_to_module(node)
                if equiv_module is not None:
                    # Resolve input from first argument
                    first_arg = node.args[0]
                    if hasattr(first_arg, 'name') and first_arg.name in node_values:
                        input_sets_op = node_values[first_arg.name]
                    else:
                        input_sets_op = current_sets

                    layer_kwargs = dict(kwargs)
                    if layer_bounds is not None and node.name in layer_bounds:
                        layer_kwargs['precomputed_bounds'] = layer_bounds[node.name]
                    else:
                        layer_kwargs.pop('precomputed_bounds', None)

                    output_sets = reach_layer(
                        equiv_module, input_sets_op,
                        method, **layer_kwargs,
                    )
                    node_values[node.name] = output_sets
                    current_sets = output_sets

                elif verbose:
                    logger.debug(
                        f'  Skipping call_function: '
                        f'{node.target}'
                    )

        elif node.op == 'call_method':
            method_name = node.target  # e.g., 'flatten', 'view', 'reshape'

            # Resolve input from first argument (the tensor/self)
            first_arg = node.args[0]
            if hasattr(first_arg, 'name') and first_arg.name in node_values:
                input_sets_op = node_values[first_arg.name]
            else:
                input_sets_op = current_sets

            if method_name == 'flatten':
                start_dim = (
                    node.args[1]
                    if len(node.args) > 1
                    else node.kwargs.get('start_dim', 1)
                )
                end_dim = (
                    node.args[2]
                    if len(node.args) > 2
                    else node.kwargs.get('end_dim', -1)
                )
                equiv_module = nn.Flatten(
                    start_dim=start_dim, end_dim=end_dim,
                )

                layer_kwargs = dict(kwargs)
                layer_kwargs.pop('precomputed_bounds', None)
                output_sets = reach_layer(
                    equiv_module, input_sets_op,
                    method, **layer_kwargs,
                )
                node_values[node.name] = output_sets
                current_sets = output_sets

            elif method_name in ('view', 'reshape'):
                shape_args = node.args[1:]
                if (len(shape_args) == 1
                        and isinstance(
                            shape_args[0], (tuple, list),
                        )):
                    target_shape = tuple(shape_args[0])
                else:
                    target_shape = tuple(shape_args)
                result_sets = _handle_reshape(input_sets_op, target_shape)
                node_values[node.name] = result_sets
                current_sets = result_sets

            elif verbose:
                logger.debug(f'  Skipping call_method: {method_name}')

        elif node.op == 'output':
            # Output node - extract final result
            if node.args and len(node.args) > 0:
                output_node = node.args[0]
                if hasattr(output_node, 'name') and output_node.name in node_values:
                    current_sets = node_values[output_node.name]

    return current_sets


def _new_node_values() -> Dict[str, List]:
    """Factory for the executor's per-node set table. The ONNX-oracle
    debugging harness swaps this for a recording dict to trace each
    node's reach sets; production behavior is a plain dict."""
    return {}


def _constant_node_names(graph_module: fx.GraphModule) -> set:
    """Names of nodes whose values depend on NO placeholder — constant
    subgraphs (weights, shapes, constant arithmetic). They carry no set
    semantics: consumers fold them via _const_eval."""
    const = set()
    for n in graph_module.graph.nodes:
        if n.op == 'placeholder' or n.op == 'output':
            continue
        if n.op == 'get_attr':
            const.add(n.name)
            continue
        deps = [a for a in list(n.args) + list(n.kwargs.values())
                if hasattr(a, 'op')]
        if all(d.name in const or d.op == 'get_attr' for d in deps) \
                and n.op in ('call_module', 'call_function', 'call_method'):
            const.add(n.name)
    return const


def _propagate_shapes(graph_module: fx.GraphModule, input_sets: List,
                      input_shape=None) -> Dict[str, Tuple[int, ...]]:
    """Annotate fx nodes with tensor shapes via torch.fx ShapeProp.

    Runs the model's REAL forward on a dummy input — a wrong dummy shape
    crashes the propagation and we return {} (consumers then raise when
    they need a shape). ``input_shape`` is the batch-stripped model input
    shape (threaded from the runner); heuristic fallback otherwise.
    """
    try:
        from torch.fx.passes.shape_prop import ShapeProp
        if input_shape is not None:
            dummy_shape = (1, *tuple(int(d) for d in input_shape))
        else:
            s0 = input_sets[0]
            if isinstance(s0, ImageStar):
                dummy_shape = (1, s0.num_channels, s0.height, s0.width)
            else:
                dummy_shape = (1, int(s0.dim))
        ShapeProp(graph_module).propagate(torch.zeros(dummy_shape))
        shapes = {}
        for n in graph_module.graph.nodes:
            tm = n.meta.get('tensor_meta')
            if tm is not None and hasattr(tm, 'shape'):
                shapes[n.name] = tuple(int(x) for x in tm.shape)
        return shapes
    except Exception:  # noqa: BLE001 — absence is handled loudly downstream
        return {}


def _handle_onnx_reduce_flat(
    module: Any,
    node: Any,
    node_values: Dict[str, List],
    node_shapes: Optional[Dict[str, Tuple[int, ...]]],
) -> Optional[List]:
    """
    Exact shape-aware ReduceSum/ReduceMean on flat sets.

    Summation over axes is the affine map A with A[j, i] = 1 exactly
    when flat input position i collapses onto flat output position j
    (mean divides by the reduced count). keepdims only changes shape
    metadata, not the flat element order, so the same map serves both.

    Returns None (fall through to the dispatcher) for ImageStar/
    ImageZono inputs, missing shape metadata, or non-affine reduce ops.
    """
    first_arg = node.args[0] if node.args else None
    if not (hasattr(first_arg, 'name') and first_arg.name in node_values):
        return None
    input_sets = node_values[first_arg.name]
    if not input_sets or not isinstance(input_sets[0], (Star, Zono, Box)):
        return None
    in_shape = (node_shapes or {}).get(first_arg.name)
    if in_shape is None or len(in_shape) == 0:
        return None

    if isinstance(module, OnnxReduceSumStaticAxes):
        op = 'ReduceSum'
        axes = module._axes
        if (axes is None or len(axes) == 0) \
                and getattr(module, '_noop_with_empty_axes', 0):
            return list(input_sets)
    elif isinstance(module, OnnxReduceStaticAxes):
        op = module.operation_type
        axes = module.axes
    else:
        return None
    if op not in ('ReduceSum', 'ReduceMean'):
        return None

    rank = len(in_shape)
    if axes is None or len(axes) == 0:
        axes = list(range(rank))
    axes = sorted({int(a) % rank for a in axes})
    size = int(np.prod(in_shape))
    if size != input_sets[0].dim:
        raise ValueError(
            f"shape metadata for {first_arg.name!r} says {tuple(in_shape)} "
            f"({size} elements) but the set has dim {input_sets[0].dim} — "
            f"an upstream op mis-sized its output")

    kept_shape = tuple(1 if i in axes else s for i, s in enumerate(in_shape))
    coords = list(np.indices(in_shape))
    for ax in axes:
        coords[ax] = np.zeros_like(coords[ax])
    out_pos = np.ravel_multi_index(coords, kept_shape).flatten()
    out_size = int(np.prod(kept_shape))
    A = np.zeros((out_size, size), dtype=np.float64)
    A[out_pos, np.arange(size)] = 1.0
    if op == 'ReduceMean':
        A /= float(np.prod([in_shape[i] for i in axes]))
    return [s.affine_map(A) for s in input_sets]


def _select_rows(input_sets: List, rows: np.ndarray, expected_size: int) -> List:
    """Apply an exact row selection (from an index-tensor mapping) to a
    list of sets. Cross-checks the mapped size against each set's actual
    dimension — a mismatch means the shape metadata is wrong for this
    set, and we raise rather than silently mis-select."""
    output = []
    for s in input_sets:
        if isinstance(s, ImageStar):
            s = _reshape_imagestar(s, (s.height * s.width * s.num_channels,))
        if isinstance(s, ImageZono):
            s = _reshape_imagezono(s, (s.height * s.width * s.num_channels,))
        dim = s.dim
        if dim != expected_size:
            raise ValueError(
                f"shape metadata says {expected_size} elements but the set "
                f"has dim {dim} — refusing to apply index selection"
            )
        if isinstance(s, Star):
            output.append(Star(s.V[rows, :], s.C, s.d,
                               s.predicate_lb, s.predicate_ub))
        elif isinstance(s, Zono):
            output.append(Zono(s.c[rows, :], s.V[rows, :]))
        elif isinstance(s, Box):
            output.append(Box(s.lb[rows, :], s.ub[rows, :]))
        else:
            raise NotImplementedError(
                f"index selection not supported for {type(s).__name__}")
    return output


def _handle_onnx_gather(module, node, node_values, graph_module,
                        node_shapes) -> Optional[List]:
    """ONNX Gather with CONSTANT indices = exact row selection."""
    if len(node.args) != 2:
        return None
    data_node, idx_node = node.args
    if not (hasattr(data_node, 'name') and data_node.name in node_values):
        return None
    try:
        indices = _get_parameter(graph_module, idx_node).numpy().astype(np.int64)
    except Exception:  # noqa: BLE001
        raise NotImplementedError(
            "Gather with non-constant indices is not supported")
    in_shape = node_shapes.get(data_node.name)
    if in_shape is None:
        raise NotImplementedError(
            "Gather requires tensor-shape tracking (ShapeProp could not "
            "annotate this model)")
    axis = int(getattr(module, '_axis', 0)) % len(in_shape)
    size = int(np.prod(in_shape))
    rows = np.take(np.arange(size).reshape(in_shape), indices,
                   axis=axis).flatten()
    return _select_rows(node_values[data_node.name], rows, size)


def _handle_onnx_transpose_flat(module, node, node_values,
                                node_shapes) -> Optional[List]:
    """Shape-aware transpose for FLAT sets (ImageStar/ImageZono are
    handled exactly in the dispatcher): with the tensor shape known, an
    axis permutation is an exact row permutation."""
    data_node = node.args[0]
    if not (hasattr(data_node, 'name') and data_node.name in node_values):
        return None
    input_sets = node_values[data_node.name]
    if input_sets and isinstance(input_sets[0], (ImageStar, ImageZono)):
        return None  # dispatcher path is exact for these
    in_shape = node_shapes.get(data_node.name)
    if in_shape is None:
        return None  # dispatcher's loud guards take over
    perm = module.perm
    if perm is None:
        perm = list(range(len(in_shape)))[::-1]
    perm = [int(p) for p in perm]
    size = int(np.prod(in_shape))
    rows = np.transpose(np.arange(size).reshape(in_shape), perm).flatten()
    return _select_rows(input_sets, rows, size)


def _get_parameter(graph_module: fx.GraphModule, node: Any) -> torch.Tensor:
    """Extract a constant tensor from a get_attr node, an OnnxConstant
    call_module, or a constant SUBGRAPH (ops applied only to constants,
    e.g. Transpose of a weight) — folded recursively."""
    return _const_eval(graph_module, node, {})


def _const_eval(graph_module: fx.GraphModule, node: Any, memo: dict) -> torch.Tensor:
    if not hasattr(node, 'op'):
        return torch.tensor(node)  # literal arg
    if node.name in memo:
        return memo[node.name]
    if node.op == 'get_attr':
        obj = graph_module
        for attr in node.target.split('.'):
            obj = getattr(obj, attr)
        result = obj.detach().cpu()
    elif node.op == 'call_module':
        module = dict(graph_module.named_modules()).get(node.target)
        if module is not None and hasattr(module, 'value'):
            result = module.value.detach().cpu()
        elif module is not None:
            args = [_const_eval(graph_module, a, memo) for a in node.args]
            with torch.no_grad():
                result = module(*args).detach().cpu()
        else:
            raise ValueError(f"not a constant subgraph: {node.name}")
    elif node.op == 'call_function':
        args = [_const_eval(graph_module, a, memo) for a in node.args]
        with torch.no_grad():
            result = node.target(*args, **node.kwargs)
        result = result.detach().cpu() if isinstance(result, torch.Tensor) \
            else torch.tensor(result)
    else:
        raise ValueError(f"not a constant node: {node.name} ({node.op})")
    memo[node.name] = result
    return result


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


def _resolve_shape(shape: Tuple[int, ...], total_size: int) -> Tuple[int, ...]:
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


def _reshape_imagestar(star: 'ImageStar', spatial_shape: Tuple[int, ...]) -> Union['Star', 'ImageStar']:
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


def _reshape_imagezono(zono: 'ImageZono', spatial_shape: Tuple[int, ...]) -> Union['Zono', 'ImageZono']:
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


def _reshape_star(star: 'Star', spatial_shape: Tuple[int, ...]) -> Union['Star', 'ImageStar']:
    """Reshape plain Star.

    Flat-Star rows are in ONNX (CHW-flattened) order — the invariant
    established by :func:`_reshape_imagestar`'s flatten branch. A 3-D
    target ``(C, H, W)`` therefore produces an ImageStar by the exact
    inverse permutation (CHW -> internal HWC storage). A reshape is a
    bijection on tensor entries, so permuting V's rows is exact:
    ``x = c + V·a`` implies ``reshape(x) = reshape(c) + reshape(V)·a``;
    constraints are untouched.

    1-D targets are the identity on a flat vector (no-op). 2-D targets
    are reinterpretation-only and stay a validated no-op (no spatial
    consumer exists for 2-D star layouts).
    """
    total = star.dim
    resolved = _resolve_shape(spatial_shape, total)
    product = 1
    for s in resolved:
        product *= s
    if product != total:
        raise ValueError(
            f"Cannot reshape Star of dim {total} to shape {resolved}"
        )

    if len(resolved) == 3:
        c_out, h_out, w_out = resolved
        n_cols = star.V.shape[1]
        V_chw = star.V.reshape(c_out, h_out, w_out, n_cols)
        V_hwc = np.transpose(V_chw, (1, 2, 0, 3))  # (H, W, C, nVar+1)
        return ImageStar(
            V_hwc, star.C, star.d, star.predicate_lb, star.predicate_ub,
            h_out, w_out, c_out
        )

    return star


def _chw_to_hwc_perm(c: int, h: int, w: int) -> np.ndarray:
    """Row permutation taking CHW-flat ordering to HWC-flat ordering."""
    return np.transpose(
        np.arange(c * h * w).reshape(c, h, w), (1, 2, 0)).flatten()


def _reshape_zono(zono: 'Zono', spatial_shape: Tuple[int, ...]) -> Union['Zono', 'ImageZono']:
    """Reshape plain Zono. Same convention and exactness argument as
    :func:`_reshape_star`; ImageZono stores flat HWC-ordered arrays."""
    total = zono.dim
    resolved = _resolve_shape(spatial_shape, total)
    product = 1
    for s in resolved:
        product *= s
    if product != total:
        raise ValueError(
            f"Cannot reshape Zono of dim {total} to shape {resolved}"
        )

    if len(resolved) == 3:
        c_out, h_out, w_out = resolved
        perm = _chw_to_hwc_perm(c_out, h_out, w_out)
        return ImageZono(zono.c[perm, :], zono.V[perm, :],
                         h_out, w_out, c_out)

    return zono


def _reshape_box(box: 'Box', spatial_shape: Tuple[int, ...]) -> 'Box':
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


def _handle_onnx_binary_op(
    module: Any,
    node: Any,
    node_values: Dict[str, List],
    graph_module: fx.GraphModule,
    set_type: Type,
    node_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Optional[List]:
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

        # ONNX broadcasting between two computed sets (e.g. (1,3,6) op
        # (1,3,1)): expand each operand to the broadcast shape via exact
        # row selection, then apply element-wise.
        shape_a = (node_shapes or {}).get(getattr(first_input, 'name', None))
        shape_b = (node_shapes or {}).get(getattr(second_input, 'name', None))
        if shape_a is not None and shape_b is not None \
                and tuple(shape_a) != tuple(shape_b):
            out_shape = np.broadcast_shapes(tuple(shape_a), tuple(shape_b))

            def _expand(sets_x, shape_x):
                size = int(np.prod(shape_x))
                if int(np.prod(out_shape)) == size \
                        and sets_x and sets_x[0].dim == size:
                    return sets_x  # same flat order, no expansion needed
                rows = np.broadcast_to(
                    np.arange(size).reshape(shape_x), out_shape).flatten()
                return _select_rows(sets_x, rows, size)

            sets_a = _expand(sets_a, shape_a)
            sets_b = _expand(sets_b, shape_b)

        if op_name == 'mul':
            return _mul_sets(sets_a, sets_b)
        elif op_name == '_onnx_div':
            return _div_sets(sets_a, sets_b)

        return _add_sets(sets_a, sets_b, op_name)

    # Case 2: one input is a parameter (get_attr / OnnxConstant), the
    # other a computed set. ONNX allows either order: Add(c, x) is
    # Add(x, c); Sub(c, x) = -(x) + c; Mul(c, x) = Mul(x, c).
    mirrored = False
    if not (hasattr(first_input, 'name')
            and first_input.name in node_values):
        if (hasattr(second_input, 'name')
                and second_input.name in node_values):
            first_input, second_input = second_input, first_input
            mirrored = True
        else:
            return None
    try:
        param_value = _get_parameter(
            graph_module, second_input).numpy().astype(np.float64)
    except Exception:  # noqa: BLE001
        return None

    if mirrored and hasattr(module, 'math_op_function'):
        op = module.math_op_function.__name__
        if 'sub' in op:
            # param - x = (-x) + param
            negated = _mul_sets_by_constant(
                node_values[first_input.name], np.array(-1.0))
            saved = node_values[first_input.name]
            node_values[first_input.name] = negated
            try:
                # re-dispatch as (neg_x) ADD param
                bias = param_value.flatten()

                def _bcast(b, dim):
                    if b.size != dim and dim % max(b.size, 1) == 0:
                        return np.tile(b, dim // b.size)
                    return b

                out = []
                for s in negated:
                    if isinstance(s, ImageStar):
                        h, w, ch = s.height, s.width, s.num_channels
                        if bias.size == h * w * ch:
                            bhwc = bias.reshape(ch, h, w).transpose(1, 2, 0)
                        elif bias.size == ch:
                            bhwc = np.broadcast_to(
                                bias.reshape(1, 1, ch), (h, w, ch))
                        else:
                            bhwc = np.full((h, w, ch), bias.flatten()[0])
                        V = s.V.copy()
                        V[:, :, :, 0] = V[:, :, :, 0] + bhwc
                        out.append(ImageStar(V, s.C, s.d, s.predicate_lb,
                                             s.predicate_ub, h, w, ch))
                    elif isinstance(s, Star):
                        b = _bcast(bias, s.dim)
                        V = s.V.copy()
                        V[:, 0:1] = V[:, 0:1] + b.reshape(-1, 1)
                        out.append(Star(V, s.C, s.d,
                                        s.predicate_lb, s.predicate_ub))
                    elif isinstance(s, ImageZono):
                        h, w, ch = s.height, s.width, s.num_channels
                        if bias.size == h * w * ch:
                            bf = bias.reshape(ch, h, w).transpose(
                                1, 2, 0).reshape(-1, 1)
                        else:
                            bf = _bcast(bias, s.c.size).reshape(-1, 1)
                        out.append(ImageZono(s.c + bf, s.V, h, w, ch))
                    elif isinstance(s, Zono):
                        b = _bcast(bias, s.c.size).reshape(-1, 1)
                        out.append(Zono(s.c + b, s.V))
                    elif isinstance(s, Box):
                        b = _bcast(bias, s.lb.size)
                        out.append(Box(s.lb + b.reshape(s.lb.shape),
                                       s.ub + b.reshape(s.ub.shape)))
                    else:
                        raise NotImplementedError(
                            f"mirrored Sub not implemented for "
                            f"{type(s).__name__}")
                return out
            finally:
                node_values[first_input.name] = saved
        if '_onnx_div' in op:
            raise NotImplementedError(
                "division BY a computed set is not supported")
        # add / mul are commutative: fall through with swapped operands

    # Get the sets from the first input
    if first_input.name not in node_values:
        return None

    input_sets_op = node_values[first_input.name]

    # ONNX broadcasting: a (M,) parameter applied to a (1, R, M) tensor
    # repeats across leading dims. On the flat set this is a tile —
    # exact via the recorded tensor shape, trailing-axis tile fallback.
    set_dim = input_sets_op[0].dim if input_sets_op else param_value.size
    if param_value.size != set_dim and not isinstance(
            input_sets_op[0] if input_sets_op else None, (ImageStar, ImageZono)):
        in_shape = (node_shapes or {}).get(
            getattr(first_input, 'name', None))
        if in_shape is not None and int(np.prod(in_shape)) != set_dim:
            raise ValueError(
                f"shape metadata for {first_input.name!r} says "
                f"{tuple(in_shape)} but the set has dim {set_dim} — an "
                f"upstream op mis-sized its output")
        if in_shape is not None and param_value.size != 1:
            out_shape = np.broadcast_shapes(in_shape, param_value.shape)
            if int(np.prod(out_shape)) != set_dim:
                # Dim-EXPANDING broadcast, e.g. (1,24,1)+(54,) -> (1,24,54):
                # exact row expansion of the set by the broadcast map.
                rows = np.broadcast_to(
                    np.arange(set_dim).reshape(in_shape), out_shape
                ).flatten()
                input_sets_op = _select_rows(input_sets_op, rows,
                                             set_dim)
                set_dim = int(np.prod(out_shape))
            param_value = np.broadcast_to(
                param_value, out_shape).flatten().copy()
        elif set_dim % max(param_value.size, 1) == 0:
            param_value = np.tile(param_value.flatten(),
                                  set_dim // param_value.size)

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

    elif set_type in (Zono, ImageZono):
        output_sets = []
        bias_reshaped = bias.reshape(-1, 1)
        for s in input_sets_op:
            if isinstance(s, ImageZono):
                h, w, c_ch = s.height, s.width, s.num_channels
                if bias_reshaped.size == c_ch:
                    bias_flat = np.tile(bias.flatten(), h * w).reshape(-1, 1)
                else:
                    bias_flat = bias_reshaped
                new_c = s.c + bias_flat
                out = ImageZono(new_c, s.V, h, w, c_ch)
            else:
                new_c = s.c + bias_reshaped
                out = Zono(new_c, s.V)
            output_sets.append(out)
        return output_sets

    elif set_type == Box:
        output_sets = []
        bias_reshaped = bias.reshape(-1, 1)
        for s in input_sets_op:
            out = Box(s.lb + bias_reshaped, s.ub + bias_reshaped)
            output_sets.append(out)
        return output_sets

    elif set_type == ImageStar:
        # set_type reflects the graph's INPUT set; mid-graph sets can
        # be flat Stars (e.g. after a flatten), so dispatch per element.
        output_sets = []
        b = np.asarray(bias, dtype=np.float64).flatten()
        for s in input_sets_op:
            if isinstance(s, ImageStar):
                # Translation moves the center plane of the 4-D V.
                h, w, c_ch = s.height, s.width, s.num_channels
                if b.size == 1:
                    bias_hwc = np.full((h, w, c_ch), b[0])
                elif b.size == h * w * c_ch:
                    # ONNX constant is (C, H, W) flat; permute to HWC
                    bias_hwc = b.reshape(c_ch, h, w).transpose(1, 2, 0)
                elif b.size == c_ch:
                    bias_hwc = np.broadcast_to(
                        b.reshape(1, 1, c_ch), (h, w, c_ch))
                else:
                    raise NotImplementedError(
                        f"ImageStar add/sub: constant of size {b.size} "
                        f"does not broadcast against (C,H,W)="
                        f"({c_ch},{h},{w})")
                new_V = s.V.copy()
                new_V[:, :, :, 0] = new_V[:, :, :, 0] + bias_hwc
                output_sets.append(ImageStar(
                    new_V, s.C, s.d, s.predicate_lb, s.predicate_ub,
                    h, w, c_ch))
            elif isinstance(s, Star):
                if b.size not in (1, s.dim):
                    raise NotImplementedError(
                        f"add/sub: constant of size {b.size} does not "
                        f"broadcast against flat Star of dim {s.dim}")
                new_V = s.V.copy()
                new_V[:, 0] = new_V[:, 0] + b
                output_sets.append(Star(
                    new_V, s.C, s.d, s.predicate_lb, s.predicate_ub))
            else:
                raise NotImplementedError(
                    f"add/sub not supported for {type(s).__name__} in an "
                    f"ImageStar graph")
        return output_sets

    else:
        raise NotImplementedError(
            f"ONNX binary operations not supported for {set_type.__name__}"
        )


def _handle_onnx_pow(
    module: Any,
    node: Any,
    node_values: Dict[str, List],
    graph_module: fx.GraphModule,
    kwargs: Dict[str, Any],
) -> Optional[List]:
    """Handle Pow(x, c) with a constant scalar exponent.

    The base (node.args[0]) is a computed set; the exponent
    (node.args[1]) is a constant. Dispatches to pow_reach by the actual
    set type. Raises (rather than silently skipping) on the unsupported
    base**x form or a non-uniform / non-integer exponent.
    """
    from n2v.nn.layer_ops import pow_reach

    base = node.args[0]
    if not (hasattr(base, 'name') and base.name in node_values):
        return None
    sets = node_values[base.name]
    if not sets:
        return None

    try:
        exp_arr = _get_parameter(
            graph_module, node.args[1]).numpy().astype(np.float64)
    except Exception:  # noqa: BLE001
        raise NotImplementedError(
            "Pow with a non-constant exponent (base**x) is not supported")

    uniq = np.unique(exp_arr)
    if uniq.size != 1:
        raise NotImplementedError(
            f"Pow with a non-uniform exponent {uniq.tolist()} is not "
            f"supported")
    p = float(uniq[0])
    if not p.is_integer() or p < 0:
        raise NotImplementedError(
            f"Pow exponent {p} not supported (only non-negative integers)")
    p = int(p)

    if p == 1:
        return list(sets)
    if p == 0:
        out = []
        for s in sets:
            if isinstance(s, (Star, ImageStar)):
                V = np.zeros_like(s.V)
                if isinstance(s, ImageStar):
                    V[:, :, :, 0] = 1.0
                    out.append(ImageStar(V, s.C, s.d, s.predicate_lb,
                                         s.predicate_ub, s.height,
                                         s.width, s.num_channels))
                else:
                    V[:, 0] = 1.0
                    out.append(Star(V, s.C, s.d, s.predicate_lb,
                                    s.predicate_ub))
            else:
                raise NotImplementedError(
                    "Pow with exponent 0 only supported for star sets")
        return out

    lp_solver = kwargs.get('lp_solver', 'default')
    first = sets[0]
    if isinstance(first, (Star, ImageStar)):
        return pow_reach.pow_star(sets, p, lp_solver=lp_solver)
    if isinstance(first, (Zono, ImageZono)):
        return pow_reach.pow_zono(sets, p)
    if isinstance(first, Box):
        return pow_reach.pow_box(sets, p)
    raise NotImplementedError(
        f"Pow not supported for {type(first).__name__}")


def _handle_onnx_matmul(
    module: Any,
    node: Any,
    node_values: Dict[str, List],
    graph_module: fx.GraphModule,
    set_type: Type,
    node_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Optional[List]:
    """Handle ONNX MatMul operations."""
    input_nodes = node.args
    if len(input_nodes) != 2:
        return None

    first_input, second_input = input_nodes

    # Weight may live in a get_attr parameter OR an OnnxConstant module —
    # _get_parameter handles both. ONNX MatMul(A, B): the constant can be
    # on EITHER side (x @ W, or W @ x).
    param_side = None
    try:
        weight_matrix = _get_parameter(
            graph_module, second_input).numpy().astype(np.float64)
        param_side = 'right'
    except Exception:  # noqa: BLE001
        try:
            weight_matrix = _get_parameter(
                graph_module, first_input).numpy().astype(np.float64)
            param_side = 'left'
            first_input = second_input  # data node
        except Exception:  # noqa: BLE001
            return None

    # Get the sets from the data input
    if not (hasattr(first_input, 'name')
            and first_input.name in node_values):
        return None

    input_sets_op = node_values[first_input.name]

    # ONNX MatMul with a 1-D operand contracts that axis (vector dot).
    # Promote to 2-D — (K,) -> (K, 1) on the right, (K,) -> (1, K) on
    # the left — and let the exact kron paths below handle batching.
    # The flat result is unchanged: squeezing the contracted axis does
    # not reorder the remaining elements.
    while weight_matrix.ndim > 2 and weight_matrix.shape[0] == 1:
        weight_matrix = weight_matrix[0]
    if weight_matrix.ndim > 2:
        raise NotImplementedError(
            f"ONNX MatMul with {weight_matrix.ndim}-D weight of shape "
            f"{weight_matrix.shape} not supported")
    if weight_matrix.ndim == 1:
        weight_matrix = (weight_matrix.reshape(1, -1)
                         if param_side == 'left'
                         else weight_matrix.reshape(-1, 1))

    if param_side == 'left' and weight_matrix.ndim == 2:
        # y = W (M,K) @ x (..., K, N): on the flat set this is
        # kron(W, I_N); N == 1 reduces to plain W. Express it through the
        # shared x @ W' path below via W' = (kron(W, I_N))^T.
        in_shape = (node_shapes or {}).get(
            getattr(first_input, 'name', None))
        K = weight_matrix.shape[1]
        size = input_sets_op[0].dim if input_sets_op else K
        if in_shape is not None:
            size = int(np.prod(in_shape))
        if size % K != 0:
            raise ValueError(
                f"MatMul(W, x) shape mismatch: weight {weight_matrix.shape}"
                f" vs data size {size}")
        N = size // K
        eff = np.kron(weight_matrix, np.eye(N)) if N > 1 else weight_matrix
        weight_matrix = eff.T  # downstream applies x @ W (i.e. W.T @ x)
        node_shapes = None  # batched-right expansion below must not re-fire

    # Batched MatMul: data (1, R, K) @ W (K, M) applies W per row. On the
    # flat set (dim R*K) this is the exact block-diagonal affine map
    # kron(I_R, W^T). R == 1 reduces to the plain case.
    in_shape = (node_shapes or {}).get(getattr(first_input, 'name', None))
    set_dim = input_sets_op[0].dim if input_sets_op else None
    if in_shape is not None and set_dim is not None \
            and int(np.prod(in_shape)) != set_dim:
        raise ValueError(
            f"shape metadata for {first_input.name!r} says "
            f"{tuple(in_shape)} ({int(np.prod(in_shape))} elements) but "
            f"the set has dim {set_dim} — an upstream op mis-sized its "
            f"output")
    if in_shape is not None and weight_matrix.ndim == 2:
        rows = int(np.prod(in_shape)) // int(weight_matrix.shape[0])
        if rows * weight_matrix.shape[0] != int(np.prod(in_shape)):
            raise ValueError(
                f"MatMul shape mismatch: input {in_shape} vs weight "
                f"{weight_matrix.shape}")
        if rows > 1:
            weight_matrix = np.kron(np.eye(rows), weight_matrix.T).T

    # Dispatch on the ACTUAL set type at this node — set_type reflects
    # the graph's input set, and mid-graph sets can be flat Stars after
    # a flatten in an ImageStar graph.
    first_set = input_sets_op[0] if input_sets_op else None
    if isinstance(first_set, (ImageStar, ImageZono)):
        raise NotImplementedError(
            f"ONNX MatMul on spatial {type(first_set).__name__} not "
            f"supported — flatten first")
    if isinstance(first_set, Star):
        set_type = Star
    elif isinstance(first_set, Zono):
        set_type = Zono
    elif isinstance(first_set, Box):
        set_type = Box

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

    elif set_type in (Zono, ImageZono):
        output_sets = []
        for s in input_sets_op:
            output_set = s.affine_map(weight_matrix.T)
            output_sets.append(output_set)
        return output_sets

    elif set_type == Box:
        output_sets = []
        for s in input_sets_op:
            output_set = s.affine_map(weight_matrix.T)
            output_sets.append(output_set)
        return output_sets

    else:
        raise NotImplementedError(
            f"ONNX MatMul not supported for {set_type.__name__}"
        )


def _coerce_set_types(sa: Any, sb: Any) -> Tuple[Any, Any]:
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


def _same_predicate_system(sa, sb) -> bool:
    """Do two stars constrain the IDENTICAL predicate vector?

    True only when predicate counts, constraint systems (C, d), and
    predicate bounds all match exactly — e.g. two branches that applied
    only affine layers to a common ancestor. Predicate COUNT alone is
    not sufficient: two branches can each append one ReLU-relaxation
    predicate (same nVar) with different constraints, and sharing those
    would be unsound.
    """
    if sa.nVar != sb.nVar:
        return False
    Ca = np.asarray(sa.C, dtype=np.float64)
    Cb = np.asarray(sb.C, dtype=np.float64)
    if Ca.shape != Cb.shape or not np.array_equal(Ca, Cb):
        return False
    da = np.asarray(sa.d, dtype=np.float64)
    db = np.asarray(sb.d, dtype=np.float64)
    if da.shape != db.shape or not np.array_equal(da, db):
        return False
    for pa, pb in ((sa.predicate_lb, sb.predicate_lb),
                   (sa.predicate_ub, sb.predicate_ub)):
        if (pa is None) != (pb is None):
            return False
        if pa is not None and not np.array_equal(
                np.asarray(pa, dtype=np.float64),
                np.asarray(pb, dtype=np.float64)):
            return False
    return True


def _join_predicates(sa, sb):
    """Block-diagonal composition of two stars' predicate systems for a
    Minkowski-sum join (cf. Tran et al., HSCC 2023, Prop. 2.8; NNV
    Star.MinkowskiSum). Returns (C, d, pred_lb, pred_ub) over the
    concatenated predicate vector [alpha_a; alpha_b]."""
    n1, n2 = sa.nVar, sb.nVar

    def _cmat(C, n):
        if C is None or not np.asarray(C).size:
            return np.zeros((0, n))
        return np.asarray(C, dtype=np.float64).reshape(-1, n)

    def _dvec(d):
        if d is None or not np.asarray(d).size:
            return np.zeros((0, 1))
        return np.asarray(d, dtype=np.float64).reshape(-1, 1)

    Ca, Cb = _cmat(sa.C, n1), _cmat(sb.C, n2)
    C = np.vstack([
        np.hstack([Ca, np.zeros((Ca.shape[0], n2))]),
        np.hstack([np.zeros((Cb.shape[0], n1)), Cb]),
    ])
    d = np.vstack([_dvec(sa.d), _dvec(sb.d)])

    if sa.predicate_lb is None and sb.predicate_lb is None:
        pred_lb = pred_ub = None
    else:
        def _pb(p, n, fill):
            if p is None:
                return np.full((n, 1), fill, dtype=np.float64)
            return np.asarray(p, dtype=np.float64).reshape(-1, 1)

        pred_lb = np.vstack([_pb(sa.predicate_lb, n1, -np.inf),
                             _pb(sb.predicate_lb, n2, -np.inf)])
        pred_ub = np.vstack([_pb(sa.predicate_ub, n1, np.inf),
                             _pb(sb.predicate_ub, n2, np.inf)])
    return C, d, pred_lb, pred_ub


def _join_star_systems(sets: List) -> Tuple[
        List[np.ndarray], np.ndarray, np.ndarray,
        Optional[np.ndarray], Optional[np.ndarray]]:
    """Map a list of Stars/ImageStars onto one joint predicate system.

    Sets whose predicate systems are identical
    (:func:`_same_predicate_system`) share variables — exact. Distinct
    systems are composed block-diagonally — sound, over-approximate
    (cross-branch correlation dropped). Zero-padding to a common width
    instead would silently identify UNRELATED predicate variables and
    can under-approximate.

    Returns (V_list, C, d, pred_lb, pred_ub) where V_list[i] is
    sets[i]'s basis flattened to 2-D and re-expressed over the joint
    predicate vector [alpha_g1; alpha_g2; ...].
    """
    group_reps = []
    assign = []
    for s in sets:
        for gi, rep in enumerate(group_reps):
            if _same_predicate_system(rep, s):
                assign.append(gi)
                break
        else:
            assign.append(len(group_reps))
            group_reps.append(s)

    sizes = [rep.nVar for rep in group_reps]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int) \
        if sizes else np.zeros(0, dtype=int)
    n = int(sum(sizes))

    C_blocks, d_blocks = [], []
    for rep, off in zip(group_reps, offsets):
        C_rep = (np.asarray(rep.C, dtype=np.float64).reshape(-1, rep.nVar)
                 if np.asarray(rep.C).size else np.zeros((0, rep.nVar)))
        if C_rep.shape[0]:
            block = np.zeros((C_rep.shape[0], n))
            block[:, off:off + rep.nVar] = C_rep
            C_blocks.append(block)
            d_blocks.append(
                np.asarray(rep.d, dtype=np.float64).reshape(-1, 1))
    joint_C = np.vstack(C_blocks) if C_blocks else np.zeros((0, n))
    joint_d = np.vstack(d_blocks) if d_blocks else np.zeros((0, 1))

    if all(rep.predicate_lb is None for rep in group_reps):
        pred_lb = pred_ub = None
    else:
        lb_parts, ub_parts = [], []
        for rep in group_reps:
            if rep.predicate_lb is None:
                lb_parts.append(np.full((rep.nVar, 1), -np.inf))
                ub_parts.append(np.full((rep.nVar, 1), np.inf))
            else:
                lb_parts.append(np.asarray(
                    rep.predicate_lb, dtype=np.float64).reshape(-1, 1))
                ub_parts.append(np.asarray(
                    rep.predicate_ub, dtype=np.float64).reshape(-1, 1))
        pred_lb = np.vstack(lb_parts)
        pred_ub = np.vstack(ub_parts)

    V_list = []
    for s, gi in zip(sets, assign):
        V_flat = s.V.reshape(-1, s.V.shape[-1])
        V = np.zeros((V_flat.shape[0], n + 1))
        V[:, 0] = V_flat[:, 0]
        off = offsets[gi]
        V[:, 1 + off:1 + off + s.nVar] = V_flat[:, 1:]
        V_list.append(V)
    return V_list, joint_C, joint_d, pred_lb, pred_ub


def _add_sets(sets_a: List, sets_b: List, op_name: str) -> List:
    """
    Element-wise addition or subtraction of two lists of sets.

    Used for residual connections where both inputs are computed sets
    (not constant parameters).

    For Star/ImageStar (NNV-parity join; tighter prefix-aligned variant
    tracked as future work in .claude/issues.md I-35):
      * If both stars constrain the IDENTICAL predicate system
        (:func:`_same_predicate_system` — e.g. affine-only branches from
        a common ancestor), the addition is exact:
        ``V_out = V1 +/- V2`` with the shared constraints preserved.
      * Otherwise (e.g. approx-star ReLU appended relaxation predicates
        on a branch), Minkowski sum via block-diagonal predicate
        composition — sound, over-approximate (inter-branch correlation
        is dropped).

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
        sign = 1.0 if 'add' in op_name else -1.0

        if isinstance(sa, ImageStar) and isinstance(sb, ImageStar):
            if _same_predicate_system(sa, sb):
                # Exact: shared predicates, element-wise basis addition.
                V_out = sa.V + sign * sb.V
                out = ImageStar(
                    V_out, sa.C, sa.d, sa.predicate_lb, sa.predicate_ub,
                    sa.height, sa.width, sa.num_channels
                )
            else:
                # Minkowski join over [alpha_a; alpha_b]. ImageStar V is
                # (H, W, C, 1 + nVar): channel 0 is the center.
                center = sa.V[..., :1] + sign * sb.V[..., :1]
                V_out = np.concatenate(
                    [center, sa.V[..., 1:], sign * sb.V[..., 1:]], axis=-1)
                C, d, plb, pub = _join_predicates(sa, sb)
                out = ImageStar(V_out, C, d, plb, pub,
                                sa.height, sa.width, sa.num_channels)
            output_sets.append(out)

        elif isinstance(sa, Star) and isinstance(sb, Star):
            if _same_predicate_system(sa, sb):
                # Exact: shared predicates, element-wise basis addition.
                V_out = sa.V + sign * sb.V
                out = Star(V_out, sa.C, sa.d,
                           sa.predicate_lb, sa.predicate_ub)
            else:
                # Minkowski join over [alpha_a; alpha_b]. Star V is
                # (dim, 1 + nVar): column 0 is the center.
                center = sa.V[:, :1] + sign * sb.V[:, :1]
                V_out = np.hstack([center, sa.V[:, 1:], sign * sb.V[:, 1:]])
                C, d, plb, pub = _join_predicates(sa, sb)
                out = Star(V_out, C, d, plb, pub)
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


def _mul_sets(sets_a: List, sets_b: List) -> List:
    """
    Element-wise multiplication of two lists of computed sets.

    For Box: standard interval arithmetic.
    For Zono: extract bounds, compute interval product, build new Zono.
    For Star: McCormick relaxation (tightest convex relaxation of bilinear terms).

    Args:
        sets_a: List of sets from the first operand
        sets_b: List of sets from the second operand

    Returns:
        List of output sets
    """
    if len(sets_a) != len(sets_b):
        raise ValueError(
            f"Cannot multiply set lists of different lengths: "
            f"{len(sets_a)} vs {len(sets_b)}"
        )

    output_sets = []

    for sa, sb in zip(sets_a, sets_b):
        # Coerce mismatched types (e.g., ImageStar * Star)
        sa, sb = _coerce_set_types(sa, sb)

        if isinstance(sa, Box) and isinstance(sb, Box):
            output_sets.append(_mul_boxes(sa, sb))
        elif isinstance(sa, Zono) and isinstance(sb, Zono):
            output_sets.append(_mul_zonos(sa, sb))
        elif isinstance(sa, Star) and isinstance(sb, Star):
            output_sets.append(_mul_stars_mccormick(sa, sb))
        else:
            raise NotImplementedError(
                f"Element-wise mul not supported for "
                f"{type(sa).__name__} and {type(sb).__name__}"
            )

    return output_sets


def _mul_boxes(a: 'Box', b: 'Box') -> 'Box':
    """Element-wise multiplication of two Boxes via interval arithmetic."""
    products = np.stack([a.lb * b.lb, a.lb * b.ub, a.ub * b.lb, a.ub * b.ub])
    return Box(products.min(axis=0), products.max(axis=0))


def _mul_zonos(a: 'Zono', b: 'Zono') -> 'Zono':
    """Element-wise multiplication of two Zonos via interval arithmetic.

    Since Zonos don't track shared generators, we fall back to
    interval bounds, compute the product interval, and construct
    a new Zono from the resulting bounds.
    """
    lb_a, ub_a = a.get_bounds()
    lb_b, ub_b = b.get_bounds()
    products = np.stack([lb_a * lb_b, lb_a * ub_b, ub_a * lb_b, ub_a * ub_b])
    out_lb = products.min(axis=0)
    out_ub = products.max(axis=0)
    return Zono.from_bounds(out_lb, out_ub)


def _mul_stars_mccormick(sa: 'Star', sb: 'Star', lp_solver: str = 'default') -> 'Star':
    """
    Element-wise multiplication of two Stars using McCormick relaxation.

    If the two Stars constrain the identical predicate system (see
    :func:`_same_predicate_system`), the shared coupling is kept exactly;
    otherwise their predicate systems are joined block-diagonally
    (:func:`_join_predicates`) — sound, over-approximate.

    For z_i = x_i * y_i with x_i in [a_i, b_i], y_i in [c_i, d_i]:
      z >= a*y + c*x - a*c   (lower envelope 1)
      z >= b*y + d*x - b*d   (lower envelope 2)
      z <= a*y + d*x - a*d   (upper envelope 1)
      z <= b*y + c*x - b*c   (upper envelope 2)

    New predicate variables z_i are introduced for each output dimension.
    """
    N = sa.dim
    if sb.dim != N:
        raise ValueError(
            f"McCormick mul: operand dims differ ({sa.dim} vs {sb.dim}); "
            f"broadcasting must be resolved by the caller")

    if _same_predicate_system(sa, sb):
        # Identical predicate systems: keep the shared (exact) coupling.
        n = sa.nVar
        sa_V, sb_V = sa.V, sb.V
        joint_C = (np.asarray(sa.C, dtype=np.float64).reshape(-1, n)
                   if np.asarray(sa.C).size else np.zeros((0, n)))
        joint_d = (np.asarray(sa.d, dtype=np.float64).reshape(-1, 1)
                   if np.asarray(sa.d).size else np.zeros((0, 1)))
        joint_pred_lb, joint_pred_ub = sa.predicate_lb, sa.predicate_ub
    else:
        # Different predicate systems (e.g. approx-star ReLU appended
        # relaxation predicates on one branch): block-diagonal join —
        # sound, over-approximate (inter-branch correlation dropped).
        # Zero-padding the smaller system instead would silently
        # identify UNRELATED predicate variables and can produce an
        # under-approximation.
        joint_C, joint_d, joint_pred_lb, joint_pred_ub = \
            _join_predicates(sa, sb)
        n1, n2 = sa.nVar, sb.nVar
        n = n1 + n2
        sa_V = np.hstack([sa.V, np.zeros((sa.V.shape[0], n2))])
        sb_V = np.hstack([sb.V[:, :1],
                          np.zeros((sb.V.shape[0], n1)),
                          sb.V[:, 1:]])

    # Get bounds for both operands via LP
    lbs_a = np.zeros(N)
    ubs_a = np.zeros(N)
    lbs_b = np.zeros(N)
    ubs_b = np.zeros(N)

    for i in range(N):
        lb_val, ub_val = sa.get_range(i, lp_solver)
        if lb_val is None or ub_val is None:
            raise ValueError(
                f"LP solver returned None for dimension {i} of first operand. "
                f"Star may be infeasible."
            )
        lbs_a[i] = lb_val
        ubs_a[i] = ub_val

        lb_val, ub_val = sb.get_range(i, lp_solver)
        if lb_val is None or ub_val is None:
            raise ValueError(
                f"LP solver returned None for dimension {i} of second operand. "
                f"Star may be infeasible."
            )
        lbs_b[i] = lb_val
        ubs_b[i] = ub_val

    a, b = lbs_a, ubs_a  # bounds on x
    c, d = lbs_b, ubs_b  # bounds on y

    # Interval arithmetic for z predicate bounds
    products = np.stack([a * c, a * d, b * c, b * d])
    z_lb = products.min(axis=0)
    z_ub = products.max(axis=0)

    # Build new V matrix: [constant_col | old_pred_cols | new_z_cols]
    # The output is z_i, so V maps the new z predicate variables to output.
    V1 = np.zeros((N, n + 1))  # zero contribution from old predicates
    V2 = np.eye(N)             # identity for new z variables
    new_V = np.hstack([V1, V2])

    # Carry forward the joint predicate constraints, padded with zeros
    # for the new z variables
    constraint_blocks_C = []
    constraint_blocks_d = []

    if joint_C.size > 0:
        constraint_blocks_C.append(
            np.hstack([joint_C, np.zeros((joint_C.shape[0], N))]))
        constraint_blocks_d.append(joint_d)

    # Extract V coefficients for x and y in terms of shared predicates
    Vx = sa_V[:, 1:n + 1]  # (N, n)
    cx = sa_V[:, 0]         # (N,)
    Vy = sb_V[:, 1:n + 1]  # (N, n)
    cy = sb_V[:, 0]         # (N,)

    # McCormick envelope constraints
    C_rows = []
    d_rows = []

    for i in range(N):
        # Lower 1: z_i >= a_i*y_i + c_i*x_i - a_i*c_i
        # Rewrite: -(z_i) + a_i*(Vy_i @ alpha + cy_i) + c_i*(Vx_i @ alpha + cx_i) <= a_i*c_i
        # => (a_i*Vy_i + c_i*Vx_i) @ alpha - z_i <= a_i*c_i - a_i*cy_i - c_i*cx_i
        row = np.zeros(n + N)
        row[:n] = a[i] * Vy[i] + c[i] * Vx[i]
        row[n + i] = -1
        rhs = a[i] * c[i] - a[i] * cy[i] - c[i] * cx[i]
        C_rows.append(row)
        d_rows.append(rhs)

        # Lower 2: z_i >= b_i*y_i + d_i*x_i - b_i*d_i
        row = np.zeros(n + N)
        row[:n] = b[i] * Vy[i] + d[i] * Vx[i]
        row[n + i] = -1
        rhs = b[i] * d[i] - b[i] * cy[i] - d[i] * cx[i]
        C_rows.append(row)
        d_rows.append(rhs)

        # Upper 1: z_i <= a_i*y_i + d_i*x_i - a_i*d_i
        row = np.zeros(n + N)
        row[:n] = -(a[i] * Vy[i] + d[i] * Vx[i])
        row[n + i] = 1
        rhs = -a[i] * d[i] + a[i] * cy[i] + d[i] * cx[i]
        C_rows.append(row)
        d_rows.append(rhs)

        # Upper 2: z_i <= b_i*y_i + c_i*x_i - b_i*c_i
        row = np.zeros(n + N)
        row[:n] = -(b[i] * Vy[i] + c[i] * Vx[i])
        row[n + i] = 1
        rhs = -b[i] * c[i] + b[i] * cy[i] + c[i] * cx[i]
        C_rows.append(row)
        d_rows.append(rhs)

    C_mc = np.array(C_rows)
    d_mc = np.array(d_rows).reshape(-1, 1)

    constraint_blocks_C.append(C_mc)
    constraint_blocks_d.append(d_mc)

    new_C = np.vstack(constraint_blocks_C)
    new_d = np.vstack(constraint_blocks_d)

    # Predicate bounds: joint predicates + new z variables
    if joint_pred_lb is not None:
        new_pred_lb = np.vstack([
            np.asarray(joint_pred_lb, dtype=np.float64).reshape(-1, 1),
            z_lb.reshape(-1, 1)])
        new_pred_ub = np.vstack([
            np.asarray(joint_pred_ub, dtype=np.float64).reshape(-1, 1),
            z_ub.reshape(-1, 1)])
    else:
        # Even with no original predicate bounds, z variables need bounds
        new_pred_lb = np.vstack([np.full((n, 1), -np.inf), z_lb.reshape(-1, 1)])
        new_pred_ub = np.vstack([np.full((n, 1), np.inf), z_ub.reshape(-1, 1)])

    return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)


def _div_sets(sets_a: List, sets_b: List, lp_solver: str = 'default') -> List:
    """
    Element-wise division z = a / b of two lists of computed sets.

    Raises if any denominator range straddles zero — there is no sound
    convex over-approximation in that case.
    """
    output_sets = []
    for sa, sb in zip(sets_a, sets_b):
        sa, sb = _coerce_set_types(sa, sb)

        if isinstance(sa, Box) and isinstance(sb, Box):
            blb = np.asarray(sb.lb, dtype=np.float64).flatten()
            bub = np.asarray(sb.ub, dtype=np.float64).flatten()
            if np.any((blb <= 0) & (bub >= 0)):
                raise NotImplementedError(
                    "Element-wise Div: denominator range straddles zero")
            alb = np.asarray(sa.lb, dtype=np.float64).flatten()
            aub = np.asarray(sa.ub, dtype=np.float64).flatten()
            q = np.stack([alb / blb, alb / bub, aub / blb, aub / bub])
            output_sets.append(Box(q.min(axis=0), q.max(axis=0)))
        elif isinstance(sa, Star) and isinstance(sb, Star):
            output_sets.append(_div_stars_mccormick(sa, sb, lp_solver))
        else:
            raise NotImplementedError(
                f"Element-wise div not supported for "
                f"{type(sa).__name__} and {type(sb).__name__}"
            )
    return output_sets


def _div_stars_mccormick(sa: 'Star', sb: 'Star', lp_solver: str = 'default') -> 'Star':
    """
    Element-wise division z = x / y of two Stars.

    Encodes the bilinear identity x_i = z_i * y_i with McCormick
    envelopes over the joint predicate system: fresh predicate
    variables z_i are bounded by interval division, and the four
    McCormick planes for the product z*y over
    [z_lb, z_ub] x [y_lb, y_ub] couple x, y and z. Sound
    over-approximation; raises if any denominator range straddles zero.

    Predicate systems are kept shared when identical and joined
    block-diagonally otherwise (same policy as _mul_stars_mccormick).
    """
    N = sa.dim
    if sb.dim != N:
        raise ValueError(
            f"McCormick div: operand dims differ ({sa.dim} vs {sb.dim}); "
            f"broadcasting must be resolved by the caller")

    if _same_predicate_system(sa, sb):
        n = sa.nVar
        sa_V, sb_V = sa.V, sb.V
        joint_C = (np.asarray(sa.C, dtype=np.float64).reshape(-1, n)
                   if np.asarray(sa.C).size else np.zeros((0, n)))
        joint_d = (np.asarray(sa.d, dtype=np.float64).reshape(-1, 1)
                   if np.asarray(sa.d).size else np.zeros((0, 1)))
        joint_pred_lb, joint_pred_ub = sa.predicate_lb, sa.predicate_ub
    else:
        joint_C, joint_d, joint_pred_lb, joint_pred_ub = \
            _join_predicates(sa, sb)
        n1, n2 = sa.nVar, sb.nVar
        n = n1 + n2
        sa_V = np.hstack([sa.V, np.zeros((sa.V.shape[0], n2))])
        sb_V = np.hstack([sb.V[:, :1],
                          np.zeros((sb.V.shape[0], n1)),
                          sb.V[:, 1:]])

    # Per-dimension ranges of numerator x and denominator y via LP
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)
    for i in range(N):
        lb_val, ub_val = sa.get_range(i, lp_solver)
        if lb_val is None or ub_val is None:
            raise ValueError(
                f"LP solver returned None for dimension {i} of numerator. "
                f"Star may be infeasible.")
        a[i], b[i] = lb_val, ub_val
        lb_val, ub_val = sb.get_range(i, lp_solver)
        if lb_val is None or ub_val is None:
            raise ValueError(
                f"LP solver returned None for dimension {i} of "
                f"denominator. Star may be infeasible.")
        c[i], d[i] = lb_val, ub_val

    if np.any((c <= 0) & (d >= 0)):
        bad = int(np.argmax((c <= 0) & (d >= 0)))
        raise NotImplementedError(
            f"Element-wise Div: denominator range [{c[bad]}, {d[bad]}] "
            f"(dimension {bad}) straddles zero")

    # Interval division bounds for the quotient z
    q = np.stack([a / c, a / d, b / c, b / d])
    z_lb = q.min(axis=0)
    z_ub = q.max(axis=0)

    # Output star: z variables only
    new_V = np.hstack([np.zeros((N, n + 1)), np.eye(N)])

    constraint_blocks_C = []
    constraint_blocks_d = []
    if joint_C.size > 0:
        constraint_blocks_C.append(
            np.hstack([joint_C, np.zeros((joint_C.shape[0], N))]))
        constraint_blocks_d.append(joint_d)

    Vx = sa_V[:, 1:n + 1]
    cx = sa_V[:, 0]
    Vy = sb_V[:, 1:n + 1]
    cy = sb_V[:, 0]

    # McCormick planes for the identity x = z*y with z in [zl, zu] and
    # y in [c, d]; rows are over [alpha | z]
    C_rows = []
    d_rows = []
    for i in range(N):
        zl, zu = z_lb[i], z_ub[i]

        # x >= zl*y + c*z - zl*c  =>  zl*y + c*z - x <= zl*c
        row = np.zeros(n + N)
        row[:n] = zl * Vy[i] - Vx[i]
        row[n + i] = c[i]
        C_rows.append(row)
        d_rows.append(zl * c[i] - zl * cy[i] + cx[i])

        # x >= zu*y + d*z - zu*d  =>  zu*y + d*z - x <= zu*d
        row = np.zeros(n + N)
        row[:n] = zu * Vy[i] - Vx[i]
        row[n + i] = d[i]
        C_rows.append(row)
        d_rows.append(zu * d[i] - zu * cy[i] + cx[i])

        # x <= zl*y + d*z - zl*d  =>  x - zl*y - d*z <= -zl*d
        row = np.zeros(n + N)
        row[:n] = Vx[i] - zl * Vy[i]
        row[n + i] = -d[i]
        C_rows.append(row)
        d_rows.append(-zl * d[i] + zl * cy[i] - cx[i])

        # x <= zu*y + c*z - zu*c  =>  x - zu*y - c*z <= -zu*c
        row = np.zeros(n + N)
        row[:n] = Vx[i] - zu * Vy[i]
        row[n + i] = -c[i]
        C_rows.append(row)
        d_rows.append(-zu * c[i] + zu * cy[i] - cx[i])

    constraint_blocks_C.append(np.array(C_rows))
    constraint_blocks_d.append(np.array(d_rows).reshape(-1, 1))

    new_C = np.vstack(constraint_blocks_C)
    new_d = np.vstack(constraint_blocks_d)

    if joint_pred_lb is not None:
        new_pred_lb = np.vstack([
            np.asarray(joint_pred_lb, dtype=np.float64).reshape(-1, 1),
            z_lb.reshape(-1, 1)])
        new_pred_ub = np.vstack([
            np.asarray(joint_pred_ub, dtype=np.float64).reshape(-1, 1),
            z_ub.reshape(-1, 1)])
    else:
        new_pred_lb = np.vstack([np.full((n, 1), -np.inf),
                                 z_lb.reshape(-1, 1)])
        new_pred_ub = np.vstack([np.full((n, 1), np.inf),
                                 z_ub.reshape(-1, 1)])

    return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)


def _mul_sets_by_constant(input_sets: List, scale: np.ndarray) -> List:
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
                # Full spatial scale. The ONNX constant is flat in
                # (C, H, W) order; permute to the HWC storage layout.
                scale_4d = scale.reshape(c, h, w) \
                    .transpose(1, 2, 0).reshape(h, w, c, 1)
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
                # ONNX constant is (C, H, W) flat; permute to HWC
                scale_flat = scale.reshape(c_ch, h, w) \
                    .transpose(1, 2, 0).reshape(-1, 1)
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


def _concat_sets(set_lists: List[List], axis: int) -> List:
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
            # Join predicate systems (exact when identical, block-
            # diagonal otherwise), then concatenate the joint-space V
            # tensors along the spatial axis.
            V_list, joint_C, joint_d, pred_lb, pred_ub = \
                _join_star_systems(sets_to_concat)
            n_cols = V_list[0].shape[1]
            V_imgs = [
                Vj.reshape(s.V.shape[0], s.V.shape[1], s.V.shape[2],
                           n_cols)
                for s, Vj in zip(sets_to_concat, V_list)
            ]
            V_out = np.concatenate(V_imgs, axis=axis)
            out = ImageStar(
                V_out, joint_C, joint_d, pred_lb, pred_ub,
                V_out.shape[0], V_out.shape[1], V_out.shape[2]
            )
            output_sets.append(out)

        elif isinstance(first, Star):
            # Join predicate systems, then vstack the joint-space V
            # matrices (input-stacked order; the caller remaps rows for
            # inner-axis concats).
            V_list, joint_C, joint_d, pred_lb, pred_ub = \
                _join_star_systems(sets_to_concat)
            V_out = np.vstack(V_list)
            output_sets.append(
                Star(V_out, joint_C, joint_d, pred_lb, pred_ub))

        elif isinstance(first, ImageZono):
            # Generator columns from different branches cannot be
            # assumed to denote the same noise symbols — compose
            # block-diagonally (sound; drops cross-branch correlation).
            total_gen = sum(s.V.shape[1] for s in sets_to_concat)
            col_off = 0
            img_c_list = []
            img_V_list = []
            for s in sets_to_concat:
                h, w, c_ch = s.height, s.width, s.num_channels
                V_joint = np.zeros((s.V.shape[0], total_gen))
                V_joint[:, col_off:col_off + s.V.shape[1]] = s.V
                col_off += s.V.shape[1]
                img_c_list.append(s.c.reshape(h, w, c_ch))
                img_V_list.append(V_joint.reshape(h, w, c_ch, total_gen))

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
            # Same reasoning as ImageZono: block-diagonal generators.
            gens = [s.V for s in sets_to_concat]
            total_rows = sum(g.shape[0] for g in gens)
            total_cols = sum(g.shape[1] for g in gens)
            V_out = np.zeros((total_rows, total_cols))
            row_off = col_off = 0
            for g in gens:
                V_out[row_off:row_off + g.shape[0],
                      col_off:col_off + g.shape[1]] = g
                row_off += g.shape[0]
                col_off += g.shape[1]

            c_out = np.vstack([s.c for s in sets_to_concat])
            output_sets.append(Zono(c_out, V_out))

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


def _handle_onnx_concat(
    module: Any,
    node: Any,
    node_values: Dict[str, List],
    node_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Optional[List]:
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
    shapes = []
    for arg in node.args:
        if hasattr(arg, 'name') and arg.name in node_values:
            sl = node_values[arg.name]
            set_lists.append(sl)
            shapes.append((node_shapes or {}).get(arg.name))
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
        return _concat_sets(set_lists, set_axis)

    # Flat sets: _concat_sets stacks inputs in order, which equals
    # concat along the OUTERMOST axis. With the true tensor shapes
    # known, remap rows exactly for inner-axis concats.
    stacked = _concat_sets(set_lists, 0)
    if shapes and all(s is not None for s in shapes):
        rank = len(shapes[0])
        axis = onnx_axis % rank
        base = 0
        blocks = []
        for shp in shapes:
            sz = int(np.prod(shp))
            blocks.append(np.arange(sz).reshape(tuple(shp)) + base)
            base += sz
        rows = np.concatenate(blocks, axis=axis).flatten()
        if np.array_equal(rows, np.arange(base)):
            return stacked
        return _select_rows(stacked, rows, base)
    if onnx_axis not in (0, 1):
        raise NotImplementedError(
            f"Concat along axis {onnx_axis} requires tensor shape "
            f"metadata, which is unavailable for node {node.name!r}")
    return stacked


def _slice_set(s: Any, slices_by_axis: Dict[int, slice]) -> Any:
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


def _handle_onnx_slice(
    module: Any,
    node: Any,
    node_values: Dict[str, List],
    graph_module: fx.GraphModule,
    node_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Optional[List]:
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

    # Shape-aware exact path: with the tensor shape known, a Slice is an
    # exact row selection via an index tensor (handles any axis — the
    # legacy path below mis-dimensions middle-axis slices on flat sets).
    in_shape = (node_shapes or {}).get(getattr(first_arg, 'name', None))
    if in_shape is not None and not isinstance(first_set,
                                               (ImageStar, ImageZono)):
        full = [slice(None)] * len(in_shape)
        ok = True
        if isinstance(module, OnnxSliceV9):
            for ax, sl in enumerate(module._pos_axes_slices):
                if ax < len(full):
                    full[ax] = sl
        else:
            try:
                def _p(i, default=None):
                    if len(node.args) > i:
                        return _get_parameter(
                            graph_module,
                            node.args[i]).numpy().astype(np.int64).flatten()
                    return default
                starts = _p(1)
                ends = _p(2)
                axes = _p(3, np.arange(len(starts)))
                steps = _p(4, np.ones(len(starts), dtype=np.int64))

                def _clip(v):
                    return None if abs(int(v)) >= 2**31 - 1 else int(v)
                for st, en, ax, sp in zip(starts, ends, axes, steps):
                    full[int(ax) % len(in_shape)] = slice(
                        _clip(st), _clip(en), int(sp))
            except Exception:  # noqa: BLE001
                ok = False
        if ok:
            size = int(np.prod(in_shape))
            rows = np.arange(size).reshape(in_shape)[tuple(full)].flatten()
            return _select_rows(input_sets, rows, size)

    slices_by_axis = {}

    if isinstance(module, OnnxSliceV9):
        # V9: module._pos_axes_slices is a list of slice objects per axis
        # Skip axis 0 (batch dim), shift remaining by -1
        for ax, sl in enumerate(module._pos_axes_slices):
            if ax == 0:
                continue  # skip batch dimension
            set_ax = ax - 1
            slices_by_axis[set_ax] = sl

    elif isinstance(module, OnnxSlice):
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
            start_val = int(starts[i])
            end_val = int(ends[i])
            step_val = int(steps[i])

            # ONNX uses very large numbers (e.g., 2^63-1) for "to end"
            if end_val > 2**30:
                end_val = None

            if ax == 0:
                # Without shape metadata, axis 0 can only be ASSUMED to
                # be the batch dim — safe solely for a no-op slice. A
                # rank-1 input sliced on its only axis (e.g. models
                # packing spec params after the image) lands here, and
                # silently skipping computes a wrong function downstream.
                if start_val == 0 and end_val is None and step_val == 1:
                    continue
                raise NotImplementedError(
                    f"Slice on axis 0 for node {node.name!r} with no "
                    f"shape metadata: cannot tell batch from data axis")
            set_ax = ax - 1  # remove batch dim

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


def _split_set(s: Any, split_sizes: List[int], axis: int) -> List:
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


def _handle_onnx_split(
    module: Any,
    node: Any,
    node_values: Dict[str, List],
    graph_module: fx.GraphModule,
    node_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Optional[List[List]]:
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

    if isinstance(module, OnnxSplit13):
        # OnnxSplit13: split sizes from second argument (dynamic) or even division
        if len(node.args) > 1:
            split_tensor = _get_parameter(graph_module, node.args[1])
            split_sizes = split_tensor.numpy().astype(int).tolist()
        else:
            # Even division by num_splits
            split_sizes = None  # handled below

    elif isinstance(module, OnnxSplit):
        # OnnxSplit: split sizes from module.split attribute
        if module.split is not None:
            split_sizes = list(module.split)
        else:
            split_sizes = None  # handled below

    # Exact shape-aware path for flat sets: split the flat index tensor
    # along the true tensor axis and select rows. The legacy flat path
    # below assumes the split axis is the outermost non-batch axis,
    # which is wrong for inner axes.
    in_shape = (node_shapes or {}).get(getattr(first_arg, 'name', None))
    if in_shape is not None and len(in_shape) > 0 \
            and not isinstance(first_set, (ImageStar, ImageZono)):
        size = int(np.prod(in_shape))
        axis = onnx_axis % len(in_shape)
        sizes = split_sizes
        if sizes is None:
            num_splits = module.num_splits
            if in_shape[axis] % num_splits != 0:
                raise ValueError(
                    f"Split: axis length {in_shape[axis]} (shape "
                    f"{tuple(in_shape)}, axis {axis}) is not divisible "
                    f"into {num_splits} equal chunks")
            sizes = [in_shape[axis] // num_splits] * num_splits
        if sum(sizes) != in_shape[axis]:
            raise ValueError(
                f"Split sizes {sizes} do not sum to axis length "
                f"{in_shape[axis]} (shape {tuple(in_shape)}, axis {axis})")
        index_tensor = np.arange(size).reshape(in_shape)
        chunk_indices = np.split(
            index_tensor, np.cumsum(sizes)[:-1], axis=axis)
        return [_select_rows(input_sets, chunk.flatten(), size)
                for chunk in chunk_indices]

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


def _reach_probabilistic(model: nn.Module, input_set: Any, **kwargs: Any) -> List:
    """
    Probabilistic reachability using conformal inference.

    This is a model-agnostic approach that works with any PyTorch model.
    """
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

    # Determine input shape for reshaping flat samples
    # (needed for models that expect spatial input, e.g. NHWC or NCHW)
    input_shape = kwargs.get('input_shape', None)
    if input_shape is None:
        if isinstance(input_set, ImageStar):
            input_shape = (input_set.num_channels, input_set.height, input_set.width)
        elif isinstance(input_set, ImageZono):
            input_shape = (input_set.num_channels, input_set.height, input_set.width)

    # Create model wrapper for numpy interface
    def model_fn(x):
        """Wrap PyTorch model as numpy-in/numpy-out callable."""
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            if input_shape is not None and len(input_shape) > 1:
                batch = x_tensor.shape[0]
                x_tensor = x_tensor.reshape(batch, *input_shape)
            output = model(x_tensor)
            return output.numpy()

    # Run probabilistic verification. ``method='probabilistic'`` is the
    # legacy alias; ``conformal_reach`` is the renamed primary entry. The
    # set-to-Box conversion above (and image-shape handling) are
    # specific to this dispatch path so we keep it as a thin wrapper.
    result = conformal_reach(
        model_fn,
        box,
        m=kwargs.get('m', 8000),
        ell=kwargs.get('ell', None),
        epsilon=kwargs.get('epsilon', 0.001),
        surrogate=kwargs.get('surrogate', 'clipping_block'),
        training_samples=kwargs.get('training_samples', None),
        pca_components=kwargs.get('pca_components', None),
        batch_size=kwargs.get('batch_size', 100),
        seed=kwargs.get('seed', None),
        verbose=kwargs.get('verbose', False),
    )

    return [result]


def _reach_hybrid(model: nn.Module, input_set: Any, **kwargs: Any) -> List:
    """
    Hybrid reachability: deterministic until threshold, then probabilistic.

    Attempts exact reachability layer by layer. If the number of stars exceeds
    max_stars or time exceeds timeout_per_layer, switches to probabilistic
    verification for the remaining layers.
    """
    max_stars = kwargs.get('max_stars', 1000)
    timeout_per_layer = kwargs.get('timeout_per_layer', 30.0)
    verbose = kwargs.get('verbose', False)

    # Trace the model if needed
    if not isinstance(model, fx.GraphModule):
        try:
            model = torch.fx.symbolic_trace(model)
        except Exception as e:
            raise TypeError(
                f"n2v requires models to be traceable by torch.fx. "
                f"Models with data-dependent control flow (e.g., "
                f"'if x.sum() > 0') or inline module instantiation "
                f"(e.g., 'nn.ReLU()(x)') are not supported. "
                f"For inline activations, use functional equivalents "
                f"(e.g., F.relu(x) instead of nn.ReLU()(x)). "
                f"Tracing failed with: {e}"
            ) from e

    # Extract ordered list of modules from fx graph
    named_modules = dict(model.named_modules())
    layer_entries = []
    for node in model.graph.nodes:
        if node.op == 'call_module':
            module = named_modules.get(node.target)
            if module is not None:
                layer_entries.append(module)
        elif node.op == 'call_function' and node.target is not operator.getitem:
            equiv = _function_node_to_module(node)
            if equiv is not None:
                layer_entries.append(equiv)
        elif node.op == 'call_method':
            if node.target == 'flatten':
                start_dim = node.args[1] if len(node.args) > 1 else 1
                layer_entries.append(nn.Flatten(start_dim=start_dim))

    if not layer_entries:
        layer_entries = [model]

    current_sets = [input_set]

    for i, layer in enumerate(layer_entries):
        if verbose:
            logger.info(f"Layer {i+1}/{len(layer_entries)}: {type(layer).__name__}")

        start_time = time.time()

        try:
            # Try deterministic reachability
            next_sets = reach_layer(layer, current_sets, 'exact', **kwargs)
            elapsed = time.time() - start_time

            # Check thresholds
            if len(next_sets) > max_stars:
                if verbose:
                    logger.info(f"  Exceeded {max_stars} stars, switching to probabilistic")
                raise _SwitchToProbabilistic()

            if elapsed > timeout_per_layer:
                if verbose:
                    logger.info(f"  Exceeded {timeout_per_layer}s timeout, switching to probabilistic")
                raise _SwitchToProbabilistic()

            current_sets = next_sets

        except (_SwitchToProbabilistic, MemoryError):
            # Switch to probabilistic for remaining layers
            remaining_model = nn.Sequential(*layer_entries[i:])

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
