"""
Unified reachability analysis for neural networks.

This module provides the core reachability computation engine that routes
computation based on set type and handles both standard PyTorch models
and ONNX GraphModules.
"""

import itertools
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
from n2v.sets.star import fresh_predicate_tag
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.layer_ops.linear_reach import linear_hexatope, linear_octatope
from n2v.utils.model_preprocessing import fuse_batchnorm, has_batchnorm
from n2v.utils.bounds_precomputation import compute_intermediate_bounds
from n2v.utils.ns_certificate import _add_dir, _div_dir, _mul_dir
from n2v.utils.lpsolver import _bound_lp_solver, _outward_lp_ranges

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
from n2v.nn.layer_ops.batchnorm_reach import (
    _batchnorm_single_box, _bn_fold_mag, _get_bn_params,
)
from n2v.nn.layer_ops.flatten_reach import _flatten_imagezono_chw
from n2v.nn.layer_ops.bilinear_matmul_reach import (
    bilinear_matmul_star,
    bilinear_matmul_zono,
    bilinear_matmul_box,
)
from n2v.nn.layer_ops.softmax_attention_reach import (
    softmax_attn_star,
    softmax_attn_zono,
    softmax_attn_box,
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

    if method in ('exact', 'approx') and isinstance(input_set, Star) \
            and not isinstance(input_set, ImageStar):
        # SOUNDNESS: a predicate box with +-inf entries makes the interval
        # bounds NaN (0 * inf), which the ReLU classifiers read as "stably
        # active" -- measured as an unsound reach set and a false UNSAT via
        # verify_specification. Replace non-finite entries with exact-LP
        # bounds over the full polytope (a sound no-op for finite boxes); an
        # infeasible input set reaches nothing. Runs after the config is
        # unpacked so the caller's LP solver is used.
        from n2v.sets.predicate_box import ensure_finite_predicate_box
        input_set, _empty = ensure_finite_predicate_box(
            input_set, lp_solver=kwargs.get('lp_solver', 'default'))
        if _empty:
            return []

    # Auto-fuse BatchNorm layers if present. A Box reach keeps BatchNorm
    # unfused: batchnorm_box rounds its own folding outward, while the rounding
    # of a pre-fused layer's parameters would escape the Box's outward bounds.
    if has_batchnorm(model) and not isinstance(input_set, Box):
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
    # Optional callback(node_name, module, input_sets), called before every
    # standard layer is reached; the zonotope bound pre-pass uses it to
    # record the sets entering each nonlinear layer.
    on_layer_input = kwargs.pop('_on_layer_input', None)

    # The one input set feeds every placeholder, so a second graph input
    # would be read as the same set over the same predicate variables as the
    # first (X1 - X2 came out a single point: a false UNSAT).
    graph_inputs = [n for n in graph_module.graph.nodes
                    if n.op == 'placeholder' and n.users]
    if len(graph_inputs) > 1:
        raise NotImplementedError(
            f"graph has {len(graph_inputs)} inputs "
            f"({', '.join(n.name for n in graph_inputs)}); reachability "
            f"takes one input set, so only single-input graphs are supported")

    # Best-effort tensor-shape annotation (for shape-dependent exact ops:
    # Gather, Slice, Transpose on flat sets). Absence of a shape makes
    # those ops raise loudly — never guess.
    input_shape = kwargs.pop('input_shape', None)
    node_shapes = _propagate_shapes(graph_module, input_sets, input_shape)
    const_names = _constant_node_names(graph_module)

    if precompute and layer_bounds is None:
        precompute_method = precompute if isinstance(precompute, str) else 'ibp'
        layer_bounds = compute_intermediate_bounds(
            graph_module, input_sets[0], method=precompute_method,
            input_shape=input_shape)

    # Get the set type from the first input
    set_type = type(input_sets[0])

    # Get named modules
    named_modules = dict(graph_module.named_modules())

    # Store intermediate values for each node
    node_values = _new_node_values()
    # Nodes whose value is tensor-shape metadata (``x.size()``,
    # ``x.shape[0]``), not a set: they only feed the batch entry of a
    # view/reshape target, which the reshape handler strips.
    shape_nodes = set()
    verbose = kwargs.get('verbose', None)
    current_sets = input_sets
    # In a graph that combines branches, the ops that pair two set lists
    # read identical predicate systems as the same variables; the new
    # predicate variables of each node get a tag of their own so that two
    # nodes never look alike (see _tag_new_predicates).
    tag_new_predicates = _graph_joins_branches(graph_module, const_names)
    tag_pending = None

    # Process each node in the computational graph
    for node in graph_module.graph.nodes:
        # Nodes come in topological order: tagging the previous node's sets
        # here happens before any consumer reads them.
        if tag_pending is not None:
            _tag_new_predicates(tag_pending, node_values)
        tag_pending = node if tag_new_predicates else None

        if node.op == 'placeholder':
            # Input node
            node_values[node.name] = current_sets

        elif node.op == 'get_attr':
            # Parameter retrieval - handled when used by operations
            pass

        elif node.op == 'call_module':
            module = named_modules.get(node.target)
            if module is None:
                if node.name in const_names:
                    continue
                raise NotImplementedError(
                    f"graph node {node.name!r} refers to module "
                    f"{node.target!r}, which the graph module does not have")

            module_type = type(module).__name__

            if verbose:
                logger.debug(f'  Processing: {node.target} ({module_type})')

            # Handle OnnxReshape
            if isinstance(module, OnnxReshape):
                input_sets_op = _resolve_set_input(
                    node.args[0], node, node_values, const_names)
                if input_sets_op is None:
                    continue  # constant reshape: folded by its consumers

                # Get target shape from second argument (frozen parameter)
                shape_node = node.args[1]
                shape_tensor = _get_parameter(graph_module, shape_node)
                target_shape = tuple(shape_tensor.numpy().astype(int))

                # Only materialize a 3-D target as an ImageStar when the result
                # actually feeds a spatial (conv/pool) consumer; otherwise keep
                # it flat (e.g. the transformer head-split reshape, which must
                # not be reinterpreted as a (C, H, W) image).
                force_flat = not _reshape_feeds_spatial(node, graph_module)
                result_sets = _handle_reshape(
                    input_sets_op, target_shape, force_flat=force_flat)
                node_values[node.name] = result_sets
                current_sets = result_sets
                continue

            # Handle OnnxConcat
            if isinstance(module, OnnxConcat):
                result_sets = _handle_onnx_concat(
                    module, node, node_values, graph_module, node_shapes)
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
            # A Pad of a constant tensor never reaches the pad handler: its
            # consumers fold it with torch (_const_eval), whatever its value.
            if (module_type == 'OnnxPadDynamic'
                    and node.name not in const_names):
                # The pad handler reaches only zero padding, so a 3rd input
                # (constant pad value) must resolve to 0. Checked on every
                # visit, not only when the pads are first stashed: a later
                # reach of the same model skips that branch. A value that
                # cannot be resolved raises rather than being taken as 0.
                if len(node.args) > 2 and node.args[2] is not None:
                    try:
                        pv = _get_parameter(
                            graph_module, node.args[2]).numpy()
                    except Exception as exc:  # noqa: BLE001 — re-raised below
                        raise NotImplementedError(
                            f"Pad constant value ({node.args[2]}) could not "
                            f"be resolved to a constant; only zero padding "
                            f"is supported") from exc
                    if np.any(np.asarray(pv) != 0):
                        raise NotImplementedError(
                            "Pad with non-zero constant value is not "
                            "supported")
                if not hasattr(module, 'pads') and len(node.args) > 1:
                    module.pads = _get_parameter(
                        graph_module, node.args[1]).numpy().astype(
                            int).tolist()

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

            # Shape-aware BatchNorm on flat (N, C, ...) sets: the per-channel
            # affine must broadcast over the non-channel axes (e.g. the ViT
            # per-token norm on (1, 48, 17)). ImageStar/plain dim==C cases
            # return None and fall through to the channel-aware dispatcher.
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                result_sets = _handle_batchnorm_flat(
                    module, node, node_values, node_shapes)
                if result_sets is not None:
                    node_values[node.name] = result_sets
                    current_sets = result_sets
                    continue

            # Shape-aware Softmax on flat sets: the attention softmax is a
            # batched (..., L, L) softmax over one axis, routed to the sound
            # correlated row bound. ImageStar / global-(1,N) softmax fall
            # through to the generic dispatcher.
            if (isinstance(module, nn.Softmax)
                    or module_type in ('OnnxSoftmaxV1V11', 'OnnxSoftmax')):
                result_sets = _handle_softmax_flat(
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
                    node_shapes, lp_solver=kwargs.get('lp_solver', 'default'),
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
                    # OnnxResize carries its scales/sizes as graph inputs
                    # (node.args[2]/[3]), not module attributes. Resolve the
                    # real constants here so the reach uses the model's actual
                    # scale factors instead of guessing them from a probe.
                    if module_type == 'OnnxResize':
                        _add_resize_scale_kwargs(graph_module, node, layer_kwargs)
                    if on_layer_input is not None:
                        on_layer_input(node.name, module, input_sets_op)
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
            # No silent skips here either (see the call_module guard): a
            # function node without a reach rule (operator.add, torch.cat,
            # torch.clamp, ...) must raise, or its consumers would compute on
            # the sets of whatever node ran before it.
            src_node = node.args[0] if node.args else None
            if node.target is operator.getitem:
                # Multi-output ops (e.g. Split) store a list of set lists
                src_val = (node_values.get(src_node.name)
                           if isinstance(src_node, fx.Node) else None)
                if (len(node.args) >= 2 and isinstance(src_val, list)
                        and len(src_val) > 0
                        and isinstance(src_val[0], list)):
                    node_values[node.name] = src_val[node.args[1]]
                    current_sets = src_val[node.args[1]]
                elif (isinstance(src_node, fx.Node)
                        and src_node.name in shape_nodes):
                    shape_nodes.add(node.name)  # e.g. x.shape[0]
                elif node.name not in const_names:
                    raise NotImplementedError(
                        f"graph node {node.name!r} (getitem on "
                        f"{getattr(src_node, 'name', src_node)!r}) is not "
                        f"supported by reachability: the source is not a "
                        f"multi-output op")
            elif (node.target is getattr and len(node.args) >= 2
                    and node.args[1] == 'shape'
                    and isinstance(src_node, fx.Node)
                    and src_node.name in node_values):
                shape_nodes.add(node.name)  # x.shape
            elif _is_inference_identity_function(node):
                # Dropout traced in eval mode: the input passes through.
                input_sets_op = _resolve_set_input(
                    src_node, node, node_values, const_names)
                if input_sets_op is None:
                    continue  # constant subgraph
                node_values[node.name] = input_sets_op
                current_sets = input_sets_op
            else:
                # Try to convert function to module equivalent
                equiv_module = _function_node_to_module(node)
                if equiv_module is not None:
                    input_sets_op = _resolve_set_input(
                        src_node, node, node_values, const_names)
                    if input_sets_op is None:
                        continue  # constant subgraph

                    layer_kwargs = dict(kwargs)
                    if layer_bounds is not None and node.name in layer_bounds:
                        layer_kwargs['precomputed_bounds'] = layer_bounds[node.name]
                    else:
                        layer_kwargs.pop('precomputed_bounds', None)

                    if on_layer_input is not None:
                        on_layer_input(node.name, equiv_module, input_sets_op)
                    output_sets = reach_layer(
                        equiv_module, input_sets_op,
                        method, **layer_kwargs,
                    )
                    node_values[node.name] = output_sets
                    current_sets = output_sets

                elif node.name not in const_names:
                    raise NotImplementedError(
                        f"graph node {node.name!r} (call_function "
                        f"{_fx_target_name(node.target)}) is not supported "
                        f"by reachability")

        elif node.op == 'call_method':
            method_name = node.target  # e.g., 'flatten', 'view', 'reshape'
            first_arg = node.args[0] if node.args else None

            if method_name in ('size', 'dim') \
                    and isinstance(first_arg, fx.Node) \
                    and first_arg.name in node_values:
                shape_nodes.add(node.name)  # x.size(), x.size(0), x.dim()
                continue
            identity = _is_identity_method(node)
            if not identity and method_name not in ('flatten', 'view',
                                                    'reshape'):
                if node.name in const_names:
                    continue
                raise NotImplementedError(
                    f"graph node {node.name!r} (call_method "
                    f"{method_name!r}) is not supported by reachability")

            # Resolve input from first argument (the tensor/self)
            input_sets_op = _resolve_set_input(
                first_arg, node, node_values, const_names)
            if input_sets_op is None:
                continue  # constant subgraph

            if identity:
                # Copy, layout or float32/float64 cast: values unchanged.
                node_values[node.name] = input_sets_op
                current_sets = input_sets_op

            elif method_name == 'flatten':
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
                # _handle_reshape strips the batch entry, so only that entry
                # may be a traced value (x.view(x.size(0), -1)).
                kept = target_shape[1:] if len(target_shape) >= 2 \
                    else target_shape
                if not all(isinstance(v, (int, np.integer)) for v in kept):
                    raise NotImplementedError(
                        f"graph node {node.name!r} ({method_name}) has a "
                        f"non-constant target shape {target_shape}")
                force_flat = not _reshape_feeds_spatial(node, graph_module)
                result_sets = _handle_reshape(
                    input_sets_op, target_shape, force_flat=force_flat)
                node_values[node.name] = result_sets
                current_sets = result_sets

        elif node.op == 'output':
            # Output node - extract final result. It must name a node that
            # produced sets; anything else would return stale sets.
            output_node = node.args[0] if node.args else None
            if isinstance(output_node, (tuple, list)) \
                    and len(output_node) == 1:
                output_node = output_node[0]
            if not (isinstance(output_node, fx.Node)
                    and output_node.name in node_values):
                raise NotImplementedError(
                    f"graph output {output_node!r} has no reachable sets "
                    f"(multiple outputs or an unsupported output node)")
            current_sets = node_values[output_node.name]

    return current_sets


def _resolve_set_input(arg: Any, node: fx.Node,
                       node_values: Dict[str, List],
                       const_names: set) -> Optional[List]:
    """Sets that ``arg`` feeds into ``node``, or None if ``node`` belongs
    to a constant subgraph (its consumers fold it via _const_eval).

    Raises instead of falling back to the most recently computed sets: an
    input that produced no sets (an unsupported upstream op) would
    otherwise be replaced by an unrelated node's sets.
    """
    if isinstance(arg, fx.Node) and arg.name in node_values:
        return node_values[arg.name]
    if node.name in const_names:
        return None
    raise NotImplementedError(
        f"graph node {node.name!r} ({node.op} "
        f"{_fx_target_name(node.target)}): its input "
        f"{getattr(arg, 'name', arg)!r} has no reachable sets (an "
        f"unsupported upstream op?)")


def _fx_target_name(target: Any) -> str:
    """Readable name of an fx node target, for error messages."""
    if isinstance(target, str):
        return target
    return getattr(target, '__name__', repr(target))


# Dropout functions, with the keyword, position and default of their
# training flag. Traced in eval mode (``training=self.training``) the flag is
# the literal False and the function returns its input unchanged.
_DROPOUT_FUNCTIONS = {
    F.dropout: ('training', 2, True),
    F.dropout1d: ('training', 2, True),
    F.dropout2d: ('training', 2, True),
    F.dropout3d: ('training', 2, True),
    F.alpha_dropout: ('training', 2, False),
    F.feature_alpha_dropout: ('training', 2, False),
    torch.dropout: ('train', 2, None),
    torch.alpha_dropout: ('train', 2, None),
    torch.feature_dropout: ('train', 2, None),
    torch.feature_alpha_dropout: ('train', 2, None),
}


def _is_inference_identity_function(node: fx.Node) -> bool:
    """Is ``node`` a dropout call whose training flag is literally False?
    With the flag True (or traced as a value) dropout is random, so the
    node is not treated as the identity."""
    try:
        spec = _DROPOUT_FUNCTIONS.get(node.target)
    except TypeError:  # unhashable target
        return False
    if spec is None:
        return False
    key, pos, default = spec
    if key in node.kwargs:
        flag = node.kwargs[key]
    elif len(node.args) > pos:
        flag = node.args[pos]
    else:
        flag = default
    return flag is False


def _is_identity_method(node: fx.Node) -> bool:
    """Is ``node`` a tensor method that returns its input's values?

    Covers copies and layout (``clone``, ``detach``, ``contiguous``) and
    casts between float32 and float64 (``float``, ``double``, ``to`` or
    ``type`` with a dtype only); reachability already treats the network's
    floating-point arithmetic as real arithmetic. Casts to other dtypes and
    device moves are not included.
    """
    args, kwargs = node.args[1:], dict(node.kwargs)
    kwargs.pop('memory_format', None)
    if node.target in ('clone', 'detach', 'contiguous', 'float', 'double'):
        return not args and not kwargs
    if node.target in ('to', 'type'):
        kwargs.pop('non_blocking', None)
        kwargs.pop('copy', None)
        dtypes = list(args)
        if 'dtype' in kwargs:
            dtypes.append(kwargs.pop('dtype'))
        return (not kwargs and len(dtypes) == 1
                and isinstance(dtypes[0], torch.dtype)
                and dtypes[0] in (torch.float32, torch.float64))
    return False


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
        # all_input_nodes includes nodes nested in list/tuple arguments
        # (torch.cat((x, y))); scanning only the top-level args would call
        # such a node constant.
        deps = n.all_input_nodes
        if all(d.name in const or d.op == 'get_attr' for d in deps) \
                and n.op in ('call_module', 'call_function', 'call_method'):
            const.add(n.name)
    return const


def _graph_joins_branches(graph_module: fx.GraphModule,
                          const_names: set) -> bool:
    """Does some node take the sets of two or more different nodes?

    Only such a node (Add, Sub, Mul, Div, Concat, set-by-set MatMul) pairs
    the predicate systems of two set lists; in a graph without one the
    systems are never compared. Constants and tensor-shape values
    (``x.size()``, ``x.shape`` and their items) do not count.
    """
    shape_like = set()
    for n in graph_module.graph.nodes:
        if n.op == 'call_method' and n.target in ('size', 'dim'):
            shape_like.add(n.name)
            continue
        if n.op == 'call_function' and n.args:
            src = n.args[0]
            if (n.target is getattr and len(n.args) >= 2
                    and n.args[1] == 'shape'):
                shape_like.add(n.name)
                continue
            if (n.target is operator.getitem and isinstance(src, fx.Node)
                    and src.name in shape_like):
                shape_like.add(n.name)
                continue
        if n.op in ('placeholder', 'get_attr', 'output') \
                or n.name in const_names:
            continue
        sources = {d.name for d in n.all_input_nodes
                   if d.op != 'get_attr' and d.name not in const_names
                   and d.name not in shape_like}
        if len(sources) > 1:
            return True
    return False


def _node_sets(value: Any) -> List:
    """The sets a node value holds: a list of sets, or a list of set lists
    (a multi-output op such as Split); nothing for any other value."""
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        if isinstance(v, list):
            out.extend(v)
        else:
            out.append(v)
    return out


def _tag_new_predicates(node: fx.Node, node_values: Dict[str, List]) -> None:
    """Give the predicate variables that ``node`` introduced an identity of
    their own, in place in ``node_values``.

    The ops that pair two set lists (:func:`_same_predicate_system`,
    :func:`_lineage_groups`) read identical predicate systems as the same
    variables. That holds for variables that come from the same node, but
    two nodes can append variables with identical rows and bounds for
    different functions: a box-only variable (Sin and Cos over a range
    holding an extremum both give ``z in [-1, 1]``; Sign across 0; an odd
    Pow across 0; a ReLU neuron kept whole within the LP pad of 0), or rows
    that do not pin the function (Round(x) and Floor(x + 0.5) both give
    ``z - x in [-1/2, 1/2]``). Sharing them made Sin(X) - Cos(X) a single
    point, a false UNSAT.

    A star of the node's output keeps its input's variables when its
    variable count and predicate box equal those of an input star (an
    affine map, an exact split's added rows, a pass-through). Every other
    star (the node appended variables, joined two systems, or rebuilt the
    star) gets one more variable, unused, with the box ``[0, t]`` for a tag
    ``t`` unique to this node (:func:`n2v.sets.star.fresh_predicate_tag`).
    The set is unchanged. The stars of one node share the tag, and the stars
    derived from them keep it in their system prefix, so they still pair
    with each other; stars of two different nodes never do. A node whose
    output only feeds the graph output is left as it is.
    """
    out = node_values.get(node.name)
    if out is None or node.op == 'placeholder' or not node.users \
            or all(u.op == 'output' for u in node.users):
        return

    def _key(s):
        return (int(s.nVar),) + tuple(
            None if p is None
            else np.asarray(p, dtype=np.float64).tobytes()
            for p in (s.predicate_lb, s.predicate_ub))

    kept = {_key(s)
            for d in node.all_input_nodes
            for s in _node_sets(node_values.get(d.name))
            if isinstance(s, (Star, ImageStar))}

    def _is_new(s):
        return isinstance(s, (Star, ImageStar)) and _key(s) not in kept

    if not any(_is_new(s) for s in _node_sets(out)):
        return
    tag = fresh_predicate_tag()

    def _tagged(s):
        return _with_tag_variable(s, tag) if _is_new(s) else s

    node_values[node.name] = [
        [_tagged(x) for x in v] if isinstance(v, list) else _tagged(v)
        for v in out]


def _with_tag_variable(s: Union[Star, ImageStar], tag: float):
    """``s`` with one more predicate variable, last, that no basis vector
    uses, bounded to ``[0, tag]``: the same set, over a system that holds
    ``tag``. The bounds go in the predicate box, and also in two rows when
    ``s`` has no complete predicate box.
    """
    n = int(s.nVar)
    V = np.concatenate([s.V, np.zeros(s.V.shape[:-1] + (1,))], axis=-1)
    C = np.asarray(s.C, dtype=np.float64)
    if C.ndim != 2 or C.shape[1] != n:
        C = C.reshape(-1, n) if C.size else np.zeros((0, n))
    d = np.asarray(s.d, dtype=np.float64).reshape(-1, 1)
    C = np.hstack([C, np.zeros((C.shape[0], 1))])
    plb, pub = s.predicate_lb, s.predicate_ub
    if plb is None or pub is None:
        rows = np.zeros((2, n + 1))
        rows[0, n], rows[1, n] = -1.0, 1.0
        C = np.vstack([C, rows])
        d = np.vstack([d, [[0.0], [tag]]])
    if plb is not None:
        plb = np.vstack([np.asarray(plb, dtype=np.float64).reshape(-1, 1),
                         [[0.0]]])
    if pub is not None:
        pub = np.vstack([np.asarray(pub, dtype=np.float64).reshape(-1, 1),
                         [[tag]]])
    if isinstance(s, ImageStar):
        out = ImageStar(V, C, d, plb, pub)
    else:
        out = Star(V, C, d, plb, pub, outer_zono=s.Z)
    out.state_lb, out.state_ub = s.state_lb, s.state_ub
    return out


class _ShapeRecorder(fx.Interpreter):
    """Runs a graph module and records the shape of every tensor-valued
    node. Unlike torch.fx ShapeProp it neither prints a traceback when the
    forward fails (failing candidate inputs are expected, see
    :func:`_propagate_shapes`) nor writes into ``node.meta``."""

    def __init__(self, module: fx.GraphModule):
        super().__init__(module)
        self.shapes: Dict[str, Tuple[int, ...]] = {}

    def run_node(self, n: fx.Node) -> Any:
        result = super().run_node(n)
        if isinstance(result, torch.Tensor):
            self.shapes[n.name] = tuple(int(x) for x in result.shape)
        return result


def _propagate_shapes(graph_module: fx.GraphModule, input_sets: List,
                      input_shape=None) -> Dict[str, Tuple[int, ...]]:
    """Tensor shape of every fx node, from one forward on a dummy input.

    Runs the model's REAL forward on a dummy input — a wrong dummy shape
    crashes the forward and we return {} (consumers then raise when
    they need a shape). ``input_shape`` is the batch-stripped model input
    shape (threaded from the runner); heuristic fallback otherwise.
    """
    try:
        # Candidate dummy shapes, tried in order. The batch-prefixed form is
        # tried first (the common case: ``input_shape`` is batch-stripped, so
        # the model wants the batch back). A rank-1-native model — e.g. one
        # packing a flattened image plus spec params into a single vector and
        # slicing axis 0 to split them — has NO batch dim, so the prefixed
        # forward crashes; the un-prefixed candidate is the correct fallback.
        if input_shape is not None:
            stripped = tuple(int(d) for d in input_shape)
            candidates = [(1, *stripped), stripped]
        else:
            s0 = input_sets[0]
            if isinstance(s0, ImageStar):
                candidates = [(1, s0.num_channels, s0.height, s0.width)]
            else:
                candidates = [(1, int(s0.dim)), (int(s0.dim),)]
        # The dummy must match the model's floating dtype (a float64 ONNX
        # model rejects a float32 input).
        dtype = next((t.dtype for t in itertools.chain(
            graph_module.parameters(), graph_module.buffers())
            if t.is_floating_point()), torch.float32)
        dtypes = [dtype] if dtype == torch.float32 else [dtype, torch.float32]
        for dtype, dummy_shape in itertools.product(dtypes, candidates):
            dummy = torch.zeros(dummy_shape, dtype=dtype)
            recorder = _ShapeRecorder(graph_module)
            try:
                with torch.no_grad():
                    recorder.run(dummy)
            except Exception:  # noqa: BLE001 — try the next candidate shape
                continue
            return recorder.shapes
        return {}
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
    # ImageZono is a Zono subclass, so it must be excluded by name: its rows
    # are in HWC order, and the reduction matrix below assumes the tensor's
    # row-major (CHW) order.
    if (not input_sets
            or isinstance(input_sets[0], (ImageStar, ImageZono))
            or not isinstance(input_sets[0], (Star, Zono, Box))):
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


def _add_resize_scale_kwargs(graph_module: fx.GraphModule, node: Any,
                             layer_kwargs: dict) -> None:
    """Resolve an OnnxResize node's scales/sizes constants from the graph
    and stash them in ``layer_kwargs`` as ``resize_scales`` / ``resize_sizes``
    for the upsample reach op.

    onnx2torch's OnnxResize takes (x, roi, scales[, sizes]) as forward
    inputs, so the factors live in node.args, not the module. Resolving
    them here lets the reach use the model's real scale factors instead of
    probing the bare module (which cannot recover them). Empty/absent
    inputs are skipped; anything non-constant is left for the op to error on.
    """
    def _maybe_const(arg):
        if arg is None or not hasattr(arg, 'op'):
            return None
        try:
            tensor = _get_parameter(graph_module, arg)
        except Exception:  # noqa: BLE001 — non-constant: let the op handle it
            return None
        return tensor if tensor.numel() > 0 else None

    args = node.args
    if len(args) > 2:
        scales = _maybe_const(args[2])
        if scales is not None:
            layer_kwargs['resize_scales'] = scales.flatten().tolist()
    if len(args) > 3:
        sizes = _maybe_const(args[3])
        if sizes is not None:
            layer_kwargs['resize_sizes'] = [int(v) for v in sizes.flatten().tolist()]


def _const_eval(graph_module: fx.GraphModule, node: Any, memo: dict) -> torch.Tensor:
    """Value of a constant node, folded with torch (see _get_parameter). The
    node of a multi-output op (e.g. Split) evaluates to a tuple of tensors,
    which its getitem consumers index."""
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
                result = module(*args)
            # A multi-output module (e.g. Split) returns a tuple of tensors,
            # which its getitem consumers index; keep it as a tuple.
            result = (_detach_outputs(result)
                      if isinstance(result, (tuple, list))
                      else result.detach().cpu())
        else:
            raise ValueError(f"not a constant subgraph: {node.name}")
    elif node.op == 'call_function':
        args = [_const_eval(graph_module, a, memo) for a in node.args]
        with torch.no_grad():
            result = node.target(*args, **node.kwargs)
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu()
        elif (isinstance(result, (tuple, list)) and result
                and all(isinstance(r, torch.Tensor) for r in result)):
            result = _detach_outputs(result)  # e.g. torch.split
        else:
            result = torch.tensor(result)
    else:
        raise ValueError(f"not a constant node: {node.name} ({node.op})")
    memo[node.name] = result
    return result


def _detach_outputs(outputs) -> tuple:
    """The outputs of a multi-output op on constants, detached and on the
    CPU. The tuple is what _const_eval returns for the op's own node; a
    caller that needs a tensor from it fails on the missing tensor methods
    instead of reading the wrong value."""
    return tuple(t.detach().cpu() for t in outputs)


def _handle_reshape(input_sets: List, target_shape: tuple,
                    force_flat: bool = False) -> List:
    """
    Reshape sets to a new shape.

    Handles ONNX Reshape operations by reordering the V matrix.
    ONNX uses NCHW format; ImageStar stores HWC internally.

    Args:
        input_sets: List of input sets
        target_shape: Target shape tuple from ONNX (includes batch dim)
        force_flat: keep an already-flat set flat even for a 3-D target
            (non-image reshape, e.g. the transformer head-split). The caller
            sets this when the reshape does not feed a spatial (conv/pool)
            consumer; a reshape preserves ONNX row-major order, so staying
            flat is exact and avoids the CHW->HWC image reinterpretation.

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
            output_sets.append(_reshape_star(s, spatial_shape, force_flat))
        elif isinstance(s, Zono):
            output_sets.append(_reshape_zono(s, spatial_shape, force_flat))
        elif isinstance(s, Box):
            output_sets.append(_reshape_box(s, spatial_shape))
        else:
            # For Hexatope/Octatope, reshape is just reinterpretation
            output_sets.append(s)

    return output_sets


# Module types that consume a spatial (N, C, H, W) layout and therefore
# require a reshape's flat output to be materialized as an ImageStar.
_SPATIAL_CONSUMERS = frozenset({
    'Conv1d', 'Conv2d', 'Conv3d', 'OnnxConv',
    'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'OnnxMaxPool',
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'OnnxAveragePool',
    'AdaptiveAvgPool2d', 'OnnxGlobalAveragePool',
})
# Pass-through ops that preserve spatial layout and may sit between a reshape
# and its spatial consumer (so the forward walk must look past them). Erring
# toward "keep flat" is soundness-safe (an actual conv reached with a flat set
# raises loudly), so only well-understood layout-preserving ops are listed.
_SPATIAL_PASSTHROUGH = frozenset({
    'ReLU', 'ReLU6', 'LeakyReLU', 'Sigmoid', 'Tanh', 'GELU', 'ELU',
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'Dropout', 'Identity',
    'OnnxBinaryMathOperation', 'OnnxClip', 'OnnxPadStatic', 'OnnxPadDynamic',
    # Layout-permuting/reshaping ops that can sit between a reshape and a conv
    # (e.g. the common Reshape -> Transpose(to NCHW) -> Conv export); the walk
    # must look past them to find the real spatial consumer.
    'OnnxTranspose', 'OnnxResize', 'OnnxSlice', 'OnnxSliceV9', 'OnnxGather',
})


def _reshape_feeds_spatial(node: Any, graph_module: fx.GraphModule) -> bool:
    """True if a Reshape's output flows into a spatial op (conv/pool), so it
    must be materialized as an ImageStar.

    A forward walk over consumers looks past layout-preserving pass-throughs
    (activations, batchnorm, elementwise). Reaching a conv/pool returns True;
    a layout-defining op (MatMul/Transpose/Gemm/Linear/Reshape/Softmax/...)
    ends that branch as non-spatial. Defaults to False (keep flat) -- the
    ONNX-correct, transformer-safe behavior -- when no spatial consumer is
    found; an actual conv reached with a flat set raises a clear error rather
    than silently mis-laying out.
    """
    seen = set()
    stack = list(node.users)
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        name = None
        if u.op == 'call_module':
            try:
                name = type(graph_module.get_submodule(u.target)).__name__
            except AttributeError:
                name = None
        if name in _SPATIAL_CONSUMERS:
            return True
        # Walk past pass-through layers and raw function/method nodes
        # (flatten/getitem/etc.); stop at layout-defining modules.
        if name in _SPATIAL_PASSTHROUGH or u.op in (
                'call_function', 'call_method'):
            stack.extend(u.users)
    return False


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


def _reshape_star(star: 'Star', spatial_shape: Tuple[int, ...],
                  force_flat: bool = False) -> Union['Star', 'ImageStar']:
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

    ``force_flat`` keeps the set flat even for a 3-D target. This is for
    non-image reshapes such as the transformer head-split
    ``(tokens, heads, dim_head)``, which must NOT be reinterpreted as a
    ``(C, H, W)`` image (the CHW->HWC permutation would corrupt the layout
    for the downstream attention matmuls). The caller sets it when the
    reshape does not feed a spatial (conv/pool) consumer; a reshape preserves
    ONNX row-major order, so staying flat is exact.
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

    if len(resolved) == 3 and not force_flat:
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


def _reshape_zono(zono: 'Zono', spatial_shape: Tuple[int, ...],
                  force_flat: bool = False) -> Union['Zono', 'ImageZono']:
    """Reshape plain Zono. Same convention and exactness argument as
    :func:`_reshape_star`; ImageZono stores flat HWC-ordered arrays.
    ``force_flat`` keeps a 3-D target flat (non-image reshape, e.g. the
    transformer head-split)."""
    total = zono.dim
    resolved = _resolve_shape(spatial_shape, total)
    product = 1
    for s in resolved:
        product *= s
    if product != total:
        raise ValueError(
            f"Cannot reshape Zono of dim {total} to shape {resolved}"
        )

    if len(resolved) == 3 and not force_flat:
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
    lp_solver: str = 'default',
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
            return _mul_sets(sets_a, sets_b, lp_solver)
        elif op_name == '_onnx_div':
            return _div_sets(sets_a, sets_b, lp_solver)

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

    negate_input = False
    if mirrored and hasattr(module, 'math_op_function'):
        op = module.math_op_function.__name__
        if 'sub' in op:
            # param - x = (-x) + param: negate the sets below and add the
            # constant through the same layout-aware path as x + param.
            negate_input = True
        elif '_onnx_div' in op:
            raise NotImplementedError(
                "division BY a computed set is not supported")
        # add / mul are commutative: fall through with swapped operands

    # Get the sets from the first input
    if first_input.name not in node_values:
        return None

    input_sets_op = node_values[first_input.name]
    if negate_input:
        input_sets_op = _mul_sets_by_constant(input_sets_op, np.array(-1.0))

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
            # Without shape metadata a tile is the broadcast only when the
            # constant has no size-1 axis after its leading ones: (1, C, 1, 1)
            # against (1, C, H, W) repeats each entry H*W times instead.
            core = np.shape(param_value)
            while core and core[0] == 1:
                core = core[1:]
            if param_value.size != 1 and 1 in core:
                raise NotImplementedError(
                    f"constant of shape {np.shape(param_value)} cannot be "
                    f"broadcast against a flat set of dim {set_dim} without "
                    f"tensor-shape metadata for {first_input.name!r}")
            param_value = np.tile(param_value.flatten(),
                                  set_dim // param_value.size)

    # Determine operation type
    if not hasattr(module, 'math_op_function'):
        return None

    op_name = 'add' if negate_input else module.math_op_function.__name__

    if op_name == 'mul':
        return _mul_sets_by_constant(input_sets_op, param_value)
    elif op_name == '_onnx_div':
        return _div_sets_by_constant(input_sets_op, param_value)
    elif 'add' in op_name:
        bias = param_value
    elif 'sub' in op_name:
        bias = -param_value
    else:
        raise NotImplementedError(
            f"Binary operation {op_name} not supported for {set_type.__name__}"
        )

    # Apply translation based on set type
    if set_type in (Hexatope, Octatope):
        # Apply via dummy linear layer with identity + bias
        linear_fn = linear_hexatope if set_type == Hexatope else linear_octatope

        output_sets = []
        # float64: a float32 layer would round the constant.
        bias_tensor = torch.tensor(bias, dtype=torch.float64).reshape(-1)

        for s in input_sets_op:
            dummy_linear = nn.Linear(s.dim, s.dim, bias=True, dtype=torch.float64)
            dummy_linear.weight.data = torch.eye(s.dim, dtype=torch.float64)
            dummy_linear.bias.data = bias_tensor
            result = linear_fn(dummy_linear, [s])
            output_sets.extend(result)

        return output_sets

    elif set_type in (Zono, ImageZono):
        output_sets = []
        bias_reshaped = bias.reshape(-1, 1)
        for s in input_sets_op:
            if isinstance(s, ImageZono):
                # ImageZono.c is flat in HWC order; the ONNX constant is
                # flat in (C, H, W) order (same layout rule as the
                # ImageStar branch below).
                h, w, c_ch = s.height, s.width, s.num_channels
                b = np.asarray(bias, dtype=np.float64).flatten()
                if b.size == 1:
                    bias_flat = np.full((h * w * c_ch, 1), b[0])
                elif b.size == h * w * c_ch:
                    bias_flat = b.reshape(c_ch, h, w).transpose(
                        1, 2, 0).reshape(-1, 1)
                elif b.size == c_ch:
                    bias_flat = np.tile(b, h * w).reshape(-1, 1)
                else:
                    raise NotImplementedError(
                        f"ImageZono add/sub: constant of size {b.size} "
                        f"does not broadcast against (C,H,W)="
                        f"({c_ch},{h},{w})")
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
            # Rounded outward, so the Box contains every x + bias.
            out = Box(_add_dir(s.lb, bias_reshaped, -1.0),
                      _add_dir(s.ub, bias_reshaped, 1.0))
            output_sets.append(out)
        return output_sets

    elif set_type in (Star, ImageStar):
        # set_type reflects the graph's INPUT set; mid-graph sets can be
        # flat Stars (e.g. after a flatten) or ImageStars (a reshape that
        # feeds a conv), so dispatch per element.
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
                    f"add/sub not supported for {type(s).__name__} in a "
                    f"{set_type.__name__} graph")
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


def _handle_bilinear_matmul(
    left_node: Any,
    right_node: Any,
    node: Any,
    node_values: Dict[str, List],
    node_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Optional[List]:
    """Sound ``set @ set`` MatMul (the attention products ``Q@K^T`` and
    ``A@V``) via the bilinear interval-matmul reach.

    Both operands are computed sets, so the result is a sound
    over-approximation, never the exact image. Standard batched-matmul
    semantics apply (last two axes are the matrix dims, leading axes batch),
    so the full ONNX tensor shapes are passed straight through. Returns
    ``None`` if either operand is not a resolvable set.
    """
    if not (hasattr(left_node, 'name') and left_node.name in node_values
            and hasattr(right_node, 'name')
            and right_node.name in node_values):
        return None
    left_sets = node_values[left_node.name]
    right_sets = node_values[right_node.name]
    if not left_sets or not right_sets:
        return None
    left_shape = (node_shapes or {}).get(left_node.name)
    right_shape = (node_shapes or {}).get(right_node.name)
    if left_shape is None or right_shape is None:
        raise NotImplementedError(
            "set@set MatMul requires tensor-shape metadata (ShapeProp could "
            f"not infer shapes for node {node.name!r})")
    left_shape = tuple(int(s) for s in left_shape)
    right_shape = tuple(int(s) for s in right_shape)
    # The bilinear kernels combine the two lists index by index; pair the
    # pieces first so every input's left and right pieces meet.
    pairs = _pair_set_lists([left_sets, right_sets])
    left_sets = [p[0] for p in pairs]
    right_sets = [p[1] for p in pairs]
    first = left_sets[0]
    if isinstance(first, Star):
        return bilinear_matmul_star(
            left_sets, right_sets, left_shape, right_shape)
    if isinstance(first, Zono):
        return bilinear_matmul_zono(
            left_sets, right_sets, left_shape, right_shape)
    if isinstance(first, Box):
        return bilinear_matmul_box(
            left_sets, right_sets, left_shape, right_shape)
    raise NotImplementedError(
        f"set@set MatMul not supported for set type {type(first).__name__}")


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
            # Neither operand is a constant: this is a set@set (bilinear)
            # MatMul -- the attention products Q@K^T and A@V -- handled by the
            # sound bilinear interval-matmul reach (a sound over-approximation,
            # never the exact image).
            return _handle_bilinear_matmul(
                first_input, second_input, node, node_values, node_shapes)

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
        # The running set's dim is the authoritative count of flat input
        # variables. node_shapes is only a hint and can be inflated by
        # broadcasting in shape inference — e.g. a 1-D `W @ x` model whose
        # bias add broadcasts (M, 1) + (M,) -> (M, M), reporting M*M elements
        # for an M-vector. Trust the set dim; fall back to in_shape / K only
        # when no set is present.
        if input_sets_op:
            size = input_sets_op[0].dim
        elif in_shape is not None:
            size = int(np.prod(in_shape))
        else:
            size = K
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
            dummy_linear = nn.Linear(s.dim, weight_matrix.shape[0], bias=False,
                                     dtype=torch.float64)
            dummy_linear.weight.data = torch.tensor(weight_matrix.T, dtype=torch.float64)
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

    # ImageZono + Zono -> flatten ImageZono in CHW order, like
    # flatten_to_star above (its c/V are stored in HWC order)
    if isinstance(sa, ImageZono) and isinstance(sb, Zono) and not isinstance(sb, ImageZono):
        return _flatten_imagezono_chw(sa), sb
    if isinstance(sb, ImageZono) and isinstance(sa, Zono) and not isinstance(sa, ImageZono):
        return sa, _flatten_imagezono_chw(sb)

    return sa, sb


def _same_predicate_system(sa, sb) -> bool:
    """Do two stars constrain the IDENTICAL predicate vector?

    True only when predicate counts, constraint systems (C, d), and
    predicate bounds all match exactly — e.g. two branches that applied
    only affine layers to a common ancestor. Predicate COUNT alone is
    not sufficient: two branches can each append one ReLU-relaxation
    predicate (same nVar) with different constraints, and sharing those
    would be unsound.

    Equal systems are read as the same predicate variables. Equal rows
    and bounds alone do not make that true: two different ops can append
    variables with the same rows and bounds for different functions (e.g.
    a box-only Sin and Cos variable). The graph reach makes it hold by
    giving each node's new variables a tag of their own
    (:func:`_tag_new_predicates`), and ops that re-encode their output
    from bounds use ``Star.fresh_from_bounds``, so equal systems come from
    the same node.
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


def _pair_set_lists(set_lists: List[List]) -> List[tuple]:
    """Operand tuples for an element-wise op over set lists (one list per
    operand, each a union of pieces).

    Each input value lies in some piece of every operand's list, and the op
    must see those pieces together. Index i of one list says nothing about
    index i of another (a parallel reach returns pieces in completion
    order), so:

      * a list with one set is broadcast across the others;
      * with several multi-piece lists, one of them is the reference. Each
        reference piece is combined, in every other list, with the pieces
        it descends from: those whose predicate system is identical to, or
        a prefix of, its own (:func:`_lineage_groups`);
      * a reference piece without such a match in a list is combined with
        every piece of that list (the lists were split independently).
        The per-tuple op joins differing predicate systems
        block-diagonally, which is sound.

    The reference is the list that gives the fewest tuples. An empty list
    (an operand that reaches nothing) gives no tuples.
    """
    lists = [list(sl) for sl in set_lists]
    if not lists or any(len(sl) == 0 for sl in lists):
        return []
    multi = [i for i, sl in enumerate(lists) if len(sl) > 1]
    if len(multi) <= 1:
        n = max(len(sl) for sl in lists)
        return [tuple(sl[j] if len(sl) > 1 else sl[0] for sl in lists)
                for j in range(n)]

    best = None
    for r in multi:
        groups = {m: _lineage_groups(lists[r], lists[m])
                  for m in multi if m != r}
        n_tuples = sum(int(np.prod([len(groups[m][p]) for m in groups]))
                       for p in range(len(lists[r])))
        if best is None or n_tuples < best[0]:
            best = (n_tuples, r, groups)
    n_tuples, r, groups = best

    tuples = []
    for p, piece in enumerate(lists[r]):
        choices = []
        for i, sl in enumerate(lists):
            if i == r:
                choices.append([piece])
            elif len(sl) == 1:
                choices.append(sl)
            else:
                choices.append([sl[q] for q in groups[i][p]])
        tuples.extend(itertools.product(*choices))
    if len(tuples) > max(len(sl) for sl in lists):
        logger.warning(
            "pieces of set lists of sizes %s do not descend from one "
            "another; combining them gives %d sets",
            [len(sl) for sl in lists], len(tuples))
    return tuples


def _predicate_system(s: Any) -> Optional[Tuple[np.ndarray, ...]]:
    """(C, d, lb, ub) of a star's predicate system as float64 arrays of
    shapes (m, n), (m,), (n,), (n,); None for a set without one."""
    if not isinstance(s, (Star, ImageStar)):
        return None
    n = int(s.nVar)
    C = np.asarray(s.C if s.C is not None else np.zeros((0, n)),
                   dtype=np.float64)
    C = C.reshape(-1, n) if C.size else np.zeros((0, n))
    d = np.asarray(s.d if s.d is not None else np.zeros(0),
                   dtype=np.float64).reshape(-1)
    if d.size != C.shape[0]:
        return None

    def _box(p, fill):
        if p is None:
            return np.full(n, fill)
        return np.asarray(p, dtype=np.float64).reshape(-1)

    lb, ub = _box(s.predicate_lb, -np.inf), _box(s.predicate_ub, np.inf)
    if lb.size != n or ub.size != n:
        return None
    # + 0.0 turns -0.0 into 0.0 so equal systems give equal bytes
    return C + 0.0, d + 0.0, lb + 0.0, ub + 0.0


def _system_key(C: np.ndarray, d: np.ndarray, lb: np.ndarray,
                ub: np.ndarray) -> tuple:
    """Hashable key of a predicate system: equal systems, equal keys."""
    return (C.shape, np.ascontiguousarray(C).tobytes(),
            np.ascontiguousarray(d).tobytes(),
            np.ascontiguousarray(lb).tobytes(),
            np.ascontiguousarray(ub).tobytes())


def _lineage_groups(ref: List, other: List) -> List[List[int]]:
    """For each piece of ``ref``, the indices of the ``other`` pieces it is
    combined with (see :func:`_pair_set_lists`).

    Set ops keep a star's predicate system in front and append to it: new
    constraint rows, and new predicate variables with zero coefficients in
    the old rows (relaxation variables, McCormick products, the second
    operand of a block-diagonal join); an exact split appends one row to
    each child. A piece derived from an ``other`` piece therefore has that
    piece's system as a prefix: the leading rows and variables, and the
    predicate box on those variables, agree exactly. This relies on the
    same premise as :func:`_same_predicate_system`: equal systems are the
    same predicate variables, which the per-node tags of
    :func:`_tag_new_predicates` make hold in the graph reach.

    When exactly one distinct system in ``other`` is identical to, or a
    prefix of, the piece's system, the piece is combined with every
    ``other`` piece that has it. A strict prefix must have at least one
    constraint row. A re-boxing op builds its fresh variables with
    ``Star.fresh_from_bounds``, whose leading tag variable keeps its system
    from equalling or extending any system built elsewhere (a plain
    ``Star.from_bounds`` system depends only on how many widths are
    nonzero). With no match, or several (split children differ in the row
    their split added, so this is rare), the piece is combined with all of
    ``other``.
    """
    every = list(range(len(other)))
    systems = [_predicate_system(s) for s in other]
    if any(sy is None for sy in systems):
        return [every] * len(ref)
    by_shape: Dict[Tuple[int, int], Dict[tuple, List[int]]] = {}
    for q, (C, d, lb, ub) in enumerate(systems):
        by_shape.setdefault(C.shape, {}).setdefault(
            _system_key(C, d, lb, ub), []).append(q)

    groups = []
    for s in ref:
        sy = _predicate_system(s)
        if sy is None:
            groups.append(every)
            continue
        C, d, lb, ub = sy
        m_s, n_s = C.shape
        found = []
        for (m, n), keyed in by_shape.items():
            if m > m_s or n > n_s:
                continue
            if (m, n) != (m_s, n_s):
                if m == 0 or np.any(C[:m, n:] != 0):
                    continue
            hit = keyed.get(_system_key(C[:m, :n], d[:m], lb[:n], ub[:n]))
            if hit is not None:
                found.append(hit)
        groups.append(found[0] if len(found) == 1 else every)
    return groups


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
    # Residual branches can carry different pieces: one branch may split
    # (exact-star ReLU -> N sets partitioning the input) while the other stays
    # affine, or both may split independently. _pair_set_lists forms the
    # pairs that cover every input; the per-pair add below stays sound (exact
    # when the predicate systems match, Minkowski-join otherwise).
    output_sets = []

    for sa, sb in _pair_set_lists([sets_a, sets_b]):
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
            # (rounded outward)
            if op_name == 'add' or 'add' in op_name:
                lb_out = _add_dir(sa.lb, sb.lb, -1.0)
                ub_out = _add_dir(sa.ub, sb.ub, 1.0)
            else:
                lb_out = _add_dir(sa.lb, -sb.ub, -1.0)
                ub_out = _add_dir(sa.ub, -sb.lb, 1.0)

            out = Box(lb_out, ub_out)
            output_sets.append(out)

        else:
            raise NotImplementedError(
                f"Residual {op_name} not supported for "
                f"{type(sa).__name__} and {type(sb).__name__}"
            )

    return output_sets


def _mul_sets(sets_a: List, sets_b: List,
              lp_solver: str = 'default') -> List:
    """
    Element-wise multiplication of two lists of computed sets.

    For Box: standard interval arithmetic.
    For Zono: extract bounds, compute interval product, build new Zono.
    For Star: McCormick relaxation (tightest convex relaxation of bilinear terms).

    Args:
        sets_a: List of sets from the first operand
        sets_b: List of sets from the second operand
        lp_solver: LP solver for the McCormick operand ranges (a solver
            outside the HiGHS family is replaced by HiGHS, see
            ``n2v.utils.lpsolver._bound_lp_solver``)

    Returns:
        List of output sets
    """
    # Pieces are paired as in _add_sets: a bilinear Mul(f, g) where f split
    # into N input-region pieces and g stayed affine (1 set) is, per piece,
    # A_i * g over A_i's constraints; independently split operands get every
    # pair. The McCormick relaxation below stays sound for each pair.
    output_sets = []

    for sa, sb in _pair_set_lists([sets_a, sets_b]):
        # Coerce mismatched types (e.g., ImageStar * Star)
        sa, sb = _coerce_set_types(sa, sb)

        if isinstance(sa, Box) and isinstance(sb, Box):
            output_sets.append(_mul_boxes(sa, sb))
        elif isinstance(sa, Zono) and isinstance(sb, Zono):
            output_sets.append(_mul_zonos(sa, sb))
        elif isinstance(sa, Star) and isinstance(sb, Star):
            output_sets.append(_mul_stars_mccormick(sa, sb, lp_solver))
        else:
            raise NotImplementedError(
                f"Element-wise mul not supported for "
                f"{type(sa).__name__} and {type(sb).__name__}"
            )

    return output_sets


def _mul_boxes(a: 'Box', b: 'Box') -> 'Box':
    """Element-wise multiplication of two Boxes via interval arithmetic,
    with each product rounded outward."""
    pairs = [(a.lb, b.lb), (a.lb, b.ub), (a.ub, b.lb), (a.ub, b.ub)]
    lo = np.min([_mul_dir(x, y, -1.0) for x, y in pairs], axis=0)
    hi = np.max([_mul_dir(x, y, 1.0) for x, y in pairs], axis=0)
    return Box(lo, hi)


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


def _mccormick_operand_ranges(star: 'Star', solver,
                              operand: str) -> Tuple[np.ndarray, np.ndarray]:
    """Per-dimension ranges of one operand of a McCormick Mul/Div, before
    the outward pad the callers apply.

    The ranges come from ``Star.get_ranges``: the LP range, or the rigorous
    interval estimate ``Star.estimate_range`` on a dimension whose LP
    returns no optimum (a solver failure, or a presolve 'infeasible' that is
    not a certificate). The envelopes need a bounded box, so a range that is
    still not finite raises.

    Args:
        star: the operand
        solver: LP solver (already mapped by ``_bound_lp_solver``)
        operand: the operand's name, for the error message

    Returns:
        ``(lo, hi)``, 1-D float64 arrays of length ``star.dim``.
    """
    lo, hi = star.get_ranges(lp_solver=solver)
    lo = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(hi, dtype=np.float64).reshape(-1)
    bad = ~(np.isfinite(lo) & np.isfinite(hi))
    if np.any(bad):
        i = int(np.argmax(bad))
        raise ValueError(
            f"McCormick: no LP optimum and no finite interval estimate for "
            f"dimension {i} of the {operand} ([{lo[i]}, {hi[i]}])")
    return lo, hi


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

    # Get bounds for both operands via LP (HiGHS whatever lp_solver is: the
    # outward pad below is sized for it, see _bound_lp_solver)
    solver = _bound_lp_solver(lp_solver)
    lbs_a, ubs_a = _mccormick_operand_ranges(sa, solver, "first operand")
    lbs_b, ubs_b = _mccormick_operand_ranges(sb, solver, "second operand")

    # SOUNDNESS: the envelopes and the z box below are valid only on a box
    # that contains the operands; pad the LP ranges outward.
    a, b = _outward_lp_ranges(lbs_a, ubs_a, sa_V[:, :n + 1])  # bounds on x
    c, d = _outward_lp_ranges(lbs_b, ubs_b, sb_V[:, :n + 1])  # bounds on y

    # Interval arithmetic for z predicate bounds (rounded outward: the
    # verification LP enforces the predicate box without slack)
    pairs = [(a, c), (a, d), (b, c), (b, d)]
    z_lb = np.min([_mul_dir(x, y, -1.0) for x, y in pairs], axis=0)
    z_ub = np.max([_mul_dir(x, y, 1.0) for x, y in pairs], axis=0)

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
    convex over-approximation in that case. Pieces are paired as in
    :func:`_add_sets` (a split numerator against a single denominator set
    gives one quotient per numerator piece).
    """
    output_sets = []
    for sa, sb in _pair_set_lists([sets_a, sets_b]):
        sa, sb = _coerce_set_types(sa, sb)

        if isinstance(sa, Box) and isinstance(sb, Box):
            blb = np.asarray(sb.lb, dtype=np.float64).flatten()
            bub = np.asarray(sb.ub, dtype=np.float64).flatten()
            if np.any((blb <= 0) & (bub >= 0)):
                raise NotImplementedError(
                    "Element-wise Div: denominator range straddles zero")
            alb = np.asarray(sa.lb, dtype=np.float64).flatten()
            aub = np.asarray(sa.ub, dtype=np.float64).flatten()
            pairs = [(alb, blb), (alb, bub), (aub, blb), (aub, bub)]
            q_lo = np.min([_div_dir(x, y, -1.0) for x, y in pairs], axis=0)
            q_hi = np.max([_div_dir(x, y, 1.0) for x, y in pairs], axis=0)
            output_sets.append(Box(q_lo, q_hi))
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

    # Per-dimension ranges of numerator x and denominator y via LP (HiGHS,
    # see _mul_stars_mccormick)
    solver = _bound_lp_solver(lp_solver)
    a, b = _mccormick_operand_ranges(sa, solver, "numerator")
    c, d = _mccormick_operand_ranges(sb, solver, "denominator")

    # SOUNDNESS: pad the LP ranges outward (see _mul_stars_mccormick),
    # before the zero-straddle check so a denominator that reaches zero
    # within the LP tolerance is rejected.
    a, b = _outward_lp_ranges(a, b, sa_V[:, :n + 1])
    c, d = _outward_lp_ranges(c, d, sb_V[:, :n + 1])

    if np.any((c <= 0) & (d >= 0)):
        bad = int(np.argmax((c <= 0) & (d >= 0)))
        raise NotImplementedError(
            f"Element-wise Div: denominator range [{c[bad]}, {d[bad]}] "
            f"(dimension {bad}) straddles zero")

    # Interval division bounds for the quotient z (rounded outward)
    pairs = [(a, c), (a, d), (b, c), (b, d)]
    z_lb = np.min([_div_dir(x, y, -1.0) for x, y in pairs], axis=0)
    z_ub = np.max([_div_dir(x, y, 1.0) for x, y in pairs], axis=0)

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
    (constant). Division of non-Box sets is handled by
    :func:`_div_sets_by_constant` passing 1/constant.

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

            if scale.size == 1:
                # Scalar broadcast: multiply every element by the scalar.
                scale_4d = scale.reshape(1, 1, 1, 1)
            elif scale.size == c:
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

            if scale.size == 1:
                # Scalar broadcast: multiply every element by the scalar.
                scale_flat = scale.reshape(1, 1)
            elif scale.size == c_ch:
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
            scale_col = np.asarray(scale, dtype=np.float64).reshape(-1, 1)
            new_lb = np.minimum(_mul_dir(scale_col, s.lb, -1.0),
                                _mul_dir(scale_col, s.ub, -1.0))
            new_ub = np.maximum(_mul_dir(scale_col, s.lb, 1.0),
                                _mul_dir(scale_col, s.ub, 1.0))
            out = Box(new_lb, new_ub)
            output_sets.append(out)

        else:
            raise NotImplementedError(
                f"Mul by constant not supported for {type(s).__name__}"
            )

    return output_sets


def _div_sets_by_constant(input_sets: List, divisor: np.ndarray) -> List:
    """Element-wise division of sets by a constant (ONNX Div whose second
    operand is a parameter).

    A Box is divided directly, each quotient rounded outward (the scale by
    ``1 / divisor`` would round the reciprocal to nearest, which the outward
    product does not cover, so the Box could miss the exact quotient). Other
    set types are scaled by ``1 / divisor`` (:func:`_mul_sets_by_constant`).

    Args:
        input_sets: List of input sets
        divisor: Divisor, scalar or one entry per flat dimension

    Returns:
        List of output sets
    """
    divisor = np.asarray(divisor, dtype=np.float64).flatten()
    if not any(isinstance(s, Box) for s in input_sets):
        return _mul_sets_by_constant(input_sets, 1.0 / divisor)
    if np.any(divisor == 0):
        raise NotImplementedError("Div by a constant with a zero entry")
    output_sets = []
    for s in input_sets:
        if isinstance(s, Box):
            c = divisor.reshape(-1, 1)
            q_lo = np.minimum(_div_dir(s.lb, c, -1.0), _div_dir(s.ub, c, -1.0))
            q_hi = np.maximum(_div_dir(s.lb, c, 1.0), _div_dir(s.ub, c, 1.0))
            output_sets.append(Box(q_lo, q_hi))
        else:
            output_sets.extend(_mul_sets_by_constant([s], 1.0 / divisor))
    return output_sets


def _concat_sets(set_lists: List[List], axis: int) -> List:
    """
    Concatenate multiple lists of sets along a specified axis.

    Inputs whose predicate systems are identical share predicate
    variables and are stacked exactly; distinct systems are joined
    block-diagonally (:func:`_join_star_systems`). The pieces of the input
    lists are combined as in :func:`_add_sets` (:func:`_pair_set_lists`).

    Args:
        set_lists: List of lists of sets, one list per concat input,
                   e.g. [[s1_a, s1_b], [s2_a, s2_b]]
        axis: Concatenation axis (0 for feature dim of flat sets,
              2 for channel dim of ImageStar HWC)

    Returns:
        List of concatenated output sets
    """
    output_sets = []

    for combo in _pair_set_lists(set_lists):
        sets_to_concat = list(combo)

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


def _handle_batchnorm_flat(
    module: Any,
    node: Any,
    node_values: Dict[str, List],
    node_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Optional[List]:
    """Shape-aware BatchNorm (inference) for FLAT sets in ``(N, C, ...)`` layout.

    BatchNorm normalizes over axis 1, and the flat set is the row-major
    flattened tensor. When C spans only part of the tensor the plain C-sized
    diagonal affine does not match the flat dim -- e.g. the ViT per-token norm
    on ``(1, 48, 17)`` (48 channels x 17 tokens = 816 flat dims). Expand the
    per-channel scale/shift to one value per flat element (broadcasting over the
    non-channel axes) and apply the elementwise affine.

    Returns ``None`` (defer to the channel-aware dispatcher) for
    ImageStar/ImageZono, when no shape metadata is available, or for the plain
    ``dim == num_features`` case the existing handler already covers.
    """
    if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return None
    first_arg = node.args[0] if node.args else None
    if not (hasattr(first_arg, 'name') and first_arg.name in node_values):
        return None
    input_sets = node_values[first_arg.name]
    if not input_sets:
        return None
    first = input_sets[0]
    if isinstance(first, (ImageStar, ImageZono)):
        return None
    shape = (node_shapes or {}).get(first_arg.name)
    if shape is None:
        return None
    shape = tuple(int(s) for s in shape)
    total = int(np.prod(shape))
    C = int(module.num_features)
    # Plain vector case (dim == C) is handled exactly by the generic path.
    if len(shape) < 2 or total == C or shape[1] != C or total != first.dim:
        return None
    scale, shift = _get_bn_params(module)
    bshape = [1] * len(shape)
    bshape[1] = C

    def _expand(v):
        return np.broadcast_to(
            np.asarray(v, dtype=np.float64).reshape(bshape), shape).reshape(-1)

    full_scale = _expand(scale)
    full_shift = _expand(shift)
    # A Box takes the elementwise map with the rounding margin of
    # batchnorm_box, which also covers the rounding of the folded scale and
    # shift against the real-number layer (affine_map is rigorous only for
    # the already-rounded ones).
    full_fold_mag = (_expand(_bn_fold_mag(module, scale))
                     if any(isinstance(s, Box) for s in input_sets) else None)
    W = None
    output_sets = []
    for s in input_sets:
        if isinstance(s, Box):
            output_sets.append(_batchnorm_single_box(
                s, full_scale, full_shift, full_fold_mag))
            continue
        if W is None:
            W = np.diag(full_scale)
        output_sets.append(s.affine_map(W, full_shift.reshape(-1, 1)))
    return output_sets


def _handle_softmax_flat(
    module: Any,
    node: Any,
    node_values: Dict[str, List],
    node_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Optional[List]:
    """Shape-aware Softmax for FLAT sets (the transformer attention softmax).

    The generic ``softmax_star`` only handles softmax over the last axis of a
    flat ``(1, N)`` tensor (one global softmax). The attention weights are a
    batched ``(..., L, L)`` softmax over a single axis, so route to the sound
    correlated row-softmax bound (``softmax_attn_*``) with the right shape and
    axis. Handles both the opset<=11 ONNX semantics (softmax over the flattened
    tail ``[axis, end)``) and the single-axis torch / opset>=13 semantics.

    Returns ``None`` (defer to the generic dispatcher) for ImageStar/ImageZono,
    LogSoftmax, or when shape metadata is unavailable.
    """
    cls = type(module).__name__
    is_v1v11 = cls == 'OnnxSoftmaxV1V11'
    if not (is_v1v11 or isinstance(module, nn.Softmax) or cls == 'OnnxSoftmax'):
        return None
    if getattr(module, 'is_log', False):
        return None
    first_arg = node.args[0] if node.args else None
    if not (hasattr(first_arg, 'name') and first_arg.name in node_values):
        return None
    input_sets = node_values[first_arg.name]
    if not input_sets:
        return None
    first = input_sets[0]
    if isinstance(first, (ImageStar, ImageZono)):
        return None
    shape = (node_shapes or {}).get(first_arg.name)
    if shape is None:
        return None
    shape = tuple(int(s) for s in shape)
    axis = getattr(module, 'axis', None)
    if axis is None:
        axis = getattr(module, 'dim', -1)
    if axis is None:
        axis = -1
    a = axis if axis >= 0 else axis + len(shape)
    if not (0 <= a < len(shape)):
        return None
    if is_v1v11:
        # opset<=11 Softmax(axis=a): coerce to 2D rows=[0,a), cols=[a,end) and
        # softmax over the flattened columns. For the attention axis (the last
        # one) this reduces to a per-row softmax over the key axis.
        rows = int(np.prod(shape[:a])) if a > 0 else 1
        cols = int(np.prod(shape[a:]))
        sm_shape, sm_axis = (rows, cols), -1
    else:
        # torch / opset>=13 Softmax: single-axis.
        sm_shape, sm_axis = shape, a
    if isinstance(first, Star):
        return softmax_attn_star(input_sets, sm_shape, axis=sm_axis)
    if isinstance(first, Zono):
        return softmax_attn_zono(input_sets, sm_shape, axis=sm_axis)
    if isinstance(first, Box):
        return softmax_attn_box(input_sets, sm_shape, axis=sm_axis)
    return None


def _const_set_like(template: Any, const_flat: np.ndarray) -> Any:
    """A constant (zero-uncertainty) set matching ``template``'s type and
    predicate system, centered at ``const_flat``.

    Lets ONNX Concat bring a constant operand (e.g. the ViT cls token) into the
    set algebra so it occupies real dimensions rather than being dropped. The
    result shares the template's predicates with all-zero generators, so the
    concatenation stays exact (no new uncertainty, no spurious correlation)."""
    const_flat = np.asarray(const_flat, dtype=np.float64).reshape(-1)
    d = const_flat.size
    if isinstance(template, Star):
        V = np.zeros((d, template.V.shape[1]))
        V[:, 0] = const_flat
        return Star(V, template.C, template.d,
                    template.predicate_lb, template.predicate_ub)
    if isinstance(template, Zono):
        V = np.zeros((d, template.V.shape[1]))
        return Zono(const_flat.reshape(-1, 1), V)
    if isinstance(template, Box):
        return Box(const_flat.copy(), const_flat.copy())
    raise NotImplementedError(
        f"Concat with a constant operand is not supported for set type "
        f"{type(template).__name__}")


def _handle_onnx_concat(
    module: Any,
    node: Any,
    node_values: Dict[str, List],
    graph_module: fx.GraphModule,
    node_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Optional[List]:
    """
    Handle ONNX Concat operations.

    Collects sets from all input nodes, maps the ONNX axis (which includes
    the batch dimension) to the set axis, and calls _concat_sets. Constant
    operands (frozen initializers, e.g. the ViT cls token) are materialized
    as zero-uncertainty sets so they occupy their dimensions instead of being
    silently dropped.

    Args:
        module: OnnxConcat module (has .axis attribute)
        node: Graph node
        node_values: Dict mapping node names to lists of sets
        graph_module: owning GraphModule (to resolve constant operands)

    Returns:
        List of concatenated output sets, or None if inputs not found
    """
    onnx_axis = module.axis

    # A constant operand (e.g. the ViT cls token) is not in node_values. Find a
    # real set first to use as the predicate-system template, then materialize
    # any constant operands as zero-uncertainty sets. Dropping them would
    # undersize the concat output (the vit_2023 `_0_concat_2` bug: the set
    # stayed at 16 tokens while the shape said 17).
    template = None
    for arg in node.args:
        if hasattr(arg, 'name') and arg.name in node_values:
            sl = node_values[arg.name]
            if sl:
                template = sl[0]
                break

    # Collect set lists from all input arguments (in operand order)
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
        elif template is not None:
            # An operand with no sets must fold to a constant tensor;
            # skipping it would undersize the output and shift the operands
            # after it.
            try:
                const_np = np.asarray(
                    _get_parameter(graph_module, arg).numpy(),
                    dtype=np.float64)
            except Exception as exc:  # noqa: BLE001 — re-raised below
                raise NotImplementedError(
                    f"Concat {node.name!r}: operand "
                    f"{getattr(arg, 'name', arg)!r} has no reachable sets "
                    f"and could not be folded to a constant") from exc
            set_lists.append([_const_set_like(template, const_np.reshape(-1))])
            shp = (node_shapes or {}).get(arg.name)
            shapes.append(shp if shp is not None else tuple(const_np.shape))

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
        s: Input set (Star, ImageStar, Zono, ImageZono, Box)
        slices_by_axis: Dict mapping axis (int) to Python slice object.
            For flat sets (Star, Zono, Box), axis 0 is the dimension axis.
            For ImageStar/ImageZono, axes are in HWC format: 0=H, 1=W, 2=C.

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

    elif isinstance(s, ImageZono):
        # Before the Zono branch (ImageZono is a Zono subclass): c and V
        # are flat in HWC order, so view them as (H, W, C, ...) and slice
        # the first three axes, as for ImageStar.
        h, w, c = s.height, s.width, s.num_channels
        n_gen = s.V.shape[1]
        idx = [slice(None)] * 4
        for ax, sl in slices_by_axis.items():
            if ax < 3:
                idx[ax] = sl
        c_img = s.c.reshape(h, w, c, 1)[tuple(idx)]
        V_img = s.V.reshape(h, w, c, n_gen)[tuple(idx)]
        h_out, w_out, c_out = c_img.shape[:3]
        return ImageZono(
            c_img.reshape(-1, 1), V_img.reshape(-1, n_gen),
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
