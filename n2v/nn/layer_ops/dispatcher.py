"""
Layer reachability dispatcher - routes computation based on layer type and set type.

Dispatches reachability computation for a single layer based on PyTorch layer type
and input set type, without requiring custom layer wrapper classes.
"""

import torch.nn as nn
from typing import List, Union

# Import set types
from n2v.sets import Star, Zono, Box, Hexatope, Octatope
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

# Import layer-specific reach functions
from . import linear_reach, relu_reach, conv2d_reach, flatten_reach
from . import maxpool2d_reach, avgpool2d_reach


def reach_layer(
    layer: nn.Module,
    input_sets: List,
    method: str = 'exact',
    **kwargs
) -> List:
    """
    Compute reachable sets through a PyTorch layer.

    Automatically detects the input set type and dispatches to the appropriate
    layer-specific implementation.

    Args:
        layer: PyTorch layer (nn.Linear, nn.ReLU, nn.Conv2d, etc.)
        input_sets: List of input sets (Star, Zono, Box, Hexatope, or Octatope)
        method: 'exact' or 'approx' (not all combinations supported)
        **kwargs: Additional options:
            - lp_solver: LP solver to use
            - verbose: Display option
            - parallel: Enable parallel processing
            - n_workers: Number of workers
            - relax_factor: Relaxation factor for approx methods
            - relax_method: Relaxation strategy

    Returns:
        List of output sets (same type as input)

    Raises:
        NotImplementedError: If layer/set combination is not supported
    """

    if not input_sets:
        return []

    # Detect set type from first input
    first_set = input_sets[0]

    # Route based on set type (including ImageStar/ImageZono as subclasses)
    if isinstance(first_set, (Star, ImageStar)):
        return _reach_layer_star(layer, input_sets, method, **kwargs)
    elif isinstance(first_set, (Zono, ImageZono)):
        return _reach_layer_zono(layer, input_sets, method, **kwargs)
    elif isinstance(first_set, Box):
        return _reach_layer_box(layer, input_sets, method, **kwargs)
    elif isinstance(first_set, Hexatope):
        return _reach_layer_hexatope(layer, input_sets, method, **kwargs)
    elif isinstance(first_set, Octatope):
        return _reach_layer_octatope(layer, input_sets, method, **kwargs)
    else:
        raise TypeError(
            f"Unsupported set type: {type(first_set).__name__}. "
            f"Supported: Star, ImageStar, Zono, ImageZono, Box, Hexatope, Octatope"
        )


def _reach_layer_star(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Star set reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_star(layer, input_sets)

    elif isinstance(layer, nn.ReLU):
        lp_solver = kwargs.get('lp_solver', 'default')
        verbose = kwargs.get('verbose', None)
        parallel = kwargs.get('parallel', None)
        n_workers = kwargs.get('n_workers', None)

        if method == 'exact':
            return relu_reach.relu_star_exact(
                input_sets, lp_solver=lp_solver, verbose=verbose,
                parallel=parallel, n_workers=n_workers
            )
        else:  # approx
            relax_factor = kwargs.get('relax_factor', 0.5)
            relax_method = kwargs.get('relax_method', 'standard')
            return relu_reach.relu_star_approx(
                input_sets, relax_factor, lp_solver, relax_method
            )

    elif isinstance(layer, nn.Conv2d):
        return conv2d_reach.conv2d_star(layer, input_sets, method=method, **kwargs)

    elif isinstance(layer, nn.MaxPool2d):
        lp_solver = kwargs.get('lp_solver', 'default')
        verbose = kwargs.get('verbose', None)
        return maxpool2d_reach.maxpool2d_star(
            layer, input_sets, method=method, lp_solver=lp_solver,
            verbose=verbose, **kwargs
        )

    elif isinstance(layer, nn.AvgPool2d):
        return avgpool2d_reach.avgpool2d_star(layer, input_sets, **kwargs)

    elif isinstance(layer, nn.Flatten):
        return flatten_reach.flatten_star(layer, input_sets)

    elif isinstance(layer, nn.Identity):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        # Recursively handle Sequential
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        raise NotImplementedError(
            f"Star reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_zono(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Zonotope reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_zono(layer, input_sets)

    elif isinstance(layer, nn.ReLU):
        return relu_reach.relu_zono_approx(input_sets)

    elif isinstance(layer, nn.MaxPool2d):
        return maxpool2d_reach.maxpool2d_zono(layer, input_sets)

    elif isinstance(layer, nn.AvgPool2d):
        return avgpool2d_reach.avgpool2d_zono(layer, input_sets)

    elif isinstance(layer, nn.Flatten):
        return flatten_reach.flatten_zono(layer, input_sets)

    elif isinstance(layer, nn.Identity):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        raise NotImplementedError(
            f"Zono reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_box(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Box reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_box(layer, input_sets)

    elif isinstance(layer, nn.ReLU):
        return relu_reach.relu_box(input_sets)

    elif isinstance(layer, nn.Flatten):
        return flatten_reach.flatten_box(layer, input_sets)

    elif isinstance(layer, nn.Identity):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        raise NotImplementedError(
            f"Box reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_hexatope(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Hexatope reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_hexatope(layer, input_sets)

    elif isinstance(layer, nn.ReLU):
        verbose = kwargs.get('verbose', None)
        if method == 'exact':
            return relu_reach.relu_hexatope_exact(input_sets, verbose=verbose)
        else:  # approx
            return relu_reach.relu_hexatope_approx(input_sets, verbose=verbose)

    elif isinstance(layer, nn.Flatten):
        return flatten_reach.flatten_hexatope(layer, input_sets)

    elif isinstance(layer, nn.Identity):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        raise NotImplementedError(
            f"Hexatope reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_octatope(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Octatope reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_octatope(layer, input_sets)

    elif isinstance(layer, nn.ReLU):
        verbose = kwargs.get('verbose', None)
        if method == 'exact':
            return relu_reach.relu_octatope_exact(input_sets, verbose=verbose)
        else:  # approx
            return relu_reach.relu_octatope_approx(input_sets, verbose=verbose)

    elif isinstance(layer, nn.Flatten):
        return flatten_reach.flatten_octatope(layer, input_sets)

    elif isinstance(layer, nn.Identity):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        raise NotImplementedError(
            f"Octatope reachability not implemented for layer type: {type(layer).__name__}"
        )
