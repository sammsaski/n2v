"""
Layer reachability dispatcher - works directly with PyTorch layer types.

Dispatches reachability computation based on PyTorch layer type without
requiring custom layer wrapper classes.
"""

import torch.nn as nn
import numpy as np
from typing import List, Union
from n2v.sets import Star, Zono, Box


def reach_layer_star(
    layer: nn.Module,
    input_stars: List[Star],
    method: str = 'exact',
    **kwargs
) -> List[Star]:
    """
    Compute reachable sets through a PyTorch layer using Star sets.

    Args:
        layer: PyTorch layer (nn.Linear, nn.ReLU, nn.Conv2d, etc.)
        input_stars: List of input Star sets
        method: 'exact' or 'approx'
        **kwargs: Additional options (lp_solver, relax_factor, etc.)

    Returns:
        List of output Star sets
    """
    from . import linear_reach, relu_reach, conv2d_reach, flatten_reach, maxpool2d_reach, avgpool2d_reach

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_star(layer, input_stars)

    elif isinstance(layer, nn.ReLU):
        lp_solver = kwargs.get('lp_solver', 'default')
        dis_opt = kwargs.get('dis_opt', None)
        if method == 'exact':
            return relu_reach.relu_star_exact(input_stars, lp_solver=lp_solver, dis_opt=dis_opt)
        else:
            relax_factor = kwargs.get('relax_factor', 0.5)
            relax_method = kwargs.get('relax_method', 'standard')
            return relu_reach.relu_star_approx(input_stars, relax_factor, lp_solver, relax_method)

    elif isinstance(layer, nn.Conv2d):
        return conv2d_reach.conv2d_star(layer, input_stars, method=method, **kwargs)

    elif isinstance(layer, nn.MaxPool2d):
        lp_solver = kwargs.get('lp_solver', 'default')
        dis_opt = kwargs.get('dis_opt', None)
        return maxpool2d_reach.maxpool2d_star(layer, input_stars, method=method, lp_solver=lp_solver, dis_opt=dis_opt, **kwargs)

    elif isinstance(layer, nn.AvgPool2d):
        return avgpool2d_reach.avgpool2d_star(layer, input_stars, **kwargs)

    elif isinstance(layer, nn.Flatten):
        return flatten_reach.flatten_star(layer, input_stars)

    elif isinstance(layer, nn.Identity):
        return input_stars

    elif isinstance(layer, nn.Sequential):
        # Recursively handle Sequential
        current_stars = input_stars
        for sublayer in layer:
            current_stars = reach_layer_star(sublayer, current_stars, method, **kwargs)
        return current_stars

    else:
        raise NotImplementedError(
            f"Star reachability not implemented for layer type: {type(layer).__name__}"
        )


def reach_layer_zono(
    layer: nn.Module,
    input_zonos: List[Zono]
) -> List[Zono]:
    """
    Compute reachable sets through a PyTorch layer using Zonotopes.

    Args:
        layer: PyTorch layer
        input_zonos: List of input Zonotopes

    Returns:
        List of output Zonotopes
    """
    from . import linear_reach, relu_reach, flatten_reach, maxpool2d_reach, avgpool2d_reach

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_zono(layer, input_zonos)

    elif isinstance(layer, nn.ReLU):
        return relu_reach.relu_zono_approx(input_zonos)

    elif isinstance(layer, nn.MaxPool2d):
        return maxpool2d_reach.maxpool2d_zono(layer, input_zonos)

    elif isinstance(layer, nn.AvgPool2d):
        return avgpool2d_reach.avgpool2d_zono(layer, input_zonos)

    elif isinstance(layer, nn.Flatten):
        return flatten_reach.flatten_zono(layer, input_zonos)

    elif isinstance(layer, nn.Identity):
        return input_zonos

    elif isinstance(layer, nn.Sequential):
        current_zonos = input_zonos
        for sublayer in layer:
            current_zonos = reach_layer_zono(sublayer, current_zonos)
        return current_zonos

    else:
        raise NotImplementedError(
            f"Zono reachability not implemented for layer type: {type(layer).__name__}"
        )


def reach_layer_box(
    layer: nn.Module,
    input_boxes: List[Box]
) -> List[Box]:
    """
    Compute reachable sets through a PyTorch layer using Boxes.

    Args:
        layer: PyTorch layer
        input_boxes: List of input Boxes

    Returns:
        List of output Boxes
    """
    from . import linear_reach, relu_reach, flatten_reach

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_box(layer, input_boxes)

    elif isinstance(layer, nn.ReLU):
        return relu_reach.relu_box(input_boxes)

    elif isinstance(layer, nn.Flatten):
        return flatten_reach.flatten_box(layer, input_boxes)

    elif isinstance(layer, nn.Identity):
        return input_boxes

    elif isinstance(layer, nn.Sequential):
        current_boxes = input_boxes
        for sublayer in layer:
            current_boxes = reach_layer_box(sublayer, current_boxes)
        return current_boxes

    else:
        raise NotImplementedError(
            f"Box reachability not implemented for layer type: {type(layer).__name__}"
        )
