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
