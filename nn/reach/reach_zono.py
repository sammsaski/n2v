"""
Zonotope-based reachability analysis for neural networks.
"""

import torch.nn as nn
from typing import List, Optional
from n2v.nn.layer_ops.dispatcher import reach_layer_zono


def reach_zono_approx(
    model: nn.Module,
    input_zonos: List,
    dis_opt: Optional[str] = None
) -> List:
    """
    Approximate reachability using Zonotopes.

    Args:
        model: PyTorch model
        input_zonos: List of input Zono sets
        dis_opt: 'display' to show progress

    Returns:
        List of output Zono sets
    """
    current_zonos = input_zonos

    layers = list(model.children())
    if not layers:
        layers = [model]

    for i, layer in enumerate(layers):
        if dis_opt == 'display':
            print(f'Layer {i+1}/{len(layers)}: {type(layer).__name__}')

        current_zonos = reach_layer_zono(layer, current_zonos)

        if dis_opt == 'display':
            print(f'  Output: {len(current_zonos)} Zono sets')

    return current_zonos
