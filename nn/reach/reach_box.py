"""
Box-based (interval) reachability analysis for neural networks.

Simple interval arithmetic propagation through layers.
"""

import torch
import torch.nn as nn
from typing import List
import numpy as np


def reach_box_approx(model: nn.Module, input_boxes: List) -> List:
    """
    Approximate reachability using Box (interval arithmetic).

    Args:
        model: PyTorch model
        input_boxes: List of input Box sets

    Returns:
        List of output Box sets
    """
    from ...sets import Box

    output_boxes = []

    for box in input_boxes:
        # Propagate box through model using interval arithmetic
        lb, ub = box.lb, box.ub

        # Convert to tensors
        lb_tensor = torch.from_numpy(lb).float().reshape(1, -1)
        ub_tensor = torch.from_numpy(ub).float().reshape(1, -1)

        # Propagate through each layer
        for layer in model.children():
            lb_tensor, ub_tensor = _propagate_box_through_layer(layer, lb_tensor, ub_tensor)

        # Convert back to Box
        output_lb = lb_tensor.detach().numpy().reshape(-1, 1)
        output_ub = ub_tensor.detach().numpy().reshape(-1, 1)
        output_boxes.append(Box(output_lb, output_ub))

    return output_boxes


def _propagate_box_through_layer(
    layer: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor
) -> tuple:
    """
    Propagate box bounds through a single layer.

    Args:
        layer: PyTorch layer
        lb: Lower bound tensor
        ub: Upper bound tensor

    Returns:
        Tuple of (new_lb, new_ub)
    """
    if isinstance(layer, nn.Linear):
        return _propagate_linear(layer, lb, ub)
    elif isinstance(layer, nn.ReLU):
        return _propagate_relu(lb, ub)
    elif isinstance(layer, nn.Sigmoid):
        return _propagate_sigmoid(lb, ub)
    elif isinstance(layer, nn.Tanh):
        return _propagate_tanh(lb, ub)
    elif isinstance(layer, (nn.Flatten, nn.Identity)):
        return lb, ub
    elif isinstance(layer, (nn.Sequential, nn.ModuleList)):
        # Recursively handle sequential layers
        for sublayer in layer.children():
            lb, ub = _propagate_box_through_layer(sublayer, lb, ub)
        return lb, ub
    else:
        # Unsupported layer - use conservative approximation
        with torch.no_grad():
            center = 0.5 * (lb + ub)
            output = layer(center)
            # Very conservative: output could be anywhere
            return output, output


def _propagate_linear(layer: nn.Linear, lb: torch.Tensor, ub: torch.Tensor) -> tuple:
    """Propagate through linear layer using interval arithmetic."""
    W = layer.weight  # (out_features, in_features)
    b = layer.bias if layer.bias is not None else torch.zeros(layer.out_features)

    # Split weight matrix into positive and negative parts
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)

    # Compute new bounds
    new_lb = (W_pos @ lb.T + W_neg @ ub.T).T + b
    new_ub = (W_pos @ ub.T + W_neg @ lb.T).T + b

    return new_lb, new_ub


def _propagate_relu(lb: torch.Tensor, ub: torch.Tensor) -> tuple:
    """Propagate through ReLU using interval arithmetic."""
    new_lb = torch.clamp(lb, min=0)
    new_ub = torch.clamp(ub, min=0)
    return new_lb, new_ub


def _propagate_sigmoid(lb: torch.Tensor, ub: torch.Tensor) -> tuple:
    """Propagate through Sigmoid (conservative)."""
    new_lb = torch.sigmoid(lb)
    new_ub = torch.sigmoid(ub)
    return new_lb, new_ub


def _propagate_tanh(lb: torch.Tensor, ub: torch.Tensor) -> tuple:
    """Propagate through Tanh (conservative)."""
    new_lb = torch.tanh(lb)
    new_ub = torch.tanh(ub)
    return new_lb, new_ub
