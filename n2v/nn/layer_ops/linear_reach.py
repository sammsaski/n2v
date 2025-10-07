"""
Linear layer reachability operations.

Works directly with PyTorch nn.Linear layers.
"""

import torch.nn as nn
import numpy as np
from typing import List
from n2v.sets import Star, Zono, Box, Hexatope, Octatope


def linear_star(layer: nn.Linear, input_stars: List[Star]) -> List[Star]:
    """
    Exact reachability for Linear layer using Star sets.

    Args:
        layer: PyTorch nn.Linear layer
        input_stars: List of input Star sets

    Returns:
        List of output Star sets
    """
    W = layer.weight.detach().cpu().numpy()  # (out_features, in_features)
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    output_stars = []
    for star in input_stars:
        if b is not None:
            b_reshaped = b.reshape(-1, 1)
            output_star = star.affine_map(W, b_reshaped)
        else:
            output_star = star.affine_map(W)
        output_stars.append(output_star)

    return output_stars


def linear_zono(layer: nn.Linear, input_zonos: List[Zono]) -> List[Zono]:
    """
    Exact reachability for Linear layer using Zonotopes.

    Args:
        layer: PyTorch nn.Linear layer
        input_zonos: List of input Zonotopes

    Returns:
        List of output Zonotopes
    """
    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    output_zonos = []
    for zono in input_zonos:
        if b is not None:
            b_reshaped = b.reshape(-1, 1)
            output_zono = zono.affine_map(W, b_reshaped)
        else:
            output_zono = zono.affine_map(W)
        output_zonos.append(output_zono)

    return output_zonos


def linear_box(layer: nn.Linear, input_boxes: List[Box]) -> List[Box]:
    """
    Exact reachability for Linear layer using Boxes.

    Args:
        layer: PyTorch nn.Linear layer
        input_boxes: List of input Boxes

    Returns:
        List of output Boxes
    """
    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    output_boxes = []
    for box in input_boxes:
        if b is not None:
            b_reshaped = b.reshape(-1, 1)
            output_box = box.affine_map(W, b_reshaped)
        else:
            output_box = box.affine_map(W)
        output_boxes.append(output_box)

    return output_boxes


def linear_hexatope(layer: nn.Linear, input_hexatopes: List[Hexatope]) -> List[Hexatope]:
    """
    Exact reachability for Linear layer using Hexatopes.

    Args:
        layer: PyTorch nn.Linear layer
        input_hexatopes: List of input Hexatopes

    Returns:
        List of output Hexatopes
    """
    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    output_hexatopes = []
    for hexatope in input_hexatopes:
        if b is not None:
            b_reshaped = b.reshape(-1, 1)
            output_hexatope = hexatope.affine_map(W, b_reshaped)
        else:
            output_hexatope = hexatope.affine_map(W)
        output_hexatopes.append(output_hexatope)

    return output_hexatopes


def linear_octatope(layer: nn.Linear, input_octatopes: List[Octatope]) -> List[Octatope]:
    """
    Exact reachability for Linear layer using Octatopes.

    Args:
        layer: PyTorch nn.Linear layer
        input_octatopes: List of input Octatopes

    Returns:
        List of output Octatopes
    """
    W = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    output_octatopes = []
    for octatope in input_octatopes:
        if b is not None:
            b_reshaped = b.reshape(-1, 1)
            output_octatope = octatope.affine_map(W, b_reshaped)
        else:
            output_octatope = octatope.affine_map(W)
        output_octatopes.append(output_octatope)

    return output_octatopes
