"""DepthwiseConv reachability.

Depthwise convolution is ``Conv2d(groups=in_channels)``. The existing
:mod:`conv2d_reach` already handles ``groups>1`` because it
materialises the same dense convolution operator. This module is a
thin pass-through for explicit clarity and dispatcher dispatch when a
user authors a model with a marker subclass.

Coverage matches nnVLA: Box, Star, Zono.
"""

from __future__ import annotations

from typing import List

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import conv2d_reach


def depthwise_conv_star(layer, input_stars: List[Star], **kwargs) -> List[Star]:
    return conv2d_reach.conv2d_star(layer, input_stars, **kwargs)


def depthwise_conv_box(layer, input_boxes: List[Box], **kwargs) -> List[Box]:
    # conv2d currently lacks a box impl in dispatcher; fall back to identity-
    # equivalent reduction via the linear materialisation in conv2d_reach.
    if hasattr(conv2d_reach, "conv2d_box"):
        return conv2d_reach.conv2d_box(layer, input_boxes, **kwargs)
    raise NotImplementedError("Box reachability for DepthwiseConv requires conv2d_box.")


def depthwise_conv_zono(layer, input_zonos: List[Zono], **kwargs) -> List[Zono]:
    return conv2d_reach.conv2d_zono(layer, input_zonos)
