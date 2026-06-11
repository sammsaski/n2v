"""ConvTokenEmbedding reachability.

A token-embedding done by a Conv1d/Conv2d. Forwarder to the appropriate
existing module based on what the wrapper exposes.
"""

from __future__ import annotations

from typing import List

import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.nn.layer_ops import conv1d_reach, conv2d_reach


def conv_token_embedding_star(layer, input_stars: List[Star], **kwargs) -> List[Star]:
    proj = getattr(layer, "proj", layer)
    if isinstance(proj, nn.Conv2d):
        return conv2d_reach.conv2d_star(proj, input_stars, **kwargs)
    return conv1d_reach.conv1d_star(proj, input_stars, **kwargs)


def conv_token_embedding_box(layer, input_boxes: List[Box], **kwargs) -> List[Box]:
    proj = getattr(layer, "proj", layer)
    if isinstance(proj, nn.Conv1d) and hasattr(conv1d_reach, "conv1d_box"):
        return conv1d_reach.conv1d_box(proj, input_boxes, **kwargs)
    raise NotImplementedError("Box reachability for ConvTokenEmbedding not yet wired.")


def conv_token_embedding_zono(layer, input_zonos: List[Zono], **kwargs) -> List[Zono]:
    proj = getattr(layer, "proj", layer)
    if isinstance(proj, nn.Conv2d):
        return conv2d_reach.conv2d_zono(proj, input_zonos)
    return conv1d_reach.conv1d_zono(proj, input_zonos, **kwargs)
