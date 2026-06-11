"""GroupNorm reachability.

GroupNorm partitions channels into ``num_groups`` groups and applies
LayerNorm independently within each group. The per-group bounds use the
same interval-mean / interval-variance derivation as :mod:`layernorm_reach`,
applied to the per-group sub-vectors of the input.

Coverage matches nnVLA: Box (IBP), Star (CROWN/IBP fallback).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch.nn as nn

from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar
from n2v.nn.layer_ops._image_shape import apply_box_lift_star
from n2v.nn.layer_ops._layernorm_star import predicate_preserving_norm_star
from n2v.nn.layer_ops._norm_utils import (
    affine_after_norm,
    interval_mean_var,
    normalised_interval,
)


def _gn_params(layer: nn.GroupNorm):
    eps = float(layer.eps)
    num_groups = int(layer.num_groups)
    num_channels = int(layer.num_channels)
    weight = layer.weight.detach().cpu().numpy().astype(np.float64) if layer.weight is not None else None
    bias = layer.bias.detach().cpu().numpy().astype(np.float64) if layer.bias is not None else None
    return num_groups, num_channels, weight, bias, eps


def _groupnorm_interval(lb: np.ndarray, ub: np.ndarray, num_groups: int, num_channels: int, eps: float):
    """Sound interval reach for GroupNorm applied to an ``(C, ...)`` input.

    The input bounds are reshaped per-channel; channels are split into
    ``num_groups`` groups and each group is bounded by the LayerNorm
    interval helper.
    """
    lb = lb.reshape(-1)
    ub = ub.reshape(-1)
    if lb.size % num_channels != 0:
        raise ValueError(
            f"GroupNorm input length {lb.size} not divisible by num_channels={num_channels}"
        )
    spatial = lb.size // num_channels
    lb_c = lb.reshape(num_channels, spatial)
    ub_c = ub.reshape(num_channels, spatial)

    channels_per_group = num_channels // num_groups
    out_lb = np.zeros_like(lb_c)
    out_ub = np.zeros_like(ub_c)

    for g in range(num_groups):
        start = g * channels_per_group
        end = start + channels_per_group
        group_lb = lb_c[start:end].reshape(-1)
        group_ub = ub_c[start:end].reshape(-1)
        n_lb, n_ub = normalised_interval(group_lb, group_ub, eps=eps)
        out_lb[start:end] = n_lb.reshape(channels_per_group, spatial)
        out_ub[start:end] = n_ub.reshape(channels_per_group, spatial)

    return out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)


def groupnorm_box(layer: nn.GroupNorm, input_boxes: List[Box]) -> List[Box]:
    num_groups, num_channels, weight, bias, eps = _gn_params(layer)
    out: List[Box] = []
    for b in input_boxes:
        norm_lb, norm_ub = _groupnorm_interval(b.lb, b.ub, num_groups, num_channels, eps)
        # weight/bias are per-channel; broadcast across spatial.
        if weight is not None or bias is not None:
            spatial = norm_lb.size // num_channels
            w_b = np.repeat(weight, spatial) if weight is not None else None
            b_b = np.repeat(bias, spatial) if bias is not None else None
            norm_lb, norm_ub = affine_after_norm(norm_lb, norm_ub, w_b, b_b)
        out.append(Box(norm_lb, norm_ub))
    return out


def groupnorm_star_approx(layer: nn.GroupNorm, input_stars: List[Star]) -> List[Star]:
    """Predicate-preserving Star reach for GroupNorm.

    GroupNorm applies LayerNorm independently within each channel group.
    We carry the input predicates through each per-group affine map and
    add per-feature slack predicates for the per-group scale intervals.
    Falls back to box-lift when the input has no predicate basis.
    """
    num_groups, num_channels, weight, bias, eps = _gn_params(layer)
    channels_per_group = num_channels // num_groups

    def _box(lb, ub):
        norm_lb, norm_ub = _groupnorm_interval(lb, ub, num_groups, num_channels, eps)
        if weight is not None or bias is not None:
            spatial = norm_lb.size // num_channels
            w_b = np.repeat(weight, spatial) if weight is not None else None
            b_b = np.repeat(bias, spatial) if bias is not None else None
            norm_lb, norm_ub = affine_after_norm(norm_lb, norm_ub, w_b, b_b)
        return norm_lb, norm_ub

    output: List[Star] = []
    for s in input_stars:
        is_image = isinstance(s, ImageStar)
        base = s.to_star() if is_image else s
        if base.V is None or base.V.size == 0 or base.dim % num_channels != 0:
            # Box-lift fallback when predicates are absent or shape unknown.
            new_star = apply_box_lift_star([base], _box)[0]
        else:
            # T0-4 (audit C4): the predicate-preserving Star reach below
            # collapses per-group sigma intervals to a single cross-group
            # [sigma_lb, sigma_ub] and subtracts the GLOBAL mean (not the
            # per-group mean) via predicate_preserving_norm_star. Concrete
            # GroupNorm normalises each channel-group independently. As with
            # LayerNorm this is masked from being directly exploitable only
            # by var_lb=0 (-> s_ub=1/sqrt(eps) huge slack); any var_lb
            # tightening turns it actively unsound.
            # Per-group reach lands in Commit 7 (T1-1). Until then, fail
            # loud whenever the input has more than one channel-group; the
            # Box path is sound and remains the workaround.
            if num_groups > 1:
                raise NotImplementedError(
                    f"GroupNorm Star reach is latently unsound for "
                    f"multi-group inputs (num_groups={num_groups} > 1) -- "
                    f"the per-group structure is collapsed to a single "
                    f"sigma interval and global mean, masked only by "
                    f"var_lb=0. Per-group reach lands in Commit 7 "
                    f"(PR12_FIX_LIST T1-1). Use groupnorm_box (sound) or "
                    f"split into num_groups=1 batches."
                )
            # Bound each group's scale (1/sigma) interval using IBP, then
            # build a Star that preserves the input predicates per group.
            lb, ub = base.estimate_ranges()
            spatial = base.dim // num_channels
            lb_c = lb.reshape(num_channels, spatial)
            ub_c = ub.reshape(num_channels, spatial)
            # Worst-case sigma across all groups (conservative single bound).
            sig_lb_list, sig_ub_list = [], []
            for g in range(num_groups):
                start = g * channels_per_group
                end = start + channels_per_group
                g_lb = lb_c[start:end].reshape(-1)
                g_ub = ub_c[start:end].reshape(-1)
                _, _, var_lb, var_ub = interval_mean_var(g_lb, g_ub)
                sig_lb_list.append(float(np.sqrt(np.asarray(var_lb).item() + eps)))
                sig_ub_list.append(float(np.sqrt(np.asarray(var_ub).item() + eps)))
            sigma_lb = min(sig_lb_list)
            sigma_ub = max(sig_ub_list)
            weight_b = np.repeat(weight, spatial) if weight is not None else None
            bias_b = np.repeat(bias, spatial) if bias is not None else None
            new_star = predicate_preserving_norm_star(
                base,
                sigma_bounds=(sigma_lb, sigma_ub),
                weight=weight_b,
                bias=bias_b,
                subtract_mean=True,
            )
        if is_image:
            new_star = new_star.to_image_star(s.height, s.width, s.num_channels)
        output.append(new_star)
    return output
