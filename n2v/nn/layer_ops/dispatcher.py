"""
Layer reachability dispatcher - routes computation based on layer type and set type.

Dispatches reachability computation for a single layer based on PyTorch layer type
and input set type, without requiring custom layer wrapper classes.
"""

import warnings
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union

# Import set types
from n2v.sets import Star, Zono, Box, Hexatope, Octatope
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

# Registry consulted as a fallback when no isinstance branch matches.
# New layers can self-register via @n2v.nn.layer_ops.register(Layer, SetCls).
from n2v.nn.layer_ops.registry import lookup as _registry_lookup

# Import layer-specific reach functions
from . import linear_reach, relu_reach, conv2d_reach, flatten_reach
from . import maxpool2d_reach, avgpool2d_reach, global_avgpool_reach
from . import batchnorm_reach
from . import pad_reach
from .pad_reach import _PAD_TYPES
from . import reduce_reach
from . import leakyrelu_reach
from . import sigmoid_reach
from . import tanh_reach
from . import conv1d_reach
from . import upsample_reach
from . import sign_reach

# Phase 1 ports: activations and normalisations from nnVLA
from . import relu6_reach
from . import elu_reach
from . import gelu_reach
from . import quickgelu_reach
from . import silu_reach
from . import hardswish_reach
from . import layernorm_reach
from . import rmsnorm_reach
from . import groupnorm_reach
from . import grn_reach

# Wrapper modules detected via isinstance (introduced in Phase 1.5)
from n2v.nn.layers.rms_norm import RMSNorm as _RMSNorm
from n2v.nn.layers.grn import GRN as _GRN

# Phase 2 ports: MLP / skip / DAG layers
from . import layerscale_reach
from . import drop_path_reach
from . import add_with_frozen_skip_reach
from . import concat_with_frozen_skip_reach
from . import dag_add_reach
from . import dag_concat_reach
from . import concat2d_reach
from . import selective_feature_fusion_reach
from . import mix_ffn_reach

from n2v.nn.layers.layer_scale import LayerScale as _LayerScale
from n2v.nn.layers.drop_path import DropPath as _DropPath
from n2v.nn.layers.add_with_frozen_skip import AddWithFrozenSkip as _AddWithFrozenSkip
from n2v.nn.layers.concat_with_frozen_skip import ConcatWithFrozenSkip as _ConcatWithFrozenSkip
from n2v.nn.layers.dag_add import DagAdd as _DagAdd
from n2v.nn.layers.dag_concat import DagConcat as _DagConcat
from n2v.nn.layers.concat2d import Concat2D as _Concat2D
from n2v.nn.layers.selective_feature_fusion import SelectiveFeatureFusion as _SFF
from n2v.nn.layers.mix_ffn import MixFFN as _MixFFN

# Phase 3 ports: attention layers
from . import softmax_attention_reach
from . import causal_mask_reach
from . import sparsemax_reach
from . import relative_attention_bias_t5_reach
from . import relative_position_bias_table_reach
from . import linear_attention_reach
from . import efficient_attention_sr_reach
from . import sparse_attention_reach
from . import cross_attention_reach
from . import grouped_query_attention_reach
from . import multi_query_attention_reach

from n2v.nn.layers.softmax_attention import SoftmaxAttention as _SoftmaxAttention
from n2v.nn.layers.causal_mask import CausalMask as _CausalMask
from n2v.nn.layers.relative_attention_bias_t5 import RelativeAttentionBiasT5 as _RelAttnBiasT5
from n2v.nn.layers.relative_position_bias_table import RelativePositionBiasTable as _RelPosBiasTable

# Phase 4 ports: embeddings & tokens
from . import embedding_reach
from . import positional_encoding_reach
from . import rope_reach
from . import cls_token_reach
from . import distillation_token_reach
from . import segment_embedding_reach

from n2v.nn.layers.cls_token import CLSToken as _CLSToken
from n2v.nn.layers.distillation_token import DistillationToken as _DistillationToken
from n2v.nn.layers.positional_encoding import PositionalEncoding as _PositionalEncoding
from n2v.nn.layers.rope import RoPE as _RoPE
# T1-3 (audit high): wrapper for SegmentEmbedding. The reach helpers in
# segment_embedding_reach existed pre-PR12 but the wrapper class was never
# imported into dispatcher.py, so any model containing a SegmentEmbedding
# raised NotImplementedError end-to-end. Now wired via isinstance branches
# in all five _reach_layer_* methods below. segment_ids is read from
# **kwargs (forwarded by the fx call_module / NeuralNetwork.reach path).
from n2v.nn.layers.segment_embedding import SegmentEmbedding as _SegmentEmbedding
# T1-7 (ViT enable): PatchEmbed routed as fx leaf via N2VTracer; needs a
# dedicated dispatcher branch since it composes Conv2d + flatten + transpose.
from n2v.nn.layers.patch_embed import PatchEmbed as _PatchEmbed
from . import patch_embed_reach
# PR-1 audit I5: OverlapPatchEmbed was an fx leaf without a dispatcher
# branch -- every set type fell through to NotImplementedError. Now wired.
from n2v.nn.layers.overlap_patch_embed import OverlapPatchEmbed as _OverlapPatchEmbed
from . import overlap_patch_embed_reach

# Phase 5 ports: conv variants & specialty
from . import tied_linear_reach
from . import conv2d_transpose_reach
from . import depthwise_conv_reach
from . import action_head_reach
from . import action_tokenizer_reach
from . import openmax_reach
from . import pooler_reach
from . import projection_head_reach
from . import conv_token_embedding_reach

from n2v.nn.layers.tied_linear import TiedLinear as _TiedLinear
from n2v.nn.layers.action_head import ActionHead as _ActionHead
from n2v.nn.layers.action_tokenizer import ActionTokenizer as _ActionTokenizer
from n2v.nn.layers.openmax import OpenMax as _OpenMax
from n2v.nn.layers.pooler import Pooler as _Pooler
from n2v.nn.layers.projection_head import ProjectionHead as _ProjectionHead

# ONNX types (onnx2torch is a required dependency)
from onnx2torch.node_converters.global_average_pool import (
    OnnxGlobalAveragePool,
    OnnxGlobalAveragePoolWithKnownInputShape,
)
from onnx2torch.node_converters.reduce import OnnxReduceStaticAxes, OnnxReduceSumStaticAxes
from onnx2torch.node_converters.resize import OnnxResize
from onnx2torch.node_converters.neg import OnnxNeg
from onnx2torch.node_converters.cast import OnnxCast
from onnx2torch.node_converters.functions import OnnxFunction
from onnx2torch.node_converters.transpose import OnnxTranspose
from onnx2torch.node_converters.flatten import OnnxFlatten

_ONNX_GAP_TYPES = (nn.AdaptiveAvgPool2d, OnnxGlobalAveragePool, OnnxGlobalAveragePoolWithKnownInputShape)
_ONNX_REDUCE_TYPES = (OnnxReduceStaticAxes, OnnxReduceSumStaticAxes)
_ONNX_RESIZE_TYPES = (nn.Upsample, OnnxResize)
_ONNX_NEG_TYPES = (OnnxNeg,)
_ONNX_CAST_TYPES = (OnnxCast,)
_ONNX_FUNCTION_TYPES = (OnnxFunction,)
_ONNX_TRANSPOSE_TYPES = (OnnxTranspose,)
_ONNX_FLATTEN_TYPES = (nn.Flatten, OnnxFlatten)


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
        return linear_reach.linear_star(
            layer, input_sets,
            expected_n_tokens=kwargs.get("n_tokens"),
        )

    elif isinstance(layer, nn.ReLU):
        lp_solver = kwargs.get('lp_solver', 'default')
        verbose = kwargs.get('verbose', False)
        parallel = kwargs.get('parallel', None)
        n_workers = kwargs.get('n_workers', None)
        precomputed_bounds = kwargs.get('precomputed_bounds', None)

        if method == 'exact':
            return relu_reach.relu_star_exact(
                input_sets, lp_solver=lp_solver, verbose=verbose,
                parallel=parallel, n_workers=n_workers,
                precomputed_bounds=precomputed_bounds,
            )
        else:  # approx
            relax_factor = kwargs.get('relax_factor', 0.5)
            relax_method = kwargs.get('relax_method', 'standard')
            return relu_reach.relu_star_approx(
                input_sets, relax_factor, lp_solver, relax_method,
                precomputed_bounds=precomputed_bounds,
            )

    elif isinstance(layer, nn.LeakyReLU):
        gamma = layer.negative_slope
        lp_solver = kwargs.get('lp_solver', 'default')
        verbose = kwargs.get('verbose', False)
        precomputed_bounds = kwargs.get('precomputed_bounds', None)
        if method == 'exact':
            return leakyrelu_reach.leakyrelu_star_exact(
                input_sets, gamma=gamma, lp_solver=lp_solver, verbose=verbose,
                precomputed_bounds=precomputed_bounds,
            )
        else:
            return leakyrelu_reach.leakyrelu_star_approx(
                input_sets, gamma=gamma, lp_solver=lp_solver,
                precomputed_bounds=precomputed_bounds,
            )

    elif isinstance(layer, nn.Sigmoid):
        lp_solver = kwargs.get('lp_solver', 'default')
        if method == 'exact':
            warnings.warn("Sigmoid does not support exact method; using approx.")
        return sigmoid_reach.sigmoid_star_approx(input_sets, lp_solver=lp_solver)

    elif isinstance(layer, nn.Tanh):
        lp_solver = kwargs.get('lp_solver', 'default')
        if method == 'exact':
            warnings.warn("Tanh does not support exact method; using approx.")
        return tanh_reach.tanh_star_approx(input_sets, lp_solver=lp_solver)

    elif isinstance(layer, nn.Conv2d):
        return conv2d_reach.conv2d_star(layer, input_sets, method=method, **kwargs)

    elif isinstance(layer, nn.Conv1d):
        return conv1d_reach.conv1d_star(layer, input_sets, **kwargs)

    elif isinstance(layer, nn.MaxPool2d):
        lp_solver = kwargs.get('lp_solver', 'default')
        verbose = kwargs.get('verbose', False)
        return maxpool2d_reach.maxpool2d_star(
            layer, input_sets, method=method, lp_solver=lp_solver,
            verbose=verbose, **kwargs
        )

    elif isinstance(layer, nn.AvgPool2d):
        return avgpool2d_reach.avgpool2d_star(layer, input_sets, **kwargs)

    elif isinstance(layer, _ONNX_GAP_TYPES):
        return global_avgpool_reach.global_avgpool_star(input_sets)

    elif isinstance(layer, _ONNX_FLATTEN_TYPES):
        return flatten_reach.flatten_star(layer, input_sets)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return batchnorm_reach.batchnorm_star(layer, input_sets)

    elif isinstance(layer, _PAD_TYPES):
        return pad_reach.pad_star(layer, input_sets)

    elif isinstance(layer, _ONNX_REDUCE_TYPES):
        return reduce_reach.reduce_star(layer, input_sets)

    elif isinstance(layer, _ONNX_RESIZE_TYPES):
        return upsample_reach.upsample_star(layer, input_sets, **kwargs)

    elif isinstance(layer, _ONNX_NEG_TYPES):
        return _neg_sets_star(input_sets)

    elif isinstance(layer, _ONNX_CAST_TYPES):
        return input_sets

    elif isinstance(layer, _ONNX_TRANSPOSE_TYPES):
        return _transpose_sets_star(layer, input_sets)

    elif _is_sign_layer(layer):
        return sign_reach.sign_star(layer, input_sets, method, **kwargs)

    elif _resolve_onnx_function_substitute(layer) is not None:
        return _reach_layer_star(
            _resolve_onnx_function_substitute(layer), input_sets, method, **kwargs
        )

    # ----- Phase 1: activations -----
    elif isinstance(layer, nn.ReLU6):
        return relu6_reach.relu6_star_approx(input_sets)
    elif isinstance(layer, nn.ELU):
        return elu_reach.elu_star_approx(input_sets, alpha=float(layer.alpha))
    elif isinstance(layer, nn.GELU):
        # T0-3 (audit C5): branch on approximate=tanh vs erf. The previous
        # unconditional erf-form routing was unsound for nn.GELU(approximate=
        # 'tanh') -- the GPT-2 / HF default -- because the tanh dip falls
        # below the erf floor.
        mode = getattr(layer, "approximate", "none")
        if mode == "tanh":
            return gelu_reach.gelu_tanh_star_approx(input_sets)
        elif mode == "none":
            return gelu_reach.gelu_star_approx(input_sets)
        else:
            raise NotImplementedError(
                f"nn.GELU(approximate={mode!r}) is not supported by reach."
            )
    elif isinstance(layer, nn.SiLU):
        return silu_reach.silu_star_approx(input_sets)
    elif isinstance(layer, nn.Hardswish):
        return hardswish_reach.hardswish_star_approx(input_sets)

    # ----- Phase 1: normalisations -----
    elif isinstance(layer, _RMSNorm):
        return rmsnorm_reach.rmsnorm_star_approx(layer, input_sets)
    elif isinstance(layer, nn.LayerNorm):
        return layernorm_reach.layernorm_star_approx(layer, input_sets)
    elif isinstance(layer, nn.GroupNorm):
        return groupnorm_reach.groupnorm_star_approx(layer, input_sets)
    elif isinstance(layer, _GRN):
        return grn_reach.grn_star_approx(layer, input_sets)

    # ----- Phase 2: elementwise-affine MLP/skip ops -----
    elif isinstance(layer, _LayerScale):
        return layerscale_reach.layerscale_star(layer, input_sets)
    elif isinstance(layer, _DropPath):
        return drop_path_reach.drop_path_star(layer, input_sets)
    elif isinstance(layer, _AddWithFrozenSkip):
        return add_with_frozen_skip_reach.add_with_frozen_skip_star(layer, input_sets)
    elif isinstance(layer, _ConcatWithFrozenSkip):
        return concat_with_frozen_skip_reach.concat_with_frozen_skip_star(layer, input_sets)
    elif isinstance(layer, _MixFFN):
        return mix_ffn_reach.mix_ffn_passthrough(layer, input_sets, method, **kwargs)

    # ----- Phase 3: single-input attention helpers -----
    elif isinstance(layer, _CausalMask):
        return causal_mask_reach.causal_mask_star(layer, input_sets)
    elif isinstance(layer, _RelAttnBiasT5):
        return relative_attention_bias_t5_reach.relative_attention_bias_t5_star(layer, input_sets)
    elif isinstance(layer, _RelPosBiasTable):
        # T1-2 (audit high): RelPosBiasTable was previously grouped with
        # RelAttnBiasT5 and routed to the T5 helper, which reads
        # layer.relative_attention_bias.weight -- an attribute the Swin
        # table does not have (its parameter is `bias_table`). Resulting
        # in AttributeError end-to-end on every set type. The dedicated
        # relative_position_bias_table_reach module existed but was never
        # invoked. Now split.
        return relative_position_bias_table_reach.relative_position_bias_table_star(
            layer, input_sets,
        )
    elif isinstance(layer, _SoftmaxAttention):
        raise NotImplementedError(
            "SoftmaxAttention requires multi-input (Q, K, V) dispatch via n2v.nn.reach."
        )

    # ----- Phase 4: embeddings & tokens -----
    elif isinstance(layer, nn.Embedding):
        return embedding_reach.embedding_star(layer, input_sets)
    elif isinstance(layer, _PositionalEncoding):
        return positional_encoding_reach.positional_encoding_star(layer, input_sets)
    elif isinstance(layer, _RoPE):
        return rope_reach.rope_star(layer, input_sets)
    elif isinstance(layer, _CLSToken):
        return cls_token_reach.cls_token_star(layer, input_sets)
    elif isinstance(layer, _DistillationToken):
        return distillation_token_reach.distillation_token_star(layer, input_sets)
    elif isinstance(layer, _SegmentEmbedding):
        return segment_embedding_reach.segment_embedding_star(
            layer, input_sets, segment_ids=kwargs.get("segment_ids"),
        )
    elif isinstance(layer, _PatchEmbed):
        return patch_embed_reach.patch_embed_star(layer, input_sets, **kwargs)
    elif isinstance(layer, _OverlapPatchEmbed):
        return overlap_patch_embed_reach.overlap_patch_embed_star(
            layer, input_sets, **kwargs,
        )

    # ----- Phase 5: conv variants & specialty -----
    elif isinstance(layer, _TiedLinear):
        return tied_linear_reach.tied_linear_star(layer, input_sets)
    elif isinstance(layer, nn.ConvTranspose2d):
        return conv2d_transpose_reach.conv2d_transpose_star(layer, input_sets)
    elif isinstance(layer, _ActionHead):
        return action_head_reach.action_head_star(layer, input_sets)
    elif isinstance(layer, _ActionTokenizer):
        return action_tokenizer_reach.action_tokenizer_star_approx(layer, input_sets)
    elif isinstance(layer, _OpenMax):
        return openmax_reach.openmax_star_approx(layer, input_sets)
    elif isinstance(layer, _Pooler):
        return pooler_reach.pooler_passthrough(layer, input_sets, method, **kwargs)
    elif isinstance(layer, _ProjectionHead):
        return projection_head_reach.projection_head_passthrough(layer, input_sets, method, **kwargs)

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        # Recursively handle Sequential
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        handler = _registry_lookup(layer, Star)
        if handler is not None:
            return handler(layer, input_sets, method, **kwargs)
        raise NotImplementedError(
            f"Star reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_zono(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Zonotope reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_zono(
            layer, input_sets,
            expected_n_tokens=kwargs.get("n_tokens"),
        )

    elif isinstance(layer, nn.ReLU):
        return relu_reach.relu_zono_approx(input_sets)

    elif isinstance(layer, nn.LeakyReLU):
        return leakyrelu_reach.leakyrelu_zono_approx(input_sets, gamma=layer.negative_slope)

    elif isinstance(layer, nn.Sigmoid):
        return sigmoid_reach.sigmoid_zono_approx(input_sets)

    elif isinstance(layer, nn.Tanh):
        return tanh_reach.tanh_zono_approx(input_sets)

    elif isinstance(layer, nn.Conv2d):
        return conv2d_reach.conv2d_zono(layer, input_sets)

    elif isinstance(layer, nn.Conv1d):
        return conv1d_reach.conv1d_zono(layer, input_sets, **kwargs)

    elif isinstance(layer, nn.MaxPool2d):
        return maxpool2d_reach.maxpool2d_zono(layer, input_sets)

    elif isinstance(layer, nn.AvgPool2d):
        return avgpool2d_reach.avgpool2d_zono(layer, input_sets)

    elif isinstance(layer, _ONNX_GAP_TYPES):
        return global_avgpool_reach.global_avgpool_zono(input_sets)

    elif isinstance(layer, _ONNX_FLATTEN_TYPES):
        return flatten_reach.flatten_zono(layer, input_sets)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return batchnorm_reach.batchnorm_zono(layer, input_sets)

    elif isinstance(layer, _PAD_TYPES):
        return pad_reach.pad_zono(layer, input_sets)

    elif isinstance(layer, _ONNX_REDUCE_TYPES):
        return reduce_reach.reduce_zono(layer, input_sets)

    elif isinstance(layer, _ONNX_RESIZE_TYPES):
        return upsample_reach.upsample_zono(layer, input_sets, **kwargs)

    elif isinstance(layer, _ONNX_NEG_TYPES):
        return _neg_sets_zono(input_sets)

    elif isinstance(layer, _ONNX_CAST_TYPES):
        return input_sets

    elif isinstance(layer, _ONNX_TRANSPOSE_TYPES):
        return _transpose_sets_zono(layer, input_sets)

    elif _is_sign_layer(layer):
        return sign_reach.sign_zono(input_sets)

    elif _resolve_onnx_function_substitute(layer) is not None:
        return _reach_layer_zono(
            _resolve_onnx_function_substitute(layer), input_sets, method, **kwargs
        )

    # ----- Phase 2: elementwise-affine MLP/skip ops -----
    elif isinstance(layer, _LayerScale):
        return layerscale_reach.layerscale_zono(layer, input_sets)
    elif isinstance(layer, _DropPath):
        return drop_path_reach.drop_path_zono(layer, input_sets)
    elif isinstance(layer, _AddWithFrozenSkip):
        return add_with_frozen_skip_reach.add_with_frozen_skip_zono(layer, input_sets)
    elif isinstance(layer, _ConcatWithFrozenSkip):
        return concat_with_frozen_skip_reach.concat_with_frozen_skip_zono(layer, input_sets)

    # ----- Phase 3: single-input attention helpers (zono coverage limited) -----
    elif isinstance(layer, _CausalMask):
        return causal_mask_reach.causal_mask_zono(layer, input_sets)
    elif isinstance(layer, _RelAttnBiasT5):
        return relative_attention_bias_t5_reach.relative_attention_bias_t5_zono(layer, input_sets)
    elif isinstance(layer, _RelPosBiasTable):
        return relative_position_bias_table_reach.relative_position_bias_table_zono(
            layer, input_sets,
        )

    # ----- Phase 4: embeddings & tokens -----
    elif isinstance(layer, nn.Embedding):
        return embedding_reach.embedding_zono(layer, input_sets)
    elif isinstance(layer, _PositionalEncoding):
        return positional_encoding_reach.positional_encoding_zono(layer, input_sets)
    elif isinstance(layer, _RoPE):
        return rope_reach.rope_zono(layer, input_sets)
    elif isinstance(layer, _CLSToken):
        return cls_token_reach.cls_token_zono(layer, input_sets)
    elif isinstance(layer, _DistillationToken):
        return distillation_token_reach.distillation_token_zono(layer, input_sets)
    elif isinstance(layer, _SegmentEmbedding):
        return segment_embedding_reach.segment_embedding_zono(
            layer, input_sets, segment_ids=kwargs.get("segment_ids"),
        )
    elif isinstance(layer, _PatchEmbed):
        return patch_embed_reach.patch_embed_zono(layer, input_sets)
    elif isinstance(layer, _OverlapPatchEmbed):
        return overlap_patch_embed_reach.overlap_patch_embed_zono(
            layer, input_sets,
        )
    # PR-1 audit C3 follow-up: MixFFN Zono route. The helper handles all
    # five set types; only Star/Box were wired at the dispatcher when C3
    # first landed.
    elif isinstance(layer, _MixFFN):
        return mix_ffn_reach.mix_ffn_passthrough(layer, input_sets, method, **kwargs)

    # ----- Phase 1 Zono routes (box-lifted, sound but loose; ViT enable) -----
    elif isinstance(layer, nn.LayerNorm):
        return layernorm_reach.layernorm_zono(layer, input_sets)
    elif isinstance(layer, nn.GELU):
        mode = getattr(layer, "approximate", "none")
        if mode == "tanh":
            return gelu_reach.gelu_tanh_zono(input_sets)
        elif mode == "none":
            return gelu_reach.gelu_zono(input_sets)
        else:
            raise NotImplementedError(
                f"nn.GELU(approximate={mode!r}) is not supported by reach."
            )

    # ----- Phase 5: conv variants & specialty -----
    elif isinstance(layer, _TiedLinear):
        return tied_linear_reach.tied_linear_zono(layer, input_sets)
    elif isinstance(layer, nn.ConvTranspose2d):
        return conv2d_transpose_reach.conv2d_transpose_zono(layer, input_sets)
    elif isinstance(layer, _ActionHead):
        return action_head_reach.action_head_zono(layer, input_sets)
    elif isinstance(layer, _Pooler):
        return pooler_reach.pooler_passthrough(layer, input_sets, method, **kwargs)
    elif isinstance(layer, _ProjectionHead):
        return projection_head_reach.projection_head_passthrough(layer, input_sets, method, **kwargs)

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        handler = _registry_lookup(layer, Zono)
        if handler is not None:
            return handler(layer, input_sets, method, **kwargs)
        raise NotImplementedError(
            f"Zono reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_box(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Box reachability through a layer."""

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_box(
            layer, input_sets,
            expected_n_tokens=kwargs.get("n_tokens"),
        )

    elif isinstance(layer, nn.ReLU):
        return relu_reach.relu_box(input_sets)

    elif isinstance(layer, nn.LeakyReLU):
        return leakyrelu_reach.leakyrelu_box(input_sets, gamma=layer.negative_slope)

    elif isinstance(layer, nn.Sigmoid):
        return sigmoid_reach.sigmoid_box(input_sets)

    elif isinstance(layer, nn.Tanh):
        return tanh_reach.tanh_box(input_sets)

    elif isinstance(layer, nn.Conv1d):
        return conv1d_reach.conv1d_box(layer, input_sets, **kwargs)

    elif isinstance(layer, _ONNX_FLATTEN_TYPES):
        return flatten_reach.flatten_box(layer, input_sets)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return batchnorm_reach.batchnorm_box(layer, input_sets)

    elif isinstance(layer, _ONNX_REDUCE_TYPES):
        return reduce_reach.reduce_box(layer, input_sets)

    elif isinstance(layer, _ONNX_NEG_TYPES):
        return _neg_sets_box(input_sets)

    elif isinstance(layer, _ONNX_CAST_TYPES):
        return input_sets

    elif isinstance(layer, _ONNX_TRANSPOSE_TYPES):
        return _transpose_sets_box(layer, input_sets)

    elif _is_sign_layer(layer):
        return sign_reach.sign_box(input_sets)

    elif _resolve_onnx_function_substitute(layer) is not None:
        return _reach_layer_box(
            _resolve_onnx_function_substitute(layer), input_sets, method, **kwargs
        )

    # ----- Phase 1: activations -----
    elif isinstance(layer, nn.ReLU6):
        return relu6_reach.relu6_box(input_sets)
    elif isinstance(layer, nn.ELU):
        return elu_reach.elu_box(input_sets, alpha=float(layer.alpha))
    elif isinstance(layer, nn.GELU):
        # T0-3 (audit C5): branch on approximate=tanh vs erf.
        mode = getattr(layer, "approximate", "none")
        if mode == "tanh":
            return gelu_reach.gelu_tanh_box(input_sets)
        elif mode == "none":
            return gelu_reach.gelu_box(input_sets)
        else:
            raise NotImplementedError(
                f"nn.GELU(approximate={mode!r}) is not supported by reach."
            )
    elif isinstance(layer, nn.SiLU):
        return silu_reach.silu_box(input_sets)
    elif isinstance(layer, nn.Hardswish):
        return hardswish_reach.hardswish_box(input_sets)

    # ----- Phase 1: normalisations -----
    elif isinstance(layer, _RMSNorm):
        return rmsnorm_reach.rmsnorm_box(layer, input_sets)
    elif isinstance(layer, nn.LayerNorm):
        return layernorm_reach.layernorm_box(layer, input_sets)
    elif isinstance(layer, nn.GroupNorm):
        return groupnorm_reach.groupnorm_box(layer, input_sets)
    elif isinstance(layer, _GRN):
        return grn_reach.grn_box(layer, input_sets)

    # ----- Phase 2: skip / MLP / DAG -----
    elif isinstance(layer, _LayerScale):
        return layerscale_reach.layerscale_box(layer, input_sets)
    elif isinstance(layer, _DropPath):
        return drop_path_reach.drop_path_box(layer, input_sets)
    elif isinstance(layer, _AddWithFrozenSkip):
        return add_with_frozen_skip_reach.add_with_frozen_skip_box(layer, input_sets)
    elif isinstance(layer, _ConcatWithFrozenSkip):
        return concat_with_frozen_skip_reach.concat_with_frozen_skip_box(layer, input_sets)
    elif isinstance(layer, (_DagAdd, _DagConcat, _Concat2D, _SFF)):
        # Multi-input ops: dispatcher's single-input path can't satisfy
        # them. The graph-level traversal in n2v.nn.reach handles their
        # multi-port inputs by calling the per-op helpers in
        # dag_add_reach / dag_concat_reach / concat2d_reach /
        # selective_feature_fusion_reach with the second-port stream in
        # ``extras``. Raising here keeps single-input dispatch sound.
        raise NotImplementedError(
            f"{type(layer).__name__} requires multi-input dispatch via n2v.nn.reach."
        )
    elif isinstance(layer, _MixFFN):
        return mix_ffn_reach.mix_ffn_passthrough(layer, input_sets, method, **kwargs)

    # ----- Phase 3: single-input attention helpers -----
    elif isinstance(layer, _CausalMask):
        return causal_mask_reach.causal_mask_box(layer, input_sets)
    elif isinstance(layer, _RelAttnBiasT5):
        return relative_attention_bias_t5_reach.relative_attention_bias_t5_box(layer, input_sets)
    elif isinstance(layer, _RelPosBiasTable):
        return relative_position_bias_table_reach.relative_position_bias_table_box(
            layer, input_sets,
        )
    elif isinstance(layer, _SoftmaxAttention):
        raise NotImplementedError(
            "SoftmaxAttention requires multi-input (Q, K, V) dispatch via n2v.nn.reach."
        )

    # ----- Phase 4: embeddings & tokens -----
    elif isinstance(layer, nn.Embedding):
        return embedding_reach.embedding_box(layer, input_sets)
    elif isinstance(layer, _PositionalEncoding):
        return positional_encoding_reach.positional_encoding_box(layer, input_sets)
    elif isinstance(layer, _RoPE):
        return rope_reach.rope_box(layer, input_sets)
    elif isinstance(layer, _CLSToken):
        return cls_token_reach.cls_token_box(layer, input_sets)
    elif isinstance(layer, _DistillationToken):
        return distillation_token_reach.distillation_token_box(layer, input_sets)
    elif isinstance(layer, _SegmentEmbedding):
        return segment_embedding_reach.segment_embedding_box(
            layer, input_sets, segment_ids=kwargs.get("segment_ids"),
        )
    elif isinstance(layer, _PatchEmbed):
        return patch_embed_reach.patch_embed_box(
            layer, input_sets,
            image_shape=kwargs.get("image_shape"),
            image_layout=kwargs.get("image_layout", "HWC"),
        )
    elif isinstance(layer, _OverlapPatchEmbed):
        return overlap_patch_embed_reach.overlap_patch_embed_box(
            layer, input_sets,
            image_shape=kwargs.get("image_shape"),
            image_layout=kwargs.get("image_layout", "HWC"),
        )

    # ----- Phase 5: conv variants & specialty -----
    elif isinstance(layer, _TiedLinear):
        return tied_linear_reach.tied_linear_box(layer, input_sets)
    elif isinstance(layer, nn.ConvTranspose2d):
        return conv2d_transpose_reach.conv2d_transpose_box(layer, input_sets)
    elif isinstance(layer, _ActionHead):
        return action_head_reach.action_head_box(layer, input_sets)
    elif isinstance(layer, _ActionTokenizer):
        return action_tokenizer_reach.action_tokenizer_box(layer, input_sets)
    elif isinstance(layer, _OpenMax):
        return openmax_reach.openmax_box(layer, input_sets)
    elif isinstance(layer, _Pooler):
        return pooler_reach.pooler_passthrough(layer, input_sets, method, **kwargs)
    elif isinstance(layer, _ProjectionHead):
        return projection_head_reach.projection_head_passthrough(layer, input_sets, method, **kwargs)

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        handler = _registry_lookup(layer, Box)
        if handler is not None:
            return handler(layer, input_sets, method, **kwargs)
        raise NotImplementedError(
            f"Box reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_hexatope(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Hexatope reachability through a layer."""

    if _resolve_onnx_function_substitute(layer) is not None:
        return _reach_layer_hexatope(
            _resolve_onnx_function_substitute(layer), input_sets, method, **kwargs
        )

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_hexatope(
            layer, input_sets,
            expected_n_tokens=kwargs.get("n_tokens"),
        )

    elif isinstance(layer, nn.ReLU):
        verbose = kwargs.get('verbose', False)
        solver = kwargs.get('solver', None)
        return relu_reach.relu_hexatope_approx(input_sets, verbose=verbose, solver=solver)

    elif isinstance(layer, _ONNX_FLATTEN_TYPES):
        return flatten_reach.flatten_hexatope(layer, input_sets)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        scale, shift = batchnorm_reach._get_bn_params(layer)
        dummy = nn.Linear(len(scale), len(scale), bias=True)
        with torch.no_grad():
            dummy.weight.copy_(torch.from_numpy(np.diag(scale)).float())
            dummy.bias.copy_(torch.from_numpy(shift).float())
        return linear_reach.linear_hexatope(dummy, input_sets)

    elif isinstance(layer, _ONNX_NEG_TYPES):
        return _neg_sets_hexatope(input_sets)

    elif isinstance(layer, _ONNX_CAST_TYPES):
        return input_sets

    elif isinstance(layer, _TiedLinear):
        return tied_linear_reach.tied_linear_hexatope(layer, input_sets)
    elif isinstance(layer, _ActionHead):
        return action_head_reach.action_head_hexatope(layer, input_sets)

    # Elementwise-affine wrappers — route via linear_reach surrogates.
    elif isinstance(layer, _LayerScale):
        return layerscale_reach.layerscale_hexatope(layer, input_sets)
    elif isinstance(layer, _AddWithFrozenSkip):
        return add_with_frozen_skip_reach.add_with_frozen_skip_hexatope(layer, input_sets)
    elif isinstance(layer, _RoPE):
        return rope_reach.rope_hexatope(layer, input_sets)
    elif isinstance(layer, _PositionalEncoding):
        return positional_encoding_reach.positional_encoding_hexatope(layer, input_sets)
    elif isinstance(layer, _SegmentEmbedding):
        return segment_embedding_reach.segment_embedding_hexatope(
            layer, input_sets, segment_ids=kwargs.get("segment_ids"),
        )
    elif isinstance(layer, _CausalMask):
        return causal_mask_reach.causal_mask_hexatope(layer, input_sets)
    elif isinstance(layer, nn.ConvTranspose2d):
        return conv2d_transpose_reach.conv2d_transpose_hexatope(layer, input_sets)
    elif isinstance(layer, _RelAttnBiasT5):
        return relative_attention_bias_t5_reach.relative_attention_bias_t5_hexatope(
            layer, input_sets
        )
    elif isinstance(layer, _RelPosBiasTable):
        return relative_position_bias_table_reach.relative_position_bias_table_hexatope(
            layer, input_sets,
        )

    # ----- Phase 1 Hexatope routes (box-lifted; ViT enable) -----
    elif isinstance(layer, nn.LayerNorm):
        return layernorm_reach.layernorm_hexatope(layer, input_sets)
    elif isinstance(layer, nn.GELU):
        mode = getattr(layer, "approximate", "none")
        if mode == "tanh":
            return gelu_reach.gelu_tanh_hexatope(input_sets)
        elif mode == "none":
            return gelu_reach.gelu_hexatope(input_sets)
        else:
            raise NotImplementedError(
                f"nn.GELU(approximate={mode!r}) is not supported by reach."
            )
    elif isinstance(layer, _PatchEmbed):
        return patch_embed_reach.patch_embed_hexatope(
            layer, input_sets,
            image_shape=kwargs.get("image_shape"),
            image_layout=kwargs.get("image_layout", "HWC"),
        )
    elif isinstance(layer, _OverlapPatchEmbed):
        return overlap_patch_embed_reach.overlap_patch_embed_hexatope(
            layer, input_sets,
            image_shape=kwargs.get("image_shape"),
            image_layout=kwargs.get("image_layout", "HWC"),
        )
    # PR-1 audit C3 follow-up: MixFFN Hexatope route (box-lifted inside
    # the helper).
    elif isinstance(layer, _MixFFN):
        return mix_ffn_reach.mix_ffn_passthrough(layer, input_sets, method, **kwargs)

    # PR-1 audit I7: CLSToken and ConcatWithFrozenSkip are fx leaves via
    # N2VTracer but previously had no Hex/Oct branches -- any end-to-end
    # ViT with these wrappers raised through ``_registry_lookup``.
    elif isinstance(layer, _CLSToken):
        return cls_token_reach.cls_token_hexatope(layer, input_sets)
    elif isinstance(layer, _ConcatWithFrozenSkip):
        return concat_with_frozen_skip_reach.concat_with_frozen_skip_hexatope(
            layer, input_sets,
        )

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        handler = _registry_lookup(layer, Hexatope)
        if handler is not None:
            return handler(layer, input_sets, method, **kwargs)
        raise NotImplementedError(
            f"Hexatope reachability not implemented for layer type: {type(layer).__name__}"
        )


def _reach_layer_octatope(layer: nn.Module, input_sets: List, method: str, **kwargs) -> List:
    """Octatope reachability through a layer."""

    if _resolve_onnx_function_substitute(layer) is not None:
        return _reach_layer_octatope(
            _resolve_onnx_function_substitute(layer), input_sets, method, **kwargs
        )

    if isinstance(layer, nn.Linear):
        return linear_reach.linear_octatope(
            layer, input_sets,
            expected_n_tokens=kwargs.get("n_tokens"),
        )

    elif isinstance(layer, nn.ReLU):
        verbose = kwargs.get('verbose', False)
        solver = kwargs.get('solver', None)
        return relu_reach.relu_octatope_approx(input_sets, verbose=verbose, solver=solver)

    elif isinstance(layer, _ONNX_FLATTEN_TYPES):
        return flatten_reach.flatten_octatope(layer, input_sets)

    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        scale, shift = batchnorm_reach._get_bn_params(layer)
        dummy = nn.Linear(len(scale), len(scale), bias=True)
        with torch.no_grad():
            dummy.weight.copy_(torch.from_numpy(np.diag(scale)).float())
            dummy.bias.copy_(torch.from_numpy(shift).float())
        return linear_reach.linear_octatope(dummy, input_sets)

    elif isinstance(layer, _ONNX_NEG_TYPES):
        return _neg_sets_octatope(input_sets)

    elif isinstance(layer, _ONNX_CAST_TYPES):
        return input_sets

    elif isinstance(layer, _TiedLinear):
        return tied_linear_reach.tied_linear_octatope(layer, input_sets)
    elif isinstance(layer, _ActionHead):
        return action_head_reach.action_head_octatope(layer, input_sets)

    # Elementwise-affine wrappers — route via linear_reach surrogates.
    elif isinstance(layer, _LayerScale):
        return layerscale_reach.layerscale_octatope(layer, input_sets)
    elif isinstance(layer, _AddWithFrozenSkip):
        return add_with_frozen_skip_reach.add_with_frozen_skip_octatope(layer, input_sets)
    elif isinstance(layer, _RoPE):
        return rope_reach.rope_octatope(layer, input_sets)
    elif isinstance(layer, _PositionalEncoding):
        return positional_encoding_reach.positional_encoding_octatope(layer, input_sets)
    elif isinstance(layer, _SegmentEmbedding):
        return segment_embedding_reach.segment_embedding_octatope(
            layer, input_sets, segment_ids=kwargs.get("segment_ids"),
        )
    elif isinstance(layer, _CausalMask):
        return causal_mask_reach.causal_mask_octatope(layer, input_sets)
    elif isinstance(layer, nn.ConvTranspose2d):
        return conv2d_transpose_reach.conv2d_transpose_octatope(layer, input_sets)
    elif isinstance(layer, _RelAttnBiasT5):
        return relative_attention_bias_t5_reach.relative_attention_bias_t5_octatope(
            layer, input_sets
        )
    elif isinstance(layer, _RelPosBiasTable):
        return relative_position_bias_table_reach.relative_position_bias_table_octatope(
            layer, input_sets,
        )

    # ----- Phase 1 Octatope routes (box-lifted; ViT enable) -----
    elif isinstance(layer, nn.LayerNorm):
        return layernorm_reach.layernorm_octatope(layer, input_sets)
    elif isinstance(layer, nn.GELU):
        mode = getattr(layer, "approximate", "none")
        if mode == "tanh":
            return gelu_reach.gelu_tanh_octatope(input_sets)
        elif mode == "none":
            return gelu_reach.gelu_octatope(input_sets)
        else:
            raise NotImplementedError(
                f"nn.GELU(approximate={mode!r}) is not supported by reach."
            )
    elif isinstance(layer, _PatchEmbed):
        return patch_embed_reach.patch_embed_octatope(
            layer, input_sets,
            image_shape=kwargs.get("image_shape"),
            image_layout=kwargs.get("image_layout", "HWC"),
        )
    elif isinstance(layer, _OverlapPatchEmbed):
        return overlap_patch_embed_reach.overlap_patch_embed_octatope(
            layer, input_sets,
            image_shape=kwargs.get("image_shape"),
            image_layout=kwargs.get("image_layout", "HWC"),
        )
    # PR-1 audit C3 follow-up: MixFFN Octatope route (box-lifted inside
    # the helper).
    elif isinstance(layer, _MixFFN):
        return mix_ffn_reach.mix_ffn_passthrough(layer, input_sets, method, **kwargs)

    # PR-1 audit I7: CLSToken and ConcatWithFrozenSkip Oct branches.
    elif isinstance(layer, _CLSToken):
        return cls_token_reach.cls_token_octatope(layer, input_sets)
    elif isinstance(layer, _ConcatWithFrozenSkip):
        return concat_with_frozen_skip_reach.concat_with_frozen_skip_octatope(
            layer, input_sets,
        )

    elif isinstance(layer, (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        return input_sets

    elif isinstance(layer, nn.Sequential):
        current_sets = input_sets
        for sublayer in layer:
            current_sets = reach_layer(sublayer, current_sets, method, **kwargs)
        return current_sets

    else:
        handler = _registry_lookup(layer, Octatope)
        if handler is not None:
            return handler(layer, input_sets, method, **kwargs)
        raise NotImplementedError(
            f"Octatope reachability not implemented for layer type: {type(layer).__name__}"
        )


# ===========================================================================
# OnnxFunction helpers — detect Sign activation
# ===========================================================================

def _is_sign_layer(layer: nn.Module) -> bool:
    """Check if layer is a Sign activation (OnnxFunction wrapping torch.sign)."""
    if isinstance(layer, _ONNX_FUNCTION_TYPES):
        return getattr(layer, 'function', None) is torch.sign
    return False


def _onnx_function_target(layer: nn.Module):
    """Return the wrapped callable of an OnnxFunction, or ``None`` otherwise."""
    if isinstance(layer, _ONNX_FUNCTION_TYPES):
        return getattr(layer, 'function', None)
    return None


# Map a wrapped function (as seen on OnnxFunction.function) → torch.nn.Module
# class to substitute when dispatching reachability. The dispatcher consults
# this *before* its isinstance chains so an ONNX-loaded model behaves the same
# as an equivalent native PyTorch model.
import torch.nn.functional as F  # noqa: E402 — kept local to avoid eager top-level import
_ONNX_FUNCTION_SUBSTITUTES: dict = {
    F.gelu: nn.GELU,
    F.silu: nn.SiLU,
    F.hardswish: nn.Hardswish,
    F.relu6: nn.ReLU6,
    F.elu: nn.ELU,
}


def _resolve_onnx_function_substitute(layer: nn.Module):
    """If ``layer`` is an OnnxFunction wrapping a known activation, return a
    fresh substitute module the dispatcher can route normally; else None."""
    target = _onnx_function_target(layer)
    if target is None:
        return None
    cls = _ONNX_FUNCTION_SUBSTITUTES.get(target)
    if cls is None:
        return None
    return cls()


# ===========================================================================
# OnnxNeg helpers — negate sets (multiply by -1)
# ===========================================================================

def _neg_sets_star(input_sets: List) -> List:
    """Negate Star/ImageStar sets: multiply V matrix by -1."""
    output = []
    for s in input_sets:
        if isinstance(s, ImageStar):
            new_V = -s.V
            output.append(ImageStar(
                new_V, s.C, s.d, s.predicate_lb, s.predicate_ub,
                s.height, s.width, s.num_channels
            ))
        else:
            new_V = -s.V
            output.append(Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub))
    return output


def _neg_sets_zono(input_sets: List) -> List:
    """Negate Zono/ImageZono sets: negate center and generators."""
    output = []
    for s in input_sets:
        if isinstance(s, ImageZono):
            output.append(ImageZono(-s.c, -s.V, s.height, s.width, s.num_channels))
        else:
            output.append(Zono(-s.c, -s.V))
    return output


def _neg_sets_box(input_sets: List) -> List:
    """Negate Box sets: swap and negate bounds."""
    output = []
    for s in input_sets:
        output.append(Box(-s.ub, -s.lb))
    return output


def _neg_sets_hexatope(input_sets: List) -> List:
    """Negate Hexatope sets via affine map with -I."""
    import torch
    output = []
    for s in input_sets:
        dummy = nn.Linear(s.dim, s.dim, bias=False)
        with torch.no_grad():
            dummy.weight.copy_(torch.from_numpy(-np.eye(s.dim)).float())
        result = linear_reach.linear_hexatope(dummy, [s])
        output.extend(result)
    return output


def _neg_sets_octatope(input_sets: List) -> List:
    """Negate Octatope sets via affine map with -I."""
    import torch
    output = []
    for s in input_sets:
        dummy = nn.Linear(s.dim, s.dim, bias=False)
        with torch.no_grad():
            dummy.weight.copy_(torch.from_numpy(-np.eye(s.dim)).float())
        result = linear_reach.linear_octatope(dummy, [s])
        output.extend(result)
    return output


# ===========================================================================
# OnnxTranspose helpers — permute dimensions of sets
# ===========================================================================

def _transpose_sets_star(layer: nn.Module, input_sets: List) -> List:
    """Permute rows of Star V matrix."""
    perm = layer.perm
    output_sets = []
    for s in input_sets:
        new_V = s.V[perm, :]
        output_sets.append(Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub))
    return output_sets


def _transpose_sets_zono(layer: nn.Module, input_sets: List) -> List:
    """Permute rows of Zono center and generators."""
    perm = layer.perm
    output_sets = []
    for s in input_sets:
        new_c = s.c[perm, :]
        new_V = s.V[perm, :]
        output_sets.append(Zono(new_c, new_V))
    return output_sets


def _transpose_sets_box(layer: nn.Module, input_sets: List) -> List:
    """Permute rows of Box bounds."""
    perm = layer.perm
    output_sets = []
    for s in input_sets:
        output_sets.append(Box(s.lb[perm, :], s.ub[perm, :]))
    return output_sets
