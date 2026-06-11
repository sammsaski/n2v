"""User-facing layer wrappers for ops not present in ``torch.nn``.

Many transformer-era components ported from nnVLA are not stock
``torch.nn.Module`` types (RMSNorm, RoPE, CausalMask, GRN, LayerScale,
DropPath, CLSToken, OpenMax, ...). This package provides thin
``nn.Module`` wrappers so users can build models in PyTorch using the
same names nnVLA uses, and the n2v dispatcher detects them via
``isinstance(layer, ...)``.

Importing any wrapper does **not** affect dispatch — registration is done
by ``n2v.nn.layer_ops.dispatcher`` via ``isinstance`` chains.
"""

from n2v.nn.layers.rms_norm import RMSNorm
from n2v.nn.layers.layer_norm_wrap import LayerNormWrap
from n2v.nn.layers.grn import GRN
from n2v.nn.layers.layer_scale import LayerScale
from n2v.nn.layers.drop_path import DropPath
from n2v.nn.layers.softmax_attention import SoftmaxAttention
from n2v.nn.layers.causal_mask import CausalMask
from n2v.nn.layers.rope import RoPE
from n2v.nn.layers.cls_token import CLSToken
from n2v.nn.layers.distillation_token import DistillationToken
from n2v.nn.layers.tied_linear import TiedLinear
from n2v.nn.layers.openmax import OpenMax
from n2v.nn.layers.action_head import ActionHead
from n2v.nn.layers.action_tokenizer import ActionTokenizer
from n2v.nn.layers.patch_embed import PatchEmbed
from n2v.nn.layers.overlap_patch_embed import OverlapPatchEmbed
from n2v.nn.layers.parallel_residual import ParallelResidual
from n2v.nn.layers.mix_ffn import MixFFN
from n2v.nn.layers.pooler import Pooler
from n2v.nn.layers.projection_head import ProjectionHead
from n2v.nn.layers.selective_feature_fusion import SelectiveFeatureFusion
from n2v.nn.layers.segment_embedding import SegmentEmbedding
from n2v.nn.layers.positional_encoding import PositionalEncoding
from n2v.nn.layers.relative_attention_bias_t5 import RelativeAttentionBiasT5
from n2v.nn.layers.relative_position_bias_table import RelativePositionBiasTable
from n2v.nn.layers.dag_add import DagAdd
from n2v.nn.layers.dag_concat import DagConcat
from n2v.nn.layers.concat2d import Concat2D
from n2v.nn.layers.add_with_frozen_skip import AddWithFrozenSkip
from n2v.nn.layers.concat_with_frozen_skip import ConcatWithFrozenSkip

__all__ = [
    "RMSNorm",
    "LayerNormWrap",
    "GRN",
    "LayerScale",
    "DropPath",
    "SoftmaxAttention",
    "CausalMask",
    "RoPE",
    "CLSToken",
    "DistillationToken",
    "TiedLinear",
    "OpenMax",
    "ActionHead",
    "ActionTokenizer",
    "PatchEmbed",
    "OverlapPatchEmbed",
    "ParallelResidual",
    "MixFFN",
    "Pooler",
    "ProjectionHead",
    "SelectiveFeatureFusion",
    "SegmentEmbedding",
    "PositionalEncoding",
    "RelativeAttentionBiasT5",
    "RelativePositionBiasTable",
    "DagAdd",
    "DagConcat",
    "Concat2D",
    "AddWithFrozenSkip",
    "ConcatWithFrozenSkip",
]
