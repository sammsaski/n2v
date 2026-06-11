"""OverlapPatchEmbed reachability — SegFormer-style overlapping patches.

``OverlapPatchEmbed.forward`` is

    x = self.proj(x)              # Conv2d with kernel > stride
    x = x.flatten(2).transpose(1, 2)
    return self.norm(x)           # LayerNorm

The Conv2d + flatten + transpose stack is identical to
:mod:`patch_embed_reach`, so we delegate to those helpers and then
apply the trailing LayerNorm.

PR-1 audit I5: this module was missing. ``N2VTracer`` leaf-treated
``OverlapPatchEmbed`` (via ``n2v.nn.layers.__all__``) but the
dispatcher had no branch -- every set-type call fell through
``_registry_lookup`` -> ``None`` -> ``NotImplementedError``. This
file closes the gap with the same box/star/zono/hex/oct shape used by
``patch_embed_reach``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn

from n2v.nn.layer_ops import patch_embed_reach, layernorm_reach


def _apply_layernorm(layer: nn.Module, sets: List, fn_name: str) -> List:
    norm: nn.LayerNorm = layer.norm  # type: ignore[attr-defined]
    fn = getattr(layernorm_reach, fn_name)
    return fn(norm, sets)


def overlap_patch_embed_zono(layer: nn.Module, input_sets: List) -> List:
    """Sound Zono reach for OverlapPatchEmbed."""
    out = patch_embed_reach.patch_embed_zono(layer, input_sets)
    return _apply_layernorm(layer, out, "layernorm_zono")


def overlap_patch_embed_star(
    layer: nn.Module, input_sets: List, **kwargs,
) -> List:
    """Sound Star reach for OverlapPatchEmbed."""
    out = patch_embed_reach.patch_embed_star(layer, input_sets, **kwargs)
    return _apply_layernorm(layer, out, "layernorm_star")


def overlap_patch_embed_box(
    layer: nn.Module,
    input_sets: List,
    image_shape: Tuple[int, int, int] | None = None,
    image_layout: str = "HWC",
) -> List:
    """Sound Box reach for OverlapPatchEmbed.

    Forwards ``image_shape`` / ``image_layout`` to
    :func:`patch_embed_reach.patch_embed_box` (audit I2/I3).
    """
    out = patch_embed_reach.patch_embed_box(
        layer, input_sets,
        image_shape=image_shape, image_layout=image_layout,
    )
    return _apply_layernorm(layer, out, "layernorm_box")


def overlap_patch_embed_hexatope(
    layer: nn.Module,
    input_sets: List,
    image_shape: Tuple[int, int, int] | None = None,
    image_layout: str = "HWC",
) -> List:
    out = patch_embed_reach.patch_embed_hexatope(
        layer, input_sets,
        image_shape=image_shape, image_layout=image_layout,
    )
    return _apply_layernorm(layer, out, "layernorm_hexatope")


def overlap_patch_embed_octatope(
    layer: nn.Module,
    input_sets: List,
    image_shape: Tuple[int, int, int] | None = None,
    image_layout: str = "HWC",
) -> List:
    out = patch_embed_reach.patch_embed_octatope(
        layer, input_sets,
        image_shape=image_shape, image_layout=image_layout,
    )
    return _apply_layernorm(layer, out, "layernorm_octatope")
