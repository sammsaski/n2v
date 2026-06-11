"""PatchEmbed reachability — Conv2d -> flatten(2) -> transpose(1, 2).

PatchEmbed maps a ``(B, C, H, W)`` image into a ``(B, L, embed_dim)``
sequence where ``L = (H // patch_size) * (W // patch_size)``. The
implementation is:

    x = self.proj(x)           # (B, embed_dim, H/p, W/p)
    return x.flatten(2).transpose(1, 2)  # (B, L, embed_dim)

We treat PatchEmbed as an fx leaf (see ``n2v/nn/_tracer.py``) and
implement reach by:

1. Running the existing Conv2d reach helper on the input set. For
   ImageZono / ImageStar inputs this yields an output set in CHW layout
   (channel-major) with the new ``(H/p, W/p, embed_dim)`` spatial shape.

2. Permuting the flat representation from channel-major
   ``(c, h, w)`` to token-major ``(h*w, c)``. After the permutation the
   set carries the **(L, embed_dim)** layout the downstream transformer
   expects when it operates on a flat ``Star/Zono`` of dim ``L * embed_dim``.

This is a sound, exact-affine reach (Conv2d + permutation) and lands as
the keystone for the ViT integration test (``tests/integration/
test_minimal_vit.py``).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch.nn as nn

from n2v.sets import Box, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops import conv2d_reach


def _channel_major_to_token_major_index(
    n_channels: int, height: int, width: int,
) -> np.ndarray:
    """Build a permutation that takes a flat ``(C, H, W)`` (channel-major)
    layout to ``(H*W, C)`` (token-major) layout.

    The output index ``perm[i]`` gives the source row in the channel-major
    set that should occupy position ``i`` in the token-major layout. For
    ``C=2, H=2, W=2`` the channel-major flat order is
    ``[c0_h0_w0, c0_h0_w1, c0_h1_w0, c0_h1_w1, c1_h0_w0, ...]`` and the
    token-major target is
    ``[c0_h0_w0, c1_h0_w0, c0_h0_w1, c1_h0_w1, c0_h1_w0, c1_h1_w0, ...]``.
    """
    # channel-major index: i_cm = c * (H*W) + h * W + w
    # token-major index:   i_tm = (h * W + w) * C + c
    # We want perm[i_tm] = i_cm.
    perm = np.empty(n_channels * height * width, dtype=np.int64)
    for c in range(n_channels):
        for h in range(height):
            for w in range(width):
                i_cm = c * (height * width) + h * width + w
                i_tm = (h * width + w) * n_channels + c
                perm[i_tm] = i_cm
    return perm


def _permute_rows_flat(
    flat_centre: np.ndarray, flat_generators: np.ndarray, perm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute rows of a flat (centre, generators) pair."""
    return flat_centre[perm], flat_generators[perm]


def _patch_embed_image_zono(layer: nn.Module, input_image_zono: ImageZono) -> Zono:
    """Apply Conv2d via ``conv2d_zono`` then flatten to token-major.

    ``conv2d_zono`` returns an ImageZono with ``(H, W, C, n_gen)`` in V
    and HWC-row-major flat positions. PatchEmbed's forward is
    ``proj(x).flatten(2).transpose(1, 2)`` which produces a
    ``(B, L=H*W, embed_dim=C)`` sequence -- i.e. token-major flat order
    ``(token_idx) * C + c`` for ``token_idx = h * W + w``.

    This token-major flat layout EQUALS the HWC-row-major flat layout
    (worked out: ``h*W*C + w*C + c == (h*W + w) * C + c``), so simply
    calling ``ImageZono.to_zono()`` produces the correct flat Zono. No
    permutation is needed.
    """
    proj: nn.Conv2d = layer.proj  # type: ignore[attr-defined]
    conv_out = conv2d_reach.conv2d_zono(proj, [input_image_zono])
    assert len(conv_out) == 1
    out_set = conv_out[0]
    if isinstance(out_set, ImageZono):
        return out_set.to_zono()
    return out_set


def _patch_embed_image_star(layer: nn.Module, input_image_star: ImageStar) -> Star:
    """Apply Conv2d via ``conv2d_star`` then flatten to token-major.

    See ``_patch_embed_image_zono`` for the layout argument.
    ``ImageStar.to_star()`` flattens in HWC order which matches the
    desired token-major layout.
    """
    proj: nn.Conv2d = layer.proj  # type: ignore[attr-defined]
    star_out = conv2d_reach.conv2d_star(proj, [input_image_star])
    assert len(star_out) == 1
    out_set = star_out[0]
    if isinstance(out_set, ImageStar):
        return out_set.to_star()
    return out_set


def patch_embed_zono(layer: nn.Module, input_sets: List) -> List[Zono]:
    """Sound Zono reach for PatchEmbed.

    Expects each input set to be an :class:`ImageZono` (with explicit
    spatial shape); a flat :class:`Zono` cannot recover the H/W needed
    by Conv2d and will raise.
    """
    out: List[Zono] = []
    for s in input_sets:
        if not isinstance(s, ImageZono):
            raise TypeError(
                f"PatchEmbed Zono reach requires ImageZono input "
                f"(needs H/W/C); got {type(s).__name__}."
            )
        out.append(_patch_embed_image_zono(layer, s))
    return out


def patch_embed_star(layer: nn.Module, input_sets: List, **kwargs) -> List[Star]:
    """Sound Star reach for PatchEmbed.

    Expects each input set to be an :class:`ImageStar`.
    """
    out: List[Star] = []
    for s in input_sets:
        if not isinstance(s, ImageStar):
            raise TypeError(
                f"PatchEmbed Star reach requires ImageStar input "
                f"(needs H/W/C); got {type(s).__name__}."
            )
        out.append(_patch_embed_image_star(layer, s))
    return out


def _chw_flat_to_hwc_flat(flat: np.ndarray, H: int, W: int, C: int) -> np.ndarray:
    """Permute a flat (C, H, W) row-major vector to (H, W, C) row-major.

    ``flat`` is shape ``(C*H*W, 1)`` or ``(C*H*W,)``. Returns the same
    column-vector shape as the input.
    """
    flat = np.asarray(flat).reshape(-1)
    chw = flat.reshape(C, H, W)
    hwc = np.transpose(chw, (1, 2, 0))
    return hwc.reshape(-1, 1)


def patch_embed_box(
    layer: nn.Module,
    input_sets: List,
    image_shape: Tuple[int, int, int] | None = None,
    image_layout: str = "HWC",
) -> List[Box]:
    """Box reach for PatchEmbed.

    Box does not carry explicit ``(H, W, C)``, but PatchEmbed.proj is a
    fully-affine Conv2d so we can recover a sound (loose) Box by lifting
    each input Box to an ImageZono using the layer's known
    ``in_channels`` and the implied image size from the Box's flat dim,
    running ``patch_embed_zono``, and taking the IBP envelope of the
    result.

    PR-1 audit I2/I3: the previous helper silently assumed
    ``ImageZono.from_bounds(c, h, w, in_c)`` HWC-flat layout AND
    inferred a square image side from pixel count. This was unsound
    in two distinct ways:

      * I2 (layout): A CHW-flat input (PyTorch's native flatten of a
        (C, H, W) tensor) was reshaped as HWC by ``ImageZono.from_bounds``,
        permuting the channels. Conv2d then operated on a permuted
        image and the reach excluded true forward outputs. Masked in
        PR-1 because the test fixture uses ``in_channels=1`` (HWC ==
        CHW for C=1).
      * I3 (shape): A non-square image (e.g. 2x8x1) whose pixel count is
        a perfect square (16) was silently reshaped as a 4x4 image, and
        the conv reach again excluded true outputs.

    Fix: accept explicit ``image_shape=(H, W, C)`` and
    ``image_layout`` ('HWC' or 'CHW') kwargs. When absent and
    ``in_channels > 1``, raise -- layout is unrecoverable. When absent
    and ``in_channels == 1``, allow the legacy square inference but
    emit a ``UserWarning`` to make the assumption auditable (CHW ==
    HWC for C=1 so this case is sound).
    """
    import warnings as _warnings
    from n2v.sets.image_zono import ImageZono
    from n2v.sets import Zono

    proj: nn.Conv2d = layer.proj  # type: ignore[attr-defined]
    in_c = int(proj.in_channels)

    if image_shape is not None:
        H, W, C = (int(v) for v in image_shape)
        if C != in_c:
            raise NotImplementedError(
                f"PatchEmbed Box reach: image_shape channels={C} "
                f"disagrees with layer.proj.in_channels={in_c}."
            )
    elif in_c > 1:
        raise NotImplementedError(
            f"PatchEmbed Box reach: in_channels={in_c} > 1 requires "
            f"explicit ``image_shape=(H, W, C)`` and "
            f"``image_layout='HWC'|'CHW'`` -- flat layout is "
            f"unrecoverable from a Box alone (audit I2/I3). Either "
            f"pass image_shape kwarg or use ImageZono input."
        )
    if image_layout not in ("HWC", "CHW"):
        raise NotImplementedError(
            f"PatchEmbed Box reach: image_layout must be 'HWC' or "
            f"'CHW', got {image_layout!r}."
        )

    out: List[Box] = []
    for box in input_sets:
        flat = box.dim
        if flat % in_c != 0:
            raise NotImplementedError(
                f"PatchEmbed Box reach: flat input dim {flat} is not "
                f"divisible by in_channels={in_c}; cannot infer H/W."
            )
        if image_shape is not None:
            H_use, W_use = H, W
            if H_use * W_use * in_c != flat:
                raise NotImplementedError(
                    f"PatchEmbed Box reach: image_shape "
                    f"({H_use}, {W_use}, {in_c}) does not match flat "
                    f"input dim {flat}."
                )
            lb_use = box.lb
            ub_use = box.ub
            if image_layout == "CHW":
                lb_use = _chw_flat_to_hwc_flat(lb_use, H_use, W_use, in_c)
                ub_use = _chw_flat_to_hwc_flat(ub_use, H_use, W_use, in_c)
        else:
            # in_c == 1 here; HWC == CHW so layout is moot, but warn so
            # the square inference is auditable.
            n_pixels = flat // in_c
            side = int(round(n_pixels ** 0.5))
            if side * side != n_pixels:
                raise NotImplementedError(
                    f"PatchEmbed Box reach: cannot infer a square image "
                    f"of {n_pixels} pixels; pass image_shape kwarg."
                )
            _warnings.warn(
                f"PatchEmbed Box reach: inferring square image "
                f"({side}, {side}, {in_c}) from flat dim {flat} without "
                f"an explicit image_shape kwarg. Pass "
                f"``image_shape=({side}, {side}, {in_c})`` to silence "
                f"this warning (see PR-1 audit I3).",
                UserWarning, stacklevel=3,
            )
            H_use = W_use = side
            lb_use = box.lb
            ub_use = box.ub
        zono_in = ImageZono.from_bounds(
            lb_use, ub_use,
            height=H_use, width=W_use, num_channels=in_c,
        )
        zono_out = _patch_embed_image_zono(layer, zono_in)
        lb, ub = zono_out.get_bounds()
        out.append(Box(
            np.asarray(lb).reshape(-1, 1), np.asarray(ub).reshape(-1, 1),
        ))
    return out


def _hex_oct_lb_ub(s):
    """Fast IBP (lb, ub) for Hex/Oct -- their zero-arg ``estimate_ranges``."""
    lb, ub = s.estimate_ranges()
    return np.asarray(lb).reshape(-1, 1), np.asarray(ub).reshape(-1, 1)


def patch_embed_hexatope(
    layer: nn.Module,
    input_sets: List,
    image_shape: Tuple[int, int, int] | None = None,
    image_layout: str = "HWC",
):
    """Sound (box-lifted) Hexatope reach for PatchEmbed.

    Lifts each Hexatope to its IBP box envelope, runs ``patch_embed_box``
    (which requires explicit image_shape for in_channels > 1; see audit
    I2/I3), then constructs a fresh Hexatope from the result.
    """
    from n2v.sets import Hexatope

    out = []
    for h in input_sets:
        lb, ub = _hex_oct_lb_ub(h)
        box_in = Box(lb, ub)
        box_out = patch_embed_box(
            layer, [box_in],
            image_shape=image_shape, image_layout=image_layout,
        )[0]
        out.append(Hexatope.from_bounds(box_out.lb, box_out.ub))
    return out


def patch_embed_octatope(
    layer: nn.Module,
    input_sets: List,
    image_shape: Tuple[int, int, int] | None = None,
    image_layout: str = "HWC",
):
    """Sound (box-lifted) Octatope reach for PatchEmbed.

    Same box-lift pattern as ``patch_embed_hexatope``.
    """
    from n2v.sets import Octatope

    out = []
    for o in input_sets:
        lb, ub = _hex_oct_lb_ub(o)
        box_in = Box(lb, ub)
        box_out = patch_embed_box(
            layer, [box_in],
            image_shape=image_shape, image_layout=image_layout,
        )[0]
        out.append(Octatope.from_bounds(box_out.lb, box_out.ub))
    return out
