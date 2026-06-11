"""MixFFN reachability — SegFormer Mix-FFN block.

``MixFFN.forward`` is:

    x = self.fc1(x)                          # (B, L, hidden_dim)
    x_2d = x.transpose(1, 2).reshape(b, d, h, h)   # (B, hidden_dim, H, W)
    x_2d = self.dwconv(x_2d)                 # depthwise Conv2d
    x = x_2d.flatten(2).transpose(1, 2)      # (B, L, hidden_dim)
    x = self.act(x)
    return self.fc2(x)                       # (B, L, dim)

The block is a fx leaf via ``N2VTracer`` (it lives in
``n2v.nn.layers.__all__``). PR-1 audit C3 caught that ``mix_ffn_passthrough``
unconditionally RAISED, making MixFFN unreachable end-to-end across every
set type -- a coverage regression with no escape.

This module implements a real reach helper that mirrors the forward:

1. Apply fc1 per-token via the existing block-tile linear helpers.
2. Reshape the post-fc1 flat reach into an ImageStar/ImageZono of
   shape ``(H, W, hidden_dim)`` (HWC == token-major flat; this
   identity is the same one ``patch_embed_reach`` relies on).
3. Apply ``dwconv`` via ``conv2d_reach``.
4. Flatten the ImageStar/ImageZono back to a token-major flat set.
5. Apply GELU per-element.
6. Apply fc2 per-token via block-tile.

Hex/Oct paths box-lift via ``estimate_ranges`` -> Box reach (which
itself routes through the ImageZono path inside) -> rebuild Hex/Oct
from bounds. Sound but loose; identical pattern to ``patch_embed_hexatope``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch.nn as nn

from n2v.sets import Box, Star, Zono, Hexatope, Octatope
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops import (
    linear_reach,
    conv2d_reach,
    gelu_reach,
)


# ---------------------------------------------------------------------------
# Shape-utility helpers
# ---------------------------------------------------------------------------


def _hw_from_L(L: int, layer: nn.Module) -> int:
    """Infer the spatial side ``H = W = sqrt(L)``. Raise on non-square L,
    matching the forward's own ``ValueError``.
    """
    h = int(round(L ** 0.5))
    if h * h != L:
        raise NotImplementedError(
            f"MixFFN reach: sequence length L={L} is not a perfect "
            f"square; the dwconv requires a square spatial layout."
        )
    return h


def _flat_to_image_zono(flat_zono: Zono, H: int, W: int, C: int) -> ImageZono:
    """Lift a flat Zono of dim ``H*W*C`` (HWC row-major == token-major)
    to an ImageZono with the same flat data.
    """
    return ImageZono(flat_zono.c, flat_zono.V, H, W, C)


def _flat_to_image_star(flat_star: Star, H: int, W: int, C: int) -> ImageStar:
    """Lift a flat Star of dim ``H*W*C`` to an ImageStar.

    ``Star.V`` is 2D ``(H*W*C, nVar+1)``; reshape to
    ``(H, W, C, nVar+1)``.
    """
    V_4d = flat_star.V.reshape(H, W, C, -1)
    return ImageStar(
        V_4d, flat_star.C, flat_star.d,
        flat_star.predicate_lb, flat_star.predicate_ub,
        H, W, C,
    )


# ---------------------------------------------------------------------------
# Per-set-type reach helpers
# ---------------------------------------------------------------------------


def _mix_ffn_zono(layer: nn.Module, z: Zono, n_tokens: int) -> Zono:
    fc1: nn.Linear = layer.fc1  # type: ignore[attr-defined]
    dwconv: nn.Conv2d = layer.dwconv  # type: ignore[attr-defined]
    act: nn.GELU = layer.act  # type: ignore[attr-defined]
    fc2: nn.Linear = layer.fc2  # type: ignore[attr-defined]

    # Step 1: fc1 per-token. Expect (L * dim,) flat -> (L * hidden,) flat.
    z_after_fc1 = linear_reach.linear_zono(
        fc1, [z], expected_n_tokens=n_tokens,
    )[0]

    # Step 2: reshape to ImageZono.
    H = _hw_from_L(n_tokens, layer)
    hidden = int(fc1.out_features)
    iz = _flat_to_image_zono(z_after_fc1, H, H, hidden)

    # Step 3: dwconv.
    iz_after_dw = conv2d_reach.conv2d_zono(dwconv, [iz])[0]

    # Step 4: flatten back to token-major (HWC == token-major).
    z_flat = iz_after_dw.to_zono() if isinstance(iz_after_dw, ImageZono) \
        else iz_after_dw

    # Step 5: GELU.
    mode = getattr(act, "approximate", "none")
    if mode == "tanh":
        z_act = gelu_reach.gelu_tanh_zono([z_flat])[0]
    elif mode == "none":
        z_act = gelu_reach.gelu_zono([z_flat])[0]
    else:
        raise NotImplementedError(
            f"MixFFN reach: nn.GELU(approximate={mode!r}) not supported."
        )

    # Step 6: fc2 per-token.
    z_out = linear_reach.linear_zono(
        fc2, [z_act], expected_n_tokens=n_tokens,
    )[0]
    return z_out


def _mix_ffn_star(layer: nn.Module, s: Star, n_tokens: int) -> Star:
    fc1: nn.Linear = layer.fc1  # type: ignore[attr-defined]
    dwconv: nn.Conv2d = layer.dwconv  # type: ignore[attr-defined]
    act: nn.GELU = layer.act  # type: ignore[attr-defined]
    fc2: nn.Linear = layer.fc2  # type: ignore[attr-defined]

    s_after_fc1 = linear_reach.linear_star(
        fc1, [s], expected_n_tokens=n_tokens,
    )[0]
    H = _hw_from_L(n_tokens, layer)
    hidden = int(fc1.out_features)
    istar = _flat_to_image_star(s_after_fc1, H, H, hidden)
    istar_after_dw = conv2d_reach.conv2d_star(dwconv, [istar])[0]
    s_flat = istar_after_dw.to_star() if isinstance(istar_after_dw, ImageStar) \
        else istar_after_dw

    mode = getattr(act, "approximate", "none")
    if mode == "tanh":
        s_act = gelu_reach.gelu_tanh_star_approx([s_flat])[0]
    elif mode == "none":
        s_act = gelu_reach.gelu_star_approx([s_flat])[0]
    else:
        raise NotImplementedError(
            f"MixFFN reach: nn.GELU(approximate={mode!r}) not supported."
        )

    s_out = linear_reach.linear_star(
        fc2, [s_act], expected_n_tokens=n_tokens,
    )[0]
    return s_out


def _mix_ffn_box(layer: nn.Module, b: Box, n_tokens: int) -> Box:
    """Box reach for MixFFN.

    The dwconv requires (H, W, C) layout, so we lift the input Box to a
    no-noise ImageZono, run the Zono chain, and project back via IBP.
    Sound but loose.
    """
    iz = ImageZono.from_bounds(
        b.lb, b.ub,
        height=_hw_from_L(n_tokens, layer),
        width=_hw_from_L(n_tokens, layer),
        num_channels=int(layer.fc1.in_features),
    )
    z_in = iz.to_zono()
    z_out = _mix_ffn_zono(layer, z_in, n_tokens)
    lb, ub = z_out.get_bounds()
    return Box(
        np.asarray(lb).reshape(-1, 1), np.asarray(ub).reshape(-1, 1),
    )


def mix_ffn_passthrough(
    layer: nn.Module,
    input_sets: List,
    method: str = "exact",
    **kwargs,
):
    """Dispatch entry point used by ``dispatcher.reach_layer`` (Box and
    Star paths share this name from the pre-PR-1 era).

    Routes to the correct per-set-type helper based on ``input_sets[0]``.

    PR-1 audit C3: this used to RAISE unconditionally. The rationale at
    the time was "n2v relies on fx to decompose MixFFN" -- but T1-7
    made MixFFN an fx leaf, so the two halves contradicted and any
    model with MixFFN became unverifiable on every set type. This
    helper now implements the forward sequence directly:
    Linear -> ImageZono/ImageStar reshape -> Conv2d -> flatten ->
    GELU -> Linear. Hex/Oct paths are box-lifted via the dedicated
    wrappers below.
    """
    n_tokens = kwargs.get("n_tokens")
    if n_tokens is None:
        n_tokens = getattr(layer, "n_tokens", None)
    if n_tokens is None:
        raise NotImplementedError(
            "MixFFN reach requires the per-token count ``n_tokens`` -- "
            "either pass ``n_tokens=L`` to reach() or set "
            "``model.n_tokens``. L must be a perfect square (audit C3)."
        )
    n_tokens = int(n_tokens)

    if not input_sets:
        return []
    first = input_sets[0]
    if isinstance(first, (ImageStar, Star)):
        if isinstance(first, ImageStar):
            return [_mix_ffn_star(layer, s.to_star(), n_tokens) for s in input_sets]
        return [_mix_ffn_star(layer, s, n_tokens) for s in input_sets]
    if isinstance(first, (ImageZono, Zono)):
        if isinstance(first, ImageZono):
            return [_mix_ffn_zono(layer, z.to_zono(), n_tokens) for z in input_sets]
        return [_mix_ffn_zono(layer, z, n_tokens) for z in input_sets]
    if isinstance(first, Box):
        return [_mix_ffn_box(layer, b, n_tokens) for b in input_sets]
    if isinstance(first, Hexatope):
        out = []
        for h in input_sets:
            lb, ub = h.estimate_ranges()
            box_in = Box(
                np.asarray(lb).reshape(-1, 1),
                np.asarray(ub).reshape(-1, 1),
            )
            box_out = _mix_ffn_box(layer, box_in, n_tokens)
            out.append(Hexatope.from_bounds(box_out.lb, box_out.ub))
        return out
    if isinstance(first, Octatope):
        out = []
        for o in input_sets:
            lb, ub = o.estimate_ranges()
            box_in = Box(
                np.asarray(lb).reshape(-1, 1),
                np.asarray(ub).reshape(-1, 1),
            )
            box_out = _mix_ffn_box(layer, box_in, n_tokens)
            out.append(Octatope.from_bounds(box_out.lb, box_out.ub))
        return out
    raise NotImplementedError(
        f"MixFFN reach: unsupported set type {type(first).__name__}."
    )
