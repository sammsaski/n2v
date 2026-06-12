"""
Linear layer reachability operations.

Works directly with PyTorch nn.Linear layers.

T1-7 (ViT enable): nn.Linear in a transformer is applied PER TOKEN to a
``(B, L, D_in)`` sequence, producing ``(B, L, D_out)``. After
``PatchEmbed`` and ``CLSToken`` the reach set is flat with dim
``L * D_in`` while ``layer.weight`` is ``(D_out, D_in)``.

The per-token map is applied BLOCKWISE: the flat basis rows are
reshaped to ``(L, D_in, ...)``, multiplied by ``W`` once via einsum,
and flattened back -- O(L * D_out * D_in) per basis column. (Copilot
review: the previous implementation materialised the block-diagonal
``kron(I_L, W)`` -- O((L*D_out) * (L*D_in)) memory/time, which OOMs at
realistic transformer token counts.)
"""

from __future__ import annotations

import warnings

import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from n2v.sets import Star, Zono, Box, Hexatope, Octatope


def _resolve_n_tokens(
    in_features: int,
    input_flat_dim: int,
    expected_n_tokens: Optional[int] = None,
) -> int:
    """Determine the per-token block count ``L`` for a flat input.

    Returns ``L >= 1``. Mirrors the audit-I8 contract of the previous
    ``_maybe_block_tile_linear``:

    * ``input_flat_dim == in_features``: L = 1 (plain Linear).
    * not divisible: raise (the concrete forward would fail too).
    * caller-declared ``expected_n_tokens`` disagreeing with the
      divisibility-inferred ``L``: raise -- the reach would silently
      verify a different function.
    * inferred ``L > 1`` with no explicit signal: ``UserWarning`` so
      the silent inference is auditable.
    """
    if input_flat_dim == in_features:
        return 1
    if input_flat_dim % in_features != 0:
        raise ValueError(
            f"Linear reach: flat input dim {input_flat_dim} is not a "
            f"multiple of in_features={in_features}; the concrete "
            f"forward would fail on this input."
        )
    L = input_flat_dim // in_features
    if expected_n_tokens is not None and int(expected_n_tokens) != L:
        raise NotImplementedError(
            f"linear block-tile: caller-provided n_tokens="
            f"{expected_n_tokens} disagrees with divisibility-inferred "
            f"L={L} for input_flat_dim={input_flat_dim} / "
            f"in_features={in_features}. The reach would silently "
            f"verify a different function. Fix the shape upstream or "
            f"correct the n_tokens kwarg."
        )
    if expected_n_tokens is None and L > 1:
        warnings.warn(
            f"linear block-tile inferred L={L} (input_flat_dim="
            f"{input_flat_dim}, in_features={in_features}) without an "
            f"explicit n_tokens signal. Pass ``n_tokens={L}`` to "
            f"reach() or set ``model.n_tokens`` to make this "
            f"verifiable. See PR-1 audit I8.",
            UserWarning, stacklevel=3,
        )
    return L


def _block_apply(M: np.ndarray, W: np.ndarray, L: int) -> np.ndarray:
    """Apply ``y = (I_L (x) W) @ M`` without materialising the Kronecker.

    ``M`` has shape ``(L * D_in, k)``; the result is ``(L * D_out, k)``.
    """
    d_in = W.shape[1]
    d_out = W.shape[0]
    k = M.shape[1]
    blocks = M.reshape(L, d_in, k)
    out = np.einsum('od,ldk->lok', W, blocks)
    return out.reshape(L * d_out, k)


def _tile_bias(b: Optional[np.ndarray], L: int) -> Optional[np.ndarray]:
    if b is None:
        return None
    return np.tile(np.asarray(b, dtype=np.float64).reshape(-1), L)


def _weights(layer: nn.Linear) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    W = layer.weight.detach().cpu().numpy().astype(np.float64)
    b = (layer.bias.detach().cpu().numpy().astype(np.float64)
         if layer.bias is not None else None)
    return W, b


def linear_star(
    layer: nn.Linear,
    input_stars: List[Star],
    expected_n_tokens: Optional[int] = None,
) -> List[Star]:
    """
    Exact reachability for Linear layer using Star sets.

    Args:
        layer: PyTorch nn.Linear layer
        input_stars: List of input Star sets
        expected_n_tokens: PR-1 audit I8 -- declared per-token count;
            the helper raises on mismatch with the inferred block count
            instead of silently verifying a different function.

    Returns:
        List of output Star sets
    """
    W, b = _weights(layer)

    output_stars = []
    for star in input_stars:
        L = _resolve_n_tokens(W.shape[1], star.dim, expected_n_tokens)
        if L == 1:
            if b is not None:
                output_stars.append(star.affine_map(W, b.reshape(-1, 1)))
            else:
                output_stars.append(star.affine_map(W))
            continue
        # ImageStar carries a 4D basis (H, W, C, nVar+1); ``_block_apply``
        # expects the 2D flat form. Flatten via ``to_star`` (HWC row-major
        # == pixel-major x channel, which is exactly the per-pixel block
        # structure the L > 1 path applies W over) and re-wrap with the
        # new channel count afterwards. Without this, an L > 1 ImageStar
        # crashed with a confusing reshape ValueError (deep-dive review).
        from n2v.sets.image_star import ImageStar as _ImageStar
        rewrap = None
        if isinstance(star, _ImageStar):
            # Re-wrapping as an image is only meaningful when the block
            # count equals the pixel count (per-pixel Linear over the
            # channel axis); otherwise return the flat Star.
            if L == star.height * star.width:
                rewrap = (star.height, star.width)
            star = star.to_star()
        new_V = _block_apply(star.V, W, L)
        bt = _tile_bias(b, L)
        if bt is not None:
            new_V[:, 0] = new_V[:, 0] + bt
        out = Star(new_V, star.C, star.d,
                   star.predicate_lb, star.predicate_ub)
        if rewrap is not None:
            # Per-pixel Linear over an image: channel count becomes
            # out_features; spatial shape is preserved.
            out = out.to_image_star(rewrap[0], rewrap[1], W.shape[0])
        output_stars.append(out)

    return output_stars


def linear_zono(
    layer: nn.Linear,
    input_zonos: List[Zono],
    expected_n_tokens: Optional[int] = None,
) -> List[Zono]:
    """
    Exact reachability for Linear layer using Zonotopes.

    Args:
        layer: PyTorch nn.Linear layer
        input_zonos: List of input Zonotopes
        expected_n_tokens: see ``linear_star``.

    Returns:
        List of output Zonotopes
    """
    W, b = _weights(layer)

    output_zonos = []
    for zono in input_zonos:
        L = _resolve_n_tokens(W.shape[1], zono.dim, expected_n_tokens)
        if L == 1:
            if b is not None:
                output_zonos.append(zono.affine_map(W, b.reshape(-1, 1)))
            else:
                output_zonos.append(zono.affine_map(W))
            continue
        new_c = _block_apply(zono.c, W, L)
        bt = _tile_bias(b, L)
        if bt is not None:
            new_c = new_c + bt.reshape(-1, 1)
        new_V = _block_apply(zono.V, W, L)
        output_zonos.append(Zono(new_c, new_V))

    return output_zonos


def linear_box(
    layer: nn.Linear,
    input_boxes: List[Box],
    expected_n_tokens: Optional[int] = None,
) -> List[Box]:
    """
    Exact reachability for Linear layer using Boxes.

    Args:
        layer: PyTorch nn.Linear layer
        input_boxes: List of input Boxes
        expected_n_tokens: see ``linear_star``.

    Returns:
        List of output Boxes
    """
    W, b = _weights(layer)
    W_pos = np.maximum(W, 0.0)
    W_neg = np.minimum(W, 0.0)

    output_boxes = []
    for box in input_boxes:
        L = _resolve_n_tokens(W.shape[1], box.dim, expected_n_tokens)
        d_in = W.shape[1]
        lb = np.asarray(box.lb, dtype=np.float64).reshape(L, d_in)
        ub = np.asarray(box.ub, dtype=np.float64).reshape(L, d_in)
        # Exact interval affine per token block.
        new_lb = lb @ W_pos.T + ub @ W_neg.T
        new_ub = ub @ W_pos.T + lb @ W_neg.T
        if b is not None:
            new_lb = new_lb + b.reshape(1, -1)
            new_ub = new_ub + b.reshape(1, -1)
        output_boxes.append(
            Box(new_lb.reshape(-1, 1), new_ub.reshape(-1, 1)))

    return output_boxes


def linear_hexatope(
    layer: nn.Linear,
    input_hexatopes: List[Hexatope],
    expected_n_tokens: Optional[int] = None,
) -> List[Hexatope]:
    """
    Exact reachability for Linear layer using Hexatopes.

    Args:
        layer: PyTorch nn.Linear layer
        input_hexatopes: List of input Hexatopes
        expected_n_tokens: see ``linear_star``.

    Returns:
        List of output Hexatopes
    """
    W, b = _weights(layer)

    output_hexatopes = []
    for hexatope in input_hexatopes:
        L = _resolve_n_tokens(W.shape[1], hexatope.dim, expected_n_tokens)
        if L == 1:
            if b is not None:
                output_hexatopes.append(hexatope.affine_map(W, b))
            else:
                output_hexatopes.append(hexatope.affine_map(W))
            continue
        center = np.asarray(
            hexatope.center, dtype=np.float64).reshape(-1, 1)
        new_center = _block_apply(center, W, L).reshape(-1)
        bt = _tile_bias(b, L)
        if bt is not None:
            new_center = new_center + bt
        new_gens = _block_apply(hexatope.generators, W, L)
        output_hexatopes.append(
            Hexatope(new_center, new_gens, hexatope.dcs.copy()))

    return output_hexatopes


def linear_octatope(
    layer: nn.Linear,
    input_octatopes: List[Octatope],
    expected_n_tokens: Optional[int] = None,
) -> List[Octatope]:
    """
    Exact reachability for Linear layer using Octatopes.

    Args:
        layer: PyTorch nn.Linear layer
        input_octatopes: List of input Octatopes
        expected_n_tokens: see ``linear_star``.

    Returns:
        List of output Octatopes
    """
    W, b = _weights(layer)

    output_octatopes = []
    for octatope in input_octatopes:
        L = _resolve_n_tokens(W.shape[1], octatope.dim, expected_n_tokens)
        if L == 1:
            if b is not None:
                output_octatopes.append(octatope.affine_map(W, b))
            else:
                output_octatopes.append(octatope.affine_map(W))
            continue
        center = np.asarray(
            octatope.center, dtype=np.float64).reshape(-1, 1)
        new_center = _block_apply(center, W, L).reshape(-1)
        bt = _tile_bias(b, L)
        if bt is not None:
            new_center = new_center + bt
        new_gens = _block_apply(octatope.generators, W, L)
        output_octatopes.append(
            Octatope(new_center, new_gens, octatope.utvpi.copy()))

    return output_octatopes
