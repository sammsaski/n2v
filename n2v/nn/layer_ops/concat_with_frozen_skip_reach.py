"""ConcatWithFrozenSkip reachability: concatenate input set with a constant.

Supported call shape (T1-4 / audit medium):

* **Single-token append** -- ``layer.dim == -1`` (or the last axis) AND
  the input set's flat dim equals ``layer.skip.numel()``. This is the
  common case (a single (D,) activation concatenated with a (D,) skip
  -> (2D,) output). The flat ``np.vstack`` matches ``torch.cat`` along
  ``dim=-1`` exactly.

* **Multi-token append** -- the reach would need to know ``L`` (or
  equivalently ``D_x``) to tile the skip per-token to match
  ``torch.cat([(B, L, D_x), (B, L, D_skip)], dim=-1)``. With only the
  flat-input dim and the layer's flat-skip dim, ``L`` is not
  recoverable -- e.g. flat=6 + skip=2 could be L=3, D_x=2 OR L=1,
  D_x=6, OR several other valid decompositions, and each produces a
  different output layout. Until an explicit shape attribute lands on
  the wrapper (see ``expected_input_shape`` follow-up), the multi-
  token path raises ``NotImplementedError`` rather than the previous
  silent ``vstack`` that mislabelled coordinates.

Anything else (``dim != -1``) likewise raises rather than silently
producing a permuted layout.

The wrapper ``ConcatWithFrozenSkip.forward`` (in
``n2v/nn/layers/concat_with_frozen_skip.py``) is independently fixed
in this commit to handle the (B, L, D) input + (D,) skip
broadcast case that previously raised a RuntimeError.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star, Zono


def _skip_vec(layer) -> np.ndarray:
    return layer.skip.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)


def _validate_call(layer, input_flat_dim: int, d_skip: int) -> None:
    if layer.dim != -1:
        raise NotImplementedError(
            f"ConcatWithFrozenSkip reach only supports dim=-1; got "
            f"dim={layer.dim}. Other dims need explicit token-shape info "
            f"(PR12_FIX_LIST T1-4)."
        )
    if input_flat_dim != d_skip:
        raise NotImplementedError(
            f"ConcatWithFrozenSkip reach: multi-token concat (input flat "
            f"dim {input_flat_dim} != skip flat dim {d_skip}) is not yet "
            f"supported -- the reach cannot recover L from flat dims "
            f"alone. Pre-PR12 the helper silently vstack'd the skip at "
            f"the END of the flat vector, mislabelling coordinates and "
            f"verifying a different function than the model computed. "
            f"Until the wrapper carries an explicit shape, single-token "
            f"(flat dim == skip dim) is the supported case. See "
            f"PR12_FIX_LIST T1-4."
        )


def concat_with_frozen_skip_box(layer, input_boxes: List[Box]) -> List[Box]:
    skip = _skip_vec(layer)
    d_skip = skip.shape[0]
    out: List[Box] = []
    for b in input_boxes:
        _validate_call(layer, b.dim, d_skip)
        out.append(Box(np.vstack([b.lb, skip]), np.vstack([b.ub, skip])))
    return out


def concat_with_frozen_skip_star(layer, input_stars: List[Star]) -> List[Star]:
    skip = _skip_vec(layer)
    d_skip = skip.shape[0]
    out: List[Star] = []
    for s in input_stars:
        _validate_call(layer, s.V.shape[0], d_skip)
        n_var = s.V.shape[1] - 1
        skip_block = np.hstack([skip, np.zeros((d_skip, n_var))])
        new_V = np.vstack([s.V, skip_block])
        out.append(Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub))
    return out


def concat_with_frozen_skip_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    skip = _skip_vec(layer)
    d_skip = skip.shape[0]
    out: List[Zono] = []
    for z in input_zonos:
        _validate_call(layer, z.c.shape[0], d_skip)
        n_gen = z.V.shape[1]
        new_c = np.vstack([z.c, skip])
        new_V = np.vstack([z.V, np.zeros((d_skip, n_gen))])
        out.append(Zono(new_c, new_V))
    return out


def concat_with_frozen_skip_hexatope(layer, input_sets):
    """Sound (box-lifted) Hexatope reach for ConcatWithFrozenSkip.

    PR-1 audit I7: previously absent. Box-lifts the input via IBP,
    runs ``concat_with_frozen_skip_box``, and rebuilds the Hexatope
    from the result. Loose but sound; symmetric to
    ``patch_embed_hexatope`` / ``cls_token_hexatope``.
    """
    from n2v.sets import Hexatope

    out = []
    for h in input_sets:
        lb, ub = h.estimate_ranges()
        box_in = Box(
            np.asarray(lb).reshape(-1, 1),
            np.asarray(ub).reshape(-1, 1),
        )
        box_out = concat_with_frozen_skip_box(layer, [box_in])[0]
        out.append(Hexatope.from_bounds(box_out.lb, box_out.ub))
    return out


def concat_with_frozen_skip_octatope(layer, input_sets):
    """Sound (box-lifted) Octatope reach for ConcatWithFrozenSkip (audit I7).

    Same box-lift pattern as ``concat_with_frozen_skip_hexatope``.
    """
    from n2v.sets import Octatope

    out = []
    for o in input_sets:
        lb, ub = o.estimate_ranges()
        box_in = Box(
            np.asarray(lb).reshape(-1, 1),
            np.asarray(ub).reshape(-1, 1),
        )
        box_out = concat_with_frozen_skip_box(layer, [box_in])[0]
        out.append(Octatope.from_bounds(box_out.lb, box_out.ub))
    return out
