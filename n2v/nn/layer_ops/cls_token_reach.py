"""CLSToken reachability: prepend a learnable token to a sequence.

Concatenation with a constant token vector routes through the same
pattern as :mod:`concat_with_frozen_skip_reach` (with the skip placed
*before* the running activation instead of after).

Hex/Oct paths (PR-1 audit I7) are box-lifted: prepending a constant
to a Hexatope/Octatope would require surgery on the half-space basis
that does not yet exist, so the helper takes the IBP envelope of the
input set and re-builds via ``set_type.from_bounds``. Sound but loose;
identical pattern to :mod:`patch_embed_reach.patch_embed_hexatope`.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box, Star, Zono


def _token(layer) -> np.ndarray:
    return layer.token.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)


def cls_token_box(layer, input_boxes: List[Box]) -> List[Box]:
    tok = _token(layer)
    return [Box(np.vstack([tok, b.lb]), np.vstack([tok, b.ub])) for b in input_boxes]


def cls_token_star(layer, input_stars: List[Star]) -> List[Star]:
    tok = _token(layer)
    out: List[Star] = []
    for s in input_stars:
        m = tok.shape[0]
        n_var = s.V.shape[1] - 1
        prepend = np.hstack([tok, np.zeros((m, n_var))])
        new_V = np.vstack([prepend, s.V])
        out.append(Star(new_V, s.C, s.d, s.predicate_lb, s.predicate_ub))
    return out


def cls_token_zono(layer, input_zonos: List[Zono]) -> List[Zono]:
    tok = _token(layer)
    out: List[Zono] = []
    for z in input_zonos:
        m = tok.shape[0]
        new_c = np.vstack([tok, z.c])
        new_V = np.vstack([np.zeros((m, z.V.shape[1])), z.V])
        out.append(Zono(new_c, new_V))
    return out


def cls_token_hexatope(layer, input_sets):
    """Sound (box-lifted) Hexatope reach for CLSToken.

    PR-1 audit I7: previously absent. The N2VTracer leaf-treats CLSToken
    so an end-to-end Hexatope ViT fell through the dispatcher and
    raised ``NotImplementedError``. This box-lifts the input via IBP,
    runs ``cls_token_box``, and rebuilds the Hexatope from the result.
    Loose but sound; symmetric to ``patch_embed_hexatope``.
    """
    from n2v.sets import Hexatope

    out = []
    for h in input_sets:
        lb, ub = h.estimate_ranges()
        box_in = Box(
            np.asarray(lb).reshape(-1, 1),
            np.asarray(ub).reshape(-1, 1),
        )
        box_out = cls_token_box(layer, [box_in])[0]
        out.append(Hexatope.from_bounds(box_out.lb, box_out.ub))
    return out


def cls_token_octatope(layer, input_sets):
    """Sound (box-lifted) Octatope reach for CLSToken (audit I7).

    Same box-lift pattern as ``cls_token_hexatope``.
    """
    from n2v.sets import Octatope

    out = []
    for o in input_sets:
        lb, ub = o.estimate_ranges()
        box_in = Box(
            np.asarray(lb).reshape(-1, 1),
            np.asarray(ub).reshape(-1, 1),
        )
        box_out = cls_token_box(layer, [box_in])[0]
        out.append(Octatope.from_bounds(box_out.lb, box_out.ub))
    return out
