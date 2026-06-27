"""
Split-selection strategies (Phase 1).

Every selector returns the ``NeuronKey`` of the unstable neuron to split next.
The two strategies under test differ *only* in the witness feeding the
``eps * influence`` score:

  * ``FaithfulSelector`` scores at the constraint-faithful witness (minimizer
    over the full predicate polytope P) -- reuses the witness already solved by
    the BaB loop for pruning, so it adds no extra LP.
  * ``BoxCornerSelector`` scores at the box-corner witness (minimizer over the
    predicate box only, ignoring C/d) -- the DRG-BaB analog. It solves its own
    box LP, which may return a point outside P.

``RandomSelector`` and ``BoundWidthSelector`` are controls that use no witness.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.refine.types import LinearSpec, NeuronKey, NeuronMeta, Witness
from n2v.refine.witness import binding_row, score_vector, violation_lp


class Selector:
    """Base class. ``name`` labels the strategy in experiment output."""

    name = "base"

    def choose(
        self,
        S_out: Star,
        spec: LinearSpec,
        meta: List[NeuronMeta],
        faithful_witness: Witness,
        lp_solver=LPSolver.DEFAULT,
    ) -> Optional[NeuronKey]:
        raise NotImplementedError


class FaithfulSelector(Selector):
    name = "faithful"

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=LPSolver.DEFAULT):
        if not meta:
            return None
        h = spec.G[faithful_witness.binding_row]
        scores, _ = score_vector(faithful_witness.alpha, meta, S_out, h)
        return meta[int(np.argmax(scores))].key


class BoxCornerSelector(Selector):
    name = "box_corner"

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=LPSolver.DEFAULT):
        if not meta:
            return None
        box = violation_lp(S_out, spec, include_Cd=False, lp_solver=lp_solver)
        if box is None:
            # Box LP infeasible (cannot happen when P is non-empty, since box >= P);
            # fall back to the faithful witness rather than failing.
            alpha, binding = faithful_witness.alpha, faithful_witness.binding_row
        else:
            alpha = box[0]
            binding = binding_row(S_out, spec, alpha)
        h = spec.G[binding]
        scores, _ = score_vector(alpha, meta, S_out, h)
        return meta[int(np.argmax(scores))].key


class RandomSelector(Selector):
    name = "random"

    def __init__(self, seed: int = 47):
        self._rng = np.random.default_rng(seed)

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=LPSolver.DEFAULT):
        if not meta:
            return None
        return meta[int(self._rng.integers(len(meta)))].key


class BoundWidthSelector(Selector):
    name = "bound_width"

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=LPSolver.DEFAULT):
        if not meta:
            return None
        widths = np.array([nm.u - nm.l for nm in meta])
        return meta[int(np.argmax(widths))].key
