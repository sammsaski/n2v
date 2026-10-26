"""
Split-selection strategies for ``verify_refine``.

Every selector returns the ``NeuronKey`` of the relaxed neuron to split next.

  * ``FaithfulSelector`` (the default) scores each relaxed neuron by
    ``eps * influence`` at the constraint-faithful witness (minimizer over the
    full predicate polytope P) -- it reuses the witness the branch and bound
    already solved for the counterexample check, so it adds no extra LP.
  * ``LayerOrderSelector`` picks the shallowest relaxed neuron first. It is
    for low-input-dimension networks, usually with ``contract_inputs``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.refine.types import LinearSpec, NeuronKey, NeuronMeta, Witness
from n2v.refine.witness import score_vector


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


class LayerOrderSelector(Selector):
    """
    Layer-ordered exhaustive branching: always split the shallowest remaining
    relaxed neuron (ties by neuron index) -- the branch order nnenum uses, as a
    plain Selector.

    Rationale: deciding every unstable neuron of layer ``l`` before touching
    layer ``l+1`` makes the node's prefix *exact* -- a fixed neuron carries no
    relaxation, so once a layer is fully fixed it contributes zero error, which
    tightens every deeper bound, stabilises more neurons, and shrinks the
    remaining tree (self-extinguishing). Only sane on low-input-dimension
    networks, where the number of *realizable* activation patterns is capped by
    input geometry (hyperplane-arrangement bound), not the 2^k syntactic count;
    in high dimension this order is catastrophic -- keep FaithfulSelector there.
    Uses no witness and solves no LP.
    """

    name = "layer_order"

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=LPSolver.DEFAULT):
        if not meta:
            return None
        return min(meta, key=lambda nm: (nm.key.layer, nm.key.neuron)).key
