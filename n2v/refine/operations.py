"""
Refinement set-operations: the two algorithms expressed as operations that take
a star and produce star(s), to be verified by ``verify_specification``.

  * ``refine(star, ...)``  -> a single, tighter star (bound tightening): re-runs
    the reach with an LP-over-P ``bound_mode`` so every triangle relaxation is
    built from tighter neuron ranges.
  * ``split(star, ...)``   -> two child stars (witness-guided branch-and-bound):
    fixes one unstable neuron ACTIVE / INACTIVE, re-running the reach for each.

Both are **full-pass**: they re-run ``relaxed_reach`` from the input with a
modification (a tighter bound mode / one extra phase fix), exactly as the prior
monolithic loop did per node. They read the star's attached ``relax_meta`` (which
relaxed neuron is which predicate variable) and its search provenance
(``fixed``/``bound_mode``); they do not mutate the input star. The network and
input set (``layers``/``input_star``) are passed as the reach environment rather
than stored on the star -- reachability is external to set objects in n2v.

These are functions in ``n2v.refine`` (not methods on the core ``Star``) on
purpose: they re-run reachability, and binding them onto the foundational
``Star`` class would invert the layering (core depending on this experimental
module). The data the star carries (``relax_meta``) is what makes the operations
read as set operations.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.refine.reach_relaxed import Layer, relaxed_reach
from n2v.refine.selectors import Selector
from n2v.refine.types import LinearSpec, NeuronMeta, Phase, Witness
from n2v.refine.witness import preactivation


def witness_phase(alpha: np.ndarray, nm: NeuronMeta) -> Phase:
    """The phase the witness ``alpha`` 'claims' for neuron ``nm`` (descend first).

    ``x_hat >= 0`` -> ACTIVE, matching the ReLU's identity branch on the boundary.
    """
    return Phase.ACTIVE if preactivation(nm, alpha) >= 0.0 else Phase.INACTIVE


def refine(
    star: Star,
    input_star: Star,
    layers: List[Layer],
    bound_mode: str,
) -> Star:
    """
    Bound-tightening refinement: return a tighter star for the same fixed-phase
    region as ``star``, by re-running the reach with LP-over-P ``bound_mode``
    ("lp_cpu" | "lp_gpu"; "box" is the un-tightened baseline).

    The set ``[[result]] == [[star]]`` (same fixed phases); only the
    representation is tighter (tighter triangle relaxations stabilise more
    neurons). The returned star carries its own ``relax_meta``/``fixed``/
    ``bound_mode``.
    """
    fixed = star.fixed or {}
    tightened, _ = relaxed_reach(input_star, layers, fixed, bound_mode=bound_mode)
    return tightened


def split(
    star: Star,
    input_star: Star,
    layers: List[Layer],
    spec: LinearSpec,
    selector: Selector,
    witness: Witness,
    *,
    lp_solver=LPSolver.DEFAULT,
) -> Optional[List[Star]]:
    """
    Witness-guided activation split: choose one unstable (triangle-relaxed) neuron
    via ``selector`` and return the two child stars that fix it ACTIVE / INACTIVE.

    The children are over the same input set with one extra phase constraint each;
    their union covers ``[[star]]``. They are returned witness-phase-first (the
    phase the witness claims is element 0) so a SAT-seeking search can descend it
    first. Returns ``None`` if there is nothing to split (no ``relax_meta``) or
    the selector returns a key absent from this star's metadata.

    ``witness`` is the faithful epigraph witness already solved by the caller (for
    the shared prune / counterexample test) -- reused here so no extra LP is
    spent for the split choice itself (the box-corner selector solves its own).
    """
    meta: List[NeuronMeta] = star.relax_meta or []
    if not meta:
        return None

    key = selector.choose(star, spec, meta, witness, lp_solver)
    if key is None:
        return None
    nm = next((m for m in meta if m.key == key), None)
    if nm is None:
        return None

    fixed = star.fixed or {}
    bound_mode = star.bound_mode or "box"
    wphase = witness_phase(witness.alpha, nm)
    other = Phase.INACTIVE if wphase == Phase.ACTIVE else Phase.ACTIVE

    children: List[Star] = []
    for phase in (wphase, other):
        child_fixed = {**fixed, key: phase}
        child, _ = relaxed_reach(input_star, layers, child_fixed, bound_mode=bound_mode)
        children.append(child)
    return children
