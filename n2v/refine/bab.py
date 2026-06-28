"""
Minimal star activation-split branch-and-bound (Phase 1).

The loop is a DFS worklist over ReLU activation splits. Crucially, the
**prune/UNSAT decision and the real-counterexample check are identical across
every selector** -- only the choice of *which neuron to split* varies. This
isolates split-selection quality as the single experimental variable (the box
arm is even handed the faithful CE check, a conservative bias toward the
baseline).

Soundness sketch: a node is pruned only when its (sound over-approximate) output
star provably does not intersect the unsafe region (faithful epigraph optimum
``t* > PRUNE_TOL``, or an empty predicate polytope). A branch ends in SAT only on
an *exact* forward pass landing in the unsafe region. Splitting is exhaustive
down to fully-fixed (exact) stars, so any true counterexample is eventually
realized -- hence UNSAT (worklist emptied with no SAT) is sound.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.selectors import FaithfulSelector, Selector
from n2v.refine.types import (
    LinearSpec,
    NeuronKey,
    NeuronMeta,
    Phase,
    RefineNode,
    RefineResult,
    Status,
)
from n2v.refine.witness import (
    PRUNE_TOL,
    is_true_counterexample,
    make_witness,
    violation_lp,
)

logger = logging.getLogger(__name__)


def _witness_phase(alpha: np.ndarray, nm: NeuronMeta) -> Phase:
    """The phase the witness 'claims' for neuron ``nm`` (descend it first)."""
    k = len(nm.preact_gens)
    x_hat = nm.preact_center + float(nm.preact_gens @ alpha[:k])
    # x_hat >= 0 -> ACTIVE, matching the ReLU's identity branch on the boundary.
    return Phase.ACTIVE if x_hat >= 0.0 else Phase.INACTIVE


def verify_refine(
    input_star: Star,
    model,
    spec: LinearSpec,
    selector: Optional[Selector] = None,
    *,
    layers=None,
    node_budget: int = 20000,
    collect_deltas: bool = False,
    bound_mode: str = "box",
    lp_solver=LPSolver.DEFAULT,
) -> RefineResult:
    """
    Refine ``input_star`` through ``model`` until the property ``G y <= g`` is
    decided (SAT/UNSAT) or ``node_budget`` is exhausted (UNKNOWN).

    Args:
        input_star: input set (a Star over the flattened input).
        model: torch FC ReLU module (or pre-extracted ``layers``).
        spec: unsafe region.
        selector: split-selection strategy (default: FaithfulSelector).
        layers: optional pre-extracted layer list (avoids re-extracting).
        node_budget: max nodes expanded before returning UNKNOWN.
        collect_deltas: if True, record per spurious node the divergence
            ``Delta = t_faithful - t_box`` and the node depth (kill-experiment
            diagnostics; costs one extra box LP per spurious node).
        lp_solver: LP backend.

    Returns:
        RefineResult.
    """
    if selector is None:
        selector = FaithfulSelector()
    if layers is None:
        layers = extract_layers(model)

    worklist: List[RefineNode] = [RefineNode()]
    nodes = 0
    max_depth = 0
    inconclusive = False  # a branch was neither proven safe nor refined -> not UNSAT
    deltas: List[float] = []
    delta_depths: List[int] = []

    while worklist:
        if nodes >= node_budget:
            return RefineResult(
                Status.UNKNOWN, nodes, max_depth,
                deltas=deltas, delta_depths=delta_depths,
            )
        node = worklist.pop()
        nodes += 1
        max_depth = max(max_depth, node.depth)

        S_out, meta = relaxed_reach(input_star, layers, node.fixed, bound_mode=bound_mode)

        # --- shared prune / UNSAT test (faithful, tight over P) ---
        res = violation_lp(S_out, spec, include_Cd=True, lp_solver=lp_solver)
        if res is None:
            continue  # empty predicate polytope -> vacuously safe
        alpha_f, t_f = res
        if t_f > PRUNE_TOL:
            continue  # provably safe over this branch -> prune

        # --- shared real-counterexample check ---
        is_ce, x_in, _ = is_true_counterexample(alpha_f, input_star, model, spec)
        if is_ce:
            return RefineResult(
                Status.SAT, nodes, max_depth, counterexample_x=x_in,
                deltas=deltas, delta_depths=delta_depths,
            )

        if not meta:
            # Exact star (no relaxed neuron) feasible-in-unsafe yet not a real CE:
            # a numerical corner case Theorem 1 rules out in exact arithmetic. Do
            # NOT prune it (that could hide a real CE) -- mark the whole search
            # inconclusive and keep exploring other branches. The final verdict
            # then degrades to UNKNOWN rather than a possibly-false UNSAT, while
            # still being able to return SAT if another branch yields a real CE.
            logger.warning("refine: feasible exact star without a real CE; inconclusive")
            inconclusive = True
            continue

        # --- diagnostics: box-vs-polytope divergence at this spurious node ---
        if collect_deltas:
            box = violation_lp(S_out, spec, include_Cd=False, lp_solver=lp_solver)
            if box is not None:
                deltas.append(t_f - box[1])
                delta_depths.append(node.depth)

        # --- split selection (THE experimental variable) ---
        wit_f = make_witness(S_out, spec, alpha_f, t_f, "faithful")
        key = selector.choose(S_out, spec, meta, wit_f, lp_solver)
        nm = next((m for m in meta if m.key == key), None)
        if nm is None:
            # Selector returned None or a key absent from this node's meta:
            # cannot refine here. Stay sound -- mark inconclusive and continue.
            inconclusive = True
            continue

        # Witness-phase-first child ordering (affects SAT fail-fast only; UNSAT
        # node counts are order-independent). Same rule for every selector.
        wphase = _witness_phase(alpha_f, nm)
        other = Phase.INACTIVE if wphase == Phase.ACTIVE else Phase.ACTIVE
        worklist.append(node.with_fixed(key, other))   # popped second
        worklist.append(node.with_fixed(key, wphase))  # popped first

    # Worklist emptied: UNSAT only if every branch was provably pruned; if any
    # branch was inconclusive, the sound verdict is UNKNOWN (never false UNSAT).
    final = Status.UNKNOWN if inconclusive else Status.UNSAT
    return RefineResult(
        final, nodes, max_depth,
        deltas=deltas, delta_depths=delta_depths,
    )
