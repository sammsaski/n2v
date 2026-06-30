"""
Thin branch-and-refine driver.

The two refinement *algorithms* live as set-operations in
``n2v.refine.operations`` (``refine`` = bound tightening, ``split`` =
witness-guided activation split); the *verdict* is decided by the project's
canonical ``n2v.utils.verify_specification.verify_specification`` (the faithful
LP-over-P disjointness test). This driver just runs the DFS worklist that
composes them:

    pop a star
      -> verify_specification UNSAT?  prune (this branch is safe)
      -> else solve the faithful witness (for the CE test + the split choice)
      -> exact forward pass on the witness lands in the unsafe region?  SAT
      -> else split the star and push the children

Soundness: a branch is pruned only when ``verify_specification`` certifies the
(sound over-approximate) output star disjoint from the unsafe region. SAT is
returned only on an *exact* forward pass landing in the unsafe region. Splitting
is exhaustive down to fully-fixed (exact) stars, so emptying the worklist with no
SAT is a sound UNSAT -- unless some branch was neither proven safe nor refinable
(an inconclusive exact star, or a selector that could not choose), in which case
the verdict degrades to UNKNOWN rather than a possibly-false UNSAT.

Only the *choice of which neuron to split* (the selector) varies across
experiments; the prune/CE logic is identical for every selector.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from n2v.sets import HalfSpace
from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.utils.verify_specification import verify_specification
from n2v.refine.operations import split
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.selectors import FaithfulSelector, Selector
from n2v.refine.types import LinearSpec, RefineResult, Status
from n2v.refine.witness import (
    is_true_counterexample,
    make_witness,
    violation_lp,
)

logger = logging.getLogger(__name__)


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
    time_budget: Optional[float] = None,
    lp_solver=LPSolver.DEFAULT,
    incremental: bool = True,
) -> RefineResult:
    """
    Refine ``input_star`` through ``model`` until the property ``G y <= g`` is
    decided (SAT/UNSAT) or the ``node_budget`` / ``time_budget`` is exhausted
    (UNKNOWN).

    Args:
        input_star: input set (a Star over the flattened input).
        model: torch FC ReLU module (used for the exact counterexample check).
        spec: unsafe region ``{y : G y <= g}``.
        selector: split-selection strategy (default: FaithfulSelector).
        layers: optional pre-extracted layer list (avoids re-extracting).
        node_budget: max nodes expanded before returning UNKNOWN.
        bound_mode: neuron-range mode for the reach, "box" | "lp_cpu" | "lp_gpu";
            "lp_cpu"/"lp_gpu" apply LP-over-P bound tightening (the ``refine``
            operation, here applied globally to every node).
        collect_deltas: if True, record per spurious node the divergence
            ``Delta = t_faithful - t_box`` and the node depth (kill-experiment
            diagnostics; costs one extra box LP per spurious node).
        time_budget: optional wall-clock limit (seconds); returns UNKNOWN when
            exceeded (checked between nodes).
        lp_solver: LP backend.

    Returns:
        RefineResult.
    """
    if selector is None:
        selector = FaithfulSelector()
    if layers is None:
        layers = extract_layers(model)

    spec_hs = HalfSpace(spec.G, spec.g)

    # The one loose star, tightened up-front when a bound-tightening mode is
    # requested (global tightening == constructing the reach at that bound_mode;
    # ``refine`` is the per-star form of the same operation).
    root, _ = relaxed_reach(input_star, layers, {}, bound_mode=bound_mode)

    worklist: List[Star] = [root]
    nodes = 0
    max_depth = 0
    inconclusive = False  # a branch was neither proven safe nor refined -> not UNSAT
    deltas: List[float] = []
    delta_depths: List[int] = []
    start = time.perf_counter()

    while worklist:
        timed_out = time_budget is not None and time.perf_counter() - start > time_budget
        if nodes >= node_budget or timed_out:
            return RefineResult(
                Status.UNKNOWN, nodes, max_depth,
                deltas=deltas, delta_depths=delta_depths,
            )
        star = worklist.pop()
        nodes += 1
        depth = len(star.fixed or {})
        max_depth = max(max_depth, depth)

        # --- shared prune / UNSAT test: the canonical faithful disjointness check ---
        if verify_specification([star], spec_hs).verdict == "UNSAT":
            continue  # provably safe over this branch -> prune

        # --- faithful witness (over P): drives the CE test and the split choice ---
        res = violation_lp(star, spec, include_Cd=True, lp_solver=lp_solver)
        if res is None:
            continue  # empty predicate polytope -> vacuously safe
        alpha_f, t_f = res

        # --- shared real-counterexample check ---
        is_ce, x_in, _ = is_true_counterexample(alpha_f, input_star, model, spec)
        if is_ce:
            return RefineResult(
                Status.SAT, nodes, max_depth, counterexample_x=x_in,
                deltas=deltas, delta_depths=delta_depths,
            )

        meta = star.relax_meta or []
        if not meta:
            # Exact star (no relaxed neuron) feasible-in-unsafe yet not a real CE:
            # a numerical corner case Theorem 1 rules out in exact arithmetic. Do
            # NOT prune it (that could hide a real CE) -- mark the search
            # inconclusive and keep exploring. The final verdict then degrades to
            # UNKNOWN rather than a possibly-false UNSAT, while still able to
            # return SAT if another branch yields a real CE.
            logger.warning("refine: feasible exact star without a real CE; inconclusive")
            inconclusive = True
            continue

        # --- diagnostics: box-vs-polytope divergence at this spurious node ---
        if collect_deltas:
            box = violation_lp(star, spec, include_Cd=False, lp_solver=lp_solver)
            if box is not None:
                deltas.append(t_f - box[1])
                delta_depths.append(depth)

        if nodes >= node_budget:
            # No budget left to expand children -> cannot refine here. Stay sound.
            inconclusive = True
            continue

        # --- split selection (THE experimental variable) ---
        wit_f = make_witness(star, spec, alpha_f, t_f, "faithful")
        children = split(star, input_star, layers, spec, selector, wit_f,
                         lp_solver=lp_solver, incremental=incremental)
        if children is None:
            # Nothing to split / selector returned an absent key: stay sound.
            inconclusive = True
            continue

        # Children are witness-phase-first (children[0] = witness phase); push so
        # that child pops first (affects SAT fail-fast only; UNSAT node counts are
        # order-independent).
        for child in reversed(children):
            worklist.append(child)

    # Worklist emptied: UNSAT only if every branch was provably pruned; if any
    # branch was inconclusive, the sound verdict is UNKNOWN (never false UNSAT).
    final = Status.UNKNOWN if inconclusive else Status.UNSAT
    return RefineResult(
        final, nodes, max_depth,
        deltas=deltas, delta_depths=delta_depths,
    )
