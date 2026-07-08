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
    certified_prune,
    certified_prune_wave,
    is_true_counterexample,
    make_witness,
    violation_lp,
    violation_witness_gpu,
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
    root: Optional[Star] = None,
    prune_mode: str = "exact",
    witness_mode: str = "exact",
    frontier_width: int = 1,
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
        bound_mode: neuron-range mode for the reach, "box" | "lp_cpu" | "lp_gpu"
            | "dual_gpu"; the non-box modes apply LP-over-P bound tightening
            (the ``refine`` operation, here applied globally to every node).
        collect_deltas: if True, record per spurious node the divergence
            ``Delta = t_faithful - t_box`` and the node depth (kill-experiment
            diagnostics; costs one extra box LP per spurious node).
        time_budget: optional wall-clock limit (seconds); returns UNKNOWN when
            exceeded (checked between nodes).
        lp_solver: LP backend.
        root: optional prebuilt root star from ``relaxed_reach(input_star,
            layers, {}, bound_mode=bound_mode)``. The reach is spec-independent,
            so a caller verifying several property disjuncts over the same input
            set can build the root once and share it (it is only read, never
            mutated -- children are fresh stars). Must match ``bound_mode``.
        prune_mode: "exact" (default) -- the canonical ``verify_specification``
            LP disjointness check per node; "dual_gpu" -- replace it with the
            NS-certified GPU margin bound (``certified_prune``): sound prunes at
            a fraction of the CPU-LP cost, may prune fewer nodes (bound
            tightness; multi-row specs are checked row-wise). Verdicts are
            unaffected either way -- only node count / wall-clock.
        witness_mode: "exact" (default) -- the exact epigraph ``violation_lp``
            per surviving node; "dual_gpu" -- the approximate GPU witness
            (``violation_witness_gpu``): dual-informed primal, no CPU LP. SAT
            stays sound (candidates are re-validated by the exact forward
            pass); split-score quality may degrade slightly.
        frontier_width: nodes popped and prune-checked per iteration (1 = plain
            DFS). Under ``prune_mode="dual_gpu"`` a wave's prunes run as ONE
            population GPU call, amortising kernel launches across the
            frontier; under exact pruning the wave is checked per node (width
            still honoured, no batching benefit). Changes exploration order
            (wave-DFS) and hence possibly node counts -- never verdicts.

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
    if root is None:
        root, _ = relaxed_reach(input_star, layers, {}, bound_mode=bound_mode)
    elif root.bound_mode != bound_mode:
        raise ValueError(
            f"prebuilt root was reached with bound_mode={root.bound_mode!r}, "
            f"but verify_refine was called with {bound_mode!r}")

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

        # --- pop a wave (frontier_width nodes; width 1 == plain DFS) and run
        # the shared prune for the whole wave (one population GPU call under
        # prune_mode="dual_gpu"; per-node exact LPs otherwise) ---
        wave = [worklist.pop()
                for _ in range(min(max(frontier_width, 1), len(worklist)))]
        if prune_mode == "dual_gpu":
            pruned_flags = certified_prune_wave(wave, spec)
        else:
            pruned_flags = [verify_specification([s], spec_hs).verdict == "UNSAT"
                            for s in wave]

        for star, pruned in zip(wave, pruned_flags):
            # per-star budget checks: without these a wide wave overshoots the
            # node budget / deadline by up to frontier_width-1 witness+CE costs
            if time_budget is not None and time.perf_counter() - start > time_budget:
                return RefineResult(
                    Status.UNKNOWN, nodes, max_depth,
                    deltas=deltas, delta_depths=delta_depths,
                )
            if nodes >= node_budget:
                break  # outer loop head returns UNKNOWN
            nodes += 1
            depth = len(star.fixed or {})
            max_depth = max(max_depth, depth)

            # --- shared prune / UNSAT test: the canonical faithful disjointness
            # check (or its NS-certified GPU replacement -- sound, fewer prunes) ---
            if pruned:
                continue  # provably safe over this branch -> prune

            # --- faithful witness (over P): drives the CE test and split choice ---
            if witness_mode == "dual_gpu":
                res = violation_witness_gpu(star, spec, lp_solver=lp_solver)
            else:
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
                # Exact star (no relaxed neuron): exact checks are decisive and
                # this case is rare, so BACKSTOP the approximate GPU prune /
                # witness with the exact machinery before poisoning the verdict.
                # This restores prune_mode/witness_mode verdict-invariance: a
                # disjoint or empty exact leaf prunes (as the exact path would),
                # a genuinely feasible one is re-witnessed exactly.
                if prune_mode == "dual_gpu" or witness_mode == "dual_gpu":
                    if verify_specification([star], spec_hs).verdict == "UNSAT":
                        continue
                    exact_res = violation_lp(star, spec, include_Cd=True,
                                             lp_solver=lp_solver)
                    if exact_res is None:
                        continue  # empty predicate polytope -> vacuously safe
                    is_ce, x_in, _ = is_true_counterexample(
                        exact_res[0], input_star, model, spec)
                    if is_ce:
                        return RefineResult(
                            Status.SAT, nodes, max_depth, counterexample_x=x_in,
                            deltas=deltas, delta_depths=delta_depths,
                        )
                # Exact star (no relaxed neuron) feasible-in-unsafe yet not a
                # real CE: a numerical corner case Theorem 1 rules out in exact
                # arithmetic. Do NOT prune it (that could hide a real CE) --
                # mark the search inconclusive and keep exploring. The final
                # verdict then degrades to UNKNOWN rather than a possibly-false
                # UNSAT, while still able to return SAT from another branch.
                logger.warning(
                    "refine: feasible exact star without a real CE; inconclusive")
                inconclusive = True
                continue

            # --- diagnostics: box-vs-polytope divergence at this spurious node ---
            if collect_deltas:
                box = violation_lp(star, spec, include_Cd=False, lp_solver=lp_solver)
                if box is not None:
                    deltas.append(t_f - box[1])
                    delta_depths.append(depth)

            if nodes >= node_budget:
                # No budget left to expand children -> cannot refine. Stay sound.
                inconclusive = True
                continue

            # --- in-node deadline check before the expensive step (children
            # reach): without this, a single node's split can overshoot the
            # budget by two child re-propagations (minutes at cora scale). ---
            if time_budget is not None and time.perf_counter() - start > time_budget:
                return RefineResult(
                    Status.UNKNOWN, nodes, max_depth,
                    deltas=deltas, delta_depths=delta_depths,
                )

            # --- split selection (THE experimental variable) ---
            wit_f = make_witness(star, spec, alpha_f, t_f, "faithful")
            children = split(star, input_star, layers, spec, selector, wit_f,
                             lp_solver=lp_solver, incremental=incremental)
            if children is None:
                # Nothing to split / selector returned an absent key: stay sound.
                inconclusive = True
                continue

            # Children are witness-phase-first (children[0] = witness phase);
            # push so that child pops first (affects SAT fail-fast only; UNSAT
            # node counts are order-independent).
            for child in reversed(children):
                worklist.append(child)

    # Worklist emptied: UNSAT only if every branch was provably pruned; if any
    # branch was inconclusive, the sound verdict is UNKNOWN (never false UNSAT).
    final = Status.UNKNOWN if inconclusive else Status.UNSAT
    return RefineResult(
        final, nodes, max_depth,
        deltas=deltas, delta_depths=delta_depths,
    )
