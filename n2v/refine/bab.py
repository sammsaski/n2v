"""
Branch and bound over ReLU activation splits on triangle-relaxed stars.

``verify_refine`` runs a depth-first search whose worklist is popped in waves
of ``frontier_width`` nodes (width 1 is plain DFS). Per node:

    pop a star
      -> certified disjoint from the unsafe region?  prune (this branch is safe)
      -> else solve a witness (for the CE test + the split choice)
      -> exact forward pass on the witness lands in the unsafe region?  SAT
      -> else split one relaxed neuron and push the two children

The prune is the exact LP disjointness check
(``n2v.utils.verify_specification.verify_specification``) or its NS-certified
GPU replacement; the split is ``n2v.refine.operations.split`` (or
``split_wave`` under certified input contraction), guided by the selector.

Soundness: a branch is pruned only when the (sound over-approximate) output
star is certified disjoint from the unsafe region. SAT is returned only on an
*exact* forward pass landing in the unsafe region. Splitting is exhaustive down
to fully-fixed (exact) stars, so emptying the worklist with no SAT is a sound
UNSAT -- unless some branch was neither proven safe nor refinable (an
inconclusive exact star, a failed LP, or a selector that could not choose), in
which case the verdict degrades to UNKNOWN rather than a possibly-false UNSAT.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import List, Optional, Sequence, Union

from n2v.sets import HalfSpace
from n2v.sets.predicate_box import ensure_finite_predicate_box
from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver, resolve as _resolve_lp
from n2v.utils.lpsolver import LPFailure
from n2v.utils.lpsolver_gpu import gpu_available
from n2v.utils.verify_specification import verify_specification
from n2v.refine.operations import split, split_wave
from n2v.refine.reach_relaxed import BOUND_MODES, extract_layers, relaxed_reach
from n2v.refine.selectors import FaithfulSelector, Selector
from n2v.refine.types import LinearSpec, RefineResult, Status
from n2v.refine.witness import (certified_prune_wave, is_true_counterexample,
                                make_witness, violation_lp,
                                violation_witness_wave)

logger = logging.getLogger(__name__)

# Gradient-iteration budget for the certified group prune. Internal constant,
# not a tuning surface: 150 iterations produced trees identical to 300 on the
# ACAS controls at half the launch-bound wall-clock (A/B 2026-07-09), and ~100
# iterations recovered 93-99% of the box->exact gap in general measurements. The
# prune is sound at ANY budget (NS-certified per iterate); the budget only
# trades wall-clock against prunes-per-node.
PRUNE_GPU_ITERS = 150

# Default configuration when a CUDA device is usable: NS-certified GPU neuron
# bounds, certified GPU prune and GPU wave witnesses over waves of 32 nodes
# (the configuration the research measured). Without CUDA the defaults fall
# back to the exact CPU checks on single nodes.
_GPU_DEFAULTS = {"bound_mode": "dual_gpu", "prune_mode": "dual_gpu",
                 "witness_mode": "dual_gpu", "frontier_width": 32}
_CPU_DEFAULTS = {"bound_mode": "box", "prune_mode": "exact",
                 "witness_mode": "exact", "frontier_width": 1}


def _resolve_defaults(bound_mode, prune_mode, witness_mode, frontier_width,
                      cuda: bool):
    """Fill every knob left as ``None`` from the GPU configuration when
    ``cuda`` is True, else from the CPU configuration. Explicit values always
    win, each knob independently."""
    base = _GPU_DEFAULTS if cuda else _CPU_DEFAULTS
    return (base["bound_mode"] if bound_mode is None else bound_mode,
            base["prune_mode"] if prune_mode is None else prune_mode,
            base["witness_mode"] if witness_mode is None else witness_mode,
            base["frontier_width"] if frontier_width is None else frontier_width)


def _most_violated(candidates):
    """The disjunct whose witness lies deepest inside its unsafe region: the
    smallest worst-row margin ``t`` (``t <= 0`` means the node's relaxation
    reaches that disjunct). ``candidates`` is a non-empty list of
    ``(spec, (alpha, t))``; returns the chosen pair."""
    return min(candidates, key=lambda c: c[1][1])


def _require_float64_model(model, who):
    """Raise ``TypeError`` unless every floating-point parameter and buffer of
    ``model`` is float64. The counterexample checks feed float64 inputs to
    ``model``, so a float32 model would fail at the first SAT candidate; and
    a silent float64 copy would verify a different function from the one the
    caller ships. A plain callable is not inspected."""
    import torch
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        return
    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_floating_point() and t.dtype != torch.float64:
            raise TypeError(
                f"{who}: model must be float64 (call model.double()); found "
                f"a {t.dtype} parameter or buffer")


def _eval_mode_model(model):
    """The model the counterexample checks run: ``model`` itself when no
    submodule is in training mode, else an eval-mode deep copy. The relaxed
    layers are always the eval-mode function (Dropout is the identity,
    BatchNorm uses its running statistics) and a SAT witness is replayed in
    eval mode, so a training-mode forward pass (random Dropout masks, BatchNorm
    batch statistics that also overwrite the caller's running buffers) must
    never decide SAT. The caller's model is never modified."""
    import torch.nn as nn

    if isinstance(model, nn.Module) and any(m.training
                                            for m in model.modules()):
        return copy.deepcopy(model).eval()
    return model


def _finite_input_star(input_star, lp_solver, who):
    """Finitize the input predicate box (interval bounds are undefined on
    infinite intervals: ``0 * inf = NaN`` reads as "stably active").

    Returns ``(star, None)`` to proceed with the finitized star, or
    ``(input_star, status)`` when the input set alone decides the run:
    UNSAT for a CONFIRMED-empty input set (it reaches nothing), UNKNOWN when
    the finitizer reported empty but the confirmation did not agree, and
    UNKNOWN when the finitizer raises ``ValueError`` (no trustworthy finite
    box exists: an unconfirmed 'infeasible' report, bound LPs that contradict
    each other, an unbounded or failed bound LP, or NaN in the star)."""
    try:
        star, empty = ensure_finite_predicate_box(input_star,
                                                  lp_solver=lp_solver)
    except ValueError as exc:
        logger.warning("%s: the input star's predicate box could not be "
                       "finitized (%s) -- UNKNOWN", who, exc)
        return input_star, Status.UNKNOWN
    if not empty:
        return star, None
    # ensure_finite_predicate_box acts on the default solve's 'infeasible',
    # which is not a certificate on tiny-scale systems (a presolve artefact);
    # the vacuous UNSAT needs the confirmed emptiness check
    # (check_feasibility -> confirm_infeasible).
    try:
        confirmed = input_star.is_empty_set(lp_solver=lp_solver)
    except LPFailure:
        confirmed = False
    if confirmed:
        logger.warning("%s: input star's predicate polytope is infeasible -- "
                       "vacuous UNSAT", who)
        return input_star, Status.UNSAT
    logger.warning("%s: the predicate-box LP reported an empty input set but "
                   "emptiness could not be confirmed -- UNKNOWN", who)
    return input_star, Status.UNKNOWN


def verify_refine(
    input_star: Star,
    model,
    spec: Union[LinearSpec, Sequence[LinearSpec]],
    selector: Optional[Selector] = None,
    *,
    layers=None,
    node_budget: int = 20000,
    time_budget: Optional[float] = None,
    lp_solver=LPSolver.DEFAULT,
    bound_mode: Optional[str] = None,
    prune_mode: Optional[str] = None,
    witness_mode: Optional[str] = None,
    frontier_width: Optional[int] = None,
    incremental: bool = True,
    contract_inputs: bool = False,
    gpu_iters: Optional[int] = None,
    warm_starts: bool = True,
) -> RefineResult:
    """
    Decide whether ``model`` maps ``input_star`` into the unsafe region
    ``{y : G y <= g}``: SAT means unsafe, with a validated counterexample;
    UNSAT means safe; UNKNOWN means undecided (a budget ran out, or some
    branch could be neither decided nor refined).

    Soundness: a node is pruned only when every live disjunct is certified
    disjoint from its output star, by the exact LP (``verify_specification``)
    or by the NS-certified GPU bound. SAT requires an exact float64 forward
    pass into the closed unsafe region at a zero-tolerance member of the input
    set. Any node that is neither decided nor refinable degrades the verdict
    to UNKNOWN.

    Configuration: ``bound_mode``, ``prune_mode``, ``witness_mode`` and
    ``frontier_width`` default to ``None``, and each ``None`` resolves
    independently at entry. When a CUDA device is usable the defaults are
    the GPU configuration ("dual_gpu", "dual_gpu", "dual_gpu", 32): certified
    GPU neuron bounds, certified GPU prune and GPU wave witnesses. Otherwise
    they are the CPU configuration ("box", "exact", "exact", 1): exact CPU
    checks on single nodes. Explicit values always win. A GPU mode requested
    on a host without CUDA runs its gradient engine on CPU torch: sound, but
    slow.

    Args:
        input_star: input set (a Star over the flattened input). A non-finite
            predicate box is finitized first (exact-LP bounds over the
            predicate polytope). An input set reported empty returns UNSAT
            only when ``Star.is_empty_set`` confirms the emptiness, and
            UNKNOWN otherwise; with highspy installed that includes every
            genuinely empty star with an infinite box entry, because the
            confirmation never certifies such a system (see
            ``_finite_input_star``). An input set for which no trustworthy
            finite box exists (the finitizer raises) returns UNKNOWN.
        model: a float64 ``nn.Sequential`` chain of the layer types
            ``extract_layers`` accepts, taking a flat ``(1, n)`` input. It is
            used for the exact counterexample check; a model with a
            non-float64 floating-point parameter or buffer raises
            ``TypeError`` (call ``model.double()``). Custom modules must pass
            ``layers=`` explicitly and are then responsible for its fidelity.
            For conv nets, either use a flat-input ``Sequential`` that
            reshapes with ``nn.Unflatten`` before its first conv, or pass a
            flat-input wrapper as ``model`` together with
            ``layers=extract_layers(inner_seq, in_shape=(C, H, W))``, where
            ``inner_seq`` is the image-input ``Sequential`` the wrapper
            calls. The counterexample checks run in eval mode: a model with a
            submodule in training mode is deep-copied and switched to eval
            mode first, and the caller's model is never modified.
        spec: unsafe region ``{y : G y <= g}``, or a LIST of such specs
            forming a DISJUNCTION (robustness properties are 9-way
            disjunctions). One tree serves all disjuncts: a node is pruned
            when EVERY disjunct is certified unreachable, SAT accepts a
            validated hit on ANY disjunct, and witness/split guidance follows
            the node's most-violated disjunct (smallest worst-row margin). A
            list is not combinable with ``contract_inputs``.
        selector: split-selection strategy (default: FaithfulSelector).
        layers: optional pre-extracted layer list (avoids re-extracting).
        node_budget: max nodes expanded before returning UNKNOWN.
        time_budget: optional wall-clock limit (seconds); returns UNKNOWN when
            exceeded (checked between nodes).
        lp_solver: LP backend (an ``LPSolver``, a solver name, or ``None``
            for the global config); an unknown name raises ``ValueError`` at
            entry, before the input box is finitized.
        bound_mode: neuron-range mode for the reach: "box" (interval bounds
            from the predicate box), "lp_cpu" (exact LP bounds over the
            predicate polytope, padded outward) or "dual_gpu" (NS-certified
            gradient-dual LP bounds). ``None``: see Configuration.
        prune_mode: "exact" -- the canonical ``verify_specification`` LP
            disjointness check per node; "dual_gpu" -- replace it with the
            NS-certified GPU margin bound (``certified_prune_wave``): sound
            prunes at a fraction of the CPU-LP cost, may prune fewer nodes
            (bound tightness; multi-row specs are checked row-wise). Verdicts
            are unaffected either way -- only node count / wall-clock.
            ``None``: see Configuration.
        witness_mode: "exact" -- the exact epigraph ``violation_lp`` per
            surviving node; "dual_gpu" -- approximate witnesses for the
            WHOLE wave in one population PDHG call
            (``violation_witness_wave``): dual-informed primal points, no CPU
            LP, launch cost paid once per wave. SAT stays sound in both modes
            (every candidate is re-validated by an exact forward pass + exact
            input membership). The "dual_gpu" witness cannot detect an empty
            predicate polytope; under ``contract_inputs`` certified-empty
            children are dropped at split time instead, and the exact-leaf
            backstop covers the rest. ``None``: see Configuration.
        frontier_width: nodes popped per wave (an int >= 1; 1 = plain DFS).
            Under ``prune_mode="dual_gpu"`` the wave prune is ONE population
            GPU call, amortising kernel launches across the frontier; under
            exact pruning the wave is checked per node (width still honoured,
            no batching benefit). Changes exploration order (wave-DFS) and
            hence possibly node counts -- never verdicts. ``None``: see
            Configuration.
        incremental: build each child by resuming the reach from the parent's
            shared-prefix checkpoint at the split layer (default True) instead
            of re-reaching from the input. Both paths are sound and give the
            same verdict. Under "box" and "lp_cpu" the child is the same star;
            under "dual_gpu" its bounds can differ (the resume warm-starts the
            parent's duals and uses the resume iteration budget). Only cost
            and node counts change.
        contract_inputs: certified input-domain contraction at every split
            (see ``operations.split_wave``), meant for low-input-dimension
            networks, usually with ``LayerOrderSelector``: each child's input
            box is tightened to the NS-certified range of the input dims over
            the parent polytope + the child's phase rows before its reach, and
            a certified-empty child is dropped for free. Sound either way.
            Forces from-scratch child reaches (``incremental`` is ignored);
            with ``bound_mode="dual_gpu"`` the whole wave's children reach in
            lockstep with one population bounds call per layer
            (``operations._reach_children_wave``).
        gpu_iters: dual-iteration budget per bound for the ROOT reach
            (dual_gpu only; None = the tighten default, 300). Child reaches use
            ``max(1, gpu_iters // 2)``, preserving the default 300:150
            root:resume ratio. Any budget is sound (every NS-certified iterate
            is a valid bound); this knob trades bound tightness for time.
        warm_starts: inherit parent duals on child reaches (default True).
            False drops dual inheritance only -- checkpoint/prefix reuse is
            ``incremental``'s knob. Sound either way.

    Returns:
        RefineResult.
    """
    if selector is None:
        selector = FaithfulSelector()
    bound_mode, prune_mode, witness_mode, frontier_width = _resolve_defaults(
        bound_mode, prune_mode, witness_mode, frontier_width, gpu_available())
    # Fail fast on unknown mode strings: the comparison sites treat unknown
    # values inconsistently (most read != "dual_gpu" as exact, the exact-leaf
    # backstop reads != "exact" as approximate), so a typo would silently run
    # a hybrid configuration nobody specified.
    if bound_mode not in BOUND_MODES:
        raise ValueError(f"unknown bound_mode {bound_mode!r} "
                         f"(expected one of {BOUND_MODES})")
    if prune_mode not in ("exact", "dual_gpu"):
        raise ValueError(f"unknown prune_mode {prune_mode!r} "
                         f"(expected 'exact' or 'dual_gpu')")
    if witness_mode not in ("exact", "dual_gpu"):
        raise ValueError(f"unknown witness_mode {witness_mode!r} "
                         f"(expected 'exact' or 'dual_gpu')")
    if not isinstance(frontier_width, int) or frontier_width < 1:
        raise ValueError(f"frontier_width must be an int >= 1 "
                         f"(got {frontier_width!r})")
    _require_float64_model(model, "verify_refine")
    model = _eval_mode_model(model)
    if layers is None:
        layers = extract_layers(model)
    # Argument checks come before the finitizer: it turns every ValueError
    # into UNKNOWN, so a caller mistake reported only from inside it (a bad
    # solver name, a malformed spec) would be hidden on infinite-box inputs.
    # The sentinel is kept, so the global config is still read per solve.
    lp_solver = _resolve_lp(lp_solver)
    if isinstance(spec, LinearSpec):
        specs: List[LinearSpec] = [spec]
    else:
        specs = list(spec)
        if not specs or not all(isinstance(s, LinearSpec) for s in specs):
            raise TypeError("spec must be a LinearSpec or a non-empty list "
                            "of LinearSpec disjuncts")
        spec = specs[0]
    if len(specs) > 1 and contract_inputs:
        raise TypeError("multi-disjunct specs are not yet supported with "
                        "contract_inputs")

    # Interval bounds are undefined on infinite predicate intervals (0 * inf =
    # NaN reads as "stably active"): finitize the input box first. An
    # unconfirmed emptiness report gives UNKNOWN, never a vacuous UNSAT.
    input_star, decided = _finite_input_star(input_star, lp_solver,
                                             "verify_refine")
    if decided is not None:
        return RefineResult(decided, 0, 0)

    specs_hs = [HalfSpace(s.G, s.g) for s in specs]

    root, _ = relaxed_reach(input_star, layers, {}, bound_mode=bound_mode,
                            gpu_iters=gpu_iters)
    # Per-node search state: the live-disjunct mask. Seeded on the root here;
    # every child copies its parent's mask when it is pushed.
    root._live_specs = list(range(len(specs)))

    # Child-reach budget derived from the root budget at the same 2:1 ratio the
    # defaults use (300 root / 150 resume); None keeps every current default.
    _resume_iters = None if gpu_iters is None else max(1, gpu_iters // 2)

    worklist: List[Star] = [root]
    nodes = 0
    max_depth = 0
    inconclusive = False  # a branch was neither proven safe nor refined -> not UNSAT
    start = time.perf_counter()

    def _unknown():
        return RefineResult(Status.UNKNOWN, nodes, max_depth)

    def _over_deadline():
        return time_budget is not None and time.perf_counter() - start > time_budget

    while worklist:
        if nodes >= node_budget or _over_deadline():
            return _unknown()

        # --- pop a wave (frontier_width nodes; width 1 == plain DFS) and run
        # the shared prune for the whole wave (one population GPU call under
        # prune_mode="dual_gpu"; per-node exact LPs otherwise) ---
        wave = [worklist.pop()
                for _ in range(min(frontier_width, len(worklist)))]
        deferred = []  # (star, witness) splits awaiting the wave-batched contraction
        # Disjunct liveness (sound subtree exclusion): a disjunct certified
        # disjoint from a node's over-approximation is disjoint from the TRUE
        # reach of every descendant (child region subset of parent region), so
        # children inherit the parent's live mask and neither prune nor
        # witness dead disjuncts again. Deep nodes typically carry one live
        # disjunct -- this converts the naive D-fold multi-disjunct cost into
        # ~1x. A node retires when its live set empties.
        live = [list(getattr(s, "_live_specs", range(len(specs)))) for s in wave]
        if prune_mode == "dual_gpu":
            for d in range(len(specs)):
                idx = [i for i in range(len(wave)) if d in live[i]]
                if not idx:
                    continue
                fl = certified_prune_wave([wave[i] for i in idx], specs[d],
                                          max_iters=PRUNE_GPU_ITERS)
                for i, f in zip(idx, fl):
                    if f:
                        live[i] = [x for x in live[i] if x != d]
        else:
            for i, s in enumerate(wave):
                # verify_specification already degrades a failed LP to UNKNOWN
                live[i] = [d for d in live[i]
                           if verify_specification([s], specs_hs[d]).verdict
                           != "UNSAT"]
        for s, lv in zip(wave, live):
            s._live_specs = lv
        pruned_flags = [not lv for lv in live]
        surv_idx = [i for i, p in enumerate(pruned_flags) if not p]

        # Deadline check between the wave prune and the (heuristic) witness
        # wave: the witness only proposes candidates and informs split scores,
        # so it is safe to skip, and without this check a wide wave overshoots
        # the deadline by one population PDHG call.
        if _over_deadline():
            return _unknown()

        # --- batched witnesses: every surviving branch's epigraph LP solved
        # in ONE population PDHG call (the per-branch form was ~67ms/branch,
        # all launch overhead) ---
        wave_wit: List = [None] * len(wave)
        wave_wit_spec: List = [None] * len(wave)
        wit_failed = [False] * len(wave)
        if witness_mode == "dual_gpu" and surv_idx:
            wave_cands = [[] for _ in wave]
            for d, sp in enumerate(specs):
                idx = [i for i in surv_idx if d in live[i]]
                if not idx:
                    continue
                try:
                    wres = violation_witness_wave([wave[i] for i in idx], sp,
                                                  lp_solver=lp_solver)
                except LPFailure:
                    # The exact-LP fallback inside the wave (degenerate stars)
                    # did not decide. Not a certificate of anything: these
                    # nodes stay undecided (verdict degrades to UNKNOWN).
                    logger.warning("refine: wave witness LP failed to decide; "
                                   "its nodes are kept undecided")
                    for i in idx:
                        wit_failed[i] = True
                    continue
                for i, r in zip(idx, wres):
                    if r is not None:
                        wave_cands[i].append((sp, r))
            for i in surv_idx:
                if wave_cands[i]:
                    wave_wit_spec[i], wave_wit[i] = _most_violated(wave_cands[i])
        if _over_deadline():
            return _unknown()

        for i, (star, pruned) in enumerate(zip(wave, pruned_flags)):
            # per-star budget checks: without these a wide wave overshoots the
            # node budget / deadline by up to frontier_width-1 witness costs
            if _over_deadline():
                return _unknown()
            if nodes >= node_budget:
                # Requeue the unexpanded survivors of this wave AND the
                # deferred split parents; clearing ``deferred`` also stops the
                # post-loop split_wave from burning unbounded work (a full
                # batched contraction + child reaches) after the budget is
                # already exhausted.
                worklist.extend(s for j, s in enumerate(wave)
                                if j >= i and not pruned_flags[j])
                worklist.extend(s for s, _ in deferred)
                deferred.clear()
                break  # outer loop head returns UNKNOWN
            nodes += 1
            depth = len(star.fixed or {})
            max_depth = max(max_depth, depth)

            # --- shared prune / UNSAT test: the canonical faithful disjointness
            # check (or its NS-certified GPU replacement -- sound, fewer prunes) ---
            if pruned:
                continue  # provably safe over this branch -> prune

            # --- faithful witness (over P): drives the CE test and split choice.
            # Multi-disjunct: guide with the node's MOST-VIOLATED disjunct. ---
            if witness_mode == "dual_gpu":
                if wit_failed[i]:
                    inconclusive = True
                    continue
                res, guide = wave_wit[i], (wave_wit_spec[i] or spec)
            else:
                cands, lp_failed = [], False
                for d in star._live_specs:
                    try:
                        r = violation_lp(star, specs[d], include_Cd=True,
                                         lp_solver=lp_solver)
                    except LPFailure:
                        # Not a certificate of emptiness: retiring this star
                        # as "vacuously safe" below would be a false UNSAT.
                        lp_failed = True
                        break
                    if r is not None:
                        cands.append((specs[d], r))
                if lp_failed:
                    logger.warning(
                        "refine: witness LP failed to decide; branch kept "
                        "undecided (verdict degrades to UNKNOWN)")
                    inconclusive = True
                    continue
                res, guide = None, spec
                if cands:
                    guide, res = _most_violated(cands)
            if res is None:
                continue  # empty predicate polytope -> vacuously safe
            alpha_f, t_f = res

            # --- real-counterexample check, for both witness modes: exact
            # float64 forward pass plus zero-tolerance input membership ---
            for sp in specs:
                is_ce, x_in, _ = is_true_counterexample(alpha_f, input_star,
                                                        model, sp)
                if is_ce:
                    return RefineResult(Status.SAT, nodes, max_depth,
                                        counterexample_x=x_in)

            meta = star.relax_meta or []
            if not meta:
                # Exact star (no relaxed neuron): exact checks are decisive and
                # this case is rare, so BACKSTOP the approximate GPU prune /
                # witness with the exact machinery before poisoning the verdict.
                # This restores prune_mode/witness_mode verdict-invariance: a
                # disjoint or empty exact leaf prunes (as the exact path would),
                # a genuinely feasible one is re-witnessed exactly.
                if prune_mode != "exact" or witness_mode != "exact":
                    lv = list(star._live_specs)
                    if all(verify_specification([star], specs_hs[d]).verdict
                           == "UNSAT" for d in lv):
                        continue
                    leaf_feasible, exact_lp_failed = False, False
                    for sp in (specs[d] for d in lv):
                        try:
                            r = violation_lp(star, sp, include_Cd=True,
                                             lp_solver=lp_solver)
                        except LPFailure:
                            exact_lp_failed = True
                            break
                        if r is None:
                            continue
                        leaf_feasible = True
                        is_ce, x_in, _ = is_true_counterexample(
                            r[0], input_star, model, sp)
                        if is_ce:
                            return RefineResult(Status.SAT, nodes, max_depth,
                                                counterexample_x=x_in)
                    if exact_lp_failed:
                        logger.warning(
                            "refine: exact-leaf witness LP failed to decide; "
                            "branch kept undecided")
                        inconclusive = True
                        continue
                    if not leaf_feasible:
                        continue  # empty predicate polytope -> vacuously safe
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

            if nodes >= node_budget:
                # No budget left to expand children -> cannot refine. Stay
                # sound: the verdict can no longer be UNSAT.
                inconclusive = True
                continue

            # --- in-node deadline check before the expensive step (children
            # reach): without this, a single node's split can overshoot the
            # budget by two child re-propagations (minutes at cora scale). ---
            if _over_deadline():
                return _unknown()

            # --- split selection ---
            wit_f = make_witness(star, guide, alpha_f, t_f, "faithful")
            if (contract_inputs and star.predicate_lb is not None
                    and input_star.predicate_lb is not None):
                # Defer: the contraction call is launch-bound, not size-bound,
                # so the whole wave's splits share ONE batched call below.
                # Boxless stars fall through to the plain split -- the same
                # graceful degradation operations.split applies; split_wave's
                # contraction requires finite predicate boxes.
                deferred.append((star, wit_f))
                continue
            children = split(star, input_star, layers, guide, selector, wit_f,
                             lp_solver=lp_solver, incremental=incremental,
                             gpu_iters=_resume_iters, warm_starts=warm_starts)
            if children is None:
                # Nothing to split / selector returned an absent key: stay
                # sound; the verdict can no longer be UNSAT.
                inconclusive = True
                continue

            # Children are witness-phase-first (children[0] = witness phase);
            # push so that child pops first (affects SAT fail-fast only; UNSAT
            # node counts are order-independent). Children inherit the
            # parent's live-disjunct mask (sound: subtree exclusion above).
            for child in reversed(children):
                child._live_specs = list(star._live_specs)
                worklist.append(child)

        if deferred:
            # Wave-batched contract-splits: one population contraction call for
            # every pending split, then per-child reaches (deadline-guarded).
            deadline = None if time_budget is None else start + time_budget
            wave_children = split_wave(deferred, input_star, layers, spec,
                                       selector, lp_solver=lp_solver,
                                       deadline=deadline,
                                       gpu_iters=_resume_iters)
            if _over_deadline():
                return _unknown()
            for (parent, _w), children in zip(deferred, wave_children):
                if children is None:
                    # Nothing to split / selector returned an absent key: stay
                    # sound; the verdict can no longer be UNSAT.
                    inconclusive = True
                    continue
                # The parent is replaced by its surviving children. A
                # certified-empty child was dropped inside split_wave (it
                # covers no points); ``children == []`` means BOTH phase
                # regions are certified empty, i.e. the parent region itself
                # is empty -- so pushing nothing is a sound prune. Children
                # inherit the live mask (sound -- a disjunct disjoint from the
                # parent is disjoint from every child).
                for child in reversed(children):
                    child._live_specs = list(parent._live_specs)
                    worklist.append(child)

    # Worklist emptied: UNSAT only if every branch was provably pruned; if any
    # branch was inconclusive, the sound verdict is UNKNOWN (never false UNSAT).
    if inconclusive:
        return _unknown()
    return RefineResult(Status.UNSAT, nodes, max_depth)
