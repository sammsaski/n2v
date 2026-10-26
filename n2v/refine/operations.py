"""
Refinement set-operations: functions that take a star and produce child stars,
to be verified by the branch and bound in ``n2v.refine.bab``.

  * ``split(star, ...)``      -> two child stars (witness-guided activation
    split): fixes one relaxed neuron ACTIVE / INACTIVE. Each child resumes the
    reach from the parent's shared-prefix checkpoint at the split layer when
    the parent carries checkpoints, and re-reaches from the input otherwise.
  * ``split_wave(jobs, ...)`` -> the children of a whole frontier wave under
    certified input-domain contraction: one batched contraction call for
    every job, then from-scratch child reaches.

They read the star's attached ``relax_meta`` (which relaxed neuron is which
predicate variable) and its search provenance (``fixed``/``bound_mode``); they
do not mutate the input star. The network and input set
(``layers``/``input_star``) are passed as the reach environment rather than
stored on the star -- reachability is external to set objects in n2v.

These are functions in ``n2v.refine`` (not methods on the core ``Star``) on
purpose: they re-run reachability, and binding them onto the foundational
``Star`` class would invert the layering (core depending on this experimental
module). The data the star carries (``relax_meta``) is what makes the operations
read as set operations.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.refine.reach_relaxed import (ConvLayer, Layer, LinearLayer, ReluLayer,
                                      _attach_provenance, _relu_layer_relaxed,
                                      apply_affine, relaxed_reach, resume_reach,
                                      resume_reach_pair)
from n2v.refine.selectors import Selector
from n2v.refine.types import LinearSpec, NeuronMeta, Phase, Witness
from n2v.refine.witness import preactivation


# Gradient-iteration budget for the group reach's per-layer bound solves
# (children are cold-started; in our measurements ~100-150 iterations recover
# 93-99% of the box->exact gap).
WAVE_BOUNDS_GPU_ITERS = 150


def witness_phase(alpha: np.ndarray, nm: NeuronMeta) -> Phase:
    """The phase the witness ``alpha`` 'claims' for neuron ``nm`` (descend first).

    ``x_hat >= 0`` -> ACTIVE, matching the ReLU's identity branch on the boundary.
    """
    return Phase.ACTIVE if preactivation(nm, alpha) >= 0.0 else Phase.INACTIVE


def split(
    star: Star,
    input_star: Star,
    layers: List[Layer],
    spec: LinearSpec,
    selector: Selector,
    witness: Witness,
    *,
    lp_solver=LPSolver.DEFAULT,
    incremental: bool = True,
    gpu_iters: Optional[int] = None,
    warm_starts: bool = True,
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
    spent for the split choice itself.

    ``incremental`` (default True) builds each child by ``resume_reach`` from the
    parent's shared-prefix checkpoint at the split layer -- reprocessing only
    layers at/after the split, reusing the parent's affine maps and OBBT bound LPs
    below it. Falls back to a full ``relaxed_reach`` if the parent carries no
    checkpoints. Under box and lp_cpu bounds the two paths give the same child
    (see test_incremental_reach); under dual_gpu the child's bounds can differ
    (warm-started parent duals, resume iteration budget). Both are sound, so
    this changes cost and node counts, never the verdict.

    For splits with certified input-domain contraction, use ``split_wave``
    (which batches the contraction GPU call across a whole frontier wave; a
    single-job wave ``[(star, witness)]`` is the one-node case).

    ``gpu_iters``: child-reach dual-iteration budget (dual_gpu only); ``None``
    keeps the current defaults (``RESUME_GPU_ITERS`` on the incremental path,
    the tighten default on the full-reach fallback). ``warm_starts=False``
    drops parent-dual inheritance on the incremental path (ablation knob;
    checkpoint reuse is ``incremental``'s job). Both are sound at any setting.
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

    use_incremental = incremental and star.checkpoints is not None

    fixed_pair = [{**fixed, key: phase} for phase in (wphase, other)]

    if use_incremental and bound_mode == "dual_gpu":
        # lockstep siblings: both children's per-layer OBBT batches merge into
        # one population GPU call (sound; verdict-equivalent, see the docstring)
        return resume_reach_pair(star, layers, fixed_pair, bound_mode, key.layer,
                                 gpu_iters=gpu_iters, warm_starts=warm_starts)
    children: List[Star] = []
    for child_fixed in fixed_pair:
        if use_incremental:
            child = resume_reach(star, layers, child_fixed, bound_mode, key.layer,
                                 gpu_iters=gpu_iters, warm_starts=warm_starts)
        else:
            child, _ = relaxed_reach(input_star, layers, child_fixed,
                                     bound_mode=bound_mode, gpu_iters=gpu_iters)
        children.append(child)
    return children


def _reach_children_wave(entries, layers, gpu_iters=None, deadline=None):
    """
    Group reach for contract-split children: every child in the popped group
    is propagated layer-by-layer together, and at each ReLU layer ALL
    children's neuron bounds are computed by the certified gradient-dual solve
    over each child's full constraint polytope in ONE population GPU call
    (``neuron_bounds_population``: the cheap predicate-box interval classifies
    what it can for free, the dual solve covers only the dims the box leaves
    crossing, results are box-intersected and NS-certified outward).

    This is the same bound discipline as ``bound_mode="dual_gpu"`` on a single
    star -- just batched across the group so the per-layer GPU launch cost is
    paid once per group instead of once per child. Control experiments
    (2026-07-09): applying it at every layer beats any layer-gated variant
    (same or smaller trees, e.g. 4.5x smaller on prop_1/3_9), and an
    additional zonotope bound rung is fully redundant under it.

    Args:
        entries: list of ``(child_input_star, child_fixed)``.
        gpu_iters: per-layer dual-solve iteration budget (default
            ``WAVE_BOUNDS_GPU_ITERS``).
        deadline: absolute ``time.perf_counter()`` deadline, checked between
            layers; on expiry returns all-``None`` (the caller must treat an
            expired deadline as UNKNOWN).

    Returns:
        Output stars aligned to ``entries`` (provenance attached; no
        checkpoints -- the contract path always re-reaches from scratch).
    """
    from n2v.refine.tighten import neuron_bounds_population

    n = len(entries)
    Ss = [e[0] for e in entries]
    fixeds = [e[1] for e in entries]
    metas: List[List[NeuronMeta]] = [[] for _ in range(n)]
    relu_idx = 0
    for layer in layers:
        if deadline is not None and time.perf_counter() > deadline:
            return [None] * n
        if isinstance(layer, (LinearLayer, ConvLayer)):
            Ss = [apply_affine(S, layer) for S in Ss]
        elif isinstance(layer, ReluLayer):
            res = neuron_bounds_population(
                Ss, gpu_iters=(gpu_iters or WAVE_BOUNDS_GPU_ITERS))
            for i in range(n):
                S2, lm, _, _ = _relu_layer_relaxed(
                    Ss[i], relu_idx, fixeds[i], "dual_gpu",
                    precomputed_bounds=(res[i][0], res[i][1]))
                Ss[i] = S2
                metas[i].extend(lm)
            relu_idx += 1
        else:
            raise TypeError(f"Unknown layer type {type(layer).__name__}")
    return [_attach_provenance(Ss[i], metas[i], fixeds[i], "dual_gpu", None)
            for i in range(n)]


def _pin_rows(nm: NeuronMeta, nVar: int, active: bool):
    """
    The extra inequality rows that un-relax neuron ``nm`` to one phase *on the
    existing output star* -- the proposal's equality-pin discipline. Returns
    ``(A, b)`` with ``A @ alpha <= b``.

    Un-relaxing must PIN the relaxed output variable with an equality, not merely
    add the pre-activation halfspace: with the fresh var ``alpha_r`` (column
    ``pred_col``) left free in its triangle, the parent witness survives unchanged
    into one child and the score is degenerate. Reading
    ``g_i(alpha) = preact_center + preact_gens . alpha``:

      active   : g_i >= 0  AND  alpha_r = g_i   (exact identity branch)
      inactive : g_i <= 0  AND  alpha_r = 0      (exact zero branch)

    matching exactly what ``reach_relaxed`` adds for a forced neuron, but applied
    to ``S_out`` with no re-propagation. The equality is encoded as the pair of
    inequalities so the (inequality-only) epigraph LP accepts it.
    """
    g = np.zeros(nVar, dtype=np.float64)
    g[: len(nm.preact_gens)] = nm.preact_gens
    e = np.zeros(nVar, dtype=np.float64)
    e[nm.pred_col] = 1.0
    c = nm.preact_center
    if active:
        # g_i >= 0 ; alpha_r <= g_i ; alpha_r >= g_i
        A = np.vstack([-g, e - g, g - e])
        b = np.array([c, c, -c])
    else:
        # g_i <= 0 ; alpha_r <= 0 ; alpha_r >= 0
        A = np.vstack([g, e, -e])
        b = np.array([-c, 0.0, 0.0])
    return A, b


def split_wave(
    stars_witnesses: List[tuple],
    input_star: Star,
    layers: List[Layer],
    spec: LinearSpec,
    selector: Selector,
    *,
    lp_solver=LPSolver.DEFAULT,
    deadline: Optional[float] = None,
    gpu_iters: Optional[int] = None,
) -> List[Optional[List[Star]]]:
    """
    Contract-split a WAVE of nodes with ONE batched domain-contraction call.

    The contraction GPU call is launch-bound, not size-bound (~the same
    wall-clock for 2 systems as for 128), so a frontier wave's splits must
    share one call: every job contributes its two children's systems (parent
    C/d + that child's phase-pin rows over the parent's own predicate box),
    the population solve certifies all input boxes at once, and only the
    per-child reaches remain per-node.

    Semantics per job are ``split``'s witness-guided activation split, plus
    certified input-domain contraction: before each child's reach, the *input
    box* is contracted to the NS-certified range of the input dims over the
    parent's polytope plus the child's phase-pin rows. The child then
    re-reaches from scratch from the contracted input star (never
    incrementally -- parent checkpoints assume the parent's box), so every
    layer's bounds see the accumulated branch constraints through the box as
    well as the rows. Sound because the contracted boxes are NS-outward (a
    feasible input is never cut); a certified-empty child (its phase region
    is infeasible over P) is dropped: the remaining child alone covers the
    parent's region.

    Args:
        stars_witnesses: list of ``(star, witness)`` jobs.
        deadline: optional absolute ``time.perf_counter()`` deadline; child
            reaches past it are skipped (their jobs stay ``None``) -- the
            caller must treat an expired deadline as UNKNOWN, never as a
            completed wave.

    Returns:
        Per job (aligned): ``None`` if there was nothing to split (no relaxed
        neuron / no selector choice -- the caller's inconclusive path), else
        the child list (2 children; 1 or 0 when phase regions are certified
        empty -- 0 means the parent region itself is empty, a sound prune).
    """
    from n2v.refine.tighten import contract_input_boxes

    k0 = input_star.nVar
    results: List[Optional[List[Star]]] = [None] * len(stars_witnesses)
    jobs = []  # (result_idx, parent bound_mode, fixed_pair)
    systems, boxes = [], []
    for idx, (star, witness) in enumerate(stars_witnesses):
        meta: List[NeuronMeta] = star.relax_meta or []
        if not meta:
            continue
        key = selector.choose(star, spec, meta, witness, lp_solver)
        nm = (next((m for m in meta if m.key == key), None)
              if key is not None else None)
        if nm is None:
            continue
        fixed = star.fixed or {}
        wphase = witness_phase(witness.alpha, nm)
        other = Phase.INACTIVE if wphase == Phase.ACTIVE else Phase.ACTIVE
        for phase in (wphase, other):
            A_pin, b_pin = _pin_rows(nm, star.nVar,
                                     active=(phase == Phase.ACTIVE))
            b_pin = b_pin.reshape(-1, 1)
            if star.C.size > 0:
                systems.append((np.vstack([star.C, A_pin]),
                                np.vstack([star.d, b_pin])))
            else:
                systems.append((A_pin, b_pin))
            boxes.append((star.predicate_lb, star.predicate_ub))
        jobs.append((idx, star.bound_mode or "box",
                     [{**fixed, key: phase} for phase in (wphase, other)]))
    if not jobs:
        return results

    contracted = contract_input_boxes(systems, boxes, k0)
    entries, owners = [], []  # owners: (result_idx, parent bound_mode)
    dual_wave = True
    for j, (idx, bound_mode, fixed_pair) in enumerate(jobs):
        results[idx] = []
        if bound_mode != "dual_gpu":
            dual_wave = False
        for c, child_fixed in enumerate(fixed_pair):
            clb, cub, empty = contracted[2 * j + c]
            if empty:
                continue  # certified-infeasible phase region: free sound prune
            child_input = Star(input_star.V, input_star.C, input_star.d,
                               clb, cub)
            entries.append((child_input, child_fixed))
            owners.append((idx, bound_mode))
    if not entries:
        return results

    if dual_wave:
        # Group reach: one population bounds call per layer for ALL children.
        children_flat = _reach_children_wave(entries, layers, gpu_iters=gpu_iters,
                                             deadline=deadline)
    else:
        # Other bound modes under contraction (testing/ablation): per-child
        # from-scratch reach at that star's own bound mode.
        children_flat = []
        for (child_input, child_fixed), (_, bound_mode) in zip(entries,
                                                               owners):
            if deadline is not None and time.perf_counter() > deadline:
                children_flat.append(None)
                continue
            child, _ = relaxed_reach(child_input, layers, child_fixed,
                                     bound_mode=bound_mode,
                                     gpu_iters=gpu_iters)
            children_flat.append(child)

    for (idx, _), child in zip(owners, children_flat):
        if child is None:
            results[idx] = None  # deadline expiry: caller returns UNKNOWN
        elif results[idx] is not None:
            results[idx].append(child)
    return results
