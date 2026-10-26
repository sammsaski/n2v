"""
Input-space branch and bound for low-input-dimension problems (e.g. ACAS Xu:
5 inputs), with batched CROWN bounds.

Instead of splitting ReLU *activations*, ``verify_refine_input_crown`` bisects
the INPUT box along one input dimension. A tighter input range tightens every
neuron's bound, so more neurons stabilise and the relaxation tightens -- often
deciding a sub-box outright with zero activation splits. This attacks **tree
size** (the wall on hard ACAS Xu instances that per-node speedups could not
move), and is the standard approach for input-dim < ~20 (alpha-beta-CROWN uses
input branching there, naming ACAS Xu specifically). Each wave of sub-boxes is
bounded by one batched CROWN pass (``n2v.refine.crown_bounds``) with no LPs.

Soundness mirrors the activation-split engine: a box is pruned only when a
sound CROWN bound (float64, widened outward by a rigorous rounding bound)
proves every point of it violates some unsafe-region row; SAT requires an
*exact* forward pass landing in the closed unsafe region at a candidate whose
predicate coordinates satisfy the input star's predicate box and ``C`` rows
at zero tolerance; a box that can be neither decided nor split further (it
hit ``min_width``) degrades the verdict to UNKNOWN rather than a
possibly-false UNSAT.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.refine.bab import (_eval_mode_model, _finite_input_star,
                            _require_float64_model)
from n2v.refine.reach_relaxed import extract_layers
from n2v.refine.types import LinearSpec, RefineResult, Status
from n2v.refine.witness import clip_to_raw_box

logger = logging.getLogger(__name__)


def verify_refine_input_crown(
    input_star: Star,
    model,
    spec: LinearSpec,
    *,
    layers=None,
    node_budget: int = 2_000_000,
    time_budget: Optional[float] = None,
    min_width: float = 1e-6,
    wave: int = 1024,
    device: str = "cpu",
    row_rule: str = "argmax",
) -> RefineResult:
    """Input-space BaB with batched-CROWN bounds — the cheap-bounds-in-bulk
    recipe the low-input-dimension regime rewards (what alpha-beta-CROWN runs
    below ~20 inputs), natively in this engine.

    Per iteration a WAVE of input sub-boxes is processed with no LPs at all:
    one batched CROWN pass bounds every box's spec margins in closed form with
    last-layer elision (prune), every box's driving-row vertex is checked by
    one batched EXACT forward pass (SAT), and survivors bisect on the
    CROWN-smear dimension (vectorized; children pushed as one chunk).

    Soundness mirrors the other drivers: a box is pruned only when a sound
    CROWN bound (float64, widened outward by a rigorous rounding bound)
    proves every point violates some unsafe-region row; a box that cannot be
    split further degrades the verdict to UNKNOWN, never a false UNSAT. The
    search runs over the input star's outward bounding box, which
    over-approximates constrained input sets (sound on the UNSAT side; the
    LP-grade drivers may be needed to DECIDE genuinely constrained inputs).

    SAT candidates are box vertices projected into the input star's predicate
    set: each vertex is mapped back to predicate coordinates (diagonal input
    maps only; candidates on non-diagonal maps are discarded), clipped into
    the predicate box, and kept only if it satisfies the ``C alpha <= d``
    rows at zero tolerance -- the same contract as
    ``is_true_counterexample``. Membership is checked on those predicate
    coordinates; the rebuilt point ``x = c + alpha * diag`` is not mapped
    back and re-checked, because that round trip can move a corner one ulp
    back outside the predicate box and reject exactly the corner
    counterexamples the projection exists to accept. For box-derived stars
    (``Box.to_star``, which carries raw bounds) the rebuilt point is snapped
    into the user's raw float64 box (``clip_to_raw_box``), which is what a
    zero-tolerance replay checks. Every candidate is then confirmed by a
    single-input exact forward pass into the closed unsafe region.

    ``model`` must be float64 (``TypeError`` otherwise). The counterexample
    checks run on an eval-mode model: a model with a submodule in training
    mode is deep-copied and switched to eval mode first, and the caller's
    model is never modified. A non-finite predicate box is finitized first,
    as in ``verify_refine``; an input set reported empty returns UNSAT only
    when ``Star.is_empty_set`` confirms the emptiness, and UNKNOWN otherwise.
    If the predicate box cannot be finitized at all (the finitizer raises
    ``ValueError``, e.g. an unconfirmed-infeasible polytope), the result is
    UNKNOWN.

    Known limitation: only the vertex minimizing the driving row is tried per
    box, so a counterexample that exists only at a corner jointly minimizing
    several spec rows (e.g. ``G = -I`` with the unsafe region at one corner)
    can stay UNKNOWN.

    ``time_budget`` is checked per wave: wall clock can overshoot by at most
    one wave's processing (~fraction of a second at default sizes);
    ``node_budget`` is exact (waves shrink to the remaining budget).
    """
    import torch

    from n2v.refine.crown_bounds import CrownBatch, prepare_layers

    _require_float64_model(model, "verify_refine_input_crown")
    model = _eval_mode_model(model)
    # Interval bounds (the CROWN search box) are undefined on infinite
    # predicate intervals: finitize first, exactly as verify_refine does
    # (an unconfirmed emptiness report gives UNKNOWN, never a vacuous UNSAT).
    input_star, decided = _finite_input_star(input_star, LPSolver.DEFAULT,
                                             "verify_refine_input_crown")
    if decided is not None:
        return RefineResult(decided, 0, 0)

    if layers is None:
        layers = extract_layers(model)
    prepared = prepare_layers(layers, device=device)
    model_device = next(model.parameters()).device

    G = np.asarray(spec.G, dtype=np.float64)
    g = np.asarray(spec.g, dtype=np.float64).flatten()
    lb0, ub0 = (b.flatten() for b in input_star.estimate_ranges())

    # SAT candidates. The search box is the input star's OUTWARD interval (a
    # sound cover for pruning), so its vertices can sit a few ulps outside
    # the input set -- exactly where corner-only counterexamples live. Map
    # each vertex back to predicate coordinates (diagonal input maps only),
    # clip it into the predicate box, require the C rows at zero tolerance,
    # and rebuild the input point; the per-point check below snaps it into
    # the user's raw box like every other witness (clip_to_raw_box).
    c_in = input_star.V[:, 0]
    V_in = input_star.V[:, 1:]
    diag = np.diag(V_in).copy() if V_in.shape[0] == V_in.shape[1] else None
    if diag is not None and not np.array_equal(V_in, np.diag(diag)):
        diag = None
    # A star without predicate variables (a point) has no predicate box and
    # no diagonal map, so the box is read only when the vertices are mapped.
    plb = pub = None
    if diag is not None:
        plb = input_star.predicate_lb.flatten()
        pub = input_star.predicate_ub.flatten()

    def _to_members(X):
        if diag is None:
            return X, np.zeros(X.shape[0], dtype=bool)
        nz = diag != 0.0
        A = np.zeros_like(X)
        A[:, nz] = (X[:, nz] - c_in[nz]) / diag[nz]
        A = np.clip(A, plb, pub)
        member = np.ones(X.shape[0], dtype=bool)
        if input_star.C.size > 0:
            member &= np.all(A @ input_star.C.T <= input_star.d.flatten(),
                             axis=1)
        return c_in + A * diag, member

    # chunked frontier: [(LB (K, n), UB (K, n), depth (K,))] — one chunk per
    # producing wave; avoids per-node tuple objects at ~10^6-node scale
    chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (lb0[None, :], ub0[None, :], np.zeros(1, dtype=np.int64))]
    n_pending = 1
    nodes = 0
    max_depth = 0
    inconclusive = False
    start = time.perf_counter()

    while n_pending:
        if nodes >= node_budget or (
                time_budget is not None
                and time.perf_counter() - start > time_budget):
            return RefineResult(Status.UNKNOWN, nodes, max_depth)
        take = min(wave, n_pending, node_budget - nodes)
        got, parts = 0, []
        while got < take:
            LBc, UBc, Dc = chunks.pop()
            if LBc.shape[0] > take - got:
                keep = LBc.shape[0] - (take - got)
                chunks.append((LBc[:keep], UBc[:keep], Dc[:keep]))
                LBc, UBc, Dc = LBc[keep:], UBc[keep:], Dc[keep:]
            parts.append((LBc, UBc, Dc))
            got += LBc.shape[0]
        LB = np.concatenate([p[0] for p in parts])
        UB = np.concatenate([p[1] for p in parts])
        DEPTH = np.concatenate([p[2] for p in parts])
        n_pending -= got
        nodes += got
        max_depth = max(max_depth, int(DEPTH.max()))

        cb = CrownBatch(prepared, LB, UB, device=device)
        margin_lo, vertices, smear = cb.margin_report(G, g, row_rule=row_rule)
        undecided = np.flatnonzero(~(margin_lo > 0.0).any(axis=1))

        if undecided.size:
            # batched EXACT forward on every undecided box's candidate vertex,
            # projected into the input set (only members can decide SAT)
            X, member = _to_members(vertices[undecided])
            with torch.no_grad():
                Y = model(torch.as_tensor(X, dtype=torch.float64)
                          .to(model_device)).cpu().numpy()
            hits = np.flatnonzero((G @ Y.T - g[:, None] <= 0.0).all(axis=0)
                                  & member)
            for h in hits:
                x = clip_to_raw_box(X[int(h)], input_star)
                # SOUNDNESS OF THE WITNESS: the batched GEMM margin rounds
                # differently from the per-point path the replay uses;
                # re-validate with a single-input forward at zero tolerance.
                with torch.no_grad():
                    y1 = (model(torch.as_tensor(x, dtype=torch.float64)
                                .unsqueeze(0).to(model_device))
                          .cpu().numpy().flatten())
                if bool(np.all(G @ y1 - g <= 0.0)):
                    return RefineResult(Status.SAT, nodes, max_depth,
                                        counterexample_x=x)

            # vectorized bisection of every undecided box on its smear dim
            lb_u, ub_u = LB[undecided], UB[undecided]
            width = ub_u - lb_u
            eligible = width > min_width
            splittable = eligible.any(axis=1)
            if not splittable.all():
                inconclusive = True          # sliver boxes: stay sound
            if splittable.any():
                lb_s, ub_s = lb_u[splittable], ub_u[splittable]
                d = np.where(eligible[splittable], smear[undecided][splittable],
                             -np.inf).argmax(axis=1)
                rows = np.arange(d.size)
                mid = 0.5 * (lb_s[rows, d] + ub_s[rows, d])
                ub_a = ub_s.copy(); ub_a[rows, d] = mid
                lb_b = lb_s.copy(); lb_b[rows, d] = mid
                child_depth = DEPTH[undecided][splittable] + 1
                chunks.append((np.concatenate([lb_s, lb_b]),
                               np.concatenate([ub_a, ub_s]),
                               np.concatenate([child_depth, child_depth])))
                n_pending += 2 * d.size

    final = Status.UNKNOWN if inconclusive else Status.UNSAT
    return RefineResult(final, nodes, max_depth)
