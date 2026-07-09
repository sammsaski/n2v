"""
Input-space branching for low-input-dimension problems (e.g. ACAS Xu: 5 inputs).

Instead of splitting ReLU *activations*, bisect the INPUT box along one input
dimension. A tighter input range tightens every neuron's bound, so more neurons
stabilise and the triangle relaxation tightens -- often deciding a sub-box
outright with zero activation splits. This attacks **tree size** (the wall on
hard ACAS Xu instances that per-node speedups could not move), and is the
standard approach for input-dim < ~20 (alpha-beta-CROWN uses input branching
there, naming ACAS Xu specifically).

Bisection halves the chosen input dimension's *predicate box* (``from_bounds``
normalises each input to ``alpha_d in [-1,1]`` with generator = half-width), so a
sub-box is just the input star with a tightened predicate -- ``V``/``C``/``d`` are
untouched. Soundness mirrors the activation-split engine: a branch is pruned only
when ``verify_specification`` certifies the (sound over-approximate) reach disjoint
from the unsafe region; SAT requires an *exact* forward pass landing in the unsafe
region; emptying the worklist with every box proven safe is a sound UNSAT, while a
box that can be neither decided nor refined (hit ``min_width``) degrades the
verdict to UNKNOWN rather than a possibly-false UNSAT.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np

from n2v.sets import HalfSpace
from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.utils.verify_specification import verify_specification
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.types import LinearSpec, RefineResult, Status
from n2v.refine.witness import binding_row, is_true_counterexample, violation_lp

logger = logging.getLogger(__name__)


def _bisect(S_in: Star, d: int) -> Tuple[Star, Star]:
    """Split the input star in two by halving input dimension ``d``'s predicate box."""
    mid = 0.5 * (float(S_in.predicate_lb[d, 0]) + float(S_in.predicate_ub[d, 0]))
    lo_ub = S_in.predicate_ub.copy(); lo_ub[d, 0] = mid       # child A: alpha_d <= mid
    hi_lb = S_in.predicate_lb.copy(); hi_lb[d, 0] = mid       # child B: alpha_d >= mid
    A = Star(S_in.V, S_in.C, S_in.d, S_in.predicate_lb.copy(), lo_ub)
    B = Star(S_in.V, S_in.C, S_in.d, hi_lb, S_in.predicate_ub.copy())
    return A, B


def _pick_dim(
    S_in: Star, out_star: Star, spec: LinearSpec, alpha: np.ndarray,
    n_in: int, min_width: float, heuristic: str,
) -> Optional[int]:
    """Choose the input dim to bisect (only among dims still wider than ``min_width``)."""
    width = (S_in.predicate_ub.flatten()[:n_in] - S_in.predicate_lb.flatten()[:n_in])
    eligible = width > min_width
    if not np.any(eligible):
        return None
    if heuristic == "widest":
        score = width.astype(np.float64)
    elif heuristic == "smear":
        # sensitivity of the binding spec row to each input dim, times its range:
        # |h . V_out[:, input_d]| * (alpha half-width). Splits the input whose
        # current range contributes most to the obstructing margin.
        h = spec.G[binding_row(out_star, spec, alpha)]
        V_in = out_star.V[:, 1:1 + n_in]                   # output cols for input dims
        score = np.abs(h @ V_in) * (width / 2.0)
    else:
        raise ValueError(f"unknown heuristic {heuristic!r}; expected 'smear'|'widest'")
    score = np.where(eligible, score, -np.inf)
    return int(np.argmax(score))


def _process_node(
    S_in: Star, model, spec: LinearSpec, layers, spec_hs: HalfSpace,
    bound_mode: str, n_in: int, min_width: float, heuristic: str, lp_solver,
):
    """
    Reach + decide one input sub-box (the per-node work, side-effect-free so it is
    safe to run concurrently across the frontier). Returns one of:
      ("safe", None)            -- provably disjoint from the unsafe region (prune)
      ("sat", x_in)             -- a real counterexample
      ("split", (childA, childB))
      ("stuck", None)           -- cannot refine further (-> UNKNOWN)
    """
    out_star, _ = relaxed_reach(S_in, layers, {}, bound_mode=bound_mode)
    if verify_specification([out_star], spec_hs).verdict == "UNSAT":
        return ("safe", None)
    res = violation_lp(out_star, spec, include_Cd=True, lp_solver=lp_solver)
    if res is None:
        return ("safe", None)
    alpha, _ = res
    is_ce, x_in, _ = is_true_counterexample(alpha, S_in, model, spec)
    if is_ce:
        return ("sat", x_in)
    d = _pick_dim(S_in, out_star, spec, alpha, n_in, min_width, heuristic)
    if d is None:
        return ("stuck", None)
    return ("split", _bisect(S_in, d))


def _reach_margin(S_in, model, spec, layers, spec_hs, bound_mode, lp_solver):
    """
    Reach one candidate input sub-box and return ``(margin, ce_x)``: the worst
    spec-row margin over the (sound over-approximate) reach -- ``+inf`` if the box
    is provably safe (disjoint from the unsafe region) -- and a concrete
    counterexample input if the box's witness is a real CE, else ``None``. The
    per-candidate work for exact input strong branching; side-effect-free, so the
    candidates are scored concurrently.
    """
    out_star, _ = relaxed_reach(S_in, layers, {}, bound_mode=bound_mode)
    if verify_specification([out_star], spec_hs).verdict == "UNSAT":
        return (np.inf, None)
    res = violation_lp(out_star, spec, include_Cd=True, lp_solver=lp_solver)
    if res is None:
        return (np.inf, None)
    alpha, t = res
    is_ce, x_in, _ = is_true_counterexample(alpha, S_in, model, spec)
    return (t, x_in if is_ce else None)


def verify_refine_input_sb(
    input_star: Star,
    model,
    spec: LinearSpec,
    *,
    layers=None,
    bound_mode: str = "box",
    node_budget: int = 100000,
    time_budget: Optional[float] = None,
    min_width: float = 1e-6,
    lp_solver=LPSolver.DEFAULT,
    n_workers: int = 8,
) -> RefineResult:
    """
    Input-space BaB with **exact strong branching**: at each node, score *every*
    candidate input dimension by actually bisecting it and reaching both children,
    then split the dimension whose worst child has the highest margin (FSB score
    ``max_d min(margin(A_d), margin(B_d))`` -- the split that drives both halves
    hardest toward safety, and decides the node outright if some dimension makes
    both halves safe). Affordable only because the input dimension is tiny
    (~5 for ACAS) and the candidate reaches are independent -> scored concurrently
    in a thread pool. Same soundness contract as ``verify_refine_input``; any
    candidate's real CE short-circuits to SAT.
    """
    if layers is None:
        layers = extract_layers(model)
    spec_hs = HalfSpace(spec.G, spec.g)
    n_in = input_star.nVar

    worklist: List[Tuple[Star, int]] = [(input_star, 0)]
    nodes = 0
    max_depth = 0
    inconclusive = False
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        while worklist:
            timed_out = time_budget is not None and time.perf_counter() - start > time_budget
            if nodes >= node_budget or timed_out:
                return RefineResult(Status.UNKNOWN, nodes, max_depth)
            S_in, depth = worklist.pop()
            nodes += 1
            max_depth = max(max_depth, depth)

            # node's own verdict (prune / CE) before scoring splits
            kind, payload = _process_node(
                S_in, model, spec, layers, spec_hs, bound_mode, n_in, min_width,
                heuristic="widest", lp_solver=lp_solver,
            )
            if kind == "sat":
                return RefineResult(Status.SAT, nodes, max_depth, counterexample_x=payload)
            if kind == "safe":
                continue
            if kind == "stuck":
                inconclusive = True
                continue

            # exact strong branching: bisect every eligible dim, reach both
            # children of each, score concurrently.
            width = (S_in.predicate_ub.flatten()[:n_in] - S_in.predicate_lb.flatten()[:n_in])
            cand = [d for d in range(n_in) if width[d] > min_width]
            pairs = {d: _bisect(S_in, d) for d in cand}
            tasks = [(d, child) for d in cand for child in pairs[d]]
            margins = list(pool.map(
                lambda c: _reach_margin(c, model, spec, layers, spec_hs, bound_mode, lp_solver),
                [c for _, c in tasks],
            ))
            by_d: dict = {}
            for (d, _), (margin, ce_x) in zip(tasks, margins):
                if ce_x is not None:
                    return RefineResult(Status.SAT, nodes, max_depth, counterexample_x=ce_x)
                by_d.setdefault(d, []).append(margin)
            d_star = max(cand, key=lambda d: min(by_d[d]))
            A, B = pairs[d_star]
            worklist.append((A, depth + 1))
            worklist.append((B, depth + 1))

    final = Status.UNKNOWN if inconclusive else Status.UNSAT
    return RefineResult(final, nodes, max_depth)


def verify_refine_input(
    input_star: Star,
    model,
    spec: LinearSpec,
    *,
    layers=None,
    bound_mode: str = "box",
    node_budget: int = 100000,
    time_budget: Optional[float] = None,
    min_width: float = 1e-6,
    heuristic: str = "smear",
    lp_solver=LPSolver.DEFAULT,
) -> RefineResult:
    """
    Decide ``G y <= g`` over ``input_star`` by (serial DFS) input-space
    branch-and-bound.

    ``heuristic``: "smear" (sensitivity-weighted, default) or "widest" (largest
    remaining input range). ``min_width`` is the smallest input-predicate range a
    dimension may be split to before the box is declared unrefinable (-> UNKNOWN).
    """
    if layers is None:
        layers = extract_layers(model)
    spec_hs = HalfSpace(spec.G, spec.g)
    n_in = input_star.nVar

    worklist: List[Tuple[Star, int]] = [(input_star, 0)]
    nodes = 0
    max_depth = 0
    inconclusive = False
    start = time.perf_counter()

    while worklist:
        timed_out = time_budget is not None and time.perf_counter() - start > time_budget
        if nodes >= node_budget or timed_out:
            return RefineResult(Status.UNKNOWN, nodes, max_depth)
        S_in, depth = worklist.pop()
        nodes += 1
        max_depth = max(max_depth, depth)
        kind, payload = _process_node(
            S_in, model, spec, layers, spec_hs, bound_mode, n_in, min_width,
            heuristic, lp_solver,
        )
        if kind == "sat":
            return RefineResult(Status.SAT, nodes, max_depth, counterexample_x=payload)
        if kind == "safe":
            continue
        if kind == "stuck":
            inconclusive = True
            continue
        A, B = payload
        worklist.append((A, depth + 1))
        worklist.append((B, depth + 1))

    final = Status.UNKNOWN if inconclusive else Status.UNSAT
    return RefineResult(final, nodes, max_depth)


def verify_refine_input_parallel(
    input_star: Star,
    model,
    spec: LinearSpec,
    *,
    layers=None,
    bound_mode: str = "box",
    node_budget: int = 100000,
    time_budget: Optional[float] = None,
    min_width: float = 1e-6,
    heuristic: str = "smear",
    lp_solver=LPSolver.DEFAULT,
    n_workers: int = 8,
    batch: Optional[int] = None,
) -> RefineResult:
    """
    Parallel-frontier input-space BaB: identical verdict to ``verify_refine_input``
    (nodes are independent sub-problems), but each wave of up to ``batch`` frontier
    nodes is reached + checked concurrently in a thread pool. The heavy per-node
    work (numpy affine maps, HiGHS LP, torch forward) releases the GIL, so threads
    give real multi-core speedup. ``batch`` defaults to ``4 * n_workers``.

    Soundness is order-independent: any node's real CE -> SAT; the frontier
    emptying with every node proven safe -> UNSAT; a node that cannot be refined,
    or hitting the budget with a non-empty frontier -> UNKNOWN (never false UNSAT).
    """
    if layers is None:
        layers = extract_layers(model)
    spec_hs = HalfSpace(spec.G, spec.g)
    n_in = input_star.nVar
    if batch is None:
        batch = 4 * n_workers

    frontier: List[Tuple[Star, int]] = [(input_star, 0)]
    nodes = 0
    max_depth = 0
    inconclusive = False
    start = time.perf_counter()

    def work(node):
        return _process_node(
            node[0], model, spec, layers, spec_hs, bound_mode, n_in, min_width,
            heuristic, lp_solver,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        while frontier:
            timed_out = time_budget is not None and time.perf_counter() - start > time_budget
            if nodes >= node_budget or timed_out:
                return RefineResult(Status.UNKNOWN, nodes, max_depth)
            wave = [frontier.pop() for _ in range(min(batch, len(frontier)))]
            nodes += len(wave)
            max_depth = max(max_depth, max(d for _, d in wave))
            results = list(pool.map(work, wave))

            children: List[Tuple[Star, int]] = []
            for (S_in, depth), (kind, payload) in zip(wave, results):
                if kind == "sat":
                    return RefineResult(Status.SAT, nodes, max_depth, counterexample_x=payload)
                if kind == "safe":
                    continue
                if kind == "stuck":
                    inconclusive = True
                    continue
                A, B = payload
                children.append((A, depth + 1))
                children.append((B, depth + 1))
            frontier.extend(children)

    final = Status.UNKNOWN if inconclusive else Status.UNSAT
    return RefineResult(final, nodes, max_depth)


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
    (CROWN, float64) bound proves every point violates some unsafe-region row;
    SAT requires an exact forward pass into the closed unsafe region at a
    point verified to be an EXACT member of the input star (zero tolerance,
    box AND C/d — solver-tolerance membership is not accepted; candidates on
    non-invertible input maps are discarded rather than trusted); a box that
    cannot be split further degrades the verdict to UNKNOWN, never a false
    UNSAT. The search runs over the input star's bounding box, which
    over-approximates constrained input sets (sound on the UNSAT side; the
    LP-grade drivers may be needed to DECIDE genuinely constrained inputs).

    ``time_budget`` is checked per wave: wall clock can overshoot by at most
    one wave's processing (~fraction of a second at default sizes);
    ``node_budget`` is exact (waves shrink to the remaining budget).
    """
    import torch

    from n2v.refine.crown_bounds import CrownBatch, prepare_layers

    if layers is None:
        layers = extract_layers(model)
    prepared = prepare_layers(layers, device=device)
    model_device = next(model.parameters()).device

    G = np.asarray(spec.G, dtype=np.float64)
    g = np.asarray(spec.g, dtype=np.float64).flatten()
    lb0, ub0 = (b.flatten() for b in input_star.estimate_ranges())
    n_in = lb0.size

    # exact zero-tolerance membership (SAT gate): recover alpha for diagonal
    # input maps (Star.from_bounds and friends); non-diagonal maps cannot be
    # inverted pointwise, so their candidates are discarded (stay sound).
    c_in = input_star.V[:, 0]
    V_in = input_star.V[:, 1:]
    diag = np.diag(V_in).copy() if V_in.shape[0] == V_in.shape[1] else None
    if diag is not None and not np.array_equal(V_in, np.diag(diag)):
        diag = None

    def _exact_member(x: np.ndarray) -> bool:
        if diag is None:
            return False
        alpha = np.zeros(n_in)
        nz = diag != 0.0
        alpha[nz] = (x[nz] - c_in[nz]) / diag[nz]
        if np.any(x[~nz] != c_in[~nz]):
            return False
        if input_star.predicate_lb is not None and np.any(
                alpha < input_star.predicate_lb.flatten()):
            return False
        if input_star.predicate_ub is not None and np.any(
                alpha > input_star.predicate_ub.flatten()):
            return False
        if input_star.C.size > 0 and np.any(
                input_star.C @ alpha > input_star.d.flatten()):
            return False
        return True

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
            # batched EXACT forward on every undecided box's candidate vertex
            X = vertices[undecided]
            with torch.no_grad():
                Y = model(torch.as_tensor(X, dtype=torch.float64)
                          .to(model_device)).cpu().numpy()
            hits = np.flatnonzero((G @ Y.T - g[:, None] <= 0.0).all(axis=0))
            for h in hits:
                x = X[int(h)]
                if _exact_member(x):
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
