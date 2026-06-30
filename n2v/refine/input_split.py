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
