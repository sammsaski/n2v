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
    Decide ``G y <= g`` over ``input_star`` by input-space branch-and-bound.

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

        out_star, _ = relaxed_reach(S_in, layers, {}, bound_mode=bound_mode)
        if verify_specification([out_star], spec_hs).verdict == "UNSAT":
            continue  # this sub-box is provably safe

        res = violation_lp(out_star, spec, include_Cd=True, lp_solver=lp_solver)
        if res is None:
            continue  # empty -> vacuously safe
        alpha, _ = res

        is_ce, x_in, _ = is_true_counterexample(alpha, S_in, model, spec)
        if is_ce:
            return RefineResult(Status.SAT, nodes, max_depth, counterexample_x=x_in)

        if nodes >= node_budget:
            inconclusive = True
            continue
        d = _pick_dim(S_in, out_star, spec, alpha, n_in, min_width, heuristic)
        if d is None:
            inconclusive = True  # cannot refine further -> stay sound (UNKNOWN)
            continue
        A, B = _bisect(S_in, d)
        worklist.append((A, depth + 1))
        worklist.append((B, depth + 1))

    final = Status.UNKNOWN if inconclusive else Status.UNSAT
    return RefineResult(final, nodes, max_depth)
