"""
GPU-accelerated input-space BaB (Stage 2 of the parallel plan).

The per-node wall was OBBT: ~600 cold CPU LPs/node at ~24ms each (~5s/node). This
driver instead advances the *whole frontier* through the network in lockstep and,
at each ReLU layer, batches every frontier node's OBBT into ONE 3D batched PDHG
(``solve_lp_population_gpu``, one ``bmm`` per step over the node axis). PT-2 found
per-node (batch=1) GPU OBBT loses on narrow ACAS; the large frontier batch flips
the regime -- measured ~50x over sequential CPU HiGHS on real ACAS deep-layer
stars, with NS-certified bounds ~1-10% looser than exact (vs zono's ~30%).

Soundness: NS bounds are an *outward* enclosure (sound at any PDHG accuracy) and
are intersected with the box (also sound), so the triangle relaxation stays a
sound over-approximation; the verdict is identical to the CPU engine (only node
counts change with bound tightness). Branch logic is the same as
``verify_refine_input``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np

from n2v.sets import HalfSpace
from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.utils.verify_specification import verify_specification
from n2v.utils.lpsolver_gpu import solve_lp_population_gpu
from n2v.refine.reach_relaxed import (
    extract_layers, LinearLayer, ReluLayer, _relu_layer_relaxed,
)
from n2v.refine.types import LinearSpec, RefineResult, Status
from n2v.refine.witness import is_true_counterexample, violation_lp
from n2v.refine.input_split import _bisect, _pick_dim


def _batched_obbt_gpu(stars: List[Star], gpu_iters: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per-star ``(lb, ub)`` neuron ranges for a population of stars sharing the
    layer width ``dim``, via one 3D batched PDHG. Heterogeneous ``nVar``/constraint
    counts are zero-padded to the population max (inert under NS). Falls back to
    the exact box interval when there are no constraints yet (m==0)."""
    B = len(stars)
    dim = stars[0].dim
    n = max(s.nVar for s in stars)
    m = max((s.C.shape[0] if s.C.size else 0) for s in stars)
    if m == 0 or n == 0:
        return [s.estimate_ranges() for s in stars]

    k = 2 * dim
    A = np.zeros((B, m, n)); b = np.zeros((B, m))
    lbx = np.zeros((B, n)); ubx = np.zeros((B, n))
    C = np.zeros((B, n, k))
    for i, s in enumerate(stars):
        nv = s.nVar
        mc = s.C.shape[0] if s.C.size else 0
        if mc:
            A[i, :mc, :nv] = s.C
            b[i, :mc] = s.d.flatten()
        lbx[i, :nv] = s.predicate_lb.flatten()
        ubx[i, :nv] = s.predicate_ub.flatten()
        gens = s.V[:, 1:]                                  # (dim, nv)
        C[i, :nv, 0::2] = gens.T                           # min objectives
        C[i, :nv, 1::2] = gens.T                           # max objectives
    flags = [True, False] * dim
    res = solve_lp_population_gpu(A, b, lbx, ubx, C, flags, max_iters=gpu_iters)  # (B, k)

    out = []
    for i, s in enumerate(stars):
        c = s.V[:, 0]
        lb = res[i, 0::2] + c
        ub = res[i, 1::2] + c
        out.append((lb.reshape(-1, 1), ub.reshape(-1, 1)))
    return out


def batched_relaxed_reach(input_stars: List[Star], layers, gpu_iters: int = 1500) -> List[Star]:
    """Advance ``B`` input stars through ``layers`` (no fixed phases -- input
    splitting) in lockstep, batching the OBBT at each ReLU on the GPU. The GPU
    bound is fed to the (tested) triangle relaxation as a tighter-than-box bound
    to intersect (the ``zono`` path), so the relaxation logic is unchanged."""
    stars = list(input_stars)
    relu_idx = 0
    for layer in layers:
        if isinstance(layer, LinearLayer):
            stars = [s.affine_map(layer.W, layer.b) for s in stars]
        elif isinstance(layer, ReluLayer):
            bounds = _batched_obbt_gpu(stars, gpu_iters)
            stars = [
                _relu_layer_relaxed(s, relu_idx, {}, bound_mode="zono", zono_bounds=bd)[0]
                for s, bd in zip(stars, bounds)
            ]
            relu_idx += 1
        else:
            raise TypeError(f"Unsupported layer {type(layer).__name__}")
    return stars


def verify_refine_input_gpu(
    input_star: Star,
    model,
    spec: LinearSpec,
    *,
    layers=None,
    node_budget: int = 200000,
    time_budget: Optional[float] = None,
    min_width: float = 1e-6,
    heuristic: str = "smear",
    lp_solver=LPSolver.DEFAULT,
    batch: int = 256,
    gpu_iters: int = 1500,
    n_workers: int = 16,
) -> RefineResult:
    """
    Input-space BaB with the whole frontier's OBBT batched on the GPU. Each wave
    of up to ``batch`` frontier nodes is reached together (``batched_relaxed_reach``,
    one GPU PDHG per layer), then the cheap per-node post-reach work (spec
    disjointness, witness CE, split selection) runs concurrently on a thread pool.
    Same verdict as ``verify_refine_input``.
    """
    if layers is None:
        layers = extract_layers(model)
    spec_hs = HalfSpace(spec.G, spec.g)
    n_in = input_star.nVar

    frontier: List[Tuple[Star, int]] = [(input_star, 0)]
    nodes = 0
    max_depth = 0
    inconclusive = False
    start = time.perf_counter()

    def decide(S_in, out_star):
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

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        while frontier:
            timed_out = time_budget is not None and time.perf_counter() - start > time_budget
            if nodes >= node_budget or timed_out:
                return RefineResult(Status.UNKNOWN, nodes, max_depth)
            wave = [frontier.pop() for _ in range(min(batch, len(frontier)))]
            nodes += len(wave)
            max_depth = max(max_depth, max(d for _, d in wave))

            out_stars = batched_relaxed_reach([s for s, _ in wave], layers, gpu_iters)
            results = list(pool.map(lambda p: decide(p[0][0], p[1]), list(zip(wave, out_stars))))

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

    return RefineResult(Status.UNKNOWN if inconclusive else Status.UNSAT, nodes, max_depth)
