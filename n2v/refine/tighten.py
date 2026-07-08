"""
Bound tightening: re-derive each neuron's range by LP over the predicate polytope
P (respecting the accumulated C/d constraints), the "tighter output set"
refinement mode.

``neuron_bounds(S, backend)`` returns the per-dimension range of star ``S`` over
``P = {C alpha <= d, predicate box}`` -- the OBBT batch (2*dim objectives sharing
one constraint matrix). Backends:

  * ``"lp_cpu"``: HiGHS (exact range) -- delegates to ``Star.get_ranges`` (the
    same batched LP-over-P the rest of n2v uses).
  * ``"lp_gpu"``: batched PDHG + Neumaier-Shcherbina (an *outward* enclosure of
    the range, sound at any solver accuracy). ``Star`` has no GPU backend, so
    this path routes the OBBT batch through ``solve_lp_batch_gpu`` directly.
  * ``"dual_gpu"``: batched projected-Adam descent on the NS dual itself
    (best-iterate certified; see ``lpsolver_gpu._grad_duals_gpu``). Same
    soundness contract as ``lp_gpu`` (NS outward), materially tighter per
    iteration on deep-layer geometries.

Compared with ``Star.estimate_ranges`` (the predicate-box interval that ignores
C/d), these bounds are tighter-or-equal and still sound, so they stabilise more
neurons and tighten the triangle relaxation -> fewer splits.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.utils.lpsolver_gpu import solve_lp_batch_gpu, solve_lp_population_gpu


def _lp_ranges(S: Star, backend: str, dims: np.ndarray,
               gpu_iters: Optional[int] = None,
               warm_duals: Optional[dict] = None,
               collect_duals: bool = False):
    """
    Batched min/max of ``S``'s output dims ``dims`` over the predicate polytope.
    Returns ``(lo, hi)`` aligned to ``dims`` (each already offset by the center).
    ``lp_cpu`` uses Star's batched HiGHS primitive; ``lp_gpu`` uses batched
    PDHG + Neumaier-Shcherbina (outward). Solving only ``dims`` (rather than all
    output dims) is what makes the OBBT pre-filter a real LP-count saving.

    ``warm_duals`` (``dual_gpu`` only): ``{dim: (y_min, y_max)}`` row-dual
    prefixes from an ancestor's solve of the same-prefix polytope — sound for
    any values, saves iterations. ``collect_duals=True`` returns
    ``(lo, hi, duals)`` with ``duals = {dim: (y_min, y_max)}`` (float32) for
    caching; otherwise returns ``(lo, hi)``.
    """
    dims = np.asarray(dims, dtype=int)
    c = S.V[:, 0]
    gens = S.V[:, 1:].astype(np.float64)
    duals_by_dim: dict = {}

    if backend == "lp_cpu":
        objectives, flags = [], []
        for i in dims:
            objectives.extend([gens[i], gens[i]])
            flags.extend([True, False])
        res = S._solve_lp_batch(objectives, flags, LPSolver.DEFAULT)
    elif backend in ("lp_gpu", "dual_gpu"):
        # NS requires finite predicate bounds; without them fall back to exact CPU.
        if S.predicate_lb is None or S.predicate_ub is None:
            return _lp_ranges(S, "lp_cpu", dims, collect_duals=collect_duals)
        objectives, flags = [], []
        for i in dims:
            objectives.extend([gens[i], gens[i]])
            flags.extend([True, False])
        A = S.C if S.C.size > 0 else None
        b = S.d.flatten() if S.d.size > 0 else None
        # "dual_gpu" (T1b): projected-Adam best-iterate descent on the NS dual —
        # tighter per iteration than truncated PDHG on hard geometries and
        # monotone by construction; ~300 iterations recovers >=97% of the
        # box->exact gap at 2-3 orders less cost than exact OBBT (T1a).
        gpu_backend, iters = (("grad", gpu_iters or 300) if backend == "dual_gpu"
                              else ("pdhg", 2000))
        warm = None
        if warm_duals is not None and gpu_backend == "grad":
            warm = []
            for i in dims:
                w = warm_duals.get(int(i))
                warm.extend([w[0], w[1]] if w is not None else [None, None])
        res = solve_lp_batch_gpu(
            objectives, A=A, b=b, lb=S.predicate_lb.flatten(),
            ub=S.predicate_ub.flatten(), minimize_flags=flags,
            backend=gpu_backend, max_iters=iters,
            warm_duals=warm, return_duals=collect_duals,
        )
        if collect_duals:
            res, out_duals = res
            # float16 cache: warm starts need only a decent point (soundness is
            # dual-value-independent) and the cache lives on every worklist star
            duals_by_dim = {
                int(i): (out_duals[2 * j].astype(np.float16),
                         out_duals[2 * j + 1].astype(np.float16))
                for j, i in enumerate(dims)
            }
    else:
        raise ValueError(
            f"unknown backend {backend!r}; expected 'lp_cpu'|'lp_gpu'|'dual_gpu'")

    lo = np.empty(dims.size, dtype=np.float64)
    hi = np.empty(dims.size, dtype=np.float64)
    for j, i in enumerate(dims):
        vmin, vmax = res[2 * j], res[2 * j + 1]
        if vmin is None or vmax is None:
            a, b2 = S.estimate_range(int(i))         # sound box fallback
            lo[j], hi[j] = a, b2
        else:
            lo[j], hi[j] = vmin + c[i], vmax + c[i]
    if collect_duals:
        # duals exist only on the dual_gpu path; other backends cache nothing
        return lo, hi, duals_by_dim
    return lo, hi


def neuron_bounds(
    S: Star, backend: str = "lp_cpu", prefilter: bool = False,
    gpu_iters: Optional[int] = None,
    warm_duals: Optional[dict] = None,
    collect_duals: bool = False,
):
    """
    Per-dimension range of ``S`` over its predicate polytope ``P``.

    Returns ``(lb, ub)`` each shape ``(dim, 1)`` (matching ``estimate_ranges``).
    For ``lp_gpu`` the values are NS-certified *outward* (lb <= true min,
    ub >= true max), so any downstream stabilisation stays sound.

    ``prefilter`` (default **False** -- the OBBT pre-filter, Phase-2 finding #1):
    the cheap predicate-box interval already proves stability for most neurons, so
    LP only the box-*unstable* dims (box ``l<0<u``) and keep the box interval for
    the rest. This is **result-identical** to LPing every dim, not just sound: a
    box-stable neuron is provably stable (the box over-approximates ``P``), gets
    the same ReLU classification, and a stable neuron's bound *value* never reaches
    the output star (it is zeroed or passed through -- only crossing/relaxed
    neurons, which are exactly the box-unstable ones, use their ``[l,u]`` to build
    the triangle). So the pre-filter only removes wasted LPs. ``prefilter=False``
    LPs every dim (for the A/B speed comparison and the equivalence test).

    Defaulting OFF is a measured choice, not caution: on ACAS Xu the deeper layers
    are nearly all box-unstable (few LPs to skip), and the subset solve here routes
    through the sequential ``_solve_lp_batch`` rather than ``get_ranges``' parallel
    path, so it *regresses* ~2.9x on that benchmark. The pre-filter pays off only
    when a large fraction of neurons are box-stable (e.g. deep BaB nodes with many
    fixed neurons); enable it there. (A future fix -- parallelise the subset solve
    -- would make it never-worse and safe to default on.)

    Every LP/NS bound is intersected with the (always-sound) box interval. For
    ``lp_cpu`` the exact HiGHS optimum already lies inside the box, so this is a
    no-op and the pre-filter is **bit-identical** to LPing every dim (HiGHS is
    batch-independent). For ``lp_gpu`` the result is **sound but not identical**:
    batched first-order PDHG/NS bounds are *outward* and mildly batch-composition-
    sensitive, so the box intersection tightens them and removes the stable-neuron
    reclassification, but crossing-dim values still differ by the PDHG tolerance
    between the two batches. Both paths remain a sound over-approximation (NS box
    enclosure), which is all stabilisation requires.
    """
    if S.nVar == 0:
        col = S.V[:, 0].reshape(-1, 1)
        return (col.copy(), col.copy(), {}) if collect_duals else (col.copy(), col.copy())

    lb, ub = S.estimate_ranges()
    lb = lb.copy()                          # don't alias S.state_lb/ub (estimate_ranges caches them)
    ub = ub.copy()
    if prefilter:
        lbf, ubf = lb.flatten(), ub.flatten()
        dims = np.flatnonzero((lbf < 0.0) & (ubf > 0.0))
    else:
        dims = np.arange(S.dim)

    if dims.size == 0:
        # everything box-stable -> no LP at all
        return (lb, ub, {}) if collect_duals else (lb, ub)

    res = _lp_ranges(S, backend, dims, gpu_iters=gpu_iters,
                     warm_duals=warm_duals, collect_duals=collect_duals)
    lo, hi = res[0], res[1]
    # Intersect with the box: tighter sound bound, no-op for exact CPU optima,
    # clips NS-outward GPU overshoot -> backend-independent, prefilter-invariant.
    lb[dims, 0] = np.maximum(lo, lb[dims, 0])
    ub[dims, 0] = np.minimum(hi, ub[dims, 0])
    if collect_duals:
        return lb, ub, res[2]
    return lb, ub


def neuron_bounds_population(stars, gpu_iters=None, warm=None):
    """``dual_gpu`` ``neuron_bounds`` for SEVERAL stars in one population GPU
    call (the frontier/sibling batching primitive).

    Semantics per star match ``neuron_bounds(S, "dual_gpu", prefilter=True)``:
    only box-unstable dims are solved, results are box-intersected, values are
    NS-certified outward. Heterogeneous shapes are zero-padded (inert under
    NS); the objective axis is laid out ``[min x kd_max | max x kd_max]`` so
    the shared ``minimize_flags`` vector is valid for every star.

    Args:
        stars: list of Stars (finite predicate boxes required; a star without
            one falls back to the scalar path).
        gpu_iters: gradient-iteration budget (default 300).
        warm: optional per-star dual dict ``{dim: (y_min, y_max)}`` covering a
            row prefix (see ``_lp_ranges``).

    Returns:
        list of ``(lb, ub, duals_dict)`` per star, shapes as ``neuron_bounds``.
    """
    results: list = [None] * len(stars)
    idxs, prep = [], []
    for i, S in enumerate(stars):
        if S.nVar == 0:
            col = S.V[:, 0].reshape(-1, 1)
            results[i] = (col.copy(), col.copy(), {})
            continue
        lb, ub = S.estimate_ranges()
        lb, ub = lb.copy(), ub.copy()
        if S.predicate_lb is None or S.predicate_ub is None:
            r = neuron_bounds(S, "lp_cpu")
            results[i] = (r[0], r[1], {})
            continue
        dims = np.flatnonzero((lb.flatten() < 0.0) & (ub.flatten() > 0.0))
        if dims.size == 0:
            results[i] = (lb, ub, {})
            continue
        idxs.append(i)
        prep.append((S, lb, ub, dims))
    if not idxs:
        return results

    B = len(idxs)
    kd_max = max(p[3].size for p in prep)
    m_max = max(p[0].C.shape[0] for p in prep) or 1
    n_max = max(p[0].nVar for p in prep)
    k = 2 * kd_max
    Ap = np.zeros((B, m_max, n_max))
    bp = np.zeros((B, m_max))
    lbp = np.zeros((B, n_max))
    ubp = np.zeros((B, n_max))
    Cp = np.zeros((B, n_max, k))
    Y0 = np.zeros((B, m_max, k)) if warm is not None else None
    for j, (S, _, _, dims) in enumerate(prep):
        mi, ni = S.C.shape[0], S.nVar
        if S.C.size:
            Ap[j, :mi, :ni] = S.C
            bp[j, :mi] = S.d.flatten()
        lbp[j, :ni] = S.predicate_lb.flatten()
        ubp[j, :ni] = S.predicate_ub.flatten()
        gens = S.V[:, 1:].astype(np.float64)
        for c, dim in enumerate(dims):
            Cp[j, :ni, c] = gens[dim]
            Cp[j, :ni, kd_max + c] = gens[dim]
            w = warm[idxs[j]].get(int(dim)) if (warm is not None
                                                and warm[idxs[j]]) else None
            if w is not None and Y0 is not None:
                r = min(w[0].size, mi)
                Y0[j, :r, c] = np.clip(w[0][:r], 0.0, None)
                r = min(w[1].size, mi)
                Y0[j, :r, kd_max + c] = np.clip(w[1][:r], 0.0, None)
    flags = [True] * kd_max + [False] * kd_max
    bounds, Y = solve_lp_population_gpu(
        Ap, bp, lbp, ubp, Cp, minimize_flags=flags,
        max_iters=(gpu_iters or 300), backend="grad", Y0=Y0, return_duals=True,
    )
    for j, i in enumerate(idxs):
        S, lb, ub, dims = prep[j]
        c0 = S.V[:, 0]
        mi = S.C.shape[0]
        duals: dict = {}
        for c, dim in enumerate(dims):
            lo = bounds[j, c] + c0[dim]
            hi = bounds[j, kd_max + c] + c0[dim]
            lb[dim, 0] = max(lo, lb[dim, 0])
            ub[dim, 0] = min(hi, ub[dim, 0])
            duals[int(dim)] = (Y[j, :mi, c].astype(np.float16),
                               Y[j, :mi, kd_max + c].astype(np.float16))
        results[i] = (lb, ub, duals)
    return results
