"""
Bound tightening: re-derive each neuron's range by LP over the predicate polytope
P (respecting the accumulated C/d constraints), the "tighter output set"
refinement mode.

``neuron_bounds(S, backend)`` returns the per-dimension range of star ``S`` over
``P = {C alpha <= d, predicate box}`` -- the OBBT batch (2*dim objectives sharing
one constraint matrix). Backends:

  * ``"lp_cpu"``: HiGHS (exact range) on every dim, through Star's batched
    LP-over-P primitive (the one behind ``Star.get_ranges``), padded outward
    by ``_lp_pad`` because an exact optimum is exact only to the solver's
    tolerance. HiGHS is used whatever solver is set globally
    (``_bound_lp_solver``): the pad is sized for it.
  * ``"dual_gpu"``: batched projected-Adam descent on the Neumaier-Shcherbina
    dual (best-iterate certified; see ``lpsolver_gpu._grad_duals_gpu``) on the
    box-unstable dims only -- an *outward* enclosure of the range, sound at any
    solver accuracy. It runs on CPU torch when no CUDA device is usable.

Compared with ``Star.estimate_ranges`` (the predicate-box interval that ignores
C/d), these bounds are tighter-or-equal and still sound, so they stabilise more
neurons and tighten the triangle relaxation -> fewer splits.

``contract_input_boxes`` is the certified input-domain contraction behind
``operations.split_wave``: for many ``(C, d)`` systems at once it tightens the
leading (input) dims of each predicate box to an NS-certified outward range
over ``{C alpha <= d, box}``, in one population solve.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.utils.lpsolver import _bound_lp_solver, _lp_pad
from n2v.utils.lpsolver_gpu import solve_lp_batch_gpu, solve_lp_population_gpu

# Iteration budget for domain contraction: the LPs are tiny (n_in variables,
# a few hundred rows) and cold-started, and contraction only needs a decent
# certified range, not the exact optimum -- ~100 iterations recover 93-99% of
# the box->exact gap even on much larger systems.
CONTRACT_GPU_ITERS = 120

_F16_MAX = float(np.finfo(np.float16).max)


def _safe_f16(y: np.ndarray) -> np.ndarray:
    """Store a dual vector in the float16 warm-start cache, safely.

    The cache exists only to give the next ascent a decent starting point, and
    soundness never depends on the dual's VALUE -- any finite nonnegative
    multiplier certifies a valid (possibly loose) bound. But it does depend on
    the value being FINITE: a magnitude above the float16 range casts to
    ``inf``, and an infinite multiplier makes the Neumaier-Shcherbina
    certificate return NaN (``0 * inf`` in the reduced-cost term). A NaN bound
    then passes neither the ``ub <= 0`` nor the ``l < 0 < u`` test in
    ``_relu_layer_relaxed``, so the neuron is silently treated as stably
    ACTIVE -- an identity pass-through with no constraint, which is unsound.

    Clamping to the representable range keeps the multiplier nonnegative and
    finite, so the warm start stays valid and can only be less tight.
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=_F16_MAX, neginf=0.0)
    return np.clip(y, 0.0, _F16_MAX).astype(np.float16)


def _lp_ranges(S: Star, backend: str, dims: np.ndarray,
               gpu_iters: Optional[int] = None,
               warm_duals: Optional[dict] = None,
               collect_duals: bool = False):
    """
    Batched min/max of ``S``'s output dims ``dims`` over the predicate polytope.
    Returns ``(lo, hi)`` aligned to ``dims`` (each already offset by the center).
    ``lp_cpu`` uses Star's batched HiGHS primitive; ``dual_gpu`` uses the
    batched NS-certified gradient-dual solve (outward).

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
        res = S._solve_lp_batch(objectives, flags,
                                _bound_lp_solver(LPSolver.DEFAULT))
    elif backend == "dual_gpu":
        # NS requires finite predicate bounds; without them fall back to exact CPU.
        if S.predicate_lb is None or S.predicate_ub is None:
            return _lp_ranges(S, "lp_cpu", dims, collect_duals=collect_duals)
        objectives, flags = [], []
        for i in dims:
            objectives.extend([gens[i], gens[i]])
            flags.extend([True, False])
        A = S.C if S.C.size > 0 else None
        b = S.d.flatten() if S.d.size > 0 else None
        # projected-Adam best-iterate descent on the NS dual, monotone by
        # construction; ~300 iterations recovers >=97% of the box->exact gap
        # at 2-3 orders less cost than exact OBBT (measured).
        iters = gpu_iters or 300
        warm = None
        if warm_duals is not None:
            warm = []
            for i in dims:
                w = warm_duals.get(int(i))
                warm.extend([w[0], w[1]] if w is not None else [None, None])
        res = solve_lp_batch_gpu(
            objectives, A=A, b=b, lb=S.predicate_lb.flatten(),
            ub=S.predicate_ub.flatten(), minimize_flags=flags,
            max_iters=iters,
            warm_duals=warm, return_duals=collect_duals,
        )
        if collect_duals:
            res, out_duals = res
            # float16 cache: warm starts need only a decent point (soundness is
            # dual-value-independent) and the cache lives on every worklist star
            duals_by_dim = {
                int(i): (_safe_f16(out_duals[2 * j]),
                         _safe_f16(out_duals[2 * j + 1]))
                for j, i in enumerate(dims)
            }
    else:
        raise ValueError(
            f"unknown backend {backend!r}; expected 'lp_cpu'|'dual_gpu'")

    lo = np.empty(dims.size, dtype=np.float64)
    hi = np.empty(dims.size, dtype=np.float64)
    for j, i in enumerate(dims):
        vmin, vmax = res[2 * j], res[2 * j + 1]
        if vmin is None or vmax is None:
            a, b2 = S.estimate_range(int(i))         # sound box fallback
            lo[j], hi[j] = a, b2
        else:
            lo[j], hi[j] = vmin + c[i], vmax + c[i]
            if backend == "lp_cpu":
                # SOUNDNESS: an exact-solver optimum is exact only to the
                # solver's tolerance -- a MIN can be reported above the true
                # minimum (measured undershoot up to 5e-10), which reads as a
                # stably-inactive neuron and under-approximates. Pad outward
                # by the tolerance, relative to magnitude. (dual_gpu is
                # NS-certified outward already.)
                lo[j] -= _lp_pad(lo[j])
                hi[j] += _lp_pad(hi[j])
    if collect_duals:
        # duals exist only on the dual_gpu path; other backends cache nothing
        return lo, hi, duals_by_dim
    return lo, hi


def neuron_bounds(
    S: Star, backend: str = "lp_cpu",
    gpu_iters: Optional[int] = None,
    warm_duals: Optional[dict] = None,
    collect_duals: bool = False,
):
    """
    Per-dimension range of ``S`` over its predicate polytope ``P``.

    Returns ``(lb, ub)`` each shape ``(dim, 1)`` (matching ``estimate_ranges``).
    ``lp_cpu`` solves every dim (exact HiGHS optimum, padded outward by
    ``_lp_pad``). ``dual_gpu`` values are NS-certified *outward* (lb <= true
    min, ub >= true max), so any downstream stabilisation stays sound.

    ``dual_gpu`` solves only the box-*unstable* dims (box ``l<0<u``) and keeps
    the box interval for the rest. This is **result-identical** to solving
    every dim, not just sound: a box-stable neuron is provably stable (the box
    over-approximates ``P``), gets the same ReLU classification, and a stable
    neuron's bound *value* never reaches the output star (it is zeroed or
    passed through -- only crossing/relaxed neurons, which are exactly the
    box-unstable ones, use their ``[l,u]`` to build the triangle).

    Every LP/NS bound is intersected with the (always-sound) box interval,
    which clips the ``lp_cpu`` padding and the NS-outward ``dual_gpu``
    overshoot; the result stays a sound over-approximation (the intersection
    of two enclosures), which is all stabilisation requires.
    """
    if S.nVar == 0:
        col = S.V[:, 0].reshape(-1, 1)
        return (col.copy(), col.copy(), {}) if collect_duals else (col.copy(), col.copy())

    lb, ub = S.estimate_ranges()
    lb = lb.copy()                          # don't alias S.state_lb/ub (estimate_ranges caches them)
    ub = ub.copy()
    if backend == "dual_gpu":
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
    # Intersect with the box: a tighter sound bound (clips the lp_cpu padding
    # and the NS-outward dual_gpu overshoot).
    lb[dims, 0] = np.maximum(lo, lb[dims, 0])
    ub[dims, 0] = np.minimum(hi, ub[dims, 0])
    if collect_duals:
        return lb, ub, res[2]
    return lb, ub


def neuron_bounds_population(stars, gpu_iters=None, warm=None, device=None):
    """``dual_gpu`` ``neuron_bounds`` for SEVERAL stars in one population GPU
    call (the frontier/sibling batching primitive).

    Semantics per star match ``neuron_bounds(S, "dual_gpu")``:
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
        device: torch device for the solve (``None``: CUDA when usable, else
            CPU).

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
                Y0[j, :r, c] = np.nan_to_num(
                    np.clip(w[0][:r], 0.0, None), nan=0.0, posinf=_F16_MAX)
                r = min(w[1].size, mi)
                Y0[j, :r, kd_max + c] = np.nan_to_num(
                    np.clip(w[1][:r], 0.0, None), nan=0.0, posinf=_F16_MAX)
    flags = [True] * kd_max + [False] * kd_max
    bounds, Y = solve_lp_population_gpu(
        Ap, bp, lbp, ubp, Cp, minimize_flags=flags,
        max_iters=(gpu_iters or 300), Y0=Y0, return_duals=True,
        device=device,
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
            duals[int(dim)] = (_safe_f16(Y[j, :mi, c]),
                               _safe_f16(Y[j, :mi, kd_max + c]))
        results[i] = (lb, ub, duals)
    return results


def contract_input_boxes(systems, boxes, n_dims, gpu_iters=None, device=None):
    """
    Batched NS-certified domain contraction (nnenum's contract-on-split, on the
    gradient-dual engine): for each constraint system ``(C, d)`` over its
    predicate box, compute a certified enclosure of the range of the LEADING
    ``n_dims`` predicate variables (the input dims -- the alpha prefix is
    preserved through the reach, so these are always the first columns) over
    ``{C alpha <= d, box}``, and intersect it with the old box.

    All systems share ONE population GPU call -- the per-call cost is launch-
    bound, not size-bound, so batch every pending split of a frontier wave into
    a single call rather than calling per split. Heterogeneous row counts and
    variable counts are zero-padded (inert under NS). Soundness: each bound is
    an NS-certified *outward* enclosure (lo <= true min, hi >= true max, for
    ANY dual iterate), so the contracted box still contains every feasible
    point -- any downstream use is sound at any solver accuracy. ``empty=True``
    is certified infeasibility: the outward enclosure itself is empty
    (certified lo > certified hi), which is only possible when the true region
    is empty -- a free, sound prune of that child.

    Args:
        systems: list of ``(C, d)`` with C shape (m_i, nVar_i), d (m_i, 1).
        boxes: list of ``(lb, ub)`` predicate boxes aligned to ``systems``,
            each (nVar_i,) or (nVar_i, 1).
        n_dims: number of leading dims to contract (= the input star's nVar).
        gpu_iters: gradient-iteration budget (default ``CONTRACT_GPU_ITERS``).
        device: torch device for the solve (``None``: CUDA when usable, else
            CPU).

    Returns:
        list of ``(new_lb, new_ub, empty)`` per system; ``new_lb``/``new_ub``
        are the contracted leading-dims box, shape ``(n_dims, 1)``.
    """
    B = len(systems)
    m_max = max(int(C.shape[0]) for C, _ in systems) or 1
    n_max = max(np.asarray(lb).size for lb, _ in boxes)
    k = 2 * n_dims
    Ap = np.zeros((B, m_max, n_max))
    bp = np.zeros((B, m_max))
    lbp = np.zeros((B, n_max))   # zero-width padding is inert under NS
    ubp = np.zeros((B, n_max))
    Cp = np.zeros((B, n_max, k))
    base = []
    for j, ((C, d), (lb, ub)) in enumerate(zip(systems, boxes)):
        lbf = np.asarray(lb, dtype=np.float64).flatten()
        ubf = np.asarray(ub, dtype=np.float64).flatten()
        if not (np.all(np.isfinite(lbf)) and np.all(np.isfinite(ubf))):
            # NS certificates are undefined on infinite intervals (0 * inf =
            # NaN would silently poison the certified bounds and every
            # downstream prune) -- fail loudly instead.
            raise ValueError(
                "contract_input_boxes requires finite predicate boxes; "
                "finitize the star first via "
                "n2v.sets.predicate_box.ensure_finite_predicate_box"
            )
        base.append((lbf[:n_dims].copy(), ubf[:n_dims].copy()))
        ni = lbf.size
        lbp[j, :ni] = lbf
        ubp[j, :ni] = ubf
        mi = int(C.shape[0])
        if mi:
            Ap[j, :mi, :ni] = C
            bp[j, :mi] = np.asarray(d).flatten()
        for i in range(n_dims):
            Cp[j, i, i] = 1.0            # min alpha_i
            Cp[j, i, n_dims + i] = 1.0   # max alpha_i
    flags = [True] * n_dims + [False] * n_dims
    bounds = solve_lp_population_gpu(
        Ap, bp, lbp, ubp, Cp, minimize_flags=flags,
        max_iters=(gpu_iters or CONTRACT_GPU_ITERS), device=device,
    )

    out = []
    for j in range(B):
        base_lb, base_ub = base[j]
        new_lb = np.maximum(base_lb, bounds[j, :n_dims])
        new_ub = np.minimum(base_ub, bounds[j, n_dims:2 * n_dims])
        empty = bool(np.any(new_lb > new_ub))
        out.append((new_lb.reshape(-1, 1), new_ub.reshape(-1, 1), empty))
    return out
