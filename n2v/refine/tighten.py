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

Compared with ``Star.estimate_ranges`` (the predicate-box interval that ignores
C/d), these bounds are tighter-or-equal and still sound, so they stabilise more
neurons and tighten the triangle relaxation -> fewer splits.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from n2v.sets.star import Star
from n2v.utils.lp_solver_enum import LPSolver
from n2v.utils.lpsolver_gpu import solve_lp_batch_gpu


def _lp_ranges(S: Star, backend: str, dims: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched min/max of ``S``'s output dims ``dims`` over the predicate polytope.
    Returns ``(lo, hi)`` aligned to ``dims`` (each already offset by the center).
    ``lp_cpu`` uses Star's batched HiGHS primitive; ``lp_gpu`` uses batched
    PDHG + Neumaier-Shcherbina (outward). Solving only ``dims`` (rather than all
    output dims) is what makes the OBBT pre-filter a real LP-count saving.
    """
    dims = np.asarray(dims, dtype=int)
    c = S.V[:, 0]
    gens = S.V[:, 1:].astype(np.float64)

    if backend == "lp_cpu":
        objectives, flags = [], []
        for i in dims:
            objectives.extend([gens[i], gens[i]])
            flags.extend([True, False])
        res = S._solve_lp_batch(objectives, flags, LPSolver.DEFAULT)
    elif backend == "lp_gpu":
        # NS requires finite predicate bounds; without them fall back to exact CPU.
        if S.predicate_lb is None or S.predicate_ub is None:
            return _lp_ranges(S, "lp_cpu", dims)
        objectives, flags = [], []
        for i in dims:
            objectives.extend([gens[i], gens[i]])
            flags.extend([True, False])
        A = S.C if S.C.size > 0 else None
        b = S.d.flatten() if S.d.size > 0 else None
        res = solve_lp_batch_gpu(
            objectives, A=A, b=b, lb=S.predicate_lb.flatten(),
            ub=S.predicate_ub.flatten(), minimize_flags=flags,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'lp_cpu'|'lp_gpu'")

    lo = np.empty(dims.size, dtype=np.float64)
    hi = np.empty(dims.size, dtype=np.float64)
    for j, i in enumerate(dims):
        vmin, vmax = res[2 * j], res[2 * j + 1]
        if vmin is None or vmax is None:
            a, b2 = S.estimate_range(int(i))         # sound box fallback
            lo[j], hi[j] = a, b2
        else:
            lo[j], hi[j] = vmin + c[i], vmax + c[i]
    return lo, hi


def neuron_bounds(
    S: Star, backend: str = "lp_cpu", prefilter: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
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
        return col.copy(), col.copy()

    lb, ub = S.estimate_ranges()
    lb = lb.copy()                          # don't alias S.state_lb/ub (estimate_ranges caches them)
    ub = ub.copy()
    if prefilter:
        lbf, ubf = lb.flatten(), ub.flatten()
        dims = np.flatnonzero((lbf < 0.0) & (ubf > 0.0))
    else:
        dims = np.arange(S.dim)

    if dims.size == 0:
        return lb, ub                       # everything box-stable -> no LP at all

    lo, hi = _lp_ranges(S, backend, dims)
    # Intersect with the box: tighter sound bound, no-op for exact CPU optima,
    # clips NS-outward GPU overshoot -> backend-independent, prefilter-invariant.
    lb[dims, 0] = np.maximum(lo, lb[dims, 0])
    ub[dims, 0] = np.minimum(hi, ub[dims, 0])
    return lb, ub
