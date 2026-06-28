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
from n2v.utils.lpsolver_gpu import solve_lp_batch_gpu


def neuron_bounds(S: Star, backend: str = "lp_cpu") -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-dimension range of ``S`` over its predicate polytope ``P``.

    Returns ``(lb, ub)`` each shape ``(dim, 1)`` (matching ``estimate_ranges``).
    For ``lp_gpu`` the values are NS-certified *outward* (lb <= true min,
    ub >= true max), so any downstream stabilisation stays sound.
    """
    if S.nVar == 0:
        col = S.V[:, 0].reshape(-1, 1)
        return col.copy(), col.copy()

    if backend == "lp_cpu":
        # The exact LP-over-P range; Star already implements this (batched HiGHS,
        # C/d-aware, handles unbounded predicates). Reuse it rather than duplicate.
        return S.get_ranges()

    if backend != "lp_gpu":
        raise ValueError(f"unknown backend {backend!r}; expected 'lp_cpu'|'lp_gpu'")

    # GPU path: NS requires finite predicate bounds; without them, fall back to
    # the (sound) CPU LP rather than emit an uncertified result.
    if S.predicate_lb is None or S.predicate_ub is None:
        return S.get_ranges()

    dim = S.dim
    c = S.V[:, 0]
    gens = S.V[:, 1:].astype(np.float64)               # (dim, nVar)
    objectives, flags = [], []
    for i in range(dim):
        objectives.append(gens[i])
        flags.append(True)   # minimize
        objectives.append(gens[i])
        flags.append(False)  # maximize
    A = S.C if S.C.size > 0 else None
    b = S.d.flatten() if S.d.size > 0 else None
    plb = S.predicate_lb.flatten()
    pub = S.predicate_ub.flatten()

    vals = solve_lp_batch_gpu(objectives, A=A, b=b, lb=plb, ub=pub, minimize_flags=flags)

    lb = np.empty((dim, 1), dtype=np.float64)
    ub = np.empty((dim, 1), dtype=np.float64)
    for i in range(dim):
        vmin, vmax = vals[2 * i], vals[2 * i + 1]
        if vmin is None or vmax is None:
            lo, hi = S.estimate_range(i)              # sound box fallback
            lb[i, 0], ub[i, 0] = lo, hi
        else:
            lb[i, 0], ub[i, 0] = vmin + c[i], vmax + c[i]
    return lb, ub
