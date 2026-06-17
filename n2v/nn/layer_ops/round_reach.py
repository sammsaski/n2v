"""
Rounding operation reachability (ONNX Round / Ceil / Floor).

round/ceil/floor are monotone, piecewise-constant step functions:
z = f(y) = y + e with a bounded rounding error e
(round: e in [-1/2, 1/2]; ceil: [0, 1]; floor: [-1, 0]), and
z in [f(l), f(u)] for y in [l, u].

The star relaxation introduces one fresh predicate variable per varying
dimension, constrained by z - y in [e_lo, e_hi] and predicate-bounded by
[f(l), f(u)]. Dimensions whose output is constant (f(l) == f(u), which
includes degenerate inputs) map exactly. Sound over-approximation.
"""

import numpy as np
from typing import List, Tuple

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar

# Value range of f(y) - y for each rounding mode. torch.round and
# np.round both round half to even, so the +/-1/2 bound is tight.
_ROUND_ERROR = {
    'round': (-0.5, 0.5),
    'ceil': (0.0, 1.0),
    'floor': (-1.0, 0.0),
}


def _np_fn_and_error(layer) -> Tuple:
    """Map an OnnxRound module to its numpy function and error interval."""
    name = getattr(layer.round_function, '__name__', None)
    if name not in _ROUND_ERROR:
        raise NotImplementedError(
            f"rounding op {name!r} not supported for reachability")
    return getattr(np, name), _ROUND_ERROR[name]


def round_star(layer, input_sets: List, lp_solver: str = 'default') -> List:
    """Star reachability for Round/Ceil/Floor."""
    npf, (e_lo, e_hi) = _np_fn_and_error(layer)
    output_sets = []
    for s in input_sets:
        star = s.to_star() if isinstance(s, ImageStar) else s
        N, n = star.dim, star.nVar

        lbs = np.zeros(N)
        ubs = np.zeros(N)
        for i in range(N):
            l_val, u_val = star.get_range(i, lp_solver)
            if l_val is None or u_val is None:
                raise ValueError(
                    f"LP solver returned None for dimension {i} of "
                    f"rounding-op input. Star may be infeasible.")
            lbs[i], ubs[i] = l_val, u_val

        z_lb = npf(lbs)
        z_ub = npf(ubs)
        const_mask = np.abs(z_ub - z_lb) < 1e-12
        vary_idx = np.where(~const_mask)[0]
        m = len(vary_idx)

        V1 = np.zeros((N, n + 1))
        V1[const_mask, 0] = z_lb[const_mask]
        V2 = np.zeros((N, m))
        V2[vary_idx, np.arange(m)] = 1.0
        new_V = np.hstack([V1, V2])

        C_old = (np.asarray(star.C, dtype=np.float64).reshape(-1, n)
                 if np.asarray(star.C).size else np.zeros((0, n)))
        d_old = (np.asarray(star.d, dtype=np.float64).reshape(-1, 1)
                 if np.asarray(star.d).size else np.zeros((0, 1)))
        blocks_C = []
        blocks_d = []
        if C_old.shape[0]:
            blocks_C.append(
                np.hstack([C_old, np.zeros((C_old.shape[0], m))]))
            blocks_d.append(d_old)

        # z_i - y_i <= e_hi and y_i - z_i <= -e_lo for varying dims
        Vy = star.V[vary_idx, 1:n + 1]
        cy = star.V[vary_idx, 0]
        rows = []
        rhs = []
        for j in range(m):
            r = np.zeros(n + m)
            r[:n] = -Vy[j]
            r[n + j] = 1.0
            rows.append(r)
            rhs.append(e_hi + cy[j])
            r = np.zeros(n + m)
            r[:n] = Vy[j]
            r[n + j] = -1.0
            rows.append(r)
            rhs.append(-e_lo - cy[j])
        if rows:
            blocks_C.append(np.array(rows))
            blocks_d.append(np.array(rhs).reshape(-1, 1))

        new_C = np.vstack(blocks_C) if blocks_C else np.zeros((0, n + m))
        new_d = np.vstack(blocks_d) if blocks_d else np.zeros((0, 1))

        if star.predicate_lb is not None:
            old_lb = np.asarray(
                star.predicate_lb, dtype=np.float64).reshape(-1, 1)
            old_ub = np.asarray(
                star.predicate_ub, dtype=np.float64).reshape(-1, 1)
        else:
            old_lb = np.full((n, 1), -np.inf)
            old_ub = np.full((n, 1), np.inf)
        new_pred_lb = np.vstack([old_lb, z_lb[vary_idx].reshape(-1, 1)])
        new_pred_ub = np.vstack([old_ub, z_ub[vary_idx].reshape(-1, 1)])

        result = Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)
        if isinstance(s, ImageStar):
            result = result.to_image_star(
                s.height, s.width, s.num_channels)
        output_sets.append(result)
    return output_sets


def round_zono(layer, input_sets: List) -> List:
    """Zono reachability: z = y + e as a fresh generator per dim."""
    _, (e_lo, e_hi) = _np_fn_and_error(layer)
    mid = (e_lo + e_hi) / 2.0
    rad = (e_hi - e_lo) / 2.0
    output_sets = []
    for z in input_sets:
        c_out = np.asarray(z.c, dtype=np.float64).reshape(-1, 1) + mid
        dim = c_out.shape[0]
        V_out = np.hstack([z.V, rad * np.eye(dim)])
        output_sets.append(Zono(c_out, V_out))
    return output_sets


def round_box(layer, input_sets: List) -> List:
    """Box reachability: rounding modes are monotone non-decreasing."""
    npf, _ = _np_fn_and_error(layer)
    return [Box(npf(b.lb), npf(b.ub)) for b in input_sets]
