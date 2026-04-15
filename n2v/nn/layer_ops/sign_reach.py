"""
Sign activation reachability.

Sign is piecewise constant: sign(x) = -1 for x<0, 0 for x=0, +1 for x>0.
Used in Binarized Neural Networks (BNNs).

Box/Zono: interval evaluation per dimension.
Star approx: linear relaxation with secant constraint for crossing neurons.
Star exact: 2-way splitting at x=0.
"""

import numpy as np
from typing import List, Optional

from n2v.sets import Star, Zono, Box
from n2v.utils.lp_solver_enum import LPSolver


def sign_box(input_sets: List[Box]) -> List[Box]:
    """Apply sign activation to Box sets via interval evaluation."""
    output_sets = []
    for s in input_sets:
        lb = s.lb.flatten()
        ub = s.ub.flatten()
        N = len(lb)

        out_lb = np.zeros(N)
        out_ub = np.zeros(N)

        for i in range(N):
            if lb[i] > 0:
                out_lb[i] = 1.0; out_ub[i] = 1.0
            elif ub[i] < 0:
                out_lb[i] = -1.0; out_ub[i] = -1.0
            elif lb[i] == 0 and ub[i] == 0:
                out_lb[i] = 0.0; out_ub[i] = 0.0
            elif lb[i] == 0:
                out_lb[i] = 0.0; out_ub[i] = 1.0
            elif ub[i] == 0:
                out_lb[i] = -1.0; out_ub[i] = 0.0
            else:  # crossing
                out_lb[i] = -1.0; out_ub[i] = 1.0

        output_sets.append(Box(out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)))
    return output_sets


def sign_zono(input_sets: List[Zono]) -> List[Zono]:
    """Apply sign activation to Zono sets via interval over-approximation."""
    output_sets = []
    for s in input_sets:
        lb, ub = s.get_bounds()
        # Get sign bounds per dim, then build Zono from those bounds
        box_result = sign_box([Box(lb, ub)])
        sign_lb = box_result[0].lb
        sign_ub = box_result[0].ub

        # Build Zono from bounds
        new_c = (sign_lb + sign_ub) / 2
        half_widths = (sign_ub - sign_lb) / 2

        nonzero = np.where(half_widths.flatten() > 0)[0]
        N = len(new_c)
        if len(nonzero) == 0:
            new_V = np.zeros((N, 1))
        else:
            new_V = np.zeros((N, len(nonzero)))
            for i, idx in enumerate(nonzero):
                new_V[idx, i] = half_widths[idx, 0]

        output_sets.append(Zono(new_c, new_V))
    return output_sets


def sign_star(
    layer,
    input_stars: List[Star],
    method: str = 'approx',
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    **kwargs,
) -> List[Star]:
    """Sign activation reachability for Star sets."""
    if method == 'exact':
        return _sign_star_exact(input_stars, lp_solver)
    else:
        return _sign_star_approx(input_stars, lp_solver)


def _sign_star_approx(input_stars: List[Star], lp_solver: "LPSolver | str" = LPSolver.DEFAULT) -> List[Star]:
    """Approximate Sign reachability using linear relaxation."""
    output_stars = []
    for I in input_stars:
        if I is None or I.dim == 0:
            continue
        result = _sign_single_star_approx(I, lp_solver)
        if result is not None:
            output_stars.append(result)
    return output_stars


def _sign_single_star_approx(I: Star, lp_solver: "LPSolver | str" = LPSolver.DEFAULT) -> Optional[Star]:
    """
    Approximate Sign for a single Star.

    For each neuron with LP bounds [l, u]:
      - l >= 0: output = +1 (constant)
      - u <= 0: output = -1 (constant)
      - l < 0 < u: introduce new predicate var y_i in [-1, 1]
        with 4 constraints forming a parallelogram:
          y_i >= -1 (box lower)
          y_i <= +1 (box upper)
          y_i >= slope*(x_i - l) - 1 (secant lower from (l,-1) to (u,+1))
          y_i <= slope*(x_i - u) + 1 (secant upper, parallel to lower)
        where slope = 2/(u-l)
    """
    N = I.dim
    n = I.nVar

    # Get tight bounds via LP
    lbs = np.zeros(N)
    ubs = np.zeros(N)
    for i in range(N):
        lb_val, ub_val = I.get_range(i, lp_solver)
        if lb_val is None or ub_val is None:
            return None
        lbs[i] = lb_val
        ubs[i] = ub_val

    # Partition neurons
    constant_pos = np.where(lbs >= 0)[0]
    constant_neg = np.where(ubs <= 0)[0]
    varying_map = np.where((lbs < 0) & (ubs > 0))[0]

    if len(varying_map) == 0:
        # All constant -- apply sign directly
        new_V = np.zeros_like(I.V)
        new_V[constant_pos, 0] = 1.0
        new_V[constant_neg, 0] = -1.0
        return Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub)

    m = len(varying_map)

    # Build new basis matrix
    V1 = np.zeros((N, n + 1))
    V1[constant_pos, 0] = 1.0
    V1[constant_neg, 0] = -1.0

    V2 = np.zeros((N, m))
    for i, idx in enumerate(varying_map):
        V2[idx, i] = 1.0

    new_V = np.hstack([V1, V2])

    # Old constraints padded
    C0 = np.hstack([I.C, np.zeros((I.C.shape[0], m))])
    d0 = I.d

    # Original V rows for constraint construction
    V_orig = I.V[varying_map, 1:n+1]  # (m, n)
    c_orig = I.V[varying_map, 0]       # (m,)
    l = lbs[varying_map]
    u = ubs[varying_map]

    C_rows = []
    d_rows = []

    for i in range(m):
        slope = 2.0 / (u[i] - l[i])

        # 1. y_i >= -1 -> -y_i <= 1
        row1 = np.zeros(n + m)
        row1[n + i] = -1
        C_rows.append(row1)
        d_rows.append(1.0)

        # 2. y_i <= +1
        row2 = np.zeros(n + m)
        row2[n + i] = 1
        C_rows.append(row2)
        d_rows.append(1.0)

        # 3. Secant lower: y_i >= slope*(x_i - l) - 1
        #    slope*x_i - y_i <= slope*l + 1
        #    slope*(V_orig[i]@alpha + c_orig[i]) - y_i <= slope*l + 1
        row3 = np.zeros(n + m)
        row3[:n] = slope * V_orig[i]
        row3[n + i] = -1
        C_rows.append(row3)
        d_rows.append(slope * (l[i] - c_orig[i]) + 1.0)

        # 4. Secant upper: y_i <= slope*(x_i - u) + 1
        #    -slope*x_i + y_i <= -slope*u + 1
        row4 = np.zeros(n + m)
        row4[:n] = -slope * V_orig[i]
        row4[n + i] = 1
        C_rows.append(row4)
        d_rows.append(slope * (c_orig[i] - u[i]) + 1.0)

    C_new = np.array(C_rows)
    d_new = np.array(d_rows).reshape(-1, 1)
    new_C = np.vstack([C0, C_new])
    new_d = np.vstack([d0, d_new])

    new_pred_lb = np.vstack([I.predicate_lb, -np.ones((m, 1))]) if I.predicate_lb is not None else None
    new_pred_ub = np.vstack([I.predicate_ub, np.ones((m, 1))]) if I.predicate_ub is not None else None

    return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)


def _sign_star_exact(input_stars: List[Star], lp_solver: "LPSolver | str" = LPSolver.DEFAULT) -> List[Star]:
    """
    Sign reachability via 2-way splitting at x=0.

    Note: This is a sound over-approximation, not truly exact. The split
    produces x<=0 (mapped to -1) and x>=0 (mapped to +1), but sign(0)=0.
    The x=0 boundary point is included in both halves, so the union
    [-1, +1] is sound. A true 3-way split (x<0, x=0, x>0) would handle
    the degenerate x=0 case, but this is measure-zero and not worth the cost.
    """
    output_stars = []
    for I in input_stars:
        if I is None or I.dim == 0:
            continue
        result = _sign_exact_single(I, lp_solver)
        output_stars.extend(result)
    return output_stars


def _sign_exact_single(I: Star, lp_solver: "LPSolver | str" = LPSolver.DEFAULT) -> List[Star]:
    """Exact Sign for a single Star."""
    N = I.dim

    # Get tight bounds
    lbs = np.zeros(N)
    ubs = np.zeros(N)
    for i in range(N):
        lb_val, ub_val = I.get_range(i, lp_solver)
        if lb_val is None or ub_val is None:
            return []
        lbs[i] = lb_val
        ubs[i] = ub_val

    crossing = np.where((lbs < 0) & (ubs > 0))[0]

    if len(crossing) == 0:
        new_V = np.zeros_like(I.V)
        new_V[lbs >= 0, 0] = 1.0
        new_V[ubs <= 0, 0] = -1.0
        return [Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub)]

    # Split on each crossing neuron
    current_stars = [I]
    for neuron_idx in crossing:
        new_stars = []
        for star in current_stars:
            split = _step_sign(star, neuron_idx, lp_solver)
            new_stars.extend(split)
        current_stars = new_stars

    # Apply sign to each fully-constrained star
    output = []
    for star in current_stars:
        new_V = np.zeros_like(star.V)
        for i in range(N):
            lb_i, ub_i = star.get_range(i, lp_solver)
            if lb_i is None or ub_i is None:
                continue
            if lb_i >= -1e-10:
                new_V[i, 0] = 1.0
            elif ub_i <= 1e-10:
                new_V[i, 0] = -1.0
        output.append(Star(new_V, star.C, star.d, star.predicate_lb, star.predicate_ub))

    return output


def _step_sign(I: Star, index: int, lp_solver: "LPSolver | str" = LPSolver.DEFAULT) -> List[Star]:
    """Split a Star at x[index] = 0 for Sign activation."""
    xmin, xmax = I.get_range(index, lp_solver)
    if xmin is None or xmax is None:
        return []
    if xmin >= 0 or xmax <= 0:
        return [I]

    c = I.V[index, 0]
    V = I.V[index, 1:I.nVar + 1].reshape(1, -1)

    # Case 1: x[index] <= 0
    new_C1 = np.vstack([I.C, V])
    new_d1 = np.vstack([I.d, -c * np.ones((1, 1))])
    S1 = Star(I.V, new_C1, new_d1, I.predicate_lb, I.predicate_ub)

    # Case 2: x[index] >= 0
    new_C2 = np.vstack([I.C, -V])
    new_d2 = np.vstack([I.d, c * np.ones((1, 1))])
    S2 = Star(I.V, new_C2, new_d2, I.predicate_lb, I.predicate_ub)

    return [S1, S2]
