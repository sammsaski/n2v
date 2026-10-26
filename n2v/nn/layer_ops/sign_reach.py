"""
Sign activation reachability.

Sign is piecewise constant: sign(x) = -1 for x<0, 0 for x=0, +1 for x>0.
Used in Binarized Neural Networks (BNNs).

Box/Zono: interval evaluation per dimension.
Star approx: a free predicate variable in [-1, 1] (or [0, 1] / [-1, 0] when a
bound is exactly 0) for each neuron whose sign is not decided.
Star exact: splitting at x = 0 into x <= 0 (-1), x = 0 (0) and x >= 0 (+1).
"""

import numpy as np
from typing import List, Optional

from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.relu_reach import _phase_range
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
    """Sign activation reachability for Star sets.

    Sign is element-wise, so ImageStar inputs are processed flat and every
    output star (exact mode may split one input into several) is converted
    back to its input's spatial shape: a flat star in the ImageStar's
    (H, W, C) order would make a following Flatten skip its reorder.
    """
    from n2v.sets.image_star import ImageStar

    output_stars = []
    for s in input_stars:
        flat = s.to_star() if isinstance(s, ImageStar) else s
        if method == 'exact':
            results = _sign_star_exact([flat], lp_solver)
        else:
            results = _sign_star_approx([flat], lp_solver)
        if isinstance(s, ImageStar):
            results = [r.to_image_star(s.height, s.width, s.num_channels)
                       for r in results]
        output_stars.extend(results)
    return output_stars


def _sign_ranges(I: Star, lp_solver: "LPSolver | str" = LPSolver.DEFAULT):
    """Enclosures ``(lo, hi)`` of every neuron of ``I`` for deciding its sign,
    or ``None`` when ``I`` is certified empty.

    Each is the star's rigorous interval bound, tightened by the padded LP
    range (``relu_reach._phase_range``) where the interval bound leaves the
    sign open. Sign is a step, so an LP error of 5e-10 at the boundary flips
    the output by 2: the raw optimum never decides a sign, and a bound of
    exactly 0 comes only from an exact computation (the interval bound, or a
    row with no nonzero generator).
    """
    lo, hi = I.estimate_ranges()
    lo = np.asarray(lo, dtype=np.float64).reshape(-1).copy()
    hi = np.asarray(hi, dtype=np.float64).reshape(-1).copy()
    decided = (lo > 0) | (hi < 0) | ((lo == 0) & (hi == 0))
    for i in np.flatnonzero(~decided):
        rng = _phase_range(I, int(i), lp_solver)
        if rng is None:
            return None
        lo[i] = np.fmax(lo[i], rng[0])
        hi[i] = np.fmin(hi[i], rng[1])
    return lo, hi


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

    For each neuron with sound bounds [l, u] (``_sign_ranges``), as in
    ``sign_box``:
      - l > 0: output = +1 (constant)
      - u < 0: output = -1 (constant)
      - l = u = 0: output = 0 (constant)
      - otherwise: a FREE predicate var y_i in [lo, hi], lo = -1 if l < 0
        else 0, hi = +1 if u > 0 else 0, so sign(0) = 0 is kept whenever 0
        is reachable. Sign is a STEP (discontinuous at 0), so no affine
        constraint relating y_i to x_i is sound across the crossing -- a
        secant coupling would exclude the true output -1 for x in (l,0) (or
        +1 for x in (0,u)), shrinking the reach set (unsound). The only
        sound single-Star relaxation is the box, with no coupling to x.
        (A tighter result needs the exact split.)

    Returns None only when ``I`` is certified empty.
    """
    N = I.dim
    n = I.nVar

    ranges = _sign_ranges(I, lp_solver)
    if ranges is None:
        return None
    lbs, ubs = ranges

    # Partition neurons (l = u = 0 is the constant 0: its row stays zero)
    constant_pos = np.where(lbs > 0)[0]
    constant_neg = np.where(ubs < 0)[0]
    varying_map = np.where(~((lbs > 0) | (ubs < 0) | ((lbs == 0) & (ubs == 0))))[0]

    m = len(varying_map)

    # Build new basis matrix
    V1 = np.zeros((N, n + 1))
    V1[constant_pos, 0] = 1.0
    V1[constant_neg, 0] = -1.0

    if m == 0:
        # All constant -- apply sign directly
        return Star(V1, I.C, I.d, I.predicate_lb, I.predicate_ub)

    V2 = np.zeros((N, m))
    for i, idx in enumerate(varying_map):
        V2[idx, i] = 1.0

    new_V = np.hstack([V1, V2])

    # Sound box relaxation only: y_i in [y_lo, y_hi], no coupling to x_i
    # (Sign is a step; any secant relating y_i to x_i is unsound).
    y_lo = np.where(lbs[varying_map] < 0, -1.0, 0.0)
    y_hi = np.where(ubs[varying_map] > 0, 1.0, 0.0)

    # Old constraints padded, then -y_i <= -y_lo_i and y_i <= y_hi_i
    C0 = np.hstack([I.C, np.zeros((I.C.shape[0], m))])
    C_new = np.zeros((2 * m, n + m))
    C_new[0::2, n:] = -np.eye(m)
    C_new[1::2, n:] = np.eye(m)
    d_new = np.empty((2 * m, 1))
    d_new[0::2, 0] = -y_lo
    d_new[1::2, 0] = y_hi
    new_C = np.vstack([C0, C_new])
    new_d = np.vstack([I.d, d_new])

    new_pred_lb = np.vstack([I.predicate_lb, y_lo.reshape(-1, 1)]) if I.predicate_lb is not None else None
    new_pred_ub = np.vstack([I.predicate_ub, y_hi.reshape(-1, 1)]) if I.predicate_ub is not None else None

    return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)


def _sign_star_exact(input_stars: List[Star], lp_solver: "LPSolver | str" = LPSolver.DEFAULT) -> List[Star]:
    """
    Sign reachability by splitting each undecided neuron at x = 0.

    A neuron whose sign is open is split three ways: x <= 0 mapped to -1,
    x = 0 (both rows) mapped to 0, and x >= 0 mapped to +1 (a side the sound
    bounds rule out is not created). The children cover the input, so the
    union is sound; it over-approximates only on the measure-zero boundary
    x = 0, which also appears in the -1 and +1 children. sign(0) = 0 needs its
    own child: the two constant halves alone give {-1, +1}, which misses the
    reachable value 0.
    """
    output_stars = []
    for I in input_stars:
        if I is None or I.dim == 0:
            continue
        result = _sign_exact_single(I, lp_solver)
        output_stars.extend(result)
    return output_stars


def _sign_exact_single(I: Star, lp_solver: "LPSolver | str" = LPSolver.DEFAULT) -> List[Star]:
    """Exact Sign for a single Star.

    Each branch carries the sign it assigns to every neuron, fixed when the
    neuron is decided (by the sound bounds, or by the side of the split the
    branch was made on); the outputs are never re-derived from LP ranges
    after the split.
    """
    N = I.dim

    ranges = _sign_ranges(I, lp_solver)
    if ranges is None:
        return []
    lbs, ubs = ranges

    signs = np.zeros(N)
    signs[lbs > 0] = 1.0
    signs[ubs < 0] = -1.0
    open_neurons = np.where(~((lbs > 0) | (ubs < 0) | ((lbs == 0) & (ubs == 0))))[0]

    # Split on each neuron whose sign is open
    branches = [(I, signs)]
    for neuron_idx in open_neurons:
        new_branches = []
        for star, star_signs in branches:
            new_branches.extend(_step_sign(
                star, int(neuron_idx), star_signs, lp_solver,
                lbs[neuron_idx], ubs[neuron_idx]))
        branches = new_branches

    # Each branch is a constant output
    output = []
    for star, star_signs in branches:
        new_V = np.zeros_like(star.V)
        new_V[:, 0] = star_signs
        output.append(Star(new_V, star.C, star.d, star.predicate_lb, star.predicate_ub))

    return output


def _step_sign(
    I: Star,
    index: int,
    signs: np.ndarray,
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    bound_lo: float = -np.inf,
    bound_hi: float = np.inf,
) -> List[tuple]:
    """Decide or split neuron ``index`` of ``I`` for Sign.

    Returns ``(star, signs)`` pairs: ``I`` itself when the sound bounds (the
    padded LP range, tightened by ``bound_lo`` / ``bound_hi``, sound bounds
    of this neuron over a star containing ``I``) decide the sign; otherwise
    the children x <= 0 (-1), x = 0 (0) and x >= 0 (+1) that the bounds
    allow. An empty list only when ``I`` is certified empty.
    """
    rng = _phase_range(I, index, lp_solver)
    if rng is None:
        return []
    lo = np.fmax(rng[0], bound_lo)
    hi = np.fmin(rng[1], bound_hi)

    def _with(star, value):
        new_signs = signs.copy()
        new_signs[index] = value
        return star, new_signs

    if lo > 0:
        return [_with(I, 1.0)]
    if hi < 0:
        return [_with(I, -1.0)]
    if lo == 0 and hi == 0:
        return [_with(I, 0.0)]

    c = I.V[index, 0]
    V = I.V[index, 1:I.nVar + 1].reshape(1, -1)
    le_C, le_d = V, -c * np.ones((1, 1))       # x[index] <= 0
    ge_C, ge_d = -V, c * np.ones((1, 1))       # x[index] >= 0

    def _cut(rows_C, rows_d):
        return Star(I.V, np.vstack([I.C] + rows_C), np.vstack([I.d] + rows_d),
                    I.predicate_lb, I.predicate_ub)

    children = []
    if lo < 0:
        # x <= 0 -> -1 (when hi == 0 the whole star already has x <= 0)
        children.append(_with(I if hi == 0 else _cut([le_C], [le_d]), -1.0))
    # x = 0 -> 0 (lo <= 0 <= hi): add each of x <= 0, x >= 0 not already
    # implied by the bounds
    zero_C, zero_d = [], []
    if hi != 0:
        zero_C.append(le_C)
        zero_d.append(le_d)
    if lo != 0:
        zero_C.append(ge_C)
        zero_d.append(ge_d)
    children.append(_with(_cut(zero_C, zero_d), 0.0))
    if hi > 0:
        # x >= 0 -> +1 (when lo == 0 the whole star already has x >= 0)
        children.append(_with(I if lo == 0 else _cut([ge_C], [ge_d]), 1.0))
    return children
