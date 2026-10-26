"""
LeakyReLU activation reachability operations.

Exact and approximate reachability for LeakyReLU activation.
Translated from MATLAB NNV LeakyReLU.m

LeakyReLU(x) = x       if x >= 0
             = gamma*x  if x < 0

where gamma is the negative_slope parameter (default 0.01).

Like ReLU, this is element-wise so it operates on 2D Star representation.
ImageStar inputs are converted to Star, processed, and converted back.
"""

import logging

import numpy as np
from typing import List, Optional
from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.utils.lp_solver_enum import LPSolver

# Profiler hooks (no-op when profiling is disabled)
from n2v.profiling import count
from n2v.nn.layer_ops._profiling import (
    record_layer_neurons,
    record_classification,
)
from n2v.nn.layer_ops.relu_reach import (
    _get_maxs,
    _get_mins,
    _intersect_precomputed,
    _phase_range,
    _phase_sliver_star,
    _thin_neuron_side,
)
from n2v.utils.ns_certificate import outward_interval_slack

logger = logging.getLogger(__name__)


def _preserve_imagestar_type(original: Star, new_star: Star) -> Star:
    """If original was ImageStar, convert new_star back."""
    if isinstance(original, ImageStar):
        return new_star.to_image_star(original.height, original.width, original.num_channels)
    return new_star


def leakyrelu_star_exact(
    input_stars: List[Star],
    gamma: float = 0.01,
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    verbose: bool = False,
    precomputed_bounds: tuple = None,
) -> List[Star]:
    """
    Exact reachability for LeakyReLU using Star sets.

    Args:
        input_stars: List of input Star sets
        gamma: Negative slope parameter
        lp_solver: LP solver to use
        verbose: Display progress
        precomputed_bounds: Optional (lb, ub) from Zono pre-pass

    Returns:
        List of output Star sets (may be more than input due to splitting)
    """
    # Profiler: static neuron count, once per layer (no-op when disabled)
    record_layer_neurons(input_stars)

    output_stars = []
    for star in input_stars:
        star_2d = star.to_star() if isinstance(star, ImageStar) else star
        result = _leakyrelu_single_star_exact(star_2d, gamma, lp_solver, verbose, precomputed_bounds)
        result = [_preserve_imagestar_type(star, s) for s in result]
        output_stars.extend(result)
    return output_stars


def _leakyrelu_single_star_exact(
    I: Star,
    gamma: float,
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    verbose: bool = False,
    precomputed_bounds: tuple = None,
) -> List[Star]:
    """Exact LeakyReLU for a single Star. Split on crossing neurons."""
    if I is None or I.dim == 0:
        return []

    lb, ub = I.estimate_ranges()
    if lb is None or ub is None:
        return []

    # Refine bounds with precomputed pre-pass bounds if available. As in
    # relu_reach._relu_single_star_exact, these round-to-nearest bounds only
    # save LPs here: they are widened by the LP pad before they may decide
    # a phase.
    if precomputed_bounds is not None:
        lb, ub = _intersect_precomputed(lb, ub, precomputed_bounds, pad=True)

    # Scale neurons that are always inactive (ub <= 0)
    inactive_map = np.where(ub.flatten() <= 0)[0]
    V = I.V.copy()
    V[inactive_map, :] = gamma * V[inactive_map, :]

    if I.Z is not None:
        c1 = I.Z.c.copy()
        c1[inactive_map] = gamma * c1[inactive_map]
        V1 = I.Z.V.copy()
        V1[inactive_map, :] = gamma * V1[inactive_map, :]
        new_Z = Zono(c1, V1)
    else:
        new_Z = None

    current_stars = [Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)]

    # Split on neurons crossing zero
    split_map = np.where((lb.flatten() < 0) & (ub.flatten() > 0))[0]

    # Profiler: classification, summed across the per-star population (inactive =
    # always-negative-slope; unstable = crossing). n_neurons at the layer level.
    record_classification(I.dim, len(inactive_map), len(split_map))

    for i, neuron_idx in enumerate(split_map):
        if verbose:
            logger.debug(f'Exact LeakyReLU_{neuron_idx} ({i+1}/{len(split_map)})')
        new_stars = []
        for star in current_stars:
            split_result = _step_leakyrelu(star, neuron_idx, gamma, lp_solver)
            new_stars.extend(split_result)
        current_stars = new_stars

    return current_stars


def _step_leakyrelu(I: Star, index: int, gamma: float, lp_solver: "LPSolver | str" = LPSolver.DEFAULT) -> List[Star]:
    """Split a single neuron for LeakyReLU (exact step reach).

    Phases are decided as in ``relu_reach._step_relu``: from the padded LP
    range, with a star dropped only when certified empty, and a raw bound
    within its pad of 0 kept in one star with a sliver variable when the
    other bound is outside its pad (a neuron whose whole range lies within
    the pads is split, which is exact, unless one side is certified empty,
    ``relu_reach._thin_neuron_side``). For the band on the positive side,
    ``f(x) = gamma*x + (1-gamma)*max(x, 0)``; on the negative side
    ``f(x) = x + (1-gamma)*max(-x, 0)``.
    """
    rng = _phase_range(I, index, lp_solver)
    if rng is None:
        return []
    xmin, xmax, pad_lo, pad_hi = rng
    max_in_band = xmax <= 2.0 * pad_hi
    min_in_band = -xmin <= 2.0 * pad_lo

    if xmin >= 0:
        count("n_resolved", 1)
        return [I]

    elif xmax <= 0:
        # Always inactive — scale by gamma
        count("n_resolved", 1)
        return [_scale_star_row(I, index, gamma)]

    side = None
    if max_in_band and min_in_band:
        side = _thin_neuron_side(I, index, xmin, xmax, pad_lo, pad_hi, lp_solver)

    if side == 1:
        count("n_resolved", 1)
        return [I]

    elif side == -1:
        count("n_resolved", 1)
        return [_scale_star_row(I, index, gamma)]

    elif max_in_band and not min_in_band:
        # Raw max within its pad of 0: gamma*x + t, t in (1-gamma)*[0, xmax]
        count("n_resolved", 1)
        return [_phase_sliver_star(I, index, gamma, (1.0 - gamma) * xmax)]

    elif min_in_band and not max_in_band:
        # Raw min within its pad of 0: x + t, t in (1-gamma)*[0, -xmin]
        count("n_resolved", 1)
        return [_phase_sliver_star(I, index, 1.0, (1.0 - gamma) * (-xmin))]

    else:
        # Split into two cases
        count("n_split", 1)
        c = I.V[index, 0]
        V_row = I.V[index, 1:I.nVar + 1].reshape(1, -1)

        # Case 1: x[index] < 0 (inactive) — add constraint V*alpha <= -c, scale by gamma
        new_C1 = np.vstack([I.C, V_row])
        new_d1 = np.vstack([I.d, -c * np.ones((1, 1))])
        new_V1 = I.V.copy()
        new_V1[index, :] = gamma * new_V1[index, :]

        if I.Z is not None:
            c1 = I.Z.c.copy()
            c1[index] = gamma * c1[index]
            V1 = I.Z.V.copy()
            V1[index, :] = gamma * V1[index, :]
            new_Z1 = Zono(c1, V1)
        else:
            new_Z1 = None

        S1 = Star(new_V1, new_C1, new_d1, I.predicate_lb, I.predicate_ub, outer_zono=new_Z1)

        # Case 2: x[index] >= 0 (active) — add constraint -V*alpha <= c, keep V
        new_C2 = np.vstack([I.C, -V_row])
        new_d2 = np.vstack([I.d, c * np.ones((1, 1))])
        S2 = Star(I.V, new_C2, new_d2, I.predicate_lb, I.predicate_ub, outer_zono=I.Z)

        return [S1, S2]


def _scale_star_row(I: Star, index: int, gamma: float) -> Star:
    """``I`` with row ``index`` (and its outer zonotope row) scaled by gamma."""
    new_V = I.V.copy()
    new_V[index, :] = gamma * new_V[index, :]

    if I.Z is not None:
        new_c = I.Z.c.copy()
        new_c[index] = gamma * new_c[index]
        new_V_z = I.Z.V.copy()
        new_V_z[index, :] = gamma * new_V_z[index, :]
        new_Z = Zono(new_c, new_V_z)
    else:
        new_Z = None

    return Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)


def leakyrelu_star_approx(
    input_stars: List[Star],
    gamma: float = 0.01,
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    precomputed_bounds: tuple = None,
) -> List[Star]:
    """
    Approximate LeakyReLU reachability using modified triangle relaxation.

    For crossing neurons with bounds [lb, ub]:
      y >= gamma * x
      y >= x
      y <= a*(x - lb) + gamma*lb,  where a = (ub - gamma*lb)/(ub - lb)

    Args:
        input_stars: List of input Stars
        gamma: Negative slope
        lp_solver: LP solver
        precomputed_bounds: Optional (lb, ub) from Zono pre-pass

    Returns:
        List of output Stars (no splitting, same count as input)
    """
    # Profiler: static neuron count, once per layer (no-op when disabled)
    record_layer_neurons(input_stars)

    output_stars = []
    for star in input_stars:
        star_2d = star.to_star() if isinstance(star, ImageStar) else star
        result = _leakyrelu_single_star_approx(star_2d, gamma, lp_solver, precomputed_bounds)
        if result is not None:
            result = _preserve_imagestar_type(star, result)
            output_stars.append(result)
    return output_stars


def _leakyrelu_single_star_approx(
    I: Star,
    gamma: float,
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    precomputed_bounds: tuple = None,
) -> Optional[Star]:
    """Approximate LeakyReLU for a single Star."""
    if I is None or I.dim == 0:
        return None

    lb_est, ub_est = I.estimate_ranges()
    if lb_est is None or ub_est is None:
        return None

    lb_est = lb_est.flatten()
    ub_est = ub_est.flatten()

    # Refine bounds with precomputed Zono pre-pass bounds if available
    if precomputed_bounds is not None:
        lb_est, ub_est = _intersect_precomputed(lb_est, ub_est, precomputed_bounds)

    # Scale definitely inactive neurons (ub <= 0)
    inactive_map = np.where(ub_est <= 0)[0]
    V = I.V.copy()
    V[inactive_map, :] = gamma * V[inactive_map, :]

    if I.Z is not None:
        c1 = I.Z.c.copy()
        c1[inactive_map] = gamma * c1[inactive_map]
        V1 = I.Z.V.copy()
        V1[inactive_map, :] = gamma * V1[inactive_map, :]
        new_Z = Zono(c1, V1)
    else:
        new_Z = None

    current_star = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)

    crossing_map_est = np.where((lb_est < 0) & (ub_est > 0))[0]
    if len(crossing_map_est) == 0:
        return current_star

    # Get tight bounds via LP for crossing neurons (padded outward; the
    # estimate where the LP returns no optimum)
    ub_tight = _get_maxs(current_star, crossing_map_est, lp_solver,
                         ub_est[crossing_map_est])

    # Neurons LP confirmed inactive
    still_positive = ub_tight > 0
    actually_inactive = ~still_positive

    if np.any(actually_inactive):
        inactive_indices = crossing_map_est[actually_inactive]
        V2 = current_star.V.copy()
        V2[inactive_indices, :] = gamma * V2[inactive_indices, :]
        if current_star.Z is not None:
            c2 = current_star.Z.c.copy()
            c2[inactive_indices] = gamma * c2[inactive_indices]
            V2_z = current_star.Z.V.copy()
            V2_z[inactive_indices, :] = gamma * V2_z[inactive_indices, :]
            new_Z2 = Zono(c2, V2_z)
        else:
            new_Z2 = None
        current_star = Star(V2, current_star.C, current_star.d,
                           current_star.predicate_lb, current_star.predicate_ub,
                           outer_zono=new_Z2)

    crossing_indices = crossing_map_est[still_positive]
    ub_for_crossing = ub_tight[still_positive]

    if len(crossing_indices) == 0:
        return current_star

    lb_for_crossing = _get_mins(current_star, crossing_indices, lp_solver,
                                lb_est[crossing_indices])

    actually_crossing = lb_for_crossing < 0
    final_indices = crossing_indices[actually_crossing]
    final_lb = lb_for_crossing[actually_crossing]
    final_ub = ub_for_crossing[actually_crossing]

    if len(final_indices) == 0:
        return current_star

    return _apply_leakyrelu_approx_multi(current_star, final_indices, final_lb, final_ub, gamma)


def _apply_leakyrelu_approx_multi(
    I: Star,
    indices: np.ndarray,
    lbs: np.ndarray,
    ubs: np.ndarray,
    gamma: float,
) -> Star:
    """
    Apply LeakyReLU triangle approximation to multiple neurons simultaneously.

    3 constraints per crossing neuron. For gamma <= 1 LeakyReLU is
    max(x, gamma*x), convex:
      y >= gamma * x     (lower bound from negative side)
      y >= x             (lower bound from positive side)
      y <= a*(x - lb) + gamma*lb,  a = (ub - gamma*lb)/(ub - lb)  (secant upper)
    For gamma > 1 it is min(x, gamma*x), concave, and every inequality flips
    (the secant becomes the lower face). y is also boxed to the function's
    range on [lb, ub]: [min(gamma*lb, 0), max(gamma*lb, ub)].
    """
    if len(indices) == 0:
        return I

    N = I.dim
    m = len(indices)
    n = I.nVar

    # New basis matrix: zero out original, add new pred vars
    V1 = I.V.copy()
    V1[indices, :] = 0

    V2 = np.zeros((N, m))
    for i, idx in enumerate(indices):
        V2[idx, i] = 1

    new_V = np.hstack([V1, V2])

    # Old constraints extended with zeros for new vars
    C0 = np.hstack([I.C, np.zeros((I.C.shape[0], m))])
    d0 = I.d

    has_box = I.predicate_lb is not None and I.predicate_ub is not None
    # +1: the two lines bound y from below and the secant from above
    # (convex, gamma <= 1); -1: the reverse (concave, gamma > 1)
    s = 1.0 if gamma <= 1.0 else -1.0

    # Constraint 1: y >= gamma * x  →  gamma*x - y <= 0
    #   gamma * V[idx, 1:n+1] * alpha - y_i <= -gamma * V[idx, 0]
    # The scaled row is stored rounded: its right-hand side moves out by a
    # rigorous bound on that rounding over the predicate box (as for C3).
    G1 = gamma * I.V[indices, 1:n+1]
    C1 = s * np.hstack([G1, -np.eye(m)])
    d1 = -s * gamma * I.V[indices, 0]
    if has_box:
        d1 = d1 + outward_interval_slack(np.abs(d1), G1, I.predicate_lb,
                                         I.predicate_ub)
    d1 = d1.reshape(-1, 1)

    # Constraint 2: y >= x  →  x - y <= 0
    #   V[idx, 1:n+1] * alpha - y_i <= -V[idx, 0]
    C2 = s * np.hstack([I.V[indices, 1:n+1], -np.eye(m)])
    d2 = -s * I.V[indices, 0:1]

    # Constraint 3: y <= a*(x - lb) + gamma*lb
    #   y - a*x <= -a*lb + gamma*lb = lb*(gamma - a)
    #   -a * V[idx, 1:n+1] * alpha + y_i <= -a*V[idx,0] + a*lb - gamma*lb + gamma*V[idx,0]
    # Simplified: a = (ub - gamma*lb) / (ub - lb)
    # SOUNDNESS: as in relu_reach._apply_triangle_approx_multi, for gamma <= 1
    # any slope a >= (ub - gamma*lb)/(ub - lb) keeps the face on or above
    # (ub, ub) and the kink, so the rounded quotient is raised by 8 units of
    # roundoff (2^-50 relative); for gamma > 1 the face is a lower bound and
    # the slope is lowered instead. lb < 0 < ub, so no epsilon in the
    # denominator (the old ``+ 1e-10`` lowered the face below (ub, ub)).
    lbs = np.asarray(lbs, dtype=np.float64).reshape(-1)
    ubs = np.asarray(ubs, dtype=np.float64).reshape(-1)
    a = ((ubs - gamma * lbs) / (ubs - lbs)) * (1.0 + s * 2.0 ** -50)
    b = a * lbs - gamma * lbs  # -a*lb + gamma*lb negated = a*lb - gamma*lb
    G3 = a.reshape(-1, 1) * I.V[indices, 1:n+1]
    C3 = s * np.hstack([-G3, np.eye(m)])
    # Rigorous bound on the rounding of the stored row over the predicate box.
    d3 = s * (a * I.V[indices, 0] - b)
    if has_box:
        d3 = d3 + outward_interval_slack(
            np.abs(a * I.V[indices, 0]) + np.abs(a * lbs) + np.abs(gamma * lbs),
            G3, I.predicate_lb, I.predicate_ub)
    d3 = d3.reshape(-1, 1)

    new_C = np.vstack([C0, C1, C2, C3])
    new_d = np.vstack([d0, d1, d2, d3])

    # Predicate bounds: the range of LeakyReLU on [lb, ub], attained at lb, 0
    # or ub (the product gamma*lb rounded outward)
    g_lo = np.nextafter(gamma * lbs, -np.inf)
    g_hi = np.nextafter(gamma * lbs, np.inf)
    y_lb = np.minimum(g_lo, 0.0)
    y_ub = np.maximum(g_hi, ubs)
    new_pred_lb = np.vstack([I.predicate_lb, y_lb.reshape(-1, 1)]) if I.predicate_lb is not None else None
    new_pred_ub = np.vstack([I.predicate_ub, y_ub.reshape(-1, 1)]) if I.predicate_ub is not None else None

    return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub, outer_zono=None)


def leakyrelu_zono_approx(input_zonos: List[Zono], gamma: float = 0.01) -> List[Zono]:
    """Approximate LeakyReLU for Zonotopes, preserving ImageZono type."""
    from n2v.sets.image_zono import ImageZono

    output = []
    for z in input_zonos:
        result = _leakyrelu_single_zono(z, gamma)
        if isinstance(z, ImageZono) and not isinstance(result, ImageZono):
            result = ImageZono(result.c, result.V, z.height, z.width, z.num_channels)
        output.append(result)
    return output


def _leakyrelu_single_zono(I: Zono, gamma: float) -> Zono:
    """Approximate LeakyReLU for a single Zonotope."""
    lb, ub = I.get_bounds()
    new_c = I.c.copy()
    new_V = I.V.copy()
    n_orig = I.V.shape[1]

    for i in range(I.dim):
        li, ui = lb[i, 0], ub[i, 0]

        if ui <= 0:
            # Always inactive — scale by gamma
            new_c[i] = gamma * I.c[i, 0]
            new_V[i, :n_orig] = gamma * I.V[i, :]

        elif li >= 0:
            # Always active — no change
            pass

        else:
            # Crosses zero — over-approximate with the exact envelope
            # band (issue #16). The secant through (li, gamma*li) and
            # (ui, ui) has slope ``a`` and intercept ``b_u``; since
            # LeakyReLU is piecewise linear with its kink at 0, the
            # residual f(x) - a*x on [li, ui] attains its extremes at
            # the kink (value 0) and at the endpoints (value b_u):
            #
            #     f(x) - a*x  in  [min(0, b_u), max(0, b_u)]
            #
            # Centre the affine part mid-band and add ONE error
            # generator of radius |b_u| / 2 (the DeepZ form; the band is
            # the tightest parallel envelope). Handles gamma > 1 too,
            # where b_u < 0 and the band flips sides.
            #
            # The previous code's ``shift`` was algebraically equal to
            # b_u (the affine part WAS the upper secant) and both of its
            # ``error`` formulas evaluated to exactly zero, so no
            # generator was ever added: the output collapsed to the
            # secant line and excluded true outputs for correlated
            # inputs (issue #16 repro: f(0,0) = (0,0) excluded by 0.99).
            a = (ui - gamma * li) / (ui - li)
            b_u = gamma * li - a * li

            new_c[i] = a * I.c[i, 0] + 0.5 * b_u
            new_V[i, :n_orig] = a * I.V[i, :]

            if b_u != 0.0:
                error_gen = np.zeros((new_V.shape[0], 1))
                error_gen[i] = 0.5 * abs(b_u)
                new_V = np.hstack([new_V, error_gen])

    return Zono(new_c, new_V)


def leakyrelu_box(input_boxes: List, gamma: float = 0.01) -> List:
    """LeakyReLU for Boxes.

    LeakyReLU is piecewise linear with its kink at 0, so on [lb, ub] its range
    is spanned by f(lb), f(ub) and, when lb < 0 < ub, f(0) = 0. This holds for
    any slope (for gamma < 0, f(lb) = gamma*lb can exceed ub and the old
    [gamma*lb, ub] left out f(0); an all-negative box came out inverted). The
    products gamma*x are rounded outward, as for the Star predicate bounds.
    """
    from n2v.sets import Box
    output = []
    for box in input_boxes:
        lb = np.asarray(box.lb, dtype=np.float64)
        ub = np.asarray(box.ub, dtype=np.float64)
        f_lo, f_hi = [], []
        for v in (lb, ub):
            neg = v < 0
            f_lo.append(np.where(neg, np.nextafter(gamma * v, -np.inf), v))
            f_hi.append(np.where(neg, np.nextafter(gamma * v, np.inf), v))
        new_lb = np.minimum(f_lo[0], f_lo[1])
        new_ub = np.maximum(f_hi[0], f_hi[1])
        crossing = (lb < 0) & (ub > 0)
        new_lb = np.where(crossing, np.minimum(new_lb, 0.0), new_lb)
        new_ub = np.where(crossing, np.maximum(new_ub, 0.0), new_ub)
        output.append(Box(new_lb, new_ub))
    return output
