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

import numpy as np
from typing import List, Optional
from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar


def _preserve_imagestar_type(original: Star, new_star: Star) -> Star:
    """If original was ImageStar, convert new_star back."""
    if isinstance(original, ImageStar):
        return new_star.to_image_star(original.height, original.width, original.num_channels)
    return new_star


def leakyrelu_star_exact(
    input_stars: List[Star],
    gamma: float = 0.01,
    lp_solver: str = 'default',
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
    lp_solver: str = 'default',
    verbose: bool = False,
    precomputed_bounds: tuple = None,
) -> List[Star]:
    """Exact LeakyReLU for a single Star. Split on crossing neurons."""
    if I is None or I.dim == 0:
        return []

    lb, ub = I.estimate_ranges()
    if lb is None or ub is None:
        return []

    # Refine bounds with precomputed Zono pre-pass bounds if available
    if precomputed_bounds is not None:
        pre_lb, pre_ub = precomputed_bounds
        lb = np.maximum(lb, pre_lb.reshape(lb.shape))
        ub = np.minimum(ub, pre_ub.reshape(ub.shape))

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

    for i, neuron_idx in enumerate(split_map):
        if verbose:
            print(f'Exact LeakyReLU_{neuron_idx} ({i+1}/{len(split_map)})')
        new_stars = []
        for star in current_stars:
            split_result = _step_leakyrelu(star, neuron_idx, gamma, lp_solver)
            new_stars.extend(split_result)
        current_stars = new_stars

    return current_stars


def _step_leakyrelu(I: Star, index: int, gamma: float, lp_solver: str = 'default') -> List[Star]:
    """Split a single neuron for LeakyReLU (exact step reach)."""
    xmin, xmax = I.get_range(index, lp_solver)

    if xmin is None or xmax is None:
        return []

    if xmin >= 0:
        return [I]

    elif xmax <= 0:
        # Always inactive — scale by gamma
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

        return [Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)]

    else:
        # Split into two cases
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


def leakyrelu_star_approx(
    input_stars: List[Star],
    gamma: float = 0.01,
    lp_solver: str = 'default',
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
    lp_solver: str = 'default',
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
        pre_lb, pre_ub = precomputed_bounds
        lb_est = np.maximum(lb_est, pre_lb.flatten())
        ub_est = np.minimum(ub_est, pre_ub.flatten())

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

    # Get tight bounds via LP for crossing neurons
    ub_tight = np.zeros(len(crossing_map_est))
    for i, idx in enumerate(crossing_map_est):
        _, ub_val = current_star.get_range(idx, lp_solver)
        ub_tight[i] = ub_val if ub_val is not None else ub_est[idx]

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

    lb_for_crossing = np.zeros(len(crossing_indices))
    for i, idx in enumerate(crossing_indices):
        lb_val, _ = current_star.get_range(idx, lp_solver)
        lb_for_crossing[i] = lb_val if lb_val is not None else lb_est[idx]

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

    3 constraints per crossing neuron:
      y >= gamma * x     (lower bound from negative side)
      y >= x             (lower bound from positive side)
      y <= a*(x - lb) + gamma*lb,  a = (ub - gamma*lb)/(ub - lb)  (secant upper)
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

    # Constraint 1: y >= gamma * x  →  gamma*x - y <= 0
    #   gamma * V[idx, 1:n+1] * alpha - y_i <= -gamma * V[idx, 0]
    C1 = np.hstack([gamma * I.V[indices, 1:n+1], -np.eye(m)])
    d1 = (-gamma * I.V[indices, 0:1])

    # Constraint 2: y >= x  →  x - y <= 0
    #   V[idx, 1:n+1] * alpha - y_i <= -V[idx, 0]
    C2 = np.hstack([I.V[indices, 1:n+1], -np.eye(m)])
    d2 = -I.V[indices, 0:1]

    # Constraint 3: y <= a*(x - lb) + gamma*lb
    #   y - a*x <= -a*lb + gamma*lb = lb*(gamma - a)
    #   -a * V[idx, 1:n+1] * alpha + y_i <= -a*V[idx,0] + a*lb - gamma*lb + gamma*V[idx,0]
    # Simplified: a = (ub - gamma*lb) / (ub - lb)
    a = (ubs - gamma * lbs) / (ubs - lbs + 1e-10)
    b = a * lbs - gamma * lbs  # -a*lb + gamma*lb negated = a*lb - gamma*lb
    C3 = np.hstack([-a.reshape(-1, 1) * I.V[indices, 1:n+1], np.eye(m)])
    d3 = (a * I.V[indices, 0] - b).reshape(-1, 1)

    new_C = np.vstack([C0, C1, C2, C3])
    new_d = np.vstack([d0, d1, d2, d3])

    # Predicate bounds: y_i in [gamma*lb_i, ub_i]
    new_pred_lb = np.vstack([I.predicate_lb, (gamma * lbs).reshape(-1, 1)]) if I.predicate_lb is not None else None
    new_pred_ub = np.vstack([I.predicate_ub, ubs.reshape(-1, 1)]) if I.predicate_ub is not None else None

    return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub, outer_zono=None)


def leakyrelu_zono_approx(input_zonos: List[Zono], gamma: float = 0.01) -> List[Zono]:
    """Approximate LeakyReLU for Zonotopes."""
    return [_leakyrelu_single_zono(z, gamma) for z in input_zonos]


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
            # Crosses zero — over-approximation
            # Secant slope: a = (ub - gamma*lb) / (ub - lb)
            a = (ui - gamma * li) / (ui - li) if (ui - li) != 0 else 1.0

            # y ≈ a*x + shift, where shift centers the approximation
            # Upper bound at lb: a*lb + shift_u = gamma*lb → shift_u = (gamma - a)*lb
            # Upper bound at ub: a*ub + shift_u vs ub → may exceed
            # Use midpoint of upper and lower intercepts
            y_at_lb = gamma * li
            y_at_ub = ui
            midpoint = 0.5 * (y_at_lb + y_at_ub)
            linear_at_mid = a * 0.5 * (li + ui)
            shift = midpoint - linear_at_mid

            new_c[i] = a * I.c[i, 0] + shift
            new_V[i, :n_orig] = a * I.V[i, :]

            # Error generator: half the gap between upper and lower envelopes
            error = 0.5 * abs(ui - gamma * li) - 0.5 * abs(a * (ui - li))
            if error < 0:
                error = 0.5 * max(abs(y_at_ub - (a * ui + shift)),
                                  abs(y_at_lb - (a * li + shift)))
            if error > 1e-10:
                error_gen = np.zeros((I.dim, 1))
                error_gen[i] = error
                new_V = np.hstack([new_V, error_gen])

    return Zono(new_c, new_V)


def leakyrelu_box(input_boxes: List, gamma: float = 0.01) -> List:
    """LeakyReLU for Boxes."""
    from n2v.sets import Box
    output = []
    for box in input_boxes:
        new_lb = np.minimum(box.lb, gamma * box.lb)
        new_lb = np.where(box.lb >= 0, box.lb, gamma * box.lb)
        new_ub = np.where(box.ub >= 0, box.ub, gamma * box.ub)
        # Handle crossing: lb < 0 and ub > 0
        crossing = (box.lb < 0) & (box.ub > 0)
        new_lb = np.where(crossing, gamma * box.lb, new_lb)
        new_ub = np.where(crossing, box.ub, new_ub)
        output.append(Box(new_lb, new_ub))
    return output
