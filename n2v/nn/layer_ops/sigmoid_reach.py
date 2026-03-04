"""
Sigmoid activation reachability operations.

Approximate reachability for Sigmoid using linear relaxation.
Translated from MATLAB NNV LogSig.m (multiStepLogSig_NoSplit).

Sigmoid is smooth and non-piecewise-linear, so only approx is supported.

Also contains _s_curve_approx_multi, a shared helper for S-shaped
activation functions (Sigmoid, Tanh) parameterized by:
  - func: the activation function
  - func_deriv: its derivative
  - f0: function value at inflection point (x=0)
  - df0: derivative at inflection point (x=0)
"""

import warnings
import numpy as np
from typing import List, Optional, Callable
from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar


def _preserve_imagestar_type(original: Star, new_star: Star) -> Star:
    """If original was ImageStar, convert new_star back."""
    if isinstance(original, ImageStar):
        return new_star.to_image_star(original.height, original.width, original.num_channels)
    return new_star


# --- Sigmoid math helpers ---

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def _sigmoid_deriv(x):
    """Sigmoid derivative: sigma(x) * (1 - sigma(x))."""
    s = _sigmoid(x)
    return s * (1.0 - s)


# --- Star approx ---

def sigmoid_star_approx(
    input_stars: List[Star],
    lp_solver: str = 'default',
) -> List[Star]:
    """
    Approximate Sigmoid reachability for Star sets.

    Uses NNV's multiStepLogSig_NoSplit algorithm with tangent/secant relaxation.

    Args:
        input_stars: List of input Stars
        lp_solver: LP solver

    Returns:
        List of output Stars (no splitting)
    """
    output_stars = []
    for star in input_stars:
        star_2d = star.to_star() if isinstance(star, ImageStar) else star
        result = _s_curve_single_star_approx(
            star_2d, _sigmoid, _sigmoid_deriv, f0=0.5, df0=0.25, lp_solver=lp_solver
        )
        if result is not None:
            result = _preserve_imagestar_type(star, result)
            output_stars.append(result)
    return output_stars


# --- Zono approx ---

def sigmoid_zono_approx(input_zonos: List[Zono]) -> List[Zono]:
    """Approximate Sigmoid for Zonotopes using interval over-approximation."""
    return [_s_curve_single_zono(z, _sigmoid) for z in input_zonos]


# --- Box ---

def sigmoid_box(input_boxes: List) -> List:
    """Sigmoid for Boxes. Monotone, so just apply to bounds."""
    from n2v.sets import Box
    return [Box(_sigmoid(box.lb), _sigmoid(box.ub)) for box in input_boxes]


# =============================================================================
# Shared S-Curve Helper (used by both Sigmoid and Tanh)
# =============================================================================

def _s_curve_single_star_approx(
    I: Star,
    func: Callable,
    func_deriv: Callable,
    f0: float,
    df0: float,
    lp_solver: str = 'default',
) -> Optional[Star]:
    """
    Approximate reachability for an S-shaped activation using NNV's
    tangent/secant relaxation.

    For each neuron with LP-computed bounds [l, u]:
      Case 1 (l >= 0, convex): 3 constraints (2 tangent upper + secant lower)
      Case 2 (u <= 0, concave): 3 constraints (2 tangent lower + secant upper)
      Case 3 (mixed): 4 constraints (parallelogram using inflection point)

    Args:
        I: Input Star
        func: Activation function (vectorized numpy)
        func_deriv: Derivative of activation function (vectorized numpy)
        f0: func(0) — function value at inflection point
        df0: func'(0) — derivative at inflection point
        lp_solver: LP solver

    Returns:
        Output Star with linear relaxation constraints
    """
    if I is None or I.dim == 0:
        return None

    lb_est, ub_est = I.estimate_ranges()
    if lb_est is None or ub_est is None:
        return None

    lb_est = lb_est.flatten()
    ub_est = ub_est.flatten()

    N = I.dim
    n = I.nVar

    # Get tight bounds via LP for all neurons
    lbs = np.zeros(N)
    ubs = np.zeros(N)
    for i in range(N):
        lb_val, ub_val = I.get_range(i, lp_solver)
        lbs[i] = lb_val if lb_val is not None else lb_est[i]
        ubs[i] = ub_val if ub_val is not None else ub_est[i]

    # Partition neurons
    constant_map = np.where(np.abs(ubs - lbs) < 1e-10)[0]
    varying_map = np.where(np.abs(ubs - lbs) >= 1e-10)[0]

    if len(varying_map) == 0:
        # All constant — just apply function
        new_V = np.zeros_like(I.V)
        new_V[:, 0] = func(I.V[:, 0])
        return Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub)

    # Evaluate function and derivative at bounds
    fl = func(lbs[varying_map])
    fu = func(ubs[varying_map])
    dfl = func_deriv(lbs[varying_map])
    dfu = func_deriv(ubs[varying_map])
    l = lbs[varying_map]
    u = ubs[varying_map]

    m = len(varying_map)

    # Build new basis matrix
    V1 = np.zeros((N, n + 1))
    # Constant neurons get their function value
    V1[constant_map, 0] = func(lbs[constant_map])
    # Varying neurons will be represented by new predicate vars

    V2 = np.zeros((N, m))
    for i, idx in enumerate(varying_map):
        V2[idx, i] = 1

    new_V = np.hstack([V1, V2])

    # Old constraints
    C0 = np.hstack([I.C, np.zeros((I.C.shape[0], m))])
    d0 = I.d

    # Build constraints per neuron based on case
    # We need the ORIGINAL V rows for constraint construction (these
    # reference the predicate variables that encode the input x)
    V_orig = I.V[varying_map, 1:n+1]  # (m, n) — predicate coefficients
    c_orig = I.V[varying_map, 0]       # (m,) — center values

    # Case classification
    convex_mask = l >= 0       # Case 1
    concave_mask = u <= 0      # Case 2
    mixed_mask = (l < 0) & (u > 0)  # Case 3

    # Secant slope for all
    secant_slope = (fu - fl) / (u - l)

    # Collect constraint rows
    C_rows = []
    d_rows = []

    for i in range(m):
        # The constraint variable layout is [alpha_1..alpha_n, y_1..y_m]
        # where alpha are the original predicates and y are new output vars

        if convex_mask[i]:
            # Case 1: Convex (l >= 0)
            # Upper 1: y <= f'(l)*(x - l) + f(l)
            # x_i = V_orig[i] @ alpha + c_orig[i]
            # So: y_i <= f'(l) * (V_orig[i] @ alpha + c_orig[i] - l) + f(l)
            # y_i - f'(l) * V_orig[i] @ alpha <= f'(l) * (c_orig[i] - l) + f(l)
            row1 = np.zeros(n + m)
            row1[:n] = -dfl[i] * V_orig[i]
            row1[n + i] = 1
            rhs1 = dfl[i] * (c_orig[i] - l[i]) + fl[i]
            C_rows.append(row1)
            d_rows.append(rhs1)

            # Upper 2: y <= f'(u)*(x - u) + f(u)
            row2 = np.zeros(n + m)
            row2[:n] = -dfu[i] * V_orig[i]
            row2[n + i] = 1
            rhs2 = dfu[i] * (c_orig[i] - u[i]) + fu[i]
            C_rows.append(row2)
            d_rows.append(rhs2)

            # Lower: y >= secant  →  secant_slope*x - y <= secant_slope*l - f(l)
            # s * (V_orig @ alpha + c) - y <= s*l - f(l)
            # s * V_orig @ alpha - y <= s*l - f(l) - s*c
            row3 = np.zeros(n + m)
            row3[:n] = secant_slope[i] * V_orig[i]
            row3[n + i] = -1
            rhs3 = secant_slope[i] * (l[i] - c_orig[i]) - fl[i]
            C_rows.append(row3)
            d_rows.append(rhs3)

        elif concave_mask[i]:
            # Case 2: Concave (u <= 0)
            # Lower 1: y >= f'(l)*(x-l) + f(l)  →  f'(l)*x - y <= f'(l)*l - f(l)
            row1 = np.zeros(n + m)
            row1[:n] = dfl[i] * V_orig[i]
            row1[n + i] = -1
            rhs1 = dfl[i] * (l[i] - c_orig[i]) - fl[i]
            C_rows.append(row1)
            d_rows.append(rhs1)

            # Lower 2: y >= f'(u)*(x-u) + f(u)
            row2 = np.zeros(n + m)
            row2[:n] = dfu[i] * V_orig[i]
            row2[n + i] = -1
            rhs2 = dfu[i] * (u[i] - c_orig[i]) - fu[i]
            C_rows.append(row2)
            d_rows.append(rhs2)

            # Upper: y <= secant
            row3 = np.zeros(n + m)
            row3[:n] = -secant_slope[i] * V_orig[i]
            row3[n + i] = 1
            rhs3 = secant_slope[i] * (c_orig[i] - l[i]) + fl[i]
            C_rows.append(row3)
            d_rows.append(rhs3)

        else:
            # Case 3: Mixed (l < 0 < u) — 4 constraints
            dmin = min(dfl[i], dfu[i])

            # Constraint 1: y >= dmin*(x - l) + f(l)
            row1 = np.zeros(n + m)
            row1[:n] = dmin * V_orig[i]
            row1[n + i] = -1
            rhs1 = dmin * (l[i] - c_orig[i]) - fl[i]
            C_rows.append(row1)
            d_rows.append(rhs1)

            # Constraint 2: y <= dmin*(x - u) + f(u)
            row2 = np.zeros(n + m)
            row2[:n] = -dmin * V_orig[i]
            row2[n + i] = 1
            rhs2 = dmin * (c_orig[i] - u[i]) + fu[i]
            C_rows.append(row2)
            d_rows.append(rhs2)

            # Intersection geometry for tighter bounds
            if abs(df0 - dmin) > 1e-10:
                gu_x = (fu[i] - dmin * u[i] - f0) / (df0 - dmin)
                gu_y = df0 * gu_x + f0
                gl_x = (fl[i] - dmin * l[i] - f0) / (df0 - dmin)
                gl_y = df0 * gl_x + f0

                # Tighter lower: y >= m_l*(x - u) + f(u)
                if abs(u[i] - gl_x) > 1e-10:
                    m_l = (fu[i] - gl_y) / (u[i] - gl_x)
                    row3 = np.zeros(n + m)
                    row3[:n] = m_l * V_orig[i]
                    row3[n + i] = -1
                    rhs3 = m_l * (u[i] - c_orig[i]) - fu[i]
                    C_rows.append(row3)
                    d_rows.append(rhs3)

                # Tighter upper: y <= m_u*(x - l) + f(l)
                if abs(l[i] - gu_x) > 1e-10:
                    m_u = (fl[i] - gu_y) / (l[i] - gu_x)
                    row4 = np.zeros(n + m)
                    row4[:n] = -m_u * V_orig[i]
                    row4[n + i] = 1
                    rhs4 = m_u * (c_orig[i] - l[i]) + fl[i]
                    C_rows.append(row4)
                    d_rows.append(rhs4)

    if len(C_rows) > 0:
        C_new = np.array(C_rows)
        d_new = np.array(d_rows).reshape(-1, 1)
        new_C = np.vstack([C0, C_new])
        new_d = np.vstack([d0, d_new])
    else:
        new_C = C0
        new_d = d0

    # Predicate bounds
    new_pred_lb = np.vstack([I.predicate_lb, fl.reshape(-1, 1)]) if I.predicate_lb is not None else None
    new_pred_ub = np.vstack([I.predicate_ub, fu.reshape(-1, 1)]) if I.predicate_ub is not None else None

    return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub, outer_zono=None)


def _s_curve_single_zono(I: Zono, func: Callable) -> Zono:
    """
    Approximate S-curve activation for a single Zonotope.

    Uses interval over-approximation: compute output interval [f(l), f(u)]
    for each dimension, create new zonotope with center at midpoint and
    error generators for the interval width.
    """
    lb, ub = I.get_bounds()
    fl = func(lb)
    fu = func(ub)

    # New center and generator for each dimension
    new_c = 0.5 * (fl + fu)
    # Error generators: one per dimension for the output interval
    half_widths = 0.5 * (fu - fl)
    new_V = np.diag(half_widths.flatten())

    return Zono(new_c, new_V)
