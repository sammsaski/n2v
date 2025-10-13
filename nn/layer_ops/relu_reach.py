"""
ReLU activation reachability operations.

Exact and approximate reachability for ReLU (Positive Linear) activation.
Translated from MATLAB NNV PosLin.m
"""

import numpy as np
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from n2v.sets import Star, Zono, Hexatope, Octatope


def relu_star_exact(
    input_stars: List[Star],
    lp_solver: str = 'default',
    dis_opt: Optional[str] = None,
    parallel: bool = None,
    n_workers: int = None
) -> List[Star]:
    """
    Exact reachability for ReLU using Star sets.

    Args:
        input_stars: List of input Star sets
        lp_solver: LP solver to use
        dis_opt: 'display' to show progress
        parallel: Enable parallel Star processing (None = use global config)
        n_workers: Number of parallel workers (None = auto-detect)

    Returns:
        List of output Star sets (may be more than input due to splitting)
    """
    # Check if we should use parallel processing
    use_parallel = _should_use_star_parallel(len(input_stars), parallel, n_workers)

    if use_parallel:
        return _relu_star_exact_parallel(input_stars, lp_solver, dis_opt, n_workers)
    else:
        # Sequential processing
        output_stars = []
        for star in input_stars:
            # Process each star through exact ReLU
            result = _relu_single_star_exact(star, lp_solver, dis_opt)
            output_stars.extend(result)
        return output_stars


def _relu_single_star_exact(
    I: Star,
    lp_solver: str = 'default',
    dis_opt: Optional[str] = None
) -> List[Star]:
    """
    Exact ReLU reachability for a single Star set.

    Algorithm:
    1. Estimate ranges for all dimensions
    2. Reset neurons that are always ≤ 0
    3. Split neurons that cross 0 boundary

    Args:
        I: Input Star
        lp_solver: LP solver
        dis_opt: Display option

    Returns:
        List of output Stars
    """
    if I is None or I.dim == 0:
        return []

    # Estimate ranges
    lb, ub = I.estimate_ranges()

    if lb is None or ub is None:
        return []

    # Neurons always inactive (ub <= 0) - reset to 0
    reset_map = np.where(ub.flatten() <= 0)[0]

    V = I.V.copy()
    V[reset_map, :] = 0

    # Update outer zonotope
    if I.Z is not None:
        c1 = I.Z.c.copy()
        c1[reset_map] = 0
        V1 = I.Z.V.copy()
        V1[reset_map, :] = 0
        new_Z = Zono(c1, V1)
    else:
        new_Z = None

    current_stars = [Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)]

    # Neurons crossing zero (lb < 0 and ub > 0) - need splitting
    split_map = np.where((lb.flatten() < 0) & (ub.flatten() > 0))[0]

    # Recursively split each uncertain neuron
    for i, neuron_idx in enumerate(split_map):
        if dis_opt == 'display':
            print(f'Exact ReLU_{neuron_idx} ({i+1}/{len(split_map)})')

        new_stars = []
        for star in current_stars:
            split_result = _step_relu(star, neuron_idx, lp_solver)
            new_stars.extend(split_result)

        current_stars = new_stars

    return current_stars


def _step_relu(I: Star, index: int, lp_solver: str = 'default') -> List[Star]:
    """
    Split a single neuron in ReLU (exact step reach).

    Args:
        I: Input Star
        index: Neuron index to split
        lp_solver: LP solver

    Returns:
        List of 1 or 2 Star sets
    """
    # Get bounds of neuron
    xmin, xmax = I.get_range(index, lp_solver)

    if xmin is None or xmax is None:
        return []

    if xmin >= 0:
        # Always active
        return [I]

    elif xmax <= 0:
        # Always inactive - zero out
        new_V = I.V.copy()
        new_V[index, :] = 0

        if I.Z is not None:
            new_c = I.Z.c.copy()
            new_c[index] = 0
            new_V_zono = I.Z.V.copy()
            new_V_zono[index, :] = 0
            new_Z = Zono(new_c, new_V_zono)
        else:
            new_Z = None

        return [Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)]

    else:
        # Split into two cases
        c = I.V[index, 0]
        V = I.V[index, 1:I.nVar + 1].reshape(1, -1)

        # Case 1: x[index] < 0 (inactive)
        new_C1 = np.vstack([I.C, V])
        new_d1 = np.vstack([I.d, -c * np.ones((1, 1))])
        new_V1 = I.V.copy()
        new_V1[index, :] = 0

        if I.Z is not None:
            c1 = I.Z.c.copy()
            c1[index] = 0
            V1 = I.Z.V.copy()
            V1[index, :] = 0
            new_Z1 = Zono(c1, V1)
        else:
            new_Z1 = None

        S1 = Star(new_V1, new_C1, new_d1, I.predicate_lb, I.predicate_ub, outer_zono=new_Z1)

        # Case 2: x[index] >= 0 (active)
        new_C2 = np.vstack([I.C, -V])
        new_d2 = np.vstack([I.d, c * np.ones((1, 1))])
        S2 = Star(I.V, new_C2, new_d2, I.predicate_lb, I.predicate_ub, outer_zono=I.Z)

        return [S1, S2]


def relu_star_approx(
    input_stars: List[Star],
    relax_factor: float = 0.5,
    lp_solver: str = 'default',
    relax_method: str = 'standard'
) -> List[Star]:
    """
    Approximate reachability for ReLU using triangle relaxation.

    Uses triangle over-approximation for neurons crossing zero instead of
    splitting. This is faster but produces wider (conservative) bounds.

    Args:
        input_stars: List of input Stars
        relax_factor: 0 = exact, 1 = maximum relaxation
        lp_solver: LP solver
        relax_method: Relaxation strategy - 'standard', 'range', 'area', 'bound'

    Returns:
        List of output Stars (no splitting, same count as input)
    """
    if relax_factor == 0.0:
        return relu_star_exact(input_stars, lp_solver)

    output_stars = []

    for star in input_stars:
        # Process each star through approximate ReLU with specified method
        if relax_method == 'range':
            result = _relu_single_star_relax_range(star, relax_factor, lp_solver)
        elif relax_method == 'area':
            result = _relu_single_star_relax_area(star, relax_factor, lp_solver)
        elif relax_method == 'bound':
            result = _relu_single_star_relax_bound(star, relax_factor, lp_solver)
        else:  # 'standard'
            result = _relu_single_star_approx(star, lp_solver)

        if result is not None:
            output_stars.append(result)

    return output_stars


def _relu_single_star_approx(
    I: Star,
    lp_solver: str = 'default'
) -> Optional[Star]:
    """
    Approximate ReLU reachability for a single Star using triangle relaxation.

    Algorithm (from MATLAB NNV PosLin.m stepReachStarApprox):
    1. Estimate ranges for all dimensions
    2. Reset neurons that are always ≤ 0 (set to zero)
    3. Keep neurons that are always > 0 (no change)
    4. For neurons crossing zero (lb < 0 < ub):
       - Add new predicate variable y representing output
       - Add triangle constraints:
         * y ≥ 0 (output non-negative)
         * y ≥ x (output at least input when active)
         * y ≤ (ub/(ub-lb))*(x - lb) (triangle upper bound)

    This avoids splitting but creates an over-approximation.

    Args:
        I: Input Star
        lp_solver: LP solver

    Returns:
        Output Star (single star, no splitting)
    """
    if I is None or I.dim == 0:
        return None

    # Estimate ranges
    lb, ub = I.estimate_ranges()

    if lb is None or ub is None:
        return None

    lb = lb.flatten()
    ub = ub.flatten()

    # Step 1: Handle neurons always inactive (ub <= 0) - reset to 0
    reset_map = np.where(ub <= 0)[0]

    V = I.V.copy()
    V[reset_map, :] = 0

    # Update outer zonotope
    if I.Z is not None:
        c1 = I.Z.c.copy()
        c1[reset_map] = 0
        V1 = I.Z.V.copy()
        V1[reset_map, :] = 0
        new_Z = Zono(c1, V1)
    else:
        new_Z = None

    current_star = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)

    # Step 2: Find neurons crossing zero (lb < 0 and ub > 0) - apply triangle approximation
    crossing_map = np.where((lb < 0) & (ub > 0))[0]

    # Process each crossing neuron by adding constraints (no splitting!)
    for idx in crossing_map:
        current_star = _step_relu_approx(current_star, idx, lb[idx], ub[idx])
        if current_star is None:
            return None

    return current_star


def _step_relu_approx(
    I: Star,
    index: int,
    lb: float,
    ub: float
) -> Optional[Star]:
    """
    Apply triangle approximation for a single ReLU neuron crossing zero.

    Adds a new predicate variable to represent the output and constrains it
    with triangle relaxation instead of splitting.

    Args:
        I: Input Star
        index: Neuron index to approximate
        lb: Lower bound of neuron (< 0)
        ub: Upper bound of neuron (> 0)

    Returns:
        Star with added predicate variable and triangle constraints
    """
    if I is None:
        return None

    n = I.nVar + 1  # Number of variables after adding new predicate

    # Triangle relaxation constraints:
    # Let y be the new predicate variable representing ReLU output at index

    # Constraint 1: y >= 0 (output is non-negative)
    # Expressed as: -y <= 0
    C1 = np.zeros((1, n))
    C1[0, -1] = -1  # Last column is new variable
    d1 = np.array([[0.0]])

    # Constraint 2: y >= x (output >= input when active)
    # Expressed as: x - y <= 0, or V[index,1:n]*alpha - y <= -V[index,0]
    C2 = np.zeros((1, n))
    C2[0, :I.nVar] = I.V[index, 1:I.nVar + 1]  # Coefficients for existing predicates
    C2[0, -1] = -1  # New variable
    d2 = np.array([[-I.V[index, 0]]])

    # Constraint 3: y <= (ub/(ub-lb)) * (x - lb)
    # Expressed as: y - (ub/(ub-lb))*x <= -ub*lb/(ub-lb)
    # Or: -[(ub/(ub-lb))*x] + y <= ub*lb/(ub-lb) + (ub/(ub-lb))*c
    if abs(ub - lb) < 1e-10:
        # Degenerate case: lb ≈ ub, can't apply triangle
        return I

    lambda_val = ub / (ub - lb)
    C3 = np.zeros((1, n))
    C3[0, :I.nVar] = -lambda_val * I.V[index, 1:I.nVar + 1]
    C3[0, -1] = 1  # New variable
    d3 = np.array([[-ub * lb / (ub - lb) + ub * I.V[index, 0] / (ub - lb)]])

    # Combine old constraints with new ones
    m = I.C.shape[0]
    C0 = np.hstack([I.C, np.zeros((m, 1))])  # Add column for new variable
    d0 = I.d

    new_C = np.vstack([C0, C1, C2, C3])
    new_d = np.vstack([d0, d1, d2, d3])

    # Update basis matrix V
    # Old neurons keep their representation, but neuron at 'index' is now represented by the new variable
    new_V = np.hstack([I.V, np.zeros((I.dim, 1))])  # Add column for new variable
    new_V[index, :] = 0  # Zero out original representation
    new_V[index, -1] = 1  # New variable directly represents this neuron's output

    # Update predicate bounds
    new_predicate_lb = np.vstack([I.predicate_lb, [[0.0]]]) if I.predicate_lb is not None else None
    new_predicate_ub = np.vstack([I.predicate_ub, [[ub]]]) if I.predicate_ub is not None else None

    # Update outer zonotope with triangle approximation
    # y ≈ lambda*x + mu, where lambda = ub/(ub-lb), mu = -0.5*ub*lb/(ub-lb)
    mu = -0.5 * ub * lb / (ub - lb)
    if I.Z is not None:
        c = I.Z.c.copy()
        c[index] = lambda_val * c[index] + mu
        V_zono = I.Z.V.copy()
        V_zono[index, :] = lambda_val * V_zono[index, :]
        # Add generator for approximation error
        I1 = np.zeros((I.dim, 1))
        I1[index] = mu
        V_zono = np.hstack([V_zono, I1])
        new_Z = Zono(c, V_zono)
    else:
        new_Z = None

    return Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, outer_zono=new_Z)


def relu_zono_approx(input_zonos: List[Zono]) -> List[Zono]:
    """
    Approximate ReLU reachability using Zonotopes.

    Args:
        input_zonos: List of input Zonotopes

    Returns:
        List of output Zonotopes (over-approximation)
    """
    output_zonos = []

    for zono in input_zonos:
        output_zonos.append(_relu_single_zono(zono))

    return output_zonos


def _relu_single_zono(I: Zono) -> Zono:
    """
    Approximate ReLU for a single Zonotope.

    Uses interval hull over-approximation.

    Args:
        I: Input Zonotope

    Returns:
        Output Zonotope (over-approximation)
    """
    lb, ub = I.get_bounds()

    new_c = I.c.copy()
    new_V = I.V.copy()
    n_orig_generators = I.V.shape[1]  # Track original number of generators

    for i in range(I.dim):
        li, ui = lb[i, 0], ub[i, 0]

        if ui <= 0:
            # Always inactive
            new_c[i] = 0
            new_V[i, :] = 0

        elif li >= 0:
            # Always active - no change
            pass

        else:
            # Crosses zero - use over-approximation
            lambda_val = ui / (ui - li) if (ui - li) != 0 else 0

            new_c[i] = lambda_val * I.c[i, 0] + 0.5 * (1 - lambda_val) * ui
            # Only modify original generators, not error generators added by previous neurons
            new_V[i, :n_orig_generators] = lambda_val * I.V[i, :]

            # Add error term as new generator
            error = 0.5 * (1 - lambda_val) * ui
            if error > 1e-10:
                error_gen = np.zeros((I.dim, 1))
                error_gen[i] = error
                new_V = np.hstack([new_V, error_gen])

    return Zono(new_c, new_V)


def relu_box(input_boxes: List) -> List:
    """
    ReLU for Boxes (interval arithmetic).

    Args:
        input_boxes: List of input Boxes

    Returns:
        List of output Boxes
    """
    from n2v.sets import Box

    output_boxes = []

    for box in input_boxes:
        new_lb = np.maximum(box.lb, 0)
        new_ub = np.maximum(box.ub, 0)
        output_boxes.append(Box(new_lb, new_ub))

    return output_boxes


def _relu_single_star_relax_range(
    I: Star,
    relax_factor: float,
    lp_solver: str = 'default'
) -> Optional[Star]:
    """
    Relaxed ReLU reachability prioritizing neurons by range width (ub - lb).

    Algorithm from MATLAB NNV PosLin.reach_relaxed_star_range:
    1. Estimate ranges for all dimensions
    2. Reset neurons with ub <= 0
    3. For neurons crossing zero (lb < 0 < ub):
       - Compute exact bounds for (1-relaxFactor) fraction with largest ranges
       - Use estimated bounds for remaining relaxFactor fraction
    4. Apply triangle approximation to all crossing neurons in one shot

    Args:
        I: Input Star
        relax_factor: Fraction of neurons to relax (0=exact, 1=max relaxation)
        lp_solver: LP solver

    Returns:
        Output Star with relaxed approximation
    """
    if I is None or I.dim == 0:
        return None

    if relax_factor < 0 or relax_factor > 1:
        raise ValueError(f"Invalid relax_factor: {relax_factor}, must be in [0,1]")

    # Step 1: Estimate ranges
    lb, ub = I.estimate_ranges()
    if lb is None or ub is None:
        return None

    lb = lb.flatten()
    ub = ub.flatten()

    # Step 2: Find and reset neurons with ub <= 0
    map1 = np.where(ub <= 0)[0]
    V = I.V.copy()
    V[map1, :] = 0

    if I.Z is not None:
        c1 = I.Z.c.copy()
        c1[map1] = 0
        V1 = I.Z.V.copy()
        V1[map1, :] = 0
        new_Z = Zono(c1, V1)
    else:
        new_Z = None

    In = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)

    # Step 3: Find neurons crossing zero
    map2 = np.where((lb < 0) & (ub > 0))[0]

    if len(map2) == 0:
        return In

    # Step 4: Prioritize by range width (ub - lb)
    n1 = round((1 - relax_factor) * len(map2))  # Number of neurons to optimize exactly
    ranges = ub[map2] - lb[map2]
    sorted_indices = np.argsort(-ranges)  # Descending order

    map21 = map2[sorted_indices[:n1]]  # Neurons with optimized ranges
    map22 = map2[sorted_indices[n1:]]  # Neurons with estimated ranges

    lb1 = lb[map22]
    ub1 = ub[map22]

    # Step 5: Optimize upper bounds of selected neurons
    xmax = _get_maxs(I, map21, lp_solver) if len(map21) > 0 else np.array([])

    # Find neurons that are actually inactive after optimization
    map3 = np.where(xmax <= 0)[0] if len(xmax) > 0 else np.array([], dtype=int)
    map4 = map21[map3] if len(map3) > 0 else np.array([], dtype=int)

    # Reset newly found inactive neurons
    map11 = np.concatenate([map1, map4]) if len(map4) > 0 else map1
    In = In if len(map4) == 0 else _reset_star_rows(In, map4)

    # Step 6: Optimize lower bounds of neurons that are still crossing
    map5 = np.where(xmax > 0)[0] if len(xmax) > 0 else np.array([], dtype=int)
    map6 = map21[map5] if len(map5) > 0 else np.array([], dtype=int)
    xmax1 = xmax[map5] if len(map5) > 0 else np.array([])

    xmin = _get_mins(I, map6, lp_solver) if len(map6) > 0 else np.array([])

    map7 = np.where(xmin < 0)[0] if len(xmin) > 0 else np.array([], dtype=int)
    map8 = map6[map7] if len(map7) > 0 else np.array([], dtype=int)
    lb2 = xmin[map7] if len(map7) > 0 else np.array([])
    ub2 = xmax1[map7] if len(map7) > 0 else np.array([])

    # Step 7: Combine all crossing neurons and apply triangle approximation
    map9 = np.concatenate([map22, map8]) if len(map8) > 0 else map22
    lb3 = np.concatenate([lb1, lb2]) if len(lb2) > 0 else lb1
    ub3 = np.concatenate([ub1, ub2]) if len(ub2) > 0 else ub1

    if len(map9) == 0:
        return In

    # Apply multi-neuron triangle approximation in one shot
    result = _apply_triangle_approx_multi(In, map9, lb3, ub3)

    return result


def _relu_single_star_relax_area(
    I: Star,
    relax_factor: float,
    lp_solver: str = 'default'
) -> Optional[Star]:
    """
    Relaxed ReLU reachability prioritizing neurons by triangle area.

    Similar to relax_range but prioritizes by area = 0.5 * |ub| * |lb|.
    Neurons with larger triangle areas get exact optimization.

    Args:
        I: Input Star
        relax_factor: Fraction of neurons to relax
        lp_solver: LP solver

    Returns:
        Output Star with relaxed approximation
    """
    if I is None or I.dim == 0:
        return None

    if relax_factor < 0 or relax_factor > 1:
        raise ValueError(f"Invalid relax_factor: {relax_factor}, must be in [0,1]")

    lb, ub = I.estimate_ranges()
    if lb is None or ub is None:
        return None

    lb = lb.flatten()
    ub = ub.flatten()

    # Reset neurons with ub <= 0
    map1 = np.where(ub <= 0)[0]
    V = I.V.copy()
    V[map1, :] = 0

    if I.Z is not None:
        c1 = I.Z.c.copy()
        c1[map1] = 0
        V1 = I.Z.V.copy()
        V1[map1, :] = 0
        new_Z = Zono(c1, V1)
    else:
        new_Z = None

    In = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)

    # Find neurons crossing zero
    map2 = np.where((lb < 0) & (ub > 0))[0]

    if len(map2) == 0:
        return In

    # Prioritize by triangle area: 0.5 * |ub| * |lb|
    n1 = round((1 - relax_factor) * len(map2))
    areas = 0.5 * np.abs(ub[map2]) * np.abs(lb[map2])
    sorted_indices = np.argsort(-areas)  # Descending order

    map21 = map2[sorted_indices[:n1]]  # Neurons with optimized ranges
    map22 = map2[sorted_indices[n1:]]  # Neurons with estimated ranges

    lb1 = lb[map22]
    ub1 = ub[map22]

    # Optimize selected neurons (same process as relax_range)
    xmax = _get_maxs(I, map21, lp_solver) if len(map21) > 0 else np.array([])

    map3 = np.where(xmax <= 0)[0] if len(xmax) > 0 else np.array([], dtype=int)
    map4 = map21[map3] if len(map3) > 0 else np.array([], dtype=int)

    map11 = np.concatenate([map1, map4]) if len(map4) > 0 else map1
    In = In if len(map4) == 0 else _reset_star_rows(In, map4)

    map5 = np.where(xmax > 0)[0] if len(xmax) > 0 else np.array([], dtype=int)
    map6 = map21[map5] if len(map5) > 0 else np.array([], dtype=int)
    xmax1 = xmax[map5] if len(map5) > 0 else np.array([])

    xmin = _get_mins(I, map6, lp_solver) if len(map6) > 0 else np.array([])

    map7 = np.where(xmin < 0)[0] if len(xmin) > 0 else np.array([], dtype=int)
    map8 = map6[map7] if len(map7) > 0 else np.array([], dtype=int)
    lb2 = xmin[map7] if len(map7) > 0 else np.array([])
    ub2 = xmax1[map7] if len(map7) > 0 else np.array([])

    map9 = np.concatenate([map22, map8]) if len(map8) > 0 else map22
    lb3 = np.concatenate([lb1, lb2]) if len(lb2) > 0 else lb1
    ub3 = np.concatenate([ub1, ub2]) if len(ub2) > 0 else ub1

    if len(map9) == 0:
        return In

    result = _apply_triangle_approx_multi(In, map9, lb3, ub3)

    return result


def _relu_single_star_relax_bound(
    I: Star,
    relax_factor: float,
    lp_solver: str = 'default'
) -> Optional[Star]:
    """
    Relaxed ReLU reachability prioritizing by individual bound magnitudes.

    Prioritizes neurons by |ub| and |lb| separately, optimizing the
    bounds with largest magnitudes.

    Args:
        I: Input Star
        relax_factor: Fraction of neurons to relax
        lp_solver: LP solver

    Returns:
        Output Star with relaxed approximation
    """
    if I is None or I.dim == 0:
        return None

    if relax_factor < 0 or relax_factor > 1:
        raise ValueError(f"Invalid relax_factor: {relax_factor}, must be in [0,1]")

    lb, ub = I.estimate_ranges()
    if lb is None or ub is None:
        return None

    lb = lb.flatten()
    ub = ub.flatten()

    # Reset neurons with ub <= 0
    map1 = np.where(ub <= 0)[0]
    V = I.V.copy()
    V[map1, :] = 0

    if I.Z is not None:
        c1 = I.Z.c.copy()
        c1[map1] = 0
        V1 = I.Z.V.copy()
        V1[map1, :] = 0
        new_Z = Zono(c1, V1)
    else:
        new_Z = None

    In = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)

    # Find neurons crossing zero
    map2 = np.where((lb < 0) & (ub > 0))[0]

    if len(map2) == 0:
        return In

    # Prioritize by bound magnitudes (both ub and |lb|)
    n1 = round((1 - relax_factor) * len(map2))
    N = len(map2)

    # Combine ub and |lb| values
    lu = np.concatenate([ub[map2], np.abs(lb[map2])])
    sorted_indices = np.argsort(-lu)  # Descending order

    # First 2*n1 indices, split between ub and lb optimizations
    selected_indices = sorted_indices[:2*n1] if 2*n1 <= len(lu) else sorted_indices
    ub_idx = selected_indices[selected_indices < N]  # Indices for ub optimization
    lb_idx = selected_indices[selected_indices >= N] - N  # Indices for lb optimization

    map21 = map2[ub_idx] if len(ub_idx) > 0 else np.array([], dtype=int)  # Optimize ub
    map22 = map2[lb_idx] if len(lb_idx) > 0 else np.array([], dtype=int)  # Optimize lb

    # Optimize upper bounds
    if len(map21) > 0:
        xmax = _get_maxs(I, map21, lp_solver)
        map3 = np.where(xmax <= 0)[0]
        map4 = map21[map3] if len(map3) > 0 else np.array([], dtype=int)
        map5 = np.where(xmax > 0)[0]
        map6 = map21[map5] if len(map5) > 0 else np.array([], dtype=int)
        map11 = np.concatenate([map1, map4]) if len(map4) > 0 else map1
    else:
        map11 = map1
        map4 = np.array([], dtype=int)
        map6 = np.array([], dtype=int)

    In = In if len(map4) == 0 else _reset_star_rows(In, map4)

    # Remove newly inactive neurons from lb optimization list
    if len(map4) > 0:
        map23 = np.setdiff1d(map22, map4)
    else:
        map23 = map22

    # Optimize lower bounds
    if len(map23) > 0:
        xmin = _get_mins(I, map23, lp_solver)
        map7 = np.where(xmin < 0)[0]
        map8 = map23[map7] if len(map7) > 0 else np.array([], dtype=int)
        map9 = np.where(xmin >= 0)[0]
        map10 = map23[map9] if len(map9) > 0 else np.array([], dtype=int)
    else:
        map8 = np.array([], dtype=int)
        map10 = np.array([], dtype=int)

    # Gather all neurons needing approximation
    # Include neurons not selected for optimization
    unselected = np.setdiff1d(map2, np.concatenate([map21, map22]))
    crossing_neurons = np.concatenate([unselected, map8]) if len(map8) > 0 else unselected

    if len(crossing_neurons) == 0:
        return In

    # Get bounds for all crossing neurons
    lbs = []
    ubs = []
    for idx in crossing_neurons:
        if idx in map8:
            # Optimized bound
            opt_idx = np.where(map8 == idx)[0][0]
            lbs.append(xmin[np.where(map23 == idx)[0][0]])
            # Need to get ub - check if it was optimized
            if idx in map6:
                ub_opt_idx = np.where(map6 == idx)[0][0]
                ubs.append(xmax[np.where(map21 == idx)[0][0]])
            else:
                ubs.append(ub[idx])
        else:
            # Not optimized, use estimated
            lbs.append(lb[idx])
            ubs.append(ub[idx])

    lb_arr = np.array(lbs)
    ub_arr = np.array(ubs)

    result = _apply_triangle_approx_multi(In, crossing_neurons, lb_arr, ub_arr)

    return result


def _reset_star_rows(I: Star, indices: np.ndarray) -> Star:
    """Helper function to reset specified rows of a star to zero."""
    V = I.V.copy()
    V[indices, :] = 0

    if I.Z is not None:
        c = I.Z.c.copy()
        c[indices] = 0
        V_zono = I.Z.V.copy()
        V_zono[indices, :] = 0
        new_Z = Zono(c, V_zono)
    else:
        new_Z = None

    return Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, outer_zono=new_Z)


def _apply_triangle_approx_multi(
    I: Star,
    indices: np.ndarray,
    lbs: np.ndarray,
    ubs: np.ndarray
) -> Star:
    """
    Apply triangle approximation to multiple neurons simultaneously.

    This is the Python equivalent of MATLAB's multipleStepReachStarApprox_at_one.
    Instead of applying triangle approximation one neuron at a time, this does
    it all at once by adding multiple predicate variables.

    Args:
        I: Input star
        indices: Array of neuron indices to approximate
        lbs: Lower bounds for each neuron
        ubs: Upper bounds for each neuron

    Returns:
        Star with triangle approximations applied
    """
    if len(indices) == 0:
        return I

    N = I.dim
    m = len(indices)  # Number of neurons to approximate
    n = I.nVar  # Number of existing predicate variables

    # Construct new basis array
    V1 = I.V.copy()
    V1[indices, :] = 0  # Zero out original representations

    # Create basis for new predicates (one per neuron)
    V2 = np.zeros((N, m))
    for i, idx in enumerate(indices):
        V2[idx, i] = 1

    new_V = np.hstack([V1, V2])

    # Construct constraints
    # Case 0: Keep old constraints
    C0 = np.hstack([I.C, np.zeros((I.C.shape[0], m))])
    d0 = I.d

    # Case 1: y[i] >= 0 for all i
    C1 = np.hstack([np.zeros((m, n)), -np.eye(m)])
    d1 = np.zeros((m, 1))

    # Case 2: y[i] >= x[i] for all i
    C2 = np.hstack([I.V[indices, 1:n+1], -np.eye(m)])
    d2 = -I.V[indices, 0:1]

    # Case 3: y[i] <= (ub[i]/(ub[i]-lb[i]))*(x[i]-lb[i])
    a = ubs / (ubs - lbs + 1e-10)  # Avoid division by zero
    b = a * lbs
    C3 = np.hstack([-a.reshape(-1, 1) * I.V[indices, 1:n+1], np.eye(m)])
    d3 = (a * I.V[indices, 0] - b).reshape(-1, 1)

    new_C = np.vstack([C0, C1, C2, C3])
    new_d = np.vstack([d0, d1, d2, d3])

    # Update predicate bounds
    new_pred_lb = np.vstack([I.predicate_lb, np.zeros((m, 1))]) if I.predicate_lb is not None else None
    new_pred_ub = np.vstack([I.predicate_ub, ubs.reshape(-1, 1)]) if I.predicate_ub is not None else None

    return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub, outer_zono=None)


def _get_maxs(star: Star, indices: np.ndarray, lp_solver: str = 'default') -> np.ndarray:
    """
    Get maximum values for multiple state dimensions.

    Equivalent to MATLAB Star.getMaxs(map).

    Args:
        star: Star set
        indices: Array of dimension indices
        lp_solver: LP solver to use

    Returns:
        Array of maximum values, one per index
    """
    n = len(indices)
    xmax = np.zeros(n)

    for i, idx in enumerate(indices):
        _, xmax[i] = star.get_range(int(idx), lp_solver)

    return xmax


def _get_mins(star: Star, indices: np.ndarray, lp_solver: str = 'default') -> np.ndarray:
    """
    Get minimum values for multiple state dimensions.

    Equivalent to MATLAB Star.getMins(map).

    Args:
        star: Star set
        indices: Array of dimension indices
        lp_solver: LP solver to use

    Returns:
        Array of minimum values, one per index
    """
    n = len(indices)
    xmin = np.zeros(n)

    for i, idx in enumerate(indices):
        xmin[i], _ = star.get_range(int(idx), lp_solver)

    return xmin


# ============================================================================
# Star-Level Parallelization Functions
# ============================================================================

def _should_use_star_parallel(n_stars: int, parallel: bool = None, n_workers: int = None) -> bool:
    """
    Determine if Star-level parallelization should be used.

    Args:
        n_stars: Number of Stars to process
        parallel: Explicit parallel setting (None = use global config)
        n_workers: Number of workers (None = auto)

    Returns:
        True if parallel processing should be used
    """
    # Need at least 2 Stars to benefit from parallelization
    if n_stars < 2:
        return False

    # Check explicit setting
    if parallel is not None:
        return parallel

    # Check global config
    try:
        from n2v.config import config as global_config
        # Use star_parallel setting if available, otherwise check if parallel is enabled
        if hasattr(global_config, 'star_parallel'):
            return global_config.star_parallel and n_stars >= 2
        elif hasattr(global_config, 'parallel_lp'):
            # If LP parallel is enabled, also enable Star parallel for n_stars >= 2
            return global_config.parallel_lp and n_stars >= 2
    except ImportError:
        pass

    # Default: use parallel if we have multiple Stars
    return n_stars >= 4  # Conservative threshold


def _get_star_workers(n_stars: int, n_workers: int = None) -> int:
    """
    Determine optimal number of workers for Star parallelization.

    Args:
        n_stars: Number of Stars to process
        n_workers: Requested workers (None = auto-detect)

    Returns:
        Number of workers to use
    """
    if n_workers is not None:
        return max(1, min(n_workers, n_stars))

    # Check global config
    try:
        from n2v.config import config as global_config
        workers = global_config.n_workers if hasattr(global_config, 'n_workers') else 4
    except ImportError:
        workers = 4

    # Don't use more workers than Stars
    return max(1, min(workers, n_stars))


def _relu_star_exact_parallel(
    input_stars: List[Star],
    lp_solver: str = 'default',
    dis_opt: Optional[str] = None,
    n_workers: int = None
) -> List[Star]:
    """
    Process multiple Stars through exact ReLU in parallel.

    Uses ProcessPoolExecutor to distribute Stars across workers.

    Args:
        input_stars: List of input Stars
        lp_solver: LP solver
        dis_opt: Display option
        n_workers: Number of workers

    Returns:
        List of output Stars
    """
    workers = _get_star_workers(len(input_stars), n_workers)

    if workers == 1 or len(input_stars) == 1:
        # Fall back to sequential
        output_stars = []
        for star in input_stars:
            result = _relu_single_star_exact(star, lp_solver, dis_opt)
            output_stars.extend(result)
        return output_stars

    # Parallel processing
    output_stars = []

    if dis_opt == 'display':
        print(f'  ⚡ Processing {len(input_stars)} Stars in parallel ({workers} workers)')

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all Stars for processing
        future_to_idx = {
            executor.submit(_relu_single_star_exact, star, lp_solver, None): idx
            for idx, star in enumerate(input_stars)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                output_stars.extend(result)
            except Exception as e:
                if dis_opt == 'display':
                    print(f'  Error processing Star: {e}')
                # Continue with other Stars

    return output_stars


def relu_hexatope(input_hexatopes: List[Hexatope]) -> List[Hexatope]:
    """
    ReLU for Hexatopes (approximate, interval-based).

    Uses interval arithmetic over-approximation. For each dimension:
    - If ub <= 0: set to zero
    - If lb >= 0: keep unchanged  
    - If crosses zero: use box over-approximation [0, max(0, ub)]

    Args:
        input_hexatopes: List of input Hexatopes

    Returns:
        List of output Hexatopes (over-approximation)
    """
    output_hexatopes = []

    for hexatope in input_hexatopes:
        # Get bounds
        lb, ub = hexatope.estimate_ranges()
        
        # Apply ReLU element-wise
        new_lb = np.maximum(lb, 0)
        new_ub = np.maximum(ub, 0)
        
        # Create new hexatope from bounds
        output_hexatope = Hexatope.from_bounds(new_lb, new_ub)
        output_hexatopes.append(output_hexatope)

    return output_hexatopes


def relu_octatope(input_octatopes: List[Octatope]) -> List[Octatope]:
    """
    ReLU for Octatopes (approximate, interval-based).

    Uses interval arithmetic over-approximation. For each dimension:
    - If ub <= 0: set to zero
    - If lb >= 0: keep unchanged
    - If crosses zero: use box over-approximation [0, max(0, ub)]

    Args:
        input_octatopes: List of input Octatopes

    Returns:
        List of output Octatopes (over-approximation)
    """
    output_octatopes = []

    for octatope in input_octatopes:
        # Get bounds
        lb, ub = octatope.estimate_ranges()

        # Apply ReLU element-wise
        new_lb = np.maximum(lb, 0)
        new_ub = np.maximum(ub, 0)

        # Create new octatope from bounds
        output_octatope = Octatope.from_bounds(new_lb, new_ub)
        output_octatopes.append(output_octatope)

    return output_octatopes


def relu_hexatope_exact(
    input_hexatopes: List[Hexatope],
    dis_opt: Optional[str] = None
) -> List[Hexatope]:
    """
    Exact reachability for ReLU using Hexatope sets with splitting.

    Similar to relu_star_exact but for Hexatopes. Splits on neurons crossing zero.

    Args:
        input_hexatopes: List of input Hexatope sets
        dis_opt: 'display' to show progress

    Returns:
        List of output Hexatope sets (may be more than input due to splitting)
    """
    output_hexatopes = []
    for hexatope in input_hexatopes:
        result = _relu_single_hexatope_exact(hexatope, dis_opt)
        output_hexatopes.extend(result)
    return output_hexatopes


def _relu_single_hexatope_exact(
    I: Hexatope,
    dis_opt: Optional[str] = None
) -> List[Hexatope]:
    """
    Exact ReLU reachability for a single Hexatope set.

    Algorithm:
    1. Estimate ranges for all dimensions
    2. Reset neurons that are always ≤ 0
    3. Split neurons that cross 0 boundary

    Args:
        I: Input Hexatope
        dis_opt: Display option

    Returns:
        List of output Hexatopes (with splitting)
    """
    if I is None or I.dim == 0:
        return []

    # Estimate ranges
    lb, ub = I.estimate_ranges()

    if lb is None or ub is None:
        return []

    # Neurons always inactive (ub <= 0) - reset to 0
    reset_map = np.where(ub.flatten() <= 0)[0]

    # Reset neurons using affine map to preserve constraints
    if len(reset_map) > 0:
        W = np.eye(I.dim, dtype=np.float64)
        for idx in reset_map:
            W[idx, idx] = 0  # Zero out inactive neurons
        b = np.zeros((I.dim, 1), dtype=np.float64)
        I = I.affine_map(W, b)

    current_hexatopes = [I]

    # Neurons crossing zero (lb < 0 and ub > 0) - need splitting
    split_map = np.where((lb.flatten() < 0) & (ub.flatten() > 0))[0]

    # Recursively split each uncertain neuron
    for i, neuron_idx in enumerate(split_map):
        if dis_opt == 'display':
            print(f'Exact ReLU_{neuron_idx} ({i+1}/{len(split_map)})')

        new_hexatopes = []
        for hexatope in current_hexatopes:
            split_result = _step_relu_hexatope(hexatope, neuron_idx)
            new_hexatopes.extend(split_result)

        current_hexatopes = new_hexatopes

    return current_hexatopes


def _step_relu_hexatope(I: Hexatope, index: int) -> List[Hexatope]:
    """
    Split a single neuron in ReLU for Hexatope (exact step reach).

    Args:
        I: Input Hexatope
        index: Neuron index to split

    Returns:
        List of 1 or 2 Hexatope sets
    """
    # Get bounds of neuron
    xmin, xmax = I.get_range(index, use_mcf=False)

    if xmin is None or xmax is None:
        return []

    if xmin >= 0:
        # Always active
        return [I]

    elif xmax <= 0:
        # Always inactive - zero out this dimension
        lb, ub = I.estimate_ranges()
        new_lb = lb.copy()
        new_ub = ub.copy()
        new_lb[index] = 0
        new_ub[index] = 0
        return [Hexatope.from_bounds(new_lb, new_ub)]

    else:
        # Split into two cases
        # Case 1: x[index] < 0 (inactive) - constrain to x[index] <= 0 and set output to 0
        # Case 2: x[index] >= 0 (active) - constrain to x[index] >= 0 and keep as is

        # Case 1: x[index] <= 0 (inactive region)
        H1 = np.zeros((1, I.dim))
        H1[0, index] = 1.0
        g1 = np.array([[0.0]])
        hex1 = I.intersect_half_space(H1, g1)

        # Apply ReLU: zero out this dimension in output
        # Do this by creating linear map that zeros dimension 'index'
        W1 = np.eye(I.dim, dtype=np.float64)
        W1[index, index] = 0
        b1 = np.zeros((I.dim, 1), dtype=np.float64)
        hex1_final = hex1.affine_map(W1, b1)

        # Case 2: x[index] >= 0 (active region)
        H2 = np.zeros((1, I.dim))
        H2[0, index] = -1.0
        g2 = np.array([[0.0]])
        hex2 = I.intersect_half_space(H2, g2)
        # No transformation needed - ReLU is identity here

        return [hex1_final, hex2]


def relu_octatope_exact(
    input_octatopes: List[Octatope],
    dis_opt: Optional[str] = None
) -> List[Octatope]:
    """
    Exact reachability for ReLU using Octatope sets with splitting.

    Similar to relu_star_exact but for Octatopes. Splits on neurons crossing zero.

    Args:
        input_octatopes: List of input Octatope sets
        dis_opt: 'display' to show progress

    Returns:
        List of output Octatope sets (may be more than input due to splitting)
    """
    output_octatopes = []
    for octatope in input_octatopes:
        result = _relu_single_octatope_exact(octatope, dis_opt)
        output_octatopes.extend(result)
    return output_octatopes


def _relu_single_octatope_exact(
    I: Octatope,
    dis_opt: Optional[str] = None
) -> List[Octatope]:
    """
    Exact ReLU reachability for a single Octatope set.

    Algorithm:
    1. Estimate ranges for all dimensions
    2. Reset neurons that are always ≤ 0
    3. Split neurons that cross 0 boundary

    Args:
        I: Input Octatope
        dis_opt: Display option

    Returns:
        List of output Octatopes (with splitting)
    """
    if I is None or I.dim == 0:
        return []

    # Estimate ranges
    lb, ub = I.estimate_ranges()

    if lb is None or ub is None:
        return []

    # Neurons always inactive (ub <= 0) - reset to 0
    reset_map = np.where(ub.flatten() <= 0)[0]

    # Reset neurons using affine map to preserve constraints
    if len(reset_map) > 0:
        W = np.eye(I.dim, dtype=np.float64)
        for idx in reset_map:
            W[idx, idx] = 0  # Zero out inactive neurons
        b = np.zeros((I.dim, 1), dtype=np.float64)
        I = I.affine_map(W, b)

    current_octatopes = [I]

    # Neurons crossing zero (lb < 0 and ub > 0) - need splitting
    split_map = np.where((lb.flatten() < 0) & (ub.flatten() > 0))[0]

    # Recursively split each uncertain neuron
    for i, neuron_idx in enumerate(split_map):
        if dis_opt == 'display':
            print(f'Exact ReLU_{neuron_idx} ({i+1}/{len(split_map)})')

        new_octatopes = []
        for octatope in current_octatopes:
            split_result = _step_relu_octatope(octatope, neuron_idx)
            new_octatopes.extend(split_result)

        current_octatopes = new_octatopes

    return current_octatopes


def _step_relu_octatope(I: Octatope, index: int) -> List[Octatope]:
    """
    Split a single neuron in ReLU for Octatope (exact step reach).

    Args:
        I: Input Octatope
        index: Neuron index to split

    Returns:
        List of 1 or 2 Octatope sets
    """
    # Get bounds of neuron
    xmin, xmax = I.get_range(index, use_mcf=False)

    if xmin is None or xmax is None:
        return []

    if xmin >= 0:
        # Always active
        return [I]

    elif xmax <= 0:
        # Always inactive - zero out this dimension
        lb, ub = I.estimate_ranges()
        new_lb = lb.copy()
        new_ub = ub.copy()
        new_lb[index] = 0
        new_ub[index] = 0
        return [Octatope.from_bounds(new_lb, new_ub)]

    else:
        # Split into two cases
        # Case 1: x[index] < 0 (inactive) - constrain to x[index] <= 0 and set output to 0
        # Case 2: x[index] >= 0 (active) - constrain to x[index] >= 0 and keep as is

        # Case 1: x[index] <= 0 (inactive region)
        H1 = np.zeros((1, I.dim))
        H1[0, index] = 1.0
        g1 = np.array([[0.0]])
        oct1 = I.intersect_half_space(H1, g1)

        # Apply ReLU: zero out this dimension in output
        # Do this by creating linear map that zeros dimension 'index'
        W1 = np.eye(I.dim, dtype=np.float64)
        W1[index, index] = 0
        b1 = np.zeros((I.dim, 1), dtype=np.float64)
        oct1_final = oct1.affine_map(W1, b1)

        # Case 2: x[index] >= 0 (active region)
        H2 = np.zeros((1, I.dim))
        H2[0, index] = -1.0
        g2 = np.array([[0.0]])
        oct2 = I.intersect_half_space(H2, g2)
        # No transformation needed - ReLU is identity here

        return [oct1_final, oct2]
