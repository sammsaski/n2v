"""
ReLU activation reachability operations.

Exact and approximate reachability for ReLU (Positive Linear) activation.
Translated from MATLAB NNV PosLin.m

Note: ReLU is element-wise, so it operates on 2D Star representation.
ImageStar inputs are converted to Star (via to_star()), processed through ReLU,
and converted back to ImageStar (via _preserve_imagestar_type()).
"""

import logging
import warnings

import numpy as np
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from n2v.sets import Star, Zono, Hexatope, Octatope
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono

# Profiler hooks (no-op when profiling is disabled)
from n2v.profiling import count, is_enabled
from n2v.nn.layer_ops._profiling import (
    record_layer_neurons,
    record_classification,
    record_relax_outcome,
)
from n2v.utils.lp_solver_enum import LPSolver
from n2v.utils.lpsolver import LPFailure, _bound_lp_solver, _lp_pad
from n2v.utils.ns_certificate import outward_interval_bounds, outward_interval_slack

logger = logging.getLogger(__name__)


# ============================================================================
# Neuron phase decisions from LP bounds (shared with leakyrelu_reach and
# sign_reach)
# ============================================================================

# SOUNDNESS: an LP optimum that decides a neuron's phase is padded outward by
# _lp_pad and solved with HiGHS (_bound_lp_solver); both live in
# n2v.utils.lpsolver, shared with every other op that uses LP optima as bounds.


def _intersect_precomputed(lb, ub, precomputed_bounds, pad: bool = False):
    """Intersect a star's interval enclosure ``(lb, ub)`` with precomputed
    pre-activation bounds (from the bound pre-pass, or a caller such as
    nn/bab.py). Returns ``(lb, ub)`` in the shape of the inputs.

    ``pad=True`` first widens the precomputed bounds by the LP pad (the exact
    paths, where they only save LPs); the approximate paths use them as given.

    Both enclosures contain every point of the star, so they have one entry
    per neuron and overlap up to rounding. An array of another size means the
    bounds belong to another node, and raises (numpy would broadcast a size-1
    array to every neuron). An interval that misses the star's own enclosure
    means the bounds describe other neurons (e.g. a pre-pass that laid an
    image out in another row order) or that the star is empty. Either way the
    precomputed bounds of the whole layer are dropped, with a warning, and the
    star's own enclosure is used, which is sound. A NaN bound is ignored.
    """
    shape = np.shape(lb)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    pre_lb, pre_ub = precomputed_bounds
    pre_lb = np.asarray(pre_lb, dtype=np.float64).reshape(-1)
    pre_ub = np.asarray(pre_ub, dtype=np.float64).reshape(-1)
    if pre_lb.size != lb.size or pre_ub.size != lb.size:
        raise ValueError(
            f"precomputed bounds have {pre_lb.size}/{pre_ub.size} entries but "
            f"the layer has {lb.size} neurons")
    if pad:
        pre_lb = pre_lb - _lp_pad(pre_lb)
        pre_ub = pre_ub + _lp_pad(pre_ub)
    new_lb = np.fmax(lb, pre_lb)
    new_ub = np.fmin(ub, pre_ub)
    disjoint = new_lb > new_ub + _lp_pad(np.fmax(np.abs(new_lb), np.abs(new_ub)))
    if np.any(disjoint):
        logger.warning(
            f"precomputed bounds miss the star's own enclosure at "
            f"{int(np.count_nonzero(disjoint))} of {lb.size} neurons (the "
            f"bounds describe other neurons, or the star is empty); "
            f"ignoring them for this layer")
        return lb.reshape(shape), ub.reshape(shape)
    return new_lb.reshape(shape), new_ub.reshape(shape)


def _certified_empty(I: Star, lp_solver: "LPSolver | str") -> bool:
    """True only if ``I`` is shown empty by the certified emptiness check
    (``Star.is_empty_set``, i.e. ``confirm_infeasible``). A solver failure is
    not evidence of emptiness and gives False."""
    try:
        return I.is_empty_set(lp_solver)
    except LPFailure as e:
        logger.debug(f'emptiness check undecided ({e}); keeping the star')
        return False


def _phase_range(I: Star, index: int, lp_solver: "LPSolver | str"):
    """Enclosure of neuron ``index`` over ``I`` for deciding its phase.

    Returns ``None`` when ``I`` is certified empty. Otherwise returns
    ``(lo, hi, pad_lo, pad_hi)`` with ``lo <= x_index <= hi`` on ``I``:

    - a row with no nonzero generator is the exact constant, with zero pad;
    - otherwise the HiGHS range, padded outward by ``pad_lo`` / ``pad_hi``;
    - if the LP returns no optimum (any non-optimal status, including a
      presolve 'infeasible' on a feasible tiny-scale system) and ``I`` is not
      certified empty, the star's rigorous interval bound, with zero pad. The
      neuron is then undecided, never dropped.

    ``pad_lo`` / ``pad_hi`` let a caller tell a bound that is within its own
    pad of 0 (the raw optimum's sign is not reliable) from a decided one.
    """
    row = I.V[index, 1:]
    if not np.any(row != 0):
        c = float(I.V[index, 0])
        return c, c, 0.0, 0.0

    solver = _bound_lp_solver(lp_solver)
    xmin, xmax = I.get_range(index, solver)
    if xmin is None or xmax is None:
        if _certified_empty(I, solver):
            return None
        if I.predicate_lb is None or I.predicate_ub is None:
            return -np.inf, np.inf, 0.0, 0.0
        lo, hi = I.estimate_range(index)
        return lo, hi, 0.0, 0.0

    pad_lo, pad_hi = _lp_pad(xmin), _lp_pad(xmax)
    return xmin - pad_lo, xmax + pad_hi, pad_lo, pad_hi


def _phase_sliver_star(I: Star, index: int, row_scale: float, width: float) -> Star:
    """``I`` with row ``index`` scaled by ``row_scale``, plus one new predicate
    variable ``t`` in ``[min(0, width), max(0, width)]`` added to that row.

    This is how a neuron whose LP bound lies within its pad of 0 is kept in ONE
    star: the output row is the stable-phase map of ``x_index`` plus ``t``,
    which covers the other phase over the ``|width|``-wide band the LP cannot
    rule out (e.g. ReLU with raw min in ``[-pad, pad)``: ``y = x + t``,
    ``t in [0, -lo]``, since ``relu(x) - x = max(0, -x) <= -lo``). The result
    over-approximates by at most ``|width|`` in that one coordinate.
    """
    t_lb, t_ub = min(0.0, width), max(0.0, width)

    new_V = np.hstack([I.V, np.zeros((I.dim, 1))])
    new_V[index, :-1] = row_scale * new_V[index, :-1]
    new_V[index, -1] = 1.0

    new_C = np.hstack([I.C, np.zeros((I.C.shape[0], 1))])
    new_d = I.d
    if I.predicate_lb is not None and I.predicate_ub is not None:
        new_pred_lb = np.vstack([I.predicate_lb, [[t_lb]]])
        new_pred_ub = np.vstack([I.predicate_ub, [[t_ub]]])
    else:
        # no predicate box to carry the bounds of t: add them as rows
        rows = np.zeros((2, I.nVar + 1))
        rows[0, -1] = -1.0
        rows[1, -1] = 1.0
        new_C = np.vstack([new_C, rows])
        new_d = np.vstack([I.d, [[-t_lb], [t_ub]]])
        new_pred_lb = new_pred_ub = None

    if I.Z is not None:
        c = I.Z.c.copy()
        c[index] = row_scale * c[index] + 0.5 * (t_lb + t_ub)
        V_z = I.Z.V.copy()
        V_z[index, :] = row_scale * V_z[index, :]
        gen = np.zeros((I.dim, 1))
        gen[index] = 0.5 * (t_ub - t_lb)
        new_Z = Zono(c, np.hstack([V_z, gen]))
    else:
        new_Z = None

    return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub, outer_zono=new_Z)


def _thin_neuron_side(I: Star, index: int, xmin: float, xmax: float,
                      pad_lo: float, pad_hi: float,
                      lp_solver: "LPSolver | str") -> Optional[int]:
    """Stable side of a neuron whose whole padded range lies within the pads.

    Such a neuron is split (exact), but when the raw LP range is on one side
    of 0 the other child is usually empty and would otherwise be carried to
    the output. Returns +1 (active) / -1 (inactive) only when ``I`` intersected
    with the other side is certified empty (``_certified_empty``), else None.
    """
    solver = _bound_lp_solver(lp_solver)
    c = I.V[index, 0]
    row = I.V[index, 1:I.nVar + 1].reshape(1, -1)
    for side, raw_on_side, a, b in ((1, xmin + pad_lo >= 0, row, -c),
                                    (-1, xmax - pad_hi <= 0, -row, c)):
        if raw_on_side:
            # the other side: row * alpha <= -c (x <= 0) or >= -c (x >= 0)
            other = Star(I.V, np.vstack([I.C, a]), np.vstack([I.d, [[b]]]),
                         I.predicate_lb, I.predicate_ub)
            if _certified_empty(other, solver):
                return side
    return None


def _preserve_imagestar_type(original: Star, new_star: Star) -> Star:
    """
    If original was an ImageStar, convert the new_star back to ImageStar
    with the same spatial dimensions.

    Args:
        original: Original input Star or ImageStar (used for dimension info)
        new_star: Output Star from ReLU processing (2D V)

    Returns:
        ImageStar if original was ImageStar, otherwise returns new_star unchanged
    """
    if isinstance(original, ImageStar):
        return new_star.to_image_star(original.height, original.width, original.num_channels)
    return new_star


def relu_star_exact(
    input_stars: List[Star],
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    verbose: bool = False,
    parallel: bool = None,
    n_workers: int = None,
    precomputed_bounds: tuple = None,
) -> List[Star]:
    """
    Exact reachability for ReLU using Star sets.

    The result is exact up to documented slivers: a neuron with one LP bound
    within the LP pad of 0 and the other outside it is kept in one star with
    an extra predicate variable of width at most about 2e-7 * (1 + |bound|)
    (``_step_relu``).

    Args:
        input_stars: List of input Star sets
        lp_solver: LP solver to use (the phase LPs use HiGHS, see
            ``_bound_lp_solver``)
        verbose: 'display' to show progress
        parallel: Enable parallel Star processing (None = use global config)
        n_workers: Number of parallel workers (None = auto-detect)
        precomputed_bounds: Optional (lb, ub) from the bound pre-pass

    Returns:
        List of output Star sets (may be more than input due to splitting)
    """
    # Profiler: static neuron count, once per layer (no-op when disabled)
    record_layer_neurons(input_stars)

    # Check if we should use parallel processing
    use_parallel = _should_use_star_parallel(len(input_stars), parallel, n_workers)

    if use_parallel:
        return _relu_star_exact_parallel(input_stars, lp_solver, verbose, n_workers, precomputed_bounds)
    else:
        # Sequential processing
        output_stars = []
        for star in input_stars:
            # Convert ImageStar to Star for 2D ReLU processing
            star_2d = star.to_star() if isinstance(star, ImageStar) else star
            # Process each star through exact ReLU
            result = _relu_single_star_exact(star_2d, lp_solver, verbose, precomputed_bounds)
            # Preserve ImageStar type for each output star
            result = [_preserve_imagestar_type(star, s) for s in result]
            output_stars.extend(result)
        return output_stars


def _relu_single_star_exact(
    I: Star,
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    verbose: bool = False,
    precomputed_bounds: tuple = None,
) -> List[Star]:
    """
    Exact ReLU reachability for a single Star set.

    Algorithm:
    1. Estimate ranges for all dimensions
    2. Reset neurons that are always ≤ 0
    3. Split neurons that cross 0 boundary (``_step_relu``)

    Args:
        I: Input Star
        lp_solver: LP solver
        verbose: Display option
        precomputed_bounds: Optional (lb, ub) from the bound pre-pass. They
            only save LPs here: each is widened by the LP pad first (see
            below), so a neuron they leave near 0 goes to ``_step_relu``.

    Returns:
        List of output Stars
    """
    if I is None or I.dim == 0:
        return []

    # Estimate ranges
    lb, ub = I.estimate_ranges()

    if lb is None or ub is None:
        return []

    # Refine bounds with precomputed pre-pass bounds if available
    if precomputed_bounds is not None:
        # SOUNDNESS: the pre-pass bounds (IBP / zonotope) are round-to-nearest
        # and can sit ulps inside the true range (measured: IBP upper bound 0.0
        # for a true max of +2.8e-17, which zeroed the neuron). Widen them by
        # the LP pad before they may decide a phase, then intersect.
        lb, ub = _intersect_precomputed(lb, ub, precomputed_bounds, pad=True)

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

    # Profiler: neuron stability classification, summed across the per-star
    # population. reset_map = always inactive (ub<=0); split_map = unstable
    # (lb<0<ub); the remainder active. n_neurons (static) is at the layer level.
    record_classification(I.dim, len(reset_map), len(split_map))

    # Recursively split each uncertain neuron
    for i, neuron_idx in enumerate(split_map):
        if verbose:
            logger.debug(f'Exact ReLU_{neuron_idx} ({i+1}/{len(split_map)})')

        new_stars = []
        for star in current_stars:
            split_result = _step_relu(star, neuron_idx, lp_solver)
            new_stars.extend(split_result)

        current_stars = new_stars

    return current_stars


def _step_relu(I: Star, index: int, lp_solver: "LPSolver | str" = LPSolver.DEFAULT) -> List[Star]:
    """
    Split a single neuron in ReLU (exact step reach).

    The phase comes from the padded LP range (``_phase_range``):

    - padded min >= 0 / padded max <= 0: stable, one exact star;
    - one raw bound within its pad of 0 (the LP cannot tell the sign of a
      band that thin) and the other outside its pad: one star with a sliver
      variable covering the band (``_phase_sliver_star``), not a split -- a
      bound of exactly 0 is common (stacked ReLUs, identity layers, zero
      biases) and splitting on it doubled the star count per such neuron;
    - otherwise two stars, x <= 0 (zeroed) and x >= 0, which cover ``I``.
      This includes a neuron whose whole range lies within the pads: the
      sliver there would be decoupled from x and many times wider than the
      range (a loss of exactness that shows on small-scale layers), while
      the split is exact. If the raw range is on one side of 0, the side is
      kept without a split when the other child is certified empty
      (``_thin_neuron_side``).

    Args:
        I: Input Star
        index: Neuron index to split
        lp_solver: LP solver (the phase LPs use HiGHS, see
            ``_bound_lp_solver``)

    Returns:
        List of 0, 1 or 2 Star sets (0 only when ``I`` is certified empty)
    """
    rng = _phase_range(I, index, lp_solver)
    if rng is None:
        return []
    xmin, xmax, pad_lo, pad_hi = rng
    max_in_band = xmax <= 2.0 * pad_hi
    min_in_band = -xmin <= 2.0 * pad_lo

    if xmin >= 0:
        # Always active (the exact LP resolved an estimated-unstable neuron)
        count("n_resolved", 1)
        return [I]

    elif xmax <= 0:
        # Always inactive - zero out (estimated-unstable resolved by exact LP)
        count("n_resolved", 1)
        return [_reset_star_rows(I, np.array([index]))]

    side = None
    if max_in_band and min_in_band:
        side = _thin_neuron_side(I, index, xmin, xmax, pad_lo, pad_hi, lp_solver)

    if side == 1:
        count("n_resolved", 1)
        return [I]

    elif side == -1:
        count("n_resolved", 1)
        return [_reset_star_rows(I, np.array([index]))]

    elif max_in_band and not min_in_band:
        # Raw max within its pad of 0: zero the row and add y in [0, xmax],
        # which covers relu(x) whatever the sign of the true max.
        count("n_resolved", 1)
        return [_phase_sliver_star(I, index, 0.0, xmax)]

    elif min_in_band and not max_in_band:
        # Raw min within its pad of 0: y = x + t with t in [0, -xmin], which
        # covers relu(x) - x = max(0, -x) whatever the sign of the true min.
        count("n_resolved", 1)
        return [_phase_sliver_star(I, index, 1.0, -xmin)]

    else:
        # Split into two cases
        count("n_split", 1)
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
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    relax_method: str = 'standard',
    precomputed_bounds: tuple = None,
) -> List[Star]:
    """
    Approximate reachability for ReLU using triangle relaxation.

    Uses triangle over-approximation for neurons crossing zero instead of
    splitting. This is faster but produces wider (conservative) bounds.

    Args:
        input_stars: List of input Stars
        relax_factor: 0 = exact, 1 = maximum relaxation
        lp_solver: LP solver (LPs that decide a phase use HiGHS, see
            ``_bound_lp_solver``)
        relax_method: Relaxation strategy - 'standard', 'range', 'area', 'bound'
        precomputed_bounds: Optional (lb, ub) of the pre-activations. Unlike
            the exact path, the approximate paths use them as given (they may
            decide a phase), so they must enclose the true range; nn/bab.py
            relies on this to pin a forced neuron's bound to exactly 0. See
            ``_intersect_precomputed`` for bounds that cannot belong to the
            layer (another size raises; missing the star's enclosure drops
            them).

    Returns:
        List of output Stars: one per input star for ``relax_factor > 0``.
        ``relax_factor == 0`` is the exact reach (``relu_star_exact``),
        which splits and can return several stars per input star.
    """
    if relax_factor == 0.0:
        return relu_star_exact(input_stars, lp_solver, precomputed_bounds=precomputed_bounds)

    # Profiler: static neuron count, once per layer (no-op when disabled).
    # (The relax_factor==0 branch above delegates to relu_star_exact, which
    # records it; record here only for the genuinely-approximate path.)
    record_layer_neurons(input_stars)

    output_stars = []

    for star in input_stars:
        # Convert ImageStar to Star for 2D ReLU processing
        star_2d = star.to_star() if isinstance(star, ImageStar) else star

        # Process each star through approximate ReLU with specified method
        if relax_method == 'range':
            result = _relu_single_star_relax_range(star_2d, relax_factor, lp_solver, precomputed_bounds)
        elif relax_method == 'area':
            result = _relu_single_star_relax_area(star_2d, relax_factor, lp_solver, precomputed_bounds)
        elif relax_method == 'bound':
            result = _relu_single_star_relax_bound(star_2d, relax_factor, lp_solver, precomputed_bounds)
        else:  # 'standard'
            result = _relu_single_star_approx(star_2d, lp_solver, precomputed_bounds)

        if result is not None:
            # Preserve ImageStar type if input was ImageStar
            result = _preserve_imagestar_type(star, result)
            output_stars.append(result)

    return output_stars


def _relu_single_star_approx(
    I: Star,
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    precomputed_bounds: tuple = None,
) -> Optional[Star]:
    """
    Approximate ReLU reachability for a single Star using triangle
    relaxation.

    Uses estimate_ranges() (predicate bounds) to classify neurons:
      1. ub <= 0: inactive (zero out)
      2. lb >= 0: active (keep as-is)
      3. lb < 0 < ub: unstable (apply triangle relaxation)

    For unstable neurons, adds triangle constraints:
      y >= 0, y >= x, y <= (ub/(ub-lb))*(x - lb)

    This may over-count unstable neurons compared to LP-based
    classification, but avoids expensive per-neuron LP solves.

    Future work: investigate whether LP-based classification
    (get_range per neuron) produces tighter bounds that justify
    the additional cost, especially for deep networks where
    over-approximation error compounds.

    Args:
        I: Input Star.
        lp_solver: LP solver (unused, kept for API compatibility).
        precomputed_bounds: Optional tighter bounds from Zono
            pre-pass as (lb, ub) arrays.

    Returns:
        Output Star (single star, no splitting), or None if
        input is empty.
    """
    if I is None or I.dim == 0:
        return None

    # Estimate ranges to classify neurons
    lb_est, ub_est = I.estimate_ranges()

    if lb_est is None or ub_est is None:
        return None

    lb_est = lb_est.flatten()
    ub_est = ub_est.flatten()

    # Refine bounds with precomputed Zono pre-pass bounds
    if precomputed_bounds is not None:
        lb_est, ub_est = _intersect_precomputed(lb_est, ub_est, precomputed_bounds)

    # Classify neurons
    reset_map = np.where(ub_est <= 0)[0]
    crossing_map = np.where((lb_est < 0) & (ub_est > 0))[0]

    # Profiler: classification + relaxation, summed across the per-star
    # population (approx never splits). n_neurons (static) is at the layer level.
    record_classification(I.dim, len(reset_map), len(crossing_map))
    count("n_relaxed", len(crossing_map))

    # Zero out inactive neurons
    V = I.V.copy()
    V[reset_map, :] = 0

    if I.Z is not None:
        c1 = I.Z.c.copy()
        c1[reset_map] = 0
        V1 = I.Z.V.copy()
        V1[reset_map, :] = 0
        new_Z = Zono(c1, V1)
    else:
        new_Z = None

    current_star = Star(
        V, I.C, I.d, I.predicate_lb, I.predicate_ub,
        outer_zono=new_Z,
    )

    if len(crossing_map) == 0:
        return current_star

    # Apply triangle relaxation for unstable neurons
    return _apply_triangle_approx_multi(
        current_star, crossing_map,
        lb_est[crossing_map], ub_est[crossing_map],
    )


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
        List of output Zonotopes (over-approximation); an ImageZono input
        gives an ImageZono output
    """
    record_layer_neurons(input_zonos)  # static neuron count, once per layer

    output_zonos = []

    for zono in input_zonos:
        result = _relu_single_zono(zono)
        # SOUNDNESS: ReLU is element-wise, so the rows stay in the input's
        # (H, W, C) order. Returned as a plain Zono, a following Flatten
        # skipped its HWC -> CHW reorder and the next Linear read the wrong
        # neurons (measured: a false UNSAT).
        if isinstance(zono, ImageZono):
            result = ImageZono(result.c, result.V, zono.height, zono.width,
                               zono.num_channels)
        output_zonos.append(result)

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
    lbf, ubf = lb.flatten(), ub.flatten()
    inactive = ubf <= 0
    # ~(lbf >= 0) instead of (lbf < 0): identical for finite bounds, but a NaN
    # lower bound (inf-inf upstream overflow) must land in CROSSING -- where
    # the NaN lam/err arithmetic poisons the output loudly, matching the old
    # per-neuron loop's else-branch -- not in the stably-active catch-all,
    # which would silently pass a possibly-negative neuron through as identity
    # (an under-approximation on the sound-reach side).
    crossing = (~inactive) & ~(lbf >= 0)

    if is_enabled():
        n_inactive = int(inactive.sum())
        n_unstable = int(crossing.sum())
        record_classification(I.dim, n_inactive, n_unstable)
        count("n_relaxed", n_unstable)

    new_c = I.c.copy()
    new_V = I.V.copy()

    if inactive.any():
        new_c[inactive] = 0.0
        new_V[inactive, :] = 0.0

    # Crossing neurons, vectorized (semantics identical to the former
    # per-neuron loop): lam-scaled row + center shift, one error generator per
    # neuron appended in neuron order -- a SINGLE hstack instead of one
    # whole-matrix copy per neuron.
    ci = np.flatnonzero(crossing)
    if ci.size:
        li, ui = lbf[ci], ubf[ci]
        den = ui - li
        lam = np.where(den != 0.0, ui / np.where(den != 0.0, den, 1.0), 0.0)
        err = 0.5 * (1.0 - lam) * ui
        new_c[ci, 0] = lam * I.c[ci, 0] + err
        new_V[ci, :] = lam[:, None] * I.V[ci, :]
        # SOUNDNESS: the error generator IS the over-approximation of the
        # ReLU graph for a crossing neuron; dropping it whenever it is merely
        # small (the old ``err > 1e-10``) under-approximated the set (measured:
        # false UNSAT). Drop only the exact zero case.
        keep = err > 0.0
        if keep.any():
            k = int(keep.sum())
            E = np.zeros((I.dim, k))
            E[ci[keep], np.arange(k)] = err[keep]
            new_V = np.hstack([new_V, E])

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
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    precomputed_bounds: tuple = None,
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

    # Refine bounds with precomputed Zono pre-pass bounds if available
    if precomputed_bounds is not None:
        lb, ub = _intersect_precomputed(lb, ub, precomputed_bounds)

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

    # Profiler: per-star classification (no-op when disabled)
    record_classification(I.dim, len(map1), len(map2))

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
    xmax = _get_maxs(I, map21, lp_solver, ub[map21]) if len(map21) > 0 else np.array([])

    # Find neurons that are actually inactive after optimization
    map3 = np.where(xmax <= 0)[0] if len(xmax) > 0 else np.array([], dtype=int)
    map4 = map21[map3] if len(map3) > 0 else np.array([], dtype=int)

    # Reset newly found inactive neurons
    np.concatenate([map1, map4]) if len(map4) > 0 else map1
    In = In if len(map4) == 0 else _reset_star_rows(In, map4)

    # Step 6: Optimize lower bounds of neurons that are still crossing
    map5 = np.where(xmax > 0)[0] if len(xmax) > 0 else np.array([], dtype=int)
    map6 = map21[map5] if len(map5) > 0 else np.array([], dtype=int)
    xmax1 = xmax[map5] if len(map5) > 0 else np.array([])

    xmin = _get_mins(I, map6, lp_solver, lb[map6]) if len(map6) > 0 else np.array([])

    map7 = np.where(xmin < 0)[0] if len(xmin) > 0 else np.array([], dtype=int)
    map8 = map6[map7] if len(map7) > 0 else np.array([], dtype=int)
    lb2 = xmin[map7] if len(map7) > 0 else np.array([])
    ub2 = xmax1[map7] if len(map7) > 0 else np.array([])

    # Step 7: Combine all crossing neurons and apply triangle approximation
    map9 = np.concatenate([map22, map8]) if len(map8) > 0 else map22
    lb3 = np.concatenate([lb1, lb2]) if len(lb2) > 0 else lb1
    ub3 = np.concatenate([ub1, ub2]) if len(ub2) > 0 else ub1

    # Profiler: relaxed vs LP-resolved (no-op when disabled)
    record_relax_outcome(len(map2), len(map9))

    if len(map9) == 0:
        return In

    # Apply multi-neuron triangle approximation in one shot
    result = _apply_triangle_approx_multi(In, map9, lb3, ub3)

    return result


def _relu_single_star_relax_area(
    I: Star,
    relax_factor: float,
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    precomputed_bounds: tuple = None,
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

    # Refine bounds with precomputed Zono pre-pass bounds if available
    if precomputed_bounds is not None:
        lb, ub = _intersect_precomputed(lb, ub, precomputed_bounds)

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

    # Profiler: per-star classification (no-op when disabled)
    record_classification(I.dim, len(map1), len(map2))

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
    xmax = _get_maxs(I, map21, lp_solver, ub[map21]) if len(map21) > 0 else np.array([])

    map3 = np.where(xmax <= 0)[0] if len(xmax) > 0 else np.array([], dtype=int)
    map4 = map21[map3] if len(map3) > 0 else np.array([], dtype=int)

    np.concatenate([map1, map4]) if len(map4) > 0 else map1
    In = In if len(map4) == 0 else _reset_star_rows(In, map4)

    map5 = np.where(xmax > 0)[0] if len(xmax) > 0 else np.array([], dtype=int)
    map6 = map21[map5] if len(map5) > 0 else np.array([], dtype=int)
    xmax1 = xmax[map5] if len(map5) > 0 else np.array([])

    xmin = _get_mins(I, map6, lp_solver, lb[map6]) if len(map6) > 0 else np.array([])

    map7 = np.where(xmin < 0)[0] if len(xmin) > 0 else np.array([], dtype=int)
    map8 = map6[map7] if len(map7) > 0 else np.array([], dtype=int)
    lb2 = xmin[map7] if len(map7) > 0 else np.array([])
    ub2 = xmax1[map7] if len(map7) > 0 else np.array([])

    map9 = np.concatenate([map22, map8]) if len(map8) > 0 else map22
    lb3 = np.concatenate([lb1, lb2]) if len(lb2) > 0 else lb1
    ub3 = np.concatenate([ub1, ub2]) if len(ub2) > 0 else ub1

    # Profiler: relaxed vs LP-resolved (no-op when disabled)
    record_relax_outcome(len(map2), len(map9))

    if len(map9) == 0:
        return In

    result = _apply_triangle_approx_multi(In, map9, lb3, ub3)

    return result


def _relu_single_star_relax_bound(
    I: Star,
    relax_factor: float,
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    precomputed_bounds: tuple = None,
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

    # Refine bounds with precomputed Zono pre-pass bounds if available
    if precomputed_bounds is not None:
        lb, ub = _intersect_precomputed(lb, ub, precomputed_bounds)

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

    # Profiler: per-star classification (no-op when disabled)
    record_classification(I.dim, len(map1), len(map2))

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

    # Optimize upper bounds via LP for the selected neurons.
    if len(map21) > 0:
        xmax = _get_maxs(I, map21, lp_solver, ub[map21])
        map3 = np.where(xmax <= 0)[0]
        map4 = map21[map3] if len(map3) > 0 else np.array([], dtype=int)
    else:
        xmax = np.array([])
        map4 = np.array([], dtype=int)

    # Neurons proven inactive by the refined upper bound: reset rows.
    In = In if len(map4) == 0 else _reset_star_rows(In, map4)

    # Remove newly inactive neurons from lb optimization list
    if len(map4) > 0:
        map23 = np.setdiff1d(map22, map4)
    else:
        map23 = map22

    # Optimize lower bounds via LP for the selected neurons.
    if len(map23) > 0:
        xmin = _get_mins(I, map23, lp_solver, lb[map23])
        map9 = np.where(xmin >= 0)[0]
        proven_active = map23[map9] if len(map9) > 0 else np.array([], dtype=int)
    else:
        xmin = np.array([])
        proven_active = np.array([], dtype=int)

    # Issue #15 fix: EVERY neuron that still crosses zero must receive
    # triangle constraints. The previous logic constrained only
    # ``unselected + (lb-optimized still-negative)`` -- ub-optimized
    # neurons whose refined max stayed positive were left with
    # unconstrained identity rows, so the output star excluded true
    # outputs by O(1) (line-star repro in issue #15: the output was the
    # unchanged input star).
    #
    # Still-crossing = map2
    #   minus proven-inactive (map4: refined ub <= 0, rows reset above)
    #   minus proven-active  (refined lb >= 0, identity rows are exact).
    refined_ub = {int(i): float(v) for i, v in zip(map21, xmax)}
    refined_lb = {int(i): float(v) for i, v in zip(map23, xmin)}

    n_drop = len(map4) + len(proven_active)
    drop = (np.concatenate([map4, proven_active])
            if n_drop > 0 else np.array([], dtype=int))
    crossing_neurons = np.setdiff1d(map2, drop)

    # Profiler: relaxed vs LP-resolved (no-op when disabled)
    record_relax_outcome(len(map2), len(crossing_neurons))

    if len(crossing_neurons) == 0:
        return In

    # Best-available bounds: LP-refined where computed, estimated otherwise.
    lb_arr = np.array(
        [refined_lb.get(int(i), lb[i]) for i in crossing_neurons])
    ub_arr = np.array(
        [refined_ub.get(int(i), ub[i]) for i in crossing_neurons])

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

    # Case 3: y[i] <= a[i]*(x[i]-lb[i]), a[i] = ub[i]/(ub[i]-lb[i])
    # SOUNDNESS: the face must pass on or above the vertex (u, u). Any slope
    # a >= u/(u-l) does (a(u-l) >= u at x=u, and a(x-l) >= 0 for x >= l), so
    # the rounded quotient is raised by 8 units of roundoff (2^-50 relative),
    # which covers the rounding of u - l and of the division. Callers pass
    # only crossing neurons (l < 0 < u), so u - l > 0 needs no guard; the old
    # ``+ 1e-10`` in the denominator lowered the face below (u, u) and cut
    # true outputs off (measured: a false UNSAT on a 1e-4-wide neuron).
    lbs = np.asarray(lbs, dtype=np.float64).reshape(-1)
    ubs = np.asarray(ubs, dtype=np.float64).reshape(-1)
    a = (ubs / (ubs - lbs)) * (1.0 + 2.0 ** -50)
    b = a * lbs
    G3 = a.reshape(-1, 1) * I.V[indices, 1:n+1]
    C3 = np.hstack([-G3, np.eye(m)])
    # The row is stored rounded (a*c - a*l and each a*v_j): move its right-hand
    # side out by a rigorous bound on that rounding over the predicate box.
    d3 = a * I.V[indices, 0] - b
    if I.predicate_lb is not None and I.predicate_ub is not None:
        d3 = d3 + outward_interval_slack(np.abs(a * I.V[indices, 0]) + np.abs(b),
                                         G3, I.predicate_lb, I.predicate_ub)
    d3 = d3.reshape(-1, 1)

    new_C = np.vstack([C0, C1, C2, C3])
    new_d = np.vstack([d0, d1, d2, d3])

    # Update predicate bounds
    new_pred_lb = np.vstack([I.predicate_lb, np.zeros((m, 1))]) if I.predicate_lb is not None else None
    new_pred_ub = np.vstack([I.predicate_ub, ubs.reshape(-1, 1)]) if I.predicate_ub is not None else None

    return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub, outer_zono=None)


def _get_maxs(
    star: Star,
    indices: np.ndarray,
    lp_solver: "LPSolver | str",
    fallback: np.ndarray,
) -> np.ndarray:
    """
    Sound upper bounds for multiple state dimensions, for phase decisions.

    Equivalent to MATLAB Star.getMaxs(map), made sound: each LP maximum is
    padded outward (``_lp_pad``, HiGHS via ``_bound_lp_solver``) and capped by
    ``fallback``. A dimension whose LP returns no optimum (failure, or a
    presolve 'infeasible') keeps ``fallback`` -- reading it as 0 zeroed a
    possibly-positive neuron. ``fmin`` lets ``fallback`` govern a NaN.

    Args:
        star: Star set
        indices: Array of dimension indices
        lp_solver: LP solver requested by the caller
        fallback: Rigorous upper bounds for ``indices`` (the interval
            estimate), used when the LP does not decide and as a cap

    Returns:
        Array of upper bounds, one per index
    """
    solver = _bound_lp_solver(lp_solver)
    fallback = np.asarray(fallback, dtype=np.float64).reshape(-1)
    xmax = fallback.copy()

    for i, idx in enumerate(indices):
        result = star.get_max(int(idx), solver)
        if result is not None:
            xmax[i] = np.fmin(result + _lp_pad(result), fallback[i])

    return xmax


def _get_mins(
    star: Star,
    indices: np.ndarray,
    lp_solver: "LPSolver | str",
    fallback: np.ndarray,
) -> np.ndarray:
    """
    Sound lower bounds for multiple state dimensions, for phase decisions.

    Equivalent to MATLAB Star.getMins(map), made sound as ``_get_maxs``: LP
    minima padded outward and floored by ``fallback``; ``fallback`` where the
    LP returns no optimum.

    Args:
        star: Star set
        indices: Array of dimension indices
        lp_solver: LP solver requested by the caller
        fallback: Rigorous lower bounds for ``indices``

    Returns:
        Array of lower bounds, one per index
    """
    solver = _bound_lp_solver(lp_solver)
    fallback = np.asarray(fallback, dtype=np.float64).reshape(-1)
    xmin = fallback.copy()

    for i, idx in enumerate(indices):
        result = star.get_min(int(idx), solver)
        if result is not None:
            xmin[i] = np.fmax(result - _lp_pad(result), fallback[i])

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
    from n2v.config import config as global_config
    # Use star_parallel setting if available, otherwise check if parallel is enabled
    if hasattr(global_config, 'star_parallel'):
        return global_config.star_parallel and n_stars >= 2
    elif hasattr(global_config, 'parallel_lp'):
        # If LP parallel is enabled, also enable Star parallel for n_stars >= 2
        return global_config.parallel_lp and n_stars >= 2

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
    from n2v.config import config as global_config
    workers = global_config.n_workers if hasattr(global_config, 'n_workers') else 4

    # Don't use more workers than Stars
    return max(1, min(workers, n_stars))


def _relu_star_exact_parallel(
    input_stars: List[Star],
    lp_solver: "LPSolver | str" = LPSolver.DEFAULT,
    verbose: bool = False,
    n_workers: int = None,
    precomputed_bounds: tuple = None,
) -> List[Star]:
    """
    Process multiple Stars through exact ReLU in parallel.

    Uses ProcessPoolExecutor to distribute Stars across workers.

    Args:
        input_stars: List of input Stars
        lp_solver: LP solver
        verbose: Display option
        n_workers: Number of workers
        precomputed_bounds: Optional precomputed (lb, ub) from Zono pre-pass

    Returns:
        List of output Stars
    """
    workers = _get_star_workers(len(input_stars), n_workers)

    if workers == 1 or len(input_stars) == 1:
        # Fall back to sequential
        output_stars = []
        for star in input_stars:
            # Convert ImageStar to Star for 2D ReLU processing
            star_2d = star.to_star() if isinstance(star, ImageStar) else star
            result = _relu_single_star_exact(star_2d, lp_solver, verbose, precomputed_bounds)
            # Preserve ImageStar type for each output star
            result = [_preserve_imagestar_type(star, s) for s in result]
            output_stars.extend(result)
        return output_stars

    # Parallel processing
    # Profiler caveat: worker PROCESSES can't share the parent's region tree, so
    # their counts (n_split/n_lp_solves/classification) are lost — totals read
    # low here. Warn once so the numbers aren't silently CPU-count-dependent.
    # (See plan §9 "process-pool record-return".) Run n_workers=1 for accuracy.
    if is_enabled():
        warnings.warn(
            "Profiler: exact-ReLU ProcessPool path active; per-worker counters "
            "are not captured (totals will read low). Use n_workers=1 for "
            "accurate profiling.",
            RuntimeWarning, stacklevel=2,
        )

    # Convert ImageStars to Stars first (conversion is fast, not worth parallelizing)
    stars_2d = [s.to_star() if isinstance(s, ImageStar) else s for s in input_stars]
    output_stars = []

    if verbose:
        logger.info(f'  Processing {len(input_stars)} Stars in parallel ({workers} workers)')

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all Stars for processing, track original star for type preservation
        future_to_orig = {
            executor.submit(_relu_single_star_exact, star_2d, lp_solver, None, precomputed_bounds): orig_star
            for star_2d, orig_star in zip(stars_2d, input_stars)
        }

        # Collect results as they complete
        for future in as_completed(future_to_orig):
            orig_star = future_to_orig[future]
            try:
                result = future.result()
            except Exception as e:
                # SOUNDNESS: a failed star (an exception in the worker, or a
                # worker process that died: BrokenProcessPool) has no output.
                # Skipping it silently dropped every state it reaches, and a
                # verdict on the remaining stars was a false UNSAT. Cancel the
                # pending work and re-raise, as the sequential path does.
                logger.error(f'Exact ReLU failed on a star in a worker: {e!r}')
                for pending in future_to_orig:
                    pending.cancel()
                raise
            # Preserve ImageStar type for each output star
            result = [_preserve_imagestar_type(orig_star, s) for s in result]
            output_stars.extend(result)

    return output_stars


def _kernel_phase_ranges(I, solver: str):
    """Ranges of a Hexatope/Octatope for deciding ReLU phases.

    The LP/MCF optimum is padded outward (``_lp_pad``) and intersected with
    the rigorous interval bound of ``c + G x`` over the kernel box
    ``[-1, 1]`` (the bound ``estimate_ranges`` uses when no solver is given).
    That bound is exact where it can be: a row with no nonzero generator (a
    dead neuron) stays ``[c, c]``, and a box bound of 0 stays 0. Padding those
    alone turned each such neuron into a crossing one, split in two with no
    emptiness check (2^k sets for k of them).

    ``fmax``/``fmin`` let the interval bound govern if an LP bound is NaN.
    """
    lb, ub = I.get_ranges(solver=solver)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1, 1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1, 1)
    lb = lb - _lp_pad(lb)
    ub = ub + _lp_pad(ub)
    box = np.ones(I.nVar)
    elb, eub = outward_interval_bounds(I.center, I.generators, -box, box)
    return (np.fmax(lb, elb.reshape(-1, 1)), np.fmin(ub, eub.reshape(-1, 1)))


def _relu_single_hexatope(I: Hexatope, solver: str = None) -> List[Hexatope]:
    """
    Sound approximate ReLU for a single Hexatope.

    For each crossing neuron, splits into active (x_i >= 0) and inactive
    (x_i <= 0, then zero out) regions. Both are kept as separate sets.
    This produces up to 2^k sets where k = number of crossing neurons.

    Each intersect_half_space call uses Algorithm 5.1 (DCS bounding box)
    which is an over-approximation, so the result is always a sound
    over-approximation of the true ReLU output.

    Args:
        I: Input Hexatope
        solver: Optional solver method ('lp' or 'mcf').

    Returns:
        List of Hexatope sets (sound over-approximation of ReLU output)
    """
    if solver is not None:
        lb, ub = _kernel_phase_ranges(I, solver)
    else:
        lb, ub = I.estimate_ranges()
    n = I.dim

    # Step 1: Zero out always-inactive neurons via affine map
    inactive = np.where(ub.flatten() <= 0)[0]
    if len(inactive) > 0:
        W = np.eye(n, dtype=np.float64)
        for idx in inactive:
            W[idx, idx] = 0.0
        b = np.zeros((n, 1), dtype=np.float64)
        I = I.affine_map(W, b)

    # Step 2: Split crossing neurons
    crossing = np.where((lb.flatten() < 0) & (ub.flatten() > 0))[0]

    # Profiler: per-set neuron classification (no-op when disabled)
    record_classification(n, len(inactive), len(crossing))

    if len(crossing) == 0:
        return [I]

    current_sets = [I]
    for idx in crossing:
        new_sets = []
        for s in current_sets:
            # Active case: x_i >= 0 (identity pass-through)
            H_active = np.zeros((1, n))
            H_active[0, idx] = -1.0  # -x_i <= 0 means x_i >= 0
            g_active = np.array([[0.0]])
            active = s.intersect_half_space(H_active, g_active, solver=solver)

            # Inactive case: x_i <= 0, then zero out dimension i
            H_inactive = np.zeros((1, n))
            H_inactive[0, idx] = 1.0  # x_i <= 0
            g_inactive = np.array([[0.0]])
            inactive_set = s.intersect_half_space(H_inactive, g_inactive, solver=solver)
            W = np.eye(n, dtype=np.float64)
            W[idx, idx] = 0.0
            b = np.zeros((n, 1), dtype=np.float64)
            inactive_set = inactive_set.affine_map(W, b)

            new_sets.append(active)
            new_sets.append(inactive_set)
        # n_split = branches added this neuron: P sets each double (P->2P).
        count("n_split", len(current_sets))
        current_sets = new_sets

    return current_sets


def relu_hexatope(input_hexatopes: List[Hexatope], solver: str = None) -> List[Hexatope]:
    """
    ReLU for Hexatopes (sound over-approximation with splitting).

    Splits crossing neurons into active + inactive regions using
    intersect_half_space (Algorithm 5.1 DCS bounding box).

    Args:
        input_hexatopes: List of input Hexatope sets
        solver: Optional solver method ('lp' or 'mcf').
    """
    record_layer_neurons(input_hexatopes)  # static neuron count, once per layer
    output_hexatopes = []
    for hexatope in input_hexatopes:
        output_hexatopes.extend(_relu_single_hexatope(hexatope, solver=solver))
    return output_hexatopes


def relu_hexatope_approx(
    input_hexatopes: List[Hexatope],
    verbose: bool = False,
    solver: str = None
) -> List[Hexatope]:
    """
    Approximate reachability for ReLU using Hexatope sets.

    Sound over-approximation: splits crossing neurons into active + inactive
    regions. Each intersect_half_space uses Algorithm 5.1 (DCS bounding box).
    """
    return relu_hexatope(input_hexatopes, solver=solver)


def _relu_single_octatope(I: Octatope, solver: str = None) -> List[Octatope]:
    """
    Sound approximate ReLU for a single Octatope.

    Same algorithm as hexatope: splits crossing neurons into active + inactive
    cases. Uses Algorithm 5.1 (UTVPI bounding box) for intersect_half_space.

    Args:
        I: Input Octatope
        solver: Optional solver method ('lp' or 'mcf').
    """
    if solver is not None:
        lb, ub = _kernel_phase_ranges(I, solver)
    else:
        lb, ub = I.estimate_ranges()
    n = I.dim

    inactive = np.where(ub.flatten() <= 0)[0]
    if len(inactive) > 0:
        W = np.eye(n, dtype=np.float64)
        for idx in inactive:
            W[idx, idx] = 0.0
        b = np.zeros((n, 1), dtype=np.float64)
        I = I.affine_map(W, b)

    crossing = np.where((lb.flatten() < 0) & (ub.flatten() > 0))[0]

    # Profiler: per-set neuron classification (no-op when disabled)
    record_classification(n, len(inactive), len(crossing))

    if len(crossing) == 0:
        return [I]

    current_sets = [I]
    for idx in crossing:
        new_sets = []
        for s in current_sets:
            H_active = np.zeros((1, n))
            H_active[0, idx] = -1.0
            g_active = np.array([[0.0]])
            active = s.intersect_half_space(H_active, g_active, solver=solver)

            H_inactive = np.zeros((1, n))
            H_inactive[0, idx] = 1.0
            g_inactive = np.array([[0.0]])
            inactive_set = s.intersect_half_space(H_inactive, g_inactive, solver=solver)
            W = np.eye(n, dtype=np.float64)
            W[idx, idx] = 0.0
            b = np.zeros((n, 1), dtype=np.float64)
            inactive_set = inactive_set.affine_map(W, b)

            new_sets.append(active)
            new_sets.append(inactive_set)
        # n_split = branches added this neuron: P sets each double (P->2P).
        count("n_split", len(current_sets))
        current_sets = new_sets

    return current_sets


def relu_octatope(input_octatopes: List[Octatope], solver: str = None) -> List[Octatope]:
    """
    ReLU for Octatopes (sound over-approximation with splitting).

    Args:
        input_octatopes: List of input Octatope sets
        solver: Optional solver method ('lp' or 'mcf').
    """
    record_layer_neurons(input_octatopes)  # static neuron count, once per layer
    output_octatopes = []
    for octatope in input_octatopes:
        output_octatopes.extend(_relu_single_octatope(octatope, solver=solver))
    return output_octatopes


def relu_octatope_approx(
    input_octatopes: List[Octatope],
    verbose: bool = False,
    solver: str = None
) -> List[Octatope]:
    """Approximate reachability for ReLU using Octatope sets."""
    return relu_octatope(input_octatopes, solver=solver)
