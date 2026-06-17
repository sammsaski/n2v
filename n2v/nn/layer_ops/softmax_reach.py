"""
Softmax reachability via sound interval relaxation.

For x in the per-dimension ranges [l, u] (LP-tightened),

    softmax_i(x) = 1 / (1 + sum_{j != i} exp(x_j - x_i))

and x_j - x_i ranges over [l_j - u_i, u_j - l_i], so

    lower_i = 1 / (1 + sum_{j != i} exp(u_j - l_i))
    upper_i = 1 / (1 + sum_{j != i} exp(l_j - u_i))

is a sound enclosure (exact on degenerate inputs). The output is a
box-shaped Star over fresh predicates — input correlations are dropped,
which is the price of the relaxation, but every true softmax output is
contained.

The whole flat vector is treated as ONE softmax group, which matches
the standard (1, N) logits layout. Softmax over an inner axis of a
higher-rank tensor would group differently and is rejected loudly.
"""

import numpy as np
from typing import List

from n2v.sets import Star, Box
from n2v.sets.image_star import ImageStar


def _softmax_bounds(lbs: np.ndarray, ubs: np.ndarray):
    """Sound per-dimension softmax bounds from input ranges."""
    n = lbs.size
    lower = np.zeros(n)
    upper = np.zeros(n)
    for i in range(n):
        others = np.arange(n) != i
        # exp args clipped to avoid overflow; 709 ~ log(float64 max)
        s_lo = np.exp(np.clip(ubs[others] - lbs[i], -745.0, 709.0)).sum()
        s_hi = np.exp(np.clip(lbs[others] - ubs[i], -745.0, 709.0)).sum()
        lower[i] = 1.0 / (1.0 + s_lo)
        upper[i] = 1.0 / (1.0 + s_hi)
    return lower, upper


def _check_dim(layer, ndim_hint: int = 2) -> None:
    """Accept only softmax over the last/feature axis of (1, N)."""
    dim = getattr(layer, 'dim', None)
    if dim is not None and dim not in (-1, 1, None):
        raise NotImplementedError(
            f"Softmax over axis {dim} is not supported — only the "
            f"feature axis of a flat (1, N) tensor")


def softmax_star(layer, input_stars: List[Star],
                 lp_solver: str = 'default') -> List[Star]:
    """Star reachability for Softmax (sound interval relaxation)."""
    _check_dim(layer)
    output_stars = []
    for s in input_stars:
        star = s.to_star() if isinstance(s, ImageStar) else s
        N = star.dim
        lbs = np.zeros(N)
        ubs = np.zeros(N)
        for i in range(N):
            l_val, u_val = star.get_range(i, lp_solver)
            if l_val is None or u_val is None:
                raise ValueError(
                    f"LP solver returned None for dimension {i} of "
                    f"Softmax input. Star may be infeasible.")
            lbs[i], ubs[i] = l_val, u_val
        lower, upper = _softmax_bounds(lbs, ubs)
        output_stars.append(Star.from_bounds(lower, upper))
    return output_stars


def softmax_box(layer, input_boxes: List) -> List:
    """Box reachability for Softmax."""
    _check_dim(layer)
    output = []
    for b in input_boxes:
        lower, upper = _softmax_bounds(
            np.asarray(b.lb, dtype=np.float64).flatten(),
            np.asarray(b.ub, dtype=np.float64).flatten())
        output.append(Box(lower, upper))
    return output
