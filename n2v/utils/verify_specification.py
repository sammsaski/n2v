"""
Verify a specification based on the intersection between the reach set and halfspaces.

This module provides functionality to verify properties by checking intersection
between the computed reachable set and the property (assumed to be the un-robust
or unsafe region to prove).

The router (verify_specification) parses the property format and dispatches to
type-specific verification functions:
- _verify_specification_box: O(n) interval arithmetic for Box/ProbabilisticBox sets
- _verify_specification_star: LP-based intersection for Star sets (default fallback)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Union, List

from n2v.sets import Star, Box, HalfSpace


def verify_specification(reach_set: Union[List[Star], List], property: Union[dict, List[dict], HalfSpace, List[HalfSpace]]) -> int:
    """
    Verify a specification based on intersection between reach set and property halfspaces.

    The property is assumed to represent the un-robust or unsafe region that we want
    to prove does NOT intersect with the reachable set.

    Dispatches to type-specific verification:
    - Box/ProbabilisticBox: interval arithmetic (fast, no LP)
    - Star: LP-based intersection (exact)
    - Other types: converted to Star as fallback

    Args:
        reach_set: Computed output set of neural network (list of set objects)
        property: Property specification, can be:
                  - dict with 'Hg' field containing HalfSpace(s)
                  - list of dicts with 'Hg' field (multiple groups, ANDed)
                  - HalfSpace object
                  - list of HalfSpace objects

    Returns:
        result: 0 -> property failed (intersection found, property violated)
                1 -> property satisfied (no intersection, property holds)
                2 -> unknown (may have intersection with both safe and unsafe)

    Note:
        - Multiple property groups (list of dicts): AND logic across groups.
          The unsafe region is the intersection of all groups. If ANY group is
          disjoint from all reach sets → satisfied.
        - Within a single group with multiple HalfSpaces: OR logic.
          If ANY halfspace intersects with ANY reach set → that group intersects.
        - Single HalfSpace: ALL reach sets must NOT intersect.
    """
    # Parse property into groups (AND of OR)
    groups = _parse_property_groups(property)

    # For AND logic across groups: if ANY group is fully disjoint → satisfied
    for group in groups:
        if _group_disjoint_from_reach_set(group, reach_set):
            return 1  # this group is disjoint → overall AND is disjoint → satisfied

    # All groups individually intersect → unknown
    return 2


def _parse_property_groups(property: Union[dict, List, HalfSpace]) -> List[List[HalfSpace]]:
    """
    Normalize property input into a list of groups (AND of OR).

    Each group is a list of HalfSpace objects (OR within group).
    Multiple groups are ANDed together.
    """
    # List of dicts: each dict is a property group (AND across groups)
    if isinstance(property, list) and len(property) > 0 and isinstance(property[0], dict):
        groups = []
        for p in property:
            hg = p['Hg']
            if isinstance(hg, HalfSpace):
                groups.append([hg])
            elif isinstance(hg, list):
                groups.append(hg)
            else:
                raise TypeError(f"Property group 'Hg' must be HalfSpace or list, got {type(hg)}")
        return groups
    elif isinstance(property, dict):
        hg = property['Hg']
        if isinstance(hg, HalfSpace):
            return [[hg]]
        elif isinstance(hg, list):
            return [hg]
        else:
            raise TypeError(f"Property 'Hg' must be HalfSpace or list, got {type(hg)}")

    # Single HalfSpace → one group with one halfspace
    if isinstance(property, HalfSpace):
        return [[property]]
    # List of HalfSpaces → one group with OR logic
    elif isinstance(property, list):
        return [property]
    else:
        raise TypeError(f"Property must be HalfSpace, list of HalfSpace, or dict with 'Hg' field, got {type(property)}")


def _group_disjoint_from_reach_set(group: List[HalfSpace], reach_set: list) -> bool:
    """
    Check if a group of halfspaces (OR) is disjoint from all reach sets.

    A group is disjoint from the reach set if for every reach set S and every
    halfspace in the group, S is disjoint from the halfspace.

    If the group has a single halfspace: all S must be disjoint from it.
    If the group has multiple halfspaces (OR): if any hs intersects any S → not disjoint.
    """
    for hs in group:
        for S in reach_set:
            if not _is_disjoint(S, hs):
                return False  # this halfspace intersects → group not disjoint
    return True  # all halfspaces disjoint from all reach sets


def _is_disjoint(S: Union[Star, Box], halfspace: HalfSpace) -> bool:
    """Check if set S is disjoint from halfspace. Dispatches by set type."""
    if isinstance(S, Box):
        return _verify_specification_box(S, halfspace)
    elif isinstance(S, Star):
        return _verify_specification_star(S, halfspace)
    elif hasattr(S, 'to_star'):
        return _verify_specification_star(S.to_star(), halfspace)
    else:
        raise TypeError(f"Cannot verify specification for {type(S)}")


def _verify_specification_box(box: Box, halfspace: HalfSpace) -> bool:
    """
    Check if a Box is disjoint from a halfspace using interval arithmetic.

    For halfspace Gx <= g with box [lb, ub], the intersection is empty iff
    there is no x in [lb, ub] satisfying all rows of Gx <= g.

    For each row i: min_{x in box} G_i·x = Σ_j min(G_ij*lb_j, G_ij*ub_j).
    If min > g_i for any row → that constraint is infeasible → disjoint.

    If all rows are individually feasible, we solve a box-bounded LP to check
    simultaneous feasibility (needed for multi-row halfspaces).

    Returns:
        True if disjoint (empty intersection), False if intersection may exist.
    """
    G = halfspace.G.astype(np.float64)
    g = halfspace.g.astype(np.float64).flatten()
    lb = box.lb.flatten()
    ub = box.ub.flatten()

    n_rows = G.shape[0]

    # Fast check: for each row, compute min(G_i · x) over the box.
    # min(G_i · x) = Σ_j min(G_ij * lb_j, G_ij * ub_j)
    for i in range(n_rows):
        row = G[i]
        min_val = np.sum(np.minimum(row * lb, row * ub))
        if min_val > g[i]:
            return True  # constraint i infeasible → disjoint

    # If only one row and it's feasible, intersection exists
    if n_rows == 1:
        return False

    # Multiple rows all individually feasible — check simultaneous feasibility
    # via box-bounded LP (only variable bounds + halfspace constraints)
    # Feasibility LP: minimize 0 subject to Gx <= g, lb <= x <= ub
    n = len(lb)
    c = np.zeros(n)
    bounds = list(zip(lb, ub))

    result = linprog(c, A_ub=G, b_ub=g, bounds=bounds, method='highs')

    # If infeasible → disjoint
    return not result.success


def _verify_specification_star(star: Star, halfspace: HalfSpace) -> bool:
    """
    Check if a Star is disjoint from a halfspace using LP-based intersection.

    Returns:
        True if disjoint (empty intersection), False if intersection may exist.
    """
    G = halfspace.G.astype(np.float64)
    g = halfspace.g.astype(np.float64)

    S = star.intersect_half_space(G, g)

    if S is None or (isinstance(S, list) and len(S) == 0) or (isinstance(S, Star) and S.is_empty_set()):
        return True  # empty intersection → disjoint
    return False
