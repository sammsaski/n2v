"""
Verify a specification based on the intersection between the reach set and halfspaces.

This module provides functionality to verify properties by checking intersection
between the computed reachable set and the property (assumed to be the un-robust
or unsafe region to prove).
"""

import numpy as np
from typing import Union, List
from nnv_py.sets import Star, HalfSpace


def verify_specification(reach_set: Union[List[Star], List], property: Union[dict, List[dict], HalfSpace, List[HalfSpace]]) -> int:
    """
    Verify a specification based on intersection between reach set and property halfspaces.

    The property is assumed to represent the un-robust or unsafe region that we want
    to prove does NOT intersect with the reachable set.

    Args:
        reach_set: Computed output set of neural network (list of Star objects)
        property: Property specification, can be:
                  - dict with 'Hg' field containing HalfSpace(s)
                  - list of dicts with 'Hg' field
                  - HalfSpace object
                  - list of HalfSpace objects

    Returns:
        result: 0 -> property failed (intersection found, property violated)
                1 -> property satisfied (no intersection, property holds)
                2 -> unknown (may have intersection with both safe and unsafe)

    Note:
        - Single HalfSpace: ALL reach sets must NOT intersect (AND logic)
        - Multiple HalfSpaces: If ANY intersects with ANY reach set -> unknown/unsafe (OR logic)
    """
    R = reach_set
    nr = len(R)  # number of output sets (for approx should be 1)

    # Process property to verify
    if isinstance(property, list) and len(property) > 0 and isinstance(property[0], dict):
        # Created from vnnlib (one or multiple halfSpaces)
        property = property[0]
        property = property['Hg']  # property transformed into HalfSpace(s)
    elif isinstance(property, dict):
        # Single dict
        property = property['Hg']

    # Ensure property is a list
    if isinstance(property, HalfSpace):
        property = [property]
    elif not isinstance(property, list):
        raise TypeError(f"Property must be HalfSpace, list of HalfSpace, or dict with 'Hg' field, got {type(property)}")

    # Begin verification
    np_halfspaces = len(property)

    if np_halfspaces == 1:
        # Only one halfspace - check if ALL reach sets do NOT intersect
        result = 1  # Assume property is satisfied (no intersection)

        for k in range(nr):
            Set = R[k]

            # Convert to Star if needed
            if not isinstance(Set, Star):
                if hasattr(Set, 'to_star'):
                    Set = Set.to_star()
                else:
                    raise TypeError(f"Cannot convert {type(Set)} to Star")

            # TODO: Handle GPU arrays if needed (in Python, likely using CuPy)
            # if isinstance(Set.V, cp.ndarray):  # CuPy array
            #     Set = Set.to_cpu()

            # Compute intersection with unsafe/not robust region
            G = property[0].G.astype(np.float64)
            g = property[0].g.astype(np.float64)

            S = Set.intersect_half_space(G, g)

            if S is None or (isinstance(S, list) and len(S) == 0):
                # No intersection with unsafe region = safe (unsat)
                result = 1
            else:
                # Intersection with safe and unsafe region = unknown or unsafe
                result = 2
                break

    else:
        # Multiple halfspaces, which means OR assertion
        # If ANY halfspace intersects with ANY reach set -> unknown/unsafe
        cp = 0  # current halfspace we are looking at (0-indexed in Python)
        result = 1  # start assuming property is unsat (no intersection)

        while cp < np_halfspaces:
            for k in range(nr):  # check every reach set vs OR property
                Set = R[k]

                # Convert to Star if needed
                if not isinstance(Set, Star):
                    if hasattr(Set, 'to_star'):
                        Set = Set.to_star()
                    else:
                        raise TypeError(f"Cannot convert {type(Set)} to Star")

                # TODO: Handle GPU arrays if needed
                # if isinstance(Set.V, cp.ndarray):
                #     Set = Set.to_cpu()

                G = property[cp].G.astype(np.float64)
                g = property[cp].g.astype(np.float64)

                S = Set.intersect_half_space(G, g)

                if S is None or (isinstance(S, list) and len(S) == 0):
                    # No intersection, continue to next reach set
                    continue
                else:
                    # Intersection found - unknown if approx, sat if exact
                    result = 2
                    return result

            cp += 1

    return result
