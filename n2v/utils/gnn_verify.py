"""
Verification specifications for GNN (GraphStar) reachability outputs.

A GNN reach set is a list of GraphStars over an (N, F_out) node-feature matrix.
Properties are expressed over the row-major flattened output space
(``index = node * F_out + feature``), matching :meth:`GraphStar.to_star`, and
delegated to the existing :func:`n2v.utils.verify_specification`, whose
``_is_disjoint`` already flattens any set exposing ``to_star``.

The typical specification is a per-node output box: prove that every selected
node-feature output stays within ``[lb, ub]``.  Under approximate (over-
approximate) reachability the verdict is ``'verified'`` or ``'unknown'``; a
definitive ``'falsified'`` requires a concrete counterexample (e.g. sampling),
which is out of scope here.
"""

from typing import List, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid a circular import at module load (n2v.sets -> n2v.utils)
    from n2v.sets import GraphStar, HalfSpace


def graph_output_index(node: int, feature: int, F: int) -> int:
    """Flattened row-major index of output[node, feature] (matches to_star)."""
    return node * F + feature


def box_violation_halfspaces(
    N: int,
    F: int,
    lb: np.ndarray,
    ub: np.ndarray,
    target_nodes: Optional[Sequence[int]] = None,
) -> List["HalfSpace"]:
    """Unsafe halfspaces for an output-box property over selected nodes.

    The property "output[node, f] in [lb[node, f], ub[node, f]] for all
    selected nodes and features" is violated iff some output leaves the box.
    Each violation is one halfspace over the flattened (N*F) space:

        out[idx] >= ub  ->  -out[idx] <= -ub
        out[idx] <= lb  ->   out[idx] <=  lb

    Passing the returned list (OR group) to :func:`verify_graph_output` proves
    the box holds iff none of these halfspaces intersect the reach set.

    Args:
        N, F: Output node and feature counts.
        lb, ub: (N, F) bound arrays.
        target_nodes: Optional node subset; defaults to all nodes.

    Returns:
        List of HalfSpace objects over the flattened output space.
    """
    from n2v.sets import HalfSpace

    lb = np.asarray(lb, dtype=np.float64).reshape(N, F)
    ub = np.asarray(ub, dtype=np.float64).reshape(N, F)
    nodes = range(N) if target_nodes is None else target_nodes
    dim = N * F

    halfspaces: List[HalfSpace] = []
    for n in nodes:
        for f in range(F):
            idx = graph_output_index(n, f, F)
            e = np.zeros((1, dim))
            e[0, idx] = 1.0
            # out[idx] <= lb  (below-lower violation)
            halfspaces.append(HalfSpace(e.copy(), np.array([lb[n, f]])))
            # out[idx] >= ub  ->  -out[idx] <= -ub  (above-upper violation)
            halfspaces.append(HalfSpace(-e.copy(), np.array([-ub[n, f]])))
    return halfspaces


def verify_graph_output(
    reach_sets: Sequence["GraphStar"],
    property: Union["HalfSpace", List["HalfSpace"], dict, List[dict]],
) -> str:
    """Verify a property against a GNN reach set.

    Args:
        reach_sets: Output GraphStars from a GNN reach.
        property: Unsafe-region specification accepted by
            :func:`verify_specification` (HalfSpace, list of HalfSpace, or
            property dict(s)).  A list of HalfSpaces is treated as one OR group
            and is verified iff *every* halfspace is disjoint from the reach set.

    Returns:
        'verified'  — reach set avoids the unsafe region (property holds),
        'unknown'   — over-approximation may intersect the unsafe region,
        'falsified' — reach set provably inside the unsafe region.
    """
    from n2v.sets import GraphStar
    from n2v.utils.verify_specification import verify_specification

    if not all(isinstance(s, GraphStar) for s in reach_sets):
        raise TypeError("verify_graph_output expects a sequence of GraphStars")

    result = verify_specification(list(reach_sets), property)
    return {"SAT": "falsified", "UNSAT": "verified", "UNKNOWN": "unknown"}[result.verdict]


def verify_node_bounds(
    reach_sets: Sequence["GraphStar"],
    lb: np.ndarray,
    ub: np.ndarray,
    target_nodes: Optional[Sequence[int]] = None,
) -> str:
    """Convenience: verify every selected node-feature output stays in [lb, ub].

    Infers (N, F) from the first reach set.  Returns 'verified' / 'unknown' /
    'falsified' as in :func:`verify_graph_output`.
    """
    if len(reach_sets) == 0:
        raise ValueError("reach_sets is empty")
    N, F = reach_sets[0].N, reach_sets[0].F
    hs = box_violation_halfspaces(N, F, lb, ub, target_nodes)
    return verify_graph_output(reach_sets, hs)
