"""Soundness invariants for verify_specification's sound (Star/Box) dispatch.

Locks the −150-critical rule for the UNSAT (holds) lane: a sound reach set may
be declared UNSAT ONLY when it is genuinely DISJOINT from the unsafe region. An
over-approximate reach set that still intersects the unsafe region MUST return
UNKNOWN, never a false UNSAT (a false UNSAT is an unsound 'holds' → −150).

These are the regression guard for the whole Track-B reach-tuning effort: any
change to the reach/relaxation paths that lets an intersecting set be reported
UNSAT must fail here.
"""

import numpy as np

from n2v.sets import Star
from n2v.sets.halfspace import HalfSpace
from n2v.utils.verify_specification import verify_specification


def _box_star(lb, ub):
    """A sound reach set that is exactly the output box [lb, ub]."""
    return Star.from_bounds(np.asarray(lb, float).reshape(-1, 1),
                            np.asarray(ub, float).reshape(-1, 1))


def test_unsat_only_when_disjoint():
    # Reach outputs in [0,1]^2; unsafe region y0 >= 2 is disjoint -> UNSAT (holds).
    rs = [_box_star([0.0, 0.0], [1.0, 1.0])]
    unsafe_far = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-2.0]]))  # -y0 <= -2
    assert verify_specification(rs, unsafe_far).verdict == 'UNSAT'


def test_overlap_is_unknown_never_false_unsat():
    # The soundness-critical direction: unsafe region y0 >= 0.5 INTERSECTS the
    # reach box [0,1]^2 -> must be UNKNOWN, never a (false) UNSAT.
    rs = [_box_star([0.0, 0.0], [1.0, 1.0])]
    overlap = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))  # -y0 <= -0.5
    assert verify_specification(rs, overlap).verdict == 'UNKNOWN'


def test_and_of_groups_unsat_if_any_group_disjoint():
    # AND across groups: safe iff ANY group is disjoint. group0 (y0>=0.5)
    # intersects, group1 (y1>=2) is disjoint -> overall UNSAT.
    rs = [_box_star([0.0, 0.0], [1.0, 1.0])]
    prop = [
        {'Hg': HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))},
        {'Hg': HalfSpace(np.array([[0.0, -1.0]]), np.array([[-2.0]]))},
    ]
    assert verify_specification(rs, prop).verdict == 'UNSAT'


def test_or_within_group_unknown_if_any_halfspace_intersects():
    # OR within a group: the group is unsafe if ANY halfspace intersects.
    # {y0>=2 (disjoint), y1>=0.5 (intersects)} -> group intersects -> UNKNOWN.
    rs = [_box_star([0.0, 0.0], [1.0, 1.0])]
    prop = [{'Hg': [HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-2.0]])),
                    HalfSpace(np.array([[0.0, -1.0]]), np.array([[-0.5]]))]}]
    assert verify_specification(rs, prop).verdict == 'UNKNOWN'


def test_multiple_reach_sets_unsat_only_if_all_disjoint():
    # A spec is safe only if EVERY reach set is disjoint from the unsafe region.
    # One set in [0,1], another in [3,4]; unsafe y0 >= 3 intersects the second
    # -> UNKNOWN (must not be UNSAT just because the first set is disjoint).
    rs = [_box_star([0.0], [1.0]), _box_star([3.0], [4.0])]
    unsafe = HalfSpace(np.array([[-1.0]]), np.array([[-3.0]]))  # y0 >= 3
    assert verify_specification(rs, unsafe).verdict == 'UNKNOWN'

    # Both disjoint from y0 >= 5 -> UNSAT.
    unsafe_far = HalfSpace(np.array([[-1.0]]), np.array([[-5.0]]))
    assert verify_specification(rs, unsafe_far).verdict == 'UNSAT'
