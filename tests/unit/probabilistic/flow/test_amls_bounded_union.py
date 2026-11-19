"""Tests for union-mass bounded AMLS.

Covers ``amls_bounded_estimate_union_mass`` and
``amls_bounded_certify_spec_union``. The union estimator runs a single
in-ball MCMC chain on ``phi_union(y) = min_j phi_halfspace_j(y)``,
estimating ``Pr[y in union of halfspaces | z ~ pi_q]`` directly. The
single-HalfSpace path remains validated by the existing Phase 5e ACAS
Xu sweep; these tests focus on the new union path.
"""
from __future__ import annotations

import numpy as np
import torch

from n2v.probabilistic.flow.amls_bounded import (
    AMLSBoundedResult,
    AMLSBoundedSpecResult,
    amls_bounded_certify_spec,
    amls_bounded_certify_spec_union,
    amls_bounded_estimate_halfspace_mass,
    amls_bounded_estimate_union_mass,
)
from n2v.sets.halfspace import HalfSpace


class _IdentityFlow:
    """Trivial 'flow' whose inverse is the identity, so the data
    distribution coincides with the latent (truncated Gaussian on
    ``||z|| <= q``). Used to give known ground-truth probabilities."""

    def inverse(self, z, **_kw):
        return z

    def forward(self, y, **_kw):
        return y

    def eval(self):
        return self

    def to(self, *_args, **_kw):
        return self


def _hs(G_row, g_scalar):
    G = np.asarray([G_row], dtype=np.float64)
    g = np.asarray([[float(g_scalar)]], dtype=np.float64)
    return HalfSpace(G, g)


def test_union_single_halfspace_matches_single_estimator():
    """For a 1-halfspace OR group, union-mass should agree with the
    single-HalfSpace estimator on detection and pi_hat (modulo the tiny
    numerical drift that the extra ``stack/min`` of size 1 introduces).
    """
    flow = _IdentityFlow()
    hs = _hs([1.0, 0.0], -3.0)  # y_0 <= -3

    single = amls_bounded_estimate_halfspace_mass(
        flow, hs, q=4.0, n_samples_per_level=500,
        max_levels=10, seed=47,
    )
    union = amls_bounded_estimate_union_mass(
        flow, [hs], q=4.0, n_samples_per_level=500,
        max_levels=10, seed=47,
    )

    assert union.detected_unsafe is single.detected_unsafe
    assert union.levels_used == single.levels_used
    # On a 1-halfspace input the union-mass and single-mass paths walk
    # the same RNG sequence and compute the same phi values, so pi_hat
    # should be identical.
    assert np.isclose(union.pi_hat, single.pi_hat)


def test_union_detects_when_any_halfspace_overlaps():
    """An OR of K halfspaces all targeting the deep ``y_0 << 0`` tail
    is detectable: any one halfspace would be detected, so the union
    surely is.
    """
    flow = _IdentityFlow()
    halfspaces = [
        _hs([1.0], -3.0),    # y_0 <= -3 (mass ~ 1.35e-3)
        _hs([1.0], -3.5),    # y_0 <= -3.5 (rarer)
        _hs([1.0], -4.0),    # y_0 <= -4 (rarer still)
    ]

    res = amls_bounded_estimate_union_mass(
        flow, halfspaces, q=5.0, n_samples_per_level=500,
        max_levels=10, seed=47,
    )
    assert res.detected_unsafe is True
    assert res.pi_hat > 0.0


def test_union_certifies_disjoint_when_far():
    """An OR of halfspaces all far from the ``||z|| <= q`` ball should
    pass the gate (no detection, small pi_upper).
    """
    flow = _IdentityFlow()
    # All halfspaces require y_0 >= 5 or larger; ||z|| <= 2 keeps
    # |y_0| <= 2 ≪ 5, so no point can satisfy them.
    halfspaces = [
        _hs([-1.0], -5.0),   # y_0 >= 5
        _hs([-1.0], -6.0),   # y_0 >= 6
        _hs([-1.0], -7.0),   # y_0 >= 7
    ]
    res = amls_bounded_estimate_union_mass(
        flow, halfspaces, q=2.0, n_samples_per_level=500,
        max_levels=10, seed=47,
    )
    assert res.detected_unsafe is False
    assert res.pi_upper < 0.05


def test_certify_spec_union_single_group_certifies():
    """One group, OR over halfspaces, all far from the ball → UNSAT."""
    flow = _IdentityFlow()
    group = [_hs([-1.0], -5.0), _hs([-1.0], -6.0)]
    spec_groups = [group]

    res = amls_bounded_certify_spec_union(
        flow, spec_groups, q=2.0, eps_2_target=0.05,
        n_samples_per_level=500, max_levels=10, seed=47,
    )
    assert isinstance(res, AMLSBoundedSpecResult)
    assert res.unsat_certified is True
    assert res.detected_any is False
    # per_hs_results is List[List[Result]]; each inner list has exactly
    # one union-mass entry.
    assert len(res.per_hs_results) == 1
    assert len(res.per_hs_results[0]) == 1
    assert isinstance(res.per_hs_results[0][0], AMLSBoundedResult)


def test_certify_spec_union_and_of_or_one_disjoint_group_unsat():
    """AND-of-OR: two groups, only the second is disjoint. UNSAT
    requires *at least one* group disjoint, so should certify.

    The first group must overlap the ball. With ``y_0 <= -3``, which the
    ball ``|z| <= 2`` does not reach either, both groups would be disjoint
    and the case could not see a spec judged by its first group only (it
    would be "detected" only through a proposal outside the ball, a defect
    of its own). ``y_0 <= -1.5`` holds 4.6 % of the ball: 500 first-level
    samples miss it with probability 6e-11, so the first group is detected.
    """
    flow = _IdentityFlow()
    group_overlap = [_hs([1.0], -1.5)]      # y_0 <= -1.5 — 4.6 % of the ball
    group_disjoint = [_hs([-1.0], -5.0)]    # y_0 >= 5 — far from ball
    spec_groups = [group_overlap, group_disjoint]

    res = amls_bounded_certify_spec_union(
        flow, spec_groups, q=2.0, eps_2_target=0.05,
        n_samples_per_level=500, max_levels=10, seed=47,
    )
    assert res.per_hs_results[0][0].detected_unsafe is True
    assert res.per_hs_results[1][0].detected_unsafe is False
    assert res.unsat_certified is True


def test_certify_spec_union_no_disjoint_group_not_certified():
    """If every group's union mass overlaps the ball, no group is
    disjoint and unsat_certified must be False."""
    flow = _IdentityFlow()
    group_a = [_hs([1.0], -3.0)]   # y_0 <= -3 — has tail mass under N(0,I)
    group_b = [_hs([-1.0], -3.0)]  # y_0 >= 3 — has tail mass under N(0,I)
    spec_groups = [group_a, group_b]

    res = amls_bounded_certify_spec_union(
        flow, spec_groups, q=4.0, eps_2_target=1e-6,
        n_samples_per_level=500, max_levels=10, seed=47,
    )
    assert res.unsat_certified is False


def test_certify_spec_union_eps_target_drives_gate():
    """When pi_upper sits between two thresholds, raising eps_2_target
    should flip the certification."""
    flow = _IdentityFlow()
    spec_groups = [[_hs([1.0], -3.0)]]  # y_0 <= -3 — pi ~ 1.35e-3

    res_strict = amls_bounded_certify_spec_union(
        flow, spec_groups, q=4.0, eps_2_target=1e-6,
        n_samples_per_level=500, max_levels=10, seed=47,
    )
    res_loose = amls_bounded_certify_spec_union(
        flow, spec_groups, q=4.0, eps_2_target=0.5,
        n_samples_per_level=500, max_levels=10, seed=47,
    )
    # Same physical pi_upper, different gate. With pi ~ 1.35e-3,
    # strict (1e-6) refuses to certify, loose (0.5) accepts the bound
    # provided no detection occurred. Detection itself depends on the
    # sample, so we don't assert unsat directly on the loose case;
    # we only assert the strict case rejects.
    assert res_strict.unsat_certified is False
    if not res_loose.detected_any:
        assert res_loose.unsat_certified is True


def test_union_rejects_empty_halfspaces():
    """Empty halfspaces list is a programming error."""
    import pytest
    flow = _IdentityFlow()
    with pytest.raises(ValueError):
        amls_bounded_estimate_union_mass(
            flow, [], q=2.0, n_samples_per_level=100, seed=47,
        )


def test_union_rejects_dim_mismatched_halfspaces():
    """All halfspaces must share input dimension."""
    import pytest
    flow = _IdentityFlow()
    hs_2d = _hs([1.0, 0.0], -3.0)
    hs_1d = _hs([1.0], -3.0)
    with pytest.raises(ValueError):
        amls_bounded_estimate_union_mass(
            flow, [hs_2d, hs_1d], q=2.0,
            n_samples_per_level=100, seed=47,
        )


def test_max_levels_caps_union_loop():
    """``max_levels`` is the hard cap on the level-splitting loop. A
    deep-tail target should otherwise need more levels than the cap."""
    flow = _IdentityFlow()
    # Far tail so the chain would normally take many levels to detect.
    halfspaces = [_hs([1.0], -8.0), _hs([1.0], -8.5)]

    capped = amls_bounded_estimate_union_mass(
        flow, halfspaces, q=4.0, n_samples_per_level=200,
        max_levels=2, seed=47,
    )
    uncapped = amls_bounded_estimate_union_mass(
        flow, halfspaces, q=4.0, n_samples_per_level=200,
        max_levels=20, seed=47,
    )
    # The cap should be tight: capped run uses exactly the cap (or
    # fewer if it terminated early); uncapped run uses more.
    assert capped.levels_used <= 2
    assert uncapped.levels_used >= capped.levels_used
