"""Tests for certify_spec_disjoint (layer 3: AND across groups).

Early-exit semantics for epsilon_2: when ``certify_spec_disjoint``
finds a disjoint group it short-circuits and stops testing later groups.
Similarly, ``certify_group_disjoint`` short-circuits on the first
not-disjoint HalfSpace within a group. The Bonferroni bound is
summed over ONLY the HalfSpaces actually executed (i.e. the ones that
ran a hypothesis test on N samples). Unexecuted HalfSpaces do not
consume error budget. So
``epsilon_2 == sum(g.epsilon_2 for g in per_group_results)`` over the
executed prefix of groups, where each ``g.epsilon_2`` already aggregates
its executed-HalfSpaces sum.
"""
from __future__ import annotations
import math
import numpy as np
import pytest

from n2v.sets.halfspace import HalfSpace


def test_single_group_spec_unsat_iff_group_disjoint():
    """A spec with one group reduces to layer 2 on that group."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    groups = [[HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))]]
    result = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=groups,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.unsat_certified is True
    assert result.certifying_group_idx == 0


def test_two_group_spec_unsat_if_any_group_disjoint():
    """Group 0 reachable, group 1 unreachable. Spec is UNSAT (one group suffices)."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    groups = [
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))],   # reachable
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))],  # unreachable
    ]
    result = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=groups,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.unsat_certified is True
    assert result.certifying_group_idx == 1


def test_no_group_disjoint_means_not_unsat():
    """Every group reachable → can't certify UNSAT."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    groups = [
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))],   # reachable
        [HalfSpace(np.array([[0.0, 1.0]]), np.array([[100.0]]))],   # reachable
    ]
    result = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=groups,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.unsat_certified is False
    assert result.certifying_group_idx is None


def test_spec_epsilon_is_bonferroni_across_executed_hs():
    """epsilon_2_spec = sum over EXECUTED (group, hs) of log(1/beta_2)/n_samples.

    Group 0 has two unreachable HalfSpaces (both disjoint → group 0 is
    disjoint → spec UNSAT, early-exit before group 1). Within group 0,
    no early exit fires (every HS is disjoint), so 2 HalfSpace tests
    run. Group 1's HalfSpace never executes. Total: 2 executed tests.
    """
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    groups = [
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
         HalfSpace(np.array([[0.0, 1.0]]), np.array([[-100.0]]))],  # 2 hs (disjoint)
        [HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-100.0]]))],  # not executed
    ]
    result = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=groups,
        n_samples=500, beta_2=0.001, seed=0,
    )
    # 2 executed halfspaces × log(1000)/500 (group 1 skipped via spec early exit)
    assert result.epsilon_2 == pytest.approx(2 * math.log(1000) / 500, rel=1e-6)


def test_spec_early_exit_on_first_disjoint_group():
    """When group 0 is disjoint, we can stop and report UNSAT
    without running later groups. per_group_results may be shorter
    than len(spec_groups)."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    groups = [
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))],  # disjoint (UNSAT witness)
        [HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))],   # reachable (would fail)
    ]
    result = certify_spec_disjoint(
        flow_ode=flow, threshold_q=3.0, spec_groups=groups,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.unsat_certified is True
    assert result.certifying_group_idx == 0
    assert len(result.per_group_results) == 1  # early exit
    # epsilon is Bonferroni over EXECUTED HalfSpaces only — group 1
    # never ran a hypothesis test so it does not consume error budget.
    assert result.epsilon_2 == pytest.approx(1 * math.log(1000) / 500, rel=1e-6)


def test_empty_spec_raises():
    """An empty spec (no groups) is invalid."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    with pytest.raises(ValueError):
        certify_spec_disjoint(
            flow_ode=flow, threshold_q=3.0, spec_groups=[],
            n_samples=500, beta_2=0.001, seed=0,
        )


# --------------------------------------------------------------------------- #
# The keywords that a spec hands to                                            #
# each group, on an identity stand-in flow (no training). The AND semantics    #
# and the union bound over every executed test, a failed first group           #
# included, are in tests/soundness/test_soundness_scenario_verify.py.          #
# --------------------------------------------------------------------------- #
class _RecordingIdentityFlow:
    """A stand-in flow: the inverse map is the identity (float64) and every
    call records its keywords and its batch size."""

    def __init__(self):
        self.calls = []

    def inverse(self, z, **kwargs):
        import torch
        self.calls.append(dict(kwargs, batch=int(z.shape[0])))
        return z.detach().to(torch.float64).clone()


def _at_least(i, bound):
    """The unsafe half-space {y_i >= bound} as G y <= g, in two dimensions."""
    G = np.zeros((1, 2))
    G[0, i] = -1.0
    return HalfSpace(G, np.array([[-bound]]))


def test_spec_hands_its_seed_to_every_group():
    """With a seed, two spec checks agree bit for bit whatever the global
    numpy state, and each member check sees the sample of the member checked
    alone with that seed. Catches a seed that the spec does not forward."""
    from n2v.probabilistic.flow.scenario_verify import (
        certify_halfspace_disjoint, certify_spec_disjoint,
    )
    spec = [[_at_least(0, 0.25)], [_at_least(1, 1.25), _at_least(0, 1.5)]]
    flow = _RecordingIdentityFlow()
    np.random.seed(11)
    a = certify_spec_disjoint(flow, 1.0, spec, n_samples=64, beta_2=0.01, seed=4)
    np.random.seed(22)
    b = certify_spec_disjoint(flow, 1.0, spec, n_samples=64, beta_2=0.01, seed=4)
    assert a.unsat_certified and a.certifying_group_idx == 1
    for ga, gb, group in zip(a.per_group_results, b.per_group_results, spec):
        for ra, rb, hs in zip(ga.per_hs_results, gb.per_hs_results, group):
            alone = certify_halfspace_disjoint(flow, 1.0, hs, n_samples=64, beta_2=0.01, seed=4)
            np.testing.assert_array_equal(ra.worst_sample, rb.worst_sample)
            np.testing.assert_array_equal(ra.worst_sample, alone.worst_sample)


def test_spec_hands_the_adaptive_keywords_to_every_group():
    """A member with a marginal certificate is re-run with 900 samples inside
    the spec: the spec epsilon is ln(1/beta)/900 for that test. Catches
    adaptive keywords that the spec does not forward."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    r = certify_spec_disjoint(_RecordingIdentityFlow(), 1.0, [[_at_least(0, 1.25)]],
                              n_samples=100, beta_2=0.001, seed=0,
                              adaptive_threshold=0.5, adaptive_n_samples=900)
    assert r.unsat_certified is True
    assert r.per_group_results[0].per_hs_results[0].n_samples_used == 900
    assert abs(r.epsilon_2 - math.log(1000.0) / 900) <= 1e-15


class _RecordingLatentFlow:
    """A stand-in flow: the inverse map is the identity (float64) and every call
    records the latent points it was handed."""

    def __init__(self):
        self.latents = []

    def inverse(self, z, **kwargs):
        import torch
        self.latents.append(z.detach().numpy().copy())
        return z.detach().to(torch.float64).clone()


@pytest.mark.parametrize("strategy", ["uniform", "qmc"])
def test_spec_hands_its_sampling_strategy_to_every_group(strategy):
    """As for the group: every half-space check of the spec draws the latent points
    that a direct check with the spec's strategy and seed draws, and the spec's default
    is 'uniform'. The first group is hit by every sample (y_0 >= -50), so the second runs
    as well. Catches a strategy that the spec does not forward and a default other than
    'uniform'. (The test does not demand that 'qmc' hit a region outside the ball: the
    promise does not ask it.)"""
    from n2v.probabilistic.flow.scenario_verify import (
        certify_halfspace_disjoint, certify_spec_disjoint,
    )
    direct = {}
    for s in ("uniform", "qmc"):
        flow = _RecordingLatentFlow()
        certify_halfspace_disjoint(flow, 0.5, _at_least(0, 50.0), n_samples=256, beta_2=0.01,
                                   seed=0, sampling_strategy=s)
        direct[s] = flow.latents[0]
    if np.array_equal(direct["uniform"], direct["qmc"]):
        pytest.fail("the two strategies must draw different latent points")
    spec = [[_at_least(0, -50.0)], [_at_least(1, 50.0)]]
    for kwargs, expected in [(dict(sampling_strategy=strategy), direct[strategy]),
                             ({}, direct["uniform"])]:
        flow = _RecordingLatentFlow()
        r = certify_spec_disjoint(flow, 0.5, spec, n_samples=256, beta_2=0.01, seed=0, **kwargs)
        assert r.unsat_certified is True and r.certifying_group_idx == 1
        assert len(flow.latents) == 2
        for z in flow.latents:
            np.testing.assert_array_equal(z, expected)


def test_spec_hands_its_ode_keywords_to_every_group():
    """Every half-space check of every group integrates the flow with the
    spec's keywords."""
    from n2v.probabilistic.flow.scenario_verify import certify_spec_disjoint
    flow = _RecordingIdentityFlow()
    certify_spec_disjoint(flow, 1.0, [[_at_least(0, 0.0)], [_at_least(1, 5.0)]], n_samples=9,
                          beta_2=0.01, seed=0, t=0.5, n_ode_steps=4, ode_method='euler',
                          ode_atol=1e-2, ode_rtol=1e-3)
    assert flow.calls == [dict(t=0.5, n_steps=4, method='euler', atol=1e-2, rtol=1e-3,
                               batch=9)] * 2
