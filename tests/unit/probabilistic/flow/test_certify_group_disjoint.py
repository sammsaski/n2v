"""Tests for certify_group_disjoint (layer 2: OR within group).

Early-exit semantics for epsilon_2: when ``certify_group_disjoint``
short-circuits on the first not-disjoint HalfSpace, only the executed
HalfSpaces contribute to the Bonferroni bound — the unexecuted ones
were not subjected to a hypothesis test (no samples drawn for them),
so they do not consume any error budget. Therefore
``epsilon_2 == sum(r.epsilon_2 for r in per_hs_results)`` over the
executed members, NOT a planned ``len(group) * log(1/beta)/N`` total.
This matches standard scenario-theory accounting for sequentially-run
hypothesis tests with adaptive stopping.
"""
from __future__ import annotations
import math
import numpy as np
import pytest

from n2v.sets.halfspace import HalfSpace


def test_single_halfspace_group_reduces_to_layer_1():
    """A group with one HalfSpace is disjoint iff that HalfSpace is disjoint."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    group = [HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))]
    result = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.disjoint is True


def test_group_not_disjoint_if_any_hs_hit():
    """If even one HalfSpace in a group is reachable, the group is not disjoint."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    # HS_A: y_0 <= -100 (unreachable). HS_B: y_0 <= 100 (trivially reachable).
    group = [
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]])),
    ]
    result = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.disjoint is False


def test_group_disjoint_epsilon_is_bonferroni_over_hs():
    """epsilon_2 = |group| * log(1/beta_2) / n_samples."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    group = [
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
        HalfSpace(np.array([[0.0, 1.0]]), np.array([[-100.0]])),
    ]
    result = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.epsilon_2 == pytest.approx(
        2 * math.log(1000) / 500, rel=1e-6
    )


def test_group_disjoint_returns_per_hs_results():
    """The result includes per-HalfSpace sub-results. The first
    HalfSpace is disjoint (unreachable), so the loop continues to the
    second (reachable), yielding a two-entry per_hs_results list and
    a group verdict of not-disjoint."""
    from n2v.probabilistic.flow.scenario_verify import (
        certify_group_disjoint, HalfSpaceDisjointResult,
    )
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    group = [
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
        HalfSpace(np.array([[0.0, 1.0]]), np.array([[100.0]])),
    ]
    result = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=500, beta_2=0.001, seed=0,
    )
    # Order matters: first HS is disjoint, so the loop proceeds to HS 2.
    assert len(result.per_hs_results) == 2
    assert all(isinstance(r, HalfSpaceDisjointResult)
               for r in result.per_hs_results)
    assert result.per_hs_results[0].disjoint is True
    assert result.per_hs_results[1].disjoint is False
    # Group is not disjoint (HS 2 is reachable).
    assert result.disjoint is False


def test_group_disjoint_early_exit_on_first_failure():
    """If the FIRST HalfSpace is not disjoint, the loop short-circuits
    and per_hs_results has just one entry."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    from tests.unit.probabilistic.flow._helpers import _train_small_2d_flow
    flow = _train_small_2d_flow(seed=0)
    # First HalfSpace is trivially reachable (y_0 <= 100).
    # Second would be disjoint (y_0 <= -100) but shouldn't run due to early exit.
    group = [
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]])),
        HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
    ]
    result = certify_group_disjoint(
        flow_ode=flow, threshold_q=3.0, group=group,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result.disjoint is False
    assert len(result.per_hs_results) == 1
    assert result.per_hs_results[0].disjoint is False
    # epsilon is Bonferroni over EXECUTED HalfSpaces only — the unexecuted
    # second HS did not consume scenario budget (no samples drawn).
    import math
    assert result.epsilon_2 == pytest.approx(1 * math.log(1000) / 500, rel=1e-6)


# --------------------------------------------------------------------------- #
# The refusal and the keywords that                                             #
# a group hands to each member check, on an identity stand-in flow (no         #
# training). The OR semantics and the summed epsilon on known shares of the    #
# set are in tests/soundness/test_soundness_scenario_verify.py.                #
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


def test_empty_group_raises():
    """A group with no member is refused (ValueError): an empty OR would be
    certified disjoint vacuously."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    with pytest.raises(ValueError, match="group"):
        certify_group_disjoint(_RecordingIdentityFlow(), 1.0, [], n_samples=10,
                               beta_2=0.01, seed=0)


def test_group_hands_its_seed_to_every_member():
    """With a seed, every member check draws the same latent sample whatever
    the global numpy state: the worst sample of each member equals the worst
    sample of the member checked alone with that seed, and two group calls
    agree bit for bit. Catches a seed that the group does not forward."""
    from n2v.probabilistic.flow.scenario_verify import (
        certify_group_disjoint, certify_halfspace_disjoint,
    )
    group = [_at_least(0, 1.25), _at_least(1, 1.5)]
    flow = _RecordingIdentityFlow()
    np.random.seed(11)
    a = certify_group_disjoint(flow, 1.0, group, n_samples=64, beta_2=0.01, seed=3)
    np.random.seed(22)
    b = certify_group_disjoint(flow, 1.0, group, n_samples=64, beta_2=0.01, seed=3)
    assert a.disjoint and b.disjoint
    for hs, ra, rb in zip(group, a.per_hs_results, b.per_hs_results):
        alone = certify_halfspace_disjoint(flow, 1.0, hs, n_samples=64, beta_2=0.01, seed=3)
        np.testing.assert_array_equal(ra.worst_sample, rb.worst_sample)
        np.testing.assert_array_equal(ra.worst_sample, alone.worst_sample)


def test_group_hands_the_adaptive_keywords_to_every_member():
    """A member whose certificate is marginal (margin at least 0.25 against
    the threshold 0.5) is re-run with 900 samples inside the group: its result
    states 900 samples and the group epsilon is the sum of the member
    epsilons, ln(1/beta)/900 + ln(1/beta)/100 for the member with margin at
    least 0.75. Catches adaptive keywords that the group does not forward."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    group = [_at_least(0, 1.25), _at_least(1, 1.75)]
    r = certify_group_disjoint(_RecordingIdentityFlow(), 1.0, group, n_samples=100,
                               beta_2=0.001, seed=0, adaptive_threshold=0.5,
                               adaptive_n_samples=900)
    assert r.disjoint is True
    assert [h.n_samples_used for h in r.per_hs_results] == [900, 100]
    assert abs(r.epsilon_2 - (math.log(1000.0) / 900 + math.log(1000.0) / 100)) <= 1e-15
    assert r.n_samples_used == 100


class _RecordingLatentFlow:
    """A stand-in flow: the inverse map is the identity (float64) and every call
    records the latent points it was handed."""

    def __init__(self):
        self.latents = []

    def inverse(self, z, **kwargs):
        import torch
        self.latents.append(z.detach().numpy().copy())
        return z.detach().to(torch.float64).clone()


def _latents_of_a_direct_check(strategy):
    """The latent points certify_halfspace_disjoint draws with this strategy and seed 0."""
    from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
    flow = _RecordingLatentFlow()
    certify_halfspace_disjoint(flow, 0.5, _at_least(0, 50.0), n_samples=256, beta_2=0.01,
                               seed=0, sampling_strategy=strategy)
    return flow.latents[0]


@pytest.mark.parametrize("strategy", ["uniform", "qmc"])
def test_group_hands_its_sampling_strategy_to_every_member(strategy):
    """Every member check draws the latent points that a direct check with the group's
    strategy and seed draws, and the group's default is 'uniform' (documented). Both
    members lie far outside every sample, so both run whatever the strategy, and the two
    strategies draw different points (checked). Catches a strategy that the group does not
    forward and a default other than 'uniform'. (The test does not demand that 'qmc' hit
    a region outside the ball: the promise does not ask it, and a sampler fixed to the
    flow set's law would not do it.)"""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    direct = {s: _latents_of_a_direct_check(s) for s in ("uniform", "qmc")}
    if np.array_equal(direct["uniform"], direct["qmc"]):
        pytest.fail("the two strategies must draw different latent points")
    group = [_at_least(1, 50.0), _at_least(0, 50.0)]
    for kwargs, expected in [(dict(sampling_strategy=strategy), direct[strategy]),
                             ({}, direct["uniform"])]:
        flow = _RecordingLatentFlow()
        r = certify_group_disjoint(flow, 0.5, group, n_samples=256, beta_2=0.01, seed=0, **kwargs)
        assert r.disjoint is True and len(flow.latents) == 2
        for z in flow.latents:
            np.testing.assert_array_equal(z, expected)


def test_group_hands_its_ode_keywords_to_every_member():
    """Every member check integrates the flow with the group's keywords."""
    from n2v.probabilistic.flow.scenario_verify import certify_group_disjoint
    flow = _RecordingIdentityFlow()
    certify_group_disjoint(flow, 1.0, [_at_least(0, 5.0), _at_least(1, 5.0)], n_samples=9,
                           beta_2=0.01, seed=0, t=0.5, n_ode_steps=4, ode_method='euler',
                           ode_atol=1e-2, ode_rtol=1e-3)
    assert flow.calls == [dict(t=0.5, n_steps=4, method='euler', atol=1e-2, rtol=1e-3,
                               batch=9)] * 2
