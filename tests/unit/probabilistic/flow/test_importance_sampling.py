"""Tests for C2 / Tilted Importance Sampling flow-set detector.

These tests exercise the self-normalised IS estimator on small synthetic
flows, plus the dispatch into ``verify_specification`` via
``verification_method='is_tilted'``.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from n2v.probabilistic.flow.importance_sampling import (
    is_tilted_certify_spec,
    is_tilted_estimate_halfspace_mass,
)
from n2v.sets.halfspace import HalfSpace


# ---------- Identity flow stub ----------


class _IdentityFlow:
    """A trivial 'flow' whose inverse is the identity.

    Lets us drive IS with a known target distribution -- the latent
    ``z ~ N(0, I_d)`` is also the data distribution.
    """

    def __init__(self):
        self.velocity_field = None

    def inverse(self, z, **_kw):
        return z

    def forward(self, y, **_kw):
        return y

    def eval(self):
        return self

    def to(self, *_args, **_kw):
        return self


# ---------- Smoke + return-shape ----------


def test_is_returns_correct_shape_and_keys():
    """Smoke: function runs and returns a populated ISResult."""
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])  # y_0 >= -10 (bulk)
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=200, lambda_tilt=5.0, seed=0,
    )
    assert res.n_samples == 200
    assert 0.0 <= res.pi_hat <= 1.0
    assert 0.0 <= res.pi_upper <= 1.0
    assert res.ess > 0
    assert res.worst_y.shape == (2,)
    assert isinstance(res.detected_unsafe, bool)
    # On a bulk halfspace IS should detect U.
    assert res.detected_unsafe is True


# ---------- Unbiasedness on a known case ----------


@pytest.mark.slow
def test_is_zero_lambda_recovers_flat_mc():
    """At lambda=0 all weights are 1 and pi_hat reduces to the flat
    Monte-Carlo indicator average. That is exactly P_flow(U) under p_z.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0]])
    g = np.array([-2.0])  # U = { y_0 <= -2 }, true mass ~0.02275
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = is_tilted_estimate_halfspace_mass(
        flow, hs,
        n_samples=10_000, lambda_tilt=0.0, seed=0,
    )
    true_mass = 0.02275  # 1 - Phi(2)
    assert res.detected_unsafe is True
    # Flat MC at N=10K is unbiased; loose 2x tolerance for noise.
    assert res.pi_hat <= 2.0 * true_mass
    assert res.pi_hat >= true_mass / 2.0


# A test here asserted that the tilted estimate is at least the flat one on
# the same samples, which is the over-estimate of DEFERRED-BUGS.md bug 4. The
# estimate is of P_flow(U), so it must be unbiased at every tilt; the contract
# is tests/contracts/test_probabilistic_contract.py
# (``test_is_tilted_estimate_is_unbiased``).


def test_is_no_detect_when_truly_disjoint():
    """U = { y_0 <= -100 } has effectively zero mass under N(0,1).
    IS should NOT detect U with n_samples=200 (no sample reaches the deep
    tail; that's the limitation IS shares with flat MC -- it does not
    sample biased toward U, only reweights).
    """
    flow = _IdentityFlow()
    G = np.array([[1.0, 0.0]])
    g = np.array([-100.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=200, lambda_tilt=5.0, seed=0,
    )
    assert res.detected_unsafe is False
    # final_phi is the MIN slack observed; outside U => slack > 0.
    assert res.final_phi > 0.0
    # pi_hat should be 0 (no detection means no in-U samples; the SN-IS
    # numerator is identically zero).
    assert res.pi_hat == 0.0


def test_is_zero_lambda_recovers_uniform_mc():
    """With lambda=0 weights are all 1 and the estimator collapses to
    flat MC: pi_hat = (in_U).mean()."""
    flow = _IdentityFlow()
    G = np.array([[-1.0]])  # y_0 >= -10 (bulk)
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=2000, lambda_tilt=0.0, seed=0,
    )
    # All samples are in U => pi_hat ~ 1.
    assert res.pi_hat > 0.99
    # ESS should be ~ N when all weights equal.
    assert abs(res.ess - res.n_samples) < 1e-6


def test_is_ess_decreases_with_lambda():
    """Larger lambda => more concentrated weights => smaller ESS."""
    flow = _IdentityFlow()
    G = np.array([[1.0]])
    g = np.array([-1.0])  # U = {y_0 <= -1}, moderate mass ~0.16
    hs = HalfSpace(G, g.reshape(-1, 1))

    res_small = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=2000, lambda_tilt=0.5, seed=0,
    )
    res_large = is_tilted_estimate_halfspace_mass(
        flow, hs, n_samples=2000, lambda_tilt=10.0, seed=0,
    )
    assert res_large.ess < res_small.ess


# ---------- Spec-level dispatcher ----------


# The two tests below asserted ``unsat_certified is True`` where no sample was
# unsafe. Sampling that found nothing is not a certificate; the answer it
# leads to is the subject of tests/contracts/test_probabilistic_contract.py
# (``test_is_tilted_never_answers_unsat_without_a_bound``, DEFERRED-BUGS.md
# bug 4). They now assert what the sampler did observe.


def test_is_certify_spec_reports_no_detection_in_an_unreachable_region():
    """One group with one HalfSpace no sample reaches: nothing is detected
    and no sample is counted as unsafe.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0, 0.0]])
    g = np.array([-100.0])
    hs = HalfSpace(G, g.reshape(-1, 1))
    res = is_tilted_certify_spec(
        flow, [[hs]],
        n_samples=200, lambda_tilt=5.0, seed=0,
    )
    assert res.detected_any is False
    (only,), = res.per_hs_results
    assert only.detected_unsafe is False
    assert only.n_in_U == 0
    assert only.n_samples == 200
    assert only.final_phi > 0.0


def test_is_certify_spec_unknown_when_group_member_detected():
    """Single group with one reachable HalfSpace: detected => no group
    is fully disjoint => unsat_certified False.
    """
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])  # y_0 >= -10 (bulk)
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))
    res = is_tilted_certify_spec(
        flow, [[hs]],
        n_samples=200, lambda_tilt=5.0, seed=0,
    )
    assert res.detected_any is True
    assert res.unsat_certified is False


def test_is_certify_spec_reports_detection_per_group():
    """Spec with TWO groups (AND across groups): the results keep the group
    structure, one list of per-HalfSpace results per group, in order.
    """
    flow = _IdentityFlow()
    hs_far = HalfSpace(
        np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    hs_bulk = HalfSpace(
        np.array([[-1.0, 0.0]]), np.array([[10.0]]))
    res = is_tilted_certify_spec(
        flow, [[hs_far], [hs_bulk]],
        n_samples=200, lambda_tilt=5.0, seed=0,
    )
    detected = [[r.detected_unsafe for r in group]
                for group in res.per_hs_results]
    assert detected == [[False], [True]]
    assert res.detected_any is True


# ---------- Options, seeds, chunks and refusals ----------


class _RecordingIdentityFlow(_IdentityFlow):
    """The identity flow, recording the keyword arguments and the latent
    batch of every call, so that a test sees what reached the flow and
    decides membership on exactly the samples that were pushed."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def inverse(self, z, **kw):
        self.calls.append((dict(kw), z.detach().numpy().astype(np.float64)))
        return z.clone()

    def samples(self):
        return np.concatenate([batch for _, batch in self.calls])


def _count_in(y, hs):
    """Samples meeting every row of ``G y <= g``, in plain numpy."""
    G = np.asarray(hs.G, dtype=np.float64)
    g = np.asarray(hs.g, dtype=np.float64).ravel()
    return int(np.all(y @ G.T <= g[None, :], axis=1).sum())


def test_is_options_reach_the_flow_and_every_halfspace_run():
    """Every option of ``is_tilted_certify_spec`` reaches each per-half-space
    run and the flow: the flow time, step count, solver and both tolerances
    arrive at ``flow.inverse`` for each of the three half-spaces; each run
    draws ``n_samples`` = 37 and reports the tilt it used. The tilt is 0, so
    each estimate is the plain fraction n_in_U / 37 and the effective sample
    size is 37 (true of any importance sampler at tilt 0). Catches a flow
    time dropped by the flow helper (the flow integrates to t = 1), a tilt
    that is not handed on (every run uses the default 5) and a solver option
    that is dropped or swapped."""
    flow = _RecordingIdentityFlow()
    a = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.0]]))
    b = HalfSpace(np.array([[0.0, 1.0], [1.0, 0.0]]),
                  np.array([[0.5], [0.25]]))
    res = is_tilted_certify_spec(
        flow, [[a], [b, a]], n_samples=37, lambda_tilt=0.0, seed=3,
        t=0.25, n_ode_steps=7, ode_method='euler', ode_atol=1e-3,
        ode_rtol=1e-4)
    assert len(flow.calls) == 3
    for kw, batch in flow.calls:
        assert kw == {'t': 0.25, 'n_steps': 7, 'method': 'euler',
                      'atol': 1e-3, 'rtol': 1e-4}
        assert batch.shape == (37, 2)
    runs = [r for group in res.per_hs_results for r in group]
    assert len(runs) == 3
    for r in runs:
        assert r.lambda_tilt == 0.0
        assert r.n_samples == 37
        assert r.pi_hat == r.n_in_U / 37
        assert abs(r.ess - 37.0) < 1e-9


def test_is_beta_reaches_every_halfspace_run():
    """``beta`` is the confidence parameter of each run's upper bound: on the
    same draws (same seed), asking for more confidence (beta 1e-6 instead of
    0.2) widens the bound of every run whose region holds some but not all
    samples. Catches a beta that ``is_tilted_certify_spec`` does not hand on
    (both calls would then use the default 0.001 and agree), and weights
    above 1 inside U (the bound then passes 1 before its cap and is 1 at
    both betas)."""
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.0]]))
    hs2 = HalfSpace(np.array([[0.0, -1.0]]), np.array([[0.5]]))
    wide = is_tilted_certify_spec(_IdentityFlow(), [[hs], [hs2]],
                                  n_samples=300, beta=1e-6, seed=4)
    narrow = is_tilted_certify_spec(_IdentityFlow(), [[hs], [hs2]],
                                    n_samples=300, beta=0.2, seed=4)
    for (w,), (n,) in zip(wide.per_hs_results, narrow.per_hs_results):
        if not 0 < n.n_in_U < 300 or w.n_in_U != n.n_in_U:
            pytest.fail('the case needs the same partial count in both runs')
        assert w.pi_upper > n.pi_upper


def test_is_same_seed_gives_the_same_run_and_another_seed_another_draw():
    """The same seed gives the same latent draw and the same result, for one
    half-space and for every run of a spec; another seed gives another
    draw, and ``seed=None`` a fresh draw on each call. Catches a generator
    created without the seed (every call draws fresh latents), a seed that
    is ignored, and ``seed=None`` read as a fixed seed (repeated runs then
    see the same samples)."""
    hs = HalfSpace(np.array([[1.0, 1.0]]), np.array([[-0.5]]))

    def run(seed):
        flow = _RecordingIdentityFlow()
        res = is_tilted_estimate_halfspace_mass(flow, hs, n_samples=100,
                                                seed=seed)
        return flow.samples(), res

    (z1, r1), (z2, r2), (z3, _) = run(7), run(7), run(8)
    assert np.array_equal(z1, z2)
    assert (r1.n_in_U, r1.final_phi, r1.pi_hat, r1.pi_upper, r1.ess) == (
        r2.n_in_U, r2.final_phi, r2.pi_hat, r2.pi_upper, r2.ess)
    assert np.array_equal(r1.worst_y, r2.worst_y)
    assert not np.array_equal(z1, z3)
    assert not np.array_equal(run(None)[0], run(None)[0])
    s1 = is_tilted_certify_spec(_IdentityFlow(), [[hs], [hs]],
                                n_samples=50, seed=7)
    s2 = is_tilted_certify_spec(_IdentityFlow(), [[hs], [hs]],
                                n_samples=50, seed=7)
    for (a,), (b,) in zip(s1.per_hs_results, s2.per_hs_results):
        assert (a.n_in_U, a.final_phi) == (b.n_in_U, b.final_phi)
        assert np.array_equal(a.worst_y, b.worst_y)


def test_is_pushes_every_sample_once_across_chunks():
    """With 23 samples in chunks of 5 the flow receives four full chunks and
    a last chunk of 3, each latent once; the count, the witness and the
    estimate are those of the 23 pushed samples. Catches a chunk loop that
    skips or truncates the last chunk (its rows are then never pushed and
    the result reads uninitialised memory)."""
    flow = _RecordingIdentityFlow()
    hs = HalfSpace(np.array([[1.0]]), np.array([[0.0]]))
    res = is_tilted_estimate_halfspace_mass(flow, hs, n_samples=23,
                                            lambda_tilt=0.0, seed=2,
                                            chunk_size=5)
    assert [batch.shape[0] for _, batch in flow.calls] == [5, 5, 5, 5, 3]
    y = flow.samples()
    assert len(np.unique(y, axis=0)) == 23
    count = _count_in(y, hs)
    assert res.n_in_U == count
    assert res.pi_hat == count / 23
    assert res.final_phi == float(y.min())
    assert np.array_equal(res.worst_y, y[np.argmin(y[:, 0])])


@pytest.mark.parametrize('entry, kwargs', [
    pytest.param('estimate', {'n_samples': 0}, id='no-sample'),
    pytest.param('estimate', {'lambda_tilt': -1e-9}, id='negative-tilt'),
    pytest.param('estimate', {'beta': 0.0}, id='beta-zero'),
    pytest.param('estimate', {'beta': 1.0}, id='beta-one'),
    pytest.param('certify_spec', {'n_samples': 0},
                 id='no-sample-through-a-spec'),
])
def test_is_refuses_a_sample_count_tilt_or_beta_out_of_range(entry, kwargs):
    """A sample count below 1, a negative tilt and a beta outside (0, 1) are
    refused with ``ValueError``, at the boundary of each range, called
    directly and through a spec. Catches a range check that admits beta = 1
    (a bound at confidence 0), beta = 0 (a crash on log(2/0)) or a negative
    tilt (weights above 1)."""
    hs = HalfSpace(np.array([[1.0]]), np.array([[0.0]]))
    with pytest.raises(ValueError):
        if entry == 'estimate':
            is_tilted_estimate_halfspace_mass(_IdentityFlow(), hs, **kwargs)
        else:
            is_tilted_certify_spec(_IdentityFlow(), [[hs]], **kwargs)


def test_is_estimate_is_zero_when_every_weight_underflows():
    """With a tilt of 1e6 and U = {y <= -50}, which no sample of the prior
    comes near, the weight exp(-lambda * phi) of every sample outside U
    underflows to 0. The estimate is still a number in [0, 1], exactly 0
    when no pushed sample is in U (the numerator is then 0, as it is today:
    0 of 50), the count is that of the pushed samples, and the bound is a
    number in (0, 1] (a valid bound is positive: U has positive mass, and a
    run that misses it is the likely one). Written so that it holds for a
    sampler that draws from a tilted proposal too. Catches the zero-weight
    guard left out (0/0, a NaN estimate) and a bound with no range term (0
    when nothing is hit)."""
    flow = _RecordingIdentityFlow()
    hs = HalfSpace(np.array([[1.0]]), np.array([[-50.0]]))
    res = is_tilted_estimate_halfspace_mass(flow, hs, n_samples=50,
                                            lambda_tilt=1e6, seed=0)
    count = _count_in(flow.samples(), hs)
    assert res.n_in_U == count
    assert res.detected_unsafe is (count > 0)
    assert math.isfinite(res.pi_hat) and 0.0 <= res.pi_hat <= 1.0
    assert (res.pi_hat == 0.0) is (count == 0)
    assert math.isfinite(res.pi_upper) and 0.0 < res.pi_upper <= 1.0


def test_is_reads_an_offset_vector_of_shape_k():
    """The half-space may carry ``g`` of shape (k,) as well as (k, 1), and
    ``G`` with fewer rows than columns: two rows in three dimensions,
    y0 <= 0 and y2 <= 0.5. The count is that of the pushed samples meeting
    both rows. Catches the input dimension read from the row count and an
    offset vector read by a column index."""
    flow = _RecordingIdentityFlow()
    hs = SimpleNamespace(G=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                         g=np.array([0.0, 0.5]))
    res = is_tilted_estimate_halfspace_mass(flow, hs, n_samples=200, seed=1)
    y = flow.samples()
    assert y.shape == (200, 3)
    assert res.n_in_U == _count_in(y, hs)
    assert res.worst_y.shape == (3,)
