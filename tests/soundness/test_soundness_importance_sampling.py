"""Soundness of the two sampling estimators of the unsafe mass of a flow set:
the tilted importance sampler (``n2v/probabilistic/flow/importance_sampling.py``)
and plain Monte Carlo on the latent ball (``n2v/probabilistic/flow/raw_mc_uniform.py``).

The unsafe region of a half-space is ``U = {y : G y <= g}``; a group is the
union of its half-spaces and a spec is the AND of its groups. The promises
checked here, each against a reference that does not come from n2v:

* a sample counts as unsafe exactly when it meets every row of some member of
  the group, and a reported witness is one of the pushed samples and lies in U
  (membership decided on the recorded samples in plain numpy);
* a stated upper bound is a valid (1 - beta) upper confidence bound on the
  mass: the Clopper-Pearson bound of plain Monte Carlo equals the exact
  binomial bound (the p at which at most k successes of n have probability
  beta, found in 50-digit arithmetic) and with no hit is never below
  1 - beta^(1/n), the least value a valid bound can take there; the tilted
  sampler's bound holds at its confidence over independent runs, at a mass
  of 1/2 and at a mass of 0.01 that most runs miss, and is 1 for a region
  that holds all but 1e-500 of the mass;
* plain Monte Carlo certifies a group only when no sample is unsafe and that
  bound at the caller's beta is at most the target, and a spec when any group
  certifies; a detector never answers UNSAT for a region it detected;
* the samples are the flow's images, at the requested flow time, of latents
  from the stated distribution: the latent ball of radius ``q`` for plain
  Monte Carlo (ball-conditional mass from a numerical integral), the standard
  normal prior for the tilted sampler at tilt 0;
* plain Monte Carlo refuses an unsafe region with no half-space with
  ``ValueError``, the tilted check never certifies one with no rows (the
  whole output space), and a flow held in float64 or with no parameters is
  answered like any other.

The tilted estimate itself at a positive tilt is wrong on today's library
(registry PROB-10 and PROB-11) and its UNSAT has no bound behind it (PROB-9);
those are marked in ``tests/contracts/test_probabilistic_contract.py`` and no
case here asserts either answer: every case below holds on today's library
and on a tilted sampler that draws from its proposal and weights by the
likelihood ratio. So a case that runs the tilted sampler at a positive tilt
counts what was pushed, uses an empty region where it needs one that no
sample can reach, and checks a bound only through its validity (a floor such
as 1 - beta^(1/n) binds only a bound that depends on the count alone).

Statistical cases fix their seeds and state the probability that a correct
implementation fails them (at most 1e-6), from a binomial tail. The stand-in
flows have no randomness and run on the CPU.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import mpmath
import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.stats import binom, norm

from n2v.probabilistic.flow.importance_sampling import (
    is_tilted_certify_spec, is_tilted_estimate_halfspace_mass)
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.raw_mc_uniform import (
    _clopper_pearson_upper, raw_mc_certify_spec, raw_mc_estimate_union_mass)
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.sets.halfspace import HalfSpace
from n2v.utils.verify_specification import (ProbVerifyConfig,
                                            verify_specification)
from tests.known_bugs import known_bug


# --------------------------------------------------------------------------- #
# Stand-in flows, regions and independent references                          #
# --------------------------------------------------------------------------- #
class _RecordingFlow:
    """A stand-in flow with no parameters. Its inverse is ``y = scale * z +
    shift``, computed by torch in the latents' dtype, and it records every
    call: the latent batch, the keyword arguments and the returned batch. A
    test decides membership on exactly the points the sampler pushed."""

    def __init__(self, scale=1.0, shift=0.0):
        self.scale, self.shift = scale, shift
        self.calls = []

    def inverse(self, z, **kwargs):
        y = z * self.scale + self.shift
        self.calls.append(SimpleNamespace(
            z=z.detach().cpu().numpy().astype(np.float64).copy(),
            kwargs=dict(kwargs),
            y=y.detach().cpu().numpy().astype(np.float64).copy()))
        return y

    def to(self, *args, **kwargs):
        return self

    def latents(self):
        return np.concatenate([call.z for call in self.calls])

    def outputs(self):
        return np.concatenate([call.y for call in self.calls])


class _ConstantVelocity(nn.Module):
    """A velocity field that is the constant ``c``: the real ``FlowODE``
    maps a latent z to z + c t at flow time t (exact for any solver). It is
    a linear layer with zero weight and bias ``c`` held in ``dtype`` (in
    float32 the product with the zero weight is exactly 0), or, with
    ``parameters=False``, a buffer: a flow with no parameters at all."""

    def __init__(self, c, dtype=torch.float32, parameters=True):
        super().__init__()
        self.lin = None
        if parameters:
            self.lin = nn.Linear(len(c), len(c), dtype=dtype)
            with torch.no_grad():
                self.lin.weight.zero_()
                self.lin.bias.copy_(torch.tensor(c, dtype=dtype))
        else:
            self.register_buffer("c", torch.tensor(c, dtype=dtype))

    def forward(self, t, y):
        if self.lin is None:
            return self.c.expand_as(y)
        return self.lin(y)


def _hs(G, g):
    return HalfSpace(np.asarray(G, dtype=np.float64),
                     np.asarray(g, dtype=np.float64).reshape(-1, 1))


def _members(group):
    return [(np.asarray(h.G, dtype=np.float64),
             np.asarray(h.g, dtype=np.float64).ravel()) for h in group]


def _margins(y, group):
    """Margin of each sample for a group (a union of half-spaces): the
    smallest over members of the largest row margin ``G_i y - g_i``. Plain
    numpy in float64 on float32 samples; exact for the dyadic G and g used
    in this file (every product and sum fits in 53 bits)."""
    return np.min([(y @ G.T - g[None, :]).max(axis=1)
                   for G, g in _members(group)], axis=0)


def _in_group(y, group):
    return _margins(np.atleast_2d(y), group) <= 0.0


def _closest_row_margin(y, group):
    """The smallest |G_i y - g_i| over every row of every member."""
    return min(float(np.abs(y @ G.T - g[None, :]).min())
               for G, g in _members(group))


def _require_clear_of_the_boundary(y, group, gap):
    """Setup check: no sample lies within ``gap`` of a row's boundary, so a
    float32 evaluation of a margin cannot change a sample's side."""
    closest = _closest_row_margin(y, group)
    if closest < gap:
        pytest.fail(f"a sample lies {closest:.2e} from a boundary; choose "
                    f"another seed for this case")


def _binomial_cdf(k, n, p):
    """P(Bin(n, p) <= k) by its defining sum (terms by the ratio of
    neighbouring binomial terms), in the caller's mpmath precision."""
    p = mpmath.mpf(p)
    term = (1 - p) ** n
    total = term
    for i in range(k):
        term = term * (n - i) / (i + 1) * p / (1 - p)
        total += term
    return total


def _exact_clopper_pearson(k, n, beta):
    """The exact one-sided (1 - beta) upper bound, in 50-digit arithmetic: 1
    when k = n, the closed form 1 - beta^(1/n) when k = 0, otherwise the p in
    (0, 1) at which P(Bin(n, p) <= k) = beta (bisection to 2^-150)."""
    with mpmath.workdps(50):
        beta = mpmath.mpf(beta)
        if k >= n:
            return mpmath.mpf(1)
        if k == 0:
            return 1 - beta ** (1 / mpmath.mpf(n))
        lo, hi = mpmath.mpf(0), mpmath.mpf(1)
        for _ in range(150):
            mid = (lo + hi) / 2
            if _binomial_cdf(k, n, mid) > beta:
                lo = mid
            else:
                hi = mid
        return hi


def _assert_is_the_exact_bound(value, k, n, beta, what):
    """``value`` is the exact Clopper-Pearson bound up to a relative 1e-9:
    a quantile at beta instead of 1 - beta, a wrong count, sample size or
    beta is far outside. (The few-ulp escape below the exact bound that
    float64 evaluation leaves is the marked case
    ``test_a_target_just_below_the_exact_bound_is_not_certified``.)"""
    exact = _exact_clopper_pearson(k, n, beta)
    assert not math.isnan(value), f"{what}: the bound is NaN"
    assert abs(mpmath.mpf(value) - exact) <= exact / 10 ** 9, (
        f"{what}: bound {value!r}, exact Clopper-Pearson bound "
        f"{mpmath.nstr(exact, 17)} (k={k}, n={n}, beta={beta})")


def _ball_mass_beyond(a, q, dim):
    """P(z_0 >= a | ||z|| <= q) for z standard normal in 1 or 2 dimensions
    (0 <= a < q): a closed form in 1-D and a one-dimensional integral over
    z_0 in 2-D, both in mpmath."""
    with mpmath.workdps(30):
        a, q = mpmath.mpf(a), mpmath.mpf(q)
        if dim == 1:
            return (mpmath.ncdf(q) - mpmath.ncdf(a)) / (2 * mpmath.ncdf(q) - 1)
        numerator = mpmath.quad(
            lambda x: mpmath.npdf(x)
            * (2 * mpmath.ncdf(mpmath.sqrt(q * q - x * x)) - 1), [a, q])
        return numerator / (1 - mpmath.exp(-q * q / 2))


def _assert_count_matches_the_mass(k, n, p, what):
    """k hits of n are a draw of Bin(n, p): k lies in the two-sided interval
    whose tails hold at most 5e-7 each, so a correct sampler fails this with
    probability at most 1e-6."""
    lo, hi = binom.ppf(5e-7, n, float(p)), binom.isf(5e-7, n, float(p))
    assert lo <= k <= hi, (f"{what}: {k} of {n} samples in U, the mass "
                           f"{float(p):.4f} gives [{lo:.0f}, {hi:.0f}]")


def _flow_set(flow, radius=2.0, dim=2):
    """A flow set whose score's flow is ``flow``: the latent ball of radius
    ``radius``, stated coverage 0.99 (m = 1000, ell = 999)."""
    return ProbabilisticSet(SimpleNamespace(flow_model=flow), threshold=radius,
                            m=1000, ell=999, epsilon=0.01, dim=dim)


# Regions in two dimensions (rows ``G y <= g``).
def _cut_region():
    """y0 >= -0.5, y1 <= 0.75 and y0 + y1 <= 0.5: three rows, each binding
    (dropping any one enlarges the region), about 26 % of a standard normal."""
    return _hs([[-1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.5, 0.75, 0.5])


def _far_region():
    """y0 >= 6 and y1 <= 0: no standard normal sample of these sizes, and no
    point of a ball of radius below 6."""
    return _hs([[-1.0, 0.0], [0.0, 1.0]], [-6.0, 0.0])


def _quadrant():
    """y0 <= 0 and y1 <= 0: a quarter of any centred ball or normal."""
    return _hs([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])


def _empty_region():
    """y0 <= -1 and y0 >= 1: no point meets both rows, so no sampler, tilted
    or not, can put a sample in it."""
    return _hs([[1.0, 0.0], [-1.0, 0.0]], [-1.0, -1.0])


def _at_least(row, value):
    """row . y >= value."""
    return _hs([[-float(c) for c in row]], [-float(value)])


# --------------------------------------------------------------------------- #
# Tilted importance sampling: what it counts and what it reports               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("region", [
    pytest.param(_cut_region, id="region-holds-part-of-the-samples"),
    pytest.param(_far_region, id="region-no-sample-reaches"),
])
def test_tilted_sampler_counts_a_sample_as_unsafe_iff_it_meets_every_row(
        region):
    """At the default tilt, with a three-row region in two dimensions: the
    unsafe count is the number of pushed samples that meet every row,
    ``detected_unsafe`` says whether there is one, and the witness is a
    pushed sample with the smallest margin, so a point of U when detected
    and the closest sample otherwise. Catches a margin taken over the
    smallest row instead of the largest (a sample meeting one row counted
    as unsafe; the witness outside the region), a detection that needs every
    sample in U, and a witness with the largest margin."""
    flow = _RecordingFlow()
    hs = region()
    result = is_tilted_estimate_halfspace_mass(flow, hs, n_samples=400,
                                               seed=11)
    y = flow.outputs()
    if y.shape != (400, 2):
        pytest.fail(f"the flow received {y.shape[0]} samples, not 400")
    phi = _margins(y, [hs])
    count = int((phi <= 0).sum())
    assert result.n_in_U == count
    assert result.detected_unsafe is (count > 0)
    assert any(np.array_equal(result.worst_y, row) for row in y), (
        "the witness is not one of the pushed samples")
    assert abs(result.final_phi - phi.min()) <= 1e-12
    assert abs(_margins(result.worst_y[None, :], [hs])[0]
               - result.final_phi) <= 1e-12
    if count:
        assert _in_group(result.worst_y, [hs])[0], (
            f"witness {result.worst_y} does not meet every row of U")


@pytest.mark.parametrize("mass, n", [
    pytest.param(0.5, 20_000, id="half-the-mass-many-samples"),
    pytest.param(0.01, 50, id="small-mass-most-runs-miss-it"),
])
def test_tilted_upper_bound_holds_at_its_stated_confidence(mass, n):
    """``pi_upper`` is a (1 - beta) upper confidence bound on P_flow(U). With
    the identity flow and U = {y <= Phi^-1(mass)}, P(U) is ``mass``: 1/2 with
    20000 samples, and 0.01 with 50 samples, where a run sees no sample in U
    with probability 0.99^50 = 0.61. Over 100 runs (seeds 0 to 99, beta =
    0.01, default tilt) a valid bound lies below P(U) in each run with
    probability at most beta, so the number of such runs exceeds the 1e-6
    upper quantile of Bin(100, 0.01) (8) with probability at most 1e-6.
    This holds for any valid bound, whatever distribution the sampler draws
    from (registry PROB-10 and PROB-11 do not enter). Catches a variance term
    divided by n outside the square root (at mass 1/2 the bound then lies
    below it in about 40 runs), a bound with no range term (0 in every run
    that misses U, about 60 runs at mass 0.01) and a bound that is a lower
    confidence bound."""
    hs = _hs([[1.0]], [norm.ppf(mass)])
    runs, beta = 100, 0.01
    below = sum(
        is_tilted_estimate_halfspace_mass(
            _RecordingFlow(), hs, n_samples=n, beta=beta, seed=seed
        ).pi_upper < mass
        for seed in range(runs))
    limit = binom.isf(1e-6, runs, beta)
    assert below <= limit, (f"pi_upper < P(U) = {mass} in {below} of {runs} "
                            f"runs; a valid bound at beta={beta} exceeds "
                            f"{limit:.0f} with probability <= 1e-6")


@pytest.mark.parametrize("n", [1, 200])
@pytest.mark.parametrize("region", [
    pytest.param(lambda: _hs([[1.0]], [50.0]), id="every-sample-in-U"),
])
def test_tilted_upper_bound_at_the_extremes(n, region):
    """U = {y <= 50} holds all but 1e-500 of the standard normal, more than
    any float64 below 1, so a valid (1 - beta) upper bound on P(U) is 1.0
    except with probability beta = 1e-6, for any sampler; today's bound
    depends only on the count and is exactly 1 whenever every sample is in
    U. Checked with one sample (its own branch) and with 200 (the Bernstein
    branch, capped at 1), at the default tilt. Catches a bound that is not
    capped at 1 and a single sample sent through the Bernstein branch (a
    division by n - 1 = 0, a NaN variance)."""
    result = is_tilted_estimate_halfspace_mass(
        _RecordingFlow(), region(), n_samples=n, beta=1e-6, seed=5)
    assert result.n_in_U == n
    assert result.pi_upper == 1.0


@pytest.mark.parametrize("spec", [
    pytest.param(lambda: _quadrant(), id="one-two-row-region"),
    pytest.param(lambda: [{"Hg": _quadrant()},
                          {"Hg": [_far_region(), _at_least([0.0, 1.0], 0.0)]}],
                 id="two-groups-each-detected"),
])
def test_tilted_check_never_answers_unsat_for_a_region_it_detects(spec):
    """Through ``verify_specification`` (``method='is_tilted'``, 200 samples,
    identity flow): every group of the spec holds at least a quarter of the
    standard normal output (a two-row quadrant; the OR of an unreachable
    region and a half-plane), so 200 samples miss a group with probability
    below 1e-24 for any seed and the answer is UNKNOWN, with no epsilon.
    Catches a detection that needs every sample in U (an UNSAT for a region
    of probability 1/4) and a group verdict that ignores a detected member."""
    result = verify_specification(
        _flow_set(_RecordingFlow()), spec(),
        config=ProbVerifyConfig(method="is_tilted", n_samples=200, seed=0))
    assert result.verdict == "UNKNOWN"
    assert result.epsilon_total is None


def test_tilted_verdict_does_not_depend_on_the_order_of_the_groups():
    """A spec is the AND of its groups, so its answer cannot depend on their
    order. Group A (y0 >= 0, half the samples) is detected in any run of
    200 samples (a miss has probability 2^-200 for samples of the prior, and
    a sampler tilted towards A misses it less often) and group B, an empty
    region (y0 <= -1 and y0 >= 1), is never detected by any sampler, so
    ``detected_any`` is true and the detection pattern is [[True], [False]]
    in one order and [[False], [True]] in the other; the answer for [A, B]
    is the answer for [B, A] (whatever a correct check answers there,
    registry PROB-9 being about which); and a spec of two detected groups is
    never certified. Catches a verdict loop that stops after the first
    group, a ``detected_any`` that is overwritten by each later half-space,
    a margin taken over the smallest row (B is then 'detected') and a
    detection that needs every sample in U."""
    a, b, a2 = _at_least([1.0, 0.0], 0.0), _empty_region(), _quadrant()
    forward = is_tilted_certify_spec(_RecordingFlow(), [[a], [b]],
                                     n_samples=200, seed=0)
    backward = is_tilted_certify_spec(_RecordingFlow(), [[b], [a]],
                                      n_samples=200, seed=0)
    assert [[r.detected_unsafe for r in g] for g in forward.per_hs_results] \
        == [[True], [False]]
    assert [[r.detected_unsafe for r in g] for g in backward.per_hs_results] \
        == [[False], [True]]
    assert forward.detected_any is True and backward.detected_any is True
    assert forward.unsat_certified == backward.unsat_certified
    both = is_tilted_certify_spec(_RecordingFlow(), [[a], [a2]],
                                  n_samples=200, seed=0)
    assert both.detected_any is True
    assert both.unsat_certified is False


# --------------------------------------------------------------------------- #
# Samples are the flow's images of latents from the stated distribution       #
# --------------------------------------------------------------------------- #
_FLOAT64_FLOW = ("draws its latents in float32 whatever the flow's dtype, so "
                 "a FlowODE held in float64 raises RuntimeError (mat1 and mat2 "
                 "must have the same dtype) instead of answering or refusing "
                 "the flow")


@pytest.mark.parametrize("estimator, dim, flow_kind", [
    pytest.param("is_tilted", 2, "float32", id="tilted-prior-2d"),
    pytest.param("is_tilted", 2, "no-parameters",
                 id="tilted-prior-2d-flow-without-parameters"),
    pytest.param("is_tilted", 2, "float64", id="tilted-prior-2d-flow-in-float64",
                 marks=known_bug(5, "the tilted sampler moves its latents to "
                                    "the flow's device but not to its dtype: "
                                    "it " + _FLOAT64_FLOW,
                                 raises=RuntimeError)),
    pytest.param("raw_mc", 1, "float32", id="plain-ball-1d"),
    pytest.param("raw_mc", 2, "float32", id="plain-ball-2d"),
    pytest.param("raw_mc", 2, "float64", id="plain-ball-2d-flow-in-float64",
                 marks=known_bug(5, "plain Monte Carlo " + _FLOAT64_FLOW,
                                 raises=RuntimeError)),
])
def test_samples_are_the_flow_images_at_the_requested_time(estimator, dim,
                                                           flow_kind):
    """A real ``FlowODE`` with constant velocity c = (-4, 1) maps z to
    z + c t; at t = 0.5 the region y0 >= -1 is z0 >= 1. Its mass is the
    normal tail 1 - Phi(1) under the tilted sampler's prior (tilt 0, so the
    samples are the plain prior), and the ball-conditional
    P(z0 >= 1 | ||z|| <= 1.5) under plain Monte Carlo (an integral in
    mpmath). The count of 4000 samples in U lies in the binomial interval of
    that mass (a correct sampler fails with probability at most 1e-6; the
    float32 rounding of the ODE steps moves the boundary by about 1e-6,
    which changes the mass by less than 1e-6). The flow is held in float32,
    in float64 (answered in its own precision, or refused with ValueError or
    NotImplementedError) or has no parameters at all (the tilted sampler
    then leaves the latents where they are). Catches a flow time that is
    not handed to the flow (at t = 1 the region is z0 >= 3), plain Monte
    Carlo samples that are not restricted to the ball of radius q or come
    from another radius, a tilted sampler that does not draw from its prior
    at tilt 0, and a flow without parameters that stops the sampler (the
    device lookup's StopIteration left uncaught)."""
    c = [-4.0, 1.0][:dim]
    flow = FlowODE(_ConstantVelocity(
        c, dtype=torch.float64 if flow_kind == "float64" else torch.float32,
        parameters=flow_kind != "no-parameters"))
    region = _at_least([1.0] + [0.0] * (dim - 1), -1.0)
    n = 4000
    try:
        hits, mass = _hits_and_mass(estimator, flow, region, n, dim)
    except (ValueError, NotImplementedError):
        if flow_kind == "float64":
            return                       # refusing a float64 flow is correct
        raise
    _assert_count_matches_the_mass(hits, n, mass, estimator)


def _hits_and_mass(estimator, flow, region, n, dim):
    if estimator == "is_tilted":
        result = is_tilted_estimate_halfspace_mass(
            flow, region, n_samples=n, lambda_tilt=0.0, t=0.5, seed=1)
        return result.n_in_U, 1 - mpmath.ncdf(1)
    result = raw_mc_estimate_union_mass(
        flow, [region], q=1.5, n_samples=n, t=0.5, seed=1, device="cpu")
    return result.final_unsafe_count, _ball_mass_beyond(1, 1.5, dim)


@pytest.mark.parametrize("dim", [1, 2, 33])
def test_plain_monte_carlo_samples_only_the_ball_of_radius_q(dim):
    """Plain Monte Carlo states the mass of U under latents restricted to
    the ball ||z|| <= q (``q`` = 1.5; dimension 33 takes the chi sampler).
    With the identity flow every pushed latent has norm at most q (up to
    the float32 rounding of the latents), and the region y0 >= q + 2^-10,
    beyond the ball, is never hit, for any seed. Catches samples from the
    untruncated normal (the region then holds 6.7 % of them) and a ball of
    the wrong radius (q squared)."""
    q = 1.5
    flow = _RecordingFlow()
    region = _at_least([1.0] + [0.0] * (dim - 1), q + 2.0 ** -10)
    result = raw_mc_estimate_union_mass(flow, [region], q=q, n_samples=2000,
                                        seed=4, device="cpu")
    norms = np.linalg.norm(flow.latents(), axis=1)
    assert norms.max() <= q * (1 + 1e-6), (
        f"a latent of norm {norms.max():.4f} is outside the ball of radius {q}")
    assert result.final_unsafe_count == 0
    assert result.detected_unsafe is False


# --------------------------------------------------------------------------- #
# Plain Monte Carlo: the Clopper-Pearson bound                                 #
# --------------------------------------------------------------------------- #
def test_clopper_pearson_upper_is_the_exact_binomial_bound():
    """``_clopper_pearson_upper(k, n, beta)`` is the one-sided (1 - beta)
    Clopper-Pearson bound: the p at which P(Bin(n, p) <= k) = beta, from its
    defining sum in 50-digit arithmetic, on a grid with k = 0, 1, about n/2,
    n - 1 and n, n = 1, 2, 7 and 50 (and 500 at k = 0), and beta = 1e-6,
    0.05 and 0.5; with k = n it is 1, and with no sample at all it is 1.
    Catches the quantile taken at beta (a lower bound), an all-hits guard
    that lets k = n through to a NaN quantile, and swapped Beta parameters."""
    assert _clopper_pearson_upper(0, 0, 0.05) == 1.0
    for beta in (1e-6, 0.05, 0.5):
        for n in (1, 2, 7, 50):
            for k in sorted({0, 1, n // 2, n - 1, n}):
                value = _clopper_pearson_upper(k, n, beta)
                if k == n:
                    assert value == 1.0, f"k = n = {n}: bound {value!r}"
                else:
                    _assert_is_the_exact_bound(value, k, n, beta,
                                               f"k={k} n={n}")
        _assert_is_the_exact_bound(_clopper_pearson_upper(0, 500, beta), 0,
                                   500, beta, "k=0 n=500")


@known_bug(13, "the Clopper-Pearson bound of plain Monte Carlo is the float64 "
               "Beta quantile at 1 - beta, rounded to nearest with no outward "
               "step: at beta = 1e-6 the rounding of 1 - beta puts it 5.6e-14 "
               "below the exact bound (n = 500, no hit), so a target between "
               "the two is certified though the bound at the stated "
               "confidence exceeds it")
def test_a_target_just_below_the_exact_bound_is_not_certified():
    """With no hit in n = 500 samples the exact (1 - beta) bound is
    1 - beta^(1/n); a target strictly below it (the largest float64 below
    the 50-digit value, beta = 1e-6) is not supported at the stated
    confidence, so the group is not certified. The bound in use must not be
    below the exact one, one-sided with zero slack."""
    beta, n = 1e-6, 500
    exact = _exact_clopper_pearson(0, n, beta)
    target = float(exact)
    while mpmath.mpf(target) >= exact:
        target = float(np.nextafter(target, 0.0))
    result = raw_mc_certify_spec(_RecordingFlow(), [[_far_region()]], q=2.0,
                                 eps_2_target=target, n_samples=n, beta=beta,
                                 seed=0, device="cpu")
    (only,), = result.per_hs_results
    if only.final_unsafe_count != 0:
        pytest.fail("the case needs a run with no hit")
    assert mpmath.mpf(only.pi_upper) >= exact, (
        f"bound {only.pi_upper!r} below the exact {mpmath.nstr(exact, 20)}")
    assert result.unsat_certified is False


# --------------------------------------------------------------------------- #
# Plain Monte Carlo: hits, witness and the bound of a group                     #
# --------------------------------------------------------------------------- #
def _two_member_group():
    """Two members that each cut the disc of radius 2.5 and do not meet:
    (y0 >= 1 and y1 >= -0.5) or (y0 + y1 <= -1.5 and y1 >= -2)."""
    return [_hs([[-1.0, 0.0], [0.0, -1.0]], [-1.0, 0.5]),
            _hs([[1.0, 1.0], [0.0, -1.0]], [-1.5, 2.0])]


def _far_group():
    """y0 >= 3 or y1 <= -3: beyond the disc of radius 2.5."""
    return [_at_least([1.0, 0.0], 3.0), _hs([[0.0, 1.0]], [-3.0])]


@pytest.mark.parametrize("group", [
    pytest.param(_two_member_group, id="union-of-two-cut-regions"),
    pytest.param(_far_group, id="union-beyond-the-ball"),
])
def test_plain_monte_carlo_counts_the_union_and_bounds_it(group):
    """One group of two members, 400 samples of the disc of radius 2.5,
    identity flow, beta = 0.01. The hit count is the number of pushed
    samples that meet every row of at least one member (decided in plain
    numpy; no sample lies within 1e-5 of a boundary, so the float32
    evaluation cannot move one), ``pi_hat`` is that count over 400 and
    ``pi_upper`` is the exact Clopper-Pearson bound at it. The witness is a
    pushed sample with the smallest union margin: a point of U when there is
    a hit, else the closest sample; ``final_phi`` is its margin up to the
    float32 evaluation (|G y| + |g| <= 10, so within 1e-5). Catches a hit
    counted on the wrong side of the boundary, a member left out of the
    union, a witness with the largest margin, and a bound that is not the
    Clopper-Pearson bound at the count."""
    members = group()
    flow = _RecordingFlow()
    n, beta = 400, 0.01
    result = raw_mc_estimate_union_mass(flow, members, q=2.5, n_samples=n,
                                        beta=beta, seed=2, device="cpu")
    y = flow.outputs()
    if y.shape != (n, 2):
        pytest.fail(f"the flow received {y.shape[0]} samples, not {n}")
    _require_clear_of_the_boundary(y, members, 1e-5)
    phi = _margins(y, members)
    count = int((phi <= 0).sum())
    assert result.final_unsafe_count == count
    assert result.pi_hat == count / n
    assert result.detected_unsafe is (count > 0)
    _assert_is_the_exact_bound(result.pi_upper, count, n, beta, "pi_upper")
    assert any(np.array_equal(result.worst_y, row) for row in y), (
        "the witness is not one of the pushed samples")
    assert abs(result.final_phi - phi.min()) <= 1e-5
    if count:
        assert _in_group(result.worst_y, members)[0], (
            f"witness {result.worst_y} is in no member of the group")
        assert result.final_phi <= 0.0
    else:
        assert result.final_phi > 0.0


def test_a_region_holding_the_whole_ball_has_estimate_one_and_bound_one():
    """y0 <= 1.5 holds every point of the disc of radius 1.5 (a float32
    latent cannot round past 1.5), so every one of 300 samples is a hit:
    ``pi_hat`` and ``pi_upper`` are 1 (the only valid bound when k = n),
    and the group is not certified at any target, 1 included. Catches hits
    counted on the wrong side (none would be counted and the group
    certified) and an all-hits case that reaches the Beta quantile (NaN)."""
    region = _at_least([-1.0, 0.0], -1.5)
    result = raw_mc_estimate_union_mass(_RecordingFlow(), [region], q=1.5,
                                        n_samples=300, seed=3, device="cpu")
    assert result.final_unsafe_count == 300
    assert result.pi_hat == 1.0
    assert result.pi_upper == 1.0
    spec = raw_mc_certify_spec(_RecordingFlow(), [[region]], q=1.5,
                               eps_2_target=1.0, n_samples=300, seed=3,
                               device="cpu")
    assert spec.unsat_certified is False
    assert spec.detected_any is True


# --------------------------------------------------------------------------- #
# Plain Monte Carlo: the certificate                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n, beta, region, target, seeds, certified", [
    # 50 samples with no hit support only 1 - 0.001^(1/50) = 0.129 > 0.01.
    pytest.param(50, 1e-3, lambda: _at_least([1.0], 1.5 + 2.0 ** -10), 0.01,
                 range(5), False, id="too-few-samples-region-beyond-the-ball"),
    # y >= 1.4 holds 1.6 % of the ball of radius 1.5, above the target;
    # 50 samples miss it in 44 % of the runs.
    pytest.param(50, 1e-3, lambda: _at_least([1.0], 1.4), 0.01, range(10),
                 False, id="too-few-samples-region-above-the-target"),
    # 500 samples with no hit support 1 - 1e-6^(1/500) = 0.02725 (0.0137 at
    # the default beta 0.001).
    pytest.param(500, 1e-6, lambda: _at_least([1.0], 1.5 + 2.0 ** -10), 0.02,
                 range(2), False, id="bound-at-the-requested-beta-above-target"),
    pytest.param(500, 1e-6, lambda: _at_least([1.0], 1.5 + 2.0 ** -10), 0.03,
                 range(2), True, id="bound-at-the-requested-beta-below-target"),
])
def test_plain_monte_carlo_certifies_only_when_the_bound_meets_the_target(
        n, beta, region, target, seeds, certified):
    """A group is certified exactly when no sample is in U and the
    Clopper-Pearson bound at the caller's beta is at most the target. With
    no hit that bound is 1 - beta^(1/n) whatever the samples, so the answer
    is exact for every seed: never certified when the bound exceeds the
    target (even when no sample was unsafe, and in the second case for a
    region whose mass, 0.016, is above the target), certified when it does
    not; the reported bound is the exact one. Catches a gate on the point
    estimate (0 when nothing is hit), a beta that is not handed on (the
    default's smaller bound certifies the third case), and a gate that never
    passes."""
    for seed in seeds:
        result = raw_mc_certify_spec(_RecordingFlow(), [[region()]], q=1.5,
                                     eps_2_target=target, n_samples=n,
                                     beta=beta, seed=seed, device="cpu")
        (only,), = result.per_hs_results
        assert result.unsat_certified is (certified and
                                          only.final_unsafe_count == 0), (
            f"seed {seed}: {only.final_unsafe_count} hits, bound "
            f"{only.pi_upper}, target {target}")
        if only.final_unsafe_count == 0:
            _assert_is_the_exact_bound(only.pi_upper, 0, n, beta,
                                       f"seed {seed}")
    if certified:
        assert result.unsat_certified is True


@pytest.mark.parametrize("order", ["detected-group-first",
                                   "detected-group-last"])
def test_a_spec_is_certified_by_any_group_and_states_the_largest_bound(order):
    """Two groups on the disc of radius 2 (500 samples, beta = 1e-6, target
    0.03): A = {y0 >= 1} holds about 20 % of it and is hit, B = {y0 >= 5}
    lies beyond it and is never hit, with bound 1 - beta^(1/500) = 0.02725.
    The spec (the AND of A and B) is certified because B is, in either
    order; ``detected_any`` is true; ``eps_2_upper`` is the largest group
    bound, which is A's exact Clopper-Pearson bound at its hit count. A spec
    of two hit groups is not certified. Catches a verdict loop that stops
    after the first group, a ``detected_any`` overwritten by the last group,
    and a largest bound that is never updated (reported as 0)."""
    a, b = [_at_least([1.0, 0.0], 1.0)], [_at_least([1.0, 0.0], 5.0)]
    groups = [a, b] if order == "detected-group-first" else [b, a]
    n, beta = 500, 1e-6
    flow = _RecordingFlow()
    result = raw_mc_certify_spec(flow, groups, q=2.0, eps_2_target=0.03,
                                 n_samples=n, beta=beta, seed=6, device="cpu")
    samples_of_a = flow.calls[0 if order == "detected-group-first" else 1].y
    hits_a = int(_in_group(samples_of_a, a).sum())
    if not 0 < hits_a < n:
        pytest.fail(f"the case needs some hits in A, got {hits_a}")
    assert result.unsat_certified is True
    assert result.detected_any is True
    _assert_is_the_exact_bound(result.eps_2_upper, hits_a, n, beta,
                               "eps_2_upper")
    assert result.eps_2_upper == max(g[0].pi_upper
                                     for g in result.per_hs_results)
    both = raw_mc_certify_spec(_RecordingFlow(), [a, [_quadrant()]], q=2.0,
                               eps_2_target=0.03, n_samples=n, beta=beta,
                               seed=6, device="cpu")
    assert both.unsat_certified is False


@pytest.mark.parametrize("seeds", [
    pytest.param("one-seed-for-every-run", id="seeded-groups-share-a-draw"),
    pytest.param(None, id="no-seed", marks=known_bug(
        13, "raw_mc_certify_spec with seed=None draws fresh samples for each "
            "group, so a spec of several groups is certified falsely with "
            "probability up to the number of groups times beta: 0.34 for two "
            "groups at beta = 0.2")),
])
def test_plain_monte_carlo_keeps_its_confidence_over_several_groups(seeds):
    """The spec [[U], [U]] (two groups, each U = {y <= c}) has the unsafe
    region U, of mass 0.0505 (c the 0.0505 normal quantile; the ball of
    radius 8 holds all but 1e-15 of the normal). With 32 samples and beta =
    0.2 a run with no hit gives the bound 1 - 0.2^(1/32) = 0.0491 <= 0.05
    = target, so a certificate is false here, and a valid check gives one in
    a run with probability at most beta. Over 1000 independent runs the
    number of certificates exceeds the 1e-6 upper quantile of Bin(1000, 0.2)
    (262) with probability at most 1e-6. With a seed per run the two groups
    share one draw and the rate is the one-group rate (about 0.19); with no
    seed each group draws its own samples and the spec is certified when
    either draw misses U."""
    region = _hs([[1.0]], [norm.ppf(0.0505)])
    runs, beta = 1000, 0.2
    certified = sum(
        raw_mc_certify_spec(
            _RecordingFlow(), [[region], [region]], q=8.0, eps_2_target=0.05,
            n_samples=32, beta=beta, device="cpu",
            seed=(run if seeds is not None else None)).unsat_certified
        for run in range(runs))
    limit = binom.isf(1e-6, runs, beta)
    assert certified <= limit, (f"{certified} false certificates in {runs} "
                                f"runs; a valid check exceeds {limit:.0f} with "
                                f"probability <= 1e-6")


def test_plain_monte_carlo_options_reach_the_flow():
    """Every option of ``raw_mc_certify_spec`` reaches the flow of every
    group: the flow time, the step count, the solver and both tolerances,
    and 37 samples per group drawn from the ball of radius 2. Catches an
    option that is not handed on (the flow then integrates to the default
    time 1, or the default 2000 samples are drawn)."""
    flow = _RecordingFlow()
    raw_mc_certify_spec(flow, [[_quadrant()], [_far_region(), _quadrant()]],
                        q=2.0, eps_2_target=0.5, n_samples=37, beta=0.2,
                        seed=3, t=0.25, n_ode_steps=7, ode_method="euler",
                        ode_atol=1e-3, ode_rtol=1e-4, device="cpu")
    assert len(flow.calls) == 2
    for call in flow.calls:
        assert call.kwargs == {"t": 0.25, "n_steps": 7, "method": "euler",
                               "atol": 1e-3, "rtol": 1e-4}
        assert call.z.shape == (37, 2)
        assert np.linalg.norm(call.z, axis=1).max() <= 2.0 * (1 + 1e-6)


def test_plain_monte_carlo_is_reproducible_for_a_seed():
    """The same seed gives the same draw and the same result whatever the
    global generators held before the call, seed 0 included; another seed
    gives another draw; and with ``seed=None`` two calls in a row draw
    different latents (the global generators are left to advance). Catches
    a seed that is applied only when it is truthy (seed 0, the default,
    would then draw from whatever state the global generator was left in),
    a seed that is ignored, and ``seed=None`` read as a fixed seed."""
    def run(seed, disturb):
        np.random.seed(disturb)
        torch.manual_seed(disturb)
        flow = _RecordingFlow()
        result = raw_mc_estimate_union_mass(flow, [_quadrant()], q=2.0,
                                            n_samples=50, seed=seed,
                                            device="cpu")
        return flow.latents(), result

    for seed in (0, 9):
        (z1, r1), (z2, r2) = run(seed, 101), run(seed, 202)
        assert np.array_equal(z1, z2), f"seed {seed} gave two different draws"
        assert (r1.final_unsafe_count, r1.final_phi) == (
            r2.final_unsafe_count, r2.final_phi)
        assert np.array_equal(r1.worst_y, r2.worst_y)
    assert not np.array_equal(run(0, 101)[0], run(9, 101)[0])
    flow = _RecordingFlow()
    for _ in range(2):
        raw_mc_estimate_union_mass(flow, [_quadrant()], q=2.0, n_samples=50,
                                   seed=None, device="cpu")
    assert not np.array_equal(flow.calls[0].z, flow.calls[1].z)


@pytest.mark.parametrize("call", [
    pytest.param(lambda: raw_mc_estimate_union_mass(
        _RecordingFlow(), [_quadrant()], q=2.0, beta=0.0, device="cpu"),
        id="beta-zero"),
    pytest.param(lambda: raw_mc_estimate_union_mass(
        _RecordingFlow(), [_quadrant()], q=2.0, beta=1.0, device="cpu"),
        id="beta-one"),
    pytest.param(lambda: raw_mc_certify_spec(
        _RecordingFlow(), [[_quadrant()]], q=2.0, eps_2_target=0.1, beta=1.5,
        device="cpu"), id="beta-above-one-through-the-spec"),
    pytest.param(lambda: raw_mc_estimate_union_mass(
        _RecordingFlow(), [], q=2.0, device="cpu"), id="group-with-no-member"),
    pytest.param(lambda: raw_mc_certify_spec(
        _RecordingFlow(), [[_quadrant()], []], q=2.0, eps_2_target=0.1,
        device="cpu"), id="spec-with-an-empty-group"),
    pytest.param(lambda: raw_mc_estimate_union_mass(
        _RecordingFlow(), [_quadrant(), _hs([[1.0, 0.0, 0.0]], [0.0])], q=2.0,
        device="cpu"), id="members-of-different-input-dimension"),
    pytest.param(lambda: raw_mc_estimate_union_mass(
        _RecordingFlow(), [_quadrant()], q=0.0, device="cpu"), id="radius-zero"),
    pytest.param(lambda: raw_mc_estimate_union_mass(
        _RecordingFlow(), [_quadrant()], q=2.0, n_samples=0, device="cpu"),
        id="no-sample", marks=known_bug(
            23, "plain Monte Carlo with n_samples=0 divides by the sample count "
                "and raises ZeroDivisionError instead of refusing the count "
                "with ValueError", raises=ZeroDivisionError)),
])
def test_plain_monte_carlo_refuses_an_input_it_cannot_read(call):
    """A beta outside (0, 1), a group or spec group with no half-space,
    members of different input dimension, a ball of radius 0 and a sample
    count of 0 are refused with ``ValueError`` before any answer. Catches a
    beta range that admits 0 or 1 (an answer with a confidence of 1 or 0)
    and a missing check that lets the call fail later on an accidental
    exception."""
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("spec, n, target, verdict", [
    pytest.param(lambda: [{"Hg": _at_least([1.0, 0.0], 1.0)},
                          {"Hg": _at_least([1.0, 0.0], 5.0)}], 500, 0.03,
                 "UNSAT", id="second-group-certifies"),
    pytest.param(lambda: _at_least([-1.0, 0.0], -2.0), 500, 0.99, "UNKNOWN",
                 id="region-holds-the-whole-disc"),
    pytest.param(lambda: _at_least([1.0, 0.0], 5.0), 50, 0.01, "UNKNOWN",
                 id="too-few-samples-for-the-target"),
])
def test_plain_monte_carlo_answers_through_verify_specification(
        spec, n, target, verdict, monkeypatch):
    """The public route, ``verify_specification`` on a flow set over the disc
    of radius 2 with ``method='raw_mc_uniform'`` and beta = 1e-6 (identity
    flow, the CPU forced): a spec whose second group lies beyond the disc
    and whose first holds part of it is UNSAT (500 samples support 0.02725
    <= 0.03); a region holding the whole disc is UNKNOWN even at target
    0.99; a region beyond the disc with 50 samples (bound 0.24 > 0.01) is
    UNKNOWN although no sample meets it. Catches a verdict loop that stops
    after the first group, hits counted on the wrong side, and a gate on the
    point estimate, at the level the user calls."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = verify_specification(
        _flow_set(_RecordingFlow()), spec(),
        config=ProbVerifyConfig(method="raw_mc_uniform", n_samples=n,
                                beta=1e-6, seed=0,
                                amls_bounded_eps_2_target=target))
    assert result.verdict == verdict


def _config(method):
    return ProbVerifyConfig(
        method=method, n_samples=50, seed=0,
        amls_bounded_eps_2_target=0.1 if method == "raw_mc_uniform" else None)


def test_plain_monte_carlo_refuses_an_unsafe_region_with_no_half_space(
        monkeypatch):
    """``verify_specification(flow_set, [])`` with ``method='raw_mc_uniform'``
    states no unsafe region and is refused with ``ValueError`` (the settled
    direction of registry SPEC-12) at the level the user calls: the parser
    makes it one empty group and the spec check refuses that group. The
    tilted check answers UNSAT on the same call, which is marked in
    ``tests/contracts/test_probabilistic_contract.py::test_an_empty_unsafe_region_is_refused_on_the_probabilistic_lane[is_tilted]``.
    Catches an empty group that is skipped (the spec then has no group and
    is answered)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _config("raw_mc_uniform")
    with pytest.raises(ValueError):
        verify_specification(_flow_set(_RecordingFlow()), [], config=config)


def test_tilted_check_never_certifies_a_region_with_no_rows():
    """A half-space with no rows is the whole output space: every sample is
    unsafe, so UNSAT is wrong for any flow set. Through
    ``verify_specification`` with ``method='is_tilted'`` the answer must not
    be UNSAT, or the region is refused with ``ValueError``; both are
    correct, as for a sound set in
    ``tests/contracts/test_verify_api.py::test_unsafe_region_with_no_rows_is_never_unsat``.
    The tilted sampler refuses it today (numpy's maximum over no rows raises
    ``ValueError``); the torch margin of plain Monte Carlo crashes on it,
    which is marked in
    ``tests/contracts/test_probabilistic_contract.py::test_an_unsafe_region_with_no_rows_is_refused_or_not_unsat[raw_mc_uniform]``.
    Catches a margin over no rows taken as 'outside U' (no sample is then
    unsafe and the spec is certified)."""
    whole_space = HalfSpace(np.zeros((0, 2)), np.zeros((0, 1)))
    config = _config("is_tilted")
    try:
        result = verify_specification(_flow_set(_RecordingFlow()), whole_space,
                                      config=config)
    except ValueError:
        return
    assert result.verdict != "UNSAT"
