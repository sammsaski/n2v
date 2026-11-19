"""Soundness cells of the scenario certifiers on a flow set and of their samplers
(`n2v/probabilistic/flow/scenario_verify.py`).

The flow set is {y : ||forward(y)|| <= q}, the image under the flow's inverse map
(latent to data) of the latent ball of radius q. The scenario certifiers draw N
latent points, map them through the inverse map and check a property on them.
The promise checked here:

* a sampler returns points of the whole ball and only of the ball, with the law
  its docstring states (the truncated Gaussian: radius by the chi law cut at q,
  direction uniform; Sobol points through the normal inverse CDF);
* a polyhedron ``G y <= g`` (the UNSAFE region, closed) that holds a known share
  of the flow set is never certified disjoint, a spec is UNSAT only when one of
  its groups (AND across groups, OR inside a group) is certified, and a safe
  property violated on a known share of the set is never 'verified';
* the stated miscoverage is the documented scenario bound ln(1/beta)/N, never
  below the smallest value N samples support at confidence 1 - beta, which is
  1 - beta^(1/N), and a group or a spec states the sum over every test it ran;
* a certificate keeps its stated confidence: over independent seeds, a region
  that holds more of the flow set than the stated miscoverage is certified no
  more often than beta allows (a binomial tail), for every sampling strategy
  and through the public route;
* a counterexample is the worst sample and lies in the set; a 'falsified'
  witness lies in the input box and its real output violates the property;
* the public route ``verify_specification`` on a whitened flow set answers the
  same.

Oracles are independent of n2v: stand-in flows whose inverse map is a fixed
affine map ``A z + c`` or a constant, for which the flow set is an ellipsoid or a
point and the share of the set that a half-space cuts off is the closed form
``P(z_1 >= s | ||z|| <= q)`` of the truncated Gaussian (a one-dimensional
integral, scipy); the exact scenario bound in mpmath; the chi and Beta laws of
scipy for the samplers, compared by the Dvoretzky-Kiefer-Wolfowitz inequality,
and the chi-square law in 60-digit arithmetic where double precision underflows.
Every statistical assertion states the probability that a correct
implementation fails it, at most 1e-6. Latent samples are cast to float32 by the
library before the inverse map, so a sample on the sphere can leave the ball by
a relative 2^-24; checks against the set carry that derived slack and no other.
"""

import math
import signal
from types import SimpleNamespace

import mpmath
import numpy as np
import pytest
import torch
from scipy import integrate, stats

from n2v.probabilistic.flow.scenario_verify import (
    _qmc_sample_latents,
    certify_group_disjoint,
    certify_halfspace_disjoint,
    certify_spec_disjoint,
    preimage_search,
    sample_empirical_latent_ball,
    sample_truncated_gaussian_ball,
    scenario_verify_halfspace,
    verify_robustness,
)
from n2v.probabilistic.flow.calibrate import compute_guarantee
from n2v.probabilistic.flow.scores import FlowScore
from n2v.probabilistic.flow.sets import AffineTransform, ProbabilisticSet
from n2v.sets.halfspace import HalfSpace
from n2v.utils.verify_specification import ProbVerifyConfig, verify_specification
from tests.known_bugs import known_bug

# A float32 cast moves each latent coordinate by at most |z_i| 2^-24, so a
# sample of the ball of radius q lies in the ball of radius q (1 + 2^-23).
_F32_REL = 2.0 ** -23


# --------------------------------------------------------------------------- #
# Stand-in flows and closed forms                                              #
# --------------------------------------------------------------------------- #
class _AffineFlow:
    """A stand-in flow: inverse (latent to data) y = A z + c, forward
    z = A^{-1} (y - c), both in float64. Its flow set is the ellipsoid
    {A z + c : ||z|| <= q}; the least value of v.y over it is
    v.c - q ||A^T v||."""

    def __init__(self, A, c):
        self.A = np.asarray(A, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)

    def inverse(self, z, **kwargs):
        zz = z.detach().to(torch.float64).numpy()
        return torch.from_numpy(zz @ self.A.T + self.c)

    def forward(self, y, **kwargs):
        yy = y.detach().to(torch.float64).numpy()
        return torch.from_numpy(np.linalg.solve(self.A, (yy - self.c).T).T)

    def latent(self, y):
        """The latent point of a data point, in float64."""
        return np.linalg.solve(self.A, np.asarray(y, dtype=np.float64) - self.c)


class _ConstantFlow:
    """A stand-in flow whose inverse map sends every latent point to y0: the flow
    set of a network whose output is constant over its input set."""

    def __init__(self, y0):
        self.y0 = torch.tensor(y0, dtype=torch.float64)

    def inverse(self, z, **kwargs):
        return self.y0.expand(z.shape[0], -1).clone()


_IDENTITY_2D = _AffineFlow(np.eye(2), np.zeros(2))
# Non-symmetric, entries above and below one in magnitude, a shift of both signs.
_SKEW_2D = _AffineFlow([[2.0, 0.5], [-0.25, 0.75]], [3.0, -1.5])


def _cap_share(q, s, d):
    """P(z_1 >= s | ||z|| <= q) for z ~ N(0, I_d): the share of the truncated
    Gaussian on the ball that a half-space at distance s from the centre cuts
    off (the same for every direction, by rotational symmetry)."""
    if d == 1:
        num = stats.norm.cdf(q) - stats.norm.cdf(s)
        return num / (2 * stats.norm.cdf(q) - 1)
    num, _ = integrate.quad(
        lambda x: stats.norm.pdf(x) * stats.chi2.cdf(max(q * q - x * x, 0.0), d - 1),
        s, q, epsabs=1e-14, epsrel=1e-12)
    return num / stats.chi2.cdf(q * q, d)


def _cut_row(flow, v, q, s):
    """The unsafe half-space {v.y <= g} that cuts off the share _cap_share(q, s)
    of the flow set of ``flow`` (an _AffineFlow): in latent space it is the
    half-space at distance s from the centre with normal -A^T v."""
    v = np.asarray(v, dtype=np.float64)
    g = v @ flow.c - s * np.linalg.norm(flow.A.T @ v)
    return HalfSpace(v[None, :], np.array([[g]]))


def _miss_row(flow, v, q, gap):
    """The unsafe half-space {v.y <= g} whose least margin over the flow set is
    exactly ``gap`` > 0."""
    v = np.asarray(v, dtype=np.float64)
    g = v @ flow.c - q * np.linalg.norm(flow.A.T @ v) - gap
    return HalfSpace(v[None, :], np.array([[g]]))


def _row_margins(hs, y):
    """max over rows of G_i y - g_i, in float64, for a point y."""
    return float(np.max(hs.G @ np.asarray(y, dtype=np.float64) - hs.g.flatten()))


def _miss_probability(share, n):
    """The probability that n independent samples all miss a region holding
    ``share`` of the sampling law."""
    return (1.0 - share) ** n


def _most_wrong_certificates(seeds, beta):
    """The most wrong certificates over ``seeds`` independent seeds that a correct
    certifier exceeds with probability at most 1e-6: each wrong certificate has probability
    at most beta, so their count is at most Binomial(seeds, beta), and this is the least k
    with P(Binomial(seeds, beta) > k) <= 1e-6 (14 for 300 seeds at beta = 0.01)."""
    k = 0
    while stats.binom.sf(k, seeds, beta) > 1e-6:
        k += 1
    return k


def _exact_scenario_bound(beta, n):
    """1 - beta^(1/N) in 50-digit arithmetic: the smallest miscoverage that N
    samples with no violation support at confidence 1 - beta."""
    with mpmath.workdps(50):
        return 1 - mpmath.power(mpmath.mpf(beta), mpmath.mpf(1) / n)


def _dkw_epsilon(n, failure):
    """Dvoretzky-Kiefer-Wolfowitz: P(sup |F_n - F| > eps) <= 2 exp(-2 n eps^2)."""
    return math.sqrt(math.log(2.0 / failure) / (2.0 * n))


def _ks_distance(u):
    """sup |F_n(t) - t| for points u that are Uniform(0, 1) under the law."""
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = len(u)
    k = np.arange(1, n + 1)
    return float(max(np.max(k / n - u), np.max(u - (k - 1) / n)))


# --------------------------------------------------------------------------- #
# The samplers of the ball                                                      #
# --------------------------------------------------------------------------- #
# (q, dim, method): rejection by default in low dimension, the switch to the
# chi method when the acceptance is below 1 % (chi2.cdf(0.09, 6) = 1.5e-5),
# chi by default above dimension 30, both explicit methods, dimension one.
_BALL_REGIMES = [
    pytest.param(2.0, 2, None, id="rejection-default-d2"),
    pytest.param(1.5, 5, "rejection", id="rejection-explicit-d5"),
    pytest.param(0.3, 6, None, id="chi-by-low-acceptance-d6"),
    pytest.param(8.0, 50, None, id="chi-default-d50"),
    pytest.param(1.0, 2, "chi", id="chi-explicit-d2"),
    pytest.param(2.5, 1, None, id="rejection-default-d1"),
    pytest.param(0.7, 1, "chi", id="chi-explicit-d1"),
]


@pytest.mark.parametrize("q, dim, method", _BALL_REGIMES)
def test_truncated_ball_sampler_draws_the_truncated_gaussian_of_the_whole_ball(q, dim, method):
    """Every sample lies in the ball (exact, up to the 1e-12 of the inverse chi
    CDF), the radius follows the chi law cut at q and the direction is uniform.
    Catches a sampler that keeps the wrong candidates (outside the ball), that
    covers only part of the ball (rejection at 0.9 q, radii halved, the chi
    radius without its truncation), that draws from the wrong law (uniform
    candidates in the positive orthant) or that normalises the directions
    along the wrong axis. Radius: the probability integral transform
    chi.cdf(R)/chi.cdf(q) is uniform; direction: (u.v + 1)/2 for a fixed
    oblique unit v is Beta((d-1)/2, (d-1)/2) (a fair sign for d = 1). Each
    law is compared by DKW at failure probability 5e-8 (total below 1e-7).
    A 20-second limit per draw catches the default method kept on rejection
    when the acceptance is 1.5e-5 (the switch to chi dropped: the draw then
    takes minutes)."""
    n = 4000
    with _TimeLimit(20.0):
        z = sample_truncated_gaussian_ball(q=q, dim=dim, n_samples=n, method=method)
    assert z.shape == (n, dim)
    radius = np.linalg.norm(z, axis=1)
    assert np.all(radius <= q * (1 + 1e-12)), f"largest radius {radius.max()} > q = {q}"

    eps = _dkw_epsilon(n, 5e-8)
    pit = stats.chi.cdf(radius, dim) / stats.chi.cdf(q, dim)
    assert _ks_distance(pit) <= eps, (
        f"radius law: KS distance {_ks_distance(pit):.4f} > {eps:.4f}")

    v = np.arange(1, dim + 1, dtype=np.float64) * np.where(np.arange(dim) % 2, -1.0, 1.0)
    v /= np.linalg.norm(v)
    proj = (z @ v) / radius
    if dim == 1:
        positive = int(np.sum(proj > 0))
        lo, hi = stats.binom.ppf([2.5e-8, 1 - 2.5e-8], n, 0.5)
        assert lo <= positive <= hi, f"{positive} of {n} directions positive"
    else:
        a = (dim - 1) / 2.0
        pit_dir = stats.beta.cdf((proj + 1) / 2, a, a)
        assert _ks_distance(pit_dir) <= eps, (
            f"direction law: KS distance {_ks_distance(pit_dir):.4f} > {eps:.4f}")


def test_qmc_latents_stratify_every_coordinate_of_the_standard_normal():
    """A scrambled Sobol sequence of n = 2^m points has exactly one point in
    each interval [k/n, (k+1)/n) of every coordinate, so after the documented
    inverse normal CDF the k-th smallest value of Phi(z_j) lies in that
    interval. Catches QMC latents that skip the inverse CDF (all in [0, 1)),
    map with a wrong scale or sign, or are not a Sobol net: the certifier's
    'qmc' strategy would then sample a part of the latent space only.
    Deterministic for every seed."""
    n, dim = 256, 3
    for seed in (0, 7):
        z = _qmc_sample_latents(n, dim, seed=seed, dtype=torch.float64).numpy()
        assert z.shape == (n, dim)
        u = np.sort(stats.norm.cdf(z), axis=0)
        k = np.arange(n)[:, None]
        assert np.all(u >= k / n - 1e-9) and np.all(u <= (k + 1) / n + 1e-9), (
            f"seed {seed}: a coordinate has two points in one stratum")


def _chi2_cdf_exact(x, d):
    """P(chi2_d <= x) in 60-digit arithmetic (the regularised lower incomplete gamma
    function), which does not underflow where the double-precision value is 0."""
    with mpmath.workdps(60):
        return mpmath.gammainc(mpmath.mpf(d) / 2, 0, mpmath.mpf(x) / 2, regularized=True)


class _IdentityFlow:
    """A stand-in flow whose inverse map is the identity, in float64, in any dimension."""

    def inverse(self, z, **kwargs):
        return z.detach().to(torch.float64).clone()


@pytest.mark.parametrize("dim, q", [
    pytest.param(1000, 10.0, id="d1000-q10"),
    pytest.param(2000, 20.0, id="d2000-q20", marks=known_bug(
        13, "sample_truncated_gaussian_ball draws the chi radius as chi.ppf(U chi.cdf(q, d)); "
            "chi.cdf(q, d) is 0 in double precision when the ball holds less than about 1e-308 "
            "of the normal (d = 2000, q = 20: 4.6e-354), so every sample is the centre of the "
            "ball and the scenario check certifies a half-space that holds half of the set")),
])
def test_chi_sampler_reaches_the_rim_when_the_ball_holds_a_tiny_share_of_the_normal(dim, q):
    """The chi method promises the truncated Gaussian in any dimension ("Exact for any
    dimension"). A ball whose radius lies far below the bulk of the chi law holds a share of
    the normal below the smallest double (P(chi2_2000 <= 400) = 4.6e-354), and the truncated
    law then puts nearly all its mass near the rim. Reference in 60-digit arithmetic: a
    radius below q/2 has probability P(chi2_d <= q^2/4) / P(chi2_d <= q^2), below 1e-284
    in both cases, so a correct sampler returns one among 200 with probability far below
    1e-6; the half-space {y_0 >= 1/64} holds at least 1/2 - (1/64) phi(0)
    P(chi2_{d-1} <= q^2) / P(chi2_d <= q^2) > 0.48 of the set (1/2 by symmetry, less the
    largest density of y_0 times the width of the strip), so 60 samples of an identity
    stand-in flow all miss it with probability below 1e-17 and the certifier must not
    certify it. Catches radii that collapse to the centre when chi.cdf(q, d) underflows.
    The neighbour (d = 1000, q = 10, a share of 5.4e-307) lies just above the underflow."""
    n = 200
    with mpmath.workdps(60):
        p_ball = _chi2_cdf_exact(q * q, dim)
        p_inner = _chi2_cdf_exact(q * q / 4, dim) / p_ball
        density = mpmath.npdf(0) * _chi2_cdf_exact(q * q, dim - 1) / p_ball
    share = 0.5 - float(density) / 64
    if not (float(n * p_inner) < 1e-6 and (1 - share) ** 60 < 1e-12):
        pytest.fail(f"the case is too weak: {float(n * p_inner)}, {share}")
    np.random.seed(0)
    radius = np.linalg.norm(sample_truncated_gaussian_ball(q=q, dim=dim, n_samples=n), axis=1)
    assert np.all(radius <= q * (1 + 1e-12))
    assert np.all(radius >= q / 2), (
        f"{int(np.sum(radius < q / 2))} of {n} radii below q/2, the smallest {radius.min()}")
    unsafe = HalfSpace(-np.eye(1, dim), np.array([[-1.0 / 64]]))
    result = certify_halfspace_disjoint(_IdentityFlow(), q, unsafe, n_samples=60,
                                        beta_2=0.001, seed=0)
    assert result.disjoint is False


class _TimeLimit:
    """Turn a call that runs longer than ``seconds`` into a TimeoutError (a
    SIGALRM handler, restored afterwards), so that a sampler that never
    returns fails the test instead of hanging the run."""

    def __init__(self, seconds):
        self.seconds = seconds

    def _alarm(self, signum, frame):
        raise TimeoutError(f"still running after {self.seconds} s")

    def __enter__(self):
        self.previous = signal.signal(signal.SIGALRM, self._alarm)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)
        return False


# A contracting affine stand-in: its inverse map applied to the data (in place
# of the forward map) gives other points that still lie inside the ball, so
# that mistake shows as wrong samples and not as a sampler that never returns.
_CONTRACT_2D = _AffineFlow([[0.5, 0.25], [-0.125, 0.75]], [0.25, -0.125])


def test_empirical_latent_sampler_draws_noisy_latents_of_the_data_inside_the_ball():
    """The samples are training outputs mapped data to latent by the flow's
    forward map, perturbed by noise and kept only inside the ball. Stand-in
    affine flow with latent points known exactly: four on the sphere of radius
    0.95 q (noise of 0.1 pushes about half of their candidates outside) and one
    outside the ball. Catches a rejection test on the base point instead of the
    noisy candidate (samples outside the ball: exact check, no slack, since the
    returned values are the tested ones), the inverse map used in place of the
    forward map and a noise scale that is ignored (with noise 0 every sample is
    one of the inside latent points, exactly), and a resampling index that never
    reaches the last training point (with noise 0 all four inside points must
    be drawn). Each call has a 20-second limit, so a sampler that never returns
    fails here."""
    flow = _CONTRACT_2D
    q = 1.5
    angles = np.array([0.3, 1.9, 3.5, 5.0])
    z_in = 0.95 * q * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    z_out = np.array([[1.25 * q, 0.0]])
    # The outside point first, so that the last training point is one of the inside ones.
    z_all = np.concatenate([z_out, z_in])
    y_train = torch.from_numpy(z_all @ flow.A.T + flow.c)

    with _TimeLimit(20.0):
        noisy = sample_empirical_latent_ball(flow, y_train, q=q, n_samples=2000,
                                             noise_sigma=0.1, seed=3)
    assert noisy.shape == (2000, 2)
    assert np.all(np.linalg.norm(noisy, axis=1) <= q), "a sample lies outside the ball"
    assert np.max(np.linalg.norm(noisy, axis=1)) > 0.95 * q, "no sample reaches the noisy rim"

    with _TimeLimit(20.0):
        exact = sample_empirical_latent_ball(flow, y_train, q=q, n_samples=200,
                                             noise_sigma=0.0, seed=3)
    latents = np.array([flow.latent(y) for y in y_train.numpy()])
    gaps = np.linalg.norm(exact[:, None, :] - latents[None, 1:, :], axis=2)
    distance = np.min(gaps, axis=1)
    assert np.all(distance <= 1e-12), (
        f"with noise 0 a sample is {distance.max()} away from every latent of the data")
    # Each of the 4 inside points is drawn with probability 1/4 per accepted sample, so one
    # missing from 200 has probability at most 4 (3/4)^200 < 1e-24.
    drawn = set(np.argmin(gaps, axis=1).tolist())
    assert drawn == {0, 1, 2, 3}, f"only the inside training points {sorted(drawn)} were drawn"


@known_bug(13, "sample_empirical_latent_ball never returns when no candidate can "
               "land in the ball (noise 0 and every latent of the data outside "
               "the ball): it loops forever instead of refusing",
           raises=TimeoutError)
def test_empirical_latent_sampler_refuses_when_no_candidate_can_be_accepted():
    """With noise 0 the candidates are the latents of the data themselves; if
    all lie outside the ball no candidate is ever accepted, and the promise
    "returns n_samples points inside the ball" cannot be kept. The sampler must
    raise ValueError. A one-second alarm turns the endless loop of today into a
    TimeoutError (the marker absorbs only that)."""
    flow = _IDENTITY_2D
    y_train = torch.tensor([[2.0, 0.0], [0.0, -3.0]], dtype=torch.float64)
    with _TimeLimit(1.0):
        with pytest.raises(ValueError):
            sample_empirical_latent_ball(flow, y_train, q=1.0, n_samples=5,
                                         noise_sigma=0.0, seed=0)


# --------------------------------------------------------------------------- #
# certify_halfspace_disjoint: a polyhedron that holds part of the set          #
# --------------------------------------------------------------------------- #
def _square_polyhedron():
    """Identity flow, q = 2: {y0 <= 0, 2 y0 + y1 <= -3}. On the ball it is the
    cap {2 y0 + y1 <= -3} (distance 3/sqrt(5)); the polyhedron of the
    transposed matrix, {y0 + 2 y1 <= 0, y1 <= -3}, misses the ball."""
    return HalfSpace(np.array([[1.0, 0.0], [2.0, 1.0]]), np.array([[0.0], [-3.0]]))


def _triangle():
    """Identity flow, q = 2: three rows in two dimensions, a triangle around the
    centre holding the disc of radius 0.5/sqrt(2)."""
    return HalfSpace(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
                     np.array([[0.5], [0.5], [0.5]]))


_CUTS = {
    # name: (flow, q, half-space, share of the set it holds, at least)
    "cap-identity-oblique": (_IDENTITY_2D, 2.0,
                             _cut_row(_IDENTITY_2D, [0.6, 0.8], 2.0, 1.2),
                             _cap_share(2.0, 1.2, 2)),
    "cap-skew-flow": (_SKEW_2D, 2.0, _cut_row(_SKEW_2D, [1.0, -2.0], 2.0, 1.2),
                      _cap_share(2.0, 1.2, 2)),
    "square-two-rows": (_IDENTITY_2D, 2.0, _square_polyhedron(),
                        _cap_share(2.0, 3 / math.sqrt(5), 2)),
    "triangle-three-rows": (_IDENTITY_2D, 2.0, _triangle(),
                            (1 - math.exp(-0.125 / 2)) / (1 - math.exp(-2.0))),
}


@pytest.mark.parametrize("strategy", ["uniform", "qmc", "qmc+antithetic"])
@pytest.mark.parametrize("cut", list(_CUTS))
def test_a_polyhedron_holding_part_of_the_flow_set_is_never_certified(cut, strategy):
    """The polyhedron holds a known share of the flow set (closed form), so N =
    400 samples all miss it with probability (1 - share)^400 <= 1e-9 per seed
    under the truncated law, and less under QMC (whose untruncated samples
    cover the ball); three seeds. Catches the worst sample taken by argmax
    (only the sample farthest outside is tested), a sampling radius below q,
    candidates in the positive orthant (the oblique cap points to negative
    coordinates), the rows of G^T read for a square G, and a sampler or
    strategy that covers part of the ball only. The reported worst sample must
    lie in the polyhedron and, for the truncated law, in the set; its margin,
    recomputed, is the reported worst margin."""
    flow, q, hs, share = _CUTS[cut]
    n = 400
    if _miss_probability(share, n) > 1e-9:
        pytest.fail(f"the case is too weak: miss probability {_miss_probability(share, n)}")
    for seed in (0, 1, 2):
        result = certify_halfspace_disjoint(flow, q, hs, n_samples=n, beta_2=0.001,
                                            seed=seed, sampling_strategy=strategy)
        assert result.disjoint is False, (
            f"seed {seed}: certified disjoint although the polyhedron holds "
            f"{share:.3f} of the flow set")
        assert result.worst_max_margin <= 0
        recomputed = _row_margins(hs, result.worst_sample)
        assert abs(recomputed - result.worst_max_margin) <= 1e-12 * (1 + np.abs(hs.g).max()), (
            f"worst_sample has margin {recomputed}, reported {result.worst_max_margin}")
        assert recomputed <= 1e-12, "the worst sample is not in the polyhedron"
        if strategy == "uniform":
            assert np.linalg.norm(flow.latent(result.worst_sample)) <= q * (1 + _F32_REL) + 1e-12


@pytest.mark.parametrize("rows", ["one-row", "two-rows"])
def test_a_polyhedron_on_the_face_of_a_point_set_is_not_disjoint(rows):
    """The polyhedron is closed: a sample exactly on a face is inside it. The
    stand-in flow sends every latent point to y0 = (0.5, -0.25) (dyadic), so
    every margin is exactly 0. Catches 'disjoint' decided by worst margin >= 0
    in place of > 0, at every level: half-space, group and spec."""
    flow = _ConstantFlow([0.5, -0.25])
    hs = {"one-row": HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.5]])),
          "two-rows": HalfSpace(np.array([[1.0, 0.0], [0.0, 1.0]]),
                                np.array([[0.5], [0.0]]))}[rows]
    result = certify_halfspace_disjoint(flow, 1.0, hs, n_samples=50, beta_2=0.01, seed=0)
    assert result.worst_max_margin == 0.0
    assert result.disjoint is False
    assert certify_group_disjoint(flow, 1.0, [hs], n_samples=50, beta_2=0.01,
                                  seed=0).disjoint is False
    assert certify_spec_disjoint(flow, 1.0, [[hs]], n_samples=50, beta_2=0.01,
                                 seed=0).unsat_certified is False


def test_a_polyhedron_that_misses_the_flow_set_is_certified_with_its_worst_sample():
    """Completeness side, with the fields checked against closed forms. One row
    at least margin 0.5 from the skewed ellipsoid: certified, the reported
    worst margin is at least the exact least margin over the set (one-sided,
    the float32 slack only), the worst sample is a point of the set in data
    space whose recomputed margin is the reported one. Two rows of which each
    alone cuts the set but whose intersection misses it (AND of rows): the
    exact least value of max(1.5 - y0, 1.5 - y1) over the ball of radius 2 is
    1.5 - sqrt(2). Catches the latent point reported as the worst sample, the
    rows combined by OR, and a sample from outside the set (a false hit)."""
    q, n = 2.0, 400
    flow = _SKEW_2D
    v = np.array([1.0, -2.0])
    hs = _miss_row(flow, v, q, 0.5)
    slack = q * np.linalg.norm(flow.A.T @ v) * _F32_REL + 1e-12
    for seed in (0, 1):
        result = certify_halfspace_disjoint(flow, q, hs, n_samples=n, beta_2=0.001, seed=seed)
        assert result.disjoint is True
        assert result.worst_max_margin >= 0.5 - slack
        assert abs(_row_margins(hs, result.worst_sample) - result.worst_max_margin) <= 1e-12 * 16
        assert np.linalg.norm(flow.latent(result.worst_sample)) <= q * (1 + _F32_REL) + 1e-12
        assert result.n_samples_used == n

    both = HalfSpace(np.array([[-1.0, 0.0], [0.0, -1.0]]), np.array([[-1.5], [-1.5]]))
    result = certify_halfspace_disjoint(_IDENTITY_2D, q, both, n_samples=n, beta_2=0.001, seed=0)
    assert result.disjoint is True
    assert result.worst_max_margin >= 1.5 - math.sqrt(2) - 2 * _F32_REL - 1e-12
    assert abs(_row_margins(both, result.worst_sample) - result.worst_max_margin) <= 1e-12


# The slab |y_0 - y_1| <= 1/25, symmetric under y -> -y.
_SYMMETRIC_SLAB = HalfSpace(np.array([[1.0, -1.0], [-1.0, 1.0]]), np.full((2, 1), 0.04))


@pytest.mark.parametrize("strategy", [
    "uniform",
    "qmc",
    pytest.param("qmc+antithetic", marks=known_bug(
        13, "the 'qmc+antithetic' strategy pairs each Sobol point z with -z and states "
            "epsilon_2 = ln(1/beta)/N for N samples, but on a region symmetric about the centre "
            "a pair is one draw (both in or both out), so the certificate rests on N/2 draws: "
            "24 of 300 seeds certify a slab that holds 2.26 % of the set at beta = 0.01")),
])
def test_a_certificate_keeps_its_confidence_on_a_region_symmetric_about_the_centre(strategy):
    """On the flow set of an identity stand-in flow with q = 4 (the disc holds 99.97 % of the
    normal, so the law of every strategy is nearly the flow set's own), the symmetric slab
    holds the share 1 - 2 P(z_1 >= c | ||z|| <= 4) = 2.26 % with c = 1/(25 sqrt 2) (closed
    form), above epsilon_2 = ln(100)/256 = 1.80 %: every certificate is wrong. The promise
    allows a wrong certificate with probability at most beta = 0.01 per independent seed, so
    more than 14 of 300 seeds certify with probability below 1e-6 (binomial tail). Catches
    antithetic pairs counted as independent samples: 256 samples of 128 pairs miss the slab
    as often as 128 draws, about (1 - 0.0226)^128 = 5.4 % for independent draws."""
    n, beta, seeds = 256, 0.01, 300
    share = 1.0 - 2.0 * _cap_share(4.0, 0.04 / math.sqrt(2), 2)
    if not share > math.log(1.0 / beta) / n:
        pytest.fail(f"the case is too weak: share {share}")
    certified = sum(
        certify_halfspace_disjoint(_IDENTITY_2D, 4.0, _SYMMETRIC_SLAB, n_samples=n, beta_2=beta,
                                   seed=seed, sampling_strategy=strategy).disjoint
        for seed in range(seeds))
    assert certified <= _most_wrong_certificates(seeds, beta), (
        f"{certified} of {seeds} seeds certify a region holding {share:.4f} of the set")


# --------------------------------------------------------------------------- #
# The stated miscoverage                                                        #
# --------------------------------------------------------------------------- #
_GRID_N = (1, 3, 64, 1000, 4099)
_GRID_BETA = (0.5, 0.05, 1e-3, 1e-9)


def _stated_epsilon(kind, n, beta):
    flow = _IDENTITY_2D
    if kind == "certify_halfspace_disjoint":
        hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-5.0]]))
        r = certify_halfspace_disjoint(flow, 1.0, hs, n_samples=n, beta_2=beta, seed=0)
        assert r.disjoint is True and r.n_samples_used == n
        return r.epsilon_2, None
    if kind == "scenario_verify_halfspace":
        r = scenario_verify_halfspace(flow, 1.0, np.array([1.0, 0.0]), 5.0, n_samples=n, beta_2=beta)
        assert r.outcome == "verified" and r.n_samples_used == n
        return r.epsilon_2, r.delta_2
    r = verify_robustness(flow, 1.0, 0, 2, 0.01, 0.99, n, beta,
                          output_shift=np.array([10.0, 0.0]))
    assert r.outcome == "verified" and r.n_samples_used == n
    return r.epsilon_2, r.delta_2


@pytest.mark.parametrize("kind", ["certify_halfspace_disjoint", "scenario_verify_halfspace",
                                  "verify_robustness"])
def test_stated_miscoverage_is_the_scenario_bound(kind):
    """On a grid of N and beta, the stated epsilon_2 is the documented
    ln(1/beta)/N (within 4 ulps) and is never below 1 - beta^(1/N), the
    smallest miscoverage that N samples with no violation support at
    confidence 1 - beta (mpmath, 50 digits); the stated delta_2 is 1 - beta.
    Catches a logarithm in the wrong base, a sample count off by a factor,
    and a confidence that is not 1 - beta."""
    for n in _GRID_N:
        for beta in _GRID_BETA:
            eps, delta = _stated_epsilon(kind, n, beta)
            documented = math.log(1.0 / beta) / n
            assert abs(eps - documented) <= 4 * math.ulp(documented), (n, beta, eps)
            assert mpmath.mpf(eps) >= _exact_scenario_bound(beta, n), (n, beta, eps)
            if delta is not None:
                assert delta == 1.0 - beta


def test_robustness_states_the_joint_miscoverage_and_confidence():
    """The joint statement of verify_robustness is epsilon_total =
    1 - (1 - epsilon_1)(1 - epsilon_2) and delta_total = delta_1 delta_2.
    Soundness, one-sided against mpmath (50 digits): the stated miscoverage is
    at least 1 - (1 - epsilon_1) beta^(1/N), the joint value with the smallest
    scenario miscoverage N samples support, and the stated confidence is at
    most delta_1 (1 - (1 - epsilon_2)^N), the confidence the N samples give
    at the stated epsilon_2. The documented formulas hold within the rounding
    of their float evaluation (4 ulps of 1 for the difference from 1, 4 ulps
    of the value for the product). Values above and below one half and close
    to one. Catches the product of the miscoverages, the smaller of the two
    confidences, and a swapped pair."""
    for eps1, delta1, n, beta in [(0.001, 0.997, 200, 0.001), (0.2, 0.6, 7, 0.3),
                                  (0.6, 0.999999, 5000, 1e-6)]:
        r = verify_robustness(_IDENTITY_2D, 1.0, 1, 2, eps1, delta1, n, beta,
                              output_shift=np.array([0.0, 10.0]))
        assert r.outcome == "verified"
        with mpmath.workdps(50):
            least_eps = 1 - (1 - mpmath.mpf(eps1)) * mpmath.power(mpmath.mpf(beta), mpmath.mpf(1) / n)
            assert mpmath.mpf(r.epsilon_total) >= least_eps, (eps1, n, beta)
            most_delta = mpmath.mpf(delta1) * (1 - (1 - mpmath.mpf(r.epsilon_2)) ** n)
            assert mpmath.mpf(r.delta_total) <= most_delta, (delta1, n, beta)
            eps2 = mpmath.log(1 / mpmath.mpf(beta)) / n
            documented_eps = float(1 - (1 - mpmath.mpf(eps1)) * (1 - eps2))
            documented_delta = float(mpmath.mpf(delta1) * (1 - mpmath.mpf(beta)))
        assert abs(r.epsilon_total - documented_eps) <= 4 * math.ulp(1.0)
        assert abs(r.delta_total - documented_delta) <= 4 * math.ulp(documented_delta)


# --------------------------------------------------------------------------- #
# Groups and specs                                                              #
# --------------------------------------------------------------------------- #
_Q = 2.0
_N = 400
_CUT_A = _cut_row(_IDENTITY_2D, [0.6, 0.8], _Q, 1.2)
_CUT_B = _cut_row(_IDENTITY_2D, [1.0, 0.0], _Q, 1.2)
_MISS_A = _miss_row(_IDENTITY_2D, [1.0, 0.0], _Q, 0.75)
_MISS_B = _miss_row(_IDENTITY_2D, [0.0, -1.0], _Q, 0.25)
_LOG = math.log(1000.0) / _N


@pytest.mark.parametrize("order", ["miss-then-cut", "cut-then-miss", "miss-cut-miss"])
def test_a_group_with_a_member_that_holds_part_of_the_set_is_not_disjoint(order):
    """A group is the OR of its members: it is disjoint from the set only when
    every member is. One member holds 8 % of the set (miss probability below
    1e-14 per seed). Catches members combined by any() in place of all(); the
    stated epsilon is the sum over the members that ran (the early exit stops
    at the member that is hit), which catches the largest member epsilon
    stated in place of the sum."""
    group, ran = {"miss-then-cut": ([_MISS_A, _CUT_A], 2),
                  "cut-then-miss": ([_CUT_A, _MISS_A], 1),
                  "miss-cut-miss": ([_MISS_A, _CUT_A, _MISS_B], 2)}[order]
    for seed in (0, 1, 2):
        r = certify_group_disjoint(_IDENTITY_2D, _Q, group, n_samples=_N, beta_2=0.001, seed=seed)
        assert r.disjoint is False
        assert len(r.per_hs_results) == ran
        assert abs(r.epsilon_2 - ran * _LOG) <= 1e-15


def test_a_group_of_members_that_all_miss_the_set_is_disjoint_with_the_summed_epsilon():
    """Every member misses the set: the group is certified and states the sum
    of the member epsilons, here 2 ln(1/beta)/N. Catches the largest member
    epsilon (one test's worth) stated for a group of two tests."""
    r = certify_group_disjoint(_IDENTITY_2D, _Q, [_MISS_A, _MISS_B], n_samples=_N,
                               beta_2=0.001, seed=0)
    assert r.disjoint is True and len(r.per_hs_results) == 2
    assert abs(r.epsilon_2 - 2 * _LOG) <= 1e-15


@pytest.mark.parametrize("spec, unsat, idx, ran", [
    pytest.param([[_CUT_A], [_CUT_B]], False, None, 2, id="every-group-hit"),
    pytest.param([[_MISS_A, _CUT_A]], False, None, 2, id="one-group-with-a-hit-member"),
    pytest.param([[_CUT_A], [_MISS_A, _MISS_B]], True, 1, 3, id="second-group-misses"),
    pytest.param([[_MISS_A], [_CUT_A]], True, 0, 1, id="first-group-misses"),
])
def test_spec_is_unsat_only_through_a_group_disjoint_from_the_set(spec, unsat, idx, ran):
    """A spec is the AND of its groups: UNSAT is certified when one group is
    disjoint from the set, and never when every group holds part of it (8 %
    each, three seeds). The stated epsilon is the sum over every half-space
    test that ran, the failed groups before the certifying one included
    (here 3 tests when group 0 fails at its only member and group 1 certifies
    with two). Catches the first group that is hit taken as certifying, a
    loop that stops after the first group, groups combined by OR, and the
    certifying group's epsilon stated alone (the tests of the earlier groups
    dropped from the union bound)."""
    for seed in (0, 1, 2):
        r = certify_spec_disjoint(_IDENTITY_2D, _Q, spec, n_samples=_N, beta_2=0.001, seed=seed)
        assert r.unsat_certified is unsat
        assert r.certifying_group_idx == idx
        executed = sum(len(g.per_hs_results) for g in r.per_group_results)
        assert executed == ran
        assert abs(r.epsilon_2 - ran * _LOG) <= 1e-15


# --------------------------------------------------------------------------- #
# The public route: verify_specification on a whitened flow set                #
# --------------------------------------------------------------------------- #
_MEAN = np.array([10.0, -3.0])
_STD = np.array([4.0, 0.5])


def _whitened_flow_set():
    """A flow set over raw outputs y = mean + std * z, ||z|| <= 2, held in the
    whitened frame by an identity stand-in flow and an AffineTransform."""
    return ProbabilisticSet(FlowScore(_AffineFlow(np.eye(2), np.zeros(2))), threshold=_Q,
                            m=1000, ell=999, epsilon=0.01, dim=2,
                            affine_transform=AffineTransform(mean=_MEAN, std=_STD))


def _raw_at_least(i, s):
    """The raw unsafe region {y_i >= mean_i + std_i s}: in the whitened frame
    the cap {z_i >= s}."""
    G = np.zeros((1, 2))
    G[0, i] = -1.0
    return HalfSpace(G, np.array([[-(_MEAN[i] + _STD[i] * s)]]))


def test_verify_specification_scenario_is_never_unsat_on_a_spec_holding_part_of_the_set():
    """Public route: the raw unsafe region holds 8 % of the flow set (whitened,
    the cap z_0 >= 1.2), so the answer is UNKNOWN for every seed (miss
    probability below 1e-14 each). Catches a spec read in the wrong frame and
    every certifier defect that turns a hit into a certificate."""
    pset = _whitened_flow_set()
    for seed in (0, 1, 2):
        result = verify_specification(pset, _raw_at_least(0, 1.2),
                                      config=ProbVerifyConfig(n_samples=_N, seed=seed))
        assert result.verdict == "UNKNOWN"


def test_verify_specification_scenario_unsat_states_the_union_bound_over_every_test():
    """Public route, a spec of two groups: the first holds part of the set
    (hit), the second is the OR of two raw half-spaces that miss it. The answer
    is UNSAT, and the stated joint miscoverage rests on all three tests that
    ran: it is at least 1 - (1 - 0.01)(1 - 3 (1 - beta^(1/N))), the union
    bound over three tests at the smallest miscoverage each supports (mpmath),
    and it is the documented 1 - (1 - 0.01)(1 - 3 ln(1/beta)/N) within the
    rounding of its float evaluation; delta_total is the set's confidence
    times 1 - beta.
    Catches the epsilon of the certifying group stated alone. A raw half-space
    that misses the raw set but would cut the set read in whitened
    coordinates checks that the spec is whitened (it must be certified)."""
    pset = _whitened_flow_set()
    spec = [{"Hg": _raw_at_least(0, 1.2)},
            {"Hg": [_raw_at_least(1, 2.5), _raw_at_least(0, 2.25)]}]
    result = verify_specification(pset, spec, config=ProbVerifyConfig(n_samples=_N, seed=0))
    assert result.verdict == "UNSAT"
    with mpmath.workdps(50):
        least = 1 - (1 - mpmath.mpf(0.01)) * (1 - 3 * _exact_scenario_bound(0.001, _N))
        assert mpmath.mpf(result.epsilon_total) >= least
        documented = float(1 - (1 - mpmath.mpf(0.01)) * (1 - 3 * mpmath.log(1000) / _N))
    assert abs(result.epsilon_total - documented) <= 4 * math.ulp(1.0)
    _, confidence = compute_guarantee(1000, 999, 0.01)
    assert abs(result.delta_total - confidence * (1 - 0.001)) <= 1e-15
    assert result.n_samples_used == _N


# The unsafe box [-1/10, 1/10]^2, inside the disc of radius 1/2.
_CENTRE_BOX = HalfSpace(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
                        np.full((4, 1), 0.1))


@pytest.mark.parametrize("strategy", [
    "uniform",
    pytest.param("qmc", marks=known_bug(
        13, "the 'qmc' strategy of the scenario check draws its latent points from the whole "
            "standard normal and never reads the set's radius, so a region inside a flow set "
            "whose ball holds a small part of the normal is sampled less often than the stated "
            "epsilon_2 assumes: verify_specification answers UNSAT with a guarantee that does "
            "not hold (33 wrong UNSAT of 300 seeds at beta = 0.01)")),
])
def test_verify_specification_scenario_keeps_its_confidence_for_a_region_inside_the_set(strategy):
    """Public route, every sampling strategy that ProbVerifyConfig accepts. The flow set is
    the disc of radius 1/2 of an identity stand-in flow; it holds 1 - e^{-1/8} = 11.75 % of
    the standard normal. The unsafe box lies inside it and holds the share (2 Phi(1/10) -
    1)^2 / (1 - e^{-1/8}) = 5.40 % of it (closed form), above epsilon_2 = ln(100)/256 = 1.80 %
    and above the stated epsilon_total, 3.76 %: every UNSAT rests on a wrong scenario
    certificate. The set is given, so its conformal layer adds no randomness, and the promise
    allows a wrong UNSAT with probability at most beta = 0.01 per independent seed: more than
    14 of 300 seeds answer UNSAT with probability below 1e-6 (binomial tail). Catches latent
    points that are not drawn from the flow set's own law (a radius not read, points outside
    the ball), which make the region inside the set rarer in the sample than the bound says."""
    n, beta, seeds = 256, 0.01, 300
    pset = ProbabilisticSet(FlowScore(_AffineFlow(np.eye(2), np.zeros(2))), threshold=0.5,
                            m=1000, ell=999, epsilon=0.02, dim=2)
    share = (2 * stats.norm.cdf(0.1) - 1) ** 2 / (1 - math.exp(-0.125))
    if not share > 1 - (1 - 0.02) * (1 - math.log(1.0 / beta) / n):
        pytest.fail(f"the case is too weak: share {share}")
    unsat = sum(
        verify_specification(pset, _CENTRE_BOX, config=ProbVerifyConfig(
            n_samples=n, beta=beta, seed=seed, sampling_strategy=strategy)).verdict == "UNSAT"
        for seed in range(seeds))
    assert unsat <= _most_wrong_certificates(seeds, beta), (
        f"{unsat} of {seeds} seeds answer UNSAT for a region holding {share:.4f} of the set")


# --------------------------------------------------------------------------- #
# scenario_verify_halfspace and verify_robustness                              #
# --------------------------------------------------------------------------- #
def test_scenario_halfspace_reports_the_largest_violation_among_given_latent_samples():
    """With the latent samples given, the margins are known exactly (dyadic
    data): y = A z + c + shift, margin = w.y - b. The property w.y <= b is
    violated only after the shift is added. The counterexample is the sample of
    largest margin, with its latent point, its raw output and its margin.
    Catches the shift subtracted, the smallest margin taken, a violation
    required of every sample, and the given samples ignored."""
    flow = _AffineFlow([[2.0, 0.5], [-0.25, 0.75]], [0.5, -1.5])
    shift = np.array([1.0, -0.5])
    z = np.array([[0.5, 0.25], [-1.0, 0.5], [0.75, -0.5], [0.25, 0.0], [-0.5, -1.0]])
    w, b = np.array([1.0, -2.0]), 6.25
    y_raw = z @ flow.A.T + flow.c + shift
    margins = y_raw @ w - b
    assert np.sum(margins > 0) == 2 and np.all((z @ flow.A.T + flow.c - shift) @ w - b <= 0)
    worst = int(np.argmax(margins))
    r = scenario_verify_halfspace(flow, 2.0, w, b, n_samples=5, beta_2=0.01,
                                  output_shift=shift, latent_samples=z)
    assert r.outcome == "unknown" and r.verified is False
    z_ce, y_ce, m_ce = r.counterexample
    np.testing.assert_array_equal(z_ce, z[worst])
    np.testing.assert_array_equal(y_ce, y_raw[worst])
    assert m_ce == margins[worst]

    r = scenario_verify_halfspace(flow, 2.0, w, b + 2.0, n_samples=5, beta_2=0.01,
                                  output_shift=shift, latent_samples=z)
    assert r.outcome == "verified" and r.counterexample is None


@pytest.mark.parametrize("q, s", [(2.0, 1.2), (0.8, 0.68)])
def test_scenario_halfspace_never_verifies_a_property_violated_on_part_of_the_set(q, s):
    """The safe property v.y <= b fails on a cap holding a known share of the
    flow set (closed form; N = 1000 samples miss it with probability below
    1e-12 per seed, three seeds), so the answer is never 'verified'. Radius
    below one included: a radius read as its square would shrink the ball
    there. The counterexample lies in the set, violates the property and its
    margin, recomputed, is the reported one."""
    flow = _IDENTITY_2D
    v = np.array([-0.6, -0.8])
    b = s
    n = 1000
    assert _miss_probability(_cap_share(q, s, 2), n) < 1e-12
    for seed in (0, 1, 2):
        np.random.seed(seed)
        r = scenario_verify_halfspace(flow, q, v, b, n_samples=n, beta_2=0.001)
        assert r.outcome != "verified"
        z_ce, y_ce, m_ce = r.counterexample
        assert m_ce > 0 and abs(float(y_ce @ v - b) - m_ce) <= 1e-12
        assert np.linalg.norm(y_ce) <= q * (1 + _F32_REL) + 1e-12
        np.testing.assert_array_equal(y_ce, z_ce.astype(np.float32).astype(np.float64))


class _RobustnessFlow(_AffineFlow):
    """Three classes; centred outputs (5 + z_0/10, z_1/10, z_2/10) in which
    class 0 always wins. With the raw shift (0, -1, 4.85) class 2 beats class 0
    where z_2 - z_0 > 1.5, a cap at distance 1.5/sqrt(2) holding 9.6 % of the
    ball of radius 2 in three dimensions; class 1 never wins. ``perm`` relabels
    the classes."""

    SHIFT = np.array([0.0, -1.0, 4.85])

    def __init__(self, perm):
        P = np.eye(3)[perm]
        super().__init__(P @ (0.1 * np.eye(3)), P @ np.array([5.0, 0.0, 0.0]))
        self.shift = P @ self.SHIFT
        self.true_class = int(np.argmax(P[:, 0]))
        self.winner = int(np.argmax(P[:, 2]))


@pytest.mark.parametrize("perm", [[0, 1, 2], [1, 0, 2], [2, 1, 0]],
                         ids=["true-first-winner-last", "true-middle-winner-last",
                              "true-last-winner-first"])
def test_robustness_is_never_verified_when_a_wrong_class_wins_on_part_of_the_set(perm):
    """A wrong class beats the true class on a cap holding 9.6 % of the flow
    set (closed form; N = 400 samples miss it with probability below 1e-17,
    three seeds), and only in raw logits: the centred flow outputs alone are
    robust. Catches the last class skipped by the loop, the margins computed
    on the centred outputs, and a joint verdict that misses a class. The
    counterexample names the winning class, lies in the set and its raw output
    has that class strictly above the true class."""
    flow = _RobustnessFlow(perm)
    q, n = 2.0, 400
    assert _miss_probability(_cap_share(q, 1.5 / math.sqrt(2), 3), n) < 1e-17
    for seed in (0, 1, 2):
        np.random.seed(seed)
        r = verify_robustness(flow, q, flow.true_class, 3, 0.01, 0.99, n, 0.001,
                              output_shift=flow.shift)
        assert r.outcome != "verified" and r.verified is False
        z_ce, y_ce, k = r.counterexample
        assert k == flow.winner
        assert y_ce[k] > y_ce[flow.true_class]
        assert np.linalg.norm(z_ce) <= q * (1 + _F32_REL) + 1e-12
        expected = z_ce.astype(np.float32).astype(np.float64) @ flow.A.T + flow.c + flow.shift
        np.testing.assert_allclose(y_ce, expected, rtol=0, atol=1e-12)


def test_robustness_reports_the_largest_margin_among_given_latent_samples():
    """With the latent samples given (dyadic), the worst (sample, class) pair
    is known exactly: the largest y[k] - y[true] over every sample and every
    class but the true one, the true class in the middle. The centred outputs
    have another worst pair. Catches the last class skipped and the margins
    computed without the shift."""
    flow = _AffineFlow(np.eye(4), np.zeros(4))
    shift = np.array([-0.5, 0.0, 0.0, 0.5])
    z = np.array([[0.25, 0.5, 0.0, 0.5], [1.0, 0.25, 0.5, 0.0], [0.0, 0.75, 0.25, 1.0]])
    y_raw = z + shift
    true = 1
    margins = np.delete(y_raw, true, axis=1) - y_raw[:, [true]]
    i, j = np.unravel_index(int(np.argmax(margins)), margins.shape)
    k = [c for c in range(4) if c != true][j]
    assert k == 3 and margins[i, j] > 0 and np.sum(margins == margins[i, j]) == 1
    centred = np.delete(z, true, axis=1) - z[:, [true]]
    assert np.unravel_index(int(np.argmax(centred)), centred.shape) != (i, j)
    r = verify_robustness(flow, 2.0, true, 4, 0.01, 0.99, 3, 0.01,
                          output_shift=shift, latent_samples=z)
    assert r.outcome == "unknown"
    z_ce, y_ce, cls = r.counterexample
    assert cls == k
    np.testing.assert_array_equal(z_ce, z[i])
    np.testing.assert_array_equal(y_ce, y_raw[i])


class _NaNFirstSample(_AffineFlow):
    """Identity flow, except that the first sample's output is NaN in every
    coordinate: a flow whose integration failed for one latent point."""

    def __init__(self, dim):
        super().__init__(np.eye(dim), np.zeros(dim))

    def inverse(self, z, **kwargs):
        y = super().inverse(z)
        y[0, :] = float("nan")
        return y


_NAN_LATENTS = np.array([[0.0, 0.0], [0.0, 0.875], [0.0, 0.75]])


@known_bug(13, "verify_robustness takes np.argmax of the class margins, which "
               "returns the NaN entry when one sample's output is NaN, so that "
               "sample hides every violating sample of its class and the set "
               "is reported 'verified'")
def test_robustness_is_not_verified_when_one_sample_output_is_nan():
    """Samples 1 and 2 have class 1 above the true class 0 by 0.875 and 0.75;
    sample 0 has a NaN output. The answer must not be 'verified' (the passing
    neighbour below runs scenario_verify_halfspace on the same samples)."""
    r = verify_robustness(_NaNFirstSample(2), 1.0, 0, 2, 0.01, 0.99, 3, 0.01,
                          latent_samples=_NAN_LATENTS)
    assert r.outcome != "verified"


def test_scenario_halfspace_is_not_verified_when_one_sample_output_is_nan():
    """Passing neighbour of the two NaN cases: the same samples, the property
    y_1 <= 0. The answer is not 'verified', because the violation test is made
    sample by sample. Catches a violation test that a NaN entry can mask."""
    r = scenario_verify_halfspace(_NaNFirstSample(2), 1.0, np.array([0.0, 1.0]), 0.0,
                                  n_samples=3, beta_2=0.01, latent_samples=_NAN_LATENTS)
    assert r.outcome != "verified"


@known_bug(13, "scenario_verify_halfspace picks the counterexample by np.argmax "
               "of the margins, which returns the NaN entry, so the reported "
               "counterexample is the sample whose output is NaN and not a "
               "violation")
def test_scenario_halfspace_counterexample_violates_when_one_sample_output_is_nan():
    """The same samples: the reported counterexample must be a violating
    sample, here sample 1 with margin 0.875 (exact, dyadic)."""
    r = scenario_verify_halfspace(_NaNFirstSample(2), 1.0, np.array([0.0, 1.0]), 0.0,
                                  n_samples=3, beta_2=0.01, latent_samples=_NAN_LATENTS)
    if r.counterexample is None:
        pytest.fail("the case needs a counterexample")
    assert r.counterexample[2] == 0.875


# --------------------------------------------------------------------------- #
# Witnesses: preimage_search and the 'falsified' outcome                        #
# --------------------------------------------------------------------------- #
_M = torch.tensor([[2.0, 0.5], [0.0, 1.0]], dtype=torch.float32)
_B0 = torch.tensor([0.25, -0.5], dtype=torch.float32)
_LB, _UB = np.array([-1.0, 0.5]), np.array([1.0, 2.0])
_WIDE_LB, _WIDE_UB = np.full(2, -2.0), np.full(2, 2.0)


def _linear_target(x):
    return x @ _M.T + _B0


def _replay(x):
    """f(x) = M x + b0 in float64, from the returned input."""
    return np.asarray(x, dtype=np.float64) @ _M.double().numpy().T + _B0.double().numpy()


def test_preimage_search_returns_an_input_of_the_box_that_reaches_the_target():
    """f(x) = M x + b0 with M non-symmetric. A target reached at x* = (0.25, 1)
    inside the dyadic box is found: x lies in the box (exact compare) and f(x),
    replayed in float64, is within the tolerance of the target; y_achieved is
    f(x). Catches the starting point of a restart returned in place of the
    optimised one."""
    torch.manual_seed(0)
    target = _replay([0.25, 1.0])
    r = preimage_search(_linear_target, target, (_LB, _UB), n_restarts=3, n_steps=300,
                        lr=0.05, tolerance=1e-3)
    assert r.found is True
    x = r.x.astype(np.float64)
    assert np.all(x >= _LB) and np.all(x <= _UB)
    assert np.linalg.norm(_replay(x) - target) < 1e-3 * (1 + 1e-6)
    assert np.linalg.norm(np.asarray(r.y_achieved, dtype=np.float64) - _replay(x)) <= 1e-5


def test_preimage_search_finds_nothing_for_a_target_reached_only_outside_the_box():
    """The target f(x*) with x* = (-1.5, 1) is reached only below the lower
    bound of x_0: over the box the distance is at least 0.5 times the least
    singular value of M (closed form, above 0.4), far above the tolerance, so
    found is False and no input or output is returned. Catches a projection
    that clamps to the upper bound only (x leaves the box and reaches the
    target) and a not-found result that still returns an input."""
    torch.manual_seed(0)
    target = _replay([-1.5, 1.0])
    least = 0.5 * float(np.linalg.svd(_M.double().numpy(), compute_uv=False).min())
    assert least > 0.4
    r = preimage_search(_linear_target, target, (_LB, _UB), n_restarts=3, n_steps=300,
                        lr=0.05, tolerance=1e-3)
    assert r.found is False
    assert r.x is None and r.y_achieved is None
    assert r.distance >= least * (1 - 1e-6)


def test_falsified_outcome_carries_an_input_of_the_box_whose_real_output_violates():
    """A 'falsified' answer is a witness: the genuine input lies in the input
    box [-2, 2]^2 (dyadic, exact compare) and its real output, replayed in float64,
    violates the property, for scenario_verify_halfspace (y_0 <= 1 fails where
    2 x_0 + x_1 / 2 + 1/4 > 1) and verify_robustness (class 1 beats class 0).
    Catches the flow counterexample checked in place of the real output."""
    torch.manual_seed(0)
    np.random.seed(0)
    r = scenario_verify_halfspace(_IDENTITY_2D, 2.0, np.array([1.0, 0.0]), 1.0, n_samples=400,
                                  beta_2=0.01, target_fn=_linear_target,
                                  input_set_bounds=(_WIDE_LB, _WIDE_UB), preimage_n_restarts=3,
                                  preimage_n_steps=300, preimage_tolerance=0.05)
    assert r.outcome == "falsified"
    x = r.genuine_input.astype(np.float64)
    assert np.all(x >= _WIDE_LB) and np.all(x <= _WIDE_UB)
    assert _replay(x)[0] > 1.0

    torch.manual_seed(0)
    np.random.seed(0)
    r = verify_robustness(_IDENTITY_2D, 2.0, 0, 2, 0.01, 0.99, 400, 0.01,
                          target_fn=_linear_target, input_set_bounds=(_WIDE_LB, _WIDE_UB),
                          preimage_n_restarts=3, preimage_n_steps=300,
                          preimage_tolerance=0.05)
    assert r.outcome == "falsified"
    x = r.genuine_input.astype(np.float64)
    assert np.all(x >= _WIDE_LB) and np.all(x <= _WIDE_UB)
    y = _replay(x)
    assert y[1] > y[0]


def _tie_target(x):
    """f(x) = (x_0, x_0): the two outputs tie at every input."""
    return torch.stack([x[..., 0], x[..., 0]], dim=-1)


def _boundary_target(x):
    """f(x) = (1/2, x_1): the first output is 1/2 at every input."""
    return torch.stack([torch.full_like(x[..., 0], 0.5), x[..., 1]], dim=-1)


@pytest.mark.parametrize("entry", ["robustness-tie", "halfspace-boundary"])
def test_a_real_output_on_the_boundary_of_the_property_is_not_a_violation(entry):
    """The flow set holds violations, the preimage search finds an input
    within its loose tolerance, but the real output lies exactly on the
    boundary of the property, which it satisfies: a tie between the true
    class and another (robustness is y[k] - y[true] <= 0), or w.y = b exactly
    (dyadic). No input violates, so the answer is 'unknown', never
    'falsified'. Catches a strict comparison at the real output read as a
    non-strict one (a witness that is not a counterexample)."""
    torch.manual_seed(0)
    np.random.seed(0)
    bounds = (np.full(2, -1.0), np.full(2, 1.0))
    if entry == "robustness-tie":
        r = verify_robustness(_IDENTITY_2D, 2.0, 0, 2, 0.01, 0.99, 200, 0.01,
                              target_fn=_tie_target, input_set_bounds=bounds,
                              preimage_n_restarts=1, preimage_n_steps=5,
                              preimage_tolerance=100.0)
    else:
        r = scenario_verify_halfspace(_IDENTITY_2D, 2.0, np.array([1.0, 0.0]), 0.5,
                                      n_samples=200, beta_2=0.01, target_fn=_boundary_target,
                                      input_set_bounds=bounds, preimage_n_restarts=1,
                                      preimage_n_steps=5, preimage_tolerance=100.0)
    if r.counterexample is None:
        pytest.fail("the case needs a violation in the flow set")
    assert r.outcome == "unknown"
    assert r.genuine_input is None


_UB_TENTH = np.array([0.1, 0.1])     # 0.1 rounds up in float32


@known_bug(13, "preimage_search projects onto the input box cast to float32 by "
               "rounding to nearest, so the returned input can lie above an "
               "upper bound (0.1 becomes 0.10000000149) and 'falsified' carries "
               "a witness outside the input set")
def test_preimage_witness_lies_in_a_box_whose_bounds_float32_cannot_hold():
    """The box [0, 0.1]^2 and f(x) = x; the target (0.1001, 0.05) pulls x_0 onto
    the upper bound. The returned input must lie in the box given, compared in
    float64 with no slack. Neighbour: the dyadic-box cases above."""
    torch.manual_seed(0)
    r = preimage_search(lambda x: x, np.array([0.1001, 0.05]), (np.zeros(2), _UB_TENTH),
                        n_restarts=2, n_steps=300, lr=0.05, tolerance=1e-3)
    if not r.found:
        pytest.fail("the case needs a found preimage")
    assert np.all(r.x.astype(np.float64) <= _UB_TENTH)


@pytest.mark.parametrize("entry", ["halfspace", "robustness"])
@known_bug(13, "scenario_verify_halfspace and verify_robustness answer 'falsified' "
               "with a genuine input outside the input box when a bound is not "
               "representable in float32 (preimage_search projects onto the "
               "float32-rounded box, and their own check of the box allows 1e-6)")
def test_falsified_witness_lies_in_a_box_whose_bounds_float32_cannot_hold(entry):
    """The same box through both callers: identity flow and identity target;
    the property y_0 <= 0.05 (halfspace) or class 0 above class 1
    (robustness, whose witness is pulled onto the upper bound of x_1). The
    witness found at the upper bound must lie in the box given (float64, no
    slack)."""
    torch.manual_seed(0)
    np.random.seed(0)
    if entry == "halfspace":
        r = scenario_verify_halfspace(_IDENTITY_2D, 1.0, np.array([1.0, 0.0]), 0.05,
                                      n_samples=200, beta_2=0.01, target_fn=lambda x: x,
                                      input_set_bounds=(np.zeros(2), _UB_TENTH),
                                      preimage_n_restarts=2, preimage_n_steps=300,
                                      preimage_tolerance=10.0)
    else:
        r = verify_robustness(_IDENTITY_2D, 1.0, 0, 2, 0.01, 0.99, 200, 0.01,
                              target_fn=lambda x: x, input_set_bounds=(np.zeros(2), _UB_TENTH),
                              preimage_n_restarts=2, preimage_n_steps=300,
                              preimage_tolerance=10.0)
    if r.outcome != "falsified":
        pytest.fail(f"the case needs a 'falsified' answer, got {r.outcome}")
    assert np.all(r.genuine_input.astype(np.float64) <= _UB_TENTH)
