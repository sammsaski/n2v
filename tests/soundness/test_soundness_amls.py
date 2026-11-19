"""Adaptive multilevel splitting: the estimate of the unsafe mass and the certificate built on it.

Covers ``n2v/probabilistic/flow/amls.py`` (``amls_estimate_halfspace_mass``,
``amls_certify_spec`` and their helpers ``_phi_halfspace_torch``,
``_push_through_flow``, ``_resolve_device``) and
``n2v/probabilistic/flow/amls_bounded.py`` (``amls_bounded_estimate_halfspace_mass``,
``amls_bounded_estimate_union_mass`` with ``_phi_union_torch``,
``amls_bounded_certify_spec``, ``amls_bounded_certify_spec_union``), called
directly and through ``verify_specification`` on a ``ProbabilisticSet``.

The promise. The unsafe set of a half-space is U = {y : G y <= g} (every row
holds); a group of half-spaces is their union. An estimator returns
``pi_hat``, an estimate of the mass of U under the flow (for the bounded
variants: under the standard Gaussian latent restricted to the ball
``||z|| <= q``), and ``pi_upper``, documented as a (1 - beta) upper bound on
that mass; ``detected_unsafe`` says that a sample landed in U, and
``worst_y`` is then a point of U. A certificate (``unsat_certified``, verdict
UNSAT) says "the unsafe mass is at most the target" and is given only when
the run did not detect U and its upper bound supports the target; a run that
could not estimate must not be read as "the mass is zero".
``amls_certify_spec`` is documented as a detector: its ``unsat_certified``
means "some group had no detection", and the verdict built on that is
registry PROB-8 (``tests/contracts/test_probabilistic_contract.py``).

Oracles. The flow is replaced by maps whose push-forward of the Gaussian
latent is known in closed form: a shift y = z + C (a stand-in object, or the
library's ``FlowODE`` with a constant velocity field, which integrates to the
same shift), and ``FlowODE`` with a linear velocity integrated by Euler in a
fixed number of steps, which is a scaling of the latent. The mass of a
half-space, or of a set whose rows constrain orthogonal directions, is a
product of normal tails (``scipy.stats.norm``); restricted to the ball it is
a one-dimensional integral (``scipy.integrate.quad``, to 1e-12).

Statistics. Every case fixes its seeds. The count of the first level is
Binomial(N, p) exactly, because the first sample is drawn independently of
everything else: it is checked against a window of binomial probability 1e-8
on each side, and the Clopper-Pearson bound of that level against the true
mass at beta = 1e-7. Cases on the splitting levels rest on the central limit
theorem of Cerou and Guyader: log(pi_hat / p) is close to normal with
variance K (1 - rho) / (rho N) (0.18 for K = 4 levels and N = 200) when each
level is sampled exactly; the correlation of the Markov chain inflates it,
and the standard deviation measured on the unmodified library is at most
1.2 (300 seeds). Each case states
the probability that a correct implementation fails it under that normal
approximation.

Convention: SAT = unsafe; the unsafe region is ``G y <= g``.
"""

import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.integrate import quad
from scipy.stats import beta as beta_distribution
from scipy.stats import binom, norm

from n2v.probabilistic.flow.amls import (amls_certify_spec,
                                         amls_estimate_halfspace_mass)
from n2v.probabilistic.flow.amls_bounded import (
    amls_bounded_certify_spec, amls_bounded_certify_spec_union,
    amls_bounded_estimate_halfspace_mass, amls_bounded_estimate_union_mass)
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.sets.halfspace import HalfSpace
from n2v.utils.verify_specification import (ProbVerifyConfig,
                                            verify_specification)
from tests.known_bugs import known_bug

# --------------------------------------------------------------------------- #
# Geometry: a 2-D latent, data y = z + C, unsafe sets written in the latent    #
# --------------------------------------------------------------------------- #
A = np.array([0.6, -0.8])          # a unit direction that is not an axis
A_PERP = np.array([0.8, 0.6])      # the unit direction orthogonal to it
C = np.array([1.0, -0.5])          # the shift of the flow (A @ C = 1, A_PERP @ C = 0.5)
E0 = np.array([1.0, 0.0])


def _unsafe(*rows, shift=C):
    """The half-space whose rows are ``r @ z <= b`` in the latent, written for
    the data ``y = z + shift``: ``r @ y <= b + r @ shift``."""
    G = np.array([r for r, _ in rows], dtype=np.float64)
    g = np.array([b + np.dot(r, shift) for r, b in rows], dtype=np.float64)
    return HalfSpace(G, g.reshape(-1, 1))


def _at_least(direction, b, shift=C):
    """``direction @ z >= b`` in the latent."""
    return _unsafe((-np.asarray(direction, float), -b), shift=shift)


def _at_most(direction, b, shift=C):
    """``direction @ z <= b`` in the latent."""
    return _unsafe((np.asarray(direction, float), b), shift=shift)


def _wedge(b):
    """Two rows that constrain orthogonal directions: ``A @ z >= b`` and
    ``A_PERP @ z <= 0.5``. Gaussian mass ``norm.sf(b) * norm.cdf(0.5)``."""
    return _unsafe((-A, -b), (A_PERP, 0.5))


FAR = _at_least(E0, 40.0)                        # mass below 1e-300
E1 = np.array([0.0, 1.0])
BOX = _unsafe((-E0, -0.25), (E0, 1.75), (-E1, 1.0), (E1, 0.75))
# 0.25 <= z0 <= 1.75 and -1 <= z1 <= 0.75: four rows, Gaussian mass 0.222
HALF_T = float(norm.isf(0.25))                   # z0 >= HALF_T has Gaussian mass 0.25


# --------------------------------------------------------------------------- #
# Mass oracles                                                                 #
# --------------------------------------------------------------------------- #
def _ball_probability(q):
    """P(||z|| <= q) for a 2-D standard normal latent (chi-square, 2 dof)."""
    return -math.expm1(-q * q / 2.0)


def _ball_slab(lo, hi, q):
    """P(lo <= e @ z <= hi | ||z|| <= q) for a unit vector e in 2-D."""
    lo, hi = max(lo, -q), min(hi, q)
    if lo >= hi:
        return 0.0

    def density(x):
        s = math.sqrt(max(q * q - x * x, 0.0))
        return norm.pdf(x) * (2.0 * norm.cdf(s) - 1.0)

    value, _ = quad(density, lo, hi, epsabs=1e-15, epsrel=1e-12, limit=200)
    return value / _ball_probability(q)


def _ball_wedge(b, q):
    """P(A @ z >= b, A_PERP @ z <= 0.5 | ||z|| <= q) in 2-D."""
    def density(x):
        s = math.sqrt(max(q * q - x * x, 0.0))
        return norm.pdf(x) * max(norm.cdf(min(0.5, s)) - norm.cdf(-s), 0.0)

    value, _ = quad(density, b, q, epsabs=1e-15, epsrel=1e-12, limit=200)
    return value / _ball_probability(q)


def _box_mass(q=None):
    """Mass of BOX; with ``q``, conditional on the ball, which holds the box."""
    mass = ((norm.cdf(1.75) - norm.cdf(0.25)) * (norm.cdf(0.75) - norm.cdf(-1.0)))
    if q is None:
        return mass
    assert math.hypot(1.75, 1.0) <= q, "the box must lie inside the ball"
    return mass / _ball_probability(q)


# --------------------------------------------------------------------------- #
# Flows with a known push-forward                                              #
# --------------------------------------------------------------------------- #
class _Shift:
    """Stand-in flow: ``inverse`` maps a latent z to the data point z + shift."""

    def __init__(self, shift=C):
        self.shift = torch.tensor(np.asarray(shift, dtype=np.float32))

    def inverse(self, z, **_ode_options):
        return z + self.shift

    def forward(self, y, **_ode_options):
        return y - self.shift

    def eval(self):
        return self

    def to(self, *_args, **_kwargs):
        return self


class _ConstantVelocity(nn.Module):
    """v(t, y) = c: the library's FlowODE integrates it to y = z + c t."""

    def __init__(self, velocity):
        super().__init__()
        self.register_buffer("velocity",
                             torch.tensor(np.asarray(velocity, np.float32)))

    def forward(self, t, y):
        return self.velocity.expand_as(y)


class _LinearVelocity(nn.Module):
    """v(t, y) = rate * y: Euler with n grid points over [0, t] gives
    y = (1 + rate * t / (n - 1)) ** (n - 1) * z."""

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, t, y):
        return self.rate * y


def _ode_shift():
    return FlowODE(_ConstantVelocity(C))


# --------------------------------------------------------------------------- #
# Checks shared by the cells                                                   #
# --------------------------------------------------------------------------- #
WINDOW = 1e-8      # binomial probability left out on each side of a count


def _binomial_window(n, p):
    """Counts outside [lo, hi] have probability at most 1e-8 on each side."""
    return int(binom.ppf(WINDOW, n, p)), int(binom.isf(WINDOW, n, p))


def _row_margins(halfspace, y):
    """``G y - g`` in float64 and the slack of n2v's float32 evaluation of it
    (casting G, g and y and a dot product of length d: at most (d + 3)
    roundings of relative size 2**-24 each, doubled)."""
    G = np.asarray(halfspace.G, np.float64)
    g = np.asarray(halfspace.g, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    margins = G @ y - g
    slack = (G.shape[1] + 3) * 2.0 ** -23 * (np.abs(G) @ np.abs(y) + np.abs(g))
    return margins, slack


def _assert_witness_in_unsafe_set(result, members, label):
    """``worst_y`` of a detection satisfies every row of some member."""
    assert result.final_phi <= 0.0, (
        f"{label}: detected, but final_phi = {result.final_phi} > 0")
    inside = []
    for hs in members:
        margins, slack = _row_margins(hs, result.worst_y)
        inside.append(bool(np.all(margins <= slack)))
    assert any(inside), (
        f"{label}: the witness {result.worst_y} lies in no member of the "
        f"unsafe set (row margins {[_row_margins(h, result.worst_y)[0] for h in members]})")


def _clt_upper(pi_hat, levels, quantile, beta, n):
    return min(1.0, pi_hat * math.exp(
        norm.isf(beta) * math.sqrt(levels * (1.0 - quantile) / (quantile * n))))


def _assert_documented_formulas(r, *, n, quantile, beta, max_levels, label):
    """The returned numbers follow the documented estimator.

    A detection with k samples in U at the last of L levels: pi_hat =
    quantile ** (L - 1) * k / N (the docstring of
    ``amls_estimate_halfspace_mass``); at the first level pi_upper is the
    Clopper-Pearson bound beta.ppf(1 - beta, k + 1, N - k) (1 when k = N);
    after splitting levels it is the CLT bound pi_hat * exp(z_(1-beta) *
    sqrt(L (1 - rho) / (rho N))) (module docstring of ``amls_bounded``). A run
    with no sample in U used every level: L = max_levels and pi_hat =
    quantile ** max_levels, with the same CLT bound. ``k / N`` is a float32
    mean in the library: relative rounding at most 2**-24, so 2**-22 here.
    """
    k, levels = r.final_unsafe_count, r.levels_used
    if k > 0:
        expected = quantile ** (levels - 1) * k / n
        assert r.pi_hat == pytest.approx(expected, rel=2.0 ** -22, abs=0.0), (
            f"{label}: pi_hat {r.pi_hat} is not quantile**(L-1) * k/N = "
            f"{expected} (L = {levels}, k = {k}, N = {n})")
    else:
        assert levels == max_levels, (
            f"{label}: no sample in U, but {levels} levels of {max_levels}")
        assert r.pi_hat == pytest.approx(quantile ** max_levels, rel=1e-12,
                                         abs=0.0), (
            f"{label}: exhausted run reports pi_hat {r.pi_hat}, not "
            f"quantile**max_levels = {quantile ** max_levels}")
    if k > 0 and levels == 1:
        expected = 1.0 if k == n else float(
            beta_distribution.ppf(1.0 - beta, k + 1, n - k))
    else:
        expected = _clt_upper(r.pi_hat, levels, quantile, beta, n)
    assert r.pi_upper == pytest.approx(expected, rel=1e-12, abs=0.0), (
        f"{label}: pi_upper {r.pi_upper} is not the documented bound "
        f"{expected} (L = {levels}, k = {k}, N = {n}, beta = {beta})")


def _same_run(a, b, label):
    """Two results of the same run, field by field."""
    for field in ("pi_hat", "pi_upper", "levels_used", "final_unsafe_count",
                  "detected_unsafe", "final_phi", "adaptive_step_used"):
        if hasattr(a, field) or hasattr(b, field):
            assert getattr(a, field) == getattr(b, field), (
                f"{label}: {field} differs: {getattr(a, field)} vs "
                f"{getattr(b, field)}")
    assert np.array_equal(a.worst_y, b.worst_y), f"{label}: worst_y differs"


def _estimate(kind, flow, unsafe, **options):
    """Run one estimator on the CPU. ``unsafe`` is a half-space or, for the
    union estimator, a list of them."""
    options.setdefault("device", "cpu")
    if kind == "unbounded":
        options.pop("q", None)
        return amls_estimate_halfspace_mass(flow, unsafe, **options)
    if kind == "bounded":
        return amls_bounded_estimate_halfspace_mass(flow, unsafe, **options)
    return amls_bounded_estimate_union_mass(flow, list(unsafe), **options)


@pytest.fixture
def cpu_only(monkeypatch):
    """``verify_specification`` passes no device: make its autodetection
    answer CPU on every host, so that no case of this file runs on a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _flow_set(flow, q):
    """A flow reach set over a 2-D output whose calibrated latent radius is q."""
    return ProbabilisticSet(SimpleNamespace(flow_model=flow), threshold=q,
                            m=1000, ell=999, epsilon=0.01, dim=2)


# --------------------------------------------------------------------------- #
# 1. The first level: an exact binomial count                                  #
# --------------------------------------------------------------------------- #
_FIRST_LEVEL = [
    # id, estimator, unsafe set (or group), its mass, q
    ("unbounded-rotated", "unbounded", _at_least(A, 0.75), norm.sf(0.75), None),
    ("unbounded-box", "unbounded", BOX, _box_mass(), None),
    ("bounded-rotated-cut-by-ball", "bounded", _at_least(A, 0.75),
     _ball_slab(0.75, math.inf, 2.5), 2.5),
    ("bounded-box-inside-ball", "bounded", BOX, _box_mass(2.5), 2.5),
    ("union-two-sides", "union", [_at_least(E0, 1.0), _at_most(E0, -1.25)],
     _ball_slab(1.0, math.inf, 2.5) + _ball_slab(-math.inf, -1.25, 2.5), 2.5),
    ("union-box-and-rotated", "union", [BOX, _at_least(A, 2.0)],
     None, 2.5),
]


def _union_box_rotated_mass(q):
    """P(z in BOX or A @ z >= 2 | ||z|| <= q): the two sets are disjoint
    (A @ z <= 0.6 * 1.75 + 0.8 * 1 = 1.85 on the box)."""
    return _box_mass(q) + _ball_slab(2.0, math.inf, q)


@pytest.mark.parametrize("label,kind,unsafe,mass,q", _FIRST_LEVEL,
                         ids=[c[0] for c in _FIRST_LEVEL])
def test_first_level_count_is_binomial_in_the_true_mass(label, kind, unsafe,
                                                         mass, q):
    """A set of mass 0.1 to 0.3 is reached by the first sample: the count of
    samples in U is Binomial(N, p), pi_hat = k / N, pi_upper is the
    Clopper-Pearson bound of k and holds the true mass, and the witness is a
    point of U. The flow is the library's FlowODE (constant velocity, so the
    data are the latent shifted by C).

    Catches: the level function over the wrong connective (minimum over the
    rows of a box: every sample counts; maximum over the members of a union:
    none counts); the flow run in the wrong direction (forward in place of
    inverse moves the samples by -C); the first-level Clopper-Pearson bound
    with a wrong parameter; a bounded sample drawn outside the ball.
    A correct implementation fails with probability at most 2e-8 (the window)
    plus 1e-7 (the coverage of the bound at beta = 1e-7).
    """
    if mass is None:
        mass = _union_box_rotated_mass(q)
    n, beta = 400, 1e-7
    members = unsafe if kind == "union" else [unsafe]
    r = _estimate(kind, _ode_shift(), unsafe, q=q, n_samples_per_level=n,
                  beta=beta, seed=11)
    lo, hi = _binomial_window(n, mass)
    k = r.final_unsafe_count
    assert r.detected_unsafe and r.levels_used == 1 and lo <= k <= hi, (
        f"{label}: first-level count {k} (levels {r.levels_used}, detected "
        f"{r.detected_unsafe}) outside the binomial window [{lo}, {hi}] of "
        f"N = {n}, p = {mass:.6f}")
    assert r.pi_hat == k / n, f"{label}: pi_hat {r.pi_hat} is not k/N = {k / n}"
    _assert_documented_formulas(r, n=n, quantile=0.1, beta=beta, max_levels=30,
                                label=label)
    assert r.pi_upper >= mass, (
        f"{label}: the (1 - 1e-7) bound {r.pi_upper} is below the true mass "
        f"{mass}")
    _assert_witness_in_unsafe_set(r, members, label)


@pytest.mark.parametrize("kind", ["unbounded", "bounded", "union"])
def test_a_set_that_holds_every_sample_has_mass_one(kind):
    """A half-space that holds the whole ball (``A @ z >= -2.6`` with q = 2.5;
    for the unbounded estimator ``A @ z >= -40``) holds every sample: the
    count is N, pi_hat = 1 and pi_upper = 1 exactly, at the first level.
    Catches: the branch for a full count (the Clopper-Pearson formula has no
    second parameter there), a count or a fraction off by one sample. Exact.
    """
    n = 60
    unsafe = _at_least(A, -40.0 if kind == "unbounded" else -2.6)
    r = _estimate(kind, _Shift(), [unsafe] if kind == "union" else unsafe,
                  q=2.5, n_samples_per_level=n, seed=3)
    assert (r.levels_used, r.final_unsafe_count, r.pi_hat, r.pi_upper) == \
        (1, n, 1.0, 1.0), (
        f"{kind}: levels {r.levels_used}, count {r.final_unsafe_count}, "
        f"pi_hat {r.pi_hat}, pi_upper {r.pi_upper} for a set of mass 1")


# --------------------------------------------------------------------------- #
# 2. Certificates on sets whose answer needs no statistics                     #
# --------------------------------------------------------------------------- #
HALF_GROUP = [_at_least(E0, HALF_T), _at_most(E0, -HALF_T), FAR]


def _certify(kind, spec_groups, **options):
    options.setdefault("device", "cpu")
    if kind == "amls":
        options.pop("q", None)
        options.pop("eps_2_target", None)
        return amls_certify_spec(_Shift(), spec_groups, **options)
    fn = (amls_bounded_certify_spec if kind == "amls_bounded"
          else amls_bounded_certify_spec_union)
    return fn(_Shift(), spec_groups, **options)


CERTIFY_KINDS = ["amls", "amls_bounded", "amls_bounded_union"]


@pytest.mark.parametrize("kind", CERTIFY_KINDS)
def test_a_group_holding_half_the_mass_is_never_certified(kind):
    """One group, the union of ``z0 >= 0.674`` and ``z0 <= -0.674`` (Gaussian
    mass 0.25 each, at least 0.2 each in the ball q = 2.5) and a half-space no
    sample reaches. Its mass is at least 0.4, so it must not be certified at
    a target of 1e-3, for any seed.

    Catches: a group certified when one member passes (``any`` for ``all``),
    a group judged by its last member only, a union read as the intersection
    of its members (empty here). A correct implementation can certify only if
    no first-level sample hits a member of mass 0.2: probability 0.8 ** 100 =
    2e-10 per seed.
    """
    for seed in range(3):
        r = _certify(kind, [HALF_GROUP], q=2.5, eps_2_target=1e-3,
                     n_samples_per_level=100, seed=seed)
        assert not r.unsat_certified and r.detected_any, (
            f"{kind}, seed {seed}: a group of mass >= 0.4 is certified "
            f"(unsat_certified={r.unsat_certified}, detected_any="
            f"{r.detected_any})")


@pytest.mark.parametrize("method", CERTIFY_KINDS)
def test_verify_specification_never_answers_unsat_for_half_the_mass(method,
                                                                    cpu_only):
    """The public route of the case above: ``verify_specification`` on a flow
    reach set (q = 2.5) with the group as one OR of half-spaces. The verdict
    must not be UNSAT. Catches the same group and union faults, and a route
    that loses a member of the group or the threshold q. Failure probability
    of a correct implementation: 2e-10 per seed."""
    for seed in range(2):
        config = ProbVerifyConfig(method=method, n_samples=100, seed=seed,
                                  amls_bounded_eps_2_target=1e-3)
        result = verify_specification(_flow_set(_Shift(), 2.5),
                                      {"Hg": list(HALF_GROUP)}, config=config)
        assert result.verdict != "UNSAT", (
            f"{method}, seed {seed}: UNSAT for an unsafe group of mass >= 0.4")


@pytest.mark.parametrize("kind", ["amls_bounded", "amls_bounded_union"])
@pytest.mark.parametrize("order", ["reachable-group-first",
                                   "reachable-group-last"])
def test_a_spec_with_one_group_out_of_reach_is_certified_in_either_order(
        kind, order):
    """Two groups (AND across groups): one that no latent of the ball reaches
    (``A @ z >= 4`` with q = 2: distance 2 from the ball), one of mass about
    0.4. The spec is safe as soon as one group is: certified, in either order
    of the groups.

    Catches: a verdict taken from the last group only, or from the first
    only; a spec judged by a group that is not certified; a chain that
    leaves the ball. The unreachable group's run exhausts 30 levels: its
    bound, 0.1 ** 30 times the CLT factor, is below the target of 1e-3. A
    correct implementation fails only if a chain confined to the ball reaches
    a set the ball does not meet: never. (``amls_certify_spec`` has no bound
    to certify with; its case is the next one.)
    """
    groups = [[_at_least(A, 4.0)], list(HALF_GROUP[:2])]
    if order == "reachable-group-first":
        groups = groups[::-1]
    r = _certify(kind, groups, q=2.0, eps_2_target=1e-3,
                 n_samples_per_level=100, seed=4)
    assert r.unsat_certified, (
        f"{kind}, {order}: a spec with a group out of reach is not certified "
        f"(results {[[(x.detected_unsafe, x.pi_upper) for x in g] for g in r.per_hs_results]})")


@pytest.mark.parametrize("order", ["reachable-group-first",
                                   "reachable-group-last"])
def test_amls_reports_the_detection_of_each_group_in_order(order):
    """``amls_certify_spec`` is a detector: what it must get right is which
    half-space of which group it detected, in the order of the spec. Two
    groups: ``FAR`` (mass below 1e-300, never reached) and the union of
    ``z0 >= 0.674`` and ``z0 <= -0.674`` (mass 0.25 each). The result has one
    list per group with one run per member, FAR is not detected and both
    members of the other group are, in either order. Whether a spec with an
    undetected group may be called certified is registry PROB-8, which this
    case leaves alone.

    Catches: results filed under the wrong group or member, a group's list
    that keeps only its last member, and a chain that drifts outward to FAR
    (a negated Metropolis-Hastings ratio). A correct implementation fails
    only if 100 first-level samples miss a set of mass 0.25 (0.75 ** 100 =
    3e-13 per member) or reach FAR (never)."""
    groups = [[FAR], list(HALF_GROUP[:2])]
    if order == "reachable-group-first":
        groups = groups[::-1]
    r = amls_certify_spec(_Shift(), groups, n_samples_per_level=100, seed=4,
                          device="cpu")
    expected = [[hs is not FAR for hs in group] for group in groups]
    detected = [[x.detected_unsafe for x in g] for g in r.per_hs_results]
    assert detected == expected and r.detected_any, (
        f"{order}: detections {detected}, expected {expected} "
        f"(detected_any {r.detected_any})")


@pytest.mark.parametrize("method", ["amls_bounded", "amls_bounded_union"])
def test_verify_specification_certifies_a_group_out_of_reach_of_the_ball(
        method, cpu_only):
    """Public route of the bounded certificates: a flow reach set with
    threshold q = 2, a spec of two groups (AND), one of them ``A @ z >= 4``.
    That group has mass 0 in the ball, so the answer is UNSAT with
    epsilon_total = 1 - (1 - epsilon)(1 - target). The unbounded mass of the
    group is 3.2e-5, which a run that ignored q would detect (UNKNOWN).
    Catches: the threshold q not handed to the bounded estimators, a group
    verdict that needs every group."""
    target = 1e-3
    pset = _flow_set(_Shift(), 2.0)
    config = ProbVerifyConfig(method=method, n_samples=100, seed=1,
                              amls_bounded_eps_2_target=target)
    result = verify_specification(
        pset, [{"Hg": [_at_least(A, 4.0)]}, {"Hg": list(HALF_GROUP[:2])}],
        config=config)
    assert result.verdict == "UNSAT", f"{method}: {result.verdict}"
    expected = 1.0 - (1.0 - pset.epsilon) * (1.0 - target)
    assert result.epsilon_total == pytest.approx(expected, rel=1e-12), (
        f"{method}: epsilon_total {result.epsilon_total}, expected {expected}")


@pytest.mark.parametrize("kind", ["amls_bounded", "amls_bounded_union"])
@pytest.mark.parametrize("case", ["bound-above-target", "detected"])
def test_the_bounded_gate_needs_both_no_detection_and_a_bound_below_target(
        kind, case):
    """The documented gate of a bounded certificate is "not detected AND
    pi_upper <= target". Two exact cases, each failing one conjunct:

    * bound-above-target: a set no sample reaches, one level only
      (max_levels = 1): nothing is detected and the point estimate 0.1 is
      below the target 0.15, but the bound, 0.1 times the CLT factor (0.25
      with N = 100), is above it;
    * detected: a set of mass >= 0.2 at a loose target 0.99: the first level
      detects it and its Clopper-Pearson bound (about 0.4) is below the target.

    Catches a gate that drops either conjunct, and a gate on the point
    estimate in place of the bound. Exact for the first case; the
    second fails a correct implementation only if no first-level sample hits a
    set of mass 0.2: 0.8 ** 100 = 2e-10.
    """
    if case == "bound-above-target":
        groups, target, options = [[FAR]], 0.15, dict(max_levels=1)
    else:
        groups, target, options = [[HALF_GROUP[0]]], 0.99, {}
    r = _certify(kind, groups, q=2.5, eps_2_target=target,
                 n_samples_per_level=100, seed=2, **options)
    (only,), = r.per_hs_results
    if case == "bound-above-target":
        assert (not only.detected_unsafe
                and only.pi_hat <= target < only.pi_upper), (
            f"setup: detected {only.detected_unsafe}, pi_hat {only.pi_hat}, "
            f"pi_upper {only.pi_upper}")
    else:
        assert only.detected_unsafe and only.pi_upper <= target, (
            f"setup: detected {only.detected_unsafe}, pi_upper {only.pi_upper}")
    assert not r.unsat_certified, (
        f"{kind}, {case}: certified with detected={only.detected_unsafe}, "
        f"pi_upper={only.pi_upper}, target={target}")


@pytest.mark.parametrize("kind", ["amls_bounded", "amls_bounded_union"])
def test_eps_2_upper_is_the_largest_upper_bound_of_the_runs(kind):
    """Two groups, each a set of mass about 0.2 that the first level reaches.
    ``eps_2_upper`` is documented as the largest pi_upper of the runs: it must
    equal the largest Clopper-Pearson bound computed here from each run's
    count (scipy), and it holds the true mass of each group (beta = 1e-7).
    Catches a running maximum of the point estimates in place of the bounds
    (the stated upper bound then lies below the confidence bound). A correct
    implementation fails the coverage with probability at most 2e-7."""
    n, beta = 100, 1e-7
    groups = [[HALF_GROUP[0]], [HALF_GROUP[1]]]
    r = _certify(kind, groups, q=2.5, eps_2_target=1e-3,
                 n_samples_per_level=n, beta=beta, seed=6)
    bounds = []
    for (res,) in r.per_hs_results:
        k = res.final_unsafe_count
        assert res.levels_used == 1 and 0 < k < n, (
            f"setup: levels {res.levels_used}, count {k}")
        bounds.append(float(beta_distribution.ppf(1.0 - beta, k + 1, n - k)))
    assert r.eps_2_upper == pytest.approx(max(bounds), rel=1e-12, abs=0.0), (
        f"{kind}: eps_2_upper {r.eps_2_upper}, largest bound {max(bounds)}")
    mass = _ball_slab(HALF_T, math.inf, 2.5)
    assert r.eps_2_upper >= mass, (
        f"{kind}: eps_2_upper {r.eps_2_upper} below the mass {mass}")


# --------------------------------------------------------------------------- #
# 3. Options, seeds and refusals                                               #
# --------------------------------------------------------------------------- #
def _linear_flow(rate):
    return FlowODE(_LinearVelocity(rate))


@pytest.mark.parametrize("kind", ["unbounded", "bounded", "union"])
def test_ode_options_reach_the_flow(kind):
    """``t``, ``n_ode_steps`` and ``ode_method`` are handed to the flow: with
    the velocity 4 y, Euler over [0, 0.5] on a grid of 2 points gives
    y = (1 + 4 * 0.5) z = 3 z, so U = {y0 >= 6} is z0 >= 2 (Gaussian mass
    0.0228, mean count 34 of 1500; in the ball q = 2.5 mass 0.0124, mean 19).
    A flow left at one of its defaults moves the data elsewhere: 100 Euler
    points give 7.24 z, RK4 7.0 z, dopri5 7.39 z, t = 1 gives 5 z, so the mean
    count is 155 to 313, far above the binomial window ([7, 71] and [1, 47]).
    Catches an ODE option that is not forwarded. Failure probability of a
    correct implementation: 2e-8 (the window) plus 8e-9 (no first-level
    hit in the ball)."""
    n = 1500
    unsafe = _at_least(E0, 6.0, shift=(0.0, 0.0))
    mass = norm.sf(2.0) if kind == "unbounded" else _ball_slab(2.0, math.inf, 2.5)
    r = _estimate(kind, _linear_flow(4.0),
                  [unsafe] if kind == "union" else unsafe,
                  q=2.5, n_samples_per_level=n, t=0.5, n_ode_steps=2,
                  ode_method="euler", seed=8)
    lo, hi = _binomial_window(n, mass)
    assert r.levels_used == 1 and lo <= r.final_unsafe_count <= hi, (
        f"{kind}: count {r.final_unsafe_count} (levels {r.levels_used}) outside "
        f"[{lo}, {hi}] for mass {mass:.4f}")


@pytest.mark.parametrize("kind", ["unbounded", "bounded", "union"])
def test_same_seed_repeats_the_run_and_another_seed_does_not(kind):
    """Documented: same seed and same device give a bit-identical run, also
    after the global numpy and torch generators moved; a different seed gives
    another run. The set (``A @ z >= 0.75``) is reached by the first sample, so
    the smallest level function value differs between seeds with probability
    one. Catches a generator that is not seeded from ``seed`` (every seed the
    same run) and a global generator that is not reseeded (the same seed
    gives two runs)."""
    unsafe = _at_least(A, 0.75)
    arg = [unsafe] if kind == "union" else unsafe

    def run(seed):
        return _estimate(kind, _Shift(), arg, q=2.5, n_samples_per_level=80,
                         seed=seed)

    first = run(5)
    np.random.rand(7)
    torch.rand(7)
    _same_run(first, run(5), f"{kind} seed 5 twice")
    other = run(6)
    assert other.final_phi != first.final_phi, (
        f"{kind}: seeds 5 and 6 give the same run (final_phi {first.final_phi})")


_SEED_NONE = "amls_certify_spec with seed=None hands None to every half-space " \
             "and the estimator seeds its generator with 0 for None, so every " \
             "half-space runs with the same random stream: the runs are not " \
             "decoupled as documented"


@pytest.mark.parametrize("layout,seed", [
    ("one-group", 9),
    ("two-groups", 9),
    pytest.param("one-group", None, marks=known_bug(13, _SEED_NONE)),
], ids=["one-group-seed-9", "two-groups-seed-9", "one-group-seed-None"])
def test_two_copies_of_one_halfspace_run_with_different_seeds(layout, seed):
    """``amls_certify_spec`` documents a different seed per half-space "so
    each AMLS run is decoupled" (the counter runs over the whole spec): two
    copies of one half-space, in one group or in two groups, must give two
    different runs, also when the caller gives no seed. Catches a half-space
    counter that never advances or restarts in every group. The first sample
    reaches the set, so its smallest level function value differs between
    two independent streams with probability one. Today ``seed=None`` gives
    the same run twice."""
    hs = _at_least(A, 0.75)
    spec = [[hs, hs]] if layout == "one-group" else [[hs], [hs]]
    r = amls_certify_spec(_Shift(), spec, n_samples_per_level=80, seed=seed,
                          device="cpu")
    first, second = [x for group in r.per_hs_results for x in group]
    assert first.final_phi != second.final_phi, (
        f"{layout}, seed {seed}: both copies ran with the same random stream")


_ODE_VARIANTS = {
    # Euler with 7 points over [0, 0.75] at velocity 2 y: y = 1.25 ** 6 z.
    "euler": dict(ode_method="euler", n_ode_steps=7, ode_atol=1e-3,
                  ode_rtol=1e-3, scale=1.25 ** 6),
    # dopri5 reads the tolerances; at 1e-2 its result differs from the
    # default 1e-5 in the low bits: y is about exp(1.5) z.
    "dopri5": dict(ode_method="dopri5", n_ode_steps=5, ode_atol=1e-2,
                   ode_rtol=1e-2, scale=math.exp(1.5)),
}


@pytest.mark.parametrize("kind", CERTIFY_KINDS)
@pytest.mark.parametrize("solver", list(_ODE_VARIANTS))
def test_certify_hands_every_option_to_the_estimator(kind, solver):
    """Every option of a certify function reaches the estimator: each run of
    the spec equals the estimator called directly with the same options (the
    bounded functions run every half-space with the caller's seed, the
    documented convention; ``amls_certify_spec`` runs its first half-space
    with the caller's seed). The options are all non-default and the flow is
    FlowODE with the velocity 2 y, so that t, the step count, the method and,
    for dopri5, the tolerances change the data; the set (A @ z >= 3.41, mass
    3.3e-4, inside q = 4) needs splitting levels, so that the chain options
    change the run. Beside it, two independent checks: an exhausted run
    follows the documented formulas with the given quantile, beta,
    max_levels and N, and ``adaptive_step_used`` is set. Catches an option
    that is not forwarded (N, adaptive_step, quantile, beta, the chain and
    ODE options)."""
    variant = dict(_ODE_VARIANTS[solver])
    scale = variant.pop("scale")
    options = dict(n_samples_per_level=120, quantile=0.2, max_levels=3,
                   n_mcmc_steps=4, mcmc_step_size=0.45, beta=0.01, seed=5,
                   t=0.75, device="cpu", **variant)
    origin = (0.0, 0.0)
    rare = _at_least(A, 3.41 * scale, shift=origin)
    far = _at_least(E0, 400.0, shift=origin)
    flow = _linear_flow(2.0)
    if kind == "amls":
        spec = [[rare, far]]
        r = amls_certify_spec(flow, spec, **options)
        direct = [amls_estimate_halfspace_mass(flow, rare, **options)]
        runs = [r.per_hs_results[0][0]]
        exhausted = r.per_hs_results[0][1]
    else:
        bounded = dict(options, q=4.0, adaptive_step=True)
        if kind == "amls_bounded":
            spec = [[rare], [far]]
            r = amls_bounded_certify_spec(flow, spec, eps_2_target=1e-3,
                                          **bounded)
            direct = [amls_bounded_estimate_halfspace_mass(flow, hs, **bounded)
                      for (hs,) in spec]
        else:
            spec = [[rare, far], [far]]
            r = amls_bounded_certify_spec_union(flow, spec, eps_2_target=1e-3,
                                                **bounded)
            direct = [amls_bounded_estimate_union_mass(flow, g, **bounded)
                      for g in spec]
        runs = [res for group in r.per_hs_results for res in group]
        exhausted = runs[-1]
        assert all(res.adaptive_step_used for res in runs), (
            f"{kind}: adaptive_step=True was not used: "
            f"{[res.adaptive_step_used for res in runs]}")
    if runs[0].levels_used == 1:
        pytest.fail(f"setup: {kind}: the first level reached the rare set, so "
                    "the chain options were never used")
    for i, (got, want) in enumerate(zip(runs, direct)):
        _same_run(got, want, f"{kind} run {i}")
    _assert_documented_formulas(exhausted, n=120, quantile=0.2, beta=0.01,
                                max_levels=3, label=f"{kind} exhausted run")


_OUT_OF_RANGE = [
    ("n_samples_per_level", 0), ("quantile", 0.0), ("quantile", 1.0),
    ("max_levels", 0), ("beta", 0.0), ("beta", 1.0),
]
_REFUSALS = [(kind, option, value)
             for kind in ("unbounded", "bounded", "union")
             for option, value in _OUT_OF_RANGE + (
                 [] if kind == "unbounded" else [("q", 0.0)])]


@pytest.mark.parametrize("kind,option,value", _REFUSALS,
                         ids=[f"{k}-{o}={v}" for k, o, v in _REFUSALS])
def test_an_option_outside_its_range_is_refused(kind, option, value):
    """Every estimator refuses with ``ValueError`` an option outside its
    documented range: N and max_levels below 1, quantile and beta outside
    (0, 1), and for the bounded ones a radius q <= 0. Catches a dropped or
    loosened range check (quantile 1 keeps every sample and credits a factor
    1 per level; beta 1 gives a bound at the median)."""
    unsafe = _at_least(A, 0.75)
    options = dict(q=2.0, n_samples_per_level=20, max_levels=2, seed=0)
    options[option] = value
    with pytest.raises(ValueError):
        _estimate(kind, _Shift(), [unsafe] if kind == "union" else unsafe,
                  **options)


def test_union_certificate_refuses_an_empty_group():
    """An empty OR group cannot be read as a set to estimate: ``ValueError``."""
    with pytest.raises(ValueError):
        amls_bounded_certify_spec_union(_Shift(), [[FAR], []], q=2.0,
                                        eps_2_target=1e-3,
                                        n_samples_per_level=20, device="cpu")


@pytest.mark.parametrize("kind", ["bounded", "union"])
def test_adaptive_step_is_the_documented_scaled_step(kind):
    """``adaptive_step=True`` is documented as the step
    ``mcmc_step_size * min(1, q / sqrt(d))``: with d = 4, q = 1 and step 0.3
    the run must equal the run with step 0.15 and no adaptation (apart from
    the flag). The set (``z0 >= 0.95`` in the unit 4-ball) needs splitting
    levels, so the chain runs. Catches the adaptive step computed but not
    used, and a scale q / d in place of q / sqrt(d)."""
    shift = np.array([1.0, -0.5, 0.25, 2.0])
    unsafe = _at_least(np.eye(4)[0], 0.95, shift=shift)
    arg = [unsafe] if kind == "union" else unsafe
    common = dict(q=1.0, n_samples_per_level=50, max_levels=12, seed=13)
    adaptive = _estimate(kind, _Shift(shift), arg, adaptive_step=True,
                         mcmc_step_size=0.3, **common)
    scaled = _estimate(kind, _Shift(shift), arg, adaptive_step=False,
                       mcmc_step_size=0.15, **common)
    if adaptive.levels_used == 1:
        pytest.fail("setup: the first level already reached the set; the "
                    "chain never ran")
    assert adaptive.adaptive_step_used and not scaled.adaptive_step_used
    for field in ("pi_hat", "pi_upper", "levels_used", "final_unsafe_count",
                  "detected_unsafe", "final_phi"):
        assert getattr(adaptive, field) == getattr(scaled, field), (
            f"{kind}: {field} {getattr(adaptive, field)} with adaptive_step, "
            f"{getattr(scaled, field)} with the documented step 0.15")


# --------------------------------------------------------------------------- #
# 4. The splitting levels: rare sets against closed-form masses                #
# --------------------------------------------------------------------------- #
SEEDS = range(20)
N_RARE = 200
# A chain that mixes within a level: 30 steps of 0.8 in place of 10 of 0.3.
# On the unmodified library the spread of log(pi_hat / p) stays at about 1.0
# (60 seeds); a chain that does
# not keep to its level set falls back to the prior within one level, so
# the levels stop progressing while each still credits a factor 0.1.
WELL_MIXED = (("n_mcmc_steps", 30), ("mcmc_step_size", 0.8))

_RARE = {
    # id: (estimator, unsafe set or group, mass, q, chain options)
    "unbounded": ("unbounded", _wedge(4.0), norm.sf(4.0) * norm.cdf(0.5), None,
                  ()),
    "unbounded-well-mixed": ("unbounded", _wedge(4.0),
                             norm.sf(4.0) * norm.cdf(0.5), None, WELL_MIXED),
    "bounded-q6": ("bounded", _wedge(4.0), _ball_wedge(4.0, 6.0), 6.0, ()),
    "bounded-q6-well-mixed": ("bounded", _wedge(4.0), _ball_wedge(4.0, 6.0),
                              6.0, WELL_MIXED),
    "bounded-q4-beyond-sqrt-q": ("bounded", _wedge(3.7), _ball_wedge(3.7, 4.0),
                                 4.0, ()),
    "union-q6-rare-member-second": ("union", [FAR, _wedge(4.0)],
                                    _ball_wedge(4.0, 6.0), 6.0, ()),
    "union-q6-well-mixed": ("union", [FAR, _wedge(4.0)], _ball_wedge(4.0, 6.0),
                            6.0, WELL_MIXED),
}


# The well-mixed configurations differ from the others only by the chain
# options, which change no code path of the witness, the formulas or the
# average: they serve the bound case alone.
_RARE_DEFAULT_CHAIN = [c for c, entry in _RARE.items() if not entry[4]]


@lru_cache(maxsize=None)
def _rare_runs(config):
    kind, unsafe, mass, q, chain = _RARE[config]
    return [_estimate(kind, _Shift(), unsafe, q=q, n_samples_per_level=N_RARE,
                      seed=seed, **dict(chain)) for seed in SEEDS]


@pytest.mark.parametrize("config", list(_RARE))
def test_bound_of_a_rare_set_is_not_far_below_its_mass(config):
    """A rare set (mass 2.2e-5; for q = 4 a set between radius 2 = sqrt(q) and
    q, mass 4.2e-5) is reached only through splitting levels. In every one of
    20 seeds pi_upper must be at least 1e-4 times the true mass.

    Catches the faults that make the bound collapse while the formulas stay
    consistent: an elite set of the samples farthest from U, an elite set of
    90 % of the samples while each level still counts 0.1, the ball test
    against q in place of q squared (the chain stays in radius sqrt(q)), the
    flow run backwards, the union chain scored on its first member only; and,
    on the well-mixed configurations, a chain that does not keep to its level
    set or a level set at the wrong quantile (the levels stop progressing and
    the bound falls to 1e-25 times the mass). A
    correct implementation: log(pi_upper / p) >= log(pi_hat / p), mean near 0,
    standard deviation at most 1.2, so a ratio of 1e-4 is 7.7 standard
    deviations away: below 1e-13 per run.
    """
    kind, unsafe, mass, q, _chain = _RARE[config]
    low = [(seed, r.levels_used, r.pi_upper / mass)
           for seed, r in zip(SEEDS, _rare_runs(config))
           if not r.pi_upper >= 1e-4 * mass]
    assert not low, (
        f"{config}: pi_upper below 1e-4 times the true mass {mass:.3e} in "
        f"(seed, levels, ratio) {low}")


@pytest.mark.parametrize("config", _RARE_DEFAULT_CHAIN)
def test_estimate_of_a_rare_set_is_within_a_factor_ten_on_average(config):
    """Over the runs that needed splitting levels (at least 15 of 20: a
    first-level hit has probability at most 0.9 % per seed, so more than 5
    hits in 20 seeds has probability below 1.2e-8), the geometric mean of
    pi_hat / p lies in [0.1, 10]. Catches a chain that drifts away from the
    Gaussian target (a negated Metropolis-Hastings ratio overestimates about
    70-fold) and every fault that collapses the estimate. A correct
    implementation: the mean of at least 15 logs has standard deviation at
    most 0.31 and a bias of at most 0.72 (the log of an unbiased estimate),
    so log 10 is 5 standard deviations away: below 1e-6."""
    kind, unsafe, mass, q, _chain = _RARE[config]
    logs = [math.log(max(r.pi_hat, 1e-300) / mass)
            for r in _rare_runs(config) if r.levels_used > 1]
    assert len(logs) >= 15, f"{config}: only {len(logs)} runs used levels"
    ratio = math.exp(sum(logs) / len(logs))
    assert 0.1 <= ratio <= 10.0, (
        f"{config}: geometric mean of pi_hat / p is {ratio:.3g} over "
        f"{len(logs)} runs")


@pytest.mark.parametrize("config", _RARE_DEFAULT_CHAIN)
def test_witness_of_a_rare_detection_lies_in_the_unsafe_set(config):
    """Every run that detects the rare set returns a witness ``worst_y`` that
    satisfies every row of a member and ``final_phi <= 0``. Catches a running
    best that keeps the largest level function value in place of the
    smallest, and a witness taken from the wrong member. Exact up to the
    float32 slack of the library's own row evaluation."""
    kind, unsafe, mass, q, _chain = _RARE[config]
    members = unsafe if kind == "union" else [unsafe]
    detected = 0
    for seed, r in zip(SEEDS, _rare_runs(config)):
        if r.final_unsafe_count > 0:
            detected += 1
            _assert_witness_in_unsafe_set(r, members, f"{config} seed {seed}")
    assert detected >= 15, f"{config}: only {detected} detections in 20 seeds"


def test_a_run_that_saw_a_point_of_u_reports_it_detected():
    """``detected_unsafe`` is documented as "at least one sample landed in U at
    any point during the run" and ``final_phi`` as the smallest level
    function value seen, "non-positive iff detected_unsafe". The unbounded
    estimator has no ball, so a proposal in U is a point of the flow's image
    in U, accepted or not. With two levels only (max_levels = 2) on the rare
    wedge, about half of 20 runs end their levels with no accepted sample in
    U although a proposal landed there. In every run detected_unsafe must be
    final_phi <= 0, and the witness of every such run is a point of U.

    Catches a run that ends its levels and drops what it saw (not detected
    with final_phi <= 0): ``amls_certify_spec`` then counts the group as
    undetected. Exact up to the float32 slack of the row evaluation."""
    unsafe = _wedge(4.0)
    runs = [_estimate("unbounded", _Shift(), unsafe, n_samples_per_level=N_RARE,
                      max_levels=2, seed=seed) for seed in SEEDS]
    seen = [(seed, r) for seed, r in zip(SEEDS, runs)
            if r.final_unsafe_count == 0 and r.final_phi <= 0.0]
    if not seen:
        pytest.fail("setup: no run ended its levels with a point of U seen "
                    "but not accepted")
    for seed, r in zip(SEEDS, runs):
        assert r.detected_unsafe == (r.final_phi <= 0.0), (
            f"seed {seed}: detected_unsafe {r.detected_unsafe} with final_phi "
            f"{r.final_phi} (levels {r.levels_used}, count "
            f"{r.final_unsafe_count})")
    for seed, r in seen:
        _assert_witness_in_unsafe_set(r, [unsafe], f"seed {seed}")


_EXHAUSTED = {
    # id: (estimator, unsafe, q): a set no sample reaches, 4 levels, quantile 0.2
    "unbounded-exhausted": ("unbounded", _at_least(A, 40.0), None),
    "bounded-exhausted": ("bounded", _at_least(A, 40.0), 2.0),
    "union-exhausted": ("union", [_at_least(A, 40.0), FAR], 2.0),
}


@pytest.mark.parametrize("config", _RARE_DEFAULT_CHAIN + list(_EXHAUSTED))
def test_returned_estimate_and_bound_follow_the_documented_formulas(config):
    """pi_hat is quantile ** (L - 1) * k / N for a detection at level L with k
    samples in U, and quantile ** max_levels after max_levels levels without
    one; pi_upper is the Clopper-Pearson bound at the first level and the CLT
    bound pi_hat * exp(z_(1-beta) sqrt(L (1 - rho) / (rho N))) otherwise.
    The rare sets run with the defaults; the exhausted ones with 4 levels,
    quantile 0.2, beta 0.01 and N = 150, so that a level budget one short, a
    level factor too many or too few, a normal quantile at beta in place of
    1 - beta and a variance in place of a standard deviation all change a
    returned number. Exact up to floating-point rounding."""
    if config in _RARE:
        runs = list(zip(SEEDS, _rare_runs(config)))
        settings = dict(n=N_RARE, quantile=0.1, beta=0.001, max_levels=30)
    else:
        kind, unsafe, q = _EXHAUSTED[config]
        settings = dict(n=150, quantile=0.2, beta=0.01, max_levels=4)
        r = _estimate(kind, _Shift(), unsafe, q=q,
                      n_samples_per_level=settings["n"],
                      quantile=settings["quantile"], beta=settings["beta"],
                      max_levels=settings["max_levels"], seed=21)
        assert not r.detected_unsafe and r.final_unsafe_count == 0
        runs = [(21, r)]
    for seed, r in runs:
        _assert_documented_formulas(r, label=f"{config} seed {seed}",
                                    **settings)


# --------------------------------------------------------------------------- #
# 5. Defects of today's library                                                #
# --------------------------------------------------------------------------- #
_STALL = "a chain that cannot move (n_mcmc_steps=0) still credits a factor " \
         "quantile per level, so a run that never left its first sample " \
         "reports a bound near quantile ** max_levels (1.6e-28) and is " \
         "certified for a set of mass 1.3e-3"


def _stall_outcome(route, steps, seed):
    """One run on U = {A @ z >= 3} (Gaussian mass 1.35e-3; 1.32e-3 in the
    ball q = 4), N = 100, beta = 1e-7. Returns (certified or bound, mass)."""
    unsafe = _at_least(A, 3.0)
    if route == "estimator":
        r = amls_estimate_halfspace_mass(_Shift(), unsafe, n_samples_per_level=100,
                                         n_mcmc_steps=steps, beta=1e-7,
                                         seed=seed, device="cpu")
        return r.pi_upper, norm.sf(3.0)
    config = ProbVerifyConfig(method=route, n_samples=100, beta=1e-7,
                              amls_n_mcmc_steps=steps, seed=seed,
                              amls_bounded_eps_2_target=1e-4)
    result = verify_specification(_flow_set(_Shift(), 4.0), unsafe,
                                  config=config)
    return result.verdict, _ball_slab(3.0, math.inf, 4.0)


@pytest.mark.parametrize("route,steps", [
    ("estimator", 10),
    pytest.param("estimator", 0, marks=known_bug(13, _STALL)),
    ("amls_bounded", 10),
    pytest.param("amls_bounded", 0, marks=known_bug(13, _STALL)),
    ("amls_bounded_union", 10),
    pytest.param("amls_bounded_union", 0, marks=known_bug(13, _STALL)),
])
def test_a_chain_that_cannot_move_is_not_read_as_a_tiny_mass(route, steps,
                                                             cpu_only):
    """The splitting levels estimate each conditional probability by moving
    the kept samples with the Markov chain. With ``n_mcmc_steps = 0`` no sample
    ever moves: the run cannot estimate anything below its first level. The
    library must then refuse (``ValueError``) or answer without reading the
    stall as a tiny mass: the unbounded bound stays above 1e-4 times the
    true mass 1.35e-3, and a bounded certificate at the target 1e-4 is not
    given (verdict not UNSAT). The neighbour with the default 10 steps keeps
    both. Five seeds; with N = 100 the first level misses the set in 88 % of
    them.

    A correct implementation fails: the certificate needs a bound below the
    target, i.e. below the true mass, which beta = 1e-7 allows with
    probability 1e-7 per seed; the unbounded bound is 7.7 standard deviations
    from 1e-4 times the mass (see the rare-set cases). Today seed 0 already
    fails: pi_upper 5e-27 for the estimator and UNSAT for both bounded
    methods (UNSAT in 9 of 10 seeds at beta = 0.001)."""
    for seed in range(5):
        try:
            outcome, mass = _stall_outcome(route, steps, seed)
        except ValueError:
            return  # refusing a chain that cannot move keeps the promise
        if route == "estimator":
            assert outcome >= 1e-4 * mass, (
                f"seed {seed}: pi_upper {outcome:.3e} for a set of mass "
                f"{mass:.3e} (n_mcmc_steps={steps})")
        else:
            assert outcome != "UNSAT", (
                f"{route}, seed {seed}: UNSAT at target 1e-4 for a set of "
                f"mass {mass:.3e} in the ball (n_mcmc_steps={steps})")


_CLT = "the bound after splitting levels uses the variance K(1-rho)/(rho N) " \
       "of independent sampling at every level, while the measured spread " \
       "of log(pi_hat / p) is several times larger (0.86 against 0.16 at " \
       "N = 1000), so the stated (1 - beta) bound lies below the true mass " \
       "in 8 to 10 % of runs at N = 200 and 20 to 35 % at the default N, " \
       "at beta = 0.001"


@pytest.mark.parametrize("regime,kind", [
    ("first-level", "bounded"),
    pytest.param("splitting", "unbounded", marks=known_bug(13, _CLT)),
    pytest.param("splitting", "bounded", marks=known_bug(13, _CLT)),
    pytest.param("splitting", "union", marks=known_bug(13, _CLT)),
])
def test_upper_bound_holds_the_true_mass_at_the_stated_confidence(regime,
                                                                  kind):
    """pi_upper is documented as a (1 - beta) upper bound (for the bounded
    estimators it is the mass bound of the certificate and of
    ``eps_2_upper``). Over 100 seeds at the default beta = 0.001, the number
    of runs whose bound lies below the true mass is Binomial(100, <= 0.001)
    for a correct bound: 5 or more has probability below 1e-6.

    first-level: a set of mass 0.05 (A @ z >= 1.645 in the ball q = 6),
    reached by the first sample of 200 (a miss has probability 3.5e-5): the
    Clopper-Pearson bound keeps its confidence. splitting: the rare set of
    mass 2.2e-5 of the cells above: today 8 to 10 of 100 runs miss (and 20
    to 35 % of runs at the default N of 1000 and 2000)."""
    if regime == "first-level":
        unsafe, mass = _at_least(A, 1.645), _ball_slab(1.645, math.inf, 6.0)
    else:
        unsafe = _wedge(4.0)
        mass = (norm.sf(4.0) * norm.cdf(0.5) if kind == "unbounded"
                else _ball_wedge(4.0, 6.0))
    arg = [FAR, unsafe] if kind == "union" else unsafe
    threshold = next(m for m in range(101) if binom.sf(m - 1, 100, 0.001) <= 1e-6)
    misses = [seed for seed in range(100)
              if _estimate(kind, _Shift(), arg, q=6.0, n_samples_per_level=200,
                           seed=seed).pi_upper < mass]
    assert len(misses) < threshold, (
        f"{regime}/{kind}: the bound is below the true mass {mass:.3e} in "
        f"{len(misses)} of 100 runs at beta = 0.001 (seeds {misses[:10]})")


_OUT_OF_BALL = "the bounded estimators keep the best level function value " \
               "over every proposal, also those rejected for leaving the " \
               "ball, so a set no latent of the ball reaches is reported " \
               "detected, with a witness outside the image of the ball " \
               "(a false UNKNOWN)"


@pytest.mark.parametrize("distance,kind", [
    (2.0, "bounded"),
    (2.0, "union"),
    pytest.param(0.2, "bounded", marks=known_bug(13, _OUT_OF_BALL)),
    pytest.param(0.2, "union", marks=known_bug(13, _OUT_OF_BALL)),
])
def test_a_set_the_ball_cannot_reach_is_not_detected(distance, kind):
    """The bounded estimators estimate the mass under the latent restricted to
    ``||z|| <= q``: a set at a positive distance from the ball has mass 0 and
    no sample of the run can land in it, so ``detected_unsafe`` must be False
    (a detection blocks the certificate). q = 2; the set is
    ``A @ z >= 2 + distance``. At distance 2 (the neighbour) no proposal of
    step 0.3 gets there; at distance 0.2 proposals that leave the ball do.
    Catches, at distance 2, a chain that accepts proposals outside the ball.
    Exact."""
    unsafe = _at_least(A, 2.0 + distance)
    arg = [unsafe] if kind == "union" else unsafe
    for seed in range(5):
        r = _estimate(kind, _Shift(), arg, q=2.0, n_samples_per_level=100,
                      max_levels=10, seed=seed)
        assert not r.detected_unsafe, (
            f"{kind}, seed {seed}: detected a set at distance {distance} from "
            f"the ball; witness {r.worst_y}, latent norm "
            f"{np.linalg.norm(r.worst_y - C):.3f} > q = 2")
