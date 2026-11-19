"""Soundness cells of the flow lane of probabilistic reachability.

Covers ``n2v/probabilistic/flow/sets.py`` (``AffineTransform``,
``ProbabilisticSet``), ``n2v/probabilistic/flow/reach.py``
(``FlowReachConfig``, ``flow_reach`` and its model forward),
``n2v/probabilistic/flow/calibrate.py`` and
``n2v/probabilistic/flow/sampling.py``.

The promise checked here:

- the set ``{y : score(y) <= q}`` that a calibration of rank ``ell`` over
  ``m`` scores builds states the guarantee that holds: coverage
  ``1 - epsilon`` with the confidence ``P(Bin(m, 1 - epsilon) <= ell - 1)``,
  the probability that the ``ell``-th smallest of ``m`` continuous scores
  lies above the ``1 - epsilon`` quantile;
- ``flow_reach`` builds that set from a calibration sample of the stated
  input box that is held out from training, in the frame of the whitening it
  stores, and states the rank, size and miscoverage it used;
- a half-space written in raw output coordinates is moved into the set's
  whitened frame without changing which points it holds, on the direct call
  and through ``verify_specification``;
- the input samplers draw from the whole stated box and from nothing else
  (calibration data from another region makes the guarantee a statement about
  another input set).

Oracles: the binomial tail in fractions; ranks of exact squared distances;
residuals in fractions on dyadic data; the closed-form coverage of a uniform
score; a recording model that hands back the samples ``flow_reach`` drew and
the outputs it saw; exact membership on dyadic bounds. Every statistical
check fixes its seeds and takes its tolerance from a binomial or beta tail, so
that a correct implementation fails it with probability at most 1e-6.

``flow_reach`` trains a flow; the cases here use a few dozen samples, one or
two epochs and a two-step Euler integration, so they check the wiring of the
calibration and not the quality of the fit (``flow/train.py`` is not covered
here).
"""

from fractions import Fraction
from math import comb
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.stats import beta as beta_dist
from scipy.stats import binom

from n2v.nn import NeuralNetwork
from n2v.probabilistic import FlowReachConfig, flow_reach
from n2v.probabilistic.flow.calibrate import calibrate, compute_guarantee
from n2v.probabilistic.flow.sampling import sample_box, sample_l_inf_ball
from n2v.probabilistic.flow.scores import BallScore
from n2v.probabilistic.flow.sets import AffineTransform, ProbabilisticSet
from n2v.sets import Box, HalfSpace
from n2v.utils.verify_specification import ProbVerifyConfig, verify_specification
from tests.known_bugs import known_bug


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #
def _exact_confidence(m, ell, epsilon):
    """P(Bin(m, 1 - epsilon) <= ell - 1) in fractions, for the float epsilon.

    The ell-th smallest of m i.i.d. continuous scores is the ell-th order
    statistic U_(ell) of m uniforms after the probability transform; the set
    covers more than 1 - epsilon exactly when U_(ell) > 1 - epsilon, which
    happens when at most ell - 1 of the m uniforms lie below 1 - epsilon. The
    sum is taken over the shorter side; ell = 0 gives 0 and ell = m + 1
    gives 1, the limits of the rank.
    """
    e = Fraction(epsilon)
    p = 1 - e
    if ell <= 0:
        return Fraction(0)
    if ell > m:
        return Fraction(1)
    if ell - 1 <= m - ell:
        return sum(comb(m, k) * p**k * e**(m - k) for k in range(ell))
    return 1 - sum(comb(m, k) * p**k * e**(m - k) for k in range(ell, m + 1))


def _two_sided(n, p, alpha):
    """[lo, hi] that holds a Bin(n, p) count with probability >= 1 - alpha."""
    return binom.ppf(alpha / 2, n, p), binom.isf(alpha / 2, n, p)


class _IdentityFlow:
    """A flow whose latent point is the data point: the set in its frame is
    the Euclidean ball of radius q, and the scenario check samples it."""

    velocity_field = None

    def inverse(self, z, **kwargs):
        return z

    def forward(self, y, **kwargs):
        return y

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


# A 3-input, 2-output affine map with dyadic weights: output count differs
# from input count, and the output spread is far from 1 per coordinate, so a
# whitening by the variance or by the wrong sample shows.
_W = np.array([[2.0, -4.0, 0.5], [-1.0, 8.0, 3.0]])
_B = np.array([1.5, -0.25])
# Dyadic bounds, both signs, widths above and below 1: float32 sampling of
# this box is exact up to rounding that stays inside the closed box.
_LB = np.array([-1.0, 0.5, 2.0])
_UB = np.array([3.0, 0.75, 6.0])


def _tiny(**overrides):
    """A config small enough for the default tier (well under a second)."""
    cfg = dict(m=24, epsilon=0.125, n_train=40, flow_epochs=1,
               flow_config="base", flow_coupling="none", flow_use_ema=False,
               infer_steps=2, infer_solver="euler", seed=5)
    cfg.update(overrides)
    return FlowReachConfig(**cfg)


class _RecordingCallable:
    """A numpy model y = W x + b (float64 output, numpy's default) that
    records every batch of inputs it receives and the outputs it returned."""

    def __init__(self):
        self.calls = []

    def __call__(self, x):
        x = np.array(x, copy=True)
        y = x.astype(np.float64) @ _W.T + _B
        self.calls.append((x, y))
        return y


class _RecordingModule(nn.Module):
    """The same map as an nn.Module with parameters, recording its calls."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.lin = nn.Linear(3, 2).to(dtype)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor(_W, dtype=dtype))
            self.lin.bias.copy_(torch.tensor(_B, dtype=dtype))
        self.calls = []

    def forward(self, x):
        y = self.lin(x)
        self.calls.append((x.detach().cpu().numpy().copy(),
                           y.detach().cpu().numpy().astype(np.float64)))
        return y


def _batch(calls, n_rows, what):
    """The one recorded call on n_rows inputs; fails the test if there is not
    exactly one (the calibration size and the training size differ). It uses
    pytest.fail, not assert, so that a strict expected-failure marker that
    absorbs AssertionError does not absorb a broken setup."""
    found = [c for c in calls if c[0].shape[0] == n_rows]
    if len(found) != 1:
        pytest.fail(
            f"expected exactly one model call on {n_rows} inputs (the {what} "
            f"sample); got calls on {[c[0].shape[0] for c in calls]}")
    return found[0]


# --------------------------------------------------------------------------- #
# Calibration and the stated guarantee                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m, ell, epsilon", [
    pytest.param(5, 1, 0.25, id="ell=1"),
    pytest.param(5, 5, 0.25, id="ell=m"),
    pytest.param(1000, 999, 0.01, id="m=1000-ell=m-1"),
])
def test_stated_guarantee_is_the_binomial_tail_of_the_rank(m, ell, epsilon):
    """The set states coverage 1 - epsilon and, as confidence, the exact
    binomial tail P(Bin(m, 1 - epsilon) <= ell - 1) computed in fractions,
    through the constructor and get_guarantee and through compute_guarantee
    alone. Catches a rank shifted by one in the constructor (ell + 1), a beta
    shape of m - ell for m + 1 - ell (a confidence too high, or NaN at
    ell = m) and coverage and confidence returned in swapped order. The
    library's beta evaluation agrees with the fractions to about 2e-16."""
    exact = float(_exact_confidence(m, ell, epsilon))
    coverage, confidence = compute_guarantee(m, ell, epsilon)
    assert coverage == 1 - epsilon
    assert abs(confidence - exact) <= 1e-12, (confidence, exact)

    pset = ProbabilisticSet(BallScore(torch.zeros(2)), threshold=1.0, m=m,
                            ell=ell, epsilon=epsilon, dim=2)
    stated_coverage, stated_confidence = pset.get_guarantee()
    assert stated_coverage == 1 - epsilon
    assert abs(stated_confidence - exact) <= 1e-12, (stated_confidence, exact)


# Twelve points around c = (1.5, -2.5) at integer offsets whose squared
# distances 1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20 are distinct, listed in
# an order that is not the order of their distances, with mixed signs.
_CENTRE = (1.5, -2.5)
_OFFSETS = [(-3, 2), (1, 0), (0, -4), (2, -2), (-1, 1), (4, 1), (-2, 0),
            (3, -1), (-1, -2), (0, 3), (3, 3), (-4, 2)]


@pytest.mark.parametrize("ell", [1, 7, 12])
def test_calibrated_set_holds_exactly_ell_of_its_calibration_points(ell):
    """A threshold calibrated at rank ell over m points makes a set that
    holds exactly the ell points of smallest score, the one whose score is
    the threshold included: that is what the rank and the stated confidence
    are computed for. The oracle ranks the exact integer squared distances.
    Catches calibrate sorting in descending order (the ell-th largest), a
    0-based rank, and contains comparing with < where the calibration counts
    the point on the level set as covered."""
    points = torch.tensor([[_CENTRE[0] + dx, _CENTRE[1] + dy]
                           for dx, dy in _OFFSETS], dtype=torch.float32)
    squared = [dx * dx + dy * dy for dx, dy in _OFFSETS]
    if len(set(squared)) != len(squared):
        pytest.fail("fixture: the squared distances must be distinct")
    order = sorted(range(len(squared)), key=lambda i: squared[i])
    expected = torch.zeros(len(squared), dtype=torch.bool)
    expected[order[:ell]] = True

    score = BallScore(torch.tensor(_CENTRE, dtype=torch.float32))
    q = calibrate(score(points), ell).item()
    pset = ProbabilisticSet(score, threshold=q, m=len(squared), ell=ell,
                            epsilon=0.25, dim=2)
    inside = pset.contains(points)
    assert torch.equal(inside, expected), (
        f"rank {ell}: the set holds the points {inside.nonzero().ravel().tolist()}, "
        f"the {ell} nearest are {sorted(order[:ell])}")


def test_stated_confidence_holds_over_repeated_calibrations():
    """The confidence a calibrated set states is not above the frequency with
    which its coverage really exceeds 1 - epsilon. Scores are |Y| for Y
    uniform on [-1, 1], so the coverage of the set {|y| <= q} is q in closed
    form. Over 2000 seeded calibrations of m = 50 points at rank 48 the count
    of sets with q > 1 - epsilon is Bin(2000, true confidence); the test
    demands it is not below the 1e-6 lower quantile of Bin(2000, stated
    confidence). Catches a descending sort (q is then the third smallest
    score), a constructor that computes the guarantee for rank ell + 1 and a
    beta shape of m - ell, each of which states a confidence the coverage
    does not have (0.966 and 0.963 against the true 0.888)."""
    m, ell, epsilon, trials = 50, 48, 0.1, 2000
    gen = torch.Generator().manual_seed(20261118)
    score = BallScore(torch.zeros(1))
    stated = None
    success = 0
    for _ in range(trials):
        y_cal = torch.rand(m, 1, generator=gen) * 2 - 1
        q = calibrate(score(y_cal), ell).item()
        pset = ProbabilisticSet(score, threshold=q, m=m, ell=ell,
                                epsilon=epsilon, dim=1)
        stated = pset.get_guarantee()[1]
        success += q > 1 - epsilon        # true coverage of {|y| <= q} is q
    floor = binom.ppf(1e-6, trials, stated)
    assert success >= floor, (
        f"{success} of {trials} calibrated sets cover more than "
        f"{1 - epsilon}; a stated confidence of {stated:.4f} needs at least "
        f"{floor:.0f}")


# --------------------------------------------------------------------------- #
# The whitened frame                                                           #
# --------------------------------------------------------------------------- #
# Different scale per axis (above and below 1, one not a power of two) and a
# mean with both signs; all dyadic, so the transformed rows are exact floats.
_MEAN3 = np.array([8.0, -4.0, 0.75])
_STD3 = np.array([4.0, 0.25, 1.5])


@pytest.mark.parametrize("G, g", [
    pytest.param([[-1.0, 2.0, 0.5]], [-3.0], id="one-row"),
    pytest.param([[0.0, 1.0, -2.0], [-1.0, 0.0, 0.25]], [[0.5], [-10.0]],
                 id="two-rows"),
    pytest.param([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 2.0]],
                 [-3.75, 0.5, -6.0], id="square"),
])
def test_whitened_halfspace_holds_the_same_points(G, g):
    """transform_halfspace is the exact coordinate change it documents: for
    every raw point y, G_w (y - mean) / std - g_w equals G y - g, so a point
    is on the same side of every row, and on a row's plane, in both frames.
    Checked in fractions on points on each plane, one unit to each side of
    it and at a few other places. Catches dividing the columns by std, scaling
    the rows instead of the columns (silent only for a square G) and dropping
    the mean shift; each turns a real violation into a whitened point outside
    the half-space, which verify_specification answers UNSAT."""
    transform = AffineTransform(mean=_MEAN3, std=_STD3)
    hs = HalfSpace(np.array(G), np.array(g))
    white = transform.transform_halfspace(hs)
    assert white.G.shape == hs.G.shape and white.g.shape == hs.g.shape

    Gf = [[Fraction(v) for v in row] for row in hs.G]
    gf = [Fraction(v) for v in hs.g.ravel()]
    Gw = [[Fraction(v) for v in row] for row in white.G]
    gw = [Fraction(v) for v in white.g.ravel()]
    mean = [Fraction(v) for v in _MEAN3]
    std = [Fraction(v) for v in _STD3]

    def residual(rows, rhs, y):
        return [sum(a * b for a, b in zip(row, y)) - r
                for row, r in zip(rows, rhs)]

    points = [[Fraction(v) for v in p] for p in
              ([0, 0, 0], [8, -4, 0.75], [12, -4.25, 2.25], [-3, 7, -1.5])]
    for row, r in zip(Gf, gf):
        norm2 = sum(a * a for a in row)
        base = [Fraction(5), Fraction(-3), Fraction(1, 3)]
        t = (r - sum(a * b for a, b in zip(row, base))) / norm2
        on_plane = [b + t * a for a, b in zip(row, base)]
        points += [on_plane,
                   [v + a / norm2 for v, a in zip(on_plane, row)],
                   [v - a / norm2 for v, a in zip(on_plane, row)]]

    for y in points:
        raw = residual(Gf, gf, y)
        y_white = [(v - mu) / s for v, mu, s in zip(y, mean, std)]
        whitened = residual(Gw, gw, y_white)
        assert whitened == raw, (
            f"point {[float(v) for v in y]}: raw residuals "
            f"{[float(v) for v in raw]}, whitened {[float(v) for v in whitened]}")


def _flow_set_with_whitening():
    """A flow set whose whitened frame is the unit ball (identity flow,
    q = 1) and whose raw image is the ellipse centred at (8, -4) with
    semi-axes 4 and 1/4."""
    return ProbabilisticSet(
        SimpleNamespace(flow_model=_IdentityFlow()), threshold=1.0, m=1000,
        ell=999, epsilon=0.01, dim=2,
        affine_transform=AffineTransform(mean=np.array([8.0, -4.0]),
                                         std=np.array([4.0, 0.25])))


@pytest.mark.parametrize("G, g, meets", [
    # {y_1 <= 0} and {y_0 >= 10}: whitened x_w >= 1/2 cuts the unit ball.
    pytest.param([[0.0, 1.0], [-1.0, 0.0]], [[0.0], [-10.0]], True,
                 id="meets-the-set"),
    # {y_1 <= -4.5} and {y_0 <= 100}: whitened y_w <= -2 misses the ball.
    pytest.param([[0.0, 1.0], [1.0, 0.0]], [[-4.5], [100.0]], False,
                 id="misses-the-set"),
])
def test_verify_specification_reads_a_raw_halfspace_in_the_sets_frame(G, g, meets):
    """Through the public route: verify_specification of a set that carries
    a whitening reads a raw-coordinate unsafe region in the set's frame. A
    region that meets the raw image of the set must not be answered UNSAT
    (the scenario check finds whitened samples in it); one that misses it is
    UNSAT. Both specs have two rows and a square G whose entries are off the
    diagonal. Catches dividing by std, scaling rows instead of columns and
    dropping the mean shift (the meeting region whitened wrongly misses the
    ball: a false UNSAT), and the row scaling on the missing region (a false
    UNKNOWN)."""
    result = verify_specification(
        _flow_set_with_whitening(), HalfSpace(np.array(G), np.array(g)),
        config=ProbVerifyConfig(method="scenario", n_samples=500, seed=0))
    if meets:
        assert result.verdict != "UNSAT", (
            "an unsafe region that meets the set was answered UNSAT")
    else:
        assert result.verdict == "UNSAT", result.verdict


# --------------------------------------------------------------------------- #
# flow_reach: the calibration it hands to the set                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("route, ell", [
    pytest.param("flow_reach-numpy-callable", None, id="flow_reach-numpy-default-ell"),
    pytest.param("flow_reach-module", 21, id="flow_reach-module-ell=21"),
    pytest.param("NeuralNetwork.reach", 21, id="NeuralNetwork.reach-ell=21"),
])
def test_flow_reach_calibrates_on_a_held_out_sample_of_the_input_box(route, ell):
    """The set flow_reach returns is the calibration of its own held-out
    sample: the model sees n_train training inputs and m calibration inputs,
    all inside the input box and no input in both; the stored whitening is
    the mean and an empirical standard deviation of the training outputs; the
    set's score is the flow integrated with the configured solver and steps;
    and the set holds exactly ell of the m whitened calibration outputs,
    ell = m - 1 when it is not given. The set states its own m, ell, epsilon,
    the output dimension and the exact binomial confidence. Catches a
    calibration sample of n_train points, a threshold at rank ell - 1, a
    default rank of m, a score that ignores the configured solver, a
    whitening by the variance or by the calibration outputs' mean, and the
    input dimension stated as the set's dimension."""
    cfg = _tiny() if ell is None else _tiny(ell=ell)
    expected_ell = cfg.m - 1 if ell is None else ell
    box = Box(_LB, _UB)
    if route == "flow_reach-numpy-callable":
        model = _RecordingCallable()
        pset = flow_reach(model, box, cfg)
    elif route == "flow_reach-module":
        model = _RecordingModule()
        pset = flow_reach(model, box, cfg)
    else:
        model = _RecordingModule()
        pset = NeuralNetwork(model).reach(box, method="flow_matching", config=cfg)

    x_tr, y_tr = _batch(model.calls, cfg.n_train, "training")
    x_ca, y_ca = _batch(model.calls, cfg.m, "calibration")
    for x, _ in model.calls:
        assert np.all(x >= _LB) and np.all(x <= _UB), "an input left the box"
    shared = {tuple(r) for r in x_tr} & {tuple(r) for r in x_ca}
    assert not shared, f"{len(shared)} calibration inputs are training inputs"

    assert (pset.m, pset.ell, pset.epsilon, pset.dim) == (
        cfg.m, expected_ell, cfg.epsilon, 2)
    assert pset.get_guarantee() == pytest.approx(
        (1 - cfg.epsilon, float(_exact_confidence(cfg.m, expected_ell, cfg.epsilon))),
        abs=1e-12)

    # Whitening: mean and an empirical std (n or n - 1) of the float32
    # training outputs. Float32 sums of 40 values of magnitude below 64 are
    # off by less than 40 * 2**-24 * 64 = 1.6e-4; relative 1e-4 on the std.
    y32 = y_tr.astype(np.float32).astype(np.float64)
    at = pset.affine_transform
    assert np.all(np.abs(at.mean - y32.mean(axis=0)) <= 2e-4), (at.mean, y32.mean(axis=0))
    std0, std1 = y32.std(axis=0, ddof=0), y32.std(axis=0, ddof=1)
    assert np.all(at.std >= std0 * (1 - 1e-4)) and np.all(at.std <= std1 * (1 + 1e-4)), (
        f"stored std {at.std}, empirical std between {std0} and {std1}")

    # The score is the configured integration of the trained flow, and the
    # set holds exactly ell of the whitened calibration outputs.
    mean32 = torch.tensor(at.mean, dtype=torch.float32)
    std32 = torch.tensor(at.std, dtype=torch.float32)
    y_white = (torch.tensor(y_ca, dtype=torch.float32) - mean32) / std32
    with torch.no_grad():
        mine = pset.score_fn.flow_model.forward(
            y_white, t=1.0, n_steps=cfg.infer_steps,
            method=cfg.infer_solver).norm(dim=1)
        theirs = pset.score_fn(y_white)
    assert torch.allclose(theirs, mine, rtol=1e-6, atol=1e-6), (
        "the set's score is not the flow integrated with the configured solver")
    held = int(pset.contains(y_white).sum())
    assert held == expected_ell, (
        f"the set holds {held} of its {cfg.m} calibration outputs; rank "
        f"{expected_ell} must hold {expected_ell}")


def test_flow_reach_draws_each_sample_with_its_documented_seed():
    """flow_reach draws the training sample with flow_seed (default: seed)
    and the calibration sample with cal_seed + 1_000_000 (default: seed +
    1_000_000), as its docstring states, so an override changes exactly its
    own sample. The reference is sample_box with the documented seed. Catches
    an override that is ignored (the master seed used for training or for
    calibration)."""
    box = Box(_LB, _UB)
    lb = torch.tensor(_LB, dtype=torch.float32)
    ub = torch.tensor(_UB, dtype=torch.float32)
    for overrides, train_seed, cal_seed in [
        (dict(seed=5), 5, 5),
        (dict(seed=5, flow_seed=9, cal_seed=13), 9, 13),
    ]:
        cfg = _tiny(**overrides)
        model = _RecordingCallable()
        flow_reach(model, box, cfg)
        x_tr, _ = _batch(model.calls, cfg.n_train, "training")
        x_ca, _ = _batch(model.calls, cfg.m, "calibration")
        np.testing.assert_array_equal(
            x_tr, sample_box(lb, ub, cfg.n_train, train_seed).numpy(),
            err_msg=f"{overrides}: training sample is not seed {train_seed}")
        np.testing.assert_array_equal(
            x_ca, sample_box(lb, ub, cfg.m, cal_seed + 1_000_000).numpy(),
            err_msg=f"{overrides}: calibration sample is not seed "
                    f"{cal_seed} + 1_000_000")


@pytest.mark.parametrize("overrides", [
    pytest.param(dict(seed=5), id="default-seeds"),
    pytest.param(dict(seed=0, flow_seed=1_000_000), id="flow_seed=cal_seed+1e6",
                 marks=known_bug(13, "flow_reach seeds the training sample with "
                                     "flow_seed and the calibration sample with "
                                     "cal_seed + 1_000_000, so flow_seed = cal_seed "
                                     "+ 1_000_000 makes them the same points: the "
                                     "flow is fitted to its calibration data and "
                                     "the stated confidence does not hold")),
])
def test_training_and_calibration_samples_share_no_point(overrides):
    """The conformal guarantee needs calibration scores exchangeable with a
    fresh output's score, so the flow (the score function) must not be fitted
    to the calibration points, whatever seeds the caller chooses."""
    model = _RecordingCallable()
    cfg = _tiny(**overrides)
    flow_reach(model, Box(_LB, _UB), cfg)
    x_tr, _ = _batch(model.calls, cfg.n_train, "training")
    x_ca, _ = _batch(model.calls, cfg.m, "calibration")
    shared = {tuple(r) for r in x_tr} & {tuple(r) for r in x_ca}
    assert not shared, f"{len(shared)} of {cfg.m} calibration inputs are training inputs"


def test_flow_reach_set_covers_fresh_outputs_at_the_stated_level():
    """End to end: the set flow_reach returns covers fresh outputs of the
    model at the level the conformal guarantee states. Its coverage C is
    Beta(ell, m + 1 - ell) over the calibration draw whatever flow was
    trained; with m = 200, ell = 199 the 5e-7 quantile c0 of C bounds it
    below, and a 5e-7 binomial quantile at c0 bounds the covered count of
    4000 fresh outputs (independent numpy draws from the box, whitened in
    float64 from the stored transform). estimate_coverage on its own seed
    must meet the same floor. A correct implementation fails with
    probability at most 1e-6 per check. Catches calibrating on a sample of
    n_train = 1000 points at rank 199 (coverage about 0.2) and any frame
    mismatch between calibration and membership."""
    cfg = FlowReachConfig(m=200, epsilon=0.02, n_train=1000, flow_epochs=2,
                          flow_config="base", flow_coupling="none",
                          flow_use_ema=False, infer_steps=2,
                          infer_solver="euler", seed=11)
    box = Box(_LB, _UB)
    model = _RecordingCallable()
    pset = flow_reach(model, box, cfg)

    n_test = 4000
    c0 = beta_dist.ppf(5e-7, cfg.ell, cfg.m + 1 - cfg.ell)
    floor = binom.ppf(5e-7, n_test, c0)
    rng = np.random.default_rng(12345)
    x = rng.uniform(_LB, _UB, size=(n_test, 3))
    y = x @ _W.T + _B
    y_white = (y - pset.affine_transform.mean) / pset.affine_transform.std
    covered = int(pset.contains(torch.tensor(y_white, dtype=torch.float32)).sum())
    assert covered >= floor, (
        f"{covered} of {n_test} fresh outputs in the set; the guarantee "
        f"needs at least {floor:.0f}")
    estimated = pset.estimate_coverage(model, box, n_test=n_test, seed=777)
    assert estimated * n_test >= floor, (estimated, floor / n_test)


class _ParameterFreeAffine(nn.Module):
    """y = W x + b with W and b built in forward: no parameters, no buffers."""

    def forward(self, x):
        w = torch.tensor(_W, dtype=x.dtype)
        b = torch.tensor(_B, dtype=x.dtype)
        return x @ w.T + b


class _BufferOnlyAffine(nn.Module):
    """y = W x + b with W and b registered as buffers only."""

    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.tensor(_W, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(_B, dtype=torch.float32))

    def forward(self, x):
        return x @ self.w.T + self.b


@pytest.mark.parametrize("kind", [
    "numpy-callable-float64-output",
    "module-without-parameters-or-buffers",
    "module-with-buffers-only",
    pytest.param("float64-module", marks=known_bug(
        13, "flow_reach feeds float32 samples to an nn.Module without casting "
           "them to the module's dtype, so a float64 model raises a torch "
           "RuntimeError (mat1 and mat2 must have the same dtype)",
        raises=RuntimeError)),
])
def test_flow_reach_answers_for_every_kind_of_model(kind):
    """flow_reach accepts any numpy callable and any nn.Module: a callable
    that returns float64 arrays, a module with neither parameters nor
    buffers (run on the CPU), one with buffers only, and a float64 module.
    Each answers with the whitening of its training outputs: the stored mean
    is the mean of W x + b over the documented training sample (float64
    reference, float32 tolerance 2e-4) and the set's dimension is the output
    count. Catches a numpy round trip that keeps float64 (the flow then
    trains on float64 data and raises) and a device lookup without the CPU
    fallback (StopIteration on a parameter-free module)."""
    model = {
        "numpy-callable-float64-output": lambda x: x.astype(np.float64) @ _W.T + _B,
        "module-without-parameters-or-buffers": _ParameterFreeAffine(),
        "module-with-buffers-only": _BufferOnlyAffine(),
        "float64-module": _RecordingModule(dtype=torch.float64),
    }[kind]
    cfg = _tiny()
    pset = flow_reach(model, Box(_LB, _UB), cfg)
    x_tr = sample_box(torch.tensor(_LB, dtype=torch.float32),
                      torch.tensor(_UB, dtype=torch.float32),
                      cfg.n_train, cfg.seed).numpy().astype(np.float64)
    reference = (x_tr @ _W.T + _B).mean(axis=0)
    assert pset.dim == 2
    assert np.all(np.abs(pset.affine_transform.mean - reference) <= 2e-4), (
        pset.affine_transform.mean, reference)


def test_flow_reach_answers_for_a_model_with_a_constant_output():
    """A model output that does not vary over the input box (here the second
    of two outputs is 3) has an empirical spread of 0, which AffineTransform
    refuses as a std; flow_reach clamps the spread to a small positive value
    and answers for this legal model. The stored whitening holds the
    constant exactly as that output's mean and a positive std, and the set
    still holds exactly ell of its m whitened calibration outputs (the
    whitened constant output is 0 for every point). Catches the clamp
    dropped: flow_reach then refuses the model with ValueError."""
    calls = []

    def model(x):
        calls.append(np.array(x, copy=True))
        first = x.astype(np.float64) @ _W[0] + _B[0]
        return np.stack([first, np.full(x.shape[0], 3.0)], axis=1)

    cfg = _tiny(ell=21)
    pset = flow_reach(model, Box(_LB, _UB), cfg)
    at = pset.affine_transform
    assert at.mean[1] == 3.0 and at.std[1] > 0, (at.mean, at.std)
    x_ca = [c for c in calls if c.shape[0] == cfg.m]
    if len(x_ca) != 1:
        pytest.fail(f"expected one calibration call on {cfg.m} inputs")
    y_ca = torch.tensor(model(x_ca[0]), dtype=torch.float32)
    y_white = ((y_ca - torch.tensor(at.mean, dtype=torch.float32))
               / torch.tensor(at.std, dtype=torch.float32))
    held = int(pset.contains(y_white).sum())
    assert held == cfg.ell, (
        f"the set holds {held} of its {cfg.m} calibration outputs, not {cfg.ell}")


@pytest.mark.parametrize("field, value", [
    ("epsilon", 0.0), ("epsilon", 1.0), ("ell", 0), ("ell", 25),
    ("flow_config", "fast"), ("flow_coupling", "ot"), ("n_train", 0),
    ("flow_epochs", 0), ("infer_steps", 0),
])
def test_flow_reach_config_refuses_values_outside_their_range(field, value):
    """Every documented range of FlowReachConfig is enforced with ValueError
    (m = 24 here, so ell = 25 is above m). The check m >= 1 has no case of
    its own: for m < 1 no ell lies in [1, m], so the rank check refuses it
    too. Catches each range check that lets its boundary value through; for
    ell = 0, calibrate then wraps index -1 to the largest score and the set
    states a NaN confidence."""
    cfg = dict(m=24)
    cfg[field] = value
    with pytest.raises(ValueError):
        FlowReachConfig(**cfg)


def test_flow_reach_builds_its_config_from_keyword_arguments():
    """flow_reach(model, box, **kwargs) is FlowReachConfig(**kwargs): the set
    states the m, ell and epsilon passed as keywords, and the model sees the
    n_train and m samples they ask for. The model refuses a batch of more
    than 100 inputs other than the 2000-point diagnostic sample, so a
    flow_reach that drops the keywords and runs the defaults (10000 training
    samples, 5000 epochs) fails at once instead of training for minutes."""
    calls = []

    def guarded(x):
        assert x.shape[0] <= 100 or x.shape[0] == 2000, (
            f"model called on {x.shape[0]} inputs")
        calls.append(x.shape[0])
        return x.astype(np.float64) @ _W.T + _B

    kwargs = dict(m=24, ell=20, epsilon=0.125, n_train=40, flow_epochs=1,
                  flow_config="base", flow_coupling="none", flow_use_ema=False,
                  infer_steps=2, infer_solver="euler", seed=5)
    pset = flow_reach(guarded, Box(_LB, _UB), **kwargs)
    assert (pset.m, pset.ell, pset.epsilon) == (24, 20, 0.125)
    assert sorted(c for c in calls if c != 2000) == [24, 40], calls


def test_flow_reach_config_defaults_are_the_documented_ones():
    """FlowReachConfig() holds the defaults its docstring states: epsilon
    0.001, m 8000, ell m - 1 = 7999, 10000 training samples, 5000 epochs,
    the 'tight' preset with Sinkhorn coupling and EMA, 30 rk4 steps and seed
    0 with no overrides. flow_reach() with no config builds exactly this one
    (too costly to train here). Catches a default rank of m and any default
    that drifts from its documentation."""
    cfg = FlowReachConfig()
    assert (cfg.epsilon, cfg.m, cfg.ell, cfg.n_train, cfg.flow_epochs) == (
        0.001, 8000, 7999, 10_000, 5000)
    assert (cfg.flow_config, cfg.flow_coupling, cfg.flow_use_ema) == (
        "tight", "sinkhorn", True)
    assert (cfg.infer_solver, cfg.infer_steps) == ("rk4", 30)
    assert (cfg.seed, cfg.flow_seed, cfg.cal_seed) == (0, None, None)


def test_flow_reach_refuses_an_unknown_solver():
    """FlowReachConfig does not check infer_solver; flow_reach refuses an
    unknown name with ValueError when the calibration integrates the flow
    (after training). Catches a silent fallback to another solver, which
    would calibrate the set with a score the caller did not ask for."""
    with pytest.raises(ValueError, match="no-such-solver"):
        flow_reach(_RecordingCallable(), Box(_LB, _UB),
                   _tiny(infer_solver="no-such-solver"))


def test_flow_reach_is_reproducible_whatever_the_global_random_state():
    """Identical seeds and config give a bit-identical set (documented),
    also when the global torch and numpy random states differ between the
    two calls: the threshold, the whitening and the score of a fixed point
    agree exactly. Catches a sample, an initialisation or a calibration
    step that draws from the global random state."""
    def run(global_seed):
        torch.manual_seed(global_seed)
        np.random.seed(global_seed)
        pset = flow_reach(_RecordingCallable(), Box(_LB, _UB), _tiny())
        probe = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
        return pset.threshold, pset.affine_transform, pset.score_fn(probe).item()

    q1, at1, s1 = run(1)
    q2, at2, s2 = run(2)
    assert q1 == q2 and s1 == s2
    np.testing.assert_array_equal(at1.mean, at2.mean)
    np.testing.assert_array_equal(at1.std, at2.std)


def test_flow_reach_refuses_a_config_together_with_kwargs():
    """config= and FlowReachConfig keyword arguments together raise
    TypeError, as documented; otherwise a keyword such as m= beside a config
    is silently dropped."""
    with pytest.raises(TypeError):
        flow_reach(_RecordingCallable(), Box(_LB, _UB), _tiny(), m=8)


def test_flow_config_selects_the_documented_preset():
    """flow_config='tight' is documented as the high-capacity preset and
    'base' as the lighter one: the flow trained under 'tight' has more
    parameters. Catches the two presets swapped or the option ignored."""
    sizes = {}
    for preset in ("base", "tight"):
        pset = flow_reach(_RecordingCallable(), Box(_LB, _UB),
                          _tiny(flow_config=preset))
        sizes[preset] = sum(p.numel() for p in
                            pset.score_fn.flow_model.velocity_field.parameters())
    assert sizes["tight"] > sizes["base"], sizes


# --------------------------------------------------------------------------- #
# The input samplers                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sampler", ["sample_box", "sample_l_inf_ball"])
def test_samplers_draw_from_the_whole_stated_box(sampler):
    """Every sample lies in the stated box (exactly: dyadic bounds), every
    face is reached as often as a uniform law says (the share of samples in
    the outer eighth next to each face), two coordinates are independent (the
    share in one corner quarter), and a coordinate of zero width is held at
    its value. Tolerances are binomial quantiles with 1e-6 split over the
    checks. Catches samples outside the box (a full width used as half-width,
    Gaussian perturbations), samples from the inner half only, one uniform
    number broadcast to every coordinate, and a degenerate coordinate
    refused."""
    n = 20000
    if sampler == "sample_box":
        lb = np.array([-3.0, 0.5, 2.0, -0.25])
        ub = np.array([5.0, 0.75, 2.0, 1.75])
        x = sample_box(torch.tensor(lb, dtype=torch.float32),
                       torch.tensor(ub, dtype=torch.float32), n, 3).numpy()
    else:
        centre, radius = np.array([-3.0, 0.5, 2.0]), 1.5
        lb, ub = centre - radius, centre + radius
        x = sample_l_inf_ball(torch.tensor(centre, dtype=torch.float32),
                              radius, n, 3, dim=3).numpy()
    x = x.astype(np.float64)
    assert x.shape == (n, lb.size)
    assert np.all(x >= lb) and np.all(x <= ub), "a sample left the box"

    wide = [k for k in range(lb.size) if ub[k] > lb[k]]
    alpha = 1e-6 / (2 * len(wide) + 1)
    lo, hi = _two_sided(n, 1 / 8, alpha)
    for k in range(lb.size):
        if ub[k] == lb[k]:
            assert np.all(x[:, k] == lb[k]), f"coordinate {k} of zero width moved"
            continue
        eighth = (ub[k] - lb[k]) / 8
        near_low = int((x[:, k] <= lb[k] + eighth).sum())
        near_high = int((x[:, k] >= ub[k] - eighth).sum())
        assert lo <= near_low <= hi, f"coordinate {k}: {near_low} near the lower face"
        assert lo <= near_high <= hi, f"coordinate {k}: {near_high} near the upper face"
    a, b = wide[0], wide[-1]
    corner = int(((x[:, a] <= (lb[a] + ub[a]) / 2)
                  & (x[:, b] <= (lb[b] + ub[b]) / 2)).sum())
    lo4, hi4 = _two_sided(n, 1 / 4, alpha)
    assert lo4 <= corner <= hi4, f"{corner} samples in one corner quarter of {n}"
