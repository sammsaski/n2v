"""Soundness cells of the conformal lane: ``conformal_reach`` and the
conformal inference behind it (``n2v/probabilistic/conformal.py``).

The promise: the returned ``ProbabilisticBox`` holds f(x), x uniform on the
input box, with probability at least 1 - epsilon, at the stated confidence
1 - BetaCDF(1 - epsilon; ell, m + 1 - ell). The cells check what that rests
on, each against an oracle computed here and never by n2v:

- the box holds at least ell of the m calibration outputs (exactly ell with
  the naive surrogate), for both surrogates, with and without PCA, through
  keywords and config=; the calibration outputs are the rows the model itself
  returned (``TestCalibrationRank``);
- with one output the box is the split-conformal interval, written down in a
  few lines of numpy (``TestOneOutputInterval``);
- the coverage of a box, exact for a model whose output distribution is a
  closed form, has the Beta distribution the stated confidence assumes
  (``TestConformalGuarantee``; a statistical test whose failure probability
  for a correct implementation is at most 1e-6);
- training and calibration inputs are fresh uniform draws of the input box
  (``TestSampling``);
- the stated confidence is the binomial tail summed in rationals.

These properties do not depend on the per-output scale tau, so they hold on
today's library although tau is the deferred PROB-6 formula (its coverage
failure with many outputs is marked in
``tests/contracts/test_probabilistic_contract.py``).
"""

from fractions import Fraction
from math import comb

import pytest
import numpy as np
from scipy.stats import beta

from n2v.probabilistic import ConformalReachConfig, conformal_reach
from n2v.sets import Box


# --------------------------------------------------------------------------- #
# Oracles shared by the cells below                                            #
# --------------------------------------------------------------------------- #
def _exact_confidence(m, ell, epsilon):
    """Confidence of rank ``ell`` of ``m`` at miscoverage ``epsilon`` (a decimal
    string), in rationals: the probability that a Binomial(m, 1 - epsilon)
    count is at most ell - 1, which is 1 - BetaCDF(1 - epsilon; ell,
    m + 1 - ell) written as a finite sum (the shorter tail is summed)."""
    eps = Fraction(epsilon)
    q = 1 - eps
    if ell - 1 <= m // 2:
        return sum(comb(m, j) * q ** j * eps ** (m - j) for j in range(ell))
    return 1 - sum(comb(m, j) * q ** j * eps ** (m - j)
                   for j in range(ell, m + 1))


class _Recorder:
    """A numpy model that keeps every input batch it is evaluated on and the
    outputs it returned, and fails the test as soon as it is asked for more
    rows than ``budget``: a run that ignores the requested sizes then fails at
    once instead of running thousands of evaluations."""

    def __init__(self, fn, budget):
        self.fn, self.budget = fn, budget
        self.inputs, self.outputs = [], []

    def __call__(self, x):
        done = sum(len(b) for b in self.inputs)
        if done + len(x) > self.budget:
            pytest.fail(f"the model was asked for {done + len(x)} rows, more "
                        f"than the {self.budget} the requested sizes need")
        y = self.fn(np.asarray(x, dtype=np.float64))
        self.inputs.append(np.array(x, copy=True))
        self.outputs.append(np.array(y, copy=True))
        return y

    def rows(self):
        return np.vstack(self.inputs), np.vstack(self.outputs)


def _contained(lb, ub, points, pad):
    """Number of rows of ``points`` inside [lb - pad, ub + pad]."""
    lb, ub = np.asarray(lb).ravel(), np.asarray(ub).ravel()
    return int(np.sum(np.all((points >= lb - pad) & (points <= ub + pad),
                             axis=1)))


# A box with intervals of different widths, both signs and ends beyond
# plus and minus 1; every end is dyadic, so a float32 draw stays inside.
_LB3 = np.array([-3.0, 0.5, -1.5])
_UB3 = np.array([1.0, 4.0, -0.25])
_LB2 = np.array([-2.5, 0.75])
_UB2 = np.array([1.5, 3.0])


def _two_outputs(x):
    """3 inputs -> 2 outputs, smooth and nonlinear, outputs of scale 1 to 10."""
    z = x @ np.array([[1.5, -0.5], [-0.75, 2.0], [2.0, 1.25]]) + np.array([4.0, -6.0])
    return z + 0.0625 * z ** 3


def _four_outputs(x):
    """2 inputs -> 4 outputs of rank 2 plus a nonlinear part outside the
    rank-2 subspace, so PCA with 2 components leaves a residual."""
    z = x @ np.array([[1.0, -2.0, 0.5, 3.0], [-1.5, 0.25, 2.0, -1.0]])
    return z + np.array([5.0, -3.0, 0.0, 1.0]) + 0.5 * np.sin(3.0 * z[:, ::-1])


def _with_constant_output(x):
    """_two_outputs and a third output that is 2.5 on the whole box: its
    errors are all zero, so only the floor of tau keeps its scores finite."""
    y = _two_outputs(x)
    return np.hstack([y, np.full((len(y), 1), 2.5)])


_RANK_CASES = [
    # model, box, options, exactly ell (naive) or at least ell, route
    pytest.param(_two_outputs, _LB3, _UB3,
                 dict(m=40, ell=37, epsilon=0.125, surrogate="naive",
                      training_samples=30), True, "keywords",
                 id="naive-3in-2out"),
    pytest.param(_four_outputs, _LB2, _UB2,
                 dict(m=40, ell=38, epsilon=0.25, surrogate="naive",
                      pca_components=2, training_samples=24), True, "keywords",
                 id="naive-pca-2in-4out"),
    pytest.param(_two_outputs, _LB3, _UB3,
                 dict(m=40, ell=39, epsilon=0.125, surrogate="clipping_block",
                      training_samples=20), False, "keywords",
                 id="clipping-3in-2out"),
    pytest.param(_four_outputs, _LB2, _UB2,
                 dict(m=40, ell=37, epsilon=0.25, surrogate="clipping_block",
                      pca_components=2, training_samples=24), False, "config",
                 id="clipping-pca-2in-4out-config"),
    pytest.param(_with_constant_output, _LB3, _UB3,
                 dict(m=40, ell=36, epsilon=0.125, surrogate="naive",
                      training_samples=16), True, "keywords",
                 id="naive-constant-output"),
]


class TestCalibrationRank:
    """The conformal box holds the calibration outputs whose score is at most
    the rank-ell score, which is what the stated guarantee rests on."""

    @pytest.mark.parametrize("fn, lb, ub, options, exactly, route", _RANK_CASES)
    def test_box_holds_ell_of_the_m_calibration_outputs(self, fn, lb, ub,
                                                        options, exactly, route):
        """The guarantee of ``conformal_reach`` holds because a calibration
        output is inside the box whenever its score is at most the ell-th
        smallest score, whatever the scale tau is; so the box holds at least
        ell of the m calibration outputs, and with the naive surrogate (box =
        centre plus and minus tau R) exactly ell. The calibration outputs are
        the last m rows the model was evaluated on (training first, then
        calibration, as the algorithm states), as the model returned them. This is independent of the per-output scale of the deferred
        PROB-6, so it holds on today's library. One case has an output that
        is constant on the box (its tau is the floor).

        Catches a threshold taken at the wrong rank or from a descending sort,
        an inflation that drops tau, scores that average instead of taking the
        maximum or drop the absolute value, an inflation added to the lower
        bound, a clipping-block box left without inflation, PCA bounds lifted
        without the interval map or without the mean, calibration errors taken
        against the PCA reconstruction, a projection of the clipping block
        that returns wrong points, wrong sizes of the training and calibration
        draws, a tau without its floor (NaN bounds on a constant output),
        and a box that states another m, ell or epsilon. The model
        fails the test at once when it is asked for more rows than the sizes
        need (keyword options dropped). Pad: 1e-12 of the output scale for the
        naive surrogate (float64 arithmetic), 1e-6 for the clipping block,
        whose projection is an LP solved to HiGHS's feasibility tolerance of
        1e-7."""
        m, ell = options["m"], options["ell"]
        t = options["training_samples"]
        model = _Recorder(fn, t + m)
        if route == "config":
            result = conformal_reach(model, Box(lb, ub),
                                     ConformalReachConfig(**options, seed=3))
        else:
            result = conformal_reach(model, Box(lb, ub), seed=3, **options)
        x, y = model.rows()
        assert len(x) == t + m, f"{len(x)} model rows, expected {t} + {m}"
        train_y, calib_y = y[:t], y[t:]
        assert (result.m, result.ell, result.epsilon) == (
            m, ell, options["epsilon"])
        assert abs(result.confidence - float(_exact_confidence(
            m, ell, str(options["epsilon"])))) <= 1e-12
        lb_out, ub_out = result.lb.ravel(), result.ub.ravel()
        assert lb_out.shape == (y.shape[1],)
        scale = 1.0 + np.max(np.abs(y))
        pad = (1e-12 if options["surrogate"] == "naive" else 1e-6) * scale
        held = _contained(lb_out, ub_out, calib_y, pad)
        assert held >= ell, (
            f"the box holds {held} of {m} calibration outputs, fewer than "
            f"ell = {ell}")
        if exactly:
            strictly = _contained(lb_out + pad, ub_out - pad, calib_y, 0.0)
            assert strictly <= ell, (
                f"the box holds {strictly} calibration outputs well inside, "
                f"more than ell = {ell}: the threshold is not the ell-th score")
            centre = train_y.mean(axis=0)
            assert np.all(np.abs((lb_out + ub_out) / 2 - centre) <= pad), (
                "the naive box is not centred on the training mean")
        elif "pca_components" not in options:
            assert np.all(lb_out <= train_y.min(axis=0)) and np.all(
                ub_out >= train_y.max(axis=0)), (
                "the clipping-block box does not hold the hull of the "
                "training outputs")


def _one_output(x):
    """3 inputs -> 1 output, nonlinear, with values on both sides of zero."""
    z = x @ np.array([[1.5], [-0.75], [2.0]]) + 1.0
    return z + 0.125 * z ** 3


class TestOneOutputInterval:
    """With one output the conformal box is the split-conformal interval."""

    @pytest.mark.parametrize("surrogate", ["naive", "clipping_block"])
    def test_one_output_box_is_the_split_conformal_interval(self, surrogate):
        """With one output the scale tau cancels (the box is c plus and minus
        tau times the ell-th smallest |e| / tau), so the box is fixed by the
        draws alone: for the naive surrogate it is the training mean plus and
        minus the ell-th smallest distance of a calibration output to it; for
        the clipping block it is [min, max] of the training outputs widened by
        the ell-th smallest distance of a calibration output to that interval
        (the L-infinity projection on a one-dimensional hull is clipping).
        Computed here from the outputs the model gave, in a few lines of
        numpy; the deferred PROB-6 does not touch it.

        Catches every wrong threshold rank, sort order, inflation, score or
        lower bound, and for the clipping block a projection that returns the
        wrong point (wrong LP weights, projections left at zero for a batch
        that was not processed) or hull bounds that are wrong. Slack: 1e-12
        of the output scale for the naive surrogate, 1e-6 for the clipping
        block (the LP's feasibility tolerance is 1e-7)."""
        m, ell, t = 40, 38, 10
        model = _Recorder(_one_output, t + m)
        result = conformal_reach(model, Box(_LB3, _UB3), m=m, ell=ell,
                                 epsilon=0.125, surrogate=surrogate,
                                 training_samples=t, seed=11)
        _, y = model.rows()
        assert y.shape == (t + m, 1)
        train, calib = y[:t, 0], y[t:, 0]
        if surrogate == "naive":
            lo = hi = train.mean()
            dist = np.abs(calib - lo)
        else:
            lo, hi = train.min(), train.max()
            dist = np.maximum(np.maximum(lo - calib, calib - hi), 0.0)
        r = np.sort(dist)[ell - 1]
        if not r > 0:
            pytest.fail("the case needs 3 calibration outputs outside the hull")
        scale = 1.0 + np.max(np.abs(y))
        tol = (1e-12 if surrogate == "naive" else 1e-6) * scale
        got_lb, got_ub = float(result.lb.ravel()[0]), float(result.ub.ravel()[0])
        assert abs(got_lb - (lo - r)) <= tol and abs(got_ub - (hi + r)) <= tol, (
            f"box [{got_lb}, {got_ub}], split-conformal interval "
            f"[{lo - r}, {hi + r}]")


def _sum_cdf(s, a0, b0, a1, b1):
    """CDF of u + v, u uniform on [a0, b0] and v uniform on [a1, b1]."""
    h = lambda z: np.maximum(z, 0.0) ** 2
    area = (h(s - a0 - a1) - h(s - a0 - b1) - h(s - b0 - a1)
            + h(s - b0 - b1)) / 2.0
    return area / ((b0 - a0) * (b1 - a1))


class TestConformalGuarantee:
    """The stated guarantee, checked where the coverage of a box is exact."""

    def test_coverage_of_a_one_output_box_has_the_stated_distribution(self):
        """The promise: Pr[coverage > 1 - epsilon] >= the stated confidence,
        for every epsilon. Model y = x0 + 2 x1 on [-3, 1] x [0.5, 2.5], so y
        is the sum of a uniform on [-3, 1] and one on [1, 5], whose CDF is a
        closed form and the coverage of an interval is exact. With one output
        and the naive surrogate the scores of fresh outputs are independent
        and identically distributed given the training mean, so the coverage
        of the box is the ell-th order statistic of m uniforms, Beta(ell,
        m + 1 - ell), exactly; then V = BetaCDF(coverage) is uniform on
        [0, 1], and every claim of the form above (each epsilon) is one point
        of V's distribution. K = 400 runs with seeds 0 to 399.

        Bound: by the Dvoretzky-Kiefer-Wolfowitz inequality with Massart's
        constant, Pr[sup |F_K - F| > t] <= 2 exp(-2 K t^2); t = 0.1347 makes
        this 1e-6, the probability that a correct implementation fails. A box
        at rank ell - 1 (m = 20, ell = 19) moves the distribution by 0.285 in
        this distance, and a box drawn from another region or with a wrong
        width moves it further. The stated confidence of every box is checked
        against the binomial sum in rationals as well."""
        m, ell, epsilon, runs = 20, 19, 0.1, 400
        a0, b0, a1, b1 = -3.0, 1.0, 1.0, 5.0
        box = Box(np.array([a0, a1 / 2]), np.array([b0, b1 / 2]))
        fn = lambda x: x[:, :1] + 2.0 * x[:, 1:2]
        stated = float(_exact_confidence(m, ell, str(epsilon)))
        v = np.empty(runs)
        for seed in range(runs):
            result = conformal_reach(_Recorder(fn, m + m // 2), box, m=m,
                                     ell=ell, epsilon=epsilon,
                                     surrogate="naive", seed=seed)
            assert abs(result.confidence - stated) <= 1e-12
            lo, hi = float(result.lb.ravel()[0]), float(result.ub.ravel()[0])
            coverage = (_sum_cdf(hi, a0, b0, a1, b1)
                        - _sum_cdf(lo, a0, b0, a1, b1))
            v[seed] = beta.cdf(coverage, ell, m + 1 - ell)
        v.sort()
        grid = np.arange(1, runs + 1) / runs
        distance = max(np.max(grid - v), np.max(v - (grid - 1.0 / runs)))
        bound = np.sqrt(np.log(2 / 1e-6) / (2 * runs))
        assert distance <= bound, (
            f"the coverage of the box is not Beta({ell}, {m + 1 - ell}): "
            f"distance {distance:.3f} > {bound:.3f}")


class TestSampling:
    """Training and calibration inputs are fresh uniform draws of the box."""

    def test_draws_are_fresh_uniform_samples_of_the_input_box(self):
        """The guarantee is about x uniform on the input box, and it needs
        calibration draws that are independent of the training draws. The
        model records every input row: t + m rows in all, each inside the box
        (its ends are dyadic, so a float32 draw cannot round outside), the
        degenerate coordinate exactly at its value, each other coordinate
        uniform on its own interval, and no calibration row equal to a
        training row (for independent float32 draws this has probability
        below 1e-12 here).

        Bound for uniformity: for each of the three coordinates the largest
        distance between the empirical CDF of the 300 rows and the uniform
        CDF is above t = 0.172 with probability at most 2 exp(-2 * 300 t^2) =
        3.9e-8 (DKW with Massart's constant), so a correct implementation
        fails with probability below 1.2e-7. Catches draws from [lb, lb], from
        [0, ub - lb] or from a permuted box, a wrong number of draws, and
        calibration draws that repeat the training draws (a reseed before
        the calibration draw)."""
        lb = np.array([-3.0, 0.5, 2.0, -1.5])
        ub = np.array([1.0, 4.0, 2.0, -0.25])
        m, t = 200, 100
        model = _Recorder(lambda x: x[:, [0, 1, 3]] * np.array([1.0, -2.0, 3.0]),
                          t + m)
        conformal_reach(model, Box(lb, ub), m=m, surrogate="naive",
                        training_samples=t, seed=5)
        x, _ = model.rows()
        assert x.shape == (t + m, 4), f"{x.shape} rows drawn, expected {t + m}"
        assert np.all((x >= lb) & (x <= ub)), "a draw lies outside the box"
        assert np.all(x[:, 2] == 2.0)
        bound = 0.172
        for k in (0, 1, 3):
            u = np.sort((x[:, k] - lb[k]) / (ub[k] - lb[k]))
            grid = np.arange(1, len(u) + 1) / len(u)
            distance = max(np.max(grid - u), np.max(u - (grid - 1 / len(u))))
            assert distance <= bound, (
                f"coordinate {k} is not uniform on its interval "
                f"(distance {distance:.3f})")
        train, calib = {tuple(r) for r in x[:t]}, [tuple(r) for r in x[t:]]
        repeated = sum(r in train for r in calib)
        assert repeated == 0, f"{repeated} calibration draws repeat training draws"
