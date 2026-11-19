"""Unit tests for nonconformity score functions."""

import math
from fractions import Fraction

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestHyperrectScore:
    """Tests for HyperrectScore."""

    def test_zero_at_center(self):
        """Score at the center should be 0."""
        from n2v.probabilistic.flow.scores import HyperrectScore

        center = torch.tensor([1.0, 2.0])
        scales = torch.tensor([1.0, 1.0])
        score_fn = HyperrectScore(center, scales)

        y = center.unsqueeze(0)  # (1, 2)
        assert score_fn(y).item() == pytest.approx(0.0)

    def test_max_normalized_deviation(self):
        """Score should be max of |y_k - c_k| / tau_k."""
        from n2v.probabilistic.flow.scores import HyperrectScore

        center = torch.tensor([0.0, 0.0])
        scales = torch.tensor([1.0, 2.0])
        score_fn = HyperrectScore(center, scales)

        y = torch.tensor([[3.0, 2.0]])  # deviations: 3/1=3, 2/2=1
        assert score_fn(y).item() == pytest.approx(3.0)

    def test_batch_computation(self):
        """Should handle batches correctly."""
        from n2v.probabilistic.flow.scores import HyperrectScore

        center = torch.tensor([0.0, 0.0])
        scales = torch.tensor([1.0, 1.0])
        score_fn = HyperrectScore(center, scales)

        y = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
        scores = score_fn(y)
        assert scores.shape == (3,)
        assert scores[0].item() == pytest.approx(1.0)
        assert scores[1].item() == pytest.approx(2.0)
        assert scores[2].item() == pytest.approx(3.0)

    def test_sublevel_set_volume(self):
        """Volume of hyperrect {y : score(y) <= q} = prod(2*q*tau_k)."""
        from n2v.probabilistic.flow.scores import HyperrectScore

        center = torch.tensor([0.0, 0.0])
        scales = torch.tensor([2.0, 3.0])
        score_fn = HyperrectScore(center, scales)

        # Volume = 2*q*2 * 2*q*3 = 4q * 6q = 24q^2
        q = torch.tensor(1.5)
        expected = (2 * 1.5 * 2) * (2 * 1.5 * 3)
        assert score_fn.sublevel_set_volume(q) == pytest.approx(expected)


class TestEllipsoidScore:
    """Tests for EllipsoidScore."""

    def test_zero_at_center(self):
        """Score at the center should be 0."""
        from n2v.probabilistic.flow.scores import EllipsoidScore

        center = torch.tensor([1.0, 2.0])
        cov_inv = torch.eye(2)
        score_fn = EllipsoidScore(center, cov_inv)

        y = center.unsqueeze(0)
        assert score_fn(y).item() == pytest.approx(0.0)

    def test_identity_covariance_equals_l2(self):
        """With identity covariance, Mahalanobis = L2 distance."""
        from n2v.probabilistic.flow.scores import EllipsoidScore

        center = torch.tensor([0.0, 0.0])
        cov_inv = torch.eye(2)
        score_fn = EllipsoidScore(center, cov_inv)

        y = torch.tensor([[3.0, 4.0]])
        assert score_fn(y).item() == pytest.approx(5.0)

    def test_batch_computation(self):
        """Should handle batches correctly."""
        from n2v.probabilistic.flow.scores import EllipsoidScore

        center = torch.tensor([0.0, 0.0])
        cov_inv = torch.eye(2)
        score_fn = EllipsoidScore(center, cov_inv)

        y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        scores = score_fn(y)
        assert scores.shape == (2,)
        assert scores[0].item() == pytest.approx(1.0)
        assert scores[1].item() == pytest.approx(1.0)


class TestBallScore:
    """Tests for BallScore."""

    def test_zero_at_center(self):
        """Score at center should be 0."""
        from n2v.probabilistic.flow.scores import BallScore

        center = torch.tensor([1.0, 2.0])
        score_fn = BallScore(center)

        y = center.unsqueeze(0)
        assert score_fn(y).item() == pytest.approx(0.0)

    def test_l2_distance(self):
        """Score should be L2 norm from center."""
        from n2v.probabilistic.flow.scores import BallScore

        center = torch.tensor([0.0, 0.0])
        score_fn = BallScore(center)

        y = torch.tensor([[3.0, 4.0]])
        assert score_fn(y).item() == pytest.approx(5.0)


class TestFlowScore:
    """Tests for FlowScore."""

    def test_output_shape(self):
        """Score should return (batch,) tensor."""
        from n2v.probabilistic.flow.scores import FlowScore

        # Use a dummy flow model that just returns its input
        class DummyFlow:
            def forward(self, y, t=1.0, **kwargs):
                # kwargs absorbs n_steps / method / atol / rtol that
                # FlowScore._integrate forwards.
                return y

        score_fn = FlowScore(DummyFlow(), t=1.0)
        y = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        scores = score_fn(y)
        assert scores.shape == (2,)

    def test_with_identity_flow(self):
        """With identity flow, FlowScore = L2 norm."""
        from n2v.probabilistic.flow.scores import FlowScore

        class IdentityFlow:
            def forward(self, y, t=1.0, **kwargs):
                # kwargs absorbs n_steps / method / atol / rtol that
                # FlowScore._integrate forwards.
                return y

        score_fn = FlowScore(IdentityFlow(), t=1.0)
        y = torch.tensor([[3.0, 4.0]])
        assert score_fn(y).item() == pytest.approx(5.0)

    def test_set_t(self):
        """set_t should update the flow time parameter."""
        from n2v.probabilistic.flow.scores import FlowScore

        class DummyFlow:
            def forward(self, y, t=1.0, **kwargs):
                # kwargs absorbs n_steps / method / atol / rtol that
                # FlowScore._integrate forwards.
                return y * t  # scale by t

        score_fn = FlowScore(DummyFlow(), t=1.0)
        y = torch.tensor([[3.0, 4.0]])

        score_at_1 = score_fn(y).item()
        score_fn.set_t(0.5)
        score_at_05 = score_fn(y).item()

        assert score_at_1 == pytest.approx(5.0)
        assert score_at_05 == pytest.approx(2.5)


class TestGMMScore:
    """Tests for GMMScore."""

    def test_output_shape(self):
        """Score returns (batch,) tensor matching input dtype."""
        from n2v.probabilistic.flow.scores import GMMScore
        rng = np.random.default_rng(0)
        y_train = torch.as_tensor(
            rng.normal(size=(200, 3)).astype(np.float32)
        )
        score_fn = GMMScore.fit(y_train, n_components=3, random_state=0)
        y_test = torch.as_tensor(rng.normal(size=(10, 3)).astype(np.float32))
        s = score_fn(y_test)
        assert s.shape == (10,)
        assert s.dtype == torch.float32

    def test_high_density_at_training_points(self):
        """Training points should score lower (higher density) than far points."""
        from n2v.probabilistic.flow.scores import GMMScore
        rng = np.random.default_rng(1)
        # Tight cluster around origin
        y_train = torch.as_tensor(
            (0.1 * rng.normal(size=(500, 2))).astype(np.float32)
        )
        score_fn = GMMScore.fit(y_train, n_components=2, random_state=1)
        # Score near origin (in distribution) vs far (out of distribution)
        near = torch.zeros((1, 2), dtype=torch.float32)
        far = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
        assert score_fn(near).item() < score_fn(far).item()

    def test_accepts_numpy_array(self):
        """fit and __call__ accept numpy arrays (not just tensors)."""
        from n2v.probabilistic.flow.scores import GMMScore
        rng = np.random.default_rng(2)
        y_train = rng.normal(size=(150, 2)).astype(np.float32)
        score_fn = GMMScore.fit(y_train, n_components=2, random_state=2)
        y_test = rng.normal(size=(5, 2)).astype(np.float32)
        s = score_fn(y_test)
        assert s.shape == (5,)


# ---------------------------------------------------------------------------
# Closed forms, options and the sameness promise.
#
# The conformal guarantee holds for any score function as long as the same
# function is used for calibration and for membership. The cases below check
# that each score is the function its docstring states (hand values, closed
# forms of a flow known exactly), that the options reach the code that reads
# them, and that the score of a point does not depend on the other points of
# its batch or on the batch size. References never come from n2v: plain Python
# with fractions for the closed-form scores, torch.linalg.matrix_exp for a
# linear flow, sklearn and numpy for the mixture.
# ---------------------------------------------------------------------------


class _LinearField(nn.Module):
    """Velocity v(t, x) = x A^T, independent of t.

    Its flow from time t down to 0 maps y to expm(-A t) y, which is the
    reference; one Euler step of size h maps y to (I - h A) y.
    """

    def __init__(self, A):
        super().__init__()
        self.register_buffer('A', A)

    def forward(self, t, x):
        return x @ self.A.T


# Non-diagonal, entries of both signs above and below one in size.
_A3 = torch.tensor([[0.5, -1.25, 0.25],
                    [0.75, -0.5, 2.0],
                    [-1.5, 0.125, 0.25]], dtype=torch.float64)


def _flow_points(n, seed):
    """n points in R^3 with coordinates of both signs, magnitudes 0 to 4."""
    g = torch.Generator().manual_seed(seed)
    return 4.0 * (2.0 * torch.rand(n, 3, generator=g, dtype=torch.float64) - 1.0)


def _exact_flow_norm(y, t):
    """||expm(-A t) y||, the flow score of the linear field in closed form."""
    E = torch.linalg.matrix_exp(-_A3 * t)
    return (y @ E.T).norm(dim=1)


def _fixed_step_matrix(method, h, k):
    """The matrix of k steps of size h of 'euler' or 'rk4' on y' = -A y.

    One Euler step is I - hA; one step of any four-stage fourth-order
    Runge-Kutta method on a linear system is the Taylor polynomial of
    expm(-hA) to degree 4.
    """
    Z = -h * _A3
    I = torch.eye(3, dtype=torch.float64)
    if method == 'euler':
        S = I + Z
    else:
        S = I + Z + Z @ Z / 2 + Z @ Z @ Z / 6 + Z @ Z @ Z @ Z / 24
    return torch.linalg.matrix_power(S, k)


class TestNonconformityScore:
    """The base class has no score of its own."""

    def test_base_class_refuses_to_score(self):
        """Catches a base class that answers (for example zeros, which make
        the sublevel set the whole space) instead of refusing, so a subclass
        that forgets ``__call__`` goes unnoticed."""
        from n2v.probabilistic.flow.scores import NonconformityScore

        with pytest.raises(NotImplementedError):
            NonconformityScore()(torch.zeros(2, 3))


class TestHyperrectSublevelSetVolume:
    """``sublevel_set_volume`` is the volume of the set the score defines."""

    @pytest.mark.parametrize('q', [0.75, 2.5])
    def test_volume_is_the_volume_of_the_box_the_score_defines(self, q):
        """The set {y : score(y) <= q} is the box c +- q tau: its corners
        score exactly q and a point a little beyond any face scores more.
        Its volume is prod(2 q tau_k), computed here in fractions. Catches a
        volume formula that drops the factor 2 or the power of q, and a score
        whose box is not the one the volume describes (a scale multiplied
        instead of divided, a missing absolute value). Three coordinates with
        scales above and below one, q above and below one; dyadic data, so
        the comparisons are exact."""
        from n2v.probabilistic.flow.scores import HyperrectScore

        c = [1.0, -2.0, 0.5]
        tau = [0.5, 4.0, 2.0]
        score = HyperrectScore(torch.tensor(c, dtype=torch.float64),
                               torch.tensor(tau, dtype=torch.float64))
        expected = math.prod(Fraction(2) * Fraction(q) * Fraction(s)
                             for s in tau)
        assert score.sublevel_set_volume(
            torch.tensor(q, dtype=torch.float64)) == float(expected)

        corners = torch.tensor(
            [[c[k] + sk * q * tau[k] for k, sk in enumerate(signs)]
             for signs in [(1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1)]],
            dtype=torch.float64)
        assert torch.equal(score(corners),
                           torch.full((4,), q, dtype=torch.float64))
        outside = []
        for k in range(3):
            for sign in (1.0, -1.0):
                p = list(c)
                p[k] = c[k] + sign * (q * tau[k] + 0.125)
                outside.append(p)
        assert (score(torch.tensor(outside, dtype=torch.float64)) > q).all()


def _closed_form_case(kind):
    """A closed-form score, its centre, and a hand formula in fractions."""
    from n2v.probabilistic.flow.scores import (
        BallScore, EllipsoidScore, HyperrectScore,
    )
    c = [1.0, -2.0, 0.5]
    if kind == 'hyperrect':
        tau = [0.5, 4.0, 2.0]

        def hand(p):
            return max(abs(Fraction(p[k]) - Fraction(c[k])) / Fraction(tau[k])
                       for k in range(3))
        return HyperrectScore(torch.tensor(c, dtype=torch.float64),
                              torch.tensor(tau, dtype=torch.float64)), hand
    if kind == 'ellipsoid':
        M = [[2.0, 0.5, 0.0], [0.5, 1.0, -0.25], [0.0, -0.25, 3.0]]

        def hand(p):
            d = [Fraction(p[k]) - Fraction(c[k]) for k in range(3)]
            quad = sum(d[i] * Fraction(M[i][j]) * d[j]
                       for i in range(3) for j in range(3))
            return math.sqrt(quad)
        return EllipsoidScore(torch.tensor(c, dtype=torch.float64),
                              torch.tensor(M, dtype=torch.float64)), hand

    def hand(p):
        return math.sqrt(sum((Fraction(p[k]) - Fraction(c[k])) ** 2
                             for k in range(3)))
    return BallScore(torch.tensor(c, dtype=torch.float64)), hand


class TestScoreOfAPointDoesNotDependOnItsBatch:
    """The score of a point alone equals its score inside any batch: exactly
    for the closed-form scores, to the integrator's tolerance for the flow
    score, and every value is the stated function of that point alone."""

    @pytest.mark.parametrize('kind', ['hyperrect', 'ellipsoid', 'ball'])
    def test_closed_form_scores_equal_the_hand_value_alone_and_in_a_batch(
            self, kind):
        """Five points on both sides of the centre in three coordinates,
        dyadic, so every sum and quadratic form is exact in float64 and the
        only rounding is in a final square root (torch's float64 square root
        can differ from the correctly rounded one by an ulp, hence the
        relative 1e-15); rows alone and in a batch must agree exactly.
        Catches a dropped absolute
        value (a point below the centre scores negative), a dropped square
        root, an L1 norm for the L2 norm, an ellipsoid that keeps only the
        diagonal of ``cov_inv`` (off-diagonal entries here), and any score
        that reads other rows of the batch (reduction over the wrong axis,
        batch statistics): each row alone and the batch in reverse order must
        give the same values."""
        score, hand = _closed_form_case(kind)
        pts = [[0.0, -2.0, 0.5], [1.25, -14.0, 1.5], [3.0, 0.0, -3.5],
               [0.75, -1.0, -0.5], [-2.5, 6.0, 0.25]]
        Y = torch.tensor(pts, dtype=torch.float64)
        expected = torch.tensor([float(hand(p)) for p in pts],
                                dtype=torch.float64)
        batch = score(Y)
        assert batch.shape == (5,)
        torch.testing.assert_close(batch, expected, rtol=1e-15, atol=0.0)
        for i in range(5):
            assert torch.equal(score(Y[i:i + 1]), batch[i:i + 1])
        assert torch.equal(score(Y.flip(0)), batch.flip(0))

    def test_flow_score_in_chunks_equals_the_exact_flow_for_every_row(self):
        """``FlowScore`` with ``batch_size=4`` on 11 points (two full chunks
        and a partial one) returns one score per point, each the norm of the
        exact flow expm(-A t) y of a linear field at t = 0.5, and each equal
        to the score of the point alone. dopri5 at atol = rtol = 1e-12 in
        float64 lands within about 1e-11 of the closed form; the bound 1e-9
        leaves a margin. Catches a chunk loop that scores only the first chunk
        or drops rows, and a score that depends on its batch beyond the
        integrator's tolerance."""
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.scores import FlowScore

        score = FlowScore(FlowODE(_LinearField(_A3)), t=0.5, n_steps=5,
                          method='dopri5', atol=1e-12, rtol=1e-12,
                          batch_size=4)
        y = _flow_points(11, seed=3)
        expected = _exact_flow_norm(y, 0.5)
        s = score(y)
        assert s.shape == (11,)
        assert (s - expected).abs().max().item() <= 1e-9
        for i in (0, 4, 10):
            alone = score(y[i:i + 1])
            assert abs(alone.item() - expected[i].item()) <= 1e-9


class TestFlowScoreIsTheNormOfTheFlow:
    """``FlowScore`` is ||phi_t(y)|| with the flow time, solver, step count
    and tolerances it was given."""

    def test_score_matches_the_exact_flow_at_the_constructor_time(self):
        """t = 0.5 given to the constructor, dopri5 with tolerances 1e-12 on
        float64 points: every score is within 1e-9 of ||expm(-A t) y||.
        Catches a score that integrates from t = 1 instead of its own t, and
        tolerances that do not reach the solver (the flow's default 1e-5
        leaves an error of about 1e-5)."""
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.scores import FlowScore

        score = FlowScore(FlowODE(_LinearField(_A3)), t=0.5, n_steps=5,
                          method='dopri5', atol=1e-12, rtol=1e-12)
        y = _flow_points(6, seed=1)
        s = score(y)
        assert s.dtype == torch.float64
        assert (s - _exact_flow_norm(y, 0.5)).abs().max().item() <= 1e-9

    @pytest.mark.parametrize('method', ['euler', 'rk4'])
    def test_fixed_step_method_reaches_the_flow(self, method):
        """``method='euler'`` or ``'rk4'`` with ``n_steps=4`` at t = 0.5 must
        give the norm of that method's polygon on a uniform grid of [0, t].
        The step count is not what this case checks (the docstring says
        n_steps steps, the code takes n_steps - 1; that is the marked case
        ``test_flow.py::TestFlowODEStepCount``), so either count is accepted
        here; the exact flow (what dopri5 returns) is neither. Catches a
        method that does not reach the flow, so 'euler' and 'rk4' silently
        run dopri5."""
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.scores import FlowScore

        t, n_steps = 0.5, 4
        score = FlowScore(FlowODE(_LinearField(_A3)), t=t, n_steps=n_steps,
                          method=method)
        y = _flow_points(5, seed=2)
        s = score(y)
        errors = []
        for k in (n_steps - 1, n_steps):
            ref = (y @ _fixed_step_matrix(method, t / k, k).T).norm(dim=1)
            errors.append((s - ref).abs().max().item())
        assert min(errors) <= 1e-10, errors

    def test_membership_of_a_probabilistic_set_follows_the_exact_flow(self):
        """The public route: ``ProbabilisticSet.contains`` with a
        ``FlowScore`` at t = 0.5. Points y = expm(A t) z with ||z|| = 0.875 q
        are members and those with ||z|| = 1.125 q are not, by the closed
        form. Catches a flow that integrates in the wrong direction, from the
        wrong time or with the wrong time input, seen through the set."""
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.scores import FlowScore
        from n2v.probabilistic.flow.sets import ProbabilisticSet

        q = 2.0
        score = FlowScore(FlowODE(_LinearField(_A3)), t=0.5, n_steps=5,
                          method='dopri5', atol=1e-10, rtol=1e-10)
        pset = ProbabilisticSet(score_fn=score, threshold=q, m=2000,
                                ell=2000, epsilon=0.01, dim=3)
        d = _flow_points(6, seed=5)
        d = d / d.norm(dim=1, keepdim=True)
        radii = torch.tensor([0.875, 1.125] * 3, dtype=torch.float64) * q
        z = d * radii.unsqueeze(1)
        y = z @ torch.linalg.matrix_exp(_A3 * 0.5).T
        inside = pset.contains(y)
        assert inside.tolist() == [True, False] * 3


class TestGMMScoreIsTheNegativeLogDensity:
    """``GMMScore`` is -log p of the fitted mixture and ``fit`` hands every
    option to ``GaussianMixture``."""

    @staticmethod
    def _two_clusters(seed):
        rng = np.random.default_rng(seed)
        a = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], 300)
        b = rng.multivariate_normal([4.0, -2.0], [[0.5, -0.3], [-0.3, 0.6]],
                                    300)
        return np.concatenate([a, b])

    def test_score_is_minus_log_density_of_the_fitted_mixture(self):
        """The reference is -log sum_k w_k N(y | mu_k, Sigma_k), written out
        in numpy from the fitted weights, means and covariances. Points in
        each cluster, between them and far away, in float64: the score keeps
        float64 and matches to a relative 1e-12, alone and in the batch.
        Catches log p returned for -log p (the sublevel set becomes the
        low-density region) and a score cast to float32."""
        from scipy.special import logsumexp
        from n2v.probabilistic.flow.scores import GMMScore

        score = GMMScore.fit(torch.as_tensor(self._two_clusters(0)),
                             n_components=2, random_state=0)
        pts = torch.tensor([[0.0, 0.0], [4.0, -2.0], [2.0, -1.0],
                            [10.0, 10.0], [-1.5, 2.0]], dtype=torch.float64)
        g = score.gmm
        ref = []
        for p in pts.numpy():
            comps = []
            for w, mu, S in zip(g.weights_, g.means_, g.covariances_):
                d = p - mu
                comps.append(math.log(w) - 0.5 * (
                    2 * math.log(2 * math.pi) + np.linalg.slogdet(S)[1]
                    + d @ np.linalg.solve(S, d)))
            ref.append(-logsumexp(comps))
        ref = torch.tensor(ref, dtype=torch.float64)
        s = score(pts)
        assert s.dtype == torch.float64
        torch.testing.assert_close(s, ref, rtol=1e-12, atol=0.0)
        for i in range(len(pts)):
            torch.testing.assert_close(score(pts[i:i + 1]), ref[i:i + 1],
                                       rtol=1e-12, atol=0.0)

    def test_fit_hands_every_option_to_gaussian_mixture(self):
        """``GMMScore.fit`` on a numpy array with ``covariance_type='diag'``,
        ``reg_covar=0.05``, ``max_iter=50``, ``random_state=3`` and two
        components must give the same mixture as ``GaussianMixture`` built
        with those options: the same covariance type and covariances and the
        same scores, exactly. The data are correlated, so a full fit differs.
        Catches an option that does not reach ``GaussianMixture``
        (``covariance_type``, ``reg_covar``, the component count, the seed)."""
        from sklearn.mixture import GaussianMixture
        from n2v.probabilistic.flow.scores import GMMScore

        X = self._two_clusters(1)
        opts = dict(n_components=2, covariance_type='diag', reg_covar=0.05,
                    max_iter=50, random_state=3)
        mine = GMMScore.fit(X, **opts)
        ref = GaussianMixture(**opts).fit(X)
        assert mine.gmm.covariance_type == 'diag'
        np.testing.assert_array_equal(mine.gmm.covariances_, ref.covariances_)
        pts = torch.tensor([[0.0, 0.0], [4.0, -2.0], [1.0, 3.0]],
                           dtype=torch.float64)
        assert torch.equal(
            mine(pts), torch.as_tensor(-ref.score_samples(pts.numpy())))


class TestWhiteningFlowScore:
    """``_WhiteningFlowScore`` whitens its input, then delegates."""

    def test_whitens_then_delegates_and_exposes_the_flow(self):
        """With mean (5, -3, 0.5) and std (4, 0.25, 1.5) around a flow score
        of a linear field at t = 0.5, the score of y is
        ||expm(-A t) ((y - mean) / std)||, and ``flow_model`` is the wrapped
        score's flow (the probabilistic dispatch of ``verify_specification``
        reads it). Catches whitening by multiplication with std, a missing
        shift, and a ``flow_model`` that is not the flow."""
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.scores import FlowScore, _WhiteningFlowScore

        flow = FlowODE(_LinearField(_A3))
        base = FlowScore(flow, t=0.5, n_steps=5, method='dopri5',
                         atol=1e-12, rtol=1e-12)
        mean = torch.tensor([5.0, -3.0, 0.5], dtype=torch.float64)
        std = torch.tensor([4.0, 0.25, 1.5], dtype=torch.float64)
        score = _WhiteningFlowScore(base, mean, std)
        y = _flow_points(5, seed=4) * 3.0 + mean
        expected = _exact_flow_norm((y - mean) / std, 0.5)
        assert (score(y) - expected).abs().max().item() <= 1e-9
        assert score.flow_model is flow
