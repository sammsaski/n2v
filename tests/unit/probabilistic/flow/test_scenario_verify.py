"""Unit tests for scenario-based verification on flow sets."""

import pytest
import numpy as np
import torch


class TestSampleTruncatedGaussianBall:
    """Tests for sample_truncated_gaussian_ball."""

    def test_output_shape(self):
        """Should return (n_samples, dim) array."""
        from n2v.probabilistic.flow.scenario_verify import (
            sample_truncated_gaussian_ball,
        )

        samples = sample_truncated_gaussian_ball(q=1.0, dim=3, n_samples=100)
        assert samples.shape == (100, 3)

    def test_all_samples_inside_ball(self):
        """All samples should satisfy ||z|| <= q."""
        from n2v.probabilistic.flow.scenario_verify import (
            sample_truncated_gaussian_ball,
        )

        q = 2.0
        samples = sample_truncated_gaussian_ball(q=q, dim=4, n_samples=500)
        norms = np.linalg.norm(samples, axis=1)
        assert np.all(norms <= q + 1e-10)

    def test_empirical_mean_near_zero(self):
        """Truncated Gaussian on a ball centered at origin has mean zero."""
        from n2v.probabilistic.flow.scenario_verify import (
            sample_truncated_gaussian_ball,
        )

        np.random.seed(0)
        samples = sample_truncated_gaussian_ball(q=2.0, dim=3, n_samples=10000)
        empirical_mean = samples.mean(axis=0)
        assert np.allclose(empirical_mean, 0.0, atol=0.1)

    def test_low_dim_uses_rejection(self):
        """For low dim, samples should come from rejection sampling and
        match a direct rejection-sample empirical distribution."""
        from n2v.probabilistic.flow.scenario_verify import (
            sample_truncated_gaussian_ball,
        )

        np.random.seed(42)
        samples = sample_truncated_gaussian_ball(q=1.5, dim=5, n_samples=200)
        assert samples.shape == (200, 5)
        assert np.all(np.linalg.norm(samples, axis=1) <= 1.5 + 1e-10)

    def test_high_dim_uses_chi_method(self):
        """For high dim, sphere + chi method should still produce valid samples."""
        from n2v.probabilistic.flow.scenario_verify import (
            sample_truncated_gaussian_ball,
        )

        np.random.seed(42)
        samples = sample_truncated_gaussian_ball(q=8.0, dim=50, n_samples=200)
        assert samples.shape == (200, 50)
        assert np.all(np.linalg.norm(samples, axis=1) <= 8.0 + 1e-10)

    def test_chi_method_explicit(self):
        """Explicitly request chi method even at low dim."""
        from n2v.probabilistic.flow.scenario_verify import (
            sample_truncated_gaussian_ball,
        )

        np.random.seed(42)
        samples = sample_truncated_gaussian_ball(
            q=1.0, dim=2, n_samples=100, method='chi'
        )
        assert samples.shape == (100, 2)
        assert np.all(np.linalg.norm(samples, axis=1) <= 1.0 + 1e-10)

    def test_invalid_q_raises(self):
        """q <= 0 should raise ValueError."""
        from n2v.probabilistic.flow.scenario_verify import (
            sample_truncated_gaussian_ball,
        )

        with pytest.raises(ValueError, match="q"):
            sample_truncated_gaussian_ball(q=0.0, dim=2, n_samples=10)

        with pytest.raises(ValueError, match="q"):
            sample_truncated_gaussian_ball(q=-1.0, dim=2, n_samples=10)


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    def test_verified_result(self):
        """Construct a verified result."""
        from n2v.probabilistic.flow.scenario_verify import ScenarioResult

        result = ScenarioResult(
            verified=True,
            outcome='verified',
            counterexample=None,
            genuine_input=None,
            epsilon_2=0.001,
            delta_2=0.999,
            n_samples_used=10000,
        )
        assert result.verified is True
        assert result.counterexample is None
        assert result.epsilon_2 == 0.001

    def test_falsified_result(self):
        """Construct a falsified result with counterexample."""
        from n2v.probabilistic.flow.scenario_verify import ScenarioResult

        z = np.array([1.0, 0.0])
        y = np.array([2.5, -1.0])
        result = ScenarioResult(
            verified=False,
            outcome='falsified',
            counterexample=(z, y, 0.5),
            genuine_input=np.array([0.3, 0.7]),
            epsilon_2=0.001,
            delta_2=0.999,
            n_samples_used=10000,
        )
        assert result.verified is False
        assert result.counterexample is not None
        assert result.counterexample[2] == 0.5


import math


class TestScenarioVerifyHalfspace:
    """Tests for single-halfspace scenario verification."""

    def _make_trained_flow(self, dim=2):
        """Helper: train a tiny flow on simple data."""
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.train import train_flow

        torch.manual_seed(42)
        vf = VelocityField(dim=dim, hidden=32, n_layers=3)
        data = torch.randn(200, dim) * 0.3
        train_flow(vf, data, n_epochs=20, batch_size=64, lr=1e-3,
                   coupling='none')
        return FlowODE(vf)

    def test_loose_spec_verifies(self):
        """A spec with huge slack should verify (no counterexample)."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )

        flow = self._make_trained_flow(dim=2)
        # Spec: y[0] <= 100 (always true)
        w = np.array([1.0, 0.0])
        b = 100.0

        result = scenario_verify_halfspace(
            flow_ode=flow, threshold_q=2.0,
            w=w, b=b,
            n_samples=200, beta_2=0.001, t=1.0,
        )
        assert result.verified is True
        assert result.counterexample is None
        assert result.epsilon_2 > 0
        assert result.epsilon_2 < 1
        assert result.delta_2 == 0.999

    def test_tight_spec_falsifies(self):
        """A spec that excludes most of the flow set should falsify."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )

        flow = self._make_trained_flow(dim=2)
        # Spec: y[0] <= -10 (always false)
        w = np.array([1.0, 0.0])
        b = -10.0

        result = scenario_verify_halfspace(
            flow_ode=flow, threshold_q=2.0,
            w=w, b=b,
            n_samples=200, beta_2=0.001, t=1.0,
        )
        assert result.verified is False
        assert result.counterexample is not None
        z, y, margin = result.counterexample
        assert z.shape == (2,)
        assert y.shape == (2,)
        assert margin > 0

    def test_epsilon_2_formula(self):
        """epsilon_2 should equal -log(beta_2) / n_samples."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )

        flow = self._make_trained_flow(dim=2)
        N = 1000
        beta_2 = 0.01

        result = scenario_verify_halfspace(
            flow_ode=flow, threshold_q=1.0,
            w=np.array([1.0, 0.0]), b=100.0,
            n_samples=N, beta_2=beta_2, t=1.0,
        )
        expected_eps = -math.log(beta_2) / N
        assert abs(result.epsilon_2 - expected_eps) < 1e-10

    def test_invalid_n_samples(self):
        """n_samples <= 0 should raise."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )

        flow = self._make_trained_flow(dim=2)
        with pytest.raises(ValueError, match="n_samples"):
            scenario_verify_halfspace(
                flow_ode=flow, threshold_q=1.0,
                w=np.array([1.0, 0.0]), b=100.0,
                n_samples=0, beta_2=0.001, t=1.0,
            )

    def test_invalid_beta_2(self):
        """beta_2 not in (0, 1) should raise."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )

        flow = self._make_trained_flow(dim=2)
        with pytest.raises(ValueError, match="beta_2"):
            scenario_verify_halfspace(
                flow_ode=flow, threshold_q=1.0,
                w=np.array([1.0, 0.0]), b=100.0,
                n_samples=100, beta_2=1.5, t=1.0,
            )


class TestRobustnessResult:
    """Tests for the joint robustness result dataclass."""

    def test_construct(self):
        """Construct a verified robustness result."""
        from n2v.probabilistic.flow.scenario_verify import RobustnessResult

        result = RobustnessResult(
            verified=True,
            outcome='verified',
            counterexample=None,
            genuine_input=None,
            epsilon_total=0.0017,
            delta_total=0.996,
            epsilon_1=0.001, delta_1=0.997,
            epsilon_2=0.0007, delta_2=0.999,
            n_classes=10,
            n_samples_used=10000,
        )
        assert result.verified is True
        assert result.epsilon_total == pytest.approx(0.0017)


class TestVerifyRobustness:
    """Tests for the verify_robustness wrapper."""

    def _make_trained_flow(self, dim=4):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.train import train_flow

        torch.manual_seed(42)
        vf = VelocityField(dim=dim, hidden=32, n_layers=3)
        # Concentrate data so that one class dominates by construction
        data = torch.randn(200, dim) * 0.1
        data[:, 2] += 5.0  # class 2 logit is dominant
        # 100 epochs are needed for the flow to actually recover the
        # dominant-class structure strongly enough for verification
        # (20 epochs leaves class-2 winning only ~50% of latent samples).
        train_flow(vf, data, n_epochs=100, batch_size=64, lr=1e-3,
                   coupling='none')
        return FlowODE(vf)

    def test_robust_classification(self):
        """A network with dominant class should verify as robust."""
        from n2v.probabilistic.flow.scenario_verify import verify_robustness

        flow = self._make_trained_flow(dim=4)
        result = verify_robustness(
            flow_ode=flow,
            threshold_q=1.5,
            true_class=2,
            n_classes=4,
            epsilon_1=0.001,
            delta_1=0.997,
            n_samples=200,
            beta_2=0.001,
            t=1.0,
        )
        assert result.verified is True
        assert result.counterexample is None
        # Joint coverage = (1 - eps1)(1 - eps2)
        expected_coverage = (1 - result.epsilon_1) * (1 - result.epsilon_2)
        expected_eps_total = 1.0 - expected_coverage
        assert abs(result.epsilon_total - expected_eps_total) < 1e-10
        # Joint confidence = delta_1 * delta_2
        expected_delta_total = result.delta_1 * result.delta_2
        assert abs(result.delta_total - expected_delta_total) < 1e-10

    def test_non_robust_classification(self):
        """A spec where another class can win should falsify."""
        from n2v.probabilistic.flow.scenario_verify import verify_robustness

        flow = self._make_trained_flow(dim=4)
        # Claim true_class is 0 even though class 2 dominates → falsify
        result = verify_robustness(
            flow_ode=flow,
            threshold_q=1.5,
            true_class=0,
            n_classes=4,
            epsilon_1=0.001,
            delta_1=0.997,
            n_samples=200,
            beta_2=0.001,
            t=1.0,
        )
        assert result.verified is False
        assert result.counterexample is not None

    def test_joint_certificate_composition(self):
        """Verify that epsilon_total and delta_total are computed correctly."""
        from n2v.probabilistic.flow.scenario_verify import verify_robustness

        flow = self._make_trained_flow(dim=4)
        result = verify_robustness(
            flow_ode=flow, threshold_q=1.5,
            true_class=2, n_classes=4,
            epsilon_1=0.005, delta_1=0.95,
            n_samples=500, beta_2=0.01,
            t=1.0,
        )
        # Per-spec eps_2 = -log(beta_2)/N (shared samples, not Bonferroni)
        expected_eps_2 = -math.log(0.01) / 500
        assert abs(result.epsilon_2 - expected_eps_2) < 1e-10
        assert result.delta_2 == 1.0 - 0.01


class TestPreimageResult:
    """Tests for PreimageResult dataclass."""

    def test_found_result(self):
        """Construct a result where preimage was found."""
        from n2v.probabilistic.flow.scenario_verify import PreimageResult

        result = PreimageResult(
            found=True,
            x=np.array([0.5, 0.5]),
            y_achieved=np.array([1.0, 0.25]),
            distance=0.0001,
        )
        assert result.found is True
        assert result.x is not None
        assert result.distance == 0.0001

    def test_not_found_result(self):
        """Construct a result where preimage was not found."""
        from n2v.probabilistic.flow.scenario_verify import PreimageResult

        result = PreimageResult(
            found=False,
            x=None,
            y_achieved=None,
            distance=5.0,
        )
        assert result.found is False
        assert result.x is None


class TestPreimageSearch:
    """Tests for preimage_search function."""

    def test_identity_function(self):
        """For f(x) = x, preimage of y should be y itself."""
        from n2v.probabilistic.flow.scenario_verify import preimage_search

        def identity(x):
            return x

        y_target = np.array([0.3, 0.7])
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        result = preimage_search(
            target_fn=identity,
            y_target=y_target,
            input_set_bounds=bounds,
            n_restarts=3,
            n_steps=200,
            lr=0.05,
        )
        assert result.found is True
        assert result.distance < 0.01
        np.testing.assert_allclose(result.x, y_target, atol=0.01)

    def test_linear_function(self):
        """For f(x) = 2x + 1, preimage of y should be (y-1)/2."""
        from n2v.probabilistic.flow.scenario_verify import preimage_search

        def linear(x):
            return 2.0 * x + 1.0

        y_target = np.array([1.5, 2.0])
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        result = preimage_search(
            target_fn=linear,
            y_target=y_target,
            input_set_bounds=bounds,
            n_restarts=3,
            n_steps=200,
            lr=0.05,
        )
        assert result.found is True
        np.testing.assert_allclose(result.x, np.array([0.25, 0.5]), atol=0.01)

    def test_preimage_not_exists(self):
        """If no x in the input set maps to y_target, should return found=False."""
        from n2v.probabilistic.flow.scenario_verify import preimage_search

        def identity(x):
            return x

        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        result = preimage_search(
            target_fn=identity,
            y_target=np.array([-5.0, -5.0]),
            input_set_bounds=bounds,
            n_restarts=5,
            n_steps=200,
            lr=0.05,
            tolerance=0.1,
        )
        assert result.found is False
        assert result.distance > 0.1

    def test_torch_model_input(self):
        """Should accept a PyTorch nn.Module as target_fn."""
        from n2v.probabilistic.flow.scenario_verify import preimage_search

        torch.manual_seed(0)
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
        )
        net.eval()

        x0 = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
        with torch.no_grad():
            y_target = net(x0).numpy().squeeze()

        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        result = preimage_search(
            target_fn=net,
            y_target=y_target,
            input_set_bounds=bounds,
            n_restarts=5,
            n_steps=300,
            lr=0.05,
        )
        assert result.found is True
        assert result.distance < 0.01


class TestScenarioVerifyHalfspaceWithPreimage:
    """Tests for scenario verification with preimage search integration."""

    def _make_trained_flow(self, dim=2):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.train import train_flow

        torch.manual_seed(42)
        vf = VelocityField(dim=dim, hidden=32, n_layers=3)
        data = torch.randn(200, dim) * 0.3
        train_flow(vf, data, n_epochs=20, batch_size=64, lr=1e-3,
                   coupling='none')
        return FlowODE(vf)

    def test_verified_outcome(self):
        """Loose spec: outcome should be 'verified'."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )

        flow = self._make_trained_flow(dim=2)
        result = scenario_verify_halfspace(
            flow_ode=flow, threshold_q=2.0,
            w=np.array([1.0, 0.0]), b=100.0,
            n_samples=200, beta_2=0.001, t=1.0,
        )
        assert result.verified is True
        assert result.outcome == 'verified'
        assert result.genuine_input is None

    def test_unknown_outcome_without_target_fn(self):
        """Violation found but no target_fn provided: outcome = 'unknown'."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )

        flow = self._make_trained_flow(dim=2)
        result = scenario_verify_halfspace(
            flow_ode=flow, threshold_q=2.0,
            w=np.array([1.0, 0.0]), b=-10.0,
            n_samples=200, beta_2=0.001, t=1.0,
        )
        assert result.verified is False
        assert result.outcome == 'unknown'
        assert result.genuine_input is None
        assert result.counterexample is not None

    @pytest.mark.slow
    def test_falsified_outcome_with_matching_target(self):
        """Target network IS the flow inverse: preimage search succeeds."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )

        flow = self._make_trained_flow(dim=2)
        # Use the flow itself as the target: psi_t(z) = target_fn(z)
        def target_fn(x):
            return flow.inverse(x, t=1.0, n_steps=100)

        result = scenario_verify_halfspace(
            flow_ode=flow, threshold_q=2.0,
            w=np.array([1.0, 0.0]), b=-10.0,
            n_samples=200, beta_2=0.001, t=1.0,
            target_fn=target_fn,
            input_set_bounds=(np.array([-3.0, -3.0]), np.array([3.0, 3.0])),
            preimage_n_restarts=3,
            preimage_n_steps=100,
            preimage_tolerance=0.1,
        )
        assert result.verified is False
        assert result.outcome == 'falsified'
        assert result.genuine_input is not None


class TestVerifyRobustnessWithPreimage:
    """Tests for verify_robustness with preimage search."""

    def _make_trained_flow(self, dim=4):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.train import train_flow

        torch.manual_seed(42)
        vf = VelocityField(dim=dim, hidden=32, n_layers=3)
        data = torch.randn(200, dim) * 0.1
        data[:, 2] += 5.0
        train_flow(vf, data, n_epochs=100, batch_size=64, lr=1e-3,
                   coupling='none')
        return FlowODE(vf)

    def test_robust_has_outcome_verified(self):
        """Robust classification gets outcome = 'verified'."""
        from n2v.probabilistic.flow.scenario_verify import verify_robustness

        flow = self._make_trained_flow(dim=4)
        result = verify_robustness(
            flow_ode=flow, threshold_q=1.5,
            true_class=2, n_classes=4,
            epsilon_1=0.001, delta_1=0.997,
            n_samples=200, beta_2=0.001, t=1.0,
        )
        assert result.outcome == 'verified'
        assert result.verified is True

    def test_non_robust_without_target_fn_is_unknown(self):
        """Flow violation found but no target_fn: outcome = 'unknown'."""
        from n2v.probabilistic.flow.scenario_verify import verify_robustness

        flow = self._make_trained_flow(dim=4)
        result = verify_robustness(
            flow_ode=flow, threshold_q=1.5,
            true_class=0, n_classes=4,
            epsilon_1=0.001, delta_1=0.997,
            n_samples=200, beta_2=0.001, t=1.0,
        )
        assert result.outcome == 'unknown'
        assert result.verified is False

    @pytest.mark.slow
    def test_non_robust_with_target_fn_is_falsified(self):
        """Flow violation + target_fn = flow itself: outcome = 'falsified'."""
        from n2v.probabilistic.flow.scenario_verify import verify_robustness

        flow = self._make_trained_flow(dim=4)

        def target_fn(x):
            return flow.inverse(x, t=1.0, n_steps=100)

        result = verify_robustness(
            flow_ode=flow, threshold_q=1.5,
            true_class=0, n_classes=4,
            epsilon_1=0.001, delta_1=0.997,
            n_samples=200, beta_2=0.001, t=1.0,
            target_fn=target_fn,
            input_set_bounds=(np.array([-3.0]*4), np.array([3.0]*4)),
            preimage_n_restarts=3,
            preimage_n_steps=100,
            preimage_tolerance=0.1,
        )
        assert result.outcome == 'falsified'
        assert result.genuine_input is not None


class TestSpecCheckBeforeFalsified:
    """Tests for the spec-violation check before marking as 'falsified'."""

    def test_halfspace_unknown_when_preimage_found_but_spec_satisfied(self):
        """If preimage search finds x in bounds but f(x) satisfies the spec,
        the outcome should be 'unknown', not 'falsified'.

        Constructed using an identity flow and a tight input bound: the flow
        reach set contains points violating the spec (ball of radius 2), but
        real inputs are restricted to [-0.3, 0.3]^2 under identity f, so no
        real output can have y[0] > 0.5. With a very loose tolerance,
        preimage search declares the candidate 'found', but the spec check
        should catch that f(x) doesn't actually violate w^T y <= 0.5.
        """
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )

        class IdentityFlowODE:
            def inverse(self, z, t=1.0, n_steps=100, **kwargs):
                return z

        flow = IdentityFlowODE()

        def target_fn(x):
            return x

        result = scenario_verify_halfspace(
            flow_ode=flow,
            threshold_q=2.0,
            w=np.array([1.0, 0.0]),
            b=0.5,
            n_samples=500,
            beta_2=0.001,
            t=1.0,
            target_fn=target_fn,
            input_set_bounds=(np.array([-0.3, -0.3]), np.array([0.3, 0.3])),
            preimage_n_restarts=3,
            preimage_n_steps=50,
            preimage_tolerance=100.0,  # very loose: any x is "found"
        )
        assert result.outcome == 'unknown'
        assert result.genuine_input is None

    def test_halfspace_still_falsifies_when_spec_actually_violated(self):
        """If preimage search finds x and f(x) truly violates the spec,
        outcome should still be 'falsified'."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )

        class IdentityFlowODE:
            def inverse(self, z, t=1.0, n_steps=100, **kwargs):
                return z

        flow = IdentityFlowODE()

        def target_fn(x):
            return x

        # Spec: y[0] <= 0. Flow set contains y[0] up to 2, bounds allow
        # real inputs with x[0] up to 3.0, so the whole ball is reachable
        # under identity f and a real spec violation exists and is
        # achievable within tolerance.
        result = scenario_verify_halfspace(
            flow_ode=flow,
            threshold_q=2.0,
            w=np.array([1.0, 0.0]),
            b=0.0,
            n_samples=500,
            beta_2=0.001,
            t=1.0,
            target_fn=target_fn,
            input_set_bounds=(np.array([-3.0, -3.0]), np.array([3.0, 3.0])),
            preimage_n_restarts=3,
            preimage_n_steps=200,
            preimage_tolerance=0.1,
        )
        assert result.outcome == 'falsified'
        assert result.genuine_input is not None
        # Verify the returned input actually satisfies target_fn(x)[0] > 0
        x_ce = result.genuine_input
        y_real = target_fn(torch.tensor(x_ce, dtype=torch.float32)).numpy()
        assert y_real[0] > 0

    def test_robustness_unknown_when_preimage_found_but_spec_satisfied(self):
        """verify_robustness: if preimage search finds x but no wrong class
        actually beats the true class at f(x), outcome is 'unknown'.

        Constructed so the flow set (ball) contains points where class 1
        beats class 0, but the tight input bounds prevent the target_fn
        (which boosts class 0) from actually producing a spec violation."""
        from n2v.probabilistic.flow.scenario_verify import verify_robustness

        class IdentityFlowODE:
            def inverse(self, z, t=1.0, n_steps=100, **kwargs):
                return z

        flow = IdentityFlowODE()

        # target_fn boosts class 0 by a large constant, so f(x)[0] is
        # always at least 10. No wrong class can ever beat class 0 at f(x).
        def target_fn(x):
            boost = torch.zeros_like(x)
            boost[..., 0] = 10.0
            return x + boost

        result = verify_robustness(
            flow_ode=flow,
            threshold_q=2.0,
            true_class=0,
            n_classes=4,
            epsilon_1=0.001,
            delta_1=0.997,
            n_samples=500,
            beta_2=0.001,
            t=1.0,
            target_fn=target_fn,
            input_set_bounds=(np.array([-1.0]*4), np.array([1.0]*4)),
            preimage_n_restarts=3,
            preimage_n_steps=50,
            preimage_tolerance=100.0,  # loose: any x is "found"
        )
        assert result.outcome == 'unknown'
        assert result.genuine_input is None


class TestSampleEmpiricalLatentBall:
    """Tests for sample_empirical_latent_ball."""

    def _make_flow_and_data(self, dim=3, n_train=200, seed=42):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.train import train_flow

        torch.manual_seed(seed)
        vf = VelocityField(dim=dim, hidden=32, n_layers=3)
        y_train = torch.randn(n_train, dim) * 0.3
        train_flow(vf, y_train, n_epochs=20, batch_size=64, lr=1e-3,
                   coupling='none')
        return FlowODE(vf), y_train

    def test_output_shape(self):
        from n2v.probabilistic.flow.scenario_verify import (
            sample_empirical_latent_ball,
        )
        flow, y_train = self._make_flow_and_data(dim=3)
        samples = sample_empirical_latent_ball(
            flow, y_train, q=2.0, n_samples=100, seed=0,
        )
        assert samples.shape == (100, 3)

    def test_all_samples_inside_ball(self):
        from n2v.probabilistic.flow.scenario_verify import (
            sample_empirical_latent_ball,
        )
        flow, y_train = self._make_flow_and_data(dim=3)
        q = 1.5
        samples = sample_empirical_latent_ball(
            flow, y_train, q=q, n_samples=500, seed=0,
        )
        norms = np.linalg.norm(samples, axis=1)
        assert np.all(norms <= q + 1e-10)

    def test_invalid_q_raises(self):
        from n2v.probabilistic.flow.scenario_verify import (
            sample_empirical_latent_ball,
        )
        flow, y_train = self._make_flow_and_data(dim=3)
        with pytest.raises(ValueError, match="q"):
            sample_empirical_latent_ball(
                flow, y_train, q=0.0, n_samples=10, seed=0,
            )

    def test_invalid_n_samples_raises(self):
        from n2v.probabilistic.flow.scenario_verify import (
            sample_empirical_latent_ball,
        )
        flow, y_train = self._make_flow_and_data(dim=3)
        with pytest.raises(ValueError, match="n_samples"):
            sample_empirical_latent_ball(
                flow, y_train, q=1.0, n_samples=0, seed=0,
            )

    def test_reproducible_with_seed(self):
        from n2v.probabilistic.flow.scenario_verify import (
            sample_empirical_latent_ball,
        )
        flow, y_train = self._make_flow_and_data(dim=3)
        s1 = sample_empirical_latent_ball(
            flow, y_train, q=2.0, n_samples=50, seed=123,
        )
        s2 = sample_empirical_latent_ball(
            flow, y_train, q=2.0, n_samples=50, seed=123,
        )
        np.testing.assert_array_equal(s1, s2)


class TestLatentSamplesParameter:
    """Tests for the `latent_samples` parameter on scenario functions."""

    def _make_trained_flow(self, dim=2):
        from n2v.probabilistic.flow.model import VelocityField
        from n2v.probabilistic.flow.ode import FlowODE
        from n2v.probabilistic.flow.train import train_flow

        torch.manual_seed(42)
        vf = VelocityField(dim=dim, hidden=32, n_layers=3)
        data = torch.randn(200, dim) * 0.3
        train_flow(vf, data, n_epochs=20, batch_size=64, lr=1e-3,
                   coupling='none')
        return FlowODE(vf)

    def test_halfspace_accepts_custom_latent_samples(self):
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )
        flow = self._make_trained_flow(dim=2)
        custom_z = np.random.default_rng(0).normal(0, 0.1, size=(200, 2))
        result = scenario_verify_halfspace(
            flow_ode=flow, threshold_q=2.0,
            w=np.array([1.0, 0.0]), b=100.0,
            n_samples=200, beta_2=0.001,
            latent_samples=custom_z,
        )
        assert result.outcome == 'verified'

    def test_halfspace_rejects_wrong_shape(self):
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )
        flow = self._make_trained_flow(dim=2)
        bad_z = np.random.default_rng(0).normal(0, 0.1, size=(100, 2))
        with pytest.raises(ValueError, match="latent_samples"):
            scenario_verify_halfspace(
                flow_ode=flow, threshold_q=2.0,
                w=np.array([1.0, 0.0]), b=100.0,
                n_samples=200, beta_2=0.001,
                latent_samples=bad_z,
            )

    def test_robustness_accepts_custom_latent_samples(self):
        from n2v.probabilistic.flow.scenario_verify import verify_robustness
        flow = self._make_trained_flow(dim=3)
        custom_z = np.random.default_rng(0).normal(0, 0.1, size=(200, 3))
        result = verify_robustness(
            flow_ode=flow, threshold_q=2.0,
            true_class=0, n_classes=3,
            epsilon_1=0.001, delta_1=0.997,
            n_samples=200, beta_2=0.001,
            latent_samples=custom_z,
        )
        # Result can be verified/falsified/unknown; we just check it ran
        assert result.outcome in ('verified', 'falsified', 'unknown')


# --------------------------------------------------------------------------- #
# Refusals, options and plumbing                                               #
# with stand-in flows whose maps are known exactly. The soundness cells of     #
# these functions are in tests/soundness/test_soundness_scenario_verify.py.    #
# --------------------------------------------------------------------------- #
class _RecordingIdentityFlow:
    """A stand-in flow: the inverse map is the identity (float64) and every
    call records its keywords and its batch size."""

    def __init__(self):
        self.calls = []

    def inverse(self, z, **kwargs):
        self.calls.append(dict(kwargs, batch=int(z.shape[0])))
        return z.detach().to(torch.float64).clone()

    def forward(self, y, **kwargs):
        return y.detach().to(torch.float64).clone()


class TestSamplerRefusals:
    """Refusals of the samplers that no other case asserts."""

    def test_unknown_sampling_method_raises(self):
        """An unknown method name is an input that cannot be read: ValueError,
        never a silent fallback to one of the two samplers."""
        from n2v.probabilistic.flow.scenario_verify import (
            sample_truncated_gaussian_ball,
        )
        for method in ('rejecton', 'CHI', ''):
            with pytest.raises(ValueError, match="method"):
                sample_truncated_gaussian_ball(q=1.0, dim=2, n_samples=10, method=method)

    def test_negative_noise_sigma_raises(self):
        """A negative noise scale is refused before the flow is called; zero
        is legal (the soundness file samples with noise 0)."""
        from n2v.probabilistic.flow.scenario_verify import (
            sample_empirical_latent_ball,
        )
        flow = _RecordingIdentityFlow()
        with pytest.raises(ValueError, match="noise_sigma"):
            sample_empirical_latent_ball(flow, torch.zeros(4, 2), q=1.0,
                                         n_samples=5, noise_sigma=-0.1, seed=0)


class TestPreimageStepHandling:
    """The optional preimage step of scenario_verify_halfspace and
    verify_robustness: when it runs, and what a failure inside it does."""

    @staticmethod
    def _halfspace(target_fn, bounds):
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )
        return scenario_verify_halfspace(
            _RecordingIdentityFlow(), threshold_q=2.0, w=np.array([1.0, 0.0]), b=0.0,
            n_samples=200, beta_2=0.01, target_fn=target_fn, input_set_bounds=bounds,
            preimage_n_restarts=1, preimage_n_steps=5)

    @staticmethod
    def _robustness(target_fn, bounds):
        from n2v.probabilistic.flow.scenario_verify import verify_robustness
        return verify_robustness(
            _RecordingIdentityFlow(), threshold_q=2.0, true_class=0, n_classes=2,
            epsilon_1=0.01, delta_1=0.99, n_samples=200, beta_2=0.01,
            target_fn=target_fn, input_set_bounds=bounds,
            preimage_n_restarts=1, preimage_n_steps=5)

    @pytest.mark.parametrize("entry", ["_halfspace", "_robustness"])
    def test_a_target_without_gradient_gives_unknown(self, entry):
        """A target network that does not propagate gradients makes the
        preimage step raise 'does not require grad'; the documented answer is
        'unknown' with the flow counterexample kept, not a crash and not
        'falsified'."""
        def no_grad_target(x):
            return x.detach() * 2.0

        bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        result = getattr(self, entry)(no_grad_target, bounds)
        assert result.outcome == 'unknown'
        assert result.genuine_input is None
        assert result.counterexample is not None

    @pytest.mark.parametrize("entry", ["_halfspace", "_robustness"])
    def test_another_runtime_error_in_the_target_is_raised(self, entry):
        """Only the gradient-flow failure is swallowed: any other RuntimeError
        of the target network reaches the caller."""
        def broken_target(x):
            raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")

        bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
            getattr(self, entry)(broken_target, bounds)

    @pytest.mark.parametrize("entry", ["_halfspace", "_robustness"])
    @pytest.mark.parametrize("given", ["target-only", "bounds-only"])
    def test_no_preimage_search_without_both_target_and_bounds(self, entry, given):
        """The search runs only when both the target network and the input box
        are given; with one of them the answer is 'unknown' and the target is
        never called."""
        calls = []

        def target(x):
            calls.append(1)
            return x

        bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        result = getattr(self, entry)(target if given == "target-only" else None,
                                      bounds if given == "bounds-only" else None)
        assert result.outcome == 'unknown'
        assert result.genuine_input is None
        assert calls == []

    @pytest.mark.parametrize("entry", ["halfspace", "robustness"])
    def test_the_preimage_keywords_reach_the_search(self, entry):
        """The target's output is a constant c that violates the property (y_0 = 5 above
        b = 0; class 1 = 5 above class 0), about 4 away from every flow sample of the unit
        disc. With preimage_tolerance 10 that constant counts as found and the answer is
        'falsified'; the default tolerance 1e-3 would leave it 'unknown'. The search makes
        n_restarts (n_steps + 1) target calls: 2 (3 + 1) = 8. Catches the tolerance, the
        restart count or the step count not handed to preimage_search."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace, verify_robustness,
        )
        calls = []
        c = torch.tensor([5.0, 0.0] if entry == "halfspace" else [0.0, 5.0])

        def constant_target(x):
            calls.append(1)
            return x * 0.0 + c

        keywords = dict(target_fn=constant_target,
                        input_set_bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
                        preimage_n_restarts=2, preimage_n_steps=3, preimage_tolerance=10.0)
        if entry == "halfspace":
            result = scenario_verify_halfspace(_RecordingIdentityFlow(), 1.0, np.array([1.0, 0.0]),
                                               0.0, n_samples=100, beta_2=0.01, **keywords)
        else:
            result = verify_robustness(_RecordingIdentityFlow(), 1.0, 0, 2, 0.01, 0.99, 100, 0.01,
                                       **keywords)
        assert result.outcome == 'falsified'
        assert len(calls) == 8


class TestOdeKeywordsReachTheFlow:
    """The flow keywords of the scenario functions reach flow_ode.inverse."""

    def test_scenario_halfspace_hands_its_ode_keywords_to_the_flow(self):
        """Explicit values arrive unchanged; the documented defaults are
        t = 1.0, 100 steps, dopri5 and tolerances 1e-5. Catches a keyword
        that is dropped or swapped (a wrong flow time integrates a different
        map)."""
        from n2v.probabilistic.flow.scenario_verify import (
            scenario_verify_halfspace,
        )
        flow = _RecordingIdentityFlow()
        scenario_verify_halfspace(flow, 1.0, np.array([1.0, 0.0]), 5.0, n_samples=17,
                                  beta_2=0.01, t=0.25, n_ode_steps=7, ode_method='euler',
                                  ode_atol=1e-3, ode_rtol=1e-4)
        scenario_verify_halfspace(flow, 1.0, np.array([1.0, 0.0]), 5.0, n_samples=17,
                                  beta_2=0.01)
        assert flow.calls == [
            dict(t=0.25, n_steps=7, method='euler', atol=1e-3, rtol=1e-4, batch=17),
            dict(t=1.0, n_steps=100, method='dopri5', atol=1e-5, rtol=1e-5, batch=17)]

    def test_robustness_hands_its_ode_keywords_to_the_flow(self):
        """As above, for verify_robustness."""
        from n2v.probabilistic.flow.scenario_verify import verify_robustness
        flow = _RecordingIdentityFlow()
        verify_robustness(flow, 1.0, 0, 2, 0.01, 0.99, 13, 0.01, t=0.5, n_ode_steps=9,
                          ode_method='rk4', ode_atol=1e-2, ode_rtol=1e-3)
        verify_robustness(flow, 1.0, 0, 2, 0.01, 0.99, 13, 0.01)
        assert flow.calls == [
            dict(t=0.5, n_steps=9, method='rk4', atol=1e-2, rtol=1e-3, batch=13),
            dict(t=1.0, n_steps=100, method='dopri5', atol=1e-5, rtol=1e-5, batch=13)]

    def test_certify_halfspace_hands_its_ode_keywords_to_the_flow(self):
        """As above, for certify_halfspace_disjoint, whose defaults are 30
        steps of rk4."""
        from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
        from n2v.sets.halfspace import HalfSpace
        flow = _RecordingIdentityFlow()
        hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-5.0]]))
        certify_halfspace_disjoint(flow, 1.0, hs, n_samples=11, beta_2=0.01, t=0.75,
                                   n_ode_steps=3, ode_method='euler', ode_atol=1e-2,
                                   ode_rtol=1e-3, seed=0)
        certify_halfspace_disjoint(flow, 1.0, hs, n_samples=11, beta_2=0.01, seed=0)
        assert flow.calls == [
            dict(t=0.75, n_steps=3, method='euler', atol=1e-2, rtol=1e-3, batch=11),
            dict(t=1.0, n_steps=30, method='rk4', atol=1e-5, rtol=1e-5, batch=11)]


class TestCertifyHalfspaceDisjointOptions:
    """Seed, adaptive escalation and sampling strategy of
    certify_halfspace_disjoint, on the identity stand-in flow with q = 1."""

    @staticmethod
    def _hs(bound):
        """The unsafe half-space {y_0 >= bound} as G y <= g."""
        from n2v.sets.halfspace import HalfSpace
        return HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-bound]]))

    @pytest.mark.parametrize("strategy", ["uniform", "qmc", "qmc+antithetic"])
    def test_the_seed_fixes_the_sample_whatever_the_global_state(self, strategy):
        """Two calls with the same seed, with the global numpy state changed in
        between, return the same worst sample and margin bit for bit; another
        seed returns another sample. Catches a seed that is not handed to the
        sampler (the sample then depends on the global state)."""
        from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
        flow = _RecordingIdentityFlow()
        np.random.seed(123)
        a = certify_halfspace_disjoint(flow, 1.0, self._hs(1.25), n_samples=64,
                                       beta_2=0.01, seed=5, sampling_strategy=strategy)
        np.random.seed(456)
        b = certify_halfspace_disjoint(flow, 1.0, self._hs(1.25), n_samples=64,
                                       beta_2=0.01, seed=5, sampling_strategy=strategy)
        c = certify_halfspace_disjoint(flow, 1.0, self._hs(1.25), n_samples=64,
                                       beta_2=0.01, seed=6, sampling_strategy=strategy)
        np.testing.assert_array_equal(a.worst_sample, b.worst_sample)
        assert a.worst_max_margin == b.worst_max_margin
        assert not np.array_equal(a.worst_sample, c.worst_sample)

    def test_a_marginal_certificate_is_escalated_to_the_larger_sample(self):
        """Every sample has margin at least 0.25 (the ball of radius 1 against
        y_0 >= 1.25), below the threshold 0.5: the result is the re-run with
        adaptive_n_samples, so it states N = 900 and ln(1/beta)/900, and the
        flow saw a first batch of 100 and a second of 900, both integrated
        with the caller's flow keywords. Catches the re-run made with the
        first-stage N and a flow keyword that the re-run drops."""
        from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
        flow = _RecordingIdentityFlow()
        ode = dict(t=0.5, n_ode_steps=7, ode_method='euler', ode_atol=1e-3, ode_rtol=1e-4)
        r = certify_halfspace_disjoint(flow, 1.0, self._hs(1.25), n_samples=100,
                                       beta_2=0.001, seed=0, adaptive_threshold=0.5,
                                       adaptive_n_samples=900, **ode)
        assert r.disjoint is True
        assert r.n_samples_used == 900
        assert abs(r.epsilon_2 - math.log(1000.0) / 900) <= 1e-15
        assert [c['batch'] for c in flow.calls] == [100, 900]
        seen = dict(t=0.5, n_steps=7, method='euler', atol=1e-3, rtol=1e-4)
        assert [{k: c[k] for k in seen} for c in flow.calls] == [seen, seen]

    @pytest.mark.parametrize("case", ["margin-above-threshold", "not-disjoint",
                                      "threshold-only", "count-only"])
    def test_no_escalation_outside_the_marginal_band(self, case):
        """No re-run when the worst margin exceeds the threshold, when the
        polyhedron is hit, or when only one of the two adaptive keywords is
        given: one flow call of N samples and ln(1/beta)/N."""
        from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
        kwargs = {"margin-above-threshold": dict(adaptive_threshold=0.1, adaptive_n_samples=900),
                  "not-disjoint": dict(adaptive_threshold=0.5, adaptive_n_samples=900),
                  "threshold-only": dict(adaptive_threshold=0.5),
                  "count-only": dict(adaptive_n_samples=900)}[case]
        bound = 0.5 if case == "not-disjoint" else 1.25
        flow = _RecordingIdentityFlow()
        r = certify_halfspace_disjoint(flow, 1.0, self._hs(bound), n_samples=100,
                                       beta_2=0.001, seed=0, **kwargs)
        assert r.disjoint is (case != "not-disjoint")
        assert r.n_samples_used == 100
        assert abs(r.epsilon_2 - math.log(1000.0) / 100) <= 1e-15
        assert [c['batch'] for c in flow.calls] == [100]

    def test_unknown_sampling_strategy_raises(self):
        """An unknown strategy is refused with ValueError (the slow-tier case
        in test_scenario_sampling_qmc.py trains a flow; this one does not)."""
        from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
        with pytest.raises(ValueError, match="sampling_strategy"):
            certify_halfspace_disjoint(_RecordingIdentityFlow(), 1.0, self._hs(1.25),
                                       n_samples=10, beta_2=0.01, seed=0,
                                       sampling_strategy='sobol')

    @pytest.mark.parametrize("beta_2", [0.0, 1.0, -0.5])
    def test_beta_outside_the_open_unit_interval_raises(self, beta_2):
        """beta_2 = 0 would state no confidence bound at all (log of
        infinity) and beta_2 = 1 a miscoverage of 0 at confidence 0: both are
        refused, as is a non-positive sample count."""
        from n2v.probabilistic.flow.scenario_verify import certify_halfspace_disjoint
        with pytest.raises(ValueError, match="beta_2"):
            certify_halfspace_disjoint(_RecordingIdentityFlow(), 1.0, self._hs(1.25),
                                       n_samples=10, beta_2=beta_2, seed=0)
        with pytest.raises(ValueError, match="n_samples"):
            certify_halfspace_disjoint(_RecordingIdentityFlow(), 1.0, self._hs(1.25),
                                       n_samples=0, beta_2=0.01, seed=0)


class TestExtractMinWorstMaxMargin:
    """_extract_min_worst_max_margin (no caller in the library today)."""

    def test_least_margin_over_every_group_and_member(self):
        """The least worst_max_margin over every member of every group,
        skipping entries without members or without a margin; None when
        nothing is left. Margins of both signs, the least in the second
        group."""
        from types import SimpleNamespace
        from n2v.probabilistic.flow.scenario_verify import _extract_min_worst_max_margin

        def hs(m):
            return SimpleNamespace(worst_max_margin=m)

        groups = [SimpleNamespace(per_hs_results=[hs(0.5), hs(2.0)]),
                  SimpleNamespace(per_hs_results=[hs(None), hs(-0.25), hs(1.5)]),
                  SimpleNamespace(per_hs_results=None),
                  SimpleNamespace()]
        assert _extract_min_worst_max_margin({'per_group_results': groups}) == -0.25
        assert _extract_min_worst_max_margin({'per_group_results': groups[2:]}) is None
        assert _extract_min_worst_max_margin({'per_group_results': []}) is None
        assert _extract_min_worst_max_margin({}) is None


class TestScenarioRefusals:
    """Refusals of scenario_verify_halfspace and verify_robustness at the edges
    of their parameter ranges."""

    @pytest.mark.parametrize("beta_2", [0.0, 1.0])
    def test_scenario_halfspace_refuses_beta_at_the_ends(self, beta_2):
        """beta_2 = 0 gives no finite epsilon and beta_2 = 1 states a
        miscoverage of 0 at confidence 0: both are refused (the existing case
        tries 1.5 only)."""
        from n2v.probabilistic.flow.scenario_verify import scenario_verify_halfspace
        with pytest.raises(ValueError, match="beta_2"):
            scenario_verify_halfspace(_RecordingIdentityFlow(), 1.0, np.array([1.0, 0.0]),
                                      5.0, n_samples=10, beta_2=beta_2)

    @pytest.mark.parametrize("bad, match", [
        (dict(n_samples=0), "n_samples"),
        (dict(beta_2=0.0), "beta_2"),
        (dict(beta_2=1.0), "beta_2"),
        (dict(epsilon_1=0.0), "epsilon_1"),
        (dict(epsilon_1=1.0), "epsilon_1"),
        (dict(delta_1=0.0), "delta_1"),
        (dict(delta_1=1.5), "delta_1"),
        (dict(true_class=3), "true_class"),
        (dict(true_class=-1), "true_class"),
        (dict(n_classes=1, true_class=0), "n_classes"),
    ])
    def test_robustness_refuses_parameters_out_of_range(self, bad, match):
        """Every documented range of verify_robustness is enforced with
        ValueError before the flow is called; delta_1 = 1 is legal."""
        from n2v.probabilistic.flow.scenario_verify import verify_robustness
        kwargs = dict(threshold_q=1.0, true_class=0, n_classes=3, epsilon_1=0.01,
                      delta_1=0.99, n_samples=10, beta_2=0.01)
        kwargs.update(bad)
        flow = _RecordingIdentityFlow()
        with pytest.raises(ValueError, match=match):
            verify_robustness(flow, **kwargs)
        assert flow.calls == []
        ok = dict(kwargs, **{k: v for k, v in dict(
            n_samples=10, beta_2=0.01, epsilon_1=0.01, delta_1=1.0, true_class=0,
            n_classes=3).items()})
        assert verify_robustness(flow, **ok).delta_1 == 1.0

    @pytest.mark.parametrize("shape", [(9, 3), (10, 2)], ids=["fewer-samples", "fewer-classes"])
    def test_robustness_rejects_latent_samples_of_the_wrong_shape(self, shape):
        """Given latent samples whose count is not n_samples (the scenario bound would be
        stated for a sample that was not drawn: ln(1/beta)/10 for 9 points) or whose width
        is not n_classes are refused with ValueError before the flow is called; the
        halfspace twin is TestLatentSamplesParameter::test_halfspace_rejects_wrong_shape."""
        from n2v.probabilistic.flow.scenario_verify import verify_robustness
        flow = _RecordingIdentityFlow()
        with pytest.raises(ValueError, match="latent_samples"):
            verify_robustness(flow, 1.0, 0, 3, 0.01, 0.99, 10, 0.01,
                              latent_samples=np.zeros(shape))
        assert flow.calls == []
