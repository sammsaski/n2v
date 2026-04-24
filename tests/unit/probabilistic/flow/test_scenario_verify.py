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
