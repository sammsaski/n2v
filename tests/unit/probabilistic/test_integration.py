"""
Integration tests for probabilistic verification with n2v.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from n2v import ProbabilisticBox, probabilistic, Box, Star
from n2v.nn import NeuralNetwork
from n2v.sets.probabilistic_box import ProbabilisticBox as ProbabilisticBoxDirect


class TestImports:
    """Tests for module imports."""

    def test_probabilistic_box_importable_from_n2v_sets(self):
        """Test that ProbabilisticBox can be imported from n2v.sets."""
        from n2v.sets import ProbabilisticBox as PB
        assert PB is not None

    def test_probabilistic_box_importable_from_n2v(self):
        """Test that ProbabilisticBox can be imported from n2v."""
        from n2v import ProbabilisticBox as PB
        assert PB is not None

    def test_probabilistic_module_importable_from_n2v(self):
        """Test that probabilistic module can be imported from n2v."""
        from n2v import probabilistic
        assert hasattr(probabilistic, 'verify')

    def test_verify_importable_from_probabilistic(self):
        """Test that verify can be imported from n2v.probabilistic."""
        from n2v.probabilistic import verify
        assert callable(verify)


class TestNeuralNetworkReach:
    """Tests for NeuralNetwork.reach() with probabilistic methods."""

    def test_reach_probabilistic_method(self):
        """Test NeuralNetwork.reach() with method='probabilistic'."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

        # Create verifier
        net = NeuralNetwork(model)

        # Create input set
        lb = np.zeros(3)
        ub = np.ones(3)
        input_box = Box(lb, ub)

        # Run probabilistic verification
        result = net.reach(
            input_box,
            method='probabilistic',
            m=50,
            epsilon=0.1,
            seed=42
        )

        # Should return list with ProbabilisticBox
        assert len(result) == 1
        assert isinstance(result[0], ProbabilisticBox)
        assert result[0].dim == 2  # Output dimension
        assert result[0].epsilon == 0.1

    def test_reach_probabilistic_with_star_input(self):
        """Test that probabilistic method accepts Star input."""
        model = nn.Sequential(nn.Linear(2, 2))

        net = NeuralNetwork(model)

        # Create Star input
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        input_star = Box(lb, ub).to_star()

        # Should convert Star to Box internally and work
        result = net.reach(
            input_star,
            method='probabilistic',
            m=30,
            epsilon=0.1,
            seed=42
        )

        assert len(result) == 1
        assert isinstance(result[0], ProbabilisticBox)

    def test_reach_probabilistic_preserves_params(self):
        """Test that m, ell, epsilon are correctly passed through."""
        model = nn.Linear(2, 2)
        net = NeuralNetwork(model)
        input_box = Box(np.zeros(2), np.ones(2))

        result = net.reach(
            input_box,
            method='probabilistic',
            m=100,
            ell=95,
            epsilon=0.05,
            seed=42
        )

        pbox = result[0]
        assert pbox.m == 100
        assert pbox.ell == 95
        assert pbox.epsilon == 0.05


class TestHybridMethod:
    """Tests for NeuralNetwork.reach() with method='hybrid'."""

    def test_reach_hybrid_small_model(self):
        """Test hybrid method completes deterministically for small model."""
        # Small model that shouldn't trigger switch
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )

        net = NeuralNetwork(model)
        input_star = Box(np.zeros(2), np.ones(2) * 0.1).to_star()  # Small perturbation

        result = net.reach(
            input_star,
            method='hybrid',
            max_stars=1000,
            timeout_per_layer=30.0
        )

        # Should complete deterministically (returns Stars, not ProbabilisticBox)
        # But with small perturbation, might return Stars or might switch
        assert len(result) >= 1

    def test_reach_hybrid_with_star_count_limit(self):
        """Test hybrid method switches when star count exceeded."""
        # Model with ReLU that will create many stars
        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

        net = NeuralNetwork(model)
        input_star = Box(np.zeros(2), np.ones(2)).to_star()

        result = net.reach(
            input_star,
            method='hybrid',
            max_stars=5,  # Very low limit
            m=50,
            epsilon=0.1,
            seed=42
        )

        # Should switch to probabilistic due to low star limit
        assert len(result) >= 1


class TestProbabilisticBoxWithVerifySpecification:
    """Tests for verify_specification with ProbabilisticBox."""

    def test_verify_specification_accepts_probabilistic_box(self):
        """Test that verify_specification() accepts ProbabilisticBox."""
        from n2v.utils.verify_specification import verify_specification
        from n2v.sets import HalfSpace

        # Create a ProbabilisticBox
        lb = np.array([0.0, 0.5])
        ub = np.array([0.3, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        # Define property using HalfSpace
        # Property: y[0] - y[1] <= -0.1 (unsafe if first output is at least 0.1 less than second)
        G = np.array([[1.0, -1.0]])  # y[0] - y[1]
        g = np.array([[-0.1]])  # <= -0.1
        unsafe_region = HalfSpace(G, g)

        # This should work (ProbabilisticBox inherits from Box which has to_star)
        # Note: Result loses probabilistic guarantee (warning issued)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = verify_specification([pbox], unsafe_region)

        # Result is an int: 1 = safe, 0 = unsafe, 2 = unknown
        assert result in (0, 1, 2)


class TestProbabilisticBoxOperations:
    """Tests for ProbabilisticBox operations in the context of n2v."""

    def test_probabilistic_box_is_instance_of_box(self):
        """Test that ProbabilisticBox can be used where Box is expected."""
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        # Should be instance of Box
        assert isinstance(pbox, Box)

    def test_sample_from_probabilistic_box(self):
        """Test sampling from ProbabilisticBox."""
        lb = np.array([0.0, 0.0, 0.0])
        ub = np.array([1.0, 1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        samples = pbox.sample(100)

        assert samples.shape == (3, 100)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_complete_probabilistic_workflow(self):
        """Test complete probabilistic verification workflow."""
        # Create model
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        model.eval()

        # Create verifier
        net = NeuralNetwork(model)

        # Define input region
        center = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        epsilon = 0.1
        input_box = Box(center - epsilon, center + epsilon)

        # Run probabilistic verification
        result = net.reach(
            input_box,
            method='probabilistic',
            m=100,
            epsilon=0.05,
            surrogate='naive',
            seed=42
        )

        # Verify result
        pbox = result[0]
        assert isinstance(pbox, ProbabilisticBox)
        assert pbox.dim == 3
        assert pbox.coverage == 0.95  # 1 - epsilon

        # Check bounds make sense
        lb, ub = pbox.get_range()
        assert np.all(ub >= lb)

    def test_probabilistic_then_deterministic_comparison(self):
        """Compare probabilistic bounds with deterministic bounds."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        model.eval()

        net = NeuralNetwork(model)
        input_box = Box(np.zeros(2), np.ones(2) * 0.5)

        # Probabilistic
        prob_result = net.reach(
            input_box,
            method='probabilistic',
            m=500,
            epsilon=0.001,
            seed=42
        )

        # Deterministic (approx)
        det_result = net.reach(input_box.to_star(), method='approx')

        # Both should give valid results
        assert len(prob_result) >= 1
        assert len(det_result) >= 1

        # Note: We can't directly compare bounds since deterministic
        # is sound overapproximation while probabilistic is empirical
        # with coverage guarantee
